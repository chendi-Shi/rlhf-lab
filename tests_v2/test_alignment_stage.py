"""Tiny random CPU fixtures only; no production model, GPU, or real-data scoring."""
import contextlib
import copy
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from alignment_lab import alignment_stage as stage, gpu_reference as gpu, study_stage as parents
from alignment_lab import alignment_process_guard as process_guard
from alignment_lab.checkpointing import build_identity, load_checkpoint, save_checkpoint
from alignment_lab.capacity_stage import compare_state
from alignment_lab.stage_artifacts import export_stage, import_stage
from alignment_lab.stage_engine import parameter_mapping
from alignment_lab.stage_engine_observed import run_ppo, run_supervised
from alignment_lab.training_telemetry import verify_checkpoint_telemetry, verify_training_telemetry, read as telemetry_read
from alignment_lab.training_data import TrainingPartition, TrainingRow
import test_study_stage as parent_fixtures
from test_stage_artifacts import backend_for
from test_reference_bridge import identity_inputs, paired_batches
from test_shared_qwen import fixture


class AlignmentStageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        assert not torch.cuda.is_initialized()

    def setUp(self):
        self.fixtures = parent_fixtures.StudyStageTests()
        self.fixtures.setUp()
        self.addCleanup(self.fixtures.tearDown)
        self.root = self.fixtures.root
        self.addCleanup(patch.stopall)
        patch.object(stage, "ROOT", self.root).start()
        for name in stage.REQUIRED_SOURCES:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_text("isolated test fixture", encoding="utf-8")
        self.fixtures.json("audit/manifest.json", {"status": "audit_complete", "data_ready": True,
            "readiness_blockers": [], "training_started": False})
        self.request = {k: v for k, v in self.fixtures.request.items()
                        if k not in {"expected_rows", "expected_updates"}}
        self.request.update(kind=stage.KIND, stage="dpo", output=str(self.root / "runs/alignment/dpo42"),
            audit_manifest_sha256=stage.sha(self.root / "audit/manifest.json"),
            source_sha256={n: stage.sha(self.root / n) for n in stage.REQUIRED_SOURCES})
        capacity = stage.read_json(self.request["capacity_acceptance_path"])
        capacity["data"]["manifest"]["sha256"] = self.request["audit_manifest_sha256"]
        self.fixtures.json("reports/capacity-acceptance.json", capacity)
        self.request["capacity_acceptance_sha256"] = stage.sha(self.request["capacity_acceptance_path"])
        self.supervisor = {"pid": 900001, "parent_pid": 0,
            "created_at_utc": "2026-10-21T00:00:00.0000000Z",
            "command_line": "python scripts/run_alignlab_alignment.py --plan tiny-fixture --execute"}
        self.fixtures.json("alignment-plan.json", {"fixture": "CPU only",
            "process_wait_policy": copy.deepcopy(process_guard.POLICY), "allowed_background_processes": []})
        self.request.update(alignment_plan_path=str(self.root / "alignment-plan.json"),
            alignment_plan_sha256=stage.sha(self.root / "alignment-plan.json"),
            workload={"scope": stage.FULL_SCOPE, "rows": 13139, "optimizer_updates": 822, "beta": 0.1},
            reference_cache={"mode": "create", "path": str(Path(self.request["output"]) / "reference-cache")},
            process_wait_policy=copy.deepcopy(process_guard.POLICY), allowed_background_processes=[],
            supervisor_processes=[self.supervisor], process_wait_spent_seconds=0.0)
        (self.root / "audit/ppo_prompts.jsonl").write_text("{}\n", encoding="utf-8")
        self.parent_receipts = self.make_parents()

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def make_parents(self):
        """Serializer fixtures labelled full-work metadata, never real acceptance."""
        files = {"base/weights.bin": self.root / "models/base/weights.bin",
            "tokenizer/tokenizer.json": self.root / "models/tokenizer/tokenizer.json",
            "tokenizer/source-manifest.json": self.root / "models/tokenizer/source-manifest.json",
            "data/manifest": self.root / "audit/manifest.json", "parent_plan": self.root / "parent-plan.json"}
        receipts, accepted_stages, campaign_stages = {}, {}, []
        for name in ("sft", "reward"):
            backend, _ = backend_for(name)
            rows, updates = parents.EXPECTED[name][:2]
            config = {"stage": name, "purpose": parents.KIND, "scope": stage.FULL_SCOPE,
                "synthetic_rows": 0, "formal_training": True, "expected_rows": rows, "expected_updates": updates}
            identity = build_identity(files, config, {} if name == "sft" else
                {"sft": Path(receipts["sft"]["artifact_dir"]) / "manifest.json"})
            if name == "reward":
                import_stage(receipts["sft"]["artifact_dir"], backend, "reward",
                    expected_manifest_sha256=receipts["sft"]["manifest_sha256"], expected_identity=identity)
            completion = {"stop_reason": "data_exhausted", "data_pass_complete": True,
                "protocol_complete": True, "full_accepted_dataset_pass": True, "scope": stage.FULL_SCOPE,
                "synthetic_rows": 0, "rows_processed": rows, "optimizer_updates": updates}
            output = self.root / "runs/parent-fixture" / name
            output.mkdir(parents=True)
            receipt = export_stage(output / "final-artifact", backend, "reward" if name == "reward" else "policy",
                stage=name, identity=identity, heads=("reward",) if name == "reward" else (), completion=completion)
            receipts[name] = {k: receipt[k] for k in ("artifact_dir", "manifest_sha256")}
            progress = {"status": "formal_parent_stage_passed", "stage": name, "formal_training": True,
                "research_complete": False, "actual_optimizer_updates": updates, "artifact": receipts[name]}
            (output / "progress.json").write_text(json.dumps(progress), encoding="utf-8")
            accepted_stages[name] = {**receipts[name], "progress_path": str(output / "progress.json"),
                "progress_sha256": stage.sha(output / "progress.json"), "rows": rows, "optimizer_updates": updates}
            campaign_stages.append({"stage": name, "status": "accepted", "artifact": receipts[name],
                "progress_sha256": stage.sha(output / "progress.json")})
        self.fixtures.json("reports/parent-campaign.json", {"status": "parents_passed",
            "parents_verified": True, "formal_training": True, "research_complete": False,
            "plan_sha256": self.request["parent_plan_sha256"], "stages": campaign_stages})
        self.fixtures.json("reports/parent-acceptance.json", {
            "schema_version": 1, "kind": "alignlab-independent-formal-parents-acceptance-v1",
            "status": "accepted_full_real_formal_parents", "all_checks_passed": True,
            "parents_verified": True, "formal_training": True, "research_complete": False,
            "processes_exited": True,
            "parent_plan": {"path": self.request["parent_plan_path"], "sha256": self.request["parent_plan_sha256"]},
            "campaign": {"path": str(self.root / "reports/parent-campaign.json"),
                "sha256": stage.sha(self.root / "reports/parent-campaign.json"), "status": "parents_passed"},
            "stages": accepted_stages, "fixture_only": True})
        self.request.update(parent_acceptance_path=str(self.root / "reports/parent-acceptance.json"),
            parent_acceptance_sha256=stage.sha(self.root / "reports/parent-acceptance.json"),
            parents={"sft": receipts["sft"]})
        return receipts

    def ppo_request(self):
        return dict(self.request, stage="ppo", output=str(self.root / "runs/alignment/ppo42"),
            parents=self.parent_receipts, reference_cache=None, workload={"scope": stage.SUBSET_SCOPE,
                "rows": 1024, "optimizer_updates": 2048, "selection_seed": 42,
                "selection_algorithm": stage.SELECTION_RULE, "kl_coef": 0.05})

    def partition(self, request):
        count = stage.DPO_ROWS if request["stage"] == "dpo" else stage.PPO_POOL_ROWS
        rows = tuple(TrainingRow(f"r{i}", f"c{i}", f"p{i}", (4, 7 + i % 5),
            (20 + i % 7, stage.EOS) if request["stage"] == "dpo" else None,
            (40 + i % 7, stage.EOS) if request["stage"] == "dpo" else None) for i in range(count))
        return TrainingPartition(rows, "preference_train" if request["stage"] == "dpo" else "ppo_prompts",
            request["audit_manifest_sha256"], "a" * 64, "isolated-tiny-CPU", b"{}")

    @contextlib.contextmanager
    def runtime(self, request, *, engine=None, forbid_read=False, process_reader=None):
        is_ppo = request["stage"] == "ppo"
        request["workload"] = dict(request["workload"], rows=2 if is_ppo else 3,
                                  optimizer_updates=4 if is_ppo else 1)
        backend, _ = backend_for(request["stage"])
        loaded = SimpleNamespace(backend=backend, role="policy", heads=("value",) if is_ppo else (),
            metadata={"fixture": "tiny random CPU; no production capacity claim"},
            device=torch.device("cpu"), autocast=contextlib.nullcontext)
        with contextlib.ExitStack() as stack:
            request_path = Path(request["output"]).parent / (Path(request["output"]).name + ".request.json")
            own = {"pid": os.getpid(), "parent_pid": self.supervisor["pid"],
                "created_at_utc": "2026-10-21T00:00:01.0000000Z",
                "command_line": f'python scripts/run_alignlab_alignment_stage.py --request "{request_path}" --execute'}
            def processes():
                snapshot = [copy.deepcopy(self.supervisor), own]
                return process_reader(snapshot, loaded) if process_reader else snapshot
            stack.enter_context(patch("alignment_lab.probe_dispatch._System",
                return_value=SimpleNamespace(processes=processes)))
            for key, value in {"DPO_ROWS": 3, "PPO_POOL_ROWS": 4, "PPO_COUNTS": (2,),
                "PAD": 0, "EOS": 2, "MAX_NEW_TOKENS": 3, "ROLLOUT_BATCH": 2, "PRECISION": "float32"}.items():
                stack.enter_context(patch.object(stage, key, value))
            partition = self.partition(request)
            stack.enter_context(patch("alignment_lab.training_data.load_training_partition", return_value=partition))
            stack.enter_context(patch("alignment_lab.gpu_probe.resources", return_value={"blockers": []}))
            stack.enter_context(patch("alignment_lab.model_loader.QUANTIZATION", "none"))
            stack.enter_context(patch("alignment_lab.model_loader.load_quantized_stage", return_value=loaded))
            stack.enter_context(patch("rlhf_lab.runtime.gpu_inventory", return_value={"fixture_only": True}))
            for name in ("set_device", "reset_peak_memory_stats", "synchronize"):
                stack.enter_context(patch.object(torch.cuda, name, return_value=None))
            for name in ("max_memory_allocated", "max_memory_reserved"):
                stack.enter_context(patch.object(torch.cuda, name, return_value=0))
            if engine:
                stack.enter_context(patch("alignment_lab.stage_engine_observed." + ("run_ppo" if is_ppo else "run_supervised"),
                                          side_effect=engine))
            if forbid_read:
                original_read = backend.read
                def guarded_read(role, callback):
                    if role == "reference":
                        raise AssertionError("Cache reuse must not forward reference")
                    return original_read(role, callback)
                stack.enter_context(patch.object(backend, "read", side_effect=guarded_read))
            yield loaded, partition

    def reference_fixture(self):
        parts = fixture()
        chosen, rejected = paired_batches(parts)
        artifacts = identity_inputs(self.root / "reference-inputs", parts[0])
        accepted = artifacts.pop("accepted_dataset_manifest_sha256")
        config = artifacts.pop("scoring_config")
        receipt = gpu.capture_frozen_reference(parts[0], "reference", artifact_files=artifacts,
            accepted_data_manifest_sha256=accepted, precision="float32", quantization="none", hash_chunk_bytes=128)
        return receipt, config, [(chosen, rejected)]

    def test_explicit_workload_and_original_budget(self):
        stage.validate_request(self.request)
        dpo = stage.stage_config(self.request)
        self.assertEqual((dpo.max_updates, dpo.effective_batch, dpo.microbatch, dpo.beta), (822, 16, 1, 0.1))
        ppo_request = self.ppo_request()
        ppo = stage.stage_config(ppo_request)
        self.assertEqual((ppo.max_responses, ppo.max_updates, ppo.rollout_batch, ppo.epochs,
            ppo.optimizer_batch, ppo.max_new_tokens, ppo.kl_coef), (1024, 2048, 16, 2, 1, 256, 0.05))
        self.assertEqual(ppo.deadline_epoch, self.request["deadline_epoch"])
        for rows in (None, True, 12748, 0):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                stage.validate_workload("ppo", dict(ppo_request["workload"], rows=rows))
        with self.assertRaises(ValueError):
            stage.validate_request(dict(self.request, seed=45))
        with self.assertRaises(ValueError):
            stage.validate_request(dict(ppo_request, reference_cache={"mode": "create"}))

    def test_dry_run_and_independent_both_parent_acceptance_required(self):
        with patch("alignment_lab.gpu_probe.resources", side_effect=AssertionError("No CUDA admission")):
            self.assertEqual(stage.run_alignment_stage(self.request)["status"], "plan_only")
        self.assertFalse(Path(self.request["output"]).exists())
        accepted = stage.read_json(self.request["parent_acceptance_path"])
        accepted["stages"].pop("reward")
        Path(self.request["parent_acceptance_path"]).write_text(json.dumps(accepted))
        self.request["parent_acceptance_sha256"] = stage.sha(self.request["parent_acceptance_path"])
        with self.assertRaisesRegex(ValueError, "both full-real"):
            stage.validate_request(self.request)  # DPO still waits for the common RM acceptance.

    def test_parent_campaign_progress_and_actual_weights_are_pinned(self):
        acceptance = stage.read_json(self.request["parent_acceptance_path"])
        progress = Path(acceptance["stages"]["sft"]["progress_path"])
        progress.write_text(progress.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            stage.validate_request(self.request)

    def test_capacity_parent_cannot_be_laundered_as_formal_parent(self):
        record = self.parent_receipts["sft"]
        path = Path(record["artifact_dir"]) / "manifest.json"
        manifest = stage.read_json(path)
        manifest["identity"]["config"]["purpose"] = "alignlab-capacity-stage-v1"
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "Capacity/partial"):
            stage._formal_parent("sft", {**record, "manifest_sha256": stage.sha(path)}, self.request)

    def test_selected_prompts_are_common_hash_sample_not_branch_seed_or_prefix(self):
        request = self.ppo_request()
        with self.runtime(request):
            pool = self.partition(request)
            selected, summary, records = stage.select_real_rows(pool, request)
            expected = sorted(pool.rows, key=lambda r: (hashlib.sha256(f"42:{r.prompt_id}".encode()).hexdigest(), r.prompt_id))[:2]
            self.assertEqual([r.record_id for r in selected], [r.record_id for r in expected])
            other, _, _ = stage.select_real_rows(pool, dict(request, seed=44))
            self.assertEqual(selected, other)
            self.assertEqual((summary["accepted_partition_rows"], summary["selected_rows"],
                summary["unique_prompt_ids"], summary["synthetic_rows"]), (4, 2, 2, 0))
            self.assertEqual([r["record_id"] for r in records], [r.record_id for r in selected])
            bad = dataclasses.replace(pool, rows=(dataclasses.replace(pool.rows[0], prompt_id=pool.rows[1].prompt_id),) + pool.rows[1:])
            with self.assertRaisesRegex(ValueError, "unique prompt"):
                stage.select_real_rows(bad, request)

    def test_bounded_cache_matches_original_real_tiny_scores_identity_and_binding(self):
        receipt, config, pairs = self.reference_fixture()
        original = gpu.create_reference_cache(receipt, self.root / "original-cache", pairs,
            scoring_config=config, chunk_tokens=stage.CHUNK_TOKENS)
        progress = []
        bounded = stage.create_bounded_cache(receipt, self.root / "bounded-cache", pairs, config=config,
            deadline_epoch=time.time() + 60, progress=lambda **state: progress.append(state))
        self.assertEqual(bounded.identity, original.identity)
        self.assertEqual(dict(bounded._cache.entries), dict(original._cache.entries))
        self.assertTrue(progress[-1]["complete"])
        self.assertEqual(progress[-1]["scored_sequences"], len(original._cache.entries))
        chosen, rejected = pairs[0]
        actual, expected = bounded.pair_scores(chosen, rejected), original.pair_scores(chosen, rejected)
        torch.testing.assert_close(actual.chosen, expected.chosen, rtol=0, atol=0)
        torch.testing.assert_close(actual.rejected, expected.rejected, rtol=0, atol=0)

    def test_real_cache_record_wait_preserves_complete_record_and_tiny_model(self):
        output = Path(self.request["output"])
        journal = output / "reference-cache-records.jsonl"
        observed = {"unknown_reads": 0, "waits": 0, "cleared": False}
        original_session = process_guard.GuardSession

        def unchanged():
            self.assertEqual(journal.read_bytes(), observed["journal"])
            self.assertTrue(all(torch.equal(p.detach(), before)
                for p, before in zip(observed["parameters"], observed["weights"])))

        def processes(snapshot, loaded):
            if observed["cleared"] or not journal.exists():
                return snapshot
            records = [json.loads(line) for line in journal.read_text().splitlines()]
            if (records[-1]["scored_sequences"] == 0
                    or records[-1]["scored_sequences"] != records[-1]["expected_unique_sequences"]):
                return snapshot
            if not observed["unknown_reads"]:
                observed["journal"] = journal.read_bytes()
                observed["scored"] = records[-1]["scored_sequences"]
                observed["parameters"] = tuple(loaded.backend._all_parameters())
                observed["weights"] = tuple(p.detach().clone() for p in observed["parameters"])
            unchanged()
            if observed["unknown_reads"] < 2:
                observed["unknown_reads"] += 1
                return snapshot + [{"pid": 900002, "parent_pid": 0,
                    "created_at_utc": "2026-10-21T00:00:02.0000000Z",
                    "command_line": "python -m pip install --target .runtime Pillow"}]
            observed["cleared"] = True
            return snapshot

        def checked_sleep(seconds):
            unchanged()
            candidates = list(output.glob(f"cache-safe-boundary-{observed['scored']:06d}-*.json"))
            self.assertEqual(len(candidates), 1)
            saved = stage.read_json(candidates[0])
            self.assertTrue(saved["complete"])
            self.assertEqual(saved["optimizer_updates"], 0)
            self.assertEqual(saved["committed_record"], json.loads(journal.read_text().splitlines()[-1]))
            self.assertEqual(saved["cache_journal"], str(journal))
            observed["waits"] += 1
            time.sleep(seconds)
            unchanged()

        def session(*args, **kwargs):
            return original_session(*args, **kwargs, sleep=checked_sleep)

        with self.runtime(self.request, process_reader=processes):
            with patch.object(process_guard, "GuardSession", side_effect=session):
                result = stage.run_alignment_stage(self.request, execute=True)
        self.assertEqual((observed["unknown_reads"], observed["waits"], observed["cleared"]), (2, 1, True))
        self.assertEqual(result["actual_optimizer_updates"], 1)
        self.assertGreaterEqual(result["process_guard"]["total_waited_seconds"], 2)
        self.assertTrue(result["process_guard"]["last"]["admission_allowed"])
        verified = process_guard.verify_guard_log(output / "process-guard.jsonl", request=self.request, progress=result)
        self.assertTrue(verified["verified"])
        # A valid old scoring record cannot replace the exact record at which
        # this particular wait began, even if all replacement hashes agree.
        log = output / "process-guard.jsonl"
        events = [json.loads(line) for line in log.read_text().splitlines()]
        wait = next(row for row in events if row["event"] == "wait_start")
        receipt_path = Path(wait["saved_boundary"]["path"])
        saved = stage.read_json(receipt_path)
        saved["committed_record"] = json.loads(journal.read_text().splitlines()[0])
        receipt_path.write_text(json.dumps(saved), encoding="utf8")
        changed_hash = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
        for row in events:
            if row.get("saved_boundary") and row["saved_boundary"]["path"] == str(receipt_path):
                row["saved_boundary"]["sha256"] = changed_hash
        log.write_text("".join(json.dumps(row) + "\n" for row in events), encoding="utf8")
        with self.assertRaisesRegex(ValueError, "committed complete scoring record"):
            process_guard.verify_guard_log(log, request=self.request, progress=result)
        self.assertFalse(torch.cuda.is_initialized())

    def test_unknown_after_initial_admission_clears_before_model_loader(self):
        calls, cleared, loads, sleeps = [0], [False], [], []
        original_session = process_guard.GuardSession
        def inventory(snapshot, loaded):
            calls[0] += 1
            if calls[0] in (3, 4):
                return snapshot + [{"pid": 900002, "parent_pid": 0,
                    "created_at_utc": "2026-10-21T00:00:02.0000000Z",
                    "command_line": "python -m pip install --target .runtime Pillow"}]
            if calls[0] > 4:
                cleared[0] = True
            return snapshot
        def waiting(seconds):
            self.assertEqual(loads, [])
            self.assertFalse(cleared[0])
            self.assertTrue((Path(self.request["output"]) / "request.json").is_file())
            sleeps.append(seconds)
            time.sleep(seconds)
        def session(*args, **kwargs):
            return original_session(*args, **kwargs, sleep=waiting)
        with self.runtime(self.request, process_reader=inventory) as (loaded, _):
            def load(*args, **kwargs):
                self.assertTrue(cleared[0])
                loads.append(True)
                return loaded
            with patch.object(process_guard, "GuardSession", side_effect=session), patch(
                    "alignment_lab.model_loader.load_quantized_stage", side_effect=load):
                result = stage.run_alignment_stage(self.request, execute=True)
        self.assertEqual((loads, sleeps), ([True], [2.0]))
        self.assertTrue(process_guard.verify_guard_log(Path(self.request["output"]) / "process-guard.jsonl",
            request=self.request, progress=result)["verified"])
        journal = Path(self.request["output"]) / "reference-cache-records.jsonl"
        records = [json.loads(line) for line in journal.read_text().splitlines()]
        records[1]["record"]["sum_logprob"] -= 0.25
        chain = "0" * 64
        for record in records:
            record["previous_sha256"] = chain
            payload = {k: v for k, v in record.items() if k != "record_sha256"}
            chain = stage.digest(payload)
            record["record_sha256"] = chain
        journal.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf8")
        with self.assertRaisesRegex(ValueError, "actually consumed reference scores"):
            process_guard.verify_guard_log(Path(self.request["output"]) / "process-guard.jsonl",
                request=self.request, progress=result)

    def test_stop_and_deadline_between_each_scored_sequence_publish_no_partial_cache(self):
        receipt, config, pairs = self.reference_fixture()
        for mode in ("stop", "deadline"):
            completed = []
            output = self.root / f"interrupted-{mode}"
            deadline = time.time() + 60
            with self.subTest(mode=mode), patch.object(receipt._owner, "read", wraps=receipt._owner.read) as read:
                with self.assertRaises(InterruptedError if mode == "stop" else TimeoutError):
                    stage.create_bounded_cache(receipt, output, pairs, config=config, deadline_epoch=deadline,
                        stop_requested=lambda: mode == "stop" and bool(completed),
                        clock=lambda: deadline if mode == "deadline" and completed else time.time(),
                        progress=lambda **state: completed.append(state) if state["scored_sequences"] else None)
                self.assertEqual(read.call_count, 1)
                self.assertFalse(output.exists())

    def test_partial_existing_cache_is_never_repaired_or_overwritten(self):
        receipt, config, pairs = self.reference_fixture()
        output = self.root / "partial"
        output.mkdir()
        (output / "identity.json").write_text("{}")
        with patch.object(receipt._owner, "read", side_effect=AssertionError("Must reject before scoring")):
            with self.assertRaises(FileExistsError):
                stage.create_bounded_cache(receipt, output, pairs, config=config, deadline_epoch=time.time() + 60)
        self.assertEqual((output / "identity.json").read_text(), "{}")

    def test_cache_reuse_rejects_subset_coverage_and_identity_file_tamper(self):
        receipt, config, pairs = self.reference_fixture()
        one = tuple({key: value[:1] for key, value in b.items()} for b in pairs[0])
        binding = gpu.create_reference_cache(receipt, self.root / "subset-cache", [one],
            scoring_config=config, chunk_tokens=stage.CHUNK_TOKENS)
        record = {"mode": "reuse", "path": str(binding.path), "manifest_sha256": binding.manifest_sha256,
                  "identity_sha256": stage.sha(binding.path / "identity.json")}
        with self.assertRaisesRegex(ValueError, "exactly the full"):
            stage.bind_complete_cache(receipt, record, pairs, config=config, deadline_epoch=time.time() + 60)
        (binding.path / "identity.json").write_bytes((binding.path / "identity.json").read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            stage.bind_complete_cache(receipt, record, [one], config=config, deadline_epoch=time.time() + 60)

    def test_checkpoint_identity_distinguishes_new_rollout_at_same_optimizer_update(self):
        schedule = stage.BoundaryCheckpoints(2048)
        writer = Mock(side_effect=lambda name: {"checkpoint_path": name, "manifest_sha256": "a" * 64})
        prior = {"optimizer_updates": 32, "offset": 16, "completed_rollouts": 1, "active_rollout": False}
        first = schedule.save(prior, 10, writer=writer, safe_return=True)
        current = {**prior, "offset": 32, "active_rollout": True}
        second = schedule.save(current, 20, writer=writer, safe_return=True)
        self.assertNotEqual(first["checkpoint_path"], second["checkpoint_path"])
        self.assertNotEqual(first["cursor_sha256"], second["cursor_sha256"])
        schedule.save(current, 21, writer=writer, safe_return=True)
        self.assertEqual(writer.call_count, 2)

    def test_tiny_dpo_full_partial_batch_and_second_seed_reuses_without_reference_forward(self):
        with self.runtime(self.request):
            first = stage.run_alignment_stage(self.request, execute=True)
        self.assertEqual(first["status"], "formal_alignment_stage_passed")
        self.assertTrue(first["completion"]["full_accepted_dataset_pass"])
        self.assertEqual(first["completion"]["rows_processed"], 3)
        self.assertEqual(first["identity"]["config"]["optimizer"]["lr"], 5e-6)
        self.assertEqual(first["identity"]["config"]["execution_revision"], "observed-v2")
        telemetry = verify_training_telemetry(Path(first["telemetry"]["path"]).parent,
            expected_manifest_sha256=first["telemetry"]["sha256"], expected_identity_sha256=stage.digest(first["identity"]),
            expected_completion=first["completion"], checkpoints=first["checkpoints"])
        self.assertIn("cold_cache_scoring_seconds", telemetry["costs"])
        self.assertGreater(telemetry["costs"]["optimizer_active_seconds"], 0)
        cache = first["reference_cache"]
        before = {p.name: stage.sha(p) for p in Path(cache["path"]).iterdir()}
        second_request = dict(self.request, seed=43, output=str(self.root / "runs/alignment/dpo43"),
            reference_cache={k: cache[k] for k in ("path", "manifest_sha256", "identity_sha256")} | {"mode": "reuse"})
        with self.runtime(second_request, forbid_read=True):
            second = stage.run_alignment_stage(second_request, execute=True)
        self.assertEqual(second["status"], "formal_alignment_stage_passed")
        self.assertEqual(second["reference_cache"]["manifest_sha256"], cache["manifest_sha256"])
        self.assertIn("warm_cache_bind_seconds", telemetry_read(second["telemetry"]["path"])["costs"])
        self.assertNotIn("cold_cache_total_seconds", telemetry_read(second["telemetry"]["path"])["costs"])
        self.assertEqual(before, {p.name: stage.sha(p) for p in Path(cache["path"]).iterdir()})
        self.assertEqual([audit["phase"] for audit in second["frozen_full_audits"]],
                         ["before_training", "after_training", "after_export"])

    def test_tiny_ppo_subset_completion_and_fresh_model_resume_keep_oldstats_optimizer_rng(self):
        request = self.ppo_request()
        with self.runtime(request) as (_, partition):
            result = stage.run_alignment_stage(request, execute=True)
            self.assertEqual(result["status"], "formal_alignment_stage_passed")
            self.assertTrue(result["completion"]["full_selected_dataset_pass"])
            self.assertFalse(result["completion"]["full_accepted_dataset_pass"])
            self.assertEqual((result["completion"]["rows_processed"], result["completion"]["optimizer_updates"],
                              result["completion"]["completed_rollouts"]), (2, 4, 1))
            self.assertTrue(result["scalar_heads_changed"]["value"])
            manifest = verify_training_telemetry(Path(result["telemetry"]["path"]).parent,
                expected_manifest_sha256=result["telemetry"]["sha256"], expected_identity_sha256=stage.digest(result["identity"]),
                expected_completion=result["completion"], checkpoints=result["checkpoints"])
            logged = telemetry_read(manifest["rollouts"][0]["path"])
            self.assertEqual(logged["generated_at_optimizer_updates"], 0)
            self.assertEqual(logged["summary"]["responses"], 2)
            first, last = result["checkpoints"][0], result["latest_safe_checkpoint"]
            self.assertTrue(first["cursor"]["active_rollout"])
            self.assertFalse(last["cursor"]["active_rollout"])
            backend, _ = backend_for("ppo")
            for parent, role, heads in (("sft", "policy", ()), ("sft", "reference", ()),
                                        ("reward", "reward", ("reward",))):
                p = self.parent_receipts[parent]
                import_stage(p["artifact_dir"], backend, role, expected_manifest_sha256=p["manifest_sha256"],
                             expected_identity=result["identity"], heads=heads)
            mapping = parameter_mapping(backend, "policy", heads=("value",))
            optimizer = torch.optim.AdamW(mapping.values(), lr=1e-6)
            generator = torch.Generator().manual_seed(999)
            restored = load_checkpoint(first["checkpoint_path"], expected_identity=result["identity"],
                parameters=mapping, optimizer=optimizer, generators={"sampling": generator},
                expected_manifest_sha256=first["manifest_sha256"])
            self.assertIsNotNone(restored["extra_state"])
            verified = verify_checkpoint_telemetry(first["checkpoint_path"],
                expected_manifest_sha256=first["manifest_sha256"],
                expected_sidecar_sha256=first["telemetry_sidecar"]["sha256"])
            rows, _, _ = stage.select_real_rows(partition, request)
            replay = run_ppo(backend, "policy", optimizer, rows, stage.stage_config(request), device="cpu",
                generator=generator, reference_role="reference", reward_role="reward",
                state=verified["cursor"], extra_state=verified["extra_state"])
            self.assertEqual(len(replay.metrics), 3)
            saved = save_checkpoint(self.root / "cpu-replay", identity=result["identity"], parameters=mapping,
                optimizer=optimizer, cursor=replay.cursor, extra_state=replay.extra_state, generators={"sampling": generator})
            expected = torch.load(Path(last["checkpoint_path"]) / "state.pt", map_location="cpu", weights_only=True)
            actual = torch.load(Path(saved["checkpoint_path"]) / "state.pt", map_location="cpu", weights_only=True)
            expected["cursor"].pop("telemetry")
            compare_state(actual, expected, exact_float=True)
            # Rehashing a modified checkpoint must not authorize altered fixed
            # PPO targets; the pre-update rollout is a separate immutable anchor.
            for field in ("advantages", "returns", "reference_logprobs", "reward_scores"):
                original = torch.load(Path(first["checkpoint_path"]) / "state.pt", map_location="cpu", weights_only=True)
                changed = copy.deepcopy(original)
                changed["extra_state"][field].view(-1)[0] += 0.125
                target = Path(first["checkpoint_path"]) / "state.pt"
                torch.save(changed, target)
                manifest_path = Path(first["checkpoint_path"]) / "manifest.json"
                receipt = stage.read_json(manifest_path)
                receipt["artifacts"]["state.pt"] = {"sha256": stage.sha(target), "size_bytes": target.stat().st_size}
                manifest_path.write_text(json.dumps(receipt), encoding="utf-8")
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "trajectory differs"):
                    verify_checkpoint_telemetry(first["checkpoint_path"],
                        expected_manifest_sha256=stage.sha(manifest_path),
                        expected_sidecar_sha256=first["telemetry_sidecar"]["sha256"])
                torch.save(original, target)

    def test_stop_or_deadline_before_sampling_preserves_clean_zero_update_checkpoint(self):
        for mode in ("stop", "deadline"):
            request = self.ppo_request()
            request["output"] = str(Path(request["output"]).with_name("unsampled-" + mode))
            dispatched = []
            def engine(*args, **kwargs):
                observer = kwargs["event_callback"]
                def event(value):
                    observer(value)
                    if value["event"] == "sampling_start":
                        dispatched.append(True)
                kwargs["event_callback"] = event
                kwargs["stop_requested"] = lambda: mode == "stop" and bool(dispatched)
                kwargs["clock"] = lambda: request["deadline_epoch"] if mode == "deadline" and dispatched else time.time()
                return run_ppo(*args, **kwargs)
            with self.subTest(mode=mode), self.runtime(request, engine=engine):
                with self.assertRaises(InterruptedError if mode == "stop" else TimeoutError):
                    stage.run_alignment_stage(request, execute=True)
            report = stage.read_json(Path(request["output"]) / "progress.json")
            self.assertEqual(report["status"], "paused" if mode == "stop" else "budget_exhausted")
            self.assertEqual(report["actual_optimizer_updates"], 0)
            receipt = report["latest_safe_checkpoint"]
            restored = verify_checkpoint_telemetry(receipt["checkpoint_path"],
                expected_manifest_sha256=receipt["manifest_sha256"],
                expected_sidecar_sha256=receipt["telemetry_sidecar"]["sha256"])
            self.assertIsNone(restored["extra_state"])
            self.assertFalse(restored["cursor"]["active_rollout"])
            self.assertFalse((Path(request["output"]) / "telemetry/manifest.json").exists())

    def test_ppo_stop_after_collection_saves_active_rollout_even_with_zero_updates(self):
        request = self.ppo_request()
        from alignment_lab.stage_engine import sample_rollout as real_sample
        def sample(*args, **kwargs):
            result = real_sample(*args, **kwargs)
            (Path(request["output"]).parent / "STOP").write_text("pause after collecting")
            return result
        with self.runtime(request), patch("alignment_lab.stage_engine.sample_rollout", side_effect=sample):
            with self.assertRaises(InterruptedError):
                stage.run_alignment_stage(request, execute=True)
        report = stage.read_json(Path(request["output"]) / "progress.json")
        self.assertEqual(report["status"], "paused")
        self.assertEqual(report["actual_optimizer_updates"], 0)
        self.assertTrue(report["cursor"]["active_rollout"])
        self.assertTrue(report["safe_resume_available"])
        saved = torch.load(Path(report["latest_safe_checkpoint"]["checkpoint_path"]) / "state.pt",
                           map_location="cpu", weights_only=True)
        self.assertIsNotNone(saved["extra_state"])
        self.assertEqual(len(saved["extra_state"]["responses"]), 2)
        self.assertFalse((Path(request["output"]) / "final-artifact").exists())

    def test_unknown_engine_failure_retains_only_committed_checkpoint(self):
        request = self.ppo_request()
        def engine(*args, **kwargs):
            callback = kwargs["checkpoint_callback"]
            def fail(cursor, extra):
                callback(cursor, extra)
                raise RuntimeError("unknown subsequent update failure")
            kwargs["checkpoint_callback"] = fail
            return run_ppo(*args, **kwargs)
        with self.runtime(request, engine=engine):
            with self.assertRaisesRegex(RuntimeError, "subsequent update"):
                stage.run_alignment_stage(request, execute=True)
        report = stage.read_json(Path(request["output"]) / "progress.json")
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["emergency_checkpoint_attempted"])
        self.assertEqual(report["latest_safe_checkpoint"]["optimizer_updates"], 1)
        self.assertNotIn("safe_engine_return", report["latest_safe_checkpoint"]["reasons"])


if __name__ == "__main__":
    unittest.main()
