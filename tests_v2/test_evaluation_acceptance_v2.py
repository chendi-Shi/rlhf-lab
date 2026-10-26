"""Small CPU fixtures for direct-v4 acceptance; no research models or GPU work."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from alignment_lab import evaluation_acceptance_v2 as audit


class FormalMetadataTests(unittest.TestCase):
    @staticmethod
    def artifact(stage="ppo", seed=42, purpose=None):
        rows, updates = {"sft": (4000, 250), "reward": (13139, 822),
                         "dpo": (13139, 822), "ppo": (1024, 2048)}[stage]
        scope = ("registered_unique_real_prompt_subset" if stage == "ppo"
                 else "full_accepted_real_training_partition")
        config = {"stage": stage, "formal_training": True, "synthetic_rows": 0}
        complete = {"synthetic_rows": 0, "stop_reason": "data_exhausted",
                    "data_pass_complete": True, "protocol_complete": True, "scope": scope,
                    "rows_processed": rows, "optimizer_updates": updates,
                    "full_accepted_dataset_pass": stage != "ppo"}
        if stage in ("sft", "reward"):
            config.update(purpose="alignlab-formal-parent-stage-v1", scope=scope,
                          expected_rows=rows, expected_updates=updates)
        else:
            config.update(purpose=purpose or "alignlab-formal-alignment-stage-r3-v1",
                          work={"seed": seed}, registered_workload={
                              "rows": rows, "optimizer_updates": updates, "scope": scope})
            complete["full_selected_dataset_pass"] = True
        return {"stage": stage, "identity": {"config": config}, "completion": complete}

    @staticmethod
    def progress(manifest, status=None):
        return {"status": status or ("formal_parent_stage_passed" if manifest["stage"] in ("sft", "reward")
                                     else "formal_alignment_stage_passed"),
                "formal_training": True, "research_complete": False,
                "identity": copy.deepcopy(manifest["identity"]),
                "completion": copy.deepcopy(manifest["completion"]),
                "actual_optimizer_updates": manifest["completion"]["optimizer_updates"]}

    def test_completed_r3_dpo_and_ppo_accept_each_registered_seed(self):
        for stage in ("dpo", "ppo"):
            for seed in (42, 43, 44):
                with self.subTest(stage=stage, seed=seed):
                    manifest = self.artifact(stage, seed)
                    audit.validate_formal_artifact(manifest, model_id=f"{stage}{seed}")
                    audit.validate_formal_progress(self.progress(manifest), manifest, model_id=f"{stage}{seed}")

    def test_completed_ppo42_throughput_resume_purpose_is_accepted(self):
        manifest = self.artifact(purpose="alignlab-ppo-throughput-resume-v2")
        audit.validate_formal_artifact(manifest, model_id="ppo42")
        audit.validate_formal_progress(self.progress(manifest), manifest, model_id="ppo42")

    def test_wrong_label_seed_unknown_seed_and_non42_resume_are_rejected(self):
        cases = ((self.artifact(seed=43), "ppo42"), (self.artifact(seed=45), None),
                 (self.artifact(seed=43, purpose="alignlab-ppo-throughput-resume-v2"), "ppo43"),
                 (self.artifact("dpo", purpose="alignlab-ppo-throughput-resume-v2"), "dpo42"))
        for manifest, model_id in cases:
            with self.subTest(config=manifest["identity"]["config"], model_id=model_id), self.assertRaises(ValueError):
                audit.validate_formal_artifact(manifest, model_id=model_id)

    def test_unknown_alignment_purpose_is_rejected(self):
        for stage in ("dpo", "ppo"):
            with self.subTest(stage=stage), self.assertRaises(ValueError):
                audit.validate_formal_artifact(self.artifact(stage, purpose="fixture-unknown-purpose"))

    def test_ppo_registered_workload_and_completed_counts_are_both_required(self):
        changes = (("registered_workload", "rows", 13139), ("registered_workload", "optimizer_updates", 822),
                   ("registered_workload", "scope", "full_accepted_real_training_partition"),
                   ("completion", "rows_processed", 1023), ("completion", "optimizer_updates", 2047),
                   ("completion", "full_accepted_dataset_pass", True))
        for section, field, value in changes:
            with self.subTest(section=section, field=field):
                manifest = self.artifact()
                target = manifest["completion"] if section == "completion" else manifest["identity"]["config"][section]
                target[field] = value
                with self.assertRaises(ValueError):
                    audit.validate_formal_artifact(manifest, model_id="ppo42")

    def test_r5_recovery_status_requires_reward_recovery_config(self):
        manifest = self.artifact("reward")
        manifest["identity"]["config"]["recovery"] = {"fixture_only": True}
        audit.validate_formal_progress(self.progress(manifest, "formal_parent_recovery_passed"),
                                       manifest, model_id="reward")
        for recovery in (None, "unbound-recovery"):
            with self.subTest(recovery=recovery):
                manifest["identity"]["config"]["recovery"] = recovery
                with self.assertRaises(ValueError):
                    audit.validate_formal_progress(self.progress(manifest, "formal_parent_recovery_passed"),
                                                   manifest, model_id="reward")
        del manifest["identity"]["config"]["recovery"]
        with self.assertRaises(ValueError):
            audit.validate_formal_progress(self.progress(manifest, "formal_parent_recovery_passed"),
                                           manifest, model_id="reward")

    def test_progress_cannot_claim_different_counts_or_research_completion(self):
        manifest = self.artifact()
        for key, value in (("actual_optimizer_updates", 2047), ("research_complete", True),
                           ("status", "formal_parent_recovery_passed")):
            with self.subTest(key=key):
                progress = self.progress(manifest)
                progress[key] = value
                with self.assertRaises(ValueError):
                    audit.validate_formal_progress(progress, manifest, model_id="ppo42")


class ProcessHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name) / "completed-training"
        self.output.mkdir()
        self.process = {
            "pid": 101, "parent_pid": 100,
            "created_at_utc": "2026-10-25T01:00:00.1234567Z",
            "command_line": "python run_alignlab_throughput_direct_v4.py --execute",
        }

    def check(self, current):
        return audit._process_history([self.process], current, [str(self.output)])

    def test_exited_historical_process_is_accepted(self):
        self.assertEqual(len(self.check([])), 1)

    def test_live_identity_is_rejected_even_if_command_or_parent_changed(self):
        variants = ({}, {"command_line": "python harmless.py"}, {"parent_pid": 999})
        for mutation in variants:
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.check([{**self.process, **mutation}])

    def test_reused_pid_requires_different_full_precision_creation_time(self):
        reused = {**self.process, "created_at_utc": "2026-10-25T01:00:00.1234568Z",
                  "command_line": "python harmless.py"}
        self.assertEqual(len(self.check([reused])), 1)

    def test_new_writer_to_protected_training_output_is_rejected(self):
        for command in ("run_alignlab_alignment.py", "train.py", "bootstrap_run.py"):
            process = {**self.process, "pid": 202,
                       "command_line": f'python {command} --output "{self.output}"'}
            with self.subTest(command=command), self.assertRaises(ValueError):
                self.check([process])

    def test_unrelated_process_does_not_reanimate_exited_history(self):
        process = {**self.process, "pid": 202, "command_line": "python harmless.py"}
        self.assertEqual(len(self.check([process])), 1)


class FinalReceiptTests(unittest.TestCase):
    def setUp(self):
        self.plan = {"path": str(Path("fixture-plan.json").resolve()), "sha256": "a" * 64}
        self.execution = {"path": str(Path("fixture-execution.json").resolve()), "sha256": "b" * 64}
        self.receipt = {
            "kind": "alignlab-direct-v4-final-inherited-acceptance", "status": "passed",
            "plan": copy.deepcopy(self.plan), "execution_manifest": copy.deepcopy(self.execution),
            "started_at_epoch": 150.0, "finished_at_epoch": 200.0,
            "stage_names": ["dpo42", "ppo42", "dpo43", "ppo43"],
        }
        self.campaign = {
            "started_at_epoch": 100.0, "finished_at_epoch": 210.0, "deadline_epoch": 250.0,
            "inherited_prefix_reverified": True,
            "stages": [
                {"name": name, "finished_at_epoch": finished}
                for name, finished in (("dpo42", 50.0), ("ppo42", 60.0),
                                       ("dpo43", 70.0), ("ppo43", 80.0),
                                       ("dpo44", 120.0), ("ppo44", 140.0))
            ],
        }

    def check(self):
        return audit._validate_final_receipt(self.receipt, self.plan, self.execution, self.campaign)

    def test_exact_receipt_bound_to_both_authorities_is_accepted(self):
        self.check()

    def test_plan_and_execution_paths_and_hashes_cannot_be_substituted(self):
        for authority in ("plan", "execution_manifest"):
            for key, value in (("path", str(Path("different.json").resolve())), ("sha256", "c" * 64)):
                with self.subTest(authority=authority, key=key):
                    before = copy.deepcopy(self.receipt)
                    self.receipt[authority][key] = value
                    with self.assertRaises(ValueError):
                        self.check()
                    self.receipt = before

    def test_inherited_stages_cannot_be_missing_duplicated_reordered_or_extended(self):
        names = self.receipt["stage_names"]
        for changed in (names[:-1], [*names[:-1], "dpo43"], list(reversed(names)), [*names, "dpo44"]):
            with self.subTest(changed=changed):
                self.receipt["stage_names"] = changed
                with self.assertRaises(ValueError):
                    self.check()

    def test_failed_or_unverified_receipt_cannot_authorize_models(self):
        for key, value in (("kind", "some-other-receipt"), ("status", "failed")):
            original = self.receipt[key]
            with self.subTest(key=key):
                self.receipt[key] = value
                with self.assertRaises(ValueError):
                    self.check()
                self.receipt[key] = original
        self.campaign["inherited_prefix_reverified"] = False
        with self.assertRaises(ValueError):
            self.check()

    def test_receipt_times_must_be_finite_ordered_and_after_seed44_completion(self):
        for key, value in (("started_at_epoch", float("nan")), ("finished_at_epoch", float("inf")),
                           ("started_at_epoch", 139.0), ("started_at_epoch", 201.0),
                           ("finished_at_epoch", 211.0)):
            original = self.receipt[key]
            with self.subTest(key=key, value=value):
                self.receipt[key] = value
                with self.assertRaises(ValueError):
                    self.check()
                self.receipt[key] = original

    def test_campaign_cannot_finish_after_its_original_deadline(self):
        self.campaign["finished_at_epoch"] = 251.0
        with self.assertRaises(ValueError):
            self.check()


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.document = self.root / "fixture.json"
        self.value = {"fixture_only": True, "count": 2}
        self.document.write_text(json.dumps(self.value), encoding="utf-8")
        self.digest = hashlib.sha256(self.document.read_bytes()).hexdigest()
        self.pin = {"path": str(self.document), "sha256": self.digest}

    def test_deduplicates_hashing_only_within_one_acceptance_pass(self):
        with patch.object(audit.legacy, "_file", wraps=audit.legacy._file) as file_hash:
            evidence = audit.Evidence()
            evidence.add(self.document, self.digest)
            self.assertEqual(evidence.read(self.pin), self.value)
            evidence.add(self.document, self.digest)
            self.assertEqual(file_hash.call_count, 1)
            self.assertEqual(set(evidence.inventory), {str(self.document)})
            audit.Evidence().read(self.pin)
            self.assertEqual(file_hash.call_count, 2)

    def test_cached_identity_rejects_conflicting_digest_and_file_drift(self):
        evidence = audit.Evidence()
        evidence.add(self.document, self.digest)
        with self.assertRaises(ValueError):
            evidence.add(self.document, "f" * 64)
        self.document.write_bytes(b"changed and differently sized fixture")
        changed_digest = hashlib.sha256(self.document.read_bytes()).hexdigest()
        with self.assertRaises(ValueError):
            evidence.add(self.document, changed_digest)
        fresh = audit.Evidence().add(self.document, changed_digest)
        self.assertEqual(fresh["sha256"], changed_digest)

    def test_read_requires_external_absolute_hash_pin_and_checks_content(self):
        for pin in ({"path": "fixture.json", "sha256": self.digest},
                    {"path": str(self.document)},
                    {"path": str(self.document), "sha256": "e" * 64}):
            with self.subTest(pin=pin), self.assertRaises(ValueError):
                audit.Evidence().read(pin)

    def test_source_inventory_authenticates_python_files_inside_project(self):
        source = self.root / "fixture.py"
        source.write_bytes(b"# CPU fixture source\n")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        with patch.object(audit, "ROOT", self.root):
            evidence = audit.Evidence()
            evidence.source_inventory({"fixture.py": digest})
            self.assertEqual(evidence.inventory[str(source)]["sha256"], digest)
            for sources in ({}, {"fixture.py": "e" * 64}, {"../outside.py": digest},
                            {"fixture.json": self.digest}):
                with self.subTest(sources=sources), self.assertRaises(ValueError):
                    audit.Evidence().source_inventory(sources)


class EvidenceResolutionTests(unittest.TestCase):
    """Resolve tiny metadata pins using each containing document's location."""
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        change = patch.object(audit, "ROOT", self.root)
        change.start()
        self.addCleanup(change.stop)

    def write(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(audit.legacy._canonical(value))
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def test_snapshot_metadata_pin_resolves_beside_its_source_document(self):
        metadata = self.write("models/snapshot/upstream-metadata.json", {"fixture_only": True})
        document = self.write("models/snapshot/snapshot-manifest.json", {
            "upstream": {**metadata, "path": "upstream-metadata.json"}})
        evidence = audit.Evidence()
        evidence.include(document)
        self.assertEqual(set(evidence.inventory), {document["path"], metadata["path"]})
        self.assertEqual(evidence.documents[metadata["path"]], {"fixture_only": True})

    def test_project_relative_pin_has_unique_root_fallback(self):
        metadata = self.write("data/source.json", {"fixture_only": True})
        document = self.write("reports/frozen/plan.json", {"source": {**metadata, "path": "data/source.json"}})
        evidence = audit.Evidence()
        evidence.include(document)
        self.assertEqual(set(evidence.inventory), {document["path"], metadata["path"]})

    def test_distinct_document_and_root_candidates_are_ambiguous_even_with_same_hash(self):
        metadata = self.write("models/snapshot/upstream.json", {"fixture_only": True})
        other = self.write("upstream.json", {"fixture_only": True})
        self.assertEqual(metadata["sha256"], other["sha256"])
        document = self.write("models/snapshot/manifest.json", {"upstream": {**metadata, "path": "upstream.json"}})
        with self.assertRaises(ValueError):
            audit.Evidence().include(document)

    def test_traversal_drive_missing_context_and_missing_reference_are_rejected(self):
        metadata = self.write("upstream.json", {"fixture_only": True})
        document = self.write("models/snapshot/manifest.json", {})
        evidence = audit.Evidence()
        evidence.include(document)
        for relative in ("../upstream.json", "..\\upstream.json", "C:upstream.json", "child:upstream.json", "missing.json"):
            with self.subTest(relative=relative), self.assertRaises(ValueError):
                evidence.include({**metadata, "path": relative}, document=document["path"])
        with self.assertRaises(ValueError):
            evidence.include({**metadata, "path": "upstream.json"})
        # A simulated symlink avoids requiring Windows symlink privileges.
        original = Path.is_symlink
        with patch.object(Path, "is_symlink", lambda path: path == Path(metadata["path"]) or original(path)):
            with self.assertRaises(ValueError):
                evidence.include({**metadata, "path": "upstream.json"}, document=document["path"])

    def test_nested_relative_include_uses_each_child_document_context(self):
        leaf = self.write("models/parts/leaf.json", {"fixture_only": "actual-child-context"})
        decoy = self.write("models/leaf.json", {"fixture_only": "wrong-parent-context"})
        child = self.write("models/parts/child.json", {"leaf": {**leaf, "path": "leaf.json"}})
        document = self.write("models/manifest.json", {"child": {**child, "path": "parts/child.json"}})
        evidence = audit.Evidence()
        evidence.include(document)
        self.assertEqual(set(evidence.inventory), {document["path"], child["path"], leaf["path"]})
        self.assertNotIn(decoy["path"], evidence.inventory)


class HistoricalEvidenceBoundaryTests(unittest.TestCase):
    """Historical JSON stays pinned without treating its old sources as live."""
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        change = patch.object(audit, "ROOT", self.root)
        change.start()
        self.addCleanup(change.stop)

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value if isinstance(value, bytes) else audit.legacy._canonical(value))
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def test_shallow_diagnostic_keeps_own_pin_without_revalidating_old_source(self):
        source = self.write("auditor.py", b"# current source\n")
        old_digest = hashlib.sha256(b"# historical source\n").hexdigest()
        diagnostic = self.write("history/diagnostic.json", {"source_sha256": {"auditor.py": old_digest}})
        evidence = audit.Evidence()
        evidence.include(diagnostic, expand_pins=False)
        self.assertEqual(set(evidence.inventory), {diagnostic["path"]})
        self.assertNotIn(source["path"], evidence.inventory)
        Path(diagnostic["path"]).write_bytes(b'{"changed_diagnostic":true}')
        for verifier in (evidence, audit.Evidence()):
            with self.subTest(cached=verifier is evidence), self.assertRaises(ValueError):
                verifier.include(diagnostic, expand_pins=False)

    def test_shallow_walk_still_authenticates_current_embedded_source_maps(self):
        source = self.write("auditor.py", b"# current source\n")
        current = {"identity": {"source_sha256": {"auditor.py": source["sha256"]}}}
        evidence = audit.Evidence()
        evidence.walk(current, expand_pins=False)
        self.assertIn(source["path"], evidence.inventory)
        current["identity"]["source_sha256"]["auditor.py"] = "f" * 64
        with self.assertRaises(ValueError):
            audit.Evidence().walk(current, expand_pins=False)

    def test_shallow_walk_still_rejects_direct_model_input_pin_drift(self):
        for name, original, changed in (
                ("tokenizer-config.json", b'{"fixture":1}', b'{"fixture":2}'),
                ("merges.txt", b"fixture merge one\n", b"fixture merge two\n")):
            with self.subTest(input=name):
                pin = self.write(name, original)
                Path(pin["path"]).write_bytes(changed)
                with self.assertRaises(ValueError):
                    audit.Evidence().walk({"model_input": pin}, expand_pins=False)

    def test_shallow_and_deep_references_cannot_conflict_on_one_document_hash(self):
        pin = self.write("diagnostic.json", {"fixture_only": True})
        for first_expand in (False, True):
            with self.subTest(first_expand=first_expand):
                evidence = audit.Evidence()
                evidence.include(pin, expand_pins=first_expand)
                with self.assertRaises(ValueError):
                    evidence.include({**pin, "sha256": "e" * 64}, expand_pins=not first_expand)

    def test_deep_walk_upgrades_shallow_document_and_rejects_changed_nested_report(self):
        mutable = self.write("history/current-report.json", {"fixture_version": 1})
        diagnostic = self.write("history/diagnostic.json", {"old_report": mutable})
        self.write("history/current-report.json", {"fixture_version": 2})
        embedded = {"historical_diagnostic": diagnostic}
        evidence = audit.Evidence()
        evidence.walk(embedded, expand_pins=False)
        self.assertEqual(set(evidence.inventory), {diagnostic["path"]})
        for verifier in (evidence, audit.Evidence()):
            with self.subTest(upgrade=verifier is evidence), self.assertRaises(ValueError):
                verifier.walk(embedded, expand_pins=True)


class FinalPayloadTests(unittest.TestCase):
    """Two tiny CPU tensors exercise actual safe decoding and exact comparison."""
    def setUp(self):
        import torch
        self.torch = torch
        self.assertFalse(torch.cuda.is_initialized())
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.artifact_dir, self.checkpoint_dir = root / "artifact", root / "checkpoint"
        self.artifact_dir.mkdir()
        self.checkpoint_dir.mkdir()
        self.progress = {"cursor": {"schema": 1, "optimizer_updates": 2, "offset": 4},
                         "completion": {"optimizer_updates": 2}}
        self.parameters = {
            "base.layer.lora_A.policy.weight": torch.tensor([0.25, -0.5], device="cpu"),
            "scalar_heads.value.weight": torch.tensor([1.0], device="cpu"),
        }
        self.checkpoint = {"parameters": self.parameters, "optimizer": {}, "rng": {},
                           "cursor": copy.deepcopy(self.progress["cursor"]), "extra_state": None}
        self.exported = {"adapter/base.layer.lora_A.weight": self.parameters[
                            "base.layer.lora_A.policy.weight"].clone(),
                         "heads/value/weight": self.parameters["scalar_heads.value.weight"].clone()}
        self.artifact_manifest = {"source_role": "policy", "heads": ["value"]}
        self.checkpoint_manifest = {}
        self.save()

    def save(self):
        for value, directory, filename, manifest in (
                (self.checkpoint, self.checkpoint_dir, "state.pt", self.checkpoint_manifest),
                (self.exported, self.artifact_dir, "weights.pt", self.artifact_manifest)):
            path = directory / filename
            self.torch.save(value, path)
            manifest["artifacts"] = {filename: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                                "size_bytes": path.stat().st_size}}
        self.artifact_manifest["tensor_specs"] = [
            {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
            for name, tensor in self.exported.items()
        ]

    def check(self):
        result = audit._verify_final_payload(self.progress, self.artifact_manifest,
            self.checkpoint_manifest, self.artifact_dir, self.checkpoint_dir)
        self.assertFalse(self.torch.cuda.is_initialized())
        return result

    def test_actual_cpu_export_matches_final_checkpoint_and_value_head(self):
        result = self.check()
        self.assertTrue(result["actual_cpu_payloads_verified"])
        self.assertTrue(result["final_checkpoint_export_equal"])
        self.assertEqual(result["tensor_count"], 2)
        self.assertEqual(result["optimizer_updates"], 2)

    def test_rehashed_export_value_dtype_shape_or_tensor_set_difference_is_rejected(self):
        key = "adapter/base.layer.lora_A.weight"
        baseline = copy.deepcopy(self.exported)
        mutations = (
            lambda: self.exported[key].add_(0.125),
            lambda: self.exported.update({key: self.exported[key].to(self.torch.float64)}),
            lambda: self.exported.update({key: self.exported[key].reshape(1, 2)}),
            lambda: self.exported.pop("heads/value/weight"),
            lambda: self.exported.update({"heads/reward/weight": self.torch.tensor([1.0], device="cpu")}),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                self.exported = copy.deepcopy(baseline)
                mutate()
                self.save()
                with self.assertRaises(ValueError):
                    self.check()

    def test_equal_nonfinite_checkpoint_and_export_are_rejected(self):
        for nonfinite in (float("inf"), float("-inf"), float("nan")):
            with self.subTest(nonfinite=nonfinite):
                self.parameters["base.layer.lora_A.policy.weight"][0] = nonfinite
                self.exported["adapter/base.layer.lora_A.weight"][0] = nonfinite
                self.save()
                with self.assertRaises(ValueError):
                    self.check()

    def test_completed_progress_must_match_actual_payload_cursor(self):
        self.checkpoint["cursor"]["optimizer_updates"] = 1
        self.save()
        with self.assertRaises(ValueError):
            self.check()

    def test_active_rollout_or_wrong_payload_schema_is_rejected(self):
        self.checkpoint["extra_state"] = {"active_rollout": True}
        self.save()
        with self.assertRaises(ValueError):
            self.check()
        self.checkpoint["extra_state"] = None
        self.checkpoint["unrecognized"] = 1
        self.save()
        with self.assertRaises(ValueError):
            self.check()

    def test_payload_hash_is_verified_before_deserialization(self):
        for manifest, filename in ((self.checkpoint_manifest, "state.pt"),
                                   (self.artifact_manifest, "weights.pt")):
            with self.subTest(filename=filename):
                original = manifest["artifacts"][filename]["sha256"]
                manifest["artifacts"][filename]["sha256"] = "e" * 64
                with self.assertRaises(ValueError):
                    self.check()
                manifest["artifacts"][filename]["sha256"] = original

    def test_export_specs_and_owned_head_are_enforced(self):
        self.artifact_manifest["tensor_specs"][0]["shape"] = [999]
        with self.assertRaises(ValueError):
            self.check()
        self.save()
        self.artifact_manifest["heads"] = []
        with self.assertRaises(ValueError):
            self.check()


class RosterStructureTests(unittest.TestCase):
    """Mocked metadata closure only; these are not an actual nine-model audit."""
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        (self.root / "reports").mkdir()
        source = self.root / "auditor_fixture.py"
        source.write_bytes(b"# Structural roster fixture only\n")
        source_pin = audit.legacy._file(source)
        for name, value in (("ROOT", self.root), ("SOURCES", (source.name,))):
            change = patch.object(audit, name, value)
            change.start()
            self.addCleanup(change.stop)
        models = {
            name: {"stage": stage, "artifact_dir": None if name == "base" else str(self.root / name),
                   "manifest_sha256": None if name == "base" else "d" * 64,
                   "progress_path": None, "progress_sha256": None}
            for name, stage in audit.MODELS.items()
        }
        self.context = {
            "models": models, "evidence_files": {str(source): source_pin},
            "historical_tensor_pins": {"archived-fixture.pt": {"fixture_only": True}},
            "final_payloads": {name: {"fixture_only": True} for name in models if name != "base"},
            "protected_paths": [str(self.root / "training")], "model_snapshot": str(self.root / "base"),
            "model_manifest_sha256": "c" * 64, "tokenizer_dir": str(self.root / "tokenizer"),
        }
        self.accepted = {
            **copy.deepcopy(self.context), "kind": audit.KIND, "schema_version": 2,
            "status": "accepted_for_evaluation", "acceptance_policy": audit.POLICY,
            "models_verified": True, "all_checks_passed": True, "processes_exited": True,
            "actual_cpu_payloads_verified": True, "cuda_initialized": False, "device": "cpu",
            "bindings": {"fixture_only": True}, "auditor_source_sha256": {source.name: source_pin["sha256"]},
            "payload_audits": {name: {"actual_cpu_payloads_verified": True,
                                      "final_checkpoint_export_equal": True}
                               for name in models if name != "base"},
        }
        self.accepted_path, self.roster_path = self.root / "accepted.json", self.root / "roster.json"
        accepted_pin = self.write(self.accepted_path, self.accepted)
        self.roster = {
            "schema_version": 2, "kind": audit.ROSTER_KIND,
            **{key: self.context[key] for key in ("model_snapshot", "model_manifest_sha256", "tokenizer_dir")},
            "models": {name: {"model_id": name,
                              **{key: row[key] for key in ("stage", "artifact_dir", "manifest_sha256")},
                              "acceptance_path": accepted_pin["path"], "acceptance_sha256": accepted_pin["sha256"]}
                       for name, row in models.items()},
        }
        change = patch.object(audit, "_context", side_effect=lambda *args, **kwargs: copy.deepcopy(self.context))
        self.rebuild = change.start()
        self.addCleanup(change.stop)

    def write(self, path, value):
        path.write_bytes(audit.legacy._canonical(value))
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def check(self, **kwargs):
        pin = self.write(self.roster_path, self.roster)
        return audit.verify_evaluation_roster(pin["path"], pin["sha256"], **kwargs)

    def rehash_acceptance(self):
        pin = self.write(self.accepted_path, self.accepted)
        for row in self.roster["models"].values():
            row.update(acceptance_path=pin["path"], acceptance_sha256=pin["sha256"])

    def test_exact_fixture_roster_reconstructs_metadata_without_payload_audit(self):
        self.assertEqual(self.check(), self.roster)
        self.assertFalse(self.rebuild.call_args.kwargs.get("payload_audit", False))

    def test_roster_requires_all_nine_models_and_one_acceptance_pin(self):
        original = copy.deepcopy(self.roster)
        del self.roster["models"]["reward"]
        with self.assertRaises(ValueError):
            self.check()
        self.roster = original
        self.roster["models"]["reward"]["acceptance_sha256"] = "e" * 64
        with self.assertRaises(ValueError):
            self.check()

    def test_rehashed_acceptance_cannot_thin_evidence_or_final_payload_proofs(self):
        original = copy.deepcopy(self.accepted)
        for key in ("evidence_files", "historical_tensor_pins", "final_payloads", "payload_audits"):
            with self.subTest(key=key):
                self.accepted = copy.deepcopy(original)
                self.accepted[key] = {}
                self.rehash_acceptance()
                with self.assertRaises(ValueError):
                    self.check()

    def test_rehashed_roster_cannot_substitute_model_or_protocol_version(self):
        original = copy.deepcopy(self.roster)
        self.roster["models"]["dpo42"]["artifact_dir"] = str(self.root / "replacement")
        with self.assertRaises(ValueError):
            self.check()
        self.roster = original
        self.roster["schema_version"] = 1
        with self.assertRaises(ValueError):
            self.check()

    def test_bound_file_verification_cannot_be_disabled(self):
        with self.assertRaises(ValueError):
            self.check(verify_bound_files=False)
        self.rebuild.assert_not_called()

    def test_failed_context_emits_diagnostic_without_accepted_models_or_roster(self):
        output = self.root / "reports" / "failed-fixture-audit"
        self.rebuild.side_effect = ValueError("fixture evidence is incomplete")
        result = audit.audit_evaluation_models(self.root / "plan.json", "a" * 64,
            self.root / "campaign.json", "b" * 64, self.root / "execution.json", "c" * 64,
            output_dir=output)
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["models_verified"])
        self.assertEqual({path.name for path in output.iterdir()}, {"report.json"})


if __name__ == "__main__":
    unittest.main()
