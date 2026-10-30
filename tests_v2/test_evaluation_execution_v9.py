"""Pure CPU checks for full denominators and capacity-bound production admission."""
import ast
import copy
from contextlib import ExitStack
import hashlib
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from alignment_lab import evaluation_execution_v9 as ex
from alignment_lab import evaluation_resources_v9 as resource
from alignment_lab.evaluation_protocol_v2 import canonical
from alignment_lab.evaluation_protocol_v2 import parse, digest, EvaluationProtocol
from tests_v2.test_evaluation_resources_v9 import fixture, process, pin_profile, resource_test_anchors
from tests_v2 import test_evaluation_execution as legacy_fixture

FakeModel, FakeCodec, dataset, write = (legacy_fixture.FakeModel, legacy_fixture.FakeCodec,
                                      legacy_fixture.dataset, legacy_fixture.write)


def attention_fixture(model_id, calls=1, contexts=(4,)):
    """Fabricated saved-output evidence; no actual CUDA or model work occurs."""
    from alignment_lab.evaluation_attention_receipt_v5 import KIND, BLOCKED
    receipt = {"kind": KIND, "enabled": model_id == "reward", "scope": "reward_only"}
    if model_id == "reward":
        observations = [{"batch": 1, "query_tokens": length, "key_tokens": length,
            "query_heads": 16, "original_kv_heads": 8, "head_dim": 128,
            "dtype": "torch.bfloat16", "device": "cuda:0", "mask": "implicit_full_causal",
            "backend": BLOCKED, "calls": calls} for length in contexts]
        receipt.update(cpu_fixture=False, efficient_support_checks=calls * len(contexts),
            efficient_supported=0, efficient_executed=0, blocked_math_executed=calls * len(contexts),
            max_block_rows=min(max(contexts), 128), query_block_limit=128, input_truncated=False,
            all_key_columns_retained=True, observations=observations)
    return receipt


class ExecutionTests(unittest.TestCase):
    """Versioned saved-artifact fixture shared with CPU analysis tests.

    Opaque synthetic weight bytes are never deserialized or executed. The full
    production metadata validators remain enabled; only dataset acquisition is
    replaced by the complete fabricated 541/2017/1327-record test datasets.
    """
    def setUp(self):
        old = legacy_fixture.ExecutionTests()
        old.setUp()
        old.stack.close()
        self.__dict__.update({k: v for k, v in old.__dict__.items() if not k.startswith("_test")})
        self.stack = ExitStack()
        for name in ex.REQUIRED_SOURCES:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_bytes(b"# explicit V2 CPU fixture source\n")
        self.roster.update(schema_version=2, kind="alignlab-accepted-evaluation-roster-v2")
        for model_id, stage in ex.MODEL_STAGES.items():
            record = self.roster["models"][model_id]
            accepted_path = Path(record["acceptance_path"])
            accepted = parse(accepted_path.read_bytes())
            accepted.update(schema_version=2, kind="alignlab-independent-evaluation-model-acceptance-v2")
            if stage != "base":
                path = Path(record["artifact_dir"]) / "manifest.json"
                manifest = parse(path.read_bytes())
                if stage in {"dpo", "ppo"}:
                    manifest["identity"]["config"]["purpose"] = ("alignlab-ppo-throughput-resume-v2"
                        if model_id == "ppo42" else "alignlab-formal-alignment-stage-r3-v1")
                for parent in manifest["identity"]["parents"]:
                    parent_pin = self.protocol["model_manifests"][parent]
                    manifest["identity"]["parents"][parent] = {
                        **parent_pin, "size_bytes": Path(parent_pin["path"]).stat().st_size}
                manifest["identity_sha256"] = digest(manifest["identity"])
                sha = write(path, manifest)
                self.protocol["model_manifests"][model_id]["sha256"] = sha
                record["manifest_sha256"] = sha
                row = accepted["models"][model_id]
                row["manifest_sha256"] = sha
                progress_path = Path(row["progress_path"])
                progress = parse(progress_path.read_bytes())
                progress["identity"] = manifest["identity"]
                progress["artifact"]["manifest_sha256"] = sha
                row["progress_sha256"] = write(progress_path, progress)
            record["acceptance_sha256"] = write(accepted_path, accepted)
        self.roster_sha = write(self.roster_path, self.roster)
        self.protocol_sha = write(self.protocol_path, self.protocol)
        profile, _, health, _ = fixture()
        self.resource_profile = pin_profile(self.root, profile, health)
        self.stack.enter_context(resource_test_anchors(profile))
        capacity_path = self.root / "capacity.fixture.json"
        self.capacity_manifest = {"path": str(capacity_path),
                                  "sha256": write(capacity_path, {"synthetic_saved_output_fixture_only": True})}
        self.stack.enter_context(patch.object(ex, "ROOT", self.root))
        self.stack.enter_context(patch.object(ex, "load_evaluation_assets", return_value=self.assets))
        self.stack.enter_context(patch.object(ex, "load_final_holdout", return_value=self.held))

    def tearDown(self):
        self.stack.close()
        self.temp.cleanup()

    def resource_evidence(self, epoch):
        return {"allocator": {"kind": "alignlab-cuda-allocator-ceiling-v2", "device": "cuda:0",
            "total_memory_bytes": 8192 * resource.MIB, "limit_bytes": 2560 * resource.MIB,
            "fraction": 0.3125, "configured_before_model_load": True},
            "capacity_peak_reserved_bytes": 2000 * resource.MIB,
            "resource_observations": [{"checked_at_epoch": epoch, "gpus": [{"index": 0,
                "free_mb": 1200, "total_mb": 8192, "utilization_percent": 10}],
                "available_ram_gb": 12, "blockers": [], "service_health_sha256": "f" * 64,
                "memory": {"reserved_bytes": 2000 * resource.MIB, "max_reserved_bytes": 2000 * resource.MIB}}]}

    def test_full_rm_bench_six_answers_per_record(self):
        ex._roster(self.roster, EvaluationProtocol(canonical(self.protocol)))
        self.assertEqual(ex._counts("rm_bench", len(self.rm.rows))["answer_scores"], 7962)


class ExecutionV9Tests(unittest.TestCase):
    def test_all_v8_function_bodies_preserved_except_explicit_version_routing(self):
        root = Path(__file__).resolve().parents[1]
        old = ast.parse((root / "alignment_lab/evaluation_execution_v8.py").read_text(encoding="utf-8"))
        new = ast.parse((root / "alignment_lab/evaluation_execution_v9.py").read_text(encoding="utf-8"))
        class V9RoutingOnly(ast.NodeTransformer):
            def visit_Constant(self, node):
                if isinstance(node.value, str):
                    for name in ("run_alignlab_evaluation_stage", "supervise_alignlab_evaluation"):
                        node.value = node.value.replace(name + "_v9.py", name + "_v8.py")
                return node

            def visit_ImportFrom(self, node):
                if node.module == "evaluation_capacity_inheritance_v9":
                    node.module = "evaluation_capacity_inheritance_v8"
                return self.generic_visit(node)
        normalized = V9RoutingOnly().visit(new)
        functions = lambda tree: {node.name: ast.dump(node) for node in tree.body if isinstance(node, ast.FunctionDef)}
        self.assertEqual(functions(old), functions(normalized))

    def test_every_inference_function_is_unchanged_from_v6(self):
        root = Path(__file__).resolve().parents[1]
        previous = ast.parse((root / "alignment_lab/evaluation_execution_v6.py").read_text(encoding="utf-8"))
        current = ast.parse((root / "alignment_lab/evaluation_execution_v9.py").read_text(encoding="utf-8"))
        functions = lambda tree: {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
        old, new = functions(previous), functions(current)
        self.assertEqual(old.keys(), new.keys())
        for name in old.keys() - {"_admit", "_capacity_peak"}:
            with self.subTest(function=name):
                self.assertEqual(ast.dump(old[name]), ast.dump(new[name]))
        class RoutingOnly(ast.NodeTransformer):
            def visit_Constant(self, node):
                if isinstance(node.value, str):
                    node.value = node.value.replace("run_alignlab_evaluation_stage_v9.py", "run_alignlab_evaluation_stage_v6.py")
                    node.value = node.value.replace("supervise_alignlab_evaluation_v9.py", "supervise_alignlab_evaluation_v6.py")
                return node
        self.assertEqual(ast.dump(old["_admit"]), ast.dump(RoutingOnly().visit(new["_admit"])))

    def test_v9_source_inventory_retains_all_v8_pins_and_exact_new_dependencies(self):
        from alignment_lab import evaluation_execution_v8 as previous
        additions = {"alignment_lab/evaluation_resources_v9.py", "tests_v2/test_evaluation_resources_v9.py",
            "alignment_lab/evaluation_execution_v9.py", "tests_v2/test_evaluation_execution_v9.py",
            "scripts/run_alignlab_evaluation_stage_v9.py",
            "alignment_lab/evaluation_capacity_inheritance_v9.py", "tests_v2/test_evaluation_capacity_inheritance_v9.py"}
        self.assertEqual(ex.REQUIRED_SOURCES, previous.REQUIRED_SOURCES | additions)
        self.assertEqual(ex.QUANTIZATION, previous.QUANTIZATION)
        self.assertEqual(ex.DENOMINATORS, previous.DENOMINATORS)
        self.assertEqual(ex.REQUEST_FIELDS, previous.REQUEST_FIELDS)

    def test_actual_attention_receipt_is_required_and_cpu_fixture_cannot_prove_execution(self):
        self.assertEqual(ex._attention_receipt(attention_fixture("base"), "base")["enabled"], False)
        receipt = attention_fixture("reward", calls=56, contexts=(5481, 2363))
        ex._attention_receipt(receipt, "reward", minimum_calls=56, required_contexts=(5481, 2363))
        with self.assertRaises(ValueError):
            ex._attention_receipt(None, "reward")
        with self.assertRaisesRegex(ValueError, "required full input context"):
            ex._attention_receipt(receipt, "reward", required_contexts=(5482,))
        with self.assertRaises(ValueError):
            ex._attention_receipt(receipt, "reward", minimum_calls=113)
        receipt["cpu_fixture"] = True
        with self.assertRaises(ValueError):
            ex._attention_receipt(receipt, "reward")

    def test_all_original_denominators_and_nine_model_roles_are_fixed(self):
        self.assertEqual(len(ex.MODEL_STAGES), 9)
        self.assertEqual(ex._counts("ifeval", 541), {"records": 541, "prompts": 541, "instructions": 834})
        self.assertEqual(ex._counts("helpsteer_rm", 2017), {"records": 2017, "answer_scores": 4034, "non_ties": 1920, "ties": 97})
        self.assertEqual(ex._counts("rm_bench", 1327), {"records": 1327, "answer_scores": 7962, "comparisons": 11943})

    def test_production_cannot_dispatch_without_full_capacity(self):
        with self.assertRaisesRegex(ValueError, "Real full nine-model"):
            ex._capacity_peak({"capacity_manifest": None})

    def test_capacity_is_verified_only_by_v9_bridge_api_with_unchanged_request(self):
        request = {"capacity_manifest": {"path": str(Path.cwd() / "capacity/manifest.json"), "sha256": "a" * 64},
            "resource_profile": {"path": "profile", "sha256": "b" * 64}, "deadline_epoch": 100,
            "protocol_path": "protocol", "protocol_sha256": "c" * 64,
            "roster_path": "roster", "roster_sha256": "d" * 64, "source_sha256": {}, "model_id": "base"}
        with patch("alignment_lab.evaluation_capacity_inheritance_v9.verify_capacity_inheritance_for_request",
                   return_value=2000 * ex.MIB) as verify:
            self.assertEqual(ex._capacity_peak(request), 2000 * ex.MIB)
            verify.assert_called_once_with(request["capacity_manifest"], request)
            verify.side_effect = ValueError("Capacity identity differs from production evaluation")
            request["deadline_epoch"] = 101
            with self.assertRaisesRegex(ValueError, "Capacity identity"):
                ex._capacity_peak(request)

    def test_shared_deadline_is_not_extended_at_boundary(self):
        request = {"output": str(Path.cwd() / "nonexistent-output-fixture"), "deadline_epoch": 100}
        with self.assertRaises(TimeoutError):
            ex._boundary(request, SimpleNamespace(now=lambda: 100))

    def test_live_worker_admission_keeps_services_and_rejects_foreign_training(self):
        profile, live, health, inventory = fixture()
        supervisor = process(100001, "python supervise_alignlab_evaluation_v9.py --plan frozen")
        wrapper = process(100002, "python run_alignlab_evaluation_stage_v9.py --request pinned", parent=supervisor["pid"])
        worker = process(os.getpid(), "python run_alignlab_evaluation_stage_v9.py --request pinned", parent=wrapper["pid"])
        live += [supervisor, wrapper, worker]
        system = SimpleNamespace(processes=lambda: copy.deepcopy(live))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "reports").mkdir()
            def pin(name, value):
                path = root / name
                raw = canonical(value)
                path.write_bytes(raw)
                return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
            binding = pin_profile(root, profile, health)
            request = {"resource_profile": binding, "deadline_epoch": 100}
            pin("reports/.alignlab-evaluation.lock", {"owner": "alignlab-primary-evaluation-campaign-v2",
                "resource_profile": binding, "deadline_epoch": 100, "pid": supervisor["pid"], "processes": [supervisor]})
            def collect(profile, **kwargs):
                return resource.collect_resources(profile, **kwargs, inventory_reader=lambda: inventory,
                    ram_reader=lambda: 12, health_reader=lambda url, timeout: health[url],
                    port_reader=lambda: [], now_fn=lambda: 1793000000)
            with resource_test_anchors(profile), patch.object(ex, "ROOT", root), patch.object(ex, "collect_resources", side_effect=collect):
                observation, owned = ex._admit(system, request=request, expected_peak_reserved_bytes=2000 * resource.MIB)
                self.assertEqual(observation["blockers"], [])
                self.assertEqual(len(owned), 3)
                live.append(process(100003, "python train_dpo.py"))
                with self.assertRaisesRegex(RuntimeError, "resources unavailable") as raised:
                    ex._admit(system, request=request, expected_peak_reserved_bytes=2000 * resource.MIB)
                self.assertIn('"pid":100003', str(raised.exception))
                self.assertIn('"parent_pid":1', str(raised.exception))
                self.assertIn('"command_line":"python train_dpo.py"', str(raised.exception))
                self.assertIn('"created_utc_ticks":"', str(raised.exception))


if __name__ == "__main__":
    unittest.main()
