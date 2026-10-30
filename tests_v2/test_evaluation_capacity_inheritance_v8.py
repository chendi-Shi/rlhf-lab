"""CPU-only capacity bridge and executable-AST fault injection."""
import ast
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from alignment_lab import evaluation_capacity_inheritance_v8 as inherit


def pin(root, name):
    return {"path": str(root / name), "sha256": (name.encode().hex() + "0" * 64)[:64]}


class AstTests(unittest.TestCase):
    def setUp(self):
        self.old = '''"""Original fixture."""
from .evaluation_resources_v6 import guard
REQUIRED_SOURCES = frozenset({"old.py"})
DTYPE = "bfloat16"
def _capacity_peak(request):
    return original_capacity(request)
def generate():
    return full_model(max_new_tokens=2048, truncate=False)
'''
        self.new = self.old.replace("evaluation_resources_v6", "evaluation_resources_v8")
        self.new = self.new.replace('frozenset({"old.py"})',
            "frozenset({" + ",".join(repr(n) for n in sorted({"old.py"}
                | inherit.PRESERVED_V7_EXECUTION_SOURCES | inherit.EXTRA_EXECUTION_SOURCES)) + "})")
        self.new = self.new.replace("def _capacity_peak(request):\n    return original_capacity(request)\n",
                                    inherit.CAPACITY_FUNCTION.strip() + "\n")

    def test_exact_bridge_and_source_extension(self):
        inherit.verify_execution_ast(self.old, self.new, execution=True)
        inherit.verify_execution_ast("from alignment_lab.evaluation_execution_v6 import run_evaluation_stage\n",
                                     "from alignment_lab.evaluation_execution_v8 import run_evaluation_stage\n")

    def test_dtype_horizon_truncation_and_model_changes_rejected(self):
        changes = [("bfloat16", "float16"), ("2048", "1024"), ("truncate=False", "truncate=True"),
                   ("full_model", "different_model"), ("return full_model", "raise RuntimeError"),
                   ("evaluation_resources_v8", "unsafe_resources")]
        for old, new in changes:
            with self.subTest(change=(old, new)), self.assertRaises(ValueError):
                inherit.verify_execution_ast(self.old, self.new.replace(old, new), execution=True)

    def test_capacity_bypass_or_extra_statement_rejected(self):
        for candidate in (self.new.replace("return verify_capacity_inheritance_for_request(binding, request)", "return 1"),
                          self.new + "\nconfigure_global_attention()\n",
                          self.new.replace("if binding is None:", "if False:")):
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                inherit.verify_execution_ast(self.old, candidate, execution=True)

    def test_missing_or_unreviewed_sources_rejected(self):
        candidates = [self.new.replace(repr("old.py") + ",", ""),
                      self.new.replace("'old.py'", "'unrelated.py'"),
                      self.new.replace("'alignment_lab/evaluation_resources_v7.py',", ""),
                      self.new.replace("frozenset({", "frozenset({'unreviewed.py',", 1)]
        for candidate in candidates:
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                inherit.verify_execution_ast(self.old, candidate, execution=True)


class CapacityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.binding = {k: pin(self.root, p) for k, p in (("prior_plan", "prior.json"),
            ("campaign", "runs/old/campaign.json"), ("closure", "closure.json"),
            ("bridge", "runs/old/capacity-inheritance.json"))}
        self.old_binding = {"original": "full V5 capacity"}
        self.prior = {"output": str(self.root / "runs/old"), "inherited_capacity": self.old_binding}
        self.campaign = {"capacity_manifest": self.binding["bridge"],
            "capacity": {"inherited_capacity": self.old_binding, "inheritance_verified": True}}
        self.bridge = {"kind": inherit.previous.KIND, "plan": self.binding["prior_plan"],
                       "inherited_capacity": self.old_binding}
        self.closure = {"closed": True}
        self.plan = {"inherited_capacity": self.binding, "resource_profile": pin(self.root, "resource.json")}
        self.docs = {self.binding[k]["path"]: value for k, value in (("prior_plan", self.prior),
            ("campaign", self.campaign), ("closure", self.closure), ("bridge", self.bridge))}
        self.result = {"inheritance_verified": True,
            "models": {m: {"frozen_audits": [{"full": True}, {"full": True}]} for m in inherit.MODEL_IDS},
            "max_reserved_bytes": {m: 1024 for m in inherit.MODEL_IDS}}
        for name, replacement in (("_read", lambda p: copy.deepcopy(self.docs[p["path"]])),
             ("verify_recovery_context", lambda *a, **k: (self.prior, self.campaign, self.closure, {}, [])),
             ("_sources", lambda *a: None)):
            p = patch.object(inherit, name, replacement)
            p.start()
            self.addCleanup(p.stop)
        p = patch.object(inherit.previous, "verify_inherited_capacity", side_effect=lambda *a: copy.deepcopy(self.result))
        self.old_verifier = p.start()
        self.addCleanup(p.stop)
        p = patch.dict(sys.modules, {"alignment_lab.evaluation_resources_v8": SimpleNamespace(
            load_profile=lambda *a: {"allocator_limit_mib": 2560}, MIB=1024**2)})
        p.start()
        self.addCleanup(p.stop)

    def test_transitive_complete_proof_retains_original_binding(self):
        result = inherit.verify_inherited_capacity(self.binding, self.plan)
        self.old_verifier.assert_called_once_with(self.old_binding, self.prior)
        self.assertEqual(result["original_v5_inherited_capacity"], self.old_binding)
        self.assertEqual(result["inherited_capacity"], self.binding)
        self.assertFalse(result["current_live_admission_performed"])
        self.assertTrue(result["v8_execution_ast_equivalent"])

    def test_cannot_promote_partial_or_missing_models(self):
        for changes in (lambda r: r["models"].pop("reward"),
                        lambda r: r["models"]["reward"].update(frozen_audits=[{}]),
                        lambda r: r.update(inheritance_verified=False),
                        lambda r: r["max_reserved_bytes"].pop("base"),
                        lambda r: r["max_reserved_bytes"].update(base=3000 * 1024**2),
                        lambda r: r["max_reserved_bytes"].update(base=True)):
            baseline = copy.deepcopy(self.result)
            changes(self.result)
            with self.subTest(change=changes), self.assertRaises(ValueError):
                inherit.verify_inherited_capacity(self.binding, self.plan)
            self.result.clear()
            self.result.update(baseline)

    def test_exact_bridge_and_previous_verifier_required(self):
        self.bridge["kind"] = "alignlab-evaluation-capacity-inheritance-v4-partial"
        with self.assertRaises(ValueError):
            inherit.verify_inherited_capacity(self.binding, self.plan)
        self.bridge["kind"] = inherit.previous.KIND
        self.old_verifier.side_effect = ValueError("old model artifact changed")
        with self.assertRaisesRegex(ValueError, "old model artifact"):
            inherit.verify_inherited_capacity(self.binding, self.plan)


class RequestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.plan = {"output": str(self.root / "runs/new"), "inherited_capacity": {"frozen": True},
            "source_sha256": {"new.py": "a" * 64}, "resource_profile": pin(self.root, "profile.json"),
            "deadline_epoch": 1793673011, "protocol": pin(self.root, "protocol.json"),
            "roster": pin(self.root, "roster.json")}
        self.binding = pin(self.root, "runs/new/capacity-inheritance.json")
        plan_pin = pin(self.root, "new-plan.json")
        self.bridge = {"kind": inherit.KIND, "plan": plan_pin, "inherited_capacity": self.plan["inherited_capacity"]}
        self.docs = {plan_pin["path"]: self.plan, self.binding["path"]: self.bridge}
        self.request = {"schema_version": 1, "kind": "alignlab-evaluation-inference-stage-v2",
            "task": "ifeval", "model_id": "ppo44", "output": str(self.root / "runs/new/ifeval-ppo44"),
            "capacity_manifest": self.binding,
            **{key: self.plan[key] for key in ("source_sha256", "resource_profile", "deadline_epoch")},
            **{key + suffix: self.plan[key][field] for key in ("protocol", "roster")
                for suffix, field in (("_path", "path"), ("_sha256", "sha256"))}}
        for name, replacement in (("_read", lambda p: copy.deepcopy(self.docs[p["path"]])),
            ("verify_inherited_capacity", lambda *a: {"max_reserved_bytes": {"ppo44": 2048, "reward": 4096}})):
            p = patch.object(inherit, name, replacement)
            p.start()
            self.addCleanup(p.stop)

    def test_only_remaining_complete_slots_can_consume_capacity(self):
        self.assertEqual(inherit.verify_capacity_bridge(self.binding, self.request), 2048)
        for task in ("helpsteer_rm", "rm_bench"):
            req = {**self.request, "task": task, "model_id": "reward",
                   "output": str(self.root / ("runs/new/" + task + "-reward"))}
            self.assertEqual(inherit.verify_capacity_bridge(self.binding, req), 4096)

    def test_slot_output_budget_protocol_and_source_drift_rejected(self):
        for changes in ({"model_id": "sft"}, {"task": "sample"}, {"output": str(self.root / "runs/old")},
            {"deadline_epoch": 1793673012}, {"source_sha256": {}}, {"protocol_sha256": "f" * 64},
            {"resource_profile": {}}, {"capacity_manifest": None}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                inherit.verify_capacity_bridge(self.binding, {**self.request, **changes})


if __name__ == "__main__":
    unittest.main()
