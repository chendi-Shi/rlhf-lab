"""Fault injection for V7 history authorization; artificial CPU-only evidence."""
import copy
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from alignment_lab import evaluation_inheritance_v7 as inherit


def proc(pid):
    return {"pid": pid, "parent_pid": 1, "created_at_utc": "2026-10-29T02:34:38.4723310Z",
            "created_utc_ticks": "639288380784723310", "command_line": f"python frozen-{pid}.py"}


def pin(root, name):
    return {"path": str(root / name), "sha256": (name.encode().hex() + "0" * 64)[:64]}


def process_snapshot_fixture(request_pin):
    """Reproduce V6 DPO42's actual runner/encoder versus manifest topology."""
    command = ["project-python.exe", "-I", "-B", "-u", "scripts/run_alignlab_evaluation_stage_v6.py",
        "--request", request_pin["path"], "--request-sha256", request_pin["sha256"], "--execute"]
    supervisors = [{**proc(51336), "parent_pid": 41256}, {**proc(47548), "parent_pid": 51336}]
    wrapper = {**proc(51908), "parent_pid": 47548, "command_line": subprocess.list2cmdline(command)}
    worker = {**proc(42452), "parent_pid": 51908,
        "command_line": subprocess.list2cmdline(["cache python.exe", *command[1:]])}
    encoder_command = ["project-python.exe", "-I", "-B",
        str(inherit.ROOT / "scripts/encode_alignlab_evaluation_capacity_v2.py")]
    encoders = [{**proc(46572), "parent_pid": 42452,
        "command_line": subprocess.list2cmdline(encoder_command)},
        {**proc(15944), "parent_pid": 46572,
         "command_line": subprocess.list2cmdline(["cache python.exe", *encoder_command[1:]])}]
    accepted = {"wrapper_pid": wrapper["pid"], "command": command,
        "request_path": request_pin["path"], "request_sha256": request_pin["sha256"],
        "known_processes": [wrapper, worker, *encoders]}
    manifest = {"processes": [wrapper, worker, *reversed(supervisors)], "pid": worker["pid"]}
    campaign = {"supervisor_processes": supervisors}
    return accepted, manifest, campaign, [wrapper, worker, *encoders, *supervisors]


class ProcessSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.args = process_snapshot_fixture({"path": str(Path.cwd() / "request.json"), "sha256": "a" * 64})

    def test_actual_v6_encoder_history_is_not_part_of_final_admission_snapshot(self):
        accepted, manifest, campaign, closed = self.args
        self.assertEqual([p["pid"] for p in accepted["known_processes"]], [51908, 42452, 46572, 15944])
        self.assertEqual([p["pid"] for p in manifest["processes"]], [51908, 42452, 47548, 51336])
        inherit._verify_original_processes(accepted, manifest, campaign, closed)

    def test_no_encoder_observed_is_valid_without_inventing_processes(self):
        accepted, manifest, campaign, closed = self.args
        accepted["known_processes"] = accepted["known_processes"][:2]
        inherit._verify_original_processes(accepted, manifest, campaign, closed)

    def test_manifest_must_have_exact_runner_and_supervisor_pairs(self):
        mutations = [lambda a, m, c, x: m["processes"].pop(0),
            lambda a, m, c, x: m["processes"].pop(),
            lambda a, m, c, x: m["processes"].append(copy.deepcopy(a["known_processes"][-1])),
            lambda a, m, c, x: m.update(pid=46572),
            lambda a, m, c, x: m["processes"].append(copy.deepcopy(m["processes"][0]))]
        for mutate in mutations:
            args = copy.deepcopy(self.args)
            mutate(*args)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                inherit._verify_original_processes(*args)

    def test_runner_command_request_or_ancestry_substitution_rejected(self):
        mutations = [lambda a, m, c, x: a["command"].__setitem__(8, "b" * 64),
            lambda a, m, c, x: a["known_processes"][0].update(command_line="python unrelated.py"),
            lambda a, m, c, x: a["known_processes"][1].update(parent_pid=51336),
            lambda a, m, c, x: c["supervisor_processes"][1].update(parent_pid=999),
            lambda a, m, c, x: a.update(wrapper_pid=46572)]
        for mutate in mutations:
            args = copy.deepcopy(self.args)
            mutate(*args)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                inherit._verify_original_processes(*args)

    def test_encoder_requires_exact_command_worker_ancestry_and_closed_identity(self):
        mutations = [lambda a, m, c, x: a["known_processes"][2].update(command_line="python -I -B unknown.py"),
            lambda a, m, c, x: a["known_processes"][2].update(
                command_line=a["known_processes"][2]["command_line"] + " --unbound"),
            lambda a, m, c, x: a["known_processes"][2].update(parent_pid=47548),
            lambda a, m, c, x: a["known_processes"][3].update(parent_pid=999),
            lambda a, m, c, x: x.pop(2),
            lambda a, m, c, x: a["known_processes"].append(copy.deepcopy(a["known_processes"][2]))]
        for mutate in mutations:
            args = copy.deepcopy(self.args)
            mutate(*args)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                inherit._verify_original_processes(*args)


def campaign_fixture(root):
    prior_pin = pin(root, "prior.json")
    bridge = pin(root, "runs/old/capacity-inheritance.json")
    prior = {"kind": "alignlab-primary-evaluation-campaign-v2", "output": str(root / "runs/old"),
        "inherited_capacity": {"old": "capacity"}, "resource_profile": pin(root, "profile.json")}
    campaign = {"kind": prior["kind"], "status": "failed", "capacity_verified": True,
        "started_at_epoch": inherit.STARTED, "deadline_epoch": inherit.DEADLINE,
        "primary_evaluation_complete": False, "research_complete": False, "formal_training": False,
        "plan_path": prior_pin["path"], "plan_sha256": prior_pin["sha256"],
        "scores": {slot: pin(root, slot.replace("/", "-") + "/scores.json") for slot in inherit.SLOTS},
        "inherited_inference": {"ifeval/base": {}}, "capacity_manifest": bridge,
        "inherited_capacity": {"bindings": prior["inherited_capacity"], "bridge": bridge},
        "supervisor_processes": [proc(10), proc(11)], "children": []}
    names = [("ifeval-base-score", "score")]
    for model in inherit.POLICIES[1:]:
        names.extend((("ifeval-" + model, "inference"), ("ifeval-" + model + "-score", "score")))
    names.append(("ifeval-ppo44", "inference"))
    for i, (name, phase) in enumerate(names):
        campaign["children"].append({"name": name, "phase": phase,
            "status": "accepted" if i < 13 else "dispatching", "exit_code": 0 if i < 13 else 1,
            "active_processes": [], "known_processes": [proc(20 + i)],
            "result": pin(root, name + "/result.json") if i < 13 else None,
            "monitoring_error": None, "stop_reason": None})
    binding = {"prior_plan": prior_pin, "campaign": pin(root, "runs/old/campaign.json"),
               "closure": pin(root, "closure.json")}
    return prior, campaign, binding


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prior, self.campaign, self.binding = campaign_fixture(self.root)

    def test_seven_accepted_then_one_failed_partial(self):
        self.assertEqual(len(inherit._campaign(self.binding, self.prior, self.campaign)), 14)

    def test_missing_extra_reordered_or_unaccepted_history_rejected(self):
        changes = [lambda c: c["children"].pop(0),
                   lambda c: c["children"].reverse(),
                   lambda c: c["children"][1].update(status="dispatching"),
                   lambda c: c["children"][1].update(exit_code=1),
                   lambda c: c["children"][1].update(active_processes=[proc(2)]),
                   lambda c: c["scores"].pop("ifeval/base"),
                   lambda c: c["scores"].update({"ifeval/ppo44": {}}),
                   lambda c: c["children"][-1].update(status="accepted", exit_code=0),
                   lambda c: c["children"][-1].update(result={"path": "partial"}),
                   lambda c: c.update(deadline_epoch=inherit.DEADLINE + 1)]
        for change in changes:
            with self.subTest(change=change):
                candidate = copy.deepcopy(self.campaign)
                change(candidate)
                with self.assertRaises(ValueError):
                    inherit._campaign(self.binding, self.prior, candidate)

    def test_partial_manifest_cannot_be_promoted(self):
        p = Path(self.prior["output"]) / "ifeval-ppo44/manifest.json"
        p.parent.mkdir(parents=True)
        p.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "PPO44 partial"):
            inherit._campaign(self.binding, self.prior, self.campaign)

    def test_ticks_exact_and_pid_reuse_not_old_process(self):
        original = proc(10)
        self.assertEqual(inherit._identity(original), inherit._identity({**original,
            "created_utc_ticks": 639288380784723310}))
        for bad in (639288380784723300, 639288380784723310.0, "639288380784723314"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                inherit._identity({**original, "created_utc_ticks": bad})
        with self.assertRaises(ValueError):
            inherit._exited([original], SimpleNamespace(processes=lambda: [{**original,
                "created_utc_ticks": 639288380784723310}]))
        reused = {**original, "created_at_utc": "2026-10-29T03:34:38.4723310Z",
                  "created_utc_ticks": 639288416784723310, "command_line": "unrelated"}
        self.assertEqual(inherit._exited([original], SimpleNamespace(processes=lambda: [reused])), 1)

    def test_source_preservation_cannot_be_weakened(self):
        prior = {inherit.previous.V6_EXECUTION: "a" * 64,
                 **{f"old{i}.py": "b" * 64 for i in range(153)}}
        current = {**prior, inherit.V7_EXECUTION: "c" * 64}
        with patch.object(inherit, "verified_bytes", return_value=b"# fixture") as read:
            inherit._sources(prior, current)
            self.assertEqual(read.call_count, 155)
            for change in (lambda d: d.pop("old1.py"), lambda d: d.update({"old1.py": "d" * 64}),
                           lambda d: d.pop(inherit.V7_EXECUTION)):
                candidate = dict(current)
                change(candidate)
                with self.assertRaises(ValueError):
                    inherit._sources(prior, candidate)

    def test_original_budget_is_required_before_expensive_proofs(self):
        prior = {key: "fixture" for key in ("kind", "schema_version", "authorization", "protocol", "roster",
            "decoder", "analysis_wall_seconds", "poll_seconds", "model_python_sha256", "scoring_python_sha256")}
        prior.update(started_at_epoch=inherit.STARTED, deadline_epoch=inherit.DEADLINE,
                     wall_seconds=inherit.WALL_SECONDS)
        for field, value in (("deadline_epoch", inherit.DEADLINE + 1), ("wall_seconds", inherit.WALL_SECONDS + 1),
                             ("roster", "replacement"), ("protocol", "replacement")):
            with self.subTest(field=field), self.assertRaises(ValueError):
                inherit._scientific_identity(prior, {**prior, field: value})


class ClosureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prior, self.campaign, self.binding = campaign_fixture(self.root)
        health = {f"field{i}": i for i in range(27)}
        health["busy"] = False
        preserved = [proc(100 + i) for i in range(5)]
        profile = {"services": [{"name": "evidoc", "json_equals": {k: v for k, v in health.items() if k != "busy"}}],
            "background_processes": preserved, "authorized_pause": {"authorization": {"sha256": "a" * 64}}}
        failed = self.campaign["children"][-1]
        failed_request_pin = pin(self.root, "runs/old/ifeval-ppo44.request.json")
        failed_request = {"model_id": "ppo44"}
        failed.update(request=failed_request, request_path=failed_request_pin["path"], request_sha256=failed_request_pin["sha256"])
        error = "Service-preserving evaluation guard: Unknown Python process outside exact frozen identities: 58104"
        partial = {"status": "failed", "completed_records": 303, "inference_complete": False,
            "request": failed_request, "error": error, "deadline_epoch": inherit.DEADLINE,
            "pid": failed["known_processes"][0]["pid"]}
        self.closure = {"kind": "alignlab-independent-quality-evaluation-v6-failure-closure",
            "status": "failed_closed_services_restored", "deadline_epoch": inherit.DEADLINE,
            "fully_accepted_policy_inference_records": 3787, "failed_partial_records": 303,
            "failed_partial_accepted": False, "policy_scores_accepted": 7,
            "primary_evaluation_complete": False, "research_complete": False,
            "restore_pending": False, "all_28_fields_match": True, "all_owned_exact_identities_exited": True,
            "guardian_status": "complete", "awake_status": "released", "preserved_processes_match": True,
            "unknown_python_at_closure": [], "same_plan_output_relaunch_forbidden": True,
            "plan": self.binding["prior_plan"], "campaign": self.binding["campaign"],
            "failure": {"stage": "ifeval-ppo44", "error_type": "RuntimeError", "exit_code": 1,
                "completed_records": 303, "unknown_process_origin_confirmed": False, "error": error},
            "failed_progress": pin(self.root, "runs/old/ifeval-ppo44/progress.json"),
            "restoration": pin(self.root, "restored.json"), "resource": pin(self.root, "resource.json"),
            "monitor": pin(self.root, "monitor.json"), "restored_process": proc(200),
            "verified_scorepins": [{"name": slot, "pin": p, "check": "verified", "official": True,
                "denominators": {"prompts": 541, "instructions": 834}} for slot, p in self.campaign["scores"].items()]}
        self.restoration = {"kind": "alignlab-evidoc-restoration-complete-v1", "status": "restored_and_verified",
            "restoration_complete": True, "restore_pending": False, "all_28_fields_match": True,
            "deadline_epoch": inherit.DEADLINE, "health": health, "pause_authorization_sha256": "a" * 64,
            "processes": [proc(200)], "preserved_processes": preserved}
        monitor = {"plan": self.binding["prior_plan"],
            "guardian": {"status": "complete", "deadline_epoch": inherit.DEADLINE,
                "plan": self.binding["prior_plan"], "guardian": proc(201)},
            "awake": {"status": "released", "deadline_epoch": inherit.DEADLINE, "helper_pid": 202,
                "helper_parent_pid": 1, "helper_created_utc": proc(202)["created_at_utc"],
                "helper_command": proc(202)["command_line"]}}
        self.resource = {"all_preserved_identity_matches": True, "restored_identity_matches": True,
            "port_8083_owner_matches": True, "health_fields_verified": 28, "all_28_fields_match": True,
            "unmatched_python_processes": [], "all_owned_exact_identities_exited": True,
            "restoration": self.closure["restoration"], "health": health, "restored_process": proc(200),
            "preserved_processes": preserved, "owned_processes": [{"expected": p, "exact_identity_present": False}
                for p in self.campaign["supervisor_processes"] + [c["known_processes"][0] for c in self.campaign["children"]]]}
        docs = {self.prior["resource_profile"]["path"]: profile, self.closure["restoration"]["path"]: self.restoration,
            self.closure["resource"]["path"]: self.resource, self.closure["monitor"]["path"]: monitor,
            self.closure["failed_progress"]["path"]: partial, failed_request_pin["path"]: failed_request}
        p = patch.object(inherit, "_read", side_effect=lambda p: docs[p["path"]])
        p.start()
        self.addCleanup(p.stop)

    def test_closed_history_includes_all_processes_and_helpers(self):
        self.assertEqual(len(inherit._closed_v6(self.binding, self.prior, self.campaign, self.closure)), 18)

    def test_incomplete_health_pending_restore_or_wrong_process_rejected(self):
        changes = [(self.closure, "restore_pending", True), (self.closure, "failed_partial_accepted", True),
            (self.restoration, "all_28_fields_match", False), (self.restoration, "processes", [proc(999)]),
            (self.resource, "port_8083_owner_matches", False), (self.resource, "health_fields_verified", 27),
            (self.resource, "unmatched_python_processes", [proc(58104)]),
            (self.resource, "owned_processes", []), (self.restoration, "preserved_processes", [])]
        for obj, key, bad in changes:
            saved = obj[key]
            obj[key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                inherit._closed_v6(self.binding, self.prior, self.campaign, self.closure)
            obj[key] = saved


class SlotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prior, self.campaign, common = campaign_fixture(self.root)
        self.prior.update(protocol=pin(self.root, "protocol.json"), roster=pin(self.root, "roster.json"),
                          decoder={"frozen": "decoder"}, source_sha256={"old.py": "a" * 64})
        self.request = {"task": "ifeval", "model_id": "sft", "output": str(self.root / "runs/old/ifeval-sft"),
            "deadline_epoch": inherit.DEADLINE, "source_sha256": self.prior["source_sha256"],
            "resource_profile": self.prior["resource_profile"], "capacity_manifest": self.campaign["capacity_manifest"],
            **{key + suffix: self.prior[key][field] for key in ("protocol", "roster")
                for suffix, field in (("_path", "path"), ("_sha256", "sha256"))}}
        self.binding = {**common, "request": pin(self.root, "runs/old/ifeval-sft.request.json"),
            "manifest": pin(self.root, "runs/old/ifeval-sft/manifest.json"),
            "score_request": pin(self.root, "runs/old/ifeval-sft-score.request.json"),
            "score": pin(self.root, "runs/old/ifeval-sft-score/scores.json")}
        self.children = {c["name"]: c for c in self.campaign["children"]}
        child = self.children["ifeval-sft"]
        identities, snapshot, supervisor_snapshot, closed = process_snapshot_fixture(self.binding["request"])
        child.update(identities)
        self.campaign.update(supervisor_snapshot)
        child.update(request=self.request, request_path=self.binding["request"]["path"],
            request_sha256=self.binding["request"]["sha256"], result=self.binding["manifest"], output=self.request["output"])
        self.manifest = {"request": self.request, "counts": dict(inherit.COUNTS), **snapshot}
        self.score_request = {"schema_version": 1, "kind": "alignlab-official-cpu-scoring-request-v1",
            "inference_manifest": self.binding["manifest"]["path"],
            "inference_manifest_sha256": self.binding["manifest"]["sha256"],
            "source_sha256": self.prior["source_sha256"], "decoder": self.prior["decoder"], "deadline_epoch": inherit.DEADLINE}
        self.children["ifeval-sft-score"].update(request=self.score_request,
            request_path=self.binding["score_request"]["path"], request_sha256=self.binding["score_request"]["sha256"],
            result=self.binding["score"])
        self.campaign["scores"]["ifeval/sft"] = self.binding["score"]
        self.plan = {"inherited_inference": {s: self.binding for s in inherit.SLOTS},
            "inherited_scores": {s: self.binding for s in inherit.SLOTS}, "source_sha256": {"new.py": "b" * 64}}
        docs = {"prior_plan": self.prior, "campaign": self.campaign, "closure": {}, "request": self.request,
                "manifest": self.manifest, "score_request": self.score_request, "score": {"fixture": "score"}}
        self.docs = {self.binding[k]["path"]: v for k, v in docs.items()}
        def context(binding, plan, **kwargs):
            return self.prior, self.campaign, {}, self.children, closed
        for target, replacement in (("_read", lambda p: copy.deepcopy(self.docs[p["path"]])),
                                    ("verify_recovery_context", context), ("_sources", lambda *a: None)):
            p = patch.object(inherit, target, replacement)
            p.start()
            self.addCleanup(p.stop)
        p = patch("alignment_lab.evaluation_supervisor_v6._verify_score", return_value=self.binding["score"])
        self.score_verifier = p.start()
        self.addCleanup(p.stop)
        self.system = SimpleNamespace(processes=lambda: [])

    def test_full_slot_retains_original_bindings_and_requires_replay(self):
        result = inherit.verify_inherited_inference(self.binding, self.plan, system=self.system)
        self.assertEqual(result["manifest_binding"], self.binding["manifest"])
        self.assertEqual(result["score_binding"], self.binding["score"])
        self.assertTrue(result["verification"]["requires_official_analysis_replay"])
        self.assertFalse(result["verification"]["partial_ppo44_inherited"])
        self.score_verifier.assert_called_once_with(Path(self.binding["score"]["path"]).parent,
                                                   self.score_request, self.prior, self.request)

    def test_score_api_returns_scoring_request(self):
        result = inherit.verify_inherited_score(self.binding, self.plan, system=self.system)
        self.assertEqual(result["request"], self.score_request)

    def test_score_and_request_substitution_rejected(self):
        changes = [lambda: self.children["ifeval-sft-score"].update(exit_code=1),
                   lambda: self.children["ifeval-sft"].update(status="dispatching"),
                   lambda: self.campaign["scores"].update({"ifeval/sft": {}}),
                   lambda: self.manifest.update(counts={"records": 303}),
                   lambda: self.request.update(model_id="ppo44"),
                   lambda: self.score_request.update(deadline_epoch=inherit.DEADLINE + 1),
                   lambda: self.plan["inherited_scores"].pop("ifeval/base")]
        baseline = copy.deepcopy((self.children, self.campaign, self.manifest, self.request,
                                  self.score_request, self.plan))
        for change in changes:
            with self.subTest(change=change):
                change()
                with self.assertRaises(ValueError):
                    inherit.verify_inherited_inference(self.binding, self.plan, system=self.system)
                # Preserve fixture aliases while restoring each mutable object.
                for obj, saved in zip((self.children, self.campaign, self.manifest, self.request,
                                      self.score_request, self.plan), baseline):
                    obj.clear()
                    obj.update(copy.deepcopy(saved))

    def test_full_original_verifier_failure_is_not_ignored(self):
        self.score_verifier.side_effect = ValueError("corrupt original output or decoder receipt")
        with self.assertRaisesRegex(ValueError, "corrupt original"):
            inherit.verify_inherited_inference(self.binding, self.plan, system=self.system)


class BaseAndRoutingTests(unittest.TestCase):
    def test_base_requires_complete_original_v5_to_v6_chain(self):
        binding = {"request": {"original": "request"}, "manifest": {"original": "manifest"}}
        request = {"task": "ifeval", "model_id": "base"}
        old_binding = {"request": binding["request"], "manifest": binding["manifest"]}
        prior = {"inherited_inference": {"ifeval/base": old_binding}}
        old = {"request": request, "manifest_binding": binding["manifest"], "accepted_record": {"accepted": True}}
        campaign = {"inherited_inference": {"ifeval/base": old}}
        with patch.object(inherit.previous, "verify_inherited_inference", return_value=old) as verify:
            self.assertEqual(inherit._base_chain(binding, prior, campaign, request), old["accepted_record"])
            verify.assert_called_once_with(old_binding, prior, system=None)
            for bad in ({**binding, "manifest": {"replacement": True}},
                        {**binding, "request": {"replacement": True}}):
                with self.assertRaises(ValueError):
                    inherit._base_chain(bad, prior, campaign, request)
            with self.assertRaises(ValueError):
                inherit._base_chain(binding, prior, {"inherited_inference": {"ifeval/base": {}}}, request)

    def test_v7_routing_wins_when_old_sources_are_preserved(self):
        request = {"source_sha256": {inherit.V7_EXECUTION: "a" * 64, inherit.previous.V6_EXECUTION: "b" * 64}}
        with patch.object(inherit, "_read", return_value={"request": request}), \
             patch.object(inherit, "verified_bytes", return_value=b"# fixture"), \
             patch("alignment_lab.evaluation_execution_v7.verify_evaluation_result", return_value={"version": 7}) as new, \
             patch.object(inherit.previous, "verify_versioned_evaluation_result") as old:
            result = inherit.verify_versioned_evaluation_result(Path.cwd(), expected_manifest_sha256="c" * 64,
                                                                 expected_request=request)
            self.assertEqual(result, {"version": 7})
            self.assertEqual(new.call_count, 1)
            old.assert_not_called()
            with self.assertRaises(ValueError):
                inherit.verify_versioned_evaluation_result(Path.cwd(), expected_manifest_sha256="c" * 64,
                                                            expected_request={"changed": True})

    def test_original_versions_use_original_verifier(self):
        request = {"source_sha256": {inherit.previous.V6_EXECUTION: "a" * 64}}
        with patch.object(inherit, "_read", return_value={"request": request}), \
             patch.object(inherit.previous, "verify_versioned_evaluation_result", return_value={"version": 6}) as old:
            self.assertEqual(inherit.verify_versioned_evaluation_result(Path.cwd(),
                expected_manifest_sha256="c" * 64), {"version": 6})
            self.assertEqual(old.call_count, 1)


if __name__ == "__main__":
    unittest.main()
