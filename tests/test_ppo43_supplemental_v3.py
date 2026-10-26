"""Adversarial evidence checks for the case-bound PPO43 supplemental verifier."""
import copy
from pathlib import Path
import random
import tempfile
import unittest

from alignment_lab import ppo43_supplemental_v3 as verifier


def fixture():
    work = {"seed": 43, "rollout_batch": 2, "optimizer_batch": 1, "epochs": 2,
            "max_responses": 4, "max_updates": 8, "max_new_tokens": 256}
    records = [{"record_id": "real-" + str(i)} for i in range(4)]
    order = list(range(4))
    random.Random(43).shuffle(order)
    baseline = {"schema": 1, "order": order, "config_sha256": "config", "data_sha256": "data"}
    events, rollouts, objectives = [], [], []
    update, total = 0, 0
    for index in range(2):
        lengths = [3 + index, 6 + index]
        total += sum(lengths)
        ids = [records[i]["record_id"] for i in order[index * 2:index * 2 + 2]]
        cursor = {**baseline, "offset": (index + 1) * 2, "optimizer_updates": update,
                  "response_tokens": total, "active_rollout": True, "completed_rollouts": index}
        rollouts.append({"rollout_index": index, "record_ids": ids,
            "generated_at_optimizer_updates": update, "cursor_before_updates": copy.deepcopy(cursor),
            "responses": [{"action_ids": list(range(n))} for n in lengths]})
        for epoch in range(2):
            indices = list(range(2))
            random.Random(43 + index * 2 + epoch).shuffle(indices)
            for offset, selected in enumerate(indices):
                info = {"rollout_index": index, "epoch": epoch, "rollout_offset": offset,
                        "target_update": update + 1, "response_indices": [selected], "record_ids": [ids[selected]]}
                for after in (False, True):
                    view = {**cursor, "optimizer_updates": update + int(after)}
                    if after and epoch == 1 and offset == 1:
                        view.update(active_rollout=False, completed_rollouts=index + 1)
                    event = {**info, "event": "optimizer_end" if after else "optimizer_start",
                             "cursor": {k: v for k, v in view.items() if k != "order"},
                             "cursor_sha256": verifier.digest(view), "order_sha256": verifier.digest(order)}
                    if after:
                        event.update(update=update + 1, action_tokens=lengths[selected], objective=-0.5, gradient_norm=2.0)
                    else:
                        event["denominator"] = lengths[selected]
                    events.append(event)
                update += 1
                objectives.append({"update": update, "responses": 1, "action_tokens": lengths[selected],
                                   "objective": -0.5, "gradient_norm": 2.0})
                cursor["optimizer_updates"] = update
    final = {**baseline, "offset": 4, "optimizer_updates": 8, "response_tokens": total,
             "active_rollout": False, "completed_rollouts": 2}
    return events, rollouts, records, work, final, objectives


class SupplementalTests(unittest.TestCase):
    def test_complete_independent_reconstruction_preserves_input(self):
        data = fixture()
        before = copy.deepcopy(data)
        result = verifier.reconstruct_objectives(*data)
        self.assertEqual(data, before)
        self.assertEqual(result["optimizer_updates"], 8)
        self.assertEqual(result["response_optimizer_uses"], 8)
        self.assertEqual(result["optimizer_action_tokens"], 40)
        self.assertTrue(result["other_four_objective_fields_exact"])

    def test_missing_duplicate_or_unpaired_optimizer_event_rejected(self):
        for mutate in (lambda e: e.pop(1), lambda e: e.insert(1, copy.deepcopy(e[1])),
                       lambda e: e.__setitem__(slice(0, 2), [e[1], e[0]])):
            data = fixture()
            mutate(data[0])
            with self.assertRaises(ValueError):
                verifier.reconstruct_objectives(*data)

    def test_tampered_indices_ids_epoch_or_offset_rejected_even_if_end_matches_start(self):
        for key, value in (("response_indices", [99]), ("record_ids", ["wrong"]),
                           ("epoch", 9), ("rollout_offset", 1), ("target_update", 2)):
            data = fixture()
            data[0][0][key] = value
            data[0][1][key] = value
            with self.assertRaises(ValueError):
                verifier.reconstruct_objectives(*data)

    def test_coordinated_returned_and_live_token_tampering_cannot_replace_trajectory_tokens(self):
        data = fixture()
        data[0][0]["denominator"] = 123
        data[0][1]["action_tokens"] = 123
        data[5][0]["action_tokens"] = 123
        with self.assertRaisesRegex(ValueError, "denominator"):
            verifier.reconstruct_objectives(*data)

    def test_returned_response_count_never_used_as_default(self):
        data = fixture()
        data[5][0]["responses"] = 17
        with self.assertRaises(ValueError):
            verifier.reconstruct_objectives(*data)

    def test_exception_does_not_accept_an_already_present_response_field(self):
        data = fixture()
        data[0][1]["responses"] = 1
        with self.assertRaisesRegex(ValueError, "historical omitted"):
            verifier.reconstruct_objectives(*data)

    def test_numerical_mismatch_nonfinite_or_fake_update_rejected(self):
        for key, value in (("objective", -0.4), ("gradient_norm", float("nan")), ("update", 2)):
            data = fixture()
            data[0][1][key] = value
            with self.assertRaises(ValueError):
                verifier.reconstruct_objectives(*data)

    def test_cursor_value_or_hash_tampering_rejected(self):
        for mutate in (lambda e: e["cursor"].update(offset=3),
                       lambda e: e.update(cursor_sha256="wrong"),
                       lambda e: e.update(order_sha256="wrong")):
            data = fixture()
            mutate(data[0][1])
            with self.assertRaisesRegex(ValueError, "cursor"):
                verifier.reconstruct_objectives(*data)

    def test_omitted_rollout_or_repeated_record_or_wrong_seed_rejected(self):
        for choice in range(3):
            data = fixture()
            if choice == 0:
                data[1].pop()
            elif choice == 1:
                data[1][1]["record_ids"] = data[1][0]["record_ids"]
            else:
                data[3]["seed"] = 44
            with self.assertRaises(ValueError):
                verifier.reconstruct_objectives(*data)

    def test_final_denominator_tampering_rejected(self):
        data = fixture()
        data[4]["response_tokens"] += 1
        with self.assertRaisesRegex(ValueError, "coverage"):
            verifier.reconstruct_objectives(*data)

    def test_binding_rejects_unrelated_campaign_and_changed_payload(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "proof.json"
            path.write_text("original", encoding="utf-8")
            pin = verifier._pin(path)
            self.assertEqual(verifier._check_pin(pin), path.resolve())
            path.write_text("changed!", encoding="utf-8")
            with self.assertRaises(ValueError):
                verifier._check_pin(pin)
            with self.assertRaises(ValueError):
                verifier.build_binding(path)

    def test_local_function_clone_keeps_postchecks_and_does_not_patch_module(self):
        namespace = {"callback": lambda: "old", "postcheck": lambda: (_ for _ in ()).throw(ValueError("postcheck"))}
        exec("def original():\n    callback()\n    postcheck()\n", namespace)
        original = namespace["original"]
        calls = []
        cloned = verifier._clone(original, callback=lambda: calls.append("new"))
        with self.assertRaisesRegex(ValueError, "postcheck"):
            cloned()
        self.assertEqual(calls, ["new"])
        self.assertEqual(namespace["callback"](), "old")
        self.assertIs(cloned.__code__, original.__code__)


if __name__ == "__main__":
    unittest.main()
