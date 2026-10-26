"""Small mocked HTTP fixtures only; no real weights, model import, GPU or network."""
import copy
import hashlib
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alignment_lab import model_snapshot as model


class Response(io.BytesIO):
    def __init__(self, body, status=200, headers=None):
        super().__init__(body)
        self.status = status
        self.headers = headers or {"Content-Length": str(len(body))}


def safetensor(name):
    header = json.dumps({name: {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
                        separators=(",", ":")).encode()
    header += b" " * (-len(header) % 8)
    return struct.pack("<Q", len(header)) + header + struct.pack("<f", 0.25)


class ModelSnapshotTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "new-snapshot"
        self.tokenizer = self.root / "existing-tokenizer"
        self.tokenizer.mkdir()
        self.contents = {name: f"official fixture {name}\n".encode() for name in model.PINNED_FILES}
        self.contents["config.json"] = b'{"model_type":"qwen3"}'
        self.contents[model.INDEX_NAME] = model._json_bytes({
            "metadata": {"total_size": 8},
            "weight_map": {"layer.0": model.WEIGHT_FILES[0], "layer.1": model.WEIGHT_FILES[1]},
        })
        for i, name in enumerate(model.WEIGHT_FILES):
            self.contents[name] = safetensor(f"layer.{i}")
        self.pins = {}
        for name, payload in self.contents.items():
            sha = hashlib.sha256(payload).hexdigest()
            lfs = sha if name.endswith(".safetensors") or name == "tokenizer.json" else None
            blob = model._blob(model._pointer(sha, len(payload))) if lfs else model._blob(payload)
            self.pins[name] = (len(payload), blob, lfs)
        self.metadata = self.make_metadata()
        for name in model.TOKENIZER_FILES:
            (self.tokenizer / name).write_bytes(self.contents[name])
        self.calls = []
        self.patcher = patch.multiple(model, PINNED_FILES=self.pins,
                                     TOKENIZER_DIRECTORY=self.tokenizer, RANGE_BYTES=64)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def make_metadata(self):
        siblings = []
        for name, (size, blob, lfs) in self.pins.items():
            item = {"rfilename": name, "size": size, "blobId": blob}
            if lfs:
                item["lfs"] = {"sha256": lfs, "size": size,
                               "pointerSize": len(model._pointer(lfs, size))}
            siblings.append(item)
        return {"id": model.REPO_ID, "sha": model.REVISION, "siblings": siblings}

    def open(self, request, timeout):
        self.calls.append((request.full_url, request.get_header("Range"), timeout))
        if request.full_url == model.API_URL:
            return Response(model._json_bytes(self.metadata))
        name = request.full_url.rsplit("/", 1)[-1]
        if request.full_url != f"https://huggingface.co/{model.REPO_ID}/resolve/{model.REVISION}/{name}":
            raise AssertionError("Unexpected request URL")
        self.assertFalse((self.output / model.MANIFEST_NAME).exists())
        body = self.contents[name]
        byte_range = request.get_header("Range")
        if byte_range:
            start, end = map(int, byte_range.removeprefix("bytes=").split("-"))
            return Response(body[start:end + 1], 206, {
                "Content-Length": str(end - start + 1),
                "Content-Range": f"bytes {start}-{end}/{len(body)}",
            })
        return Response(body)

    def create(self):
        return model.download_snapshot(self.output, opener=self.open)

    def anchor(self):
        return hashlib.sha256((self.output / model.MANIFEST_NAME).read_bytes()).hexdigest()

    def edit_manifest(self, change):
        path = self.output / model.MANIFEST_NAME
        value = json.loads(path.read_bytes())
        change(value)
        path.write_bytes(model._json_bytes(value))
        return self.anchor()

    def test_default_plan_contacts_metadata_only_and_creates_nothing(self):
        before = sorted(p.name for p in self.root.iterdir())
        plan = model.plan_snapshot(self.output, opener=self.open)
        self.assertEqual(plan["mode"], "plan")
        self.assertFalse(plan["large_weights_downloaded"])
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0][0], model.API_URL)
        self.assertEqual(before, sorted(p.name for p in self.root.iterdir()))
        self.assertEqual(plan["weights_bytes"], sum(len(self.contents[n]) for n in model.WEIGHT_FILES))

    def test_complete_snapshot_verified_read_only_and_anchor_required(self):
        result = self.create()
        before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.output.iterdir()}
        self.assertEqual(model.verify_snapshot(self.output, result["manifest_sha256"]), result["manifest"])
        self.assertEqual(before, {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.output.iterdir()})
        with self.assertRaisesRegex(ValueError, "manifest SHA256"):
            model.verify_snapshot(self.output, "0" * 64)
        with self.assertRaisesRegex(ValueError, "mandatory"):
            model.verify_snapshot(self.output, result["manifest_sha256"], verify_hashes=False)
        self.assertFalse((self.root / f".{self.output.name}.download.lock").exists())
        self.assertTrue(any(r[1] for r in self.calls))

    def test_no_overwrite_complete_or_ambiguous_partial_directory(self):
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            self.create()
        self.assertEqual(self.calls, [])
        self.output.rmdir()
        self.create()
        snapshot = {p.name: p.read_bytes() for p in self.output.iterdir()}
        with self.assertRaises(FileExistsError):
            self.create()
        self.assertEqual(snapshot, {p.name: p.read_bytes() for p in self.output.iterdir()})

    def test_stale_lock_is_never_stolen(self):
        lock = self.root / f".{self.output.name}.download.lock"
        lock.write_bytes(b"original lock")
        with self.assertRaises(FileExistsError):
            self.create()
        self.assertFalse(self.output.exists())
        self.assertEqual(lock.read_bytes(), b"original lock")
        self.assertEqual(len(self.calls), 1)

    def test_tamper_weight_and_reanchor_manifest_still_fails_upstream_hash(self):
        self.create()
        name = model.WEIGHT_FILES[0]
        path = self.output / name
        altered = path.read_bytes()[:-1] + b"X"
        path.write_bytes(altered)
        anchor = self.edit_manifest(lambda m: m["files"][name].update(sha256=hashlib.sha256(altered).hexdigest()))
        with self.assertRaisesRegex(ValueError, "LFS/Git"):
            model.verify_snapshot(self.output, anchor)

    def test_non_lfs_git_blob_is_checked_even_if_local_sha_reanchored(self):
        self.create()
        name = "README.md"
        altered = b"X" * len(self.contents[name])
        (self.output / name).write_bytes(altered)
        anchor = self.edit_manifest(lambda m: m["files"][name].update(sha256=hashlib.sha256(altered).hexdigest()))
        with self.assertRaisesRegex(ValueError, "Git blob"):
            model.verify_snapshot(self.output, anchor)

    def test_extra_or_missing_inventory_and_wrong_revision_rejected(self):
        result = self.create()
        extra = self.output / "pytorch_model.bin"
        extra.write_bytes(b"not allowed")
        with self.assertRaisesRegex(ValueError, "untracked"):
            model.verify_snapshot(self.output, result["manifest_sha256"])
        extra.unlink()
        anchor = self.edit_manifest(lambda m: m.update(revision="0" * 40))
        with self.assertRaisesRegex(ValueError, "identity"):
            model.verify_snapshot(self.output, anchor)
        (self.output / model.WEIGHT_FILES[1]).unlink()
        with self.assertRaisesRegex(ValueError, "partial"):
            model.verify_snapshot(self.output, anchor)

    def test_actual_config_and_index_structure_checked(self):
        result = self.create()
        self.assertEqual(model.verify_snapshot(self.output, result["manifest_sha256"])["model_type"], "qwen3")
        (self.output / "config.json").write_bytes(b'{"model_type":"gpt2"}')
        with self.assertRaisesRegex(ValueError, "model_type"):
            model._model_structure(self.output)
        (self.output / "config.json").write_bytes(self.contents["config.json"])
        (self.output / model.INDEX_NAME).write_bytes(b'{"weight_map":{"a":"../evil.safetensors"}}')
        with self.assertRaisesRegex(ValueError, "index"):
            model._model_structure(self.output)

    def test_upstream_wrong_sha_size_lfs_pointer_or_inventory_rejected(self):
        mutations = (
            lambda d: d.update(sha="0" * 40),
            lambda d: d["siblings"][0].update(size=True),
            lambda d: d["siblings"].append(copy.deepcopy(d["siblings"][0])),
            lambda d: next(x for x in d["siblings"] if "lfs" in x)["lfs"].update(sha256="0" * 64),
        )
        original = copy.deepcopy(self.metadata)
        for mutate in mutations:
            self.metadata = copy.deepcopy(original)
            mutate(self.metadata)
            with self.assertRaises(ValueError):
                model.fetch_metadata(opener=self.open)
        self.assertFalse(self.output.exists())

    def test_failure_preserves_partial_lock_and_evidence_without_manifest(self):
        def broken(request, timeout):
            result = self.open(request, timeout)
            if request.get_header("Range"):
                result.status = 200
            return result
        with self.assertRaisesRegex(ValueError, "byte range"):
            model.download_snapshot(self.output, opener=broken)
        self.assertFalse((self.output / model.MANIFEST_NAME).exists())
        self.assertTrue(list(self.output.glob("*.part")))
        self.assertTrue((self.root / f".{self.output.name}.download.lock").exists())
        log = next(self.root.glob("*.jsonl"))
        last = json.loads(log.read_bytes().splitlines()[-1])
        self.assertEqual(last["status"], "failed")
        self.assertTrue(last["lock_retained"])
        with self.assertRaises(FileExistsError):
            self.create()

    def test_truncated_body_rejected_without_final_manifest(self):
        def truncated(request, timeout):
            response = self.open(request, timeout)
            if request.full_url != model.API_URL:
                body = response.getvalue()[:-1]
                return Response(body, response.status, response.headers)
            return response
        with self.assertRaisesRegex(ValueError, "Truncated"):
            model.download_snapshot(self.output, opener=truncated)
        self.assertFalse((self.output / model.MANIFEST_NAME).exists())

    def test_insufficient_space_stops_before_destination_or_lock(self):
        with patch.object(model.shutil, "disk_usage", return_value=SimpleNamespace(free=0)):
            with self.assertRaisesRegex(OSError, "space"):
                self.create()
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["existing-tokenizer"])
        self.assertEqual(len(self.calls), 1)

    def test_existing_tokenizer_conflict_is_detected_before_weights_and_on_verify(self):
        result = self.create()
        existing = self.tokenizer / "config.json"
        existing.write_bytes(b'{"model_type":"qwen4"}')
        with self.assertRaisesRegex(ValueError, "Git blob"):
            model.verify_snapshot(self.output, result["manifest_sha256"])
        self.calls.clear()
        with self.assertRaisesRegex(ValueError, "Git blob"):
            model.plan_snapshot(self.root / "another", opener=self.open)
        self.assertEqual(len(self.calls), 1)

    def test_metadata_only_and_tensor_or_pickle_modules_are_not_imported(self):
        # This stdlib module must never initialize a framework merely to plan.
        self.assertNotIn("torch", model.__dict__)
        self.assertNotIn("transformers", model.__dict__)
        with self.assertRaisesRegex(ValueError, "timeout"):
            model.fetch_metadata(timeout=0, opener=self.open)


if __name__ == "__main__":
    unittest.main()
