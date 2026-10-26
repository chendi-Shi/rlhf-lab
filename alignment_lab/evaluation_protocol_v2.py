"""CPU-only, immutable inputs and identities for the independent final evaluation.

No tokenizer/model is instantiated. No length filter is available. Loading a
protocol is not evidence that its model artifacts completed formal training;
the caller must also enforce the campaign's artifact lineage and model roster.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping

ASSET_MANIFEST_SHA256 = "6950abb56fc6ae6044b0283c53a7b5fc3e64d57a43ea8ba863621e2274505720"
ASSET_REPORT_SHA256 = "15a45c0c93d95a0da82ff87dd1d0481261ce4b6a367994f0a3d5018e1d6cba9c"
HOLDOUT_MANIFEST_SHA256 = "84d059ba111c37d8f89d70cec6db6bd6c4fd7aef8d4cf933343d67556a204e0b"
IFEVAL_INPUT = "ifeval/instruction_following_eval/data/input_data.jsonl"
RM_INPUT = "rm_bench/data/total_dataset.json"
IFEVAL_PACKAGES = ("absl-py", "langdetect", "nltk", "immutabledict")
DOMAINS = ("chat", "math", "code", "safety")
SERIALIZATION = "qwen3-nonthinking-nfc-completion-v3"


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def parse(payload: bytes):
    def reject(value):
        raise ValueError(f"Nonfinite JSON: {value}")
    return json.loads(payload, object_pairs_hook=_object, parse_constant=reject)


def _sha(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value):
        raise ValueError("Expected lowercase SHA256")
    return value


def verified_bytes(path: Path, expected_sha256: str) -> bytes:
    """Hash the bytes actually returned, refusing links and changing files."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected regular file: {path}")
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        payload = stream.read()
        after = os.fstat(stream.fileno())
    signature = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
    if signature(before) != signature(after) or signature(after) != signature(path.stat()):
        raise ValueError(f"File changed while reading: {path}")
    if hashlib.sha256(payload).hexdigest() != _sha(expected_sha256):
        raise ValueError(f"SHA256 mismatch: {path}")
    return payload


def _relative(root: Path, name: str) -> Path:
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name
            or any(part in ("", ".", "..") for part in name.split("/"))):
        raise ValueError("Unsafe inventory path")
    path = root.joinpath(*name.split("/"))
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Inventory path escapes asset directory")
    if any(p.is_symlink() for p in (path, *path.parents) if p != root.parent):
        raise ValueError("Symlink in inventory path")
    return path


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _index(rows, field):
    result = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each record must be an object")
        key = row.get(field)
        if type(key) not in (str, int) or key == "" or key in result:
            raise ValueError(f"Missing, invalid or duplicate {field}: {key!r}")
        result[key] = row
    return result


@dataclass(frozen=True)
class EvaluationDataset:
    name: str
    id_field: str
    _rows: bytes
    _identity: bytes
    _lengths: bytes = b"{}"

    @property
    def rows(self) -> tuple[dict, ...]:
        return tuple(parse(self._rows))

    @property
    def identity(self) -> dict:
        return parse(self._identity)

    @property
    def lengths(self) -> dict:
        return parse(self._lengths)

    @property
    def ids(self) -> tuple:
        return tuple(row[self.id_field] for row in self.rows)

    def verify(self):
        rows, identity = self.rows, self.identity
        if (len(_index(rows, self.id_field)) != identity["record_count"]
                or digest(list(rows)) != identity["records_sha256"]
                or digest(list(self.ids)) != identity["ordered_ids_sha256"]):
            raise ValueError("Dataset content differs from its immutable identity")


@dataclass(frozen=True)
class EvaluationAssets:
    manifest_path: str
    manifest_sha256: str
    _manifest: bytes
    ifeval: EvaluationDataset
    rm_bench: EvaluationDataset

    @property
    def manifest(self):
        return parse(self._manifest)

    def read(self, relative_path: str) -> bytes:
        verified_bytes(Path(self.manifest_path), self.manifest_sha256)
        inventory = {r["path"]: r for r in self.manifest["files"]}
        if relative_path not in inventory:
            raise ValueError("File is outside the accepted asset inventory")
        item = inventory[relative_path]
        result = verified_bytes(_relative(Path(self.manifest_path).parent, relative_path), item["sha256"])
        if len(result) != item["bytes"]:
            raise ValueError("Asset size mismatch")
        return result

    def verify(self):
        manifest = parse(verified_bytes(Path(self.manifest_path), self.manifest_sha256))
        if canonical(manifest) != self._manifest:
            raise ValueError("Asset identity changed")
        root = Path(self.manifest_path).parent
        names = [r["path"] for r in manifest["files"]]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate inventory path")
        actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
        if actual != set(names) | {Path(self.manifest_path).name}:
            raise ValueError("Missing or extra asset files")
        for item in manifest["files"]:
            self.read(item["path"])


def load_evaluation_assets(manifest_path, expected_manifest_sha256: str) -> EvaluationAssets:
    """Only the complete, externally pinned 2026-10-20 asset version is accepted."""
    if _sha(expected_manifest_sha256) != ASSET_MANIFEST_SHA256:
        raise ValueError("Unaccepted evaluation asset revision")
    path = Path(manifest_path).resolve()
    manifest = parse(verified_bytes(path, expected_manifest_sha256))
    if manifest.get("kind") != "alignlab-official-evaluation-assets-v1":
        raise ValueError("Unsupported asset manifest")
    empty = EvaluationDataset("", "", b"[]", b"{}")
    assets = EvaluationAssets(str(path), expected_manifest_sha256, canonical(manifest), empty, empty)
    assets.verify()
    # split on physical JSONL newlines, not Unicode U+2028 inside strings.
    ifeval = [parse(line) for line in assets.read(IFEVAL_INPUT).split(b"\n") if line.strip()]
    rm = parse(assets.read(RM_INPUT))
    if (len(ifeval) != 541 or len(_index(ifeval, "key")) != 541
            or len({r["prompt"] for r in ifeval}) != 541
            or sum(len(r["instruction_id_list"]) for r in ifeval) != 834):
        raise ValueError("Full IFEval 541/834 denominator required")
    if any(len(r["kwargs"]) != len(r["instruction_id_list"]) for r in ifeval):
        raise ValueError("IFEval instruction/kwargs mismatch")
    if len(rm) != 1327 or len(_index(rm, "id")) != 1327:
        raise ValueError("Full RM-Bench denominator required")
    if any(len(r["chosen"]) != 3 or len(r["rejected"]) != 3 for r in rm):
        raise ValueError("RM-Bench requires all six style responses")
    audit = parse(assets.read("STRUCTURE_AND_LENGTH_AUDIT.json"))
    if audit["serialization_errors"] or audit["serialization_version"] != SERIALIZATION:
        raise ValueError("Accepted lossless serialization is required")
    if_lengths = {}
    rm_lengths = {}
    for row in audit["records"]:
        if row["dataset"] == "ifeval":
            if_lengths[str(row["key"])] = row["prompt_tokens"]
        else:
            sides = rm_lengths.setdefault(row["id"], {"chosen": [None] * 3, "rejected": [None] * 3})
            sides[row["side"]][row["style"]] = row["total_tokens"]
    for name, rows, lengths in (("ifeval", ifeval, if_lengths), ("rm_bench", rm, rm_lengths)):
        if len(rows) != len(lengths):
            raise ValueError(f"Incomplete accepted length inventory: {name}")
    def dataset(name, rows, field, file, lengths):
        identity = {"dataset": name, "asset_manifest_sha256": expected_manifest_sha256,
                    "input_sha256": hashlib.sha256(assets.read(file)).hexdigest(),
                    "record_count": len(rows), "records_sha256": digest(rows),
                    "ordered_ids_sha256": digest([r[field] for r in rows])}
        return EvaluationDataset(name, field, canonical(rows), canonical(identity), canonical(lengths))
    return EvaluationAssets(str(path), expected_manifest_sha256, canonical(manifest),
                            dataset("ifeval", ifeval, "key", IFEVAL_INPUT, if_lengths),
                            dataset("rm_bench", rm, "id", RM_INPUT, rm_lengths))


def load_final_holdout(audit_dir, expected_manifest_sha256: str) -> EvaluationDataset:
    """Load all official validation rows; train length limits are never applied."""
    if _sha(expected_manifest_sha256) != HOLDOUT_MANIFEST_SHA256:
        raise ValueError("Unaccepted final holdout revision")
    directory = Path(audit_dir).resolve()
    manifest = parse(verified_bytes(directory / "manifest.json", expected_manifest_sha256))
    identity = manifest["identity"]
    if (manifest.get("data_ready") is not True or manifest.get("status") != "audit_complete"
            or manifest.get("readiness_blockers") != []
            or digest(identity) != manifest["identity_sha256"]
            or identity["serialization_version"] != SERIALIZATION):
        raise ValueError("Final holdout audit is not accepted")
    source = identity["inputs"]["validation"]
    verified_bytes(Path(source["path"]), source["sha256"])
    payload = verified_bytes(directory / "final_holdout.jsonl", manifest["artifacts"]["final_holdout.jsonl"])
    rows = [parse(line) for line in payload.split(b"\n") if line.strip()]
    if (len(rows) != 2017
            or {r["source_index"] for r in rows} != set(range(2017))
            or any(r["source_split"] != "validation" for r in rows)):
        raise ValueError("Complete official validation IDs are required")
    ties, non_ties, lengths = [], [], {}
    for row in rows:
        _sha(row["record_id"])
        _integer(row["source_index"], "source_index")
        # Content hashes legitimately repeat in the original 2017 rows. The
        # source position distinguishes observations without dropping any.
        row["evaluation_id"] = f"validation:{row['source_index']}:{row['record_id']}"
        preference = row["overall_preference"]
        if type(preference) is not int or preference not in range(-3, 4):
            raise ValueError("Invalid official preference")
        if not isinstance(row["cluster_id"], str) or not row["cluster_id"]:
            raise ValueError("Missing holdout cluster")
        expected = ((None, None) if preference == 0 else
                    ((row["response1"], row["response2"]) if preference < 0 else
                     (row["response2"], row["response1"])))
        if (row["chosen"], row["rejected"]) != expected:
            raise ValueError("Chosen/rejected differs from official preference direction")
        (ties if preference == 0 else non_ties).append(row["evaluation_id"])
        tokens = row["tokens"]
        for key in ("prompt", "response1", "response2"):
            _integer(tokens[key], key, 1)
        lengths[row["evaluation_id"]] = [tokens["prompt"] + tokens["response1"],
                                         tokens["prompt"] + tokens["response2"]]
    if len(non_ties) != 1920 or len(ties) != 97:
        raise ValueError("Final holdout must retain 1920 non-ties and 97 ties")
    result_identity = {"dataset": "helpsteer3_final_holdout", "manifest_path": str(directory / "manifest.json"),
                       "manifest_sha256": expected_manifest_sha256, "input_sha256": hashlib.sha256(payload).hexdigest(),
                       "raw_validation_sha256": source["sha256"], "record_count": 2017,
                       "non_tie_ids": non_ties, "tie_ids": ties,
                       "id_field": "evaluation_id", "content_record_ids_are_unique": False,
                       "unique_content_record_ids": len({r["record_id"] for r in rows}),
                       "split_counts": {"validation": 2017, "train": 0},
                       "records_sha256": digest(rows),
                       "ordered_ids_sha256": digest([r["evaluation_id"] for r in rows]),
                       "length_filter_applied": False, "serialization_version": SERIALIZATION}
    return EvaluationDataset("helpsteer3_final_holdout", "evaluation_id", canonical(rows),
                             canonical(result_identity), canonical(lengths))


def dependency_versions():
    result = {}
    for package in IFEVAL_PACKAGES:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


@dataclass(frozen=True)
class EvaluationProtocol:
    _payload: bytes

    @property
    def identity(self):
        return parse(self._payload)

    @property
    def sha256(self):
        return hashlib.sha256(self._payload).hexdigest()

    def verify_bindings(self):
        value = self.identity
        for group in ("model_manifests", "source_files", "nltk_resource_files"):
            for record in value[group].values():
                verified_bytes(Path(record["path"]), record["sha256"])
        verified_bytes(Path(value["asset_manifest"]["path"]), value["asset_manifest"]["sha256"])
        verified_bytes(Path(value["holdout_manifest"]["path"]), value["holdout_manifest"]["sha256"])


def build_evaluation_protocol(assets: EvaluationAssets, holdout: EvaluationDataset, *,
                              model_manifests: Mapping[str, tuple[str, str]],
                              scorer_seed: int, bootstrap_seed: int,
                              nltk_resource_files: Mapping[str, tuple[str, str]] | None = None,
                              nltk_data_dir: str | None = None) -> EvaluationProtocol:
    """Bind existing manifest bytes, never future/missing paths or asserted hashes.

    The roster can be expanded only by creating a new protocol identity. A
    campaign must separately require its complete preregistered model roster.
    Missing evaluator dependencies are recorded and later prevent scoring.
    """
    assets.verify()
    for dataset in (assets.ifeval, assets.rm_bench, holdout):
        dataset.verify()
    if holdout.name != "helpsteer3_final_holdout" or holdout.identity["record_count"] != 2017:
        raise ValueError("Complete final holdout is required")
    if not model_manifests:
        raise ValueError("At least one existing model manifest is required")
    def bindings(values):
        records = {}
        for name, (path, expected) in values.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Manifest/resource name required")
            path = Path(path).resolve()
            verified_bytes(path, expected)
            records[name] = {"path": str(path), "sha256": expected}
        return records
    source_files = {}
    for name in ("evaluation_protocol_v2.py", "official_evaluation_v2.py"):
        path = Path(__file__).with_name(name)
        source_files[name] = (str(path), hashlib.sha256(path.read_bytes()).hexdigest())
    holdout_identity = holdout.identity
    resource_bindings = bindings(nltk_resource_files or {})
    if nltk_data_dir is not None:
        data_dir = Path(nltk_data_dir)
        if data_dir.is_symlink() or not data_dir.is_dir():
            raise ValueError("Fixed NLTK data directory must exist")
        nltk_data_dir = str(data_dir.resolve())
        if any(not Path(record["path"]).is_relative_to(data_dir.resolve())
               for record in resource_bindings.values()):
            raise ValueError("NLTK resource file escapes the fixed data directory")
    value = {
        "schema_version": 1, "kind": "alignlab-independent-evaluation-protocol-v1",
        "asset_manifest": {"path": assets.manifest_path, "sha256": assets.manifest_sha256},
        "holdout_manifest": {"path": holdout_identity["manifest_path"], "sha256": holdout_identity["manifest_sha256"]},
        "dataset_identities": {d.name: d.identity for d in (assets.ifeval, assets.rm_bench, holdout)},
        "model_manifests": bindings(model_manifests), "source_files": bindings(source_files),
        "nltk_resource_files": resource_bindings, "nltk_data_dir": nltk_data_dir,
        "dependency_versions": dependency_versions(),
        "scorer_seed": _integer(scorer_seed, "scorer_seed"),
        "bootstrap_seed": _integer(bootstrap_seed, "bootstrap_seed"),
        "bootstrap_replicates": 2000, "serialization_version": SERIALIZATION,
        "generation": {"do_sample": False, "enable_thinking": False, "max_new_tokens": 2048,
                       "eos_token_id": 151645, "pad_token_id": 151643, "vocab_size": 151936,
                       "input_truncation": False,
                       "response_rendering": "decode original generated IDs except terminal EOS; skip_special_tokens=False; no whitespace cleanup"},
        "denominators": {"ifeval_prompts": 541, "ifeval_instructions": 834,
                         "helpsteer3_all": 2017, "helpsteer3_non_ties": 1920, "helpsteer3_ties": 97,
                         "rm_bench_records": 1327, "rm_bench_answers": 7962, "rm_bench_comparisons": 11943},
        "primary_metric": "ifeval_strict_prompt_accuracy", "quality_selection_allowed": False,
        "training_use_allowed": False, "probability_is_quality_claim": False,
    }
    result = EvaluationProtocol(canonical(value))
    result.verify_bindings()
    return result


def ordered_results(dataset: EvaluationDataset, results) -> tuple[dict, ...]:
    dataset.verify()
    rows = list(results)
    canonical(rows)  # Reject NaN/Infinity in both scored and diagnostic fields.
    indexed = _index(rows, dataset.id_field)
    expected = dataset.ids
    if set(indexed) != set(expected):
        raise ValueError("Missing or extra result IDs; complete denominator required")
    return tuple(indexed[key] for key in expected)


def _output_identity(record, protocol: EvaluationProtocol, model_id: str):
    identity = protocol.identity
    if model_id not in identity["model_manifests"]:
        raise ValueError("Model is absent from the fixed evaluation identity")
    if (record.get("model_id") != model_id or record.get("protocol_sha256") != protocol.sha256
            or record.get("model_manifest_sha256") != identity["model_manifests"][model_id]["sha256"]):
        raise ValueError("Result model/protocol identity mismatch")
    if record.get("input_truncated") is not False:
        raise ValueError("Full context required; truncated or unspecified input is rejected")


def validate_generation_records(dataset: EvaluationDataset, results, *,
                                protocol: EvaluationProtocol, model_id: str,
                                decode_tokens) -> tuple[dict, ...]:
    """Validate all IDs and exact decoded output. Caller supplies its bound CPU decoder."""
    if dataset.name != "ifeval":
        raise ValueError("Generation schema currently supports official IFEval only")
    if not callable(decode_tokens):
        raise ValueError("A bound CPU token decoder is required")
    protocol.verify_bindings()
    if protocol.identity["dataset_identities"].get(dataset.name) != dataset.identity:
        raise ValueError("Dataset differs from the frozen evaluation protocol")
    ordered = ordered_results(dataset, results)
    settings = protocol.identity["generation"]
    lengths = dataset.lengths
    for source, row in zip(dataset.rows, ordered):
        _output_identity(row, protocol, model_id)
        if row.get("prompt") != source["prompt"] or not isinstance(row.get("response"), str):
            raise ValueError("Exact original prompt and response string required")
        if (type(row.get("generation_seed")) is not int
                or row["generation_seed"] != protocol.identity["scorer_seed"]):
            raise ValueError("Generation seed differs from protocol")
        if type(row.get("prompt_tokens")) is not int or row["prompt_tokens"] != lengths[str(source["key"])]:
            raise ValueError("Full prompt token length mismatch")
        ids = row.get("generated_token_ids")
        if (not isinstance(ids, (list, tuple)) or not ids or len(ids) > settings["max_new_tokens"]
                or any(type(i) is not int or not 0 <= i < settings["vocab_size"] for i in ids)):
            raise ValueError("Invalid generated action IDs or generation length")
        eos = settings["eos_token_id"]
        if row.get("termination") == "eos":
            if ids[-1] != eos or eos in ids[:-1]:
                raise ValueError("EOS termination must be the first real terminal EOS")
            response_ids = ids[:-1]
        elif row.get("termination") == "length":
            if len(ids) != settings["max_new_tokens"] or eos in ids:
                raise ValueError("Length termination requires full budget and no EOS")
            response_ids = ids
        else:
            raise ValueError("Incomplete generation termination")
        if decode_tokens(tuple(response_ids)) != row["response"]:
            raise ValueError("Response differs from decoded original actions")
    return tuple(parse(canonical(ordered)))
