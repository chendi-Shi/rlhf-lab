"""Independent CPU scoring of saved outputs; no model loading or generation.

IFEval executes the accepted upstream functions, with no substitute checker.
RM-Bench's arithmetic is independently implemented and tested against the two
unmodified upstream aggregation functions extracted from the pinned source.
"""
from __future__ import annotations

from contextlib import contextmanager
import importlib
import math
from pathlib import Path
import random
import sys
import types

from .evaluation_protocol_v2 import (
    DOMAINS, EvaluationAssets, EvaluationDataset, EvaluationProtocol,
    _finite, _integer, _output_identity, canonical, dependency_versions,
    ordered_results, validate_generation_records,
)


@contextmanager
def _nltk_path(directory, *, isolate_cache=False):
    if directory is None:
        yield
        return
    nltk = importlib.import_module("nltk")
    root = Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Fixed NLTK data directory is missing")
    previous_path = list(nltk.data.path)
    previous_cache = dict(nltk.data._resource_cache)
    punkt_factory = getattr(nltk.tokenize, "_get_punkt_tokenizer", None)
    try:
        nltk.data.path[:] = [str(root.resolve())]
        if isolate_cache:
            nltk.data.clear_cache()
            if punkt_factory is not None and hasattr(punkt_factory, "cache_clear"):
                punkt_factory.cache_clear()
        yield
    finally:
        nltk.data.path[:] = previous_path
        if isolate_cache:
            nltk.data.clear_cache()
            nltk.data._resource_cache.update(previous_cache)
            if punkt_factory is not None and hasattr(punkt_factory, "cache_clear"):
                punkt_factory.cache_clear()


def ifeval_dependency_diagnostics(assets: EvaluationAssets, *, nltk_data_dir=None) -> dict:
    source = "ifeval/instruction_following_eval/requirements.txt"
    raw = assets.read(source)
    versions = dependency_versions()
    missing = [name for name, version in versions.items() if version is None]
    resources, issues = {}, []
    if versions["nltk"] is not None:
        import hashlib
        nltk = importlib.import_module("nltk")
        version = tuple(int(part) for part in versions["nltk"].split(".")[:2])
        # NLTK >=3.9 redirects the old pickle URI in the official source to
        # punkt_tab before looking for that pickle file.
        required = (["tokenizers/punkt_tab/english/"] if version >= (3, 9)
                    else ["tokenizers/punkt/english.pickle"])
        with _nltk_path(nltk_data_dir):
            for resource in required:
                try:
                    location = Path(str(nltk.data.find(resource)))
                    files = sorted(location.rglob("*")) if location.is_dir() else [location]
                    if not files or any(not p.is_file() or p.is_symlink() for p in files):
                        raise ValueError("Only ordinary local NLTK resource files are supported")
                    for path in files:
                        resources[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
                except (LookupError, ValueError, OSError) as exc:
                    issues.append({"resource": resource, "reason": str(exc)})
    else:
        issues.append({"resource": "nltk punkt / version-dependent punkt_tab",
                       "reason": "nltk is not installed"})
    import hashlib
    return {"ready": not missing and not issues, "missing_packages": missing,
            "installed_versions": versions, "resource_issues": issues,
            "resource_files": resources, "requirements_source": {
                "path": str(Path(assets.manifest_path).parent / source),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "official_text": raw.decode("utf-8")},
            "fallback_available": False, "installation_performed": False}


@contextmanager
def _official_ifeval(assets, scorer_seed, nltk_data_dir=None):
    """Load verified upstream bytes without pycache or permanent module/RNG changes."""
    names = ["instruction_following_eval"] + [
        "instruction_following_eval." + name for name in
        ("instructions_util", "instructions", "instructions_registry", "evaluation_lib")]
    if any(name in sys.modules for name in names):
        raise RuntimeError("Existing IFEval modules prevent an isolated verified import")
    rng_state = random.getstate()
    langdetect = importlib.import_module("langdetect")
    old_language_seed = langdetect.DetectorFactory.seed
    factory_module = importlib.import_module("langdetect.detector_factory")
    existing_factory = getattr(factory_module, "_factory", None)
    had_instance_seed = existing_factory is not None and "seed" in vars(existing_factory)
    old_instance_seed = vars(existing_factory).get("seed") if existing_factory is not None else None
    package = types.ModuleType(names[0])
    package.__path__ = []  # Only explicitly verified module bytes are executable.
    package.__package__ = names[0]
    try:
        random.seed(scorer_seed)
        langdetect.DetectorFactory.seed = scorer_seed
        if existing_factory is not None:
            existing_factory.seed = scorer_seed
        sys.modules[names[0]] = package
        for name in names[1:]:
            relative = "ifeval/instruction_following_eval/" + name.rsplit(".", 1)[1] + ".py"
            payload = assets.read(relative)
            module = types.ModuleType(name)
            module.__package__ = names[0]
            module.__file__ = str(Path(assets.manifest_path).parent / relative)
            sys.modules[name] = module
            setattr(package, name.rsplit(".", 1)[1], module)
            exec(compile(payload, module.__file__, "exec"), module.__dict__)
        with _nltk_path(nltk_data_dir, isolate_cache=True):
            yield sys.modules[names[-1]]
    finally:
        for name in reversed(names):
            sys.modules.pop(name, None)
        random.setstate(rng_state)
        langdetect.DetectorFactory.seed = old_language_seed
        if existing_factory is not None:
            if had_instance_seed:
                existing_factory.seed = old_instance_seed
            else:
                vars(existing_factory).pop("seed", None)
        # A factory first created by this scoring call is not retained with a
        # hidden seeded state in a caller that previously had no factory.
        factory_module._factory = existing_factory


def score_ifeval(assets: EvaluationAssets, results, *, protocol: EvaluationProtocol,
                 model_id: str, decode_tokens) -> dict:
    assets.verify()
    directory = protocol.identity.get("nltk_data_dir") if protocol is not None else None
    diagnostics = ifeval_dependency_diagnostics(assets, nltk_data_dir=directory)
    if not diagnostics["ready"]:
        raise RuntimeError("Official IFEval dependencies/resources unavailable: " +
                           canonical(diagnostics).decode("utf-8"))
    if protocol.identity["dependency_versions"] != diagnostics["installed_versions"]:
        raise ValueError("Scorer package versions changed after protocol freeze")
    if directory is None:
        raise ValueError("A fixed NLTK data directory must be declared before official scoring")
    declared = {record["path"]: record["sha256"] for record in
                protocol.identity["nltk_resource_files"].values()}
    if declared != diagnostics["resource_files"]:
        raise ValueError("Exact NLTK resource files must be pinned before official scoring")
    ordered = validate_generation_records(assets.ifeval, results, protocol=protocol,
                                          model_id=model_id, decode_tokens=decode_tokens)
    responses = {row["prompt"]: row["response"] for row in ordered}
    result = {"protocol_sha256": protocol.sha256, "model_id": model_id,
              "official_evaluator": True, "denominators": {"prompts": 541, "instructions": 834},
              "dependency_diagnostics": diagnostics, "per_prompt": [], "metrics": {}}
    with _official_ifeval(assets, protocol.identity["scorer_seed"], directory) as official:
        by_mode = {}
        for mode in ("strict", "loose"):
            checker = getattr(official, "test_instruction_following_" + mode)
            outputs = []
            for raw in assets.ifeval.rows:
                inp = official.InputExample(**raw)
                output = checker(inp, responses)
                flags = output.follow_instruction_list
                if (len(flags) != len(raw["instruction_id_list"])
                        or any(type(value) is not bool for value in flags)
                        or output.follow_all_instructions != all(flags)):
                    raise RuntimeError("Unexpected official IFEval output schema")
                outputs.append({"key": raw["key"], "instruction_id_list": raw["instruction_id_list"],
                                "follow_instruction_list": flags,
                                "follow_all_instructions": output.follow_all_instructions})
            by_mode[mode] = outputs
            result["metrics"][mode + "_prompt_accuracy"] = sum(r["follow_all_instructions"] for r in outputs) / 541
            result["metrics"][mode + "_instruction_accuracy"] = sum(sum(r["follow_instruction_list"]) for r in outputs) / 834
        result["per_prompt"] = [{"key": strict["key"], "strict": strict, "loose": loose}
                                for strict, loose in zip(by_mode["strict"], by_mode["loose"])]
    return result


def _rm_metrics(rows) -> dict:
    """The official four-domain macro and 3x3 style rule; score ties are failures."""
    groups = {domain: [] for domain in DOMAINS}
    for row in rows:
        domain = row["domain"]
        if domain in ("safety-refuse", "safety-response"):
            domain = "safety"
        if domain not in groups:
            raise ValueError("Unknown RM-Bench domain")
        chosen, rejected = row["score_chosen"], row["score_rejected"]
        if (not isinstance(chosen, (list, tuple)) or not isinstance(rejected, (list, tuple))
                or len(chosen) != 3 or len(rejected) != 3):
            raise ValueError("All three chosen/rejected styles are required")
        matrix = [[int(_finite(chosen[i], "chosen score") > _finite(rejected[j], "rejected score"))
                   for j in range(3)] for i in range(3)]
        groups[domain].append(matrix)
    metrics, details = {}, {}
    for domain, matrices in groups.items():
        if not matrices:
            raise ValueError("All four RM-Bench domains are required")
        matrix = [[sum(m[i][j] for m in matrices) / len(matrices) for j in range(3)] for i in range(3)]
        hard = sum(matrix[i][j] for i in range(3) for j in range(3) if i < j) / 3
        normal = sum(matrix[i][i] for i in range(3)) / 3
        easy = sum(matrix[i][j] for i in range(3) for j in range(3) if i > j) / 3
        metrics[domain] = (hard + normal + easy) / 3
        details[domain] = {"records": len(matrices), "accuracy_matrix": matrix,
                           "hard_acc": hard, "normal_acc": normal, "easy_acc": easy}
    for kind in ("hard_acc", "normal_acc", "easy_acc"):
        metrics[kind] = sum(d[kind] for d in details.values()) / 4
    metrics["total_avg_acc"] = sum(metrics[domain] for domain in DOMAINS) / 4
    return {"metrics": metrics, "domains": details}


def aggregate_rm_bench(assets: EvaluationAssets, results, *, protocol: EvaluationProtocol,
                       model_id: str) -> dict:
    assets.verify()
    protocol.verify_bindings()
    data = assets.rm_bench
    if protocol.identity["dataset_identities"].get(data.name) != data.identity:
        raise ValueError("RM-Bench identity differs from protocol")
    ordered = ordered_results(data, results)
    lengths = data.lengths
    rows = []
    for source, output in zip(data.rows, ordered):
        _output_identity(output, protocol, model_id)
        expected = lengths[source["id"]]
        for side in ("chosen", "rejected"):
            actual = output.get("input_lengths_" + side)
            if (not isinstance(actual, list) or any(type(n) is not int for n in actual)
                    or actual != expected[side]):
                raise ValueError("Full RM-Bench input lengths required; no truncation")
        rows.append({"id": source["id"], "domain": source["domain"],
                     "score_chosen": output["score_chosen"], "score_rejected": output["score_rejected"]})
    result = _rm_metrics(rows)
    result.update(protocol_sha256=protocol.sha256, model_id=model_id,
                  denominators={"records": len(rows), "answer_scores": 6 * len(rows), "comparisons": 9 * len(rows)},
                  input_truncation=False, raw_micro_average_is_official=False)
    return result


def aggregate_helpsteer_rm(holdout: EvaluationDataset, results, *, protocol: EvaluationProtocol,
                          model_id: str) -> dict:
    """RM pair fit on all existing responses, never a generated-response quality claim."""
    protocol.verify_bindings()
    if (holdout.name != "helpsteer3_final_holdout"
            or protocol.identity["dataset_identities"].get(holdout.name) != holdout.identity):
        raise ValueError("Final holdout identity differs from protocol")
    ordered = ordered_results(holdout, results)
    correct, losses, ties = [], [], []
    lengths = holdout.lengths
    for source, row in zip(holdout.rows, ordered):
        _output_identity(row, protocol, model_id)
        if (row.get("record_id") != source["record_id"]
                or type(row.get("source_index")) is not int or row["source_index"] != source["source_index"]):
            raise ValueError("Original content ID and source observation index must be preserved")
        actual = row.get("input_lengths")
        if (not isinstance(actual, list) or any(type(n) is not int for n in actual)
                or actual != lengths[source[holdout.id_field]]):
            raise ValueError("Full final holdout context required; no length filtering")
        one, two = _finite(row["response1_score"], "response1 score"), _finite(row["response2_score"], "response2 score")
        if not math.isfinite(one - two):
            raise ValueError("Nonfinite reward margin")
        preference = source["overall_preference"]
        if preference == 0:
            ties.append(abs(one - two))
            continue
        margin = (one - two) if preference < 0 else (two - one)
        correct.append(margin > 0)
        losses.append(max(-margin, 0) + math.log1p(math.exp(-abs(margin))))
    return {"protocol_sha256": protocol.sha256, "model_id": model_id,
            "denominators": {"all": len(ordered), "non_ties": len(correct), "ties": len(ties)},
            "pair_accuracy": sum(correct) / len(correct), "pair_log_loss": sum(losses) / len(losses),
            "tie_mean_absolute_margin": sum(ties) / len(ties),
            "ties_assigned_binary_labels": False, "new_response_human_quality_claim": False}


def paired_cluster_bootstrap(left: dict, right: dict, cluster_ids: dict, *,
                             seed: int, replicates: int = 2000, confidence: float = .95) -> dict:
    """Within one trained seed, sample clusters and retain every member together.

    Record IDs are unique mapping keys; repeated cluster IDs are deliberately
    allowed. The estimand is the mean record-level difference, recomputed with
    cluster multiplicity in each resample, not a flattened across-seed sample.
    """
    if not left or set(left) != set(right) or set(left) != set(cluster_ids):
        raise ValueError("Paired scores and clusters must have identical nonempty record IDs")
    _integer(seed, "seed")
    _integer(replicates, "replicates", 2000)
    if type(confidence) not in (int, float) or not 0 < confidence < 1:
        raise ValueError("Confidence must lie strictly between zero and one")
    grouped = {}
    for key in sorted(left, key=lambda x: (type(x).__name__, str(x))):
        cluster = cluster_ids[key]
        if not isinstance(cluster, str) or not cluster:
            raise ValueError("Every record needs a nonempty cluster ID")
        difference = _finite(left[key], "left") - _finite(right[key], "right")
        if not math.isfinite(difference):
            raise ValueError("Nonfinite paired difference")
        grouped.setdefault(cluster, []).append(difference)
    clusters = [grouped[key] for key in sorted(grouped)]
    totals = [(sum(values), len(values)) for values in clusters]
    rng, draws = random.Random(seed), []
    for _ in range(replicates):
        selected = [totals[rng.randrange(len(totals))] for _ in totals]
        draws.append(sum(item[0] for item in selected) / sum(item[1] for item in selected))
    draws.sort()
    def quantile(probability):
        position = (len(draws) - 1) * probability
        low, high = math.floor(position), math.ceil(position)
        return draws[low] + (draws[high] - draws[low]) * (position - low)
    tail = (1 - confidence) / 2
    return {"estimate": sum(total for total, _ in totals) / len(left),
            "confidence_interval": [quantile(tail), quantile(1 - tail)],
            "confidence": confidence, "replicates": replicates, "bootstrap_seed": seed,
            "records": len(left), "clusters": len(clusters),
            "conditional_on_fixed_models": True, "across_seed_pooling": False}
