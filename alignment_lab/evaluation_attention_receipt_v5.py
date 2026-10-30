"""Pure-stdlib validation of observed RM attention execution, safe for supervisors."""
import copy


KIND = "alignlab-reward-attention-execution-v5"
QUERY_BLOCK_TOKENS = 128
EFFICIENT = "torch_efficient_sdpa"
BLOCKED = "torch_math_sdpa_query_blocks_full_kv"
_OBSERVATION_FIELDS = {
    "batch", "query_tokens", "key_tokens", "query_heads", "original_kv_heads", "head_dim",
    "dtype", "device", "mask", "backend", "calls",
}
_COUNTERS = {"efficient_support_checks", "efficient_supported", "efficient_executed",
             "blocked_math_executed", "max_block_rows"}
_RM_FIELDS = {"kind", "enabled", "scope", "cpu_fixture", "query_block_limit",
              "input_truncated", "all_key_columns_retained", "observations"} | _COUNTERS


def validate_attention_execution_receipt(receipt, *, enabled, minimum_calls=1):
    """Accept completed observations, never eligibility-only evidence as execution."""
    if type(enabled) is not bool or type(minimum_calls) is not int or minimum_calls < 0:
        raise ValueError("Explicit enabled bool and nonnegative minimum calls required")
    if (type(receipt) is not dict or receipt.get("kind") != KIND
            or receipt.get("enabled") is not enabled or receipt.get("scope") != "reward_only"):
        raise ValueError("Exact V5 reward attention receipt required")
    if not enabled:
        if set(receipt) != {"kind", "enabled", "scope"}:
            raise ValueError("Policy models must not report modified RM attention")
        return copy.deepcopy(receipt)
    if (set(receipt) != _RM_FIELDS or type(receipt["cpu_fixture"]) is not bool
            or type(receipt["query_block_limit"]) is not int
            or receipt["query_block_limit"] != QUERY_BLOCK_TOKENS
            or receipt["input_truncated"] is not False
            or receipt["all_key_columns_retained"] is not True
            or type(receipt["observations"]) is not list
            or any(type(receipt[k]) is not int or receipt[k] < 0 for k in _COUNTERS)):
        raise ValueError("Reward attention execution schema or fixed contract differs")
    counts = {EFFICIENT: 0, BLOCKED: 0}
    seen, maximum_block = set(), 0
    for row in receipt["observations"]:
        if (type(row) is not dict or set(row) != _OBSERVATION_FIELDS
                or any(type(row[k]) is not int or row[k] < 1 for k in
                       ("batch", "query_tokens", "key_tokens", "query_heads", "original_kv_heads", "head_dim", "calls"))
                or row["query_tokens"] != row["key_tokens"]
                or row["query_heads"] % row["original_kv_heads"] != 0
                or row["backend"] not in counts
                or row["mask"] not in {"implicit_full_causal", "torch.bool", "torch.float32", "torch.bfloat16"}):
            raise ValueError("Invalid or truncated full-KV observation")
        if receipt["cpu_fixture"]:
            if row["device"] != "cpu" or row["dtype"] not in {"torch.float32", "torch.bfloat16"} or row["backend"] != BLOCKED:
                raise ValueError("CPU fixture cannot claim production CUDA/efficient execution")
        elif (row["device"] != "cuda:0" or row["dtype"] != "torch.bfloat16"
                or row["batch"] != 1 or row["query_heads"] != 16 or row["original_kv_heads"] != 8
                or row["head_dim"] != 128 or row["query_tokens"] > 40960):
            raise ValueError("Production Qwen shape, dtype or device changed")
        signature = tuple(row[k] for k in sorted(_OBSERVATION_FIELDS - {"calls"}))
        if signature in seen:
            raise ValueError("Duplicate attention observation must be aggregated")
        seen.add(signature)
        counts[row["backend"]] += row["calls"]
        if row["backend"] == BLOCKED:
            maximum_block = max(maximum_block, min(row["query_tokens"], QUERY_BLOCK_TOKENS))
    total = sum(counts.values())
    if (total < minimum_calls or counts[EFFICIENT] != receipt["efficient_executed"]
            or counts[BLOCKED] != receipt["blocked_math_executed"]
            or maximum_block != receipt["max_block_rows"]
            or receipt["efficient_supported"] != receipt["efficient_executed"]
            or receipt["efficient_support_checks"] != (0 if receipt["cpu_fixture"] else total)):
        raise ValueError("Eligibility, completed executions and observation counters do not agree")
    return copy.deepcopy(receipt)
