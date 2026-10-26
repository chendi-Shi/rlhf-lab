"""Pure token-list transformations, shared between training and generation."""

import re


def encode_prompt(tokenizer, prompt, config):
    if not prompt.strip():
        raise ValueError("Empty prompt after tokenization")
    # In this dataset the article has no summary marker; chosen/rejected often do.
    text = prompt.rstrip() + "\n\nTL;DR:"
    tokens = tokenizer.encode(text, add_special_tokens=False)
    # Preserve the explicit task boundary. Long articles lose earlier context in this pilot.
    return tokens[-config["max_prompt_tokens"]:]


def encode_response(tokenizer, response, config):
    response = re.sub(r"^\s*TL\s*;\s*DR\s*:\s*", "", response, flags=re.IGNORECASE).strip()
    if not response:
        raise ValueError("Empty summary after removing the dataset prefix")
    tokens = tokenizer.encode(" " + response, add_special_tokens=False)
    if not tokens:
        raise ValueError("Empty response after tokenization")
    if tokens[-1] == tokenizer.eos_token_id:
        tokens = tokens[:-1]
    budget = config["max_response_tokens"]
    if config.get("completion_eos_policy") == "natural" and len(tokens) >= budget:
        # A length limit is not an observed end of a human summary.
        return tokens[:budget]
    return tokens[: budget - 1] + [tokenizer.eos_token_id]


def encode_sft(tokenizer, row, config):
    prompt = encode_prompt(tokenizer, row["prompt"], config)
    answer = encode_response(tokenizer, row["response"], config)
    return {"input_ids": prompt + answer, "attention_mask": [1] * (len(prompt) + len(answer)),
            "labels": [-100] * len(prompt) + answer}


def encode_pair(tokenizer, row, config):
    prompt = encode_prompt(tokenizer, row["prompt"], config)
    chosen = encode_response(tokenizer, row["chosen"], config)
    rejected = encode_response(tokenizer, row["rejected"], config)
    if chosen == rejected:
        return None  # Truncation can remove the only difference; such a pair has no preference signal.
    return {"chosen_ids": prompt + chosen, "rejected_ids": prompt + rejected}
