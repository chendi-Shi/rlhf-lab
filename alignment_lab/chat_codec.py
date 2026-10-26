"""Qwen3 non-thinking completion serialization, protocol v3 (native NFC).

The official pinned tokenizer renders only the generation prompt with
``enable_thinking=False``. Prompt and raw response are encoded separately with
``add_special_tokens=False``; exactly one assistant EOS is appended to the raw
response IDs. The concatenation is the canonical training/evaluation sequence.
It need not equal a single BPE encoding of the joined text at its boundary.

The official final-assistant template is intentionally never used: it strips
leading newlines and may reinterpret reasoning tags. This codec preserves raw
response whitespace, supports an empty EOS-only response, and adds no trailing
inter-message newline. The pinned tokenizer's native NFC normalization is
explicitly retained; raw text is preserved separately. Exact decoding of the
NFC-normalized segments is required, without cleanup or skipped
special tokens. Completion masks include the response and EOS only, excluding
all history, assistant headers and the non-thinking empty reasoning block.
Terminal EOS is 151645 (<|im_end|>); padding is 151643 (<|endoftext|>).
The upstream generation config also treats 151643 as an EOS, so downstream
generation must explicitly adopt this codec's terminal-EOS contract. Padding
masks are derived from sequence lengths, never from ``token_id != pad_id``.

The module imports neither transformers nor torch. Training, auditing, reference
caching and evaluation must instantiate this same codec with the same pinned
tokenizer; an alternative serialization is a different experiment identity.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import unicodedata
from typing import Any

SERIALIZATION_VERSION = "qwen3-nonthinking-nfc-completion-v3"
TERMINAL_EOS_TOKEN_ID = 151645
PAD_TOKEN_ID = 151643
RESERVED_CHAT_CONTROLS = ("<|im_start|>", "<|im_end|>", "<think>", "</think>",
                          "<tool_call>", "<tool_response>")
NONTHINKING_GENERATION_SUFFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def validate_context(context: Any) -> list[dict[str, str]]:
    """Allow empty message strings, but reject tools/multimodal/control syntax."""
    if not isinstance(context, list) or not context:
        raise ValueError("invalid_context")
    result = []
    for message in context:
        if (not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant"}
                or not isinstance(message.get("content"), str)):
            raise ValueError("invalid_context_message")
        if set(message) - {"role", "content"}:
            raise ValueError("unsupported_context_message_fields")
        if any(tag in message["content"] for tag in RESERVED_CHAT_CONTROLS):
            raise ValueError("reserved_chat_control_in_context")
        result.append({"role": message["role"], "content": message["content"]})
    if result[-1]["role"] != "user":
        raise ValueError("context_must_end_with_user")
    return result


def validate_response(response: Any) -> str:
    if not isinstance(response, str):
        raise ValueError("invalid_response_type")
    if any(tag in response for tag in RESERVED_CHAT_CONTROLS):
        raise ValueError("reserved_chat_control_in_response")
    return response


@dataclass(frozen=True)
class EncodedPrompt:
    rendered: str
    input_ids: tuple[int, ...]
    normalized_rendered: str


@dataclass(frozen=True)
class EncodedCompletion:
    prompt: EncodedPrompt
    raw_response: str
    completion_ids: tuple[int, ...]

    @property
    def input_ids(self) -> tuple[int, ...]:
        return self.prompt.input_ids + self.completion_ids

    @property
    def completion_mask(self) -> tuple[bool, ...]:
        return (False,) * len(self.prompt.input_ids) + (True,) * len(self.completion_ids)

    @property
    def labels(self) -> tuple[int, ...]:
        return (-100,) * len(self.prompt.input_ids) + self.completion_ids

    def pad_to(self, length: int) -> dict[str, tuple]:
        """Right-pad using actual lengths; never infer masks from token values."""
        if type(length) is not int or length < len(self.input_ids):
            raise ValueError("padding_length_must_not_truncate")
        extra = length - len(self.input_ids)
        return {"input_ids": self.input_ids + (PAD_TOKEN_ID,) * extra,
                "attention_mask": (True,) * len(self.input_ids) + (False,) * extra,
                "completion_mask": self.completion_mask + (False,) * extra,
                "labels": self.labels + (-100,) * extra}


class Qwen3ChatCodec:
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer
        if (tokenizer.eos_token != "<|im_end|>" or type(tokenizer.eos_token_id) is not int
                or tokenizer.eos_token_id != TERMINAL_EOS_TOKEN_ID
                or tokenizer.pad_token_id != PAD_TOKEN_ID):
            raise ValueError("unsupported_qwen3_assistant_eos")
        self.eos_token = tokenizer.eos_token
        self.eos_token_id = tokenizer.eos_token_id
        self.special_token_ids = frozenset(tokenizer.all_special_ids)
        state = tokenizer.backend_tokenizer.normalizer.__getstate__()
        if json.loads(state) != {"type": "NFC"}:
            raise ValueError("unsupported_tokenizer_normalization; require pinned NFC")

    @staticmethod
    def normalized(text: str) -> str:
        return unicodedata.normalize("NFC", text)

    def _encode(self, text: str) -> tuple[int, ...]:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not isinstance(ids, list) or any(type(token) is not int for token in ids):
            raise ValueError("tokenizer_encode_did_not_return_token_id_list")
        return tuple(ids)

    def _decode(self, ids: tuple[int, ...]) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=False,
                                     clean_up_tokenization_spaces=False)

    def _reject_other_special_tokens(self, text: str) -> None:
        if any(token and token in text for token in self.tokenizer.all_special_tokens):
            raise ValueError("reserved_tokenizer_special_token_in_raw_text")

    def validate_input(self, context: Any, *responses: Any) -> None:
        """Reject literal controls before clustering; do not rewrite raw text."""
        for message in validate_context(context):
            self._reject_other_special_tokens(message["content"])
        for response in responses:
            self._reject_other_special_tokens(validate_response(response))

    def _raw_ids(self, text: str) -> tuple[int, ...]:
        self._reject_other_special_tokens(text)
        ids = self._encode(text)
        if any(token in self.special_token_ids for token in ids):
            raise ValueError("reserved_tokenizer_special_id_in_raw_text")
        return ids

    def encode_prompt(self, context: Any) -> EncodedPrompt:
        context = validate_context(context)
        for message in context:
            self._raw_ids(message["content"])
        rendered = self.tokenizer.apply_chat_template(context, tokenize=False, add_generation_prompt=True,
                                                      enable_thinking=False)
        if not isinstance(rendered, str) or not rendered.endswith(NONTHINKING_GENERATION_SUFFIX):
            raise ValueError("nonthinking_generation_prefix_missing_or_unsupported")
        ids = self._encode(rendered)
        normalized_rendered = self.normalized(rendered)
        if self._decode(ids) != normalized_rendered:
            raise ValueError("prompt_tokenization_is_not_lossless")
        return EncodedPrompt(rendered, ids, normalized_rendered)

    def encode_completion(self, prompt: EncodedPrompt, raw_response: Any) -> EncodedCompletion:
        response = validate_response(raw_response)
        # Do not encode a final-assistant message or strip any response bytes.
        raw_ids = self._raw_ids(response)
        if self.eos_token_id in raw_ids:
            raise ValueError("raw_response_contains_assistant_eos")
        completion_ids = raw_ids + (self.eos_token_id,)
        result = EncodedCompletion(prompt, response, completion_ids)
        normalized_response = self.normalized(response)
        expected = prompt.normalized_rendered + normalized_response + self.eos_token
        if self._decode(result.input_ids) != expected:
            raise ValueError("concatenated_prompt_completion_is_not_lossless")
        if self._decode(completion_ids) != normalized_response + self.eos_token:
            raise ValueError("response_tokenization_is_not_lossless")
        return result

    def encode(self, context: Any, raw_response: Any) -> EncodedCompletion:
        return self.encode_completion(self.encode_prompt(context), raw_response)
