#!/usr/bin/env python3
"""Standalone, content-addressed Behavior Patch Bundle v1 tool.

This file is generated. It treats bundles, contracts, probes, checkpoint
indexes, and output paths as untrusted data. The only trusted executable input
accepted by the verifier is the adapter explicitly named by the operator.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import os
import platform
import re
import shutil
import sys
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import Tensor, nn

_TINY_MAX_CONFIG_BYTES = 1024 * 1024
_TINY_MAX_CHECKPOINT_BYTES = 16 * 1024**3
_TINY_MAX_TENSORS = 10_000
_TINY_MAX_MODEL_ELEMENTS = 100_000_000
_TINY_MAX_GENERATED_TOKENS = 4096


def _tiny_unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate tiny-model configuration key: {key!r}")
        result[key] = value
    return result


def _tiny_reject_constant(value: str) -> None:
    raise ValueError(f"non-finite tiny-model configuration value: {value}")


class GenerationPolicyLike(Protocol):
    mode: str
    max_new_tokens: int
    seed: int
    temperature: float
    stop_on_eos: bool


@dataclass(frozen=True, slots=True)
class ModelBatch:
    input_ids: Tensor
    attention_mask: Tensor

    def __post_init__(self) -> None:
        if self.input_ids.ndim != 2 or self.attention_mask.shape != self.input_ids.shape:
            raise ValueError("input IDs and attention mask must be equally shaped matrices")
        if self.input_ids.dtype != torch.long or self.attention_mask.dtype != torch.bool:
            raise ValueError("invalid tiny-model batch dtypes")

    def to(self, device: torch.device | str) -> ModelBatch:
        return ModelBatch(self.input_ids.to(device), self.attention_mask.to(device))


@dataclass(frozen=True, slots=True)
class GeneratedSample:
    token_ids: tuple[int, ...]
    text: str
    finished: bool


@dataclass(frozen=True, slots=True)
class TinyConfig:
    vocab_size: int = 259
    max_sequence_length: int = 128
    hidden_size: int = 32
    intermediate_size: int = 64
    num_layers: int = 2
    num_heads: int = 4
    rms_norm_epsilon: float = 1e-6
    tie_word_embeddings: bool = True
    initialization_seed: int = 17

    def __post_init__(self) -> None:
        if self.vocab_size != 259:
            raise ValueError("standalone tiny adapter requires the fixed 259-token vocabulary")
        if not 2 <= self.max_sequence_length <= 65_536:
            raise ValueError("invalid maximum sequence length")
        if not 4 <= self.hidden_size <= 16_384 or self.hidden_size % self.num_heads:
            raise ValueError("hidden size must be positive and divisible by the head count")
        if not 1 <= self.num_layers <= 256 or not 1 <= self.num_heads <= 256:
            raise ValueError("invalid layer or head count")
        if self.intermediate_size < self.hidden_size:
            raise ValueError("intermediate size must be at least the hidden size")
        if not 0 < self.rms_norm_epsilon < 1:
            raise ValueError("invalid RMSNorm epsilon")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> TinyConfig:
        allowed = {
            "hidden_size",
            "initialization_seed",
            "intermediate_size",
            "max_sequence_length",
            "num_heads",
            "num_layers",
            "rms_norm_epsilon",
            "tie_word_embeddings",
            "vocab_size",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown tiny-model configuration fields: {sorted(unknown)}")

        def integer(name: str, default: int) -> int:
            item = value.get(name, default)
            if isinstance(item, bool) or not isinstance(item, int):
                raise ValueError(f"tiny-model configuration field must be an integer: {name}")
            return item

        epsilon = value.get("rms_norm_epsilon", 1e-6)
        if isinstance(epsilon, bool) or not isinstance(epsilon, int | float):
            raise ValueError("RMSNorm epsilon must be numeric")
        tied = value.get("tie_word_embeddings", True)
        if not isinstance(tied, bool):
            raise ValueError("tie_word_embeddings must be boolean")
        return cls(
            vocab_size=integer("vocab_size", 259),
            max_sequence_length=integer("max_sequence_length", 128),
            hidden_size=integer("hidden_size", 32),
            intermediate_size=integer("intermediate_size", 64),
            num_layers=integer("num_layers", 2),
            num_heads=integer("num_heads", 4),
            rms_norm_epsilon=float(epsilon),
            tie_word_embeddings=tied,
            initialization_seed=integer("initialization_seed", 17),
        )


class TinyTokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    byte_offset = 3
    vocab_size = 259

    def encode(self, text: str, *, add_bos: bool = True, add_eos: bool = False) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        tokens = [self.byte_offset + byte for byte in text.encode("utf-8")]
        if add_bos:
            tokens.insert(0, self.bos_token_id)
        if add_eos:
            tokens.append(self.eos_token_id)
        return tokens

    def decode(self, token_ids: Sequence[int], *, skip_special_tokens: bool = True) -> str:
        output = bytearray()
        for token in token_ids:
            if isinstance(token, bool) or not isinstance(token, int):
                raise ValueError("token IDs must be integers")
            if not 0 <= token < self.vocab_size:
                raise ValueError(f"invalid token ID: {token}")
            if token < self.byte_offset:
                if skip_special_tokens:
                    continue
                output.extend(("<pad>", "<bos>", "<eos>")[token].encode())
            else:
                output.append(token - self.byte_offset)
        return output.decode("utf-8", errors="replace")

    def batch(self, texts: Sequence[str], *, add_bos: bool = True) -> ModelBatch:
        if not texts:
            raise ValueError("cannot encode an empty batch")
        encoded = [self.encode(text, add_bos=add_bos) for text in texts]
        width = max(map(len, encoded))
        ids = torch.full((len(encoded), width), self.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(encoded), width), dtype=torch.bool)
        for row, tokens in enumerate(encoded):
            ids[row, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
            mask[row, : len(tokens)] = True
        return ModelBatch(ids, mask)


class TinyRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, epsilon: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.epsilon = epsilon

    def forward(self, hidden: Tensor) -> Tensor:
        variance = hidden.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden * torch.rsqrt(variance.to(hidden.dtype) + self.epsilon)
        return normalized * self.weight


class TinySelfAttention(nn.Module):
    def __init__(self, config: TinyConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_size = config.hidden_size // config.num_heads
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, hidden: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        batch, length, width = hidden.shape

        def split(value: Tensor) -> Tensor:
            return value.view(batch, length, self.num_heads, self.head_size).transpose(1, 2)

        query, key, value = (
            split(self.q_proj(hidden)),
            split(self.k_proj(hidden)),
            split(self.v_proj(hidden)),
        )
        scores = query @ key.transpose(-1, -2) / math.sqrt(self.head_size)
        causal = torch.ones((length, length), device=hidden.device, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(causal, torch.finfo(scores.dtype).min)
        if attention_mask is not None:
            key_mask = ~attention_mask.to(dtype=torch.bool)[:, None, None, :]
            scores = scores.masked_fill(key_mask, torch.finfo(scores.dtype).min)
        probabilities = F.softmax(scores.float(), dim=-1).to(hidden.dtype)
        context = probabilities @ value
        context = context.transpose(1, 2).contiguous().view(batch, length, width)
        return cast(Tensor, self.o_proj(context))


class TinyMLP(nn.Module):
    def __init__(self, config: TinyConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden: Tensor) -> Tensor:
        return cast(Tensor, self.down_proj(F.silu(self.gate_proj(hidden)) * self.up_proj(hidden)))


class TinyBlock(nn.Module):
    def __init__(self, config: TinyConfig) -> None:
        super().__init__()
        self.input_norm = TinyRMSNorm(config.hidden_size, config.rms_norm_epsilon)
        self.attention = TinySelfAttention(config)
        self.post_attention_norm = TinyRMSNorm(config.hidden_size, config.rms_norm_epsilon)
        self.mlp = TinyMLP(config)

    def forward(self, hidden: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        hidden = hidden + self.attention(self.input_norm(hidden), attention_mask)
        return cast(Tensor, hidden + self.mlp(self.post_attention_norm(hidden)))


@dataclass(frozen=True, slots=True)
class TinyCausalLMOutput:
    logits: Tensor
    hidden_states: tuple[Tensor, ...]


class TinyCausalLM(nn.Module):
    def __init__(self, config: TinyConfig) -> None:
        super().__init__()
        self.config = config
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.initialization_seed)
            self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
            self.position_embedding = nn.Embedding(config.max_sequence_length, config.hidden_size)
            self.layers = nn.ModuleList(TinyBlock(config) for _ in range(config.num_layers))
            self.final_norm = TinyRMSNorm(config.hidden_size, config.rms_norm_epsilon)
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self.apply(self._initialize)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear | nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        *,
        output_hidden_states: bool = False,
    ) -> TinyCausalLMOutput:
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError("input IDs must be a rank-2 long tensor")
        _, length = input_ids.shape
        if length > self.config.max_sequence_length:
            raise ValueError("input exceeds maximum sequence length")
        if attention_mask is not None and attention_mask.shape != input_ids.shape:
            raise ValueError("attention mask shape mismatch")
        positions = torch.arange(length, device=input_ids.device)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
        states: list[Tensor] = [hidden] if output_hidden_states else []
        for layer in self.layers:
            hidden = layer(hidden, attention_mask)
            if output_hidden_states:
                states.append(hidden)
        hidden = self.final_norm(hidden)
        if output_hidden_states:
            states.append(hidden)
        return TinyCausalLMOutput(self.lm_head(hidden), tuple(states))


def _read_config(root: Path) -> TinyConfig:
    config_path = root / "config.json"
    if config_path.is_symlink() or not config_path.is_file():
        raise ValueError("tiny checkpoint requires a regular config.json")
    if config_path.stat().st_size > _TINY_MAX_CONFIG_BYTES:
        raise ValueError("tiny-model configuration exceeds the size limit")
    try:
        value = json.loads(
            config_path.read_text(encoding="utf-8"),
            object_pairs_hook=_tiny_unique_object,
            parse_constant=_tiny_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, OSError) as error:
        raise ValueError("malformed tiny-model configuration") from error
    if not isinstance(value, dict):
        raise ValueError("tiny-model configuration must be an object")
    allowed = {"architectures", "model_config", "model_type", "schema_version"}
    if set(value) - allowed:
        raise ValueError("unknown top-level tiny-model configuration fields")
    if value.get("schema_version") != 1:
        raise ValueError("unsupported tiny-model configuration version")
    if value.get("model_type") != "modelpact_tiny_causal_lm":
        raise ValueError("checkpoint is not the supported tiny causal LM")
    architecture = value.get("architectures")
    if architecture is not None and architecture != ["TinyCausalLM"]:
        raise ValueError("unexpected tiny-model architecture declaration")
    raw = value.get("model_config")
    if not isinstance(raw, dict) or not all(isinstance(key, str) for key in raw):
        raise ValueError("malformed tiny-model configuration payload")
    return TinyConfig.from_mapping(raw)


def _expected_shapes(config: TinyConfig) -> dict[str, tuple[int, ...]]:
    hidden = config.hidden_size
    intermediate = config.intermediate_size
    shapes: dict[str, tuple[int, ...]] = {
        "final_norm.weight": (hidden,),
        "lm_head.weight": (config.vocab_size, hidden),
        "position_embedding.weight": (config.max_sequence_length, hidden),
        "token_embedding.weight": (config.vocab_size, hidden),
    }
    for index in range(config.num_layers):
        prefix = f"layers.{index}"
        shapes.update(
            {
                f"{prefix}.attention.k_proj.weight": (hidden, hidden),
                f"{prefix}.attention.o_proj.weight": (hidden, hidden),
                f"{prefix}.attention.q_proj.weight": (hidden, hidden),
                f"{prefix}.attention.v_proj.weight": (hidden, hidden),
                f"{prefix}.input_norm.weight": (hidden,),
                f"{prefix}.mlp.down_proj.weight": (hidden, intermediate),
                f"{prefix}.mlp.gate_proj.weight": (intermediate, hidden),
                f"{prefix}.mlp.up_proj.weight": (intermediate, hidden),
                f"{prefix}.post_attention_norm.weight": (hidden,),
            }
        )
    elements = sum(math.prod(shape) for shape in shapes.values())
    if elements > _TINY_MAX_MODEL_ELEMENTS:
        raise ValueError("tiny-model state exceeds the standalone adapter element limit")
    return shapes


def _load_tensors(root: Path, shapes: Mapping[str, tuple[int, ...]]) -> dict[str, Tensor]:
    if (root / "model.safetensors.index.json").exists():
        raise ValueError("standalone tiny adapter does not support sharded checkpoints")
    candidates = sorted(root.glob("*.safetensors"))
    if len(candidates) != 1:
        raise ValueError("tiny checkpoint must contain exactly one SafeTensors file")
    tensor_path = candidates[0]
    if tensor_path.is_symlink() or not tensor_path.is_file():
        raise ValueError("tiny checkpoint tensor file must be regular")
    if tensor_path.stat().st_size > _TINY_MAX_CHECKPOINT_BYTES:
        raise ValueError("tiny checkpoint exceeds the file-size limit")
    tensors = load_file(tensor_path, device="cpu")
    if len(tensors) > _TINY_MAX_TENSORS or set(tensors) != set(shapes):
        raise ValueError("tiny checkpoint state keys do not match the declared architecture")
    total = 0
    for name, tensor in tensors.items():
        if tensor.layout != torch.strided or not tensor.is_floating_point():
            raise ValueError(f"unsupported tensor representation: {name}")
        if tuple(tensor.shape) != shapes[name]:
            raise ValueError(f"tiny checkpoint tensor shape mismatch: {name}")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"tiny checkpoint contains a non-finite tensor: {name}")
        total += tensor.numel()
        if total > _TINY_MAX_MODEL_ELEMENTS:
            raise ValueError("tiny checkpoint state exceeds the element limit")
    return tensors


class TinyModelAdapter:
    adapter_id = "modelpact.tiny_causal_lm.v1"

    def __init__(self) -> None:
        self._tokenizer = TinyTokenizer()

    def load(
        self,
        checkpoint: str,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> TinyCausalLM:
        root = Path(checkpoint)
        if root.is_symlink() or not root.is_dir():
            raise ValueError("tiny checkpoint must be a regular directory")
        config = _read_config(root)
        shapes = _expected_shapes(config)
        tensors = _load_tensors(root, shapes)
        if config.tie_word_embeddings and not torch.equal(
            tensors["token_embedding.weight"], tensors["lm_head.weight"]
        ):
            raise ValueError("checkpoint violates its tied embedding alias")
        model = TinyCausalLM(config)
        model.load_state_dict(tensors, strict=True)
        return model.to(device=device, dtype=dtype)

    def tokenizer(self) -> TinyTokenizer:
        return self._tokenizer

    def prepare(self, model: nn.Module) -> None:
        if not isinstance(model, TinyCausalLM):
            raise TypeError("tiny adapter requires TinyCausalLM")
        model.eval()

    def forward_logits(self, model: nn.Module, batch: ModelBatch) -> Tensor:
        if not isinstance(model, TinyCausalLM):
            raise TypeError("tiny adapter requires TinyCausalLM")
        moved = batch.to(next(model.parameters()).device)
        output = cast(TinyCausalLMOutput, model(moved.input_ids, moved.attention_mask))
        return output.logits

    def generate(
        self,
        model: nn.Module,
        batch: ModelBatch,
        policy: GenerationPolicyLike,
    ) -> list[GeneratedSample]:
        if not isinstance(model, TinyCausalLM):
            raise TypeError("tiny adapter requires TinyCausalLM")
        if policy.mode not in {"greedy", "sample"}:
            raise ValueError("unsupported generation mode")
        if not 1 <= policy.max_new_tokens <= _TINY_MAX_GENERATED_TOKENS:
            raise ValueError("generation token budget is outside limits")
        if not math.isfinite(policy.temperature) or policy.temperature <= 0:
            raise ValueError("generation temperature must be finite and positive")
        device = next(model.parameters()).device
        moved = batch.to(device)
        sequences = [
            moved.input_ids[row, moved.attention_mask[row]].clone()
            for row in range(moved.input_ids.shape[0])
        ]
        generated: list[list[int]] = [[] for _ in sequences]
        finished = [False] * len(sequences)
        generator = torch.Generator(device=device).manual_seed(policy.seed)
        prior_mode = model.training
        model.eval()
        try:
            with torch.no_grad():
                for _ in range(policy.max_new_tokens):
                    for row, sequence in enumerate(sequences):
                        if finished[row]:
                            continue
                        if sequence.numel() >= model.config.max_sequence_length:
                            finished[row] = True
                            continue
                        logits = model(sequence[None, :]).logits[0, -1]
                        if policy.mode == "greedy":
                            next_token = int(torch.argmax(logits).item())
                        else:
                            probabilities = F.softmax(logits.float() / policy.temperature, dim=-1)
                            next_token = int(
                                torch.multinomial(probabilities, 1, generator=generator).item()
                            )
                        generated[row].append(next_token)
                        sequences[row] = torch.cat(
                            (
                                sequence,
                                torch.tensor([next_token], device=device, dtype=torch.long),
                            )
                        )
                        if policy.stop_on_eos and next_token == self._tokenizer.eos_token_id:
                            finished[row] = True
                    if all(finished):
                        break
        finally:
            model.train(prior_mode)
        return [
            GeneratedSample(tuple(tokens), self._tokenizer.decode(tokens), finished[index])
            for index, tokens in enumerate(generated)
        ]

import json as _hf_json_module
import math as _hf_math
import os as _hf_os
from collections.abc import Sequence as _HFSequence
from dataclasses import dataclass as _hf_dataclass
from pathlib import Path as _HFPath
from typing import Protocol as _HFProtocol
from typing import cast as _hf_cast

import torch as _hf_torch
from safetensors import safe_open as _hf_safe_open
from torch import Tensor as _HFTensor
from torch import nn as _hf_nn

_HF_MAX_CONFIG_BYTES = 16 * 1024 * 1024
_HF_MAX_CHECKPOINT_BYTES = 16 * 1024**3
_HF_MAX_FILES = 10_000
_HF_MAX_TENSORS = 100_000
_HF_MAX_GENERATED_TOKENS = 4096
_HF_UNSAFE_WEIGHT_SUFFIXES = frozenset(
    {".bin", ".ckpt", ".h5", ".msgpack", ".pickle", ".pkl", ".pt", ".pth"}
)


def _hf_unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate Hugging Face configuration key: {key!r}")
        result[key] = value
    return result


def _hf_reject_constant(value: str) -> None:
    raise ValueError(f"non-finite Hugging Face configuration value: {value}")


def _hf_validate_depth(value: object, depth: int = 0) -> None:
    if depth > 32:
        raise ValueError("Hugging Face configuration exceeds the nesting limit")
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("Hugging Face configuration keys must be strings")
            _hf_validate_depth(item, depth + 1)
    elif isinstance(value, list):
        for item in value:
            _hf_validate_depth(item, depth + 1)


def _hf_read_configuration(path: _HFPath) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("local Hugging Face checkpoint requires a regular config.json")
    if path.stat().st_size > _HF_MAX_CONFIG_BYTES:
        raise ValueError("Hugging Face configuration exceeds the size limit")
    try:
        value = _hf_json_module.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_hf_unique_object,
            parse_constant=_hf_reject_constant,
        )
    except (OSError, UnicodeDecodeError, _hf_json_module.JSONDecodeError) as error:
        raise ValueError("malformed Hugging Face configuration") from error
    if not isinstance(value, dict):
        raise ValueError("Hugging Face configuration must be an object")
    _hf_validate_depth(value)
    if not isinstance(value.get("model_type"), str):
        raise ValueError("Hugging Face configuration has no model_type")
    return value


def _hf_checkpoint_root(checkpoint: str) -> _HFPath:
    root = _HFPath(checkpoint)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Hugging Face checkpoint must be a local regular directory")
    _hf_read_configuration(root / "config.json")
    entries = list(root.rglob("*"))
    if len(entries) > _HF_MAX_FILES:
        raise ValueError("Hugging Face checkpoint contains too many files")
    for entry in entries:
        if entry.is_symlink():
            raise ValueError("Hugging Face checkpoint may not contain symlinks")
        if entry.is_file() and entry.suffix.lower() in _HF_UNSAFE_WEIGHT_SUFFIXES:
            raise ValueError("Hugging Face checkpoint contains a non-SafeTensors weight file")
    if (root / "model.safetensors.index.json").exists():
        raise ValueError("standalone Hugging Face verification does not support sharded weights")
    tensors = sorted(root.glob("*.safetensors"))
    if len(tensors) != 1 or tensors[0].name != "model.safetensors":
        raise ValueError("Hugging Face checkpoint requires exactly one model.safetensors file")
    tensor_path = tensors[0]
    if tensor_path.is_symlink() or not tensor_path.is_file():
        raise ValueError("Hugging Face SafeTensors checkpoint must be a regular file")
    if tensor_path.stat().st_size > _HF_MAX_CHECKPOINT_BYTES:
        raise ValueError("Hugging Face SafeTensors checkpoint exceeds the size limit")
    try:
        # SafeTensors does not currently publish callable typing for safe_open.
        with _hf_safe_open(  # type: ignore[no-untyped-call]
            tensor_path, framework="pt", device="cpu"
        ) as handle:
            keys = handle.keys()
            if not keys or len(keys) > _HF_MAX_TENSORS:
                raise ValueError("Hugging Face checkpoint has an invalid tensor count")
    except ValueError:
        raise
    except Exception as error:
        raise ValueError("invalid Hugging Face SafeTensors checkpoint") from error
    return root.resolve(strict=True)


class _HFGenerationPolicyLike(_HFProtocol):
    mode: str
    max_new_tokens: int
    seed: int
    temperature: float
    top_k: int | None
    top_p: float
    stop_on_eos: bool


@_hf_dataclass(frozen=True, slots=True)
class HuggingFaceModelBatch:
    input_ids: _HFTensor
    attention_mask: _HFTensor

    def __post_init__(self) -> None:
        if self.input_ids.ndim != 2 or self.attention_mask.shape != self.input_ids.shape:
            raise ValueError("input IDs and attention mask must be equally shaped matrices")
        if self.input_ids.dtype != _hf_torch.long or self.attention_mask.dtype != _hf_torch.bool:
            raise ValueError("invalid Hugging Face batch dtypes")

    def to(self, device: _hf_torch.device | str) -> HuggingFaceModelBatch:
        return HuggingFaceModelBatch(self.input_ids.to(device), self.attention_mask.to(device))


@_hf_dataclass(frozen=True, slots=True)
class HuggingFaceGeneratedSample:
    token_ids: tuple[int, ...]
    text: str
    finished: bool


class StandaloneHuggingFaceTokenizer:
    def __init__(self, tokenizer: object) -> None:
        self._tokenizer = tokenizer

    @property
    def pad_token_id(self) -> int:
        value = getattr(self._tokenizer, "pad_token_id", None)
        if value is None:
            value = getattr(self._tokenizer, "eos_token_id", None)
        if value is None:
            raise ValueError("local tokenizer has neither a pad token nor an EOS fallback")
        return int(value)

    @property
    def bos_token_id(self) -> int:
        value = getattr(self._tokenizer, "bos_token_id", None)
        if value is None:
            raise ValueError("local tokenizer has no BOS token")
        return int(value)

    @property
    def eos_token_id(self) -> int:
        value = getattr(self._tokenizer, "eos_token_id", None)
        if value is None:
            raise ValueError("local tokenizer has no EOS token")
        return int(value)

    def encode(self, text: str, *, add_bos: bool = True, add_eos: bool = False) -> list[int]:
        encode = getattr(self._tokenizer, "encode", None)
        if not callable(encode):
            raise TypeError("local tokenizer has no encode method")
        tokens = [int(token) for token in encode(text, add_special_tokens=False)]
        if add_bos:
            tokens.insert(0, self.bos_token_id)
        if add_eos:
            tokens.append(self.eos_token_id)
        return tokens

    def decode(self, token_ids: _HFSequence[int], *, skip_special_tokens: bool = True) -> str:
        decode = getattr(self._tokenizer, "decode", None)
        if not callable(decode):
            raise TypeError("local tokenizer has no decode method")
        return str(decode(list(token_ids), skip_special_tokens=skip_special_tokens))

    def batch(self, texts: _HFSequence[str], *, add_bos: bool = True) -> HuggingFaceModelBatch:
        if not texts:
            raise ValueError("cannot encode an empty batch")
        encoded = [self.encode(text, add_bos=add_bos) for text in texts]
        width = max(map(len, encoded))
        ids = _hf_torch.full((len(encoded), width), self.pad_token_id, dtype=_hf_torch.long)
        mask = _hf_torch.zeros_like(ids, dtype=_hf_torch.bool)
        for row, tokens in enumerate(encoded):
            start = width - len(tokens)
            ids[row, start:] = _hf_torch.tensor(tokens, dtype=_hf_torch.long)
            mask[row, start:] = True
        return HuggingFaceModelBatch(ids, mask)


class StandaloneHuggingFaceModelAdapter:
    adapter_id = "modelpact.huggingface_causal_lm.local.v1"

    def __init__(self) -> None:
        self._tokenizer_adapter: StandaloneHuggingFaceTokenizer | None = None

    def load(
        self,
        checkpoint: str,
        *,
        device: _hf_torch.device | str = "cpu",
        dtype: _hf_torch.dtype = _hf_torch.float32,
    ) -> _hf_nn.Module:
        root = _hf_checkpoint_root(checkpoint)
        # Enforce offline behavior in this process even when the caller did not
        # configure the conventional Hugging Face environment variables.
        _hf_os.environ["HF_HUB_OFFLINE"] = "1"
        _hf_os.environ["TRANSFORMERS_OFFLINE"] = "1"
        _hf_os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "standalone Hugging Face verification requires Transformers and its "
                f"local dependencies: {error}"
            ) from error
        tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
            str(root),
            local_files_only=True,
            trust_remote_code=False,
        )
        loaded = AutoModelForCausalLM.from_pretrained(
            str(root),
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            torch_dtype=dtype,
        )
        if not isinstance(loaded, _hf_nn.Module):
            raise TypeError("Transformers returned a non-module causal LM")
        model_config = getattr(loaded, "config", None)
        if getattr(model_config, "is_encoder_decoder", False) is True:
            raise ValueError("checkpoint is not a decoder-only causal language model")
        if model_config is not None:
            model_config.use_cache = False
        self._tokenizer_adapter = StandaloneHuggingFaceTokenizer(tokenizer)
        model = _hf_cast(_hf_nn.Module, loaded)
        model.eval()
        return model.to(device)

    def tokenizer(self) -> StandaloneHuggingFaceTokenizer:
        if self._tokenizer_adapter is None:
            raise RuntimeError("load a checkpoint before requesting its tokenizer")
        return self._tokenizer_adapter

    def prepare(self, model: _hf_nn.Module) -> None:
        model.eval()
        model_config = getattr(model, "config", None)
        if model_config is not None:
            model_config.use_cache = False

    def forward_logits(self, model: _hf_nn.Module, batch: HuggingFaceModelBatch) -> _HFTensor:
        device = next(model.parameters()).device
        moved = batch.to(device)
        output = model(input_ids=moved.input_ids, attention_mask=moved.attention_mask)
        logits = getattr(output, "logits", None)
        if not isinstance(logits, _HFTensor):
            raise TypeError("causal-LM adapter expected tensor logits")
        return logits

    def generate(
        self,
        model: _hf_nn.Module,
        batch: HuggingFaceModelBatch,
        policy: _HFGenerationPolicyLike,
    ) -> list[HuggingFaceGeneratedSample]:
        if policy.mode not in {"greedy", "sample"}:
            raise ValueError("unsupported generation mode")
        if not 1 <= policy.max_new_tokens <= _HF_MAX_GENERATED_TOKENS:
            raise ValueError("generation token budget is outside limits")
        if not _hf_math.isfinite(policy.temperature) or policy.temperature <= 0:
            raise ValueError("generation temperature must be finite and positive")
        if not _hf_math.isfinite(policy.top_p) or not 0 < policy.top_p <= 1:
            raise ValueError("generation top_p must be in (0, 1]")
        if policy.top_k is not None and policy.top_k < 1:
            raise ValueError("generation top_k must be positive")
        tokenizer = self.tokenizer()
        device = next(model.parameters()).device
        moved = batch.to(device)
        prior_mode = model.training
        model.eval()
        try:
            fork_devices = (
                [device.index if device.index is not None else _hf_torch.cuda.current_device()]
                if device.type == "cuda"
                else []
            )
            with _hf_torch.no_grad(), _hf_torch.random.fork_rng(devices=fork_devices):
                _hf_torch.manual_seed(policy.seed)
                if device.type == "cuda":
                    _hf_torch.cuda.manual_seed_all(policy.seed)
                generate = getattr(model, "generate", None)
                if not callable(generate):
                    raise TypeError("causal-LM module has no generate method")
                arguments: dict[str, object] = {
                    "input_ids": moved.input_ids,
                    "attention_mask": moved.attention_mask,
                    "do_sample": policy.mode == "sample",
                    "max_new_tokens": policy.max_new_tokens,
                    "eos_token_id": (tokenizer.eos_token_id if policy.stop_on_eos else None),
                    "pad_token_id": tokenizer.pad_token_id,
                    "use_cache": False,
                }
                if policy.mode == "sample":
                    arguments["temperature"] = policy.temperature
                    arguments["top_p"] = policy.top_p
                    if policy.top_k is not None:
                        arguments["top_k"] = policy.top_k
                output = generate(**arguments)
        finally:
            model.train(prior_mode)
        if not isinstance(output, _HFTensor):
            raise TypeError("causal-LM generate returned a non-tensor result")
        prompt_width = moved.input_ids.shape[1]
        samples: list[HuggingFaceGeneratedSample] = []
        for row in range(output.shape[0]):
            token_ids = tuple(int(token) for token in output[row, prompt_width:].tolist())
            finished = bool(token_ids and token_ids[-1] == tokenizer.eos_token_id)
            samples.append(
                HuggingFaceGeneratedSample(token_ids, tokenizer.decode(token_ids), finished)
            )
        return samples


MODE = "verify"
EXPECTED_PATCH_ID = "sha256:92c91fa7a5270f3c80d9f35c36ff46e9d56a9cc9d0435a783ad3192643db2c61"
EXPECTED_EVIDENCE_ID = "sha256:3c192d649ce52eb7899373b2809a9726777e695bfb8f0c1e503be6a68b4c1b00"
DEFAULT_PATCH_RELATIVE = "."
DEFAULT_PATCH = (
    None
    if DEFAULT_PATCH_RELATIVE is None
    else str(
        (
            Path(__file__).resolve().parent
            / Path(*PurePosixPath(DEFAULT_PATCH_RELATIVE).parts)
        ).resolve()
    )
)

MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_CONTRACT_BYTES = 2 * 1024 * 1024
MAX_PROBE_BYTES = 64 * 1024 * 1024
MAX_LINE_BYTES = 2 * 1024 * 1024
MAX_FILE_BYTES = 16 * 1024**3
MAX_AUXILIARY_BYTES = 1024**3
MAX_TENSORS = 100_000
MAX_TENSOR_ELEMENTS = 1 << 40
MAX_DELTA_ELEMENTS = 1 << 34
MAX_TARGETS = 100_000
MAX_DEPTH = 32
MAX_SUM_TERMS = 4096
MAX_RECORDS = 100_000
MAX_TEXT = 1_000_000
DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

AUXILIARY_FILES = frozenset(
    {
        "added_tokens.json",
        "config.json",
        "generation_config.json",
        "merges.txt",
        "preprocessor_config.json",
        "special_tokens_map.json",
        "spiece.model",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "vocab.json",
    }
)


class ToolError(Exception):
    def __init__(self, message: str, *, outcome: str = "FAIL", exit_code: int = 2) -> None:
        super().__init__(message)
        self.outcome = outcome
        self.exit_code = exit_code


class UnsupportedError(ToolError):
    def __init__(self, message: str) -> None:
        super().__init__(message, outcome="UNSUPPORTED", exit_code=3)


class InconclusiveError(ToolError):
    def __init__(self, message: str) -> None:
        super().__init__(message, outcome="INCONCLUSIVE", exit_code=4)


def _reject_constant(value: str) -> None:
    raise ToolError(f"non-finite JSON number is forbidden: {value}")


def _object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ToolError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _validate_data(value: object, *, maximum_depth: int = 64, maximum_nodes: int = 200_000) -> None:
    nodes = 0
    stack: list[tuple[object, int, str]] = [(value, 0, "$")]
    while stack:
        item, depth, location = stack.pop()
        nodes += 1
        if nodes > maximum_nodes:
            raise ToolError("data document exceeds node limit")
        if depth > maximum_depth:
            raise ToolError(f"data document exceeds depth limit at {location}")
        if item is None or isinstance(item, (str, bool, int)):
            if isinstance(item, str) and (len(item) > MAX_TEXT or "\x00" in item):
                raise ToolError(f"invalid bounded string at {location}")
            continue
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ToolError(f"non-finite number at {location}")
            continue
        if isinstance(item, dict):
            if len(item) > 100_000:
                raise ToolError(f"object is too large at {location}")
            for key, child in item.items():
                if not isinstance(key, str) or len(key) > 4096 or "\x00" in key:
                    raise ToolError(f"invalid object key at {location}")
                stack.append((child, depth + 1, f"{location}.{key}"))
            continue
        if isinstance(item, list):
            if len(item) > 100_000:
                raise ToolError(f"array is too large at {location}")
            for index, child in enumerate(item):
                stack.append((child, depth + 1, f"{location}[{index}]"))
            continue
        raise ToolError(f"unsupported data type at {location}: {type(item).__name__}")


def _loads_json(raw: bytes, *, maximum_bytes: int = MAX_JSON_BYTES) -> object:
    if len(raw) > maximum_bytes:
        raise ToolError("JSON document exceeds size limit")
    try:
        text = raw.decode("utf-8-sig")
        value = json.loads(
            text,
            object_pairs_hook=_object_pairs,
            parse_constant=_reject_constant,
        )
    except UnicodeDecodeError as error:
        raise ToolError("JSON document is not UTF-8") from error
    except json.JSONDecodeError as error:
        raise ToolError(f"malformed JSON at line {error.lineno}, column {error.colno}") from error
    _validate_data(value)
    return value


def _canonical(value: object) -> bytes:
    _validate_data(value)
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ToolError("value is outside the canonical JSON domain") from error


def _tagged_digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path, *, maximum_bytes: int = MAX_FILE_BYTES) -> str:
    if path.is_symlink() or not path.is_file():
        raise ToolError(f"expected a regular file: {path.name}")
    size = path.stat().st_size
    if size < 0 or size > maximum_bytes:
        raise ToolError(f"file exceeds size limit: {path.name}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _hash_canonical(value: object) -> str:
    return _tagged_digest(_canonical(value))


def _exact_fields(value: dict[str, object], allowed: set[str], context: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ToolError(f"unknown {context} field(s): {', '.join(sorted(unknown))}")


def _mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ToolError(f"{context} must be an object")
    return value


def _string(value: object, context: str, *, maximum: int = 4096) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum or "\x00" in value:
        raise ToolError(f"{context} must be a non-empty bounded string")
    return value


def _finite(value: object, context: str, *, default: float | None = None) -> float:
    if value is None and default is not None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ToolError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ToolError(f"{context} must be finite")
    return result


def _safe_relative(value: object, context: str = "path") -> str:
    text = _string(value, context)
    normalized = text.replace("\\", "/")
    pure = PurePosixPath(normalized)
    if (
        normalized.startswith("/")
        or Path(text).is_absolute()
        or Path(text).drive
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ToolError(f"unsafe relative {context}: {text}")
    return pure.as_posix()


def _inside(root: Path, candidate: Path) -> bool:
    try:
        return os.path.commonpath((str(root.resolve()), str(candidate.resolve()))) == str(root.resolve())
    except ValueError:
        return False


def _safe_file(root: Path, relative: object, *, maximum_bytes: int = MAX_FILE_BYTES) -> Path:
    safe = _safe_relative(relative, "bundle artifact path")
    current = root
    for part in PurePosixPath(safe).parts:
        current = current / part
        if current.is_symlink():
            raise ToolError(f"bundle artifacts may not traverse symlinks: {safe}")
    if not current.is_file() or not _inside(root, current):
        raise ToolError(f"bundle artifact is not a regular in-bundle file: {safe}")
    if current.stat().st_size > maximum_bytes:
        raise ToolError(f"bundle artifact exceeds size limit: {safe}")
    return current


def _regular_directory(path: Path, context: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise ToolError(f"{context} must be a regular directory")
    return path.resolve()


def _tensor_content_hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(device="cpu").contiguous()
    raw = value.view(torch.uint8).numpy().tobytes()
    prefix = f"{value.dtype}|{tuple(value.shape)}|".encode("utf-8")
    return _tagged_digest(prefix + raw)


def _load_tensor_file(path: Path) -> dict[str, torch.Tensor]:
    if path.is_symlink() or not path.is_file():
        raise ToolError("SafeTensors input must be a regular file")
    size = path.stat().st_size
    if size <= 0 or size > MAX_FILE_BYTES:
        raise ToolError("SafeTensors input has an invalid size")
    result: dict[str, torch.Tensor] = {}
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = sorted(handle.keys())
            if not keys or len(keys) > MAX_TENSORS:
                raise ToolError("SafeTensors key count is outside limits")
            for key in keys:
                if not key or len(key) > 2048 or "\x00" in key:
                    raise ToolError("SafeTensors contains an invalid tensor key")
                tensor = handle.get_tensor(key)
                if tensor.numel() > MAX_TENSOR_ELEMENTS:
                    raise ToolError(f"tensor exceeds element limit: {key}")
                if tensor.layout != torch.strided:
                    raise ToolError(f"unsupported tensor layout: {key}")
                result[key] = tensor.clone()
    except ToolError:
        raise
    except Exception as error:
        raise ToolError(f"invalid SafeTensors data: {type(error).__name__}") from error
    return result


def _checkpoint_file(checkpoint: Path) -> tuple[Path, Path | None]:
    if checkpoint.is_symlink():
        raise ToolError("checkpoint may not be a symlink")
    if checkpoint.is_file():
        if checkpoint.suffix != ".safetensors":
            raise ToolError("checkpoint file must use SafeTensors")
        return checkpoint.resolve(), None
    root = _regular_directory(checkpoint, "checkpoint")
    if (root / "model.safetensors.index.json").exists():
        raise UnsupportedError("standalone R1 scripts support only unsharded checkpoints")
    candidates = sorted(root.glob("*.safetensors"))
    if len(candidates) != 1:
        raise UnsupportedError("checkpoint directory must contain exactly one unsharded SafeTensors file")
    if candidates[0].is_symlink():
        raise ToolError("checkpoint tensor file may not be a symlink")
    return candidates[0].resolve(), root


def _checkpoint_tensors(checkpoint: Path) -> tuple[dict[str, torch.Tensor], str, Path, Path | None]:
    tensor_file, root = _checkpoint_file(checkpoint)
    tensors = _load_tensor_file(tensor_file)
    hashes = {key: _tensor_content_hash(value) for key, value in sorted(tensors.items())}
    fingerprint = _hash_canonical({"schema_version": 1, "tensor_hashes": hashes})
    return tensors, fingerprint, tensor_file, root


def _validate_manifest(value: object) -> dict[str, object]:
    manifest = _mapping(value, "patch manifest")
    allowed = {
        "artifact_hashes",
        "base_signature",
        "compiler_configuration",
        "delta_representation",
        "merged_from",
        "name",
        "parent_patches",
        "patch_id",
        "preserves",
        "provides",
        "rebased_from",
        "requires",
        "schema_version",
        "source_diff_bundle",
        "target_module_schema_hash",
        "tool_version",
        "verification_policy_hash",
    }
    _exact_fields(manifest, allowed, "patch manifest")
    if manifest.get("schema_version") != 1:
        raise UnsupportedError("unsupported Patch Bundle manifest version")
    if manifest.get("delta_representation") != "additive_low_rank_sparse_v1":
        raise UnsupportedError("unsupported delta representation")
    for field in ("patch_id", "target_module_schema_hash"):
        digest = _string(manifest.get(field), field)
        if DIGEST_RE.fullmatch(digest) is None:
            raise ToolError(f"invalid digest in {field}")
    _string(manifest.get("tool_version"), "tool_version", maximum=256)
    _string(manifest.get("name"), "name", maximum=256)
    for field in ("provides", "preserves", "requires", "parent_patches", "merged_from"):
        items = manifest.get(field)
        if not isinstance(items, list) or not all(isinstance(item, str) for item in items):
            raise ToolError(f"{field} must be a string array")
        if items != sorted(set(items)):
            raise ToolError(f"{field} must be sorted and unique")
    for field in ("rebased_from", "source_diff_bundle", "verification_policy_hash"):
        item = manifest.get(field)
        if item is not None and not isinstance(item, str):
            raise ToolError(f"{field} must be a string or null")
    base = _mapping(manifest.get("base_signature"), "base_signature")
    _exact_fields(
        base,
        {
            "schema_version",
            "adapter_id",
            "architecture_hash",
            "state_schema_hash",
            "checkpoint_hash",
            "tokenizer_hash",
            "chat_template_hash",
            "generation_config_hash",
            "configuration_hash",
        },
        "base signature",
    )
    checkpoint_hash = base.get("checkpoint_hash")
    if not isinstance(checkpoint_hash, str) or DIGEST_RE.fullmatch(checkpoint_hash) is None:
        raise ToolError("base_signature.checkpoint_hash is required")
    configuration_hash = base.get("configuration_hash")
    if configuration_hash is not None and (
        not isinstance(configuration_hash, str)
        or DIGEST_RE.fullmatch(configuration_hash) is None
    ):
        raise ToolError("base_signature.configuration_hash must be a SHA-256 digest")
    _mapping(manifest.get("compiler_configuration"), "compiler_configuration")
    artifacts = _mapping(manifest.get("artifact_hashes"), "artifact_hashes")
    if len(artifacts) > MAX_TENSORS:
        raise ToolError("artifact hash map exceeds limit")
    for relative, digest in artifacts.items():
        _safe_relative(relative, "artifact path")
        if not isinstance(digest, str) or DIGEST_RE.fullmatch(digest) is None:
            raise ToolError(f"invalid artifact digest: {relative}")
    if not {"delta-program.json", "tensors.safetensors"}.issubset(artifacts):
        raise ToolError("bundle omits mandatory delta artifacts")
    identity_payload = {key: item for key, item in manifest.items() if key != "patch_id"}
    identity_payload["artifact_hashes"] = {
        path: digest
        for path, digest in artifacts.items()
        if path in {"delta-program.json", "tensors.safetensors"}
        or path.startswith("contracts/")
    }
    observed = _hash_canonical(identity_payload)
    if manifest["patch_id"] != observed:
        raise ToolError("patch identity does not match canonical manifest content")
    if manifest["patch_id"] != EXPECTED_PATCH_ID:
        raise ToolError("bundle patch ID differs from the ID pinned in this script")
    evidence_id = _hash_canonical(
        {
            "artifact_hashes": {
                path: digest
                for path, digest in sorted(artifacts.items())
                if path not in {"apply_patch.py", "verify_patch.py", "certificate.json"}
            },
            "patch_id": manifest["patch_id"],
            "schema_version": 1,
        }
    )
    if evidence_id != EXPECTED_EVIDENCE_ID:
        raise ToolError(
            "bundle evidence identity differs from the identity pinned in this script"
        )
    return manifest


def _load_bundle(path: Path) -> tuple[Path, dict[str, object], dict[str, object], dict[str, torch.Tensor]]:
    root = _regular_directory(path, "patch bundle")
    manifest_path = _safe_file(root, "manifest.json", maximum_bytes=MAX_JSON_BYTES)
    manifest = _validate_manifest(_loads_json(manifest_path.read_bytes()))
    artifacts = _mapping(manifest["artifact_hashes"], "artifact_hashes")
    for relative in sorted(artifacts):
        expected = artifacts[relative]
        actual = _sha256_file(_safe_file(root, relative))
        if actual != expected:
            raise ToolError(f"patch artifact hash mismatch: {relative}")
    program_value = _loads_json(
        _safe_file(root, "delta-program.json", maximum_bytes=MAX_JSON_BYTES).read_bytes()
    )
    program = _validate_program(program_value)
    patch_tensors = _load_tensor_file(_safe_file(root, "tensors.safetensors"))
    _validate_program_tensors(program, patch_tensors)
    return root, manifest, program, patch_tensors


def _valid_reference(value: object, context: str) -> str:
    text = _string(value, context, maximum=2048)
    normalized = text.replace("\\", "/")
    if normalized.startswith("/") or ".." in PurePosixPath(normalized).parts:
        raise ToolError(f"unsafe tensor reference in {context}")
    return text


def _validate_op(value: object, *, depth: int = 0) -> dict[str, object]:
    if depth > MAX_DEPTH:
        raise ToolError("delta expression exceeds depth limit")
    operation = _mapping(value, "delta operation")
    kind = operation.get("op")
    if kind == "low_rank_matrix":
        _exact_fields(operation, {"op", "left", "right", "scale"}, "low-rank operation")
        _valid_reference(operation.get("left"), "low-rank left factor")
        _valid_reference(operation.get("right"), "low-rank right factor")
        _finite(operation.get("scale", 1.0), "low-rank scale")
    elif kind == "sparse_matrix":
        _exact_fields(
            operation,
            {"op", "indices", "values", "shape", "scale"},
            "sparse operation",
        )
        _valid_reference(operation.get("indices"), "sparse indices")
        _valid_reference(operation.get("values"), "sparse values")
        shape = operation.get("shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in shape)
            or shape[0] * shape[1] > MAX_DELTA_ELEMENTS
        ):
            raise ToolError("sparse matrix shape is invalid or exceeds limits")
        _finite(operation.get("scale", 1.0), "sparse scale")
    elif kind == "vector":
        _exact_fields(operation, {"op", "tensor", "scale"}, "vector operation")
        _valid_reference(operation.get("tensor"), "vector tensor")
        _finite(operation.get("scale", 1.0), "vector scale")
    elif kind == "alias":
        _exact_fields(operation, {"op", "target"}, "alias operation")
        _valid_reference(operation.get("target"), "alias target")
    elif kind == "sum":
        _exact_fields(operation, {"op", "terms"}, "sum operation")
        terms = operation.get("terms")
        if not isinstance(terms, list) or not terms or len(terms) > MAX_SUM_TERMS:
            raise ToolError("sum must have a bounded non-empty term list")
        for term in terms:
            _validate_op(term, depth=depth + 1)
    else:
        raise UnsupportedError(f"unknown delta operation: {kind!r}")
    return operation


def _validate_program(value: object) -> dict[str, object]:
    program = _mapping(value, "delta program")
    _exact_fields(program, {"schema_version", "targets"}, "delta program")
    if program.get("schema_version") != 1:
        raise UnsupportedError("unsupported Delta Program version")
    targets = _mapping(program.get("targets"), "delta targets")
    if not targets or len(targets) > MAX_TARGETS:
        raise ToolError("delta target count is outside limits")
    for name in sorted(targets):
        _valid_reference(name, "delta target")
        _validate_op(targets[name])
    return program


def _is_float(tensor: torch.Tensor) -> bool:
    return tensor.is_floating_point()


def _materializer(
    program: dict[str, object], patch_tensors: dict[str, torch.Tensor]
) -> Any:
    targets = _mapping(program["targets"], "delta targets")
    active: set[str] = set()
    cache: dict[str, torch.Tensor] = {}

    def tensor(name_value: object) -> torch.Tensor:
        name = _valid_reference(name_value, "patch tensor")
        if name not in patch_tensors:
            raise ToolError(f"missing patch tensor: {name}")
        return patch_tensors[name]

    def operation(value: object, depth: int = 0) -> torch.Tensor:
        if depth > MAX_DEPTH:
            raise ToolError("delta expression exceeds depth limit")
        item = _mapping(value, "delta operation")
        kind = item["op"]
        scale = _finite(item.get("scale", 1.0), "delta scale")
        if kind == "low_rank_matrix":
            left, right = tensor(item["left"]), tensor(item["right"])
            if (
                left.ndim != 2
                or right.ndim != 2
                or left.shape[1] <= 0
                or left.shape[1] != right.shape[0]
                or left.dtype != right.dtype
                or not _is_float(left)
            ):
                raise ToolError("invalid low-rank factor shapes or dtypes")
            return (left @ right) * scale
        if kind == "sparse_matrix":
            indices, values = tensor(item["indices"]), tensor(item["values"])
            shape = item["shape"]
            assert isinstance(shape, list)
            if (
                indices.ndim != 2
                or indices.shape[1] != 2
                or indices.dtype not in {torch.int32, torch.int64}
                or values.ndim != 1
                or values.shape[0] != indices.shape[0]
                or not _is_float(values)
            ):
                raise ToolError("invalid sparse matrix indices or values")
            cpu_indices = indices.detach().cpu().to(torch.int64)
            if cpu_indices.numel():
                if bool((cpu_indices < 0).any()):
                    raise ToolError("sparse indices may not be negative")
                if bool((cpu_indices[:, 0] >= shape[0]).any()) or bool(
                    (cpu_indices[:, 1] >= shape[1]).any()
                ):
                    raise ToolError("sparse index is out of bounds")
                flat = cpu_indices[:, 0] * shape[1] + cpu_indices[:, 1]
                if flat.numel() > 1 and not bool(torch.all(flat[1:] > flat[:-1])):
                    raise ToolError("sparse indices must be strictly sorted and unique")
            result = torch.zeros(tuple(shape), dtype=values.dtype, device=values.device)
            if cpu_indices.numel():
                result[cpu_indices[:, 0], cpu_indices[:, 1]] = values * scale
            return result
        if kind == "vector":
            value_tensor = tensor(item["tensor"])
            if value_tensor.ndim != 1 or not _is_float(value_tensor):
                raise ToolError("vector delta must be a rank-one floating tensor")
            return value_tensor * scale
        if kind == "alias":
            return resolve(_string(item["target"], "alias target"))
        if kind == "sum":
            terms = item["terms"]
            assert isinstance(terms, list)
            values = [operation(term, depth + 1) for term in terms]
            if any(value.shape != values[0].shape or value.dtype != values[0].dtype for value in values[1:]):
                raise ToolError("sum terms must have equal shapes and dtypes")
            result = values[0]
            for value_tensor in values[1:]:
                result = result + value_tensor
            return result
        raise UnsupportedError(f"unknown delta operation: {kind!r}")

    def resolve(name: str) -> torch.Tensor:
        if name in cache:
            return cache[name]
        if name in active:
            raise ToolError(f"delta alias cycle at {name}")
        if name not in targets:
            raise ToolError(f"delta alias refers to unknown target: {name}")
        active.add(name)
        try:
            result = operation(targets[name])
            cache[name] = result
            return result
        finally:
            active.remove(name)

    return resolve


def _validate_program_tensors(
    program: dict[str, object], patch_tensors: dict[str, torch.Tensor]
) -> None:
    targets = _mapping(program["targets"], "delta targets")
    resolve = _materializer(program, patch_tensors)
    for name in sorted(targets):
        delta = resolve(name)
        if delta.numel() > MAX_DELTA_ELEMENTS:
            raise ToolError(f"materialized delta exceeds element limit: {name}")


def _apply_program(
    state: dict[str, torch.Tensor],
    program: dict[str, object],
    patch_tensors: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    targets = _mapping(program["targets"], "delta targets")
    missing = set(targets) - set(state)
    if missing:
        raise ToolError(f"checkpoint lacks patch target(s): {', '.join(sorted(missing))}")
    resolve = _materializer(program, patch_tensors)
    result: dict[str, torch.Tensor] = {}
    for name in sorted(state):
        base = state[name]
        if name not in targets:
            result[name] = base.clone()
            continue
        delta = resolve(name)
        if delta.shape != base.shape:
            raise ToolError(f"base/delta shape mismatch for {name}")
        if delta.dtype != base.dtype:
            raise ToolError(f"base/delta dtype mismatch for {name}")
        if not _is_float(base):
            raise ToolError(f"refusing to patch a non-floating tensor: {name}")
        result[name] = base + delta
    return result


def _snapshot_source(tensor_file: Path, root: Path | None) -> dict[str, str]:
    paths = [tensor_file]
    if root is not None:
        paths.extend(
            root / name
            for name in sorted(AUXILIARY_FILES)
            if (root / name).exists()
        )
    result: dict[str, str] = {}
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ToolError(f"checkpoint input may not be a symlink: {path.name}")
        result[path.name] = _sha256_file(path, maximum_bytes=MAX_AUXILIARY_BYTES if path != tensor_file else MAX_FILE_BYTES)
    return result


def _safe_output_parent(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    current = target.parent
    while True:
        if current.is_symlink():
            raise ToolError("output path may not traverse a symlink")
        if current.parent == current:
            break
        current = current.parent
    return target.parent.resolve()


def _materialize(
    base_checkpoint: Path,
    output: Path,
    root: Path,
    manifest: dict[str, object],
    program: dict[str, object],
    patch_tensors: dict[str, torch.Tensor],
) -> dict[str, object]:
    if output.exists() or output.is_symlink():
        raise ToolError("output already exists")
    state, base_hash, tensor_file, source_root = _checkpoint_tensors(base_checkpoint)
    base_signature = _mapping(manifest["base_signature"], "base_signature")
    expected_base = base_signature["checkpoint_hash"]
    if base_hash != expected_base:
        raise ToolError(f"base checkpoint fingerprint mismatch: expected {expected_base}, observed {base_hash}")
    _assert_base_identity(base_checkpoint, base_signature, base_hash)
    source_resolved = base_checkpoint.resolve()
    output_resolved = output.parent.resolve() / output.name
    if output_resolved == source_resolved or (source_root is not None and _inside(source_root, output_resolved)):
        raise ToolError("output must not equal or be nested within the source checkpoint")
    before = _snapshot_source(tensor_file, source_root)
    patched = _apply_program(state, program, patch_tensors)
    parent = _safe_output_parent(output)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=parent))
    try:
        output_tensor = temporary / "model.safetensors"
        save_file(
            {key: patched[key].detach().cpu().contiguous() for key in sorted(patched)},
            output_tensor,
            # Transformers requires the reserved ``format`` metadata value to
            # identify its framework. Keep ModelPact provenance in a separate
            # data-only metadata field so the independently materialized
            # checkpoint remains loadable by the reviewed local HF adapter.
            metadata={"format": "pt", "modelpact_format": "modelpact-materialized-v1"},
        )
        copied: list[str] = []
        if source_root is not None:
            for name in sorted(AUXILIARY_FILES):
                source = source_root / name
                if not source.exists():
                    continue
                if source.is_symlink() or not source.is_file():
                    raise ToolError(f"checkpoint auxiliary input may not be a symlink: {name}")
                if source.stat().st_size > MAX_AUXILIARY_BYTES:
                    raise ToolError(f"checkpoint auxiliary file exceeds size limit: {name}")
                shutil.copyfile(source, temporary / name)
                copied.append(name)
        output_hashes = {key: _tensor_content_hash(value) for key, value in sorted(patched.items())}
        output_checkpoint_hash = _hash_canonical(
            {"schema_version": 1, "tensor_hashes": output_hashes}
        )
        materialization = {
            "auxiliary_files": copied,
            "base_checkpoint_hash": base_hash,
            "output_checkpoint_hash": output_checkpoint_hash,
            "output_file": "model.safetensors",
            "output_tensor_hashes": output_hashes,
            "patch_ids": [manifest["patch_id"]],
            "resolved_delta_program_hash": _hash_canonical(program),
            "schema_version": 1,
            "source_file_hashes": before,
        }
        (temporary / "materialization-manifest.json").write_bytes(_canonical(materialization))
        after = _snapshot_source(tensor_file, source_root)
        if after != before:
            raise ToolError("source checkpoint changed during materialization")
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "applied_targets": sorted(_mapping(program["targets"], "delta targets")),
        "base_checkpoint_hash": base_hash,
        "command": "apply",
        "outcome": "PASS",
        "output": str(output.resolve()),
        "output_checkpoint_hash": output_checkpoint_hash,
        "patch_id": manifest["patch_id"],
        "schema_version": 1,
        "source_unchanged": True,
    }


@dataclass(frozen=True)
class GenerationPolicy:
    mode: str = "greedy"
    max_new_tokens: int = 128
    seed: int = 0
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float = 1.0
    stop_sequences: tuple[str, ...] = ()
    stop_on_eos: bool = True


def _load_contract(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    if len(raw) > MAX_CONTRACT_BYTES:
        raise ToolError(f"contract exceeds size limit: {path.name}")
    suffix = path.suffix.lower()
    if suffix == ".json":
        value = _loads_json(raw, maximum_bytes=MAX_CONTRACT_BYTES)
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml
            from yaml.tokens import AliasToken, AnchorToken, TagToken
        except ImportError as error:
            raise UnsupportedError("PyYAML is required to verify YAML contracts") from error
        try:
            text = raw.decode("utf-8-sig")
            for token in yaml.scan(text):
                if isinstance(token, (AliasToken, AnchorToken, TagToken)):
                    raise ToolError("YAML aliases, anchors, and explicit tags are forbidden")
            value = yaml.safe_load(text)
        except ToolError:
            raise
        except (UnicodeDecodeError, yaml.YAMLError) as error:
            raise ToolError(f"malformed YAML contract: {path.name}") from error
        _validate_data(value, maximum_depth=MAX_DEPTH)
    else:
        raise UnsupportedError(f"unsupported contract format: {path.suffix}")
    return _validate_contract(value)


ASSERTION_OPTIONS = {
    "token_log_probability": {"token_id", "token", "position", "minimum", "maximum"},
    "sequence_log_probability": {"sequence", "normalize", "minimum", "maximum"},
    "sequence_margin": {"preferred", "dispreferred", "minimum_margin"},
    "multiple_choice_margin": {"choices", "correct_choice", "minimum_margin"},
    "exact_match": {"expected", "case_sensitive", "minimum_pass_rate"},
    "normalized_exact_match": {"expected", "unicode_normalization", "minimum_pass_rate"},
    "regular_expression": {"pattern", "full_match", "case_sensitive", "minimum_pass_rate"},
    "json_parse": {"minimum_pass_rate"},
    "json_schema": {"schema", "schema_file", "minimum_pass_rate"},
    "free_generation_match": {
        "expected", "match_type", "pattern", "full_match", "case_sensitive", "minimum_pass_rate"
    },
    "reference_kl": {"temperature", "maximum_mean", "maximum_item", "maximum_quantile"},
    "base_kl": {"temperature", "maximum_mean", "maximum_item", "maximum_quantile"},
    "generation_length": {"unit", "minimum", "maximum"},
    "perplexity": {"maximum", "maximum_mean", "maximum_item"},
}


def _validate_assertion(value: object) -> dict[str, object]:
    assertion = _mapping(value, "verification assertion")
    kind = _string(assertion.get("type"), "assertion type", maximum=128)
    if kind not in ASSERTION_OPTIONS:
        raise UnsupportedError(f"unsupported assertion type: {kind}")
    _exact_fields(
        assertion,
        {"id", "type", "source"} | ASSERTION_OPTIONS[kind],
        f"{kind} assertion",
    )
    _string(assertion.get("id"), "assertion id", maximum=128)
    _safe_relative(assertion.get("source"), "assertion source")
    for field in ("minimum", "maximum", "minimum_pass_rate", "minimum_margin", "maximum_mean", "maximum_item", "temperature"):
        if field in assertion:
            _finite(assertion[field], f"assertion {field}")
    if "minimum_pass_rate" in assertion and not 0 <= float(assertion["minimum_pass_rate"]) <= 1:
        raise ToolError("minimum_pass_rate must be in [0, 1]")
    if "maximum_quantile" in assertion:
        quantile = _mapping(assertion["maximum_quantile"], "maximum_quantile")
        _exact_fields(quantile, {"q", "value"}, "maximum_quantile")
        q = _finite(quantile.get("q"), "maximum_quantile.q")
        limit = _finite(quantile.get("value"), "maximum_quantile.value")
        if not 0 < q <= 1 or limit < 0:
            raise ToolError("maximum_quantile has invalid bounds")
    return assertion


def _validate_contract(value: object) -> dict[str, object]:
    contract = _mapping(value, "behavior contract")
    allowed = {
        "schema_version", "id", "contract_version", "description", "model_requirements",
        "compile", "verify", "holdout", "statistics", "generation"
    }
    _exact_fields(contract, allowed, "behavior contract")
    if contract.get("schema_version") != 1:
        raise UnsupportedError("unsupported Behavior Contract version")
    _string(contract.get("id"), "contract id", maximum=128)
    version = contract.get("contract_version", 1)
    if isinstance(version, bool) or not isinstance(version, int) or version <= 0:
        raise ToolError("contract_version must be a positive integer")
    requirements = _mapping(contract.get("model_requirements", {}), "model_requirements")
    _exact_fields(
        requirements,
        {"tokenizer_hash", "base_signature", "architecture_hash", "state_schema_hash", "adapter_id", "output_semantics"},
        "model_requirements",
    )
    if requirements.get("output_semantics", "causal_lm") != "causal_lm":
        raise UnsupportedError("only causal_lm contract semantics are supported")
    compile_section = _mapping(contract.get("compile", {"objectives": []}), "compile")
    _exact_fields(compile_section, {"objectives"}, "compile")
    objectives = compile_section.get("objectives", [])
    if not isinstance(objectives, list) or len(objectives) > 10_000:
        raise ToolError("compile objectives are malformed")
    verify = _mapping(contract.get("verify"), "verify")
    _exact_fields(verify, {"targets", "guards"}, "verify")
    for group in ("targets", "guards"):
        assertions = verify.get(group, [])
        if not isinstance(assertions, list) or len(assertions) > 50_000:
            raise ToolError(f"verify.{group} is malformed")
        verify[group] = [_validate_assertion(item) for item in assertions]
    holdout = _mapping(contract.get("holdout", {"sealed": True, "unseal_policy": "final_candidate_only"}), "holdout")
    _exact_fields(holdout, {"sealed", "targets", "guards", "unseal_policy"}, "holdout")
    if not isinstance(holdout.get("sealed", True), bool):
        raise ToolError("holdout.sealed must be boolean")
    for field in ("targets", "guards"):
        if field in holdout:
            _safe_relative(holdout[field], f"holdout {field}")
    statistics = _mapping(contract.get("statistics", {}), "statistics")
    _exact_fields(statistics, {"confidence_level", "bootstrap_samples", "bootstrap_seed", "multiple_comparison"}, "statistics")
    generation = _mapping(contract.get("generation", {}), "generation")
    _exact_fields(generation, {"mode", "max_new_tokens", "temperature", "top_k", "top_p", "seeds", "stop_sequences"}, "generation")
    mode = generation.get("mode", "greedy")
    if mode not in {"greedy", "sample"}:
        raise UnsupportedError(f"unsupported generation mode: {mode}")
    maximum = generation.get("max_new_tokens", 128)
    if isinstance(maximum, bool) or not isinstance(maximum, int) or not 1 <= maximum <= 4096:
        raise ToolError("generation max_new_tokens is outside standalone limits")
    temperature = _finite(generation.get("temperature", 1.0), "generation temperature")
    top_p = _finite(generation.get("top_p", 1.0), "generation top_p")
    top_k = generation.get("top_k")
    if temperature <= 0 or not 0 < top_p <= 1:
        raise ToolError("generation temperature/top_p are outside limits")
    if top_k is not None and (
        isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 10_000_000
    ):
        raise ToolError("generation top_k is outside limits")
    stops = generation.get("stop_sequences", [])
    if not isinstance(stops, list) or len(stops) > 128 or any(
        not isinstance(stop, str) or not stop or len(stop) > 4096 or "\x00" in stop
        for stop in stops
    ):
        raise ToolError("generation stop sequences are invalid")
    seeds = generation.get("seeds", [0])
    if not isinstance(seeds, list) or not seeds or len(seeds) > 1024 or any(
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or seed >= 2**63 for seed in seeds
    ):
        raise ToolError("generation seeds are invalid")
    return contract


def _tokenizer_hash(checkpoint: Path) -> str:
    _, root = _checkpoint_file(checkpoint)
    records: list[dict[str, str]] = []
    if root is not None:
        tokenizer_names = {
            "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
            "added_tokens.json", "vocab.json", "merges.txt", "spiece.model", "tokenizer.model"
        }
        for name in sorted(tokenizer_names):
            path = root / name
            if path.exists():
                if path.is_symlink() or not path.is_file():
                    raise ToolError(f"tokenizer identity input may not be a symlink: {name}")
                records.append({"name": name, "sha256": _sha256_file(path, maximum_bytes=MAX_AUXILIARY_BYTES)})
    return _hash_canonical({"files": records, "schema_version": 1})


def _identity_json(checkpoint: Path, name: str, default: object) -> object:
    _, root = _checkpoint_file(checkpoint)
    if root is None:
        return default
    path = root / name
    if not path.exists():
        return default
    if path.is_symlink() or not path.is_file():
        raise ToolError(f"model identity input may not be a symlink: {name}")
    if path.stat().st_size > MAX_AUXILIARY_BYTES:
        raise ToolError(f"model identity input exceeds size limit: {name}")
    return _loads_json(path.read_bytes(), maximum_bytes=MAX_AUXILIARY_BYTES)


def _configuration_hash(checkpoint: Path) -> str:
    value = _identity_json(checkpoint, "config.json", {})
    configuration = _mapping(value, "model configuration")
    excluded = {"_name_or_path", "transformers_version", "torch_dtype"}
    canonical = {key: configuration[key] for key in sorted(configuration) if key not in excluded}
    return _hash_canonical(canonical)


def _chat_template_hash(checkpoint: Path) -> str:
    value = _identity_json(checkpoint, "tokenizer_config.json", {})
    configuration = _mapping(value, "tokenizer configuration")
    return _hash_canonical({"chat_template": configuration.get("chat_template")})


def _generation_config_hash(checkpoint: Path) -> str:
    return _hash_canonical(_identity_json(checkpoint, "generation_config.json", {}))


def _assert_base_identity(
    checkpoint: Path,
    base_signature: dict[str, object],
    checkpoint_hash: str,
) -> None:
    observed = {
        "checkpoint_hash": checkpoint_hash,
        "tokenizer_hash": _tokenizer_hash(checkpoint),
        "chat_template_hash": _chat_template_hash(checkpoint),
        "generation_config_hash": _generation_config_hash(checkpoint),
        "configuration_hash": _configuration_hash(checkpoint),
    }
    for name, actual in observed.items():
        expected = base_signature.get(name)
        if expected is None and name == "configuration_hash":
            # Legacy bundles did not expose this independently recomputable
            # component of architecture identity.
            continue
        if actual != expected:
            raise ToolError(
                f"base model identity mismatch for {name}: expected {expected}, observed {actual}"
            )


def _load_probe_rows(root: Path, relative: str, artifacts: dict[str, object]) -> list[dict[str, object]]:
    safe = _safe_relative(relative, "probe source")
    if safe not in artifacts:
        raise ToolError(f"probe source is not content-addressed by the patch manifest: {safe}")
    path = _safe_file(root, safe, maximum_bytes=MAX_PROBE_BYTES)
    rows: list[dict[str, object]] = []
    total = 0
    with path.open("rb") as stream:
        for line_number, raw in enumerate(stream, start=1):
            total += len(raw)
            if total > MAX_PROBE_BYTES or len(raw) > MAX_LINE_BYTES:
                raise ToolError(f"probe source exceeds limits: {safe}")
            if not raw.strip():
                continue
            value = _loads_json(raw, maximum_bytes=MAX_LINE_BYTES)
            row = _mapping(value, f"probe record {line_number}")
            prompt = row.get("prompt", row.get("input"))
            _string(prompt, f"probe prompt at line {line_number}", maximum=MAX_TEXT)
            rows.append(row)
            if len(rows) > MAX_RECORDS:
                raise ToolError(f"probe record count exceeds limit: {safe}")
    if not rows:
        raise InconclusiveError(f"probe source contains no records: {safe}")
    return rows


def _trusted_adapter(specification: str, *, patch_root: Path) -> object:
    module_name, separator, attribute_name = specification.partition(":")
    identifier = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$")
    if (
        not separator
        or identifier.fullmatch(module_name) is None
        or re.fullmatch(r"[A-Za-z_]\w*", attribute_name) is None
    ):
        raise ToolError("trusted adapter must be specified as module:attribute")
    module_path = patch_root.joinpath(*module_name.split("."))
    bundle_candidates = [module_path]
    if module_path.parent.is_dir():
        bundle_candidates.extend(module_path.parent.glob(module_path.name + ".*"))
    if any(candidate.exists() or candidate.is_symlink() for candidate in bundle_candidates):
        raise ToolError("trusted adapter code must be located outside the untrusted patch bundle")
    original_path = list(sys.path)
    safe_path: list[str] = []
    for item in original_path:
        candidate = Path(item or os.getcwd())
        if _inside(patch_root, candidate):
            continue
        safe_path.append(item)
    try:
        sys.path[:] = safe_path
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            raise ModuleNotFoundError(module_name)
        origins = []
        if spec.origin not in {None, "built-in", "frozen"}:
            origins.append(Path(spec.origin))
        if spec.submodule_search_locations is not None:
            origins.extend(Path(item) for item in spec.submodule_search_locations)
        if any(_inside(patch_root, origin) for origin in origins):
            raise ToolError("trusted adapter code must be located outside the untrusted patch bundle")
        module = importlib.import_module(module_name)
        candidate = getattr(module, attribute_name)
        adapter = candidate() if isinstance(candidate, type) else candidate
    except ToolError:
        raise
    except Exception as error:
        raise InconclusiveError(f"trusted adapter could not be loaded: {type(error).__name__}: {error}") from error
    finally:
        sys.path[:] = original_path
    for name in ("load", "tokenizer", "generate", "forward_logits"):
        if not callable(getattr(adapter, name, None)):
            raise ToolError(f"trusted adapter does not implement required method: {name}")
    return adapter


def _selected_adapter(
    adapter_specification: str | None,
    adapter_kind: str | None,
    *,
    patch_root: Path,
) -> object:
    if adapter_kind == "tiny":
        return TinyModelAdapter()
    if adapter_kind == "huggingface":
        return StandaloneHuggingFaceModelAdapter()
    if adapter_kind is not None:
        raise UnsupportedError(f"unsupported built-in adapter kind: {adapter_kind}")
    if adapter_specification is None:
        raise ToolError("verification requires a built-in adapter kind or trusted local adapter")
    return _trusted_adapter(adapter_specification, patch_root=patch_root)


def _row_value(row: dict[str, object], assertion: dict[str, object], name: str, default: object = None) -> object:
    values = row.get("values")
    if isinstance(values, dict) and name in values:
        return values[name]
    if name in row:
        return row[name]
    return assertion.get(name, default)


def _prompt(row: dict[str, object]) -> str:
    value = row.get("prompt", row.get("input"))
    assert isinstance(value, str)
    return value


def _sample_id(row: dict[str, object], index: int, seed: int | None = None) -> str:
    value = row.get("id", row.get("sample_id", f"sample-{index:06d}"))
    result = _string(value, "sample id", maximum=4096)
    return f"{result}@seed-{seed}" if seed is not None else result


def _batch(adapter: object, text: str) -> object:
    tokenizer = adapter.tokenizer()
    batch = tokenizer.batch([text])
    return batch


def _generate(adapter: object, model: object, prompt: str, generation: dict[str, object], seed: int) -> tuple[str, tuple[int, ...]]:
    temperature = _finite(generation.get("temperature", 1.0), "generation temperature")
    policy = GenerationPolicy(
        mode=str(generation.get("mode", "greedy")),
        max_new_tokens=int(generation.get("max_new_tokens", 128)),
        seed=seed,
        temperature=temperature,
        top_k=generation.get("top_k") if isinstance(generation.get("top_k"), int) else None,
        top_p=float(generation.get("top_p", 1.0)),
        stop_sequences=tuple(generation.get("stop_sequences", [])),
    )
    try:
        with torch.inference_mode():
            samples = adapter.generate(model, _batch(adapter, prompt), policy)
    except Exception as error:
        raise InconclusiveError(f"adapter generation failed: {type(error).__name__}: {error}") from error
    if not isinstance(samples, (list, tuple)) or len(samples) != 1:
        raise InconclusiveError("adapter generation must return one sample for one prompt")
    sample = samples[0]
    text = getattr(sample, "text", None)
    token_ids = getattr(sample, "token_ids", None)
    if not isinstance(text, str) or len(text) > MAX_TEXT or "\x00" in text:
        raise InconclusiveError("adapter returned invalid generated text")
    if not isinstance(token_ids, (list, tuple)) or any(
        isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in token_ids
    ):
        raise InconclusiveError("adapter returned invalid generated token IDs")
    return text, tuple(token_ids)


def _safe_regex(pattern_value: object, *, case_sensitive: bool) -> re.Pattern[str]:
    pattern = _string(pattern_value, "regular expression", maximum=4096)
    if "(?" in pattern or re.search(r"\\[1-9]", pattern) or re.search(r"(?:\*|\+|\{[^}]+\})\s*(?:\*|\+|\{)", pattern):
        raise UnsupportedError("regular expression uses unsupported high-risk constructs")
    try:
        return re.compile(pattern, 0 if case_sensitive else re.IGNORECASE)
    except re.error as error:
        raise ToolError(f"invalid regular expression: {error}") from error


def _normalized(text: str, *, case_sensitive: bool = False) -> str:
    result = " ".join(unicodedata.normalize("NFKC", text).split())
    return result if case_sensitive else result.casefold()


def _strict_json_text(text: str) -> object:
    return _loads_json(text.encode("utf-8"), maximum_bytes=2 * 1024 * 1024)


def _schema_errors(instance: object, schema_value: object, path: str = "$", depth: int = 0) -> list[str]:
    if depth > 64:
        raise UnsupportedError("JSON schema evaluation exceeds depth limit")
    schema = _mapping(schema_value, "JSON schema")
    supported = {
        "$schema", "title", "description", "type", "enum", "const", "properties", "required",
        "additionalProperties", "items", "minItems", "maxItems", "uniqueItems", "minLength",
        "maxLength", "pattern", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"
    }
    unknown = set(schema) - supported
    if unknown:
        raise UnsupportedError(f"unsupported JSON schema keyword(s): {', '.join(sorted(unknown))}")
    errors: list[str] = []
    expected_type = schema.get("type")
    checks = {
        "object": lambda value: isinstance(value, dict),
        "array": lambda value: isinstance(value, list),
        "string": lambda value: isinstance(value, str),
        "number": lambda value: isinstance(value, (int, float)) and not isinstance(value, bool),
        "integer": lambda value: isinstance(value, int) and not isinstance(value, bool),
        "boolean": lambda value: isinstance(value, bool),
        "null": lambda value: value is None,
    }
    if expected_type is not None:
        if expected_type not in checks:
            raise UnsupportedError(f"unsupported JSON schema type: {expected_type}")
        if not checks[expected_type](instance):
            return [f"{path}: expected {expected_type}"]
    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: value is not in enum")
    if "const" in schema and instance != schema["const"]:
        errors.append(f"{path}: value differs from const")
    if isinstance(instance, dict):
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            raise ToolError("JSON schema properties must be an object")
        required = schema.get("required", [])
        if not isinstance(required, list) or not all(isinstance(key, str) for key in required):
            raise ToolError("JSON schema required must be a string array")
        for key in required:
            if key not in instance:
                errors.append(f"{path}: missing required property {key!r}")
        for key, child in properties.items():
            if key in instance:
                errors.extend(_schema_errors(instance[key], child, f"{path}.{key}", depth + 1))
        additional = schema.get("additionalProperties", True)
        for key in instance:
            if key in properties:
                continue
            if additional is False:
                errors.append(f"{path}: additional property {key!r} is forbidden")
            elif isinstance(additional, dict):
                errors.extend(_schema_errors(instance[key], additional, f"{path}.{key}", depth + 1))
            elif additional is not True:
                raise ToolError("additionalProperties must be boolean or a schema")
    if isinstance(instance, list):
        if isinstance(schema.get("minItems"), int) and len(instance) < int(schema["minItems"]):
            errors.append(f"{path}: has fewer than minItems")
        if isinstance(schema.get("maxItems"), int) and len(instance) > int(schema["maxItems"]):
            errors.append(f"{path}: has more than maxItems")
        if schema.get("uniqueItems") is True:
            encoded = [_canonical(item) for item in instance]
            if len(encoded) != len(set(encoded)):
                errors.append(f"{path}: items are not unique")
        if "items" in schema:
            for index, child in enumerate(instance):
                errors.extend(_schema_errors(child, schema["items"], f"{path}[{index}]", depth + 1))
    if isinstance(instance, str):
        if isinstance(schema.get("minLength"), int) and len(instance) < int(schema["minLength"]):
            errors.append(f"{path}: shorter than minLength")
        if isinstance(schema.get("maxLength"), int) and len(instance) > int(schema["maxLength"]):
            errors.append(f"{path}: longer than maxLength")
        if "pattern" in schema and _safe_regex(schema["pattern"], case_sensitive=True).search(instance) is None:
            errors.append(f"{path}: does not match pattern")
    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        numeric = float(instance)
        comparisons = {
            "minimum": lambda bound: numeric >= bound,
            "maximum": lambda bound: numeric <= bound,
            "exclusiveMinimum": lambda bound: numeric > bound,
            "exclusiveMaximum": lambda bound: numeric < bound,
        }
        for name, predicate in comparisons.items():
            if name in schema and not predicate(_finite(schema[name], f"schema {name}")):
                errors.append(f"{path}: violates {name}")
    return errors


def _generated_result(
    kind: str,
    assertion: dict[str, object],
    row: dict[str, object],
    text: str,
    token_ids: tuple[int, ...],
    schema: object | None,
) -> tuple[bool, object, str | None]:
    case_sensitive = bool(assertion.get("case_sensitive", True))
    if kind == "exact_match":
        expected = _string(_row_value(row, assertion, "expected"), "expected output", maximum=MAX_TEXT)
        left, right = (text, expected) if case_sensitive else (text.casefold(), expected.casefold())
        return left == right, left == right, None
    if kind == "normalized_exact_match":
        expected = _string(_row_value(row, assertion, "expected"), "expected output", maximum=MAX_TEXT)
        passed = _normalized(text, case_sensitive=case_sensitive) == _normalized(expected, case_sensitive=case_sensitive)
        return passed, passed, None
    if kind == "regular_expression":
        pattern = _row_value(row, assertion, "pattern")
        compiled = _safe_regex(pattern, case_sensitive=case_sensitive)
        match = compiled.fullmatch(text) if assertion.get("full_match", False) else compiled.search(text)
        return match is not None, match is not None, None
    if kind in {"json_parse", "json_schema"}:
        try:
            parsed = _strict_json_text(text)
            errors = _schema_errors(parsed, schema) if kind == "json_schema" else []
            return not errors, not errors, errors[0] if errors else None
        except UnsupportedError:
            raise
        except ToolError as error:
            return False, False, str(error)
    if kind == "free_generation_match":
        match_type = assertion.get("match_type", "exact")
        if match_type == "regex":
            pattern = _row_value(row, assertion, "pattern", _row_value(row, assertion, "expected"))
            compiled = _safe_regex(pattern, case_sensitive=case_sensitive)
            passed = compiled.fullmatch(text) is not None if assertion.get("full_match", False) else compiled.search(text) is not None
        else:
            expected = _string(_row_value(row, assertion, "expected"), "expected output", maximum=MAX_TEXT)
            actual_cmp = text if case_sensitive else text.casefold()
            expected_cmp = expected if case_sensitive else expected.casefold()
            if match_type == "exact":
                passed = actual_cmp == expected_cmp
            elif match_type == "normalized":
                passed = _normalized(text, case_sensitive=case_sensitive) == _normalized(expected, case_sensitive=case_sensitive)
            elif match_type == "contains":
                passed = expected_cmp in actual_cmp
            else:
                raise UnsupportedError(f"unsupported free_generation match_type: {match_type}")
        return passed, passed, None
    if kind == "generation_length":
        unit = assertion.get("unit", "tokens")
        if unit == "tokens":
            value = float(len(token_ids))
        elif unit == "characters":
            value = float(len(text))
        elif unit == "words":
            value = float(len(text.split()))
        else:
            raise UnsupportedError(f"unsupported generation length unit: {unit}")
        minimum = assertion.get("minimum")
        maximum = assertion.get("maximum")
        passed = (minimum is None or value >= float(minimum)) and (maximum is None or value <= float(maximum))
        return passed, value, None
    raise UnsupportedError(f"unsupported generative assertion: {kind}")


def _input_ids(batch: object) -> torch.Tensor:
    value = getattr(batch, "input_ids", None)
    if value is None and isinstance(batch, dict):
        value = batch.get("input_ids")
    if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[0] != 1:
        raise InconclusiveError("adapter batch does not expose one-row input_ids")
    return value.to(torch.long)


def _logits(adapter: object, model: object, text: str) -> tuple[torch.Tensor, torch.Tensor]:
    batch = _batch(adapter, text)
    try:
        with torch.inference_mode():
            logits = adapter.forward_logits(model, batch)
    except Exception as error:
        raise InconclusiveError(f"adapter logits failed: {type(error).__name__}: {error}") from error
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3 or logits.shape[0] != 1:
        raise InconclusiveError("adapter logits must have shape [1, time, vocabulary]")
    if not bool(torch.isfinite(logits).all()):
        raise InconclusiveError("adapter logits contain non-finite values")
    return logits[0].detach().cpu(), _input_ids(batch)[0].detach().cpu()


def _sequence_score(adapter: object, model: object, prompt: str, sequence: str, normalize: bool = False) -> float:
    logits, ids = _logits(adapter, model, prompt + sequence)
    if ids.numel() < 2 or logits.shape[0] < ids.numel() - 1:
        raise InconclusiveError("adapter logits do not cover the scored sequence")
    scores = torch.log_softmax(logits[: ids.numel() - 1], dim=-1).gather(1, ids[1:, None]).squeeze(1)
    return float(scores.mean() if normalize else scores.sum())


def _continuous_pass(values: list[float], assertion: dict[str, object]) -> tuple[bool, float, float]:
    conditions: list[tuple[bool, float, float]] = []
    if "minimum" in assertion:
        observed, limit = min(values), float(assertion["minimum"])
        conditions.append((observed >= limit, observed, observed - limit))
    if "maximum" in assertion:
        observed, limit = max(values), float(assertion["maximum"])
        conditions.append((observed <= limit, observed, limit - observed))
    if "maximum_mean" in assertion:
        observed, limit = sum(values) / len(values), float(assertion["maximum_mean"])
        conditions.append((observed <= limit, observed, limit - observed))
    if "maximum_item" in assertion:
        observed, limit = max(values), float(assertion["maximum_item"])
        conditions.append((observed <= limit, observed, limit - observed))
    if "maximum_quantile" in assertion:
        spec = assertion["maximum_quantile"]
        assert isinstance(spec, dict)
        q, limit = float(spec["q"]), float(spec["value"])
        ordered = sorted(values)
        observed = ordered[max(0, math.ceil(q * len(ordered)) - 1)]
        conditions.append((observed <= limit, observed, limit - observed))
    if not conditions:
        raise UnsupportedError("continuous assertion has no supported threshold")
    worst = min(conditions, key=lambda item: item[2])
    return all(item[0] for item in conditions), worst[1], worst[2]


def _distribution_values(
    kind: str,
    assertion: dict[str, object],
    rows: list[dict[str, object]],
    adapter: object,
    model: object,
    base_model: object,
) -> tuple[list[float], list[dict[str, object]]]:
    values: list[float] = []
    prompts: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        prompt = _prompt(row)
        if kind == "sequence_log_probability":
            sequence = _string(_row_value(row, assertion, "sequence", ""), "sequence", maximum=MAX_TEXT)
            value = _sequence_score(adapter, model, prompt, sequence, bool(assertion.get("normalize", False)))
        elif kind == "perplexity":
            sequence = str(_row_value(row, assertion, "sequence", ""))
            log_probability = _sequence_score(adapter, model, prompt, sequence, True)
            value = math.exp(-log_probability)
        elif kind == "token_log_probability":
            logits, ids = _logits(adapter, model, prompt)
            position = int(_row_value(row, assertion, "position", -1))
            token_id_raw = _row_value(row, assertion, "token_id")
            if token_id_raw is None:
                target_index = position if position >= 0 else ids.numel() + position
                if not 1 <= target_index < ids.numel():
                    raise InconclusiveError("token probability target position is outside the sequence")
                logit_index, token_id = target_index - 1, int(ids[target_index])
            else:
                token_id = int(token_id_raw)
                logit_index = position if position >= 0 else logits.shape[0] + position
            if not 0 <= logit_index < logits.shape[0] or not 0 <= token_id < logits.shape[1]:
                raise InconclusiveError("token probability index is out of range")
            value = float(torch.log_softmax(logits[logit_index], dim=-1)[token_id])
        elif kind in {"sequence_margin", "multiple_choice_margin"}:
            if kind == "sequence_margin":
                correct = _string(_row_value(row, assertion, "preferred"), "preferred sequence", maximum=MAX_TEXT)
                alternatives = [_string(_row_value(row, assertion, "dispreferred"), "dispreferred sequence", maximum=MAX_TEXT)]
            else:
                correct = _string(_row_value(row, assertion, "correct_choice"), "correct choice", maximum=MAX_TEXT)
                choices = _row_value(row, assertion, "choices")
                if not isinstance(choices, list) or not all(isinstance(choice, str) for choice in choices):
                    raise UnsupportedError("multiple_choice_margin requires string choices")
                alternatives = [choice for choice in choices if choice != correct]
            if not alternatives:
                raise UnsupportedError("margin assertion requires an alternative")
            value = _sequence_score(adapter, model, prompt, correct) - max(
                _sequence_score(adapter, model, prompt, alternative) for alternative in alternatives
            )
        elif kind == "base_kl":
            student, _ = _logits(adapter, model, prompt)
            reference, _ = _logits(adapter, base_model, prompt)
            if student.shape != reference.shape:
                raise InconclusiveError("patched and base logits have different shapes")
            temperature = float(assertion.get("temperature", 1.0))
            student_log = torch.log_softmax(student / temperature, dim=-1)
            reference_log = torch.log_softmax(reference / temperature, dim=-1)
            value = float((reference_log.exp() * (reference_log - student_log)).sum(-1).mean() * temperature**2)
        elif kind == "reference_kl":
            student, _ = _logits(adapter, model, prompt)
            supplied = _row_value(row, assertion, "reference_logits")
            try:
                reference = torch.tensor(supplied, dtype=student.dtype)
            except Exception as error:
                raise UnsupportedError("reference_kl requires bounded reference_logits in probe data") from error
            if reference.numel() > MAX_DELTA_ELEMENTS or reference.shape != student.shape or not bool(torch.isfinite(reference).all()):
                raise ToolError("reference logits have invalid shape or values")
            temperature = float(assertion.get("temperature", 1.0))
            student_log = torch.log_softmax(student / temperature, dim=-1)
            reference_log = torch.log_softmax(reference / temperature, dim=-1)
            value = float((reference_log.exp() * (reference_log - student_log)).sum(-1).mean() * temperature**2)
        else:
            raise UnsupportedError(f"unsupported distribution assertion: {kind}")
        if not math.isfinite(value):
            raise InconclusiveError(f"non-finite metric for {kind}")
        threshold = float(assertion.get("minimum_margin", 0.0)) if kind in {"sequence_margin", "multiple_choice_margin"} else None
        passed = value >= threshold if threshold is not None else True
        prompts.append(
            {
                "margin": value - threshold if threshold is not None else None,
                "metric": kind,
                "outcome": "PASS" if passed else "FAIL",
                "output_hash": None,
                "prompt_hash": _tagged_digest(prompt.encode("utf-8")),
                "sample_id": _sample_id(row, index),
                "value": value,
            }
        )
        values.append(value)
    return values, prompts


GENERATION_ASSERTIONS = {
    "exact_match", "normalized_exact_match", "regular_expression", "json_parse",
    "json_schema", "free_generation_match", "generation_length"
}


def _assertion_result(
    root: Path,
    artifacts: dict[str, object],
    assertion: dict[str, object],
    source: str,
    adapter: object,
    model: object,
    base_model: object,
    generation: dict[str, object],
    role: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    rows = _load_probe_rows(root, source, artifacts)
    kind = str(assertion["type"])
    free_records: list[dict[str, object]] = []
    if kind in GENERATION_ASSERTIONS:
        schema: object | None = assertion.get("schema")
        if kind == "json_schema" and schema is None:
            schema_file = _safe_relative(assertion.get("schema_file"), "JSON schema file")
            if schema_file not in artifacts:
                raise ToolError("JSON schema file is not content-addressed by the patch manifest")
            schema = _loads_json(_safe_file(root, schema_file, maximum_bytes=MAX_CONTRACT_BYTES).read_bytes(), maximum_bytes=MAX_CONTRACT_BYTES)
        prompt_metrics: list[dict[str, object]] = []
        seeds = generation.get("seeds", [0])
        assert isinstance(seeds, list)
        for index, row in enumerate(rows):
            for seed in seeds:
                assert isinstance(seed, int)
                text, token_ids = _generate(adapter, model, _prompt(row), generation, seed)
                passed, value, message = _generated_result(kind, assertion, row, text, token_ids, schema)
                metric = {
                    "diagnostics": {"parser_error": message},
                    "metric": kind,
                    "outcome": "PASS" if passed else "FAIL",
                    "output_hash": _tagged_digest(text.encode("utf-8")),
                    "prompt_hash": _tagged_digest(_prompt(row).encode("utf-8")),
                    "sample_id": _sample_id(row, index, seed),
                    "token_count": len(token_ids),
                    "value": value,
                }
                prompt_metrics.append(metric)
                free_records.append(
                    {
                        "assertion_id": assertion["id"],
                        "decoding_policy": generation,
                        "output_hash": metric["output_hash"],
                        "parser_result": {"message": message, "passed": passed},
                        "prompt_hash": metric["prompt_hash"],
                        "role": role,
                        "sample_id": metric["sample_id"],
                        "seed": seed,
                        "token_ids_hash": _hash_canonical(list(token_ids)),
                    }
                )
        pass_rate = sum(item["outcome"] == "PASS" for item in prompt_metrics) / len(prompt_metrics)
        threshold = float(assertion.get("minimum_pass_rate", 1.0))
        passed = pass_rate >= threshold
        result = {
            "assertion_id": assertion["id"],
            "assertion_type": kind,
            "margin": pass_rate - threshold,
            "metric": "pass_rate",
            "outcome": "PASS" if passed else "FAIL",
            "prompt_metrics": prompt_metrics,
            "role": role,
            "source": source,
            "value": pass_rate,
        }
        return result, free_records
    values, prompt_metrics = _distribution_values(kind, assertion, rows, adapter, model, base_model)
    if kind in {"sequence_margin", "multiple_choice_margin"}:
        threshold = float(assertion.get("minimum_margin", 0.0))
        passed = min(values) >= threshold
        observed, margin = min(values), min(values) - threshold
    else:
        passed, observed, margin = _continuous_pass(values, assertion)
        for metric in prompt_metrics:
            value = float(metric["value"])
            item_passed = True
            if "minimum" in assertion:
                item_passed = item_passed and value >= float(assertion["minimum"])
            item_maximum = assertion.get("maximum_item", assertion.get("maximum"))
            if item_maximum is not None:
                item_passed = item_passed and value <= float(item_maximum)
            metric["outcome"] = "PASS" if item_passed else "FAIL"
    result = {
        "assertion_id": assertion["id"],
        "assertion_type": kind,
        "margin": margin,
        "metric": kind,
        "outcome": "PASS" if passed else "FAIL",
        "prompt_metrics": prompt_metrics,
        "role": role,
        "source": source,
        "value": observed,
    }
    return result, free_records


def _combine(outcomes: list[str]) -> str:
    if "FAIL" in outcomes:
        return "FAIL"
    if "UNSUPPORTED" in outcomes:
        return "UNSUPPORTED"
    if "INCONCLUSIVE" in outcomes:
        return "INCONCLUSIVE"
    return "PASS" if outcomes else "NOT_APPLICABLE"


def _contract_paths(root: Path, artifacts: dict[str, object], requested: list[str]) -> list[str]:
    if requested:
        paths = [_safe_relative(item, "contract path") for item in requested]
    else:
        paths = [
            item for item in sorted(artifacts)
            if (
                (
                    len(Path(item).parts) == 2
                    and Path(item).parts[0] == "contracts"
                    and (
                        Path(item).stem in {"target", "preservation"}
                        or Path(item).stem.startswith("contract-")
                    )
                )
                or (
                    len(Path(item).parts) == 4
                    and Path(item).parts[:2] == ("contracts", "parents")
                    and Path(item).name == "contract.json"
                )
            )
            and Path(item).suffix.lower() in {".yaml", ".yml", ".json"}
        ]
    if not paths:
        raise UnsupportedError("patch bundle contains no executable behavior contract")
    for item in paths:
        if item not in artifacts:
            raise ToolError(f"contract is not content-addressed by the patch manifest: {item}")
        _safe_file(root, item, maximum_bytes=MAX_CONTRACT_BYTES)
    return sorted(set(paths))


def _contract_resource_path(
    contract_path: str,
    relative: object,
    artifacts: dict[str, object],
) -> str:
    resource = _safe_relative(relative, "contract resource")
    # Retain compatibility with older bundles that explicitly used a
    # bundle-root-relative resource path. New bundles use contract-relative
    # paths, including contracts nested below contracts/parents/<hash>/.
    if resource in artifacts:
        return resource
    candidate = (Path(contract_path).parent / resource).as_posix()
    return _safe_relative(candidate, "contract-relative resource")


def _verify(
    base_checkpoint: Path,
    patch_path: Path,
    adapter_spec: str | None,
    adapter_kind: str | None,
    requested_contracts: list[str],
    include_holdout: bool,
    dtype_name: str,
) -> dict[str, object]:
    root, manifest, program, patch_tensors = _load_bundle(patch_path)
    artifacts = _mapping(manifest["artifact_hashes"], "artifact_hashes")
    _, base_hash, _, _ = _checkpoint_tensors(base_checkpoint)
    base_signature = _mapping(manifest["base_signature"], "base_signature")
    expected_base = base_signature["checkpoint_hash"]
    if base_hash != expected_base:
        raise ToolError(f"base checkpoint fingerprint mismatch: expected {expected_base}, observed {base_hash}")
    _assert_base_identity(base_checkpoint, base_signature, base_hash)
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    if dtype_name not in dtype_map:
        raise ToolError("unsupported execution dtype")
    adapter = _selected_adapter(adapter_spec, adapter_kind, patch_root=root)
    expected_adapter = _mapping(manifest["base_signature"], "base_signature").get("adapter_id")
    observed_adapter = getattr(adapter, "adapter_id", None)
    if expected_adapter is not None and observed_adapter != expected_adapter:
        raise ToolError(f"adapter identity mismatch: expected {expected_adapter}, observed {observed_adapter}")
    contract_paths = _contract_paths(root, artifacts, requested_contracts)
    contracts: list[tuple[str, dict[str, object]]] = []
    seen_contracts: set[str] = set()
    for path in contract_paths:
        contract = _load_contract(
            _safe_file(root, path, maximum_bytes=MAX_CONTRACT_BYTES)
        )
        identity = _hash_canonical(contract)
        if identity in seen_contracts:
            continue
        seen_contracts.add(identity)
        contracts.append((path, contract))
    tokenizer_hash = _tokenizer_hash(base_checkpoint)
    compatibility_errors: list[str] = []
    for path, contract in contracts:
        requirements = _mapping(contract.get("model_requirements", {}), "model_requirements")
        comparisons = {
            "adapter_id": observed_adapter,
            "tokenizer_hash": tokenizer_hash,
            "architecture_hash": base_signature.get("architecture_hash"),
            "state_schema_hash": base_signature.get("state_schema_hash"),
        }
        for name, observed in comparisons.items():
            expected = requirements.get(name)
            if expected is not None and expected != observed:
                compatibility_errors.append(f"{path}: {name} mismatch: expected {expected}, observed {observed}")
        expected_signature = requirements.get("base_signature")
        if expected_signature is not None:
            candidates = {base_hash, _hash_canonical(base_signature)}
            if expected_signature not in candidates:
                compatibility_errors.append(f"{path}: base_signature mismatch")
    if compatibility_errors:
        raise ToolError("; ".join(compatibility_errors))
    with tempfile.TemporaryDirectory(prefix="modelpact-independent-verify-") as temporary_name:
        patched_checkpoint = Path(temporary_name) / "patched"
        _materialize(base_checkpoint, patched_checkpoint, root, manifest, program, patch_tensors)
        try:
            model = adapter.load(str(patched_checkpoint), device="cpu", dtype=dtype_map[dtype_name])
            base_model = adapter.load(str(base_checkpoint), device="cpu", dtype=dtype_map[dtype_name])
            prepare = getattr(adapter, "prepare", None)
            if callable(prepare):
                prepare(model)
                prepare(base_model)
        except Exception as error:
            raise InconclusiveError(f"trusted adapter model loading failed: {type(error).__name__}: {error}") from error
        results: list[dict[str, object]] = []
        free_generation: list[dict[str, object]] = []
        warnings: list[str] = []
        target_count = 0
        guard_count = 0
        for path, contract in contracts:
            verify = _mapping(contract["verify"], "verify")
            generation = _mapping(contract.get("generation", {}), "generation")
            for group, role in (("targets", "target"), ("guards", "guard")):
                assertions = verify.get(group, [])
                assert isinstance(assertions, list)
                target_count += len(assertions) if group == "targets" else 0
                guard_count += len(assertions) if group == "guards" else 0
                for assertion_value in assertions:
                    assertion = _mapping(assertion_value, "assertion")
                    execution_assertion = dict(assertion)
                    execution_assertion["source"] = _contract_resource_path(
                        path,
                        assertion["source"],
                        artifacts,
                    )
                    if "schema_file" in assertion:
                        execution_assertion["schema_file"] = _contract_resource_path(
                            path,
                            assertion["schema_file"],
                            artifacts,
                        )
                    try:
                        result, generated = _assertion_result(
                            root,
                            artifacts,
                            execution_assertion,
                            str(execution_assertion["source"]),
                            adapter,
                            model,
                            base_model,
                            generation,
                            role,
                        )
                    except UnsupportedError as error:
                        result = {
                            "assertion_id": assertion["id"], "assertion_type": assertion["type"],
                            "margin": None, "message": str(error), "metric": assertion["type"],
                            "outcome": "UNSUPPORTED", "prompt_metrics": [], "role": role,
                            "source": assertion["source"], "value": None,
                        }
                        generated = []
                    except (InconclusiveError, ToolError) as error:
                        result = {
                            "assertion_id": assertion["id"], "assertion_type": assertion["type"],
                            "margin": None, "message": str(error), "metric": assertion["type"],
                            "outcome": "INCONCLUSIVE", "prompt_metrics": [], "role": role,
                            "source": assertion["source"], "value": None,
                        }
                        generated = []
                    results.append(result)
                    free_generation.extend(generated)
            holdout = _mapping(contract.get("holdout", {}), "holdout")
            if include_holdout:
                for group, role, source_key in (
                    ("targets", "holdout_target", "targets"),
                    ("guards", "holdout_guard", "guards"),
                ):
                    source_override = holdout.get(source_key)
                    if source_override is None:
                        continue
                    scoped_source = _contract_resource_path(path, source_override, artifacts)
                    assertions = verify.get(group, [])
                    assert isinstance(assertions, list)
                    for assertion_value in assertions:
                        assertion = _mapping(assertion_value, "assertion")
                        execution_assertion = dict(assertion)
                        execution_assertion["source"] = _contract_resource_path(
                            path,
                            assertion["source"],
                            artifacts,
                        )
                        if "schema_file" in assertion:
                            execution_assertion["schema_file"] = _contract_resource_path(
                                path,
                                assertion["schema_file"],
                                artifacts,
                            )
                        try:
                            result, generated = _assertion_result(
                                root, artifacts, execution_assertion, scoped_source, adapter, model,
                                base_model, generation, role,
                            )
                        except UnsupportedError as error:
                            result = {
                                "assertion_id": f"{assertion['id']}@{role}", "assertion_type": assertion["type"],
                                "margin": None, "message": str(error), "metric": assertion["type"],
                                "outcome": "UNSUPPORTED", "prompt_metrics": [], "role": role,
                                "source": scoped_source, "value": None,
                            }
                            generated = []
                        except (InconclusiveError, ToolError) as error:
                            result = {
                                "assertion_id": f"{assertion['id']}@{role}", "assertion_type": assertion["type"],
                                "margin": None, "message": str(error), "metric": assertion["type"],
                                "outcome": "INCONCLUSIVE", "prompt_metrics": [], "role": role,
                                "source": scoped_source, "value": None,
                            }
                            generated = []
                        else:
                            result["assertion_id"] = f"{assertion['id']}@{role}"
                        results.append(result)
                        free_generation.extend(generated)
            elif holdout.get("targets") is not None or holdout.get("guards") is not None:
                warnings.append(f"{path}: sealed holdout was not executed")
    outcomes = [str(result["outcome"]) for result in results]
    unsupported_claims: list[str] = []
    if target_count == 0:
        unsupported_claims.append("TARGET_ASSERTIONS_VERIFIED")
        outcomes.append("UNSUPPORTED")
    if guard_count == 0:
        unsupported_claims.append("PRESERVATION_ASSERTIONS_VERIFIED")
        outcomes.append("UNSUPPORTED")
    if warnings and not include_holdout:
        unsupported_claims.append("SEALED_HOLDOUT_VERIFIED")
    if any(str(result["assertion_type"]) in GENERATION_ASSERTIONS for result in results) and not free_generation:
        unsupported_claims.append("FREE_GENERATION_VERIFIED")
        outcomes.append("INCONCLUSIVE")
    outcome = _combine(outcomes)
    probe_hashes = {
        path: artifacts[path]
        for path in sorted({str(result["source"]) for result in results})
        if path in artifacts
    }
    report: dict[str, object] = {
        "artifact_hashes": dict(sorted(artifacts.items())),
        "base_checkpoint_hash": base_hash,
        "command": "verify",
        "contract_hashes": {path: artifacts[path] for path in contract_paths},
        "environment": {
            "modelpact_importable": importlib.util.find_spec("modelpact") is not None,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "python_no_site": bool(sys.flags.no_site),
            "python_safe_path": bool(sys.flags.safe_path),
            "safetensors": getattr(sys.modules.get("safetensors"), "__version__", "unknown"),
            "torch": torch.__version__,
        },
        "free_generation_results": free_generation,
        "model_adapter_id": observed_adapter,
        "outcome": outcome,
        "patch_id": manifest["patch_id"],
        "probe_hashes": probe_hashes,
        "schema_version": 1,
        "tokenizer_hash": tokenizer_hash,
        "unsupported_claims": sorted(set(unsupported_claims)),
        "verification_results": results,
        "warnings": sorted(set(warnings)),
        "wording": "Verified under the declared contracts, probe spaces, generation policy, environment, and search budget.",
    }
    report["result_hash"] = _hash_canonical(report)
    return report


def _write_report(path_value: str | None, result: dict[str, object]) -> None:
    if path_value is None:
        return
    target = Path(path_value)
    if target.exists() or target.is_symlink():
        raise ToolError("report output already exists")
    parent = _safe_output_parent(target)
    descriptor, name = tempfile.mkstemp(prefix=f".{target.name}.", dir=parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        temporary.write_bytes(_canonical(result))
        if target.exists() or target.is_symlink():
            raise ToolError("report output already exists")
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _parser() -> argparse.ArgumentParser:
    if MODE == "apply":
        parser = argparse.ArgumentParser(description="Independently apply a Behavior Patch Bundle v1")
        parser.add_argument("base", help="unsharded SafeTensors checkpoint file or directory")
        parser.add_argument("output", help="new checkpoint directory; must not exist")
        parser.add_argument("--patch", default=DEFAULT_PATCH, help="Patch Bundle v1 directory")
        return parser
    parser = argparse.ArgumentParser(description="Independently verify a Behavior Patch Bundle v1")
    parser.add_argument("base", help="unsharded SafeTensors checkpoint file or directory")
    parser.add_argument("--patch", default=DEFAULT_PATCH, help="Patch Bundle v1 directory")
    adapters = parser.add_mutually_exclusive_group(required=True)
    adapters.add_argument(
        "--adapter-kind",
        choices=("tiny", "huggingface"),
        help="reviewed adapter implementation embedded in this generated verifier",
    )
    adapters.add_argument(
        "--adapter",
        help="separately trusted local adapter as module:attribute; never loaded from the bundle",
    )
    parser.add_argument("--contract", action="append", default=[], help="content-addressed bundle-relative contract")
    parser.add_argument("--include-holdout", action="store_true", help="execute declared sealed holdout probes")
    parser.add_argument("--dtype", default="float32", choices=("float16", "bfloat16", "float32", "float64"))
    parser.add_argument("--output", help="write the newly generated verification report atomically")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if MODE == "apply":
            root, manifest, program, patch_tensors = _load_bundle(Path(args.patch))
            result = _materialize(
                Path(args.base), Path(args.output), root, manifest, program, patch_tensors
            )
        elif MODE == "verify":
            result = _verify(
                Path(args.base),
                Path(args.patch),
                args.adapter,
                args.adapter_kind,
                list(args.contract),
                bool(args.include_holdout),
                args.dtype,
            )
            _write_report(args.output, result)
        else:
            raise UnsupportedError(f"invalid generated tool mode: {MODE}")
        print(_canonical(result).decode("utf-8"))
        outcome = result.get("outcome")
        return {"PASS": 0, "FAIL": 2, "UNSUPPORTED": 3, "INCONCLUSIVE": 4}.get(str(outcome), 4)
    except ToolError as error:
        result = {
            "command": MODE,
            "error": str(error),
            "outcome": error.outcome,
            "patch_id": EXPECTED_PATCH_ID,
            "schema_version": 1,
        }
        print(_canonical(result).decode("utf-8"))
        return error.exit_code
    except Exception as error:
        result = {
            "command": MODE,
            "error": f"unexpected execution failure: {type(error).__name__}: {error}",
            "outcome": "INCONCLUSIVE",
            "patch_id": EXPECTED_PATCH_ID,
            "schema_version": 1,
        }
        print(_canonical(result).decode("utf-8"))
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
