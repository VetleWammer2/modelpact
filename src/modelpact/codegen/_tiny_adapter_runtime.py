#!/usr/bin/env python3
"""Reviewed source fragment embedded in generated standalone tiny-LM verifiers.

This module is package-owned code, never copied into or imported from an
untrusted patch bundle. Generated verifiers expose it only through the explicit
``--adapter-kind tiny`` selection. Runtime dependencies are limited to PyTorch
and SafeTensors.
"""

from __future__ import annotations

# MODELPACT_BUILTIN_TINY_ADAPTER_BEGIN
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


# MODELPACT_BUILTIN_TINY_ADAPTER_END
