"""Deterministic internal decoder-only causal language model research harness."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from modelpact.adapters.base import (
    ActivationPoint,
    GeneratedSample,
    GenerationPolicy,
    ModelBatch,
    PatchableModule,
)
from modelpact.checkpoints.safetensors import save_safetensors_atomic
from modelpact.checkpoints.store import load_checkpoint_tensors
from modelpact.models.schema import ModelStateSchema, inspect_state_schema
from modelpact.util.atomic import atomic_write_text
from modelpact.util.canonical_json import canonical_dumps


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
        if not 4 <= self.vocab_size <= 1_000_000:
            raise ValueError("invalid vocabulary size")
        if not 2 <= self.max_sequence_length <= 65536:
            raise ValueError("invalid maximum sequence length")
        if not 4 <= self.hidden_size <= 16384 or self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be positive and divisible by num_heads")
        if not 1 <= self.num_layers <= 256 or not 1 <= self.num_heads <= 256:
            raise ValueError("invalid layer/head count")
        if self.intermediate_size < self.hidden_size:
            raise ValueError("intermediate_size must be at least hidden_size")
        if not 0 < self.rms_norm_epsilon < 1:
            raise ValueError("invalid RMSNorm epsilon")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> TinyConfig:
        fields = {field.name for field in cls.__dataclass_fields__.values()}
        unknown = set(value) - fields
        if unknown:
            raise ValueError(f"unknown tiny model config fields: {sorted(unknown)}")
        integer_fields = {
            "vocab_size",
            "max_sequence_length",
            "hidden_size",
            "intermediate_size",
            "num_layers",
            "num_heads",
            "initialization_seed",
        }
        for name in integer_fields.intersection(value):
            if isinstance(value[name], bool) or not isinstance(value[name], int):
                raise ValueError(f"tiny model config field must be an integer: {name}")
        if "tie_word_embeddings" in value and not isinstance(value["tie_word_embeddings"], bool):
            raise ValueError("tie_word_embeddings must be boolean")
        epsilon = value.get("rms_norm_epsilon")
        if epsilon is not None and (
            isinstance(epsilon, bool) or not isinstance(epsilon, int | float)
        ):
            raise ValueError("rms_norm_epsilon must be numeric")
        try:
            return cls(**dict(value))  # type: ignore[arg-type]
        except TypeError as error:
            raise ValueError("malformed tiny model config") from error


@dataclass(frozen=True, slots=True)
class TinyTrainingConfig:
    steps: int = 100
    batch_size: int = 8
    learning_rate: float = 3e-3
    weight_decay: float = 0.0
    gradient_clip_norm: float = 1.0
    seed: int = 101

    def __post_init__(self) -> None:
        if not 1 <= self.steps <= 1_000_000:
            raise ValueError("training steps must be between 1 and 1,000,000")
        if not 1 <= self.batch_size <= 65_536:
            raise ValueError("training batch size is invalid")
        if not 0 < self.learning_rate <= 1:
            raise ValueError("learning rate must be in (0, 1]")
        if self.weight_decay < 0 or self.gradient_clip_norm <= 0:
            raise ValueError("weight decay and gradient clipping values are invalid")


class TinyTokenizer:
    """A fixed UTF-8 byte tokenizer with three special tokens."""

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
            if not isinstance(token, int) or not 0 <= token < self.vocab_size:
                raise ValueError(f"invalid token ID: {token}")
            if token < self.byte_offset:
                if skip_special_tokens:
                    continue
                # Explicit textual markers avoid silently producing invalid bytes.
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

    def configuration(self) -> dict[str, object]:
        return {
            "bos_token_id": self.bos_token_id,
            "byte_offset": self.byte_offset,
            "eos_token_id": self.eos_token_id,
            "kind": "utf8_byte",
            "pad_token_id": self.pad_token_id,
            "schema_version": 1,
            "vocab_size": self.vocab_size,
        }


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
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        *,
        output_hidden_states: bool = False,
    ) -> TinyCausalLMOutput:
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError("input_ids must be a rank-2 long tensor")
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
        return TinyCausalLMOutput(logits=self.lm_head(hidden), hidden_states=tuple(states))


class TinyModelAdapter:
    adapter_id = "modelpact.tiny_causal_lm.v1"

    def __init__(self, tokenizer: TinyTokenizer | None = None) -> None:
        self._tokenizer = tokenizer or TinyTokenizer()

    def load(
        self,
        checkpoint: str,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> TinyCausalLM:
        root = Path(checkpoint)
        config_path = root / "config.json"
        if root.is_symlink() or not config_path.is_file() or config_path.is_symlink():
            raise ValueError("tiny checkpoint requires a regular config.json")
        if config_path.stat().st_size > 1024 * 1024:
            raise ValueError("tiny model config exceeds size limit")
        value = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("model_type") != "modelpact_tiny_causal_lm":
            raise ValueError("not a ModelPact tiny model checkpoint")
        raw_config = value.get("model_config")
        if not isinstance(raw_config, dict):
            raise ValueError("malformed tiny model configuration")
        config = TinyConfig.from_dict(raw_config)
        if config.vocab_size != self._tokenizer.vocab_size:
            raise ValueError("checkpoint/tokenizer vocabulary mismatch")
        tensors = load_checkpoint_tensors(root)
        if config.tie_word_embeddings:
            embedding = tensors.get("token_embedding.weight")
            head = tensors.get("lm_head.weight")
            if embedding is None or head is None or not torch.equal(embedding, head):
                raise ValueError("checkpoint violates tied embedding alias")
        model = TinyCausalLM(config)
        missing, unexpected = model.load_state_dict(tensors, strict=False)
        if missing or unexpected:
            raise ValueError(
                f"checkpoint state mismatch; missing={missing}, unexpected={unexpected}"
            )
        return model.to(device=device, dtype=dtype)

    def tokenizer(self) -> TinyTokenizer:
        return self._tokenizer

    def prepare(self, model: nn.Module) -> None:
        if not isinstance(model, TinyCausalLM):
            raise TypeError("TinyModelAdapter requires TinyCausalLM")
        model.eval()

    def forward_logits(self, model: nn.Module, batch: ModelBatch) -> Tensor:
        if not isinstance(model, TinyCausalLM):
            raise TypeError("TinyModelAdapter requires TinyCausalLM")
        device = next(model.parameters()).device
        moved = batch.to(device)
        output = cast(TinyCausalLMOutput, model(moved.input_ids, moved.attention_mask))
        return output.logits

    def generate(
        self, model: nn.Module, batch: ModelBatch, policy: GenerationPolicy
    ) -> list[GeneratedSample]:
        if not isinstance(model, TinyCausalLM):
            raise TypeError("TinyModelAdapter requires TinyCausalLM")
        device = next(model.parameters()).device
        moved = batch.to(device)
        sequences = [
            moved.input_ids[row, moved.attention_mask[row].bool()].clone()
            for row in range(moved.input_ids.shape[0])
        ]
        generated: list[list[int]] = [[] for _ in sequences]
        finished = [False] * len(sequences)
        generator = torch.Generator(device=device)
        generator.manual_seed(policy.seed)
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
                        sequence = torch.cat(
                            (sequence, torch.tensor([next_token], device=device, dtype=torch.long))
                        )
                        sequences[row] = sequence
                        if policy.stop_on_eos and next_token == self._tokenizer.eos_token_id:
                            finished[row] = True
                    if all(finished):
                        break
        finally:
            model.train(prior_mode)
        return [
            GeneratedSample(
                token_ids=tuple(tokens),
                text=self._tokenizer.decode(tokens),
                finished=finished[index],
            )
            for index, tokens in enumerate(generated)
        ]

    def patchable_modules(self, model: nn.Module) -> Iterable[PatchableModule]:
        if not isinstance(model, TinyCausalLM):
            raise TypeError("TinyModelAdapter requires TinyCausalLM")
        for path, module in model.named_modules():
            if isinstance(module, nn.Linear):
                names = ("weight",) + (("bias",) if module.bias is not None else ())
                yield PatchableModule(path, module, names, "linear")
            elif isinstance(module, nn.Embedding):
                yield PatchableModule(path, module, ("weight",), "embedding")
            elif isinstance(module, TinyRMSNorm):
                yield PatchableModule(path, module, ("weight",), "rms_norm")

    def activation_points(self, model: nn.Module) -> Iterable[ActivationPoint]:
        if not isinstance(model, TinyCausalLM):
            raise TypeError("TinyModelAdapter requires TinyCausalLM")
        for index, layer in enumerate(model.layers):
            yield ActivationPoint(f"layers.{index}", layer, "residual_stream")
        yield ActivationPoint("final_norm", model.final_norm, "final_residual_stream")

    def state_schema(self, model: nn.Module) -> ModelStateSchema:
        if not isinstance(model, TinyCausalLM):
            raise TypeError("TinyModelAdapter requires TinyCausalLM")
        return inspect_state_schema(model)


def save_tiny_checkpoint(
    model: TinyCausalLM,
    output: str | Path,
    *,
    tokenizer: TinyTokenizer | None = None,
) -> Path:
    """Atomically create a complete deterministic tiny-model checkpoint."""

    target = Path(output)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    tokenizer = tokenizer or TinyTokenizer()
    try:
        # Clone every state key, including tied aliases, into data-only tensors.
        state = {
            key: value.detach().cpu().clone() for key, value in sorted(model.state_dict().items())
        }
        save_safetensors_atomic(temporary / "model.safetensors", state, overwrite=False)
        config = {
            "architectures": ["TinyCausalLM"],
            "model_config": model.config.to_dict(),
            "model_type": "modelpact_tiny_causal_lm",
            "schema_version": 1,
        }
        atomic_write_text(temporary / "config.json", canonical_dumps(config), overwrite=False)
        atomic_write_text(
            temporary / "tokenizer.json",
            canonical_dumps(tokenizer.configuration()),
            overwrite=False,
        )
        atomic_write_text(
            temporary / "tokenizer_config.json",
            canonical_dumps({"chat_template": None, **tokenizer.configuration()}),
            overwrite=False,
        )
        atomic_write_text(
            temporary / "generation_config.json",
            canonical_dumps({"do_sample": False, "eos_token_id": tokenizer.eos_token_id}),
            overwrite=False,
        )
        os.replace(temporary, target)
        return target
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def train_tiny_causal_lm(
    model: TinyCausalLM,
    texts: Sequence[str],
    *,
    tokenizer: TinyTokenizer | None = None,
    config: TinyTrainingConfig | None = None,
) -> tuple[float, ...]:
    """Run deterministic next-token training on a finite local text corpus.

    This deliberately small trainer is the CPU research harness used by ModelPactBench;
    it is not a distributed-training abstraction.
    """

    tokenizer = tokenizer or TinyTokenizer()
    training = config or TinyTrainingConfig()
    if not texts:
        raise ValueError("tiny-model training requires at least one example")
    sequences = []
    for text in texts:
        tokens = tokenizer.encode(text, add_bos=True, add_eos=True)
        tokens = tokens[: model.config.max_sequence_length]
        if len(tokens) < 2:
            raise ValueError("training examples must contain at least two tokens")
        sequences.append(tokens)
    device = next(model.parameters()).device
    generator = torch.Generator(device="cpu").manual_seed(training.seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )
    prior_mode = model.training
    model.train()
    losses = []
    try:
        for _ in range(training.steps):
            selected = torch.randint(
                len(sequences),
                (training.batch_size,),
                generator=generator,
            ).tolist()
            batch_sequences = [sequences[index] for index in selected]
            width = max(map(len, batch_sequences))
            input_ids = torch.full(
                (training.batch_size, width),
                tokenizer.pad_token_id,
                dtype=torch.long,
                device=device,
            )
            attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for row, tokens in enumerate(batch_sequences):
                length = len(tokens)
                input_ids[row, :length] = torch.tensor(tokens, dtype=torch.long, device=device)
                attention_mask[row, :length] = True
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids, attention_mask).logits[:, :-1, :]
            labels = input_ids[:, 1:]
            label_mask = attention_mask[:, 1:]
            per_token = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                labels.reshape(-1),
                reduction="none",
            ).view_as(labels)
            loss = (per_token * label_mask).sum() / label_mask.sum().clamp_min(1)
            loss.backward()  # type: ignore[no-untyped-call]
            nn.utils.clip_grad_norm_(model.parameters(), training.gradient_clip_norm)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    finally:
        model.train(prior_mode)
    return tuple(losses)
