"""Small deterministic adapter used to execute the generic extraction CLI."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor, nn

from modelpact.adapters.base import (
    ActivationPoint,
    GeneratedSample,
    GenerationPolicy,
    ModelBatch,
    PatchableModule,
)
from modelpact.models.schema import ModelStateSchema, inspect_state_schema
from modelpact.util.atomic import atomic_write_text
from modelpact.util.canonical_json import canonical_dumps


class ExtractionTestTokenizer:
    pad_token_id = 0
    bos_token_id = 0
    eos_token_id = 1
    vocab_size = 2

    @staticmethod
    def _domain(text: str) -> int:
        return 0 if "TARGET" in text.upper() else 1

    def encode(self, text: str, *, add_bos: bool = True, add_eos: bool = False) -> list[int]:
        del add_bos, add_eos
        return [self._domain(text)]

    def decode(self, token_ids: Sequence[int], *, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return "".join("A" if token == 0 else "B" for token in token_ids)

    def batch(self, texts: Sequence[str], *, add_bos: bool = True) -> ModelBatch:
        del add_bos
        ids = torch.tensor([[self._domain(text)] for text in texts], dtype=torch.long)
        return ModelBatch(ids, torch.ones_like(ids, dtype=torch.bool))


class ExtractionTestModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2, bias=False)


class ExtractionTestAdapter:
    adapter_id = "modelpact.tests.extraction.v1"

    def __init__(self) -> None:
        self._tokenizer = ExtractionTestTokenizer()

    def load(
        self,
        checkpoint: str,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> nn.Module:
        model = ExtractionTestModel().to(device=device, dtype=dtype)
        state = load_file(str(Path(checkpoint) / "model.safetensors"), device=str(device))
        model.load_state_dict({name: value.to(dtype=dtype) for name, value in state.items()})
        return model

    def tokenizer(self) -> ExtractionTestTokenizer:
        return self._tokenizer

    def prepare(self, model: nn.Module) -> None:
        model.eval()

    def forward_logits(self, model: nn.Module, batch: ModelBatch) -> Tensor:
        features = torch.nn.functional.one_hot(batch.input_ids, num_classes=2).to(
            device=next(model.parameters()).device,
            dtype=next(model.parameters()).dtype,
        )
        return model.linear(features)  # type: ignore[attr-defined, no-any-return]

    def generate(
        self,
        model: nn.Module,
        batch: ModelBatch,
        policy: GenerationPolicy,
    ) -> list[GeneratedSample]:
        del policy
        logits = self.forward_logits(model, batch)
        tokens = logits[:, -1].argmax(dim=-1).detach().cpu().tolist()
        return [
            GeneratedSample((int(token),), self._tokenizer.decode((int(token),)), True)
            for token in tokens
        ]

    def patchable_modules(self, model: nn.Module) -> Iterable[PatchableModule]:
        yield PatchableModule("linear", model.linear, ("weight",), "linear")  # type: ignore[attr-defined]

    def activation_points(self, model: nn.Module) -> Iterable[ActivationPoint]:
        yield ActivationPoint("linear", model.linear, "logits")  # type: ignore[attr-defined]

    def state_schema(self, model: nn.Module) -> ModelStateSchema:
        return inspect_state_schema(model)


def save_extraction_checkpoint(path: Path, *, target: bool) -> None:
    path.mkdir(parents=True, exist_ok=False)
    model = ExtractionTestModel()
    with torch.no_grad():
        if target:
            model.linear.weight.copy_(torch.tensor([[-2.0, -2.0], [2.0, 2.0]]))
        else:
            model.linear.weight.copy_(torch.tensor([[2.0, 2.0], [-2.0, -2.0]]))
    save_file(
        {name: value.detach().contiguous() for name, value in model.state_dict().items()},
        str(path / "model.safetensors"),
    )
    atomic_write_text(
        path / "config.json",
        canonical_dumps({"architecture": "extraction-test", "schema_version": 1}) + "\n",
    )
