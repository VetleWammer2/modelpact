from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from modelpact.adapters.tiny_lm import TinyCausalLM, TinyConfig, TinyModelAdapter
from modelpact.contracts import (
    AssertionType,
    GenerationMode,
    GenerationPolicy,
    VerificationAssertion,
)
from modelpact.verify import (
    ModelBackedRecordProvider,
    ProbeDataError,
    VerificationRole,
    load_json_schemas,
    load_probe_records,
)
from modelpact.verify.engine import UnsupportedRecordProviderError


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def make_provider(
    tmp_path: Path, *, policy: GenerationPolicy | None = None
) -> ModelBackedRecordProvider:
    adapter = TinyModelAdapter()
    torch.manual_seed(7)
    model = TinyCausalLM(
        TinyConfig(
            max_sequence_length=64,
            hidden_size=8,
            intermediate_size=16,
            num_layers=1,
            num_heads=2,
        )
    )
    base = TinyCausalLM(model.config)
    base.load_state_dict(model.state_dict())
    reference = TinyCausalLM(model.config)
    reference.load_state_dict(model.state_dict())
    return ModelBackedRecordProvider(
        adapter=adapter,
        model=model,
        base_model=base,
        reference_model=reference,
        contract_root=tmp_path,
        generation_policy=policy or GenerationPolicy(max_new_tokens=2, seeds=(0,)),
    )


def spec(kind: AssertionType, **options: object) -> VerificationAssertion:
    return VerificationAssertion("check", kind, "probes.jsonl", options)


def test_strict_probe_loader_hashes_prompts_and_rejects_unknown_fields(tmp_path: Path) -> None:
    prompt = "hello"
    write_jsonl(tmp_path / "probes.jsonl", [{"id": "p", "prompt": prompt}])
    records = load_probe_records(tmp_path, "probes.jsonl")
    assert records[0]["id"] == "p"
    write_jsonl(tmp_path / "bad.jsonl", [{"prompt": prompt, "precomputed_pass": True}])
    with pytest.raises(ProbeDataError, match="unknown probe field"):
        load_probe_records(tmp_path, "bad.jsonl")
    with pytest.raises(ValueError):
        load_probe_records(tmp_path, "../outside.jsonl")


def test_model_provider_computes_logits_base_and_reference_distributions(tmp_path: Path) -> None:
    write_jsonl(tmp_path / "probes.jsonl", [{"id": "p", "prompt": "abc"}])
    provider = make_provider(tmp_path)
    for kind in (AssertionType.REFERENCE_KL, AssertionType.BASE_KL):
        records = provider.records_for(
            spec(kind, maximum_mean=0.1),
            source="probes.jsonl",
            role=VerificationRole.GUARD,
            holdout_capability=None,
        )
        assert len(records) == 1
        assert records[0].logits is not None
        reference = (
            records[0].reference_logits
            if kind is AssertionType.REFERENCE_KL
            else records[0].base_logits
        )
        assert reference is not None
        assert torch.equal(records[0].logits, reference)
    assert provider.probe_hashes["probes.jsonl"].startswith("sha256:")


def test_model_provider_uses_content_hashed_reference_logits_without_teacher(
    tmp_path: Path,
) -> None:
    adapter = TinyModelAdapter()
    torch.manual_seed(17)
    model = TinyCausalLM(
        TinyConfig(
            max_sequence_length=64,
            hidden_size=8,
            intermediate_size=16,
            num_layers=1,
            num_heads=2,
        )
    )
    prompt = "abc"
    batch = adapter.tokenizer().batch((prompt,), add_bos=True)
    with torch.no_grad():
        logits = adapter.forward_logits(model, batch)[0]
    length = int(batch.attention_mask[0].sum().item())
    reference = logits[:length].detach().cpu()
    write_jsonl(
        tmp_path / "probes.jsonl",
        [{"id": "p", "prompt": prompt, "reference_logits": reference.tolist()}],
    )
    provider = ModelBackedRecordProvider(
        adapter=adapter,
        model=model,
        contract_root=tmp_path,
        generation_policy=GenerationPolicy(max_new_tokens=1, seeds=(0,)),
    )

    records = provider.records_for(
        spec(AssertionType.REFERENCE_KL, maximum_mean=0.1),
        source="probes.jsonl",
        role=VerificationRole.TARGET,
        holdout_capability=None,
    )

    torch.testing.assert_close(records[0].reference_logits, reference)


def test_model_provider_scores_choices_with_real_forward_passes(tmp_path: Path) -> None:
    write_jsonl(
        tmp_path / "probes.jsonl",
        [{"id": "choice", "prompt": "pick ", "choices": ["A", "B"], "correct_choice": "A"}],
    )
    provider = make_provider(tmp_path)
    records = provider.records_for(
        spec(
            AssertionType.MULTIPLE_CHOICE_MARGIN,
            choices=["A", "B"],
            correct_choice="A",
            minimum_margin=-100.0,
        ),
        source="probes.jsonl",
        role=VerificationRole.TARGET,
        holdout_capability=None,
    )
    scores = records[0].values["choice_log_probabilities"]
    assert isinstance(scores, dict)
    assert set(scores) == {"A", "B"}
    assert all(isinstance(value, float) for value in scores.values())


def test_model_provider_generates_and_records_evidence(tmp_path: Path) -> None:
    write_jsonl(tmp_path / "probes.jsonl", [{"id": "g", "prompt": "x", "expected": "z"}])
    provider = make_provider(tmp_path, policy=GenerationPolicy(max_new_tokens=2, seeds=(4, 2)))
    records = provider.records_for(
        spec(AssertionType.EXACT_MATCH, expected="z"),
        source="probes.jsonl",
        role=VerificationRole.TARGET,
        holdout_capability=None,
    )
    assert len(records) == 2
    evidence = provider.generation_evidence()
    assert len(evidence) == 2
    assert {item.seed for item in evidence} == {2, 4}
    assert all(item.output_hash.startswith("sha256:") for item in evidence)
    # Re-requesting the same source uses cached executions and does not duplicate evidence.
    provider.records_for(
        spec(AssertionType.EXACT_MATCH, expected="z"),
        source="probes.jsonl",
        role=VerificationRole.TARGET,
        holdout_capability=None,
    )
    assert len(provider.generation_evidence()) == 2


def test_unrepresentable_generation_policy_is_unsupported(tmp_path: Path) -> None:
    write_jsonl(tmp_path / "probes.jsonl", [{"prompt": "x", "expected": "y"}])
    provider = make_provider(
        tmp_path,
        policy=GenerationPolicy(
            mode=GenerationMode.SAMPLE,
            max_new_tokens=2,
            top_p=0.8,
            seeds=(1,),
        ),
    )
    with pytest.raises(UnsupportedRecordProviderError, match="cannot represent"):
        provider.records_for(
            spec(AssertionType.EXACT_MATCH, expected="y"),
            source="probes.jsonl",
            role=VerificationRole.TARGET,
            holdout_capability=None,
        )


def test_schema_loader_is_bounded_and_data_only(tmp_path: Path) -> None:
    (tmp_path / "schema.json").write_text(
        json.dumps({"type": "object", "required": ["answer"]}), encoding="utf-8"
    )
    schemas = load_json_schemas(tmp_path, ("schema.json",))
    assert schemas["schema.json"]["type"] == "object"
    (tmp_path / "not-schema.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ProbeDataError, match="must be an object"):
        load_json_schemas(tmp_path, ("not-schema.json",))
