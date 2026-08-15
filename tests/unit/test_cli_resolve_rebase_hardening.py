from __future__ import annotations

import json
from pathlib import Path

import torch
from typer.testing import CliRunner

from modelpact import __version__
from modelpact.adapters.tiny_lm import (
    TinyCausalLM,
    TinyConfig,
    TinyModelAdapter,
    save_tiny_checkpoint,
)
from modelpact.cli import app
from modelpact.contracts.ast import BehaviorContract
from modelpact.contracts.parser import parse_contract
from modelpact.models.manifest import ModelManifest, build_model_manifest
from modelpact.patch.ast import DeltaProgram, VectorDelta
from modelpact.patch.bundle import create_patch_bundle, load_patch_bundle, missing_bundle_artifacts
from modelpact.util.hashing import sha256_file
from tests.support.semantic_fixtures import (
    constant_output_model,
    learned_behavior_bundle,
    searchable_behavior_contract,
)

RUNNER = CliRunner()


def _checkpoint(path: Path) -> ModelManifest:
    model = TinyCausalLM(
        TinyConfig(
            hidden_size=8,
            intermediate_size=8,
            num_layers=1,
            num_heads=2,
            max_sequence_length=16,
        )
    )
    save_tiny_checkpoint(model, path)
    adapter = TinyModelAdapter()
    loaded = adapter.load(str(path), device="cpu", dtype=torch.float32)
    return build_model_manifest(loaded, checkpoint=path, adapter_id=adapter.adapter_id)


def _contract(
    manifest: ModelManifest,
    identifier: str,
    *,
    targets: bool,
    guards: bool,
    target_length: int = 1,
    maximum_guard_kl: float = 10.0,
) -> BehaviorContract:
    target_assertions = (
        [
            {
                "id": "generated-length",
                "maximum": target_length,
                "minimum": target_length,
                "source": "probes/validation.jsonl",
                "type": "generation_length",
            }
        ]
        if targets
        else []
    )
    guard_assertions = (
        [
            {
                "id": "preserve-base",
                "maximum_mean": maximum_guard_kl,
                "source": "guards/validation.jsonl",
                "type": "base_kl",
            }
        ]
        if guards
        else []
    )
    objectives = (
        [
            {
                "id": "imitate",
                "source": "probes/train.jsonl",
                "type": "teacher_cross_entropy",
            }
        ]
        if targets
        else []
    )
    return parse_contract(
        {
            "compile": {"objectives": objectives},
            "contract_version": 1,
            "generation": {"max_new_tokens": 1, "mode": "greedy", "seeds": [0]},
            "holdout": {"sealed": True, "unseal_policy": "final_candidate_only"},
            "id": identifier,
            "model_requirements": {
                "adapter_id": "modelpact.tiny_causal_lm.v1",
                "architecture_hash": manifest.signature.architecture_hash,
                "base_signature": manifest.signature.signature_hash,
                "output_semantics": "causal_lm",
                "state_schema_hash": manifest.signature.state_schema_hash,
                "tokenizer_hash": manifest.signature.tokenizer_hash,
            },
            "schema_version": 1,
            "statistics": {
                "bootstrap_samples": 10,
                "bootstrap_seed": 19,
                "confidence_level": 0.95,
            },
            "verify": {"guards": guard_assertions, "targets": target_assertions},
        }
    )


def _single_contract_bundle(
    path: Path,
    manifest: ModelManifest,
    identifier: str,
    *,
    target_length: int = 1,
    delta_value: float = 0.0,
    maximum_guard_kl: float = 10.0,
) -> Path:
    contract = _contract(
        manifest,
        identifier,
        targets=True,
        guards=True,
        target_length=target_length,
        maximum_guard_kl=maximum_guard_kl,
    )
    contract_bytes = (json.dumps(contract.to_dict(), sort_keys=True) + "\n").encode()
    bundle = create_patch_bundle(
        path,
        name=identifier,
        base_signature=manifest.signature.to_dict(),
        state_schema=manifest.state_schema,
        program=DeltaProgram({"final_norm.weight": VectorDelta("zero")}),
        tensors={"zero": torch.full((8,), delta_value)},
        tool_version=__version__,
        contracts={
            "contracts/guards/validation.jsonl": b'{"id":"g","prompt":"control"}\n',
            "contracts/preservation.yaml": contract_bytes,
            "contracts/probes/train.jsonl": b'{"id":"t","prompt":"x","target":"a"}\n',
            "contracts/probes/validation.jsonl": b'{"id":"v","prompt":"x"}\n',
            "contracts/target.yaml": contract_bytes,
        },
        provides=(contract.contract_id,),
        preserves=(contract.contract_id,),
    )
    return bundle.path


def test_resolve_emits_complete_patch_certificate_and_executed_audit(tmp_path: Path) -> None:
    checkpoint = tmp_path / "base"
    manifest = _checkpoint(checkpoint)
    left = _single_contract_bundle(tmp_path / "left", manifest, "left")
    right = _single_contract_bundle(tmp_path / "right", manifest, "right")
    specification = tmp_path / "stack.toml"
    specification.write_text(
        "\n".join(
            (
                "schema_version = 1",
                f'base = "{checkpoint.as_posix()}"',
                f'patches = ["{left.as_posix()}", "{right.as_posix()}"]',
                "[policy]",
                "repair_conflicts = false",
                "subset_audit_budget = 3",
            )
        ),
        encoding="utf-8",
    )
    output = tmp_path / "resolved"

    result = RUNNER.invoke(
        app,
        ["resolve", str(specification), "--output", str(output), "--json"],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    resolved_path = output / "resolved-patch"
    bundle = load_patch_bundle(resolved_path, state_schema=manifest.state_schema)
    assert payload["resolved_patch_id"] == bundle.manifest.patch_id
    assert missing_bundle_artifacts(bundle.manifest) == ()
    assert bundle.evidence_id in (resolved_path / "verify_patch.py").read_text(encoding="utf-8")
    lock = json.loads((output / "stack.lock.json").read_text(encoding="utf-8"))
    assert lock["certificate_hash"].startswith("sha256:")
    assert lock["audit_hash"] == sha256_file(output / "composition-audit.json")
    assert lock["resolved_artifact_hash"] == sha256_file(resolved_path / "manifest.json")
    audit = json.loads((output / "composition-audit.json").read_text(encoding="utf-8"))
    assert audit["audit"]["executed_subset_count"] == 3
    assert audit["audit"]["search_space_exhausted"] is True
    independent = RUNNER.invoke(
        app,
        [
            "verify",
            str(resolved_path),
            "--base",
            str(checkpoint),
            "--adapter",
            "tiny",
            "--json",
        ],
    )
    assert independent.exit_code == 0, independent.stdout


def test_resolve_invokes_real_tiny_repair_and_reports_budget_failure_honestly(
    tmp_path: Path,
) -> None:
    torch.manual_seed(1)
    torch.set_num_threads(1)
    checkpoint = tmp_path / "base"
    manifest = _checkpoint(checkpoint)
    left = _single_contract_bundle(
        tmp_path / "left",
        manifest,
        "left",
        delta_value=0.5,
        maximum_guard_kl=0.001,
    )
    right = _single_contract_bundle(
        tmp_path / "right",
        manifest,
        "right",
        delta_value=0.5,
        maximum_guard_kl=0.001,
    )
    specification = tmp_path / "stack.toml"
    specification.write_text(
        "\n".join(
            (
                "schema_version = 1",
                f'base = "{checkpoint.as_posix()}"',
                f'patches = ["{left.as_posix()}", "{right.as_posix()}"]',
                "[policy]",
                "repair_conflicts = true",
                "subset_audit_budget = 0",
            )
        ),
        encoding="utf-8",
    )
    output = tmp_path / "resolved"

    result = RUNNER.invoke(
        app,
        ["resolve", str(specification), "--output", str(output), "--json"],
    )

    assert result.exit_code == 2, result.stdout
    payload = json.loads(result.stdout)
    assert payload["resolution"] == "EMPIRICAL_FAILURE"
    assert payload["resolved_patch"] is None
    resolution = json.loads((output / "resolution.json").read_text(encoding="utf-8"))
    assert resolution["compiler_invoked"] is True
    assert resolution["compiler_evidence"]["steps_executed"] == 200
    assert resolution["compiler_evidence"]["budget_exhausted"] is True
    assert resolution["compiler_evidence"]["candidate_delta"] is None
    assert not (output / "resolved-patch").exists()


def test_resolve_successful_repair_executes_cegis_and_minimization(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "base"
    model = constant_output_model(hidden_size=8, num_heads=2, output="A")
    save_tiny_checkpoint(model, checkpoint)
    adapter = TinyModelAdapter()
    loaded = adapter.load(str(checkpoint), device="cpu", dtype=torch.float32)
    manifest = build_model_manifest(
        loaded,
        checkpoint=checkpoint,
        adapter_id=adapter.adapter_id,
    )
    left_contract = searchable_behavior_contract(
        manifest,
        "repair-left",
        expected="Q",
        maximum_guard_kl=30.0,
    )
    right_contract = searchable_behavior_contract(
        manifest,
        "repair-right",
        expected="Q",
        maximum_guard_kl=30.0,
    )
    left = learned_behavior_bundle(
        tmp_path / "left",
        manifest,
        left_contract,
        source_output="A",
        target_output="Q",
    )
    right = learned_behavior_bundle(
        tmp_path / "right",
        manifest,
        right_contract,
        source_output="A",
        target_output="Q",
    )
    specification = tmp_path / "stack.toml"
    specification.write_text(
        "\n".join(
            (
                "schema_version = 1",
                f'base = "{checkpoint.as_posix()}"',
                f'patches = ["{left.path.as_posix()}", "{right.path.as_posix()}"]',
                "[policy]",
                "repair_conflicts = true",
                "subset_audit_budget = 0",
            )
        ),
        encoding="utf-8",
    )
    output = tmp_path / "resolved"

    result = RUNNER.invoke(
        app,
        ["resolve", str(specification), "--output", str(output), "--json"],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["resolution"] == "VERIFIED_COMPOSITE_PATCH"
    resolution = json.loads((output / "resolution.json").read_text(encoding="utf-8"))
    assert resolution["compiler_invoked"] is True
    assert resolution["cegis"]["model_executions"] > 0
    assert resolution["cegis"]["search_executions"]
    assert "UNMINIMIZED" not in resolution["minimization"]["claims"]
    resolved = output / "resolved-patch"
    compile_evidence = json.loads(
        (resolved / "evidence" / "compile.json").read_text(encoding="utf-8")
    )
    assert compile_evidence["cegis"]["compilation_candidates"]
    minimization = json.loads(
        (resolved / "evidence" / "minimization.json").read_text(encoding="utf-8")
    )
    assert minimization["candidates"][0]["operation"] == "verify:initial"
    assert minimization["candidates"][0]["candidate_id"].startswith("sha256:")


def test_rebase_executes_contract_union_and_explicit_new_base_policy(tmp_path: Path) -> None:
    checkpoint = tmp_path / "base"
    manifest = _checkpoint(checkpoint)
    target_contract = _contract(manifest, "target-behavior", targets=True, guards=False)
    guard_contract = _contract(manifest, "source-guard", targets=False, guards=True)
    target_bytes = (json.dumps(target_contract.to_dict(), sort_keys=True) + "\n").encode()
    guard_bytes = (json.dumps(guard_contract.to_dict(), sort_keys=True) + "\n").encode()
    source_bundle = create_patch_bundle(
        tmp_path / "source-patch",
        name="split-contract-patch",
        base_signature=manifest.signature.to_dict(),
        state_schema=manifest.state_schema,
        program=DeltaProgram({"final_norm.weight": VectorDelta("zero")}),
        tensors={"zero": torch.zeros(8)},
        tool_version=__version__,
        contracts={
            "contracts/guards/validation.jsonl": b'{"id":"g","prompt":"control"}\n',
            "contracts/preservation.yaml": guard_bytes,
            "contracts/probes/train.jsonl": b'{"id":"t","prompt":"x","target":"a"}\n',
            "contracts/probes/validation.jsonl": b'{"id":"v","prompt":"x"}\n',
            "contracts/target.yaml": target_bytes,
        },
        provides=(target_contract.contract_id,),
        preserves=(guard_contract.contract_id,),
    )
    policy = _contract(manifest, "new-base-controls", targets=False, guards=True)
    policy_path = tmp_path / "new-base-policy.json"
    policy_path.write_text(json.dumps(policy.to_dict(), sort_keys=True), encoding="utf-8")
    guards = tmp_path / "guards"
    guards.mkdir()
    (guards / "validation.jsonl").write_text('{"id":"g","prompt":"control"}\n', encoding="utf-8")
    output = tmp_path / "rebased"

    result = RUNNER.invoke(
        app,
        [
            "rebase",
            str(source_bundle.path),
            "--from-base",
            str(checkpoint),
            "--onto",
            str(checkpoint),
            "--source-adapter",
            "tiny",
            "--target-adapter",
            "tiny",
            "--new-base-policy",
            str(policy_path),
            "--output",
            str(output),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert len(payload["verification"]) == 3
    rebased = load_patch_bundle(output, state_schema=manifest.state_schema)
    assert len(rebased.manifest.provides) == 1
    assert len(rebased.manifest.preserves) == 2
    assert missing_bundle_artifacts(rebased.manifest) == ()
    assert rebased.evidence_id in (output / "verify_patch.py").read_text(encoding="utf-8")
    certificate = json.loads((output / "certificate.json").read_text(encoding="utf-8"))
    assert len(certificate["contract_hashes"]) == 3
    compile_evidence = json.loads(
        (output / "evidence" / "compile.json").read_text(encoding="utf-8")
    )
    assert len(compile_evidence["source_teacher_verification"]) == 2
    assert compile_evidence["new_base_policy"] == policy_path.resolve().as_posix()
    independent = RUNNER.invoke(
        app,
        [
            "verify",
            str(output),
            "--base",
            str(checkpoint),
            "--adapter",
            "tiny",
            "--json",
        ],
    )
    assert independent.exit_code == 0, independent.stdout


def test_rebase_rejects_a_source_patched_teacher_that_fails_contracts(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "base"
    manifest = _checkpoint(checkpoint)
    source_patch = _single_contract_bundle(
        tmp_path / "source-patch",
        manifest,
        "invalid-source-teacher",
        target_length=2,
    )
    output = tmp_path / "rebase-failed"

    result = RUNNER.invoke(
        app,
        [
            "rebase",
            str(source_patch),
            "--from-base",
            str(checkpoint),
            "--onto",
            str(checkpoint),
            "--source-adapter",
            "tiny",
            "--target-adapter",
            "tiny",
            "--output",
            str(output),
            "--json",
        ],
    )

    assert result.exit_code == 2, result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "SOURCE_PATCHED_TEACHER_FAILED"
    assert "invalid teacher" in payload["reason"]
    assert (output / "rebase-evidence.json").is_file()
