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
from modelpact.patch.bundle import PatchBundle, create_patch_bundle

RUNNER = CliRunner()
HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64


def _contract(*, contract_id: str = "cli-contract", base_hash: str = HASH_A) -> str:
    return f"""
schema_version: 1
id: {contract_id}
contract_version: 1
model_requirements:
  tokenizer_hash: {HASH_A}
  base_signature: {base_hash}
  output_semantics: causal_lm
compile:
  objectives:
    - id: imitate
      type: teacher_cross_entropy
      source: probes/train.jsonl
verify:
  targets:
    - id: exact
      type: exact_match
      source: probes/validation.jsonl
      expected: ok
      minimum_pass_rate: 1.0
  guards:
    - id: preserve
      type: base_kl
      source: guards/validation.jsonl
      maximum_mean: 0.1
holdout:
  sealed: true
  unseal_policy: final_candidate_only
statistics:
  confidence_level: 0.95
  bootstrap_samples: 10
  bootstrap_seed: 7
generation:
  mode: greedy
  max_new_tokens: 4
"""


def _executable_contract(manifest: ModelManifest, identifier: str) -> BehaviorContract:
    return parse_contract(
        {
            "compile": {
                "objectives": [
                    {
                        "id": "imitate",
                        "source": "probes/train.jsonl",
                        "type": "teacher_cross_entropy",
                    }
                ]
            },
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
                "bootstrap_seed": 3,
                "confidence_level": 0.95,
            },
            "verify": {
                "guards": [
                    {
                        "id": "preserve",
                        "maximum_mean": 10.0,
                        "source": "guards/validation.jsonl",
                        "type": "base_kl",
                    }
                ],
                "targets": [
                    {
                        "id": "one-token",
                        "maximum": 1,
                        "minimum": 1,
                        "source": "probes/validation.jsonl",
                        "type": "generation_length",
                    }
                ],
            },
        }
    )


def _zero_bundle(
    path: Path,
    manifest: ModelManifest,
    contract: BehaviorContract,
    *,
    requires: tuple[str, ...] = (),
) -> PatchBundle:
    contract_bytes = (json.dumps(contract.to_dict(), sort_keys=True) + "\n").encode()
    return create_patch_bundle(
        path,
        name=contract.id,
        base_signature=manifest.signature.to_dict(),
        state_schema=manifest.state_schema,
        program=DeltaProgram({"final_norm.weight": VectorDelta("zero")}),
        tensors={"zero": torch.zeros(8)},
        tool_version=__version__,
        contracts={
            "contracts/guards/validation.jsonl": b'{"id":"g","prompt":"x"}\n',
            "contracts/preservation.yaml": contract_bytes,
            "contracts/probes/train.jsonl": b'{"id":"t","prompt":"x","target":"a"}\n',
            "contracts/probes/validation.jsonl": b'{"id":"v","prompt":"x"}\n',
            "contracts/target.yaml": contract_bytes,
        },
        provides=(contract.contract_id,),
        preserves=(f"{contract.contract_id}:guards",),
        requires=requires,
    )


def test_root_help_lists_the_complete_command_surface() -> None:
    result = RUNNER.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in (
        "scan",
        "diff",
        "contract",
        "compile",
        "extract",
        "inspect",
        "apply",
        "verify",
        "compose",
        "merge",
        "audit",
        "rebase",
        "revert",
        "resolve",
        "emit",
        "benchmark",
    ):
        assert command in result.stdout
    assert RUNNER.invoke(app, ["contract", "--help"]).exit_code == 0
    assert RUNNER.invoke(app, ["emit", "--help"]).exit_code == 0


def test_contract_validate_inspect_hash_and_static_failure(tmp_path: Path) -> None:
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text(_contract(), encoding="utf-8")
    second.write_text(
        _contract(contract_id="second-contract", base_hash=HASH_B),
        encoding="utf-8",
    )

    validated = RUNNER.invoke(app, ["contract", "validate", str(first), "--json"])
    assert validated.exit_code == 0
    validation = json.loads(validated.stdout)
    assert validation["status"] == "PASS"
    assert validation["contract_id"] == "cli-contract"

    inspected = RUNNER.invoke(app, ["contract", "inspect", str(first), "--json"])
    assert inspected.exit_code == 0
    assert json.loads(inspected.stdout)["contract"]["schema_version"] == 1

    hashed = RUNNER.invoke(app, ["contract", "hash", str(first), "--json"])
    assert hashed.exit_code == 0
    assert json.loads(hashed.stdout)["contract_hash"] == validation["contract_hash"]

    static = RUNNER.invoke(
        app,
        ["contract", "check-static", str(first), str(second), "--json"],
    )
    assert static.exit_code == 2
    assert json.loads(static.stdout)["status"] == "STATIC_CONTRACT_CONTRADICTION"


def test_scan_tiny_emits_a_stable_model_manifest(tmp_path: Path) -> None:
    checkpoint = tmp_path / "tiny"
    output = tmp_path / "manifest.json"
    model = TinyCausalLM(
        TinyConfig(
            hidden_size=8,
            intermediate_size=8,
            num_layers=1,
            num_heads=2,
            max_sequence_length=16,
        )
    )
    save_tiny_checkpoint(model, checkpoint)

    result = RUNNER.invoke(
        app,
        [
            "scan",
            "--model",
            "tiny",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    written = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["manifest"]["signature"] == written["signature"]
    assert written["patchable_parameter_count"] > 0


def test_runtime_apply_is_an_honest_unsupported_state(tmp_path: Path) -> None:
    base = tmp_path / "base"
    patch = tmp_path / "patch"
    base.mkdir()
    patch.mkdir()
    result = RUNNER.invoke(
        app,
        [
            "apply",
            str(base),
            str(patch),
            "--output",
            str(tmp_path / "out"),
            "--mode",
            "runtime",
            "--json",
        ],
    )
    assert result.exit_code == 4
    payload = json.loads(result.stdout)
    assert payload["status"] == "UNSUPPORTED"
    assert "Python process" in payload["reason"]


def test_benchmark_runs_a_real_exhaustive_experiment() -> None:
    result = RUNNER.invoke(app, ["benchmark", "closure_matrix", "--json"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "PASS"
    assert payload["result"]["executed_subsets"] == 63
    assert payload["result"]["search_space_exhausted"] is True


def test_tiny_merge_invokes_and_verifies_a_real_joint_optimization(tmp_path: Path) -> None:
    checkpoint = tmp_path / "tiny"
    model = TinyCausalLM(
        TinyConfig(
            hidden_size=8,
            intermediate_size=8,
            num_layers=1,
            num_heads=2,
            max_sequence_length=16,
        )
    )
    save_tiny_checkpoint(model, checkpoint)
    adapter = TinyModelAdapter()
    loaded = adapter.load(str(checkpoint), device="cpu", dtype=torch.float32)
    manifest = build_model_manifest(
        loaded,
        checkpoint=checkpoint,
        adapter_id=adapter.adapter_id,
    )
    left_contract = _executable_contract(manifest, "left-behavior")
    right_contract = _executable_contract(manifest, "right-behavior")
    left = _zero_bundle(tmp_path / "left", manifest, left_contract)
    right = _zero_bundle(tmp_path / "right", manifest, right_contract)

    result = RUNNER.invoke(
        app,
        [
            "merge",
            str(left.path),
            str(right.path),
            "--base",
            str(checkpoint),
            "--output",
            str(tmp_path / "merged"),
            "--adapter",
            "tiny",
            "--force-recompile",
            "--maximum-steps",
            "2",
            "--max-rank",
            "1",
            "--max-modules",
            "1",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["disposition"] == "SEMANTIC_MERGE_VERIFIED"
    assert payload["compiler_invoked"] is True
    assert payload["compilation"]["steps_executed"] == 2
    assert payload["compilation"]["diagnostics"]["real_optimization"] is True
    assert payload["compilation"]["diagnostics"]["parent_teacher_objectives"] == 2
    assert (tmp_path / "merged" / "resolved-delta.safetensors").is_file()


def test_cross_architecture_tiny_rebase_recompiles_and_packages(tmp_path: Path) -> None:
    source_checkpoint = tmp_path / "source"
    target_checkpoint = tmp_path / "target"
    source_model = TinyCausalLM(
        TinyConfig(
            hidden_size=8,
            intermediate_size=8,
            num_layers=1,
            num_heads=2,
            max_sequence_length=16,
        )
    )
    target_model = TinyCausalLM(
        TinyConfig(
            hidden_size=12,
            intermediate_size=12,
            num_layers=1,
            num_heads=3,
            max_sequence_length=16,
        )
    )
    save_tiny_checkpoint(source_model, source_checkpoint)
    save_tiny_checkpoint(target_model, target_checkpoint)
    adapter = TinyModelAdapter()
    loaded = adapter.load(str(source_checkpoint), device="cpu", dtype=torch.float32)
    manifest = build_model_manifest(
        loaded,
        checkpoint=source_checkpoint,
        adapter_id=adapter.adapter_id,
    )
    contract = _executable_contract(manifest, "portable-behavior")
    patch = _zero_bundle(tmp_path / "source-patch", manifest, contract)
    output = tmp_path / "rebased"

    result = RUNNER.invoke(
        app,
        [
            "rebase",
            str(patch.path),
            "--from-base",
            str(source_checkpoint),
            "--onto",
            str(target_checkpoint),
            "--source-adapter",
            "tiny",
            "--target-adapter",
            "tiny",
            "--output",
            str(output),
            "--maximum-steps",
            "2",
            "--max-rank",
            "1",
            "--max-modules",
            "1",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["claim"] == "SEMANTIC_REBASE_VERIFIED"
    assert payload["disposition"] == "SEMANTIC_REBASE_VERIFIED"
    assert payload["optimization_steps"] == 2
    rebased_manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert rebased_manifest["rebased_from"] == patch.manifest.patch_id
    assert rebased_manifest["compiler_configuration"]["mode"] == "semantic_recompile"
    assert (output / "certificate.json").is_file()


def test_resolve_rejects_an_omitted_declared_dependency(tmp_path: Path) -> None:
    checkpoint = tmp_path / "tiny"
    model = TinyCausalLM(
        TinyConfig(
            hidden_size=8,
            intermediate_size=8,
            num_layers=1,
            num_heads=2,
            max_sequence_length=16,
        )
    )
    save_tiny_checkpoint(model, checkpoint)
    adapter = TinyModelAdapter()
    loaded = adapter.load(str(checkpoint), device="cpu", dtype=torch.float32)
    manifest = build_model_manifest(
        loaded,
        checkpoint=checkpoint,
        adapter_id=adapter.adapter_id,
    )
    contract = _executable_contract(manifest, "dependent-behavior")
    patch = _zero_bundle(
        tmp_path / "dependent",
        manifest,
        contract,
        requires=("sha256:" + "c" * 64,),
    )
    stack_spec = tmp_path / "stack.toml"
    stack_spec.write_text(
        "\n".join(
            (
                "schema_version = 1",
                f'base = "{checkpoint.as_posix()}"',
                f'patches = ["{patch.path.as_posix()}"]',
                "[policy]",
                "repair_conflicts = false",
                "subset_audit_budget = 0",
            )
        ),
        encoding="utf-8",
    )
    output = tmp_path / "resolved"
    result = RUNNER.invoke(
        app,
        ["resolve", str(stack_spec), "--output", str(output), "--json"],
    )
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["status"] == "ERROR"
    assert "omits declared dependencies" in payload["error"]
    assert not output.exists()


def test_resolve_and_revert_execute_remaining_stack_contracts(tmp_path: Path) -> None:
    checkpoint = tmp_path / "tiny"
    model = TinyCausalLM(
        TinyConfig(
            hidden_size=8,
            intermediate_size=8,
            num_layers=1,
            num_heads=2,
            max_sequence_length=16,
        )
    )
    save_tiny_checkpoint(model, checkpoint)
    adapter = TinyModelAdapter()
    loaded = adapter.load(str(checkpoint), device="cpu", dtype=torch.float32)
    manifest = build_model_manifest(
        loaded,
        checkpoint=checkpoint,
        adapter_id=adapter.adapter_id,
    )
    contract = parse_contract(
        {
            "compile": {"objectives": []},
            "contract_version": 1,
            "generation": {"max_new_tokens": 1, "mode": "greedy", "seeds": [0]},
            "holdout": {"sealed": True, "unseal_policy": "final_candidate_only"},
            "id": "stack-contract",
            "model_requirements": {
                "adapter_id": adapter.adapter_id,
                "architecture_hash": manifest.signature.architecture_hash,
                "base_signature": manifest.signature.signature_hash,
                "output_semantics": "causal_lm",
                "state_schema_hash": manifest.signature.state_schema_hash,
                "tokenizer_hash": manifest.signature.tokenizer_hash,
            },
            "schema_version": 1,
            "statistics": {
                "bootstrap_samples": 10,
                "bootstrap_seed": 3,
                "confidence_level": 0.95,
            },
            "verify": {
                "guards": [],
                "targets": [
                    {
                        "id": "one-token",
                        "maximum": 1,
                        "minimum": 1,
                        "source": "data/validation.jsonl",
                        "type": "generation_length",
                    }
                ],
            },
        }
    )
    patch = tmp_path / "patch"
    contract_bytes = (json.dumps(contract.to_dict(), sort_keys=True) + "\n").encode()
    bundle = create_patch_bundle(
        patch,
        name="zero-stack-patch",
        base_signature=manifest.signature.to_dict(),
        state_schema=manifest.state_schema,
        program=DeltaProgram({"final_norm.weight": VectorDelta("zero")}),
        tensors={"zero": torch.zeros(8)},
        tool_version=__version__,
        contracts={
            "contracts/data/validation.jsonl": b'{"id":"p","prompt":"x"}\n',
            "contracts/preservation.yaml": contract_bytes,
            "contracts/target.yaml": contract_bytes,
        },
        provides=(contract.contract_id,),
    )
    stack_spec = tmp_path / "stack.toml"
    stack_spec.write_text(
        "\n".join(
            (
                "schema_version = 1",
                f'base = "{checkpoint.as_posix()}"',
                f'patches = ["{patch.as_posix()}"]',
                "[policy]",
                "repair_conflicts = false",
                "subset_audit_budget = 0",
            )
        ),
        encoding="utf-8",
    )
    resolved_path = tmp_path / "resolved"
    resolved = RUNNER.invoke(
        app,
        ["resolve", str(stack_spec), "--output", str(resolved_path), "--json"],
    )
    assert resolved.exit_code == 0, resolved.stdout
    lockfile = resolved_path / "stack.lock.json"
    assert lockfile.is_file()
    assert json.loads(resolved.stdout)["resolution"] == "NAIVE_ADDITIVE_STACK"

    reverted_path = tmp_path / "reverted"
    reverted = RUNNER.invoke(
        app,
        [
            "revert",
            str(lockfile),
            "--remove",
            bundle.manifest.patch_id,
            "--output",
            str(reverted_path),
            "--adapter",
            "tiny",
            "--json",
        ],
    )
    assert reverted.exit_code == 0, reverted.stdout
    result = json.loads(reverted.stdout)
    assert result["reversion_grade"] == "BASE_HASH_RESTORED"
    assert result["lock"]["patch_hashes"] == {}
