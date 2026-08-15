from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from pytest import MonkeyPatch
from typer.testing import CliRunner

from modelpact import __version__
from modelpact.adapters.base import GenerationPolicy as AdapterGenerationPolicy
from modelpact.adapters.tiny_lm import (
    TinyCausalLM,
    TinyConfig,
    TinyModelAdapter,
    TinyTokenizer,
    TinyTrainingConfig,
    save_tiny_checkpoint,
    train_tiny_causal_lm,
)
from modelpact.cli import EXIT_FAILED, _verified_diff_manifest, app
from modelpact.contracts.ast import BehaviorContract
from modelpact.contracts.parser import parse_contract
from modelpact.models.manifest import ModelManifest, build_model_manifest
from modelpact.patch.ast import (
    DeltaProgram,
    LowRankMatrixDelta,
    SparseMatrixDelta,
    Sum,
    VectorDelta,
)
from modelpact.patch.bundle import PatchBundle, create_patch_bundle, load_patch_bundle
from modelpact.patch.mount import mount_patch
from modelpact.util.hashing import hash_canonical, sha256_file
from tests.support.semantic_fixtures import (
    brittle_case_behavior_bundle,
    case_brittle_model,
    constant_output_model,
    learned_behavior_bundle,
    searchable_behavior_contract,
)

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
        preserves=(contract.contract_id,),
        requires=requires,
    )


def _holdout_bundle(
    path: Path,
    manifest: ModelManifest,
    *,
    visible_expected: str,
    holdout_expected: str | None = None,
) -> PatchBundle:
    contract = parse_contract(
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
            "holdout": {
                "sealed": True,
                "targets": "holdout/targets.jsonl",
                "unseal_policy": "final_candidate_only",
            },
            "id": "sealed-holdout-failure",
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
                        "id": "match-generation",
                        "minimum_pass_rate": 1.0,
                        "source": "probes/validation.jsonl",
                        "type": "free_generation_match",
                    }
                ],
            },
        }
    )
    contract_bytes = (json.dumps(contract.to_dict(), sort_keys=True) + "\n").encode()
    visible_row = json.dumps(
        {"expected": visible_expected, "id": "visible", "prompt": "x"},
        sort_keys=True,
    )
    holdout_row = json.dumps(
        {
            "expected": (
                visible_expected + "-must-fail" if holdout_expected is None else holdout_expected
            ),
            "id": "sealed",
            "prompt": "x",
        },
        sort_keys=True,
    )
    return create_patch_bundle(
        path,
        name=contract.id,
        base_signature=manifest.signature.to_dict(),
        state_schema=manifest.state_schema,
        program=DeltaProgram({"final_norm.weight": VectorDelta("zero")}),
        tensors={"zero": torch.zeros(8)},
        tool_version=__version__,
        contracts={
            "contracts/guards/validation.jsonl": b'{"id":"g","prompt":"control"}\n',
            "contracts/holdout/targets.jsonl": (holdout_row + "\n").encode(),
            "contracts/preservation.yaml": contract_bytes,
            "contracts/probes/train.jsonl": b'{"id":"t","prompt":"x","target":"a"}\n',
            "contracts/probes/validation.jsonl": (visible_row + "\n").encode(),
            "contracts/target.yaml": contract_bytes,
        },
        provides=(contract.contract_id,),
        preserves=(contract.contract_id,),
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


def test_runtime_apply_executes_mount_and_exact_unmount(tmp_path: Path) -> None:
    base = tmp_path / "base"
    model = TinyCausalLM(
        TinyConfig(
            hidden_size=8,
            intermediate_size=8,
            num_layers=1,
            num_heads=2,
            max_sequence_length=16,
        )
    )
    save_tiny_checkpoint(model, base)
    adapter = TinyModelAdapter()
    loaded = adapter.load(str(base), device="cpu", dtype=torch.float32)
    manifest = build_model_manifest(loaded, checkpoint=base, adapter_id=adapter.adapter_id)

    def runtime_bundle(path: Path, identifier: str, target: str, scale: float) -> PatchBundle:
        contract = _executable_contract(manifest, identifier)
        contract_bytes = (json.dumps(contract.to_dict(), sort_keys=True) + "\n").encode()
        return create_patch_bundle(
            path,
            name=identifier,
            base_signature=manifest.signature.to_dict(),
            state_schema=manifest.state_schema,
            program=DeltaProgram({target: VectorDelta("delta")}),
            tensors={"delta": torch.full((8,), scale)},
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

    first = runtime_bundle(tmp_path / "patch-first", "runtime-first", "final_norm.weight", 0.125)
    second = runtime_bundle(
        tmp_path / "patch-second",
        "runtime-second",
        "layers.0.input_norm.weight",
        -0.0625,
    )
    source_before = {
        path.relative_to(base).as_posix(): sha256_file(path)
        for path in sorted(base.rglob("*"))
        if path.is_file()
    }
    output = tmp_path / "runtime-stack.json"
    result = RUNNER.invoke(
        app,
        [
            "apply",
            str(base),
            str(second.path),
            str(first.path),
            "--output",
            str(output),
            "--mode",
            "runtime",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    record = json.loads(output.read_text(encoding="utf-8"))
    expected_order = sorted((first.manifest.patch_id, second.manifest.patch_id))
    assert payload["status"] == record["status"] == "PASS"
    assert record["artifact_kind"] == "RUNTIME_STACK_EXECUTION_V1"
    assert record["execution_id"] == hash_canonical(
        {key: value for key, value in record.items() if key != "execution_id"}
    )
    assert record["mode"] == "runtime"
    assert record["ephemeral"] is True
    assert record["persistent"] is False
    assert record["mount"]["executed"] is True
    assert record["mount"]["tensors_match_base_plus_resolved_delta"] is True
    assert all(check["matches_expected"] for check in record["mount"]["target_checks"])
    assert record["unmount"]["executed"] is True
    assert record["unmount"]["base_state_bitwise_restored"] is True
    assert record["unmount"]["grade"] == "RUNTIME_UNMOUNT_EXACT"
    assert record["source_checkpoint_identity_unchanged"] is True
    assert record["patch_order"] == expected_order
    assert "unmounted before command exit" in record["warning"]

    second_output = tmp_path / "runtime-stack-reordered.json"
    reordered = RUNNER.invoke(
        app,
        [
            "apply",
            str(base),
            str(first.path),
            str(second.path),
            "--output",
            str(second_output),
            "--mode",
            "runtime",
            "--json",
        ],
    )
    assert reordered.exit_code == 0, reordered.stdout
    assert second_output.read_bytes() == output.read_bytes()
    source_after = {
        path.relative_to(base).as_posix(): sha256_file(path)
        for path in sorted(base.rglob("*"))
        if path.is_file()
    }
    assert source_after == source_before


def test_extraction_authenticates_diff_artifacts_and_model_identities(tmp_path: Path) -> None:
    checkpoint = tmp_path / "tiny"
    save_tiny_checkpoint(
        TinyCausalLM(
            TinyConfig(
                hidden_size=8,
                intermediate_size=8,
                num_layers=1,
                num_heads=2,
                max_sequence_length=16,
            )
        ),
        checkpoint,
    )
    adapter = TinyModelAdapter()
    model = adapter.load(str(checkpoint), device="cpu", dtype=torch.float32)
    manifest = build_model_manifest(model, checkpoint=checkpoint, adapter_id=adapter.adapter_id)
    bundle = tmp_path / "diff"
    bundle.mkdir()
    (bundle / "clusters.json").write_text("[]", encoding="utf-8")
    (bundle / "witnesses.parquet").write_bytes(b"bounded-fixture")
    diff_manifest = {
        "artifact_hashes": {
            "clusters.json": sha256_file(bundle / "clusters.json"),
            "witnesses.parquet": sha256_file(bundle / "witnesses.parquet"),
        },
        "configuration": {
            "base_signature": manifest.signature.to_dict(),
            "target_signature": manifest.signature.to_dict(),
        },
        "schema_version": 1,
    }
    (bundle / "manifest.json").write_text(
        json.dumps(diff_manifest, sort_keys=True), encoding="utf-8"
    )
    assert (
        _verified_diff_manifest(
            bundle,
            base_manifest=manifest,
            target_manifest=manifest,
        )["schema_version"]
        == 1
    )

    (bundle / "witnesses.parquet").write_bytes(b"mutated")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        _verified_diff_manifest(bundle, base_manifest=manifest, target_manifest=manifest)
    (bundle / "witnesses.parquet").write_bytes(b"bounded-fixture")
    malformed_hash = json.loads(json.dumps(diff_manifest))
    malformed_hash["artifact_hashes"]["clusters.json"] = "sha256:" + "A" * 64
    (bundle / "manifest.json").write_text(
        json.dumps(malformed_hash, sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="malformed artifact hashes"):
        _verified_diff_manifest(bundle, base_manifest=manifest, target_manifest=manifest)
    wrong = json.loads(json.dumps(diff_manifest))
    wrong["configuration"]["base_signature"]["checkpoint_hash"] = "sha256:" + "f" * 64
    (bundle / "manifest.json").write_text(json.dumps(wrong, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="base signature"):
        _verified_diff_manifest(bundle, base_manifest=manifest, target_manifest=manifest)


def test_verify_executes_every_distinct_bundled_contract(tmp_path: Path) -> None:
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
    target_value = json.loads(
        json.dumps(_executable_contract(manifest, "target-contract").to_dict())
    )
    target_value["verify"]["guards"] = []
    target_contract = parse_contract(target_value)
    preservation_value = json.loads(
        json.dumps(_executable_contract(manifest, "preservation-contract").to_dict())
    )
    preservation_value["verify"]["targets"] = []
    preservation_contract = parse_contract(preservation_value)
    patch = tmp_path / "patch"
    create_patch_bundle(
        patch,
        name="two-contract-patch",
        base_signature=manifest.signature.to_dict(),
        state_schema=manifest.state_schema,
        program=DeltaProgram({"final_norm.weight": VectorDelta("zero")}),
        tensors={"zero": torch.zeros(8)},
        tool_version=__version__,
        contracts={
            "contracts/guards/validation.jsonl": b'{"id":"g","prompt":"x"}\n',
            "contracts/preservation.yaml": (
                json.dumps(preservation_contract.to_dict(), sort_keys=True) + "\n"
            ).encode(),
            "contracts/probes/train.jsonl": b'{"id":"t","prompt":"x","target":"a"}\n',
            "contracts/probes/validation.jsonl": b'{"id":"v","prompt":"x"}\n',
            "contracts/target.yaml": (
                json.dumps(target_contract.to_dict(), sort_keys=True) + "\n"
            ).encode(),
        },
        provides=(target_contract.contract_id,),
        preserves=(preservation_contract.contract_id,),
    )

    result = RUNNER.invoke(
        app,
        [
            "verify",
            str(patch),
            "--base",
            str(checkpoint),
            "--adapter",
            "tiny",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "PASS"
    assert len(payload["reports"]) == 2
    assert {item["contract_id"] for item in payload["reports"]} == {
        "target-contract",
        "preservation-contract",
    }
    reports_by_id = {item["contract_id"]: item for item in payload["reports"]}
    assert len(reports_by_id["target-contract"]["target_results"]) == 1
    assert reports_by_id["target-contract"]["guard_results"] == []
    assert len(reports_by_id["preservation-contract"]["guard_results"]) == 1
    assert reports_by_id["preservation-contract"]["target_results"] == []
    assert len(payload["certificates"]) == 1
    assert set(payload["certificate"]["contract_hashes"]) == {
        "preservation-contract",
        "target-contract",
    }
    assert "report" not in payload


def test_verify_policy_cannot_replace_failing_bundled_claims(tmp_path: Path) -> None:
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
    manifest = build_model_manifest(loaded, checkpoint=checkpoint, adapter_id=adapter.adapter_id)

    failing_value = json.loads(
        json.dumps(_executable_contract(manifest, "bundled-failure").to_dict())
    )
    failing_value["verify"]["targets"][0]["minimum"] = 2
    failing_value["verify"]["targets"][0]["maximum"] = 2
    failing_contract = parse_contract(failing_value)
    patch = tmp_path / "patch"
    _zero_bundle(patch, manifest, failing_contract)

    policy = _executable_contract(manifest, "additional-policy")
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy.to_dict(), sort_keys=True), encoding="utf-8")
    (tmp_path / "probes").mkdir()
    (tmp_path / "guards").mkdir()
    (tmp_path / "probes" / "validation.jsonl").write_text(
        '{"id":"v","prompt":"x"}\n', encoding="utf-8"
    )
    (tmp_path / "guards" / "validation.jsonl").write_text(
        '{"id":"g","prompt":"x"}\n', encoding="utf-8"
    )

    result = RUNNER.invoke(
        app,
        [
            "verify",
            str(patch),
            "--base",
            str(checkpoint),
            "--adapter",
            "tiny",
            "--policy",
            str(policy_path),
            "--json",
        ],
    )

    assert result.exit_code == 2, result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "FAIL"
    assert {item["contract_id"] for item in payload["reports"]} == {
        "additional-policy",
        "bundled-failure",
    }
    assert set(payload["certificate"]["contract_hashes"]) == {
        "additional-policy",
        "bundled-failure",
    }


def test_benchmark_runs_a_real_exhaustive_experiment() -> None:
    result = RUNNER.invoke(app, ["benchmark", "closure_matrix", "--json"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "PASS"
    assert payload["result"]["executed_subsets"] == 63
    assert payload["result"]["search_space_exhausted"] is True


def test_benchmark_propagates_a_structured_failure(monkeypatch: MonkeyPatch) -> None:
    import modelpact.modelpactbench.runner as benchmark_runner

    monkeypatch.setattr(
        benchmark_runner,
        "run_selected",
        lambda _name: {"status": "FAIL", "success": False, "failed_stage": "verification"},
    )
    result = RUNNER.invoke(app, ["benchmark", "closure_matrix", "--json"])
    assert result.exit_code == 2
    payload = json.loads(result.stdout)
    assert payload["status"] == "FAIL"
    assert payload["result"]["failed_stage"] == "verification"


def test_benchmark_terminal_fields_fail_closed() -> None:
    from modelpact.modelpactbench.runner import benchmark_succeeded

    with pytest.raises(ValueError, match="success field"):
        benchmark_succeeded({"executed_subsets": 63})
    assert benchmark_succeeded({"status": "PASS", "success": True}) is True
    assert benchmark_succeeded({"status": "PASS", "success": False}) is False
    assert benchmark_succeeded({"status": "FAIL", "success": True}) is False


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
    merged_bundle = load_patch_bundle(
        tmp_path / "merged",
        state_schema=manifest.state_schema,
    )
    assert merged_bundle.manifest.patch_id == payload["patch_id"]
    assert payload["artifact_kind"] == "BEHAVIOR_PATCH_BUNDLE_V1"


def test_tiny_semantic_merge_executes_union_cegis_and_minimization(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "tiny-behavioral-base"
    model = case_brittle_model()
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
        "left-portable-output",
        expected="Q",
    )
    right_contract = searchable_behavior_contract(
        manifest,
        "right-portable-output",
        expected="Q",
    )
    left = brittle_case_behavior_bundle(tmp_path / "left-behavior", manifest, left_contract)
    right = brittle_case_behavior_bundle(tmp_path / "right-behavior", manifest, right_contract)
    output = tmp_path / "semantic-merge"

    result = RUNNER.invoke(
        app,
        [
            "merge",
            str(left.path),
            str(right.path),
            "--base",
            str(checkpoint),
            "--output",
            str(output),
            "--adapter",
            "tiny",
            "--force-recompile",
            "--maximum-steps",
            "200",
            "--max-rank",
            "1",
            "--max-modules",
            "1",
            "--cegis-rounds",
            "1",
            "--cegis-search-budget",
            "4",
            "--minimization-budget",
            "6",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["disposition"] == "SEMANTIC_MERGE_VERIFIED"
    assert payload["cegis"]["rounds_executed"] == 1
    assert payload["cegis"]["model_executions"] > 0
    assert len(payload["cegis"]["compilation_candidates"]) == 2
    assert (
        sum(
            len(round_result["target_counterexamples"]) + len(round_result["guard_counterexamples"])
            for round_result in payload["cegis"]["rounds"]
        )
        >= 1
    )
    assert all(execution["proposals"] for execution in payload["cegis"]["search_executions"])
    assert "UNMINIMIZED" not in payload["minimization"]["claims"]
    assert payload["minimization"]["verification_budget_used"] >= 1
    assert payload["holdout_execution_counts"] == {
        left_contract.contract_id: 1,
        right_contract.contract_id: 1,
    }
    compile_evidence = json.loads(
        (output / "evidence" / "compile.json").read_text(encoding="utf-8")
    )
    assert compile_evidence["cegis"]["search_executions"]
    minimization = json.loads(
        (output / "evidence" / "minimization.json").read_text(encoding="utf-8")
    )
    assert minimization["candidates"][0]["operation"] == "verify:initial"
    assert minimization["candidates"][0]["candidate_id"].startswith("sha256:")
    merged = load_patch_bundle(output, state_schema=manifest.state_schema)
    with mount_patch(
        loaded,
        merged.program,
        merged.tensors,
        state_schema=manifest.state_schema,
    ):
        assert (
            adapter.generate(
                loaded,
                adapter.tokenizer().batch(["x"]),
                AdapterGenerationPolicy(max_new_tokens=1),
            )[0].text
            == "Q"
        )


def test_compose_emits_executed_baselines_and_explicit_interaction_evidence(
    tmp_path: Path,
) -> None:
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

    def structured_bundle(path: Path, contract: BehaviorContract, sign: float) -> PatchBundle:
        contract_bytes = (json.dumps(contract.to_dict(), sort_keys=True) + "\n").encode()
        return create_patch_bundle(
            path,
            name=contract.id,
            base_signature=manifest.signature.to_dict(),
            state_schema=manifest.state_schema,
            program=DeltaProgram(
                {
                    "layers.0.attention.q_proj.weight": Sum(
                        (
                            LowRankMatrixDelta("b", "a"),
                            SparseMatrixDelta("indices", "values", (8, 8)),
                        )
                    )
                }
            ),
            tensors={
                "a": torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
                "b": torch.tensor([[sign * 1e-6], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]),
                "indices": torch.tensor([[1, 1]], dtype=torch.int64),
                "values": torch.tensor([sign * 1e-6]),
            },
            tool_version=__version__,
            contracts={
                "contracts/guards/validation.jsonl": (
                    json.dumps(
                        {"id": f"g-{path.name}", "prompt": "x"},
                        sort_keys=True,
                    )
                    + "\n"
                ).encode(),
                "contracts/preservation.yaml": contract_bytes,
                "contracts/probes/train.jsonl": b'{"id":"t","prompt":"x","target":"a"}\n',
                "contracts/probes/validation.jsonl": (
                    json.dumps(
                        {"id": f"v-{path.name}", "prompt": "x"},
                        sort_keys=True,
                    )
                    + "\n"
                ).encode(),
                "contracts/target.yaml": contract_bytes,
            },
            provides=(contract.contract_id,),
            preserves=(contract.contract_id,),
        )

    left = structured_bundle(
        tmp_path / "left", _executable_contract(manifest, "left-behavior"), 1.0
    )
    right = structured_bundle(
        tmp_path / "right", _executable_contract(manifest, "right-behavior"), -1.0
    )
    output = tmp_path / "composition"

    result = RUNNER.invoke(
        app,
        [
            "compose",
            str(left.path),
            str(right.path),
            "--base",
            str(checkpoint),
            "--output",
            str(output),
            "--adapter",
            "tiny",
            "--degradation-tolerance",
            "0.1",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["claim"] == "COMPOSITION_CLOSED"
    assert payload["base_verification"]["margins"]
    assert set(payload["singleton_verifications"]) == {
        left.manifest.patch_id,
        right.manifest.patch_id,
    }
    assert len(payload["interaction_margins"]) == 2
    composite = load_patch_bundle(output, state_schema=manifest.state_schema)
    assert composite.manifest.patch_id == payload["patch_id"]
    assert payload["artifact_kind"] == "BEHAVIOR_PATCH_BUNDLE_V1"
    interactions = json.loads(
        (output / "evidence" / "interactions.json").read_text(encoding="utf-8")
    )
    assert interactions["contract_margin_interactions"]["status"] == "AVAILABLE"
    assert len(interactions["contract_margin_interactions"]["values"]) == 2
    pair = interactions["pairwise_diagnostics"][0]
    assert pair["module_overlap"]["jaccard"] == 1.0
    assert pair["sparse_index_overlap"]["status"] == "AVAILABLE"
    assert pair["sparse_index_overlap"]["values"][0]["jaccard"] == 1.0
    assert pair["low_rank_subspace"]["status"] == "AVAILABLE"
    principal_angles = pair["low_rank_subspace"]["values"][0]["principal_angles"]
    assert principal_angles["column_space"]["radians"][0] == 0.0
    assert principal_angles["row_space"]["radians"][0] == 0.0
    assert pair["gradient_cosine_similarity"]["status"] == "NOT_AVAILABLE"
    assert pair["activation_delta_similarity"]["status"] == "NOT_AVAILABLE"
    assert pair["output_interaction_residual"]["status"] == "NOT_AVAILABLE"
    assert interactions["pairwise_parameter_overlap"][0]["module_overlap"]["jaccard"] == 1.0

    applied = RUNNER.invoke(
        app,
        [
            "apply",
            str(checkpoint),
            str(output),
            "--output",
            str(tmp_path / "materialized-composite"),
            "--adapter",
            "tiny",
            "--json",
        ],
    )
    assert applied.exit_code == 0, applied.stdout

    standalone = subprocess.run(  # noqa: S603 - generated script and exact interpreter
        [
            sys.executable,
            str(output / "verify_patch.py"),
            str(checkpoint),
            "--adapter-kind",
            "tiny",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert standalone.returncode == 0, standalone.stdout + standalone.stderr
    assert json.loads(standalone.stdout)["outcome"] == "PASS"


def test_compose_packages_one_successful_sealed_holdout_execution(tmp_path: Path) -> None:
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
    generated = adapter.generate(
        loaded,
        adapter.tokenizer().batch(["x"]),
        AdapterGenerationPolicy(max_new_tokens=1),
    )[0].text
    patch = _holdout_bundle(
        tmp_path / "patch",
        manifest,
        visible_expected=generated,
        holdout_expected=generated,
    )
    output = tmp_path / "composition"

    result = RUNNER.invoke(
        app,
        [
            "compose",
            str(patch.path),
            "--base",
            str(checkpoint),
            "--output",
            str(output),
            "--adapter",
            "tiny",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["holdout_execution_counts"] == {patch.manifest.provides[0]: 1}
    composite = load_patch_bundle(output, state_schema=manifest.state_schema)
    assert (output / "contracts" / "holdout" / "targets.jsonl").is_file()
    certificate = json.loads((output / "certificate.json").read_text(encoding="utf-8"))
    assert certificate["sealed_holdout_result"]["outcome"] == "PASS"
    assert composite.manifest.patch_id == payload["patch_id"]


def test_audit_retains_distinct_execution_evidence_for_identical_resolved_deltas(
    tmp_path: Path,
) -> None:
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
    pool = tmp_path / "pool"
    _zero_bundle(
        pool / "left",
        manifest,
        _executable_contract(manifest, "audit-left-behavior"),
    )
    _zero_bundle(
        pool / "right",
        manifest,
        _executable_contract(manifest, "audit-right-behavior"),
    )
    output = tmp_path / "audit"

    result = RUNNER.invoke(
        app,
        [
            "audit",
            "--base",
            str(checkpoint),
            "--patch-dir",
            str(pool),
            "--output",
            str(output),
            "--adapter",
            "tiny",
            "--subset-budget",
            "3",
            "--exhaustive-threshold",
            "2",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    executed = json.loads((output / "executed-verification.json").read_text(encoding="utf-8"))
    subset_execution_ids = [key for key in executed if ":audit:subset:" in key]
    assert len(subset_execution_ids) == 3
    assert len(set(subset_execution_ids)) == 3
    assert all(":contracts-" in key for key in subset_execution_ids)

    audit = json.loads((output / "audit.json").read_text(encoding="utf-8"))["audit"]
    evaluations = audit["evaluations"]
    assert len(evaluations) == 3
    assert all(item["result_hash"].startswith("sha256:") for item in evaluations)
    assert len({item["result_hash"] for item in evaluations}) == 3
    assert all(item["metadata"]["execution_evidence_id"] in executed for item in evaluations)


def test_audit_returns_failed_exit_code_when_a_failing_subset_is_found(
    tmp_path: Path,
) -> None:
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
    contract_value = _executable_contract(manifest, "audit-failing-behavior").to_dict()
    target = contract_value["verify"]["targets"][0]
    target["minimum"] = 2
    target["maximum"] = 2
    pool = tmp_path / "pool"
    _zero_bundle(
        pool / "failing",
        manifest,
        parse_contract(contract_value),
    )
    output = tmp_path / "audit"

    result = RUNNER.invoke(
        app,
        [
            "audit",
            "--base",
            str(checkpoint),
            "--patch-dir",
            str(pool),
            "--output",
            str(output),
            "--adapter",
            "tiny",
            "--subset-budget",
            "1",
            "--exhaustive-threshold",
            "1",
            "--json",
        ],
    )

    assert result.exit_code == EXIT_FAILED, result.stdout
    payload = json.loads(result.stdout)
    assert "FAILING_SUBSET_FOUND" in payload["status"]
    assert payload["audit"]["failing_subsets"]
    assert (output / "audit.json").is_file()


@pytest.mark.parametrize("command", ["compose", "merge"])
def test_composition_final_holdout_failure_is_executed_once_and_rejects_candidate(
    tmp_path: Path,
    command: str,
) -> None:
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
    generated = adapter.generate(
        loaded,
        adapter.tokenizer().batch(["x"]),
        AdapterGenerationPolicy(max_new_tokens=1),
    )[0].text
    patch = _holdout_bundle(
        tmp_path / "patch",
        manifest,
        visible_expected=generated,
    )
    output = tmp_path / "failed-composition"

    result = RUNNER.invoke(
        app,
        [
            command,
            str(patch.path),
            "--base",
            str(checkpoint),
            "--output",
            str(output),
            "--adapter",
            "tiny",
            "--json",
        ],
    )

    assert result.exit_code == 2, result.stdout
    payload = json.loads(result.stdout)
    assert payload["claim"] == "SEMANTIC_CONFLICT"
    assert payload["artifact_kind"] == "COMPOSITION_FAILURE_EVIDENCE_V1"
    assert payload["holdout_execution_counts"] == {patch.manifest.provides[0]: 1}
    if command == "merge":
        assert payload["disposition"] == "FINAL_CANDIDATE_FAILED_HOLDOUT"
    evidence = json.loads((output / "verification.json").read_text(encoding="utf-8"))
    report = evidence["verification"]["reports"][patch.manifest.provides[0]]
    assert report["holdout_outcome"] == "FAIL"


def test_merge_detects_probe_backed_exact_output_contradiction(tmp_path: Path) -> None:
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

    def make_contract(identifier: str) -> BehaviorContract:
        return parse_contract(
            {
                "compile": {"objectives": []},
                "contract_version": 1,
                "generation": {"max_new_tokens": 1, "mode": "greedy", "seeds": [0]},
                "holdout": {"sealed": True, "unseal_policy": "final_candidate_only"},
                "id": identifier,
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
                            "id": "exact-output",
                            "minimum_pass_rate": 1.0,
                            "source": "probes/validation.jsonl",
                            "type": "exact_match",
                        }
                    ],
                },
            }
        )

    def make_bundle(path: Path, contract: BehaviorContract, expected: str) -> PatchBundle:
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
                "contracts/guards/validation.jsonl": b'{"id":"g","prompt":"control"}\n',
                "contracts/preservation.yaml": contract_bytes,
                "contracts/probes/validation.jsonl": (
                    json.dumps(
                        {"expected": expected, "id": "shared", "prompt": "same prompt"},
                        sort_keys=True,
                    )
                    + "\n"
                ).encode(),
                "contracts/target.yaml": contract_bytes,
            },
            provides=(contract.contract_id,),
            preserves=(contract.contract_id,),
        )

    left_contract = make_contract("requires-left-output")
    right_contract = make_contract("requires-right-output")
    left = make_bundle(tmp_path / "left", left_contract, "left")
    right = make_bundle(tmp_path / "right", right_contract, "right")
    output = tmp_path / "merged"

    result = RUNNER.invoke(
        app,
        [
            "merge",
            str(left.path),
            str(right.path),
            "--base",
            str(checkpoint),
            "--output",
            str(output),
            "--adapter",
            "tiny",
            "--maximum-steps",
            "2",
            "--json",
        ],
    )

    assert result.exit_code == 2, result.stdout
    payload = json.loads(result.stdout)
    assert payload["disposition"] == "STATIC_CONTRACT_CONTRADICTION"
    assert payload["compiler_invoked"] is False
    witnesses = payload["naive_composition"]["contradictions"]
    assert len(witnesses) == 1
    assert witnesses[0]["code"] == "INCOMPATIBLE_EXACT_REQUIREMENTS"
    assert witnesses[0]["contract_ids"] == [
        "requires-left-output",
        "requires-right-output",
    ]


def test_cross_architecture_tiny_rebase_recompiles_and_packages(tmp_path: Path) -> None:
    source_checkpoint = tmp_path / "source"
    target_checkpoint = tmp_path / "target"
    source_model = constant_output_model(
        hidden_size=8,
        num_heads=2,
        output="A",
    )
    target_model = constant_output_model(
        hidden_size=12,
        num_heads=3,
        output="B",
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
    contract = searchable_behavior_contract(
        manifest,
        "portable-learned-behavior",
        expected="Q",
    )
    patch = learned_behavior_bundle(
        tmp_path / "source-patch",
        manifest,
        contract,
        source_output="A",
        target_output="Q",
    )
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
            "200",
            "--max-rank",
            "1",
            "--max-modules",
            "1",
            "--cegis-rounds",
            "1",
            "--cegis-search-budget",
            "4",
            "--minimization-budget",
            "4",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["claim"] == "SEMANTIC_REBASE_VERIFIED"
    assert payload["disposition"] == "SEMANTIC_REBASE_VERIFIED"
    assert payload["optimization_steps"] == 200
    assert payload["cegis"]["model_executions"] > 0
    assert payload["cegis"]["compilation_candidates"]
    assert payload["cegis"]["search_executions"][0]["proposals"]
    assert "UNMINIMIZED" not in payload["minimization"]["claims"]
    rebased_manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert rebased_manifest["rebased_from"] == patch.manifest.patch_id
    assert rebased_manifest["compiler_configuration"]["mode"] == "semantic_recompile"
    assert (output / "certificate.json").is_file()
    minimization = json.loads(
        (output / "evidence" / "minimization.json").read_text(encoding="utf-8")
    )
    assert minimization["verification_budget_used"] >= 1
    assert minimization["candidates"][0]["operation"] == "verify:initial"
    assert minimization["candidates"][0]["candidate_id"].startswith("sha256:")
    assert "UNMINIMIZED" not in minimization["claims"]
    target_loaded = adapter.load(
        str(target_checkpoint),
        device="cpu",
        dtype=torch.float32,
    )
    assert (
        adapter.generate(
            target_loaded,
            adapter.tokenizer().batch(["x"]),
            AdapterGenerationPolicy(max_new_tokens=1),
        )[0].text
        == "B"
    )
    rebased_bundle = load_patch_bundle(
        output,
        state_schema=adapter.state_schema(target_loaded),
    )
    with mount_patch(
        target_loaded,
        rebased_bundle.program,
        rebased_bundle.tensors,
        state_schema=adapter.state_schema(target_loaded),
    ):
        assert (
            adapter.generate(
                target_loaded,
                adapter.tokenizer().batch(["x"]),
                AdapterGenerationPolicy(max_new_tokens=1),
            )[0].text
            == "Q"
        )


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
    assert "unsatisfied required contracts" in payload["error"]
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
                "guards": [
                    {
                        "id": "preserve-stack-control",
                        "maximum_mean": 10.0,
                        "source": "data/guards.jsonl",
                        "type": "base_kl",
                    }
                ],
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
            "contracts/data/guards.jsonl": b'{"id":"g","prompt":"control"}\n',
            "contracts/data/validation.jsonl": b'{"id":"p","prompt":"x"}\n',
            "contracts/preservation.yaml": contract_bytes,
            "contracts/target.yaml": contract_bytes,
        },
        provides=(contract.contract_id,),
        preserves=(contract.contract_id,),
    )
    remaining_value = json.loads(json.dumps(contract.to_dict()))
    remaining_value["id"] = "remaining-stack-contract"
    remaining_contract = parse_contract(remaining_value)
    remaining_contract_bytes = (
        json.dumps(remaining_contract.to_dict(), sort_keys=True) + "\n"
    ).encode()
    remaining_patch = tmp_path / "remaining-patch"
    remaining_bundle = create_patch_bundle(
        remaining_patch,
        name="remaining-zero-stack-patch",
        base_signature=manifest.signature.to_dict(),
        state_schema=manifest.state_schema,
        program=DeltaProgram({"final_norm.weight": VectorDelta("zero")}),
        tensors={"zero": torch.zeros(8)},
        tool_version=__version__,
        contracts={
            "contracts/data/guards.jsonl": b'{"id":"g","prompt":"control"}\n',
            "contracts/data/validation.jsonl": b'{"id":"p","prompt":"x"}\n',
            "contracts/preservation.yaml": remaining_contract_bytes,
            "contracts/target.yaml": remaining_contract_bytes,
        },
        provides=(remaining_contract.contract_id,),
        preserves=(remaining_contract.contract_id,),
    )
    stack_spec = tmp_path / "stack.toml"
    stack_spec.write_text(
        "\n".join(
            (
                "schema_version = 1",
                f'base = "{checkpoint.as_posix()}"',
                (f'patches = ["{patch.as_posix()}", "{remaining_patch.as_posix()}"]'),
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

    tokenizer_config = checkpoint / "tokenizer_config.json"
    original_tokenizer_config = tokenizer_config.read_bytes()
    tokenizer_value = json.loads(original_tokenizer_config)
    tokenizer_value["chat_template"] = "changed after stack resolution"
    tokenizer_config.write_text(json.dumps(tokenizer_value), encoding="utf-8")
    rejected_remaining_path = tmp_path / "rejected-remaining-stack"
    rejected_remaining = RUNNER.invoke(
        app,
        [
            "revert",
            str(lockfile),
            "--remove",
            bundle.manifest.patch_id,
            "--output",
            str(rejected_remaining_path),
            "--adapter",
            "tiny",
            "--json",
        ],
    )
    assert rejected_remaining.exit_code != 0
    rejected_remaining_payload = json.loads(rejected_remaining.stdout)
    assert "base model manifest hash no longer matches" in rejected_remaining_payload["error"]
    assert not rejected_remaining_path.exists()
    tokenizer_config.write_bytes(original_tokenizer_config)

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
    assert result["reversion_grade"] == "VERIFIED_LOGICAL_STACK_RECONSTRUCTED"
    assert set(result["lock"]["patch_hashes"]) == {remaining_bundle.manifest.patch_id}
    assert result["reversion_grade"] not in {
        "BASE_HASH_RESTORED",
        "NUMERIC_DELTA_INVERSE",
        "RUNTIME_UNMOUNT_EXACT",
        "SEMANTIC_STACK_RECOMPILED",
    }

    generation_config = checkpoint / "generation_config.json"
    original_generation_config = generation_config.read_bytes()
    generation_value = json.loads(original_generation_config)
    generation_value["do_sample"] = True
    generation_config.write_text(json.dumps(generation_value), encoding="utf-8")
    rejected_empty_path = tmp_path / "rejected-empty-stack"
    rejected_empty = RUNNER.invoke(
        app,
        [
            "revert",
            str(reverted_path / "stack.lock.json"),
            "--remove",
            remaining_bundle.manifest.patch_id,
            "--output",
            str(rejected_empty_path),
            "--adapter",
            "tiny",
            "--json",
        ],
    )
    assert rejected_empty.exit_code != 0
    rejected_empty_payload = json.loads(rejected_empty.stdout)
    assert "base model manifest hash no longer matches" in rejected_empty_payload["error"]
    assert not rejected_empty_path.exists()
    generation_config.write_bytes(original_generation_config)

    empty_path = tmp_path / "empty-stack"
    empty = RUNNER.invoke(
        app,
        [
            "revert",
            str(reverted_path / "stack.lock.json"),
            "--remove",
            remaining_bundle.manifest.patch_id,
            "--output",
            str(empty_path),
            "--adapter",
            "tiny",
            "--json",
        ],
    )
    assert empty.exit_code == 0, empty.stdout
    empty_result = json.loads(empty.stdout)
    assert empty_result["reversion_grade"] == "BASE_HASH_RESTORED"
    assert empty_result["lock"]["patch_hashes"] == {}


def test_compile_executes_bounded_cegis_before_minimization_and_holdout(
    tmp_path: Path,
) -> None:
    torch.set_num_threads(1)
    checkpoint = tmp_path / "tiny-cegis"
    tokenizer = TinyTokenizer()
    model = TinyCausalLM(
        TinyConfig(
            hidden_size=16,
            intermediate_size=32,
            num_layers=1,
            num_heads=2,
            max_sequence_length=24,
            initialization_seed=107,
        )
    )
    train_tiny_causal_lm(
        model,
        (
            *("F:aR" for _ in range(8)),
            *("F:bB" for _ in range(8)),
            *("C:q1" for _ in range(8)),
        ),
        tokenizer=tokenizer,
        config=TinyTrainingConfig(
            steps=160,
            batch_size=24,
            learning_rate=0.02,
            seed=109,
        ),
    )
    save_tiny_checkpoint(model, checkpoint, tokenizer=tokenizer)
    adapter = TinyModelAdapter(tokenizer)
    manifest = build_model_manifest(
        model,
        checkpoint=checkpoint,
        adapter_id=adapter.adapter_id,
        architecture_config=model.config.to_dict(),
    )
    specification = tmp_path / "contract.yaml"
    specification.write_text(
        json.dumps(
            {
                "compile": {
                    "objectives": [
                        {
                            "id": "teach-update",
                            "source": "probes/train.jsonl",
                            "type": "teacher_cross_entropy",
                        }
                    ]
                },
                "contract_version": 1,
                "generation": {
                    "max_new_tokens": 1,
                    "mode": "greedy",
                    "seeds": [0],
                },
                "holdout": {
                    "guards": "holdout/guards.jsonl",
                    "sealed": True,
                    "targets": "holdout/targets.jsonl",
                    "unseal_policy": "final_candidate_only",
                },
                "id": "tiny-generic-cegis",
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
                    "bootstrap_samples": 16,
                    "bootstrap_seed": 113,
                    "confidence_level": 0.95,
                },
                "verify": {
                    "guards": [
                        {
                            "id": "preserve-neighbors",
                            "maximum_item": 5.0,
                            "maximum_mean": 5.0,
                            "source": "guards/validation.jsonl",
                            "type": "base_kl",
                        }
                    ],
                    "targets": [
                        {
                            "id": "generate-update",
                            "minimum_pass_rate": 1.0,
                            "source": "probes/validation.jsonl",
                            "type": "free_generation_match",
                        }
                    ],
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    resources = {
        "probes/train.jsonl": {"id": "train", "prompt": "F:a", "target": "G"},
        "probes/validation.jsonl": {
            "expected": "G",
            "id": "validation",
            "prompt": "F:a",
        },
        "guards/validation.jsonl": {"id": "guard", "prompt": "F:b"},
        "holdout/targets.jsonl": {
            "expected": "G",
            "id": "sealed-target",
            "prompt": "F:a",
        },
        "holdout/guards.jsonl": {"id": "sealed-guard", "prompt": "C:q"},
    }
    for relative, record in resources.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    output = tmp_path / "compiled"

    invocation = RUNNER.invoke(
        app,
        [
            "compile",
            "--base",
            "tiny",
            "--checkpoint",
            str(checkpoint),
            "--spec",
            str(specification),
            "--output",
            str(output),
            "--max-rank",
            "2",
            "--max-modules",
            "4",
            "--steps",
            "140",
            "--cegis-rounds",
            "1",
            "--cegis-search-budget",
            "5",
            "--minimization-budget",
            "6",
            "--seed",
            "127",
            "--json",
        ],
    )

    assert invocation.exit_code == 0, invocation.stdout
    payload = json.loads(invocation.stdout)
    assert payload["status"] == "PASS"
    assert payload["holdout_outcome"] == "PASS"
    cegis = payload["cegis"]
    assert cegis["rounds_executed"] > 0
    assert cegis["model_executions"] > 0
    assert cegis["search_budget_per_domain_per_round"] == 5
    assert cegis["post_minimization_working_set_passed"] is True
    assert any(round_result["target_counterexamples"] for round_result in cegis["rounds"])
    assert all(
        "sealed" not in example["record_id"]
        for example in (
            *cegis["working_target_examples"],
            *cegis["working_guard_examples"],
        )
    )
    compile_evidence = json.loads(
        (output / "evidence" / "compile.json").read_text(encoding="utf-8")
    )
    assert compile_evidence["cegis"]["post_minimization_working_set_passed"] is True
    compiled_contract = parse_contract(json.loads(specification.read_text(encoding="utf-8")))
    bundle = load_patch_bundle(output)
    assert bundle.manifest.provides == (compiled_contract.contract_id,)
    assert bundle.manifest.preserves == (compiled_contract.contract_id,)
    certificate = json.loads((output / "certificate.json").read_text(encoding="utf-8"))
    assert certificate["counterexample_search"]["rounds_executed"] > 0

    unsupported_value = json.loads(specification.read_text(encoding="utf-8"))
    unsupported_value["id"] = "tiny-unsupported-cegis-search"
    unsupported_value["verify"]["targets"] = [
        {
            "id": "unsupported-length-search",
            "maximum": 1,
            "minimum": 1,
            "source": "probes/validation.jsonl",
            "type": "generation_length",
        }
    ]
    unsupported_spec = tmp_path / "unsupported-contract.yaml"
    unsupported_spec.write_text(
        json.dumps(unsupported_value, sort_keys=True),
        encoding="utf-8",
    )
    unsupported = RUNNER.invoke(
        app,
        [
            "compile",
            "--base",
            "tiny",
            "--checkpoint",
            str(checkpoint),
            "--spec",
            str(unsupported_spec),
            "--output",
            str(tmp_path / "unsupported-output"),
            "--cegis-rounds",
            "1",
            "--json",
        ],
    )
    assert unsupported.exit_code != 0
    unsupported_payload = json.loads(unsupported.stdout)
    assert unsupported_payload["status"] == "UNSUPPORTED"
    assert "unsupported-length-search" in unsupported_payload["reason"]
