from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import modelpact.cli as cli_module
from modelpact.cli import _read_lock, _verify_locked_patch_manifests, app
from modelpact.compose.stack import (
    MAX_STACK_LOCK_BYTES,
    MAX_STACK_LOCK_CONTRACTS,
    MAX_STACK_LOCK_NODES,
    MAX_STACK_LOCK_OBJECT_KEYS,
    MAX_STACK_LOCK_PATCHES,
    MAX_STACK_LOCK_STRING_LENGTH,
    StackLock,
)
from modelpact.contracts import BehaviorContract, EvaluationRecord, canonical_contract_json
from modelpact.contracts.parser import DEFAULT_LIMITS, parse_contract
from modelpact.models.manifest import ModelSignature
from modelpact.patch.bundle import MAX_MANIFEST_BYTES
from modelpact.rebase.direct import RebaseCompatibility
from modelpact.rebase.evidence import RebaseEvidence
from modelpact.status import RebaseClaim, VerificationOutcome
from modelpact.util.canonical_json import canonical_dumps
from modelpact.util.hashing import hash_canonical, sha256_file
from modelpact.verify import (
    ExecutionIdentity,
    MappingRecordProvider,
    VerificationCertificate,
    build_certificate,
    verify_contract,
)

RUNNER = CliRunner()
_STACK_VERIFICATION_POLICY = {"kind": "stack-lock-security-test"}
_STACK_VERIFICATION_POLICY_HASH = hash_canonical(_STACK_VERIFICATION_POLICY)


def _digest(index: int) -> str:
    return f"sha256:{index:064x}"


def _canonical_bytes(value: object) -> bytes:
    return (canonical_dumps(value) + "\n").encode("utf-8")


def _write_canonical(path: Path, value: object) -> None:
    path.write_bytes(_canonical_bytes(value))


def _behavior_contract(
    name: str,
    *,
    objective: bool = False,
    target: bool = False,
    guard: bool = False,
) -> BehaviorContract:
    objectives: list[dict[str, object]] = []
    targets: list[dict[str, object]] = []
    guards: list[dict[str, object]] = []
    if objective:
        objectives.append(
            {
                "id": f"{name}-objective",
                "source": "objective.jsonl",
                "type": "teacher_cross_entropy",
                "weight": 1.0,
            }
        )
    if target:
        targets.append(
            {
                "id": f"{name}-target",
                "minimum": -10.0,
                "source": "targets.jsonl",
                "type": "token_log_probability",
            }
        )
    if guard:
        guards.append(
            {
                "id": f"{name}-guard",
                "maximum_mean": 1.0,
                "source": "guards.jsonl",
                "type": "base_kl",
            }
        )
    return parse_contract(
        {
            "compile": {"objectives": objectives},
            "contract_version": 1,
            "generation": {"max_new_tokens": 1, "mode": "greedy", "seeds": [0]},
            "holdout": {"sealed": True, "unseal_policy": "final_candidate_only"},
            "id": name,
            "model_requirements": {
                "output_semantics": "causal_lm",
                "tokenizer_hash": _digest(24),
            },
            "schema_version": 1,
            "statistics": {
                "bootstrap_samples": 2,
                "bootstrap_seed": 1,
                "confidence_level": 0.95,
            },
            "verify": {"guards": guards, "targets": targets},
        }
    )


def _locked_target_contract() -> BehaviorContract:
    return _behavior_contract("locked-target", target=True)


def _rebase_evidence(
    manifest_value: dict[str, object],
    *,
    contract_id: str,
    source_patch_id: str,
    source_base_hash: str = _digest(81),
) -> RebaseEvidence:
    signature_value = manifest_value["base_signature"]
    assert isinstance(signature_value, dict)
    target_signature = ModelSignature.from_dict(signature_value).signature_hash
    return RebaseEvidence(
        source_patch_id=source_patch_id,
        source_base_hash=source_base_hash,
        target_base_hash=target_signature,
        claim=RebaseClaim.SEMANTIC_REBASE_VERIFIED,
        compatibility=RebaseCompatibility.DIRECT_PHYSICAL_TRANSFER.value,
        direct_attempted=True,
        direct_outcome=VerificationOutcome.FAIL.value,
        recompile_attempted=True,
        recompile_steps=17,
        recompile_restarts=1,
        budget_exhausted=False,
        old_patched_behavior={contract_id: 0.75},
        new_patched_behavior={contract_id: 0.5},
        new_base_preservation={f"{contract_id}:guards": 0.25},
        patch_complexity_before={"parameters": 64},
        patch_complexity_after={"active_modules": 1, "parameters": 32},
        warnings=(),
    )


def _historical_source_manifest(contract: BehaviorContract) -> dict[str, object]:
    return _manifest_value(
        name="historical-source-patch",
        base_hash=_digest(84),
        contract_hashes=(contract.contract_id,),
        preserves=(contract.contract_id,),
    )


def _attach_contracts(
    root: Path,
    manifest_value: dict[str, object],
    contracts: tuple[BehaviorContract, ...],
) -> None:
    artifact_hashes = manifest_value["artifact_hashes"]
    assert isinstance(artifact_hashes, dict)
    for relative in tuple(artifact_hashes):
        if relative.startswith("contracts/"):
            del artifact_hashes[relative]
    contracts_root = root / "contracts"
    contracts_root.mkdir(exist_ok=True)
    for index, contract in enumerate(sorted(contracts, key=lambda item: item.contract_id)):
        relative = f"contracts/contract-{index:04d}.json"
        contract_path = root / relative
        contract_path.write_bytes((canonical_contract_json(contract) + "\n").encode("utf-8"))
        artifact_hashes[relative] = sha256_file(contract_path)


def _write_patch_manifest(
    root: Path,
    manifest_value: dict[str, object],
    *,
    contracts: tuple[BehaviorContract, ...],
) -> str:
    _attach_contracts(root, manifest_value, contracts)
    patch_id = _rehash_manifest(manifest_value)
    _write_canonical(root / "manifest.json", manifest_value)
    return patch_id


def _rehash_manifest(value: dict[str, object]) -> str:
    payload = dict(value)
    payload.pop("patch_id", None)
    artifact_hashes = payload.get("artifact_hashes")
    assert isinstance(artifact_hashes, dict)
    payload["artifact_hashes"] = {
        path: digest
        for path, digest in artifact_hashes.items()
        if path in {"delta-program.json", "tensors.safetensors"} or path.startswith("contracts/")
    }
    patch_id = hash_canonical(payload)
    value["patch_id"] = patch_id
    return patch_id


def _manifest_value(
    *,
    name: str,
    base_hash: str,
    contract_hashes: tuple[str, ...],
    preserves: tuple[str, ...] = (),
    requires: tuple[str, ...] = (),
    parent_patches: tuple[str, ...] = (),
) -> dict[str, object]:
    state_schema_hash = _digest(20)
    payload: dict[str, object] = {
        "artifact_hashes": {},
        "base_signature": {
            "adapter_id": "modelpact.test.v1",
            "architecture_hash": _digest(21),
            "chat_template_hash": _digest(22),
            "checkpoint_hash": base_hash,
            "generation_config_hash": _digest(23),
            "schema_version": 1,
            "state_schema_hash": state_schema_hash,
            "tokenizer_hash": _digest(24),
        },
        "compiler_configuration": {},
        "delta_representation": "additive_low_rank_sparse_v1",
        "merged_from": [],
        "name": name,
        "parent_patches": list(parent_patches),
        "preserves": list(preserves),
        "provides": list(contract_hashes),
        "rebased_from": None,
        "requires": list(requires),
        "schema_version": 1,
        "source_diff_bundle": None,
        "target_module_schema_hash": state_schema_hash,
        "tool_version": "0.1.0-test",
        "verification_policy_hash": _STACK_VERIFICATION_POLICY_HASH,
    }
    result = dict(payload)
    _rehash_manifest(result)
    return result


def _stack_certificate(
    manifest_value: dict[str, object],
    *,
    contract_hashes: tuple[str, ...],
    rebase_evidence: RebaseEvidence | None = None,
    failing: bool = False,
) -> VerificationCertificate:
    signature_value = manifest_value["base_signature"]
    patch_id = manifest_value["patch_id"]
    assert isinstance(signature_value, dict)
    assert isinstance(patch_id, str)
    signature = ModelSignature.from_dict(signature_value)
    contract = parse_contract(
        {
            "compile": {"objectives": []},
            "contract_version": 1,
            "generation": {"max_new_tokens": 1, "mode": "greedy", "seeds": [0]},
            "holdout": {"sealed": True, "unseal_policy": "final_candidate_only"},
            "id": "stack-lock-certificate-security",
            "model_requirements": {
                "output_semantics": "causal_lm",
                "tokenizer_hash": signature.tokenizer_hash,
            },
            "schema_version": 1,
            "statistics": {
                "bootstrap_samples": 2,
                "bootstrap_seed": 1,
                "confidence_level": 0.95,
            },
            "verify": {
                "guards": [
                    {
                        "id": "guard",
                        "maximum_mean": 1.0,
                        "source": "guards.jsonl",
                        "type": "base_kl",
                    }
                ],
                "targets": [
                    {
                        "id": "score",
                        "minimum": -10.0,
                        "source": "probes.jsonl",
                        "type": "token_log_probability",
                    }
                ],
            },
        }
    )
    report = verify_contract(
        contract,
        identity=ExecutionIdentity(
            signature.adapter_id,
            signature.signature_hash,
            signature.tokenizer_hash,
        ),
        provider=MappingRecordProvider(
            {
                "guards.jsonl": (
                    EvaluationRecord(
                        "guard",
                        "guard",
                        values={"base_kl": 99.0 if failing else 0.0},
                    ),
                ),
                "probes.jsonl": (
                    EvaluationRecord(
                        "probe",
                        "probe",
                        values={"token_log_probability": -999.0 if failing else -1.0},
                    ),
                ),
            }
        ),
    )
    artifact_hashes = manifest_value["artifact_hashes"]
    assert isinstance(artifact_hashes, dict)
    rebase_result = (
        None
        if rebase_evidence is None
        else {
            "claim": rebase_evidence.claim.value,
            "evidence": rebase_evidence.to_dict(),
            "new_base_guard_ids": sorted(rebase_evidence.new_base_preservation),
            "source_base_hash": rebase_evidence.source_base_hash,
            "source_patch_id": rebase_evidence.source_patch_id,
            "target_base_hash": rebase_evidence.target_base_hash,
        }
    )
    return build_certificate(
        report,
        contract,
        patch_id=patch_id,
        checkpoint_hashes={"base": signature.checkpoint_hash},
        artifact_hashes={
            path: digest for path, digest in artifact_hashes.items() if path != "certificate.json"
        },
        verification_policy=_STACK_VERIFICATION_POLICY,
        contract_hashes={
            f"contract-{index}": contract_hash
            for index, contract_hash in enumerate(contract_hashes)
        },
        rebase_result=rebase_result,
    )


def _write_resolved_bundle(
    resolved: Path,
    manifest_value: dict[str, object],
    *,
    contracts: tuple[BehaviorContract, ...],
) -> VerificationCertificate:
    artifact_hashes = manifest_value["artifact_hashes"]
    assert isinstance(artifact_hashes, dict)
    artifact_hashes.pop("certificate.json", None)
    _attach_contracts(resolved, manifest_value, contracts)
    _rehash_manifest(manifest_value)
    certificate = _stack_certificate(
        manifest_value,
        contract_hashes=tuple(sorted(contract.contract_id for contract in contracts)),
    )
    certificate_path = resolved / "certificate.json"
    certificate_path.write_bytes((certificate.canonical_json() + "\n").encode("utf-8"))
    artifact_hashes["certificate.json"] = sha256_file(certificate_path)
    _write_canonical(resolved / "manifest.json", manifest_value)
    return certificate


def _rewrite_resolved_certificate(
    resolved: Path,
    manifest_value: dict[str, object],
    *,
    contract_hashes: tuple[str, ...],
    rebase_evidence: RebaseEvidence | None = None,
    failing: bool = False,
) -> VerificationCertificate:
    artifact_hashes = manifest_value["artifact_hashes"]
    assert isinstance(artifact_hashes, dict)
    artifact_hashes.pop("certificate.json", None)
    _rehash_manifest(manifest_value)
    certificate = _stack_certificate(
        manifest_value,
        contract_hashes=contract_hashes,
        rebase_evidence=rebase_evidence,
        failing=failing,
    )
    certificate_path = resolved / "certificate.json"
    certificate_path.write_bytes((certificate.canonical_json() + "\n").encode("utf-8"))
    artifact_hashes["certificate.json"] = sha256_file(certificate_path)
    _write_canonical(resolved / "manifest.json", manifest_value)
    return certificate


def _lock_fixture(tmp_path: Path) -> tuple[Path, Path, str, dict[str, object]]:
    base_hash = _digest(2)
    contract = _locked_target_contract()
    contract_hash = contract.contract_id
    patch = tmp_path / "patch"
    patch.mkdir()
    manifest = patch / "manifest.json"
    manifest_value = _manifest_value(
        name="locked-patch",
        base_hash=base_hash,
        contract_hashes=(contract_hash,),
    )
    patch_id = _write_patch_manifest(patch, manifest_value, contracts=(contract,))
    base = tmp_path / "base"
    base.mkdir()
    resolved = tmp_path / "resolved"
    resolved.mkdir()
    resolved_manifest = resolved / "manifest.json"
    resolved_value = _manifest_value(
        name="resolved-patch",
        base_hash=base_hash,
        contract_hashes=(contract_hash,),
        parent_patches=(patch_id,),
    )
    certificate = _write_resolved_bundle(
        resolved,
        resolved_value,
        contracts=(contract,),
    )
    value: dict[str, object] = {
        "audit_hash": None,
        "base_hash": base_hash,
        "certificate_hash": certificate.certificate_hash,
        "contract_hashes": [contract_hash],
        "extensions": {
            "modelpact_cli": {
                "base_manifest_hash": _digest(4),
                "base_path": base.resolve().as_posix(),
                "dependency_order": [patch_id],
                "patch_paths": {patch_id: patch.resolve().as_posix()},
                "resolved_patch_path": resolved.resolve().as_posix(),
            }
        },
        "patch_hashes": {patch_id: sha256_file(manifest)},
        "resolution": "NAIVE_ADDITIVE_STACK",
        "resolved_artifact_hash": sha256_file(resolved_manifest),
        "schema_version": 1,
        "verification_policy_hash": _STACK_VERIFICATION_POLICY_HASH,
    }
    lock = tmp_path / "stack.lock.json"
    _write_canonical(lock, value)
    return lock, manifest, patch_id, value


def _write_variant(tmp_path: Path, index: int, value: dict[str, object]) -> Path:
    path = tmp_path / f"invalid-{index}.lock.json"
    _write_canonical(path, value)
    return path


def _add_dependent_patch(
    tmp_path: Path,
    value: dict[str, object],
    *,
    provider_id: str,
) -> tuple[str, Path]:
    base_hash = value["base_hash"]
    contract_hashes = value["contract_hashes"]
    assert isinstance(base_hash, str)
    assert isinstance(contract_hashes, list)
    assert len(contract_hashes) == 1
    provided = contract_hashes[0]
    assert isinstance(provided, str)

    provider_contract = _locked_target_contract()
    assert provider_contract.contract_id == provided
    dependent_contract = _behavior_contract("dependent-target", target=True)
    dependent_contract_id = dependent_contract.contract_id
    dependent_value = _manifest_value(
        name="dependent-patch",
        base_hash=base_hash,
        contract_hashes=(dependent_contract_id,),
        requires=(provided,),
    )
    dependent = tmp_path / "dependent-patch"
    dependent.mkdir()
    dependent_manifest = dependent / "manifest.json"
    dependent_id = _write_patch_manifest(
        dependent,
        dependent_value,
        contracts=(dependent_contract,),
    )

    patch_hashes = value["patch_hashes"]
    extensions = value["extensions"]
    assert isinstance(patch_hashes, dict)
    assert isinstance(extensions, dict)
    cli_extension = extensions["modelpact_cli"]
    assert isinstance(cli_extension, dict)
    patch_paths = cli_extension["patch_paths"]
    assert isinstance(patch_paths, dict)
    patch_hashes[dependent_id] = sha256_file(dependent_manifest)
    patch_paths[dependent_id] = dependent.resolve().as_posix()
    cli_extension["dependency_order"] = [provider_id, dependent_id]
    all_contract_hashes = sorted((provided, dependent_contract_id))
    value["contract_hashes"] = all_contract_hashes

    resolved_path = cli_extension["resolved_patch_path"]
    assert isinstance(resolved_path, str)
    resolved_manifest = Path(resolved_path) / "manifest.json"
    resolved_value = _manifest_value(
        name="resolved-patch",
        base_hash=base_hash,
        contract_hashes=tuple(all_contract_hashes),
        parent_patches=tuple(sorted((provider_id, dependent_id))),
    )
    certificate = _write_resolved_bundle(
        resolved_manifest.parent,
        resolved_value,
        contracts=(provider_contract, dependent_contract),
    )
    value["resolved_artifact_hash"] = sha256_file(resolved_manifest)
    value["certificate_hash"] = certificate.certificate_hash
    return dependent_id, dependent_manifest


def _replace_locked_patch_identity(
    value: dict[str, object],
    *,
    old_patch_id: str,
    new_patch_id: str,
    manifest_path: Path,
) -> None:
    patch_hashes = value["patch_hashes"]
    extensions = value["extensions"]
    assert isinstance(patch_hashes, dict)
    assert isinstance(extensions, dict)
    cli_extension = extensions["modelpact_cli"]
    assert isinstance(cli_extension, dict)
    patch_paths = cli_extension["patch_paths"]
    dependency_order = cli_extension["dependency_order"]
    assert isinstance(patch_paths, dict)
    assert isinstance(dependency_order, list)
    bundle_path = patch_paths.pop(old_patch_id)
    patch_hashes.pop(old_patch_id)
    patch_hashes[new_patch_id] = sha256_file(manifest_path)
    patch_paths[new_patch_id] = bundle_path
    cli_extension["dependency_order"] = [
        new_patch_id if patch_id == old_patch_id else patch_id for patch_id in dependency_order
    ]


def _rewrite_resolved_bundle(
    value: dict[str, object],
    manifest_value: dict[str, object],
    *,
    contracts: tuple[BehaviorContract, ...],
) -> None:
    extensions = value["extensions"]
    assert isinstance(extensions, dict)
    cli_extension = extensions["modelpact_cli"]
    assert isinstance(cli_extension, dict)
    resolved_path_value = cli_extension["resolved_patch_path"]
    assert isinstance(resolved_path_value, str)
    resolved_path = Path(resolved_path_value)
    certificate = _write_resolved_bundle(
        resolved_path,
        manifest_value,
        contracts=contracts,
    )
    value["resolved_artifact_hash"] = sha256_file(resolved_path / "manifest.json")
    value["certificate_hash"] = certificate.certificate_hash


def test_stack_lock_parser_rejects_unknown_malformed_and_inconsistent_data(
    tmp_path: Path,
) -> None:
    _lock, _manifest, patch_id, original = _lock_fixture(tmp_path)
    variants: list[dict[str, object]] = []

    unknown = copy.deepcopy(original)
    unknown["unexpected"] = True
    variants.append(unknown)

    bad_hash = copy.deepcopy(original)
    bad_hash["base_hash"] = "not-a-digest"
    variants.append(bad_hash)

    bad_resolution = copy.deepcopy(original)
    bad_resolution["resolution"] = "CLOSED"
    variants.append(bad_resolution)

    noninteger_schema = copy.deepcopy(original)
    noninteger_schema["schema_version"] = 1.0
    variants.append(noninteger_schema)

    missing_success_artifact = copy.deepcopy(original)
    missing_success_artifact["resolved_artifact_hash"] = None
    variants.append(missing_success_artifact)

    missing_success_certificate = copy.deepcopy(original)
    missing_success_certificate["resolution"] = "VERIFIED_COMPOSITE_PATCH"
    missing_success_certificate["certificate_hash"] = None
    variants.append(missing_success_certificate)

    failed_with_success_artifacts = copy.deepcopy(original)
    failed_with_success_artifacts["resolution"] = "EMPIRICAL_FAILURE"
    variants.append(failed_with_success_artifacts)

    duplicate_contract = copy.deepcopy(original)
    duplicate_contract["contract_hashes"] = [_digest(3), _digest(3)]
    variants.append(duplicate_contract)

    contracts_without_patches = copy.deepcopy(original)
    contracts_without_patches["patch_hashes"] = {}
    extensions = contracts_without_patches["extensions"]
    assert isinstance(extensions, dict)
    cli_extension = extensions["modelpact_cli"]
    assert isinstance(cli_extension, dict)
    cli_extension["dependency_order"] = []
    cli_extension["patch_paths"] = {}
    cli_extension["resolved_patch_path"] = None
    contracts_without_patches["certificate_hash"] = None
    contracts_without_patches["resolved_artifact_hash"] = original["base_hash"]
    variants.append(contracts_without_patches)

    unknown_extension = copy.deepcopy(original)
    extensions = unknown_extension["extensions"]
    assert isinstance(extensions, dict)
    cli_extension = extensions["modelpact_cli"]
    assert isinstance(cli_extension, dict)
    cli_extension["unexpected"] = True
    variants.append(unknown_extension)

    relative_path = copy.deepcopy(original)
    extensions = relative_path["extensions"]
    assert isinstance(extensions, dict)
    cli_extension = extensions["modelpact_cli"]
    assert isinstance(cli_extension, dict)
    cli_extension["base_path"] = "relative/base"
    variants.append(relative_path)

    network_path = copy.deepcopy(original)
    extensions = network_path["extensions"]
    assert isinstance(extensions, dict)
    cli_extension = extensions["modelpact_cli"]
    assert isinstance(cli_extension, dict)
    cli_extension["base_path"] = "//server/share/model"
    variants.append(network_path)

    traversal_path = copy.deepcopy(original)
    extensions = traversal_path["extensions"]
    assert isinstance(extensions, dict)
    cli_extension = extensions["modelpact_cli"]
    assert isinstance(cli_extension, dict)
    cli_extension["base_path"] = f"{cli_extension['base_path']}/../outside"
    variants.append(traversal_path)

    backslash_path = copy.deepcopy(original)
    extensions = backslash_path["extensions"]
    assert isinstance(extensions, dict)
    cli_extension = extensions["modelpact_cli"]
    assert isinstance(cli_extension, dict)
    base_path = cli_extension["base_path"]
    assert isinstance(base_path, str)
    cli_extension["base_path"] = base_path.replace("/", "\\")
    variants.append(backslash_path)

    redundant_separator_path = copy.deepcopy(original)
    extensions = redundant_separator_path["extensions"]
    assert isinstance(extensions, dict)
    cli_extension = extensions["modelpact_cli"]
    assert isinstance(cli_extension, dict)
    base_path = cli_extension["base_path"]
    assert isinstance(base_path, str)
    cli_extension["base_path"] = base_path.replace("/", "//", 1)
    variants.append(redundant_separator_path)

    null_extension = copy.deepcopy(original)
    null_extension["extensions"] = {"modelpact_cli": None}
    variants.append(null_extension)

    missing_patch_path = copy.deepcopy(original)
    extensions = missing_patch_path["extensions"]
    assert isinstance(extensions, dict)
    cli_extension = extensions["modelpact_cli"]
    assert isinstance(cli_extension, dict)
    cli_extension["patch_paths"] = {}
    variants.append(missing_patch_path)

    missing_dependency = copy.deepcopy(original)
    extensions = missing_dependency["extensions"]
    assert isinstance(extensions, dict)
    cli_extension = extensions["modelpact_cli"]
    assert isinstance(cli_extension, dict)
    cli_extension["dependency_order"] = []
    variants.append(missing_dependency)

    too_many_patches = copy.deepcopy(original)
    too_many_patches["patch_hashes"] = {
        _digest(index + 100): _digest(10) for index in range(MAX_STACK_LOCK_PATCHES + 1)
    }
    variants.append(too_many_patches)

    too_many_contracts = copy.deepcopy(original)
    too_many_contracts["contract_hashes"] = [
        _digest(3) for _ in range(MAX_STACK_LOCK_CONTRACTS + 1)
    ]
    variants.append(too_many_contracts)

    for index, value in enumerate(variants):
        with pytest.raises(ValueError):
            _read_lock(_write_variant(tmp_path, index, value))

    parsed = _read_lock(_lock)
    assert isinstance(parsed.lock, StackLock)
    assert set(parsed.lock.patch_hashes) == {patch_id}


@pytest.mark.parametrize("unsafe_component", ["trailing.", "trailing ", "artifact:stream"])
def test_stack_lock_parser_rejects_windows_ambiguous_absolute_path_components(
    tmp_path: Path,
    unsafe_component: str,
) -> None:
    _lock, _manifest, _patch_id, value = _lock_fixture(tmp_path)
    extensions = value["extensions"]
    assert isinstance(extensions, dict)
    cli_extension = extensions["modelpact_cli"]
    assert isinstance(cli_extension, dict)
    base_path = cli_extension["base_path"]
    assert isinstance(base_path, str)
    cli_extension["base_path"] = f"{base_path}/{unsafe_component}"

    with pytest.raises(ValueError, match="unsafe|canonical|Windows|path"):
        _read_lock(_write_variant(tmp_path, 90, value))


def test_stack_lock_parser_rejects_duplicate_truncated_and_trailing_json(
    tmp_path: Path,
) -> None:
    _lock, _manifest, _patch_id, value = _lock_fixture(tmp_path)
    encoded = canonical_dumps(value).encode("utf-8")
    variants = {
        "duplicate": encoded[:-1] + b',"schema_version":1}\n',
        "nonfinite": encoded.replace(b'"schema_version":1', b'"schema_version":NaN'),
        "truncated": encoded[:-8],
        "trailing": encoded + b"\n{}\n",
    }

    for name, content in variants.items():
        path = tmp_path / f"{name}.lock.json"
        path.write_bytes(content)
        with pytest.raises(ValueError):
            _read_lock(path)


def test_stack_lock_parser_rejects_oversized_file_before_parsing(tmp_path: Path) -> None:
    path = tmp_path / "oversized.lock.json"
    with path.open("wb") as stream:
        stream.truncate(MAX_STACK_LOCK_BYTES + 1)

    with pytest.raises(ValueError, match="size limit"):
        _read_lock(path)


def test_stack_lock_parser_accepts_canonical_json_without_a_final_lf(tmp_path: Path) -> None:
    _lock, _manifest, patch_id, value = _lock_fixture(tmp_path)
    path = tmp_path / "canonical-without-lf.lock.json"
    path.write_bytes(canonical_dumps(value).encode("utf-8"))

    assert set(_read_lock(path).lock.patch_hashes) == {patch_id}


def test_stack_lock_parser_rejects_noncanonical_bytes(tmp_path: Path) -> None:
    _lock, _manifest, _patch_id, value = _lock_fixture(tmp_path)
    canonical = canonical_dumps(value).encode("utf-8")
    variants = {
        "bom": b"\xef\xbb\xbf" + canonical + b"\n",
        "crlf": canonical + b"\r\n",
        "pretty": (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    }

    for name, content in variants.items():
        path = tmp_path / f"noncanonical-{name}.lock.json"
        path.write_bytes(content)
        with pytest.raises(ValueError, match="canonical"):
            _read_lock(path)


def test_stack_lock_parser_enforces_depth_node_object_and_string_limits(
    tmp_path: Path,
) -> None:
    _lock, _manifest, _patch_id, original = _lock_fixture(tmp_path)

    nested: object = "leaf"
    for _ in range(18):
        nested = {"nested": nested}
    excessive_depth = copy.deepcopy(original)
    excessive_depth["unexpected"] = nested

    excessive_nodes = copy.deepcopy(original)
    excessive_nodes["unexpected"] = [None] * (MAX_STACK_LOCK_NODES + 1)

    excessive_object = copy.deepcopy(original)
    excessive_object["unexpected"] = {
        f"key-{index}": None for index in range(MAX_STACK_LOCK_OBJECT_KEYS + 1)
    }

    excessive_string = copy.deepcopy(original)
    excessive_string["unexpected"] = "x" * (MAX_STACK_LOCK_STRING_LENGTH + 1)

    variants = (
        ("depth", excessive_depth, "depth|nesting"),
        ("nodes", excessive_nodes, "node"),
        ("object", excessive_object, "object|key"),
        ("string", excessive_string, "string"),
    )
    for name, value, message in variants:
        path = tmp_path / f"excessive-{name}.lock.json"
        _write_canonical(path, value)
        with pytest.raises(ValueError, match=message):
            _read_lock(path)


@pytest.mark.parametrize(
    ("name", "key", "message"),
    [
        ("nul", "attacker\x00key", "NUL|object key"),
        (
            "oversized",
            "k" * (MAX_STACK_LOCK_STRING_LENGTH + 1),
            "object key|string.*limit",
        ),
    ],
)
def test_stack_lock_parser_rejects_nul_and_oversized_object_keys(
    tmp_path: Path,
    name: str,
    key: str,
    message: str,
) -> None:
    _lock, _manifest, _patch_id, value = _lock_fixture(tmp_path)
    value[key] = None
    path = tmp_path / f"invalid-{name}-object-key.lock.json"
    _write_canonical(path, value)

    with pytest.raises(ValueError, match=message):
        _read_lock(path)


def test_aggregate_executable_contract_reference_limit_precedes_artifact_access(
    tmp_path: Path,
) -> None:
    lock, _manifest, _patch_id, value = _lock_fixture(tmp_path)
    base_hash = value["base_hash"]
    extensions = value["extensions"]
    assert isinstance(base_hash, str)
    assert isinstance(extensions, dict)
    cli_extension = extensions["modelpact_cli"]
    assert isinstance(cli_extension, dict)

    references_per_manifest = 9_091
    manifest_count = 11
    assert references_per_manifest * manifest_count > (
        cli_module.MAX_STACK_LOCK_ARTIFACT_REFERENCES
    )
    patch_hashes: dict[str, str] = {}
    patch_paths: dict[str, str] = {}
    for manifest_index in range(manifest_count):
        bundle = tmp_path / f"many-contract-references-{manifest_index}"
        bundle.mkdir()
        manifest_value = _manifest_value(
            name=f"many-contract-references-{manifest_index}",
            base_hash=base_hash,
            contract_hashes=(),
        )
        artifact_hashes = manifest_value["artifact_hashes"]
        assert isinstance(artifact_hashes, dict)
        artifact_hashes.update(
            {
                f"contracts/contract-{reference_index:05d}.json": _digest(90)
                for reference_index in range(references_per_manifest)
            }
        )
        patch_id = _rehash_manifest(manifest_value)
        manifest_path = bundle / "manifest.json"
        _write_canonical(manifest_path, manifest_value)
        patch_hashes[patch_id] = sha256_file(manifest_path)
        patch_paths[patch_id] = bundle.resolve().as_posix()

    value["patch_hashes"] = dict(sorted(patch_hashes.items()))
    value["contract_hashes"] = []
    cli_extension["patch_paths"] = dict(sorted(patch_paths.items()))
    cli_extension["dependency_order"] = sorted(patch_hashes)
    _write_canonical(lock, value)

    with pytest.raises(ValueError, match="aggregate.*executable|reference.*limit"):
        _verify_locked_patch_manifests(_read_lock(lock))


def test_locked_manifest_size_is_rejected_before_hashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock, manifest, _patch_id, value = _lock_fixture(tmp_path)
    with manifest.open("wb") as stream:
        stream.truncate(MAX_MANIFEST_BYTES + 1)
    _write_canonical(lock, value)
    parsed = _read_lock(lock)

    def forbidden_hash(_path: str | Path, *, max_bytes: int | None = None) -> str:
        del max_bytes
        raise AssertionError("oversized manifest must be rejected before hashing")

    monkeypatch.setattr(cli_module, "sha256_file", forbidden_hash)
    with pytest.raises(ValueError, match="exceeds the size limit"):
        _verify_locked_patch_manifests(parsed)


def test_locked_manifest_hash_is_explicitly_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock, _manifest, _patch_id, _value = _lock_fixture(tmp_path)
    parsed = _read_lock(lock)
    calls: list[int | None] = []

    def bounded_hash(path: str | Path, *, max_bytes: int | None = None) -> str:
        calls.append(max_bytes)
        return sha256_file(path, max_bytes=max_bytes)

    monkeypatch.setattr(cli_module, "sha256_file", bounded_hash)
    _verify_locked_patch_manifests(parsed)
    assert calls
    assert set(calls) == {MAX_MANIFEST_BYTES, DEFAULT_LIMITS.max_bytes}


def test_locked_manifest_hash_mutation_is_rejected(tmp_path: Path) -> None:
    lock, _manifest, patch_id, value = _lock_fixture(tmp_path)
    patch_hashes = value["patch_hashes"]
    assert isinstance(patch_hashes, dict)
    patch_hashes[patch_id] = _digest(99)
    _write_canonical(lock, value)

    with pytest.raises(ValueError, match="manifest.*changed|hash"):
        _verify_locked_patch_manifests(_read_lock(lock))


def test_rehashed_manifest_cannot_be_substituted_for_locked_patch_identity(
    tmp_path: Path,
) -> None:
    lock, manifest, patch_id, value = _lock_fixture(tmp_path)
    base_hash = value["base_hash"]
    contract_hashes = value["contract_hashes"]
    assert isinstance(base_hash, str)
    assert isinstance(contract_hashes, list)
    substituted = _manifest_value(
        name="attacker-rehashed-substitute",
        base_hash=base_hash,
        contract_hashes=tuple(contract_hashes),
    )
    assert substituted["patch_id"] != patch_id
    _write_canonical(manifest, substituted)
    patch_hashes = value["patch_hashes"]
    assert isinstance(patch_hashes, dict)
    patch_hashes[patch_id] = sha256_file(manifest)
    _write_canonical(lock, value)

    with pytest.raises(ValueError, match="identity|patch_id"):
        _verify_locked_patch_manifests(_read_lock(lock))


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("base_hash", _digest(40), "base"),
        ("contract_hashes", [_digest(41)], "contract"),
    ],
)
def test_locked_manifest_metadata_must_match_stack_identity(
    tmp_path: Path,
    field: str,
    replacement: object,
    message: str,
) -> None:
    lock, _manifest, _patch_id, value = _lock_fixture(tmp_path)
    value[field] = replacement
    _write_canonical(lock, value)

    with pytest.raises(ValueError, match=message):
        _verify_locked_patch_manifests(_read_lock(lock))


def test_locked_dependency_order_must_match_manifest_requirements(tmp_path: Path) -> None:
    lock, _manifest, provider_id, value = _lock_fixture(tmp_path)
    dependent_id, _dependent_manifest = _add_dependent_patch(
        tmp_path,
        value,
        provider_id=provider_id,
    )
    extensions = value["extensions"]
    assert isinstance(extensions, dict)
    cli_extension = extensions["modelpact_cli"]
    assert isinstance(cli_extension, dict)
    cli_extension["dependency_order"] = [dependent_id, provider_id]
    _write_canonical(lock, value)

    with pytest.raises(ValueError, match="dependency.*order"):
        _verify_locked_patch_manifests(_read_lock(lock))


def test_preservation_only_contract_cannot_satisfy_a_patch_requirement(
    tmp_path: Path,
) -> None:
    lock, provider_manifest, old_provider_id, value = _lock_fixture(tmp_path)
    base_hash = value["base_hash"]
    assert isinstance(base_hash, str)
    preservation_contract = _behavior_contract("preservation-only", guard=True)
    preserved_contract = preservation_contract.contract_id

    provider_value = _manifest_value(
        name="preservation-only-patch",
        base_hash=base_hash,
        contract_hashes=(),
        preserves=(preserved_contract,),
    )
    provider_id = _write_patch_manifest(
        provider_manifest.parent,
        provider_value,
        contracts=(preservation_contract,),
    )

    dependent_contract = _behavior_contract("requires-preserved-target", target=True)
    dependent_contract_id = dependent_contract.contract_id
    dependent_value = _manifest_value(
        name="requires-preserved-contract",
        base_hash=base_hash,
        contract_hashes=(dependent_contract_id,),
        requires=(preserved_contract,),
    )
    dependent_path = tmp_path / "requires-preserved-contract"
    dependent_path.mkdir()
    dependent_manifest = dependent_path / "manifest.json"
    dependent_id = _write_patch_manifest(
        dependent_path,
        dependent_value,
        contracts=(dependent_contract,),
    )

    extensions = value["extensions"]
    patch_hashes = value["patch_hashes"]
    assert isinstance(extensions, dict)
    assert isinstance(patch_hashes, dict)
    cli_extension = extensions["modelpact_cli"]
    assert isinstance(cli_extension, dict)
    patch_paths = cli_extension["patch_paths"]
    assert isinstance(patch_paths, dict)
    provider_path = patch_paths.pop(old_provider_id)
    patch_hashes.clear()
    patch_hashes.update(
        {
            provider_id: sha256_file(provider_manifest),
            dependent_id: sha256_file(dependent_manifest),
        }
    )
    patch_paths.update(
        {
            provider_id: provider_path,
            dependent_id: dependent_path.resolve().as_posix(),
        }
    )
    cli_extension["dependency_order"] = [provider_id, dependent_id]
    all_contract_hashes = sorted((preserved_contract, dependent_contract_id))
    value["contract_hashes"] = all_contract_hashes

    resolved_path = cli_extension["resolved_patch_path"]
    assert isinstance(resolved_path, str)
    resolved_value = _manifest_value(
        name="resolved-preservation-only-dependency",
        base_hash=base_hash,
        contract_hashes=(dependent_contract_id,),
        preserves=(preserved_contract,),
        requires=(preserved_contract,),
        parent_patches=tuple(sorted((provider_id, dependent_id))),
    )
    certificate = _write_resolved_bundle(
        Path(resolved_path),
        resolved_value,
        contracts=(preservation_contract, dependent_contract),
    )
    value["resolved_artifact_hash"] = sha256_file(Path(resolved_path) / "manifest.json")
    value["certificate_hash"] = certificate.certificate_hash
    _write_canonical(lock, value)

    with pytest.raises(ValueError, match="unsatisfied required contract"):
        _verify_locked_patch_manifests(_read_lock(lock))


def test_objective_only_contract_is_in_the_exact_locked_contract_set(
    tmp_path: Path,
) -> None:
    lock, manifest, old_patch_id, value = _lock_fixture(tmp_path)
    base_hash = value["base_hash"]
    assert isinstance(base_hash, str)
    objective_contract = _behavior_contract("objective-only", objective=True)
    source_value = _manifest_value(
        name="objective-only-patch",
        base_hash=base_hash,
        contract_hashes=(),
    )
    patch_id = _write_patch_manifest(
        manifest.parent,
        source_value,
        contracts=(objective_contract,),
    )
    _replace_locked_patch_identity(
        value,
        old_patch_id=old_patch_id,
        new_patch_id=patch_id,
        manifest_path=manifest,
    )
    value["contract_hashes"] = [objective_contract.contract_id]
    resolved_value = _manifest_value(
        name="resolved-objective-only-patch",
        base_hash=base_hash,
        contract_hashes=(),
        parent_patches=(patch_id,),
    )
    _rewrite_resolved_bundle(
        value,
        resolved_value,
        contracts=(objective_contract,),
    )
    _write_canonical(lock, value)

    _verify_locked_patch_manifests(_read_lock(lock))

    value["contract_hashes"] = []
    _write_canonical(lock, value)
    with pytest.raises(ValueError, match="contract identities"):
        _verify_locked_patch_manifests(_read_lock(lock))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("merged_subset", "merged_from.*subset"),
        ("rebase_with_parents", "rebase lineage"),
        ("source_diff_with_parents", "source-diff lineage"),
    ],
)
def test_rehashed_input_patch_rejects_inconsistent_lineage_modes(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    lock, manifest, old_patch_id, value = _lock_fixture(tmp_path)
    source_value = json.loads(manifest.read_bytes())
    if mutation == "merged_subset":
        source_value["parent_patches"] = [_digest(70)]
        source_value["merged_from"] = [_digest(71)]
    elif mutation == "rebase_with_parents":
        source_value["parent_patches"] = [_digest(70)]
        source_value["rebased_from"] = _digest(71)
    else:
        source_value["parent_patches"] = [_digest(70)]
        source_value["source_diff_bundle"] = _digest(71)
    patch_id = _rehash_manifest(source_value)
    _write_canonical(manifest, source_value)
    _replace_locked_patch_identity(
        value,
        old_patch_id=old_patch_id,
        new_patch_id=patch_id,
        manifest_path=manifest,
    )

    extensions = value["extensions"]
    assert isinstance(extensions, dict)
    cli_extension = extensions["modelpact_cli"]
    assert isinstance(cli_extension, dict)
    resolved_path_value = cli_extension["resolved_patch_path"]
    assert isinstance(resolved_path_value, str)
    resolved_value = json.loads((Path(resolved_path_value) / "manifest.json").read_bytes())
    resolved_value["parent_patches"] = [patch_id]
    _rewrite_resolved_bundle(
        value,
        resolved_value,
        contracts=(_locked_target_contract(),),
    )
    _write_canonical(lock, value)

    with pytest.raises(ValueError, match=message):
        _verify_locked_patch_manifests(_read_lock(lock))


@pytest.mark.parametrize("mismatch_location", ["member", "resolved"])
def test_same_checkpoint_with_different_full_base_signature_is_rejected(
    tmp_path: Path,
    mismatch_location: str,
) -> None:
    lock, _manifest, provider_id, value = _lock_fixture(tmp_path)
    dependent_id, dependent_manifest = _add_dependent_patch(
        tmp_path,
        value,
        provider_id=provider_id,
    )
    extensions = value["extensions"]
    assert isinstance(extensions, dict)
    cli_extension = extensions["modelpact_cli"]
    assert isinstance(cli_extension, dict)
    resolved_path_value = cli_extension["resolved_patch_path"]
    assert isinstance(resolved_path_value, str)
    resolved_manifest = Path(resolved_path_value) / "manifest.json"
    resolved_value = json.loads(resolved_manifest.read_bytes())
    if mismatch_location == "member":
        dependent_value = json.loads(dependent_manifest.read_bytes())
        dependent_signature = dependent_value["base_signature"]
        assert isinstance(dependent_signature, dict)
        dependent_signature["generation_config_hash"] = _digest(72)
        replacement_id = _rehash_manifest(dependent_value)
        _write_canonical(dependent_manifest, dependent_value)
        _replace_locked_patch_identity(
            value,
            old_patch_id=dependent_id,
            new_patch_id=replacement_id,
            manifest_path=dependent_manifest,
        )
        resolved_value["parent_patches"] = sorted((provider_id, replacement_id))
    else:
        resolved_signature = resolved_value["base_signature"]
        assert isinstance(resolved_signature, dict)
        resolved_signature["generation_config_hash"] = _digest(72)
    _rewrite_resolved_bundle(
        value,
        resolved_value,
        contracts=(
            _locked_target_contract(),
            _behavior_contract("dependent-target", target=True),
        ),
    )
    _write_canonical(lock, value)

    with pytest.raises(ValueError, match="full base signature"):
        _verify_locked_patch_manifests(_read_lock(lock))


def test_locked_patch_paths_must_be_unique(tmp_path: Path) -> None:
    lock, _manifest, provider_id, value = _lock_fixture(tmp_path)
    dependent_id, _dependent_manifest = _add_dependent_patch(
        tmp_path,
        value,
        provider_id=provider_id,
    )
    extensions = value["extensions"]
    assert isinstance(extensions, dict)
    cli_extension = extensions["modelpact_cli"]
    assert isinstance(cli_extension, dict)
    patch_paths = cli_extension["patch_paths"]
    assert isinstance(patch_paths, dict)
    patch_paths[dependent_id] = patch_paths[provider_id]
    _write_canonical(lock, value)

    with pytest.raises(ValueError, match="path.*unique|duplicate.*path"):
        parsed = _read_lock(lock)
        _verify_locked_patch_manifests(parsed)


def test_resolved_patch_manifest_hash_substitution_is_rejected(tmp_path: Path) -> None:
    lock, _manifest, _patch_id, value = _lock_fixture(tmp_path)
    value["resolved_artifact_hash"] = _digest(50)
    _write_canonical(lock, value)

    with pytest.raises(ValueError, match="resolved.*changed|resolved.*hash"):
        _verify_locked_patch_manifests(_read_lock(lock))


def test_resolved_patch_path_substitution_is_rejected(tmp_path: Path) -> None:
    lock, _manifest, _patch_id, value = _lock_fixture(tmp_path)
    substitute = tmp_path / "substituted-resolved"
    substitute.mkdir()
    (substitute / "manifest.json").write_bytes(b'{"substituted":true}\n')
    extensions = value["extensions"]
    assert isinstance(extensions, dict)
    cli_extension = extensions["modelpact_cli"]
    assert isinstance(cli_extension, dict)
    cli_extension["resolved_patch_path"] = substitute.resolve().as_posix()
    _write_canonical(lock, value)

    with pytest.raises(ValueError, match="resolved.*changed|resolved.*hash"):
        _verify_locked_patch_manifests(_read_lock(lock))


def test_rehashed_resolved_contract_artifact_substitution_is_rejected(
    tmp_path: Path,
) -> None:
    lock, _manifest, _patch_id, value = _lock_fixture(tmp_path)
    extensions = value["extensions"]
    contract_hashes = value["contract_hashes"]
    assert isinstance(extensions, dict)
    assert isinstance(contract_hashes, list)
    cli_extension = extensions["modelpact_cli"]
    assert isinstance(cli_extension, dict)
    resolved_path_value = cli_extension["resolved_patch_path"]
    assert isinstance(resolved_path_value, str)
    resolved_path = Path(resolved_path_value)
    resolved_manifest = resolved_path / "manifest.json"
    resolved_value = json.loads(resolved_manifest.read_bytes())
    artifact_hashes = resolved_value["artifact_hashes"]
    assert isinstance(artifact_hashes, dict)
    contract_relative = next(
        relative for relative in artifact_hashes if relative.startswith("contracts/")
    )
    substituted_contract = _behavior_contract("attacker-substitute", target=True)
    contract_path = resolved_path / contract_relative
    contract_path.write_bytes(
        (canonical_contract_json(substituted_contract) + "\n").encode("utf-8")
    )
    artifact_hashes[contract_relative] = sha256_file(contract_path)
    certificate = _rewrite_resolved_certificate(
        resolved_path,
        resolved_value,
        contract_hashes=tuple(contract_hashes),
    )
    value["resolved_artifact_hash"] = sha256_file(resolved_manifest)
    value["certificate_hash"] = certificate.certificate_hash
    _write_canonical(lock, value)

    with pytest.raises(ValueError, match="resolved.*contract|contract.*resolved"):
        _verify_locked_patch_manifests(_read_lock(lock))


@pytest.mark.parametrize("evidence_kind", ["corrupt", "mismatched_lineage"])
def test_rebased_input_rejects_corrupt_or_inconsistent_evidence(
    tmp_path: Path,
    evidence_kind: str,
) -> None:
    lock, manifest, old_patch_id, value = _lock_fixture(tmp_path)
    base_hash = value["base_hash"]
    assert isinstance(base_hash, str)
    contract = _behavior_contract("rebased-input", target=True, guard=True)
    historical_source = _historical_source_manifest(contract)
    rebased_from = historical_source["patch_id"]
    historical_signature_value = historical_source["base_signature"]
    assert isinstance(rebased_from, str)
    assert isinstance(historical_signature_value, dict)
    historical_signature = ModelSignature.from_dict(historical_signature_value).signature_hash
    source_value = _manifest_value(
        name="rebased-input-patch",
        base_hash=base_hash,
        contract_hashes=(contract.contract_id,),
        preserves=(contract.contract_id,),
    )
    source_value["rebased_from"] = rebased_from
    _attach_contracts(manifest.parent, source_value, (contract,))
    evidence_path = manifest.parent / "evidence" / "rebase.json"
    evidence_path.parent.mkdir()
    source_manifest_path = manifest.parent / "evidence" / "source-manifest.json"
    _write_canonical(source_manifest_path, historical_source)
    if evidence_kind == "corrupt":
        evidence_path.write_bytes(b'{"schema_version":1,"truncated":')
    else:
        inconsistent = _rebase_evidence(
            source_value,
            contract_id=contract.contract_id,
            source_patch_id=_digest(82),
            source_base_hash=historical_signature,
        )
        _write_canonical(evidence_path, inconsistent.to_dict())
    artifact_hashes = source_value["artifact_hashes"]
    assert isinstance(artifact_hashes, dict)
    artifact_hashes["evidence/rebase.json"] = sha256_file(evidence_path)
    artifact_hashes["evidence/source-manifest.json"] = sha256_file(source_manifest_path)
    patch_id = _rehash_manifest(source_value)
    _write_canonical(manifest, source_value)
    _replace_locked_patch_identity(
        value,
        old_patch_id=old_patch_id,
        new_patch_id=patch_id,
        manifest_path=manifest,
    )
    value["contract_hashes"] = [contract.contract_id]
    resolved_value = _manifest_value(
        name="resolved-rebased-input",
        base_hash=base_hash,
        contract_hashes=(contract.contract_id,),
        preserves=(contract.contract_id,),
        parent_patches=(patch_id,),
    )
    _rewrite_resolved_bundle(value, resolved_value, contracts=(contract,))
    _write_canonical(lock, value)

    with pytest.raises(ValueError, match="Rebase Evidence|rebase evidence|rebase.*lineage"):
        _verify_locked_patch_manifests(_read_lock(lock))


@pytest.mark.parametrize("location", ["input", "resolved"])
def test_non_rebased_manifest_rejects_rebase_evidence(
    tmp_path: Path,
    location: str,
) -> None:
    lock, manifest, patch_id, value = _lock_fixture(tmp_path)
    contract = _locked_target_contract()
    contract_hashes = value["contract_hashes"]
    extensions = value["extensions"]
    assert isinstance(contract_hashes, list)
    assert isinstance(extensions, dict)
    cli_extension = extensions["modelpact_cli"]
    assert isinstance(cli_extension, dict)
    resolved_path_value = cli_extension["resolved_patch_path"]
    assert isinstance(resolved_path_value, str)
    resolved_path = Path(resolved_path_value)
    target_manifest_path = manifest if location == "input" else resolved_path / "manifest.json"
    target_value = json.loads(target_manifest_path.read_bytes())
    historical_source = _historical_source_manifest(contract)
    historical_patch_id = historical_source["patch_id"]
    historical_signature_value = historical_source["base_signature"]
    assert isinstance(historical_patch_id, str)
    assert isinstance(historical_signature_value, dict)
    historical_signature = ModelSignature.from_dict(historical_signature_value).signature_hash
    evidence = _rebase_evidence(
        target_value,
        contract_id=contract.contract_id,
        source_patch_id=historical_patch_id,
        source_base_hash=historical_signature,
    )
    evidence_path = target_manifest_path.parent / "evidence" / "rebase.json"
    evidence_path.parent.mkdir()
    _write_canonical(evidence_path, evidence.to_dict())
    artifact_hashes = target_value["artifact_hashes"]
    assert isinstance(artifact_hashes, dict)
    artifact_hashes["evidence/rebase.json"] = sha256_file(evidence_path)
    if location == "resolved":
        source_manifest_path = target_manifest_path.parent / "evidence" / "source-manifest.json"
        _write_canonical(source_manifest_path, historical_source)
        artifact_hashes["evidence/source-manifest.json"] = sha256_file(source_manifest_path)
    unchanged_patch_id = _rehash_manifest(target_value)
    if location == "input":
        assert unchanged_patch_id == patch_id
        _write_canonical(manifest, target_value)
        patch_hashes = value["patch_hashes"]
        assert isinstance(patch_hashes, dict)
        patch_hashes[patch_id] = sha256_file(manifest)
    else:
        certificate = _rewrite_resolved_certificate(
            resolved_path,
            target_value,
            contract_hashes=tuple(contract_hashes),
            rebase_evidence=evidence,
        )
        value["resolved_artifact_hash"] = sha256_file(target_manifest_path)
        value["certificate_hash"] = certificate.certificate_hash
    _write_canonical(lock, value)

    with pytest.raises(ValueError, match="non-rebased|Rebase Evidence|rebase evidence"):
        _verify_locked_patch_manifests(_read_lock(lock))


def test_locked_certificate_hash_substitution_is_rejected(tmp_path: Path) -> None:
    lock, _manifest, _patch_id, value = _lock_fixture(tmp_path)
    value["certificate_hash"] = _digest(52)
    _write_canonical(lock, value)

    with pytest.raises(ValueError, match="certificate.*hash|certificate.*identity"):
        _verify_locked_patch_manifests(_read_lock(lock))


def test_rehashed_certificate_cannot_be_substituted_for_resolved_patch_identity(
    tmp_path: Path,
) -> None:
    lock, _manifest, _patch_id, value = _lock_fixture(tmp_path)
    extensions = value["extensions"]
    assert isinstance(extensions, dict)
    cli_extension = extensions["modelpact_cli"]
    assert isinstance(cli_extension, dict)
    resolved_path_value = cli_extension["resolved_patch_path"]
    assert isinstance(resolved_path_value, str)
    resolved_path = Path(resolved_path_value)
    certificate_path = resolved_path / "certificate.json"
    certificate_value = json.loads(certificate_path.read_bytes())
    certificate_value["patch_id"] = _digest(53)
    certificate_payload = dict(certificate_value)
    certificate_payload.pop("certificate_hash")
    certificate_value["certificate_hash"] = hash_canonical(certificate_payload)
    _write_canonical(certificate_path, certificate_value)

    resolved_manifest = resolved_path / "manifest.json"
    resolved_value = json.loads(resolved_manifest.read_bytes())
    artifact_hashes = resolved_value["artifact_hashes"]
    assert isinstance(artifact_hashes, dict)
    artifact_hashes["certificate.json"] = sha256_file(certificate_path)
    _rehash_manifest(resolved_value)
    _write_canonical(resolved_manifest, resolved_value)
    value["resolved_artifact_hash"] = sha256_file(resolved_manifest)
    value["certificate_hash"] = certificate_value["certificate_hash"]
    _write_canonical(lock, value)

    with pytest.raises(ValueError, match="patch_id mismatch|certificate.*patch"):
        _verify_locked_patch_manifests(_read_lock(lock))


def test_successful_resolution_cannot_pin_a_failing_certificate(tmp_path: Path) -> None:
    """A pinned certificate is not the same as a passing one.

    Every field stays self-consistent and correctly hashed; the only change is
    that the composite failed both its target and its preservation contract.
    """

    lock, _manifest, _patch_id, value = _lock_fixture(tmp_path)
    extensions = value["extensions"]
    assert isinstance(extensions, dict)
    cli_extension = extensions["modelpact_cli"]
    assert isinstance(cli_extension, dict)
    resolved_path_value = cli_extension["resolved_patch_path"]
    assert isinstance(resolved_path_value, str)
    resolved_path = Path(resolved_path_value)
    resolved_manifest = resolved_path / "manifest.json"
    resolved_value = json.loads(resolved_manifest.read_bytes())
    contract_hashes = value["contract_hashes"]
    assert isinstance(contract_hashes, list)
    certificate = _rewrite_resolved_certificate(
        resolved_path,
        resolved_value,
        contract_hashes=tuple(contract_hashes),
        failing=True,
    )
    assert certificate.verification_outcome is VerificationOutcome.FAIL
    value["resolved_artifact_hash"] = sha256_file(resolved_manifest)
    value["certificate_hash"] = certificate.certificate_hash
    _write_canonical(lock, value)

    with pytest.raises(ValueError, match="non-passing certificate"):
        _verify_locked_patch_manifests(_read_lock(lock))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("lineage", "lineage"),
        ("requirements", "require"),
        ("contract_roles", "contract.*role"),
    ],
)
def test_rehashed_resolved_patch_cannot_change_stack_semantics(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    lock, _manifest, patch_id, value = _lock_fixture(tmp_path)
    extensions = value["extensions"]
    assert isinstance(extensions, dict)
    cli_extension = extensions["modelpact_cli"]
    assert isinstance(cli_extension, dict)
    resolved_path = cli_extension["resolved_patch_path"]
    assert isinstance(resolved_path, str)
    resolved_manifest = Path(resolved_path) / "manifest.json"
    resolved_value = json.loads(resolved_manifest.read_bytes())
    if mutation == "lineage":
        resolved_value["merged_from"] = [patch_id]
        resolved_value["rebased_from"] = _digest(60)
    elif mutation == "requirements":
        resolved_value["requires"] = [_digest(61)]
    else:
        contract_hashes = value["contract_hashes"]
        assert isinstance(contract_hashes, list)
        resolved_value["provides"] = []
        resolved_value["preserves"] = contract_hashes
    contract_hashes = value["contract_hashes"]
    assert isinstance(contract_hashes, list)
    certificate = _write_resolved_bundle(
        resolved_manifest.parent,
        resolved_value,
        contracts=(_locked_target_contract(),),
    )
    value["resolved_artifact_hash"] = sha256_file(resolved_manifest)
    value["certificate_hash"] = certificate.certificate_hash
    _write_canonical(lock, value)

    with pytest.raises(ValueError, match=message):
        _verify_locked_patch_manifests(_read_lock(lock))


def test_revert_rejects_nonregular_manifest_before_model_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock, manifest, patch_id, _value = _lock_fixture(tmp_path)
    manifest.unlink()
    manifest.mkdir()
    model_loaded = False

    def forbidden_model_load(*_args: object, **_kwargs: object) -> object:
        nonlocal model_loaded
        model_loaded = True
        raise AssertionError("the base model must not load before manifest preflight")

    monkeypatch.setattr(cli_module, "_load_model", forbidden_model_load)
    result = RUNNER.invoke(
        app,
        [
            "revert",
            str(lock),
            "--remove",
            patch_id,
            "--output",
            str(tmp_path / "output"),
            "--adapter",
            "tiny",
            "--json",
        ],
    )
    assert result.exit_code != 0
    assert model_loaded is False
    payload = json.loads(result.stdout)
    assert payload["status"] == "ERROR"
    assert "regular file" in payload["error"]


def test_locked_symlink_manifest_is_rejected(tmp_path: Path) -> None:
    lock, manifest, _patch_id, value = _lock_fixture(tmp_path)
    target = tmp_path / "outside-manifest.json"
    target.write_bytes(b"{}\n")
    manifest.unlink()
    try:
        manifest.symlink_to(target)
    except OSError as error:
        pytest.skip(f"symlink creation is unavailable: {error}")
    patch_hashes = value["patch_hashes"]
    assert isinstance(patch_hashes, dict)
    patch_hashes[next(iter(patch_hashes))] = sha256_file(target)
    _write_canonical(lock, value)

    with pytest.raises(ValueError, match="regular file"):
        _verify_locked_patch_manifests(_read_lock(lock))
