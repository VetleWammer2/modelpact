"""ModelPact command-line interface.

The CLI is intentionally a thin orchestration boundary around the typed research
engines.  Trusted adapter specifications may execute local Python; every other
input is parsed as bounded data by the subsystem that owns its schema.
"""

# Ruff B008 is intentional: Typer evaluates declarative Option/Argument metadata
# when command functions are registered.
# ruff: noqa: B008

from __future__ import annotations

import copy
import dataclasses
import json
import math
import shutil
import tempfile
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar, cast

import torch
import typer
from safetensors.torch import save_file
from torch import Tensor, nn

from modelpact import __version__
from modelpact.compose.stack import STACK_LOCK_FIELDS, StackLock
from modelpact.loading import load_trusted_adapter, parse_dtype
from modelpact.models.manifest import ModelManifest, build_model_manifest
from modelpact.models.schema import ModelStateSchema
from modelpact.patch.ast import Alias, DeltaOp, DeltaProgram, LowRankMatrixDelta, VectorDelta
from modelpact.patch.bundle import PatchBundle, load_patch_bundle, missing_bundle_artifacts
from modelpact.status import AuditClaim, VerificationOutcome
from modelpact.util.atomic import atomic_write_text
from modelpact.util.canonical_json import canonical_dumps, strict_json_loads
from modelpact.util.hashing import (
    hash_canonical,
    is_sha256_digest,
    sha256_bytes,
    sha256_file,
)

app = typer.Typer(
    name="modelpact",
    help="Compile, compose, rebase, audit, and revert learned behavior.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)
contract_app = typer.Typer(help="Validate and compare Behavior Contract v1 documents.")
emit_app = typer.Typer(help="Emit package-independent patch utilities.")
app.add_typer(contract_app, name="contract")
app.add_typer(emit_app, name="emit")


EXIT_ERROR = 1
EXIT_FAILED = 2
EXIT_INCONCLUSIVE = 3
EXIT_UNSUPPORTED = 4

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class _CommandResult:
    payload: Mapping[str, object]
    exit_code: int = 0


@dataclass(frozen=True, slots=True)
class _LoadedModel:
    adapter: Any
    model: nn.Module
    manifest: ModelManifest


def _jsonable(value: object) -> object:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Tensor):
        return {
            "dtype": str(value.dtype).removeprefix("torch."),
            "shape": list(value.shape),
        }
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_jsonable(item) for item in value]
    return value


def _emit(payload: Mapping[str, object], *, compact: bool) -> None:
    normalized = cast(dict[str, object], _jsonable(payload))
    rendered = (
        canonical_dumps(normalized)
        if compact
        else json.dumps(normalized, ensure_ascii=False, indent=2, sort_keys=True)
    )
    typer.echo(rendered)


def _invoke(operation: Callable[[], _CommandResult], *, compact: bool) -> None:
    try:
        result = operation()
    except typer.Exit:
        raise
    except Exception as error:
        _emit(
            {
                "error": str(error),
                "error_type": type(error).__name__,
                "status": "ERROR",
            },
            compact=compact,
        )
        raise typer.Exit(EXIT_ERROR) from None
    _emit(result.payload, compact=compact)
    if result.exit_code:
        raise typer.Exit(result.exit_code)


def _load_model(
    adapter_spec: str,
    checkpoint: Path,
    *,
    device: str,
    dtype: str,
) -> _LoadedModel:
    adapter = load_trusted_adapter(adapter_spec)
    model = adapter.load(str(checkpoint), device=device, dtype=parse_dtype(dtype))
    adapter.prepare(model)
    manifest = build_model_manifest(
        model,
        checkpoint=checkpoint,
        adapter_id=adapter.adapter_id,
    )
    return _LoadedModel(adapter, model, manifest)


def _write_json(path: Path, value: object, *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    atomic_write_text(path, canonical_dumps(_jsonable(value)) + "\n", overwrite=overwrite)


def _model_payload(loaded: _LoadedModel) -> dict[str, object]:
    manifest = loaded.manifest
    patchable = [
        {
            "kind": module.kind,
            "parameter_names": list(module.parameter_names),
            "path": module.path,
        }
        for module in loaded.adapter.patchable_modules(loaded.model)
    ]
    return {
        "adapter_id": loaded.adapter.adapter_id,
        "manifest": manifest.to_dict(),
        "manifest_hash": manifest.manifest_hash,
        "patchable_modules": patchable,
        "status": "PASS",
    }


@app.command("scan")
def scan_command(
    model: str = typer.Option(
        ..., "--model", help="Trusted adapter: tiny, hf, or module:attribute."
    ),
    checkpoint: Path = typer.Option(..., "--checkpoint", exists=True, readable=True),
    output: Path | None = typer.Option(None, "--output"),
    device: str = typer.Option("cpu", "--device"),
    dtype: str = typer.Option("float32", "--dtype"),
    json_output: bool = typer.Option(False, "--json", help="Emit compact canonical JSON."),
) -> None:
    """Fingerprint a trusted local model and emit Model Manifest v1."""

    def operation() -> _CommandResult:
        loaded = _load_model(model, checkpoint, device=device, dtype=dtype)
        payload = _model_payload(loaded)
        if output is not None:
            _write_json(output, loaded.manifest.to_dict())
            payload["output"] = output.as_posix()
        return _CommandResult(payload)

    _invoke(operation, compact=json_output)


def _contract_payload(contract: Any, *, include_document: bool) -> dict[str, object]:
    payload: dict[str, object] = {
        "contract_hash": contract.contract_id,
        "contract_id": contract.id,
        "contract_version": contract.contract_version,
        "guard_assertions": len(contract.guards),
        "holdout_configured": contract.holdout.configured,
        "objectives": len(contract.objectives),
        "schema_version": contract.schema_version,
        "status": "PASS",
        "target_assertions": len(contract.targets),
    }
    if include_document:
        payload["contract"] = contract.to_dict()
    return payload


@contract_app.command("validate")
def contract_validate_command(
    path: Path = typer.Argument(..., exists=True, readable=True),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Parse, normalize, and validate a contract without executing probes."""

    from modelpact.contracts.parser import load_contract

    _invoke(
        lambda: _CommandResult(_contract_payload(load_contract(path), include_document=False)),
        compact=json_output,
    )


@contract_app.command("inspect")
def contract_inspect_command(
    path: Path = typer.Argument(..., exists=True, readable=True),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Print the canonical, typed contract representation."""

    from modelpact.contracts.parser import load_contract

    _invoke(
        lambda: _CommandResult(_contract_payload(load_contract(path), include_document=True)),
        compact=json_output,
    )


@contract_app.command("hash")
def contract_hash_command(
    path: Path = typer.Argument(..., exists=True, readable=True),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Compute the stable content hash of a normalized contract."""

    from modelpact.contracts.parser import load_contract

    def operation() -> _CommandResult:
        contract = load_contract(path)
        return _CommandResult(
            {"contract_hash": contract.contract_id, "contract_id": contract.id, "status": "PASS"}
        )

    _invoke(operation, compact=json_output)


@contract_app.command("check-static")
def contract_static_command(
    paths: list[Path] = typer.Argument(..., exists=True, readable=True),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Report only contradictions justified by supported static fragments."""

    from modelpact.contracts.parser import load_contract
    from modelpact.contracts.static import ProbeRecord, check_static_contracts
    from modelpact.verify.provider import load_probe_records

    def operation() -> _CommandResult:
        contracts = [load_contract(path) for path in paths]
        records_by_contract: dict[str, dict[str, tuple[ProbeRecord, ...]]] = {}
        unavailable: list[str] = []
        for contract, path in zip(contracts, paths, strict=True):
            sources: dict[str, tuple[ProbeRecord, ...]] = {}
            for source in sorted({item.source for item in (*contract.targets, *contract.guards)}):
                candidate = path.parent / source
                if not candidate.is_file():
                    unavailable.append(f"{contract.id}:{source}")
                    continue
                sources[source] = cast(
                    tuple[ProbeRecord, ...], load_probe_records(path.parent, source)
                )
            records_by_contract[contract.id] = sources
        result = check_static_contracts(
            contracts,
            records_by_contract=records_by_contract,
        )
        return _CommandResult(
            {
                "checked_contracts": list(result.checked_contracts),
                "conclusion": result.conclusion,
                "requirements_examined": result.requirements_examined,
                "status": result.status.value,
                "unavailable_probe_sources": sorted(unavailable),
                "witnesses": _jsonable(result.witnesses),
            },
            EXIT_FAILED if result.contradictory else 0,
        )

    _invoke(operation, compact=json_output)


def _probe_prompts(path: Path, *, maximum_prompts: int) -> tuple[str, ...]:
    from modelpact.contracts.parser import load_data_file
    from modelpact.probes.dataset import load_jsonl
    from modelpact.probes.grammar import TemplateGrammar

    if maximum_prompts <= 0:
        raise ValueError("maximum_prompts must be positive")
    if path.suffix.lower() == ".jsonl":
        prompts = tuple(probe.prompt for probe in load_jsonl(path, max_probes=maximum_prompts))
    else:
        value = load_data_file(path)
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            prompts = tuple(cast(list[str], value))
        elif isinstance(value, Mapping):
            raw_prompts = value.get("prompts")
            templates = value.get("templates")
            variables = value.get("variables")
            if isinstance(raw_prompts, list) and all(isinstance(item, str) for item in raw_prompts):
                prompts = tuple(cast(list[str], raw_prompts))
            elif (
                isinstance(templates, list)
                and all(isinstance(item, str) for item in templates)
                and isinstance(variables, Mapping)
                and all(
                    isinstance(key, str)
                    and isinstance(items, list)
                    and all(isinstance(item, str) for item in items)
                    for key, items in variables.items()
                )
            ):
                grammar = TemplateGrammar(
                    tuple(cast(list[str], templates)),
                    {key: tuple(cast(list[str], items)) for key, items in variables.items()},
                    maximum_outputs=maximum_prompts,
                )
                prompts = grammar.expand()
            else:
                raise ValueError("probe space must declare prompts or templates plus variables")
        else:
            raise ValueError("probe space must be JSONL, a string list, or a grammar object")
    prompts = tuple(dict.fromkeys(prompts))
    if not prompts or len(prompts) > maximum_prompts:
        raise ValueError("probe space is empty or exceeds its configured bound")
    return prompts


@app.command("diff")
def diff_command(
    base: str = typer.Option(..., "--base", help="Trusted base adapter."),
    base_checkpoint: Path = typer.Option(..., "--base-checkpoint", exists=True),
    target: str = typer.Option(..., "--target", help="Trusted target adapter."),
    target_checkpoint: Path = typer.Option(..., "--target-checkpoint", exists=True),
    probe_space: Path = typer.Option(..., "--probe-space", exists=True, readable=True),
    output: Path = typer.Option(..., "--output"),
    search_budget: int = typer.Option(256, "--search-budget", min=1, max=1_000_000),
    divergence_threshold: float = typer.Option(0.01, "--divergence-threshold", min=0.0),
    maximum_clusters: int = typer.Option(8, "--maximum-clusters", min=1, max=10_000),
    seed: int = typer.Option(0, "--seed", min=0),
    device: str = typer.Option("cpu", "--device"),
    dtype: str = typer.Option("float32", "--dtype"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Search, minimize, fingerprint, and cluster scoped model differences."""

    from modelpact.diff.cluster import deterministic_agglomerative
    from modelpact.diff.engine import DiffConfig, find_difference_witnesses
    from modelpact.diff.report import write_difference_bundle

    def operation() -> _CommandResult:
        base_loaded = _load_model(base, base_checkpoint, device=device, dtype=dtype)
        target_loaded = _load_model(target, target_checkpoint, device=device, dtype=dtype)
        if base_loaded.adapter.adapter_id != target_loaded.adapter.adapter_id:
            return _CommandResult(
                {
                    "reason": "diff currently requires one adapter identity for both local models",
                    "status": "UNSUPPORTED",
                },
                EXIT_UNSUPPORTED,
            )
        if (
            base_loaded.manifest.signature.tokenizer_hash
            != target_loaded.manifest.signature.tokenizer_hash
        ):
            return _CommandResult(
                {"reason": "tokenizer fingerprints differ", "status": "UNSUPPORTED"},
                EXIT_UNSUPPORTED,
            )
        prompts = _probe_prompts(probe_space, maximum_prompts=search_budget)
        config = DiffConfig(
            divergence_threshold=divergence_threshold,
            search_budget=search_budget,
            seed=seed,
        )
        execution = find_difference_witnesses(
            base_loaded.adapter,
            base_loaded.model,
            target_loaded.model,
            prompts,
            config=config,
        )
        clusters = deterministic_agglomerative(
            execution.witnesses,
            maximum_clusters=maximum_clusters,
        )
        manifest = write_difference_bundle(
            output,
            execution.witnesses,
            clusters,
            configuration={
                "base_signature": base_loaded.manifest.signature.to_dict(),
                "diff": _jsonable(config),
                "probe_space_hash": sha256_file(probe_space),
                "target_signature": target_loaded.manifest.signature.to_dict(),
            },
        )
        return _CommandResult(
            {
                "clusters": len(clusters),
                "manifest_hash": hash_canonical(manifest),
                "output": output.as_posix(),
                "prompts_evaluated": execution.prompts_evaluated,
                "scope": "executed_probe_space",
                "status": "PASS",
                "tokens_processed": execution.tokens_processed,
                "wall_seconds": execution.wall_seconds,
                "witnesses": len(execution.witnesses),
            }
        )

    _invoke(operation, compact=json_output)


def _contract_paths(bundle: PatchBundle) -> tuple[Path, ...]:
    def executable(relative: str) -> bool:
        path = Path(relative)
        parts = path.parts
        if path.suffix.lower() not in {".json", ".yaml", ".yml"}:
            return False
        if len(parts) == 2 and parts[0] == "contracts":
            return path.stem in {"target", "preservation"} or path.stem.startswith("contract-")
        return (
            len(parts) == 4
            and parts[:2] == ("contracts", "parents")
            and path.name == "contract.json"
        )

    paths = [
        bundle.path / relative
        for relative in sorted(bundle.manifest.artifact_hashes)
        if executable(relative)
    ]
    if not paths:
        raise ValueError("patch bundle carries no executable contracts")
    return tuple(paths)


def _load_bundle_contracts(bundle: PatchBundle) -> dict[str, tuple[Any, Path]]:
    from modelpact.contracts.parser import load_contract

    contracts: dict[str, tuple[Any, Path]] = {}
    for path in _contract_paths(bundle):
        contract = load_contract(path)
        prior = contracts.get(contract.contract_id)
        if prior is not None and prior[0].to_dict() != contract.to_dict():
            raise ValueError("contract hash collision inside patch bundle")
        contracts[contract.contract_id] = (contract, path)
    return contracts


def _alias_map(schema: ModelStateSchema) -> dict[str, str]:
    return {
        member: group.canonical
        for group in schema.aliases
        for member in group.members
        if member != group.canonical
    }


def _bundle_dense_delta(bundle: PatchBundle) -> dict[str, Tensor]:
    return {
        target: bundle.program.materialize(target, bundle.tensors).detach().cpu()
        for target in sorted(bundle.program.targets)
    }


def _dense_program(
    deltas: Mapping[str, Tensor],
    schema: ModelStateSchema,
) -> tuple[DeltaProgram, dict[str, Tensor]]:
    """Represent executed dense deltas in the safe additive IR.

    Matrices use ``delta @ I``.  This is intentionally an in-memory resolution
    representation, not a compactness claim.
    """

    if not deltas:
        raise ValueError("a resolved patch delta cannot be empty")
    canonical = dict(deltas)
    for group in schema.aliases:
        selected = [member for member in group.members if member in canonical]
        if not selected:
            continue
        first = canonical[selected[0]]
        if any(not torch.equal(first, canonical[member]) for member in selected[1:]):
            raise ValueError(f"resolved delta disagrees across aliases: {group.members}")
        for member in selected:
            del canonical[member]
        canonical[group.canonical] = first

    operations: dict[str, DeltaOp] = {}
    tensors: dict[str, Tensor] = {}
    for index, (target, delta) in enumerate(sorted(canonical.items())):
        schema.tensor(target)
        value = delta.detach().cpu().contiguous()
        if value.ndim == 1:
            key = f"resolved.{index:08d}.vector"
            tensors[key] = value
            operations[target] = VectorDelta(key)
        elif value.ndim == 2:
            left = f"resolved.{index:08d}.left"
            right = f"resolved.{index:08d}.right"
            tensors[left] = value
            tensors[right] = torch.eye(value.shape[1], dtype=value.dtype)
            operations[target] = LowRankMatrixDelta(left, right)
        else:
            raise ValueError(f"resolved delta target has unsupported rank: {target}")
    for group in schema.aliases:
        if group.canonical not in operations:
            continue
        for member in group.members:
            if member != group.canonical:
                operations[member] = Alias(group.canonical)
    program = DeltaProgram(dict(sorted(operations.items())))
    program.validate(tensors, schema)
    return program, tensors


def _factorized_program(
    factors: Mapping[str, tuple[Tensor, Tensor]],
    schema: ModelStateSchema,
) -> tuple[DeltaProgram, dict[str, Tensor]]:
    """Serialize executed matrix factors while preserving declared aliases."""

    if not factors:
        raise ValueError("a factorized patch delta cannot be empty")
    aliases = _alias_map(schema)
    canonical: dict[str, tuple[Tensor, Tensor]] = {}
    for name, (left, right) in sorted(factors.items()):
        target = aliases.get(name, name)
        schema.tensor(target)
        value = (left.detach().cpu().contiguous(), right.detach().cpu().contiguous())
        prior = canonical.get(target)
        if prior is not None and not torch.equal(prior[0] @ prior[1], value[0] @ value[1]):
            raise ValueError(f"factorized delta disagrees across aliases: {target}")
        canonical[target] = value
    operations: dict[str, DeltaOp] = {}
    tensors: dict[str, Tensor] = {}
    for index, (target, (left, right)) in enumerate(sorted(canonical.items())):
        left_name = f"factorized.{index:08d}.left"
        right_name = f"factorized.{index:08d}.right"
        tensors[left_name] = left
        tensors[right_name] = right
        operations[target] = LowRankMatrixDelta(left_name, right_name)
    for group in schema.aliases:
        if group.canonical not in operations:
            continue
        for member in group.members:
            if member != group.canonical:
                operations[member] = Alias(group.canonical)
    program = DeltaProgram(dict(sorted(operations.items())))
    program.validate(tensors, schema)
    return program, tensors


def _copy_contract_artifacts(
    bundles: Sequence[PatchBundle],
) -> tuple[dict[str, bytes], tuple[str, ...]]:
    from modelpact.contracts.parser import load_contract

    artifacts: dict[str, bytes] = {}
    identifiers: set[str] = set()
    for bundle in bundles:
        for path in _contract_paths(bundle):
            contract = load_contract(path)
            identifiers.add(contract.contract_id)
            name = f"contracts/contract-{contract.contract_id.removeprefix('sha256:')}.json"
            artifacts[name] = (canonical_dumps(contract.to_dict()) + "\n").encode()
    return dict(sorted(artifacts.items())), tuple(sorted(identifiers))


def _schema_files(contract: Any) -> tuple[str, ...]:
    values = {
        item.option("schema_file")
        for item in (*contract.targets, *contract.guards)
        if isinstance(item.option("schema_file"), str)
    }
    return tuple(sorted(cast(set[str], values)))


def _contract_resource_artifacts(contract: Any, contract_path: Path) -> dict[str, bytes]:
    from modelpact.contracts.parser import resolve_contract_resource

    resources = {
        item.source for item in (*contract.objectives, *contract.targets, *contract.guards)
    }
    resources.update(_schema_files(contract))
    resources.update(
        source
        for source in (contract.holdout.targets, contract.holdout.guards)
        if source is not None
    )
    result: dict[str, bytes] = {}
    for relative in sorted(resources):
        source = resolve_contract_resource(contract_path, relative)
        result[f"contracts/{Path(relative).as_posix()}"] = source.read_bytes()
    return result


def _verification_report(
    *,
    loaded: _LoadedModel,
    model: nn.Module,
    base_model: nn.Module,
    contract: Any,
    contract_path: Path,
    candidate_id: str,
    include_holdout: bool,
) -> Any:
    from modelpact.contracts.holdout import HoldoutPhase, SealedHoldoutGate
    from modelpact.verify.engine import ExecutionIdentity, verify_contract
    from modelpact.verify.provider import ModelBackedRecordProvider, load_json_schemas

    provider = ModelBackedRecordProvider(
        adapter=loaded.adapter,
        model=model,
        base_model=base_model,
        contract_root=contract_path.parent,
        generation_policy=contract.generation,
    )
    schemas = load_json_schemas(contract_path.parent, _schema_files(contract))
    gate = None
    capability = None
    execute_holdout = include_holdout and contract.holdout.configured
    if execute_holdout:
        gate = SealedHoldoutGate(contract, allow_independent=True)
        capability = gate.authorize(
            phase=HoldoutPhase.INDEPENDENT_VERIFICATION,
            candidate_id=candidate_id,
        )
    signature = loaded.manifest.signature
    identity = ExecutionIdentity(
        adapter_id=loaded.adapter.adapter_id,
        base_signature=signature.signature_hash,
        tokenizer_hash=signature.tokenizer_hash,
        architecture_hash=signature.architecture_hash,
        state_schema_hash=signature.state_schema_hash,
    )
    return verify_contract(
        contract,
        identity=identity,
        provider=provider,
        schemas=schemas,
        include_holdout=execute_holdout,
        holdout_gate=gate,
        holdout_capability=capability,
    )


def _outcome_exit(outcome: VerificationOutcome) -> int:
    if outcome is VerificationOutcome.PASS:
        return 0
    if outcome is VerificationOutcome.FAIL:
        return EXIT_FAILED
    if outcome is VerificationOutcome.UNSUPPORTED:
        return EXIT_UNSUPPORTED
    return EXIT_INCONCLUSIVE


def _certificate_union(
    reports: Sequence[Any],
    contract_entries: Sequence[tuple[Any, Path]],
) -> tuple[Any, Any, dict[str, str], dict[str, object]]:
    """Construct one certificate view over an executed contract union.

    Raw contract reports remain independently inspectable.  This synthetic view
    exists only to prevent a target-only or guard-only document from being
    mistaken for a complete patch certificate while still supporting bundles
    that intentionally split target and preservation contracts.
    """

    from modelpact.contracts.ast import BehaviorContract, HoldoutPolicy
    from modelpact.status import PatchClaim
    from modelpact.verify.engine import VerificationReport, combine_outcomes

    if not reports or len(reports) != len(contract_entries):
        raise ValueError("certificate union requires matching reports and contracts")
    identities = {hash_canonical(report.identity.to_dict()) for report in reports}
    if len(identities) != 1:
        raise ValueError("contract reports were executed under different model identities")

    scoped_contracts = [item[0] for item in contract_entries]

    def scoped(identifier: str, contract_hash: str) -> str:
        prefix = f"c{contract_hash.removeprefix('sha256:')[:10]}-"
        return prefix + identifier[: 128 - len(prefix)]

    objectives = tuple(
        dataclasses.replace(
            objective,
            id=scoped(objective.id, contract.contract_id),
        )
        for contract in scoped_contracts
        for objective in contract.objectives
    )
    targets = tuple(
        dataclasses.replace(
            assertion,
            id=scoped(assertion.id, contract.contract_id),
        )
        for contract in scoped_contracts
        for assertion in contract.targets
    )
    guards = tuple(
        dataclasses.replace(
            assertion,
            id=scoped(assertion.id, contract.contract_id),
        )
        for contract in scoped_contracts
        for assertion in contract.guards
    )
    union_id = (
        "bundle-union-"
        + hash_canonical(
            sorted(contract.contract_id for contract in scoped_contracts)
        ).removeprefix("sha256:")[:16]
    )
    first_contract = scoped_contracts[0]
    union_contract = BehaviorContract(
        schema_version=1,
        id=union_id,
        contract_version=1,
        model_requirements=first_contract.model_requirements,
        objectives=objectives,
        targets=targets,
        guards=guards,
        holdout=HoldoutPolicy(
            sealed=True,
            unseal_policy=first_contract.holdout.unseal_policy,
        ),
        statistics=first_contract.statistics,
        generation=first_contract.generation,
        description="Deterministic certificate view over separately executed bundle contracts.",
    )

    def scoped_results(attribute: str) -> tuple[Any, ...]:
        return tuple(
            dataclasses.replace(
                result,
                assertion_id=scoped(result.assertion_id, contract.contract_id),
            )
            for report, contract in zip(reports, scoped_contracts, strict=True)
            for result in getattr(report, attribute)
        )

    target_results = scoped_results("target_results")
    guard_results = scoped_results("guard_results")
    holdout_targets = scoped_results("holdout_target_results")
    holdout_guards = scoped_results("holdout_guard_results")
    outcome = combine_outcomes(tuple(report.outcome for report in reports))
    warnings = {warning for report in reports for warning in report.warnings}
    if not target_results or not guard_results:
        if outcome is VerificationOutcome.PASS:
            outcome = VerificationOutcome.INCONCLUSIVE
        warnings.add(
            "successful patch certification requires executed target and preservation assertions"
        )
    holdout_outcome = combine_outcomes(tuple(report.holdout_outcome for report in reports))
    unsupported = {claim for report in reports for claim in report.unsupported_claims}
    if guard_results and all(item.outcome is VerificationOutcome.PASS for item in guard_results):
        unsupported.discard(PatchClaim.PRESERVATION_ASSERTIONS_VERIFIED.value)
    if (holdout_targets or holdout_guards) and holdout_outcome is VerificationOutcome.PASS:
        unsupported.discard(PatchClaim.SEALED_HOLDOUT_VERIFIED.value)

    probe_hashes: dict[str, str] = {}
    compatibility_errors: list[str] = []
    free_generation = []
    for report, contract in zip(reports, scoped_contracts, strict=True):
        prefix = contract.contract_id.removeprefix("sha256:")[:12]
        probe_hashes.update(
            {f"{prefix}/{name}": digest for name, digest in report.probe_hashes.items()}
        )
        compatibility_errors.extend(
            f"{contract.id}: {error}" for error in report.compatibility_errors
        )
        free_generation.extend(report.free_generation_records)
    union_report = VerificationReport(
        schema_version=1,
        contract_id=union_contract.id,
        contract_hash=union_contract.contract_id,
        identity=reports[0].identity,
        outcome=outcome,
        target_results=target_results,
        guard_results=guard_results,
        holdout_target_results=holdout_targets,
        holdout_guard_results=holdout_guards,
        holdout_outcome=holdout_outcome,
        free_generation_records=tuple(free_generation),
        probe_hashes=probe_hashes,
        compatibility_errors=tuple(sorted(compatibility_errors)),
        warnings=tuple(sorted(warnings)),
        unsupported_claims=tuple(sorted(unsupported)),
    )
    identifiers = [contract.id for contract in scoped_contracts]
    duplicate_ids = {item for item in identifiers if identifiers.count(item) > 1}
    contract_hashes: dict[str, str] = {
        (
            f"{contract.id}@{contract.contract_id.removeprefix('sha256:')[:12]}"
            if contract.id in duplicate_ids
            else contract.id
        ): contract.contract_id
        for contract in scoped_contracts
    }
    policy: dict[str, object] = {
        "contracts": {
            contract.contract_id: {
                "generation": contract.generation.to_dict(),
                "statistics": contract.statistics.to_dict(),
            }
            for contract in scoped_contracts
        }
    }
    return union_report, union_contract, contract_hashes, policy


@app.command("inspect")
def inspect_command(
    patch: Path = typer.Argument(..., exists=True),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Re-hash and inspect a Behavior Patch Bundle without executing it."""

    def operation() -> _CommandResult:
        bundle = load_patch_bundle(patch)
        tensor_summary = {
            name: {
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "shape": list(tensor.shape),
            }
            for name, tensor in sorted(bundle.tensors.items())
        }
        return _CommandResult(
            {
                "delta_program": bundle.program.to_dict(),
                "bundle_id": bundle.bundle_id,
                "evidence_id": bundle.evidence_id,
                "estimated_patch_bytes": bundle.program.estimate_bytes(bundle.tensors),
                "manifest": bundle.manifest.to_dict(),
                "missing_complete_bundle_artifacts": list(
                    missing_bundle_artifacts(bundle.manifest)
                ),
                "status": "PASS",
                "tensors": tensor_summary,
            }
        )

    _invoke(operation, compact=json_output)


@app.command("apply")
def apply_command(
    base_checkpoint: Path = typer.Argument(..., exists=True),
    patches: list[Path] = typer.Argument(..., exists=True),
    output: Path = typer.Option(..., "--output"),
    adapter_spec: str = typer.Option("tiny", "--adapter"),
    mode: str = typer.Option("materialize", "--mode"),
    max_shard_size: int = typer.Option(2 * 1024**3, "--max-shard-size", min=1),
    device: str = typer.Option("cpu", "--device"),
    dtype: str = typer.Option("float32", "--dtype"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Materialize or ephemerally execute a deterministic additive patch stack."""

    from modelpact.checkpoints.safetensors import tensor_content_hash
    from modelpact.compose.closure import PatchOperand, additive_compose
    from modelpact.patch.fold import materialize_patch
    from modelpact.patch.mount import mount_patch
    from modelpact.patch.validate import validate_base_signature

    def operation() -> _CommandResult:
        if mode not in {"materialize", "runtime"}:
            raise ValueError("--mode must be either 'materialize' or 'runtime'")
        if not patches:
            raise ValueError("at least one patch is required")
        loaded = _load_model(adapter_spec, base_checkpoint, device=device, dtype=dtype)
        schema = loaded.manifest.state_schema
        bundles = [load_patch_bundle(path, state_schema=schema) for path in patches]
        for bundle in bundles:
            validate_base_signature(bundle.manifest.base_signature, loaded.manifest.signature)
        operands = [
            PatchOperand(
                patch_id=bundle.manifest.patch_id,
                base_signature=loaded.manifest.signature.signature_hash,
                module_schema_hash=schema.schema_hash,
                delta=_bundle_dense_delta(bundle),
                contract_ids=tuple(
                    sorted(set(bundle.manifest.provides) | set(bundle.manifest.preserves))
                )
                or (f"unscoped:{bundle.manifest.patch_id}",),
            )
            for bundle in bundles
        ]
        resolved = additive_compose(operands, aliases=_alias_map(schema))
        program, tensors = _dense_program(resolved, schema)
        patch_order = sorted(bundle.manifest.patch_id for bundle in bundles)

        if mode == "runtime":
            resolved_output = output.resolve(strict=False)
            resolved_checkpoint = base_checkpoint.resolve()
            if resolved_output == resolved_checkpoint or (
                resolved_checkpoint.is_dir() and resolved_checkpoint in resolved_output.parents
            ):
                raise ValueError("runtime descriptor output cannot be inside the source checkpoint")

            def runtime_value(target: str) -> Tensor:
                module_path, separator, parameter_name = target.rpartition(".")
                module = loaded.model.get_submodule(module_path) if separator else loaded.model
                value = getattr(module, parameter_name)
                if not isinstance(value, Tensor):
                    raise TypeError(f"runtime patch target is not a tensor: {target}")
                return value

            before_hashes = {
                name: tensor_content_hash(value)
                for name, value in sorted(loaded.model.state_dict().items())
            }
            before_state_hash = hash_canonical(
                {"schema_version": 1, "tensor_hashes": before_hashes}
            )
            base_targets = {
                target: runtime_value(target).detach().clone() for target in sorted(program.targets)
            }
            target_devices = {value.device for value in base_targets.values()}
            if len(target_devices) != 1:
                raise ValueError("runtime apply requires all resolved targets on one device")
            runtime_device = next(iter(target_devices))
            runtime_tensors = {
                name: value.to(device=runtime_device) for name, value in sorted(tensors.items())
            }

            session = mount_patch(
                loaded.model,
                program,
                runtime_tensors,
                state_schema=schema,
            )
            target_checks: list[dict[str, object]] = []
            try:
                for target in sorted(program.targets):
                    delta = program.materialize(target, runtime_tensors)
                    expected = base_targets[target] + delta
                    mounted = runtime_value(target).detach()
                    matches = (
                        mounted.shape == expected.shape
                        and mounted.dtype == expected.dtype
                        and mounted.device == expected.device
                        and torch.equal(mounted, expected)
                    )
                    target_checks.append(
                        {
                            "base_tensor_hash": tensor_content_hash(base_targets[target]),
                            "delta_tensor_hash": tensor_content_hash(delta),
                            "expected_tensor_hash": tensor_content_hash(expected),
                            "matches_expected": matches,
                            "mounted_tensor_hash": tensor_content_hash(mounted),
                            "target": target,
                        }
                    )
                if not all(bool(check["matches_expected"]) for check in target_checks):
                    raise RuntimeError(
                        "runtime-mounted tensor did not equal base plus resolved delta"
                    )
            finally:
                session.unmount()
                after_hashes = {
                    name: tensor_content_hash(value)
                    for name, value in sorted(loaded.model.state_dict().items())
                }
                if before_hashes != after_hashes:
                    raise RuntimeError("runtime unmount did not restore the base state bitwise")

            after_manifest = build_model_manifest(
                loaded.model,
                checkpoint=base_checkpoint,
                adapter_id=loaded.adapter.adapter_id,
            )
            if after_manifest.to_dict() != loaded.manifest.to_dict():
                raise RuntimeError("source checkpoint identity changed during runtime execution")

            resolved_tensor_hashes = {
                target: tensor_content_hash(value) for target, value in sorted(resolved.items())
            }
            record: dict[str, object] = {
                "adapter_id": loaded.adapter.adapter_id,
                "artifact_kind": "RUNTIME_STACK_EXECUTION_V1",
                "base_signature": loaded.manifest.signature.to_dict(),
                "ephemeral": True,
                "execution_environment": {
                    "device": str(runtime_device),
                    "requested_dtype": dtype,
                },
                "mode": "runtime",
                "mount": {
                    "executed": True,
                    "mounted_parameter_count": len(session.mounted_parameters),
                    "target_checks": target_checks,
                    "tensors_match_base_plus_resolved_delta": True,
                },
                "patch_order": patch_order,
                "persistent": False,
                "resolved_delta_hash": hash_canonical(
                    {"schema_version": 1, "tensor_hashes": resolved_tensor_hashes}
                ),
                "resolved_target_count": len(resolved),
                "schema_version": 1,
                "session_scope": "command_process",
                "source_checkpoint_identity_unchanged": True,
                "status": "PASS",
                "unmount": {
                    "base_state_bitwise_restored": True,
                    "base_state_hash": before_state_hash,
                    "executed": True,
                    "grade": "RUNTIME_UNMOUNT_EXACT",
                },
                "warning": (
                    "The runtime mount was executed and unmounted before command exit; "
                    "this descriptor does not preserve an in-memory mount."
                ),
            }
            record["execution_id"] = hash_canonical(record)
            _write_json(output, record)
            return _CommandResult({**record, "output": output.as_posix()})

        manifest = materialize_patch(
            base_checkpoint,
            output,
            program,
            tensors,
            state_schema=schema,
            max_shard_size=max_shard_size,
            patch_ids=tuple(sorted(bundle.manifest.patch_id for bundle in bundles)),
        )
        return _CommandResult(
            {
                "materialization_manifest": manifest,
                "output": output.as_posix(),
                "patch_order": patch_order,
                "resolved_delta_hash": hash_canonical(program.to_dict()),
                "status": "PASS",
            }
        )

    _invoke(operation, compact=json_output)


@app.command("verify")
def verify_command(
    patch: Path = typer.Argument(..., exists=True),
    base_checkpoint: Path = typer.Option(..., "--base", exists=True),
    adapter_spec: str = typer.Option(..., "--adapter"),
    policy: Path | None = typer.Option(None, "--policy", exists=True, readable=True),
    certificate_output: Path | None = typer.Option(None, "--certificate-output"),
    include_holdout: bool = typer.Option(True, "--include-holdout/--skip-holdout"),
    device: str = typer.Option("cpu", "--device"),
    dtype: str = typer.Option("float32", "--dtype"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Independently mount, execute, and re-certificate a patch contract."""

    from modelpact.contracts.parser import load_contract
    from modelpact.patch.mount import mount_bundle
    from modelpact.verify.certificate import build_certificate, write_certificate

    def operation() -> _CommandResult:
        loaded = _load_model(adapter_spec, base_checkpoint, device=device, dtype=dtype)
        unpatched = copy.deepcopy(loaded.model)
        bundle = load_patch_bundle(patch, state_schema=loaded.manifest.state_schema)
        bundled_contracts = _load_bundle_contracts(bundle)
        contract_entries_list = [
            value for _, value in sorted(bundled_contracts.items(), key=lambda item: item[0])
        ]
        if policy is not None:
            policy_contract = load_contract(policy)
            prior = bundled_contracts.get(policy_contract.contract_id)
            if prior is not None:
                if prior[0].to_dict() != policy_contract.to_dict():
                    raise ValueError("verification policy contract identity collision")
            else:
                contract_entries_list.append((policy_contract, policy))
        contract_entries = tuple(contract_entries_list)
        with mount_bundle(
            loaded.model,
            bundle,
            loaded.manifest.signature,
            state_schema=loaded.manifest.state_schema,
        ):
            reports = tuple(
                _verification_report(
                    loaded=loaded,
                    model=loaded.model,
                    base_model=unpatched,
                    contract=contract,
                    contract_path=contract_path,
                    candidate_id=bundle.manifest.patch_id,
                    include_holdout=include_holdout,
                )
                for contract, contract_path in contract_entries
            )
        artifact_paths = tuple(sorted(bundle.manifest.artifact_hashes))
        artifact_hashes = {
            relative: sha256_file(bundle.path / relative) for relative in artifact_paths
        }
        union_report, union_contract, contract_hashes, verification_policy = _certificate_union(
            reports, contract_entries
        )
        certificate = build_certificate(
            union_report,
            union_contract,
            patch_id=bundle.manifest.patch_id,
            checkpoint_hashes=loaded.manifest.checkpoint_tensor_hashes,
            artifact_hashes=artifact_hashes,
            verification_policy=verification_policy,
            contract_hashes=contract_hashes,
            patch_structure={
                "active_targets": sorted(bundle.program.targets),
                "patch_bytes": bundle.program.estimate_bytes(bundle.tensors),
            },
            additional_warnings=(
                "Certificate was regenerated from model execution; bundled outcomes "
                "were not trusted.",
            ),
        )
        certificate_output_value: str | None = None
        if certificate_output is not None:
            write_certificate(certificate, certificate_output, overwrite=False)
            certificate_output_value = certificate_output.as_posix()
        overall = union_report.outcome
        payload: dict[str, object] = {
            "bundle_id": bundle.bundle_id,
            "evidence_id": bundle.evidence_id,
            "certificate": certificate.to_dict(),
            "certificate_output": certificate_output_value,
            "certificates": [certificate.to_dict()],
            "prompt_failures": _jsonable(
                tuple(failure for report in reports for failure in report.prompt_failures)
            ),
            "reports": [report.to_dict() for report in reports],
            "status": overall.value,
        }
        if len(reports) == 1:
            payload["report"] = reports[0].to_dict()
        return _CommandResult(payload, _outcome_exit(overall))

    _invoke(operation, compact=json_output)


def _report_sections(report: Any) -> tuple[dict[str, object], dict[str, object]]:
    validation = {
        "compatibility_errors": list(report.compatibility_errors),
        "guard_results": [item.to_dict() for item in report.guard_results],
        "outcome": report.outcome.value,
        "prompt_failures": _jsonable(report.prompt_failures),
        "schema_version": 1,
        "target_results": [item.to_dict() for item in report.target_results],
        "warnings": list(report.warnings),
    }
    holdout = {
        "guard_results": [item.to_dict() for item in report.holdout_guard_results],
        "outcome": report.holdout_outcome.value,
        "schema_version": 1,
        "target_results": [item.to_dict() for item in report.holdout_target_results],
    }
    return validation, holdout


@app.command("compile")
def compile_command(
    base: str = typer.Option(..., "--base", help="Trusted local adapter."),
    checkpoint: Path = typer.Option(..., "--checkpoint", exists=True),
    spec: Path = typer.Option(..., "--spec", exists=True, readable=True),
    output: Path = typer.Option(..., "--output"),
    max_rank: int = typer.Option(16, "--max-rank", min=1, max=4096),
    max_modules: int = typer.Option(12, "--max-modules", min=1, max=100_000),
    steps: int = typer.Option(200, "--steps", min=1, max=10_000_000),
    cegis_rounds: int = typer.Option(0, "--cegis-rounds", min=0, max=10_000),
    cegis_search_budget: int = typer.Option(
        32,
        "--cegis-search-budget",
        min=1,
        max=100_000,
        help="Executed mutations per target/guard domain in each CEGIS round.",
    ),
    minimization_budget: int = typer.Option(
        32,
        "--minimization-budget",
        min=1,
        max=100_000,
    ),
    seed: int = typer.Option(0, "--seed", min=0),
    device: str = typer.Option("cpu", "--device"),
    dtype: str = typer.Option("float32", "--dtype"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Compile and independently validate a bounded low-rank patch candidate."""

    from modelpact.codegen import emit_apply_script, emit_verify_script
    from modelpact.compiler.contracts import prepare_contract
    from modelpact.compiler.generic_cegis import (
        GenericCEGISRun,
        GenericCEGISUnsupportedError,
        candidate_satisfies_working_set,
        run_generic_cegis,
    )
    from modelpact.compiler.minimize import minimize_patch
    from modelpact.compiler.optimize import OptimizerConfig, compile_low_rank_patch
    from modelpact.compiler.package import compilation_delta_program, compile_evidence
    from modelpact.contracts.parser import load_contract
    from modelpact.patch.bundle import attach_bundle_artifacts, create_patch_bundle
    from modelpact.patch.mount import mount_patch
    from modelpact.verify.certificate import build_certificate

    def operation() -> _CommandResult:
        loaded = _load_model(base, checkpoint, device=device, dtype=dtype)
        contract = load_contract(spec)
        if not contract.targets or not contract.guards:
            return _CommandResult(
                {
                    "reason": (
                        "successful patch compilation requires at least one target assertion "
                        "and one preservation guard assertion"
                    ),
                    "status": VerificationOutcome.UNSUPPORTED.value,
                },
                EXIT_UNSUPPORTED,
            )
        prepared = prepare_contract(loaded.adapter, loaded.model, contract, spec)
        configuration = OptimizerConfig(
            maximum_rank=max_rank,
            maximum_modules=max_modules,
            steps=steps,
            seed=seed,
        )
        cegis_run: GenericCEGISRun | None = None
        if cegis_rounds:
            try:
                cegis_run = run_generic_cegis(
                    loaded.adapter,
                    loaded.model,
                    contract,
                    spec,
                    prepared,
                    configuration,
                    maximum_rounds=cegis_rounds,
                    search_budget_per_domain_per_round=cegis_search_budget,
                )
            except GenericCEGISUnsupportedError as error:
                return _CommandResult(
                    {
                        "reason": str(error),
                        "requested_cegis_rounds": cegis_rounds,
                        "search_budget_per_domain_per_round": cegis_search_budget,
                        "status": VerificationOutcome.UNSUPPORTED.value,
                    },
                    EXIT_UNSUPPORTED,
                )
            result = cegis_run.candidate
            cegis_evidence = cegis_run.to_dict()
        else:
            result = compile_low_rank_patch(
                loaded.model,
                prepared.objectives,
                prepared.guards,
                config=configuration,
            )
            cegis_evidence = {
                "outcome": "NOT_EXECUTED",
                "reason": "--cegis-rounds was zero",
                "rounds_executed": 0,
                "schema_version": 1,
            }
        evidence = {**compile_evidence(result), "cegis": cegis_evidence}
        if not result.feasible:
            output.mkdir(parents=True, exist_ok=False)
            _write_json(output / "compile-failure.json", evidence)
            return _CommandResult(
                {
                    "evidence": evidence,
                    "output": output.as_posix(),
                    "status": result.status.value,
                },
                EXIT_FAILED,
            )

        unpatched = copy.deepcopy(loaded.model)

        def candidate_passes(candidate: dict[str, Tensor]) -> bool:
            parameter_deltas = {
                f"{name}.weight": value for name, value in sorted(candidate.items())
            }
            candidate_program, candidate_tensors = _dense_program(
                parameter_deltas,
                loaded.manifest.state_schema,
            )
            with mount_patch(
                loaded.model,
                candidate_program,
                candidate_tensors,
                state_schema=loaded.manifest.state_schema,
            ):
                candidate_report = _verification_report(
                    loaded=loaded,
                    model=loaded.model,
                    base_model=unpatched,
                    contract=contract,
                    contract_path=spec,
                    candidate_id=f"minimize:{hash_canonical(candidate_program.to_dict())}",
                    include_holdout=False,
                )
            contract_passed = candidate_report.outcome is VerificationOutcome.PASS
            working_set_passed = cegis_run is None or candidate_satisfies_working_set(
                cegis_run,
                loaded.adapter,
                unpatched,
                contract,
                candidate,
            )
            return contract_passed and working_set_passed

        try:
            minimization = minimize_patch(
                result.deltas,
                candidate_passes,
                verification_budget=minimization_budget,
                seed=seed,
                initial_factors=result.factors,
            )
        except ValueError as error:
            output.mkdir(parents=True, exist_ok=False)
            failure = {
                "compiler": evidence,
                "reason": str(error),
                "schema_version": 1,
                "status": VerificationOutcome.FAIL.value,
            }
            _write_json(output / "validation-failure.json", failure)
            return _CommandResult(failure, EXIT_FAILED)
        result = dataclasses.replace(
            result,
            deltas={name: value.detach().clone() for name, value in minimization.deltas.items()},
            factors={
                name: (left.detach().clone(), right.detach().clone())
                for name, (left, right) in minimization.factors.items()
            },
            active_modules=tuple(sorted(minimization.factors)),
            ranks={name: left.shape[1] for name, (left, _right) in minimization.factors.items()},
            metadata={
                **result.metadata,
                "minimization_claims": [item.value for item in minimization.claims],
                "minimization_verification_budget_used": minimization.verification_budget_used,
            },
        )
        post_minimization_cegis_passed = cegis_run is None or candidate_satisfies_working_set(
            cegis_run,
            loaded.adapter,
            unpatched,
            contract,
            result.deltas,
        )
        if not post_minimization_cegis_passed:
            raise RuntimeError("minimized candidate regressed the accumulated CEGIS working set")
        cegis_evidence = {
            **cegis_evidence,
            "post_minimization_working_set_passed": post_minimization_cegis_passed,
        }
        evidence["cegis"] = cegis_evidence
        program, tensors = compilation_delta_program(result, loaded.manifest.state_schema)
        with mount_patch(
            loaded.model,
            program,
            tensors,
            state_schema=loaded.manifest.state_schema,
        ):
            report = _verification_report(
                loaded=loaded,
                model=loaded.model,
                base_model=unpatched,
                contract=contract,
                contract_path=spec,
                candidate_id=f"candidate:{hash_canonical(program.to_dict())}",
                include_holdout=True,
            )
        validation, holdout = _report_sections(report)
        contract_bytes = (canonical_dumps(contract.to_dict()) + "\n").encode()
        probe_hashes = dict(sorted(report.probe_hashes.items()))
        supplemental = {
            "evidence/compile.json": (canonical_dumps(evidence) + "\n").encode(),
            "evidence/holdout.json": (canonical_dumps(holdout) + "\n").encode(),
            "evidence/minimization.json": (
                canonical_dumps(
                    {
                        "candidates": [
                            {
                                "active_modules": list(item.active_modules),
                                "operation": item.operation,
                                "passed": item.passed,
                                "ranks": dict(sorted(item.ranks.items())),
                            }
                            for item in minimization.candidates
                        ],
                        "claims": [item.value for item in minimization.claims],
                        "schema_version": 1,
                        "verification_budget": minimization_budget,
                        "verification_budget_used": minimization.verification_budget_used,
                    }
                )
                + "\n"
            ).encode(),
            "evidence/validation.json": (canonical_dumps(validation) + "\n").encode(),
            "probes/hashes.json": (canonical_dumps(probe_hashes) + "\n").encode(),
            "probes/manifest.json": (
                canonical_dumps(
                    {
                        "roles": ["compile", "validation", "guard", "holdout"],
                        "schema_version": 1,
                        "source_hashes": probe_hashes,
                    }
                )
                + "\n"
            ).encode(),
            "report.md": (
                "# ModelPact compilation report\n\n"
                f"Compiler status: {result.status.value}\n\n"
                f"Executed verification: {report.outcome.value}\n\n"
                "Claims apply only to the declared contracts and executed probes.\n"
            ).encode(),
        }
        contract_artifacts = {
            "contracts/preservation.yaml": contract_bytes,
            "contracts/target.yaml": contract_bytes,
            **_contract_resource_artifacts(contract, spec),
        }
        bundle = create_patch_bundle(
            output,
            name=contract.id,
            base_signature=loaded.manifest.signature.to_dict(),
            state_schema=loaded.manifest.state_schema,
            program=program,
            tensors=tensors,
            tool_version=__version__,
            contracts=contract_artifacts,
            supplemental_artifacts=supplemental,
            provides=(contract.contract_id,),
            preserves=(contract.contract_id,),
            verification_policy_hash=hash_canonical(
                {
                    "generation": contract.generation.to_dict(),
                    "statistics": contract.statistics.to_dict(),
                }
            ),
            compiler_configuration={
                **cast(Mapping[str, object], _jsonable(configuration)),
                "cegis_rounds": cegis_rounds,
                "cegis_search_budget_per_domain_per_round": cegis_search_budget,
            },
        )
        certificate = build_certificate(
            report,
            contract,
            patch_id=bundle.manifest.patch_id,
            checkpoint_hashes=loaded.manifest.checkpoint_tensor_hashes,
            artifact_hashes=dict(bundle.manifest.artifact_hashes),
            counterexample_search=cegis_evidence,
            patch_structure={
                "active_modules": list(result.active_modules),
                "module_ranks": dict(sorted(result.ranks.items())),
                "patch_bytes": program.estimate_bytes(tensors),
            },
            objectives_optimized=True,
            minimized_within_budget=True,
        )
        temporary = Path(tempfile.mkdtemp(prefix="modelpact-codegen-"))
        try:
            apply_path = emit_apply_script(
                output, temporary / "apply_patch.py", will_live_in_bundle=True
            )
            verify_path = emit_verify_script(
                output, temporary / "verify_patch.py", will_live_in_bundle=True
            )
            bundle = attach_bundle_artifacts(
                output,
                {
                    "apply_patch.py": apply_path.read_bytes(),
                    "certificate.json": (certificate.canonical_json() + "\n").encode(),
                    "verify_patch.py": verify_path.read_bytes(),
                },
                state_schema=loaded.manifest.state_schema,
                require_complete=True,
            )
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
        return _CommandResult(
            {
                "active_modules": list(result.active_modules),
                "certificate_hash": certificate.certificate_hash,
                "cegis": {
                    **cegis_evidence,
                },
                "holdout_outcome": report.holdout_outcome.value,
                "output": output.as_posix(),
                "patch_bytes": program.estimate_bytes(tensors),
                "patch_id": bundle.manifest.patch_id,
                "bundle_id": bundle.bundle_id,
                "evidence_id": bundle.evidence_id,
                "minimization_claims": [item.value for item in minimization.claims],
                "minimization_verification_budget_used": minimization.verification_budget_used,
                "status": report.outcome.value,
                "verification_result_hash": report.result_hash,
            },
            _outcome_exit(report.outcome),
        )

    _invoke(operation, compact=json_output)


_DIFF_MAX_ARTIFACTS = 4096
_DIFF_MAX_TOTAL_ARTIFACT_BYTES = 4 * 1024**3
_DIFF_MAX_GENERIC_ARTIFACT_BYTES = 2 * 1024**3
_DIFF_MAX_CLUSTERS_BYTES = 16 * 1024**2
_DIFF_MAX_WITNESSES_BYTES = 256 * 1024**2
_DIFF_MAX_WITNESS_UNCOMPRESSED_BYTES = 512 * 1024**2
_DIFF_MAX_WITNESSES = 100_000
_DIFF_WITNESS_FIELDS = frozenset(
    {
        "witness_id",
        "input_hash",
        "original_input",
        "minimized_input",
        "divergence_metrics",
        "base_output_hash",
        "target_output_hash",
        "activation_fingerprint",
        "gradient_fingerprint",
        "prompt_fingerprint",
        "provenance",
    }
)


def _bounded_witness_value(value: object, *, location: str) -> None:
    nodes = 0

    def visit(item: object, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > 10_000 or depth > 8:
            raise ValueError(f"{location} exceeds nested value limits")
        if item is None or isinstance(item, bool | int):
            return
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError(f"{location} contains a non-finite number")
            return
        if isinstance(item, str):
            if len(item) > 65_536 or "\x00" in item:
                raise ValueError(f"{location} contains an oversized or invalid string")
            return
        if isinstance(item, Mapping):
            if len(item) > 1_000 or any(not isinstance(key, str) for key in item):
                raise ValueError(f"{location} contains a malformed mapping")
            for key, nested in item.items():
                visit(key, depth + 1)
                visit(nested, depth + 1)
            return
        if isinstance(item, Sequence) and not isinstance(item, str | bytes | bytearray):
            if len(item) > 10_000:
                raise ValueError(f"{location} contains an oversized sequence")
            for nested in item:
                visit(nested, depth + 1)
            return
        raise ValueError(f"{location} contains unsupported value type {type(item).__name__}")

    visit(value, 0)


def _witness_from_row(row: Mapping[str, object]) -> Any:
    from modelpact.diff.witnesses import DifferenceWitness

    required = {
        "witness_id",
        "input_hash",
        "original_input",
        "minimized_input",
        "divergence_metrics",
        "base_output_hash",
        "target_output_hash",
    }
    unknown = set(row) - _DIFF_WITNESS_FIELDS
    if unknown:
        raise ValueError(f"difference witness row has unknown fields: {sorted(unknown)}")
    if not required <= set(row):
        raise ValueError(f"difference witness row is missing fields: {sorted(required - set(row))}")
    for name in ("witness_id", "input_hash", "base_output_hash", "target_output_hash"):
        if not is_sha256_digest(row[name]):
            raise ValueError(f"difference witness {name} must be a lowercase sha256 digest")
    for name in ("original_input", "minimized_input"):
        text = row[name]
        if not isinstance(text, str) or not text or len(text) > 1_000_000 or "\x00" in text:
            raise ValueError(f"difference witness {name} is invalid or oversized")
    metrics = row["divergence_metrics"]
    provenance = row.get("provenance", {})
    if not isinstance(metrics, Mapping) or not isinstance(provenance, Mapping):
        raise ValueError("malformed difference witness mappings")
    if len(metrics) > 128 or any(
        not isinstance(key, str)
        or not key
        or len(key) > 128
        or isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        for key, value in metrics.items()
    ):
        raise ValueError("difference witness metrics are malformed or non-finite")
    _bounded_witness_value(provenance, location="difference witness provenance")

    def float_tuple(name: str) -> tuple[float, ...]:
        value = row.get(name, ())
        if (
            not isinstance(value, Sequence)
            or isinstance(value, str | bytes | bytearray)
            or len(value) > 4096
        ):
            raise ValueError(f"malformed difference witness field: {name}")
        converted = tuple(float(item) for item in value)
        if any(not math.isfinite(item) for item in converted):
            raise ValueError(f"non-finite difference witness field: {name}")
        return converted

    return DifferenceWitness(
        witness_id=cast(str, row["witness_id"]),
        input_hash=cast(str, row["input_hash"]),
        original_input=cast(str, row["original_input"]),
        minimized_input=cast(str, row["minimized_input"]),
        divergence_metrics={str(key): float(value) for key, value in metrics.items()},
        base_output_hash=cast(str, row["base_output_hash"]),
        target_output_hash=cast(str, row["target_output_hash"]),
        activation_fingerprint=float_tuple("activation_fingerprint"),
        gradient_fingerprint=float_tuple("gradient_fingerprint"),
        prompt_fingerprint=float_tuple("prompt_fingerprint"),
        provenance={str(key): value for key, value in provenance.items()},
    )


def _load_extraction_clusters(path: Path) -> tuple[Mapping[str, object], ...]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > _DIFF_MAX_CLUSTERS_BYTES:
        raise ValueError("difference cluster index is missing, linked, or oversized")
    value = strict_json_loads(path.read_bytes(), max_depth=16)
    if not isinstance(value, list) or len(value) > _DIFF_MAX_WITNESSES:
        raise ValueError("malformed or oversized difference cluster index")
    clusters: list[Mapping[str, object]] = []
    cluster_ids: set[str] = set()
    assigned_witnesses: set[str] = set()
    total_references = 0
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"difference cluster {index} must be an object")
        cluster_id = item.get("cluster_id")
        witness_ids = item.get("witness_ids")
        if (
            not isinstance(cluster_id, str)
            or not cluster_id
            or len(cluster_id) > 128
            or cluster_id in cluster_ids
        ):
            raise ValueError(f"difference cluster {index} has an invalid or duplicate ID")
        if (
            not isinstance(witness_ids, list)
            or not witness_ids
            or len(witness_ids) > _DIFF_MAX_WITNESSES
            or any(not is_sha256_digest(identifier) for identifier in witness_ids)
            or len(witness_ids) != len(set(cast(list[str], witness_ids)))
        ):
            raise ValueError(f"difference cluster {cluster_id} has malformed witness IDs")
        identifiers = set(cast(list[str], witness_ids))
        if assigned_witnesses & identifiers:
            raise ValueError("difference witnesses cannot belong to multiple clusters")
        cluster_ids.add(cluster_id)
        assigned_witnesses.update(identifiers)
        total_references += len(identifiers)
        if total_references > _DIFF_MAX_WITNESSES:
            raise ValueError("difference cluster witness references exceed limit")
        for numeric_name in ("dispersion", "uncertainty"):
            numeric = item.get(numeric_name)
            if numeric is not None and (
                isinstance(numeric, bool)
                or not isinstance(numeric, int | float)
                or not math.isfinite(float(numeric))
            ):
                raise ValueError(f"difference cluster {cluster_id} has non-finite metadata")
        clusters.append(item)
    return tuple(clusters)


def _load_extraction_witnesses(path: Path) -> tuple[Any, ...]:
    import pyarrow.parquet as parquet  # type: ignore[import-untyped]

    if path.is_symlink() or not path.is_file() or path.stat().st_size > _DIFF_MAX_WITNESSES_BYTES:
        raise ValueError("difference witness table is missing, linked, or oversized")
    try:
        source = parquet.ParquetFile(path)
    except Exception as error:
        raise ValueError("difference witness table is not valid Parquet") from error
    metadata = source.metadata
    if (
        metadata.num_rows <= 0
        or metadata.num_rows > _DIFF_MAX_WITNESSES
        or metadata.num_columns > 128
        or metadata.num_row_groups > 10_000
    ):
        raise ValueError("difference witness table dimensions exceed limits")
    names = set(source.schema_arrow.names)
    required = {
        "witness_id",
        "input_hash",
        "original_input",
        "minimized_input",
        "divergence_metrics",
        "base_output_hash",
        "target_output_hash",
    }
    if not required <= names or names - _DIFF_WITNESS_FIELDS:
        raise ValueError("difference witness table has an unsupported schema")
    uncompressed = sum(
        metadata.row_group(index).total_byte_size for index in range(metadata.num_row_groups)
    )
    if uncompressed > _DIFF_MAX_WITNESS_UNCOMPRESSED_BYTES:
        raise ValueError("difference witness table uncompressed size exceeds limit")
    witnesses: list[Any] = []
    try:
        for batch in source.iter_batches(batch_size=256):
            for raw in batch.to_pylist():
                if not isinstance(raw, Mapping):
                    raise ValueError("difference witness row must be an object")
                witnesses.append(_witness_from_row(cast(Mapping[str, object], raw)))
                if len(witnesses) > _DIFF_MAX_WITNESSES:
                    raise ValueError("difference witness table exceeds row limit")
    except ValueError:
        raise
    except Exception as error:
        raise ValueError("difference witness table could not be decoded safely") from error
    if len(witnesses) != metadata.num_rows:
        raise ValueError("difference witness table row count changed during decoding")
    return tuple(witnesses)


def _verified_diff_manifest(
    diff_bundle: Path,
    *,
    base_manifest: ModelManifest,
    target_manifest: ModelManifest,
) -> Mapping[str, object]:
    """Authenticate a scoped diff bundle before reading extraction inputs."""

    if diff_bundle.is_symlink() or not diff_bundle.is_dir():
        raise ValueError("difference bundle must be a regular directory")
    root = diff_bundle.resolve()
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("difference bundle omits a regular manifest.json")
    if manifest_path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("difference bundle manifest exceeds size limit")
    value = strict_json_loads(manifest_path.read_bytes())
    if not isinstance(value, Mapping) or value.get("schema_version") != 1:
        raise ValueError("malformed or unsupported difference bundle manifest")
    artifacts = value.get("artifact_hashes")
    if not isinstance(artifacts, Mapping) or not all(
        isinstance(relative, str) and is_sha256_digest(digest)
        for relative, digest in artifacts.items()
    ):
        raise ValueError("difference bundle has malformed artifact hashes")
    if not artifacts or len(artifacts) > _DIFF_MAX_ARTIFACTS:
        raise ValueError("difference bundle artifact count exceeds limit")
    required = {"clusters.json", "witnesses.parquet"}
    if not required <= set(artifacts):
        raise ValueError("difference bundle omits extraction artifacts")
    total_bytes = 0
    for relative, expected in sorted(cast(Mapping[str, str], artifacts).items()):
        path = Path(relative.replace("\\", "/"))
        if (
            not relative
            or len(relative) > 4096
            or path.is_absolute()
            or path.drive
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError(f"unsafe difference artifact path: {relative}")
        candidate = root / path
        if (
            candidate.is_symlink()
            or not candidate.is_file()
            or root not in candidate.resolve().parents
        ):
            raise ValueError(f"difference artifact is not a regular in-bundle file: {relative}")
        limit = {
            "clusters.json": _DIFF_MAX_CLUSTERS_BYTES,
            "witnesses.parquet": _DIFF_MAX_WITNESSES_BYTES,
        }.get(relative, _DIFF_MAX_GENERIC_ARTIFACT_BYTES)
        size = candidate.stat().st_size
        total_bytes += size
        if size > limit or total_bytes > _DIFF_MAX_TOTAL_ARTIFACT_BYTES:
            raise ValueError(f"difference artifact exceeds extraction limits: {relative}")
        if sha256_file(candidate, max_bytes=limit) != expected:
            raise ValueError(f"difference artifact hash mismatch: {relative}")
    configuration = value.get("configuration")
    if not isinstance(configuration, Mapping):
        raise ValueError("difference bundle omits its model identities")
    expected_base = configuration.get("base_signature")
    expected_target = configuration.get("target_signature")
    if expected_base != base_manifest.signature.to_dict():
        raise ValueError("difference bundle base signature does not match the extraction base")
    if expected_target != target_manifest.signature.to_dict():
        raise ValueError("difference bundle target signature does not match the extraction target")
    return value


@app.command("extract")
def extract_command(
    diff_bundle: Path = typer.Argument(..., exists=True),
    cluster: str = typer.Option(..., "--cluster"),
    base: str = typer.Option(..., "--base"),
    base_checkpoint: Path = typer.Option(..., "--base-checkpoint", exists=True),
    target: str = typer.Option(..., "--target"),
    target_checkpoint: Path = typer.Option(..., "--target-checkpoint", exists=True),
    output: Path = typer.Option(..., "--output"),
    max_rank: int = typer.Option(8, "--max-rank", min=1, max=4096),
    max_modules: int = typer.Option(8, "--max-modules", min=1, max=100_000),
    steps: int = typer.Option(200, "--steps", min=1, max=10_000_000),
    cegis_rounds: int = typer.Option(2, "--cegis-rounds", min=1, max=10_000),
    search_budget: int = typer.Option(8, "--search-budget", min=1, max=1_000_000),
    validation_probes: int = typer.Option(2, "--validation-probes", min=1, max=10_000),
    holdout_probes: int = typer.Option(2, "--holdout-probes", min=1, max=10_000),
    minimization_budget: int = typer.Option(32, "--minimization-budget", min=1, max=1_000_000),
    max_new_tokens: int = typer.Option(32, "--max-new-tokens", min=1, max=4096),
    seed: int = typer.Option(0, "--seed", min=0),
    device: str = typer.Option("cpu", "--device"),
    dtype: str = typer.Option("float32", "--dtype"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Extract one empirical difference cluster while guarding nonselected clusters."""

    from modelpact.adapters.base import GenerationPolicy as AdapterGenerationPolicy
    from modelpact.codegen import emit_apply_script, emit_verify_script
    from modelpact.compiler.extract import (
        apply_dense_deltas,
        build_extraction_prompt_roles,
        run_extraction_cegis,
    )
    from modelpact.compiler.minimize import minimize_patch
    from modelpact.compiler.optimize import OptimizerConfig
    from modelpact.compiler.package import compilation_delta_program, compile_evidence
    from modelpact.compiler.result import CompilationResult
    from modelpact.contracts.parser import parse_contract
    from modelpact.patch.bundle import attach_bundle_artifacts, create_patch_bundle
    from modelpact.verify.certificate import build_certificate

    def operation() -> _CommandResult:
        base_loaded = _load_model(base, base_checkpoint, device=device, dtype=dtype)
        target_loaded = _load_model(target, target_checkpoint, device=device, dtype=dtype)
        if base_loaded.adapter.adapter_id != target_loaded.adapter.adapter_id:
            return _CommandResult(
                {
                    "reason": "extraction requires a shared adapter identity",
                    "status": "UNSUPPORTED",
                },
                EXIT_UNSUPPORTED,
            )
        if (
            base_loaded.manifest.signature.tokenizer_hash
            != target_loaded.manifest.signature.tokenizer_hash
        ):
            return _CommandResult(
                {"reason": "extraction requires compatible tokenizers", "status": "UNSUPPORTED"},
                EXIT_UNSUPPORTED,
            )
        diff_manifest = _verified_diff_manifest(
            diff_bundle,
            base_manifest=base_loaded.manifest,
            target_manifest=target_loaded.manifest,
        )
        cluster_values = _load_extraction_clusters(diff_bundle / "clusters.json")
        matching = [
            item
            for item in cluster_values
            if isinstance(item, dict) and item.get("cluster_id") == cluster
        ]
        if len(matching) != 1 or not isinstance(matching[0].get("witness_ids"), list):
            raise ValueError(f"unknown or malformed difference cluster: {cluster}")
        selected_ids = set(cast(list[str], matching[0]["witness_ids"]))
        witnesses = _load_extraction_witnesses(diff_bundle / "witnesses.parquet")
        selected = tuple(item for item in witnesses if item.witness_id in selected_ids)
        nonselected = tuple(item for item in witnesses if item.witness_id not in selected_ids)
        if {item.witness_id for item in selected} != selected_ids:
            raise ValueError("cluster references witnesses absent from witnesses.parquet")
        optimizer = OptimizerConfig(
            maximum_rank=max_rank,
            maximum_modules=max_modules,
            steps=steps,
            seed=seed,
        )
        roles = build_extraction_prompt_roles(
            selected,
            nonselected,
            maximum_rounds=cegis_rounds,
            search_budget_per_domain_per_round=search_budget,
            validation_probes_per_domain=validation_probes,
            holdout_probes_per_domain=holdout_probes,
            seed=seed,
        )
        extraction = run_extraction_cegis(
            base_loaded.adapter,
            base_loaded.model,
            target_loaded.model,
            roles,
            optimizer_config=optimizer,
            maximum_rounds=cegis_rounds,
            search_budget_per_domain_per_round=search_budget,
        )
        result = extraction.compiler_result.detached_cpu()
        final_attempt = extraction.attempts[-1]
        extraction_payload = {
            "cegis": extraction.to_dict(),
            "nonselected_base_kl": final_attempt.nonselected_base_kl,
            "nonselected_witness_ids": [item.witness_id for item in nonselected],
            "prompt_roles": roles.to_dict(),
            "selected_teacher_kl": final_attempt.selected_teacher_kl,
            "selected_witness_ids": [item.witness_id for item in selected],
            "source_witness_set_hash": diff_manifest.get("witness_set_hash"),
            "working_set_passed": final_attempt.validation_passed,
        }
        if not result.feasible:
            output.mkdir(parents=True, exist_ok=False)
            _write_json(
                output / "extraction-failure.json",
                {"compiler": compile_evidence(result), "extraction": extraction_payload},
            )
            return _CommandResult(
                {"evidence": extraction_payload, "status": result.status.value},
                EXIT_FAILED,
            )
        if not final_attempt.validation_passed:
            output.mkdir(parents=True, exist_ok=False)
            _write_json(
                output / "extraction-failure.json",
                {
                    "compiler": compile_evidence(result),
                    "extraction": extraction_payload,
                    "reason": "final CEGIS candidate failed its differentiable working set",
                    "status": "VALIDATION_FAILED",
                },
            )
            return _CommandResult(
                {
                    "evidence": extraction_payload,
                    "reason": "final CEGIS candidate failed its differentiable working set",
                    "status": "VALIDATION_FAILED",
                },
                EXIT_FAILED,
            )
        policy = AdapterGenerationPolicy(mode="greedy", max_new_tokens=max_new_tokens, seed=seed)

        def generated(model: nn.Module, prompts: Sequence[str]) -> tuple[str, ...]:
            return tuple(
                base_loaded.adapter.generate(
                    model,
                    base_loaded.adapter.tokenizer().batch([prompt]),
                    policy,
                )[0].text
                for prompt in prompts
            )

        def reference_logits(model: nn.Module, prompt: str) -> list[list[float]]:
            batch = base_loaded.adapter.tokenizer().batch([prompt])
            with torch.no_grad():
                logits = base_loaded.adapter.forward_logits(model, batch)[0]
            length = int(batch.attention_mask[0].sum().item())
            return cast(
                list[list[float]],
                logits[:length].detach().to(dtype=torch.float32, device="cpu").tolist(),
            )

        working_targets = extraction.result.working_target_examples
        working_guards = extraction.result.working_guard_examples
        compile_target_rows = [
            {
                "id": f"compile-target-{index:06d}",
                "prompt": prompt,
                "teacher_logits": reference_logits(target_loaded.model, prompt),
            }
            for index, prompt in enumerate(working_targets)
        ]
        compile_guard_rows = [
            {"id": f"compile-guard-{index:06d}", "prompt": prompt}
            for index, prompt in enumerate(working_guards)
        ]
        search_target_rows = [
            {"id": f"search-target-{index:06d}", "prompt": prompt}
            for index, prompt in enumerate(roles.search_targets)
        ]
        search_guard_rows = [
            {"id": f"search-guard-{index:06d}", "prompt": prompt}
            for index, prompt in enumerate(roles.search_guards)
        ]
        validation_target_outputs = generated(target_loaded.model, roles.validation_targets)
        validation_guard_outputs = generated(base_loaded.model, roles.validation_guards)
        validation_target_rows = [
            {
                "expected": expected,
                "id": f"validation-target-{index:06d}",
                "prompt": prompt,
                "reference_logits": reference_logits(target_loaded.model, prompt),
            }
            for index, (prompt, expected) in enumerate(
                zip(roles.validation_targets, validation_target_outputs, strict=True)
            )
        ]
        validation_guard_rows = [
            {
                "expected": expected,
                "id": f"validation-guard-{index:06d}",
                "prompt": prompt,
            }
            for index, (prompt, expected) in enumerate(
                zip(roles.validation_guards, validation_guard_outputs, strict=True)
            )
        ]
        contract_value: dict[str, object] = {
            "compile": {
                "objectives": [
                    {
                        "id": "selected-teacher-distribution",
                        "source": "data/compile-targets.jsonl",
                        "type": "teacher_kl",
                        "weight": 1.0,
                    },
                    {
                        "id": "preserve-base-distribution",
                        "source": "data/compile-guards.jsonl",
                        "type": "base_kl",
                        "weight": 1.0,
                    },
                ]
            },
            "contract_version": 1,
            "generation": {"max_new_tokens": max_new_tokens, "mode": "greedy", "seeds": [seed]},
            "holdout": {"sealed": True, "unseal_policy": "final_candidate_only"},
            "id": f"extraction-{cluster}",
            "model_requirements": {
                "adapter_id": base_loaded.adapter.adapter_id,
                "architecture_hash": base_loaded.manifest.signature.architecture_hash,
                "base_signature": base_loaded.manifest.signature.signature_hash,
                "output_semantics": "causal_lm",
                "state_schema_hash": base_loaded.manifest.signature.state_schema_hash,
                "tokenizer_hash": base_loaded.manifest.signature.tokenizer_hash,
            },
            "schema_version": 1,
            "statistics": {
                "bootstrap_samples": 200,
                "bootstrap_seed": seed,
                "confidence_level": 0.95,
            },
            "verify": {
                "guards": [
                    {
                        "id": "retain-nonselected-generations",
                        "minimum_pass_rate": 1.0,
                        "source": "data/validation-guards.jsonl",
                        "type": "free_generation_match",
                    },
                    {
                        "id": "retain-nonselected-distribution",
                        "maximum_item": 0.08,
                        "maximum_mean": 0.02,
                        "source": "data/validation-guards.jsonl",
                        "type": "base_kl",
                    },
                ],
                "targets": [
                    {
                        "id": "transfer-selected-generations",
                        "minimum_pass_rate": 1.0,
                        "source": "data/validation-targets.jsonl",
                        "type": "free_generation_match",
                    },
                    {
                        "id": "transfer-selected-distribution",
                        "maximum_item": 0.20,
                        "maximum_mean": 0.05,
                        "source": "data/validation-targets.jsonl",
                        "type": "reference_kl",
                    },
                ],
            },
        }
        contract_value["holdout"] = {
            "guards": "data/holdout-guards.jsonl",
            "sealed": True,
            "targets": "data/holdout-targets.jsonl",
            "unseal_policy": "final_candidate_only",
        }
        contract = parse_contract(contract_value)
        temporary_contract_root = Path(tempfile.mkdtemp(prefix="modelpact-extract-contract-"))
        try:
            contract_path = temporary_contract_root / "target.yaml"
            _write_json(contract_path, contract.to_dict())

            def jsonl(rows_value: Sequence[Mapping[str, object]]) -> bytes:
                return b"".join((canonical_dumps(dict(row)) + "\n").encode() for row in rows_value)

            data = {
                "data/compile-guards.jsonl": jsonl(compile_guard_rows),
                "data/compile-targets.jsonl": jsonl(compile_target_rows),
                "data/search-guards.jsonl": jsonl(search_guard_rows),
                "data/search-targets.jsonl": jsonl(search_target_rows),
                "data/validation-guards.jsonl": jsonl(validation_guard_rows),
                "data/validation-targets.jsonl": jsonl(validation_target_rows),
            }
            for relative, content in data.items():
                target_path = temporary_contract_root / relative
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.write_bytes(content)

            validation_reports: list[Any] = []

            def validates_visible_contracts(deltas: dict[str, Tensor]) -> bool:
                candidate = apply_dense_deltas(base_loaded.model, deltas)
                visible_report = _verification_report(
                    loaded=base_loaded,
                    model=candidate,
                    base_model=base_loaded.model,
                    contract=contract,
                    contract_path=contract_path,
                    candidate_id=f"extraction:{cluster}:minimization",
                    include_holdout=False,
                )
                validation_reports.append(visible_report)
                return visible_report.outcome is VerificationOutcome.PASS

            try:
                minimization = minimize_patch(
                    result.deltas,
                    validates_visible_contracts,
                    verification_budget=minimization_budget,
                    seed=seed,
                    initial_factors=result.factors,
                )
            except ValueError as error:
                output.mkdir(parents=True, exist_ok=False)
                visible = validation_reports[-1].to_dict() if validation_reports else None
                _write_json(
                    output / "extraction-failure.json",
                    {
                        "compiler": compile_evidence(result),
                        "extraction": extraction_payload,
                        "reason": str(error),
                        "status": "VALIDATION_FAILED",
                        "validation": visible,
                    },
                )
                return _CommandResult(
                    {
                        "evidence": extraction_payload,
                        "reason": str(error),
                        "status": "VALIDATION_FAILED",
                    },
                    EXIT_FAILED,
                )
            if not minimization.factors:
                output.mkdir(parents=True, exist_ok=False)
                _write_json(
                    output / "extraction-failure.json",
                    {
                        "compiler": compile_evidence(result),
                        "extraction": extraction_payload,
                        "reason": "the unpatched base already passed the selected visible contract",
                        "status": VerificationOutcome.NOT_APPLICABLE.value,
                    },
                )
                return _CommandResult(
                    {
                        "reason": (
                            "the unpatched base already passed the selected visible contract; "
                            "no behavior patch is applicable"
                        ),
                        "status": VerificationOutcome.NOT_APPLICABLE.value,
                    },
                    EXIT_INCONCLUSIVE,
                )
            minimized_factors = {
                name: (left.detach().clone(), right.detach().clone())
                for name, (left, right) in sorted(minimization.factors.items())
            }
            result = CompilationResult(
                status=result.status,
                deltas={name: left @ right for name, (left, right) in minimized_factors.items()},
                factors=minimized_factors,
                active_modules=tuple(sorted(minimized_factors)),
                ranks={
                    name: int(left.shape[1]) for name, (left, _right) in minimized_factors.items()
                },
                evidence=list(result.evidence),
                best_step=result.best_step,
                best_target_loss=result.best_target_loss,
                violated_constraints=dict(result.violated_constraints),
                warnings=list(result.warnings),
                metadata={
                    **result.metadata,
                    "post_compile_minimization": True,
                    "validation_candidate_executions": len(validation_reports),
                },
            ).detached_cpu()
            program, tensors = compilation_delta_program(result, base_loaded.manifest.state_schema)

            # This is the first point at which sealed-holdout reference outcomes
            # are accessed. Exactly one selected candidate is evaluated below.
            holdout_target_outputs = generated(target_loaded.model, roles.holdout_targets)
            holdout_guard_outputs = generated(base_loaded.model, roles.holdout_guards)
            holdout_target_rows = [
                {
                    "expected": expected,
                    "id": f"holdout-target-{index:06d}",
                    "prompt": prompt,
                    "reference_logits": reference_logits(target_loaded.model, prompt),
                }
                for index, (prompt, expected) in enumerate(
                    zip(roles.holdout_targets, holdout_target_outputs, strict=True)
                )
            ]
            holdout_guard_rows = [
                {
                    "expected": expected,
                    "id": f"holdout-guard-{index:06d}",
                    "prompt": prompt,
                }
                for index, (prompt, expected) in enumerate(
                    zip(roles.holdout_guards, holdout_guard_outputs, strict=True)
                )
            ]
            data.update(
                {
                    "data/holdout-guards.jsonl": jsonl(holdout_guard_rows),
                    "data/holdout-targets.jsonl": jsonl(holdout_target_rows),
                }
            )
            for relative in ("data/holdout-guards.jsonl", "data/holdout-targets.jsonl"):
                target_path = temporary_contract_root / relative
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.write_bytes(data[relative])
            final_candidate = apply_dense_deltas(base_loaded.model, result.deltas)
            report = _verification_report(
                loaded=base_loaded,
                model=final_candidate,
                base_model=base_loaded.model,
                contract=contract,
                contract_path=contract_path,
                candidate_id=f"extraction:{cluster}:final",
                include_holdout=True,
            )
            validation, holdout = _report_sections(report)
            extraction_payload = {
                **extraction_payload,
                "holdout_candidate_executions": 1,
                "holdout_outcome": report.holdout_outcome.value,
                "validation_candidate_executions": len(validation_reports) + 1,
                "validation_passed": all(
                    item.outcome is VerificationOutcome.PASS
                    for item in (*report.target_results, *report.guard_results)
                ),
            }
            minimization_payload = {
                "schema_version": 1,
                "claims": [claim.value for claim in minimization.claims],
                "verification_budget": minimization_budget,
                "verification_budget_used": minimization.verification_budget_used,
                "candidates": [
                    {
                        "operation": candidate.operation,
                        "active_modules": list(candidate.active_modules),
                        "ranks": dict(sorted(candidate.ranks.items())),
                        "passed": candidate.passed,
                    }
                    for candidate in minimization.candidates
                ],
            }
            contract_bytes = (canonical_dumps(contract.to_dict()) + "\n").encode()
            contract_artifacts = {
                "contracts/preservation.yaml": contract_bytes,
                "contracts/target.yaml": contract_bytes,
                **{f"contracts/{relative}": content for relative, content in data.items()},
            }
            probe_hashes = {
                relative: sha256_bytes(content) for relative, content in sorted(data.items())
            }
            bundle = create_patch_bundle(
                output,
                name=f"extraction-{cluster}",
                base_signature=base_loaded.manifest.signature.to_dict(),
                state_schema=base_loaded.manifest.state_schema,
                program=program,
                tensors=tensors,
                tool_version=__version__,
                contracts=contract_artifacts,
                supplemental_artifacts={
                    "evidence/compile.json": (
                        canonical_dumps(
                            {
                                "cegis": extraction.to_dict(),
                                "compiler": compile_evidence(result),
                                "schema_version": 1,
                            }
                        )
                        + "\n"
                    ).encode(),
                    "evidence/holdout.json": (
                        canonical_dumps(
                            {
                                **holdout,
                                "candidate_evaluations": 1,
                                "unseal_policy": "final_candidate_only",
                            }
                        )
                        + "\n"
                    ).encode(),
                    "evidence/minimization.json": (
                        canonical_dumps(minimization_payload) + "\n"
                    ).encode(),
                    "evidence/validation.json": (
                        canonical_dumps({**validation, "extraction": extraction_payload}) + "\n"
                    ).encode(),
                    "probes/hashes.json": (canonical_dumps(probe_hashes) + "\n").encode(),
                    "probes/manifest.json": (
                        canonical_dumps(
                            {
                                "contract_hash": contract.contract_id,
                                "roles": roles.to_dict(),
                                "source_diff_hash": sha256_file(diff_bundle / "manifest.json"),
                                "source_hashes": probe_hashes,
                                "schema_version": 1,
                            }
                        )
                        + "\n"
                    ).encode(),
                    "report.md": (
                        "# Selective extraction report\n\n"
                        f"Selected empirical cluster: {cluster}\n\n"
                        f"Executed validation and sealed holdout: {report.outcome.value}\n\n"
                        f"CEGIS stop reason: {extraction.result.stop_reason.value}\n\n"
                        + (
                            "This is a verified candidate bundle.\n\n"
                            if report.outcome is VerificationOutcome.PASS
                            else (
                                "This bundle records a failed candidate and is not verified "
                                "for deployment.\n\n"
                            )
                        )
                        + "Claims are scoped to the recorded prompts, local mutation search, "
                        "generation policy, and budgets.\n"
                    ).encode(),
                },
                provides=(contract.contract_id,),
                preserves=(contract.contract_id,),
                source_diff_bundle=sha256_file(diff_bundle / "manifest.json"),
                compiler_configuration=cast(
                    Mapping[str, object],
                    _jsonable(
                        {
                            "cegis_rounds": cegis_rounds,
                            "holdout_probes": holdout_probes,
                            "minimization_budget": minimization_budget,
                            "optimizer": optimizer,
                            "search_budget": search_budget,
                            "validation_probes": validation_probes,
                        }
                    ),
                ),
            )
            certificate = build_certificate(
                report,
                contract,
                patch_id=bundle.manifest.patch_id,
                checkpoint_hashes=base_loaded.manifest.checkpoint_tensor_hashes,
                artifact_hashes=dict(bundle.manifest.artifact_hashes),
                patch_structure={
                    "active_modules": list(result.active_modules),
                    "module_ranks": dict(sorted(result.ranks.items())),
                    "patch_bytes": program.estimate_bytes(tensors),
                },
                counterexample_search=extraction.to_dict(),
                minimization_result=minimization_payload,
                objectives_optimized=True,
                minimized_within_budget=any(
                    claim.value != "UNMINIMIZED" for claim in minimization.claims
                ),
            )
            codegen_root = Path(tempfile.mkdtemp(prefix="modelpact-codegen-"))
            try:
                apply_path = emit_apply_script(
                    output, codegen_root / "apply_patch.py", will_live_in_bundle=True
                )
                verify_path = emit_verify_script(
                    output, codegen_root / "verify_patch.py", will_live_in_bundle=True
                )
                bundle = attach_bundle_artifacts(
                    output,
                    {
                        "apply_patch.py": apply_path.read_bytes(),
                        "certificate.json": (certificate.canonical_json() + "\n").encode(),
                        "verify_patch.py": verify_path.read_bytes(),
                    },
                    state_schema=base_loaded.manifest.state_schema,
                    require_complete=True,
                )
            finally:
                shutil.rmtree(codegen_root, ignore_errors=True)
        finally:
            shutil.rmtree(temporary_contract_root, ignore_errors=True)
        effective_outcome = report.outcome
        validation_outcome = VerificationOutcome.PASS
        if any(
            item.outcome is not VerificationOutcome.PASS
            for item in (*report.target_results, *report.guard_results)
        ):
            validation_outcome = VerificationOutcome.FAIL
        if report.holdout_outcome is VerificationOutcome.FAIL:
            status_value = "HOLDOUT_FAILED"
        elif validation_outcome is VerificationOutcome.FAIL:
            status_value = "VALIDATION_FAILED"
        else:
            status_value = effective_outcome.value
        return _CommandResult(
            {
                "deployable": effective_outcome is VerificationOutcome.PASS,
                "extraction": extraction_payload,
                "output": output.as_posix(),
                "patch_id": bundle.manifest.patch_id,
                "status": status_value,
                "verification": report.to_dict(),
            },
            _outcome_exit(effective_outcome),
        )

    _invoke(operation, compact=json_output)


def _report_margin(report: Any) -> float:
    evaluations = (*report.target_results, *report.guard_results)
    margins = [item.margin for item in evaluations if item.margin is not None]
    if report.outcome is VerificationOutcome.PASS:
        return min(margins, default=1.0)
    negative = [value for value in margins if value < 0.0]
    return min(negative, default=-1.0)


@dataclass(frozen=True, slots=True)
class _TeacherLogitsExample:
    batch: Any
    logits: Tensor


def _model_with_dense_delta(
    model: nn.Module,
    delta: Mapping[str, Tensor],
    state_schema: ModelStateSchema,
) -> nn.Module:
    """Materialize a validated dense delta on a private model copy.

    Alias groups are resolved to one physical parameter, so a tied embedding and
    output head are changed exactly once.  This helper is used only for trusted
    in-process optimization; patch files still flow through the safe Delta IR.
    """

    result = copy.deepcopy(model)
    aliases = _alias_map(state_schema)
    canonical: dict[str, Tensor] = {}
    for name, value in sorted(delta.items()):
        target = aliases.get(name, name)
        specification = state_schema.tensor(target)
        if not specification.patchable:
            raise ValueError(f"semantic compilation targeted non-patchable state: {target}")
        prior = canonical.get(target)
        if prior is not None:
            if (
                prior.shape != value.shape
                or prior.dtype != value.dtype
                or not torch.equal(prior, value)
            ):
                raise ValueError(f"resolved delta disagrees across aliases: {target}")
            continue
        canonical[target] = value

    parameters = dict(result.named_parameters(remove_duplicate=False))
    with torch.no_grad():
        for name, value in sorted(canonical.items()):
            parameter = parameters.get(name)
            if parameter is None:
                raise ValueError(f"delta target is not a model parameter: {name}")
            if tuple(parameter.shape) != tuple(value.shape):
                raise ValueError(f"delta shape mismatch for {name}")
            if str(parameter.dtype).removeprefix("torch.") != state_schema.tensor(name).dtype:
                raise ValueError(f"loaded parameter dtype disagrees with schema: {name}")
            parameter.add_(value.to(device=parameter.device, dtype=parameter.dtype))
    return result


def _compilation_dense_delta(result: Any, state_schema: ModelStateSchema) -> dict[str, Tensor]:
    from modelpact.compiler.package import compilation_delta_program

    program, tensors = compilation_delta_program(result, state_schema)
    return {
        target: program.materialize(target, tensors).detach().cpu()
        for target in sorted(program.targets)
    }


def _sum_dense_deltas(
    left: Mapping[str, Tensor],
    right: Mapping[str, Tensor],
    *,
    base_signature: str,
    state_schema: ModelStateSchema,
) -> dict[str, Tensor]:
    from modelpact.compose.closure import PatchOperand, additive_compose

    operands = [
        PatchOperand(
            patch_id="semantic-initial",
            base_signature=base_signature,
            module_schema_hash=state_schema.schema_hash,
            delta=left,
            contract_ids=("semantic-initial",),
        ),
        PatchOperand(
            patch_id="semantic-repair",
            base_signature=base_signature,
            module_schema_hash=state_schema.schema_hash,
            delta=right,
            contract_ids=("semantic-repair",),
        ),
    ]
    return additive_compose(operands, aliases=_alias_map(state_schema))


def _teacher_kl_objective(
    *,
    adapter: Any,
    teacher_model: nn.Module,
    objective_id: str,
    batches: Sequence[Any],
) -> Any:
    from modelpact.compiler.constraints import DifferentiableObjective

    examples: list[_TeacherLogitsExample] = []
    with torch.no_grad():
        for prepared in batches:
            batch = prepared.batch
            logits = adapter.forward_logits(teacher_model, batch).detach().cpu()
            examples.append(_TeacherLogitsExample(batch, logits))
    if not examples:
        raise ValueError("a parent behavior teacher requires at least one target-domain probe")

    def loss(candidate: nn.Module, example: _TeacherLogitsExample) -> Tensor:
        student = adapter.forward_logits(candidate, example.batch).to(torch.float64)
        teacher = example.logits.to(device=student.device, dtype=student.dtype)
        teacher_log = torch.log_softmax(teacher, dim=-1)
        student_log = torch.log_softmax(student, dim=-1)
        per_token = (teacher_log.exp() * (teacher_log - student_log)).sum(dim=-1)
        mask = example.batch.attention_mask.to(device=per_token.device, dtype=per_token.dtype)
        return cast(Tensor, (per_token * mask).sum() / mask.sum().clamp_min(1))

    return DifferentiableObjective(objective_id, tuple(examples), loss)


def _joint_tiny_compiler(
    context: _CompositionContext,
    *,
    maximum_rank: int,
    maximum_modules: int,
    seed: int,
    semantic_target_examples: tuple[Any, ...] = (),
    semantic_guard_examples: tuple[Any, ...] = (),
) -> Any:
    from modelpact.compiler.constraints import DifferentiableConstraint, DifferentiableObjective
    from modelpact.compiler.contracts import prepare_contract
    from modelpact.compiler.optimize import OptimizerConfig, compile_low_rank_patch
    from modelpact.compiler.semantic_cegis import differentiable_refinement_problem
    from modelpact.compose.merge import JointCompilationResult, SemanticMergeRequest

    def compile_joint(request: SemanticMergeRequest) -> JointCompilationResult:
        prepared = {
            identifier: prepare_contract(
                context.loaded.adapter,
                context.loaded.model,
                contract,
                path,
            )
            for identifier, (contract, path) in sorted(context.contracts.items())
        }
        declared_objectives: list[DifferentiableObjective] = []
        guards: list[DifferentiableConstraint] = []
        for contract_id, item in sorted(prepared.items()):
            declared_objectives.extend(
                DifferentiableObjective(
                    f"contract:{contract_id}:{objective.objective_id}",
                    objective.batches,
                    objective.loss,
                    objective.weight,
                )
                for objective in item.objectives
            )
            guards.extend(
                DifferentiableConstraint(
                    f"contract:{contract_id}:{guard.constraint_id}",
                    guard.batches,
                    guard.measure,
                    guard.maximum,
                )
                for guard in item.guards
            )

        parent_teacher_objectives: list[DifferentiableObjective] = []
        by_patch = {operand.patch_id: operand for operand in context.operands}
        for patch_id in request.parent_patch_ids:
            operand = by_patch[patch_id]
            teacher = _model_with_dense_delta(
                context.loaded.model,
                operand.delta,
                context.loaded.manifest.state_schema,
            )
            for contract_id in operand.contract_ids:
                batches = tuple(
                    example
                    for objective in prepared[contract_id].objectives
                    for example in objective.batches
                )
                if batches:
                    parent_teacher_objectives.append(
                        _teacher_kl_objective(
                            adapter=context.loaded.adapter,
                            teacher_model=teacher,
                            objective_id=f"parent:{patch_id}:{contract_id}",
                            batches=batches,
                        )
                    )

        refinement_objectives, refinement_guards = differentiable_refinement_problem(
            context.loaded.adapter,
            context.loaded.model,
            semantic_target_examples,
            semantic_guard_examples,
        )
        objectives = (
            *declared_objectives,
            *parent_teacher_objectives,
            *refinement_objectives,
        )
        guards.extend(refinement_guards)
        if not objectives:
            return JointCompilationResult(
                candidate_delta=None,
                optimization_succeeded=False,
                budget_exhausted=False,
                steps_executed=0,
                restarts_executed=0,
                failure_reason=(
                    "the union contracts expose no differentiable objective or "
                    "target-domain parent-teacher probes"
                ),
                diagnostics={"missing_data_contract": "compile.objectives[].source"},
            )

        initialized = _model_with_dense_delta(
            context.loaded.model,
            request.initial_delta,
            context.loaded.manifest.state_schema,
        )
        result = compile_low_rank_patch(
            initialized,
            tuple(objectives),
            tuple(guards),
            config=OptimizerConfig(
                maximum_rank=maximum_rank,
                maximum_modules=maximum_modules,
                steps=request.budget.maximum_steps,
                seed=seed,
                patience=max(1, min(50, request.budget.maximum_steps)),
            ),
        )
        residual = (
            _compilation_dense_delta(result, context.loaded.manifest.state_schema)
            if result.feasible
            else {}
        )
        candidate = None
        if result.feasible:
            candidate = (
                _sum_dense_deltas(
                    request.initial_delta,
                    residual,
                    base_signature=request.base_signature,
                    state_schema=context.loaded.manifest.state_schema,
                )
                if residual
                else {
                    name: value.detach().cpu().clone()
                    for name, value in sorted(request.initial_delta.items())
                }
            )
        executed_steps = len(result.evidence)
        return JointCompilationResult(
            candidate_delta=candidate,
            optimization_succeeded=result.feasible,
            budget_exhausted=(
                not result.feasible and executed_steps >= request.budget.maximum_steps
            ),
            steps_executed=executed_steps,
            restarts_executed=1,
            violated_contracts=tuple(sorted(result.violated_constraints)),
            diagnostics={
                "active_modules": list(result.active_modules),
                "declared_objectives": len(declared_objectives),
                "initialization": "summed_parent_delta_plus_low_rank_residual",
                "parent_teacher_objectives": len(parent_teacher_objectives),
                "semantic_cegis_guard_constraints": len(refinement_guards),
                "semantic_cegis_target_objectives": len(refinement_objectives),
                "ranks": dict(sorted(result.ranks.items())),
                "real_optimization": executed_steps > 0,
            },
            failure_reason=None if result.feasible else "; ".join(result.warnings),
        )

    return compile_joint


def _joint_as_compilation(result: Any) -> Any:
    """Adapt one joint-compiler execution to the generic CEGIS candidate type."""

    from modelpact.compiler.result import CompilationResult, CompilationStatus

    feasible = bool(result.optimization_succeeded and result.candidate_delta is not None)
    deltas = (
        {
            name: value.detach().cpu().clone()
            for name, value in sorted(result.candidate_delta.items())
        }
        if feasible
        else {}
    )
    return CompilationResult(
        status=(
            CompilationStatus.FEASIBLE if feasible else CompilationStatus.INFEASIBLE_WITHIN_BUDGET
        ),
        deltas=deltas,
        factors={},
        active_modules=tuple(sorted(deltas)),
        ranks={name: min(value.shape) if value.ndim == 2 else 0 for name, value in deltas.items()},
        violated_constraints=dict.fromkeys(result.violated_contracts, -1.0),
        warnings=[result.failure_reason] if result.failure_reason else [],
        metadata={
            "budget_exhausted": result.budget_exhausted,
            "optimization_steps": result.steps_executed,
            "restarts_executed": result.restarts_executed,
        },
    )


def _refine_tiny_semantic_merge(
    result: Any,
    context: _CompositionContext,
    *,
    maximum_steps: int,
    maximum_rank: int,
    maximum_modules: int,
    cegis_rounds: int,
    cegis_search_budget: int,
    minimization_budget: int,
    seed: int,
) -> tuple[Any, dict[str, object], dict[str, object], Mapping[str, tuple[Tensor, Tensor]] | None]:
    """Search and minimize a visible-contract-verified tiny semantic merge."""

    from modelpact.compiler.minimize import minimize_patch
    from modelpact.compiler.semantic_cegis import (
        candidate_satisfies_semantic_working_set,
        run_semantic_cegis,
    )
    from modelpact.compose.merge import (
        JointCompilationResult,
        MergeBudget,
        MergeDisposition,
        SemanticMergeRequest,
    )
    from modelpact.status import CompositionClaim

    if not result.verified or not result.compiler_invoked:
        raise ValueError("semantic refinement requires a verified recompiled candidate")
    request = SemanticMergeRequest(
        parent_patch_ids=tuple(sorted(result.parent_patch_ids)),
        base_signature=context.loaded.manifest.signature.signature_hash,
        module_schema_hash=context.loaded.manifest.state_schema.schema_hash,
        contract_ids=tuple(sorted(result.contract_ids)),
        initial_delta=result.delta,
        parent_deltas={
            operand.patch_id: operand.delta
            for operand in sorted(context.operands, key=lambda item: item.patch_id)
        },
        budget=MergeBudget(maximum_steps=maximum_steps),
    )

    joint_candidates: list[JointCompilationResult] = []

    def compile_candidate(target_examples: tuple[Any, ...], guard_examples: tuple[Any, ...]) -> Any:
        compiler = _joint_tiny_compiler(
            context,
            maximum_rank=maximum_rank,
            maximum_modules=maximum_modules,
            seed=seed,
            semantic_target_examples=target_examples,
            semantic_guard_examples=guard_examples,
        )
        joint = compiler(request)
        joint_candidates.append(joint)
        return _joint_as_compilation(joint)

    def candidate_model(delta: Mapping[str, Tensor]) -> nn.Module:
        return _model_with_dense_delta(
            context.loaded.model,
            delta,
            context.loaded.manifest.state_schema,
        )

    cegis = run_semantic_cegis(
        context.loaded.adapter,
        context.loaded.model,
        context.contracts,
        compile_candidate=compile_candidate,
        candidate_model=candidate_model,
        maximum_rounds=cegis_rounds,
        search_budget_per_domain_per_round=cegis_search_budget,
        seed=seed,
    )
    cegis_evidence = cegis.to_dict()
    if not cegis.candidate.feasible:
        latest = joint_candidates[-1]
        failed = dataclasses.replace(
            result,
            disposition=MergeDisposition.EMPIRICALLY_INFEASIBLE_WITHIN_BUDGET,
            claim=CompositionClaim.EMPIRICALLY_INFEASIBLE_WITHIN_BUDGET,
            delta=latest.candidate_delta or {},
            compilation=latest,
            warnings=(
                *result.warnings,
                "bounded semantic CEGIS recompilation was infeasible",
            ),
        )
        return failed, cegis_evidence, {}, None

    visible = context.execute(
        cegis.candidate.deltas,
        result.contract_ids,
        include_holdout=False,
        execution_phase="semantic-cegis",
        execution_label="selected-visible-candidate",
    )
    visible_passed = (
        visible.outcome is VerificationOutcome.PASS
        and set(result.contract_ids) <= set(visible.by_contract())
        and all(item.passed for item in visible.margins)
        and candidate_satisfies_semantic_working_set(
            cegis,
            context.loaded.adapter,
            context.loaded.model,
            context.contracts,
            candidate_model,
            cegis.candidate.deltas,
        )
    )
    if not visible_passed:
        failed = dataclasses.replace(
            result,
            disposition=MergeDisposition.RECOMPILED_CANDIDATE_FAILED_VERIFICATION,
            claim=CompositionClaim.SEMANTIC_CONFLICT,
            delta=cegis.candidate.deltas,
            verification=visible,
            compilation=joint_candidates[-1],
            warnings=(
                *result.warnings,
                "semantic CEGIS candidate failed its accumulated visible working set",
            ),
        )
        return failed, cegis_evidence, {}, None

    def minimization_verifier(candidate: dict[str, Tensor]) -> bool:
        if not candidate:
            return False
        if not candidate_satisfies_semantic_working_set(
            cegis,
            context.loaded.adapter,
            context.loaded.model,
            context.contracts,
            candidate_model,
            candidate,
        ):
            return False
        verification = context.execute(
            candidate,
            result.contract_ids,
            include_holdout=False,
            execution_phase="semantic-minimization",
            execution_label=hash_canonical(
                {
                    name: {
                        "dtype": str(value.dtype),
                        "shape": list(value.shape),
                        "sum": float(value.to(torch.float64).sum().item()),
                    }
                    for name, value in sorted(candidate.items())
                }
            ),
        )
        return bool(
            verification.outcome is VerificationOutcome.PASS
            and set(result.contract_ids) <= set(verification.by_contract())
            and all(item.passed for item in verification.margins)
        )

    minimized = minimize_patch(
        dict(cegis.candidate.deltas),
        minimization_verifier,
        verification_budget=minimization_budget,
        seed=seed,
    )
    final_visible = context.execute(
        minimized.deltas,
        result.contract_ids,
        include_holdout=False,
        execution_phase="semantic-minimization",
        execution_label="selected-minimized-candidate",
    )
    claims = [item.value for item in minimized.claims]
    if "UNMINIMIZED" in claims:
        raise RuntimeError("semantic merge minimization produced no bounded minimality claim")
    minimization_evidence: dict[str, object] = {
        "candidates": cast(list[object], _jsonable(minimized.candidates)),
        "claims": claims,
        "schema_version": 1,
        "verification_budget": minimization_budget,
        "verification_budget_used": minimized.verification_budget_used,
    }
    latest = joint_candidates[-1]
    updated_compilation = dataclasses.replace(
        latest,
        candidate_delta=minimized.deltas,
        diagnostics={
            **dict(latest.diagnostics),
            "cegis": cegis_evidence,
            "minimization": minimization_evidence,
        },
    )
    factorized: Mapping[str, tuple[Tensor, Tensor]] | None = None
    if minimized.factors and set(minimized.factors) == set(minimized.deltas):
        factorized = minimized.factors
    refined = dataclasses.replace(
        result,
        disposition=MergeDisposition.SEMANTIC_MERGE_VERIFIED,
        claim=CompositionClaim.COMPOSITION_CLOSED,
        delta=minimized.deltas,
        verification=final_visible,
        compilation=updated_compilation,
    )
    return refined, cegis_evidence, minimization_evidence, factorized


@dataclass(slots=True)
class _CompositionContext:
    loaded: _LoadedModel
    bundles: tuple[PatchBundle, ...]
    operands: tuple[Any, ...]
    contracts: dict[str, tuple[Any, Path]]
    reports: dict[str, Any]
    execution_reports: dict[str, dict[str, Any]] = dataclasses.field(default_factory=dict)
    holdout_execution_counts: dict[str, int] = dataclasses.field(default_factory=dict)
    execution_sequence: int = 0
    last_execution_id: str | None = None
    last_execution_result_hash: str | None = None

    def execute(
        self,
        delta: Mapping[str, Tensor],
        contract_ids: tuple[str, ...],
        *,
        include_holdout: bool = False,
        execution_phase: str = "visible",
        execution_label: str | None = None,
    ) -> Any:
        from modelpact.compose.closure import ContractMargin, MarginKind, VerificationReport
        from modelpact.patch.mount import mount_patch

        normalized_contract_ids = tuple(sorted(set(contract_ids)))
        model = copy.deepcopy(self.loaded.model)
        reports: dict[str, Any] = {}

        def verify_candidate(candidate_id: str) -> None:
            for contract_id in normalized_contract_ids:
                contract, path = self.contracts[contract_id]
                if include_holdout and contract.holdout.configured:
                    count = self.holdout_execution_counts.get(contract_id, 0)
                    if count:
                        raise RuntimeError(
                            f"sealed holdout already executed for contract {contract_id}"
                        )
                    # Count access before execution: an adapter or scorer failure
                    # does not make the sealed data reusable for another candidate.
                    self.holdout_execution_counts[contract_id] = count + 1
                reports[contract_id] = _verification_report(
                    loaded=self.loaded,
                    model=model,
                    base_model=self.loaded.model,
                    contract=contract,
                    contract_path=path,
                    candidate_id=candidate_id,
                    include_holdout=include_holdout,
                )

        if delta:
            from modelpact.checkpoints.safetensors import tensor_content_hash

            program, tensors = _dense_program(delta, self.loaded.manifest.state_schema)
            candidate_id = "composition:" + hash_canonical(
                {
                    "program": program.to_dict(),
                    "tensors": {
                        name: tensor_content_hash(value) for name, value in sorted(tensors.items())
                    },
                }
            ).removeprefix("sha256:")
            self.execution_sequence += 1
            label = execution_label or f"execution-{self.execution_sequence:08d}"
            contract_set_hash = hash_canonical(normalized_contract_ids).removeprefix("sha256:")
            execution_id = f"{execution_phase}:{label}:contracts-{contract_set_hash}:{candidate_id}"
            with mount_patch(
                model,
                program,
                tensors,
                state_schema=self.loaded.manifest.state_schema,
            ):
                verify_candidate(candidate_id)
        else:
            candidate_id = "composition:base"
            self.execution_sequence += 1
            label = execution_label or f"execution-{self.execution_sequence:08d}"
            contract_set_hash = hash_canonical(normalized_contract_ids).removeprefix("sha256:")
            execution_id = f"{execution_phase}:{label}:contracts-{contract_set_hash}:{candidate_id}"
            verify_candidate(candidate_id)
        self.reports = reports
        self.execution_reports[execution_id] = reports
        self.last_execution_id = execution_id
        self.last_execution_result_hash = hash_canonical(
            {
                "execution_id": execution_id,
                "verification_result_hashes": {
                    contract_id: report.result_hash
                    for contract_id, report in sorted(reports.items())
                },
            }
        )
        outcomes = [report.outcome for report in reports.values()]
        if any(item is VerificationOutcome.FAIL for item in outcomes):
            outcome = VerificationOutcome.FAIL
        elif any(item is VerificationOutcome.UNSUPPORTED for item in outcomes):
            outcome = VerificationOutcome.UNSUPPORTED
        elif any(item is VerificationOutcome.INCONCLUSIVE for item in outcomes):
            outcome = VerificationOutcome.INCONCLUSIVE
        else:
            outcome = VerificationOutcome.PASS
        margins = tuple(
            ContractMargin(
                contract_id,
                MarginKind.TARGET,
                _report_margin(reports[contract_id]),
                details={
                    "candidate_id": candidate_id,
                    "execution_id": execution_id,
                    "execution_phase": execution_phase,
                    "verification_result_hash": reports[contract_id].result_hash,
                },
            )
            for contract_id in sorted(reports)
        )
        return VerificationReport(outcome, margins)


def _composition_context(
    *,
    adapter_spec: str,
    base_checkpoint: Path,
    patch_paths: Sequence[Path],
    device: str,
    dtype: str,
) -> _CompositionContext:
    from modelpact.compose.closure import PatchOperand
    from modelpact.patch.validate import validate_base_signature

    if not patch_paths:
        raise ValueError("composition requires at least one patch")
    loaded = _load_model(adapter_spec, base_checkpoint, device=device, dtype=dtype)
    bundles = tuple(
        load_patch_bundle(path, state_schema=loaded.manifest.state_schema) for path in patch_paths
    )
    contracts: dict[str, tuple[Any, Path]] = {}
    operands = []
    for bundle in bundles:
        validate_base_signature(bundle.manifest.base_signature, loaded.manifest.signature)
        bundle_contracts = _load_bundle_contracts(bundle)
        for identifier, value in bundle_contracts.items():
            if identifier in contracts and contracts[identifier][0].to_dict() != value[0].to_dict():
                raise ValueError("contract identity collision across patch bundles")
            contracts[identifier] = value
        operands.append(
            PatchOperand(
                patch_id=bundle.manifest.patch_id,
                base_signature=loaded.manifest.signature.signature_hash,
                module_schema_hash=loaded.manifest.state_schema.schema_hash,
                delta=_bundle_dense_delta(bundle),
                contract_ids=tuple(sorted(bundle_contracts)),
            )
        )
    return _CompositionContext(loaded, bundles, tuple(operands), contracts, {})


def _static_checker(context: _CompositionContext) -> Any:
    from modelpact.contracts.static import ProbeRecord, check_static_contracts
    from modelpact.verify.provider import load_probe_records

    def check(contract_ids: tuple[str, ...]) -> tuple[Any, ...]:
        selected = [context.contracts[item] for item in contract_ids]
        records_by_contract: dict[str, dict[str, tuple[ProbeRecord, ...]]] = {}
        for contract, path in selected:
            source_records: dict[str, tuple[ProbeRecord, ...]] = {}
            for source in sorted({item.source for item in (*contract.targets, *contract.guards)}):
                candidate = path.parent / source
                if not candidate.is_file():
                    continue
                source_records[source] = cast(
                    tuple[ProbeRecord, ...],
                    load_probe_records(path.parent, source),
                )
            records_by_contract[contract.id] = source_records
        result = check_static_contracts(
            [contract for contract, _path in selected],
            records_by_contract=records_by_contract,
        )
        return result.witnesses

    return check


def _composition_payload(result: Any, reports: Mapping[str, Any]) -> dict[str, object]:
    verification = None
    if result.verification is not None:
        verification = {
            "margins": _jsonable(result.verification.margins),
            "outcome": result.verification.outcome.value,
            "reports": {
                identifier: report.to_dict() for identifier, report in sorted(reports.items())
            },
        }
    return {
        "base_verification": _jsonable(result.base_verification),
        "claim": result.claim.value,
        "contract_ids": list(result.contract_ids),
        "contradictions": _jsonable(result.contradictions),
        "degraded_contracts": list(result.degraded_contracts),
        "degradation_tolerance": result.degradation_tolerance,
        "evidence_gaps": list(result.evidence_gaps),
        "interaction_margins": dict(sorted(result.interaction_margins.items())),
        "patch_ids": list(result.patch_ids),
        "singleton_verifications": _jsonable(result.singleton_verifications),
        "structural_errors": list(result.structural_errors),
        "unverified_contracts": list(result.unverified_contracts),
        "verification": verification,
    }


def _execute_final_composition_candidate(
    result: Any,
    context: _CompositionContext,
    *,
    delta: Mapping[str, Tensor],
) -> Any:
    """Unseal holdout once, only after visible probes select a closed candidate."""

    from modelpact.status import CompositionClaim

    if not result.closed:
        return result
    final = context.execute(
        delta,
        result.contract_ids,
        include_holdout=True,
        execution_phase="final",
        execution_label="compose:selected-candidate",
    )
    reported = set(final.by_contract())
    missing = tuple(sorted(set(result.contract_ids) - reported))
    passed = (
        final.outcome is VerificationOutcome.PASS
        and not missing
        and all(margin.passed for margin in final.margins)
    )
    gaps = list(result.evidence_gaps)
    if not passed:
        gaps.append("selected final candidate failed validation or sealed holdout verification")
    return dataclasses.replace(
        result,
        claim=(
            CompositionClaim.COMPOSITION_CLOSED if passed else CompositionClaim.SEMANTIC_CONFLICT
        ),
        verification=final,
        unverified_contracts=missing,
        evidence_gaps=tuple(sorted(set(gaps))),
    )


def _composition_contract_artifacts(
    context: _CompositionContext,
    *,
    include_holdout_resources: bool,
) -> tuple[
    dict[str, bytes],
    tuple[tuple[Any, str], ...],
    dict[str, str],
]:
    """Copy exact contracts and contract-relative resources without collisions."""

    from modelpact.contracts.parser import resolve_contract_resource
    from modelpact.util.hashing import sha256_bytes

    artifacts: dict[str, bytes] = {}
    entries: list[tuple[Any, str]] = []
    resource_hashes: dict[str, str] = {}

    def add(relative: str, content: bytes) -> None:
        prior = artifacts.get(relative)
        if prior is not None and prior != content:
            raise ValueError(f"composition resource collision: {relative}")
        artifacts[relative] = content

    for index, (identifier, (contract, contract_path)) in enumerate(
        sorted(context.contracts.items())
    ):
        token = identifier.removeprefix("sha256:")
        root = Path("contracts") if index == 0 else Path("contracts/parents") / token
        contract_bytes = (canonical_dumps(contract.to_dict()) + "\n").encode()
        if index == 0:
            add("contracts/target.yaml", contract_bytes)
            add("contracts/preservation.yaml", contract_bytes)
            relative_contract = "contracts/target.yaml"
        else:
            relative_contract = (root / "contract.json").as_posix()
            add(relative_contract, contract_bytes)
        entries.append((contract, relative_contract))

        resources = {
            item.source for item in (*contract.objectives, *contract.targets, *contract.guards)
        }
        resources.update(_schema_files(contract))
        if include_holdout_resources:
            resources.update(
                source
                for source in (contract.holdout.targets, contract.holdout.guards)
                if source is not None
            )
        for source_name in sorted(resources):
            source = resolve_contract_resource(contract_path, source_name)
            destination = (root / Path(source_name)).as_posix()
            content = source.read_bytes()
            add(destination, content)
            resource_hashes[destination] = sha256_bytes(content)
    return (
        dict(sorted(artifacts.items())),
        tuple(entries),
        dict(sorted(resource_hashes.items())),
    )


def _composition_interaction_payload(
    result: Any,
    context: _CompositionContext,
) -> dict[str, object]:
    from modelpact.compose.interactions import (
        low_rank_factors_by_target,
        low_rank_subspace_diagnostics,
        module_overlap,
        sparse_index_overlap,
        sparse_indices_by_target,
    )

    bundles_by_id = {
        operand.patch_id: bundle
        for operand, bundle in zip(context.operands, context.bundles, strict=True)
    }
    pairs = []
    ordered_operands = tuple(sorted(context.operands, key=lambda item: item.patch_id))
    for left_index, left in enumerate(ordered_operands):
        for right in ordered_operands[left_index + 1 :]:
            overlap = module_overlap(tuple(left.delta), tuple(right.delta))
            left_bundle = bundles_by_id[left.patch_id]
            right_bundle = bundles_by_id[right.patch_id]
            left_sparse = sparse_indices_by_target(left_bundle.program, left_bundle.tensors)
            right_sparse = sparse_indices_by_target(right_bundle.program, right_bundle.tensors)
            sparse_diagnostics: dict[str, object]
            if left_sparse or right_sparse:
                sparse_diagnostics = {
                    "status": "AVAILABLE",
                    "values": _jsonable(sparse_index_overlap(left_sparse, right_sparse)),
                }
            else:
                sparse_diagnostics = {
                    "reason": "neither serialized patch program contains a sparse matrix delta",
                    "status": "NOT_AVAILABLE",
                }
            left_factors = low_rank_factors_by_target(left_bundle.program, left_bundle.tensors)
            right_factors = low_rank_factors_by_target(right_bundle.program, right_bundle.tensors)
            shared_low_rank = sorted(set(left_factors) & set(right_factors))
            if shared_low_rank:
                subspace_diagnostics: dict[str, object] = {
                    "status": "AVAILABLE",
                    "values": [
                        {
                            "module": module,
                            "principal_angles": _jsonable(
                                low_rank_subspace_diagnostics(
                                    left_left_factor=left_factors[module][0],
                                    left_right_factor=left_factors[module][1],
                                    right_left_factor=right_factors[module][0],
                                    right_right_factor=right_factors[module][1],
                                )
                            ),
                        }
                        for module in shared_low_rank
                    ],
                }
            else:
                subspace_diagnostics = {
                    "reason": (
                        "the serialized patch programs have no shared target with low-rank factors"
                    ),
                    "status": "NOT_AVAILABLE",
                }
            pairs.append(
                {
                    "activation_delta_similarity": {
                        "reason": (
                            "no aligned activation-delta samples were retained for these "
                            "composition executions"
                        ),
                        "status": "NOT_AVAILABLE",
                    },
                    "gradient_cosine_similarity": {
                        "reason": (
                            "no aligned contract-gradient samples were retained for these "
                            "patch bundles"
                        ),
                        "status": "NOT_AVAILABLE",
                    },
                    "left": left.patch_id,
                    "low_rank_subspace": subspace_diagnostics,
                    "module_overlap": _jsonable(overlap),
                    "output_interaction_residual": {
                        "reason": (
                            "verification retained signed margins and output hashes, not "
                            "aligned raw output tensors"
                        ),
                        "status": "NOT_AVAILABLE",
                    },
                    "right": right.patch_id,
                    "sparse_index_overlap": sparse_diagnostics,
                }
            )
    contract_interactions: dict[str, object] = {
        "status": "AVAILABLE" if result.interaction_margins else "NOT_AVAILABLE",
        "values": dict(sorted(result.interaction_margins.items())),
    }
    if not result.interaction_margins:
        contract_interactions["reason"] = (
            "base, both singleton, and pair-composition margins are required"
        )
    return {
        "contract_margin_interactions": contract_interactions,
        "executed_verification_reports": {
            candidate_id: {
                contract_id: report.to_dict() for contract_id, report in sorted(reports.items())
            }
            for candidate_id, reports in sorted(context.execution_reports.items())
        },
        "pairwise_diagnostics": pairs,
        "pairwise_parameter_overlap": [
            {
                "left": item["left"],
                "module_overlap": item["module_overlap"],
                "right": item["right"],
            }
            for item in pairs
        ],
        "warning": "parameter overlap is diagnostic only; executed contracts are authoritative",
    }


def _write_composition_artifacts(
    output: Path,
    *,
    result: Any,
    context: _CompositionContext,
) -> dict[str, object]:
    """Write inspectable failure evidence; successful candidates become bundles."""

    from modelpact.util.atomic import atomic_write_bytes

    output.mkdir(parents=True, exist_ok=False)
    tensor_path = output / "resolved-delta.safetensors"
    if result.resolved_delta:
        save_file(
            {
                key: value.detach().cpu().contiguous()
                for key, value in sorted(result.resolved_delta.items())
            },
            str(tensor_path),
        )
    contract_artifacts, _entries, _hashes = _composition_contract_artifacts(
        context,
        include_holdout_resources=bool(context.holdout_execution_counts),
    )
    for relative, content in contract_artifacts.items():
        atomic_write_bytes(output / relative, content, overwrite=False)
    _write_json(output / "verification.json", _composition_payload(result, context.reports))
    _write_json(
        output / "interactions.json",
        _composition_interaction_payload(result, context),
    )
    manifest = {
        "artifact_hashes": {
            path.relative_to(output).as_posix(): sha256_file(path)
            for path in sorted(item for item in output.rglob("*") if item.is_file())
        },
        "claim": result.claim.value,
        "contract_ids": list(result.contract_ids),
        "holdout_execution_counts": dict(sorted(context.holdout_execution_counts.items())),
        "holdout_resources_included": bool(context.holdout_execution_counts),
        "patch_ids": list(result.patch_ids),
        "schema_version": 1,
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def _create_composite_patch_bundle(
    output: Path,
    *,
    result: Any,
    context: _CompositionContext,
    operation: str,
    compiler_evidence: Mapping[str, object] | None = None,
    cegis_evidence: Mapping[str, object] | None = None,
    minimization_evidence: Mapping[str, object] | None = None,
    factorized: Mapping[str, tuple[Tensor, Tensor]] | None = None,
) -> tuple[PatchBundle, Any]:
    """Package one final, holdout-tested composition as Behavior Patch Bundle v1."""

    from modelpact.codegen import emit_apply_script, emit_verify_script
    from modelpact.patch.bundle import attach_bundle_artifacts, create_patch_bundle
    from modelpact.verify.certificate import build_certificate

    if not result.closed or result.verification is None:
        raise ValueError("only a verified final composition can be packaged")
    if set(result.contract_ids) - set(context.reports):
        raise ValueError("final composition reports omit union contracts")

    if factorized is not None:
        program, tensors = _factorized_program(
            factorized,
            context.loaded.manifest.state_schema,
        )
    else:
        program, tensors = _dense_program(
            result.resolved_delta,
            context.loaded.manifest.state_schema,
        )
    contract_artifacts, contract_entries, resource_hashes = _composition_contract_artifacts(
        context,
        include_holdout_resources=True,
    )
    interactions = _composition_interaction_payload(result, context)
    composition_evidence = _composition_payload(result, context.reports)
    validation_evidence: dict[str, object] = {}
    holdout_evidence: dict[str, object] = {}
    for identifier, report in sorted(context.reports.items()):
        validation, holdout = _report_sections(report)
        validation_evidence[identifier] = validation
        holdout_evidence[identifier] = holdout
    holdout_evidence["execution_counts"] = dict(sorted(context.holdout_execution_counts.items()))
    holdout_evidence["schema_version"] = 1
    compile_evidence = {
        "operation": operation,
        "cegis": (
            dict(cegis_evidence) if cegis_evidence is not None else {"outcome": "NOT_APPLICABLE"}
        ),
        "optimization": (
            dict(compiler_evidence)
            if compiler_evidence is not None
            else {"outcome": "NOT_APPLICABLE"}
        ),
        "parent_patch_ids": list(result.patch_ids),
        "schema_version": 1,
    }
    supplemental = {
        "evidence/compile.json": (canonical_dumps(compile_evidence) + "\n").encode(),
        "evidence/composition.json": (canonical_dumps(composition_evidence) + "\n").encode(),
        "evidence/holdout.json": (canonical_dumps(holdout_evidence) + "\n").encode(),
        "evidence/interactions.json": (canonical_dumps(interactions) + "\n").encode(),
        "evidence/minimization.json": (
            canonical_dumps(
                dict(minimization_evidence)
                if minimization_evidence is not None
                else {
                    "claim": "UNMINIMIZED",
                    "reason": "resolved composition was not minimized",
                    "schema_version": 1,
                }
            )
            + "\n"
        ).encode(),
        "evidence/validation.json": (
            canonical_dumps({"contracts": validation_evidence, "schema_version": 1}) + "\n"
        ).encode(),
        "probes/hashes.json": (canonical_dumps(resource_hashes) + "\n").encode(),
        "probes/manifest.json": (
            canonical_dumps(
                {
                    "roles": ["compile", "validation", "guard", "holdout"],
                    "schema_version": 1,
                    "source_hashes": resource_hashes,
                }
            )
            + "\n"
        ).encode(),
        "report.md": (
            "# ModelPact composite patch report\n\n"
            f"Operation: {operation}\n\n"
            f"Closure: {result.claim.value}\n\n"
            "The selected final candidate executed every declared visible contract and "
            "each configured sealed holdout once. Claims remain scoped to those probes.\n"
        ).encode(),
    }
    policy = {
        identifier: {
            "generation": contract.generation.to_dict(),
            "statistics": contract.statistics.to_dict(),
        }
        for identifier, (contract, _path) in sorted(context.contracts.items())
    }
    parents = tuple(sorted(result.patch_ids))
    provided_contracts = tuple(
        sorted(
            identifier
            for identifier, (contract, _path) in context.contracts.items()
            if contract.targets
        )
    )
    preserved_contracts = tuple(
        sorted(
            identifier
            for identifier, (contract, _path) in context.contracts.items()
            if contract.guards
        )
    )
    required_contracts = tuple(
        sorted(
            {identifier for bundle in context.bundles for identifier in bundle.manifest.requires}
        )
    )
    bundle = create_patch_bundle(
        output,
        name=f"{operation}-" + "-".join(item.removeprefix("sha256:")[:8] for item in parents),
        base_signature=context.loaded.manifest.signature.to_dict(),
        state_schema=context.loaded.manifest.state_schema,
        program=program,
        tensors=tensors,
        tool_version=__version__,
        contracts=contract_artifacts,
        supplemental_artifacts=supplemental,
        provides=provided_contracts,
        preserves=preserved_contracts,
        requires=required_contracts,
        verification_policy_hash=hash_canonical(policy),
        parent_patches=parents,
        merged_from=parents if operation == "semantic-merge" else (),
        compiler_configuration={
            "cegis_rounds": (
                cegis_evidence.get("maximum_rounds") if cegis_evidence is not None else 0
            ),
            "operation": operation,
            "representation": (
                "low_rank_additive_delta"
                if factorized is not None
                else "resolved_dense_additive_delta"
            ),
        },
    )

    ordered_reports = tuple(
        context.reports[contract.contract_id] for contract, _relative in contract_entries
    )
    certificate_entries = tuple(
        (contract, output / relative) for contract, relative in contract_entries
    )
    union_report, union_contract, contract_hashes, verification_policy = _certificate_union(
        ordered_reports,
        certificate_entries,
    )
    certificate = build_certificate(
        union_report,
        union_contract,
        patch_id=bundle.manifest.patch_id,
        checkpoint_hashes=context.loaded.manifest.checkpoint_tensor_hashes,
        artifact_hashes=dict(bundle.manifest.artifact_hashes),
        verification_policy=verification_policy,
        contract_hashes=contract_hashes,
        patch_structure={
            "active_targets": sorted(program.targets),
            "patch_bytes": program.estimate_bytes(tensors),
        },
        composition_result={
            "claim": result.claim.value,
            "parent_patch_ids": list(parents),
        },
        counterexample_search=cegis_evidence,
        interaction_diagnostics=interactions,
        minimization_result=minimization_evidence,
        minimized_within_budget=minimization_evidence is not None,
        additional_warnings=(
            (
                "Composite delta uses a deterministic dense resolution representation; "
                "no minimality claim is made."
                if minimization_evidence is None
                else (
                    "Minimality is bounded to the recorded executed module and rank "
                    "candidates; it is not a global-minimum claim."
                )
            ),
        ),
    )
    codegen_root = Path(tempfile.mkdtemp(prefix="modelpact-composition-codegen-"))
    try:
        apply_path = emit_apply_script(
            output, codegen_root / "apply_patch.py", will_live_in_bundle=True
        )
        verify_path = emit_verify_script(
            output, codegen_root / "verify_patch.py", will_live_in_bundle=True
        )
        bundle = attach_bundle_artifacts(
            output,
            {
                "apply_patch.py": apply_path.read_bytes(),
                "certificate.json": (certificate.canonical_json() + "\n").encode(),
                "verify_patch.py": verify_path.read_bytes(),
            },
            state_schema=context.loaded.manifest.state_schema,
            require_complete=True,
        )
    finally:
        shutil.rmtree(codegen_root, ignore_errors=True)
    return bundle, certificate


@app.command("compose")
def compose_command(
    patches: list[Path] = typer.Argument(..., exists=True),
    base_checkpoint: Path = typer.Option(..., "--base", exists=True),
    output: Path = typer.Option(..., "--output"),
    adapter_spec: str = typer.Option("tiny", "--adapter"),
    device: str = typer.Option("cpu", "--device"),
    dtype: str = typer.Option("float32", "--dtype"),
    degradation_tolerance: float | None = typer.Option(
        None,
        "--degradation-tolerance",
        min=0.0,
        max=1_000_000.0,
        help=(
            "Classify a still-passing contract as degraded when its singleton "
            "margin drops by more than this bound."
        ),
    ),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Add patches and execute the union contracts without repairing failures."""

    from modelpact.compose.closure import verify_contract_closure

    def operation() -> _CommandResult:
        context = _composition_context(
            adapter_spec=adapter_spec,
            base_checkpoint=base_checkpoint,
            patch_paths=patches,
            device=device,
            dtype=dtype,
        )
        result = verify_contract_closure(
            context.operands,
            executor=context.execute,
            aliases=_alias_map(context.loaded.manifest.state_schema),
            contradiction_checker=_static_checker(context),
            degradation_tolerance=degradation_tolerance,
            execute_baselines=True,
        )
        result = _execute_final_composition_candidate(
            result,
            context,
            delta=result.resolved_delta,
        )
        patch_id = None
        certificate_hash = None
        if result.closed:
            bundle, certificate = _create_composite_patch_bundle(
                output,
                result=result,
                context=context,
                operation="compose",
            )
            manifest: Mapping[str, object] = bundle.manifest.to_dict()
            patch_id = bundle.manifest.patch_id
            certificate_hash = certificate.certificate_hash
            artifact_kind = "BEHAVIOR_PATCH_BUNDLE_V1"
        else:
            manifest = _write_composition_artifacts(
                output,
                result=result,
                context=context,
            )
            artifact_kind = "COMPOSITION_FAILURE_EVIDENCE_V1"
        payload = {
            **_composition_payload(result, context.reports),
            "artifact_kind": artifact_kind,
            "certificate_hash": certificate_hash,
            "holdout_execution_counts": dict(sorted(context.holdout_execution_counts.items())),
            "manifest_hash": hash_canonical(manifest),
            "output": output.as_posix(),
            "patch_id": patch_id,
            "status": result.claim.value,
        }
        return _CommandResult(payload, 0 if result.closed else EXIT_FAILED)

    _invoke(operation, compact=json_output)


@app.command("merge")
def merge_command(
    patches: list[Path] = typer.Argument(..., exists=True),
    base_checkpoint: Path = typer.Option(..., "--base", exists=True),
    output: Path = typer.Option(..., "--output"),
    adapter_spec: str = typer.Option("tiny", "--adapter"),
    maximum_steps: int = typer.Option(200, "--maximum-steps", min=1, max=10_000_000),
    max_rank: int = typer.Option(16, "--max-rank", min=1, max=4096),
    max_modules: int = typer.Option(12, "--max-modules", min=1, max=100_000),
    cegis_rounds: int = typer.Option(2, "--cegis-rounds", min=1, max=10_000),
    cegis_search_budget: int = typer.Option(
        16,
        "--cegis-search-budget",
        min=1,
        max=100_000,
    ),
    minimization_budget: int = typer.Option(
        32,
        "--minimization-budget",
        min=1,
        max=100_000,
    ),
    seed: int = typer.Option(0, "--seed", min=0),
    force_recompile: bool = typer.Option(False, "--force-recompile"),
    device: str = typer.Option("cpu", "--device"),
    dtype: str = typer.Option("float32", "--dtype"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Return a closed sum or jointly recompile a failing tiny-model composition."""

    from modelpact.compose.merge import (
        JointCompilationResult,
        MergeBudget,
        MergeDisposition,
        semantic_merge,
    )
    from modelpact.status import CompositionClaim

    def operation() -> _CommandResult:
        context = _composition_context(
            adapter_spec=adapter_spec,
            base_checkpoint=base_checkpoint,
            patch_paths=patches,
            device=device,
            dtype=dtype,
        )

        from modelpact.compose.merge import SemanticMergeRequest

        def unavailable_compiler(request: SemanticMergeRequest) -> JointCompilationResult:
            _ = request
            return JointCompilationResult(
                candidate_delta=None,
                optimization_succeeded=False,
                budget_exhausted=False,
                steps_executed=0,
                restarts_executed=0,
                failure_reason=(
                    "joint CLI recompilation is currently supported only by the built-in "
                    "tiny adapter; custom and Hugging Face adapters require an explicit "
                    "trusted compiler integration"
                ),
            )

        compiler = (
            _joint_tiny_compiler(
                context,
                maximum_rank=max_rank,
                maximum_modules=max_modules,
                seed=seed,
            )
            if adapter_spec == "tiny"
            else unavailable_compiler
        )

        result = semantic_merge(
            context.operands,
            executor=context.execute,
            compiler=compiler,
            budget=MergeBudget(maximum_steps=maximum_steps),
            aliases=_alias_map(context.loaded.manifest.state_schema),
            contradiction_checker=_static_checker(context),
            force_recompile=force_recompile,
            execute_baselines=True,
        )
        cegis_evidence: Mapping[str, object] | None = None
        minimization_evidence: Mapping[str, object] | None = None
        factorized: Mapping[str, tuple[Tensor, Tensor]] | None = None
        if result.verified and result.compiler_invoked:
            from modelpact.compiler.semantic_cegis import SemanticCEGISUnsupportedError

            try:
                (
                    result,
                    raw_cegis_evidence,
                    raw_minimization_evidence,
                    factorized,
                ) = _refine_tiny_semantic_merge(
                    result,
                    context,
                    maximum_steps=maximum_steps,
                    maximum_rank=max_rank,
                    maximum_modules=max_modules,
                    cegis_rounds=cegis_rounds,
                    cegis_search_budget=cegis_search_budget,
                    minimization_budget=minimization_budget,
                    seed=seed,
                )
                cegis_evidence = raw_cegis_evidence
                minimization_evidence = raw_minimization_evidence or None
            except SemanticCEGISUnsupportedError as error:
                result = dataclasses.replace(
                    result,
                    disposition=MergeDisposition.COMPILER_FAILED,
                    claim=CompositionClaim.SEMANTIC_CONFLICT,
                    warnings=(
                        *result.warnings,
                        f"semantic CEGIS unsupported: {error}",
                    ),
                )
                cegis_evidence = {
                    "outcome": "UNSUPPORTED",
                    "reason": str(error),
                    "schema_version": 1,
                }
        selected_before_holdout = result.verified
        if selected_before_holdout:
            final_verification = context.execute(
                result.delta,
                result.contract_ids,
                include_holdout=True,
                execution_phase="final",
                execution_label="merge:selected-candidate",
            )
            reported = set(final_verification.by_contract())
            final_passed = (
                final_verification.outcome is VerificationOutcome.PASS
                and set(result.contract_ids) <= reported
                and all(margin.passed for margin in final_verification.margins)
            )
            result = dataclasses.replace(
                result,
                disposition=(
                    result.disposition
                    if final_passed
                    else MergeDisposition.FINAL_CANDIDATE_FAILED_HOLDOUT
                ),
                claim=(result.claim if final_passed else CompositionClaim.SEMANTIC_CONFLICT),
                verification=final_verification,
                warnings=(
                    result.warnings
                    if final_passed
                    else (
                        *result.warnings,
                        "selected merge candidate failed final validation or sealed holdout",
                    )
                ),
            )
        artifact_result = result.naive_composition
        if selected_before_holdout:
            artifact_result = dataclasses.replace(
                artifact_result,
                claim=result.claim,
                resolved_delta=result.delta,
                verification=result.verification,
                degraded_contracts=(),
                unverified_contracts=(),
            )
        patch_id = None
        certificate_hash = None
        if result.verified:
            raw_compilation = _jsonable(result.compilation)
            compiler_evidence = (
                cast(Mapping[str, object], raw_compilation)
                if isinstance(raw_compilation, Mapping)
                else None
            )
            bundle, certificate = _create_composite_patch_bundle(
                output,
                result=artifact_result,
                context=context,
                operation="semantic-merge",
                compiler_evidence=compiler_evidence,
                cegis_evidence=cegis_evidence,
                minimization_evidence=minimization_evidence,
                factorized=factorized,
            )
            manifest: Mapping[str, object] = bundle.manifest.to_dict()
            patch_id = bundle.manifest.patch_id
            certificate_hash = certificate.certificate_hash
            artifact_kind = "BEHAVIOR_PATCH_BUNDLE_V1"
        else:
            manifest = _write_composition_artifacts(
                output,
                result=artifact_result,
                context=context,
            )
            artifact_kind = "COMPOSITION_FAILURE_EVIDENCE_V1"
        payload: dict[str, object] = {
            "artifact_kind": artifact_kind,
            "certificate_hash": certificate_hash,
            "claim": result.claim.value,
            "compiler_invoked": result.compiler_invoked,
            "contract_ids": list(result.contract_ids),
            "disposition": result.disposition.value,
            "holdout_execution_counts": dict(sorted(context.holdout_execution_counts.items())),
            "cegis": dict(cegis_evidence) if cegis_evidence is not None else None,
            "minimization": (
                dict(minimization_evidence) if minimization_evidence is not None else None
            ),
            "manifest_hash": hash_canonical(manifest),
            # Full reports are phase-keyed in evidence/interactions.json. The
            # context now points at the final candidate, so do not mislabel
            # those reports as the earlier naive-composition execution.
            "naive_composition": _composition_payload(result.naive_composition, {}),
            "output": output.as_posix(),
            "parent_patch_ids": list(result.parent_patch_ids),
            "patch_id": patch_id,
            "status": result.disposition.value,
            "warnings": list(result.warnings),
        }
        if result.compilation is not None:
            payload["compilation"] = _jsonable(result.compilation)
        if result.verified:
            return _CommandResult(payload)
        if result.disposition is MergeDisposition.COMPILER_FAILED:
            payload["status"] = "UNSUPPORTED"
            return _CommandResult(payload, EXIT_UNSUPPORTED)
        return _CommandResult(payload, EXIT_FAILED)

    _invoke(operation, compact=json_output)


@app.command("audit")
def audit_command(
    base_checkpoint: Path = typer.Option(..., "--base", exists=True),
    patch_dir: Path = typer.Option(..., "--patch-dir", exists=True, file_okay=False),
    output: Path = typer.Option(..., "--output"),
    contracts_policy: Path | None = typer.Option(None, "--contracts", exists=True),
    adapter_spec: str = typer.Option("tiny", "--adapter"),
    subset_budget: int = typer.Option(300, "--subset-budget", min=1),
    max_order: int | None = typer.Option(None, "--max-order", min=1),
    exhaustive_threshold: int = typer.Option(6, "--exhaustive-threshold", min=0, max=24),
    surrogate_degree: int = typer.Option(3, "--surrogate-degree", min=1, max=3),
    seed: int = typer.Option(0, "--seed", min=0),
    device: str = typer.Option("cpu", "--device"),
    dtype: str = typer.Option("float32", "--dtype"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Execute exhaustive or active higher-order composition verification."""

    from modelpact.audit.active import AuditConfig, SubsetEvaluation, audit_patch_pool
    from modelpact.compose.closure import verify_contract_closure
    from modelpact.contracts.parser import load_contract

    def operation() -> _CommandResult:
        paths = tuple(
            sorted(
                child
                for child in patch_dir.iterdir()
                if child.is_dir() and (child / "manifest.json").is_file()
            )
        )
        context = _composition_context(
            adapter_spec=adapter_spec,
            base_checkpoint=base_checkpoint,
            patch_paths=paths,
            device=device,
            dtype=dtype,
        )
        policy_entries: list[tuple[Any, Path]] = []
        if contracts_policy is not None:
            candidates = (
                (contracts_policy,)
                if contracts_policy.is_file()
                else tuple(
                    sorted(
                        path
                        for path in contracts_policy.iterdir()
                        if path.is_file() and path.suffix.lower() in {".json", ".yaml", ".yml"}
                    )
                )
            )
            if not candidates:
                raise ValueError("external audit policy contains no Behavior Contract files")
            for path in candidates:
                contract = load_contract(path)
                prior = context.contracts.get(contract.contract_id)
                if prior is not None and prior[0].to_dict() != contract.to_dict():
                    raise ValueError("external audit policy contract identity collision")
                context.contracts[contract.contract_id] = (contract, path)
                policy_entries.append((contract, path))
        policy_ids = tuple(sorted(contract.contract_id for contract, _path in policy_entries))
        by_id = {
            operand.patch_id: dataclasses.replace(
                operand,
                contract_ids=tuple(sorted({*operand.contract_ids, *policy_ids})),
            )
            for operand in context.operands
        }
        providers: dict[str, list[str]] = {}
        for bundle in context.bundles:
            for contract_id in bundle.manifest.provides:
                providers.setdefault(contract_id, []).append(bundle.manifest.patch_id)
        dependency_map: dict[str, tuple[str, ...]] = {}
        for bundle in context.bundles:
            required_patches: set[str] = set()
            for contract_id in bundle.manifest.requires:
                if contract_id in bundle.manifest.provides:
                    continue
                provider_candidates: tuple[str, ...] = tuple(
                    sorted(dict.fromkeys(providers.get(contract_id, [])))
                )
                if not provider_candidates:
                    raise ValueError(
                        f"patch {bundle.manifest.patch_id} requires unavailable contract "
                        f"{contract_id}"
                    )
                if len(provider_candidates) != 1:
                    raise ValueError(
                        f"patch {bundle.manifest.patch_id} has an ambiguously provided "
                        f"required contract {contract_id}: {list(provider_candidates)}"
                    )
                required_patches.add(provider_candidates[0])
            dependency_map[bundle.manifest.patch_id] = tuple(sorted(required_patches))

        def oracle(subset: tuple[str, ...]) -> SubsetEvaluation:
            if not subset:
                baseline_report = context.execute(
                    {},
                    tuple(sorted(context.contracts)),
                    execution_label="audit:empty-stack",
                )
                return SubsetEvaluation(
                    (),
                    {item.contract_id: item.margin for item in baseline_report.margins},
                    baseline_report.outcome,
                    metadata={
                        "composition_claim": "EMPTY_STACK_BASELINE",
                        "execution_evidence_id": context.last_execution_id,
                    },
                    result_hash=context.last_execution_result_hash,
                )
            execution_count_before = context.execution_sequence

            def execute_subset(delta: Mapping[str, Tensor], contract_ids: tuple[str, ...]) -> Any:
                return context.execute(
                    delta,
                    contract_ids,
                    execution_label="audit:subset:" + ",".join(sorted(subset)),
                )

            result = verify_contract_closure(
                [by_id[item] for item in subset],
                executor=execute_subset,
                aliases=_alias_map(context.loaded.manifest.state_schema),
                contradiction_checker=_static_checker(context),
            )
            if result.verification is None:
                margins = dict.fromkeys(result.contract_ids, -1.0)
                outcome = VerificationOutcome.FAIL
            else:
                margins = {item.contract_id: item.margin for item in result.verification.margins}
                outcome = result.verification.outcome
            violated = tuple(sorted(key for key, value in margins.items() if value < 0.0))
            executed = context.execution_sequence > execution_count_before
            return SubsetEvaluation(
                subset,
                margins,
                outcome,
                violated_contracts=violated,
                metadata={
                    "composition_claim": result.claim.value,
                    "execution_evidence_id": (context.last_execution_id if executed else None),
                },
                result_hash=(context.last_execution_result_hash if executed else None),
            )

        result = audit_patch_pool(
            tuple(by_id),
            oracle=oracle,
            config=AuditConfig(
                subset_budget=subset_budget,
                maximum_order=max_order,
                exhaustive_threshold=exhaustive_threshold,
                surrogate_degree=surrogate_degree,
                seed=seed,
            ),
            dependencies=dependency_map,
        )
        payload = {
            "audit": _jsonable(result),
            "evidence_wording": (
                "all combinations executed"
                if result.search_space_exhausted
                else "no unexecuted combination is described as safe"
            ),
            "policy_hash": (
                None
                if contracts_policy is None
                else hash_canonical(
                    {contract.contract_id: sha256_file(path) for contract, path in policy_entries}
                )
            ),
            "status": [claim.value for claim in result.claims],
        }
        output.mkdir(parents=True, exist_ok=False)
        _write_json(output / "audit.json", payload)
        _write_json(
            output / "executed-verification.json",
            {
                candidate_id: {
                    contract_id: report.to_dict() for contract_id, report in sorted(reports.items())
                }
                for candidate_id, reports in sorted(context.execution_reports.items())
            },
        )
        _write_json(
            output / "manifest.json",
            {
                "artifact_hashes": {
                    "audit.json": sha256_file(output / "audit.json"),
                    "executed-verification.json": sha256_file(
                        output / "executed-verification.json"
                    ),
                },
                "claims": [claim.value for claim in result.claims],
                "schema_version": 1,
            },
        )
        exit_code = EXIT_FAILED if AuditClaim.FAILING_SUBSET_FOUND in result.claims else 0
        return _CommandResult({**payload, "output": output.as_posix()}, exit_code)

    _invoke(operation, compact=json_output)


def _descriptor(loaded: _LoadedModel) -> Any:
    from modelpact.rebase.direct import BaseModelDescriptor

    signature = loaded.manifest.signature
    return BaseModelDescriptor(
        signature=signature.signature_hash,
        architecture_id=signature.architecture_hash,
        module_schema_hash=signature.state_schema_hash,
        tokenizer_hash=signature.tokenizer_hash,
        output_semantics="causal_lm",
        module_shapes={item.name: item.shape for item in loaded.manifest.state_schema.tensors},
        family_id=loaded.adapter.adapter_id,
    )


def _retarget_contract(contract: Any, loaded: _LoadedModel) -> Any:
    from modelpact.contracts.parser import parse_contract

    value = contract.to_dict()
    value["model_requirements"] = {
        "adapter_id": loaded.adapter.adapter_id,
        "architecture_hash": loaded.manifest.signature.architecture_hash,
        "base_signature": loaded.manifest.signature.signature_hash,
        "output_semantics": "causal_lm",
        "state_schema_hash": loaded.manifest.signature.state_schema_hash,
        "tokenizer_hash": loaded.manifest.signature.tokenizer_hash,
    }
    return parse_contract(value)


def _evaluation_margin(evaluations: Sequence[Any]) -> float:
    if not evaluations:
        return -1.0
    values = [item.margin for item in evaluations if item.margin is not None]
    if all(item.outcome is VerificationOutcome.PASS for item in evaluations):
        return min(values, default=1.0)
    return min((value for value in values if value < 0.0), default=-1.0)


def _tiny_rebase_components(
    *,
    source: _LoadedModel,
    target: _LoadedModel,
    bundle: PatchBundle,
    contracts: Mapping[str, tuple[Any, Path]],
    source_contract_ids: tuple[str, ...],
    dense: Mapping[str, Tensor],
    maximum_rank: int,
    maximum_modules: int,
    seed: int,
) -> tuple[Any, Any, Any]:
    from modelpact.compiler.constraints import DifferentiableConstraint, DifferentiableObjective
    from modelpact.compiler.contracts import prepare_contract
    from modelpact.compiler.optimize import OptimizerConfig, compile_low_rank_patch
    from modelpact.compiler.result import CompilationResult, CompilationStatus
    from modelpact.compiler.semantic_cegis import differentiable_refinement_problem
    from modelpact.rebase.compile import BehavioralRecompileResult, TeacherContext
    from modelpact.rebase.direct import RebaseVerification
    from modelpact.verify.engine import combine_outcomes

    target_contracts = {
        identifier: (_retarget_contract(contract, target), path)
        for identifier, (contract, path) in sorted(contracts.items())
    }
    reports: dict[str, Any] = {}

    def applier(candidate: Mapping[str, Tensor], _descriptor: Any) -> Mapping[str, Tensor]:
        return candidate

    def verifier(
        candidate: object,
        target_contract_ids: tuple[str, ...],
        guard_contract_ids: tuple[str, ...],
    ) -> Any:
        if not isinstance(candidate, Mapping):
            raise TypeError("rebase candidate must be a tensor mapping")
        typed_candidate = {
            str(name): value
            for name, value in candidate.items()
            if isinstance(name, str) and isinstance(value, Tensor)
        }
        if len(typed_candidate) != len(candidate):
            raise TypeError("rebase candidate contains a non-tensor delta")
        model = _model_with_dense_delta(
            target.model,
            typed_candidate,
            target.manifest.state_schema,
        )
        requested_contracts = set(target_contract_ids)
        requested_contracts.update(
            identifier.removesuffix(":guards") for identifier in guard_contract_ids
        )
        local_reports: dict[str, Any] = {}
        for identifier in sorted(requested_contracts):
            contract, path = target_contracts[identifier]
            local_reports[identifier] = _verification_report(
                loaded=target,
                model=model,
                base_model=target.model,
                contract=contract,
                contract_path=path,
                candidate_id=f"rebase:{bundle.manifest.patch_id}",
                include_holdout=False,
            )
        reports.clear()
        reports.update(local_reports)
        target_margins = {
            identifier: _evaluation_margin(local_reports[identifier].target_results)
            for identifier in target_contract_ids
        }
        guard_margins = {
            identifier: _evaluation_margin(
                local_reports[identifier.removesuffix(":guards")].guard_results
            )
            for identifier in guard_contract_ids
        }
        component_outcomes = [
            item.outcome
            for identifier in target_contract_ids
            for item in local_reports[identifier].target_results
        ]
        component_outcomes.extend(
            item.outcome
            for identifier in guard_contract_ids
            for item in local_reports[identifier.removesuffix(":guards")].guard_results
        )
        outcome = combine_outcomes(component_outcomes)
        if outcome is VerificationOutcome.NOT_APPLICABLE:
            outcome = VerificationOutcome.FAIL
        failures = tuple(
            sorted(
                f"{identifier}:{len(report.prompt_failures)}-prompt-failures"
                for identifier, report in local_reports.items()
                if report.prompt_failures
            )
        )
        return RebaseVerification(
            outcome,
            target_margins=target_margins,
            guard_margins=guard_margins,
            prompt_failures=failures,
        )

    def teacher_builder(_request: Any) -> Any:
        prepared = {
            identifier: prepare_contract(
                target.adapter,
                target.model,
                contract,
                path,
            )
            for identifier, (contract, path) in target_contracts.items()
        }
        source_teacher = _model_with_dense_delta(
            source.model,
            dense,
            source.manifest.state_schema,
        )
        teacher_objectives: list[DifferentiableObjective] = []
        declared_objectives: list[DifferentiableObjective] = []
        guards: list[DifferentiableConstraint] = []
        evidence_count = 0
        old_margins: dict[str, float] = {}
        source_identifiers = set(source_contract_ids)
        for identifier, item in sorted(prepared.items()):
            batches = tuple(
                example for objective in item.objectives for example in objective.batches
            )
            if identifier in source_identifiers:
                evidence_count += len(batches)
            if batches and identifier in source_identifiers:
                teacher_objectives.append(
                    _teacher_kl_objective(
                        adapter=target.adapter,
                        teacher_model=source_teacher,
                        objective_id=f"source-teacher:{identifier}",
                        batches=batches,
                    )
                )
            declared_objectives.extend(
                DifferentiableObjective(
                    f"contract:{identifier}:{objective.objective_id}",
                    objective.batches,
                    objective.loss,
                    objective.weight,
                )
                for objective in item.objectives
            )
            guards.extend(
                DifferentiableConstraint(
                    f"new-base:{identifier}:{guard.constraint_id}",
                    guard.batches,
                    guard.measure,
                    guard.maximum,
                )
                for guard in item.guards
            )
            if identifier in source_identifiers:
                original_contract, original_path = contracts[identifier]
                old_report = _verification_report(
                    loaded=source,
                    model=source_teacher,
                    base_model=source.model,
                    contract=original_contract,
                    contract_path=original_path,
                    candidate_id=f"source-teacher:{bundle.manifest.patch_id}",
                    include_holdout=False,
                )
                old_margins[identifier] = _report_margin(old_report)
        payload = {
            "declared_objectives": tuple(declared_objectives),
            "guards": tuple(guards),
            "teacher_objectives": tuple(teacher_objectives),
        }
        return TeacherContext(payload, target.model, old_margins, evidence_count)

    def recompiler(request: Any) -> Any:
        context = cast(Mapping[str, object], request.old_patched_teacher)
        declared = cast(tuple[DifferentiableObjective, ...], context["declared_objectives"])
        teacher_objectives = cast(
            tuple[DifferentiableObjective, ...], context["teacher_objectives"]
        )
        guards = cast(tuple[DifferentiableConstraint, ...], context["guards"])
        objectives = (*declared, *teacher_objectives)
        if not teacher_objectives:
            return BehavioralRecompileResult(
                candidate_delta=None,
                optimization_succeeded=False,
                budget_exhausted=False,
                steps_executed=0,
                restarts_executed=0,
                failure_reason=(
                    "the patch contracts expose no compile objective probes from which "
                    "to observe the source patched teacher"
                ),
            )
        initial_delta: Mapping[str, Tensor] = dense if request.direct_transfer.attempted else {}
        initialized = _model_with_dense_delta(
            target.model,
            initial_delta,
            target.manifest.state_schema,
        )
        result = compile_low_rank_patch(
            initialized,
            objectives,
            guards,
            config=OptimizerConfig(
                maximum_rank=maximum_rank,
                maximum_modules=maximum_modules,
                steps=request.budget.maximum_steps,
                seed=seed,
                patience=max(1, min(50, request.budget.maximum_steps)),
            ),
        )
        residual = (
            _compilation_dense_delta(result, target.manifest.state_schema)
            if result.feasible
            else {}
        )
        candidate: Mapping[str, Tensor] | None
        if residual and initial_delta:
            candidate = _sum_dense_deltas(
                initial_delta,
                residual,
                base_signature=target.manifest.signature.signature_hash,
                state_schema=target.manifest.state_schema,
            )
        else:
            candidate = residual or None
        executed_steps = len(result.evidence)
        return BehavioralRecompileResult(
            candidate_delta=candidate,
            optimization_succeeded=result.feasible,
            budget_exhausted=(
                not result.feasible and executed_steps >= request.budget.maximum_steps
            ),
            steps_executed=executed_steps,
            restarts_executed=1,
            violated_contracts=tuple(sorted(result.violated_constraints)),
            complexity={
                "active_modules": len(result.active_modules),
                "parameters": sum(value.numel() for value in residual.values()),
                "total_rank": sum(result.ranks.values()),
            },
            failure_reason=None if result.feasible else "; ".join(result.warnings),
        )

    refinement_teacher: Any | None = None

    def refinement_compiler(
        initial_delta: Mapping[str, Tensor],
        target_examples: tuple[Any, ...],
        guard_examples: tuple[Any, ...],
        maximum_steps: int,
    ) -> CompilationResult:
        nonlocal refinement_teacher
        if refinement_teacher is None:
            refinement_teacher = teacher_builder(None)
        teacher_payload = cast(Mapping[str, object], refinement_teacher.old_patched_teacher)
        declared = cast(
            tuple[DifferentiableObjective, ...],
            teacher_payload["declared_objectives"],
        )
        teacher_objectives = cast(
            tuple[DifferentiableObjective, ...],
            teacher_payload["teacher_objectives"],
        )
        declared_guards = cast(
            tuple[DifferentiableConstraint, ...],
            teacher_payload["guards"],
        )
        refinement_objectives, refinement_guards = differentiable_refinement_problem(
            target.adapter,
            target.model,
            target_examples,
            guard_examples,
        )
        initialized = _model_with_dense_delta(
            target.model,
            initial_delta,
            target.manifest.state_schema,
        )
        compiled = compile_low_rank_patch(
            initialized,
            (*declared, *teacher_objectives, *refinement_objectives),
            (*declared_guards, *refinement_guards),
            config=OptimizerConfig(
                maximum_rank=maximum_rank,
                maximum_modules=maximum_modules,
                steps=maximum_steps,
                seed=seed,
                patience=max(1, min(50, maximum_steps)),
            ),
        )
        residual = (
            _compilation_dense_delta(compiled, target.manifest.state_schema)
            if compiled.feasible
            else {}
        )
        if compiled.feasible:
            final_delta = (
                _sum_dense_deltas(
                    initial_delta,
                    residual,
                    base_signature=target.manifest.signature.signature_hash,
                    state_schema=target.manifest.state_schema,
                )
                if residual
                else {
                    name: value.detach().cpu().clone()
                    for name, value in sorted(initial_delta.items())
                }
            )
            status = CompilationStatus.FEASIBLE
        else:
            final_delta = {}
            status = CompilationStatus.INFEASIBLE_WITHIN_BUDGET
        return CompilationResult(
            status=status,
            deltas=final_delta,
            factors={},
            active_modules=tuple(sorted(final_delta)),
            ranks={
                name: min(value.shape) if value.ndim == 2 else 0
                for name, value in final_delta.items()
            },
            evidence=list(compiled.evidence),
            best_step=compiled.best_step,
            best_target_loss=compiled.best_target_loss,
            violated_constraints=dict(compiled.violated_constraints),
            warnings=list(compiled.warnings),
            metadata={
                **dict(compiled.metadata),
                "optimization_steps": len(compiled.evidence),
                "semantic_cegis_guard_constraints": len(refinement_guards),
                "semantic_cegis_target_objectives": len(refinement_objectives),
            },
        )

    return applier, verifier, (teacher_builder, recompiler, refinement_compiler, reports)


@app.command("rebase")
def rebase_command(
    patch: Path = typer.Argument(..., exists=True),
    from_base: Path = typer.Option(..., "--from-base", exists=True),
    onto: Path = typer.Option(..., "--onto", exists=True),
    target_adapter: str = typer.Option(..., "--target-adapter"),
    output: Path = typer.Option(..., "--output"),
    source_adapter: str | None = typer.Option(None, "--source-adapter"),
    new_base_policy: Path | None = typer.Option(
        None,
        "--new-base-policy",
        exists=True,
        readable=True,
        help="Optional guard-only Behavior Contract evaluated relative to the new base.",
    ),
    maximum_steps: int = typer.Option(200, "--maximum-steps", min=1, max=10_000_000),
    max_rank: int = typer.Option(16, "--max-rank", min=1, max=4096),
    max_modules: int = typer.Option(12, "--max-modules", min=1, max=100_000),
    cegis_rounds: int = typer.Option(2, "--cegis-rounds", min=1, max=10_000),
    cegis_search_budget: int = typer.Option(
        16,
        "--cegis-search-budget",
        min=1,
        max=100_000,
    ),
    minimization_budget: int = typer.Option(
        32,
        "--minimization-budget",
        min=1,
        max=100_000,
    ),
    seed: int = typer.Option(0, "--seed", min=0),
    device: str = typer.Option("cpu", "--device"),
    dtype: str = typer.Option("float32", "--dtype"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Verify direct transfer, then behaviorally recompile failed tiny-model transfers."""

    from modelpact.codegen import emit_apply_script, emit_verify_script
    from modelpact.compiler.minimize import minimize_patch
    from modelpact.contracts.parser import load_contract
    from modelpact.patch.bundle import attach_bundle_artifacts, create_patch_bundle
    from modelpact.patch.validate import validate_base_signature
    from modelpact.rebase.compile import (
        BehavioralRecompileResult,
        RebaseBudget,
        RebaseDisposition,
        RebaseRequest,
        TeacherContext,
        semantic_rebase,
    )
    from modelpact.rebase.direct import RebasePatch
    from modelpact.status import RebaseClaim
    from modelpact.verify.certificate import build_certificate

    def operation() -> _CommandResult:
        effective_source_adapter = source_adapter or target_adapter
        source = _load_model(
            effective_source_adapter,
            from_base,
            device=device,
            dtype=dtype,
        )
        target = _load_model(target_adapter, onto, device=device, dtype=dtype)
        bundle = load_patch_bundle(patch, state_schema=source.manifest.state_schema)
        validate_base_signature(bundle.manifest.base_signature, source.manifest.signature)
        source_contracts = _load_bundle_contracts(bundle)
        source_contract_ids = tuple(sorted(source_contracts))
        target_contract_ids = tuple(
            sorted(
                identifier
                for identifier, (contract, _path) in source_contracts.items()
                if contract.targets
            )
        )
        preservation_contract_ids = tuple(
            sorted(
                f"{identifier}:guards"
                for identifier, (contract, _path) in source_contracts.items()
                if contract.guards
            )
        )
        if not target_contract_ids or not preservation_contract_ids:
            return _CommandResult(
                {
                    "reason": (
                        "semantic rebase certification requires at least one target assertion "
                        "and one preservation assertion across the bundled contract union"
                    ),
                    "status": "UNSUPPORTED",
                },
                EXIT_UNSUPPORTED,
            )
        all_contracts = dict(source_contracts)
        new_base_guard_ids: tuple[str, ...] = ()
        if new_base_policy is not None:
            policy_contract = load_contract(new_base_policy)
            if policy_contract.targets or not policy_contract.guards:
                raise ValueError(
                    "--new-base-policy must be a guard-only Behavior Contract with at "
                    "least one preservation assertion"
                )
            prior = all_contracts.get(policy_contract.contract_id)
            if prior is not None and prior[0].to_dict() != policy_contract.to_dict():
                raise ValueError("new-base policy contract identity collision")
            all_contracts[policy_contract.contract_id] = (policy_contract, new_base_policy)
            new_base_guard_ids = (f"{policy_contract.contract_id}:guards",)
        retargeted_contracts = {
            identifier: (_retarget_contract(contract, target), path)
            for identifier, (contract, path) in sorted(all_contracts.items())
        }
        dense = _bundle_dense_delta(bundle)

        source_teacher_model = _model_with_dense_delta(
            source.model,
            dense,
            source.manifest.state_schema,
        )
        source_teacher_reports = {
            identifier: _verification_report(
                loaded=source,
                model=source_teacher_model,
                base_model=source.model,
                contract=contract,
                contract_path=path,
                candidate_id=f"source-teacher-validation:{bundle.manifest.patch_id}",
                include_holdout=False,
            )
            for identifier, (contract, path) in sorted(source_contracts.items())
        }
        source_teacher_passed = all(
            report.outcome is VerificationOutcome.PASS for report in source_teacher_reports.values()
        )
        if not source_teacher_passed:
            output.mkdir(parents=True, exist_ok=False)
            evidence = {
                "reason": (
                    "source patched teacher failed its bundled visible target or preservation "
                    "contracts; rebase refused to distill an invalid teacher"
                ),
                "schema_version": 1,
                "source_patch_id": bundle.manifest.patch_id,
                "source_teacher_verification": {
                    identifier: report.to_dict()
                    for identifier, report in sorted(source_teacher_reports.items())
                },
                "status": "SOURCE_PATCHED_TEACHER_FAILED",
            }
            _write_json(output / "rebase-evidence.json", evidence)
            return _CommandResult(evidence, EXIT_FAILED)

        applier, verifier, tiny_components = _tiny_rebase_components(
            source=source,
            target=target,
            bundle=bundle,
            contracts=all_contracts,
            source_contract_ids=source_contract_ids,
            dense=dense,
            maximum_rank=max_rank,
            maximum_modules=max_modules,
            seed=seed,
        )
        tiny_teacher_builder, tiny_recompiler, tiny_refinement_compiler, _reports = tiny_components
        if effective_source_adapter == "tiny" and target_adapter == "tiny":
            teacher_builder = tiny_teacher_builder
            recompiler = tiny_recompiler
        else:

            def teacher_builder(_request: Any) -> Any:
                return TeacherContext(
                    None,
                    None,
                    {},
                    0,
                )

            def recompiler(_request: Any) -> Any:
                return BehavioralRecompileResult(
                    candidate_delta=None,
                    optimization_succeeded=False,
                    budget_exhausted=False,
                    steps_executed=0,
                    restarts_executed=0,
                    failure_reason=(
                        "custom and Hugging Face semantic recompilation require an explicit "
                        "trusted compiler integration"
                    ),
                )

        result = semantic_rebase(
            RebaseRequest(
                patch=RebasePatch(
                    patch_id=bundle.manifest.patch_id,
                    source_base_signature=source.manifest.signature.signature_hash,
                    delta=dense,
                    target_contract_ids=target_contract_ids,
                    preservation_contract_ids=preservation_contract_ids,
                ),
                source_base=_descriptor(source),
                target_base=_descriptor(target),
                new_base_guard_ids=new_base_guard_ids,
                budget=RebaseBudget(maximum_steps),
                allow_cross_architecture=True,
                compiler_configuration={
                    "maximum_modules": max_modules,
                    "maximum_rank": max_rank,
                    "seed": seed,
                },
            ),
            applier=applier,
            verifier=verifier,
            teacher_builder=teacher_builder,
            recompiler=recompiler,
        )
        if not result.verified:
            output.mkdir(parents=True, exist_ok=False)
            evidence = {
                "claim": result.claim.value,
                "disposition": result.disposition.value,
                "evidence": result.evidence.to_dict(),
                "recompile": _jsonable(result.recompile),
                "source_patch_id": bundle.manifest.patch_id,
                "status": result.claim.value,
            }
            _write_json(output / "rebase-evidence.json", evidence)
            if result.disposition is RebaseDisposition.INSUFFICIENT_TEACHER_EVIDENCE:
                evidence["status"] = "UNSUPPORTED"
                return _CommandResult(evidence, EXIT_UNSUPPORTED)
            if result.claim is RebaseClaim.REBASE_INCONCLUSIVE:
                return _CommandResult(evidence, EXIT_INCONCLUSIVE)
            return _CommandResult(evidence, EXIT_FAILED)

        semantic = result.claim is RebaseClaim.SEMANTIC_REBASE_VERIFIED
        semantic_cegis_run: Any | None = None
        cegis_evidence: dict[str, object] = {
            "outcome": "NOT_APPLICABLE",
            "reason": "direct transplant required no semantic recompilation",
            "schema_version": 1,
        }
        if semantic:
            from modelpact.compiler.semantic_cegis import (
                SemanticCEGISUnsupportedError,
                candidate_satisfies_semantic_working_set,
                run_semantic_cegis,
            )

            def candidate_model_for_search(delta: Mapping[str, Tensor]) -> nn.Module:
                return _model_with_dense_delta(
                    target.model,
                    delta,
                    target.manifest.state_schema,
                )

            def compile_refinement(
                target_examples: tuple[Any, ...],
                guard_examples: tuple[Any, ...],
            ) -> Any:
                return tiny_refinement_compiler(
                    result.delta,
                    target_examples,
                    guard_examples,
                    maximum_steps,
                )

            try:
                semantic_cegis_run = run_semantic_cegis(
                    target.adapter,
                    target.model,
                    retargeted_contracts,
                    compile_candidate=compile_refinement,
                    candidate_model=candidate_model_for_search,
                    maximum_rounds=cegis_rounds,
                    search_budget_per_domain_per_round=cegis_search_budget,
                    seed=seed,
                )
            except SemanticCEGISUnsupportedError as error:
                output.mkdir(parents=True, exist_ok=False)
                cegis_evidence = {
                    "outcome": "UNSUPPORTED",
                    "reason": str(error),
                    "schema_version": 1,
                }
                evidence = {
                    "cegis": cegis_evidence,
                    "claim": RebaseClaim.REBASE_INCONCLUSIVE.value,
                    "reason": "semantic rebase requires executable bounded CEGIS semantics",
                    "source_patch_id": bundle.manifest.patch_id,
                    "status": "UNSUPPORTED",
                }
                _write_json(output / "rebase-evidence.json", evidence)
                return _CommandResult(evidence, EXIT_UNSUPPORTED)
            cegis_evidence = semantic_cegis_run.to_dict()
            if not semantic_cegis_run.candidate.feasible:
                output.mkdir(parents=True, exist_ok=False)
                evidence = {
                    "cegis": cegis_evidence,
                    "claim": RebaseClaim.REBASE_FAILED.value,
                    "reason": "semantic CEGIS recompilation was infeasible within budget",
                    "source_patch_id": bundle.manifest.patch_id,
                    "status": RebaseClaim.REBASE_FAILED.value,
                }
                _write_json(output / "rebase-evidence.json", evidence)
                return _CommandResult(evidence, EXIT_FAILED)
            refined_verification = verifier(
                semantic_cegis_run.candidate.deltas,
                target_contract_ids,
                tuple(sorted({*preservation_contract_ids, *new_base_guard_ids})),
            )
            accumulated_passed = candidate_satisfies_semantic_working_set(
                semantic_cegis_run,
                target.adapter,
                target.model,
                retargeted_contracts,
                candidate_model_for_search,
                semantic_cegis_run.candidate.deltas,
            )
            if (
                not refined_verification.passes(
                    target_contract_ids,
                    tuple(sorted({*preservation_contract_ids, *new_base_guard_ids})),
                )
                or not accumulated_passed
            ):
                output.mkdir(parents=True, exist_ok=False)
                evidence = {
                    "cegis": cegis_evidence,
                    "claim": RebaseClaim.REBASE_FAILED.value,
                    "reason": (
                        "semantic CEGIS candidate failed the union contract or accumulated "
                        "visible working set"
                    ),
                    "source_patch_id": bundle.manifest.patch_id,
                    "status": RebaseClaim.REBASE_FAILED.value,
                    "verification": _jsonable(refined_verification),
                }
                _write_json(output / "rebase-evidence.json", evidence)
                return _CommandResult(evidence, EXIT_FAILED)
            updated_recompile = (
                dataclasses.replace(
                    result.recompile,
                    candidate_delta=semantic_cegis_run.candidate.deltas,
                )
                if result.recompile is not None
                else None
            )
            result = dataclasses.replace(
                result,
                delta=semantic_cegis_run.candidate.deltas,
                recompile=updated_recompile,
                verification=refined_verification,
                evidence=dataclasses.replace(
                    result.evidence,
                    new_patched_behavior=refined_verification.target_margins,
                    new_base_preservation=refined_verification.guard_margins,
                ),
            )
        minimization_evidence: dict[str, object] = {
            "claim": "UNMINIMIZED",
            "reason": (
                "direct transplant preserves the source patch representation"
                if not semantic
                else "semantic recompilation did not produce a delta to minimize"
            ),
            "schema_version": 1,
        }
        factorized: Mapping[str, tuple[Tensor, Tensor]] | None = None
        if semantic and result.delta:

            def minimization_verifier(candidate: dict[str, Tensor]) -> bool:
                if not candidate:
                    return False
                if semantic_cegis_run is not None:
                    from modelpact.compiler.semantic_cegis import (
                        candidate_satisfies_semantic_working_set,
                    )

                    if not candidate_satisfies_semantic_working_set(
                        semantic_cegis_run,
                        target.adapter,
                        target.model,
                        retargeted_contracts,
                        candidate_model_for_search,
                        candidate,
                    ):
                        return False
                verification = verifier(
                    candidate,
                    target_contract_ids,
                    tuple(sorted({*preservation_contract_ids, *new_base_guard_ids})),
                )
                return bool(
                    verification.passes(
                        target_contract_ids,
                        tuple(sorted({*preservation_contract_ids, *new_base_guard_ids})),
                    )
                )

            minimization = minimize_patch(
                dict(result.delta),
                minimization_verifier,
                verification_budget=minimization_budget,
                seed=seed,
            )
            minimized_verification = verifier(
                minimization.deltas,
                target_contract_ids,
                tuple(sorted({*preservation_contract_ids, *new_base_guard_ids})),
            )
            complexity = {
                "active_modules": len(minimization.deltas),
                "parameters": sum(
                    left.numel() + right.numel() for left, right in minimization.factors.values()
                ),
                "total_rank": sum(left.shape[1] for left, _right in minimization.factors.values()),
            }
            updated_recompile = (
                dataclasses.replace(
                    result.recompile,
                    candidate_delta=minimization.deltas,
                    complexity=complexity,
                )
                if result.recompile is not None
                else None
            )
            result = dataclasses.replace(
                result,
                delta=minimization.deltas,
                recompile=updated_recompile,
                verification=minimized_verification,
                evidence=dataclasses.replace(
                    result.evidence,
                    new_patched_behavior=minimized_verification.target_margins,
                    new_base_preservation=minimized_verification.guard_margins,
                    patch_complexity_after=complexity,
                ),
            )
            if minimization.factors and set(minimization.factors) == set(minimization.deltas):
                factorized = minimization.factors
            minimization_claims = [claim.value for claim in minimization.claims]
            minimization_evidence = {
                "candidates": _jsonable(minimization.candidates),
                "claims": minimization_claims,
                "schema_version": 1,
                "verification_budget": minimization_budget,
                "verification_budget_used": minimization.verification_budget_used,
            }
            if "UNMINIMIZED" in minimization_claims:
                raise RuntimeError(
                    "semantic rebase minimization produced no bounded minimality claim"
                )

        if factorized is None:
            program, tensors = _dense_program(result.delta, target.manifest.state_schema)
        else:
            program, tensors = _factorized_program(
                factorized,
                target.manifest.state_schema,
            )
        candidate_model = _model_with_dense_delta(
            target.model,
            result.delta,
            target.manifest.state_schema,
        )
        final_reports = {
            identifier: _verification_report(
                loaded=target,
                model=candidate_model,
                base_model=target.model,
                contract=contract,
                contract_path=path,
                candidate_id=f"final-rebase:{bundle.manifest.patch_id}",
                include_holdout=True,
            )
            for identifier, (contract, path) in sorted(retargeted_contracts.items())
        }
        from modelpact.verify.engine import combine_outcomes

        final_outcome = combine_outcomes(tuple(report.outcome for report in final_reports.values()))
        if final_outcome is not VerificationOutcome.PASS:
            output.mkdir(parents=True, exist_ok=False)
            holdout_failed = any(
                contract.holdout.configured
                and final_reports[identifier].holdout_outcome is not VerificationOutcome.PASS
                for identifier, (contract, _path) in retargeted_contracts.items()
            )
            evidence = {
                "claim": result.claim.value,
                "disposition": result.disposition.value,
                "evidence": result.evidence.to_dict(),
                "reason": "final candidate failed validation or sealed holdout",
                "status": "HOLDOUT_FAILED" if holdout_failed else final_outcome.value,
                "verification": {
                    identifier: report.to_dict()
                    for identifier, report in sorted(final_reports.items())
                },
            }
            _write_json(output / "rebase-evidence.json", evidence)
            return _CommandResult(evidence, _outcome_exit(final_outcome))

        validation_evidence: dict[str, object] = {}
        holdout_evidence: dict[str, object] = {}
        for identifier, report in sorted(final_reports.items()):
            validation, holdout = _report_sections(report)
            validation_evidence[identifier] = validation
            holdout_evidence[identifier] = holdout
        packaging_context = _CompositionContext(
            target,
            (bundle,),
            (),
            dict(retargeted_contracts),
            {},
        )
        contract_artifacts, contract_entries, resource_hashes = _composition_contract_artifacts(
            packaging_context,
            include_holdout_resources=True,
        )
        compilation_evidence = {
            "cegis": cegis_evidence,
            "direct_transplant": not semantic,
            "new_base_policy": (
                new_base_policy.resolve().as_posix()
                if new_base_policy is not None
                else "retargeted bundled preservation contracts"
            ),
            "optimization_steps": (
                result.recompile.steps_executed if result.recompile is not None else 0
            ),
            "rebase": result.evidence.to_dict(),
            "recompile": _jsonable(result.recompile),
            "schema_version": 1,
            "source_teacher_verification": {
                identifier: report.to_dict()
                for identifier, report in sorted(source_teacher_reports.items())
            },
        }
        rebased = create_patch_bundle(
            output,
            name=f"{bundle.manifest.name}-rebased",
            base_signature=target.manifest.signature.to_dict(),
            state_schema=target.manifest.state_schema,
            program=program,
            tensors=tensors,
            tool_version=__version__,
            contracts=contract_artifacts,
            supplemental_artifacts={
                "evidence/compile.json": (canonical_dumps(compilation_evidence) + "\n").encode(),
                "evidence/holdout.json": (
                    canonical_dumps({"contracts": holdout_evidence, "schema_version": 1}) + "\n"
                ).encode(),
                "evidence/minimization.json": (
                    canonical_dumps(minimization_evidence) + "\n"
                ).encode(),
                "evidence/rebase.json": (
                    canonical_dumps(result.evidence.to_dict()) + "\n"
                ).encode(),
                "evidence/validation.json": (
                    canonical_dumps({"contracts": validation_evidence, "schema_version": 1}) + "\n"
                ).encode(),
                "probes/hashes.json": (canonical_dumps(resource_hashes) + "\n").encode(),
                "probes/manifest.json": (
                    canonical_dumps(
                        {
                            "roles": ["compile", "validation", "guard", "holdout"],
                            "schema_version": 1,
                            "source_hashes": resource_hashes,
                        }
                    )
                    + "\n"
                ).encode(),
                "report.md": (
                    "# Semantic rebase report\n\n"
                    f"Rebase claim: {result.claim.value}\n\n"
                    "The candidate was independently executed against target and preservation "
                    "contracts on the target base.\n"
                ).encode(),
            },
            provides=tuple(
                sorted(
                    contract.contract_id
                    for contract, _path in retargeted_contracts.values()
                    if contract.targets
                )
            ),
            preserves=tuple(
                sorted(
                    contract.contract_id
                    for contract, _path in retargeted_contracts.values()
                    if contract.guards
                )
            ),
            requires=bundle.manifest.requires,
            verification_policy_hash=hash_canonical(
                {
                    contract.contract_id: {
                        "generation": contract.generation.to_dict(),
                        "statistics": contract.statistics.to_dict(),
                    }
                    for contract, _path in retargeted_contracts.values()
                }
            ),
            rebased_from=bundle.manifest.patch_id,
            compiler_configuration={
                "cegis_rounds": cegis_rounds if semantic else 0,
                "cegis_search_budget_per_domain_per_round": (
                    cegis_search_budget if semantic else 0
                ),
                "maximum_modules": max_modules,
                "maximum_rank": max_rank,
                "mode": "semantic_recompile" if semantic else "direct_transplant",
                "optimization_steps": compilation_evidence["optimization_steps"],
                "seed": seed,
            },
        )
        ordered_reports = tuple(
            final_reports[identifier] for identifier in sorted(retargeted_contracts)
        )
        certificate_entries = tuple(
            (contract, output / relative) for contract, relative in contract_entries
        )
        union_report, union_contract, contract_hashes, verification_policy = _certificate_union(
            ordered_reports, certificate_entries
        )
        certificate = build_certificate(
            union_report,
            union_contract,
            patch_id=rebased.manifest.patch_id,
            checkpoint_hashes=target.manifest.checkpoint_tensor_hashes,
            artifact_hashes=dict(rebased.manifest.artifact_hashes),
            verification_policy=verification_policy,
            contract_hashes=contract_hashes,
            patch_structure={
                "active_targets": sorted(program.targets),
                "patch_bytes": program.estimate_bytes(tensors),
            },
            counterexample_search=cegis_evidence,
            minimization_result=minimization_evidence,
            minimized_within_budget=semantic,
            rebase_result={
                "claim": result.claim.value,
                "source_base_hash": source.manifest.signature.signature_hash,
                "source_patch_id": bundle.manifest.patch_id,
                "target_base_hash": target.manifest.signature.signature_hash,
                "evidence": result.evidence.to_dict(),
                "new_base_guard_ids": list(new_base_guard_ids),
            },
        )
        codegen_root = Path(tempfile.mkdtemp(prefix="modelpact-codegen-"))
        try:
            apply_path = emit_apply_script(
                output, codegen_root / "apply_patch.py", will_live_in_bundle=True
            )
            verify_path = emit_verify_script(
                output, codegen_root / "verify_patch.py", will_live_in_bundle=True
            )
            rebased = attach_bundle_artifacts(
                output,
                {
                    "apply_patch.py": apply_path.read_bytes(),
                    "certificate.json": (certificate.canonical_json() + "\n").encode(),
                    "verify_patch.py": verify_path.read_bytes(),
                },
                state_schema=target.manifest.state_schema,
                require_complete=True,
            )
        finally:
            shutil.rmtree(codegen_root, ignore_errors=True)
        return _CommandResult(
            {
                "claim": result.claim.value,
                "cegis": cegis_evidence,
                "disposition": result.disposition.value,
                "minimization": minimization_evidence,
                "output": output.as_posix(),
                "optimization_steps": compilation_evidence["optimization_steps"],
                "patch_id": rebased.manifest.patch_id,
                "rebased_from": bundle.manifest.patch_id,
                "status": "PASS",
                "verification": {
                    identifier: report.to_dict()
                    for identifier, report in sorted(final_reports.items())
                },
            }
        )

    _invoke(operation, compact=json_output)


@emit_app.command("apply")
def emit_apply_command(
    patch: Path = typer.Argument(..., exists=True),
    output: Path = typer.Option(..., "--output"),
    overwrite: bool = typer.Option(False, "--overwrite"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Emit a standalone, patch-ID-pinned SafeTensors application tool."""

    from modelpact.codegen import emit_apply_script

    def operation() -> _CommandResult:
        result = emit_apply_script(patch, output, overwrite=overwrite)
        return _CommandResult(
            {
                "output": result.as_posix(),
                "script_hash": sha256_file(result),
                "status": "PASS",
            }
        )

    _invoke(operation, compact=json_output)


@emit_app.command("verify")
def emit_verify_command(
    patch: Path = typer.Argument(..., exists=True),
    output: Path = typer.Option(..., "--output"),
    overwrite: bool = typer.Option(False, "--overwrite"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Emit a standalone verifier that never imports ModelPact."""

    from modelpact.codegen import emit_verify_script

    def operation() -> _CommandResult:
        result = emit_verify_script(patch, output, overwrite=overwrite)
        return _CommandResult(
            {
                "output": result.as_posix(),
                "script_hash": sha256_file(result),
                "status": "PASS",
            }
        )

    _invoke(operation, compact=json_output)


@app.command("benchmark")
def benchmark_command(
    name: str = typer.Argument(
        ...,
        help=(
            "forkbench, closure_matrix, collusion, merge, rebase, "
            "rebase_cross_architecture, cegis, r1_loop, or huggingface_local"
        ),
    ),
    output: Path | None = typer.Option(None, "--output"),
    artifacts: Path | None = typer.Option(None, "--artifacts"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Run one deterministic, machine-readable ModelPactBench experiment."""

    from modelpact.modelpactbench.runner import benchmark_succeeded, run_selected

    def operation() -> _CommandResult:
        result = (
            run_selected(name)
            if artifacts is None
            else run_selected(name, artifact_output=artifacts)
        )
        succeeded = benchmark_succeeded(result)
        result_status = result.get("status")
        status = (
            "PASS" if succeeded else (result_status if isinstance(result_status, str) else "FAIL")
        )
        payload = {
            "benchmark": name,
            "result": result,
            "result_hash": hash_canonical(result),
            "status": status,
        }
        if output is not None:
            _write_json(output, result)
            payload["output"] = output.as_posix()
        if artifacts is not None:
            payload["artifacts"] = artifacts.as_posix()
        return _CommandResult(payload, 0 if succeeded else EXIT_FAILED)

    _invoke(operation, compact=json_output)


def _read_stack_spec(path: Path) -> tuple[Path, tuple[Path, ...], bool, int, str]:
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("stack specification exceeds size limit")
    with path.open("rb") as stream:
        value = tomllib.load(stream)
    if not isinstance(value, dict):
        raise ValueError("stack specification must be a TOML table")
    unknown = set(value) - {"schema_version", "base", "patches", "policy"}
    if unknown:
        raise ValueError(f"unknown stack specification fields: {sorted(unknown)}")
    if value.get("schema_version", 1) != 1:
        raise ValueError("only Patch Stack specification version 1 is supported")
    raw_base = value.get("base")
    raw_patches = value.get("patches")
    raw_policy = value.get("policy", {})
    if not isinstance(raw_base, str) or not raw_base:
        raise ValueError("stack base must be a non-empty path string")
    if (
        not isinstance(raw_patches, list)
        or not raw_patches
        or not all(isinstance(item, str) and item for item in raw_patches)
    ):
        raise ValueError("stack patches must be a non-empty array of path strings")
    if not isinstance(raw_policy, dict):
        raise ValueError("stack policy must be a table")
    policy_unknown = set(raw_policy) - {"repair_conflicts", "subset_audit_budget"}
    if policy_unknown:
        raise ValueError(f"unknown stack policy fields: {sorted(policy_unknown)}")
    repair = raw_policy.get("repair_conflicts", False)
    audit_budget = raw_policy.get("subset_audit_budget", 0)
    if not isinstance(repair, bool):
        raise ValueError("repair_conflicts must be boolean")
    if isinstance(audit_budget, bool) or not isinstance(audit_budget, int) or audit_budget < 0:
        raise ValueError("subset_audit_budget must be a non-negative integer")
    root = path.parent.resolve()

    def resolved_path(raw: str) -> Path:
        candidate = Path(raw)
        return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()

    base = resolved_path(raw_base)
    patches = tuple(resolved_path(cast(str, item)) for item in raw_patches)
    policy_hash = hash_canonical({"repair_conflicts": repair, "subset_audit_budget": audit_budget})
    return base, patches, repair, audit_budget, policy_hash


def _stack_dependency_map(context: _CompositionContext) -> dict[str, tuple[str, ...]]:
    """Resolve contract requirements to unambiguous provider patch identities."""

    providers: dict[str, list[str]] = {}
    for bundle in context.bundles:
        for contract_id in bundle.manifest.provides:
            providers.setdefault(contract_id, []).append(bundle.manifest.patch_id)
    dependencies: dict[str, tuple[str, ...]] = {}
    for bundle in context.bundles:
        required_patches: set[str] = set()
        for contract_id in bundle.manifest.requires:
            if contract_id in bundle.manifest.provides:
                continue
            candidates = tuple(sorted(set(providers.get(contract_id, ()))))
            if not candidates:
                raise ValueError(
                    "patch stack has unsatisfied required contracts: "
                    f"patch {bundle.manifest.patch_id} requires {contract_id}"
                )
            if len(candidates) != 1:
                raise ValueError(
                    f"patch {bundle.manifest.patch_id} has an ambiguously provided "
                    f"required contract {contract_id}: {list(candidates)}"
                )
            required_patches.add(candidates[0])
        dependencies[bundle.manifest.patch_id] = tuple(sorted(required_patches))
    return dependencies


def _execute_stack_audit(
    context: _CompositionContext,
    *,
    subset_budget: int,
) -> Any | None:
    """Execute the stack policy's bounded audit without accessing sealed holdouts."""

    if subset_budget == 0:
        return None
    from modelpact.audit.active import AuditConfig, SubsetEvaluation, audit_patch_pool
    from modelpact.compose.closure import verify_contract_closure

    if subset_budget < len(context.operands):
        raise ValueError(
            "subset_audit_budget must be zero or permit verification of every singleton "
            f"({len(context.operands)} required)"
        )
    by_id = {operand.patch_id: operand for operand in context.operands}
    dependencies = _stack_dependency_map(context)

    def oracle(subset: tuple[str, ...]) -> SubsetEvaluation:
        if not subset:
            execution = context.execute(
                {},
                tuple(sorted(context.contracts)),
                execution_label="resolve-audit:empty-stack",
            )
            return SubsetEvaluation(
                (),
                {item.contract_id: item.margin for item in execution.margins},
                execution.outcome,
                metadata={"execution_evidence_id": context.last_execution_id},
                result_hash=context.last_execution_result_hash,
            )
        before = context.execution_sequence
        result = verify_contract_closure(
            [by_id[item] for item in subset],
            executor=lambda delta, contract_ids: context.execute(
                delta,
                contract_ids,
                execution_label="resolve-audit:subset:" + ",".join(subset),
            ),
            aliases=_alias_map(context.loaded.manifest.state_schema),
            contradiction_checker=_static_checker(context),
        )
        if result.verification is None:
            margins = dict.fromkeys(result.contract_ids, -1.0)
            outcome = VerificationOutcome.FAIL
        else:
            margins = {item.contract_id: item.margin for item in result.verification.margins}
            outcome = result.verification.outcome
        executed = context.execution_sequence > before
        return SubsetEvaluation(
            subset,
            margins,
            outcome,
            violated_contracts=tuple(
                sorted(identifier for identifier, margin in margins.items() if margin < 0.0)
            ),
            metadata={
                "composition_claim": result.claim.value,
                "execution_evidence_id": context.last_execution_id if executed else None,
            },
            result_hash=context.last_execution_result_hash if executed else None,
        )

    patch_count = len(by_id)
    possible_subsets = (1 << patch_count) - 1
    return audit_patch_pool(
        tuple(by_id),
        oracle=oracle,
        config=AuditConfig(
            subset_budget=subset_budget,
            maximum_order=patch_count,
            exhaustive_threshold=(patch_count if subset_budget >= possible_subsets else 0),
            seed=0,
        ),
        dependencies=dependencies,
    )


def _resolve_stack_paths(
    *,
    base_checkpoint: Path,
    patch_paths: Sequence[Path],
    adapter_spec: str,
    output: Path,
    repair_conflicts: bool,
    subset_audit_budget: int,
    verification_policy_hash: str,
    device: str,
    dtype: str,
) -> tuple[dict[str, object], int]:
    from modelpact.compose.closure import verify_contract_closure
    from modelpact.compose.merge import (
        MergeBudget,
        semantic_merge,
    )
    from modelpact.compose.stack import (
        PatchLineage,
        PatchReference,
        StackResolutionExecution,
        StackResolutionKind,
        resolve_stack,
    )

    context = _composition_context(
        adapter_spec=adapter_spec,
        base_checkpoint=base_checkpoint,
        patch_paths=patch_paths,
        device=device,
        dtype=dtype,
    )
    _stack_dependency_map(context)
    result = verify_contract_closure(
        context.operands,
        executor=context.execute,
        aliases=_alias_map(context.loaded.manifest.state_schema),
        contradiction_checker=_static_checker(context),
        execute_baselines=True,
    )
    audit = _execute_stack_audit(context, subset_budget=subset_audit_budget)
    compiler_invoked = False
    compiler_evidence: Mapping[str, object] | None = None
    cegis_evidence: Mapping[str, object] | None = None
    minimization_evidence: Mapping[str, object] | None = None
    factorized: Mapping[str, tuple[Tensor, Tensor]] | None = None
    resolution_result = result
    warnings: list[str] = []
    if not result.closed and repair_conflicts and not result.contradictions:
        if adapter_spec != "tiny":
            warnings.append(
                "semantic stack repair is supported only by the built-in tiny adapter; "
                "custom and Hugging Face adapters require a trusted compiler integration"
            )
        else:
            merged = semantic_merge(
                context.operands,
                executor=context.execute,
                compiler=_joint_tiny_compiler(
                    context,
                    maximum_rank=16,
                    maximum_modules=12,
                    seed=0,
                ),
                budget=MergeBudget(maximum_steps=200),
                aliases=_alias_map(context.loaded.manifest.state_schema),
                contradiction_checker=_static_checker(context),
                force_recompile=True,
                execute_baselines=True,
            )
            compiler_invoked = merged.compiler_invoked
            raw_compilation = _jsonable(merged.compilation)
            if isinstance(raw_compilation, Mapping):
                compiler_evidence = cast(Mapping[str, object], raw_compilation)
            if merged.verified:
                from modelpact.compiler.semantic_cegis import SemanticCEGISUnsupportedError

                try:
                    (
                        merged,
                        raw_cegis,
                        raw_minimization,
                        factorized,
                    ) = _refine_tiny_semantic_merge(
                        merged,
                        context,
                        maximum_steps=200,
                        maximum_rank=16,
                        maximum_modules=12,
                        cegis_rounds=2,
                        cegis_search_budget=16,
                        minimization_budget=32,
                        seed=0,
                    )
                    cegis_evidence = raw_cegis
                    minimization_evidence = raw_minimization or None
                    raw_compilation = _jsonable(merged.compilation)
                    if isinstance(raw_compilation, Mapping):
                        compiler_evidence = cast(Mapping[str, object], raw_compilation)
                except SemanticCEGISUnsupportedError as error:
                    cegis_evidence = {
                        "outcome": "UNSUPPORTED",
                        "reason": str(error),
                        "schema_version": 1,
                    }
                    warnings.append(f"semantic CEGIS unsupported: {error}")
                if merged.verified and minimization_evidence is not None:
                    resolution_result = dataclasses.replace(
                        merged.naive_composition,
                        claim=merged.claim,
                        resolved_delta=merged.delta,
                        verification=merged.verification,
                        degraded_contracts=(),
                        unverified_contracts=(),
                    )
            else:
                warnings.extend(merged.warnings)

    selected_before_holdout = resolution_result.closed
    if selected_before_holdout:
        resolution_result = _execute_final_composition_candidate(
            resolution_result,
            context,
            delta=resolution_result.resolved_delta,
        )

    output.mkdir(parents=True, exist_ok=False)
    audit_hash: str | None = None
    if audit is not None:
        raw_audit = _jsonable(audit)
        if not isinstance(raw_audit, dict):
            raise TypeError("serialized stack audit must be an object")
        raw_audit.update(
            {
                "executed_subset_count": audit.executed_subset_count,
                "total_model_executions": audit.total_model_executions,
            }
        )
        audit_payload = {
            "audit": raw_audit,
            "executed_verification": {
                execution_id: {
                    contract_id: report.to_dict() for contract_id, report in sorted(reports.items())
                }
                for execution_id, reports in sorted(context.execution_reports.items())
                if "resolve-audit:" in execution_id
            },
            "schema_version": 1,
        }
        _write_json(output / "composition-audit.json", audit_payload)
        audit_hash = sha256_file(output / "composition-audit.json")

    resolved_hash: str | None = None
    certificate_hash: str | None = None
    resolved_bundle: PatchBundle | None = None
    if resolution_result.closed:
        resolved_bundle, certificate = _create_composite_patch_bundle(
            output / "resolved-patch",
            result=resolution_result,
            context=context,
            operation="semantic-merge" if compiler_invoked else "resolve",
            compiler_evidence=compiler_evidence,
            cegis_evidence=cegis_evidence,
            minimization_evidence=minimization_evidence,
            factorized=factorized,
        )
        resolved_hash = sha256_file(resolved_bundle.path / "manifest.json")
        certificate_hash = certificate.certificate_hash
        kind = (
            StackResolutionKind.VERIFIED_COMPOSITE_PATCH
            if compiler_invoked
            else StackResolutionKind.NAIVE_ADDITIVE_STACK
        )
        exit_code = 0
    elif resolution_result.claim.value == "STATIC_CONTRACT_CONTRADICTION":
        kind = StackResolutionKind.STATIC_CONTRADICTION
        exit_code = EXIT_FAILED
    elif repair_conflicts and adapter_spec != "tiny":
        kind = StackResolutionKind.UNSUPPORTED
        exit_code = EXIT_UNSUPPORTED
    else:
        kind = StackResolutionKind.EMPIRICAL_FAILURE
        exit_code = EXIT_FAILED
    union_contract_hash = hash_canonical(list(resolution_result.contract_ids))
    execution = StackResolutionExecution(
        kind=kind,
        resolved_artifact_hash=resolved_hash,
        verification_policy_hash=verification_policy_hash,
        union_contract_hash=union_contract_hash,
        certificate_hash=certificate_hash,
        audit_hash=audit_hash,
        warnings=tuple(sorted(set(warnings))),
    )
    base_hash = context.loaded.manifest.signature.checkpoint_hash
    references = tuple(
        PatchReference(
            patch_id=bundle.manifest.patch_id,
            patch_hash=sha256_file(bundle.path / "manifest.json"),
            base_hash=base_hash,
            contract_hashes=tuple(sorted(_load_bundle_contracts(bundle))),
            artifact_hash=hash_canonical(dict(bundle.manifest.artifact_hashes)),
            requires=tuple(sorted(bundle.manifest.requires)),
            provides=tuple(sorted(bundle.manifest.provides)),
            lineage=PatchLineage(
                parent_patches=bundle.manifest.parent_patches,
                merged_from=bundle.manifest.merged_from,
                rebased_from=bundle.manifest.rebased_from,
                source_diff=bundle.manifest.source_diff_bundle,
            ),
        )
        for bundle in context.bundles
    )
    resolved = resolve_stack(
        base_hash=base_hash,
        patches=references,
        resolver=lambda _request: execution,
        repair_conflicts=repair_conflicts,
        subset_audit_budget=subset_audit_budget,
    )
    patch_paths_by_id = {
        bundle.manifest.patch_id: bundle.path.resolve().as_posix() for bundle in context.bundles
    }
    lock_value = {
        **resolved.lock.to_dict(),
        "extensions": {
            "modelpact_cli": {
                "base_manifest_hash": context.loaded.manifest.manifest_hash,
                "base_path": base_checkpoint.resolve().as_posix(),
                "dependency_order": list(resolved.dependency_order),
                "patch_paths": patch_paths_by_id,
                "resolved_patch_path": (
                    resolved_bundle.path.resolve().as_posix()
                    if resolved_bundle is not None
                    else None
                ),
            }
        },
    }
    _write_json(output / "stack.lock.json", lock_value)
    evidence = {
        "audit_hash": audit_hash,
        "certificate_hash": certificate_hash,
        "compiler_evidence": compiler_evidence,
        "compiler_invoked": compiler_invoked,
        "cegis": cegis_evidence,
        "minimization": minimization_evidence,
        "composition": _composition_payload(resolution_result, context.reports),
        "lock_hash": sha256_file(output / "stack.lock.json"),
        "resolution": kind.value,
        "resolved_patch_id": (
            resolved_bundle.manifest.patch_id if resolved_bundle is not None else None
        ),
        "schema_version": 1,
        "warnings": list(execution.warnings),
    }
    _write_json(output / "resolution.json", evidence)
    return {
        "lock": lock_value,
        "lock_hash": evidence["lock_hash"],
        "output": output.as_posix(),
        "resolution": kind.value,
        "resolved_patch": (
            resolved_bundle.path.as_posix() if resolved_bundle is not None else None
        ),
        "resolved_patch_id": evidence["resolved_patch_id"],
        "status": "PASS" if exit_code == 0 else kind.value,
    }, exit_code


@app.command("resolve")
def resolve_command(
    stack_spec: Path = typer.Argument(..., exists=True, readable=True),
    output: Path = typer.Option(..., "--output"),
    adapter_spec: str = typer.Option("tiny", "--adapter"),
    device: str = typer.Option("cpu", "--device"),
    dtype: str = typer.Option("float32", "--dtype"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Resolve a declarative stack and pin every executed identity in a lockfile."""

    def operation() -> _CommandResult:
        base, patches, repair, audit_budget, policy_hash = _read_stack_spec(stack_spec)
        resolved_payload, exit_code = _resolve_stack_paths(
            base_checkpoint=base,
            patch_paths=patches,
            adapter_spec=adapter_spec,
            output=output,
            repair_conflicts=repair,
            subset_audit_budget=audit_budget,
            verification_policy_hash=policy_hash,
            device=device,
            dtype=dtype,
        )
        return _CommandResult(resolved_payload, exit_code)

    _invoke(operation, compact=json_output)


_MAX_STACK_LOCK_BYTES = 16 * 1024**2
_MAX_STACK_LOCK_PATH_CHARS = 4_096
_MAX_STACK_MANIFEST_AGGREGATE_BYTES = 512 * 1024**2
_CLI_LOCK_EXTENSION_FIELDS = frozenset(
    {
        "base_manifest_hash",
        "base_path",
        "dependency_order",
        "patch_paths",
        "resolved_patch_path",
    }
)


@dataclass(frozen=True, slots=True)
class _ModelPactCLILockExtension:
    base_manifest_hash: str
    base_path: Path
    dependency_order: tuple[str, ...]
    patch_paths: Mapping[str, Path]
    resolved_patch_path: Path | None


@dataclass(frozen=True, slots=True)
class _ParsedStackLock:
    lock: StackLock
    extension: _ModelPactCLILockExtension | None


def _absolute_lock_path(value: object, *, field_name: str) -> Path:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > _MAX_STACK_LOCK_PATH_CHARS
        or "\x00" in value
    ):
        raise ValueError(f"lockfile {field_name} is empty, oversized, or contains NUL")
    normalized = value.replace("\\", "/")
    if normalized.startswith("//"):
        raise ValueError(f"lockfile {field_name} must not use a UNC or network path")
    normalized_parts = normalized.split("/")
    if any(part in {".", ".."} for part in normalized_parts):
        raise ValueError(f"lockfile {field_name} must not contain traversal components")
    candidate = Path(value)
    if not candidate.is_absolute():
        raise ValueError(f"lockfile {field_name} must be an absolute local path")
    return candidate


def _parse_cli_lock_extension(
    value: object,
    *,
    lock: StackLock,
) -> _ModelPactCLILockExtension:
    if not isinstance(value, Mapping):
        raise ValueError("lockfile extensions.modelpact_cli must be an object")
    unknown = set(value) - _CLI_LOCK_EXTENSION_FIELDS
    required = _CLI_LOCK_EXTENSION_FIELDS - {"resolved_patch_path"}
    missing = required - set(value)
    if unknown:
        raise ValueError(f"unknown ModelPact CLI lock extension fields: {sorted(unknown)}")
    if missing:
        raise ValueError(f"missing ModelPact CLI lock extension fields: {sorted(missing)}")
    base_manifest_hash = value.get("base_manifest_hash")
    if not is_sha256_digest(base_manifest_hash):
        raise ValueError("lockfile base_manifest_hash must be a tagged SHA-256 digest")

    raw_patch_paths = value.get("patch_paths")
    if not isinstance(raw_patch_paths, Mapping):
        raise ValueError("lockfile patch_paths must be an object")
    if set(raw_patch_paths) != set(lock.patch_hashes):
        raise ValueError("lockfile patch_paths must exactly cover patch_hashes")
    patch_paths: dict[str, Path] = {}
    for patch_id, raw_path in raw_patch_paths.items():
        if not isinstance(patch_id, str):
            raise ValueError("lockfile patch_paths keys must be strings")
        patch_paths[patch_id] = _absolute_lock_path(
            raw_path,
            field_name=f"patch_paths.{patch_id}",
        )

    raw_dependency_order = value.get("dependency_order")
    if not isinstance(raw_dependency_order, list) or not all(
        is_sha256_digest(item) for item in raw_dependency_order
    ):
        raise ValueError("lockfile dependency_order must contain patch SHA-256 identities")
    dependency_order = tuple(cast(list[str], raw_dependency_order))
    if len(set(dependency_order)) != len(dependency_order):
        raise ValueError("lockfile dependency_order must not contain duplicates")
    if set(dependency_order) != set(lock.patch_hashes):
        raise ValueError("lockfile dependency_order must exactly cover patch_hashes")

    raw_resolved_path = value.get("resolved_patch_path")
    resolved_path = (
        None
        if raw_resolved_path is None
        else _absolute_lock_path(raw_resolved_path, field_name="resolved_patch_path")
    )
    return _ModelPactCLILockExtension(
        base_manifest_hash=cast(str, base_manifest_hash),
        base_path=_absolute_lock_path(value.get("base_path"), field_name="base_path"),
        dependency_order=dependency_order,
        patch_paths=dict(sorted(patch_paths.items())),
        resolved_patch_path=resolved_path,
    )


def _read_lock(path: Path) -> _ParsedStackLock:
    if path.is_symlink() or not path.is_file():
        raise ValueError("stack lockfile must be a regular file")
    if path.stat().st_size > _MAX_STACK_LOCK_BYTES:
        raise ValueError("stack lockfile exceeds size limit")
    value = strict_json_loads(path.read_bytes(), max_depth=16)
    if not isinstance(value, Mapping):
        raise ValueError("malformed Patch Stack Lockfile v1")
    allowed = STACK_LOCK_FIELDS | {"extensions"}
    unknown = set(value) - allowed
    missing = allowed - set(value)
    if unknown:
        raise ValueError(f"unknown Patch Stack Lockfile v1 fields: {sorted(unknown)}")
    if missing:
        raise ValueError(f"missing Patch Stack Lockfile v1 fields: {sorted(missing)}")
    lock = StackLock.from_dict({field: value[field] for field in STACK_LOCK_FIELDS})
    extensions = value.get("extensions")
    if not isinstance(extensions, Mapping):
        raise ValueError("lockfile extensions must be an object")
    unknown_extensions = set(extensions) - {"modelpact_cli"}
    if unknown_extensions:
        raise ValueError(f"unknown lockfile extensions: {sorted(unknown_extensions)}")
    extension = (
        _parse_cli_lock_extension(extensions["modelpact_cli"], lock=lock)
        if "modelpact_cli" in extensions
        else None
    )
    return _ParsedStackLock(lock=lock, extension=extension)


def _verify_locked_patch_manifests(parsed: _ParsedStackLock) -> None:
    """Bound and authenticate every locked manifest before loading the base model."""

    from modelpact.patch.bundle import MAX_MANIFEST_BYTES

    if parsed.extension is None:
        return
    bounded: list[tuple[str, Path, str]] = []
    aggregate = 0
    for patch_id, expected in sorted(parsed.lock.patch_hashes.items()):
        bundle_path = parsed.extension.patch_paths[patch_id]
        if bundle_path.is_symlink() or not bundle_path.is_dir():
            raise ValueError(f"locked patch path is not a regular directory: {patch_id}")
        manifest_path = bundle_path / "manifest.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ValueError(f"locked patch manifest is not a regular file: {patch_id}")
        size = manifest_path.stat().st_size
        if size > MAX_MANIFEST_BYTES:
            raise ValueError(f"locked patch manifest exceeds the size limit: {patch_id}")
        aggregate += size
        if aggregate > _MAX_STACK_MANIFEST_AGGREGATE_BYTES:
            raise ValueError("locked patch manifests exceed the aggregate size limit")
        bounded.append((patch_id, manifest_path, expected))
    for patch_id, manifest_path, expected in bounded:
        actual = sha256_file(manifest_path, max_bytes=MAX_MANIFEST_BYTES)
        if actual != expected:
            raise ValueError(f"locked patch manifest changed: {patch_id}")


@app.command("revert")
def revert_command(
    lockfile: Path = typer.Argument(..., exists=True, readable=True),
    remove: str = typer.Option(..., "--remove"),
    output: Path = typer.Option(..., "--output"),
    adapter_spec: str = typer.Option(..., "--adapter"),
    repair_conflicts: bool = typer.Option(False, "--repair-conflicts"),
    subset_audit_budget: int = typer.Option(0, "--subset-audit-budget", min=0),
    device: str = typer.Option("cpu", "--device"),
    dtype: str = typer.Option("float32", "--dtype"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Remove one logical patch, reconstruct the remainder, and execute its contracts."""

    def operation() -> _CommandResult:
        parsed = _read_lock(lockfile)
        patch_hashes = dict(parsed.lock.patch_hashes)
        if remove not in patch_hashes:
            raise ValueError(f"patch is not present in the locked stack: {remove}")
        if parsed.extension is None:
            return _CommandResult(
                {
                    "reason": "lockfile does not carry local path resolution metadata",
                    "reversion_grade": "REVERT_FAILED",
                    "status": "UNSUPPORTED",
                },
                EXIT_UNSUPPORTED,
            )
        _verify_locked_patch_manifests(parsed)
        base_checkpoint = parsed.extension.base_path
        remaining_ids = tuple(sorted(set(patch_hashes) - {remove}))
        loaded = _load_model(adapter_spec, base_checkpoint, device=device, dtype=dtype)
        if loaded.manifest.signature.checkpoint_hash != parsed.lock.base_hash:
            raise ValueError("base checkpoint hash no longer matches the lockfile")
        if loaded.manifest.manifest_hash != parsed.extension.base_manifest_hash:
            raise ValueError("base model manifest hash no longer matches the lockfile")
        if not remaining_ids:
            output.mkdir(parents=True, exist_ok=False)
            lock_value: dict[str, object] = {
                "audit_hash": None,
                "base_hash": parsed.lock.base_hash,
                "certificate_hash": None,
                "contract_hashes": [],
                "extensions": {
                    "modelpact_cli": {
                        "base_manifest_hash": loaded.manifest.manifest_hash,
                        "base_path": base_checkpoint.resolve().as_posix(),
                        "dependency_order": [],
                        "patch_paths": {},
                    }
                },
                "patch_hashes": {},
                "resolution": "NAIVE_ADDITIVE_STACK",
                "resolved_artifact_hash": loaded.manifest.signature.checkpoint_hash,
                "schema_version": 1,
                "verification_policy_hash": parsed.lock.verification_policy_hash,
            }
            _write_json(output / "stack.lock.json", lock_value)
            base_payload: dict[str, object] = {
                "lock": lock_value,
                "output": output.as_posix(),
                "removed_patch_id": remove,
                "reversion_grade": "BASE_HASH_RESTORED",
                "status": "PASS",
                "warning": (
                    "restoration uses the original pinned base; no delta subtraction occurred"
                ),
            }
            _write_json(output / "revert-evidence.json", base_payload)
            return _CommandResult(base_payload)
        remaining_paths = tuple(parsed.extension.patch_paths[item] for item in remaining_ids)
        policy_hash = parsed.lock.verification_policy_hash
        resolved_payload, exit_code = _resolve_stack_paths(
            base_checkpoint=base_checkpoint,
            patch_paths=remaining_paths,
            adapter_spec=adapter_spec,
            output=output,
            repair_conflicts=repair_conflicts,
            subset_audit_budget=subset_audit_budget,
            verification_policy_hash=policy_hash,
            device=device,
            dtype=dtype,
        )
        resolved_payload.update(
            {
                "removed_patch_id": remove,
                "reversion_grade": (
                    "VERIFIED_LOGICAL_STACK_RECONSTRUCTED" if exit_code == 0 else "REVERT_FAILED"
                ),
                "warning": (
                    "the remaining stack was reconstructed from the untouched base and original "
                    "patches; no floating-point inverse was described as bitwise exact"
                ),
            }
        )
        _write_json(output / "revert-evidence.json", resolved_payload)
        return _CommandResult(resolved_payload, exit_code)

    _invoke(operation, compact=json_output)


if __name__ == "__main__":
    app()
