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
from modelpact.loading import load_trusted_adapter, parse_dtype
from modelpact.models.manifest import ModelManifest, build_model_manifest
from modelpact.models.schema import ModelStateSchema
from modelpact.patch.ast import Alias, DeltaOp, DeltaProgram, LowRankMatrixDelta, VectorDelta
from modelpact.patch.bundle import PatchBundle, load_patch_bundle, missing_bundle_artifacts
from modelpact.status import VerificationOutcome
from modelpact.util.atomic import atomic_write_text
from modelpact.util.canonical_json import canonical_dumps
from modelpact.util.hashing import hash_canonical, sha256_file

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
    paths = [
        bundle.path / relative
        for relative in sorted(bundle.manifest.artifact_hashes)
        if relative.startswith("contracts/")
        and Path(relative).suffix.lower() in {".json", ".yaml", ".yml"}
        and len(Path(relative).parts) == 2
        and (
            Path(relative).stem in {"target", "preservation"}
            or Path(relative).stem.startswith("contract-")
        )
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
    """Materialize one patch or a deterministic additive patch stack."""

    from modelpact.compose.closure import PatchOperand, additive_compose
    from modelpact.patch.fold import materialize_patch
    from modelpact.patch.validate import validate_base_signature

    def operation() -> _CommandResult:
        if mode != "materialize":
            return _CommandResult(
                {
                    "reason": (
                        "runtime mounts live inside a Python process; use modelpact.patch.mount "
                        "or modelpact verify for an executed mount"
                    ),
                    "status": "UNSUPPORTED",
                },
                EXIT_UNSUPPORTED,
            )
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
                "patch_order": sorted(bundle.manifest.patch_id for bundle in bundles),
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
        contract_path = policy or _contract_paths(bundle)[0]
        contract = load_contract(contract_path)
        with mount_bundle(
            loaded.model,
            bundle,
            loaded.manifest.signature,
            state_schema=loaded.manifest.state_schema,
        ):
            report = _verification_report(
                loaded=loaded,
                model=loaded.model,
                base_model=unpatched,
                contract=contract,
                contract_path=contract_path,
                candidate_id=bundle.manifest.patch_id,
                include_holdout=include_holdout,
            )
        artifact_paths = tuple(sorted(bundle.manifest.artifact_hashes))
        certificate = build_certificate(
            report,
            contract,
            patch_id=bundle.manifest.patch_id,
            checkpoint_hashes=loaded.manifest.checkpoint_tensor_hashes,
            artifact_hashes={
                relative: sha256_file(bundle.path / relative) for relative in artifact_paths
            },
            patch_structure={
                "active_targets": sorted(bundle.program.targets),
                "patch_bytes": bundle.program.estimate_bytes(bundle.tensors),
            },
            additional_warnings=(
                "Certificate was regenerated from model execution; bundled outcomes "
                "were not trusted.",
            ),
        )
        if certificate_output is not None:
            write_certificate(certificate, certificate_output, overwrite=False)
        return _CommandResult(
            {
                "certificate": certificate.to_dict(),
                "certificate_output": (
                    None if certificate_output is None else certificate_output.as_posix()
                ),
                "prompt_failures": _jsonable(report.prompt_failures),
                "report": report.to_dict(),
                "status": report.outcome.value,
            },
            _outcome_exit(report.outcome),
        )

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
    seed: int = typer.Option(0, "--seed", min=0),
    device: str = typer.Option("cpu", "--device"),
    dtype: str = typer.Option("float32", "--dtype"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Compile and independently validate a bounded low-rank patch candidate."""

    from modelpact.codegen import emit_apply_script, emit_verify_script
    from modelpact.compiler.contracts import prepare_contract
    from modelpact.compiler.optimize import OptimizerConfig, compile_low_rank_patch
    from modelpact.compiler.package import compilation_delta_program, compile_evidence
    from modelpact.contracts.parser import load_contract
    from modelpact.patch.bundle import attach_bundle_artifacts, create_patch_bundle
    from modelpact.patch.mount import mount_patch
    from modelpact.verify.certificate import build_certificate

    def operation() -> _CommandResult:
        if cegis_rounds:
            return _CommandResult(
                {
                    "reason": (
                        "the generic CEGIS engine requires an explicit bounded prompt-space "
                        "search callback; this command has no --search-space input"
                    ),
                    "requested_cegis_rounds": cegis_rounds,
                    "status": "UNSUPPORTED",
                },
                EXIT_UNSUPPORTED,
            )
        loaded = _load_model(base, checkpoint, device=device, dtype=dtype)
        contract = load_contract(spec)
        prepared = prepare_contract(loaded.adapter, loaded.model, contract, spec)
        configuration = OptimizerConfig(
            maximum_rank=max_rank,
            maximum_modules=max_modules,
            steps=steps,
            seed=seed,
        )
        result = compile_low_rank_patch(
            loaded.model,
            prepared.objectives,
            prepared.guards,
            config=configuration,
        )
        evidence = compile_evidence(result)
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

        program, tensors = compilation_delta_program(result, loaded.manifest.state_schema)
        unpatched = copy.deepcopy(loaded.model)
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
                        "claim": "UNMINIMIZED",
                        "reason": "no minimization budget was requested",
                        "schema_version": 1,
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
            preserves=(f"{contract.contract_id}:guards",),
            verification_policy_hash=hash_canonical(
                {
                    "generation": contract.generation.to_dict(),
                    "statistics": contract.statistics.to_dict(),
                }
            ),
            compiler_configuration=cast(Mapping[str, object], _jsonable(configuration)),
        )
        certificate = build_certificate(
            report,
            contract,
            patch_id=bundle.manifest.patch_id,
            checkpoint_hashes=loaded.manifest.checkpoint_tensor_hashes,
            artifact_hashes=dict(bundle.manifest.artifact_hashes),
            counterexample_search={
                "outcome": "NOT_EXECUTED",
                "reason": "no counterexample search space was supplied",
                "rounds": 0,
            },
            patch_structure={
                "active_modules": list(result.active_modules),
                "module_ranks": dict(sorted(result.ranks.items())),
                "patch_bytes": program.estimate_bytes(tensors),
            },
            objectives_optimized=True,
        )
        temporary = Path(tempfile.mkdtemp(prefix="modelpact-codegen-"))
        try:
            apply_path = emit_apply_script(output, temporary / "apply_patch.py")
            verify_path = emit_verify_script(output, temporary / "verify_patch.py")
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
                "holdout_outcome": report.holdout_outcome.value,
                "output": output.as_posix(),
                "patch_bytes": program.estimate_bytes(tensors),
                "patch_id": bundle.manifest.patch_id,
                "status": report.outcome.value,
                "verification_result_hash": report.result_hash,
            },
            _outcome_exit(report.outcome),
        )

    _invoke(operation, compact=json_output)


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
    if not required <= set(row):
        raise ValueError(f"difference witness row is missing fields: {sorted(required - set(row))}")
    metrics = row["divergence_metrics"]
    provenance = row.get("provenance", {})
    if not isinstance(metrics, Mapping) or not isinstance(provenance, Mapping):
        raise ValueError("malformed difference witness mappings")

    def float_tuple(name: str) -> tuple[float, ...]:
        value = row.get(name, ())
        if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
            raise ValueError(f"malformed difference witness field: {name}")
        return tuple(float(item) for item in value)

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
    max_new_tokens: int = typer.Option(32, "--max-new-tokens", min=1, max=4096),
    seed: int = typer.Option(0, "--seed", min=0),
    device: str = typer.Option("cpu", "--device"),
    dtype: str = typer.Option("float32", "--dtype"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Extract one empirical difference cluster while guarding nonselected clusters."""

    import pyarrow.parquet as parquet  # type: ignore[import-untyped]

    from modelpact.adapters.base import GenerationPolicy as AdapterGenerationPolicy
    from modelpact.codegen import emit_apply_script, emit_verify_script
    from modelpact.compiler.extract import extract_behavior_cluster
    from modelpact.compiler.optimize import OptimizerConfig
    from modelpact.compiler.package import compilation_delta_program, compile_evidence
    from modelpact.contracts.parser import parse_contract
    from modelpact.patch.bundle import attach_bundle_artifacts, create_patch_bundle
    from modelpact.patch.mount import mount_patch
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
        cluster_values = json.loads((diff_bundle / "clusters.json").read_text(encoding="utf-8"))
        if not isinstance(cluster_values, list):
            raise ValueError("malformed difference cluster index")
        matching = [
            item
            for item in cluster_values
            if isinstance(item, dict) and item.get("cluster_id") == cluster
        ]
        if len(matching) != 1 or not isinstance(matching[0].get("witness_ids"), list):
            raise ValueError(f"unknown or malformed difference cluster: {cluster}")
        selected_ids = set(cast(list[str], matching[0]["witness_ids"]))
        rows = parquet.read_table(diff_bundle / "witnesses.parquet").to_pylist()
        witnesses = tuple(_witness_from_row(cast(Mapping[str, object], row)) for row in rows)
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
        extraction = extract_behavior_cluster(
            base_loaded.adapter,
            base_loaded.model,
            target_loaded.model,
            selected,
            nonselected,
            optimizer_config=optimizer,
        )
        result = extraction.compiler_result
        extraction_payload = {
            "nonselected_base_kl": extraction.nonselected_base_kl,
            "nonselected_witness_ids": list(extraction.nonselected_witness_ids),
            "selected_teacher_kl": extraction.selected_teacher_kl,
            "selected_witness_ids": list(extraction.selected_witness_ids),
            "validation_passed": extraction.validation_passed,
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

        selected_prompts = tuple(item.minimized_input for item in selected)
        nonselected_prompts = tuple(item.minimized_input for item in nonselected)
        target_outputs = generated(target_loaded.model, selected_prompts)
        base_outputs = generated(base_loaded.model, nonselected_prompts)
        compile_rows = [
            {"id": f"selected-{index:06d}", "prompt": prompt, "target": expected}
            for index, (prompt, expected) in enumerate(
                zip(selected_prompts, target_outputs, strict=True)
            )
        ]
        target_rows = [
            {"expected": expected, "id": f"selected-{index:06d}", "prompt": prompt}
            for index, (prompt, expected) in enumerate(
                zip(selected_prompts, target_outputs, strict=True)
            )
        ]
        guard_rows = [
            {"expected": expected, "id": f"guard-{index:06d}", "prompt": prompt}
            for index, (prompt, expected) in enumerate(
                zip(nonselected_prompts, base_outputs, strict=True)
            )
        ]
        contract_value: dict[str, object] = {
            "compile": {
                "objectives": [
                    {
                        "id": "selected-teacher-sequences",
                        "source": "data/compile-targets.jsonl",
                        "type": "teacher_cross_entropy",
                        "weight": 1.0,
                    }
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
                "guards": (
                    [
                        {
                            "id": "retain-nonselected-generations",
                            "minimum_pass_rate": 1.0,
                            "source": "data/guards.jsonl",
                            "type": "free_generation_match",
                        }
                    ]
                    if guard_rows
                    else []
                ),
                "targets": [
                    {
                        "id": "transfer-selected-generations",
                        "minimum_pass_rate": 1.0,
                        "source": "data/validation-targets.jsonl",
                        "type": "free_generation_match",
                    }
                ],
            },
        }
        contract = parse_contract(contract_value)
        program, tensors = compilation_delta_program(result, base_loaded.manifest.state_schema)
        temporary_contract_root = Path(tempfile.mkdtemp(prefix="modelpact-extract-contract-"))
        try:
            contract_path = temporary_contract_root / "target.yaml"
            _write_json(contract_path, contract.to_dict())

            def jsonl(rows_value: Sequence[Mapping[str, object]]) -> bytes:
                return b"".join((canonical_dumps(dict(row)) + "\n").encode() for row in rows_value)

            data = {
                "data/compile-targets.jsonl": jsonl(compile_rows),
                "data/guards.jsonl": jsonl(guard_rows),
                "data/validation-targets.jsonl": jsonl(target_rows),
            }
            for relative, content in data.items():
                target_path = temporary_contract_root / relative
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.write_bytes(content)
            unpatched = copy.deepcopy(base_loaded.model)
            with mount_patch(
                base_loaded.model,
                program,
                tensors,
                state_schema=base_loaded.manifest.state_schema,
            ):
                report = _verification_report(
                    loaded=base_loaded,
                    model=base_loaded.model,
                    base_model=unpatched,
                    contract=contract,
                    contract_path=contract_path,
                    candidate_id=f"extraction:{cluster}",
                    include_holdout=False,
                )
            validation, holdout = _report_sections(report)
            contract_bytes = (canonical_dumps(contract.to_dict()) + "\n").encode()
            contract_artifacts = {
                "contracts/preservation.yaml": contract_bytes,
                "contracts/target.yaml": contract_bytes,
                **{f"contracts/{relative}": content for relative, content in data.items()},
            }
            probe_hashes = {
                relative: hash_canonical(content.decode("utf-8"))
                for relative, content in sorted(data.items())
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
                        canonical_dumps(compile_evidence(result)) + "\n"
                    ).encode(),
                    "evidence/holdout.json": (canonical_dumps(holdout) + "\n").encode(),
                    "evidence/minimization.json": (
                        canonical_dumps({"claim": "UNMINIMIZED", "schema_version": 1}) + "\n"
                    ).encode(),
                    "evidence/validation.json": (
                        canonical_dumps({**validation, "extraction": extraction_payload}) + "\n"
                    ).encode(),
                    "probes/hashes.json": (canonical_dumps(probe_hashes) + "\n").encode(),
                    "probes/manifest.json": (
                        canonical_dumps(
                            {
                                "source_diff": (diff_bundle / "manifest.json").as_posix(),
                                "source_hashes": probe_hashes,
                                "schema_version": 1,
                            }
                        )
                        + "\n"
                    ).encode(),
                    "report.md": (
                        "# Selective extraction report\n\n"
                        f"Selected empirical cluster: {cluster}\n\n"
                        f"Executed verification: {report.outcome.value}\n"
                    ).encode(),
                },
                provides=(contract.contract_id,),
                preserves=(f"{contract.contract_id}:guards",),
                source_diff_bundle=sha256_file(diff_bundle / "manifest.json"),
                compiler_configuration=cast(Mapping[str, object], _jsonable(optimizer)),
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
                objectives_optimized=True,
                additional_warnings=(
                    "No sealed holdout was defined by the selected difference bundle.",
                ),
            )
            codegen_root = Path(tempfile.mkdtemp(prefix="modelpact-codegen-"))
            try:
                apply_path = emit_apply_script(output, codegen_root / "apply_patch.py")
                verify_path = emit_verify_script(output, codegen_root / "verify_patch.py")
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
        effective_outcome = (
            report.outcome if extraction.validation_passed else VerificationOutcome.FAIL
        )
        return _CommandResult(
            {
                "extraction": extraction_payload,
                "output": output.as_posix(),
                "patch_id": bundle.manifest.patch_id,
                "status": effective_outcome.value,
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
) -> Any:
    from modelpact.compiler.constraints import DifferentiableConstraint, DifferentiableObjective
    from modelpact.compiler.contracts import prepare_contract
    from modelpact.compiler.optimize import OptimizerConfig, compile_low_rank_patch
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

        objectives = (*declared_objectives, *parent_teacher_objectives)
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
        candidate = (
            _sum_dense_deltas(
                request.initial_delta,
                residual,
                base_signature=request.base_signature,
                state_schema=context.loaded.manifest.state_schema,
            )
            if residual
            else None
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
                "ranks": dict(sorted(result.ranks.items())),
                "real_optimization": executed_steps > 0,
            },
            failure_reason=None if result.feasible else "; ".join(result.warnings),
        )

    return compile_joint


@dataclass(slots=True)
class _CompositionContext:
    loaded: _LoadedModel
    bundles: tuple[PatchBundle, ...]
    operands: tuple[Any, ...]
    contracts: dict[str, tuple[Any, Path]]
    reports: dict[str, Any]

    def execute(self, delta: Mapping[str, Tensor], contract_ids: tuple[str, ...]) -> Any:
        from modelpact.compose.closure import ContractMargin, MarginKind, VerificationReport
        from modelpact.patch.mount import mount_patch

        model = copy.deepcopy(self.loaded.model)
        program, tensors = _dense_program(delta, self.loaded.manifest.state_schema)
        reports: dict[str, Any] = {}
        with mount_patch(
            model,
            program,
            tensors,
            state_schema=self.loaded.manifest.state_schema,
        ):
            for contract_id in contract_ids:
                contract, path = self.contracts[contract_id]
                reports[contract_id] = _verification_report(
                    loaded=self.loaded,
                    model=model,
                    base_model=self.loaded.model,
                    contract=contract,
                    contract_path=path,
                    candidate_id=f"composition:{hash_canonical(program.to_dict())}",
                    include_holdout=False,
                )
        self.reports = reports
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
                details={"verification_result_hash": reports[contract_id].result_hash},
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
    from modelpact.contracts.static import check_static_contracts

    def check(contract_ids: tuple[str, ...]) -> tuple[Any, ...]:
        result = check_static_contracts([context.contracts[item][0] for item in contract_ids])
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
        "claim": result.claim.value,
        "contract_ids": list(result.contract_ids),
        "contradictions": _jsonable(result.contradictions),
        "degraded_contracts": list(result.degraded_contracts),
        "interaction_margins": dict(sorted(result.interaction_margins.items())),
        "patch_ids": list(result.patch_ids),
        "structural_errors": list(result.structural_errors),
        "unverified_contracts": list(result.unverified_contracts),
        "verification": verification,
    }


def _write_composition_artifacts(
    output: Path,
    *,
    result: Any,
    context: _CompositionContext,
) -> dict[str, object]:
    from modelpact.compose.interactions import module_overlap

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
    contracts_dir = output / "contracts"
    contracts_dir.mkdir()
    for identifier, (contract, _path) in sorted(context.contracts.items()):
        _write_json(
            contracts_dir / f"contract-{identifier.removeprefix('sha256:')}.json",
            contract.to_dict(),
        )
    pairs = []
    for left_index, left in enumerate(context.operands):
        for right in context.operands[left_index + 1 :]:
            overlap = module_overlap(tuple(left.delta), tuple(right.delta))
            pairs.append(
                {
                    "left": left.patch_id,
                    "module_overlap": _jsonable(overlap),
                    "right": right.patch_id,
                }
            )
    payload = _composition_payload(result, context.reports)
    _write_json(output / "verification.json", payload)
    interaction = {
        "contract_margin_interactions": dict(sorted(result.interaction_margins.items())),
        "pairwise_parameter_overlap": pairs,
        "warning": "parameter overlap is diagnostic only; executed contracts are authoritative",
    }
    _write_json(output / "interactions.json", interaction)
    manifest = {
        "artifact_hashes": {
            path.relative_to(output).as_posix(): sha256_file(path)
            for path in sorted(item for item in output.rglob("*") if item.is_file())
        },
        "claim": result.claim.value,
        "contract_ids": list(result.contract_ids),
        "patch_ids": list(result.patch_ids),
        "schema_version": 1,
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


@app.command("compose")
def compose_command(
    patches: list[Path] = typer.Argument(..., exists=True),
    base_checkpoint: Path = typer.Option(..., "--base", exists=True),
    output: Path = typer.Option(..., "--output"),
    adapter_spec: str = typer.Option("tiny", "--adapter"),
    device: str = typer.Option("cpu", "--device"),
    dtype: str = typer.Option("float32", "--dtype"),
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
        )
        manifest = _write_composition_artifacts(output, result=result, context=context)
        payload = {
            **_composition_payload(result, context.reports),
            "manifest_hash": hash_canonical(manifest),
            "output": output.as_posix(),
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
        )
        artifact_result = result.naive_composition
        if result.verified:
            artifact_result = dataclasses.replace(
                artifact_result,
                claim=result.claim,
                resolved_delta=result.delta,
                verification=result.verification,
                degraded_contracts=(),
                unverified_contracts=(),
            )
        manifest = _write_composition_artifacts(
            output,
            result=artifact_result,
            context=context,
        )
        payload: dict[str, object] = {
            "claim": result.claim.value,
            "compiler_invoked": result.compiler_invoked,
            "contract_ids": list(result.contract_ids),
            "disposition": result.disposition.value,
            "manifest_hash": hash_canonical(manifest),
            "naive_composition": _composition_payload(result.naive_composition, context.reports),
            "output": output.as_posix(),
            "parent_patch_ids": list(result.parent_patch_ids),
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

    def operation() -> _CommandResult:
        if contracts_policy is not None:
            return _CommandResult(
                {
                    "policy_hash": sha256_file(contracts_policy),
                    "reason": (
                        "external stack-policy contracts do not yet have a v1 parser; "
                        "bundle-carried contracts can be audited without --contracts"
                    ),
                    "status": "UNSUPPORTED",
                },
                EXIT_UNSUPPORTED,
            )
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
        by_id = {operand.patch_id: operand for operand in context.operands}

        def oracle(subset: tuple[str, ...]) -> SubsetEvaluation:
            if not subset:
                return SubsetEvaluation((), {}, VerificationOutcome.NOT_APPLICABLE)
            result = verify_contract_closure(
                [by_id[item] for item in subset],
                executor=context.execute,
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
            return SubsetEvaluation(
                subset,
                margins,
                outcome,
                violated_contracts=violated,
                metadata={"composition_claim": result.claim.value},
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
            dependencies={
                bundle.manifest.patch_id: bundle.manifest.requires for bundle in context.bundles
            },
        )
        payload = {
            "audit": _jsonable(result),
            "evidence_wording": (
                "all combinations executed"
                if result.search_space_exhausted
                else "no unexecuted combination is described as safe"
            ),
            "status": [claim.value for claim in result.claims],
        }
        output.mkdir(parents=True, exist_ok=False)
        _write_json(output / "audit.json", payload)
        _write_json(
            output / "manifest.json",
            {
                "artifact_hashes": {"audit.json": sha256_file(output / "audit.json")},
                "claims": [claim.value for claim in result.claims],
                "schema_version": 1,
            },
        )
        return _CommandResult({**payload, "output": output.as_posix()})

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
    dense: Mapping[str, Tensor],
    maximum_rank: int,
    maximum_modules: int,
    seed: int,
) -> tuple[Any, Any, Any]:
    from modelpact.compiler.constraints import DifferentiableConstraint, DifferentiableObjective
    from modelpact.compiler.contracts import prepare_contract
    from modelpact.compiler.optimize import OptimizerConfig, compile_low_rank_patch
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
        for identifier, item in sorted(prepared.items()):
            batches = tuple(
                example for objective in item.objectives for example in objective.batches
            )
            evidence_count += len(batches)
            if batches:
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

    return applier, verifier, (teacher_builder, recompiler, reports)


@app.command("rebase")
def rebase_command(
    patch: Path = typer.Argument(..., exists=True),
    from_base: Path = typer.Option(..., "--from-base", exists=True),
    onto: Path = typer.Option(..., "--onto", exists=True),
    target_adapter: str = typer.Option(..., "--target-adapter"),
    output: Path = typer.Option(..., "--output"),
    source_adapter: str | None = typer.Option(None, "--source-adapter"),
    maximum_steps: int = typer.Option(200, "--maximum-steps", min=1, max=10_000_000),
    max_rank: int = typer.Option(16, "--max-rank", min=1, max=4096),
    max_modules: int = typer.Option(12, "--max-modules", min=1, max=100_000),
    seed: int = typer.Option(0, "--seed", min=0),
    device: str = typer.Option("cpu", "--device"),
    dtype: str = typer.Option("float32", "--dtype"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Verify direct transfer, then behaviorally recompile failed tiny-model transfers."""

    from modelpact.codegen import emit_apply_script, emit_verify_script
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
        contracts = _load_bundle_contracts(bundle)
        if len(contracts) != 1:
            return _CommandResult(
                {
                    "contract_count": len(contracts),
                    "reason": (
                        "Behavior Patch Bundle v1 has no executable multi-contract certificate "
                        "aggregation schema; rebase currently requires one deduplicated contract"
                    ),
                    "status": "UNSUPPORTED",
                },
                EXIT_UNSUPPORTED,
            )
        original_contract, original_path = next(iter(contracts.values()))
        if not original_contract.guards:
            return _CommandResult(
                {
                    "reason": "semantic rebase certification requires preservation assertions",
                    "status": "UNSUPPORTED",
                },
                EXIT_UNSUPPORTED,
            )
        contract = _retarget_contract(original_contract, target)
        dense = _bundle_dense_delta(bundle)
        applier, verifier, tiny_components = _tiny_rebase_components(
            source=source,
            target=target,
            bundle=bundle,
            contracts=contracts,
            dense=dense,
            maximum_rank=max_rank,
            maximum_modules=max_modules,
            seed=seed,
        )
        tiny_teacher_builder, tiny_recompiler, _reports = tiny_components
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
                    target_contract_ids=(original_contract.contract_id,),
                    preservation_contract_ids=(f"{original_contract.contract_id}:guards",),
                ),
                source_base=_descriptor(source),
                target_base=_descriptor(target),
                new_base_guard_ids=(),
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

        program, tensors = _dense_program(result.delta, target.manifest.state_schema)
        candidate_model = _model_with_dense_delta(
            target.model,
            result.delta,
            target.manifest.state_schema,
        )
        report = _verification_report(
            loaded=target,
            model=candidate_model,
            base_model=target.model,
            contract=contract,
            contract_path=original_path,
            candidate_id=f"final-rebase:{bundle.manifest.patch_id}",
            include_holdout=True,
        )
        final_outcome = report.outcome
        if contract.holdout.configured and report.holdout_outcome is not VerificationOutcome.PASS:
            final_outcome = report.holdout_outcome
        if final_outcome is not VerificationOutcome.PASS:
            output.mkdir(parents=True, exist_ok=False)
            evidence = {
                "claim": result.claim.value,
                "disposition": result.disposition.value,
                "evidence": result.evidence.to_dict(),
                "reason": "final candidate failed validation or sealed holdout",
                "status": (
                    "HOLDOUT_FAILED"
                    if report.outcome is VerificationOutcome.PASS
                    else report.outcome.value
                ),
                "verification": report.to_dict(),
            }
            _write_json(output / "rebase-evidence.json", evidence)
            return _CommandResult(evidence, _outcome_exit(final_outcome))

        validation, holdout = _report_sections(report)
        contract_bytes = (canonical_dumps(contract.to_dict()) + "\n").encode()
        contract_resources = _contract_resource_artifacts(original_contract, original_path)
        semantic = result.claim is RebaseClaim.SEMANTIC_REBASE_VERIFIED
        compilation_evidence = {
            "direct_transplant": not semantic,
            "optimization_steps": (
                result.recompile.steps_executed if result.recompile is not None else 0
            ),
            "rebase": result.evidence.to_dict(),
            "recompile": _jsonable(result.recompile),
            "schema_version": 1,
        }
        rebased = create_patch_bundle(
            output,
            name=f"{bundle.manifest.name}-rebased",
            base_signature=target.manifest.signature.to_dict(),
            state_schema=target.manifest.state_schema,
            program=program,
            tensors=tensors,
            tool_version=__version__,
            contracts={
                "contracts/preservation.yaml": contract_bytes,
                "contracts/target.yaml": contract_bytes,
                **contract_resources,
            },
            supplemental_artifacts={
                "evidence/compile.json": (canonical_dumps(compilation_evidence) + "\n").encode(),
                "evidence/holdout.json": (canonical_dumps(holdout) + "\n").encode(),
                "evidence/minimization.json": (
                    canonical_dumps({"claim": "UNMINIMIZED", "schema_version": 1}) + "\n"
                ).encode(),
                "evidence/validation.json": (canonical_dumps(validation) + "\n").encode(),
                "probes/hashes.json": (
                    canonical_dumps(dict(sorted(report.probe_hashes.items()))) + "\n"
                ).encode(),
                "probes/manifest.json": (
                    canonical_dumps(
                        {"schema_version": 1, "source_hashes": dict(report.probe_hashes)}
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
            provides=(contract.contract_id,),
            preserves=(f"{contract.contract_id}:guards",),
            rebased_from=bundle.manifest.patch_id,
            compiler_configuration={
                "maximum_modules": max_modules,
                "maximum_rank": max_rank,
                "mode": "semantic_recompile" if semantic else "direct_transplant",
                "optimization_steps": compilation_evidence["optimization_steps"],
                "seed": seed,
            },
        )
        certificate = build_certificate(
            report,
            contract,
            patch_id=rebased.manifest.patch_id,
            checkpoint_hashes=target.manifest.checkpoint_tensor_hashes,
            artifact_hashes=dict(rebased.manifest.artifact_hashes),
            patch_structure={
                "active_targets": sorted(program.targets),
                "patch_bytes": program.estimate_bytes(tensors),
            },
            rebase_result={
                "claim": result.claim.value,
                "source_base_hash": source.manifest.signature.signature_hash,
                "source_patch_id": bundle.manifest.patch_id,
                "target_base_hash": target.manifest.signature.signature_hash,
                "evidence": result.evidence.to_dict(),
            },
        )
        codegen_root = Path(tempfile.mkdtemp(prefix="modelpact-codegen-"))
        try:
            apply_path = emit_apply_script(output, codegen_root / "apply_patch.py")
            verify_path = emit_verify_script(output, codegen_root / "verify_patch.py")
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
                "disposition": result.disposition.value,
                "output": output.as_posix(),
                "optimization_steps": compilation_evidence["optimization_steps"],
                "patch_id": rebased.manifest.patch_id,
                "rebased_from": bundle.manifest.patch_id,
                "status": "PASS",
                "verification": report.to_dict(),
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
            "rebase_cross_architecture, or cegis"
        ),
    ),
    output: Path | None = typer.Option(None, "--output"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Run one deterministic, machine-readable ModelPactBench experiment."""

    from modelpact.modelpactbench.runner import run_selected

    def operation() -> _CommandResult:
        result = run_selected(name)
        payload = {
            "benchmark": name,
            "result": result,
            "result_hash": hash_canonical(result),
            "status": "PASS",
        }
        if output is not None:
            _write_json(output, result)
            payload["output"] = output.as_posix()
        return _CommandResult(payload)

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
    supplied_patch_ids = {bundle.manifest.patch_id for bundle in context.bundles}
    missing_dependencies = {
        bundle.manifest.patch_id: tuple(sorted(set(bundle.manifest.requires) - supplied_patch_ids))
        for bundle in context.bundles
        if set(bundle.manifest.requires) - supplied_patch_ids
    }
    if missing_dependencies:
        raise ValueError(
            f"patch stack omits declared dependencies: {canonical_dumps(missing_dependencies)}"
        )
    result = verify_contract_closure(
        context.operands,
        executor=context.execute,
        aliases=_alias_map(context.loaded.manifest.state_schema),
        contradiction_checker=_static_checker(context),
    )
    output.mkdir(parents=True, exist_ok=False)
    resolved_hash: str | None = None
    if result.closed:
        resolved_path = output / "resolved-delta.safetensors"
        save_file(
            {
                key: value.detach().cpu().contiguous()
                for key, value in sorted(result.resolved_delta.items())
            },
            str(resolved_path),
        )
        resolved_hash = sha256_file(resolved_path)
        kind = StackResolutionKind.NAIVE_ADDITIVE_STACK
        exit_code = 0
    elif result.claim.value == "STATIC_CONTRACT_CONTRADICTION":
        kind = StackResolutionKind.STATIC_CONTRADICTION
        exit_code = EXIT_FAILED
    elif repair_conflicts:
        kind = StackResolutionKind.UNSUPPORTED
        exit_code = EXIT_UNSUPPORTED
    else:
        kind = StackResolutionKind.EMPIRICAL_FAILURE
        exit_code = EXIT_FAILED
    union_contract_hash = hash_canonical(list(result.contract_ids))
    execution = StackResolutionExecution(
        kind=kind,
        resolved_artifact_hash=resolved_hash,
        verification_policy_hash=verification_policy_hash,
        union_contract_hash=union_contract_hash,
        warnings=(
            (
                "repair was requested, but no concrete semantic-merge compiler backend "
                "is selected by the CLI"
            ),
        )
        if kind is StackResolutionKind.UNSUPPORTED
        else (),
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
            }
        },
    }
    _write_json(output / "stack.lock.json", lock_value)
    evidence = {
        "composition": _composition_payload(result, context.reports),
        "lock_hash": sha256_file(output / "stack.lock.json"),
        "resolution": kind.value,
        "schema_version": 1,
    }
    _write_json(output / "resolution.json", evidence)
    return {
        "lock": lock_value,
        "lock_hash": evidence["lock_hash"],
        "output": output.as_posix(),
        "resolution": kind.value,
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


def _read_lock(path: Path) -> dict[str, object]:
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("stack lockfile exceeds size limit")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("malformed Patch Stack Lockfile v1")
    required = {"base_hash", "patch_hashes", "verification_policy_hash", "extensions"}
    if not required <= set(value):
        raise ValueError(f"lockfile is missing fields: {sorted(required - set(value))}")
    patch_hashes = value.get("patch_hashes")
    extensions = value.get("extensions")
    if not isinstance(patch_hashes, dict) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in patch_hashes.items()
    ):
        raise ValueError("lockfile patch_hashes must be a string mapping")
    if not isinstance(extensions, dict):
        raise ValueError("lockfile extensions must be an object")
    return cast(dict[str, object], value)


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
        value = _read_lock(lockfile)
        patch_hashes = cast(dict[str, str], value["patch_hashes"])
        if remove not in patch_hashes:
            raise ValueError(f"patch is not present in the locked stack: {remove}")
        extensions = cast(dict[str, object], value["extensions"])
        cli_extension = extensions.get("modelpact_cli")
        if not isinstance(cli_extension, dict):
            return _CommandResult(
                {
                    "reason": "lockfile does not carry local path resolution metadata",
                    "reversion_grade": "REVERT_FAILED",
                    "status": "UNSUPPORTED",
                },
                EXIT_UNSUPPORTED,
            )
        base_value = cli_extension.get("base_path")
        path_values = cli_extension.get("patch_paths")
        if (
            not isinstance(base_value, str)
            or not isinstance(path_values, dict)
            or not all(
                isinstance(key, str) and isinstance(item, str) for key, item in path_values.items()
            )
        ):
            raise ValueError("malformed ModelPact CLI lock extension")
        base_checkpoint = Path(base_value)
        remaining_ids = tuple(sorted(set(patch_hashes) - {remove}))
        loaded = _load_model(adapter_spec, base_checkpoint, device=device, dtype=dtype)
        if loaded.manifest.signature.checkpoint_hash != value.get("base_hash"):
            raise ValueError("base checkpoint hash no longer matches the lockfile")
        for patch_id, expected_manifest_hash in sorted(patch_hashes.items()):
            path_value = path_values.get(patch_id)
            if not isinstance(path_value, str):
                raise ValueError(f"lockfile has no path for patch {patch_id}")
            actual = sha256_file(Path(path_value) / "manifest.json")
            if actual != expected_manifest_hash:
                raise ValueError(f"locked patch manifest changed: {patch_id}")
        if not remaining_ids:
            output.mkdir(parents=True, exist_ok=False)
            lock_value = {
                "audit_hash": None,
                "base_hash": value["base_hash"],
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
                "verification_policy_hash": value["verification_policy_hash"],
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
        remaining_paths = tuple(Path(cast(str, path_values[item])) for item in remaining_ids)
        policy_hash = cast(str, value["verification_policy_hash"])
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
                "reversion_grade": ("RUNTIME_UNMOUNT_EXACT" if exit_code == 0 else "REVERT_FAILED"),
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
