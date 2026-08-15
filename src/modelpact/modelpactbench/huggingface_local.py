"""Offline Hugging Face integration benchmark with real learned patches.

The benchmark creates a tiny GPT-NeoX causal LM and a WordLevel tokenizer from
local Python objects, writes only SafeTensors checkpoints, reloads every model
through :class:`HuggingFaceCausalLMAdapter`, and then executes extraction,
verification, additive composition, and direct-first semantic rebase.  It never
uses a hub identifier or a network-backed ``from_pretrained`` call.
"""

from __future__ import annotations

import copy
import importlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn

from modelpact import __version__
from modelpact.adapters.base import GenerationPolicy as AdapterGenerationPolicy
from modelpact.adapters.huggingface import HuggingFaceCausalLMAdapter
from modelpact.codegen.apply import emit_apply_script
from modelpact.codegen.verify import emit_verify_script
from modelpact.compiler.extract import apply_dense_deltas, extract_behavior_cluster
from modelpact.compiler.optimize import OptimizerConfig
from modelpact.compiler.package import compilation_delta_program, compile_evidence
from modelpact.compiler.result import CompilationResult
from modelpact.compose.closure import (
    ContractMargin,
    MarginKind,
    PatchOperand,
    verify_contract_closure,
)
from modelpact.compose.closure import (
    VerificationReport as ClosureVerificationReport,
)
from modelpact.contracts import (
    AssertionType,
    BehaviorContract,
    CompileObjective,
    GenerationMode,
    GenerationPolicy,
    HoldoutPhase,
    HoldoutPolicy,
    ModelRequirements,
    ObjectiveType,
    SealedHoldoutGate,
    StatisticsPolicy,
    VerificationAssertion,
    canonical_contract_json,
)
from modelpact.diff.metrics import symmetric_kl
from modelpact.diff.witnesses import DifferenceWitness
from modelpact.models.manifest import ModelManifest, build_model_manifest
from modelpact.patch.bundle import PatchBundle, attach_bundle_artifacts, create_patch_bundle
from modelpact.patch.mount import mount_patch
from modelpact.rebase.compile import (
    BehavioralRecompileRequest,
    BehavioralRecompileResult,
    RebaseBudget,
    RebaseRequest,
    TeacherContext,
    semantic_rebase,
)
from modelpact.rebase.direct import (
    BaseModelDescriptor,
    RebasePatch,
    RebaseVerification,
)
from modelpact.status import VerificationOutcome
from modelpact.util.atomic import atomic_write_bytes, atomic_write_text
from modelpact.util.canonical_json import canonical_dumps
from modelpact.util.hashing import hash_canonical, sha256_bytes
from modelpact.verify.certificate import build_certificate, validate_certificate
from modelpact.verify.engine import ExecutionIdentity, VerificationReport, verify_contract
from modelpact.verify.provider import ModelBackedRecordProvider


@dataclass(frozen=True, slots=True)
class HuggingFaceLocalConfig:
    """Bounded deterministic resource configuration for the generated fixture."""

    base_steps: int = 180
    teacher_steps: int = 100
    compiler_steps: int = 180
    rebase_steps: int = 180
    hidden_size: int = 16
    intermediate_size: int = 32
    seed: int = 90210

    def __post_init__(self) -> None:
        values = (
            self.base_steps,
            self.teacher_steps,
            self.compiler_steps,
            self.rebase_steps,
            self.hidden_size,
            self.intermediate_size,
        )
        if any(isinstance(value, bool) or value <= 0 for value in values):
            raise ValueError("Hugging Face benchmark budgets must be positive integers")


DEFAULT_HUGGINGFACE_LOCAL_CONFIG = HuggingFaceLocalConfig()

_VOCABULARY = {
    "<pad>": 0,
    "<bos>": 1,
    "<eos>": 2,
    "<unk>": 3,
    "fact_a": 4,
    "fact_b": 5,
    "control": 6,
    "R": 7,
    "B": 8,
    "C": 9,
    "X": 10,
    "Y": 11,
}
_BASE_BEHAVIOR = {"fact_a": "R", "fact_b": "B", "control": "C"}
_TEACHER_A_BEHAVIOR = {**_BASE_BEHAVIOR, "fact_a": "X"}
_TEACHER_B_BEHAVIOR = {**_BASE_BEHAVIOR, "fact_b": "Y"}
_BASE_V2_BEHAVIOR = {**_BASE_BEHAVIOR, "control": "Y"}


def huggingface_dependencies_available() -> bool:
    """Return whether the optional, fully local fixture dependencies import."""

    try:
        __import__("transformers")
        __import__("tokenizers")
    except ImportError:
        return False
    return True


def _require_huggingface() -> tuple[Any, Any, Any, Any]:
    try:
        from tokenizers import Tokenizer  # type: ignore[import-untyped]
        from tokenizers.models import WordLevel  # type: ignore[import-untyped]
        from tokenizers.pre_tokenizers import WhitespaceSplit  # type: ignore[import-untyped]
        from transformers import GPTNeoXConfig, GPTNeoXForCausalLM, PreTrainedTokenizerFast
    except ImportError as error:  # pragma: no cover - exercised by optional-dependency callers
        raise RuntimeError(
            "Benchmark G requires the optional 'huggingface' dependencies"
        ) from error
    return (
        (Tokenizer, WordLevel, WhitespaceSplit),
        GPTNeoXConfig,
        GPTNeoXForCausalLM,
        PreTrainedTokenizerFast,
    )


def _tokenizer() -> Any:
    tokenizers, _, _, tokenizer_class = _require_huggingface()
    tokenizer_type, word_level_type, whitespace_type = tokenizers
    backend = tokenizer_type(word_level_type(_VOCABULARY, unk_token="<unk>"))  # noqa: S106
    backend.pre_tokenizer = whitespace_type()
    tokenizer = tokenizer_class(
        tokenizer_object=backend,
        bos_token="<bos>",  # noqa: S106
        eos_token="<eos>",  # noqa: S106
        unk_token="<unk>",  # noqa: S106
        pad_token="<pad>",  # noqa: S106
        model_max_length=16,
    )
    tokenizer.padding_side = "left"
    return tokenizer


def _new_model(config: HuggingFaceLocalConfig) -> Any:
    _, configuration_type, model_type, _ = _require_huggingface()
    torch.manual_seed(config.seed)
    configuration = configuration_type(
        vocab_size=len(_VOCABULARY),
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=1,
        num_attention_heads=4,
        max_position_embeddings=16,
        rotary_pct=0.25,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        bos_token_id=_VOCABULARY["<bos>"],
        eos_token_id=_VOCABULARY["<eos>"],
        pad_token_id=_VOCABULARY["<pad>"],
        tie_word_embeddings=False,
        use_cache=False,
    )
    return model_type(configuration)


def _train_mapping(
    model: Any,
    behavior: Mapping[str, str],
    *,
    steps: int,
    learning_rate: float,
    seed: int,
) -> tuple[float, ...]:
    """Train exact one-token continuations using real causal-LM logits."""

    torch.manual_seed(seed)
    model.train()
    prompts = tuple(sorted(behavior))
    input_ids = torch.tensor(
        [[_VOCABULARY["<bos>"], _VOCABULARY[prompt]] for prompt in prompts],
        dtype=torch.long,
    )
    attention_mask = torch.ones_like(input_ids)
    targets = torch.tensor([_VOCABULARY[behavior[prompt]] for prompt in prompts])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    losses: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits: Tensor = output.logits[:, -1, :]
        loss: Tensor = torch.nn.functional.cross_entropy(logits, targets)
        loss.backward()  # type: ignore[no-untyped-call]  # PyTorch stub gap.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    return tuple(losses)


def _save_checkpoint(model: Any, tokenizer: Any, path: Path) -> Path:
    if path.exists():
        raise FileExistsError(path)
    path.mkdir(parents=True)
    model.save_pretrained(path, safe_serialization=True, max_shard_size="64MB")
    tokenizer.save_pretrained(path)
    if list(path.glob("*.bin")):
        raise RuntimeError("generated Hugging Face fixture unexpectedly wrote pickle weights")
    tensor_files = sorted(path.glob("*.safetensors"))
    if not tensor_files:
        raise RuntimeError("generated Hugging Face fixture contains no SafeTensors weights")
    return path


def _load_local(path: Path) -> tuple[HuggingFaceCausalLMAdapter, nn.Module]:
    # These environment settings are defense in depth around local_files_only.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    adapter = HuggingFaceCausalLMAdapter()
    model = adapter.load(str(path), device="cpu", dtype=torch.float32)
    adapter.prepare(model)
    return adapter, model


def _generate(adapter: HuggingFaceCausalLMAdapter, model: nn.Module, prompt: str) -> str:
    sample = adapter.generate(
        model,
        adapter.tokenizer().batch((prompt,)),
        AdapterGenerationPolicy(mode="greedy", max_new_tokens=1, seed=0),
    )[0]
    return sample.text.strip()


def _outputs(
    adapter: HuggingFaceCausalLMAdapter,
    model: nn.Module,
    prompts: Sequence[str] = ("fact_a", "fact_b", "control"),
) -> dict[str, str]:
    return {prompt: _generate(adapter, model, prompt) for prompt in prompts}


def _witness(
    adapter: HuggingFaceCausalLMAdapter,
    base: nn.Module,
    target: nn.Module,
    prompt: str,
    *,
    role: str,
) -> DifferenceWitness:
    batch = adapter.tokenizer().batch((prompt,))
    with torch.no_grad():
        base_logits = adapter.forward_logits(base, batch)[:, -1].detach().cpu()
        target_logits = adapter.forward_logits(target, batch)[:, -1].detach().cpu()
    divergence = float(symmetric_kl(base_logits, target_logits).item())
    base_output = _generate(adapter, base, prompt)
    target_output = _generate(adapter, target, prompt)
    return DifferenceWitness.create(
        original_input=prompt,
        minimized_input=prompt,
        divergence_metrics={
            "symmetric_kl": divergence,
            "generation_changed": float(base_output != target_output),
        },
        base_output={"text": base_output},
        target_output={"text": target_output},
        provenance={"benchmark": "HuggingFaceLocal", "role": role},
    )


def _compile_patch(
    adapter: HuggingFaceCausalLMAdapter,
    base: nn.Module,
    teacher: nn.Module,
    prompt: str,
    guards: tuple[str, ...],
    config: HuggingFaceLocalConfig,
    *,
    seed: int,
) -> CompilationResult:
    selected = (_witness(adapter, base, teacher, prompt, role="selected"),)
    evidence = extract_behavior_cluster(
        adapter,
        base,
        teacher,
        selected,
        (),
        additional_guards=guards,
        optimizer_config=OptimizerConfig(
            maximum_rank=2,
            maximum_modules=2,
            steps=config.compiler_steps,
            learning_rate=0.03,
            patience=config.compiler_steps,
            complexity_weight=1e-8,
            seed=seed,
        ),
        maximum_selected_kl=0.05,
        maximum_nonselected_base_kl=0.08,
    )
    if not evidence.compiler_result.feasible or not evidence.validation_passed:
        raise RuntimeError(
            f"real Hugging Face extraction failed for {prompt}: {evidence.compiler_result.warnings}"
        )
    return evidence.compiler_result


def _jsonl(rows: Sequence[Mapping[str, object]]) -> bytes:
    return ("".join(canonical_dumps(dict(row)) + "\n" for row in rows)).encode()


def _reference_logits(
    adapter: HuggingFaceCausalLMAdapter,
    model: nn.Module,
    prompt: str,
) -> list[list[float]]:
    batch = adapter.tokenizer().batch((prompt,), add_bos=True)
    with torch.no_grad():
        logits = adapter.forward_logits(model, batch)
    length = int(batch.attention_mask[0].sum().item())
    return logits[0, :length].detach().to(torch.float32).cpu().tolist()


def _probe_payloads(
    *,
    adapter: HuggingFaceCausalLMAdapter,
    teacher: nn.Module,
    prompt: str,
    expected: str,
    guard_expected: Mapping[str, str],
) -> dict[str, bytes]:
    guards = tuple(sorted(guard_expected))
    return {
        "probes/compile-targets.jsonl": _jsonl(({"id": "compile-target", "prompt": prompt},)),
        "probes/validation-targets.jsonl": _jsonl(
            (
                {
                    "id": "validation-target",
                    "prompt": prompt,
                    "expected": expected,
                    "reference_logits": _reference_logits(adapter, teacher, prompt),
                },
            )
        ),
        "probes/validation-guards.jsonl": _jsonl(
            tuple(
                {
                    "id": f"validation-guard-{index}",
                    "prompt": guard,
                    "expected": guard_expected[guard],
                }
                for index, guard in enumerate(guards)
            )
        ),
        # The holdout records are separately gated and never passed to extraction.
        "probes/holdout-targets.jsonl": _jsonl(
            (
                {
                    "id": "holdout-target",
                    "prompt": f"{prompt} {prompt}",
                    "expected": expected,
                    "reference_logits": _reference_logits(adapter, teacher, f"{prompt} {prompt}"),
                },
            )
        ),
        "probes/holdout-guards.jsonl": _jsonl(
            (
                {
                    "id": "holdout-guard",
                    "prompt": f"{guards[-1]} {guards[-1]}",
                    "expected": guard_expected[guards[-1]],
                },
            )
        ),
    }


def _contract(
    manifest: ModelManifest,
    *,
    identifier: str,
) -> BehaviorContract:
    requirements = ModelRequirements(
        tokenizer_hash=manifest.signature.tokenizer_hash,
        base_signature=manifest.signature.signature_hash,
        architecture_hash=manifest.signature.architecture_hash,
        state_schema_hash=manifest.signature.state_schema_hash,
        adapter_id=manifest.signature.adapter_id,
    )
    return BehaviorContract(
        schema_version=1,
        id=identifier,
        contract_version=1,
        model_requirements=requirements,
        objectives=(
            CompileObjective(
                f"{identifier}-teacher-kl",
                ObjectiveType.TEACHER_KL,
                "probes/compile-targets.jsonl",
            ),
        ),
        targets=(
            VerificationAssertion(
                f"{identifier}-generation",
                AssertionType.FREE_GENERATION_MATCH,
                "probes/validation-targets.jsonl",
                {"minimum_pass_rate": 1.0},
            ),
            VerificationAssertion(
                f"{identifier}-reference-kl",
                AssertionType.REFERENCE_KL,
                "probes/validation-targets.jsonl",
                {"maximum_mean": 4.0, "maximum_item": 6.0},
            ),
        ),
        guards=(
            VerificationAssertion(
                f"{identifier}-guard-generation",
                AssertionType.FREE_GENERATION_MATCH,
                "probes/validation-guards.jsonl",
                {"minimum_pass_rate": 1.0},
            ),
            VerificationAssertion(
                f"{identifier}-guard-kl",
                AssertionType.BASE_KL,
                "probes/validation-guards.jsonl",
                # The provider evaluates every prompt position, including the
                # BOS-to-input-token distribution; exact deployed continuation
                # preservation remains the hard guard in the preceding rule.
                {"maximum_mean": 8.0, "maximum_item": 10.0},
            ),
        ),
        holdout=HoldoutPolicy(
            sealed=True,
            targets="probes/holdout-targets.jsonl",
            guards="probes/holdout-guards.jsonl",
        ),
        statistics=StatisticsPolicy(
            confidence_level=0.95,
            bootstrap_samples=64,
            bootstrap_seed=71237,
        ),
        generation=GenerationPolicy(
            mode=GenerationMode.GREEDY,
            max_new_tokens=1,
            seeds=(0,),
        ),
        description=(
            "Finite offline GPT-NeoX behavior contract; no claim extends beyond "
            "the declared local probes."
        ),
    )


def _preservation_contract(contract: BehaviorContract) -> BehaviorContract:
    return BehaviorContract(
        schema_version=1,
        id=f"{contract.id}-preservation",
        contract_version=1,
        model_requirements=contract.model_requirements,
        objectives=(),
        targets=(),
        guards=contract.guards,
        holdout=HoldoutPolicy(sealed=True),
        statistics=contract.statistics,
        generation=contract.generation,
        description="Preservation projection of the executed finite contract.",
    )


def _write_probe_workspace(root: Path, probes: Mapping[str, bytes]) -> Path:
    workspace = root / "verification-inputs"
    for relative, content in sorted(probes.items()):
        atomic_write_bytes(workspace / relative, content, overwrite=False)
    return workspace


def _verify(
    adapter: HuggingFaceCausalLMAdapter,
    base: nn.Module,
    teacher: nn.Module,
    patched: nn.Module,
    contract: BehaviorContract,
    manifest: ModelManifest,
    workspace: Path,
    *,
    candidate_id: str,
) -> tuple[VerificationReport, SealedHoldoutGate]:
    provider = ModelBackedRecordProvider(
        adapter=adapter,
        model=patched,
        base_model=base,
        reference_model=teacher,
        contract_root=workspace,
        generation_policy=contract.generation,
    )
    identity = ExecutionIdentity(
        adapter_id=manifest.signature.adapter_id,
        base_signature=manifest.signature.signature_hash,
        tokenizer_hash=manifest.signature.tokenizer_hash,
        architecture_hash=manifest.signature.architecture_hash,
        state_schema_hash=manifest.signature.state_schema_hash,
    )
    gate = SealedHoldoutGate(contract)
    gate.select_final_candidate(candidate_id)
    capability = gate.authorize(
        phase=HoldoutPhase.FINAL_CANDIDATE,
        candidate_id=candidate_id,
    )
    report = verify_contract(
        contract,
        identity=identity,
        provider=provider,
        include_holdout=True,
        holdout_gate=gate,
        holdout_capability=capability,
    )
    return report, gate


def _bundle_report(
    name: str,
    report: VerificationReport,
    *,
    output: Mapping[str, str],
) -> bytes:
    return (
        f"# {name}\n\n"
        "Verified under the declared contracts, probe spaces, generation policy, "
        "environment, and search budget.\n\n"
        f"- Validation: `{report.outcome.value}`\n"
        f"- Sealed holdout: `{report.holdout_outcome.value}`\n"
        f"- Executed outputs: `{canonical_dumps(dict(sorted(output.items())))}`\n"
        "- Scope: finite locally generated GPT-NeoX probes only.\n"
    ).encode()


def _complete_bundle(
    root: Path,
    *,
    name: str,
    result: CompilationResult,
    manifest: ModelManifest,
    contract: BehaviorContract,
    probes: Mapping[str, bytes],
    base: nn.Module,
    teacher: nn.Module,
    adapter: HuggingFaceCausalLMAdapter,
    checkpoint_hashes: Mapping[str, str],
    compiler_configuration: Mapping[str, object],
    rebased_from: str | None = None,
) -> tuple[PatchBundle, VerificationReport, dict[str, str], bool]:
    program, tensors = compilation_delta_program(result, manifest.state_schema)
    preservation = _preservation_contract(contract)
    policy = {
        "statistics": contract.statistics.to_dict(),
        "generation": contract.generation.to_dict(),
    }
    initial = create_patch_bundle(
        root,
        name=name,
        base_signature=manifest.signature.to_dict(),
        state_schema=manifest.state_schema,
        program=program,
        tensors=tensors,
        tool_version=__version__,
        contracts={
            "contracts/target.yaml": (canonical_contract_json(contract) + "\n").encode(),
            "contracts/preservation.yaml": (canonical_contract_json(preservation) + "\n").encode(),
            **{f"contracts/{relative}": content for relative, content in sorted(probes.items())},
        },
        provides=(contract.contract_id,),
        preserves=tuple(sorted({contract.contract_id, preservation.contract_id})),
        verification_policy_hash=hash_canonical(policy),
        compiler_configuration=compiler_configuration,
        rebased_from=rebased_from,
    )
    workspace = _write_probe_workspace(root.parent / f".{root.name}-work", probes)
    runtime = copy.deepcopy(base)
    before = {name: value.detach().clone() for name, value in runtime.state_dict().items()}
    session = mount_patch(
        runtime,
        initial.program,
        initial.tensors,
        state_schema=manifest.state_schema,
    )
    try:
        report, gate = _verify(
            adapter,
            base,
            teacher,
            runtime,
            contract,
            manifest,
            workspace,
            candidate_id=initial.manifest.patch_id,
        )
        output = _outputs(adapter, runtime)
    finally:
        session.unmount()
    unmount_exact = all(
        torch.equal(value, before[name]) for name, value in runtime.state_dict().items()
    )
    if report.outcome is not VerificationOutcome.PASS:
        raise RuntimeError(f"{name} failed validation: {report.prompt_failures}")
    if report.holdout_outcome is not VerificationOutcome.PASS:
        raise RuntimeError(f"{name} failed sealed holdout")
    probe_hashes = {key: sha256_bytes(value) for key, value in sorted(probes.items())}
    certificate = build_certificate(
        report,
        contract,
        patch_id=initial.manifest.patch_id,
        checkpoint_hashes=checkpoint_hashes,
        artifact_hashes=initial.manifest.artifact_hashes,
        verification_policy=policy,
        counterexample_search={
            "outcome": "NO_FAILURE_FOUND_WITHIN_BUDGET",
            "executions": 3,
            "scope": "fixed neighboring-token guards",
        },
        patch_structure={
            "active_modules": list(result.active_modules),
            "module_ranks": dict(sorted(result.ranks.items())),
            "factor_parameters": sum(
                left.numel() + right.numel() for left, right in result.factors.values()
            ),
            "tensor_bytes": sum(value.numel() * value.element_size() for value in tensors.values()),
        },
        minimization_result={"outcome": "UNMINIMIZED"},
        objectives_optimized=True,
        minimized_within_budget=False,
        additional_warnings=(
            "Generated-fixture result covers finite local probes and one deterministic seed.",
        ),
    )
    probe_manifest = {
        "schema_version": 1,
        "roles": {
            "compile": ["probes/compile-targets.jsonl"],
            "validation": [
                "probes/validation-targets.jsonl",
                "probes/validation-guards.jsonl",
            ],
            "sealed_holdout": [
                "probes/holdout-targets.jsonl",
                "probes/holdout-guards.jsonl",
            ],
        },
        "contract_hash": contract.contract_id,
    }
    enriched = attach_bundle_artifacts(
        initial.path,
        {
            **probes,
            "probes/manifest.json": (canonical_dumps(probe_manifest) + "\n").encode(),
            "probes/hashes.json": (canonical_dumps(probe_hashes) + "\n").encode(),
            "evidence/compile.json": (
                canonical_dumps(
                    {
                        **compile_evidence(result),
                        "mode": "behavior_cluster_extraction",
                        "selected_probe_hash": probe_hashes["probes/compile-targets.jsonl"],
                        "guard_probe_hash": probe_hashes["probes/validation-guards.jsonl"],
                        "checkpoint_hashes": dict(sorted(checkpoint_hashes.items())),
                    }
                )
                + "\n"
            ).encode(),
            "evidence/validation.json": (canonical_dumps(report.to_dict()) + "\n").encode(),
            "evidence/holdout.json": (
                canonical_dumps(
                    {
                        "schema_version": 1,
                        "outcome": report.holdout_outcome.value,
                        "access_records": [
                            {
                                "sequence": record.sequence,
                                "contract_hash": record.contract_hash,
                                "candidate_id": record.candidate_id,
                                "phase": record.phase.value,
                                "role": record.role.value,
                                "source": record.source,
                            }
                            for record in gate.access_records
                        ],
                        "targets": [item.to_dict() for item in report.holdout_target_results],
                        "guards": [item.to_dict() for item in report.holdout_guard_results],
                    }
                )
                + "\n"
            ).encode(),
            "evidence/minimization.json": (
                canonical_dumps({"schema_version": 1, "outcome": "UNMINIMIZED"}) + "\n"
            ).encode(),
            "certificate.json": (certificate.canonical_json() + "\n").encode(),
            "report.md": _bundle_report(name, report, output=output),
        },
        state_schema=manifest.state_schema,
    )
    with tempfile.TemporaryDirectory(prefix="modelpact-hf-codegen-", dir=root.parent) as temp:
        temporary = Path(temp)
        apply_script = emit_apply_script(
            enriched.path,
            temporary / "apply_patch.py",
            will_live_in_bundle=True,
        ).read_bytes()
        verify_script = emit_verify_script(
            enriched.path,
            temporary / "verify_patch.py",
            will_live_in_bundle=True,
        ).read_bytes()
    completed = attach_bundle_artifacts(
        enriched.path,
        {"apply_patch.py": apply_script, "verify_patch.py": verify_script},
        state_schema=manifest.state_schema,
        require_complete=True,
    )
    validate_certificate(certificate, artifact_root=completed.path)
    return completed, report, output, unmount_exact


def _generation_margin(
    adapter: HuggingFaceCausalLMAdapter,
    model: nn.Module,
    prompt: str,
    expected: str,
) -> float:
    batch = adapter.tokenizer().batch((prompt,))
    expected_id = _VOCABULARY[expected]
    with torch.no_grad():
        logits = adapter.forward_logits(model, batch)[0, -1].float()
    selected = logits[expected_id]
    other = logits.clone()
    other[expected_id] = torch.finfo(other.dtype).min
    return float((selected - other.max()).detach())


def _composition(
    adapter: HuggingFaceCausalLMAdapter,
    base: nn.Module,
    manifest: ModelManifest,
    patch_a: PatchBundle,
    result_a: CompilationResult,
    patch_b: PatchBundle,
    result_b: CompilationResult,
) -> tuple[dict[str, object], dict[str, Tensor]]:
    target_a = "huggingface-local-fact-a"
    target_b = "huggingface-local-fact-b"
    preservation = "huggingface-local-controls"
    operands = (
        PatchOperand(
            patch_id=patch_a.manifest.patch_id,
            base_signature=manifest.signature.signature_hash,
            module_schema_hash=manifest.state_schema.schema_hash,
            delta=result_a.deltas,
            contract_ids=(target_a, preservation),
            verified_margins={target_a: 1.0, preservation: 1.0},
        ),
        PatchOperand(
            patch_id=patch_b.manifest.patch_id,
            base_signature=manifest.signature.signature_hash,
            module_schema_hash=manifest.state_schema.schema_hash,
            delta=result_b.deltas,
            contract_ids=(target_b, preservation),
            verified_margins={target_b: 1.0, preservation: 1.0},
        ),
    )

    observed_outputs: dict[str, str] = {}

    def execute(
        delta: Mapping[str, Tensor], contract_ids: tuple[str, ...]
    ) -> ClosureVerificationReport:
        candidate = apply_dense_deltas(base, dict(delta))
        observed_outputs.update(_outputs(adapter, candidate))
        specifications = {
            target_a: ("fact_a", "X", MarginKind.TARGET),
            target_b: ("fact_b", "Y", MarginKind.TARGET),
            preservation: ("control", "C", MarginKind.GUARD),
        }
        margins = tuple(
            ContractMargin(
                contract_id,
                specifications[contract_id][2],
                _generation_margin(
                    adapter,
                    candidate,
                    specifications[contract_id][0],
                    specifications[contract_id][1],
                ),
                {"executed_output": observed_outputs[specifications[contract_id][0]]},
            )
            for contract_id in contract_ids
        )
        outcome = (
            VerificationOutcome.PASS
            if all(margin.passed for margin in margins)
            else VerificationOutcome.FAIL
        )
        return ClosureVerificationReport(outcome, margins)

    closure = verify_contract_closure(operands, executor=execute)
    payload: dict[str, object] = {
        "schema_version": 1,
        "claim": closure.claim.value,
        "patch_ids": list(closure.patch_ids),
        "contract_ids": list(closure.contract_ids),
        "outputs": dict(sorted(observed_outputs.items())),
        "margins": (
            [
                {
                    "contract_id": item.contract_id,
                    "kind": item.kind.value,
                    "margin": item.margin,
                    "passed": item.passed,
                }
                for item in closure.verification.margins
            ]
            if closure.verification is not None
            else []
        ),
        "executed_contract_closure": closure.verification is not None,
    }
    return payload, dict(closure.resolved_delta)


def _descriptor(manifest: ModelManifest, family_id: str) -> BaseModelDescriptor:
    return BaseModelDescriptor(
        signature=manifest.signature.signature_hash,
        architecture_id=manifest.signature.architecture_hash,
        module_schema_hash=manifest.signature.state_schema_hash,
        tokenizer_hash=manifest.signature.tokenizer_hash,
        output_semantics="causal_lm",
        module_shapes={
            tensor.name: tensor.shape
            for tensor in manifest.state_schema.tensors
            if tensor.patchable
        },
        family_id=family_id,
    )


def _rebase(
    adapter: HuggingFaceCausalLMAdapter,
    source_base: nn.Module,
    source_manifest: ModelManifest,
    target_base: nn.Module,
    target_manifest: ModelManifest,
    source_patch: PatchBundle,
    source_result: CompilationResult,
    config: HuggingFaceLocalConfig,
) -> tuple[dict[str, object], CompilationResult]:
    source_descriptor = _descriptor(source_manifest, "generated-gpt-neox")
    target_descriptor = _descriptor(target_manifest, "generated-gpt-neox")
    target_id = "huggingface-local-fact-a"
    preservation_id = "huggingface-local-fact-a-preservation"
    new_guard_id = "huggingface-local-v2-controls"
    source_patched_teacher = apply_dense_deltas(source_base, source_result.deltas)

    def apply(delta: Mapping[str, Tensor], target: BaseModelDescriptor) -> object:
        del target
        return apply_dense_deltas(target_base, dict(delta))

    def verify(
        candidate: object,
        target_contract_ids: tuple[str, ...],
        guard_contract_ids: tuple[str, ...],
    ) -> RebaseVerification:
        if not isinstance(candidate, nn.Module):
            raise TypeError("rebase candidate is not a module")
        target_margin = _generation_margin(adapter, candidate, "fact_a", "X")
        guard_margin = min(
            _generation_margin(adapter, candidate, "fact_b", "B"),
            _generation_margin(adapter, candidate, "control", "Y"),
        )
        target_margins = dict.fromkeys(target_contract_ids, target_margin)
        guard_margins = dict.fromkeys(guard_contract_ids, guard_margin)
        outcome = (
            VerificationOutcome.PASS
            if target_margin >= 0.0 and guard_margin >= 0.0
            else VerificationOutcome.FAIL
        )
        failures = tuple(
            prompt
            for prompt, value in (("fact_a", target_margin), ("new-base-controls", guard_margin))
            if value < 0.0
        )
        return RebaseVerification(outcome, target_margins, guard_margins, failures)

    semantic_compilations: list[CompilationResult] = []

    def teacher_builder(request: RebaseRequest) -> TeacherContext:
        del request
        return TeacherContext(
            old_patched_teacher=source_patched_teacher,
            new_unpatched_teacher=target_base,
            old_behavior_margins={target_id: 1.0},
            evidence_count=3,
        )

    def recompile(request: BehavioralRecompileRequest) -> BehavioralRecompileResult:
        selected = (
            _witness(adapter, target_base, source_patched_teacher, "fact_a", role="rebase"),
        )
        evidence = extract_behavior_cluster(
            adapter,
            target_base,
            source_patched_teacher,
            selected,
            (),
            additional_guards=("fact_b", "control"),
            optimizer_config=OptimizerConfig(
                maximum_rank=2,
                maximum_modules=2,
                steps=config.rebase_steps,
                learning_rate=0.03,
                patience=config.rebase_steps,
                complexity_weight=1e-8,
                seed=config.seed + 19,
            ),
            maximum_selected_kl=0.08,
            maximum_nonselected_base_kl=0.08,
        )
        candidate = evidence.compiler_result
        semantic_compilations.append(candidate)
        return BehavioralRecompileResult(
            candidate_delta=candidate.deltas if candidate.feasible else None,
            optimization_succeeded=candidate.feasible,
            budget_exhausted=False,
            steps_executed=len(candidate.evidence),
            restarts_executed=1,
            best_target_margins={target_id: -evidence.selected_teacher_kl},
            best_guard_margins={new_guard_id: -evidence.nonselected_base_kl},
            violated_contracts=tuple(sorted(candidate.violated_constraints)),
            complexity={
                "active_modules": len(candidate.active_modules),
                "total_rank": sum(candidate.ranks.values()),
            },
            failure_reason=None if candidate.feasible else "bounded extraction was infeasible",
        )

    request = RebaseRequest(
        patch=RebasePatch(
            patch_id=source_patch.manifest.patch_id,
            source_base_signature=source_manifest.signature.signature_hash,
            delta=source_result.deltas,
            target_contract_ids=(target_id,),
            preservation_contract_ids=(preservation_id,),
        ),
        source_base=source_descriptor,
        target_base=target_descriptor,
        new_base_guard_ids=(new_guard_id,),
        budget=RebaseBudget(config.rebase_steps),
        compiler_configuration={"maximum_rank": 2, "maximum_modules": 2},
    )
    result = semantic_rebase(
        request,
        applier=apply,
        verifier=verify,
        teacher_builder=teacher_builder,
        recompiler=recompile,
    )
    if not result.verified:
        raise RuntimeError(f"Hugging Face rebase failed: {result.disposition.value}")
    rebased_compilation = semantic_compilations[-1] if semantic_compilations else source_result
    candidate = apply_dense_deltas(target_base, dict(result.delta))
    payload: dict[str, object] = {
        "schema_version": 1,
        "claim": result.claim.value,
        "disposition": result.disposition.value,
        "direct_attempted": result.direct_transfer.attempted,
        "direct_verified": result.direct_transfer.verified,
        "semantic_recompile_executed": result.recompile is not None,
        "source_patch_id": source_patch.manifest.patch_id,
        "source_base_hash": source_manifest.signature.signature_hash,
        "target_base_hash": target_manifest.signature.signature_hash,
        "outputs": _outputs(adapter, candidate),
        "evidence": result.evidence.to_dict(),
    }
    return payload, rebased_compilation


def _isolated_standalone_verification(
    bundle: PatchBundle,
    base_checkpoint: Path,
) -> dict[str, object]:
    """Execute the generated verifier without making ModelPact importable."""

    discovered_roots: set[Path] = set()
    for module_name in (
        "filelock",
        "fsspec",
        "huggingface_hub",
        "numpy",
        "packaging",
        "regex",
        "requests",
        "safetensors",
        "tokenizers",
        "torch",
        "tqdm",
        "transformers",
        "yaml",
    ):
        module = importlib.import_module(module_name)
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str):
            raise RuntimeError(f"cannot locate standalone dependency: {module_name}")
        discovered_roots.add(Path(module_file).resolve().parents[1])
    dependency_roots = [
        resolved
        for item in sys.path
        if item and (resolved := Path(item).resolve()) in discovered_roots
    ]
    dependency_roots.extend(sorted(discovered_roots - set(dependency_roots)))

    with tempfile.TemporaryDirectory(
        prefix="modelpact-hf-standalone-", dir=bundle.path.parent
    ) as temporary_name:
        isolation_root = Path(temporary_name)
        environment = dict(os.environ)
        environment.update(
            {
                "HF_DATASETS_OFFLINE": "1",
                "HF_HOME": str(isolation_root / "huggingface"),
                "HF_HUB_CACHE": str(isolation_root / "huggingface" / "hub"),
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "HF_HUB_OFFLINE": "1",
                "PYTHONNOUSERSITE": "1",
                "PYTHONPATH": os.pathsep.join(str(path) for path in dependency_roots),
                "TRANSFORMERS_OFFLINE": "1",
            }
        )
        preflight = subprocess.run(  # noqa: S603 - exact interpreter and constant program
            [
                sys.executable,
                "-S",
                "-P",
                "-c",
                (
                    "import importlib.util; "
                    "assert importlib.util.find_spec('modelpact') is None; "
                    "assert importlib.util.find_spec('transformers') is not None"
                ),
            ],
            cwd=isolation_root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if preflight.returncode != 0:
            raise RuntimeError(
                "standalone Hugging Face isolation preflight failed: "
                f"{preflight.stdout}{preflight.stderr}"
            )
        process = subprocess.run(  # noqa: S603 - exact generated script and interpreter
            [
                sys.executable,
                "-S",
                "-P",
                str((bundle.path / "verify_patch.py").resolve()),
                str(base_checkpoint.resolve()),
                "--patch",
                str(bundle.path.resolve()),
                "--adapter-kind",
                "huggingface",
                "--include-holdout",
            ],
            cwd=isolation_root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
        try:
            value = json.loads(process.stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                "standalone Hugging Face verifier emitted malformed JSON: "
                f"{process.stdout}{process.stderr}"
            ) from error
        if not isinstance(value, dict):
            raise RuntimeError("standalone Hugging Face verifier result is not an object")
        if process.returncode != 0 or value.get("outcome") != "PASS":
            raise RuntimeError(
                f"standalone Hugging Face verifier failed: {process.stdout}{process.stderr}"
            )
        results = value.get("verification_results")
        if not isinstance(results, list) or not all(isinstance(item, dict) for item in results):
            raise RuntimeError("standalone Hugging Face verifier omitted assertion evidence")
        roles = {str(item.get("role")) for item in results}
        if not {"holdout_target", "holdout_guard"}.issubset(roles):
            raise RuntimeError("standalone Hugging Face verifier did not execute sealed holdout")
        if any(item.get("outcome") != "PASS" for item in results):
            raise RuntimeError("standalone Hugging Face assertion evidence contains a failure")
        if value.get("model_adapter_id") != HuggingFaceCausalLMAdapter.adapter_id:
            raise RuntimeError("standalone Hugging Face verifier used the wrong adapter")
        if value.get("unsupported_claims") != []:
            raise RuntimeError("standalone Hugging Face verifier left unsupported claims")
        return value


def _run_at(root: Path, config: HuggingFaceLocalConfig) -> dict[str, object]:
    started = time.perf_counter()
    root.mkdir(parents=True, exist_ok=False)
    prior_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        tokenizer = _tokenizer()
        base_raw = _new_model(config)
        base_losses = _train_mapping(
            base_raw,
            _BASE_BEHAVIOR,
            steps=config.base_steps,
            learning_rate=0.02,
            seed=config.seed + 1,
        )
        teacher_a_raw = copy.deepcopy(base_raw)
        teacher_b_raw = copy.deepcopy(base_raw)
        teacher_a_losses = _train_mapping(
            teacher_a_raw,
            _TEACHER_A_BEHAVIOR,
            steps=config.teacher_steps,
            learning_rate=0.01,
            seed=config.seed + 2,
        )
        teacher_b_losses = _train_mapping(
            teacher_b_raw,
            _TEACHER_B_BEHAVIOR,
            steps=config.teacher_steps,
            learning_rate=0.01,
            seed=config.seed + 3,
        )
        base_v2_raw = copy.deepcopy(base_raw)
        base_v2_losses = _train_mapping(
            base_v2_raw,
            _BASE_V2_BEHAVIOR,
            steps=config.teacher_steps,
            learning_rate=0.01,
            seed=config.seed + 4,
        )
        checkpoint_root = root / "generated-checkpoints"
        base_path = _save_checkpoint(base_raw, tokenizer, checkpoint_root / "base")
        teacher_a_path = _save_checkpoint(teacher_a_raw, tokenizer, checkpoint_root / "teacher-a")
        teacher_b_path = _save_checkpoint(teacher_b_raw, tokenizer, checkpoint_root / "teacher-b")
        base_v2_path = _save_checkpoint(base_v2_raw, tokenizer, checkpoint_root / "base-v2")

        # Discard in-memory training objects: every downstream stage uses the
        # same safe adapter path exposed to operators.
        del base_raw, teacher_a_raw, teacher_b_raw, base_v2_raw
        adapter, base = _load_local(base_path)
        _, teacher_a = _load_local(teacher_a_path)
        _, teacher_b = _load_local(teacher_b_path)
        _, base_v2 = _load_local(base_v2_path)
        base_manifest = build_model_manifest(
            base,
            checkpoint=base_path,
            adapter_id=adapter.adapter_id,
        )
        teacher_a_manifest = build_model_manifest(
            teacher_a,
            checkpoint=teacher_a_path,
            adapter_id=adapter.adapter_id,
        )
        teacher_b_manifest = build_model_manifest(
            teacher_b,
            checkpoint=teacher_b_path,
            adapter_id=adapter.adapter_id,
        )
        base_v2_manifest = build_model_manifest(
            base_v2,
            checkpoint=base_v2_path,
            adapter_id=adapter.adapter_id,
        )
        setup = {
            "base": _outputs(adapter, base),
            "teacher_a": _outputs(adapter, teacher_a),
            "teacher_b": _outputs(adapter, teacher_b),
            "base_v2": _outputs(adapter, base_v2),
        }
        expected_setup = {
            "base": _BASE_BEHAVIOR,
            "teacher_a": _TEACHER_A_BEHAVIOR,
            "teacher_b": _TEACHER_B_BEHAVIOR,
            "base_v2": _BASE_V2_BEHAVIOR,
        }
        if setup != expected_setup:
            raise RuntimeError(f"generated Hugging Face fixtures did not converge: {setup}")

        compile_started = time.perf_counter()
        result_a = _compile_patch(
            adapter,
            base,
            teacher_a,
            "fact_a",
            ("fact_b", "control"),
            config,
            seed=config.seed + 11,
        )
        result_b = _compile_patch(
            adapter,
            base,
            teacher_b,
            "fact_b",
            ("fact_a", "control"),
            config,
            seed=config.seed + 12,
        )
        compilation_seconds = time.perf_counter() - compile_started

        probes_a = _probe_payloads(
            adapter=adapter,
            teacher=teacher_a,
            prompt="fact_a",
            expected="X",
            guard_expected={"fact_b": "B", "control": "C"},
        )
        probes_b = _probe_payloads(
            adapter=adapter,
            teacher=teacher_b,
            prompt="fact_b",
            expected="Y",
            guard_expected={"fact_a": "R", "control": "C"},
        )
        contract_a = _contract(base_manifest, identifier="huggingface-local-fact-a")
        contract_b = _contract(base_manifest, identifier="huggingface-local-fact-b")
        patch_a, report_a, outputs_a, unmount_a = _complete_bundle(
            root / "patch-fact-a",
            name="huggingface-local-fact-a",
            result=result_a,
            manifest=base_manifest,
            contract=contract_a,
            probes=probes_a,
            base=base,
            teacher=teacher_a,
            adapter=adapter,
            checkpoint_hashes={
                "base": base_manifest.signature.checkpoint_hash,
                "teacher": teacher_a_manifest.signature.checkpoint_hash,
            },
            compiler_configuration={
                "maximum_rank": 2,
                "maximum_modules": 2,
                "steps": config.compiler_steps,
                "seed": config.seed + 11,
            },
        )
        patch_b, report_b, outputs_b, unmount_b = _complete_bundle(
            root / "patch-fact-b",
            name="huggingface-local-fact-b",
            result=result_b,
            manifest=base_manifest,
            contract=contract_b,
            probes=probes_b,
            base=base,
            teacher=teacher_b,
            adapter=adapter,
            checkpoint_hashes={
                "base": base_manifest.signature.checkpoint_hash,
                "teacher": teacher_b_manifest.signature.checkpoint_hash,
            },
            compiler_configuration={
                "maximum_rank": 2,
                "maximum_modules": 2,
                "steps": config.compiler_steps,
                "seed": config.seed + 12,
            },
        )
        standalone_a = _isolated_standalone_verification(patch_a, base_path)
        standalone_b = _isolated_standalone_verification(patch_b, base_path)
        standalone_reports = {
            "patch-fact-a": standalone_a,
            "patch-fact-b": standalone_b,
        }
        standalone_summaries: dict[str, object] = {}
        for standalone_name, standalone_report in sorted(standalone_reports.items()):
            verification_results = cast(
                list[dict[str, object]], standalone_report["verification_results"]
            )
            standalone_summaries[standalone_name] = {
                "adapter_kind": "huggingface",
                "include_holdout": True,
                "model_adapter_id": standalone_report["model_adapter_id"],
                "modelpact_importable": False,
                "outcome": standalone_report["outcome"],
                "result_hash": standalone_report["result_hash"],
                "verified_roles": sorted({str(item["role"]) for item in verification_results}),
            }
        atomic_write_text(
            root / "standalone-verification.json",
            canonical_dumps(standalone_reports) + "\n",
        )
        composition, _ = _composition(
            adapter,
            base,
            base_manifest,
            patch_a,
            result_a,
            patch_b,
            result_b,
        )
        atomic_write_text(root / "composition.json", canonical_dumps(composition) + "\n")

        rebase_payload, rebased_result = _rebase(
            adapter,
            base,
            base_manifest,
            base_v2,
            base_v2_manifest,
            patch_a,
            result_a,
            config,
        )
        atomic_write_text(root / "rebase.json", canonical_dumps(rebase_payload) + "\n")
        rebased_candidate = apply_dense_deltas(base_v2, rebased_result.deltas)

        success = bool(
            report_a.outcome is VerificationOutcome.PASS
            and report_a.holdout_outcome is VerificationOutcome.PASS
            and report_b.outcome is VerificationOutcome.PASS
            and report_b.holdout_outcome is VerificationOutcome.PASS
            and standalone_a["outcome"] == "PASS"
            and standalone_b["outcome"] == "PASS"
            and composition["claim"] == "COMPOSITION_CLOSED"
            and composition["outputs"] == {"control": "C", "fact_a": "X", "fact_b": "Y"}
            and rebase_payload["claim"]
            in {"DIRECT_TRANSPLANT_VERIFIED", "SEMANTIC_REBASE_VERIFIED"}
            and _outputs(adapter, rebased_candidate)
            == {"control": "Y", "fact_a": "X", "fact_b": "B"}
            and unmount_a
            and unmount_b
        )
        result: dict[str, object] = {
            "schema_version": 1,
            "suite": "ModelPactBench",
            "benchmark": "HuggingFaceLocal",
            "status": "PASS" if success else "FAIL",
            "success": success,
            "offline": True,
            "network_policy": "LOCAL_FILES_ONLY_WITH_OFFLINE_ENVIRONMENT",
            "adapter": {
                "id": adapter.adapter_id,
                "local_files_only": True,
                "trust_remote_code": False,
                "safe_tensors_only": True,
            },
            "model": {
                "architecture": "GPTNeoXForCausalLM",
                "hidden_size": config.hidden_size,
                "layers": 1,
                "attention_heads": 4,
                "vocabulary_size": len(_VOCABULARY),
                "base_signature": base_manifest.signature.signature_hash,
                "base_v2_signature": base_v2_manifest.signature.signature_hash,
                "tokenizer_hash": base_manifest.signature.tokenizer_hash,
            },
            "training": {
                "base_steps": len(base_losses),
                "base_initial_loss": base_losses[0],
                "base_final_loss": base_losses[-1],
                "teacher_a_steps": len(teacher_a_losses),
                "teacher_a_final_loss": teacher_a_losses[-1],
                "teacher_b_steps": len(teacher_b_losses),
                "teacher_b_final_loss": teacher_b_losses[-1],
                "base_v2_steps": len(base_v2_losses),
                "base_v2_final_loss": base_v2_losses[-1],
                "outputs": setup,
            },
            "patches": [
                {
                    "patch_id": patch_a.manifest.patch_id,
                    "contract_id": contract_a.contract_id,
                    "outcome": report_a.outcome.value,
                    "holdout_outcome": report_a.holdout_outcome.value,
                    "outputs": outputs_a,
                    "active_modules": list(result_a.active_modules),
                    "total_rank": sum(result_a.ranks.values()),
                    "bundle_bytes": sum(
                        path.stat().st_size for path in patch_a.path.rglob("*") if path.is_file()
                    ),
                    "runtime_unmount_exact": unmount_a,
                },
                {
                    "patch_id": patch_b.manifest.patch_id,
                    "contract_id": contract_b.contract_id,
                    "outcome": report_b.outcome.value,
                    "holdout_outcome": report_b.holdout_outcome.value,
                    "outputs": outputs_b,
                    "active_modules": list(result_b.active_modules),
                    "total_rank": sum(result_b.ranks.values()),
                    "bundle_bytes": sum(
                        path.stat().st_size for path in patch_b.path.rglob("*") if path.is_file()
                    ),
                    "runtime_unmount_exact": unmount_b,
                },
            ],
            "standalone_verification": standalone_summaries,
            "composition": composition,
            "rebase": rebase_payload,
            "performance": {
                "compiler_wall_seconds": compilation_seconds,
                "total_wall_seconds": time.perf_counter() - started,
                "compiler_steps": 2 * config.compiler_steps,
            },
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "platform": platform.platform(),
                "cuda_available": torch.cuda.is_available(),
            },
            "evidence_scope": {
                "contract_coverage": ["FINITE_PARTIAL", "SEALED_HOLDOUT"],
                "composition_coverage": "EXECUTED_DECLARED_STACK",
                "reproducibility": "DETERMINISTIC_WITHIN_ENVIRONMENT",
                "claim": (
                    "Verified under the declared contracts, probe spaces, generation policy, "
                    "environment, and search budget."
                ),
            },
            "limitations": [
                (
                    "The generated model is a tiny GPT-NeoX fixture, not a claim about "
                    "other checkpoints."
                ),
                (
                    "One deterministic training seed was executed; no comparative "
                    "significance is claimed."
                ),
                "The sealed holdout is finite and intentionally small for CPU integration testing.",
                "No GPU measurement was performed by this CPU workflow.",
            ],
        }
        atomic_write_text(root / "result.json", canonical_dumps(result) + "\n")
        return result
    finally:
        torch.set_num_threads(prior_threads)


def run_huggingface_local(
    output: str | Path | None = None,
    *,
    config: HuggingFaceLocalConfig = DEFAULT_HUGGINGFACE_LOCAL_CONFIG,
) -> dict[str, object]:
    """Execute Benchmark G, optionally retaining all locally generated evidence."""

    if not huggingface_dependencies_available():
        raise RuntimeError("Benchmark G requires optional Transformers and tokenizers packages")
    if output is not None:
        target = Path(output)
        if target.exists():
            raise FileExistsError(target)
        return _run_at(target, config)
    with tempfile.TemporaryDirectory(prefix="modelpact-huggingface-local-") as temporary:
        return _run_at(Path(temporary) / "run", config)


__all__ = [
    "DEFAULT_HUGGINGFACE_LOCAL_CONFIG",
    "HuggingFaceLocalConfig",
    "huggingface_dependencies_available",
    "run_huggingface_local",
]
