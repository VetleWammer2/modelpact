from __future__ import annotations

import json
from pathlib import Path

import torch

from modelpact import __version__
from modelpact.adapters.tiny_lm import TinyCausalLM, TinyConfig, TinyTokenizer
from modelpact.contracts.ast import BehaviorContract
from modelpact.contracts.parser import parse_contract
from modelpact.models.manifest import ModelManifest
from modelpact.patch.ast import Alias, DeltaProgram, LowRankMatrixDelta
from modelpact.patch.bundle import PatchBundle, create_patch_bundle


def constant_output_model(
    *,
    hidden_size: int,
    num_heads: int,
    output: str,
) -> TinyCausalLM:
    if len(output.encode("utf-8")) != 1:
        raise ValueError("fixture output must be one UTF-8 byte")
    model = TinyCausalLM(
        TinyConfig(
            hidden_size=hidden_size,
            intermediate_size=hidden_size,
            num_layers=1,
            num_heads=num_heads,
            max_sequence_length=96,
        )
    )
    token = TinyTokenizer.byte_offset + output.encode("utf-8")[0]
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        for name, parameter in model.named_parameters():
            if "norm.weight" in name:
                parameter.fill_(1.0)
        model.token_embedding.weight.fill_(1.0)
        model.token_embedding.weight[token].fill_(3.0)
    return model


def case_brittle_model(*, hidden_size: int = 8, num_heads: int = 2) -> TinyCausalLM:
    """Emit ``A`` everywhere; an e0-only patch can change ``x`` but not ``X``."""

    model = TinyCausalLM(
        TinyConfig(
            hidden_size=hidden_size,
            intermediate_size=hidden_size,
            num_layers=1,
            num_heads=num_heads,
            max_sequence_length=96,
            tie_word_embeddings=False,
        )
    )
    tokenizer = TinyTokenizer()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        for name, parameter in model.named_parameters():
            if "norm.weight" in name:
                parameter.fill_(1.0)
        model.token_embedding.weight[:, 0] = 1.0
        model.token_embedding.weight[tokenizer.byte_offset + ord("X")].zero_()
        model.token_embedding.weight[tokenizer.byte_offset + ord("X"), 1] = 1.0
        model.lm_head.weight[tokenizer.byte_offset + ord("A"), :2] = 2.0
    return model


def searchable_behavior_contract(
    manifest: ModelManifest,
    identifier: str,
    *,
    expected: str,
    maximum_guard_kl: float = 100.0,
) -> BehaviorContract:
    return parse_contract(
        {
            "compile": {
                "objectives": [
                    {
                        "id": "imitate-visible-behavior",
                        "source": "probes/train.jsonl",
                        "type": "teacher_cross_entropy",
                    }
                ]
            },
            "contract_version": 1,
            "generation": {"max_new_tokens": 1, "mode": "greedy", "seeds": [0]},
            "holdout": {
                "guards": "holdout/guards.jsonl",
                "sealed": True,
                "targets": "holdout/targets.jsonl",
                "unseal_policy": "final_candidate_only",
            },
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
                "bootstrap_seed": 29,
                "confidence_level": 0.95,
            },
            "verify": {
                "guards": [
                    {
                        "id": "preserve-new-base-distribution",
                        "maximum_mean": maximum_guard_kl,
                        "source": "guards/validation.jsonl",
                        "type": "base_kl",
                    }
                ],
                "targets": [
                    {
                        "id": "generate-selected-token",
                        "minimum_pass_rate": 1.0,
                        "source": "probes/validation.jsonl",
                        "type": "free_generation_match",
                    }
                ],
            },
        }
    )


def learned_behavior_bundle(
    path: Path,
    manifest: ModelManifest,
    contract: BehaviorContract,
    *,
    source_output: str,
    target_output: str,
) -> PatchBundle:
    tokenizer = TinyTokenizer()
    source_token = tokenizer.byte_offset + source_output.encode("utf-8")[0]
    target_token = tokenizer.byte_offset + target_output.encode("utf-8")[0]
    width = manifest.state_schema.tensor("lm_head.weight").shape[1]
    left = torch.zeros((tokenizer.vocab_size, 1), dtype=torch.float32)
    left[source_token, 0] = -2.0
    left[target_token, 0] = 3.0
    right = torch.ones((1, width), dtype=torch.float32)
    contract_bytes = (json.dumps(contract.to_dict(), sort_keys=True) + "\n").encode()
    expected = json.dumps(
        {"expected": target_output, "id": "visible", "prompt": "x"},
        sort_keys=True,
    ).encode()
    guard = b'{"id":"guard","prompt":"control"}'
    holdout_target = json.dumps(
        {"expected": target_output, "id": "sealed", "prompt": "x?"},
        sort_keys=True,
    ).encode()
    holdout_guard = b'{"id":"sealed-guard","prompt":"neighbor"}'
    return create_patch_bundle(
        path,
        name=contract.id,
        base_signature=manifest.signature.to_dict(),
        state_schema=manifest.state_schema,
        program=DeltaProgram(
            {
                "lm_head.weight": LowRankMatrixDelta("behavior.left", "behavior.right"),
                "token_embedding.weight": Alias("lm_head.weight"),
            }
        ),
        tensors={"behavior.left": left, "behavior.right": right},
        tool_version=__version__,
        contracts={
            "contracts/guards/validation.jsonl": guard + b"\n",
            "contracts/holdout/guards.jsonl": holdout_guard + b"\n",
            "contracts/holdout/targets.jsonl": holdout_target + b"\n",
            "contracts/preservation.yaml": contract_bytes,
            "contracts/probes/train.jsonl": (
                json.dumps(
                    {"id": "train", "prompt": "x", "target": target_output},
                    sort_keys=True,
                ).encode()
                + b"\n"
            ),
            "contracts/probes/validation.jsonl": expected + b"\n",
            "contracts/target.yaml": contract_bytes,
        },
        provides=(contract.contract_id,),
        preserves=(contract.contract_id,),
    )


def brittle_case_behavior_bundle(
    path: Path,
    manifest: ModelManifest,
    contract: BehaviorContract,
) -> PatchBundle:
    tokenizer = TinyTokenizer()
    width = manifest.state_schema.tensor("lm_head.weight").shape[1]
    left = torch.zeros((tokenizer.vocab_size, 1), dtype=torch.float32)
    left[tokenizer.byte_offset + ord("Q"), 0] = 3.0
    right = torch.zeros((1, width), dtype=torch.float32)
    right[0, 0] = 1.0
    contract_bytes = (json.dumps(contract.to_dict(), sort_keys=True) + "\n").encode()
    return create_patch_bundle(
        path,
        name=contract.id,
        base_signature=manifest.signature.to_dict(),
        state_schema=manifest.state_schema,
        program=DeltaProgram(
            {"lm_head.weight": LowRankMatrixDelta("behavior.left", "behavior.right")}
        ),
        tensors={"behavior.left": left, "behavior.right": right},
        tool_version=__version__,
        contracts={
            "contracts/guards/validation.jsonl": b'{"id":"guard","prompt":"control"}\n',
            "contracts/holdout/guards.jsonl": (b'{"id":"sealed-guard","prompt":"neighbor"}\n'),
            "contracts/holdout/targets.jsonl": (b'{"expected":"Q","id":"sealed","prompt":"x?"}\n'),
            "contracts/preservation.yaml": contract_bytes,
            "contracts/probes/train.jsonl": (b'{"id":"train","prompt":"x","target":"Q"}\n'),
            "contracts/probes/validation.jsonl": (
                b'{"expected":"Q","id":"visible","prompt":"x"}\n'
            ),
            "contracts/target.yaml": contract_bytes,
        },
        provides=(contract.contract_id,),
        preserves=(contract.contract_id,),
    )
