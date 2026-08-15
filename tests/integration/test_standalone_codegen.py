from __future__ import annotations

import json
import os
import py_compile
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy
import packaging
import safetensors
import torch
from safetensors.torch import load_file, save_file

from modelpact.codegen import emit_apply_script, emit_verify_script
from modelpact.contracts.parser import parse_contract
from modelpact.models.aliases import AliasGroup
from modelpact.models.fingerprint import (
    chat_template_fingerprint,
    checkpoint_tensor_fingerprint,
    configuration_fingerprint,
    generation_config_fingerprint,
    tokenizer_fingerprint,
)
from modelpact.models.schema import ModelStateSchema, TensorSpec
from modelpact.patch.ast import (
    Alias,
    DeltaProgram,
    LowRankMatrixDelta,
    SparseMatrixDelta,
    Sum,
    VectorDelta,
)
from modelpact.patch.bundle import create_patch_bundle
from modelpact.util.hashing import hash_canonical, sha256_file


def _isolated_environment(directory: Path) -> dict[str, str]:
    """Expose dependencies but do not process the checkout's site .pth files."""

    dependency_roots = {
        Path(torch.__file__).resolve().parents[1],
        Path(safetensors.__file__).resolve().parents[1],
        Path(numpy.__file__).resolve().parents[1],
        Path(packaging.__file__).resolve().parents[1],
    }
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(directory), *(str(path) for path in sorted(dependency_roots))]
    )
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def _run(
    script: Path,
    *arguments: str | Path,
    cwd: Path,
    environment: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - test executes the generated script by exact path
        [sys.executable, "-S", str(script), *(str(argument) for argument in arguments)],
        cwd=cwd,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _result(process: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    assert process.stdout.strip(), process.stderr
    value = json.loads(process.stdout)
    assert isinstance(value, dict)
    return value


def _write_adapter(path: Path) -> None:
    path.write_text(
        """from types import SimpleNamespace
from safetensors.torch import load_file


class Tokenizer:
    def batch(self, texts):
        return list(texts)


class Adapter:
    adapter_id = "fixture_adapter.v1"

    def load(self, checkpoint, *, device, dtype):
        del dtype
        return load_file(checkpoint + "/model.safetensors", device=str(device))

    def tokenizer(self):
        return Tokenizer()

    def prepare(self, model):
        del model

    def generate(self, model, batch, policy):
        del policy
        prompt = batch[0]
        if prompt == "target":
            text = "ok" if float(model["vector"][0]) > 0.5 else "not-ok"
        else:
            text = "stable"
        return [SimpleNamespace(text=text, token_ids=tuple(map(ord, text)), finished=True)]

    def forward_logits(self, model, batch):
        raise RuntimeError("this fixture uses free-generation assertions only")
""",
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    base = tmp_path / "base"
    base.mkdir()
    state = {
        "matrix": torch.zeros((2, 2), dtype=torch.float32),
        "tied_a": torch.zeros((2, 2), dtype=torch.float32),
        "tied_b": torch.zeros((2, 2), dtype=torch.float32),
        "vector": torch.zeros(2, dtype=torch.float32),
    }
    save_file(state, base / "model.safetensors")
    (base / "config.json").write_text('{"fixture":true}', encoding="utf-8")
    (base / "tokenizer.json").write_text('{"fixture":true}', encoding="utf-8")
    (base / "tokenizer_config.json").write_text(
        '{"chat_template":"fixture-template"}', encoding="utf-8"
    )
    (base / "generation_config.json").write_text('{"max_new_tokens":8}', encoding="utf-8")
    (base / "modeling.py").write_text("raise RuntimeError('must not be copied')", encoding="utf-8")
    checkpoint_hash, _ = checkpoint_tensor_fingerprint(base)
    tensor_specs = tuple(
        TensorSpec(name, tuple(tensor.shape), "float32", True, "fixture")
        for name, tensor in sorted(state.items())
    )
    schema = ModelStateSchema(
        schema_version=1,
        tensors=tensor_specs,
        modules=(),
        aliases=(AliasGroup("tied_a", ("tied_a", "tied_b")),),
    )
    tensors = {
        "left": torch.tensor([[1.0], [2.0]]),
        "right": torch.tensor([[3.0, 4.0]]),
        "sparse_indices": torch.tensor([[0, 1]], dtype=torch.int64),
        "sparse_values": torch.tensor([2.0]),
        "tied_left": torch.ones((2, 1)),
        "tied_right": torch.full((1, 2), 0.5),
        "vector_delta": torch.tensor([1.0, -1.0]),
    }
    program = DeltaProgram(
        {
            "matrix": Sum(
                (
                    LowRankMatrixDelta("left", "right"),
                    SparseMatrixDelta("sparse_indices", "sparse_values", (2, 2)),
                )
            ),
            "tied_a": LowRankMatrixDelta("tied_left", "tied_right"),
            "tied_b": Alias("tied_a"),
            "vector": VectorDelta("vector_delta"),
        }
    )
    contract = {
        "schema_version": 1,
        "id": "standalone-codegen",
        "contract_version": 1,
        "model_requirements": {
            "adapter_id": "fixture_adapter.v1",
            "tokenizer_hash": tokenizer_fingerprint(base),
            "output_semantics": "causal_lm",
        },
        "compile": {"objectives": []},
        "verify": {
            "targets": [
                {
                    "id": "target-output",
                    "type": "exact_match",
                    "source": "contracts/probes/targets.jsonl",
                    "minimum_pass_rate": 1.0,
                }
            ],
            "guards": [
                {
                    "id": "stable-control",
                    "type": "normalized_exact_match",
                    "source": "contracts/probes/guards.jsonl",
                    "minimum_pass_rate": 1.0,
                }
            ],
        },
        "holdout": {"sealed": True, "unseal_policy": "final_candidate_only"},
        "statistics": {
            "confidence_level": 0.95,
            "bootstrap_samples": 20,
            "bootstrap_seed": 7,
            "multiple_comparison": "none",
        },
        "generation": {"mode": "greedy", "max_new_tokens": 8, "seeds": [5]},
    }
    parsed_contract = parse_contract(contract)
    contract_bytes = json.dumps(contract, sort_keys=True).encode()
    contracts = {
        "contracts/target.yaml": contract_bytes,
        "contracts/probes/targets.jsonl": b'{"id":"t","prompt":"target","expected":"ok"}\n',
        "contracts/probes/guards.jsonl": b'{"id":"g","prompt":"guard","expected":" stable "}\n',
    }
    bundle = create_patch_bundle(
        tmp_path / "patch",
        name="standalone-codegen",
        base_signature={
            "adapter_id": "fixture_adapter.v1",
            "architecture_hash": hash_canonical({"fixture": "architecture"}),
            "chat_template_hash": chat_template_fingerprint(base),
            "checkpoint_hash": checkpoint_hash,
            "configuration_hash": configuration_fingerprint(base),
            "generation_config_hash": generation_config_fingerprint(base),
            "schema_version": 1,
            "state_schema_hash": schema.schema_hash,
            "tokenizer_hash": tokenizer_fingerprint(base),
        },
        state_schema=schema,
        program=program,
        tensors=tensors,
        tool_version="0.1.0",
        contracts=contracts,
        supplemental_artifacts={"evidence/compile.json": b'{"schema_version":1}'},
        provides=(parsed_contract.contract_id,),
        preserves=(parsed_contract.contract_id,),
    )
    return base, bundle.path


def test_standalone_apply_isolated_and_detects_tampering(tmp_path: Path) -> None:
    base, bundle = _fixture(tmp_path)
    script = emit_apply_script(bundle, tmp_path / "apply_patch.py")
    source = script.read_text(encoding="utf-8")
    assert re.search(r"^\s*(?:from|import)\s+modelpact\b", source, re.MULTILINE) is None
    assert str(tmp_path.resolve()) not in source
    environment = _isolated_environment(tmp_path)
    preflight = subprocess.run(  # noqa: S603 - exact interpreter and constant code
        [
            sys.executable,
            "-S",
            "-c",
            "import importlib.util; assert importlib.util.find_spec('modelpact') is None",
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert preflight.returncode == 0, preflight.stderr
    before = {path.name: sha256_file(path) for path in base.iterdir() if path.is_file()}
    process = _run(script, base, tmp_path / "output", cwd=tmp_path, environment=environment)
    assert process.returncode == 0, process.stdout + process.stderr
    result = _result(process)
    assert result["outcome"] == "PASS"
    after = {path.name: sha256_file(path) for path in base.iterdir() if path.is_file()}
    assert before == after
    output = load_file(tmp_path / "output" / "model.safetensors")
    assert torch.equal(output["matrix"], torch.tensor([[3.0, 6.0], [6.0, 8.0]]))
    assert torch.equal(output["vector"], torch.tensor([1.0, -1.0]))
    assert torch.equal(output["tied_a"], output["tied_b"])
    assert (tmp_path / "output" / "config.json").read_bytes() == (base / "config.json").read_bytes()
    assert not (tmp_path / "output" / "modeling.py").exists()

    tampered = tmp_path / "tampered-patch"
    shutil.copytree(bundle, tampered)
    tensor_path = tampered / "tensors.safetensors"
    changed_bytes = bytearray(tensor_path.read_bytes())
    changed_bytes[-1] ^= 1
    tensor_path.write_bytes(changed_bytes)
    process = _run(
        script,
        "--patch",
        tampered,
        base,
        tmp_path / "tampered-output",
        cwd=tmp_path,
        environment=environment,
    )
    assert process.returncode != 0
    assert _result(process)["outcome"] == "FAIL"
    assert not (tmp_path / "tampered-output").exists()

    rebound = tmp_path / "rebound-evidence-patch"
    shutil.copytree(bundle, rebound)
    evidence_path = rebound / "evidence" / "compile.json"
    evidence_path.write_bytes(b'{"schema_version":1,"tampered":true}')
    manifest_path = rebound / "manifest.json"
    rebound_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rebound_manifest["artifact_hashes"]["evidence/compile.json"] = sha256_file(evidence_path)
    manifest_path.write_text(json.dumps(rebound_manifest, sort_keys=True), encoding="utf-8")
    process = _run(
        script,
        "--patch",
        rebound,
        base,
        tmp_path / "rebound-output",
        cwd=tmp_path,
        environment=environment,
    )
    assert process.returncode != 0
    assert "evidence identity" in str(_result(process)["error"])
    assert not (tmp_path / "rebound-output").exists()

    traversal = tmp_path / "traversal-patch"
    shutil.copytree(bundle, traversal)
    manifest_path = traversal / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_hashes"]["../escape"] = "sha256:" + "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    process = _run(
        script,
        "--patch",
        traversal,
        base,
        tmp_path / "traversal-output",
        cwd=tmp_path,
        environment=environment,
    )
    assert process.returncode != 0
    assert "unsafe relative artifact path" in str(_result(process)["error"])
    assert not (tmp_path / "traversal-output").exists()

    wrong_base = tmp_path / "wrong-base"
    shutil.copytree(base, wrong_base)
    replacement = wrong_base / "replacement.safetensors"
    save_file(
        {
            "matrix": torch.zeros((2, 2)),
            "tied_a": torch.zeros((2, 2)),
            "tied_b": torch.zeros((2, 2)),
            "vector": torch.tensor([10.0, 0.0]),
        },
        replacement,
    )
    (wrong_base / "model.safetensors").unlink()
    replacement.rename(wrong_base / "model.safetensors")
    process = _run(
        script,
        wrong_base,
        tmp_path / "wrong-output",
        cwd=tmp_path,
        environment=environment,
    )
    assert process.returncode != 0
    assert "fingerprint mismatch" in str(_result(process)["error"])
    assert not (tmp_path / "wrong-output").exists()

    identity_mutations = {
        "config.json": b'{"fixture":false}',
        "tokenizer.json": b'{"fixture":false}',
        "tokenizer_config.json": b'{"chat_template":"changed"}',
        "generation_config.json": b'{"max_new_tokens":9}',
    }
    for index, (name, content) in enumerate(sorted(identity_mutations.items())):
        changed = tmp_path / f"identity-{index}"
        shutil.copytree(base, changed)
        (changed / name).write_bytes(content)
        changed_output = tmp_path / f"identity-output-{index}"
        process = _run(
            script,
            changed,
            changed_output,
            cwd=tmp_path,
            environment=environment,
        )
        assert process.returncode != 0
        assert "identity mismatch" in str(_result(process)["error"])
        assert not changed_output.exists()


def test_standalone_verify_reexecutes_contracts_without_package(tmp_path: Path) -> None:
    base, bundle = _fixture(tmp_path)
    _write_adapter(tmp_path / "fixture_adapter.py")
    script = emit_verify_script(bundle, tmp_path / "verify_patch.py")
    source = script.read_text(encoding="utf-8")
    assert "import modelpact" not in source
    assert str(tmp_path.resolve()) not in source
    environment = _isolated_environment(tmp_path)
    report = tmp_path / "verification.json"
    process = _run(
        script,
        base,
        "--adapter",
        "fixture_adapter:Adapter",
        "--output",
        report,
        cwd=tmp_path,
        environment=environment,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    result = _result(process)
    assert result["outcome"] == "PASS"
    assert result["unsupported_claims"] == []
    assert {item["role"] for item in result["verification_results"]} == {"target", "guard"}
    assert len(result["free_generation_results"]) == 2
    assert json.loads(report.read_text(encoding="utf-8"))["result_hash"] == result["result_hash"]

    tampered = tmp_path / "verify-tampered"
    shutil.copytree(bundle, tampered)
    probe = tampered / "contracts" / "probes" / "targets.jsonl"
    probe.write_text('{"id":"t","prompt":"target","expected":"not-ok"}\n', encoding="utf-8")
    process = _run(
        script,
        base,
        "--patch",
        tampered,
        "--adapter",
        "fixture_adapter:Adapter",
        cwd=tmp_path,
        environment=environment,
    )
    assert process.returncode != 0
    assert "artifact hash mismatch" in str(_result(process)["error"])

    bundle_script = emit_verify_script(bundle, bundle / "independent_verify.py")
    execution_marker = tmp_path / "bundle-adapter-executed"
    (bundle / "bundle_adapter.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(execution_marker)!r}).write_text('executed', encoding='utf-8')\n"
        "class Adapter:\n"
        "    pass\n",
        encoding="utf-8",
    )
    process = _run(
        bundle_script,
        base,
        "--adapter",
        "bundle_adapter:Adapter",
        cwd=bundle,
        environment=environment,
    )
    assert process.returncode != 0
    assert "outside the untrusted patch bundle" in str(_result(process)["error"])
    assert not execution_marker.exists()

    bytecode_source = bundle / "bytecode_adapter.py"
    bytecode_marker = tmp_path / "bytecode-adapter-executed"
    bytecode_source.write_text(
        "from pathlib import Path\n"
        f"Path({str(bytecode_marker)!r}).write_text('executed', encoding='utf-8')\n"
        "class Adapter:\n"
        "    pass\n",
        encoding="utf-8",
    )
    py_compile.compile(
        str(bytecode_source),
        cfile=str(bundle / "bytecode_adapter.pyc"),
        doraise=True,
    )
    bytecode_source.unlink()
    process = _run(
        bundle_script,
        base,
        "--adapter",
        "bytecode_adapter:Adapter",
        cwd=bundle,
        environment=environment,
    )
    assert process.returncode != 0
    assert "outside the untrusted patch bundle" in str(_result(process)["error"])
    assert not bytecode_marker.exists()
