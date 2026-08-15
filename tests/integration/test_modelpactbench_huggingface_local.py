from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from typer.testing import CliRunner

pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

from modelpact.cli import app
from modelpact.modelpactbench.huggingface_local import run_huggingface_local
from modelpact.patch.bundle import MANDATORY_BUNDLE_ARTIFACTS, load_patch_bundle
from modelpact.verify.certificate import read_certificate, validate_certificate


@pytest.mark.integration
@pytest.mark.slow
def test_generated_local_huggingface_workflow_is_real_and_offline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "huggingface-local"
    result = run_huggingface_local(output)

    assert result["suite"] == "ModelPactBench"
    assert result["benchmark"] == "HuggingFaceLocal"
    assert result["status"] == "PASS"
    assert result["success"] is True
    assert result["offline"] is True
    assert result["network_policy"] == "LOCAL_FILES_ONLY_WITH_OFFLINE_ENVIRONMENT"
    assert result["adapter"] == {
        "id": "modelpact.huggingface_causal_lm.local.v1",
        "local_files_only": True,
        "safe_tensors_only": True,
        "trust_remote_code": False,
    }

    training = result["training"]
    assert isinstance(training, dict)
    assert training["outputs"] == {
        "base": {"control": "C", "fact_a": "R", "fact_b": "B"},
        "base_v2": {"control": "Y", "fact_a": "R", "fact_b": "B"},
        "teacher_a": {"control": "C", "fact_a": "X", "fact_b": "B"},
        "teacher_b": {"control": "C", "fact_a": "R", "fact_b": "Y"},
    }

    patches = result["patches"]
    assert isinstance(patches, list)
    assert len(patches) == 2
    for name, patch_result in zip(("patch-fact-a", "patch-fact-b"), patches, strict=True):
        assert patch_result["outcome"] == "PASS"
        assert patch_result["holdout_outcome"] == "PASS"
        assert patch_result["runtime_unmount_exact"] is True
        bundle = load_patch_bundle(output / name)
        assert bundle.manifest.patch_id == patch_result["patch_id"]
        assert set(bundle.manifest.artifact_hashes) >= MANDATORY_BUNDLE_ARTIFACTS
        apply_script = (bundle.path / "apply_patch.py").read_text(encoding="utf-8")
        verify_script = (bundle.path / "verify_patch.py").read_text(encoding="utf-8")
        assert "import modelpact" not in apply_script
        assert "import modelpact" not in verify_script
        assert 'choices=("tiny", "huggingface")' in verify_script
        assert "local_files_only=True" in verify_script
        assert "trust_remote_code=False" in verify_script
        assert "use_safetensors=True" in verify_script
        certificate = read_certificate(bundle.path / "certificate.json")
        validate_certificate(certificate, artifact_root=bundle.path)
        assert certificate.patch_id == bundle.manifest.patch_id
        assert "FREE_GENERATION_VERIFIED" in certificate.claims
        assert "SEALED_HOLDOUT_VERIFIED" in certificate.claims

    standalone = result["standalone_verification"]
    assert isinstance(standalone, dict)
    assert set(standalone) == {"patch-fact-a", "patch-fact-b"}
    for verification in standalone.values():
        assert verification["outcome"] == "PASS"
        assert verification["adapter_kind"] == "huggingface"
        assert verification["modelpact_importable"] is False
        assert verification["include_holdout"] is True
        assert verification["model_adapter_id"] == result["adapter"]["id"]
        assert {"holdout_target", "holdout_guard"}.issubset(verification["verified_roles"])

    composition = result["composition"]
    assert isinstance(composition, dict)
    assert composition["claim"] == "COMPOSITION_CLOSED"
    assert composition["executed_contract_closure"] is True
    assert composition["outputs"] == {"control": "C", "fact_a": "X", "fact_b": "Y"}
    assert all(item["passed"] for item in composition["margins"])

    rebase = result["rebase"]
    assert isinstance(rebase, dict)
    assert rebase["claim"] in {
        "DIRECT_TRANSPLANT_VERIFIED",
        "SEMANTIC_REBASE_VERIFIED",
    }
    assert rebase["outputs"] == {"control": "Y", "fact_a": "X", "fact_b": "B"}
    assert (output / "composition.json").is_file()
    assert (output / "rebase.json").is_file()
    assert (output / "standalone-verification.json").is_file()
    assert (output / "result.json").is_file()
    assert not list((output / "generated-checkpoints").rglob("*.bin"))
    assert len(list((output / "generated-checkpoints").rglob("*.safetensors"))) == 4

    materialized = output / "materialized-fact-a"
    apply_result = CliRunner().invoke(
        app,
        [
            "apply",
            str(output / "generated-checkpoints" / "base"),
            str(output / "patch-fact-a"),
            "--output",
            str(materialized),
            "--adapter",
            "huggingface",
            "--max-shard-size",
            "4096",
            "--json",
        ],
    )
    assert apply_result.exit_code == 0, apply_result.stdout
    apply_payload = json.loads(apply_result.stdout)
    assert apply_payload["status"] == "PASS"
    assert apply_payload["output"] == materialized.as_posix()
    tensor_files = sorted(materialized.glob("*.safetensors"))
    assert len(tensor_files) > 1
    for tensor_file in tensor_files:
        with safe_open(tensor_file, framework="pt", device="cpu") as handle:  # type: ignore[no-untyped-call]
            assert handle.metadata() == {
                "format": "pt",
                "modelpact_format": "modelpact-materialized-v1",
            }

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    reloaded = AutoModelForCausalLM.from_pretrained(
        materialized,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        materialized,
        local_files_only=True,
        trust_remote_code=False,
    )
    encoded = tokenizer("fact_a", return_tensors="pt")
    with torch.no_grad():
        token = int(reloaded(**encoded, use_cache=False).logits[0, -1].argmax().item())
    assert tokenizer.decode([token], skip_special_tokens=True).strip() == "X"
