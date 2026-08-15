"""Behavior Patch Bundle v1 manifest and stable patch identity."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from modelpact.models.manifest import ModelSignature
from modelpact.util.hashing import hash_canonical, is_sha256_digest


@dataclass(frozen=True, slots=True)
class PatchManifest:
    schema_version: int
    tool_version: str
    patch_id: str
    name: str
    base_signature: Mapping[str, object]
    target_module_schema_hash: str
    delta_representation: str
    provides: tuple[str, ...]
    preserves: tuple[str, ...]
    requires: tuple[str, ...]
    artifact_hashes: Mapping[str, str]
    verification_policy_hash: str | None = None
    parent_patches: tuple[str, ...] = ()
    merged_from: tuple[str, ...] = ()
    rebased_from: str | None = None
    source_diff_bundle: str | None = None
    compiler_configuration: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError(f"unsupported patch manifest version: {self.schema_version}")
        if not self.name or len(self.name) > 256:
            raise ValueError("invalid patch name")
        if self.delta_representation != "additive_low_rank_sparse_v1":
            raise ValueError(f"unsupported delta representation: {self.delta_representation}")
        if self.patch_id and not is_sha256_digest(self.patch_id):
            raise ValueError("patch_id must be a tagged SHA-256 digest")
        if not is_sha256_digest(self.target_module_schema_hash):
            raise ValueError("target_module_schema_hash must be a tagged SHA-256 digest")
        signature = ModelSignature.from_dict(self.base_signature)
        if signature.state_schema_hash != self.target_module_schema_hash:
            raise ValueError("patch target schema hash differs from its base signature")
        for collection in (
            self.provides,
            self.preserves,
            self.requires,
            self.parent_patches,
            self.merged_from,
        ):
            if tuple(sorted(set(collection))) != collection:
                raise ValueError("manifest identity lists must be unique and sorted")
        for name, values in (
            ("provides", self.provides),
            ("preserves", self.preserves),
            ("requires", self.requires),
        ):
            if any(not is_sha256_digest(value) for value in values):
                raise ValueError(f"{name} must contain contract SHA-256 identities")
        for path, digest in self.artifact_hashes.items():
            if (
                not isinstance(path, str)
                or not isinstance(digest, str)
                or not path
                or path.startswith(("/", "\\"))
                or ".." in path.replace("\\", "/").split("/")
            ):
                raise ValueError(f"unsafe artifact path: {path}")
            if not is_sha256_digest(digest):
                raise ValueError(f"invalid artifact digest: {path}")
        if self.verification_policy_hash is not None and not is_sha256_digest(
            self.verification_policy_hash
        ):
            raise ValueError("verification_policy_hash must be a tagged SHA-256 digest")
        for name, values in (
            ("parent_patches", self.parent_patches),
            ("merged_from", self.merged_from),
        ):
            if any(not is_sha256_digest(value) for value in values):
                raise ValueError(f"{name} must contain tagged SHA-256 digests")
        for name, value in (
            ("rebased_from", self.rebased_from),
            ("source_diff_bundle", self.source_diff_bundle),
        ):
            if value is not None and not is_sha256_digest(value):
                raise ValueError(f"{name} must be a tagged SHA-256 digest")

    def _payload(self, *, identity_only: bool) -> dict[str, object]:
        artifact_hashes = self.artifact_hashes
        if identity_only:
            # Evidence and generated helpers may refer to the already-computed
            # patch ID. Only the canonical delta, factors, and contracts are
            # inputs to that ID, avoiding a certificate/hash cycle.
            artifact_hashes = {
                path: digest
                for path, digest in self.artifact_hashes.items()
                if path in {"delta-program.json", "tensors.safetensors"}
                or path.startswith("contracts/")
            }

        return {
            "artifact_hashes": dict(sorted(artifact_hashes.items())),
            "base_signature": dict(self.base_signature),
            "compiler_configuration": dict(self.compiler_configuration),
            "delta_representation": self.delta_representation,
            "merged_from": list(self.merged_from),
            "name": self.name,
            "parent_patches": list(self.parent_patches),
            "preserves": list(self.preserves),
            "provides": list(self.provides),
            "rebased_from": self.rebased_from,
            "requires": list(self.requires),
            "schema_version": self.schema_version,
            "source_diff_bundle": self.source_diff_bundle,
            "target_module_schema_hash": self.target_module_schema_hash,
            "tool_version": self.tool_version,
            "verification_policy_hash": self.verification_policy_hash,
        }

    def identity_payload(self) -> dict[str, object]:
        """Return the stable patch-ID input, excluding post-ID evidence artifacts."""

        return self._payload(identity_only=True)

    def evidence_payload(self) -> dict[str, object]:
        """Return the immutable evidence-bearing identity pinned by emitted tools.

        Generated helpers and the certificate are excluded because they embed
        the core patch identity. All executable contracts, probes, evidence,
        reports, deltas, and tensors are included.
        """

        return {
            "artifact_hashes": {
                path: digest
                for path, digest in sorted(self.artifact_hashes.items())
                if path not in {"apply_patch.py", "verify_patch.py", "certificate.json"}
            },
            "patch_id": self.patch_id,
            "schema_version": 1,
        }

    @property
    def evidence_id(self) -> str:
        return hash_canonical(self.evidence_payload())

    def computed_patch_id(self) -> str:
        return hash_canonical(self.identity_payload())

    def validate_identity(self) -> None:
        if self.patch_id != self.computed_patch_id():
            raise ValueError("patch manifest identity does not match its content")

    def to_dict(self) -> dict[str, object]:
        return {"patch_id": self.patch_id, **self._payload(identity_only=False)}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> PatchManifest:
        allowed = {
            "artifact_hashes",
            "base_signature",
            "compiler_configuration",
            "delta_representation",
            "merged_from",
            "name",
            "parent_patches",
            "patch_id",
            "preserves",
            "provides",
            "rebased_from",
            "requires",
            "schema_version",
            "source_diff_bundle",
            "target_module_schema_hash",
            "tool_version",
            "verification_policy_hash",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown patch manifest fields: {sorted(unknown)}")

        def string_list(key: str) -> tuple[str, ...]:
            item = value.get(key, [])
            if not isinstance(item, list) or not all(isinstance(part, str) for part in item):
                raise ValueError(f"invalid patch manifest field: {key}")
            return tuple(item)

        maps = {}
        for key in ("base_signature", "artifact_hashes", "compiler_configuration"):
            item = value.get(key, {})
            if not isinstance(item, Mapping) or not all(isinstance(part, str) for part in item):
                raise ValueError(f"invalid patch manifest field: {key}")
            maps[key] = dict(item)
        required_strings = (
            "tool_version",
            "patch_id",
            "name",
            "target_module_schema_hash",
            "delta_representation",
        )
        strings: dict[str, str] = {}
        for key in required_strings:
            item = value.get(key)
            if not isinstance(item, str):
                raise ValueError(f"invalid patch manifest field: {key}")
            strings[key] = item
        optionals: dict[str, str | None] = {}
        for key in ("verification_policy_hash", "rebased_from", "source_diff_bundle"):
            item = value.get(key)
            if item is not None and not isinstance(item, str):
                raise ValueError(f"invalid patch manifest field: {key}")
            optionals[key] = item
        schema_version = value.get("schema_version")
        if not isinstance(schema_version, int):
            raise ValueError("invalid patch manifest schema version")
        return cls(
            schema_version=schema_version,
            tool_version=strings["tool_version"],
            patch_id=strings["patch_id"],
            name=strings["name"],
            base_signature=maps["base_signature"],
            target_module_schema_hash=strings["target_module_schema_hash"],
            delta_representation=strings["delta_representation"],
            provides=string_list("provides"),
            preserves=string_list("preserves"),
            requires=string_list("requires"),
            artifact_hashes=maps["artifact_hashes"],
            verification_policy_hash=optionals["verification_policy_hash"],
            parent_patches=string_list("parent_patches"),
            merged_from=string_list("merged_from"),
            rebased_from=optionals["rebased_from"],
            source_diff_bundle=optionals["source_diff_bundle"],
            compiler_configuration=maps["compiler_configuration"],
        )
