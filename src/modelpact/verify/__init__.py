"""Independent behavior-patch verification and certificate generation."""

from modelpact.verify.certificate import (
    CertificateError,
    CertificateExpectations,
    CertificateIntegrityError,
    VerificationCertificate,
    build_certificate,
    loads_certificate,
    read_certificate,
    validate_certificate,
    write_certificate,
)
from modelpact.verify.engine import (
    AssertionRecordProvider,
    ExecutionIdentity,
    MappingRecordProvider,
    VerificationReport,
    VerificationRole,
    combine_outcomes,
    verify_contract,
)
from modelpact.verify.generation import (
    FreeGenerationRecord,
    GeneratedOutput,
    GenerationBackend,
    GenerationExecution,
    GenerationRequest,
    execute_free_generation,
)
from modelpact.verify.independent import (
    IndependentVerificationResult,
    hash_artifacts,
    independently_verify,
)
from modelpact.verify.provider import (
    ModelBackedRecordProvider,
    ProbeDataError,
    ProbeLimits,
    load_json_schemas,
    load_probe_records,
)

__all__ = [
    "AssertionRecordProvider",
    "CertificateError",
    "CertificateExpectations",
    "CertificateIntegrityError",
    "ExecutionIdentity",
    "FreeGenerationRecord",
    "GeneratedOutput",
    "GenerationBackend",
    "GenerationExecution",
    "GenerationRequest",
    "IndependentVerificationResult",
    "MappingRecordProvider",
    "ModelBackedRecordProvider",
    "ProbeDataError",
    "ProbeLimits",
    "VerificationCertificate",
    "VerificationReport",
    "VerificationRole",
    "build_certificate",
    "combine_outcomes",
    "execute_free_generation",
    "hash_artifacts",
    "independently_verify",
    "load_json_schemas",
    "load_probe_records",
    "loads_certificate",
    "read_certificate",
    "validate_certificate",
    "verify_contract",
    "write_certificate",
]
