# Trust model

Trusted model adapters are arbitrary local Python and may execute anything the
current user can execute. Load only adapters you trust.

Contracts, patch manifests, delta programs, certificates, Rebase Evidence,
SafeTensors metadata, checkpoint indexes, probe manifests, stack lockfiles,
tensor keys, and output paths are untrusted data. Parsing them cannot import
code, run scripts, use pickle/eval, or contact a network. Bundle scripts are
output artifacts; ModelPact never executes them merely because they are present.

Bare Rebase Evidence v1 and on-disk stack lock records cross the same hostile
JSON boundary as certificates. Their readers require bounded canonical UTF-8
JSON (with at most the writer's final LF), reject duplicate keys, non-finite
numbers, excessive size/depth/counts/strings, unknown or missing closed-schema
fields, malformed hashes, invalid enumerations, and inconsistent semantic state.
There is no lenient or auto-correcting mode. Rebase Evidence carries an
`evidence_hash` over its canonical payload; recomputing that hash cannot bypass
claim/attempt/budget invariants or caller-pinned source, target, claim, and
contract-identity expectations.

A successful rebased bundle requires two reserved, hash-pinned canonical
artifacts: bare `evidence/rebase.json` and
`evidence/source-manifest.json`. The companion source manifest self-validates and
binds the evidence to the source patch ID, full source-base signature, and source
target-contract identities. The target manifest independently binds the target
base and target-versus-preservation contract roles. A rebase-bearing certificate
must name both artifacts; artifact-root certificate validation parses the target
manifest, both lineage artifacts, and the executable contracts so a rehashed
record cannot swap target and guard roles while preserving their union.

Bundle-relative paths have one portable POSIX spelling. Readers reject
case-insensitive collisions, alternate case for fixed reserved paths, Windows
device names and alternate-data-stream syntax, forbidden/control characters,
components ending in a dot or space, traversal, and symlink resolution. These
checks prevent a platform alias from turning a hostile file into a different
logical artifact.

A stack lock that names local patch paths is not trusted merely because its JSON
is valid. Before model loading, preflight rejects traversal and symlink
resolution, non-regular or oversized records, manifest-file hash mismatch,
patch-ID substitution, and base/contract/dependency inconsistencies. It strictly
parses every input and resolved manifest, any rebase/source-manifest lineage
records they carry, every executable target/guard contract, and the resolved
certificate. Contract documents must match their manifest roles, and input plus
resolved manifests may contain at most 100,000 aggregate artifact references. A
hash detects mutation relative to a pinned value; it is not a signature or proof
of publisher identity.

Failed `modelpact rebase` runs may emit `rebase-evidence.json` diagnostic reports.
Those CLI reports are not bare Rebase Evidence v1 and do not establish a rebase
claim. The normative bare record in a successful rebased bundle is
`evidence/rebase.json`, where its lineage and target-base identities are checked
against the source and target manifests and certificate.

Local Hugging Face loading sets `local_files_only=True` and
`trust_remote_code=False`. Users opt into model downloads outside ModelPact.
R1 has no signature or registry infrastructure; a SHA-256 identity detects
mutation but does not identify a publisher.
