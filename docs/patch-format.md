# Patch format

Delta Program v1 supports low-rank matrices, sparse matrices, vectors, aliases,
and sums. It is data-only and additive. Patch bundles use canonical JSON and
SafeTensors, pin their base state schema and aliases, and are content addressed.
The automatic compiler currently emits low-rank deltas for linear weights; the
broader operations are runtime/materialization capabilities. Complete release
bundles contain contracts, probes, evidence, a certificate, report, and generated
helpers, but low-level bundle APIs require an explicit complete-bundle check.
Identity has three layers. `patch_id` names the executable delta, contracts,
base compatibility, claims, and lineage. `evidence_id` additionally binds the
probes, evidence, and reports and is pinned by generated tools. `bundle_id`
addresses the final manifest including its certificate and generated helpers.
This avoids certificate hash cycles without allowing claim-bearing evidence to
be silently replaced under the same evidence identity.
Checkpoint folding preflights SafeTensors metadata, aliases, and the delta
program, then reads and writes one deterministic planned output shard at a time.
It does not claim constant memory: patch factors, target deltas, SafeTensors'
write snapshot, and oversized individual tensors remain resident costs. The
materialization manifest records the exact shard bounds, measured I/O, and a
platform high-water RSS value or an explicit unavailable result. See
[SPEC.md](../SPEC.md) for normative semantics and limits.
