# Patch format

Delta Program v1 supports low-rank matrices, sparse matrices, vectors, aliases,
and sums. It is data-only and additive. Patch bundles use canonical JSON and
SafeTensors, pin their base state schema and aliases, and are content addressed.
The automatic compiler currently emits low-rank deltas for linear weights; the
broader operations are runtime/materialization capabilities. Complete release
bundles contain contracts, probes, evidence, a certificate, report, and generated
helpers, but low-level bundle APIs require an explicit complete-bundle check.
Checkpoint folding currently loads the full state into host memory. See
[SPEC.md](../SPEC.md) for normative fields and limits.
