# Certificates

A certificate records identities, policies, prompt-level execution, intervals,
search budgets, counterexamples, complexity, minimization, interaction/rebase
evidence, environment, warnings, and unsupported claims. Some subsystem fields
can honestly record `NOT_EXECUTED`, `UNMINIMIZED`, or `NOT_APPLICABLE`.

Parsing validates the strict schema, self-hash, referenced hashes when requested,
known claims, and selected claim/evidence consistency; parsing alone does not run
a model. Independent verification re-hashes the caller-declared artifact set and
re-executes through a supplied model-backed provider rather than trusting the
bundled certificate.

Each assertion evidence record carries its executed `acceptance_policy`.
Certificate parsing recomputes pass-rate or continuous-threshold margins from
prompt metrics and permits item failures only when the declared aggregate policy
allows them. A self-hash alone cannot turn an out-of-threshold value into PASS.

The permitted conclusion is: “Verified under the declared contracts, probe
spaces, generation policy, environment, and search budget.”

Certificates name the stable core `patch_id`; independently emitted tools also
pin the bundle's `evidence_id`. A changed evidence file therefore invalidates
the standalone verification input even if the delta and its core `patch_id`
remain unchanged. The full `bundle_id` is the content hash of the final
manifest and is the appropriate identity for byte-complete distribution.
