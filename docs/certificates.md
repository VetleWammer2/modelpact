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

The permitted conclusion is: “Verified under the declared contracts, probe
spaces, generation policy, environment, and search budget.”
