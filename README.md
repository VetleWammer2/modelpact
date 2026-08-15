# ModelPact

Compile, compose, rebase, audit, and revert learned behavior.

ModelPact is an experimental system for packaging learned parameter changes with
executable target and preservation contracts. This repository is under active R1
implementation; benchmark claims will be added only from committed, reproducible
artifacts produced by the real tiny-model workflow.

The trust boundary is strict: model adapters are trusted local Python code, while
contracts, patch bundles, manifests, certificates, checkpoints, and lockfiles are
validated as untrusted data. ModelPact does not automatically download models or
enable Hugging Face remote code.

See `SPEC.md` and `TECHNICAL_NOTE.md` for the artifact and research design as they
land. The command-line interface is installed as `modelpact`.
