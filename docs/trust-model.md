# Trust model

Trusted model adapters are arbitrary local Python and may execute anything the
current user can execute. Load only adapters you trust.

Contracts, patch manifests, delta programs, certificates, SafeTensors metadata,
checkpoint indexes, probe manifests, lockfiles, tensor keys, and output paths are
untrusted data. Parsing them cannot import code, run scripts, use pickle/eval, or
contact a network. Bundle scripts are output artifacts; ModelPact never executes
them merely because they are present.

Local Hugging Face loading sets `local_files_only=True` and
`trust_remote_code=False`. Users opt into model downloads outside ModelPact.
R1 has no signature or registry infrastructure; a SHA-256 identity detects
mutation but does not identify a publisher.

