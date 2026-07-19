# Research Run Manifest

`engine.core.run_manifest` writes self-hashing provenance records. V2 validates
its schema before write and verifies both schema and integrity on read. Use
`read_manifest(path, verify=False)` only for forensic inspection of damaged
files. V1 files remain readable after integrity verification, but are never
treated as promotion evidence because they have no structured test evidence.

V2 promotion requires a clean committed worktree, source hashes, a permitted
holdout status, and at least one successful `test_evidence` record bound to the
manifest commit. Evidence records command, exit code, UTC start/completion,
artifact/JUnit hash, Python version, and dependency-snapshot hash. The legacy
`required_tests_passed` boolean is retained for compatibility but cannot promote.

Artifact paths can be `Path` values or `ArtifactBinding(physical_path,
logical_path)`. Bytes are hashed from the physical staged file; the manifest key
is the normalized relative logical publication path. Traversal, absolute paths,
missing physical files, and duplicate logical paths are rejected.
