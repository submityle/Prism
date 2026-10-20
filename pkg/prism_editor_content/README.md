# prism_editor_content

This crate provides the renderer- and filesystem-independent part of the
editor content pipeline: a deterministic asset catalog, dependency and
reverse-reference queries, validation, and a bounded import job state machine.

Actual file readers, importers, thumbnail generation, Cook execution, and
version-control providers are adapters above these contracts.
