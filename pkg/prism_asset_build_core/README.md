# `prism_asset_build_core`

This crate contains deterministic asset-build protocol primitives and the
local offline build core. `BakeService` is a deliberately narrower adapter for
in-process `RuntimeDerived` work:

- it admits a caller-selected bounded number of requests and rejects when full;
- it invokes a caller-provided Bake Kernel with explicit input identity,
  kernel/recipe identity, version, and canonical settings;
- it returns an owned in-memory result with a deterministic request key and
  content descriptor;
- it can use an explicitly injected, thread-safe kernel registry keyed by the
  stable `kernel_id` in the request;
- it can retain successful results in an explicitly bounded process-local
  runtime cache, limited by both `max_entries` and `max_bytes`;
- it never writes DDC, CAS, manifests, release packages, or remote services.

The runtime cache is best-effort and non-persistent. A cache hit is keyed only
by the deterministic `ArtifactKey` derived from the request. Failed, panicked,
or cancelled jobs are never inserted. Concurrent requests for the same key
share one kernel execution; each caller has its own cancellation subscription,
and cancellation removes only that subscription unless it is the last active
subscriber. Eviction is deterministic: while either bound is exceeded, the
lexicographically largest `ArtifactKey` is removed. A zero cache limit
disables storage rather than meaning unbounded capacity.

`BakeService::new` remains a compatibility entry point with no cache and a
single directly supplied kernel. Use `new_with_cache` for bounded caching and
`new_with_registry` when unknown stable kernel IDs must be rejected. The
registry is caller-owned and injected; there is no implicit global registry.

`BakeService` is not the offline `BuildExecutor`, an unbounded job queue, a
remote service, or a substitute for a Bake Kernel. Kernels are expected to be
pure and deterministic for their declared inputs. Cancellation is cooperative:
an unpreemptible kernel may finish its current call, but cancellation prevents
result publication when it wins before the service's publication boundary.

The offline build path remains responsible for processors, DDC/CAS, manifest
publication, and packaging. The in-process service only decides admission,
execution handoff, cancellation, and ownership of the returned memory result.
