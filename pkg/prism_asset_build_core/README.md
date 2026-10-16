# `prism_asset_build_core`

This crate contains deterministic asset-build protocol primitives and the
local offline build core. `BakeService` is a deliberately narrower adapter for
in-process `RuntimeDerived` work:

- it admits a caller-selected bounded number of requests and rejects when full;
- it invokes a caller-provided Bake Kernel with explicit input identity,
  kernel/recipe identity, version, and canonical settings;
- it returns an owned in-memory result with a deterministic request key and
  content descriptor;
- it does not write DDC, CAS, manifests, release packages, or any external
  runtime cache.

`BakeService` is not the offline `BuildExecutor`, an unbounded job queue, a
remote service, or a substitute for a Bake Kernel. Kernels are expected to be
pure and deterministic for their declared inputs. Cancellation is cooperative:
an unpreemptible kernel may finish its current call, but cancellation prevents
result publication when it wins before the service's publication boundary.

The offline build path remains responsible for processors, DDC/CAS, manifest
publication, and packaging. The in-process service only decides admission,
execution handoff, cancellation, and ownership of the returned memory result.
