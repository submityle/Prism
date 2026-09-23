# Prism Render Visibility

Backend-neutral unified visibility contracts and deterministic CPU reference implementation. It consumes GPU Scene snapshots, view-family state, geometry LOD metadata, and Material ABI classification, producing shared visible work for raster, virtual geometry, VSM, GI, ray scenes, picking, and offline rendering.

`benchmark_cpu_reference` provides an in-process throughput harness for fixed
scene/view fixtures. Record CPU baselines after warmup and across multiple
iterations; RenderApp GPU dispatch, parity, overflow, and pipeline readiness are
reported separately by `PrismVisibilityDiagnostics`.

Draw bins are deterministic and include geometry generation, resolved LOD,
pipeline/material class, logical vertex/index buffer classes, topology, and pass
mask. Candidate-table offsets and indirect-command offsets are separate because
sparse scene capacity and live command capacity do not share the same stride.
