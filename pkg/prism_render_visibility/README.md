# Prism Render Visibility

Backend-neutral unified visibility contracts and deterministic CPU reference implementation. It consumes GPU Scene snapshots, view-family state, geometry LOD metadata, and Material ABI classification, producing shared visible work for raster, virtual geometry, VSM, GI, ray scenes, picking, and offline rendering.

`benchmark_cpu_reference` provides an in-process throughput harness for fixed
scene/view fixtures. Record CPU baselines after warmup and across multiple
iterations; RenderApp GPU dispatch, parity, overflow, and pipeline readiness are
reported separately by `PrismVisibilityDiagnostics`.
