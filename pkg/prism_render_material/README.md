# Prism Render Material

Versioned Material ABI shared by raster, virtual geometry, shadows, GI, ray tracing, NPR, custom shading, and offline rendering. The package owns stable material identity, normalized closure IR, compact runtime records, validation, and deterministic lowering.

It contains no Unreal Engine source. UE Substrate and MaterialCache are architectural references only; Prism owns its ABI and implementation.

## Generated layout maintenance

`src/surface_layout.rs` is generated from `pkg/prism_material_schema/schema/surface.toml`; do not edit it by hand. The committed CPU layout must remain byte-identical to the schema generator output because the crate's `codegen_freshness` tests guard the CPU layout and GPU decoder twins.

When the schema or generator changes, regenerate the products with:

```text
cargo run -p prism_material_schema --bin prism-material-codegen -- write
```

If a task is scoped to this crate only, preserve the existing `render_scene` shader product and copy or review only the generated `src/surface_layout.rs` change. Verify with:

```text
cargo test -p prism_render_material
```
