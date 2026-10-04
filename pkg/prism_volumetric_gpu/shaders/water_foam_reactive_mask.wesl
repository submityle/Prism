// `wgpu` compute twin of the dependency-free `CPU` golden foam reactive mask
// (`prism_render_architecture::water::foam::reactive_mask_into`).
//
// The foam reactive mask is a per-cell visibility predicate: `true` wherever
// foam coverage meets a shading threshold. The renderer uses it to gate the
// foam shading pass to the cells that actually carry foam. This kernel
// evaluates that predicate for a whole grid in one dispatch, keeping the mask
// on device next to the foam field the `water_foam_step` twin evolves there.
//
// The predicate is embarrassingly parallel: one invocation owns one cell and
// reads only that cell's own coverage, so there is no write hazard, no barrier
// and no workgroup memory. Each cell performs the identical single
// floating-point comparison `coverage >= threshold` the golden performs — no
// arithmetic at all — so the readback is exactly bit-identical to the golden.
// Each boolean is stored as a `u32` (`1` set, `0` clear). The kernel uses only
// the portable core-`WGSL` subset (a comparison, `select` and index math), so
// it runs unmodified on Metal, Vulkan and DX12.

// Uniform parameters for one dispatch. Layout matches `Params` in
// `src/water_foam_reactive_mask.rs` (`16` bytes, 16-byte aligned).
struct Params {
    // Number of cells (`>= 1`).
    count: u32,
    // Coverage at or above which a cell is in the mask.
    threshold: f32,
    // Padding to the 16-byte alignment the uniform address space requires.
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
// Row-major per-cell foam coverage.
@group(0) @binding(1) var<storage, read> field: array<f32>;
// Row-major per-cell mask flag (`1` set, `0` clear).
@group(0) @binding(2) var<storage, read_write> out_mask: array<u32>;

@compute @workgroup_size(256)
fn mask_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.count {
        return;
    }
    // Identical single comparison to the golden `field[i] >= threshold`.
    out_mask[idx] = select(0u, 1u, field[idx] >= params.threshold);
}
