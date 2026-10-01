//! The water-surface *meshing* compute pass: the bridge that turns the
//! assembled ocean displacement/normal textures into the four per-vertex
//! storage arrays the raster draw (`water_surface_raster.wesl`) reads back.
//!
//! The spectrum assemble pass (`water_spectrum_assemble` in
//! `water_spectrum_fft.wesl`) ends by writing two `rgba32float` textures: the
//! displacement field `(Dx, height, Dz, J)` and the normal/foam field
//! `(nx, ny, nz, foam)`. The raster vertex stage instead reads *per-vertex
//! storage arrays* (`base_positions`, `surface_uvs`, `displacement`,
//! `normal_foam`) indexed by `@builtin(vertex_index)`. This kernel is the
//! sampling pass that closes that gap: one invocation per surface vertex
//! samples both textures at the vertex's lattice `uv` and scatters the result
//! into the four arrays, exactly the producer→consumer split `WaveWorks`,
//! `Crest` and `UE5` Water run when they build their projected-grid / clipmap
//! surface from the cascade textures.
//!
//! The `CPU` twin of this pass is
//! `prism_render_architecture::water::gpu::surface_mesh`: the output slot
//! numbering (`base_positions`=1, `surface_uvs`=2, `displacement`=3,
//! `normal_foam`=4) mirrors the raster draw's `@group(0)` storage bindings, and
//! the 64-lane linear sweep matches `SURFACE_MESH_LANES`, so the producer here
//! and the consumer in `water_surface_raster.wesl` can never silently disagree.
//!
//! Entry-point name (`water_surface_mesh`) is byte-identical to the `CPU`
//! contract `prism_render_architecture::water::kernels::WaterKernel::SurfaceMesh`
//! so the pipeline cache keys on it.

// Per-dispatch constants describing the surface lattice and its world placement.
struct WaterSurfaceMeshParams {
    // Lattice resolution: vertices along x (x) and z (y), the total vertex
    // count the sweep bounds against (z), and the linear lane width (w,
    // informational; the dispatch is sized host-side).
    grid_dims: vec4<u32>,
    // World-space position of the patch's `(ix=0, iz=0)` corner (xyz); w unused.
    patch_origin: vec4<f32>,
    // Patch world extent along x (x) and z (y), the `uv` tiling scale that maps
    // the lattice onto the cascade textures (z), and the cascade-blend weight
    // applied to the sampled displacement (w).
    patch_extent: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: WaterSurfaceMeshParams;
// Assembled displacement field `(Dx, height, Dz, J)`, sampled at each vertex uv.
@group(0) @binding(1) var src_displacement: texture_2d<f32>;
// Assembled normal/foam field `(nx, ny, nz, foam)`, sampled at each vertex uv.
@group(0) @binding(2) var src_normal: texture_2d<f32>;
// Linear-clamp sampler shared by both source textures.
@group(0) @binding(3) var src_sampler: sampler;
// Undisplaced base lattice position (xyz; w = 1), draw `@binding(1)`.
@group(0) @binding(4) var<storage, read_write> out_base_positions: array<vec4<f32>>;
// Per-vertex texture coordinate (xy; zw pad), draw `@binding(2)`.
@group(0) @binding(5) var<storage, read_write> out_surface_uvs: array<vec4<f32>>;
// Sampled displacement (xyz world offset; w Jacobian), draw `@binding(3)`.
@group(0) @binding(6) var<storage, read_write> out_displacement: array<vec4<f32>>;
// Sampled surface normal (xyz, unit) and foam coverage (w), draw `@binding(4)`.
@group(0) @binding(7) var<storage, read_write> out_normal_foam: array<vec4<f32>>;

// Unit-normalize `v`, falling back to world up for a degenerate (near-zero)
// input so a flat-sea texel never yields a NaN normal.
fn normalize_or_up(v: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    if len_sq > 1.0e-12 {
        return v * inverseSqrt(len_sq);
    }
    return vec3<f32>(0.0, 1.0, 0.0);
}

// Normalized lattice coordinate in `[0, 1]` for index `idx` on an axis with
// `verts` vertices. A single-vertex (or empty) axis collapses to `0`, and the
// denominator is guarded so the discarded branch never divides by zero.
fn lattice_coord(idx: u32, verts: u32) -> f32 {
    let denom = f32(max(verts, 2u) - 1u);
    return select(f32(idx) / denom, 0.0, verts <= 1u);
}

/// Assemble one surface vertex: sample the displacement/normal textures at the
/// vertex's lattice `uv` and scatter the result into the four per-vertex arrays
/// the raster draw reads. A flat 64-lane sweep over the patch vertex count; the
/// tail invocations past `vertex_count` early-out so no out-of-range store runs.
@compute @workgroup_size(64, 1, 1)
fn water_surface_mesh(@builtin(global_invocation_id) gid: vec3<u32>) {
    let vertex_count = params.grid_dims.z;
    let i = gid.x;
    if i >= vertex_count {
        return;
    }

    let verts_x = params.grid_dims.x;
    let verts_z = params.grid_dims.y;
    let ix = i % max(verts_x, 1u);
    let iz = i / max(verts_x, 1u);

    let fx = lattice_coord(ix, verts_x);
    let fz = lattice_coord(iz, verts_z);

    let base = params.patch_origin.xyz
        + vec3<f32>(fx * params.patch_extent.x, 0.0, fz * params.patch_extent.y);
    let uv = vec2<f32>(fx, fz) * params.patch_extent.z;

    let disp = textureSampleLevel(src_displacement, src_sampler, uv, 0.0);
    let norm = textureSampleLevel(src_normal, src_sampler, uv, 0.0);
    let blend = params.patch_extent.w;

    out_base_positions[i] = vec4<f32>(base, 1.0);
    out_surface_uvs[i] = vec4<f32>(uv, 0.0, 0.0);
    out_displacement[i] = vec4<f32>(disp.xyz * blend, disp.w);
    out_normal_foam[i] = vec4<f32>(normalize_or_up(norm.xyz), clamp(norm.w, 0.0, 1.0));
}
