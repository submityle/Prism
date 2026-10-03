// water_surface_mesh.wesl
//
// Water-surface meshing compute kernel (entry `water_surface_mesh`). The bridge
// between the spectral/assemble stages (which end by writing *textures*: an
// assembled displacement field and a normal/foam field) and the raster surface
// draw (which reads *per-vertex storage arrays*). One invocation assembles one
// surface vertex of a regular `verts_x` by `verts_z` lattice: it writes the
// undisplaced base position and the patch UV, then point-samples the two
// assembled textures at that UV and scatters the results into the per-vertex
// displacement and normal/foam arrays the draw consumes.
//
// Shipping oceans (WaveWorks / Crest / UE5 Water) run the identical
// producer->consumer split: evaluate the displacement/normal fields into GPU
// textures once per cascade, then sample them while building the
// clipmap/projected-grid vertices. This kernel is that sampling pass factored
// into its own dispatch so the draw node stays a dumb index-buffer executor.
//
// Sampling is a nearest-texel point fetch via `textureLoad` (clamp-to-edge on
// the UV->texel map), chosen so the result is fully deterministic and
// reproducible by the CPU twin without depending on host sampler state. This
// matches a nearest/clamp sampler but removes the sampler-state ambiguity.
//
// Deterministic classical numerics; no AI/ML. No transcendental math (only
// integer bookkeeping, a divide for the UV, and a floor for the texel map).
//
// This is the standalone contract twin of
// `prism_render_architecture::water::gpu::surface_mesh_kernel` (the bit-exact
// CPU reference). The entry-point name is byte-identical to the
// `WaterKernel::SurfaceMesh::wesl_entry_point` contract, and the ABI matches its
// descriptor: four storage buffers (the per-vertex output arrays), one uniform
// param block, two sampled textures (the assembled displacement + normal/foam
// fields), no storage textures, a 64-lane linear tile, and the `Vertices`
// dispatch domain. The sandbox has no GPU, so `prism_render_architecture` (a
// zero-dependency crate) ships this source and covers it with the structural
// ABI test; the standalone naga/wesl compile check runs out of tree, and the
// CPU twin is the vertex-for-vertex proof.

// Per-dispatch surface-meshing scalars.
struct MeshParams {
    // Lattice vertex counts along x and z; the dispatch assembles
    // verts_x * verts_z vertices total.
    verts_x: u32,
    verts_z: u32,
    // Assembled source texture dimensions (texels). A zero dimension disables
    // sampling and writes a zero fetch.
    tex_width: u32,
    tex_height: u32,
    // World spacing between adjacent lattice vertices (m).
    cell_size: f32,
    // World-space origin of the patch's (vx = 0, vz = 0) corner.
    origin_x: f32,
    origin_z: f32,
    // Padding lane for a 16-byte-aligned uniform tail; ignored.
    pad: f32,
}

@group(0) @binding(0) var<uniform> params: MeshParams;
@group(0) @binding(1) var disp_tex: texture_2d<f32>;
@group(0) @binding(2) var normal_tex: texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> base_positions: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> surface_uvs: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> out_displacement: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> out_normal_foam: array<vec4<f32>>;

// Nearest-texel point fetch of `tex` at normalised UV (u, w), clamped to the
// last texel on each axis. A zero-sized texture yields a zero fetch.
fn sample_texel(tex: texture_2d<f32>, u: f32, w: f32) -> vec4<f32> {
    if (params.tex_width == 0u || params.tex_height == 0u) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    let tx = min(u32(floor(u * f32(params.tex_width))), params.tex_width - 1u);
    let ty = min(u32(floor(w * f32(params.tex_height))), params.tex_height - 1u);
    return textureLoad(tex, vec2<i32>(i32(tx), i32(ty)), 0);
}

@compute @workgroup_size(64, 1, 1)
fn water_surface_mesh(@builtin(global_invocation_id) gid: vec3<u32>) {
    let v = gid.x;
    let total = params.verts_x * params.verts_z;
    if (v >= total) {
        return;
    }

    let vx = v % params.verts_x;
    let vz = v / params.verts_x;

    // Undisplaced base lattice position: y = 0 (the still plane), w = 1 so the
    // draw can treat it as a homogeneous point.
    let world_x = params.origin_x + f32(vx) * params.cell_size;
    let world_z = params.origin_z + f32(vz) * params.cell_size;
    base_positions[v] = vec4<f32>(world_x, 0.0, world_z, 1.0);

    // Normalised patch UV in [0, 1]; a single-vertex axis pins its UV to 0 by
    // dividing through a clamped denominator rather than zero.
    let denom_x = max(params.verts_x - 1u, 1u);
    let denom_z = max(params.verts_z - 1u, 1u);
    let u = f32(vx) / f32(denom_x);
    let w = f32(vz) / f32(denom_z);
    surface_uvs[v] = vec4<f32>(u, w, 0.0, 0.0);

    // Point-sample the assembled fields at the vertex UV. Displacement carries
    // xyz world offset (w = foldover/Jacobian); normal/foam carries xyz normal
    // (w = foam coverage). Both pass through unchanged to the draw arrays.
    out_displacement[v] = sample_texel(disp_tex, u, w);
    out_normal_foam[v] = sample_texel(normal_tex, u, w);
}
