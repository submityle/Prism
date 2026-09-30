// Bounded uniform-grid cell hashing, device kernel.
//
// One invocation per particle. Each thread maps its position to a dense linear
// cell index and writes that index as the sort key, plus its own particle index
// as the payload the radix sort will carry alongside the key. The cell math
// mirrors `cpu_cell_index` in `src/grid/cpu.rs` exactly: floor the per-axis
// local coordinate, clamp it into `[0, dim - 1]` in floating point, cast, then
// combine as `x + y * nx + z * nx * ny`. Because the arithmetic is identical
// and integer-valued, the emitted keys match the CPU twin bit-for-bit.
//
// Provenance: bounded uniform grid of Green, "Particle Simulation using CUDA"
// (NVIDIA 2008). No Unreal Engine source or derived code.

struct Params {
    // Grid minimum corner.
    origin_x: f32,
    origin_y: f32,
    origin_z: f32,
    // Uniform cell edge length.
    cell_size: f32,
    // Cell counts along x, y, z.
    nx: u32,
    ny: u32,
    nz: u32,
    // Number of particles.
    n: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
// Particle positions; xyz = position, w unused.
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
// Output: per-particle dense linear cell index (the sort key).
@group(0) @binding(2) var<storage, read_write> keys: array<u32>;
// Output: per-particle identity payload the sort carries with the key.
@group(0) @binding(3) var<storage, read_write> indices: array<u32>;

fn axis_coord(p: f32, origin: f32, dim: u32) -> u32 {
    let hi = f32(dim - 1u);
    let coord = clamp(floor((p - origin) / params.cell_size), 0.0, hi);
    return u32(coord);
}

@compute @workgroup_size(64)
fn hash(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) {
        return;
    }
    let p = positions[i].xyz;
    let cx = axis_coord(p.x, params.origin_x, params.nx);
    let cy = axis_coord(p.y, params.origin_y, params.ny);
    let cz = axis_coord(p.z, params.origin_z, params.nz);
    keys[i] = cx + cy * params.nx + cz * params.nx * params.ny;
    indices[i] = i;
}
