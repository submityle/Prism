// GPU MLS-MPM grid velocity update (finalise + gravity + boundary).
//
// One invocation per grid node. Given the accumulated node mass and momentum
// from the P2G scatter, this kernel reproduces the CPU golden grid update that
// the solver runs between P2G and G2P:
//
//   1. finalise: v = mass > 0 ? momentum / mass : 0
//   2. external force: nodes with mass gain v += gravity * dt
//   3. wall boundary: within `thickness` nodes of any domain face, apply the
//      Sticky / Slip / Separate rule to the node velocity.
//
// This is the device twin of `Grid::finalize_velocity`,
// `Grid::add_velocity_to_active(gravity * dt)`, and `solver::apply_grid_boundary`.
//
// Unlike the P2G scatter this is a pure per-node map with no cross-invocation
// contention, so it reads and writes plain `f32` buffers (no fixed-point
// atomics): mass and the packed momentum come in, the finalised velocity goes
// out.
//
// Provenance: the affine PIC grid update and the standard MPM wall boundary
// conditions (Stomakhin et al. 2013; Jiang et al. 2015) are standard, publicly
// documented techniques. No Unreal Engine source or derived code.

// Boundary condition selectors. Must match `BoundaryMode` in the Rust harness.
const MPM_BOUNDARY_STICKY: u32 = 0u;
const MPM_BOUNDARY_SLIP: u32 = 1u;
const MPM_BOUNDARY_SEPARATE: u32 = 2u;

// Uniform parameter block. Mirrors `GridUpdateParams` in the Rust harness.
struct GridUpdateParams {
    // xyz = gravity acceleration, w = time step dt.
    gravity_dt: vec4<f32>,
    // x = nx, y = ny, z = nz, w = node count.
    dims: vec4<u32>,
    // x = boundary thickness (in nodes), y = boundary mode, zw = padding.
    bounds: vec4<u32>,
};

@group(0) @binding(0) var<uniform> params: GridUpdateParams;
@group(0) @binding(1) var<storage, read> grid_mass: array<f32>;
@group(0) @binding(2) var<storage, read> grid_momentum: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> grid_velocity: array<vec4<f32>>;

// Finalises one grid node's velocity, adds gravity, and enforces the wall
// boundary condition. Mirrors the CPU golden grid update exactly, including the
// empty-node skip.
@compute @workgroup_size(64)
fn grid_update(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.dims.w) {
        return;
    }
    let mass = grid_mass[idx];
    if (mass <= 0.0) {
        grid_velocity[idx] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        return;
    }

    // Finalise momentum -> velocity, then apply gravity.
    var v = grid_momentum[idx].xyz / mass;
    v = v + params.gravity_dt.xyz * params.gravity_dt.w;

    // Reconstruct the node's integer coordinate from the flat index
    // (flat = i + nx * (j + ny * k)).
    let nx = params.dims.x;
    let ny = params.dims.y;
    let i = i32(idx % nx);
    let j = i32((idx / nx) % ny);
    let k = i32(idx / (nx * ny));

    let t = i32(params.bounds.x);
    let nxi = i32(nx);
    let nyi = i32(ny);
    let nzi = i32(params.dims.z);

    let low_x = i < t;
    let low_y = j < t;
    let low_z = k < t;
    let high_x = i >= nxi - t;
    let high_y = j >= nyi - t;
    let high_z = k >= nzi - t;

    if (low_x || low_y || low_z || high_x || high_y || high_z) {
        let mode = params.bounds.y;
        if (mode == MPM_BOUNDARY_STICKY) {
            v = vec3<f32>(0.0, 0.0, 0.0);
        } else if (mode == MPM_BOUNDARY_SLIP) {
            if (low_x || high_x) { v.x = 0.0; }
            if (low_y || high_y) { v.y = 0.0; }
            if (low_z || high_z) { v.z = 0.0; }
        } else {
            // Separate: zero the wall-normal component only when it points into
            // the wall.
            if (low_x && v.x < 0.0) { v.x = 0.0; }
            if (high_x && v.x > 0.0) { v.x = 0.0; }
            if (low_y && v.y < 0.0) { v.y = 0.0; }
            if (high_y && v.y > 0.0) { v.y = 0.0; }
            if (low_z && v.z < 0.0) { v.z = 0.0; }
            if (high_z && v.z > 0.0) { v.z = 0.0; }
        }
    }

    grid_velocity[idx] = vec4<f32>(v, 0.0);
}
