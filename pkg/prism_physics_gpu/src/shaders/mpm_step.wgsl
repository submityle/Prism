// GPU MLS-MPM full-step orchestrator (clear -> P2G -> grid update -> G2P+clamp).
//
// This fused module reproduces, stage by stage, the arithmetic of the
// standalone `mpm_p2g.wgsl`, `mpm_grid_update.wgsl`, and `mpm_g2p.wgsl`
// kernels, but over one resident set of grid and particle buffers so that a
// multi-step advance never crosses the bus between steps. It is the device
// twin of the CPU golden `MpmSolver::step` (clear grid, particle_to_grid,
// finalize_velocity, add gravity, apply_grid_boundary, grid_to_particle,
// clamp_particles).
//
// The host concatenates `mpm_math.wgsl` (the shared pure-function library)
// ahead of this module, so every `mpm_*` helper referenced below is in scope;
// this file adds only the shared binding scheme and the four entry points.
//
// One shared bind group holds every buffer; each entry point touches only the
// subset it needs. The four entry points run as four separate compute passes so
// the implicit inter-pass memory barrier makes each stage's writes visible to
// the next, reproducing the sequential solver order:
//
//   1. step_clear   - zero the fixed-point atomic grid accumulators.
//   2. p2g_scatter  - affine scatter of mass/momentum into the atomics.
//   3. grid_update  - dequantise, finalise v = p/m, add gravity, wall boundary.
//   4. g2p_gather   - gather velocity + affine C, advect, update F, snow return
//                     mapping, then clamp the position into the domain interior.
//
// Fixed-point accumulation: `f32` atomics are not portable, so mass and each
// momentum component are quantised to `i32` (multiply by a fixed scale, round,
// `atomicAdd`) exactly as the standalone P2G scatter does; the grid update
// dequantises them with `atomicLoad`. The scale is 2^22, matching the
// standalone kernels so the fused step reuses the same verified quantisation.
//
// Provenance: the affine MLS-MPM transfer with the folded stress term (Hu et
// al. 2018; Jiang et al. 2015), the fixed-corotated / snow plasticity model
// (Stomakhin et al. 2013), and the standard MPM wall boundary conditions are
// all standard, publicly documented techniques. No Unreal Engine source or
// derived code.

// Fixed-point scale for accumulated mass. Must match `MASS_SCALE` in the Rust
// harness and the standalone `mpm_p2g.wgsl`.
const MPM_STEP_MASS_SCALE: f32 = 4194304.0;
// Fixed-point scale for accumulated momentum components.
const MPM_STEP_MOMENTUM_SCALE: f32 = 4194304.0;

// Boundary condition selectors. Must match `BoundaryMode` in the Rust harness.
const MPM_STEP_BOUNDARY_STICKY: u32 = 0u;
const MPM_STEP_BOUNDARY_SLIP: u32 = 1u;
const MPM_STEP_BOUNDARY_SEPARATE: u32 = 2u;

// Uniform parameter block. Mirrors `StepParams` in the Rust harness.
struct StepParams {
    // xyz = grid origin, w = cell size dx.
    origin_dx: vec4<f32>,
    // x = shear modulus μ0, y = first Lamé λ0, z = hardening ξ, w = time step dt.
    material: vec4<f32>,
    // xyz = gravity acceleration, w = time step dt (mirrors material.w).
    gravity_dt: vec4<f32>,
    // x = critical compression θc, y = critical stretch θs, zw = padding.
    snow: vec4<f32>,
    // x = nx, y = ny, z = nz, w = particle count.
    dims: vec4<u32>,
    // x = grid node count, yzw = padding.
    grid: vec4<u32>,
    // x = boundary thickness (nodes), y = boundary mode, z = plastic flag (0/1),
    // w = padding.
    bounds: vec4<u32>,
    // xyz = clamp lower corner, w = padding.
    clamp_lo: vec4<f32>,
    // xyz = clamp upper corner, w = padding.
    clamp_hi: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: StepParams;
// Particle state. Positions, velocities, affine C, deformation F, and Jp are
// updated in place by `g2p_gather`; masses and volumes are read only.
@group(0) @binding(1) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> affine: array<mat3x3<f32>>;
@group(0) @binding(4) var<storage, read_write> deformation: array<mat3x3<f32>>;
@group(0) @binding(5) var<storage, read> masses: array<f32>;
@group(0) @binding(6) var<storage, read> volumes: array<f32>;
@group(0) @binding(7) var<storage, read_write> plastic_det: array<f32>;
// Fixed-point atomic grid accumulators.
@group(0) @binding(8) var<storage, read_write> grid_mass: array<atomic<i32>>;
@group(0) @binding(9) var<storage, read_write> grid_mom_x: array<atomic<i32>>;
@group(0) @binding(10) var<storage, read_write> grid_mom_y: array<atomic<i32>>;
@group(0) @binding(11) var<storage, read_write> grid_mom_z: array<atomic<i32>>;
// Finalised node velocity field written by `grid_update`, read by `g2p_gather`.
@group(0) @binding(12) var<storage, read_write> grid_velocity: array<vec4<f32>>;

// Quantises `value * scale` to the nearest integer for atomic accumulation.
fn mpm_step_quantise(value: f32, scale: f32) -> i32 {
    return i32(round(value * scale));
}

// The 3x3 identity used to build `I + dt * C`.
fn mpm_step_identity3() -> mat3x3<f32> {
    return mat3x3<f32>(
        vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(0.0, 0.0, 1.0),
    );
}

// Stage 1: zero the fixed-point atomic grid accumulators. One invocation per
// grid node. Re-run at the start of every step so accumulation starts clean,
// mirroring `Grid::clear`.
@compute @workgroup_size(64)
fn step_clear(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.grid.x) {
        return;
    }
    atomicStore(&grid_mass[idx], 0);
    atomicStore(&grid_mom_x[idx], 0);
    atomicStore(&grid_mom_y[idx], 0);
    atomicStore(&grid_mom_z[idx], 0);
}

// Stage 2: scatter each particle's mass and affine momentum to its 27
// neighbouring grid nodes via fixed-point atomic adds. Mirrors the CPU golden
// P2G inner loop exactly, including the out-of-bounds node skip.
@compute @workgroup_size(64)
fn p2g_scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if (p >= params.dims.w) {
        return;
    }
    let origin = params.origin_dx.xyz;
    let dx = params.origin_dx.w;
    let dt = params.material.w;
    let inv_dx2 = 1.0 / (dx * dx);
    let dinv = 4.0 * inv_dx2;

    let nx = i32(params.dims.x);
    let ny = i32(params.dims.y);
    let nz = i32(params.dims.z);

    let x = positions[p].xyz;
    let v = velocities[p].xyz;
    let mass = masses[p];
    let f = deformation[p];

    var harden: f32 = 1.0;
    if (params.bounds.z == 1u) {
        harden = mpm_hardening_factor(params.material.z, plastic_det[p]);
    }
    let mu = params.material.x * harden;
    let lambda = params.material.y * harden;

    let pf = mpm_corotated_pf(f, mu, lambda);
    let stress = pf * (-dt * volumes[p] * dinv);
    let affine_p = stress + affine[p] * mass;

    let w = mpm_quad_weights(x, origin, dx);
    for (var i: i32 = 0; i < 3; i = i + 1) {
        for (var j: i32 = 0; j < 3; j = j + 1) {
            for (var k: i32 = 0; k < 3; k = k + 1) {
                let ni = w.base.x + i;
                let nj = w.base.y + j;
                let nk = w.base.z + k;
                if (ni < 0 || nj < 0 || nk < 0 || ni >= nx || nj >= ny || nk >= nz) {
                    continue;
                }
                let weight = w.wx[i] * w.wy[j] * w.wz[k];
                let dpos = vec3<f32>(
                    (f32(i) - w.fx.x) * dx,
                    (f32(j) - w.fx.y) * dx,
                    (f32(k) - w.fx.z) * dx,
                );
                let momentum = (v * mass) + affine_p * dpos;
                let idx = ni + nx * (nj + ny * nk);
                atomicAdd(&grid_mass[idx], mpm_step_quantise(weight * mass, MPM_STEP_MASS_SCALE));
                let wm = weight * momentum;
                atomicAdd(&grid_mom_x[idx], mpm_step_quantise(wm.x, MPM_STEP_MOMENTUM_SCALE));
                atomicAdd(&grid_mom_y[idx], mpm_step_quantise(wm.y, MPM_STEP_MOMENTUM_SCALE));
                atomicAdd(&grid_mom_z[idx], mpm_step_quantise(wm.z, MPM_STEP_MOMENTUM_SCALE));
            }
        }
    }
}

// Stage 3: dequantise the atomic accumulators, finalise v = momentum / mass,
// add gravity to nodes with mass, and enforce the wall boundary within
// `thickness` nodes of any face. One invocation per grid node. Mirrors
// `Grid::finalize_velocity`, `Grid::add_velocity_to_active`, and
// `apply_grid_boundary` in that order.
@compute @workgroup_size(64)
fn grid_update(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.grid.x) {
        return;
    }
    let mass = f32(atomicLoad(&grid_mass[idx])) / MPM_STEP_MASS_SCALE;
    if (mass <= 0.0) {
        grid_velocity[idx] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        return;
    }

    let momentum = vec3<f32>(
        f32(atomicLoad(&grid_mom_x[idx])) / MPM_STEP_MOMENTUM_SCALE,
        f32(atomicLoad(&grid_mom_y[idx])) / MPM_STEP_MOMENTUM_SCALE,
        f32(atomicLoad(&grid_mom_z[idx])) / MPM_STEP_MOMENTUM_SCALE,
    );
    var v = momentum / mass;
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
        if (mode == MPM_STEP_BOUNDARY_STICKY) {
            v = vec3<f32>(0.0, 0.0, 0.0);
        } else if (mode == MPM_STEP_BOUNDARY_SLIP) {
            if (low_x || high_x) { v.x = 0.0; }
            if (low_y || high_y) { v.y = 0.0; }
            if (low_z || high_z) { v.z = 0.0; }
        } else {
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

// Stage 4: gather each particle's velocity and affine matrix C from the 27
// surrounding node velocities, advect the position, update the deformation
// gradient, apply the snow return mapping when plasticity is enabled, and clamp
// the advected position into the domain interior. Updates the particle state in
// place. Mirrors the CPU golden `grid_to_particle` followed by
// `clamp_particles`.
@compute @workgroup_size(64)
fn g2p_gather(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if (p >= params.dims.w) {
        return;
    }
    let origin = params.origin_dx.xyz;
    let dx = params.origin_dx.w;
    let dt = params.material.w;
    let dinv = 4.0 / (dx * dx);

    let nx = i32(params.dims.x);
    let ny = i32(params.dims.y);
    let nz = i32(params.dims.z);

    let x = positions[p].xyz;
    let w = mpm_quad_weights(x, origin, dx);

    var vel = vec3<f32>(0.0, 0.0, 0.0);
    var cmat = mat3x3<f32>(
        vec3<f32>(0.0, 0.0, 0.0),
        vec3<f32>(0.0, 0.0, 0.0),
        vec3<f32>(0.0, 0.0, 0.0),
    );
    for (var i: i32 = 0; i < 3; i = i + 1) {
        for (var j: i32 = 0; j < 3; j = j + 1) {
            for (var k: i32 = 0; k < 3; k = k + 1) {
                let ni = w.base.x + i;
                let nj = w.base.y + j;
                let nk = w.base.z + k;
                if (ni < 0 || nj < 0 || nk < 0 || ni >= nx || nj >= ny || nk >= nz) {
                    continue;
                }
                let idx = ni + nx * (nj + ny * nk);
                let gv = grid_velocity[idx].xyz;
                let weight = w.wx[i] * w.wy[j] * w.wz[k];
                let dpos = vec3<f32>(
                    (f32(i) - w.fx.x) * dx,
                    (f32(j) - w.fx.y) * dx,
                    (f32(k) - w.fx.z) * dx,
                );
                vel = vel + weight * gv;
                cmat = cmat + mpm_outer(gv, dpos) * (weight * dinv);
            }
        }
    }

    var new_x = x + dt * vel;
    let f = deformation[p];
    let f_trial = (mpm_step_identity3() + cmat * dt) * f;

    var new_f: mat3x3<f32>;
    var new_jp: f32 = plastic_det[p];
    if (params.bounds.z == 1u) {
        let upd = mpm_snow_return_mapping(f_trial, plastic_det[p], params.snow.x, params.snow.y);
        new_f = upd.deformation;
        new_jp = upd.plastic_det;
    } else {
        new_f = f_trial;
    }

    // Clamp the advected position so its stencil stays inside the domain,
    // matching `clamp_particles`. Only the position is clamped.
    let lo = params.clamp_lo.xyz;
    let hi = params.clamp_hi.xyz;
    new_x = vec3<f32>(
        clamp(new_x.x, lo.x, hi.x),
        clamp(new_x.y, lo.y, hi.y),
        clamp(new_x.z, lo.z, hi.z),
    );

    positions[p] = vec4<f32>(new_x, 0.0);
    velocities[p] = vec4<f32>(vel, 0.0);
    affine[p] = cmat;
    deformation[p] = new_f;
    plastic_det[p] = new_jp;
}
