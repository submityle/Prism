// GPU MLS-MPM grid-to-particle (G2P) affine gather.
//
// Each invocation handles one particle: it gathers the finalised node
// velocities from its 27 surrounding grid nodes into an APIC velocity `vel` and
// affine matrix `C`, advects the particle by `x += dt * vel`, updates the
// deformation gradient `F_trial = (I + dt * C) * F`, and (when plasticity is
// enabled) applies the snow return mapping to split `F_trial` into an elastic
// deformation and a new plastic determinant `Jp`. This is the device twin of
// the CPU golden `transfer::grid_to_particle`.
//
// The host concatenates `mpm_math.wgsl` (the shared pure-function library)
// ahead of this module, so every `mpm_*` helper referenced below is in scope;
// this file adds only the bindings and the entry point.
//
// Unlike the P2G scatter this is a pure per-particle gather with no
// cross-invocation contention, so it reads and writes plain `f32` buffers with
// no fixed-point atomics: the finalised node velocities come in, the updated
// particle state (position, velocity, affine C, deformation F, plastic Jp) goes
// out.
//
// Provenance: the APIC gather (Jiang et al. 2015), the MLS-MPM deformation
// update (Hu et al. 2018), and the snow return mapping (Stomakhin et al. 2013)
// are standard, publicly documented techniques. No Unreal Engine source or
// derived code.

// Uniform parameter block. Mirrors `G2pParams` in the Rust harness.
struct G2pParams {
    // xyz = grid origin, w = cell size dx.
    origin_dx: vec4<f32>,
    // x = time step dt, y = critical compression θc, z = critical stretch θs,
    // w = padding.
    step: vec4<f32>,
    // x = nx, y = ny, z = nz, w = particle count.
    dims: vec4<u32>,
    // x = plastic-enabled flag (0/1), yzw = padding.
    flags: vec4<u32>,
};

@group(0) @binding(0) var<uniform> params: G2pParams;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> deformation: array<mat3x3<f32>>;
@group(0) @binding(3) var<storage, read> plastic_det: array<f32>;
@group(0) @binding(4) var<storage, read> grid_velocity: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> out_positions: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> out_velocities: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> out_affine: array<mat3x3<f32>>;
@group(0) @binding(8) var<storage, read_write> out_deformation: array<mat3x3<f32>>;
@group(0) @binding(9) var<storage, read_write> out_plastic_det: array<f32>;

// The 3x3 identity used to build `I + dt * C`.
fn mpm_identity3() -> mat3x3<f32> {
    return mat3x3<f32>(
        vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(0.0, 0.0, 1.0),
    );
}

// Gathers one particle's velocity and affine matrix from the grid, advects it,
// and updates its deformation gradient. Mirrors the CPU golden G2P inner loop
// exactly, including the out-of-bounds node skip.
@compute @workgroup_size(64)
fn g2p_gather(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if (p >= params.dims.w) {
        return;
    }
    let origin = params.origin_dx.xyz;
    let dx = params.origin_dx.w;
    let dt = params.step.x;
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

    let new_x = x + dt * vel;
    let f = deformation[p];
    let f_trial = (mpm_identity3() + cmat * dt) * f;

    var new_f: mat3x3<f32>;
    var new_jp: f32 = plastic_det[p];
    if (params.flags.x == 1u) {
        let upd = mpm_snow_return_mapping(f_trial, plastic_det[p], params.step.y, params.step.z);
        new_f = upd.deformation;
        new_jp = upd.plastic_det;
    } else {
        new_f = f_trial;
    }

    out_positions[p] = vec4<f32>(new_x, 0.0);
    out_velocities[p] = vec4<f32>(vel, 0.0);
    out_affine[p] = cmat;
    out_deformation[p] = new_f;
    out_plastic_det[p] = new_jp;
}
