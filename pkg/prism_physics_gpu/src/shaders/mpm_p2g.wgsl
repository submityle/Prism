// GPU MLS-MPM particle-to-grid (P2G) affine scatter.
//
// Each invocation handles one particle: it evaluates the fixed-corotated
// stress, folds the MLS-MPM internal-force term and the affine (APIC) momentum
// into a single affine matrix, and scatters mass and momentum to the 27
// surrounding grid nodes via fixed-point atomic adds. This is the device twin
// of the CPU golden `transfer::particle_to_grid`.
//
// The host concatenates `mpm_math.wgsl` (the shared pure-function library)
// ahead of this module, so every `mpm_*` helper referenced below is in scope;
// this file adds only the bindings, the fixed-point atomics, and the entry
// point.
//
// Fixed-point accumulation: `f32` atomics are not portable, so mass and each
// momentum component are quantised to `i32` (multiply by a fixed scale, round,
// `atomicAdd`) exactly as the fluid affine scatter does. The host dequantises
// on read-back. The scales are chosen so the accumulated magnitudes stay well
// within `i32` range for the tested configurations.
//
// Provenance: the affine MLS-MPM scatter with the folded stress term (Hu et al.
// 2018; Jiang et al. 2015) and the fixed-corotated model (Stomakhin et al.
// 2013) are standard, publicly documented techniques. No Unreal Engine source
// or derived code.

// Fixed-point scale for accumulated mass.
const MPM_MASS_SCALE: f32 = 65536.0;
// Fixed-point scale for accumulated momentum components.
const MPM_MOMENTUM_SCALE: f32 = 65536.0;

// Uniform parameter block. Mirrors `P2gParams` in the Rust harness.
struct P2gParams {
    // xyz = grid origin, w = cell size dx.
    origin_dx: vec4<f32>,
    // x = shear modulus μ0, y = first Lamé λ0, z = hardening ξ, w = time step dt.
    material: vec4<f32>,
    // x = nx, y = ny, z = nz, w = particle count.
    dims: vec4<u32>,
    // x = plastic-enabled flag (0/1), yzw = padding.
    flags: vec4<u32>,
};

@group(0) @binding(0) var<uniform> params: P2gParams;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> velocities: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> affine: array<mat3x3<f32>>;
@group(0) @binding(4) var<storage, read> deformation: array<mat3x3<f32>>;
@group(0) @binding(5) var<storage, read> masses: array<f32>;
@group(0) @binding(6) var<storage, read> volumes: array<f32>;
@group(0) @binding(7) var<storage, read> plastic_det: array<f32>;
@group(0) @binding(8) var<storage, read_write> grid_mass: array<atomic<i32>>;
@group(0) @binding(9) var<storage, read_write> grid_mom_x: array<atomic<i32>>;
@group(0) @binding(10) var<storage, read_write> grid_mom_y: array<atomic<i32>>;
@group(0) @binding(11) var<storage, read_write> grid_mom_z: array<atomic<i32>>;

// Quantises `value * scale` to the nearest integer for atomic accumulation.
fn mpm_quantise(value: f32, scale: f32) -> i32 {
    return i32(round(value * scale));
}

// Scatters one particle's mass and affine momentum to its 27 neighbouring grid
// nodes. Mirrors the CPU golden P2G inner loop exactly, including the
// out-of-bounds node skip.
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
    if (params.flags.x == 1u) {
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
                atomicAdd(&grid_mass[idx], mpm_quantise(weight * mass, MPM_MASS_SCALE));
                let wm = weight * momentum;
                atomicAdd(&grid_mom_x[idx], mpm_quantise(wm.x, MPM_MOMENTUM_SCALE));
                atomicAdd(&grid_mom_y[idx], mpm_quantise(wm.y, MPM_MOMENTUM_SCALE));
                atomicAdd(&grid_mom_z[idx], mpm_quantise(wm.z, MPM_MOMENTUM_SCALE));
            }
        }
    }
}
