// GPU MLS-MPM constitutive probe: evaluates the per-particle constitutive
// kernels in isolation so a real-device parity test can pin the ported
// arithmetic (fixed-corotated stress, polar rotation, snow return-mapping)
// against the CPU golden twin before those kernels are fused into the P2G/G2P
// pipeline.
//
// The host concatenates `mpm_math.wgsl` (the shared pure-function library)
// ahead of this module, so every `mpm_*` helper referenced below is already in
// scope; this file adds only the bindings and the single compute entry point.
//
// Provenance: the fixed-corotated energy and snow return-mapping (Stomakhin et
// al. 2013) and the affine MLS-MPM transfer conventions (Hu et al. 2018;
// Jiang et al. 2015) are standard, publicly documented techniques. No Unreal
// Engine source or derived code.

// Uniform parameter block. Mirrors `ProbeParams` in the Rust harness.
struct ProbeParams {
    // x = particle count, y = plastic-enabled flag (0/1), zw = padding.
    counts: vec4<u32>,
    // x = shear modulus μ0, y = first Lamé λ0, z = hardening ξ, w = padding.
    mat: vec4<f32>,
    // x = critical compression θc, y = critical stretch θs, zw = padding.
    plast: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: ProbeParams;
@group(0) @binding(1) var<storage, read> f_in: array<mat3x3<f32>>;
@group(0) @binding(2) var<storage, read> jp_in: array<f32>;
@group(0) @binding(3) var<storage, read_write> pf_out: array<mat3x3<f32>>;
@group(0) @binding(4) var<storage, read_write> polar_out: array<mat3x3<f32>>;
@group(0) @binding(5) var<storage, read_write> f_elastic_out: array<mat3x3<f32>>;
@group(0) @binding(6) var<storage, read_write> jp_out: array<f32>;

// Evaluates the constitutive kernels for one particle per invocation. The
// hardening branch mirrors the CPU P2G path: when plasticity is enabled the
// Lamé parameters are scaled by `hardening_factor(ξ, Jp)`, otherwise they are
// used as-is. The stress `P Fᵀ`, the polar rotation `R`, and the snow
// return-mapping (corrected elastic `F` and new `Jp`) are written to their
// respective output buffers.
@compute @workgroup_size(64)
fn constitutive_probe(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if (p >= params.counts.x) {
        return;
    }
    let f = f_in[p];
    var harden: f32 = 1.0;
    if (params.counts.y == 1u) {
        harden = mpm_hardening_factor(params.mat.z, jp_in[p]);
    }
    let mu = params.mat.x * harden;
    let lambda = params.mat.y * harden;
    pf_out[p] = mpm_corotated_pf(f, mu, lambda);
    polar_out[p] = mpm_polar_rotation(f);
    let upd = mpm_snow_return_mapping(f, jp_in[p], params.plast.x, params.plast.y);
    f_elastic_out[p] = upd.deformation;
    jp_out[p] = upd.plastic_det;
}
