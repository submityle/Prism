// water_spectrum_ifft.wesl
//
// Reference spectral inverse transform (entry `water_spectrum_ifft`): the
// O(N^4) direct-sum IFFT the production butterfly path (water_fft_bitrev /
// water_fft_stage / water_fft_normalize, driven per spectral_plan.rs) is checked
// against. One invocation owns one spatial texel (gx = column -> +x, gy = row ->
// +z) and direct-sums the inverse transform over every frequency cell, so the
// pass is exact for any grid edge N (not just the radix-2 sizes the butterfly
// requires) and depends on nothing but the two h0 spectra and the frame time.
//
// Shipping oceans (Tessendorf / WaveWorks / Crest / UE5 Water) never ship this
// quartic sum at runtime -- they factor the transform into the separable
// radix-2 butterfly. It ships here as the deterministic golden the fast path and
// the CPU synthesis (super::super::synthesis::synthesize_surface) reproduce:
// same centred spectrum, same dispersion, same Hermitian phase advance.
//
// Per texel it evolves every amplitude h(k, t) = h0(+k) e^{+i w t} +
// conj(h0(-k)) e^{-i w t} and accumulates eight real fields via the physical
// phase theta = k . x (k centred at ((j - N/2),(i - N/2)) * 2*PI/L, x = texel *
// L/N): surface height, the two Tessendorf choppiness displacements, the two
// surface slopes (for the normal), and the three horizontal displacement
// gradients (for the folding Jacobian). It writes a displacement texture
// (Dx, height, Dz, Jacobian) and a normal texture (nx, ny, nz, foam).
//
// Deterministic classical numerics; no AI/ML. The only float intrinsic is sqrt
// (dispersion + the normal's normalize); sin/cos use the crate's libm-free
// sin_approx/cos_approx, byte-identical to water_fft_stage.wesl and the Rust
// super::super::{sin_approx, cos_approx} the twin calls.
//
// This is the contract twin of
// `prism_render_architecture::water::gpu::spectrum_ifft_kernel`. The entry name
// matches `WaterKernel::SpectrumIfft::wesl_entry_point` and the ABI matches its
// descriptor: two read storage buffers (the h0(+k)/h0(-k) complex spectra), one
// uniform param block, two rgba32float storage textures (displacement + normal),
// an 8x8 texel tile, and the Grid2d dispatch domain. The sandbox has no GPU, so
// the crate ships this source, covers it with the structural ABI test, and the
// CPU twin is the texel-for-texel proof; the standalone naga check runs out of
// tree.

const PI: f32 = 3.1415927;
const FRAC_PI_2: f32 = 1.5707964;
const TWO_PI: f32 = 6.2831855;
const GRAVITY: f32 = 9.81;
// Smallest |k| treated as non-zero when forming the choppiness direction k/|k|;
// mirrors synthesis::K_EPS.
const K_EPS: f32 = 1.0e-6;
// Normal degeneracy floor; mirrors the crate EPS_LEN_SQ.
const EPS_LEN_SQ: f32 = 1.0e-12;

// Per-dispatch spectral inverse-transform scalars.
struct IfftParams {
    // Grid edge N; the dispatch covers an N by N spatial grid and sums over the
    // N by N centred frequency grid.
    resolution: u32,
    // Frame time (s) the amplitudes are advanced to.
    time: f32,
    // Spatial patch size L (m) the grid tiles.
    patch_size: f32,
    // Tessendorf choppiness scale (0 = pure height, 1 = full horizontal drag).
    choppiness: f32,
}

@group(0) @binding(0) var<uniform> params: IfftParams;
@group(0) @binding(1) var<storage, read> h0: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> h0_neg: array<vec2<f32>>;
@group(0) @binding(3) var out_displacement: texture_storage_2d<rgba32float, write>;
@group(0) @binding(4) var out_normal: texture_storage_2d<rgba32float, write>;

// Round half away from zero, matching Rust `f32::round`.
fn round_rust(x: f32) -> f32 {
    return sign(x) * floor(abs(x) + 0.5);
}

// Reduce an angle to [-PI, PI]; mirrors the golden `wrap_pi`.
fn wrap_pi(x: f32) -> f32 {
    let k = round_rust(x / TWO_PI);
    return x - k * TWO_PI;
}

// Hand-rolled sine via range reduction + seventh-order Taylor; mirrors the
// golden `sin_approx`.
fn sin_approx(x: f32) -> f32 {
    var r = wrap_pi(x);
    if (r > FRAC_PI_2) {
        r = PI - r;
    } else if (r < -FRAC_PI_2) {
        r = -PI - r;
    }
    let x2 = r * r;
    return r * (1.0 + x2 * (-1.0 / 6.0 + x2 * (1.0 / 120.0 + x2 * (-1.0 / 5040.0))));
}

// Hand-rolled cosine via `cos(x) = sin(x + PI/2)`; mirrors `cos_approx`.
fn cos_approx(x: f32) -> f32 {
    return sin_approx(x + FRAC_PI_2);
}

// Complex product, mirroring `Complex::mul`.
fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// Deep-water dispersion `w(k) = sqrt(g*k)`, 0 for non-positive k; mirrors
// `spectrum::dispersion`.
fn dispersion(k: f32) -> f32 {
    if (k <= 0.0) {
        return 0.0;
    }
    return sqrt(GRAVITY * k);
}

// Advance an amplitude pair to time `t`, mirroring `spectrum::advance_amplitude`:
// h0 * e^{+i w t} + conj(h0_neg) * e^{-i w t}.
fn advance_amplitude(h0v: vec2<f32>, h0negv: vec2<f32>, omega: f32, t: f32) -> vec2<f32> {
    let theta = omega * t;
    let forward = cmul(h0v, vec2<f32>(cos_approx(theta), sin_approx(theta)));
    let conj_neg = vec2<f32>(h0negv.x, -h0negv.y);
    let backward = cmul(conj_neg, vec2<f32>(cos_approx(-theta), sin_approx(-theta)));
    return forward + backward;
}

@compute @workgroup_size(8, 8, 1)
fn water_spectrum_ifft(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = params.resolution;
    if (gid.x >= n || gid.y >= n) {
        return;
    }

    let nf = f32(n);
    let half = nf * 0.5;
    let k_scale = TWO_PI / params.patch_size;
    let cell = params.patch_size / nf;
    // Spatial position of this texel (column -> +x, row -> +z).
    let px = f32(gid.x) * cell;
    let pz = f32(gid.y) * cell;

    let len0 = arrayLength(&h0);
    let len1 = arrayLength(&h0_neg);

    // Complex accumulators for the eight real fields.
    var acc_h = vec2<f32>(0.0, 0.0);
    var acc_dx = vec2<f32>(0.0, 0.0);
    var acc_dz = vec2<f32>(0.0, 0.0);
    var acc_sx = vec2<f32>(0.0, 0.0);
    var acc_sz = vec2<f32>(0.0, 0.0);
    var acc_gxx = vec2<f32>(0.0, 0.0);
    var acc_gzz = vec2<f32>(0.0, 0.0);
    var acc_gxz = vec2<f32>(0.0, 0.0);

    var i = 0u;
    while (i < n) {
        let kz = (f32(i) - half) * k_scale;
        var j = 0u;
        while (j < n) {
            let idx = i * n + j;
            // Out-of-range cells contribute zero (short/empty spectra).
            if (idx < len0 && idx < len1) {
                let kx = (f32(j) - half) * k_scale;
                let k_mag = sqrt(kx * kx + kz * kz);
                let omega = dispersion(k_mag);
                let h = advance_amplitude(h0[idx], h0_neg[idx], omega, params.time);

                // Physical phasor e^{+i k . x}.
                let theta = kx * px + kz * pz;
                let phasor = vec2<f32>(cos_approx(theta), sin_approx(theta));
                let hp = cmul(h, phasor);
                acc_h = acc_h + hp;

                // Surface slopes: i*kx*h and i*kz*h, then phasor.
                acc_sx = acc_sx + cmul(cmul(vec2<f32>(0.0, kx), h), phasor);
                acc_sz = acc_sz + cmul(cmul(vec2<f32>(0.0, kz), h), phasor);

                if (k_mag > K_EPS) {
                    let inv = params.choppiness / k_mag;
                    // Tessendorf horizontal displacement: -i * (component/|k|) * h.
                    let dxh = cmul(vec2<f32>(0.0, -kx * inv), h);
                    let dzh = cmul(vec2<f32>(0.0, -kz * inv), h);
                    acc_dx = acc_dx + cmul(dxh, phasor);
                    acc_dz = acc_dz + cmul(dzh, phasor);
                    // Displacement gradients for the folding Jacobian:
                    // dDx/dx = i*kx*(dir_x*h), dDz/dz = i*kz*(dir_z*h),
                    // dDx/dz = i*kz*(dir_x*h) ( = dDz/dx ).
                    acc_gxx = acc_gxx + cmul(cmul(vec2<f32>(0.0, kx), dxh), phasor);
                    acc_gzz = acc_gzz + cmul(cmul(vec2<f32>(0.0, kz), dzh), phasor);
                    acc_gxz = acc_gxz + cmul(cmul(vec2<f32>(0.0, kz), dxh), phasor);
                }
            }
            j = j + 1u;
        }
        i = i + 1u;
    }

    let norm = 1.0 / (nf * nf);
    let height = acc_h.x * norm;
    let disp_x = acc_dx.x * norm;
    let disp_z = acc_dz.x * norm;
    let slope_x = acc_sx.x * norm;
    let slope_z = acc_sz.x * norm;
    let d_dx_dx = acc_gxx.x * norm;
    let d_dz_dz = acc_gzz.x * norm;
    let d_dx_dz = acc_gxz.x * norm;

    // Folding Jacobian and a monotonic foam proxy.
    let jacobian = (1.0 + d_dx_dx) * (1.0 + d_dz_dz) - d_dx_dz * d_dx_dz;
    let foam = max(0.0, 1.0 - jacobian);

    // Surface normal from slopes: normalize((-slope_x, 1, -slope_z)).
    var nx = 0.0;
    var ny = 1.0;
    var nz = 0.0;
    let len_sq = slope_x * slope_x + 1.0 + slope_z * slope_z;
    if (len_sq > EPS_LEN_SQ) {
        let r = 1.0 / sqrt(len_sq);
        nx = -slope_x * r;
        ny = 1.0 * r;
        nz = -slope_z * r;
    }

    let coord = vec2<i32>(i32(gid.x), i32(gid.y));
    textureStore(out_displacement, coord, vec4<f32>(disp_x, height, disp_z, jacobian));
    textureStore(out_normal, coord, vec4<f32>(nx, ny, nz, foam));
}
