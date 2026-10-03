// water_spectrum_evolve.wesl
//
// Spectral evolve stage of the separable Tessendorf ocean (entry
// `water_spectrum_evolve`): stage 1 of the three-stage spectral_plan.rs
// schedule (evolve -> butterfly -> assemble). One invocation owns one frequency
// cell (gx = column -> kx index, gy = row -> kz index). It advances the two
// initial amplitudes h0(+k)/h0(-k) to the frame time and packs the eight real
// output-field spectra into the four complex buffers the separable radix-2
// butterfly (water_fft_bitrev / water_fft_stage / water_fft_normalize) consumes.
//
// Shipping oceans (Tessendorf / WaveWorks / Crest / UE5 Water) never direct-sum
// the O(N^4) inverse transform at runtime (that is water_spectrum_ifft, the
// reference golden). They factor it into a separable FFT, and amortise the
// transform by packing two real fields per complex transform: for two real
// spatial fields a, b the packed spectrum A(k) + i B(k) inverse-transforms to
// a(x) + i b(x), so the eight real fields (height, the two choppiness
// displacements, the two slopes, the three displacement gradients) need only
// four complex butterflies. This kernel writes those four packed spectra; the
// assemble stage reads Re -> field A, Im -> field B back out.
//
// The eight fields and their frequency-domain amplitudes per cell are identical
// to the water_spectrum_ifft accumulators (before its phasor sum): height = h,
// slopeX = i*kx*h, slopeZ = i*kz*h, dispX = -i*(kx/|k|)*chop*h, dispZ likewise,
// gradXx = i*kx*dispX, gradZz = i*kz*dispZ, gradXz = i*kz*dispX. The fixed
// field->buffer packing mirrors spectral_plan::field_slot (two consecutive
// fields of SpectralRealField::ALL share one complex buffer, first as real part,
// second as imaginary part):
//   buf0 = height + i*dispX, buf1 = dispZ + i*slopeX,
//   buf2 = slopeZ + i*gradXx, buf3 = gradZz + i*gradXz.
//
// Deterministic classical numerics; no AI/ML. The only float intrinsic is sqrt
// (dispersion); sin/cos use the crate's libm-free sin_approx/cos_approx,
// byte-identical to water_spectrum_ifft.wesl and the Rust super::super::{
// sin_approx, cos_approx} the twin calls.
//
// This is the contract twin of
// `prism_render_architecture::water::gpu::spectrum_evolve_kernel`. The entry
// name matches `WaterKernel::SpectrumEvolve::wesl_entry_point` and the ABI
// matches its descriptor: six storage buffers (two read h0(+k)/h0(-k) complex
// spectra, four written packed complex buffers), one uniform param block, no
// textures, an 8x8 tile, and the Grid2d dispatch domain. The sandbox has no
// GPU, so the crate ships this source, covers it with the structural ABI test,
// and the CPU twin is the cell-for-cell proof; the standalone naga check runs
// out of tree.

const PI: f32 = 3.1415927;
const FRAC_PI_2: f32 = 1.5707964;
const TWO_PI: f32 = 6.2831855;
const GRAVITY: f32 = 9.81;
// Smallest |k| treated as non-zero when forming the choppiness direction k/|k|;
// mirrors synthesis::K_EPS and water_spectrum_ifft.wesl.
const K_EPS: f32 = 1.0e-6;

// Per-dispatch spectral evolve scalars (identical layout to IfftParams).
struct EvolveParams {
    // Grid edge N; the dispatch covers the N by N centred frequency grid.
    resolution: u32,
    // Frame time (s) the amplitudes are advanced to.
    time: f32,
    // Spatial patch size L (m) the grid tiles.
    patch_size: f32,
    // Tessendorf choppiness scale (0 = pure height, 1 = full horizontal drag).
    choppiness: f32,
}

@group(0) @binding(0) var<uniform> params: EvolveParams;
@group(0) @binding(1) var<storage, read> h0: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> h0_neg: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> out0: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> out1: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read_write> out2: array<vec2<f32>>;
@group(0) @binding(6) var<storage, read_write> out3: array<vec2<f32>>;

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

// Pack two complex spectra into one: a + i*b. With a = (ar, ai), b = (br, bi),
// i*b = (-bi, br), so a + i*b = (ar - bi, ai + br). Inverse-transforming this
// buffer yields a(x) + i*b(x) for the Hermitian field pair, which assemble
// unpacks as Re -> field a, Im -> field b.
fn cpack(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x - b.y, a.y + b.x);
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
fn water_spectrum_evolve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = params.resolution;
    if (gid.x >= n || gid.y >= n) {
        return;
    }

    let nf = f32(n);
    let half = nf * 0.5;
    let k_scale = TWO_PI / params.patch_size;

    let idx = gid.y * n + gid.x;
    let len0 = arrayLength(&h0);
    let len1 = arrayLength(&h0_neg);

    // Short/empty spectra: a cell with no backing amplitude evolves to zero in
    // every packed buffer (a deterministic skip, never a garbled read).
    var height = vec2<f32>(0.0, 0.0);
    var disp_x = vec2<f32>(0.0, 0.0);
    var disp_z = vec2<f32>(0.0, 0.0);
    var slope_x = vec2<f32>(0.0, 0.0);
    var slope_z = vec2<f32>(0.0, 0.0);
    var grad_xx = vec2<f32>(0.0, 0.0);
    var grad_zz = vec2<f32>(0.0, 0.0);
    var grad_xz = vec2<f32>(0.0, 0.0);

    if (idx < len0 && idx < len1) {
        let kx = (f32(gid.x) - half) * k_scale;
        let kz = (f32(gid.y) - half) * k_scale;
        let k_mag = sqrt(kx * kx + kz * kz);
        let omega = dispersion(k_mag);
        let h = advance_amplitude(h0[idx], h0_neg[idx], omega, params.time);

        height = h;
        // Surface slopes i*kx*h and i*kz*h (zero at k=0, so unconditional).
        slope_x = cmul(vec2<f32>(0.0, kx), h);
        slope_z = cmul(vec2<f32>(0.0, kz), h);

        if (k_mag > K_EPS) {
            let inv = params.choppiness / k_mag;
            // Tessendorf horizontal displacement -i*(component/|k|)*h.
            let dxh = cmul(vec2<f32>(0.0, -kx * inv), h);
            let dzh = cmul(vec2<f32>(0.0, -kz * inv), h);
            disp_x = dxh;
            disp_z = dzh;
            // Displacement gradients for the folding Jacobian.
            grad_xx = cmul(cmul(vec2<f32>(0.0, kx), dxh), vec2<f32>(1.0, 0.0));
            grad_zz = cmul(cmul(vec2<f32>(0.0, kz), dzh), vec2<f32>(1.0, 0.0));
            grad_xz = cmul(cmul(vec2<f32>(0.0, kz), dxh), vec2<f32>(1.0, 0.0));
        }
    }

    // Pack the eight fields into the four complex buffers (spectral_plan field
    // ordering: height, dispX, dispZ, slopeX, slopeZ, gradXx, gradZz, gradXz).
    if (idx < arrayLength(&out0)) {
        out0[idx] = cpack(height, disp_x);
    }
    if (idx < arrayLength(&out1)) {
        out1[idx] = cpack(disp_z, slope_x);
    }
    if (idx < arrayLength(&out2)) {
        out2[idx] = cpack(slope_z, grad_xx);
    }
    if (idx < arrayLength(&out3)) {
        out3[idx] = cpack(grad_zz, grad_xz);
    }
}
