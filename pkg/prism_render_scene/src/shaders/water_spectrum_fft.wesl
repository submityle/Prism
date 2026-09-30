//! `GPU` ocean spectral kernels for the O(N log N) butterfly path: a packed
//! evolve pass that bakes the eight `Tessendorf` spectral fields into four
//! complex grids, and an assemble pass that turns the four inverse-transformed
//! grids back into the displacement/normal textures.
//!
//! This is the production replacement for the O(N²)-per-texel direct summation
//! in `water_ocean.wesl` (`water_spectrum_ifft`). The direct sum evaluates
//! `field(x) = sum_k factor(k) h(k) e^{i k·x}` texel-by-texel, which is O(N⁴)
//! over the whole `N*N` grid. Here the same result is obtained by four separable
//! inverse `FFT`s (each O(N² log N)) driven by `water_butterfly.wesl`:
//!
//! 1. `water_spectrum_evolve` advances every spectral amplitude to the current
//!    frame (the Hermitian phase advance) and multiplies in the eight per-field
//!    spectral factors, then packs them pairwise into four complex grids
//!    (`G0..G3`) written at index `m*N + n`.
//! 2. Each `G_j` is run through the separable butterfly inverse `FFT2`
//!    (`water_butterfly.wesl`: `bitrev` + `log2 N` radix-2 stages per axis, then
//!    one `1/N²` normalize), producing `g_j` at index `px*N + py`.
//! 3. `water_spectrum_assemble` reads the four transformed grids, undoes the
//!    frequency-centering shift and the transform normalization, unpacks the
//!    eight real fields, applies the choppiness scale, and writes the same
//!    displacement `(Dx, height, Dz, J)` and normal `(nx, ny, nz, foam)`
//!    textures the direct sum produced.
//!
//! Packing is legal because all eight spatial fields are real-valued (their
//! spectra are Hermitian: `factor(-k) h(-k) = conj(factor(k) h(k))`), so a
//! single complex inverse transform of `A_hat + i*B_hat` yields `Re = A`,
//! `Im = B`. The four pairings are `G0 = height + i*Dx`, `G1 = Dz + i*slopeX`,
//! `G2 = slopeZ + i*dDxdx`, `G3 = dDzdz + i*dDxdz`.
//!
//! Scaling/shift derivation (must be matched exactly or parity fails): the
//! direct sum uses centered frequencies `k = (m - N/2) dk` with `dk = 2*PI/L`
//! and world position `x = (px, py) * L/N`, so the exponent is
//! `k·x = 2*PI (m·px + n·py)/N - PI (px + py)`. The butterfly transform computes
//! `g(px,py) = (1/N²) sum F e^{i 2*PI (m·px + n·py)/N}`. Hence
//! `field(px,py) = (-1)^(px+py) · N² · g(px,py)`, where `N²` cancels the
//! butterfly's `1/N²` and `(-1)^(px+py) = e^{-i PI (px+py)}` compensates the
//! `N/2` frequency shift.
//!
//! Buffer indexing is consistent end to end: evolve writes `F` at `m*N + n`
//! (`m = gid.x`, `n = gid.y`); the butterfly `ifft2` (row then column, matching
//! the `CPU` `fft::ifft2`) outputs `g` at `px*N + py`; assemble reads `gx*N + gy`
//! and stores texel `(gx, gy)`. The three agree cell-for-cell.
//!
//! Entry-point names (`water_spectrum_evolve`, `water_spectrum_assemble`) are
//! the byte-identical strings the host
//! `prism_render_architecture::water::kernels::WaterKernel::wesl_entry_point`
//! contract emits for the packed spectral variants, so the pipeline cache keys
//! resolve against this module. The two kernels declare disjoint binding ranges
//! in one module (naga validates the whole module in a single compile), so an
//! auto-derived layout materializes only the bindings each entry point touches.

/// Squared-length / wave-number floor below which a `1/k` division is skipped,
/// so nothing divides by ~0 (mirrors the `CPU`
/// `prism_render_architecture::water::EPS_LEN_SQ`).
const WATER_EPS_LEN_SQ: f32 = 1.0e-12;

/// Deep-water gravitational acceleration (m/s²); mirrors the `CPU`
/// `prism_render_architecture::water::GRAVITY`.
const WATER_GRAVITY: f32 = 9.81;

/// `2*PI`, the full-turn constant used to build wave vectors from patch size.
const WATER_TWO_PI: f32 = 6.28318530717958647692;

// -----------------------------------------------------------------------------
// Complex arithmetic (vec2: .x = real, .y = imaginary), mirroring the `CPU`
// `prism_render_architecture::water::spectrum::Complex` methods.
// -----------------------------------------------------------------------------

/// Complex product `(a.re + i*a.im)(b.re + i*b.im)`; mirrors `Complex::mul`.
fn c_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

/// Complex conjugate `re - i*im`; mirrors `Complex::conj`.
fn c_conj(a: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x, -a.y);
}

/// Unit phasor `e^{i*theta} = cos(theta) + i*sin(theta)`; mirrors
/// `Complex::mul_phasor`'s rotor.
fn c_phasor(theta: f32) -> vec2<f32> {
    return vec2<f32>(cos(theta), sin(theta));
}

/// Packs two complex spectral coefficients into one complex grid cell as
/// `a + i*b = (a.re - b.im) + i*(a.im + b.re)`. Legal only for Hermitian `a`,
/// `b` (real spatial fields), which is the case for every field packed here.
fn c_pack(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x - b.y, a.y + b.x);
}

/// Deep-water dispersion `omega(k) = sqrt(g*k)`; mirrors `spectrum::dispersion`.
/// Non-positive `k` yields `0`.
fn water_dispersion(k: f32) -> f32 {
    if k <= 0.0 {
        return 0.0;
    }
    return sqrt(WATER_GRAVITY * k);
}

/// Hermitian phase advance to time `t`; mirrors `spectrum::advance_amplitude`.
/// `h(k, t) = h0 e^{i*omega*t} + conj(h0_neg) e^{-i*omega*t}`, the pairing that
/// keeps the inverse-`FFT` height field real-valued.
fn water_advance_amplitude(h0: vec2<f32>, h0_neg: vec2<f32>, omega: f32, t: f32) -> vec2<f32> {
    let theta = omega * t;
    let forward = c_mul(h0, c_phasor(theta));
    let backward = c_mul(c_conj(h0_neg), c_phasor(-theta));
    return forward + backward;
}

/// Choppy-surface Jacobian `J = (1 + dDx_dx)(1 + dDz_dz) - dDx_dz * dDz_dx`;
/// mirrors `spectrum::surface_jacobian`. `J <= threshold` marks a folded crest.
fn water_surface_jacobian(d_dx_dx: f32, d_dz_dz: f32, d_dx_dz: f32, d_dz_dx: f32) -> f32 {
    return (1.0 + d_dx_dx) * (1.0 + d_dz_dz) - d_dx_dz * d_dz_dx;
}

/// Per-frame constants for the spectral pass. Byte-identical to the
/// `WaterSpectrumParams` in `water_ocean.wesl` and to the host
/// `prism_render_scene::water::abi::GpuWaterSpectrumParams`.
struct WaterSpectrumParams {
    /// Grid resolution `N`; the spectrum buffers are `N*N` complex amplitudes
    /// and the output textures are `N*N` texels.
    grid_size: u32,
    /// Physical patch size `L` (m): the wave vector at cell `(m, n)` is
    /// `2*PI/L * (m - N/2, n - N/2)`.
    patch_size: f32,
    /// Simulation time `t` (s) fed to the Hermitian phase advance.
    time: f32,
    /// Horizontal choppiness scale `lambda` applied to the xz displacement and
    /// its Jacobian derivatives (the `Tessendorf` `-i*k_hat` displacement).
    choppiness: f32,
    /// Fold threshold: `J <= foam_threshold` flags a breaking crest (whitecap).
    foam_threshold: f32,
}

// -----------------------------------------------------------------------------
// Packed evolve kernel (`water_spectrum_evolve`).
//
// Bindings 0..6 belong to this entry point only.
// -----------------------------------------------------------------------------

// Initial complex spectral amplitude `h0(+k)` at grid index `m*N + n`. Read-only.
@group(0) @binding(0) var<storage, read> evolve_h0: array<vec2<f32>>;
// Initial complex spectral amplitude `h0(-k)`, the Hermitian partner. Read-only.
@group(0) @binding(1) var<storage, read> evolve_h0_neg: array<vec2<f32>>;
// Per-frame spectral scalars (grid size, patch size, time, choppiness, foam).
@group(0) @binding(2) var<uniform> evolve_params: WaterSpectrumParams;
// Packed complex grid `G0 = height + i*Dx`, written at `m*N + n`.
@group(0) @binding(3) var<storage, read_write> packed_g0: array<vec2<f32>>;
// Packed complex grid `G1 = Dz + i*slopeX`.
@group(0) @binding(4) var<storage, read_write> packed_g1: array<vec2<f32>>;
// Packed complex grid `G2 = slopeZ + i*dDxdx`.
@group(0) @binding(5) var<storage, read_write> packed_g2: array<vec2<f32>>;
// Packed complex grid `G3 = dDzdz + i*dDxdz`.
@group(0) @binding(6) var<storage, read_write> packed_g3: array<vec2<f32>>;

/// Packed spectral evolve: for every grid cell `(m, n)`, advance the amplitude
/// to the current frame and multiply in the eight per-field spectral factors,
/// then pack the eight complex coefficients into four complex grids for the
/// separable inverse `FFT`. Same `8x8` tiling as the direct-sum kernel.
@compute @workgroup_size(8, 8, 1)
fn water_spectrum_evolve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = evolve_params.grid_size;
    if gid.x >= n || gid.y >= n {
        return;
    }

    let dk = WATER_TWO_PI / max(evolve_params.patch_size, WATER_EPS_LEN_SQ);
    let half = f32(n) * 0.5;

    let m = gid.x;
    let nn = gid.y;
    let idx = m * n + nn;

    let kx = (f32(m) - half) * dk;
    let kz = (f32(nn) - half) * dk;
    let k_sq = kx * kx + kz * kz;

    // Advance the spectral amplitude to the current frame (Hermitian pairing).
    let omega = water_dispersion(sqrt(k_sq));
    let h = water_advance_amplitude(
        evolve_h0[idx],
        evolve_h0_neg[idx],
        omega,
        evolve_params.time,
    );

    // Height uses the unit spectral factor and is always present.
    let height_hat = h;

    // The remaining seven factors vanish at the zero-frequency (k ~ 0) cell.
    var disp_x_hat = vec2<f32>(0.0, 0.0);
    var disp_z_hat = vec2<f32>(0.0, 0.0);
    var slope_x_hat = vec2<f32>(0.0, 0.0);
    var slope_z_hat = vec2<f32>(0.0, 0.0);
    var dxdx_hat = vec2<f32>(0.0, 0.0);
    var dzdz_hat = vec2<f32>(0.0, 0.0);
    var dxdz_hat = vec2<f32>(0.0, 0.0);
    if k_sq > WATER_EPS_LEN_SQ {
        let inv_k = 1.0 / sqrt(k_sq);
        // Choppy displacement factor -i * k_hat (per axis).
        disp_x_hat = c_mul(vec2<f32>(0.0, -kx * inv_k), h);
        disp_z_hat = c_mul(vec2<f32>(0.0, -kz * inv_k), h);
        // Slope factor i * k (per axis) → surface gradient.
        slope_x_hat = c_mul(vec2<f32>(0.0, kx), h);
        slope_z_hat = c_mul(vec2<f32>(0.0, kz), h);
        // Displacement gradients (real spectral factors).
        dxdx_hat = c_mul(vec2<f32>(kx * kx * inv_k, 0.0), h);
        dzdz_hat = c_mul(vec2<f32>(kz * kz * inv_k, 0.0), h);
        dxdz_hat = c_mul(vec2<f32>(kx * kz * inv_k, 0.0), h);
    }

    // Pack pairwise: G0 = height + i*Dx, G1 = Dz + i*slopeX,
    // G2 = slopeZ + i*dDxdx, G3 = dDzdz + i*dDxdz.
    packed_g0[idx] = c_pack(height_hat, disp_x_hat);
    packed_g1[idx] = c_pack(disp_z_hat, slope_x_hat);
    packed_g2[idx] = c_pack(slope_z_hat, dxdx_hat);
    packed_g3[idx] = c_pack(dzdz_hat, dxdz_hat);
}

// -----------------------------------------------------------------------------
// Assemble kernel (`water_spectrum_assemble`).
//
// Reads the four inverse-transformed grids (via `water_butterfly.wesl`) and
// writes the displacement/normal textures. Binding 2 (params) is shared with
// the evolve entry point; bindings 7..12 belong to this entry point only.
// -----------------------------------------------------------------------------

// Inverse-transformed `g0` (= `ifft2(G0)`), indexed `px*N + py`. Read-only.
@group(0) @binding(7) var<storage, read> assemble_g0: array<vec2<f32>>;
// Inverse-transformed `g1`.
@group(0) @binding(8) var<storage, read> assemble_g1: array<vec2<f32>>;
// Inverse-transformed `g2`.
@group(0) @binding(9) var<storage, read> assemble_g2: array<vec2<f32>>;
// Inverse-transformed `g3`.
@group(0) @binding(10) var<storage, read> assemble_g3: array<vec2<f32>>;
// Displacement output: `xyz = (Dx, height, Dz)`, `w` = folding Jacobian `J`.
@group(0) @binding(11) var spectrum_displacement: texture_storage_2d<rgba32float, write>;
// Normal output: `xyz` = world-space surface normal, `w` = whitecap/foam mask.
@group(0) @binding(12) var spectrum_normal: texture_storage_2d<rgba32float, write>;

/// Assemble the ocean surface at one output texel from the four inverse-`FFT`
/// grids: undo the frequency-shift sign and the transform normalization, unpack
/// the eight real fields, apply choppiness, and write the displacement and
/// normal textures identically to the direct-sum kernel.
@compute @workgroup_size(8, 8, 1)
fn water_spectrum_assemble(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = evolve_params.grid_size;
    if gid.x >= n || gid.y >= n {
        return;
    }

    let gx = gid.x;
    let gy = gid.y;
    let idx = gx * n + gy;

    // field = (-1)^(px+py) * N² * g. The N² cancels the butterfly's 1/N²
    // normalization; the sign compensates the N/2 frequency-centering shift.
    let sign = 1.0 - 2.0 * f32((gx + gy) & 1u);
    let s = f32(n * n) * sign;

    let c0 = assemble_g0[idx] * s;
    let c1 = assemble_g1[idx] * s;
    let c2 = assemble_g2[idx] * s;
    let c3 = assemble_g3[idx] * s;

    // Unpack: G0 = height + i*Dx, G1 = Dz + i*slopeX,
    // G2 = slopeZ + i*dDxdx, G3 = dDzdz + i*dDxdz.
    let height = c0.x;
    let disp_x_raw = c0.y;
    let disp_z_raw = c1.x;
    let slope_x = c1.y;
    let slope_z = c2.x;
    let dxdx_raw = c2.y;
    let dzdz_raw = c3.x;
    let dxdz_raw = c3.y;

    let lambda = evolve_params.choppiness;
    let disp_x = lambda * disp_x_raw;
    let disp_z = lambda * disp_z_raw;
    let dxdx = lambda * dxdx_raw;
    let dzdz = lambda * dzdz_raw;
    let dxdz = lambda * dxdz_raw;

    // Folding Jacobian from the (choppiness-scaled) horizontal gradients; the
    // xz-symmetric cross term means dDx/dz == dDz/dx here.
    let jacobian = water_surface_jacobian(dxdx, dzdz, dxdz, dxdz);

    // Surface normal from the height slopes: N = normalize(-dh/dx, 1, -dh/dz).
    let normal = normalize(vec3<f32>(-slope_x, 1.0, -slope_z));

    // Whitecap mask: 1 where the crest has folded past the threshold.
    var foam = 0.0;
    if jacobian <= evolve_params.foam_threshold {
        foam = 1.0;
    }

    let texel = vec2<i32>(i32(gx), i32(gy));
    textureStore(spectrum_displacement, texel, vec4<f32>(disp_x, height, disp_z, jacobian));
    textureStore(spectrum_normal, texel, vec4<f32>(normal, foam));
}
