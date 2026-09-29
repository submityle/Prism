//! `GPU` ocean surface kernels: spectral inverse-`FFT` evolution + analytic
//! `Gerstner` displacement (compute twin of the `CPU` golden).
//!
//! Compute twin of the classical `Tessendorf` spectral core in
//! `prism_render_architecture::water::spectrum` (the `Phillips` / `JONSWAP` /
//! Pierson-Moskowitz (`PM`) energy densities, the deep-water dispersion
//! `omega(k) = sqrt(g*k)`, the Hermitian phase advance
//! `h(k, t) = h0(k) e^{i*omega*t} + conj(h0(-k)) e^{-i*omega*t}`, and the
//! choppy-surface Jacobian fold test) plus the analytic multi-wave `Gerstner`
//! trochoid sum used for the near-field/interactive tier. The sandbox has no
//! `GPU`, so this shader is validated by naga compilation only (see
//! `shader_tests`); on real hardware the direct-summation inverse `FFT` below
//! is replaced by a separable Cooley-Tukey butterfly `FFT` (the O(N log N)
//! two-pass row/column transform), but the per-texel result is identical.
//!
//! The entry-point names below are byte-identical to the ones the host
//! `prism_render_architecture::water::kernels::WaterKernel::wesl_entry_point`
//! contract emits, so the pipeline cache keys resolve against this module:
//! `water_spectrum_ifft` and `water_gerstner_displace`. Their bind-group
//! layouts differ (the spectral pass binds two spectrum buffers, the `Gerstner`
//! pass binds one wave-train buffer), so the two kernels declare disjoint
//! binding ranges in this one module and each entry point touches only its own
//! subset — naga still validates the whole module in one compile.
//!
//! Buffer/texture layout mirrors `spectrum.rs`: complex spectral amplitudes are
//! `vec2<f32>` (`.x` = real, `.y` = imaginary); the displacement map packs
//! `xyz = (Dx, height, Dz)` with `w` = the folding Jacobian `J`; the normal map
//! packs `xyz` = the world-space surface normal with `w` = the whitecap/foam
//! mask derived from the fold test. Grid texel `(px, py)` maps to the world
//! patch position `(px, py) * patch_size / grid_size`, matching the `CPU` patch
//! sampling.

/// Squared-length / wave-number floor below which normalization or a `1/k`
/// division is skipped, so nothing divides by ~0 (mirrors the `CPU`
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
/// `Complex::mul_phasor`'s rotor. Uses the native `sin`/`cos` (the `CPU`
/// hand-rolls `sin_approx`/`cos_approx` only because it forbids `libm`; the
/// `GPU` intrinsics are the exact same function).
fn c_phasor(theta: f32) -> vec2<f32> {
    return vec2<f32>(cos(theta), sin(theta));
}

/// Deep-water dispersion `omega(k) = sqrt(g*k)`; mirrors
/// `spectrum::dispersion`. Non-positive `k` yields `0`.
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

// -----------------------------------------------------------------------------
// Spectral inverse-`FFT` kernel (`water_spectrum_ifft`).
// -----------------------------------------------------------------------------

/// Per-frame constants for the spectral pass. Mirrors the scalars the `CPU`
/// spectral driver derives once per patch per frame.
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
    /// Fold threshold: `J <= foam_threshold` flags a breaking crest (whitecap),
    /// matching `spectrum::is_wave_folding`.
    foam_threshold: f32,
}

// Initial complex spectral amplitude `h0(+k)` at grid index `m*N + n`
// (`.x` = real, `.y` = imaginary). Read-only input from the `CPU` spectrum bake.
@group(0) @binding(0) var<storage, read> spectrum_h0: array<vec2<f32>>;
// Initial complex spectral amplitude `h0(-k)` at the mirrored index; the
// Hermitian partner that keeps the height field real. Read-only.
@group(0) @binding(1) var<storage, read> spectrum_h0_neg: array<vec2<f32>>;
// Per-frame spectral scalars (grid size, patch size, time, choppiness, foam).
@group(0) @binding(2) var<uniform> spectrum_params: WaterSpectrumParams;
// Displacement output: `xyz = (Dx, height, Dz)`, `w` = folding Jacobian `J`.
@group(0) @binding(3) var spectrum_displacement: texture_storage_2d<rgba32float, write>;
// Normal output: `xyz` = world-space surface normal, `w` = whitecap/foam mask.
@group(0) @binding(4) var spectrum_normal: texture_storage_2d<rgba32float, write>;

/// Spectral inverse-`FFT` pass: advance every spectral amplitude to the current
/// frame, then evaluate the ocean surface at one output texel by direct
/// summation of the inverse transform.
///
/// One invocation owns one spatial texel `(px, py)` of the height/displacement
/// field. It sums over the full `N*N` wave-vector grid:
///
/// * height       `h(x)  = Re Σ H(k, t) e^{i k·x}`,
/// * displacement `D(x)  = Re Σ (-i k_hat) H(k, t) e^{i k·x}` (the choppy xz
///   offset, scaled by `choppiness`),
/// * slope        `∇h    = Re Σ (i k) H(k, t) e^{i k·x}` (→ surface normal), and
/// * the horizontal displacement gradients `dDx/dx`, `dDz/dz`, `dDx/dz`,
///   `dDz/dx` that feed the folding Jacobian `J`.
///
/// The direct O(N²)-per-texel sum is the compilable reference; on real hardware
/// it is replaced by a separable butterfly `FFT` (row transform, then column
/// transform, O(N log N) total). Both compute the identical field; the sum is
/// kept here because it is deterministic and needs no scratch/ping-pong bindings
/// to validate under naga.
@compute @workgroup_size(8, 8, 1)
fn water_spectrum_ifft(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = spectrum_params.grid_size;
    if gid.x >= n || gid.y >= n {
        return;
    }

    let inv_n = 1.0 / max(f32(n), 1.0);
    let cell = spectrum_params.patch_size * inv_n;
    // World patch position of this texel.
    let world_x = f32(gid.x) * cell;
    let world_z = f32(gid.y) * cell;
    // Base spatial frequency step (rad/m) for one grid cell in k-space.
    let dk = WATER_TWO_PI / max(spectrum_params.patch_size, WATER_EPS_LEN_SQ);
    let half = f32(n) * 0.5;

    // Complex accumulators; the real part of each is taken at the end.
    var acc_height = vec2<f32>(0.0, 0.0);
    var acc_disp_x = vec2<f32>(0.0, 0.0);
    var acc_disp_z = vec2<f32>(0.0, 0.0);
    var acc_slope_x = vec2<f32>(0.0, 0.0);
    var acc_slope_z = vec2<f32>(0.0, 0.0);
    var acc_dxdx = vec2<f32>(0.0, 0.0);
    var acc_dzdz = vec2<f32>(0.0, 0.0);
    var acc_dxdz = vec2<f32>(0.0, 0.0);

    for (var m: u32 = 0u; m < n; m = m + 1u) {
        let kx = (f32(m) - half) * dk;
        for (var nn: u32 = 0u; nn < n; nn = nn + 1u) {
            let kz = (f32(nn) - half) * dk;
            let idx = m * n + nn;

            let k_sq = kx * kx + kz * kz;
            // Advance the spectral amplitude to the current frame (Hermitian).
            let omega = water_dispersion(sqrt(k_sq));
            let h = water_advance_amplitude(
                spectrum_h0[idx],
                spectrum_h0_neg[idx],
                omega,
                spectrum_params.time,
            );

            // Inverse-transform kernel e^{i k·x} for this texel.
            let phase = kx * world_x + kz * world_z;
            let hk = c_mul(h, c_phasor(phase));

            // Height: unit spectral factor.
            acc_height = acc_height + hk;

            if k_sq > WATER_EPS_LEN_SQ {
                let inv_k = 1.0 / sqrt(k_sq);
                // Choppy displacement factor -i * k_hat (per axis).
                let disp_fx = vec2<f32>(0.0, -kx * inv_k);
                let disp_fz = vec2<f32>(0.0, -kz * inv_k);
                acc_disp_x = acc_disp_x + c_mul(disp_fx, hk);
                acc_disp_z = acc_disp_z + c_mul(disp_fz, hk);

                // Slope factor i * k (per axis) → surface gradient.
                acc_slope_x = acc_slope_x + c_mul(vec2<f32>(0.0, kx), hk);
                acc_slope_z = acc_slope_z + c_mul(vec2<f32>(0.0, kz), hk);

                // Displacement gradients: d/dx of Dx = (i kx)(-i kx k_hat) etc,
                // which reduce to the real factors below.
                acc_dxdx = acc_dxdx + c_mul(vec2<f32>(kx * kx * inv_k, 0.0), hk);
                acc_dzdz = acc_dzdz + c_mul(vec2<f32>(kz * kz * inv_k, 0.0), hk);
                acc_dxdz = acc_dxdz + c_mul(vec2<f32>(kx * kz * inv_k, 0.0), hk);
            }
        }
    }

    let lambda = spectrum_params.choppiness;
    let height = acc_height.x;
    let disp_x = lambda * acc_disp_x.x;
    let disp_z = lambda * acc_disp_z.x;

    // Folding Jacobian from the (choppiness-scaled) horizontal gradients; the
    // xz-symmetric cross term means dDx/dz == dDz/dx here.
    let dxdx = lambda * acc_dxdx.x;
    let dzdz = lambda * acc_dzdz.x;
    let dxdz = lambda * acc_dxdz.x;
    let jacobian = water_surface_jacobian(dxdx, dzdz, dxdz, dxdz);

    // Surface normal from the height slopes: N = normalize(-dh/dx, 1, -dh/dz).
    let normal = normalize(vec3<f32>(-acc_slope_x.x, 1.0, -acc_slope_z.x));

    // Whitecap mask: 1 where the crest has folded past the threshold.
    var foam = 0.0;
    if jacobian <= spectrum_params.foam_threshold {
        foam = 1.0;
    }

    let texel = vec2<i32>(i32(gid.x), i32(gid.y));
    textureStore(spectrum_displacement, texel, vec4<f32>(disp_x, height, disp_z, jacobian));
    textureStore(spectrum_normal, texel, vec4<f32>(normal, foam));
}

// -----------------------------------------------------------------------------
// Analytic `Gerstner` displacement kernel (`water_gerstner_displace`).
// -----------------------------------------------------------------------------

/// One `Gerstner` (trochoidal) wave train. `Gerstner` waves move surface points
/// in circular/elliptical orbits, sharpening crests and flattening troughs —
/// the classic art-directable near-field ocean (as in UE5's water). Byte layout
/// is 8 `f32` (32 bytes): a unit xz direction, amplitude, wavelength, steepness,
/// a speed multiplier, an initial phase, and one pad lane for 16-byte alignment.
struct GerstnerWave {
    /// Unit wave-propagation direction, x component (xz plane).
    dir_x: f32,
    /// Unit wave-propagation direction, z component (xz plane).
    dir_z: f32,
    /// Crest amplitude `A` (m).
    amplitude: f32,
    /// Wavelength `L` (m); the wave number is `k = 2*PI / L`.
    wavelength: f32,
    /// Steepness `Q` in `[0, 1]`: `0` = round sine wave, `1` = sharp cusp. Kept
    /// per-wave here; the host clamps the sum so crests never self-intersect.
    steepness: f32,
    /// Phase-speed multiplier on the deep-water `omega = sqrt(g*k)` (1 = physical).
    speed: f32,
    /// Constant phase offset (rad) so multiple trains de-correlate.
    phase: f32,
    /// Padding lane for 16-byte struct alignment; ignored.
    _pad: f32,
}

/// Per-frame constants for the `Gerstner` pass.
struct WaterGerstnerParams {
    /// Grid resolution `N`; output textures are `N*N` texels.
    grid_size: u32,
    /// Physical patch size `L` (m) mapping texel `(px, py)` to world xz.
    patch_size: f32,
    /// Simulation time `t` (s) advancing every wave's phase.
    time: f32,
    /// Number of active `GerstnerWave` entries to sum.
    wave_count: u32,
    /// Rest water level (world y, m) added to the summed vertical displacement.
    base_level: f32,
}

// Active `Gerstner` wave trains to superpose (read-only input).
@group(0) @binding(5) var<storage, read> gerstner_waves: array<GerstnerWave>;
// Per-frame `Gerstner` scalars (grid size, patch size, time, wave count, level).
@group(0) @binding(6) var<uniform> gerstner_params: WaterGerstnerParams;
// Displacement output: `xyz = (Dx, y, Dz)` world-space offset (y includes base).
@group(0) @binding(7) var gerstner_displacement: texture_storage_2d<rgba32float, write>;
// Normal output: `xyz` = analytic world-space surface normal, `w` unused (0).
@group(0) @binding(8) var gerstner_normal: texture_storage_2d<rgba32float, write>;

/// Analytic multi-wave `Gerstner` displacement pass: superpose `wave_count`
/// trochoidal wave trains at one output texel, writing the world-space
/// displacement and the closed-form surface normal.
///
/// For each wave with unit direction `D`, wave number `k = 2*PI/L`, angular
/// frequency `omega = speed * sqrt(g*k)`, and steepness `Q`, at rest position
/// `p = (x, z)` and phase `theta = k (D·p) - omega t + phase`:
///
/// * horizontal `+= Q A D cos(theta)` (the trochoidal crest bunching), and
/// * vertical   `+= A sin(theta)`.
///
/// The normal is the exact `Gerstner` closed form (GPU Gems): with `WA = k A`,
/// `n.xz -= D * WA * cos(theta)` and `n.y -= Q * WA * sin(theta)` accumulated
/// from an upright surface, then normalized. Deterministic: a fixed wave set and
/// time produce the same texel on every dispatch.
@compute @workgroup_size(8, 8, 1)
fn water_gerstner_displace(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = gerstner_params.grid_size;
    if gid.x >= n || gid.y >= n {
        return;
    }

    let inv_n = 1.0 / max(f32(n), 1.0);
    let cell = gerstner_params.patch_size * inv_n;
    // Rest-plane xz position sampled by this texel.
    let rest_x = f32(gid.x) * cell;
    let rest_z = f32(gid.y) * cell;

    var disp = vec3<f32>(0.0, 0.0, 0.0);
    // Normal accumulator, starting from the flat upright surface (0, 1, 0).
    var nrm = vec3<f32>(0.0, 1.0, 0.0);

    let count = gerstner_params.wave_count;
    for (var i: u32 = 0u; i < count; i = i + 1u) {
        let w = gerstner_waves[i];
        if w.wavelength <= 0.0 {
            continue;
        }
        // Normalize the stored direction so a non-unit input still behaves.
        let dir_len_sq = w.dir_x * w.dir_x + w.dir_z * w.dir_z;
        if dir_len_sq <= WATER_EPS_LEN_SQ {
            continue;
        }
        let inv_dir_len = 1.0 / sqrt(dir_len_sq);
        let dx = w.dir_x * inv_dir_len;
        let dz = w.dir_z * inv_dir_len;

        let k = WATER_TWO_PI / w.wavelength;
        let omega = w.speed * water_dispersion(k);
        let theta = k * (dx * rest_x + dz * rest_z) - omega * gerstner_params.time + w.phase;
        let c = cos(theta);
        let s = sin(theta);

        let qa = w.steepness * w.amplitude;
        disp.x = disp.x + qa * dx * c;
        disp.z = disp.z + qa * dz * c;
        disp.y = disp.y + w.amplitude * s;

        // Closed-form Gerstner normal contribution (WA = k * A).
        let wa = k * w.amplitude;
        nrm.x = nrm.x - dx * wa * c;
        nrm.z = nrm.z - dz * wa * c;
        nrm.y = nrm.y - w.steepness * wa * s;
    }

    var normal = vec3<f32>(0.0, 1.0, 0.0);
    if dot(nrm, nrm) > WATER_EPS_LEN_SQ {
        normal = normalize(nrm);
    }

    let out_pos = vec3<f32>(disp.x, gerstner_params.base_level + disp.y, disp.z);
    let texel = vec2<i32>(i32(gid.x), i32(gid.y));
    textureStore(gerstner_displacement, texel, vec4<f32>(out_pos, 0.0));
    textureStore(gerstner_normal, texel, vec4<f32>(normal, 0.0));
}
