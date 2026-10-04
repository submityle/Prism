// Ocean initial-spectrum twin: reproduces the dependency-free CPU golden
// `build_initial_spectrum` (in `prism_render_architecture::water::initial_spectrum`)
// on device, one thread per spectral cell.
//
// This closes the last CPU-only link of the spectral-ocean chain. The surface
// synthesis twin (`water_surface_synth`) already runs
// `advance -> ifft2 -> surface` on device from a host-built `h0(k)` / `h0(-k)`;
// this kernel builds that time-zero field itself, so the whole
// `initial_spectrum -> advance -> ifft2 -> surface` path is GPU self-consistent
// and real-device verifiable end to end — the construction shipping oceans
// (`WaveWorks`, `Crest`, UE5 Water) run once per sea state.
//
// `Tessendorf`'s construction draws, for every grid cell, an independent
// complex Gaussian `xi = xi_r + i*xi_i`, then scales it by the square root of
// the (halved) energy density at that cell's wave vector:
//
//   h0(k)  = sqrt(E(k)  / 2) * xi(cell)
//   h0(-k) = sqrt(E(-k) / 2) * xi(mirror-cell)
//
// The `-k` amplitude reuses the *mirror* cell's own Gaussian so the pair is
// Hermitian in the way `advance_amplitude` expects, which keeps the evolved
// height field real. The Gaussian field is deterministic: a `splitmix32`
// integer hash of the cell index and the field seed feeds the classical
// Irwin-Hall / central-limit approximation (twelve uniforms minus six), so the
// same sea state rebuilds identically with no `RNG` and no transcendental.
//
// This step is embarrassingly parallel: one invocation owns one cell, with no
// barrier and no workgroup memory. The mirror cell's Gaussian is recomputed
// independently per thread (the CPU golden pre-draws every cell's Gaussian only
// as an optimisation; `cell_gaussian` is a pure function of the cell index and
// seed, so the GPU reproduces it without any shared storage). Only
// `build_initial_spectrum` (the full-band field, `band == None`) is twinned;
// the cascade band logic stays on the CPU for a later kernel.
//
// Only the single permitted float intrinsic `sqrt` is used; the energy spectra
// go through the hand-rolled `exp_approx` mirrored bit-for-bit from the CPU
// `water::mod` math, since the determinism policy forbids `exp`/`pow`.
// Everything else is integer hash plus multiply/add in the portable core-`WGSL`
// subset — no `sin`, `cos`, `exp`, `pow`, no optional device feature — so the
// kernel runs unmodified on Metal, Vulkan and DX12. WGSL `u32` arithmetic wraps
// on overflow, matching the CPU golden's `wrapping_*` hash ops exactly.
//
// Provenance: 孪生自本仓 `prism_render_architecture::water::initial_spectrum`；
// standard Tessendorf spectral-ocean construction; no Unreal Engine source or
// derived code.

// Two-pi, matching `TAU` used for the wave-number grid scale in the CPU golden
// (`core::f32::consts::TAU`); the high-precision literal rounds to the same f32.
const TWO_PI: f32 = 6.28318530717958647692;

// Squared-length epsilon below which a wave vector carries no energy and the
// wind direction degenerates to zero, matching `water::EPS_LEN_SQ`.
const EPS_LEN_SQ: f32 = 1.0e-12;

// Spectrum-kind discriminants, matching the host `SpectrumKindTag` mapping of
// `prism_render_architecture::water::spectrum::SpectrumKind`.
const KIND_PHILLIPS: u32 = 0u;
const KIND_JONSWAP: u32 = 1u;
const KIND_PIERSON_MOSKOWITZ: u32 = 2u;

// Uniform parameters for one dispatch. Matches `Params` in
// `src/water_initial_spectrum.rs` (48 bytes, 16-byte aligned).
struct Params {
    // Grid resolution `N` (the field holds `N*N` complex amplitudes).
    n: u32,
    // Deterministic field seed.
    seed: u32,
    // Statistical spectrum shape (see the `KIND_*` discriminants).
    kind: u32,
    // Directional-spread exponent; the `|k_hat . w_hat|^{2n}` power.
    directional_exponent: u32,
    // Downwind vector `wind` (its magnitude is the wind speed `V`, m/s).
    wind_x: f32,
    wind_y: f32,
    // Overall energy scale (`Phillips` `A` / `JONSWAP` alpha).
    amplitude: f32,
    // Peak-enhancement factor `gamma` for `JONSWAP`.
    peak_enhancement: f32,
    // Shortest wavelength (m) kept before high-frequency suppression.
    min_wavelength: f32,
    // Spatial patch size `L` (m) the wave-number grid tiles.
    patch_size: f32,
    // Gravitational constant `g`, set to the golden `water::GRAVITY`.
    gravity: f32,
    // Padding to a 16-byte multiple.
    pad: f32,
}

// A complex spectral amplitude, matching `GpuComplex` in the host (8 bytes).
struct Complex {
    re: f32,
    im: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> out_h0: array<Complex>;
@group(0) @binding(2) var<storage, read_write> out_h0_neg: array<Complex>;

// `splitmix32`: a fast, well-mixed integer hash, the deterministic entropy
// source. Pure integer arithmetic; mirrors `splitmix32` in the CPU golden.
fn splitmix32(x0: u32) -> u32 {
    var x = x0;
    x = x ^ (x >> 16u);
    x = x * 0x7feb352du;
    x = x ^ (x >> 15u);
    x = x * 0x846ca68bu;
    x = x ^ (x >> 16u);
    return x;
}

// A uniform in `[0, 1)` from a 24-bit hash draw, matching `uniform_01`.
fn uniform_01(bits: u32) -> f32 {
    return f32(bits >> 8u) / 16777216.0;
}

// One standard-normal draw by the Irwin-Hall central-limit approximation:
// twelve uniforms summed and recentred to zero mean, unit variance. Mirrors
// `standard_gaussian`.
fn standard_gaussian(base: u32, stream: u32) -> f32 {
    var acc = 0.0;
    for (var i = 0u; i < 12u; i = i + 1u) {
        let h = splitmix32(base ^ splitmix32(stream * 0x9e3779b9u + i));
        acc = acc + uniform_01(h);
    }
    return acc - 6.0;
}

// The per-cell complex Gaussian `xi_r + i*xi_i`, deterministic in the cell
// index and the field seed. Mirrors `cell_gaussian`.
fn cell_gaussian(cell: u32, seed: u32) -> vec2<f32> {
    let base = splitmix32((cell + 1u) * 0x27d4eb2fu ^ seed);
    return vec2<f32>(standard_gaussian(base, 0u), standard_gaussian(base, 1u));
}

// Hand-rolled `exp` approximation (2^12 squaring steps of `1 + x/4096`, clamped
// non-negative), mirroring `water::exp_approx` bit-for-bit.
fn exp_approx(x: f32) -> f32 {
    var base = 1.0 + x / 4096.0;
    if (base < 0.0) {
        base = 0.0;
    }
    for (var i = 0u; i < 12u; i = i + 1u) {
        base = base * base;
    }
    return base;
}

// Unit vector, or zero when the squared length is at or below `EPS_LEN_SQ`.
// Mirrors `Vec2::normalize_or_zero`.
fn normalize_or_zero(v: vec2<f32>) -> vec2<f32> {
    let len_sq = dot(v, v);
    if (len_sq > EPS_LEN_SQ) {
        return v * (1.0 / sqrt(len_sq));
    }
    return vec2<f32>(0.0, 0.0);
}

// `|k_hat . w_hat|^{2n}` directional alignment, shared by all three spectra.
fn directional_weight(k_vec: vec2<f32>, k: f32) -> f32 {
    let wind = vec2<f32>(params.wind_x, params.wind_y);
    let wind_dir = normalize_or_zero(wind);
    let k_dir = k_vec * (1.0 / k);
    var align = dot(k_dir, wind_dir);
    if (align < 0.0) {
        align = -align;
    }
    var directional = 1.0;
    for (var i = 0u; i < params.directional_exponent; i = i + 1u) {
        directional = directional * align * align;
    }
    return directional;
}

// `Phillips` energy density at wave vector `k_vec`. Mirrors `phillips`.
fn phillips(k_vec: vec2<f32>) -> f32 {
    let k_sq = dot(k_vec, k_vec);
    if (k_sq < EPS_LEN_SQ) {
        return 0.0;
    }
    let k = sqrt(k_sq);
    let wind = vec2<f32>(params.wind_x, params.wind_y);
    let v = sqrt(dot(wind, wind));
    let big_l = v * v / params.gravity;
    if (big_l <= 0.0) {
        return 0.0;
    }
    let k_l = k * big_l;
    let long_cut = exp_approx(-1.0 / (k_l * k_l));
    let falloff = 1.0 / (k_sq * k_sq);
    let directional = directional_weight(k_vec, k);
    let l = params.min_wavelength;
    let short_cut = exp_approx(-(k * l) * (k * l));
    return params.amplitude * long_cut * falloff * directional * short_cut;
}

// Deep-water dispersion `omega(k) = sqrt(g*k)`; non-positive `k` yields `0`.
fn dispersion(k: f32) -> f32 {
    if (k <= 0.0) {
        return 0.0;
    }
    return sqrt(params.gravity * k);
}

// Pierson-Moskowitz omnidirectional density at angular frequency `omega`.
// Mirrors `pierson_moskowitz`.
fn pierson_moskowitz(omega: f32) -> f32 {
    if (omega <= 0.0) {
        return 0.0;
    }
    let wind = vec2<f32>(params.wind_x, params.wind_y);
    let v = sqrt(dot(wind, wind));
    if (v <= 0.0) {
        return 0.0;
    }
    let omega_p = params.gravity / v;
    let ratio = omega_p / omega;
    let r2 = ratio * ratio;
    let ratio4 = r2 * r2;
    let omega2 = omega * omega;
    let omega5 = omega2 * omega2 * omega;
    let g2 = params.gravity * params.gravity;
    return params.amplitude * g2 / omega5 * exp_approx(-1.25 * ratio4);
}

// `JONSWAP` omnidirectional density at angular frequency `omega`. Mirrors
// `jonswap`.
fn jonswap(omega: f32) -> f32 {
    let base = pierson_moskowitz(omega);
    if (base <= 0.0) {
        return 0.0;
    }
    let wind = vec2<f32>(params.wind_x, params.wind_y);
    let v = sqrt(dot(wind, wind));
    let omega_p = params.gravity / v;
    var sigma = 0.09;
    if (omega <= omega_p) {
        sigma = 0.07;
    }
    let denom = 2.0 * sigma * sigma * omega_p * omega_p;
    let diff = omega - omega_p;
    let r = exp_approx(-(diff * diff) / denom);
    let gamma = params.peak_enhancement;
    let enhancement = 1.0 + (gamma - 1.0) * r;
    return base * enhancement;
}

// Initial energy density for a wave vector under any spectrum. Mirrors
// `energy_density`.
fn energy_density(k_vec: vec2<f32>) -> f32 {
    if (params.kind == KIND_PHILLIPS) {
        return phillips(k_vec);
    }
    let k_sq = dot(k_vec, k_vec);
    if (k_sq < EPS_LEN_SQ) {
        return 0.0;
    }
    let k = sqrt(k_sq);
    let omega = dispersion(k);
    var omni = 0.0;
    if (params.kind == KIND_JONSWAP) {
        omni = jonswap(omega);
    } else {
        omni = pierson_moskowitz(omega);
    }
    let directional = directional_weight(k_vec, k);
    return omni * directional * params.amplitude;
}

@compute @workgroup_size(256)
fn spectrum_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = params.n;
    if (n == 0u) {
        return;
    }
    let cells = n * n;
    let idx = gid.x;
    if (idx >= cells) {
        return;
    }

    let i = idx / n;
    let j = idx % n;
    let half = f32(n) * 0.5;
    let k_scale = TWO_PI / params.patch_size;

    let kz = (f32(i) - half) * k_scale;
    let kx = (f32(j) - half) * k_scale;
    let k_vec = vec2<f32>(kx, kz);

    // +k amplitude: this cell's own Gaussian.
    let energy = energy_density(k_vec);
    let amp = sqrt(max(energy * 0.5, 0.0));
    let g = cell_gaussian(idx, params.seed);
    out_h0[idx] = Complex(amp * g.x, amp * g.y);

    // -k amplitude: the mirror cell's Gaussian; energy is symmetric in k -> -k,
    // so the amplitude matches but the draw differs.
    let mi = (n - i) % n;
    let mj = (n - j) % n;
    let g_neg = cell_gaussian(mi * n + mj, params.seed);
    let energy_neg = energy_density(k_vec * -1.0);
    let amp_neg = sqrt(max(energy_neg * 0.5, 0.0));
    out_h0_neg[idx] = Complex(amp_neg * g_neg.x, amp_neg * g_neg.y);
}
