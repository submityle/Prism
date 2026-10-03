// Ocean surface-synthesis twin: reproduces the dependency-free CPU golden
// `synthesize_surface` (in `prism_render_architecture::water::synthesis`) on
// device, one workgroup per surface patch.
//
// This is the step that turns a spectral ocean into something an author can
// actually see: it evolves the time-zero complex amplitudes `h0(k)` / `h0(-k)`
// to `h(k, t)`, inverse-transforms that frequency grid into a spatial height
// field, and adds the horizontal `Tessendorf` choppiness displacement — the
// same chain shipping oceans (`WaveWorks`, `Crest`, UE5 Water) run every frame.
// The earlier `water_fft` twin pinned the 1D butterfly in isolation; this twin
// pins the whole `initial_spectrum -> spectrum_advance -> ifft2 -> surface`
// chain end to end, so the spectral path is validated as a unit, not just per
// operator.
//
// Each workgroup synthesises one `n x n` patch cooperatively in workgroup
// memory (`n <= 32`, so `n*n <= 1024` threads). The three output fields
// (height, +x displacement, +z displacement) each go through the same separable
// inverse `FFT` plus the `(-1)^(row+col)` recentring sign, mirroring the golden
// bit for bit:
//
//   advance: h(k,t) = h0(k) e^{+i w t} + conj(h0(-k)) e^{-i w t}
//   height:  real( ifft2( h(k,t) ) ) * (-1)^(row+col)
//   choppy:  real( ifft2( -i (k/|k|) h(k,t) ) ) * (-1)^(row+col)
//
// The 2D inverse is separable: a raw butterfly over every row, a raw butterfly
// over every column, then a single `1/(n*n)` scale — exactly matching the CPU
// `fft::ifft2`'s `transform2`. The dispersion `omega = sqrt(g*k)` uses the one
// permitted float intrinsic `sqrt`; every phasor goes through the hand-rolled
// `sin_approx`/`cos_approx` mirrored bit-for-bit from the CPU `water::mod` math
// (range reduction `wrap_pi` with round-half-away ties, fold into
// `[-PI/2, PI/2]`, seventh-order Taylor), since the determinism policy forbids
// the hardware transcendentals. WGSL's native `round` rounds ties to even, so
// `round_ties_away` reproduces Rust's `f32::round`. Everything else is
// multiply/add plus `workgroupBarrier` in the portable core-`WGSL` subset — no
// `sin`, `cos`, `exp`, `pow` — so the kernel runs unmodified on Metal, Vulkan
// and DX12.
//
// Provenance: 孪生自本仓 `prism_render_architecture::water::synthesis`；standard
// separable Cooley-Tukey ocean synthesis; no Unreal Engine source or derived
// code.

// Angle constants matching `PI`, `TWO_PI`, `FRAC_PI_2` in the CPU math module
// (high-precision literals round to the same f32 bits).
const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;
const FRAC_PI_2: f32 = 1.57079632679489661923;

// Smallest wave-number magnitude treated as non-zero when forming the
// horizontal-displacement direction `k / |k|`, matching `synthesis::K_EPS`.
const K_EPS: f32 = 1.0e-6;

// Largest resolution the single-workgroup scratch supports: `n <= 32` keeps the
// thread count `n*n <= 1024` (within Metal's and Vulkan's workgroup limits) and
// the four scratch planes at 16 KiB. The host asserts `n <= MAX_N`.
const MAX_N: u32 = 32u;

// Uniform parameters for one dispatch. Matches `Params` in
// `src/water_surface_synth.rs`.
struct Params {
    // Patch resolution `N` (each patch holds `N*N` samples).
    n: u32,
    // `log2(n)`, the number of butterfly stages per 1D transform.
    bits: u32,
    // Number of `n*n` patches laid out back to back (one workgroup each).
    count: u32,
    // Padding to a 16-byte boundary.
    pad: u32,
    // Spatial patch size `L` (m) the grid tiles.
    patch_size: f32,
    // Evolution time `t` (s).
    time: f32,
    // Horizontal choppiness scale (`0` = pure height, `1` = full Tessendorf).
    choppiness: f32,
    // Gravitational acceleration `g` feeding the dispersion `sqrt(g*k)`.
    gravity: f32,
}

// One complex sample. Matches `GpuComplex` in `src/water_surface_synth.rs`.
struct Complex {
    re: f32,
    im: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> h0: array<Complex>;
@group(0) @binding(2) var<storage, read> h0_neg: array<Complex>;
@group(0) @binding(3) var<storage, read_write> out_height: array<f32>;
@group(0) @binding(4) var<storage, read_write> out_disp_x: array<f32>;
@group(0) @binding(5) var<storage, read_write> out_disp_z: array<f32>;

// Per-workgroup scratch for one patch: the evolved spectrum (held across all
// three fields) and the live transform buffer, each split into real/imag planes.
var<workgroup> spec_re: array<f32, 1024>;
var<workgroup> spec_im: array<f32, 1024>;
var<workgroup> work_re: array<f32, 1024>;
var<workgroup> work_im: array<f32, 1024>;

// Round halves away from zero, matching Rust's `f32::round`. WGSL's native
// `round` rounds ties to even, so it cannot be used here.
fn round_ties_away(v: f32) -> f32 {
    let r = floor(abs(v) + 0.5);
    return select(r, -r, v < 0.0);
}

// Reduce an angle to [-PI, PI], mirroring `math::wrap_pi`.
fn wrap_pi(x: f32) -> f32 {
    let kk = round_ties_away(x / TWO_PI);
    return x - kk * TWO_PI;
}

// Hand-rolled sine, mirroring `water::sin_approx` bit for bit.
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

// Hand-rolled cosine via `cos(x) = sin(x + PI/2)`, mirroring `water::cos_approx`.
fn cos_approx(x: f32) -> f32 {
    return sin_approx(x + FRAC_PI_2);
}

// Complex product `(a.x + i a.y)(b.x + i b.y)`, mirroring `Complex::mul`.
fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// Reverse the low `bits` bits of `x`, mirroring `fft::reverse_bits`.
fn reverse_bits_u(x: u32, bits: u32) -> u32 {
    var v = x;
    var r = 0u;
    var b = 0u;
    loop {
        if (b >= bits) {
            break;
        }
        r = (r << 1u) | (v & 1u);
        v = v >> 1u;
        b = b + 1u;
    }
    return r;
}

// Separable inverse `FFT` of the `n x n` grid held in `work_re`/`work_im`,
// mirroring `fft::ifft2`'s `transform2`: a raw inverse butterfly over every row,
// a raw inverse butterfly over every column, then one `1/(n*n)` scale. Called
// unconditionally by all invocations, so every `workgroupBarrier` sits in
// uniform control flow; the stage loop runs `log2(n)` times (driven by uniform
// `n`), and the per-element work is guarded without guarding the barriers.
fn ifft2_shared(lid: u32, n: u32, bits: u32) {
    let total = n * n;
    let in_range = lid < total;
    let row = lid / n;
    let col = lid % n;
    let half_threads = total / 2u;

    // ---- Row pass: inverse transform each row along the column index ----
    // Decimation-in-time reorder: swap each element with its bit-reversed
    // partner once (the smaller index drives the disjoint-pair swap).
    if (in_range) {
        let rev = reverse_bits_u(col, bits);
        if (rev > col) {
            let a = row * n + col;
            let b = row * n + rev;
            let tr = work_re[a];
            let ti = work_im[a];
            work_re[a] = work_re[b];
            work_im[a] = work_im[b];
            work_re[b] = tr;
            work_im[b] = ti;
        }
    }
    workgroupBarrier();

    var len = 2u;
    loop {
        if (len > n) {
            break;
        }
        let half = len / 2u;
        let per_row = n / 2u;
        if (lid < half_threads) {
            let brow = lid / per_row;
            let kk = lid % per_row;
            let group = kk / half;
            let k = kk % half;
            let start = brow * n + group * len;
            let theta = TWO_PI / f32(len) * f32(k);
            let tw = vec2<f32>(cos_approx(theta), sin_approx(theta));
            let topi = start + k;
            let boti = start + k + half;
            let top = vec2<f32>(work_re[topi], work_im[topi]);
            let bottom = cmul(vec2<f32>(work_re[boti], work_im[boti]), tw);
            work_re[topi] = top.x + bottom.x;
            work_im[topi] = top.y + bottom.y;
            work_re[boti] = top.x - bottom.x;
            work_im[boti] = top.y - bottom.y;
        }
        workgroupBarrier();
        len = len * 2u;
    }

    // ---- Column pass: inverse transform each column along the row index ----
    if (in_range) {
        let rev = reverse_bits_u(row, bits);
        if (rev > row) {
            let a = row * n + col;
            let b = rev * n + col;
            let tr = work_re[a];
            let ti = work_im[a];
            work_re[a] = work_re[b];
            work_im[a] = work_im[b];
            work_re[b] = tr;
            work_im[b] = ti;
        }
    }
    workgroupBarrier();

    len = 2u;
    loop {
        if (len > n) {
            break;
        }
        let half = len / 2u;
        let per_col = n / 2u;
        if (lid < half_threads) {
            let bcol = lid / per_col;
            let kk = lid % per_col;
            let group = kk / half;
            let k = kk % half;
            let startrow = group * len;
            let theta = TWO_PI / f32(len) * f32(k);
            let tw = vec2<f32>(cos_approx(theta), sin_approx(theta));
            let topi = (startrow + k) * n + bcol;
            let boti = (startrow + k + half) * n + bcol;
            let top = vec2<f32>(work_re[topi], work_im[topi]);
            let bottom = cmul(vec2<f32>(work_re[boti], work_im[boti]), tw);
            work_re[topi] = top.x + bottom.x;
            work_im[topi] = top.y + bottom.y;
            work_re[boti] = top.x - bottom.x;
            work_im[boti] = top.y - bottom.y;
        }
        workgroupBarrier();
        len = len * 2u;
    }

    // Single inverse scale `1/(n*n)`, matching `transform2`'s one normalisation.
    if (in_range) {
        let inv = 1.0 / f32(total);
        work_re[lid] = work_re[lid] * inv;
        work_im[lid] = work_im[lid] * inv;
    }
    workgroupBarrier();
}

@compute @workgroup_size(1024)
fn synth_main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid3: vec3<u32>,
) {
    let n = params.n;
    let bits = params.bits;
    let lid = lid3.x;
    let total = n * n;
    let base = wid.x * total;
    let in_range = lid < total;
    let row = lid / n;
    let col = lid % n;

    // Centred wave vector for this texel: k = ((col - N/2), (row - N/2)) * 2PI/L.
    let halfn = f32(n) * 0.5;
    let k_scale = TWO_PI / params.patch_size;
    let kz = (f32(row) - halfn) * k_scale;
    let kx = (f32(col) - halfn) * k_scale;
    let k_mag = sqrt(kx * kx + kz * kz);

    // Deep-water dispersion omega = sqrt(g*k); zero for the DC term, matching
    // `spectrum::dispersion`'s `k <= 0` guard.
    var omega = 0.0;
    if (k_mag > 0.0) {
        omega = sqrt(params.gravity * k_mag);
    }

    // Evolve the amplitude to `time`:
    // h(k,t) = h0(k) e^{+i w t} + conj(h0(-k)) e^{-i w t}, matching
    // `spectrum::advance_amplitude` (`mul_phasor(theta)` / `mul_phasor(-theta)`).
    if (in_range) {
        let theta = omega * params.time;
        let ph = vec2<f32>(cos_approx(theta), sin_approx(theta));
        let phn = vec2<f32>(cos_approx(-theta), sin_approx(-theta));
        let a = vec2<f32>(h0[base + lid].re, h0[base + lid].im);
        let bn = vec2<f32>(h0_neg[base + lid].re, h0_neg[base + lid].im);
        let forward = cmul(a, ph);
        let conj_bn = vec2<f32>(bn.x, -bn.y);
        let backward = cmul(conj_bn, phn);
        spec_re[lid] = forward.x + backward.x;
        spec_im[lid] = forward.y + backward.y;
    }
    workgroupBarrier();

    // ---- Vertical height: real( ifft2(spectrum) ) * (-1)^(row+col) ----
    if (in_range) {
        work_re[lid] = spec_re[lid];
        work_im[lid] = spec_im[lid];
    }
    workgroupBarrier();
    ifft2_shared(lid, n, bits);
    if (in_range) {
        var sign = 1.0;
        if (((row + col) & 1u) != 0u) {
            sign = -1.0;
        }
        out_height[base + lid] = work_re[lid] * sign;
    }
    workgroupBarrier();

    // ---- Horizontal +x choppiness: real( ifft2(-i (kx/|k|) h) ) * sign ----
    if (in_range) {
        if (k_mag > K_EPS) {
            let inv = params.choppiness / k_mag;
            let dir = vec2<f32>(0.0, -kx * inv);
            let s = vec2<f32>(spec_re[lid], spec_im[lid]);
            let v = cmul(dir, s);
            work_re[lid] = v.x;
            work_im[lid] = v.y;
        } else {
            work_re[lid] = 0.0;
            work_im[lid] = 0.0;
        }
    }
    workgroupBarrier();
    ifft2_shared(lid, n, bits);
    if (in_range) {
        var sign = 1.0;
        if (((row + col) & 1u) != 0u) {
            sign = -1.0;
        }
        out_disp_x[base + lid] = work_re[lid] * sign;
    }
    workgroupBarrier();

    // ---- Horizontal +z choppiness: real( ifft2(-i (kz/|k|) h) ) * sign ----
    if (in_range) {
        if (k_mag > K_EPS) {
            let inv = params.choppiness / k_mag;
            let dir = vec2<f32>(0.0, -kz * inv);
            let s = vec2<f32>(spec_re[lid], spec_im[lid]);
            let v = cmul(dir, s);
            work_re[lid] = v.x;
            work_im[lid] = v.y;
        } else {
            work_re[lid] = 0.0;
            work_im[lid] = 0.0;
        }
    }
    workgroupBarrier();
    ifft2_shared(lid, n, bits);
    if (in_range) {
        var sign = 1.0;
        if (((row + col) & 1u) != 0u) {
            sign = -1.0;
        }
        out_disp_z[base + lid] = work_re[lid] * sign;
    }
}
