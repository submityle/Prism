// 1D radix-2 `Cooley-Tukey` butterfly `FFT` twin: reproduces the dependency-free
// CPU golden `fft`/`ifft` (in `prism_render_architecture::water::fft`) on device,
// one workgroup per transform.
//
// The spectral ocean (`Tessendorf`) synthesises its height field by inverse-
// transforming a grid of time-advanced complex amplitudes. The production path
// is a separable `Cooley-Tukey` butterfly, `O(N log N)` per row/column, which
// this kernel mirrors exactly so a `GPU` ping-pong butterfly pass can be pinned
// against the scalar reference. Each workgroup transforms one length-`n` array
// cooperatively in workgroup memory: it bit-reverses the input, runs `log2(n)`
// butterfly stages separated by `workgroupBarrier`, and (on the inverse path)
// scales by `1/n`, matching the golden's conventions bit for bit:
//
//   fft:  X[k] = sum_n x[n] e^{-i 2*PI k n / N}   (no scale)
//   ifft: x[n] = (1/N) sum_k X[k] e^{+i 2*PI k n / N}
//
// Non-power-of-two (and `n <= 1`) lengths are copied through verbatim, matching
// the golden's "skip, do not crash" contract (and leaving the inverse scale off,
// exactly as `ifft` does for those lengths).
//
// The transcendental `sin`/`cos` are forbidden by the determinism policy, so the
// twiddle phasors use the hand-rolled `sin_approx`/`cos_approx` mirrored bit-for-
// bit from the CPU `water::mod` math (range reduction `wrap_pi` with round-half-
// away ties, fold into [-PI/2, PI/2], seventh-order Taylor). WGSL's native
// `round` rounds ties to even, so `round_ties_away` reproduces Rust's
// `f32::round`. Everything else is multiply/add in the portable core-`WGSL`
// subset — no `sin`, `cos`, `exp`, `pow` — so the kernel runs unmodified on
// Metal, Vulkan and DX12.
//
// Provenance: 孪生自本仓 `prism_render_architecture::water::fft`；standard radix-2
// Cooley-Tukey butterfly；no Unreal Engine source or derived code.

// Angle constants matching `PI`, `TWO_PI`, `FRAC_PI_2` in the CPU math module
// (high-precision literals round to the same f32 bits).
const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;
const FRAC_PI_2: f32 = 1.57079632679489661923;

// Largest transform length the workgroup-shared scratch supports; the host
// asserts `n <= MAX_N`. 256 keeps the thread count portable (<= Metal's and
// Vulkan's conservative workgroup limits) and the scratch at 2 KiB.
const MAX_N: u32 = 256u;

// Uniform parameters for one dispatch. Matches `Params` in `src/water_fft.rs`.
struct Params {
    // Transform length (elements per array). Each workgroup transforms one array.
    n: u32,
    // Non-zero selects the inverse transform (`+` twiddle sign and `1/n` scale).
    inverse: u32,
    // Number of length-`n` arrays laid out back to back in the buffers.
    count: u32,
    // `log2(n)` for the power-of-two path; ignored on the passthrough path.
    bits: u32,
}

// One complex sample. Matches `GpuComplex` in `src/water_fft.rs` (8 bytes).
struct Complex {
    re: f32,
    im: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<Complex>;
@group(0) @binding(2) var<storage, read_write> output: array<Complex>;

// Per-workgroup scratch for one transform, split into real/imag planes.
var<workgroup> sh_re: array<f32, 256>;
var<workgroup> sh_im: array<f32, 256>;

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

// `true` when `n` is a positive power of two, mirroring `fft::is_power_of_two`.
fn is_pow2(n: u32) -> bool {
    return n != 0u && (n & (n - 1u)) == 0u;
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

@compute @workgroup_size(256)
fn fft_main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid3: vec3<u32>,
) {
    let n = params.n;
    let lid = lid3.x;
    let base = wid.x * n;

    // Passthrough for `n <= 1` or non-power-of-two: copy input to output
    // verbatim (no reorder, no butterfly, no inverse scale), matching the
    // golden's deterministic no-op for out-of-contract lengths.
    if (n <= 1u || !is_pow2(n)) {
        if (lid < n) {
            output[base + lid].re = input[base + lid].re;
            output[base + lid].im = input[base + lid].im;
        }
        return;
    }

    // Decimation-in-time load: place `input[i]` at its bit-reversed index, so
    // the shared buffer equals the golden's in-place reorder result.
    if (lid < n) {
        let dst = reverse_bits_u(lid, params.bits);
        sh_re[dst] = input[base + lid].re;
        sh_im[dst] = input[base + lid].im;
    }
    workgroupBarrier();

    // Butterfly stages: `len` doubles from 2 to `n`. One thread drives one of
    // the `n/2` butterflies per stage; those butterflies touch disjoint index
    // pairs, so no intra-stage barrier is needed, only one between stages.
    let sign = select(-1.0, 1.0, params.inverse != 0u);
    var len = 2u;
    loop {
        if (len > n) {
            break;
        }
        let half = len / 2u;
        let num = n / 2u;
        if (lid < num) {
            let group = lid / half;
            let k = lid % half;
            let start = group * len;
            let theta = sign * TWO_PI / f32(len) * f32(k);
            let tw = vec2<f32>(cos_approx(theta), sin_approx(theta));
            let topi = start + k;
            let boti = start + k + half;
            let top = vec2<f32>(sh_re[topi], sh_im[topi]);
            let bottom = cmul(vec2<f32>(sh_re[boti], sh_im[boti]), tw);
            sh_re[topi] = top.x + bottom.x;
            sh_im[topi] = top.y + bottom.y;
            sh_re[boti] = top.x - bottom.x;
            sh_im[boti] = top.y - bottom.y;
        }
        workgroupBarrier();
        len = len * 2u;
    }

    // Store, applying the `1/n` scale only on the inverse path (the forward
    // transform applies none), matching `ifft`/`fft`.
    if (lid < n) {
        var re = sh_re[lid];
        var im = sh_im[lid];
        if (params.inverse != 0u) {
            let inv = 1.0 / f32(n);
            re = re * inv;
            im = im * inv;
        }
        output[base + lid].re = re;
        output[base + lid].im = im;
    }
}
