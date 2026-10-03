// Water ocean-spectrum inverse-FFT radix-2 butterfly stage compute kernel
// (entry point `water_fft_stage`).
//
// After the bit-reversal reorder, the separable inverse `Cooley-Tukey` transform
// runs `log2(n)` butterfly stages per axis, the span doubling 2, 4, …, N. On the
// `GPU` each stage is one out-of-place ping-pong pass: every invocation writes
// its own line position by combining the two source elements of its butterfly
// pair with the stage twiddle. For a stage of span `len` (half = len/2) the line
// splits into groups of `len`; position `s + k` (k in 0..half) is the pair's
// "top" and `s + k + half` its "bottom", and the inverse butterfly is
//   out[s + k]        = top + W_k * bottom
//   out[s + k + half] = top - W_k * bottom,  W_k = e^{+i 2*PI k / len}.
//
// This mirrors the butterfly loop of the CPU golden `fft::transform` (its
// `inverse` branch, sign = +1) element-for-element: the complex multiply and
// add match `Complex::mul`/`Complex::add`, and the twiddle phasor uses the
// crate's hand-rolled `sin_approx`/`cos_approx` so the pass stays `libm`-free.
// The one subtlety is the range-reduction round in `wrap_pi`: Rust's
// `f32::round` rounds halves away from zero, whereas `WGSL`'s builtin `round`
// rounds halves to even, so this shader reduces with an explicit
// away-from-zero `round_rust` to stay faithful to the golden phase.
//
// ABI (matches `WaterKernel::FftStage.descriptor()`: 2 storage buffers,
// 1 uniform, 0 textures, 8x8 tile, Grid2d domain):
//   @binding(0) src    : array<vec2<f32>>, the n*n complex grid (re, im)
//   @binding(1) dst     : array<vec2<f32>>, the stage output grid (written)
//   @binding(2) params  : FftStageParams uniform (n, axis, len, log2n)
//
// `axis` selects the transform direction: 0 combines along rows (the x index),
// 1 along columns (the y index). `len` is the current span; `log2n` is carried
// for uniform-layout parity with the reorder/normalize passes and unused here.

struct FftStageParams {
    n: u32,
    axis: u32,
    len: u32,
    log2n: u32,
}

@group(0) @binding(0) var<storage, read> src: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: FftStageParams;

const PI: f32 = 3.1415927;
const FRAC_PI_2: f32 = 1.5707964;
const TWO_PI: f32 = 6.2831855;

// Round half away from zero, matching Rust `f32::round` (not `WGSL` round,
// which ties to even).
fn round_rust(x: f32) -> f32 {
    return sign(x) * floor(abs(x) + 0.5);
}

// Reduce an angle to `[-PI, PI]` by subtracting the nearest whole multiple of
// `2*PI`; mirrors the golden `wrap_pi`.
fn wrap_pi(x: f32) -> f32 {
    let k = round_rust(x / TWO_PI);
    return x - k * TWO_PI;
}

// Hand-rolled sine via range reduction + seventh-order Taylor polynomial,
// mirroring the golden `sin_approx`.
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

// Hand-rolled cosine via `cos(x) = sin(x + PI/2)`, mirroring `cos_approx`.
fn cos_approx(x: f32) -> f32 {
    return sin_approx(x + FRAC_PI_2);
}

// Linear buffer index of line position `pos` for this invocation's fixed line.
fn elem_index(pos: u32, gid: vec3<u32>) -> u32 {
    if (params.axis == 0u) {
        return gid.y * params.n + pos;
    }
    return pos * params.n + gid.x;
}

@compute @workgroup_size(8, 8, 1)
fn water_fft_stage(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.n || gid.y >= params.n) {
        return;
    }
    let len = params.len;
    if (len < 2u) {
        // Degenerate span: pass the element through unchanged.
        let idx = elem_index(select(gid.y, gid.x, params.axis == 0u), gid);
        dst[idx] = src[idx];
        return;
    }
    let half = len / 2u;
    let pos = select(gid.y, gid.x, params.axis == 0u);
    let group = pos / len;
    let local = pos - group * len;
    let is_top = local < half;

    let k = select(local - half, local, is_top);
    let top_pos = select(pos - half, pos, is_top);
    let bot_pos = select(pos, pos + half, is_top);

    let theta = TWO_PI / f32(len) * f32(k);
    let tw = vec2<f32>(cos_approx(theta), sin_approx(theta));

    let top = src[elem_index(top_pos, gid)];
    let b = src[elem_index(bot_pos, gid)];
    let prod = vec2<f32>(b.x * tw.x - b.y * tw.y, b.x * tw.y + b.y * tw.x);

    let out_idx = elem_index(pos, gid);
    dst[out_idx] = select(
        vec2<f32>(top.x - prod.x, top.y - prod.y),
        vec2<f32>(top.x + prod.x, top.y + prod.y),
        is_top,
    );
}
