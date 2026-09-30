//! `GPU` ping-pong butterfly `FFT` — real-device twin of the `CPU` golden
//! `prism_render_architecture::water::fft` (radix-2 `Cooley-Tukey`, `DIT`).
//!
//! Every shipping ocean (`WaveWorks`, `Crest`, `UE5` Water) inverts the
//! `Tessendorf` spectrum with a separable butterfly `FFT` rather than the
//! `O(N^4)` direct sum the reference `water_ocean.wesl` spells out. This module
//! is that butterfly, decomposed into three tiny compute passes the host
//! ping-pongs so a real device reproduces the `CPU` golden `ifft2` to
//! `float32` rounding:
//!
//!   1. `water_fft_bitrev`   — decimation-in-time reorder along one axis
//!                             (`dst[pos] = src[reverse_bits(pos)]`).
//!   2. `water_fft_stage`    — one radix-2 butterfly stage of length `len`
//!                             (host invokes it for `len = 2, 4, ... , N`).
//!   3. `water_fft_normalize`— the single `1/(N*N)` inverse scale at the end.
//!
//! A full inverse of an `N*N` grid is: bitrev + `log2 N` stages along the rows
//! (axis 0), the same along the columns (axis 1), then one normalize — exactly
//! the row-then-column, normalise-once separable order the `CPU` `transform2`
//! uses. `axis` selects the traversal (0 = along `x` within a row, 1 = along
//! `y` within a column); `sign` is fixed positive here because this module only
//! serves the inverse transform (the production path never needs the forward
//! `FFT` on device). Each pass reads `fft_src` and writes a distinct `fft_dst`,
//! so it is race-free and deterministic regardless of thread scheduling; the
//! host swaps the two bindings between passes.
//!
//! The twiddle phasors use the hardware `cos`/`sin`; the parity test therefore
//! compares against a `libm`-trig `DIT` reference computed host-side rather than
//! the crate's polynomial `sin_approx`/`cos_approx`, isolating the butterfly
//! algorithm from the trig-approximation gap. This module is deliberately kept
//! out of `WaterKernel::ALL`: it is an algorithm-parity proof, not yet wired
//! into the cross-crate production dispatch/pipeline, which still runs the
//! direct-sum `water_spectrum_ifft`.

/// `2*PI`, matching the `CPU` golden's `TWO_PI` constant to full `f32` width.
const FFT_TWO_PI: f32 = 6.28318530717958647692;

/// Drives one butterfly pass. `n` is the grid edge, `axis` selects row (0) or
/// column (1) traversal, `len` is the current stage length (unused by the
/// reorder / normalize passes), and `log2n` is `log2(n)` for the bit reversal.
struct FftParams {
    n: u32,
    axis: u32,
    len: u32,
    log2n: u32,
}

@group(0) @binding(0) var<storage, read> fft_src: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> fft_dst: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> fft_params: FftParams;

/// Complex product `(a.x + i a.y)(b.x + i b.y)`, matching the `CPU`
/// `Complex::mul`.
fn c_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

/// Reverses the low `bits` bits of `x` — the decimation-in-time permutation
/// index, mirroring the `CPU` `reverse_bits`.
fn reverse_bits_low(x: u32, bits: u32) -> u32 {
    var reversed = 0u;
    var v = x;
    for (var b = 0u; b < bits; b = b + 1u) {
        reversed = (reversed << 1u) | (v & 1u);
        v = v >> 1u;
    }
    return reversed;
}

/// Row-major index of element `pos` on line `line`. `axis == 0` walks a row
/// (`line * n + pos`); `axis == 1` walks a column (`pos * n + line`).
fn lin_index(line: u32, pos: u32, n: u32, axis: u32) -> u32 {
    if axis == 0u {
        return line * n + pos;
    }
    return pos * n + line;
}

/// Decimation-in-time reorder: `dst[line, pos] = src[line, reverse_bits(pos)]`.
@compute @workgroup_size(8, 8, 1)
fn water_fft_bitrev(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = fft_params.n;
    let pos = gid.x;
    let line = gid.y;
    if pos >= n || line >= n {
        return;
    }
    let src_pos = reverse_bits_low(pos, fft_params.log2n);
    let dst_i = lin_index(line, pos, n, fft_params.axis);
    let src_i = lin_index(line, src_pos, n, fft_params.axis);
    fft_dst[dst_i] = fft_src[src_i];
}

/// One radix-2 inverse butterfly stage of length `len`. Each invocation owns
/// one butterfly `pair`: it combines `i0 = start + k` with `i1 = i0 + half`
/// using the twiddle `e^{+i 2*PI k / len}` (positive sign = inverse). The
/// `n/2` pairs per line cover all `n` elements exactly once, so `fft_dst` is
/// fully written and the host ping-pong is safe.
@compute @workgroup_size(8, 8, 1)
fn water_fft_stage(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = fft_params.n;
    let len = fft_params.len;
    let half = len / 2u;
    let pair = gid.x;
    let line = gid.y;
    if pair >= n / 2u || line >= n {
        return;
    }
    let block = pair / half;
    let k = pair % half;
    let start = block * len;
    let i0 = start + k;
    let i1 = start + k + half;
    let theta = FFT_TWO_PI * f32(k) / f32(len);
    let tw = vec2<f32>(cos(theta), sin(theta));
    let axis = fft_params.axis;
    let a = fft_src[lin_index(line, i0, n, axis)];
    let b = c_mul(fft_src[lin_index(line, i1, n, axis)], tw);
    fft_dst[lin_index(line, i0, n, axis)] = a + b;
    fft_dst[lin_index(line, i1, n, axis)] = a - b;
}

/// Final inverse scale: `dst = src * (1 / (N*N))`, applied once over the whole
/// grid after both axes are transformed (mirrors the `CPU` `transform2` tail).
@compute @workgroup_size(8, 8, 1)
fn water_fft_normalize(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = fft_params.n;
    let x = gid.x;
    let y = gid.y;
    if x >= n || y >= n {
        return;
    }
    let i = y * n + x;
    let inv = 1.0 / f32(n * n);
    fft_dst[i] = fft_src[i] * inv;
}
