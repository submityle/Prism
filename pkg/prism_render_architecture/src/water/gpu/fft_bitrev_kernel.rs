//! Water ocean-spectrum `FFT` bit-reversal reorder compute kernel: the `WESL`
//! shader plus its bit-exact `CPU` twin.
//!
//! A decimation-in-time radix-2 `Cooley-Tukey` butterfly must permute its input
//! into bit-reversed order before the in-place butterfly stages combine the
//! right sub-transforms. The separable 2D inverse transform runs this reorder
//! once per axis as an out-of-place ping-pong pass: every invocation writes its
//! own `(x, y)` complex texel by reading the source element whose index along
//! the active axis is bit-reversed. Bit reversal is an involution, so reading
//! the reversed source is the same permutation as writing the reversed
//! destination, and each output slot is written exactly once.
//!
//! [`WATER_FFT_BITREV_WESL`] is the shader (entry point `water_fft_bitrev`);
//! [`dispatch_fft_bit_reverse`] is its bit-exact `CPU` twin. Because the sandbox
//! has no `GPU`, the twin is the correctness proof: it consumes the identical
//! buffer `ABI` ([`WaterKernel::FftBitReverse`](super::super::kernels::WaterKernel)
//! — two storage buffers, one uniform param block, no textures, 8x8 tile,
//! `Grid2d` domain) and reconstructs the shader's integer index permutation.
//! The permutation mirrors the decimation-in-time reorder step of the `CPU`
//! golden [`fft::transform`](super::super::fft) (its private `reverse_bits`),
//! so the pass is bit-exact — it is pure integer index math with no float
//! arithmetic and nothing for the determinism policy to constrain.

use alloc::vec;
use alloc::vec::Vec;

/// `WESL` source of the water `FFT` bit-reversal reorder compute kernel.
///
/// Bound into the crate so the shader ships and is covered by the structural
/// test; the standalone `naga`/`wesl` compile check runs out of tree, because
/// `prism_render_architecture` is a zero-dependency crate.
pub const WATER_FFT_BITREV_WESL: &str = include_str!("water_fft_bitrev.wesl");

/// Number of `f32` lanes in one complex grid element: `(re, im)`.
pub const FFT_COMPLEX_FLOATS: usize = 2;

/// Uniform parameter block for the bit-reversal reorder pass.
///
/// `n` is the per-axis grid extent (`n == 1 << log2n`), `log2n` the bit width to
/// reverse, and `axis` the transform direction: `0` reorders along rows (the x
/// index), `1` reorders along columns (the y index).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FftBitReverseParams {
    /// Per-axis grid extent; must be a power of two.
    pub n: u32,
    /// Bit width to reverse (`n == 1 << log2n`).
    pub log2n: u32,
    /// Transform axis: `0` reorders rows (x), `1` reorders columns (y).
    pub axis: u32,
}

/// Reverses the low `bits` bits of `x`, the bit-reversal permutation index.
///
/// Mirrors the shader's `reverse_bits_low` shift-accumulate loop. Kept
/// independent of the golden [`fft`](super::super::fft) `reverse_bits` so the
/// parity test proves the transcription rather than asserting a tautology.
#[must_use]
fn reverse_bits_twin(x: u32, bits: u32) -> u32 {
    let mut reversed = 0u32;
    let mut v = x;
    let mut b = 0u32;
    while b < bits {
        reversed = (reversed << 1) | (v & 1);
        v >>= 1;
        b += 1;
    }
    reversed
}

/// Bit-exact `CPU` twin of the `water_fft_bitrev` kernel.
///
/// `src` is the row-major `n*n` complex grid, two `f32` lanes `(re, im)` per
/// element. Returns the reordered grid in which every `(x, y)` texel is read
/// from the source whose index along `params.axis` is bit-reversed. A degenerate
/// request — a zero or non-power-of-two extent, or a buffer that is not exactly
/// `n*n` complex elements — is left untouched (the grid is returned verbatim),
/// matching the golden transform's "non-power-of-two is a no-op" contract.
#[must_use]
pub fn dispatch_fft_bit_reverse(src: &[f32], params: FftBitReverseParams) -> Vec<f32> {
    let n = params.n as usize;
    if n == 0 || !super::super::fft::is_power_of_two(n) || src.len() != n * n * FFT_COMPLEX_FLOATS {
        return src.to_vec();
    }

    let mut out = vec![0.0_f32; src.len()];
    let mut y = 0u32;
    while y < params.n {
        let mut x = 0u32;
        while x < params.n {
            let (sx, sy) = if params.axis == 0 {
                (reverse_bits_twin(x, params.log2n), y)
            } else {
                (x, reverse_bits_twin(y, params.log2n))
            };
            let dst_idx = (y as usize * n + x as usize) * FFT_COMPLEX_FLOATS;
            let src_idx = (sy as usize * n + sx as usize) * FFT_COMPLEX_FLOATS;
            out[dst_idx] = src[src_idx];
            out[dst_idx + 1] = src[src_idx + 1];
            x += 1;
        }
        y += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::super::super::fft::is_power_of_two;
    use super::*;

    /// Independent bit-reversal reference built the opposite way to
    /// [`reverse_bits_twin`]'s shift-accumulate loop: it reads bit `i` of `x`
    /// and places it at mirror position `bits - 1 - i`. Two structurally
    /// different implementations agreeing proves the transcription, not a
    /// tautology.
    fn ref_reverse(x: u32, bits: u32) -> u32 {
        let mut out = 0u32;
        for i in 0..bits {
            if (x >> i) & 1 == 1 {
                out |= 1 << (bits - 1 - i);
            }
        }
        out
    }

    #[test]
    fn reverse_bits_matches_independent_reference() {
        // Three bit widths across their full value range: the shift-accumulate
        // twin must agree with the mirror-placement reference everywhere.
        for bits in [1u32, 2, 3, 4, 5] {
            let span = 1u32 << bits;
            let mut x = 0u32;
            while x < span {
                assert_eq!(
                    reverse_bits_twin(x, bits),
                    ref_reverse(x, bits),
                    "x={x} bits={bits}"
                );
                x += 1;
            }
        }
    }

    #[test]
    fn row_reorder_pulls_from_bit_reversed_source() {
        // A 4x4 grid (log2n = 2) reordered along rows: output (x, y) must carry
        // the complex pair from source (ref_reverse(x), y), both lanes exact.
        let n = 4u32;
        let log2n = 2u32;
        let mut src = vec![0.0_f32; (n * n) as usize * FFT_COMPLEX_FLOATS];
        // Encode each element so re/im are distinguishable and position-coded.
        for idx in 0..(n * n) as usize {
            src[idx * FFT_COMPLEX_FLOATS] = idx as f32; // re
            src[idx * FFT_COMPLEX_FLOATS + 1] = -(idx as f32); // im
        }
        let params = FftBitReverseParams { n, log2n, axis: 0 };
        let out = dispatch_fft_bit_reverse(&src, params);

        for y in 0..n {
            for x in 0..n {
                let sx = ref_reverse(x, log2n);
                let dst = (y * n + x) as usize * FFT_COMPLEX_FLOATS;
                let want = (y * n + sx) as usize * FFT_COMPLEX_FLOATS;
                assert_eq!(out[dst].to_bits(), src[want].to_bits(), "re x={x} y={y}");
                assert_eq!(
                    out[dst + 1].to_bits(),
                    src[want + 1].to_bits(),
                    "im x={x} y={y}"
                );
            }
        }
    }

    #[test]
    fn column_reorder_pulls_from_bit_reversed_source() {
        // The same grid reordered along columns (axis = 1): output (x, y) must
        // carry source (x, ref_reverse(y)).
        let n = 4u32;
        let log2n = 2u32;
        let mut src = vec![0.0_f32; (n * n) as usize * FFT_COMPLEX_FLOATS];
        for idx in 0..(n * n) as usize {
            src[idx * FFT_COMPLEX_FLOATS] = idx as f32;
            src[idx * FFT_COMPLEX_FLOATS + 1] = -(idx as f32);
        }
        let params = FftBitReverseParams { n, log2n, axis: 1 };
        let out = dispatch_fft_bit_reverse(&src, params);

        for y in 0..n {
            for x in 0..n {
                let sy = ref_reverse(y, log2n);
                let dst = (y * n + x) as usize * FFT_COMPLEX_FLOATS;
                let want = (sy * n + x) as usize * FFT_COMPLEX_FLOATS;
                assert_eq!(out[dst].to_bits(), src[want].to_bits(), "re x={x} y={y}");
                assert_eq!(
                    out[dst + 1].to_bits(),
                    src[want + 1].to_bits(),
                    "im x={x} y={y}"
                );
            }
        }
    }

    #[test]
    fn reorder_is_an_involution() {
        // Bit reversal is self-inverse: reordering twice along the same axis
        // restores the original grid bit-for-bit.
        let n = 8u32;
        let log2n = 3u32;
        let mut src = vec![0.0_f32; (n * n) as usize * FFT_COMPLEX_FLOATS];
        for idx in 0..(n * n) as usize {
            src[idx * FFT_COMPLEX_FLOATS] = (idx * 2 + 1) as f32;
            src[idx * FFT_COMPLEX_FLOATS + 1] = (idx * 3 + 5) as f32;
        }
        let params = FftBitReverseParams { n, log2n, axis: 0 };
        let once = dispatch_fft_bit_reverse(&src, params);
        let twice = dispatch_fft_bit_reverse(&once, params);
        let mut i = 0usize;
        while i < src.len() {
            assert_eq!(twice[i].to_bits(), src[i].to_bits(), "lane {i}");
            i += 1;
        }
    }

    #[test]
    fn degenerate_requests_are_left_untouched() {
        // Non-power-of-two extent and a mis-sized buffer are both no-ops: the
        // grid is returned verbatim rather than panicking or garbling.
        assert!(!is_power_of_two(6));
        let three = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let not_pow2 = FftBitReverseParams {
            n: 6,
            log2n: 3,
            axis: 0,
        };
        let out = dispatch_fft_bit_reverse(&three, not_pow2);
        assert_eq!(out, three, "non-power-of-two extent is a no-op");

        // Correct power-of-two extent but a buffer that is not n*n complex.
        let mis_sized = vec![1.0_f32, 2.0, 3.0];
        let pow2 = FftBitReverseParams {
            n: 4,
            log2n: 2,
            axis: 0,
        };
        let out = dispatch_fft_bit_reverse(&mis_sized, pow2);
        assert_eq!(out, mis_sized, "mis-sized buffer is a no-op");
    }

    #[test]
    fn wesl_kernel_declares_expected_abi() {
        assert!(WATER_FFT_BITREV_WESL.contains("fn water_fft_bitrev"));
        assert!(WATER_FFT_BITREV_WESL.contains("@workgroup_size(8, 8, 1)"));
        assert!(WATER_FFT_BITREV_WESL.contains("struct FftBitReverseParams"));
        assert!(WATER_FFT_BITREV_WESL.contains("var<storage, read_write> dst"));
    }

    #[test]
    fn stride_is_consistent() {
        assert_eq!(FFT_COMPLEX_FLOATS, 2);
    }
}
