//! Water ocean-spectrum inverse-`FFT` radix-2 butterfly stage compute kernel:
//! the `WESL` shader plus its bit-exact `CPU` twin.
//!
//! After the bit-reversal reorder, the separable inverse `Cooley-Tukey`
//! transform runs `log2(n)` butterfly stages per axis, the span doubling
//! 2, 4, …, N. On the `GPU` each stage is one out-of-place ping-pong pass:
//! every invocation writes its own line position by combining the two source
//! elements of its butterfly pair with the stage twiddle. For a stage of span
//! `len` (half = len/2) the line splits into groups of `len`; position
//! `s + k` (k in 0..half) is the pair's "top" and `s + k + half` its "bottom",
//! and the inverse butterfly (twiddle sign `+1`) is
//!   `out[s + k]        = top + W_k * bottom`
//!   `out[s + k + half] = top - W_k * bottom`,  `W_k = e^{+i 2*PI k / len}`.
//!
//! [`WATER_FFT_STAGE_WESL`] is the shader (entry point `water_fft_stage`);
//! [`dispatch_fft_stage`] is its bit-exact `CPU` twin. Because the sandbox has
//! no `GPU`, the twin is the correctness proof: it consumes the identical
//! buffer `ABI` ([`WaterKernel::FftStage`](super::super::kernels::WaterKernel)
//! — two storage buffers, one uniform param block, no textures, 8x8 tile,
//! `Grid2d` domain) and reconstructs the shader's butterfly. It mirrors the
//! butterfly loop of the `CPU` golden [`fft::transform`](super::super::fft)
//! (its `inverse` branch, sign = `+1`) element-for-element: the twiddle phasor
//! goes through the crate's hand-rolled
//! [`sin_approx`](super::super::sin_approx) /
//! [`cos_approx`](super::super::cos_approx), and the complex multiply and add
//! reuse [`Complex::mul`](super::super::spectrum::Complex::mul) /
//! [`Complex::add`](super::super::spectrum::Complex::add), so the twin is
//! bit-exact with the golden by construction. Chained bit-reversal → every
//! stage → normalization reproduces [`fft::ifft2`](super::super::fft::ifft2)
//! lane-for-lane; the composition test proves that tie rather than asserting a
//! tautology.
//!
//! The shader carries one `GPU`-fidelity subtlety the twin does not need: the
//! range-reduction round inside `wrap_pi`. Rust `f32::round` rounds halves away
//! from zero whereas the `WGSL` builtin `round` ties to even, so the shader
//! reduces with an explicit away-from-zero `round_rust`. On the `CPU` twin the
//! crate's own `sin_approx`/`cos_approx` are reused directly, so the twin is
//! trivially bit-exact with the golden and nothing is left for the determinism
//! policy to constrain.

use alloc::vec;
use alloc::vec::Vec;

use super::super::spectrum::Complex;
use super::super::{cos_approx, sin_approx, TWO_PI};
use super::fft_bitrev_kernel::FFT_COMPLEX_FLOATS;

/// `WESL` source of the water inverse-`FFT` butterfly stage compute kernel.
///
/// Bound into the crate so the shader ships and is covered by the structural
/// test; the standalone `naga`/`wesl` compile check runs out of tree, because
/// `prism_render_architecture` is a zero-dependency crate.
pub const WATER_FFT_STAGE_WESL: &str = include_str!("water_fft_stage.wesl");

/// Uniform parameter block for one inverse-`FFT` butterfly stage.
///
/// The field layout matches the planner's
/// [`FftPassParams`](super::fft_plan::FftPassParams): `n` is the per-axis grid
/// extent (`n == 1 << log2n`), `axis` the transform direction (`0` combines
/// rows, `1` columns), `len` the current butterfly span (2, 4, …, n), and
/// `log2n` is carried for uniform-layout parity with the reorder/normalize
/// passes (unused by the butterfly itself).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FftStageParams {
    /// Per-axis grid extent; must be a power of two.
    pub n: u32,
    /// Transform axis: `0` combines along rows (x), `1` along columns (y).
    pub axis: u32,
    /// Butterfly span for this stage (2, 4, …, n).
    pub len: u32,
    /// Bit width (`n == 1 << log2n`); unused here, kept for layout parity.
    pub log2n: u32,
}

/// Linear complex-element index of line position `pos` for the fixed line that
/// invocation `(gx, gy)` owns. Mirrors the shader's `elem_index`.
#[inline]
fn elem_index(pos: usize, gx: usize, gy: usize, n: usize, axis: u32) -> usize {
    if axis == 0 {
        gy * n + pos
    } else {
        pos * n + gx
    }
}

/// Reads the complex element at linear index `idx` from the `(re, im)` lane
/// buffer `src`.
#[inline]
fn read_complex(src: &[f32], idx: usize) -> Complex {
    Complex::new(
        src[idx * FFT_COMPLEX_FLOATS],
        src[idx * FFT_COMPLEX_FLOATS + 1],
    )
}

/// Bit-exact `CPU` twin of the `water_fft_stage` kernel.
///
/// `src` is the row-major `n*n` complex grid, two `f32` lanes `(re, im)` per
/// element. Returns the grid after one inverse-`FFT` butterfly stage of span
/// `params.len` along `params.axis`. A degenerate request — a zero or
/// non-power-of-two extent, a buffer that is not exactly `n*n` complex
/// elements, or a span outside `2..=n` — is left untouched (the grid is
/// returned verbatim), matching the golden transform's "skip, do not crash"
/// contract for out-of-contract inputs.
#[must_use]
pub fn dispatch_fft_stage(src: &[f32], params: FftStageParams) -> Vec<f32> {
    let n = params.n as usize;
    if n == 0 || !super::super::fft::is_power_of_two(n) || src.len() != n * n * FFT_COMPLEX_FLOATS {
        return src.to_vec();
    }
    let len = params.len as usize;
    if len < 2 || len > n {
        // Degenerate span: identity pass-through, matching the shader.
        return src.to_vec();
    }
    let half = len / 2;
    // Mirror the shader's `step = TWO_PI / f32(len)` then `theta = step * k`.
    let step = TWO_PI / (params.len as f32);

    let mut out = vec![0.0_f32; src.len()];
    let mut gy = 0usize;
    while gy < n {
        let mut gx = 0usize;
        while gx < n {
            let pos = if params.axis == 0 { gx } else { gy };
            let local = pos - (pos / len) * len;
            let is_top = local < half;

            let k = if is_top { local } else { local - half };
            let top_pos = if is_top { pos } else { pos - half };
            let bot_pos = if is_top { pos + half } else { pos };

            let theta = step * (k as f32);
            let twiddle = Complex::new(cos_approx(theta), sin_approx(theta));

            let top = read_complex(src, elem_index(top_pos, gx, gy, n, params.axis));
            let bottom = read_complex(src, elem_index(bot_pos, gx, gy, n, params.axis));
            let prod = bottom.mul(twiddle);

            // golden: top.add(bottom*W) for the top slot, top.add(-(bottom*W))
            // for the bottom slot — bit-identical to +/- of the same product.
            let value = if is_top {
                top.add(prod)
            } else {
                top.add(Complex::new(-prod.re, -prod.im))
            };

            let out_idx = elem_index(pos, gx, gy, n, params.axis) * FFT_COMPLEX_FLOATS;
            out[out_idx] = value.re;
            out[out_idx + 1] = value.im;
            gx += 1;
        }
        gy += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::super::super::fft::{ifft2, is_power_of_two};
    use super::super::super::spectrum::Complex;
    use super::super::super::{cos_approx, sin_approx};
    use super::super::fft_bitrev_kernel::{dispatch_fft_bit_reverse, FftBitReverseParams};
    use super::super::fft_normalize_kernel::{dispatch_fft_normalize, FftNormalizeParams};
    use super::*;

    /// Flattens a complex grid into the `(re, im)` lane buffer the kernels
    /// consume.
    fn flatten(grid: &[Complex]) -> Vec<f32> {
        let mut out = vec![0.0_f32; grid.len() * FFT_COMPLEX_FLOATS];
        let mut i = 0usize;
        while i < grid.len() {
            out[i * FFT_COMPLEX_FLOATS] = grid[i].re;
            out[i * FFT_COMPLEX_FLOATS + 1] = grid[i].im;
            i += 1;
        }
        out
    }

    /// Builds a deterministic `n*n` complex grid with position-coded lanes.
    fn sample_grid(n: usize) -> Vec<Complex> {
        let mut grid = Vec::with_capacity(n * n);
        let mut idx = 0usize;
        while idx < n * n {
            let f = idx as f32;
            grid.push(Complex::new(0.25 * f - 1.0, 2.0 - 0.1 * f));
            idx += 1;
        }
        grid
    }

    #[test]
    fn pipeline_composition_matches_golden_ifft2() {
        // The full `GPU` inverse-FFT pipeline — bit-reversal, then every
        // butterfly stage, per axis, then the `1/(N*N)` normalization — must
        // reproduce the `CPU` golden `ifft2` lane-for-lane. This is the genuine
        // tie: the stage twin is chained with its sibling reorder/normalize
        // twins and compared against the independent separable transform.
        for &n in &[2usize, 4, 8] {
            let log2n = n.trailing_zeros();
            let grid = sample_grid(n);
            let golden = ifft2(&grid, n);

            let mut buf = flatten(&grid);
            // Axis 0 (rows), then axis 1 (columns): reorder, all stages.
            for axis in [0u32, 1u32] {
                buf = dispatch_fft_bit_reverse(
                    &buf,
                    FftBitReverseParams {
                        n: n as u32,
                        log2n,
                        axis,
                    },
                );
                let mut len = 2u32;
                while (len as usize) <= n {
                    buf = dispatch_fft_stage(
                        &buf,
                        FftStageParams {
                            n: n as u32,
                            axis,
                            len,
                            log2n,
                        },
                    );
                    len *= 2;
                }
            }
            // Final `1/(N*N)` fold.
            buf = dispatch_fft_normalize(&buf, FftNormalizeParams { n: n as u32 });

            let mut i = 0usize;
            while i < n * n {
                assert_eq!(
                    buf[i * FFT_COMPLEX_FLOATS].to_bits(),
                    golden[i].re.to_bits(),
                    "n={n} re element {i}"
                );
                assert_eq!(
                    buf[i * FFT_COMPLEX_FLOATS + 1].to_bits(),
                    golden[i].im.to_bits(),
                    "n={n} im element {i}"
                );
                i += 1;
            }
        }
    }

    #[test]
    fn first_stage_combines_adjacent_pairs() {
        // Stage len = 2 on a 4x4 grid along rows: each pair (2j, 2j+1) becomes
        // (sum, difference) because the twiddle W_0 = 1. Recomputed independently
        // from the source pairs.
        let n = 4usize;
        let grid = sample_grid(n);
        let src = flatten(&grid);
        let out = dispatch_fft_stage(
            &src,
            FftStageParams {
                n: n as u32,
                axis: 0,
                len: 2,
                log2n: 2,
            },
        );

        let mut y = 0usize;
        while y < n {
            let mut j = 0usize;
            while j < n / 2 {
                // W_0 = e^{+i*0}, built from the crate's polynomial trig
                // exactly as the kernel does (cos_approx(0) is ~0.99996, not a
                // literal 1), so this stays an independent recomputation rather
                // than a tautology.
                let w0 = Complex::new(cos_approx(0.0), sin_approx(0.0));
                let a = grid[y * n + 2 * j];
                let prod = grid[y * n + 2 * j + 1].mul(w0);
                let sum = a.add(prod);
                let diff = a.add(Complex::new(-prod.re, -prod.im));
                let top = (y * n + 2 * j) * FFT_COMPLEX_FLOATS;
                let bot = (y * n + 2 * j + 1) * FFT_COMPLEX_FLOATS;
                assert_eq!(out[top].to_bits(), sum.re.to_bits(), "sum re y={y} j={j}");
                assert_eq!(
                    out[top + 1].to_bits(),
                    sum.im.to_bits(),
                    "sum im y={y} j={j}"
                );
                assert_eq!(out[bot].to_bits(), diff.re.to_bits(), "diff re y={y} j={j}");
                assert_eq!(
                    out[bot + 1].to_bits(),
                    diff.im.to_bits(),
                    "diff im y={y} j={j}"
                );
                j += 1;
            }
            y += 1;
        }
    }

    #[test]
    fn column_axis_combines_along_y() {
        // Stage len = 2 along columns (axis = 1): the sum/difference pairs run
        // vertically, leaving each column independent.
        let n = 4usize;
        let grid = sample_grid(n);
        let src = flatten(&grid);
        let out = dispatch_fft_stage(
            &src,
            FftStageParams {
                n: n as u32,
                axis: 1,
                len: 2,
                log2n: 2,
            },
        );

        let mut x = 0usize;
        while x < n {
            let mut j = 0usize;
            while j < n / 2 {
                let w0 = Complex::new(cos_approx(0.0), sin_approx(0.0));
                let a = grid[(2 * j) * n + x];
                let prod = grid[(2 * j + 1) * n + x].mul(w0);
                let sum = a.add(prod);
                let diff = a.add(Complex::new(-prod.re, -prod.im));
                let top = ((2 * j) * n + x) * FFT_COMPLEX_FLOATS;
                let bot = ((2 * j + 1) * n + x) * FFT_COMPLEX_FLOATS;
                assert_eq!(out[top].to_bits(), sum.re.to_bits(), "sum re x={x} j={j}");
                assert_eq!(out[bot].to_bits(), diff.re.to_bits(), "diff re x={x} j={j}");
                j += 1;
            }
            x += 1;
        }
    }

    #[test]
    fn degenerate_requests_are_left_untouched() {
        // Non-power-of-two extent, mis-sized buffer, and an out-of-range span
        // are all no-ops: the grid is returned verbatim rather than panicking.
        assert!(!is_power_of_two(6));
        let not_pow2 = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = dispatch_fft_stage(
            &not_pow2,
            FftStageParams {
                n: 6,
                axis: 0,
                len: 2,
                log2n: 3,
            },
        );
        assert_eq!(out, not_pow2, "non-power-of-two extent is a no-op");

        let mis_sized = vec![1.0_f32, 2.0, 3.0];
        let out = dispatch_fft_stage(
            &mis_sized,
            FftStageParams {
                n: 4,
                axis: 0,
                len: 2,
                log2n: 2,
            },
        );
        assert_eq!(out, mis_sized, "mis-sized buffer is a no-op");

        // Correct 4x4 buffer but a span of 1 (< 2): identity pass-through.
        let grid = flatten(&sample_grid(4));
        let out = dispatch_fft_stage(
            &grid,
            FftStageParams {
                n: 4,
                axis: 0,
                len: 1,
                log2n: 2,
            },
        );
        assert_eq!(out, grid, "span < 2 is an identity pass-through");
    }

    #[test]
    fn wesl_kernel_declares_expected_abi() {
        assert!(WATER_FFT_STAGE_WESL.contains("fn water_fft_stage"));
        assert!(WATER_FFT_STAGE_WESL.contains("@workgroup_size(8, 8, 1)"));
        assert!(WATER_FFT_STAGE_WESL.contains("struct FftStageParams"));
        assert!(WATER_FFT_STAGE_WESL.contains("var<storage, read_write> dst"));
        assert!(WATER_FFT_STAGE_WESL.contains("round_rust"));
    }

    #[test]
    fn stride_is_consistent() {
        assert_eq!(FFT_COMPLEX_FLOATS, 2);
    }
}
