//! Water ocean-spectrum inverse-`FFT` normalization compute kernel: the `WESL`
//! shader plus its bit-exact `CPU` twin.
//!
//! The separable 2D inverse transform runs its unnormalized butterfly along
//! every row and column; forward and inverse butterflies differ only in twiddle
//! sign, so the round trip is scaled up by the element count. The inverse must
//! divide the whole grid by `N*N` once, after the final butterfly axis, to
//! recover the spatial ocean field. This pass is that single scalar scaling:
//! each invocation multiplies its own `(x, y)` complex texel by `1 / (n*n)`.
//!
//! [`WATER_FFT_NORMALIZE_WESL`] is the shader (entry point
//! `water_fft_normalize`); [`dispatch_fft_normalize`] is its bit-exact `CPU`
//! twin. Because the sandbox has no `GPU`, the twin is the correctness proof: it
//! consumes the identical buffer `ABI`
//! ([`WaterKernel::FftNormalize`](super::super::kernels::WaterKernel) — two
//! storage buffers, one uniform param block, no textures, 8x8 tile, `Grid2d`
//! domain) and reconstructs the shader's scalar scaling. It mirrors the
//! normalization tail of the `CPU` golden
//! [`fft::transform2`](super::super::fft) (its `inverse` branch's `1 / (N*N)`
//! fold): `n*n` is a power of two, so both the product and its reciprocal are
//! exact in `f32` and the per-lane multiply introduces no rounding, keeping the
//! pass bit-exact with nothing for the determinism policy to constrain.

use alloc::vec;
use alloc::vec::Vec;

/// `WESL` source of the water inverse-`FFT` normalization compute kernel.
///
/// Bound into the crate so the shader ships and is covered by the structural
/// test; the standalone `naga`/`wesl` compile check runs out of tree, because
/// `prism_render_architecture` is a zero-dependency crate.
pub const WATER_FFT_NORMALIZE_WESL: &str = include_str!("water_fft_normalize.wesl");

/// Number of `f32` lanes in one complex grid element: `(re, im)`.
pub const FFT_COMPLEX_FLOATS: usize = 2;

/// Uniform parameter block for the inverse-`FFT` normalization pass.
///
/// `n` is the per-axis grid extent (`n == 1 << log2n`); the scaling factor is
/// `1 / (n*n)`, matching the 2D inverse transform's `1/(N*N)` fold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FftNormalizeParams {
    /// Per-axis grid extent; must be a power of two.
    pub n: u32,
}

/// Bit-exact `CPU` twin of the `water_fft_normalize` kernel.
///
/// `src` is the row-major `n*n` complex grid, two `f32` lanes `(re, im)` per
/// element. Returns the grid with every complex lane multiplied by
/// `1 / (n*n)`. A degenerate request — a zero or non-power-of-two extent, or a
/// buffer that is not exactly `n*n` complex elements — is left untouched (the
/// grid is returned verbatim), matching the golden transform's "mis-sized or
/// non-power-of-two is a no-op" contract.
#[must_use]
pub fn dispatch_fft_normalize(src: &[f32], params: FftNormalizeParams) -> Vec<f32> {
    let n = params.n as usize;
    if n == 0 || !super::super::fft::is_power_of_two(n) || src.len() != n * n * FFT_COMPLEX_FLOATS {
        return src.to_vec();
    }

    // Mirror the shader's `f32(params.n * params.n)` product then reciprocal.
    let total = (params.n * params.n) as f32;
    let inv = 1.0 / total;

    let mut out = vec![0.0_f32; src.len()];
    let mut i = 0usize;
    while i < src.len() {
        out[i] = src[i] * inv;
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::super::super::fft::{ifft2, is_power_of_two};
    use super::super::super::spectrum::Complex;
    use super::*;

    #[test]
    fn scales_every_lane_by_inverse_element_count() {
        // An arbitrary 4x4 complex grid: every lane must equal the source lane
        // times `1 / (n*n)`, bit-for-bit, independently recomputed.
        let n = 4u32;
        let count = (n * n) as usize;
        let mut src = vec![0.0_f32; count * FFT_COMPLEX_FLOATS];
        let mut k = 0usize;
        while k < src.len() {
            src[k] = (k as f32) * 0.5 - 3.0;
            k += 1;
        }
        let out = dispatch_fft_normalize(&src, FftNormalizeParams { n });

        let inv = 1.0 / ((n * n) as f32);
        let mut i = 0usize;
        while i < src.len() {
            assert_eq!(out[i].to_bits(), (src[i] * inv).to_bits(), "lane {i}");
            i += 1;
        }
    }

    #[test]
    fn factor_matches_golden_ifft2_of_an_impulse() {
        // The inverse 2D transform of a unit impulse at the origin is a uniform
        // grid whose every element is exactly `1 / (n*n)` (the DC term spreads
        // to all texels through both butterfly axes, then the `1/(N*N)` fold
        // scales it). Normalizing an all-ones grid must therefore reproduce the
        // golden `ifft2` output bit-for-bit — a genuine golden tie, not a
        // tautology, since the reference comes from the full transform.
        let n = 4u32;
        let count = (n * n) as usize;

        let mut impulse = vec![Complex::new(0.0, 0.0); count];
        impulse[0] = Complex::new(1.0, 0.0);
        let golden = ifft2(&impulse, n as usize);

        let mut ones = vec![0.0_f32; count * FFT_COMPLEX_FLOATS];
        let mut j = 0usize;
        while j < count {
            ones[j * FFT_COMPLEX_FLOATS] = 1.0;
            ones[j * FFT_COMPLEX_FLOATS + 1] = 0.0;
            j += 1;
        }
        let out = dispatch_fft_normalize(&ones, FftNormalizeParams { n });

        let mut i = 0usize;
        while i < count {
            assert_eq!(
                out[i * FFT_COMPLEX_FLOATS].to_bits(),
                golden[i].re.to_bits(),
                "re element {i}"
            );
            assert_eq!(
                out[i * FFT_COMPLEX_FLOATS + 1].to_bits(),
                golden[i].im.to_bits(),
                "im element {i}"
            );
            i += 1;
        }
    }

    #[test]
    fn degenerate_requests_are_left_untouched() {
        // Non-power-of-two extent and a mis-sized buffer are both no-ops.
        assert!(!is_power_of_two(6));
        let not_pow2 = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = dispatch_fft_normalize(&not_pow2, FftNormalizeParams { n: 6 });
        assert_eq!(out, not_pow2, "non-power-of-two extent is a no-op");

        let mis_sized = vec![1.0_f32, 2.0, 3.0];
        let out = dispatch_fft_normalize(&mis_sized, FftNormalizeParams { n: 4 });
        assert_eq!(out, mis_sized, "mis-sized buffer is a no-op");
    }

    #[test]
    fn wesl_kernel_declares_expected_abi() {
        assert!(WATER_FFT_NORMALIZE_WESL.contains("fn water_fft_normalize"));
        assert!(WATER_FFT_NORMALIZE_WESL.contains("@workgroup_size(8, 8, 1)"));
        assert!(WATER_FFT_NORMALIZE_WESL.contains("struct FftNormalizeParams"));
        assert!(WATER_FFT_NORMALIZE_WESL.contains("var<storage, read_write> dst"));
    }

    #[test]
    fn stride_is_consistent() {
        assert_eq!(FFT_COMPLEX_FLOATS, 2);
    }
}
