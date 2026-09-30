//! Host-side bridge from the architecture-layer butterfly `FFT` plan to the
//! packed `#[repr(C)]` uniform the `water_butterfly.wesl` passes bind.
//!
//! The arch crate is dependency-free, so its
//! [`FftPassParams`](prism_render_architecture::water::gpu::FftPassParams) is a
//! plain integer record it cannot mark `bytemuck::Pod`. This module is the one
//! place that crosses that boundary: it copies each scheduled pass payload into
//! [`GpuWaterFftParams`], the `Pod`/`Zeroable` mirror the renderer uploads with
//! `bytemuck`, so the same deterministic plan drives both the `CPU` golden
//! transform and the on-device dispatch.
//!
//! Keeping the conversion here (rather than inside the dispatch recorder) keeps
//! the recorder free of arch-type imports and lets the mapping be unit-tested
//! against the golden [`plan_inverse_fft2`] output without a `GPU` in scope.

use super::abi::GpuWaterFftParams;
use prism_render_architecture::water::gpu::{plan_inverse_fft2, FftPass, FftPassParams};

/// Packs one scheduled pass payload into the uploadable uniform mirror.
///
/// The field order is identical on both sides (`n`, `axis`, `len`, `log2n`), so
/// this is a lossless copy; the named-field spelling keeps it robust against a
/// future reordering of either record.
#[must_use]
pub(crate) fn fft_pass_uniform(params: FftPassParams) -> GpuWaterFftParams {
    GpuWaterFftParams {
        n: params.n,
        axis: params.axis,
        len: params.len,
        log2n: params.log2n,
    }
}

/// Packs an ordered butterfly plan into the per-pass uniform payloads, in the
/// exact dispatch order the recorder replays them.
///
/// One [`GpuWaterFftParams`] is produced per [`FftPass`]; the renderer uploads
/// them one at a time (re-writing the resident `fft_params` uniform before each
/// dispatch) so a single small uniform slot drives every pass.
#[must_use]
pub(crate) fn fft_pass_uniforms(passes: &[FftPass]) -> Vec<GpuWaterFftParams> {
    passes.iter().map(|p| fft_pass_uniform(p.params)).collect()
}

/// Plans the inverse `FFT` for a grid of edge `n` and packs every scheduled pass
/// into upload-ready uniforms in dispatch order.
///
/// The renderer calls this once per body to seed the resident `fft_params`
/// slot; a non-power-of-two `n` yields an empty plan (and thus no uniforms).
#[must_use]
pub(crate) fn fft_pass_uniforms_for(n: u32) -> Vec<GpuWaterFftParams> {
    fft_pass_uniforms(&plan_inverse_fft2(n))
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_architecture::water::gpu::{
        inverse_fft2_pass_count, plan_inverse_fft2, FftAxis, FftEntry,
    };

    #[test]
    fn single_pass_copies_every_field() {
        let src = FftPassParams {
            n: 256,
            axis: FftAxis::Column.index(),
            len: 8,
            log2n: 8,
        };
        let dst = fft_pass_uniform(src);
        assert_eq!(dst.n, 256);
        assert_eq!(dst.axis, 1);
        assert_eq!(dst.len, 8);
        assert_eq!(dst.log2n, 8);
    }

    #[test]
    fn plan_maps_to_one_uniform_per_pass() {
        for &n in &[16u32, 64, 256] {
            let passes = plan_inverse_fft2(n);
            let uniforms = fft_pass_uniforms(&passes);
            assert_eq!(uniforms.len(), passes.len());
            assert_eq!(uniforms.len(), inverse_fft2_pass_count(n));
            // Every payload carries the same grid edge `N`.
            for u in &uniforms {
                assert_eq!(u.n, n);
            }
        }
    }

    #[test]
    fn bit_reversal_and_normalize_zero_the_span() {
        // The reorder and the final normalize do not read `len`; the plan pins
        // it to a stable `0` and the mapping must preserve that.
        let passes = plan_inverse_fft2(16);
        for pass in &passes {
            let u = fft_pass_uniform(pass.params);
            match pass.entry {
                FftEntry::BitReversal | FftEntry::Normalize => assert_eq!(u.len, 0),
                FftEntry::Butterfly => assert!(u.len >= 2 && u.len.is_power_of_two()),
            }
        }
    }

    #[test]
    fn empty_plan_yields_no_uniforms() {
        // Non-power-of-two and degenerate edges produce an empty plan.
        assert!(fft_pass_uniforms(&plan_inverse_fft2(0)).is_empty());
        assert!(fft_pass_uniforms(&plan_inverse_fft2(1)).is_empty());
        assert!(fft_pass_uniforms(&plan_inverse_fft2(24)).is_empty());
    }

    #[test]
    fn axis_selectors_are_row_then_column() {
        // The separable plan transforms every row first (`axis == 0`) then every
        // column (`axis == 1`); the mapped selectors must reflect that order.
        let uniforms = fft_pass_uniforms(&plan_inverse_fft2(16));
        let first_axis = uniforms.first().map(|u| u.axis);
        assert_eq!(first_axis, Some(0));
        // The final pass is the shared normalize, pinned to axis `0`.
        assert_eq!(uniforms.last().map(|u| u.axis), Some(0));
        // A column pass (`axis == 1`) appears in the middle of the plan.
        assert!(uniforms.iter().any(|u| u.axis == 1));
    }
}
