//! ABI shared between the exposure compute passes and `shaders/exposure.wesl`.
//!
//! Auto-exposure runs in two compute passes over the resolved HDR scene colour:
//! a **histogram build** that bins per-pixel log-luminance, and a
//! single-invocation **resolve** that reduces the histogram to a
//! percentile-trimmed average, eases the persistent eye-adaptation state toward
//! it, and writes the exposure multiplier the composite applies. This module
//! owns the two immediate (push-constant) blocks those entry points consume,
//! each mirroring its WGSL twin byte-for-byte so machines with and without a
//! GPU agree with the CPU golden in [`prism_render_shading::exposure`].

use bevy_math::UVec2;
use bytemuck::{Pod, Zeroable};
use prism_render_shading::{
    AutoExposureSettings, EyeAdaptation, HistogramPercentiles, HistogramRange,
};

/// Number of log-luminance bins, matching `EXPOSURE_BIN_COUNT` in
/// `exposure.wesl` and the storage buffers both passes bind. A 64-bin histogram
/// is the AAA norm: fine enough for stable metering, small enough to reduce in
/// a single invocation.
pub(crate) const EXPOSURE_BIN_COUNT: u32 = 64;

/// Compute workgroup edge the histogram build dispatches in, matching the
/// `@workgroup_size(8, 8)` in `exposure.wesl`. One invocation per pixel, one
/// workgroup per 8x8 tile.
pub(crate) const EXPOSURE_HISTOGRAM_WORKGROUP_SIZE: u32 = 8;

/// Immediate block consumed by `exposure.wesl`'s `build_histogram` entry point.
///
/// Mirrors the shader's `HistogramParams`: the `[min, max]` log2-luminance
/// window the bins span, then the framebuffer extent used to drop the ragged
/// tile edge. Four scalars fill exactly the 16-byte immediate alignment with no
/// implicit padding.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuExposureHistogramConfig {
    /// Log2 luminance mapped to the first bin's lower edge.
    pub min_log2: f32,
    /// Log2 luminance mapped to the last bin's upper edge.
    pub max_log2: f32,
    /// Framebuffer width in texels.
    pub width: u32,
    /// Framebuffer height in texels.
    pub height: u32,
}

impl GpuExposureHistogramConfig {
    /// Builds the histogram config from the golden log2 window and the
    /// framebuffer extent.
    pub(crate) fn from_view(range: HistogramRange, size: UVec2) -> Self {
        Self {
            min_log2: range.min_log2_luminance,
            max_log2: range.max_log2_luminance,
            width: size.x,
            height: size.y,
        }
    }
}

/// Immediate block consumed by `exposure.wesl`'s `resolve_exposure` entry point.
///
/// Mirrors the shader's `ResolveParams`: the histogram log2 window, the
/// percentile trim, the auto-exposure EV100 clamp + compensation, the
/// eye-adaptation speeds, and the wall-clock frame delta the exponential
/// adaptation integrates over. Twelve `f32`s fill 48 bytes, a multiple of the
/// 16-byte immediate alignment; the two trailing pads keep the block explicit.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuExposureResolveConfig {
    /// Log2 luminance mapped to the first bin's lower edge.
    pub min_log2: f32,
    /// Log2 luminance mapped to the last bin's upper edge.
    pub max_log2: f32,
    /// Fraction of the darkest samples to trim before averaging.
    pub low_percent: f32,
    /// Fraction of the brightest samples to trim before averaging.
    pub high_percent: f32,
    /// Lower EV100 clamp on the metered exposure.
    pub min_ev100: f32,
    /// Upper EV100 clamp on the metered exposure.
    pub max_ev100: f32,
    /// Exposure compensation in stops added to the metered EV100.
    pub compensation_stops: f32,
    /// Eye-adaptation rate when brightening (dark -> light), in stops/second.
    pub speed_up: f32,
    /// Eye-adaptation rate when darkening (light -> dark), in stops/second.
    pub speed_down: f32,
    /// Wall-clock seconds since the previous resolve, driving the exponential
    /// adaptation response.
    pub delta_seconds: f32,
    /// Padding to keep the block a multiple of the 16-byte immediate alignment.
    pub _pad0: f32,
    /// Padding to keep the block a multiple of the 16-byte immediate alignment.
    pub _pad1: f32,
}

impl GpuExposureResolveConfig {
    /// Builds the resolve config from the golden exposure settings and the
    /// per-frame wall-clock delta.
    pub(crate) fn new(
        range: HistogramRange,
        percentiles: HistogramPercentiles,
        settings: AutoExposureSettings,
        adaptation: EyeAdaptation,
        delta_seconds: f32,
    ) -> Self {
        Self {
            min_log2: range.min_log2_luminance,
            max_log2: range.max_log2_luminance,
            low_percent: percentiles.low,
            high_percent: percentiles.high,
            min_ev100: settings.min_ev100,
            max_ev100: settings.max_ev100,
            compensation_stops: settings.compensation_stops,
            speed_up: adaptation.speed_up,
            speed_down: adaptation.speed_down,
            delta_seconds,
            _pad0: 0.0,
            _pad1: 0.0,
        }
    }
}

/// Persistent per-view exposure state carried across frames, mirroring the
/// shader's `ExposureState`. `adapted_exposure` is the multiplier the composite
/// applies; `adapted_luminance` is the smoothed metered luminance the next
/// frame eases from; `average_luminance` is the raw metered value (diagnostic).
/// Initialised so the first frame is a no-op multiply and adaptation eases from
/// a mid-grey rather than from black (which would clamp to an over-exposed EV).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuExposureState {
    /// Exposure multiplier the composite applies to radiance.
    pub adapted_exposure: f32,
    /// Smoothed metered luminance the next frame's adaptation eases from.
    pub adapted_luminance: f32,
    /// Raw percentile-trimmed metered luminance (diagnostic).
    pub average_luminance: f32,
    /// Padding to the 16-byte storage alignment.
    pub _pad0: f32,
}

impl Default for GpuExposureState {
    fn default() -> Self {
        Self {
            adapted_exposure: 1.0,
            adapted_luminance: 1.0,
            average_luminance: 0.0,
            _pad0: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{align_of, size_of};

    #[test]
    fn histogram_config_matches_the_shader_immediate_layout() {
        // Two f32 window bounds plus the two u32 extent fields fill exactly the
        // 16-byte immediate alignment with no implicit padding.
        assert_eq!(size_of::<GpuExposureHistogramConfig>(), 16);
        assert_eq!(align_of::<GpuExposureHistogramConfig>(), 4);
    }

    #[test]
    fn histogram_config_folds_in_the_golden_window_and_extent() {
        let config = GpuExposureHistogramConfig::from_view(
            HistogramRange::default(),
            UVec2::new(1920, 1080),
        );
        assert_eq!(config.min_log2, -10.0);
        assert_eq!(config.max_log2, 12.0);
        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
    }

    #[test]
    fn resolve_config_matches_the_shader_immediate_layout() {
        // Twelve f32 (ten live + two pad) fill 48 bytes, a multiple of the
        // 16-byte immediate alignment.
        assert_eq!(size_of::<GpuExposureResolveConfig>(), 48);
        assert_eq!(align_of::<GpuExposureResolveConfig>(), 4);
    }

    #[test]
    fn resolve_config_folds_in_the_golden_defaults() {
        let config = GpuExposureResolveConfig::new(
            HistogramRange::default(),
            HistogramPercentiles::default(),
            AutoExposureSettings::default(),
            EyeAdaptation::default(),
            1.0 / 60.0,
        );
        assert_eq!(config.min_log2, -10.0);
        assert_eq!(config.max_log2, 12.0);
        assert_eq!(config.low_percent, 0.5);
        assert_eq!(config.high_percent, 0.1);
        assert_eq!(config.min_ev100, -8.0);
        assert_eq!(config.max_ev100, 16.0);
        assert_eq!(config.compensation_stops, 0.0);
        assert_eq!(config.speed_up, 3.0);
        assert_eq!(config.speed_down, 1.0);
        assert_eq!(config.delta_seconds, 1.0 / 60.0);
    }

    #[test]
    fn state_matches_the_shader_layout_and_defaults_to_a_no_op_multiply() {
        assert_eq!(size_of::<GpuExposureState>(), 16);
        assert_eq!(align_of::<GpuExposureState>(), 4);
        let state = GpuExposureState::default();
        assert_eq!(state.adapted_exposure, 1.0);
        assert_eq!(state.adapted_luminance, 1.0);
        assert_eq!(state.average_luminance, 0.0);
    }

    #[test]
    fn workgroup_edge_and_bin_count_match_the_shader() {
        assert_eq!(EXPOSURE_HISTOGRAM_WORKGROUP_SIZE, 8);
        assert_eq!(EXPOSURE_BIN_COUNT, 64);
    }
}
