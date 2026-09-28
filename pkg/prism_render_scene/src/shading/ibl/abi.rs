//! ABI shared between the IBL precompute passes and their WESL twins.
//!
//! Two split-sum halves are precomputed on the GPU:
//!
//! * the view-independent environment BRDF ("DFG") table
//!   (`shaders/brdf_lut.wesl`), driven by [`GpuBrdfLutConfig`), and
//! * the prefiltered radiance cube-map mip chain
//!   (`shaders/env_prefilter.wesl`), driven per output mip by
//!   [`GpuPrefilterConfig`].
//!
//! Both configs are 16-byte `#[repr(C)]` immediate blocks mirroring their
//! shader structs field-for-field so machines with and without a GPU agree.

use bytemuck::{Pod, Zeroable};

/// Workgroup size (per axis) of the `integrate_brdf_lut` compute entry point.
///
/// Must match `@workgroup_size(N, N, 1)` in `shaders/brdf_lut.wesl`; the
/// dispatch rounds the table resolution up to a multiple of this on both axes.
pub(crate) const BRDF_LUT_WORKGROUP_SIZE: u32 = 8;

/// Workgroup size (per axis) of the `prefilter_env_map` compute entry point.
///
/// Must match `@workgroup_size(N, N, 1)` in `shaders/env_prefilter.wesl`; each
/// output mip is dispatched at `div_ceil(mip_size, N)` groups on x/y and six
/// groups on z (one per cube face).
pub(crate) const ENV_PREFILTER_WORKGROUP_SIZE: u32 = 8;

/// Immediate (push-constant) block consumed by `brdf_lut.wesl`.
///
/// Only `sample_count` is live; the three trailing `u32`s pad the block out to
/// the 16-byte alignment WGSL requires of an immediate struct and mirror the
/// shader's `BrdfLutConfig` field-for-field.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq, Eq)]
pub(crate) struct GpuBrdfLutConfig {
    /// GGX importance samples integrated per texel (clamped `>= 1` in-shader).
    pub sample_count: u32,
    /// Padding to satisfy the 16-byte immediate alignment.
    pub _pad0: u32,
    /// Padding to satisfy the 16-byte immediate alignment.
    pub _pad1: u32,
    /// Padding to satisfy the 16-byte immediate alignment.
    pub _pad2: u32,
}

impl GpuBrdfLutConfig {
    /// Builds a config for `sample_count` GGX importance samples, clamping the
    /// count to at least one so the shader's integral never divides by zero.
    pub(crate) const fn new(sample_count: u32) -> Self {
        Self {
            sample_count: if sample_count == 0 { 1 } else { sample_count },
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        }
    }
}

/// Immediate (push-constant) block consumed by `env_prefilter.wesl`, uploaded
/// once per output mip level.
///
/// Mirrors the shader's `PrefilterConfig` field-for-field: the perceptual
/// `roughness` baked into this mip, the GGX importance-sample count, the edge
/// length of the output face in texels, and one `u32` of padding rounding the
/// block up to the 16-byte immediate alignment.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuPrefilterConfig {
    /// Perceptual roughness baked into this mip, in `[0, 1]`.
    pub roughness: f32,
    /// GGX importance samples convolved per output texel (clamped `>= 1`).
    pub sample_count: u32,
    /// Edge length of this output mip's cube face in texels.
    pub mip_size: u32,
    /// Padding to satisfy the 16-byte immediate alignment.
    pub _pad0: u32,
}

impl GpuPrefilterConfig {
    /// Builds a config for one output mip.
    ///
    /// `sample_count` is clamped to at least one so the convolution never
    /// divides by zero; `roughness` is clamped to `[0, 1]` to match the golden
    /// and the shader's expectations.
    pub(crate) fn new(roughness: f32, sample_count: u32, mip_size: u32) -> Self {
        Self {
            roughness: roughness.clamp(0.0, 1.0),
            sample_count: sample_count.max(1),
            mip_size,
            _pad0: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_matches_the_shader_immediate_layout() {
        // `BrdfLutConfig` in brdf_lut.wesl is four `u32`s = 16 bytes, and the
        // immediate block must be 16-byte aligned.
        assert_eq!(size_of::<GpuBrdfLutConfig>(), 16);
        assert_eq!(align_of::<GpuBrdfLutConfig>(), 4);
    }

    #[test]
    fn new_clamps_the_sample_count_to_at_least_one() {
        assert_eq!(GpuBrdfLutConfig::new(0).sample_count, 1);
        assert_eq!(GpuBrdfLutConfig::new(1024).sample_count, 1024);
    }

    #[test]
    fn prefilter_config_matches_the_shader_immediate_layout() {
        // `PrefilterConfig` in env_prefilter.wesl is `f32 + u32 + u32 + u32` =
        // 16 bytes, 16-byte aligned as an immediate block.
        assert_eq!(size_of::<GpuPrefilterConfig>(), 16);
        assert_eq!(align_of::<GpuPrefilterConfig>(), 4);
    }

    #[test]
    fn prefilter_config_clamps_roughness_and_sample_count() {
        let low = GpuPrefilterConfig::new(-1.0, 0, 32);
        assert_eq!(low.roughness, 0.0);
        assert_eq!(low.sample_count, 1);
        assert_eq!(low.mip_size, 32);

        let high = GpuPrefilterConfig::new(2.0, 256, 8);
        assert_eq!(high.roughness, 1.0);
        assert_eq!(high.sample_count, 256);
        assert_eq!(high.mip_size, 8);
    }
}
