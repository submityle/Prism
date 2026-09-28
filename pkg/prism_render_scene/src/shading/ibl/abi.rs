//! ABI shared between the DFG lookup-table precompute and `shaders/brdf_lut.wesl`.
//!
//! The split-sum environment BRDF ("DFG") table depends only on view angle and
//! roughness, so it is generated once into a global `Rg16Float` texture rather
//! than per view.  The only per-dispatch state the kernel needs is the GGX
//! importance-sample count; the rest (texel-centre `n_dot_v` / `roughness`) is
//! derived from `textureDimensions` inside the shader, exactly like the CPU
//! golden `DfgLut::generate`.

use bytemuck::{Pod, Zeroable};

/// Workgroup size (per axis) of the `integrate_brdf_lut` compute entry point.
///
/// Must match `@workgroup_size(N, N, 1)` in `shaders/brdf_lut.wesl`; the
/// dispatch rounds the table resolution up to a multiple of this on both axes.
pub(crate) const BRDF_LUT_WORKGROUP_SIZE: u32 = 8;

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
}
