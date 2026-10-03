//! ABI shared between the DDGI irradiance-volume sample pass and its WESL twin
//! (`shaders/ddgi_sample.wesl`).
//!
//! The subsystem carries one uniform-buffer block describing the probe lattice
//! and field metadata ([`GpuDdgiVolume`], the std430/uniform twin of the golden
//! [`prism_render_shading::gi::irradiance_volume`] `ProbeGrid` plus the
//! octahedral resolutions and the Majercik/RTXGI tunables), and one immediate
//! (push-constant) block carrying the per-dispatch reconstruction transform
//! ([`GpuDdgiSampleParams`]).
//!
//! WGSL gives `vec3<f32>` a 16-byte alignment but a 12-byte size, so a trailing
//! scalar packs into the slot that follows each `vec3`: `origin` + the
//! `irradiance_interior` `u32` fill one 16-byte row, `spacing` +
//! `depth_interior` the next, and `counts` + a padding `u32` the third. The
//! scalar tail then packs to the 16-byte boundary, giving an 80-byte block with
//! no implicit padding (asserted by the tests). The immediate block leads with
//! the `mat4x4` for its 16-byte alignment, giving `64 + 8 + 8 = 80` bytes.

use bevy_math::Mat4;
use bytemuck::{Pod, Zeroable};

/// Workgroup size (per axis) of the DDGI sample compute entry point.
///
/// Must match `@workgroup_size(N, N, 1)` in `ddgi_sample.wesl`; the dispatch
/// rounds its target extent up to a multiple of this on both axes and the
/// shader bounds-checks every invocation.
pub(crate) const DDGI_WORKGROUP_SIZE: u32 = 8;

/// Uniform-buffer twin of the WESL `DdgiVolume` struct: the probe lattice plus
/// the octahedral field metadata and the Majercik/RTXGI tunables.
///
/// Mirrors the golden [`prism_render_shading::gi::irradiance_volume`]
/// `ProbeGrid` (`origin` / `spacing` / `counts`) and adds the device-side
/// octahedral interior resolutions and the probe-update / sampling tunables
/// (`normal_bias`, `view_bias`, `hysteresis`, `depth_sharpness`, `intensity`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuDdgiVolume {
    /// World-space position of probe coordinate `(0, 0, 0)`.
    pub origin: [f32; 3],
    /// Interior octahedral resolution of each irradiance probe (texels per
    /// axis, excluding the one-texel gutter border).
    pub irradiance_interior: u32,
    /// World-space spacing between adjacent probes along each axis.
    pub spacing: [f32; 3],
    /// Interior octahedral resolution of each depth / visibility probe.
    pub depth_interior: u32,
    /// Probe counts along each axis (each component is `>= 1`).
    pub counts: [i32; 3],
    /// Padding to the 16-byte row boundary after `counts`.
    pub _pad_counts: u32,
    /// World-space normal bias pulling the shading point along its normal
    /// before the probe lookup (reduces self-intersection / light leak).
    pub normal_bias: f32,
    /// World-space self-shadow bias subtracted from the probe->point distance
    /// before the Chebyshev visibility test.
    pub view_bias: f32,
    /// Temporal hysteresis used by the probe-update pass (kept here for ABI
    /// parity so the sample and update passes share one block).
    pub hysteresis: f32,
    /// Depth cosine exponent used by the probe-update pass (kept for parity).
    pub depth_sharpness: f32,
    /// Artistic gain baked into the resolved irradiance.
    pub intensity: f32,
    /// Padding to the 16-byte immediate boundary.
    pub _pad0: f32,
    /// Padding to the 16-byte immediate boundary.
    pub _pad1: f32,
    /// Padding to the 16-byte immediate boundary.
    pub _pad2: f32,
}

impl GpuDdgiVolume {
    /// Builds the volume block from the golden lattice and the device-side
    /// octahedral resolutions / tunables.
    ///
    /// `counts` is clamped to `>= 1` per axis to match the golden
    /// `ProbeGrid::new` sanitisation and keep the probe-storage indexing in
    /// range; `irradiance_interior` / `depth_interior` are clamped to `>= 1`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        origin: [f32; 3],
        spacing: [f32; 3],
        counts: [i32; 3],
        irradiance_interior: u32,
        depth_interior: u32,
        normal_bias: f32,
        view_bias: f32,
        hysteresis: f32,
        depth_sharpness: f32,
        intensity: f32,
    ) -> Self {
        Self {
            origin,
            irradiance_interior: irradiance_interior.max(1),
            spacing,
            depth_interior: depth_interior.max(1),
            counts: [counts[0].max(1), counts[1].max(1), counts[2].max(1)],
            _pad_counts: 0,
            normal_bias,
            view_bias,
            hysteresis,
            depth_sharpness,
            intensity,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        }
    }

    /// Total number of probes in the lattice (`counts.x * counts.y * counts.z`).
    pub(crate) fn probe_count(&self) -> u32 {
        (self.counts[0].max(1) as u32)
            * (self.counts[1].max(1) as u32)
            * (self.counts[2].max(1) as u32)
    }
}

/// Immediate (push-constant) twin of the WESL `SampleParams` struct.
///
/// Carries the clip->world transform (to reconstruct world position from the
/// depth buffer), the framebuffer extent (to bounds-check each invocation and
/// form the sampling UV), the near-plane distance (reserved for depth
/// linearisation), and an artistic gain applied on top of the volume intensity.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuDdgiSampleParams {
    /// Clip -> world transform, column-major via [`Mat4::to_cols_array`];
    /// leads the block for its 16-byte alignment.
    pub clip_to_world: [f32; 16],
    /// Full-resolution framebuffer extent in texels (`vec2<f32>`).
    pub screen_size: [f32; 2],
    /// Positive near-plane distance (reserved for depth linearisation).
    pub near: f32,
    /// Artistic gain applied on top of [`GpuDdgiVolume::intensity`].
    pub intensity: f32,
}

impl GpuDdgiSampleParams {
    /// Builds the immediate block from the clip->world transform, the
    /// framebuffer extent, the near-plane distance and the artistic gain.
    pub(crate) fn new(
        clip_to_world: Mat4,
        screen_size: [f32; 2],
        near: f32,
        intensity: f32,
    ) -> Self {
        Self {
            clip_to_world: clip_to_world.to_cols_array(),
            screen_size,
            near,
            intensity,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn volume_is_the_80_byte_uniform_block() {
        // vec3+u32 (16) x3 + five scalars + three pads (32) = 80 bytes, a
        // multiple of 16 with no implicit padding.
        assert_eq!(size_of::<GpuDdgiVolume>(), 80);
        assert_eq!(align_of::<GpuDdgiVolume>(), 4);
    }

    #[test]
    fn sample_params_is_the_80_byte_immediate_block() {
        // mat4x4 (64) + vec2<f32> (8) + two scalars (8) = 80 bytes.
        assert_eq!(size_of::<GpuDdgiSampleParams>(), 80);
        assert_eq!(align_of::<GpuDdgiSampleParams>(), 4);
    }

    #[test]
    fn workgroup_constant_matches_the_shader() {
        assert_eq!(DDGI_WORKGROUP_SIZE, 8);
    }

    #[test]
    fn volume_new_sanitises_counts_and_resolutions() {
        let v = GpuDdgiVolume::new(
            [1.0, 2.0, 3.0],
            [0.5, 0.5, 0.5],
            [0, -4, 8],
            0,
            0,
            0.1,
            0.1,
            0.97,
            50.0,
            1.0,
        );
        assert_eq!(v.counts, [1, 1, 8]);
        assert_eq!(v.irradiance_interior, 1);
        assert_eq!(v.depth_interior, 1);
        assert_eq!(v.probe_count(), 8);
        assert_eq!(v._pad_counts, 0);
        assert_eq!(v.origin, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn sample_params_round_trips_the_matrix() {
        let params = GpuDdgiSampleParams::new(Mat4::IDENTITY, [1920.0, 1080.0], 0.1, 1.0);
        assert_eq!(params.clip_to_world, Mat4::IDENTITY.to_cols_array());
        assert_eq!(params.screen_size, [1920.0, 1080.0]);
        assert_eq!(params.near, 0.1);
        assert_eq!(params.intensity, 1.0);
    }
}
