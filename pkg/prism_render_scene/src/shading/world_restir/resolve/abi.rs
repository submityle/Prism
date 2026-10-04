//! Device ABI for the world-space `ReSTIR` direct-illumination resolve pass.
//!
//! The resolve runs one compute invocation per screen pixel: it reconstructs
//! the pixel's world shading point from the SSR prepass depth + packed normal
//! (byte-identical to the visible-point producer), re-hashes it into the same
//! `SHARC` cell the inject pass claimed, linear-probes the resident reservoir
//! table for that cell's finalised reservoir and — on a hit — re-evaluates the
//! stored light sample under a reconnection shift, writing the pre-BRDF direct
//! irradiance + hit confidence into the `gi_out` export. This module freezes
//! that pass's immediate (push-constant) block, byte-for-byte with the WESL
//! twin `shaders/world_restir_resolve.wesl`.

use bevy_math::{Mat4, UVec2, Vec3};
use bytemuck::{Pod, Zeroable};

use super::super::settings::PrismWorldRestirSettings;

/// Workgroup edge (both x and y) of the `resolve_main` entry point.
///
/// Must match `@workgroup_size(N, N, 1)` in `world_restir_resolve.wesl`; the
/// dispatch rounds the framebuffer up to a multiple of this on each axis and
/// the shader bounds-checks every invocation against the live `(screen_w,
/// screen_h)`.
pub(crate) const RESOLVE_WORKGROUP_SIZE: u32 = 8;

/// Immediate (push-constant) block consumed by the `resolve_main` entry point.
///
/// Layout (two `mat4x4<f32>` + four `vec4` lanes, std430, no implicit padding =
/// 192 bytes):
/// 0..16.  `view_from_clip` (inverse reverse-Z projection; reconstructs the
///         view-space shading point from a device depth).
/// 16..32. `world_from_view` (lifts the reconstructed point + its normal into
///         world space for the `SHARC` hash).
/// 32..36. `camera_base` = (camera_position.xyz, base_cell_size).
/// 36..40. `jitter_level` = (jitter.xyz, level_scale).
/// 40..44. `dims` = (capacity, normal_resolution, screen_width, screen_height).
/// 44..48. `intensity_pad` = (intensity, 0, 0, 0).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuWorldRestirResolveParams {
    /// Clip -> view (inverse reverse-Z projection), column-major.
    pub view_from_clip: [f32; 16],
    /// View -> world, column-major.
    pub world_from_view: [f32; 16],
    /// (camera_position.xyz, base_cell_size): the hash-grid anchor + level-0
    /// cell edge, forwarded to the golden `grid_level` so the resolve selects
    /// the identical cell size the inject pass claimed the slot at.
    pub camera_base: [f32; 4],
    /// (jitter.xyz, level_scale): grid-phase jitter (zero, matching the
    /// spatial-only inject/fill passes) + the distance-to-cell-size scale.
    pub jitter_level: [f32; 4],
    /// (capacity, normal_resolution, screen_width, screen_height): the
    /// open-addressing modulus + probe-window bound, the normal-bin grid side,
    /// and the framebuffer extent / per-invocation bounds.
    pub dims: [u32; 4],
    /// (intensity, 0, 0, 0): the artistic gain multiplying the resolved
    /// irradiance, mirroring the seed/fill blocks so the magnitudes agree.
    pub intensity_pad: [f32; 4],
}

impl GpuWorldRestirResolveParams {
    /// Builds the resolve immediate block from the per-view matrices, the camera
    /// world position, the framebuffer size and the live settings.
    ///
    /// `jitter` is held at zero to match the spatial-only inject/fill passes so
    /// the resolve re-hashes each pixel into the identical `SHARC` cell the
    /// inject pass claimed its slot in; `capacity` is floored at `1` exactly as
    /// the producer passes do.
    pub(crate) fn new(
        view_from_clip: Mat4,
        world_from_view: Mat4,
        camera_position: Vec3,
        screen: UVec2,
        settings: &PrismWorldRestirSettings,
    ) -> Self {
        Self {
            view_from_clip: view_from_clip.to_cols_array(),
            world_from_view: world_from_view.to_cols_array(),
            camera_base: [
                camera_position.x,
                camera_position.y,
                camera_position.z,
                settings.base_cell_size,
            ],
            jitter_level: [0.0, 0.0, 0.0, settings.level_scale],
            dims: [
                settings.capacity.max(1),
                settings.normal_resolution,
                screen.x,
                screen.y,
            ],
            intensity_pad: [settings.intensity, 0.0, 0.0, 0.0],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{align_of, size_of};

    #[test]
    fn params_is_the_192_byte_six_lane_block() {
        // Two mat4x4 (64 B each) + four vec4 (16 B each) = 192 B, every lane
        // naturally 16-aligned so there is no implicit tail padding. A drift
        // here would mismatch `set_immediates` against the shader's
        // `var<immediate>` block.
        assert_eq!(size_of::<GpuWorldRestirResolveParams>(), 192);
        assert_eq!(align_of::<GpuWorldRestirResolveParams>(), 4);
    }

    #[test]
    fn builder_packs_matrices_column_major_and_settings_in_place() {
        let view_from_clip = Mat4::from_cols_array(&core::array::from_fn(|i| i as f32));
        let world_from_view = Mat4::from_cols_array(&core::array::from_fn(|i| (i as f32) * 2.0));
        let settings = PrismWorldRestirSettings {
            capacity: 4096,
            base_cell_size: 2.5,
            level_scale: 0.75,
            normal_resolution: 8,
            intensity: 1.5,
            ..Default::default()
        };
        let params = GpuWorldRestirResolveParams::new(
            view_from_clip,
            world_from_view,
            Vec3::new(1.0, 2.0, 3.0),
            UVec2::new(1920, 1080),
            &settings,
        );
        assert_eq!(params.view_from_clip, view_from_clip.to_cols_array());
        assert_eq!(params.world_from_view, world_from_view.to_cols_array());
        assert_eq!(params.camera_base, [1.0, 2.0, 3.0, 2.5]);
        // Grid-phase jitter is held at zero to match the spatial-only producer
        // passes; a non-zero phase here would re-hash pixels into the wrong
        // cell and miss every claimed slot.
        assert_eq!(params.jitter_level, [0.0, 0.0, 0.0, 0.75]);
        assert_eq!(params.dims, [4096, 8, 1920, 1080]);
        assert_eq!(params.intensity_pad, [1.5, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn builder_floors_capacity_at_one() {
        let settings = PrismWorldRestirSettings {
            capacity: 0,
            ..Default::default()
        };
        let params = GpuWorldRestirResolveParams::new(
            Mat4::IDENTITY,
            Mat4::IDENTITY,
            Vec3::ZERO,
            UVec2::new(64, 64),
            &settings,
        );
        assert_eq!(params.dims[0], 1);
    }

    #[test]
    fn dispatch_constant_matches_the_wesl_twin() {
        // `@workgroup_size(8, 8, 1)` in `world_restir_resolve.wesl`.
        assert_eq!(RESOLVE_WORKGROUP_SIZE, 8);
    }
}
