//! ABI shared between the depth-of-field compute passes and
//! `shaders/dof.wesl`.
//!
//! Unlike the single-block motion-blur ABI, the three `DoF` passes each carry
//! their *own* immediate (push-constant) block, mirroring the shader's three
//! `var<immediate>` globals ([`GpuDofCocParams`], [`GpuDofGatherParams`],
//! [`GpuDofCompositeParams`]). `naga` prunes the immediate globals a given entry
//! point does not touch, so each specialized pipeline sees exactly one block —
//! the same per-entry-point pruning the motion-blur bindings rely on. Every
//! field mirrors its shader struct byte-for-byte so machines with and without a
//! GPU agree with the CPU golden in [`prism_render_shading::dof`].
//!
//! The `coc` block leads with a `mat4x4<f32>` inverse projection, which forces
//! the struct's size to a multiple of the 16-byte immediate alignment WGSL
//! requires: 64 (matrix) + 8 (extent) + 24 (six optics `f32`) + 4 (near) + 12
//! (three `f32` pads) = 112 bytes. The `gather` and `composite` blocks are a
//! tight 16 bytes each (`vec2<u32>` extent + two trailing 4-byte scalars).

use bevy_math::{Mat4, UVec2};
use bytemuck::{Pod, Zeroable};

use super::settings::PrismDofSettings;

/// Workgroup size (per axis) of every depth-of-field compute entry point.
///
/// Must match `@workgroup_size(N, N, 1)` in `dof.wesl`; each dispatch rounds
/// its target extent up to a multiple of this on both axes and the shaders
/// bounds-check every invocation.
pub(crate) const DOF_WORKGROUP_SIZE: u32 = 8;

/// Immediate (push-constant) block consumed by the `dof_coc` entry point.
///
/// Mirrors the shader's `GpuDofCocParams`: the `view_from_clip` inverse
/// projection (reconstructs a linear view-space distance from reverse-Z device
/// depth), the framebuffer extent, the thin-lens optics, the scene-unit->mm
/// scale and the near plane, then three trailing `f32` pads to the 16-byte
/// immediate alignment the matrix forces on the struct.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuDofCocParams {
    /// Clip -> view (inverse projection); reconstructs a linear view-space
    /// position from the reverse-Z device depth. Uploaded column-major via
    /// [`Mat4::to_cols_array`].
    pub view_from_clip: [f32; 16],
    /// Full-resolution framebuffer extent in texels.
    pub screen_size: [u32; 2],
    /// Distance the lens is focused at, in millimetres (golden `focus_distance`).
    pub focus_distance_mm: f32,
    /// Lens focal length in millimetres (golden `focal_length`).
    pub focal_length_mm: f32,
    /// Aperture f-number / f-stop (golden `aperture_f_stop`).
    pub aperture_f_stop: f32,
    /// Sensor width in millimetres (golden `sensor_size` / `sensor_width_mm`).
    pub sensor_width_mm: f32,
    /// Rendered image width in pixels for the mm->pixel `CoC` conversion.
    pub image_width_px: f32,
    /// Scene-unit -> millimetre scale (the golden optics are in mm while the
    /// reconstructed view distance is in scene units).
    pub scene_units_to_mm: f32,
    /// Positive near-plane distance in front of the camera along -Z.
    pub near: f32,
    /// Padding to satisfy the 16-byte immediate alignment.
    pub _pad0: f32,
    /// Padding to satisfy the 16-byte immediate alignment.
    pub _pad1: f32,
    /// Padding to satisfy the 16-byte immediate alignment.
    pub _pad2: f32,
}

/// Immediate block consumed by the `dof_gather` entry point.
///
/// Mirrors the shader's `GpuDofGatherParams`: the framebuffer extent, the max
/// `CoC` radius the disk gather is bounded by, and the per-field spiral tap count.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuDofGatherParams {
    /// Full-resolution framebuffer extent in texels.
    pub screen_size: [u32; 2],
    /// Largest gather radius (pixels); bounds the near/far disks.
    pub max_coc_pixels: f32,
    /// Disk taps per field (near and far each spiral this many samples).
    pub sample_count: u32,
}

/// Immediate block consumed by the `dof_composite` entry point.
///
/// Mirrors the shader's `GpuDofCompositeParams`: the framebuffer extent, the max
/// `CoC` radius the blend saturates at (golden `max_coc_pixels`), and the global
/// effect scale (golden `enabled_scale`; 0 disables `DoF`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuDofCompositeParams {
    /// Full-resolution framebuffer extent in texels.
    pub screen_size: [u32; 2],
    /// Largest gather radius (pixels) the blend saturates at.
    pub max_coc_pixels: f32,
    /// Global effect scale in `[0, 1]`; 0 disables `DoF`.
    pub enabled_scale: f32,
}

impl GpuDofCocParams {
    /// Builds the CoC-prepass block from the inverse projection, the framebuffer
    /// extent and the live [`PrismDofSettings`].
    ///
    /// The matrix is uploaded column-major (via [`Mat4::to_cols_array`]) so the
    /// WGSL `mat4x4<f32>` multiply agrees byte-for-byte with the reconstruction
    /// the SSR prepass validated.
    pub(crate) fn from_settings(
        view_from_clip: Mat4,
        size: UVec2,
        settings: &PrismDofSettings,
    ) -> Self {
        Self {
            view_from_clip: view_from_clip.to_cols_array(),
            screen_size: [size.x, size.y],
            focus_distance_mm: settings.focus_distance_mm,
            focal_length_mm: settings.focal_length_mm,
            aperture_f_stop: settings.aperture_f_stop,
            sensor_width_mm: settings.sensor_width_mm,
            image_width_px: settings.image_width_px,
            scene_units_to_mm: settings.scene_units_to_mm,
            near: settings.near,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        }
    }
}

impl GpuDofGatherParams {
    /// Builds the gather block from the framebuffer extent and the live
    /// [`PrismDofSettings`], flooring the tap count at one so a zero request
    /// still runs the centre tap rather than dividing by an empty gather.
    pub(crate) fn from_settings(size: UVec2, settings: &PrismDofSettings) -> Self {
        Self {
            screen_size: [size.x, size.y],
            max_coc_pixels: settings.max_coc_pixels,
            sample_count: settings.sample_count.max(1),
        }
    }
}

impl GpuDofCompositeParams {
    /// Builds the composite block from the framebuffer extent and the live
    /// [`PrismDofSettings`].
    pub(crate) fn from_settings(size: UVec2, settings: &PrismDofSettings) -> Self {
        Self {
            screen_size: [size.x, size.y],
            max_coc_pixels: settings.max_coc_pixels,
            enabled_scale: settings.enabled_scale,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coc_params_is_the_112_byte_immediate_block() {
        // mat4x4 (64) + extent (8) + six optics f32 (24) + near (4) + three
        // f32 pads (12) fill 112 bytes, a multiple of the 16-byte immediate
        // alignment the mat4x4 field forces on the struct.
        assert_eq!(size_of::<GpuDofCocParams>(), 112);
        assert_eq!(align_of::<GpuDofCocParams>(), 4);
    }

    #[test]
    fn gather_and_composite_params_are_tight_16_byte_blocks() {
        assert_eq!(size_of::<GpuDofGatherParams>(), 16);
        assert_eq!(align_of::<GpuDofGatherParams>(), 4);
        assert_eq!(size_of::<GpuDofCompositeParams>(), 16);
        assert_eq!(align_of::<GpuDofCompositeParams>(), 4);
    }

    #[test]
    fn workgroup_constant_matches_the_shader() {
        assert_eq!(DOF_WORKGROUP_SIZE, 8);
    }

    #[test]
    fn coc_from_settings_uploads_the_matrix_column_major_and_folds_optics() {
        let inv = Mat4::from_cols_array(&[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ]);
        let settings = PrismDofSettings::default();
        let params = GpuDofCocParams::from_settings(inv, UVec2::new(1920, 1080), &settings);
        assert_eq!(params.view_from_clip, inv.to_cols_array());
        assert_eq!(params.screen_size, [1920, 1080]);
        // Golden `DofParams` / `DofCamera` optical defaults.
        assert_eq!(params.focus_distance_mm, 2000.0);
        assert_eq!(params.focal_length_mm, 50.0);
        assert_eq!(params.aperture_f_stop, 16.0);
        assert_eq!(params.sensor_width_mm, 36.0);
        assert_eq!(params.image_width_px, 1920.0);
        assert_eq!(params.scene_units_to_mm, 1000.0);
        assert_eq!(params._pad0, 0.0);
        assert_eq!(params._pad1, 0.0);
        assert_eq!(params._pad2, 0.0);
    }

    #[test]
    fn gather_from_settings_floors_the_sample_count_to_one() {
        let settings = PrismDofSettings {
            sample_count: 0,
            ..Default::default()
        };
        let params = GpuDofGatherParams::from_settings(UVec2::new(64, 64), &settings);
        assert_eq!(params.sample_count, 1);
        assert_eq!(params.screen_size, [64, 64]);
        assert_eq!(params.max_coc_pixels, 32.0);
    }

    #[test]
    fn composite_from_settings_folds_the_blend_tunables() {
        let settings = PrismDofSettings::default();
        let params = GpuDofCompositeParams::from_settings(UVec2::new(800, 600), &settings);
        assert_eq!(params.screen_size, [800, 600]);
        assert_eq!(params.max_coc_pixels, 32.0);
        // Golden default is disabled (identity blend).
        assert_eq!(params.enabled_scale, 0.0);
    }
}
