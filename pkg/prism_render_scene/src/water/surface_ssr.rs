//! The water-surface pass's **screen-space reflection** `@group(3)` plumbing:
//! the `CPU` half that lets the §5 lighting fork march the engine's existing
//! reverse-Z Hi-Z "nearest depth" pyramid (built by the opaque
//! [`ViewSsrTextures`](crate::shading::ViewSsrTextures) prepass) directly inside
//! the transparent water fragment stage.
//!
//! ## Why a forward march instead of sampling the resolved `SSR` buffer
//!
//! The opaque `SSR` chain (`shading::ssr`) traces reflections for every opaque
//! pixel and resolves them into a view texture *before* the water surface is
//! drawn. That resolved buffer stores, for each screen pixel, the reflection of
//! the **opaque surface that occupies that pixel** — for a water pixel that is
//! the submerged terrain behind the water, not the water surface itself.
//! Sampling it on the water would reflect the lake bed, which is physically
//! wrong. Instead the water fragment reconstructs its own view-space position
//! and normal, reflects the view ray about the animated surface normal, and
//! marches the shared Hi-Z pyramid itself, sampling the current-frame
//! `scene_color` at the hit. This reuses the engine's existing depth pyramid
//! (no extra prepass) and matches the `CPU` golden
//! [`prism_render_shading::screen_space`] march convention the opaque
//! `ssr.wesl` twin already encodes.
//!
//! The sibling [`super::surface_pipeline`] slice owns the raster pipelines and
//! the `@group(0)`/`@group(1)` layouts; [`super::surface_vsm`] owns the
//! `@group(2)` virtual-shadow-map group. This slice adds the fourth bind group.
//!
//! ## Why a separate fallback resource
//!
//! Mirroring [`super::surface_vsm::WaterVsmFallback`]: the
//! [`WaterSurfacePipelines`](super::surface_pipeline::WaterSurfacePipelines)
//! specializer is a pure-`CPU` descriptor bag with no [`RenderDevice`], so the
//! draw node must bind a real, format-correct Hi-Z texture even on a view with
//! no resident [`ViewSsrTextures`](crate::shading::ViewSsrTextures) (feature off
//! or first frame). [`WaterSsrFallback`] holds a 1x1 `R32Float` view, built once
//! at `RenderStartup` by [`init_water_ssr_fallback`]; the draw node clears the
//! `sample_enable` bit whenever it binds the fallback so the shader skips the
//! march and keeps the image-based reflection.

use bevy_ecs::prelude::*;
use bevy_material::bind_group_layout_entries::{
    binding_types::{texture_2d, uniform_buffer_sized},
    BindGroupLayoutEntries,
};
use bevy_math::Mat4;
use bevy_render::{
    render_resource::{
        Extent3d, ShaderStages, TextureDescriptor, TextureDimension, TextureFormat,
        TextureSampleType, TextureUsages, TextureView, TextureViewDescriptor,
    },
    renderer::RenderDevice,
};
use bytemuck::{Pod, Zeroable};

/// `GPU`-side mirror of the shader's `WaterSsrConfig` uniform (the water
/// `@group(3) @binding(1)` block).
///
/// Layout matches the `WGSL` struct byte-for-byte: three `mat4x4<f32>` (64 B
/// each at offsets 0/64/128) precede the scalars. `clip_from_view` projects the
/// reflected view ray to framebuffer `UV` + reverse-Z device depth,
/// `view_from_clip` is unused by the forward path (kept for parity with the
/// opaque `SsrConfig` and future depth-reconstruction use), and
/// `view_from_world` rotates the animated world-space surface normal and
/// position into view space. The two trailing `u32`s pad the block to the
/// 16-byte alignment the `WGSL` uniform address space requires for a struct
/// whose widest member is a `mat4x4<f32>`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuWaterSsrConfig {
    /// View -> clip (reverse-Z), column-major.
    pub clip_from_view: [f32; 16],
    /// Clip -> view (inverse projection), column-major.
    pub view_from_clip: [f32; 16],
    /// World -> view, column-major (rotates the surface normal/position).
    pub view_from_world: [f32; 16],
    /// Positive near-plane distance in front of the camera along `-Z`.
    pub near: f32,
    /// View-space march length.
    pub max_distance: f32,
    /// Device-depth thin-surface tolerance for accepting a hit.
    pub thickness: f32,
    /// Hard iteration cap for the hierarchical march.
    pub max_iterations: u32,
    /// Screen-border confidence fade start (golden `SsrConfidenceParams`).
    pub edge_fade_start: f32,
    /// Perceptual roughness below which `SSR` is fully trusted.
    pub full_roughness: f32,
    /// Perceptual roughness at/above which `SSR` is fully replaced by `IBL`.
    pub max_roughness: f32,
    /// Fractional march distance at which the distance fade begins.
    pub distance_fade_start: f32,
    /// Finest pyramid mip the march refines to (usually 0).
    pub most_detailed_mip: u32,
    /// `1` when a resident Hi-Z pyramid is bound; `0` binds the fallback and the
    /// shader skips the march (keeps the image-based reflection).
    pub sample_enable: u32,
    /// Padding to the 16-byte block alignment.
    pub _pad0: u32,
    /// Padding to the 16-byte block alignment.
    pub _pad1: u32,
}

impl GpuWaterSsrConfig {
    /// Build the uniform from the view's reverse-Z projection and world->view
    /// transform. The confidence/march tunables adopt the opaque `SSR` golden
    /// defaults (`SsrMarchConfig`/`SsrConfidenceParams`) so the water march
    /// agrees with the shared [`prism_render_shading::screen_space`] reference.
    /// All three matrices upload column-major (via [`Mat4::to_cols_array`]) so
    /// the `WGSL` `mat4x4<f32>` multiply agrees byte-for-byte.
    pub(crate) fn new(
        clip_from_view: Mat4,
        view_from_clip: Mat4,
        view_from_world: Mat4,
        near: f32,
        max_distance: f32,
        sample_enable: bool,
    ) -> Self {
        Self {
            clip_from_view: clip_from_view.to_cols_array(),
            view_from_clip: view_from_clip.to_cols_array(),
            view_from_world: view_from_world.to_cols_array(),
            near,
            max_distance,
            // Golden `SsrMarchConfig::default()`.
            thickness: 0.02,
            max_iterations: 128,
            // Golden `SsrConfidenceParams::default()`, widened on the roughness
            // axis: a water surface is a near-mirror, so trust the march across
            // the full authored roughness range rather than fading it out early.
            edge_fade_start: 0.1,
            full_roughness: 0.3,
            max_roughness: 0.8,
            distance_fade_start: 0.7,
            most_detailed_mip: 0,
            sample_enable: u32::from(sample_enable),
            _pad0: 0,
            _pad1: 0,
        }
    }
}

/// Builds the water-surface `@group(3)` `SSR` layout entries (two bindings), in
/// the exact `@binding(n)` order `water_surface_raster.wesl` declares:
///
/// 0. the reverse-Z Hi-Z "nearest depth" pyramid (non-filterable float; the
///    march reads it with `textureLoad`, so it needs no sampler), and
/// 1. the [`GpuWaterSsrConfig`] uniform.
///
/// Declared [`ShaderStages::FRAGMENT`]: the water surface marches the pyramid in
/// its fragment stage, not a compute pass.
pub(crate) fn ssr_layout_entries() -> BindGroupLayoutEntries<2> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::FRAGMENT,
        (
            texture_2d(TextureSampleType::Float { filterable: false }),
            uniform_buffer_sized(false, None),
        ),
    )
}

/// Format-correct fallback Hi-Z view bound into the surface pass's `@group(3)`
/// on any view without a resident [`ViewSsrTextures`](crate::shading::ViewSsrTextures).
///
/// Built once at `RenderStartup` by [`init_water_ssr_fallback`]; read by the
/// raster draw node ([`super::surface_draw`]). The draw node clears the config's
/// `sample_enable` bit whenever it binds this fallback, so the shader never
/// actually marches it — but the bind group is still structurally complete.
#[derive(Resource)]
pub(crate) struct WaterSsrFallback {
    /// A 1x1 `R32Float` view, format-matched to the real
    /// [`ViewSsrTextures`](crate::shading::ViewSsrTextures) Hi-Z pyramid so the
    /// bind group is valid when no pyramid is resident.
    pub(crate) hzb_view: TextureView,
}

/// `RenderStartup` initializer that builds the surface pass's `SSR` fallback
/// Hi-Z view and inserts [`WaterSsrFallback`].
pub(crate) fn init_water_ssr_fallback(mut commands: Commands, device: Res<RenderDevice>) {
    let dummy_hzb = device.create_texture(&TextureDescriptor {
        label: Some("prism water surface ssr dummy hzb"),
        size: Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        // Format-matched to `ViewSsrTextures`'s `SSR_HZB_FORMAT` (`R32Float`).
        format: TextureFormat::R32Float,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let hzb_view = dummy_hzb.create_view(&TextureViewDescriptor::default());

    commands.insert_resource(WaterSsrFallback { hzb_view });
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{align_of, size_of};

    #[test]
    fn config_matches_the_wgsl_uniform_block_size_and_alignment() {
        // Three `mat4x4<f32>` (192 B) + ten 4-byte scalars + two 4-byte pads =
        // 240 B, a multiple of the 16-byte `WGSL` uniform block alignment.
        assert_eq!(size_of::<GpuWaterSsrConfig>(), 240);
        assert_eq!(align_of::<GpuWaterSsrConfig>(), 4);
    }

    #[test]
    fn new_packs_the_enable_bit_and_golden_defaults() {
        let enabled = GpuWaterSsrConfig::new(
            Mat4::IDENTITY,
            Mat4::IDENTITY,
            Mat4::IDENTITY,
            0.1,
            40.0,
            true,
        );
        assert_eq!(enabled.sample_enable, 1);
        assert_eq!(enabled.max_iterations, 128);
        assert_eq!(enabled.most_detailed_mip, 0);
        assert_eq!(enabled._pad0, 0);
        assert_eq!(enabled._pad1, 0);

        let disabled = GpuWaterSsrConfig::new(
            Mat4::IDENTITY,
            Mat4::IDENTITY,
            Mat4::IDENTITY,
            0.1,
            40.0,
            false,
        );
        assert_eq!(disabled.sample_enable, 0);
    }

    #[test]
    fn new_uploads_each_matrix_column_major() {
        let clip = Mat4::from_cols_array(&[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ]);
        let cfg = GpuWaterSsrConfig::new(clip, Mat4::IDENTITY, Mat4::IDENTITY, 0.1, 40.0, true);
        assert_eq!(cfg.clip_from_view, clip.to_cols_array());
        assert_eq!(cfg.view_from_clip, Mat4::IDENTITY.to_cols_array());
        assert_eq!(cfg.view_from_world, Mat4::IDENTITY.to_cols_array());
    }

    #[test]
    fn layout_declares_two_fragment_bindings() {
        let entries = ssr_layout_entries();
        // `BindGroupLayoutEntries` derefs to the built `[BindGroupLayoutEntry]`.
        assert_eq!(entries.len(), 2);
        for entry in entries.iter() {
            assert!(entry.visibility.contains(ShaderStages::FRAGMENT));
        }
    }
}
