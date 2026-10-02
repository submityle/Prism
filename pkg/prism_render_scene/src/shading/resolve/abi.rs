//! ABI shared between the shading-resolve compute pass and
//! `shaders/shading_resolve.wesl`.
//!
//! The resolve entry point is dispatched once per `MaterialShadingClass` with
//! an indirect argument buffer, so the only per-dispatch state it needs is the
//! active class index plus the framebuffer dimensions and the world-space view
//! position used to build the shading frame.  Everything else is read from the
//! storage buffers bound by [`super::bind_groups`].

use bevy_math::Vec3;
use bytemuck::{Pod, Zeroable};
use prism_render_shading::ClipmapConfig;

/// Workgroup size of the `shading_resolve` compute entry point.
///
/// Must match `@workgroup_size(...)` in `shaders/shading_resolve.wesl` and the
/// `local_invocation` bound checks against the per-class work count.
pub(crate) const RESOLVE_WORKGROUP_SIZE: u32 = 64;

/// Immediate (push-constant) block consumed by `shading_resolve.wesl`.
///
/// `shading_class` selects which compacted worklist slice this dispatch drains
/// (`work[class_offsets[class] + local]`).  `view_position` is padded to a
/// 16-byte boundary so the trailing `vec4<f32>` lands on its natural alignment
/// and the Rust/WGSL layouts agree byte-for-byte.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuShadingResolveParams {
    /// Active `MaterialShadingClass` discriminant for this dispatch.
    pub shading_class: u32,
    /// Framebuffer width in pixels (used to unflatten `pixel_index`).
    pub width: u32,
    /// Framebuffer height in pixels.
    pub height: u32,
    /// Packed feature bits; see [`RESOLVE_FLAG_GTAO`] and
    /// [`RESOLVE_FLAG_IBL_SPECULAR`].  Doubles as the 16-byte alignment word
    /// ahead of the `vec4<f32>`.
    pub flags: u32,
    /// World-space camera position; `w` is unused padding.
    pub view_position: [f32; 4],
}

/// `flags` bit selecting the screen-space GTAO occlusion multiply in the
/// resolve shader.  Mirrors `RESOLVE_FLAG_GTAO` in `shading_resolve.wesl`.
pub(crate) const RESOLVE_FLAG_GTAO: u32 = 1 << 0;

/// `flags` bit signalling that the prefiltered environment cube and the DFG
/// table are resident, so the resolve samples the real split-sum specular
/// reflection instead of the low-frequency SH-radiance fallback.  Mirrors
/// `RESOLVE_FLAG_IBL_SPECULAR` in `shading_resolve.wesl`.
pub(crate) const RESOLVE_FLAG_IBL_SPECULAR: u32 = 1 << 1;

/// Uniform block bound at group 6 of the shading-resolve pass, carrying the
/// virtual-shadow-map (VSM) sampling state consumed by the inline
/// `sample_virtual_shadow` twin in `shading_resolve.wesl`.
///
/// The clipmap + physical-atlas geometry fields mirror
/// [`super::super::virtual_shadow::GpuVsmSampleParams`] exactly (same clamps),
/// so the resolve pass addresses virtual pages byte-for-byte with the page-mark
/// and standalone-sample passes.  On top of that this block carries the run-time
/// switch (`enable`) and the light's orthonormal clipmap basis
/// (`light_right`/`light_up`/`light_forward`) so the shader can project a world
/// receiver onto the light plane without a separate matrix upload, plus the
/// `depth_scale`/`depth_bias` that map the receiver's along-light distance into
/// the atlas's stored NDC depth range.
///
/// # Layout
///
/// 96 bytes, std140-clean for a WGSL `uniform`: the first twelve 4-byte scalars
/// fill three 16-byte rows (0..48), then each `vec3<f32>` basis axis lands on a
/// 16-byte boundary (48, 64, 80) with its trailing `f32` filling the fourth
/// column, so the plain `#[repr(C)]` scalar packing is byte-identical to the
/// shader's `VsmResolveParams` with no explicit padding words between the axes.
///
/// # Depth mapping (reconciled with the Slice A caster projection)
///
/// `depth_scale`/`depth_bias` reproduce the exact NDC depth the caster-depth
/// raster writes into the atlas `.r`.  That projection
/// (`super::super::virtual_shadow`'s `atlas::projection`) encodes a world point
/// as `clip.z = (world . light_forward) / (2 D) + 0.5`, clamped to the wgpu
/// `[0, 1]` range, where `D` is the symmetric depth half-extent sized to the
/// coarsest clipmap window (`page_world_size(levels - 1) * pages_per_level_edge`,
/// floored at `1e-3`).  So the resolve reference depth is
/// `reference_depth = (world . light_forward) * depth_scale + depth_bias` with
/// `depth_scale = 1 / (2 D)` and `depth_bias = 0.5` -- the plane through the
/// **world origin** sits at `0.5`, a caster `D` towards the light at the near
/// plane `0` and one `D` away at the far plane `1`.  Measuring from the world
/// origin (not the camera) matches the projection, whose look-at reference is
/// `Vec3::ZERO`; the sampler then compares `reference_depth <= stored => lit`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuVsmResolveParams {
    /// Number of clipmap levels (mirrors [`ClipmapConfig::levels`], `>= 1`).
    pub levels: u32,
    /// Edge length of a page in texels (mirrors [`ClipmapConfig::page_size`]).
    pub page_size: u32,
    /// Pages along one edge of a level's resident window.
    pub pages_per_level_edge: u32,
    /// World page coordinate mapped to key coordinate `0`.
    pub page_coord_bias: i32,
    /// World-space edge length of one texel at the finest level.
    pub level0_texel_world_size: f32,
    /// View distance at or below which level `0` is selected.
    pub level0_max_distance: f32,
    /// Number of physical pages backing the atlas.
    pub physical_pages: u32,
    /// Physical pages along one edge of the square atlas grid.
    pub physical_pages_per_edge: u32,
    /// Soft-shadow PCF box half-width in physical texels.
    pub pcf_radius: i32,
    /// `1` when the resolve should sample the VSM for the primary directional
    /// light, `0` to skip the VSM branch entirely and use the cascaded path.
    pub enable: u32,
    /// Padding word keeping the switch row a whole 16-byte row.
    pub pad0: u32,
    /// Padding word keeping the switch row a whole 16-byte row.
    pub pad1: u32,
    /// Right axis of the light's clipmap plane (unit, `world_xy.x` axis).
    pub light_right: [f32; 3],
    /// Maps the receiver's along-light distance into NDC depth: `1 / (2 D)`,
    /// matching the caster projection's `z` slope (see the type docs).
    pub depth_scale: f32,
    /// Up axis of the light's clipmap plane (unit, `world_xy.y` axis).
    pub light_up: [f32; 3],
    /// Constant NDC depth bias folded into the reference depth: `0.5`, the
    /// world-origin plane the caster projection centres its depth slab on.
    pub depth_bias: f32,
    /// Forward axis of the light (the direction its rays travel); the along-light
    /// projection axis for the reference depth.
    pub light_forward: [f32; 3],
    /// Padding filling the fourth column of the forward-axis row.
    pub pad2: f32,
}

/// Floor on the caster depth half-extent, mirroring
/// `atlas::projection`'s `MIN_DEPTH_HALF_EXTENT`, so the `1 / (2 D)` depth scale
/// stays finite for a degenerate zero-window clipmap.
const MIN_DEPTH_HALF_EXTENT: f32 = 1.0e-3;

/// NDC depth of the world-origin plane the caster projection centres its
/// symmetric depth slab on; the resolve reference depth is biased by it.
const DEPTH_ORIGIN_PLANE_BIAS: f32 = 0.5;

impl GpuVsmResolveParams {
    /// Builds the resolve VSM params from the clipmap config, the physical atlas
    /// geometry, the PCF radius, the run-time enable switch and the light's
    /// orthonormal clipmap basis.
    ///
    /// The clipmap / atlas fields use the identical clamps as
    /// [`super::super::virtual_shadow::GpuVsmSampleParams::new`] so the two
    /// blocks address pages the same way.  `depth_scale`/`depth_bias` reproduce
    /// the caster-depth projection's NDC `z` encoding (see the type docs):
    /// `depth_scale = 1 / (2 D)` and `depth_bias = 0.5`, where `D` is the
    /// coarsest clipmap window (`page_world_size(levels - 1) *
    /// pages_per_level_edge`, floored at `1e-3`) the projection sizes its depth
    /// slab to.
    pub(crate) fn new(
        clipmap: &ClipmapConfig,
        physical_pages: u32,
        physical_pages_per_edge: u32,
        pcf_radius: i32,
        enable: bool,
        light_right: Vec3,
        light_up: Vec3,
        light_forward: Vec3,
    ) -> Self {
        let levels = u32::from(clipmap.levels.max(1));
        // Symmetric world-space depth half-extent the caster projection's
        // `[0, 1]` range spans, recomputed identically to
        // `atlas::projection::caster_depth_half_extent` (which is private to the
        // atlas module): the coarsest level's full resident window, floored so
        // the reciprocal stays finite for a degenerate clipmap.
        let coarsest = clipmap.level_count().saturating_sub(1);
        let depth_half_extent = (clipmap.page_world_size(coarsest)
            * f32::from(clipmap.pages_per_level_edge.max(1)))
        .max(MIN_DEPTH_HALF_EXTENT);
        // Reproduce the projection's `clip.z = (world . forward) / (2 D) + 0.5`:
        // slope `1 / (2 D)` about the world-origin plane at `0.5`.
        let depth_scale = 1.0 / (2.0 * depth_half_extent);
        Self {
            levels,
            page_size: u32::from(clipmap.page_size.max(1)),
            pages_per_level_edge: u32::from(clipmap.pages_per_level_edge.max(1)),
            page_coord_bias: clipmap.page_coord_bias,
            level0_texel_world_size: clipmap.level0_texel_world_size,
            level0_max_distance: clipmap.level0_max_distance,
            physical_pages,
            physical_pages_per_edge: physical_pages_per_edge.max(1),
            pcf_radius: pcf_radius.max(0),
            enable: u32::from(enable),
            pad0: 0,
            pad1: 0,
            light_right: light_right.to_array(),
            depth_scale,
            light_up: light_up.to_array(),
            depth_bias: DEPTH_ORIGIN_PLANE_BIAS,
            light_forward: light_forward.to_array(),
            pad2: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_params_layout_matches_the_wgsl_immediate_block() {
        // vec4<f32> forces 16-byte alignment; the four leading u32 scalars fill
        // the first 16 bytes exactly, so the whole block is 32 bytes.
        assert_eq!(size_of::<GpuShadingResolveParams>(), 32);
        assert_eq!(align_of::<GpuShadingResolveParams>(), 4);
        assert_eq!(RESOLVE_WORKGROUP_SIZE, 64);
    }

    #[test]
    fn resolve_feature_flags_are_distinct_single_bits() {
        // The shader ANDs `flags` with each mask independently, so they must be
        // disjoint powers of two.
        assert_eq!(RESOLVE_FLAG_GTAO, 1);
        assert_eq!(RESOLVE_FLAG_IBL_SPECULAR, 2);
        assert_eq!(RESOLVE_FLAG_GTAO & RESOLVE_FLAG_IBL_SPECULAR, 0);
    }
}

#[cfg(test)]
mod vsm_resolve_tests {
    use super::*;
    use crate::shading::virtual_shadow::{window_slot_count, GpuVsmSampleParams};
    use core::mem::offset_of;
    use prism_render_shading::ClipmapConfig;

    fn clipmap() -> ClipmapConfig {
        ClipmapConfig {
            levels: 4,
            page_size: 128,
            pages_per_level_edge: 8,
            level0_texel_world_size: 0.1,
            level0_max_distance: 10.0,
            page_coord_bias: 32_768,
        }
    }

    #[test]
    fn resolve_vsm_params_layout_is_std140_clean_96_bytes() {
        // Three 16-byte scalar rows then three vec3+scalar rows: the vec3 axes
        // must land on their 16-byte boundaries with the trailing scalar filling
        // the fourth column, or the GPU would read the basis shifted.
        assert_eq!(size_of::<GpuVsmResolveParams>(), 96);
        assert_eq!(align_of::<GpuVsmResolveParams>(), 4);
        assert_eq!(offset_of!(GpuVsmResolveParams, levels), 0);
        assert_eq!(offset_of!(GpuVsmResolveParams, level0_texel_world_size), 16);
        assert_eq!(offset_of!(GpuVsmResolveParams, pcf_radius), 32);
        assert_eq!(offset_of!(GpuVsmResolveParams, light_right), 48);
        assert_eq!(offset_of!(GpuVsmResolveParams, depth_scale), 60);
        assert_eq!(offset_of!(GpuVsmResolveParams, light_up), 64);
        assert_eq!(offset_of!(GpuVsmResolveParams, depth_bias), 76);
        assert_eq!(offset_of!(GpuVsmResolveParams, light_forward), 80);
        assert_eq!(offset_of!(GpuVsmResolveParams, pad2), 92);
    }

    #[test]
    fn clipmap_fields_match_the_sample_params_oracle() {
        // The resolve block's clipmap/atlas fields must clamp and map exactly
        // like the frozen `GpuVsmSampleParams`, so the resolve twin addresses
        // pages the same way the standalone sample pass does.
        let c = clipmap();
        let oracle = GpuVsmSampleParams::new(&c, 4096, 64, 1);
        let params = GpuVsmResolveParams::new(&c, 4096, 64, 1, true, Vec3::X, Vec3::Y, Vec3::NEG_Z);
        assert_eq!(params.levels, oracle.levels);
        assert_eq!(params.page_size, oracle.page_size);
        assert_eq!(params.pages_per_level_edge, oracle.pages_per_level_edge);
        assert_eq!(params.page_coord_bias, oracle.page_coord_bias);
        assert_eq!(
            params.level0_texel_world_size,
            oracle.level0_texel_world_size
        );
        assert_eq!(params.level0_max_distance, oracle.level0_max_distance);
        assert_eq!(params.physical_pages, oracle.physical_pages);
        assert_eq!(
            params.physical_pages_per_edge,
            oracle.physical_pages_per_edge
        );
        assert_eq!(params.pcf_radius, oracle.pcf_radius);
        // Slot space the page-table buffer must cover for this clipmap.
        assert_eq!(window_slot_count(&c), 4 * 8 * 8);
    }

    #[test]
    fn enable_switch_and_basis_are_packed_verbatim() {
        let c = clipmap();
        let on = GpuVsmResolveParams::new(
            &c,
            16,
            4,
            2,
            true,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, -1.0, 0.0),
        );
        assert_eq!(on.enable, 1);
        assert_eq!(on.light_right, [1.0, 0.0, 0.0]);
        assert_eq!(on.light_up, [0.0, 0.0, 1.0]);
        assert_eq!(on.light_forward, [0.0, -1.0, 0.0]);
        assert_eq!(on.pad0, 0);
        assert_eq!(on.pad1, 0);
        assert_eq!(on.pad2, 0.0);

        let off = GpuVsmResolveParams::new(&c, 16, 4, 2, false, Vec3::X, Vec3::Y, Vec3::NEG_Z);
        assert_eq!(off.enable, 0);
    }

    #[test]
    fn degenerate_geometry_is_clamped_like_the_oracle() {
        let c = clipmap();
        let params = GpuVsmResolveParams::new(&c, 0, 0, -3, true, Vec3::X, Vec3::Y, Vec3::NEG_Z);
        // Atlas edge clamped to >= 1 and PCF radius to >= 0 so the shader never
        // divides by zero or loops with a negative bound.
        assert_eq!(params.physical_pages_per_edge, 1);
        assert_eq!(params.pcf_radius, 0);
    }

    #[test]
    fn depth_scale_bias_match_the_caster_projection_ndc() {
        // The caster-depth projection (atlas/projection.rs) writes
        //   clip.z = (world . light_forward) / (2 D) + 0.5
        // into the atlas `.r`, where D = caster_depth_half_extent =
        //   page_world_size(coarsest) * pages_per_level_edge, coarsest = levels-1.
        // The resolve reference depth must use the *identical* mapping so a page
        // compare (`reference_depth <= stored => lit`) is meaningful, hence
        //   depth_scale = 1 / (2 D), depth_bias = 0.5 (the world-origin plane).
        // For this clipmap: page_world_size(3) = 0.1 * 2^3 * 128 = 102.4,
        // D = 102.4 * 8 = 819.2, so depth_scale = 1 / 1638.4, depth_bias = 0.5.
        let c = clipmap();
        let params = GpuVsmResolveParams::new(&c, 16, 4, 0, true, Vec3::X, Vec3::Y, Vec3::NEG_Z);
        let d = c.page_world_size(c.level_count() - 1) * f32::from(c.pages_per_level_edge);
        assert!((d - 819.2).abs() < 1.0e-3, "D = {}", d);
        assert!((params.depth_scale - 1.0 / (2.0 * d)).abs() < 1.0e-9);
        assert!((params.depth_bias - 0.5).abs() < 1.0e-9);
        // A caster `D` towards the light (world . forward = -D) lands on the near
        // plane 0; the world-origin plane on 0.5; a `D` away on the far plane 1 --
        // exactly the caster projection's `clip.z` endpoints.
        assert!(((-d) * params.depth_scale + params.depth_bias).abs() < 1.0e-6);
        assert!((0.0 * params.depth_scale + params.depth_bias - 0.5).abs() < 1.0e-6);
        assert!((d * params.depth_scale + params.depth_bias - 1.0).abs() < 1.0e-6);
    }
}
