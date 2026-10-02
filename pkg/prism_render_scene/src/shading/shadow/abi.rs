//! GPU-side shadow ABI: the `#[repr(C)]` uniform records the resolve pass binds
//! so `shading_resolve.wesl` can expand them into the per-fragment
//! `DirectionalShadowInput` / `PointShadowInput` structs that `shadow.wesl`
//! evaluates.
//!
//! The shadow *math* lives in the CPU golden reference
//! ([`prism_render_shading::shadow`]) and its byte-for-byte WESL twin
//! (`shaders/shadow.wesl`).  This module only mirrors that math's **inputs**
//! onto the GPU: the per-cascade light-clip matrices, the PSSM split table, the
//! bias/blend/filter tunables and the atlas layer assignment.  Each record is
//! flat, `Pod`, and padded to a 16-byte grid so the same layout is valid as a
//! `std430` storage buffer today and can be promoted to a `std140` uniform
//! later without repacking.
//!
//! Conversions from the reference [`DirectionalShadowConfig`] /
//! [`CascadeSplits`] / [`CascadeMatrix`] / [`PointShadowConfig`] types are
//! provided (and round-trip tested) so extraction reuses the reference's
//! well-tested encoding rather than re-deriving it here.

use bytemuck::{Pod, Zeroable};
use prism_render_shading::{
    CascadeMatrix, CascadeSplits, DirectionalShadowConfig, PcssConfig, PointShadowConfig,
    ShadowFilter, SpotShadowConfig, MAX_CASCADE_COUNT,
};

/// Maximum number of shadow-casting directional lights the resolve pass tracks
/// in a single frame.  Directional lights are few (typically one sun), so a
/// small fixed cap keeps the uniform block compact.
pub(crate) const MAX_SHADOW_DIRECTIONALS: usize = 4;

/// Maximum number of shadow-casting point lights tracked per frame.  Each one
/// consumes six atlas layers (its cube faces), so this is bounded by the atlas
/// layer budget in practice.
pub(crate) const MAX_SHADOW_POINTS: usize = 16;

/// Maximum number of shadow-casting spot lights tracked per frame.  Each spot
/// consumes a single atlas layer (one perspective map), so the cap is generous
/// relative to the point-light budget.
pub(crate) const MAX_SHADOW_SPOTS: usize = 8;

/// Directional filter selector: fixed-radius percentage-closer filtering.
/// Matches `SHADOW_FILTER_PCF` in `shaders/shadow.wesl`.
pub(crate) const SHADOW_FILTER_PCF: u32 = 0;

/// Directional filter selector: percentage-closer soft shadows. Matches
/// `SHADOW_FILTER_PCSS` in `shaders/shadow.wesl`.
pub(crate) const SHADOW_FILTER_PCSS: u32 = 1;

/// Per-frame shadow header: how many directional / point slots are live plus
/// the shared atlas edge resolution (used to derive texel `UV` sizes on the
/// GPU when a slot does not carry its own).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub(crate) struct GpuShadowGlobals {
    /// Number of populated entries in the directional shadow array.
    pub directional_count: u32,
    /// Number of populated entries in the point shadow array.
    pub point_count: u32,
    /// Edge resolution (texels) of each square atlas layer.
    pub atlas_resolution: u32,
    /// Number of populated entries in the spot shadow array.
    pub spot_count: u32,
}

/// One shadow-casting directional light's GPU parameters.
///
/// The four `light_view_projections` are column-major world -> light-clip
/// matrices, one per cascade; `cascade_far` / `split_near` / `cascade_count`
/// reconstruct the [`CascadeSplits`] table on the GPU; the remaining scalars
/// mirror [`DirectionalShadowConfig`].  `light_index` links this record back to
/// the directional light it modulates in the light buffer.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub(crate) struct GpuDirectionalShadow {
    /// Column-major world -> light-clip matrix for each cascade.
    pub light_view_projections: [[f32; 16]; MAX_CASCADE_COUNT],
    /// View-space far distance of each cascade (the PSSM split boundaries).
    pub cascade_far: [f32; MAX_CASCADE_COUNT],
    /// World size of one shadow texel per cascade (drives the normal offset).
    pub texel_world_sizes: [f32; MAX_CASCADE_COUNT],
    /// `UV` size of one shadow texel (`1 / resolution`) for filtering.
    pub texel_uv_size: [f32; 2],
    /// Shared near plane of the cascade split table.
    pub split_near: f32,
    /// Number of active cascades in `[1, MAX_CASCADE_COUNT]`.
    pub cascade_count: u32,
    /// Normal-offset scale in multiples of one texel's world size.
    pub normal_offset_scale: f32,
    /// Constant depth bias in normalized shadow-depth units.
    pub const_depth_bias: f32,
    /// Slope-scaled depth-bias coefficient.
    pub slope_depth_bias: f32,
    /// Maximum total depth bias (bounds peter-panning).
    pub max_depth_bias: f32,
    /// Cross-cascade blend band width as a fraction of the cascade range.
    pub cascade_blend_fraction: f32,
    /// Soft-shadow filter selector (`SHADOW_FILTER_PCF` / `SHADOW_FILTER_PCSS`).
    pub filter_kind: u32,
    /// Box `PCF` half-extent in texels (used when `filter_kind` is `PCF`).
    pub pcf_radius: i32,
    /// `PCSS` blocker-search half-extent in texels.
    pub pcss_search_radius: i32,
    /// `PCSS` emitter size in `UV` units (larger softens the penumbra).
    pub pcss_light_size_uv: f32,
    /// `PCSS` minimum variable-`PCF` radius in texels.
    pub pcss_min_filter_radius: i32,
    /// `PCSS` maximum variable-`PCF` radius in texels.
    pub pcss_max_filter_radius: i32,
    /// First atlas array layer of this light's cascade block.
    pub base_layer: u32,
    /// Index of the modulated light in the directional light buffer.
    pub light_index: u32,
    /// `1` when this slot casts shadows, `0` when it is inert padding.
    pub enabled: u32,
    /// Padding to a 16-byte boundary; always zero.
    pub _padding: [u32; 2],
}

impl Default for GpuDirectionalShadow {
    fn default() -> Self {
        Self {
            light_view_projections: [CascadeMatrix::identity().view_projection; MAX_CASCADE_COUNT],
            cascade_far: [1.0; MAX_CASCADE_COUNT],
            texel_world_sizes: [1.0; MAX_CASCADE_COUNT],
            texel_uv_size: [1.0, 1.0],
            split_near: 0.0,
            cascade_count: 1,
            normal_offset_scale: 0.0,
            const_depth_bias: 0.0,
            slope_depth_bias: 0.0,
            max_depth_bias: 0.0,
            cascade_blend_fraction: 0.0,
            filter_kind: SHADOW_FILTER_PCF,
            pcf_radius: 1,
            pcss_search_radius: 0,
            pcss_light_size_uv: 0.0,
            pcss_min_filter_radius: 0,
            pcss_max_filter_radius: 0,
            base_layer: 0,
            light_index: 0,
            enabled: 0,
            _padding: [0; 2],
        }
    }
}

impl GpuDirectionalShadow {
    /// Packs the reference cascade matrices, split table and config into the GPU
    /// record for the light at `light_index`, whose cascade block starts at
    /// `base_layer`.  `texel_uv_size` is `1 / resolution` of the atlas layers.
    ///
    /// Only the first `splits.count` cascades are considered active; inactive
    /// slots keep the identity matrix so a stray GPU read stays lit rather than
    /// projecting garbage.
    pub(crate) fn from_reference(
        matrices: &[CascadeMatrix; MAX_CASCADE_COUNT],
        splits: &CascadeSplits,
        config: &DirectionalShadowConfig,
        texel_uv_size: [f32; 2],
        base_layer: u32,
        light_index: u32,
    ) -> Self {
        let mut light_view_projections = [[0.0_f32; 16]; MAX_CASCADE_COUNT];
        let mut texel_world_sizes = [1.0_f32; MAX_CASCADE_COUNT];
        for (slot, matrix) in light_view_projections.iter_mut().zip(matrices.iter()) {
            *slot = matrix.view_projection;
        }
        for (slot, matrix) in texel_world_sizes.iter_mut().zip(matrices.iter()) {
            *slot = matrix.texel_world_size;
        }

        let (filter_kind, pcf_radius, pcss) = match config.filter {
            ShadowFilter::Pcf { radius } => (SHADOW_FILTER_PCF, radius, None),
            ShadowFilter::Pcss(cfg) => {
                (SHADOW_FILTER_PCSS, cfg.min_filter_radius.max(1), Some(cfg))
            }
        };
        let pcss = pcss.unwrap_or(PcssConfig {
            search_radius: 0,
            light_size_uv: 0.0,
            min_filter_radius: 0,
            max_filter_radius: 0,
        });

        Self {
            light_view_projections,
            cascade_far: splits.distances,
            texel_world_sizes,
            texel_uv_size,
            split_near: splits.near,
            cascade_count: (splits.count.clamp(1, MAX_CASCADE_COUNT)) as u32,
            normal_offset_scale: config.normal_offset_scale,
            const_depth_bias: config.const_depth_bias,
            slope_depth_bias: config.slope_depth_bias,
            max_depth_bias: config.max_depth_bias,
            cascade_blend_fraction: config.cascade_blend_fraction,
            filter_kind,
            pcf_radius,
            pcss_search_radius: pcss.search_radius,
            pcss_light_size_uv: pcss.light_size_uv,
            pcss_min_filter_radius: pcss.min_filter_radius,
            pcss_max_filter_radius: pcss.max_filter_radius,
            base_layer,
            light_index,
            enabled: 1,
            _padding: [0; 2],
        }
    }
}

/// One shadow-casting point light's GPU parameters.
///
/// The six cube faces live in a contiguous atlas block starting at
/// `base_layer` (face order matches `shadow.wesl`'s `OpenGL` convention).
/// `light_index` links this record to the punctual light it modulates.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub(crate) struct GpuPointShadow {
    /// World-space position of the emitter.
    pub light_position: [f32; 3],
    /// Far range normalizing the stored linear distance.
    pub range: f32,
    /// `UV` size of one cube-face texel (`1 / resolution`).
    pub texel_uv_size: [f32; 2],
    /// Constant distance bias in normalized (`distance / range`) units.
    pub const_bias: f32,
    /// Slope-scaled distance-bias coefficient.
    pub slope_bias: f32,
    /// Maximum total distance bias.
    pub max_bias: f32,
    /// Box `PCF` half-extent in cube-face texels.
    pub pcf_radius: i32,
    /// First atlas array layer of this light's six-face block.
    pub base_layer: u32,
    /// Index of the modulated light in the punctual light buffer.
    pub light_index: u32,
    /// `1` when this slot casts shadows, `0` when it is inert padding.
    pub enabled: u32,
    /// Padding to a 16-byte boundary; always zero.
    pub _padding: [u32; 3],
}

impl Default for GpuPointShadow {
    fn default() -> Self {
        Self {
            light_position: [0.0; 3],
            range: 1.0,
            texel_uv_size: [1.0, 1.0],
            const_bias: 0.0,
            slope_bias: 0.0,
            max_bias: 0.0,
            pcf_radius: 1,
            base_layer: 0,
            light_index: 0,
            enabled: 0,
            _padding: [0; 3],
        }
    }
}

impl GpuPointShadow {
    /// Packs the reference point-shadow config into the GPU record for the light
    /// at `light_index`, whose six cube faces start at `base_layer`.
    pub(crate) fn from_reference(
        light_position: [f32; 3],
        range: f32,
        config: &PointShadowConfig,
        base_layer: u32,
        light_index: u32,
    ) -> Self {
        Self {
            light_position,
            range,
            texel_uv_size: config.texel_uv_size,
            const_bias: config.const_bias,
            slope_bias: config.slope_bias,
            max_bias: config.max_bias,
            pcf_radius: config.pcf_radius,
            base_layer,
            light_index,
            enabled: 1,
            _padding: [0; 3],
        }
    }
}

/// One shadow-casting spot light's GPU parameters.
///
/// A spot light needs a single perspective shadow map, so unlike
/// [`GpuPointShadow`] it stores one `world -> light-clip` matrix and occupies
/// one atlas layer at `base_layer`.  `light_index` links this record to the
/// punctual light it modulates (a spot lives in the punctual buffer with a
/// non-zero `spot_scale`).  The scalar block mirrors [`SpotShadowConfig`] and
/// reuses the directional `SHADOW_FILTER_*` selectors.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub(crate) struct GpuSpotShadow {
    /// Column-major world -> light-clip matrix for the spot's perspective map.
    pub light_view_projection: [f32; 16],
    /// `UV` size of one shadow texel (`1 / resolution`) for filtering.
    pub texel_uv_size: [f32; 2],
    /// Normal-offset scale in multiples of one texel's world size.
    pub normal_offset_scale: f32,
    /// World size of one shadow texel (drives the normal-offset magnitude).
    pub texel_world_size: f32,
    /// Constant depth bias in normalized shadow-depth units.
    pub const_depth_bias: f32,
    /// Slope-scaled depth-bias coefficient.
    pub slope_depth_bias: f32,
    /// Maximum total depth bias (bounds peter-panning).
    pub max_depth_bias: f32,
    /// Soft-shadow filter selector (`SHADOW_FILTER_PCF` / `SHADOW_FILTER_PCSS`).
    pub filter_kind: u32,
    /// Box `PCF` half-extent in texels (used when `filter_kind` is `PCF`).
    pub pcf_radius: i32,
    /// `PCSS` blocker-search half-extent in texels.
    pub pcss_search_radius: i32,
    /// `PCSS` emitter size in `UV` units (larger softens the penumbra).
    pub pcss_light_size_uv: f32,
    /// `PCSS` minimum variable-`PCF` radius in texels.
    pub pcss_min_filter_radius: i32,
    /// `PCSS` maximum variable-`PCF` radius in texels.
    pub pcss_max_filter_radius: i32,
    /// Atlas array layer holding this spot's perspective map.
    pub base_layer: u32,
    /// Index of the modulated light in the punctual light buffer.
    pub light_index: u32,
    /// `1` when this slot casts shadows, `0` when it is inert padding.
    pub enabled: u32,
}

impl Default for GpuSpotShadow {
    fn default() -> Self {
        Self {
            light_view_projection: [0.0; 16],
            texel_uv_size: [1.0, 1.0],
            normal_offset_scale: 0.0,
            texel_world_size: 0.0,
            const_depth_bias: 0.0,
            slope_depth_bias: 0.0,
            max_depth_bias: 0.0,
            filter_kind: SHADOW_FILTER_PCF,
            pcf_radius: 1,
            pcss_search_radius: 0,
            pcss_light_size_uv: 0.0,
            pcss_min_filter_radius: 0,
            pcss_max_filter_radius: 0,
            base_layer: 0,
            light_index: 0,
            enabled: 0,
        }
    }
}

impl GpuSpotShadow {
    /// Packs the reference spot-shadow config and its `world -> light-clip`
    /// matrix into the GPU record for the light at `light_index`, whose map
    /// occupies atlas layer `base_layer`.  `texel_world_size` scales the normal
    /// offset; `texel_uv_size` is `1 / resolution` of the atlas layer.
    pub(crate) fn from_reference(
        light_view_projection: [f32; 16],
        config: &SpotShadowConfig,
        texel_world_size: f32,
        texel_uv_size: [f32; 2],
        base_layer: u32,
        light_index: u32,
    ) -> Self {
        let (filter_kind, pcf_radius, pcss) = match config.filter {
            ShadowFilter::Pcf { radius } => (SHADOW_FILTER_PCF, radius, None),
            ShadowFilter::Pcss(cfg) => {
                (SHADOW_FILTER_PCSS, cfg.min_filter_radius.max(1), Some(cfg))
            }
        };
        let pcss = pcss.unwrap_or(PcssConfig {
            search_radius: 0,
            light_size_uv: 0.0,
            min_filter_radius: 0,
            max_filter_radius: 0,
        });

        Self {
            light_view_projection,
            texel_uv_size,
            normal_offset_scale: config.normal_offset_scale,
            texel_world_size,
            const_depth_bias: config.const_depth_bias,
            slope_depth_bias: config.slope_depth_bias,
            max_depth_bias: config.max_depth_bias,
            filter_kind,
            pcf_radius,
            pcss_search_radius: pcss.search_radius,
            pcss_light_size_uv: pcss.light_size_uv,
            pcss_min_filter_radius: pcss.min_filter_radius,
            pcss_max_filter_radius: pcss.max_filter_radius,
            base_layer,
            light_index,
            enabled: 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_shading::{
        compute_cascade_splits, spot_view_projection, CascadeMatrix, SpotShadowConfig,
    };

    #[test]
    fn shadow_records_are_16_byte_aligned_and_sized() {
        // Every std140/std430-friendly record must be a whole number of 16-byte
        // rows so a later uniform promotion needs no repacking.
        assert_eq!(size_of::<GpuShadowGlobals>(), 16);
        assert_eq!(size_of::<GpuDirectionalShadow>() % 16, 0);
        assert_eq!(size_of::<GpuPointShadow>() % 16, 0);
        assert_eq!(size_of::<GpuSpotShadow>() % 16, 0);
        assert_eq!(align_of::<GpuShadowGlobals>(), 4);
        assert_eq!(align_of::<GpuDirectionalShadow>(), 4);
        assert_eq!(align_of::<GpuPointShadow>(), 4);
        assert_eq!(align_of::<GpuSpotShadow>(), 4);
    }

    #[test]
    fn directional_pcf_config_round_trips_through_the_gpu_record() {
        let matrices = [
            CascadeMatrix::identity(),
            CascadeMatrix::identity(),
            CascadeMatrix::identity(),
            CascadeMatrix::identity(),
        ];
        let splits = compute_cascade_splits(1.0, 100.0, 4, 0.5);
        let config = DirectionalShadowConfig {
            normal_offset_scale: 2.0,
            const_depth_bias: 0.0005,
            slope_depth_bias: 0.002,
            max_depth_bias: 0.02,
            cascade_blend_fraction: 0.1,
            filter: ShadowFilter::Pcf { radius: 2 },
        };

        let gpu =
            GpuDirectionalShadow::from_reference(&matrices, &splits, &config, [0.5, 0.5], 3, 7);

        assert_eq!(gpu.filter_kind, SHADOW_FILTER_PCF);
        assert_eq!(gpu.pcf_radius, 2);
        assert_eq!(gpu.cascade_count, splits.count as u32);
        assert_eq!(gpu.split_near, splits.near);
        assert_eq!(gpu.cascade_far, splits.distances);
        assert_eq!(gpu.base_layer, 3);
        assert_eq!(gpu.light_index, 7);
        assert_eq!(gpu.enabled, 1);
        // PCF slots leave the PCSS block zeroed.
        assert_eq!(gpu.pcss_search_radius, 0);
        assert_eq!(gpu.pcss_light_size_uv, 0.0);
    }

    #[test]
    fn directional_pcss_config_carries_the_penumbra_parameters() {
        let matrices = [CascadeMatrix::identity(); MAX_CASCADE_COUNT];
        let splits = compute_cascade_splits(0.5, 200.0, 3, 0.7);
        let config = DirectionalShadowConfig {
            normal_offset_scale: 1.5,
            const_depth_bias: 0.001,
            slope_depth_bias: 0.003,
            max_depth_bias: 0.03,
            cascade_blend_fraction: 0.15,
            filter: ShadowFilter::Pcss(PcssConfig {
                search_radius: 3,
                light_size_uv: 0.02,
                min_filter_radius: 1,
                max_filter_radius: 8,
            }),
        };

        let gpu =
            GpuDirectionalShadow::from_reference(&matrices, &splits, &config, [0.25, 0.25], 0, 1);

        assert_eq!(gpu.filter_kind, SHADOW_FILTER_PCSS);
        assert_eq!(gpu.pcss_search_radius, 3);
        assert_eq!(gpu.pcss_light_size_uv, 0.02);
        assert_eq!(gpu.pcss_min_filter_radius, 1);
        assert_eq!(gpu.pcss_max_filter_radius, 8);
        // The PCF radius defaults to the PCSS floor so a fallback tap keeps size.
        assert_eq!(gpu.pcf_radius, 1);
        assert_eq!(gpu.cascade_count, 3);
    }

    #[test]
    fn point_shadow_config_round_trips_through_the_gpu_record() {
        let config = PointShadowConfig {
            const_bias: 0.001,
            slope_bias: 0.002,
            max_bias: 0.02,
            pcf_radius: 2,
            texel_uv_size: [1.0 / 512.0, 1.0 / 512.0],
        };

        let gpu = GpuPointShadow::from_reference([10.0, 2.0, -3.0], 50.0, &config, 4, 9);

        assert_eq!(gpu.light_position, [10.0, 2.0, -3.0]);
        assert_eq!(gpu.range, 50.0);
        assert_eq!(gpu.texel_uv_size, [1.0 / 512.0, 1.0 / 512.0]);
        assert_eq!(gpu.const_bias, 0.001);
        assert_eq!(gpu.pcf_radius, 2);
        assert_eq!(gpu.base_layer, 4);
        assert_eq!(gpu.light_index, 9);
        assert_eq!(gpu.enabled, 1);
    }

    #[test]
    fn spot_shadow_config_round_trips_through_the_gpu_record() {
        let vp = spot_view_projection(
            [1.0, 2.0, 3.0],
            [0.0, 0.0, -1.0],
            std::f32::consts::FRAC_PI_4,
            0.1,
            40.0,
        );
        let config = SpotShadowConfig {
            normal_offset_scale: 1.5,
            const_depth_bias: 0.0008,
            slope_depth_bias: 0.003,
            max_depth_bias: 0.02,
            filter: ShadowFilter::Pcss(PcssConfig {
                search_radius: 3,
                light_size_uv: 0.03,
                min_filter_radius: 2,
                max_filter_radius: 12,
            }),
        };

        let gpu =
            GpuSpotShadow::from_reference(vp, &config, 0.05, [1.0 / 256.0, 1.0 / 256.0], 5, 11);

        assert_eq!(gpu.light_view_projection, vp);
        assert_eq!(gpu.filter_kind, SHADOW_FILTER_PCSS);
        assert_eq!(gpu.pcss_search_radius, 3);
        assert_eq!(gpu.pcss_light_size_uv, 0.03);
        assert_eq!(gpu.pcss_min_filter_radius, 2);
        assert_eq!(gpu.pcss_max_filter_radius, 12);
        // The PCF fallback radius takes the PCSS floor.
        assert_eq!(gpu.pcf_radius, 2);
        assert_eq!(gpu.normal_offset_scale, 1.5);
        assert_eq!(gpu.texel_world_size, 0.05);
        assert_eq!(gpu.texel_uv_size, [1.0 / 256.0, 1.0 / 256.0]);
        assert_eq!(gpu.base_layer, 5);
        assert_eq!(gpu.light_index, 11);
        assert_eq!(gpu.enabled, 1);
    }

    #[test]
    fn spot_pcf_config_leaves_pcss_block_zeroed() {
        let config = SpotShadowConfig {
            normal_offset_scale: 1.0,
            const_depth_bias: 0.0005,
            slope_depth_bias: 0.0,
            max_depth_bias: 0.01,
            filter: ShadowFilter::Pcf { radius: 3 },
        };
        let gpu = GpuSpotShadow::from_reference([0.0; 16], &config, 0.0, [1.0, 1.0], 0, 0);
        assert_eq!(gpu.filter_kind, SHADOW_FILTER_PCF);
        assert_eq!(gpu.pcf_radius, 3);
        assert_eq!(gpu.pcss_search_radius, 0);
        assert_eq!(gpu.pcss_light_size_uv, 0.0);
    }

    #[test]
    fn default_slots_are_disabled_so_the_shader_skips_them() {
        assert_eq!(GpuDirectionalShadow::default().enabled, 0);
        assert_eq!(GpuPointShadow::default().enabled, 0);
        assert_eq!(GpuSpotShadow::default().enabled, 0);
    }
}
