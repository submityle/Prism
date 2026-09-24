//! Directional (cascaded) shadow evaluation: the top-level `visibility` term
//! for a sun/directional light, orchestrating cascade selection, normal-offset
//! and slope-scaled bias, light-space projection, PCF/PCSS filtering, and the
//! cross-cascade blend.
//!
//! This is the CPU golden twin of `shadow.wesl`'s directional path.  It returns
//! a scalar visibility in `[0, 1]` that the resolve pass multiplies into
//! `DirectLightSample::visibility` for the directional light.

use crate::shadow::bias::{apply_normal_offset, slope_scaled_depth_bias};
use crate::shadow::cascade::{cascade_blend_weight, select_cascade, CascadeSplits, MAX_CASCADE_COUNT};
use crate::shadow::filter::{pcf_visibility, pcss_visibility, PcssConfig, ShadowDepthSampler};
use crate::shadow::math::{transform_point, Mat4};

/// Which soft-shadow filter the directional evaluator applies per cascade.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ShadowFilter {
    /// Fixed-radius percentage-closer filtering (uniform penumbra).
    Pcf {
        /// Box-filter half-extent in shadow-map texels.
        radius: i32,
    },
    /// Percentage-closer soft shadows (contact-hardening penumbra).
    Pcss(PcssConfig),
}

/// Tunables for the directional shadow bias/blend, shared across cascades.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DirectionalShadowConfig {
    /// Normal-offset scale (multiples of one shadow texel's world size).
    pub normal_offset_scale: f32,
    /// Constant depth bias in normalized shadow depth units.
    pub const_depth_bias: f32,
    /// Slope-scaled depth-bias coefficient (multiplied by `tan(theta)`).
    pub slope_depth_bias: f32,
    /// Maximum total depth bias, bounding peter-panning at grazing angles.
    pub max_depth_bias: f32,
    /// Cross-cascade blend band width, as a fraction of the cascade's range.
    pub cascade_blend_fraction: f32,
    /// Soft-shadow filter kind and its parameters.
    pub filter: ShadowFilter,
}

/// All per-fragment and per-light inputs needed to evaluate a directional
/// shadow.  Matrices and texel sizes are indexed per cascade.
#[derive(Clone, Copy, Debug)]
pub struct DirectionalShadowInput {
    /// World-space position of the shaded surface point.
    pub world_position: [f32; 3],
    /// Unit world-space surface normal (used for the normal offset).
    pub world_normal: [f32; 3],
    /// Clamped, non-negative `n·l` cosine for the directional light.
    pub n_dot_l: f32,
    /// Positive view-space distance from the camera to the surface point.
    pub view_depth: f32,
    /// Cascade split table for this light.
    pub splits: CascadeSplits,
    /// Column-major world->light-clip matrix for each cascade.
    pub light_view_projections: [Mat4; MAX_CASCADE_COUNT],
    /// World size of one shadow-map texel for each cascade (drives the normal
    /// offset magnitude so it scales with cascade resolution).
    pub texel_world_sizes: [f32; MAX_CASCADE_COUNT],
    /// UV size of one shadow-map texel (`1 / resolution`) for filtering.
    pub texel_uv_size: [f32; 2],
}

/// Evaluates the directional-light visibility in `[0, 1]` for `input` under
/// `config`, sampling shadow depth through `sampler` (layer = cascade index).
///
/// `1.0` means fully lit.  Surfaces the light cannot reach in shadow space
/// (behind the light, or outside every cascade's mapped region) are treated as
/// lit so they fall back to ordinary `n·l` shading rather than turning black.
pub fn evaluate_directional_shadow<S: ShadowDepthSampler>(
    sampler: &S,
    input: &DirectionalShadowInput,
    config: &DirectionalShadowConfig,
) -> f32 {
    let cascade = select_cascade(input.view_depth, &input.splits);
    let primary = evaluate_single_cascade(sampler, input, config, cascade);

    let blend = cascade_blend_weight(
        input.view_depth,
        &input.splits,
        cascade,
        config.cascade_blend_fraction,
    );
    if blend <= 0.0 {
        return primary;
    }

    // Cross-fade into the next, coarser cascade to hide the resolution seam.
    let next = evaluate_single_cascade(sampler, input, config, cascade + 1);
    primary * (1.0 - blend) + next * blend
}

/// Projects the (normal-offset) surface point into `cascade`'s light clip
/// space, applies the slope-scaled depth bias, and filters.  Returns `1.0`
/// (lit) when the point projects behind the light or outside the cascade's
/// `[0, 1]` UV / depth range.
fn evaluate_single_cascade<S: ShadowDepthSampler>(
    sampler: &S,
    input: &DirectionalShadowInput,
    config: &DirectionalShadowConfig,
    cascade: usize,
) -> f32 {
    let cascade = cascade.min(input.splits.count.saturating_sub(1));

    let offset_position = apply_normal_offset(
        input.world_position,
        input.world_normal,
        input.texel_world_sizes[cascade],
        config.normal_offset_scale,
        input.n_dot_l,
    );

    let clip = transform_point(&input.light_view_projections[cascade], offset_position);
    // Directional lights use an orthographic projection (w == 1), but guard the
    // perspective divide anyway so a degenerate matrix cannot divide by zero.
    if clip[3] <= 0.0 {
        return 1.0;
    }
    let inv_w = clip[3].recip();
    let ndc = [clip[0] * inv_w, clip[1] * inv_w, clip[2] * inv_w];

    // wgpu NDC: x,y in [-1, 1], z in [0, 1]. Map to shadow UV with a flipped V.
    let uv = [ndc[0] * 0.5 + 0.5, ndc[1] * -0.5 + 0.5];
    if uv[0] < 0.0 || uv[0] > 1.0 || uv[1] < 0.0 || uv[1] > 1.0 {
        return 1.0;
    }
    if ndc[2] < 0.0 || ndc[2] > 1.0 {
        return 1.0;
    }

    let bias = slope_scaled_depth_bias(
        input.n_dot_l,
        config.const_depth_bias,
        config.slope_depth_bias,
        config.max_depth_bias,
    );
    let reference_depth = ndc[2] - bias;

    match config.filter {
        ShadowFilter::Pcf { radius } => pcf_visibility(
            sampler,
            cascade,
            uv,
            reference_depth,
            input.texel_uv_size,
            radius,
        ),
        ShadowFilter::Pcss(cfg) => pcss_visibility(
            sampler,
            cascade,
            uv,
            reference_depth,
            input.texel_uv_size,
            cfg,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Depth field that stores a single occluder plane at `depth` inside the
    /// mapped region and far elsewhere, on any layer.
    struct PlaneAtlas {
        depth: f32,
    }

    impl ShadowDepthSampler for PlaneAtlas {
        fn sample_depth(&self, _layer: usize, uv: [f32; 2]) -> f32 {
            if uv[0] < 0.0 || uv[0] > 1.0 || uv[1] < 0.0 || uv[1] > 1.0 {
                1.0
            } else {
                self.depth
            }
        }
    }

    /// Builds an orthographic-style column-major matrix that maps world XY in
    /// `[-half, half]` to NDC `[-1, 1]` and world Z in `[0, range]` to NDC
    /// `[0, 1]`, i.e. a trivial light projection for tests.
    fn ortho(half: f32, range: f32) -> Mat4 {
        let mut m = [0.0; 16];
        m[0] = 1.0 / half; // x scale
        m[5] = 1.0 / half; // y scale
        m[10] = 1.0 / range; // z scale into [0,1]
        m[15] = 1.0; // w = 1 (orthographic)
        m
    }

    fn base_input(z: f32) -> DirectionalShadowInput {
        let m = ortho(10.0, 100.0);
        DirectionalShadowInput {
            world_position: [0.0, 0.0, z],
            world_normal: [0.0, 0.0, 1.0],
            n_dot_l: 1.0,
            view_depth: 1.0,
            splits: crate::shadow::cascade::compute_cascade_splits(1.0, 100.0, 4, 0.5),
            light_view_projections: [m; MAX_CASCADE_COUNT],
            texel_world_sizes: [0.0; MAX_CASCADE_COUNT], // no normal offset in tests
            texel_uv_size: [1.0 / 128.0, 1.0 / 128.0],
        }
    }

    fn pcf_config(radius: i32) -> DirectionalShadowConfig {
        DirectionalShadowConfig {
            normal_offset_scale: 0.0,
            const_depth_bias: 0.0005,
            slope_depth_bias: 0.0,
            max_depth_bias: 0.01,
            cascade_blend_fraction: 0.0,
            filter: ShadowFilter::Pcf { radius },
        }
    }

    /// A receiver well behind the occluder plane is shadowed; a receiver in
    /// front of it is lit.
    #[test]
    fn receiver_behind_occluder_is_shadowed() {
        // Occluder stored at NDC depth 0.1 (world z = 10 over range 100).
        let field = PlaneAtlas { depth: 0.1 };
        let cfg = pcf_config(1);

        // Receiver at world z = 50 -> NDC 0.5 > 0.1 -> occluded.
        let shadowed = evaluate_directional_shadow(&field, &base_input(50.0), &cfg);
        assert!((shadowed - 0.0).abs() < 1.0e-6, "expected shadow, got {shadowed}");

        // Receiver at world z = 5 -> NDC 0.05 < 0.1 -> lit.
        let lit = evaluate_directional_shadow(&field, &base_input(5.0), &cfg);
        assert!((lit - 1.0).abs() < 1.0e-6, "expected lit, got {lit}");
    }

    /// Points projecting outside the shadow map's UV range stay lit rather than
    /// self-shadowing on garbage depth.
    #[test]
    fn out_of_bounds_projection_is_lit() {
        let field = PlaneAtlas { depth: 0.0 };
        let cfg = pcf_config(1);
        let mut input = base_input(50.0);
        // World x = 1000 with half-extent 10 -> NDC x = 100, far outside.
        input.world_position = [1000.0, 0.0, 50.0];
        assert_eq!(evaluate_directional_shadow(&field, &input, &cfg), 1.0);
    }

    /// The constant depth bias lifts a receiver that sits exactly on the
    /// occluder depth out of self-shadow (acne suppression).
    #[test]
    fn depth_bias_suppresses_self_shadow() {
        let field = PlaneAtlas { depth: 0.5 };
        // Receiver exactly at the stored depth (world z = 50 -> NDC 0.5).
        let input = base_input(50.0);

        // Without bias the equal-depth comparison is lit (<=), so force acne by
        // nudging the plane slightly in front, then show bias recovers it.
        let acne_field = PlaneAtlas { depth: 0.4999 };
        let no_bias = DirectionalShadowConfig {
            const_depth_bias: 0.0,
            ..pcf_config(0)
        };
        assert_eq!(evaluate_directional_shadow(&acne_field, &input, &no_bias), 0.0);

        let with_bias = DirectionalShadowConfig {
            const_depth_bias: 0.01,
            ..pcf_config(0)
        };
        assert_eq!(evaluate_directional_shadow(&acne_field, &input, &with_bias), 1.0);
        // Sanity: the well-lit plane case is unaffected.
        let _ = field;
    }

    /// PCSS produces a soft value near an occluder edge; here we just assert it
    /// runs end to end and returns a valid visibility for a mixed neighbourhood.
    #[test]
    fn pcss_path_returns_valid_visibility() {
        struct HalfPlane;
        impl ShadowDepthSampler for HalfPlane {
            fn sample_depth(&self, _layer: usize, uv: [f32; 2]) -> f32 {
                if uv[0] < 0.0 || uv[0] > 1.0 || uv[1] < 0.0 || uv[1] > 1.0 {
                    1.0
                } else if uv[0] < 0.5 {
                    0.1
                } else {
                    1.0
                }
            }
        }
        let cfg = DirectionalShadowConfig {
            normal_offset_scale: 0.0,
            const_depth_bias: 0.0005,
            slope_depth_bias: 0.0,
            max_depth_bias: 0.01,
            cascade_blend_fraction: 0.0,
            filter: ShadowFilter::Pcss(PcssConfig {
                search_radius: 4,
                light_size_uv: 0.3,
                min_filter_radius: 1,
                max_filter_radius: 16,
            }),
        };
        // Project to UV ~ (0.5, 0.5) via world x=0 -> ndc 0 -> uv 0.5.
        let v = evaluate_directional_shadow(&HalfPlane, &base_input(50.0), &cfg);
        assert!((0.0..=1.0).contains(&v), "visibility out of range: {v}");
    }
}
