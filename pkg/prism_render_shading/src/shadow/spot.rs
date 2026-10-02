//! Spot-light shadow evaluation: the `visibility` term for a cone-restricted
//! punctual light whose depth is stored in a single perspective shadow-atlas
//! layer.
//!
//! A spot light illuminates a cone, so unlike a point light it needs only one
//! shadow map rather than a cube: the depth pass rasterizes the scene through a
//! perspective frustum whose full vertical field of view matches the cone's
//! outer angle, and this evaluator projects the shaded point through that same
//! `world -> light-clip` matrix, does the (mandatory) perspective divide, and
//! filters the stored NDC depth. The cone's angular intensity falloff is a
//! lighting term handled by the resolve pass, not a shadow-visibility term, so
//! this module only answers "is the receiver occluded from the light".
//!
//! This is the CPU golden twin of `shadow.wesl`'s spot-light path. It returns a
//! scalar visibility in `[0, 1]` (`1.0` fully lit). Points that project behind
//! the light (`w <= 0`) or outside the mapped `[0, 1]` UV / depth range fall
//! back to lit so they shade with ordinary `n·l` rather than turning black.

use crate::shadow::bias::{apply_normal_offset, slope_scaled_depth_bias};
use crate::shadow::directional::ShadowFilter;
use crate::shadow::filter::{pcf_visibility, pcss_visibility, ShadowDepthSampler};
use crate::shadow::math::{look_at_rh, mul, perspective_rh_01, transform_point, Mat4};

/// Builds the column-major `world -> light-clip` matrix a spot light rasterizes
/// its shadow depth through, and that [`evaluate_spot_shadow`] projects the
/// shaded point with. Mirrors `glam::Mat4::perspective_rh` composed with
/// `look_at_rh` so an uploaded matrix and this reference are byte-identical.
///
/// `outer_angle` is the cone's half-angle (axis to rim, radians); the frustum's
/// full vertical field of view is `2 * outer_angle` so the cone rim lands on
/// the shadow map's edge. `near`/`far` bound the frustum along the cone axis.
/// The up hint switches to `+Z` when the axis is near-vertical to keep the
/// look-at basis well-conditioned.
pub fn spot_view_projection(
    position: [f32; 3],
    direction: [f32; 3],
    outer_angle: f32,
    near: f32,
    far: f32,
) -> Mat4 {
    let near = near.max(1.0e-4);
    let far = far.max(near + 1.0e-4);
    // Full vertical FOV = 2 * outer half-angle; clamp below `PI` so the
    // perspective `tan(fov/2)` stays finite even for a fully clamped cone.
    let fov_y = (outer_angle * 2.0).clamp(1.0e-3, core::f32::consts::PI - 1.0e-3);
    let proj = perspective_rh_01(fov_y, 1.0, near, far);

    // Choose an up hint not colinear with the cone axis so `look_at_rh`'s
    // `cross(forward, up)` does not collapse when the light points up / down.
    let up = if direction[0].abs() < 1.0e-3 && direction[2].abs() < 1.0e-3 {
        [0.0, 0.0, 1.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let center = [
        position[0] + direction[0],
        position[1] + direction[1],
        position[2] + direction[2],
    ];
    let view = look_at_rh(position, center, up);
    mul(&proj, &view)
}

/// Bias/filter tunables for a spot-light shadow. Shares [`ShadowFilter`] with
/// the directional path so PCF / PCSS parameters mean the same thing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpotShadowConfig {
    /// Normal-offset scale (multiples of one shadow texel's world size).
    pub normal_offset_scale: f32,
    /// Constant depth bias in normalized shadow depth units.
    pub const_depth_bias: f32,
    /// Slope-scaled depth-bias coefficient (multiplied by `tan(theta)`).
    pub slope_depth_bias: f32,
    /// Maximum total depth bias, bounding peter-panning at grazing angles.
    pub max_depth_bias: f32,
    /// Soft-shadow filter kind and its parameters.
    pub filter: ShadowFilter,
}

/// All per-fragment and per-light inputs needed to evaluate a spot shadow.
#[derive(Clone, Copy, Debug)]
pub struct SpotShadowInput {
    /// World-space position of the shaded surface point.
    pub world_position: [f32; 3],
    /// Unit world-space surface normal (used for the normal offset).
    pub world_normal: [f32; 3],
    /// Clamped, non-negative `n·l` cosine for the spot light.
    pub n_dot_l: f32,
    /// Column-major `world -> light-clip` matrix from [`spot_view_projection`].
    pub light_view_projection: Mat4,
    /// World size of one shadow-map texel (drives the normal-offset magnitude).
    pub texel_world_size: f32,
    /// UV size of one shadow-map texel (`1 / resolution`) for filtering.
    pub texel_uv_size: [f32; 2],
    /// Atlas layer this spot's depth lives on (the sampler's `layer` argument).
    pub layer: usize,
}

/// Evaluates the spot-light visibility in `[0, 1]` for `input` under `config`,
/// sampling shadow depth through `sampler` on `input.layer`.
///
/// `1.0` means fully lit. Surfaces the light cannot reach in shadow space
/// (behind the light, or outside the mapped `[0, 1]` UV / depth region) are
/// treated as lit so they fall back to ordinary `n·l` shading.
pub fn evaluate_spot_shadow<S: ShadowDepthSampler>(
    sampler: &S,
    input: &SpotShadowInput,
    config: &SpotShadowConfig,
) -> f32 {
    let offset_position = apply_normal_offset(
        input.world_position,
        input.world_normal,
        input.texel_world_size,
        config.normal_offset_scale,
        input.n_dot_l,
    );

    let clip = transform_point(&input.light_view_projection, offset_position);
    // Spot lights are perspective (`w == -z_eye`), so the divide is mandatory
    // and `w <= 0` means the point is at or behind the light plane: treat lit.
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
            input.layer,
            uv,
            reference_depth,
            input.texel_uv_size,
            radius,
        ),
        ShadowFilter::Pcss(cfg) => pcss_visibility(
            sampler,
            input.layer,
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
    use crate::shadow::filter::PcssConfig;

    /// Depth field storing a single occluder plane at `depth` inside the mapped
    /// region and far (`1.0`) elsewhere / off-map, on any layer.
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

    fn pcf_config(radius: i32) -> SpotShadowConfig {
        SpotShadowConfig {
            normal_offset_scale: 0.0,
            const_depth_bias: 0.0005,
            slope_depth_bias: 0.0,
            max_depth_bias: 0.01,
            filter: ShadowFilter::Pcf { radius },
        }
    }

    /// A light at the origin aiming down `-Z`; a point on the axis a distance
    /// `dist` in front projects to UV (0.5, 0.5) on the shadow map.
    fn axis_input(dist: f32) -> SpotShadowInput {
        let vp = spot_view_projection(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            core::f32::consts::FRAC_PI_4, // 45-degree half-angle cone
            0.1,
            100.0,
        );
        SpotShadowInput {
            world_position: [0.0, 0.0, -dist],
            world_normal: [0.0, 0.0, 1.0],
            n_dot_l: 1.0,
            light_view_projection: vp,
            texel_world_size: 0.0, // disable normal offset in tests
            texel_uv_size: [1.0 / 128.0, 1.0 / 128.0],
            layer: 0,
        }
    }

    /// The view-projection must send an on-axis point in front of the light to
    /// the shadow map centre (0.5, 0.5) with a valid `[0, 1]` NDC depth.
    #[test]
    fn on_axis_point_projects_to_map_centre() {
        let vp = spot_view_projection(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            core::f32::consts::FRAC_PI_4,
            0.1,
            100.0,
        );
        let clip = transform_point(&vp, [0.0, 0.0, -10.0]);
        assert!(
            clip[3] > 0.0,
            "point in front must have w > 0, got {}",
            clip[3]
        );
        let inv_w = clip[3].recip();
        let uv = [clip[0] * inv_w * 0.5 + 0.5, clip[1] * inv_w * -0.5 + 0.5];
        assert!((uv[0] - 0.5).abs() < 1.0e-6, "u = {}", uv[0]);
        assert!((uv[1] - 0.5).abs() < 1.0e-6, "v = {}", uv[1]);
        let ndc_z = clip[2] * inv_w;
        assert!((0.0..=1.0).contains(&ndc_z), "ndc z out of range: {ndc_z}");
    }

    /// A receiver farther than the stored occluder is shadowed; one nearer than
    /// it is lit.
    #[test]
    fn receiver_behind_occluder_is_shadowed() {
        let cfg = pcf_config(1);

        // Perspective NDC depth is strongly non-linear, so derive the two
        // receiver depths and store the occluder plane exactly between them.
        let ndc_z = |dist: f32| {
            let clip = transform_point(&axis_input(dist).light_view_projection, [0.0, 0.0, -dist]);
            clip[2] / clip[3]
        };
        let near_ndc = ndc_z(0.2);
        let far_ndc = ndc_z(1.0);
        assert!(
            near_ndc < far_ndc,
            "near {near_ndc} should be < far {far_ndc}"
        );
        let plane = 0.5 * (near_ndc + far_ndc);

        let field = PlaneAtlas { depth: plane };
        // Farther receiver is behind the occluder plane -> shadowed.
        let shadowed = evaluate_spot_shadow(&field, &axis_input(1.0), &cfg);
        assert!(
            (shadowed - 0.0).abs() < 1.0e-6,
            "expected shadow, got {shadowed}"
        );

        // Nearer receiver is in front of the occluder plane -> lit.
        let lit = evaluate_spot_shadow(&field, &axis_input(0.2), &cfg);
        assert!((lit - 1.0).abs() < 1.0e-6, "expected lit, got {lit}");
    }

    /// A point outside the cone (behind the light) projects with `w <= 0` and
    /// stays lit rather than sampling garbage depth.
    #[test]
    fn point_behind_light_is_lit() {
        let field = PlaneAtlas { depth: 0.0 };
        let cfg = pcf_config(1);
        let mut input = axis_input(10.0);
        // Move the receiver behind the light (positive Z, light aims -Z).
        input.world_position = [0.0, 0.0, 10.0];
        assert_eq!(evaluate_spot_shadow(&field, &input, &cfg), 1.0);
    }

    /// A point projecting outside the cone's UV footprint stays lit.
    #[test]
    fn out_of_cone_projection_is_lit() {
        let field = PlaneAtlas { depth: 0.0 };
        let cfg = pcf_config(1);
        let mut input = axis_input(10.0);
        // Far off the axis at a shallow depth: outside the 45-degree cone.
        input.world_position = [1000.0, 0.0, -10.0];
        assert_eq!(evaluate_spot_shadow(&field, &input, &cfg), 1.0);
    }

    /// The constant depth bias lifts a receiver sitting essentially on the
    /// occluder depth out of self-shadow (acne suppression).
    #[test]
    fn depth_bias_suppresses_self_shadow() {
        // Stored plane a hair in front of the receiver's own depth so the
        // unbiased comparison shadows it.
        let receiver = axis_input(50.0);
        let clip = transform_point(&receiver.light_view_projection, [0.0, 0.0, -50.0]);
        let ndc_z = clip[2] / clip[3];
        let acne_field = PlaneAtlas {
            depth: ndc_z - 1.0e-4,
        };

        let no_bias = SpotShadowConfig {
            const_depth_bias: 0.0,
            ..pcf_config(0)
        };
        assert_eq!(evaluate_spot_shadow(&acne_field, &receiver, &no_bias), 0.0);

        let with_bias = SpotShadowConfig {
            const_depth_bias: 0.01,
            ..pcf_config(0)
        };
        assert_eq!(
            evaluate_spot_shadow(&acne_field, &receiver, &with_bias),
            1.0
        );
    }

    /// PCSS runs end to end over a mixed neighbourhood and returns a valid
    /// soft visibility.
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
        let cfg = SpotShadowConfig {
            normal_offset_scale: 0.0,
            const_depth_bias: 0.0005,
            slope_depth_bias: 0.0,
            max_depth_bias: 0.01,
            filter: ShadowFilter::Pcss(PcssConfig {
                search_radius: 4,
                light_size_uv: 0.3,
                min_filter_radius: 1,
                max_filter_radius: 16,
            }),
        };
        let v = evaluate_spot_shadow(&HalfPlane, &axis_input(50.0), &cfg);
        assert!((0.0..=1.0).contains(&v), "visibility out of range: {v}");
    }

    /// A vertical cone axis must not produce a degenerate matrix (NaNs); the
    /// up-hint switch keeps the basis well-conditioned.
    #[test]
    fn vertical_axis_matrix_is_finite() {
        let vp = spot_view_projection(
            [0.0, 5.0, 0.0],
            [0.0, -1.0, 0.0],
            core::f32::consts::FRAC_PI_4,
            0.1,
            100.0,
        );
        assert!(
            vp.iter().all(|c| c.is_finite()),
            "matrix had non-finite: {vp:?}"
        );
        // A point below the light on the axis projects in front (w > 0).
        let clip = transform_point(&vp, [0.0, 0.0, 0.0]);
        assert!(clip[3] > 0.0, "w = {}", clip[3]);
    }
}
