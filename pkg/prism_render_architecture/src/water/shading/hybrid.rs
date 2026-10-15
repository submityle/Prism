//! Hybrid water lighting response (design §5.4).
//!
//! One water body can present a physical body of water offshore and a stylized
//! treatment at the shoreline and in the shallows. This planner evaluates both
//! the [`super::pbr`] and [`super::npr`] sub-responses and a blend weight driven
//! by shore proximity and water depth, then carries an orthogonal stylized
//! overlay opacity for hand-painted layers over the physical base
//! (`Arcane`-style). The blend weights always sum to one. Pure and
//! deterministic.

use super::super::EPS;
use super::npr::plan_npr;
use super::pbr::plan_pbr;
use super::{
    HybridBlendParams, HybridResponse, NprShadingParams, PbrShadingParams, SurfaceShadingInputs,
};

/// Normalized inputs shared by the CPU planner and the raster shader.
///
/// Both response arms consume the same rule: the stylized region is the maximum
/// of the shore and shallow-water signals. Keeping the signals normalized
/// before this function makes the contract directly portable to WESL without
/// importing any host-side distance or depth units into the shader.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HybridWeightInputs {
    /// Normalized shore-proximity signal (`1` at the shore, `0` offshore).
    pub shore_signal: f32,
    /// Normalized shallow-water signal (`1` at zero depth, `0` in deep water).
    pub shallow_signal: f32,
    /// Opacity of the stylized overlay over the blended base.
    pub overlay_opacity: f32,
}

/// The shared CPU/GPU hybrid weight contract.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HybridWeights {
    /// Weight toward the stylized response.
    pub npr_weight: f32,
    /// Weight toward the physically based response.
    pub pbr_weight: f32,
    /// Clamped opacity of the orthogonal stylized overlay.
    pub overlay_opacity: f32,
}

/// One resolved response arm consumed by the resident `MaskBlend`
/// composition pass.
///
/// The raster response arms already contain their lighting decisions.  The
/// compositor must not re-run a frontend, so the contract carries the final
/// premultiplied colour together with the two scalar diagnostics that are
/// commonly consumed by temporal and water-specific post-processing.  Alpha
/// is the fourth colour component and therefore remains part of the same
/// interpolation operation as the legacy `vec4` path.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HybridResponseArm {
    /// Premultiplied RGBA response produced by one frontend.
    pub color: [f32; 4],
    /// Reflection contribution represented by this response arm.
    pub reflection: f32,
    /// Focused-light caustic contribution represented by this response arm.
    pub caustics: f32,
}

/// Unified CPU input/output contract for the legacy Hybrid and `MaskBlend`
/// equivalence gate.
///
/// This is intentionally a resolved contract: geometry, reflection lookup,
/// caustic projection, and response shading happen before composition on both
/// paths.  Keeping those services out of this type prevents the golden from
/// silently inventing a second water renderer while still making every value
/// that the compositor can change explicit and testable.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HybridCompositionInput {
    /// Physically based response arm.
    pub pbr: HybridResponseArm,
    /// Stylized response arm.
    pub npr: HybridResponseArm,
    /// Normalized shore/depth signals and authored overlay opacity.
    pub weights: HybridWeightInputs,
    /// Surface foam coverage used by the reactive history gate.
    pub foam: f32,
    /// Current-minus-previous screen-space motion.
    pub motion: [f32; 2],
}

/// Resolved output of the resident `MaskBlend` composition contract.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HybridCompositionOutput {
    /// Final premultiplied RGBA colour.
    pub color: [f32; 4],
    /// Final reflection diagnostic after the same two-stage blend as colour.
    pub reflection: f32,
    /// Final caustic diagnostic after the same two-stage blend as colour.
    pub caustics: f32,
    /// Explicit alpha/transparency output, equal to `color[3]`.
    pub transparency: f32,
    /// Motion vector forwarded without modification by composition.
    pub motion: [f32; 2],
    /// Resolved weight toward the stylized arm.
    pub npr_weight: f32,
    /// Resolved weight toward the physical arm.
    pub pbr_weight: f32,
    /// Reactive history-rejection coverage.
    pub reactive_mask: f32,
    /// Effective overlay factor after the reactive floor is applied.
    pub overlay_factor: f32,
}

/// Resolve normalized hybrid weights.
///
/// This is the behavior contract mirrored by `water_hybrid_weight_contract`
/// in `water_surface_raster.wesl`: clamp both region signals, take their
/// maximum for the `NPR` weight, and use its complement for `PBR`.
#[must_use]
pub fn resolve_hybrid_weights(inputs: HybridWeightInputs) -> HybridWeights {
    let shore_signal = inputs.shore_signal.clamp(0.0, 1.0);
    let shallow_signal = inputs.shallow_signal.clamp(0.0, 1.0);
    let npr_weight = shore_signal.max(shallow_signal);

    HybridWeights {
        npr_weight,
        pbr_weight: 1.0 - npr_weight,
        overlay_opacity: inputs.overlay_opacity.clamp(0.0, 1.0),
    }
}

/// Convert physical shore/depth measurements into the normalized contract.
#[must_use]
pub fn resolve_hybrid_geometry_weights(
    blend: HybridBlendParams,
    inputs: SurfaceShadingInputs,
) -> HybridWeights {
    let shore_signal = if blend.shore_blend_dist > EPS {
        (1.0 - inputs.dist_to_shore / blend.shore_blend_dist).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let shallow_signal = if blend.deep_blend_depth > EPS {
        (1.0 - inputs.depth / blend.deep_blend_depth).clamp(0.0, 1.0)
    } else {
        0.0
    };

    resolve_hybrid_weights(HybridWeightInputs {
        shore_signal,
        shallow_signal,
        overlay_opacity: blend.overlay_opacity,
    })
}

/// Blend one scalar response using the shared region weights and then apply
/// the orthogonal stylized overlay.
#[must_use]
pub fn compose_hybrid_scalar(pbr: f32, npr: f32, weights: HybridWeights) -> f32 {
    let blended = pbr * weights.pbr_weight + npr * weights.npr_weight;
    blended * (1.0 - weights.overlay_opacity) + npr * weights.overlay_opacity
}

fn mix_rgba(left: [f32; 4], right: [f32; 4], factor: f32) -> [f32; 4] {
    [
        left[0] * (1.0 - factor) + right[0] * factor,
        left[1] * (1.0 - factor) + right[1] * factor,
        left[2] * (1.0 - factor) + right[2] * factor,
        left[3] * (1.0 - factor) + right[3] * factor,
    ]
}

fn mix_scalar(left: f32, right: f32, factor: f32) -> f32 {
    left * (1.0 - factor) + right * factor
}

/// Evaluate the resident `MaskBlend` composition contract.
///
/// This is the CPU twin of the response-plane compositor.  It deliberately
/// performs the two stages visible in the legacy shader: first blend PBR/NPR
/// by the normalized region mask, then apply the stylized overlay using the
/// reactive floor.  Keeping this function separate from the legacy test
/// reference makes the S4b gate a real A/B comparison instead of a duplicate
/// call to one implementation.
#[must_use]
pub fn compose_mask_blend(input: HybridCompositionInput) -> HybridCompositionOutput {
    let weights = resolve_hybrid_weights(input.weights);
    let blended_color = mix_rgba(input.pbr.color, input.npr.color, weights.npr_weight);
    let blended_reflection = mix_scalar(
        input.pbr.reflection,
        input.npr.reflection,
        weights.npr_weight,
    );
    let blended_caustics = mix_scalar(input.pbr.caustics, input.npr.caustics, weights.npr_weight);
    let reactive_mask = hybrid_reactive_mask(input.foam, input.motion, blended_color[3]);
    let overlay_factor = (weights.overlay_opacity * reactive_mask.max(0.25)).clamp(0.0, 1.0);
    let color = mix_rgba(blended_color, input.npr.color, overlay_factor);

    HybridCompositionOutput {
        color,
        reflection: mix_scalar(blended_reflection, input.npr.reflection, overlay_factor),
        caustics: mix_scalar(blended_caustics, input.npr.caustics, overlay_factor),
        transparency: color[3],
        motion: input.motion,
        npr_weight: weights.npr_weight,
        pbr_weight: weights.pbr_weight,
        reactive_mask,
        overlay_factor,
    }
}

/// Current-to-previous motion convention used by the water raster output.
#[must_use]
pub fn hybrid_motion_vector(current_uv: [f32; 2], previous_uv: [f32; 2]) -> [f32; 2] {
    [
        current_uv[0] - previous_uv[0],
        current_uv[1] - previous_uv[1],
    ]
}

/// Reactive coverage used to gate the hybrid overlay without adding a new
/// render target to the existing water surface ABI.
#[must_use]
pub fn hybrid_reactive_mask(foam: f32, motion: [f32; 2], transparency: f32) -> f32 {
    let motion_strength = (motion[0] * motion[0] + motion[1] * motion[1]).sqrt();
    foam.clamp(0.0, 1.0)
        .max((motion_strength * 32.0).clamp(0.0, 1.0))
        .max((1.0 - transparency).clamp(0.0, 1.0))
}

/// Resolve the [`HybridResponse`] for one surface sample.
///
/// The stylized weight rises toward `1` near the shore and in the shallows and
/// falls to `0` in deep, offshore water; the physical weight is its complement.
#[must_use]
pub fn plan_hybrid(
    blend: HybridBlendParams,
    pbr_params: PbrShadingParams,
    npr_params: NprShadingParams,
    inputs: SurfaceShadingInputs,
) -> HybridResponse {
    let pbr = plan_pbr(pbr_params, inputs);
    let npr = plan_npr(npr_params, inputs);
    let weights = resolve_hybrid_geometry_weights(blend, inputs);

    HybridResponse {
        pbr,
        npr,
        npr_weight: weights.npr_weight,
        pbr_weight: weights.pbr_weight,
        overlay_opacity: weights.overlay_opacity,
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::underwater::RgbColor;
    use super::*;

    fn pbr_params() -> PbrShadingParams {
        PbrShadingParams {
            f0_override: 0.0,
            foam_fold_threshold: 1.0,
            base_roughness: 0.08,
            grazing_transmission: 0.6,
            ssr_min_confidence: 0.5,
            rt_min_budget: 0.25,
        }
    }

    fn npr_params() -> NprShadingParams {
        NprShadingParams {
            color_bands: 4,
            specular_threshold: 0.7,
            foam_edge_width: 1.5,
            halftone_scale: 8.0,
            flow_line_gain: 0.5,
        }
    }

    fn blend_params() -> HybridBlendParams {
        HybridBlendParams {
            shore_blend_dist: 3.0,
            deep_blend_depth: 2.0,
            overlay_opacity: 0.35,
        }
    }

    fn inputs_at(dist_to_shore: f32, depth: f32) -> SurfaceShadingInputs {
        SurfaceShadingInputs {
            cos_view: 0.5,
            ior: 1.33,
            jacobian: 0.4,
            water_color: RgbColor {
                r: 0.1,
                g: 0.3,
                b: 0.5,
            },
            specular_intensity: 0.8,
            caustic_intensity: 0.6,
            flow_speed: 1.2,
            depth,
            dist_to_shore,
            ssr_confidence: 0.7,
            ray_budget: 0.5,
        }
    }

    #[test]
    fn weights_sum_to_one() {
        let cases = [(0.0, 0.0), (1.0, 1.0), (5.0, 5.0), (0.5, 3.0)];
        for (dist, depth) in cases {
            let r = plan_hybrid(
                blend_params(),
                pbr_params(),
                npr_params(),
                inputs_at(dist, depth),
            );
            assert!(
                (r.pbr_weight + r.npr_weight - 1.0).abs() < EPS,
                "weights sum {} at ({dist},{depth})",
                r.pbr_weight + r.npr_weight
            );
            assert!((0.0..=1.0).contains(&r.npr_weight));
            assert!((0.0..=1.0).contains(&r.pbr_weight));
        }
    }

    #[test]
    fn stylized_at_shore_physical_offshore() {
        let shore = plan_hybrid(
            blend_params(),
            pbr_params(),
            npr_params(),
            inputs_at(0.0, 0.0),
        );
        assert!(
            (shore.npr_weight - 1.0).abs() < EPS,
            "shore should be fully NPR"
        );

        let offshore = plan_hybrid(
            blend_params(),
            pbr_params(),
            npr_params(),
            inputs_at(10.0, 10.0),
        );
        assert!(
            offshore.npr_weight.abs() < EPS,
            "deep offshore should be fully PBR"
        );
        assert!((offshore.pbr_weight - 1.0).abs() < EPS);
    }

    #[test]
    fn npr_weight_decreases_with_shore_distance() {
        let near = plan_hybrid(
            blend_params(),
            pbr_params(),
            npr_params(),
            inputs_at(0.5, 10.0),
        );
        let far = plan_hybrid(
            blend_params(),
            pbr_params(),
            npr_params(),
            inputs_at(2.5, 10.0),
        );
        assert!(near.npr_weight > far.npr_weight);
    }

    #[test]
    fn sub_responses_match_standalone_planners() {
        let inputs = inputs_at(1.0, 1.0);
        let r = plan_hybrid(blend_params(), pbr_params(), npr_params(), inputs);
        assert_eq!(r.pbr, plan_pbr(pbr_params(), inputs));
        assert_eq!(r.npr, plan_npr(npr_params(), inputs));
    }

    #[test]
    fn plan_hybrid_is_deterministic() {
        let inputs = inputs_at(1.0, 1.0);
        assert_eq!(
            plan_hybrid(blend_params(), pbr_params(), npr_params(), inputs),
            plan_hybrid(blend_params(), pbr_params(), npr_params(), inputs)
        );
    }

    #[test]
    fn normalized_weight_contract_matches_geometry_planner() {
        let inputs = inputs_at(1.0, 1.0);
        let direct = resolve_hybrid_weights(HybridWeightInputs {
            shore_signal: 2.0 / 3.0,
            shallow_signal: 0.5,
            overlay_opacity: 0.35,
        });
        let geometry = resolve_hybrid_geometry_weights(blend_params(), inputs);
        assert!((direct.npr_weight - geometry.npr_weight).abs() <= EPS);
        assert!((direct.pbr_weight - geometry.pbr_weight).abs() <= EPS);
        assert!((direct.overlay_opacity - geometry.overlay_opacity).abs() <= EPS);
    }

    #[test]
    fn overlay_is_consumed_after_region_blend() {
        let weights = HybridWeights {
            npr_weight: 0.25,
            pbr_weight: 0.75,
            overlay_opacity: 0.5,
        };
        assert!((compose_hybrid_scalar(0.0, 1.0, weights) - 0.625).abs() < EPS);
    }

    #[test]
    fn motion_and_reactive_contracts_are_bounded() {
        assert_eq!(
            hybrid_motion_vector([0.75, 0.25], [0.5, 0.5]),
            [0.25, -0.25]
        );
        assert!((0.0..=1.0).contains(&hybrid_reactive_mask(0.3, [0.01, 0.0], 0.8)));
        assert_eq!(hybrid_reactive_mask(0.0, [0.0, 0.0], 1.0), 0.0);
    }
}
