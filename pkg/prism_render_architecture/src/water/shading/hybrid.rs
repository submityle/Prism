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

    let shore_factor = if blend.shore_blend_dist > EPS {
        (1.0 - inputs.dist_to_shore / blend.shore_blend_dist).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let shallow_factor = if blend.deep_blend_depth > EPS {
        (1.0 - inputs.depth / blend.deep_blend_depth).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let npr_weight = shore_factor.max(shallow_factor).clamp(0.0, 1.0);
    let pbr_weight = 1.0 - npr_weight;
    let overlay_opacity = blend.overlay_opacity.clamp(0.0, 1.0);

    HybridResponse {
        pbr,
        npr,
        npr_weight,
        pbr_weight,
        overlay_opacity,
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
}
