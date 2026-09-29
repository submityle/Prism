//! Stylized (`NPR`) water lighting response (design §5.2).
//!
//! The stylized frontend consumes the exact same shared advanced base as `PBR`
//! (see [`super::super::SharedBaseServices`]); it only reinterprets the resolved
//! lighting. It ramp-quantizes the water color (`Zelda`/toon), turns the
//! specular highlight into a hard toon block, draws a hand-painted shoreline
//! foam edge (`Zelda` shallows), stylizes caustics as a `halftone` dot pattern,
//! and traces flow-aligned stylization lines. Every function is pure.

use super::super::underwater::RgbColor;
use super::super::EPS;
use super::{NprResponse, NprShadingParams, SurfaceShadingInputs};

/// Quantize `t` into `bands` discrete steps.
///
/// Clamps `t` into `0..=1`, then snaps it to the lower edge of one of `bands`
/// equal intervals, so the result is a non-decreasing step function bounded in
/// `[0, 1]`. `bands` is floored to `1` to stay well defined. This is the toon
/// color-ramp primitive.
#[must_use]
pub fn quantize_ramp(t: f32, bands: u32) -> f32 {
    let b = bands.max(1) as f32;
    let idx = (t.clamp(0.0, 1.0) * b) as u32;
    (idx as f32 / b).min(1.0)
}

/// Ramp-quantize each channel of an `RGB` color.
fn quantize_color(color: RgbColor, bands: u32) -> RgbColor {
    RgbColor {
        r: quantize_ramp(color.r, bands),
        g: quantize_ramp(color.g, bands),
        b: quantize_ramp(color.b, bands),
    }
}

/// Resolve the [`NprResponse`] for one surface sample.
#[must_use]
pub fn plan_npr(params: NprShadingParams, inputs: SurfaceShadingInputs) -> NprResponse {
    let ramp_color = quantize_color(inputs.water_color, params.color_bands);

    let toon_specular = if inputs.specular_intensity >= params.specular_threshold {
        1.0
    } else {
        0.0
    };

    let foam_edge = if params.foam_edge_width > EPS {
        (1.0 - inputs.dist_to_shore / params.foam_edge_width).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let halftone_coverage = inputs.caustic_intensity.clamp(0.0, 1.0);
    let flow_line = (inputs.flow_speed * params.flow_line_gain).clamp(0.0, 1.0);

    NprResponse {
        ramp_color,
        toon_specular,
        foam_edge,
        halftone_coverage,
        halftone_scale: params.halftone_scale,
        flow_line,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_params() -> NprShadingParams {
        NprShadingParams {
            color_bands: 4,
            specular_threshold: 0.7,
            foam_edge_width: 1.5,
            halftone_scale: 8.0,
            flow_line_gain: 0.5,
        }
    }

    fn fixture_inputs() -> SurfaceShadingInputs {
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
            depth: 1.0,
            dist_to_shore: 1.0,
            ssr_confidence: 0.7,
            ray_budget: 0.5,
        }
    }

    #[test]
    fn quantize_is_bounded_and_monotonic() {
        let mut prev = quantize_ramp(0.0, 4);
        let mut t = 0.0;
        while t <= 1.0 + EPS {
            let cur = quantize_ramp(t, 4);
            assert!((0.0..=1.0).contains(&cur), "quantized {cur} out of range");
            assert!(cur >= prev - EPS, "quantize decreased at {t}");
            prev = cur;
            t += 0.02;
        }
    }

    #[test]
    fn quantize_snaps_within_a_band() {
        // Two inputs in the same quarter-band snap to the same step.
        assert!((quantize_ramp(0.05, 4) - quantize_ramp(0.2, 4)).abs() < EPS);
        // A band boundary produces a distinct step.
        assert!(quantize_ramp(0.3, 4) > quantize_ramp(0.2, 4));
    }

    #[test]
    fn quantize_handles_zero_bands() {
        // Flooring bands to one keeps the call well defined.
        assert!((0.0..=1.0).contains(&quantize_ramp(0.5, 0)));
    }

    #[test]
    fn ramp_color_is_quantized_per_channel() {
        let params = fixture_params();
        let inputs = fixture_inputs();
        let out = plan_npr(params, inputs).ramp_color;
        assert!((out.r - quantize_ramp(inputs.water_color.r, params.color_bands)).abs() < EPS);
        assert!((out.g - quantize_ramp(inputs.water_color.g, params.color_bands)).abs() < EPS);
        assert!((out.b - quantize_ramp(inputs.water_color.b, params.color_bands)).abs() < EPS);
    }

    #[test]
    fn toon_specular_is_a_hard_block() {
        let params = fixture_params();
        let mut inputs = fixture_inputs();
        inputs.specular_intensity = 0.9;
        assert!((plan_npr(params, inputs).toon_specular - 1.0).abs() < EPS);
        inputs.specular_intensity = 0.5;
        assert!(plan_npr(params, inputs).toon_specular.abs() < EPS);
    }

    #[test]
    fn foam_edge_fades_away_from_shore() {
        let params = fixture_params();
        let mut inputs = fixture_inputs();
        inputs.dist_to_shore = 0.0;
        let at_shore = plan_npr(params, inputs).foam_edge;
        inputs.dist_to_shore = 0.75;
        let mid = plan_npr(params, inputs).foam_edge;
        inputs.dist_to_shore = 5.0;
        let offshore = plan_npr(params, inputs).foam_edge;
        assert!((at_shore - 1.0).abs() < EPS, "shore should be full foam");
        assert!(mid < at_shore && mid > offshore);
        assert!(offshore.abs() < EPS, "far offshore should carry no edge");
    }

    #[test]
    fn flow_line_grows_with_flow_and_saturates() {
        let params = fixture_params();
        let mut inputs = fixture_inputs();
        inputs.flow_speed = 0.0;
        assert!(plan_npr(params, inputs).flow_line.abs() < EPS);
        inputs.flow_speed = 1.0;
        let slow = plan_npr(params, inputs).flow_line;
        inputs.flow_speed = 100.0;
        let fast = plan_npr(params, inputs).flow_line;
        assert!(fast > slow);
        assert!((fast - 1.0).abs() < EPS, "flow line should saturate at 1");
    }

    #[test]
    fn plan_npr_is_deterministic() {
        let params = fixture_params();
        let inputs = fixture_inputs();
        assert_eq!(plan_npr(params, inputs), plan_npr(params, inputs));
    }
}
