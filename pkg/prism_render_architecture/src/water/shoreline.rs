//! Shoreline orchestration: one deterministic plan per surface sample.
//!
//! The [`waterline`](super::waterline) and [`wetness`](super::wetness) modules
//! each own one slice of the land/water boundary: the former derives the soft
//! above/below transition and the shallow shoreline band from world heights,
//! the latter tracks how a surface soaks, dries, darkens, and pools over time.
//! Neither knows about the other. This module stitches them into a single pure
//! step so the shading pass and the `SWE` puddle seeding read one coherent
//! record instead of re-deriving overlapping quantities, and so a `GPU`-side
//! consumer sees a stable, allocation-free struct.
//!
//! Everything here is a pure, deterministic function of caller-supplied inputs.
//! There is no hidden state, no sampling, and no `f32` equality; the module
//! only forwards to the upstream helpers and assembles their results. The
//! advanced moisture state returned in [`ShorelinePlan`] is the fresh state to
//! carry into the next frame.

use super::waterline::{
    is_underwater, shoreline_band, submersion_depth, waterline_weight, WaterlineParams,
};
use super::wetness::{
    capillary_height, is_puddle, step_moisture, wet_albedo_scale, SurfaceMoisture, WetnessParams,
};

/// Static tuning shared by every sample of one shoreline surface.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShorelineProfile {
    /// Waterline transition and shoreline-band tuning.
    pub waterline: WaterlineParams,
    /// Wetness, capillary, and puddle tuning.
    pub wetness: WetnessParams,
    /// Puddle drainage rate (meters per second) applied each step.
    pub drain_rate: f32,
}

/// Per-sample, per-frame inputs to a shoreline plan.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShorelineInputs {
    /// World height of the surface sample, in meters.
    pub sample_y: f32,
    /// World height of the local water surface, in meters.
    pub water_surface_y: f32,
    /// Total water column depth at the sample, in meters.
    pub water_depth: f32,
    /// Distance of the sample above the waterline, in meters, for the
    /// capillary band. Ignored for submerged samples by the upstream helper.
    pub dist_above_water: f32,
    /// Moisture state carried in from the previous frame.
    pub moisture: SurfaceMoisture,
    /// Normalized rain drive for wetting and puddle fill.
    pub rain_rate: f32,
    /// Timestep in seconds.
    pub dt: f32,
}

/// The fully resolved shoreline decision for one sample.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShorelinePlan {
    /// Signed submersion depth (`water_surface_y - sample_y`).
    pub submersion: f32,
    /// Soft waterline weight in `0..=1`.
    pub waterline_weight: f32,
    /// Shoreline-band weight in `0..=1`.
    pub shoreline_band: f32,
    /// Advanced moisture state to carry into the next frame.
    pub moisture: SurfaceMoisture,
    /// Albedo multiplier from the advanced wetness.
    pub albedo_scale: f32,
    /// Capillary wet-band height at the sample, in meters.
    pub capillary_height: f32,
    /// Whether the advanced puddle depth crosses the puddle threshold.
    pub is_puddle: bool,
}

/// Resolves a complete [`ShorelinePlan`] for one surface sample.
///
/// Forwards to the upstream waterline and wetness helpers and assembles their
/// results. Water contact is decided by [`is_underwater`], which drives the
/// moisture step; the advanced moisture then feeds the albedo, capillary, and
/// puddle outputs so the whole plan is internally consistent. Pure and
/// deterministic: identical inputs always yield an identical plan.
#[must_use]
pub fn plan_shoreline(profile: ShorelineProfile, inputs: ShorelineInputs) -> ShorelinePlan {
    let submersion = submersion_depth(inputs.sample_y, inputs.water_surface_y);
    let waterline_weight =
        waterline_weight(inputs.sample_y, inputs.water_surface_y, profile.waterline);
    let shoreline_band = shoreline_band(
        inputs.sample_y,
        inputs.water_surface_y,
        inputs.water_depth,
        profile.waterline,
    );
    let water_contact = is_underwater(inputs.sample_y, inputs.water_surface_y);
    let moisture = step_moisture(
        inputs.moisture,
        profile.wetness,
        water_contact,
        inputs.rain_rate,
        profile.drain_rate,
        inputs.dt,
    );
    let albedo_scale = wet_albedo_scale(moisture.wetness, profile.wetness);
    let capillary_height =
        capillary_height(moisture.wetness, inputs.dist_above_water, profile.wetness);
    let is_puddle = is_puddle(moisture.puddle_depth, profile.wetness);
    ShorelinePlan {
        submersion,
        waterline_weight,
        shoreline_band,
        moisture,
        albedo_scale,
        capillary_height,
        is_puddle,
    }
}

#[cfg(test)]
mod tests {
    use super::super::EPS;
    use super::*;

    const WATERLINE: WaterlineParams = WaterlineParams {
        transition_half_width: 0.25,
        shoreline_depth: 1.0,
    };
    const WETNESS: WetnessParams = WetnessParams {
        max_capillary_height: 0.5,
        absorb_rate: 2.0,
        dry_rate: 0.5,
        darkening_strength: 0.4,
        puddle_threshold: 0.02,
    };
    const PROFILE: ShorelineProfile = ShorelineProfile {
        waterline: WATERLINE,
        wetness: WETNESS,
        drain_rate: 0.1,
    };

    fn dry_state() -> SurfaceMoisture {
        SurfaceMoisture {
            wetness: 0.0,
            puddle_depth: 0.0,
        }
    }

    #[test]
    fn submerged_sample_is_fully_weighted_and_soaks() {
        let inputs = ShorelineInputs {
            sample_y: -1.0,
            water_surface_y: 1.0,
            water_depth: 3.0,
            dist_above_water: 0.0,
            moisture: dry_state(),
            rain_rate: 0.0,
            dt: 1.0,
        };
        let plan = plan_shoreline(PROFILE, inputs);
        // Deep below the surface: fully submerged weight.
        assert!((plan.waterline_weight - 1.0).abs() < EPS);
        assert!(plan.submersion > 0.0);
        // Direct water contact wets the surface from bone dry.
        assert!(plan.moisture.wetness > inputs.moisture.wetness);
        // All outputs stay in their declared ranges.
        assert!((0.0..=1.0).contains(&plan.waterline_weight));
        assert!((0.0..=1.0).contains(&plan.shoreline_band));
        assert!((0.0..=1.0).contains(&plan.moisture.wetness));
    }

    #[test]
    fn dry_sample_above_water_wets_under_rain() {
        let inputs = ShorelineInputs {
            sample_y: 2.0,
            water_surface_y: 1.0,
            water_depth: 0.0,
            dist_above_water: 1.0,
            moisture: dry_state(),
            rain_rate: 0.8,
            dt: 1.0,
        };
        let plan = plan_shoreline(PROFILE, inputs);
        // Above water: no submersion, no shoreline band.
        assert!(plan.submersion < 0.0);
        assert!((0.0..=1.0).contains(&plan.waterline_weight));
        // Rain drives wetness up from dry.
        assert!(plan.moisture.wetness > 0.0);
    }

    #[test]
    fn albedo_scale_stays_in_band_and_darkens_with_wetness() {
        let lower = 1.0 - WETNESS.darkening_strength;
        let light = ShorelineInputs {
            sample_y: 2.0,
            water_surface_y: 1.0,
            water_depth: 0.0,
            dist_above_water: 0.2,
            moisture: dry_state(),
            rain_rate: 0.2,
            dt: 1.0,
        };
        let heavy = ShorelineInputs {
            moisture: SurfaceMoisture {
                wetness: 0.9,
                puddle_depth: 0.0,
            },
            rain_rate: 1.0,
            ..light
        };
        let light_plan = plan_shoreline(PROFILE, light);
        let heavy_plan = plan_shoreline(PROFILE, heavy);
        assert!((lower..=1.0).contains(&light_plan.albedo_scale));
        assert!((lower..=1.0).contains(&heavy_plan.albedo_scale));
        // A wetter surface is never brighter.
        assert!(heavy_plan.albedo_scale <= light_plan.albedo_scale + EPS);
    }

    #[test]
    fn heavy_rain_fills_a_puddle() {
        let inputs = ShorelineInputs {
            sample_y: 2.0,
            water_surface_y: 1.0,
            water_depth: 0.0,
            dist_above_water: 0.5,
            moisture: dry_state(),
            rain_rate: 0.5,
            dt: 1.0,
        };
        let plan = plan_shoreline(PROFILE, inputs);
        // Rain minus drainage leaves standing water past the threshold.
        assert!(plan.moisture.puddle_depth > WETNESS.puddle_threshold);
        assert!(plan.is_puddle);
    }

    #[test]
    fn no_rain_dry_surface_leaves_no_puddle() {
        let inputs = ShorelineInputs {
            sample_y: 2.0,
            water_surface_y: 1.0,
            water_depth: 0.0,
            dist_above_water: 0.5,
            moisture: dry_state(),
            rain_rate: 0.0,
            dt: 1.0,
        };
        let plan = plan_shoreline(PROFILE, inputs);
        assert!(!plan.is_puddle);
        assert!(plan.moisture.puddle_depth >= 0.0);
    }

    #[test]
    fn plan_is_deterministic() {
        let inputs = ShorelineInputs {
            sample_y: 0.3,
            water_surface_y: 0.5,
            water_depth: 0.4,
            dist_above_water: 0.1,
            moisture: SurfaceMoisture {
                wetness: 0.25,
                puddle_depth: 0.01,
            },
            rain_rate: 0.3,
            dt: 0.5,
        };
        let a = plan_shoreline(PROFILE, inputs);
        let b = plan_shoreline(PROFILE, inputs);
        assert_eq!(a, b);
    }

    #[test]
    fn all_outputs_stay_bounded_across_a_sweep() {
        let lower = 1.0 - WETNESS.darkening_strength;
        let mut sample = -1.0_f32;
        while sample <= 2.0 {
            let inputs = ShorelineInputs {
                sample_y: sample,
                water_surface_y: 0.5,
                water_depth: (2.0 - sample).max(0.0),
                dist_above_water: (sample - 0.5).max(0.0),
                moisture: SurfaceMoisture {
                    wetness: 0.3,
                    puddle_depth: 0.0,
                },
                rain_rate: 0.4,
                dt: 0.5,
            };
            let plan = plan_shoreline(PROFILE, inputs);
            assert!((0.0..=1.0).contains(&plan.waterline_weight));
            assert!((0.0..=1.0).contains(&plan.shoreline_band));
            assert!((0.0..=1.0).contains(&plan.moisture.wetness));
            assert!(plan.moisture.puddle_depth >= 0.0);
            assert!((lower..=1.0).contains(&plan.albedo_scale));
            assert!((0.0..=WETNESS.max_capillary_height).contains(&plan.capillary_height));
            sample += 0.1;
        }
    }
}
