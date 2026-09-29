//! Waterline mask and above/below-water transition.
//!
//! The waterline is where the animated water surface crosses solid geometry
//! (a shoreline, a pier piling, a swimmer's torso). Rendering needs three
//! things there: a boolean above/below test to route underwater shading, a
//! soft transition weight so the surface intersection does not alias into a
//! hard line, and a shallow-water shoreline band that seeds wet sand and
//! shore foam. This module derives all three from a sample's world height and
//! the local water-surface height.
//!
//! Everything is a pure, deterministic function of caller-supplied heights and
//! depths; there is no sampling of textures or geometry here, only signed
//! comparisons, division, and clamping (no float equality). Transition weights
//! are clamped to `0..=1` and are monotonic in submersion depth.

use super::EPS;

/// Tuning for the soft waterline and its shoreline band.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterlineParams {
    /// Half-width of the soft transition straddling the surface, in meters.
    /// A sample this far below the surface reads fully submerged; this far
    /// above reads fully dry.
    pub transition_half_width: f32,
    /// Total water depth below which a submerged sample is "shallow" and part
    /// of the shoreline band, in meters.
    pub shoreline_depth: f32,
}

/// Signed submersion depth: `water_surface_y - sample_y`.
///
/// Positive when the sample sits below the surface (submerged), negative when
/// it is above (in air), and near zero right at the waterline.
#[must_use]
pub fn submersion_depth(sample_y: f32, water_surface_y: f32) -> f32 {
    water_surface_y - sample_y
}

/// `true` when the sample is at or below the water surface.
#[must_use]
pub fn is_underwater(sample_y: f32, water_surface_y: f32) -> bool {
    submersion_depth(sample_y, water_surface_y) >= 0.0
}

/// Soft waterline weight in `0..=1`: `0` fully in air, `1` fully submerged.
///
/// Linearly ramps across `2 * transition_half_width` centred on the surface,
/// so the intersection fades smoothly instead of aliasing. Monotonic
/// non-decreasing in submersion depth. A degenerate (zero) band collapses to a
/// hard step at the surface.
#[must_use]
pub fn waterline_weight(sample_y: f32, water_surface_y: f32, params: WaterlineParams) -> f32 {
    let depth = submersion_depth(sample_y, water_surface_y);
    let half = params.transition_half_width;
    if half <= EPS {
        return if depth >= 0.0 { 1.0 } else { 0.0 };
    }
    // depth = -half -> 0, depth = +half -> 1.
    ((depth + half) / (2.0 * half)).clamp(0.0, 1.0)
}

/// Shoreline-band weight in `0..=1` for a submerged, shallow sample.
///
/// Zero for dry samples and for deep water; it rises toward `1` as the total
/// water depth drops below `shoreline_depth`, marking the wet, breaking edge
/// where shore foam and wet-sand darkening live. Monotonic non-increasing in
/// water depth.
#[must_use]
pub fn shoreline_band(
    sample_y: f32,
    water_surface_y: f32,
    water_depth: f32,
    params: WaterlineParams,
) -> f32 {
    if !is_underwater(sample_y, water_surface_y) {
        return 0.0;
    }
    let reach = if params.shoreline_depth > EPS {
        params.shoreline_depth
    } else {
        return 0.0;
    };
    let clamped_depth = water_depth.max(0.0);
    (1.0 - clamped_depth / reach).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const PARAMS: WaterlineParams = WaterlineParams {
        transition_half_width: 0.25,
        shoreline_depth: 1.0,
    };

    #[test]
    fn submersion_and_underwater_agree() {
        assert!(submersion_depth(1.0, 2.0) > 0.0);
        assert!(is_underwater(1.0, 2.0));
        assert!(submersion_depth(3.0, 2.0) < 0.0);
        assert!(!is_underwater(3.0, 2.0));
        // Exactly at the surface counts as underwater (inclusive).
        assert!(is_underwater(2.0, 2.0));
    }

    #[test]
    fn waterline_weight_is_monotonic_and_bounded() {
        let surface = 5.0;
        let mut prev = waterline_weight(surface + 2.0, surface, PARAMS);
        let mut y = surface + 2.0;
        // Sweep from high above down to well below; weight must not decrease.
        while y >= surface - 2.0 {
            let w = waterline_weight(y, surface, PARAMS);
            assert!((0.0..=1.0).contains(&w), "weight out of range: {w}");
            assert!(w + EPS >= prev, "weight must rise as the sample submerges");
            prev = w;
            y -= 0.05;
        }
        // Endpoints saturate.
        assert_eq!(waterline_weight(surface + 1.0, surface, PARAMS), 0.0);
        assert_eq!(waterline_weight(surface - 1.0, surface, PARAMS), 1.0);
        // Midpoint sits at one half.
        assert!((waterline_weight(surface, surface, PARAMS) - 0.5).abs() < EPS);
    }

    #[test]
    fn zero_band_is_a_hard_step() {
        let params = WaterlineParams {
            transition_half_width: 0.0,
            ..PARAMS
        };
        assert_eq!(waterline_weight(4.9, 5.0, params), 1.0);
        assert_eq!(waterline_weight(5.1, 5.0, params), 0.0);
    }

    #[test]
    fn shoreline_band_only_fires_underwater_and_grows_toward_shore() {
        // Dry sample: never in the band.
        assert_eq!(shoreline_band(6.0, 5.0, 0.1, PARAMS), 0.0);
        // Submerged deep water: no shoreline.
        assert_eq!(shoreline_band(4.0, 5.0, 5.0, PARAMS), 0.0);
        // Submerged shallow water: rises as depth shrinks.
        let shallow = shoreline_band(4.9, 5.0, 0.2, PARAMS);
        let shallower = shoreline_band(4.95, 5.0, 0.05, PARAMS);
        assert!(shallow > 0.0);
        assert!(shallower >= shallow);
    }

    #[test]
    fn shoreline_band_is_monotonic_in_water_depth() {
        let mut prev = shoreline_band(4.0, 5.0, 0.0, PARAMS);
        let mut depth = 0.0;
        while depth <= 1.5 {
            let w = shoreline_band(4.0, 5.0, depth, PARAMS);
            assert!((0.0..=1.0).contains(&w));
            assert!(w <= prev + EPS, "band must not grow with depth");
            prev = w;
            depth += 0.05;
        }
    }
}
