//! Wetness, capillary rise, puddles, and rain coupling for shoreline surfaces.
//!
//! Solid geometry near water does not switch instantly from dry to wet at the
//! waterline. Capillary action wicks moisture a short distance up above the
//! surface, wet material reads darker and glossier, and depressions collect
//! standing water that drains slowly. Weather feeds the same fields: rain wets
//! exposed surfaces and fills puddles, while a dry spell evaporates both back
//! out. This module derives all of that as pure decisions over a caller-owned
//! moisture state, so the shading pass can darken albedo and the `SWE` solver
//! can seed puddle ripples without either owning the bookkeeping.
//!
//! A surface's wetness is a saturation scalar in `0..=1`: `0` bone dry, `1`
//! fully soaked. Absorption and drying are exponential envelopes built on the
//! shared non-negative `exp_approx` (never `f32::exp`), so both are smooth and
//! strictly monotonic in time. Capillary height, albedo darkening, and puddle
//! depth are likewise monotonic in their drivers and clamped to physical
//! ranges; nothing here tests `f32` equality or divides by an unchecked zero.

use super::{exp_approx, EPS};

/// Tuning for the wetness, capillary, and puddle response of a surface.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WetnessParams {
    /// Maximum capillary rise above the waterline, in meters. Also the reach
    /// over which the wet band fades to nothing; `<= 0` disables capillary
    /// rise entirely.
    pub max_capillary_height: f32,
    /// Absorption rate (per second) while a surface is in contact with water
    /// or under full rain. Larger values soak the surface faster.
    pub absorb_rate: f32,
    /// Drying rate (per second) while a surface is exposed and rain-free.
    /// Larger values dry the surface faster.
    pub dry_rate: f32,
    /// Peak fractional albedo darkening at full saturation, in `0..=1`. A wet
    /// surface's albedo is scaled toward `1 - darkening_strength`.
    pub darkening_strength: f32,
    /// Accumulated puddle depth (meters) above which a cell counts as a puddle
    /// worth seeding into the shallow-water solver.
    pub puddle_threshold: f32,
}

/// Per-surface moisture state carried between frames.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceMoisture {
    /// Wetness saturation in `0..=1`.
    pub wetness: f32,
    /// Standing puddle depth in meters (`>= 0`).
    pub puddle_depth: f32,
}

/// Absorbs moisture over `dt` at `rate`, driving saturation toward `1`.
///
/// Returns `1 - (1 - w) * exp(-rate * dt)`: the remaining dry fraction decays
/// exponentially. The result stays in `0..=1`, never drops below the input,
/// and rises monotonically as either `dt` or `rate` grows.
#[must_use]
pub fn absorb(wetness: f32, rate: f32, dt: f32) -> f32 {
    let w = wetness.clamp(0.0, 1.0);
    let dry_fraction = (1.0 - w) * exp_approx(-rate.max(0.0) * dt.max(0.0));
    (1.0 - dry_fraction).clamp(0.0, 1.0)
}

/// Dries moisture over `dt` at `rate`, driving saturation toward `0`.
///
/// Returns `w * exp(-rate * dt)` via the shared non-negative `exp_approx`. The
/// result is non-negative, never exceeds the input, and decreases
/// monotonically as either `dt` or `rate` grows.
#[must_use]
pub fn dry(wetness: f32, rate: f32, dt: f32) -> f32 {
    let w = wetness.clamp(0.0, 1.0);
    w * exp_approx(-rate.max(0.0) * dt.max(0.0))
}

/// Advances a surface's wetness saturation for one step.
///
/// Direct water contact soaks at the full `absorb_rate`. Otherwise rain wets
/// the surface at a rate scaled by the (clamped) rain intensity, and a
/// rain-free exposed surface dries at `dry_rate`. `rain_rate` is a normalized
/// wetting drive; only its non-negative part up to `1` contributes. The result
/// is always a valid saturation in `0..=1`.
#[must_use]
pub fn update_wetness(
    current: f32,
    params: WetnessParams,
    water_contact: bool,
    rain_rate: f32,
    dt: f32,
) -> f32 {
    if water_contact {
        return absorb(current, params.absorb_rate, dt);
    }
    let rain = rain_rate.max(0.0);
    if rain > EPS {
        absorb(current, params.absorb_rate * rain.min(1.0), dt)
    } else {
        dry(current, params.dry_rate, dt)
    }
}

/// Albedo multiplier for a surface at the given saturation.
///
/// Scales linearly from `1` (dry) down to `1 - darkening_strength` (fully wet),
/// so wet material reads darker. Monotonically non-increasing in wetness and
/// clamped to `1 - darkening_strength ..= 1`.
#[must_use]
pub fn wet_albedo_scale(wetness: f32, params: WetnessParams) -> f32 {
    let w = wetness.clamp(0.0, 1.0);
    let darkening = params.darkening_strength.clamp(0.0, 1.0);
    (1.0 - darkening * w).clamp(1.0 - darkening, 1.0)
}

/// Capillary wet-band height at a point `dist_above_water` meters above the
/// waterline for a surface at the given saturation.
///
/// Rises to `max_capillary_height` right at the waterline for a fully soaked
/// surface and fades linearly to zero by the top of the reach. Monotonically
/// non-increasing in distance above the water, monotonically non-decreasing in
/// saturation, and clamped to `0 ..= max_capillary_height`. A non-positive
/// reach disables the effect.
#[must_use]
pub fn capillary_height(wetness: f32, dist_above_water: f32, params: WetnessParams) -> f32 {
    let reach = params.max_capillary_height;
    if reach <= EPS {
        return 0.0;
    }
    let w = wetness.clamp(0.0, 1.0);
    let falloff = (1.0 - dist_above_water.max(0.0) / reach).clamp(0.0, 1.0);
    reach * w * falloff
}

/// Integrates a puddle's standing depth over `dt`.
///
/// Rain adds depth and drainage removes it: `depth + (rain_rate - drain_rate) *
/// dt`, clamped at zero so a puddle never goes negative. Monotonically
/// non-decreasing in `rain_rate` and non-increasing in `drain_rate`.
#[must_use]
pub fn puddle_depth(accumulated: f32, rain_rate: f32, drain_rate: f32, dt: f32) -> f32 {
    let base = accumulated.max(0.0);
    (base + (rain_rate.max(0.0) - drain_rate.max(0.0)) * dt.max(0.0)).max(0.0)
}

/// `true` when a puddle is deep enough to seed shallow-water ripples.
///
/// A non-positive threshold means any positive depth counts.
#[must_use]
pub fn is_puddle(depth: f32, params: WetnessParams) -> bool {
    let threshold = params.puddle_threshold.max(0.0);
    if threshold <= EPS {
        return depth > EPS;
    }
    depth >= threshold
}

/// Advances both wetness and puddle depth for one weather/contact step.
///
/// Couples the two fields to the same rain drive: [`update_wetness`] soaks or
/// dries the surface film while [`puddle_depth`] fills or drains the standing
/// pool. Direct water contact both soaks the film and (in the absence of
/// drainage exceeding it) is expected to keep any pool full via the caller's
/// `rain_rate`/`drain_rate` choice. The returned state is always valid: wetness
/// in `0..=1` and puddle depth `>= 0`.
#[must_use]
pub fn step_moisture(
    state: SurfaceMoisture,
    params: WetnessParams,
    water_contact: bool,
    rain_rate: f32,
    drain_rate: f32,
    dt: f32,
) -> SurfaceMoisture {
    let wetness = update_wetness(state.wetness, params, water_contact, rain_rate, dt);
    let puddle_depth = puddle_depth(state.puddle_depth, rain_rate, drain_rate, dt);
    SurfaceMoisture {
        wetness,
        puddle_depth,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PARAMS: WetnessParams = WetnessParams {
        max_capillary_height: 0.5,
        absorb_rate: 2.0,
        dry_rate: 0.5,
        darkening_strength: 0.4,
        puddle_threshold: 0.02,
    };

    #[test]
    fn absorb_rises_monotonically_toward_one() {
        let mut prev = absorb(0.0, PARAMS.absorb_rate, 0.0);
        let mut dt = 0.0;
        while dt <= 5.0 {
            let w = absorb(0.0, PARAMS.absorb_rate, dt);
            assert!((0.0..=1.0).contains(&w), "wetness out of range: {w}");
            assert!(w + EPS >= prev, "absorption must not decrease over time");
            prev = w;
            dt += 0.1;
        }
        // Longer soaking approaches full saturation.
        assert!(absorb(0.0, PARAMS.absorb_rate, 10.0) > 0.99);
        // Zero time leaves the input untouched.
        assert!((absorb(0.3, PARAMS.absorb_rate, 0.0) - 0.3).abs() < EPS);
    }

    #[test]
    fn dry_falls_monotonically_toward_zero() {
        let mut prev = dry(1.0, PARAMS.dry_rate, 0.0);
        let mut dt = 0.0;
        while dt <= 10.0 {
            let w = dry(1.0, PARAMS.dry_rate, dt);
            assert!((0.0..=1.0).contains(&w), "wetness out of range: {w}");
            assert!(w <= prev + EPS, "drying must not increase over time");
            prev = w;
            dt += 0.2;
        }
        // Never negative, and never above the starting saturation.
        assert!(dry(1.0, PARAMS.dry_rate, 100.0) >= 0.0);
        assert!(dry(0.5, PARAMS.dry_rate, 1.0) <= 0.5 + EPS);
    }

    #[test]
    fn wet_albedo_scale_darkens_monotonically() {
        assert!((wet_albedo_scale(0.0, PARAMS) - 1.0).abs() < EPS);
        let fully = wet_albedo_scale(1.0, PARAMS);
        assert!((fully - (1.0 - PARAMS.darkening_strength)).abs() < EPS);
        let mut prev = wet_albedo_scale(0.0, PARAMS);
        let mut w = 0.0;
        while w <= 1.0 {
            let s = wet_albedo_scale(w, PARAMS);
            assert!((1.0 - PARAMS.darkening_strength..=1.0).contains(&s));
            assert!(s <= prev + EPS, "wetter surfaces must not brighten");
            prev = s;
            w += 0.05;
        }
    }

    #[test]
    fn capillary_height_falls_with_distance_and_rises_with_wetness() {
        // Monotonic decrease as we climb away from the waterline.
        let mut prev = capillary_height(1.0, 0.0, PARAMS);
        let mut dist = 0.0;
        while dist <= 0.6 {
            let h = capillary_height(1.0, dist, PARAMS);
            assert!((0.0..=PARAMS.max_capillary_height).contains(&h));
            assert!(h <= prev + EPS, "capillary band must not grow with height");
            prev = h;
            dist += 0.02;
        }
        // At the waterline a fully soaked surface reaches the maximum.
        assert!((capillary_height(1.0, 0.0, PARAMS) - PARAMS.max_capillary_height).abs() < EPS);
        // Beyond the reach it is dry.
        assert_eq!(capillary_height(1.0, 1.0, PARAMS), 0.0);
        // Monotonic increase with saturation at a fixed height.
        assert!(capillary_height(0.8, 0.1, PARAMS) >= capillary_height(0.2, 0.1, PARAMS));
        // Degenerate reach disables the effect.
        let no_reach = WetnessParams {
            max_capillary_height: 0.0,
            ..PARAMS
        };
        assert_eq!(capillary_height(1.0, 0.0, no_reach), 0.0);
    }

    #[test]
    fn puddle_depth_accumulates_non_negatively() {
        // Rain fills a puddle.
        let filled = puddle_depth(0.0, 0.01, 0.0, 1.0);
        assert!(filled > 0.0);
        // More rain fills faster.
        assert!(puddle_depth(0.0, 0.02, 0.0, 1.0) >= filled);
        // Drainage empties it but never past zero.
        let drained = puddle_depth(0.05, 0.0, 1.0, 1.0);
        assert!(drained >= 0.0);
        assert_eq!(drained, 0.0);
        // Stronger drainage never leaves more water behind.
        assert!(puddle_depth(0.1, 0.0, 0.5, 0.1) >= puddle_depth(0.1, 0.0, 1.0, 0.1));
    }

    #[test]
    fn is_puddle_respects_threshold() {
        assert!(!is_puddle(0.01, PARAMS));
        assert!(is_puddle(0.03, PARAMS));
        // Non-positive threshold: any positive depth is a puddle.
        let any = WetnessParams {
            puddle_threshold: 0.0,
            ..PARAMS
        };
        assert!(is_puddle(0.001, any));
        assert!(!is_puddle(0.0, any));
    }

    #[test]
    fn update_wetness_routes_contact_rain_and_drying() {
        // Contact soaks toward full saturation.
        let soaked = update_wetness(0.0, PARAMS, true, 0.0, 1.0);
        assert!(soaked > 0.0);
        // Rain wets an exposed surface, but slower than direct contact.
        let rained = update_wetness(0.0, PARAMS, false, 0.5, 1.0);
        assert!(rained > 0.0);
        assert!(rained <= soaked + EPS);
        // A dry, rain-free surface loses moisture.
        let dried = update_wetness(0.8, PARAMS, false, 0.0, 1.0);
        assert!(dried < 0.8);
    }

    #[test]
    fn step_moisture_couples_wetness_and_puddle() {
        let start = SurfaceMoisture {
            wetness: 0.0,
            puddle_depth: 0.0,
        };
        // Rain wets the film and fills the pool together.
        let after_rain = step_moisture(start, PARAMS, false, 0.03, 0.0, 1.0);
        assert!(after_rain.wetness > 0.0);
        assert!(after_rain.puddle_depth > 0.0);
        // A dry spell dries the film and drains the pool, both staying valid.
        let after_dry = step_moisture(after_rain, PARAMS, false, 0.0, 0.05, 1.0);
        assert!(after_dry.wetness < after_rain.wetness);
        assert!(after_dry.puddle_depth >= 0.0);
        assert!((0.0..=1.0).contains(&after_dry.wetness));
    }
}
