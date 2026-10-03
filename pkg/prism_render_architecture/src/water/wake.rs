//! Ship-hull `Kelvin` wake planning: deterministic interaction-source trains.
//!
//! A displacement hull moving across the water drags a wake behind it. On deep
//! water that wake is confined to the `Kelvin` wedge, a V whose half-angle is
//! `arcsin(1/3)` (about 19.47 degrees) off the ship's track, independent of
//! speed. Inside the wedge live two wave families: a *transverse* train whose
//! crests run across the track, and two *divergent* trains that feather along
//! the wedge boundary (the cusp lines). This matches the art-directable boat
//! wakes of `Sea of Thieves`, `AC`, and `RDR2` at the algorithm level.
//!
//! This module owns the pure, deterministic geometry and amplitude math that
//! turns a hull's motion into a bounded set of [`WakeSource`] stamps. Each
//! stamp is an analytic `Gerstner`-style disturbance that the pipeline submits
//! into a water body as an interaction source (see
//! [`InteractionSourceHandle`]); the actual field injection runs elsewhere.
//!
//! Only `sqrt` is used, there are no `f32` equality tests, and there is no
//! AI/ML. Degenerate input (slow hull, zero heading, invalid config) yields an
//! empty plan rather than a panic.

use super::{InteractionSourceHandle, Vec2, EPS, TWO_PI};
use alloc::vec::Vec;

/// Sine of the `Kelvin` wake half-angle (`1/3` on deep water).
pub const KELVIN_HALF_ANGLE_SIN: f32 = 1.0 / 3.0;

/// Cosine of the `Kelvin` wake half-angle (`2*sqrt(2)/3`).
pub const KELVIN_HALF_ANGLE_COS: f32 = 0.942_809_f32;

/// The `Kelvin` wake half-angle in radians (`arcsin(1/3)`, about 19.47 deg).
pub const KELVIN_HALF_ANGLE_RAD: f32 = 0.339_836_9;

/// Deep-water transverse wavelength of a wake driven at `speed`.
///
/// The dominant transverse wave travels with the ship, so its phase speed
/// equals `speed`; deep-water gravity waves obey `c^2 = g*lambda/(2*pi)`, which
/// gives `lambda = 2*pi*speed^2/g`. The result grows with the square of the
/// speed, so a faster hull leaves longer, more widely spaced crests. The value
/// is non-negative and zero for a stationary hull or a non-positive gravity.
#[must_use]
pub fn transverse_wavelength(speed: f32, gravity: f32) -> f32 {
    let g = gravity.max(0.0);
    if g <= EPS {
        return 0.0;
    }
    let v = speed.max(0.0);
    TWO_PI * v * v / g
}

/// Length-based `Froude` number `Fr = speed / sqrt(g*length)`.
///
/// `Fr` ranks how hard a hull of a given length drives its wake: higher `Fr`
/// means a larger, more divergent wake. The value is non-negative and zero for
/// a degenerate length or gravity.
#[must_use]
pub fn froude_length(speed: f32, length_m: f32, gravity: f32) -> f32 {
    let denom = (gravity.max(0.0) * length_m.max(0.0)).max(0.0);
    if denom <= EPS {
        return 0.0;
    }
    speed.max(0.0) / denom.sqrt()
}

/// Which wake family a [`WakeSource`] belongs to.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum WakeTrain {
    /// Crests across the track, propagating along the stern direction.
    Transverse,
    /// Port-side divergent arm along the `Kelvin` wedge boundary.
    DivergentPort,
    /// Starboard-side divergent arm along the `Kelvin` wedge boundary.
    DivergentStarboard,
}

/// Kinematic state of one hull on the horizontal water plane.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HullMotion {
    /// Hull reference point (stern/centre) on the water plane.
    pub position: Vec2,
    /// Heading; need not be unit length (it is normalised internally).
    pub forward: Vec2,
    /// Forward speed through the water (metres per second).
    pub speed: f32,
    /// Beam (width) of the hull in metres; widens the cusp offset.
    pub beam_m: f32,
    /// Draft (submerged depth) in metres; deeper hulls displace more water.
    pub draft_m: f32,
}

/// Tuning for how a hull sheds wake interaction sources.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WakeConfig {
    /// Gravitational acceleration used for the wavelength law.
    pub gravity: f32,
    /// Below this speed the hull leaves no wake.
    pub min_speed: f32,
    /// Along-track spacing between emitted source stamps (metres).
    pub source_spacing_m: f32,
    /// How far behind the hull the wake persists (metres).
    pub trail_length_m: f32,
    /// Speed at which the emission amplitude is half of its asymptote.
    pub reference_speed: f32,
    /// Amplitude scale at full drive and unit draft.
    pub base_amplitude: f32,
    /// Hard cap on the number of stamps per train.
    pub max_sources: u32,
}

impl WakeConfig {
    /// Returns `true` when every field is finite-positive where required.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.gravity > EPS
            && self.source_spacing_m > EPS
            && self.trail_length_m > EPS
            && self.reference_speed > EPS
            && self.base_amplitude >= 0.0
            && self.min_speed >= 0.0
            && self.max_sources > 0
    }
}

/// One analytic wake disturbance stamped into a water body.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WakeSource {
    /// Stamp centre on the water plane.
    pub position: Vec2,
    /// Unit propagation direction of the wave crest normal.
    pub direction: Vec2,
    /// Peak displacement amplitude (non-negative, decays with age).
    pub amplitude: f32,
    /// Dominant wavelength of this stamp.
    pub wavelength: f32,
    /// Normalised age `0..=1`: zero at the hull, one at the trail end.
    pub age_frac: f32,
    /// Which wake family this stamp belongs to.
    pub train: WakeTrain,
}

/// Emission amplitude a hull sheds this frame, before per-stamp age decay.
///
/// Returns zero below `min_speed`. Above it the amplitude rises monotonically
/// with speed through a saturating factor `s/(s+reference_speed)` in `0..1`,
/// and scales with draft because a deeper hull displaces more water. The value
/// is non-negative.
#[must_use]
pub fn emission_amplitude(hull: &HullMotion, cfg: &WakeConfig) -> f32 {
    if !cfg.is_valid() || hull.speed < cfg.min_speed {
        return 0.0;
    }
    let drive = (hull.speed - cfg.min_speed).max(0.0);
    let speed_factor = drive / (drive + cfg.reference_speed);
    let draft_scale = 1.0 + hull.draft_m.max(0.0);
    cfg.base_amplitude.max(0.0) * speed_factor * draft_scale
}

/// Rotate `v` by the angle whose cosine/sine are `cos_t`/`sin_t`.
fn rotate(v: Vec2, cos_t: f32, sin_t: f32) -> Vec2 {
    Vec2::new(v.x * cos_t - v.y * sin_t, v.x * sin_t + v.y * cos_t)
}

/// Plan the full set of wake interaction sources behind a moving hull.
///
/// Emits up to `cap = min(max_sources, floor(trail_length/spacing))` stations
/// along the track; each station sheds three stamps (one transverse, two
/// divergent on the `Kelvin` wedge). Stamp amplitude decays linearly to zero at
/// the trail end, so stamps farther behind the hull are weaker. Output order is
/// deterministic (station-major, then transverse, port, starboard). A slow
/// hull, a zero heading, or an invalid config yields an empty plan.
#[must_use]
pub fn plan_wake_sources(hull: &HullMotion, cfg: &WakeConfig) -> Vec<WakeSource> {
    let mut out = Vec::new();
    let emit = emission_amplitude(hull, cfg);
    if emit <= EPS {
        return out;
    }
    let fwd = hull.forward.normalize_or_zero();
    // A unit heading has length_squared == 1; the zero vector has 0.
    if fwd.length_squared() < 0.5 {
        return out;
    }
    let back = fwd.scale(-1.0);
    let perp = Vec2::new(-fwd.y, fwd.x);
    let wavelength = transverse_wavelength(hull.speed, cfg.gravity);

    let stations = (cfg.trail_length_m / cfg.source_spacing_m) as u32;
    let cap = stations.min(cfg.max_sources);
    let wedge_tan = KELVIN_HALF_ANGLE_SIN / KELVIN_HALF_ANGLE_COS;
    let half_beam = 0.5 * hull.beam_m.max(0.0);

    for i in 0..cap {
        let dist = (f32::from(u16::try_from(i + 1).unwrap_or(u16::MAX))) * cfg.source_spacing_m;
        let age = (dist / cfg.trail_length_m).min(1.0);
        let amplitude = emit * (1.0 - age).max(0.0);
        let centre = hull.position.add(back.scale(dist));
        let lateral = half_beam + dist * wedge_tan;

        out.push(WakeSource {
            position: centre,
            direction: back,
            amplitude,
            wavelength,
            age_frac: age,
            train: WakeTrain::Transverse,
        });
        let port_dir = rotate(back, KELVIN_HALF_ANGLE_COS, KELVIN_HALF_ANGLE_SIN);
        out.push(WakeSource {
            position: centre.add(perp.scale(lateral)),
            direction: port_dir,
            amplitude,
            wavelength,
            age_frac: age,
            train: WakeTrain::DivergentPort,
        });
        let star_dir = rotate(back, KELVIN_HALF_ANGLE_COS, -KELVIN_HALF_ANGLE_SIN);
        out.push(WakeSource {
            position: centre.add(perp.scale(-lateral)),
            direction: star_dir,
            amplitude,
            wavelength,
            age_frac: age,
            train: WakeTrain::DivergentStarboard,
        });
    }
    out
}

/// Pair each planned stamp with a sequential interaction-source handle.
///
/// Handles are assigned `base, base+1, ...` in plan order, so a caller can
/// submit the stamps as distinct interaction sources on a water body. The
/// pairing is deterministic and the handles are unique within the batch.
#[must_use]
pub fn assign_handles(
    sources: &[WakeSource],
    base: u32,
) -> Vec<(InteractionSourceHandle, WakeSource)> {
    let mut out = Vec::with_capacity(sources.len());
    for (i, src) in sources.iter().enumerate() {
        let id = base.wrapping_add(u32::try_from(i).unwrap_or(u32::MAX));
        out.push((InteractionSourceHandle(id), *src));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::water::GRAVITY;

    const CFG: WakeConfig = WakeConfig {
        gravity: GRAVITY,
        min_speed: 0.5,
        source_spacing_m: 2.0,
        trail_length_m: 20.0,
        reference_speed: 6.0,
        base_amplitude: 1.0,
        max_sources: 64,
    };

    fn hull(speed: f32) -> HullMotion {
        HullMotion {
            position: Vec2::new(10.0, -4.0),
            forward: Vec2::new(1.0, 0.0),
            speed,
            beam_m: 3.0,
            draft_m: 1.5,
        }
    }

    #[test]
    fn slow_hull_emits_no_wake() {
        let h = hull(0.2);
        assert!(plan_wake_sources(&h, &CFG).is_empty());
        assert!(emission_amplitude(&h, &CFG) < EPS);
    }

    #[test]
    fn invalid_config_yields_no_sources() {
        let bad = WakeConfig {
            source_spacing_m: 0.0,
            ..CFG
        };
        assert!(plan_wake_sources(&hull(5.0), &bad).is_empty());
        assert!(!bad.is_valid());
    }

    #[test]
    fn zero_heading_yields_no_sources() {
        let h = HullMotion {
            forward: Vec2::ZERO,
            ..hull(5.0)
        };
        assert!(plan_wake_sources(&h, &CFG).is_empty());
    }

    #[test]
    fn sources_lie_behind_the_hull() {
        let h = hull(5.0);
        let plan = plan_wake_sources(&h, &CFG);
        assert!(!plan.is_empty());
        for s in &plan {
            let rel = s.position.sub(h.position);
            // Behind means a negative projection onto the heading.
            assert!(rel.dot(h.forward) < -EPS, "stamp must sit astern");
        }
    }

    #[test]
    fn amplitude_decays_with_age_along_the_transverse_train() {
        let plan = plan_wake_sources(&hull(5.0), &CFG);
        let trans: Vec<f32> = plan
            .iter()
            .filter(|s| s.train == WakeTrain::Transverse)
            .map(|s| s.amplitude)
            .collect();
        assert!(trans.len() > 2);
        for w in trans.windows(2) {
            assert!(w[0] > w[1], "farther stamps must be weaker");
        }
    }

    #[test]
    fn faster_hull_makes_a_bigger_wake() {
        let slow = emission_amplitude(&hull(2.0), &CFG);
        let fast = emission_amplitude(&hull(8.0), &CFG);
        assert!(fast > slow);
        assert!(slow > 0.0);
    }

    #[test]
    fn faster_hull_has_a_longer_transverse_wavelength() {
        let slow = transverse_wavelength(2.0, GRAVITY);
        let fast = transverse_wavelength(8.0, GRAVITY);
        assert!(fast > slow);
        assert!(slow > 0.0);
        assert!(transverse_wavelength(0.0, GRAVITY) < EPS);
        assert!(transverse_wavelength(5.0, 0.0) < EPS);
    }

    #[test]
    fn divergent_arms_sit_on_the_kelvin_wedge() {
        let plan = plan_wake_sources(&hull(5.0), &CFG);
        let back = Vec2::new(-1.0, 0.0);
        let port = plan
            .iter()
            .find(|s| s.train == WakeTrain::DivergentPort)
            .unwrap();
        let star = plan
            .iter()
            .find(|s| s.train == WakeTrain::DivergentStarboard)
            .unwrap();
        // Unit directions whose dot with the stern axis is cos(half-angle).
        assert!((port.direction.length() - 1.0).abs() < 1e-4);
        assert!((port.direction.dot(back) - KELVIN_HALF_ANGLE_COS).abs() < 1e-4);
        assert!((star.direction.dot(back) - KELVIN_HALF_ANGLE_COS).abs() < 1e-4);
        // Port and starboard are mirror images across the track.
        assert!((port.direction.x - star.direction.x).abs() < 1e-4);
        assert!((port.direction.y + star.direction.y).abs() < 1e-4);
    }

    #[test]
    fn source_count_respects_cap_and_spacing() {
        // 20 / 2 = 10 stations, each sheds 3 stamps.
        let plan = plan_wake_sources(&hull(5.0), &CFG);
        assert_eq!(plan.len(), 30);
        let capped = WakeConfig {
            max_sources: 3,
            ..CFG
        };
        assert_eq!(plan_wake_sources(&hull(5.0), &capped).len(), 9);
    }

    #[test]
    fn plan_is_deterministic() {
        let a = plan_wake_sources(&hull(5.0), &CFG);
        let b = plan_wake_sources(&hull(5.0), &CFG);
        assert_eq!(a, b);
    }

    #[test]
    fn deeper_draft_makes_a_bigger_wake() {
        let shallow = HullMotion {
            draft_m: 0.5,
            ..hull(5.0)
        };
        let deep = HullMotion {
            draft_m: 3.0,
            ..hull(5.0)
        };
        assert!(emission_amplitude(&deep, &CFG) > emission_amplitude(&shallow, &CFG));
    }

    #[test]
    fn handles_are_sequential_and_unique() {
        let plan = plan_wake_sources(&hull(5.0), &CFG);
        let tagged = assign_handles(&plan, 100);
        assert_eq!(tagged.len(), plan.len());
        for (i, (h, _)) in tagged.iter().enumerate() {
            assert_eq!(h.0, 100 + u32::try_from(i).unwrap());
        }
    }

    #[test]
    fn froude_length_is_monotone_and_zero_safe() {
        let slow = froude_length(2.0, 10.0, GRAVITY);
        let fast = froude_length(8.0, 10.0, GRAVITY);
        assert!(fast > slow);
        assert!(slow > 0.0);
        assert!(froude_length(5.0, 0.0, GRAVITY) < EPS);
        assert!(froude_length(5.0, 10.0, 0.0) < EPS);
    }
}
