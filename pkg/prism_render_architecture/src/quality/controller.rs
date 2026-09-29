//! A deterministic proportional-integral (`PID`-family) adaptive quality controller.
//!
//! This is the concrete [`QualityController`](super::QualityController) the
//! architecture ships as its reference: a closed-loop feedback controller that
//! watches the measured frame time against the target and steers a single
//! scalar *quality level* `q in [0, 1]` up or down. Every knob a renderer can
//! trade for performance — render scale, geometry error tolerance, shadow page
//! budget, and global-illumination ray density — is a monotone function of that
//! one level, so the controller can never push two knobs in contradictory
//! directions.
//!
//! The loop is a proportional-integral (`PI`) controller, the practical subset
//! of the `PID` family used for frame-time governance (a derivative term reacts
//! to frame-time noise and is deliberately omitted):
//!
//! * **Proportional** — reacts to the current normalized frame-time error.
//! * **Integral** — accumulates persistent error so a steady overshoot is
//!   eventually driven out, with anti-windup clamping.
//! * **Hysteresis** — a deadband around the target suppresses dithering when
//!   the frame time is "close enough", which is what keeps the image from
//!   visibly pulsing between quality levels.
//!
//! Everything is basic arithmetic (`+ - * /`, plus `round`/`clamp`-style
//! branches); no transcendental functions are used, so the controller is
//! bit-deterministic: identical input sequences from an identical start produce
//! identical decisions. The `GPU`-side consumption of these decisions (viewport
//! resize, `LOD` error push, shadow allocator budget) is out of scope and
//! pending the `GPU` backend.

use super::{clamp_unit, lerp, lerp_u32, QualityController, QualityDecision, QualityTarget};

/// Bytes charged per shadow page when converting a memory budget into a page
/// cap. A representative virtual-shadow page footprint; the exact figure only
/// scales the memory-derived ceiling and does not affect determinism.
pub const SHADOW_PAGE_BYTES: u64 = 128 * 1024;

/// The per-knob range the quality level is mapped through.
///
/// Each field pairs the value at the *lowest* quality (`q = 0`) with the value
/// at the *highest* quality (`q = 1`). For most knobs "better" means "larger",
/// but `geometry_error` inverts: higher quality means a *smaller* tolerated
/// screen-space error, so its `best` is below its `worst`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QualityBounds {
    /// Render scale at lowest / highest quality (both in `(0, 1]`).
    pub render_scale: (f32, f32),
    /// Geometry error in pixels at lowest / highest quality. `worst >= best`.
    pub geometry_error_pixels: (f32, f32),
    /// Shadow page budget at lowest / highest quality.
    pub shadow_page_budget: (u32, u32),
    /// `GI` ray scale at lowest / highest quality (both in `(0, 1]`).
    pub gi_ray_scale: (f32, f32),
}

impl QualityBounds {
    /// Maps a quality level `q in [0, 1]` to a concrete decision.
    ///
    /// `q` is clamped into range first (so a stray value cannot escape the
    /// bounds). Each knob is a linear interpolation between its lowest- and
    /// highest-quality endpoints, which makes the whole decision a monotone
    /// function of `q`.
    #[must_use]
    pub fn decision_at(self, q: f32) -> QualityDecision {
        let q = clamp_unit(q);
        QualityDecision {
            render_scale: lerp(self.render_scale.0, self.render_scale.1, q),
            geometry_error_pixels: lerp(
                self.geometry_error_pixels.0,
                self.geometry_error_pixels.1,
                q,
            ),
            shadow_page_budget: lerp_u32(self.shadow_page_budget.0, self.shadow_page_budget.1, q),
            gi_ray_scale: lerp(self.gi_ray_scale.0, self.gi_ray_scale.1, q),
        }
    }
}

impl Default for QualityBounds {
    fn default() -> Self {
        Self {
            // Lowest quality renders at half scale; highest renders natively.
            render_scale: (0.5, 1.0),
            // Lowest quality tolerates a 4 px error; highest holds a quarter px.
            geometry_error_pixels: (4.0, 0.25),
            // Shadow page budget scales 8x between the extremes.
            shadow_page_budget: (1024, 8192),
            // GI ray density scales 4x between the extremes.
            gi_ray_scale: (0.25, 1.0),
        }
    }
}

/// A closed-loop proportional-integral adaptive quality controller.
///
/// Construct with [`Self::new`] (or [`Default`]), then call
/// [`QualityController::update`] once per frame. The controller holds the
/// current quality level and the integral accumulator between calls; it is
/// otherwise a pure function of its inputs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdaptiveQualityController {
    bounds: QualityBounds,
    /// Proportional gain applied to the normalized frame-time error.
    proportional_gain: f32,
    /// Integral gain applied to the accumulated normalized error.
    integral_gain: f32,
    /// Half-width of the deadband around the target, as a fraction of the
    /// target frame time. Errors inside `+/- deadband` are treated as "on
    /// target" and freeze the quality level (hysteresis).
    deadband: f32,
    /// Anti-windup clamp on the magnitude of the integral accumulator.
    integral_limit: f32,
    /// Current quality level in `[0, 1]`.
    quality: f32,
    /// Accumulated normalized error.
    integral: f32,
}

impl AdaptiveQualityController {
    /// Builds a controller with explicit gains and bounds.
    ///
    /// `initial_quality` is clamped into `[0, 1]`; the integral accumulator
    /// starts at zero. Gains are stored as given (a caller supplying negative
    /// gains would invert the loop, which is their responsibility).
    #[must_use]
    pub fn new(
        bounds: QualityBounds,
        proportional_gain: f32,
        integral_gain: f32,
        deadband: f32,
        integral_limit: f32,
        initial_quality: f32,
    ) -> Self {
        Self {
            bounds,
            proportional_gain,
            integral_gain,
            deadband,
            integral_limit,
            quality: clamp_unit(initial_quality),
            integral: 0.0,
        }
    }

    /// The current quality level in `[0, 1]`.
    #[must_use]
    pub fn quality(self) -> f32 {
        self.quality
    }

    /// The bounds this controller maps the quality level through.
    #[must_use]
    pub fn bounds(self) -> QualityBounds {
        self.bounds
    }

    /// Clamps the integral accumulator to `+/- integral_limit` (anti-windup).
    ///
    /// Written with an explicit `NaN`/low branch first so a `NaN` accumulator
    /// (which cannot arise from finite inputs but is guarded defensively)
    /// collapses to the lower bound instead of propagating, and so the form
    /// does not trip the `manual_clamp` lint.
    #[must_use]
    fn clamp_integral(&self, value: f32) -> f32 {
        let lo = -self.integral_limit;
        let hi = self.integral_limit;
        if value.is_nan() || value < lo {
            lo
        } else if value > hi {
            hi
        } else {
            value
        }
    }

    /// Applies the memory budget as a ceiling on the shadow page budget.
    ///
    /// Shadow pages are the dominant resident cost the controller governs, so a
    /// tight memory target caps them regardless of the frame-time-derived
    /// quality level. The cap is monotone in `memory_bytes`.
    fn apply_memory_cap(memory_bytes: u64, decision: &mut QualityDecision) {
        let page_cap = (memory_bytes / SHADOW_PAGE_BYTES).min(u32::MAX as u64) as u32;
        if decision.shadow_page_budget > page_cap {
            decision.shadow_page_budget = page_cap;
        }
    }
}

impl Default for AdaptiveQualityController {
    fn default() -> Self {
        Self::new(QualityBounds::default(), 0.5, 0.1, 0.05, 2.0, 1.0)
    }
}

impl QualityController for AdaptiveQualityController {
    fn update(&mut self, target: QualityTarget, measured_frame_ms: f32) -> QualityDecision {
        // Non-adaptive views run at full quality; reset the accumulator so a
        // later switch back to adaptive starts clean.
        if !target.adaptive {
            self.quality = 1.0;
            self.integral = 0.0;
            let mut decision = self.bounds.decision_at(self.quality);
            Self::apply_memory_cap(target.memory_bytes, &mut decision);
            return decision;
        }

        // A non-positive or non-finite target/measurement gives no usable
        // error signal, so hold the current level rather than reacting to
        // garbage.
        let target_ms = target.frame_time_ms;
        let usable = target_ms.is_finite()
            && target_ms > 0.0
            && measured_frame_ms.is_finite()
            && measured_frame_ms >= 0.0;

        if usable {
            // Positive error == too slow (measured above target) == reduce
            // quality; negative error == headroom == raise quality.
            let error = (measured_frame_ms - target_ms) / target_ms;

            if error.abs() > self.deadband {
                // Outside the deadband: run the PI loop.
                self.integral = self.clamp_integral(self.integral + error);
                let control = self.proportional_gain * error + self.integral_gain * self.integral;
                // Reducing quality when error is positive means subtracting the
                // control signal from the level.
                self.quality = clamp_unit(self.quality - control);
            }
            // Inside the deadband: hysteresis holds both the level and the
            // accumulator, which is what prevents visible quality dithering.
        }

        let mut decision = self.bounds.decision_at(self.quality);
        Self::apply_memory_cap(target.memory_bytes, &mut decision);
        decision
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1e-5
    }

    fn target(ms: f32) -> QualityTarget {
        QualityTarget {
            frame_time_ms: ms,
            memory_bytes: u64::MAX,
            adaptive: true,
        }
    }

    #[test]
    fn mapping_is_monotone_in_quality() {
        let b = QualityBounds::default();
        let low = b.decision_at(0.0);
        let high = b.decision_at(1.0);
        // Better quality: higher render scale, lower geometry error, more
        // shadow pages, denser GI.
        assert!(high.render_scale > low.render_scale);
        assert!(high.geometry_error_pixels < low.geometry_error_pixels);
        assert!(high.shadow_page_budget > low.shadow_page_budget);
        assert!(high.gi_ray_scale > low.gi_ray_scale);
        // Endpoints hit the declared bounds exactly.
        assert!(approx(high.render_scale, 1.0));
        assert!(approx(low.render_scale, 0.5));
    }

    #[test]
    fn quality_clamps_at_bounds() {
        let b = QualityBounds::default();
        let over = b.decision_at(5.0);
        let under = b.decision_at(-5.0);
        assert!(approx(over.render_scale, 1.0));
        assert!(approx(under.render_scale, 0.5));
    }

    #[test]
    fn over_budget_downgrades_quality() {
        let mut c = AdaptiveQualityController::default();
        let start = c.quality();
        // Sustained 2x frame time drives the loop down.
        for _ in 0..8 {
            c.update(target(16.0), 32.0);
        }
        assert!(c.quality() < start, "quality should drop: {}", c.quality());
        let d = c.bounds().decision_at(c.quality());
        assert!(d.render_scale < 1.0);
        assert!(d.geometry_error_pixels > 0.25);
        assert!(d.gi_ray_scale < 1.0);
    }

    #[test]
    fn headroom_recovers_quality() {
        let mut c =
            AdaptiveQualityController::new(QualityBounds::default(), 0.5, 0.1, 0.05, 2.0, 0.2);
        let start = c.quality();
        // Frame time well under target -> raise quality.
        for _ in 0..8 {
            c.update(target(16.0), 4.0);
        }
        assert!(c.quality() > start, "quality should rise: {}", c.quality());
    }

    #[test]
    fn hysteresis_holds_inside_deadband() {
        let mut c = AdaptiveQualityController::default();
        // Within +/- 5% of target: 16.0 * 1.02 = 16.32.
        let first = c.update(target(16.0), 16.32);
        let q_after_first = c.quality();
        for _ in 0..10 {
            let d = c.update(target(16.0), 16.32);
            assert_eq!(d, first);
        }
        assert!(approx(c.quality(), q_after_first));
    }

    #[test]
    fn is_deterministic_across_instances() {
        let seq = [40.0f32, 12.0, 33.0, 16.0, 22.0, 9.0, 18.0, 30.0];
        let mut a = AdaptiveQualityController::default();
        let mut b = AdaptiveQualityController::default();
        for &m in &seq {
            let da = a.update(target(16.0), m);
            let db = b.update(target(16.0), m);
            assert_eq!(da, db);
        }
        assert!(approx(a.quality(), b.quality()));
    }

    #[test]
    fn non_adaptive_pins_full_quality() {
        let mut c =
            AdaptiveQualityController::new(QualityBounds::default(), 0.5, 0.1, 0.05, 2.0, 0.2);
        let t = QualityTarget {
            frame_time_ms: 16.0,
            memory_bytes: u64::MAX,
            adaptive: false,
        };
        let d = c.update(t, 100.0);
        assert!(approx(c.quality(), 1.0));
        assert!(approx(d.render_scale, 1.0));
    }

    #[test]
    fn garbage_measurement_holds_level() {
        let mut c = AdaptiveQualityController::default();
        let before = c.quality();
        c.update(target(16.0), f32::NAN);
        c.update(target(f32::NAN), 16.0);
        c.update(target(0.0), 16.0);
        assert!(approx(c.quality(), before));
    }

    #[test]
    fn memory_budget_caps_shadow_pages() {
        let mut c = AdaptiveQualityController::default();
        // Budget for exactly 4 pages.
        let t = QualityTarget {
            frame_time_ms: 16.0,
            memory_bytes: SHADOW_PAGE_BYTES * 4,
            adaptive: true,
        };
        let d = c.update(t, 16.0);
        assert_eq!(d.shadow_page_budget, 4);
    }

    #[test]
    fn integral_anti_windup_is_bounded() {
        let mut c = AdaptiveQualityController::default();
        // Hammer with a huge sustained overshoot; quality bottoms out and stays
        // clamped rather than diverging.
        for _ in 0..100 {
            c.update(target(16.0), 320.0);
        }
        assert!(approx(c.quality(), 0.0));
        let d = c.bounds().decision_at(c.quality());
        assert!(approx(d.render_scale, 0.5));
    }
}
