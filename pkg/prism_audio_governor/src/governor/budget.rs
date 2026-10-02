//! CPU-budget telemetry: converting per-block render cost into a smoothed load.
//!
//! The quality governor runs a closed loop around a single scalar: how much of
//! the per-block CPU budget the audio render actually consumed. The telemetry
//! ring (design section 21) reports the wall-clock cost of each rendered block;
//! this module turns a raw `render_time / budget_time` ratio into a stable,
//! exponentially smoothed `load` that the governor compares against hysteresis
//! thresholds at block boundaries. Smoothing is what keeps a single spiky block
//! from flipping the whole quality ladder.
//!
//! A `load` of `1.0` means the render used exactly its budget; below `1.0` is
//! headroom, above `1.0` is an overrun that risks a buffer underflow.
//!
//! # Real-time contract
//!
//! [`BudgetTracker::observe`] performs no allocation, takes no locks, and cannot
//! panic; it sanitises non-finite and negative input to a safe value. It is
//! intended to be called once per block boundary off the hot DSP path, but is
//! cheap enough to be harmless anywhere.
//!
//! # Determinism
//!
//! The only state is two scalars updated by a first-order recursion, so a given
//! sequence of inputs always yields the same `load`. No transcendental math is
//! involved here, so the result is bit-reproducible across targets.
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//!
//! Feeds the smoothed load into [`crate::governor::hysteresis`], which decides
//! block-boundary quality transitions, and into
//! [`crate::governor::QualityGovernor`], which owns the closed loop. The raw
//! input ratio is produced upstream by the telemetry ring (design section 21).

use prism_audio_core::math::Sample;

/// Converts a raw render cost and its budget into a dimensionless load ratio.
///
/// Returns `render_seconds / budget_seconds`, i.e. the fraction of the budget
/// consumed. A non-positive or non-finite budget yields `0.0` (treated as "no
/// measurable pressure") and a non-finite render time is treated as zero cost,
/// so the governor never acts on garbage telemetry.
///
/// # Examples
///
/// ```
/// # use prism_audio_governor::governor::budget::cost_ratio;
/// // Used 3 ms of a 6 ms budget -> half the budget.
/// assert!((cost_ratio(0.003, 0.006) - 0.5).abs() < 1e-6);
/// // A zero budget degrades gracefully to no pressure.
/// assert_eq!(cost_ratio(0.003, 0.0), 0.0);
/// ```
#[inline]
#[must_use]
pub fn cost_ratio(render_seconds: Sample, budget_seconds: Sample) -> Sample {
    if !budget_seconds.is_finite() || budget_seconds <= 0.0 {
        return 0.0;
    }
    let render = if render_seconds.is_finite() && render_seconds > 0.0 {
        render_seconds
    } else {
        0.0
    };
    render / budget_seconds
}

/// Exponentially smoothed CPU-budget load, the governor's single feedback input.
///
/// Each observed per-block ratio is folded into a running estimate with a
/// configurable smoothing coefficient, so transient spikes are attenuated while
/// sustained pressure is tracked. The tracked value is clamped to a sane upper
/// bound so a pathological overrun cannot drive the state to infinity.
#[derive(Debug, Clone)]
pub struct BudgetTracker {
    /// Smoothing coefficient in `[0, 1)`: the weight given to the *previous*
    /// estimate. `0.0` means no smoothing (follow the latest block exactly),
    /// values near `1.0` mean very heavy smoothing.
    smoothing: Sample,
    /// Current smoothed load estimate.
    load: Sample,
    /// Most recent raw (unsmoothed) ratio, retained for observability.
    last_raw: Sample,
}

/// Hard ceiling applied to the smoothed load so one catastrophic overrun cannot
/// saturate the state to a non-finite value. Ten times budget is already a deep
/// overload; the governor will be pinned at its lowest tier well before this.
pub const MAX_TRACKED_LOAD: Sample = 10.0;

impl BudgetTracker {
    /// Creates a tracker with the given smoothing coefficient, starting from an
    /// idle load of `0.0`.
    ///
    /// `smoothing` is clamped to `[0, 0.999]`; it is the weight retained from
    /// the previous estimate on each [`observe`](BudgetTracker::observe) call.
    #[must_use]
    pub fn new(smoothing: Sample) -> Self {
        Self {
            smoothing: sanitize_smoothing(smoothing),
            load: 0.0,
            last_raw: 0.0,
        }
    }

    /// Folds one block's cost ratio into the smoothed estimate and returns the
    /// updated load.
    ///
    /// The input is sanitised: non-finite values are treated as `0.0` and
    /// negative values are clamped to `0.0`. The result is clamped to
    /// `[0, MAX_TRACKED_LOAD]`.
    #[inline]
    pub fn observe(&mut self, block_cost_ratio: Sample) -> Sample {
        let raw = if block_cost_ratio.is_finite() && block_cost_ratio > 0.0 {
            block_cost_ratio
        } else {
            0.0
        };
        self.last_raw = raw;
        let a = self.smoothing;
        let next = a * self.load + (1.0 - a) * raw;
        self.load = next.clamp(0.0, MAX_TRACKED_LOAD);
        self.load
    }

    /// Returns the current smoothed load without observing a new block.
    #[must_use]
    #[inline]
    pub fn load(&self) -> Sample {
        self.load
    }

    /// Returns the most recent raw (unsmoothed) ratio passed to
    /// [`observe`](BudgetTracker::observe).
    #[must_use]
    #[inline]
    pub fn last_raw(&self) -> Sample {
        self.last_raw
    }

    /// Returns the configured (already sanitised) smoothing coefficient.
    #[must_use]
    #[inline]
    pub fn smoothing(&self) -> Sample {
        self.smoothing
    }

    /// Resets the smoothed and raw state back to idle.
    #[inline]
    pub fn reset(&mut self) {
        self.load = 0.0;
        self.last_raw = 0.0;
    }
}

/// Clamps a smoothing coefficient into the usable `[0, 0.999]` range, mapping
/// non-finite input to `0.0`.
#[inline]
fn sanitize_smoothing(smoothing: Sample) -> Sample {
    if smoothing.is_finite() {
        smoothing.clamp(0.0, 0.999)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: Sample = 1e-6;

    #[test]
    fn cost_ratio_divides() {
        assert!((cost_ratio(0.003, 0.006) - 0.5).abs() < EPS);
        assert!((cost_ratio(0.006, 0.006) - 1.0).abs() < EPS);
    }

    #[test]
    fn cost_ratio_handles_degenerate_budget() {
        assert!(cost_ratio(0.003, 0.0).abs() < EPS);
        assert!(cost_ratio(0.003, -1.0).abs() < EPS);
        assert!(cost_ratio(0.003, Sample::NAN).abs() < EPS);
    }

    #[test]
    fn cost_ratio_sanitizes_render_time() {
        assert!(cost_ratio(Sample::NAN, 0.006).abs() < EPS);
        assert!(cost_ratio(-1.0, 0.006).abs() < EPS);
    }

    #[test]
    fn no_smoothing_follows_input() {
        let mut t = BudgetTracker::new(0.0);
        assert!((t.observe(0.7) - 0.7).abs() < EPS);
        assert!((t.observe(0.2) - 0.2).abs() < EPS);
    }

    #[test]
    fn smoothing_attenuates_single_spike() {
        let mut t = BudgetTracker::new(0.9);
        // One spike to 2.0 from idle should move only a little.
        let after = t.observe(2.0);
        assert!(after < 0.3, "after={after}");
        assert!(after > 0.0);
    }

    #[test]
    fn sustained_load_converges() {
        let mut t = BudgetTracker::new(0.8);
        let mut load = 0.0;
        for _ in 0..200 {
            load = t.observe(0.9);
        }
        assert!((load - 0.9).abs() < 1e-3, "load={load}");
    }

    #[test]
    fn load_is_clamped_to_ceiling() {
        let mut t = BudgetTracker::new(0.0);
        let load = t.observe(1.0e9);
        assert!((load - MAX_TRACKED_LOAD).abs() < EPS);
    }

    #[test]
    fn reset_returns_to_idle() {
        let mut t = BudgetTracker::new(0.5);
        t.observe(0.9);
        t.reset();
        assert!(t.load().abs() < EPS);
        assert!(t.last_raw().abs() < EPS);
    }

    #[test]
    fn smoothing_coefficient_is_clamped() {
        assert!((BudgetTracker::new(5.0).smoothing() - 0.999).abs() < EPS);
        assert!(BudgetTracker::new(-1.0).smoothing().abs() < EPS);
        assert!(BudgetTracker::new(Sample::NAN).smoothing().abs() < EPS);
    }
}
