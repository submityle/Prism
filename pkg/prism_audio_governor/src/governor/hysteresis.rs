//! Block-boundary hysteresis: deciding when to raise or lower quality.
//!
//! A naive controller that lowered quality the instant load crossed a threshold
//! and raised it the instant load dropped back would "chatter" -- oscillating
//! between tiers every block and producing audible pumping. The classic fix is
//! hysteresis: separate `raise` and `lower` thresholds (a dead band between
//! them) plus a dwell requirement so a decision only fires after the condition
//! has held for several consecutive blocks.
//!
//! This module implements that policy over the smoothed load produced by
//! [`crate::governor::budget`]. High load asks to *lower* quality; low load
//! asks to *raise* it; anything inside the dead band holds the current tier and
//! relaxes both dwell counters.
//!
//! # Real-time contract
//!
//! [`Hysteresis::update`] is branch-light, allocation-free, lock-free, and
//! panic-free. It holds three small integers and three scalars of state.
//!
//! # Determinism
//!
//! State evolves by integer counting and scalar comparison only, so a given
//! input sequence always yields the same transition sequence.
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//!
//! Consumes the load from [`crate::governor::budget::BudgetTracker`] and emits a
//! [`Transition`] consumed by [`crate::governor::QualityGovernor`] to step the
//! quality tier within the limits set by [`crate::governor::power`].

use prism_audio_core::math::Sample;

/// The decision a [`Hysteresis`] controller makes at a block boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum Transition {
    /// Keep the current quality tier.
    Hold,
    /// Headroom is available: step one tier up toward higher quality.
    Raise,
    /// Budget pressure is sustained: step one tier down toward lower quality.
    Lower,
}

impl Transition {
    /// Returns the signed tier delta this transition implies: `+1` for
    /// [`Raise`](Transition::Raise), `-1` for [`Lower`](Transition::Lower),
    /// `0` for [`Hold`](Transition::Hold).
    #[must_use]
    #[inline]
    pub const fn delta(self) -> i32 {
        match self {
            Transition::Hold => 0,
            Transition::Raise => 1,
            Transition::Lower => -1,
        }
    }
}

/// A dual-threshold, dwell-gated hysteresis controller over a scalar load.
///
/// `lower_at` is the load above which the controller wants to drop quality;
/// `raise_at` is the load below which it wants to add quality. The two define a
/// dead band `[raise_at, lower_at]` in which the controller holds. A decision
/// only fires once the relevant condition has held for `hold_blocks`
/// consecutive updates, after which the counter resets and must re-accumulate.
#[derive(Debug, Clone)]
pub struct Hysteresis {
    /// Load at or above which quality should be lowered (upper threshold).
    lower_at: Sample,
    /// Load at or below which quality may be raised (lower threshold).
    raise_at: Sample,
    /// Consecutive qualifying blocks required before a transition fires.
    hold_blocks: u32,
    /// Consecutive blocks the load has sat in the "too hot" region.
    hot_count: u32,
    /// Consecutive blocks the load has sat in the "cool" region.
    cool_count: u32,
}

impl Hysteresis {
    /// Creates a controller with the given dead band and dwell requirement.
    ///
    /// `raise_at` and `lower_at` are sanitised: non-finite values fall back to
    /// sensible defaults and the pair is ordered so `raise_at <= lower_at`,
    /// guaranteeing a non-empty dead band. `hold_blocks` is forced to at least
    /// `1` so a decision always requires at least one qualifying block.
    #[must_use]
    pub fn new(raise_at: Sample, lower_at: Sample, hold_blocks: u32) -> Self {
        let raise = if raise_at.is_finite() { raise_at } else { 0.5 };
        let lower = if lower_at.is_finite() { lower_at } else { 0.9 };
        let (raise, lower) = if raise <= lower {
            (raise, lower)
        } else {
            (lower, raise)
        };
        Self {
            lower_at: lower,
            raise_at: raise,
            hold_blocks: hold_blocks.max(1),
            hot_count: 0,
            cool_count: 0,
        }
    }

    /// Feeds one block's load into the controller and returns its decision.
    ///
    /// When the load is at or above `lower_at` the hot counter advances and the
    /// cool counter clears; when at or below `raise_at` the reverse happens; in
    /// the dead band both counters relax to zero. A counter reaching
    /// `hold_blocks` fires the corresponding transition and resets.
    #[inline]
    pub fn update(&mut self, load: Sample) -> Transition {
        let load = if load.is_finite() { load.max(0.0) } else { 0.0 };

        if load >= self.lower_at {
            self.cool_count = 0;
            self.hot_count = self.hot_count.saturating_add(1);
            if self.hot_count >= self.hold_blocks {
                self.hot_count = 0;
                return Transition::Lower;
            }
            Transition::Hold
        } else if load <= self.raise_at {
            self.hot_count = 0;
            self.cool_count = self.cool_count.saturating_add(1);
            if self.cool_count >= self.hold_blocks {
                self.cool_count = 0;
                return Transition::Raise;
            }
            Transition::Hold
        } else {
            // Inside the dead band: relax both counters, hold the tier.
            self.hot_count = 0;
            self.cool_count = 0;
            Transition::Hold
        }
    }

    /// Returns the upper (lower-quality) threshold.
    #[must_use]
    #[inline]
    pub fn lower_at(&self) -> Sample {
        self.lower_at
    }

    /// Returns the lower (raise-quality) threshold.
    #[must_use]
    #[inline]
    pub fn raise_at(&self) -> Sample {
        self.raise_at
    }

    /// Returns the dwell requirement in blocks.
    #[must_use]
    #[inline]
    pub fn hold_blocks(&self) -> u32 {
        self.hold_blocks
    }

    /// Clears both dwell counters without changing the thresholds.
    #[inline]
    pub fn reset(&mut self) {
        self.hot_count = 0;
        self.cool_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delta_matches_direction() {
        assert_eq!(Transition::Raise.delta(), 1);
        assert_eq!(Transition::Lower.delta(), -1);
        assert_eq!(Transition::Hold.delta(), 0);
    }

    #[test]
    fn ordering_is_normalized() {
        let h = Hysteresis::new(0.9, 0.5, 3);
        assert!(h.raise_at() <= h.lower_at());
        assert!((h.raise_at() - 0.5).abs() < 1e-6);
        assert!((h.lower_at() - 0.9).abs() < 1e-6);
    }

    #[test]
    fn dwell_is_required_before_lowering() {
        let mut h = Hysteresis::new(0.5, 0.9, 3);
        assert_eq!(h.update(1.0), Transition::Hold);
        assert_eq!(h.update(1.0), Transition::Hold);
        assert_eq!(h.update(1.0), Transition::Lower);
    }

    #[test]
    fn dwell_is_required_before_raising() {
        let mut h = Hysteresis::new(0.5, 0.9, 2);
        assert_eq!(h.update(0.1), Transition::Hold);
        assert_eq!(h.update(0.1), Transition::Raise);
    }

    #[test]
    fn dead_band_holds_and_relaxes() {
        let mut h = Hysteresis::new(0.5, 0.9, 2);
        // One hot block then a dead-band block clears the hot counter.
        assert_eq!(h.update(1.0), Transition::Hold);
        assert_eq!(h.update(0.7), Transition::Hold);
        // Need two fresh hot blocks again to fire.
        assert_eq!(h.update(1.0), Transition::Hold);
        assert_eq!(h.update(1.0), Transition::Lower);
    }

    #[test]
    fn opposite_region_clears_counter() {
        let mut h = Hysteresis::new(0.5, 0.9, 2);
        assert_eq!(h.update(1.0), Transition::Hold); // hot=1
        assert_eq!(h.update(0.1), Transition::Hold); // cool=1, hot=0
        assert_eq!(h.update(1.0), Transition::Hold); // hot=1 again, no fire
    }

    #[test]
    fn hold_blocks_is_at_least_one() {
        let mut h = Hysteresis::new(0.5, 0.9, 0);
        assert_eq!(h.hold_blocks(), 1);
        assert_eq!(h.update(1.0), Transition::Lower);
    }

    #[test]
    fn non_finite_load_is_safe() {
        let mut h = Hysteresis::new(0.5, 0.9, 1);
        // NaN -> treated as 0.0 -> cool region -> raise.
        assert_eq!(h.update(Sample::NAN), Transition::Raise);
    }

    #[test]
    fn reset_clears_counters() {
        let mut h = Hysteresis::new(0.5, 0.9, 3);
        h.update(1.0);
        h.update(1.0);
        h.reset();
        assert_eq!(h.update(1.0), Transition::Hold);
    }
}
