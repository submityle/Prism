//! Change-detection primitives: monotonic [`Tick`]s and per-value
//! [`ComponentTicks`] (design §10).
//!
//! Change detection is the backbone of "cost ∝ change": a system records the
//! world [`Tick`] it last ran at (`last_run`) and the tick the current run is
//! stamped with (`this_run`). Every component value carries two ticks — when it
//! was *added* and when it was last *changed*. A value is "visible" to a system
//! as added/changed when its tick lies in the half-open window
//! `(last_run, this_run]`.
//!
//! # Wraparound
//!
//! Ticks are `u32` counters that wrap after ~4 billion increments. Naive
//! `a < b` comparison would misbehave across a wrap, so comparisons are done on
//! the *relative age* `this_run.wrapping_sub(tick)`, clamped to
//! [`Tick::MAX_CHANGE_AGE`]. Periodically clamping very old ticks
//! ([`Tick::check_tick`]) keeps every stored tick within one `MAX_CHANGE_AGE`
//! window of `this_run`, so the relative-age comparison is always unambiguous.
//! This is the standard scheme used by production tick-based ECS kernels.

/// A monotonically increasing world change counter.
///
/// A `Tick` is a logical timestamp, not a wall-clock time. The world bumps its
/// change tick as systems run; component writes stamp the current tick so later
/// systems can tell what changed since they last ran.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct Tick(u32);

impl Tick {
    /// The tick used for "has never run" / initial state. Chosen as `0` so a
    /// freshly default-constructed [`Tick`] reads as the oldest possible time.
    pub const ZERO: Tick = Tick(0);

    /// The largest representable tick value.
    pub const MAX: Tick = Tick(u32::MAX);

    /// Maximum age (in ticks) that change detection can reliably distinguish.
    ///
    /// Any stored tick older than this relative to `this_run` is clamped by
    /// [`Tick::check_tick`] so the wrapping comparison never aliases a very old
    /// tick onto a very new one. Mirrors the conservative bound used by
    /// tick-based ECS designs: `u32::MAX - (2 * CHECK_TICK_THRESHOLD - 1)`.
    pub const MAX_CHANGE_AGE: u32 = u32::MAX - (2 * Self::CHECK_TICK_THRESHOLD - 1);

    /// Suggested interval between [`check_tick`](Tick::check_tick) passes.
    pub const CHECK_TICK_THRESHOLD: u32 = 518_400_000;

    /// Construct a tick from a raw counter value.
    #[inline]
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    /// The raw counter value.
    #[inline]
    pub const fn get(self) -> u32 {
        self.0
    }

    /// Overwrite the raw counter value.
    #[inline]
    pub fn set(&mut self, value: u32) {
        self.0 = value;
    }

    /// The wrapping age of `self` measured backwards from `other`
    /// (`other - self`), i.e. how many ticks elapsed from `self` up to `other`.
    #[inline]
    pub const fn age_since(self, other: Tick) -> u32 {
        other.0.wrapping_sub(self.0)
    }

    /// Whether a value stamped at `self` should be considered new to a system
    /// whose window is `(last_run, this_run]`.
    ///
    /// Comparisons use clamped relative ages so the result is correct across a
    /// `u32` wrap, provided ticks are periodically clamped via
    /// [`Tick::check_tick`].
    #[inline]
    pub fn is_newer_than(self, last_run: Tick, this_run: Tick) -> bool {
        let ticks_since_insert = self.age_since(this_run).min(Self::MAX_CHANGE_AGE);
        let ticks_since_system = last_run.age_since(this_run).min(Self::MAX_CHANGE_AGE);
        ticks_since_system > ticks_since_insert
    }

    /// Clamp `self` if it has aged past [`MAX_CHANGE_AGE`](Tick::MAX_CHANGE_AGE)
    /// relative to `this_run`, returning `true` if it was clamped.
    ///
    /// Run periodically over every stored tick so no tick can wrap all the way
    /// around and masquerade as a recent one.
    #[inline]
    pub fn check_tick(&mut self, this_run: Tick) -> bool {
        let age = self.age_since(this_run);
        if age > Self::MAX_CHANGE_AGE {
            self.0 = this_run.0.wrapping_sub(Self::MAX_CHANGE_AGE);
            true
        } else {
            false
        }
    }
}

/// The pair of ticks recorded for a single component value: when it was added
/// and when it was last changed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ComponentTicks {
    /// The tick at which this value was first inserted onto its entity.
    pub added: Tick,
    /// The tick at which this value was most recently written.
    pub changed: Tick,
}

impl ComponentTicks {
    /// New ticks for a value just inserted at `change_tick` (added == changed).
    #[inline]
    pub const fn new(change_tick: Tick) -> Self {
        Self {
            added: change_tick,
            changed: change_tick,
        }
    }

    /// Whether the value was added within `(last_run, this_run]`.
    #[inline]
    pub fn is_added(&self, last_run: Tick, this_run: Tick) -> bool {
        self.added.is_newer_than(last_run, this_run)
    }

    /// Whether the value was changed (or added) within `(last_run, this_run]`.
    #[inline]
    pub fn is_changed(&self, last_run: Tick, this_run: Tick) -> bool {
        self.changed.is_newer_than(last_run, this_run)
    }

    /// Record a write at `change_tick` by advancing the changed tick.
    #[inline]
    pub fn set_changed(&mut self, change_tick: Tick) {
        self.changed = change_tick;
    }

    /// Clamp both ticks against `this_run` (see [`Tick::check_tick`]).
    #[inline]
    pub fn check_ticks(&mut self, this_run: Tick) {
        self.added.check_tick(this_run);
        self.changed.check_tick(this_run);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_zero() {
        assert_eq!(Tick::default(), Tick::ZERO);
        assert_eq!(Tick::default().get(), 0);
    }

    #[test]
    fn newer_than_basic_window() {
        // System last ran at 2, this run is 5. Values stamped in (2, 5] are new.
        let last = Tick::new(2);
        let now = Tick::new(5);
        assert!(!Tick::new(2).is_newer_than(last, now)); // boundary: not newer
        assert!(Tick::new(3).is_newer_than(last, now));
        assert!(Tick::new(5).is_newer_than(last, now));
        assert!(!Tick::new(1).is_newer_than(last, now)); // older than last_run
    }

    #[test]
    fn newer_than_handles_wraparound() {
        // last_run just below the wrap, this_run just past it.
        let last = Tick::new(u32::MAX - 1);
        let now = Tick::new(2); // wrapped: age(last->now) == 3
        assert!(Tick::new(0).is_newer_than(last, now)); // stamped after wrap
        assert!(Tick::new(1).is_newer_than(last, now));
        assert!(Tick::new(2).is_newer_than(last, now));
        assert!(!Tick::new(u32::MAX - 1).is_newer_than(last, now)); // == last
        assert!(!Tick::new(u32::MAX - 5).is_newer_than(last, now)); // before last
    }

    #[test]
    fn check_tick_clamps_overly_old() {
        let now = Tick::new(Tick::MAX_CHANGE_AGE + 100);
        let mut old = Tick::new(1); // age ~ MAX_CHANGE_AGE + 99 > MAX_CHANGE_AGE
        assert!(old.check_tick(now));
        assert_eq!(old.age_since(now), Tick::MAX_CHANGE_AGE);
        // A fresh tick within the window is untouched.
        let mut fresh = Tick::new(Tick::MAX_CHANGE_AGE + 50);
        assert!(!fresh.check_tick(now));
        assert_eq!(fresh.get(), Tick::MAX_CHANGE_AGE + 50);
    }

    #[test]
    fn component_ticks_added_then_changed() {
        let mut t = ComponentTicks::new(Tick::new(10));
        assert!(t.is_added(Tick::new(5), Tick::new(10)));
        assert!(t.is_changed(Tick::new(5), Tick::new(10)));
        // A later system (last_run = 10) sees neither until a new write.
        assert!(!t.is_added(Tick::new(10), Tick::new(12)));
        assert!(!t.is_changed(Tick::new(10), Tick::new(12)));
        // Writing at 12 shows as changed (but not added) to that system.
        t.set_changed(Tick::new(12));
        assert!(!t.is_added(Tick::new(10), Tick::new(12)));
        assert!(t.is_changed(Tick::new(10), Tick::new(12)));
    }
}
