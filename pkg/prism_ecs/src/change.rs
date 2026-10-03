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

// ---------------------------------------------------------------------------
// Ref / Mut — change-detecting component borrows
// ---------------------------------------------------------------------------

/// A shared borrow of a component value together with its change-detection
/// metadata (design §10).
///
/// Yielded by a `Ref<T>` query term. Derefs to `&T`; additionally reports
/// [`is_added`](Ref::is_added) / [`is_changed`](Ref::is_changed) relative to the
/// querying system's half-open observer window `(last_run, this_run]`.
pub struct Ref<'w, T: ?Sized> {
    value: &'w T,
    added: Tick,
    changed: Tick,
    last_run: Tick,
    this_run: Tick,
}

impl<'w, T: ?Sized> Ref<'w, T> {
    /// Wrap a shared borrow with its ticks and the observer window.
    #[inline]
    pub(crate) fn new(
        value: &'w T,
        added: Tick,
        changed: Tick,
        last_run: Tick,
        this_run: Tick,
    ) -> Self {
        Self {
            value,
            added,
            changed,
            last_run,
            this_run,
        }
    }

    /// Whether the value was added within the observer window
    /// `(last_run, this_run]`.
    #[inline]
    pub fn is_added(&self) -> bool {
        self.added.is_newer_than(self.last_run, self.this_run)
    }

    /// Whether the value was changed (or added) within `(last_run, this_run]`.
    #[inline]
    pub fn is_changed(&self) -> bool {
        self.changed.is_newer_than(self.last_run, self.this_run)
    }

    /// The tick at which the value was first added to its entity.
    #[inline]
    pub fn added_tick(&self) -> Tick {
        self.added
    }

    /// The tick at which the value was most recently changed.
    #[inline]
    pub fn changed_tick(&self) -> Tick {
        self.changed
    }

    /// Consume the wrapper, returning the underlying shared reference.
    #[inline]
    pub fn into_inner(self) -> &'w T {
        self.value
    }
}

impl<T: ?Sized> core::ops::Deref for Ref<'_, T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        self.value
    }
}

/// An exclusive borrow of a component value that **bumps its changed tick** on
/// mutable access (design §10).
///
/// Yielded by a `&mut T` or `Option<&mut T>` query term. Derefs to `&T`
/// immutably and to `&mut T` mutably; any [`DerefMut`](core::ops::DerefMut) (or
/// [`into_inner`](Mut::into_inner)) records a write at `this_run`, which is what
/// later makes [`Changed<T>`](crate::query::Changed) fire. An equal-value guard
/// can avoid the bump via [`bypass_change_detection`](Mut::bypass_change_detection).
pub struct Mut<'w, T: ?Sized> {
    value: &'w mut T,
    changed: &'w mut Tick,
    added: Tick,
    last_run: Tick,
    this_run: Tick,
}

impl<'w, T: ?Sized> Mut<'w, T> {
    /// Wrap an exclusive borrow with a mutable handle to its changed tick, the
    /// added tick, and the observer window.
    #[inline]
    pub(crate) fn new(
        value: &'w mut T,
        changed: &'w mut Tick,
        added: Tick,
        last_run: Tick,
        this_run: Tick,
    ) -> Self {
        Self {
            value,
            changed,
            added,
            last_run,
            this_run,
        }
    }

    /// Whether the value was added within the observer window
    /// `(last_run, this_run]`.
    #[inline]
    pub fn is_added(&self) -> bool {
        self.added.is_newer_than(self.last_run, self.this_run)
    }

    /// Whether the value was changed (or added) within `(last_run, this_run]`.
    ///
    /// Reflects writes recorded *before* this call; a subsequent `DerefMut` in
    /// the same run will of course make it return `true` afterwards.
    #[inline]
    pub fn is_changed(&self) -> bool {
        self.changed.is_newer_than(self.last_run, self.this_run)
    }

    /// Record a write at `this_run` without going through `DerefMut`.
    #[inline]
    pub fn set_changed(&mut self) {
        *self.changed = self.this_run;
    }

    /// Access the value mutably **without** bumping the changed tick.
    ///
    /// Use when a write is known to be a no-op (an equal-value guard, design
    /// §10) to avoid spurious change propagation.
    #[inline]
    pub fn bypass_change_detection(&mut self) -> &mut T {
        self.value
    }

    /// Consume the wrapper, recording a write and returning the exclusive
    /// reference bound to `'w`.
    #[inline]
    pub fn into_inner(self) -> &'w mut T {
        *self.changed = self.this_run;
        self.value
    }
}

impl<T: ?Sized> core::ops::Deref for Mut<'_, T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        self.value
    }
}

impl<T: ?Sized> core::ops::DerefMut for Mut<'_, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        *self.changed = self.this_run;
        self.value
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
