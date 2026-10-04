//! [`WorldTimeDomain`]: an isolated per-world time context, and [`WorldSet`]:
//! a bundle of worlds advanced from one shared real delta.
//!
//! Each world owns its own `scale`, `pause` flag, and deterministic
//! [`TickClock`](crate::TickClock) accumulator. Advancing one world touches
//! only that world's state, so a paused editor-preview world and a running main
//! world coexist without interference, and each world's `scale` is independent.
//! A world can emit a deterministic audit digest of its key state for
//! double-run comparison (see the [`audit`](crate::multiworld) tooling).

use super::audit::StateHasher;
use alloc::vec::Vec;
use crate::{Duration, RationalStep, TickClock};

/// An independent time context for one World.
///
/// The *effective* scale is the requested scale, or `0.0` while paused. On
/// [`advance`](Self::advance) the incoming real delta is scaled by it, fed into
/// the world's own deterministic accumulator, and drained into integer ticks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldTimeDomain {
    /// This world's deterministic fixed-step accumulator and tick counter.
    clock: TickClock,
    /// Requested local time scale (`>= 0`; non-finite sanitized to `1.0`).
    scale: f64,
    /// Whether this world is frozen (effective scale `0.0`).
    paused: bool,
}

impl WorldTimeDomain {
    /// A world with the given exact step, scale `1.0`, not paused.
    #[inline]
    #[must_use]
    pub const fn new(step: RationalStep) -> Self {
        Self {
            clock: TickClock::new(step),
            scale: 1.0,
            paused: false,
        }
    }

    /// A world stepping at `hz` ticks per second, scale `1.0`, not paused.
    ///
    /// # Panics
    /// Panics if `hz` is zero.
    #[inline]
    #[must_use]
    pub fn from_hz(hz: u64) -> Self {
        Self::new(RationalStep::from_hz(hz))
    }

    /// Builder: set the requested time scale (sanitized like
    /// [`set_scale`](Self::set_scale)).
    #[inline]
    #[must_use]
    pub fn with_scale(mut self, scale: f64) -> Self {
        self.set_scale(scale);
        self
    }

    /// Builder: start this world paused.
    #[inline]
    #[must_use]
    pub fn paused(mut self) -> Self {
        self.paused = true;
        self
    }

    /// The requested scale, ignoring pause.
    #[inline]
    #[must_use]
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// Set the requested scale. Clamped to `>= 0`; non-finite values are
    /// ignored (the previous scale is kept).
    #[inline]
    pub fn set_scale(&mut self, scale: f64) {
        if scale.is_finite() {
            self.scale = scale.max(0.0);
        }
    }

    /// The effective scale actually applied: the requested scale, or `0.0`
    /// while paused.
    #[inline]
    #[must_use]
    pub fn effective_scale(&self) -> f64 {
        if self.paused {
            0.0
        } else {
            self.scale
        }
    }

    /// Pause this world (effective scale becomes `0.0`; requested scale kept).
    #[inline]
    pub fn pause(&mut self) {
        self.paused = true;
    }

    /// Unpause this world, restoring the requested scale.
    #[inline]
    pub fn unpause(&mut self) {
        self.paused = false;
    }

    /// Whether this world is paused.
    #[inline]
    #[must_use]
    pub fn is_paused(&self) -> bool {
        self.paused
    }

    /// Advance this world by a shared real delta, applying its own scale/pause.
    ///
    /// Returns how many fixed ticks ran this frame (bounded by the clock's
    /// `max_substeps` guard). A paused or zero-scale world runs zero ticks and
    /// its accumulator is untouched.
    #[inline]
    pub fn advance(&mut self, real_delta: Duration) -> u64 {
        let scaled = self.scale_delta(real_delta);
        self.clock.accumulate(scaled);
        self.clock.expend_all()
    }

    /// Scale a real delta by this world's effective scale.
    #[inline]
    #[must_use]
    fn scale_delta(&self, real_delta: Duration) -> Duration {
        let s = self.effective_scale();
        if s == 1.0 {
            real_delta
        } else if s == 0.0 {
            Duration::ZERO
        } else {
            real_delta.mul_f64(s)
        }
    }

    /// This world's current integer tick count (its authoritative clock).
    #[inline]
    #[must_use]
    pub fn tick(&self) -> u64 {
        self.clock.tick()
    }

    /// Leftover accumulator in sub-units of `1/den` nanoseconds.
    #[inline]
    #[must_use]
    pub fn overstep_subunits(&self) -> u128 {
        self.clock.overstep_subunits()
    }

    /// Interpolation alpha in `[0, 1)` for this world's presentation layer.
    #[inline]
    #[must_use]
    pub fn overstep_fraction(&self) -> f32 {
        self.clock.overstep_fraction()
    }

    /// Elapsed time derived from the integer tick count.
    #[inline]
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.clock.elapsed()
    }

    /// The exact rational step this world ticks at.
    #[inline]
    #[must_use]
    pub fn step(&self) -> RationalStep {
        self.clock.step()
    }

    /// Borrow the backing deterministic clock.
    #[inline]
    #[must_use]
    pub fn clock(&self) -> &TickClock {
        &self.clock
    }

    /// Reset this world's clock to tick 0 (keeping step, scale and pause).
    #[inline]
    pub fn reset(&mut self) {
        self.clock.reset();
    }

    /// Fold this world's key deterministic state into `hasher` in a fixed
    /// order (tick, accumulator, step num/den, scale bits, pause flag).
    #[inline]
    pub fn hash_state(&self, hasher: &mut StateHasher) {
        let step = self.clock.step();
        hasher.write_u64(self.clock.tick());
        hasher.write_u128(self.clock.overstep_subunits());
        hasher.write_u64(step.nanos_num());
        hasher.write_u64(step.nanos_den());
        hasher.write_f64_bits(self.scale);
        hasher.write_u8(u8::from(self.paused));
    }

    /// A standalone audit digest of this world's key state.
    #[inline]
    #[must_use]
    pub fn audit_hash(&self) -> u64 {
        let mut hasher = StateHasher::new();
        self.hash_state(&mut hasher);
        hasher.finish()
    }
}

/// A bundle of independent [`WorldTimeDomain`]s advanced from one shared real
/// delta. Each world applies its own scale/pause, so they stay isolated.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct WorldSet {
    /// The worlds, in a stable order (the index is the world id).
    worlds: Vec<WorldTimeDomain>,
}

impl WorldSet {
    /// An empty set.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            worlds: Vec::new(),
        }
    }

    /// Add a world and return its index (its stable world id).
    #[inline]
    pub fn push(&mut self, world: WorldTimeDomain) -> usize {
        let id = self.worlds.len();
        self.worlds.push(world);
        id
    }

    /// Number of worlds.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.worlds.len()
    }

    /// Whether the set has no worlds.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.worlds.is_empty()
    }

    /// Borrow the world at `index`.
    #[inline]
    #[must_use]
    pub fn world(&self, index: usize) -> Option<&WorldTimeDomain> {
        self.worlds.get(index)
    }

    /// Mutably borrow the world at `index` (e.g. to pause or re-scale it).
    #[inline]
    pub fn world_mut(&mut self, index: usize) -> Option<&mut WorldTimeDomain> {
        self.worlds.get_mut(index)
    }

    /// All worlds in id order.
    #[inline]
    #[must_use]
    pub fn worlds(&self) -> &[WorldTimeDomain] {
        &self.worlds
    }

    /// Advance every world by the same real delta; each applies its own
    /// scale/pause independently.
    #[inline]
    pub fn advance_all(&mut self, real_delta: Duration) {
        for world in &mut self.worlds {
            world.advance(real_delta);
        }
    }

    /// A single audit digest over every world's key state, in id order. Feed
    /// this into an [`AuditTrail`](crate::multiworld::AuditTrail) per frame.
    #[inline]
    #[must_use]
    pub fn audit_hash(&self) -> u64 {
        let mut hasher = StateHasher::new();
        hasher.write_u64(self.worlds.len() as u64);
        for world in &self.worlds {
            world.hash_state(&mut hasher);
        }
        hasher.finish()
    }
}
