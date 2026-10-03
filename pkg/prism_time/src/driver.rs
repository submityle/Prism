//! [`TimeDriver`]: an ECS-agnostic frame-advance pipeline.
//!
//! A frame driver wires the building blocks together without depending on any
//! particular ECS. Each frame you hand it the real wall-clock delta and it:
//! 1. records the real delta into [`FrameStats`] and checks the [`FrameBudget`];
//! 2. gates the delta through the [`FrameStepper`] (frame-step debugging);
//! 3. advances the real clock by the *ungated* real delta (the real clock never
//!    pauses);
//! 4. advances the virtual clock by the gated delta (so pause/step/scale apply);
//! 5. feeds the resulting virtual delta into the fixed accumulator; and
//! 6. points the default clock at [`Virtual`] for the variable-update phase.
//!
//! The fixed-update loop is then driven with [`TimeDriver::expend_fixed`], which
//! points the default clock at [`Fixed`] for each fixed step and back at
//! [`Virtual`] when the steps are drained.

use crate::{Clocks, DefaultSource, Duration, FrameBudget, FrameStats, FrameStepper};

/// A frame report describing what one [`TimeDriver::advance`] produced.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct FrameReport {
    /// The real wall-clock delta for the frame (unaffected by pause/step).
    pub real_delta: Duration,
    /// The virtual delta produced after gating and scaling.
    pub virtual_delta: Duration,
    /// Whether the frame released time through a single frame-step while paused.
    pub stepped: bool,
    /// Whether the real frame delta exceeded the [`FrameBudget`].
    pub over_budget: bool,
}

/// An ECS-agnostic frame-advance pipeline bundling the clocks, the frame-step
/// gate, a stats ring of `N` samples, and a frame budget.
#[derive(Clone, Copy, Debug)]
pub struct TimeDriver<const N: usize> {
    clocks: Clocks,
    stepper: FrameStepper,
    stats: FrameStats<N>,
    budget: FrameBudget,
}

impl<const N: usize> TimeDriver<N> {
    /// Create a driver with the given per-frame [`FrameBudget`] and defaults for
    /// the clocks and frame-stepper.
    #[inline]
    pub fn new(budget: FrameBudget) -> Self {
        Self {
            clocks: Clocks::new(),
            stepper: FrameStepper::new(),
            stats: FrameStats::new(),
            budget,
        }
    }

    /// Create a driver whose budget targets `hz` frames per second.
    ///
    /// # Panics
    /// Panics if `hz` is not strictly positive and finite.
    #[inline]
    pub fn from_hz(hz: f64) -> Self {
        Self::new(FrameBudget::from_hz(hz))
    }

    /// Shared access to the clock bundle.
    #[inline]
    pub fn clocks(&self) -> &Clocks {
        &self.clocks
    }

    /// Mutable access to the clock bundle.
    #[inline]
    pub fn clocks_mut(&mut self) -> &mut Clocks {
        &mut self.clocks
    }

    /// Shared access to the frame-step gate.
    #[inline]
    pub fn stepper(&self) -> &FrameStepper {
        &self.stepper
    }

    /// Mutable access to the frame-step gate (pause, resume, request steps).
    #[inline]
    pub fn stepper_mut(&mut self) -> &mut FrameStepper {
        &mut self.stepper
    }

    /// Shared access to the frame-time stats ring.
    #[inline]
    pub fn stats(&self) -> &FrameStats<N> {
        &self.stats
    }

    /// Shared access to the frame budget.
    #[inline]
    pub fn budget(&self) -> &FrameBudget {
        &self.budget
    }

    /// Mutable access to the frame budget.
    #[inline]
    pub fn budget_mut(&mut self) -> &mut FrameBudget {
        &mut self.budget
    }

    /// Advance one frame from the real wall-clock `real_delta`.
    ///
    /// Records stats, gates through the stepper, advances the real and virtual
    /// clocks, feeds the fixed accumulator, and leaves the default clock
    /// pointing at [`Virtual`]. Returns a [`FrameReport`].
    #[inline]
    pub fn advance(&mut self, real_delta: Duration) -> FrameReport {
        self.stats.record(real_delta);
        let over_budget = self.budget.is_over_budget(real_delta);

        let was_paused = self.stepper.is_paused();
        let gated = self.stepper.next_delta(real_delta);
        let stepped = was_paused && !gated.is_zero();

        // The real clock tracks wall time regardless of pause/step.
        self.clocks.real_mut().update_with_delta(real_delta);
        // Virtual time sees only the gated delta, then applies scale/pause.
        self.clocks.virtual_time_mut().advance_by(gated);
        let virtual_delta = self.clocks.virtual_time().delta();
        // Fixed accumulator is driven by the (scaled) virtual delta.
        self.clocks.fixed_mut().accumulate(virtual_delta);
        // Variable-update phase reads the virtual clock by default.
        self.clocks.set_source(DefaultSource::Virtual);

        FrameReport {
            real_delta,
            virtual_delta,
            stepped,
            over_budget,
        }
    }

    /// Drain one fixed timestep if available, pointing the default clock at the
    /// active phase.
    ///
    /// Returns `true` and points the default clock at [`Fixed`](crate::Fixed)
    /// when a step ran; returns `false` and points it back at [`Virtual`] when
    /// the accumulator is drained. Drive the fixed schedule with
    /// `while driver.expend_fixed() { .. }`.
    #[inline]
    pub fn expend_fixed(&mut self) -> bool {
        if self.clocks.fixed_mut().expend() {
            self.clocks.set_source(DefaultSource::Fixed);
            true
        } else {
            self.clocks.set_source(DefaultSource::Virtual);
            false
        }
    }
}
