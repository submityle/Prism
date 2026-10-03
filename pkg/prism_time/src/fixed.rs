//! [`Time<Fixed>`]: a deterministic fixed-timestep accumulator.
//!
//! You feed the virtual delta each frame via [`Time::<Fixed>::accumulate`],
//! then drain whole timesteps with [`Time::<Fixed>::expend`] (true while a step
//! is available). The leftover is [`Time::<Fixed>::overstep`], whose fraction of
//! a timestep is the interpolation alpha. The accumulator is capped at
//! `max_substeps` timesteps so a frame hitch cannot trigger an unbounded run of
//! fixed steps (the spiral of death).

use crate::{Duration, Time, TimeKind};

/// Context for [`Time<Fixed>`]: the fixed timestep and its accumulator.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fixed {
    /// Length of one fixed step.
    timestep: Duration,
    /// Accumulated time not yet consumed by a fixed step.
    overstep: Duration,
    /// Maximum number of timesteps the accumulator may hold (death-spiral
    /// guard). Excess accumulated time is dropped.
    max_substeps: u32,
}

impl Fixed {
    /// Default fixed timestep: `1/64 s` (`15.625 ms`).
    pub const DEFAULT_TIMESTEP: Duration = Duration::from_micros(15_625);
    /// Default cap on accumulated timesteps per frame.
    pub const DEFAULT_MAX_SUBSTEPS: u32 = 8;
}

impl Default for Fixed {
    #[inline]
    fn default() -> Self {
        Self {
            timestep: Self::DEFAULT_TIMESTEP,
            overstep: Duration::ZERO,
            max_substeps: Self::DEFAULT_MAX_SUBSTEPS,
        }
    }
}

impl TimeKind for Fixed {}

impl Time<Fixed> {
    /// Create a fixed clock with the default `1/64 s` timestep.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a fixed clock from a timestep duration.
    ///
    /// # Panics
    /// Panics if `timestep` is zero (a zero step would loop forever).
    #[inline]
    pub fn from_duration(timestep: Duration) -> Self {
        assert!(!timestep.is_zero(), "fixed timestep must be non-zero");
        let mut time = Self::default();
        time.context.timestep = timestep;
        time
    }

    /// Create a fixed clock from a rate in hertz (steps per second).
    ///
    /// # Panics
    /// Panics if `hz` is not strictly positive and finite.
    #[inline]
    pub fn from_hz(hz: f64) -> Self {
        assert!(hz > 0.0 && hz.is_finite(), "fixed rate must be positive");
        Self::from_duration(Duration::from_secs_f64(1.0 / hz))
    }

    /// The fixed timestep.
    #[inline]
    pub fn timestep(&self) -> Duration {
        self.context.timestep
    }

    /// Set the fixed timestep.
    ///
    /// # Panics
    /// Panics if `timestep` is zero.
    #[inline]
    pub fn set_timestep(&mut self, timestep: Duration) {
        assert!(!timestep.is_zero(), "fixed timestep must be non-zero");
        self.context.timestep = timestep;
    }

    /// Set the fixed timestep from a rate in hertz.
    ///
    /// # Panics
    /// Panics if `hz` is not strictly positive and finite.
    #[inline]
    pub fn set_timestep_hz(&mut self, hz: f64) {
        assert!(hz > 0.0 && hz.is_finite(), "fixed rate must be positive");
        self.set_timestep(Duration::from_secs_f64(1.0 / hz));
    }

    /// The max-substeps cap applied to the accumulator.
    #[inline]
    pub fn max_substeps(&self) -> u32 {
        self.context.max_substeps
    }

    /// Set the max-substeps cap. A value of `0` drops all accumulated time.
    #[inline]
    pub fn set_max_substeps(&mut self, max_substeps: u32) {
        self.context.max_substeps = max_substeps;
    }

    /// Feed a virtual delta into the accumulator.
    ///
    /// The accumulator is capped at `max_substeps` timesteps so later
    /// [`expend`](Self::expend) calls run a bounded number of steps.
    #[inline]
    pub fn accumulate(&mut self, delta: Duration) {
        let ctx = &mut self.context;
        ctx.overstep = ctx.overstep.saturating_add(delta);
        let cap = ctx.timestep.saturating_mul(ctx.max_substeps);
        if ctx.overstep > cap {
            ctx.overstep = cap;
        }
    }

    /// Consume one timestep if the accumulator holds at least one.
    ///
    /// Returns `true` and advances the clock by one timestep when a step was
    /// available, otherwise `false`. Drive the fixed schedule with
    /// `while time.expend() { .. }`.
    #[inline]
    pub fn expend(&mut self) -> bool {
        let timestep = self.context.timestep;
        if let Some(rest) = self.context.overstep.checked_sub(timestep) {
            self.context.overstep = rest;
            self.advance_generic(timestep);
            true
        } else {
            false
        }
    }

    /// Drain every available timestep, returning how many ran this frame.
    ///
    /// Convenience over looping [`expend`](Self::expend); bounded by
    /// `max_substeps` because [`accumulate`](Self::accumulate) caps the
    /// accumulator.
    #[inline]
    pub fn expend_all(&mut self) -> u32 {
        let mut steps = 0;
        while self.expend() {
            steps += 1;
        }
        steps
    }

    /// Accumulated time not yet consumed by a fixed step.
    #[inline]
    pub fn overstep(&self) -> Duration {
        self.context.overstep
    }

    /// Overstep as a fraction of one timestep (`[0, 1)` after draining); the
    /// interpolation alpha for the presentation layer (`f32`).
    #[inline]
    pub fn overstep_fraction(&self) -> f32 {
        self.context.overstep.as_secs_f32() / self.context.timestep.as_secs_f32()
    }

    /// Overstep fraction as `f64`.
    #[inline]
    pub fn overstep_fraction_f64(&self) -> f64 {
        self.context.overstep.as_secs_f64() / self.context.timestep.as_secs_f64()
    }
}
