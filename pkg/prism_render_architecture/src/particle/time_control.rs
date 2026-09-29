//! Time-dilation, pause, and fixed-step *policy* contracts for the particle
//! subsystem (design §25, §26).
//!
//! Production `VFX` stacks let an artist run one emitter in slow-motion while
//! the rest of the scene runs at full speed: Unreal `Niagara` exposes a global
//! and a per-emitter *time dilation*, and `Frostbite`'s FX stack drives spawn
//! and aging from the same scaled clock. This module is the `CPU`-verifiable
//! contract layer for exactly that behavior. It does **not** re-implement the
//! integrator — the actual motion advance lives in [`super::simulation`] and
//! the substep *schedule* math lives in [`super::stability`]. Here we only turn
//! a raw frame `dt` plus a set of scales into the four derived quantities every
//! caller needs:
//!
//! 1. the **scaled `dt`** an emitter is advanced by this frame,
//! 2. the **fixed-step count** a `Fixed` policy should run (with a
//!    spiral-of-death clamp),
//! 3. the **substep count** a `CFL`-style stability limit implies for that
//!    scaled `dt`, and
//! 4. the **scaled spawn rate and age delta** so emission and over-life aging
//!    track the same dilated clock as motion.
//!
//! # No transcendental math, no float equality
//! Every routine uses only multiply, divide, comparison, and `floor` — no
//! `sin`/`exp`/`ceil`. Float-to-integer casts rely on Rust's saturating
//! `as u32` conversion so an absurd `dt` can never wrap. Floating-point values
//! are **never** compared with `==` / `!=`; a single tolerance
//! [`CMP_EPS`](self) guards every "is this zero / does a remainder survive"
//! decision, which keeps the pause detection and the substep rounding
//! deterministic and identical to a future `GPU` compute path.
//!
//! # Determinism (design §29)
//! Given the same inputs, every function here returns the same integer counts
//! and the same scaled scalars, and [`FixedStepAccumulator`] advances its
//! remainder by a fixed rule, so recorded playback reproduces the identical
//! per-emitter clock on replay.

/// Absolute tolerance for every floating-point comparison in this module.
///
/// The contract crate forbids `==` / `!=` on `f32`, so "is this value zero",
/// "is this emitter paused", and "did a sub-step remainder survive" are all
/// decided against this epsilon instead. It is deliberately coarse relative to
/// a per-frame `dt` (seconds) so a scale of a few ULPs above zero still reads
/// as paused.
pub const CMP_EPS: f32 = 1.0e-6;

/// Clamps a scalar into `[0, +inf)`, mapping `NaN` to `0.0`.
///
/// Used wherever a scaled `dt`, spawn rate, or age delta must never go negative
/// and must never propagate a `NaN` into the simulation.
#[must_use]
fn clamp_non_negative(value: f32) -> f32 {
    if value.is_nan() || value < 0.0 {
        0.0
    } else {
        value
    }
}

/// A non-negative time-scale multiplier (a slow-motion / fast-forward factor).
///
/// `1.0` is real time, `0.5` is half speed, `0.0` is frozen. Construction
/// clamps the input into `[0, +inf)` and folds `NaN` to `0.0`, so a
/// `TimeScale` can never carry a negative or non-finite-below-zero factor into
/// the `dt` math. The inner field is private precisely so the clamp cannot be
/// bypassed by a struct literal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimeScale(f32);

impl TimeScale {
    /// Real-time scale (`1.0`): the neutral element of scale multiplication.
    pub const IDENTITY: Self = Self(1.0);

    /// Fully frozen scale (`0.0`).
    pub const PAUSED: Self = Self(0.0);

    /// Builds a `TimeScale`, clamping `scale` to `>= 0.0` and mapping `NaN`
    /// to `0.0`.
    #[must_use]
    pub fn new(scale: f32) -> Self {
        if scale.is_nan() || scale < 0.0 {
            Self(0.0)
        } else {
            Self(scale)
        }
    }

    /// Returns the clamped scale factor.
    #[must_use]
    pub fn get(self) -> f32 {
        self.0
    }

    /// Returns `true` when this scale is at (or within [`CMP_EPS`](self) of)
    /// zero, i.e. the emitter would not advance under it.
    #[must_use]
    pub fn is_paused_like(self) -> bool {
        self.0 <= CMP_EPS
    }
}

impl Default for TimeScale {
    /// Defaults to real time ([`TimeScale::IDENTITY`]).
    fn default() -> Self {
        Self::IDENTITY
    }
}

/// The composed time-dilation state of a single emitter (design §26).
///
/// The effective scale is the product of a scene-wide `global_scale` and a
/// per-emitter `local_scale`, gated by an explicit `paused` flag. This mirrors
/// `Niagara`'s split between a global time dilation and a per-emitter override:
/// pausing the scene freezes every emitter, and an emitter can additionally run
/// slower than the scene without touching the global clock.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EmitterTimeControl {
    /// Scene-wide dilation applied to every emitter.
    pub global_scale: TimeScale,
    /// Per-emitter dilation applied on top of [`Self::global_scale`].
    pub local_scale: TimeScale,
    /// Hard pause: when `true` the effective scale is exactly `0.0`.
    pub paused: bool,
}

impl EmitterTimeControl {
    /// The effective scale for this frame: `0.0` when paused, otherwise the
    /// product of the global and local scales.
    ///
    /// Both factors are already clamped `>= 0.0`, so the product is too.
    #[must_use]
    pub fn effective_scale(self) -> f32 {
        if self.paused {
            0.0
        } else {
            self.global_scale.get() * self.local_scale.get()
        }
    }

    /// Scales a raw frame `dt` by [`Self::effective_scale`], clamped to
    /// `>= 0.0` (and never `NaN`).
    ///
    /// This is the single `dt` the rest of the pipeline should advance the
    /// emitter by; a negative or `NaN` `raw_dt` collapses to `0.0` rather than
    /// running time backwards.
    #[must_use]
    pub fn effective_dt(self, raw_dt: f32) -> f32 {
        clamp_non_negative(raw_dt * self.effective_scale())
    }

    /// Returns `true` when the emitter will not advance this frame, either
    /// because it is paused or because the effective scale is within
    /// [`CMP_EPS`](self) of zero.
    #[must_use]
    pub fn is_effectively_paused(self) -> bool {
        self.effective_scale() <= CMP_EPS
    }
}

impl Default for EmitterTimeControl {
    /// Defaults to real time, unpaused, with identity scales.
    fn default() -> Self {
        Self {
            global_scale: TimeScale::IDENTITY,
            local_scale: TimeScale::IDENTITY,
            paused: false,
        }
    }
}

/// How a frame is decomposed into simulation steps (design §25).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StepPolicy {
    /// Advance by the whole scaled `dt` in one variable-length step.
    Variable,
    /// Advance by a fixed `step` seconds, accumulating any remainder frame to
    /// frame (see [`FixedStepAccumulator`]).
    Fixed {
        /// The fixed per-step timestep in seconds.
        step: f32,
    },
}

impl StepPolicy {
    /// Returns `true` for the [`StepPolicy::Fixed`] variant.
    #[must_use]
    pub fn is_fixed(self) -> bool {
        matches!(self, StepPolicy::Fixed { .. })
    }
}

/// A reference fixed-timestep accumulator for [`StepPolicy::Fixed`].
///
/// Each frame it banks the scaled `dt` and releases as many whole `step`s as
/// have accumulated, keeping the sub-step remainder for the next frame so the
/// simulation clock does not drift. A `max_steps` cap breaks the
/// *spiral-of-death* — the failure mode where a slow frame queues more steps
/// than the next frame can afford, which queues still more — by discarding the
/// backlog that exceeds the cap instead of letting `accumulated` grow without
/// bound.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FixedStepAccumulator {
    /// Unconsumed time banked toward the next fixed step, always kept in
    /// `[0, step)` after each [`Self::advance`].
    pub accumulated: f32,
}

impl FixedStepAccumulator {
    /// Banks `dt` and returns how many whole `step`s to run this frame.
    ///
    /// The count is `floor(accumulated / step)` clamped to `max_steps`; the
    /// consumed time (`count * step`) is deducted, leaving a remainder shorter
    /// than one `step`. When the demanded count exceeds `max_steps` the surplus
    /// backlog is dropped (spiral-of-death guard) rather than carried forward.
    ///
    /// Invalid inputs are handled without panicking: a non-positive or `NaN`
    /// `step` yields `0` and leaves the accumulator untouched, and a negative
    /// or `NaN` `dt` banks nothing.
    pub fn advance(&mut self, dt: f32, step: f32, max_steps: u32) -> u32 {
        // A non-positive or NaN step has no well-defined division; refuse to
        // advance rather than divide by (near) zero.
        if step.is_nan() || step <= CMP_EPS {
            return 0;
        }

        self.accumulated += clamp_non_negative(dt);

        // `as u32` saturates, so an enormous backlog cannot wrap the count.
        let demanded = (self.accumulated / step).floor();
        let demanded_steps = demanded as u32;
        let steps = demanded_steps.min(max_steps);

        self.accumulated -= steps as f32 * step;

        if demanded_steps > max_steps {
            // Spiral-of-death guard: we could not run the whole backlog, so
            // collapse the remaining time to the sub-step remainder instead of
            // carrying an ever-growing debt into the next frame.
            let whole = (self.accumulated / step).floor();
            self.accumulated -= whole * step;
        }

        steps
    }
}

/// Derives a `CFL`-style substep count for a scaled `dt`.
///
/// Given the effective `dt` and a stability ceiling `max_substep_dt` (the
/// largest step that stays stable for this emitter), this returns how many
/// equal substeps cover the frame: conceptually `ceil(effective_dt /
/// max_substep_dt)`, but computed *without* `f32::ceil`. The integer part comes
/// from the saturating `as u32` truncation, and one extra step is added only
/// when a real remainder survives the [`CMP_EPS`](self) tolerance
/// (`n * max_substep_dt + CMP_EPS < effective_dt`).
///
/// The result is clamped to `max_substeps`. It is `0` when the frame is
/// effectively empty (`effective_dt <= CMP_EPS`) and at least `1` otherwise
/// (subject to the `max_substeps` cap). A non-positive `max_substep_dt` cannot
/// be divided, so a non-empty frame falls back to a single capped step.
#[must_use]
pub fn substep_count(effective_dt: f32, max_substep_dt: f32, max_substeps: u32) -> u32 {
    if effective_dt.is_nan() || effective_dt <= CMP_EPS {
        return 0;
    }
    if max_substep_dt.is_nan() || max_substep_dt <= CMP_EPS {
        // No usable ceiling: run one step, still honoring the hard cap.
        return max_substeps.min(1);
    }

    let mut n = (effective_dt / max_substep_dt) as u32;
    if (n as f32) * max_substep_dt + CMP_EPS < effective_dt {
        n = n.saturating_add(1);
    }
    if n == 0 {
        n = 1;
    }
    n.min(max_substeps)
}

/// Scales an emitter's base spawn rate (particles per second) by the emitter's
/// effective time scale, clamped to `>= 0.0`.
///
/// Slowing an emitter must slow its emission too, otherwise slow-motion smoke
/// would emit at full density; a paused emitter spawns nothing.
#[must_use]
pub fn scaled_spawn_rate(base_rate: f32, ctrl: EmitterTimeControl) -> f32 {
    clamp_non_negative(base_rate * ctrl.effective_scale())
}

/// The age increment for a particle this frame under the emitter's clock.
///
/// Aging tracks the same dilated `dt` as motion and emission, so this is simply
/// [`EmitterTimeControl::effective_dt`]; a paused emitter ages nothing.
#[must_use]
pub fn scaled_age_delta(raw_dt: f32, ctrl: EmitterTimeControl) -> f32 {
    ctrl.effective_dt(raw_dt)
}

#[cfg(test)]
mod tests {
    use super::{
        scaled_age_delta, scaled_spawn_rate, substep_count, EmitterTimeControl,
        FixedStepAccumulator, StepPolicy, TimeScale, CMP_EPS,
    };

    /// Absolute-tolerance float comparison used throughout the tests, since the
    /// crate forbids `==` on `f32`.
    fn approx(value: f32, expected: f32) -> bool {
        (value - expected).abs() < CMP_EPS
    }

    #[test]
    fn time_scale_clamps_negative_and_nan_to_zero() {
        assert!(approx(TimeScale::new(-1.0).get(), 0.0));
        assert!(approx(TimeScale::new(f32::NAN).get(), 0.0));
        assert!(approx(TimeScale::new(0.5).get(), 0.5));
        assert!(approx(TimeScale::IDENTITY.get(), 1.0));
    }

    #[test]
    fn time_scale_is_paused_like_at_zero() {
        assert!(TimeScale::new(0.0).is_paused_like());
        assert!(TimeScale::PAUSED.is_paused_like());
        assert!(TimeScale::new(-3.0).is_paused_like());
        assert!(!TimeScale::new(1.0).is_paused_like());
    }

    #[test]
    fn effective_dt_is_zero_when_paused() {
        let ctrl = EmitterTimeControl {
            global_scale: TimeScale::new(1.0),
            local_scale: TimeScale::new(1.0),
            paused: true,
        };
        assert!(approx(ctrl.effective_scale(), 0.0));
        assert!(approx(ctrl.effective_dt(0.016), 0.0));
        assert!(ctrl.is_effectively_paused());
    }

    #[test]
    fn effective_dt_multiplies_global_and_local() {
        let ctrl = EmitterTimeControl {
            global_scale: TimeScale::new(0.5),
            local_scale: TimeScale::new(0.5),
            paused: false,
        };
        assert!(approx(ctrl.effective_scale(), 0.25));
        assert!(approx(ctrl.effective_dt(0.1), 0.025));
        assert!(!ctrl.is_effectively_paused());
    }

    #[test]
    fn effective_dt_clamps_negative_raw_dt() {
        let ctrl = EmitterTimeControl::default();
        assert!(approx(ctrl.effective_dt(-0.5), 0.0));
        assert!(approx(ctrl.effective_dt(f32::NAN), 0.0));
    }

    #[test]
    fn step_policy_is_fixed_flag() {
        assert!(!StepPolicy::Variable.is_fixed());
        assert!(StepPolicy::Fixed { step: 0.01 }.is_fixed());
    }

    #[test]
    fn fixed_accumulator_accumulates_and_deducts() {
        let mut acc = FixedStepAccumulator::default();
        // 0.025 with a 0.01 step -> 2 steps, 0.005 remainder.
        assert_eq!(acc.advance(0.025, 0.01, 8), 2);
        assert!(approx(acc.accumulated, 0.005));
        // Add another 0.02 -> 0.025 banked -> 2 steps, 0.005 remainder again.
        assert_eq!(acc.advance(0.02, 0.01, 8), 2);
        assert!(approx(acc.accumulated, 0.005));
    }

    #[test]
    fn fixed_accumulator_keeps_sub_step_remainder() {
        let mut acc = FixedStepAccumulator::default();
        // Less than one step: no step runs, all banked.
        assert_eq!(acc.advance(0.004, 0.01, 8), 0);
        assert!(approx(acc.accumulated, 0.004));
        // Now enough for exactly one step, remainder 0.002.
        assert_eq!(acc.advance(0.008, 0.01, 8), 1);
        assert!(approx(acc.accumulated, 0.002));
    }

    #[test]
    fn fixed_accumulator_clamps_spiral_of_death() {
        let mut acc = FixedStepAccumulator::default();
        // 1.0s at a 0.01 step demands 100 steps, capped to 4.
        assert_eq!(acc.advance(1.0, 0.01, 4), 4);
        // Backlog is dropped: remainder stays below one step, so the next frame
        // does not inherit the debt.
        assert!(acc.accumulated < 0.01);
        assert!(acc.accumulated >= 0.0);
        // A normal follow-up frame therefore behaves normally.
        assert_eq!(acc.advance(0.02, 0.01, 4), 2);
    }

    #[test]
    fn fixed_accumulator_rejects_invalid_step() {
        let mut acc = FixedStepAccumulator::default();
        assert_eq!(acc.advance(0.05, 0.0, 8), 0);
        assert_eq!(acc.advance(0.05, f32::NAN, 8), 0);
        assert!(approx(acc.accumulated, 0.0));
    }

    #[test]
    fn substep_count_zero_for_empty_frame() {
        assert_eq!(substep_count(0.0, 0.01, 8), 0);
        assert_eq!(substep_count(CMP_EPS * 0.5, 0.01, 8), 0);
        assert_eq!(substep_count(f32::NAN, 0.01, 8), 0);
    }

    #[test]
    fn substep_count_exact_division() {
        // 0.04 / 0.01 == 4 exactly, no extra step.
        assert_eq!(substep_count(0.04, 0.01, 16), 4);
    }

    #[test]
    fn substep_count_rounds_up_on_remainder() {
        // 0.045 / 0.01 -> 4 whole + remainder -> 5.
        assert_eq!(substep_count(0.045, 0.01, 16), 5);
        // A tiny frame still needs one step.
        assert_eq!(substep_count(0.002, 0.01, 16), 1);
    }

    #[test]
    fn substep_count_clamps_to_max() {
        // 0.5 / 0.01 -> 50 demanded, capped to 8.
        assert_eq!(substep_count(0.5, 0.01, 8), 8);
    }

    #[test]
    fn substep_count_falls_back_without_ceiling() {
        // Non-positive max_substep_dt: one capped step for a non-empty frame.
        assert_eq!(substep_count(0.02, 0.0, 8), 1);
        assert_eq!(substep_count(0.02, 0.0, 0), 0);
    }

    #[test]
    fn scaled_spawn_rate_tracks_effective_scale() {
        let ctrl = EmitterTimeControl {
            global_scale: TimeScale::new(0.5),
            local_scale: TimeScale::new(1.0),
            paused: false,
        };
        assert!(approx(scaled_spawn_rate(100.0, ctrl), 50.0));

        let paused = EmitterTimeControl {
            global_scale: TimeScale::new(1.0),
            local_scale: TimeScale::new(1.0),
            paused: true,
        };
        assert!(approx(scaled_spawn_rate(100.0, paused), 0.0));
        // Negative base rate clamps to zero.
        assert!(approx(scaled_spawn_rate(-5.0, ctrl), 0.0));
    }

    #[test]
    fn scaled_age_delta_matches_effective_dt() {
        let ctrl = EmitterTimeControl {
            global_scale: TimeScale::new(0.5),
            local_scale: TimeScale::new(0.5),
            paused: false,
        };
        assert!(approx(scaled_age_delta(0.1, ctrl), ctrl.effective_dt(0.1)));
        assert!(approx(scaled_age_delta(0.1, ctrl), 0.025));
    }
}
