//! Emitter warmup / pre-roll contracts for the Ember particle engine.
//!
//! Production VFX engines (Unreal `Niagara`'s emitter *warmup*, Unity `VFX
//! Graph`'s pre-warm, `PopcornFX`'s pre-roll) face the same authoring problem:
//! a freshly spawned emitter looks *empty* on its first visible frame. A
//! campfire that should already be burning, a dust cloud that should already
//! fill a room, or a waterfall that should already reach the ground all start
//! from zero particles unless the effect is *warmed up* — simulated for a
//! fixed span of virtual time *before* the first rendered frame so the pool
//! has reached its steady-state population.
//!
//! This module owns the small, purely `CPU`-verifiable contract that turns an
//! authored warmup duration into a concrete, deterministic plan:
//!
//! 1. [`WarmupConfig`] — the authored request: a total warmup duration and a
//!    fixed per-tick step.
//! 2. [`WarmupPlan`] — the solved schedule: how many ticks to run, the `dt`
//!    each tick advances, and the frame index the first warmup tick uses.
//! 3. [`plan_warmup`] — the solver that converts a config into a plan, clamping
//!    the tick count against a caller-supplied cost ceiling.
//! 4. [`warmup_frame_index`] — the deterministic frame-index sequence a warmup
//!    tick feeds into the stateless hash `RNG` (see
//!    [`super::determinism`]), chosen so warmup draws never collide with the
//!    real frames that follow.
//! 5. [`clamp_warmup_spawn`] — the capacity clamp that keeps a warmup's
//!    accumulated spawn count inside the emitter pool.
//!
//! Every routine here is deterministic, allocation-free, and total: no input
//! panics, out-of-range ticks return [`None`] rather than aborting, and all
//! integer arithmetic states its overflow semantics explicitly with
//! `saturating_*` / `wrapping_*`. Following the sibling modules, floating-point
//! magnitudes are compared against a local [`CMP_EPS`] epsilon and never with
//! `==` / `!=`, and only non-transcendental arithmetic is used (multiply,
//! divide, integer floor via `as u32`).

/// Absolute epsilon for float magnitude comparisons in this module.
///
/// The warmup solver only ever asks "is this duration effectively zero?" and
/// "did integer truncation drop a fractional tick?", so a single small
/// absolute epsilon is sufficient; there is no accumulation that would demand a
/// relative tolerance. Matches the epsilon used by the sibling
/// [`super::frame_pipeline`] and `dual_backend` modules.
pub const CMP_EPS: f32 = 1.0e-6;

/// The authored warmup request (design: emitter warmup / pre-roll).
///
/// Both fields are wall-clock seconds of *virtual* simulation time. The plan
/// solver ([`plan_warmup`]) treats them as a request, not a guarantee: a zero
/// or negative duration disables warmup, and the resulting tick count is
/// clamped against a caller-supplied ceiling so a mis-authored multi-minute
/// warmup can never stall a level load.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WarmupConfig {
    /// Total virtual time to pre-simulate before the first rendered frame.
    ///
    /// Values at or below [`CMP_EPS`] disable warmup entirely.
    pub warmup_time: f32,
    /// Fixed virtual `dt` advanced by each warmup tick.
    ///
    /// A smaller tick yields a more accurate (and more expensive) warmup;
    /// values at or below [`CMP_EPS`] disable warmup entirely because no finite
    /// number of ticks could cover the requested duration.
    pub warmup_tick: f32,
}

impl WarmupConfig {
    /// Constructs a warmup config from a duration and a per-tick step.
    #[must_use]
    pub const fn new(warmup_time: f32, warmup_tick: f32) -> Self {
        Self {
            warmup_time,
            warmup_tick,
        }
    }

    /// A disabled warmup: zero duration, so no ticks are ever scheduled.
    pub const DISABLED: Self = Self {
        warmup_time: 0.0,
        warmup_tick: 0.0,
    };

    /// Returns `true` when this config requests no warmup.
    ///
    /// A config is disabled when either the total duration or the per-tick step
    /// is effectively zero (or negative): the first case asks for no virtual
    /// time, and the second could never be covered by a finite tick count.
    #[must_use]
    pub fn is_disabled(self) -> bool {
        self.warmup_time <= CMP_EPS || self.warmup_tick <= CMP_EPS
    }
}

/// The solved, deterministic warmup schedule (design: emitter warmup).
///
/// A plan is produced by [`plan_warmup`] and is fully determined by its inputs,
/// so two peers (or a `CPU` reference and a future `GPU` kernel) that solve the
/// same config against the same base frame index derive an identical plan.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WarmupPlan {
    /// Number of fixed-`dt` ticks to run before the first rendered frame.
    ///
    /// Zero means warmup is disabled; the pool starts empty.
    pub tick_count: u32,
    /// Virtual `dt` each tick advances (equal to the config's `warmup_tick`,
    /// clamped to be non-negative).
    pub dt_per_tick: f32,
    /// Frame index fed to the hash `RNG` for the *first* warmup tick.
    ///
    /// Warmup ticks occupy the `tick_count` frame indices immediately *before*
    /// `base_frame_index`, so their random draws never collide with the real
    /// frames that follow. See [`warmup_frame_index`].
    pub start_frame_index: u64,
}

impl WarmupPlan {
    /// A plan that performs no warmup, anchored at `base_frame_index`.
    ///
    /// Used when the config is disabled or the cost ceiling is zero. The
    /// `dt_per_tick` is zero because no tick will ever consume it, and the
    /// start frame index degenerates to the base (there is no preceding
    /// interval to reserve).
    #[must_use]
    pub const fn disabled(base_frame_index: u64) -> Self {
        Self {
            tick_count: 0,
            dt_per_tick: 0.0,
            start_frame_index: base_frame_index,
        }
    }

    /// Returns `true` when the plan schedules at least one warmup tick.
    #[must_use]
    pub const fn is_active(&self) -> bool {
        self.tick_count > 0
    }
}

/// Solves a [`WarmupConfig`] into a concrete [`WarmupPlan`].
///
/// # Tick count
///
/// When the config is disabled (see [`WarmupConfig::is_disabled`]) or the cost
/// ceiling `max_ticks` is zero, the result performs no warmup. Otherwise the
/// tick count is `ceil(warmup_time / warmup_tick)`, computed with integer
/// arithmetic rather than `f32::ceil` (a transcendental-adjacent helper the
/// workspace lint forbids): the quotient is truncated toward zero with an
/// `as u32` cast, and a single fractional remainder is detected by checking
/// whether the truncated ticks fall short of the requested duration by more
/// than [`CMP_EPS`]. The count is then clamped to `max_ticks` so an emitter's
/// warmup cost is bounded regardless of authoring.
///
/// # Frame interval
///
/// The plan reserves the `tick_count` frame indices immediately before
/// `base_frame_index`: `start_frame_index = base_frame_index.wrapping_sub(
/// tick_count)`. Because the hash `RNG` is a pure function of the frame index
/// (design §29), placing warmup on a disjoint, contiguous interval guarantees
/// warmup draws are decorrelated from — and reproducible alongside — the real
/// frames that follow. `wrapping_sub` gives well-defined behavior even when
/// `base_frame_index` is near zero; the indices simply wrap into the high end
/// of the `u64` range, which is still a disjoint contiguous block.
///
/// # Determinism & totality
///
/// The function never panics and allocates nothing. `dt_per_tick` is the
/// config's `warmup_tick` clamped to be non-negative so a downstream integrator
/// never steps backwards in time.
#[must_use]
pub fn plan_warmup(config: WarmupConfig, base_frame_index: u64, max_ticks: u32) -> WarmupPlan {
    if config.is_disabled() || max_ticks == 0 {
        return WarmupPlan::disabled(base_frame_index);
    }

    // Integer-floor of the exact quotient, then add one tick if truncation
    // dropped a fractional remainder. `warmup_tick > CMP_EPS` here (the config
    // is not disabled), so the divide is well-conditioned and the quotient is
    // finite and non-negative.
    let quotient = config.warmup_time / config.warmup_tick;
    let truncated = quotient as u32;
    let covered = truncated as f32 * config.warmup_tick;
    let tick_count = if covered + CMP_EPS < config.warmup_time {
        truncated.saturating_add(1)
    } else {
        truncated
    }
    .min(max_ticks);

    // A clamp to zero ticks (only reachable if `max_ticks` were zero, already
    // handled above) would still be a valid disabled plan; guard anyway so the
    // frame interval stays meaningful.
    if tick_count == 0 {
        return WarmupPlan::disabled(base_frame_index);
    }

    let dt_per_tick = if config.warmup_tick < 0.0 {
        0.0
    } else {
        config.warmup_tick
    };
    let start_frame_index = base_frame_index.wrapping_sub(tick_count as u64);

    WarmupPlan {
        tick_count,
        dt_per_tick,
        start_frame_index,
    }
}

/// Returns the hash-`RNG` frame index for warmup tick `tick`, or [`None`].
///
/// Warmup ticks run on the contiguous interval
/// `[start_frame_index, start_frame_index + tick_count)`, so tick `i` uses
/// `start_frame_index + i`. A `tick` at or beyond `plan.tick_count` is out of
/// range and yields [`None`] rather than panicking. The addition uses
/// `wrapping_add` to match the `wrapping_sub` the plan used to reserve the
/// interval: when the interval wraps the `u64` boundary the sequence stays
/// contiguous and, by construction, ends exactly at `base_frame_index`.
#[must_use]
pub fn warmup_frame_index(plan: &WarmupPlan, tick: u32) -> Option<u64> {
    if tick >= plan.tick_count {
        return None;
    }
    Some(plan.start_frame_index.wrapping_add(tick as u64))
}

/// Clamps a warmup's accumulated spawn request to the emitter pool capacity.
///
/// Across all warmup ticks an emitter may *request* more spawns than the pool
/// can hold; the pool is a fixed-capacity ring, so the steady-state population
/// can never exceed `capacity`. This returns `min(requested_total, capacity)`
/// as a `u32`, saturating rather than wrapping when `requested_total` exceeds
/// the `u32` range, so an absurdly large accumulated request collapses to the
/// capacity instead of silently truncating to a small value. Never panics.
#[must_use]
pub fn clamp_warmup_spawn(requested_total: u64, capacity: u32) -> u32 {
    let capped = requested_total.min(capacity as u64);
    // `capped <= capacity <= u32::MAX`, so this cast is exact; the explicit
    // `min` above rules out any wrapping truncation.
    capped as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASE: u64 = 10_000;

    #[test]
    fn zero_time_disables_warmup() {
        let plan = plan_warmup(WarmupConfig::new(0.0, 0.25), BASE, 64);
        assert_eq!(plan.tick_count, 0);
        assert!(!plan.is_active());
        assert_eq!(plan.start_frame_index, BASE);
        assert_eq!(warmup_frame_index(&plan, 0), None);
    }

    #[test]
    fn zero_tick_disables_warmup() {
        let plan = plan_warmup(WarmupConfig::new(1.0, 0.0), BASE, 64);
        assert_eq!(plan.tick_count, 0);
        assert_eq!(warmup_frame_index(&plan, 0), None);
        assert_eq!(warmup_frame_index(&plan, 5), None);
    }

    #[test]
    fn negative_inputs_disable_warmup() {
        assert!(WarmupConfig::new(-1.0, 0.25).is_disabled());
        assert!(WarmupConfig::new(1.0, -0.25).is_disabled());
        let plan = plan_warmup(WarmupConfig::new(-1.0, 0.25), BASE, 64);
        assert_eq!(plan.tick_count, 0);
    }

    #[test]
    fn disabled_constant_config_is_disabled() {
        assert!(WarmupConfig::DISABLED.is_disabled());
        let plan = plan_warmup(WarmupConfig::DISABLED, BASE, 64);
        assert!(!plan.is_active());
    }

    #[test]
    fn max_ticks_zero_disables_warmup() {
        let plan = plan_warmup(WarmupConfig::new(1.0, 0.25), BASE, 0);
        assert_eq!(plan.tick_count, 0);
        assert_eq!(plan.start_frame_index, BASE);
    }

    #[test]
    fn exact_division_gives_exact_tick_count() {
        // 1.0 / 0.25 == 4 exactly, no remainder tick.
        let plan = plan_warmup(WarmupConfig::new(1.0, 0.25), BASE, 64);
        assert_eq!(plan.tick_count, 4);
        assert!((plan.dt_per_tick - 0.25).abs() < CMP_EPS);
    }

    #[test]
    fn fractional_remainder_rounds_up() {
        // 1.0 / 0.3 == 3.33..., so ceil is 4.
        let plan = plan_warmup(WarmupConfig::new(1.0, 0.3), BASE, 64);
        assert_eq!(plan.tick_count, 4);
    }

    #[test]
    fn single_tick_when_time_below_tick() {
        // 0.1 / 0.25 truncates to 0 but leaves a remainder, so one tick runs.
        let plan = plan_warmup(WarmupConfig::new(0.1, 0.25), BASE, 64);
        assert_eq!(plan.tick_count, 1);
    }

    #[test]
    fn tick_count_is_clamped_by_max_ticks() {
        // 10.0 / 0.25 == 40, but the ceiling is 8.
        let plan = plan_warmup(WarmupConfig::new(10.0, 0.25), BASE, 8);
        assert_eq!(plan.tick_count, 8);
    }

    #[test]
    fn start_frame_is_contiguous_interval_before_base() {
        let plan = plan_warmup(WarmupConfig::new(1.0, 0.25), BASE, 64);
        assert_eq!(plan.tick_count, 4);
        // The interval [start, start + tick_count) ends exactly at base.
        assert_eq!(plan.start_frame_index, BASE - 4);
        assert_eq!(
            plan.start_frame_index + plan.tick_count as u64,
            BASE,
            "warmup interval must abut base_frame_index"
        );
    }

    #[test]
    fn warmup_frame_indices_are_consecutive_and_bounded() {
        let plan = plan_warmup(WarmupConfig::new(1.0, 0.25), BASE, 64);
        assert_eq!(warmup_frame_index(&plan, 0), Some(BASE - 4));
        assert_eq!(warmup_frame_index(&plan, 1), Some(BASE - 3));
        assert_eq!(warmup_frame_index(&plan, 2), Some(BASE - 2));
        assert_eq!(warmup_frame_index(&plan, 3), Some(BASE - 1));
        // The last real-frame-adjacent index is base - 1, never base itself.
        assert_eq!(warmup_frame_index(&plan, 4), None);
        assert_eq!(warmup_frame_index(&plan, 100), None);
    }

    #[test]
    fn warmup_interval_near_zero_base_wraps_without_panic() {
        let plan = plan_warmup(WarmupConfig::new(1.0, 0.25), 2, 64);
        assert_eq!(plan.tick_count, 4);
        // 2.wrapping_sub(4) wraps to u64::MAX - 1.
        assert_eq!(plan.start_frame_index, 2u64.wrapping_sub(4));
        // The sequence still ends exactly at the base frame index.
        let last = warmup_frame_index(&plan, plan.tick_count - 1).unwrap();
        assert_eq!(last.wrapping_add(1), 2);
    }

    #[test]
    fn clamp_spawn_below_capacity_is_unchanged() {
        assert_eq!(clamp_warmup_spawn(100, 4096), 100);
    }

    #[test]
    fn clamp_spawn_above_capacity_saturates_to_capacity() {
        assert_eq!(clamp_warmup_spawn(9000, 4096), 4096);
    }

    #[test]
    fn clamp_spawn_at_capacity_is_capacity() {
        assert_eq!(clamp_warmup_spawn(4096, 4096), 4096);
    }

    #[test]
    fn clamp_spawn_huge_request_does_not_wrap() {
        // A request far beyond u32 range must collapse to capacity, not wrap
        // into a small value.
        assert_eq!(clamp_warmup_spawn(u64::MAX, 1024), 1024);
        assert_eq!(clamp_warmup_spawn(u64::MAX, u32::MAX), u32::MAX);
    }

    #[test]
    fn clamp_spawn_zero_capacity_is_zero() {
        assert_eq!(clamp_warmup_spawn(500, 0), 0);
    }
}
