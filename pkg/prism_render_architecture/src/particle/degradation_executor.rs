//! Per-emitter progressive runtime degradation executor (design §28).
//!
//! This module is the **execution loop** that closes the gap between the two
//! halves of the §28 runtime-degradation story that already live in the
//! particle subsystem but never actually drive each other:
//!
//! * [`super::platform`] owns the *policy*: the fixed, ordered
//!   [`super::platform::DEGRADATION_LADDER`] (reduce spawn, disable
//!   sorting/`OIT`, lower volume resolution, simplify shading, reduce update
//!   rate, cull the renderer, then pause the simulation), the
//!   `(priority, screen coverage, distance)` ordering key through
//!   [`super::platform::order_degradation_candidates`], and the *global*
//!   over-budget staircase [`super::platform::resolve_runtime_degradation`]
//!   that returns a single ladder depth for the whole frame.
//! * [`super::perf_budget`] owns the *measurement*: the three §28 budget
//!   dimensions (live particle count, `VRAM` bytes, `GPU` milliseconds)
//!   accumulated in a [`super::perf_budget::BudgetLedger`] over a
//!   [`super::perf_budget::TripleBudget`], reported as a
//!   [`super::perf_budget::BudgetReport`], and folded by
//!   [`super::perf_budget::arbitrate`] into a scalar
//!   [`super::perf_budget::BudgetPressure`] signal.
//!
//! What neither sibling owns is the **per-emitter progressive loop** §28 calls
//! for: starting from the *least important* emitter, walk the ladder one rung at
//! a time, recompute the reclaimed budget after every applied
//! [`super::platform::DegradationAction`], and stop the instant the
//! [`super::perf_budget::BudgetPressure`] falls back within budget. Rather than
//! slamming every emitter to the same global depth, this executor sheds the
//! smallest amount of detail that clears the overspend, concentrating the pain
//! on the emitters the ordering key already deemed least important.
//!
//! # Algorithm
//! 1. Charge every emitter's current cost into a
//!    [`super::perf_budget::BudgetLedger`] and arbitrate the resulting pressure.
//!    If the frame is already within budget, assign every emitter ladder level
//!    `0` (no action) and return.
//! 2. Otherwise order the emitters least-important-first with
//!    [`super::platform::order_degradation_candidates`].
//! 3. Walk that order. For each emitter, apply ladder rungs one at a time; each
//!    rung reclaims a deterministic fraction of that emitter's *remaining* cost
//!    (see [`recovered_for_actions`]). After every rung, rebuild the ledger from
//!    the reduced costs, re-[`super::perf_budget::arbitrate`], and stop the whole
//!    loop as soon as the pressure clears.
//! 4. If the ladder is exhausted on every emitter and the frame is still over
//!    budget, report the plan as unresolved so the caller can escalate.
//!
//! # Determinism
//! Every reclaim fraction is a fixed-point basis-point constant applied with
//! integer multiply/divide (`f32` only for the millisecond dimension, and then
//! only `+ - * /`), no transcendental function is used except the `.sqrt()`
//! inside [`super::Vec3::distance`] when deriving a candidate from world
//! positions, and no bare `f32` equality is performed. The same inputs therefore
//! yield a bit-identical [`DegradationPlan`], so a future `GPU`-side reproduction
//! of the same decisions is bit-exact. This module performs the real
//! degradation arbitration only; it is not a hash, checksum, codec, or random
//! generator.

use alloc::vec::Vec;

use super::platform::{
    order_degradation_candidates, DegradationAction, DegradationCandidate, DEGRADATION_LADDER,
};
use super::perf_budget::{arbitrate, BudgetLedger, BudgetPressure, FrameStage, FrameTimeReport, TripleBudget};
use super::{EmitterHandle, Vec3};

/// Fixed-point denominator for reclaim fractions.
///
/// Each ladder rung's reclaim share is stored in basis points (parts per
/// `10_000`) so the particle and `VRAM` dimensions reclaim through integer
/// multiply/divide and stay deterministic across platforms.
const RECOVERY_BASIS: u32 = 10_000;

/// A single emitter's current per-dimension runtime cost (design §28).
///
/// These are the three §28 budget dimensions measured for one emitter: live
/// particle count, `VRAM` footprint in bytes, and `GPU` milliseconds. The
/// executor reclaims fractions of these as it walks the ladder.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EmitterRuntimeCost {
    /// Live particle count contributed by this emitter.
    pub particles: u32,
    /// `VRAM` footprint of this emitter, in bytes.
    pub vram_bytes: u64,
    /// `GPU` time spent on this emitter this frame, in milliseconds.
    pub gpu_ms: f32,
}

impl EmitterRuntimeCost {
    /// A zero-cost emitter (used as the residual of a fully paused emitter).
    pub const ZERO: Self = Self {
        particles: 0,
        vram_bytes: 0,
        gpu_ms: 0.0,
    };

    /// Builds a cost from its three dimensions.
    #[must_use]
    pub const fn new(particles: u32, vram_bytes: u64, gpu_ms: f32) -> Self {
        Self {
            particles,
            vram_bytes,
            gpu_ms,
        }
    }
}

/// One live emitter offered to the executor: its ordering key plus its cost.
///
/// The `candidate` carries the `(priority, screen coverage, distance)` key the
/// executor orders by (reusing [`super::platform::order_degradation_candidates`]);
/// the `cost` is the per-dimension load the executor reclaims from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EmitterBudgetLoad {
    /// The §28 ordering key for this emitter.
    pub candidate: DegradationCandidate,
    /// This emitter's current per-dimension runtime cost.
    pub cost: EmitterRuntimeCost,
}

/// The budget reclaimed by applying one or more ladder rungs.
///
/// Reported both per-rung internally and as the plan-wide total in
/// [`DegradationPlan::total_recovered`]. The integer dimensions saturate so a
/// pathological accumulation can never wrap.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RecoveredBudget {
    /// Particle slots reclaimed.
    pub particles: u32,
    /// `VRAM` bytes reclaimed.
    pub vram_bytes: u64,
    /// `GPU` milliseconds reclaimed.
    pub gpu_ms: f32,
}

impl RecoveredBudget {
    /// A no-reclaim total.
    pub const ZERO: Self = Self {
        particles: 0,
        vram_bytes: 0,
        gpu_ms: 0.0,
    };

    /// Sums two reclaim totals, saturating the integer dimensions.
    #[must_use]
    fn add(self, other: Self) -> Self {
        Self {
            particles: self.particles.saturating_add(other.particles),
            vram_bytes: self.vram_bytes.saturating_add(other.vram_bytes),
            gpu_ms: self.gpu_ms + other.gpu_ms,
        }
    }
}

/// The degradation decision assigned to a single emitter.
///
/// `level` is how many rungs of [`super::platform::DEGRADATION_LADDER`] were
/// applied (`0` means untouched, `7` means paused); [`EmitterDegradation::actions`]
/// expands it into the exact ordered prefix of actions. `residual` is the
/// emitter's cost after the reclaim.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EmitterDegradation {
    /// Emitter this decision applies to.
    pub handle: EmitterHandle,
    /// Number of ladder rungs applied, in `0..=7`.
    pub level: usize,
    /// This emitter's cost after the applied rungs reclaimed their share.
    pub residual: EmitterRuntimeCost,
}

impl EmitterDegradation {
    /// The ordered prefix of ladder actions applied to this emitter.
    ///
    /// Equivalent to [`super::platform::active_degradation_actions`] for this
    /// emitter's `level`; empty when the emitter was left untouched.
    #[must_use]
    pub fn actions(&self) -> &'static [DegradationAction] {
        &DEGRADATION_LADDER[..self.level.min(DEGRADATION_LADDER.len())]
    }

    /// Whether any rung was applied to this emitter.
    #[must_use]
    pub fn is_degraded(&self) -> bool {
        self.level > 0
    }
}

/// The result of a progressive degradation pass over a set of emitters.
///
/// Assignments are returned in the caller's input order (not the degradation
/// order) so a caller can index them positionally; use
/// [`DegradationPlan::assignment_for`] to look one up by handle.
#[derive(Clone, Debug, PartialEq)]
pub struct DegradationPlan {
    /// Per-emitter decisions, in the caller's input order.
    pub assignments: Vec<EmitterDegradation>,
    /// The final arbitrated pressure after the pass.
    pub final_pressure: BudgetPressure,
    /// `true` when the pass drove the frame back within budget.
    pub resolved_within_budget: bool,
    /// Total budget reclaimed across every emitter.
    pub total_recovered: RecoveredBudget,
}

impl DegradationPlan {
    /// Looks up the decision for one emitter by handle.
    #[must_use]
    pub fn assignment_for(&self, handle: EmitterHandle) -> Option<&EmitterDegradation> {
        self.assignments.iter().find(|a| a.handle == handle)
    }

    /// The deepest ladder level any single emitter was driven to.
    #[must_use]
    pub fn max_level(&self) -> usize {
        self.assignments.iter().map(|a| a.level).max().unwrap_or(0)
    }
}

/// Per-rung reclaim shares in basis points: `(particles, VRAM, GPU)`.
///
/// Each tuple names the fraction of an emitter's *remaining* cost that the rung
/// reclaims. The numbers encode what the §28 ladder physically does:
/// `ReduceSpawn` halves the particle inflow (and with it most of its `VRAM` and
/// a chunk of `GPU`); the sorting/`OIT`, volume-resolution, shading, and
/// update-rate rungs chip `GPU` (and some `VRAM`) without touching the live
/// count; `CullRenderer` stops drawing (most remaining `GPU`, some `VRAM`); and
/// `PauseSimulation` is the last resort that reclaims everything that is left.
#[must_use]
const fn reclaim_bps(action: DegradationAction) -> (u32, u32, u32) {
    match action {
        // Halve spawn: ~50% particles, ~40% VRAM, ~25% GPU of what remains.
        DegradationAction::ReduceSpawn => (5_000, 4_000, 2_500),
        // Drop sorting + route through cheap OIT: GPU only, with a little VRAM.
        DegradationAction::DisableSortingOit => (0, 500, 1_500),
        // Coarser volume grid: frees VRAM and some GPU, not the live count.
        DegradationAction::LowerVolumeResolution => (0, 3_500, 2_000),
        // Cheaper shading closure: GPU only.
        DegradationAction::SimplifyShading => (0, 0, 2_500),
        // Simulate every N frames: amortizes a large GPU share.
        DegradationAction::ReduceUpdateRate => (0, 0, 3_500),
        // Stop drawing while still simulating: most remaining GPU, some VRAM.
        DegradationAction::CullRenderer => (0, 2_000, 7_000),
        // Last resort: reclaim everything that is left on every dimension.
        DegradationAction::PauseSimulation => (RECOVERY_BASIS, RECOVERY_BASIS, RECOVERY_BASIS),
    }
}

/// Multiplies a `u32` by a basis-point fraction, rounding toward zero.
///
/// Uses a `u64` intermediate so the product never overflows; with `bps` capped
/// at [`RECOVERY_BASIS`] the result never exceeds `value`.
#[must_use]
const fn mul_bps_u32(value: u32, bps: u32) -> u32 {
    ((value as u64 * bps as u64) / RECOVERY_BASIS as u64) as u32
}

/// Multiplies a `u64` by a basis-point fraction, rounding toward zero.
///
/// Uses a `u128` intermediate so the product never overflows; with `bps` capped
/// at [`RECOVERY_BASIS`] the result never exceeds `value`.
#[must_use]
const fn mul_bps_u64(value: u64, bps: u32) -> u64 {
    ((value as u128 * bps as u128) / RECOVERY_BASIS as u128) as u64
}

/// Applies one ladder rung to an emitter's remaining cost, returning the
/// reclaimed amount and shrinking `remaining` in place.
///
/// Because every reclaim fraction is at most [`RECOVERY_BASIS`], each dimension
/// shrinks monotonically toward zero and the reclaimed amount is never negative.
fn apply_action(remaining: &mut EmitterRuntimeCost, action: DegradationAction) -> RecoveredBudget {
    let (particle_bps, vram_bps, gpu_bps) = reclaim_bps(action);

    let particles = mul_bps_u32(remaining.particles, particle_bps);
    let vram_bytes = mul_bps_u64(remaining.vram_bytes, vram_bps);
    let gpu_ms = remaining.gpu_ms * (gpu_bps as f32 / RECOVERY_BASIS as f32);

    remaining.particles -= particles;
    remaining.vram_bytes -= vram_bytes;
    remaining.gpu_ms = (remaining.gpu_ms - gpu_ms).max(0.0);

    RecoveredBudget {
        particles,
        vram_bytes,
        gpu_ms,
    }
}

/// Returns the cumulative budget reclaimed by applying the first `level` rungs
/// of [`super::platform::DEGRADATION_LADDER`] to an emitter of the given cost.
///
/// This is the pure building block the executor loops over, exposed so callers
/// can preview a reclaim without running a whole pass. The reclaim is monotonic:
/// every dimension of the returned total is non-decreasing in `level`.
#[must_use]
pub fn recovered_for_actions(cost: EmitterRuntimeCost, level: usize) -> RecoveredBudget {
    let depth = level.min(DEGRADATION_LADDER.len());
    let mut remaining = cost;
    let mut total = RecoveredBudget::ZERO;
    for action in &DEGRADATION_LADDER[..depth] {
        total = total.add(apply_action(&mut remaining, *action));
    }
    total
}

/// Builds a [`super::platform::DegradationCandidate`] from world positions.
///
/// The camera distance is derived with [`super::Vec3::distance`] (the only
/// transcendental call, a single `.sqrt()`), so a caller holding world
/// positions can produce an ordering key without reaching for a math library.
#[must_use]
pub fn candidate_from_world(
    handle: EmitterHandle,
    priority: u32,
    screen_coverage: f32,
    emitter_position: Vec3,
    camera_position: Vec3,
) -> DegradationCandidate {
    DegradationCandidate {
        handle,
        priority,
        screen_coverage,
        distance: emitter_position.distance(camera_position),
    }
}

/// A frame-time report that contributes no pressure of its own.
///
/// The executor arbitrates purely over the §28 triple budget; the frame-time
/// dimension of [`super::perf_budget::arbitrate`] is neutralized by reporting a
/// zero overspend against the budget's `GPU`-time ceiling, so the pressure the
/// loop stops on reflects the particle / `VRAM` / `GPU` dimensions it manages.
#[must_use]
fn neutral_frame_report(budget: TripleBudget) -> FrameTimeReport {
    FrameTimeReport {
        stage_overspend_ms: [0.0; FrameStage::COUNT],
        total_measured_ms: 0.0,
        total_target_ms: budget.max_gpu_ms,
        total_overspend_ms: 0.0,
    }
}

/// Charges every remaining emitter cost into the ledger and arbitrates the
/// resulting [`super::perf_budget::BudgetPressure`].
fn pressure_of(
    budget: TripleBudget,
    remaining: &[EmitterRuntimeCost],
    ledger: &mut BudgetLedger,
) -> BudgetPressure {
    ledger.reset();
    for cost in remaining {
        ledger.charge_particles(cost.particles);
        ledger.charge_vram(cost.vram_bytes);
        ledger.charge_gpu_ms(cost.gpu_ms);
    }
    arbitrate(&ledger.report(), &neutral_frame_report(budget))
}

/// Resolves the least-important-first processing order of the input loads.
///
/// Delegates the ordering key to
/// [`super::platform::order_degradation_candidates`] and then maps the ordered
/// candidates back to indices into `loads`. Positional matching with a used-mask
/// keeps the result deterministic even if two loads share a handle.
fn processing_order(loads: &[EmitterBudgetLoad]) -> Vec<usize> {
    let mut ordered: Vec<DegradationCandidate> = loads.iter().map(|load| load.candidate).collect();
    order_degradation_candidates(&mut ordered);

    let mut used = alloc::vec![false; loads.len()];
    let mut order = Vec::with_capacity(loads.len());
    for candidate in &ordered {
        for (index, load) in loads.iter().enumerate() {
            if !used[index] && load.candidate.handle == candidate.handle {
                used[index] = true;
                order.push(index);
                break;
            }
        }
    }
    order
}

/// Assembles the final [`DegradationPlan`] from the loop state.
fn finish(
    loads: &[EmitterBudgetLoad],
    remaining: &[EmitterRuntimeCost],
    levels: &[usize],
    final_pressure: BudgetPressure,
    resolved_within_budget: bool,
) -> DegradationPlan {
    let mut assignments = Vec::with_capacity(loads.len());
    let mut total_recovered = RecoveredBudget::ZERO;
    for (index, load) in loads.iter().enumerate() {
        let residual = remaining[index];
        assignments.push(EmitterDegradation {
            handle: load.candidate.handle,
            level: levels[index],
            residual,
        });
        total_recovered = total_recovered.add(RecoveredBudget {
            particles: load.cost.particles.saturating_sub(residual.particles),
            vram_bytes: load.cost.vram_bytes.saturating_sub(residual.vram_bytes),
            gpu_ms: (load.cost.gpu_ms - residual.gpu_ms).max(0.0),
        });
    }
    DegradationPlan {
        assignments,
        final_pressure,
        resolved_within_budget,
        total_recovered,
    }
}

/// Runs the §28 per-emitter progressive degradation loop.
///
/// Starting from the least-important emitter, applies ladder rungs one at a time
/// and recomputes the reclaimed budget after each, stopping the instant the
/// arbitrated [`super::perf_budget::BudgetPressure`] falls back within budget.
/// When the frame already fits, every emitter is assigned level `0`. When the
/// whole ladder is exhausted on every emitter and the frame is still over
/// budget, the returned plan's `resolved_within_budget` is `false`.
#[must_use]
pub fn execute_degradation(budget: TripleBudget, loads: &[EmitterBudgetLoad]) -> DegradationPlan {
    let mut remaining: Vec<EmitterRuntimeCost> = loads.iter().map(|load| load.cost).collect();
    let mut levels: Vec<usize> = alloc::vec![0; loads.len()];
    let mut ledger = BudgetLedger::new(budget);

    let mut pressure = pressure_of(budget, &remaining, &mut ledger);
    if !pressure.is_over_budget() {
        return finish(loads, &remaining, &levels, pressure, true);
    }

    for &index in &processing_order(loads) {
        while levels[index] < DEGRADATION_LADDER.len() {
            let action = DEGRADATION_LADDER[levels[index]];
            apply_action(&mut remaining[index], action);
            levels[index] += 1;
            pressure = pressure_of(budget, &remaining, &mut ledger);
            if !pressure.is_over_budget() {
                return finish(loads, &remaining, &levels, pressure, true);
            }
        }
    }

    finish(loads, &remaining, &levels, pressure, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute tolerance for comparing expected `f32`s in assertions.
    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1.0e-5
    }

    fn load(handle: u32, priority: u32, cost: EmitterRuntimeCost) -> EmitterBudgetLoad {
        EmitterBudgetLoad {
            candidate: DegradationCandidate {
                handle: EmitterHandle(handle),
                priority,
                screen_coverage: 0.5,
                distance: 10.0,
            },
            cost,
        }
    }

    fn roomy_budget() -> TripleBudget {
        TripleBudget {
            max_particles: 1_000_000,
            max_vram_bytes: 1_000_000_000,
            max_gpu_ms: 1_000.0,
        }
    }

    #[test]
    fn within_budget_assigns_no_actions() {
        let budget = roomy_budget();
        let loads = [
            load(0, 1, EmitterRuntimeCost::new(100, 1_000, 1.0)),
            load(1, 5, EmitterRuntimeCost::new(100, 1_000, 1.0)),
        ];
        let plan = execute_degradation(budget, &loads);
        assert!(plan.resolved_within_budget);
        assert_eq!(plan.max_level(), 0);
        for assignment in &plan.assignments {
            assert_eq!(assignment.level, 0);
            assert!(assignment.actions().is_empty());
            assert!(!assignment.is_degraded());
        }
        assert_eq!(plan.total_recovered, RecoveredBudget::ZERO);
    }

    #[test]
    fn empty_input_is_vacuously_resolved() {
        let plan = execute_degradation(roomy_budget(), &[]);
        assert!(plan.resolved_within_budget);
        assert!(plan.assignments.is_empty());
        assert_eq!(plan.max_level(), 0);
    }

    #[test]
    fn degrades_lowest_priority_emitter_first() {
        // Only the particle dimension is over (1_200 vs a 1_000 ceiling); a
        // single ReduceSpawn on the lower-priority emitter clears it.
        let budget = TripleBudget {
            max_particles: 1_000,
            max_vram_bytes: 1_000_000,
            max_gpu_ms: 1_000.0,
        };
        let loads = [
            load(10, 5, EmitterRuntimeCost::new(600, 4_000, 4.0)), // high priority
            load(20, 1, EmitterRuntimeCost::new(600, 4_000, 4.0)), // low priority
        ];
        let plan = execute_degradation(budget, &loads);
        assert!(plan.resolved_within_budget);

        let low = plan.assignment_for(EmitterHandle(20)).unwrap();
        let high = plan.assignment_for(EmitterHandle(10)).unwrap();
        // The least-important emitter sheds first; the important one is untouched.
        assert_eq!(low.level, 1);
        assert_eq!(low.actions(), &[DegradationAction::ReduceSpawn]);
        assert_eq!(high.level, 0);
        // ReduceSpawn reclaims half the low emitter's 600 particles.
        assert_eq!(low.residual.particles, 300);
        assert_eq!(plan.total_recovered.particles, 300);
    }

    #[test]
    fn walks_rungs_one_at_a_time_until_pressure_clears() {
        // One emitter, only the GPU dimension over: 4.0 ms against a 2.0 ms
        // ceiling. Each rung reclaims a share of what remains, so it takes four
        // rungs to drop back under budget.
        let budget = TripleBudget {
            max_particles: 1_000_000,
            max_vram_bytes: 1_000_000_000,
            max_gpu_ms: 2.0,
        };
        let loads = [load(7, 1, EmitterRuntimeCost::new(100, 1_000, 4.0))];
        let plan = execute_degradation(budget, &loads);
        assert!(plan.resolved_within_budget);

        let only = plan.assignment_for(EmitterHandle(7)).unwrap();
        assert_eq!(only.level, 4);
        assert_eq!(
            only.actions(),
            &[
                DegradationAction::ReduceSpawn,
                DegradationAction::DisableSortingOit,
                DegradationAction::LowerVolumeResolution,
                DegradationAction::SimplifyShading,
            ]
        );
        // 4.0 -> 3.0 -> 2.55 -> 2.04 -> 1.53 ms of residual GPU cost.
        assert!(approx(only.residual.gpu_ms, 1.53));
        assert!(only.residual.gpu_ms <= budget.max_gpu_ms);
    }

    #[test]
    fn each_rung_reclaims_monotonically() {
        let cost = EmitterRuntimeCost::new(10_000, 1_000_000, 8.0);
        let mut previous = RecoveredBudget::ZERO;
        for level in 0..=DEGRADATION_LADDER.len() {
            let recovered = recovered_for_actions(cost, level);
            assert!(recovered.particles >= previous.particles);
            assert!(recovered.vram_bytes >= previous.vram_bytes);
            assert!(recovered.gpu_ms >= previous.gpu_ms - 1.0e-6);
            previous = recovered;
        }
        // The full ladder ends in PauseSimulation, reclaiming everything.
        let full = recovered_for_actions(cost, DEGRADATION_LADDER.len());
        assert_eq!(full.particles, cost.particles);
        assert_eq!(full.vram_bytes, cost.vram_bytes);
        assert!(approx(full.gpu_ms, cost.gpu_ms));
    }

    #[test]
    fn extreme_overspend_walks_to_pause_simulation() {
        // A tiny budget against a huge single emitter: only the final
        // PauseSimulation rung (which zeroes every dimension) can clear it.
        let budget = TripleBudget {
            max_particles: 1,
            max_vram_bytes: 1,
            max_gpu_ms: 0.5,
        };
        let loads = [load(3, 1, EmitterRuntimeCost::new(100_000, 50_000_000, 100.0))];
        let plan = execute_degradation(budget, &loads);
        assert!(plan.resolved_within_budget);

        let only = plan.assignment_for(EmitterHandle(3)).unwrap();
        assert_eq!(only.level, DEGRADATION_LADDER.len());
        assert_eq!(
            *only.actions().last().unwrap(),
            DegradationAction::PauseSimulation
        );
        assert_eq!(only.residual, EmitterRuntimeCost::ZERO);
    }

    #[test]
    fn multiple_emitters_degrade_in_priority_order() {
        // Enough overspend that the two least-important emitters must shed
        // before the frame clears; the highest-priority emitter stays untouched.
        let budget = TripleBudget {
            max_particles: 900,
            max_vram_bytes: 1_000_000,
            max_gpu_ms: 1_000.0,
        };
        let loads = [
            load(10, 9, EmitterRuntimeCost::new(600, 4_000, 2.0)), // highest priority
            load(20, 1, EmitterRuntimeCost::new(600, 4_000, 2.0)), // lowest priority
            load(30, 5, EmitterRuntimeCost::new(600, 4_000, 2.0)), // middle priority
        ];
        let plan = execute_degradation(budget, &loads);
        assert!(plan.resolved_within_budget);

        // Total particles 1_800 vs a 900 ceiling (only the particle dimension is
        // over). Only ReduceSpawn and PauseSimulation shed particles, so the
        // lowest-priority emitter is walked all the way to PauseSimulation before
        // the middle emitter's ReduceSpawn finally clears the overspend; the
        // highest-priority emitter is never touched.
        let highest = plan.assignment_for(EmitterHandle(10)).unwrap();
        let lowest = plan.assignment_for(EmitterHandle(20)).unwrap();
        assert_eq!(highest.level, 0);
        assert!(lowest.level >= 1);
        assert!(lowest.level >= highest.level);
    }

    #[test]
    fn deterministic_bit_identical_plans() {
        let budget = TripleBudget {
            max_particles: 1_000,
            max_vram_bytes: 20_000,
            max_gpu_ms: 6.0,
        };
        let loads = [
            load(1, 3, EmitterRuntimeCost::new(400, 8_000, 3.0)),
            load(2, 1, EmitterRuntimeCost::new(500, 9_000, 4.0)),
            load(3, 2, EmitterRuntimeCost::new(450, 7_000, 3.5)),
        ];
        let first = execute_degradation(budget, &loads);
        let second = execute_degradation(budget, &loads);
        assert_eq!(first, second);
    }

    #[test]
    fn candidate_from_world_uses_vector_distance() {
        let candidate = candidate_from_world(
            EmitterHandle(9),
            4,
            0.25,
            Vec3::new(3.0, 4.0, 0.0),
            Vec3::ZERO,
        );
        assert_eq!(candidate.handle, EmitterHandle(9));
        assert_eq!(candidate.priority, 4);
        assert!(approx(candidate.screen_coverage, 0.25));
        assert!(approx(candidate.distance, 5.0));
    }
}
