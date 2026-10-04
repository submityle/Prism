//! Real-device parity for the cloth sleep-gate twin:
//! [`GpuClothSleepUpdate`](prism_volumetric_gpu::cloth_sleep_update::GpuClothSleepUpdate)
//! must reproduce the `CPU` golden
//! `prism_render_architecture::cloth::sleep::SleepTracker::update` (together with
//! `SleepParams::sanitized`), the hysteretic gate that puts a garment that has
//! come to rest to sleep so it stops consuming the per-frame deformation budget.
//!
//! The oracle here is an independent re-implementation of that closed form — the
//! `clamp_non_negative` sanitize of the two thresholds, the lift of the wake
//! threshold to at least the sleep threshold, and the wake-priority /
//! quiet-accumulate / dead-band-reset state machine — written out directly so
//! the test never imports `prism_render_architecture` or `prism_physics_core`.
//! It mirrors the reference branch for branch, including the strict `>` / `<`
//! threshold compares (distinct from the `>=` / `<=` gates elsewhere), the
//! `NaN`-indicator-is-energetic rule that forbids latching a garment asleep, and
//! the `frames_to_sleep.max(1)` dwell that still needs one quiet frame when the
//! authored dwell is zero.
//!
//! The per-scene `max_kinetic_indicator` reduction and the rest of the module
//! are not twinned: this kernel is the pure per-garment transition, one thread
//! per tracker.
//!
//! The fixtures cover the branches the kernel must honor: an awake quiet run
//! that reaches the dwell and sleeps, an energetic frame that wakes a sleeper, a
//! dead-band frame that resets the quiet counter, a `NaN` indicator that wakes a
//! sleeper, a `sanitized` lift of the wake threshold above the sleep threshold,
//! indicators landing exactly on each threshold that confirm the strict
//! compares do not fire, a zero dwell that still needs one quiet frame, and a
//! mixed multi-element batch that validates the `std430` stride. A `512`-step
//! sweep over random state, quiet counter, indicator (with occasional `NaN`
//! injection), thresholds and dwell follows, plus an empty batch the host
//! short-circuits with no dispatch.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Every output is discrete (`state`, `quiet_frames`, `valid`), so parity is an
//! exact integer comparison; the `f32` thresholds only steer branches. The
//! sweep keeps the indicator a safe margin away from the sanitized `lin'` and
//! `wake'` compare points so `f32` noise never flips a strict `<` / `>` branch,
//! and injects `NaN` indicators directly (always energetic, so no knee to
//! avoid). `valid` is always `1`: the golden has no rejected input.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::cloth::sleep`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::cloth_sleep_update::{ClothSleepUpdateQuery, GpuClothSleepUpdate};
use prism_volumetric_gpu::GpuContext;

/// Largest magnitude accepted as finite, matching the kernel and golden.
const F32_MAX_FINITE: f32 = 3.4e38;

/// Portable finiteness test, replicated identically to the kernel: `x == x` is
/// the `NaN` self-compare (a `NaN` is the only value unequal to itself) and the
/// magnitude guard rejects the two infinities.
fn is_finite(x: f32) -> bool {
    (x == x) && (x.abs() < F32_MAX_FINITE)
}

/// Clamps a scalar to `[0, inf)`, mapping negatives and non-finite values to
/// `0`, mirroring the golden `clamp_non_negative`.
fn clamp_non_negative(v: f32) -> f32 {
    if is_finite(v) && v > 0.0 {
        v
    } else {
        0.0
    }
}

/// Independent oracle for one query, returning `(state, quiet_frames, valid)`.
///
/// Reproduces `SleepParams::sanitized` then `SleepTracker::update`: wake takes
/// priority, an awake garment accumulates quiet frames strictly below the sleep
/// threshold and sleeps at `max(frames_to_sleep, 1)`, and any other frame
/// resets the counter. `valid` is always `1`.
fn oracle(q: &ClothSleepUpdateQuery) -> (u32, u32, u32) {
    let lin = clamp_non_negative(q.linear_threshold);
    let wake_raw = clamp_non_negative(q.wake_threshold);
    let wake = if wake_raw > lin { wake_raw } else { lin };

    let mut st = q.state;
    let mut quiet = q.quiet_frames;

    // Wake takes priority: a non-finite indicator or one strictly above the
    // wake threshold is energetic, so a NaN can never latch a garment asleep.
    let energetic = !is_finite(q.indicator) || q.indicator > wake;

    if st == 1 {
        // Sleeping: only an energetic frame wakes it; otherwise hold.
        if energetic {
            st = 0;
            quiet = 0;
        }
    } else {
        // Awake.
        if energetic {
            quiet = 0;
        } else if is_finite(q.indicator) && q.indicator < lin {
            // Quiet frame: saturating increment, then sleep at the dwell.
            quiet = quiet.saturating_add(1);
            if quiet >= q.frames_to_sleep.max(1) {
                st = 1;
            }
        } else {
            // Dead band: awake, but making no progress toward sleep.
            quiet = 0;
        }
    }

    (st, quiet, 1)
}

/// Asserts GPU-vs-oracle parity for a single query. Every output is discrete,
/// so all three fields are compared exactly.
fn assert_parity(ctx: &GpuContext, gpu: &GpuClothSleepUpdate, q: ClothSleepUpdateQuery) {
    let results = gpu.evaluate(ctx, &[q]);
    assert_eq!(results.len(), 1, "one result per query");
    let r = results[0];
    let (state, quiet, valid) = oracle(&q);
    assert_eq!(r.valid, valid, "valid mismatch: query={q:?}");
    assert_eq!(r.state, state, "state mismatch: query={q:?}");
    assert_eq!(r.quiet_frames, quiet, "quiet_frames mismatch: query={q:?}");
}

/// Golden-leaning thresholds reused across several named fixtures: a tiny rest
/// threshold with a wake threshold two orders of magnitude higher.
const LIN: f32 = 1.0e-4;
const WAKE: f32 = 1.0e-2;

#[test]
fn awake_quiet_run_reaches_dwell_and_sleeps() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothSleepUpdate::new(&ctx);
    // Awake with two quiet frames already banked; a third quiet frame (indicator
    // 0, strictly below the sleep threshold) reaches the dwell of 3 and sleeps.
    assert_parity(
        &ctx,
        &gpu,
        ClothSleepUpdateQuery::new(0, 2, 0.0, LIN, 3, WAKE),
    );
}

#[test]
fn energetic_frame_wakes_sleeper() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothSleepUpdate::new(&ctx);
    // A sleeping garment with a long quiet history wakes the instant its
    // indicator (1.0) crosses strictly above the wake threshold, clearing quiet.
    assert_parity(
        &ctx,
        &gpu,
        ClothSleepUpdateQuery::new(1, 9, 1.0, LIN, 3, WAKE),
    );
}

#[test]
fn dead_band_frame_resets_quiet() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothSleepUpdate::new(&ctx);
    // Indicator 1e-3 sits in the dead band (above sleep 1e-4, below wake 1e-2):
    // an awake garment neither sleeps nor wakes, but resets its quiet counter.
    assert_parity(
        &ctx,
        &gpu,
        ClothSleepUpdateQuery::new(0, 5, 1.0e-3, LIN, 3, WAKE),
    );
}

#[test]
fn nan_indicator_wakes_sleeper() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothSleepUpdate::new(&ctx);
    // A non-finite indicator reads as energetic, so a NaN can never latch a
    // garment asleep: the sleeper wakes and clears its quiet counter.
    assert_parity(
        &ctx,
        &gpu,
        ClothSleepUpdateQuery::new(1, 5, f32::NAN, LIN, 3, WAKE),
    );
}

#[test]
fn sanitize_lifts_wake_above_sleep() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothSleepUpdate::new(&ctx);
    // Authored with wake (1e-4) below sleep (1e-2); sanitize lifts wake to the
    // sleep threshold. Indicator 5e-3 is then NOT energetic (<= lifted wake 1e-2)
    // and IS quiet (< sleep 1e-2), so quiet increments to 1. Were the wake not
    // lifted, 5e-3 > 1e-4 would read as energetic and clear quiet instead.
    assert_parity(
        &ctx,
        &gpu,
        ClothSleepUpdateQuery::new(0, 0, 5.0e-3, 1.0e-2, 3, 1.0e-4),
    );
}

#[test]
fn indicator_equal_sleep_threshold_is_not_quiet() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothSleepUpdate::new(&ctx);
    // Indicator exactly equal to the sleep threshold is NOT strictly below it,
    // so the quiet branch does not fire; it falls into the dead band and resets.
    assert_parity(
        &ctx,
        &gpu,
        ClothSleepUpdateQuery::new(0, 4, 0.2, 0.2, 3, 0.5),
    );
}

#[test]
fn indicator_equal_wake_threshold_is_not_energetic() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothSleepUpdate::new(&ctx);
    // Indicator exactly equal to the wake threshold is NOT strictly above it, so
    // a sleeper is not woken and simply holds its state and quiet counter.
    assert_parity(
        &ctx,
        &gpu,
        ClothSleepUpdateQuery::new(1, 3, 0.5, 0.1, 3, 0.5),
    );
}

#[test]
fn zero_dwell_still_needs_one_quiet_frame() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothSleepUpdate::new(&ctx);
    // frames_to_sleep == 0 is treated as at least one: the first quiet frame
    // takes quiet to 1, meets the max(0, 1) dwell, and sleeps.
    assert_parity(
        &ctx,
        &gpu,
        ClothSleepUpdateQuery::new(0, 0, 0.0, LIN, 0, WAKE),
    );
}

#[test]
fn mixed_batch_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothSleepUpdate::new(&ctx);
    // A multi-element batch exercises the std430 stride: adjacent slots must
    // decode independently and in order, spanning both states, several quiet
    // counters, finite and NaN indicators, and distinct thresholds/dwells.
    let queries = [
        ClothSleepUpdateQuery::new(0, 2, 0.0, LIN, 3, WAKE),
        ClothSleepUpdateQuery::new(1, 9, 1.0, LIN, 3, WAKE),
        ClothSleepUpdateQuery::new(0, 5, 1.0e-3, LIN, 3, WAKE),
        ClothSleepUpdateQuery::new(1, 5, f32::NAN, LIN, 3, WAKE),
        ClothSleepUpdateQuery::new(0, 0, 5.0e-3, 1.0e-2, 3, 1.0e-4),
        ClothSleepUpdateQuery::new(0, 0, 0.0, LIN, 0, WAKE),
    ];
    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(results.iter()) {
        let (state, quiet, valid) = oracle(q);
        assert_eq!(r.valid, valid, "batch valid mismatch: query={q:?}");
        assert_eq!(r.state, state, "batch state mismatch: query={q:?}");
        assert_eq!(
            r.quiet_frames, quiet,
            "batch quiet_frames mismatch: query={q:?}"
        );
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothSleepUpdate::new(&ctx);
    assert!(
        gpu.evaluate(&ctx, &[]).is_empty(),
        "empty batch returns an empty vector with no dispatch"
    );
}

/// A small deterministic linear-congruential generator so the sweep needs no
/// external randomness. Constants are the Numerical Recipes values.
struct Lcg {
    state: u32,
}

impl Lcg {
    fn new(seed: u32) -> Self {
        Lcg { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        self.state
    }

    /// A `[0, 1)` fraction built from the top bits, keeping the fixture pure
    /// integer host-side with no transcendental call.
    fn next_unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }

    /// A `[lo, hi)` fraction.
    fn next_range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_unit()
    }
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothSleepUpdate::new(&ctx);
    let mut rng = Lcg::new(0x5E_3A_C1_07);
    // Margin kept well above any f32 noise so the indicator never sits on a
    // strict `<` / `>` knee at the sanitized sleep or wake compare point.
    const KNEE_MARGIN: f32 = 0.01;
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        let state = rng.next_u32() % 2;
        let quiet = rng.next_u32() % 11; // [0, 10]
        let lin = rng.next_range(0.0, 1.0);
        let wake = rng.next_range(0.0, 1.0);
        let frames = 1 + rng.next_u32() % 8; // [1, 8]

        // Occasionally inject a NaN indicator: always energetic, so there is no
        // threshold knee to avoid and it can be pushed directly.
        if rng.next_u32() % 17 == 0 {
            queries.push(ClothSleepUpdateQuery::new(
                state,
                quiet,
                f32::NAN,
                lin,
                frames,
                wake,
            ));
            continue;
        }

        let indicator = rng.next_range(-1.0, 2.0);
        // The sanitized compare points: lin' = lin (already non-negative) and
        // wake' = max(wake, lin). Reject indicators within the margin of either.
        let wake_eff = wake.max(lin);
        if (indicator - lin).abs() < KNEE_MARGIN || (indicator - wake_eff).abs() < KNEE_MARGIN {
            continue;
        }
        queries.push(ClothSleepUpdateQuery::new(
            state, quiet, indicator, lin, frames, wake,
        ));
    }

    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(results.iter()) {
        let (state, quiet, valid) = oracle(q);
        assert_eq!(r.valid, valid, "sweep valid mismatch: query={q:?}");
        assert_eq!(r.state, state, "sweep state mismatch: query={q:?}");
        assert_eq!(
            r.quiet_frames, quiet,
            "sweep quiet_frames mismatch: query={q:?}"
        );
    }
}
