//! Real-device parity for the guide-strand groom sleep-gate twin:
//! [`GpuHairSleepUpdate`](prism_volumetric_gpu::hair_sleep_update::GpuHairSleepUpdate)
//! must reproduce the `CPU` golden `update_sleep` of
//! `prism_render_architecture::hair::sleep`.
//!
//! The oracle here is an independent re-implementation of that closed form —
//! the wake-takes-priority test `!is_finite(e) || e >= wake_above`, the
//! already-asleep hold, the quiet accumulation `is_finite(e) && e <=
//! sleep_below`, the saturating quiet-frame counter and the
//! `quiet_frames >= max(frames_to_sleep, 1)` sleep latch, plus the dead-band
//! reset to the awake state — written out directly so the test never imports
//! `prism_render_architecture` or `prism_physics_core`.
//!
//! The fixtures cover a quiet run that latches to sleep, an asleep groom
//! holding its state, a dead-band frame that clears an awake groom's progress,
//! a `NaN` motion that wakes, a `+inf` motion that wakes, the inclusive
//! boundary `motion == wake_above` (confirms `>=` wakes), the inclusive
//! boundary `motion == sleep_below` (confirms `<=` counts quiet) and the
//! `frames_to_sleep == 0` lift to `1`. A mixed batch validates the `std430`
//! stride, a `512`-query `LCG` sweep (with `NaN` injection) follows, and an
//! empty batch the host short-circuits with no dispatch.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//!
//! # Parity criterion
//!
//! Every output channel (`asleep`, `quiet_frames`, `valid`) is discrete and is
//! compared exactly with `assert_eq!`; there is no continuous output. The
//! random sweep keeps finite samples clear of the exact `motion == wake_above`
//! and `motion == sleep_below` ties so a host/device ordering difference on the
//! inclusive compares cannot flip a discrete channel; dedicated named fixtures
//! pin those exact-boundary cases.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::hair::sleep`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::hair_sleep_update::{
    GpuHairSleepUpdate, HairSleepUpdateQuery, HairSleepUpdateResult,
};
use prism_volumetric_gpu::GpuContext;

/// Independent host re-implementation of `hair::sleep::update_sleep`, flattened
/// into the `(asleep, quiet_frames, valid)` triple the twin encodes. No
/// threshold sanitization, matching the reference verbatim.
fn oracle(q: &HairSleepUpdateQuery) -> HairSleepUpdateResult {
    let energy = q.motion_energy;
    let finite = energy.is_finite();
    let energetic = !finite || energy >= q.wake_above;
    if energetic {
        return awake();
    }
    if q.asleep == 1 {
        // Asleep and not energetic enough to wake: hold the incoming state.
        return HairSleepUpdateResult {
            asleep: 1,
            quiet_frames: q.quiet_frames,
            valid: 1,
        };
    }
    let quiet = finite && energy <= q.sleep_below;
    if quiet {
        let quiet_frames = q.quiet_frames.saturating_add(1);
        let asleep = u32::from(quiet_frames >= q.frames_to_sleep.max(1));
        HairSleepUpdateResult {
            asleep,
            quiet_frames,
            valid: 1,
        }
    } else {
        awake()
    }
}

/// The awake reset state: `asleep = 0`, `quiet_frames = 0`, `valid = 1`.
fn awake() -> HairSleepUpdateResult {
    HairSleepUpdateResult {
        asleep: 0,
        quiet_frames: 0,
        valid: 1,
    }
}

/// Asserts the GPU result for one query matches the oracle. Every channel is a
/// discrete `u32`, so all three are compared with exact equality.
fn assert_result(got: HairSleepUpdateResult, want: HairSleepUpdateResult, label: &str) {
    assert_eq!(got.valid, want.valid, "valid mismatch: {label}");
    assert_eq!(got.asleep, want.asleep, "asleep mismatch: {label}");
    assert_eq!(
        got.quiet_frames, want.quiet_frames,
        "quiet_frames mismatch: {label}"
    );
}

/// Asserts a single-query GPU result matches the oracle.
fn assert_parity(ctx: &GpuContext, gpu: &GpuHairSleepUpdate, q: HairSleepUpdateQuery, label: &str) {
    let got = gpu.evaluate(ctx, std::slice::from_ref(&q));
    assert_eq!(got.len(), 1, "one result per query: {label}");
    assert_result(got[0], oracle(&q), label);
}

#[test]
fn quiet_run_latches_to_sleep() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairSleepUpdate::new(&ctx);
    // Awake with two quiet frames already banked; a third quiet frame (energy
    // 0.1 <= sleep_below 0.5) reaches frames_to_sleep = 3 and latches asleep.
    let q = HairSleepUpdateQuery::new(0, 2, 0.1, 0.5, 1.0, 3);
    let want = oracle(&q);
    assert_eq!(want.asleep, 1, "fixture sanity: should latch asleep");
    assert_eq!(want.quiet_frames, 3, "fixture sanity: counter advances");
    assert_parity(&ctx, &gpu, q, "quiet_run_latches_to_sleep");
}

#[test]
fn asleep_holds_in_dead_band() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairSleepUpdate::new(&ctx);
    // Asleep with energy 0.7 in the dead band (sleep_below 0.5, wake_above 1.0):
    // not energetic, so the groom holds its {asleep, quiet_frames} unchanged.
    let q = HairSleepUpdateQuery::new(1, 5, 0.7, 0.5, 1.0, 3);
    let want = oracle(&q);
    assert_eq!(want.asleep, 1, "fixture sanity: stays asleep");
    assert_eq!(want.quiet_frames, 5, "fixture sanity: counter held");
    assert_parity(&ctx, &gpu, q, "asleep_holds_in_dead_band");
}

#[test]
fn dead_band_clears_awake_progress() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairSleepUpdate::new(&ctx);
    // Awake with four quiet frames banked, but energy 0.7 is in the dead band:
    // neither energetic nor quiet, so progress resets to the awake state.
    let q = HairSleepUpdateQuery::new(0, 4, 0.7, 0.5, 1.0, 6);
    let want = oracle(&q);
    assert_eq!(want.asleep, 0, "fixture sanity: stays awake");
    assert_eq!(want.quiet_frames, 0, "fixture sanity: counter cleared");
    assert_parity(&ctx, &gpu, q, "dead_band_clears_awake_progress");
}

#[test]
fn nan_motion_wakes() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairSleepUpdate::new(&ctx);
    // A NaN motion energy counts as motion, so an asleep groom wakes and the
    // counter clears; a NaN can never latch a groom asleep.
    let q = HairSleepUpdateQuery::new(1, 3, f32::NAN, 0.5, 1.0, 3);
    let want = oracle(&q);
    assert_eq!(want.asleep, 0, "fixture sanity: NaN wakes");
    assert_eq!(want.quiet_frames, 0, "fixture sanity: counter cleared");
    assert_parity(&ctx, &gpu, q, "nan_motion_wakes");
}

#[test]
fn infinite_motion_wakes() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairSleepUpdate::new(&ctx);
    // Positive infinity is non-finite and so counts as motion: the groom wakes.
    let q = HairSleepUpdateQuery::new(1, 2, f32::INFINITY, 0.5, 1.0, 3);
    let want = oracle(&q);
    assert_eq!(want.asleep, 0, "fixture sanity: inf wakes");
    assert_parity(&ctx, &gpu, q, "infinite_motion_wakes");
}

#[test]
fn energy_equal_wake_above_wakes() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairSleepUpdate::new(&ctx);
    // Energy exactly equal to wake_above wakes, confirming the inclusive `>=`.
    let q = HairSleepUpdateQuery::new(1, 2, 1.0, 0.5, 1.0, 3);
    let want = oracle(&q);
    assert_eq!(want.asleep, 0, "fixture sanity: >= wakes at equality");
    assert_parity(&ctx, &gpu, q, "energy_equal_wake_above_wakes");
}

#[test]
fn energy_equal_sleep_below_counts_quiet() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairSleepUpdate::new(&ctx);
    // Energy exactly equal to sleep_below counts as quiet, confirming the
    // inclusive `<=`; with frames_to_sleep = 1 it latches on this first frame.
    let q = HairSleepUpdateQuery::new(0, 0, 0.5, 0.5, 1.0, 1);
    let want = oracle(&q);
    assert_eq!(want.asleep, 1, "fixture sanity: <= counts quiet, latches");
    assert_eq!(want.quiet_frames, 1, "fixture sanity: counter advances");
    assert_parity(&ctx, &gpu, q, "energy_equal_sleep_below_counts_quiet");
}

#[test]
fn frames_to_sleep_zero_lifts_to_one() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairSleepUpdate::new(&ctx);
    // frames_to_sleep = 0 is lifted to 1 by max(_, 1), so one quiet frame
    // suffices to sleep.
    let q = HairSleepUpdateQuery::new(0, 0, 0.1, 0.5, 1.0, 0);
    let want = oracle(&q);
    assert_eq!(want.asleep, 1, "fixture sanity: zero lifts to one");
    assert_eq!(want.quiet_frames, 1, "fixture sanity: counter advances");
    assert_parity(&ctx, &gpu, q, "frames_to_sleep_zero_lifts_to_one");
}

#[test]
fn mixed_batch_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairSleepUpdate::new(&ctx);
    // A >=2-element batch mixing every branch validates the std430 stride end
    // to end: latch-to-sleep, asleep hold, dead-band clear, NaN wake, the
    // wake_above boundary and the sleep_below boundary.
    let queries = vec![
        HairSleepUpdateQuery::new(0, 2, 0.1, 0.5, 1.0, 3),
        HairSleepUpdateQuery::new(1, 5, 0.7, 0.5, 1.0, 3),
        HairSleepUpdateQuery::new(0, 4, 0.7, 0.5, 1.0, 6),
        HairSleepUpdateQuery::new(1, 3, f32::NAN, 0.5, 1.0, 3),
        HairSleepUpdateQuery::new(1, 2, 1.0, 0.5, 1.0, 3),
        HairSleepUpdateQuery::new(0, 0, 0.5, 0.5, 1.0, 1),
        HairSleepUpdateQuery::new(0, 0, 0.1, 0.5, 1.0, 0),
    ];
    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (i, (q, r)) in queries.iter().zip(results.iter()).enumerate() {
        assert_result(*r, oracle(q), &format!("mixed batch index {i}"));
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairSleepUpdate::new(&ctx);
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

    /// A `[lo, hi]` integer.
    fn next_u32_range(&mut self, lo: u32, hi: u32) -> u32 {
        lo + self.next_u32() % (hi - lo + 1)
    }
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairSleepUpdate::new(&ctx);
    let mut rng = Lcg::new(0x5A_3C_71_E9);
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        let asleep = rng.next_u32() & 1;
        let quiet_frames = rng.next_u32_range(0, 10);
        let sleep_below = rng.next_range(0.0, 1.0);
        let wake_above = rng.next_range(0.0, 1.0);
        let frames_to_sleep = rng.next_u32_range(1, 8);

        // Every tenth sample injects a non-finite motion energy. NaN and inf
        // are unambiguously energetic for both host and device, so they need no
        // tie rejection.
        let inject = rng.next_u32() % 10 == 0;
        let motion_energy = if inject {
            if rng.next_u32() & 1 == 0 {
                f32::NAN
            } else {
                f32::INFINITY
            }
        } else {
            let m = rng.next_range(-1.0, 2.0);
            // Reject samples sitting on either inclusive-compare knee so the
            // discrete channels cannot flip on a rounding tie between host and
            // device.
            if (m - wake_above).abs() < 1e-2 || (m - sleep_below).abs() < 1e-2 {
                continue;
            }
            m
        };

        queries.push(HairSleepUpdateQuery::new(
            asleep,
            quiet_frames,
            motion_energy,
            sleep_below,
            wake_above,
            frames_to_sleep,
        ));
    }

    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (i, (q, r)) in queries.iter().zip(results.iter()).enumerate() {
        assert_result(*r, oracle(q), &format!("sweep index {i}"));
    }
}
