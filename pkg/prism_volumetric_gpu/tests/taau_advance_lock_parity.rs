//! Real-device parity for the thin-feature lock-transition numeric-core twin:
//! [`GpuTaauAdvanceLock`](prism_volumetric_gpu::taau_advance_lock::GpuTaauAdvanceLock)
//! must reproduce the `CPU` golden single-frame transition
//! [`advance_lock`](prism_render_architecture::temporal_upscale::lock::advance_lock)
//! across disocclusion, luma-break, decay, expiry, create/refresh, unlocked and
//! randomized fixtures.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The golden
//! [`advance_lock`](prism_render_architecture::temporal_upscale::lock::advance_lock)
//! is public, so it is called directly as the oracle: each `GPU` result is
//! pinned against the golden evaluated on the identical previous
//! [`LockState`](prism_render_architecture::temporal_upscale::lock::LockState)
//! and this frame's `current_luma`, `thin_strength` and `disoccluded` verdict.
//! The golden's
//! [`is_locked`](prism_render_architecture::temporal_upscale::lock::LockState::is_locked)
//! flag is compared exactly against the `locked` word the kernel derives.
//!
//! # Parity criterion
//!
//! The next `lifetime` is one of `0`, the literal `INITIAL_LOCK_LIFETIME`, or a
//! single `previous_lifetime - 1` subtract, and the next `luma` is a copy of
//! `0`, `previous_luma` or `current_luma`, so the two engines agree to within a
//! legal last-place difference; both scalars are asserted within
//! `abs_diff <= 1e-4` or `rel_diff <= 1e-3`. The `locked` flag is a discrete
//! decision and is asserted exactly (`==`).
//!
//! # Conditioning
//!
//! The only comparison whose two sides are *computed* rather than copied inputs
//! is the luma-drift break `drift <= tolerance`; fixtures and the randomized
//! sweep keep that margin well clear of zero so `CPU` and `GPU` cannot straddle
//! it. The `thin_strength >= LOCK_CREATION_THRESHOLD` test and the
//! `previous_lifetime > 0` / `lifetime - 1 > 0` tests compare input-identical
//! operands, so they cannot diverge, but the sweep still keeps `thin_strength`
//! a clear margin off its threshold for good measure. The randomized sweep
//! draws from a host-side integer generator, with no transcendental method,
//! matching the house rules.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::temporal_upscale::lock`；无第三方引擎源码或衍生代码。

use prism_render_architecture::temporal_upscale::lock::{
    advance_lock, LockState, INITIAL_LOCK_LIFETIME, LOCK_BREAK_LUMA_TOLERANCE,
    LOCK_CREATION_THRESHOLD,
};
use prism_volumetric_gpu::taau_advance_lock::{
    GpuTaauAdvanceLock, TaauAdvanceLockQuery, TaauAdvanceLockResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on a scalar. A `GPU` subtract may land a few units in
/// the last place from the scalar reference; `1e-4` admits that legal slack
/// while still failing a wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in the
/// last place exceed the absolute floor.
const REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
}

/// Evaluates the golden transition on the same inputs the query carries,
/// building the previous [`LockState`] from the query's scalar fields.
fn oracle(q: &TaauAdvanceLockQuery) -> TaauAdvanceLockResult {
    let previous = LockState {
        lifetime: q.previous_lifetime,
        luma: q.previous_luma,
    };
    let next = advance_lock(previous, q.current_luma, q.thin_strength, q.disoccluded);
    TaauAdvanceLockResult {
        lifetime: next.lifetime,
        luma: next.luma,
        locked: next.is_locked(),
    }
}

/// Pins one `GPU` pixel result against the golden oracle: both scalars within
/// tolerance and the discrete `locked` flag exact.
fn check_pixel(idx: usize, got: &TaauAdvanceLockResult, want: &TaauAdvanceLockResult) {
    assert!(
        close(got.lifetime, want.lifetime),
        "pixel {idx} lifetime: gpu {} vs cpu {}",
        got.lifetime,
        want.lifetime
    );
    assert!(
        close(got.luma, want.luma),
        "pixel {idx} luma: gpu {} vs cpu {}",
        got.luma,
        want.luma
    );
    assert_eq!(
        got.locked, want.locked,
        "pixel {idx} locked: gpu {} vs cpu {}",
        got.locked, want.locked
    );
}

/// Dispatches every pixel and pins each result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuTaauAdvanceLock, queries: &[TaauAdvanceLockQuery]) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (q, result)) in queries.iter().zip(got.iter()).enumerate() {
        let want = oracle(q);
        check_pixel(idx, result, &want);
    }
}

/// A tiny integer linear-congruential generator; only integer work, so no
/// transcendental appears. Returns the raw high bits as a `u32`.
fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 33) as u32
}

/// Draws a scalar in `[0.0, 1.0)` at milli resolution from `state`.
fn unit(state: &mut u64) -> f32 {
    (lcg(state) % 1000) as f32 / 1000.0
}

/// Draws a previous lifetime across three regimes: unlocked (`0`), a short lock
/// that will decay to zero this frame (`[0.3, 0.8)`), or a sustaining lock
/// (`[1.3, 4.5)`). Every value is a copy, so both engines branch on it
/// identically.
fn previous_lifetime(state: &mut u64) -> f32 {
    match lcg(state) % 3 {
        0 => 0.0,
        1 => 0.3 + (lcg(state) % 500) as f32 / 1000.0,
        _ => 1.3 + (lcg(state) % 3200) as f32 / 1000.0,
    }
}

/// Whether a random sample is clear of the one computed-vs-computed guard, the
/// luma-drift break. Replicates the golden `drift`/`tolerance` host-side with
/// only `+`/`*`/`abs` and keeps a clear margin on either side so `CPU` and
/// `GPU` stay on the same side of it. Also keeps `thin_strength` a clear margin
/// off its creation threshold.
fn well_conditioned(previous_luma: f32, current_luma: f32, thin_strength: f32) -> bool {
    let drift = (current_luma - previous_luma).abs();
    let tolerance = LOCK_BREAK_LUMA_TOLERANCE * (previous_luma.abs() + 1.0e-3);
    (drift - tolerance).abs() >= 0.02 && (thin_strength - LOCK_CREATION_THRESHOLD).abs() >= 0.02
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping taau_advance_lock parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuTaauAdvanceLock::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn disocclusion_drops_a_live_lock() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTaauAdvanceLock::new(&ctx);
    // A disoccluded pixel invalidates its past: the live lock is dropped and the
    // unlocked default is emitted regardless of a strong thin feature.
    let q = TaauAdvanceLockQuery::new(3.0, 0.5, 0.5, 0.5, true);
    check(&ctx, &gpu, &[q]);
}

#[test]
fn thin_feature_creates_a_fresh_lock() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTaauAdvanceLock::new(&ctx);
    // An unlocked pixel with a strong thin feature creates a full-lifetime lock
    // capturing the current luma.
    let q = TaauAdvanceLockQuery::new(0.0, 0.0, 0.7, 0.5, false);
    check(&ctx, &gpu, &[q]);
}

#[test]
fn thin_feature_supersedes_a_decaying_lock() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTaauAdvanceLock::new(&ctx);
    // A fresh thin feature wins over a surviving decayed lock: the lifetime is
    // refreshed to the full value and the luma is recaptured.
    let q = TaauAdvanceLockQuery::new(3.0, 0.5, 0.5, 0.8, false);
    check(&ctx, &gpu, &[q]);
}

#[test]
fn stable_lock_decays_by_one_frame() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTaauAdvanceLock::new(&ctx);
    // A lock whose luma is unchanged and with no thin feature loses one frame of
    // lifetime and keeps its captured luma.
    let q = TaauAdvanceLockQuery::new(3.0, 0.5, 0.5, 0.0, false);
    check(&ctx, &gpu, &[q]);
}

#[test]
fn lock_expires_at_zero_lifetime() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTaauAdvanceLock::new(&ctx);
    // A one-frame lock decays to zero lifetime and so becomes unlocked.
    let q = TaauAdvanceLockQuery::new(1.0, 0.5, 0.5, 0.0, false);
    check(&ctx, &gpu, &[q]);
}

#[test]
fn shading_change_breaks_a_lock() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTaauAdvanceLock::new(&ctx);
    // A luma that departs far past the relative tolerance drops the lock even
    // though its lifetime had not expired.
    let q = TaauAdvanceLockQuery::new(3.0, 0.5, 2.0, 0.0, false);
    check(&ctx, &gpu, &[q]);
}

#[test]
fn unlocked_pixel_without_feature_stays_unlocked() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTaauAdvanceLock::new(&ctx);
    // An unlocked pixel with no thin feature stays unlocked.
    let q = TaauAdvanceLockQuery::new(0.0, 0.0, 0.3, 0.1, false);
    check(&ctx, &gpu, &[q]);
}

#[test]
fn mixed_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTaauAdvanceLock::new(&ctx);
    // A deterministic batch spanning disocclusion, create, supersede, decay,
    // expiry, break and unlocked cases, dispatched together so the per-thread
    // indexing and the contiguous output slots are both exercised.
    let queries = [
        TaauAdvanceLockQuery::new(3.0, 0.5, 0.5, 0.5, true),
        TaauAdvanceLockQuery::new(0.0, 0.0, 0.7, 0.5, false),
        TaauAdvanceLockQuery::new(3.0, 0.5, 0.5, 0.8, false),
        TaauAdvanceLockQuery::new(3.0, 0.5, 0.5, 0.0, false),
        TaauAdvanceLockQuery::new(1.0, 0.5, 0.5, 0.0, false),
        TaauAdvanceLockQuery::new(3.0, 0.5, 2.0, 0.0, false),
        TaauAdvanceLockQuery::new(0.0, 0.0, 0.3, 0.1, false),
        TaauAdvanceLockQuery::new(INITIAL_LOCK_LIFETIME, 0.3, 0.3, 0.0, false),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTaauAdvanceLock::new(&ctx);
    let mut state = 0x0f1e_2d3c_4b5a_6978_u64;
    let mut queries = Vec::new();
    // Several workgroups' worth of random pixels across the three lifetime
    // regimes, a mix of thin strengths and disocclusion verdicts, each clear of
    // the luma-drift guard tie.
    let mut made = 0u32;
    let mut tries = 0u32;
    while made < 256 && tries < 20_000 {
        tries += 1;
        let prev_life = previous_lifetime(&mut state);
        let prev_luma = unit(&mut state);
        let cur_luma = unit(&mut state);
        let thin = unit(&mut state);
        let disoccluded = lcg(&mut state).is_multiple_of(2);
        if !well_conditioned(prev_luma, cur_luma, thin) {
            continue;
        }
        queries.push(TaauAdvanceLockQuery::new(
            prev_life,
            prev_luma,
            cur_luma,
            thin,
            disoccluded,
        ));
        made += 1;
    }
    assert!(
        made >= 256,
        "expected 256 well-conditioned pixels, got {made}"
    );
    check(&ctx, &gpu, &queries);
}
