//! Real-device parity for the thin-feature lock numeric-core twin:
//! [`GpuTaauThinFeatureLock`](prism_volumetric_gpu::taau_thin_feature_lock::GpuTaauThinFeatureLock)
//! must reproduce the `CPU` golden
//! [`lock`](prism_render_architecture::temporal_upscale::lock) closed forms —
//! the thin-feature strength of a cross neighborhood and the rejection scale of
//! a lock state — across bright-line, dark-point, edge, flat, locked, unlocked
//! and randomized fixtures.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The golden's
//! [`thin_feature_strength`](prism_render_architecture::temporal_upscale::lock::thin_feature_strength)
//! and
//! [`rejection_scale`](prism_render_architecture::temporal_upscale::lock::rejection_scale)
//! are public, so they are called directly as the oracle: each `GPU` result is
//! pinned against the golden evaluated on the identical inputs. The lock state
//! is built through the public
//! [`LockState`](prism_render_architecture::temporal_upscale::lock::LockState)
//! fields, and its
//! [`is_locked`](prism_render_architecture::temporal_upscale::lock::LockState::is_locked)
//! flag is the `locked` word the query carries, so the `GPU` and the golden take
//! the same branch.
//!
//! # Parity criterion
//!
//! Both scalars thread through subtracts, a divide and `min`/`max`/`clamp`, so a
//! `GPU` divide may land a few units in the last place from the scalar
//! reference; they are asserted within `abs_diff <= 1e-4` or `rel_diff <= 1e-3`.
//!
//! # Conditioning
//!
//! Fixtures keep the local luma `range` and the `bright`/`dark` separation well
//! clear of zero (or deliberately at zero for the non-feature cases), and the
//! lock `lifetime` well clear of the `0` unlocked boundary, so `CPU` and `GPU`
//! stay on the same side of every guard. The randomized sweep draws its colors
//! and lifetimes from a host-side integer generator, with no transcendental
//! method, matching the house rules.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::temporal_upscale::lock`；无第三方引擎源码或衍生代码。

use prism_render_architecture::temporal_upscale::lock::{
    rejection_scale, thin_feature_strength, LockState, INITIAL_LOCK_LIFETIME,
};
use prism_volumetric_gpu::taau_thin_feature_lock::{
    GpuTaauThinFeatureLock, TaauThinFeatureLockQuery, TaauThinFeatureLockResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on a scalar. A `GPU` divide may land a few units in the
/// last place from the scalar reference; `1e-4` admits that legal slack while
/// still failing a wrong port.
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

/// A gray color of a given intensity.
fn gray(v: f32) -> [f32; 3] {
    [v, v, v]
}

/// Builds a query from the cross colors and a lock state, taking the `locked`
/// flag from the golden
/// [`is_locked`](prism_render_architecture::temporal_upscale::lock::LockState::is_locked)
/// so the `GPU` and the golden branch identically.
fn query(
    center: [f32; 3],
    north: [f32; 3],
    south: [f32; 3],
    west: [f32; 3],
    east: [f32; 3],
    lock: LockState,
) -> TaauThinFeatureLockQuery {
    TaauThinFeatureLockQuery::new(
        center,
        north,
        south,
        west,
        east,
        lock.is_locked(),
        lock.lifetime,
    )
}

/// Evaluates the golden closed forms on the same inputs the query carries.
fn oracle(q: &TaauThinFeatureLockQuery) -> TaauThinFeatureLockResult {
    let strength = thin_feature_strength(q.center, q.north, q.south, q.west, q.east);
    let lock = LockState {
        lifetime: q.lifetime,
        luma: 0.0,
    };
    TaauThinFeatureLockResult {
        strength,
        rejection: rejection_scale(lock),
    }
}

/// Pins one `GPU` pixel result against the golden oracle: both scalars within
/// tolerance.
fn check_pixel(idx: usize, got: &TaauThinFeatureLockResult, want: &TaauThinFeatureLockResult) {
    assert!(
        close(got.strength, want.strength),
        "pixel {idx} strength: gpu {} vs cpu {}",
        got.strength,
        want.strength
    );
    assert!(
        close(got.rejection, want.rejection),
        "pixel {idx} rejection: gpu {} vs cpu {}",
        got.rejection,
        want.rejection
    );
}

/// Dispatches every pixel and pins each result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuTaauThinFeatureLock, queries: &[TaauThinFeatureLockQuery]) {
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

/// Draws a channel value in `[0.0, 1.0)` at milli resolution from `state`.
fn channel(state: &mut u64) -> f32 {
    (lcg(state) % 1000) as f32 / 1000.0
}

/// Draws a lock lifetime in `[0.5, 4.5]` at milli resolution, well clear of the
/// `0` unlocked boundary.
fn lifetime(state: &mut u64) -> f32 {
    0.5 + (lcg(state) % 4000) as f32 / 1000.0
}

/// Rec. 709 luminance, replicated host-side to reject random crosses that land
/// near a guard boundary. Uses only `+`/`*`, no transcendental method.
fn luma(c: [f32; 3]) -> f32 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

/// Whether a random cross is clear of both guard ties: its local luma `range` is
/// comfortably positive (so the divide is well-conditioned) and its
/// `separation` is clearly positive or clearly negative (so the `separation <=
/// 0` guard cannot flip between `CPU` and `GPU`). Uses only `min`/`max` and
/// never an `f32` `==`.
fn well_conditioned(
    center: [f32; 3],
    north: [f32; 3],
    south: [f32; 3],
    west: [f32; 3],
    east: [f32; 3],
) -> bool {
    let c = luma(center);
    let n = luma(north);
    let s = luma(south);
    let w = luma(west);
    let e = luma(east);
    let v_max = n.max(s);
    let v_min = n.min(s);
    let h_max = w.max(e);
    let h_min = w.min(e);
    let ring_max = v_max.max(h_max);
    let ring_min = v_min.min(h_min);
    let range = ring_max.max(c) - ring_min.min(c);
    if range < 0.05 {
        return false;
    }
    let bright = (c - v_max).min(c - h_max);
    let dark = (v_min - c).min(h_min - c);
    let separation = bright.max(dark);
    // Keep the separation a clear margin off the zero guard on either side.
    separation.abs() >= 0.02
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping taau_thin_feature_lock parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuTaauThinFeatureLock::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn bright_line_is_a_strong_feature() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTaauThinFeatureLock::new(&ctx);
    // Center far above all four neighbors: a crisp bright thin feature with a
    // separation and range well clear of zero.
    let q = query(
        gray(1.0),
        gray(0.0),
        gray(0.0),
        gray(0.0),
        gray(0.0),
        LockState::default(),
    );
    check(&ctx, &gpu, &[q]);
}

#[test]
fn dark_point_is_a_strong_feature() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTaauThinFeatureLock::new(&ctx);
    // Center far below all four neighbors: a dark thin feature.
    let q = query(
        gray(0.0),
        gray(1.0),
        gray(1.0),
        gray(1.0),
        gray(1.0),
        LockState::default(),
    );
    check(&ctx, &gpu, &[q]);
}

#[test]
fn edge_is_not_a_thin_feature() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTaauThinFeatureLock::new(&ctx);
    // Center matches one side of each axis (a step edge, not an isolated line):
    // not a two-axis extremum, so the strength is zero.
    let q = query(
        gray(1.0),
        gray(1.0),
        gray(0.0),
        gray(1.0),
        gray(0.0),
        LockState::default(),
    );
    check(&ctx, &gpu, &[q]);
}

#[test]
fn flat_cross_has_no_feature() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTaauThinFeatureLock::new(&ctx);
    // A uniform cross has a zero local luma range: the guard returns zero.
    let q = query(
        gray(0.5),
        gray(0.5),
        gray(0.5),
        gray(0.5),
        gray(0.5),
        LockState::default(),
    );
    check(&ctx, &gpu, &[q]);
}

#[test]
fn partial_feature_scores_between_zero_and_one() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTaauThinFeatureLock::new(&ctx);
    // A center above both axes but by less than the full range: a mid strength
    // that exercises the (separation / range).min(1) branch away from a tie.
    let q = query(
        gray(0.7),
        gray(0.2),
        gray(0.1),
        gray(0.15),
        gray(0.25),
        LockState::default(),
    );
    check(&ctx, &gpu, &[q]);
}

#[test]
fn unlocked_pixel_keeps_full_rejection() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTaauThinFeatureLock::new(&ctx);
    // An unlocked pixel keeps the full rejection of 1.
    let q = query(
        gray(0.4),
        gray(0.3),
        gray(0.6),
        gray(0.5),
        gray(0.2),
        LockState::default(),
    );
    check(&ctx, &gpu, &[q]);
}

#[test]
fn full_lock_strongly_suppresses_rejection() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTaauThinFeatureLock::new(&ctx);
    // A fresh full-lifetime lock nearly zeroes the rejection.
    let lock = LockState {
        lifetime: INITIAL_LOCK_LIFETIME,
        luma: 0.5,
    };
    let q = query(
        gray(0.6),
        gray(0.1),
        gray(0.2),
        gray(0.15),
        gray(0.05),
        lock,
    );
    check(&ctx, &gpu, &[q]);
}

#[test]
fn half_decayed_lock_restores_some_rejection() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTaauThinFeatureLock::new(&ctx);
    // A half-decayed lock eases the suppression back toward the full rejection.
    let lock = LockState {
        lifetime: INITIAL_LOCK_LIFETIME * 0.5,
        luma: 0.5,
    };
    let q = query(gray(0.3), gray(0.8), gray(0.7), gray(0.9), gray(0.6), lock);
    check(&ctx, &gpu, &[q]);
}

#[test]
fn mixed_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTaauThinFeatureLock::new(&ctx);
    // A deterministic batch spanning bright, dark, edge, flat, locked and
    // unlocked cases, dispatched together so the per-thread indexing and the
    // contiguous output slots are both exercised.
    let queries = [
        query(
            gray(1.0),
            gray(0.0),
            gray(0.0),
            gray(0.0),
            gray(0.0),
            LockState::default(),
        ),
        query(
            gray(0.0),
            gray(1.0),
            gray(1.0),
            gray(1.0),
            gray(1.0),
            LockState {
                lifetime: INITIAL_LOCK_LIFETIME,
                luma: 0.0,
            },
        ),
        query(
            gray(0.5),
            gray(0.5),
            gray(0.5),
            gray(0.5),
            gray(0.5),
            LockState {
                lifetime: 1.0,
                luma: 0.5,
            },
        ),
        query(
            [0.8, 0.2, 0.4],
            [0.1, 0.1, 0.2],
            [0.2, 0.1, 0.1],
            [0.15, 0.05, 0.1],
            [0.1, 0.2, 0.15],
            LockState {
                lifetime: 2.5,
                luma: 0.3,
            },
        ),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTaauThinFeatureLock::new(&ctx);
    let mut state = 0x0f1e_2d3c_4b5a_6978_u64;
    let mut queries = Vec::new();
    // Several workgroups' worth of random pixels: arbitrary cross colors and a
    // mix of locked and unlocked states, each clear of a guard boundary.
    let mut made = 0u32;
    let mut tries = 0u32;
    // Reject-sample pixels clear of the separation/range guard ties so CPU and
    // GPU stay on the same side of every discrete decision.
    while made < 256 && tries < 20_000 {
        tries += 1;
        let center = [
            channel(&mut state),
            channel(&mut state),
            channel(&mut state),
        ];
        let north = [
            channel(&mut state),
            channel(&mut state),
            channel(&mut state),
        ];
        let south = [
            channel(&mut state),
            channel(&mut state),
            channel(&mut state),
        ];
        let west = [
            channel(&mut state),
            channel(&mut state),
            channel(&mut state),
        ];
        let east = [
            channel(&mut state),
            channel(&mut state),
            channel(&mut state),
        ];
        if !well_conditioned(center, north, south, west, east) {
            continue;
        }
        // Lock about half the pixels; unlocked ones carry a zero lifetime.
        let lock = if made.is_multiple_of(2) {
            LockState {
                lifetime: lifetime(&mut state),
                luma: channel(&mut state),
            }
        } else {
            LockState::default()
        };
        queries.push(query(center, north, south, west, east, lock));
        made += 1;
    }
    assert!(
        made >= 256,
        "expected 256 well-conditioned pixels, got {made}"
    );
    check(&ctx, &gpu, &queries);
}
