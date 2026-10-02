//! Real-device parity for the particle time-control twin:
//! [`GpuTimeControl`](prism_volumetric_gpu::time_control::GpuTimeControl) must
//! reproduce the `CPU` golden
//! [`time_control`](prism_render_architecture::particle::time_control) across a
//! real-time emitter, a slow-motion emitter, a fast-forward emitter, a hard
//! pause, a substep count clamped to `max_substeps`, the no-ceiling fallback
//! when `max_substep_dt` is non-positive, and a randomized batch of
//! clearly-conditioned emitters compared element-for-element.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each query is a fixed, non-reorderable sequence of multiplies, compares and
//! one divide, so `CPU` and `GPU` evaluate the same closed form in the same
//! order. The continuous fields are not bit-exact: a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place. The comparison therefore
//! allows `abs_diff <= 1e-4` or `rel_diff <= 1e-3` on the `f32` fields, while
//! the integer `substep_count` and the `is_effectively_paused` flag are
//! compared exactly.
//!
//! # Conditioning
//!
//! Every fixture is deliberately well away from a branch tie: unpaused scales
//! sit far above the `CMP_EPS` pause threshold while paused emitters pin the
//! scale to exactly zero, and the ratio `effective_dt / max_substep_dt` keeps
//! its fractional part clear of an integer so the substep ceiling rounds the
//! same way on both devices regardless of a few units in the last place of
//! slack. The randomized batch rejection-samples until both margins hold.
//!
//! Provenance: twinned from this repository's
//! [`time_control`](prism_render_architecture::particle::time_control); no
//! third-party engine source or derived code.

use prism_render_architecture::particle::time_control::{
    scaled_age_delta, scaled_spawn_rate, substep_count, EmitterTimeControl, TimeScale,
};
use prism_volumetric_gpu::time_control::{GpuTimeControl, TimeControlQuery, TimeControlResult};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound. A `GPU` may fuse a multiply-add the scalar reference
/// leaves separate, perturbing the low mantissa bits by a few units in the last
/// place; `1e-4` admits that legal slack while still failing a wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in
/// the last place exceed the absolute floor.
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

/// Builds an emitter control state from two raw scale factors and a pause flag,
/// routing both scales through the clamping `TimeScale` constructor.
fn ctrl(global: f32, local: f32, paused: bool) -> EmitterTimeControl {
    EmitterTimeControl {
        global_scale: TimeScale::new(global),
        local_scale: TimeScale::new(local),
        paused,
    }
}

/// A tiny integer linear-congruential generator; only integer and divide work,
/// so no transcendental appears. Returns a value in `[0, 1)`.
fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let bits = (*state >> 40) as u32;
    (bits & 0x00ff_ffff) as f32 / 16_777_216.0
}

/// A pseudo-random value in `[lo, hi)` drawn from `state`.
fn uniform(state: &mut u64, lo: f32, hi: f32) -> f32 {
    lo + lcg(state) * (hi - lo)
}

/// Builds a clearly-conditioned emitter query by rejection sampling: unpaused
/// scales stay far above the pause threshold and the substep ratio keeps its
/// fractional part clear of an integer so the ceiling rounds the same way on
/// both devices.
fn rand_query(state: &mut u64) -> TimeControlQuery {
    loop {
        let paused = lcg(state) < 0.2;
        let global = uniform(state, 0.25, 2.0);
        let local = uniform(state, 0.25, 2.0);
        let raw_dt = uniform(state, 0.004, 0.05);
        let base_rate = uniform(state, 0.0, 500.0);
        let max_substep_dt = uniform(state, 0.004, 0.02);
        let max_substeps = 1 + (lcg(state) * 8.0) as u32;

        let control = ctrl(global, local, paused);
        let query = TimeControlQuery::new(control, raw_dt, base_rate, max_substep_dt, max_substeps);

        // A paused emitter freezes everything, so no ceiling tie can arise.
        if paused {
            return query;
        }

        // Keep the substep ratio's fractional part clear of an integer so the
        // truncation rounds identically on both devices.
        let effective_dt = control.effective_dt(raw_dt);
        let ratio = effective_dt / max_substep_dt;
        let frac = ratio - ratio.floor();
        if !(0.15..=0.85).contains(&frac) {
            continue;
        }
        return query;
    }
}

/// Pins one `GPU` result against the `CPU` golden for `query`: the effective
/// scale, effective `dt`, scaled spawn rate and scaled age delta must agree
/// within bound, while the substep count and the effectively-paused flag must
/// match exactly.
fn pin(idx: usize, query: &TimeControlQuery, got: &TimeControlResult) {
    let want_scale = query.control.effective_scale();
    let want_dt = query.control.effective_dt(query.raw_dt);
    let want_spawn = scaled_spawn_rate(query.base_rate, query.control);
    let want_age = scaled_age_delta(query.raw_dt, query.control);
    let want_substeps = substep_count(want_dt, query.max_substep_dt, query.max_substeps);
    let want_paused = query.control.is_effectively_paused();

    assert!(
        close(got.effective_scale, want_scale),
        "query {idx} effective_scale: gpu {} vs cpu {}",
        got.effective_scale,
        want_scale
    );
    assert!(
        close(got.effective_dt, want_dt),
        "query {idx} effective_dt: gpu {} vs cpu {}",
        got.effective_dt,
        want_dt
    );
    assert!(
        close(got.scaled_spawn_rate, want_spawn),
        "query {idx} scaled_spawn_rate: gpu {} vs cpu {}",
        got.scaled_spawn_rate,
        want_spawn
    );
    assert!(
        close(got.scaled_age_delta, want_age),
        "query {idx} scaled_age_delta: gpu {} vs cpu {}",
        got.scaled_age_delta,
        want_age
    );
    assert_eq!(
        got.substep_count, want_substeps,
        "query {idx} substep_count: gpu {} vs cpu {}",
        got.substep_count, want_substeps
    );
    assert_eq!(
        got.is_effectively_paused, want_paused,
        "query {idx} is_effectively_paused: gpu {} vs cpu {}",
        got.is_effectively_paused, want_paused
    );
}

/// Dispatches `queries` on the `GPU` and pins every result against the
/// reference.
fn check(ctx: &GpuContext, gpu: &GpuTimeControl, queries: &[TimeControlQuery]) {
    let got = gpu.eval(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (query, result)) in queries.iter().zip(got.iter()).enumerate() {
        pin(idx, query, result);
    }
}

#[test]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTimeControl::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn real_time_emitter_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTimeControl::new(&ctx);
    // Full speed: scale 1.0, a 16 ms frame over a 10 ms ceiling demands two
    // substeps (ratio 1.6, clear of an integer).
    let query = TimeControlQuery::new(ctrl(1.0, 1.0, false), 0.016, 100.0, 0.01, 8);
    check(&ctx, &gpu, &[query]);
}

#[test]
fn slow_motion_emitter_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTimeControl::new(&ctx);
    // Quarter speed: scale 0.25, so a 20 ms frame scales to 5 ms and a 10 ms
    // ceiling demands a single substep (ratio 0.5).
    let query = TimeControlQuery::new(ctrl(1.0, 0.25, false), 0.02, 200.0, 0.01, 8);
    check(&ctx, &gpu, &[query]);
}

#[test]
fn fast_forward_emitter_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTimeControl::new(&ctx);
    // Triple speed: scale 3.0, a 10 ms frame scales to 30 ms and an 8 ms ceiling
    // demands four substeps (ratio 3.75, clear of an integer).
    let query = TimeControlQuery::new(ctrl(2.0, 1.5, false), 0.01, 150.0, 0.008, 8);
    check(&ctx, &gpu, &[query]);
}

#[test]
fn paused_emitter_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTimeControl::new(&ctx);
    // A hard pause freezes the scale to exactly zero: no dt, no spawn, no aging,
    // zero substeps and the effectively-paused flag set.
    let query = TimeControlQuery::new(ctrl(1.0, 1.0, true), 0.016, 100.0, 0.01, 8);
    check(&ctx, &gpu, &[query]);
}

#[test]
fn substep_count_clamps_to_max() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTimeControl::new(&ctx);
    // Five-times speed: a 10 ms frame scales to 50 ms, which a 2 ms ceiling would
    // split into 25 substeps, clamped hard to the four-step cap.
    let query = TimeControlQuery::new(ctrl(1.0, 5.0, false), 0.01, 100.0, 0.002, 4);
    check(&ctx, &gpu, &[query]);
}

#[test]
fn no_ceiling_fallback_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTimeControl::new(&ctx);
    // A non-positive max_substep_dt has no usable ceiling: a non-empty frame
    // falls back to a single capped step, and a zero cap yields zero steps.
    let queries = [
        TimeControlQuery::new(ctrl(1.0, 1.0, false), 0.016, 100.0, 0.0, 8),
        TimeControlQuery::new(ctrl(1.0, 1.0, false), 0.016, 100.0, 0.0, 0),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn mixed_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTimeControl::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    // One batch mixing the deterministic fixtures with many random emitters,
    // dispatched together so the per-thread indexing and the contiguous storage
    // layout are both exercised, then pinned element-for-element.
    let mut queries = vec![
        TimeControlQuery::new(ctrl(1.0, 1.0, false), 0.016, 100.0, 0.01, 8),
        TimeControlQuery::new(ctrl(1.0, 0.25, false), 0.02, 200.0, 0.01, 8),
        TimeControlQuery::new(ctrl(1.0, 1.0, true), 0.016, 100.0, 0.01, 8),
        TimeControlQuery::new(ctrl(1.0, 5.0, false), 0.01, 100.0, 0.002, 4),
    ];
    for _ in 0..60 {
        queries.push(rand_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn many_random_emitters_sweep_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTimeControl::new(&ctx);
    let mut state = 0x0bad_c0de_cafe_f00d_u64;
    // A larger sweep of clearly-conditioned emitters (several workgroups' worth)
    // pins every reported field across many random dilation states.
    let queries: Vec<TimeControlQuery> = (0..200).map(|_| rand_query(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}
