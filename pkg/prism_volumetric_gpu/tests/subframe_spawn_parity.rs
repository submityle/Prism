//! Real-device parity for the sub-frame spawn twin:
//! [`GpuSubframeSpawn`](prism_volumetric_gpu::subframe_spawn::GpuSubframeSpawn)
//! must reproduce the `CPU` golden
//! [`subframe_spawn`](prism_render_architecture::particle::subframe_spawn)
//! across the fractional spawn accumulator (integer count plus surviving carry),
//! the centered sub-frame fraction of the `i`-th spawn, the position and scalar
//! linear interpolations, and the one-shot burst crossing test.
//!
//! The fixtures cover the shapes the golden unit tests call out: a product that
//! lands on a whole particle, a product that preserves a fractional carry, a
//! zero rate and a zero `dt` (both emit nothing and keep the carry), a negative
//! rate and a negative `dt` (the guard emits nothing), a single centered spawn
//! and a four-way centered schedule, interpolation endpoints and midpoints, and
//! a burst that fires inside its interval but not across the shared boundary. A
//! final randomized batch rejection-samples `rate * dt + carry` away from any
//! integer boundary and keeps the interpolation fraction in `[0.2, 0.8]`, so
//! `CPU` and `GPU` stay on the same side of the `floor` regardless of a few
//! units in the last place of slack.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The spawn count and the crossing flag are discrete classifications, so `CPU`
//! and `GPU` must agree exactly: the comparison is an exact `==` on the `u32`
//! count and on the crossing `bool`. The carry, the fraction and the two
//! interpolations thread through multiplies, adds and one guarded division, so
//! they are compared under tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`,
//! `REL_FLOOR = 1e-6`).
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::subframe_spawn`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::subframe_spawn::{
    interpolate_position, interpolate_scalar, SpawnAccumulator, SpawnBurst, SubframeSchedule,
};
use prism_volumetric_gpu::subframe_spawn::{GpuSubframeSpawn, SubframeSpawnQuery};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous quantities.
const ABS_EPS: f32 = 1.0e-4;
/// Relative tolerance for the continuous quantities.
const REL_EPS: f32 = 1.0e-3;
/// Floor for the relative-tolerance denominator.
const REL_FLOOR: f32 = 1.0e-6;

/// Mixed absolute / relative tolerance comparison for one `f32` lane.
fn approx(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= ABS_EPS {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale <= REL_EPS
}

/// Tolerant comparison of two `vec3` triples.
fn approx3(a: [f32; 3], b: [f32; 3]) -> bool {
    approx(a[0], b[0]) && approx(a[1], b[1]) && approx(a[2], b[2])
}

/// Builds a query from its parts; helpers below override only what they need.
#[expect(
    clippy::too_many_arguments,
    reason = "the twin bundles every independent routine's inputs into one query"
)]
fn query(
    carry: f32,
    rate: f32,
    dt: f32,
    frac_count: u32,
    frac_index: u32,
    prev_pos: [f32; 3],
    curr_pos: [f32; 3],
    frac: f32,
    prev_scalar: f32,
    curr_scalar: f32,
    burst_time: f32,
    prev_time: f32,
    curr_time: f32,
) -> SubframeSpawnQuery {
    SubframeSpawnQuery {
        carry,
        rate,
        dt,
        frac_count,
        frac_index,
        prev_pos,
        curr_pos,
        frac,
        prev_scalar,
        curr_scalar,
        burst_time,
        prev_time,
        curr_time,
    }
}

/// Asserts every twinned answer for one query matches the `CPU` golden.
fn assert_parity(gpu: &GpuSubframeSpawn, ctx: &GpuContext, q: &SubframeSpawnQuery) {
    let got = gpu.evaluate(ctx, std::slice::from_ref(q));
    assert_eq!(got.len(), 1, "one result per query");
    let g = got[0];

    let (whole, next) = SpawnAccumulator { carry: q.carry }.accumulate(q.rate, q.dt);
    assert_eq!(
        g.spawn_count, whole,
        "spawn count mismatch: gpu {} vs cpu {whole}",
        g.spawn_count
    );
    assert!(
        approx(g.next_carry, next.carry),
        "carry mismatch: gpu {} vs cpu {}",
        g.next_carry,
        next.carry
    );

    let frac = SubframeSchedule::new(q.frac_count).fraction(q.frac_index);
    assert!(
        approx(g.fraction, frac),
        "fraction mismatch: gpu {} vs cpu {frac}",
        g.fraction
    );

    let pos = interpolate_position(q.prev_pos, q.curr_pos, q.frac);
    assert!(
        approx3(g.interp_pos, pos),
        "interp_pos mismatch: gpu {:?} vs cpu {pos:?}",
        g.interp_pos
    );

    let scalar = interpolate_scalar(q.prev_scalar, q.curr_scalar, q.frac);
    assert!(
        approx(g.interp_scalar, scalar),
        "interp_scalar mismatch: gpu {} vs cpu {scalar}",
        g.interp_scalar
    );

    let due = SpawnBurst::new(1, q.burst_time).is_due(q.prev_time, q.curr_time);
    assert_eq!(
        g.is_due, due,
        "is_due mismatch for burst {} in ({}, {}]",
        q.burst_time, q.prev_time, q.curr_time
    );
}

#[test]
fn accumulator_whole_and_fractional() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSubframeSpawn::new(&ctx);
    // 2 particles/s * 0.5 s == exactly 1 (an exact product, floor is bit-stable).
    let whole = query(
        0.0,
        2.0,
        0.5,
        4,
        1,
        [1.0, 2.0, 3.0],
        [5.0, -2.0, 9.0],
        0.5,
        3.0,
        7.0,
        1.0,
        0.0,
        0.5,
    );
    assert_parity(&gpu, &ctx, &whole);
    // 2.5 * 1.0 == 2 with 0.5 carried; the carry is well away from a boundary.
    let fractional = query(
        0.0,
        2.5,
        1.0,
        4,
        2,
        [0.0, 0.0, 0.0],
        [4.0, 8.0, -12.0],
        0.5,
        3.0,
        7.0,
        2.0,
        1.0,
        3.0,
    );
    assert_parity(&gpu, &ctx, &fractional);
}

#[test]
fn zero_and_negative_inputs_emit_nothing() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSubframeSpawn::new(&ctx);
    // A zero rate and a zero dt both emit nothing and keep the carry unchanged.
    let zero_rate = query(
        0.4,
        0.0,
        1.0 / 60.0,
        8,
        3,
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
        0.3,
        1.0,
        2.0,
        0.5,
        0.0,
        1.0,
    );
    assert_parity(&gpu, &ctx, &zero_rate);
    let zero_dt = query(
        0.4,
        120.0,
        0.0,
        8,
        4,
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
        0.7,
        1.0,
        2.0,
        0.5,
        0.0,
        1.0,
    );
    assert_parity(&gpu, &ctx, &zero_dt);
    // A negative rate and a negative dt both hit the guard: zero count, carry
    // preserved.
    let neg_rate = query(
        0.25,
        -5.0,
        0.5,
        8,
        5,
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
        0.4,
        1.0,
        2.0,
        0.5,
        0.0,
        1.0,
    );
    assert_parity(&gpu, &ctx, &neg_rate);
    let neg_dt = query(
        0.25,
        5.0,
        -0.5,
        8,
        6,
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
        0.6,
        1.0,
        2.0,
        0.5,
        0.0,
        1.0,
    );
    assert_parity(&gpu, &ctx, &neg_dt);
}

#[test]
fn fraction_is_centered() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSubframeSpawn::new(&ctx);
    // A single spawn sits at the frame center; a four-way schedule is centered.
    let single = query(
        0.0,
        1.0,
        0.3,
        1,
        0,
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        0.5,
        0.0,
        1.0,
        0.5,
        0.0,
        1.0,
    );
    assert_parity(&gpu, &ctx, &single);
    for i in 0..4u32 {
        let q = query(
            0.0,
            1.0,
            0.3,
            4,
            i,
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            0.5,
            0.0,
            1.0,
            0.5,
            0.0,
            1.0,
        );
        assert_parity(&gpu, &ctx, &q);
    }
    // A zero spawn count has no spawn to place: the fraction falls back to 0.0.
    let empty = query(
        0.0,
        1.0,
        0.3,
        0,
        0,
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        0.5,
        0.0,
        1.0,
        0.5,
        0.0,
        1.0,
    );
    assert_parity(&gpu, &ctx, &empty);
}

#[test]
fn interpolation_hits_endpoints_and_midpoint() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSubframeSpawn::new(&ctx);
    let prev = [1.0, 2.0, 3.0];
    let curr = [5.0, -2.0, 9.0];
    for frac in [0.0, 0.5, 1.0] {
        let q = query(
            0.0, 1.0, 0.3, 4, 1, prev, curr, frac, 3.0, 7.0, 0.5, 0.0, 1.0,
        );
        assert_parity(&gpu, &ctx, &q);
    }
}

#[test]
fn burst_fires_once_across_the_boundary() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSubframeSpawn::new(&ctx);
    // Inside the interval it fires; the next frame's exclusive lower edge does
    // not, so the burst releases exactly once across the shared boundary.
    let inside = query(
        0.0,
        1.0,
        0.3,
        4,
        1,
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        0.5,
        0.0,
        1.0,
        1.5,
        1.0,
        2.0,
    );
    assert_parity(&gpu, &ctx, &inside);
    let after = query(
        0.0,
        1.0,
        0.3,
        4,
        1,
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        0.5,
        0.0,
        1.0,
        2.0,
        2.0,
        3.0,
    );
    assert_parity(&gpu, &ctx, &after);
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
fn range(state: &mut u64, lo: f32, hi: f32) -> f32 {
    lo + lcg(state) * (hi - lo)
}

/// A pseudo-random vector with each component in `[-span, span)`.
fn rand_vec(state: &mut u64, span: f32) -> [f32; 3] {
    [
        range(state, -span, span),
        range(state, -span, span),
        range(state, -span, span),
    ]
}

/// Builds a well-conditioned query by rejection sampling: `rate * dt + carry`
/// is accepted only when its fractional part lands in `[0.2, 0.8]`, so the
/// `floor` is bit-stable across devices, and the interpolation fraction stays in
/// `[0.2, 0.8]`, well clear of either endpoint.
fn random_query(state: &mut u64) -> SubframeSpawnQuery {
    let (carry, rate, dt) = loop {
        let carry = range(state, 0.0, 1.0);
        let rate = range(state, 0.0, 20.0);
        let dt = range(state, 0.0, 2.0);
        let total = rate * dt + carry;
        let fract = total - total.floor();
        if (0.2..=0.8).contains(&fract) {
            break (carry, rate, dt);
        }
    };
    let frac_count = 1 + (lcg(state) * 63.0) as u32;
    let frac_index = (lcg(state) * frac_count as f32) as u32;
    let frac = range(state, 0.2, 0.8);
    let prev_time = range(state, -5.0, 5.0);
    let curr_time = prev_time + range(state, 0.1, 3.0);
    // Spread the burst instant across and beyond the interval so some fixtures
    // fire and some do not; a direct compare of stored inputs cannot tie.
    let burst_time = range(state, prev_time - 1.0, curr_time + 1.0);
    query(
        carry,
        rate,
        dt,
        frac_count,
        frac_index.min(frac_count - 1),
        rand_vec(state, 10.0),
        rand_vec(state, 10.0),
        frac,
        range(state, -8.0, 8.0),
        range(state, -8.0, 8.0),
        burst_time,
        prev_time,
        curr_time,
    )
}

#[test]
fn random_batch_matches_elementwise() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSubframeSpawn::new(&ctx);
    let mut state = 0x51b3_f00d_c0ff_ee11_u64;
    let batch: Vec<SubframeSpawnQuery> = (0..256).map(|_| random_query(&mut state)).collect();
    let got = gpu.evaluate(&ctx, &batch);
    assert_eq!(got.len(), batch.len());
    for q in &batch {
        assert_parity(&gpu, &ctx, q);
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSubframeSpawn::new(&ctx);
    // No dispatch is issued and the result vector is empty.
    assert!(gpu.evaluate(&ctx, &[]).is_empty());
}
