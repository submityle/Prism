//! Real-device parity for the `tonemap` twin:
//! [`GpuTonemap`](prism_volumetric_gpu::tonemap::GpuTonemap) must reproduce the
//! `CPU` golden
//! [`tonemap`](prism_render_architecture::particle::tonemap) across the full
//! `exposure` -> operator -> `gamma`-encode map pipeline for every operator, the
//! bare exposure product, the integer `exposure_from_stops` factor (positive,
//! negative, zero and clamped stops), each tone curve and the `gamma`
//! encode/decode evaluated at a scalar, the selected operator apply, and a
//! randomized batch compared field-for-field.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each operator is a fixed, non-reorderable sequence of multiplies, adds,
//! divides and one `sqrt`, so `CPU` and `GPU` evaluate the same closed form in
//! the same order. They are not bit-exact: a `GPU` may fuse a multiply-add the
//! scalar reference leaves separate, perturbing the low mantissa bits by a few
//! units in the last place. The comparison therefore allows `abs_diff <= 1e-4`
//! or `rel_diff <= 1e-3` on every `f32` field.
//!
//! # Conditioning
//!
//! Every random fixture stays clear of the reference guards: the `white` point
//! is drawn in `[0.5, 8.0)` so its square sits far above `MIN_DENOM` and the
//! extended-`Reinhard` inner division never takes its guard, the tone-curve
//! denominators `1 + x` stay at or above `1.0` for the non-negative inputs, and
//! the stop count stays within `[-20, 20]` so the integer shift matches the
//! reference without approaching `MAX_STOPS`. The clamped-stop and zero-input
//! fixtures that do exercise a guard are deterministic and land on the exact
//! same guarded value on both devices.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::tonemap`；
//! no third-party engine source or derived code.
#![forbid(unsafe_code)]

use prism_render_architecture::particle::tonemap::TonemapOperator;
use prism_volumetric_gpu::tonemap::{golden, GpuTonemap, TonemapQuery, TonemapResult};
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

/// The three operators in a stable order, cycled by index so a batch covers
/// every enum variant.
const OPERATORS: [TonemapOperator; 3] = [
    TonemapOperator::Reinhard,
    TonemapOperator::ReinhardExtended,
    TonemapOperator::Aces,
];

/// Builds a well-conditioned random query: positive interior `rgb`/`scalar`, a
/// `white` far above the guard, an `encoded` channel in `[0, 1)`, a positive
/// `exposure` and a stop count well within `MAX_STOPS`. The operator is chosen
/// by `index` so a batch cycles through all three variants.
fn rand_query(state: &mut u64, index: usize) -> TonemapQuery {
    let rgb = [
        range(state, 0.0, 4.0),
        range(state, 0.0, 4.0),
        range(state, 0.0, 4.0),
    ];
    let scalar = range(state, 0.0, 8.0);
    let encoded = range(state, 0.0, 1.0);
    let exposure = range(state, 0.25, 4.0);
    // White in [0.5, 8.0) keeps white^2 >= 0.25, far above MIN_DENOM, so the
    // extended-Reinhard inner division never takes its guard.
    let white = range(state, 0.5, 8.0);
    // Stops in [-20, 20] shift exactly on both devices, clear of MAX_STOPS.
    let stops = (range(state, 0.0, 41.0) as i32) - 20;
    TonemapQuery::new(
        rgb,
        scalar,
        encoded,
        exposure,
        white,
        OPERATORS[index % 3],
        stops,
    )
}

/// A well-conditioned deterministic query used wherever a particular operator
/// is under test and the other inputs just need to be clear of the guards.
fn baseline(operator: TonemapOperator) -> TonemapQuery {
    TonemapQuery::new([0.3, 1.7, 5.0], 2.5, 0.6, 1.25, 4.0, operator, 3)
}

/// Pins one `GPU` result against the `CPU` golden for `query`: every field of
/// the pipeline output and all seven scalar operators must agree within the
/// documented tolerance.
fn pin(idx: usize, query: &TonemapQuery, got: &TonemapResult) {
    let want = golden(query);
    let fields = [
        ("mapped.r", got.mapped[0], want.mapped[0]),
        ("mapped.g", got.mapped[1], want.mapped[1]),
        ("mapped.b", got.mapped[2], want.mapped[2]),
        ("exposed.r", got.exposed[0], want.exposed[0]),
        ("exposed.g", got.exposed[1], want.exposed[1]),
        ("exposed.b", got.exposed[2], want.exposed[2]),
        (
            "exposure_from_stops",
            got.exposure_from_stops,
            want.exposure_from_stops,
        ),
        ("reinhard", got.reinhard, want.reinhard),
        (
            "reinhard_extended",
            got.reinhard_extended,
            want.reinhard_extended,
        ),
        ("aces", got.aces, want.aces),
        ("encode", got.encode, want.encode),
        ("decode", got.decode, want.decode),
        ("operator_apply", got.operator_apply, want.operator_apply),
    ];
    for (name, g, c) in fields {
        assert!(close(g, c), "query {idx} {name}: gpu {g} vs cpu {c}");
    }
}

/// Dispatches `queries` on the `GPU` and pins every result against the
/// reference.
fn check(ctx: &GpuContext, gpu: &GpuTonemap, queries: &[TonemapQuery]) {
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
    let gpu = GpuTonemap::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn every_operator_pipeline_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTonemap::new(&ctx);
    // One baseline per operator, dispatched together so the per-channel map
    // pipeline is pinned for every enum variant at once.
    let queries: Vec<TonemapQuery> = OPERATORS.into_iter().map(baseline).collect();
    check(&ctx, &gpu, &queries);
}

#[test]
fn exposure_from_stops_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTonemap::new(&ctx);
    // Sweep positive, negative and zero stops plus two magnitudes beyond
    // MAX_STOPS so the clamp-then-shift path is pinned against the reference.
    let stops = [-40, -31, -8, -1, 0, 1, 8, 31, 40];
    let queries: Vec<TonemapQuery> = stops
        .into_iter()
        .map(|s| {
            TonemapQuery::new(
                [0.5, 0.5, 0.5],
                1.0,
                0.5,
                1.0,
                2.0,
                TonemapOperator::Reinhard,
                s,
            )
        })
        .collect();
    check(&ctx, &gpu, &queries);
}

#[test]
fn zero_scalar_and_black_rgb_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTonemap::new(&ctx);
    // A zero scalar and black RGB land on exactly zero on both devices for every
    // operator, pinning the black-maps-to-black contract.
    let queries: Vec<TonemapQuery> = OPERATORS
        .into_iter()
        .map(|op| TonemapQuery::new([0.0, 0.0, 0.0], 0.0, 0.0, 1.5, 4.0, op, 0))
        .collect();
    check(&ctx, &gpu, &queries);
}

#[test]
fn bright_input_saturates_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTonemap::new(&ctx);
    // A very bright input drives each operator deep into its saturating shoulder
    // and the map clamp, pinning the near-one tail on both devices.
    let queries: Vec<TonemapQuery> = OPERATORS
        .into_iter()
        .map(|op| TonemapQuery::new([250.0, 500.0, 1000.0], 800.0, 0.95, 4.0, 3.0, op, 5))
        .collect();
    check(&ctx, &gpu, &queries);
}

#[test]
fn mixed_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTonemap::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    // One batch mixing the per-operator baselines with many random queries,
    // dispatched together so the per-thread indexing and the contiguous storage
    // layout are both exercised, then pinned field-for-field.
    let mut queries: Vec<TonemapQuery> = OPERATORS.into_iter().map(baseline).collect();
    for i in 0..48 {
        queries.push(rand_query(&mut state, i));
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn many_queries_sweep_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTonemap::new(&ctx);
    let mut state = 0x0bad_c0de_cafe_f00d_u64;
    // A larger sweep (several workgroups' worth) pins every operator across many
    // random queries, cycling the operator so all variants recur.
    let queries: Vec<TonemapQuery> = (0..200).map(|i| rand_query(&mut state, i)).collect();
    check(&ctx, &gpu, &queries);
}
