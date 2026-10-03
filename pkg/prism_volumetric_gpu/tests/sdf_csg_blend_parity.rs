//! Real-device parity for the smooth constructive-solid-geometry blend twin:
//! [`GpuSdfCsgBlend`](prism_volumetric_gpu::sdf_csg_blend::GpuSdfCsgBlend)
//! must reproduce the `CPU` closed forms of
//! `prism_render_architecture::ray_scene::sdf_csg` — the smooth union, smooth
//! intersection and smooth subtraction, each reported as a `(distance, blend)`
//! pair where `blend` in `[0, 1]` is the material weight of the second operand
//! at the rounded seam — across the fillet transition, the hard-operator
//! fallback and a randomized sweep compared query-for-query.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! This wave forbids depending on the golden crate, so the host oracle is an
//! *independent* reimplementation of the same three closed forms: the shared
//! quadratic polynomial `h = max(k - |a - b|, 0) / k`, the fillet offset
//! `h * h * k * 0.25`, and the blend `m = h * h * 0.5` selected against the
//! operand ordering. Because the reference and this oracle are both scalar
//! `f32`, a `GPU == oracle` pass is direct evidence the ported kernel computes
//! the same distances and blends the reference does.
//!
//! # Parity criterion
//!
//! Each distance and blend threads through products and quotients, so a `GPU`
//! result may land a few units in the last place from the scalar oracle; each
//! is asserted within `abs_diff <= 1e-4` or `rel_diff <= 1e-3`, with a
//! `rel_diff` floor of `1e-6` so a near-zero expected value (a hard-`0` blend)
//! does not inflate the relative error.
//!
//! # Conditioning
//!
//! The smooth-branch distance and blend are continuous polynomials in `a`, `b`
//! and `k`, and even across the operand tie-break the blend is continuous (the
//! two selection arms coincide at `0.5` when `a == b`), so a last-place
//! difference never becomes a jump. The one genuine hazard is a `k` approaching
//! zero, which divides the polynomial by a vanishing radius; the named fixtures
//! and the randomized sweep keep `k` in a safe positive band, while dedicated
//! fixtures exercise the hard `k <= 0` fallback where the result is an exact
//! `min`/`max` with a hard `0`/`1` blend.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::sdf_csg`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::sdf_csg_blend::{GpuSdfCsgBlend, SdfCsgBlendQuery, SdfCsgBlendResult};
use prism_volumetric_gpu::GpuContext;

/// Absolute bound on each distance and blend. A `GPU` multiply/divide may land
/// a few units in the last place from the scalar oracle; `1e-4` admits that
/// legal slack while still failing a wrong port.
const DIST_ABS: f32 = 1.0e-4;

/// Relative bound on each distance and blend, applied for larger magnitudes
/// where a few units in the last place exceed the absolute floor.
const DIST_REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value (a
/// hard-`0` blend) does not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within the given absolute or relative
/// bound (relative error floored at `REL_FLOOR`).
fn close(a: f32, b: f32, abs_eps: f32, rel_eps: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= abs_eps || rel <= rel_eps
}

/// Independent reimplementation of the reference smooth union blend: a
/// quadratic fillet subtracted from `min(a, b)`, reporting the second-operand
/// weight. A non-positive `k` falls back to the hard union with the `a <= b`
/// tie-break.
fn smooth_union_blend_oracle(a: f32, b: f32, k: f32) -> (f32, f32) {
    if k <= 0.0 {
        return (a.min(b), if a <= b { 0.0 } else { 1.0 });
    }
    let h = ((k - (a - b).abs()).max(0.0)) / k;
    let m = h * h * 0.5;
    let distance = a.min(b) - h * h * k * 0.25;
    let blend = if a < b { m } else { 1.0 - m };
    (distance, blend)
}

/// Independent reimplementation of the reference smooth intersection blend: the
/// dual of the union, a quadratic fillet added to `max(a, b)`. A non-positive
/// `k` falls back to the hard intersection with the `a >= b` tie-break.
fn smooth_intersection_blend_oracle(a: f32, b: f32, k: f32) -> (f32, f32) {
    if k <= 0.0 {
        return (a.max(b), if a >= b { 0.0 } else { 1.0 });
    }
    let h = ((k - (a - b).abs()).max(0.0)) / k;
    let m = h * h * 0.5;
    let distance = a.max(b) + h * h * k * 0.25;
    let blend = if a > b { m } else { 1.0 - m };
    (distance, blend)
}

/// Independent reimplementation of the reference smooth subtraction blend:
/// carve `b` out of `a` as a smooth intersection with the complement `-b`.
fn smooth_subtraction_blend_oracle(a: f32, b: f32, k: f32) -> (f32, f32) {
    smooth_intersection_blend_oracle(a, -b, k)
}

/// Computes all three reference `(distance, blend)` pairs for one query.
fn oracle(q: &SdfCsgBlendQuery) -> SdfCsgBlendResult {
    let (union_distance, union_blend) = smooth_union_blend_oracle(q.a, q.b, q.k);
    let (intersection_distance, intersection_blend) =
        smooth_intersection_blend_oracle(q.a, q.b, q.k);
    let (subtraction_distance, subtraction_blend) = smooth_subtraction_blend_oracle(q.a, q.b, q.k);
    SdfCsgBlendResult {
        union_distance,
        union_blend,
        intersection_distance,
        intersection_blend,
        subtraction_distance,
        subtraction_blend,
    }
}

/// Pins one `GPU` result against the host oracle under the tolerance.
fn check_one(idx: usize, got: &SdfCsgBlendResult, want: &SdfCsgBlendResult) {
    assert!(
        close(got.union_distance, want.union_distance, DIST_ABS, DIST_REL),
        "query {idx} union_distance: gpu {} vs cpu {}",
        got.union_distance,
        want.union_distance
    );
    assert!(
        close(got.union_blend, want.union_blend, DIST_ABS, DIST_REL),
        "query {idx} union_blend: gpu {} vs cpu {}",
        got.union_blend,
        want.union_blend
    );
    assert!(
        close(
            got.intersection_distance,
            want.intersection_distance,
            DIST_ABS,
            DIST_REL
        ),
        "query {idx} intersection_distance: gpu {} vs cpu {}",
        got.intersection_distance,
        want.intersection_distance
    );
    assert!(
        close(
            got.intersection_blend,
            want.intersection_blend,
            DIST_ABS,
            DIST_REL
        ),
        "query {idx} intersection_blend: gpu {} vs cpu {}",
        got.intersection_blend,
        want.intersection_blend
    );
    assert!(
        close(
            got.subtraction_distance,
            want.subtraction_distance,
            DIST_ABS,
            DIST_REL
        ),
        "query {idx} subtraction_distance: gpu {} vs cpu {}",
        got.subtraction_distance,
        want.subtraction_distance
    );
    assert!(
        close(
            got.subtraction_blend,
            want.subtraction_blend,
            DIST_ABS,
            DIST_REL
        ),
        "query {idx} subtraction_blend: gpu {} vs cpu {}",
        got.subtraction_blend,
        want.subtraction_blend
    );
}

/// Dispatches every query and pins each result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuSdfCsgBlend, queries: &[SdfCsgBlendQuery]) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (q, result)) in queries.iter().zip(got.iter()).enumerate() {
        let want = oracle(q);
        check_one(idx, result, &want);
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

/// Draws a `f32` in `[lo, hi)` from the generator, using only integer-to-float
/// division (no transcendental).
fn uniform(state: &mut u64, lo: f32, hi: f32) -> f32 {
    let u = lcg(state) as f32 * (1.0 / 4_294_967_296.0);
    lo + (hi - lo) * u
}

/// Builds one query from the two distance values and the fillet radius, in the
/// public constructor's argument order.
fn make_query(a: f32, b: f32, k: f32) -> SdfCsgBlendQuery {
    SdfCsgBlendQuery::new(a, b, k)
}

/// Returns whether a query keeps the two distance values within the fillet
/// radius, so the quadratic polynomial does not saturate (`h > 0`) and the
/// blend lands in the interesting transition region. Rejecting the trivial
/// saturated points keeps the randomized sweep exercising the smooth branch.
fn well_conditioned(q: &SdfCsgBlendQuery) -> bool {
    (q.a - q.b).abs() < q.k
}

/// A fixed battery of named cases spanning the fillet transition, the clamp
/// saturation boundary and both hard-fallback tie-breaks.
fn fixture_queries() -> Vec<SdfCsgBlendQuery> {
    vec![
        // Equal distances: deepest fillet, blend sits at 0.5 for every operator.
        make_query(1.0, 1.0, 0.5),
        // Second operand nearer (a < b), inside the fillet: union blend < 0.5.
        make_query(0.3, 0.7, 1.0),
        // First operand nearer (a > b), inside the fillet: union blend > 0.5.
        make_query(0.7, 0.3, 1.0),
        // Clamp saturation: |a - b| exceeds k, so h = 0 and the blend is hard.
        make_query(0.0, 5.0, 1.0),
        // Hard fallback with zero radius: exact min/max with a hard 0/1 blend.
        make_query(2.0, 5.0, 0.0),
        // Hard fallback with negative radius and the opposite ordering.
        make_query(5.0, 2.0, -1.0),
        // Subtraction transition: a and the complement -b lie within the fillet.
        make_query(0.5, -0.4, 1.0),
        // Mixed negative distances inside a wide fillet, every field live.
        make_query(-0.6, -0.2, 1.5),
        // Mixed with a tight fillet and a near-crossing pair.
        make_query(0.15, -0.05, 0.5),
    ]
}

/// Builds one well-conditioned random query: two distance values in `[-1, 1]`
/// and a fillet radius in a safe positive band `[0.1, 2.0]`, rejection-sampled
/// so the two values stay within the fillet and the smooth branch is live.
fn random_query(state: &mut u64) -> SdfCsgBlendQuery {
    loop {
        let a = uniform(state, -1.0, 1.0);
        let b = uniform(state, -1.0, 1.0);
        let k = uniform(state, 0.1, 2.0);
        let q = make_query(a, b, k);
        if well_conditioned(&q) {
            return q;
        }
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping sdf_csg_blend parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuSdfCsgBlend::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn union_equal_distance_blends_half() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfCsgBlend::new(&ctx);
    let q = fixture_queries()[0];
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    assert!(
        close(got[0].union_blend, 0.5, DIST_ABS, DIST_REL),
        "equal distances blend at the fillet midpoint: {}",
        got[0].union_blend
    );
}

#[test]
fn intersection_equal_distance_blends_half() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfCsgBlend::new(&ctx);
    let q = fixture_queries()[0];
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    assert!(
        close(got[0].intersection_blend, 0.5, DIST_ABS, DIST_REL),
        "equal distances blend at the fillet midpoint: {}",
        got[0].intersection_blend
    );
}

#[test]
fn union_nearer_second_operand_blends_below_half() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfCsgBlend::new(&ctx);
    let q = fixture_queries()[1];
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    assert!(
        got[0].union_blend < 0.5,
        "with a < b the union blend weighs the first operand more: {}",
        got[0].union_blend
    );
}

#[test]
fn union_nearer_first_operand_blends_above_half() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfCsgBlend::new(&ctx);
    let q = fixture_queries()[2];
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    assert!(
        got[0].union_blend > 0.5,
        "with a > b the union blend weighs the second operand more: {}",
        got[0].union_blend
    );
}

#[test]
fn clamp_saturation_gives_hard_blend() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfCsgBlend::new(&ctx);
    let q = fixture_queries()[3];
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    // |a - b| exceeds k so h = 0: the union distance is the exact min and the
    // blend collapses to the hard selection (a < b -> 0).
    assert!(
        close(got[0].union_distance, 0.0, DIST_ABS, DIST_REL),
        "saturated union distance is the hard min: {}",
        got[0].union_distance
    );
    assert!(
        close(got[0].union_blend, 0.0, DIST_ABS, DIST_REL),
        "saturated union blend is the hard 0: {}",
        got[0].union_blend
    );
}

#[test]
fn hard_fallback_zero_radius_matches_min_max() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfCsgBlend::new(&ctx);
    let q = fixture_queries()[4];
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    // k == 0 falls back to the hard operators: a = 2, b = 5.
    assert!(
        close(got[0].union_distance, 2.0, DIST_ABS, DIST_REL),
        "hard union distance is min(a, b): {}",
        got[0].union_distance
    );
    assert!(
        close(got[0].union_blend, 0.0, DIST_ABS, DIST_REL),
        "hard union blend is 0 when a <= b: {}",
        got[0].union_blend
    );
    assert!(
        close(got[0].intersection_distance, 5.0, DIST_ABS, DIST_REL),
        "hard intersection distance is max(a, b): {}",
        got[0].intersection_distance
    );
    assert!(
        close(got[0].intersection_blend, 1.0, DIST_ABS, DIST_REL),
        "hard intersection blend is 1 when a < b: {}",
        got[0].intersection_blend
    );
}

#[test]
fn hard_fallback_negative_radius_matches_min_max() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfCsgBlend::new(&ctx);
    let q = fixture_queries()[5];
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    // k < 0 falls back to the hard operators: a = 5, b = 2.
    assert!(
        close(got[0].union_distance, 2.0, DIST_ABS, DIST_REL),
        "hard union distance is min(a, b): {}",
        got[0].union_distance
    );
    assert!(
        close(got[0].union_blend, 1.0, DIST_ABS, DIST_REL),
        "hard union blend is 1 when a > b: {}",
        got[0].union_blend
    );
}

#[test]
fn subtraction_transition_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfCsgBlend::new(&ctx);
    let q = fixture_queries()[6];
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    assert!(
        got[0].subtraction_blend >= 0.0 && got[0].subtraction_blend <= 1.0,
        "the subtraction blend stays in the unit interval: {}",
        got[0].subtraction_blend
    );
}

#[test]
fn fixture_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfCsgBlend::new(&ctx);
    check(&ctx, &gpu, &fixture_queries());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfCsgBlend::new(&ctx);
    let mut state = 0x0f0e_0d0c_0b0a_0908_u64;
    let mut queries = fixture_queries();
    // Several workgroups' worth of random, well-conditioned triples pin all
    // three operators' distance and blend across a wide span of parameters.
    for _ in 0..512 {
        queries.push(random_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}
