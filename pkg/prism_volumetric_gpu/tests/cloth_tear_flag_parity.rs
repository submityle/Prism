//! Real-device parity test for the cloth constraint-tearing twin.
//!
//! Each case evaluates one or more [`ClothTearFlagQuery`] values on the GPU and
//! pins the returned [`ClothTearFlagResult`] against an independent `f32`
//! reimplementation of `prism_physics_core::soft::damage::tearing::tear_flag`.
//! The oracle is reimplemented here from first principles; this test never
//! depends on the golden crate.
//!
//! The published outputs `tear` and `valid` are discrete `u32` flags, so they
//! are pinned with an exact `==`. The only continuous intermediate is the
//! tensile strain; the named fixtures place the strain either well above or
//! well below the break threshold (and one case exactly on it, confirming the
//! strict `>` does not tear), and the random sweep keeps the sampled strain a
//! safe margin clear of the `strain == break_strain` knee so a few units in the
//! last place cannot flip the boolean.
//!
//! Every case short-circuits to a skip when no headless adapter is available,
//! so the suite is inert on a machine without a GPU and exercises the real
//! device elsewhere.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::soft::damage::tearing::tear_flag`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::cloth_tear_flag::{
    ClothTearFlagQuery, ClothTearFlagResult, GpuClothTearFlag,
};
use prism_volumetric_gpu::GpuContext;

/// Smallest rest length considered valid; mirrors `EPS_REST` in the golden.
const EPS_REST: f32 = 1.0e-9;

/// Independent `f32` reimplementation of the tearing decision for one query, in
/// the same arithmetic order as the kernel: a `NaN`/negative threshold maps to
/// `+inf`, a degenerate rest length never tears, otherwise the tensile strain
/// is compared to the sanitized threshold with a strict `>`.
fn oracle(query: &ClothTearFlagQuery) -> ClothTearFlagResult {
    let break_strain = if query.break_strain.is_nan() || query.break_strain < 0.0 {
        f32::INFINITY
    } else {
        query.break_strain
    };
    let tears = if query.rest_length <= EPS_REST {
        false
    } else {
        let strain = (query.length - query.rest_length) / query.rest_length;
        strain > break_strain
    };
    ClothTearFlagResult {
        tear: u32::from(tears),
        valid: 1,
    }
}

/// Pins one `GPU` result against the independent host oracle: both discrete
/// flags must match exactly.
fn pin(idx: usize, query: &ClothTearFlagQuery, result: &ClothTearFlagResult) {
    let want = oracle(query);
    assert_eq!(
        result.tear, want.tear,
        "query {idx}: tear gpu={} oracle={} (rest={} len={} bs={})",
        result.tear, want.tear, query.rest_length, query.length, query.break_strain
    );
    assert_eq!(
        result.valid, want.valid,
        "query {idx}: valid gpu={} oracle={}",
        result.valid, want.valid
    );
}

/// Evaluates `queries` on-device and pins every result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuClothTearFlag, queries: &[ClothTearFlagQuery]) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (query, result)) in queries.iter().zip(got.iter()).enumerate() {
        pin(idx, query, result);
    }
}

/// 64-bit linear-congruential step (`Knuth`/`PCG` constants), returning the
/// high word so the stream has good spread without any transcendental math.
fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 32) as u32
}

/// A deterministic pseudo-random `f32` in `[0, 1]`.
fn unit01(state: &mut u64) -> f32 {
    lcg(state) as f32 / u32::MAX as f32
}

/// A deterministic, well-conditioned random query: a positive rest length well
/// above `EPS_REST`, a break strain in `[0.1, 2.0]`, and a current length
/// chosen so the tensile strain sits a safe margin (`>= 0.1`) either above or
/// below the threshold, never on the knee.
fn rand_query(state: &mut u64) -> ClothTearFlagQuery {
    let rest_length = 0.5 + unit01(state) * 4.5;
    let break_strain = 0.1 + unit01(state) * 1.9;
    let above = lcg(state) & 1 == 0;
    let strain = if above {
        // Clearly over the threshold: [bs + 0.1, bs + 1.5].
        break_strain + 0.1 + unit01(state) * 1.4
    } else {
        // Clearly under the threshold: [-0.5, bs - 0.1].
        let lo = -0.5;
        let hi = break_strain - 0.1;
        lo + unit01(state) * (hi - lo)
    };
    let length = rest_length * (1.0 + strain);
    ClothTearFlagQuery::new(rest_length, length, break_strain)
}

#[test]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothTearFlag::new(&ctx);
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty input must return an empty vector");
}

#[test]
fn taut_edge_tears() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothTearFlag::new(&ctx);
    // strain = (2.0 - 1.0) / 1.0 = 1.0 > 0.5 -> tears.
    let query = ClothTearFlagQuery::new(1.0, 2.0, 0.5);
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&query));
    assert_eq!(got[0].tear, 1, "an over-stretched edge should tear");
    pin(0, &query, &got[0]);
}

#[test]
fn slack_edge_does_not_tear() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothTearFlag::new(&ctx);
    // strain = 0.2 < 0.5 -> survives.
    let query = ClothTearFlagQuery::new(1.0, 1.2, 0.5);
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&query));
    assert_eq!(got[0].tear, 0, "a lightly stretched edge should survive");
    pin(0, &query, &got[0]);
}

#[test]
fn strain_exactly_at_threshold_does_not_tear() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothTearFlag::new(&ctx);
    // strain = (1.5 - 1.0) / 1.0 = 0.5 == break_strain; strict `>` keeps it.
    let query = ClothTearFlagQuery::new(1.0, 1.5, 0.5);
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&query));
    assert_eq!(
        got[0].tear, 0,
        "a strain exactly on the threshold must not tear (strict comparison)"
    );
    pin(0, &query, &got[0]);
}

#[test]
fn nan_break_strain_never_tears() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothTearFlag::new(&ctx);
    // strain = 4.0, but a NaN threshold sanitizes to +inf -> nothing tears.
    let query = ClothTearFlagQuery::new(1.0, 5.0, f32::NAN);
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&query));
    assert_eq!(got[0].tear, 0, "a NaN threshold must tear nothing");
    pin(0, &query, &got[0]);
}

#[test]
fn negative_break_strain_never_tears() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothTearFlag::new(&ctx);
    // A negative threshold sanitizes to +inf -> nothing tears, despite strain = 4.0.
    let query = ClothTearFlagQuery::new(1.0, 5.0, -1.0);
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&query));
    assert_eq!(got[0].tear, 0, "a negative threshold must tear nothing");
    pin(0, &query, &got[0]);
}

#[test]
fn degenerate_rest_never_tears() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothTearFlag::new(&ctx);
    // rest_length <= EPS_REST is degenerate: never tears, even at huge strain.
    let query = ClothTearFlagQuery::new(1.0e-12, 5.0, 0.1);
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&query));
    assert_eq!(got[0].tear, 0, "a degenerate rest length must never tear");
    assert_eq!(
        got[0].valid, 1,
        "a degenerate edge is still a valid decision"
    );
    pin(0, &query, &got[0]);
}

#[test]
fn compression_does_not_tear() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothTearFlag::new(&ctx);
    // strain = (1.0 - 2.0) / 2.0 = -0.5 < 0.1 -> compression never tears.
    let query = ClothTearFlagQuery::new(2.0, 1.0, 0.1);
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&query));
    assert_eq!(got[0].tear, 0, "a compressed edge should never tear");
    pin(0, &query, &got[0]);
}

#[test]
fn stride_regression_two_element_batch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothTearFlag::new(&ctx);
    // A tearing edge followed by a surviving one: a wrong per-element stride
    // would cross-contaminate the two decisions.
    let queries = [
        ClothTearFlagQuery::new(1.0, 2.0, 0.5),
        ClothTearFlagQuery::new(1.0, 1.1, 0.5),
    ];
    let got = gpu.evaluate(&ctx, &queries);
    assert_eq!(got.len(), 2, "both results must be returned");
    assert_eq!(got[0].tear, 1, "first edge tears");
    assert_eq!(got[1].tear, 0, "second edge survives");
    check(&ctx, &gpu, &queries);
}

#[test]
fn mixed_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothTearFlag::new(&ctx);
    let mut queries = vec![
        ClothTearFlagQuery::new(1.0, 2.0, 0.5),
        ClothTearFlagQuery::new(1.0, 1.2, 0.5),
        ClothTearFlagQuery::new(1.0, 1.5, 0.5),
        ClothTearFlagQuery::new(1.0e-12, 5.0, 0.1),
        ClothTearFlagQuery::new(1.0, 5.0, f32::NAN),
        ClothTearFlagQuery::new(2.0, 1.0, 0.1),
    ];
    let mut state: u64 = 0x5151_7EA2_C10B_1234;
    for _ in 0..48 {
        queries.push(rand_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothTearFlag::new(&ctx);
    let mut state: u64 = 0x0CEA_1F10_7A6B_9D55;
    let queries: Vec<ClothTearFlagQuery> = (0..512).map(|_| rand_query(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}
