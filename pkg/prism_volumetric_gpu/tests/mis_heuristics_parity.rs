//! Real-device parity for the multiple-importance-sampling weights twin:
//! [`GpuMisHeuristics`](prism_volumetric_gpu::mis_heuristics::GpuMisHeuristics)
//! must reproduce the two `CPU` golden closed forms
//! [`balance_heuristic`](prism_render_architecture::reference_pt::mis::balance_heuristic)
//! and
//! [`power_heuristic`](prism_render_architecture::reference_pt::mis::power_heuristic)
//! across the vanishing-density degeneracies, the equal-density midpoint, the
//! lopsided-ratio regime that separates the two heuristics, and a randomized
//! sweep compared query-for-query.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! Both golden functions are public, so the expected values come straight from
//! calling [`balance_heuristic`] and [`power_heuristic`] on the host — no
//! reimplementation of the formula. A `GPU == golden` pass is therefore direct
//! evidence the ported kernel computes the same weights.
//!
//! # Parity criterion
//!
//! The golden promotes both densities to `f64` before dividing; `WGSL` has no
//! `f64`, so the kernel divides in `f32`. The balance weight is a single divide
//! and is pinned within `abs_diff <= 1e-5` or `rel_diff <= 1e-4`; the power
//! weight squares first, so it is pinned within the looser `abs_diff <= 1e-4`
//! or `rel_diff <= 1e-3`. Both use a `1e-6` relative-error floor. The
//! both-densities-zero branch returns an exact `0` on both sides.
//!
//! # Conditioning
//!
//! Every fixture and random draw keeps the densities inside `[0, 1e3]`, where
//! the squared sum stays far from `f32` overflow and the `f32` and `f64`
//! divides agree within the documented tolerance.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::reference_pt::mis`；无第三方引擎源码或衍生代码。

use prism_render_architecture::reference_pt::mis::{balance_heuristic, power_heuristic};
use prism_volumetric_gpu::mis_heuristics::{
    GpuMisHeuristics, MisHeuristicsQuery, MisHeuristicsResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on the balance weight, a single divide.
const BALANCE_EPS: f32 = 1.0e-5;

/// Relative parity bound on the balance weight.
const BALANCE_REL: f32 = 1.0e-4;

/// Absolute parity bound on the power weight, which squares the densities first
/// and so admits a wider last-place slack.
const POWER_EPS: f32 = 1.0e-4;

/// Relative parity bound on the power weight.
const POWER_REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within the given absolute or relative
/// bound.
fn close(a: f32, b: f32, abs_eps: f32, rel_eps: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= abs_eps || rel <= rel_eps
}

/// Computes the expected result straight from the public golden functions, the
/// faithful oracle the `GPU` is pinned against.
fn oracle(q: &MisHeuristicsQuery) -> MisHeuristicsResult {
    MisHeuristicsResult {
        balance: balance_heuristic(q.pdf_a, q.pdf_b),
        power: power_heuristic(q.pdf_a, q.pdf_b),
    }
}

/// Pins one `GPU` result against the in-host oracle: the balance weight at the
/// tight tolerance and the power weight at the squared-magnitude tolerance.
fn check_one(idx: usize, got: &MisHeuristicsResult, want: &MisHeuristicsResult) {
    assert!(
        close(got.balance, want.balance, BALANCE_EPS, BALANCE_REL),
        "query {idx} balance: gpu {} vs cpu {}",
        got.balance,
        want.balance
    );
    assert!(
        close(got.power, want.power, POWER_EPS, POWER_REL),
        "query {idx} power: gpu {} vs cpu {}",
        got.power,
        want.power
    );
}

/// Dispatches every query and pins each result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuMisHeuristics, queries: &[MisHeuristicsQuery]) {
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

/// Draws a density in `[0.0, 1000.0]` at milli resolution, inside the range
/// where the `f32` kernel and the `f64` golden agree.
fn density(state: &mut u64) -> f32 {
    (lcg(state) % 1_000_001) as f32 / 1000.0
}

/// Draws a random, well-conditioned query from `state`.
fn random_query(state: &mut u64) -> MisHeuristicsQuery {
    MisHeuristicsQuery::new(density(state), density(state))
}

/// The deterministic fixture batch: both degeneracies, each single-zero case,
/// the equal-density midpoint and a spread of lopsided and mid ratios.
fn fixture_queries() -> Vec<MisHeuristicsQuery> {
    vec![
        // both zero -> both weights 0 (the <= 0 guard fires).
        MisHeuristicsQuery::new(0.0, 0.0),
        // a > 0, b = 0 -> a dominates, both weights 1.
        MisHeuristicsQuery::new(3.0, 0.0),
        // a = 0, b > 0 -> a vanishes, both weights 0.
        MisHeuristicsQuery::new(0.0, 7.0),
        // equal densities -> balance 0.5, power 0.5.
        MisHeuristicsQuery::new(2.0, 2.0),
        // equal densities at a different magnitude -> still 0.5 / 0.5.
        MisHeuristicsQuery::new(250.0, 250.0),
        // lopsided a >> b -> power pushes closer to 1 than balance.
        MisHeuristicsQuery::new(1000.0, 0.1),
        // lopsided b >> a -> power pushes closer to 0 than balance.
        MisHeuristicsQuery::new(0.1, 1000.0),
        // mild asymmetry.
        MisHeuristicsQuery::new(3.0, 1.0),
        // another mid ratio.
        MisHeuristicsQuery::new(10.0, 40.0),
        // small magnitudes.
        MisHeuristicsQuery::new(0.002, 0.006),
        // near the upper bound of the conditioned range.
        MisHeuristicsQuery::new(999.5, 500.25),
    ]
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping mis_heuristics parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuMisHeuristics::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn both_densities_zero_is_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMisHeuristics::new(&ctx);
    let q = MisHeuristicsQuery::new(0.0, 0.0);
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    // The <= 0 guard returns an exact zero on both sides.
    assert!(close(got[0].balance, 0.0, BALANCE_EPS, BALANCE_REL));
    assert!(close(got[0].power, 0.0, POWER_EPS, POWER_REL));
}

#[test]
fn one_density_zero_endpoints() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMisHeuristics::new(&ctx);
    // a > 0, b = 0 -> both weights 1; a = 0, b > 0 -> both weights 0.
    let a_only = MisHeuristicsQuery::new(5.0, 0.0);
    let b_only = MisHeuristicsQuery::new(0.0, 5.0);
    let got = gpu.evaluate(&ctx, &[a_only, b_only]);
    assert_eq!(got.len(), 2);
    check_one(0, &got[0], &oracle(&a_only));
    check_one(1, &got[1], &oracle(&b_only));
    assert!(close(got[0].balance, 1.0, BALANCE_EPS, BALANCE_REL));
    assert!(close(got[0].power, 1.0, POWER_EPS, POWER_REL));
    assert!(close(got[1].balance, 0.0, BALANCE_EPS, BALANCE_REL));
    assert!(close(got[1].power, 0.0, POWER_EPS, POWER_REL));
}

#[test]
fn equal_densities_are_one_half() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMisHeuristics::new(&ctx);
    let q = MisHeuristicsQuery::new(42.0, 42.0);
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    // Equal densities split both heuristics evenly.
    assert!(close(got[0].balance, 0.5, BALANCE_EPS, BALANCE_REL));
    assert!(close(got[0].power, 0.5, POWER_EPS, POWER_REL));
}

#[test]
fn balance_is_symmetric_and_sums_to_one() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMisHeuristics::new(&ctx);
    // Dispatching both orderings must give complementary weights that sum to 1
    // whenever at least one density is positive, for both heuristics.
    let forward = MisHeuristicsQuery::new(12.0, 37.0);
    let reverse = MisHeuristicsQuery::new(37.0, 12.0);
    let got = gpu.evaluate(&ctx, &[forward, reverse]);
    assert_eq!(got.len(), 2);
    check_one(0, &got[0], &oracle(&forward));
    check_one(1, &got[1], &oracle(&reverse));
    assert!(
        close(
            got[0].balance + got[1].balance,
            1.0,
            BALANCE_EPS,
            BALANCE_REL
        ),
        "balance(a,b) + balance(b,a) must be 1"
    );
    assert!(
        close(got[0].power + got[1].power, 1.0, POWER_EPS, POWER_REL),
        "power(a,b) + power(b,a) must be 1"
    );
}

#[test]
fn lopsided_ratio_separates_the_heuristics() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMisHeuristics::new(&ctx);
    // When a >> b the power heuristic sharpens the dominance: its weight for the
    // larger density sits strictly above the balance weight (and both above
    // 0.5).
    let q = MisHeuristicsQuery::new(1000.0, 0.1);
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    assert!(
        got[0].power > got[0].balance,
        "power should dominate harder than balance when a >> b"
    );
    assert!(
        got[0].balance > 0.5,
        "the larger density keeps the balance weight above half"
    );
}

#[test]
fn fixture_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMisHeuristics::new(&ctx);
    check(&ctx, &gpu, &fixture_queries());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMisHeuristics::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    let mut queries = fixture_queries();
    // Several workgroups' worth of random, well-conditioned queries pin every
    // reported balance and power weight across the full `[0, 1e3]` span.
    for _ in 0..512 {
        queries.push(random_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}
