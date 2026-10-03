//! Real-device parity for the soft-barrier contact twin:
//! [`GpuBarrierContactForce`](prism_volumetric_gpu::barrier_contact_force::GpuBarrierContactForce)
//! must reproduce the `CPU` golden scalar pair
//! [`barrier_energy`](prism_render_architecture::hair::barrier_contact::barrier_energy)
//! and
//! [`barrier_force_magnitude`](prism_render_architecture::hair::barrier_contact::barrier_force_magnitude)
//! across the free region, the active window, the floor-clamp, a non-finite
//! distance, a zero stiffness, every `sanitize` fallback branch, a mixed batch,
//! and a randomized sweep compared query-for-query.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The expected energy and force are produced by calling the golden
//! [`barrier_energy`](prism_render_architecture::hair::barrier_contact::barrier_energy)
//! and
//! [`barrier_force_magnitude`](prism_render_architecture::hair::barrier_contact::barrier_force_magnitude)
//! directly with the original (possibly non-finite) `d`, so a `GPU == oracle`
//! pass establishes `GPU == golden`. The `friction_mu` field is irrelevant to
//! both and is set to a fixed `0.3` when building the params.
//!
//! # Parity criterion
//!
//! The energy and force thread through divides, so a `GPU` divide may land a few
//! units in the last place from the scalar reference; they are asserted within
//! `abs_diff <= 1e-5` or `rel_diff <= 1e-4`.
//!
//! # Conditioning
//!
//! Every randomized fixture is deliberately away from a discrete tie: the
//! distance is kept clear of the free-region boundary `dhat`, the floor
//! `d_floor`, and the `hi = dhat * 0.5` guard so the `CPU` and `GPU` take the
//! same `sanitize` and free-region branches.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::hair::barrier_contact`；无第三方引擎源码或衍生代码。

use prism_render_architecture::hair::barrier_contact::{
    barrier_energy, barrier_force_magnitude, BarrierParams,
};
use prism_volumetric_gpu::barrier_contact_force::{
    BarrierContactForceQuery, BarrierContactForceResult, GpuBarrierContactForce,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on a scalar. A `GPU` divide may land a few units in the
/// last place from the scalar reference; `1e-5` admits that legal slack while
/// still failing a wrong port.
const EPS: f32 = 1.0e-5;

/// Relative parity bound, applied for larger magnitudes where a few units in the
/// last place exceed the absolute floor.
const REL: f32 = 1.0e-4;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
}

/// Builds the golden params for a query, with the energy/force-irrelevant
/// `friction_mu` fixed at `0.3`.
fn params_of(q: &BarrierContactForceQuery) -> BarrierParams {
    BarrierParams {
        dhat: q.dhat,
        stiffness: q.stiffness,
        d_floor: q.d_floor,
        friction_mu: 0.3,
    }
}

/// The in-host oracle: the golden energy and force for a query, called with the
/// original `d` so parity is honest.
fn oracle(q: &BarrierContactForceQuery) -> BarrierContactForceResult {
    let p = params_of(q);
    BarrierContactForceResult {
        energy: barrier_energy(q.d, p),
        force: barrier_force_magnitude(q.d, p),
    }
}

/// Pins one `GPU` result against the in-host oracle: both scalars within
/// tolerance.
fn check_query(idx: usize, got: &BarrierContactForceResult, want: &BarrierContactForceResult) {
    assert!(
        close(got.energy, want.energy),
        "query {idx} energy: gpu {} vs cpu {}",
        got.energy,
        want.energy
    );
    assert!(
        close(got.force, want.force),
        "query {idx} force: gpu {} vs cpu {}",
        got.force,
        want.force
    );
}

/// Dispatches `queries` and pins every result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuBarrierContactForce, queries: &[BarrierContactForceQuery]) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (idx, (g, q)) in got.iter().zip(queries.iter()).enumerate() {
        let want = oracle(q);
        check_query(idx, g, &want);
    }
}

/// A small host-side `LCG`, used only to drive fixture generation; the kernel
/// and oracle stay free of any transcendental math.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Lcg {
        Lcg { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.state >> 33) as u32
    }

    fn next_f32(&mut self, lo: f32, hi: f32) -> f32 {
        let u = (self.next_u32() >> 8) as f32 * (1.0 / 16_777_216.0);
        lo + (hi - lo) * u
    }
}

/// Builds a well-conditioned random query: the distance is kept clear of the
/// free-region boundary, the floor, and the `hi` guard, and `d_floor` stays
/// comfortably below `hi` so no `sanitize` branch sits on a tie.
fn conditioned_query(rng: &mut Lcg) -> BarrierContactForceQuery {
    let dhat = rng.next_f32(0.008, 0.03);
    let hi = dhat * 0.5;
    // Keep the floor strictly under hi (no truncation tie) and above MIN.
    let d_floor = rng.next_f32(0.0004, hi * 0.6);
    let stiffness = rng.next_f32(0.2, 60.0);
    // Choose the free region or the active window, each with a safe margin.
    let d = if (rng.next_u32() & 1) == 0 {
        // Free region, comfortably past dhat.
        rng.next_f32(dhat * 1.2, dhat * 2.5)
    } else {
        // Active window, well inside (d_floor, dhat).
        let span = dhat - d_floor;
        rng.next_f32(d_floor + span * 0.12, dhat - span * 0.12)
    };
    BarrierContactForceQuery::new(d, dhat, stiffness, d_floor)
}

#[test]
fn empty_batch_returns_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBarrierContactForce::new(&ctx);
    // An empty batch issues no dispatch (a storage buffer cannot be zero-sized)
    // and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn distance_past_dhat_is_free_region() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBarrierContactForce::new(&ctx);
    // d >= dhat: both outputs are exactly zero.
    let q = BarrierContactForceQuery::new(0.05, 0.01, 50.0, 0.001);
    let got = gpu.evaluate(&ctx, &[q]);
    let want = oracle(&q);
    assert!(
        want.energy.abs() <= EPS,
        "fixture must be in the free region"
    );
    assert!(
        want.force.abs() <= EPS,
        "fixture must be in the free region"
    );
    check_query(0, &got[0], &want);
}

#[test]
fn distance_inside_active_window() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBarrierContactForce::new(&ctx);
    // d strictly inside (d_floor, dhat): positive energy and force.
    let q = BarrierContactForceQuery::new(0.005, 0.01, 50.0, 0.001);
    let got = gpu.evaluate(&ctx, &[q]);
    let want = oracle(&q);
    assert!(want.energy > 0.0, "fixture must be active");
    assert!(want.force > 0.0, "fixture must be active");
    check_query(0, &got[0], &want);
}

#[test]
fn distance_below_floor_is_clamped() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBarrierContactForce::new(&ctx);
    // d < d_floor: the evaluation clamps to d_floor (large-but-finite output).
    let q = BarrierContactForceQuery::new(0.0002, 0.01, 50.0, 0.001);
    let got = gpu.evaluate(&ctx, &[q]);
    let want = oracle(&q);
    assert!(want.energy.is_finite() && want.energy > 0.0);
    assert!(want.force.is_finite() && want.force > 0.0);
    check_query(0, &got[0], &want);
}

#[test]
fn nonfinite_distance_is_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBarrierContactForce::new(&ctx);
    // Both NaN and +inf distances are treated as "far": zero energy and force.
    let queries = [
        BarrierContactForceQuery::new(f32::NAN, 0.01, 50.0, 0.001),
        BarrierContactForceQuery::new(f32::INFINITY, 0.01, 50.0, 0.001),
    ];
    let got = gpu.evaluate(&ctx, &queries);
    for (idx, (g, q)) in got.iter().zip(queries.iter()).enumerate() {
        let want = oracle(q);
        assert!(want.energy.abs() <= EPS, "golden non-finite d energy");
        assert!(want.force.abs() <= EPS, "golden non-finite d force");
        check_query(idx, g, &want);
    }
}

#[test]
fn zero_stiffness_is_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBarrierContactForce::new(&ctx);
    // A zero stiffness scales both outputs to zero even inside the window.
    let q = BarrierContactForceQuery::new(0.005, 0.01, 0.0, 0.001);
    let got = gpu.evaluate(&ctx, &[q]);
    let want = oracle(&q);
    assert!(want.energy.abs() <= EPS, "zero stiffness energy");
    assert!(want.force.abs() <= EPS, "zero stiffness force");
    check_query(0, &got[0], &want);
}

#[test]
fn nonpositive_dhat_falls_back() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBarrierContactForce::new(&ctx);
    // dhat <= 0 is sanitized to the default 1e-2, so an active d yields a
    // positive response against that default window.
    let q = BarrierContactForceQuery::new(0.004, -1.0, 10.0, 0.0008);
    let got = gpu.evaluate(&ctx, &[q]);
    let want = oracle(&q);
    assert!(want.energy > 0.0, "fallback dhat should re-open the window");
    check_query(0, &got[0], &want);
}

#[test]
fn floor_truncated_by_hi_guard() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBarrierContactForce::new(&ctx);
    // d_floor > dhat * 0.5 is pulled down to the hi guard; d sits inside the
    // resulting window.
    let q = BarrierContactForceQuery::new(0.006, 0.01, 25.0, 0.02);
    let got = gpu.evaluate(&ctx, &[q]);
    let want = oracle(&q);
    assert!(
        want.energy > 0.0,
        "window must stay open after the hi guard"
    );
    check_query(0, &got[0], &want);
}

#[test]
fn nonpositive_floor_falls_back() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBarrierContactForce::new(&ctx);
    // d_floor <= 0 is sanitized to the default 1e-3.
    let q = BarrierContactForceQuery::new(0.005, 0.01, 30.0, -0.5);
    let got = gpu.evaluate(&ctx, &[q]);
    let want = oracle(&q);
    assert!(want.energy > 0.0, "fallback floor keeps the window open");
    check_query(0, &got[0], &want);
}

#[test]
fn healthy_mid_range_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBarrierContactForce::new(&ctx);
    let q = BarrierContactForceQuery::new(0.004, 0.012, 12.0, 0.0008);
    check(&ctx, &gpu, &[q]);
}

#[test]
fn mixed_batch_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBarrierContactForce::new(&ctx);
    // Several of the above regimes dispatched together so the per-thread
    // indexing and the contiguous output slots are both exercised.
    let queries = [
        BarrierContactForceQuery::new(0.05, 0.01, 50.0, 0.001),
        BarrierContactForceQuery::new(0.005, 0.01, 50.0, 0.001),
        BarrierContactForceQuery::new(0.0002, 0.01, 50.0, 0.001),
        BarrierContactForceQuery::new(f32::NAN, 0.01, 50.0, 0.001),
        BarrierContactForceQuery::new(0.005, 0.01, 0.0, 0.001),
        BarrierContactForceQuery::new(0.004, -1.0, 10.0, 0.0008),
        BarrierContactForceQuery::new(0.006, 0.01, 25.0, 0.02),
        BarrierContactForceQuery::new(0.005, 0.01, 30.0, -0.5),
        BarrierContactForceQuery::new(0.004, 0.012, 12.0, 0.0008),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBarrierContactForce::new(&ctx);
    let mut rng = Lcg::new(0x0ba7_7126_c047_ac7e_u64);
    let mut queries = Vec::new();
    // Several workgroups' worth of well-conditioned queries pin every reported
    // result across a wide span of distances and params.
    for _ in 0..384 {
        queries.push(conditioned_query(&mut rng));
    }
    check(&ctx, &gpu, &queries);
}
