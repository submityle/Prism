//! Real-device parity for the `ReSTIR` GI reconnection-Jacobian twin:
//! [`GpuRestirGiReconnectionJacobian`](prism_volumetric_gpu::restir_gi_reconnection_jacobian::GpuRestirGiReconnectionJacobian)
//! must reproduce the `CPU` golden
//! [`reconnection_jacobian`](prism_render_architecture::lighting::restir_gi::reconnection_jacobian) —
//! the cosine-weighted inverse-square change-of-measure ratio — across the
//! identity (`src == dst`) case, a hand-computed value, the reciprocal
//! relation, deliberate degenerate reconnections and a randomized sweep.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! [`reconnection_jacobian`](prism_render_architecture::lighting::restir_gi::reconnection_jacobian)
//! is public, so each `GPU` Jacobian is pinned directly against a host call: a
//! [`ShadingPoint`](prism_render_architecture::lighting::restir_gi::ShadingPoint)
//! pair and a
//! [`GiSample`](prism_render_architecture::lighting::restir_gi::GiSample) are
//! rebuilt from the query and evaluated on the host. The shading-point normals
//! and the sample `radiance` are irrelevant to the Jacobian, so they carry
//! placeholder values.
//!
//! # Parity criterion
//!
//! The Jacobian is a continuous `f32` ratio and is asserted within
//! `abs_diff <= 1e-4` or `rel_diff <= 1e-3`. A degenerate reconnection returns
//! an exact `0` on both sides, which the tolerance also admits.
//!
//! # Conditioning
//!
//! The randomized sweep keeps every reconnection well clear of the degenerate
//! thresholds — the sample point is separated from both shading points and the
//! sample normal is non-grazing — so a last-place difference in the shared
//! `+ - * / sqrt` sequence can never flip the rejection branch and desync the
//! two sides.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::lighting::restir_gi::reconnection_jacobian`；无第三方引擎源码或衍生代码。

use prism_render_architecture::lighting::restir_gi::{
    reconnection_jacobian, GiSample, ShadingPoint,
};
use prism_volumetric_gpu::restir_gi_reconnection_jacobian::{
    GpuRestirGiReconnectionJacobian, RestirGiReconnectionJacobianQuery,
    RestirGiReconnectionJacobianResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on the Jacobian. A `GPU` `sqrt`/divide may land a few
/// units in the last place from the scalar reference; `1e-4` admits that legal
/// slack while still failing a wrong port.
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

/// Computes the expected Jacobian in-host via the golden `reconnection_jacobian`:
/// the faithful oracle the `GPU` is pinned against. The shading-point normals
/// and sample `radiance` are unused by the Jacobian, so they carry placeholders.
fn oracle(q: &RestirGiReconnectionJacobianQuery) -> f32 {
    let src = ShadingPoint::new(q.src_position, [0.0, 0.0, 1.0]);
    let dst = ShadingPoint::new(q.dst_position, [0.0, 0.0, 1.0]);
    let sample = GiSample {
        sample_point: q.sample_point,
        sample_normal: q.sample_normal,
        radiance: [0.0; 3],
    };
    reconnection_jacobian(src, dst, &sample)
}

/// Pins one `GPU` Jacobian against the in-host oracle within tolerance.
fn check_sample(idx: usize, got: &RestirGiReconnectionJacobianResult, want: f32) {
    assert!(
        close(got.jacobian, want),
        "sample {idx} jacobian: gpu {} vs cpu {}",
        got.jacobian,
        want
    );
}

/// Dispatches every sample and pins each result against the oracle.
fn check(
    ctx: &GpuContext,
    gpu: &GpuRestirGiReconnectionJacobian,
    queries: &[RestirGiReconnectionJacobianQuery],
) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "one result per query must come back"
    );
    for (idx, (res, q)) in got.iter().zip(queries.iter()).enumerate() {
        check_sample(idx, res, oracle(q));
    }
}

/// A 64-bit linear-congruential generator (`PCG`-style multiplier); only
/// integer work, so no transcendental appears. Returns the raw high bits.
fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 33) as u32
}

/// Draws a coordinate in `[-2.0, 2.0)` at milli resolution from `state`.
fn coord(state: &mut u64) -> f32 {
    (lcg(state) % 4000) as f32 / 1000.0 - 2.0
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping restir_gi_reconnection_jacobian parity: no wgpu adapter");
        return;
    };
    let gpu = GpuRestirGiReconnectionJacobian::new(&ctx);
    // An empty batch must short-circuit on the host (a storage buffer cannot be
    // zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn identity_reconnection_is_unity() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRestirGiReconnectionJacobian::new(&ctx);
    // src == dst: the measure is unchanged, so J = 1.
    let q = RestirGiReconnectionJacobianQuery::new(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.3, -0.2, 2.0],
        [0.0, 0.0, -1.0],
    );
    let got = gpu.evaluate(&ctx, &[q]);
    assert!(
        close(got[0].jacobian, 1.0),
        "identity J = {}",
        got[0].jacobian
    );
    check(&ctx, &gpu, &[q]);
}

#[test]
fn hand_computed_value_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRestirGiReconnectionJacobian::new(&ctx);
    // Sample point at origin, normal +z. src straight above at z=2 (cosθ=1,
    // d²=4); dst along (3,0,4)/5 (cosθ=4/5, d²=25). J = (0.8/25)/(1/4) = 0.128.
    let q = RestirGiReconnectionJacobianQuery::new(
        [0.0, 0.0, 2.0],
        [3.0, 0.0, 4.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    );
    let got = gpu.evaluate(&ctx, &[q]);
    assert!(
        close(got[0].jacobian, 0.128),
        "hand J = {}",
        got[0].jacobian
    );
    check(&ctx, &gpu, &[q]);
}

#[test]
fn reciprocal_pair_multiplies_to_unity() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRestirGiReconnectionJacobian::new(&ctx);
    // J(a→b) and J(b→a) are reciprocals, so their product is 1. Both directions
    // are also pinned against the oracle individually by `check`.
    let a = [-0.5, 0.1, 0.0];
    let b = [0.6, -0.3, 0.0];
    let p = [0.1, 0.2, 2.0];
    let n = [0.0, 0.0, -1.0];
    let fwd = RestirGiReconnectionJacobianQuery::new(a, b, p, n);
    let bwd = RestirGiReconnectionJacobianQuery::new(b, a, p, n);
    let got = gpu.evaluate(&ctx, &[fwd, bwd]);
    assert!(
        close(got[0].jacobian * got[1].jacobian, 1.0),
        "reciprocal product {} * {}",
        got[0].jacobian,
        got[1].jacobian
    );
    check(&ctx, &gpu, &[fwd, bwd]);
}

#[test]
fn degenerate_reconnections_are_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRestirGiReconnectionJacobian::new(&ctx);
    // (a) Grazing source normal: direction perpendicular to the sample normal
    //     drives cos_src → 0, so den < GEOM_EPS and J = 0.
    let grazing = RestirGiReconnectionJacobianQuery::new(
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 2.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    );
    // (b) Coincident source and sample point: d_src2 < GEOM_EPS, so J = 0.
    let coincident = RestirGiReconnectionJacobianQuery::new(
        [0.5, 0.5, 0.5],
        [0.0, 0.0, 2.0],
        [0.5, 0.5, 0.5],
        [0.0, 0.0, 1.0],
    );
    let got = gpu.evaluate(&ctx, &[grazing, coincident]);
    assert!(
        close(got[0].jacobian, 0.0),
        "grazing J = {}",
        got[0].jacobian
    );
    assert!(
        close(got[1].jacobian, 0.0),
        "coincident J = {}",
        got[1].jacobian
    );
    check(&ctx, &gpu, &[grazing, coincident]);
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRestirGiReconnectionJacobian::new(&ctx);
    let mut state = 0x51ed_270b_a3c7_19e4_u64;
    let mut queries = Vec::new();
    // Several workgroups' worth of random reconnections, each conditioned so
    // both shading points are well separated from the sample point and the
    // sample normal is firmly non-grazing. That keeps both sides on the
    // compute branch, so the tolerance assertion holds and the degenerate
    // branch cannot flip.
    while queries.len() < 256 {
        let src = [coord(&mut state), coord(&mut state), coord(&mut state)];
        let dst = [coord(&mut state), coord(&mut state), coord(&mut state)];
        let sp = [coord(&mut state), coord(&mut state), coord(&mut state)];
        // A unit-ish normal drawn from a non-zero vector; reject near-zero
        // vectors so normalization is well defined.
        let nraw = [coord(&mut state), coord(&mut state), coord(&mut state)];
        let nlen2 = nraw[0] * nraw[0] + nraw[1] * nraw[1] + nraw[2] * nraw[2];
        if nlen2 < 0.25 {
            continue;
        }
        let inv = 1.0 / nlen2.sqrt();
        let n = [nraw[0] * inv, nraw[1] * inv, nraw[2] * inv];

        // Reconnection vectors and their squared lengths.
        let ts = [src[0] - sp[0], src[1] - sp[1], src[2] - sp[2]];
        let td = [dst[0] - sp[0], dst[1] - sp[1], dst[2] - sp[2]];
        let ds2 = ts[0] * ts[0] + ts[1] * ts[1] + ts[2] * ts[2];
        let dd2 = td[0] * td[0] + td[1] * td[1] + td[2] * td[2];
        // Keep both points well separated from the sample point (d² >= 0.25,
        // i.e. d >= 0.5, far above sqrt(GEOM_EPS) = 1e-4).
        if ds2 < 0.25 || dd2 < 0.25 {
            continue;
        }
        // Non-grazing source cosine so `den` stays firmly above GEOM_EPS.
        let cos_src = (n[0] * ts[0] + n[1] * ts[1] + n[2] * ts[2]).abs() / ds2.sqrt();
        if cos_src < 0.1 {
            continue;
        }
        queries.push(RestirGiReconnectionJacobianQuery::new(src, dst, sp, n));
    }
    check(&ctx, &gpu, &queries);
}
