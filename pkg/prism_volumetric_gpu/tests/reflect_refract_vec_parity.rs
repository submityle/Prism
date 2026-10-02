//! Real-device parity for the reflect/refract optical-response twin:
//! [`GpuReflectRefractVec`](prism_volumetric_gpu::reflect_refract_vec::GpuReflectRefractVec)
//! must reproduce the `CPU` golden
//! [`reflect_refract_vec`](prism_render_architecture::particle::reflect_refract_vec)
//! across the mirror reflection, the vector-form `Snell` refraction (both below
//! the critical angle and past it, where total internal reflection fires), the
//! standalone total-internal-reflection flag, the `Schlick` `Fresnel`
//! reflectance and its normal-incidence seed, and the coefficient-of-restitution
//! velocity bounce, plus a randomized batch of clearly-conditioned queries
//! compared element-for-element.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each query is a fixed, non-reorderable sequence of multiplies, adds, divides
//! and one `sqrt`, so `CPU` and `GPU` evaluate the same closed form in the same
//! order. They are not bit-exact: a `GPU` may fuse a multiply-add the scalar
//! reference leaves separate, perturbing the low mantissa bits by a few units in
//! the last place. The comparison therefore allows `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3` on every continuous `f32` field, while the
//! `refract_valid` and `is_total_internal_reflection` classification flags are
//! compared bit-exactly.
//!
//! # Conditioning
//!
//! Every fixture is deliberately well away from the one branch tie — the
//! `Snell` discriminant `k = 1 - eta² (1 - cos_i²)` at the critical angle.
//! Below-critical fixtures keep `k` clearly positive, the total-internal-
//! reflection fixtures keep it clearly negative, and the randomized batch
//! rejection-samples until `|k| > 0.08`, so `CPU` and `GPU` land on the same
//! side of the `k < 0` test regardless of a few units in the last place of
//! slack. Refractive indices are kept positive so `n1 + n2` never approaches
//! the `fresnel_r0` divide's zero, and normals are unit length so the
//! restitution normalise never trips its epsilon guard.
//!
//! Provenance: twinned from this repository's
//! [`reflect_refract_vec`](prism_render_architecture::particle::reflect_refract_vec);
//! no third-party engine source or derived code.

use prism_render_architecture::particle::reflect_refract_vec::{
    fresnel_schlick_r0, fresnel_schlick_reflectance, is_total_internal_reflection, reflect,
    reflect_with_restitution, refract,
};
use prism_volumetric_gpu::reflect_refract_vec::{
    GpuReflectRefractVec, ReflectRefractQuery, ReflectRefractResult,
};
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

/// Asserts two 3-vectors agree channel-for-channel within the parity bound.
fn close_vec(label: &str, idx: usize, got: [f32; 3], want: [f32; 3]) {
    assert!(
        close(got[0], want[0]) && close(got[1], want[1]) && close(got[2], want[2]),
        "query {idx} {label}: gpu ({}, {}, {}) vs cpu ({}, {}, {})",
        got[0],
        got[1],
        got[2],
        want[0],
        want[1],
        want[2]
    );
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

/// A pseudo-random value in `[-span, span)` drawn from `state`.
fn signed(state: &mut u64, span: f32) -> f32 {
    (lcg(state) * 2.0 - 1.0) * span
}

/// A pseudo-random 3-vector with each component in `[-span, span)`.
fn rand_vec3(state: &mut u64, span: f32) -> [f32; 3] {
    [
        signed(state, span),
        signed(state, span),
        signed(state, span),
    ]
}

/// The squared length of a 3-vector.
fn len_sq(v: [f32; 3]) -> f32 {
    v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
}

/// The dot product of two 3-vectors.
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// The unit vector in the direction of `v`, via `f32::sqrt` (no transcendental).
fn unit(v: [f32; 3]) -> [f32; 3] {
    let inv = 1.0 / len_sq(v).sqrt();
    [v[0] * inv, v[1] * inv, v[2] * inv]
}

/// Builds a clearly-conditioned query by rejection sampling: `incident` and
/// `normal` are well-spread unit vectors, `eta` spans both the entering and the
/// exiting regime, and the pair is accepted only when the `Snell` discriminant
/// `k = 1 - eta² (1 - cos_i²)` sits a safe margin (`|k| > 0.08`) off the
/// critical-angle tie, so `CPU` and `GPU` share the `k < 0` branch. Indices are
/// kept positive so `n1 + n2` clears the `fresnel_r0` divide.
fn well_conditioned(state: &mut u64) -> ReflectRefractQuery {
    loop {
        let raw_i = rand_vec3(state, 1.0);
        let raw_n = rand_vec3(state, 1.0);
        // Reject near-zero vectors so the unit normalise is well defined.
        if len_sq(raw_i) < 0.09 || len_sq(raw_n) < 0.09 {
            continue;
        }
        let incident = unit(raw_i);
        let normal = unit(raw_n);
        let eta = 0.4 + lcg(state) * 1.4;
        let cos_i = -dot(incident, normal);
        let k = 1.0 - eta * eta * (1.0 - cos_i * cos_i);
        // Stay a safe margin off the critical-angle branch tie.
        if k.abs() < 0.08 {
            continue;
        }
        let velocity = rand_vec3(state, 3.0);
        let restitution = lcg(state);
        let cos_theta = lcg(state);
        let r0 = 0.02 + lcg(state) * 0.3;
        let n1 = 1.0 + lcg(state) * 1.5;
        let n2 = 1.0 + lcg(state) * 1.5;
        return ReflectRefractQuery::new(
            incident,
            normal,
            velocity,
            eta,
            restitution,
            cos_theta,
            r0,
            n1,
            n2,
        );
    }
}

/// Pins one `GPU` result against the `CPU` golden for `query`: the mirror
/// reflection, the `Snell` transmission with its validity flag, the standalone
/// total-internal-reflection flag, the `Schlick` reflectance and its seed, and
/// the restitution bounce must all agree within bound (flags bit-exactly).
fn pin(idx: usize, query: &ReflectRefractQuery, got: &ReflectRefractResult) {
    let want_reflect = reflect(query.incident, query.normal);
    close_vec("reflect_dir", idx, got.reflect_dir, want_reflect);

    let want_refract = refract(query.incident, query.normal, query.eta);
    assert_eq!(
        got.refract_valid,
        want_refract.is_some(),
        "query {idx} refract_valid: gpu {} vs cpu {}",
        got.refract_valid,
        want_refract.is_some()
    );
    match want_refract {
        Some(dir) => close_vec("refract_dir", idx, got.refract_dir, dir),
        None => close_vec("refract_dir", idx, got.refract_dir, [0.0, 0.0, 0.0]),
    }

    let want_tir = is_total_internal_reflection(query.incident, query.normal, query.eta);
    assert_eq!(
        got.is_total_internal_reflection, want_tir,
        "query {idx} is_total_internal_reflection: gpu {} vs cpu {}",
        got.is_total_internal_reflection, want_tir
    );

    let want_fresnel = fresnel_schlick_reflectance(query.cos_theta, query.r0);
    assert!(
        close(got.fresnel_reflectance, want_fresnel),
        "query {idx} fresnel_reflectance: gpu {} vs cpu {}",
        got.fresnel_reflectance,
        want_fresnel
    );

    let want_r0 = fresnel_schlick_r0(query.n1, query.n2);
    assert!(
        close(got.fresnel_r0, want_r0),
        "query {idx} fresnel_r0: gpu {} vs cpu {}",
        got.fresnel_r0,
        want_r0
    );

    let want_restitution =
        reflect_with_restitution(query.velocity, query.normal, query.restitution);
    close_vec("restitution_dir", idx, got.restitution_dir, want_restitution);
}

/// Dispatches `queries` on the `GPU` and pins every result against the
/// reference.
fn check(ctx: &GpuContext, gpu: &GpuReflectRefractVec, queries: &[ReflectRefractQuery]) {
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
    let gpu = GpuReflectRefractVec::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn normal_incidence_identity_refraction_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuReflectRefractVec::new(&ctx);
    // Head-on incidence with eta = 1: the transmitted ray equals the incident
    // ray (no bend, no TIR) and the mirror reflection flips it back.
    let query = ReflectRefractQuery::new(
        [0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -2.0, 0.0],
        1.0,
        1.0,
        1.0,
        0.04,
        1.0,
        1.5,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn below_critical_oblique_refraction_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuReflectRefractVec::new(&ctx);
    // An oblique ray with eta = 0.9 (into a denser medium) bends but never
    // reflects totally, so refract returns Some and the TIR flag is false.
    let incident = unit([1.0, -1.0, 0.0]);
    let query = ReflectRefractQuery::new(
        incident,
        [0.0, 1.0, 0.0],
        [2.0, -3.0, 1.0],
        0.9,
        0.5,
        0.5,
        0.08,
        1.0,
        1.33,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn total_internal_reflection_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuReflectRefractVec::new(&ctx);
    // Dense -> rare (eta = 1.5) at 60° incidence is well past the critical
    // angle: k is clearly negative, so refract returns None (zero direction,
    // refract_valid false) and the TIR flag is true.
    let incident = unit([0.866_025, -0.5, 0.0]);
    let query = ReflectRefractQuery::new(
        incident,
        [0.0, 1.0, 0.0],
        [1.0, -1.0, 0.5],
        1.5,
        0.25,
        0.3,
        0.2,
        1.5,
        1.0,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn restitution_and_fresnel_fixtures_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuReflectRefractVec::new(&ctx);
    // A spread of restitution coefficients and Fresnel configurations, each with
    // a benign below-critical refraction so every function is exercised at once.
    let incident = unit([0.3, -0.9, 0.2]);
    let queries = [
        // Perfectly elastic bounce; grazing cosine near 1 keeps Fresnel near r0.
        ReflectRefractQuery::new(
            incident,
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            1.0,
            1.0,
            0.95,
            0.04,
            1.0,
            1.5,
        ),
        // Zero restitution (pure slide); mid cosine lifts Fresnel above r0.
        ReflectRefractQuery::new(
            incident,
            [0.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
            0.9,
            0.0,
            0.5,
            0.1,
            1.0,
            2.4,
        ),
        // Half restitution; near-zero cosine drives Fresnel toward one.
        ReflectRefractQuery::new(
            incident,
            [0.0, 1.0, 0.0],
            [2.0, -3.0, 0.0],
            0.8,
            0.5,
            0.05,
            0.02,
            1.33,
            1.0,
        ),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn mixed_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuReflectRefractVec::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    // One batch mixing the deterministic fixtures with many clearly-conditioned
    // random queries, dispatched together so the per-thread indexing and the
    // contiguous storage layout are both exercised, then pinned element-wise.
    let mut queries = vec![
        ReflectRefractQuery::new(
            [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -2.0, 0.0],
            1.0,
            1.0,
            1.0,
            0.04,
            1.0,
            1.5,
        ),
        ReflectRefractQuery::new(
            unit([0.866_025, -0.5, 0.0]),
            [0.0, 1.0, 0.0],
            [1.0, -1.0, 0.5],
            1.5,
            0.25,
            0.3,
            0.2,
            1.5,
            1.0,
        ),
    ];
    for _ in 0..48 {
        queries.push(well_conditioned(&mut state));
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn many_random_queries_sweep_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuReflectRefractVec::new(&ctx);
    let mut state = 0x0bad_c0de_cafe_f00d_u64;
    // A larger sweep of clearly-conditioned queries (several workgroups' worth)
    // pins every reported field across many random reflect/refract geometries.
    let queries: Vec<ReflectRefractQuery> =
        (0..200).map(|_| well_conditioned(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}
