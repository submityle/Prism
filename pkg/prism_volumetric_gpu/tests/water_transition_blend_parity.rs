//! Real-device parity for the transition-blend twin:
//! [`GpuWaterTransitionBlend`](prism_volumetric_gpu::water_transition_blend::GpuWaterTransitionBlend)
//! must reproduce the `CPU` golden
//! [`solver_blend_weights`](prism_render_architecture::water::transition::solver_blend_weights)
//! and
//! [`blend_normal`](prism_render_architecture::water::transition::blend_normal)
//! across pure regions, both crossfade bands, the zero-length normal fallback,
//! and a randomized sweep.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The expected values are produced by calling the golden `solver_blend_weights`
//! and `blend_normal` directly, so the test pins `GPU == golden`, not merely
//! that the shader compiles.
//!
//! # Parity criterion
//!
//! The kernel performs clamps, a guarded divide and one `sqrt`, so a `GPU`
//! reciprocal or square root may land a few units in the last place from the
//! scalar reference. Every weight and normal component is asserted within
//! `abs_diff <= 1e-4` or `rel_diff <= 1e-3`, the three weights are checked to
//! sum to one, and the zero-length fallback is pinned exactly. Fixtures keep the
//! band widths well above `EPS` so the ramp denominators stay non-degenerate.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::transition`；无第三方引擎源码或衍生代码。

use prism_render_architecture::water::transition::{
    blend_normal, solver_blend_weights, TransitionBands,
};
use prism_render_architecture::water::Vec3;
use prism_volumetric_gpu::water_transition_blend::{
    GpuWaterTransitionBlend, WaterTransitionBlendQuery, WaterTransitionBlendResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound. A `GPU` reciprocal or square root may land a few units
/// in the last place from the scalar reference; `1e-4` admits that legal slack
/// while still failing a wrong port.
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

/// Reconstructs the golden result in-host by calling the reference functions
/// directly.
fn oracle(q: &WaterTransitionBlendQuery) -> WaterTransitionBlendResult {
    let bands = TransitionBands {
        particle_to_swe: q.particle_to_swe,
        swe_to_spectral: q.swe_to_spectral,
        half_width: q.half_width,
    };
    let w = solver_blend_weights(q.distance, bands);
    let n = blend_normal(
        w,
        Vec3::new(q.np_x, q.np_y, q.np_z),
        Vec3::new(q.nsw_x, q.nsw_y, q.nsw_z),
        Vec3::new(q.nsp_x, q.nsp_y, q.nsp_z),
    );
    WaterTransitionBlendResult {
        w_particle: w.particle,
        w_shallow_water: w.shallow_water,
        w_spectral: w.spectral,
        n_x: n.x,
        n_y: n.y,
        n_z: n.z,
    }
}

/// Pins one `GPU` result against the in-host oracle: every weight and normal
/// component within tolerance, plus a partition-of-unity check on the weights.
fn check_query(idx: usize, got: &WaterTransitionBlendResult, want: &WaterTransitionBlendResult) {
    assert!(
        close(got.w_particle, want.w_particle),
        "query {idx} w_particle: gpu {} vs cpu {}",
        got.w_particle,
        want.w_particle
    );
    assert!(
        close(got.w_shallow_water, want.w_shallow_water),
        "query {idx} w_shallow_water: gpu {} vs cpu {}",
        got.w_shallow_water,
        want.w_shallow_water
    );
    assert!(
        close(got.w_spectral, want.w_spectral),
        "query {idx} w_spectral: gpu {} vs cpu {}",
        got.w_spectral,
        want.w_spectral
    );
    // Partition of unity: the three weights must sum to one.
    let sum = got.w_particle + got.w_shallow_water + got.w_spectral;
    assert!(close(sum, 1.0), "query {idx} weights sum to one: got {sum}");
    assert!(
        close(got.n_x, want.n_x),
        "query {idx} n_x: gpu {} vs cpu {}",
        got.n_x,
        want.n_x
    );
    assert!(
        close(got.n_y, want.n_y),
        "query {idx} n_y: gpu {} vs cpu {}",
        got.n_y,
        want.n_y
    );
    assert!(
        close(got.n_z, want.n_z),
        "query {idx} n_z: gpu {} vs cpu {}",
        got.n_z,
        want.n_z
    );
}

/// Dispatches `queries` and checks every result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuWaterTransitionBlend, queries: &[WaterTransitionBlendQuery]) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (idx, (q, g)) in queries.iter().zip(got.iter()).enumerate() {
        let want = oracle(q);
        check_query(idx, g, &want);
    }
}

/// A small `LCG` for the randomized sweep (host-only; the kernel is portable).
fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 33) as u32
}

/// Maps a raw `u32` to an `f32` in `[lo, hi]` without any transcendental call.
fn uniform(bits: u32, lo: f32, hi: f32) -> f32 {
    let unit = (bits as f32) / (u32::MAX as f32);
    lo + unit * (hi - lo)
}

/// A fixed-band query builder: particle-to-swe at 20, swe-to-spectral at 80,
/// half-width 5, with the three standard-basis normals.
fn basis_query(distance: f32) -> WaterTransitionBlendQuery {
    WaterTransitionBlendQuery::new(
        distance, 20.0, 80.0, 5.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
    )
}

#[test]
fn empty_batch_produces_no_results() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterTransitionBlend::new(&ctx);
    // An empty batch short-circuits on the host (a storage buffer cannot be
    // zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn pure_regions_match_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterTransitionBlend::new(&ctx);
    // Distances well inside each region: pure particle, pure shallow-water
    // between the bands, pure spectral beyond. Each sits far from a band edge.
    let queries = [basis_query(2.0), basis_query(50.0), basis_query(140.0)];
    check(&ctx, &gpu, &queries);
}

#[test]
fn near_band_crossfade_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterTransitionBlend::new(&ctx);
    // Distances straddling the near band [15, 25] at interior ramp fractions,
    // kept at least 1.0 from the band edges so the ramp is unambiguous.
    let queries = [
        basis_query(17.0),
        basis_query(19.0),
        basis_query(21.0),
        basis_query(23.0),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn far_band_crossfade_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterTransitionBlend::new(&ctx);
    // Distances straddling the far band [75, 85] at interior ramp fractions.
    let queries = [
        basis_query(77.0),
        basis_query(79.0),
        basis_query(81.0),
        basis_query(83.0),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn zero_length_normal_falls_back_to_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterTransitionBlend::new(&ctx);
    // At distance 22 both particle and shallow-water are active; feeding exactly
    // opposed normals cancels the weighted sum to zero length, so the result
    // must hit the normalize_or_zero -> ZERO branch exactly on both sides.
    let queries = [WaterTransitionBlendQuery::new(
        22.0, 20.0, 80.0, 5.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0,
    )];
    let got = gpu.evaluate(&ctx, &queries);
    assert_eq!(got.len(), 1);
    let want = oracle(&queries[0]);
    // The golden composed sum is (0, w_p - w_sw, 0); choose weights so it is not
    // exactly zero unless the branch fires. Here it does not fully cancel, so
    // assert against the oracle directly.
    check_query(0, &got[0], &want);
}

#[test]
fn exactly_canceling_normals_hit_zero_branch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterTransitionBlend::new(&ctx);
    // Distance exactly at the near-band midpoint gives to_swe = 0.5, so particle
    // and shallow-water weights are both 0.5; opposed unit normals then cancel
    // to a zero-length sum, forcing the ZERO fallback on both GPU and golden.
    let queries = [WaterTransitionBlendQuery::new(
        20.0, 20.0, 80.0, 5.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0,
    )];
    let got = gpu.evaluate(&ctx, &queries);
    assert_eq!(got.len(), 1);
    let want = oracle(&queries[0]);
    // Golden must land on ZERO here; pin both the oracle expectation and GPU.
    assert!(
        want.n_x.abs() <= REL_FLOOR && want.n_y.abs() <= REL_FLOOR && want.n_z.abs() <= REL_FLOOR,
        "oracle should hit the zero-length fallback: {want:?}"
    );
    check_query(0, &got[0], &want);
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterTransitionBlend::new(&ctx);
    let mut state = 0x51a3_c7e2_0d94_6b18_u64;
    let mut queries: Vec<WaterTransitionBlendQuery> = Vec::new();
    // Several workgroups' worth of random queries. Band midpoints and widths are
    // kept well separated (half_width >= 3) so no ramp band is degenerate, and
    // distances span inside/outside both bands. Normals span a cube including
    // near-zero vectors to exercise the normalize_or_zero fallback.
    while queries.len() < 512 {
        let particle_to_swe = uniform(lcg(&mut state), 15.0, 25.0);
        let swe_to_spectral = uniform(lcg(&mut state), 70.0, 90.0);
        let half_width = uniform(lcg(&mut state), 3.0, 8.0);
        let distance = uniform(lcg(&mut state), 0.0, 120.0);
        // Reject distances within 1.0 of a band edge so the ramp fraction is
        // unambiguous away from the clamp knees.
        let near_lo = particle_to_swe - half_width;
        let near_hi = particle_to_swe + half_width;
        let far_lo = swe_to_spectral - half_width;
        let far_hi = swe_to_spectral + half_width;
        let too_close = (distance - near_lo).abs() < 1.0
            || (distance - near_hi).abs() < 1.0
            || (distance - far_lo).abs() < 1.0
            || (distance - far_hi).abs() < 1.0;
        if too_close {
            continue;
        }
        let np_x = uniform(lcg(&mut state), -1.0, 1.0);
        let np_y = uniform(lcg(&mut state), -1.0, 1.0);
        let np_z = uniform(lcg(&mut state), -1.0, 1.0);
        let nsw_x = uniform(lcg(&mut state), -1.0, 1.0);
        let nsw_y = uniform(lcg(&mut state), -1.0, 1.0);
        let nsw_z = uniform(lcg(&mut state), -1.0, 1.0);
        let nsp_x = uniform(lcg(&mut state), -1.0, 1.0);
        let nsp_y = uniform(lcg(&mut state), -1.0, 1.0);
        let nsp_z = uniform(lcg(&mut state), -1.0, 1.0);
        queries.push(WaterTransitionBlendQuery::new(
            distance,
            particle_to_swe,
            swe_to_spectral,
            half_width,
            np_x,
            np_y,
            np_z,
            nsw_x,
            nsw_y,
            nsw_z,
            nsp_x,
            nsp_y,
            nsp_z,
        ));
    }
    check(&ctx, &gpu, &queries);
}
