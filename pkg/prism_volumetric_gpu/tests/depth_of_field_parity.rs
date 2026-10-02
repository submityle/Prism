//! Real-device parity for the depth-of-field twin:
//! [`GpuDepthOfField`](prism_volumetric_gpu::depth_of_field::GpuDepthOfField)
//! must reproduce the `CPU` golden
//! [`depth_of_field`](prism_render_architecture::particle::depth_of_field)
//! across a near-defocus depth (negative signed `CoC`, interior gather radius),
//! a far-defocus depth (positive signed `CoC`), a saturated far depth whose
//! radius clamps to the maximum, a wide-aperture near depth that also
//! saturates, a near-focus depth with a tiny `CoC`, and a randomized batch of
//! mixed apertures compared element-for-element.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each depth is a fixed, non-reorderable sequence of multiplies, adds and
//! divides, so `CPU` and `GPU` evaluate the same closed form in the same order.
//! They are not bit-exact: a `GPU` may fuse a multiply-add the scalar reference
//! leaves separate, perturbing the low mantissa bits by a few units in the last
//! place. The comparison therefore allows `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3` on every `f32` field, and bit equality on the integer
//! near/far classification.
//!
//! # Conditioning
//!
//! Every fixture is kept well away from the discrete guard cracks: the depth is
//! comfortably positive, the focus distance comfortably exceeds the focal
//! length, the denominator is far from zero, the maximum radius is far above the
//! `smoothstep` span floor, and the depth is a safe margin from the focus plane
//! so the near/far classification stays on the same side regardless of a few
//! units in the last place of slack. The random batch enforces the same margins
//! by rejection sampling.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::depth_of_field`；
//! no third-party engine source or derived code.

use prism_render_architecture::particle::depth_of_field::DofParams;
use prism_volumetric_gpu::depth_of_field::{
    golden, DepthOfFieldQuery, DepthOfFieldSample, GpuDepthOfField,
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

/// Builds a clearly-conditioned depth-of-field query by rejection sampling: the
/// lens keeps the focus distance comfortably above the focal length and the
/// maximum radius well above the span floor, and the depth is kept positive and
/// a safe margin away from the focus plane, so both the division guards and the
/// near/far classification stay off every tie.
fn rand_query(state: &mut u64) -> DepthOfFieldQuery {
    loop {
        let focus = range(state, 4.0, 12.0);
        let focal_length = range(state, 0.5, 2.0);
        let aperture = range(state, 0.5, 3.0);
        let max_coc = range(state, 0.05, 0.45);
        let depth = range(state, 1.0, 20.0);
        if focus - focal_length < 1.0 {
            continue;
        }
        if (depth - focus).abs() < 0.5 {
            continue;
        }
        let params = DofParams::new(focus, focal_length, aperture, max_coc);
        return DepthOfFieldQuery::new(params, depth);
    }
}

/// Pins one `GPU` result against the `CPU` golden for `query`: every continuous
/// field must agree within bound and the near/far classification must match
/// exactly.
fn pin(idx: usize, query: &DepthOfFieldQuery, got: &DepthOfFieldSample) {
    let want = golden(query);
    assert!(
        close(got.signed_coc, want.signed_coc),
        "query {idx} signed_coc: gpu {} vs cpu {}",
        got.signed_coc,
        want.signed_coc
    );
    assert!(
        close(got.coc_diameter, want.coc_diameter),
        "query {idx} coc_diameter: gpu {} vs cpu {}",
        got.coc_diameter,
        want.coc_diameter
    );
    assert!(
        close(got.radius, want.radius),
        "query {idx} radius: gpu {} vs cpu {}",
        got.radius,
        want.radius
    );
    assert!(
        close(got.bokeh_scale, want.bokeh_scale),
        "query {idx} bokeh_scale: gpu {} vs cpu {}",
        got.bokeh_scale,
        want.bokeh_scale
    );
    assert!(
        close(got.blur_fade, want.blur_fade),
        "query {idx} blur_fade: gpu {} vs cpu {}",
        got.blur_fade,
        want.blur_fade
    );
    assert!(
        close(got.energy_attenuation, want.energy_attenuation),
        "query {idx} energy_attenuation: gpu {} vs cpu {}",
        got.energy_attenuation,
        want.energy_attenuation
    );
    assert_eq!(
        got.is_near, want.is_near,
        "query {idx} is_near: gpu {} vs cpu {}",
        got.is_near, want.is_near
    );
}

/// Dispatches `queries` on the `GPU` and pins every result against the
/// reference.
fn check(ctx: &GpuContext, gpu: &GpuDepthOfField, queries: &[DepthOfFieldQuery]) {
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
    let gpu = GpuDepthOfField::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn near_defocus_interior_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDepthOfField::new(&ctx);
    // depth 6 sits in front of a focus plane at 10 (near defocus), with a
    // generous max radius so the gather radius stays strictly interior and the
    // smoothstep is evaluated in its cubic region, not at an edge.
    let query = DepthOfFieldQuery::new(DofParams::new(10.0, 2.0, 1.0, 0.5), 6.0);
    check(&ctx, &gpu, &[query]);
}

#[test]
fn far_defocus_interior_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDepthOfField::new(&ctx);
    // depth 20 sits behind the focus plane at 10 (far defocus); the max radius
    // again keeps the gather radius interior.
    let query = DepthOfFieldQuery::new(DofParams::new(10.0, 2.0, 1.0, 0.5), 20.0);
    check(&ctx, &gpu, &[query]);
}

#[test]
fn far_defocus_saturated_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDepthOfField::new(&ctx);
    // A tight max radius with a deep depth: the raw thin-lens radius exceeds the
    // clamp, so both devices saturate to the maximum via the min() guard.
    let query = DepthOfFieldQuery::new(DofParams::new(10.0, 1.0, 2.0, 0.05), 40.0);
    check(&ctx, &gpu, &[query]);
}

#[test]
fn wide_aperture_near_saturated_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDepthOfField::new(&ctx);
    // A wide aperture and a shallow near depth produce a large CoC that clamps to
    // the maximum, exercising the near-defocus branch at saturation.
    let query = DepthOfFieldQuery::new(DofParams::new(8.0, 2.0, 3.0, 0.6), 3.0);
    check(&ctx, &gpu, &[query]);
}

#[test]
fn near_focus_small_coc_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDepthOfField::new(&ctx);
    // depth 9 is one unit in front of the focus plane at 10: a small negative
    // CoC with a margin that keeps the near/far classification unambiguous.
    let query = DepthOfFieldQuery::new(DofParams::new(10.0, 1.5, 1.2, 0.4), 9.0);
    check(&ctx, &gpu, &[query]);
}

#[test]
fn mixed_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDepthOfField::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    // One batch mixing deterministic fixtures with many random queries,
    // dispatched together so the per-thread indexing and the contiguous storage
    // layout are both exercised, then pinned element-for-element.
    let mut queries = vec![
        DepthOfFieldQuery::new(DofParams::new(10.0, 2.0, 1.0, 0.5), 6.0),
        DepthOfFieldQuery::new(DofParams::new(10.0, 2.0, 1.0, 0.5), 20.0),
        DepthOfFieldQuery::new(DofParams::new(8.0, 2.0, 3.0, 0.6), 3.0),
    ];
    for _ in 0..48 {
        queries.push(rand_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn many_queries_sweep_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDepthOfField::new(&ctx);
    let mut state = 0x0bad_c0de_cafe_f00d_u64;
    // A larger sweep (several workgroups' worth) pins every field across many
    // random lenses and depths of mixed aperture.
    let queries: Vec<DepthOfFieldQuery> = (0..200).map(|_| rand_query(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}
