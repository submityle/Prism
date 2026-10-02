//! Real-device parity for the per-query contact-distance twin:
//! [`GpuHairContactDistance`] must reproduce the `CPU` goldens
//! [`point_plane_signed_distance`](prism_render_architecture::hair::barrier_contact::point_plane_signed_distance)
//! and
//! [`point_point_distance`](prism_render_architecture::hair::barrier_contact::point_point_distance)
//! for a batch of `(p, plane_point, plane_normal)` triples, emitting one
//! `(signed_plane_distance, point_point_distance)` pair per query in input
//! order.
//!
//! # Parity criterion
//!
//! Both primitives are a single closed-form evaluation (a `dot` and a reciprocal
//! `sqrt` for the plane distance, one subtract-and-length for the point
//! distance) with no chained recurrence, so the only `CPU` vs `GPU` divergence
//! is legal fused-multiply-add contraction and correctly rounded `sqrt`/division.
//! Each output lane is asserted to within `abs_diff < 1e-4` or
//! `rel_diff < 1e-3`. The zero-normal signed-distance branch is asserted
//! bit-exact `0`.
//!
//! The suite drives points straddling a plane (both signed-distance signs
//! present, so neither a zero-writing nor a constant-writing no-op kernel could
//! pass), a non-unit normal (exercising the normalization), a zero normal
//! (bit-exact zero signed distance while the point distance stays non-zero), an
//! empty batch (handled with no dispatch), and a 100-query batch that crosses
//! the 64-wide dispatch boundary.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. All inputs are explicit decimal
//! literals so test data stays deterministic without transcendental divergence.
//!
//! Provenance: standard analytic point/plane distance plus a `wgpu` compute
//! dispatch; no Unreal Engine source or derived code.

use prism_hair_gpu::contact_distance::{
    reference_point_plane_signed_distance, reference_point_point_distance, GpuHairContactDistance,
};
use prism_hair_gpu::GpuContext;
use prism_render_architecture::hair::barrier_contact::Vec3;

/// Acquires a headless context, or `None` (with a skip notice) when the host has
/// no `wgpu` adapter so the suite stays green off-device.
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn context_or_skip(label: &str) -> Option<GpuContext> {
    match GpuContext::try_headless() {
        Some(ctx) => Some(ctx),
        None => {
            eprintln!("skipping {label}: no wgpu adapter on this host");
            None
        }
    }
}

/// True when `got` matches `want` within the fused-multiply-add tolerance.
fn close(got: f32, want: f32) -> bool {
    let diff = (got - want).abs();
    diff < 1.0e-4 || diff <= 1.0e-3 * want.abs()
}

/// Asserts every query's output pair matches the `CPU` goldens.
fn assert_batch_matches(gpu: &[(f32, f32)], queries: &[(Vec3, Vec3, Vec3)]) {
    assert_eq!(gpu.len(), queries.len(), "one output pair per query");
    for (i, &(p, pp, nn)) in queries.iter().enumerate() {
        let want_signed = reference_point_plane_signed_distance(p, pp, nn);
        let want_point = reference_point_point_distance(p, pp);
        let (got_signed, got_point) = gpu[i];
        assert!(
            close(got_signed, want_signed),
            "query {i} signed distance: gpu {got_signed} vs cpu {want_signed}",
        );
        assert!(
            close(got_point, want_point),
            "query {i} point distance: gpu {got_point} vs cpu {want_point}",
        );
    }
}

#[test]
fn gpu_plane_distances_match_cpu() {
    let Some(ctx) = context_or_skip("gpu_plane_distances_match_cpu") else {
        return;
    };
    let twin = GpuHairContactDistance::new(&ctx);
    // A plane through (0, 1, 0) with the unit +Y normal: points above read a
    // positive signed distance, points below a negative one.
    let plane_point = Vec3::new(0.0, 1.0, 0.0);
    let normal = Vec3::new(0.0, 1.0, 0.0);
    let queries = [
        (Vec3::new(0.3, 2.4, -0.5), plane_point, normal),
        (Vec3::new(-1.2, 0.2, 0.7), plane_point, normal),
        (Vec3::new(2.0, 1.0, 2.0), plane_point, normal),
        (Vec3::new(0.5, 3.7, 1.1), plane_point, normal),
    ];
    let gpu = twin.eval(&ctx, &queries);
    assert_batch_matches(&gpu, &queries);
    // Both signs must appear (query 0 above, query 1 below) and point distances
    // must be strictly positive, so no constant/zero kernel could pass.
    assert!(
        gpu[0].0 > 0.0,
        "point above plane must have positive distance"
    );
    assert!(
        gpu[1].0 < 0.0,
        "point below plane must have negative distance"
    );
    for (i, &(_, point_dist)) in gpu.iter().enumerate() {
        assert!(
            point_dist > 0.0,
            "point distance {i} must be strictly positive"
        );
    }
}

#[test]
fn gpu_non_unit_normal_is_normalized() {
    let Some(ctx) = context_or_skip("gpu_non_unit_normal_is_normalized") else {
        return;
    };
    let twin = GpuHairContactDistance::new(&ctx);
    // A non-unit, off-axis normal: the kernel must normalize it before the
    // projection, matching the golden's `normalize_or_zero`.
    let plane_point = Vec3::new(1.0, 0.0, -2.0);
    let queries = [
        (
            Vec3::new(2.0, 0.5, -1.0),
            plane_point,
            Vec3::new(0.0, 3.0, 0.0),
        ),
        (
            Vec3::new(-0.5, 1.5, 0.5),
            plane_point,
            Vec3::new(2.0, 2.0, 2.0),
        ),
        (
            Vec3::new(3.3, -1.1, 0.9),
            plane_point,
            Vec3::new(5.0, -1.0, 2.0),
        ),
    ];
    let gpu = twin.eval(&ctx, &queries);
    assert_batch_matches(&gpu, &queries);
    // The off-axis projections must be non-trivial (not all zero), so the
    // normalize-then-dot path is genuinely exercised.
    let any_nonzero = gpu.iter().any(|&(d, _)| d.abs() > 1.0e-3);
    assert!(any_nonzero, "normalized projections must be non-zero");
}

#[test]
fn gpu_zero_normal_is_bit_exact_zero() {
    let Some(ctx) = context_or_skip("gpu_zero_normal_is_bit_exact_zero") else {
        return;
    };
    let twin = GpuHairContactDistance::new(&ctx);
    // A (numerically) zero normal has no orientation: the signed distance is
    // exactly zero on both sides, but the point distance is still the plain
    // separation and must stay non-zero.
    let queries = [(
        Vec3::new(1.5, -2.0, 0.5),
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::ZERO,
    )];
    let gpu = twin.eval(&ctx, &queries);
    assert_batch_matches(&gpu, &queries);
    assert_eq!(
        gpu[0].0.to_bits(),
        0.0_f32.to_bits(),
        "zero-normal signed distance must be bit-exact zero, got {}",
        gpu[0].0,
    );
    assert!(
        gpu[0].1 > 0.0,
        "point distance for distinct points must stay non-zero, got {}",
        gpu[0].1,
    );
}

#[test]
fn gpu_empty_batch_is_empty() {
    let Some(ctx) = context_or_skip("gpu_empty_batch_is_empty") else {
        return;
    };
    let twin = GpuHairContactDistance::new(&ctx);
    let gpu = twin.eval(&ctx, &[]);
    assert!(gpu.is_empty(), "an empty batch must return no pairs");
}

#[test]
fn gpu_many_queries_cross_workgroup_boundary() {
    let Some(ctx) = context_or_skip("gpu_many_queries_cross_workgroup_boundary") else {
        return;
    };
    let twin = GpuHairContactDistance::new(&ctx);
    // 100 queries sweep the query point along y across a +Y plane, crossing the
    // 64-wide dispatch boundary so threads in distinct workgroups each recover
    // their own value.
    let plane_point = Vec3::new(0.0, 1.0, 0.0);
    let normal = Vec3::new(0.0, 1.0, 0.0);
    let queries: Vec<(Vec3, Vec3, Vec3)> = (0..100)
        .map(|i| {
            (
                Vec3::new(0.25, 0.13 + (i as f32) * 0.07, -0.4),
                plane_point,
                normal,
            )
        })
        .collect();
    let gpu = twin.eval(&ctx, &queries);
    assert_batch_matches(&gpu, &queries);
}
