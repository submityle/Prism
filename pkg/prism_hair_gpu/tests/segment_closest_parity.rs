//! Real-device parity for the per-pair segment-segment closest-point twin:
//! [`GpuSegmentClosest`] must reproduce the `CPU` golden
//! [`segment_segment_closest`](prism_render_architecture::hair::barrier_contact::segment_segment_closest)
//! for a batch of segment pairs, emitting one
//! `(closest_on_first, closest_on_second, distance)` triple per pair in input
//! order.
//!
//! # Parity criterion
//!
//! The closest-pair solve is a single closed-form evaluation (a handful of
//! dot/clamp/divide/multiply-add plus one `sqrt` for the distance) with no
//! chained recurrence, so the only `CPU` vs `GPU` divergence is legal
//! fused-multiply-add contraction. Each component and the distance is asserted
//! to within `abs_diff < 1e-4` or `rel_diff < 1e-3`. Every pair here is chosen
//! so the branch taken (interior solve, parallel guard, clamped endpoint,
//! degenerate point) is far from its decision boundary, so a stray `fma`
//! cannot flip the branch and the two sides evaluate the identical formula.
//!
//! The suite drives a skew (non-parallel) interior crossing, a pair of parallel
//! segments (the `denom`-zero guard), a point-vs-segment pair (first segment
//! degenerate), a segment-vs-point pair (second segment degenerate), a
//! point-vs-point pair (both degenerate, distance is the plain point distance),
//! an empty batch (handled with no dispatch), and a 100-pair batch that crosses
//! the 64-wide dispatch boundary.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-`WGSL`,
//! so it needs no optional device feature. All endpoints are integer-coordinate
//! points — never `f32::sin`/`cos` — so test data stays deterministic without
//! introducing transcendental divergence.
//!
//! Provenance: standard Ericson segment-segment closest-point construction plus
//! a `wgpu` compute dispatch; no Unreal Engine source or derived code.

use prism_hair_gpu::segment_closest::{reference_segment_closest, GpuSegmentClosest, SegmentPair};
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

/// Asserts every pair's closest-point triple matches the `CPU` golden
/// component for component.
fn assert_batch_matches(gpu: &[(Vec3, Vec3, f32)], pairs: &[SegmentPair]) {
    assert_eq!(gpu.len(), pairs.len(), "one output triple per input pair");
    for (i, pair) in pairs.iter().enumerate() {
        let (wc1, wc2, wdist) = reference_segment_closest(pair.p1, pair.q1, pair.p2, pair.q2);
        let (gc1, gc2, gdist) = gpu[i];
        assert!(
            close(gc1.x, wc1.x) && close(gc1.y, wc1.y) && close(gc1.z, wc1.z),
            "pair {i} c1: gpu ({}, {}, {}) vs cpu ({}, {}, {})",
            gc1.x,
            gc1.y,
            gc1.z,
            wc1.x,
            wc1.y,
            wc1.z,
        );
        assert!(
            close(gc2.x, wc2.x) && close(gc2.y, wc2.y) && close(gc2.z, wc2.z),
            "pair {i} c2: gpu ({}, {}, {}) vs cpu ({}, {}, {})",
            gc2.x,
            gc2.y,
            gc2.z,
            wc2.x,
            wc2.y,
            wc2.z,
        );
        assert!(
            close(gdist, wdist),
            "pair {i} dist: gpu {gdist} vs cpu {wdist}",
        );
    }
}

/// A skew (non-parallel) crossing whose closest points land at the interior
/// `s = t = 0.5`, offset by `shift` along `x` so a batch stays deterministic.
fn skew_pair(shift: f32) -> SegmentPair {
    SegmentPair {
        p1: Vec3::new(shift, 0.0, 0.0),
        q1: Vec3::new(shift + 4.0, 0.0, 0.0),
        p2: Vec3::new(shift + 2.0, -2.0, 2.0),
        q2: Vec3::new(shift + 2.0, 2.0, 2.0),
    }
}

#[test]
fn gpu_skew_interior_matches_cpu() {
    let Some(ctx) = context_or_skip("gpu_skew_interior_matches_cpu") else {
        return;
    };
    let twin = GpuSegmentClosest::new(&ctx);
    let pairs = [skew_pair(0.0)];
    let gpu = twin.eval(&ctx, &pairs);
    assert_batch_matches(&gpu, &pairs);
    // The skew crossing is a true interior solve: the distance between the two
    // parallel planes is 2, so a kernel that collapsed to an endpoint could not
    // pass.
    assert!(
        close(gpu[0].2, 2.0),
        "skew interior distance must be 2, got {}",
        gpu[0].2,
    );
}

#[test]
fn gpu_parallel_matches_cpu() {
    let Some(ctx) = context_or_skip("gpu_parallel_matches_cpu") else {
        return;
    };
    let twin = GpuSegmentClosest::new(&ctx);
    // Two parallel segments along x, offset by 3 in y and staggered in x so the
    // parallel `denom`-zero guard fires and `t0` lands clearly below 0.
    let pairs = [SegmentPair {
        p1: Vec3::new(0.0, 0.0, 0.0),
        q1: Vec3::new(4.0, 0.0, 0.0),
        p2: Vec3::new(1.0, 3.0, 0.0),
        q2: Vec3::new(5.0, 3.0, 0.0),
    }];
    let gpu = twin.eval(&ctx, &pairs);
    assert_batch_matches(&gpu, &pairs);
    assert!(
        close(gpu[0].2, 3.0),
        "parallel offset distance must be 3, got {}",
        gpu[0].2,
    );
}

#[test]
fn gpu_first_segment_degenerate_matches_cpu() {
    let Some(ctx) = context_or_skip("gpu_first_segment_degenerate_matches_cpu") else {
        return;
    };
    let twin = GpuSegmentClosest::new(&ctx);
    // Segment 1 collapses to the point (1, 1, 1); segment 2 runs along y.
    let pairs = [SegmentPair {
        p1: Vec3::new(1.0, 1.0, 1.0),
        q1: Vec3::new(1.0, 1.0, 1.0),
        p2: Vec3::new(0.0, 0.0, 0.0),
        q2: Vec3::new(0.0, 4.0, 0.0),
    }];
    let gpu = twin.eval(&ctx, &pairs);
    assert_batch_matches(&gpu, &pairs);
}

#[test]
fn gpu_second_segment_degenerate_matches_cpu() {
    let Some(ctx) = context_or_skip("gpu_second_segment_degenerate_matches_cpu") else {
        return;
    };
    let twin = GpuSegmentClosest::new(&ctx);
    // Segment 2 collapses to the point (1, 2, 0); segment 1 runs along x.
    let pairs = [SegmentPair {
        p1: Vec3::new(0.0, 0.0, 0.0),
        q1: Vec3::new(4.0, 0.0, 0.0),
        p2: Vec3::new(1.0, 2.0, 0.0),
        q2: Vec3::new(1.0, 2.0, 0.0),
    }];
    let gpu = twin.eval(&ctx, &pairs);
    assert_batch_matches(&gpu, &pairs);
    assert!(
        close(gpu[0].2, 2.0),
        "point-to-segment distance must be 2, got {}",
        gpu[0].2,
    );
}

#[test]
fn gpu_both_segments_degenerate_matches_cpu() {
    let Some(ctx) = context_or_skip("gpu_both_segments_degenerate_matches_cpu") else {
        return;
    };
    let twin = GpuSegmentClosest::new(&ctx);
    // Both segments collapse to points: the distance is the plain 3-4-5 point
    // distance.
    let pairs = [SegmentPair {
        p1: Vec3::new(1.0, 2.0, 3.0),
        q1: Vec3::new(1.0, 2.0, 3.0),
        p2: Vec3::new(4.0, 6.0, 3.0),
        q2: Vec3::new(4.0, 6.0, 3.0),
    }];
    let gpu = twin.eval(&ctx, &pairs);
    assert_batch_matches(&gpu, &pairs);
    assert!(
        close(gpu[0].2, 5.0),
        "point-to-point distance must be 5, got {}",
        gpu[0].2,
    );
}

#[test]
fn gpu_empty_batch_is_empty() {
    let Some(ctx) = context_or_skip("gpu_empty_batch_is_empty") else {
        return;
    };
    let twin = GpuSegmentClosest::new(&ctx);
    let gpu = twin.eval(&ctx, &[]);
    assert!(gpu.is_empty(), "an empty batch must return no triples");
}

#[test]
fn gpu_many_pairs_cross_workgroup_boundary() {
    let Some(ctx) = context_or_skip("gpu_many_pairs_cross_workgroup_boundary") else {
        return;
    };
    let twin = GpuSegmentClosest::new(&ctx);
    // 100 skew crossings cross the 64-wide dispatch boundary; each is an
    // interior solve translated along x, so threads in distinct workgroups must
    // each recover their own distance of 2.
    let pairs: Vec<SegmentPair> = (0..100).map(|i| skew_pair(i as f32)).collect();
    let gpu = twin.eval(&ctx, &pairs);
    assert_batch_matches(&gpu, &pairs);
    for (i, triple) in gpu.iter().enumerate() {
        assert!(
            close(triple.2, 2.0),
            "pair {i} skew distance must be 2, got {}",
            triple.2,
        );
    }
}
