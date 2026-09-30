//! Real-device parity for the isolated arc-length resampling twin:
//! [`GpuHairResample`] must reproduce the `CPU` golden
//! [`resample_strand`](prism_render_architecture::hair::groom_import::resample_strand)
//! /
//! [`resample_groom`](prism_render_architecture::hair::groom_import::resample_groom)
//! for a batch of raw guide polylines. The suite drives a straight single
//! segment, an unequal two-segment polyline, a many-segment curve, the pinned
//! endpoints (root and tip), the single-point and all-coincident zero-length
//! degeneracies, the out-of-bounds and zero-length range skips (twinning
//! `resample_groom`'s host filter), the empty no-op, and a large multi-workgroup
//! batch that crosses the 64-wide dispatch boundary.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The endpoints are pinned exactly (`t == 0` at the root, `t == 1.0` at the
//! tip); with integer-valued input vertices `a + (b - a)` is exact, so those
//! outputs are asserted *exactly*. Interior points divide `(d - d0) / span` and
//! take a `sqrt` per segment, either of which a `GPU` may fuse, so they can
//! differ by a few low-mantissa `ULP`; those are asserted with the point within
//! `1e-4` and each component within `abs_diff < 1e-4` or `rel_diff < 1e-3`. All
//! interior samples sit inside segment interiors so both sides pick the same
//! segment. No `sin`/`cos` appears anywhere; all vertices are explicit literals.
//!
//! Provenance: standard arc-length polyline resampling plus `wgpu` compute
//! dispatch; no Unreal Engine source or derived code.

use prism_hair_gpu::resample::{
    reference_resample_groom, reference_resample_strand, GpuHairResample,
};
use prism_hair_gpu::GpuContext;
use prism_render_architecture::hair::dynamics::Vec3;
use prism_render_architecture::hair::groom_import::{RawStrandRange, StrandAttributes};

fn v(x: f32, y: f32, z: f32) -> Vec3 {
    Vec3::new(x, y, z)
}

/// A `RawStrandRange` with placeholder attributes (attributes never affect the
/// resampling math).
fn range(start: usize, len: usize) -> RawStrandRange {
    RawStrandRange {
        start,
        len,
        attributes: StrandAttributes::new(0.02, 0.005, [0.0, 0.0], 7),
    }
}

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

/// Asserts one scalar component matches within the documented fma tolerance.
fn assert_close(got: f32, expected: f32, label: &str) {
    let abs_diff = (got - expected).abs();
    let rel_diff = abs_diff / expected.abs().max(1e-6);
    assert!(
        abs_diff < 1e-4 || rel_diff < 1e-3,
        "{label}: gpu {got}, cpu {expected} (abs {abs_diff}, rel {rel_diff})"
    );
}

/// Asserts one scalar component is bit-identical (zero difference), avoiding a
/// float `==` (the `float_cmp` lint) by bounding the absolute difference by
/// `0.0` — pinned endpoints must not drift at all.
fn assert_exact(got: f32, expected: f32, label: &str) {
    assert!(
        (got - expected).abs() <= 0.0,
        "{label}: {got} != {expected}"
    );
}

/// Compares every GPU point against the golden `resample_groom` positions,
/// asserting each strand's endpoints exactly and interior points within the fma
/// tolerance.
fn assert_matches_groom(
    got: &[Vec3],
    points: &[Vec3],
    strands: &[RawStrandRange],
    target_points: u32,
) {
    let groom = reference_resample_groom(points, strands, target_points);
    let pps = groom.points_per_strand;
    assert_eq!(
        got.len(),
        groom.positions.len(),
        "flat length: gpu {}, cpu {}",
        got.len(),
        groom.positions.len()
    );
    for (i, (g, c)) in got.iter().zip(groom.positions.iter()).enumerate() {
        let within_strand = i % pps;
        let is_endpoint = within_strand == 0 || within_strand == pps - 1;
        if is_endpoint {
            assert_exact(g.x, c.x, "endpoint.x");
            assert_exact(g.y, c.y, "endpoint.y");
            assert_exact(g.z, c.z, "endpoint.z");
        } else {
            assert_close(g.x, c.x, "interior.x");
            assert_close(g.y, c.y, "interior.y");
            assert_close(g.z, c.z, "interior.z");
        }
    }
}

#[test]
fn single_segment_midpoint() {
    let Some(ctx) = context_or_skip("single_segment_midpoint") else {
        return;
    };
    let kernel = GpuHairResample::new(&ctx);
    let points = [v(0.0, 0.0, 0.0), v(4.0, 0.0, 0.0)];
    let strands = [range(0, 2)];
    let got = kernel.eval(&ctx, &points, &strands, 3);
    assert_eq!(got.len(), 3);
    // Root, arc-length midpoint, tip.
    assert_exact(got[0].x, 0.0, "root.x");
    assert_close(got[1].x, 2.0, "mid.x");
    assert_exact(got[2].x, 4.0, "tip.x");
    assert_matches_groom(&got, &points, &strands, 3);
}

#[test]
fn unequal_two_segment_polyline() {
    let Some(ctx) = context_or_skip("unequal_two_segment_polyline") else {
        return;
    };
    let kernel = GpuHairResample::new(&ctx);
    // A short first leg and a long second leg so the resample lands the interior
    // samples inside different segments.
    let points = [v(0.0, 0.0, 0.0), v(1.0, 0.0, 0.0), v(1.0, 6.0, 0.0)];
    let strands = [range(0, 3)];
    let got = kernel.eval(&ctx, &points, &strands, 5);
    assert_eq!(got.len(), 5);
    assert_matches_groom(&got, &points, &strands, 5);
}

#[test]
fn many_segment_curve() {
    let Some(ctx) = context_or_skip("many_segment_curve") else {
        return;
    };
    let kernel = GpuHairResample::new(&ctx);
    // An eight-vertex zig-zag polyline (explicit literals, no trig); target 6.
    let points = [
        v(0.0, 0.0, 0.0),
        v(2.0, 1.0, 0.0),
        v(4.0, 0.0, 1.0),
        v(6.0, 2.0, 0.0),
        v(8.0, 1.0, 2.0),
        v(10.0, 3.0, 1.0),
        v(12.0, 0.0, 0.0),
        v(14.0, 2.0, 3.0),
    ];
    let strands = [range(0, 8)];
    let got = kernel.eval(&ctx, &points, &strands, 6);
    assert_eq!(got.len(), 6);
    assert_matches_groom(&got, &points, &strands, 6);
}

#[test]
fn endpoints_pinned_exact() {
    let Some(ctx) = context_or_skip("endpoints_pinned_exact") else {
        return;
    };
    let kernel = GpuHairResample::new(&ctx);
    let points = [
        v(-3.0, 5.0, 2.0),
        v(1.0, 5.0, 2.0),
        v(1.0, 9.0, 2.0),
        v(1.0, 9.0, 7.0),
    ];
    let strands = [range(0, 4)];
    let got = kernel.eval(&ctx, &points, &strands, 7);
    assert_eq!(got.len(), 7);
    // The first output is the true root, the last is the true tip; both exact.
    assert_exact(got[0].x, -3.0, "root.x");
    assert_exact(got[0].y, 5.0, "root.y");
    assert_exact(got[0].z, 2.0, "root.z");
    assert_exact(got[6].x, 1.0, "tip.x");
    assert_exact(got[6].y, 9.0, "tip.y");
    assert_exact(got[6].z, 7.0, "tip.z");
    assert_matches_groom(&got, &points, &strands, 7);
}

#[test]
fn single_raw_point() {
    let Some(ctx) = context_or_skip("single_raw_point") else {
        return;
    };
    let kernel = GpuHairResample::new(&ctx);
    let points = [v(3.0, -2.0, 5.0)];
    let strands = [range(0, 1)];
    let got = kernel.eval(&ctx, &points, &strands, 4);
    let expected = reference_resample_strand(&points, 4);
    assert_eq!(got.len(), expected.len());
    for (g, c) in got.iter().zip(expected.iter()) {
        assert_exact(g.x, c.x, "copy.x");
        assert_exact(g.y, c.y, "copy.y");
        assert_exact(g.z, c.z, "copy.z");
    }
}

#[test]
fn all_coincident_zero_length() {
    let Some(ctx) = context_or_skip("all_coincident_zero_length") else {
        return;
    };
    let kernel = GpuHairResample::new(&ctx);
    let points = [v(1.0, 2.0, 3.0), v(1.0, 2.0, 3.0), v(1.0, 2.0, 3.0)];
    let strands = [range(0, 3)];
    let got = kernel.eval(&ctx, &points, &strands, 5);
    assert_eq!(got.len(), 5);
    // Zero total length: every output collapses to the root, exactly.
    for g in &got {
        assert_exact(g.x, 1.0, "coincident.x");
        assert_exact(g.y, 2.0, "coincident.y");
        assert_exact(g.z, 3.0, "coincident.z");
    }
}

#[test]
fn skips_out_of_bounds_and_zero_len_ranges() {
    let Some(ctx) = context_or_skip("skips_out_of_bounds_and_zero_len_ranges") else {
        return;
    };
    let kernel = GpuHairResample::new(&ctx);
    let points = [
        v(0.0, 0.0, 0.0),
        v(2.0, 0.0, 0.0),
        v(2.0, 4.0, 0.0),
        v(0.0, 0.0, 0.0),
        v(0.0, 3.0, 0.0),
    ];
    // Valid, out-of-bounds (start+len past the buffer), zero-length, valid.
    let strands = [range(0, 3), range(3, 9), range(1, 0), range(3, 2)];
    let got = kernel.eval(&ctx, &points, &strands, 4);
    // Only the two valid strands survive → 2 * 4 points.
    assert_eq!(got.len(), 8);
    assert_matches_groom(&got, &points, &strands, 4);
}

#[test]
fn empty_input_is_noop() {
    let Some(ctx) = context_or_skip("empty_input_is_noop") else {
        return;
    };
    let kernel = GpuHairResample::new(&ctx);
    let got = kernel.eval(&ctx, &[], &[], 5);
    assert!(got.is_empty());
    // A non-empty buffer with no surviving strand is also a no-op.
    let points = [v(0.0, 0.0, 0.0), v(1.0, 0.0, 0.0)];
    let strands = [range(5, 3), range(0, 0)];
    let got2 = kernel.eval(&ctx, &points, &strands, 5);
    assert!(got2.is_empty());
}

#[test]
fn multi_workgroup_batch() {
    let Some(ctx) = context_or_skip("multi_workgroup_batch") else {
        return;
    };
    let kernel = GpuHairResample::new(&ctx);
    // 130 strands (> two 64-wide workgroups), each a 4-vertex polyline packed
    // back to back in one shared buffer, with lengths varied per strand.
    let mut points = Vec::new();
    let mut strands = Vec::new();
    for k in 0..130usize {
        let start = points.len();
        let f = (k % 7) as f32;
        points.push(v(0.0, 0.0, 0.0));
        points.push(v(1.0 + f, 0.0, 0.0));
        points.push(v(1.0 + f, 2.0 + f, 0.0));
        points.push(v(1.0 + f, 2.0 + f, 3.0 + f));
        strands.push(range(start, 4));
    }
    let got = kernel.eval(&ctx, &points, &strands, 5);
    assert_eq!(got.len(), 130 * 5);
    assert_matches_groom(&got, &points, &strands, 5);
}
