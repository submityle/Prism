//! Real-device parity for the frustum-plane signed-distance twin:
//! [`GpuSignedDistance`] must reproduce the CPU golden
//! [`Plane::signed_distance`](prism_render_architecture::virtual_geometry::Plane::signed_distance)
//! for every query — the three-term dot product plus the plane distance,
//! evaluated left to right.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The value is `n0*p0 + n1*p1 + n2*p2 + d`. A GPU may contract the products
//! into fused-multiply-adds and reassociate the sum, so the twin is bit-exact
//! against the sequential golden only when every product and every partial sum
//! is exactly representable — then no rounding occurs and fusion / ordering are
//! immaterial. Every query below uses small integer / dyadic operands whose
//! products and running sums stay well inside the 24-bit mantissa, so the
//! result is bit-for-bit identical and asserted with zero tolerance.
//!
//! Provenance: standard inward-plane signed-distance test for frustum culling;
//! no Unreal Engine source or derived code.

use prism_render_architecture::virtual_geometry::Plane;
use prism_virtual_geometry_gpu::{GpuContext, GpuSignedDistance, SignedDistanceQuery};

/// Asserts the twin matches the golden bit-for-bit for every query — valid only
/// when the query's products and partial sums are exactly representable, which
/// every case below guarantees by using small integer / dyadic operands.
fn assert_bit_exact(ctx: &GpuContext, queries: &[SignedDistanceQuery]) {
    let gpu = GpuSignedDistance::new(ctx).evaluate(ctx, queries);
    assert_eq!(gpu.len(), queries.len(), "one distance per query");
    for (i, q) in queries.iter().enumerate() {
        let expected = Plane::new(q.normal, q.distance).signed_distance(q.point);
        assert_eq!(
            gpu[i].to_bits(),
            expected.to_bits(),
            "signed distance must be bit-exact for query {i}: gpu {} ({:#010x}), cpu {expected} ({:#010x})",
            gpu[i],
            gpu[i].to_bits(),
            expected.to_bits(),
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_signed_distance_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping signed-distance parity: no wgpu adapter on this host");
        return;
    };
    // Axis-aligned unit normals and dyadic oblique normals; small integer /
    // dyadic points and distances so every product and partial sum is exact.
    // Signs span interior (positive), exterior (negative) and on-plane (zero).
    let queries = [
        // +X plane through origin, point in front -> +3 (interior).
        SignedDistanceQuery {
            normal: [1.0, 0.0, 0.0],
            point: [3.0, -7.0, 2.0],
            distance: 0.0,
        },
        // -X plane through origin, same point -> -3 (exterior).
        SignedDistanceQuery {
            normal: [-1.0, 0.0, 0.0],
            point: [3.0, -7.0, 2.0],
            distance: 0.0,
        },
        // +Y plane offset by +4, point below -> -6 + 4 = -2 (exterior).
        SignedDistanceQuery {
            normal: [0.0, 1.0, 0.0],
            point: [10.0, -6.0, 5.0],
            distance: 4.0,
        },
        // +Z plane offset by -2, point on it -> 2 + (-2) = 0 (on plane).
        SignedDistanceQuery {
            normal: [0.0, 0.0, 1.0],
            point: [1.0, 1.0, 2.0],
            distance: -2.0,
        },
        // Dyadic oblique normal (0.5, 0.5, 0): 0.5*4 + 0.5*8 + 0 - 2 = 4.
        SignedDistanceQuery {
            normal: [0.5, 0.5, 0.0],
            point: [4.0, 8.0, 100.0],
            distance: -2.0,
        },
        // Dyadic oblique normal (0.25, -0.5, 0.125):
        // 0.25*8 - 0.5*4 + 0.125*16 + 1 = 2 - 2 + 2 + 1 = 3.
        SignedDistanceQuery {
            normal: [0.25, -0.5, 0.125],
            point: [8.0, 4.0, 16.0],
            distance: 1.0,
        },
    ];
    assert_bit_exact(&ctx, &queries);

    // Spot-check the exact scalars the arithmetic must produce.
    let gpu = GpuSignedDistance::new(&ctx).evaluate(&ctx, &queries);
    assert_eq!(gpu[0], 3.0, "+X plane, point x=3 -> +3");
    assert_eq!(gpu[1], -3.0, "-X plane, point x=3 -> -3");
    assert_eq!(gpu[2], -2.0, "+Y plane +4 offset, y=-6 -> -2");
    assert_eq!(gpu[3], 0.0, "+Z plane -2 offset, point on plane -> 0");
    assert_eq!(gpu[4], 4.0, "dyadic oblique normal -> 4");
    assert_eq!(gpu[5], 3.0, "dyadic oblique normal -> 3");
}

#[test]
fn gpu_signed_distance_handles_many_queries() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    // More than one workgroup (>64) to exercise the dispatch tiling and
    // per-thread independence. Small integer operands keep every result exact.
    let mut queries: Vec<SignedDistanceQuery> = Vec::new();
    for k in 0..200i32 {
        let px = f32::from(i16::try_from(k % 21 - 10).expect("small range fits i16"));
        let py = f32::from(i16::try_from(k % 13 - 6).expect("small range fits i16"));
        let pz = f32::from(i16::try_from(k % 9 - 4).expect("small range fits i16"));
        let d = f32::from(i16::try_from(k % 7 - 3).expect("small range fits i16"));
        // Axis-aligned unit normal rotates through the three axes so each query
        // reduces to one exact product plus an exact integer distance.
        let normal = match k % 3 {
            0 => [1.0, 0.0, 0.0],
            1 => [0.0, -1.0, 0.0],
            _ => [0.0, 0.0, 1.0],
        };
        queries.push(SignedDistanceQuery {
            normal,
            point: [px, py, pz],
            distance: d,
        });
    }
    assert_bit_exact(&ctx, &queries);
}

#[test]
fn empty_input_yields_empty_output() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let out = GpuSignedDistance::new(&ctx).evaluate(&ctx, &[]);
    assert!(out.is_empty(), "no queries yields no distances");
}
