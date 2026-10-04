//! Real-device parity for the oriented-bounding-box containment twin:
//! [`GpuObbContainsPoint`](prism_volumetric_gpu::obb_contains_point::GpuObbContainsPoint)
//! must reproduce the `CPU` golden `contains_point` of
//! `prism_physics_core::collider::obb::Obb`, which offsets the point by the box
//! centre, builds an adaptive tolerance from the largest half-extent, and tests
//! the three signed axis projections against the toleranced half-extents.
//!
//! The oracle here is an independent re-implementation of that closed form — the
//! offset `d = point - center`, the adaptive tolerance
//! `tol = 1e-4 + 1e-4 * max_element(half_extents)` and the conjunction of
//! `|dot(d, axis_i)| <= half_extents_i + tol` — written in pure `f32` without
//! `glam`, so the test never imports `prism_physics_core` or
//! `prism_render_architecture`.
//!
//! The fixtures cover a point deep inside the box, a point far outside, a point
//! near a face (reject-sampled off the knee), a box with large half-extents (so
//! the adaptive tolerance scales up), a rotated non-axis-aligned frame, and a
//! batch of distinct queries mixing `contains = 1` and `contains = 0` that
//! catches any `std430` stride aliasing. A reject-sampled sweep over random
//! frames and points follows, plus an empty batch the host short-circuits with
//! no dispatch.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//!
//! # Parity criterion
//!
//! The output is purely discrete, so `contains` and `valid` are both compared
//! exactly. The intermediate projections thread only through dot products and
//! multiply-adds; the sweep reject-samples any query within `1e-2` of a face
//! knee so a fused multiply-add on the device can never flip the verdict.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::obb`；无第三方引擎
//! 源码或衍生代码。

use prism_volumetric_gpu::obb_contains_point::{GpuObbContainsPoint, ObbContainsPointQuery};
use prism_volumetric_gpu::GpuContext;

/// The resolved oracle answer: the containment flag and the validity flag.
struct Expected {
    contains: u32,
    valid: u32,
}

/// The independent oracle for one query, mirroring the on-device kernel's branch
/// structure exactly, including the adaptive tolerance and the three-axis
/// conjunction. Written in pure `f32` with no `glam` dependency.
fn oracle(q: &ObbContainsPointQuery) -> Expected {
    let d = [q.px - q.cx, q.py - q.cy, q.pz - q.cz];
    let max_el = q.hx.max(q.hy).max(q.hz);
    let tol = 1.0e-4 + 1.0e-4 * max_el;

    let p0 = (d[0] * q.a0x + d[1] * q.a0y + d[2] * q.a0z).abs();
    let p1 = (d[0] * q.a1x + d[1] * q.a1y + d[2] * q.a1z).abs();
    let p2 = (d[0] * q.a2x + d[1] * q.a2y + d[2] * q.a2z).abs();

    let inside = p0 <= q.hx + tol && p1 <= q.hy + tol && p2 <= q.hz + tol;
    Expected {
        contains: u32::from(inside),
        valid: 1,
    }
}

/// Dispatches one query and asserts the resolved verdict.
fn assert_parity(ctx: &GpuContext, gpu: &GpuObbContainsPoint, q: ObbContainsPointQuery) {
    let got = gpu.evaluate(ctx, std::slice::from_ref(&q));
    assert_eq!(got.len(), 1, "one result per query");
    assert_result(&got[0], &q);
}

/// Asserts parity for a whole batch, so the shared dispatch exercises the
/// `std430` stride.
fn assert_batch(ctx: &GpuContext, gpu: &GpuObbContainsPoint, queries: &[ObbContainsPointQuery]) {
    let results = gpu.evaluate(ctx, queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(results.iter()) {
        assert_result(r, q);
    }
}

/// Compares one `GPU` result against the oracle for the same query. Both fields
/// are discrete and compared exactly.
fn assert_result(
    r: &prism_volumetric_gpu::obb_contains_point::ObbContainsPointResult,
    q: &ObbContainsPointQuery,
) {
    let e = oracle(q);
    assert_eq!(r.valid, e.valid, "valid flag mismatch: query={q:?}");
    assert_eq!(
        r.contains, e.contains,
        "contains flag mismatch: gpu={} cpu={} query={q:?}",
        r.contains, e.contains
    );
}

/// The world axis-aligned basis, used by the axis-aligned fixtures.
const AXIS_X: [f32; 3] = [1.0, 0.0, 0.0];
const AXIS_Y: [f32; 3] = [0.0, 1.0, 0.0];
const AXIS_Z: [f32; 3] = [0.0, 0.0, 1.0];

#[test]
fn point_deep_inside_is_contained() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuObbContainsPoint::new(&ctx);
    // A point near the box centre, well within every half-extent.
    let q = ObbContainsPointQuery::new(
        [0.0, 0.0, 0.0],
        AXIS_X,
        AXIS_Y,
        AXIS_Z,
        [2.0, 1.0, 0.5],
        [0.3, -0.2, 0.1],
    );
    assert_parity(&ctx, &gpu, q);
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(got[0].contains, 1, "interior point must be contained");
}

#[test]
fn point_far_outside_is_rejected() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuObbContainsPoint::new(&ctx);
    // A point well beyond the x half-extent along the box's own x axis.
    let q = ObbContainsPointQuery::new(
        [0.0, 0.0, 0.0],
        AXIS_X,
        AXIS_Y,
        AXIS_Z,
        [1.0, 1.0, 1.0],
        [5.0, 0.0, 0.0],
    );
    assert_parity(&ctx, &gpu, q);
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(got[0].contains, 0, "far point must be rejected");
}

#[test]
fn point_near_face_inside_the_knee() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuObbContainsPoint::new(&ctx);
    // A point just inside the +x face, pushed well clear of the half-extent knee
    // (tol here is ~1.0002e-4, and the margin below is 0.05) so CPU and GPU agree.
    let q = ObbContainsPointQuery::new(
        [0.0, 0.0, 0.0],
        AXIS_X,
        AXIS_Y,
        AXIS_Z,
        [1.0, 1.0, 1.0],
        [0.95, 0.0, 0.0],
    );
    assert_parity(&ctx, &gpu, q);
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(
        got[0].contains, 1,
        "point inside the knee must be contained"
    );
}

#[test]
fn large_half_extents_scale_the_tolerance() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuObbContainsPoint::new(&ctx);
    // With a large half-extent the adaptive tolerance grows (tol ~ 1e-4 * 1e4 =
    // 1.0001). A point just over the nominal x half-extent but inside the
    // widened band is still contained; the oracle computes the same tol.
    let he = 10_000.0_f32;
    let tol = 1.0e-4 + 1.0e-4 * he;
    let q = ObbContainsPointQuery::new(
        [0.0, 0.0, 0.0],
        AXIS_X,
        AXIS_Y,
        AXIS_Z,
        [he, he, he],
        [he + 0.5 * tol, 0.0, 0.0],
    );
    assert_parity(&ctx, &gpu, q);
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(
        got[0].contains, 1,
        "point inside the widened tolerance band must be contained"
    );
}

/// Builds an orthonormal frame rotated by a precomputed angle about the world
/// `z` axis, so the fixtures exercise non-axis-aligned boxes. The cos/sin
/// constants are host-side `f32` literals (the test is not `WGSL`).
fn rotated_frame_z(cos_t: f32, sin_t: f32) -> ([f32; 3], [f32; 3], [f32; 3]) {
    let axis0 = [cos_t, sin_t, 0.0];
    let axis1 = [-sin_t, cos_t, 0.0];
    let axis2 = [0.0, 0.0, 1.0];
    (axis0, axis1, axis2)
}

#[test]
fn rotated_box_contains_point_in_local_frame() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuObbContainsPoint::new(&ctx);
    // A box rotated 30 degrees about z (cos/sin precomputed). The query point is
    // placed at a known local offset (0.5, 0.0) along the box axes, well inside
    // the half-extents, then expressed in world space via the same axes.
    let cos_t = 0.866_025_4_f32;
    let sin_t = 0.5_f32;
    let (axis0, axis1, axis2) = rotated_frame_z(cos_t, sin_t);
    let center = [1.0, -2.0, 0.5];
    // Local coords (0.5, 0.0, 0.0) -> world = center + 0.5*axis0.
    let point = [
        center[0] + 0.5 * axis0[0],
        center[1] + 0.5 * axis0[1],
        center[2] + 0.5 * axis0[2],
    ];
    let q = ObbContainsPointQuery::new(center, axis0, axis1, axis2, [1.0, 1.0, 1.0], point);
    assert_parity(&ctx, &gpu, q);
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(
        got[0].contains, 1,
        "point inside rotated box must be contained"
    );
}

#[test]
fn rotated_box_rejects_point_outside_local_frame() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuObbContainsPoint::new(&ctx);
    // Same rotated box, but the query point is far along the local x axis, well
    // past the half-extent, so it is rejected.
    let cos_t = 0.866_025_4_f32;
    let sin_t = 0.5_f32;
    let (axis0, axis1, axis2) = rotated_frame_z(cos_t, sin_t);
    let center = [1.0, -2.0, 0.5];
    let point = [
        center[0] + 3.0 * axis0[0],
        center[1] + 3.0 * axis0[1],
        center[2] + 3.0 * axis0[2],
    ];
    let q = ObbContainsPointQuery::new(center, axis0, axis1, axis2, [1.0, 1.0, 1.0], point);
    assert_parity(&ctx, &gpu, q);
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(
        got[0].contains, 0,
        "point outside rotated box must be rejected"
    );
}

#[test]
fn batch_stride_mixes_inside_and_outside() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuObbContainsPoint::new(&ctx);
    // A batch of distinct queries, deliberately mixing contained and rejected
    // verdicts, exercises the std430 query/result stride: every slot must read
    // and write its own non-aliased data.
    let cos_t = 0.866_025_4_f32;
    let sin_t = 0.5_f32;
    let (r0, r1, r2) = rotated_frame_z(cos_t, sin_t);
    let queries = [
        ObbContainsPointQuery::new(
            [0.0, 0.0, 0.0],
            AXIS_X,
            AXIS_Y,
            AXIS_Z,
            [2.0, 1.0, 0.5],
            [0.3, -0.2, 0.1],
        ),
        ObbContainsPointQuery::new(
            [0.0, 0.0, 0.0],
            AXIS_X,
            AXIS_Y,
            AXIS_Z,
            [1.0, 1.0, 1.0],
            [5.0, 0.0, 0.0],
        ),
        ObbContainsPointQuery::new(
            [1.0, -2.0, 0.5],
            r0,
            r1,
            r2,
            [1.0, 1.0, 1.0],
            [1.0 + 0.5 * r0[0], -2.0 + 0.5 * r0[1], 0.5 + 0.5 * r0[2]],
        ),
        ObbContainsPointQuery::new(
            [1.0, -2.0, 0.5],
            r0,
            r1,
            r2,
            [1.0, 1.0, 1.0],
            [1.0 + 3.0 * r0[0], -2.0 + 3.0 * r0[1], 0.5 + 3.0 * r0[2]],
        ),
    ];
    assert_batch(&ctx, &gpu, &queries);
}

/// A small deterministic linear-congruential generator so the sweep needs no
/// external randomness. Constants are the Numerical Recipes values.
struct Lcg {
    state: u32,
}

impl Lcg {
    fn new(seed: u32) -> Self {
        Lcg { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        self.state
    }

    /// A `[0, 1)` fraction built from the top bits, keeping the fixture pure
    /// integer host-side with no transcendental call.
    fn next_unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }

    /// A `[lo, hi)` fraction.
    fn next_range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_unit()
    }
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuObbContainsPoint::new(&ctx);
    let mut rng = Lcg::new(0x0B_B0_17_73);
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        // A rotated frame about z built from a precomputed angle fraction. The
        // host may use a lookup-free rational rotation to stay orthonormal: pick
        // a random unit (c, s) from a Pythagorean-style parameterisation.
        let m = rng.next_range(-4.0, 4.0);
        let denom = 1.0 + m * m;
        let cos_t = (1.0 - m * m) / denom;
        let sin_t = 2.0 * m / denom;
        let axis0 = [cos_t, sin_t, 0.0];
        let axis1 = [-sin_t, cos_t, 0.0];
        let axis2 = [0.0, 0.0, 1.0];

        let center = [
            rng.next_range(-3.0, 3.0),
            rng.next_range(-3.0, 3.0),
            rng.next_range(-3.0, 3.0),
        ];
        let half_extents = [
            rng.next_range(0.3, 2.5),
            rng.next_range(0.3, 2.5),
            rng.next_range(0.3, 2.5),
        ];
        let point = [
            rng.next_range(-5.0, 5.0),
            rng.next_range(-5.0, 5.0),
            rng.next_range(-5.0, 5.0),
        ];

        let q = ObbContainsPointQuery::new(center, axis0, axis1, axis2, half_extents, point);

        // Reject-sample any query within 1e-2 of a face knee on any axis so a
        // fused multiply-add on the device cannot flip the discrete verdict.
        let d = [
            point[0] - center[0],
            point[1] - center[1],
            point[2] - center[2],
        ];
        let max_el = half_extents[0].max(half_extents[1]).max(half_extents[2]);
        let tol = 1.0e-4 + 1.0e-4 * max_el;
        let proj = [
            (d[0] * axis0[0] + d[1] * axis0[1] + d[2] * axis0[2]).abs(),
            (d[0] * axis1[0] + d[1] * axis1[1] + d[2] * axis1[2]).abs(),
            (d[0] * axis2[0] + d[1] * axis2[1] + d[2] * axis2[2]).abs(),
        ];
        let knee_ok = (proj[0] - (half_extents[0] + tol)).abs() > 1.0e-2
            && (proj[1] - (half_extents[1] + tol)).abs() > 1.0e-2
            && (proj[2] - (half_extents[2] + tol)).abs() > 1.0e-2;
        if !knee_ok {
            continue;
        }
        queries.push(q);
    }

    assert_batch(&ctx, &gpu, &queries);
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuObbContainsPoint::new(&ctx);
    assert!(
        gpu.evaluate(&ctx, &[]).is_empty(),
        "empty batch returns an empty vector with no dispatch"
    );
}
