//! Real-device parity for the point vs *oriented bounding box* (`OBB`)
//! nearest-point twin:
//! [`GpuClosestPointObb`](prism_volumetric_gpu::closest_point_obb::GpuClosestPointObb)
//! must reproduce the `CPU` golden
//! [`Obb::closest_point`](prism_render_architecture::particle::closest_point_obb::Obb::closest_point)
//! across an empty batch, an interior point that maps to itself, a single-face
//! clamp, a beyond-corner clamp, a rotated-frame clamp, a thin (non-degenerate)
//! slab, a rigid-rotation invariance check and a large pseudo-random batch of
//! rotated boxes compared lane for lane. Each expected reading is taken straight
//! from the reference `closest_point` entry point — the full `CPU` path — not
//! from a re-implementation.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each query is a fixed, non-reorderable sequence of three per-axis clamps and
//! one length, so `CPU` and `GPU` evaluate the same closed form in the same
//! associativity. They are not bit-exact: a `GPU` may fuse a multiply-add the
//! scalar reference leaves separate, perturbing the low mantissa bits by a few
//! units in the last place. The comparison therefore allows `abs_diff <= 1e-4`
//! or `rel_diff <= 1e-3` on the nearest point and the two distances and asserts
//! an *exact* match on the discrete inside flag. For the random batch an inside
//! disagreement is tolerated only when the query sits inside a narrow tie band
//! around a face (`||proj| - half| <= 1e-2`), the only place where a legal `ULP`
//! perturbation can flip the `<=` half-extent verdict; the named fixtures are
//! all placed clear of every face, edge and corner so they assert exact flags,
//! points and distances unconditionally.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::closest_point_obb`；
//! standard per-axis interval clamp of a point against an `OBB`; no third-party
//! engine source or derived code.

use prism_render_architecture::particle::closest_point_obb::{Obb, Vec3};
use prism_volumetric_gpu::closest_point_obb::{
    ClosestPointObbQuery, ClosestPointObbResult, GpuClosestPointObb,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on the nearest point and the two distances. A `GPU` may
/// fuse a multiply-add the scalar reference leaves separate, perturbing the low
/// mantissa bits by a few units in the last place; `1e-4` admits that legal
/// slack while still failing a genuinely wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in the
/// last place exceed the absolute floor.
const REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Half-width of the tie band around a face inside which the `<=` half-extent
/// verdict can legally flip under a `ULP`-scale perturbation, so an inside-flag
/// disagreement there is tolerated for the random batch (never for the
/// clear-of-boundary fixtures).
const TIE: f32 = 1.0e-2;

/// One-quarter-turn cosine/sine literal (`1 / sqrt(2)`), written as a plain
/// constant so the fixtures never call a transcendental function.
const SQRT_1_2: f32 = core::f32::consts::FRAC_1_SQRT_2;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
}

/// Returns whether two vectors agree component-wise within the parity bound.
fn close_vec(a: Vec3, b: Vec3) -> bool {
    close(a.x, b.x) && close(a.y, b.y) && close(a.z, b.z)
}

/// The reference golden path for one query, calling the public `closest_point`
/// entry point directly so the comparison runs the full `CPU` contract rather
/// than a re-implementation.
fn golden(q: &ClosestPointObbQuery) -> ClosestPointObbResult {
    let r = q.obb.closest_point(q.point);
    ClosestPointObbResult {
        point: r.point,
        distance: r.distance,
        distance_squared: r.distance_squared,
        inside: r.inside,
    }
}

/// The axis-aligned unit box `[-1, 1]` on every axis built as a degenerate
/// `OBB`, the fixture most of the named cases probe.
fn unit_aabb() -> Obb {
    Obb::from_aabb(Vec3::ZERO, [1.0, 1.0, 1.0])
}

/// A box rotated 45 degrees about `z`, centered at the origin, half 1 — a
/// genuinely oriented frame whose axes are exact `1 / sqrt(2)` literals so the
/// fixture stays transcendental-free.
fn rot45() -> Obb {
    Obb::new(
        Vec3::ZERO,
        [
            Vec3::new(SQRT_1_2, SQRT_1_2, 0.0),
            Vec3::new(-SQRT_1_2, SQRT_1_2, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ],
        [1.0, 1.0, 1.0],
    )
}

/// Rotates a point 45 degrees about `z` using the `1 / sqrt(2)` literal, so the
/// rigid-rotation fixture is built with pure arithmetic and never a
/// transcendental call.
fn rot_z45(v: Vec3) -> Vec3 {
    Vec3::new(
        v.x * SQRT_1_2 - v.y * SQRT_1_2,
        v.x * SQRT_1_2 + v.y * SQRT_1_2,
        v.z,
    )
}

/// Builds one query from a query point and an oriented box.
fn query(point: Vec3, obb: Obb) -> ClosestPointObbQuery {
    ClosestPointObbQuery { point, obb }
}

/// Runs the `GPU` dispatch and asserts strict lane-for-lane parity against the
/// `CPU` golden: the inside flag matches exactly and the nearest point and both
/// distances match within tolerance. Returns the `GPU` readings for extra
/// per-test assertions. Use only for fixtures placed clear of every face, edge
/// and corner.
fn check(
    ctx: &GpuContext,
    gpu: &GpuClosestPointObb,
    queries: &[ClosestPointObbQuery],
) -> Vec<ClosestPointObbResult> {
    let got = gpu.eval(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (lane, (g, q)) in got.iter().zip(queries.iter()).enumerate() {
        let want = golden(q);
        assert_eq!(
            g.inside, want.inside,
            "lane {lane}: inside gpu {} vs cpu {}",
            g.inside, want.inside
        );
        assert!(
            close_vec(g.point, want.point),
            "lane {lane}: point gpu {:?} vs cpu {:?}",
            g.point,
            want.point
        );
        assert!(
            close(g.distance, want.distance),
            "lane {lane}: distance gpu {} vs cpu {}",
            g.distance,
            want.distance
        );
        assert!(
            close(g.distance_squared, want.distance_squared),
            "lane {lane}: distance_squared gpu {} vs cpu {}",
            g.distance_squared,
            want.distance_squared
        );
    }
    got
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

/// Builds a proper orthonormal frame from a (bias-nonzero) quaternion through
/// the standard quaternion-to-matrix columns — pure arithmetic plus one `sqrt`
/// for the normalization, so the random frames stay transcendental-free and
/// exactly orthonormal (the solver assumes orthonormal axes). Returns the three
/// local axes.
fn quat_frame(qx: f32, qy: f32, qz: f32, qw: f32) -> [Vec3; 3] {
    let n = (qx * qx + qy * qy + qz * qz + qw * qw).sqrt();
    let x = qx / n;
    let y = qy / n;
    let z = qz / n;
    let w = qw / n;
    [
        Vec3::new(
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + z * w),
            2.0 * (x * z - y * w),
        ),
        Vec3::new(
            2.0 * (x * y - z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z + x * w),
        ),
        Vec3::new(
            2.0 * (x * z + y * w),
            2.0 * (y * z - x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
    ]
}

/// Whether the query lies within the tie band around any face of its box, where
/// a `ULP`-scale perturbation could legally flip the `<=` half-extent inside
/// verdict. Computed from the public `Vec3` dot product (not the result), so it
/// gates only the boolean tolerance and never substitutes for the golden.
fn near_face(q: &ClosestPointObbQuery) -> bool {
    let rel = q.point.minus(q.obb.center);
    q.obb
        .axes
        .iter()
        .zip(q.obb.half.iter())
        .any(|(axis, &half)| (rel.dot(*axis).abs() - half).abs() <= TIE)
}

#[test]
fn empty_batch_is_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClosestPointObb::new(&ctx);
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch yields an empty result");
}

#[test]
fn interior_point_maps_to_itself() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClosestPointObb::new(&ctx);
    // A point well inside the unit box clamps to itself: zero distance, inside.
    let p = Vec3::new(0.25, -0.5, 0.75);
    let got = check(&ctx, &gpu, &[query(p, unit_aabb())]);
    assert!(got[0].inside, "interior point must read inside");
    assert!(close(got[0].distance, 0.0), "distance {}", got[0].distance);
    assert!(
        close_vec(got[0].point, p),
        "nearest {:?} should equal the query",
        got[0].point
    );
}

#[test]
fn single_face_clamp_outside() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClosestPointObb::new(&ctx);
    // Outside along +x only: nearest is the +x face point (1, 0.3, -0.4), the
    // other two coordinates untouched; distance exactly 1, clearly outside.
    let got = check(&ctx, &gpu, &[query(Vec3::new(2.0, 0.3, -0.4), unit_aabb())]);
    assert!(!got[0].inside, "exterior point must read outside");
    assert!(close(got[0].distance, 1.0), "distance {}", got[0].distance);
    assert!(
        close_vec(got[0].point, Vec3::new(1.0, 0.3, -0.4)),
        "nearest {:?}",
        got[0].point
    );
}

#[test]
fn beyond_corner_clamps_to_corner() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClosestPointObb::new(&ctx);
    // Beyond the (+,+,+) corner on every axis: the nearest point is the corner
    // (1, 1, 1), well clear of any face/edge tie, and the point is outside.
    let got = check(&ctx, &gpu, &[query(Vec3::new(3.0, 4.0, 5.0), unit_aabb())]);
    assert!(!got[0].inside, "beyond-corner point must read outside");
    assert!(
        close_vec(got[0].point, Vec3::new(1.0, 1.0, 1.0)),
        "nearest {:?}",
        got[0].point
    );
    // distance^2 = 2^2 + 3^2 + 4^2 = 29.
    assert!(
        close(got[0].distance_squared, 29.0),
        "distance_squared {}",
        got[0].distance_squared
    );
}

#[test]
fn rotated_frame_clamp() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClosestPointObb::new(&ctx);
    // A point outside a 45-degrees-rotated unit box along world +x. Both in-plane
    // projections clamp to opposite faces, so the nearest point is (sqrt 2, 0, 0)
    // and the point is clearly outside.
    let got = check(&ctx, &gpu, &[query(Vec3::new(3.0, 0.0, 0.5), rot45())]);
    assert!(!got[0].inside, "point outside a rotated box must read outside");
    assert!(
        close_vec(got[0].point, Vec3::new(core::f32::consts::SQRT_2, 0.0, 0.5)),
        "nearest {:?}",
        got[0].point
    );
}

#[test]
fn thin_slab_box_clamp() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClosestPointObb::new(&ctx);
    // A world-aligned slab thin in y (half 0.5), long in x (half 3). A point high
    // above the top face clamps straight down to (0, 0.5, 0), distance 4.5. The
    // box is thin but non-degenerate, so no axis collapses.
    let obb = Obb::new(
        Vec3::ZERO,
        [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ],
        [3.0, 0.5, 1.0],
    );
    let got = check(&ctx, &gpu, &[query(Vec3::new(0.0, 5.0, 0.0), obb)]);
    assert!(!got[0].inside, "point above the slab must read outside");
    assert!(close(got[0].distance, 4.5), "distance {}", got[0].distance);
    assert!(
        close_vec(got[0].point, Vec3::new(0.0, 0.5, 0.0)),
        "nearest {:?}",
        got[0].point
    );
}

#[test]
fn rigid_rotation_leaves_distance_invariant() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClosestPointObb::new(&ctx);
    // Rotating both the box frame and the query by a common rotation leaves the
    // distance invariant and rigidly rotates the nearest point. The unrotated
    // case (p against the axis-aligned unit box) has nearest (1, 0.3, -0.4) and
    // distance 1; the rotated case must reproduce the rotated point and the same
    // distance.
    let p = Vec3::new(2.0, 0.3, -0.4);
    let rotated_box = rot45();
    let rotated_p = rot_z45(p);
    let got = check(&ctx, &gpu, &[query(rotated_p, rotated_box)]);
    assert!(!got[0].inside, "rotated exterior point must read outside");
    assert!(close(got[0].distance, 1.0), "distance {}", got[0].distance);
    let want_point = rot_z45(Vec3::new(1.0, 0.3, -0.4));
    assert!(
        close_vec(got[0].point, want_point),
        "nearest {:?} vs {want_point:?}",
        got[0].point
    );
}

#[test]
fn random_batch_matches_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClosestPointObb::new(&ctx);
    let mut state = 0x_c105_e570_0bb0_1234_u64;

    let mut saw_inside = false;
    let mut saw_outside = false;

    for round in 0u32..8 {
        let mut queries = Vec::with_capacity(128);
        for i in 0..128u32 {
            // A box centered somewhere in [-4, 4] with half extents in
            // [0.5, 2.0], never degenerate, carried in a random orthonormal frame.
            let center = Vec3::new(
                lcg(&mut state) * 8.0 - 4.0,
                lcg(&mut state) * 8.0 - 4.0,
                lcg(&mut state) * 8.0 - 4.0,
            );
            let half = [
                0.5 + lcg(&mut state) * 1.5,
                0.5 + lcg(&mut state) * 1.5,
                0.5 + lcg(&mut state) * 1.5,
            ];
            // A quaternion with a positive w bias is always nonzero, so the frame
            // is well defined and orthonormal.
            let axes = quat_frame(
                lcg(&mut state) * 2.0 - 1.0,
                lcg(&mut state) * 2.0 - 1.0,
                lcg(&mut state) * 2.0 - 1.0,
                0.5 + lcg(&mut state),
            );
            let obb = Obb::new(center, axes, half);

            // Alternate between a point drawn tightly around the center (often
            // inside) and one on a loose shell (clearly outside), so the batch
            // exercises both verdict classes.
            let spread = if (i & 1) == 0 { 1.5 } else { 7.0 };
            let point = Vec3::new(
                center.x + lcg(&mut state) * spread * 2.0 - spread,
                center.y + lcg(&mut state) * spread * 2.0 - spread,
                center.z + lcg(&mut state) * spread * 2.0 - spread,
            );
            queries.push(query(point, obb));
        }

        let got = gpu.eval(&ctx, &queries);
        assert_eq!(got.len(), queries.len(), "one result per query");
        for (lane, (g, q)) in got.iter().zip(queries.iter()).enumerate() {
            let want = golden(q);

            // The inside flag matches exactly unless the query sits in the tie
            // band around a face, the only place a ULP perturbation can flip the
            // `<=` half-extent verdict.
            if g.inside != want.inside {
                assert!(
                    near_face(q),
                    "round {round} lane {lane}: inside disagreement off the tie band: gpu {} cpu {}",
                    g.inside,
                    want.inside
                );
            }

            // The nearest point and both distances are continuous in the query,
            // so they are compared within tolerance on every lane.
            assert!(
                close_vec(g.point, want.point),
                "round {round} lane {lane}: point gpu {:?} vs cpu {:?}",
                g.point,
                want.point
            );
            assert!(
                close(g.distance, want.distance),
                "round {round} lane {lane}: distance gpu {} vs cpu {}",
                g.distance,
                want.distance
            );
            assert!(
                close(g.distance_squared, want.distance_squared),
                "round {round} lane {lane}: distance_squared gpu {} vs cpu {}",
                g.distance_squared,
                want.distance_squared
            );

            saw_inside |= want.inside;
            saw_outside |= !want.inside;
        }
    }

    // A large random spread must exercise both verdict classes, so the test is
    // not trivially passing on an all-inside or all-outside batch.
    assert!(
        saw_inside && saw_outside,
        "random batch should produce both inside and outside verdicts"
    );
}
