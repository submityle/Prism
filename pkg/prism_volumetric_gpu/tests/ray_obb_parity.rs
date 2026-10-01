//! Real-device parity for the ray vs *oriented bounding box* (`OBB`) twin:
//! [`GpuRayObb`](prism_volumetric_gpu::ray_obb::GpuRayObb) must reproduce the
//! `CPU` golden
//! [`Obb::span`](prism_render_architecture::particle::ray_obb::Obb::span),
//! [`Obb::first_hit`](prism_render_architecture::particle::ray_obb::Obb::first_hit)
//! and
//! [`Obb::intersects`](prism_render_architecture::particle::ray_obb::Obb::intersects)
//! across an empty batch, a frontal penetration, a clear miss driven by a
//! parallel-slab guard, an origin inside the box, a thin slab box, a box wholly
//! behind the origin, a rotated-frame hit (point and outward normal both
//! rigidly rotated), a sub-`EPS` near-zero direction axis guard, zero-length
//! directions and a large pseudo-random batch of rotated boxes compared lane for
//! lane. Each expected verdict is taken straight from the reference `span`,
//! `first_hit` and `intersects` entry points — the full `CPU` path — not from a
//! re-implementation.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each query is a fixed, non-reorderable sequence of three guarded divisions
//! and a running interval intersection, so `CPU` and `GPU` evaluate the same
//! closed form in the same associativity. They are not bit-exact: a `GPU` may
//! fuse a multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place. The comparison therefore
//! allows `abs_diff <= 1e-4` or `rel_diff <= 1e-3` on the chord, crossing, point
//! and normal values and asserts an *exact* match on the discrete hit flags. For
//! the random batch a verdict disagreement is tolerated only when the chord sits
//! inside a narrow tie band (a grazing chord `|t_exit - t_enter| <= 1e-2` or a
//! forward-visibility boundary `|t_exit| <= 1e-2`), the only place where a legal
//! `ULP` perturbation can flip a `<=` or `>=` verdict; the named fixtures are all
//! placed clear of such boundaries and clear of box corners so they assert exact
//! flags, points and normals unconditionally.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::ray_obb`；
//! standard slab-method ray/`OBB` intersection; no third-party engine source or
//! derived code.

use prism_render_architecture::particle::ray_obb::{Obb, Ray, Vec3};
use prism_volumetric_gpu::ray_obb::{GpuRayObb, RayObbQuery, RayObbResult};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on the chord, crossing, point and normal values. A
/// `GPU` may fuse a multiply-add the scalar reference leaves separate,
/// perturbing the low mantissa bits by a few units in the last place; `1e-4`
/// admits that legal slack while still failing a genuinely wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in the
/// last place exceed the absolute floor.
const REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Half-width of the tie band inside which a `<=`/`>=` verdict can legally flip
/// under a `ULP`-scale perturbation, so a boolean disagreement there is
/// tolerated for the random batch (never for the clear-of-boundary fixtures).
const TIE: f32 = 1.0e-2;

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

/// The reference golden path for one query, calling the public `span`,
/// `first_hit` and `intersects` entry points directly so the comparison runs the
/// full `CPU` contract rather than a re-implementation. Returns the ordered
/// chord, the nearest forward hit parameter, that hit's point and normal, and the
/// forward predicate.
fn golden(q: &RayObbQuery) -> (Option<(f32, f32)>, Option<(f32, Vec3, Vec3)>, bool) {
    let span = q.obb.span(q.ray);
    let hit = q.obb.first_hit(q.ray).map(|h| (h.t, h.point, h.normal));
    let intersects = q.obb.intersects(q.ray);
    (span, hit, intersects)
}

/// The axis-aligned unit box `[-1, 1]` on every axis built as a degenerate
/// `OBB`, the fixture most of the named cases probe.
fn unit_cube() -> Obb {
    Obb::from_aabb(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0))
}

/// Builds one query from a ray and an oriented box.
fn query(origin: Vec3, dir: Vec3, obb: Obb) -> RayObbQuery {
    RayObbQuery {
        ray: Ray::new(origin, dir),
        obb,
    }
}

/// Rotates a vector by `angle` radians about the `z` axis; used host-side only
/// (never in the kernel) to build genuinely oriented frames and the matching
/// rotated rays for the rotation-invariance fixture.
#[expect(
    clippy::disallowed_methods,
    reason = "host-side fixture frame construction only; both the CPU golden and the GPU query consume the identical precomputed axes, so cross-platform libm determinism is irrelevant and the kernel itself uses no transcendental functions"
)]
fn rot_z(v: Vec3, angle: f32) -> Vec3 {
    let (s, c) = angle.sin_cos();
    Vec3::new(v.x * c - v.y * s, v.x * s + v.y * c, v.z)
}

/// Builds a proper orthonormal frame by rotating the world basis about `z`, then
/// `y`, then `x`; host-side only. Returns `(axis_u, axis_v, axis_w)`.
#[expect(
    clippy::disallowed_methods,
    reason = "host-side fixture frame construction only; both the CPU golden and the GPU query consume the identical precomputed axes, so cross-platform libm determinism is irrelevant and the kernel itself uses no transcendental functions"
)]
fn frame(az: f32, ay: f32, ax: f32) -> (Vec3, Vec3, Vec3) {
    let rz = |v: Vec3| rot_z(v, az);
    let ry = |v: Vec3| {
        let (s, c) = ay.sin_cos();
        Vec3::new(v.x * c + v.z * s, v.y, -v.x * s + v.z * c)
    };
    let rx = |v: Vec3| {
        let (s, c) = ax.sin_cos();
        Vec3::new(v.x, v.y * c - v.z * s, v.y * s + v.z * c)
    };
    let r = |v: Vec3| rx(ry(rz(v)));
    (
        r(Vec3::new(1.0, 0.0, 0.0)),
        r(Vec3::new(0.0, 1.0, 0.0)),
        r(Vec3::new(0.0, 0.0, 1.0)),
    )
}

/// Runs the `GPU` dispatch and asserts strict lane-for-lane parity against the
/// `CPU` golden: the two hit flags match exactly, the chord endpoints match
/// within tolerance when the line hits, and the nearest forward crossing (plus
/// its point and normal) matches within tolerance when the forward ray hits.
/// Returns the `GPU` verdicts for extra per-test assertions. Use only for
/// fixtures placed clear of every boundary and corner.
fn check(ctx: &GpuContext, gpu: &GpuRayObb, queries: &[RayObbQuery]) -> Vec<RayObbResult> {
    let got = gpu.eval(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (lane, (g, q)) in got.iter().zip(queries.iter()).enumerate() {
        let (span, hit, intersects) = golden(q);
        assert_eq!(
            g.span_hit,
            span.is_some(),
            "lane {lane}: span_hit gpu {} vs cpu {}",
            g.span_hit,
            span.is_some()
        );
        assert_eq!(
            g.forward_hit, intersects,
            "lane {lane}: forward_hit gpu {} vs cpu {intersects}",
            g.forward_hit
        );
        if let Some((t_enter, t_exit)) = span {
            assert!(
                close(g.t_enter, t_enter),
                "lane {lane}: t_enter gpu {} vs cpu {t_enter}",
                g.t_enter
            );
            assert!(
                close(g.t_exit, t_exit),
                "lane {lane}: t_exit gpu {} vs cpu {t_exit}",
                g.t_exit
            );
        }
        if let Some((t, point, normal)) = hit {
            assert!(
                close(g.first_t, t),
                "lane {lane}: first_t gpu {} vs cpu {t}",
                g.first_t
            );
            assert!(
                close_vec(g.point, point),
                "lane {lane}: point gpu {:?} vs cpu {point:?}",
                g.point
            );
            assert!(
                close_vec(g.normal, normal),
                "lane {lane}: normal gpu {:?} vs cpu {normal:?}",
                g.normal
            );
        }
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

#[test]
fn empty_batch_is_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayObb::new(&ctx);
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch yields an empty result");
}

#[test]
fn frontal_penetration_with_parallel_guards() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayObb::new(&ctx);
    // Straight down +x against the unit cube: the v and w axis projections of the
    // direction are exactly zero, so both take the parallel-slab guard with the
    // origin inside the slab. Chord [4, 6], forward hit entering the -x face at 4.
    let q = query(
        Vec3::new(-5.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        unit_cube(),
    );
    let got = check(&ctx, &gpu, &[q]);
    assert!(
        got[0].span_hit && got[0].forward_hit,
        "frontal ray should hit"
    );
    assert!(close(got[0].t_enter, 4.0), "t_enter {}", got[0].t_enter);
    assert!(close(got[0].t_exit, 6.0), "t_exit {}", got[0].t_exit);
    assert!(close(got[0].first_t, 4.0), "first {}", got[0].first_t);
    assert!(
        close_vec(got[0].normal, Vec3::new(-1.0, 0.0, 0.0)),
        "normal {:?}",
        got[0].normal
    );
}

#[test]
fn parallel_guard_clear_miss() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayObb::new(&ctx);
    // Same +x ray, but offset to y = 9 — far outside the y slab. The y-axis
    // parallel guard rejects the whole query, so both readings must be an exact
    // miss. This is the dedicated "projected dir component zero and projected
    // origin outside the slab" guard case; its boolean verdict must match.
    let q = query(
        Vec3::new(-5.0, 9.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        unit_cube(),
    );
    let got = check(&ctx, &gpu, &[q]);
    assert!(
        !got[0].span_hit && !got[0].forward_hit,
        "a ray parallel to and outside the y slab must miss both readings"
    );
}

#[test]
fn origin_inside_box_exits_forward() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayObb::new(&ctx);
    // Origin at the center: the chord straddles zero (t_enter < 0 <= t_exit) and
    // a forward ray still hits, with its first visible crossing at the +x exit
    // face whose outward normal is +x.
    let q = query(
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        unit_cube(),
    );
    let got = check(&ctx, &gpu, &[q]);
    assert!(
        got[0].span_hit && got[0].forward_hit,
        "inside origin should hit"
    );
    assert!(close(got[0].t_enter, -1.0), "t_enter {}", got[0].t_enter);
    assert!(close(got[0].t_exit, 1.0), "t_exit {}", got[0].t_exit);
    assert!(close(got[0].first_t, 1.0), "first {}", got[0].first_t);
    assert!(
        close_vec(got[0].normal, Vec3::new(1.0, 0.0, 0.0)),
        "normal {:?}",
        got[0].normal
    );
}

#[test]
fn thin_slab_box_hit() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayObb::new(&ctx);
    // A world-aligned slab thin in y (half 0.5) and long in x (half 3). A ray
    // straight down -y hits the top (+y) face at t = 4.5, point (0, 0.5, 0),
    // outward normal +y.
    let obb = Obb::new(
        Vec3::ZERO,
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(3.0, 0.5, 1.0),
    );
    let q = query(Vec3::new(0.0, 5.0, 0.0), Vec3::new(0.0, -1.0, 0.0), obb);
    let got = check(&ctx, &gpu, &[q]);
    assert!(got[0].span_hit && got[0].forward_hit, "top ray should hit");
    assert!(close(got[0].first_t, 4.5), "first {}", got[0].first_t);
    assert!(
        close_vec(got[0].point, Vec3::new(0.0, 0.5, 0.0)),
        "point {:?}",
        got[0].point
    );
    assert!(
        close_vec(got[0].normal, Vec3::new(0.0, 1.0, 0.0)),
        "normal {:?}",
        got[0].normal
    );
}

#[test]
fn box_behind_origin_is_line_only() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayObb::new(&ctx);
    // Origin past the box: the infinite line still crosses it with a negative
    // chord [-6, -4], but the forward ray reports a miss (t_exit < 0).
    let q = query(
        Vec3::new(5.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        unit_cube(),
    );
    let got = check(&ctx, &gpu, &[q]);
    assert!(
        got[0].span_hit && !got[0].forward_hit,
        "a box behind the origin is a line hit but a forward miss"
    );
    assert!(close(got[0].t_enter, -6.0), "t_enter {}", got[0].t_enter);
    assert!(close(got[0].t_exit, -4.0), "t_exit {}", got[0].t_exit);
}

#[test]
fn rotated_frame_hit_is_rigidly_transformed() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayObb::new(&ctx);
    // The reference guarantees that rotating both the box frame and the ray by a
    // common rotation leaves the hit parameter invariant and rigidly rotates the
    // point and normal. Rotate a unit cube and the frontal +x ray by +0.6 rad
    // about z and confirm the GPU reproduces the rotated point and normal.
    let angle = 0.6_f32;
    let (u, v, w) = frame(angle, 0.0, 0.0);
    let rotated = Obb::new(Vec3::ZERO, u, v, w, Vec3::splat(1.0));
    let origin = rot_z(Vec3::new(-5.0, 0.0, 0.0), angle);
    let dir = rot_z(Vec3::new(1.0, 0.0, 0.0), angle);
    let q = query(origin, dir, rotated);
    let got = check(&ctx, &gpu, &[q]);
    assert!(
        got[0].span_hit && got[0].forward_hit,
        "rotated frontal ray should hit"
    );
    // Parameter is invariant under the common rotation: still entry at 4.
    assert!(close(got[0].first_t, 4.0), "first {}", got[0].first_t);
    let want_point = rot_z(Vec3::new(-1.0, 0.0, 0.0), angle);
    let want_normal = rot_z(Vec3::new(-1.0, 0.0, 0.0), angle);
    assert!(
        close_vec(got[0].point, want_point),
        "point {:?} vs {want_point:?}",
        got[0].point
    );
    assert!(
        close_vec(got[0].normal, want_normal),
        "normal {:?} vs {want_normal:?}",
        got[0].normal
    );
}

#[test]
fn near_zero_direction_axis_guard() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayObb::new(&ctx);
    // The y projection of the direction is 1e-8, below the EPS = 1e-6 parallel
    // floor, so the y axis takes the parallel guard. With the origin at y = 9,
    // far outside the slab, both devices must classify it as parallel-and-outside
    // and report a clean miss — exercising the sub-EPS direction guard, not a
    // finite division.
    let q = query(
        Vec3::new(-5.0, 9.0, 0.0),
        Vec3::new(1.0, 1.0e-8, 0.0),
        unit_cube(),
    );
    let got = check(&ctx, &gpu, &[q]);
    assert!(
        !got[0].span_hit && !got[0].forward_hit,
        "a sub-EPS direction component outside the slab must miss"
    );
}

#[test]
fn zero_direction_is_a_miss() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayObb::new(&ctx);
    // A zero-length direction is rejected as a non-ray on both devices, whether
    // the origin is inside the box or outside it, so a point never spans the
    // infinite sentinel interval.
    let inside = query(
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 0.0),
        unit_cube(),
    );
    let outside = query(
        Vec3::new(5.0, 5.0, 5.0),
        Vec3::new(0.0, 0.0, 0.0),
        unit_cube(),
    );
    let got = check(&ctx, &gpu, &[inside, outside]);
    assert!(
        got.iter().all(|r| !r.span_hit && !r.forward_hit),
        "a zero-length direction must miss both readings everywhere"
    );
}

#[test]
fn random_batch_matches_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayObb::new(&ctx);
    let mut state = 0x_0bb0_a1b2_c3d4_e5f6_u64;

    let mut saw_span_hit = false;
    let mut saw_span_miss = false;
    let mut saw_forward_hit = false;

    for _round in 0u32..8 {
        let mut queries = Vec::with_capacity(128);
        for _ in 0..128 {
            // A box centered somewhere in [-4, 4] with half extents in
            // [0.5, 2.0], never degenerate, carried in a random orthonormal frame.
            let center = Vec3::new(
                lcg(&mut state) * 8.0 - 4.0,
                lcg(&mut state) * 8.0 - 4.0,
                lcg(&mut state) * 8.0 - 4.0,
            );
            let half = Vec3::new(
                0.5 + lcg(&mut state) * 1.5,
                0.5 + lcg(&mut state) * 1.5,
                0.5 + lcg(&mut state) * 1.5,
            );
            let (u, v, w) = frame(
                lcg(&mut state) * 6.2831855,
                lcg(&mut state) * 6.2831855,
                lcg(&mut state) * 6.2831855,
            );
            let obb = Obb::new(center, u, v, w, half);

            // Origin on a loose shell around the box center so misses stay
            // clearly misses rather than hovering on a face.
            let origin = Vec3::new(
                center.x + lcg(&mut state) * 16.0 - 8.0,
                center.y + lcg(&mut state) * 16.0 - 8.0,
                center.z + lcg(&mut state) * 16.0 - 8.0,
            );
            // Aim at the center plus a random offset wider than the box, mixing
            // clear hits with clear misses.
            let target = Vec3::new(
                center.x + lcg(&mut state) * 6.0 - 3.0,
                center.y + lcg(&mut state) * 6.0 - 3.0,
                center.z + lcg(&mut state) * 6.0 - 3.0,
            );
            let mut dir = Vec3::new(
                target.x - origin.x,
                target.y - origin.y,
                target.z - origin.z,
            );
            // Occasionally force one world direction component to exactly zero to
            // exercise the parallel-slab guard against a rotated axis.
            let axis = (lcg(&mut state) * 4.0) as u32;
            match axis {
                0 => dir.x = 0.0,
                1 => dir.y = 0.0,
                2 => dir.z = 0.0,
                _ => {}
            }
            // Guard against an accidental zero-length direction.
            let dd = dir.x * dir.x + dir.y * dir.y + dir.z * dir.z;
            if dd < 0.25 {
                dir.x += 1.0;
            }
            queries.push(query(origin, dir, obb));
        }

        let got = gpu.eval(&ctx, &queries);
        assert_eq!(got.len(), queries.len(), "one result per query");
        for (lane, (g, q)) in got.iter().zip(queries.iter()).enumerate() {
            let (span, hit, intersects) = golden(q);

            // Boolean flags match exactly unless the chord sits inside the tie
            // band, the only place a ULP perturbation can legally flip a verdict.
            if g.span_hit != span.is_some() {
                let gap = if g.span_hit {
                    (g.t_exit - g.t_enter).abs()
                } else if let Some((e, x)) = span {
                    (x - e).abs()
                } else {
                    f32::INFINITY
                };
                assert!(
                    gap <= TIE,
                    "lane {lane}: span_hit disagreement off the tie band: gpu {} cpu {} gap {gap}",
                    g.span_hit,
                    span.is_some()
                );
            }
            if g.forward_hit != intersects {
                let near = if g.span_hit {
                    g.t_exit.abs() <= TIE || (g.t_exit - g.t_enter).abs() <= TIE
                } else if let Some((e, x)) = span {
                    x.abs() <= TIE || (x - e).abs() <= TIE
                } else {
                    false
                };
                assert!(
                    near,
                    "lane {lane}: forward_hit disagreement off the tie band: gpu {} cpu {intersects}",
                    g.forward_hit
                );
            }

            // Values are compared only where both devices agree on a hit.
            if g.span_hit
                && let Some((t_enter, t_exit)) = span
            {
                assert!(
                    close(g.t_enter, t_enter),
                    "lane {lane}: t_enter gpu {} vs cpu {t_enter}",
                    g.t_enter
                );
                assert!(
                    close(g.t_exit, t_exit),
                    "lane {lane}: t_exit gpu {} vs cpu {t_exit}",
                    g.t_exit
                );
            }
            if g.forward_hit
                && let Some((t, _point, _normal)) = hit
            {
                assert!(
                    close(g.first_t, t),
                    "lane {lane}: first_t gpu {} vs cpu {t}",
                    g.first_t
                );
            }

            saw_span_hit |= span.is_some();
            saw_span_miss |= span.is_none();
            saw_forward_hit |= intersects;
        }
    }

    // A large random spread must exercise both verdict classes, so the test is
    // not trivially passing on an all-hit or all-miss batch.
    assert!(
        saw_span_hit && saw_span_miss && saw_forward_hit,
        "random batch should produce span hits, span misses and forward hits"
    );
}
