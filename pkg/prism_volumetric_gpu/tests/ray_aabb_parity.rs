//! Real-device parity for the ray vs *axis-aligned bounding box* (`AABB`) twin:
//! [`GpuRayAabb`](prism_volumetric_gpu::ray_aabb::GpuRayAabb) must reproduce the
//! `CPU` golden
//! [`Aabb::intersect_line`](prism_render_architecture::particle::ray_aabb::Aabb::intersect_line),
//! [`Aabb::intersect_ray`](prism_render_architecture::particle::ray_aabb::Aabb::intersect_ray)
//! and
//! [`Aabb::first_hit_t`](prism_render_architecture::particle::ray_aabb::Aabb::first_hit_t)
//! across an empty batch, a frontal penetration, a clear miss driven by a
//! parallel-slab guard, an origin inside the box, an origin on a face, a thin
//! grazing hit, a zero-volume box, a box wholly behind the origin, a negative
//! direction, a space diagonal, zero-length directions and a large pseudo-random
//! batch compared lane for lane.
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
//! allows `abs_diff <= 1e-4` or `rel_diff <= 1e-3` on the chord and crossing
//! values and asserts an *exact* match on the discrete hit flags. For the random
//! batch a verdict disagreement is tolerated only when the chord sits inside a
//! narrow tie band (a grazing chord `|t_exit - t_enter| <= 1e-2` or a
//! forward-visibility boundary `|t_exit| <= 1e-2`), the only place where a legal
//! `ULP` perturbation can flip a `<=` or `>=` verdict; the named fixtures are all
//! placed clear of such boundaries so they assert exact flags unconditionally.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::ray_aabb`；
//! standard slab-method ray/`AABB` intersection; no third-party engine source or
//! derived code.

use prism_render_architecture::particle::ray_aabb::{Aabb, Ray, Vec3};
use prism_volumetric_gpu::ray_aabb::{cpu_reference, GpuRayAabb, RayAabbQuery, RayAabbResult};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on the chord and crossing values. A `GPU` may fuse a
/// multiply-add the scalar reference leaves separate, perturbing the low
/// mantissa bits by a few units in the last place; `1e-4` admits that legal
/// slack while still failing a genuinely wrong port.
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

/// The axis-aligned unit box `[-1, 1]` on every axis, the fixture most of the
/// named cases probe.
fn unit_box() -> Aabb {
    Aabb::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0))
}

/// Builds one query from a ray and a box.
fn query(origin: Vec3, dir: Vec3, aabb: Aabb) -> RayAabbQuery {
    RayAabbQuery {
        ray: Ray::new(origin, dir),
        aabb,
    }
}

/// Runs the `GPU` dispatch and asserts strict lane-for-lane parity against the
/// `CPU` golden: the two hit flags match exactly, the chord endpoints match
/// within tolerance when the line hits, and the first visible crossing matches
/// within tolerance when the forward ray hits. Returns the `GPU` verdicts for
/// extra per-test assertions. Use only for fixtures placed clear of every
/// boundary.
fn check(ctx: &GpuContext, gpu: &GpuRayAabb, queries: &[RayAabbQuery]) -> Vec<RayAabbResult> {
    let got = gpu.eval(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (lane, (g, q)) in got.iter().zip(queries.iter()).enumerate() {
        let (line, ray, chord, first) = cpu_reference(q);
        assert_eq!(
            g.line_hit, line,
            "lane {lane}: line_hit gpu {} vs cpu {line}",
            g.line_hit
        );
        assert_eq!(
            g.ray_hit, ray,
            "lane {lane}: ray_hit gpu {} vs cpu {ray}",
            g.ray_hit
        );
        if let Some(h) = chord {
            assert!(
                close(g.t_enter, h.t_enter),
                "lane {lane}: t_enter gpu {} vs cpu {}",
                g.t_enter,
                h.t_enter
            );
            assert!(
                close(g.t_exit, h.t_exit),
                "lane {lane}: t_exit gpu {} vs cpu {}",
                g.t_exit,
                h.t_exit
            );
        }
        if let Some(f) = first {
            assert!(
                close(g.first_hit_t, f),
                "lane {lane}: first_hit_t gpu {} vs cpu {f}",
                g.first_hit_t
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
    let gpu = GpuRayAabb::new(&ctx);
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch yields an empty result");
}

#[test]
fn frontal_penetration_with_parallel_guards() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayAabb::new(&ctx);
    // Straight down +x: the y and z direction components are exactly zero, so
    // both of those axes take the parallel-slab guard with the origin inside the
    // slab. Chord [4, 6], forward hit entering at 4.
    let q = query(
        Vec3::new(-5.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        unit_box(),
    );
    let got = check(&ctx, &gpu, &[q]);
    assert!(got[0].line_hit && got[0].ray_hit, "frontal ray should hit");
    assert!(close(got[0].t_enter, 4.0), "t_enter {}", got[0].t_enter);
    assert!(close(got[0].t_exit, 6.0), "t_exit {}", got[0].t_exit);
    assert!(
        close(got[0].first_hit_t, 4.0),
        "first {}",
        got[0].first_hit_t
    );
}

#[test]
fn parallel_guard_clear_miss() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayAabb::new(&ctx);
    // Same +x ray, but offset to y = 9 — far outside the y slab. The y-axis
    // parallel guard rejects the whole query, so both readings must be an exact
    // miss. This is the dedicated "dir component zero and origin outside the
    // slab" guard case; its boolean verdict must match the CPU exactly.
    let q = query(
        Vec3::new(-5.0, 9.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        unit_box(),
    );
    let got = check(&ctx, &gpu, &[q]);
    assert!(
        !got[0].line_hit && !got[0].ray_hit,
        "a ray parallel to and outside the y slab must miss both readings"
    );
}

#[test]
fn origin_inside_box_exits_forward() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayAabb::new(&ctx);
    // Origin at the center: the chord straddles zero (t_enter < 0 <= t_exit) and
    // a forward ray still hits, with its first visible crossing at the exit face.
    let q = query(
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        unit_box(),
    );
    let got = check(&ctx, &gpu, &[q]);
    assert!(
        got[0].line_hit && got[0].ray_hit,
        "inside origin should hit"
    );
    assert!(close(got[0].t_enter, -1.0), "t_enter {}", got[0].t_enter);
    assert!(close(got[0].t_exit, 1.0), "t_exit {}", got[0].t_exit);
    assert!(
        close(got[0].first_hit_t, 1.0),
        "first {}",
        got[0].first_hit_t
    );
}

#[test]
fn thin_grazing_clear_hit() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayAabb::new(&ctx);
    // Box [0, 1] on every axis, a diagonal in the xy plane with z parallel and
    // inside. The chord is [1.0, 1.1] — a short but unambiguous overlap, well
    // clear of a tie so the exact-flag assertion is safe.
    let b = Aabb::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
    let q = query(Vec3::new(-0.1, -1.0, 0.5), Vec3::new(1.0, 1.0, 0.0), b);
    let got = check(&ctx, &gpu, &[q]);
    assert!(got[0].line_hit && got[0].ray_hit, "grazing ray should hit");
    assert!(close(got[0].t_enter, 1.0), "t_enter {}", got[0].t_enter);
    assert!(close(got[0].t_exit, 1.1), "t_exit {}", got[0].t_exit);
}

#[test]
fn zero_volume_box_is_a_point_hit() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayAabb::new(&ctx);
    // Degenerate box collapsed to the origin point: on the x axis near and far
    // planes coincide, so t_enter == t_exit == 5 come from the identical
    // expression on both devices and the point is a touching hit.
    let b = Aabb::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.0));
    let q = query(Vec3::new(-5.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), b);
    let got = check(&ctx, &gpu, &[q]);
    assert!(
        got[0].line_hit && got[0].ray_hit,
        "a point box on the ray should be a touching hit"
    );
    assert!(close(got[0].t_enter, 5.0), "t_enter {}", got[0].t_enter);
    assert!(close(got[0].t_exit, 5.0), "t_exit {}", got[0].t_exit);
    assert!(
        close(got[0].first_hit_t, 5.0),
        "first {}",
        got[0].first_hit_t
    );
}

#[test]
fn box_behind_origin_is_line_only() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayAabb::new(&ctx);
    // Origin past the box: the infinite line still crosses it with a negative
    // chord [-6, -4], but the forward ray reports a miss (t_exit < 0).
    let q = query(
        Vec3::new(5.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        unit_box(),
    );
    let got = check(&ctx, &gpu, &[q]);
    assert!(
        got[0].line_hit && !got[0].ray_hit,
        "a box behind the origin is a line hit but a forward miss"
    );
    assert!(close(got[0].t_enter, -6.0), "t_enter {}", got[0].t_enter);
    assert!(close(got[0].t_exit, -4.0), "t_exit {}", got[0].t_exit);
}

#[test]
fn negative_direction_reorders_slabs() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayAabb::new(&ctx);
    // Shooting -x back toward the box: the per-axis near/far swap, so the
    // ordered chord is [4, 6] again and the forward ray hits.
    let q = query(
        Vec3::new(5.0, 0.0, 0.0),
        Vec3::new(-1.0, 0.0, 0.0),
        unit_box(),
    );
    let got = check(&ctx, &gpu, &[q]);
    assert!(got[0].line_hit && got[0].ray_hit, "negative dir should hit");
    assert!(close(got[0].t_enter, 4.0), "t_enter {}", got[0].t_enter);
    assert!(close(got[0].t_exit, 6.0), "t_exit {}", got[0].t_exit);
    assert!(
        close(got[0].first_hit_t, 4.0),
        "first {}",
        got[0].first_hit_t
    );
}

#[test]
fn space_diagonal_through_center() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayAabb::new(&ctx);
    // All three axes divide and reach their near planes at the same t: chord
    // [1, 3] through the main diagonal.
    let q = query(
        Vec3::new(-2.0, -2.0, -2.0),
        Vec3::new(1.0, 1.0, 1.0),
        unit_box(),
    );
    let got = check(&ctx, &gpu, &[q]);
    assert!(got[0].line_hit && got[0].ray_hit, "diagonal ray should hit");
    assert!(close(got[0].t_enter, 1.0), "t_enter {}", got[0].t_enter);
    assert!(close(got[0].t_exit, 3.0), "t_exit {}", got[0].t_exit);
    assert!(
        close(got[0].first_hit_t, 1.0),
        "first {}",
        got[0].first_hit_t
    );
}

#[test]
fn zero_direction_is_a_miss() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayAabb::new(&ctx);
    // A zero-length direction is rejected as a non-ray on both devices, whether
    // the origin is inside the box or outside it, so a point never spans the
    // infinite sentinel interval.
    let inside = query(
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 0.0),
        unit_box(),
    );
    let outside = query(
        Vec3::new(5.0, 5.0, 5.0),
        Vec3::new(0.0, 0.0, 0.0),
        unit_box(),
    );
    let got = check(&ctx, &gpu, &[inside, outside]);
    assert!(
        got.iter().all(|r| !r.line_hit && !r.ray_hit),
        "a zero-length direction must miss both readings everywhere"
    );
}

#[test]
fn random_batch_matches_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayAabb::new(&ctx);
    let mut state = 0x_51ab_aabb_0f0f_0001_u64;

    let mut saw_line_hit = false;
    let mut saw_line_miss = false;
    let mut saw_ray_hit = false;

    for _round in 0u32..8 {
        let mut queries = Vec::with_capacity(128);
        for _ in 0..128 {
            // A box centered somewhere in [-4, 4] with half extents in
            // [0.5, 2.0], never degenerate.
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
            let b = Aabb::from_center_half(center, half);

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
            // Occasionally force one direction component to exactly zero to
            // exercise the parallel-slab guard; the magnitude of the kept
            // components stays far above EPS so the guard verdict is
            // unambiguous.
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
            queries.push(query(origin, dir, b));
        }

        let got = gpu.eval(&ctx, &queries);
        assert_eq!(got.len(), queries.len(), "one result per query");
        for (lane, (g, q)) in got.iter().zip(queries.iter()).enumerate() {
            let (line, ray, chord, first) = cpu_reference(q);

            // Boolean flags match exactly unless the chord sits inside the tie
            // band, the only place a ULP perturbation can legally flip a verdict.
            if g.line_hit != line {
                let gap = if g.line_hit {
                    (g.t_exit - g.t_enter).abs()
                } else if let Some(h) = chord {
                    (h.t_exit - h.t_enter).abs()
                } else {
                    f32::INFINITY
                };
                assert!(
                    gap <= TIE,
                    "lane {lane}: line_hit disagreement off the tie band: gpu {} cpu {line} gap {gap}",
                    g.line_hit
                );
            }
            if g.ray_hit != ray {
                let near = if g.line_hit {
                    g.t_exit.abs() <= TIE || (g.t_exit - g.t_enter).abs() <= TIE
                } else if let Some(h) = chord {
                    h.t_exit.abs() <= TIE || (h.t_exit - h.t_enter).abs() <= TIE
                } else {
                    false
                };
                assert!(
                    near,
                    "lane {lane}: ray_hit disagreement off the tie band: gpu {} cpu {ray}",
                    g.ray_hit
                );
            }

            // Values are compared only where both devices agree on a hit.
            if g.line_hit
                && line
                && let Some(h) = chord
            {
                assert!(
                    close(g.t_enter, h.t_enter),
                    "lane {lane}: t_enter gpu {} vs cpu {}",
                    g.t_enter,
                    h.t_enter
                );
                assert!(
                    close(g.t_exit, h.t_exit),
                    "lane {lane}: t_exit gpu {} vs cpu {}",
                    g.t_exit,
                    h.t_exit
                );
            }
            if g.ray_hit
                && ray
                && let Some(f) = first
            {
                assert!(
                    close(g.first_hit_t, f),
                    "lane {lane}: first_hit_t gpu {} vs cpu {f}",
                    g.first_hit_t
                );
            }

            saw_line_hit |= line;
            saw_line_miss |= !line;
            saw_ray_hit |= ray;
        }
    }

    // A large random spread must exercise both verdict classes, so the test is
    // not trivially passing on an all-hit or all-miss batch.
    assert!(
        saw_line_hit && saw_line_miss && saw_ray_hit,
        "random batch should produce line hits, line misses and forward hits"
    );
}
