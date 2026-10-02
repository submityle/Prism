//! Real-device parity for the minimum-enclosing-ball primitive twin:
//! [`GpuWelzlMinSphere`](prism_volumetric_gpu::welzl_min_sphere::GpuWelzlMinSphere)
//! must reproduce the stateless building blocks of the `CPU` golden
//! [`welzl_min_sphere`](prism_render_architecture::particle::welzl_min_sphere) —
//! the [`Sphere2::contains`](prism_render_architecture::particle::welzl_min_sphere::Sphere2::contains)
//! and [`Sphere3::contains`](prism_render_architecture::particle::welzl_min_sphere::Sphere3::contains)
//! predicates and the closed-form diameter / circumcircle / circumsphere
//! constructions — across containment points clearly inside and outside, the
//! two-point diameters, acute triangles whose circumscribed circle is the
//! minimum enclosing ball, an acute tetrahedron whose circumscribed sphere is
//! the minimum enclosing ball, the collinear / coplanar degeneracies, a mixed
//! batch dispatched together, and randomized sweeps compared element for
//! element against the golden public API.
//!
//! The stateful Welzl recursion (fixed-seed shuffle, data-dependent recursion,
//! growing support set) that
//! [`min_enclosing_circle`](prism_render_architecture::particle::welzl_min_sphere::min_enclosing_circle)
//! and
//! [`min_enclosing_sphere`](prism_render_architecture::particle::welzl_min_sphere::min_enclosing_sphere)
//! run is deliberately **not** twinned; it remains on the host. These tests use
//! those host entry points purely as the oracle for the leaves: fixtures are
//! chosen so the minimum-enclosing ball of a tiny set *is* the closed-form
//! circum ball, which lets the host result stand in for the (private) golden
//! construction functions.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each query is a fixed, non-reorderable sequence of multiplies, adds and
//! divides, so `CPU` and `GPU` evaluate the same closed form in the same order.
//! The continuous center and radius are compared with `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`; the containment `bool` and the degeneracy classification
//! are pinned by exact equality.
//!
//! # Conditioning
//!
//! Every fixture is well away from a branch tie: containment points sit clearly
//! inside or outside the ball (rejection sampling keeps randomized points off
//! the boundary band), the circum fixtures are acute (every support point on
//! the ball boundary, determinant far above the compare epsilon), and the
//! degeneracies are exactly collinear / coplanar (determinant identically zero
//! on both devices). This keeps `CPU` and `GPU` on the same side of every
//! branch regardless of a few units in the last place of slack.
//!
//! Provenance: twinned from this repository's
//! [`welzl_min_sphere`](prism_render_architecture::particle::welzl_min_sphere);
//! no third-party engine source or derived code.

use prism_render_architecture::particle::welzl_min_sphere::{
    min_enclosing_circle, min_enclosing_sphere, Sphere2, Sphere3,
};
use prism_volumetric_gpu::welzl_min_sphere::{
    GpuWelzlMinSphere, WelzlMinSphereQuery, WelzlMinSphereResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on the continuous center / radius fields.
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

/// Unwraps a [`WelzlMinSphereResult::Circle`], panicking on any other variant.
fn as_circle(result: WelzlMinSphereResult) -> ([f32; 2], f32) {
    match result {
        WelzlMinSphereResult::Circle { center, radius } => (center, radius),
        other => panic!("expected a circle result, got {other:?}"),
    }
}

/// Unwraps a [`WelzlMinSphereResult::Ball`], panicking on any other variant.
fn as_ball(result: WelzlMinSphereResult) -> ([f32; 3], f32) {
    match result {
        WelzlMinSphereResult::Ball { center, radius } => (center, radius),
        other => panic!("expected a ball result, got {other:?}"),
    }
}

/// Unwraps a [`WelzlMinSphereResult::Contains`], panicking on any other variant.
fn as_contains(result: WelzlMinSphereResult) -> bool {
    match result {
        WelzlMinSphereResult::Contains(hit) => hit,
        other => panic!("expected a containment result, got {other:?}"),
    }
}

/// Pins a planar circle result against a golden [`Sphere2`].
fn pin_circle(idx: usize, got: WelzlMinSphereResult, want: Sphere2) {
    let (center, radius) = as_circle(got);
    assert!(
        close(center[0], want.center[0]) && close(center[1], want.center[1]),
        "circle {idx} center: gpu {center:?} vs cpu {:?}",
        want.center
    );
    assert!(
        close(radius, want.radius),
        "circle {idx} radius: gpu {radius} vs cpu {}",
        want.radius
    );
}

/// Pins a spatial ball result against a golden [`Sphere3`].
fn pin_ball(idx: usize, got: WelzlMinSphereResult, want: Sphere3) {
    let (center, radius) = as_ball(got);
    assert!(
        close(center[0], want.center[0])
            && close(center[1], want.center[1])
            && close(center[2], want.center[2]),
        "ball {idx} center: gpu {center:?} vs cpu {:?}",
        want.center
    );
    assert!(
        close(radius, want.radius),
        "ball {idx} radius: gpu {radius} vs cpu {}",
        want.radius
    );
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

/// A pseudo-random value in `[-span, span)` drawn from `state`.
fn signed(state: &mut u64, span: f32) -> f32 {
    (lcg(state) * 2.0 - 1.0) * span
}

/// A pseudo-random planar point with each component in `[-span, span)`.
fn rand2(state: &mut u64, span: f32) -> [f32; 2] {
    [signed(state, span), signed(state, span)]
}

/// A pseudo-random spatial point with each component in `[-span, span)`.
fn rand3(state: &mut u64, span: f32) -> [f32; 3] {
    [
        signed(state, span),
        signed(state, span),
        signed(state, span),
    ]
}

/// A pseudo-random unit direction, drawn by rejection sampling and normalized
/// with `sqrt` (allowed; not a transcendental). Rejects tiny / long draws so
/// the normalization is well conditioned.
fn rand_dir3(state: &mut u64) -> [f32; 3] {
    loop {
        let v = rand3(state, 1.0);
        let len_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
        if (0.2..=1.0).contains(&len_sq) {
            let inv = 1.0 / len_sq.sqrt();
            return [v[0] * inv, v[1] * inv, v[2] * inv];
        }
    }
}

/// Squared distance between two spatial points.
fn dist2_3(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

#[test]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWelzlMinSphere::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn contains_2d_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWelzlMinSphere::new(&ctx);
    // A fixed circle; points chosen clearly inside, clearly outside, and well
    // inside near the center. Each GPU bool must equal the golden predicate.
    let ball = Sphere2 {
        center: [1.0, -2.0],
        radius: 3.0,
    };
    let points = [[1.0, -2.0], [3.0, -2.0], [1.0, 2.0], [5.0, 5.0]];
    let queries: Vec<WelzlMinSphereQuery> = points
        .iter()
        .map(|&point| WelzlMinSphereQuery::Contains2 {
            center: ball.center,
            radius: ball.radius,
            point,
        })
        .collect();
    let got = gpu.eval(&ctx, &queries);
    for (idx, (&point, result)) in points.iter().zip(got).enumerate() {
        assert_eq!(
            as_contains(result),
            ball.contains(point),
            "contains2 {idx} point {point:?}"
        );
    }
}

#[test]
fn contains_3d_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWelzlMinSphere::new(&ctx);
    let ball = Sphere3 {
        center: [0.0, 1.0, -1.0],
        radius: 2.5,
    };
    let points = [
        [0.0, 1.0, -1.0],
        [2.0, 1.0, -1.0],
        [0.0, 3.0, 0.0],
        [5.0, 5.0, 5.0],
    ];
    let queries: Vec<WelzlMinSphereQuery> = points
        .iter()
        .map(|&point| WelzlMinSphereQuery::Contains3 {
            center: ball.center,
            radius: ball.radius,
            point,
        })
        .collect();
    let got = gpu.eval(&ctx, &queries);
    for (idx, (&point, result)) in points.iter().zip(got).enumerate() {
        assert_eq!(
            as_contains(result),
            ball.contains(point),
            "contains3 {idx} point {point:?}"
        );
    }
}

#[test]
fn circle_diameter_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWelzlMinSphere::new(&ctx);
    // The minimum enclosing circle of two points is their diameter circle.
    let a = [-2.0, 1.0];
    let b = [4.0, 5.0];
    let want = min_enclosing_circle(&[a, b]).expect("non-empty");
    let got = gpu.eval(&ctx, &[WelzlMinSphereQuery::CircleDiameter2 { a, b }]);
    pin_circle(0, got[0], want);
}

#[test]
fn circumcircle_2d_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWelzlMinSphere::new(&ctx);
    // An acute triangle: its circumscribed circle is the minimum enclosing
    // circle, so the host oracle equals the closed-form circumcircle.
    let a = [0.0, 0.0];
    let b = [4.0, 0.0];
    let c = [2.0, 3.0];
    let want = min_enclosing_circle(&[a, b, c]).expect("non-empty");
    let got = gpu.eval(&ctx, &[WelzlMinSphereQuery::Circumcircle2 { a, b, c }]);
    pin_circle(0, got[0], want);
}

#[test]
fn sphere_diameter_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWelzlMinSphere::new(&ctx);
    let a = [-1.0, 2.0, 0.5];
    let b = [3.0, -2.0, 4.5];
    let want = min_enclosing_sphere(&[a, b]).expect("non-empty");
    let got = gpu.eval(&ctx, &[WelzlMinSphereQuery::SphereDiameter3 { a, b }]);
    pin_ball(0, got[0], want);
}

#[test]
fn circumcircle_3d_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWelzlMinSphere::new(&ctx);
    // An acute triangle lifted off the z = 0 plane; its circumscribed circle
    // (center in the triangle plane) is the minimum enclosing sphere.
    let a = [0.0, 0.0, 1.0];
    let b = [4.0, 0.0, 1.0];
    let c = [2.0, 3.0, 1.0];
    let want = min_enclosing_sphere(&[a, b, c]).expect("non-empty");
    let got = gpu.eval(&ctx, &[WelzlMinSphereQuery::Circumcircle3 { a, b, c }]);
    pin_ball(0, got[0], want);
}

#[test]
fn circumsphere_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWelzlMinSphere::new(&ctx);
    // A regular tetrahedron (cube-diagonal vertices): acute, so its
    // circumscribed sphere is the minimum enclosing sphere, centered at the
    // origin with radius sqrt(3).
    let a = [1.0, 1.0, 1.0];
    let b = [1.0, -1.0, -1.0];
    let c = [-1.0, 1.0, -1.0];
    let d = [-1.0, -1.0, 1.0];
    let want = min_enclosing_sphere(&[a, b, c, d]).expect("non-empty");
    let got = gpu.eval(&ctx, &[WelzlMinSphereQuery::Circumsphere4 { a, b, c, d }]);
    pin_ball(0, got[0], want);
}

#[test]
fn collinear_circumcircle_is_degenerate() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWelzlMinSphere::new(&ctx);
    // Exactly collinear points: the determinant is identically zero on both
    // devices, so the degeneracy guard fires and reports a degenerate result.
    let a = [0.0, 0.0];
    let b = [1.0, 1.0];
    let c = [2.0, 2.0];
    let got = gpu.eval(&ctx, &[WelzlMinSphereQuery::Circumcircle2 { a, b, c }]);
    assert_eq!(got[0], WelzlMinSphereResult::Degenerate);
}

#[test]
fn coplanar_circumsphere_is_degenerate() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWelzlMinSphere::new(&ctx);
    // Four exactly coplanar points (all z = 0): the 3x3 determinant is zero, so
    // the circumsphere guard reports a degenerate result.
    let a = [0.0, 0.0, 0.0];
    let b = [1.0, 0.0, 0.0];
    let c = [0.0, 1.0, 0.0];
    let d = [1.0, 1.0, 0.0];
    let got = gpu.eval(&ctx, &[WelzlMinSphereQuery::Circumsphere4 { a, b, c, d }]);
    assert_eq!(got[0], WelzlMinSphereResult::Degenerate);
}

#[test]
fn mixed_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWelzlMinSphere::new(&ctx);
    // One batch mixing every op so the per-thread indexing and the contiguous
    // storage layout are both exercised, then pinned element for element.
    let ball2 = Sphere2 {
        center: [0.0, 0.0],
        radius: 2.0,
    };
    let ball3 = Sphere3 {
        center: [1.0, 1.0, 1.0],
        radius: 3.0,
    };
    let tri2 = ([0.0, 0.0], [4.0, 0.0], [2.0, 3.0]);
    let tri3 = ([0.0, 0.0, 2.0], [4.0, 0.0, 2.0], [2.0, 3.0, 2.0]);
    let tet = (
        [1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
    );

    let queries = vec![
        WelzlMinSphereQuery::Contains2 {
            center: ball2.center,
            radius: ball2.radius,
            point: [0.5, 0.5],
        },
        WelzlMinSphereQuery::Contains3 {
            center: ball3.center,
            radius: ball3.radius,
            point: [10.0, 0.0, 0.0],
        },
        WelzlMinSphereQuery::CircleDiameter2 {
            a: [-2.0, 1.0],
            b: [4.0, 5.0],
        },
        WelzlMinSphereQuery::Circumcircle2 {
            a: tri2.0,
            b: tri2.1,
            c: tri2.2,
        },
        WelzlMinSphereQuery::SphereDiameter3 {
            a: [-1.0, 2.0, 0.5],
            b: [3.0, -2.0, 4.5],
        },
        WelzlMinSphereQuery::Circumcircle3 {
            a: tri3.0,
            b: tri3.1,
            c: tri3.2,
        },
        WelzlMinSphereQuery::Circumsphere4 {
            a: tet.0,
            b: tet.1,
            c: tet.2,
            d: tet.3,
        },
    ];
    let got = gpu.eval(&ctx, &queries);
    assert_eq!(got.len(), queries.len());

    assert_eq!(as_contains(got[0]), ball2.contains([0.5, 0.5]));
    assert_eq!(as_contains(got[1]), ball3.contains([10.0, 0.0, 0.0]));
    pin_circle(
        2,
        got[2],
        min_enclosing_circle(&[[-2.0, 1.0], [4.0, 5.0]]).unwrap(),
    );
    pin_circle(
        3,
        got[3],
        min_enclosing_circle(&[tri2.0, tri2.1, tri2.2]).unwrap(),
    );
    pin_ball(
        4,
        got[4],
        min_enclosing_sphere(&[[-1.0, 2.0, 0.5], [3.0, -2.0, 4.5]]).unwrap(),
    );
    pin_ball(
        5,
        got[5],
        min_enclosing_sphere(&[tri3.0, tri3.1, tri3.2]).unwrap(),
    );
    pin_ball(
        6,
        got[6],
        min_enclosing_sphere(&[tet.0, tet.1, tet.2, tet.3]).unwrap(),
    );
}

#[test]
fn contains_random_sweep_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWelzlMinSphere::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;

    // Random spheres with random query points, each kept clearly off the
    // boundary band so the GPU and CPU classifications never straddle a tie.
    let mut queries: Vec<WelzlMinSphereQuery> = Vec::new();
    let mut wants: Vec<bool> = Vec::new();
    while queries.len() < 128 {
        let center = rand3(&mut state, 4.0);
        let radius = 1.0 + lcg(&mut state) * 4.0;
        let point = rand3(&mut state, 10.0);
        let d = dist2_3(point, center).sqrt();
        // Reject the boundary band |d - radius| < 0.1 so the bool is unambiguous.
        if (d - radius).abs() < 0.1 {
            continue;
        }
        let ball = Sphere3 { center, radius };
        wants.push(ball.contains(point));
        queries.push(WelzlMinSphereQuery::Contains3 {
            center,
            radius,
            point,
        });
    }

    let got = gpu.eval(&ctx, &queries);
    for (idx, (result, &want)) in got.iter().zip(wants.iter()).enumerate() {
        assert_eq!(as_contains(*result), want, "contains sweep {idx}");
    }
}

#[test]
fn circumcircle_2d_random_sweep_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWelzlMinSphere::new(&ctx);
    let mut state = 0x0bad_c0de_cafe_f00d_u64;

    // Rejection-sample acute triangles: a triangle whose minimum enclosing
    // circle passes through all three vertices (every squared distance to the
    // center matches the squared radius) is acute, so its circumcircle is that
    // minimum enclosing circle. That lets the host oracle stand in for the
    // closed-form circumcircle while staying far from the collinear crack.
    let mut queries: Vec<WelzlMinSphereQuery> = Vec::new();
    let mut wants: Vec<Sphere2> = Vec::new();
    while queries.len() < 64 {
        let a = rand2(&mut state, 6.0);
        let b = rand2(&mut state, 6.0);
        let c = rand2(&mut state, 6.0);
        let mec = min_enclosing_circle(&[a, b, c]).expect("non-empty");
        // Reject near-degenerate (tiny / huge circumradius) and non-acute
        // triangles (a vertex strictly inside the minimum enclosing circle).
        if !(0.5..=40.0).contains(&mec.radius) {
            continue;
        }
        // Keep only well-shaped acute triangles. Acuteness guarantees the host
        // minimum enclosing circle *is* the three-point circumcircle the kernel
        // builds: an obtuse triangle's MEC collapses to a two-point diameter (a
        // genuinely different circle), so a loose on-boundary band would feed
        // the kernel a mismatched oracle. The lower angle bound additionally
        // keeps the circumcenter solve well-conditioned — a near-right angle
        // vanishes the determinant and amplifies the last-place `CPU`/`GPU` gap
        // past the parity band even though each result is individually correct.
        // Require every interior angle in `[30, 85]` degrees by bounding the
        // normalized edge dot (the cosine) at each vertex.
        let cos_at = |p: [f32; 2], q: [f32; 2], r: [f32; 2]| -> f32 {
            let u = [q[0] - p[0], q[1] - p[1]];
            let v = [r[0] - p[0], r[1] - p[1]];
            let dot = u[0] * v[0] + u[1] * v[1];
            let lu = (u[0] * u[0] + u[1] * u[1]).sqrt();
            let lv = (v[0] * v[0] + v[1] * v[1]).sqrt();
            dot / (lu * lv)
        };
        // `cos 85 deg` (angle <= 85, acute margin) .. `cos 30 deg` (angle >= 30).
        let acute_band = 0.0872_f32..=0.8660_f32;
        let well_shaped = [cos_at(a, b, c), cos_at(b, a, c), cos_at(c, a, b)]
            .iter()
            .all(|cos| acute_band.contains(cos));
        if !well_shaped {
            continue;
        }
        wants.push(mec);
        queries.push(WelzlMinSphereQuery::Circumcircle2 { a, b, c });
    }

    let got = gpu.eval(&ctx, &queries);
    for (idx, (&result, &want)) in got.iter().zip(wants.iter()).enumerate() {
        pin_circle(idx, result, want);
    }
}

#[test]
fn circumsphere_random_sweep_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWelzlMinSphere::new(&ctx);
    let mut state = 0x51ab_cd00_1357_9bdf_u64;

    // Build four points on a known sphere (center + radius * unit direction),
    // then accept only when the host minimum enclosing sphere reproduces that
    // sphere: that means all four points surround the center and sit on the
    // boundary, so the circumsphere is the minimum enclosing sphere. The host
    // oracle then stands in for the closed-form circumsphere, far from the
    // coplanar crack.
    let mut queries: Vec<WelzlMinSphereQuery> = Vec::new();
    let mut wants: Vec<Sphere3> = Vec::new();
    while queries.len() < 64 {
        let center = rand3(&mut state, 3.0);
        let radius = 2.0 + lcg(&mut state) * 3.0;
        let on = |dir: [f32; 3]| {
            [
                center[0] + radius * dir[0],
                center[1] + radius * dir[1],
                center[2] + radius * dir[2],
            ]
        };
        let a = on(rand_dir3(&mut state));
        let b = on(rand_dir3(&mut state));
        let c = on(rand_dir3(&mut state));
        let d = on(rand_dir3(&mut state));
        let mes = min_enclosing_sphere(&[a, b, c, d]).expect("non-empty");
        // Accept only when the host sphere matches the generating sphere, i.e.
        // the circumsphere is the minimum enclosing sphere.
        if !(close(mes.radius, radius)
            && close(mes.center[0], center[0])
            && close(mes.center[1], center[1])
            && close(mes.center[2], center[2]))
        {
            continue;
        }
        wants.push(mes);
        queries.push(WelzlMinSphereQuery::Circumsphere4 { a, b, c, d });
    }

    let got = gpu.eval(&ctx, &queries);
    for (idx, (&result, &want)) in got.iter().zip(wants.iter()).enumerate() {
        pin_ball(idx, result, want);
    }
}
