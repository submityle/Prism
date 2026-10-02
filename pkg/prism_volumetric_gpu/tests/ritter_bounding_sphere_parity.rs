//! Real-device parity for the Ritter approximate smallest-enclosing-sphere twin:
//! [`GpuRitterBoundingSphere`](prism_volumetric_gpu::ritter_bounding_sphere::GpuRitterBoundingSphere)
//! must reproduce the `CPU` golden
//! [`ritter_bounding_sphere`](prism_render_architecture::particle::ritter_bounding_sphere::ritter_bounding_sphere)
//! across an empty batch, an empty point-set, a single point, two points on an
//! axis and on a diagonal, a collinear run, a clearly dominant `z`-axis pair, a
//! clearly dominant `y`-axis pair, coincident points plus one outlier, an
//! asymmetric eight-vertex box, large coordinates, a mixed batch and a large
//! pseudo-random batch compared lane for lane.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Both sides run the identical seed-and-grow construction in the identical
//! per-point order, so they evaluate the same closed form in the same
//! associativity. They are not bit-exact: a `GPU` may fuse a multiply-add the
//! scalar reference leaves separate, perturbing the low mantissa bits. The
//! comparison therefore allows `abs_diff <= 1e-4` or `rel_diff <= 1e-3` on the
//! center and radius and asserts an *exact* match on the discrete `valid` flag.
//! The one branch whose flip can perturb the result beyond that tolerance is the
//! seed axis-pair selection (choosing a different extremal pair yields a
//! genuinely different seed); the named fixtures all have a clearly dominant
//! span, and the random batch rejects any point-set whose two largest spans are
//! within a small margin, so a legal `ULP` perturbation can never flip the
//! selected pair. A flipped grow-pass decision is self-stabilizing: its growth
//! magnitude is proportional to `dist - radius`, which vanishes at the boundary,
//! so it never moves the sphere beyond the tolerance.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::ritter_bounding_sphere`；
//! Jack Ritter's *Graphics Gems* bounding-sphere construction; no third-party
//! engine source or derived code.

use prism_render_architecture::particle::ritter_bounding_sphere::{Sphere, Vec3};
use prism_volumetric_gpu::ritter_bounding_sphere::{
    cpu_reference, GpuRitterBoundingSphere, GpuRitterSphere, RitterQuery, MAX_POINTS,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on the center and radius. A `GPU` may fuse a
/// multiply-add the scalar reference leaves separate, perturbing the low
/// mantissa bits; `1e-4` admits that legal slack while still failing a genuinely
/// wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in the
/// last place exceed the absolute floor.
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

/// Builds one query from a point slice.
fn query(points: &[Vec3]) -> RitterQuery {
    RitterQuery {
        points: points.to_vec(),
    }
}

/// Asserts strict lane-for-lane parity of one `GPU` sphere against the `CPU`
/// golden: the `valid` flag matches exactly, and when valid the center and
/// radius match within tolerance. Use only for fixtures with a clearly dominant
/// seed span.
fn assert_parity(lane: usize, got: &GpuRitterSphere, want: Option<Sphere>) {
    match want {
        None => assert_eq!(
            got.valid, 0,
            "lane {lane}: empty set should report valid 0, got {}",
            got.valid
        ),
        Some(s) => {
            assert_eq!(
                got.valid, 1,
                "lane {lane}: non-empty set should report valid 1, got {}",
                got.valid
            );
            assert!(
                close(got.center[0], s.center.x),
                "lane {lane}: center.x gpu {} vs cpu {}",
                got.center[0],
                s.center.x
            );
            assert!(
                close(got.center[1], s.center.y),
                "lane {lane}: center.y gpu {} vs cpu {}",
                got.center[1],
                s.center.y
            );
            assert!(
                close(got.center[2], s.center.z),
                "lane {lane}: center.z gpu {} vs cpu {}",
                got.center[2],
                s.center.z
            );
            assert!(
                close(got.radius, s.radius),
                "lane {lane}: radius gpu {} vs cpu {}",
                got.radius,
                s.radius
            );
        }
    }
}

/// Runs the `GPU` dispatch and asserts strict lane-for-lane parity against the
/// `CPU` golden for every query, returning the `GPU` spheres for extra per-test
/// assertions.
fn check(
    ctx: &GpuContext,
    gpu: &GpuRitterBoundingSphere,
    queries: &[RitterQuery],
) -> Vec<GpuRitterSphere> {
    let got = gpu.eval(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (lane, (g, q)) in got.iter().zip(queries.iter()).enumerate() {
        assert_parity(lane, g, cpu_reference(q));
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

/// Returns the three squared axis spans of a point-set, replicating the golden
/// seed pass so the random harness can reject seed-selection near-ties.
fn axis_spans(points: &[Vec3]) -> (f32, f32, f32) {
    let first = points[0];
    let mut min_x = first;
    let mut max_x = first;
    let mut min_y = first;
    let mut max_y = first;
    let mut min_z = first;
    let mut max_z = first;
    for &p in points {
        if p.x < min_x.x {
            min_x = p;
        }
        if p.x > max_x.x {
            max_x = p;
        }
        if p.y < min_y.y {
            min_y = p;
        }
        if p.y > max_y.y {
            max_y = p;
        }
        if p.z < min_z.z {
            min_z = p;
        }
        if p.z > max_z.z {
            max_z = p;
        }
    }
    (
        max_x.sub(min_x).length_squared(),
        max_y.sub(min_y).length_squared(),
        max_z.sub(min_z).length_squared(),
    )
}

/// Whether the point-set has a clearly dominant seed span, so a `ULP`-scale
/// perturbation cannot flip the selected extremal pair between `CPU` and `GPU`.
/// A degenerate (all-coincident) set has every span zero and seeds any axis to
/// the same zero-radius sphere, so it is always stable.
fn clear_dominant(points: &[Vec3]) -> bool {
    if points.len() < 2 {
        return true;
    }
    let (sx, sy, sz) = axis_spans(points);
    let mut spans = [sx, sy, sz];
    spans.sort_by(|a, b| a.partial_cmp(b).expect("spans are finite"));
    let top = spans[2];
    let second = spans[1];
    // All spans tiny: coincident cloud, zero-radius sphere, no ambiguity.
    top <= REL_FLOOR || top >= second * 1.05 + 1.0e-3
}

#[test]
fn empty_batch_is_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRitterBoundingSphere::new(&ctx);
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch yields an empty result");
}

#[test]
fn empty_point_set_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRitterBoundingSphere::new(&ctx);
    let q = query(&[]);
    let got = check(&ctx, &gpu, &[q]);
    assert_eq!(got[0].valid, 0, "an empty point-set is invalid");
}

#[test]
fn single_point_centers_on_it() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRitterBoundingSphere::new(&ctx);
    let q = query(&[Vec3::new(3.0, -2.0, 7.0)]);
    let got = check(&ctx, &gpu, &[q]);
    assert_eq!(got[0].valid, 1, "a single point is valid");
    assert!(
        got[0].radius <= EPS,
        "radius {} should be ~0",
        got[0].radius
    );
    assert!(
        close(got[0].center[0], 3.0),
        "center.x {}",
        got[0].center[0]
    );
    assert!(
        close(got[0].center[1], -2.0),
        "center.y {}",
        got[0].center[1]
    );
    assert!(
        close(got[0].center[2], 7.0),
        "center.z {}",
        got[0].center[2]
    );
}

#[test]
fn two_points_axis_radius_half_distance() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRitterBoundingSphere::new(&ctx);
    // Distance 10 along the (0, 6, 8) diagonal; radius ~5, center at midpoint.
    let q = query(&[Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 6.0, 8.0)]);
    let got = check(&ctx, &gpu, &[q]);
    assert!(close(got[0].radius, 5.0), "radius {}", got[0].radius);
    assert!(
        close(got[0].center[1], 3.0),
        "center.y {}",
        got[0].center[1]
    );
    assert!(
        close(got[0].center[2], 4.0),
        "center.z {}",
        got[0].center[2]
    );
}

#[test]
fn two_points_diagonal_center_and_radius() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRitterBoundingSphere::new(&ctx);
    // Distance 3 along (2, 2, 1); radius ~1.5, center (1, 1, 0.5).
    let q = query(&[Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 2.0, 1.0)]);
    let got = check(&ctx, &gpu, &[q]);
    assert!(close(got[0].radius, 1.5), "radius {}", got[0].radius);
    assert!(
        close(got[0].center[0], 1.0),
        "center.x {}",
        got[0].center[0]
    );
    assert!(
        close(got[0].center[1], 1.0),
        "center.y {}",
        got[0].center[1]
    );
    assert!(
        close(got[0].center[2], 0.5),
        "center.z {}",
        got[0].center[2]
    );
}

#[test]
fn collinear_x_center_at_midpoint() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRitterBoundingSphere::new(&ctx);
    // Clearly dominant x span; center at the extremes' midpoint (1, 0, 0).
    let q = query(&[
        Vec3::new(-2.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(4.0, 0.0, 0.0),
    ]);
    let got = check(&ctx, &gpu, &[q]);
    assert!(close(got[0].radius, 3.0), "radius {}", got[0].radius);
    assert!(
        close(got[0].center[0], 1.0),
        "center.x {}",
        got[0].center[0]
    );
}

#[test]
fn dominant_z_axis_pair() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRitterBoundingSphere::new(&ctx);
    // The largest spread is clearly along z, so the seed must use that pair.
    let q = query(&[
        Vec3::new(0.0, 0.0, -10.0),
        Vec3::new(1.0, 1.0, 0.0),
        Vec3::new(-1.0, -1.0, 0.0),
        Vec3::new(0.0, 0.0, 10.0),
    ]);
    let got = check(&ctx, &gpu, &[q]);
    assert!(got[0].radius + EPS >= 10.0, "radius {}", got[0].radius);
}

#[test]
fn dominant_y_axis_pair() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRitterBoundingSphere::new(&ctx);
    let q = query(&[
        Vec3::new(0.0, -8.0, 0.0),
        Vec3::new(1.0, 0.0, 1.0),
        Vec3::new(-1.0, 0.0, -1.0),
        Vec3::new(0.0, 8.0, 0.0),
    ]);
    let got = check(&ctx, &gpu, &[q]);
    assert!(got[0].radius + EPS >= 8.0, "radius {}", got[0].radius);
}

#[test]
fn coincident_plus_one_outlier() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRitterBoundingSphere::new(&ctx);
    let base = Vec3::new(2.0, 2.0, 2.0);
    let q = query(&[base, base, base, Vec3::new(2.0, 2.0, 8.0)]);
    let got = check(&ctx, &gpu, &[q]);
    assert!(close(got[0].radius, 3.0), "radius {}", got[0].radius);
}

#[test]
fn asymmetric_box_eight_vertices() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRitterBoundingSphere::new(&ctx);
    // Half extents 1, 2, 4: the z axis clearly dominates, so there is no
    // seed-selection tie even though all eight corners are present.
    let q = query(&[
        Vec3::new(-1.0, -2.0, -4.0),
        Vec3::new(1.0, -2.0, -4.0),
        Vec3::new(-1.0, 2.0, -4.0),
        Vec3::new(1.0, 2.0, -4.0),
        Vec3::new(-1.0, -2.0, 4.0),
        Vec3::new(1.0, -2.0, 4.0),
        Vec3::new(-1.0, 2.0, 4.0),
        Vec3::new(1.0, 2.0, 4.0),
    ]);
    assert!(clear_dominant(&q.points), "fixture must be seed-stable");
    check(&ctx, &gpu, &[q]);
}

#[test]
fn large_coordinates() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRitterBoundingSphere::new(&ctx);
    let q = query(&[
        Vec3::new(-6.0e5, 10.0, -20.0),
        Vec3::new(6.0e5, 25.0, 15.0),
        Vec3::new(1.0e5, -30.0, 40.0),
    ]);
    assert!(clear_dominant(&q.points), "fixture must be seed-stable");
    check(&ctx, &gpu, &[q]);
}

#[test]
fn mixed_batch_in_one_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRitterBoundingSphere::new(&ctx);
    let queries = [
        query(&[]),
        query(&[Vec3::new(1.0, 2.0, 3.0)]),
        query(&[Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 6.0, 8.0)]),
        query(&[
            Vec3::new(0.0, 0.0, -10.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 10.0),
        ]),
    ];
    let got = check(&ctx, &gpu, &queries);
    assert_eq!(got[0].valid, 0, "empty lane is invalid");
    assert_eq!(got[1].valid, 1, "single-point lane is valid");
    assert_eq!(got[2].valid, 1, "two-point lane is valid");
    assert_eq!(got[3].valid, 1, "z-dominant lane is valid");
}

#[test]
fn random_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRitterBoundingSphere::new(&ctx);

    let mut state = 0x5eed_1234_abcd_0001_u64;
    let mut queries: Vec<RitterQuery> = Vec::new();
    let mut saw_single = false;
    let mut saw_many = false;
    while queries.len() < 96 {
        // Point count in [1, MAX_POINTS].
        let n = 1 + ((lcg(&mut state) * MAX_POINTS as f32) as usize).min(MAX_POINTS - 1);
        let mut points = Vec::with_capacity(n);
        for _ in 0..n {
            points.push(Vec3::new(
                lcg(&mut state) * 20.0 - 10.0,
                lcg(&mut state) * 20.0 - 10.0,
                lcg(&mut state) * 20.0 - 10.0,
            ));
        }
        // Reject seed-selection near-ties so a ULP flip cannot pick a different
        // extremal pair and produce a genuinely different seed.
        if !clear_dominant(&points) {
            continue;
        }
        if n == 1 {
            saw_single = true;
        }
        if n >= 8 {
            saw_many = true;
        }
        queries.push(RitterQuery { points });
    }

    let got = gpu.eval(&ctx, &queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (lane, (g, q)) in got.iter().zip(queries.iter()).enumerate() {
        assert_parity(lane, g, cpu_reference(q));
    }

    assert!(
        saw_single && saw_many,
        "random batch should exercise single-point and many-point sets"
    );
}
