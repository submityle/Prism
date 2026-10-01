//! Real-device parity for the sphere-`AABB` proximity twin:
//! [`GpuSphereAabb`](prism_volumetric_gpu::sphere_aabb::GpuSphereAabb) must
//! reproduce the `CPU` golden
//! [`query`](prism_render_architecture::particle::sphere_aabb::query) across an
//! empty batch, a clearly separated pair, a pair overlapping with the centre
//! outside, a sphere whose centre lies inside the box (closest point = centre),
//! a centre resting exactly on a face (the `distance ≈ 0` branch boundary), a
//! degenerate zero-volume box, a near-zero-radius sphere, and a large random
//! batch — all kept well clear of the inside/outside branch split and the
//! tangent boundary so the discrete flags are reproduced exactly.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full dispatch-and-
//! readback on any real device such as an Apple `M`-series `GPU`. The kernel is
//! portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The continuous fields (closest point, outside distance, separation,
//! penetration depth, separation normal) are compared with `abs_diff <= 1e-4`
//! or `rel_diff <= 1e-3` (relative floor `1e-6`) — loose enough to admit a
//! legal fused multiply-add contraction yet tight enough to fail a wrong port
//! (a swapped clamp, a dropped nearest-face candidate, a flipped normal sign, a
//! wrong penetration formula). The two discrete flags (`intersecting`,
//! `center_inside_box`) are compared for an *exact* match. Every fixture is
//! placed well clear of the `CMP_EPS` branch split and the `dist_sq ==
//! radius_sq` tangent, so a `GPU`'s fused multiply-add cannot flip a branch or
//! a flag.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::sphere_aabb`;
//! canonical closest-point sphere-`AABB` test; no third-party engine source or
//! derived code.

use prism_render_architecture::particle::sphere_aabb::{query, Aabb, Proximity, Sphere, Vec3};
use prism_volumetric_gpu::sphere_aabb::{GpuSphereAabb, SphereAabbQuery};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on the continuous fields.
const EPS_ABS: f32 = 1.0e-4;

/// Relative parity bound on the continuous fields.
const EPS_REL: f32 = 1.0e-3;

/// Floor keeping the relative-error denominator away from zero.
const REL_FLOOR: f32 = 1.0e-6;

/// Clearance margin keeping every random fixture away from the inside/outside
/// branch split, the tangent boundary and a nearest-face tie, so the discrete
/// flags and the chosen face are reproduced exactly on both paths.
const CLEARANCE: f32 = 0.05;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS_ABS || rel <= EPS_REL
}

/// Asserts two 3-vectors agree component-wise within [`close`].
fn close_vec(a: Vec3, b: Vec3, what: &str, idx: usize) {
    assert!(
        close(a.x, b.x) && close(a.y, b.y) && close(a.z, b.z),
        "{what} mismatch at query {idx}: gpu ({}, {}, {}), cpu ({}, {}, {})",
        a.x,
        a.y,
        a.z,
        b.x,
        b.y,
        b.z,
    );
}

/// Builds a sphere-`AABB` query.
fn q(center: [f32; 3], radius: f32, min: [f32; 3], max: [f32; 3]) -> SphereAabbQuery {
    SphereAabbQuery {
        sphere: Sphere::new(Vec3::new(center[0], center[1], center[2]), radius),
        aabb: Aabb::new(
            Vec3::new(min[0], min[1], min[2]),
            Vec3::new(max[0], max[1], max[2]),
        ),
    }
}

/// Asserts an exact-flag, tolerant-continuous match of one `GPU`
/// [`Proximity`] against the `CPU` golden.
fn assert_parity(got: Proximity, want: Proximity, idx: usize) {
    close_vec(got.closest_point, want.closest_point, "closest_point", idx);
    assert!(
        close(got.outside_distance, want.outside_distance),
        "outside_distance mismatch at query {idx}: gpu {}, cpu {}",
        got.outside_distance,
        want.outside_distance,
    );
    assert!(
        close(got.separation, want.separation),
        "separation mismatch at query {idx}: gpu {}, cpu {}",
        got.separation,
        want.separation,
    );
    assert!(
        close(got.penetration_depth, want.penetration_depth),
        "penetration_depth mismatch at query {idx}: gpu {}, cpu {}",
        got.penetration_depth,
        want.penetration_depth,
    );
    close_vec(got.normal, want.normal, "normal", idx);
    assert_eq!(
        got.intersecting, want.intersecting,
        "intersecting mismatch at query {idx}",
    );
    assert_eq!(
        got.center_inside_box, want.center_inside_box,
        "center_inside_box mismatch at query {idx}",
    );
}

/// Runs the `GPU` dispatch and asserts per-lane parity against the `CPU`
/// golden, returning the `GPU` proximities for extra assertions.
fn check(ctx: &GpuContext, gpu: &GpuSphereAabb, queries: &[SphereAabbQuery]) -> Vec<Proximity> {
    let got = gpu.eval(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one proximity per query");
    for (idx, (&g, qq)) in got.iter().zip(queries.iter()).enumerate() {
        let want = query(qq.sphere, qq.aabb);
        assert_parity(g, want, idx);
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

/// Whether `qq` is placed well clear of every decision boundary, so its
/// discrete flags and nearest face are unambiguous on both paths.
fn well_clear(qq: &SphereAabbQuery) -> bool {
    let aabb = qq.aabb;
    let center = qq.sphere.center;
    let radius = qq.sphere.radius.max(0.0);
    let closest = center.clamp(aabb.min, aabb.max);
    let delta = center.sub(closest);
    let outside_distance = delta.length();

    if aabb.contains_point(center) {
        // Inside branch: require a unique nearest face with a clear margin so a
        // fused multiply-add cannot flip the chosen face.
        let mut faces = [
            center.x - aabb.min.x,
            aabb.max.x - center.x,
            center.y - aabb.min.y,
            aabb.max.y - center.y,
            center.z - aabb.min.z,
            aabb.max.z - center.z,
        ];
        faces.sort_by(|a, b| a.partial_cmp(b).unwrap());
        return faces[0] > CLEARANCE && (faces[1] - faces[0]) > CLEARANCE;
    }

    // Outside branch: stay clear of the inside/outside split and the tangent.
    outside_distance > CLEARANCE && (outside_distance - radius).abs() > CLEARANCE
}

#[test]
fn empty_batch_is_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSphereAabb::new(&ctx);
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "empty batch should produce no proximities");
}

#[test]
fn clearly_separated_pair() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSphereAabb::new(&ctx);
    // Centre 3 units past the +x face, radius 1 => no overlap, separation 2.
    let queries = [q([5.0, 1.0, 1.0], 1.0, [0.0, 0.0, 0.0], [2.0, 2.0, 2.0])];
    let got = check(&ctx, &gpu, &queries);
    assert!(!got[0].intersecting, "far pair must not intersect");
    assert!(!got[0].center_inside_box, "far centre is outside the box");
}

#[test]
fn overlapping_center_outside() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSphereAabb::new(&ctx);
    // Centre 0.5 past the +x face, radius 1 => overlap, centre still outside.
    let queries = [q([2.5, 1.0, 1.0], 1.0, [0.0, 0.0, 0.0], [2.0, 2.0, 2.0])];
    let got = check(&ctx, &gpu, &queries);
    assert!(got[0].intersecting, "overlapping pair must intersect");
    assert!(!got[0].center_inside_box, "centre is just outside the box");
    assert!(
        got[0].penetration_depth > 0.0,
        "an overlapping outside centre has positive penetration",
    );
}

#[test]
fn center_inside_box_unique_face() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSphereAabb::new(&ctx);
    // Centre deep inside a [0,10]^3 box, nearest the -x face by a clear margin,
    // so the inside branch and its outward normal are unambiguous.
    let queries = [q([2.0, 5.0, 5.0], 0.5, [0.0, 0.0, 0.0], [10.0, 10.0, 10.0])];
    let got = check(&ctx, &gpu, &queries);
    assert!(got[0].center_inside_box, "centre lies inside the box");
    assert!(
        got[0].intersecting,
        "a centre inside the box always intersects"
    );
    close_vec(got[0].normal, Vec3::new(-1.0, 0.0, 0.0), "normal", 0);
    close_vec(
        got[0].closest_point,
        Vec3::new(2.0, 5.0, 5.0),
        "closest_point",
        0,
    );
}

#[test]
fn center_exactly_on_face_boundary() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSphereAabb::new(&ctx);
    // Centre resting exactly on the +x face: dist_sq is exactly 0, so both
    // paths take the inside branch (0 is not > CMP_EPS) deterministically, and
    // the inclusive contains_point reports the centre inside the box.
    let queries = [q([2.0, 1.0, 1.0], 0.5, [0.0, 0.0, 0.0], [2.0, 2.0, 2.0])];
    let got = check(&ctx, &gpu, &queries);
    assert!(
        got[0].intersecting,
        "a surface contact counts as intersecting"
    );
    assert!(
        got[0].center_inside_box,
        "an on-face centre is inclusively inside",
    );
}

#[test]
fn degenerate_zero_volume_box() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSphereAabb::new(&ctx);
    // A zero-volume box (min == max) is a point; the reference clamps against
    // it without special casing, so the twin must agree. The sphere clearly
    // encloses the point (distance 3, radius 5) so the flags are unambiguous.
    let queries = [q([3.0, 0.0, 0.0], 5.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])];
    let got = check(&ctx, &gpu, &queries);
    assert!(
        got[0].intersecting,
        "the sphere encloses the degenerate point"
    );
    assert!(
        !got[0].center_inside_box,
        "the centre is 3 units from the point"
    );
}

#[test]
fn near_zero_radius_sphere() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSphereAabb::new(&ctx);
    // A near-zero-radius sphere is effectively a point; placed clearly outside
    // (distance 3) it does not intersect, placed clearly inside it does.
    let queries = [
        q([5.0, 1.0, 1.0], 1.0e-6, [0.0, 0.0, 0.0], [2.0, 2.0, 2.0]),
        q([1.0, 1.0, 1.0], 1.0e-6, [0.0, 0.0, 0.0], [10.0, 10.0, 10.0]),
    ];
    let got = check(&ctx, &gpu, &queries);
    assert!(!got[0].intersecting, "a point sphere far outside misses");
    assert!(
        got[1].center_inside_box,
        "a point sphere inside is contained"
    );
}

#[test]
fn random_batch_matches_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSphereAabb::new(&ctx);
    let mut state = 0x_51b3_a2cf_0e17_0001_u64;

    for round in 0u32..8 {
        let mut queries: Vec<SphereAabbQuery> = Vec::with_capacity(128);
        while queries.len() < 128 {
            // A box of random position and size.
            let bx = lcg(&mut state) * 20.0 - 10.0;
            let by = lcg(&mut state) * 20.0 - 10.0;
            let bz = lcg(&mut state) * 20.0 - 10.0;
            let sx = lcg(&mut state) * 6.0 + 1.0;
            let sy = lcg(&mut state) * 6.0 + 1.0;
            let sz = lcg(&mut state) * 6.0 + 1.0;
            // Spread centres across inside, overlapping and far-outside regions.
            let cx = bx + lcg(&mut state) * 24.0 - 12.0;
            let cy = by + lcg(&mut state) * 24.0 - 12.0;
            let cz = bz + lcg(&mut state) * 24.0 - 12.0;
            let radius = lcg(&mut state) * 6.0;
            let candidate = q(
                [cx, cy, cz],
                radius,
                [bx, by, bz],
                [bx + sx, by + sy, bz + sz],
            );
            // Only keep fixtures well clear of every boundary so the discrete
            // flags and nearest face are reproduced exactly.
            if well_clear(&candidate) {
                queries.push(candidate);
            }
        }
        // `check` asserts per-lane parity against the CPU golden.
        let got = check(&ctx, &gpu, &queries);
        // Sanity: a large random spread should produce both intersecting and
        // non-intersecting lanes, so the test is not trivially passing.
        let any_hit = got.iter().any(|p| p.intersecting);
        let any_miss = got.iter().any(|p| !p.intersecting);
        assert!(
            any_hit && any_miss,
            "round {round}: expected a mix of verdicts"
        );
    }
}
