//! Real-device parity for the analytic `ray`-disk / `ray`-annulus twin:
//! [`GpuRayDisk`](prism_volumetric_gpu::ray_disk::GpuRayDisk) must reproduce the
//! `CPU` golden
//! [`ray_disk`](prism_render_architecture::particle::ray_disk) across a
//! front-face hit (a ray striking the side the normal points toward), a
//! back-face hit (a ray striking the opposite side), a parallel miss (a
//! direction lying in the plane so `dir·n` vanishes), a behind-origin miss (the
//! plane crossing has `t < 0`), near-rim hits that land clearly inside or
//! clearly outside the inner / outer radius, degenerate disk and band
//! primitives that never intersect, and a randomized batch of clearly
//! conditioned disk and annulus hits compared element-for-element.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each query is a fixed, non-reorderable sequence of multiplies, adds, one
//! guarded division and at most one `sqrt`, so `CPU` and `GPU` evaluate the same
//! closed form in the same order. They are not bit-exact: a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place. The comparison therefore
//! allows `abs_diff <= 1e-4` or `rel_diff <= 1e-3` on the `f32` fields while
//! pinning the hit flag and the face code exactly.
//!
//! # Conditioning
//!
//! Every fixture is deliberately well away from each branch boundary: a hit's
//! squared radial distance differs from `radius²` (and from `inner²`) by at
//! least `MARGIN`, so `CPU` and `GPU` fold the same inside / outside verdict; the
//! parallel miss uses a direction exactly in the plane (`dir·n = 0`, far below
//! the epsilon); the behind-origin miss has `t` clearly negative; and no fixture
//! is tuned to sit on a rim. This keeps `CPU` and `GPU` on the same side of
//! every branch regardless of a few units in the last place of slack.
//!
//! Provenance: twinned from this repository's
//! [`ray_disk`](prism_render_architecture::particle::ray_disk); no third-party
//! engine source or derived code.

use prism_render_architecture::particle::ray_disk::{Annulus, Disk, HitFace, Ray, Vec3};
use prism_volumetric_gpu::ray_disk::{
    GpuRayDisk, GpuRayDiskHit, RayDiskQuery, RayDiskTarget, FACE_BACK, FACE_FRONT,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound. A `GPU` may fuse a multiply-add the scalar reference
/// leaves separate, perturbing the low mantissa bits by a few units in the last
/// place; `1e-4` admits that legal slack while still failing a wrong port.
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

/// Asserts a three-lane array agrees channel-for-channel within the parity
/// bound against a reference [`Vec3`].
fn close_vec(label: &str, idx: usize, got: [f32; 3], want: Vec3) {
    assert!(
        close(got[0], want.x) && close(got[1], want.y) && close(got[2], want.z),
        "query {idx} {label}: gpu ({}, {}, {}) vs cpu ({}, {}, {})",
        got[0],
        got[1],
        got[2],
        want.x,
        want.y,
        want.z
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

/// A pseudo-random vector with each component in `[-span, span)`.
fn rand_vec(state: &mut u64, span: f32) -> Vec3 {
    Vec3::new(
        signed(state, span),
        signed(state, span),
        signed(state, span),
    )
}

/// A unit vector drawn from `state`, retried until it is comfortably non-zero so
/// the normalization is well conditioned.
fn rand_unit(state: &mut u64) -> Vec3 {
    loop {
        let v = rand_vec(state, 1.0);
        if v.length_squared() > 0.2 {
            return v.normalize_or_zero();
        }
    }
}

/// Returns a unit vector lying in the plane whose normal is `normal`, built by
/// projecting a random vector onto the plane and retrying until the residual is
/// comfortably non-zero so the normalization stays well conditioned.
fn in_plane_unit(state: &mut u64, normal: Vec3) -> Vec3 {
    loop {
        let w = rand_vec(state, 1.0);
        let residual = w.minus(normal.scale(w.dot(normal)));
        if residual.length_squared() > 0.2 {
            return residual.normalize_or_zero();
        }
    }
}

/// The canonical unit-`z` disk at the origin with radius `2`, used by the
/// hand-built fixtures.
fn unit_disk() -> Disk {
    Disk::new(Vec3::ZERO, Vec3::new(0.0, 0.0, 1.0), 2.0)
}

/// The canonical unit-`z` annulus at the origin with inner `1`, outer `2`.
fn unit_annulus() -> Annulus {
    Annulus::new(Vec3::ZERO, Vec3::new(0.0, 0.0, 1.0), 1.0, 2.0)
}

/// Encodes the reference [`HitFace`] into the kernel's discrete face code.
fn face_code(face: HitFace) -> u32 {
    match face {
        HitFace::Front => FACE_FRONT,
        HitFace::Back => FACE_BACK,
    }
}

/// Pins one `GPU` result against the `CPU` golden for `query`: the hit flag and
/// the face code must agree exactly, and the ray parameter, point and normal
/// must agree within the parity bound.
fn pin(idx: usize, query: &RayDiskQuery, got: &GpuRayDiskHit) {
    let want = match query.target {
        RayDiskTarget::Disk(d) => d.intersect(query.ray),
        RayDiskTarget::Annulus(a) => a.intersect(query.ray),
    };
    match want {
        Some(w) => {
            assert_eq!(got.hit, 1, "query {idx}: expected a hit");
            assert_eq!(
                got.face,
                face_code(w.face),
                "query {idx}: face code must match the reference"
            );
            assert!(
                close(got.t, w.t),
                "query {idx} t: gpu {} vs cpu {}",
                got.t,
                w.t
            );
            close_vec("point", idx, got.point, w.point);
            close_vec("normal", idx, got.normal, w.normal);
        }
        None => {
            assert_eq!(got.hit, 0, "query {idx}: expected a miss");
        }
    }
}

/// Dispatches `queries` on the `GPU` and pins every result against the reference.
fn check(ctx: &GpuContext, gpu: &GpuRayDisk, queries: &[RayDiskQuery]) {
    let got = gpu.eval(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (query, result)) in queries.iter().zip(got.iter()).enumerate() {
        pin(idx, query, result);
    }
}

/// Builds a clearly-conditioned disk hit: a random disk whose plane the ray
/// crosses at a point whose radial distance is well inside the radius, with the
/// direction dominated by the normal component so `dir·n` is comfortably away
/// from zero. The sign of the normal component alternates the struck face.
fn disk_hit_query(state: &mut u64) -> RayDiskQuery {
    let normal = rand_unit(state);
    let center = rand_vec(state, 4.0);
    let radius = 1.5 + lcg(state) * 2.0;

    let radial = radius * (0.2 + lcg(state) * 0.5);
    let in_dir = in_plane_unit(state, normal);
    let target = center.plus(in_dir.scale(radial));

    // A direction dominated by the (signed) normal component, plus a small
    // in-plane tilt, so the crossing is well conditioned and oblique.
    let normal_sign = if lcg(state) < 0.5 { 1.0 } else { -1.0 };
    let tilt = in_plane_unit(state, normal);
    let dir = normal
        .scale(normal_sign * (0.6 + lcg(state) * 0.4))
        .plus(tilt.scale(signed(state, 0.3)))
        .normalize_or_zero();
    let dist = 3.0 + lcg(state) * 4.0;
    let origin = target.minus(dir.scale(dist));

    RayDiskQuery::disk(Ray::new(origin, dir), Disk::new(center, normal, radius))
}

/// Builds a clearly-conditioned annulus hit: like [`disk_hit_query`] but with
/// the crossing's radial distance landing well inside the `[inner, outer]` band.
fn annulus_hit_query(state: &mut u64) -> RayDiskQuery {
    let normal = rand_unit(state);
    let center = rand_vec(state, 4.0);
    let inner = 1.0 + lcg(state) * 1.5;
    let outer = inner + 1.5 + lcg(state) * 2.0;

    let band = outer - inner;
    let radial = inner + band * (0.25 + lcg(state) * 0.5);
    let in_dir = in_plane_unit(state, normal);
    let target = center.plus(in_dir.scale(radial));

    let normal_sign = if lcg(state) < 0.5 { 1.0 } else { -1.0 };
    let tilt = in_plane_unit(state, normal);
    let dir = normal
        .scale(normal_sign * (0.6 + lcg(state) * 0.4))
        .plus(tilt.scale(signed(state, 0.3)))
        .normalize_or_zero();
    let dist = 3.0 + lcg(state) * 4.0;
    let origin = target.minus(dir.scale(dist));

    RayDiskQuery::annulus(
        Ray::new(origin, dir),
        Annulus::new(center, normal, inner, outer),
    )
}

#[test]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayDisk::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn disk_front_face_hit_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayDisk::new(&ctx);
    // A ray from +z travelling -z strikes the front face (dir·n < 0) at radius
    // 0.5, clearly inside radius 2.
    let query = RayDiskQuery::disk(
        Ray::new_normalized(Vec3::new(0.5, 0.0, 5.0), Vec3::new(0.0, 0.0, -1.0)),
        unit_disk(),
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn disk_back_face_hit_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayDisk::new(&ctx);
    // A ray from -z travelling +z strikes the back face (dir·n > 0) at radius
    // 0.5, clearly inside radius 2.
    let query = RayDiskQuery::disk(
        Ray::new_normalized(Vec3::new(0.5, 0.0, -5.0), Vec3::new(0.0, 0.0, 1.0)),
        unit_disk(),
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn disk_parallel_direction_misses() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayDisk::new(&ctx);
    // A direction lying in the disk plane makes dir·n exactly zero, far below the
    // epsilon, so the reference rejects it as a non-intersection.
    let query = RayDiskQuery::disk(
        Ray::new_normalized(Vec3::new(0.0, 0.0, 1.0), Vec3::new(1.0, 0.0, 0.0)),
        unit_disk(),
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn disk_behind_origin_misses() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayDisk::new(&ctx);
    // The plane crossing is at t = -5, clearly behind the origin, so the forward
    // half-line never reaches the disk.
    let query = RayDiskQuery::disk(
        Ray::new_normalized(Vec3::new(0.0, 0.0, 5.0), Vec3::new(0.0, 0.0, 1.0)),
        unit_disk(),
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn disk_just_inside_rim_hits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayDisk::new(&ctx);
    // Radial distance 1.9 (squared 3.61) is clearly inside radius 2 (squared 4),
    // a margin of 0.39 in squared space.
    let query = RayDiskQuery::disk(
        Ray::new_normalized(Vec3::new(1.9, 0.0, 5.0), Vec3::new(0.0, 0.0, -1.0)),
        unit_disk(),
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn disk_just_outside_rim_misses() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayDisk::new(&ctx);
    // Radial distance 2.1 (squared 4.41) is clearly outside radius 2 (squared 4),
    // a margin of 0.41 in squared space, so it is a miss.
    let query = RayDiskQuery::disk(
        Ray::new_normalized(Vec3::new(2.1, 0.0, 5.0), Vec3::new(0.0, 0.0, -1.0)),
        unit_disk(),
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn disk_degenerate_never_hits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayDisk::new(&ctx);
    // A zero-radius disk is degenerate: it encloses no area and never reports a
    // hit, matching the reference is_degenerate short-circuit.
    let query = RayDiskQuery::disk(
        Ray::new_normalized(Vec3::new(0.0, 0.0, 5.0), Vec3::new(0.0, 0.0, -1.0)),
        Disk::new(Vec3::ZERO, Vec3::new(0.0, 0.0, 1.0), 0.0),
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn annulus_band_hit_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayDisk::new(&ctx);
    // Radial distance 1.5 (squared 2.25) lands clearly inside the band
    // [inner 1, outer 2] (squared [1, 4]), striking the front face.
    let query = RayDiskQuery::annulus(
        Ray::new_normalized(Vec3::new(1.5, 0.0, 5.0), Vec3::new(0.0, 0.0, -1.0)),
        unit_annulus(),
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn annulus_inside_hole_misses() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayDisk::new(&ctx);
    // Radial distance 0.5 (squared 0.25) lands clearly inside the hole
    // (inner squared 1), a margin of 0.75, so it misses the band.
    let query = RayDiskQuery::annulus(
        Ray::new_normalized(Vec3::new(0.5, 0.0, 5.0), Vec3::new(0.0, 0.0, -1.0)),
        unit_annulus(),
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn annulus_outside_rim_misses() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayDisk::new(&ctx);
    // Radial distance 2.2 (squared 4.84) lands clearly outside the outer radius
    // (squared 4), a margin of 0.84, so it misses the band.
    let query = RayDiskQuery::annulus(
        Ray::new_normalized(Vec3::new(2.2, 0.0, 5.0), Vec3::new(0.0, 0.0, -1.0)),
        unit_annulus(),
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn annulus_near_inner_edge_hits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayDisk::new(&ctx);
    // Radial distance 1.2 (squared 1.44) is clearly inside the band yet near the
    // inner rim, a margin of 0.44 above inner squared 1.
    let query = RayDiskQuery::annulus(
        Ray::new_normalized(Vec3::new(1.2, 0.0, -5.0), Vec3::new(0.0, 0.0, 1.0)),
        unit_annulus(),
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn annulus_degenerate_band_never_hits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayDisk::new(&ctx);
    // The outer radius does not exceed the inner one, so there is no band: the
    // annulus is degenerate and never reports a hit.
    let query = RayDiskQuery::annulus(
        Ray::new_normalized(Vec3::new(0.0, 0.0, 5.0), Vec3::new(0.0, 0.0, -1.0)),
        Annulus::new(Vec3::ZERO, Vec3::new(0.0, 0.0, 1.0), 2.0, 2.0),
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn mixed_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayDisk::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    // One batch mixing the deterministic fixtures with many random disk and
    // annulus hits, dispatched together so the per-thread indexing and the
    // contiguous storage layout are both exercised, then pinned element-for-
    // element.
    let mut queries = vec![
        RayDiskQuery::disk(
            Ray::new_normalized(Vec3::new(0.5, 0.0, 5.0), Vec3::new(0.0, 0.0, -1.0)),
            unit_disk(),
        ),
        RayDiskQuery::disk(
            Ray::new_normalized(Vec3::new(0.5, 0.0, -5.0), Vec3::new(0.0, 0.0, 1.0)),
            unit_disk(),
        ),
        RayDiskQuery::disk(
            Ray::new_normalized(Vec3::new(2.1, 0.0, 5.0), Vec3::new(0.0, 0.0, -1.0)),
            unit_disk(),
        ),
        RayDiskQuery::annulus(
            Ray::new_normalized(Vec3::new(1.5, 0.0, 5.0), Vec3::new(0.0, 0.0, -1.0)),
            unit_annulus(),
        ),
        RayDiskQuery::annulus(
            Ray::new_normalized(Vec3::new(0.5, 0.0, 5.0), Vec3::new(0.0, 0.0, -1.0)),
            unit_annulus(),
        ),
    ];
    for _ in 0..24 {
        queries.push(disk_hit_query(&mut state));
        queries.push(annulus_hit_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn many_disk_hits_sweep_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayDisk::new(&ctx);
    let mut state = 0x0bad_c0de_cafe_f00d_u64;
    // A larger sweep of clearly-conditioned disk hits (several workgroups' worth)
    // pins the hit parameter, point and oriented normal across many random
    // geometries.
    let queries: Vec<RayDiskQuery> = (0..200).map(|_| disk_hit_query(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}

#[test]
fn many_annulus_hits_sweep_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayDisk::new(&ctx);
    let mut state = 0x00c0_ffee_1234_5678_u64;
    // A larger sweep of clearly-conditioned annulus hits pins the same fields
    // across many random band geometries.
    let queries: Vec<RayDiskQuery> = (0..200).map(|_| annulus_hit_query(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}
