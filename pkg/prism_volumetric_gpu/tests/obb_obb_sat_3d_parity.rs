//! Real-device parity for the 3D oriented-bounding-box vs oriented-bounding-box
//! `SAT` twin:
//! [`GpuObbSat3d`](prism_volumetric_gpu::obb_obb_sat_3d::GpuObbSat3d) must
//! reproduce the `CPU` golden
//! [`intersects`](prism_render_architecture::particle::obb_obb_sat_3d::intersects)
//! across an empty batch, axis-aligned overlap and separation, an exact face
//! contact, a hair-thin gap, full containment, boxes rotated `45` degrees about
//! `z` (overlapping and separated), a thin slab, a corner overlap, a diagonal
//! far separation and a large pseudo-random batch compared lane for lane.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The twinned contract is a single per-pair boolean, so parity is an *exact*
//! match on the intersect flag. A `GPU` may fuse a multiply-add the scalar
//! reference leaves separate, perturbing the low mantissa bits by a few units in
//! the last place, which can flip the `>` separation verdict only for a pair
//! sitting on the contact boundary. The named fixtures are all placed clear of
//! that boundary and assert the flag unconditionally; the random batch
//! reject-samples away from the boundary (keeping only pairs whose deciding axis
//! leaves a margin wider than the tie band) so it, too, asserts exact equality.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::obb_obb_sat_3d`；
//! standard Gottschalk `OBBTree` fifteen-axis `SAT` overlap test; no third-party
//! engine source or derived code.

use prism_render_architecture::particle::obb_obb_sat_3d::{
    Obb, Vec3, CONTACT_EPS, DEGENERATE_AXIS_EPS,
};
use prism_volumetric_gpu::obb_obb_sat_3d::{
    cpu_reference, GpuObbSat3d, ObbSat3dQuery, ObbSat3dResult,
};
use prism_volumetric_gpu::GpuContext;

/// A `45`-degree sine/cosine literal (no transcendental call at runtime),
/// matching the golden test fixtures.
const HALF_SQRT2: f32 = 0.707_106_77;

/// Half-width of the tie band around the contact boundary inside which a `>`
/// separation verdict can legally flip under a `ULP`-scale perturbation. The
/// random batch keeps only pairs whose deciding axis leaves a margin wider than
/// this, so every retained pair asserts an exact flag.
const TIE: f32 = 1.0e-2;

/// Builds one query from an ordered pair of boxes.
fn query(a: Obb, b: Obb) -> ObbSat3dQuery {
    ObbSat3dQuery { a, b }
}

/// A unit cube (`half = 1`) at `center`, axis-aligned.
fn unit_cube_at(center: Vec3) -> Obb {
    Obb::axis_aligned(center, 1.0, 1.0, 1.0)
}

/// A box rotated `45` degrees about the world `z` axis.
fn rot_z_45(center: Vec3, half_x: f32, half_y: f32, half_z: f32) -> Obb {
    Obb::new(
        center,
        Vec3::new(HALF_SQRT2, HALF_SQRT2, 0.0),
        Vec3::new(-HALF_SQRT2, HALF_SQRT2, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        half_x,
        half_y,
        half_z,
    )
}

/// Runs the `GPU` dispatch and asserts strict lane-for-lane parity against the
/// `CPU` golden: the intersect flag matches exactly. Returns the `GPU` verdicts
/// for extra per-test assertions. Use only for fixtures placed clear of the
/// contact boundary.
fn check(ctx: &GpuContext, gpu: &GpuObbSat3d, queries: &[ObbSat3dQuery]) -> Vec<ObbSat3dResult> {
    let got = gpu.eval(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (lane, (g, q)) in got.iter().zip(queries.iter()).enumerate() {
        let want = cpu_reference(q);
        assert_eq!(
            g.intersects, want,
            "lane {lane}: intersects gpu {} vs cpu {want}",
            g.intersects
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

/// The signed excess `center_offset - radius_sum` of the two boxes on `axis`;
/// positive means `axis` separates them. Mirrors the golden `separated_on_axis`
/// numerator so the test can measure the margin to the contact boundary.
fn excess(a: &Obb, b: &Obb, axis: Vec3) -> f32 {
    let center_offset = b.center.sub(a.center).dot(axis).abs();
    let radius_sum = a.projected_radius(axis) + b.projected_radius(axis);
    center_offset - radius_sum
}

/// The largest per-axis excess across the fifteen `SAT` candidate axes (face
/// normals plus non-degenerate edge-cross axes). The pair is separated when this
/// exceeds [`CONTACT_EPS`]; its distance from `CONTACT_EPS` is the robustness
/// margin used to reject-sample the random batch away from the tie band.
fn max_excess(a: &Obb, b: &Obb) -> f32 {
    let a_axes = a.axes();
    let b_axes = b.axes();
    let mut best = f32::NEG_INFINITY;
    for face in a_axes.iter().chain(b_axes.iter()) {
        best = best.max(excess(a, b, *face));
    }
    for edge_a in a_axes.iter() {
        for edge_b in b_axes.iter() {
            let cross = edge_a.cross(*edge_b);
            if cross.length_squared() <= DEGENERATE_AXIS_EPS {
                continue;
            }
            best = best.max(excess(a, b, cross.normalized()));
        }
    }
    best
}

/// Draws a random orthonormal basis via Gram-Schmidt on two random vectors; no
/// transcendental appears (only `sqrt` through `Vec3::normalized`), matching the
/// golden's own `sqrt`-only normalization.
fn random_basis(state: &mut u64) -> (Vec3, Vec3, Vec3) {
    loop {
        let u = Vec3::new(
            lcg(state) * 2.0 - 1.0,
            lcg(state) * 2.0 - 1.0,
            lcg(state) * 2.0 - 1.0,
        );
        if u.length_squared() < 0.2 {
            continue;
        }
        let x = u.normalized();
        let v = Vec3::new(
            lcg(state) * 2.0 - 1.0,
            lcg(state) * 2.0 - 1.0,
            lcg(state) * 2.0 - 1.0,
        );
        let ortho = v.sub(x.mul(x.dot(v)));
        if ortho.length_squared() < 0.2 {
            continue;
        }
        let y = ortho.normalized();
        let z = x.cross(y);
        return (x, y, z);
    }
}

#[test]
fn empty_batch_is_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuObbSat3d::new(&ctx);
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch yields an empty result");
}

#[test]
fn axis_aligned_overlap_at_origin() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuObbSat3d::new(&ctx);
    let a = unit_cube_at(Vec3::new(0.0, 0.0, 0.0));
    let b = unit_cube_at(Vec3::new(0.5, 0.0, 0.0));
    let got = check(&ctx, &gpu, &[query(a, b)]);
    assert!(got[0].intersects, "overlapping unit cubes should intersect");
}

#[test]
fn separated_along_x() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuObbSat3d::new(&ctx);
    let a = unit_cube_at(Vec3::new(0.0, 0.0, 0.0));
    let b = unit_cube_at(Vec3::new(3.0, 0.0, 0.0));
    let got = check(&ctx, &gpu, &[query(a, b)]);
    assert!(!got[0].intersects, "boxes 3 apart on x must be separated");
}

#[test]
fn face_contact_touching_counts_as_intersection() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuObbSat3d::new(&ctx);
    // Centers exactly 2 apart on x, each half-extent 1: faces just touch, which
    // the CONTACT_EPS slack reports as an intersection on both devices.
    let a = unit_cube_at(Vec3::new(0.0, 0.0, 0.0));
    let b = unit_cube_at(Vec3::new(2.0, 0.0, 0.0));
    let got = check(&ctx, &gpu, &[query(a, b)]);
    assert!(got[0].intersects, "an exact face contact counts as a hit");
}

#[test]
fn tiny_gap_is_separated() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuObbSat3d::new(&ctx);
    // A 0.01 gap sits a full tie band past contact, so the verdict is robust.
    let a = unit_cube_at(Vec3::new(0.0, 0.0, 0.0));
    let b = unit_cube_at(Vec3::new(2.1, 0.0, 0.0));
    let got = check(&ctx, &gpu, &[query(a, b)]);
    assert!(!got[0].intersects, "a clear gap must be separated");
}

#[test]
fn containment_small_inside_large() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuObbSat3d::new(&ctx);
    let big = Obb::axis_aligned(Vec3::new(0.0, 0.0, 0.0), 5.0, 5.0, 5.0);
    let small = Obb::axis_aligned(Vec3::new(1.0, -1.0, 0.5), 0.25, 0.25, 0.25);
    let got = check(&ctx, &gpu, &[query(big, small), query(small, big)]);
    assert!(
        got[0].intersects && got[1].intersects,
        "containment intersects in either order"
    );
}

#[test]
fn rotated_z_45_overlap() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuObbSat3d::new(&ctx);
    let a = unit_cube_at(Vec3::new(0.0, 0.0, 0.0));
    let b = rot_z_45(Vec3::new(1.0, 0.0, 0.0), 1.0, 1.0, 1.0);
    let got = check(&ctx, &gpu, &[query(a, b)]);
    assert!(got[0].intersects, "a close z-rotated box overlaps");
}

#[test]
fn rotated_z_45_separation() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuObbSat3d::new(&ctx);
    let a = unit_cube_at(Vec3::new(0.0, 0.0, 0.0));
    let b = rot_z_45(Vec3::new(3.0, 0.0, 0.0), 1.0, 1.0, 1.0);
    let got = check(&ctx, &gpu, &[query(a, b)]);
    assert!(!got[0].intersects, "a distant z-rotated box is separated");
}

#[test]
fn thin_slab_overlap_and_separation() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuObbSat3d::new(&ctx);
    let a = Obb::axis_aligned(Vec3::new(0.0, 0.0, 0.0), 5.0, 5.0, 0.05);
    let touching = Obb::axis_aligned(Vec3::new(0.0, 0.0, 0.04), 0.5, 0.5, 0.05);
    let apart = Obb::axis_aligned(Vec3::new(0.0, 0.0, 0.5), 0.5, 0.5, 0.05);
    let got = check(&ctx, &gpu, &[query(a, touching), query(a, apart)]);
    assert!(got[0].intersects, "overlapping thin slabs intersect");
    assert!(!got[1].intersects, "a lifted thin slab separates");
}

#[test]
fn corner_overlap_and_diagonal_separation() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuObbSat3d::new(&ctx);
    let a = unit_cube_at(Vec3::new(0.0, 0.0, 0.0));
    let corner = unit_cube_at(Vec3::new(1.9, 1.9, 1.9));
    let far = unit_cube_at(Vec3::new(4.0, 4.0, 4.0));
    let got = check(&ctx, &gpu, &[query(a, corner), query(a, far)]);
    assert!(got[0].intersects, "a +++ corner overlap intersects");
    assert!(!got[1].intersects, "a far diagonal box is separated");
}

#[test]
fn rotated_about_x_overlap() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuObbSat3d::new(&ctx);
    let a = unit_cube_at(Vec3::new(0.0, 0.0, 0.0));
    let b = Obb::new(
        Vec3::new(0.0, 1.0, 1.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, HALF_SQRT2, HALF_SQRT2),
        Vec3::new(0.0, -HALF_SQRT2, HALF_SQRT2),
        1.0,
        1.0,
        1.0,
    );
    let got = check(&ctx, &gpu, &[query(a, b)]);
    assert!(got[0].intersects, "an x-rotated box overlaps the unit cube");
}

#[test]
fn random_batch_matches_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuObbSat3d::new(&ctx);
    let mut state = 0x_0bb0_5a73_1234_0001_u64;

    let mut saw_hit = false;
    let mut saw_miss = false;

    for _round in 0u32..8 {
        let mut queries = Vec::with_capacity(128);
        while queries.len() < 128 {
            let (ax, ay, az) = random_basis(&mut state);
            let (bx, by, bz) = random_basis(&mut state);
            let a = Obb::new(
                Vec3::new(
                    lcg(&mut state) * 4.0 - 2.0,
                    lcg(&mut state) * 4.0 - 2.0,
                    lcg(&mut state) * 4.0 - 2.0,
                ),
                ax,
                ay,
                az,
                0.5 + lcg(&mut state) * 1.5,
                0.5 + lcg(&mut state) * 1.5,
                0.5 + lcg(&mut state) * 1.5,
            );
            // Place b near a with an offset wide enough to mix clear overlaps
            // with clear separations.
            let b = Obb::new(
                Vec3::new(
                    a.center.x + lcg(&mut state) * 8.0 - 4.0,
                    a.center.y + lcg(&mut state) * 8.0 - 4.0,
                    a.center.z + lcg(&mut state) * 8.0 - 4.0,
                ),
                bx,
                by,
                bz,
                0.5 + lcg(&mut state) * 1.5,
                0.5 + lcg(&mut state) * 1.5,
                0.5 + lcg(&mut state) * 1.5,
            );
            // Reject pairs sitting on the contact boundary, where a legal ULP
            // perturbation could flip the `>` separation verdict.
            if (max_excess(&a, &b) - CONTACT_EPS).abs() < TIE {
                continue;
            }
            queries.push(query(a, b));
        }

        let got = gpu.eval(&ctx, &queries);
        assert_eq!(got.len(), queries.len(), "one result per query");
        for (lane, (g, q)) in got.iter().zip(queries.iter()).enumerate() {
            let want = cpu_reference(q);
            assert_eq!(
                g.intersects, want,
                "lane {lane}: intersects gpu {} vs cpu {want}",
                g.intersects
            );
            saw_hit |= want;
            saw_miss |= !want;
        }
    }

    assert!(
        saw_hit && saw_miss,
        "the random batch should exercise both overlapping and separated pairs"
    );
}
