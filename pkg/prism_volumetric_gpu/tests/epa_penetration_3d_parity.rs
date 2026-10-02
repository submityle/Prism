//! Real-device parity for the 3D `Expanding-Polytope-Algorithm` (`EPA`)
//! penetration twin:
//! [`GpuEpaPenetration3d`](prism_volumetric_gpu::epa_penetration_3d::GpuEpaPenetration3d)
//! must reproduce the `CPU` golden
//! [`penetration`](prism_render_architecture::particle::epa_penetration_3d::penetration)
//! across an empty batch, a zero-vertex degenerate pair, single-axis box
//! overlaps with a known dominant depth and normal (`x`, `y`, `z`, deep and
//! shallow), far-apart and diagonally-separated disjoint boxes, corner and
//! off-axis box overlaps whose tied faces leave the normal ambiguous, offset
//! octahedra and a tilted tetrahedron against a box that force several `EPA`
//! refinements, a mixed named batch and a large pseudo-random batch of
//! axis-aligned box pairs compared lane for lane.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The overlap verdict is a discrete classification built from `f32` magnitude
//! and sign comparisons against the reference bands, so the comparison always
//! asserts an exact `==` on the `0`/`1` `has_penetration` flag. Every fixture —
//! named and random — is placed either clearly overlapping or clearly separated
//! (a margin far wider than any legal `ULP`-scale perturbation), so no verdict
//! sits in a tie band where a fused multiply-add could flip it.
//!
//! For the `normal` and `depth` the `CPU` and `GPU` are not bit-exact: a `GPU`
//! may fuse a multiply-add the scalar reference leaves separate, perturbing the
//! low mantissa bits. The continuous fields are therefore compared within
//! `abs_diff <= 1e-4` or `rel_diff <= 1e-3`, the `depth` already pinned into the
//! reference's `EPA_TOLERANCE` convergence band. That `normal`/`depth` check is
//! applied only to fixtures with a single dominant overlap axis, where the
//! nearest face is unique (or a set of coplanar triangles that all carry the
//! same normal and distance) so both devices must agree. For symmetric corner,
//! octahedron and tetrahedron overlaps several faces tie at the minimum
//! distance, and the reference's order-preserving horizon rebuild and the
//! twin's order-independent one may settle on different tied faces with the same
//! depth but a different normal, so those fixtures assert the flag only.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::epa_penetration_3d`；
//! textbook 3D `GJK` + `EPA` penetration solver；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::epa_penetration_3d::{
    cpu_reference, EpaPenetration3dQuery, GpuEpa, GpuEpaPenetration3d,
};
use prism_volumetric_gpu::GpuContext;

/// Fixed-capacity vertex count per convex cloud in the twin's `std430` layout;
/// the test fixtures stay well within it.
const CAP: usize = 16;

/// Absolute parity bound for the continuous `normal` and `depth` fields. A
/// `GPU` may fuse a multiply-add the scalar reference leaves separate,
/// perturbing the low mantissa bits by a few units in the last place; `1e-4`
/// admits that legal slack while still failing a wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in
/// the last place exceed the absolute floor.
const REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Half-width of the rejection band for the random batch: a candidate pair is
/// kept only when every axis overlap exceeds this margin or at least one axis
/// gap exceeds it, so the verdict is unambiguous and far from the contact
/// boundary.
const MARGIN: f32 = 0.3;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
}

/// Packs two vertex clouds into an [`EpaPenetration3dQuery`], zero-padding the
/// unused fixed-capacity slots.
fn make(a: &[[f32; 3]], b: &[[f32; 3]]) -> EpaPenetration3dQuery {
    assert!(a.len() <= CAP && b.len() <= CAP, "cloud exceeds capacity");
    let mut verts_a = [[0.0_f32; 3]; CAP];
    let mut verts_b = [[0.0_f32; 3]; CAP];
    verts_a[..a.len()].copy_from_slice(a);
    verts_b[..b.len()].copy_from_slice(b);
    EpaPenetration3dQuery {
        verts_a,
        count_a: a.len() as u32,
        verts_b,
        count_b: b.len() as u32,
    }
}

/// Runs the `GPU` dispatch and asserts strict lane-for-lane parity on the
/// `0`/`1` `has_penetration` flag only. Returns the `GPU` results for extra
/// per-test assertions. Use for ambiguous overlaps where tied faces leave the
/// `normal` non-deterministic across the two horizon rebuild orders.
fn check_flag(
    ctx: &GpuContext,
    gpu: &GpuEpaPenetration3d,
    queries: &[EpaPenetration3dQuery],
) -> Vec<GpuEpa> {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (lane, (g, q)) in got.iter().zip(queries.iter()).enumerate() {
        let cpu = cpu_reference(q);
        assert_eq!(
            g.has_penetration, cpu.has_penetration,
            "lane {lane}: has_penetration gpu {} vs cpu {}",
            g.has_penetration, cpu.has_penetration
        );
    }
    got
}

/// Runs the `GPU` dispatch and asserts full lane-for-lane parity: the
/// `has_penetration` flag matches exactly, and on overlap the `normal` and
/// `depth` match within the continuous-field tolerance. Use only for fixtures
/// with a single dominant overlap axis, where the nearest face is unique and
/// both devices must settle on the same `normal`.
fn check_full(
    ctx: &GpuContext,
    gpu: &GpuEpaPenetration3d,
    queries: &[EpaPenetration3dQuery],
) -> Vec<GpuEpa> {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (lane, (g, q)) in got.iter().zip(queries.iter()).enumerate() {
        let cpu = cpu_reference(q);
        assert_eq!(
            g.has_penetration, cpu.has_penetration,
            "lane {lane}: has_penetration gpu {} vs cpu {}",
            g.has_penetration, cpu.has_penetration
        );
        if cpu.has_penetration == 1 {
            assert!(
                close(g.normal[0], cpu.normal[0])
                    && close(g.normal[1], cpu.normal[1])
                    && close(g.normal[2], cpu.normal[2]),
                "lane {lane}: normal gpu {:?} vs cpu {:?}",
                g.normal,
                cpu.normal
            );
            assert!(
                close(g.depth, cpu.depth),
                "lane {lane}: depth gpu {} vs cpu {}",
                g.depth,
                cpu.depth
            );
        }
    }
    got
}

/// The eight corners of an axis-aligned cube of half-extent `h` centered at
/// `c`, mirroring the golden `boxed` fixture helper.
fn boxed(c: [f32; 3], h: f32) -> Vec<[f32; 3]> {
    let mut out: Vec<[f32; 3]> = Vec::new();
    for sx in [-1.0_f32, 1.0] {
        for sy in [-1.0_f32, 1.0] {
            for sz in [-1.0_f32, 1.0] {
                out.push([c[0] + sx * h, c[1] + sy * h, c[2] + sz * h]);
            }
        }
    }
    out
}

/// The six vertices of an axis-aligned octahedron of radius `r` centered at
/// `c`, mirroring the golden `octa` fixture helper.
fn octa(c: [f32; 3], r: f32) -> Vec<[f32; 3]> {
    vec![
        [c[0] + r, c[1], c[2]],
        [c[0] - r, c[1], c[2]],
        [c[0], c[1] + r, c[2]],
        [c[0], c[1] - r, c[2]],
        [c[0], c[1], c[2] + r],
        [c[0], c[1], c[2] - r],
    ]
}

/// The four vertices of a regular-ish tetrahedron centered at `c`, scaled by
/// `s`, mirroring the golden `tetra_body` fixture helper.
fn tetra_body(c: [f32; 3], s: f32) -> Vec<[f32; 3]> {
    vec![
        [c[0] + s, c[1] + s, c[2] + s],
        [c[0] + s, c[1] - s, c[2] - s],
        [c[0] - s, c[1] + s, c[2] - s],
        [c[0] - s, c[1] - s, c[2] + s],
    ]
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
    let gpu = GpuEpaPenetration3d::new(&ctx);
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch yields an empty result");
}

#[test]
fn zero_vertex_cloud_reports_no_penetration() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuEpaPenetration3d::new(&ctx);
    // Cloud a is empty (count 0); there is no support point so no overlap.
    let q = make(&[], &boxed([0.0, 0.0, 0.0], 1.0));
    let got = check_full(&ctx, &gpu, &[q]);
    assert_eq!(got[0].has_penetration, 0, "empty cloud has no penetration");
}

#[test]
fn overlap_along_x_known_depth_and_normal() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuEpaPenetration3d::new(&ctx);
    // Boxes overlap by 0.5 along x, full on y and z: the nearest face is the
    // unique +x face, so the normal is unambiguous.
    let q = make(&boxed([0.0, 0.0, 0.0], 1.0), &boxed([1.5, 0.0, 0.0], 1.0));
    let got = check_full(&ctx, &gpu, &[q]);
    assert_eq!(got[0].has_penetration, 1, "boxes overlap along x");
    assert!(got[0].normal[0].abs() > 0.9, "normal {:?}", got[0].normal);
}

#[test]
fn overlap_along_y_known_depth_and_normal() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuEpaPenetration3d::new(&ctx);
    let q = make(&boxed([0.0, 0.0, 0.0], 1.0), &boxed([0.0, 1.25, 0.0], 1.0));
    let got = check_full(&ctx, &gpu, &[q]);
    assert_eq!(got[0].has_penetration, 1, "boxes overlap along y");
    assert!(got[0].normal[1].abs() > 0.9, "normal {:?}", got[0].normal);
}

#[test]
fn overlap_along_z_known_depth_and_normal() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuEpaPenetration3d::new(&ctx);
    let q = make(&boxed([0.0, 0.0, 0.0], 1.0), &boxed([0.0, 0.0, 1.75], 1.0));
    let got = check_full(&ctx, &gpu, &[q]);
    assert_eq!(got[0].has_penetration, 1, "boxes overlap along z");
    assert!(got[0].normal[2].abs() > 0.9, "normal {:?}", got[0].normal);
}

#[test]
fn deep_penetration_along_x() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuEpaPenetration3d::new(&ctx);
    // A deep overlap (centers only 0.2 apart) along a single dominant axis.
    let q = make(&boxed([0.0, 0.0, 0.0], 1.0), &boxed([0.2, 0.0, 0.0], 1.0));
    let got = check_full(&ctx, &gpu, &[q]);
    assert_eq!(got[0].has_penetration, 1, "deep overlap");
    assert!(got[0].depth > 0.0, "depth {}", got[0].depth);
}

#[test]
fn shallow_penetration_along_x() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuEpaPenetration3d::new(&ctx);
    // A thin overlap (~0.02) along x, still a unique dominant axis.
    let q = make(&boxed([0.0, 0.0, 0.0], 1.0), &boxed([1.98, 0.0, 0.0], 1.0));
    let got = check_full(&ctx, &gpu, &[q]);
    assert_eq!(got[0].has_penetration, 1, "thin overlap");
    assert!(got[0].depth > 0.0, "depth {}", got[0].depth);
}

#[test]
fn far_boxes_do_not_penetrate() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuEpaPenetration3d::new(&ctx);
    let q = make(&boxed([0.0, 0.0, 0.0], 1.0), &boxed([3.0, 0.0, 0.0], 1.0));
    let got = check_full(&ctx, &gpu, &[q]);
    assert_eq!(got[0].has_penetration, 0, "far boxes are disjoint");
}

#[test]
fn diagonally_separated_boxes_do_not_penetrate() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuEpaPenetration3d::new(&ctx);
    let q = make(&boxed([0.0, 0.0, 0.0], 1.0), &boxed([2.5, 2.5, 2.5], 1.0));
    let got = check_full(&ctx, &gpu, &[q]);
    assert_eq!(got[0].has_penetration, 0, "diagonal gap is disjoint");
}

#[test]
fn diagonal_corner_overlap_flag_only() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuEpaPenetration3d::new(&ctx);
    // A symmetric corner overlap: three faces tie at the minimum distance so
    // the normal is not deterministic across horizon rebuild orders.
    let q = make(&boxed([0.0, 0.0, 0.0], 1.0), &boxed([1.2, 1.2, 1.2], 1.0));
    let got = check_flag(&ctx, &gpu, &[q]);
    assert_eq!(got[0].has_penetration, 1, "corner overlap penetrates");
}

#[test]
fn off_axis_overlap_flag_only() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuEpaPenetration3d::new(&ctx);
    // Two axes overlap by a comparable amount, so the nearest face can tie.
    let q = make(&boxed([0.0, 0.0, 0.0], 1.0), &boxed([1.4, 0.9, 0.0], 1.0));
    let got = check_flag(&ctx, &gpu, &[q]);
    assert_eq!(got[0].has_penetration, 1, "off-axis overlap penetrates");
}

#[test]
fn offset_octahedra_flag_only() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuEpaPenetration3d::new(&ctx);
    // Octahedra force several EPA refinements; the run must still terminate and
    // agree on the overlap flag.
    let q = make(&octa([0.0, 0.0, 0.0], 1.5), &octa([0.4, 0.3, 0.2], 1.5));
    let got = check_flag(&ctx, &gpu, &[q]);
    assert_eq!(got[0].has_penetration, 1, "offset octahedra overlap");
}

#[test]
fn tilted_tetrahedron_vs_box_flag_only() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuEpaPenetration3d::new(&ctx);
    // A tetrahedron that is not axis aligned overlapping a box.
    let q = make(
        &tetra_body([0.0, 0.0, 0.0], 1.2),
        &boxed([0.6, 0.4, 0.3], 1.0),
    );
    let got = check_flag(&ctx, &gpu, &[q]);
    assert_eq!(got[0].has_penetration, 1, "tetra overlaps box");
}

#[test]
fn mixed_named_batch_matches_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuEpaPenetration3d::new(&ctx);
    // A single dispatch mixing hits and misses so one batch exercises both
    // verdict classes through the shared pipeline.
    let queries = [
        make(&boxed([0.0, 0.0, 0.0], 1.0), &boxed([1.5, 0.0, 0.0], 1.0)),
        make(&boxed([0.0, 0.0, 0.0], 1.0), &boxed([3.0, 0.0, 0.0], 1.0)),
        make(&boxed([0.0, 0.0, 0.0], 1.0), &boxed([0.0, 0.0, 1.75], 1.0)),
        make(&boxed([0.0, 0.0, 0.0], 1.0), &boxed([2.5, 2.5, 2.5], 1.0)),
    ];
    let got = check_flag(&ctx, &gpu, &queries);
    assert_eq!(got[0].has_penetration, 1);
    assert_eq!(got[1].has_penetration, 0);
    assert_eq!(got[2].has_penetration, 1);
    assert_eq!(got[3].has_penetration, 0);
}

#[test]
fn random_batch_matches_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuEpaPenetration3d::new(&ctx);

    let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut queries: Vec<EpaPenetration3dQuery> = Vec::with_capacity(256);
    while queries.len() < 256 {
        // Two axis-aligned boxes with random centers and half extents.
        let acx = lcg(&mut state) * 10.0 - 5.0;
        let acy = lcg(&mut state) * 10.0 - 5.0;
        let acz = lcg(&mut state) * 10.0 - 5.0;
        let ah = 0.6 + lcg(&mut state) * 1.2;
        let bcx = lcg(&mut state) * 10.0 - 5.0;
        let bcy = lcg(&mut state) * 10.0 - 5.0;
        let bcz = lcg(&mut state) * 10.0 - 5.0;
        let bh = 0.6 + lcg(&mut state) * 1.2;

        // Signed per-axis overlap of the two intervals (positive = overlap).
        let ox = (acx + ah).min(bcx + bh) - (acx - ah).max(bcx - bh);
        let oy = (acy + ah).min(bcy + bh) - (acy - ah).max(bcy - bh);
        let oz = (acz + ah).min(bcz + bh) - (acz - ah).max(bcz - bh);

        // Keep only pairs that clearly overlap on all three axes or clearly
        // separate on at least one axis, so the verdict is unambiguous and far
        // from the contact boundary.
        let clear_overlap = ox > MARGIN && oy > MARGIN && oz > MARGIN;
        let clear_separate = ox < -MARGIN || oy < -MARGIN || oz < -MARGIN;
        if !(clear_overlap || clear_separate) {
            continue;
        }

        let a = boxed([acx, acy, acz], ah);
        let b = boxed([bcx, bcy, bcz], bh);
        queries.push(make(&a, &b));
    }

    let got = gpu.evaluate(&ctx, &queries);
    assert_eq!(got.len(), queries.len(), "one result per query");

    let mut saw_hit = false;
    let mut saw_miss = false;
    for (lane, (g, q)) in got.iter().zip(queries.iter()).enumerate() {
        let cpu = cpu_reference(q);
        assert_eq!(
            g.has_penetration, cpu.has_penetration,
            "lane {lane}: has_penetration gpu {} vs cpu {}",
            g.has_penetration, cpu.has_penetration
        );
        saw_hit |= cpu.has_penetration == 1;
        saw_miss |= cpu.has_penetration == 0;
    }

    assert!(
        saw_hit && saw_miss,
        "random batch should produce both overlaps and separations"
    );
}
