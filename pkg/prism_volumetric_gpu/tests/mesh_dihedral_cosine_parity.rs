//! Real-device parity for the per-interior-edge dihedral geometry twin:
//! [`GpuDihedralCosine`](prism_volumetric_gpu::mesh_dihedral_cosine::GpuDihedralCosine)
//! must reproduce the per-edge closed form of the golden
//! `prism_render_architecture::ray_scene::mesh_dihedral_cosine::dihedral_cosines`
//! across an empty batch, a coplanar pair, a right-angle fold, a ridge/valley
//! reverse fold, a zero-area degenerate face and a large pseudo-random sweep
//! compared lane for lane.
//!
//! The host oracle in this file re-derives the closed form independently (it
//! does **not** import the golden crate): the per-face `normalize(cross(...))`
//! unit normal, the clamped `cosine = dot(n0, n1)` and the clamped
//! `signed_sine = dot(cross(n0, n1), e)` along the unit edge direction `e`,
//! plus the zero-area degenerate sentinel. A passing run is therefore direct
//! evidence the ported kernel folds the same geometry, not merely that its
//! shader compiles.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each query is a fixed, non-reorderable sequence of cross and dot products
//! and two `sqrt`-based normalizations, so `CPU` and `GPU` evaluate the same
//! closed form in the same associativity. They are not bit-exact: a `GPU` may
//! fuse a multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place. The comparison therefore
//! allows `abs_diff <= 1e-4` or `rel_diff <= 1e-3` on the continuous `cosine`
//! and `signed_sine` values yet asserts an *exact* match on the discrete
//! `degenerate` flag. Every fixture and every random fold is placed clear of
//! the zero-area boundary, the only place a legal `ULP` perturbation can flip
//! that flag, so the exact-flag assertion is unconditional.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::mesh_dihedral_cosine`；
//! 无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::mesh_dihedral_cosine::{
    DihedralCosineQuery, DihedralCosineResult, GpuDihedralCosine,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on the cosine and signed-sine. A `GPU` may fuse a
/// multiply-add the scalar reference leaves separate, perturbing the low
/// mantissa bits by a few units in the last place; `1e-4` admits that legal
/// slack while still failing a genuinely wrong port.
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

/// Componentwise `a - b`.
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Euclidean dot product.
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Right-handed cross product `a × b`.
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Unit-length form of `v`, or `None` when `v` is (near) zero. Mirrors the
/// golden `normalize`, which fails when `len_sq <= 0`.
fn normalize(v: [f32; 3]) -> Option<[f32; 3]> {
    let len_sq = dot(v, v);
    if len_sq <= 0.0 {
        return None;
    }
    let inv = 1.0 / len_sq.sqrt();
    Some([v[0] * inv, v[1] * inv, v[2] * inv])
}

/// Unit normal of the triangle `(a, b, c)` in its own winding, or `None` when
/// the triangle is zero-area. Mirrors the golden `face_unit_normal`.
fn face_normal(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> Option<[f32; 3]> {
    normalize(cross(sub(b, a), sub(c, a)))
}

/// Independent host re-derivation of the golden per-edge dihedral block plus
/// the kernel's zero-area sentinel. It does not call the reference crate; it
/// reimplements the closed form so the parity assertion compares two
/// independent solutions.
fn oracle(q: &DihedralCosineQuery) -> DihedralCosineResult {
    let n0 = face_normal(q.tri0[0], q.tri0[1], q.tri0[2]);
    let n1 = face_normal(q.tri1[0], q.tri1[1], q.tri1[2]);
    let (Some(n0), Some(n1)) = (n0, n1) else {
        return DihedralCosineResult {
            cosine: 0.0,
            signed_sine: 0.0,
            degenerate: true,
        };
    };
    let cosine = dot(n0, n1).clamp(-1.0, 1.0);
    let signed_sine = match normalize(sub(q.edge_b, q.edge_a)) {
        Some(e) => dot(cross(n0, n1), e).clamp(-1.0, 1.0),
        None => 0.0,
    };
    DihedralCosineResult {
        cosine,
        signed_sine,
        degenerate: false,
    }
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

/// Maps a `[0, 1)` sample into `[lo, hi)`.
fn uniform(x: f32, lo: f32, hi: f32) -> f32 {
    lo + (hi - lo) * x
}

/// Runs the `GPU` dispatch and asserts strict lane-for-lane parity against the
/// independent host oracle: the `degenerate` flag matches exactly, and the
/// `cosine` and `signed_sine` match within tolerance. Returns the `GPU`
/// verdicts for extra per-test assertions. Use only for queries placed clear of
/// the zero-area boundary.
fn check(
    ctx: &GpuContext,
    gpu: &GpuDihedralCosine,
    queries: &[DihedralCosineQuery],
) -> Vec<DihedralCosineResult> {
    let got = gpu.eval(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (lane, (g, q)) in got.iter().zip(queries.iter()).enumerate() {
        let o = oracle(q);
        assert_eq!(
            g.degenerate, o.degenerate,
            "lane {lane}: degenerate gpu {} vs cpu {}",
            g.degenerate, o.degenerate
        );
        assert!(
            close(g.cosine, o.cosine),
            "lane {lane}: cosine gpu {} vs cpu {}",
            g.cosine,
            o.cosine
        );
        assert!(
            close(g.signed_sine, o.signed_sine),
            "lane {lane}: signed_sine gpu {} vs cpu {}",
            g.signed_sine,
            o.signed_sine
        );
    }
    got
}

#[test]
fn empty_batch_is_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDihedralCosine::new(&ctx);
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch yields an empty result");
}

#[test]
fn coplanar_pair_has_unit_cosine() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDihedralCosine::new(&ctx);

    // Two flat triangles sharing the x-axis edge, both lying in the z = 0
    // plane with the same +z normal, so the dihedral is perfectly open.
    let a = [0.0, 0.0, 0.0];
    let b = [1.0, 0.0, 0.0];
    let q = DihedralCosineQuery::new([a, b, [0.5, 1.0, 0.0]], [b, a, [0.5, -1.0, 0.0]], a, b);

    let got = check(&ctx, &gpu, &[q]);
    assert!(close(got[0].cosine, 1.0), "coplanar faces give cosine 1");
    assert!(!got[0].degenerate, "a flat pair is well conditioned");
}

#[test]
fn right_angle_fold_has_zero_cosine() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDihedralCosine::new(&ctx);

    // One flat face in z = 0 with +z normal; the opposite face stands up in the
    // x-z plane so the two normals meet at a right angle.
    let a = [0.0, 0.0, 0.0];
    let b = [1.0, 0.0, 0.0];
    let q = DihedralCosineQuery::new([a, b, [0.5, 1.0, 0.0]], [b, a, [0.5, 0.0, -1.0]], a, b);

    let got = check(&ctx, &gpu, &[q]);
    assert!(
        got[0].cosine.abs() <= EPS,
        "a right-angle fold gives cosine 0, got {}",
        got[0].cosine
    );
    assert!(!got[0].degenerate, "a right-angle fold is well conditioned");
}

#[test]
fn reverse_fold_flips_signed_sine() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDihedralCosine::new(&ctx);

    // The same fold lifted in +z and in -z: a ridge versus a valley. The signed
    // sine must take opposite signs even though the cosine (fold sharpness) is
    // identical.
    let a = [0.0, 0.0, 0.0];
    let b = [1.0, 0.0, 0.0];
    let up = DihedralCosineQuery::new([a, b, [0.5, 1.0, 0.0]], [b, a, [0.5, -1.0, 0.7]], a, b);
    let down = DihedralCosineQuery::new([a, b, [0.5, 1.0, 0.0]], [b, a, [0.5, -1.0, -0.7]], a, b);

    let got = check(&ctx, &gpu, &[up, down]);
    assert!(close(got[0].cosine, got[1].cosine), "same fold sharpness");
    assert!(
        got[0].signed_sine * got[1].signed_sine < 0.0,
        "ridge and valley take opposite signed-sine signs: {} vs {}",
        got[0].signed_sine,
        got[1].signed_sine
    );
}

#[test]
fn degenerate_face_flags_sentinel() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDihedralCosine::new(&ctx);

    // The first face is collinear (zero area, hence no normal); the kernel must
    // flag the lane degenerate and emit the sentinel zeros.
    let a = [0.0, 0.0, 0.0];
    let b = [1.0, 0.0, 0.0];
    let q = DihedralCosineQuery::new(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        [b, a, [0.5, -1.0, 0.3]],
        a,
        b,
    );

    let got = gpu.eval(&ctx, &[q]);
    assert_eq!(got.len(), 1, "one result per query");
    assert!(got[0].degenerate, "a zero-area face is degenerate");
    // The sentinel zeros are exact, not merely close.
    assert_eq!(
        got[0].cosine.to_bits(),
        0.0_f32.to_bits(),
        "sentinel cosine 0"
    );
    assert_eq!(
        got[0].signed_sine.to_bits(),
        0.0_f32.to_bits(),
        "sentinel signed_sine 0"
    );
}

#[test]
fn random_sweep_matches_lane_for_lane() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDihedralCosine::new(&ctx);

    let mut state: u64 = 0x0f0e_0d0c_0b0a_0908;
    let mut queries: Vec<DihedralCosineQuery> = Vec::new();

    for _ in 0..512 {
        // Random shared edge and two random apex points. The edge is given a
        // comfortable length and each face a comfortable area so the sample
        // stays clear of the zero-area boundary where a legal ULP perturbation
        // could flip the discrete degenerate flag.
        let a = [
            uniform(lcg(&mut state), -1.0, 1.0),
            uniform(lcg(&mut state), -1.0, 1.0),
            uniform(lcg(&mut state), -1.0, 1.0),
        ];
        let b = [
            a[0] + uniform(lcg(&mut state), 0.5, 1.5),
            a[1] + uniform(lcg(&mut state), -1.5, 1.5),
            a[2] + uniform(lcg(&mut state), -1.5, 1.5),
        ];
        let c0 = [
            uniform(lcg(&mut state), -1.5, 1.5),
            uniform(lcg(&mut state), -1.5, 1.5),
            uniform(lcg(&mut state), -1.5, 1.5),
        ];
        let c1 = [
            uniform(lcg(&mut state), -1.5, 1.5),
            uniform(lcg(&mut state), -1.5, 1.5),
            uniform(lcg(&mut state), -1.5, 1.5),
        ];

        let tri0 = [a, b, c0];
        let tri1 = [b, a, c1];

        // Reject any sample whose faces or edge are near degenerate.
        let cr0 = cross(sub(tri0[1], tri0[0]), sub(tri0[2], tri0[0]));
        let cr1 = cross(sub(tri1[1], tri1[0]), sub(tri1[2], tri1[0]));
        let edge = sub(b, a);
        if dot(cr0, cr0) < 0.25 || dot(cr1, cr1) < 0.25 || dot(edge, edge) < 0.25 {
            continue;
        }

        queries.push(DihedralCosineQuery::new(tri0, tri1, a, b));
    }

    assert!(
        !queries.is_empty(),
        "sweep should retain conditioned queries"
    );
    let got = check(&ctx, &gpu, &queries);
    assert!(
        got.iter().all(|g| !g.degenerate),
        "every conditioned fold is non-degenerate"
    );
}
