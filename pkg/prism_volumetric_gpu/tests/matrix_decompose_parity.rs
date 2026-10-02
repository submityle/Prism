//! Real-device parity for the matrix-decomposition twin:
//! [`GpuMatrixDecompose`](prism_volumetric_gpu::matrix_decompose::GpuMatrixDecompose)
//! must reproduce the `CPU` golden
//! [`matrix_decompose`](prism_render_architecture::particle::matrix_decompose)
//! across the `3x3` operators (`mul_vec3`, `mul_mat3`, `transpose`,
//! `determinant`), the `quaternion` round trip (`mat3_from_quat`,
//! `quat_from_mat3`), the `4x4` affine point transform (`mul_point`), and the
//! affine decompose / compose split (`decompose_affine`, `compose_trs`).
//!
//! The fixtures cover the shapes the golden unit tests call out: the identity
//! and a non-trivial `90`-degree rotation, all four trace / `Shepperd` pivot
//! branches (identity for the positive trace plus the three `180`-degree axis
//! rotations whose largest diagonal pivot selects each remaining branch, with
//! every `sqrt` argument held at `4`, far from zero), a non-uniform positive
//! scale, a reflection (negative determinant folding one scale negative), a
//! rotation-plus-scale decompose / compose round trip, and a degenerate
//! zero-length basis column that must fall back to its identity axis. Every
//! rotation is written from integer entries or a hand-normalized `quaternion`
//! built from `sqrt`, so the fixtures stay pure and use no transcendental math.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The matrix, vector and scalar entries thread through multiplies, adds, one
//! guarded division and at most one `sqrt`, so `CPU` and `GPU` are compared under
//! tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`).
//! Because a `quaternion` and its negation encode the same rotation, recovered
//! `quaternion`s are compared under the double-cover "same sign or wholly
//! negated" rule the golden tests use.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::matrix_decompose`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::matrix_decompose::{
    compose_trs, decompose_affine, mat3_from_quat, quat_from_mat3, Mat3, Mat4, Trs,
};
use prism_volumetric_gpu::matrix_decompose::{
    GpuMatrixDecompose, MatrixDecomposeQuery, MatrixDecomposeResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous quantities.
const ABS_EPS: f32 = 1.0e-4;
/// Relative tolerance for the continuous quantities.
const REL_EPS: f32 = 1.0e-3;
/// Floor for the relative-tolerance denominator.
const REL_FLOOR: f32 = 1.0e-6;

/// Mixed absolute / relative tolerance comparison for one `f32` lane.
fn approx(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= ABS_EPS {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale <= REL_EPS
}

/// Tolerant comparison of two `3`-vectors.
fn approx3(a: [f32; 3], b: [f32; 3]) -> bool {
    approx(a[0], b[0]) && approx(a[1], b[1]) && approx(a[2], b[2])
}

/// Tolerant comparison of two column-major `3x3` matrices.
fn approx_mat3(a: &Mat3, b: &Mat3) -> bool {
    (0..3).all(|c| approx3(a.cols[c], b.cols[c]))
}

/// Tolerant comparison of two column-major `4x4` matrices.
fn approx_mat4(a: &Mat4, b: &Mat4) -> bool {
    (0..4).all(|c| (0..4).all(|r| approx(a.cols[c][r], b.cols[c][r])))
}

/// Equivalent up to the `quaternion` double cover (`q` and `-q` rotate
/// identically); copied from the golden `quat_equiv` test helper.
fn quat_equiv(a: [f32; 4], b: [f32; 4]) -> bool {
    let same = approx(a[0], b[0]) && approx(a[1], b[1]) && approx(a[2], b[2]) && approx(a[3], b[3]);
    let flipped =
        approx(a[0], -b[0]) && approx(a[1], -b[1]) && approx(a[2], -b[2]) && approx(a[3], -b[3]);
    same || flipped
}

/// Half of `sqrt(2)`'s reciprocal: the half-angle sine / cosine of a
/// `90`-degree rotation, built from `sqrt` so no trigonometry runs.
fn half() -> f32 {
    0.5_f32.sqrt()
}

/// A unit `quaternion` for a rotation about `axis` from a half-angle
/// sine / cosine pair.
fn quat_axis(axis: [f32; 3], sin_half: f32, cos_half: f32) -> [f32; 4] {
    [
        axis[0] * sin_half,
        axis[1] * sin_half,
        axis[2] * sin_half,
        cos_half,
    ]
}

/// Runs one query on device and returns its single result.
fn solve(
    gpu: &GpuMatrixDecompose,
    ctx: &GpuContext,
    q: MatrixDecomposeQuery,
) -> MatrixDecomposeResult {
    let got = gpu.evaluate(ctx, std::slice::from_ref(&q));
    assert_eq!(got.len(), 1, "one result per query");
    got[0]
}

/// Unwraps a [`MatrixDecomposeResult::Vec3`].
fn as_vec3(r: MatrixDecomposeResult) -> [f32; 3] {
    match r {
        MatrixDecomposeResult::Vec3(v) => v,
        other => panic!("expected Vec3, got {other:?}"),
    }
}

/// Unwraps a [`MatrixDecomposeResult::Mat3`].
fn as_mat3(r: MatrixDecomposeResult) -> Mat3 {
    match r {
        MatrixDecomposeResult::Mat3(m) => m,
        other => panic!("expected Mat3, got {other:?}"),
    }
}

/// Unwraps a [`MatrixDecomposeResult::Scalar`].
fn as_scalar(r: MatrixDecomposeResult) -> f32 {
    match r {
        MatrixDecomposeResult::Scalar(s) => s,
        other => panic!("expected Scalar, got {other:?}"),
    }
}

/// Unwraps a [`MatrixDecomposeResult::Quat`].
fn as_quat(r: MatrixDecomposeResult) -> [f32; 4] {
    match r {
        MatrixDecomposeResult::Quat(q) => q,
        other => panic!("expected Quat, got {other:?}"),
    }
}

/// Unwraps a [`MatrixDecomposeResult::Mat4`].
fn as_mat4(r: MatrixDecomposeResult) -> Mat4 {
    match r {
        MatrixDecomposeResult::Mat4(m) => m,
        other => panic!("expected Mat4, got {other:?}"),
    }
}

/// Unwraps a [`MatrixDecomposeResult::Trs`].
fn as_trs(r: MatrixDecomposeResult) -> Trs {
    match r {
        MatrixDecomposeResult::Trs(t) => t,
        other => panic!("expected Trs, got {other:?}"),
    }
}

/// A general affine matrix with an integer rotation-scale-shear block and a
/// translation column.
fn sample_mat4() -> Mat4 {
    let mut m = Mat4::identity();
    m.cols[0] = [2.0, 0.0, 0.0, 0.0];
    m.cols[1] = [0.0, 3.0, 0.0, 0.0];
    m.cols[2] = [0.0, 0.0, 4.0, 0.0];
    m.cols[3] = [10.0, 20.0, 30.0, 1.0];
    m
}

#[test]
fn mat3_mul_vec3_and_mul_mat3_match() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMatrixDecompose::new(&ctx);
    let m = Mat3::from_cols([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]);
    let v = [1.0, -2.0, 0.5];
    let got = as_vec3(solve(
        &gpu,
        &ctx,
        MatrixDecomposeQuery::MulVec3 {
            matrix: m,
            vector: v,
        },
    ));
    assert!(approx3(got, m.mul_vec3(v)), "mul_vec3 {got:?}");

    let rhs = Mat3::from_cols([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 2.0]);
    let got = as_mat3(solve(
        &gpu,
        &ctx,
        MatrixDecomposeQuery::MulMat3 { lhs: m, rhs },
    ));
    assert!(approx_mat3(&got, &m.mul_mat3(&rhs)), "mul_mat3");
}

#[test]
fn mat3_transpose_and_determinant_match() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMatrixDecompose::new(&ctx);
    let m = Mat3::from_cols([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]);
    let got = as_mat3(solve(
        &gpu,
        &ctx,
        MatrixDecomposeQuery::Transpose { matrix: m },
    ));
    assert!(approx_mat3(&got, &m.transpose()), "transpose");

    let got = as_scalar(solve(
        &gpu,
        &ctx,
        MatrixDecomposeQuery::Determinant { matrix: m },
    ));
    assert!(approx(got, m.determinant()), "determinant {got}");

    let diag = Mat3::from_cols([2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]);
    let got = as_scalar(solve(
        &gpu,
        &ctx,
        MatrixDecomposeQuery::Determinant { matrix: diag },
    ));
    assert!(approx(got, diag.determinant()), "diag determinant {got}");
}

#[test]
fn mat4_mul_point_applies_block_and_translation() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMatrixDecompose::new(&ctx);
    let m = sample_mat4();
    let p = [1.0, 2.0, 3.0];
    let got = as_vec3(solve(
        &gpu,
        &ctx,
        MatrixDecomposeQuery::MulPoint {
            matrix: m,
            point: p,
        },
    ));
    assert!(approx3(got, m.mul_point(p)), "mul_point {got:?}");

    let got = as_vec3(solve(
        &gpu,
        &ctx,
        MatrixDecomposeQuery::MulPoint {
            matrix: Mat4::identity(),
            point: [4.0, -7.0, 2.0],
        },
    ));
    assert!(approx3(got, [4.0, -7.0, 2.0]), "identity mul_point {got:?}");
}

#[test]
fn mat3_from_quat_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMatrixDecompose::new(&ctx);
    // Identity quaternion yields the identity matrix.
    let got = as_mat3(solve(
        &gpu,
        &ctx,
        MatrixDecomposeQuery::Mat3FromQuat {
            quat: [0.0, 0.0, 0.0, 1.0],
        },
    ));
    assert!(
        approx_mat3(&got, &mat3_from_quat([0.0, 0.0, 0.0, 1.0])),
        "identity quat"
    );

    // 90 degrees about +Z sends +X to +Y.
    let q = quat_axis([0.0, 0.0, 1.0], half(), half());
    let got = as_mat3(solve(
        &gpu,
        &ctx,
        MatrixDecomposeQuery::Mat3FromQuat { quat: q },
    ));
    assert!(approx_mat3(&got, &mat3_from_quat(q)), "90deg Z quat");
    assert!(
        approx3(got.mul_vec3([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0]),
        "rotates +X to +Y"
    );
}

#[test]
fn quat_from_mat3_hits_all_shepperd_branches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMatrixDecompose::new(&ctx);
    // Each matrix selects a distinct trace / Shepperd branch; the sqrt argument
    // is held at 4 in every case, far from zero.
    let cases = [
        // trace > 0 (trace = 3).
        Mat3::identity(),
        // m00 is the largest pivot: 180 degrees about X.
        Mat3::from_cols([1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]),
        // m11 is the largest pivot: 180 degrees about Y.
        Mat3::from_cols([-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]),
        // m22 is the largest pivot (else branch): 180 degrees about Z.
        Mat3::from_cols([-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]),
    ];
    for m in cases {
        let got = as_quat(solve(
            &gpu,
            &ctx,
            MatrixDecomposeQuery::QuatFromMat3 { matrix: m },
        ));
        assert!(
            quat_equiv(got, quat_from_mat3(&m)),
            "quat_from_mat3 branch mismatch: gpu {got:?} vs cpu {:?}",
            quat_from_mat3(&m)
        );
    }
}

#[test]
fn quat_mat3_round_trip_recovers_rotation() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMatrixDecompose::new(&ctx);
    // Hand-normalized axis [1, 2, 2] (length 3) with a 90-degree half angle.
    let axis = [1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0];
    let q = quat_axis(axis, half(), half());
    let r = mat3_from_quat(q);
    let got = as_quat(solve(
        &gpu,
        &ctx,
        MatrixDecomposeQuery::QuatFromMat3 { matrix: r },
    ));
    assert!(quat_equiv(got, quat_from_mat3(&r)), "round-trip vs cpu");
    assert!(quat_equiv(got, q), "round-trip vs source quat");
}

#[test]
fn decompose_and_compose_round_trip() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMatrixDecompose::new(&ctx);

    // Identity decomposes to the unit transform.
    let trs = as_trs(solve(
        &gpu,
        &ctx,
        MatrixDecomposeQuery::DecomposeAffine {
            matrix: Mat4::identity(),
        },
    ));
    let cpu = decompose_affine(&Mat4::identity());
    assert!(
        approx3(trs.translation, cpu.translation),
        "identity translation"
    );
    assert!(approx3(trs.scale, cpu.scale), "identity scale");
    assert!(quat_equiv(trs.rotation, cpu.rotation), "identity rotation");

    // Rotation + non-uniform positive scale + translation: compose on device,
    // then decompose on device, and check the round trip reproduces the matrix.
    let q = quat_axis([1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0], half(), half());
    let source = Trs {
        translation: [1.0, 2.0, 3.0],
        rotation: q,
        scale: [2.0, 3.0, 4.0],
    };
    let composed = as_mat4(solve(
        &gpu,
        &ctx,
        MatrixDecomposeQuery::ComposeTrs { trs: source },
    ));
    assert!(
        approx_mat4(&composed, &compose_trs(&source)),
        "compose vs cpu"
    );

    let trs = as_trs(solve(
        &gpu,
        &ctx,
        MatrixDecomposeQuery::DecomposeAffine { matrix: composed },
    ));
    let recomposed = as_mat4(solve(&gpu, &ctx, MatrixDecomposeQuery::ComposeTrs { trs }));
    assert!(
        approx_mat4(&recomposed, &composed),
        "decompose/compose round trip"
    );
    assert!(
        quat_equiv(trs.rotation, source.rotation),
        "recovered rotation"
    );
    assert!(approx3(trs.scale, source.scale), "recovered scale");
}

#[test]
fn decompose_negative_determinant_folds_one_scale() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMatrixDecompose::new(&ctx);
    // Mirror the X axis: the determinant is negative, so exactly one scale
    // component is negative and the recovered rotation stays proper.
    let mut m = Mat4::identity();
    m.cols[0] = [-1.0, 0.0, 0.0, 0.0];
    let trs = as_trs(solve(
        &gpu,
        &ctx,
        MatrixDecomposeQuery::DecomposeAffine { matrix: m },
    ));
    let cpu = decompose_affine(&m);
    assert!(
        approx3(trs.scale, cpu.scale),
        "scale {:?} vs {:?}",
        trs.scale,
        cpu.scale
    );
    assert_eq!(
        trs.scale.iter().filter(|s| **s < 0.0).count(),
        1,
        "exactly one negative scale"
    );
    let recomposed = as_mat4(solve(&gpu, &ctx, MatrixDecomposeQuery::ComposeTrs { trs }));
    assert!(approx_mat4(&recomposed, &m), "reflection recomposes");

    // Non-uniform scale plus reflection on Y.
    let mut m = Mat4::identity();
    m.cols[0] = [2.0, 0.0, 0.0, 0.0];
    m.cols[1] = [0.0, -3.0, 0.0, 0.0];
    m.cols[2] = [0.0, 0.0, 4.0, 0.0];
    let trs = as_trs(solve(
        &gpu,
        &ctx,
        MatrixDecomposeQuery::DecomposeAffine { matrix: m },
    ));
    let recomposed = as_mat4(solve(&gpu, &ctx, MatrixDecomposeQuery::ComposeTrs { trs }));
    assert!(approx_mat4(&recomposed, &m), "YZ reflection recomposes");
}

#[test]
fn decompose_degenerate_column_falls_back_to_axis() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMatrixDecompose::new(&ctx);
    // A zero-length first basis column drives the scale to zero; the de-scaled
    // rotation falls back to the identity X axis instead of dividing by ~zero.
    let mut m = Mat4::identity();
    m.cols[0] = [0.0, 0.0, 0.0, 0.0];
    let trs = as_trs(solve(
        &gpu,
        &ctx,
        MatrixDecomposeQuery::DecomposeAffine { matrix: m },
    ));
    let cpu = decompose_affine(&m);
    assert!(approx3(trs.scale, cpu.scale), "degenerate scale");
    assert!(
        quat_equiv(trs.rotation, cpu.rotation),
        "degenerate rotation fallback"
    );
}

#[test]
fn batch_of_mixed_ops_matches_elementwise() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMatrixDecompose::new(&ctx);
    // A batch exercises the one-thread-per-query flattening across different
    // tags; each result must be independent of its neighbours.
    let m = Mat3::from_cols([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]);
    let q = quat_axis([0.0, 0.0, 1.0], half(), half());
    let batch = [
        MatrixDecomposeQuery::MulVec3 {
            matrix: m,
            vector: [1.0, 1.0, 1.0],
        },
        MatrixDecomposeQuery::Transpose { matrix: m },
        MatrixDecomposeQuery::Determinant { matrix: m },
        MatrixDecomposeQuery::Mat3FromQuat { quat: q },
        MatrixDecomposeQuery::QuatFromMat3 {
            matrix: mat3_from_quat(q),
        },
        MatrixDecomposeQuery::MulPoint {
            matrix: sample_mat4(),
            point: [1.0, 2.0, 3.0],
        },
    ];
    let got = gpu.evaluate(&ctx, &batch);
    assert_eq!(got.len(), batch.len());
    assert!(
        approx3(as_vec3(got[0]), m.mul_vec3([1.0, 1.0, 1.0])),
        "batch mul_vec3"
    );
    assert!(
        approx_mat3(&as_mat3(got[1]), &m.transpose()),
        "batch transpose"
    );
    assert!(
        approx(as_scalar(got[2]), m.determinant()),
        "batch determinant"
    );
    assert!(
        approx_mat3(&as_mat3(got[3]), &mat3_from_quat(q)),
        "batch mat3_from_quat"
    );
    assert!(
        quat_equiv(as_quat(got[4]), quat_from_mat3(&mat3_from_quat(q))),
        "batch quat_from_mat3"
    );
    assert!(
        approx3(as_vec3(got[5]), sample_mat4().mul_point([1.0, 2.0, 3.0])),
        "batch mul_point"
    );
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMatrixDecompose::new(&ctx);
    // No dispatch is issued and the result vector is empty.
    assert!(gpu.evaluate(&ctx, &[]).is_empty());
}
