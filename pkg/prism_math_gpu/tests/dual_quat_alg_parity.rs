//! Real-device parity for the §24.1 GPU dual-quaternion-algebra shader mirror.
//!
//! Each test builds a batch of dual-quaternion operand pairs, evaluates the
//! operation on the CPU with [`prism_math::DualQuat`] and runs the same op on a
//! real `GPU` from the single-sourced
//! [`WGSL_DUAL_QUAT_ALG`](prism_math::shader_mirror::WGSL_DUAL_QUAT_ALG) (plus
//! [`WGSL_QUAT_ALG`](prism_math::shader_mirror::WGSL_QUAT_ALG)) fragments.
//!
//! The conjugate and inverse are exact sign flips (compared strictly); the
//! composition product and normalization are float arithmetic and agree within a
//! small absolute+relative epsilon. It skips gracefully when no adapter is
//! available so it still passes on a device-less CI image while running the full
//! dispatch on a real `GPU`.

use prism_math::{DualQuat, Quat, Vec3};
use prism_math_gpu::{DualQuatOp, GpuContext, GpuDualQuatAlg};

/// Acquires a device, or prints a skip note and returns `None` on hosts without
/// a usable adapter.
#[expect(
    clippy::print_stderr,
    reason = "test-only skip note when no GPU adapter is present"
)]
fn with_gpu() -> Option<GpuContext> {
    match GpuContext::try_headless() {
        Some(ctx) => Some(ctx),
        None => {
            eprintln!("skipping: no usable GPU adapter on this host");
            None
        }
    }
}

/// Absolute+relative closeness for the float ops.
fn close_slice(gpu: [f32; 4], cpu: [f32; 4]) -> bool {
    gpu.iter().zip(cpu).all(|(a, b)| {
        let diff = (a - b).abs();
        diff <= 1.0e-5 + 1.0e-5 * a.abs().max(b.abs())
    })
}

/// A spread of dual quaternions: identity, pure translations, unit rigid
/// transforms, and a non-unit dual quaternion (for normalize).
fn dual_quats() -> [DualQuat; 5] {
    [
        DualQuat::IDENTITY,
        DualQuat::from_translation(Vec3::new(1.0, -2.0, 0.5)),
        DualQuat::from_rotation_translation(
            Quat::from_xyzw(0.0, 0.0, 0.0, 1.0),
            Vec3::new(3.0, 4.0, -5.0),
        ),
        DualQuat::from_rotation_translation(
            Quat::from_xyzw(0.5, 0.5, 0.5, 0.5),
            Vec3::new(-1.0, 2.0, 0.0),
        ),
        // Non-unit: exercises normalize's re-orthogonalization.
        DualQuat::from_real_dual(
            Quat::from_xyzw(1.0, 2.0, 3.0, 4.0),
            Quat::from_xyzw(0.1, 0.2, 0.3, 0.4),
        ),
    ]
}

/// Packs a dual quaternion into the `[real, dual]` lane layout.
fn lanes(dq: DualQuat) -> [[f32; 4]; 2] {
    [dq.real.to_array(), dq.dual.to_array()]
}

#[test]
fn dq_mul_matches_cpu() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuDualQuatAlg::new(&ctx);
    let dqs = dual_quats();
    let pairs: Vec<[[f32; 4]; 4]> = (0..dqs.len())
        .map(|i| {
            let a = lanes(dqs[i]);
            let b = lanes(dqs[(i + 1) % dqs.len()]);
            [a[0], a[1], b[0], b[1]]
        })
        .collect();

    let got = kernel.map(&ctx, DualQuatOp::Mul, &pairs);
    assert_eq!(got.len(), dqs.len());
    for i in 0..dqs.len() {
        let expected = lanes(dqs[i] * dqs[(i + 1) % dqs.len()]);
        assert!(
            close_slice(got[i][0], expected[0]) && close_slice(got[i][1], expected[1]),
            "mul[{i}] gpu={:?} cpu={expected:?}",
            got[i]
        );
    }
}

#[test]
fn dq_conjugate_inverse_match_cpu() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuDualQuatAlg::new(&ctx);
    let dqs = dual_quats();
    let placeholder = [[0.0f32; 4]; 2];
    let pairs: Vec<[[f32; 4]; 4]> = dqs
        .iter()
        .map(|dq| {
            let a = lanes(*dq);
            [a[0], a[1], placeholder[0], placeholder[1]]
        })
        .collect();

    let conj = kernel.map(&ctx, DualQuatOp::Conjugate, &pairs);
    let inv = kernel.map(&ctx, DualQuatOp::Inverse, &pairs);
    for i in 0..dqs.len() {
        let c = lanes(dqs[i].conjugate());
        assert_eq!(conj[i], c, "conj[{i}]");
        assert_eq!(inv[i], c, "inv[{i}]");
    }
}

#[test]
fn dq_normalize_matches_cpu() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuDualQuatAlg::new(&ctx);
    let dqs = dual_quats();
    let placeholder = [[0.0f32; 4]; 2];
    let pairs: Vec<[[f32; 4]; 4]> = dqs
        .iter()
        .map(|dq| {
            let a = lanes(*dq);
            [a[0], a[1], placeholder[0], placeholder[1]]
        })
        .collect();

    let got = kernel.map(&ctx, DualQuatOp::Normalize, &pairs);
    for i in 0..dqs.len() {
        let expected = lanes(dqs[i].normalize());
        assert!(
            close_slice(got[i][0], expected[0]) && close_slice(got[i][1], expected[1]),
            "normalize[{i}] gpu={:?} cpu={expected:?}",
            got[i]
        );
    }
}

#[test]
fn empty_batch_returns_empty() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuDualQuatAlg::new(&ctx);
    assert!(kernel.map(&ctx, DualQuatOp::Mul, &[]).is_empty());
}
