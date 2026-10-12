//! Real-device parity for the §24.1 GPU quaternion-algebra shader mirror.
//!
//! Each test builds a batch of quaternion operand pairs, evaluates the operation
//! on the CPU with [`prism_math::Quat`] and runs the same op on a real `GPU` from
//! the single-sourced
//! [`WGSL_QUAT_ALG`](prism_math::shader_mirror::WGSL_QUAT_ALG) fragment.
//!
//! The conjugate and inverse are exact sign flips (compared strictly); the
//! Hamilton product and normalization are float arithmetic and agree within a
//! small absolute+relative epsilon (Metal fast-math). It skips gracefully when
//! no adapter is available so it still passes on a device-less CI image while
//! running the full dispatch on a real `GPU`.

use prism_math::Quat;
use prism_math_gpu::{GpuContext, GpuQuatAlg, QuatOp};

/// `1/sqrt(2)`, the axis-angle half-angle sine/cosine for a 90° rotation.
const FRAC_1_SQRT_2: f32 = core::f32::consts::FRAC_1_SQRT_2;

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

/// A spread of quaternions: identity, a rotation, a general unit quat, a
/// non-unit quat (for normalize), and an axis-angle-ish quat.
fn quats() -> [Quat; 5] {
    [
        Quat::from_xyzw(0.0, 0.0, 0.0, 1.0),
        Quat::from_xyzw(0.5, 0.5, 0.5, 0.5),
        Quat::from_xyzw(-0.2, 0.4, -0.6, 0.8),
        Quat::from_xyzw(1.0, 2.0, 3.0, 4.0),
        Quat::from_xyzw(0.0, FRAC_1_SQRT_2, 0.0, FRAC_1_SQRT_2),
    ]
}

#[test]
fn quat_mul_matches_cpu() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuQuatAlg::new(&ctx);
    let qs = quats();
    let pairs: Vec<[[f32; 4]; 2]> = (0..qs.len())
        .map(|i| {
            let a = qs[i];
            let b = qs[(i + 1) % qs.len()];
            [a.to_array(), b.to_array()]
        })
        .collect();

    let got = kernel.map(&ctx, QuatOp::Mul, &pairs);
    assert_eq!(got.len(), qs.len());
    for (i, pair) in pairs.iter().enumerate() {
        let a = Quat::from_xyzw(pair[0][0], pair[0][1], pair[0][2], pair[0][3]);
        let b = Quat::from_xyzw(pair[1][0], pair[1][1], pair[1][2], pair[1][3]);
        let expected = (a * b).to_array();
        assert!(
            close_slice(got[i], expected),
            "mul[{i}] gpu={:?} cpu={expected:?}",
            got[i]
        );
    }
}

#[test]
fn quat_conjugate_inverse_match_cpu() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuQuatAlg::new(&ctx);
    let qs = quats();
    let identity = [0.0, 0.0, 0.0, 0.0];
    let pairs: Vec<[[f32; 4]; 2]> = qs.iter().map(|q| [q.to_array(), identity]).collect();

    let conj = kernel.map(&ctx, QuatOp::Conjugate, &pairs);
    let inv = kernel.map(&ctx, QuatOp::Inverse, &pairs);
    assert_eq!(conj.len(), qs.len());
    assert_eq!(inv.len(), qs.len());
    for (i, q) in qs.iter().enumerate() {
        // Conjugate/inverse are exact sign flips: strict equality.
        assert_eq!(conj[i], q.conjugate().to_array(), "conj[{i}]");
        assert_eq!(inv[i], q.inverse().to_array(), "inv[{i}]");
        assert_eq!(conj[i], inv[i], "conj == inv[{i}]");
    }
}

#[test]
fn quat_normalize_matches_cpu() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuQuatAlg::new(&ctx);
    let qs = quats();
    let identity = [0.0, 0.0, 0.0, 0.0];
    let pairs: Vec<[[f32; 4]; 2]> = qs.iter().map(|q| [q.to_array(), identity]).collect();

    let got = kernel.map(&ctx, QuatOp::Normalize, &pairs);
    assert_eq!(got.len(), qs.len());
    for (i, q) in qs.iter().enumerate() {
        let expected = q.normalize().to_array();
        assert!(
            close_slice(got[i], expected),
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
    let kernel = GpuQuatAlg::new(&ctx);
    assert!(kernel.map(&ctx, QuatOp::Mul, &[]).is_empty());
}
