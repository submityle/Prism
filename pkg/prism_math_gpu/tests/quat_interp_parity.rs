//! Real-device parity for the §24.1 GPU quaternion-interpolation shader mirror.
//!
//! Each test builds a batch of unit-quaternion pairs and interpolation factors,
//! blends them on the CPU with [`prism_math::Quat::slerp`] / [`Quat::nlerp`],
//! runs the same blend on a real `GPU` from the single-sourced
//! [`WGSL_QUAT_INTERP`](prism_math::shader_mirror::WGSL_QUAT_INTERP) fragment,
//! and asserts the results agree within a small tolerance (up to the global
//! `q`/`-q` double-cover sign, which denotes the same rotation).
//!
//! `slerp` evaluates `acos`/`sin`/division and `nlerp` evaluates `normalize`;
//! Metal compiles WGSL under fast-math while the CPU routes transcendentals
//! through `libm`, so the twin agrees within a small absolute+relative epsilon,
//! not bit-for-bit. It skips gracefully when no adapter is available so it still
//! passes on a device-less CI image while running the full dispatch on a real
//! `GPU`.

use prism_math::{Quat, Vec3};
use prism_math_gpu::{GpuContext, GpuQuatInterp};

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

/// Absolute+relative closeness, matching the fast-math tolerance contract.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    diff <= 1.0e-4 + 1.0e-4 * a.abs().max(b.abs())
}

/// Compares two quaternions up to the `q`/`-q` double-cover sign.
fn assert_quat_close(gpu: Quat, cpu: Quat, tag: &str) {
    let dot = gpu.x * cpu.x + gpu.y * cpu.y + gpu.z * cpu.z + gpu.w * cpu.w;
    // Canonicalize sign: `q` and `-q` are the same rotation.
    let gpu = if dot < 0.0 {
        Quat::from_xyzw(-gpu.x, -gpu.y, -gpu.z, -gpu.w)
    } else {
        gpu
    };
    assert!(
        close(gpu.x, cpu.x)
            && close(gpu.y, cpu.y)
            && close(gpu.z, cpu.z)
            && close(gpu.w, cpu.w),
        "{tag}: quat mismatch gpu=({},{},{},{}) cpu=({},{},{},{})",
        gpu.x,
        gpu.y,
        gpu.z,
        gpu.w,
        cpu.x,
        cpu.y,
        cpu.z,
        cpu.w
    );
}

/// A spread of unit quaternions spanning several axes and angles, including a
/// near-colinear pair (small angular separation) to exercise the `slerp`→
/// `nlerp` fallback branch, plus identity passthrough.
fn sample_pairs() -> (Vec<Quat>, Vec<Quat>, Vec<f32>) {
    let axes = [
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(1.0, 1.0, 0.0).normalize(),
        Vec3::new(1.0, 2.0, 3.0).normalize(),
        Vec3::new(-2.0, 1.0, 1.0).normalize(),
    ];
    // Angles chosen well away from the π wrap and the fallback threshold so the
    // discrete branch choice is unambiguous between CPU and GPU.
    let angle_a = [0.0, 0.3, 0.9, 1.5, 2.2, 2.8];
    let angle_b = [1.1, 1.7, 2.4, 0.6, 0.2, 2.9];
    let ts = [0.0, 0.25, 0.5, 0.75, 1.0, 0.4];

    let mut a = Vec::new();
    let mut b = Vec::new();
    let mut t = Vec::new();
    for i in 0..axes.len() {
        a.push(Quat::from_axis_angle(axes[i], angle_a[i]));
        b.push(Quat::from_axis_angle(axes[i], angle_b[i]));
        t.push(ts[i]);
    }
    // Near-colinear pair about the same axis with a tiny angular gap → forces
    // the `slerp` colinear fallback to `nlerp`.
    let axis = Vec3::new(0.0, 1.0, 0.0);
    a.push(Quat::from_axis_angle(axis, 0.500));
    b.push(Quat::from_axis_angle(axis, 0.5005));
    t.push(0.5);
    // Identity → rotation passthrough.
    a.push(Quat::IDENTITY);
    b.push(Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), 1.2));
    t.push(0.33);
    (a, b, t)
}

#[test]
fn slerp_matches_cpu() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let interp = GpuQuatInterp::new(&ctx);
    let (a, b, t) = sample_pairs();
    let gpu = interp.slerp(&ctx, &a, &b, &t);
    assert_eq!(gpu.len(), a.len());
    for i in 0..a.len() {
        let cpu = a[i].slerp(b[i], t[i]);
        assert_quat_close(gpu[i], cpu, &format!("slerp[{i}]"));
    }
}

#[test]
fn nlerp_matches_cpu() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let interp = GpuQuatInterp::new(&ctx);
    let (a, b, t) = sample_pairs();
    let gpu = interp.nlerp(&ctx, &a, &b, &t);
    assert_eq!(gpu.len(), a.len());
    for i in 0..a.len() {
        let cpu = a[i].nlerp(b[i], t[i]);
        assert_quat_close(gpu[i], cpu, &format!("nlerp[{i}]"));
    }
}

#[test]
fn large_batch_crosses_workgroups() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let interp = GpuQuatInterp::new(&ctx);
    // Deterministic spread across many workgroups (> 64) without any direct
    // transcendental calls in the test itself.
    let axis = Vec3::new(1.0, 2.0, 2.0).normalize();
    let mut a = Vec::new();
    let mut b = Vec::new();
    let mut t = Vec::new();
    for i in 0..257u32 {
        let fa = (i % 7) as f32 * 0.3;
        let fb = (i % 5) as f32 * 0.4 + 0.2;
        a.push(Quat::from_axis_angle(axis, fa));
        b.push(Quat::from_axis_angle(axis, fb));
        t.push((i % 11) as f32 / 10.0);
    }
    let gpu = interp.slerp(&ctx, &a, &b, &t);
    assert_eq!(gpu.len(), a.len());
    for i in 0..a.len() {
        let cpu = a[i].slerp(b[i], t[i]);
        assert_quat_close(gpu[i], cpu, &format!("batch[{i}]"));
    }
}

#[test]
fn empty_batches_return_empty() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let interp = GpuQuatInterp::new(&ctx);
    assert!(interp.slerp(&ctx, &[], &[], &[]).is_empty());
    assert!(interp.nlerp(&ctx, &[], &[], &[]).is_empty());
}
