//! Real-device parity for the §24.1 projection-matrix shader mirror.
//!
//! Each test builds a projection matrix with a CPU constructor
//! ([`prism_math::projection`]), builds the same matrix on a real `GPU` from the
//! single-sourced [`WGSL_PROJECTION_RH`] fragment, and asserts the 16 entries
//! agree within a small absolute+relative tolerance. The builders divide and
//! multiply floats and Metal compiles WGSL under fast-math (FMA contraction /
//! reassociation), so the §24.1 contract is a *tolerance* round-trip, not a
//! bit-exact one. The suite skips gracefully when no adapter is available so it
//! still passes on a device-less CI image while running the full dispatch on a
//! real `GPU`.

use prism_math::Mat4;
use prism_math::projection::{orthographic_rh, perspective_reverse_z_rh, perspective_rh};
use prism_math_gpu::GpuContext;
use prism_math_gpu::GpuProjection;
use std::f32::consts::FRAC_PI_2;

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

/// Absolute+relative tolerance comparison for a single matrix entry.
///
/// Projection entries span small reciprocals (`1/(far-near)`) to order-unity
/// focal terms, so an epsilon scaled by the operand magnitude is the right
/// yardstick; `1e-5` admits last-ULP FMA rounding while rejecting any real
/// formula/operand-order/layout drift.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    let scale = a.abs().max(b.abs()).max(1.0);
    diff <= 1.0e-5 * scale
}

fn assert_mat_close(cpu: Mat4, gpu: Mat4) {
    let c = [cpu.x_axis, cpu.y_axis, cpu.z_axis, cpu.w_axis];
    let g = [gpu.x_axis, gpu.y_axis, gpu.z_axis, gpu.w_axis];
    for (col, (cc, gg)) in c.iter().zip(g.iter()).enumerate() {
        assert!(
            close(cc.x, gg.x) && close(cc.y, gg.y) && close(cc.z, gg.z) && close(cc.w, gg.w),
            "column {col} drifted: cpu=({}, {}, {}, {}) gpu=({}, {}, {}, {})",
            cc.x, cc.y, cc.z, cc.w, gg.x, gg.y, gg.z, gg.w,
        );
    }
}

#[test]
fn perspective_rh_matches_cpu_reference() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuProjection::new(&ctx);
    let cases = [
        (FRAC_PI_2, 16.0 / 9.0, 0.1, 100.0),
        (1.0, 1.0, 1.0, 1000.0),
        (2.0, 21.0 / 9.0, 0.05, 500.0),
    ];
    for (fovy, aspect, near, far) in cases {
        let cpu = perspective_rh(fovy, aspect, near, far);
        let gpu = kernel.perspective_rh(&ctx, fovy, aspect, near, far);
        assert_mat_close(cpu, gpu);
    }
}

#[test]
fn perspective_reverse_z_rh_matches_cpu_reference() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuProjection::new(&ctx);
    let cases = [
        (FRAC_PI_2, 16.0 / 9.0, 0.1, 1000.0),
        (1.2, 4.0 / 3.0, 0.01, 10_000.0),
    ];
    for (fovy, aspect, near, far) in cases {
        let cpu = perspective_reverse_z_rh(fovy, aspect, near, far);
        let gpu = kernel.perspective_reverse_z_rh(&ctx, fovy, aspect, near, far);
        assert_mat_close(cpu, gpu);
    }
}

#[test]
fn orthographic_rh_matches_cpu_reference() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuProjection::new(&ctx);
    let cases = [
        (-2.0, 2.0, -1.0, 1.0, 1.0, 11.0),
        (0.0, 1920.0, 0.0, 1080.0, 0.1, 50.0),
        (-10.0, 10.0, -10.0, 10.0, -5.0, 5.0),
    ];
    for (l, r, b, t, near, far) in cases {
        let cpu = orthographic_rh(l, r, b, t, near, far);
        let gpu = kernel.orthographic_rh(&ctx, l, r, b, t, near, far);
        assert_mat_close(cpu, gpu);
    }
}
