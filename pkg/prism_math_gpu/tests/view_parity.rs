//! Real-device parity for the §24.1 view (look-at) matrix shader mirror.
//!
//! Each test builds a right-handed camera view matrix with a CPU constructor
//! ([`prism_math::projection::look_at_rh`] / [`look_to_rh`]), builds the same
//! matrix on a real `GPU` from the single-sourced
//! [`WGSL_LOOK_AT_RH`](prism_math::shader_mirror::WGSL_LOOK_AT_RH) fragment, and
//! asserts the 16 entries agree within a small absolute+relative tolerance. The
//! basis derivation normalizes (reciprocal square root) and takes cross
//! products; Metal compiles WGSL under fast-math (lower-precision `rsqrt`, FMA
//! contraction / reassociation), so the §24.1 contract is a *tolerance*
//! round-trip, not a bit-exact one. The suite skips gracefully when no adapter
//! is available so it still passes on a device-less CI image while running the
//! full dispatch on a real `GPU`.

use prism_math::Mat4;
use prism_math::Vec3;
use prism_math::projection::{look_at_rh, look_to_rh};
use prism_math_gpu::GpuContext;
use prism_math_gpu::GpuView;

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
/// The basis derivation's `rsqrt` carries more rounding than a plain multiply
/// and add, so the view epsilon is a touch looser than the projection suite's
/// `1e-5`: `5e-5` still rejects any real formula/operand-order/layout drift
/// while admitting the fast-math normalize/cross rounding.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    let scale = a.abs().max(b.abs()).max(1.0);
    diff <= 5.0e-5 * scale
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
fn look_at_rh_matches_cpu_reference() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuView::new(&ctx);
    let cases = [
        (
            Vec3::new(0.0, 0.0, 5.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ),
        (
            Vec3::new(3.0, 4.0, 5.0),
            Vec3::new(1.0, 0.0, -2.0),
            Vec3::new(0.0, 1.0, 0.0),
        ),
        (
            Vec3::new(-7.0, 2.5, 1.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ),
    ];
    for (eye, target, up) in cases {
        let cpu = look_at_rh(eye, target, up);
        let gpu = kernel.look_at_rh(&ctx, eye, target, up);
        assert_mat_close(cpu, gpu);
    }
}

#[test]
fn look_to_rh_matches_cpu_reference() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuView::new(&ctx);
    let cases = [
        (
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::new(0.0, 1.0, 0.0),
        ),
        (
            Vec3::new(10.0, -3.0, 2.0),
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::new(0.0, 1.0, 0.0),
        ),
        (
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ),
    ];
    for (eye, dir, up) in cases {
        let cpu = look_to_rh(eye, dir, up);
        let gpu = kernel.look_to_rh(&ctx, eye, dir, up);
        assert_mat_close(cpu, gpu);
    }
}
