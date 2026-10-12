//! Real-device parity for the §24.1 GPU 4x4 transform shader mirror.
//!
//! Each test transforms a batch of points/vectors on the CPU with
//! [`prism_math::Mat4`] and on a real `GPU` from the single-sourced
//! [`WGSL_TRANSFORM`](prism_math::shader_mirror::WGSL_TRANSFORM) fragment,
//! asserting agreement within a small tolerance.
//!
//! It skips gracefully when no adapter is available.

use prism_math::projection::perspective_rh;
use prism_math::{Mat4, Quat, Vec3};
use prism_math_gpu::{GpuContext, GpuMat4};

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

/// Absolute+relative closeness.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    diff <= 1.0e-5 + 1.0e-5 * a.abs().max(b.abs())
}

/// The four column-major columns of `m`, matching the CPU `Mat4` layout.
fn cols(m: Mat4) -> [[f32; 4]; 4] {
    [
        m.x_axis.to_array(),
        m.y_axis.to_array(),
        m.z_axis.to_array(),
        m.w_axis.to_array(),
    ]
}

/// An affine TRS matrix (no perspective divide) and a perspective matrix (which
/// exercises the `w` divide).
fn matrices() -> [Mat4; 2] {
    [
        Mat4::from_scale_rotation_translation(
            Vec3::new(2.0, 0.5, 1.5),
            Quat::from_xyzw(0.2, -0.3, 0.1, 0.9).normalize(),
            Vec3::new(3.0, -2.0, 5.0),
        ),
        perspective_rh(core::f32::consts::FRAC_PI_4, 16.0 / 9.0, 0.1, 100.0),
    ]
}

/// Points in front of the camera plane (all `z < 0`), so the perspective
/// matrix's `w = -z` divide stays finite and sign-definite.
fn points() -> [[f32; 3]; 5] {
    [
        [0.0, 0.0, -1.0],
        [1.0, 2.0, -3.0],
        [-4.0, 0.5, -0.5],
        [10.0, -10.0, -10.0],
        [0.25, -0.75, -2.5],
    ]
}

#[test]
fn transform_points_matches_cpu() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuMat4::new(&ctx);
    let ps = points();
    for m in matrices() {
        let got = kernel.transform_points(&ctx, &cols(m), &ps);
        assert_eq!(got.len(), ps.len());
        for (i, p) in ps.iter().enumerate() {
            let c = m.transform_point3(Vec3::new(p[0], p[1], p[2]));
            for (axis, (g, e)) in got[i].iter().zip([c.x, c.y, c.z]).enumerate() {
                assert!(close(*g, e), "point transform[{i}][{axis}] gpu={g} cpu={e}");
            }
        }
    }
}

#[test]
fn transform_vectors_matches_cpu() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuMat4::new(&ctx);
    let vs = points();
    for m in matrices() {
        let got = kernel.transform_vectors(&ctx, &cols(m), &vs);
        assert_eq!(got.len(), vs.len());
        for (i, v) in vs.iter().enumerate() {
            let c = m.transform_vector3(Vec3::new(v[0], v[1], v[2]));
            for (axis, (g, e)) in got[i].iter().zip([c.x, c.y, c.z]).enumerate() {
                assert!(
                    close(*g, e),
                    "vector transform[{i}][{axis}] gpu={g} cpu={e}"
                );
            }
        }
    }
}

#[test]
fn empty_batch_returns_empty() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuMat4::new(&ctx);
    assert!(kernel
        .transform_points(&ctx, &cols(Mat4::IDENTITY), &[])
        .is_empty());
}
