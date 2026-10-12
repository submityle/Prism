//! Real-device parity for the §24.1 GPU TRS affine-transform shader mirror.
//!
//! Each test applies a per-element `(scale, rotation, translation)` to a batch of
//! points/vectors on the CPU with [`prism_math::Affine3`] and on a real `GPU`
//! from the single-sourced
//! [`WGSL_AFFINE`](prism_math::shader_mirror::WGSL_AFFINE) fragment, asserting
//! agreement within a small tolerance.
//!
//! It skips gracefully when no adapter is available.

use prism_math::{Affine3, Quat, Vec3};
use prism_math_gpu::{affine_item, GpuAffine3, GpuContext};

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

/// TRS fixtures: unit and non-unit scales, several rotations, translations.
fn fixtures() -> [(Vec3, Quat, Vec3, Vec3); 5] {
    [
        (
            Vec3::ONE,
            Quat::IDENTITY,
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(0.5, -0.5, 0.25),
        ),
        (
            Vec3::new(2.0, 0.5, 1.5),
            Quat::from_xyzw(0.2, -0.3, 0.1, 0.9).normalize(),
            Vec3::new(-2.0, 4.0, 0.0),
            Vec3::new(1.0, 1.0, -1.0),
        ),
        (
            Vec3::new(0.25, 3.0, 1.0),
            Quat::from_xyzw(0.707_106_77, 0.0, 0.0, 0.707_106_77),
            Vec3::new(0.0, -1.0, 5.0),
            Vec3::new(-3.0, 2.0, 0.5),
        ),
        (
            Vec3::new(1.0, 1.0, 1.0),
            Quat::from_xyzw(0.0, 0.6, 0.8, 0.0),
            Vec3::ZERO,
            Vec3::new(2.0, -2.0, 2.0),
        ),
        (
            Vec3::new(-1.0, 2.0, 0.5),
            Quat::from_xyzw(0.1, 0.2, 0.3, 0.927_361_9),
            Vec3::new(7.0, -6.0, 5.0),
            Vec3::new(0.0, 0.0, 0.0),
        ),
    ]
}

#[test]
fn affine_transform_points_matches_cpu() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuAffine3::new(&ctx);
    let fx = fixtures();
    let items: Vec<_> = fx
        .iter()
        .map(|(s, q, t, p)| {
            affine_item(
                [s.x, s.y, s.z],
                [q.x, q.y, q.z, q.w],
                [t.x, t.y, t.z],
                [p.x, p.y, p.z],
            )
        })
        .collect();

    let got = kernel.transform_points(&ctx, &items);
    assert_eq!(got.len(), fx.len());
    for (i, (s, q, t, p)) in fx.iter().enumerate() {
        let a = Affine3::from_scale_rotation_translation(*s, *q, *t);
        let c = a.transform_point3(*p);
        for (axis, (g, e)) in got[i].iter().zip([c.x, c.y, c.z]).enumerate() {
            assert!(close(*g, e), "point[{i}][{axis}] gpu={g} cpu={e}");
        }
    }
}

#[test]
fn affine_transform_vectors_matches_cpu() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuAffine3::new(&ctx);
    let fx = fixtures();
    let items: Vec<_> = fx
        .iter()
        .map(|(s, q, t, v)| {
            affine_item(
                [s.x, s.y, s.z],
                [q.x, q.y, q.z, q.w],
                [t.x, t.y, t.z],
                [v.x, v.y, v.z],
            )
        })
        .collect();

    let got = kernel.transform_vectors(&ctx, &items);
    assert_eq!(got.len(), fx.len());
    for (i, (s, q, t, v)) in fx.iter().enumerate() {
        let a = Affine3::from_scale_rotation_translation(*s, *q, *t);
        let c = a.transform_vector3(*v);
        for (axis, (g, e)) in got[i].iter().zip([c.x, c.y, c.z]).enumerate() {
            assert!(close(*g, e), "vector[{i}][{axis}] gpu={g} cpu={e}");
        }
    }
}

#[test]
fn empty_batch_returns_empty() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuAffine3::new(&ctx);
    assert!(kernel.transform_points(&ctx, &[]).is_empty());
    assert!(kernel.transform_vectors(&ctx, &[]).is_empty());
}
