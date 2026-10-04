//! Real-device parity for the §24.1 dual-quaternion skinning shader mirror.
//!
//! Each test blends a set of weighted bone dual quaternions and transforms a
//! point on the CPU ([`prism_math::DualQuat::blend_weighted`] +
//! [`transform_point3`]), does the same 4-influence blend on a real `GPU` from
//! the single-sourced
//! [`WGSL_DUAL_QUAT_SKIN`](prism_math::shader_mirror::WGSL_DUAL_QUAT_SKIN) +
//! [`WGSL_QUAT_ROTATE`](prism_math::shader_mirror::WGSL_QUAT_ROTATE) fragments,
//! and asserts the skinned point agrees within a small absolute+relative
//! tolerance. The blend renormalizes (reciprocal square root) and composes
//! Hamilton products; Metal compiles WGSL under fast-math (lower-precision
//! `rsqrt`, FMA contraction / reassociation), so the §24.1 contract is a
//! *tolerance* round-trip, not a bit-exact one. The suite skips gracefully when
//! no adapter is available so it still passes on a device-less CI image while
//! running the full dispatch on a real `GPU`.

use prism_math::DualQuat;
use prism_math::Quat;
use prism_math::Vec3;
use prism_math_gpu::GpuContext;
use prism_math_gpu::{GpuDualQuatSkin, Influence};
use std::f32::consts::FRAC_PI_3;
use std::f32::consts::FRAC_PI_4;

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

/// Absolute+relative tolerance comparison for a single coordinate.
///
/// The blend's `rsqrt` normalize plus the Hamilton-product translation recovery
/// carry more rounding than a plain multiply-add, so the skinning epsilon is a
/// touch looser than the projection suite's `1e-5`: `5e-5` still rejects any
/// real algorithm/operand-order/layout drift while admitting the fast-math
/// rounding.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    let scale = a.abs().max(b.abs()).max(1.0);
    diff <= 5.0e-5 * scale
}

fn assert_vec_close(cpu: Vec3, gpu: Vec3) {
    assert!(
        close(cpu.x, gpu.x) && close(cpu.y, gpu.y) && close(cpu.z, gpu.z),
        "skinned point drifted: cpu=({}, {}, {}) gpu=({}, {}, {})",
        cpu.x, cpu.y, cpu.z, gpu.x, gpu.y, gpu.z,
    );
}

fn dq(axis: Vec3, angle: f32, t: Vec3) -> DualQuat {
    DualQuat::from_rotation_translation(Quat::from_axis_angle(axis.normalize(), angle), t)
}

fn cpu_skin(influences: &[Influence], p: Vec3) -> Vec3 {
    let items: Vec<(DualQuat, f32)> = influences
        .iter()
        .map(|inf| (inf.transform, inf.weight))
        .collect();
    DualQuat::blend_weighted(&items).transform_point3(p)
}

#[test]
fn single_bone_matches_cpu_reference() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuDualQuatSkin::new(&ctx);
    let influences = [Influence {
        transform: dq(Vec3::new(0.0, 1.0, 0.0), FRAC_PI_3, Vec3::new(2.0, -1.0, 3.0)),
        weight: 1.0,
    }];
    let p = Vec3::new(1.0, 2.0, -0.5);
    assert_vec_close(cpu_skin(&influences, p), kernel.skin_point(&ctx, &influences, p));
}

#[test]
fn two_bone_blend_matches_cpu_reference() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuDualQuatSkin::new(&ctx);
    let influences = [
        Influence {
            transform: dq(Vec3::new(0.0, 1.0, 0.0), FRAC_PI_4, Vec3::new(1.0, 0.0, 0.0)),
            weight: 0.6,
        },
        Influence {
            transform: dq(Vec3::new(1.0, 0.0, 0.0), -FRAC_PI_3, Vec3::new(0.0, 2.0, -1.0)),
            weight: 0.4,
        },
    ];
    let p = Vec3::new(0.5, 1.5, 2.0);
    assert_vec_close(cpu_skin(&influences, p), kernel.skin_point(&ctx, &influences, p));
}

#[test]
fn four_bone_blend_matches_cpu_reference() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuDualQuatSkin::new(&ctx);
    let influences = [
        Influence {
            transform: dq(Vec3::new(0.0, 1.0, 0.0), FRAC_PI_4, Vec3::new(1.0, 0.0, 0.0)),
            weight: 0.4,
        },
        Influence {
            transform: dq(Vec3::new(1.0, 0.0, 0.0), FRAC_PI_3, Vec3::new(0.0, 1.0, 0.0)),
            weight: 0.3,
        },
        Influence {
            transform: dq(Vec3::new(0.0, 0.0, 1.0), -FRAC_PI_4, Vec3::new(0.0, 0.0, 1.0)),
            weight: 0.2,
        },
        Influence {
            transform: dq(Vec3::new(1.0, 1.0, 0.0), FRAC_PI_3, Vec3::new(-1.0, 1.0, 0.5)),
            weight: 0.1,
        },
    ];
    let p = Vec3::new(2.0, -1.0, 0.5);
    assert_vec_close(cpu_skin(&influences, p), kernel.skin_point(&ctx, &influences, p));
}

#[test]
fn opposite_hemisphere_blend_matches_cpu_reference() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuDualQuatSkin::new(&ctx);
    // Second bone deliberately in the opposite hemisphere (negated) so the
    // pivot-flip path is exercised on both sides.
    let base = dq(Vec3::new(0.0, 1.0, 0.0), FRAC_PI_3, Vec3::new(1.0, 0.0, 0.0));
    let influences = [
        Influence {
            transform: base,
            weight: 0.5,
        },
        Influence {
            transform: base.negated(),
            weight: 0.5,
        },
    ];
    let p = Vec3::new(1.0, 1.0, 1.0);
    assert_vec_close(cpu_skin(&influences, p), kernel.skin_point(&ctx, &influences, p));
}
