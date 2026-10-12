//! Real-device parity for the §24.6 batched spherical-harmonic evaluation
//! kernels.
//!
//! Each test builds a batch of (coefficients, direction) queries, evaluates them
//! on the CPU with [`prism_math::Sh2::eval`] / [`prism_math::Sh3::eval`], runs
//! the same batch on a real `GPU` from the single-sourced
//! [`WGSL_SH3_EVAL`](prism_math::shader_mirror::WGSL_SH3_EVAL) fragment (the
//! order-2 basis is its first 9 bands), and asserts agreement within tolerance.
//!
//! The evaluation is a sum of 9 or 16 polynomial-weighted multiply-adds; Metal
//! fast-math may contract/reassociate, so agreement is a small absolute+relative
//! epsilon. It skips gracefully when no adapter is available.

use prism_math::{Sh2, Sh3, Vec3};
use prism_math_gpu::{GpuContext, GpuShBatch, GpuShQuery};

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

/// Absolute+relative closeness for the multiply-add accumulations.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    diff <= 5.0e-5 + 5.0e-5 * a.abs().max(b.abs())
}

/// A spread of unit query directions (axis-aligned and oblique).
fn dirs() -> [[f32; 3]; 5] {
    [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.577_350_26, 0.577_350_26, 0.577_350_26],
        [0.3, -0.5, 0.8],
    ]
}

fn unit(d: [f32; 3]) -> Vec3 {
    Vec3::new(d[0], d[1], d[2]).normalize()
}

#[test]
fn sh2_batch_matches_cpu() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuShBatch::new(&ctx);

    let coeff_sets: [[f32; 9]; 3] = [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, -0.2, 0.3, -0.4, 0.1, 0.6, -0.3, 0.2, -0.5],
        [1.2, 0.7, -0.9, 0.4, -0.8, 0.35, 1.1, -0.6, 0.9],
    ];
    let ds = dirs();

    let mut queries = Vec::new();
    for c in &coeff_sets {
        for d in &ds {
            let dn = unit(*d);
            queries.push(GpuShQuery::from_coeffs9(c, [dn.x, dn.y, dn.z]));
        }
    }
    let got = kernel.eval2(&ctx, &queries);
    assert_eq!(got.len(), queries.len());

    let mut i = 0;
    for c in &coeff_sets {
        let sh = Sh2::from_coeffs(*c);
        for d in &ds {
            let expected = sh.eval(unit(*d));
            assert!(
                close(got[i], expected),
                "sh2[{i}] gpu={} cpu={expected}",
                got[i]
            );
            i += 1;
        }
    }
}

#[test]
fn sh3_batch_matches_cpu() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuShBatch::new(&ctx);

    let coeff_sets: [[f32; 16]; 2] = [
        [
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
        [
            0.6, 0.4, -0.3, 0.25, -0.2, 0.5, 0.7, -0.15, 0.9, 0.3, -0.8, 0.55, -0.45, 0.65, -0.35,
            0.75,
        ],
    ];
    let ds = dirs();

    let mut queries = Vec::new();
    for c in &coeff_sets {
        for d in &ds {
            let dn = unit(*d);
            queries.push(GpuShQuery::from_coeffs16(c, [dn.x, dn.y, dn.z]));
        }
    }
    let got = kernel.eval3(&ctx, &queries);
    assert_eq!(got.len(), queries.len());

    let mut i = 0;
    for c in &coeff_sets {
        let sh = Sh3::from_coeffs(*c);
        for d in &ds {
            let expected = sh.eval(unit(*d));
            assert!(
                close(got[i], expected),
                "sh3[{i}] gpu={} cpu={expected}",
                got[i]
            );
            i += 1;
        }
    }
}

#[test]
fn empty_batch_returns_empty() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuShBatch::new(&ctx);
    assert!(kernel.eval2(&ctx, &[]).is_empty());
    assert!(kernel.eval3(&ctx, &[]).is_empty());
}
