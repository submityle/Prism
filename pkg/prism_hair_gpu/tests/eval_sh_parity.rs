//! Real-device parity for the isolated hair second-order `SH` transmittance
//! reconstruction twin: [`GpuHairEvalSh`] must reproduce the `CPU` golden
//! [`reference_eval_sh`](prism_hair_gpu::eval_sh::reference_eval_sh)
//! (built on
//! [`eval_sh`](prism_render_architecture::hair::dual_scatter_sh::eval_sh)
//! composed with
//! [`sh_basis`](prism_render_architecture::hair::dual_scatter_sh::sh_basis))
//! for a batch of query directions against one shared nine-coefficient set. The
//! suite drives a constant-only lobe (direction-independent), the three signed
//! coordinate axes, a degenerate zero direction collapsing to the pole,
//! non-finite directions sanitising to the same value, a coefficient set whose
//! raw reconstruction goes negative and must clamp to zero, a mixed batch, the
//! empty no-op, and a large multi-workgroup batch that crosses the 64-wide
//! dispatch boundary.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-`WGSL`,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each reconstructed value is a `dot`, a `sqrt` and a divide chain a `GPU` may
//! fuse, so every value is asserted within `abs_diff < 1e-4` or
//! `rel_diff < 1e-3`. Beyond matching the golden element by element, every
//! output is asserted finite and non-negative (the clamped-transmittance
//! invariant). No `sin`/`cos` appears anywhere; all directions are explicit
//! literals or integer-derived fractions and all comparisons use a tolerance,
//! never a float `==`/`!=`.
//!
//! Provenance: standard orthonormal real spherical-harmonic reconstruction plus
//! `wgpu` compute dispatch; no third-party engine source or derived code.

use prism_hair_gpu::eval_sh::{reference_eval_sh, GpuHairEvalSh};
use prism_hair_gpu::GpuContext;
use prism_render_architecture::hair::dual_scatter_sh::{ShCoeffs, Vec3};

/// Acquires a headless context, or `None` (with a skip notice) when the host has
/// no `wgpu` adapter so the suite stays green off-device.
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn context_or_skip(label: &str) -> Option<GpuContext> {
    match GpuContext::try_headless() {
        Some(ctx) => Some(ctx),
        None => {
            eprintln!("skipping {label}: no wgpu adapter on this host");
            None
        }
    }
}

/// Asserts two scalars agree within the documented fma tolerance.
fn assert_close(got: f32, want: f32, what: &str) {
    let abs = (got - want).abs();
    let rel = abs / want.abs().max(1.0);
    assert!(
        abs < 1e-4 || rel < 1e-3,
        "{what}: got {got}, want {want} (abs {abs}, rel {rel})"
    );
}

/// Asserts a whole batch matches the `CPU` golden direction by direction, and
/// that every output is finite and non-negative.
fn assert_batch(got: &[f32], coeffs: ShCoeffs, dirs: &[Vec3]) {
    assert_eq!(got.len(), dirs.len(), "one value per direction");
    for (i, &dir) in dirs.iter().enumerate() {
        let want = reference_eval_sh(coeffs, dir);
        assert_close(got[i], want, &format!("direction {i}"));
        assert!(
            got[i].is_finite() && got[i] >= 0.0,
            "direction {i} must be finite and non-negative, got {}",
            got[i]
        );
    }
}

/// Dispatches one batch through the device twin.
fn run(ctx: &GpuContext, coeffs: ShCoeffs, dirs: &[Vec3]) -> Vec<f32> {
    GpuHairEvalSh::new(ctx).eval(ctx, coeffs, dirs)
}

#[test]
fn constant_lobe_is_direction_independent() {
    let Some(ctx) = context_or_skip("constant_lobe_is_direction_independent") else {
        return;
    };
    // Only the Y00 term is non-zero, so the reconstruction is the constant
    // 0.282095 * c0 along every (non-degenerate) direction.
    let coeffs = ShCoeffs {
        c: [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    };
    let dirs = [
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 2.0, 0.0),
        Vec3::new(-3.0, 0.0, 4.0),
        Vec3::new(1.0, 1.0, 1.0),
    ];
    let got = run(&ctx, coeffs, &dirs);
    assert_batch(&got, coeffs, &dirs);
    let expected = 0.282095 * 2.0;
    for (i, &v) in got.iter().enumerate() {
        assert_close(v, expected, &format!("constant lobe {i}"));
    }
}

#[test]
fn signed_coordinate_axes_match_golden() {
    let Some(ctx) = context_or_skip("signed_coordinate_axes_match_golden") else {
        return;
    };
    // A full set of coefficients evaluated along the six signed axes exercises
    // every linear and quadratic basis term independently.
    let coeffs = ShCoeffs {
        c: [0.5, 0.3, -0.2, 0.4, 0.1, -0.15, 0.25, 0.05, -0.1],
    };
    let dirs = [
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(-1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(0.0, 0.0, -1.0),
    ];
    assert_batch(&run(&ctx, coeffs, &dirs), coeffs, &dirs);
}

#[test]
fn unnormalized_direction_matches_golden() {
    let Some(ctx) = context_or_skip("unnormalized_direction_matches_golden") else {
        return;
    };
    // The shader normalises each direction internally (3-4-5 and 1-2-2 triples
    // give exact unit lengths of 5 and 3), so an unnormalised input must match
    // the golden, which normalises the same way.
    let coeffs = ShCoeffs {
        c: [0.4, 0.2, 0.1, -0.3, 0.2, 0.1, -0.25, 0.15, 0.05],
    };
    let dirs = [
        Vec3::new(3.0, 4.0, 0.0),
        Vec3::new(0.0, 3.0, 4.0),
        Vec3::new(1.0, 2.0, 2.0),
        Vec3::new(-2.0, -1.0, 2.0),
    ];
    assert_batch(&run(&ctx, coeffs, &dirs), coeffs, &dirs);
}

#[test]
fn degenerate_direction_collapses_to_pole() {
    let Some(ctx) = context_or_skip("degenerate_direction_collapses_to_pole") else {
        return;
    };
    // A zero-length direction normalises to zero on both sides; the basis there
    // leaves only Y00 and Y20 (= 0.315392 * (3*0 - 1)) non-zero. The device must
    // reproduce the golden's degenerate value exactly within tolerance.
    let coeffs = ShCoeffs {
        c: [1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 2.0, 0.5, 0.5],
    };
    let dirs = [Vec3::new(0.0, 0.0, 0.0)];
    assert_batch(&run(&ctx, coeffs, &dirs), coeffs, &dirs);
}

#[test]
fn non_finite_direction_sanitizes() {
    let Some(ctx) = context_or_skip("non_finite_direction_sanitizes") else {
        return;
    };
    // NaN/+inf/-inf components each sanitise to 0 (collapsing to the pole) exactly
    // like the golden, and the result stays finite and non-negative.
    let coeffs = ShCoeffs {
        c: [0.6, 0.2, 0.1, 0.1, 0.0, 0.0, 0.3, 0.0, 0.0],
    };
    let dirs = [
        Vec3::new(f32::NAN, 0.0, 1.0),
        Vec3::new(0.0, f32::INFINITY, 0.0),
        Vec3::new(f32::NEG_INFINITY, f32::NAN, f32::INFINITY),
        Vec3::new(0.0, 0.0, 1.0),
    ];
    let got = run(&ctx, coeffs, &dirs);
    assert_batch(&got, coeffs, &dirs);
    assert!(
        got.iter().all(|v| v.is_finite()),
        "values must stay finite for non-finite directions, got {got:?}"
    );
}

#[test]
fn negative_reconstruction_clamps_to_zero() {
    let Some(ctx) = context_or_skip("negative_reconstruction_clamps_to_zero") else {
        return;
    };
    // A negative Y00 coefficient with the rest zero drives the raw reconstruction
    // negative; transmittance is non-negative, so both the golden and the device
    // clamp to exactly 0.
    let coeffs = ShCoeffs {
        c: [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    };
    let dirs = [
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
    ];
    let got = run(&ctx, coeffs, &dirs);
    assert_batch(&got, coeffs, &dirs);
    for (i, &v) in got.iter().enumerate() {
        assert_close(v, 0.0, &format!("clamped lobe {i}"));
    }
}

#[test]
fn mixed_batch_matches_golden() {
    let Some(ctx) = context_or_skip("mixed_batch_matches_golden") else {
        return;
    };
    // A single batch mixing axes, unnormalised, degenerate and non-finite
    // directions preserves order and matches the golden direction by direction.
    let coeffs = ShCoeffs {
        c: [0.7, -0.3, 0.2, 0.4, -0.1, 0.15, 0.3, -0.2, 0.1],
    };
    let dirs = [
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(3.0, 4.0, 0.0),
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(-1.0, 2.0, -2.0),
        Vec3::new(f32::NAN, 1.0, 0.0),
        Vec3::new(1.0, -1.0, 1.0),
    ];
    assert_batch(&run(&ctx, coeffs, &dirs), coeffs, &dirs);
}

#[test]
fn empty_batch_is_noop() {
    let Some(ctx) = context_or_skip("empty_batch_is_noop") else {
        return;
    };
    let coeffs = ShCoeffs {
        c: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    };
    let got = run(&ctx, coeffs, &[]);
    assert!(got.is_empty(), "empty batch yields no values");
}

#[test]
fn large_batch_crosses_workgroup_boundary() {
    let Some(ctx) = context_or_skip("large_batch_crosses_workgroup_boundary") else {
        return;
    };
    // 130 directions span three 64-wide workgroups over a deterministic sweep of
    // integer-derived components (with every 13th slot forced non-finite to
    // exercise the sanitiser across the dispatch boundary).
    let coeffs = ShCoeffs {
        c: [0.5, 0.2, -0.1, 0.3, 0.1, -0.05, 0.2, 0.1, -0.15],
    };
    let mut dirs = Vec::new();
    for k in 0u32..130 {
        if k % 13 == 0 {
            dirs.push(Vec3::new(f32::NAN, f32::INFINITY, f32::NEG_INFINITY));
        } else {
            let x = ((k % 7) as f32) - 3.0;
            let y = ((k % 5) as f32) - 2.0;
            let z = ((k % 9) as f32) - 4.0;
            dirs.push(Vec3::new(x, y, z));
        }
    }
    assert_batch(&run(&ctx, coeffs, &dirs), coeffs, &dirs);
}
