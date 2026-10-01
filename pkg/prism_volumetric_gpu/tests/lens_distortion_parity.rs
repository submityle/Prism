//! Real-device parity for the `Brown-Conrady` lens-distortion twin:
//! [`GpuLensDistortion`](prism_volumetric_gpu::lens_distortion::GpuLensDistortion)
//! must reproduce the `CPU` golden
//! [`lens_distortion`](prism_render_architecture::particle::lens_distortion)
//! across the forward map
//! [`distort`](prism_render_architecture::particle::lens_distortion::distort),
//! the fixed-point inverse
//! [`undistort`](prism_render_architecture::particle::lens_distortion::undistort)
//! and the scalar radial map
//! [`distort_radius`](prism_render_architecture::particle::lens_distortion::distort_radius),
//! over barrel, pincushion and decentred lenses, a grid of normalized
//! coordinates spanning every quadrant plus the optical centre, and a batch of
//! radii — each compared point-for-point.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernels are portable core-`WGSL`, so they need no optional device
//! feature.
//!
//! # Parity criterion
//!
//! Each output coordinate is a fixed, non-reorderable sequence of multiplies,
//! adds and (for the inverse) divides, so `CPU` and `GPU` evaluate the same
//! closed form in the same order. They are not bit-exact: a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place, and across the ten inverse
//! iterations that perturbation compounds. The comparison therefore allows
//! `abs_diff <= 1e-4` or `rel_diff <= 1e-3` — loose enough to admit a legal
//! fused multiply-add contraction, yet tight enough to fail a genuinely wrong
//! port (a swapped tangential sign, a dropped radial term, a wrong iteration
//! count).
//!
//! # Fixture conditioning
//!
//! The coefficients are the physically moderate barrel/pincushion/decentred
//! lenses the `CPU` reference itself tests, and every sample point stays inside
//! `|coord| <= 0.5` (radius `<= ~0.6`). There the radial factor stays close to
//! one, so the inverse division never approaches zero and no fixture sits in a
//! degenerate region; there are no buckets, so no tie handling is involved.
//!
//! Provenance: standard `Brown-Conrady` lens distortion; no third-party engine
//! source or derived code.

#![expect(
    clippy::print_stderr,
    reason = "test prints a skip notice when no GPU adapter is available"
)]

use prism_render_architecture::particle::lens_distortion::{
    distort, distort_radius, undistort, DistortionCoeffs,
};
use prism_volumetric_gpu::lens_distortion::GpuLensDistortion;
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound. A `GPU` may fuse a multiply-add the scalar reference
/// leaves separate, perturbing the low mantissa bits by a few units in the last
/// place; `1e-4` admits that legal slack, compounded over the inverse
/// iterations, while still failing a genuinely wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in
/// the last place exceed the absolute floor.
const REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
}

/// A representative barrel lens (negative leading radial term), matching the
/// `CPU` reference fixture.
fn barrel() -> DistortionCoeffs {
    DistortionCoeffs::new(-0.25, 0.06, -0.01, 0.0, 0.0)
}

/// A representative pincushion lens (positive leading radial term), matching the
/// `CPU` reference fixture.
fn pincushion() -> DistortionCoeffs {
    DistortionCoeffs::new(0.2, 0.03, 0.0, 0.0, 0.0)
}

/// A lens with a decentring (tangential) component, matching the `CPU`
/// reference fixture.
fn decentred() -> DistortionCoeffs {
    DistortionCoeffs::new(-0.15, 0.02, 0.0, 0.01, -0.008)
}

/// The sample grid of normalized coordinates. Every quadrant, both axes and the
/// optical centre, all inside `|coord| <= 0.5` so the radial factor stays well
/// conditioned.
fn sample_points() -> Vec<[f32; 2]> {
    let axis = [-0.5f32, -0.35, -0.2, -0.05, 0.0, 0.05, 0.2, 0.35, 0.5];
    let mut points = Vec::new();
    for &x in &axis {
        for &y in &axis {
            points.push([x, y]);
        }
    }
    points
}

/// The sample radii for the scalar radial map, inside the well-conditioned
/// range.
fn sample_radii() -> Vec<f32> {
    vec![0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
}

/// Asserts the `GPU` forward map matches the `CPU` golden point-for-point.
fn check_distort(ctx: &GpuContext, gpu: &GpuLensDistortion, coeffs: &DistortionCoeffs) {
    let points = sample_points();
    let got = gpu.distort(ctx, coeffs, &points);
    assert_eq!(got.len(), points.len(), "point count must match the input");
    for (idx, (g, &p)) in got.iter().zip(points.iter()).enumerate() {
        let want = distort(coeffs, p);
        for channel in 0..2 {
            assert!(
                close(g[channel], want[channel]),
                "distort point {idx} channel {channel}: gpu {} vs cpu {}",
                g[channel],
                want[channel]
            );
        }
    }
}

/// Asserts the `GPU` inverse map matches the `CPU` golden point-for-point.
fn check_undistort(ctx: &GpuContext, gpu: &GpuLensDistortion, coeffs: &DistortionCoeffs) {
    let points = sample_points();
    let got = gpu.undistort(ctx, coeffs, &points);
    assert_eq!(got.len(), points.len(), "point count must match the input");
    for (idx, (g, &p)) in got.iter().zip(points.iter()).enumerate() {
        let want = undistort(coeffs, p);
        for channel in 0..2 {
            assert!(
                close(g[channel], want[channel]),
                "undistort point {idx} channel {channel}: gpu {} vs cpu {}",
                g[channel],
                want[channel]
            );
        }
    }
}

#[test]
fn distort_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping lens_distortion_parity: no GPU adapter available");
        return;
    };
    let gpu = GpuLensDistortion::new(&ctx);
    for coeffs in [barrel(), pincushion(), decentred()] {
        check_distort(&ctx, &gpu, &coeffs);
    }
}

#[test]
fn undistort_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping lens_distortion_parity: no GPU adapter available");
        return;
    };
    let gpu = GpuLensDistortion::new(&ctx);
    for coeffs in [barrel(), pincushion(), decentred()] {
        check_undistort(&ctx, &gpu, &coeffs);
    }
}

#[test]
fn distort_radius_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping lens_distortion_parity: no GPU adapter available");
        return;
    };
    let gpu = GpuLensDistortion::new(&ctx);
    let radii = sample_radii();
    for coeffs in [barrel(), pincushion(), decentred()] {
        let got = gpu.distort_radius(&ctx, &coeffs, &radii);
        assert_eq!(got.len(), radii.len(), "radius count must match the input");
        for (idx, (&g, &r)) in got.iter().zip(radii.iter()).enumerate() {
            let want = distort_radius(&coeffs, r);
            assert!(
                close(g, want),
                "distort_radius {idx}: gpu {g} vs cpu {want} (r {r})"
            );
        }
    }
}

#[test]
fn forward_inverse_round_trip_on_device() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping lens_distortion_parity: no GPU adapter available");
        return;
    };
    let gpu = GpuLensDistortion::new(&ctx);
    // Distorting then undistorting on-device must return the original points to
    // within tolerance, confirming the two kernels invert each other and agree
    // with the reference's own round-trip property.
    let coeffs = decentred();
    let points = sample_points();
    let there = gpu.distort(&ctx, &coeffs, &points);
    let back = gpu.undistort(&ctx, &coeffs, &there);
    assert_eq!(back.len(), points.len());
    for (idx, (b, &p)) in back.iter().zip(points.iter()).enumerate() {
        for channel in 0..2 {
            assert!(
                close(b[channel], p[channel]),
                "round trip point {idx} channel {channel}: {} vs {}",
                b[channel],
                p[channel]
            );
        }
    }
}

#[test]
fn empty_input_is_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping lens_distortion_parity: no GPU adapter available");
        return;
    };
    let gpu = GpuLensDistortion::new(&ctx);
    let coeffs = barrel();
    // An empty slice must yield an empty result with no dispatch issued, since a
    // storage buffer cannot be zero-sized.
    let empty: [[f32; 2]; 0] = [];
    assert!(gpu.distort(&ctx, &coeffs, &empty).is_empty());
    assert!(gpu.undistort(&ctx, &coeffs, &empty).is_empty());
    assert!(gpu.distort_radius(&ctx, &coeffs, &[]).is_empty());
}
