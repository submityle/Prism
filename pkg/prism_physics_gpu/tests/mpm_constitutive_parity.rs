//! Real-device parity: the `GPU` MLS-MPM constitutive probe must reproduce the
//! `CPU` golden constitutive kernels in [`prism_physics_core::mpm`] within a
//! tight tolerance, per particle and across a batch of deformation gradients.
//!
//! The probe isolates the highest-risk MPM arithmetic before it is fused into
//! the transfer pipeline: the fixed-corotated stress `P Fᵀ` (which internally
//! runs the sqrt-based signed `SVD` and the polar rotation), the polar
//! rotation `R` on its own, and the snow return-mapping (the clamped elastic
//! deformation gradient and the updated plastic determinant `Jp`). The batch
//! deliberately mixes the identity, a general non-symmetric deformation, a
//! shear, a reflection (`det F < 0`, exercising the signed-`SVD` sign flip), a
//! strong compression, and a stretch, and runs it both with plasticity
//! disabled and with plasticity plus hardening enabled so the Lamé scaling
//! path is covered.
//!
//! The device reassociates its sums differently from the host (fused
//! multiply-add, differing rounding) and runs the Jacobi `SVD` and the
//! range-reduced exponential in `f32`, so parity is checked within a small
//! absolute-plus-relative tolerance rather than bit-for-bit.
//!
//! On a headless host with no `wgpu` adapter the test skips (with a printed
//! notice) instead of failing.
//!
//! Provenance: the fixed-corotated energy and snow return-mapping (Stomakhin et
//! al. 2013) and the affine MLS-MPM conventions (Hu et al. 2018; Jiang et al.
//! 2015) are standard, publicly documented techniques. No Unreal Engine source
//! or derived code.

use glam::{Mat3, Vec3};
use prism_physics_core::mpm::{
    corotated_pf, hardening_factor, polar_rotation, snow_return_mapping, MpmMaterial,
    SnowPlasticity,
};
use prism_physics_gpu::{ConstitutiveOutput, GpuContext, GpuMpmConstitutive};

/// The batch of trial deformation gradients exercised by the parity test.
fn deformation_batch() -> Vec<Mat3> {
    vec![
        // Identity: rotation is identity, stress is zero, snow leaves F intact.
        Mat3::IDENTITY,
        // A general, mildly anisotropic deformation with off-diagonal terms.
        Mat3::from_cols(
            Vec3::new(1.20, 0.10, -0.05),
            Vec3::new(0.08, 0.95, 0.06),
            Vec3::new(-0.04, 0.03, 1.10),
        ),
        // A simple shear (unit determinant, nontrivial rotation).
        Mat3::from_cols(
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.25, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ),
        // A reflection: negative determinant exercises the signed-SVD sign flip.
        Mat3::from_cols(
            Vec3::new(-1.05, 0.02, 0.0),
            Vec3::new(0.03, 0.98, 0.01),
            Vec3::new(0.0, 0.02, 1.04),
        ),
        // Strong isotropic compression (clamped hard by the snow floor).
        Mat3::from_cols(
            Vec3::new(0.80, 0.0, 0.0),
            Vec3::new(0.0, 0.82, 0.0),
            Vec3::new(0.0, 0.0, 0.85),
        ),
        // Strong stretch (clamped by the snow ceiling).
        Mat3::from_cols(
            Vec3::new(1.30, 0.05, 0.0),
            Vec3::new(0.0, 1.25, 0.02),
            Vec3::new(0.01, 0.0, 1.35),
        ),
    ]
}

/// Per-particle plastic determinants covering under- and over-compaction.
fn plastic_dets(count: usize) -> Vec<f32> {
    let seeds = [1.0_f32, 0.90, 1.10, 0.75, 1.20, 0.98];
    (0..count).map(|i| seeds[i % seeds.len()]).collect()
}

/// Largest absolute-plus-relative element error between two matrices.
fn matrix_error(a: Mat3, b: Mat3) -> f32 {
    let cols = [
        (a.x_axis, b.x_axis),
        (a.y_axis, b.y_axis),
        (a.z_axis, b.z_axis),
    ];
    let mut worst = 0.0_f32;
    for (ca, cb) in cols {
        for k in 0..3 {
            let diff = (ca[k] - cb[k]).abs();
            let scale = 1.0 + ca[k].abs().max(cb[k].abs());
            worst = worst.max(diff / scale);
        }
    }
    worst
}

/// Computes the CPU golden constitutive outputs for one configuration.
fn golden(
    deformations: &[Mat3],
    dets: &[f32],
    mu0: f32,
    lambda0: f32,
    plasticity: &SnowPlasticity,
    plastic: bool,
) -> ConstitutiveOutput {
    let mut pf = Vec::with_capacity(deformations.len());
    let mut polar = Vec::with_capacity(deformations.len());
    let mut f_elastic = Vec::with_capacity(deformations.len());
    let mut plastic_det = Vec::with_capacity(deformations.len());
    for (f, &jp) in deformations.iter().zip(dets) {
        let harden = if plastic {
            hardening_factor(plasticity.hardening, jp)
        } else {
            1.0
        };
        pf.push(corotated_pf(*f, mu0 * harden, lambda0 * harden));
        polar.push(polar_rotation(*f));
        let upd = snow_return_mapping(*f, jp, plasticity);
        f_elastic.push(upd.deformation);
        plastic_det.push(upd.plastic_det);
    }
    ConstitutiveOutput {
        pf,
        polar,
        f_elastic,
        plastic_det,
    }
}

/// Asserts that a `GPU` output batch tracks the `CPU` golden within `tol`.
#[expect(
    clippy::print_stderr,
    reason = "the measured worst-case error must reach the test log"
)]
fn assert_batch_matches(gpu: &ConstitutiveOutput, cpu: &ConstitutiveOutput, tol: f32, label: &str) {
    assert_eq!(gpu.pf.len(), cpu.pf.len(), "{label}: pf length mismatch");
    let mut worst = 0.0_f32;
    for i in 0..cpu.pf.len() {
        worst = worst.max(matrix_error(gpu.pf[i], cpu.pf[i]));
        worst = worst.max(matrix_error(gpu.polar[i], cpu.polar[i]));
        worst = worst.max(matrix_error(gpu.f_elastic[i], cpu.f_elastic[i]));
        let jp_diff = (gpu.plastic_det[i] - cpu.plastic_det[i]).abs();
        let jp_scale = 1.0 + cpu.plastic_det[i].abs();
        worst = worst.max(jp_diff / jp_scale);
    }
    eprintln!("{label}: worst constitutive error = {worst:e}");
    assert!(
        worst < tol,
        "{label}: GPU constitutive probe diverged from CPU golden (worst = {worst:e}, tol = {tol:e})",
    );
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on adapter-less hosts"
)]
fn gpu_mpm_constitutive_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU MPM constitutive parity: no wgpu adapter on this host");
        return;
    };
    let probe = GpuMpmConstitutive::new(&ctx);

    let deformations = deformation_batch();
    let dets = plastic_dets(deformations.len());
    let material = MpmMaterial::default();
    let (lambda0, mu0) = material.lame();
    let plasticity = SnowPlasticity::default();

    // Elastic-only: plasticity disabled, Lamé parameters used unscaled.
    let gpu_elastic = probe.evaluate(
        &ctx,
        &deformations,
        &dets,
        mu0,
        lambda0,
        plasticity.hardening,
        plasticity.critical_compression,
        plasticity.critical_stretch,
        false,
    );
    let cpu_elastic = golden(&deformations, &dets, mu0, lambda0, &plasticity, false);
    assert_batch_matches(&gpu_elastic, &cpu_elastic, 2.0e-4, "elastic");

    // Plastic + hardening: the Lamé parameters are scaled per particle by
    // `hardening_factor(ξ, Jp)`, exercising the range-reduced exponential.
    let gpu_plastic = probe.evaluate(
        &ctx,
        &deformations,
        &dets,
        mu0,
        lambda0,
        plasticity.hardening,
        plasticity.critical_compression,
        plasticity.critical_stretch,
        true,
    );
    let cpu_plastic = golden(&deformations, &dets, mu0, lambda0, &plasticity, true);
    assert_batch_matches(&gpu_plastic, &cpu_plastic, 2.0e-4, "plastic");
}
