//! Real-device parity for the resident, GPU-driven MLS-MPM solver.
//!
//! [`GpuMpmResident`] keeps the particle and grid buffers live on the device
//! across command submissions so a real-time loop can upload once, step every
//! frame, and only read the state back when it renders. This test pins the two
//! invariants that make that split trustworthy:
//!
//! 1. **One-shot equivalence.** A resident run of `k` steps returns exactly the
//!    same particle state as a single [`GpuMpmStep::advance`] of `k` steps, so
//!    the resident path introduces no numerical drift of its own — both drive
//!    the identical fused kernels over the identical uniform layout.
//! 2. **Cross-submission residency.** Splitting those `k` steps across several
//!    separate `step` calls (distinct command submissions, i.e. frames) with a
//!    non-destructive `snapshot` in between leaves the final state unchanged,
//!    proving the device buffers genuinely carry the simulation forward without
//!    a host round-trip.
//!
//! A third case drives a hundred-thousand-particle batch to confirm the
//! resident path scales to the milestone particle count and produces finite,
//! in-domain state; it reports the wall-clock time for the stepped submission.
//!
//! On a headless host with no `wgpu` adapter every case skips (with a printed
//! notice) instead of failing.
//!
//! Provenance: the MLS-MPM transfer (Hu et al. 2018), the affine `P2G`/`G2P`
//! conventions (Jiang et al. 2015), and the fixed-corotated / snow plasticity
//! model (Stomakhin et al. 2013) are standard, publicly documented techniques.
//! No Unreal Engine source or derived code.

use glam::{Mat3, Vec3};
use prism_physics_core::mpm::{MpmMaterial, SnowPlasticity};
use prism_physics_gpu::{
    BoundaryMode, GpuContext, GpuMpmResident, GpuMpmStep, StepConfig, StepInputs, StepParticles,
};

/// Time step shared by every case.
const DT: f32 = 1.0e-3;

/// Gravity acceleration shared by every case.
fn gravity() -> Vec3 {
    Vec3::new(0.0, -9.81, 0.0)
}

/// Grid geometry shared by the interior-cluster cases.
struct GridGeometry {
    nx: usize,
    ny: usize,
    nz: usize,
    dx: f32,
    origin: Vec3,
}

/// A `6x6x6` lattice matching the transfer-slice geometry.
fn geometry() -> GridGeometry {
    GridGeometry {
        nx: 6,
        ny: 6,
        nz: 6,
        dx: 0.5,
        origin: Vec3::new(-0.25, -0.25, -0.25),
    }
}

/// The uploaded particle batch as owned host arrays.
struct Batch {
    positions: Vec<Vec3>,
    velocities: Vec<Vec3>,
    affine: Vec<Mat3>,
    deformation: Vec<Mat3>,
    masses: Vec<f32>,
    volumes: Vec<f32>,
    plastic_det: Vec<f32>,
}

impl Batch {
    /// Borrows the batch as [`StepInputs`].
    fn inputs(&self) -> StepInputs<'_> {
        StepInputs {
            positions: &self.positions,
            velocities: &self.velocities,
            affine: &self.affine,
            deformation: &self.deformation,
            masses: &self.masses,
            volumes: &self.volumes,
            plastic_det: &self.plastic_det,
        }
    }
}

/// The compact interior cluster around the domain centre `(1, 1, 1)`, matching
/// the full-step parity test so the batch exercises every arithmetic path.
fn cluster() -> Batch {
    let positions = vec![
        Vec3::new(0.95, 1.00, 1.05),
        Vec3::new(1.05, 0.95, 1.00),
        Vec3::new(1.00, 1.05, 0.95),
        Vec3::new(0.90, 1.10, 1.00),
        Vec3::new(1.10, 0.90, 1.05),
    ];
    let velocities = vec![
        Vec3::new(0.30, -0.20, 0.10),
        Vec3::new(-0.15, 0.25, -0.05),
        Vec3::new(0.05, 0.05, 0.20),
        Vec3::new(0.10, 0.15, -0.10),
        Vec3::new(-0.05, -0.10, 0.15),
    ];
    let affine = vec![
        Mat3::from_cols(
            Vec3::new(0.20, 0.05, -0.03),
            Vec3::new(-0.04, 0.15, 0.02),
            Vec3::new(0.01, -0.02, 0.18),
        ),
        Mat3::from_cols(
            Vec3::new(-0.10, 0.03, 0.05),
            Vec3::new(0.06, -0.12, 0.01),
            Vec3::new(-0.02, 0.04, 0.09),
        ),
        Mat3::from_cols(
            Vec3::new(0.08, -0.02, 0.03),
            Vec3::new(0.01, 0.11, -0.05),
            Vec3::new(-0.03, 0.02, 0.07),
        ),
        Mat3::from_cols(
            Vec3::new(0.05, 0.02, -0.01),
            Vec3::new(-0.02, 0.07, 0.03),
            Vec3::new(0.01, -0.03, 0.06),
        ),
        Mat3::from_cols(
            Vec3::new(0.03, -0.01, 0.02),
            Vec3::new(0.02, 0.09, -0.02),
            Vec3::new(-0.01, 0.03, 0.05),
        ),
    ];
    let deformation = vec![
        Mat3::from_cols(
            Vec3::new(1.12, 0.06, -0.03),
            Vec3::new(0.05, 0.94, 0.04),
            Vec3::new(-0.02, 0.03, 1.08),
        ),
        Mat3::from_cols(
            Vec3::new(0.97, 0.02, 0.05),
            Vec3::new(-0.03, 1.06, 0.02),
            Vec3::new(0.04, -0.01, 0.99),
        ),
        Mat3::from_cols(
            Vec3::new(1.05, -0.02, 0.02),
            Vec3::new(0.03, 1.01, -0.04),
            Vec3::new(-0.01, 0.02, 0.96),
        ),
        Mat3::from_cols(
            Vec3::new(1.02, 0.03, -0.01),
            Vec3::new(-0.02, 0.98, 0.02),
            Vec3::new(0.01, -0.02, 1.05),
        ),
        Mat3::from_cols(
            Vec3::new(0.99, 0.04, -0.02),
            Vec3::new(0.02, 1.03, 0.03),
            Vec3::new(-0.03, 0.01, 1.01),
        ),
    ];
    let masses = vec![1.30, 0.85, 1.10, 0.95, 1.05];
    let volumes = vec![0.05, 0.04, 0.05, 0.05, 0.05];
    let plastic_det = vec![0.96, 1.05, 1.02, 1.10, 0.98];
    Batch {
        positions,
        velocities,
        affine,
        deformation,
        masses,
        volumes,
        plastic_det,
    }
}

/// Builds the shared [`StepConfig`] for `geom`, advancing `steps` steps with the
/// snow plasticity toggle `plastic`.
fn config(geom: &GridGeometry, plastic: bool, steps: usize) -> StepConfig {
    let (lambda0, mu0) = MpmMaterial::default().lame();
    let plasticity = SnowPlasticity::default();
    StepConfig {
        dims: (geom.nx, geom.ny, geom.nz),
        origin: geom.origin,
        dx: geom.dx,
        dt: DT,
        gravity: gravity(),
        mu0,
        lambda0,
        hardening: plasticity.hardening,
        theta_c: plasticity.critical_compression,
        theta_s: plasticity.critical_stretch,
        boundary: BoundaryMode::Slip,
        boundary_thickness: 2,
        plastic,
        steps,
    }
}

/// Returns the worst absolute difference between two particle states across
/// every field (positions, velocities, affine `C`, deformation `F`, `Jp`).
fn worst_diff(a: &StepParticles, b: &StepParticles) -> f32 {
    assert_eq!(a.positions.len(), b.positions.len(), "particle count");
    let mut worst = 0.0_f32;
    for i in 0..a.positions.len() {
        for c in 0..3 {
            worst = worst.max((a.positions[i][c] - b.positions[i][c]).abs());
            worst = worst.max((a.velocities[i][c] - b.velocities[i][c]).abs());
        }
        let (ca, cb) = (a.affine[i].to_cols_array(), b.affine[i].to_cols_array());
        let (da, db) = (
            a.deformation[i].to_cols_array(),
            b.deformation[i].to_cols_array(),
        );
        for k in 0..9 {
            worst = worst.max((ca[k] - cb[k]).abs());
            worst = worst.max((da[k] - db[k]).abs());
        }
        worst = worst.max((a.plastic_det[i] - b.plastic_det[i]).abs());
    }
    worst
}

/// The resident path of `k` steps must reproduce a one-shot advance of `k`
/// steps exactly (same kernels, same uniform layout, same device).
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice and measured error must reach the test log"
)]
fn resident_matches_one_shot_advance() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping resident MPM parity: no wgpu adapter on this host");
        return;
    };
    let geom = geometry();
    let batch = cluster();

    for plastic in [false, true] {
        let steps = 12;
        let cfg = config(&geom, plastic, steps);

        let step = GpuMpmStep::new(&ctx);
        let one_shot = step.advance(&ctx, &batch.inputs(), &cfg);

        let mut resident = GpuMpmResident::new(&ctx);
        resident.upload(&ctx, &batch.inputs(), &cfg);
        resident.step(&ctx, steps);
        let got = resident.snapshot(&ctx);

        let worst = worst_diff(&got, &one_shot);
        let label = if plastic { "plastic" } else { "elastic" };
        eprintln!("resident vs one-shot ({label}): worst abs diff = {worst:e}");
        assert!(
            worst < 1.0e-6,
            "{label}: resident advance diverged from one-shot (worst = {worst:e})",
        );
    }
}

/// Splitting the advance across separate `step` submissions with a
/// non-destructive `snapshot` in between must leave the final state identical to
/// a single advance of the combined step count.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice and measured error must reach the test log"
)]
fn resident_persists_across_submissions() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping resident MPM residency: no wgpu adapter on this host");
        return;
    };
    let geom = geometry();
    let batch = cluster();
    let cfg_full = config(&geom, true, 12);

    let step = GpuMpmStep::new(&ctx);
    let one_shot = step.advance(&ctx, &batch.inputs(), &cfg_full);

    let mut resident = GpuMpmResident::new(&ctx);
    resident.upload(&ctx, &batch.inputs(), &cfg_full);
    // First frame: six steps, then a non-destructive read-back.
    resident.step(&ctx, 6);
    let mid = resident.snapshot(&ctx);
    // Second frame: six more steps continuing from the resident state.
    resident.step(&ctx, 6);
    let end = resident.snapshot(&ctx);

    // The mid snapshot must not have disturbed the resident buffers: the tail
    // six steps still land on the full twelve-step state.
    let worst_end = worst_diff(&end, &one_shot);
    eprintln!("resident split 6+6 vs one-shot 12: worst abs diff = {worst_end:e}");
    assert!(
        worst_end < 1.0e-6,
        "resident residency diverged after a mid snapshot (worst = {worst_end:e})",
    );

    // Sanity: the mid snapshot is genuinely a partial state, not the final one.
    let mid_vs_end = worst_diff(&mid, &one_shot);
    assert!(
        mid_vs_end > 1.0e-6,
        "mid snapshot unexpectedly equals the final state (diff = {mid_vs_end:e})",
    );
}

/// The resident solver must scale to the milestone particle count and produce
/// finite, in-domain state.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice and throughput report must reach the test log"
)]
fn resident_scales_to_many_particles() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping resident MPM scale: no wgpu adapter on this host");
        return;
    };
    // A larger domain so a hundred-thousand-particle block fits well inside the
    // clamp margin.
    let geom = GridGeometry {
        nx: 32,
        ny: 32,
        nz: 32,
        dx: 0.5,
        origin: Vec3::new(-0.25, -0.25, -0.25),
    };
    let cfg = config(&geom, false, 8);

    // A jittered lattice of particles inside the safe interior sub-box
    // [3, 12]^3, well clear of the clamp margin.
    let side = 47_usize; // 47^3 = 103_823 particles.
    let count = side * side * side;
    let lo = 3.0_f32;
    let span = 9.0_f32;
    let mut positions = Vec::with_capacity(count);
    let mut velocities = Vec::with_capacity(count);
    for iz in 0..side {
        for iy in 0..side {
            for ix in 0..side {
                let fx = ix as f32 / (side - 1) as f32;
                let fy = iy as f32 / (side - 1) as f32;
                let fz = iz as f32 / (side - 1) as f32;
                positions.push(Vec3::new(lo + fx * span, lo + fy * span, lo + fz * span));
                // A gentle swirl so the transfers do real work.
                velocities.push(Vec3::new(fy - 0.5, 0.5 - fx, fz - 0.5) * 0.2);
            }
        }
    }
    let affine = vec![Mat3::ZERO; count];
    let deformation = vec![Mat3::IDENTITY; count];
    let masses = vec![1.0_f32; count];
    let volumes = vec![0.01_f32; count];
    let plastic_det = vec![1.0_f32; count];
    let batch = Batch {
        positions,
        velocities,
        affine,
        deformation,
        masses,
        volumes,
        plastic_det,
    };

    let mut resident = GpuMpmResident::new(&ctx);
    resident.upload(&ctx, &batch.inputs(), &cfg);
    assert_eq!(resident.particle_count(), count);

    let start = std::time::Instant::now();
    resident.step(&ctx, cfg.steps);
    let out = resident.snapshot(&ctx);
    let elapsed = start.elapsed();
    eprintln!(
        "resident MPM: {count} particles x {} steps in {:.1} ms",
        cfg.steps,
        elapsed.as_secs_f64() * 1.0e3,
    );

    assert_eq!(out.positions.len(), count);
    // Every particle must stay finite and within the domain interior clamp.
    let margin = 2.0 * geom.dx;
    let dom_lo = geom.origin + Vec3::splat(margin);
    let dom_hi = geom.origin
        + Vec3::new(
            (geom.nx - 1) as f32 * geom.dx,
            (geom.ny - 1) as f32 * geom.dx,
            (geom.nz - 1) as f32 * geom.dx,
        )
        - Vec3::splat(margin);
    for p in &out.positions {
        assert!(p.is_finite(), "position went non-finite: {p:?}");
        for c in 0..3 {
            assert!(
                p[c] >= dom_lo[c] - 1.0e-3 && p[c] <= dom_hi[c] + 1.0e-3,
                "particle left the domain interior: {p:?}",
            );
        }
    }
    for v in &out.velocities {
        assert!(v.is_finite(), "velocity went non-finite: {v:?}");
    }
}
