//! Real-device parity: the `GPU` fused MLS-MPM advance must reproduce the `CPU`
//! golden [`prism_physics_core::mpm`] `MpmSolver::advance`, particle for
//! particle, after a multi-step run.
//!
//! This is the capstone MPM parity test. Where the earlier slices pinned each
//! transfer kernel in isolation (constitutive probe, `P2G` scatter, grid update,
//! `G2P` gather), this test closes the loop: it runs the full solver order
//! (clear grid → `P2G` → finalise → gravity → boundary → `G2P` → clamp) for
//! several steps on both paths and compares the final particle state. Because
//! the `GPU` keeps every grid and particle buffer resident across steps, only
//! the initial upload and final read-back cross the bus, so this also exercises
//! the resident-buffer round-trip and the per-step `step_clear` reset.
//!
//! The batch is a compact interior cluster around the domain centre so the
//! stencils overlap heavily (the transfers must agree on shared nodes) yet the
//! advected positions never reach the clamp margin — otherwise the clamp would
//! mask a genuine divergence by pinning both paths to the same wall. Particles
//! carry non-identity deformation gradients, nonzero velocities, and nonzero
//! affine matrices so every arithmetic path runs each step.
//!
//! Errors compound over the multi-step run: the device runs the Jacobi `SVD` in
//! `f32`, reassociates its transfer sums differently from the host, and the
//! `P2G` scatter goes through a fixed-point quantisation. Parity is therefore
//! checked within a looser absolute-plus-relative tolerance than the
//! single-kernel slices, but still tight enough to catch any structural error.
//!
//! On a headless host with no `wgpu` adapter the test skips (with a printed
//! notice) instead of failing.
//!
//! Provenance: the MLS-MPM transfer (Hu et al. 2018), the affine `P2G`/`G2P`
//! conventions (Jiang et al. 2015), and the fixed-corotated / snow plasticity
//! model (Stomakhin et al. 2013) are standard, publicly documented techniques.
//! No Unreal Engine source or derived code.

use glam::{Mat3, Vec3};
use prism_physics_core::mpm::{
    Grid, MaterialPoints, MpmConfig, MpmMaterial, MpmSolver, SnowPlasticity,
};
use prism_physics_gpu::{
    BoundaryMode, GpuContext, GpuMpmStep, StepConfig, StepInputs, StepParticles,
};

/// Time step shared by the `CPU` golden and the `GPU` kernel.
const DT: f32 = 1.0e-3;

/// Number of full steps advanced on both paths.
const STEPS: usize = 12;

/// Gravity acceleration shared by both paths.
fn gravity() -> Vec3 {
    Vec3::new(0.0, -9.81, 0.0)
}

/// Grid geometry shared by the `CPU` golden and the `GPU` kernel.
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

/// One particle's full state.
struct Particle {
    position: Vec3,
    velocity: Vec3,
    affine: Mat3,
    deformation: Mat3,
    mass: f32,
    volume: f32,
    plastic_det: f32,
}

/// Builds the interior particle cluster around the domain centre `(1, 1, 1)`.
///
/// With `dx = 0.5` and thickness `2` the clamp interval is `[0.75, 1.25]` per
/// axis; every particle sits well inside it and the small velocities plus
/// gravity keep them there for the whole run, so the clamp never fires and the
/// parity check measures the transfer arithmetic rather than a shared wall.
fn particles() -> Vec<Particle> {
    vec![
        Particle {
            position: Vec3::new(0.95, 1.00, 1.05),
            velocity: Vec3::new(0.30, -0.20, 0.10),
            affine: Mat3::from_cols(
                Vec3::new(0.20, 0.05, -0.03),
                Vec3::new(-0.04, 0.15, 0.02),
                Vec3::new(0.01, -0.02, 0.18),
            ),
            deformation: Mat3::from_cols(
                Vec3::new(1.12, 0.06, -0.03),
                Vec3::new(0.05, 0.94, 0.04),
                Vec3::new(-0.02, 0.03, 1.08),
            ),
            mass: 1.30,
            volume: 0.05,
            plastic_det: 0.96,
        },
        Particle {
            position: Vec3::new(1.05, 0.95, 1.00),
            velocity: Vec3::new(-0.15, 0.25, -0.05),
            affine: Mat3::from_cols(
                Vec3::new(-0.10, 0.03, 0.05),
                Vec3::new(0.06, -0.12, 0.01),
                Vec3::new(-0.02, 0.04, 0.09),
            ),
            deformation: Mat3::from_cols(
                Vec3::new(0.97, 0.02, 0.05),
                Vec3::new(-0.03, 1.06, 0.02),
                Vec3::new(0.04, -0.01, 0.99),
            ),
            mass: 0.85,
            volume: 0.04,
            plastic_det: 1.05,
        },
        Particle {
            position: Vec3::new(1.00, 1.05, 0.95),
            velocity: Vec3::new(0.05, 0.05, 0.20),
            affine: Mat3::from_cols(
                Vec3::new(0.08, -0.02, 0.03),
                Vec3::new(0.01, 0.11, -0.05),
                Vec3::new(-0.03, 0.02, 0.07),
            ),
            deformation: Mat3::from_cols(
                Vec3::new(1.05, -0.02, 0.02),
                Vec3::new(0.03, 1.01, -0.04),
                Vec3::new(-0.01, 0.02, 0.96),
            ),
            mass: 1.10,
            volume: 0.05,
            plastic_det: 1.02,
        },
        Particle {
            position: Vec3::new(0.90, 1.10, 1.00),
            velocity: Vec3::new(0.10, 0.15, -0.10),
            affine: Mat3::from_cols(
                Vec3::new(0.05, 0.02, -0.01),
                Vec3::new(-0.02, 0.07, 0.03),
                Vec3::new(0.01, -0.03, 0.06),
            ),
            deformation: Mat3::from_cols(
                Vec3::new(1.02, 0.03, -0.01),
                Vec3::new(-0.02, 0.98, 0.02),
                Vec3::new(0.01, -0.02, 1.05),
            ),
            mass: 0.95,
            volume: 0.05,
            plastic_det: 1.10,
        },
        Particle {
            position: Vec3::new(1.10, 0.90, 1.05),
            velocity: Vec3::new(-0.05, -0.10, 0.15),
            affine: Mat3::from_cols(
                Vec3::new(0.03, -0.01, 0.02),
                Vec3::new(0.02, 0.09, -0.02),
                Vec3::new(-0.01, 0.03, 0.05),
            ),
            deformation: Mat3::from_cols(
                Vec3::new(0.99, 0.04, -0.02),
                Vec3::new(0.02, 1.03, 0.03),
                Vec3::new(-0.03, 0.01, 1.01),
            ),
            mass: 1.05,
            volume: 0.05,
            plastic_det: 0.98,
        },
    ]
}

/// Loads the particle batch into a [`MaterialPoints`] `SoA` container.
fn material_points(batch: &[Particle]) -> MaterialPoints {
    let mut mp = MaterialPoints::with_capacity(batch.len());
    for particle in batch {
        mp.spawn(
            particle.position,
            particle.velocity,
            particle.mass,
            particle.volume,
        );
    }
    for (idx, particle) in batch.iter().enumerate() {
        mp.affine_mut()[idx] = particle.affine;
        mp.deformation_mut()[idx] = particle.deformation;
        mp.plastic_det_mut()[idx] = particle.plastic_det;
    }
    mp
}

/// Runs the `CPU` golden `MpmSolver::advance` for [`STEPS`] steps and returns the
/// final particle state.
fn golden(batch: &[Particle], geom: &GridGeometry, plastic: bool) -> StepParticles {
    let mp = material_points(batch);
    let mut cfg = MpmConfig::new(DT, gravity());
    cfg.plastic = plastic;
    cfg.plasticity = SnowPlasticity::default();
    let material = MpmMaterial::default();
    let grid = Grid::new(geom.nx, geom.ny, geom.nz, geom.dx, geom.origin);

    let mut solver = MpmSolver::new(mp, grid, material, cfg);
    solver.advance(STEPS);

    StepParticles {
        positions: solver.particles.positions().to_vec(),
        velocities: solver.particles.velocities().to_vec(),
        affine: solver.particles.affine().to_vec(),
        deformation: solver.particles.deformation().to_vec(),
        plastic_det: solver.particles.plastic_det().to_vec(),
    }
}

/// Runs the `GPU` fused advance for [`STEPS`] steps and returns the final state.
fn device(
    step: &GpuMpmStep,
    ctx: &GpuContext,
    batch: &[Particle],
    geom: &GridGeometry,
    plastic: bool,
) -> StepParticles {
    let positions: Vec<Vec3> = batch.iter().map(|p| p.position).collect();
    let velocities: Vec<Vec3> = batch.iter().map(|p| p.velocity).collect();
    let affine: Vec<Mat3> = batch.iter().map(|p| p.affine).collect();
    let deformation: Vec<Mat3> = batch.iter().map(|p| p.deformation).collect();
    let masses: Vec<f32> = batch.iter().map(|p| p.mass).collect();
    let volumes: Vec<f32> = batch.iter().map(|p| p.volume).collect();
    let plastic_det: Vec<f32> = batch.iter().map(|p| p.plastic_det).collect();

    // The CPU golden takes the material Lamé pair as `(λ, μ)`; the kernel wants
    // `μ0` then `λ0`.
    let (lambda0, mu0) = MpmMaterial::default().lame();
    let plasticity = SnowPlasticity::default();

    let inputs = StepInputs {
        positions: &positions,
        velocities: &velocities,
        affine: &affine,
        deformation: &deformation,
        masses: &masses,
        volumes: &volumes,
        plastic_det: &plastic_det,
    };
    let cfg = StepConfig {
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
        steps: STEPS,
    };
    step.advance(ctx, &inputs, &cfg)
}

/// Returns the worst absolute-plus-relative error between two `Vec3` slices,
/// anchored to a small fraction of the field peak.
fn worst_vec3(gpu: &[Vec3], cpu: &[Vec3]) -> f32 {
    let peak = cpu
        .iter()
        .flat_map(|v| [v.x.abs(), v.y.abs(), v.z.abs()])
        .fold(0.0_f32, f32::max);
    let floor = 1.0e-4 * peak.max(1.0e-6);
    let mut worst = 0.0_f32;
    for n in 0..cpu.len() {
        for c in 0..3 {
            let diff = (gpu[n][c] - cpu[n][c]).abs();
            let scale = floor + gpu[n][c].abs().max(cpu[n][c].abs());
            worst = worst.max(diff / scale);
        }
    }
    worst
}

/// Returns the worst absolute-plus-relative error between two `Mat3` slices.
fn worst_mat3(gpu: &[Mat3], cpu: &[Mat3]) -> f32 {
    let peak = cpu
        .iter()
        .flat_map(|m| {
            let c = m.to_cols_array();
            (0..9).map(move |i| c[i].abs())
        })
        .fold(0.0_f32, f32::max);
    let floor = 1.0e-4 * peak.max(1.0e-6);
    let mut worst = 0.0_f32;
    for n in 0..cpu.len() {
        let g = gpu[n].to_cols_array();
        let c = cpu[n].to_cols_array();
        for i in 0..9 {
            let diff = (g[i] - c[i]).abs();
            let scale = floor + g[i].abs().max(c[i].abs());
            worst = worst.max(diff / scale);
        }
    }
    worst
}

/// Returns the worst absolute-plus-relative error between two `f32` slices.
fn worst_scalar(gpu: &[f32], cpu: &[f32]) -> f32 {
    let peak = cpu.iter().fold(0.0_f32, |m, &v| m.max(v.abs()));
    let floor = 1.0e-4 * peak.max(1.0e-6);
    let mut worst = 0.0_f32;
    for n in 0..cpu.len() {
        let diff = (gpu[n] - cpu[n]).abs();
        let scale = floor + gpu[n].abs().max(cpu[n].abs());
        worst = worst.max(diff / scale);
    }
    worst
}

/// Asserts the `GPU` particle state tracks the `CPU` golden within `tol`.
#[expect(
    clippy::print_stderr,
    reason = "the measured worst-case error must reach the test log"
)]
fn assert_particles_match(gpu: &StepParticles, cpu: &StepParticles, tol: f32, label: &str) {
    assert_eq!(gpu.positions.len(), cpu.positions.len(), "{label}: count");

    let pos = worst_vec3(&gpu.positions, &cpu.positions);
    let vel = worst_vec3(&gpu.velocities, &cpu.velocities);
    let affine = worst_mat3(&gpu.affine, &cpu.affine);
    let deform = worst_mat3(&gpu.deformation, &cpu.deformation);
    let jp = worst_scalar(&gpu.plastic_det, &cpu.plastic_det);

    let worst = pos.max(vel).max(affine).max(deform).max(jp);
    eprintln!(
        "{label}: worst step error = {worst:e} (pos {pos:e}, vel {vel:e}, C {affine:e}, F {deform:e}, Jp {jp:e})"
    );
    assert!(
        worst < tol,
        "{label}: GPU advance diverged from CPU golden (worst = {worst:e}, tol = {tol:e})",
    );
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on adapter-less hosts"
)]
fn gpu_mpm_step_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU MPM step parity: no wgpu adapter on this host");
        return;
    };
    let step = GpuMpmStep::new(&ctx);

    let batch = particles();
    let geom = geometry();

    // Elastic path: the trial deformation is kept as-is, Jp unchanged.
    let gpu_elastic = device(&step, &ctx, &batch, &geom, false);
    let cpu_elastic = golden(&batch, &geom, false);
    assert_particles_match(&gpu_elastic, &cpu_elastic, 5.0e-3, "elastic");

    // Plastic path: the trial deformation is split by the snow return mapping
    // each step, so the f32 SVD error compounds over the run.
    let gpu_plastic = device(&step, &ctx, &batch, &geom, true);
    let cpu_plastic = golden(&batch, &geom, true);
    assert_particles_match(&gpu_plastic, &cpu_plastic, 5.0e-3, "plastic");
}
