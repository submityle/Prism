//! Real-device parity: the `GPU` MLS-MPM grid-to-particle (`G2P`) affine gather
//! must reproduce the `CPU` golden [`prism_physics_core::mpm`] `grid_to_particle`
//! transfer, particle for particle, within a tight tolerance.
//!
//! The gather is the final transfer kernel: from the finalised node velocities
//! it reconstructs each particle's `APIC` velocity and affine matrix `C`,
//! advects the particle by `x += dt · vel`, updates the deformation gradient
//! `F_trial = (I + dt · C) · F`, and (when plasticity is enabled) applies the
//! snow return mapping to split `F_trial` into an elastic deformation and a new
//! plastic determinant `Jp`.
//!
//! To produce a physically meaningful finalised velocity field, the batch is
//! first run through the `CPU` `P2G` scatter, then the grid is finalised, has
//! gravity added, and its wall boundary applied — exactly the state the solver
//! hands to `G2P`. That single `f32` velocity field feeds both the `CPU` golden
//! gather and the `GPU` kernel, so the parity check isolates the gather itself.
//!
//! The batch deliberately drives every arithmetic path: particles carry
//! non-identity deformation gradients (so the trial gradient is non-trivial and,
//! in the plastic case, the sqrt-based signed `SVD` return mapping runs),
//! nonzero velocities and affine matrices feed the initial `P2G`, and they are
//! placed so their stencils gather from shared nodes and some stencils reach the
//! boundary layer. Both the elastic and the plastic paths are covered.
//!
//! The device runs the Jacobi `SVD` in `f32` and reassociates its gather sums
//! differently from the host, so parity is checked within a small
//! absolute-plus-relative tolerance rather than bit-for-bit.
//!
//! On a headless host with no `wgpu` adapter the test skips (with a printed
//! notice) instead of failing.
//!
//! Provenance: the `APIC` gather (Jiang et al. 2015), the MLS-MPM deformation
//! update (Hu et al. 2018), and the snow return mapping (Stomakhin et al. 2013)
//! are standard, publicly documented techniques. No Unreal Engine source or
//! derived code.

use glam::{Mat3, Vec3};
use prism_physics_core::mpm::{
    apply_grid_boundary, grid_to_particle, particle_to_grid, Grid, MaterialPoints, MpmConfig,
    MpmMaterial, SnowPlasticity,
};
use prism_physics_gpu::{G2pParticles, GpuContext, GpuMpmG2p};

/// Time step shared by the `CPU` golden and the `GPU` kernel.
const DT: f32 = 1.0e-3;

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

impl GridGeometry {
    /// The number of grid nodes.
    fn node_count(&self) -> usize {
        self.nx * self.ny * self.nz
    }
}

/// A `6x6x6` lattice matching the `P2G` parity geometry.
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

/// Builds the test particle batch. Positions overlap on shared interior nodes
/// and the last particle's stencil reaches the grid boundary, so the gather sees
/// both interior and boundary-affected node velocities.
fn particles() -> Vec<Particle> {
    vec![
        Particle {
            position: Vec3::new(1.10, 1.05, 0.95),
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
            position: Vec3::new(1.28, 1.22, 1.05),
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
            position: Vec3::new(0.90, 1.40, 1.30),
            velocity: Vec3::new(0.05, 0.05, 0.40),
            affine: Mat3::from_cols(
                Vec3::new(0.08, -0.02, 0.03),
                Vec3::new(0.01, 0.11, -0.05),
                Vec3::new(-0.03, 0.02, -0.07),
            ),
            deformation: Mat3::from_cols(
                Vec3::new(1.04, -0.05, 0.02),
                Vec3::new(0.03, 1.01, -0.04),
                Vec3::new(-0.01, 0.02, 1.03),
            ),
            mass: 1.05,
            volume: 0.045,
            plastic_det: 0.90,
        },
        Particle {
            position: Vec3::new(0.35, 0.40, 0.45),
            velocity: Vec3::new(0.10, 0.15, -0.20),
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

/// Runs `P2G` + finalise + gravity + boundary and returns the finalised node
/// velocity field the solver hands to `G2P`, in flat index order.
fn finalised_grid_velocity(batch: &[Particle], geom: &GridGeometry, plastic: bool) -> Vec<Vec3> {
    let mp = material_points(batch);
    let mut cfg = MpmConfig::new(DT, gravity());
    cfg.plastic = plastic;
    cfg.plasticity = SnowPlasticity::default();
    let material = MpmMaterial::default();

    let mut grid = Grid::new(geom.nx, geom.ny, geom.nz, geom.dx, geom.origin);
    particle_to_grid(&mp, &mut grid, &cfg, &material);
    grid.finalize_velocity();
    grid.add_velocity_to_active(cfg.gravity * cfg.dt);
    apply_grid_boundary(&mut grid, &cfg);

    let mut velocity = vec![Vec3::ZERO; geom.node_count()];
    for k in 0..geom.nz {
        for j in 0..geom.ny {
            for i in 0..geom.nx {
                let flat = grid.flat(i, j, k);
                velocity[flat] = grid.velocity_at(i, j, k);
            }
        }
    }
    velocity
}

/// Runs the `CPU` golden `G2P` gather against the shared finalised velocity
/// field and returns the updated particle state.
fn golden(
    batch: &[Particle],
    geom: &GridGeometry,
    velocity: &[Vec3],
    plastic: bool,
) -> G2pParticles {
    let mut mp = material_points(batch);
    let mut cfg = MpmConfig::new(DT, gravity());
    cfg.plastic = plastic;
    cfg.plasticity = SnowPlasticity::default();
    let material = MpmMaterial::default();

    // Seed a fresh grid with the shared finalised velocity field via
    // `set_velocity`, so the CPU gather reads the exact same node velocities the
    // GPU kernel consumes. `mass_at` is left zero, which G2P never reads.
    let mut grid = Grid::new(geom.nx, geom.ny, geom.nz, geom.dx, geom.origin);
    for k in 0..geom.nz {
        for j in 0..geom.ny {
            for i in 0..geom.nx {
                let flat = grid.flat(i, j, k);
                grid.set_velocity(i, j, k, velocity[flat]);
            }
        }
    }

    grid_to_particle(&mut mp, &grid, &cfg, &material);

    G2pParticles {
        positions: mp.positions().to_vec(),
        velocities: mp.velocities().to_vec(),
        affine: mp.affine().to_vec(),
        deformation: mp.deformation().to_vec(),
        plastic_det: mp.plastic_det().to_vec(),
    }
}

/// Runs the `GPU` `G2P` gather against the shared finalised velocity field.
fn device(
    gather: &GpuMpmG2p,
    ctx: &GpuContext,
    batch: &[Particle],
    geom: &GridGeometry,
    velocity: &[Vec3],
    plastic: bool,
) -> G2pParticles {
    let positions: Vec<Vec3> = batch.iter().map(|p| p.position).collect();
    let deformation: Vec<Mat3> = batch.iter().map(|p| p.deformation).collect();
    let plastic_det: Vec<f32> = batch.iter().map(|p| p.plastic_det).collect();
    let plasticity = SnowPlasticity::default();
    gather.gather(
        ctx,
        &positions,
        &deformation,
        &plastic_det,
        velocity,
        geom.nx,
        geom.ny,
        geom.nz,
        geom.origin,
        geom.dx,
        DT,
        plasticity.critical_compression,
        plasticity.critical_stretch,
        plastic,
    )
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
fn assert_particles_match(gpu: &G2pParticles, cpu: &G2pParticles, tol: f32, label: &str) {
    assert_eq!(gpu.positions.len(), cpu.positions.len(), "{label}: count");

    let pos = worst_vec3(&gpu.positions, &cpu.positions);
    let vel = worst_vec3(&gpu.velocities, &cpu.velocities);
    let affine = worst_mat3(&gpu.affine, &cpu.affine);
    let deform = worst_mat3(&gpu.deformation, &cpu.deformation);
    let jp = worst_scalar(&gpu.plastic_det, &cpu.plastic_det);

    let worst = pos.max(vel).max(affine).max(deform).max(jp);
    eprintln!(
        "{label}: worst G2P error = {worst:e} (pos {pos:e}, vel {vel:e}, C {affine:e}, F {deform:e}, Jp {jp:e})"
    );
    assert!(
        worst < tol,
        "{label}: GPU G2P gather diverged from CPU golden (worst = {worst:e}, tol = {tol:e})",
    );
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on adapter-less hosts"
)]
fn gpu_mpm_g2p_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU MPM G2P parity: no wgpu adapter on this host");
        return;
    };
    let gather = GpuMpmG2p::new(&ctx);

    let batch = particles();
    let geom = geometry();

    // Elastic path: the trial deformation is kept as-is, Jp unchanged.
    let vel_elastic = finalised_grid_velocity(&batch, &geom, false);
    let gpu_elastic = device(&gather, &ctx, &batch, &geom, &vel_elastic, false);
    let cpu_elastic = golden(&batch, &geom, &vel_elastic, false);
    assert_particles_match(&gpu_elastic, &cpu_elastic, 2.0e-3, "elastic");

    // Plastic path: the trial deformation is split by the snow return mapping,
    // exercising the sqrt-based signed SVD on device.
    let vel_plastic = finalised_grid_velocity(&batch, &geom, true);
    let gpu_plastic = device(&gather, &ctx, &batch, &geom, &vel_plastic, true);
    let cpu_plastic = golden(&batch, &geom, &vel_plastic, true);
    assert_particles_match(&gpu_plastic, &cpu_plastic, 2.0e-3, "plastic");
}
