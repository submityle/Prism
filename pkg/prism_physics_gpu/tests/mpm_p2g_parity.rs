//! Real-device parity: the `GPU` MLS-MPM particle-to-grid (`P2G`) affine scatter
//! must reproduce the `CPU` golden [`prism_physics_core::mpm`] `particle_to_grid`
//! transfer, node for node, within a tight tolerance.
//!
//! The scatter is the first fused transfer kernel: it evaluates the
//! fixed-corotated stress, folds it with the `APIC` affine momentum, and
//! distributes mass and momentum to each particle's 27 surrounding grid nodes
//! via fixed-point atomic adds. The parity target is the grid state *before*
//! velocity finalisation, so [`prism_physics_core::mpm::Grid::velocity_at`]
//! returns the raw accumulated momentum, which is what the device kernel
//! produces.
//!
//! The batch deliberately drives every arithmetic path: particles carry
//! non-identity deformation gradients (so the corotated stress is nonzero and
//! the sqrt-based signed `SVD` runs), nonzero affine matrices `C`, and nonzero
//! velocities, and they are placed so their `3x3` stencils overlap (exercising
//! the atomic accumulation) and so some stencils spill out of bounds (exercising
//! the node skip). Both the elastic and the plastic+hardening Lamé-scaling paths
//! are covered.
//!
//! The device reassociates its sums differently from the host, runs the Jacobi
//! `SVD` in `f32`, and accumulates through a fixed-point quantisation whose
//! resolution is `1/65536 ≈ 1.5e-5` per add, so parity is checked within a
//! small absolute-plus-relative tolerance rather than bit-for-bit.
//!
//! On a headless host with no `wgpu` adapter the test skips (with a printed
//! notice) instead of failing.
//!
//! Provenance: the affine MLS-MPM scatter (Hu et al. 2018; Jiang et al. 2015)
//! and the fixed-corotated model (Stomakhin et al. 2013) are standard, publicly
//! documented techniques. No Unreal Engine source or derived code.

use glam::{Mat3, Vec3};
use prism_physics_core::mpm::{
    particle_to_grid, Grid, MaterialPoints, MpmConfig, MpmMaterial, SnowPlasticity,
};
use prism_physics_gpu::{GpuContext, GpuMpmP2g, P2gGrid};

/// Grid geometry shared by the `CPU` golden and the `GPU` scatter.
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

/// One particle's full state for the scatter.
struct Particle {
    position: Vec3,
    velocity: Vec3,
    affine: Mat3,
    deformation: Mat3,
    mass: f32,
    volume: f32,
    plastic_det: f32,
}

/// Builds the test grid: a `6x6x6` lattice with unit-ish cells placed so most
/// particles sit interior but a couple of stencils reach the boundary.
fn geometry() -> GridGeometry {
    GridGeometry {
        nx: 6,
        ny: 6,
        nz: 6,
        dx: 0.5,
        origin: Vec3::new(-0.25, -0.25, -0.25),
    }
}

/// Builds the test particle batch. Positions are chosen so stencils overlap on
/// shared interior nodes and so the last particle's stencil partly leaves the
/// grid, exercising the out-of-bounds node skip.
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
        // Placed near the far corner so its 3x3 stencil spills out of bounds.
        Particle {
            position: Vec3::new(2.35, 2.35, 2.35),
            velocity: Vec3::new(0.20, 0.10, -0.30),
            affine: Mat3::from_cols(
                Vec3::new(0.05, 0.02, -0.01),
                Vec3::new(-0.03, 0.07, 0.04),
                Vec3::new(0.02, -0.05, 0.06),
            ),
            deformation: Mat3::from_cols(
                Vec3::new(1.08, 0.03, -0.02),
                Vec3::new(0.02, 0.96, 0.03),
                Vec3::new(-0.04, 0.01, 1.05),
            ),
            mass: 0.70,
            volume: 0.03,
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

/// Runs the `CPU` golden `P2G` scatter and reads back the pre-finalisation grid
/// (mass and accumulated momentum) as a [`P2gGrid`].
fn golden(
    batch: &[Particle],
    geom: &GridGeometry,
    material: &MpmMaterial,
    plastic: bool,
) -> P2gGrid {
    let mp = material_points(batch);
    let mut cfg = MpmConfig::new(1.0e-3, Vec3::new(0.0, -9.81, 0.0));
    cfg.plastic = plastic;
    cfg.plasticity = SnowPlasticity::default();
    let mut grid = Grid::new(geom.nx, geom.ny, geom.nz, geom.dx, geom.origin);
    particle_to_grid(&mp, &mut grid, &cfg, material);

    let mut mass = vec![0.0_f32; geom.node_count()];
    let mut momentum = vec![Vec3::ZERO; geom.node_count()];
    for k in 0..geom.nz {
        for j in 0..geom.ny {
            for i in 0..geom.nx {
                let flat = grid.flat(i, j, k);
                mass[flat] = grid.mass_at(i, j, k);
                // Before finalize_velocity, velocity_at returns accumulated momentum.
                momentum[flat] = grid.velocity_at(i, j, k);
            }
        }
    }
    P2gGrid { mass, momentum }
}

/// Runs the `GPU` scatter for one configuration.
fn device(
    scatter: &GpuMpmP2g,
    ctx: &GpuContext,
    batch: &[Particle],
    geom: &GridGeometry,
    material: &MpmMaterial,
    plasticity: &SnowPlasticity,
    plastic: bool,
) -> P2gGrid {
    let positions: Vec<Vec3> = batch.iter().map(|p| p.position).collect();
    let velocities: Vec<Vec3> = batch.iter().map(|p| p.velocity).collect();
    let affine: Vec<Mat3> = batch.iter().map(|p| p.affine).collect();
    let deformation: Vec<Mat3> = batch.iter().map(|p| p.deformation).collect();
    let masses: Vec<f32> = batch.iter().map(|p| p.mass).collect();
    let volumes: Vec<f32> = batch.iter().map(|p| p.volume).collect();
    let plastic_dets: Vec<f32> = batch.iter().map(|p| p.plastic_det).collect();
    let (lambda0, mu0) = material.lame();
    scatter.scatter(
        ctx,
        &positions,
        &velocities,
        &affine,
        &deformation,
        &masses,
        &volumes,
        &plastic_dets,
        geom.nx,
        geom.ny,
        geom.nz,
        geom.origin,
        geom.dx,
        1.0e-3,
        mu0,
        lambda0,
        plasticity.hardening,
        plastic,
    )
}

/// Asserts that the `GPU` grid tracks the `CPU` golden node for node within
/// `tol` (absolute-plus-relative).
#[expect(
    clippy::print_stderr,
    reason = "the measured worst-case error must reach the test log"
)]
fn assert_grid_matches(gpu: &P2gGrid, cpu: &P2gGrid, tol: f32, label: &str) {
    assert_eq!(
        gpu.mass.len(),
        cpu.mass.len(),
        "{label}: mass length mismatch"
    );
    assert_eq!(
        gpu.momentum.len(),
        cpu.momentum.len(),
        "{label}: momentum length mismatch"
    );

    // Absolute floor for the relative metric, scaled to the peak field
    // magnitude. Fringe nodes carry physically negligible mass/momentum near
    // the fixed-point resolution floor; anchoring the denominator to a small
    // fraction of the peak keeps those nodes from dominating the metric while
    // still holding significant nodes to a tight relative error.
    let peak_mass = cpu.mass.iter().fold(0.0_f32, |m, &v| m.max(v.abs()));
    let peak_mom = cpu
        .momentum
        .iter()
        .flat_map(|m| [m.x.abs(), m.y.abs(), m.z.abs()])
        .fold(0.0_f32, f32::max);
    let mass_floor = 1.0e-3 * peak_mass.max(1.0e-6);
    let mom_floor = 1.0e-3 * peak_mom.max(1.0e-6);

    let mut worst = 0.0_f32;
    for n in 0..cpu.mass.len() {
        let mdiff = (gpu.mass[n] - cpu.mass[n]).abs();
        let mscale = mass_floor + gpu.mass[n].abs().max(cpu.mass[n].abs());
        worst = worst.max(mdiff / mscale);
        for c in 0..3 {
            let gc = gpu.momentum[n][c];
            let cc = cpu.momentum[n][c];
            let diff = (gc - cc).abs();
            let scale = mom_floor + gc.abs().max(cc.abs());
            worst = worst.max(diff / scale);
        }
    }
    eprintln!("{label}: worst P2G node error = {worst:e}");
    assert!(
        worst < tol,
        "{label}: GPU P2G scatter diverged from CPU golden (worst = {worst:e}, tol = {tol:e})",
    );
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on adapter-less hosts"
)]
fn gpu_mpm_p2g_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU MPM P2G parity: no wgpu adapter on this host");
        return;
    };
    let scatter = GpuMpmP2g::new(&ctx);

    let batch = particles();
    let geom = geometry();
    let material = MpmMaterial::default();
    let plasticity = SnowPlasticity::default();

    // Elastic-only: Lamé parameters unscaled.
    let gpu_elastic = device(&scatter, &ctx, &batch, &geom, &material, &plasticity, false);
    let cpu_elastic = golden(&batch, &geom, &material, false);
    assert_grid_matches(&gpu_elastic, &cpu_elastic, 2.0e-3, "elastic");

    // Plastic + hardening: Lamé parameters scaled per particle by
    // hardening_factor(ξ, Jp), exercising the range-reduced exponential.
    let gpu_plastic = device(&scatter, &ctx, &batch, &geom, &material, &plasticity, true);
    let cpu_plastic = golden(&batch, &geom, &material, true);
    assert_grid_matches(&gpu_plastic, &cpu_plastic, 2.0e-3, "plastic");
}
