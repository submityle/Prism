//! Real-device parity: the `GPU` MLS-MPM grid velocity update must reproduce
//! the `CPU` golden node-wise finalise / gravity / boundary sequence the solver
//! runs between `P2G` and `G2P`, node for node, within a tight tolerance.
//!
//! The kernel under test consumes the accumulated grid state (node mass and
//! momentum) from the `P2G` scatter and, per node, finalises `v = momentum /
//! mass` on nodes with positive mass, adds the gravity increment `gravity · dt`,
//! and enforces the selected wall boundary condition within `thickness` nodes of
//! each domain face. It is the device twin of
//! [`prism_physics_core::mpm::Grid::finalize_velocity`],
//! `Grid::add_velocity_to_active(gravity · dt)`, and
//! [`prism_physics_core::mpm::apply_grid_boundary`], run in that order.
//!
//! To isolate the grid update from the `P2G` fixed-point quantisation, the
//! `CPU` golden and the `GPU` kernel are fed the *same* `f32` mass and momentum
//! arrays: a single `CPU` `P2G` scatter produces a realistic accumulated grid,
//! its mass and pre-finalisation momentum are read out once, and both paths map
//! that identical input. The kernel is a pure per-node `f32` map with no
//! fixed-point atomics, so the only divergence is `f32` reassociation of
//! `momentum / mass + gravity · dt`, and parity holds to a very tight
//! tolerance.
//!
//! The particle batch is placed so that grid nodes fall inside the boundary
//! layer of every domain face (low and high on all three axes), and the node
//! velocities take both signs, exercising every branch of all three boundary
//! modes. All three of [`prism_physics_core::mpm::BoundaryCondition`]'s variants
//! are checked under nonzero gravity.
//!
//! On a headless host with no `wgpu` adapter the test skips (with a printed
//! notice) instead of failing.
//!
//! Provenance: the affine PIC grid update and the standard MPM wall boundary
//! conditions (Stomakhin et al. 2013; Jiang et al. 2015) are standard, publicly
//! documented techniques. No Unreal Engine source or derived code.

use glam::{Mat3, Vec3};
use prism_physics_core::mpm::{
    apply_grid_boundary, particle_to_grid, BoundaryCondition, Grid, MaterialPoints, MpmConfig,
    MpmMaterial,
};
use prism_physics_gpu::{BoundaryMode, GpuContext, GpuMpmGridUpdate};

/// Time step shared by the `CPU` golden and the `GPU` kernel.
const DT: f32 = 1.0e-3;

/// Gravity acceleration shared by both paths (nonzero on all axes so the
/// boundary logic sees varied velocity signs).
fn gravity() -> Vec3 {
    Vec3::new(1.5, -9.81, -0.75)
}

/// Grid geometry shared by the `CPU` golden and the `GPU` kernel.
struct GridGeometry {
    nx: usize,
    ny: usize,
    nz: usize,
    dx: f32,
    origin: Vec3,
    thickness: usize,
}

impl GridGeometry {
    /// The number of grid nodes.
    fn node_count(&self) -> usize {
        self.nx * self.ny * self.nz
    }
}

/// A `6x6x6` lattice with `dx = 0.5` and origin at the world origin. With a
/// boundary thickness of two, the low layer is nodes `{0, 1}` and the high layer
/// is nodes `{4, 5}` on each axis.
fn geometry() -> GridGeometry {
    GridGeometry {
        nx: 6,
        ny: 6,
        nz: 6,
        dx: 0.5,
        origin: Vec3::ZERO,
        thickness: 2,
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

/// Builds the particle batch. Positions are chosen so grid nodes fall inside the
/// boundary layer of every face (a particle near the low corner lights up nodes
/// `{0, 1, 2}`; one near the high corner lights up `{3, 4, 5}`) while a couple of
/// interior particles keep untouched nodes in play. Velocities span both signs
/// so the `Separate` mode exercises both its into-wall and out-of-wall branches.
fn particles() -> Vec<Particle> {
    let base_affine = Mat3::from_cols(
        Vec3::new(0.12, 0.04, -0.02),
        Vec3::new(-0.03, 0.10, 0.05),
        Vec3::new(0.01, -0.04, 0.14),
    );
    let base_deformation = Mat3::from_cols(
        Vec3::new(1.08, 0.05, -0.02),
        Vec3::new(0.04, 0.95, 0.03),
        Vec3::new(-0.03, 0.02, 1.06),
    );
    let make = |position: Vec3, velocity: Vec3, mass: f32, plastic_det: f32| Particle {
        position,
        velocity,
        affine: base_affine,
        deformation: base_deformation,
        mass,
        volume: 0.05,
        plastic_det,
    };
    vec![
        // Low corner: lights up the low boundary layer on all three axes.
        make(
            Vec3::new(0.30, 0.30, 0.30),
            Vec3::new(-0.40, -0.30, -0.20),
            1.20,
            0.96,
        ),
        // High corner: lights up the high boundary layer on all three axes.
        make(
            Vec3::new(2.20, 2.20, 2.20),
            Vec3::new(0.35, 0.45, 0.25),
            1.10,
            1.04,
        ),
        // Mixed faces: low x, high y.
        make(
            Vec3::new(0.30, 2.20, 1.25),
            Vec3::new(0.25, -0.35, 0.15),
            0.90,
            0.92,
        ),
        // Mixed faces: high x, low y, high z.
        make(
            Vec3::new(2.20, 0.30, 2.20),
            Vec3::new(-0.30, 0.40, -0.20),
            0.95,
            1.05,
        ),
        // Interior: keeps some non-boundary nodes populated.
        make(
            Vec3::new(1.25, 1.25, 1.25),
            Vec3::new(0.10, 0.20, -0.15),
            1.05,
            0.98,
        ),
        // Low z / high (x-ish) edge for extra coverage.
        make(
            Vec3::new(1.25, 0.30, 0.30),
            Vec3::new(0.20, -0.25, 0.30),
            0.80,
            1.02,
        ),
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

/// Runs one `CPU` `P2G` scatter and returns the accumulated node mass and
/// pre-finalisation momentum (before `finalize_velocity`, `velocity_at` returns
/// the raw accumulated momentum). This identical `f32` state feeds both paths.
fn scatter_grid_state(batch: &[Particle], geom: &GridGeometry) -> (Vec<f32>, Vec<Vec3>) {
    let mp = material_points(batch);
    let cfg = MpmConfig::new(DT, gravity());
    let material = MpmMaterial::default();
    let mut grid = Grid::new(geom.nx, geom.ny, geom.nz, geom.dx, geom.origin);
    particle_to_grid(&mp, &mut grid, &cfg, &material);

    let mut mass = vec![0.0_f32; geom.node_count()];
    let mut momentum = vec![Vec3::ZERO; geom.node_count()];
    for k in 0..geom.nz {
        for j in 0..geom.ny {
            for i in 0..geom.nx {
                let flat = grid.flat(i, j, k);
                mass[flat] = grid.mass_at(i, j, k);
                momentum[flat] = grid.velocity_at(i, j, k);
            }
        }
    }
    (mass, momentum)
}

/// Runs the `CPU` golden grid update for one boundary mode, seeding a fresh grid
/// with the shared `mass`/`momentum` state, finalising velocity, adding gravity,
/// and applying the wall boundary. Returns the finalised node velocity in flat
/// index order.
fn golden(
    mass: &[f32],
    momentum: &[Vec3],
    geom: &GridGeometry,
    boundary: BoundaryCondition,
) -> Vec<Vec3> {
    // Seed a fresh grid with the shared mass/momentum state. `accumulate` is the
    // core grid's only mutating entry point and adds into node state, so adding
    // each populated node's (mass, momentum) pair exactly once restores the
    // grid to the identical accumulated state the GPU kernel consumes.
    let mut cfg = MpmConfig::new(DT, gravity());
    cfg.boundary = boundary;
    cfg.boundary_thickness = geom.thickness;

    let mut grid = Grid::new(geom.nx, geom.ny, geom.nz, geom.dx, geom.origin);
    for k in 0..geom.nz {
        for j in 0..geom.ny {
            for i in 0..geom.nx {
                let flat = grid.flat(i, j, k);
                if mass[flat] > 0.0 {
                    grid.accumulate(i as i32, j as i32, k as i32, mass[flat], momentum[flat]);
                }
            }
        }
    }

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

/// Maps a core [`BoundaryCondition`] to the `GPU` harness [`BoundaryMode`].
fn to_mode(boundary: BoundaryCondition) -> BoundaryMode {
    match boundary {
        BoundaryCondition::Sticky => BoundaryMode::Sticky,
        BoundaryCondition::Slip => BoundaryMode::Slip,
        BoundaryCondition::Separate => BoundaryMode::Separate,
    }
}

/// Asserts the `GPU` node velocities track the `CPU` golden node for node within
/// `tol` (absolute-plus-relative, anchored to a small fraction of the peak).
#[expect(
    clippy::print_stderr,
    reason = "the measured worst-case error must reach the test log"
)]
fn assert_velocity_matches(gpu: &[Vec3], cpu: &[Vec3], tol: f32, label: &str) {
    assert_eq!(gpu.len(), cpu.len(), "{label}: node count mismatch");

    let peak = cpu
        .iter()
        .flat_map(|v| [v.x.abs(), v.y.abs(), v.z.abs()])
        .fold(0.0_f32, f32::max);
    let floor = 1.0e-4 * peak.max(1.0e-6);

    let mut worst = 0.0_f32;
    for n in 0..cpu.len() {
        for c in 0..3 {
            let gc = gpu[n][c];
            let cc = cpu[n][c];
            let diff = (gc - cc).abs();
            let scale = floor + gc.abs().max(cc.abs());
            worst = worst.max(diff / scale);
        }
    }
    eprintln!("{label}: worst grid-update node error = {worst:e}");
    assert!(
        worst < tol,
        "{label}: GPU grid update diverged from CPU golden (worst = {worst:e}, tol = {tol:e})",
    );
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on adapter-less hosts"
)]
fn gpu_mpm_grid_update_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU MPM grid-update parity: no wgpu adapter on this host");
        return;
    };
    let update = GpuMpmGridUpdate::new(&ctx);

    let geom = geometry();
    let batch = particles();
    let (mass, momentum) = scatter_grid_state(&batch, &geom);

    for boundary in [
        BoundaryCondition::Sticky,
        BoundaryCondition::Slip,
        BoundaryCondition::Separate,
    ] {
        let cpu = golden(&mass, &momentum, &geom, boundary);
        let gpu = update.update(
            &ctx,
            &mass,
            &momentum,
            geom.nx,
            geom.ny,
            geom.nz,
            gravity(),
            DT,
            geom.thickness,
            to_mode(boundary),
        );
        assert_velocity_matches(&gpu, &cpu, 1.0e-4, &format!("{boundary:?}"));
    }
}
