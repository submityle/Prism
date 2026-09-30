//! Real-device parity for the `GPU` Voronoi fragment-assignment classifier.
//!
//! [`GpuVoronoiAssign`] bins a point cloud into the Voronoi cells of a set of
//! fracture seed sites. This test pins its two guarantees against the
//! [`cpu_assign_cells`] golden twin:
//!
//! 1. **Cell index is exact.** With well-separated sites the nearest site is
//!    unambiguous, so the owning cell index the kernel emits must match the twin
//!    for every point, bit-for-bit.
//! 2. **Clearance matches tightly.** The distance to the nearest cell wall
//!    carries a normalising square root, so `GPU`/`CPU` reassociation perturbs
//!    only the low bits; the two must agree within a tight absolute tolerance.
//!
//! A larger case drives a hundred-thousand-point cloud to confirm the classifier
//! scales and reports the wall-clock time. On a headless host with no `wgpu`
//! adapter every case skips instead of failing.
//!
//! Provenance: nearest-site Voronoi membership and perpendicular-bisector cell
//! walls are standard, publicly documented computational-geometry results. No
//! Unreal Engine source or derived code.

use glam::Vec3;
use prism_physics_gpu::{
    cpu_assign_cells, CellAssignment, GpuContext, GpuVoronoiAssign, VoronoiAssignConfig, NO_CELL,
};

/// A deterministic `xorshift64*` stream, so the point clouds are reproducible
/// without pulling in an RNG dependency.
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Rng {
        Rng { state: seed | 1 }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// A float in `[0, 1)`.
    fn next_unit(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u32 << 24) as f32)
    }

    /// A point uniformly inside the cube `[-half, half]^3`.
    fn next_in_cube(&mut self, half: f32) -> Vec3 {
        Vec3::new(
            (self.next_unit() * 2.0 - 1.0) * half,
            (self.next_unit() * 2.0 - 1.0) * half,
            (self.next_unit() * 2.0 - 1.0) * half,
        )
    }
}

/// Asserts a `GPU` assignment batch matches the `CPU` golden: cell indices
/// exactly, and finite clearances within `tol` (saturated no-wall clearances are
/// treated as a shared "very large" sentinel).
fn assert_parity(gpu: &[CellAssignment], cpu: &[CellAssignment], tol: f32) {
    assert_eq!(gpu.len(), cpu.len(), "assignment count");
    for (i, (g, c)) in gpu.iter().zip(cpu).enumerate() {
        assert_eq!(g.cell, c.cell, "point {i}: cell index diverged");
        if c.clearance.is_infinite() {
            assert!(
                g.clearance >= 1.0e29,
                "point {i}: expected saturated clearance, got {}",
                g.clearance,
            );
        } else {
            let diff = (g.clearance - c.clearance).abs();
            assert!(
                diff <= tol,
                "point {i}: clearance diverged by {diff:e} (gpu {}, cpu {})",
                g.clearance,
                c.clearance,
            );
        }
    }
}

/// Cell indices must be exact and clearances tight for a well-separated site
/// cloud and a random query batch, exercising every arithmetic path.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice and measured error must reach the test log"
)]
fn assign_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping fracture Voronoi parity: no wgpu adapter on this host");
        return;
    };
    let mut rng = Rng::new(0xF00D_BEEF);
    // Twelve well-separated sites and a few hundred query points in the same box.
    let sites: Vec<Vec3> = (0..12).map(|_| rng.next_in_cube(1.0)).collect();
    let points: Vec<Vec3> = (0..500).map(|_| rng.next_in_cube(1.0)).collect();
    let cfg = VoronoiAssignConfig::default();

    let cpu = cpu_assign_cells(&sites, &points, &cfg);
    let solver = GpuVoronoiAssign::new(&ctx);
    let gpu = solver.assign(&ctx, &sites, &points, &cfg);

    let mut worst = 0.0_f32;
    for (g, c) in gpu.iter().zip(&cpu) {
        if c.clearance.is_finite() {
            worst = worst.max((g.clearance - c.clearance).abs());
        }
    }
    eprintln!("fracture Voronoi: worst clearance abs diff = {worst:e}");
    assert_parity(&gpu, &cpu, 1.0e-5);
}

/// An empty site cloud reports every point as [`NO_CELL`] with a saturated
/// clearance, and an empty point batch returns nothing without dispatching.
#[test]
fn degenerate_batches_are_handled() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let cfg = VoronoiAssignConfig::default();
    let solver = GpuVoronoiAssign::new(&ctx);

    let points = vec![Vec3::ZERO, Vec3::new(1.0, 2.0, 3.0)];
    let no_sites = solver.assign(&ctx, &[], &points, &cfg);
    assert_eq!(no_sites.len(), 2);
    for a in &no_sites {
        assert_eq!(a.cell, NO_CELL);
        assert!(a.clearance >= 1.0e29);
    }

    let empty = solver.assign(&ctx, &[Vec3::ZERO], &[], &cfg);
    assert!(empty.is_empty());
}

/// A single site owns every point with an infinite clearance (no walls); the
/// `CPU` and `GPU` sentinels must agree.
#[test]
fn single_site_owns_everything() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let cfg = VoronoiAssignConfig::default();
    let sites = vec![Vec3::new(0.25, -0.5, 0.75)];
    let mut rng = Rng::new(0x1234_5678);
    let points: Vec<Vec3> = (0..64).map(|_| rng.next_in_cube(2.0)).collect();

    let cpu = cpu_assign_cells(&sites, &points, &cfg);
    let solver = GpuVoronoiAssign::new(&ctx);
    let gpu = solver.assign(&ctx, &sites, &points, &cfg);

    for a in &cpu {
        assert_eq!(a.cell, 0);
        assert!(a.clearance.is_infinite());
    }
    assert_parity(&gpu, &cpu, 1.0e-5);
}

/// The classifier must scale to a hundred-thousand-point cloud and stay exact.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice and throughput report must reach the test log"
)]
fn scales_to_many_points() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping fracture Voronoi scale: no wgpu adapter on this host");
        return;
    };
    let mut rng = Rng::new(0xACE1_2025);
    let sites: Vec<Vec3> = (0..64).map(|_| rng.next_in_cube(1.0)).collect();
    let count = 200_000_usize;
    let points: Vec<Vec3> = (0..count).map(|_| rng.next_in_cube(1.0)).collect();
    let cfg = VoronoiAssignConfig::default();

    let solver = GpuVoronoiAssign::new(&ctx);
    let start = std::time::Instant::now();
    let gpu = solver.assign(&ctx, &sites, &points, &cfg);
    let elapsed = start.elapsed();
    eprintln!(
        "fracture Voronoi: {count} points x {} sites in {:.1} ms",
        sites.len(),
        elapsed.as_secs_f64() * 1.0e3,
    );

    let cpu = cpu_assign_cells(&sites, &points, &cfg);
    assert_parity(&gpu, &cpu, 1.0e-5);
    // Every point must land in a real cell.
    for a in &gpu {
        assert!(a.cell < sites.len() as u32, "point escaped the site set");
    }
}
