//! Real-device parity for the `GPU` per-fragment aggregator against its `CPU`
//! golden twin.
//!
//! Every test acquires a headless device with `GpuContext::try_headless()` and
//! skips cleanly when no adapter is available (for example inside a sandbox),
//! so the suite is a no-op rather than a failure off a real `GPU`.
//!
//! Provenance: rigid-body mass/centroid/inertia formulas are textbook mechanics
//! and fixed-point atomic accumulation is a standard `GPU` reduction. No Unreal
//! Engine source or derived code.

use glam::{Mat3, Vec3};
use prism_physics_gpu::context::GpuContext;
use prism_physics_gpu::{
    cpu_aggregate_fragments, cpu_assign_cells, AggregateConfig, GpuFragmentAggregate,
    GpuVoronoiAssign, VoronoiAssignConfig,
};

/// A tiny deterministic `xorshift64*` generator so the tests need no external
/// crate for reproducible point clouds.
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
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// A float in `[lo, hi)`, bounded so the fixed-point accumulators stay well
    /// inside the documented `i32` overflow budget.
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        let unit = (self.next_u64() >> 40) as f32 / (1_u64 << 24) as f32;
        lo + unit * (hi - lo)
    }
}

/// Worst absolute component difference between two inertia tensors.
fn inertia_abs_diff(a: Mat3, b: Mat3) -> f32 {
    let mut worst = 0.0_f32;
    for col in 0..3 {
        for row in 0..3 {
            worst = worst.max((a.col(col)[row] - b.col(col)[row]).abs());
        }
    }
    worst
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice and measured error must reach the test log"
)]
fn aggregate_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("no GPU adapter; skipping aggregate_matches_cpu_golden");
        return;
    };

    let n_cells = 24_usize;
    let n_points = 4_000_usize;
    let mut rng = Rng::new(0x0005_1ce6);

    let mut points = Vec::with_capacity(n_points);
    let mut masses = Vec::with_capacity(n_points);
    let mut cells = Vec::with_capacity(n_points);
    for _ in 0..n_points {
        points.push(Vec3::new(
            rng.range(-4.0, 4.0),
            rng.range(-4.0, 4.0),
            rng.range(-4.0, 4.0),
        ));
        masses.push(rng.range(0.5, 1.5));
        cells.push((rng.next_u64() % n_cells as u64) as u32);
    }

    let config = AggregateConfig::default();
    let cpu = cpu_aggregate_fragments(n_cells, &points, &masses, &cells, &config);
    let gpu =
        GpuFragmentAggregate::new(&ctx).aggregate(&ctx, n_cells, &points, &masses, &cells, &config);
    assert_eq!(cpu.len(), gpu.len());

    let mut worst_mass = 0.0_f32;
    let mut worst_centroid = 0.0_f32;
    let mut worst_inertia = 0.0_f32;
    for (c, g) in cpu.iter().zip(&gpu) {
        // Mass is a de-quantised integer sum, so it is bit-identical.
        assert_eq!(c.mass.to_bits(), g.mass.to_bits(), "mass must be bit-exact");
        worst_mass = worst_mass.max((c.mass - g.mass).abs());
        worst_centroid = worst_centroid.max((c.centroid - g.centroid).length());
        worst_inertia = worst_inertia.max(inertia_abs_diff(c.inertia, g.inertia));
    }
    eprintln!(
        "fracture aggregate: worst mass diff = {worst_mass:e}, centroid = {worst_centroid:e}, inertia = {worst_inertia:e}"
    );
    assert!(
        worst_centroid <= 1.0e-5,
        "centroid diff {worst_centroid} too large"
    );
    assert!(
        worst_inertia <= 1.0e-2,
        "inertia diff {worst_inertia} too large"
    );
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice and measured error must reach the test log"
)]
fn degenerate_batches_are_handled() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("no GPU adapter; skipping degenerate_batches_are_handled");
        return;
    };
    let agg = GpuFragmentAggregate::new(&ctx);
    let config = AggregateConfig::default();

    // Zero cells: empty result without dispatching.
    let none = agg.aggregate(&ctx, 0, &[Vec3::ZERO], &[1.0], &[0], &config);
    assert!(none.is_empty());

    // No points: every cell reports a zero seed.
    let empty_pts = agg.aggregate(&ctx, 3, &[], &[], &[], &config);
    assert_eq!(empty_pts.len(), 3);
    for frag in empty_pts {
        assert_eq!(frag.mass, 0.0);
        assert_eq!(frag.centroid, Vec3::ZERO);
        assert_eq!(frag.inertia, Mat3::ZERO);
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice and measured error must reach the test log"
)]
fn full_pipeline_classify_then_aggregate_matches_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("no GPU adapter; skipping full_pipeline_classify_then_aggregate_matches_cpu");
        return;
    };

    // Well-separated seed sites so classification is integer-exact, then bin a
    // point cloud into their Voronoi cells and aggregate the fragments — the
    // real destruction pipeline end to end.
    let sites = vec![
        Vec3::new(-3.0, -3.0, -3.0),
        Vec3::new(3.0, -3.0, -2.0),
        Vec3::new(-2.5, 3.0, 2.5),
        Vec3::new(3.0, 2.5, -3.0),
        Vec3::new(0.0, 0.0, 3.0),
    ];
    let n_cells = sites.len();

    let n_points = 6_000_usize;
    let mut rng = Rng::new(0xf00d_1234);
    let mut points = Vec::with_capacity(n_points);
    let mut masses = Vec::with_capacity(n_points);
    for _ in 0..n_points {
        points.push(Vec3::new(
            rng.range(-4.0, 4.0),
            rng.range(-4.0, 4.0),
            rng.range(-4.0, 4.0),
        ));
        masses.push(rng.range(0.5, 1.5));
    }

    let assign_cfg = VoronoiAssignConfig::default();
    let classifier = GpuVoronoiAssign::new(&ctx);
    let gpu_cells: Vec<u32> = classifier
        .assign(&ctx, &sites, &points, &assign_cfg)
        .into_iter()
        .map(|a| a.cell)
        .collect();
    let cpu_cells: Vec<u32> = cpu_assign_cells(&sites, &points, &assign_cfg)
        .into_iter()
        .map(|a| a.cell)
        .collect();
    assert_eq!(gpu_cells, cpu_cells, "classification must be integer-exact");

    let agg_cfg = AggregateConfig::default();
    let cpu = cpu_aggregate_fragments(n_cells, &points, &masses, &gpu_cells, &agg_cfg);
    let gpu = GpuFragmentAggregate::new(&ctx)
        .aggregate(&ctx, n_cells, &points, &masses, &gpu_cells, &agg_cfg);

    let mut worst_inertia = 0.0_f32;
    let mut total_mass = 0.0_f32;
    for (c, g) in cpu.iter().zip(&gpu) {
        assert_eq!(c.mass.to_bits(), g.mass.to_bits(), "mass must be bit-exact");
        assert!((c.centroid - g.centroid).length() <= 1.0e-5);
        worst_inertia = worst_inertia.max(inertia_abs_diff(c.inertia, g.inertia));
        total_mass += g.mass;
    }
    eprintln!(
        "pipeline: {} fragments, total mass {total_mass:.2}, worst inertia diff = {worst_inertia:e}",
        gpu.len()
    );
    assert!(
        worst_inertia <= 1.0e-2,
        "inertia diff {worst_inertia} too large"
    );
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice and measured error must reach the test log"
)]
fn scales_to_many_points() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("no GPU adapter; skipping scales_to_many_points");
        return;
    };

    let n_cells = 40_usize;
    let n_points = 100_000_usize;
    let mut rng = Rng::new(0xabcd_ef01);
    let mut points = Vec::with_capacity(n_points);
    let mut masses = Vec::with_capacity(n_points);
    let mut cells = Vec::with_capacity(n_points);
    for _ in 0..n_points {
        points.push(Vec3::new(
            rng.range(-4.0, 4.0),
            rng.range(-4.0, 4.0),
            rng.range(-4.0, 4.0),
        ));
        masses.push(rng.range(0.5, 1.5));
        cells.push((rng.next_u64() % n_cells as u64) as u32);
    }

    let config = AggregateConfig::default();
    let cpu = cpu_aggregate_fragments(n_cells, &points, &masses, &cells, &config);
    let agg = GpuFragmentAggregate::new(&ctx);
    let start = std::time::Instant::now();
    let gpu = agg.aggregate(&ctx, n_cells, &points, &masses, &cells, &config);
    let elapsed = start.elapsed();
    eprintln!(
        "fracture aggregate: {n_points} points x {n_cells} cells in {:.1} ms",
        elapsed.as_secs_f64() * 1_000.0
    );

    let mut worst_inertia = 0.0_f32;
    for (c, g) in cpu.iter().zip(&gpu) {
        assert_eq!(c.mass.to_bits(), g.mass.to_bits(), "mass must be bit-exact");
        assert!((c.centroid - g.centroid).length() <= 1.0e-5);
        worst_inertia = worst_inertia.max(inertia_abs_diff(c.inertia, g.inertia));
    }
    eprintln!("fracture aggregate: worst inertia diff at scale = {worst_inertia:e}");
    assert!(
        worst_inertia <= 1.0e-1,
        "inertia diff {worst_inertia} too large"
    );
}
