//! Real-device parity for the `GPU` per-fragment bounds builder against its
//! `CPU` golden twin.
//!
//! Every test acquires a headless device with `GpuContext::try_headless()` and
//! skips cleanly when no adapter is available (for example inside a sandbox),
//! so the suite is a no-op rather than a failure off a real `GPU`.
//!
//! Provenance: axis-aligned extrema and a box-centred bounding sphere are
//! elementary geometry and fixed-point atomic reduction is a standard `GPU`
//! technique. No Unreal Engine source or derived code.

use glam::Vec3;
use prism_physics_gpu::context::GpuContext;
use prism_physics_gpu::{
    cpu_assign_cells, cpu_bounds_fragments, BoundsConfig, GpuFragmentBounds, GpuVoronoiAssign,
    VoronoiAssignConfig,
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

    /// A float in `[lo, hi)`.
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        let unit = (self.next_u64() >> 40) as f32 / (1_u64 << 24) as f32;
        lo + unit * (hi - lo)
    }

    /// A `u32` in `[0, n)`.
    fn below(&mut self, n: u32) -> u32 {
        (self.next_u64() % u64::from(n)) as u32
    }
}

/// Asserts the two boxes are bit-identical component by component; the extrema
/// are de-quantised integers, so `GPU` and `CPU` must agree exactly.
fn assert_box_bit_exact(a_min: Vec3, a_max: Vec3, b_min: Vec3, b_max: Vec3) {
    for axis in 0..3 {
        assert_eq!(
            a_min[axis].to_bits(),
            b_min[axis].to_bits(),
            "box minimum must be bit-exact on axis {axis}"
        );
        assert_eq!(
            a_max[axis].to_bits(),
            b_max[axis].to_bits(),
            "box maximum must be bit-exact on axis {axis}"
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice and measured error must reach the test log"
)]
fn bounds_match_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("no GPU adapter; skipping bounds_match_cpu_golden");
        return;
    };

    let n_cells = 24_usize;
    let n_points = 4_000_usize;
    let mut rng = Rng::new(0x0005_1ce6);
    let mut points = Vec::with_capacity(n_points);
    let mut cells = Vec::with_capacity(n_points);
    for _ in 0..n_points {
        points.push(Vec3::new(
            rng.range(-4.0, 4.0),
            rng.range(-4.0, 4.0),
            rng.range(-4.0, 4.0),
        ));
        cells.push(rng.below(n_cells as u32));
    }

    let cfg = BoundsConfig::default();
    let cpu = cpu_bounds_fragments(n_cells, &points, &cells, &cfg);
    let gpu = GpuFragmentBounds::new(&ctx).compute(&ctx, n_cells, &points, &cells, &cfg);

    assert_eq!(cpu.len(), gpu.len());
    let mut worst_center = 0.0_f32;
    let mut worst_radius = 0.0_f32;
    for (c, g) in cpu.iter().zip(&gpu) {
        assert_box_bit_exact(c.aabb_min, c.aabb_max, g.aabb_min, g.aabb_max);
        worst_center = worst_center.max((c.sphere_center - g.sphere_center).length());
        worst_radius = worst_radius.max((c.sphere_radius - g.sphere_radius).abs());
    }
    assert!(
        worst_center <= 1.0e-6,
        "sphere centre diff {worst_center:e}"
    );
    assert!(
        worst_radius <= 1.0e-3,
        "sphere radius diff {worst_radius:e}"
    );
    eprintln!(
        "fracture bounds: worst centre diff = {worst_center:e}, radius diff = {worst_radius:e}"
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
    let cfg = BoundsConfig::default();
    let builder = GpuFragmentBounds::new(&ctx);

    // No cells: an empty result with no dispatch.
    let empty = builder.compute(&ctx, 0, &[], &[], &cfg);
    assert!(empty.is_empty());

    // No points: every cell reports a zeroed proxy.
    let zeroed = builder.compute(&ctx, 3, &[], &[], &cfg);
    assert_eq!(zeroed.len(), 3);
    for b in zeroed {
        assert_eq!(b.aabb_min, Vec3::ZERO);
        assert_eq!(b.aabb_max, Vec3::ZERO);
        assert_eq!(b.sphere_center, Vec3::ZERO);
        assert_eq!(b.sphere_radius, 0.0);
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice and measured error must reach the test log"
)]
fn full_pipeline_classify_then_bounds_matches_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("no GPU adapter; skipping full_pipeline_classify_then_bounds_matches_cpu");
        return;
    };

    // Well-separated seed sites so classification is integer-exact, then bin a
    // point cloud into their Voronoi cells and build the fragment bounds — the
    // broad-phase seeding pipeline end to end.
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
    for _ in 0..n_points {
        points.push(Vec3::new(
            rng.range(-4.0, 4.0),
            rng.range(-4.0, 4.0),
            rng.range(-4.0, 4.0),
        ));
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

    let cfg = BoundsConfig::default();
    let cpu = cpu_bounds_fragments(n_cells, &points, &gpu_cells, &cfg);
    let gpu = GpuFragmentBounds::new(&ctx).compute(&ctx, n_cells, &points, &gpu_cells, &cfg);

    let mut worst_radius = 0.0_f32;
    for (c, g) in cpu.iter().zip(&gpu) {
        assert_box_bit_exact(c.aabb_min, c.aabb_max, g.aabb_min, g.aabb_max);
        worst_radius = worst_radius.max((c.sphere_radius - g.sphere_radius).abs());
    }
    assert!(
        worst_radius <= 1.0e-3,
        "sphere radius diff {worst_radius:e}"
    );

    // Every point must fall inside both its fragment's box and sphere.
    for (p, &cell) in points.iter().zip(&gpu_cells) {
        let b = &gpu[cell as usize];
        let inside_box = p.x >= b.aabb_min.x - 1.0e-3
            && p.x <= b.aabb_max.x + 1.0e-3
            && p.y >= b.aabb_min.y - 1.0e-3
            && p.y <= b.aabb_max.y + 1.0e-3
            && p.z >= b.aabb_min.z - 1.0e-3
            && p.z <= b.aabb_max.z + 1.0e-3;
        assert!(inside_box, "point escapes its fragment box");
        let d = (*p - b.sphere_center).length();
        assert!(d <= b.sphere_radius + 1.0e-3, "point escapes its sphere");
    }
    eprintln!("pipeline bounds: {n_cells} fragments, worst radius diff = {worst_radius:e}");
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
    let mut cells = Vec::with_capacity(n_points);
    for _ in 0..n_points {
        points.push(Vec3::new(
            rng.range(-6.0, 6.0),
            rng.range(-6.0, 6.0),
            rng.range(-6.0, 6.0),
        ));
        cells.push(rng.below(n_cells as u32));
    }

    let cfg = BoundsConfig::default();
    let cpu = cpu_bounds_fragments(n_cells, &points, &cells, &cfg);
    let builder = GpuFragmentBounds::new(&ctx);

    let start = std::time::Instant::now();
    let gpu = builder.compute(&ctx, n_cells, &points, &cells, &cfg);
    let elapsed = start.elapsed();

    let mut worst_radius = 0.0_f32;
    for (c, g) in cpu.iter().zip(&gpu) {
        assert_box_bit_exact(c.aabb_min, c.aabb_max, g.aabb_min, g.aabb_max);
        worst_radius = worst_radius.max((c.sphere_radius - g.sphere_radius).abs());
    }
    assert!(
        worst_radius <= 1.0e-2,
        "sphere radius diff {worst_radius:e}"
    );
    eprintln!(
        "fracture bounds: {n_points} points x {n_cells} cells in {:.1} ms, worst radius diff = {worst_radius:e}",
        elapsed.as_secs_f64() * 1.0e3
    );
}
