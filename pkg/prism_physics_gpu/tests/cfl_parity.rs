//! Real-device parity for the `GPU` `CFL` velocity reduction against its `CPU`
//! golden twin.
//!
//! Every test acquires a headless device with `GpuContext::try_headless()` and
//! skips cleanly when no adapter is available (for example inside a sandbox),
//! so the suite is a no-op rather than a failure off a real `GPU`.
//!
//! Provenance: the `CFL` condition is a classical, openly published stability
//! criterion and shared-memory tree reduction is a standard `GPU` technique. No
//! Unreal Engine source or derived code.

use glam::Vec3;
use prism_physics_gpu::context::GpuContext;
use prism_physics_gpu::{cpu_cfl_dt, cpu_max_speed, CflConfig, GpuCflReduce};

/// A tiny deterministic `xorshift64*` generator so the tests need no external
/// crate for reproducible velocity fields.
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
        lo + (hi - lo) * unit
    }
}

/// Relative tolerance for the reduced speed: the fold is exact on the bit
/// patterns and only the closing square root (with `GPU` fused multiply-add in
/// the dot product) can perturb the low bits.
const TOL: f32 = 1.0e-4;

fn close(a: f32, b: f32) -> bool {
    (a - b).abs() <= TOL * a.abs().max(1.0)
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice and measured error must reach the test log"
)]
fn max_speed_matches_cpu_on_a_random_field() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("no GPU adapter; skipping max_speed_matches_cpu_on_a_random_field");
        return;
    };

    let mut rng = Rng::new(0x0c51_f10a);
    let n = 12_000_usize;
    let mut vel = Vec::with_capacity(n);
    for _ in 0..n {
        vel.push(Vec3::new(
            rng.range(-50.0, 50.0),
            rng.range(-50.0, 50.0),
            rng.range(-50.0, 50.0),
        ));
    }

    let cpu = cpu_max_speed(&vel);
    let gpu = GpuCflReduce::new(&ctx).max_speed(&ctx, &vel);
    assert!(close(gpu, cpu), "max speed cpu {cpu} vs gpu {gpu}");
    eprintln!("cfl reduce: {n} velocities, cpu {cpu} vs gpu {gpu}");
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log"
)]
fn degenerate_fields_reduce_cleanly() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("no GPU adapter; skipping degenerate_fields_reduce_cleanly");
        return;
    };
    let reducer = GpuCflReduce::new(&ctx);

    // Empty field: no dispatch, zero speed.
    assert_eq!(reducer.max_speed(&ctx, &[]), 0.0);

    // Single element: its own speed (3-4-0 is a 3-4-5 triangle).
    let one = reducer.max_speed(&ctx, &[Vec3::new(3.0, 4.0, 0.0)]);
    assert!(close(one, 5.0), "single speed {one}");

    // A field at rest reduces to zero.
    let still = reducer.max_speed(&ctx, &[Vec3::ZERO, Vec3::ZERO, Vec3::ZERO]);
    assert_eq!(still, 0.0);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice and measured error must reach the test log"
)]
fn suggest_dt_clamps_across_the_three_regimes() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("no GPU adapter; skipping suggest_dt_clamps_across_the_three_regimes");
        return;
    };
    let reducer = GpuCflReduce::new(&ctx);
    let cfg = CflConfig {
        cfl_number: 0.5,
        cell_size: 1.0,
        dt_min: 1.0e-4,
        dt_max: 1.0e-2,
    };

    // A still scene takes the largest step.
    let rest = reducer.suggest_dt(&ctx, &[Vec3::ZERO], &cfg);
    assert_eq!(rest, cfg.dt_max);

    // A very fast particle is clamped to the floor.
    let fast = reducer.suggest_dt(&ctx, &[Vec3::new(1.0e5, 0.0, 0.0)], &cfg);
    assert_eq!(fast, cfg.dt_min);

    // A moderate speed follows the formula: 0.5 * 1 / 100 = 5e-3, inside range.
    let mid = reducer.suggest_dt(&ctx, &[Vec3::new(60.0, 80.0, 0.0)], &cfg);
    let expected = cpu_cfl_dt(&[Vec3::new(60.0, 80.0, 0.0)], &cfg);
    assert!(close(mid, expected), "mid dt {mid} vs {expected}");
    assert!(
        mid > cfg.dt_min && mid < cfg.dt_max,
        "mid dt {mid} not interior"
    );
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice and measured timing must reach the test log"
)]
fn scales_to_many_velocities() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("no GPU adapter; skipping scales_to_many_velocities");
        return;
    };

    let n = 1_000_000_usize;
    let mut rng = Rng::new(0xbeef_cafe);
    let mut vel = Vec::with_capacity(n);
    for _ in 0..n {
        vel.push(Vec3::new(
            rng.range(-100.0, 100.0),
            rng.range(-100.0, 100.0),
            rng.range(-100.0, 100.0),
        ));
    }

    let cpu = cpu_max_speed(&vel);
    let reducer = GpuCflReduce::new(&ctx);
    let start = std::time::Instant::now();
    let gpu = reducer.max_speed(&ctx, &vel);
    let elapsed = start.elapsed();

    assert!(close(gpu, cpu), "max speed cpu {cpu} vs gpu {gpu}");
    eprintln!(
        "cfl reduce: {n} velocities in {:.1} ms, cpu {cpu} vs gpu {gpu}",
        elapsed.as_secs_f64() * 1.0e3
    );
}
