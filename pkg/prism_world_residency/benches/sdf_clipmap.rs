//! Reproducible CPU update benchmark for the sorted and toroidal SDF clipmaps.
//!
//! Run with `cargo bench --offline -p prism_world_residency --bench sdf_clipmap`.
//! The standard Cargo bench profile is optimized; this measures host CPU
//! planning only and does not include GPU work or represent product frame time.
#![expect(
    clippy::print_stdout,
    reason = "the standalone benchmark reports its measurements"
)]

use std::hint::black_box;
use std::time::Instant;

use prism_world_residency::{
    BrickCoord, CpuSdfClipmap, SdfClipmapConfig, SdfClipmapUpdate, SdfToroidalClipmap,
};

const WARMUP_UPDATES: usize = 128;
const MEASURED_UPDATES: usize = 1_024;
const MIP_LEVELS: u8 = 6;
const RING_RADIUS: u32 = 24;
const UPDATE_BUDGET_BYTES: usize = 8 * 1024 * 1024;

#[derive(Clone, Copy, Debug)]
enum Workload {
    SmallNegativeRoll,
    LargeJump,
}

impl Workload {
    const ALL: [Self; 2] = [Self::SmallNegativeRoll, Self::LargeJump];

    const fn name(self) -> &'static str {
        match self {
            Self::SmallNegativeRoll => "small_roll_negative_multimip",
            Self::LargeJump => "large_jump_disjoint_windows",
        }
    }

    fn center(self, step: usize) -> BrickCoord {
        match self {
            Self::SmallNegativeRoll => {
                // Keep coordinates negative while moving one brick in X per
                // update and one brick in Z every fourth update. Coarser mips
                // therefore change at deterministic, distinct intervals.
                let step = i32::try_from(step).expect("benchmark step fits i32");
                BrickCoord::new(-8_192 + step, -17, -4_096 + step / 4, 0)
            }
            Self::LargeJump => {
                let step = i32::try_from(step).expect("benchmark step fits i32");
                let side = if step % 2 == 0 { -1 } else { 1 };
                BrickCoord::new(side * 1_000_000, 113, -side * 750_000, 0)
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Sample {
    elapsed_ns: u128,
    delta_coords: usize,
}

fn delta_count(update: SdfClipmapUpdate) -> usize {
    black_box(update.len())
}

fn run_sorted(workload: Workload, config: SdfClipmapConfig) -> Vec<Sample> {
    let start = workload.center(0);
    let (mut planner, initial) = CpuSdfClipmap::with_center(config, start)
        .expect("sorted planner initial layout must fit benchmark config");
    black_box(initial.len());

    for step in 1..=WARMUP_UPDATES {
        black_box(
            planner
                .update_center(workload.center(step))
                .expect("sorted planner warmup update must succeed")
                .len(),
        );
    }

    let mut samples = Vec::with_capacity(MEASURED_UPDATES);
    for step in WARMUP_UPDATES + 1..=WARMUP_UPDATES + MEASURED_UPDATES {
        let center = black_box(workload.center(step));
        let start = Instant::now();
        let update = planner
            .update_center(center)
            .expect("sorted planner measured update must succeed");
        let elapsed_ns = start.elapsed().as_nanos();
        let delta_coords = delta_count(update);
        samples.push(Sample {
            elapsed_ns,
            delta_coords,
        });
    }
    black_box(planner.center());
    samples
}

fn run_toroidal(workload: Workload, config: SdfClipmapConfig) -> Vec<Sample> {
    let start = workload.center(0);
    let (mut planner, initial) = SdfToroidalClipmap::with_center(config, start)
        .expect("toroidal planner initial layout must fit benchmark config");
    black_box(initial.len());

    for step in 1..=WARMUP_UPDATES {
        black_box(
            planner
                .update_center(workload.center(step))
                .expect("toroidal planner warmup update must succeed")
                .len(),
        );
    }

    let mut samples = Vec::with_capacity(MEASURED_UPDATES);
    for step in WARMUP_UPDATES + 1..=WARMUP_UPDATES + MEASURED_UPDATES {
        let center = black_box(workload.center(step));
        let start = Instant::now();
        let update = planner
            .update_center(center)
            .expect("toroidal planner measured update must succeed");
        let elapsed_ns = start.elapsed().as_nanos();
        let delta_coords = delta_count(update);
        samples.push(Sample {
            elapsed_ns,
            delta_coords,
        });
    }
    black_box(planner.center());
    samples
}

fn percentile_index(length: usize, percentile: usize) -> usize {
    assert!(length > 0);
    assert!((1..=100).contains(&percentile));
    (length * percentile).div_ceil(100).saturating_sub(1)
}

fn report(planner_name: &str, mut samples: Vec<Sample>) {
    let mut by_time = samples.clone();
    by_time.sort_by_key(|sample| sample.elapsed_ns);
    samples.sort_by_key(|sample| sample.delta_coords);

    let at_time_percentile = |percentile| by_time[percentile_index(by_time.len(), percentile)];
    let at_delta_percentile =
        |percentile| samples[percentile_index(samples.len(), percentile)].delta_coords;
    let p50 = at_time_percentile(50);
    let p95 = at_time_percentile(95);
    let p99 = at_time_percentile(99);
    let min_delta = samples.first().expect("samples are non-empty").delta_coords;
    let max_delta = samples.last().expect("samples are non-empty").delta_coords;

    println!("  planner={planner_name}");
    println!(
        "    update_us p50={:.3} p95={:.3} p99={:.3}",
        p50.elapsed_ns as f64 / 1_000.0,
        p95.elapsed_ns as f64 / 1_000.0,
        p99.elapsed_ns as f64 / 1_000.0,
    );
    println!(
        "    logical_delta_coords entering+exiting min={} p50={} p95={} p99={} max={}",
        min_delta,
        at_delta_percentile(50),
        at_delta_percentile(95),
        at_delta_percentile(99),
        max_delta,
    );
    println!(
        "    delta_at_time_percentiles p50={} p95={} p99={}",
        p50.delta_coords, p95.delta_coords, p99.delta_coords,
    );
    println!("    measured_updates={MEASURED_UPDATES} warmup_updates={WARMUP_UPDATES}");
}

fn main() {
    let config = SdfClipmapConfig::new(MIP_LEVELS, RING_RADIUS, UPDATE_BUDGET_BYTES)
        .expect("benchmark clipmap configuration is valid");
    println!("prism_world_residency sdf_clipmap CPU benchmark");
    println!("  profile=cargo bench default optimized profile");
    println!("  config.mip_levels={MIP_LEVELS}");
    println!("  config.ring_radius_bricks_per_mip={RING_RADIUS}");
    println!(
        "  config.cells_per_mip={} config.total_layout_cells={}",
        config.cells_per_mip().expect("valid footprint"),
        config.layout_cell_count().expect("valid layout"),
    );
    println!("  config.max_update_bytes={UPDATE_BUDGET_BYTES}");
    println!("  warmup_updates={WARMUP_UPDATES} measured_updates={MEASURED_UPDATES}");
    println!("  timing_scope=update_center only; initial construction excluded");
    println!("  interpretation=host CPU planner only; not GPU time or product frame time");

    for workload in Workload::ALL {
        println!("scenario={}", workload.name());
        let sorted = run_sorted(workload, config);
        let toroidal = run_toroidal(workload, config);
        assert_eq!(sorted.len(), toroidal.len());
        for (step, (sorted_sample, toroidal_sample)) in sorted.iter().zip(&toroidal).enumerate() {
            assert_eq!(
                sorted_sample.delta_coords,
                toroidal_sample.delta_coords,
                "logical delta count diverged in {} sample {step}",
                workload.name(),
            );
        }
        report("CpuSdfClipmap", sorted);
        report("SdfToroidalClipmap", toroidal);
    }
}
