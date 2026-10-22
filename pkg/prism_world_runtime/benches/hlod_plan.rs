//! Release-mode HLOD selection latency across representative catalog sizes.
//!
//! This is a diagnostic benchmark, not a pass/fail performance gate. It keeps
//! the catalog warm, measures repeated plans, and reports per-request batch
//! latency percentiles. Run with `cargo bench -p prism_world_runtime --bench
//! hlod_plan`.

use std::hint::black_box;
use std::time::Instant;

use prism_world_residency::{HlodCluster, HlodLevel};
use prism_world_runtime::{HlodDistanceRequest, HlodRuntime, HlodSelection, HlodSelectionBudget};

const SAMPLE_COUNT: usize = 101;
const TARGET_REQUESTS_PER_SAMPLE: usize = 16_384;

fn make_runtime(cluster_count: usize) -> HlodRuntime {
    let max_output_bytes = cluster_count
        .checked_mul(size_of::<HlodSelection>())
        .expect("benchmark output budget fits usize");
    let budget = HlodSelectionBudget::new(cluster_count, cluster_count, max_output_bytes)
        .expect("benchmark budget is valid");
    let mut runtime = HlodRuntime::new(budget);
    for id in 0..u64::try_from(cluster_count).expect("cluster count fits u64") {
        let levels = vec![
            HlodLevel {
                asset_key: id * 4 + 1,
                enter_distance: 0,
                transition_width: 8,
            },
            HlodLevel {
                asset_key: id * 4 + 2,
                enter_distance: 128,
                transition_width: 8,
            },
            HlodLevel {
                asset_key: id * 4 + 3,
                enter_distance: 512,
                transition_width: 8,
            },
        ];
        runtime
            .insert(HlodCluster::new(id, vec![id], levels).expect("valid benchmark cluster"))
            .expect("cluster fits catalog budget");
    }
    runtime
}

fn percentile(sorted: &[f64], numerator: usize, denominator: usize) -> f64 {
    let index = (sorted.len() - 1) * numerator / denominator;
    sorted[index]
}

fn measure(request_count: usize) -> (f64, f64, f64) {
    let mut runtime = make_runtime(request_count);
    let requests: Vec<_> = (0..request_count)
        .map(|id| HlodDistanceRequest {
            cluster_id: u64::try_from(id).expect("request id fits u64"),
            distance: 0,
        })
        .collect();
    let rounds = (TARGET_REQUESTS_PER_SAMPLE / request_count).clamp(4, 1_024);

    // Warm allocations, ordered-map paths, and active-level hysteresis entries.
    for _ in 0..rounds {
        black_box(runtime.plan(black_box(&requests)).expect("warm-up plan"));
    }

    let mut samples = Vec::with_capacity(SAMPLE_COUNT);
    for _ in 0..SAMPLE_COUNT {
        let start = Instant::now();
        for _ in 0..rounds {
            black_box(runtime.plan(black_box(&requests)).expect("measured plan"));
        }
        let elapsed = start.elapsed().as_nanos() as f64;
        samples.push(elapsed / (rounds * request_count) as f64);
    }
    samples.sort_by(f64::total_cmp);
    (
        percentile(&samples, 50, 100),
        percentile(&samples, 95, 100),
        percentile(&samples, 99, 100),
    )
}

fn main() {
    println!("HLOD plan steady-state latency (ns/request; diagnostic only)");
    println!("clusters=requests  p50  p95  p99");
    for request_count in [16, 64, 256, 1_024] {
        let (p50, p95, p99) = measure(request_count);
        println!("{request_count:>16}  {p50:>8.2}  {p95:>8.2}  {p99:>8.2}");
    }
}
