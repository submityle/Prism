//! Reproducible ECS batch scale benchmark for `StateTree`.

#![expect(
    clippy::print_stdout,
    reason = "the benchmark output is the measured report"
)]

use std::env;
use std::hint::black_box;
use std::time::Instant;

use prism_ecs::world::World;
use prism_state_tree_core::{MicrostepLimit, StateId, StateNode, TreeDef};
use prism_state_tree_ecs::{BatchRuntime, StateTreeAsset, StateTreeInstance};

const DEFAULT_INSTANCES: usize = 10_000;
const SAMPLES: usize = 24;
const FRAMES_PER_SAMPLE: usize = 4;

fn asset() -> StateTreeAsset {
    let plan = TreeDef {
        root: StateId(0),
        nodes: vec![
            StateNode::compound(StateId(0), None, vec![StateId(1)], StateId(1), None),
            StateNode::atomic(StateId(1), Some(StateId(0))),
        ],
    }
    .compile()
    .expect("benchmark tree compiles");
    StateTreeAsset::new(plan)
}

fn percentile(samples: &mut [u128], numerator: usize, denominator: usize) -> u128 {
    samples.sort_unstable();
    let index = ((samples.len() - 1) * numerator).div_ceil(denominator);
    samples[index]
}

fn main() {
    let instances = env::var("STATE_TREE_ECS_BENCH_INSTANCES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_INSTANCES);
    let asset = asset();
    let mut world = World::new();
    for _ in 0..instances {
        world.spawn(
            StateTreeInstance::new(asset.clone_shared(), 2, MicrostepLimit(4)).expect("instance"),
        );
    }
    let mut batch = BatchRuntime::with_capacity(instances).expect("gather capacity");
    for _ in 0..2 {
        let report = batch.tick_all(&mut world).expect("warm-up batch");
        black_box(report.len());
    }

    let mut samples = [0_u128; SAMPLES];
    for sample in &mut samples {
        let start = Instant::now();
        for _ in 0..FRAMES_PER_SAMPLE {
            let report = batch.tick_all(&mut world).expect("quiet batch");
            black_box(report.completed_count());
        }
        *sample = start.elapsed().as_nanos() / FRAMES_PER_SAMPLE as u128;
    }
    let p50 = percentile(&mut samples, 50, 100);
    let p95 = percentile(&mut samples, 95, 100);
    let p99 = percentile(&mut samples, 99, 100);
    println!("prism_state_tree_ecs batch_scale");
    println!("  instances       : {instances}");
    println!("  frames/sample   : {FRAMES_PER_SAMPLE}");
    println!("  p50 ns/frame    : {p50}");
    println!("  p95 ns/frame    : {p95}");
    println!("  p99 ns/frame    : {p99}");
    println!("  gather_capacity : {}", batch.capacity());
}
