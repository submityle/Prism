//! Reproducible StateTree runtime scale benchmark.
//!
//! The benchmark reports percentile latency and allocation counts for quiet
//! instances. It is intentionally a plain binary so it runs offline and can
//! be copied into CI without a benchmark framework.

#![expect(
    clippy::print_stdout,
    reason = "the benchmark's output is its machine-readable human report"
)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::env;
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use prism_state_tree_core::{Instance, MicrostepLimit, NoopLifecycle, StateId, StateNode, TreeDef};

const DEFAULT_INSTANCES: usize = 100_000;
const SAMPLES: usize = 32;
const FRAMES_PER_SAMPLE: usize = 4;

struct CountingAllocator;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static DEALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: forwarding the caller-provided valid layout to the system allocator.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        DEALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: forwarding the allocation returned by this allocator.
        unsafe { System.dealloc(pointer, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn plan() -> prism_state_tree_core::Plan {
    TreeDef {
        root: StateId(0),
        nodes: vec![
            StateNode::compound(StateId(0), None, vec![StateId(1)], StateId(1), None),
            StateNode::atomic(StateId(1), Some(StateId(0))),
        ],
    }
    .compile()
    .expect("benchmark tree compiles")
}

fn percentile(samples: &mut [u128], numerator: usize, denominator: usize) -> u128 {
    samples.sort_unstable();
    let index = ((samples.len() - 1) * numerator).div_ceil(denominator);
    samples[index]
}

fn main() {
    let instances = env::var("STATE_TREE_BENCH_INSTANCES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_INSTANCES);
    let plan = std::sync::Arc::new(plan());
    let mut runtimes = Vec::with_capacity(instances);
    for _ in 0..instances {
        runtimes.push(
            Instance::new_shared(plan.clone(), 1, MicrostepLimit(4))
                .expect("benchmark instance allocates with valid capacity"),
        );
    }

    let mut callback = NoopLifecycle;
    for runtime in &mut runtimes {
        runtime
            .start(&mut callback)
            .expect("initial stabilization succeeds");
    }
    black_box(runtimes.len());
    ALLOCATIONS.store(0, Ordering::Relaxed);
    DEALLOCATIONS.store(0, Ordering::Relaxed);

    let mut samples = [0_u128; SAMPLES];
    for sample in &mut samples {
        let start = Instant::now();
        for _ in 0..FRAMES_PER_SAMPLE {
            for runtime in &mut runtimes {
                runtime
                    .run(&mut callback)
                    .expect("quiet runtime stays stable");
                black_box(runtime.active_states());
            }
        }
        *sample = start.elapsed().as_nanos() / FRAMES_PER_SAMPLE as u128;
    }

    let p50 = percentile(&mut samples, 50, 100);
    let p95 = percentile(&mut samples, 95, 100);
    let p99 = percentile(&mut samples, 99, 100);
    let allocations = ALLOCATIONS.load(Ordering::Relaxed);
    let deallocations = DEALLOCATIONS.load(Ordering::Relaxed);

    println!("prism_state_tree_core runtime_scale");
    println!("  instances       : {instances}");
    println!("  frames/sample   : {FRAMES_PER_SAMPLE}");
    println!("  p50 ns/frame    : {p50}");
    println!("  p95 ns/frame    : {p95}");
    println!("  p99 ns/frame    : {p99}");
    println!("  allocations     : {allocations}");
    println!("  deallocations   : {deallocations}");
    assert_eq!(
        allocations, 0,
        "quiet runtime must not allocate after warm-up"
    );
    assert_eq!(
        deallocations, 0,
        "quiet runtime must not deallocate after warm-up"
    );
}
