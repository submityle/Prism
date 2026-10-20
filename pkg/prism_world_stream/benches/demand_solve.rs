//! Ten-thousand-cell demand solve benchmark.
//!
//! This is a dependency-free `harness = false` benchmark. It reports the
//! best solve time, admitted-action throughput, and allocation behavior for a
//! deterministic multi-source demand workload.
#![expect(
    clippy::print_stdout,
    reason = "the benchmark reports its timing and allocation results"
)]
#![expect(
    unsafe_code,
    reason = "the benchmark uses a forwarding allocator to count allocations"
)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use prism_world_stream::{AabbI64, CellManifest, DemandSource, GlobalPos, SectorId};
use prism_world_stream::{StreamBudget, StreamScheduler};

const MANIFESTS: usize = 10_000;
const DEMANDS: usize = 16;
const PASSES: usize = 8;

struct CountingAllocator;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static ALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);
static DEALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static DEALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        ALLOCATED_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        // SAFETY: the caller supplied a valid allocation layout.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        DEALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        DEALLOCATED_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        // SAFETY: the pointer and layout came from this allocator.
        unsafe { System.dealloc(pointer, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[derive(Clone, Copy)]
struct AllocationStats {
    allocations: usize,
    allocated_bytes: usize,
    deallocations: usize,
    deallocated_bytes: usize,
}

fn reset_stats() {
    ALLOCATIONS.store(0, Ordering::Relaxed);
    ALLOCATED_BYTES.store(0, Ordering::Relaxed);
    DEALLOCATIONS.store(0, Ordering::Relaxed);
    DEALLOCATED_BYTES.store(0, Ordering::Relaxed);
}

fn stats() -> AllocationStats {
    AllocationStats {
        allocations: ALLOCATIONS.load(Ordering::Relaxed),
        allocated_bytes: ALLOCATED_BYTES.load(Ordering::Relaxed),
        deallocations: DEALLOCATIONS.load(Ordering::Relaxed),
        deallocated_bytes: DEALLOCATED_BYTES.load(Ordering::Relaxed),
    }
}

fn workload() -> (Vec<DemandSource>, Vec<CellManifest>, u64) {
    let mut demands = Vec::with_capacity(DEMANDS);
    for index in 0..DEMANDS {
        let center = SectorId::new(0, index as i64 * 97 - 700, 0, 0);
        demands.push(
            DemandSource::new(center, 10_000, 12_000, 100 + index as u16)
                .expect("benchmark demand radii are ordered")
                .with_prediction(SectorId::new(0, center.x + 1, 0, 0)),
        );
    }

    let mut manifests = Vec::with_capacity(MANIFESTS);
    let mut expected_bytes = 0u64;
    for index in 0..MANIFESTS {
        let x = index as i64 - (MANIFESTS / 2) as i64;
        let y = (index % 17) as i64 - 8;
        let z = (index % 31) as i64 - 15;
        let sector = SectorId::new(0, x, y, z);
        let io_bytes = 4_096 + (index % 7) as u64 * 512;
        manifests.push(CellManifest {
            sector,
            bounds: AabbI64::new(
                GlobalPos::new(x * 1000, y * 1000, z * 1000),
                GlobalPos::new(x * 1000 + 999, y * 1000 + 999, z * 1000 + 999),
            )
            .expect("benchmark bounds are ordered"),
            scene_key: index as u128 + 1,
            content_version: 1,
            derived_hash: index as u128 ^ 0xA5A5,
            io_bytes,
            data_layers: prism_world_stream::DataLayerMask::EMPTY,
        });
        expected_bytes += io_bytes;
    }
    (demands, manifests, expected_bytes)
}

fn sample(
    demands: &[DemandSource],
    manifests: &[CellManifest],
) -> (f64, usize, u64, AllocationStats) {
    reset_stats();
    let mut scheduler = StreamScheduler::new();
    let start = Instant::now();
    let plan = scheduler.plan(
        demands,
        manifests,
        StreamBudget {
            io_bytes: u64::MAX,
            max_actions: MANIFESTS,
        },
    );
    let elapsed = start.elapsed().as_secs_f64();
    let actions = plan.actions.len();
    let admitted_bytes = plan.admitted_bytes;
    black_box(&plan);
    drop(plan);
    drop(scheduler);
    (elapsed, actions, admitted_bytes, stats())
}

fn main() {
    let (demands, manifests, expected_bytes) = workload();
    let _ = sample(&demands, &manifests);

    let mut best = f64::INFINITY;
    let mut best_actions = 0;
    let mut best_bytes = 0;
    let mut best_stats = AllocationStats {
        allocations: 0,
        allocated_bytes: 0,
        deallocations: 0,
        deallocated_bytes: 0,
    };
    for _ in 0..PASSES {
        let (elapsed, actions, admitted_bytes, allocation_stats) = sample(&demands, &manifests);
        if elapsed < best {
            best = elapsed;
            best_actions = actions;
            best_bytes = admitted_bytes;
            best_stats = allocation_stats;
        }
    }

    assert_eq!(
        best_actions, MANIFESTS,
        "all benchmark manifests must be admitted"
    );
    assert_eq!(
        best_bytes, expected_bytes,
        "all benchmark bytes must be admitted"
    );
    let solves_per_second = 1.0 / best;
    let actions_per_second = best_actions as f64 / best;
    println!("prism_world_stream demand_solve");
    println!("  manifests       : {MANIFESTS}");
    println!("  demand sources  : {DEMANDS}");
    println!("  best solve      : {:.3} ms", best * 1e3);
    println!("  solves/s        : {:.2}", solves_per_second);
    println!("  actions/s       : {:.2} M", actions_per_second / 1e6);
    println!(
        "  allocations     : {} ({:.1} KiB)",
        best_stats.allocations,
        best_stats.allocated_bytes as f64 / 1024.0
    );
    println!(
        "  deallocations   : {} ({:.1} KiB)",
        best_stats.deallocations,
        best_stats.deallocated_bytes as f64 / 1024.0
    );
}
