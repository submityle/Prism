//! Facade tick benchmark over streaming and residency policy plans.
//!
//! This keeps the facade path observable in addition to the focused crate
//! benchmarks, using the repository's dependency-free custom harness.
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

use prism_world::{WorldBudget, WorldRuntime};
use prism_world_residency::{
    MaterializationBudget, ResidencyPolicy, ResidencyRecord, ResidencyState,
};
use prism_world_stream::{AabbI64, CellManifest, DemandSource, GlobalPos, SectorId, StreamBudget};

const ITEMS: usize = 10_000;
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

fn fixtures() -> (Vec<DemandSource>, Vec<CellManifest>, Vec<ResidencyRecord>) {
    let mut demands = Vec::with_capacity(1);
    let center = SectorId::new(0, 0, 0, 0);
    demands.push(DemandSource::new(center, 10_000, 12_000, 100).expect("radii are ordered"));

    let mut manifests = Vec::with_capacity(ITEMS);
    let mut records = Vec::with_capacity(ITEMS);
    for index in 0..ITEMS {
        let x = index as i64 - (ITEMS / 2) as i64;
        let sector = SectorId::new(0, x, 0, 0);
        manifests.push(CellManifest {
            sector,
            bounds: AabbI64::new(
                GlobalPos::new(x * 1000, 0, 0),
                GlobalPos::new(x * 1000 + 999, 999, 999),
            )
            .expect("bounds are ordered"),
            scene_key: index as u128 + 1,
            content_version: 1,
            derived_hash: index as u128,
            io_bytes: 4096,
        });
        records.push(ResidencyRecord::new(
            index as u64,
            ResidencyState::TierC,
            (index % 128) as u32,
            index as u32,
            4,
            2048,
        ));
    }
    (demands, manifests, records)
}

fn sample(
    demands: &[DemandSource],
    manifests: &[CellManifest],
    records: &[ResidencyRecord],
) -> (f64, usize, usize, AllocationStats) {
    reset_stats();
    let policy = ResidencyPolicy::new(16, 32, 96, 160, 0).expect("policy is ordered");
    let mut world = WorldRuntime::new(prism_world_residency::ResidencyScheduler::new(policy));
    let start = Instant::now();
    let plan = world.tick(
        demands,
        manifests,
        records,
        120,
        WorldBudget {
            stream: StreamBudget {
                io_bytes: u64::MAX,
                max_actions: ITEMS,
            },
            materialization: MaterializationBudget::new(ITEMS as u32 * 8, u64::MAX),
        },
    );
    let elapsed = start.elapsed().as_secs_f64();
    let stream_actions = plan.stream.actions.len();
    let residency_actions = plan.residency.actions.len();
    black_box(&plan);
    drop(plan);
    drop(world);
    (elapsed, stream_actions, residency_actions, stats())
}

fn main() {
    let (demands, manifests, records) = fixtures();
    let _ = sample(&demands, &manifests, &records);

    let mut best = f64::INFINITY;
    let mut best_stream_actions = 0;
    let mut best_residency_actions = 0;
    let mut best_stats = AllocationStats {
        allocations: 0,
        allocated_bytes: 0,
        deallocations: 0,
        deallocated_bytes: 0,
    };
    for _ in 0..PASSES {
        let (elapsed, stream_actions, residency_actions, allocation_stats) =
            sample(&demands, &manifests, &records);
        if elapsed < best {
            best = elapsed;
            best_stream_actions = stream_actions;
            best_residency_actions = residency_actions;
            best_stats = allocation_stats;
        }
    }

    assert_eq!(
        best_stream_actions, ITEMS,
        "stream plan must cover all cells"
    );
    assert_eq!(
        best_residency_actions, ITEMS,
        "residency plan must cover all records"
    );
    println!("prism_world world_tick");
    println!("  items           : {ITEMS}");
    println!("  best tick       : {:.3} ms", best * 1e3);
    println!(
        "  items/s         : {:.2} M",
        ITEMS as f64 * 2.0 / best / 1e6
    );
    println!("  stream actions  : {best_stream_actions}");
    println!("  residency acts  : {best_residency_actions}");
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
