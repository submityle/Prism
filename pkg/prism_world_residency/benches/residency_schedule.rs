//! Ten-thousand-record residency scheduling benchmark.
//!
//! The benchmark uses the repository's dependency-free custom harness and
//! reports scheduling throughput together with heap activity per planning pass.
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

use prism_world_residency::{MaterializationBudget, ResidencyAction, ResidencyPolicy};
use prism_world_residency::{ResidencyRecord, ResidencyScheduler, ResidencyState};

const RECORDS: usize = 10_000;
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

fn records() -> Vec<ResidencyRecord> {
    let mut records = Vec::with_capacity(RECORDS);
    for index in 0..RECORDS {
        let state = match index % 3 {
            0 => ResidencyState::TierA,
            1 => ResidencyState::TierB,
            _ => ResidencyState::TierC,
        };
        records.push(ResidencyRecord::new(
            index as u64,
            state,
            (index % 240) as u32,
            (RECORDS - index) as u32,
            8 + (index % 5) as u32,
            4096 + (index % 11) as u64 * 512,
        ));
    }
    records
}

fn sample(
    records: &[ResidencyRecord],
    scheduler: ResidencyScheduler,
) -> (f64, usize, usize, AllocationStats) {
    reset_stats();
    let start = Instant::now();
    let plan = scheduler.plan(
        120,
        records,
        MaterializationBudget::new(120_000, 64 * 1024 * 1024),
    );
    let elapsed = start.elapsed().as_secs_f64();
    let mut promotions = 0;
    let mut demotions = 0;
    for action in &plan.actions {
        match action {
            ResidencyAction::Promote { .. } => promotions += 1,
            ResidencyAction::Demote { .. } => demotions += 1,
            ResidencyAction::Keep { .. } => {}
        }
    }
    assert_eq!(plan.actions.len(), RECORDS, "every record needs one action");
    black_box(&plan);
    drop(plan);
    (elapsed, promotions, demotions, stats())
}

fn main() {
    let policy = ResidencyPolicy::new(16, 32, 96, 160, 0).expect("benchmark policy is ordered");
    let scheduler = ResidencyScheduler::new(policy);
    let records = records();
    let _ = sample(&records, scheduler);

    let mut best = f64::INFINITY;
    let mut best_promotions = 0;
    let mut best_demotions = 0;
    let mut best_stats = AllocationStats {
        allocations: 0,
        allocated_bytes: 0,
        deallocations: 0,
        deallocated_bytes: 0,
    };
    for _ in 0..PASSES {
        let (elapsed, promotions, demotions, allocation_stats) = sample(&records, scheduler);
        if elapsed < best {
            best = elapsed;
            best_promotions = promotions;
            best_demotions = demotions;
            best_stats = allocation_stats;
        }
    }

    println!("prism_world_residency residency_schedule");
    println!("  records         : {RECORDS}");
    println!("  best schedule   : {:.3} ms", best * 1e3);
    println!("  records/s       : {:.2} M", RECORDS as f64 / best / 1e6);
    println!("  promotions      : {best_promotions}");
    println!("  demotions       : {best_demotions}");
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
