//! CRDT merge and persistence codec benchmark at ten-thousand transactions.
//!
//! This dependency-free custom harness reports transaction throughput and
//! allocation behavior for merge, log encode, and log decode separately.
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

use prism_world_persist::{decode_log, encode_log, AppendOnlyLog, CrdtState};
use prism_world_persist::{ActorId, FieldId, LogicalClock, OverrideDelta, OverrideValue};
use prism_world_persist::{PersistentId, Transaction, TransactionId};

const TRANSACTIONS: usize = 10_000;
const PASSES: usize = 6;

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

fn transactions() -> Vec<Transaction> {
    let mut transactions = Vec::with_capacity(TRANSACTIONS);
    for index in 0..TRANSACTIONS {
        let field = FieldId((index % 31) as u64);
        let delta = OverrideDelta::single(field, OverrideValue::Unsigned(index as u64))
            .expect("benchmark delta is valid");
        let inverse = OverrideDelta::single(field, OverrideValue::Delete)
            .expect("benchmark inverse is valid");
        transactions.push(
            Transaction::new(
                TransactionId(index as u128 + 1),
                ActorId((index % 4) as u128 + 1),
                LogicalClock(index as u64 + 1),
                index as u64,
                PersistentId(index as u128),
                delta,
                inverse,
            )
            .expect("benchmark transaction is valid"),
        );
    }
    transactions
}

fn build_log(transactions: &[Transaction]) -> AppendOnlyLog {
    let mut log = AppendOnlyLog::new();
    for transaction in transactions {
        log.append(transaction.clone())
            .expect("benchmark log is ordered");
    }
    log
}

fn sample_merge(transactions: &[Transaction]) -> (f64, AllocationStats) {
    reset_stats();
    let start = Instant::now();
    let mut state = CrdtState::new();
    state
        .merge(transactions.iter())
        .expect("benchmark merge is valid");
    let elapsed = start.elapsed().as_secs_f64();
    assert_eq!(
        state.len(),
        TRANSACTIONS,
        "every transaction has one winning register"
    );
    black_box(&state);
    drop(state);
    (elapsed, stats())
}

fn sample_encode(log: &AppendOnlyLog) -> (f64, usize, AllocationStats) {
    reset_stats();
    let start = Instant::now();
    let bytes = encode_log(log).expect("benchmark log encodes");
    let elapsed = start.elapsed().as_secs_f64();
    let length = bytes.len();
    black_box(&bytes);
    drop(bytes);
    (elapsed, length, stats())
}

fn sample_decode(bytes: &[u8]) -> (f64, usize, AllocationStats) {
    reset_stats();
    let start = Instant::now();
    let log = decode_log(bytes).expect("benchmark bytes decode");
    let elapsed = start.elapsed().as_secs_f64();
    let length = log.len();
    black_box(&log);
    drop(log);
    (elapsed, length, stats())
}

fn main() {
    let transactions = transactions();
    let log = build_log(&transactions);
    let encoded = encode_log(&log).expect("benchmark fixture encodes");

    let _ = sample_merge(&transactions);
    let _ = sample_encode(&log);
    let _ = sample_decode(&encoded);

    let mut best_merge = f64::INFINITY;
    let mut merge_stats = AllocationStats {
        allocations: 0,
        allocated_bytes: 0,
        deallocations: 0,
        deallocated_bytes: 0,
    };
    let mut best_encode = f64::INFINITY;
    let mut encoded_length = 0;
    let mut encode_stats = merge_stats;
    let mut best_decode = f64::INFINITY;
    let mut decoded_length = 0;
    let mut decode_stats = merge_stats;
    for _ in 0..PASSES {
        let (merge_time, current_merge_stats) = sample_merge(&transactions);
        if merge_time < best_merge {
            best_merge = merge_time;
            merge_stats = current_merge_stats;
        }

        let (encode_time, current_length, current_encode_stats) = sample_encode(&log);
        if encode_time < best_encode {
            best_encode = encode_time;
            encoded_length = current_length;
            encode_stats = current_encode_stats;
        }

        let (decode_time, current_decoded_length, current_decode_stats) = sample_decode(&encoded);
        if decode_time < best_decode {
            best_decode = decode_time;
            decoded_length = current_decoded_length;
            decode_stats = current_decode_stats;
        }
    }

    assert_eq!(encoded_length, encoded.len(), "encoded size must be stable");
    assert_eq!(
        decoded_length, TRANSACTIONS,
        "decoded log length must match input"
    );
    let input_mib = encoded.len() as f64 / (1024.0 * 1024.0);
    println!("prism_world_persist crdt_codec");
    println!("  transactions    : {TRANSACTIONS}");
    println!("  encoded bytes   : {} ({input_mib:.2} MiB)", encoded.len());
    print_result("crdt merge", best_merge, TRANSACTIONS, merge_stats);
    print_result("log encode", best_encode, TRANSACTIONS, encode_stats);
    print_result("log decode", best_decode, TRANSACTIONS, decode_stats);
}

fn print_result(name: &str, seconds: f64, items: usize, stats: AllocationStats) {
    println!(
        "  {name:12}: {:.3} ms ({:.2} Mtx/s)",
        seconds * 1e3,
        items as f64 / seconds / 1e6
    );
    println!(
        "    allocations : {} ({:.1} KiB)",
        stats.allocations,
        stats.allocated_bytes as f64 / 1024.0
    );
    println!(
        "    deallocations: {} ({:.1} KiB)",
        stats.deallocations,
        stats.deallocated_bytes as f64 / 1024.0
    );
}
