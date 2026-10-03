//! Virtual-memory commit/touch/decommit throughput (roadmap M-core "基准即规格").
//!
//! The design spec (`docs/prism_platform_design_zh.md`, §9/§22) makes the
//! page-level VM layer the substrate every engine allocator and memory budget
//! sits on: a large buffer reserves a contiguous *virtual* range up front and
//! pays for physical pages only as it commits and touches them. The cost that
//! matters is therefore the reserve -> commit -> first-touch -> decommit
//! lifecycle, which is dominated by OS syscalls (`mmap`/`mprotect`/`madvise`
//! on Unix) and first-touch page-fault servicing. This benchmark drives that
//! full lifecycle over a multi-megabyte reservation and reports the committed
//! throughput in pages/s and bytes/s, plus the per-call overhead of the
//! monotonic [`clock::now`](prism_platform::now) read that every profiler and
//! frame timer leans on.
//!
//! A fast lifecycle that never actually mapped usable memory would be
//! worthless, so every committed page is stamped with a position-dependent
//! pattern through its raw pointer and read back into a checksum; the checksum
//! is compared against an independent re-computation, and a mismatch (or a page
//! count of zero) fails the run loudly. The clock loop likewise asserts the
//! samples are non-decreasing, so a stuck or non-monotonic clock cannot post a
//! fast time.
//!
//! Dependency-free `harness = false` plain `main` using [`std::time::Instant`]:
//!
//! ```text
//! cargo bench -p prism_platform --bench vm_commit
//! ```
//!
//! # Provenance
//!
//! This is an original benchmark authored for Prism. It contains no Unreal
//! Engine source or derived code.
#![expect(
    clippy::print_stdout,
    reason = "a benchmark binary reports its timing results to stdout"
)]
#![expect(
    unsafe_code,
    reason = "the benchmark writes and reads committed VM pages through raw pointers to prove the commit actually mapped usable memory"
)]

use std::hint::black_box;
use std::time::Instant;

use prism_platform::{Protection, Reservation, now, page_size, virtual_memory_supported};

/// Target reservation size in bytes (rounded to whole pages).
const RESERVE_BYTES: usize = 64 * 1024 * 1024;
/// Timed passes for the VM lifecycle; best-of is reported.
const VM_PASSES: usize = 16;
/// Clock reads per timed pass.
const CLOCK_READS: usize = 1 << 20;
/// Timed passes for the clock-read micro-bench.
const CLOCK_PASSES: usize = 32;

/// Position-dependent byte pattern, integer-only (no stdlib trig: the workspace
/// forbids `f32::sin` et al. in benches), so the checksum guard is cheap and
/// deterministic.
fn stamp_byte(page_index: usize) -> u8 {
    let mix = page_index.wrapping_mul(2_654_435_761);
    (mix ^ (mix >> 13)) as u8
}

/// Commit `res` fully as read-write, stamp one byte at the head of every page,
/// and fold those bytes into a checksum. Writing the first byte of each page is
/// enough to force a first-touch fault per page while keeping the loop bounded
/// by page faults rather than memory bandwidth.
///
/// # Safety
///
/// `res` must be a live reservation of at least `len` bytes; the committed
/// range `[0, len)` is written and read through its raw pointer and is not
/// aliased elsewhere for the duration of this call.
unsafe fn commit_touch_checksum(res: &Reservation, len: usize, page: usize) -> u64 {
    res.commit(0, len, Protection::ReadWrite)
        .expect("commit of a fresh reservation should succeed");
    let base = res.as_mut_ptr();
    let pages = len / page;
    let mut checksum: u64 = 0;
    for p in 0..pages {
        let byte = stamp_byte(p);
        // SAFETY: `p * page` is < len and the whole range is committed RW.
        unsafe {
            let slot = base.add(p * page);
            slot.write_volatile(byte);
            checksum = checksum.wrapping_add(u64::from(slot.read_volatile()));
        }
    }
    checksum
}

/// Independent re-computation of the checksum produced by
/// [`commit_touch_checksum`], used purely as a correctness oracle.
fn expected_checksum(pages: usize) -> u64 {
    let mut checksum: u64 = 0;
    for p in 0..pages {
        checksum = checksum.wrapping_add(u64::from(stamp_byte(p)));
    }
    checksum
}

fn main() {
    println!("prism_platform :: vm_commit benchmark");
    println!("(original Prism benchmark; no Unreal Engine code)");

    if !virtual_memory_supported() {
        println!("virtual memory unsupported on this platform; nothing to measure");
        return;
    }

    let page = page_size();
    assert!(page >= 4096, "page size should be at least 4 KiB, got {page}");

    let len = (RESERVE_BYTES / page) * page;
    let pages = len / page;
    assert!(pages > 0, "reservation must span at least one page");
    let oracle = expected_checksum(pages);

    // --- VM reserve/commit/touch/decommit lifecycle ------------------------
    let mut best_vm = f64::INFINITY;
    for _ in 0..VM_PASSES {
        let res = Reservation::reserve(len).expect("reserve should succeed");
        let start = Instant::now();
        // SAFETY: `res` is a fresh, exclusively-owned reservation of `len` bytes.
        let checksum = unsafe { commit_touch_checksum(&res, len, page) };
        let elapsed = start.elapsed().as_secs_f64();
        assert_eq!(
            black_box(checksum),
            oracle,
            "committed pages did not read back the stamped pattern"
        );
        res.decommit(0, len)
            .expect("decommit of a committed range should succeed");
        if elapsed < best_vm {
            best_vm = elapsed;
        }
    }
    let pages_per_sec = pages as f64 / best_vm;
    let bytes_per_sec = len as f64 / best_vm;
    println!(
        "VM commit+touch: {pages} pages ({} MiB) in {:.3} ms  =>  {:.2} Mpages/s, {:.2} GiB/s",
        len / (1024 * 1024),
        best_vm * 1e3,
        pages_per_sec / 1e6,
        bytes_per_sec / (1024.0 * 1024.0 * 1024.0),
    );

    // --- Monotonic clock read overhead -------------------------------------
    let mut best_clock = f64::INFINITY;
    for _ in 0..CLOCK_PASSES {
        let mut prev = now();
        let first = prev;
        let start = Instant::now();
        for _ in 0..CLOCK_READS {
            let t = black_box(now());
            assert!(t.0 >= prev.0, "monotonic clock went backwards");
            prev = t;
        }
        let elapsed = start.elapsed().as_secs_f64();
        assert!(
            prev.saturating_since(first) < u64::MAX,
            "clock span overflowed"
        );
        if elapsed < best_clock {
            best_clock = elapsed;
        }
    }
    let ns_per_read = best_clock * 1e9 / CLOCK_READS as f64;
    let reads_per_sec = CLOCK_READS as f64 / best_clock;
    println!(
        "clock::now(): {CLOCK_READS} reads in {:.3} ms  =>  {:.2} ns/read, {:.2} Mreads/s",
        best_clock * 1e3,
        ns_per_read,
        reads_per_sec / 1e6,
    );

    println!("guards passed: checksum == oracle ({oracle}); clock monotonic over all passes");
}
