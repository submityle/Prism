//! M6 tests for the `alloc-track` allocation tracker.
//!
//! The tracker's counters are process-global statics. In the test binary the
//! `TrackingAllocator` is *not* installed as the global allocator, so the only
//! writers to those statics are the explicit calls below — which lets us assert
//! exact, symmetric accounting. All global-touching work is kept in a single
//! test to avoid races on the shared statics.

use crate::alloc_track::{
    self, AllocSnapshot, TrackingAllocator, register_tag, snapshot, tag_report, tag_scope,
};
use core::alloc::{GlobalAlloc, Layout};
use std::alloc::System;

#[test]
fn live_allocations_is_alloc_minus_free_saturating() {
    let snap = AllocSnapshot {
        live_bytes: 0,
        peak_bytes: 0,
        total_allocated: 0,
        total_freed: 0,
        alloc_count: 10,
        free_count: 4,
    };
    assert_eq!(snap.live_allocations(), 6);

    // Saturates rather than underflowing when frees somehow exceed allocs.
    let weird = AllocSnapshot {
        alloc_count: 1,
        free_count: 5,
        ..snap
    };
    assert_eq!(weird.live_allocations(), 0);
}

#[test]
fn register_tag_is_idempotent() {
    let a = register_tag("prism::test::idempotent").expect("tag slot available");
    let b = register_tag("prism::test::idempotent").expect("tag slot available");
    assert_eq!(a, b, "same name must map to the same id");
    assert_eq!(a.index(), b.index());
}

#[test]
#[expect(
    unsafe_code,
    reason = "exercising GlobalAlloc requires unsafe; this verifies the \
              tracker's real byte accounting end-to-end"
)]
fn accounting_is_exact_and_symmetric() {
    // Serialize all global-counter manipulation into this one test.
    let alloc = TrackingAllocator::new(System);
    alloc_track::set_enabled(true);
    alloc_track::reset_all();

    let base = snapshot();
    assert_eq!(base.live_bytes, 0);
    assert_eq!(base.total_allocated, 0);
    assert_eq!(base.alloc_count, 0);

    let layout = Layout::from_size_align(4096, 16).unwrap();

    // One allocation: live/total/count all move by exactly the layout size.
    // SAFETY: `layout` has non-zero size and valid alignment.
    let p1 = unsafe { alloc.alloc(layout) };
    assert!(!p1.is_null(), "backing System allocator must succeed");
    let after_alloc = snapshot();
    assert_eq!(after_alloc.live_bytes, 4096);
    assert_eq!(after_alloc.total_allocated, 4096);
    assert_eq!(after_alloc.alloc_count, 1);
    assert_eq!(after_alloc.free_count, 0);
    assert_eq!(after_alloc.peak_bytes, 4096);

    // Second allocation pushes the peak.
    // SAFETY: `layout` has non-zero size and valid alignment.
    let p2 = unsafe { alloc.alloc(layout) };
    assert!(!p2.is_null());
    assert_eq!(snapshot().live_bytes, 8192);
    assert_eq!(snapshot().peak_bytes, 8192);

    // Freeing is symmetric: live drops, total_freed rises, peak stays.
    // SAFETY: `p1` came from `alloc.alloc(layout)` and is freed once with the same layout.
    unsafe { alloc.dealloc(p1, layout) };
    let after_free = snapshot();
    assert_eq!(after_free.live_bytes, 4096);
    assert_eq!(after_free.total_freed, 4096);
    assert_eq!(after_free.free_count, 1);
    assert_eq!(after_free.peak_bytes, 8192, "peak is a high-water mark");

    // SAFETY: `p2` came from `alloc.alloc(layout)` and is freed once with the same layout.
    unsafe { alloc.dealloc(p2, layout) };
    let drained = snapshot();
    assert_eq!(drained.live_bytes, 0);
    assert_eq!(drained.live_allocations(), 0);
    assert_eq!(drained.total_allocated, drained.total_freed);

    // reset_peak lowers the high-water mark to current live (0 here).
    alloc_track::reset_peak();
    assert_eq!(snapshot().peak_bytes, 0);

    // Per-tag attribution: allocations inside a tag scope are charged to it.
    let tag = register_tag("prism::test::accounting").expect("tag slot available");
    {
        let _scope = tag_scope(tag);
        // SAFETY: `layout` has non-zero size and valid alignment.
        let p3 = unsafe { alloc.alloc(layout) };
        assert!(!p3.is_null());
        // SAFETY: `p3` came from the matching `alloc.alloc(layout)` above.
        unsafe { alloc.dealloc(p3, layout) };
    }
    // Outside the scope: not charged.
    // SAFETY: `layout` has non-zero size and valid alignment.
    let p4 = unsafe { alloc.alloc(layout) };
    assert!(!p4.is_null());
    // SAFETY: `p4` came from the matching `alloc.alloc(layout)` above.
    unsafe { alloc.dealloc(p4, layout) };

    let report = tag_report();
    let stat = report
        .iter()
        .find(|s| s.id == tag)
        .expect("registered tag appears in report");
    assert_eq!(stat.allocated_bytes, 4096, "exactly one tagged allocation");
    assert_eq!(stat.alloc_count, 1);

    // Disabling makes the allocator a pure pass-through for accounting.
    let before = snapshot();
    alloc_track::set_enabled(false);
    assert!(!alloc_track::is_enabled());
    // SAFETY: `layout` has non-zero size and valid alignment.
    let p5 = unsafe { alloc.alloc(layout) };
    assert!(!p5.is_null());
    // SAFETY: `p5` came from the matching `alloc.alloc(layout)` above.
    unsafe { alloc.dealloc(p5, layout) };
    let after = snapshot();
    assert_eq!(before.total_allocated, after.total_allocated);
    assert_eq!(before.alloc_count, after.alloc_count);

    alloc_track::set_enabled(true);
}
