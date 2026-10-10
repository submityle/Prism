//! Allocation tracking (`alloc-track` feature, design §17 / §24.3, M6).
//!
//! This module owns the process-global allocation accounting used by the
//! `mem` reconciliation layer. It is split into:
//!
//! - shared counters + tag bookkeeping (this file),
//! - [`TrackingAllocator`](tracking::TrackingAllocator): a zero-overhead,
//!   header-free [`GlobalAlloc`] wrapper that accounts live/peak/cumulative
//!   bytes exactly and attributes *cumulative* bytes to per-callsite tags, and
//! - [`LiveTrackingAllocator`](live::LiveTrackingAllocator): a header-based
//!   wrapper that additionally delivers exact per-tag **live** bytes by
//!   stamping each allocation with the tag that owned it, so a later free (on
//!   any thread, under any scope) decrements the correct tag.
//!
//! Both allocators feed the same shared counters. The header-free allocator is
//! the zero-cost default; the live allocator trades a small per-allocation
//! header (and copy-based `realloc`) for precise per-tag residency, which is
//! the §24.3 leak-attribution signal.
//!
//! The hot path performs only relaxed atomic adds and a thread-local read, and
//! never allocates, locks, or recurses into the allocator — a hard requirement
//! for a type installed as `#[global_allocator]`.
//!
//! Allocation tracking is the only area in the crate that uses `unsafe`, as
//! implementing [`GlobalAlloc`] inherently requires it. The crate-level
//! `#![forbid(unsafe_code)]` is relaxed to `deny` only when this feature is on
//! (see `lib.rs`); every `unsafe` site carries a `SAFETY` justification and
//! lives in the `tracking`/`live` submodules.

extern crate alloc;

use alloc::vec::Vec;
use core::cell::Cell;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;

mod live;
mod tracking;

pub use live::LiveTrackingAllocator;
pub use tracking::TrackingAllocator;

/// Maximum number of distinct allocation tags.
pub const MAX_TAGS: usize = 32;

/// A no-op sentinel meaning "no tag is active on this thread".
pub(crate) const UNTAGGED: usize = usize::MAX;

static LIVE_BYTES: AtomicU64 = AtomicU64::new(0);
static PEAK_BYTES: AtomicU64 = AtomicU64::new(0);
static TOTAL_ALLOCATED: AtomicU64 = AtomicU64::new(0);
static TOTAL_FREED: AtomicU64 = AtomicU64::new(0);
static ALLOC_COUNT: AtomicU64 = AtomicU64::new(0);
static FREE_COUNT: AtomicU64 = AtomicU64::new(0);
static ENABLED: AtomicBool = AtomicBool::new(true);

static TAG_BYTES: [AtomicU64; MAX_TAGS] = [const { AtomicU64::new(0) }; MAX_TAGS];
static TAG_ALLOCS: [AtomicU64; MAX_TAGS] = [const { AtomicU64::new(0) }; MAX_TAGS];
/// Per-tag bytes currently live, maintained only by [`LiveTrackingAllocator`].
static TAG_LIVE_BYTES: [AtomicU64; MAX_TAGS] = [const { AtomicU64::new(0) }; MAX_TAGS];

/// Registered tag names, keyed by [`TagId`] index. Touched only off the hot
/// path (registration + reporting), never from inside an allocation.
static TAG_NAMES: Mutex<TagNames> = Mutex::new(TagNames::new());

struct TagNames {
    names: [Option<&'static str>; MAX_TAGS],
    len: usize,
}

impl TagNames {
    const fn new() -> Self {
        Self {
            names: [None; MAX_TAGS],
            len: 0,
        }
    }
}

std::thread_local! {
    /// The active tag id for the current thread (`UNTAGGED` when none).
    static CURRENT_TAG: Cell<usize> = const { Cell::new(UNTAGGED) };
}

/// An identifier for a registered allocation tag.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TagId(usize);

impl TagId {
    /// The tag's slot index in the fixed tag table.
    pub fn index(self) -> usize {
        self.0
    }
}

/// Register (or look up) an allocation tag by name, returning its stable id.
///
/// Registration is idempotent: the same name always maps to the same [`TagId`].
/// Returns `None` once [`MAX_TAGS`] distinct tags have been registered.
pub fn register_tag(name: &'static str) -> Option<TagId> {
    let mut guard = TAG_NAMES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    for (index, slot) in guard.names.iter().enumerate().take(guard.len) {
        if *slot == Some(name) {
            return Some(TagId(index));
        }
    }
    if guard.len >= MAX_TAGS {
        return None;
    }
    let index = guard.len;
    guard.names[index] = Some(name);
    guard.len += 1;
    Some(TagId(index))
}

/// A `RAII` guard that makes `tag` the active tag for the current thread,
/// restoring the previous tag on drop. Allocations made on this thread while it
/// lives are attributed to `tag`.
#[derive(Debug)]
pub struct TagScope {
    previous: usize,
}

impl TagScope {
    /// Enter a scope attributing allocations to `tag`.
    pub fn new(tag: TagId) -> Self {
        let previous = CURRENT_TAG.with(|cell| cell.replace(tag.0));
        Self { previous }
    }
}

impl Drop for TagScope {
    fn drop(&mut self) {
        // Best-effort restore; `try_with` tolerates late-teardown access.
        let _ = CURRENT_TAG.try_with(|cell| cell.set(self.previous));
    }
}

/// Enter an allocation-tag scope for the current thread (see [`TagScope`]).
pub fn tag_scope(tag: TagId) -> TagScope {
    TagScope::new(tag)
}

#[inline]
fn current_tag() -> Option<usize> {
    match CURRENT_TAG.try_with(Cell::get) {
        Ok(id) if id < MAX_TAGS => Some(id),
        _ => None,
    }
}

/// The current thread's raw active tag index, or [`UNTAGGED`] when none is set
/// (or the thread-local is unavailable during teardown).
#[inline]
pub(crate) fn current_tag_raw() -> usize {
    match CURRENT_TAG.try_with(Cell::get) {
        Ok(id) if id < MAX_TAGS => id,
        _ => UNTAGGED,
    }
}

#[inline]
pub(crate) fn record_alloc(size: usize) {
    if !ENABLED.load(Ordering::Relaxed) {
        return;
    }
    let size = size as u64;
    ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
    TOTAL_ALLOCATED.fetch_add(size, Ordering::Relaxed);
    let live = LIVE_BYTES.fetch_add(size, Ordering::Relaxed) + size;

    // Bump the running peak to `live` if it is a new high-water mark.
    let mut peak = PEAK_BYTES.load(Ordering::Relaxed);
    while live > peak {
        match PEAK_BYTES.compare_exchange_weak(peak, live, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(observed) => peak = observed,
        }
    }

    if let Some(id) = current_tag() {
        TAG_BYTES[id].fetch_add(size, Ordering::Relaxed);
        TAG_ALLOCS[id].fetch_add(1, Ordering::Relaxed);
    }
}

#[inline]
pub(crate) fn record_free(size: usize) {
    if !ENABLED.load(Ordering::Relaxed) {
        return;
    }
    let size = size as u64;
    FREE_COUNT.fetch_add(1, Ordering::Relaxed);
    TOTAL_FREED.fetch_add(size, Ordering::Relaxed);

    // Saturating subtract guards the enable/disable boundary (a block allocated
    // while disabled could be freed while enabled); live bytes never underflow.
    let mut current = LIVE_BYTES.load(Ordering::Relaxed);
    loop {
        let next = current.saturating_sub(size);
        match LIVE_BYTES.compare_exchange_weak(current, next, Ordering::Relaxed, Ordering::Relaxed)
        {
            Ok(_) => break,
            Err(observed) => current = observed,
        }
    }
}

/// Add `size` to the per-tag live residency for `tag` (no-op when untagged or
/// out of range). Called by [`LiveTrackingAllocator`] on allocation.
#[inline]
pub(crate) fn record_tag_live_alloc(tag: usize, size: usize) {
    if tag < MAX_TAGS {
        TAG_LIVE_BYTES[tag].fetch_add(size as u64, Ordering::Relaxed);
    }
}

/// Subtract `size` from the per-tag live residency for `tag`, saturating at
/// zero (no-op when untagged or out of range). Called by
/// [`LiveTrackingAllocator`] on deallocation of a block stamped with `tag`.
#[inline]
pub(crate) fn record_tag_live_free(tag: usize, size: usize) {
    if tag >= MAX_TAGS {
        return;
    }
    let slot = &TAG_LIVE_BYTES[tag];
    let mut current = slot.load(Ordering::Relaxed);
    let size = size as u64;
    loop {
        let next = current.saturating_sub(size);
        match slot.compare_exchange_weak(current, next, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(observed) => current = observed,
        }
    }
}

/// An immutable snapshot of the global allocation counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AllocSnapshot {
    /// Bytes currently live (allocated and not yet freed).
    pub live_bytes: u64,
    /// Highest `live_bytes` ever observed since the last peak reset.
    pub peak_bytes: u64,
    /// Cumulative bytes ever allocated.
    pub total_allocated: u64,
    /// Cumulative bytes ever freed.
    pub total_freed: u64,
    /// Cumulative allocation operations.
    pub alloc_count: u64,
    /// Cumulative free operations.
    pub free_count: u64,
}

impl AllocSnapshot {
    /// Live allocation operations (allocations not yet matched by a free).
    pub fn live_allocations(&self) -> u64 {
        self.alloc_count.saturating_sub(self.free_count)
    }
}

/// Capture a consistent-enough snapshot of the global counters.
///
/// Reads are individually atomic (`Relaxed`); the snapshot is a point-in-time
/// readout suitable for frame-boundary reporting, not a serialized transaction.
pub fn snapshot() -> AllocSnapshot {
    AllocSnapshot {
        live_bytes: LIVE_BYTES.load(Ordering::Relaxed),
        peak_bytes: PEAK_BYTES.load(Ordering::Relaxed),
        total_allocated: TOTAL_ALLOCATED.load(Ordering::Relaxed),
        total_freed: TOTAL_FREED.load(Ordering::Relaxed),
        alloc_count: ALLOC_COUNT.load(Ordering::Relaxed),
        free_count: FREE_COUNT.load(Ordering::Relaxed),
    }
}

/// Per-tag accounting for one registered tag.
///
/// `allocated_bytes`/`alloc_count` are cumulative and maintained by both
/// allocators. `live_bytes` is the exact bytes currently outstanding under the
/// tag; it is maintained only by [`LiveTrackingAllocator`] and stays `0` under
/// the header-free [`TrackingAllocator`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TagStat {
    /// Tag identifier.
    pub id: TagId,
    /// Registered tag name.
    pub name: &'static str,
    /// Cumulative bytes allocated under this tag.
    pub allocated_bytes: u64,
    /// Cumulative allocations under this tag.
    pub alloc_count: u64,
    /// Bytes currently live under this tag (0 unless the live allocator is in
    /// use).
    pub live_bytes: u64,
}

/// Report per-tag accounting for every registered tag, in registration order.
pub fn tag_report() -> Vec<TagStat> {
    let guard = TAG_NAMES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut out = Vec::with_capacity(guard.len);
    for (index, slot) in guard.names.iter().enumerate().take(guard.len) {
        if let Some(name) = *slot {
            out.push(TagStat {
                id: TagId(index),
                name,
                allocated_bytes: TAG_BYTES[index].load(Ordering::Relaxed),
                alloc_count: TAG_ALLOCS[index].load(Ordering::Relaxed),
                live_bytes: TAG_LIVE_BYTES[index].load(Ordering::Relaxed),
            });
        }
    }
    out
}

/// Enable or disable accounting. While disabled the allocator is a pure
/// pass-through (counters stand still); intended to be toggled at quiescent
/// boundaries. Live-byte subtraction saturates so a toggle can never underflow.
pub fn set_enabled(enabled: bool) {
    ENABLED.store(enabled, Ordering::Relaxed);
}

/// Whether accounting is currently enabled.
pub fn is_enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// Reset the peak high-water mark down to the current live byte count.
pub fn reset_peak() {
    PEAK_BYTES.store(LIVE_BYTES.load(Ordering::Relaxed), Ordering::Relaxed);
}

/// Reset every global and per-tag counter to zero.
///
/// This zeroes `live_bytes` too, so only call it when no tracked allocation is
/// outstanding (e.g. a controlled test or a hard measurement boundary);
/// otherwise later frees will saturate against zero.
pub fn reset_all() {
    LIVE_BYTES.store(0, Ordering::Relaxed);
    PEAK_BYTES.store(0, Ordering::Relaxed);
    TOTAL_ALLOCATED.store(0, Ordering::Relaxed);
    TOTAL_FREED.store(0, Ordering::Relaxed);
    ALLOC_COUNT.store(0, Ordering::Relaxed);
    FREE_COUNT.store(0, Ordering::Relaxed);
    for slot in &TAG_BYTES {
        slot.store(0, Ordering::Relaxed);
    }
    for slot in &TAG_ALLOCS {
        slot.store(0, Ordering::Relaxed);
    }
    for slot in &TAG_LIVE_BYTES {
        slot.store(0, Ordering::Relaxed);
    }
}
