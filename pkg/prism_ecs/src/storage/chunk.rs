//! Physical chunk change summaries: the coarse half of double-layer change
//! detection described in the design doc (§7 脏块访问器, §10 chunk 版本号).
//!
//! # What a "chunk" is here
//!
//! Storage uses physical chunk blocks owned by [`Table`](crate::storage::Table).
//! A chunk is a
//! fixed-size window of `rows_per_chunk` consecutive rows — and tracks one
//! change-version [`Tick`] per window. Every column of a table is sliced into
//! the same windows (same `rows_per_chunk`), so chunk index `c` always maps to
//! the same row range `[c * rpc, (c + 1) * rpc)` across every column.
//!
//! # Core invariant
//!
//! For every chunk `c`:
//!
//! ```text
//! summary(c) contains every changed-tick of every live row in chunk c
//! ```
//!
//! The summary is a conservative circular interval over all u32 ticks, not a
//! numeric maximum. Intersecting it with the observer window never produces a
//! false negative, including for historical windows and tick wraparound. It may
//! over-approximate (flag a chunk whose only recently-changed row has since been
//! removed), which the design explicitly permits — the coarse layer yields a
//! *superset* of changed rows and the per-row ticks remain the fine filter.
//!
//! # Interior mutability
//!
//! A chunk version cell is shared by every row in the chunk, so the `&mut T` /
//! [`Mut`](crate::change::Mut) fetch cannot thread a `&mut Tick` to it: two
//! `Mut`s into the same chunk (e.g. `iter_mut().collect()`) would alias the one
//! cell, which is undefined behaviour. Each cell therefore lives **inside its
//! chunk block's header** (design §5.3) and is reached and bumped through a raw
//! pointer to [`ChunkChange`]. All shared accesses use atomics: parallel row
//! batches may split the SAME chunk. The metadata does not synchronize component
//! bytes; the scheduler still provides unique access to each written row.

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;
use core::cell::UnsafeCell;
use core::hint::spin_loop;
use core::mem::size_of;
use core::ops::{Deref, DerefMut};
use core::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering};

use crate::change::Tick;
use crate::component::ComponentId;

/// An atomic, conservative circular interval of changed ticks plus a write
/// revision. Embedded in a physical block, shared by every row of its column.
///
/// The interval includes obsolete ticks until the periodic clamping pass;
/// false positives are allowed, false negatives are not. `revision` is separate
/// from the tick summary: replaying an older tick still invalidates caches.
pub struct ChunkChange {
    range: AtomicU64,
    revision: AtomicU64,
    // Non-owning link into the enclosing ChunkVersions' stable boxed tree.
    // Structural edits and dropping the tracker require exclusive access;
    // detach/drop clear leaf links before the tree's allocations are released.
    parent: AtomicPtr<ChunkChange>,
    // Stable, heap-owned candidate journal. Only leaf cells use these fields;
    // branch nodes leave them null/zero.
    event_log: AtomicPtr<ChangeLog>,
    event_chunk: AtomicUsize,
    event_generation: AtomicU64,
    last_event_batch: AtomicU64,
    candidate_visit: AtomicU64,
}

#[derive(Clone, Copy)]
struct ChunkEvent {
    chunk: usize,
    generation: u64,
}

struct ChangeBatch {
    id: u64,
    chunks: Vec<ChunkEvent>,
}

/// A change-volume index grouped by the tick at which chunks changed.
///
/// The interval tree remains the conservative correctness index. This journal
/// is the strict-incremental discovery path: a narrow observer window skips
/// old batches without looking at their chunk members, then visits only the
/// chunks recorded in matching batches. The lock is deliberately tiny and is
/// used only while appending or taking a read snapshot; payload access remains
/// governed by ECS scheduling.
struct ChangeLog {
    locked: AtomicBool,
    batches: UnsafeCell<BTreeMap<u32, ChangeBatch>>,
    next_id: AtomicU64,
    next_visit: AtomicU64,
}

// SAFETY: the log's spin lock serializes all access to its interior map, and
// the map contains only owned event data.
unsafe impl Send for ChangeLog {}
// SAFETY: shared access always acquires the log's spin lock before touching
// the interior map.
unsafe impl Sync for ChangeLog {}

struct ChangeLogGuard<'a> {
    log: &'a ChangeLog,
}

impl Deref for ChangeLogGuard<'_> {
    type Target = BTreeMap<u32, ChangeBatch>;

    fn deref(&self) -> &Self::Target {
        // SAFETY: the guard owns the log's spin lock.
        unsafe { &*self.log.batches.get() }
    }
}

impl DerefMut for ChangeLogGuard<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: the guard owns the log's spin lock.
        unsafe { &mut *self.log.batches.get() }
    }
}

impl Drop for ChangeLogGuard<'_> {
    fn drop(&mut self) {
        self.log.locked.store(false, Ordering::Release);
    }
}

impl ChangeLog {
    fn new() -> Self {
        Self {
            locked: AtomicBool::new(false),
            batches: UnsafeCell::new(BTreeMap::new()),
            next_id: AtomicU64::new(1),
            next_visit: AtomicU64::new(1),
        }
    }

    fn lock(&self) -> ChangeLogGuard<'_> {
        while self
            .locked
            .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            spin_loop();
        }
        ChangeLogGuard { log: self }
    }

    fn append(&self, cell: &ChunkChange, tick: Tick) {
        let mut batches = self.lock();
        let key = tick.get();
        let id = if let Some(batch) = batches.get(&key) {
            batch.id
        } else {
            let id = self.next_id.fetch_add(1, Ordering::Relaxed);
            batches.insert(
                key,
                ChangeBatch {
                    id,
                    chunks: Vec::new(),
                },
            );
            id
        };

        // One chunk appears at most once in a contiguous tick batch. If an
        // older tick is replayed after a newer one, it gets a new batch and is
        // intentionally kept: wrapping/history semantics must remain exact.
        if cell.last_event_batch.load(Ordering::Relaxed) != id {
            cell.last_event_batch.store(id, Ordering::Relaxed);
            let event = ChunkEvent {
                chunk: cell.event_chunk.load(Ordering::Relaxed),
                generation: cell.event_generation.load(Ordering::Relaxed),
            };
            batches.get_mut(&key).unwrap().chunks.push(event);
        }
    }

    fn visit_matching(&self, last: Tick, now: Tick, mut visit: impl FnMut(ChunkEvent)) -> usize {
        let batches = self.lock();
        let width = last.age_since(now).min(Tick::MAX_CHANGE_AGE);
        if width == 0 {
            return 0;
        }
        let start = now.get().wrapping_sub(width - 1);
        let mut tested = 0;
        let mut visit_range = |range: core::ops::RangeInclusive<u32>| {
            for (_, batch) in batches.range(range) {
                tested += 1;
                for &event in &batch.chunks {
                    visit(event);
                }
            }
        };
        if start <= now.get() {
            visit_range(start..=now.get());
        } else {
            visit_range(start..=u32::MAX);
            visit_range(0..=now.get());
        }
        tested
    }

    fn retain_recent(&self, now: Tick) {
        let mut batches = self.lock();
        batches.retain(|&tick, _| Tick::new(tick).age_since(now) <= Tick::MAX_CHANGE_AGE);
    }

    fn next_visit_id(&self) -> u64 {
        self.next_visit.fetch_add(1, Ordering::Relaxed).max(1)
    }
}

impl Default for ChunkChange {
    fn default() -> Self {
        Self::new()
    }
}

impl ChunkChange {
    // start=MAX,length=MAX is reserved for empty. A full interval is canonical
    // start=0,length=MAX, so no inhabited interval collides with this sentinel.
    const EMPTY: u64 = u64::MAX;

    /// An initialized, empty summary.
    pub const fn new() -> Self {
        Self {
            range: AtomicU64::new(Self::EMPTY),
            revision: AtomicU64::new(0),
            parent: AtomicPtr::new(core::ptr::null_mut()),
            event_log: AtomicPtr::new(core::ptr::null_mut()),
            event_chunk: AtomicUsize::new(0),
            event_generation: AtomicU64::new(0),
            last_event_batch: AtomicU64::new(0),
            candidate_visit: AtomicU64::new(0),
        }
    }

    fn pack(start: u32, length: u32) -> u64 {
        if length == u32::MAX {
            u64::from(length) << 32
        } else {
            u64::from(start) | (u64::from(length) << 32)
        }
    }

    fn unpack(range: u64) -> (u32, u32) {
        (range as u32, (range >> 32) as u32)
    }

    fn contains(start: u32, length: u32, tick: u32) -> bool {
        tick.wrapping_sub(start) <= length
    }

    /// Include a changed tick. Safe for concurrent writers of distinct rows in
    /// one chunk. Relaxed atomics suffice: ECS access barriers synchronize the
    /// values themselves; these atomics only merge metadata without lost writes.
    #[inline]
    pub fn record(&self, tick: Tick) {
        self.include(Self::pack(tick.get(), 0));
        self.revision.fetch_add(1, Ordering::Relaxed);
        let log = self.event_log.load(Ordering::Acquire);
        if !log.is_null() {
            // SAFETY: the owning ChunkVersions keeps the log alive while the
            // cell is attached; detachment clears this pointer exclusively.
            unsafe { (&*log).append(self, tick) };
        }
    }

    // Smallest of the two start-anchored arcs containing both complete arcs.
    // Merely inserting their endpoints is insufficient for wide/wrapping arcs.
    fn union(a: u64, b: u64) -> u64 {
        if a == Self::EMPTY {
            return b;
        }
        if b == Self::EMPTY {
            return a;
        }
        let (a_start, a_len) = Self::unpack(a);
        let (b_start, b_len) = Self::unpack(b);
        let from_a = u64::from(b_start.wrapping_sub(a_start)) + u64::from(b_len);
        let from_b = u64::from(a_start.wrapping_sub(b_start)) + u64::from(a_len);
        let a_length = from_a.max(u64::from(a_len));
        let b_length = from_b.max(u64::from(b_len));
        if a_length <= b_length {
            Self::pack(a_start, a_length.min(u64::from(u32::MAX)) as u32)
        } else {
            Self::pack(b_start, b_length.min(u64::from(u32::MAX)) as u32)
        }
    }

    fn include(&self, interval: u64) {
        if interval == Self::EMPTY {
            return;
        }
        let mut old = self.range.load(Ordering::Relaxed);
        loop {
            let new = Self::union(old, interval);
            if old == new {
                return;
            }
            match self
                .range
                .compare_exchange_weak(old, new, Ordering::Relaxed, Ordering::Relaxed)
            {
                Ok(_) => {
                    let parent = self.parent.load(Ordering::Relaxed);
                    if !parent.is_null() {
                        // SAFETY: ChunkVersions owns every ancestor in stable
                        // boxed storage. Its exclusive structural operations
                        // cannot overlap shared recording or traversal.
                        unsafe { (&*parent).include(new) };
                    }
                    return;
                }
                Err(actual) => old = actual,
            }
        }
    }

    /// Representative upper endpoint. Not an ordered maximum: use
    /// [`may_have_changed`](Self::may_have_changed) for observer-window tests.
    pub fn version(&self) -> Tick {
        let range = self.range.load(Ordering::Relaxed);
        if range == Self::EMPTY {
            return Tick::ZERO;
        }
        let (start, length) = Self::unpack(range);
        Tick::new(start.wrapping_add(length))
    }

    /// Write revision, independent of explicit or wrapping tick values.
    pub fn revision(&self) -> u64 {
        self.revision.load(Ordering::Relaxed)
    }

    /// Whether the summary intersects the exact modular window used by
    /// [`Tick::is_newer_than`]. This also supports observers of historical ticks.
    #[inline]
    pub fn may_have_changed(&self, last_run: Tick, this_run: Tick) -> bool {
        let width = last_run.age_since(this_run).min(Tick::MAX_CHANGE_AGE);
        if width == 0 {
            return false;
        }
        let range = self.range.load(Ordering::Relaxed);
        if range == Self::EMPTY {
            return false;
        }
        let (start, length) = Self::unpack(range);
        let window_start = this_run.get().wrapping_sub(width - 1);
        // Two inhabited circular arcs intersect iff one contains the other's
        // start (including when an arc is the entire ring).
        Self::contains(start, length, window_start)
            || Self::contains(window_start, width - 1, start)
    }

    fn check_tick(&mut self, now: Tick) {
        let range = *self.range.get_mut();
        if range == Self::EMPTY {
            return;
        }
        let (start, length) = Self::unpack(range);
        let end = start.wrapping_add(length);
        let start_age = Tick::new(start).age_since(now).min(Tick::MAX_CHANGE_AGE);
        let end_age = Tick::new(end).age_since(now).min(Tick::MAX_CHANGE_AGE);
        let youngest = if Self::contains(start, length, now.get()) {
            0
        } else {
            start_age.min(end_age)
        };
        let oldest = if Self::contains(start, length, now.get().wrapping_sub(Tick::MAX_CHANGE_AGE))
        {
            Tick::MAX_CHANGE_AGE
        } else {
            start_age.max(end_age)
        };
        *self.range.get_mut() = Self::pack(now.get().wrapping_sub(oldest), oldest - youngest);
    }
}

/// Target physical chunk size the logical windows approximate (design §5.3).
///
/// The row count per chunk is chosen so one chunk's worth of a single column is
/// roughly this many bytes, matching the DOTS-style 16 KiB chunk the physical
/// layout will eventually use.
pub const TARGET_CHUNK_BYTES: usize = 16 * 1024;

/// Hard cap on rows per chunk, so a zero-sized (marker) component — which has no
/// natural byte-derived limit — still partitions into finite windows.
const MAX_ROWS_PER_CHUNK: usize = 1 << 16;

/// Rows per logical chunk for a column whose element is `bytes_per_row` bytes.
///
/// Zero-sized rows (marker components, or a table of only ZSTs) fall back to the
/// [`MAX_ROWS_PER_CHUNK`] cap; otherwise the count is `TARGET_CHUNK_BYTES /
/// bytes_per_row`, clamped to at least one row and at most the cap.
#[inline]
pub fn rows_per_chunk(bytes_per_row: usize) -> usize {
    match TARGET_CHUNK_BYTES.checked_div(bytes_per_row) {
        // A zero-sized row (division by zero) has no byte-derived limit.
        None => MAX_ROWS_PER_CHUNK,
        Some(rows) => rows.clamp(1, MAX_ROWS_PER_CHUNK),
    }
}

/// Per-chunk change-versions for one column, partitioning the column's rows into
/// fixed `rows_per_chunk` windows.
///
/// The cells are **embedded in the owning chunk blocks** (design §5.3: a chunk
/// carries its own `change_version`): the table attaches each cell pointer when
/// it allocates a block and detaches it before freeing the block, so this type
/// never allocates the cells itself. The pointer vector grows and shrinks in
/// lockstep with the row count — it always holds exactly
/// `ceil(len / rows_per_chunk)` pointers.
pub struct ChunkVersions {
    rows_per_chunk: usize,
    /// One pointer per chunk window, into that chunk's header cell. `cells[c]`
    /// upper-bounds the newest changed-tick among rows
    /// `[c * rpc, (c + 1) * rpc)`.
    cells: Vec<*mut ChunkChange>,
    // Heap-indexed binary summaries: slot i-1 describes heap node i. Leaves
    // occupy [leaf_capacity, leaf_capacity + cells.len()) in the same heap,
    // but remain embedded in physical blocks. Boxes keep parent links stable
    // when this vector or the enclosing column moves.
    #[allow(clippy::vec_box)]
    branches: Vec<Box<ChunkChange>>,
    leaf_capacity: usize,
    events: Box<ChangeLog>,
    next_generation: u64,
}

/// Work performed by an indexed dirty-chunk traversal.
///
/// Counts metadata reads, not payload bytes, so complexity regressions can be
/// tested without machine-dependent timing thresholds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ChunkVisitStats {
    /// Number of atomic interval summaries inspected, including internal nodes.
    pub summaries_tested: usize,
    /// Number of tick batches inspected in the strict-incremental journal.
    pub event_batches_tested: usize,
    /// Number of conservative matching leaf chunks delivered to the visitor.
    pub chunks_returned: usize,
}

// SAFETY: the pointers reference heap-allocated chunk blocks owned by the same
// table. Moving the owner to another thread does not relocate those blocks;
// unique ownership transfers all access to the cells. Shared access is governed
// by the Sync discipline below, and cells are detached before blocks are freed.
unsafe impl Send for ChunkVersions {}

// SAFETY: the only non-`Sync` field is the raw cell-pointer vector. The cells
// live in the owning chunk blocks; all shared reads and writes are atomic.
// Attachment, detachment and clamping require exclusive access to the tracker.
unsafe impl Sync for ChunkVersions {}

impl ChunkVersions {
    /// A fresh, empty version set partitioning rows into `rows_per_chunk`
    /// windows.
    ///
    /// # Panics
    /// Panics if `rows_per_chunk` is zero.
    #[inline]
    pub fn new(rows_per_chunk: usize) -> Self {
        assert!(rows_per_chunk > 0, "rows_per_chunk must be non-zero");
        Self {
            rows_per_chunk,
            cells: Vec::new(),
            branches: Vec::new(),
            leaf_capacity: 1,
            events: Box::new(ChangeLog::new()),
            next_generation: 1,
        }
    }

    /// The fixed row-window size.
    #[inline]
    pub fn rows_per_chunk(&self) -> usize {
        self.rows_per_chunk
    }

    /// Attach the changed-version cell of a newly-allocated chunk block.
    ///
    /// # Safety
    /// `cell` must point at a live, aligned, initialized `ChunkChange` in writable,
    /// stable-address storage. Its owner must keep that storage alive until
    /// the cell is detached or this tracker is dropped, including when the
    /// tracker or its owner moves to another thread.
    ///
    /// External aliases may call shared atomic methods, but must not replace or
    /// exclusively borrow the cell while any tracker/row handle can access it.
    /// The cell must not already be attached to this or any other tracker.
    #[inline]
    pub unsafe fn attach_cell(&mut self, cell: *mut ChunkChange) {
        if self.cells.len() == self.leaf_capacity {
            self.grow_index();
        }
        // SAFETY: the caller guarantees a live, writable ChunkChange.
        // SAFETY: the owner detaches the live cell before freeing its block.
        // SAFETY: the owner detaches the live cell before freeing its block.
        let cell_ref = unsafe { &*cell };
        cell_ref
            .event_chunk
            .store(self.cells.len(), Ordering::Relaxed);
        cell_ref
            .event_generation
            .store(self.next_generation, Ordering::Relaxed);
        cell_ref.last_event_batch.store(0, Ordering::Relaxed);
        self.next_generation = self.next_generation.wrapping_add(1).max(1);
        self.cells.push(cell);
        self.link_leaf(self.cells.len() - 1);
    }

    /// Detach the trailing chunk's cell (its block is freed by the owner).
    #[inline]
    pub fn detach_cell(&mut self) {
        let cell = self
            .cells
            .pop()
            .expect("detach_cell on a version set with no chunks");
        // SAFETY: the owner detaches the live cell before freeing its block.
        unsafe { &*cell }
            .parent
            .store(core::ptr::null_mut(), Ordering::Relaxed);
        // SAFETY: `cell` came from the live cell list and remains valid until
        // the owner finishes detaching it; this shared reference only clears
        // the cell's event-log pointer.
        let cell_ref = unsafe { &*cell };
        cell_ref
            .event_log
            .store(core::ptr::null_mut(), Ordering::Release);
        cell_ref.event_generation.store(0, Ordering::Relaxed);
        cell_ref.last_event_batch.store(0, Ordering::Relaxed);
        self.refresh_ancestors(self.leaf_capacity + self.cells.len());
    }

    fn grow_index(&mut self) {
        let capacity = self
            .leaf_capacity
            .checked_mul(2)
            .expect("chunk index capacity overflow");
        let branches = (1..capacity)
            .map(|_| Box::new(ChunkChange::new()))
            .collect();
        // Allocate first. From here on rewiring is infallible and exclusively
        // borrowed, so no row writer can follow an old parent during the swap.
        self.branches = branches;
        self.leaf_capacity = capacity;
        for node in 2..capacity {
            let parent = &mut *self.branches[node / 2 - 1] as *mut ChunkChange;
            self.branches[node - 1]
                .parent
                .store(parent, Ordering::Relaxed);
        }
        for chunk in 0..self.cells.len() {
            self.link_leaf(chunk);
        }
    }

    fn link_leaf(&self, chunk: usize) {
        let parent = if self.leaf_capacity == 1 {
            core::ptr::null_mut()
        } else {
            (&*self.branches[(self.leaf_capacity + chunk) / 2 - 1] as *const ChunkChange).cast_mut()
        };
        // SAFETY: attached cells and stable boxed ancestors are live. This is
        // called only with exclusive structural access to the tracker.
        let cell = unsafe { &*self.cells[chunk] };
        cell.parent.store(parent, Ordering::Relaxed);
        cell.event_chunk.store(chunk, Ordering::Relaxed);
        cell.event_log.store(
            (&*self.events as *const ChangeLog).cast_mut(),
            Ordering::Release,
        );
        if !parent.is_null() {
            // SAFETY: parent is a live branch owned by self.
            unsafe { &*parent }.include(cell.range.load(Ordering::Relaxed));
        }
    }

    fn node(&self, node: usize) -> Option<&ChunkChange> {
        if node < self.leaf_capacity {
            Some(&self.branches[node - 1])
        } else {
            self.cells.get(node - self.leaf_capacity).map(|&cell| {
                // SAFETY: every attached cell stays live until detachment.
                unsafe { &*cell }
            })
        }
    }

    fn refresh_branch(&mut self, node: usize) {
        let range =
            [node * 2, node * 2 + 1]
                .into_iter()
                .fold(ChunkChange::EMPTY, |range, child| {
                    ChunkChange::union(
                        range,
                        self.node(child).map_or(ChunkChange::EMPTY, |cell| {
                            cell.range.load(Ordering::Relaxed)
                        }),
                    )
                });
        *self.branches[node - 1].range.get_mut() = range;
    }

    fn refresh_ancestors(&mut self, mut node: usize) {
        while node > 1 {
            node /= 2;
            self.refresh_branch(node);
        }
    }

    /// Visit conservative matching chunks in ascending row order, pruning
    /// entire clean subtrees instead of checking every physical chunk.
    ///
    /// A clean nonempty column tests one root summary. Sparse changes inspect
    /// the journal's matching tick batches and only their recorded leaves;
    /// rebuild/restore gaps conservatively fall back to the interval tree.
    /// The candidate list is proportional to the reported changes and does not
    /// consume changes from other observers.
    pub fn visit_changed(
        &self,
        last: Tick,
        now: Tick,
        mut visit: impl FnMut(usize),
    ) -> ChunkVisitStats {
        let mut stats = ChunkVisitStats::default();
        if self.cells.is_empty() {
            return stats;
        }
        stats.summaries_tested = 1;
        if !self
            .node(1)
            .is_some_and(|root| root.may_have_changed(last, now))
        {
            return stats;
        }

        let visit_id = self.events.next_visit_id();
        let mut candidates = Vec::new();
        stats.event_batches_tested = self.events.visit_matching(last, now, |event| {
            if let Some(&cell) = self.cells.get(event.chunk) {
                // SAFETY: attached cells remain live during shared traversal.
                let cell = unsafe { &*cell };
                if cell.event_generation.load(Ordering::Acquire) == event.generation
                    && cell.candidate_visit.swap(visit_id, Ordering::AcqRel) != visit_id
                {
                    candidates.push(event.chunk);
                }
            }
        });
        radix_sort(&mut candidates);
        candidates.dedup();
        if !candidates.is_empty() {
            stats.chunks_returned = candidates.len();
            for chunk in candidates {
                visit(chunk);
            }
        } else {
            // Rebuild/restore paths can reconstruct a conservative interval
            // without a row write event. Preserve the old tree as a complete
            // correctness fallback for that case.
            self.visit_node_after_root(1, last, now, &mut visit, &mut stats);
        }
        stats
    }

    fn visit_node_after_root(
        &self,
        node: usize,
        last: Tick,
        now: Tick,
        visit: &mut impl FnMut(usize),
        stats: &mut ChunkVisitStats,
    ) {
        let Some(cell) = self.node(node) else {
            return;
        };
        if node != 1 {
            stats.summaries_tested += 1;
            if !cell.may_have_changed(last, now) {
                return;
            }
        }
        if node >= self.leaf_capacity {
            stats.chunks_returned += 1;
            visit(node - self.leaf_capacity);
        } else {
            self.visit_node(node * 2, last, now, visit, stats);
            self.visit_node(node * 2 + 1, last, now, visit, stats);
        }
    }

    /// Find the first conservative matching chunk in `[start, end)`, bounded
    /// by the live chunk count. Clean subtrees are pruned without allocation.
    ///
    /// Testing the starting leaf first keeps dense iteration constant-work per
    /// chunk; a cold gap uses the interval tree instead of a linear scan. The
    /// returned statistics count all summary reads, including that first leaf.
    pub fn next_changed(
        &self,
        start: usize,
        end: usize,
        last: Tick,
        now: Tick,
    ) -> (Option<usize>, ChunkVisitStats) {
        let end = end.min(self.cells.len());
        let mut stats = ChunkVisitStats::default();
        if start >= end {
            return (None, stats);
        }
        stats.summaries_tested = 1;
        if self.may_have_changed(start, last, now) {
            stats.chunks_returned = 1;
            return (Some(start), stats);
        }
        let found = self.find_changed_node(
            1,
            0,
            self.leaf_capacity,
            start + 1,
            end,
            last,
            now,
            &mut stats,
        );
        stats.chunks_returned = usize::from(found.is_some());
        (found, stats)
    }

    #[allow(clippy::too_many_arguments)]
    fn find_changed_node(
        &self,
        node: usize,
        first: usize,
        width: usize,
        start: usize,
        end: usize,
        last: Tick,
        now: Tick,
        stats: &mut ChunkVisitStats,
    ) -> Option<usize> {
        if first >= end || first + width <= start {
            return None;
        }
        let cell = self.node(node)?;
        stats.summaries_tested += 1;
        if !cell.may_have_changed(last, now) {
            return None;
        }
        if width == 1 {
            return Some(first);
        }
        let half = width / 2;
        self.find_changed_node(node * 2, first, half, start, end, last, now, stats)
            .or_else(|| {
                self.find_changed_node(
                    node * 2 + 1,
                    first + half,
                    half,
                    start,
                    end,
                    last,
                    now,
                    stats,
                )
            })
    }

    fn visit_node(
        &self,
        node: usize,
        last: Tick,
        now: Tick,
        visit: &mut impl FnMut(usize),
        stats: &mut ChunkVisitStats,
    ) {
        let Some(cell) = self.node(node) else {
            return;
        };
        stats.summaries_tested += 1;
        if !cell.may_have_changed(last, now) {
            return;
        }
        if node >= self.leaf_capacity {
            stats.chunks_returned += 1;
            visit(node - self.leaf_capacity);
        } else {
            self.visit_node(node * 2, last, now, visit, stats);
            self.visit_node(node * 2 + 1, last, now, visit, stats);
        }
    }

    /// No-op: the cell storage lives in the table's chunk blocks, so the table
    /// reserves/grows it, not this type.
    pub(crate) fn reserve_rows(&mut self, _additional_rows: usize) {}

    /// Number of chunk windows currently tracked (`ceil(len / rpc)`).
    #[inline]
    pub fn chunk_count(&self) -> usize {
        self.cells.len()
    }

    /// The chunk index owning `row`.
    #[inline]
    pub fn chunk_of(&self, row: usize) -> usize {
        row / self.rows_per_chunk
    }

    /// The number of chunk windows needed to hold `len` rows.
    #[inline]
    fn chunks_for_len(&self, len: usize) -> usize {
        len.div_ceil(self.rows_per_chunk)
    }

    /// The changed-version of chunk `chunk`.
    ///
    /// # Panics
    /// Panics if `chunk >= chunk_count()`.
    #[inline]
    pub fn version(&self, chunk: usize) -> Tick {
        // SAFETY: attached cell stays live; the shared read is atomic.
        unsafe { (&*self.cells[chunk]).version() }
    }

    /// Conservative observer-window predicate for one chunk.
    pub fn may_have_changed(&self, chunk: usize, last: Tick, now: Tick) -> bool {
        // SAFETY: attached cell stays live; the shared read is atomic.
        unsafe { (&*self.cells[chunk]).may_have_changed(last, now) }
    }

    /// Revision of one chunk's column, including writes with explicit old ticks.
    pub fn revision(&self, chunk: usize) -> u64 {
        // SAFETY: attached cell stays live; the shared read is atomic.
        unsafe { (&*self.cells[chunk]).revision() }
    }

    /// Raw pointer to the changed-version cell owning `row`, for interior-mutable
    /// stamping through a shared `&self` (the `&mut T` / `Mut<T>` fetch).
    ///
    /// # Safety
    /// `row` must be `< len` of the owning column, and the caller must hold
    /// unique access to `row` (the same discipline as
    /// [`Column::changed_tick_ptr`](crate::storage::Column::changed_tick_ptr)).
    /// Only shared atomic methods may be used through the returned pointer;
    /// callers must not replace the cell or form an exclusive reference to it.
    #[inline]
    pub unsafe fn chunk_changed_ptr(&self, row: usize) -> *mut ChunkChange {
        let chunk = self.chunk_of(row);
        // SAFETY: the caller guarantees `row < len` of the owning column, so
        // `chunk < cells.len()`; the pointer targets a live header cell.
        unsafe { *self.cells.get_unchecked(chunk) }
    }

    /// Fold `tick` into the version of the chunk owning `row`, keeping
    /// whichever is more recent (never lowering the stored upper bound). Used by
    /// the structural write paths that already hold `&mut Column`
    /// (overwrite-in-place, structural tick stamping).
    ///
    /// # Panics
    /// Panics if `row` is not covered by an existing chunk window.
    #[inline]
    pub fn bump_row(&mut self, row: usize, tick: Tick) {
        let chunk = self.chunk_of(row);
        self.bump_chunk(chunk, tick);
    }

    /// Fold `tick` into chunk `chunk`'s version, keeping whichever is more
    /// recent (never lowering the stored upper bound).
    #[inline]
    fn bump_chunk(&mut self, chunk: usize, tick: Tick) {
        let cell = self.cells[chunk];
        // SAFETY: the attached cell stays live. Atomic merging also handles
        // old structural ticks and wrap without assuming a half-ring ordering.
        unsafe { (&*cell).record(tick) };
    }

    /// Account for a freshly appended row (the column just grew to `new_len`),
    /// whose value carries `tick` as its changed-tick.
    ///
    /// Grows the version vector to cover the new row and folds `tick` into the
    /// (possibly new) last chunk so the upper-bound invariant holds.
    pub fn on_push(&mut self, new_len: usize, tick: Tick) {
        debug_assert!(new_len >= 1);
        debug_assert!(
            self.cells.len() >= self.chunks_for_len(new_len),
            "the owning table must attach the chunk cell before a row is pushed",
        );
        let last_row = new_len - 1;
        let chunk = self.chunk_of(last_row);
        self.bump_chunk(chunk, tick);
    }

    /// Account for a swap-remove: the row at `removed_row` was removed and (if it
    /// was not already the last row) the former last row — carrying `moved_tick`
    /// — was swapped into its place. `new_len` is the row count *after* removal.
    ///
    /// Folds `moved_tick` into the destination chunk (the relocated row may be
    /// newer than that chunk's current bound) and then shrinks the version
    /// vector to `ceil(new_len / rpc)`.
    pub fn on_swap_remove(&mut self, removed_row: usize, new_len: usize, moved_tick: Tick) {
        // A row actually moved into `removed_row` only when it was not the last
        // row; after removal that position is in-range exactly when
        // `removed_row < new_len`.
        if removed_row < new_len {
            let chunk = self.chunk_of(removed_row);
            self.bump_chunk(chunk, moved_tick);
        }
        // The owning table detaches the trailing cell when it frees the block.
    }

    /// Clamp every chunk version against `this_run` so none can wrap around and
    /// masquerade as recent (mirrors [`Column::check_change_ticks`]).
    pub fn check_ticks(&mut self, this_run: Tick) {
        self.events.retain_recent(this_run);
        for &cell in &self.cells {
            // SAFETY: live chunk-block header cell.
            unsafe { (&mut *cell).check_tick(this_run) };
        }
        for node in (1..self.leaf_capacity).rev() {
            self.refresh_branch(node);
        }
    }

    /// Rebuild from live, already clamped row ticks during the periodic pass.
    /// Dropped/overwritten rows no longer widen the summary after this pass.
    pub(crate) fn rebuild(&mut self, chunk: usize, ticks: impl Iterator<Item = Tick>, now: Tick) {
        let mut youngest = u32::MAX;
        let mut oldest = 0;
        let cell_ptr = self.cells[chunk];
        for tick in ticks {
            let age = tick.age_since(now).min(Tick::MAX_CHANGE_AGE);
            youngest = youngest.min(age);
            oldest = oldest.max(age);
            // Rebuild can be reached after restore/clamping without a normal
            // write path. Feed all live row ticks into the journal so the
            // incremental path remains complete, not merely an optimization.
            // SAFETY: the cell is attached and the log is owned by self.
            let log = unsafe { (&*cell_ptr).event_log.load(Ordering::Acquire) };
            if !log.is_null() {
                // SAFETY: the attached cell and its owning log remain live for
                // the duration of this exclusive rebuild.
                unsafe { (&*log).append(&*cell_ptr, tick) };
            }
        }
        // SAFETY: the owner holds exclusive access to the column; no row handle
        // survives the periodic tick-clamping pass.
        let cell = unsafe { &mut *self.cells[chunk] };
        *cell.range.get_mut() = if youngest == u32::MAX {
            ChunkChange::EMPTY
        } else {
            ChunkChange::pack(now.get().wrapping_sub(oldest), oldest - youngest)
        };
        // Clamped added ticks can change even if changed ticks do not. Snapshot
        // keys must invalidate for either metadata change, not just endpoint edits.
        *cell.revision.get_mut() = cell.revision.get_mut().wrapping_add(1);
        self.refresh_ancestors(self.leaf_capacity + chunk);
    }
}

/// Sort machine-word chunk indices in linear time for a fixed-width usize.
/// This keeps event discovery `O(K)` after tick-range lookup while preserving
/// the public ascending chunk order. The eight-bit radix pass count is fixed
/// for the target platform, so it is not a data-dependent logarithmic factor.
fn radix_sort(values: &mut Vec<usize>) {
    if values.len() < 2 {
        return;
    }
    let mut scratch = vec![0; values.len()];
    let bytes = size_of::<usize>();
    for pass in 0..bytes {
        let shift = pass * 8;
        let mut counts = [0usize; 256];
        for &value in values.iter() {
            counts[(value >> shift) & 0xff] += 1;
        }
        let mut offset = 0;
        for count in &mut counts {
            let next = offset + *count;
            *count = offset;
            offset = next;
        }
        for &value in values.iter() {
            let slot = &mut counts[(value >> shift) & 0xff];
            scratch[*slot] = value;
            *slot += 1;
        }
        core::mem::swap(values, &mut scratch);
    }
}

impl Drop for ChunkVersions {
    fn drop(&mut self) {
        // The attach contract keeps physical headers alive until tracker drop.
        // Row metadata may outlive this index in low-level storage consumers;
        // clearing links prevents future writes following freed branch boxes.
        for &cell in &self.cells {
            // SAFETY: cells are still live and no shared traversal or recording
            // can overlap exclusive destruction of their owning tracker.
            unsafe { &*cell }
                .parent
                .store(core::ptr::null_mut(), Ordering::Relaxed);
            // SAFETY: the tracker exclusively owns destruction of attached
            // cells, so each pointer is still live here.
            unsafe { &*cell }
                .event_log
                .store(core::ptr::null_mut(), Ordering::Release);
            // SAFETY: see the preceding safety comment.
            unsafe { &*cell }
                .event_generation
                .store(0, Ordering::Relaxed);
        }
    }
}

/// A world-space axis-aligned bounding box a consumer derived for one chunk
/// window (design §5.3 `ChunkMeta` AABB; used by render culling, GPU resident
/// columns §15, and HLOD §13).
///
/// This is a plain value type: the ECS only *stores* it per chunk, never
/// computes it. A consumer (e.g. the render extractor) fills it in and re-fills
/// it whenever the window's [`ChunkMeta::structural_epoch`] advances.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChunkBounds {
    /// Component-wise minimum corner.
    pub min: [f32; 3],
    /// Component-wise maximum corner.
    pub max: [f32; 3],
}

impl ChunkBounds {
    /// Build bounds from explicit corners.
    #[inline]
    pub const fn new(min: [f32; 3], max: [f32; 3]) -> Self {
        Self { min, max }
    }

    /// Whether every component of `min` is `<=` the matching component of
    /// `max` (so the box is non-empty / not inverted).
    #[inline]
    pub fn is_well_formed(&self) -> bool {
        (0..3).all(|i| self.min[i] <= self.max[i])
    }

    /// Whether the point `p` lies inside the box (inclusive on both faces).
    #[inline]
    pub fn contains(&self, p: [f32; 3]) -> bool {
        (0..3).all(|i| self.min[i] <= p[i] && p[i] <= self.max[i])
    }

    /// Grow this box to also contain `other` (component-wise min/max).
    #[inline]
    pub fn union(&mut self, other: &ChunkBounds) {
        for i in 0..3 {
            self.min[i] = self.min[i].min(other.min[i]);
            self.max[i] = self.max[i].max(other.max[i]);
        }
    }
}

/// Per-chunk metadata sidecar (design §5.3 `ChunkMeta`): the optional,
/// consumer-facing fields a chunk window carries besides its change version.
///
/// The ECS owns only the *storage* and the *structural epoch*; the derived
/// fields ([`bounds`](Self::bounds), [`lod`](Self::lod), [`batch_key`](Self::batch_key),
/// [`cell`](Self::cell)) are written by consumers. The contract that makes the
/// cache sound:
///
/// * [`structural_epoch`](Self::structural_epoch) is a monotonic counter the
///   table bumps whenever the **set of rows in this window changes** — a spawn
///   into the window, or a swap-remove that adds or drops a row of it. It never
///   changes for value-only writes (those bump the chunk's change version in
///   [`ChunkVersions`] instead).
/// * A consumer that derived `bounds`/`lod`/`batch_key`/`cell` from the window's
///   rows may keep using them **only while the epoch it cached still matches**;
///   once it differs, the derived fields are stale and must be recomputed.
///
/// This split keeps the kernel independent of any particular derivation (a
/// renderer's AABB, a physics broadphase's fat bounds, a network interest set)
/// while giving them a cheap, correct invalidation signal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChunkMeta {
    structural_epoch: u64,
    /// The `structural_epoch` value `bounds` was computed at (or 0 before the
    /// first computation), so a producer can tell whether its cached bounds
    /// still correspond to this window's row set.
    bounds_epoch: u64,
    /// The chunk's `change_version` when `bounds` was computed, so a producer can
    /// tell whether any value moved since (design §15 fat-bounds update).
    bounds_version: Tick,
    bounds_revision: u64,
    /// Column whose values produced the cached bounds. Revisions are local to
    /// a column, so equal revisions from different columns are not equivalent.
    bounds_source: Option<ComponentId>,
    /// Cached world-space bounds for this window, or `None` when a consumer has
    /// not (re)computed them for the current epoch.
    pub bounds: Option<ChunkBounds>,
    /// Visibility bit for culling (design §13.2 / §15); defaults to `true` so a
    /// freshly-created window is not accidentally culled.
    pub visible: bool,
    /// Level-of-detail bucket (design §13.2 / §23.7 processor LOD).
    pub lod: u8,
    /// Batch key (design §6 `SharedComponent` / §15): windows with equal keys may
    /// merge into one GPU draw.
    pub batch_key: u64,
    /// Owning world-partition cell id (design §13.1); `0` means "no cell".
    pub cell: u64,
}

impl Default for ChunkMeta {
    #[inline]
    fn default() -> Self {
        Self {
            structural_epoch: 0,
            bounds_epoch: 0,
            bounds_version: Tick::default(),
            bounds_revision: 0,
            bounds_source: None,
            bounds: None,
            visible: true,
            lod: 0,
            batch_key: 0,
            cell: 0,
        }
    }
}

impl ChunkMeta {
    /// The monotonic structural epoch — see the type-level contract.
    #[inline]
    pub fn structural_epoch(&self) -> u64 {
        self.structural_epoch
    }

    /// Advance the structural epoch. Called by the owning [`Table`] whenever the
    /// window's row set changes; consumers only ever read it.
    ///
    /// [`Table`]: crate::storage::Table
    #[inline]
    pub(crate) fn bump_structural(&mut self) {
        self.structural_epoch = self.structural_epoch.wrapping_add(1);
    }

    /// The `structural_epoch` at which [`bounds`](Self::bounds) was computed.
    #[inline]
    pub fn bounds_epoch(&self) -> u64 {
        self.bounds_epoch
    }

    /// The chunk change-version at which [`bounds`](Self::bounds) was computed.
    #[inline]
    pub fn bounds_version(&self) -> Tick {
        self.bounds_version
    }

    /// Write revision at which the cached bounds were computed.
    pub fn bounds_revision(&self) -> u64 {
        self.bounds_revision
    }

    /// Source column of bounds produced by `Table::recompute_chunk_bounds`.
    /// Manually supplied bounds have no column provenance.
    pub fn bounds_source(&self) -> Option<ComponentId> {
        self.bounds_source
    }

    /// Store freshly computed bounds together with the structural epoch and
    /// change-version they were computed from, so a later pass can skip this
    /// window while nothing has moved.
    #[inline]
    pub fn set_bounds(&mut self, bounds: Option<ChunkBounds>, epoch: u64, version: Tick) {
        self.bounds = bounds;
        self.bounds_epoch = epoch;
        self.bounds_version = version;
        self.bounds_revision = 0;
        self.bounds_source = None;
    }

    pub(crate) fn set_column_bounds(
        &mut self,
        bounds: Option<ChunkBounds>,
        epoch: u64,
        version: Tick,
        revision: u64,
        source: ComponentId,
    ) {
        self.set_bounds(bounds, epoch, version);
        self.bounds_revision = revision;
        self.bounds_source = Some(source);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_per_chunk_scales_with_byte_size() {
        // 16 KiB / 16 B = 1024 rows.
        assert_eq!(rows_per_chunk(16), TARGET_CHUNK_BYTES / 16);
        // Huge rows clamp to at least one row per chunk.
        assert_eq!(rows_per_chunk(TARGET_CHUNK_BYTES * 2), 1);
        // Zero-sized rows fall back to the hard cap.
        assert_eq!(rows_per_chunk(0), MAX_ROWS_PER_CHUNK);
    }

    /// A backing store for version cells (standing in for the chunk block
    /// header) plus a `ChunkVersions` whose cells point into it. The backing
    /// cells are boxed so growing the fixture never invalidates attached cells.
    struct Fixture {
        cv: ChunkVersions,
        #[allow(clippy::vec_box)]
        backing: Vec<Box<ChunkChange>>,
    }

    impl Fixture {
        fn new(rows_per_chunk: usize) -> Self {
            Self {
                cv: ChunkVersions::new(rows_per_chunk),
                backing: Vec::new(),
            }
        }

        /// Attach header cells so the version set covers `len` rows.
        fn cover(&mut self, len: usize) {
            let needed = len.div_ceil(self.cv.rows_per_chunk());
            while self.backing.len() < needed {
                self.backing.push(Box::new(ChunkChange::new()));
            }
            while self.cv.chunk_count() < needed {
                let i = self.cv.chunk_count();
                let ptr = &mut *self.backing[i] as *mut ChunkChange;
                // SAFETY: the boxed cell stays live until after cv is dropped,
                // and each cell is attached to exactly one tracker.
                unsafe { self.cv.attach_cell(ptr) };
            }
        }
    }

    #[test]
    fn push_grows_chunks_and_bounds_versions() {
        let rpc = 4;
        let mut f = Fixture::new(rpc);
        assert_eq!(f.cv.chunk_count(), 0);

        // Fill the first chunk; each push bumps chunk 0 to a newer tick.
        for row in 0..rpc {
            f.cover(row + 1);
            f.cv.on_push(row + 1, Tick::new((row + 1) as u32));
        }
        assert_eq!(f.cv.chunk_count(), 1);
        assert_eq!(f.cv.version(0), Tick::new(rpc as u32));

        // One more row spills into a second chunk.
        f.cover(rpc + 1);
        f.cv.on_push(rpc + 1, Tick::new(99));
        assert_eq!(f.cv.chunk_count(), 2);
        assert_eq!(f.cv.chunk_of(rpc), 1);
        assert_eq!(f.cv.version(1), Tick::new(99));
        // The first chunk's bound is unchanged.
        assert_eq!(f.cv.version(0), Tick::new(rpc as u32));
    }

    #[test]
    fn bump_never_lowers_the_bound() {
        let mut f = Fixture::new(4);
        f.cover(1);
        f.cv.on_push(1, Tick::new(10));
        // A stale (older) tick must not lower the recorded upper bound.
        f.cover(2);
        f.cv.on_push(2, Tick::new(3));
        assert_eq!(f.cv.version(0), Tick::new(10));
    }

    #[test]
    fn swap_remove_folds_moved_tick_and_shrinks() {
        let rpc = 4;
        let mut f = Fixture::new(rpc);
        // Two full chunks: rows 0..8.
        for row in 0..(2 * rpc) {
            f.cover(row + 1);
            f.cv.on_push(row + 1, Tick::new(1));
        }
        assert_eq!(f.cv.chunk_count(), 2);

        // Remove row 0; the former last row (tick 42) swaps into chunk 0.
        f.cv.on_swap_remove(0, 2 * rpc - 1, Tick::new(42));
        assert_eq!(f.cv.version(0), Tick::new(42));
        // 7 rows still need 2 chunks.
        assert_eq!(f.cv.chunk_count(), 2);

        // Drain the second chunk entirely (rows 7,6,5,4 removed as last rows).
        for len in (rpc..(2 * rpc - 1)).rev() {
            f.cv.on_swap_remove(len, len, Tick::new(0));
            // The owning table detaches the trailing cell when it frees a block.
            if f.cv.chunk_count() > len.div_ceil(rpc) {
                f.cv.detach_cell();
            }
        }
        assert_eq!(f.cv.chunk_count(), 1);
    }

    #[test]
    fn check_ticks_clamps_versions() {
        let mut f = Fixture::new(4);
        f.cover(1);
        f.cv.on_push(1, Tick::new(1));
        let now = Tick::new(Tick::MAX_CHANGE_AGE + 100);
        f.cv.check_ticks(now);
        assert_eq!(f.cv.version(0).age_since(now), Tick::MAX_CHANGE_AGE);
    }

    fn indexed(f: &Fixture, last: Tick, now: Tick) -> (Vec<usize>, ChunkVisitStats) {
        let mut chunks = Vec::new();
        let stats = f.cv.visit_changed(last, now, |chunk| chunks.push(chunk));
        (chunks, stats)
    }

    fn assert_index_matches_leaves(f: &Fixture, last: Tick, now: Tick) {
        let expected: Vec<_> = (0..f.cv.chunk_count())
            .filter(|&chunk| f.cv.may_have_changed(chunk, last, now))
            .collect();
        let (actual, stats) = indexed(f, last, now);
        assert_eq!(actual, expected, "window {last:?}..{now:?}");
        assert_eq!(stats.chunks_returned, actual.len());
    }

    #[test]
    fn indexed_clean_and_sparse_work_does_not_scan_all_chunks() {
        let mut f = Fixture::new(1);
        f.cover(4097);
        for chunk in 0..f.cv.chunk_count() {
            f.cv.bump_row(chunk, Tick::new(1));
        }
        let (chunks, clean) = indexed(&f, Tick::new(1), Tick::new(2));
        assert!(chunks.is_empty());
        assert_eq!(clean.summaries_tested, 1);

        f.backing[2048].record(Tick::new(2));
        let (chunks, sparse) = indexed(&f, Tick::new(1), Tick::new(2));
        assert_eq!(chunks, [2048]);
        assert!(sparse.summaries_tested <= 27, "{sparse:?}");
        // Traversal is non-consuming: independent observers see the same set.
        assert_eq!(indexed(&f, Tick::new(1), Tick::new(2)), (chunks, sparse));
        assert_eq!(
            indexed(&f, Tick::new(2), Tick::new(3)).1.summaries_tested,
            1
        );
    }

    #[test]
    fn journal_skips_old_change_batches() {
        let mut f = Fixture::new(1);
        f.cover(4097);
        for chunk in 0..f.cv.chunk_count() {
            f.cv.bump_row(chunk, Tick::new(10));
        }

        // A later sparse write must not inspect the old batch's 4097 chunk
        // members. The batch header is checked, but its members are skipped.
        f.cv.bump_row(2048, Tick::new(11));
        let (chunks, stats) = indexed(&f, Tick::new(10), Tick::new(11));
        assert_eq!(chunks, [2048]);
        assert_eq!(stats.event_batches_tested, 1);
        assert_eq!(stats.summaries_tested, 1, "{stats:?}");
    }

    #[test]
    fn detached_event_cannot_dirty_a_reused_chunk_slot() {
        let mut f = Fixture::new(1);
        f.cover(2);
        f.cv.bump_row(1, Tick::new(7));
        f.cv.detach_cell();

        // A real table allocates a fresh initialized header when a trailing
        // block is recreated. Model that reset while reusing the same backing
        // address, then attach it again and ensure the old journal event is
        // rejected by the new attachment generation.
        f.backing[1]
            .range
            .store(ChunkChange::EMPTY, Ordering::Relaxed);
        // SAFETY: the boxed cell is live, aligned, and remains owned by the
        // fixture until it is detached.
        unsafe { f.cv.attach_cell(&mut *f.backing[1]) };
        let (chunks, stats) = indexed(&f, Tick::new(6), Tick::new(7));
        assert!(chunks.is_empty(), "stale detached event leaked: {chunks:?}");
        assert_eq!(stats.chunks_returned, 0);
    }

    #[test]
    fn journal_preserves_tick_wraparound() {
        let mut f = Fixture::new(1);
        f.cover(2);
        f.cv.bump_row(0, Tick::MAX);
        f.cv.bump_row(1, Tick::ZERO);

        let (wrapped, _) = indexed(&f, Tick::MAX, Tick::ZERO);
        assert_eq!(wrapped, [1]);
        let (pre_wrap, _) = indexed(&f, Tick::new(u32::MAX - 1), Tick::MAX);
        assert_eq!(pre_wrap, [0]);
    }

    #[test]
    fn journal_cleanup_discards_expired_batches() {
        let mut f = Fixture::new(1);
        f.cover(1);
        f.cv.bump_row(0, Tick::new(1));
        assert_eq!(f.cv.events.lock().len(), 1);

        let now = Tick::new(Tick::MAX_CHANGE_AGE + 2);
        f.cv.check_ticks(now);
        assert!(f.cv.events.lock().is_empty());
    }

    #[test]
    fn journal_indexes_out_of_order_ticks_without_scanning_unrelated_batches() {
        let mut f = Fixture::new(1);
        f.cover(3);
        f.cv.bump_row(0, Tick::new(30));
        f.cv.bump_row(1, Tick::new(10));
        f.cv.bump_row(2, Tick::new(20));

        let (chunks, stats) = indexed(&f, Tick::new(19), Tick::new(20));
        assert_eq!(chunks, [2]);
        assert_eq!(stats.event_batches_tested, 1);
    }

    #[test]
    fn journal_restores_chunk_order_after_out_of_order_tick_batches() {
        let mut f = Fixture::new(1);
        f.cover(3);
        f.cv.bump_row(2, Tick::new(10));
        f.cv.bump_row(1, Tick::new(20));
        f.cv.bump_row(0, Tick::new(30));

        let (chunks, stats) = indexed(&f, Tick::ZERO, Tick::new(30));
        assert_eq!(chunks, [0, 1, 2]);
        assert_eq!(stats.event_batches_tested, 3);
    }

    #[test]
    fn journal_deduplicates_multi_tick_writes_linearly() {
        let mut f = Fixture::new(1);
        f.cover(2);
        for tick in 1..=64 {
            f.cv.bump_row(0, Tick::new(tick));
            if tick % 2 == 0 {
                f.cv.bump_row(1, Tick::new(tick));
            }
        }

        let (chunks, stats) = indexed(&f, Tick::ZERO, Tick::new(64));
        assert_eq!(chunks, [0, 1]);
        assert_eq!(stats.chunks_returned, 2);
        assert_eq!(stats.event_batches_tested, 64);
    }

    #[test]
    fn indexed_growth_detach_reuse_clamp_and_historical_windows() {
        let mut f = Fixture::new(1);
        let mut random = 0x9e37_79b9_u32;
        let boundaries = [
            0,
            1,
            2,
            Tick::MAX_CHANGE_AGE,
            1 << 31,
            u32::MAX - 1,
            u32::MAX,
        ];
        for len in 1..=133 {
            f.cover(len);
            random ^= random << 13;
            random ^= random >> 17;
            random ^= random << 5;
            f.cv.bump_row(len - 1, Tick::new(random));
            for &last in &boundaries {
                for &now in &boundaries {
                    assert_index_matches_leaves(&f, Tick::new(last), Tick::new(now));
                }
            }
        }
        for count in (0..133).rev() {
            f.cv.detach_cell();
            assert!(f.backing[count].parent.load(Ordering::Relaxed).is_null());
            f.backing[count].record(Tick::new(700));
            assert_index_matches_leaves(&f, Tick::new(699), Tick::new(700));
        }
        assert_eq!(
            indexed(&f, Tick::ZERO, Tick::new(701)).1,
            ChunkVisitStats::default()
        );
        f.cover(79);
        f.cv.check_ticks(Tick::new(1000));
        for chunk in 0..79 {
            f.cv.rebuild(chunk, [Tick::new(1000)].into_iter(), Tick::new(1000));
        }
        assert_eq!(
            indexed(&f, Tick::new(1000), Tick::new(1001))
                .1
                .summaries_tested,
            1
        );
        f.backing[78].record(Tick::new(1001));
        assert_eq!(indexed(&f, Tick::new(1000), Tick::new(1001)).0, [78]);
        for &last in &boundaries {
            for &now in &boundaries {
                assert_index_matches_leaves(&f, Tick::new(last), Tick::new(now));
            }
        }
    }

    #[test]
    fn tracker_drop_unlinks_live_headers() {
        let mut cell = Box::new(ChunkChange::new());
        let mut other = Box::new(ChunkChange::new());
        {
            let mut cv = ChunkVersions::new(1);
            // SAFETY: distinct live boxed cells outlive the tracker.
            unsafe {
                cv.attach_cell(&mut *cell);
                cv.attach_cell(&mut *other);
            }
            assert!(!cell.parent.load(Ordering::Relaxed).is_null());
        }
        assert!(cell.parent.load(Ordering::Relaxed).is_null());
        assert!(other.parent.load(Ordering::Relaxed).is_null());
        cell.record(Tick::new(12));
        other.record(Tick::new(13));
    }

    #[cfg(feature = "std")]
    #[test]
    fn parallel_leaf_writes_propagate_to_shared_ancestors() {
        let mut f = Fixture::new(1);
        f.cover(257);
        std::thread::scope(|scope| {
            for worker in 0..8 {
                let backing = &f.backing;
                scope.spawn(move || {
                    for chunk in (worker..backing.len()).step_by(8) {
                        for tick in 1..=16 {
                            backing[chunk].record(Tick::new(tick));
                        }
                    }
                });
            }
        });
        for tick in 1..=16 {
            let (chunks, _) = indexed(&f, Tick::new(tick - 1), Tick::new(tick));
            assert_eq!(chunks, (0..257).collect::<Vec<_>>());
        }
        assert_eq!(
            indexed(&f, Tick::new(16), Tick::new(17)).1.summaries_tested,
            1
        );
    }

    #[test]
    fn wrap_and_historical_windows_never_miss() {
        let cell = ChunkChange::new();
        assert!(!cell.may_have_changed(Tick::MAX, Tick::new(1)));
        cell.record(Tick::MAX);
        cell.record(Tick::new(1));
        assert_eq!(cell.version(), Tick::new(1));
        assert!(cell.may_have_changed(Tick::MAX, Tick::new(1)));
        assert!(cell.may_have_changed(Tick::new(u32::MAX - 1), Tick::MAX));
        assert!(!cell.may_have_changed(Tick::new(1), Tick::new(2)));
        assert!(!cell.may_have_changed(Tick::new(1), Tick::new(1)));
        let revision = cell.revision();
        cell.record(Tick::MAX); // explicit older write, even if already covered
        assert_eq!(cell.revision(), revision + 1);
        assert!(cell.may_have_changed(Tick::new(u32::MAX - 1), Tick::MAX));
    }

    #[test]
    fn arbitrary_tick_summaries_and_clamping_match_row_oracle() {
        let boundaries = [
            0,
            1,
            2,
            Tick::CHECK_TICK_THRESHOLD,
            Tick::MAX_CHANGE_AGE,
            Tick::MAX_CHANGE_AGE + 1,
            1 << 31,
            u32::MAX - 2,
            u32::MAX - 1,
            u32::MAX,
        ];
        let mut random = 0x89ab_cdef_u32;
        for case in 0..64 {
            let cell = ChunkChange::new();
            let mut rows = Vec::new();
            for i in 0..20 {
                random ^= random << 13;
                random ^= random >> 17;
                random ^= random << 5;
                let tick = Tick::new(if i < boundaries.len() {
                    boundaries[(i + case) % boundaries.len()]
                } else {
                    random
                });
                rows.push(tick);
                cell.record(tick);
                for &now in &boundaries {
                    for &last in &boundaries {
                        let now = Tick::new(now);
                        let last = Tick::new(last);
                        assert!(
                            !rows.iter().any(|t| t.is_newer_than(last, now))
                                || cell.may_have_changed(last, now),
                            "miss before clamp: case={case} row={i} last={last:?} now={now:?}"
                        );
                    }
                }
            }
            for &now in &boundaries {
                let mut clamped = ChunkChange::new();
                for &row in &rows {
                    clamped.record(row);
                }
                let now = Tick::new(now);
                clamped.check_tick(now);
                let expected: Vec<_> = rows
                    .iter()
                    .map(|&t| {
                        let mut t = t;
                        t.check_tick(now);
                        t
                    })
                    .collect();
                for &observer in &boundaries {
                    for &last in &boundaries {
                        let last = Tick::new(last);
                        let observer = Tick::new(observer);
                        assert!(!expected.iter().any(|t| t.is_newer_than(last, observer))
                            || clamped.may_have_changed(last, observer),
                            "miss after clamp: case={case} last={last:?} now={now:?} observer={observer:?}");
                    }
                }
            }
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn concurrent_rows_merge_one_atomic_chunk_summary() {
        let cell = ChunkChange::new();
        std::thread::scope(|scope| {
            for worker in 0..8 {
                let cell = &cell;
                scope.spawn(move || {
                    for row in 0..1024 {
                        cell.record(Tick::new(1 + worker * 1024 + row));
                    }
                });
            }
        });
        assert_eq!(cell.revision(), 8 * 1024);
        for tick in 1..=8 * 1024 {
            assert!(cell.may_have_changed(Tick::new(tick - 1), Tick::new(tick)));
        }
        assert!(!cell.may_have_changed(Tick::new(8 * 1024), Tick::new(8 * 1024 + 1)));
    }

    #[test]
    fn chunk_meta_defaults_are_visible_and_uncomputed() {
        let m = ChunkMeta::default();
        assert_eq!(m.structural_epoch(), 0);
        assert!(m.bounds.is_none());
        assert!(m.visible, "a fresh window must not be culled by default");
        assert_eq!(m.lod, 0);
        assert_eq!(m.batch_key, 0);
        assert_eq!(m.cell, 0);
    }

    #[test]
    fn chunk_meta_structural_bump_is_monotonic() {
        let mut m = ChunkMeta::default();
        for expected in 1..=5 {
            m.bump_structural();
            assert_eq!(m.structural_epoch(), expected);
        }
        // Available fields can be written by a consumer without touching the epoch.
        m.lod = 3;
        m.batch_key = 42;
        assert_eq!(m.structural_epoch(), 5);
    }

    #[test]
    fn chunk_bounds_well_formed_contains_and_union() {
        let a = ChunkBounds::new([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]);
        assert!(a.is_well_formed());
        assert!(a.contains([1.0, 1.0, 1.0]));
        assert!(a.contains([0.0, 2.0, 1.5]));
        assert!(!a.contains([2.5, 1.0, 1.0]));

        let inverted = ChunkBounds::new([1.0, 0.0, 0.0], [0.0, 1.0, 1.0]);
        assert!(!inverted.is_well_formed());

        let mut u = a;
        u.union(&ChunkBounds::new([-1.0, 1.0, 0.5], [1.5, 3.0, 1.0]));
        assert_eq!(u, ChunkBounds::new([-1.0, 0.0, 0.0], [2.0, 3.0, 2.0]));
    }
}
