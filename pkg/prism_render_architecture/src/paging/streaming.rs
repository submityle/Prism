//! Per-frame page-streaming reconciliation: the controller that ties the
//! residency table, the physical slot pool and the page-data store into one
//! coherent streaming loop.
//!
//! The three lower layers each own one concern and nothing else:
//!
//! * [`ResidencyTable`] tracks *what* each page's lifecycle state, priority and
//!   recency are.
//! * [`PagePool`] decides *which physical slot* a resident page occupies.
//! * [`PageStorage`] holds *the bytes* those slots contain.
//!
//! On their own they do not stream anything; a controller must, each frame,
//! turn the frame's coalesced page requests into slot allocations, data
//! uploads, residency transitions and - when the budget is exceeded -
//! prioritized evictions. [`PageStreamManager`] is that controller. It keeps the
//! invariant that the pool holds exactly the resident pages ([`PagePool::len`]
//! equals [`ResidencyTable::resident_count`]) and that every resident page's
//! slot holds its uploaded bytes, so the exported [`PageStreamManager::entries`]
//! table and [`PageStreamManager::storage`] buffer are exactly the inputs the
//! `GPU`-side page-table resolve and page-data gather twins consume.
//!
//! # Streaming policy
//!
//! Reconciliation mirrors a Nanite-style prioritized streamer:
//!
//! 1. The frame's [`RequestBatch`] is flushed into the residency table, marking
//!    unseen pages [`PageResidency::Requested`](super::PageResidency::Requested) and raising re-seen pages'
//!    priority and recency.
//! 2. Pages whose recorded priority the budget can no longer justify are evicted
//!    down to the resident budget, never touching a page used this frame.
//! 3. Pending pages are streamed in highest priority first while free slots
//!    remain. When the pool is full, a pending page displaces the lowest
//!    priority resident page that was not used this frame *and* is strictly
//!    lower priority than the incoming page; if none qualifies the remaining
//!    (lower priority) pending pages are deferred to a later frame.
//!
//! The controller is GPU-independent and deterministic: every container it
//! consults iterates in key order and every tie is broken by key, so a given
//! request stream and page source reproduce the same slot table and eviction
//! sequence run to run.
//!
//! Provenance: standard priority-ranked residency streaming; no Unreal Engine
//! source or derived code.

use alloc::vec::Vec;

use super::page_pool::{PagePool, PagePoolError};
use super::page_storage::{PageStorage, PageStorageError};
use super::{RequestBatch, ResidencyTable};

/// Supplies the page-data words backing a page that is about to become
/// resident.
///
/// The streaming controller calls [`load`](PageSource::load) exactly once per
/// stream-in, when it has decided a page will occupy a physical slot. An
/// implementation typically reads the page from a mapped asset or decompresses
/// it; the returned slice must hold exactly [`PageStorage::page_words`] words or
/// the stream-in is rejected as a size mismatch.
pub trait PageSource<K> {
    /// Returns the `page_words` words backing `key`.
    fn load(&mut self, key: K) -> Vec<u32>;
}

/// Why a page could not be admitted this frame.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StreamError {
    /// The pool ran out of physical slots and no resident page could be
    /// displaced. The page stays [`PageResidency::Requested`](super::PageResidency::Requested) for a later frame.
    NoSlotAvailable,
    /// The page source returned a payload whose word count does not match the
    /// pool page size, so it could not be uploaded.
    BadPageSize {
        /// Words the source returned.
        got: usize,
        /// Words one slot holds.
        expected: u32,
    },
}

impl core::fmt::Display for StreamError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            StreamError::NoSlotAvailable => {
                write!(f, "no physical slot available and nothing evictable")
            }
            StreamError::BadPageSize { got, expected } => {
                write!(f, "page source returned {got} words but the pool page size is {expected}")
            }
        }
    }
}

impl core::error::Error for StreamError {}

/// What one [`reconcile`](PageStreamManager::reconcile) pass did, for telemetry
/// and tests.
///
/// The three lists are disjoint and each is key-ordered where it is built from a
/// key-ordered scan, so a reconcile is fully described by its report.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReconcileReport<K> {
    /// Pages newly uploaded and marked resident this frame.
    pub streamed_in: Vec<K>,
    /// Resident pages dropped this frame to respect the budget or make room.
    pub evicted: Vec<K>,
    /// Requested pages left pending because no slot could be freed for them.
    pub deferred: Vec<K>,
}

impl<K> Default for ReconcileReport<K> {
    /// An empty report: no stream-ins, evictions or deferrals.
    ///
    /// Hand-written rather than derived so the empty report exists for every
    /// key type `K`, without the spurious `K: Default` bound a derive adds.
    fn default() -> Self {
        Self { streamed_in: Vec::new(), evicted: Vec::new(), deferred: Vec::new() }
    }
}

impl<K> ReconcileReport<K> {
    /// Whether nothing changed this frame.
    #[must_use]
    pub fn is_noop(&self) -> bool {
        self.streamed_in.is_empty() && self.evicted.is_empty() && self.deferred.is_empty()
    }
}

/// A bounded page-streaming controller composing the residency table, slot pool
/// and page-data store.
///
/// Construct it with the physical slot capacity, the fixed page size and a
/// resident budget (which must not exceed capacity), then each frame flush the
/// frame's requests through [`reconcile`](PageStreamManager::reconcile).
#[derive(Clone, Debug)]
pub struct PageStreamManager<K: Copy + Ord> {
    residency: ResidencyTable<K>,
    pool: PagePool<K>,
    storage: PageStorage,
    resident_budget: usize,
}

impl<K: Copy + Ord> PageStreamManager<K> {
    /// Creates a controller over `capacity` physical slots of `page_words` words
    /// each, keeping at most `resident_budget` pages resident.
    ///
    /// # Panics
    ///
    /// Panics when `resident_budget` exceeds `capacity`, since the budget can
    /// never justify more resident pages than the pool can physically hold.
    #[must_use]
    pub fn new(capacity: u32, page_words: u32, resident_budget: usize) -> Self {
        assert!(
            resident_budget <= capacity as usize,
            "resident budget must not exceed physical slot capacity"
        );
        Self {
            residency: ResidencyTable::new(),
            pool: PagePool::new(capacity),
            storage: PageStorage::new(capacity, page_words),
            resident_budget,
        }
    }

    /// Number of physical slots the pool holds.
    #[must_use]
    pub const fn capacity(&self) -> u32 {
        self.pool_capacity()
    }

    #[inline]
    #[must_use]
    const fn pool_capacity(&self) -> u32 {
        // `PagePool::capacity` is not `const`, so cache-free access goes through
        // the stored budget bound instead where a const context is needed.
        self.storage.capacity()
    }

    /// Words per physical slot.
    #[must_use]
    pub const fn page_words(&self) -> u32 {
        self.storage.page_words()
    }

    /// The most pages the controller will keep resident at once.
    #[must_use]
    pub const fn resident_budget(&self) -> usize {
        self.resident_budget
    }

    /// Number of pages currently resident (equal to the number of used slots).
    #[must_use]
    pub fn resident_count(&self) -> usize {
        self.residency.resident_count()
    }

    /// Read-only view of the residency table.
    #[must_use]
    pub const fn residency(&self) -> &ResidencyTable<K> {
        &self.residency
    }

    /// Read-only view of the page-data store.
    #[must_use]
    pub const fn storage(&self) -> &PageStorage {
        &self.storage
    }

    /// Resolves a resident page to its physical slot, or [`None`] if it is not
    /// resident.
    #[must_use]
    pub fn slot_of(&self, key: K) -> Option<u32> {
        self.pool.slot_of(key)
    }

    /// Exports the key-ordered `(key, slot)` residency map.
    ///
    /// This is exactly the sorted table the `GPU` page-table resolve twin
    /// binary-searches, so a frame's reconcile output can be handed straight to
    /// it.
    #[must_use]
    pub fn entries(&self) -> Vec<(K, u32)> {
        self.pool.entries()
    }

    /// Reconciles one frame's page requests against the resident set.
    ///
    /// Returns a [`ReconcileReport`] describing the stream-ins, evictions and
    /// deferrals. See the [module docs](self) for the streaming policy.
    pub fn reconcile<S: PageSource<K>>(
        &mut self,
        batch: &RequestBatch<K>,
        frame: u64,
        source: &mut S,
    ) -> ReconcileReport<K> {
        let mut report = ReconcileReport::default();

        // 1. Fold the frame's coalesced requests into the residency table.
        batch.flush(&mut self.residency, frame);

        // 2. Evict down to the resident budget, protecting pages used this
        //    frame. This covers a lowered budget or leftover over-residency.
        for victim in self.residency.select_evictions(self.resident_budget, frame) {
            self.drop_resident(victim);
            report.evicted.push(victim);
        }

        // 3. Rank pending stream-in candidates: highest priority first, ties by
        //    key so the sequence is deterministic.
        let mut pending = self.residency.pending_requests();
        pending.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        for (key, priority) in pending {
            if self.resident_count() >= self.resident_budget {
                // At budget: try to displace a strictly-lower-priority resident
                // page that was not used this frame.
                match self.pick_replacement_victim(priority, frame) {
                    Some(victim) => {
                        self.drop_resident(victim);
                        report.evicted.push(victim);
                    }
                    None => {
                        // No room and nothing displaceable; every remaining
                        // pending page is lower priority, so defer them all.
                        report.deferred.push(key);
                        continue;
                    }
                }
            }
            match self.stream_in(key, source) {
                Ok(()) => report.streamed_in.push(key),
                Err(StreamError::NoSlotAvailable) => report.deferred.push(key),
                Err(StreamError::BadPageSize { .. }) => {
                    // A malformed source payload cannot be admitted; leave the
                    // page requested so a later frame with fixed data can retry.
                    report.deferred.push(key);
                }
            }
        }

        report
    }

    /// Streams one page in: loads its data, allocates a slot, uploads and marks
    /// it resident. Validates the payload size before touching the pool so a bad
    /// payload leaves no half-allocated slot.
    fn stream_in<S: PageSource<K>>(&mut self, key: K, source: &mut S) -> Result<(), StreamError> {
        let page = source.load(key);
        let expected = self.storage.page_words() as usize;
        if page.len() != expected {
            return Err(StreamError::BadPageSize {
                got: page.len(),
                expected: self.storage.page_words(),
            });
        }
        let slot = match self.pool.allocate(key) {
            Ok(slot) => slot,
            Err(PagePoolError::PoolFull { .. }) => return Err(StreamError::NoSlotAvailable),
        };
        match self.storage.upload(slot, &page) {
            Ok(()) => {}
            Err(PageStorageError::SlotOutOfRange { .. } | PageStorageError::PageSizeMismatch { .. }) => {
                // The slot came from the pool and the size was validated above,
                // so this is unreachable; roll the slot back defensively rather
                // than leaving an allocated-but-empty slot.
                self.pool.free(key);
                return Err(StreamError::BadPageSize {
                    got: page.len(),
                    expected: self.storage.page_words(),
                });
            }
        }
        self.residency.mark_resident(key);
        Ok(())
    }

    /// Drops a resident page from all three layers, keeping them in lockstep.
    fn drop_resident(&mut self, key: K) {
        if let Some(slot) = self.pool.free(key) {
            self.storage.clear_slot(slot);
        }
        self.residency.evict(key);
    }

    /// Picks the lowest-priority resident page eligible to be displaced by an
    /// incoming page of `incoming_priority`: not used this frame and strictly
    /// lower priority. Ties break toward the older page. [`None`] when nothing
    /// qualifies.
    fn pick_replacement_victim(&self, incoming_priority: f32, frame: u64) -> Option<K> {
        self.residency
            .resident_entries()
            .into_iter()
            .filter(|(_, e)| e.last_used_frame < frame && e.priority < incoming_priority)
            .min_by(|a, b| {
                a.1.priority
                    .total_cmp(&b.1.priority)
                    .then(a.1.last_used_frame.cmp(&b.1.last_used_frame))
                    .then(a.0.cmp(&b.0))
            })
            .map(|(k, _)| k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::collections::BTreeMap;
    use alloc::vec;

    /// A page source that hands out a slot-fill pattern derived from the key so
    /// uploaded bytes are distinct and checkable, with an overridable page size
    /// to exercise the bad-payload path.
    struct MapSource {
        page_words: u32,
        overrides: BTreeMap<u32, Vec<u32>>,
    }

    impl MapSource {
        fn new(page_words: u32) -> Self {
            Self {
                page_words,
                overrides: BTreeMap::new(),
            }
        }
    }

    impl PageSource<u32> for MapSource {
        fn load(&mut self, key: u32) -> Vec<u32> {
            if let Some(v) = self.overrides.get(&key) {
                return v.clone();
            }
            (0..self.page_words).map(|w| key * 100 + w).collect()
        }
    }

    fn batch(refs: &[(u32, f32)]) -> RequestBatch<u32> {
        let mut b = RequestBatch::new();
        for (k, p) in refs {
            b.record(*k, *p);
        }
        b
    }

    #[test]
    fn streams_in_requested_pages_and_uploads_their_data() {
        let mut mgr = PageStreamManager::<u32>::new(4, 3, 4);
        let mut src = MapSource::new(3);
        let report = mgr.reconcile(&batch(&[(1, 1.0), (2, 5.0)]), 1, &mut src);
        assert_eq!(report.streamed_in.len(), 2);
        assert!(report.evicted.is_empty() && report.deferred.is_empty());
        assert_eq!(mgr.resident_count(), 2);
        // Data landed in each page's slot.
        let slot1 = mgr.slot_of(1).unwrap();
        assert_eq!(mgr.storage().slot_words(slot1), Some(&[100, 101, 102][..]));
        let slot2 = mgr.slot_of(2).unwrap();
        assert_eq!(mgr.storage().slot_words(slot2), Some(&[200, 201, 202][..]));
        // Pool and residency stay in lockstep.
        assert_eq!(mgr.entries().len(), 2);
    }

    #[test]
    fn defers_lowest_priority_when_over_budget() {
        let mut mgr = PageStreamManager::<u32>::new(2, 2, 2);
        let mut src = MapSource::new(2);
        let report = mgr.reconcile(&batch(&[(1, 1.0), (2, 9.0), (3, 5.0)]), 1, &mut src);
        // Budget 2: the two highest priorities (9, 5) stream in, lowest defers.
        assert_eq!(report.streamed_in, vec![2, 3]);
        assert_eq!(report.deferred, vec![1]);
        assert_eq!(mgr.resident_count(), 2);
        assert!(mgr.slot_of(2).is_some() && mgr.slot_of(3).is_some());
        assert!(mgr.slot_of(1).is_none());
    }

    #[test]
    fn higher_priority_page_displaces_unused_resident() {
        let mut mgr = PageStreamManager::<u32>::new(2, 2, 2);
        let mut src = MapSource::new(2);
        // Frame 1 fills both slots at low priority.
        mgr.reconcile(&batch(&[(1, 1.0), (2, 2.0)]), 1, &mut src);
        assert_eq!(mgr.resident_count(), 2);
        // Frame 2 requests only a new high-priority page; neither resident is
        // touched this frame, so the lowest-priority one (key 1) is displaced.
        let report = mgr.reconcile(&batch(&[(3, 8.0)]), 2, &mut src);
        assert_eq!(report.streamed_in, vec![3]);
        assert_eq!(report.evicted, vec![1]);
        assert!(mgr.slot_of(3).is_some());
        assert!(mgr.slot_of(1).is_none());
        assert!(mgr.slot_of(2).is_some());
    }

    #[test]
    fn page_used_this_frame_is_never_displaced() {
        let mut mgr = PageStreamManager::<u32>::new(2, 2, 2);
        let mut src = MapSource::new(2);
        mgr.reconcile(&batch(&[(1, 1.0), (2, 2.0)]), 1, &mut src);
        // Frame 2 re-requests both residents (so both are used this frame) plus
        // a higher-priority newcomer. Nothing is displaceable, so it defers.
        let report = mgr.reconcile(&batch(&[(1, 1.0), (2, 2.0), (3, 9.0)]), 2, &mut src);
        assert_eq!(report.streamed_in, Vec::<u32>::new());
        assert_eq!(report.deferred, vec![3]);
        assert_eq!(mgr.resident_count(), 2);
    }

    #[test]
    fn lowered_budget_evicts_down_next_frame() {
        let mut mgr = PageStreamManager::<u32>::new(3, 2, 3);
        let mut src = MapSource::new(2);
        mgr.reconcile(&batch(&[(1, 1.0), (2, 2.0), (3, 3.0)]), 1, &mut src);
        assert_eq!(mgr.resident_count(), 3);
        // Shrink the budget and reconcile an empty frame: the two lowest
        // priority, unused pages are evicted down to the new budget.
        mgr.resident_budget = 1;
        let report = mgr.reconcile(&RequestBatch::new(), 2, &mut src);
        assert_eq!(report.evicted, vec![1, 2]);
        assert_eq!(mgr.resident_count(), 1);
        assert!(mgr.slot_of(3).is_some());
    }

    #[test]
    fn bad_payload_defers_without_allocating() {
        let mut mgr = PageStreamManager::<u32>::new(2, 3, 2);
        let mut src = MapSource::new(3);
        src.overrides.insert(1, vec![0, 0]); // wrong length for a 3-word page
        let report = mgr.reconcile(&batch(&[(1, 5.0), (2, 1.0)]), 1, &mut src);
        assert_eq!(report.streamed_in, vec![2]);
        assert_eq!(report.deferred, vec![1]);
        assert!(mgr.slot_of(1).is_none());
        // The bad page left no slot allocated, so key 2 still fit.
        assert!(mgr.slot_of(2).is_some());
    }

    #[test]
    fn evicted_slot_data_is_cleared() {
        let mut mgr = PageStreamManager::<u32>::new(1, 2, 1);
        let mut src = MapSource::new(2);
        mgr.reconcile(&batch(&[(1, 1.0)]), 1, &mut src);
        let slot = mgr.slot_of(1).unwrap();
        assert_eq!(mgr.storage().slot_words(slot), Some(&[100, 101][..]));
        // Frame 2: a higher-priority page displaces key 1 and reuses its slot.
        mgr.reconcile(&batch(&[(2, 9.0)]), 2, &mut src);
        let reused = mgr.slot_of(2).unwrap();
        assert_eq!(reused, slot, "the freed slot is reused");
        assert_eq!(mgr.storage().slot_words(reused), Some(&[200, 201][..]));
    }
}
