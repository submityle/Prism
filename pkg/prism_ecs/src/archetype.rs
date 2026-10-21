//! Archetypes: canonical grouping by components, shared values and relation pairs.
//!
//! An [`Archetype`] owns the [`Table`] that stores every entity sharing one
//! [`ComponentSet`], shared bindings and fragmenting [`RelationPair`]s. The
//! [`Archetypes`] registry maps each distinct identity to a stable
//! [`ArchetypeId`] so entities with the same identity are iterated together.
//!
//! Structural changes (adding/removing a component) move an entity from one
//! archetype to another. For M0 the destination archetype is resolved by a
//! hash lookup on the target component set; the per-archetype add/remove edge
//! cache that turns this into an O(1) pointer-follow is an M2 refinement
//! (design §5.3, §9).

use alloc::boxed::Box;
use alloc::collections::{BTreeMap, BinaryHeap, VecDeque};
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::cell::UnsafeCell;
use core::cmp::Reverse;
use core::ops::Bound::{Excluded, Unbounded};
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::collections::HashMap;
use crate::component::{ComponentId, ComponentSet, Components};
use crate::entity::Entity;
use crate::storage::{SharedValue, SharedValueId, Table};

/// Maximum retained write/structural events. Lagging readers explicitly rebuild.
pub const COLUMN_CHANGE_LOG_CAPACITY: usize = 8192;
/// Maximum retained permanent-archetype-eviction IDs. Lagging query caches
/// conservatively filter their own cached IDs against tombstone state.
pub(crate) const ARCHETYPE_EVICTION_LOG_CAPACITY: usize = 4096;
static NEXT_COLUMN_LOG: AtomicU64 = AtomicU64::new(1);

/// Independent extraction position, unrelated to simulation/change ticks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColumnChangeCursor {
    log: u64,
    revision: u64,
}

/// A physical column write or row-membership change. `component == None`
/// affects every column; `chunk == None` announces a newly created archetype.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColumnChange {
    /// Stable archetype address within this registry.
    pub archetype: ArchetypeId,
    /// Written component, or all components for structural changes.
    pub component: Option<ComponentId>,
    /// Affected chunk, including a removed tail chunk.
    pub chunk: Option<usize>,
}

/// One non-destructive journal snapshot. Only acknowledge `cursor` after the
/// consumer succeeds. A gap (including initial subscription/registry replacement)
/// requires a full baseline; the returned events alone are then insufficient.
pub struct ColumnChanges {
    /// Position of this snapshot; other consumers have their own positions.
    pub cursor: ColumnChangeCursor,
    /// Whether retained events cannot establish a complete incremental delta.
    pub gap: bool,
    /// Events after the supplied cursor, in revision order. Empty on a gap.
    pub events: Vec<ColumnChange>,
}

/// Result and deterministic work count for the query-only filtered journal read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FilteredColumnChangeRun {
    pub change: ColumnChange,
    /// Number of selected retained events represented by this run.
    pub event_count: usize,
}

pub(crate) struct FilteredColumnChanges {
    pub cursor: ColumnChangeCursor,
    pub gap: bool,
    /// Consecutive events for the same archetype/chunk are coalesced. Structural
    /// events without a chunk remain individual records.
    pub events: Vec<FilteredColumnChangeRun>,
    /// Relevant event records selected/materialized for merge, including
    /// structural records. Binary-search probes over revision metadata and
    /// unrelated writes are not counted.
    pub records_scanned: usize,
}

#[inline]
fn push_column_change_run(runs: &mut Vec<FilteredColumnChangeRun>, change: ColumnChange) {
    if change.chunk.is_some()
        && let Some(previous) = runs.last_mut()
        && previous.change.archetype == change.archetype
        && previous.change.chunk == change.chunk
    {
        previous.event_count += 1;
    } else {
        runs.push(FilteredColumnChangeRun {
            change,
            event_count: 1,
        });
    }
}

struct ColumnJournalState {
    revision: u64,
    events: VecDeque<ColumnChange>,
    component_events: HashMap<ComponentId, VecDeque<u64>>,
    structural_events: VecDeque<u64>,
    evicted_component_revision: HashMap<ComponentId, u64>,
    evicted_structural_revision: u64,
    precision_floor: u64,
}

/// Shared by the registry, tables and stable boxed column write journals.
/// No raw ownership links are added to physical chunk headers.
pub(crate) struct ColumnJournal {
    id: u64,
    active: AtomicBool,
    locked: AtomicBool,
    state: UnsafeCell<ColumnJournalState>,
}

// SAFETY: all accesses to state are serialized by the acquire/release lock.
unsafe impl Send for ColumnJournal {}
// SAFETY: the state contains only owned events and is never accessed unlocked.
unsafe impl Sync for ColumnJournal {}

struct ColumnJournalGuard<'a>(&'a ColumnJournal);
impl core::ops::Deref for ColumnJournalGuard<'_> {
    type Target = ColumnJournalState;
    fn deref(&self) -> &Self::Target {
        // SAFETY: this guard holds the lock until drop.
        unsafe { &*self.0.state.get() }
    }
}
impl core::ops::DerefMut for ColumnJournalGuard<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: the lock provides exclusive access to state.
        unsafe { &mut *self.0.state.get() }
    }
}
impl Drop for ColumnJournalGuard<'_> {
    fn drop(&mut self) {
        self.0.locked.store(false, Ordering::Release);
    }
}

impl ColumnJournal {
    fn new() -> Self {
        Self {
            id: NEXT_COLUMN_LOG.fetch_add(1, Ordering::Relaxed),
            active: AtomicBool::new(false),
            locked: AtomicBool::new(false),
            state: UnsafeCell::new(ColumnJournalState {
                revision: 0,
                events: VecDeque::new(),
                component_events: HashMap::new(),
                structural_events: VecDeque::new(),
                evicted_component_revision: HashMap::new(),
                evicted_structural_revision: 0,
                precision_floor: 0,
            }),
        }
    }

    fn lock(&self) -> ColumnJournalGuard<'_> {
        while self
            .locked
            .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            core::hint::spin_loop();
        }
        ColumnJournalGuard(self)
    }

    pub(crate) fn record(&self, event: ColumnChange) {
        // Unsubscribed worlds pay no journal allocation or additional lock.
        if !self.active.load(Ordering::Acquire) {
            return;
        }
        let mut state = self.lock();
        state.revision = state
            .revision
            .checked_add(1)
            .expect("column revision exhausted");
        if state.events.len() == COLUMN_CHANGE_LOG_CAPACITY {
            let evicted_revision = state.revision - COLUMN_CHANGE_LOG_CAPACITY as u64;
            let evicted = state.events.pop_front().expect("full journal has front");
            match evicted.component {
                Some(component) => {
                    let index = state
                        .component_events
                        .get_mut(&component)
                        .expect("retained component event has an index");
                    let removed_revision = index.pop_front();
                    debug_assert_eq!(removed_revision, Some(evicted_revision));
                    if index.is_empty() {
                        state.component_events.remove(&component);
                    }
                    state
                        .evicted_component_revision
                        .insert(component, evicted_revision);
                    if state.evicted_component_revision.len() > COLUMN_CHANGE_LOG_CAPACITY
                        && let Some((&old_component, &oldest_evicted)) =
                            state.evicted_component_revision.iter().next()
                    {
                        state.evicted_component_revision.remove(&old_component);
                        // Dropping any component watermark is safe if every
                        // cursor before its last evicted revision falls back.
                        // This keeps steady-state eviction O(1), without a
                        // sorted metadata index on the write hot path.
                        state.precision_floor = state.precision_floor.max(oldest_evicted);
                    }
                }
                None => {
                    let removed_revision = state.structural_events.pop_front();
                    debug_assert_eq!(removed_revision, Some(evicted_revision));
                    state.evicted_structural_revision = evicted_revision;
                }
            }
        }
        let revision = state.revision;
        state.events.push_back(event);
        match event.component {
            Some(component) => state
                .component_events
                .entry(component)
                .or_default()
                .push_back(revision),
            None => state.structural_events.push_back(revision),
        }
    }

    fn since(&self, cursor: Option<ColumnChangeCursor>) -> ColumnChanges {
        let state = self.lock();
        self.active.store(true, Ordering::Release);
        let current = ColumnChangeCursor {
            log: self.id,
            revision: state.revision,
        };
        let offset = cursor.filter(|c| c.log == self.id).and_then(|c| {
            let first = state.revision - state.events.len() as u64;
            (first..=state.revision)
                .contains(&c.revision)
                .then(|| (c.revision - first) as usize)
        });
        let events = offset.map_or_else(Vec::new, |start| {
            state.events.range(start..).copied().collect()
        });
        ColumnChanges {
            cursor: current,
            gap: offset.is_none(),
            events,
        }
    }

    fn since_components(
        &self,
        cursor: Option<ColumnChangeCursor>,
        components: &[ComponentId],
    ) -> FilteredColumnChanges {
        let state = self.lock();
        self.active.store(true, Ordering::Release);
        let current = ColumnChangeCursor {
            log: self.id,
            revision: state.revision,
        };
        let Some(cursor) = cursor.filter(|cursor| {
            cursor.log == self.id
                && cursor.revision <= state.revision
                && cursor.revision >= state.precision_floor
        }) else {
            return FilteredColumnChanges {
                cursor: current,
                gap: true,
                events: Vec::new(),
                records_scanned: 0,
            };
        };

        let sorted_components = if components.windows(2).all(|pair| pair[0] < pair[1]) {
            None
        } else {
            let mut sorted = components.to_vec();
            sorted.sort_unstable();
            sorted.dedup();
            Some(sorted)
        };
        let component_ids = sorted_components.as_deref().unwrap_or(components);
        if state.evicted_structural_revision > cursor.revision
            || component_ids.iter().any(|component| {
                state
                    .evicted_component_revision
                    .get(component)
                    .is_some_and(|revision| *revision > cursor.revision)
            })
        {
            return FilteredColumnChanges {
                cursor: current,
                gap: true,
                events: Vec::new(),
                records_scanned: 0,
            };
        }

        let first_retained = state.revision - state.events.len() as u64;
        let mut records_scanned = 0;
        let mut sources = Vec::with_capacity(component_ids.len() + 1);
        for component in component_ids {
            if let Some(index) = state.component_events.get(component) {
                let start = lower_bound_revision(index, cursor.revision);
                records_scanned += index.len() - start;
                sources.push((index, start));
            }
        }
        let structural_start = lower_bound_revision(&state.structural_events, cursor.revision);
        records_scanned += state.structural_events.len() - structural_start;
        if structural_start < state.structural_events.len() {
            sources.push((&state.structural_events, structural_start));
        }

        // Allocate according to the number of distinct runs, not raw writes.
        // Clustered row writes commonly collapse thousands of records to a
        // handful of chunks; scattered workloads grow this buffer as needed.
        let mut events: Vec<FilteredColumnChangeRun> = Vec::new();
        if sources.len() <= 4 {
            // The common query reads one or a few components. A stack cursor
            // merge avoids allocating and mutating a heap once per journal
            // event; retain the heap path only for high-arity queries.
            let mut cursors = [0usize; 4];
            loop {
                let next = sources
                    .iter()
                    .enumerate()
                    .filter_map(|(source, (revisions, start))| {
                        revisions
                            .get(start + cursors[source])
                            .copied()
                            .map(|r| (r, source))
                    })
                    .min();
                let Some((revision, source)) = next else {
                    break;
                };
                let offset = (revision - first_retained - 1) as usize;
                push_column_change_run(&mut events, state.events[offset]);
                cursors[source] += 1;
            }
        } else {
            // Every source is strictly ordered and sources are disjoint (each
            // event belongs to one component index or the structural index).
            // Merge in O(k log m), where k is selected event count and m is
            // source count, without sorting a materialized revision list.
            let mut heap = BinaryHeap::with_capacity(sources.len());
            for (source, (revisions, position)) in sources.iter().enumerate() {
                if let Some(&revision) = revisions.get(*position) {
                    heap.push(Reverse((revision, source)));
                }
            }
            while let Some(Reverse((revision, source))) = heap.pop() {
                let offset = (revision - first_retained - 1) as usize;
                push_column_change_run(&mut events, state.events[offset]);

                let (revisions, position) = &mut sources[source];
                *position += 1;
                if let Some(&next) = revisions.get(*position) {
                    heap.push(Reverse((next, source)));
                }
            }
        }
        FilteredColumnChanges {
            cursor: current,
            gap: false,
            events,
            records_scanned,
        }
    }

    fn cursor(&self) -> ColumnChangeCursor {
        let state = self.lock();
        self.active.store(true, Ordering::Release);
        ColumnChangeCursor {
            log: self.id,
            revision: state.revision,
        }
    }
}

/// First position whose revision is strictly greater than `cursor_revision`.
fn lower_bound_revision(revisions: &VecDeque<u64>, cursor_revision: u64) -> usize {
    let (mut low, mut high) = (0, revisions.len());
    while low < high {
        let mid = low + (high - low) / 2;
        if revisions[mid] <= cursor_revision {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    low
}

/// A stable, dense identifier for an [`Archetype`] within a world.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct ArchetypeId(u32);

impl ArchetypeId {
    /// Sentinel used by [`EntityLocation::EMPTY`](crate::entity::EntityLocation).
    pub const INVALID: Self = Self(u32::MAX);

    /// The archetype holding entities with no components (the "empty"
    /// archetype), always allocated at construction.
    pub const EMPTY: Self = Self(0);

    /// Construct an [`ArchetypeId`] from its raw index.
    #[inline]
    pub const fn new(index: u32) -> Self {
        Self(index)
    }

    /// The raw dense index.
    #[inline]
    pub const fn index(self) -> u32 {
        self.0
    }
}

/// A single shared-component binding carried by an archetype: the shared
/// [`ComponentId`], the interned [`SharedValueId`] every entity here shares,
/// and a canonical [`Arc`] handle keeping the value alive for the archetype's
/// lifetime (design §6 SharedComponent). Bindings are stored sorted by
/// [`ComponentId`] so lookup is a binary search and the derived archetype key
/// is canonical.
pub type SharedBinding = (ComponentId, SharedValueId, Arc<dyn SharedValue>);

/// A fragmenting relation's archetype key: its kind and full generational
/// target. Unlike a packed pair key, this never aliases a recycled target.
pub type RelationPair = (ComponentId, Entity);

/// One archetype: a (non-shared) component set plus the table storing its
/// entities, together with the shared-component value bindings that split it
/// from otherwise-identical archetypes (design §6), and fragmenting relation
/// pairs (design §11). Permanently evicted stable-ID entries retain only their
/// ID, empty metadata/storage, and eviction marker.
pub struct Archetype {
    id: ArchetypeId,
    components: ComponentSet,
    shared: Box<[SharedBinding]>,
    pairs: Box<[RelationPair]>,
    /// Evicted stable-ID tombstones replace their backing table with an empty
    /// shell and release component/relation/shared-key allocations. The slot
    /// remains addressable to preserve stable ArchetypeIds.
    table: Table,
    evicted: bool,
}

impl Archetype {
    /// This archetype's id.
    #[inline]
    pub fn id(&self) -> ArchetypeId {
        self.id
    }

    /// The exact set of component ids every entity in a live archetype has.
    /// Permanently evicted archetypes release this metadata and return the
    /// empty set; use [`Archetype::is_evicted`] to distinguish that tombstone
    /// from the live empty archetype.
    #[inline]
    pub fn components(&self) -> &ComponentSet {
        &self.components
    }

    /// Whether entities here have the table/sparse component `id`. Shared
    /// bindings are *not* reported here (they are archetype-level batch keys,
    /// not table columns); query [`Archetype::shared_binding`] for those.
    /// Permanently evicted tombstones return `false` for every component.
    #[inline]
    pub fn contains(&self, id: ComponentId) -> bool {
        self.components.contains(id)
    }

    /// The interned [`SharedValueId`] bound to shared component `id` in this
    /// archetype, or `None` if this archetype has no binding for `id` (design
    /// §6). This is the precise, per-archetype membership test for a shared
    /// component. Permanently evicted tombstones return `None`.
    #[inline]
    pub fn shared_binding(&self, id: ComponentId) -> Option<SharedValueId> {
        self.shared
            .binary_search_by_key(&id, |&(cid, _, _)| cid)
            .ok()
            .map(|i| self.shared[i].1)
    }

    /// A canonical [`Arc`] handle to the interned value bound to shared
    /// component `id` in this archetype, or `None` if unbound (design §6). The
    /// value is immutable; downcast via
    /// [`SharedValue::as_any`](crate::storage::SharedValue::as_any) to read it.
    /// Permanently evicted tombstones return `None`.
    #[inline]
    pub fn shared_arc(&self, id: ComponentId) -> Option<&Arc<dyn SharedValue>> {
        self.shared
            .binary_search_by_key(&id, |&(cid, _, _)| cid)
            .ok()
            .map(|i| &self.shared[i].2)
    }

    /// All shared-component bindings on this live archetype, sorted by
    /// [`ComponentId`] (design §6). Permanently evicted tombstones return an
    /// empty slice.
    #[inline]
    pub fn shared_bindings(&self) -> &[SharedBinding] {
        &self.shared
    }

    /// Fragmenting relation pairs shared by every entity in this live
    /// archetype. Sorted and duplicate-free; pairs are identity, not component
    /// columns. Evicted tombstones release this metadata and return an empty
    /// slice.
    #[inline]
    pub fn relation_pairs(&self) -> &[RelationPair] {
        &self.pairs
    }

    /// Number of entities in this archetype. Permanently evicted tombstones
    /// have length zero.
    #[inline]
    pub fn len(&self) -> usize {
        self.table.len()
    }

    /// Whether this archetype currently holds no entities.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
    }

    /// Whether this stable-id tombstone was permanently evicted from lookup
    /// indexes after its last shared-value reference was released.
    ///
    /// Evicted archetypes remain addressable through [`Archetypes::get`] and
    /// [`Archetypes::iter`] to preserve stable IDs, but their component,
    /// shared-binding, relation-pair, and table storage metadata is released.
    /// This distinguishes an evicted tombstone from a live empty archetype.
    #[inline]
    pub fn is_evicted(&self) -> bool {
        self.evicted
    }

    /// Immutable access to the backing table.
    #[inline]
    pub fn table(&self) -> &Table {
        &self.table
    }

    /// Mutable access to the backing table.
    #[inline]
    pub fn table_mut(&mut self) -> &mut Table {
        &mut self.table
    }

    /// Reserve room for `additional` more rows in this archetype's table
    /// (design §17 原型预热).
    pub(crate) fn reserve(&mut self, additional: usize) {
        self.table.reserve(additional);
    }

    /// Drop storage and all archetype identity metadata after permanent
    /// eviction. The stable ID remains addressable, while evicted IDs are
    /// excluded before any table access or query matching.
    fn release_evicted_storage(&mut self) {
        debug_assert!(self.evicted);
        debug_assert!(self.is_empty());
        self.table = Table::new(core::iter::empty());
        self.components = ComponentSet::default();
        self.shared = Box::default();
        self.pairs = Box::default();
    }
}

/// The canonical identity of an archetype: its non-shared [`ComponentSet`]
/// plus its sorted shared-value bindings and fragmenting relation pairs. Two
/// archetypes with different shared values or pairs have distinct keys
/// (design §6, §11). Shared bindings
/// are reduced to `(ComponentId, SharedValueId)` here (the [`Arc`] handle is
/// identity-irrelevant and lives on the [`Archetype`]).
#[derive(Clone, PartialEq, Eq, Hash, Debug, Default)]
struct ArchetypeKey {
    set: ComponentSet,
    shared: Box<[(ComponentId, SharedValueId)]>,
    pairs: Box<[RelationPair]>,
}

impl ArchetypeKey {
    /// Build a canonical key, sorting the shared bindings by [`ComponentId`].
    fn new(
        set: ComponentSet,
        mut shared: Vec<(ComponentId, SharedValueId)>,
        mut pairs: Vec<RelationPair>,
    ) -> Self {
        shared.sort_unstable_by_key(|&(cid, _)| cid);
        pairs.sort_unstable();
        pairs.dedup();
        Self {
            set,
            shared: shared.into_boxed_slice(),
            pairs: pairs.into_boxed_slice(),
        }
    }

    /// Derive the key for an archetype's table set and (already-materialized)
    /// shared bindings.
    fn from_bindings(set: &ComponentSet, shared: &[SharedBinding], pairs: &[RelationPair]) -> Self {
        Self::new(
            set.clone(),
            shared.iter().map(|&(cid, sid, _)| (cid, sid)).collect(),
            pairs.to_vec(),
        )
    }
}

/// A sorted archetype posting. Small postings stay in one bounded vector;
/// larger postings use an ordered block directory. IDs are allocated in
/// monotonically increasing order, so insertion appends. Non-tail eviction
/// compacts only touched blocks for large postings.
fn remove_small_posting_ids(posting: &mut Vec<ArchetypeId>, removed: &[ArchetypeId]) {
    if let Some(start) = posting.len().checked_sub(removed.len())
        && posting.get(start..) == Some(removed)
    {
        posting.truncate(start);
    } else {
        posting.retain(|id| removed.binary_search(id).is_err());
    }
}

#[derive(Default)]
pub(crate) struct ArchetypePosting {
    small: Vec<ArchetypeId>,
    blocks: BTreeMap<ArchetypeId, Vec<ArchetypeId>>,
    len: usize,
}

impl ArchetypePosting {
    const BLOCK_CAPACITY: usize = 128;

    fn push(&mut self, id: ArchetypeId) {
        if self.blocks.is_empty() && self.small.len() < Self::BLOCK_CAPACITY {
            self.small.push(id);
            self.len += 1;
            return;
        }
        if self.blocks.is_empty() {
            let small = core::mem::take(&mut self.small);
            self.blocks.insert(
                *small.first().expect("full small posting is non-empty"),
                small,
            );
        }
        if self
            .blocks
            .last_key_value()
            .is_none_or(|(_, block)| block.len() == Self::BLOCK_CAPACITY)
        {
            self.blocks
                .insert(id, Vec::with_capacity(Self::BLOCK_CAPACITY));
        }
        self.blocks
            .last_entry()
            .expect("posting block was added")
            .get_mut()
            .push(id);
        self.len += 1;
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub(crate) fn blocks(&self) -> impl Iterator<Item = &[ArchetypeId]> {
        core::iter::once(self.small.as_slice())
            .chain(self.blocks.values().map(Vec::as_slice))
            .filter(|block| !block.is_empty())
    }

    pub(crate) fn iter(&self) -> ArchetypePostingIter<'_> {
        ArchetypePostingIter {
            small: self.small.iter(),
            blocks: self.blocks.values(),
            block: [].iter(),
        }
    }

    #[cfg(test)]
    pub(crate) fn contains(&self, id: ArchetypeId) -> bool {
        if self.blocks.is_empty() {
            return self.small.binary_search(&id).is_ok();
        }
        self.blocks
            .range(..=id)
            .next_back()
            .is_some_and(|(_, block)| {
                block.last().is_some_and(|&last| id <= last) && block.binary_search(&id).is_ok()
            })
    }

    fn remove_sorted(&mut self, removed: &[ArchetypeId]) {
        if removed.is_empty() || self.is_empty() {
            return;
        }
        if self.blocks.is_empty() {
            let old_len = self.small.len();
            remove_small_posting_ids(&mut self.small, removed);
            self.len -= old_len - self.small.len();
            return;
        }

        // Keep the common newest-archetype churn path allocation-free.
        if self.len >= removed.len()
            && self
                .blocks
                .values()
                .rev()
                .flat_map(|block| block.iter().rev())
                .take(removed.len())
                .copied()
                .eq(removed.iter().rev().copied())
        {
            self.truncate(self.len - removed.len());
            return;
        }

        // Locate each touched block through the ordered directory and compact
        // only that block. Removed IDs are sorted, so each block is visited at
        // most once per batch.
        let mut cursor = 0;
        while cursor < removed.len() {
            let id = removed[cursor];
            let Some((&key, block)) = self.blocks.range(..=id).next_back() else {
                cursor += 1;
                continue;
            };
            let upper = *block.last().expect("posting blocks are non-empty");
            if upper < id {
                cursor += 1;
                continue;
            }
            let end = cursor + removed[cursor..].partition_point(|&candidate| candidate <= upper);
            let old_block = self.blocks.remove(&key).expect("located posting block");
            let mut compacted = Vec::with_capacity(Self::BLOCK_CAPACITY);
            let mut removal_cursor = cursor;
            for candidate in old_block {
                if removed.get(removal_cursor) == Some(&candidate) {
                    removal_cursor += 1;
                    self.len -= 1;
                } else {
                    compacted.push(candidate);
                }
            }
            if let Some(&new_key) = compacted.first() {
                self.blocks.insert(new_key, compacted);
                self.coalesce(new_key);
            }
            cursor = end;
        }
        if self.len <= Self::BLOCK_CAPACITY {
            self.small = self.blocks.values().flatten().copied().collect();
            self.blocks.clear();
        }
    }

    fn truncate(&mut self, new_len: usize) {
        if self.blocks.is_empty() {
            self.small.truncate(new_len);
            self.len = new_len;
            return;
        }
        while self.len > new_len {
            let (key, block_len) = self
                .blocks
                .last_key_value()
                .map(|(&key, block)| (key, block.len()))
                .expect("non-empty posting");
            let excess = self.len - new_len;
            if excess >= block_len {
                self.blocks.remove(&key);
                self.len -= block_len;
            } else {
                let mut block = self.blocks.remove(&key).expect("last block exists");
                block.truncate(block.len() - excess);
                self.len = new_len;
                self.blocks.insert(key, block);
            }
        }
        if self.len <= Self::BLOCK_CAPACITY {
            self.small = self.blocks.values().flatten().copied().collect();
            self.blocks.clear();
        }
    }

    fn coalesce(&mut self, mut key: ArchetypeId) {
        if let Some(previous) = self.blocks.range(..key).next_back().map(|(&key, _)| key) {
            let combined_len = self.blocks[&previous].len() + self.blocks[&key].len();
            if combined_len <= Self::BLOCK_CAPACITY {
                let mut merged = self
                    .blocks
                    .remove(&previous)
                    .expect("previous block exists");
                let mut current = self.blocks.remove(&key).expect("current block exists");
                merged.append(&mut current);
                self.blocks.insert(previous, merged);
                key = previous;
            }
        }

        if let Some(next) = self
            .blocks
            .range((Excluded(key), Unbounded))
            .next()
            .map(|(&key, _)| key)
        {
            let combined_len = self.blocks[&key].len() + self.blocks[&next].len();
            if combined_len <= Self::BLOCK_CAPACITY {
                let mut merged = self.blocks.remove(&key).expect("current block exists");
                let mut following = self.blocks.remove(&next).expect("next block exists");
                merged.append(&mut following);
                self.blocks.insert(key, merged);
            }
        }
    }
}

/// Allocation-free iterator over a posting's inline vector and ordered blocks.
pub(crate) struct ArchetypePostingIter<'a> {
    small: core::slice::Iter<'a, ArchetypeId>,
    blocks: alloc::collections::btree_map::Values<'a, ArchetypeId, Vec<ArchetypeId>>,
    block: core::slice::Iter<'a, ArchetypeId>,
}

impl Iterator for ArchetypePostingIter<'_> {
    type Item = ArchetypeId;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(&id) = self.small.next() {
            return Some(id);
        }
        loop {
            if let Some(&id) = self.block.next() {
                return Some(id);
            }
            self.block = self.blocks.next()?.iter();
        }
    }
}

/// The per-world registry of archetypes.
pub struct Archetypes {
    archetypes: Vec<Archetype>,
    by_key: HashMap<ArchetypeKey, ArchetypeId>,
    /// Inverted table-column index. Each posting list is ordered by
    /// `ArchetypeId`, matching the registry's stable insertion order. Entries
    /// are removed when a shared-value archetype is permanently evicted.
    /// Dirty-query fallback uses it to avoid testing every matched archetype
    /// for every accessed table column.
    by_component: HashMap<ComponentId, ArchetypePosting>,
    /// Shared-value reverse index. Each posting is ordered by stable
    /// `ArchetypeId` and contains every non-evicted archetype bound to the
    /// `(component, value)` pair, including currently live archetypes.
    by_shared_binding: HashMap<(ComponentId, SharedValueId), ArchetypePosting>,
    column_journal: Arc<ColumnJournal>,
    /// Recent eviction IDs let query caches skip full snapshot compaction when
    /// churn is unrelated to their matches. Old entries are discarded at a
    /// fixed bound; lagging query caches then conservatively inspect their own
    /// matched IDs against tombstone state.
    evicted_ids: VecDeque<ArchetypeId>,
    /// Absolute number of evictions, used to distinguish a retained suffix
    /// from a query cursor that fell behind the bounded log.
    eviction_count: usize,
}

impl Archetypes {
    /// Create a registry pre-populated with the empty archetype at
    /// [`ArchetypeId::EMPTY`].
    pub fn new() -> Self {
        let mut this = Self {
            archetypes: Vec::new(),
            by_key: HashMap::default(),
            by_component: HashMap::default(),
            by_shared_binding: HashMap::default(),
            column_journal: Arc::new(ColumnJournal::new()),
            evicted_ids: VecDeque::new(),
            eviction_count: 0,
        };
        let empty = ComponentSet::default();
        let id = this.insert(empty, &[], &[], &Components::new());
        debug_assert_eq!(id, ArchetypeId::EMPTY);
        this
    }

    /// Number of addressable stable-ID slots, including permanently evicted
    /// tombstones.
    #[inline]
    pub fn len(&self) -> usize {
        self.archetypes.len()
    }

    /// Whether there are no archetypes (never true after [`Archetypes::new`]).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.archetypes.is_empty()
    }

    /// Borrow an archetype by id, including a permanently evicted tombstone.
    #[inline]
    pub fn get(&self, id: ArchetypeId) -> Option<&Archetype> {
        self.archetypes.get(id.0 as usize)
    }

    /// Mutably borrow an archetype by id.
    #[inline]
    pub fn get_mut(&mut self, id: ArchetypeId) -> Option<&mut Archetype> {
        self.archetypes.get_mut(id.0 as usize)
    }

    /// Mutably borrow two distinct archetypes at once (for structural moves).
    ///
    /// # Panics
    /// Panics if `a == b` or either id is out of range.
    pub fn get_pair_mut(
        &mut self,
        a: ArchetypeId,
        b: ArchetypeId,
    ) -> (&mut Archetype, &mut Archetype) {
        assert_ne!(a, b, "cannot borrow the same archetype twice");
        let (lo, hi) = (a.0.min(b.0) as usize, a.0.max(b.0) as usize);
        let (left, right) = self.archetypes.split_at_mut(hi);
        let lo_ref = &mut left[lo];
        let hi_ref = &mut right[0];
        if a.0 < b.0 {
            (lo_ref, hi_ref)
        } else {
            (hi_ref, lo_ref)
        }
    }

    /// All stable-ID slots in id order, including permanently evicted
    /// tombstones. Use [`Archetype::is_evicted`] to distinguish tombstones
    /// from live empty archetypes.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &Archetype> {
        self.archetypes.iter()
    }

    /// Archetypes whose table component set contains `component`, in stable
    /// ascending id order. The returned index includes empty archetypes that
    /// remain addressable by key and shared-value variants, but excludes
    /// permanently evicted tombstones. Callers must still apply their query
    /// match and live-row checks.
    #[inline]
    pub(crate) fn with_component(&self, component: ComponentId) -> Option<&ArchetypePosting> {
        self.by_component.get(&component)
    }

    /// Archetypes bound to one interned shared value, in stable id order.
    /// Permanently evicted tombstones are removed from these postings.
    #[cfg(test)]
    #[inline]
    pub(crate) fn with_shared_binding(
        &self,
        component: ComponentId,
        value: SharedValueId,
    ) -> Option<&ArchetypePosting> {
        self.by_shared_binding.get(&(component, value))
    }

    /// Current absolute eviction cursor.
    #[inline]
    pub(crate) fn eviction_cursor(&self) -> usize {
        self.eviction_count
    }

    /// Retained evictions since `cursor`, or `None` when the cursor predates
    /// the bounded log and the caller must conservatively inspect its cache.
    #[inline]
    pub(crate) fn evictions_since(
        &self,
        cursor: usize,
    ) -> Option<core::iter::Skip<alloc::collections::vec_deque::Iter<'_, ArchetypeId>>> {
        let retained_start = self.eviction_count - self.evicted_ids.len();
        (cursor >= retained_start && cursor <= self.eviction_count)
            .then(|| self.evicted_ids.iter().skip(cursor - retained_start))
    }

    /// Discover writes/structural edits in O(retained events since cursor), not
    /// O(archetypes/chunks). A successful call materializes every retained event
    /// after the cursor, so its deterministic record-scan count is `events.len()`;
    /// a gap materializes/scans zero delta records. Repeated same-tick writes
    /// advance this cursor too. Payload access still requires the usual ECS
    /// read/write scheduling barrier.
    pub fn column_changes_since(&self, cursor: Option<ColumnChangeCursor>) -> ColumnChanges {
        self.column_journal.since(cursor)
    }

    /// Discover structural changes and writes to selected components without
    /// scanning unrelated retained journal records. `component_ids` may be in
    /// any order and may contain duplicates. Structural records (`component ==
    /// None`) are always included. The journal remains globally capped at
    /// [`COLUMN_CHANGE_LOG_CAPACITY`]; if a selected record may have been
    /// evicted, this returns a gap and no events so callers can rebuild a
    /// conservative baseline. Evictions known to affect only unrelated
    /// components do not force a gap unless bounded eviction metadata itself
    /// overflowed, in which case older cursors conservatively gap.
    pub(crate) fn column_changes_since_components(
        &self,
        cursor: Option<ColumnChangeCursor>,
        component_ids: &[ComponentId],
    ) -> FilteredColumnChanges {
        self.column_journal.since_components(cursor, component_ids)
    }

    /// Establish an incremental baseline without materializing retained events.
    /// Used after a conservative fallback has covered the current tick window.
    #[inline]
    pub(crate) fn column_change_cursor(&self) -> ColumnChangeCursor {
        self.column_journal.cursor()
    }

    /// All archetypes mutably, in id order.
    #[inline]
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Archetype> {
        self.archetypes.iter_mut()
    }

    /// Get the id for `set`, creating the archetype (and its empty table) if it
    /// does not yet exist. `components` supplies the layout/drop metadata needed
    /// to build the table columns.
    pub fn get_or_create(&mut self, set: &ComponentSet, components: &Components) -> ArchetypeId {
        self.get_or_create_shared(set, &[], components)
    }

    /// Get the id for the archetype identified by table component `set` *and*
    /// the given shared-value `bindings`, creating it (and its empty table) if
    /// it does not yet exist (design §6 SharedComponent archetype split).
    ///
    /// `bindings` need not be sorted; the derived key is canonicalized. The
    /// supplied [`Arc`] handles are retained on the created archetype to keep
    /// each interned value alive for the archetype's lifetime (shared values
    /// are immortal once an archetype binds them). `components` supplies the
    /// layout/drop metadata for the table columns of `set`.
    pub fn get_or_create_shared(
        &mut self,
        set: &ComponentSet,
        bindings: &[SharedBinding],
        components: &Components,
    ) -> ArchetypeId {
        self.get_or_create_with_pairs(set, bindings, &[], components)
    }

    /// Resolve an archetype by components, shared bindings and fragmenting
    /// relation pairs. Input pairs may be unsorted or repeated. No column or
    /// component lifecycle is created for a pair.
    pub fn get_or_create_with_pairs(
        &mut self,
        set: &ComponentSet,
        bindings: &[SharedBinding],
        pairs: &[RelationPair],
        components: &Components,
    ) -> ArchetypeId {
        let key = ArchetypeKey::from_bindings(set, bindings, pairs);
        if let Some(&id) = self.by_key.get(&key) {
            return id;
        }
        self.insert(set.clone(), bindings, &key.pairs, components)
    }

    /// Drop the `by_key` lookup entry of every **empty** archetype whose shared
    /// bindings include `(cid, sid)`, after that `(cid, sid)` has had its last
    /// reference released from its value pool (design §6 SharedComponent 生命周期).
    ///
    /// A freed [`SharedValueId`] may be recycled by the pool to name a *different*
    /// value. If a now-stale empty archetype keyed on the old `(cid, sid)` stayed
    /// in `by_key`, a later spawn of that different value with the same table set
    /// would collide with it in [`Archetypes::get_or_create_shared`] and route the
    /// entity into an archetype still holding the *old* value's [`Arc`] — a silent
    /// wrong-value read. Evicting the lookup entry closes that aliasing window.
    ///
    /// The [`Archetype`] struct is left tombstoned in the `archetypes` vec so
    /// existing [`ArchetypeId`]s (dense vec indices) stay stable; its key and
    /// inverted-index entries are removed, so the tombstone can never again be
    /// returned by a lookup and a fresh archetype is minted on demand instead.
    /// This is safe because `refcount(cid, sid) == 0` implies no live entity
    /// binds `(cid, sid)`, so every archetype binding it is empty and
    /// unreferenced by any [`EntityLocation`](crate::entity::EntityLocation).
    pub fn evict_shared_binding(&mut self, cid: ComponentId, sid: SharedValueId) {
        let Self {
            archetypes,
            by_key,
            by_component,
            by_shared_binding,
            evicted_ids: eviction_log,
            eviction_count,
            ..
        } = self;
        let mut evicted: Vec<ArchetypeId> = by_shared_binding
            .get(&(cid, sid))
            .into_iter()
            .flat_map(|posting| posting.blocks().flatten().copied())
            .filter(|id| archetypes[id.index() as usize].is_empty())
            .collect();

        if evicted.is_empty() {
            return;
        }

        // Keep stable ArchetypeIds in `archetypes`, but reclaim their lookup
        // and inverted-index entries. Reconstructing each canonical key avoids
        // scanning the global `by_key` map during a shared-value churn event.
        evicted.sort_unstable();
        for &id in &evicted {
            let arch = &archetypes[id.index() as usize];
            let key = ArchetypeKey::from_bindings(&arch.components, &arch.shared, &arch.pairs);
            let removed = by_key.remove(&key);
            debug_assert_eq!(removed, Some(id));
            archetypes[id.index() as usize].evicted = true;
        }
        *eviction_count = eviction_count
            .checked_add(evicted.len())
            .expect("eviction count cannot exceed the archetype registry length");
        let overflow = eviction_log.len().saturating_sub(
            ARCHETYPE_EVICTION_LOG_CAPACITY - evicted.len().min(ARCHETYPE_EVICTION_LOG_CAPACITY),
        );
        if evicted.len() >= ARCHETYPE_EVICTION_LOG_CAPACITY {
            eviction_log.clear();
            eviction_log.extend(
                evicted[evicted.len() - ARCHETYPE_EVICTION_LOG_CAPACITY..]
                    .iter()
                    .copied(),
            );
        } else {
            for _ in 0..overflow {
                eviction_log.pop_front();
            }
            eviction_log.extend(evicted.iter().copied());
        }

        let mut component_evictions = Vec::new();
        let mut binding_evictions = Vec::new();
        for &id in &evicted {
            let arch = &archetypes[id.index() as usize];
            component_evictions.extend(
                arch.components
                    .ids()
                    .iter()
                    .map(|&component| (component, id)),
            );
            binding_evictions.extend(
                arch.shared
                    .iter()
                    .map(|&(component, value, _)| ((component, value), id)),
            );
        }
        component_evictions.sort_unstable();
        binding_evictions.sort_unstable();
        binding_evictions.dedup();

        // All data needed to clean the postings is now copied into the
        // eviction batches. Release each stable tombstone's heavyweight table
        // and shared-value Arcs before touching the postings.
        for &id in &evicted {
            archetypes[id.index() as usize].release_evicted_storage();
        }

        let mut removed_ids = Vec::new();
        let mut index = 0;
        while index < component_evictions.len() {
            let component = component_evictions[index].0;
            let end = component_evictions[index..]
                .iter()
                .position(|&(candidate, _)| candidate != component)
                .map_or(component_evictions.len(), |offset| index + offset);
            removed_ids.clear();
            removed_ids.extend(component_evictions[index..end].iter().map(|&(_, id)| id));
            let remove_entry = if let Some(postings) = by_component.get_mut(&component) {
                postings.remove_sorted(&removed_ids);
                postings.is_empty()
            } else {
                false
            };
            if remove_entry {
                by_component.remove(&component);
            }
            index = end;
        }

        index = 0;
        while index < binding_evictions.len() {
            let binding = binding_evictions[index].0;
            let end = binding_evictions[index..]
                .iter()
                .position(|&(candidate, _)| candidate != binding)
                .map_or(binding_evictions.len(), |offset| index + offset);
            removed_ids.clear();
            removed_ids.extend(binding_evictions[index..end].iter().map(|&(_, id)| id));
            let remove_entry = if let Some(postings) = by_shared_binding.get_mut(&binding) {
                postings.remove_sorted(&removed_ids);
                postings.is_empty()
            } else {
                false
            };
            if remove_entry {
                by_shared_binding.remove(&binding);
            }
            index = end;
        }
    }

    fn insert(
        &mut self,
        set: ComponentSet,
        bindings: &[SharedBinding],
        pairs: &[RelationPair],
        components: &Components,
    ) -> ArchetypeId {
        let id = ArchetypeId(self.archetypes.len() as u32);
        for &component in set.ids() {
            self.by_component.entry(component).or_default().push(id);
        }
        let mut shared_keys: Vec<_> = bindings
            .iter()
            .map(|&(component, value, _)| (component, value))
            .collect();
        shared_keys.sort_unstable();
        shared_keys.dedup();
        for binding in shared_keys {
            self.by_shared_binding.entry(binding).or_default().push(id);
        }
        let columns = set.ids().iter().map(|&cid| {
            let info = components
                .info(cid)
                .expect("component in archetype set must be registered");
            (cid, info.layout(), info.drop_fn())
        });
        let mut table = Table::new(columns);
        table.attach_column_journal(id, self.column_journal.clone());
        let mut shared: Vec<SharedBinding> = bindings.to_vec();
        shared.sort_unstable_by_key(|&(cid, _, _)| cid);
        let shared = shared.into_boxed_slice();
        let key = ArchetypeKey::from_bindings(&set, &shared, pairs);
        let pairs = key.pairs.clone();
        self.archetypes.push(Archetype {
            id,
            components: set,
            shared,
            pairs,
            table,
            evicted: false,
        });
        self.by_key.insert(key, id);
        self.column_journal.record(ColumnChange {
            archetype: id,
            component: None,
            chunk: None,
        });
        id
    }
}

impl Default for Archetypes {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component::{Component, Components};
    use alloc::vec;

    struct A(#[allow(dead_code)] u32);
    impl Component for A {}
    struct B(#[allow(dead_code)] u64);
    impl Component for B {}

    fn column_event() -> ColumnChange {
        ColumnChange {
            archetype: ArchetypeId::EMPTY,
            component: None,
            chunk: Some(0),
        }
    }

    fn component_event(component: ComponentId, chunk: usize) -> ColumnChange {
        ColumnChange {
            archetype: ArchetypeId::EMPTY,
            component: Some(component),
            chunk: Some(chunk),
        }
    }

    fn coalesce_column_changes(
        events: impl IntoIterator<Item = ColumnChange>,
    ) -> Vec<FilteredColumnChangeRun> {
        let mut runs: Vec<FilteredColumnChangeRun> = Vec::new();
        for change in events {
            if change.chunk.is_some()
                && let Some(previous) = runs.last_mut()
                && previous.change.archetype == change.archetype
                && previous.change.chunk == change.chunk
            {
                previous.event_count += 1;
            } else {
                runs.push(FilteredColumnChangeRun {
                    change,
                    event_count: 1,
                });
            }
        }
        runs
    }

    #[test]
    fn filtered_column_journal_merges_selected_components_and_structural_events_in_order() {
        let journal = ColumnJournal::new();
        let a = ComponentId::new(1);
        let b = ComponentId::new(2);
        let unrelated = ComponentId::new(3);
        let baseline = journal.since(None).cursor;
        journal.record(component_event(b, 0));
        journal.record(component_event(unrelated, 1));
        journal.record(component_event(a, 2));
        journal.record(column_event());
        journal.record(component_event(a, 3));

        let full = journal.since(Some(baseline));
        let filtered = journal.since_components(Some(baseline), &[b, a, b]);
        let expected: Vec<_> = full
            .events
            .iter()
            .copied()
            .filter(|event| event.component.map_or(true, |id| id == a || id == b))
            .collect();
        assert!(!full.gap && !filtered.gap);
        assert_eq!(filtered.events, coalesce_column_changes(expected));
        assert_eq!(filtered.events.len(), 4);
        assert_eq!(filtered.records_scanned, 4);
        // The legacy full API's scan count is defined as the full returned
        // delta length; it deliberately has no added public stats field.
        assert_eq!(full.events.len(), 5);
        assert_eq!(
            filtered
                .events
                .iter()
                .map(|run| run.change.chunk)
                .collect::<Vec<_>>(),
            vec![Some(0), Some(2), Some(0), Some(3)]
        );
    }

    #[test]
    fn filtered_column_journal_coalesces_only_consecutive_same_chunk_events() {
        let journal = ColumnJournal::new();
        let component = ComponentId::new(1);
        let baseline = journal.since(None).cursor;
        journal.record(component_event(component, 4));
        journal.record(component_event(component, 4));
        journal.record(component_event(component, 5));
        journal.record(column_event());
        journal.record(component_event(component, 4));

        let filtered = journal.since_components(Some(baseline), &[component]);
        assert!(!filtered.gap);
        assert_eq!(filtered.records_scanned, 5);
        assert_eq!(
            filtered.events,
            vec![
                FilteredColumnChangeRun {
                    change: component_event(component, 4),
                    event_count: 2,
                },
                FilteredColumnChangeRun {
                    change: component_event(component, 5),
                    event_count: 1,
                },
                FilteredColumnChangeRun {
                    change: column_event(),
                    event_count: 1,
                },
                FilteredColumnChangeRun {
                    change: component_event(component, 4),
                    event_count: 1,
                },
            ]
        );
    }

    #[test]
    fn filtered_column_journal_k_way_merge_preserves_high_source_order() {
        let journal = ColumnJournal::new();
        let selected: Vec<_> = (0..32).map(|id| ComponentId::new(id + 1)).collect();
        let unrelated = ComponentId::new(10_000);
        let baseline = journal.since(None).cursor;

        for round in 0..4 {
            for (source, &component) in selected.iter().enumerate() {
                journal.record(component_event(component, round * 32 + source));
            }
            journal.record(component_event(unrelated, round));
            if round % 2 == 0 {
                journal.record(column_event());
            }
        }

        let full = journal.since(Some(baseline));
        let filtered = journal.since_components(Some(baseline), &selected);
        let expected: Vec<_> = full
            .events
            .iter()
            .copied()
            .filter(|event| event.component.is_none_or(|id| selected.contains(&id)))
            .collect();

        assert!(!full.gap && !filtered.gap);
        assert_eq!(filtered.events, coalesce_column_changes(expected));
        assert_eq!(filtered.records_scanned, selected.len() * 4 + 2);
    }

    #[test]
    fn filtered_column_journal_allows_unrelated_evictions_within_precision_window() {
        let journal = ColumnJournal::new();
        let selected = ComponentId::new(1);
        let unrelated = ComponentId::new(2);
        let baseline = journal.since(None).cursor;
        for _ in 0..COLUMN_CHANGE_LOG_CAPACITY {
            journal.record(component_event(unrelated, 9));
        }
        journal.record(component_event(selected, 7));

        // The selected record itself is still retained; the first unrelated
        // record was evicted. Filtered reads can prove that no selected record
        // was lost and need not rebuild.
        let filtered = journal.since_components(Some(baseline), &[selected]);
        assert!(!filtered.gap);
        assert_eq!(
            filtered.events,
            vec![FilteredColumnChangeRun {
                change: component_event(selected, 7),
                event_count: 1,
            }]
        );
        assert_eq!(filtered.records_scanned, 1);

        // The compatibility API remains a global snapshot and therefore gaps
        // after any retained record has been evicted.
        let full = journal.since(Some(baseline));
        assert!(full.gap);
        assert!(full.events.is_empty());
    }

    #[test]
    fn filtered_column_journal_gaps_when_selected_or_structural_record_is_evicted() {
        let selected_journal = ColumnJournal::new();
        let selected = ComponentId::new(1);
        let unrelated = ComponentId::new(2);
        let selected_baseline = selected_journal.since(None).cursor;
        selected_journal.record(component_event(selected, 0));
        for _ in 0..COLUMN_CHANGE_LOG_CAPACITY {
            selected_journal.record(component_event(unrelated, 1));
        }
        let selected_gap = selected_journal.since_components(Some(selected_baseline), &[selected]);
        assert!(selected_gap.gap);
        assert!(selected_gap.events.is_empty());
        assert_eq!(selected_gap.records_scanned, 0);

        let structural_journal = ColumnJournal::new();
        let structural_baseline = structural_journal.since(None).cursor;
        structural_journal.record(column_event());
        for _ in 0..COLUMN_CHANGE_LOG_CAPACITY {
            structural_journal.record(component_event(unrelated, 2));
        }
        let structural_gap =
            structural_journal.since_components(Some(structural_baseline), &[selected]);
        assert!(structural_gap.gap);
        assert!(structural_gap.events.is_empty());
        assert_eq!(structural_gap.records_scanned, 0);
    }

    #[test]
    fn filtered_column_journal_bounds_eviction_metadata_and_gaps_old_cursors() {
        let journal = ColumnJournal::new();
        let baseline = journal.since(None).cursor;
        for index in 0..(COLUMN_CHANGE_LOG_CAPACITY * 2 + 1) {
            journal.record(component_event(ComponentId::new(index as u32), 0));
        }

        let state = journal.lock();
        assert_eq!(state.events.len(), COLUMN_CHANGE_LOG_CAPACITY);
        assert!(state.evicted_component_revision.len() <= COLUMN_CHANGE_LOG_CAPACITY);
        let precision_cursor = ColumnChangeCursor {
            log: journal.id,
            revision: state.precision_floor,
        };
        drop(state);

        let old = journal.since_components(Some(baseline), &[]);
        assert!(old.gap);
        assert!(old.events.is_empty());
        let at_precision_floor = journal.since_components(Some(precision_cursor), &[]);
        assert!(!at_precision_floor.gap);
        assert!(at_precision_floor.events.is_empty());
        assert_eq!(at_precision_floor.records_scanned, 0);
    }

    #[test]
    fn column_journal_is_lazy_and_cursors_are_independent() {
        let journal = ColumnJournal::new();
        journal.record(column_event());
        assert_eq!(journal.lock().revision, 0);
        assert!(journal.lock().events.is_empty());

        let baseline = journal.since(None);
        assert!(baseline.gap);
        journal.record(column_event());
        let first = journal.since(Some(baseline.cursor));
        let independent = journal.since(Some(baseline.cursor));
        assert!(!first.gap && !independent.gap);
        assert_eq!(first.events.len(), 1);
        assert_eq!(independent.events.len(), 1);
        assert_eq!(first.cursor, independent.cursor);
        assert!(journal.since(Some(first.cursor)).events.is_empty());

        let replacement = ColumnJournal::new().since(Some(first.cursor));
        assert!(replacement.gap);
        assert!(replacement.events.is_empty());
    }

    #[test]
    fn column_journal_retention_boundary_requires_explicit_rebaseline() {
        let journal = ColumnJournal::new();
        let baseline = journal.since(None).cursor;
        for _ in 0..COLUMN_CHANGE_LOG_CAPACITY {
            journal.record(column_event());
        }
        let retained = journal.since(Some(baseline));
        assert!(!retained.gap);
        assert_eq!(retained.events.len(), COLUMN_CHANGE_LOG_CAPACITY);

        journal.record(column_event());
        let expired = journal.since(Some(baseline));
        assert!(expired.gap);
        assert!(expired.events.is_empty());
        let recent = journal.since(Some(retained.cursor));
        assert!(!recent.gap);
        assert_eq!(recent.events.len(), 1);
        assert!(!journal.since(Some(expired.cursor)).gap);
    }

    #[test]
    fn column_journal_concurrent_writers_preserve_every_revision() {
        let journal = ColumnJournal::new();
        let baseline = journal.since(None).cursor;
        std::thread::scope(|scope| {
            for _ in 0..4 {
                scope.spawn(|| {
                    for _ in 0..1000 {
                        journal.record(column_event());
                    }
                });
            }
        });
        let delta = journal.since(Some(baseline));
        assert!(!delta.gap);
        assert_eq!(delta.events.len(), 4000);
        assert_eq!(delta.cursor.revision, baseline.revision + 4000);
    }

    #[test]
    fn column_journal_revision_exhaustion_panics_without_wrapping_or_lock_leak() {
        let journal = ColumnJournal::new();
        journal.since(None);
        journal.lock().revision = u64::MAX - 1;
        journal.record(column_event());
        let last = journal.since(None).cursor;
        assert_eq!(last.revision, u64::MAX);

        let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            journal.record(column_event());
        }));
        assert!(failed.is_err());
        // Exhaustion is fail-stop, not rollover recovery. The RAII lock must
        // nevertheless be released during unwinding, and no cursor may wrap.
        let state = journal.lock();
        assert_eq!(state.revision, u64::MAX);
        assert_eq!(state.events.len(), 1);
    }

    #[test]
    fn empty_archetype_exists_at_id_zero() {
        let archetypes = Archetypes::new();
        assert_eq!(archetypes.len(), 1);
        let empty = archetypes.get(ArchetypeId::EMPTY).unwrap();
        assert!(empty.components().is_empty());
        assert!(empty.is_empty());
    }

    #[test]
    fn component_posting_removes_sparse_ids_across_blocks_and_keeps_order() {
        let mut posting = ArchetypePosting::default();
        for id in 0..513 {
            posting.push(ArchetypeId(id));
        }

        let removed = [
            ArchetypeId(2),
            ArchetypeId(127),
            ArchetypeId(128),
            ArchetypeId(256),
            ArchetypeId(511),
        ];
        posting.remove_sorted(&removed);
        let actual: Vec<_> = posting.iter().collect();
        let expected: Vec<_> = (0..513)
            .map(ArchetypeId)
            .filter(|id| removed.binary_search(id).is_err())
            .collect();
        assert_eq!(actual, expected);
        assert_eq!(posting.len(), 513 - removed.len());
        assert!(actual.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(removed.iter().all(|&id| !posting.contains(id)));

        // The append-only registry contract still holds after holes are
        // compacted from earlier blocks.
        posting.push(ArchetypeId(700));
        assert!(posting
            .iter()
            .last()
            .is_some_and(|id| id == ArchetypeId(700)));
    }

    #[test]
    fn component_posting_truncates_multiple_suffix_blocks() {
        let mut posting = ArchetypePosting::default();
        for id in 0..385 {
            posting.push(ArchetypeId(id));
        }
        let removed: Vec<_> = (260..385).map(ArchetypeId).collect();
        posting.remove_sorted(&removed);
        assert_eq!(
            posting.iter().collect::<Vec<_>>(),
            (0..260).map(ArchetypeId).collect::<Vec<_>>()
        );
        assert_eq!(posting.len(), 260);
    }

    #[test]
    fn component_posting_coalesces_adjacent_sparse_blocks() {
        let mut posting = ArchetypePosting::default();
        for id in 0..256 {
            posting.push(ArchetypeId(id));
        }
        let removed: Vec<_> = (0..64).chain(192..256).map(ArchetypeId).collect();
        posting.remove_sorted(&removed);
        assert!(posting.blocks.is_empty());
        assert_eq!(posting.small.len(), ArchetypePosting::BLOCK_CAPACITY);
        assert_eq!(
            posting.iter().collect::<Vec<_>>(),
            (64..192).map(ArchetypeId).collect::<Vec<_>>()
        );
    }

    #[test]
    fn get_or_create_dedups_by_set() {
        let mut comps = Components::new();
        let a = comps.register::<A>();
        let b = comps.register::<B>();
        let mut archetypes = Archetypes::new();

        let set_ab = ComponentSet::from_ids([a, b]);
        let set_ba = ComponentSet::from_ids([b, a]);
        let id1 = archetypes.get_or_create(&set_ab, &comps);
        let id2 = archetypes.get_or_create(&set_ba, &comps);
        assert_eq!(id1, id2);
        assert_eq!(archetypes.len(), 2); // empty + {A,B}

        let set_a = ComponentSet::from_ids([a]);
        let id3 = archetypes.get_or_create(&set_a, &comps);
        assert_ne!(id1, id3);
        assert_eq!(archetypes.len(), 3);
        assert!(archetypes.get(id3).unwrap().contains(a));
        assert!(!archetypes.get(id3).unwrap().contains(b));
        assert_eq!(
            archetypes
                .with_component(a)
                .unwrap()
                .iter()
                .collect::<Vec<_>>(),
            [id1, id3]
        );
        assert_eq!(
            archetypes
                .with_component(b)
                .unwrap()
                .iter()
                .collect::<Vec<_>>(),
            [id1]
        );
        assert!(archetypes.with_component(ComponentId::new(99)).is_none());
    }
}
