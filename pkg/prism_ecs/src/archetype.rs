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
use alloc::collections::{BinaryHeap, VecDeque};
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::cell::UnsafeCell;
use core::cmp::Reverse;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::collections::HashMap;
use crate::component::{ComponentId, ComponentSet, Components};
use crate::entity::Entity;
use crate::storage::{SharedValue, SharedValueId, Table};

/// Maximum retained write/structural events. Lagging readers explicitly rebuild.
pub const COLUMN_CHANGE_LOG_CAPACITY: usize = 8192;
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
pub(crate) struct FilteredColumnChanges {
    pub cursor: ColumnChangeCursor,
    pub gap: bool,
    pub events: Vec<ColumnChange>,
    /// Relevant event records selected/materialized for merge, including
    /// structural records. Binary-search probes over revision metadata and
    /// unrelated writes are not counted.
    pub records_scanned: usize,
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

        // Every source is already strictly ordered by revision, and sources
        // are disjoint (one event has either one component or is structural).
        // Merge the selected streams directly instead of materializing all
        // revisions and sorting them. This keeps the filtered read at
        // O(k log m), where k is the selected event count and m is the number
        // of selected sources, rather than O(k log k). The output is copied
        // while the journal lock is held so eviction cannot invalidate an
        // event's offset before it is materialized.
        let mut heap = BinaryHeap::with_capacity(sources.len());
        for (source, (revisions, position)) in sources.iter().enumerate() {
            if let Some(&revision) = revisions.get(*position) {
                heap.push(Reverse((revision, source)));
            }
        }
        let mut events = Vec::with_capacity(records_scanned);
        while let Some(Reverse((revision, source))) = heap.pop() {
            // `first_retained` is the revision immediately before the first
            // retained event; convert event revision to zero-based deque
            // position.
            let offset = (revision - first_retained - 1) as usize;
            events.push(state.events[offset]);

            let (revisions, position) = &mut sources[source];
            *position += 1;
            if let Some(&next) = revisions.get(*position) {
                heap.push(Reverse((next, source)));
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
/// pairs (design §11).
pub struct Archetype {
    id: ArchetypeId,
    components: ComponentSet,
    shared: Box<[SharedBinding]>,
    pairs: Box<[RelationPair]>,
    table: Table,
    evicted: bool,
}

impl Archetype {
    /// This archetype's id.
    #[inline]
    pub fn id(&self) -> ArchetypeId {
        self.id
    }

    /// The exact set of component ids every entity here has.
    #[inline]
    pub fn components(&self) -> &ComponentSet {
        &self.components
    }

    /// Whether entities here have the table/sparse component `id`. Shared
    /// bindings are *not* reported here (they are archetype-level batch keys,
    /// not table columns); query [`Archetype::shared_binding`] for those.
    #[inline]
    pub fn contains(&self, id: ComponentId) -> bool {
        self.components.contains(id)
    }

    /// The interned [`SharedValueId`] bound to shared component `id` in this
    /// archetype, or `None` if this archetype has no binding for `id` (design
    /// §6). This is the precise, per-archetype membership test for a shared
    /// component.
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
    #[inline]
    pub fn shared_arc(&self, id: ComponentId) -> Option<&Arc<dyn SharedValue>> {
        self.shared
            .binary_search_by_key(&id, |&(cid, _, _)| cid)
            .ok()
            .map(|i| &self.shared[i].2)
    }

    /// All shared-component bindings on this archetype, sorted by
    /// [`ComponentId`] (design §6).
    #[inline]
    pub fn shared_bindings(&self) -> &[SharedBinding] {
        &self.shared
    }

    /// Fragmenting relation pairs shared by every entity in this archetype.
    /// Sorted and duplicate-free; pairs are identity, not component columns.
    #[inline]
    pub fn relation_pairs(&self) -> &[RelationPair] {
        &self.pairs
    }

    /// Number of entities in this archetype.
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
    #[inline]
    pub(crate) fn is_evicted(&self) -> bool {
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

/// The per-world registry of archetypes.
pub struct Archetypes {
    archetypes: Vec<Archetype>,
    by_key: HashMap<ArchetypeKey, ArchetypeId>,
    /// Inverted table-column index. Each posting list is ordered by
    /// `ArchetypeId`, matching the registry's stable insertion order. Entries
    /// are removed when a shared-value archetype is permanently evicted.
    /// Dirty-query fallback uses it to avoid testing every matched archetype
    /// for every accessed table column.
    by_component: HashMap<ComponentId, Vec<ArchetypeId>>,
    column_journal: Arc<ColumnJournal>,
    eviction_generation: u64,
}

impl Archetypes {
    /// Create a registry pre-populated with the empty archetype at
    /// [`ArchetypeId::EMPTY`].
    pub fn new() -> Self {
        let mut this = Self {
            archetypes: Vec::new(),
            by_key: HashMap::default(),
            by_component: HashMap::default(),
            column_journal: Arc::new(ColumnJournal::new()),
            eviction_generation: 0,
        };
        let empty = ComponentSet::default();
        let id = this.insert(empty, &[], &[], &Components::new());
        debug_assert_eq!(id, ArchetypeId::EMPTY);
        this
    }

    /// Number of archetypes.
    #[inline]
    pub fn len(&self) -> usize {
        self.archetypes.len()
    }

    /// Whether there are no archetypes (never true after [`Archetypes::new`]).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.archetypes.is_empty()
    }

    /// Borrow an archetype by id.
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

    /// All archetypes, in id order.
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
    pub(crate) fn with_component(&self, component: ComponentId) -> &[ArchetypeId] {
        self.by_component
            .get(&component)
            .map_or(&[], |archetypes| archetypes.as_slice())
    }

    /// Revision incremented when one or more permanently evicted shared-value
    /// archetypes are removed from lookup indexes.
    #[inline]
    pub(crate) fn eviction_generation(&self) -> u64 {
        self.eviction_generation
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
    /// existing [`ArchetypeId`]s (dense vec indices) stay stable; only the
    /// `by_key` entry is removed, so the tombstone can never again be returned by
    /// a lookup and a fresh archetype is minted on demand instead. This is safe
    /// because `refcount(cid, sid) == 0` implies no live entity binds `(cid, sid)`,
    /// so every archetype binding it is empty and unreferenced by any
    /// [`EntityLocation`](crate::entity::EntityLocation).
    pub fn evict_shared_binding(&mut self, cid: ComponentId, sid: SharedValueId) {
        let Self {
            archetypes,
            by_key,
            by_component,
            eviction_generation,
            ..
        } = self;
        let mut evicted: Vec<ArchetypeId> = by_key
            .iter()
            .filter_map(|(_key, id)| {
                let arch = &archetypes[id.index() as usize];
                (arch.is_empty()
                    && arch
                        .shared_bindings()
                        .iter()
                        .any(|&(c, s, _)| c == cid && s == sid))
                .then_some(*id)
            })
            .collect();

        if evicted.is_empty() {
            return;
        }

        // This revision is only a cache-compaction hint. Wrapping can at worst
        // leave empty stable-id tombstones in a QueryState snapshot until its
        // next structural reset; it must never interrupt shared-value release.
        *eviction_generation = eviction_generation.wrapping_add(1);

        // Keep stable ArchetypeIds in `archetypes`, but reclaim their inverted
        // index entries. Batch affected components so each posting is filtered
        // once even when one shared value had many table-set variants.
        evicted.sort_unstable();
        for &id in &evicted {
            archetypes[id.index() as usize].evicted = true;
        }
        by_key.retain(|_key, id| evicted.binary_search(id).is_err());

        let mut affected_components = Vec::new();
        for &id in &evicted {
            affected_components.extend_from_slice(archetypes[id.index() as usize].components.ids());
        }
        affected_components.sort_unstable();
        affected_components.dedup();
        for component in affected_components {
            let remove_entry = if let Some(postings) = by_component.get_mut(&component) {
                postings.retain(|id| evicted.binary_search(id).is_err());
                postings.is_empty()
            } else {
                false
            };
            if remove_entry {
                by_component.remove(&component);
            }
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
        assert_eq!(filtered.events, expected);
        assert_eq!(filtered.events.len(), 4);
        assert_eq!(filtered.records_scanned, 4);
        // The legacy full API's scan count is defined as the full returned
        // delta length; it deliberately has no added public stats field.
        assert_eq!(full.events.len(), 5);
        assert_eq!(
            filtered
                .events
                .iter()
                .map(|event| event.chunk)
                .collect::<Vec<_>>(),
            vec![Some(0), Some(2), Some(0), Some(3)]
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
        assert_eq!(filtered.events, expected);
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
        assert_eq!(filtered.events, vec![component_event(selected, 7)]);
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
        assert_eq!(archetypes.with_component(a), &[id1, id3]);
        assert_eq!(archetypes.with_component(b), &[id1]);
        assert!(archetypes.with_component(ComponentId::new(99)).is_empty());
    }
}
