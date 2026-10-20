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
use alloc::collections::VecDeque;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::cell::UnsafeCell;
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
#[derive(Clone, Copy, Debug)]
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

struct ColumnJournalState {
    revision: u64,
    events: VecDeque<ColumnChange>,
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
            state.events.pop_front();
        }
        state.events.push_back(event);
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
        ColumnChanges {
            cursor: current,
            gap: offset.is_none(),
            events: offset.map_or_else(Vec::new, |start| {
                state.events.range(start..).copied().collect()
            }),
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
    column_journal: Arc<ColumnJournal>,
}

impl Archetypes {
    /// Create a registry pre-populated with the empty archetype at
    /// [`ArchetypeId::EMPTY`].
    pub fn new() -> Self {
        let mut this = Self {
            archetypes: Vec::new(),
            by_key: HashMap::default(),
            column_journal: Arc::new(ColumnJournal::new()),
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

    /// Discover writes/structural edits in O(retained events since cursor), not
    /// O(archetypes/chunks). Repeated same-tick writes advance this cursor too.
    /// Payload access still requires the usual ECS read/write scheduling barrier.
    pub fn column_changes_since(&self, cursor: Option<ColumnChangeCursor>) -> ColumnChanges {
        self.column_journal.since(cursor)
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
            archetypes, by_key, ..
        } = self;
        by_key.retain(|_key, id| {
            let arch = &archetypes[id.index() as usize];
            !(arch.is_empty()
                && arch
                    .shared_bindings()
                    .iter()
                    .any(|&(c, s, _)| c == cid && s == sid))
        });
    }

    fn insert(
        &mut self,
        set: ComponentSet,
        bindings: &[SharedBinding],
        pairs: &[RelationPair],
        components: &Components,
    ) -> ArchetypeId {
        let id = ArchetypeId(self.archetypes.len() as u32);
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
    }
}
