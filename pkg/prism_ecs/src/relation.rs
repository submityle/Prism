//! Relation core data structures (design §11, §23.2, §23.3).
//!
//! A **relation** is a flecs-style typed edge between entities, expressed as a
//! *pair* `(Relation, Target)`. The *relation kind* is identified by a
//! [`ComponentId`] — relation kinds are zero-sized marker component types
//! registered through the ordinary [`Components`](crate::component::Components)
//! registry — and the *target* is either a concrete [`Entity`] or the wildcard
//! `*` used by pattern queries.
//!
//! This module is the **data / index / registry layer only**. It owns:
//!
//! - the compact encoded [`PairKey`] used for pair matching,
//! - the per-kind [`RelationKind`] metadata (fragmenting / transitive /
//!   exclusive flags plus [`CleanupPolicy`] cleanup rules),
//! - the [`Relations`] registry that maps a relation kind to its metadata,
//! - the global [`RelationIndex`] (generation-aware bidirectional adjacency)
//!   used for non-fragmenting relations and every reverse / cascade query, and
//! - pure [`CascadePlan`] planning for cascade-delete.
//!
//! Nothing here mutates a [`World`](crate::world::World); wiring the index into
//! `World` spawn/despawn and query paths is a separate step. Every operation is
//! implemented against the standard library's `alloc` collections and contains
//! no `unsafe`.

use alloc::collections::{BTreeSet, VecDeque};
use alloc::vec::Vec;
use core::sync::atomic::{AtomicUsize, Ordering};

use crate::collections::HashMap;
use crate::component::ComponentId;
use crate::entity::Entity;

/// Sentinel target index that encodes the wildcard `*` inside a [`PairKey`] or
/// a [`TargetId`]. No live entity can occupy `u32::MAX` as a slot index, so the
/// value is unambiguous.
pub const WILDCARD_TARGET: u32 = u32::MAX;

/// A compact identifier for a relation *kind*.
///
/// This is simply the dense index of the marker [`ComponentId`] that names the
/// relation, re-wrapped so the packed [`PairKey`] layout is explicit and
/// self-documenting. Convert freely to and from [`ComponentId`].
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct RelationId(u32);

impl RelationId {
    /// Construct a [`RelationId`] from its raw dense index.
    #[inline]
    pub const fn new(index: u32) -> Self {
        Self(index)
    }

    /// The raw dense index backing this id.
    #[inline]
    pub const fn index(self) -> u32 {
        self.0
    }

    /// Reinterpret a relation-kind [`ComponentId`] as a [`RelationId`].
    #[inline]
    pub const fn from_component(id: ComponentId) -> Self {
        Self(id.index())
    }

    /// Recover the relation-kind [`ComponentId`] this id refers to.
    #[inline]
    pub const fn component_id(self) -> ComponentId {
        ComponentId::new(self.0)
    }
}

/// A compact identifier for a pair *target*.
///
/// Holds either a live entity's slot index or the [`WILDCARD_TARGET`] sentinel.
/// Only the slot index is stored; the full generational [`Entity`] handle is
/// retained separately by the [`RelationIndex`], whose internal adjacency keys
/// also include the generation so raw staged edits cannot alias recycled slots.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct TargetId(u32);

impl TargetId {
    /// The wildcard target `*`.
    pub const WILDCARD: Self = Self(WILDCARD_TARGET);

    /// Construct a concrete target from a raw entity slot index.
    #[inline]
    pub const fn entity_index(index: u32) -> Self {
        Self(index)
    }

    /// Construct a concrete target from a live [`Entity`] handle.
    #[inline]
    pub const fn from_entity(entity: Entity) -> Self {
        Self(entity.index())
    }

    /// The raw slot index (or [`WILDCARD_TARGET`]) backing this target.
    #[inline]
    pub const fn index(self) -> u32 {
        self.0
    }

    /// Whether this target is the wildcard `*`.
    #[inline]
    pub const fn is_wildcard(self) -> bool {
        self.0 == WILDCARD_TARGET
    }
}

/// A relation pair target: a concrete [`Entity`] or the wildcard `*`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum RelationTarget {
    /// A concrete target entity.
    Entity(Entity),
    /// The wildcard `*`, matching every target of a relation in pattern
    /// queries such as `query_pair(ChildOf, *)` (design §11).
    Wildcard,
}

impl RelationTarget {
    /// The concrete entity, if this is not the wildcard.
    #[inline]
    pub const fn entity(self) -> Option<Entity> {
        match self {
            RelationTarget::Entity(e) => Some(e),
            RelationTarget::Wildcard => None,
        }
    }

    /// Whether this target is the wildcard `*`.
    #[inline]
    pub const fn is_wildcard(self) -> bool {
        matches!(self, RelationTarget::Wildcard)
    }

    /// The compact [`TargetId`] encoding of this target.
    #[inline]
    pub const fn target_id(self) -> TargetId {
        match self {
            RelationTarget::Entity(e) => TargetId::from_entity(e),
            RelationTarget::Wildcard => TargetId::WILDCARD,
        }
    }
}

/// A fully-specified relation pair `(Relation, Target)` (design §11).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Pair {
    /// The relation kind, identified by its marker [`ComponentId`].
    pub relation: ComponentId,
    /// The pair target (concrete entity or wildcard).
    pub target: RelationTarget,
}

impl Pair {
    /// Construct a pair targeting a concrete entity.
    #[inline]
    pub const fn new(relation: ComponentId, target: Entity) -> Self {
        Self {
            relation,
            target: RelationTarget::Entity(target),
        }
    }

    /// Construct a wildcard pair `(relation, *)` for pattern queries.
    #[inline]
    pub const fn wildcard(relation: ComponentId) -> Self {
        Self {
            relation,
            target: RelationTarget::Wildcard,
        }
    }

    /// Encode this pair into its compact [`PairKey`].
    #[inline]
    pub const fn key(self) -> PairKey {
        PairKey::encode(
            RelationId::from_component(self.relation),
            self.target.target_id(),
        )
    }
}

/// A compact `u64` encoding of a `(relation, target-index)` pair, packing the
/// relation's dense id into the high 32 bits and the target slot index (or
/// [`WILDCARD_TARGET`]) into the low 32 bits.
///
/// This is the fast hash-map key used by the [`RelationIndex`]. Because only a
/// 32-bit slot index is retained, [`PairKey`] alone cannot distinguish entity
/// generations; the index pairs each key with the full anchor [`Entity`] so no
/// information is lost in practice.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct PairKey(u64);

impl PairKey {
    /// Pack a relation id and a target id into a key.
    #[inline]
    pub const fn encode(relation: RelationId, target: TargetId) -> Self {
        Self(((relation.index() as u64) << 32) | (target.index() as u64))
    }

    /// Reconstruct a key from its raw bits.
    #[inline]
    pub const fn from_raw(bits: u64) -> Self {
        Self(bits)
    }

    /// The raw `u64` bits of this key.
    #[inline]
    pub const fn raw(self) -> u64 {
        self.0
    }

    /// The relation id encoded in the high 32 bits.
    #[inline]
    pub const fn relation(self) -> RelationId {
        RelationId::new((self.0 >> 32) as u32)
    }

    /// The target id encoded in the low 32 bits.
    #[inline]
    pub const fn target(self) -> TargetId {
        TargetId::entity_index(self.0 as u32)
    }

    /// Whether this key's target is the wildcard `*`.
    #[inline]
    pub const fn is_wildcard(self) -> bool {
        self.target().is_wildcard()
    }

    /// Whether this (concrete) key is matched by `query`.
    ///
    /// Relations must be identical. The target matches when `query` is a
    /// wildcard, or when both targets refer to the same slot index. This is the
    /// primitive behind `query_pair(relation, *)` (design §11).
    #[inline]
    pub const fn matches(self, query: PairKey) -> bool {
        self.relation().index() == query.relation().index()
            && (query.is_wildcard() || self.target().index() == query.target().index())
    }
}

/// Cleanup policy applied when the counterpart of a relation edge is destroyed
/// (design §23.2). Used for both the `OnDelete` and `OnDeleteTarget` slots of a
/// [`RelationKind`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum CleanupPolicy {
    /// Remove only the dangling pair from the holder (the default).
    #[default]
    Remove,
    /// Cascade-destroy the holder (e.g. `ChildOf`: deleting a parent deletes
    /// its children).
    Delete,
    /// Reject deletion of a referenced target before applying the cascade,
    /// including in release builds.
    Panic,
}

/// Per-kind relation metadata (design §11 / §23.2 / §23.3).
///
/// Built with [`RelationKind::new`] and the fluent `with_*` setters:
///
/// ```ignore
/// let child_of = RelationKind::new()
///     .with_fragmenting(true)
///     .with_exclusive(true)
///     .with_on_delete_target(CleanupPolicy::Delete);
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct RelationKind {
    /// Low-cardinality relation whose pairs are folded into the archetype (the
    /// pair *fragments* the archetype graph, design §11). High-cardinality
    /// relations set this to `false` and live only in the bypass
    /// [`RelationIndex`] to avoid archetype explosion (design §23.3).
    pub fragmenting: bool,
    /// Whether the relation is transitive, i.e. `a R b` and `b R c` imply
    /// `a R c` under the transitive closure (e.g. `LocatedIn`, design §11).
    pub transitive: bool,
    /// Whether the relation is exclusive: each source holds at most one target,
    /// and adding a new target evicts the previous one (e.g. `ChildOf`,
    /// `DockedTo`, design §23.3).
    pub exclusive: bool,
    /// Policy applied to existing edges when the relation *kind* itself is
    /// removed from a source holder (design §23.2).
    pub on_delete: CleanupPolicy,
    /// Policy applied to a holder when the *target* it points at is destroyed
    /// (design §23.2). This is the policy that drives cascade-delete planning.
    pub on_delete_target: CleanupPolicy,
}

impl RelationKind {
    /// Create relation metadata with all defaults: non-fragmenting,
    /// non-transitive, non-exclusive, and [`CleanupPolicy::Remove`] on both
    /// deletion slots.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the [`fragmenting`](Self::fragmenting) flag.
    #[inline]
    #[must_use]
    pub const fn with_fragmenting(mut self, fragmenting: bool) -> Self {
        self.fragmenting = fragmenting;
        self
    }

    /// Set the [`transitive`](Self::transitive) flag.
    #[inline]
    #[must_use]
    pub const fn with_transitive(mut self, transitive: bool) -> Self {
        self.transitive = transitive;
        self
    }

    /// Set the [`exclusive`](Self::exclusive) flag.
    #[inline]
    #[must_use]
    pub const fn with_exclusive(mut self, exclusive: bool) -> Self {
        self.exclusive = exclusive;
        self
    }

    /// Set the [`on_delete`](Self::on_delete) policy.
    #[inline]
    #[must_use]
    pub const fn with_on_delete(mut self, policy: CleanupPolicy) -> Self {
        self.on_delete = policy;
        self
    }

    /// Set the [`on_delete_target`](Self::on_delete_target) policy.
    #[inline]
    #[must_use]
    pub const fn with_on_delete_target(mut self, policy: CleanupPolicy) -> Self {
        self.on_delete_target = policy;
        self
    }
}

/// A statically-typed relation kind whose per-kind [`RelationKind`] metadata is
/// fixed at compile time (design §11 / §23.2 / §23.3).
///
/// Every relation kind is first a [`Component`](crate::component::Component):
/// its [`ComponentId`] keys both the kind metadata in [`Relations`] and the
/// edges in the bypass [`RelationIndex`]. Implementing `Relation` additionally
/// pins the kind's fragmenting / transitive / exclusive flags and
/// [`CleanupPolicy`] rules as the associated constant [`KIND`](Relation::KIND),
/// so [`World::register_relation_type`](crate::world::World::register_relation_type)
/// can install them without the caller restating the metadata at every call
/// site.
///
/// Prefer `#[derive(Relation)]` (re-exported from the crate prelude) over a
/// hand-written impl; the derive maps `#[relation(...)]` attributes onto the
/// fields of [`RelationKind`].
///
/// ```ignore
/// use prism_ecs::prelude::*;
///
/// #[derive(Component, Relation)]
/// #[relation(fragmenting, exclusive, on_delete_target = "Delete")]
/// struct ChildOf;
/// ```
pub trait Relation: crate::component::Component {
    /// The compile-time relation metadata installed by
    /// [`World::register_relation_type`](crate::world::World::register_relation_type).
    const KIND: RelationKind;
}

/// A single directed relation edge `source --relation--> target`, produced by
/// index removal and cascade planning.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct CascadeEdge {
    /// The relation kind of the edge.
    pub relation: ComponentId,
    /// The holder (source) of the edge.
    pub source: Entity,
    /// The target the edge points at.
    pub target: Entity,
}

/// A pure, side-effect-free plan describing how a despawn should propagate over
/// relations (design §23.2). The caller (`World`) applies it; this module never
/// mutates world state while planning.
#[derive(Clone, Debug, Default)]
pub struct CascadePlan {
    /// Edges that must be unlinked from the index while their holders survive
    /// (the [`CleanupPolicy::Remove`] outcome, plus the outgoing edges of every
    /// entity being deleted).
    pub removals: Vec<CascadeEdge>,
    /// Additional holder entities that must be recursively despawned because a
    /// target they point at is being deleted under [`CleanupPolicy::Delete`].
    /// The original root entity is *not* included; the caller already despawns
    /// it.
    pub deletions: Vec<Entity>,
    /// Edges that violate a [`CleanupPolicy::Panic`] policy. The caller must
    /// reject the deletion before applying any removals or deletions.
    pub panics: Vec<CascadeEdge>,
}

/// Internal adjacency value: the full anchor entity keyed by an adjacency key plus
/// its ordered list of neighbours. For forward edges the anchor is the source
/// and the neighbours are its targets; for reverse edges the anchor is the
/// target and the neighbours are its sources.
type Adjacency = (Entity, Vec<Entity>);

/// The packed key deliberately omits generation for wildcard matching. The
/// live adjacency must retain it: staged edits can contain both a stale handle
/// and its replacement before the world filters invalid endpoints.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct AdjacencyKey {
    pair: PairKey,
    generation: u32,
}

impl AdjacencyKey {
    fn relation(self) -> RelationId {
        self.pair.relation()
    }
}

/// An opaque cache key for one version of a relation index.
///
/// Compare the entire key, not just its revision. Clones and new indexes have
/// distinct identities, and revision exhaustion rotates the identity rather
/// than wrapping to a previously observable key. Keys are process-local, not
/// deterministic simulation state or serialized identifiers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RelationIndexVersion {
    identity: usize,
    revision: u64,
}

fn allocate_index_identity(counter: &AtomicUsize) -> usize {
    // Never wrap or reuse an identity, including after a caught panic.
    counter
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
        .expect("relation index identity counter exhausted")
}

impl RelationIndexVersion {
    fn fresh() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(1);
        Self {
            identity: allocate_index_identity(&NEXT),
            revision: 0,
        }
    }

    /// Process-unique identity of this index's current revision sequence.
    pub fn identity(self) -> usize {
        self.identity
    }

    /// Monotonic mutation revision within this identity.
    pub fn revision(self) -> u64 {
        self.revision
    }
}

/// Maximum retained source changes per relation kind. Unrelated kinds cannot
/// evict one another's history. Memory is bounded by registered/used kinds
/// times this capacity, not by the number of mutations or readers.
pub const RELATION_CHANGE_CAPACITY: usize = 1024;

/// Reader position in an index's source-change history. Cloning an index or
/// exhausting its history sequence invalidates old cursors; consumers must
/// rebuild when [`RelationIndex::changed_sources_since`] returns `None`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RelationChangeCursor(RelationIndexVersion);

/// Conservative source delta: reread these sources' authoritative targets.
#[derive(Debug)]
pub struct RelationChanges {
    /// Full generational handles; repeated mutations may repeat a source.
    pub sources: Vec<Entity>,
    /// Retained entries visited (only this kind, strictly after the cursor).
    pub entries_examined: usize,
    /// Binary-search comparisons used to seek past retained older entries.
    /// Separate from delivered entries so clean-reader discovery is measurable.
    pub lookup_steps: usize,
}

#[derive(Clone, Debug, Default)]
struct KindChanges {
    floor: u64,
    entries: VecDeque<(u64, Entity)>,
}

#[derive(Clone, Debug)]
struct RelationJournal {
    cursor: RelationChangeCursor,
    kinds: HashMap<ComponentId, KindChanges>,
}

impl Default for RelationJournal {
    fn default() -> Self {
        Self {
            cursor: RelationChangeCursor(RelationIndexVersion::fresh()),
            kinds: HashMap::default(),
        }
    }
}

impl RelationJournal {
    fn begin(&mut self) {
        if let Some(next) = self.cursor.0.revision.checked_add(1) {
            self.cursor.0.revision = next;
        } else {
            *self = Self::default();
        }
    }

    fn record(&mut self, relation: ComponentId, source: Entity) {
        let kind = self.kinds.entry(relation).or_default();
        if kind.entries.len() == RELATION_CHANGE_CAPACITY {
            let (revision, _) = kind.entries.pop_front().expect("full journal");
            kind.floor = revision;
        }
        kind.entries.push_back((self.cursor.0.revision, source));
    }
}

/// The bidirectional index for all relations (design §11).
///
/// Maintains two mirrored adjacency maps so both directions are O(1) to query:
///
/// - **forward** `(relation, source) -> [targets]`, and
/// - **reverse** `(relation, target) -> [sources]`.
///
/// The reverse map powers reverse queries and cascade-delete without scanning
/// every entity. High-cardinality relations route here instead of fragmenting
/// the archetype graph (design §23.3).
#[derive(Debug)]
pub struct RelationIndex {
    /// `(relation, source) -> (source, [target...])`.
    forward: HashMap<AdjacencyKey, Adjacency>,
    /// `(relation, target) -> (target, [source...])`.
    reverse: HashMap<AdjacencyKey, Adjacency>,
    // Enumerate only an entity's incident kinds during cascade/despawn. Full
    // generational keys keep recycled slots independent, just like adjacency.
    outgoing: HashMap<Entity, BTreeSet<ComponentId>>,
    incoming: HashMap<Entity, BTreeSet<ComponentId>>,
    version: RelationIndexVersion,
    journal: RelationJournal,
}

impl Default for RelationIndex {
    fn default() -> Self {
        Self {
            forward: HashMap::default(),
            reverse: HashMap::default(),
            outgoing: HashMap::default(),
            incoming: HashMap::default(),
            version: RelationIndexVersion::fresh(),
            journal: RelationJournal::default(),
        }
    }
}

impl Clone for RelationIndex {
    fn clone(&self) -> Self {
        Self {
            forward: self.forward.clone(),
            reverse: self.reverse.clone(),
            outgoing: self.outgoing.clone(),
            incoming: self.incoming.clone(),
            version: RelationIndexVersion::fresh(),
            journal: RelationJournal::default(),
        }
    }
}

/// Build the forward key for `(relation, source)`.
#[inline]
fn forward_key(relation: ComponentId, source: Entity) -> AdjacencyKey {
    AdjacencyKey {
        pair: PairKey::encode(
            RelationId::from_component(relation),
            TargetId::from_entity(source),
        ),
        generation: source.generation(),
    }
}

/// Build the reverse key for `(relation, target)`.
#[inline]
fn reverse_key(relation: ComponentId, target: Entity) -> AdjacencyKey {
    forward_key(relation, target)
}

/// Push `edge` into `edges` only if an identical edge is not already present,
/// keeping the removal/panic lists free of duplicates.
#[inline]
fn push_unique(
    edges: &mut Vec<CascadeEdge>,
    seen: &mut BTreeSet<(ComponentId, Entity, Entity)>,
    edge: CascadeEdge,
) {
    if seen.insert((edge.relation, edge.source, edge.target)) {
        edges.push(edge);
    }
}

impl RelationIndex {
    /// Create an empty index.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Cache key covering all edge mutations, including raw registry edits.
    ///
    /// Moving an index preserves its key; cloning or replacing it with a new
    /// index does not. A no-op insertion/removal/clear preserves the key.
    #[inline]
    pub fn version(&self) -> RelationIndexVersion {
        self.version
    }

    /// Current position, captured before invoking any mutation callbacks.
    pub fn change_cursor(&self) -> RelationChangeCursor {
        self.journal.cursor
    }

    /// Source changes of one kind since `cursor`, or `None` on a history gap,
    /// clone/replacement or future cursor. Empty deltas do not visit any edges.
    /// Lookup seeks into the bounded kind log before visiting recent entries.
    pub fn changed_sources_since(
        &self,
        relation: ComponentId,
        cursor: RelationChangeCursor,
    ) -> Option<RelationChanges> {
        let current = self.journal.cursor.0;
        if cursor.0.identity != current.identity || cursor.0.revision > current.revision {
            return None;
        }
        let Some(kind) = self.journal.kinds.get(&relation) else {
            return Some(RelationChanges {
                sources: Vec::new(),
                entries_examined: 0,
                lookup_steps: 0,
            });
        };
        if cursor.0.revision < kind.floor {
            return None;
        }
        let mut lookup_steps = 0;
        let start = kind.entries.partition_point(|(rev, _)| {
            lookup_steps += 1;
            *rev <= cursor.0.revision
        });
        let sources = kind
            .entries
            .iter()
            .skip(start)
            .map(|(_, source)| *source)
            .collect();
        Some(RelationChanges {
            sources,
            entries_examined: kind.entries.len() - start,
            lookup_steps,
        })
    }

    /// Raw staged edits may replace/clone the entire registry. Reconcile the
    /// final adjacency against the live graph and inherit its reader lineage,
    /// rather than treating a fresh clone identity as an all-source mutation.
    /// This bulk-publication path is O(old + new buckets + retained history).
    pub(crate) fn inherit_changes_from(&mut self, previous: &Self) {
        self.journal = previous.journal.clone();
        let mut changed = Vec::new();
        for (key, (source, targets)) in &previous.forward {
            if self.targets(key.relation().component_id(), *source) != targets.as_slice() {
                changed.push((key.relation().component_id(), *source));
            }
        }
        for (key, (source, _)) in &self.forward {
            if !previous.forward.contains_key(key) {
                changed.push((key.relation().component_id(), *source));
            }
        }
        if !changed.is_empty() {
            self.journal.begin();
            for (relation, source) in changed {
                self.journal.record(relation, source);
            }
        }
    }

    fn mark_changed(&mut self) {
        // Obtain a fresh identity before touching adjacency on exhaustion, so
        // even a caught allocation-counter panic leaves the old graph intact.
        self.version = match self.version.revision.checked_add(1) {
            Some(revision) => RelationIndexVersion {
                revision,
                ..self.version
            },
            None => RelationIndexVersion::fresh(),
        };
        self.journal.begin();
    }

    /// Remove all edges, retaining relation-kind metadata in the registry.
    /// Clearing an already empty index is a no-op.
    pub fn clear(&mut self) {
        if !self.is_empty() {
            self.mark_changed();
            for (key, (source, _)) in &self.forward {
                self.journal.record(key.relation().component_id(), *source);
            }
            self.forward.clear();
            self.reverse.clear();
            self.outgoing.clear();
            self.incoming.clear();
        }
    }

    /// Whether the index holds no edges at all.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.forward.is_empty() && self.reverse.is_empty()
    }

    /// Number of distinct `(relation, source)` adjacency buckets held.
    #[inline]
    pub fn source_bucket_count(&self) -> usize {
        self.forward.len()
    }

    /// Total number of directed edges held in the forward index.
    ///
    /// Equal to the sum of every source bucket's target count. This is the
    /// `(relation, source) -> target` edge total, mirroring what
    /// [`iter_edges`](Self::iter_edges) yields, and is used by the relation
    /// graph diagnostics (design §16.6).
    #[inline]
    pub fn edge_count(&self) -> usize {
        self.forward
            .values()
            .map(|(_, targets)| targets.len())
            .sum()
    }

    /// Iterate every directed edge as `(relation, source, target)`.
    ///
    /// Walks the forward adjacency, expanding each `(relation, source)` bucket
    /// into one tuple per target. Iteration order follows the underlying hash
    /// map and is therefore unspecified; callers that need determinism must
    /// sort. Read-only; powers the relation graph diagnostics (design §16.6).
    pub fn iter_edges(&self) -> impl Iterator<Item = (ComponentId, Entity, Entity)> + '_ {
        self.forward.iter().flat_map(|(key, (source, targets))| {
            let relation = key.relation().component_id();
            let source = *source;
            targets
                .iter()
                .map(move |&target| (relation, source, target))
        })
    }

    /// Add the edge `source --relation--> target`.
    ///
    /// When `exclusive` is set the source keeps at most one target: any prior
    /// targets are evicted first and the previous target (if any) is returned
    /// so the caller can run cleanup for it. For non-exclusive relations a
    /// duplicate edge is a no-op. Returns `Some(old_target)` only when an
    /// exclusive add evicted a *different* previous target.
    pub fn add(
        &mut self,
        relation: ComponentId,
        source: Entity,
        target: Entity,
        exclusive: bool,
    ) -> Option<Entity> {
        let fkey = forward_key(relation, source);
        let mut evicted = None;

        let existing = self.targets(relation, source);
        if (exclusive && existing == [target]) || (!exclusive && existing.contains(&target)) {
            return None;
        }
        self.mark_changed();
        self.journal.record(relation, source);
        if exclusive && let Some((_, existing)) = self.forward.get(&fkey) {
            let olds: Vec<Entity> = existing.clone();
            evicted = olds.first().copied();
            for old in olds {
                self.remove_from_reverse(relation, source, old);
            }
            self.forward.remove(&fkey);
        }

        self.forward
            .entry(fkey)
            .or_insert_with(|| (source, Vec::new()))
            .1
            .push(target);
        self.reverse
            .entry(reverse_key(relation, target))
            .or_insert_with(|| (target, Vec::new()))
            .1
            .push(source);
        self.outgoing.entry(source).or_default().insert(relation);
        self.incoming.entry(target).or_default().insert(relation);

        evicted
    }

    /// Remove the edge `source --relation--> target`. Returns whether it
    /// existed.
    pub fn remove(&mut self, relation: ComponentId, source: Entity, target: Entity) -> bool {
        if !self.targets(relation, source).contains(&target) {
            return false;
        }
        self.mark_changed();
        self.journal.record(relation, source);
        let removed = self.remove_from_forward(relation, source, target);
        if removed {
            self.remove_from_reverse(relation, source, target);
        }
        removed
    }

    /// Remove every edge touching `entity` as either source or target, cleaning
    /// up both maps. Returns the removed edges so the caller can drive cascade
    /// processing.
    pub fn remove_all_for_entity(&mut self, entity: Entity) -> Vec<CascadeEdge> {
        let mut affected = Vec::new();

        // Edges where `entity` is the source.
        let fkeys: Vec<AdjacencyKey> = self
            .outgoing
            .get(&entity)
            .into_iter()
            .flatten()
            .map(|&relation| forward_key(relation, entity))
            .collect();
        let rkeys: Vec<AdjacencyKey> = self
            .incoming
            .get(&entity)
            .into_iter()
            .flatten()
            .map(|&relation| reverse_key(relation, entity))
            .collect();
        if !fkeys.is_empty() || !rkeys.is_empty() {
            self.mark_changed();
        }
        self.outgoing.remove(&entity);
        self.incoming.remove(&entity);
        for fkey in fkeys {
            if let Some((_, targets)) = self.forward.remove(&fkey) {
                let relation = fkey.relation().component_id();
                self.journal.record(relation, entity);
                for target in targets {
                    affected.push(CascadeEdge {
                        relation,
                        source: entity,
                        target,
                    });
                    self.remove_from_reverse(relation, entity, target);
                }
            }
        }

        // Edges where `entity` is the target.
        for rkey in rkeys {
            if let Some((_, sources)) = self.reverse.remove(&rkey) {
                let relation = rkey.relation().component_id();
                for source in sources {
                    self.journal.record(relation, source);
                    affected.push(CascadeEdge {
                        relation,
                        source,
                        target: entity,
                    });
                    self.remove_from_forward(relation, source, entity);
                }
            }
        }

        affected
    }

    /// The targets of `source` under `relation`, or an empty slice.
    #[inline]
    pub fn targets(&self, relation: ComponentId, source: Entity) -> &[Entity] {
        match self.forward.get(&forward_key(relation, source)) {
            Some((anchor, targets)) if *anchor == source => targets.as_slice(),
            _ => &[],
        }
    }

    /// The sources that point at `target` under `relation`, or an empty slice.
    #[inline]
    pub fn sources(&self, relation: ComponentId, target: Entity) -> &[Entity] {
        match self.reverse.get(&reverse_key(relation, target)) {
            Some((anchor, sources)) if *anchor == target => sources.as_slice(),
            _ => &[],
        }
    }

    /// The outgoing edges of `entity` as `(relation, target)` pairs.
    /// Visits only this entity's incident kinds and edges, not global buckets.
    pub fn outgoing_edges(&self, entity: Entity) -> Vec<(ComponentId, Entity)> {
        let mut out = Vec::new();
        for &relation in self.outgoing.get(&entity).into_iter().flatten() {
            for &target in self.targets(relation, entity) {
                out.push((relation, target));
            }
        }
        out
    }

    /// The incoming edges of `entity` as `(relation, source)` pairs.
    /// Visits only this entity's incident kinds and edges, not global buckets.
    pub fn incoming_edges(&self, entity: Entity) -> Vec<(ComponentId, Entity)> {
        let mut out = Vec::new();
        for &relation in self.incoming.get(&entity).into_iter().flatten() {
            for &source in self.sources(relation, entity) {
                out.push((relation, source));
            }
        }
        out
    }

    /// Resolve a pair query into concrete `(source, target)` edges.
    ///
    /// A [`RelationTarget::Wildcard`] yields every edge of `relation`; a
    /// concrete target yields all sources pointing at it (design §11).
    pub fn query_pair(
        &self,
        relation: ComponentId,
        target: RelationTarget,
    ) -> Vec<(Entity, Entity)> {
        let mut out = Vec::new();
        match target {
            RelationTarget::Wildcard => {
                let rel = relation.index();
                for (key, (anchor, targets)) in self.forward.iter() {
                    if key.relation().index() == rel {
                        for &t in targets {
                            out.push((*anchor, t));
                        }
                    }
                }
            }
            RelationTarget::Entity(t) => {
                for &source in self.sources(relation, t) {
                    out.push((source, t));
                }
            }
        }
        out
    }

    /// Compute the transitive closure of targets reachable from `source` under
    /// `relation` (design §11), excluding `source` itself.
    ///
    /// Uses an iterative depth-first walk with a visited set, so cyclic graphs
    /// terminate. Only meaningful when the relation kind is
    /// [`transitive`](RelationKind::transitive); the computation itself is
    /// policy-agnostic.
    pub fn transitive_targets(&self, relation: ComponentId, source: Entity) -> Vec<Entity> {
        let mut out = Vec::new();
        let mut visited: Vec<Entity> = Vec::new();
        let mut stack: Vec<Entity> = Vec::new();

        visited.push(source);
        stack.push(source);

        while let Some(current) = stack.pop() {
            for &target in self.targets(relation, current) {
                if !visited.contains(&target) {
                    visited.push(target);
                    out.push(target);
                    stack.push(target);
                }
            }
        }

        out
    }

    /// Remove `target` from the forward adjacency of `(relation, source)`,
    /// pruning the bucket when it empties. Returns whether an edge was removed.
    fn remove_from_forward(
        &mut self,
        relation: ComponentId,
        source: Entity,
        target: Entity,
    ) -> bool {
        let fkey = forward_key(relation, source);
        let Some((_, targets)) = self.forward.get_mut(&fkey) else {
            return false;
        };
        let Some(pos) = targets.iter().position(|&t| t == target) else {
            return false;
        };
        targets.remove(pos);
        if targets.is_empty() {
            self.forward.remove(&fkey);
            if let Some(kinds) = self.outgoing.get_mut(&source) {
                kinds.remove(&relation);
                if kinds.is_empty() {
                    self.outgoing.remove(&source);
                }
            }
        }
        true
    }

    /// Remove `source` from the reverse adjacency of `(relation, target)`,
    /// pruning the bucket when it empties. Returns whether an edge was removed.
    fn remove_from_reverse(
        &mut self,
        relation: ComponentId,
        source: Entity,
        target: Entity,
    ) -> bool {
        let rkey = reverse_key(relation, target);
        let Some((_, sources)) = self.reverse.get_mut(&rkey) else {
            return false;
        };
        let Some(pos) = sources.iter().position(|&s| s == source) else {
            return false;
        };
        sources.remove(pos);
        if sources.is_empty() {
            self.reverse.remove(&rkey);
            if let Some(kinds) = self.incoming.get_mut(&target) {
                kinds.remove(&relation);
                if kinds.is_empty() {
                    self.incoming.remove(&target);
                }
            }
        }
        true
    }
}

/// The per-world relation registry: relation-kind metadata plus the global
/// bypass [`RelationIndex`].
///
/// High-level [`add`](Self::add) / [`remove`](Self::remove) consult the
/// registered [`RelationKind`] (e.g. honouring exclusivity) before delegating
/// to the index, and [`plan_cascade`](Self::plan_cascade) applies the per-kind
/// [`CleanupPolicy`] to produce a [`CascadePlan`].
#[derive(Clone, Debug, Default)]
pub struct Relations {
    /// Relation-kind metadata keyed by the marker [`ComponentId`].
    kinds: HashMap<ComponentId, RelationKind>,
    /// The global non-fragmenting adjacency index.
    index: RelationIndex,
}

impl Relations {
    /// Create an empty registry.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register (or overwrite) the metadata for a relation kind.
    #[inline]
    pub fn register(&mut self, relation: ComponentId, kind: RelationKind) {
        self.kinds.insert(relation, kind);
        if kind.exclusive {
            self.normalize_exclusive(relation);
        }
    }

    /// Canonical fragmenting identity for a live source. Iterate registered
    /// kinds, not all graph edges, so unrelated holders do not increase the
    /// work of a single source's structural change.
    pub fn fragmenting_pairs(&self, source: Entity) -> Vec<(ComponentId, Entity)> {
        let mut pairs = Vec::new();
        for (&relation, kind) in &self.kinds {
            if kind.fragmenting {
                pairs.extend(
                    self.index
                        .targets(relation, source)
                        .iter()
                        .map(|&t| (relation, t)),
                );
            }
        }
        pairs.sort_unstable();
        pairs.dedup();
        pairs
    }

    /// Bulk/raw index edits may bypass exclusive insertion. Keep the most
    /// recently inserted target when enabling or reconciling exclusivity.
    pub(crate) fn normalize_exclusive(&mut self, relation: ComponentId) {
        let sources: Vec<_> = self
            .index
            .forward
            .iter()
            .filter(|(key, (_, targets))| {
                key.relation().component_id() == relation && targets.len() > 1
            })
            .map(|(_, (source, targets))| (*source, *targets.last().unwrap()))
            .collect();
        for (source, target) in sources {
            self.index.add(relation, source, target, true);
        }
    }

    /// Look up the metadata for a relation kind, if registered.
    #[inline]
    pub fn kind(&self, relation: ComponentId) -> Option<&RelationKind> {
        self.kinds.get(&relation)
    }

    /// Whether a relation kind has been registered.
    #[inline]
    pub fn is_registered(&self, relation: ComponentId) -> bool {
        self.kinds.contains_key(&relation)
    }

    /// Whether a relation kind is exclusive. Unregistered kinds are treated as
    /// non-exclusive.
    #[inline]
    pub fn is_exclusive(&self, relation: ComponentId) -> bool {
        self.kinds.get(&relation).is_some_and(|k| k.exclusive)
    }

    /// Whether a relation kind is transitive. Unregistered kinds are treated as
    /// non-transitive.
    #[inline]
    pub fn is_transitive(&self, relation: ComponentId) -> bool {
        self.kinds.get(&relation).is_some_and(|k| k.transitive)
    }

    /// Number of registered relation kinds.
    #[inline]
    pub fn kind_count(&self) -> usize {
        self.kinds.len()
    }

    /// Iterate every registered relation kind as `(relation, &metadata)`.
    ///
    /// Iteration order follows the underlying hash map and is unspecified;
    /// callers that need determinism must sort. Read-only; powers the relation
    /// graph diagnostics (design §16.6).
    pub fn iter_kinds(&self) -> impl Iterator<Item = (ComponentId, &RelationKind)> + '_ {
        self.kinds.iter().map(|(&id, kind)| (id, kind))
    }

    /// Shared access to the underlying index.
    #[inline]
    pub fn index(&self) -> &RelationIndex {
        &self.index
    }

    /// Mutable access to the underlying index.
    #[inline]
    pub fn index_mut(&mut self) -> &mut RelationIndex {
        &mut self.index
    }

    /// Add the edge `source --relation--> target`, honouring the relation
    /// kind's [`exclusive`](RelationKind::exclusive) flag. Returns the evicted
    /// previous target for an exclusive relation, as per
    /// [`RelationIndex::add`].
    #[inline]
    pub fn add(&mut self, relation: ComponentId, source: Entity, target: Entity) -> Option<Entity> {
        let exclusive = self.is_exclusive(relation);
        self.index.add(relation, source, target, exclusive)
    }

    /// Remove the edge `source --relation--> target`. Returns whether it
    /// existed.
    #[inline]
    pub fn remove(&mut self, relation: ComponentId, source: Entity, target: Entity) -> bool {
        self.index.remove(relation, source, target)
    }

    /// Plan how despawning `root` should cascade over the relation graph,
    /// without mutating anything (design §23.2).
    ///
    /// Starting from `root`, every entity scheduled for deletion contributes
    /// its outgoing edges to [`CascadePlan::removals`]. For each incoming edge,
    /// the holder is handled per the relation kind's
    /// [`on_delete_target`](RelationKind::on_delete_target) policy:
    ///
    /// - [`CleanupPolicy::Remove`] records a removal only,
    /// - [`CleanupPolicy::Delete`] schedules the holder for recursive deletion,
    /// - [`CleanupPolicy::Panic`] records a violation.
    ///
    /// A visited set makes the walk cycle-safe, and `root` is never listed in
    /// [`CascadePlan::deletions`].
    pub fn plan_cascade(&self, root: Entity) -> CascadePlan {
        let mut plan = CascadePlan::default();
        let mut visited = BTreeSet::new();
        let mut removals_seen = BTreeSet::new();
        let mut panics_seen = BTreeSet::new();
        let mut queue: Vec<Entity> = Vec::new();

        visited.insert(root);
        queue.push(root);

        while let Some(entity) = queue.pop() {
            // Outgoing edges vanish with the deleted holder.
            for (relation, target) in self.index.outgoing_edges(entity) {
                push_unique(
                    &mut plan.removals,
                    &mut removals_seen,
                    CascadeEdge {
                        relation,
                        source: entity,
                        target,
                    },
                );
            }

            // Incoming edges are governed by the target-deletion policy.
            for (relation, source) in self.index.incoming_edges(entity) {
                let policy = self
                    .kinds
                    .get(&relation)
                    .map(|k| k.on_delete_target)
                    .unwrap_or_default();
                let edge = CascadeEdge {
                    relation,
                    source,
                    target: entity,
                };
                match policy {
                    CleanupPolicy::Remove => {
                        push_unique(&mut plan.removals, &mut removals_seen, edge)
                    }
                    CleanupPolicy::Delete => {
                        push_unique(&mut plan.removals, &mut removals_seen, edge);
                        if visited.insert(source) {
                            queue.push(source);
                            plan.deletions.push(source);
                        }
                    }
                    CleanupPolicy::Panic => push_unique(&mut plan.panics, &mut panics_seen, edge),
                }
            }
        }

        plan
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::num::NonZeroU32;

    /// Build a deterministic [`Entity`] for tests via its stable bit layout.
    fn ent(index: u32, generation: u32) -> Entity {
        let bits = ((generation as u64) << 32) | (index as u64);
        Entity::from_bits(bits).expect("non-zero generation")
    }

    fn rel(index: u32) -> ComponentId {
        ComponentId::new(index)
    }

    fn assert_endpoint_indexes(idx: &RelationIndex) {
        let mut outgoing = HashMap::<Entity, BTreeSet<ComponentId>>::default();
        let mut incoming = HashMap::<Entity, BTreeSet<ComponentId>>::default();
        for (relation, source, target) in idx.iter_edges() {
            outgoing.entry(source).or_default().insert(relation);
            incoming.entry(target).or_default().insert(relation);
            assert!(idx.sources(relation, target).contains(&source));
        }
        assert_eq!(idx.outgoing, outgoing);
        assert_eq!(idx.incoming, incoming);
        for (&entity, kinds) in &outgoing {
            let expected: BTreeSet<_> = kinds
                .iter()
                .flat_map(|&r| {
                    idx.targets(r, entity)
                        .iter()
                        .map(move |&target| (r, target))
                })
                .collect();
            assert_eq!(
                idx.outgoing_edges(entity)
                    .into_iter()
                    .collect::<BTreeSet<_>>(),
                expected
            );
        }
        for (&entity, kinds) in &incoming {
            let expected: BTreeSet<_> = kinds
                .iter()
                .flat_map(|&r| {
                    idx.sources(r, entity)
                        .iter()
                        .map(move |&source| (r, source))
                })
                .collect();
            assert_eq!(
                idx.incoming_edges(entity)
                    .into_iter()
                    .collect::<BTreeSet<_>>(),
                expected
            );
        }
        assert_eq!(
            idx.forward.values().map(|(_, v)| v.len()).sum::<usize>(),
            idx.reverse.values().map(|(_, v)| v.len()).sum::<usize>()
        );
    }

    #[test]
    fn endpoint_indexes_match_adjacency_after_mixed_mutations() {
        let mut idx = RelationIndex::new();
        let mut expected = BTreeSet::new();
        let mut random = 0xa70d_98bf_13cc_49e1_u64;
        let mut next = || {
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            random
        };
        for step in 0..1024 {
            // Adjacent test handles share a slot but not its generation.
            let source = next() as u32 % 24;
            let source = ent(source / 2, source % 2 + 1);
            let target = next() as u32 % 24;
            let target = ent(target / 2, target % 2 + 1);
            let r = rel(next() as u32 % 4);
            match next() % 8 {
                0..=2 => {
                    idx.add(r, source, target, false);
                    expected.insert((r, source, target));
                }
                3 => {
                    idx.add(r, source, target, true);
                    expected.retain(|&(kind, s, _)| kind != r || s != source);
                    expected.insert((r, source, target));
                }
                4 => {
                    assert_eq!(
                        idx.remove(r, source, target),
                        expected.remove(&(r, source, target))
                    );
                }
                5 => {
                    let removed = idx.remove_all_for_entity(source);
                    let before = expected.len();
                    expected.retain(|&(_, s, t)| s != source && t != source);
                    assert_eq!(removed.len(), before - expected.len());
                }
                6 => idx = idx.clone(),
                _ if step % 31 == 0 => {
                    idx.clear();
                    expected.clear();
                }
                _ => {}
            }
            assert_eq!(
                idx.iter_edges().collect::<BTreeSet<_>>(),
                expected,
                "step {step}"
            );
            assert_endpoint_indexes(&idx);
        }
        idx.clear();
        assert_endpoint_indexes(&idx);
        assert!(idx.outgoing.is_empty() && idx.incoming.is_empty());
    }

    #[test]
    fn endpoint_lookup_and_removal_ignore_large_unrelated_graph() {
        for size in [64, 8192] {
            let mut idx = RelationIndex::new();
            let (r, a, b, reused) = (rel(1), ent(0, 1), ent(1, 1), ent(0, 2));
            for i in 10..size + 10 {
                idx.add(rel(2), ent(i, 1), ent(i + size, 1), true);
            }
            idx.add(r, a, a, false);
            idx.add(r, a, b, false);
            idx.add(r, b, a, false);
            idx.add(rel(3), reused, b, false);
            assert_eq!(idx.outgoing[&a].len(), 1);
            assert_eq!(idx.incoming[&a].len(), 1);
            assert_eq!(idx.outgoing_edges(a), [(r, a), (r, b)]);
            assert_eq!(idx.incoming_edges(a), [(r, a), (r, b)]);
            let cursor = idx.change_cursor();
            assert_eq!(idx.remove_all_for_entity(a).len(), 3);
            assert_eq!(idx.edge_count(), size as usize + 1);
            assert_eq!(idx.targets(rel(3), reused), [b]);
            assert!(idx.outgoing_edges(a).is_empty());
            assert!(idx.incoming_edges(a).is_empty());
            assert_eq!(
                idx.changed_sources_since(r, cursor)
                    .unwrap()
                    .sources
                    .into_iter()
                    .collect::<BTreeSet<_>>(),
                [a, b].into_iter().collect()
            );
            assert_endpoint_indexes(&idx);
        }
    }

    #[test]
    fn source_journal_exact_capacity_boundary_and_independent_readers() {
        let mut idx = RelationIndex::new();
        let (r, target) = (rel(1), ent(0, 1));
        let slow = idx.change_cursor();
        idx.add(r, ent(1, 1), target, true);
        let fast = idx.change_cursor();
        for i in 2..=RELATION_CHANGE_CAPACITY {
            idx.add(r, ent(i as u32, 1), target, true);
        }
        let delta = idx.changed_sources_since(r, slow).unwrap();
        assert_eq!(delta.entries_examined, RELATION_CHANGE_CAPACITY);
        assert_eq!(delta.sources.first(), Some(&ent(1, 1)));
        assert!(delta.lookup_steps <= 12);
        let current = idx.change_cursor();
        assert!(idx
            .changed_sources_since(r, current)
            .unwrap()
            .sources
            .is_empty());
        // Reading does not consume another reader's retained history.
        assert_eq!(
            idx.changed_sources_since(r, fast).unwrap().sources.len(),
            RELATION_CHANGE_CAPACITY - 1
        );
        idx.add(r, ent(RELATION_CHANGE_CAPACITY as u32 + 1, 1), target, true);
        assert!(idx.changed_sources_since(r, slow).is_none());
        assert_eq!(
            idx.changed_sources_since(r, fast).unwrap().sources.len(),
            RELATION_CHANGE_CAPACITY
        );
        assert_eq!(
            idx.changed_sources_since(r, current).unwrap().sources,
            [ent(RELATION_CHANGE_CAPACITY as u32 + 1, 1)]
        );
    }

    #[test]
    fn source_journal_is_bounded_per_kind_and_noops_do_not_advance() {
        let mut idx = RelationIndex::new();
        let (r, unrelated, a, b) = (rel(1), rel(2), ent(1, 1), ent(2, 1));
        let initial = idx.change_cursor();
        idx.clear();
        idx.remove(r, a, b);
        assert_eq!(idx.change_cursor(), initial);
        idx.add(r, a, b, true);
        let baseline = idx.change_cursor();
        idx.add(r, a, b, true);
        assert_eq!(idx.change_cursor(), baseline);
        for i in 0..RELATION_CHANGE_CAPACITY + 7 {
            idx.add(unrelated, a, ent(10 + i as u32, 1), true);
        }
        assert!(idx.changed_sources_since(unrelated, baseline).is_none());
        let delta = idx.changed_sources_since(r, baseline).unwrap();
        assert_eq!(delta.entries_examined, 0);
        assert!(delta.sources.is_empty());
        assert_eq!(
            idx.journal.kinds[&unrelated].entries.len(),
            RELATION_CHANGE_CAPACITY
        );
        assert_eq!(idx.changed_sources_since(r, initial).unwrap().sources, [a]);
        idx.clear();
        assert_eq!(idx.changed_sources_since(r, baseline).unwrap().sources, [a]);
        assert!(idx
            .changed_sources_since(r, idx.change_cursor())
            .unwrap()
            .sources
            .is_empty());
    }

    #[test]
    fn source_journal_covers_incoming_removals_and_full_generations() {
        let mut idx = RelationIndex::new();
        let (r, a, b, next_a) = (rel(1), ent(1, 1), ent(2, 1), ent(1, 2));
        idx.add(r, a, a, false);
        idx.add(r, a, b, false);
        idx.add(r, b, a, false);
        idx.add(r, next_a, a, false);
        let cursor = idx.change_cursor();
        idx.remove_all_for_entity(a);
        let sources = idx.changed_sources_since(r, cursor).unwrap().sources;
        assert_eq!(
            sources.into_iter().collect::<BTreeSet<_>>(),
            [a, b, next_a].into_iter().collect()
        );
        assert!(idx.is_empty());
    }

    #[test]
    fn source_journal_raw_publication_inherits_only_net_source_changes() {
        let mut idx = RelationIndex::new();
        let (r, a, b, c) = (rel(1), ent(1, 1), ent(2, 1), ent(3, 1));
        idx.add(r, a, b, true);
        let cursor = idx.change_cursor();
        let mut staged = idx.clone();
        assert!(staged.changed_sources_since(r, cursor).is_none());
        staged.inherit_changes_from(&idx);
        assert_eq!(staged.change_cursor(), cursor);
        staged.add(r, a, c, true);
        staged.inherit_changes_from(&idx);
        assert_eq!(
            staged.changed_sources_since(r, cursor).unwrap().sources,
            [a]
        );
        let mut empty = RelationIndex::new();
        empty.inherit_changes_from(&staged);
        assert_eq!(
            empty
                .changed_sources_since(r, staged.change_cursor())
                .unwrap()
                .sources,
            [a]
        );
    }

    #[test]
    fn source_journal_batch_gap_and_sequence_exhaustion_are_explicit() {
        let mut idx = RelationIndex::new();
        let (r, target) = (rel(1), ent(0, 1));
        for i in 1..=RELATION_CHANGE_CAPACITY + 1 {
            idx.add(r, ent(i as u32, 1), target, true);
        }
        let before_clear = idx.change_cursor();
        idx.clear(); // One batch with more sources than the log can hold.
        assert!(idx.changed_sources_since(r, before_clear).is_none());
        assert!(idx
            .changed_sources_since(r, idx.change_cursor())
            .unwrap()
            .sources
            .is_empty());
        idx.journal.cursor.0.revision = u64::MAX;
        let exhausted = idx.change_cursor();
        idx.add(r, ent(1, 1), target, true);
        assert!(idx.changed_sources_since(r, exhausted).is_none());
        let current = idx.change_cursor();
        idx.add(r, ent(2, 1), target, true);
        assert_eq!(
            idx.changed_sources_since(r, current).unwrap().sources,
            [ent(2, 1)]
        );
    }

    #[test]
    fn index_version_tracks_effective_mutations_and_noops() {
        let mut idx = RelationIndex::new();
        let r = rel(1);
        let (a, b, c) = (ent(1, 1), ent(2, 1), ent(3, 1));
        let initial = idx.version();
        idx.clear();
        assert!(!idx.remove(r, a, b));
        assert!(idx.remove_all_for_entity(a).is_empty());
        assert_eq!(idx.version(), initial);
        idx.add(r, a, b, false);
        let first = idx.version();
        assert_eq!(first.identity(), initial.identity());
        assert_eq!(first.revision(), initial.revision() + 1);
        idx.add(r, a, b, false);
        idx.add(r, a, b, true);
        assert_eq!(idx.version(), first);
        idx.add(r, a, c, false);
        assert_eq!(idx.version().revision(), first.revision() + 1);
        // Normalizing a raw non-exclusive adjacency is a real mutation even
        // when the chosen exclusive target was already present.
        idx.add(r, a, b, true);
        assert_eq!(idx.version().revision(), first.revision() + 2);
        assert_eq!(idx.targets(r, a), &[b]);
        assert!(idx.sources(r, c).is_empty());
        assert_eq!(idx.add(r, a, c, true), Some(b));
        assert_eq!(idx.version().revision(), first.revision() + 3);
        assert!(idx.sources(r, b).is_empty());
        assert!(idx.remove(r, a, c));
        let empty = idx.version();
        assert_eq!(empty.revision(), first.revision() + 4);
        idx.clear();
        assert_eq!(idx.version(), empty);
    }

    #[test]
    fn index_version_remove_all_is_one_mutation_and_generation_safe() {
        let mut idx = RelationIndex::new();
        let r = rel(1);
        let (a, b, next_a) = (ent(1, 1), ent(2, 1), ent(1, 2));
        for (source, target) in [(a, a), (a, b), (b, a), (next_a, b)] {
            idx.add(r, source, target, false);
        }
        let before = idx.version();
        let removed = idx.remove_all_for_entity(a);
        assert_eq!(removed.len(), 3); // Self-loop is reported only once.
        assert_eq!(idx.version().revision(), before.revision() + 1);
        assert_eq!(idx.edge_count(), 1);
        assert_eq!(idx.targets(r, next_a), &[b]);
        assert_eq!(idx.sources(r, b), &[next_a]);
        let before = idx.version();
        assert!(idx.remove_all_for_entity(a).is_empty());
        assert_eq!(idx.version(), before);
        idx.clear();
        assert_eq!(idx.version().revision(), before.revision() + 1);
        assert!(idx.is_empty());
    }

    #[test]
    fn index_versions_distinguish_clones_and_raw_replacements() {
        let mut registry = Relations::new();
        let (r, a, b) = (rel(1), ent(1, 1), ent(2, 1));
        let empty = registry.index().version();
        registry.index_mut().add(r, a, b, true);
        assert_ne!(registry.index().version(), empty);
        let original = registry.index().version();
        let mut cloned = registry.clone();
        assert_ne!(cloned.index().version().identity(), original.identity());
        assert_eq!(cloned.index().targets(r, a), &[b]);
        let cloned_version = cloned.index().version();
        cloned.index_mut().clear();
        assert_ne!(cloned.index().version(), cloned_version);
        assert_eq!(registry.index().targets(r, a), &[b]);
        cloned.clone_from(&registry);
        assert_ne!(cloned.index().version(), original);
        assert_ne!(
            cloned.index().version().identity(),
            cloned_version.identity()
        );
        *registry.index_mut() = RelationIndex::new();
        assert_ne!(registry.index().version().identity(), original.identity());
        assert!(registry.index().is_empty());
    }

    #[test]
    fn index_revision_exhaustion_rotates_identity_for_every_mutation() {
        for mutation in 0..4 {
            let mut idx = RelationIndex::new();
            let (r, a, b, c) = (rel(1), ent(1, 1), ent(2, 1), ent(3, 1));
            idx.add(r, a, b, true);
            idx.version.revision = u64::MAX;
            let exhausted = idx.version();
            idx.add(r, a, b, true);
            assert!(!idx.remove(r, a, c));
            assert!(idx.remove_all_for_entity(c).is_empty());
            assert_eq!(idx.version(), exhausted);
            match mutation {
                0 => {
                    idx.add(r, a, c, true);
                }
                1 => {
                    assert!(idx.remove(r, a, b));
                }
                2 => {
                    assert_eq!(idx.remove_all_for_entity(a).len(), 1);
                }
                _ => idx.clear(),
            }
            assert_ne!(idx.version().identity(), exhausted.identity());
            assert_eq!(idx.version().revision(), 0);
            let remaining = [c];
            assert_eq!(
                idx.targets(r, a),
                if mutation == 0 { &remaining[..] } else { &[] }
            );
            idx.add(r, b, a, true);
            assert_eq!(idx.version().revision(), 1);
        }
    }

    #[test]
    fn index_identity_exhaustion_never_wraps_even_after_caught_panics() {
        // Test an isolated allocator, never exhaust the process-global counter.
        let counter = AtomicUsize::new(usize::MAX - 1);
        assert_eq!(allocate_index_identity(&counter), usize::MAX - 1);
        for _ in 0..2 {
            assert!(std::panic::catch_unwind(|| allocate_index_identity(&counter)).is_err());
            assert_eq!(counter.load(Ordering::Relaxed), usize::MAX);
        }
    }

    #[test]
    fn pair_key_encode_decode_roundtrips() {
        let e = ent(7, 3);
        let pair = Pair::new(rel(5), e);
        let key = pair.key();
        assert_eq!(key.relation(), RelationId::new(5));
        assert_eq!(key.target().index(), 7);
        assert!(!key.is_wildcard());
        assert_eq!(PairKey::from_raw(key.raw()), key);

        let wild = Pair::wildcard(rel(5)).key();
        assert!(wild.is_wildcard());
        assert_eq!(wild.relation(), RelationId::new(5));
        assert_eq!(wild.target(), TargetId::WILDCARD);
    }

    #[test]
    fn pair_key_wildcard_matching() {
        let concrete = Pair::new(rel(2), ent(9, 1)).key();
        let same = Pair::new(rel(2), ent(9, 4)).key(); // same slot, differs only by gen
        let other = Pair::new(rel(2), ent(10, 1)).key();
        let wild = Pair::wildcard(rel(2)).key();
        let wild_other = Pair::wildcard(rel(3)).key();

        assert!(concrete.matches(wild));
        assert!(concrete.matches(same));
        assert!(!concrete.matches(other));
        assert!(!concrete.matches(wild_other));
    }

    #[test]
    fn add_remove_sources_targets() {
        let mut idx = RelationIndex::new();
        let r = rel(1);
        let (a, b, c) = (ent(1, 1), ent(2, 1), ent(3, 1));

        assert_eq!(idx.add(r, a, b, false), None);
        assert_eq!(idx.add(r, a, c, false), None);
        // Duplicate is a no-op.
        assert_eq!(idx.add(r, a, b, false), None);

        assert_eq!(idx.targets(r, a), &[b, c]);
        assert_eq!(idx.sources(r, b), &[a]);
        assert_eq!(idx.sources(r, c), &[a]);

        assert!(idx.remove(r, a, b));
        assert!(!idx.remove(r, a, b));
        assert_eq!(idx.targets(r, a), &[c]);
        assert_eq!(idx.sources(r, b), &[] as &[Entity]);

        assert!(idx.remove(r, a, c));
        assert!(idx.is_empty());
    }

    #[test]
    fn adjacency_keys_isolate_generations_in_both_directions() {
        let mut idx = RelationIndex::new();
        let r = rel(1);
        let (old_source, source) = (ent(1, 1), ent(1, 2));
        let (old_target, target) = (ent(2, 1), ent(2, 2));
        for s in [old_source, source] {
            for t in [old_target, target] {
                idx.add(r, s, t, false);
            }
        }
        assert_eq!(idx.edge_count(), 4);
        assert_eq!(idx.targets(r, source), &[old_target, target]);
        assert_eq!(idx.sources(r, target), &[old_source, source]);
        assert_eq!(idx.add(r, source, target, true), Some(old_target));
        assert_eq!(idx.targets(r, source), &[target]);
        assert_eq!(idx.targets(r, old_source), &[old_target, target]);
        assert_eq!(idx.sources(r, old_target), &[old_source]);
        idx.remove_all_for_entity(old_source);
        assert_eq!(idx.sources(r, target), &[source]);
        assert_eq!(idx.targets(r, source), &[target]);
        assert!(!idx.remove(r, old_source, target));
        assert!(!idx.remove(r, source, old_target));
        assert!(idx.remove(r, source, target));
        assert!(idx.is_empty());
    }

    #[test]
    fn exclusive_eviction() {
        let mut idx = RelationIndex::new();
        let r = rel(4);
        let (child, p1, p2) = (ent(1, 1), ent(2, 1), ent(3, 1));

        assert_eq!(idx.add(r, child, p1, true), None);
        assert_eq!(idx.targets(r, child), &[p1]);

        // Re-adding the same target evicts nothing.
        assert_eq!(idx.add(r, child, p1, true), None);

        // New target evicts the old one and reports it.
        assert_eq!(idx.add(r, child, p2, true), Some(p1));
        assert_eq!(idx.targets(r, child), &[p2]);
        assert_eq!(idx.sources(r, p1), &[] as &[Entity]);
        assert_eq!(idx.sources(r, p2), &[child]);
    }

    #[test]
    fn query_pair_wildcard_and_concrete() {
        let mut idx = RelationIndex::new();
        let r = rel(2);
        let (a, b, t1, t2) = (ent(1, 1), ent(2, 1), ent(5, 1), ent(6, 1));
        idx.add(r, a, t1, false);
        idx.add(r, a, t2, false);
        idx.add(r, b, t1, false);

        let mut wild = idx.query_pair(r, RelationTarget::Wildcard);
        wild.sort_by_key(|(s, t)| (s.index(), t.index()));
        assert_eq!(wild, alloc::vec![(a, t1), (a, t2), (b, t1)]);

        let mut concrete = idx.query_pair(r, RelationTarget::Entity(t1));
        concrete.sort_by_key(|(s, _)| s.index());
        assert_eq!(concrete, alloc::vec![(a, t1), (b, t1)]);
    }

    #[test]
    fn transitive_closure_with_cycle() {
        let mut idx = RelationIndex::new();
        let r = rel(3);
        let (a, b, c, d) = (ent(1, 1), ent(2, 1), ent(3, 1), ent(4, 1));
        // a -> b -> c -> d, plus a cycle d -> b.
        idx.add(r, a, b, false);
        idx.add(r, b, c, false);
        idx.add(r, c, d, false);
        idx.add(r, d, b, false);

        let mut closure = idx.transitive_targets(r, a);
        closure.sort_by_key(|e| e.index());
        assert_eq!(closure, alloc::vec![b, c, d]);
        // Source itself is excluded.
        assert!(!closure.contains(&a));

        // Pure cycle terminates.
        let mut idx2 = RelationIndex::new();
        idx2.add(r, a, b, false);
        idx2.add(r, b, a, false);
        let c2 = idx2.transitive_targets(r, a);
        assert_eq!(c2, alloc::vec![b]);
    }

    #[test]
    fn remove_all_for_entity_bidirectional() {
        let mut idx = RelationIndex::new();
        let r = rel(1);
        let (a, b, c) = (ent(1, 1), ent(2, 1), ent(3, 1));
        // a -> b (a is source), c -> a (a is target).
        idx.add(r, a, b, false);
        idx.add(r, c, a, false);

        let affected = idx.remove_all_for_entity(a);
        assert_eq!(affected.len(), 2);
        assert!(affected.contains(&CascadeEdge {
            relation: r,
            source: a,
            target: b
        }));
        assert!(affected.contains(&CascadeEdge {
            relation: r,
            source: c,
            target: a
        }));

        // Both maps are fully cleaned.
        assert!(idx.is_empty());
        assert_eq!(idx.targets(r, a), &[] as &[Entity]);
        assert_eq!(idx.sources(r, a), &[] as &[Entity]);
    }

    #[test]
    fn plan_cascade_remove_policy() {
        let mut relations = Relations::new();
        let r = rel(1);
        relations.register(r, RelationKind::new()); // default Remove
        let (parent, child) = (ent(1, 1), ent(2, 1));
        relations.add(r, child, parent); // child points at parent

        let plan = relations.plan_cascade(parent);
        assert!(plan.deletions.is_empty());
        assert!(plan.panics.is_empty());
        assert!(plan.removals.contains(&CascadeEdge {
            relation: r,
            source: child,
            target: parent
        }));
    }

    #[test]
    fn plan_cascade_delete_policy_recursive() {
        let mut relations = Relations::new();
        let r = rel(1);
        relations.register(
            r,
            RelationKind::new()
                .with_exclusive(true)
                .with_on_delete_target(CleanupPolicy::Delete),
        );
        let (root, child, grandchild) = (ent(1, 1), ent(2, 1), ent(3, 1));
        relations.add(r, child, root); // child ChildOf root
        relations.add(r, grandchild, child); // grandchild ChildOf child

        let plan = relations.plan_cascade(root);
        assert!(plan.panics.is_empty());
        assert!(plan.deletions.contains(&child));
        assert!(plan.deletions.contains(&grandchild));
        assert_eq!(plan.deletions.len(), 2);
    }

    #[test]
    fn plan_cascade_delete_policy_cycle_safe() {
        let mut relations = Relations::new();
        let r = rel(1);
        relations.register(
            r,
            RelationKind::new().with_on_delete_target(CleanupPolicy::Delete),
        );
        let (a, b) = (ent(1, 1), ent(2, 1));
        // Mutual: a -> b and b -> a under a Delete-on-target relation.
        relations.add(r, a, b);
        relations.add(r, b, a);

        let plan = relations.plan_cascade(a);
        // b is pulled in once; a (root) is never listed; termination guaranteed.
        assert_eq!(plan.deletions, alloc::vec![b]);
    }

    #[test]
    fn plan_cascade_deduplicates_diamond_and_cycle_without_reordering() {
        let mut relations = Relations::new();
        let r = rel(1);
        relations.register(
            r,
            RelationKind::new().with_on_delete_target(CleanupPolicy::Delete),
        );
        let (root, a, b, shared) = (ent(1, 1), ent(2, 1), ent(3, 1), ent(4, 1));
        for (source, target) in [
            (a, root),
            (b, root),
            (shared, a),
            (shared, b),
            (root, shared),
        ] {
            relations.add(r, source, target);
        }
        let plan = relations.plan_cascade(root);
        // Preserve first encounter / ordered adjacency, not set iteration order.
        assert_eq!(plan.deletions, alloc::vec![a, b, shared]);
        assert_eq!(
            plan.removals
                .iter()
                .map(|edge| (edge.source, edge.target))
                .collect::<Vec<_>>(),
            alloc::vec![
                (root, shared),
                (a, root),
                (b, root),
                (shared, b),
                (shared, a)
            ]
        );
        assert!(plan.panics.is_empty());
    }

    #[test]
    fn plan_cascade_panic_policy() {
        let mut relations = Relations::new();
        let r = rel(1);
        relations.register(
            r,
            RelationKind::new().with_on_delete_target(CleanupPolicy::Panic),
        );
        let (target, holder) = (ent(1, 1), ent(2, 1));
        relations.add(r, holder, target);

        let plan = relations.plan_cascade(target);
        assert!(plan.deletions.is_empty());
        assert!(plan.panics.contains(&CascadeEdge {
            relation: r,
            source: holder,
            target
        }));
    }

    #[test]
    fn relation_kind_builder() {
        let k = RelationKind::new()
            .with_fragmenting(true)
            .with_transitive(true)
            .with_exclusive(true)
            .with_on_delete(CleanupPolicy::Delete)
            .with_on_delete_target(CleanupPolicy::Panic);
        assert!(k.fragmenting && k.transitive && k.exclusive);
        assert_eq!(k.on_delete, CleanupPolicy::Delete);
        assert_eq!(k.on_delete_target, CleanupPolicy::Panic);
        assert_eq!(CleanupPolicy::default(), CleanupPolicy::Remove);

        // Needed so `NonZeroU32` import is exercised and the helper stays sound.
        assert!(NonZeroU32::new(1).is_some());
    }

    #[test]
    fn registry_lookup_and_flags() {
        let mut relations = Relations::new();
        let r = rel(9);
        assert!(!relations.is_registered(r));
        assert!(!relations.is_exclusive(r));
        assert!(!relations.is_transitive(r));
        relations.register(
            r,
            RelationKind::new()
                .with_exclusive(true)
                .with_transitive(true),
        );
        assert!(relations.is_registered(r));
        assert!(relations.is_exclusive(r));
        assert!(relations.is_transitive(r));
        assert_eq!(relations.kind(r).map(|k| k.exclusive), Some(true));
    }
}
