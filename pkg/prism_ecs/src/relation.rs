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
//! - the compact encoded [`PairKey`] used as a fast hash-map key,
//! - the per-kind [`RelationKind`] metadata (fragmenting / transitive /
//!   exclusive flags plus [`CleanupPolicy`] cleanup rules),
//! - the [`Relations`] registry that maps a relation kind to its metadata,
//! - the global non-fragmenting bypass [`RelationIndex`] (bidirectional
//!   adjacency) used for high-cardinality relations and reverse / cascade
//!   queries, and
//! - pure [`CascadePlan`] planning for cascade-delete.
//!
//! Nothing here mutates a [`World`](crate::world::World); wiring the index into
//! `World` spawn/despawn and query paths is a separate step. Every operation is
//! implemented against the standard library's `alloc` collections and contains
//! no `unsafe`.

use alloc::vec::Vec;

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
/// retained separately by the [`RelationIndex`] (the index only ever holds
/// edges between currently-live entities, so an index resolves unambiguously).
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
    /// Treat the dangling edge as a bug: a debug assertion / recorded violation
    /// that catches illegal deletions.
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
    /// Edges that violated a [`CleanupPolicy::Panic`] policy — a dangling
    /// deletion the caller may assert on in debug builds.
    pub panics: Vec<CascadeEdge>,
}

/// Internal adjacency value: the full anchor entity keyed by a [`PairKey`] plus
/// its ordered list of neighbours. For forward edges the anchor is the source
/// and the neighbours are its targets; for reverse edges the anchor is the
/// target and the neighbours are its sources.
type Adjacency = (Entity, Vec<Entity>);

/// The global non-fragmenting bypass index for relations (design §11).
///
/// Maintains two mirrored adjacency maps so both directions are O(1) to query:
///
/// - **forward** `(relation, source) -> [targets]`, and
/// - **reverse** `(relation, target) -> [sources]`.
///
/// The reverse map powers reverse queries and cascade-delete without scanning
/// every entity. High-cardinality relations route here instead of fragmenting
/// the archetype graph (design §23.3).
#[derive(Clone, Debug, Default)]
pub struct RelationIndex {
    /// `(relation, source) -> (source, [target...])`.
    forward: HashMap<PairKey, Adjacency>,
    /// `(relation, target) -> (target, [source...])`.
    reverse: HashMap<PairKey, Adjacency>,
}

/// Build the forward key for `(relation, source)`.
#[inline]
fn forward_key(relation: ComponentId, source: Entity) -> PairKey {
    PairKey::encode(
        RelationId::from_component(relation),
        TargetId::from_entity(source),
    )
}

/// Build the reverse key for `(relation, target)`.
#[inline]
fn reverse_key(relation: ComponentId, target: Entity) -> PairKey {
    PairKey::encode(
        RelationId::from_component(relation),
        TargetId::from_entity(target),
    )
}

/// Push `edge` into `edges` only if an identical edge is not already present,
/// keeping the removal/panic lists free of duplicates.
#[inline]
fn push_unique(edges: &mut Vec<CascadeEdge>, edge: CascadeEdge) {
    if !edges.contains(&edge) {
        edges.push(edge);
    }
}

impl RelationIndex {
    /// Create an empty index.
    #[inline]
    pub fn new() -> Self {
        Self::default()
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

        if exclusive {
            if let Some((_, existing)) = self.forward.get(&fkey) {
                // Already pointing exactly at `target`: nothing to do.
                if existing.len() == 1 && existing[0] == target {
                    return None;
                }
                let olds: Vec<Entity> = existing.clone();
                evicted = olds.first().copied();
                for old in olds {
                    self.remove_from_reverse(relation, source, old);
                }
                self.forward.remove(&fkey);
            }
        } else if self.targets(relation, source).contains(&target) {
            return None;
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

        evicted
    }

    /// Remove the edge `source --relation--> target`. Returns whether it
    /// existed.
    pub fn remove(&mut self, relation: ComponentId, source: Entity, target: Entity) -> bool {
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
        let fkeys: Vec<PairKey> = self
            .forward
            .iter()
            .filter(|(_, (anchor, _))| *anchor == entity)
            .map(|(key, _)| *key)
            .collect();
        for fkey in fkeys {
            if let Some((_, targets)) = self.forward.remove(&fkey) {
                let relation = fkey.relation().component_id();
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
        let rkeys: Vec<PairKey> = self
            .reverse
            .iter()
            .filter(|(_, (anchor, _))| *anchor == entity)
            .map(|(key, _)| *key)
            .collect();
        for rkey in rkeys {
            if let Some((_, sources)) = self.reverse.remove(&rkey) {
                let relation = rkey.relation().component_id();
                for source in sources {
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
            Some((_, targets)) => targets.as_slice(),
            None => &[],
        }
    }

    /// The sources that point at `target` under `relation`, or an empty slice.
    #[inline]
    pub fn sources(&self, relation: ComponentId, target: Entity) -> &[Entity] {
        match self.reverse.get(&reverse_key(relation, target)) {
            Some((_, sources)) => sources.as_slice(),
            None => &[],
        }
    }

    /// The outgoing edges of `entity` as `(relation, target)` pairs.
    pub fn outgoing_edges(&self, entity: Entity) -> Vec<(ComponentId, Entity)> {
        let mut out = Vec::new();
        for (key, (anchor, targets)) in self.forward.iter() {
            if *anchor == entity {
                let relation = key.relation().component_id();
                for &target in targets {
                    out.push((relation, target));
                }
            }
        }
        out
    }

    /// The incoming edges of `entity` as `(relation, source)` pairs.
    pub fn incoming_edges(&self, entity: Entity) -> Vec<(ComponentId, Entity)> {
        let mut out = Vec::new();
        for (key, (anchor, sources)) in self.reverse.iter() {
            if *anchor == entity {
                let relation = key.relation().component_id();
                for &source in sources {
                    out.push((relation, source));
                }
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
        let mut visited: Vec<Entity> = Vec::new();
        let mut queue: Vec<Entity> = Vec::new();

        visited.push(root);
        queue.push(root);

        while let Some(entity) = queue.pop() {
            // Outgoing edges vanish with the deleted holder.
            for (relation, target) in self.index.outgoing_edges(entity) {
                push_unique(
                    &mut plan.removals,
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
                    CleanupPolicy::Remove => push_unique(&mut plan.removals, edge),
                    CleanupPolicy::Delete => {
                        push_unique(&mut plan.removals, edge);
                        if !visited.contains(&source) {
                            visited.push(source);
                            queue.push(source);
                            plan.deletions.push(source);
                        }
                    }
                    CleanupPolicy::Panic => push_unique(&mut plan.panics, edge),
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
