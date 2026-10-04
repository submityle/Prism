//! §24.1 Static transform baking & batch merging (`Static Baking`).
//!
//! In a large world the overwhelming majority of entities never move: level
//! geometry, buildings, terrain décor. Re-propagating their world transforms
//! every frame — and re-checking whether they are dirty — is pure waste. This
//! module takes nodes the caller has marked **static** (the design doc's
//! `TransformStatic` marker) and does two things once, at load/build time:
//!
//! 1. **Bake & freeze.** Pre-multiply each static subtree's local [`Transform`]
//!    chain into a world [`GlobalTransform`] and store the frozen result. Once
//!    baked, a static subtree leaves the propagation set entirely: runtime pays
//!    zero propagation and zero dirty-checking for it. This is the single
//!    biggest propagation saving in a mostly-static world.
//! 2. **Batch merge.** Collapse many same-key (e.g. same-material) static meshes
//!    into one world-space batch — either a merged list of world instance
//!    matrices or a single merged world-space vertex buffer with their local
//!    [`Transform`] baked away — to cut draw calls and instance-matrix uploads
//!    (the analogue of UE static-mesh merging / Unity static batching).
//!
//! Baking is explicitly reversible: [`StaticBaker::unfreeze_subtree`] returns a
//! subtree to the dynamic set when something that was static must start moving,
//! and [`StaticBaker::invalidate_subtree`] marks baked data stale (bumping a
//! generation counter a cache can watch) so a later [`StaticBaker::bake`]
//! refreshes exactly those subtrees.
//!
//! Everything here is `no_std` + `alloc`, pure deterministic math — no threads,
//! no clock, no ECS dependency — so the baking core can be unit-tested against a
//! level-by-level oracle.

use alloc::vec::Vec;

use prism_math::{Affine3, Vec3};

use crate::hierarchy::{Hierarchy, HierarchyError, NodeId};
use crate::{GlobalTransform, Transform};

/// Outcome of one [`StaticBaker::bake`] call.
///
/// A caller can assert that re-baking an already-fresh scene costs nothing
/// (`baked == 0`), or that a load-time bake collapsed the expected number of
/// static subtrees.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct BakeStats {
    /// Number of static nodes whose world transform this call (re)computed.
    pub baked: usize,
    /// Number of static subtree roots swept (static nodes with no static
    /// parent).
    pub roots: usize,
}

/// Owns the static mark, the frozen world transforms, and the per-node validity
/// of a transform graph's static subset.
///
/// It is a parallel-array companion to a [`crate::TransformGraph`] (or any
/// hierarchy + locals): slot `i` corresponds to [`NodeId::new`]`(i)`. Keep it
/// the same length as the graph with [`StaticBaker::ensure_len`] after spawning
/// nodes.
#[derive(Clone, Debug, Default)]
pub struct StaticBaker {
    /// `is_static[i]` ⇒ node `i` has been marked static and is a candidate for
    /// freezing.
    is_static: Vec<bool>,
    /// Frozen world transform of node `i`, meaningful only where `valid[i]`.
    baked: Vec<GlobalTransform>,
    /// `valid[i]` ⇒ `baked[i]` is a fresh, usable frozen world transform.
    valid: Vec<bool>,
    /// Bumped on every mutation that stales baked data, so a downstream cache
    /// (merged batches, GPU residency) can cheaply detect it must rebuild.
    generation: u64,
}

impl StaticBaker {
    /// Create an empty baker.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            is_static: Vec::new(),
            baked: Vec::new(),
            valid: Vec::new(),
            generation: 0,
        }
    }

    /// Create a baker sized for `len` nodes, nothing static yet.
    #[inline]
    #[must_use]
    pub fn with_len(len: usize) -> Self {
        let mut this = Self::new();
        this.ensure_len(len);
        this
    }

    /// Number of node slots tracked.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.is_static.len()
    }

    /// Whether no node slots are tracked.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.is_static.is_empty()
    }

    /// The current generation counter. It increases whenever baked data is
    /// invalidated, unfrozen, or a static mark is cleared, so a cache keyed on
    /// it can detect staleness in O(1).
    #[inline]
    #[must_use]
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Grow the tracked length to at least `len` (e.g. after spawning nodes),
    /// defaulting new slots to dynamic + unbaked. Never shrinks.
    pub fn ensure_len(&mut self, len: usize) {
        if len > self.is_static.len() {
            self.is_static.resize(len, false);
            self.baked.resize(len, GlobalTransform::IDENTITY);
            self.valid.resize(len, false);
        }
    }

    /// Mark `node` static (a freeze candidate). Marking does not bake; call
    /// [`StaticBaker::bake`] to compute the frozen world transform.
    ///
    /// # Panics
    /// Panics if `node` is out of bounds for the tracked length.
    #[inline]
    pub fn mark_static(&mut self, node: NodeId) {
        assert!(node.index() < self.len(), "mark_static: node out of bounds");
        if !self.is_static[node.index()] {
            self.is_static[node.index()] = true;
            // A newly marked node has no fresh bake yet.
            self.valid[node.index()] = false;
            self.generation += 1;
        }
    }

    /// Clear `node`'s static mark (dropping any frozen data for it), returning
    /// `true` if it had been static. Prefer [`StaticBaker::unfreeze_subtree`]
    /// to return a whole subtree to the dynamic set.
    ///
    /// # Panics
    /// Panics if `node` is out of bounds.
    pub fn clear_static(&mut self, node: NodeId) -> bool {
        assert!(node.index() < self.len(), "clear_static: node out of bounds");
        let was = self.is_static[node.index()];
        if was {
            self.is_static[node.index()] = false;
            self.valid[node.index()] = false;
            self.generation += 1;
        }
        was
    }

    /// Whether `node` is marked static.
    ///
    /// # Panics
    /// Panics if `node` is out of bounds.
    #[inline]
    #[must_use]
    pub fn is_static(&self, node: NodeId) -> bool {
        self.is_static[node.index()]
    }

    /// Whether `node` currently holds a fresh frozen world transform (static
    /// *and* baked since its last invalidation).
    ///
    /// # Panics
    /// Panics if `node` is out of bounds.
    #[inline]
    #[must_use]
    pub fn is_baked(&self, node: NodeId) -> bool {
        self.is_static[node.index()] && self.valid[node.index()]
    }

    /// The frozen world transform of a baked static `node`.
    ///
    /// # Panics
    /// Panics if `node` is out of bounds or is not currently baked (see
    /// [`StaticBaker::is_baked`]); use [`StaticBaker::try_baked`] for a checked
    /// read.
    #[inline]
    #[must_use]
    pub fn baked(&self, node: NodeId) -> GlobalTransform {
        assert!(
            self.is_baked(node),
            "baked: node is not a freshly baked static node"
        );
        self.baked[node.index()]
    }

    /// The frozen world transform of `node`, or `None` if it is not currently a
    /// freshly baked static node.
    ///
    /// # Panics
    /// Panics if `node` is out of bounds.
    #[inline]
    #[must_use]
    pub fn try_baked(&self, node: NodeId) -> Option<GlobalTransform> {
        if self.is_baked(node) {
            Some(self.baked[node.index()])
        } else {
            None
        }
    }

    /// Collect the static subtree roots: static nodes whose parent is not
    /// static (or who have no parent). Each is the entry point of a maximal
    /// connected static component. Returned in ascending [`NodeId`] order.
    #[must_use]
    pub fn static_roots(&self, hierarchy: &Hierarchy) -> Vec<NodeId> {
        let n = self.len().min(hierarchy.len());
        let mut roots = Vec::new();
        for i in 0..n {
            if !self.is_static[i] {
                continue;
            }
            let node = NodeId::new(i as u32);
            let parent_static = match hierarchy.parent(node) {
                Some(parent) => self.is_static.get(parent.index()).copied().unwrap_or(false),
                None => false,
            };
            if !parent_static {
                roots.push(node);
            }
        }
        roots
    }

    /// Bake (freeze) every static subtree's world transform.
    ///
    /// For each static subtree root the world transform is seeded from its
    /// parent: a world root (`parent == None`) seeds from identity, and a static
    /// root whose parent is **dynamic** seeds from `parent_globals[parent]`
    /// (so a caller propagates the dynamic set first, then bakes statics
    /// hanging off it). Within a static component the sweep is strictly
    /// parent-before-child, composing `baked[child] = baked[parent] * local`
    /// in affine space exactly like [`crate::propagation::propagate`], so the
    /// frozen result is bit-for-bit what a full propagation would have produced.
    ///
    /// Baking is **incremental**: only nodes whose cached freeze is invalid
    /// (newly [`StaticBaker::mark_static`]ed or [`StaticBaker::invalidate_subtree`]d)
    /// are recomputed; a component that is already fresh is traversed but
    /// performs zero compositions, so re-baking an unchanged scene reports
    /// `baked == 0`.
    ///
    /// `parent_globals` must have exactly one entry per node (pass
    /// [`crate::propagation::identity_globals`] for a fully static scene whose
    /// static roots are all world roots). Baking marks the recomputed nodes
    /// valid and bumps the [`StaticBaker::generation`] when any work was done.
    ///
    /// # Errors
    /// - [`HierarchyError::LengthMismatch`] if `locals` or `parent_globals`
    ///   length differs from `hierarchy.len()`, or the baker is shorter than
    ///   the hierarchy.
    /// - [`HierarchyError::Cycle`] if the hierarchy is not a forest.
    pub fn bake(
        &mut self,
        hierarchy: &Hierarchy,
        locals: &[Transform],
        parent_globals: &[GlobalTransform],
    ) -> Result<BakeStats, HierarchyError> {
        let n = hierarchy.len();
        if locals.len() != n || parent_globals.len() != n || self.len() < n {
            return Err(HierarchyError::LengthMismatch);
        }
        // A cycle would make the subtree DFS below loop forever.
        hierarchy.validate()?;

        let roots = self.static_roots(hierarchy);
        let mut baked = 0usize;
        // Reusable parent-before-child queue so a child reads an already-fresh
        // parent. We recompute only *invalid* nodes (newly marked or
        // invalidated); a node whose cached bake is still valid is left as-is,
        // so re-baking an unchanged scene performs zero compositions.
        let mut queue: Vec<NodeId> = Vec::new();
        for &root in &roots {
            queue.clear();
            queue.push(root);
            let mut head = 0;
            while head < queue.len() {
                let node = queue[head];
                head += 1;
                if !self.valid[node.index()] {
                    // The invalid set is subtree-closed, so this node's parent
                    // is either dynamic, or static-and-already-fresh, or a
                    // static node recomputed earlier in this same sweep —
                    // in every case `baked[parent]` is correct here.
                    let world = match hierarchy.parent(node) {
                        None => GlobalTransform::from_transform(&locals[node.index()]),
                        Some(parent) if self.is_static[parent.index()] => {
                            self.baked[parent.index()].mul_transform(&locals[node.index()])
                        }
                        Some(parent) => {
                            parent_globals[parent.index()].mul_transform(&locals[node.index()])
                        }
                    };
                    self.baked[node.index()] = world;
                    self.valid[node.index()] = true;
                    baked += 1;
                }
                for &child in hierarchy.children(node) {
                    if self.is_static[child.index()] {
                        queue.push(child);
                    }
                }
            }
        }

        if baked > 0 {
            self.generation += 1;
        }
        Ok(BakeStats {
            baked,
            roots: roots.len(),
        })
    }

    /// Mark the frozen data of the static subtree rooted at `node` **stale**
    /// without unmarking it static, so the next [`StaticBaker::bake`]
    /// recomputes exactly this component. Bumps the generation. Returns the
    /// number of static nodes invalidated.
    ///
    /// Use this when a static node's local was edited in place but it should
    /// stay static (e.g. a one-off re-authoring at load time).
    ///
    /// # Panics
    /// Panics if `node` is out of bounds.
    pub fn invalidate_subtree(&mut self, hierarchy: &Hierarchy, node: NodeId) -> usize {
        let mut count = 0usize;
        self.for_each_static_descendant(hierarchy, node, |this, n| {
            if this.valid[n.index()] {
                this.valid[n.index()] = false;
                count += 1;
            }
        });
        if count > 0 {
            self.generation += 1;
        }
        count
    }

    /// Return the static subtree rooted at `node` to the dynamic set: clear the
    /// static mark and drop the frozen data for every static node in the
    /// component. Bumps the generation. Returns the unfrozen nodes in
    /// parent-before-child order (the set a caller re-admits to propagation).
    ///
    /// # Panics
    /// Panics if `node` is out of bounds.
    pub fn unfreeze_subtree(&mut self, hierarchy: &Hierarchy, node: NodeId) -> Vec<NodeId> {
        let mut out = Vec::new();
        self.for_each_static_descendant(hierarchy, node, |this, n| {
            this.is_static[n.index()] = false;
            this.valid[n.index()] = false;
            out.push(n);
        });
        if !out.is_empty() {
            self.generation += 1;
        }
        out
    }

    /// Visit every node in the maximal static component containing `node`, in
    /// parent-before-child order, calling `f(self, node)`. If `node` itself is
    /// not static, visits nothing.
    fn for_each_static_descendant(
        &mut self,
        hierarchy: &Hierarchy,
        node: NodeId,
        mut f: impl FnMut(&mut Self, NodeId),
    ) {
        if node.index() >= self.len() || !self.is_static[node.index()] {
            return;
        }
        // Parent-before-child via a queue (BFS keeps parents ahead of children).
        let mut queue: Vec<NodeId> = Vec::new();
        queue.push(node);
        let mut head = 0;
        while head < queue.len() {
            let current = queue[head];
            head += 1;
            f(self, current);
            for &child in hierarchy.children(current) {
                if self.is_static[child.index()] {
                    queue.push(child);
                }
            }
        }
    }
}

/// A key used to group static meshes that can be merged into one batch, e.g. a
/// material / pipeline id. Equal keys are mergeable.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct BatchKey(pub u64);

/// A merged group of static instances sharing a [`BatchKey`]: the world-space
/// matrices a renderer uploads once as a single instanced/indirect draw instead
/// of one per entity.
#[derive(Clone, Debug, PartialEq)]
pub struct InstanceBatch {
    /// The shared merge key.
    pub key: BatchKey,
    /// World-space instance matrices, one per merged source, in `sources` order.
    pub instances: Vec<Affine3>,
    /// The source nodes contributing to this batch, in merge order.
    pub sources: Vec<NodeId>,
}

/// Group baked static instances by [`BatchKey`] into [`InstanceBatch`]es.
///
/// `keyed` pairs each candidate node with its merge key. Only nodes that are
/// currently baked (see [`StaticBaker::is_baked`]) contribute; unbaked or
/// dynamic entries are skipped. Batches are returned sorted by ascending key,
/// and within a batch the sources keep their order of appearance in `keyed`, so
/// the output is fully deterministic.
#[must_use]
pub fn merge_instances(baker: &StaticBaker, keyed: &[(NodeId, BatchKey)]) -> Vec<InstanceBatch> {
    let mut batches: Vec<InstanceBatch> = Vec::new();
    for &(node, key) in keyed {
        let Some(world) = baker.try_baked(node) else {
            continue;
        };
        match batches.iter_mut().find(|b| b.key == key) {
            Some(batch) => {
                batch.instances.push(world.affine());
                batch.sources.push(node);
            }
            None => batches.push(InstanceBatch {
                key,
                instances: alloc::vec![world.affine()],
                sources: alloc::vec![node],
            }),
        }
    }
    batches.sort_by_key(|b| b.key);
    batches
}

/// One static mesh offered to [`merge_meshes`]: a source node, its merge key,
/// and its mesh vertices expressed in the node's **local** space.
#[derive(Clone, Copy, Debug)]
pub struct MeshSource<'a> {
    /// The node whose baked world transform bakes these vertices into world
    /// space.
    pub node: NodeId,
    /// The merge key; sources with equal keys merge into one [`MergedMesh`].
    pub key: BatchKey,
    /// Mesh vertices in the node's local space.
    pub vertices: &'a [Vec3],
}

/// A single merged world-space vertex buffer for one [`BatchKey`]: the result
/// of baking several static meshes' local vertices into world space and
/// concatenating them, so their individual local [`Transform`]s are dropped and
/// the group draws as one mesh.
#[derive(Clone, Debug, PartialEq)]
pub struct MergedMesh {
    /// The shared merge key.
    pub key: BatchKey,
    /// All merged vertices, already in world space, in `sources` order.
    pub vertices: Vec<Vec3>,
    /// Per-source back-references: the contributing node and the half-open
    /// range of `vertices` it produced.
    pub sources: Vec<(NodeId, core::ops::Range<usize>)>,
}

/// Bake and concatenate same-key static meshes into merged world-space vertex
/// buffers (the design's "pre-merge to world space, drop each `Transform`").
///
/// Every source's local vertices are transformed by that node's baked world
/// transform (via [`GlobalTransform::transform_point`]) and appended to its
/// key's merged buffer, with a recorded source range. Only baked static sources
/// contribute; others are skipped. Output batches are sorted by ascending key
/// and sources keep input order, so merging is deterministic.
#[must_use]
pub fn merge_meshes(baker: &StaticBaker, sources: &[MeshSource<'_>]) -> Vec<MergedMesh> {
    let mut merged: Vec<MergedMesh> = Vec::new();
    for src in sources {
        let Some(world) = baker.try_baked(src.node) else {
            continue;
        };
        let slot = match merged.iter_mut().find(|m| m.key == src.key) {
            Some(existing) => existing,
            None => {
                merged.push(MergedMesh {
                    key: src.key,
                    vertices: Vec::new(),
                    sources: Vec::new(),
                });
                merged.last_mut().expect("just pushed")
            }
        };
        let start = slot.vertices.len();
        for &v in src.vertices {
            slot.vertices.push(world.transform_point(v));
        }
        let end = slot.vertices.len();
        slot.sources.push((src.node, start..end));
    }
    merged.sort_by_key(|m| m.key);
    merged
}
