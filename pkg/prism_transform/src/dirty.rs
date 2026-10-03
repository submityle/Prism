//! M2 incremental, dirty-subtree propagation.
//!
//! The M1 [`propagate`](crate::propagation::propagate) pass recomputes every
//! node's world transform on every call. For a mostly-static scene that is
//! wasteful: if nothing moved, nothing needs to be recomputed. This module adds
//! an *incremental* pass that uses the M1 [`ChangeTicks`] to recompute only the
//! subtrees whose world pose can actually have changed, and skips clean
//! subtrees entirely.
//!
//! ## Why "a dirty node forces its whole subtree"
//! A node's world pose is `global[node] = global[parent] * local[node]`. If a
//! node's *local* changed, its own world pose changes, and because every
//! descendant composes through it, every descendant's world pose can change
//! too — even descendants whose own local is untouched. So the unit of
//! recomputation is a **subtree rooted at a changed node**, not a single node.
//!
//! ## Minimal dirty roots
//! If a changed node already has a changed ancestor, the ancestor's subtree
//! sweep will recompute it anyway, so it is redundant to treat it as its own
//! starting point. The [`DirtyPropagator`] therefore collects only the
//! **dirty roots** — changed nodes with no changed ancestor — and sweeps each
//! dirty root's subtree in parent-before-child order. Dirty roots are pairwise
//! disjoint (no dirty root is an ancestor of another), so no node is recomputed
//! twice. Each dirty root's parent is, by definition, clean, so its cached
//! [`GlobalTransform`] is still valid and can seed the sweep.
//!
//! ## Near-zero static cost
//! When nothing changed, the pass performs a single cheap scan of the
//! change-tick state, finds no dirty roots, and performs **zero** world-matrix
//! compositions. [`DirtyStats::recomputed`] reports exactly how many world
//! transforms were (re)written, so a static scene can be asserted to cost
//! nothing.
//!
//! ## Equivalence to the full pass
//! For any sequence of local edits (including reparenting, which marks the
//! moved node), the globals produced by [`DirtyPropagator::propagate`] followed
//! by [`ChangeTicks::end_pass`] are identical to those a full
//! [`propagate`](crate::propagation::propagate) would produce from the same
//! locals and hierarchy: each node is recomputed with the same
//! `global[parent] * local[node]` arithmetic, and every node whose result could
//! differ is inside some dirty root's subtree.

use alloc::vec::Vec;

use crate::change::ChangeTicks;
use crate::hierarchy::{Hierarchy, HierarchyError, NodeId};
use crate::{GlobalTransform, Transform};

/// Outcome of one [`DirtyPropagator::propagate`] call: how much work the
/// incremental pass actually did.
///
/// This is the "benchmark as spec" hook for M2: a static scene must yield
/// `recomputed == 0`, and a localized edit must yield a `recomputed` equal to
/// the size of the affected subtree(s) rather than the whole forest.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct DirtyStats {
    /// Number of world transforms (re)written this pass. Equals the summed size
    /// of every dirty root's subtree, and `0` for a static scene.
    pub recomputed: usize,
    /// Number of dirty roots swept this pass (changed nodes with no changed
    /// ancestor).
    pub dirty_roots: usize,
}

/// Incremental, dirty-subtree world-transform propagator.
///
/// It owns reusable scratch buffers so repeated passes over the same hierarchy
/// do not reallocate. Create one with [`DirtyPropagator::new`] (or
/// [`Default`]) and call [`DirtyPropagator::propagate`] each frame; the owning
/// [`crate::TransformGraph`] does this for you via
/// [`crate::TransformGraph::propagate_incremental`].
#[derive(Clone, Debug, Default)]
pub struct DirtyPropagator {
    /// `marked[i]` is `true` when node `i`'s local changed since the last pass.
    marked: Vec<bool>,
    /// The dirty roots collected for the current pass.
    roots: Vec<NodeId>,
    /// Explicit DFS stack used to sweep a dirty root's subtree.
    stack: Vec<NodeId>,
}

impl DirtyPropagator {
    /// Create a propagator with empty scratch buffers.
    #[inline]
    pub const fn new() -> Self {
        Self { marked: Vec::new(), roots: Vec::new(), stack: Vec::new() }
    }

    /// Run one incremental pass, recomputing only the dirty subtrees and
    /// leaving clean subtrees' cached globals untouched.
    ///
    /// `ticks` decides what changed: a node is dirty when
    /// [`ChangeTicks::is_changed`] is true for it. Call [`ChangeTicks::end_pass`]
    /// *after* this returns to close the change epoch (the owning
    /// [`crate::TransformGraph`] does so).
    ///
    /// Returns the [`DirtyStats`] describing the work performed.
    ///
    /// # Errors
    /// - [`HierarchyError::LengthMismatch`] if `locals`, `globals`, or `ticks`
    ///   do not all have exactly one slot per node.
    /// - [`HierarchyError::Cycle`] if the hierarchy is not a forest (an
    ///   ancestor walk would otherwise not terminate). This is detected
    ///   defensively via [`Hierarchy::validate`] only when there is work to do.
    pub fn propagate(
        &mut self,
        hierarchy: &Hierarchy,
        ticks: &ChangeTicks,
        locals: &[Transform],
        globals: &mut [GlobalTransform],
    ) -> Result<DirtyStats, HierarchyError> {
        let n = hierarchy.len();
        if locals.len() != n || globals.len() != n || ticks.len() != n {
            return Err(HierarchyError::LengthMismatch);
        }

        // Cheap dirty-set scan. For a static scene every slot is clean and we
        // return here having done no world-matrix work at all.
        self.marked.clear();
        self.marked.resize(n, false);
        let mut any_marked = false;
        for i in 0..n {
            if ticks.is_changed(NodeId::new(i as u32)) {
                self.marked[i] = true;
                any_marked = true;
            }
        }
        if !any_marked {
            return Ok(DirtyStats::default());
        }

        // A cycle would make the ancestor walk below loop forever; validate
        // defensively now that we know there is work to do.
        hierarchy.validate()?;

        // Collect minimal dirty roots: a marked node with no marked ancestor.
        self.roots.clear();
        for i in 0..n {
            if !self.marked[i] {
                continue;
            }
            let node = NodeId::new(i as u32);
            let mut cursor = hierarchy.parent(node);
            let mut has_marked_ancestor = false;
            while let Some(parent) = cursor {
                if self.marked[parent.index()] {
                    has_marked_ancestor = true;
                    break;
                }
                cursor = hierarchy.parent(parent);
            }
            if !has_marked_ancestor {
                self.roots.push(node);
            }
        }

        // Sweep each dirty root's subtree in parent-before-child order. A node
        // is written before any of its children are visited, so each child
        // reads an already-updated parent global. Dirty roots are disjoint, so
        // no node is written twice.
        let dirty_roots = self.roots.len();
        let mut recomputed = 0usize;
        for ri in 0..self.roots.len() {
            let root = self.roots[ri];
            self.stack.clear();
            self.stack.push(root);
            while let Some(node) = self.stack.pop() {
                let local = &locals[node.index()];
                let world = match hierarchy.parent(node) {
                    None => GlobalTransform::from_transform(local),
                    Some(parent) => globals[parent.index()].mul_transform(local),
                };
                globals[node.index()] = world;
                recomputed += 1;
                for &child in hierarchy.children(node) {
                    self.stack.push(child);
                }
            }
        }

        Ok(DirtyStats { recomputed, dirty_roots })
    }
}
