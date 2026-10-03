//! Single-threaded, full-pass hierarchy propagation.
//!
//! Given a [`Hierarchy`], a buffer of authoritative local [`Transform`]s, and a
//! matching output buffer of [`GlobalTransform`]s, [`propagate`] computes every
//! node's world transform in a single parent-before-child sweep:
//!
//! ```text
//! global[node] = global[parent] * local[node]      // non-root
//! global[root] = local[root].affine()              // root
//! ```
//!
//! Composition is done in [`Affine3`](prism_math::Affine3) space inside
//! [`GlobalTransform::mul_transform`], so accumulated **non-uniform scale** is
//! preserved exactly (including the shear it can introduce deeper in the tree).
//! Local data is read-only here: the pass never writes back into `locals`,
//! honoring the "Local is authoritative, Global is only a cache" invariant.
//!
//! M1 recomputes the whole forest every call. The traversal is factored into
//! [`propagate_in_order`] so a later milestone can supply a pruned,
//! dirty-subtree order (see [`crate::change`]) without changing the math.

use alloc::vec::Vec;

use crate::hierarchy::{Hierarchy, HierarchyError, NodeId};
use crate::{GlobalTransform, Transform};

/// Run a full propagation pass, writing a world transform for every node.
///
/// `locals[i]` and `globals[i]` are the local and world transforms of node `i`.
///
/// # Errors
/// - [`HierarchyError::LengthMismatch`] if either buffer length differs from
///   the number of nodes.
/// - [`HierarchyError::Cycle`] if the hierarchy is not a forest.
pub fn propagate(
    hierarchy: &Hierarchy,
    locals: &[Transform],
    globals: &mut [GlobalTransform],
) -> Result<(), HierarchyError> {
    if locals.len() != hierarchy.len() || globals.len() != hierarchy.len() {
        return Err(HierarchyError::LengthMismatch);
    }
    let order = hierarchy.compute_order()?;
    propagate_in_order(hierarchy, &order, locals, globals);
    Ok(())
}

/// Propagate world transforms for the nodes in `order`.
///
/// `order` must list every node to update with each parent appearing before
/// its children, and every referenced parent's world transform must already be
/// valid in `globals` (trivially true for the full order from
/// [`Hierarchy::compute_order`]). A future incremental pass can pass a pruned
/// order of just the dirty subtrees here.
///
/// # Panics
/// Panics if any id in `order`, or any parent reachable from it, is out of
/// bounds for `locals`/`globals`.
pub fn propagate_in_order(
    hierarchy: &Hierarchy,
    order: &[NodeId],
    locals: &[Transform],
    globals: &mut [GlobalTransform],
) {
    for &node in order {
        let local = &locals[node.index()];
        let world = match hierarchy.parent(node) {
            None => GlobalTransform::from_transform(local),
            Some(parent) => globals[parent.index()].mul_transform(local),
        };
        globals[node.index()] = world;
    }
}

/// Allocate a world-transform buffer sized for `hierarchy`, initialized to
/// identity. Convenience for callers that do not already own a buffer.
#[inline]
pub fn identity_globals(hierarchy: &Hierarchy) -> Vec<GlobalTransform> {
    let mut globals = Vec::with_capacity(hierarchy.len());
    globals.resize(hierarchy.len(), GlobalTransform::IDENTITY);
    globals
}
