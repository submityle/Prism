//! The canonical parent relation and the hierarchy operations built on it
//! (design §11 / §12 / §23.6).
//!
//! [`ChildOf`] is the engine's one well-known hierarchy relation: an *exclusive*
//! relation (an entity has at most one parent — inserting a new one evicts the
//! old, design §23.3) whose target is deleted alongside its children
//! ([`CleanupPolicy::Delete`] on the target side, design §23.2). Everything
//! hierarchy-shaped in the engine — scoped spawning, recursive despawn, event
//! bubbling, transform propagation — walks this relation, so it lives here
//! rather than in every game's own crate.
//!
//! Two entry points use it directly:
//!
//! - [`EntityCommands::with_children`](crate::command::EntityCommands::with_children)
//!   builds a subtree with **scope commands**: children are spawned *and*
//!   attached inside one nested builder, and the queue replays them in
//!   insertion order (spawn before the relation add), which is what makes the
//!   scope deterministic under the deferred command model (design §9).
//! - [`World::despawn_recursive`] removes a subtree deepest-first.

use alloc::vec::Vec;

use crate::collections::HashMap;
use crate::component::Component;
use crate::entity::Entity;
use crate::relation::{CleanupPolicy, Relation, RelationKind};
use crate::world::World;

/// "This entity is a child of that one" — the engine's canonical hierarchy
/// relation (design §11 / §23.6).
///
/// The relation is directional: `child --ChildOf--> parent`. Read the parent of
/// an entity with [`World::relation_targets::<ChildOf>`](World::relation_targets)
/// and the children of a parent with
/// [`World::relation_sources::<ChildOf>`](World::relation_sources).
///
/// # Metadata
/// - **exclusive**: one parent per entity; adding a second target evicts the
///   first (design §23.3).
/// - **`on_delete_target = Delete`**: destroying a parent destroys its children
///   through the relation cascade (design §23.2), so a plain
///   [`World::despawn`] of a parent can never leave orphaned children behind.
/// - **fragmenting**: hierarchies are low-cardinality, which is the case design
///   §11 routes into the archetype. (The flag is metadata today: the relation
///   index stores the edges while the fragmenting storage path is built out.)
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ChildOf;

impl Component for ChildOf {}

impl Relation for ChildOf {
    const KIND: RelationKind = RelationKind {
        fragmenting: true,
        transitive: false,
        exclusive: true,
        on_delete: CleanupPolicy::Remove,
        on_delete_target: CleanupPolicy::Delete,
    };
}

impl World {
    /// Register [`ChildOf`] with its canonical kind if it is not registered yet.
    ///
    /// Idempotent, and called automatically by the hierarchy entry points, so a
    /// world that never builds a hierarchy pays nothing.
    pub fn register_child_of(&mut self) {
        self.register_relation_type::<ChildOf>();
    }

    /// Despawn `entity` and every descendant reachable through [`ChildOf`],
    /// deepest-first, and return the size of the removed subtree (design §23.6).
    ///
    /// Returns `0` for an entity that is already dead. The walk carries a
    /// visited set, so a malformed relation cycle terminates instead of
    /// looping. Descendants are despawned before their ancestors, so observers
    /// and hooks along the way always see a child whose parent still exists.
    ///
    /// The count is what the walk found — the subtree size — rather than "how
    /// many `despawn` calls returned `true`": a delete-policy cascade triggered
    /// partway through can remove an entity before its own turn, and the count
    /// must not depend on that interleaving.
    ///
    /// This does not *depend* on [`ChildOf`]'s delete policy — the same subtree
    /// goes away even if the game re-registered the relation as `Remove`. With
    /// the canonical kind the cascade would remove the children anyway; doing it
    /// explicitly keeps the operation a single deterministic statement.
    pub fn despawn_recursive(&mut self, entity: Entity) -> usize {
        if !self.contains(entity) {
            return 0;
        }
        self.register_child_of();

        // Depth-first pre-order; reversing it despawns children before parents.
        let mut order: Vec<Entity> = Vec::new();
        let mut seen: HashMap<Entity, ()> = HashMap::new();
        let mut stack: Vec<Entity> = Vec::new();
        stack.push(entity);
        while let Some(current) = stack.pop() {
            if !self.contains(current) || seen.insert(current, ()).is_some() {
                continue;
            }
            order.push(current);
            for &child in self.relation_sources::<ChildOf>(current) {
                stack.push(child);
            }
        }

        let subtree = order.len();
        for entity in order.into_iter().rev() {
            self.despawn(entity);
        }
        subtree
    }
}
