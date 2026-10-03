//! World-level owning-group registry (design §6 存储模型 四态 "OwningGroup",
//! §17 "owning group 完美打包", §20 不变量 "owning group 单一拥有者").
//!
//! [`crate::storage::OwningGroup`] is the pure packing structure: given a stream
//! of "entity gained / lost / dropped" events it keeps the entities that own
//! **all** of a fixed component set packed into a contiguous, hole-free prefix
//! so iterating the group is a branch-free linear scan. On its own it does not
//! know which entities satisfy the group, nor does it enforce the single-owner
//! invariant across groups — the module docs there explicitly defer that to
//! "the `World`/group registry, a follow-up task".
//!
//! This registry is that owner. It:
//!
//! * assigns each declared group a dense [`OwningGroupId`] and stores it;
//! * enforces the single-owner invariant — a component may be owned by at most
//!   one group, with a conflicting [`World::register_owning_group`] rejected via
//!   [`OwningGroupError::AlreadyOwned`] rather than silently corrupting packing;
//! * answers, for a batch of components that just changed on an entity, which
//!   groups could be affected, so the `World` only re-evaluates membership for
//!   the handful of groups that actually care.
//!
//! The registry stays deliberately "dumb" about *whether* an entity satisfies a
//! group: that predicate needs the `World`'s archetype + sparse-set state, so
//! the `World` computes it and drives [`OwningGroup::insert`] /
//! [`OwningGroup::remove`] on the owned structure (see
//! [`World::update_owning_groups`](crate::world::World)).

use alloc::vec::Vec;

use crate::component::ComponentId;
use crate::entity::Entity;
use crate::storage::{OwningGroup, OwningGroupId};

/// Why [`World::register_owning_group`](crate::world::World::register_owning_group)
/// refused to declare an owning group.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum OwningGroupError {
    /// The group would own a component already claimed by another group,
    /// violating the single-owner invariant (design §6 "单一拥有者约束，冲突在
    /// 初始化期报错"; §20 不变量). Two groups cannot each impose a packing order
    /// on the same component.
    AlreadyOwned {
        /// The component that is already owned.
        component: ComponentId,
        /// The group that already owns it.
        owner: OwningGroupId,
    },
    /// The owned set was empty; a group must claim at least one component.
    Empty,
}

/// The set of owning groups declared on a [`World`](crate::world::World).
///
/// Groups are stored densely, indexed by the raw value of their
/// [`OwningGroupId`]. `owners` is the single-owner side-table: one
/// `(component, group)` pair per owned component, scanned to detect conflicts
/// at registration and to find affected groups on a structural change. The
/// group count is tiny (one per super-hot query), so linear scans here are
/// cheaper than a hash map and keep the registry `no_std`-trivial.
#[derive(Debug, Default)]
pub(crate) struct OwningGroupRegistry {
    groups: Vec<OwningGroup>,
    owners: Vec<(ComponentId, OwningGroupId)>,
}

impl OwningGroupRegistry {
    /// An empty registry.
    pub(crate) fn new() -> Self {
        Self {
            groups: Vec::new(),
            owners: Vec::new(),
        }
    }

    /// Whether no owning group has been declared. The hot structural-change
    /// paths check this first and skip all owning-group work when it is `true`,
    /// so worlds that never use owning groups pay nothing.
    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.groups.is_empty()
    }

    /// The number of declared owning groups.
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.groups.len()
    }

    /// Declare a new owning group over `owned`, enforcing the single-owner
    /// invariant.
    ///
    /// Duplicate ids within `owned` are collapsed by [`OwningGroup::new`]; the
    /// conflict check and owner table use that canonical set.
    pub(crate) fn register(
        &mut self,
        owned: &[ComponentId],
    ) -> Result<OwningGroupId, OwningGroupError> {
        if owned.is_empty() {
            return Err(OwningGroupError::Empty);
        }
        for &component in owned {
            if let Some(&(_, owner)) = self.owners.iter().find(|(c, _)| *c == component) {
                return Err(OwningGroupError::AlreadyOwned { component, owner });
            }
        }
        let id = OwningGroupId::new(self.groups.len() as u32);
        let group = OwningGroup::new(id, owned);
        for &component in group.owned() {
            self.owners.push((component, id));
        }
        self.groups.push(group);
        Ok(id)
    }

    /// Borrow the group with id `id`, or `None` if `id` was never declared here.
    #[inline]
    pub(crate) fn get(&self, id: OwningGroupId) -> Option<&OwningGroup> {
        self.groups.get(id.index() as usize)
    }

    /// Mutably borrow the group with id `id`.
    #[inline]
    pub(crate) fn get_mut(&mut self, id: OwningGroupId) -> Option<&mut OwningGroup> {
        self.groups.get_mut(id.index() as usize)
    }

    /// The deduplicated ids of every group that owns at least one of the
    /// components in `changed`. Used by the `World` to re-evaluate membership
    /// only for groups a structural change could actually affect.
    pub(crate) fn groups_touching(&self, changed: &[ComponentId]) -> Vec<OwningGroupId> {
        let mut out: Vec<OwningGroupId> = Vec::new();
        for &(component, group) in &self.owners {
            if changed.contains(&component) && !out.contains(&group) {
                out.push(group);
            }
        }
        out
    }

    /// Drop `entity` from every group it is tracked in. Used on despawn, where
    /// the entity ceases to exist regardless of which components it held.
    pub(crate) fn remove_entity(&mut self, entity: Entity) {
        for group in &mut self.groups {
            group.remove(entity);
        }
    }

    /// The ids of every declared group, in registration order. Collected into a
    /// `Vec` so the caller can re-borrow the registry mutably per id (e.g. to
    /// rebuild each group's membership after a `restore` replaces storage).
    pub(crate) fn ids(&self) -> Vec<OwningGroupId> {
        (0..self.groups.len() as u32)
            .map(OwningGroupId::new)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::OwningGroupError;
    use crate::component::{Component, StorageType};
    use crate::world::World;

    #[derive(Debug, PartialEq)]
    struct A(u32);
    #[derive(Debug, PartialEq)]
    struct B(u32);
    #[derive(Debug, PartialEq)]
    struct C(u32);
    struct Toggle;

    impl Component for A {}
    impl Component for B {}
    impl Component for C {}
    impl Component for Toggle {
        const STORAGE: StorageType = StorageType::SparseSet;
    }

    #[test]
    fn empty_owned_set_is_rejected() {
        let mut world = World::new();
        assert_eq!(
            world.register_owning_group(&[]),
            Err(OwningGroupError::Empty)
        );
        assert_eq!(world.owning_group_count(), 0);
    }

    #[test]
    fn single_owner_conflict_is_rejected() {
        let mut world = World::new();
        let a = world.register_component::<A>();
        let b = world.register_component::<B>();
        let c = world.register_component::<C>();

        let first = world.register_owning_group(&[a, b]).unwrap();
        // A second group may not also claim `a`.
        assert_eq!(
            world.register_owning_group(&[a, c]),
            Err(OwningGroupError::AlreadyOwned {
                component: a,
                owner: first,
            })
        );
        // A disjoint group is fine.
        assert!(world.register_owning_group(&[c]).is_ok());
        assert_eq!(world.owning_group_count(), 2);
    }

    #[test]
    fn spawn_packs_only_full_members() {
        let mut world = World::new();
        let a = world.register_component::<A>();
        let b = world.register_component::<B>();
        let group = world.register_owning_group(&[a, b]).unwrap();

        let full = world.spawn((A(1), B(2)));
        let partial = world.spawn(A(3));

        let g = world.owning_group(group).unwrap();
        assert_eq!(g.len(), 1);
        assert!(g.contains(full));
        assert!(!g.contains(partial));
        assert!(!g.is_tracked(partial));
        // The packed prefix is exactly the member set, hole-free.
        assert_eq!(g.packed(), &[full]);
    }

    #[test]
    fn insert_completes_and_remove_breaks_membership() {
        let mut world = World::new();
        let a = world.register_component::<A>();
        let b = world.register_component::<B>();
        let group = world.register_owning_group(&[a, b]).unwrap();

        let e = world.spawn(A(1));
        assert!(!world.owning_group(group).unwrap().contains(e));

        // Gaining the final component packs it.
        world.insert(e, B(2));
        assert!(world.owning_group(group).unwrap().contains(e));
        assert_eq!(world.owning_group(group).unwrap().len(), 1);

        // Losing an owned component unpacks it entirely.
        world.remove::<B>(e);
        assert!(!world.owning_group(group).unwrap().contains(e));
        assert_eq!(world.owning_group(group).unwrap().len(), 0);
    }

    #[test]
    fn despawn_drops_membership() {
        let mut world = World::new();
        let a = world.register_component::<A>();
        let b = world.register_component::<B>();
        let group = world.register_owning_group(&[a, b]).unwrap();

        let e = world.spawn((A(1), B(2)));
        assert!(world.owning_group(group).unwrap().contains(e));

        world.despawn(e);
        assert!(world.owning_group(group).unwrap().is_empty());
    }

    #[test]
    fn registration_retroactively_packs_existing_entities() {
        let mut world = World::new();
        let a = world.register_component::<A>();
        let b = world.register_component::<B>();

        let full_a = world.spawn((A(1), B(2)));
        let full_b = world.spawn((A(3), B(4)));
        let _partial = world.spawn(A(5));

        // Group declared *after* the entities already exist.
        let group = world.register_owning_group(&[a, b]).unwrap();
        let g = world.owning_group(group).unwrap();
        assert_eq!(g.len(), 2);
        assert!(g.contains(full_a));
        assert!(g.contains(full_b));
    }

    #[test]
    fn sparse_owned_component_participates() {
        let mut world = World::new();
        let a = world.register_component::<A>();
        let toggle = world.register_component::<Toggle>();
        let group = world.register_owning_group(&[a, toggle]).unwrap();

        let e = world.spawn(A(1));
        assert!(!world.owning_group(group).unwrap().contains(e));

        // Toggling the sparse component on completes the group...
        world.insert(e, Toggle);
        assert!(world.owning_group(group).unwrap().contains(e));

        // ...and toggling it off breaks membership without an archetype move.
        world.remove::<Toggle>(e);
        assert!(!world.owning_group(group).unwrap().contains(e));
    }

    #[test]
    fn unused_registry_is_empty_fast_path() {
        let mut world = World::new();
        // No groups declared: structural ops must not panic and the count is 0.
        let e = world.spawn((A(1), B(2)));
        world.insert(e, C(3));
        world.remove::<C>(e);
        world.despawn(e);
        assert_eq!(world.owning_group_count(), 0);
    }

    /// An `on_remove` hook that re-inserts the final owned component, completing
    /// the `[A, B]` group on the still-live entity midway through its own
    /// despawn — exercising the re-entrancy window closed by the post-free
    /// second `remove_entity`.
    fn readd_b_on_remove(ctx: crate::component_hooks::HookContext<'_>) {
        ctx.world.insert(ctx.entity, B(0));
    }

    #[test]
    fn despawn_hook_reinsertion_leaves_no_dead_entity_packed() {
        let mut world = World::new();
        let a = world.register_component::<A>();
        let b = world.register_component::<B>();
        let _c = world.register_component::<C>();
        let group = world.register_owning_group(&[a, b]).unwrap();

        // Hook C's removal (which fires during despawn) to re-add B.
        world.register_component_hooks::<C>(
            crate::component_hooks::ComponentHooks::new().with_on_remove(readd_b_on_remove),
        );

        // `e` owns A and C but not B, so it starts outside the group.
        let e = world.spawn((A(1), C(2)));
        assert!(!world.owning_group(group).unwrap().contains(e));

        // Despawn fires C's on_remove, which re-inserts B and re-tracks the
        // still-live `e`; the free that follows must drop it again.
        world.despawn(e);

        let g = world.owning_group(group).unwrap();
        assert!(g.is_empty(), "dead entity must not linger packed");
        assert!(!g.contains(e));
    }
}
