//! Deterministic scene instantiation using public `prism_ecs` APIs.
//!
//! Opaque serialized column bytes are never reinterpreted as Rust values.
//! Plans with payload columns or entity-valued byte slots must be materialized
//! through `TypedSpawnPlan`, which constructs actual typed Bundles.

use crate::identity::LocalEntityId;
use crate::instance::SceneInstance;
use crate::plan::{InstantiationPlan, PlanError};
use crate::remap::RemapTable;
use crate::residency::ResidencyTier;
use alloc::boxed::Box;
use alloc::vec::Vec;
use prism_asset::StableGuid;
use prism_ecs::bundle::Bundle;
use prism_ecs::entity::Entity;
use prism_ecs::hierarchy::ChildOf;
use prism_ecs::world::World;

type SpawnFn = Box<
    dyn FnOnce(&mut SpawnContext<'_>, Entity, &RemapTable) -> Result<(), TypedSpawnError>
        + Send
        + Sync,
>;

/// Restricted world access granted to a typed scene constructor.
///
/// A constructor can materialize its own reserved entity and add a
/// `ChildOf` relation between entities in the same instance. It cannot spawn,
/// despawn, reserve, or mutate arbitrary external entities, which keeps a
/// returned constructor error transactional for the instance-owned state.
pub struct SpawnContext<'a> {
    world: &'a mut World,
    owned: &'a RemapTable,
    owner: Entity,
}

impl SpawnContext<'_> {
    /// Materializes one entity reserved for this scene instance.
    pub fn spawn_at<B: Bundle>(
        &mut self,
        entity: Entity,
        bundle: B,
    ) -> Result<(), SpawnContextError> {
        if entity != self.owner {
            return Err(SpawnContextError::EntityNotCurrent(entity));
        }
        if !self.owned.entities().any(|(_, owned)| owned == entity)
            || !self.world.entities().contains(entity)
        {
            return Err(SpawnContextError::EntityNotOwned(entity));
        }
        if !self
            .world
            .entities()
            .location(entity)
            .is_some_and(prism_ecs::entity::EntityLocation::is_empty)
        {
            return Err(SpawnContextError::EntityAlreadyMaterialized(entity));
        }
        self.world.spawn_at(entity, bundle);
        Ok(())
    }

    /// Adds an instance-local `ChildOf` relation.
    pub fn add_child_of(&mut self, child: Entity, parent: Entity) -> Result<(), SpawnContextError> {
        if !self.owned.entities().any(|(_, owned)| owned == child)
            || !self.owned.entities().any(|(_, owned)| owned == parent)
            || !self.world.entities().contains(child)
            || !self.world.entities().contains(parent)
            || self
                .world
                .entities()
                .location(child)
                .is_none_or(prism_ecs::entity::EntityLocation::is_empty)
            || self
                .world
                .entities()
                .location(parent)
                .is_none_or(prism_ecs::entity::EntityLocation::is_empty)
        {
            return Err(SpawnContextError::EntityNotOwned(child));
        }
        if child == parent || would_create_child_cycle(self.world, child, parent) {
            return Err(SpawnContextError::HierarchyCycle { child, parent });
        }
        self.world.register_child_of();
        self.world.add_relation::<ChildOf>(child, parent);
        Ok(())
    }
}

/// Error returned when a restricted constructor addresses an external entity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpawnContextError {
    /// The entity is not owned by the scene instance being materialized.
    EntityNotOwned(Entity),
    /// The instance-owned entity was already materialized by an earlier
    /// constructor invocation.
    EntityAlreadyMaterialized(Entity),
    /// The constructor attempted to materialize a different local entity.
    EntityNotCurrent(Entity),
    /// The relation would make the instance hierarchy cyclic.
    HierarchyCycle {
        /// Proposed child entity.
        child: Entity,
        /// Proposed parent entity.
        parent: Entity,
    },
}

/// Explicit failure returned by a typed entity constructor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TypedSpawnError {
    /// Caller-defined stable diagnostic code.
    pub code: u32,
}

/// Typed per-entity materializers for one instantiation plan.
#[derive(Default)]
pub struct TypedSpawnPlan {
    entries: Vec<(LocalEntityId, SpawnFn)>,
}

impl TypedSpawnPlan {
    /// Creates an empty typed plan.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds an actual Bundle value for a planned local entity.
    #[must_use]
    pub fn with_bundle<B: Bundle>(mut self, local: LocalEntityId, bundle: B) -> Self {
        self.entries.push((
            local,
            Box::new(move |context, entity, _| {
                context
                    .spawn_at(entity, bundle)
                    .map_err(|_| TypedSpawnError { code: u32::MAX })
            }),
        ));
        self
    }

    /// Adds a constructor with access to the complete remap.
    ///
    /// The constructor is expected to return errors rather than panic. Since
    /// this crate is `no_std`, unwinding cannot be caught portably; a panic
    /// during materialization does not provide the rollback guarantee of a
    /// returned [`TypedSpawnError`].
    #[must_use]
    pub fn with_constructor(
        mut self,
        local: LocalEntityId,
        constructor: impl FnOnce(&mut SpawnContext<'_>, Entity, &RemapTable) -> Result<(), TypedSpawnError>
            + Send
            + Sync
            + 'static,
    ) -> Self {
        self.entries.push((local, Box::new(constructor)));
        self
    }
}

/// A live instance and its independent local-to-world mapping.
#[derive(Clone, Debug, PartialEq)]
pub struct SceneSpawn {
    /// Local zero is the root; empty plans have no root.
    pub root: Option<Entity>,
    /// The private local-to-world mapping. Use [`Self::resolve`] or
    /// [`Self::remap`] for read-only access; callers cannot register arbitrary
    /// entities as instance-owned through the spawn result.
    remap: RemapTable,
}

impl SceneSpawn {
    #[cfg(feature = "materialize")]
    pub(crate) fn from_remap(remap: RemapTable) -> Self {
        let root = remap.get(LocalEntityId::new(0));
        Self { root, remap }
    }

    /// Builds the instance anchor once the caller supplies the template identity
    /// that is intentionally absent from `InstantiationPlan`.
    #[must_use]
    pub fn scene_instance(&self, template: StableGuid, tier: ResidencyTier) -> SceneInstance {
        SceneInstance::new(template, self.remap.clone(), tier)
    }

    /// Resolves one local entity to its live handle.
    #[must_use]
    pub fn resolve(&self, local: LocalEntityId) -> Option<Entity> {
        self.remap.get(local)
    }

    /// Returns the immutable local-to-world mapping for inspection.
    #[must_use]
    pub fn remap(&self) -> &RemapTable {
        &self.remap
    }
}

/// Strongly typed instantiation failures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InstantiateError {
    /// Plan structure is invalid.
    InvalidPlan(PlanError),
    /// Opaque column data requires a typed materializer.
    TypedSpawnRequired {
        /// Component whose payload could not be materialized generically.
        component: crate::overrides::ComponentTypeId,
    },
    /// Opaque entity-reference bytes require a typed materializer.
    TypedFieldRemapRequired,
    /// Child appears under multiple parents.
    MultipleParents {
        /// Local entity with conflicting parent declarations.
        child: LocalEntityId,
    },
    /// Hierarchy contains a cycle.
    HierarchyCycle {
        /// Local entity at which the cycle was detected.
        local: LocalEntityId,
    },
    /// Typed constructor set does not exactly match plan locals.
    TypedLocalsMismatch {
        /// Local entity missing from, or unexpectedly present in, the constructors.
        local: LocalEntityId,
    },
    /// The World already has deferred entity reservations, so this spawn
    /// cannot safely establish ownership of the reservation range.
    OutstandingReservations,
    /// Constructor failed; all reserved instance entities were rolled back.
    TypedSpawnFailed {
        /// Local entity whose constructor failed.
        local: LocalEntityId,
        /// Caller-defined constructor error code.
        code: u32,
    },
    /// A constructor relation conflicts with the plan's hierarchy.
    DynamicHierarchyCycle {
        /// The child local id.
        child: LocalEntityId,
        /// The parent local id.
        parent: LocalEntityId,
    },
}

/// Spawns a plan with no opaque component payloads or entity-reference slots.
pub fn spawn_plan(
    world: &mut World,
    plan: &InstantiationPlan,
) -> Result<SceneSpawn, InstantiateError> {
    plan.validate().map_err(InstantiateError::InvalidPlan)?;
    if let Some(column) = plan
        .groups
        .iter()
        .flat_map(|group| group.columns.iter())
        .next()
    {
        return Err(InstantiateError::TypedSpawnRequired {
            component: column.component,
        });
    }
    if !plan.entity_valued_fields.is_empty() {
        return Err(InstantiateError::TypedFieldRemapRequired);
    }
    let mut typed = TypedSpawnPlan::new();
    for local in plan
        .groups
        .iter()
        .flat_map(|group| group.locals.iter().copied())
    {
        typed.entries.push((
            local,
            Box::new(|context, entity, _| {
                context
                    .spawn_at(entity, ())
                    .map_err(|_| TypedSpawnError { code: u32::MAX })
            }),
        ));
    }
    spawn_typed_plan(world, plan, typed)
}

/// Spawns typed bundles deterministically. All entities are reserved first, so
/// constructors can remap references even when their targets are later locals.
pub fn spawn_typed_plan(
    world: &mut World,
    plan: &InstantiationPlan,
    mut typed: TypedSpawnPlan,
) -> Result<SceneSpawn, InstantiateError> {
    plan.validate().map_err(InstantiateError::InvalidPlan)?;
    // The current materializer is deliberately typed-only: accepting a plan
    // with opaque columns or entity-valued byte slots here would silently drop
    // payloads and references. Keep that contract explicit at this boundary.
    if let Some(column) = plan
        .groups
        .iter()
        .flat_map(|group| group.columns.iter())
        .next()
    {
        return Err(InstantiateError::TypedSpawnRequired {
            component: column.component,
        });
    }
    if !plan.entity_valued_fields.is_empty() {
        return Err(InstantiateError::TypedFieldRemapRequired);
    }
    if world.entities().has_outstanding_reservations() {
        return Err(InstantiateError::OutstandingReservations);
    }
    let mut locals: Vec<_> = plan
        .groups
        .iter()
        .flat_map(|group| group.locals.iter().copied())
        .collect();
    locals.sort_unstable();
    typed.entries.sort_unstable_by_key(|(id, _)| *id);
    let mut planned = locals.iter().copied();
    let mut provided = typed.entries.iter().map(|(id, _)| *id);
    loop {
        match (planned.next(), provided.next()) {
            (Some(expected), Some(actual)) if expected == actual => {}
            (Some(expected), Some(actual)) => {
                return Err(InstantiateError::TypedLocalsMismatch {
                    local: expected.min(actual),
                });
            }
            (Some(local), None) | (None, Some(local)) => {
                return Err(InstantiateError::TypedLocalsMismatch { local });
            }
            (None, None) => break,
        }
    }
    let mut remap = RemapTable::with_base_capacity(plan.entity_count as usize);
    for _ in 0..plan.entity_count {
        remap.push_base(world.entities().reserve_entity());
    }
    world.flush_reserved();

    for (local, constructor) in typed.entries {
        let entity = remap.get(local).expect("validated plan local has mapping");
        let result = {
            let mut context = SpawnContext {
                world,
                owned: &remap,
                owner: entity,
            };
            constructor(&mut context, entity, &remap)
        };
        if let Err(error) = result {
            rollback(world, &remap, plan);
            return Err(InstantiateError::TypedSpawnFailed {
                local,
                code: error.code,
            });
        }
        if !world.contains(entity) {
            rollback(world, &remap, plan);
            return Err(InstantiateError::TypedSpawnFailed {
                local,
                code: u32::MAX,
            });
        }
    }
    world.register_child_of();
    for &(child, parent) in plan.hierarchy.iter() {
        let child_entity = remap.get(child).expect("validated child");
        let parent_entity = remap.get(parent).expect("validated parent");
        if would_create_child_cycle(world, child_entity, parent_entity) {
            rollback(world, &remap, plan);
            return Err(InstantiateError::DynamicHierarchyCycle { child, parent });
        }
        world.add_relation::<ChildOf>(child_entity, parent_entity);
    }

    let root = remap.get(LocalEntityId::new(0));
    Ok(SceneSpawn { root, remap })
}

pub(crate) fn would_create_child_cycle(world: &World, child: Entity, parent: Entity) -> bool {
    let mut cursor = parent;
    loop {
        if cursor == child {
            return true;
        }
        let Some(next) = world.relation_targets::<ChildOf>(cursor).first().copied() else {
            return false;
        };
        cursor = next;
    }
}

fn rollback(world: &mut World, remap: &RemapTable, plan: &InstantiationPlan) {
    for &(child, parent) in plan.hierarchy.iter() {
        if let (Some(child), Some(parent)) = (remap.get(child), remap.get(parent)) {
            world.remove_relation::<ChildOf>(child, parent);
        }
    }
    for (_, entity) in remap.entities() {
        world.despawn(entity);
    }
}

/// Despawns exactly the entities owned by the instance.
pub fn despawn_instance(world: &mut World, spawn: &SceneSpawn) -> usize {
    for (_, child) in spawn.remap.entities() {
        let parents: Vec<_> = world.relation_targets::<ChildOf>(child).to_vec();
        for parent in parents {
            world.remove_relation::<ChildOf>(child, parent);
        }
    }
    // ChildOf deletes children when its target is destroyed. Detach external
    // children first so this operation cannot cascade outside the instance.
    for (_, parent) in spawn.remap.entities() {
        let children: Vec<_> = world.relation_sources::<ChildOf>(parent).to_vec();
        for child in children {
            world.remove_relation::<ChildOf>(child, parent);
        }
    }
    let mut removed = 0;
    for (_, entity) in spawn.remap.entities() {
        if world.despawn(entity) {
            removed += 1;
        }
    }
    removed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::overrides::{ArchetypeSignature, ComponentTypeId};
    use crate::plan::{ArchetypeGroup, FieldSlot};
    use alloc::boxed::Box;
    use alloc::vec;
    use core::mem::size_of;
    use prism_ecs::component::Component;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct Marker(u32);
    impl Component for Marker {}

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct Link(Entity);
    impl Component for Link {}

    fn plan(locals: &[u32], hierarchy: &[(u32, u32)]) -> InstantiationPlan {
        InstantiationPlan {
            entity_count: locals.len() as u32,
            groups: if locals.is_empty() {
                Box::new([])
            } else {
                Box::new([ArchetypeGroup::new(
                    ArchetypeSignature::default(),
                    locals.iter().copied().map(LocalEntityId::new).collect(),
                    Box::new([]),
                )])
            },
            entity_valued_fields: Box::new([]),
            hierarchy: if hierarchy.is_empty() && !locals.is_empty() {
                locals
                    .iter()
                    .copied()
                    .filter(|&local| local != 0)
                    .map(|local| (LocalEntityId::new(local), LocalEntityId::new(0)))
                    .collect()
            } else {
                hierarchy
                    .iter()
                    .map(|&(c, p)| (LocalEntityId::new(c), LocalEntityId::new(p)))
                    .collect()
            },
            reserved_local_base: LocalEntityId::RESERVED_BASE,
        }
    }

    #[test]
    fn empty_plan_spawns_and_despawns_nothing() {
        let mut world = World::new();
        let spawned = spawn_plan(&mut world, &plan(&[], &[])).unwrap();
        assert_eq!(spawned.root, None);
        assert!(spawned.remap.is_empty());
        assert_eq!(despawn_instance(&mut world, &spawned), 0);
    }

    #[test]
    fn entity_only_hierarchy_is_remapped_and_despawned() {
        let mut world = World::new();
        let spawned = spawn_plan(&mut world, &plan(&[0, 1, 2], &[(1, 0), (2, 1)])).unwrap();
        let root = spawned.resolve(LocalEntityId::new(0)).unwrap();
        let child = spawned.resolve(LocalEntityId::new(1)).unwrap();
        let grandchild = spawned.resolve(LocalEntityId::new(2)).unwrap();
        assert_eq!(spawned.root, Some(root));
        assert!(world.has_relation::<ChildOf>(child, root));
        assert!(world.has_relation::<ChildOf>(grandchild, child));
        assert_eq!(despawn_instance(&mut world, &spawned), 3);
        assert!(!world.contains(root));
        assert!(!world.contains(child));
        assert!(!world.contains(grandchild));
    }

    #[test]
    fn typed_bundle_and_two_instances_are_isolated() {
        let plan = plan(&[0, 1], &[]);
        let mut world = World::new();
        let a = spawn_typed_plan(
            &mut world,
            &plan,
            TypedSpawnPlan::new()
                .with_bundle(LocalEntityId::new(0), Marker(10))
                .with_bundle(LocalEntityId::new(1), Marker(11)),
        )
        .unwrap();
        let b = spawn_typed_plan(
            &mut world,
            &plan,
            TypedSpawnPlan::new()
                .with_bundle(LocalEntityId::new(0), Marker(20))
                .with_bundle(LocalEntityId::new(1), Marker(21)),
        )
        .unwrap();
        let a0 = a.resolve(LocalEntityId::new(0)).unwrap();
        let b0 = b.resolve(LocalEntityId::new(0)).unwrap();
        assert_ne!(a0, b0);
        assert_eq!(world.get::<Marker>(a0), Some(&Marker(10)));
        assert_eq!(world.get::<Marker>(b0), Some(&Marker(20)));
        assert_eq!(despawn_instance(&mut world, &a), 2);
        assert!(world.contains(b0));
    }

    #[test]
    fn typed_constructor_can_remap_a_later_local() {
        let plan = plan(&[0, 1], &[]);
        let mut world = World::new();
        let spawned = spawn_typed_plan(
            &mut world,
            &plan,
            TypedSpawnPlan::new()
                .with_constructor(LocalEntityId::new(0), |context, entity, remap| {
                    let target = remap.get(LocalEntityId::new(1)).unwrap();
                    context.spawn_at(entity, Link(target)).unwrap();
                    Ok(())
                })
                .with_bundle(LocalEntityId::new(1), Marker(7)),
        )
        .unwrap();
        let source = spawned.resolve(LocalEntityId::new(0)).unwrap();
        let target = spawned.resolve(LocalEntityId::new(1)).unwrap();
        assert_eq!(world.get::<Link>(source), Some(&Link(target)));
    }

    #[test]
    fn constructor_can_only_materialize_its_current_local() {
        let mut world = World::new();
        let result = spawn_typed_plan(
            &mut world,
            &plan(&[0, 1], &[]),
            TypedSpawnPlan::new()
                .with_constructor(LocalEntityId::new(0), |context, _, remap| {
                    let other = remap.get(LocalEntityId::new(1)).unwrap();
                    assert!(matches!(
                        context.spawn_at(other, Marker(1)),
                        Err(SpawnContextError::EntityNotCurrent(_))
                    ));
                    Err(TypedSpawnError { code: 7 })
                })
                .with_bundle(LocalEntityId::new(1), Marker(2)),
        );
        assert_eq!(
            result,
            Err(InstantiateError::TypedSpawnFailed {
                local: LocalEntityId::new(0),
                code: 7,
            })
        );
        assert_eq!(world.entity_count(), 0);
    }

    #[test]
    fn constructor_hierarchy_cycle_is_rejected_and_rolled_back() {
        let mut world = World::new();
        let result = spawn_typed_plan(
            &mut world,
            &plan(&[0, 1], &[(1, 0)]),
            TypedSpawnPlan::new()
                .with_bundle(LocalEntityId::new(0), Marker(0))
                .with_constructor(LocalEntityId::new(1), |context, entity, remap| {
                    context.spawn_at(entity, Marker(1)).unwrap();
                    context
                        .add_child_of(remap.get(LocalEntityId::new(0)).unwrap(), entity)
                        .unwrap();
                    Ok(())
                }),
        );
        assert!(matches!(
            result,
            Err(InstantiateError::DynamicHierarchyCycle { .. })
        ));
        assert_eq!(world.entity_count(), 0);
    }

    #[test]
    fn typed_materializer_rejects_opaque_plan_payloads() {
        let mut payload_plan = plan(&[0], &[]);
        payload_plan.groups[0].signature =
            ArchetypeSignature::new(vec![ComponentTypeId::of::<Marker>()]);
        payload_plan.groups[0].columns = Box::new([crate::plan::Column::new(
            ComponentTypeId::of::<Marker>(),
            vec![0; size_of::<Marker>()].into_boxed_slice(),
        )]);
        let mut world = World::new();
        let result = spawn_typed_plan(
            &mut world,
            &payload_plan,
            TypedSpawnPlan::new().with_bundle(LocalEntityId::new(0), Marker(5)),
        );
        assert!(matches!(
            result,
            Err(InstantiateError::TypedSpawnRequired { .. })
        ));
        assert_eq!(world.entity_count(), 0);
    }

    #[test]
    fn spawn_refuses_to_flush_reservations_owned_by_the_caller() {
        let mut world = World::new();
        let pending = world.entities().reserve_entity();
        assert_eq!(
            spawn_plan(&mut world, &plan(&[0], &[])),
            Err(InstantiateError::OutstandingReservations)
        );
        assert!(world.entities().has_outstanding_reservations());
        world.flush_reserved();
        assert!(world.contains(pending));
    }

    #[test]
    fn despawn_does_not_cascade_into_external_children() {
        let mut world = World::new();
        let spawned = spawn_plan(&mut world, &plan(&[0], &[])).unwrap();
        let root = spawned.root.unwrap();
        let external = world.spawn(());
        world.add_relation::<ChildOf>(external, root);

        assert_eq!(despawn_instance(&mut world, &spawned), 1);
        assert!(!world.contains(root));
        assert!(world.contains(external));
        assert!(!world.has_relation::<ChildOf>(external, root));
    }

    #[test]
    fn despawn_removes_reserved_entities_owned_by_instance() {
        let mut world = World::new();
        let mut spawned = spawn_plan(&mut world, &plan(&[0], &[])).unwrap();
        let reserved = world.spawn(Marker(99));
        assert!(spawned.remap.set(LocalEntityId::reserved(0), reserved));

        assert_eq!(despawn_instance(&mut world, &spawned), 2);
        assert!(!world.contains(spawned.root.unwrap()));
        assert!(!world.contains(reserved));
    }

    #[test]
    fn constructor_failure_rolls_back_every_reserved_entity() {
        let mut world = World::new();
        let before = world.entity_count();
        let result = spawn_typed_plan(
            &mut world,
            &plan(&[0, 1, 2], &[(1, 0), (2, 0)]),
            TypedSpawnPlan::new()
                .with_bundle(LocalEntityId::new(0), Marker(0))
                .with_constructor(LocalEntityId::new(1), |_, _, _| {
                    Err(TypedSpawnError { code: 42 })
                })
                .with_bundle(LocalEntityId::new(2), Marker(2)),
        );
        assert_eq!(
            result,
            Err(InstantiateError::TypedSpawnFailed {
                local: LocalEntityId::new(1),
                code: 42
            })
        );
        assert_eq!(world.entity_count(), before);
        assert!(world.query::<&Marker>().iter(&world).next().is_none());
    }

    #[test]
    fn rejects_opaque_payload_before_mutating_world() {
        let mut payload_plan = plan(&[0], &[]);
        payload_plan.groups[0].signature =
            ArchetypeSignature::new(vec![ComponentTypeId::of::<Marker>()]);
        payload_plan.groups[0].columns = Box::new([crate::plan::Column::new(
            ComponentTypeId::of::<Marker>(),
            vec![0; size_of::<Marker>()].into_boxed_slice(),
        )]);
        let mut world = World::new();
        assert!(matches!(
            spawn_plan(&mut world, &payload_plan),
            Err(InstantiateError::TypedSpawnRequired { .. })
        ));
        assert_eq!(world.entity_count(), 0);
    }

    #[test]
    fn rejects_invalid_field_slot_before_mutating_world() {
        let mut value_plan = plan(&[0], &[]);
        value_plan.entity_valued_fields = Box::new([FieldSlot {
            local: LocalEntityId::new(0),
            group: 0,
            column: 0,
            byte_offset: 0,
        }]);
        let mut world = World::new();
        assert_eq!(
            spawn_plan(&mut world, &value_plan),
            Err(InstantiateError::InvalidPlan(
                PlanError::FieldSlotOutOfRange {
                    slot: value_plan.entity_valued_fields[0],
                }
            ))
        );
        assert_eq!(world.entity_count(), 0);
    }
}
