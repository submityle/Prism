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

type SpawnFn =
    Box<dyn FnOnce(&mut World, Entity, &RemapTable) -> Result<(), TypedSpawnError> + Send + Sync>;

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
            Box::new(move |world, entity, _| {
                world.spawn_at(entity, bundle);
                Ok(())
            }),
        ));
        self
    }

    /// Adds a constructor with access to the complete remap.
    #[must_use]
    pub fn with_constructor(
        mut self,
        local: LocalEntityId,
        constructor: impl FnOnce(&mut World, Entity, &RemapTable) -> Result<(), TypedSpawnError>
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
    /// The independent local-to-world mapping.
    pub remap: RemapTable,
}

impl SceneSpawn {
    /// Builds the instance anchor once the caller supplies the template identity
    /// that is intentionally absent from `InstantiationPlan`.
    #[must_use]
    pub fn scene_instance(&self, template: StableGuid, tier: ResidencyTier) -> SceneInstance {
        SceneInstance::new(template, self.remap.clone(), tier)
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
    /// Constructor failed; all reserved instance entities were rolled back.
    TypedSpawnFailed {
        /// Local entity whose constructor failed.
        local: LocalEntityId,
        /// Caller-defined constructor error code.
        code: u32,
    },
}

fn validate_hierarchy(plan: &InstantiationPlan) -> Result<(), InstantiateError> {
    let mut parents = Vec::with_capacity(plan.hierarchy.len());
    for &(child, parent) in plan.hierarchy.iter() {
        if parents.iter().any(|(seen, _)| *seen == child) {
            return Err(InstantiateError::MultipleParents { child });
        }
        parents.push((child, parent));
    }
    for &(start, _) in parents.iter() {
        let mut cursor = start;
        let mut steps = 0usize;
        while let Some((_, parent)) = parents.iter().find(|(child, _)| *child == cursor) {
            cursor = *parent;
            steps += 1;
            if cursor == start || steps > parents.len() {
                return Err(InstantiateError::HierarchyCycle { local: start });
            }
        }
    }
    Ok(())
}

/// Spawns a plan with no opaque component payloads or entity-reference slots.
pub fn spawn_plan(
    world: &mut World,
    plan: &InstantiationPlan,
) -> Result<SceneSpawn, InstantiateError> {
    plan.validate().map_err(InstantiateError::InvalidPlan)?;
    validate_hierarchy(plan)?;
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
            Box::new(|world, entity, _| {
                world.spawn_at(entity, ());
                Ok(())
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
    validate_hierarchy(plan)?;
    let locals: Vec<_> = plan
        .groups
        .iter()
        .flat_map(|group| group.locals.iter().copied())
        .collect();
    if typed.entries.len() != locals.len() {
        let local = locals
            .iter()
            .copied()
            .find(|id| !typed.entries.iter().any(|(have, _)| have == id))
            .or_else(|| {
                typed
                    .entries
                    .iter()
                    .map(|(id, _)| *id)
                    .find(|id| !locals.contains(id))
            })
            .unwrap_or(LocalEntityId::new(0));
        return Err(InstantiateError::TypedLocalsMismatch { local });
    }
    for &local in &locals {
        if typed.entries.iter().filter(|(id, _)| *id == local).count() != 1 {
            return Err(InstantiateError::TypedLocalsMismatch { local });
        }
    }
    if !plan.entity_valued_fields.is_empty() && typed.entries.is_empty() {
        return Err(InstantiateError::TypedFieldRemapRequired);
    }

    let mut remap = RemapTable::with_base_capacity(plan.entity_count as usize);
    for _ in 0..plan.entity_count {
        remap.push_base(world.entities().reserve_entity());
    }
    world.flush_reserved();

    for &local in &locals {
        let entity = remap.get(local).expect("validated plan local has mapping");
        let index = typed
            .entries
            .iter()
            .position(|(id, _)| *id == local)
            .expect("prevalidated typed local");
        let (_, constructor) = typed.entries.swap_remove(index);
        if let Err(error) = constructor(world, entity, &remap) {
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
        world.add_relation::<ChildOf>(
            remap.get(child).expect("validated child"),
            remap.get(parent).expect("validated parent"),
        );
    }

    let root = remap.get(LocalEntityId::new(0));
    Ok(SceneSpawn { root, remap })
}

fn rollback(world: &mut World, remap: &RemapTable, plan: &InstantiationPlan) {
    for &(child, parent) in plan.hierarchy.iter() {
        if let (Some(child), Some(parent)) = (remap.get(child), remap.get(parent)) {
            world.remove_relation::<ChildOf>(child, parent);
        }
    }
    for (_, entity) in remap.base_entities() {
        world.despawn(entity);
    }
}

/// Despawns exactly the entities owned by the instance.
pub fn despawn_instance(world: &mut World, spawn: &SceneSpawn) -> usize {
    for (_, child) in spawn.remap.base_entities() {
        let parents: Vec<_> = world.relation_targets::<ChildOf>(child).to_vec();
        for parent in parents {
            world.remove_relation::<ChildOf>(child, parent);
        }
    }
    // ChildOf deletes children when its target is destroyed. Detach external
    // children first so this operation cannot cascade outside the instance.
    for (_, parent) in spawn.remap.base_entities() {
        let children: Vec<_> = world.relation_sources::<ChildOf>(parent).to_vec();
        for child in children {
            world.remove_relation::<ChildOf>(child, parent);
        }
    }
    let mut removed = 0;
    for (_, entity) in spawn.remap.base_entities() {
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
            hierarchy: hierarchy
                .iter()
                .map(|&(c, p)| (LocalEntityId::new(c), LocalEntityId::new(p)))
                .collect(),
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
        let root = spawned.remap.get(LocalEntityId::new(0)).unwrap();
        let child = spawned.remap.get(LocalEntityId::new(1)).unwrap();
        let grandchild = spawned.remap.get(LocalEntityId::new(2)).unwrap();
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
        let a0 = a.remap.get(LocalEntityId::new(0)).unwrap();
        let b0 = b.remap.get(LocalEntityId::new(0)).unwrap();
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
                .with_constructor(LocalEntityId::new(0), |world, entity, remap| {
                    let target = remap.get(LocalEntityId::new(1)).unwrap();
                    world.spawn_at(entity, Link(target));
                    Ok(())
                })
                .with_bundle(LocalEntityId::new(1), Marker(7)),
        )
        .unwrap();
        let source = spawned.remap.get(LocalEntityId::new(0)).unwrap();
        let target = spawned.remap.get(LocalEntityId::new(1)).unwrap();
        assert_eq!(world.get::<Link>(source), Some(&Link(target)));
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
    fn constructor_failure_rolls_back_every_reserved_entity() {
        let mut world = World::new();
        let before = world.entity_count();
        let result = spawn_typed_plan(
            &mut world,
            &plan(&[0, 1, 2], &[(1, 0)]),
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
