//! Transactional replay of live scene overrides.
//!
//! This first runtime slice deliberately covers reflected property overrides only.
//! Structural operations are rejected until a component decoder and an ECS undo
//! contract are supplied; accepting them here would silently lose serialized data.

use alloc::boxed::Box;
use alloc::vec::Vec;

use crate::instantiate::SceneSpawn;
use crate::overrides::{ComponentTypeId, OverrideSet, PropertyOverride};
use prism_ecs::entity::Entity;
use prism_ecs::world::World;
use prism_reflect::{Reflect, StableTypeId};

/// Failure while replaying a live override set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OverrideReplayError {
    /// Structural operations need a typed decoder and transactional ECS support.
    StructuralOperationsUnsupported,
    /// A property target is not owned by the scene spawn.
    TargetNotOwned(crate::identity::LocalEntityId),
    /// The target does not carry the reflected component.
    MissingComponent {
        /// The live target entity.
        entity: Entity,
        /// The component requested by the override.
        component: ComponentTypeId,
    },
    /// The property value/path could not be applied to its staged component.
    Property(crate::property_apply::PropertyApplyError),
    /// The reflected component rejected the staged value during commit.
    WriteRejected {
        /// The live target entity.
        entity: Entity,
        /// The component whose write was rejected.
        component: ComponentTypeId,
    },
    /// Restoring a previously committed component failed.
    RollbackFailed {
        /// The live target entity whose original value could not be restored.
        entity: Entity,
        /// The component whose original value could not be restored.
        component: ComponentTypeId,
    },
}

struct StagedComponent {
    entity: Entity,
    component: ComponentTypeId,
    original: Box<dyn Reflect>,
    properties: Vec<PropertyOverride>,
    staged: Option<Box<dyn Reflect>>,
}

/// Replays all reflected property overrides in one transaction.
///
/// Every target is resolved through `spawn` and every property group is checked
/// against a cloned reflected value before the first ECS write. If a later write
/// is rejected, already-written components are restored from their original
/// clones. Structural operations are rejected before mutation because this API
/// does not yet have a safe serialized-component decoder. ECS change ticks and
/// observer side effects are not rolled back by the field-value restoration.
pub fn replay_overrides(
    world: &mut World,
    spawn: &SceneSpawn,
    overrides: &OverrideSet,
) -> Result<(), OverrideReplayError> {
    if !overrides.structural().is_empty() {
        return Err(OverrideReplayError::StructuralOperationsUnsupported);
    }

    let mut staged = Vec::<StagedComponent>::new();
    for property in overrides.properties() {
        let entity = spawn
            .resolve(property.target)
            .ok_or(OverrideReplayError::TargetNotOwned(property.target))?;
        let Some(current) =
            world.get_reflect(entity, StableTypeId::from_raw(property.component.raw()))
        else {
            return Err(OverrideReplayError::MissingComponent {
                entity,
                component: property.component,
            });
        };
        let Some(group) = staged
            .iter_mut()
            .find(|group| group.entity == entity && group.component == property.component)
        else {
            staged.push(StagedComponent {
                entity,
                component: property.component,
                original: current.reflect_clone(),
                properties: vec![property.clone()],
                staged: None,
            });
            continue;
        };
        group.properties.push(property.clone());
    }

    // Validate every group against an isolated clone before touching the world.
    for group in &mut staged {
        let Some(current) =
            world.get_reflect(group.entity, StableTypeId::from_raw(group.component.raw()))
        else {
            return Err(OverrideReplayError::MissingComponent {
                entity: group.entity,
                component: group.component,
            });
        };
        let mut probe = current.reflect_clone();
        for property in &group.properties {
            crate::property_apply::apply_property_to_reflect_for_replay(probe.as_mut(), property)
                .map_err(OverrideReplayError::Property)?;
        }
        group.staged = Some(probe);
    }

    for index in 0..staged.len() {
        let (entity, component, staged_value) = {
            let group = &mut staged[index];
            (
                group.entity,
                group.component,
                group.staged.take().expect("validated staged value"),
            )
        };
        if !world.insert_reflect(entity, staged_value.as_ref()) {
            for previous in staged.iter().take(index) {
                if !world.insert_reflect(previous.entity, previous.original.as_ref()) {
                    return Err(OverrideReplayError::RollbackFailed {
                        entity: previous.entity,
                        component: previous.component,
                    });
                }
            }
            return Err(OverrideReplayError::WriteRejected { entity, component });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::LocalEntityId;
    use crate::instantiate::{spawn_typed_plan, TypedSpawnPlan};
    use crate::overrides::OverrideValue;
    use crate::path::FieldPath;
    use crate::plan::{ArchetypeGroup, InstantiationPlan};
    use prism_ecs::component::Component;
    use prism_reflect::Reflect;

    #[derive(Clone, Debug, Reflect, PartialEq)]
    struct Settings {
        speed: u64,
        label: String,
    }
    impl Component for Settings {}

    fn plan(count: u32) -> InstantiationPlan {
        InstantiationPlan {
            entity_count: count,
            groups: if count == 0 {
                Box::new([])
            } else {
                Box::new([ArchetypeGroup::new(
                    Default::default(),
                    (0..count).map(LocalEntityId::new).collect(),
                    Box::new([]),
                )])
            },
            entity_valued_fields: Box::new([]),
            hierarchy: (1..count)
                .map(|local| (LocalEntityId::new(local), LocalEntityId::new(0)))
                .collect(),
            reserved_local_base: LocalEntityId::RESERVED_BASE,
        }
    }

    fn property(target: u32, path: &str, value: OverrideValue) -> PropertyOverride {
        PropertyOverride::new(
            LocalEntityId::new(target),
            ComponentTypeId::of::<Settings>(),
            FieldPath::parse(path).unwrap(),
            value,
        )
    }

    #[test]
    fn replays_properties_through_instance_remap() {
        let mut world = World::new();
        world.register_reflect_component::<Settings>();
        let spawn = spawn_typed_plan(
            &mut world,
            &plan(1),
            TypedSpawnPlan::new().with_bundle(
                LocalEntityId::new(0),
                Settings {
                    speed: 2,
                    label: "old".into(),
                },
            ),
        )
        .unwrap();
        let mut overrides = OverrideSet::new();
        overrides.set_property(property(0, "speed", OverrideValue::U64(9)));
        overrides.set_property(property(0, "label", OverrideValue::Str("new".into())));

        replay_overrides(&mut world, &spawn, &overrides).unwrap();
        assert_eq!(
            world
                .get::<Settings>(spawn.resolve(LocalEntityId::new(0)).unwrap())
                .unwrap(),
            &Settings {
                speed: 9,
                label: "new".into()
            }
        );
    }

    #[test]
    fn invalid_later_group_leaves_earlier_group_untouched() {
        let mut world = World::new();
        world.register_reflect_component::<Settings>();
        let spawn = spawn_typed_plan(
            &mut world,
            &plan(2),
            TypedSpawnPlan::new()
                .with_bundle(
                    LocalEntityId::new(0),
                    Settings {
                        speed: 2,
                        label: "a".into(),
                    },
                )
                .with_bundle(
                    LocalEntityId::new(1),
                    Settings {
                        speed: 3,
                        label: "b".into(),
                    },
                ),
        )
        .unwrap();
        let mut overrides = OverrideSet::new();
        overrides.set_property(property(0, "speed", OverrideValue::U64(9)));
        overrides.set_property(property(1, "speed", OverrideValue::I64(9)));

        assert!(matches!(
            replay_overrides(&mut world, &spawn, &overrides),
            Err(OverrideReplayError::Property(_))
        ));
        assert_eq!(
            world
                .get::<Settings>(spawn.resolve(LocalEntityId::new(0)).unwrap())
                .unwrap()
                .speed,
            2
        );
    }
}
