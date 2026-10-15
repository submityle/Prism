//! High-level scene runtime entry points.
//!
//! `spawn_typed_plan` is intentionally low-level: it only owns reservation and
//! typed materialization. This module adds the public scene transaction that
//! callers need in production. Template validation, override validation,
//! anchor installation, override replay, and rollback are one operation, so a
//! failed scene load cannot leave a partially visible instance in the world.

use crate::instantiate::{
    despawn_instance, spawn_typed_plan, InstantiateError, SceneAttachError, SceneSpawn,
    TypedSpawnPlan,
};
use crate::override_replay::OverrideReplayError;
use crate::overrides::{OverrideError, OverrideSet};
use crate::residency::ResidencyTier;
use crate::template::{SceneTemplate, TemplateError};
use alloc::boxed::Box;
use prism_ecs::world::World;

/// Failure while loading a complete scene instance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SceneSpawnError {
    /// The template is not internally consistent.
    Template(TemplateError),
    /// The sparse override set does not match the template plan.
    Overrides(OverrideError),
    /// Reservation or typed materialization failed.
    Instantiate(InstantiateError),
    /// The root anchor could not be installed.
    Attach(SceneAttachError),
    /// The override transaction failed after the entities were materialized.
    Replay(OverrideReplayError),
}

/// Spawns a complete scene transaction.
///
/// The typed plan is the safe bridge from serialized scene data to concrete
/// ECS components. `overrides` is validated against the same template plan
/// before any entity is reserved. If anchor installation or override replay
/// fails, every entity owned by this call is despawned before the error is
/// returned. On success, the root anchor retains a clone of the sparse delta.
pub fn spawn_scene(
    world: &mut World,
    template: &SceneTemplate,
    typed: TypedSpawnPlan,
    overrides: Option<&OverrideSet>,
    tier: ResidencyTier,
) -> Result<SceneSpawn, SceneSpawnError> {
    template.validate().map_err(SceneSpawnError::Template)?;
    if let Some(overrides) = overrides {
        overrides
            .validate_for_plan(&template.plan)
            .map_err(SceneSpawnError::Overrides)?;
    }

    let spawn = spawn_typed_plan(world, &template.plan, typed)
        .map_err(SceneSpawnError::Instantiate)?;
    if let Err(error) = spawn.attach_instance(world, template.guid, tier) {
        despawn_instance(world, &spawn);
        return Err(SceneSpawnError::Attach(error));
    }

    if let Some(overrides) = overrides.filter(|value| !value.is_empty()) {
        if let Err(error) = crate::override_replay::replay_overrides(world, &spawn, overrides) {
            despawn_instance(world, &spawn);
            return Err(SceneSpawnError::Replay(error));
        }
        if let Some(root) = spawn.root {
            if let Some(mut instance) = world.get_mut::<crate::SceneInstance>(root) {
                instance.overrides = Some(Box::new(overrides.clone()));
            }
        }
    }
    Ok(spawn)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::LocalEntityId;
    use crate::plan::{ArchetypeGroup, InstantiationPlan};
    use crate::overrides::ArchetypeSignature;
    use alloc::boxed::Box;
    use prism_asset::StableGuid;

    fn template() -> SceneTemplate {
        SceneTemplate::new(
            StableGuid::from_u128(0x55),
            InstantiationPlan {
                entity_count: 1,
                groups: Box::new([ArchetypeGroup::new(
                    ArchetypeSignature::default(),
                    Box::new([LocalEntityId::new(0)]),
                    Box::new([]),
                )]),
                entity_valued_fields: Box::new([]),
                hierarchy: Box::new([]),
                reserved_local_base: LocalEntityId::RESERVED_BASE,
            },
        )
    }

    #[test]
    fn empty_scene_transaction_installs_and_cleans_up_anchor() {
        let mut world = World::new();
        let scene = template();
        let spawn = spawn_scene(
            &mut world,
            &scene,
            TypedSpawnPlan::new().with_bundle(LocalEntityId::new(0), ()),
            None,
            ResidencyTier::Interactive,
        )
        .unwrap();
        let root = spawn.root.unwrap();
        assert_eq!(world.get::<crate::SceneInstance>(root).unwrap().template, scene.guid);
        assert_eq!(crate::despawn_instance(&mut world, &spawn), 1);
        assert_eq!(world.entity_count(), 0);
    }
}
