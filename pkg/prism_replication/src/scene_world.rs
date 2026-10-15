//! Scene/world identity adapters over the real Prism scene and ECS APIs.
//!
//! `prism_world` owns streaming and residency policy. This module does not
//! duplicate that policy; it only resolves a stable `PersistentId` through a
//! scene remap and provides an explicit ownership-safe scene despawn boundary.

use core::fmt;

use prism_ecs::entity::Entity;
use prism_ecs::world::World;
use prism_scene::{
    despawn_instance, LocalEntityId, PersistentEntityMap, PersistentId, SceneInstance, SceneSpawn,
};

use crate::ecs::{EcsReplication, EcsReplicationError};
use crate::mapping::{AuthorityId, EntityId, Revision};

/// Resolve stable replication ids through a spawned scene's real remap table.
pub struct SceneEntityResolver<'a> {
    spawn: &'a SceneSpawn,
    identities: &'a PersistentEntityMap,
}

impl<'a> SceneEntityResolver<'a> {
    /// Bind a scene spawn to its persistent local-id map.
    #[must_use]
    pub const fn new(spawn: &'a SceneSpawn, identities: &'a PersistentEntityMap) -> Self {
        Self { spawn, identities }
    }

    /// Resolve a stable replication identity to the live ECS entity.
    #[must_use]
    pub fn resolve(&self, stable: EntityId) -> Option<Entity> {
        let persistent = PersistentId::from_u128(stable.as_u128());
        let local = self.identities.local_for_persistent(persistent)?;
        self.spawn.resolve(local)
    }

    /// Resolve by a scene-local identity, retaining the scene's tombstone
    /// semantics for removed structural-override entities.
    #[must_use]
    pub fn resolve_local(&self, local: LocalEntityId) -> Option<Entity> {
        self.spawn.resolve(local)
    }
}

/// Resolve stable replication ids through a live `SceneInstance` anchor.
pub struct SceneInstanceResolver<'a> {
    instance: &'a SceneInstance,
    identities: &'a PersistentEntityMap,
}

impl<'a> SceneInstanceResolver<'a> {
    /// Bind a live scene anchor to its persistent local-id map.
    #[must_use]
    pub const fn new(instance: &'a SceneInstance, identities: &'a PersistentEntityMap) -> Self {
        Self {
            instance,
            identities,
        }
    }

    /// Resolve a stable replication identity to the live ECS entity.
    #[must_use]
    pub fn resolve(&self, stable: EntityId) -> Option<Entity> {
        let persistent = PersistentId::from_u128(stable.as_u128());
        let local = self.identities.local_for_persistent(persistent)?;
        self.instance.resolve(local)
    }
}

/// Error from the ownership-aware scene/world adapter.
#[derive(Debug)]
pub enum SceneWorldReplicationError {
    /// The stable id is not present in the scene identity map or remap.
    UnknownSceneEntity(EntityId),
    /// The ECS replication provider rejected adoption or cleanup.
    Ecs(EcsReplicationError),
}

impl fmt::Display for SceneWorldReplicationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownSceneEntity(id) => {
                write!(f, "stable scene entity {} is not live", id.as_u128())
            }
            Self::Ecs(error) => write!(f, "scene ECS replication failed: {error}"),
        }
    }
}

impl core::error::Error for SceneWorldReplicationError {}

impl From<EcsReplicationError> for SceneWorldReplicationError {
    fn from(error: EcsReplicationError) -> Self {
        Self::Ecs(error)
    }
}

/// A narrow scene/world bridge. The scene owns entity membership; the ECS
/// provider owns reflected component mapping and values.
pub struct SceneWorldReplication<'a> {
    world: &'a mut World,
    spawn: &'a SceneSpawn,
    identities: &'a PersistentEntityMap,
}

impl<'a> SceneWorldReplication<'a> {
    /// Create a bridge over one live scene spawn and its authoritative world.
    #[must_use]
    pub const fn new(
        world: &'a mut World,
        spawn: &'a SceneSpawn,
        identities: &'a PersistentEntityMap,
    ) -> Self {
        Self {
            world,
            spawn,
            identities,
        }
    }

    /// Resolve one stable scene/entity id through the actual scene remap.
    #[must_use]
    pub fn resolve(&self, stable: EntityId) -> Option<Entity> {
        SceneEntityResolver::new(self.spawn, self.identities).resolve(stable)
    }

    /// Register all reflected components of an existing scene entity in the
    /// ECS replication mapping without spawning a duplicate entity.
    pub fn adopt(
        &mut self,
        replication: &mut EcsReplication,
        stable: EntityId,
        authority: AuthorityId,
        revision: Revision,
    ) -> Result<Entity, SceneWorldReplicationError> {
        let entity = self
            .resolve(stable)
            .ok_or(SceneWorldReplicationError::UnknownSceneEntity(stable))?;
        replication.adopt_entity(self.world, entity, stable, authority, revision)?;
        Ok(entity)
    }

    /// Despawn the complete scene instance through `prism_scene` and remove
    /// every identity known by the scene map from the replication mapping.
    pub fn despawn(self, replication: &mut EcsReplication) -> usize {
        let removed = despawn_instance(self.world, self.spawn);
        for binding in self.identities.bindings() {
            // `despawn_instance` invalidates the scene remap before returning.
            // Forget every identity unconditionally: the mapping removal is
            // idempotent, while checking `spawn.resolve` after the despawn
            // would leave all adopted mappings behind as stale entries.
            replication.forget_entity(EntityId::from_u128(binding.persistent.to_u128()));
        }
        removed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::boxed::Box;
    use alloc::vec;
    use prism_ecs::component::Component;
    use prism_reflect::Reflect;
    use prism_scene::{spawn_plan, ArchetypeGroup, ArchetypeSignature, InstantiationPlan};

    #[derive(Clone, Debug, PartialEq, Reflect)]
    struct ReplicatedTransform {
        x: i32,
    }
    impl Component for ReplicatedTransform {}

    fn spawn_one(world: &mut World) -> SceneSpawn {
        let local = LocalEntityId::new(0);
        let plan = InstantiationPlan {
            entity_count: 1,
            groups: Box::new([ArchetypeGroup::new(
                ArchetypeSignature::default(),
                Box::new([local]),
                Box::new([]),
            )]),
            entity_valued_fields: Box::new([]),
            hierarchy: Box::new([]),
            reserved_local_base: LocalEntityId::RESERVED_BASE,
        };
        spawn_plan(world, &plan).unwrap()
    }

    #[test]
    fn scene_spawn_resolves_and_despawns_through_real_scene_api() {
        let mut world = World::new();
        let spawn = spawn_one(&mut world);
        let persistent = PersistentId::from_u128(77);
        let identities = PersistentEntityMap::new(vec![prism_scene::EntityBinding::new(
            LocalEntityId::new(0),
            persistent,
        )])
        .unwrap();
        let stable = EntityId::from_u128(persistent.to_u128());
        let entity = SceneEntityResolver::new(&spawn, &identities)
            .resolve(stable)
            .unwrap();
        assert!(world.contains(entity));
        assert_eq!(despawn_instance(&mut world, &spawn), 1);
        assert!(!world.contains(entity));
    }

    #[test]
    fn scene_despawn_forgets_adopted_replication_mappings_after_remap_invalidation() {
        let mut world = World::new();
        let spawn = spawn_one(&mut world);
        let persistent = PersistentId::from_u128(88);
        let identities = PersistentEntityMap::new(vec![prism_scene::EntityBinding::new(
            LocalEntityId::new(0),
            persistent,
        )])
        .unwrap();
        let stable = EntityId::from_u128(persistent.to_u128());
        let entity = SceneEntityResolver::new(&spawn, &identities)
            .resolve(stable)
            .unwrap();
        assert!(world.insert(entity, ReplicatedTransform { x: 4 }));

        let key = crate::mapping::ComponentKey::new(crate::mapping::SchemaId::from_u128(9), 1);
        let mut replication = EcsReplication::new(
            crate::ecs::EcsReplicationLimits::default(),
            crate::mapping::MappingLimits::new(2, 8),
        );
        replication
            .register_component::<ReplicatedTransform>(&mut world, key, 1)
            .unwrap();
        let mut bridge = SceneWorldReplication::new(&mut world, &spawn, &identities);
        bridge
            .adopt(
                &mut replication,
                stable,
                AuthorityId::from_u128(3),
                Revision::new(1),
            )
            .unwrap();
        assert_eq!(replication.mapping().entry_len(), 1);

        assert_eq!(bridge.despawn(&mut replication), 1);
        assert_eq!(replication.mapping().entry_len(), 0);
        assert!(!world.contains(entity));
    }
}
