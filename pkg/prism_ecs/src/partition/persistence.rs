//! Storage-independent inspection for scene-owned durable cell streaming.
//! The kernel exposes membership, never filesystem paths or archive policy.

use alloc::vec::Vec;

use crate::{component::ComponentId, entity::Entity, world::World};

/// Exact concrete component membership, including sparse and shared storage.
/// Returns `None` for missing or not-yet-materialized reserved entities.
/// Reflection-only enumeration is insufficient: it silently omits values for
/// which the application forgot to register a serializer.
pub fn entity_component_ids(world: &World, entity: Entity) -> Option<Vec<ComponentId>> {
    let location = world.entities().location(entity)?;
    if location.is_empty() {
        return None;
    }
    let archetype = world.archetypes().get(location.archetype_id)?;
    let mut ids = archetype.components().ids().to_vec();
    ids.extend(world.sparse_sets().ids_for(entity));
    ids.extend(archetype.shared_bindings().iter().map(|binding| binding.0));
    ids.sort_unstable();
    ids.dedup();
    Some(ids)
}

/// Whether arbitrary application lifecycle code could run during archive
/// insertion/despawn. Persistent owners can reject this configuration before
/// starting a transaction rather than claim to undo arbitrary hook effects.
/// This deliberately conservative world-wide gate includes change observers.
pub fn has_lifecycle_callbacks(world: &World) -> bool {
    world.components().has_hooks()
        || !world.observers.is_empty()
        || !world.change_observers.is_empty()
}
