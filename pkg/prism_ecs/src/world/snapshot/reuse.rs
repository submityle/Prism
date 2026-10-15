//! Explicit value semantics and storage provenance for immutable capture buffers.

use alloc::vec::Vec;
use core::alloc::Layout;
use core::any::TypeId;

use crate::change::Tick;
use crate::collections::HashMap;
use crate::component::{Component, ComponentId};
use crate::entity::Entity;
use crate::resource::Resource;
use crate::world::World;

/// A component whose cloned values can be shared by immutable snapshots.
///
/// Register with [`World::register_snapshot_component_reusable`]. Ordinary
/// Clone-only components retain conservative, independently cloned storage.
///
/// # Safety
/// A clone must preserve the component's value and isolate it from subsequent
/// mutation of the source. No observable part of a captured value may change
/// through shared references or external aliases, including during Clone,
/// PartialEq, Hash or Drop. Owned String/Vec fields and immutable Arc payloads
/// are suitable; Arc<Mutex<_>>, atomics used as component state, and pointers
/// to externally mutable state are not. Diagnostic clone counters external to
/// the value are allowed, but must not affect its value. All live value changes
/// must use change-tracked ECS writes (raw callers must stamp change metadata).
pub unsafe trait SnapshotValue: Component + Clone {}

#[derive(Default)]
struct ReusableComponents(HashMap<ComponentId, (TypeId, Layout, u64)>);

impl Resource for ReusableComponents {}

impl World {
    /// Enable immutable chunk buffers for a component with explicit value
    /// semantics. This does not infer immutability from Clone, Hash or Eq.
    /// Re-register after runtime schema replacement to renew the contract.
    pub fn register_snapshot_component_reusable<T: SnapshotValue>(&mut self) -> ComponentId {
        let id = self.register_snapshot_component::<T>();
        let generation = self.components.schema_generation();
        self.resources.init::<ReusableComponents>();
        self.resources
            .get_mut::<ReusableComponents>()
            .expect("reuse registry was initialized")
            .0
            .insert(id, (TypeId::of::<T>(), Layout::new::<T>(), generation));
        id
    }
}

pub(super) fn reusable(world: &World, id: ComponentId) -> bool {
    let Some(registry) = world.resources.get::<ReusableComponents>() else {
        return false;
    };
    let Some(&(type_id, layout, generation)) = registry.0.get(&id) else {
        return false;
    };
    world.components.info(id).is_some_and(|info| {
        info.type_id() == Some(type_id)
            && info.layout() == layout
            && world.components.schema_generation() == generation
    })
}

/// Keys contain metadata only, never value bytes, hashes or mutable pointers.
#[derive(Clone, PartialEq, Eq, Hash)]
pub(super) enum ChunkKey {
    Table {
        incarnation: u64,
        structural_epoch: u64,
        version: Tick,
        revision: u64,
        rows: usize,
    },
    // Sparse storage has no coarse version. Scan membership, ticks and
    // independent insertion/write identities, never clean value payloads.
    Sparse {
        index: usize,
        entities: Vec<Entity>,
        added: Vec<Tick>,
        changed: Vec<Tick>,
        revisions: Vec<(u64, u64)>,
    },
}

impl ChunkKey {
    fn written_at(&self, tick: Tick) -> bool {
        match self {
            Self::Table { version, .. } => *version == tick,
            Self::Sparse { changed, .. } => changed.contains(&tick),
        }
    }
}

pub(super) struct ChunkOrigin {
    pub(super) key: ChunkKey,
    pub(super) archetype_generation: u64,
    pub(super) capture_tick: Tick,
    /// Incremental capture advances the world tick before releasing its unique
    /// borrow. Subsequent writes therefore cannot alias the captured tick.
    pub(super) sealed: bool,
}

impl ChunkOrigin {
    pub(super) fn reusable_in(&self, world: &World) -> bool {
        self.archetype_generation == world.archetype_generation()
            && self.capture_tick.age_since(world.change_tick()) < Tick::MAX_CHANGE_AGE
            // A read-only capture cannot seal a hot chunk: another write in
            // that same tick would keep its version unchanged. Cold chunks are
            // safe, and uniquely borrowed incremental captures seal all chunks.
            && (self.sealed || !self.key.written_at(self.capture_tick))
    }
}
