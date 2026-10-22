//! World-persistent scene ownership and explicit lifecycle operations.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use prism_ecs::resource::Resource;
use prism_world_persist::PersistentId;

use super::{
    despawn_instance, ResidencyBinding, SceneEcsHost, SceneEcsProvider, SceneSpawn, SectorId,
};

/// Kept private so callers cannot accidentally replace another host's tokens.
#[derive(Default)]
pub(super) struct SceneOwnership {
    pub(super) cells: BTreeMap<SectorId, SceneSpawn>,
    pub(super) retired_cells: Vec<(SectorId, SceneSpawn)>,
    pub(super) bindings: BTreeMap<u64, ResidencyBinding>,
    pub(super) persistence_objects: BTreeMap<SectorId, Vec<PersistentId>>,
}

impl Resource for SceneOwnership {}

impl<P: SceneEcsProvider> SceneEcsHost<'_, P> {
    pub(super) fn ownership(&self) -> &SceneOwnership {
        self.world.resource::<SceneOwnership>()
    }

    pub(super) fn ownership_mut(&mut self) -> &mut SceneOwnership {
        self.world.resource_mut::<SceneOwnership>()
    }

    pub(super) fn destroy_cell(&mut self, sector: SectorId, spawn: &SceneSpawn) {
        // Despawn hooks may unwind. Keep the original token and bindings in the
        // world until destruction returns, including during the hooks themselves.
        despawn_instance(self.world, spawn);
        let ownership = self.ownership_mut();
        ownership.cells.remove(&sector);
        // Preserve the persistent object list after eviction for later reload.
        ownership
            .bindings
            .retain(|_, binding| binding.sector != sector);
    }

    /// Retries teardown for old scene trees retained after a published reload.
    ///
    /// If an ECS despawn hook panics, the current and later tokens remain
    /// world-owned and this method may be called again after fixing the hook.
    /// Returns the number of successfully retired scenes.
    pub fn retry_retired_cells(&mut self) -> usize {
        let mut retired_count = 0;
        loop {
            let Some((_, spawn)) = self.ownership().retired_cells.last().cloned() else {
                return retired_count;
            };
            despawn_instance(self.world, &spawn);
            self.ownership_mut().retired_cells.pop();
            retired_count += 1;
        }
    }

    /// Despawns all registered cell scenes and clears every residency binding.
    ///
    /// Returns the number of removed cell ownership tokens. This uses the same
    /// scene despawn boundary as individual unloads, never provider callbacks or
    /// ignored provider errors. Repeated shutdowns return zero.
    ///
    /// Despawn hook panics propagate. Successfully destroyed cells stay removed;
    /// the interrupted cell and all pending cells retain their tokens and
    /// bindings for handoff or retry. Entity destruction is not rolled back.
    ///
    /// The caller must finish persistence handoff first and reconcile its world
    /// runtime's streaming state separately. This does not perform IO, update
    /// residency records, or clean up provider-owned external resources.
    pub fn shutdown(&mut self) -> usize {
        let retired = self.retry_retired_cells();
        let mut count = 0;
        while let Some((sector, spawn)) = self
            .ownership()
            .cells
            .first_key_value()
            .map(|(&sector, spawn)| (sector, spawn.clone()))
        {
            self.destroy_cell(sector, &spawn);
            count += 1;
        }
        // Clear bindings to absent cells only after the entire shutdown succeeds.
        self.ownership_mut().bindings.clear();
        count + retired
    }

    /// Releases the world borrow and returns provider state for explicit handoff.
    ///
    /// Registered cells and bindings stay in the world, just as on host drop;
    /// this does not unload scenes or implicitly shut down external services.
    #[must_use]
    pub fn into_provider(self) -> P {
        self.provider
    }
}
