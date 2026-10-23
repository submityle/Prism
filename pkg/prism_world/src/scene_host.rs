//! Feature-gated scene/ECS application boundary for [`crate::WorldRuntime`].
//!
//! The policy runtime does not know scene templates, typed component
//! constructors, or how a residency id maps to a template-local entity. This
//! module supplies those runtime facts without weakening scene ownership:
//!
//! - a provider creates cells through a typed `prism_scene` entry point;
//! - cell removal always goes through [`prism_scene::despawn_instance`];
//! - tier changes use explicit typed callbacks because `prism_scene` has no
//!   generic inverse materialization API;
//! - an unbound or unloaded residency target is rejected instead of silently
//!   mutating an unrelated ECS entity.

use crate::{ResidencyBatchError, ResidencyTransition};
use alloc::collections::BTreeSet;
use alloc::string::ToString;
use alloc::vec::Vec;
use prism_ecs::entity::Entity;
use prism_ecs::world::World;
use prism_scene::{despawn_instance, LocalEntityId, SceneInstance, SceneSpawn};
use prism_world_persist::{
    AppendOnlyLog, FileJournal, LogError, ObjectState, PersistentId as WorldPersistentId, Snapshot,
    Timeline, TimelineError, Transaction, WorldState,
};
use prism_world_stream::SectorId;

mod ownership;
mod persistence_bridge;
mod plan_application;

use ownership::SceneOwnership;
pub use persistence_bridge::{
    SceneEntityMap, SceneFieldBinding, SceneFieldValueKind, SceneObjectBinding,
    ScenePersistenceBatch, ScenePersistenceBridgeError, ScenePersistenceSchema,
};

#[cfg(test)]
mod regression_tests;

/// The scene-local target associated with one world residency id.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResidencyBinding {
    /// Cell containing the target scene instance.
    pub sector: SectorId,
    /// Template-local entity resolved through the live scene spawn.
    pub local: LocalEntityId,
}

/// Persisted state for one world object owned by a cell.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CellPersistenceState {
    /// Explicit world-persistence identity supplied by the provider.
    pub object: WorldPersistentId,
    /// State reconstructed from the append-only world log.
    pub state: ObjectState,
}

/// Provider-captured scene state and explicit persistence identities for one
/// atomic sparse-property plus PRSI companion journal batch.
#[derive(Clone, Debug, PartialEq)]
pub struct ScenePersistenceCapture {
    /// Schema-owned conversion from scene fields to world fields.
    pub schema: ScenePersistenceSchema,
    /// Canonical scene snapshot diff against the supplied prior state.
    pub batch: ScenePersistenceBatch,
    /// Author for every transaction in this capture.
    pub actor: prism_world_persist::ActorId,
    /// Monotonic logical clock for every transaction in this capture.
    pub clock: prism_world_persist::LogicalClock,
    /// IDs in canonical object-delta order.
    pub transaction_ids: Vec<prism_world_persist::TransactionId>,
    /// Stable identity for the PRSI companion transaction.
    pub companion_transaction_id: prism_world_persist::TransactionId,
    /// Explicit world object storing the PRSI companion.
    pub companion_object: WorldPersistentId,
    /// Explicit opaque-byte field storing the PRSI companion.
    pub companion_field: prism_world_persist::FieldId,
}

/// Fully resolved scene target passed to a provider residency batch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SceneResidencyTransition {
    /// Stable world residency identity.
    pub residency_id: u64,
    /// Cell owning the target.
    pub sector: SectorId,
    /// Template-local target identity.
    pub local: LocalEntityId,
    /// Current ECS entity for the target.
    pub entity: Entity,
    /// Committed source tier.
    pub from: prism_world_residency::ResidencyState,
    /// Requested destination tier.
    pub to: prism_world_residency::ResidencyState,
}

/// Typed scene/ECS operations required by the world host.
pub trait SceneEcsProvider {
    /// Provider-specific failure type.
    type Error;

    /// Creates one cell scene and returns its ownership handle.
    ///
    /// Implementations must call a legal `prism_scene` spawn entry point such
    /// as `spawn_scene` or `spawn_typed_plan`, attach the returned instance
    /// anchor, and return the resulting [`SceneSpawn`].
    fn spawn_cell(
        &mut self,
        world: &mut World,
        sector: SectorId,
    ) -> Result<SceneSpawn, Self::Error>;

    /// Builds a staged replacement for an already-loaded cell.
    ///
    /// The provider must not mutate or destroy `previous`. It must build the
    /// replacement from the current template, apply applicable instance
    /// overrides to its ECS components, and return matching instance metadata.
    /// The host applies typed persistence state separately through
    /// `replay_cell_persistence`. The host preserves the old sparse override
    /// set as the authoritative source of truth, but cannot apply it itself
    /// because component decoders/registries are provider-specific. If staging
    /// fails, the provider must clean up every entity and external resource
    /// created during staging. The host validates the returned anchor and all
    /// current residency bindings before publishing it.
    fn stage_cell_reload(
        &mut self,
        world: &mut World,
        sector: SectorId,
        previous: &SceneSpawn,
        previous_instance: &SceneInstance,
    ) -> Result<SceneSpawn, Self::Error>;

    /// Transitions a bound target between typed scene representations.
    ///
    /// The provider receives both tiers because it owns the representation
    /// schema and must distinguish direct transitions such as proxy to
    /// interactive from transitions through the streamed tier.
    fn transition_residency(
        &mut self,
        world: &mut World,
        residency_id: u64,
        sector: SectorId,
        local: LocalEntityId,
        entity: Entity,
        from: prism_world_residency::ResidencyState,
        to: prism_world_residency::ResidencyState,
    ) -> Result<(), Self::Error>;

    /// Applies resolved transitions in order and reports the successful prefix.
    /// Providers with a real batch materialization path may override this; the
    /// default preserves the single-transition contract.
    fn transition_residency_batch(
        &mut self,
        world: &mut World,
        transitions: &[SceneResidencyTransition],
    ) -> Result<(), (usize, Self::Error)> {
        for (completed, transition) in transitions.iter().enumerate() {
            if let Err(error) = self.transition_residency(
                world,
                transition.residency_id,
                transition.sector,
                transition.local,
                transition.entity,
                transition.from,
                transition.to,
            ) {
                return Err((completed, error));
            }
        }
        Ok(())
    }

    /// Returns every world object ID owned by this cell, including any object
    /// that stores its PRSI companion. Scene IDs and world persistence IDs are
    /// separate domains; providers make the mapping explicit here rather than
    /// relying on representation coincidence. This list seeds load/reload
    /// state before the first eviction, including after process restart.
    fn persistence_objects(
        &mut self,
        _sector: SectorId,
    ) -> Result<Vec<WorldPersistentId>, Self::Error> {
        Ok(Vec::new())
    }

    /// Captures scene-native persistence data for host-owned transaction
    /// construction. The host computes exact inverses and appends property plus
    /// companion as one batch. `prior` is replayed from the same authoritative
    /// log being appended, and `next_sequence` is that log's next sequence.
    fn capture_scene_persistence(
        &mut self,
        world: &World,
        sector: SectorId,
        prior: &WorldState,
        next_sequence: u64,
    ) -> Result<ScenePersistenceCapture, Self::Error>;

    /// Applies reconstructed world state to a newly spawned cell before its
    /// ownership is published. Mapping fields into scene overrides/components
    /// remains the provider's typed responsibility. This is required so a
    /// provider cannot silently accept a persisted load without restoring it.
    fn replay_cell_persistence(
        &mut self,
        world: &mut World,
        sector: SectorId,
        states: &[CellPersistenceState],
    ) -> Result<(), Self::Error>;
}

/// Configuration errors for stable residency-to-scene bindings.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SceneEcsHostConfigError {
    /// The id is already bound to a different target.
    ConflictingBinding {
        /// Stable residency id supplied by the caller.
        residency_id: u64,
        /// Existing binding retained by the host.
        existing: ResidencyBinding,
        /// New binding rejected by the host.
        requested: ResidencyBinding,
    },
}

/// Runtime failure while applying a world operation to the scene/ECS host.
#[derive(Debug, Eq, PartialEq)]
pub enum SceneEcsHostError<E> {
    /// A batch reload listed the same sector more than once.
    DuplicateReloadSector,
    /// The provider rejected a typed operation.
    Provider(E),
    /// A provider returned a spawn without a live scene anchor.
    SceneNotAttached(SectorId),
    /// A load would replace an existing scene ownership token.
    CellAlreadyLoaded(SectorId),
    /// A staged reload did not return a distinct, anchored replacement.
    InvalidReloadReplacement(SectorId),
    /// A staged replacement aliases one or more entities from the live scene.
    AliasedReloadReplacement(SectorId),
    /// A current residency binding could not be resolved in the staged scene.
    ReloadBindingUnavailable {
        /// World residency id whose target would be lost.
        residency_id: u64,
        /// Cell being reloaded.
        sector: SectorId,
        /// Template-local entity from the existing binding.
        local: LocalEntityId,
    },
    /// An unload or residency operation addressed a cell that is not loaded.
    CellNotLoaded(SectorId),
    /// A residency operation has no stable scene binding.
    ResidencyNotBound(u64),
    /// A binding refers to a local entity removed from its scene instance.
    LocalEntityUnavailable {
        /// Stable residency id supplied by the world plan.
        residency_id: u64,
        /// Cell containing the binding.
        sector: SectorId,
        /// Template-local entity that could not be resolved.
        local: LocalEntityId,
    },
    /// Persistence rejected the complete cell transaction batch.
    PersistenceAppend(LogError),
    /// Durable journal rejected or failed the complete cell transaction batch.
    DurableAppend(alloc::string::String),
    /// The world persistence log could not be replayed for a cell.
    PersistenceReplay(TimelineError),
    /// Scene-native persistence data could not be converted into transactions.
    PersistenceBridge(ScenePersistenceBridgeError),
}

/// A scene-owned ECS host implementing [`crate::SceneHost`].
///
/// Cell ownership and bindings live in the authoritative world's private
/// resource, not in this temporary borrow. Dropping the host leaves scenes
/// registered for takeover by a new host with a compatible provider. Use
/// [`Self::shutdown`] for explicit scene teardown after persistence handoff.
///
/// Apply complete frames through [`crate::WorldRuntime::apply_scene_plan`] to
/// reconcile stream eviction with residency actions in that same plan. The
/// low-level [`crate::SceneHost`] operations remain strict and have no implicit
/// frame scope.
pub struct SceneEcsHost<'world, P> {
    world: &'world mut World,
    provider: P,
}

impl<'world, P: SceneEcsProvider> SceneEcsHost<'world, P> {
    /// Creates a host, resuming this world's registered cells and bindings.
    ///
    /// There is one scene ownership registry per world, independent of provider
    /// type. A replacement provider must support the existing scene templates;
    /// provider-specific state is not saved in the registry. Use
    /// [`Self::into_provider`] when that state must be handed over as well.
    pub fn new(world: &'world mut World, provider: P) -> Self {
        world.init_resource::<SceneOwnership>();
        Self { world, provider }
    }

    /// Borrows the authoritative ECS world.
    #[must_use]
    pub fn world(&self) -> &World {
        self.world
    }

    /// Mutably borrows the authoritative ECS world for provider setup.
    pub fn world_mut(&mut self) -> &mut World {
        self.world
    }

    /// Borrows the typed provider.
    #[must_use]
    pub fn provider(&self) -> &P {
        &self.provider
    }

    /// Mutably borrows the typed provider.
    pub fn provider_mut(&mut self) -> &mut P {
        &mut self.provider
    }

    /// Returns the scene ownership token for a loaded sector.
    #[must_use]
    pub fn loaded_cell(&self, sector: SectorId) -> Option<&SceneSpawn> {
        self.ownership().cells.get(&sector)
    }

    /// Returns the number of live cell ownership tokens.
    #[must_use]
    pub fn loaded_cell_count(&self) -> usize {
        self.ownership().cells.len()
    }

    /// Returns the number of previously published scene trees awaiting teardown.
    #[must_use]
    pub fn retired_cell_count(&self) -> usize {
        self.ownership().retired_cells.len()
    }

    /// Returns the binding for a world residency id.
    #[must_use]
    pub fn residency_binding(&self, residency_id: u64) -> Option<ResidencyBinding> {
        self.ownership().bindings.get(&residency_id).copied()
    }

    /// Adds a stable residency-to-scene target mapping.
    pub fn bind_residency(
        &mut self,
        residency_id: u64,
        binding: ResidencyBinding,
    ) -> Result<(), SceneEcsHostConfigError> {
        if let Some(existing) = self.residency_binding(residency_id) {
            if existing != binding {
                return Err(SceneEcsHostConfigError::ConflictingBinding {
                    residency_id,
                    existing,
                    requested: binding,
                });
            }
            return Ok(());
        }
        self.ownership_mut().bindings.insert(residency_id, binding);
        Ok(())
    }

    /// Removes a residency binding and returns it when present.
    pub fn unbind_residency(&mut self, residency_id: u64) -> Option<ResidencyBinding> {
        self.ownership_mut().bindings.remove(&residency_id)
    }

    fn target(
        &self,
        residency_id: u64,
    ) -> Result<(ResidencyBinding, Entity), SceneEcsHostError<P::Error>> {
        let binding = self
            .residency_binding(residency_id)
            .ok_or(SceneEcsHostError::ResidencyNotBound(residency_id))?;
        let spawn = self
            .ownership()
            .cells
            .get(&binding.sector)
            .ok_or(SceneEcsHostError::CellNotLoaded(binding.sector))?;
        let entity =
            spawn
                .resolve(binding.local)
                .ok_or(SceneEcsHostError::LocalEntityUnavailable {
                    residency_id,
                    sector: binding.sector,
                    local: binding.local,
                })?;
        Ok((binding, entity))
    }

    fn load_cell(&mut self, sector: SectorId) -> Result<(), SceneEcsHostError<P::Error>> {
        if self.loaded_cell(sector).is_some() {
            return self.reload_cell(sector);
        }
        let spawn = self
            .provider
            .spawn_cell(self.world, sector)
            .map_err(SceneEcsHostError::Provider)?;
        let attached = spawn
            .root
            .is_some_and(|root| self.world.get::<SceneInstance>(root).is_some());
        if !attached {
            despawn_instance(self.world, &spawn);
            return Err(SceneEcsHostError::SceneNotAttached(sector));
        }
        self.ownership_mut().cells.insert(sector, spawn);
        Ok(())
    }

    /// Atomically publishes a provider-staged scene replacement.
    ///
    /// The prior cell remains registered and live until the staged scene has a
    /// distinct attached root and resolves every current residency binding. On
    /// any validation failure the staged scene is despawned and the old scene
    /// remains authoritative. Despawn hooks for the old scene run only after
    /// ownership has switched to the replacement.
    pub fn reload_cell(&mut self, sector: SectorId) -> Result<(), SceneEcsHostError<P::Error>> {
        self.reload_cell_with_state(sector, &[], None)
    }

    /// Stages and publishes several cell replacements as one ownership batch.
    ///
    /// Sectors are processed in sorted order, independent of caller order. A
    /// duplicate sector is rejected before invoking the provider. All staged
    /// replacements are validated before any ownership token changes; a
    /// recoverable staging, validation, or persistence replay failure destroys
    /// every completed staged scene and leaves all previous cells authoritative.
    /// Replacement entities are required to be disjoint from all currently
    /// owned and already staged cells. The batch guarantee covers the host's
    /// ownership registry only: arbitrary provider side effects and ECS
    /// observers are not rollbackable. Old-scene teardown happens after
    /// publication; if a despawn hook panics, the remaining old tokens stay in
    /// the world-owned retirement queue and can be retried with
    /// [`Self::retry_retired_cells`]. A provider must clean up incomplete
    /// staging work before returning an error.
    pub fn reload_cells_atomic(
        &mut self,
        sectors: &[SectorId],
    ) -> Result<(), SceneEcsHostError<P::Error>> {
        let mut sectors = sectors.to_vec();
        sectors.sort_unstable();
        if sectors.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(SceneEcsHostError::DuplicateReloadSector);
        }
        if let Some(&sector) = sectors
            .iter()
            .find(|sector| !self.ownership().cells.contains_key(sector))
        {
            return Err(SceneEcsHostError::CellNotLoaded(sector));
        }
        if sectors.is_empty() {
            return Ok(());
        }

        let mut protected_entities: BTreeSet<_> = self
            .ownership()
            .cells
            .values()
            .flat_map(|spawn| spawn.remap().entities().map(|(_, entity)| entity))
            .collect();
        let mut staged: Vec<(SectorId, SceneSpawn, SceneSpawn, ())> =
            Vec::with_capacity(sectors.len());
        for sector in sectors {
            let previous = self
                .ownership()
                .cells
                .get(&sector)
                .cloned()
                .expect("all target cells were validated");
            let Some(old_root) = previous.root else {
                self.cleanup_staged_reloads(&staged);
                return Err(SceneEcsHostError::InvalidReloadReplacement(sector));
            };
            let Some(previous_instance) = self.world.get::<SceneInstance>(old_root).cloned() else {
                self.cleanup_staged_reloads(&staged);
                return Err(SceneEcsHostError::SceneNotAttached(sector));
            };
            let replacement = match self.provider.stage_cell_reload(
                self.world,
                sector,
                &previous,
                &previous_instance,
            ) {
                Ok(replacement) => replacement,
                Err(error) => {
                    self.cleanup_staged_reloads(&staged);
                    return Err(SceneEcsHostError::Provider(error));
                }
            };

            let replacement_entities: Vec<_> = replacement
                .remap()
                .entities()
                .map(|(_, entity)| entity)
                .collect();
            let aliases_live_or_staged = replacement_entities
                .iter()
                .any(|candidate| protected_entities.contains(candidate));
            if aliases_live_or_staged {
                self.cleanup_staged_reloads(&staged);
                return Err(SceneEcsHostError::AliasedReloadReplacement(sector));
            }

            let replacement_valid = replacement.root.is_some_and(|root| {
                replacement.resolve(LocalEntityId::new(0)) == Some(root)
                    && self.world.get::<SceneInstance>(root).is_some()
                    && self.world.entities().contains(root)
            }) && replacement.remap().entities().all(|(_, entity)| {
                self.world.entities().contains(entity)
                    && self
                        .world
                        .entities()
                        .location(entity)
                        .is_some_and(|location| !location.is_empty())
            });
            if !replacement_valid {
                despawn_instance(self.world, &replacement);
                self.cleanup_staged_reloads(&staged);
                return Err(SceneEcsHostError::InvalidReloadReplacement(sector));
            }
            if let Some((&residency_id, binding)) =
                self.ownership().bindings.iter().find(|(_, binding)| {
                    binding.sector == sector && replacement.resolve(binding.local).is_none()
                })
            {
                let error = SceneEcsHostError::ReloadBindingUnavailable {
                    residency_id,
                    sector,
                    local: binding.local,
                };
                despawn_instance(self.world, &replacement);
                self.cleanup_staged_reloads(&staged);
                return Err(error);
            }

            let new_root = replacement.root.expect("replacement root was validated");
            let Some(mut new_instance) = self.world.get_mut::<SceneInstance>(new_root) else {
                despawn_instance(self.world, &replacement);
                self.cleanup_staged_reloads(&staged);
                return Err(SceneEcsHostError::InvalidReloadReplacement(sector));
            };
            new_instance.overrides = previous_instance.overrides.clone();
            new_instance.persistent = previous_instance.persistent;
            new_instance.tier = previous_instance.tier;
            protected_entities.extend(replacement_entities);
            staged.push((sector, previous, replacement, ()));
        }

        {
            let ownership = self.ownership_mut();
            for (sector, previous, replacement, _) in &staged {
                ownership.cells.insert(*sector, replacement.clone());
                ownership.retired_cells.push((*sector, previous.clone()));
            }
        }
        self.retry_retired_cells();
        Ok(())
    }

    /// Stages, replays durable cell state, and atomically publishes a group of
    /// replacements against one authoritative persistence snapshot.
    pub fn reload_cells_atomic_with_persistence(
        &mut self,
        sectors: &[SectorId],
        log: &AppendOnlyLog,
    ) -> Result<(), SceneEcsHostError<P::Error>> {
        let mut sectors = sectors.to_vec();
        sectors.sort_unstable();
        if sectors.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(SceneEcsHostError::DuplicateReloadSector);
        }
        if let Some(&sector) = sectors
            .iter()
            .find(|sector| !self.ownership().cells.contains_key(sector))
        {
            return Err(SceneEcsHostError::CellNotLoaded(sector));
        }
        if sectors.is_empty() {
            return Ok(());
        }
        let world_state = Timeline::replay(&Snapshot::new(0, WorldState::new()), log)
            .map_err(SceneEcsHostError::PersistenceReplay)?;
        let mut persistence = Vec::with_capacity(sectors.len());
        for &sector in &sectors {
            let objects = if let Some(objects) = self.ownership().persistence_objects.get(&sector) {
                objects.clone()
            } else {
                self.provider
                    .persistence_objects(sector)
                    .map_err(SceneEcsHostError::Provider)?
            };
            let states = objects
                .iter()
                .map(|&object| CellPersistenceState {
                    object,
                    state: world_state.object(object).cloned().unwrap_or_default(),
                })
                .collect::<Vec<_>>();
            persistence.push((sector, objects, states));
        }

        let mut staged: Vec<(SectorId, SceneSpawn, SceneSpawn, Vec<WorldPersistentId>)> =
            Vec::with_capacity(sectors.len());
        let mut protected_entities: BTreeSet<_> = self
            .ownership()
            .cells
            .values()
            .flat_map(|spawn| spawn.remap().entities().map(|(_, entity)| entity))
            .collect();
        for (sector, objects, states) in persistence {
            let previous = self.ownership().cells[&sector].clone();
            let Some(old_root) = previous.root else {
                self.cleanup_staged_reloads(&staged);
                return Err(SceneEcsHostError::InvalidReloadReplacement(sector));
            };
            let Some(previous_instance) = self.world.get::<SceneInstance>(old_root).cloned() else {
                self.cleanup_staged_reloads(&staged);
                return Err(SceneEcsHostError::SceneNotAttached(sector));
            };
            let replacement = match self.provider.stage_cell_reload(
                self.world,
                sector,
                &previous,
                &previous_instance,
            ) {
                Ok(replacement) => replacement,
                Err(error) => {
                    self.cleanup_staged_reloads(&staged);
                    return Err(SceneEcsHostError::Provider(error));
                }
            };
            let replacement_entities: Vec<_> = replacement
                .remap()
                .entities()
                .map(|(_, entity)| entity)
                .collect();
            if replacement_entities
                .iter()
                .any(|entity| protected_entities.contains(entity))
            {
                self.cleanup_staged_reloads(&staged);
                return Err(SceneEcsHostError::AliasedReloadReplacement(sector));
            }
            let valid = replacement.root.is_some_and(|root| {
                replacement.resolve(LocalEntityId::new(0)) == Some(root)
                    && self.world.get::<SceneInstance>(root).is_some()
                    && self.world.entities().contains(root)
            }) && replacement_entities.iter().all(|entity| {
                self.world.entities().contains(*entity)
                    && self
                        .world
                        .entities()
                        .location(*entity)
                        .is_some_and(|location| !location.is_empty())
            });
            if !valid {
                despawn_instance(self.world, &replacement);
                self.cleanup_staged_reloads(&staged);
                return Err(SceneEcsHostError::InvalidReloadReplacement(sector));
            }
            if let Some((&residency_id, binding)) =
                self.ownership().bindings.iter().find(|(_, binding)| {
                    binding.sector == sector && replacement.resolve(binding.local).is_none()
                })
            {
                let error = SceneEcsHostError::ReloadBindingUnavailable {
                    residency_id,
                    sector,
                    local: binding.local,
                };
                despawn_instance(self.world, &replacement);
                self.cleanup_staged_reloads(&staged);
                return Err(error);
            }
            if let Err(error) = self
                .provider
                .replay_cell_persistence(self.world, sector, &states)
            {
                despawn_instance(self.world, &replacement);
                self.cleanup_staged_reloads(&staged);
                return Err(SceneEcsHostError::Provider(error));
            }
            let new_root = replacement.root.expect("replacement root was validated");
            let Some(mut instance) = self.world.get_mut::<SceneInstance>(new_root) else {
                despawn_instance(self.world, &replacement);
                self.cleanup_staged_reloads(&staged);
                return Err(SceneEcsHostError::InvalidReloadReplacement(sector));
            };
            instance.overrides = previous_instance.overrides.clone();
            instance.persistent = previous_instance.persistent;
            instance.tier = previous_instance.tier;
            protected_entities.extend(replacement_entities);
            staged.push((sector, previous, replacement, objects));
        }

        {
            let ownership = self.ownership_mut();
            for (sector, previous, replacement, objects) in &staged {
                ownership.cells.insert(*sector, replacement.clone());
                ownership
                    .persistence_objects
                    .insert(*sector, objects.clone());
                ownership.retired_cells.push((*sector, previous.clone()));
            }
        }
        self.retry_retired_cells();
        Ok(())
    }

    fn cleanup_staged_reloads<T>(&mut self, staged: &[(SectorId, SceneSpawn, SceneSpawn, T)]) {
        for (_, _, replacement, _) in staged.iter().rev() {
            despawn_instance(self.world, replacement);
        }
    }

    fn reload_cell_with_state(
        &mut self,
        sector: SectorId,
        states: &[CellPersistenceState],
        persistence_objects: Option<Vec<WorldPersistentId>>,
    ) -> Result<(), SceneEcsHostError<P::Error>> {
        let previous = self
            .ownership()
            .cells
            .get(&sector)
            .cloned()
            .ok_or(SceneEcsHostError::CellNotLoaded(sector))?;
        let old_root = previous
            .root
            .ok_or(SceneEcsHostError::InvalidReloadReplacement(sector))?;
        let previous_instance = self
            .world
            .get::<SceneInstance>(old_root)
            .cloned()
            .ok_or(SceneEcsHostError::SceneNotAttached(sector))?;
        let replacement = self
            .provider
            .stage_cell_reload(self.world, sector, &previous, &previous_instance)
            .map_err(SceneEcsHostError::Provider)?;
        let previous_entities: BTreeSet<_> = previous
            .remap()
            .entities()
            .map(|(_, entity)| entity)
            .collect();
        let aliases_previous = replacement
            .remap()
            .entities()
            .any(|(_, candidate)| previous_entities.contains(&candidate));
        if aliases_previous {
            // Do not run teardown on an aliased token: that could destroy part
            // or all of the still-authoritative old scene. The provider owns
            // cleanup for a contract-violating staging result.
            return Err(SceneEcsHostError::AliasedReloadReplacement(sector));
        }
        let replacement_valid = replacement.root.is_some_and(|root| {
            replacement.resolve(LocalEntityId::new(0)) == Some(root)
                && self.world.get::<SceneInstance>(root).is_some()
                && self.world.entities().contains(root)
        }) && replacement.remap().entities().all(|(_, entity)| {
            self.world.entities().contains(entity)
                && self
                    .world
                    .entities()
                    .location(entity)
                    .is_some_and(|location| !location.is_empty())
        });
        if !replacement_valid {
            despawn_instance(self.world, &replacement);
            return Err(SceneEcsHostError::InvalidReloadReplacement(sector));
        }
        if let Some((&residency_id, binding)) =
            self.ownership().bindings.iter().find(|(_, binding)| {
                binding.sector == sector && replacement.resolve(binding.local).is_none()
            })
        {
            let error = SceneEcsHostError::ReloadBindingUnavailable {
                residency_id,
                sector,
                local: binding.local,
            };
            despawn_instance(self.world, &replacement);
            return Err(error);
        }

        // Only invoke provider replay after proving the staged token is valid
        // and disjoint from the live scene. The callback can then target the
        // provider's staged entity map without risking writes to old entities.
        if let Err(error) = self
            .provider
            .replay_cell_persistence(self.world, sector, states)
        {
            despawn_instance(self.world, &replacement);
            return Err(SceneEcsHostError::Provider(error));
        }

        // Template replacement is allowed to rebuild the anchor, but instance
        // state belongs to the live instance and must survive independently of
        // provider policy. The provider may already have replayed typed payload
        // changes; the host now reinstates the authoritative sparse delta and
        // stable identity on the staged anchor before publishing it.
        let new_root = replacement.root.expect("replacement root was validated");
        let Some(mut new_instance) = self.world.get_mut::<SceneInstance>(new_root) else {
            despawn_instance(self.world, &replacement);
            return Err(SceneEcsHostError::InvalidReloadReplacement(sector));
        };
        new_instance.overrides = previous_instance.overrides.clone();
        new_instance.persistent = previous_instance.persistent;
        new_instance.tier = previous_instance.tier;
        let ownership = self.ownership_mut();
        ownership.cells.insert(sector, replacement);
        ownership.retired_cells.push((sector, previous));
        if let Some(objects) = persistence_objects {
            ownership.persistence_objects.insert(sector, objects);
        }
        self.retry_retired_cells();
        Ok(())
    }

    fn load_cell_with_persistence(
        &mut self,
        sector: SectorId,
        log: &AppendOnlyLog,
    ) -> Result<(), SceneEcsHostError<P::Error>> {
        if self.loaded_cell(sector).is_some() {
            let mut objects = self
                .ownership()
                .persistence_objects
                .get(&sector)
                .cloned()
                .unwrap_or_default();
            for object in self
                .provider
                .persistence_objects(sector)
                .map_err(SceneEcsHostError::Provider)?
            {
                if !objects.contains(&object) {
                    objects.push(object);
                }
            }
            let world_state = Timeline::replay(&Snapshot::new(0, WorldState::new()), log)
                .map_err(SceneEcsHostError::PersistenceReplay)?;
            let states: Vec<_> = objects
                .iter()
                .map(|&object| CellPersistenceState {
                    object,
                    state: world_state.object(object).cloned().unwrap_or_default(),
                })
                .collect();
            return self.reload_cell_with_state(sector, &states, Some(objects));
        }
        let mut objects = self
            .ownership()
            .persistence_objects
            .get(&sector)
            .cloned()
            .unwrap_or_default();
        for object in self
            .provider
            .persistence_objects(sector)
            .map_err(SceneEcsHostError::Provider)?
        {
            if !objects.contains(&object) {
                objects.push(object);
            }
        }
        let world_state = Timeline::replay(&Snapshot::new(0, WorldState::new()), log)
            .map_err(SceneEcsHostError::PersistenceReplay)?;
        let states: Vec<_> = objects
            .iter()
            .map(|&object| CellPersistenceState {
                object,
                state: world_state.object(object).cloned().unwrap_or_default(),
            })
            .collect();

        let spawn = self
            .provider
            .spawn_cell(self.world, sector)
            .map_err(SceneEcsHostError::Provider)?;
        let attached = spawn
            .root
            .is_some_and(|root| self.world.get::<SceneInstance>(root).is_some());
        if !attached {
            despawn_instance(self.world, &spawn);
            return Err(SceneEcsHostError::SceneNotAttached(sector));
        }
        if let Err(error) = self
            .provider
            .replay_cell_persistence(self.world, sector, &states)
        {
            despawn_instance(self.world, &spawn);
            return Err(SceneEcsHostError::Provider(error));
        }
        let ownership = self.ownership_mut();
        ownership.cells.insert(sector, spawn);
        ownership.persistence_objects.insert(sector, objects);
        Ok(())
    }

    fn unload_cell(&mut self, sector: SectorId) -> Result<(), SceneEcsHostError<P::Error>> {
        let Some(spawn) = self.ownership().cells.get(&sector).cloned() else {
            return Err(SceneEcsHostError::CellNotLoaded(sector));
        };
        self.destroy_cell(sector, &spawn);
        Ok(())
    }

    fn unload_cell_with_persistence(
        &mut self,
        sector: SectorId,
        log: &mut AppendOnlyLog,
    ) -> Result<(), SceneEcsHostError<P::Error>> {
        let Some(spawn) = self.ownership().cells.get(&sector).cloned() else {
            return Err(SceneEcsHostError::CellNotLoaded(sector));
        };
        let transactions =
            self.capture_persistence_transactions(sector, log, log.next_sequence())?;
        log.append_batch(&transactions)
            .map_err(SceneEcsHostError::PersistenceAppend)?;
        let ownership = self.ownership_mut();
        let objects = ownership.persistence_objects.entry(sector).or_default();
        for transaction in &transactions {
            let object = transaction.object();
            if !objects.contains(&object) {
                objects.push(object);
            }
        }
        self.destroy_cell(sector, &spawn);
        Ok(())
    }

    fn unload_cell_with_journal(
        &mut self,
        sector: SectorId,
        journal: &mut FileJournal,
    ) -> Result<(), SceneEcsHostError<P::Error>> {
        let Some(spawn) = self.ownership().cells.get(&sector).cloned() else {
            return Err(SceneEcsHostError::CellNotLoaded(sector));
        };
        let next_sequence = journal.cursor().next_sequence();
        let prior = Timeline::replay(&Snapshot::new(0, WorldState::new()), journal.log())
            .map_err(SceneEcsHostError::PersistenceReplay)?;
        let transactions =
            self.capture_scene_transactions(sector, &prior, next_sequence)?;
        journal
            .append_batch(&transactions)
            .map_err(|error| SceneEcsHostError::DurableAppend(error.to_string()))?;
        let ownership = self.ownership_mut();
        let objects = ownership.persistence_objects.entry(sector).or_default();
        for transaction in &transactions {
            let object = transaction.object();
            if !objects.contains(&object) {
                objects.push(object);
            }
        }
        self.destroy_cell(sector, &spawn);
        Ok(())
    }

    fn capture_persistence_transactions(
        &mut self,
        sector: SectorId,
        log: &AppendOnlyLog,
        next_sequence: u64,
    ) -> Result<Vec<Transaction>, SceneEcsHostError<P::Error>> {
        let prior = Timeline::replay(&Snapshot::new(0, WorldState::new()), log)
            .map_err(SceneEcsHostError::PersistenceReplay)?;
        self.capture_scene_transactions(sector, &prior, next_sequence)
    }

    fn capture_scene_transactions(
        &mut self,
        sector: SectorId,
        prior: &WorldState,
        next_sequence: u64,
    ) -> Result<Vec<Transaction>, SceneEcsHostError<P::Error>> {
        let capture = self
            .provider
            .capture_scene_persistence(self.world, sector, prior, next_sequence)
            .map_err(SceneEcsHostError::Provider)?;
        capture
            .schema
            .transactions_for_batch_with_companion(
                &capture.batch,
                prior,
                capture.actor,
                capture.clock,
                next_sequence,
                &capture.transaction_ids,
                capture.companion_transaction_id,
                capture.companion_object,
                capture.companion_field,
            )
            .map_err(SceneEcsHostError::PersistenceBridge)
    }
}

impl<P: SceneEcsProvider> crate::SceneHost for SceneEcsHost<'_, P> {
    type Error = SceneEcsHostError<P::Error>;

    fn load_cell(&mut self, sector: SectorId) -> Result<(), Self::Error> {
        self.load_cell(sector)
    }

    fn reload_cells_atomic(&mut self, sectors: &[SectorId]) -> Result<bool, Self::Error> {
        self.reload_cells_atomic(sectors).map(|()| true)
    }

    fn unload_cell(&mut self, sector: SectorId) -> Result<(), Self::Error> {
        self.unload_cell(sector)
    }

    fn transition_residency(
        &mut self,
        residency_id: u64,
        from: prism_world_residency::ResidencyState,
        to: prism_world_residency::ResidencyState,
    ) -> Result<(), Self::Error> {
        let (binding, entity) = self.target(residency_id)?;
        self.provider
            .transition_residency(
                self.world,
                residency_id,
                binding.sector,
                binding.local,
                entity,
                from,
                to,
            )
            .map_err(SceneEcsHostError::Provider)
    }

    fn transition_residency_batch(
        &mut self,
        transitions: &[ResidencyTransition],
    ) -> Result<(), ResidencyBatchError<Self::Error>> {
        let mut resolved = Vec::with_capacity(transitions.len());
        for transition in transitions {
            let (binding, entity) =
                self.target(transition.id)
                    .map_err(|error| ResidencyBatchError {
                        completed: 0,
                        error,
                    })?;
            resolved.push(SceneResidencyTransition {
                residency_id: transition.id,
                sector: binding.sector,
                local: binding.local,
                entity,
                from: transition.from,
                to: transition.to,
            });
        }
        self.provider
            .transition_residency_batch(self.world, &resolved)
            .map_err(|(completed, error)| ResidencyBatchError {
                completed,
                error: SceneEcsHostError::Provider(error),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::boxed::Box;
    use alloc::vec;
    use alloc::vec::Vec;
    use prism_asset::StableGuid;
    use prism_ecs::component::Component;
    use prism_reflect::Reflect;
    use prism_scene::{
        replay_overrides, spawn_typed_plan, ArchetypeGroup, ArchetypeSignature, InstantiationPlan,
        ResidencyTier, SceneInstancePersistence, TypedSpawnPlan,
    };
    use prism_world_persist::{
        ActorId, FieldId, LogicalClock, OverrideDelta, OverrideValue, PersistentId, TransactionId,
    };

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub(super) struct Marker(pub(super) u32);

    impl Component for Marker {}

    #[derive(Clone, Debug, Reflect, PartialEq)]
    pub(super) struct ReloadSettings {
        pub(super) value: u64,
    }

    impl Component for ReloadSettings {}

    #[derive(Default)]
    pub(super) struct Provider {
        pub(super) events: Vec<(prism_world_residency::ResidencyState, u64, Entity)>,
        pub(super) fail_transition: bool,
        pub(super) fail_replay: bool,
        pub(super) stage_attempts: usize,
        pub(super) fail_stage_on: Option<usize>,
        pub(super) persistent_object: Option<WorldPersistentId>,
        pub(super) cell_entity: Option<Entity>,
        pub(super) scene_property_delta: bool,
    }

    impl SceneEcsProvider for Provider {
        type Error = ();

        fn spawn_cell(
            &mut self,
            world: &mut World,
            sector: SectorId,
        ) -> Result<SceneSpawn, Self::Error> {
            let plan = InstantiationPlan {
                entity_count: 1,
                groups: vec![ArchetypeGroup::new(
                    ArchetypeSignature::default(),
                    vec![LocalEntityId::new(0)].into_boxed_slice(),
                    Box::new([]),
                )]
                .into_boxed_slice(),
                entity_valued_fields: Box::new([]),
                hierarchy: Box::new([]),
                reserved_local_base: LocalEntityId::RESERVED_BASE,
            };
            let spawn = spawn_typed_plan(
                world,
                &plan,
                TypedSpawnPlan::new().with_bundle(
                    LocalEntityId::new(0),
                    (Marker(7), ReloadSettings { value: 7 }),
                ),
            )
            .map_err(|_| ())?;
            spawn
                .attach_instance(
                    world,
                    StableGuid::from_u128(sector.morton_hint() as u128),
                    ResidencyTier::Streamed,
                )
                .map_err(|_| ())?;
            self.cell_entity = spawn.resolve(LocalEntityId::new(0));
            Ok(spawn)
        }

        fn stage_cell_reload(
            &mut self,
            world: &mut World,
            sector: SectorId,
            _previous: &SceneSpawn,
            previous_instance: &SceneInstance,
        ) -> Result<SceneSpawn, Self::Error> {
            self.stage_attempts += 1;
            if self.fail_stage_on == Some(self.stage_attempts) {
                return Err(());
            }
            world.register_reflect_component::<ReloadSettings>();
            let plan = InstantiationPlan {
                entity_count: 1,
                groups: vec![ArchetypeGroup::new(
                    ArchetypeSignature::default(),
                    vec![LocalEntityId::new(0)].into_boxed_slice(),
                    Box::new([]),
                )]
                .into_boxed_slice(),
                entity_valued_fields: Box::new([]),
                hierarchy: Box::new([]),
                reserved_local_base: LocalEntityId::RESERVED_BASE,
            };
            let spawn = spawn_typed_plan(
                world,
                &plan,
                TypedSpawnPlan::new().with_bundle(
                    LocalEntityId::new(0),
                    (Marker(70), ReloadSettings { value: 70 }),
                ),
            )
            .map_err(|_| ())?;
            spawn
                .attach_instance(
                    world,
                    StableGuid::from_u128(sector.morton_hint() as u128 + 100),
                    ResidencyTier::Streamed,
                )
                .map_err(|_| ())?;
            if let Some(overrides) = previous_instance.overrides() {
                if let Err(error) = replay_overrides(world, &spawn, overrides) {
                    despawn_instance(world, &spawn);
                    let _ = error;
                    return Err(());
                }
            }
            self.cell_entity = spawn.resolve(LocalEntityId::new(0));
            Ok(spawn)
        }

        fn transition_residency(
            &mut self,
            world: &mut World,
            residency_id: u64,
            _sector: SectorId,
            _local: LocalEntityId,
            entity: Entity,
            _from: prism_world_residency::ResidencyState,
            to: prism_world_residency::ResidencyState,
        ) -> Result<(), Self::Error> {
            if self.fail_transition {
                return Err(());
            }
            world.get_mut::<Marker>(entity).ok_or(())?.0 = match to {
                prism_world_residency::ResidencyState::TierA => 9,
                prism_world_residency::ResidencyState::TierB => 6,
                prism_world_residency::ResidencyState::TierC => 3,
            };
            self.events.push((to, residency_id, entity));
            Ok(())
        }

        fn persistence_objects(
            &mut self,
            _sector: SectorId,
        ) -> Result<Vec<WorldPersistentId>, Self::Error> {
            Ok(self.persistent_object.into_iter().collect())
        }

        fn capture_scene_persistence(
            &mut self,
            world: &World,
            sector: SectorId,
            _prior: &WorldState,
            _next_sequence: u64,
        ) -> Result<ScenePersistenceCapture, Self::Error> {
            let companion_id = 9009;
            let mut deltas = Vec::new();
            let mut transaction_ids = Vec::new();
            if let Some(object) = self.persistent_object {
                let value = if self.scene_property_delta {
                    12
                } else {
                    i64::from(
                        world
                            .get::<Marker>(self.cell_entity.ok_or(())?)
                            .ok_or(())?
                            .0,
                    )
                };
                deltas.push((
                    object,
                    OverrideDelta::single(FieldId(1), OverrideValue::Integer(value))
                        .map_err(|_| ())?,
                ));
                transaction_ids.push(TransactionId(1000));
            }
            let schema = ScenePersistenceSchema::new(Vec::new()).map_err(|_| ())?;
            Ok(ScenePersistenceCapture {
                schema,
                batch: ScenePersistenceBatch {
                    deltas,
                    scene_snapshot: SceneInstancePersistence {
                        template: StableGuid::from_u128(sector.morton_hint() as u128),
                        instance: None,
                        properties: Vec::new(),
                        structural: Vec::new(),
                    },
                },
                actor: ActorId(9),
                clock: LogicalClock(9),
                transaction_ids,
                companion_transaction_id: TransactionId(companion_id),
                companion_object: WorldPersistentId(u128::from(
                    9909u64.wrapping_add(sector.morton_hint()),
                )),
                companion_field: FieldId(9909),
            })
        }

        fn replay_cell_persistence(
            &mut self,
            world: &mut World,
            _sector: SectorId,
            states: &[CellPersistenceState],
        ) -> Result<(), Self::Error> {
            if self.fail_replay {
                return Err(());
            }
            let Some(object) = self.persistent_object else {
                return Ok(());
            };
            if let Some(OverrideValue::Integer(value)) = states
                .iter()
                .find(|state| state.object == object)
                .and_then(|state| state.state.get(FieldId(1)))
            {
                world
                    .get_mut::<Marker>(self.cell_entity.ok_or(())?)
                    .ok_or(())?
                    .0 = u32::try_from(*value).map_err(|_| ())?;
            }
            Ok(())
        }
    }

    fn host<'a>(world: &'a mut World) -> SceneEcsHost<'a, Provider> {
        SceneEcsHost::new(world, Provider::default())
    }

    #[test]
    fn load_and_unload_use_scene_ownership_boundary() {
        let mut world = World::new();
        let sector = SectorId::new(0, 2, -1, 4);
        let mut host = host(&mut world);
        crate::SceneHost::load_cell(&mut host, sector).unwrap();
        assert_eq!(host.loaded_cell_count(), 1);
        let root = host.loaded_cell(sector).unwrap().root.unwrap();
        crate::SceneHost::unload_cell(&mut host, sector).unwrap();
        assert_eq!(host.loaded_cell_count(), 0);
        drop(host);
        assert!(!world.contains(root));
        assert_eq!(world.entity_count(), 0);
    }

    #[test]
    fn structured_capture_filters_persisted_properties_and_companion_before_append() {
        let mut world = World::new();
        let sector = SectorId::new(0, 17, -2, 3);
        let provider = Provider {
            ..Provider::default()
        };
        let mut host = SceneEcsHost::new(&mut world, provider);
        crate::SceneHost::load_cell(&mut host, sector).unwrap();
        let mut log = AppendOnlyLog::new();
        let companion = SceneInstancePersistence {
            template: StableGuid::from_u128(sector.morton_hint() as u128),
            instance: None,
            properties: Vec::new(),
            structural: Vec::new(),
        };
        let bytes = prism_scene::encode_persistence(&companion).unwrap();
        let companion_object = PersistentId(u128::from(9909u64.wrapping_add(sector.morton_hint())));
        let delta = OverrideDelta::single(FieldId(9909), OverrideValue::Bytes(bytes)).unwrap();
        log.append(
            Transaction::new(
                TransactionId(9009),
                ActorId(9),
                LogicalClock(9),
                0,
                companion_object,
                delta.clone(),
                delta,
            )
            .unwrap(),
        )
        .unwrap();
        host.unload_cell_with_persistence(sector, &mut log).unwrap();
        assert_eq!(log.len(), 1);
        assert!(host.loaded_cell(sector).is_none());
    }

    #[test]
    fn structured_capture_appends_companion_and_sparse_delta_in_one_memory_batch() {
        let mut world = World::new();
        let sector = SectorId::new(0, 23, 1, -5);
        let provider = Provider {
            persistent_object: Some(WorldPersistentId(7700)),
            scene_property_delta: true,
            ..Provider::default()
        };
        let mut host = SceneEcsHost::new(&mut world, provider);
        crate::SceneHost::load_cell(&mut host, sector).unwrap();
        let prior_delta = OverrideDelta::single(FieldId(1), OverrideValue::Integer(11)).unwrap();
        let mut log = AppendOnlyLog::new();
        log.append(
            Transaction::new(
                TransactionId(1),
                ActorId(1),
                LogicalClock(1),
                0,
                PersistentId(7700),
                prior_delta.clone(),
                prior_delta,
            )
            .unwrap(),
        )
        .unwrap();
        host.unload_cell_with_persistence(sector, &mut log).unwrap();
        assert_eq!(log.len(), 3);
        assert_eq!(log.get(1).unwrap().sequence(), 1);
        assert_eq!(log.get(1).unwrap().object(), PersistentId(7700));
        assert_eq!(
            log.get(1).unwrap().delta().get(FieldId(1)),
            Some(&OverrideValue::Integer(12))
        );
        assert_eq!(
            log.get(1).unwrap().inverse().get(FieldId(1)),
            Some(&OverrideValue::Integer(11))
        );
        assert_eq!(log.get(2).unwrap().sequence(), 2);
        assert_eq!(
            log.get(2).unwrap().object(),
            PersistentId(u128::from(9909u64.wrapping_add(sector.morton_hint())))
        );
        assert!(matches!(
            log.get(2).unwrap().delta().get(FieldId(9909)),
            Some(OverrideValue::Bytes(_))
        ));
    }

    #[test]
    fn residency_operations_resolve_typed_scene_targets() {
        let mut world = World::new();
        let sector = SectorId::new(0, 0, 0, 0);
        let mut host = host(&mut world);
        crate::SceneHost::load_cell(&mut host, sector).unwrap();
        let root = host.loaded_cell(sector).unwrap().root.unwrap();
        host.bind_residency(
            41,
            ResidencyBinding {
                sector,
                local: LocalEntityId::new(0),
            },
        )
        .unwrap();

        crate::SceneHost::transition_residency(
            &mut host,
            41,
            prism_world_residency::ResidencyState::TierC,
            prism_world_residency::ResidencyState::TierA,
        )
        .unwrap();
        assert_eq!(host.world().get::<Marker>(root), Some(&Marker(9)));
        crate::SceneHost::transition_residency(
            &mut host,
            41,
            prism_world_residency::ResidencyState::TierA,
            prism_world_residency::ResidencyState::TierC,
        )
        .unwrap();
        assert_eq!(host.world().get::<Marker>(root), Some(&Marker(3)));
        assert_eq!(host.provider().events.len(), 2);
        assert_eq!(
            host.provider().events[0].0,
            prism_world_residency::ResidencyState::TierA
        );
        assert_eq!(
            host.provider().events[1].0,
            prism_world_residency::ResidencyState::TierC
        );
    }

    #[test]
    fn residency_batch_resolves_all_targets_before_provider_work() {
        let mut world = World::new();
        let sector = SectorId::new(0, 0, 0, 0);
        let mut host = host(&mut world);
        crate::SceneHost::load_cell(&mut host, sector).unwrap();
        host.bind_residency(
            41,
            ResidencyBinding {
                sector,
                local: LocalEntityId::new(0),
            },
        )
        .unwrap();

        let transitions = [
            ResidencyTransition {
                id: 41,
                from: prism_world_residency::ResidencyState::TierC,
                to: prism_world_residency::ResidencyState::TierA,
            },
            ResidencyTransition {
                id: 99,
                from: prism_world_residency::ResidencyState::TierC,
                to: prism_world_residency::ResidencyState::TierA,
            },
        ];
        let result = crate::SceneHost::transition_residency_batch(&mut host, &transitions);
        assert_eq!(
            result,
            Err(ResidencyBatchError {
                completed: 0,
                error: SceneEcsHostError::ResidencyNotBound(99),
            })
        );
        assert!(host.provider().events.is_empty());
        let root = host.loaded_cell(sector).unwrap().root.unwrap();
        assert_eq!(host.world().get::<Marker>(root), Some(&Marker(7)));
    }

    #[test]
    fn invalid_targets_are_rejected_and_unload_clears_bindings() {
        let mut world = World::new();
        let sector = SectorId::new(0, 0, 0, 0);
        let mut host = host(&mut world);
        assert_eq!(
            crate::SceneHost::transition_residency(
                &mut host,
                9,
                prism_world_residency::ResidencyState::TierC,
                prism_world_residency::ResidencyState::TierA,
            ),
            Err(SceneEcsHostError::ResidencyNotBound(9))
        );
        crate::SceneHost::load_cell(&mut host, sector).unwrap();
        host.bind_residency(
            9,
            ResidencyBinding {
                sector,
                local: LocalEntityId::new(0),
            },
        )
        .unwrap();
        assert_eq!(
            host.bind_residency(
                9,
                ResidencyBinding {
                    sector,
                    local: LocalEntityId::new(1),
                },
            ),
            Err(SceneEcsHostConfigError::ConflictingBinding {
                residency_id: 9,
                existing: ResidencyBinding {
                    sector,
                    local: LocalEntityId::new(0),
                },
                requested: ResidencyBinding {
                    sector,
                    local: LocalEntityId::new(1),
                },
            })
        );
        crate::SceneHost::unload_cell(&mut host, sector).unwrap();
        assert_eq!(host.residency_binding(9), None);
    }
}
