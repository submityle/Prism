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

use prism_ecs::entity::Entity;
use prism_ecs::world::World;
use prism_scene::{despawn_instance, LocalEntityId, SceneInstance, SceneSpawn};
use prism_world_stream::SectorId;

mod ownership;
mod plan_application;

use ownership::SceneOwnership;

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

    /// Materializes a bound target using its resolved typed ECS entity.
    ///
    /// This callback is explicit because the provider owns the scene template
    /// and component construction policy that the generic world facade cannot
    /// infer.
    fn materialize(
        &mut self,
        world: &mut World,
        residency_id: u64,
        sector: SectorId,
        local: LocalEntityId,
        entity: Entity,
    ) -> Result<(), Self::Error>;

    /// Dematerializes a bound target using its resolved typed ECS entity.
    ///
    /// `prism_scene` currently exposes no generic inverse operation, so this
    /// callback is the provider-owned, verifiable dematerialization boundary.
    fn dematerialize(
        &mut self,
        world: &mut World,
        residency_id: u64,
        sector: SectorId,
        local: LocalEntityId,
        entity: Entity,
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
    /// The provider rejected a typed operation.
    Provider(E),
    /// A provider returned a spawn without a live scene anchor.
    SceneNotAttached(SectorId),
    /// A load would replace an existing scene ownership token.
    CellAlreadyLoaded(SectorId),
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
            return Err(SceneEcsHostError::CellAlreadyLoaded(sector));
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

    fn unload_cell(&mut self, sector: SectorId) -> Result<(), SceneEcsHostError<P::Error>> {
        let Some(spawn) = self.ownership().cells.get(&sector).cloned() else {
            return Err(SceneEcsHostError::CellNotLoaded(sector));
        };
        self.destroy_cell(sector, &spawn);
        Ok(())
    }
}

impl<P: SceneEcsProvider> crate::SceneHost for SceneEcsHost<'_, P> {
    type Error = SceneEcsHostError<P::Error>;

    fn load_cell(&mut self, sector: SectorId) -> Result<(), Self::Error> {
        self.load_cell(sector)
    }

    fn unload_cell(&mut self, sector: SectorId) -> Result<(), Self::Error> {
        self.unload_cell(sector)
    }

    fn promote(&mut self, residency_id: u64) -> Result<(), Self::Error> {
        let (binding, entity) = self.target(residency_id)?;
        self.provider
            .materialize(
                self.world,
                residency_id,
                binding.sector,
                binding.local,
                entity,
            )
            .map_err(SceneEcsHostError::Provider)
    }

    fn demote(&mut self, residency_id: u64) -> Result<(), Self::Error> {
        let (binding, entity) = self.target(residency_id)?;
        self.provider
            .dematerialize(
                self.world,
                residency_id,
                binding.sector,
                binding.local,
                entity,
            )
            .map_err(SceneEcsHostError::Provider)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::boxed::Box;
    use alloc::vec;
    use prism_asset::StableGuid;
    use prism_ecs::component::Component;
    use prism_scene::{
        spawn_typed_plan, ArchetypeGroup, ArchetypeSignature, InstantiationPlan, ResidencyTier,
        TypedSpawnPlan,
    };

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub(super) struct Marker(pub(super) u32);

    impl Component for Marker {}

    #[derive(Default)]
    pub(super) struct Provider {
        pub(super) events: vec::Vec<(u8, u64, Entity)>,
        pub(super) fail_dematerialize: bool,
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
                TypedSpawnPlan::new().with_bundle(LocalEntityId::new(0), Marker(7)),
            )
            .map_err(|_| ())?;
            spawn
                .attach_instance(
                    world,
                    StableGuid::from_u128(sector.morton_hint() as u128),
                    ResidencyTier::Streamed,
                )
                .map_err(|_| ())?;
            Ok(spawn)
        }

        fn materialize(
            &mut self,
            world: &mut World,
            residency_id: u64,
            _sector: SectorId,
            _local: LocalEntityId,
            entity: Entity,
        ) -> Result<(), Self::Error> {
            world.get_mut::<Marker>(entity).ok_or(())?.0 = 9;
            self.events.push((1, residency_id, entity));
            Ok(())
        }

        fn dematerialize(
            &mut self,
            world: &mut World,
            residency_id: u64,
            _sector: SectorId,
            _local: LocalEntityId,
            entity: Entity,
        ) -> Result<(), Self::Error> {
            if self.fail_dematerialize {
                return Err(());
            }
            world.get_mut::<Marker>(entity).ok_or(())?.0 = 3;
            self.events.push((2, residency_id, entity));
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

        crate::SceneHost::promote(&mut host, 41).unwrap();
        assert_eq!(host.world().get::<Marker>(root), Some(&Marker(9)));
        crate::SceneHost::demote(&mut host, 41).unwrap();
        assert_eq!(host.world().get::<Marker>(root), Some(&Marker(3)));
        assert_eq!(host.provider().events.len(), 2);
        assert_eq!(host.provider().events[0].0, 1);
        assert_eq!(host.provider().events[1].0, 2);
    }

    #[test]
    fn invalid_targets_are_rejected_and_unload_clears_bindings() {
        let mut world = World::new();
        let sector = SectorId::new(0, 0, 0, 0);
        let mut host = host(&mut world);
        assert_eq!(
            crate::SceneHost::promote(&mut host, 9),
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
