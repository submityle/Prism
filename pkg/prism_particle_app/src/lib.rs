//! Bevy-free Prism App/ECS integration for the Ember CPU particle runtime.
//!
//! [`ParticleRuntimePlugin`] advances each [`EmitterRuntimeComponent`] once
//! per `FixedUpdate` using the configured fixed timestep. During `Last`, it
//! publishes a one-way, flat [`ParticleSnapshot`] in the main world. The
//! snapshot owns one reusable contiguous `Vec` and is suitable for downstream
//! render extraction; this crate does not depend on or install a renderer.

use alloc::vec::Vec;
use core::time::Duration;

use prism_app::SubAppLabel;
use prism_app::{App, EngineClocks, FixedUpdate, Last, Plugin};
use prism_ecs::extract::ExtractResource;
use prism_ecs::prelude::Entity;
use prism_ecs::prelude::{Component, Query, Res, ResMut, Resource};
use prism_ecs::world::World;
use prism_render_architecture::particle::runtime::{EmitterRuntime, ParticleRecord};

extern crate alloc;

/// An ECS component owning one already-constructed Ember emitter runtime.
#[derive(Component)]
pub struct EmitterRuntimeComponent(pub EmitterRuntime);

impl EmitterRuntimeComponent {
    /// Wraps an already validated runtime without changing its configuration.
    #[must_use]
    pub const fn new(runtime: EmitterRuntime) -> Self {
        Self(runtime)
    }

    /// Shared access to the owned runtime.
    #[must_use]
    pub const fn runtime(&self) -> &EmitterRuntime {
        &self.0
    }

    /// Mutable access to the owned runtime.
    pub fn runtime_mut(&mut self) -> &mut EmitterRuntime {
        &mut self.0
    }
}

/// A live particle paired with its source entity and stable pool slot.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParticleSnapshotEntry {
    /// Main-world entity whose component owns the emitter.
    pub emitter_entity: Entity,
    /// Stable slot in that emitter's fixed-capacity particle pool.
    pub slot: u32,
    /// Current particle data copied from the authoritative CPU runtime.
    pub particle: ParticleRecord,
}

/// Stable contiguous range for one emitter in [`ParticleSnapshot`].
///
/// The range refers to the snapshot's packed `particles` slice. Keeping this
/// metadata beside the flat payload lets render consumers route or batch
/// emitters without rescanning every particle or relying on ECS iteration
/// order after extraction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParticleEmitterSnapshot {
    /// Main-world entity owning the emitter runtime.
    pub emitter_entity: Entity,
    /// First entry in [`ParticleSnapshot::particles`].
    pub first_particle: u32,
    /// Number of live particles belonging to the emitter.
    pub particle_count: u32,
}

/// Main-world render handoff containing every live particle for the latest
/// `Last` phase. Its storage is retained and reused across snapshot updates.
#[derive(Resource, Debug, Default)]
pub struct ParticleSnapshot {
    frame: u64,
    particles: Vec<ParticleSnapshotEntry>,
    emitters: Vec<ParticleEmitterSnapshot>,
}

/// Render-world projection of [`ParticleSnapshot`].
///
/// The projection deliberately exposes only the contiguous particle payload;
/// the authoritative emitter components and runtime remain in the main world.
#[derive(Clone, Debug, Default, PartialEq, Resource)]
pub struct ExtractedParticleSnapshot {
    frame: u64,
    particles: Vec<ParticleSnapshotEntry>,
    emitters: Vec<ParticleEmitterSnapshot>,
}

impl ExtractedParticleSnapshot {
    /// Frame sequence number copied from the main-world snapshot.
    #[must_use]
    pub const fn frame(&self) -> u64 {
        self.frame
    }

    /// Live particles copied for render preparation.
    #[must_use]
    pub fn particles(&self) -> &[ParticleSnapshotEntry] {
        &self.particles
    }

    /// Packed live-particle ranges grouped by source emitter.
    #[must_use]
    pub fn emitters(&self) -> &[ParticleEmitterSnapshot] {
        &self.emitters
    }

    /// Number of live particles in the extracted projection.
    #[must_use]
    pub fn len(&self) -> usize {
        self.particles.len()
    }

    /// Whether the extracted projection is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.particles.is_empty()
    }
}

impl ExtractResource for ParticleSnapshot {
    type Out = ExtractedParticleSnapshot;

    fn extract(&self, _main: &World) -> Option<Self::Out> {
        Some(ExtractedParticleSnapshot {
            frame: self.frame,
            particles: self.particles.clone(),
            emitters: self.emitters.clone(),
        })
    }
}

/// Installs the one-way main-world snapshot extraction on a secondary app.
///
/// The caller supplies the secondary app label so this crate stays independent
/// of any particular renderer crate or renderer-specific label type.
pub fn install_particle_snapshot_extract<L>(app: &mut App, label: L) -> &mut App
where
    L: SubAppLabel,
{
    app.add_extract(label, |main, secondary| {
        extract_particle_snapshot(main, secondary);
    })
}

/// Copies the latest snapshot into a secondary world, reusing its allocation.
pub fn extract_particle_snapshot(main: &World, secondary: &mut World) {
    secondary.init_resource::<ExtractedParticleSnapshot>();
    let extracted = secondary.resource_mut::<ExtractedParticleSnapshot>();
    let Some(source) = main.get_resource::<ParticleSnapshot>() else {
        extracted.frame = extracted.frame.wrapping_add(1);
        extracted.particles.clear();
        return;
    };
    extracted.frame = source.frame;
    extracted.particles.clear();
    extracted.particles.extend_from_slice(&source.particles);
    extracted.emitters.clear();
    extracted.emitters.extend_from_slice(&source.emitters);
}

impl ParticleSnapshot {
    /// Frame sequence number, incremented each time the snapshot is rebuilt.
    #[must_use]
    pub const fn frame(&self) -> u64 {
        self.frame
    }

    /// Contiguous live-particle entries in deterministic ECS/emitter order.
    #[must_use]
    pub fn particles(&self) -> &[ParticleSnapshotEntry] {
        &self.particles
    }

    /// Packed live-particle ranges grouped by source emitter.
    #[must_use]
    pub fn emitters(&self) -> &[ParticleEmitterSnapshot] {
        &self.emitters
    }

    /// Number of live particles represented by this snapshot.
    #[must_use]
    pub fn len(&self) -> usize {
        self.particles.len()
    }

    /// Whether the most recently built snapshot contains no live particles.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.particles.is_empty()
    }

    /// Allocated entry capacity, useful for diagnostics and reuse validation.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.particles.capacity()
    }
}

/// Installs fixed-step simulation and main-world snapshot production.
#[derive(Clone, Copy, Debug, Default)]
pub struct ParticleRuntimePlugin;

impl Plugin for ParticleRuntimePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ParticleSnapshot>()
            .add_systems(FixedUpdate, step_emitters)
            .add_systems(Last, rebuild_snapshot);
    }
}

fn step_emitters(mut emitters: Query<&mut EmitterRuntimeComponent>, clocks: Res<EngineClocks>) {
    let fixed_delta = clocks.fixed().timestep().as_secs_f32();
    for mut emitter in emitters.iter_mut() {
        emitter.runtime_mut().step(fixed_delta);
    }
}

fn rebuild_snapshot(
    emitters: Query<(Entity, &EmitterRuntimeComponent)>,
    mut snapshot: ResMut<ParticleSnapshot>,
) {
    snapshot.particles.clear();
    snapshot.emitters.clear();
    for (entity, emitter) in emitters.iter() {
        let first_particle = snapshot.particles.len();
        snapshot
            .particles
            .extend(emitter.runtime().live_particles().map(|(slot, particle)| {
                ParticleSnapshotEntry {
                    emitter_entity: entity,
                    slot,
                    particle,
                }
            }));
        let particle_count = snapshot.particles.len() - first_particle;
        snapshot.emitters.push(ParticleEmitterSnapshot {
            emitter_entity: entity,
            first_particle: first_particle as u32,
            particle_count: particle_count as u32,
        });
    }
    snapshot.frame = snapshot.frame.wrapping_add(1);
}

/// Sets the fixed timestep used by the particle bridge.
///
/// The application fixed-loop driver uses the same clock for `FixedUpdate`;
/// this helper keeps tests and headless hosts explicit about the step size.
pub fn set_particle_fixed_timestep(app: &mut App, timestep: Duration) {
    assert!(
        !timestep.is_zero(),
        "particle fixed timestep must be non-zero"
    );
    app.world_mut()
        .resource_mut::<EngineClocks>()
        .fixed_mut()
        .set_timestep(timestep);
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_app::App;
    use prism_ecs::prelude::World;
    use prism_render_architecture::particle::emitter::{
        BurstSpawn, Emitter, EmitterShape, SpawnParams,
    };
    use prism_render_architecture::particle::runtime::EmitterRuntimeConfig;
    use prism_render_architecture::particle::{EmitterHandle, Vec3};

    fn runtime() -> EmitterRuntime {
        let config = EmitterRuntimeConfig {
            capacity: 8,
            bursts: alloc::vec![BurstSpawn {
                time: 0.1,
                count: 1,
            }],
            lifetime: 0.2,
            ..EmitterRuntimeConfig::default()
        };
        EmitterRuntime::new(
            Emitter::new(
                EmitterHandle(7),
                EmitterShape::Point,
                SpawnParams::default(),
            ),
            config,
            Vec3::ZERO,
            Vec3::ZERO,
        )
        .expect("test emitter configuration is valid")
    }

    fn run_phase(world: &mut World, phase: impl prism_ecs::prelude::ScheduleLabel) {
        world.run_schedule(phase);
    }

    #[test]
    fn fixed_update_advances_runtime_by_fixed_timestep() {
        let mut app = App::new();
        app.add_plugins(ParticleRuntimePlugin);
        set_particle_fixed_timestep(&mut app, Duration::from_millis(100));
        app.world_mut()
            .spawn(EmitterRuntimeComponent::new(runtime()));

        run_phase(app.world_mut(), FixedUpdate);
        run_phase(app.world_mut(), Last);
        let snapshot = app.world().resource::<ParticleSnapshot>();
        assert_eq!(snapshot.len(), 1);
        assert_eq!(snapshot.particles()[0].particle.age, 0.0);

        run_phase(app.world_mut(), FixedUpdate);
        run_phase(app.world_mut(), Last);
        let snapshot = app.world().resource::<ParticleSnapshot>();
        assert_eq!(snapshot.len(), 1);
        assert!((snapshot.particles()[0].particle.age - 0.1).abs() < 1.0e-6);
    }

    #[test]
    fn last_publishes_contiguous_live_particles_and_retires_expired_entries() {
        let mut app = App::new();
        app.add_plugins(ParticleRuntimePlugin);
        set_particle_fixed_timestep(&mut app, Duration::from_millis(100));
        let emitter_entity = app
            .world_mut()
            .spawn(EmitterRuntimeComponent::new(runtime()));

        run_phase(app.world_mut(), FixedUpdate);
        run_phase(app.world_mut(), Last);
        let first = app.world().resource::<ParticleSnapshot>();
        assert_eq!(first.particles()[0].emitter_entity, emitter_entity);
        assert_eq!(first.particles()[0].slot, 0);
        let capacity = first.capacity();
        let first_ptr = first.particles().as_ptr();

        run_phase(app.world_mut(), FixedUpdate);
        run_phase(app.world_mut(), Last);
        assert_eq!(app.world().resource::<ParticleSnapshot>().len(), 1);

        run_phase(app.world_mut(), FixedUpdate);
        run_phase(app.world_mut(), Last);
        let expired = app.world().resource::<ParticleSnapshot>();
        assert!(expired.is_empty());
        assert_eq!(expired.capacity(), capacity);
        assert_eq!(expired.particles().as_ptr(), first_ptr);
        assert_eq!(expired.frame(), 3);
    }

    #[test]
    fn snapshot_extracts_as_a_one_way_secondary_projection() {
        let mut main = World::new();
        main.insert_resource(ParticleSnapshot {
            frame: 11,
            particles: vec![ParticleSnapshotEntry {
                emitter_entity: Entity::from_bits(1_u64 << 32).expect("valid test entity"),
                slot: 3,
                particle: ParticleRecord {
                    position: Vec3::ZERO,
                    velocity: Vec3::ZERO,
                    age: 0.25,
                    lifetime: 1.0,
                    particle_id: 7,
                    size: 1.0,
                },
            }],
            emitters: vec![ParticleEmitterSnapshot {
                emitter_entity: Entity::from_bits(1_u64 << 32).expect("valid test entity"),
                first_particle: 0,
                particle_count: 1,
            }],
        });
        let mut secondary = World::new();
        extract_particle_snapshot(&main, &mut secondary);
        let extracted = secondary.resource::<ExtractedParticleSnapshot>();
        let first_ptr = extracted.particles.as_ptr();
        assert_eq!(extracted.frame(), 11);
        assert_eq!(extracted.len(), 1);
        assert_eq!(extracted.particles()[0].slot, 3);
        assert_eq!(extracted.emitters()[0].particle_count, 1);

        extract_particle_snapshot(&main, &mut secondary);
        assert_eq!(
            secondary
                .resource::<ExtractedParticleSnapshot>()
                .particles
                .as_ptr(),
            first_ptr
        );
    }

    #[test]
    fn snapshot_keeps_one_packed_range_per_emitter() {
        let mut app = App::new();
        app.add_plugins(ParticleRuntimePlugin);
        set_particle_fixed_timestep(&mut app, Duration::from_millis(100));
        let first = app
            .world_mut()
            .spawn(EmitterRuntimeComponent::new(runtime()));
        let second = app
            .world_mut()
            .spawn(EmitterRuntimeComponent::new(runtime()));

        run_phase(app.world_mut(), FixedUpdate);
        run_phase(app.world_mut(), Last);
        let snapshot = app.world().resource::<ParticleSnapshot>();
        assert_eq!(snapshot.emitters().len(), 2);
        assert_eq!(snapshot.emitters()[0].emitter_entity, first);
        assert_eq!(snapshot.emitters()[1].emitter_entity, second);
        assert_eq!(snapshot.emitters()[0].first_particle, 0);
        assert_eq!(snapshot.emitters()[0].particle_count, 1);
        assert_eq!(snapshot.emitters()[1].first_particle, 1);
        assert_eq!(snapshot.emitters()[1].particle_count, 1);
        assert_eq!(snapshot.len(), 2);
    }
}
