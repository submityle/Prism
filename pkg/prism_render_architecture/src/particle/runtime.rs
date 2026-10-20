//! Deterministic CPU particle runtime.
//!
//! This is the executable reference path for an Ember system. It owns the
//! emitter pools, applies the authored spawn policy, advances live particles
//! with the shared integrators and force field, and exposes frame statistics
//! that the GPU backend mirrors. The implementation is engine-neutral: it
//! depends only on the particle contracts in this crate and does not depend on
//! Bevy or any application ECS.

use alloc::vec::Vec;

use super::emitter::{build_spawn, sample_shape, BurstSpawn, Emitter, SpawnParams, UnitCursor};
use super::forces::ForceField;
use super::pool::{CapacityPolicy, ParticlePool};
use super::simulation::{hash_rng, integrate, unit_f32, Kinematics};
use super::{EmitterHandle, IntegratorKind, ParticleSystemHandle, Vec3};

/// A particle record stored in the CPU reference pool.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ParticleRecord {
    /// Stable slot position.
    pub position: Vec3,
    /// Current velocity.
    pub velocity: Vec3,
    /// Elapsed lifetime in seconds.
    pub age: f32,
    /// Particle lifetime in seconds.
    pub lifetime: f32,
    /// Stable identifier used by deterministic random streams.
    pub particle_id: u32,
    /// Authoring size carried into render extraction.
    pub size: f32,
}

/// Configuration shared by one emitter runtime.
#[derive(Clone, Debug, PartialEq)]
pub struct EmitterRuntimeConfig {
    /// Fixed pool capacity.
    pub capacity: u32,
    /// Behavior when a spawn request exceeds capacity.
    pub capacity_policy: CapacityPolicy,
    /// Continuous spawn rate in particles per second.
    pub rate_per_second: f32,
    /// One-shot bursts in emitter-local time.
    pub bursts: Vec<BurstSpawn>,
    /// Initial particle values.
    pub spawn: SpawnParams,
    /// Lifetime assigned to each new particle.
    pub lifetime: f32,
    /// Initial particle size.
    pub size: f32,
    /// Integration method.
    pub integrator: IntegratorKind,
    /// Acceleration and force terms.
    pub forces: ForceField,
    /// Deterministic emitter seed.
    pub seed: u32,
}

impl Default for EmitterRuntimeConfig {
    fn default() -> Self {
        Self {
            capacity: 1024,
            capacity_policy: CapacityPolicy::DiscardNewest,
            rate_per_second: 0.0,
            bursts: Vec::new(),
            spawn: SpawnParams::default(),
            lifetime: 1.0,
            size: 1.0,
            integrator: IntegratorKind::SemiImplicitEuler,
            forces: ForceField::new(),
            seed: 0,
        }
    }
}

impl EmitterRuntimeConfig {
    /// Validates configuration values before allocation.
    pub fn validate(&self) -> Result<(), RuntimeError> {
        if self.capacity == 0 {
            return Err(RuntimeError::ZeroCapacity);
        }
        if !self.rate_per_second.is_finite() || self.rate_per_second < 0.0 {
            return Err(RuntimeError::NonFinite("rate_per_second"));
        }
        if !self.lifetime.is_finite() || self.lifetime <= 0.0 {
            return Err(RuntimeError::NonFinite("lifetime"));
        }
        if !self.size.is_finite() || self.size < 0.0 {
            return Err(RuntimeError::NonFinite("size"));
        }
        if !self.spawn.speed.is_finite() || !self.spawn.inherit_velocity.is_finite() {
            return Err(RuntimeError::NonFinite("spawn"));
        }
        for burst in &self.bursts {
            if !burst.time.is_finite() || burst.time < 0.0 {
                return Err(RuntimeError::NonFinite("burst.time"));
            }
        }
        Ok(())
    }
}

/// Runtime errors that prevent construction of an invalid pool.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RuntimeError {
    /// The emitter cannot allocate a zero-sized pool.
    ZeroCapacity,
    /// A named floating-point input was non-finite or outside its domain.
    NonFinite(&'static str),
    /// An emitter handle is already present in the system.
    DuplicateEmitterHandle(EmitterHandle),
}

/// Per-frame counters returned by [`EmitterRuntime::step`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct EmitterStepStats {
    /// Requested spawns before capacity policy.
    pub requested_spawns: u32,
    /// Spawns accepted by the pool.
    pub spawned: u32,
    /// Spawns discarded by the capacity policy.
    pub discarded: u32,
    /// Particles killed by lifetime expiration.
    pub expired: u32,
    /// Number of live particles after the frame.
    pub alive: u32,
}

/// A single executable emitter instance.
#[derive(Clone, Debug, PartialEq)]
pub struct EmitterRuntime {
    emitter: Emitter,
    config: EmitterRuntimeConfig,
    pool: ParticlePool,
    particles: Vec<ParticleRecord>,
    origin: Vec3,
    emitter_velocity: Vec3,
    time: f32,
    frame: u32,
    next_particle_id: u32,
    last_stats: EmitterStepStats,
}

impl EmitterRuntime {
    /// Creates an emitter runtime with a validated fixed-capacity pool.
    pub fn new(
        emitter: Emitter,
        config: EmitterRuntimeConfig,
        origin: Vec3,
        emitter_velocity: Vec3,
    ) -> Result<Self, RuntimeError> {
        config.validate()?;
        let capacity = config.capacity as usize;
        Ok(Self {
            emitter,
            pool: ParticlePool::with_capacity(config.capacity),
            particles: alloc::vec![ParticleRecord::default(); capacity],
            config,
            origin,
            emitter_velocity,
            time: 0.0,
            frame: 0,
            next_particle_id: 0,
            last_stats: EmitterStepStats::default(),
        })
    }

    /// Stable emitter identity.
    #[must_use]
    pub const fn handle(&self) -> EmitterHandle {
        self.emitter.handle
    }

    /// Current live count.
    #[must_use]
    pub fn alive_count(&self) -> u32 {
        self.pool.alive_count()
    }

    /// Immutable pool view for render extraction and diagnostics.
    #[must_use]
    pub fn pool(&self) -> &ParticlePool {
        &self.pool
    }

    /// Returns a slot record when the slot is live.
    #[must_use]
    pub fn particle(&self, slot: u32) -> Option<ParticleRecord> {
        if self.pool.is_alive(slot) {
            self.particles.get(slot as usize).copied()
        } else {
            None
        }
    }

    /// Returns the last frame's counters.
    #[must_use]
    pub const fn last_stats(&self) -> EmitterStepStats {
        self.last_stats
    }

    /// Updates the emitter transform inputs used by subsequent spawns.
    pub fn set_transform(&mut self, origin: Vec3, emitter_velocity: Vec3) {
        self.origin = origin;
        self.emitter_velocity = emitter_velocity;
    }

    /// Advances one frame. Invalid or zero `dt` is a no-op and returns zeros.
    pub fn step(&mut self, dt: f32) -> EmitterStepStats {
        if !dt.is_finite() || dt <= 0.0 {
            self.last_stats = EmitterStepStats {
                alive: self.pool.alive_count(),
                ..EmitterStepStats::default()
            };
            return self.last_stats;
        }

        self.pool.begin_frame();
        self.time += dt;
        let mut expired = 0;
        for slot in 0..self.config.capacity {
            if !self.pool.is_alive(slot) {
                continue;
            }
            let record = &mut self.particles[slot as usize];
            let old = *record;
            let next = integrate(
                self.config.integrator,
                Kinematics::new(old.position, old.velocity),
                dt,
                |p, v| self.config.forces.accumulate(p, v),
            );
            record.position = next.position;
            record.velocity = next.velocity;
            record.age += dt;
            if record.age >= record.lifetime {
                self.pool.kill(slot);
                self.particles[slot as usize] = ParticleRecord::default();
                expired += 1;
            }
        }

        // Match the GPU frame contract: existing particles simulate and expire
        // before new slots are allocated, so a newborn is never aged during
        // the frame in which it is emitted.
        let requested = self.emitter.spawn_count(
            self.config.rate_per_second,
            dt,
            &self.config.bursts,
            self.time,
        );
        let slots = self
            .emitter
            .allocate(&mut self.pool, requested, self.config.capacity_policy);
        for &slot in &slots {
            let particle_id = self.next_particle_id;
            self.next_particle_id = self.next_particle_id.wrapping_add(1);
            let mut samples = [0.0; 8];
            for (stream, value) in samples.iter_mut().enumerate() {
                *value = unit_f32(hash_rng(
                    particle_id,
                    self.config.seed,
                    stream as u32,
                    self.frame,
                ));
            }
            let mut cursor = UnitCursor::new(&samples);
            let sample = sample_shape(self.emitter.shape, &mut cursor);
            let state = build_spawn(
                self.origin,
                self.emitter_velocity,
                sample,
                self.config.spawn,
            );
            self.particles[slot as usize] = ParticleRecord {
                position: state.position,
                velocity: state.velocity,
                age: 0.0,
                lifetime: self.config.lifetime,
                particle_id,
                size: self.config.size,
            };
        }

        self.frame = self.frame.wrapping_add(1);
        let spawned = self.pool.spawn_count();
        self.last_stats = EmitterStepStats {
            requested_spawns: requested,
            spawned,
            discarded: requested.saturating_sub(spawned),
            expired,
            alive: self.pool.alive_count(),
        };
        self.last_stats
    }
}

/// A complete engine-neutral particle system containing multiple emitters.
#[derive(Clone, Debug, PartialEq)]
pub struct ParticleSystemRuntime {
    handle: ParticleSystemHandle,
    emitters: Vec<EmitterRuntime>,
    frame: u32,
}

impl ParticleSystemRuntime {
    /// Creates an empty system.
    #[must_use]
    pub fn new(handle: ParticleSystemHandle) -> Self {
        Self {
            handle,
            emitters: Vec::new(),
            frame: 0,
        }
    }

    /// System identity.
    #[must_use]
    pub const fn handle(&self) -> ParticleSystemHandle {
        self.handle
    }

    /// Adds an emitter, rejecting duplicate handles.
    pub fn add_emitter(&mut self, emitter: EmitterRuntime) -> Result<(), RuntimeError> {
        if self
            .emitters
            .iter()
            .any(|existing| existing.handle() == emitter.handle())
        {
            return Err(RuntimeError::DuplicateEmitterHandle(emitter.handle()));
        }
        self.emitters.push(emitter);
        Ok(())
    }

    /// Advances every emitter in insertion order for deterministic results.
    pub fn step(&mut self, dt: f32) -> Vec<EmitterStepStats> {
        let mut stats = Vec::with_capacity(self.emitters.len());
        for emitter in &mut self.emitters {
            stats.push(emitter.step(dt));
        }
        self.frame = self.frame.wrapping_add(1);
        stats
    }

    /// Immutable emitter list for extraction and diagnostics.
    #[must_use]
    pub fn emitters(&self) -> &[EmitterRuntime] {
        &self.emitters
    }

    /// Current system frame.
    #[must_use]
    pub const fn frame(&self) -> u32 {
        self.frame
    }
}

#[cfg(test)]
mod tests {
    use super::super::emitter::{EmitterShape, SpawnParams};
    use super::super::forces::{Force, ForceEntry};
    use super::*;

    fn emitter(rate: f32, lifetime: f32, capacity: u32) -> EmitterRuntime {
        let config = EmitterRuntimeConfig {
            capacity,
            rate_per_second: rate,
            lifetime,
            spawn: SpawnParams {
                speed: 1.0,
                ..SpawnParams::default()
            },
            ..EmitterRuntimeConfig::default()
        };
        EmitterRuntime::new(
            Emitter::new(EmitterHandle(1), EmitterShape::Point, config.spawn),
            config,
            Vec3::ZERO,
            Vec3::ZERO,
        )
        .expect("valid emitter")
    }

    #[test]
    fn runtime_spawns_integrates_and_expires() {
        let mut runtime = emitter(10.0, 0.15, 8);
        let first = runtime.step(0.1);
        assert_eq!(first.spawned, 1);
        assert_eq!(first.alive, 1);
        let record = runtime.particle(0).expect("live particle");
        assert_eq!(
            record.position.z, 0.0,
            "new particles do not age on birth frame"
        );
        let second = runtime.step(0.1);
        assert_eq!(second.spawned, 1);
        assert_eq!(second.expired, 0);
        assert_eq!(second.alive, 2);
        assert!(runtime.particle(0).expect("first particle").position.z > 0.0);
        let third = runtime.step(0.1);
        assert_eq!(third.expired, 1);
        assert_eq!(third.alive, 2);
    }

    #[test]
    fn runtime_applies_force_field_deterministically() {
        let mut config = EmitterRuntimeConfig {
            capacity: 4,
            rate_per_second: 10.0,
            lifetime: 2.0,
            spawn: SpawnParams::default(),
            ..EmitterRuntimeConfig::default()
        };
        config.forces.push(ForceEntry::new(Force::Gravity {
            acceleration: Vec3::new(0.0, -10.0, 0.0),
        }));
        let make = || {
            EmitterRuntime::new(
                Emitter::new(EmitterHandle(3), EmitterShape::Point, config.spawn),
                config.clone(),
                Vec3::ZERO,
                Vec3::ZERO,
            )
            .expect("valid emitter")
        };
        let mut a = make();
        let mut b = make();
        a.step(0.1);
        b.step(0.1);
        assert_eq!(a.particle(0), b.particle(0));
        assert_eq!(a.particle(0).expect("live particle").position.y, 0.0);
        a.step(0.1);
        b.step(0.1);
        assert_eq!(a.particle(0), b.particle(0));
        assert!(a.particle(0).expect("live particle").position.y < 0.0);
    }

    #[test]
    fn capacity_policy_reports_dropped_spawns() {
        let mut runtime = emitter(100.0, 10.0, 2);
        let stats = runtime.step(0.1);
        assert_eq!(stats.requested_spawns, 10);
        assert_eq!(stats.spawned, 2);
        assert_eq!(stats.discarded, 8);
    }
}
