//! ECS fixed-step integration for material-point simulations.
//!
//! MPM is intentionally kept outside [`PhysicsWorldResource`]: it has a
//! particle/grid state model rather than rigid-body and soft-body arenas. This
//! module owns the ECS lifecycle and exposes the backend-independent particle
//! output while [`prism_physics_gpu::MpmRuntime`] owns CPU/GPU execution.

use std::collections::BTreeMap;

use prism_app::EngineClocks;
use prism_ecs::prelude::*;
use prism_physics_core::mpm::MpmSolver;
use prism_physics_gpu::{
    MpmBackend, MpmBackendPolicy, MpmFallbackReason, MpmRuntime, MpmRuntimeConfig, MpmRuntimeError,
};

/// Authored MPM state attached to an ECS entity.
///
/// The solver is copied into the fixed-step runtime when the entity is first
/// synchronized. Editing this component replaces the runtime at the next
/// [`crate::PhysicsSystems::Sync`] boundary; the live particle state is exposed by
/// [`MpmParticleSnapshot`] and is never written back into this authored value.
#[derive(Component, Clone, Debug)]
pub struct MpmBinding {
    /// Initial material-point, grid, material and numerical configuration.
    pub solver: MpmSolver,
    /// Backend acquisition policy.
    pub backend: MpmBackendPolicy,
    /// Maximum runtime steps accepted by one bridge advance call. The ECS
    /// bridge requests one step per fixed schedule invocation.
    pub max_steps_per_advance: usize,
}

impl MpmBinding {
    /// Creates a CPU-authoritative binding with a one-step budget.
    #[must_use]
    pub fn cpu(solver: MpmSolver) -> Self {
        Self::with_backend(solver, MpmBackendPolicy::Cpu, 1)
    }

    /// Creates a binding with an explicit CPU/GPU policy and bounded budget.
    #[must_use]
    pub const fn with_backend(
        solver: MpmSolver,
        backend: MpmBackendPolicy,
        max_steps_per_advance: usize,
    ) -> Self {
        Self {
            solver,
            backend,
            max_steps_per_advance,
        }
    }

    /// Creates a preferred-GPU binding. A missing adapter is reported and the
    /// same fixed-step runtime continues on the CPU reference.
    #[must_use]
    pub fn prefer_gpu(solver: MpmSolver) -> Self {
        Self::with_backend(solver, MpmBackendPolicy::PreferGpu, 1)
    }

    /// Returns a binding with the requested runtime step budget.
    #[must_use]
    pub fn with_step_budget(mut self, max_steps_per_advance: usize) -> Self {
        self.max_steps_per_advance = max_steps_per_advance;
        self
    }
}

/// Backend-neutral, versioned particle output published after a fixed step.
///
/// All columns are index-aligned and have the same length. `version` changes
/// exactly once for every successfully published fixed-step snapshot.
#[derive(Component, Clone, Debug, PartialEq)]
pub struct MpmParticleSnapshot {
    /// Monotonic snapshot sequence for this ECS entity, wrapping only at the
    /// representable `u64` boundary.
    pub version: u64,
    /// Fixed-step tick that produced this snapshot.
    pub tick: u64,
    /// World-space particle positions.
    pub positions: Vec<glam::Vec3>,
    /// World-space particle velocities.
    pub velocities: Vec<glam::Vec3>,
    /// APIC affine velocity matrices.
    pub affine: Vec<glam::Mat3>,
    /// Particle deformation gradients.
    pub deformation: Vec<glam::Mat3>,
    /// Snow plastic determinant per particle.
    pub plastic_det: Vec<f32>,
}

impl MpmParticleSnapshot {
    fn from_step(
        version: u64,
        tick: u64,
        particles: prism_physics_gpu::mpm::StepParticles,
    ) -> Self {
        Self {
            version,
            tick,
            positions: particles.positions,
            velocities: particles.velocities,
            affine: particles.affine,
            deformation: particles.deformation,
            plastic_det: particles.plastic_det,
        }
    }

    fn valid(&self) -> bool {
        let count = self.positions.len();
        self.velocities.len() == count
            && self.affine.len() == count
            && self.deformation.len() == count
            && self.plastic_det.len() == count
            && self.positions.iter().all(|v| v.is_finite())
            && self.velocities.iter().all(|v| v.is_finite())
            && self.affine.iter().all(|v| v.is_finite())
            && self.deformation.iter().all(|v| v.is_finite())
            && self.plastic_det.iter().all(|v| v.is_finite() && *v > 0.0)
    }
}

/// CPU or GPU authority that produced the current runtime state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MpmAuthority {
    /// The deterministic core MPM solver is authoritative.
    Cpu,
    /// The resident wgpu MPM pipeline is authoritative.
    Gpu,
}

/// A bridge diagnostic tied to an ECS entity, or to the whole MPM subsystem.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MpmIssue {
    /// Solver columns, grid, material or numerical values are invalid.
    InvalidAuthoring,
    /// The runtime could not be created under the requested policy.
    RuntimeInit(MpmRuntimeError),
    /// The authored solver dt does not match the engine fixed clock.
    FixedStepMismatch,
    /// The runtime capped the requested fixed-step work.
    StepBudgetCapped,
    /// GPU initialization selected an explicit CPU fallback.
    BackendFallback(MpmFallbackReason),
    /// Backend output could not be read into a portable snapshot.
    Snapshot(MpmRuntimeError),
    /// A snapshot was non-finite or had inconsistent columns.
    InvalidSnapshot,
}

/// One diagnostic emitted by the latest fixed-step MPM tick.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MpmDiagnostic {
    /// The entity whose runtime produced the diagnostic, or `None` for a
    /// subsystem-wide condition.
    pub entity: Option<Entity>,
    /// Diagnostic classification.
    pub issue: MpmIssue,
}

/// Latest fixed-step MPM diagnostics. History is intentionally bounded to one
/// tick; consumers that need a durable record should copy this resource.
#[derive(Resource, Default)]
pub struct MpmDiagnostics {
    /// Number of fixed MPM ticks attempted.
    pub tick: u64,
    /// Fixed simulation time accepted by the latest tick, or zero when no
    /// runtime advanced. This is wall-clock simulation time, not the sum of
    /// work across independent runtimes; [`Self::steps`] reports aggregate
    /// runtime work.
    pub simulated_dt: f32,
    /// Number of runtime steps executed by the latest tick.
    pub steps: usize,
    /// Diagnostics emitted while processing the latest tick.
    pub issues: Vec<MpmDiagnostic>,
}

struct RuntimeEntry {
    runtime: MpmRuntime,
    authored: MpmSolver,
    authored_backend: MpmBackendPolicy,
    authored_step_budget: usize,
    latest: Option<MpmParticleSnapshot>,
    publish_pending: bool,
}

/// ECS-owned MPM runtimes and their latest portable snapshots.
///
/// Runtime internals remain private so entity lifecycle, authority changes and
/// output publication cannot be bypassed between fixed-step phases.
#[derive(Resource, Default)]
pub struct MpmRuntimeResource {
    entries: BTreeMap<Entity, RuntimeEntry>,
}

impl MpmRuntimeResource {
    /// Returns whether an entity currently owns an MPM runtime.
    #[must_use]
    pub fn contains(&self, entity: Entity) -> bool {
        self.entries.contains_key(&entity)
    }

    /// Returns the latest published snapshot for an entity.
    #[must_use]
    pub fn snapshot(&self, entity: Entity) -> Option<&MpmParticleSnapshot> {
        self.entries
            .get(&entity)
            .and_then(|entry| entry.latest.as_ref())
    }

    /// Returns the current runtime authority and counters.
    #[must_use]
    pub fn info(&self, entity: Entity) -> Option<MpmRuntimeInfo> {
        self.entries.get(&entity).map(|entry| MpmRuntimeInfo {
            authority: authority(entry.runtime.backend()),
            fallback: entry.runtime.fallback_reason(),
            fixed_tick: entry.runtime.fixed_tick(),
            snapshot_version: entry.latest.as_ref().map_or(0, |s| s.version),
            particle_count: entry.runtime.cpu_solver().particles.len(),
        })
    }

    /// Removes an entity runtime and returns its latest snapshot if present.
    pub fn remove(&mut self, entity: Entity) -> Option<MpmParticleSnapshot> {
        self.entries.remove(&entity).and_then(|entry| entry.latest)
    }

    /// Number of ECS-owned MPM runtimes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no ECS-owned MPM runtime exists.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Read-only runtime state exposed without exposing backend handles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MpmRuntimeInfo {
    /// Current authority.
    pub authority: MpmAuthority,
    /// Initialization fallback, if the requested policy preferred GPU.
    pub fallback: Option<MpmFallbackReason>,
    /// Number of fixed steps executed by this runtime.
    pub fixed_tick: u64,
    /// Latest published snapshot version, or zero before publication.
    pub snapshot_version: u64,
    /// Number of particles in the runtime.
    pub particle_count: usize,
}

fn authority(backend: MpmBackend) -> MpmAuthority {
    match backend {
        MpmBackend::Cpu => MpmAuthority::Cpu,
        MpmBackend::Gpu => MpmAuthority::Gpu,
    }
}

fn same_solver(a: &MpmSolver, b: &MpmSolver) -> bool {
    a.particles == b.particles
        && a.grid == b.grid
        && a.material == b.material
        && a.config == b.config
}

fn valid_solver(solver: &MpmSolver) -> bool {
    let cfg = solver.config;
    let material = solver.material;
    cfg.dt.is_finite()
        && cfg.dt > 0.0
        && cfg.gravity.is_finite()
        && solver.grid.nx() >= 5
        && solver.grid.ny() >= 5
        && solver.grid.nz() >= 5
        && solver.grid.dx().is_finite()
        && solver.grid.dx() > 0.0
        && solver.grid.origin().is_finite()
        && cfg.boundary_thickness >= 2
        && cfg.boundary_thickness.saturating_mul(2) < solver.grid.nx()
        && cfg.boundary_thickness.saturating_mul(2) < solver.grid.ny()
        && cfg.boundary_thickness.saturating_mul(2) < solver.grid.nz()
        && material.youngs_modulus.is_finite()
        && material.youngs_modulus > 0.0
        && material.poisson_ratio.is_finite()
        && material.poisson_ratio > -1.0
        && material.poisson_ratio < 0.5
        && material.density.is_finite()
        && material.density > 0.0
        && solver.particles.positions().iter().all(|v| v.is_finite())
        && solver.particles.velocities().iter().all(|v| v.is_finite())
        && solver.particles.affine().iter().all(|v| v.is_finite())
        && solver.particles.deformation().iter().all(|v| v.is_finite())
        && solver
            .particles
            .masses()
            .iter()
            .all(|v| v.is_finite() && *v > 0.0)
        && solver
            .particles
            .volumes()
            .iter()
            .all(|v| v.is_finite() && *v > 0.0)
        && solver
            .particles
            .plastic_det()
            .iter()
            .all(|v| v.is_finite() && *v > 0.0)
}

fn issue(diagnostics: &mut MpmDiagnostics, entity: Option<Entity>, issue: MpmIssue) {
    diagnostics.issues.push(MpmDiagnostic { entity, issue });
}

pub(crate) fn sync(world: &mut World) {
    let mut runtimes = world.remove_resource::<MpmRuntimeResource>().unwrap();
    let mut diagnostics = world.remove_resource::<MpmDiagnostics>().unwrap();
    diagnostics.tick = diagnostics.tick.wrapping_add(1);
    diagnostics.simulated_dt = 0.0;
    diagnostics.steps = 0;
    diagnostics.issues.clear();

    let live: BTreeMap<_, _> = world
        .query::<(Entity, &MpmBinding)>()
        .iter(world)
        .map(|(entity, binding)| (entity, binding.clone()))
        .collect();

    let stale: Vec<_> = runtimes
        .entries
        .keys()
        .copied()
        .filter(|entity| !live.contains_key(entity))
        .collect();
    for entity in stale {
        runtimes.remove(entity);
        world.remove::<MpmParticleSnapshot>(entity);
    }

    for (entity, binding) in live {
        if !valid_solver(&binding.solver) || binding.max_steps_per_advance == 0 {
            runtimes.remove(entity);
            world.remove::<MpmParticleSnapshot>(entity);
            issue(&mut diagnostics, Some(entity), MpmIssue::InvalidAuthoring);
            continue;
        }

        let rebuild = runtimes.entries.get(&entity).is_none_or(|entry| {
            !same_solver(&entry.authored, &binding.solver)
                || entry.authored_backend != binding.backend
                || entry.authored_step_budget != binding.max_steps_per_advance
        });
        if !rebuild {
            if let Some(entry) = runtimes.entries.get(&entity) {
                if let Some(reason) = entry.runtime.fallback_reason() {
                    issue(
                        &mut diagnostics,
                        Some(entity),
                        MpmIssue::BackendFallback(reason),
                    );
                }
            }
            continue;
        }

        let config = MpmRuntimeConfig {
            backend: binding.backend,
            max_steps_per_advance: binding.max_steps_per_advance,
        };
        match MpmRuntime::new(binding.solver.clone(), config) {
            Ok(runtime) => {
                if let Some(reason) = runtime.fallback_reason() {
                    issue(
                        &mut diagnostics,
                        Some(entity),
                        MpmIssue::BackendFallback(reason),
                    );
                }
                runtimes.entries.insert(
                    entity,
                    RuntimeEntry {
                        runtime,
                        authored: binding.solver,
                        authored_backend: binding.backend,
                        authored_step_budget: binding.max_steps_per_advance,
                        latest: None,
                        publish_pending: false,
                    },
                );
            }
            Err(error) => {
                runtimes.remove(entity);
                world.remove::<MpmParticleSnapshot>(entity);
                issue(&mut diagnostics, Some(entity), MpmIssue::RuntimeInit(error));
            }
        }
    }

    world.insert_resource(runtimes);
    world.insert_resource(diagnostics);
}

pub(crate) fn step(world: &mut World) {
    let Some(clocks) = world.get_resource::<EngineClocks>() else {
        return;
    };
    let fixed_dt = clocks.fixed().delta_secs();
    let mut runtimes = world.remove_resource::<MpmRuntimeResource>().unwrap();
    let mut diagnostics = world.remove_resource::<MpmDiagnostics>().unwrap();
    diagnostics.simulated_dt = 0.0;
    let mut entities: Vec<_> = runtimes.entries.keys().copied().collect();
    entities.sort();
    for entity in entities {
        let entry = runtimes
            .entries
            .get_mut(&entity)
            .expect("sorted runtime entity must remain present");
        entry.publish_pending = false;
        let runtime_dt = entry.runtime.fixed_dt();
        let matches = fixed_dt.is_finite()
            && fixed_dt > 0.0
            && runtime_dt.is_finite()
            && (runtime_dt - fixed_dt).abs()
                <= 4.0 * f32::EPSILON * runtime_dt.abs().max(fixed_dt.abs()).max(1.0);
        if !matches {
            issue(&mut diagnostics, Some(entity), MpmIssue::FixedStepMismatch);
            continue;
        }
        let report = entry.runtime.advance(1);
        entry.publish_pending = report.steps > 0;
        diagnostics.steps = diagnostics.steps.saturating_add(report.steps);
        if report.steps > 0 {
            diagnostics.simulated_dt = fixed_dt;
        }
        if report.capped {
            issue(&mut diagnostics, Some(entity), MpmIssue::StepBudgetCapped);
        }
    }
    world.insert_resource(runtimes);
    world.insert_resource(diagnostics);
}

pub(crate) fn writeback(world: &mut World) {
    let tick = world.resource::<MpmDiagnostics>().tick;
    let mut runtimes = world.remove_resource::<MpmRuntimeResource>().unwrap();
    let mut diagnostics = world.remove_resource::<MpmDiagnostics>().unwrap();
    let mut outputs = Vec::new();
    let mut entities: Vec<_> = runtimes.entries.keys().copied().collect();
    entities.sort();
    for entity in entities {
        let entry = runtimes
            .entries
            .get_mut(&entity)
            .expect("sorted runtime entity must remain present");
        if !std::mem::take(&mut entry.publish_pending) {
            continue;
        }
        let version = entry
            .latest
            .as_ref()
            .map_or(1, |snapshot| snapshot.version.wrapping_add(1));
        let snapshot = match entry.runtime.snapshot() {
            Ok(particles) => MpmParticleSnapshot::from_step(version, tick, particles),
            Err(error) => {
                issue(&mut diagnostics, Some(entity), MpmIssue::Snapshot(error));
                continue;
            }
        };
        if !snapshot.valid() {
            issue(&mut diagnostics, Some(entity), MpmIssue::InvalidSnapshot);
            continue;
        }
        entry.latest = Some(snapshot.clone());
        outputs.push((entity, snapshot));
    }
    for (entity, snapshot) in outputs {
        if world.get::<MpmBinding>(entity).is_some() {
            world.insert(entity, (snapshot,));
        }
    }
    world.insert_resource(runtimes);
    world.insert_resource(diagnostics);
}
