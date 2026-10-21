//! ECS fixed-step integration for material-point simulations.
//!
//! MPM is intentionally kept outside [`PhysicsWorldResource`]: it has a
//! particle/grid state model rather than rigid-body and soft-body arenas. This
//! module owns the ECS lifecycle and exposes the backend-independent particle
//! output while [`prism_physics_gpu::MpmRuntime`] owns CPU/GPU execution.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use prism_app::EngineClocks;
use prism_ecs::prelude::*;
use prism_physics_core::mpm::MpmSolver;
use prism_physics_core::reconstruct::{
    triangulate_bounded, FieldBudget, FieldBuildError, ScalarField, SurfaceBudget,
    SurfaceBuildError, SurfaceMesh,
};
use prism_physics_gpu::context::GpuContext;
use prism_physics_gpu::mpm::gpu::surface::{
    ResidentSurface, ResidentSurfaceConfig, ResidentSurfaceError,
};
use prism_physics_gpu::{
    MpmBackend, MpmBackendPolicy, MpmFallbackReason, MpmRuntime, MpmRuntimeConfig, MpmRuntimeError,
};

/// Optional native host shared with a resident-surface consumer. Adopt once,
/// then clone this context into all producers/consumers. Changing its adoption
/// identity (or adding/removing the resource) rebuilds runtime epochs; replacing
/// it with a matching clone does not. Old leases remain independently valid on
/// their old device until consumers retire them.
#[derive(Resource, Clone)]
pub struct MpmGpuHost(pub GpuContext);

/// Opt-in GPU surface publication. The portable CPU path is unchanged unless
/// particle readback is explicitly disabled here. CPU fallback and requested
/// CPU surface reconstruction still publish portable particle snapshots.
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct MpmResidentSurfaceConfig {
    /// Fixed sampling domain, kernel and publication resource ceilings.
    pub surface: ResidentSurfaceConfig,
    /// Whether to also publish portable particle snapshots on the GPU backend.
    /// CPU fallback or requested CPU surface reconstruction still requires them.
    pub readback_particles: bool,
}

impl MpmResidentSurfaceConfig {
    /// Requests GPU-only output, without implicit particle/geometry readback.
    #[must_use]
    pub const fn resident_only(surface: ResidentSurfaceConfig) -> Self {
        Self {
            surface,
            readback_particles: false,
        }
    }
}

/// Immutable native publication; retain a clone through confirmed GPU
/// completion. This is not a portable rollback checkpoint or a completion fence.
#[derive(Component, Clone, Debug)]
pub struct MpmResidentSurfaceSnapshot {
    /// Runtime incarnation; changes when the binding or native host is replaced.
    pub epoch: u64,
    /// Advances on successful publication, including configuration-only edits.
    pub revision: u64,
    /// Source simulation version, independent of CPU readback policy.
    pub version: u64,
    /// Bridge tick of the last accepted simulation step, not the retry tick.
    pub tick: u64,
    /// Native buffer lease to retain through confirmed consumer GPU completion.
    pub surface: ResidentSurface,
}

/// Authored MPM state attached to an ECS entity.
///
/// The solver is copied into the fixed-step runtime when the entity is first
/// synchronized. Editing the solver, backend or step budget replaces the runtime
/// at the next [`crate::PhysicsSystems::Sync`] boundary; surface-only edits retain
/// its simulation epoch. Publication observers may edit authoring in place: old
/// outputs are retired before writeback returns, then retried at the next Sync.
/// The live particle state is exposed by [`MpmParticleSnapshot`] and is never
/// written back into this authored value. Edits must use ECS change detection.
#[derive(Component, Clone, Debug)]
pub struct MpmBinding {
    /// Initial material-point, grid, material and numerical configuration.
    pub solver: MpmSolver,
    /// Backend acquisition policy.
    pub backend: MpmBackendPolicy,
    /// Maximum runtime steps accepted by one bridge advance call. The ECS
    /// bridge requests one step per fixed schedule invocation.
    pub max_steps_per_advance: usize,
    /// Optional CPU surface reconstruction performed after each published step.
    pub surface: Option<MpmSurfaceConfig>,
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
            surface: None,
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

    /// Enables bounded particle-cloud surface reconstruction after each step.
    #[must_use]
    pub fn with_surface_reconstruction(mut self, config: MpmSurfaceConfig) -> Self {
        self.surface = Some(config);
        self
    }
}

/// Bounded particle-to-isosurface reconstruction settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MpmSurfaceConfig {
    /// Particle kernel support radius in world units.
    pub radius: f32,
    /// Scalar-field grid spacing in world units.
    pub dx: f32,
    /// Density-field isovalue.
    pub iso: f32,
    /// Number of empty grid nodes padded beyond the support radius.
    pub padding_cells: usize,
    /// Maximum scalar-field nodes allocated per reconstruction.
    pub max_field_nodes: usize,
    /// Maximum cells marched per reconstruction.
    pub max_cells: usize,
    /// Maximum particle/grid-node kernel evaluations.
    pub max_sample_visits: usize,
    /// Maximum generated unique surface vertices.
    pub max_surface_vertices: usize,
    /// Maximum generated surface triangles.
    pub max_surface_triangles: usize,
}

impl Default for MpmSurfaceConfig {
    fn default() -> Self {
        Self {
            radius: 0.1,
            dx: 0.05,
            iso: 0.1,
            padding_cells: 1,
            max_field_nodes: 2_000_000,
            max_cells: 300_000,
            max_sample_visits: 64_000_000,
            max_surface_vertices: 1_000_000,
            max_surface_triangles: 2_000_000,
        }
    }
}

impl MpmSurfaceConfig {
    fn valid(self) -> bool {
        self.radius.is_finite()
            && self.radius > 0.0
            && self.dx.is_finite()
            && self.dx > 0.0
            && self.iso.is_finite()
            && self.max_field_nodes >= 8
            && self.max_cells > 0
            && self.max_sample_visits > 0
            && self.max_surface_vertices > 0
            && self.max_surface_triangles > 0
    }
}

/// Backend-neutral, versioned particle output published after a fixed step.
///
/// All columns are index-aligned and have the same length. `version` changes
/// exactly once for every successfully published fixed-step snapshot.
#[derive(Component, Clone, Debug, PartialEq)]
pub struct MpmParticleSnapshot {
    /// Runtime generation, unique within the owning runtime resource.
    pub epoch: u64,
    /// Monotonic snapshot sequence within this runtime generation.
    pub version: u64,
    /// Fixed-step tick that produced this snapshot.
    pub tick: u64,
    /// World-space particle positions.
    pub positions: Arc<[glam::Vec3]>,
    /// World-space particle velocities.
    pub velocities: Arc<[glam::Vec3]>,
    /// APIC affine velocity matrices.
    pub affine: Arc<[glam::Mat3]>,
    /// Particle deformation gradients.
    pub deformation: Arc<[glam::Mat3]>,
    /// Snow plastic determinant per particle.
    pub plastic_det: Arc<[f32]>,
}

/// Versioned renderer-independent isosurface generated from one MPM snapshot.
#[derive(Component, Clone, Debug, PartialEq)]
pub struct MpmSurfaceSnapshot {
    /// Runtime generation shared with the source particle snapshot.
    pub epoch: u64,
    /// Mesh publication sequence within the generation. Configuration-only
    /// edits can advance this without advancing the source particle version.
    pub revision: u64,
    /// Source particle snapshot version.
    pub version: u64,
    /// Fixed-step tick shared with the source particle snapshot.
    pub tick: u64,
    /// Reconstructed world-space triangle mesh.
    pub mesh: Arc<SurfaceMesh>,
}

impl MpmParticleSnapshot {
    fn from_step(
        epoch: u64,
        version: u64,
        tick: u64,
        particles: prism_physics_gpu::mpm::StepParticles,
    ) -> Self {
        Self {
            epoch,
            version,
            tick,
            positions: particles.positions.into(),
            velocities: particles.velocities.into(),
            affine: particles.affine.into(),
            deformation: particles.deformation.into(),
            plastic_det: particles.plastic_det.into(),
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
    /// Native publication rejected before submission. Data-dependent GPU
    /// failures are instead exposed by the lease status/zero indirect draw.
    ResidentSurface(ResidentSurfaceError),
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
    /// Surface reconstruction settings contain invalid values.
    InvalidSurfaceConfig,
    /// Field sampling, cell work or surface output exceeds an authored ceiling.
    SurfaceBudgetExceeded,
    /// The allocator refused the bounded field, mesh or edge-cache allocation.
    SurfaceAllocationFailed,
    /// Surface reconstruction produced invalid mesh data.
    InvalidSurface,
    /// A snapshot was non-finite or had inconsistent columns.
    InvalidSnapshot,
    /// A generation or publication sequence exhausted its representable range.
    SequenceExhausted,
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
    epoch: u64,
    authored_tick: Tick,
    runtime: MpmRuntime,
    authored: MpmSolver,
    authored_backend: MpmBackendPolicy,
    authored_step_budget: usize,
    latest: Option<MpmParticleSnapshot>,
    publish_pending: bool,
    surface_config: Option<MpmSurfaceConfig>,
    surface_pending: bool,
    surface_revision: u64,
    host: Option<GpuContext>,
    resident_config: Option<MpmResidentSurfaceConfig>,
    resident_pending: bool,
    resident_revision: u64,
    source_version: u64,
    source_tick: u64,
}

/// ECS-owned MPM runtimes and their latest portable snapshots.
///
/// Runtime internals remain private so entity lifecycle, authority changes and
/// output publication cannot be bypassed between fixed-step phases.
#[derive(Resource, Default)]
pub struct MpmRuntimeResource {
    entries: BTreeMap<Entity, RuntimeEntry>,
    last_epoch: u64,
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
            epoch: entry.epoch,
            authority: authority(entry.runtime.backend()),
            fallback: entry.runtime.fallback_reason(),
            fixed_tick: entry.runtime.fixed_tick(),
            snapshot_version: entry.source_version,
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
    /// Runtime generation, changed whenever authoring replaces the simulation.
    pub epoch: u64,
    /// Current authority.
    pub authority: MpmAuthority,
    /// Initialization fallback, if the requested policy preferred GPU.
    pub fallback: Option<MpmFallbackReason>,
    /// Number of fixed steps executed by this runtime.
    pub fixed_tick: u64,
    /// Latest accepted source version (also advances without CPU readback).
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

#[derive(Resource, Default)]
struct RemovedAuthoring(BTreeSet<Entity>, BTreeSet<Entity>);

pub(crate) fn init(world: &mut World) {
    if !world.contains_resource::<RemovedAuthoring>() {
        world.init_resource::<RemovedAuthoring>();
        world.observe::<MpmBinding, _>(LifecycleEvent::Remove, |ctx| {
            ctx.world
                .resource_mut::<RemovedAuthoring>()
                .0
                .insert(ctx.entity);
        });
        world.observe::<MpmResidentSurfaceConfig, _>(LifecycleEvent::Remove, |ctx| {
            ctx.world
                .resource_mut::<RemovedAuthoring>()
                .1
                .insert(ctx.entity);
        });
    }
}

pub(crate) fn sync(world: &mut World) {
    let host = world
        .get_resource::<MpmGpuHost>()
        .map(|host| host.0.clone());
    let mut runtimes = world.remove_resource::<MpmRuntimeResource>().unwrap();
    let mut diagnostics = world.remove_resource::<MpmDiagnostics>().unwrap();
    diagnostics.simulated_dt = 0.0;
    diagnostics.steps = 0;
    diagnostics.issues.clear();
    let Some(tick) = diagnostics.tick.checked_add(1) else {
        let entities: Vec<_> = runtimes.entries.keys().copied().collect();
        runtimes.entries.clear();
        issue(&mut diagnostics, None, MpmIssue::SequenceExhausted);
        world.insert_resource(runtimes);
        world.insert_resource(diagnostics);
        for entity in entities {
            world.remove::<MpmParticleSnapshot>(entity);
            world.remove::<MpmSurfaceSnapshot>(entity);
            world.remove::<MpmResidentSurfaceSnapshot>(entity);
        }
        return;
    };
    diagnostics.tick = tick;
    let removed = std::mem::take(&mut world.resource_mut::<RemovedAuthoring>().0);
    let removed_resident = std::mem::take(&mut world.resource_mut::<RemovedAuthoring>().1);
    let mut clear_particles = BTreeSet::new();
    let mut clear_surfaces = BTreeSet::new();
    let mut clear_resident = BTreeSet::new();

    let live: BTreeMap<_, _> = world
        .query::<(Entity, Ref<MpmBinding>)>()
        .iter(world)
        .map(|(entity, binding)| (entity, (binding.changed_tick(), binding.surface)))
        .collect();

    let stale: Vec<_> = runtimes
        .entries
        .keys()
        .copied()
        .filter(|entity| removed.contains(entity) || !live.contains_key(entity))
        .collect();
    for entity in stale {
        runtimes.remove(entity);
        clear_particles.insert(entity);
        clear_surfaces.insert(entity);
        clear_resident.insert(entity);
    }

    for (entity, (authored_tick, surface)) in live {
        let resident_config = world.get::<MpmResidentSurfaceConfig>(entity).copied();
        if resident_config.is_none() {
            clear_resident.insert(entity);
        }
        // Configuration cleanup is independent of successful simulation work.
        // A rejected step must never keep a disabled or incompatible surface.
        if surface.is_none_or(|config| !config.valid()) {
            clear_surfaces.insert(entity);
        }
        if surface.is_some_and(|config| !config.valid()) {
            issue(
                &mut diagnostics,
                Some(entity),
                MpmIssue::InvalidSurfaceConfig,
            );
        }
        if let Some(entry) = runtimes.entries.get_mut(&entity) {
            if !same_resident_config(entry.resident_config, resident_config)
                || removed_resident.contains(&entity)
            {
                clear_resident.insert(entity);
                entry.resident_config = resident_config;
                entry.resident_pending = true;
                // Readback can be re-enabled without advancing simulation.
                entry.surface_pending = true;
            }
            if entry.runtime.backend() == MpmBackend::Gpu
                && resident_config.is_some_and(|config| !config.readback_particles)
                && surface.is_none()
            {
                entry.latest = None;
                clear_particles.insert(entity);
            }
            if !same_surface_config(entry.surface_config, surface) {
                clear_surfaces.insert(entity);
                entry.surface_config = surface;
                entry.surface_pending = true;
            }
            entry.authored_tick.check_tick(world.change_tick());
            if entry.authored_tick == authored_tick && same_host(entry.host.as_ref(), host.as_ref())
            {
                if let Some(reason) = entry.runtime.fallback_reason() {
                    issue(
                        &mut diagnostics,
                        Some(entity),
                        MpmIssue::BackendFallback(reason),
                    );
                }
                continue;
            }
        }
        // Only authoring changes copy/compare the potentially large initial
        // particle/grid state. Unchanged fixed ticks are O(entity count).
        let binding = world.get::<MpmBinding>(entity).unwrap().clone();
        if !valid_solver(&binding.solver) || binding.max_steps_per_advance == 0 {
            runtimes.remove(entity);
            clear_particles.insert(entity);
            clear_surfaces.insert(entity);
            clear_resident.insert(entity);
            issue(&mut diagnostics, Some(entity), MpmIssue::InvalidAuthoring);
            continue;
        }

        let rebuild = runtimes.entries.get(&entity).is_none_or(|entry| {
            !same_solver(&entry.authored, &binding.solver)
                || entry.authored_backend != binding.backend
                || entry.authored_step_budget != binding.max_steps_per_advance
                || !same_host(entry.host.as_ref(), host.as_ref())
        });
        if !rebuild {
            if let Some(entry) = runtimes.entries.get_mut(&entity) {
                entry.authored_tick = authored_tick;
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
        // An old generation's output is invalid even if initializing or
        // stepping its replacement fails.
        clear_particles.insert(entity);
        clear_surfaces.insert(entity);
        clear_resident.insert(entity);
        let Some(epoch) = runtimes.last_epoch.checked_add(1) else {
            runtimes.remove(entity);
            issue(&mut diagnostics, Some(entity), MpmIssue::SequenceExhausted);
            continue;
        };
        let runtime = match host.as_ref() {
            Some(host) => {
                MpmRuntime::with_gpu_context(binding.solver.clone(), config, host.clone())
            }
            None => MpmRuntime::new(binding.solver.clone(), config),
        };
        match runtime {
            Ok(runtime) => {
                runtimes.last_epoch = epoch;
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
                        epoch,
                        authored_tick,
                        runtime,
                        authored: binding.solver,
                        authored_backend: binding.backend,
                        authored_step_budget: binding.max_steps_per_advance,
                        latest: None,
                        publish_pending: false,
                        surface_config: binding.surface,
                        surface_pending: true,
                        surface_revision: 0,
                        host: host.clone(),
                        resident_config,
                        resident_pending: false,
                        resident_revision: 0,
                        source_version: 0,
                        source_tick: 0,
                    },
                );
            }
            Err(error) => {
                runtimes.remove(entity);
                issue(&mut diagnostics, Some(entity), MpmIssue::RuntimeInit(error));
            }
        }
    }

    world.insert_resource(runtimes);
    world.insert_resource(diagnostics);
    // Structural observers must see the reconciled public resources, not a
    // temporarily missing runtime/diagnostic resource during exclusive work.
    for entity in clear_particles {
        world.remove::<MpmParticleSnapshot>(entity);
    }
    for entity in clear_surfaces {
        world.remove::<MpmSurfaceSnapshot>(entity);
    }
    for entity in clear_resident {
        world.remove::<MpmResidentSurfaceSnapshot>(entity);
    }
}

fn same_host(a: Option<&GpuContext>, b: Option<&GpuContext>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => a.shares_device(b),
        _ => false,
    }
}

// Configuration identity must be reflexive even for invalid authored NaNs.
// Such a config rejects that surface, not otherwise valid portable outputs.
fn same_surface_config(a: Option<MpmSurfaceConfig>, b: Option<MpmSurfaceConfig>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => {
            a.radius.to_bits() == b.radius.to_bits()
                && a.dx.to_bits() == b.dx.to_bits()
                && a.iso.to_bits() == b.iso.to_bits()
                && a.padding_cells == b.padding_cells
                && a.max_field_nodes == b.max_field_nodes
                && a.max_cells == b.max_cells
                && a.max_sample_visits == b.max_sample_visits
                && a.max_surface_vertices == b.max_surface_vertices
                && a.max_surface_triangles == b.max_surface_triangles
        }
        _ => false,
    }
}

fn same_resident_config(
    a: Option<MpmResidentSurfaceConfig>,
    b: Option<MpmResidentSurfaceConfig>,
) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => {
            a.readback_particles == b.readback_particles
                && a.surface.origin.to_array().map(f32::to_bits)
                    == b.surface.origin.to_array().map(f32::to_bits)
                && a.surface.nodes == b.surface.nodes
                && a.surface.dx.to_bits() == b.surface.dx.to_bits()
                && a.surface.radius.to_bits() == b.surface.radius.to_bits()
                && a.surface.iso.to_bits() == b.surface.iso.to_bits()
                && a.surface.max_field_nodes == b.surface.max_field_nodes
                && a.surface.max_cells == b.surface.max_cells
                && a.surface.max_sample_visits == b.surface.max_sample_visits
                && a.surface.max_vertices == b.surface.max_vertices
                && a.surface.max_live_publications == b.surface.max_live_publications
        }
        _ => false,
    }
}

fn runtime_authoring_matches(world: &World, entity: Entity, epoch: u64) -> bool {
    if world.resource::<RemovedAuthoring>().0.contains(&entity) {
        return false;
    }
    let Some(binding) = world.get_ref::<MpmBinding>(entity) else {
        return false;
    };
    let Some(entry) = world.resource::<MpmRuntimeResource>().entries.get(&entity) else {
        return false;
    };
    let mut authored_tick = entry.authored_tick;
    authored_tick.check_tick(world.change_tick());
    entry.epoch == epoch
        && entry.authored_backend == binding.backend
        && entry.authored_step_budget == binding.max_steps_per_advance
        && same_host(
            entry.host.as_ref(),
            world.get_resource::<MpmGpuHost>().map(|host| &host.0),
        )
        // Multiple observer writes can share a tick. In that case compare
        // values, not just ticks; ordinary unchanged frames avoid scanning the
        // potentially large authored solver. Bypassing change detection is not
        // supported by this authoring contract.
        && ((authored_tick == binding.changed_tick()
            && binding.changed_tick() != world.change_tick())
            || same_solver(&entry.authored, &binding.solver))
}

fn publication_authoring_matches(world: &World, entity: Entity, epoch: u64) -> bool {
    if !runtime_authoring_matches(world, entity, epoch) {
        return false;
    }
    let entry = &world.resource::<MpmRuntimeResource>().entries[&entity];
    !world.resource::<RemovedAuthoring>().1.contains(&entity)
        && same_surface_config(
            world.get::<MpmBinding>(entity).unwrap().surface,
            entry.surface_config,
        )
        && same_resident_config(
            world.get::<MpmResidentSurfaceConfig>(entity).copied(),
            entry.resident_config,
        )
}

// Reconcile before invoking removal observers, so they cannot obtain a stale
// portable snapshot through MpmRuntimeResource either. Configuration-only edits
// keep the simulation epoch, but retry all outputs against the next Sync's
// accepted configuration, even when no further simulation step is accepted.
fn discard_stale_publications(world: &mut World, entity: Entity, epoch: u64) -> bool {
    let runtime_matches = runtime_authoring_matches(world, entity, epoch);
    let mut changed = false;
    {
        let runtimes = world.resource_mut::<MpmRuntimeResource>();
        if runtime_matches {
            if let Some(entry) = runtimes.entries.get_mut(&entity) {
                entry.latest = None;
                entry.surface_pending = true;
                entry.resident_pending = true;
            }
        } else {
            changed = runtimes.entries.remove(&entity).is_some();
        }
    }
    // Consumer-held native leases remain valid; only ECS's ownership is retired.
    changed |= world.remove::<MpmParticleSnapshot>(entity);
    changed |= world.remove::<MpmSurfaceSnapshot>(entity);
    changed |= world.remove::<MpmResidentSurfaceSnapshot>(entity);
    changed
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
    let mut particles = Vec::new();
    let mut surfaces = BTreeMap::new();
    let mut resident_surfaces = BTreeMap::new();
    let mut entities: Vec<_> = runtimes
        .entries
        .iter()
        .map(|(&entity, entry)| (entity, entry.epoch))
        .collect();
    entities.sort();
    for &(entity, _) in &entities {
        let entry = runtimes
            .entries
            .get_mut(&entity)
            .expect("sorted runtime entity must remain present");
        if std::mem::take(&mut entry.publish_pending) {
            let Some(version) = entry.source_version.checked_add(1) else {
                issue(&mut diagnostics, Some(entity), MpmIssue::SequenceExhausted);
                continue;
            };
            entry.source_version = version;
            entry.source_tick = tick;
            entry.surface_pending = true;
            entry.resident_pending = true;
        }
        let readback = entry.runtime.backend() == MpmBackend::Cpu
            || entry.surface_config.is_some()
            || entry
                .resident_config
                .is_none_or(|config| config.readback_particles);
        if readback
            && entry.source_version > 0
            && entry
                .latest
                .as_ref()
                .is_none_or(|s| s.version != entry.source_version)
        {
            match entry.runtime.snapshot() {
                Ok(data) => {
                    let snapshot = MpmParticleSnapshot::from_step(
                        entry.epoch,
                        entry.source_version,
                        entry.source_tick,
                        data,
                    );
                    if snapshot.valid() {
                        entry.latest = Some(snapshot.clone());
                        particles.push((entity, snapshot));
                        entry.surface_pending = true;
                    } else {
                        issue(&mut diagnostics, Some(entity), MpmIssue::InvalidSnapshot);
                    }
                }
                Err(error) => issue(&mut diagnostics, Some(entity), MpmIssue::Snapshot(error)),
            }
        }
        if entry.source_version > 0 && std::mem::take(&mut entry.resident_pending) {
            if let Some(config) = entry.resident_config {
                match entry.resident_revision.checked_add(1) {
                    None => {
                        resident_surfaces.insert(entity, None);
                        issue(&mut diagnostics, Some(entity), MpmIssue::SequenceExhausted);
                    }
                    Some(revision) => match entry.runtime.publish_surface(config.surface) {
                        Ok(surface) => {
                            entry.resident_revision = revision;
                            resident_surfaces.insert(
                                entity,
                                Some(MpmResidentSurfaceSnapshot {
                                    epoch: entry.epoch,
                                    revision,
                                    version: entry.source_version,
                                    tick: entry.source_tick,
                                    surface,
                                }),
                            );
                        }
                        Err(error) => {
                            resident_surfaces.insert(entity, None);
                            // Drop ECS's stale lease at commit. A held consumer
                            // lease may delay publication; retry even if the
                            // next simulation step is rejected by its clock.
                            entry.resident_pending = error == ResidentSurfaceError::InFlightLimit;
                            issue(
                                &mut diagnostics,
                                Some(entity),
                                MpmIssue::ResidentSurface(error),
                            );
                        }
                    },
                }
            } else {
                resident_surfaces.insert(entity, None);
            }
        }
        if std::mem::take(&mut entry.surface_pending) {
            if let Some(snapshot) = entry.latest.as_ref() {
                outputs.push((entity, snapshot.clone(), entry.surface_config));
            }
        }
    }
    for (entity, snapshot, config) in outputs {
        if world.get::<MpmBinding>(entity).is_some() {
            let Some(config) = config else {
                surfaces.insert(entity, None);
                continue;
            };
            if !config.valid() {
                surfaces.insert(entity, None);
                continue;
            }
            let field = match ScalarField::from_particles_bounded(
                &snapshot.positions,
                config.radius,
                config.dx,
                config.padding_cells,
                FieldBudget {
                    max_nodes: config.max_field_nodes,
                    max_cells: config.max_cells,
                    max_sample_visits: config.max_sample_visits,
                },
            ) {
                Ok(field) => field,
                Err(error) => {
                    surfaces.insert(entity, None);
                    issue(&mut diagnostics, Some(entity), surface_field_issue(error));
                    continue;
                }
            };
            let mesh = match triangulate_bounded(
                &field,
                config.iso,
                SurfaceBudget {
                    max_cells: config.max_cells,
                    max_vertices: config.max_surface_vertices,
                    max_triangles: config.max_surface_triangles,
                },
            ) {
                Ok(mesh) => mesh,
                Err(error) => {
                    surfaces.insert(entity, None);
                    issue(&mut diagnostics, Some(entity), surface_mesh_issue(error));
                    continue;
                }
            };
            let valid_mesh = mesh.indices_are_valid()
                && mesh.normals.len() == mesh.positions.len()
                && mesh.positions.iter().all(|v| v.is_finite())
                && mesh.normals.iter().all(|v| v.is_finite());
            if !valid_mesh {
                surfaces.insert(entity, None);
                issue(&mut diagnostics, Some(entity), MpmIssue::InvalidSurface);
                continue;
            }
            let entry = runtimes.entries.get_mut(&entity).unwrap();
            let Some(revision) = entry.surface_revision.checked_add(1) else {
                surfaces.insert(entity, None);
                issue(&mut diagnostics, Some(entity), MpmIssue::SequenceExhausted);
                continue;
            };
            entry.surface_revision = revision;
            surfaces.insert(
                entity,
                Some(MpmSurfaceSnapshot {
                    epoch: snapshot.epoch,
                    revision,
                    version: snapshot.version,
                    tick: snapshot.tick,
                    mesh: Arc::new(mesh),
                }),
            );
        }
    }
    world.insert_resource(runtimes);
    world.insert_resource(diagnostics);
    // Publication is a separate commit phase: insertion/removal can invoke
    // user observers, which must be able to inspect the committed resources.
    for (entity, snapshot) in particles {
        let epoch = snapshot.epoch;
        if publication_authoring_matches(world, entity, epoch) {
            world.insert(entity, (snapshot,));
        }
        if !publication_authoring_matches(world, entity, epoch) {
            discard_stale_publications(world, entity, epoch);
        }
    }
    for (entity, surface) in surfaces {
        if let Some(surface) = surface {
            let epoch = surface.epoch;
            if publication_authoring_matches(world, entity, epoch) {
                world.insert(entity, (surface,));
            }
            if !publication_authoring_matches(world, entity, epoch) {
                discard_stale_publications(world, entity, epoch);
            }
        } else {
            world.remove::<MpmSurfaceSnapshot>(entity);
        }
    }
    for (entity, surface) in resident_surfaces {
        if let Some(surface) = surface {
            // Earlier portable output insertions can invoke user observers.
            // Never commit a native lease for authoring they just removed or
            // replaced, even when removal/reinsert keeps the same value.
            let epoch = surface.epoch;
            if publication_authoring_matches(world, entity, epoch) {
                world.insert(entity, (surface,));
                // Its own insertion observers can also replace authoring.
                if publication_authoring_matches(world, entity, epoch) {
                    continue;
                }
            }
            discard_stale_publications(world, entity, epoch);
        }
        world.remove::<MpmResidentSurfaceSnapshot>(entity);
    }
    // A later entity's observer (including a removal observer) can edit one
    // already committed above. Sweep to quiescence: each additional pass must
    // retire an entry or an output, rather than merely setting retry flags.
    loop {
        let mut changed = false;
        for &(entity, epoch) in &entities {
            if !publication_authoring_matches(world, entity, epoch) {
                changed |= discard_stale_publications(world, entity, epoch);
            }
        }
        if !changed {
            break;
        }
    }
}

fn surface_field_issue(error: FieldBuildError) -> MpmIssue {
    match error {
        FieldBuildError::InvalidInput | FieldBuildError::GridOverflow => MpmIssue::InvalidSurface,
        FieldBuildError::NodeBudgetExceeded
        | FieldBuildError::CellBudgetExceeded
        | FieldBuildError::SampleBudgetExceeded => MpmIssue::SurfaceBudgetExceeded,
        FieldBuildError::AllocationFailed => MpmIssue::SurfaceAllocationFailed,
    }
}

fn surface_mesh_issue(error: SurfaceBuildError) -> MpmIssue {
    match error {
        SurfaceBuildError::InvalidField => MpmIssue::InvalidSurface,
        SurfaceBuildError::CellBudgetExceeded
        | SurfaceBuildError::VertexBudgetExceeded
        | SurfaceBuildError::TriangleBudgetExceeded => MpmIssue::SurfaceBudgetExceeded,
        SurfaceBuildError::AllocationFailed => MpmIssue::SurfaceAllocationFailed,
    }
}
