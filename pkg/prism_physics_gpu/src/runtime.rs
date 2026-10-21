//! Backend-selectable fixed-step runtime for the core MLS-MPM state.
//!
//! This module is independent of any ECS. It owns the fixed-step boundary,
//! backend selection, bounded catch-up work, and explicit snapshot points.
//! GPU mode keeps particle and grid buffers resident through
//! GpuMpmResident; CPU mode uses prism_physics_core as the numerical
//! reference.

use prism_physics_core::mpm::{BoundaryCondition, MaterialPoints, MpmSolver};

use crate::context::GpuContext;
use crate::mpm::gpu::surface::{ResidentSurface, ResidentSurfaceConfig, ResidentSurfaceError};
use crate::mpm::{BoundaryMode, GpuMpmResident, StepConfig, StepInputs, StepParticles};

/// Selects how the runtime acquires its execution backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MpmBackendPolicy {
    /// Always use the CPU reference implementation.
    Cpu,
    /// Prefer GPU and explicitly report a CPU fallback when unavailable.
    PreferGpu,
    /// Require a usable GPU and fail construction otherwise.
    RequireGpu,
}

/// Backend executing fixed steps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MpmBackend {
    /// The core CPU MPM solver.
    Cpu,
    /// The resident wgpu MPM pipeline.
    Gpu,
}

/// Reason a preferred GPU runtime fell back to CPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MpmFallbackReason {
    /// The adopted device cannot hold/dispatch the authored resident state.
    DeviceLimits,
    /// No adapter or device could be acquired.
    GpuUnavailable,
    /// The state is empty and is handled directly by CPU.
    EmptyState,
}

/// Runtime construction and synchronization errors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MpmRuntimeError {
    /// The adopted device cannot hold/dispatch the authored resident state.
    DeviceLimits,
    /// A required GPU could not be acquired.
    GpuUnavailable,
    /// A required GPU cannot be initialized for an empty state.
    EmptyGpuState,
    /// The fixed-step catch-up budget is zero.
    InvalidStepBudget,
    /// A backend snapshot did not match the core particle columns.
    StateLengthMismatch,
}

/// Runtime options.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MpmRuntimeConfig {
    /// Backend acquisition policy.
    pub backend: MpmBackendPolicy,
    /// Maximum fixed steps accepted by one advance call.
    pub max_steps_per_advance: usize,
}

impl Default for MpmRuntimeConfig {
    fn default() -> Self {
        Self {
            backend: MpmBackendPolicy::PreferGpu,
            max_steps_per_advance: 8,
        }
    }
}

/// Result of one bounded fixed-step advance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MpmAdvanceReport {
    /// Number of fixed steps executed.
    pub steps: usize,
    /// Backend that executed the steps.
    pub backend: MpmBackend,
    /// Whether the request exceeded the configured budget.
    pub capped: bool,
}

/// A backend-selectable, fixed-step MLS-MPM runtime.
pub struct MpmRuntime {
    cpu: MpmSolver,
    backend: MpmBackend,
    fallback: Option<MpmFallbackReason>,
    context: Option<GpuContext>,
    resident: Option<GpuMpmResident>,
    gpu_config: StepConfig,
    max_steps_per_advance: usize,
    fixed_tick: u64,
}

impl MpmRuntime {
    /// Creates a runtime from a core solver and backend policy.
    pub fn new(solver: MpmSolver, config: MpmRuntimeConfig) -> Result<Self, MpmRuntimeError> {
        Self::create(solver, config, None)
    }

    /// Uses the host's exact logical device and submission queue, rather than
    /// requesting an independent headless device. Clone one adopted context
    /// for physics and rendering so native publications can be imported.
    pub fn with_gpu_context(
        solver: MpmSolver,
        config: MpmRuntimeConfig,
        context: GpuContext,
    ) -> Result<Self, MpmRuntimeError> {
        Self::create(solver, config, Some(context))
    }

    fn create(
        solver: MpmSolver,
        config: MpmRuntimeConfig,
        context: Option<GpuContext>,
    ) -> Result<Self, MpmRuntimeError> {
        if config.max_steps_per_advance == 0 {
            return Err(MpmRuntimeError::InvalidStepBudget);
        }
        let gpu_config = gpu_config(&solver);
        let mut runtime = Self {
            cpu: solver,
            backend: MpmBackend::Cpu,
            fallback: None,
            context: None,
            resident: None,
            gpu_config,
            max_steps_per_advance: config.max_steps_per_advance,
            fixed_tick: 0,
        };

        match config.backend {
            MpmBackendPolicy::Cpu => {}
            MpmBackendPolicy::PreferGpu | MpmBackendPolicy::RequireGpu => {
                if runtime.cpu.particles.is_empty() {
                    if config.backend == MpmBackendPolicy::RequireGpu {
                        return Err(MpmRuntimeError::EmptyGpuState);
                    }
                    runtime.fallback = Some(MpmFallbackReason::EmptyState);
                } else if let Some(context) = context.or_else(GpuContext::try_headless) {
                    if !gpu_fits(&context, &runtime.cpu) {
                        if config.backend == MpmBackendPolicy::RequireGpu {
                            return Err(MpmRuntimeError::DeviceLimits);
                        }
                        runtime.fallback = Some(MpmFallbackReason::DeviceLimits);
                        return Ok(runtime);
                    }
                    let mut resident = GpuMpmResident::new(&context);
                    resident.upload(
                        &context,
                        &inputs(&runtime.cpu.particles),
                        &runtime.gpu_config,
                    );
                    runtime.context = Some(context);
                    runtime.resident = Some(resident);
                    runtime.backend = MpmBackend::Gpu;
                } else if config.backend == MpmBackendPolicy::RequireGpu {
                    return Err(MpmRuntimeError::GpuUnavailable);
                } else {
                    runtime.fallback = Some(MpmFallbackReason::GpuUnavailable);
                }
            }
        }
        Ok(runtime)
    }

    /// Generates a resident triangle-list lease from the live GPU particles.
    /// This never calls snapshot, maps buffers, or waits for the device.
    pub fn publish_surface(
        &mut self,
        config: ResidentSurfaceConfig,
    ) -> Result<ResidentSurface, ResidentSurfaceError> {
        let context = self
            .context
            .as_ref()
            .ok_or(ResidentSurfaceError::BackendUnavailable)?;
        self.resident
            .as_mut()
            .ok_or(ResidentSurfaceError::BackendUnavailable)?
            .publish_surface(context, config)
    }

    /// Returns the backend executing fixed steps.
    #[must_use]
    pub const fn backend(&self) -> MpmBackend {
        self.backend
    }

    /// Returns the reason for a preferred-GPU CPU fallback, if any.
    #[must_use]
    pub const fn fallback_reason(&self) -> Option<MpmFallbackReason> {
        self.fallback
    }

    /// Returns the number of fixed steps executed since construction.
    #[must_use]
    pub const fn fixed_tick(&self) -> u64 {
        self.fixed_tick
    }

    /// Returns the configured fixed step duration.
    #[must_use]
    pub const fn fixed_dt(&self) -> f32 {
        self.cpu.config.dt
    }

    /// Returns the CPU state, synchronized at the latest snapshot boundary.
    #[must_use]
    pub fn cpu_solver(&self) -> &MpmSolver {
        &self.cpu
    }

    /// Advances at most the configured number of fixed steps.
    pub fn advance(&mut self, requested_steps: usize) -> MpmAdvanceReport {
        let steps = requested_steps.min(self.max_steps_per_advance);
        if steps > 0 {
            match self.backend {
                MpmBackend::Cpu => self.cpu.advance(steps),
                MpmBackend::Gpu => {
                    let context = self.context.as_ref().expect("GPU backend context");
                    let resident = self.resident.as_ref().expect("GPU backend resident");
                    resident.step(context, steps);
                }
            }
            self.fixed_tick = self.fixed_tick.saturating_add(steps as u64);
        }
        MpmAdvanceReport {
            steps,
            backend: self.backend,
            capped: requested_steps > steps,
        }
    }

    /// Synchronizes the current backend state into a portable particle state.
    pub fn snapshot(&mut self) -> Result<StepParticles, MpmRuntimeError> {
        let snapshot = match self.backend {
            MpmBackend::Cpu => cpu_snapshot(&self.cpu.particles),
            MpmBackend::Gpu => {
                let context = self.context.as_ref().expect("GPU backend context");
                let resident = self.resident.as_ref().expect("GPU backend resident");
                resident.snapshot(context)
            }
        };
        let count = self.cpu.particles.len();
        if snapshot.positions.len() != count
            || snapshot.velocities.len() != count
            || snapshot.affine.len() != count
            || snapshot.deformation.len() != count
            || snapshot.plastic_det.len() != count
        {
            return Err(MpmRuntimeError::StateLengthMismatch);
        }
        if self.backend == MpmBackend::Gpu {
            apply_snapshot(&mut self.cpu.particles, &snapshot);
        }
        Ok(snapshot)
    }

    /// Synchronizes GPU state and switches subsequent steps to CPU.
    pub fn switch_to_cpu(&mut self) -> Result<(), MpmRuntimeError> {
        if self.backend == MpmBackend::Gpu {
            self.snapshot()?;
            self.context = None;
            self.resident = None;
            self.backend = MpmBackend::Cpu;
        }
        Ok(())
    }
}

fn gpu_fits(context: &GpuContext, solver: &MpmSolver) -> bool {
    let limits = context.device().limits();
    let max = limits
        .max_buffer_size
        .min(u64::from(limits.max_storage_buffer_binding_size));
    let Some(nodes) = solver
        .grid
        .nx()
        .checked_mul(solver.grid.ny())
        .and_then(|n| n.checked_mul(solver.grid.nz()))
    else {
        return false;
    };
    let particles = solver.particles.len();
    limits.max_storage_buffers_per_shader_stage >= 12
        && limits.max_bindings_per_bind_group >= 13
        && limits.max_compute_invocations_per_workgroup >= 64
        && limits.max_compute_workgroup_size_x >= 64
        && nodes <= u32::MAX as usize
        && particles <= u32::MAX as usize
        && (nodes as u64)
            .checked_mul(16)
            .is_some_and(|bytes| bytes <= max)
        && (particles as u64)
            .checked_mul(48)
            .is_some_and(|bytes| bytes <= max)
        && nodes.div_ceil(64) <= limits.max_compute_workgroups_per_dimension as usize
        && particles.div_ceil(64) <= limits.max_compute_workgroups_per_dimension as usize
}

fn inputs(particles: &MaterialPoints) -> StepInputs<'_> {
    StepInputs {
        positions: particles.positions(),
        velocities: particles.velocities(),
        affine: particles.affine(),
        deformation: particles.deformation(),
        masses: particles.masses(),
        volumes: particles.volumes(),
        plastic_det: particles.plastic_det(),
    }
}

fn cpu_snapshot(particles: &MaterialPoints) -> StepParticles {
    StepParticles {
        positions: particles.positions().to_vec(),
        velocities: particles.velocities().to_vec(),
        affine: particles.affine().to_vec(),
        deformation: particles.deformation().to_vec(),
        plastic_det: particles.plastic_det().to_vec(),
    }
}

fn apply_snapshot(particles: &mut MaterialPoints, snapshot: &StepParticles) {
    particles
        .positions_mut()
        .copy_from_slice(&snapshot.positions);
    particles
        .velocities_mut()
        .copy_from_slice(&snapshot.velocities);
    particles.affine_mut().copy_from_slice(&snapshot.affine);
    particles
        .deformation_mut()
        .copy_from_slice(&snapshot.deformation);
    particles
        .plastic_det_mut()
        .copy_from_slice(&snapshot.plastic_det);
}

fn gpu_config(solver: &MpmSolver) -> StepConfig {
    let (lambda0, mu0) = solver.material.lame();
    let plasticity = solver.config.plasticity;
    StepConfig {
        dims: (solver.grid.nx(), solver.grid.ny(), solver.grid.nz()),
        origin: solver.grid.origin(),
        dx: solver.grid.dx(),
        dt: solver.config.dt,
        gravity: solver.config.gravity,
        mu0,
        lambda0,
        hardening: plasticity.hardening,
        theta_c: plasticity.critical_compression,
        theta_s: plasticity.critical_stretch,
        boundary: match solver.config.boundary {
            BoundaryCondition::Sticky => BoundaryMode::Sticky,
            BoundaryCondition::Slip => BoundaryMode::Slip,
            BoundaryCondition::Separate => BoundaryMode::Separate,
        },
        boundary_thickness: solver.config.boundary_thickness,
        plastic: solver.config.plastic,
        steps: 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;
    use prism_physics_core::mpm::{Grid, MpmConfig, MpmMaterial};

    fn solver() -> MpmSolver {
        let mut particles = MaterialPoints::new();
        particles.spawn(Vec3::new(0.9, 0.9, 0.9), Vec3::ZERO, 1.0, 1.0e-3);
        MpmSolver::new(
            particles,
            Grid::new(16, 16, 16, 0.1, Vec3::ZERO),
            MpmMaterial::default(),
            MpmConfig::new(1.0e-3, Vec3::new(0.0, -9.81, 0.0)),
        )
    }

    #[test]
    fn cpu_runtime_has_fixed_step_and_snapshot_contract() {
        let mut runtime = MpmRuntime::new(
            solver(),
            MpmRuntimeConfig {
                backend: MpmBackendPolicy::Cpu,
                max_steps_per_advance: 2,
            },
        )
        .expect("valid CPU runtime");
        let report = runtime.advance(4);
        assert_eq!(report.steps, 2);
        assert!(report.capped);
        assert_eq!(runtime.fixed_tick(), 2);
        assert_eq!(runtime.snapshot().expect("CPU snapshot").positions.len(), 1);
    }

    #[test]
    fn preferred_gpu_empty_state_has_explicit_cpu_fallback() {
        let solver = MpmSolver::new(
            MaterialPoints::new(),
            Grid::new(8, 8, 8, 0.1, Vec3::ZERO),
            MpmMaterial::default(),
            MpmConfig::default(),
        );
        let runtime = MpmRuntime::new(
            solver,
            MpmRuntimeConfig {
                backend: MpmBackendPolicy::PreferGpu,
                max_steps_per_advance: 1,
            },
        )
        .expect("empty preferred-GPU runtime falls back");
        assert_eq!(runtime.backend(), MpmBackend::Cpu);
        assert_eq!(
            runtime.fallback_reason(),
            Some(MpmFallbackReason::EmptyState)
        );
    }
}
