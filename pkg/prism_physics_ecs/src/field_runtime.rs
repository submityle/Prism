//! Explicit analytic-field compute authority for fixed-step XPBD.

use prism_ecs::prelude::Resource;

pub use prism_physics_core::solver::xpbd::FieldStepError;
pub use prism_physics_gpu::field::{FieldGpuError, GpuFieldRuntime};

/// Optional caller-owned GPU field runtime. Insert before a fixed tick to use
/// GPU analytic-field sampling on every XPBD substep; remove to use the core
/// CPU default again. Existing bodies, field bindings and solver state survive.
///
/// The caller must acquire/configure the runtime's device explicitly. The
/// physics plugin never initializes this resource, acquires a device, or falls
/// back to CPU sampling after a GPU error. Only field evaluation changes
/// authority: XPBD and the remaining dynamics still run on the CPU.
/// A rejected substep restores pending forces and ECS-wrench sleep/wake side
/// effects, including idle timers; earlier completed substeps remain committed.
#[derive(Resource)]
pub struct FieldRuntimeResource(pub GpuFieldRuntime);

/// Latest fixed tick's field-authority status, including partial completion.
/// Cleared on every attempted tick, including CPU and invalid-configuration
/// ticks. Accepted time is published in [`crate::PhysicsDiagnostics::simulated_dt`].
#[derive(Resource, Debug, Default)]
pub struct FieldRuntimeDiagnostics {
    /// Same attempted fixed tick as [`crate::PhysicsDiagnostics::tick`].
    pub tick: u64,
    /// Whether an explicit GPU runtime was present for this tick. A failed GPU
    /// tick does not change authority or silently complete on the CPU.
    pub gpu_authority: bool,
    /// Configured number of substeps requested for this tick.
    pub requested_substeps: u32,
    /// Successfully committed substeps, excluding the rejected substep.
    pub completed_substeps: u32,
    /// Full core rejection, with `completed_substeps` relative to the entire
    /// fixed tick rather than to the bridge's one-substep solver invocation.
    pub error: Option<FieldStepError<FieldGpuError>>,
}
