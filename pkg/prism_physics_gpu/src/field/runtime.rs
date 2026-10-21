//! Explicit GPU field authority for core world and XPBD fixed-substep APIs.

use prism_physics_core::dynamics::field::FieldBatchSampler;

use super::{FieldBody, FieldGpuError, FieldRegistry, FieldSample, GpuFieldSampler};
use crate::context::GpuContext;

/// Owns a reusable compute pipeline and its device outside the physics world.
///
/// Pass this to `XpbdSolver::step_with_field_sampler` to evaluate analytic
/// fields on the GPU before every CPU XPBD substep. Results are read back and
/// applied to the live body's force accumulator, not a detached CPU twin.
/// This hybrid path has a synchronization/readback cost; it does not claim
/// resident zero-copy simulation or superiority for small body batches.
pub struct GpuFieldRuntime {
    context: GpuContext,
    sampler: GpuFieldSampler,
}

impl GpuFieldRuntime {
    /// Uses an explicitly acquired device. No device selection or fallback is
    /// hidden inside a simulation step.
    pub fn new(context: GpuContext) -> Result<Self, FieldGpuError> {
        let sampler = GpuFieldSampler::new(&context)?;
        Ok(Self { context, sampler })
    }

    /// Actual adapter/device authority used by this runtime.
    #[must_use]
    pub fn context(&self) -> &GpuContext {
        &self.context
    }
}

impl FieldBatchSampler for GpuFieldRuntime {
    type Error = FieldGpuError;

    fn sample_batch(
        &mut self,
        registry: &FieldRegistry,
        bodies: &[FieldBody],
    ) -> Result<Vec<FieldSample>, Self::Error> {
        self.sampler.sample(&self.context, registry, bodies)
    }
}
