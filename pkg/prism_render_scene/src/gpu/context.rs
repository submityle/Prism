//! The Prism RHI GPU context owned by the render world.
//!
//! During the `bevy_render` → Prism RHI port this is the single place the
//! renderer obtains a device and queue. Subsystems migrate onto it one at a
//! time; until the port finishes the legacy `bevy_render` device still exists
//! alongside it, so resources may only be shared between subsystems that have
//! both migrated.
//!
//! The context is backend-agnostic in shape (it exposes only [`prism_render_driver`]
//! traits) even though the shipped backend is wgpu: the concrete handle pair is
//! created through the RHI's `Instance` facade, so a different backend slots in
//! by changing the constructor alone.

use prism_render_driver_wgpu::{WgpuDevice, WgpuInstance, WgpuQueue};

/// Why the Prism GPU context could not be created.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum PrismGpuInitError {
    /// No usable graphics adapter was found on this host.
    NoAdapter,
}

/// The Prism RHI device and queue the render world draws with.
#[derive(bevy_ecs::resource::Resource)]
pub struct PrismGpu {
    /// The RHI device; the factory for every GPU resource.
    device: WgpuDevice,
    /// The RHI queue; uploads and submission.
    queue: WgpuQueue,
}

impl PrismGpu {
    /// Brings up a Prism RHI device/queue pair on the default adapter.
    ///
    /// # Errors
    ///
    /// [`PrismGpuInitError::NoAdapter`] when the host has no usable adapter, so
    /// a headless/GPU-less run can degrade instead of aborting.
    pub fn new() -> Result<Self, PrismGpuInitError> {
        let instance = WgpuInstance::new();
        let (device, queue) = instance
            .request_default_device()
            .map_err(|_| PrismGpuInitError::NoAdapter)?;
        Ok(Self { device, queue })
    }

    /// The RHI device.
    #[must_use]
    pub fn device(&self) -> &WgpuDevice {
        &self.device
    }

    /// The RHI queue.
    #[must_use]
    pub fn queue(&self) -> &WgpuQueue {
        &self.queue
    }
}

impl core::fmt::Debug for PrismGpu {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PrismGpu").finish_non_exhaustive()
    }
}
