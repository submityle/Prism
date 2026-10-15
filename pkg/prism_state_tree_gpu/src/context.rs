//! Best-effort native `wgpu` device acquisition for the `StateTree` twin.

use wgpu::{
    Adapter, BackendOptions, Backends, Device, DeviceDescriptor, Instance, InstanceDescriptor,
    InstanceFlags, PollType, Queue, RequestAdapterOptions,
};

/// Blocks the current thread until a `wgpu` future resolves.
pub fn block_on<F: Future>(future: F) -> F::Output {
    futures_lite::future::block_on(future)
}

/// Long-lived `wgpu` handles used by `StateTree` compute submissions.
pub struct GpuContext {
    #[expect(dead_code, reason = "kept alive for adapter and device lifetime")]
    instance: Instance,
    adapter: Adapter,
    device: Device,
    queue: Queue,
}

impl GpuContext {
    /// Attempts to acquire a headless native compute device.
    ///
    /// Returns `None` when the host has no usable Metal, Vulkan, or D3D12
    /// adapter, or when logical-device creation is rejected by the driver.
    #[must_use]
    pub fn try_headless() -> Option<Self> {
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::METAL | Backends::VULKAN | Backends::DX12,
            flags: InstanceFlags::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            display: None,
            backend_options: BackendOptions::default(),
        });
        let adapter = block_on(instance.request_adapter(&RequestAdapterOptions::default())).ok()?;
        let (device, queue) = block_on(adapter.request_device(&DeviceDescriptor {
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .ok()?;
        Some(Self {
            instance,
            adapter,
            device,
            queue,
        })
    }

    /// Returns the logical device used for resource creation.
    #[must_use]
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Returns the submission queue.
    #[must_use]
    pub fn queue(&self) -> &Queue {
        &self.queue
    }

    /// Returns immutable information about the selected physical adapter.
    #[must_use]
    pub fn adapter_info(&self) -> wgpu::AdapterInfo {
        self.adapter.get_info()
    }

    /// Waits until all previously submitted work has completed.
    ///
    /// # Panics
    ///
    /// Panics if the driver reports a device-loss error while polling.
    pub fn wait(&self) {
        self.device
            .poll(PollType::wait_indefinitely())
            .expect("StateTree GPU device poll should complete submitted work");
    }
}
