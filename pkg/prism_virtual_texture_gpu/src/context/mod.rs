//! `GPU` device acquisition and submission helpers.
//!
//! [`GpuContext`] wraps the four long-lived `wgpu` handles (`Instance`,
//! `Adapter`, `Device`, `Queue`) that every kernel dispatch needs. Acquisition
//! is best-effort by default: [`GpuContext::try_headless`] returns [`None`] on
//! a host with no usable adapter so a test suite can skip gracefully. Setting
//! `PRISM_REQUIRE_GPU` (to any value) makes adapter or device acquisition
//! failures panic, allowing CI to require real `GPU` execution explicitly.
//!
//! This is a deliberate local copy of the identical helper in
//! `prism_physics_gpu::context`: each `*_gpu` twin owns its own tiny device
//! bootstrap so the render reorder feature never couples to the physics crate's
//! private module tree.
//!
//! Provenance: standard `wgpu` initialisation; no Unreal Engine source or
//! derived code.

use wgpu::{
    Adapter, BackendOptions, Backends, Device, DeviceDescriptor, Instance, InstanceDescriptor,
    InstanceFlags, PollType, Queue, RequestAdapterOptions,
};

const REQUIRE_GPU_ENV: &str = "PRISM_REQUIRE_GPU";

fn gpu_resource_or_skip<T, E: std::fmt::Display>(
    result: Result<T, E>,
    stage: &'static str,
    required: bool,
) -> Option<T> {
    match result {
        Ok(resource) => Some(resource),
        Err(error) if required => panic!(
            "[prism_virtual_texture_gpu] {REQUIRE_GPU_ENV} is set, but {stage} acquisition failed: {error}"
        ),
        Err(_) => None,
    }
}

/// Blocks the current thread until `future` resolves.
///
/// A thin re-export of [`futures_lite::future::block_on`] so callers do not
/// need a direct dependency on an async executor just to drive the handful of
/// `wgpu` setup futures.
pub fn block_on<F: Future>(future: F) -> F::Output {
    futures_lite::future::block_on(future)
}

/// The long-lived `wgpu` handles shared by every compute kernel in this crate.
pub struct GpuContext {
    /// The `wgpu` instance the adapter was requested from.
    #[expect(dead_code, reason = "kept alive so the adapter and device stay valid")]
    instance: Instance,
    /// The physical adapter backing [`GpuContext::device`].
    #[expect(dead_code, reason = "kept alive so the device stays valid")]
    adapter: Adapter,
    /// The logical device used to allocate resources and pipelines.
    device: Device,
    /// The queue used to submit command buffers.
    queue: Queue,
}

impl GpuContext {
    /// Attempts to acquire a headless compute device across the native backends.
    ///
    /// Returns [`None`] when no adapter or device can be obtained, unless
    /// `PRISM_REQUIRE_GPU` is set (to any value), in which case acquisition
    /// failure panics with a diagnostic identifying the failed stage.
    #[must_use]
    pub fn try_headless() -> Option<GpuContext> {
        let required = std::env::var_os(REQUIRE_GPU_ENV).is_some();
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::METAL | Backends::VULKAN | Backends::DX12,
            flags: InstanceFlags::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            display: None,
            backend_options: BackendOptions::default(),
        });
        let adapter = gpu_resource_or_skip(
            block_on(instance.request_adapter(&RequestAdapterOptions::default())),
            "adapter",
            required,
        )?;
        let (device, queue) = gpu_resource_or_skip(
            block_on(adapter.request_device(&DeviceDescriptor {
                required_limits: adapter.limits(),
                ..Default::default()
            })),
            "device",
            required,
        )?;
        Some(GpuContext {
            instance,
            adapter,
            device,
            queue,
        })
    }

    /// Returns the logical device.
    #[must_use]
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Returns the submission queue.
    #[must_use]
    pub fn queue(&self) -> &Queue {
        &self.queue
    }

    /// Blocks until all previously submitted work on this device completes.
    ///
    /// # Panics
    ///
    /// Panics if the device is lost while waiting, which indicates a broken
    /// adapter and cannot be recovered from within a single dispatch.
    pub fn wait(&self) {
        self.device
            .poll(PollType::wait_indefinitely())
            .expect("device poll should complete the submitted work");
    }
}

#[cfg(test)]
mod tests {
    use super::gpu_resource_or_skip;

    #[test]
    fn unavailable_gpu_resource_skips_in_best_effort_mode() {
        assert_eq!(
            gpu_resource_or_skip::<(), _>(Err("adapter unavailable"), "adapter", false),
            None
        );
    }

    #[test]
    #[should_panic(expected = "PRISM_REQUIRE_GPU is set, but device acquisition failed")]
    fn unavailable_gpu_resource_fails_in_required_mode() {
        let _ = gpu_resource_or_skip::<(), _>(Err("device unavailable"), "device", true);
    }
}
