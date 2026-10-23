//! `GPU` device acquisition and submission helpers.
//!
//! [`GpuContext`] wraps the four long-lived `wgpu` handles (`Instance`,
//! `Adapter`, `Device`, `Queue`) that every kernel dispatch needs. Acquisition
//! is deliberately *best-effort*: [`GpuContext::try_headless`] returns [`None`]
//! on a host with no usable adapter (some `CI` images) so a test suite can skip
//! gracefully rather than fail, while still exercising the full dispatch on any
//! machine with a real device such as an Apple `M`-series `GPU`.
//!
//! This is a deliberate local copy of the identical helper in the sibling
//! `*_gpu` twins: each twin owns its own tiny device bootstrap so this
//! hierarchy twin never couples to another crate's private module tree.
//!
//! Provenance: standard `wgpu` initialisation; no Unreal Engine or Unity source
//! or derived code.

use core::sync::atomic::{AtomicU64, Ordering};
use wgpu::{
    Adapter, BackendOptions, Backends, Device, DeviceDescriptor, Instance, InstanceDescriptor,
    InstanceFlags, PollType, Queue, RequestAdapterOptions,
};

static NEXT_DEVICE_GENERATION: AtomicU64 = AtomicU64::new(1);

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
    device_generation: u64,
}

impl GpuContext {
    /// Attempts to acquire a headless compute device across the native backends.
    ///
    /// Returns [`None`] when no adapter or device can be obtained so callers can
    /// skip `GPU` work instead of panicking on a host without a usable adapter.
    #[must_use]
    pub fn try_headless() -> Option<GpuContext> {
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
        let device_generation = NEXT_DEVICE_GENERATION
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(1)
            })
            .ok()?;
        Some(GpuContext {
            instance,
            adapter,
            device,
            queue,
            device_generation,
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

    /// Stable identity for this logical device lifetime. GPU-owned caches use
    /// it to reject handles from another context and rebuild after recreation.
    #[must_use]
    pub(crate) fn device_generation(&self) -> u64 {
        self.device_generation
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
