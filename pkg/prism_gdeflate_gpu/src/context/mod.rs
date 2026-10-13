//! `GPU` device acquisition and submission helpers.
//!
//! [`GpuContext`] wraps the four long-lived `wgpu` handles (`Instance`,
//! `Adapter`, `Device`, `Queue`) that every kernel dispatch needs. Acquisition
//! is deliberately *best-effort*: [`GpuContext::try_headless`] returns [`None`]
//! on a host with no usable adapter (some `CI` images) so a test suite can skip
//! gracefully rather than fail, while still exercising the full dispatch on any
//! machine with a real device such as an Apple `M`-series `GPU`.
//!
//! This is a deliberate local copy of the identical helper in
//! `prism_physics_gpu::context`: each `*_gpu` twin owns its own tiny device
//! bootstrap so this `GDeflate` de-interleave twin never couples to another
//! crate's private module tree.
//!
//! Provenance: standard `wgpu` initialisation; no Unreal Engine source or
//! derived code.

use alloc::string::String;
use core::fmt;

use wgpu::{
    Adapter, BackendOptions, Backends, Device, DeviceDescriptor, Instance, InstanceDescriptor,
    InstanceFlags, PollType, Queue, RequestAdapterOptions,
};

/// Blocks the current thread until `future` resolves.
///
/// A thin re-export of [`futures_lite::future::block_on`] so callers do not
/// need a direct dependency on an async executor just to drive the handful of
/// `wgpu` setup futures.
pub fn block_on<F: Future>(future: F) -> F::Output {
    futures_lite::future::block_on(future)
}

/// Why a real compute device could not be acquired.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GpuProbeError {
    /// No adapter was exposed by the enabled native backends.
    NoAdapter(String),
    /// An adapter was found, but its logical device could not be created.
    Device(String),
}

impl fmt::Display for GpuProbeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoAdapter(message) => write!(formatter, "no usable GPU adapter: {message}"),
            Self::Device(message) => write!(formatter, "GPU device creation failed: {message}"),
        }
    }
}

impl core::error::Error for GpuProbeError {}

/// The long-lived `wgpu` handles shared by every compute kernel in this crate.
pub struct GpuContext {
    /// The `wgpu` instance the adapter was requested from.
    #[expect(dead_code, reason = "kept alive so the adapter and device stay valid")]
    instance: Instance,
    /// The physical adapter backing [`GpuContext::device`].
    #[expect(dead_code, reason = "kept alive so the device stays valid")]
    adapter: Adapter,
    /// Snapshot of the physical adapter identity used for diagnostics.
    adapter_info: wgpu::AdapterInfo,
    /// The logical device used to allocate resources and pipelines.
    device: Device,
    /// The queue used to submit command buffers.
    queue: Queue,
}

impl GpuContext {
    /// Attempts to acquire a headless compute device across the native backends.
    ///
    /// Returns [`None`] when no adapter or device can be obtained so callers can
    /// skip `GPU` work instead of panicking on a host without a usable adapter.
    #[must_use]
    pub fn try_headless() -> Option<GpuContext> {
        Self::probe().ok()
    }

    /// Acquires a headless compute device and preserves the concrete failure
    /// reason when the host cannot provide one.
    pub fn probe() -> Result<GpuContext, GpuProbeError> {
        let instance = Instance::new(InstanceDescriptor {
            // The crate enables Metal, Vulkan, and DX12 in its manifest. Keep
            // this list in one place so diagnostics describe the actual probe.
            backends: Backends::METAL | Backends::VULKAN | Backends::DX12,
            flags: InstanceFlags::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            display: None,
            backend_options: BackendOptions::default(),
        });
        let adapter = block_on(instance.request_adapter(&RequestAdapterOptions::default()))
            .map_err(|error| GpuProbeError::NoAdapter(error.to_string()))?;
        let adapter_info = adapter.get_info();
        let (device, queue) = block_on(adapter.request_device(&DeviceDescriptor {
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .map_err(|error| GpuProbeError::Device(error.to_string()))?;
        Ok(GpuContext {
            instance,
            adapter,
            adapter_info,
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

    /// Returns the adapter identity and backend selected by `wgpu`.
    #[must_use]
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.adapter_info
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
