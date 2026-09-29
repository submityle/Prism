//! `GPU` device acquisition and submission helpers.
//!
//! [`GpuContext`] wraps the four long-lived `wgpu` handles (`Instance`,
//! `Adapter`, `Device`, `Queue`) every dispatch needs. Acquisition is
//! deliberately *best-effort*: [`GpuContext::try_headless`] returns [`None`] on
//! a host with no usable adapter so a test suite can skip gracefully rather
//! than fail, while still exercising the full dispatch on any machine with a
//! real device such as an Apple `M`-series `GPU`.
//!
//! # Optional 64-bit atomics
//!
//! The portable depth-only twin only needs `atomic<u32>`, but the full
//! `(depth << 32) | payload` vis-buffer key needs a 64-bit atomic maximum.
//! [`GpuContext::try_headless`] opportunistically enables
//! [`Features::SHADER_INT64`] plus [`Features::SHADER_INT64_ATOMIC_MIN_MAX`]
//! when the adapter advertises them, and records the result in
//! [`GpuContext::supports_u64_atomics`]. Callers that need the payload key
//! probe that flag and skip when it is `false` rather than compiling a kernel
//! the backend cannot run.
//!
//! Provenance: standard `wgpu` initialisation; no Unreal Engine source or
//! derived code.

use wgpu::{
    Adapter, BackendOptions, Backends, Device, DeviceDescriptor, Features, Instance,
    InstanceDescriptor, InstanceFlags, PollType, Queue, RequestAdapterOptions,
};

/// Blocks the current thread until `future` resolves.
///
/// A thin re-export of [`futures_lite::future::block_on`] so callers do not
/// need a direct dependency on an async executor just to drive the handful of
/// `wgpu` setup futures.
pub fn block_on<F: Future>(future: F) -> F::Output {
    futures_lite::future::block_on(future)
}

/// The `wgpu` features required to composite the 64-bit vis-buffer key: a
/// 64-bit integer type plus a 64-bit atomic maximum on storage buffers.
fn u64_atomic_features() -> Features {
    Features::SHADER_INT64 | Features::SHADER_INT64_ATOMIC_MIN_MAX
}

/// The long-lived `wgpu` handles shared by the rasterizer dispatch.
pub struct GpuContext {
    /// The `wgpu` instance the adapter was requested from.
    #[expect(dead_code, reason = "kept alive so the adapter it produced stays valid")]
    instance: Instance,
    /// The physical adapter backing [`GpuContext::device`].
    #[expect(dead_code, reason = "kept alive so the device it produced stays valid")]
    adapter: Adapter,
    /// The logical device used to allocate resources and pipelines.
    device: Device,
    /// The queue used to submit command buffers.
    queue: Queue,
    /// Whether the device was created with the 64-bit atomic-maximum features
    /// the payload vis-buffer key requires.
    supports_u64_atomics: bool,
}

impl GpuContext {
    /// Attempts to acquire a headless compute device across the native backends.
    ///
    /// Returns [`None`] when no adapter or device can be obtained so callers can
    /// skip `GPU` work instead of panicking on a host without a usable adapter.
    /// When the adapter advertises the 64-bit atomic features they are enabled
    /// so the payload vis-buffer key becomes available; otherwise only the
    /// portable depth-only path works and [`GpuContext::supports_u64_atomics`]
    /// returns `false`.
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
        // Enable the 64-bit atomic features only when the adapter has them, so
        // the device request never fails on a backend without the extension.
        let wanted = u64_atomic_features();
        let supports_u64_atomics = adapter.features().contains(wanted);
        let required_features = if supports_u64_atomics {
            wanted
        } else {
            Features::empty()
        };
        let (device, queue) = block_on(adapter.request_device(&DeviceDescriptor {
            required_features,
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .ok()?;
        Some(GpuContext {
            instance,
            adapter,
            device,
            queue,
            supports_u64_atomics,
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

    /// Whether this device can run the 64-bit payload vis-buffer kernel.
    ///
    /// `true` only when the adapter advertised, and the device enabled, both
    /// [`Features::SHADER_INT64`] and [`Features::SHADER_INT64_ATOMIC_MIN_MAX`].
    /// The portable depth-only [`crate::GpuSoftwareRaster`] does not depend on
    /// this; only [`crate::GpuPayloadRaster`] does.
    #[must_use]
    pub fn supports_u64_atomics(&self) -> bool {
        self.supports_u64_atomics
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
