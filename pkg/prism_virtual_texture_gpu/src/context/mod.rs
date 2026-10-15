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

use alloc::sync::Arc;
use std::sync::Mutex;

use wgpu::{
    Adapter, BackendOptions, Backends, Device, DeviceDescriptor, DeviceLostReason, Instance,
    InstanceDescriptor, InstanceFlags, PollType, Queue, RequestAdapterOptions,
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
    _instance: Instance,
    /// The physical adapter backing [`GpuContext::device`].
    adapter: Adapter,
    /// The logical device used to allocate resources and pipelines.
    device: Device,
    /// The queue used to submit command buffers.
    queue: Queue,
    lost: Arc<Mutex<Option<DeviceLostState>>>,
}

/// The adapter classes accepted by the real-GPU gate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuAdapterClass {
    /// A hardware-integrated GPU.
    Integrated,
    /// A hardware-discrete GPU.
    Discrete,
    /// A CPU or software adapter.
    Cpu,
    /// An adapter class not accepted by the real-GPU gate.
    Other,
}

impl GpuAdapterClass {
    /// Whether the adapter is integrated or discrete hardware.
    #[must_use]
    pub const fn is_real(self) -> bool {
        matches!(self, Self::Integrated | Self::Discrete)
    }
}

/// Evidence captured from the adapter selected by wgpu.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuAdapterEvidence {
    /// Adapter name reported by wgpu.
    pub name: String,
    /// Backend selected by wgpu.
    pub backend: String,
    /// Normalized adapter class used by the hardware gate.
    pub class: GpuAdapterClass,
}

/// Device-loss evidence retained after wgpu invokes its callback.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeviceLostState {
    /// Reason reported by wgpu for the device loss.
    pub reason: DeviceLostReason,
    /// Diagnostic message reported by wgpu.
    pub message: String,
}

/// Errors that can be recovered from by switching to a CPU path or rebuilding
/// the GPU context.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GpuError {
    /// The logical device has been lost and must be recreated.
    DeviceLost(DeviceLostState),
    /// Polling the device failed before submitted work completed.
    Poll(String),
    /// The asynchronous readback mapping callback reported an error.
    ReadbackMapFailed(String),
    /// The mapped range could not be acquired after a successful map callback.
    ReadbackRangeUnavailable(String),
    /// A host-side count exceeded the `u32` ABI accepted by a shader.
    AbiLimitExceeded {
        /// Name of the count crossing the shader ABI boundary.
        resource: &'static str,
        /// Host-side count that could not be represented.
        value: usize,
    },
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DeviceLost(state) => {
                write!(f, "device lost ({:?}): {}", state.reason, state.message)
            }
            Self::Poll(message) => write!(f, "GPU poll failed: {message}"),
            Self::ReadbackMapFailed(message) => {
                write!(f, "GPU readback buffer mapping failed: {message}")
            }
            Self::ReadbackRangeUnavailable(message) => {
                write!(f, "GPU readback mapped range is unavailable: {message}")
            }
            Self::AbiLimitExceeded { resource, value } => {
                write!(
                    f,
                    "GPU ABI count for {resource} is not representable: {value}"
                )
            }
        }
    }
}

impl std::error::Error for GpuError {}

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
        let lost = Arc::new(Mutex::new(None));
        let callback_lost = Arc::clone(&lost);
        device.set_device_lost_callback(move |reason, message| {
            if let Ok(mut state) = callback_lost.lock() {
                *state = Some(DeviceLostState { reason, message });
            }
        });
        Some(GpuContext {
            _instance: instance,
            adapter,
            device,
            queue,
            lost,
        })
    }

    /// Acquires a device only when wgpu reports an integrated or discrete GPU.
    /// CPU/software adapters are deliberately rejected so hardware-gated tests
    /// cannot pass by silently running on a software implementation.
    #[must_use]
    pub fn try_real_headless() -> Option<GpuContext> {
        let context = Self::try_headless()?;
        if context.is_real_gpu() {
            Some(context)
        } else if std::env::var_os("PRISM_REQUIRE_GPU").is_some() {
            panic!(
                "PRISM_REQUIRE_GPU is set, but selected adapter is not a real GPU: {:?}",
                context.adapter_evidence()
            );
        } else {
            None
        }
    }

    /// Returns the adapter identity and hardware class selected by wgpu.
    #[must_use]
    pub fn adapter_evidence(&self) -> GpuAdapterEvidence {
        let info = self.adapter.get_info();
        GpuAdapterEvidence {
            name: info.name,
            backend: format!("{:?}", info.backend),
            class: match info.device_type {
                wgpu::DeviceType::IntegratedGpu => GpuAdapterClass::Integrated,
                wgpu::DeviceType::DiscreteGpu => GpuAdapterClass::Discrete,
                wgpu::DeviceType::Cpu => GpuAdapterClass::Cpu,
                _ => GpuAdapterClass::Other,
            },
        }
    }

    /// Returns true only for integrated or discrete GPU adapters.
    #[must_use]
    pub fn is_real_gpu(&self) -> bool {
        self.adapter_evidence().class.is_real()
    }

    /// Returns device-loss evidence if the wgpu device has reported loss.
    #[must_use]
    pub fn device_lost(&self) -> Option<DeviceLostState> {
        self.lost.lock().ok().and_then(|state| state.clone())
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
    /// Returns a device-loss or polling error instead of panicking, so callers
    /// can discard the context and continue through a CPU fallback.
    pub fn wait(&self) -> Result<(), GpuError> {
        if let Some(state) = self.device_lost() {
            return Err(GpuError::DeviceLost(state));
        }
        self.device
            .poll(PollType::wait_indefinitely())
            .map_err(|error| GpuError::Poll(error.to_string()))?;
        if let Some(state) = self.device_lost() {
            return Err(GpuError::DeviceLost(state));
        }
        Ok(())
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
