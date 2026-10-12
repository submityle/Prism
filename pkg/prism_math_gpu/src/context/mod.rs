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
//! shader-mirror twin never couples to another crate's private module tree.
//!
//! Provenance: standard `wgpu` initialisation; no Unreal Engine or Unity source
//! or derived code.

use wgpu::{
    Adapter, BackendOptions, Backends, Device, DeviceDescriptor, Instance, InstanceDescriptor,
    InstanceFlags, PollType, Queue, RequestAdapterOptions,
};

/// When set to a non-empty value other than `0`, adapter/device acquisition
/// failures become hard errors. GPU CI should set this to prevent a green run
/// that only exercised the device-free parse tests.
const REQUIRE_GPU_ENV: &str = "PRISM_MATH_GPU_REQUIRED";
/// When set, prints the selected adapter name, backend, and device type.
const GPU_DIAGNOSTICS_ENV: &str = "PRISM_MATH_GPU_DIAGNOSTICS";

fn env_flag(name: &str) -> bool {
    flag_value(std::env::var_os(name))
}

fn flag_value(value: Option<std::ffi::OsString>) -> bool {
    value.is_some_and(|value| value != "0" && !value.is_empty())
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
    /// Returns [`None`] when no adapter or device can be obtained so callers can
    /// skip `GPU` work instead of panicking on a host without a usable adapter.
    #[must_use]
    #[expect(
        clippy::print_stderr,
        reason = "opt-in adapter diagnostics are intentionally written to stderr"
    )]
    pub fn try_headless() -> Option<GpuContext> {
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::METAL | Backends::VULKAN | Backends::DX12,
            flags: InstanceFlags::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            display: None,
            backend_options: BackendOptions::default(),
        });
        let required = env_flag(REQUIRE_GPU_ENV);
        let adapter = match block_on(instance.request_adapter(&RequestAdapterOptions::default())) {
            Ok(adapter) => adapter,
            Err(err) => {
                if required {
                    panic!("{REQUIRE_GPU_ENV} is enabled, but no GPU adapter is available: {err}");
                }
                return None;
            }
        };
        let info = adapter.get_info();
        let (device, queue) = match block_on(adapter.request_device(&DeviceDescriptor {
            required_limits: adapter.limits(),
            ..Default::default()
        })) {
            Ok(device_queue) => device_queue,
            Err(err) => {
                if required {
                    panic!(
                        "{REQUIRE_GPU_ENV} is enabled, but requesting a device from {} ({:?}) failed: {err}",
                        info.name,
                        info.backend
                    );
                }
                return None;
            }
        };
        if env_flag(GPU_DIAGNOSTICS_ENV) {
            eprintln!(
                "prism_math_gpu selected adapter: {} ({:?}, {:?})",
                info.name, info.backend, info.device_type
            );
        }
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
    #[test]
    fn env_flag_accepts_nonzero_values_only() {
        assert!(!super::flag_value(None));
        assert!(!super::flag_value(Some("".into())));
        assert!(!super::flag_value(Some("0".into())));
        assert!(super::flag_value(Some("1".into())));
        assert!(super::flag_value(Some("true".into())));
    }
}
