//! The null [`Instance`] and its single CPU adapter.
//!
//! The instance is the RHI entry point: it enumerates exactly one adapter (a
//! [`DeviceType::Cpu`] / [`Backend::Noop`] device), brings up a
//! [`NullDevice`]/[`NullQueue`] pair from it, and creates [`NullSurface`]s.
//! Window handles are accepted with the same bound the frozen trait declares
//! but never inspected, so surface creation cannot fail on any host.

use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;

use prism_render_driver as rhi;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use rhi::{
    Adapter, AdapterInfo, Backend, DeviceCapabilities, DeviceDescriptor, DeviceError, DeviceType,
    Features, Instance, Limits, RequestAdapterOptions, SurfaceError,
};

use crate::device::NullDevice;
use crate::queue::NullQueue;
use crate::surface::NullSurface;

/// The null RHI instance: the entry point for adapter enumeration and
/// device/surface creation.
#[derive(Clone, Copy, Debug)]
pub struct NullInstance {
    caps: DeviceCapabilities,
}

impl NullInstance {
    /// Creates a null instance.
    #[must_use]
    pub const fn new() -> Self {
        Self::with_capabilities(DeviceCapabilities {
            backend: Backend::Noop,
            features: Features::NONE,
            limits: Limits::baseline(),
        })
    }

    /// Configures the advertised contract for resource-lifecycle and graph tests.
    ///
    /// This does not execute shaders or emulate optional hardware functionality.
    /// The backend identity remains [`Backend::Noop`] regardless of the input.
    #[must_use]
    pub const fn with_capabilities(mut caps: DeviceCapabilities) -> Self {
        caps.backend = Backend::Noop;
        Self { caps }
    }
}

impl Default for NullInstance {
    fn default() -> Self {
        Self::new()
    }
}

/// The null backend's single adapter: a CPU/[`Backend::Noop`] device that
/// advertises the instance's explicitly configured feature and limit contract.
pub struct NullAdapter {
    /// The adapter's identifying description.
    info: AdapterInfo,
    /// The capabilities the adapter offers.
    caps: DeviceCapabilities,
}

impl NullAdapter {
    /// Builds an adapter with the instance's advertised contract.
    fn new(caps: DeviceCapabilities) -> Self {
        Self {
            info: AdapterInfo {
                name: String::from("Prism Null Adapter"),
                backend: Backend::Noop,
                device_type: DeviceType::Cpu,
                vendor_id: 0,
                device_id: 0,
                driver: Some(String::from("prism_render_driver_null")),
            },
            caps,
        }
    }
}

impl Adapter for NullAdapter {
    fn info(&self) -> &AdapterInfo {
        &self.info
    }

    fn capabilities(&self) -> &DeviceCapabilities {
        &self.caps
    }
}

impl Instance for NullInstance {
    type Adapter = NullAdapter;
    type Device = NullDevice;
    type Queue = NullQueue;
    type Surface = NullSurface;

    fn enumerate_adapters(&self) -> Vec<NullAdapter> {
        alloc::vec![NullAdapter::new(self.caps)]
    }

    fn request_adapter(&self, _options: &RequestAdapterOptions) -> Option<NullAdapter> {
        Some(NullAdapter::new(self.caps))
    }

    fn request_adapter_for_surface(
        &self,
        _options: &RequestAdapterOptions,
        _surface: &NullSurface,
    ) -> Option<NullAdapter> {
        Some(NullAdapter::new(self.caps))
    }

    fn create_device(
        &self,
        adapter: &NullAdapter,
        descriptor: &DeviceDescriptor,
    ) -> Result<(NullDevice, NullQueue), DeviceError> {
        let caps = adapter.capabilities();
        if !caps.supports(descriptor.required_features, &descriptor.required_limits) {
            return Err(DeviceError::Request(alloc::format!(
                "null adapter cannot satisfy requested features {} / limits {:?}",
                descriptor.required_features.bits(),
                descriptor.required_limits
            )));
        }
        Ok((NullDevice::new(*caps), NullQueue::new()))
    }

    fn create_surface<W>(
        &self,
        _window: Arc<W>,
        device: &NullDevice,
        _queue: &NullQueue,
    ) -> Result<NullSurface, SurfaceError>
    where
        W: HasWindowHandle + HasDisplayHandle + Send + Sync + 'static,
    {
        Ok(NullSurface::new(device.clone()))
    }
}
