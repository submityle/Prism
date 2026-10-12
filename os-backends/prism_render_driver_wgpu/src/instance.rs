//! The [`WgpuInstance`]: wgpu instance creation, adapter enumeration and
//! selection, and blocking device/queue bring-up.
//!
//! The frozen RHI traits are synchronous, so every wgpu async call here is
//! driven to completion with [`futures_lite::future::block_on`].

use alloc::vec::Vec;
use alloc::sync::Arc;
use core::fmt;

use prism_render_driver as rhi;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use crate::convert;
use crate::device::WgpuDevice;
use crate::queue::WgpuQueue;
use crate::surface::WgpuSurface;

/// A wgpu physical device behind the frozen [`rhi::Adapter`] trait.
pub struct WgpuAdapter {
    /// The underlying wgpu adapter.
    adapter: wgpu::Adapter,
    /// The RHI-facing adapter description.
    info: rhi::AdapterInfo,
    /// The features/limits the adapter can offer.
    caps: rhi::DeviceCapabilities,
}

impl WgpuAdapter {
    /// Wraps a wgpu adapter and derives its RHI description and capabilities.
    #[must_use]
    pub fn new(adapter: wgpu::Adapter) -> Self {
        let info = adapter.get_info();
        let backend = convert::backend_from_wgpu(info.backend);
        let rhi_info = rhi::AdapterInfo {
            name: info.name,
            backend,
            device_type: convert::device_type_from_wgpu(info.device_type),
            vendor_id: info.vendor,
            device_id: info.device,
            driver: if info.driver.is_empty() {
                None
            } else {
                Some(info.driver)
            },
        };
        let caps = rhi::DeviceCapabilities {
            backend,
            features: convert::features_from_wgpu(adapter.features()),
            limits: convert::limits_from_wgpu(&adapter.limits()),
        };
        Self {
            adapter,
            info: rhi_info,
            caps,
        }
    }

    /// Borrows the underlying wgpu adapter.
    #[must_use]
    pub fn wgpu_adapter(&self) -> &wgpu::Adapter {
        &self.adapter
    }
}

impl rhi::Adapter for WgpuAdapter {
    fn info(&self) -> &rhi::AdapterInfo {
        &self.info
    }

    fn capabilities(&self) -> &rhi::DeviceCapabilities {
        &self.caps
    }
}

/// A wgpu instance: the entry point for enumerating adapters and bringing up a
/// [`WgpuDevice`]/[`WgpuQueue`] pair.
pub struct WgpuInstance {
    /// The underlying wgpu instance.
    instance: wgpu::Instance,
}

/// An error raised while bringing up a [`WgpuDevice`] from an adapter.
#[derive(Debug)]
pub enum DeviceCreationError {
    /// No adapter matched the requested selection options.
    NoAdapter,
    /// The adapter rejected the device request (features/limits mismatch, or a
    /// driver-level failure).
    RequestDevice(wgpu::RequestDeviceError),
}

impl fmt::Display for DeviceCreationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoAdapter => f.write_str("no suitable wgpu adapter was found"),
            Self::RequestDevice(err) => write!(f, "wgpu device request failed: {err}"),
        }
    }
}

impl std::error::Error for DeviceCreationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::NoAdapter => None,
            Self::RequestDevice(err) => Some(err),
        }
    }
}

impl Default for WgpuInstance {
    fn default() -> Self {
        Self::new()
    }
}

impl WgpuInstance {
    /// Creates an instance with wgpu's default backend/flag configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            instance: wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle()),
        }
    }

    /// Wraps an already-configured wgpu instance.
    #[must_use]
    pub fn from_wgpu(instance: wgpu::Instance) -> Self {
        Self { instance }
    }

    /// Borrows the underlying wgpu instance, e.g. to create a surface.
    #[must_use]
    pub fn wgpu_instance(&self) -> &wgpu::Instance {
        &self.instance
    }

    /// Enumerates every adapter across all compiled-in backends.
    #[must_use]
    pub fn enumerate_adapters(&self) -> Vec<wgpu::Adapter> {
        futures_lite::future::block_on(self.instance.enumerate_adapters(wgpu::Backends::all()))
    }

    /// Requests an adapter matching `power_preference`, blocking until the
    /// request resolves. Returns `None` if no adapter is available.
    #[must_use]
    pub fn request_adapter(
        &self,
        power_preference: wgpu::PowerPreference,
        force_fallback_adapter: bool,
    ) -> Option<wgpu::Adapter> {
        self.request_adapter_with_surface(power_preference, force_fallback_adapter, None)
    }

    /// Requests an adapter that can present to `compatible_surface` (when
    /// given), blocking until the request resolves.
    #[must_use]
    pub fn request_adapter_with_surface(
        &self,
        power_preference: wgpu::PowerPreference,
        force_fallback_adapter: bool,
        compatible_surface: Option<&wgpu::Surface<'_>>,
    ) -> Option<wgpu::Adapter> {
        futures_lite::future::block_on(self.instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference,
                force_fallback_adapter,
                compatible_surface,
                apply_limit_buckets: false,
            },
        ))
        .ok()
    }

    /// Brings up a [`WgpuDevice`]/[`WgpuQueue`] pair from `adapter`.
    ///
    /// Requests every non-experimental feature the adapter supports (plus
    /// `IMMEDIATES` for push constants when available) and the adapter's full
    /// limits, then reports the device's actual features/limits through
    /// [`rhi::RenderDevice::capabilities`].
    pub fn create_device(
        &self,
        adapter: &wgpu::Adapter,
    ) -> Result<(WgpuDevice, WgpuQueue), DeviceCreationError> {
        let supported = adapter.features();

        // Everything the RHI knows how to map, minus the experimental features
        // that would additionally require the experimental-features opt-in.
        let mappable = convert::features_to_wgpu(rhi::Features::all())
            & !(wgpu::Features::EXPERIMENTAL_RAY_QUERY | wgpu::Features::EXPERIMENTAL_MESH_SHADER);
        let mut required_features = mappable & supported;
        if supported.contains(wgpu::Features::IMMEDIATES) {
            required_features |= wgpu::Features::IMMEDIATES;
        }

        self.bring_up(adapter, required_features, adapter.limits(), None)
    }

    /// Brings up a device constrained to the RHI `features` and `limits` the
    /// caller actually needs.
    ///
    /// Unlike [`create_device`](Self::create_device), which opportunistically
    /// enables everything the adapter supports, this requests exactly the
    /// mapped `features` (intersected with the adapter's supported set so an
    /// unsupported request cannot panic) and the `limits` translated into wgpu limits. Use this when a renderer wants a device
    /// that provably satisfies a known RHI limit/feature floor.
    pub fn create_device_with(
        &self,
        adapter: &wgpu::Adapter,
        features: rhi::Features,
        limits: rhi::Limits,
    ) -> Result<(WgpuDevice, WgpuQueue), DeviceCreationError> {
        let required_features = convert::features_to_wgpu(features) & adapter.features();
        self.bring_up(adapter, required_features, convert::limits_to_wgpu(limits), None)
    }

    /// Shared device/queue bring-up: issues the blocking `request_device`,
    /// derives real capabilities from the created device, and wires the
    /// shared device state into the queue.
    fn bring_up(
        &self,
        adapter: &wgpu::Adapter,
        required_features: wgpu::Features,
        required_limits: wgpu::Limits,
        label: Option<&str>,
    ) -> Result<(WgpuDevice, WgpuQueue), DeviceCreationError> {
        let (device, queue) =
            futures_lite::future::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: label.or(Some("prism_render_driver_wgpu device")),
                required_features,
                required_limits,
                ..Default::default()
            }))
            .map_err(DeviceCreationError::RequestDevice)?;

        let caps = rhi::DeviceCapabilities {
            backend: convert::backend_from_wgpu(adapter.get_info().backend),
            features: convert::features_from_wgpu(device.features()),
            limits: convert::limits_from_wgpu(&device.limits()),
        };

        let wgpu_device = WgpuDevice::from_parts(adapter.clone(), device, caps);
        let inner = wgpu_device.inner();
        let wgpu_queue = WgpuQueue::from_parts(queue, inner);
        Ok((wgpu_device, wgpu_queue))
    }

    /// Convenience bring-up: selects an adapter by `power_preference` and
    /// creates a device/queue from it.
    pub fn request_device(
        &self,
        power_preference: wgpu::PowerPreference,
    ) -> Result<(WgpuDevice, WgpuQueue), DeviceCreationError> {
        let adapter = self
            .request_adapter(power_preference, false)
            .ok_or(DeviceCreationError::NoAdapter)?;
        self.create_device(&adapter)
    }

    /// Convenience bring-up requiring no wgpu types from the caller: selects a
    /// high-performance adapter and brings up a device/queue, or returns
    /// [`DeviceCreationError::NoAdapter`] when no GPU is present.
    ///
    /// This is the integration-friendly entry point for callers (such as the
    /// crate's GPU-gated tests) that depend only on `prism_render_driver_wgpu`
    /// and never name a `wgpu` type directly.
    pub fn request_default_device(&self) -> Result<(WgpuDevice, WgpuQueue), DeviceCreationError> {
        self.request_device(wgpu::PowerPreference::HighPerformance)
    }
}

impl rhi::Instance for WgpuInstance {
    type Adapter = WgpuAdapter;
    type Device = WgpuDevice;
    type Queue = WgpuQueue;
    type Surface = WgpuSurface;

    fn enumerate_adapters(&self) -> Vec<WgpuAdapter> {
        self.enumerate_adapters()
            .into_iter()
            .map(WgpuAdapter::new)
            .collect()
    }

    fn request_adapter(&self, options: &rhi::RequestAdapterOptions) -> Option<WgpuAdapter> {
        let preference = convert::power_preference_to_wgpu(options.power_preference);
        self.request_adapter(preference, options.force_fallback)
            .map(WgpuAdapter::new)
    }

    fn request_adapter_for_surface(
        &self,
        options: &rhi::RequestAdapterOptions,
        surface: &WgpuSurface,
    ) -> Option<WgpuAdapter> {
        let preference = convert::power_preference_to_wgpu(options.power_preference);
        self.request_adapter_with_surface(
            preference,
            options.force_fallback,
            Some(surface.wgpu_surface()),
        )
        .map(WgpuAdapter::new)
    }

    fn create_device(
        &self,
        adapter: &WgpuAdapter,
        descriptor: &rhi::DeviceDescriptor,
    ) -> Result<(WgpuDevice, WgpuQueue), rhi::DeviceError> {
        // Intersect the request with what the adapter supports so an
        // over-specified descriptor degrades instead of failing the request.
        let features = convert::features_to_wgpu(descriptor.required_features)
            & adapter.adapter.features();
        let limits = convert::limits_to_wgpu(descriptor.required_limits);
        self.bring_up(
            &adapter.adapter,
            features,
            limits,
            descriptor.label.as_deref(),
        )
        .map_err(|error| rhi::DeviceError::Request(alloc::format!("{error}")))
    }

    fn create_surface<W>(
        &self,
        window: Arc<W>,
        device: &WgpuDevice,
        queue: &WgpuQueue,
    ) -> Result<WgpuSurface, rhi::SurfaceError>
    where
        W: HasWindowHandle + HasDisplayHandle + Send + Sync + 'static,
    {
        WgpuSurface::create(self, window, device.inner(), queue.wgpu_queue().clone(), None)
            .map_err(|_| rhi::SurfaceError::Unsupported)
    }
}
