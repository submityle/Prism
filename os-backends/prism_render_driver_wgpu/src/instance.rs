//! The [`WgpuInstance`]: wgpu instance creation, adapter enumeration and
//! selection, and blocking device/queue bring-up.
//!
//! The frozen RHI traits are synchronous, so every wgpu async call here is
//! driven to completion with [`futures_lite::future::block_on`].

use alloc::sync::Arc;
use alloc::vec::Vec;
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

/// The acceptance classification of a wgpu adapter.
///
/// This is intentionally stricter than merely obtaining a logical device:
/// null/software paths can create a device too, but they are not evidence that
/// a render slice executed on physical GPU hardware.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AdapterClass {
    /// An integrated or discrete physical GPU.
    Hardware,
    /// A CPU/software rasterizer.
    Software,
    /// A virtual or paravirtualized GPU.
    Virtual,
    /// An adapter whose type is not known well enough for hardware claims.
    Unknown,
}

/// Adapter identity and the acceptance classification derived from it.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct AdapterEvidence {
    /// The backend-reported adapter identity.
    pub info: rhi::AdapterInfo,
    /// The conservative acceptance class.
    pub class: AdapterClass,
}

/// Returns whether acceptance tests must require a physical GPU adapter.
#[must_use]
pub fn require_hardware_gpu() -> bool {
    matches!(std::env::var("PRISM_REQUIRE_GPU"), Ok(value) if value == "1")
}

impl AdapterEvidence {
    /// Classifies an RHI adapter description without creating a device.
    #[must_use]
    pub fn from_info(info: &rhi::AdapterInfo) -> Self {
        let class = match info.device_type {
            rhi::DeviceType::IntegratedGpu | rhi::DeviceType::DiscreteGpu => AdapterClass::Hardware,
            rhi::DeviceType::Cpu => AdapterClass::Software,
            rhi::DeviceType::VirtualGpu => AdapterClass::Virtual,
            rhi::DeviceType::Other => AdapterClass::Unknown,
        };
        Self {
            info: info.clone(),
            class,
        }
    }

    /// Returns whether this adapter is suitable for a physical-GPU acceptance
    /// claim.
    #[must_use]
    pub const fn is_hardware(&self) -> bool {
        matches!(self.class, AdapterClass::Hardware)
    }
}

#[cfg(test)]
mod adapter_evidence_tests {
    use super::*;

    fn info(device_type: rhi::DeviceType) -> rhi::AdapterInfo {
        rhi::AdapterInfo {
            name: "test-adapter".into(),
            backend: rhi::Backend::Metal,
            device_type,
            vendor_id: 1,
            device_id: 2,
            driver: Some("test-driver".into()),
        }
    }

    #[test]
    fn physical_gpu_classes_are_the_only_hardware_acceptance_class() {
        assert!(AdapterEvidence::from_info(&info(rhi::DeviceType::IntegratedGpu)).is_hardware());
        assert!(AdapterEvidence::from_info(&info(rhi::DeviceType::DiscreteGpu)).is_hardware());
        assert!(!AdapterEvidence::from_info(&info(rhi::DeviceType::Cpu)).is_hardware());
        assert!(!AdapterEvidence::from_info(&info(rhi::DeviceType::VirtualGpu)).is_hardware());
        assert!(!AdapterEvidence::from_info(&info(rhi::DeviceType::Other)).is_hardware());
    }
}

impl fmt::Display for AdapterEvidence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "name={:?} backend={:?} device_type={:?} vendor=0x{:04x} device=0x{:04x} driver={:?} class={:?}",
            self.info.name,
            self.info.backend,
            self.info.device_type,
            self.info.vendor_id,
            self.info.device_id,
            self.info.driver,
            self.class,
        )
    }
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

/// Why a physical-GPU acceptance device could not be created.
#[derive(Debug)]
pub enum HardwareDeviceError {
    /// No adapter was visible to wgpu.
    NoAdapter,
    /// An adapter was visible, but it is not physical-GPU evidence.
    NonHardware(AdapterEvidence),
    /// The adapter was hardware-classed but logical-device creation failed.
    Device(DeviceCreationError),
}

impl fmt::Display for HardwareDeviceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoAdapter => f.write_str("no wgpu adapter was found"),
            Self::NonHardware(evidence) => {
                write!(f, "adapter is not physical-GPU evidence: {evidence}")
            }
            Self::Device(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for HardwareDeviceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Device(error) => Some(error),
            Self::NoAdapter | Self::NonHardware(_) => None,
        }
    }
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

    /// Returns the adapter selected by wgpu's explicit software fallback gate,
    /// classified for diagnostics without creating a logical device.
    #[must_use]
    pub fn fallback_adapter_evidence(&self) -> Option<AdapterEvidence> {
        self.request_adapter(wgpu::PowerPreference::HighPerformance, true)
            .map(|adapter| AdapterEvidence::from_info(&WgpuAdapter::new(adapter).info))
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

    /// Converts an RHI device request using the same configuration as
    /// [`rhi::Instance::create_device`], without requesting a device.
    ///
    /// Mapped features are intersected with the adapter's supported set;
    /// limits use the existing RHI-to-wgpu conversion without additional
    /// clamping. Other wgpu descriptor fields keep their defaults, including
    /// the experimental-feature opt-in. A rejected limits/experimental-feature
    /// request remains an error from `wgpu::Adapter::request_device`.
    ///
    /// The label borrows only from `descriptor`, with the same fallback label
    /// as normal RHI bring-up when it is absent. External callers can request
    /// the original native tuple with this descriptor, retain native clones
    /// for sharing, and pass the tuple to [`WgpuDevice::own_original`]. This
    /// conversion does not authenticate tuple ancestry or enable extra features.
    #[must_use]
    pub fn device_descriptor<'a>(
        adapter: &wgpu::Adapter,
        descriptor: &'a rhi::DeviceDescriptor,
    ) -> wgpu::DeviceDescriptor<'a> {
        wgpu::DeviceDescriptor {
            label: descriptor
                .label
                .as_deref()
                .or(Some("prism_render_driver_wgpu device")),
            required_features: convert::features_to_wgpu(descriptor.required_features)
                & adapter.features(),
            required_limits: convert::limits_to_wgpu(descriptor.required_limits),
            ..Default::default()
        }
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
        self.bring_up(
            adapter,
            required_features,
            convert::limits_to_wgpu(limits),
            None,
        )
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

    /// Requests a device only when the selected adapter is classified as an
    /// integrated or discrete physical GPU.
    ///
    /// This gate is for acceptance tests and diagnostics. The ordinary
    /// [`Self::request_default_device`] path remains permissive because some
    /// applications intentionally opt into software or virtual adapters.
    pub fn request_default_hardware_device(
        &self,
    ) -> Result<(WgpuDevice, WgpuQueue), HardwareDeviceError> {
        let mut non_hardware = None;
        let mut integrated = None;
        let mut discrete = None;
        for adapter in self.enumerate_adapters() {
            let evidence = AdapterEvidence::from_info(&WgpuAdapter::new(adapter.clone()).info);
            if !evidence.is_hardware() {
                non_hardware.get_or_insert(evidence);
                continue;
            }
            match evidence.info.device_type {
                rhi::DeviceType::DiscreteGpu => discrete = Some(adapter),
                rhi::DeviceType::IntegratedGpu => integrated = Some(adapter),
                _ => unreachable!("hardware adapter classification is exhaustive"),
            }
        }
        let Some(adapter) = discrete.or(integrated) else {
            return Err(non_hardware
                .map(HardwareDeviceError::NonHardware)
                .unwrap_or(HardwareDeviceError::NoAdapter));
        };
        self.create_device(&adapter)
            .map_err(HardwareDeviceError::Device)
    }

    /// Returns evidence for the adapter chosen by the normal permissive path.
    #[must_use]
    pub fn default_adapter_evidence(&self) -> Option<AdapterEvidence> {
        self.request_adapter(wgpu::PowerPreference::HighPerformance, false)
            .map(|adapter| AdapterEvidence::from_info(&WgpuAdapter::new(adapter).info))
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
        let descriptor = Self::device_descriptor(&adapter.adapter, descriptor);
        self.bring_up(
            &adapter.adapter,
            descriptor.required_features,
            descriptor.required_limits,
            descriptor.label,
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
        WgpuSurface::create(
            self,
            window,
            device.inner(),
            queue.wgpu_queue().clone(),
            None,
        )
        .map_err(|_| rhi::SurfaceError::Unsupported)
    }
}
