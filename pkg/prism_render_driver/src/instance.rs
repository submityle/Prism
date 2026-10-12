//! Instance and Adapter: the RHI entry point and physical-device selection.
//!
//! A backend's [`Instance`] is the root of the RHI: it enumerates
//! [`Adapter`]s (physical devices), creates a [`Surface`] for a window, and
//! brings up a device/queue pair from a chosen adapter. It is the seam that lets
//! the renderer switch backends without naming any graphics API: upper layers
//! are generic over the instance's associated types and only ever touch the
//! frozen traits.
//!
//! Window handles are accepted as owned [`Arc`]s of any
//! `raw-window-handle`-implementing type. The RHI carries only the handle ABI
//! (never a windowing system), matching the layering in the design: the window
//! crate owns the OS window, the RHI turns its handle into a surface.

use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::fmt;

use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use crate::capabilities::{Backend, DeviceCapabilities, Features, Limits};
use crate::device::{RenderDevice, RenderQueue};
use crate::surface::{Surface, SurfaceError};

/// The kind of physical device an [`Adapter`] represents.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum DeviceType {
    /// The type could not be determined.
    #[default]
    Other,
    /// A GPU integrated into the CPU package (shared memory).
    IntegratedGpu,
    /// A discrete GPU with its own memory.
    DiscreteGpu,
    /// A virtualized / paravirtualized GPU.
    VirtualGpu,
    /// A software rasterizer running on the CPU.
    Cpu,
}

/// How the runtime prefers to pick an adapter when several are available.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum PowerPreference {
    /// No preference; let the backend decide.
    #[default]
    None,
    /// Prefer a low-power adapter (typically integrated).
    LowPower,
    /// Prefer the highest-performance adapter (typically discrete).
    HighPerformance,
}

/// A machine-identifying description of an adapter, for selection and logs.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct AdapterInfo {
    /// A human-readable adapter name (e.g. `"Apple M3 Pro"`).
    pub name: String,
    /// The backend that owns this adapter.
    pub backend: Backend,
    /// The kind of physical device.
    pub device_type: DeviceType,
    /// The PCI/USB vendor id, or `0` when unknown.
    pub vendor_id: u32,
    /// The device id within the vendor, or `0` when unknown.
    pub device_id: u32,
    /// A driver/version description, or `None` when unknown.
    pub driver: Option<String>,
}

/// Options constraining [`Instance::request_adapter`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct RequestAdapterOptions {
    /// The preferred power/performance trade-off.
    pub power_preference: PowerPreference,
    /// Allow a software/fallback (CPU) adapter when no hardware adapter exists.
    pub force_fallback: bool,
}

/// A request to create a device from a chosen adapter.
///
/// The backend intersects `required_features`/`required_limits` with what the
/// adapter actually supports; the device it returns reports its real
/// capabilities through [`RenderDevice::capabilities`].
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct DeviceDescriptor {
    /// A debug label surfaced in GPU tooling.
    pub label: Option<String>,
    /// Features the device must enable.
    pub required_features: Features,
    /// Limits the device must guarantee.
    pub required_limits: Limits,
}

impl Default for DeviceDescriptor {
    fn default() -> Self {
        Self {
            label: None,
            required_features: Features::NONE,
            required_limits: Limits::default(),
        }
    }
}

/// Why a device could not be brought up from an adapter.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum DeviceError {
    /// The adapter rejected the device request (features/limits mismatch, or a
    /// driver-level failure); the string carries the backend's diagnostic.
    Request(String),
}

impl fmt::Display for DeviceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Request(detail) => write!(f, "device request was rejected: {detail}"),
        }
    }
}

/// A physical device an [`Instance`] can enumerate and create a device from.
pub trait Adapter {
    /// The adapter's identifying description.
    fn info(&self) -> &AdapterInfo;

    /// The capabilities (features + limits) this adapter can offer.
    fn capabilities(&self) -> &DeviceCapabilities;
}

/// The RHI entry point: a backend instance that owns adapters and surfaces.
///
/// Associated types carry the backend's concrete device/queue/surface so the
/// upper layer is generic and never names a graphics API. Adding a backend is
/// then a new `Instance` implementation, with the renderer unchanged.
pub trait Instance {
    /// The adapter handle type.
    type Adapter: Adapter;
    /// The device handle type this instance brings up.
    type Device: RenderDevice;
    /// The queue handle type this instance brings up.
    type Queue: RenderQueue;
    /// The surface handle type this instance creates.
    type Surface: Surface;

    /// Enumerates every adapter the backend can see.
    fn enumerate_adapters(&self) -> Vec<Self::Adapter>;

    /// Requests an adapter matching `options`, or `None` when none is available.
    fn request_adapter(&self, options: &RequestAdapterOptions) -> Option<Self::Adapter>;

    /// Requests an adapter that can present to `surface`.
    fn request_adapter_for_surface(
        &self,
        options: &RequestAdapterOptions,
        surface: &Self::Surface,
    ) -> Option<Self::Adapter>;

    /// Brings up a device/queue pair from `adapter`.
    ///
    /// # Errors
    ///
    /// Returns [`DeviceError`] when the adapter rejects the request.
    fn create_device(
        &self,
        adapter: &Self::Adapter,
        descriptor: &DeviceDescriptor,
    ) -> Result<(Self::Device, Self::Queue), DeviceError>;

    /// Creates a surface targeting `window`.
    ///
    /// The window is owned (an [`Arc`]), so the surface has no borrow of the
    /// caller and can outlive the call site.
    ///
    /// # Errors
    ///
    /// Returns [`SurfaceError`] when the platform cannot create a surface for
    /// the handle.
    fn create_surface<W>(
        &self,
        window: Arc<W>,
        device: &Self::Device,
        queue: &Self::Queue,
    ) -> Result<Self::Surface, SurfaceError>
    where
        W: HasWindowHandle + HasDisplayHandle + Send + Sync + 'static;
}
