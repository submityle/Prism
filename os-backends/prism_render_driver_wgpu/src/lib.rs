//! wgpu 30 backend for `prism_render_driver`.
//!
//! This crate implements the frozen render hardware interface (RHI) defined in
//! [`prism_render_driver`] on top of wgpu 30. It provides:
//!
//! - [`WgpuInstance`]: wgpu instance creation, adapter enumeration/selection,
//!   and blocking device/queue bring-up.
//! - [`WgpuDevice`]: a [`prism_render_driver::RenderDevice`] that owns a
//!   `wgpu::Device` and every GPU resource behind generational ids.
//! - [`WgpuQueue`]: a [`prism_render_driver::RenderQueue`] that uploads data and
//!   replays backend-agnostic command buffers through a `wgpu::CommandEncoder`.
//!
//! Window-surface support (configure/acquire/present) is available behind the
//! `surface` feature.
//!
//! Device loss (a driver reset or crash, or an explicit `destroy`) is reported
//! through [`RenderDevice::take_device_lost`](prism_render_driver::RenderDevice::take_device_lost);
//! see [`DeviceLostReason`]. This applies to devices created by this backend
//! and original native tuples passed to [`WgpuDevice::own_original`], which
//! takes over the loss callback and creates fresh RHI resource tables. Tuple
//! ancestry is caller-supplied, not authenticated. External native acquisition
//! can use [`WgpuInstance::device_descriptor`] for the same configuration as
//! [`prism_render_driver::Instance::create_device`].
//! [`WgpuDevice::adopt`] instead preserves the host's loss callback and leaves
//! recovery to the host. [`WgpuQueue::encode`] supports record-only native
//! handoff without transferring submission ownership to Prism.

extern crate alloc;

mod convert;
mod device;
mod instance;
mod loss;
mod queue;
mod surface;

/// GPU-gated bindless descriptor-array conformance tests.
#[cfg(test)]
mod bindless_tests;

pub use device::{BackendError, ShaderSourceKind, WgpuDevice};
pub use instance::{
    require_hardware_gpu, AdapterClass, AdapterEvidence, DeviceCreationError, HardwareDeviceError,
    WgpuAdapter, WgpuInstance,
};
pub use prism_render_driver::DeviceLostReason;
pub use queue::WgpuQueue;
pub use surface::{SurfaceCreateError, WgpuFrame, WgpuSurface};
