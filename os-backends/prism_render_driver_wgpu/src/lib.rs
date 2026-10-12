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
//! through [`WgpuDevice::take_device_lost`]; see [`DeviceLostReason`].

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

pub use device::WgpuDevice;
pub use instance::{DeviceCreationError, WgpuAdapter, WgpuInstance};
pub use loss::DeviceLostReason;
pub use queue::WgpuQueue;
pub use surface::{SurfaceCreateError, WgpuFrame, WgpuSurface};
