//! # `prism_render_driver_null`
//!
//! A headless reference implementation of Prism's render hardware interface
//! ([`prism_render_driver`]) that runs everywhere: no GPU, no window, no
//! driver. It exists so CI and GPU-less hosts can drive the render graph and
//! every upper layer through the frozen RHI traits without hardware.
//!
//! This is a real backend, not a stub: every trait method is implemented and
//! every `create_*` call mints a generation-checked id in a
//! [`prism_render_driver::GenerationalSlotMap`]. Stale-id detection, slot
//! recycling, capability reporting, and the surface configure → acquire →
//! present cycle all behave as they do on a hardware backend. What is absent
//! is only the native GPU object behind each id, which is why the design calls
//! it the *null* backend: it observes the full RHI contract and produces no
//! pixels.
//!
//! ## Layout
//! - [`NullInstance`] / [`NullAdapter`]: the entry point and its single
//!   [`prism_render_driver::DeviceType::Cpu`] /
//!   [`prism_render_driver::Backend::Noop`] adapter.
//! - [`NullDevice`]: the [`prism_render_driver::RenderDevice`] resource factory.
//! - [`NullQueue`]: the no-op [`prism_render_driver::RenderQueue`].
//! - [`NullSurface`] / [`NullFrame`]: the CPU-side [`prism_render_driver::Surface`].

extern crate alloc;

mod device;
mod instance;
mod queue;
mod surface;

#[cfg(test)]
mod tests;

pub use device::NullDevice;
pub use instance::{NullAdapter, NullInstance};
pub use queue::NullQueue;
pub use surface::{NullFrame, NullSurface};
