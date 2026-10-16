//! The Prism RHI GPU context and its render-world plugin.
//!
//! This module is the migration seam for replacing `bevy_render` with Prism's
//! RHI. It owns one device/queue pair created through the RHI's `Instance`
//! facade and hands it to the render world as the [`PrismGpu`] resource.

mod context;
pub(crate) mod fabric;
mod plugin;
pub(crate) mod rhi;

pub use context::{PrismGpu, PrismGpuInitError};
pub use plugin::PrismGpuPlugin;
