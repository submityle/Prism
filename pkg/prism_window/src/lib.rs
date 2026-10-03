//! # `prism_window`
//!
//! Prism's window kernel: a deterministic, backend-agnostic model of window
//! configuration, per-window runtime state, display/monitor info, present
//! mode, and window-level events. OS/windowing backends (winit, SDL, native)
//! live in separate crates; they translate platform messages into
//! [`WindowEvent`]s, drive [`Window`] state with [`Window::apply`], and read
//! back the desired [`WindowAttributes`] to realize on the OS.
//!
//! It is a pure, deterministic, `no_std + alloc` type system with no `unsafe`.
//! Geometry conversions avoid `std` float intrinsics (no `round`/`floor`, no
//! `libm`), and the [`WindowEvent`] stream is `Copy` for cheap record/replay.
//!
//! ## Layout
//! - Geometry: [`geometry`] (physical/logical sizes, positions, scale).
//! - Resolution: [`resolution`] (size + scale, resize constraints).
//! - Enums: [`mode`] (window/present/level/alpha/theme), [`cursor`].
//! - Displays: [`monitor`] (monitor + video mode).
//! - State & stream: [`window`] ([`Window`], [`WindowAttributes`]), [`event`].
#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]

extern crate alloc;

pub mod cursor;
pub mod event;
pub mod geometry;
pub mod mode;
pub mod monitor;
pub mod resolution;
pub mod window;

pub use cursor::{CursorGrabMode, CursorIcon, CursorOptions};
pub use event::WindowEvent;
pub use geometry::{LogicalPosition, LogicalSize, PhysicalPosition, PhysicalSize};
pub use mode::{CompositeAlphaMode, PresentMode, WindowLevel, WindowMode, WindowTheme};
pub use monitor::{Monitor, MonitorId, VideoMode};
pub use resolution::{WindowResizeConstraints, WindowResolution};
pub use window::{Window, WindowAttributes, WindowId};

/// The common types most consumers import.
pub mod prelude {
    pub use crate::cursor::{CursorGrabMode, CursorIcon, CursorOptions};
    pub use crate::event::WindowEvent;
    pub use crate::geometry::{LogicalPosition, LogicalSize, PhysicalPosition, PhysicalSize};
    pub use crate::mode::{CompositeAlphaMode, PresentMode, WindowLevel, WindowMode, WindowTheme};
    pub use crate::monitor::{Monitor, MonitorId, VideoMode};
    pub use crate::resolution::{WindowResizeConstraints, WindowResolution};
    pub use crate::window::{Window, WindowAttributes, WindowId};
}

#[cfg(test)]
mod tests;
