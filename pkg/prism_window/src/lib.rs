//! # `prism_window`
//!
//! Prism's window kernel: a deterministic, backend-agnostic model of window
//! configuration, per-window runtime state, display/monitor info, present
//! mode, window-level events, and the command/event ABI that connects it to
//! an OS backend. OS/windowing backends (winit, SDL, native) live in separate
//! crates; they translate platform messages into [`WindowEventEnvelope`]s,
//! drive [`WindowRegistry`] state with [`WindowRegistry::apply`], consume
//! [`WindowCommandEnvelope`]s, and acknowledge them with
//! [`WindowCommandResult`]s.
//!
//! It is a pure, deterministic, `no_std + alloc` type system with no `unsafe`.
//! Geometry conversions avoid `std` float intrinsics (no `round`/`floor`, no
//! `libm`); the [`WindowEvent`] stream and its [`WindowEventEnvelope`] are
//! `Copy` for cheap record/replay. Timestamps are plain nanoseconds
//! ([`MonotonicTimestamp`]) so recordings are byte-stable.
//!
//! ## Layout
//! - Geometry: [`geometry`] (physical/logical sizes, positions, scale).
//! - Resolution: [`resolution`] (size + scale, resize constraints).
//! - Enums: [`mode`] (window/present/level/alpha/theme), [`cursor`].
//! - Displays: [`monitor`] (monitor + video mode).
//! - State: [`window`] ([`Window`], [`WindowAttributes`]), [`registry`].
//! - Stream & ABI: [`event`], [`envelope`], [`command`], [`result`],
//!   [`capabilities`].
#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]

extern crate alloc;

pub mod capabilities;
pub mod command;
pub mod cursor;
pub mod envelope;
pub mod event;
pub mod geometry;
pub mod mode;
pub mod monitor;
pub mod registry;
pub mod resolution;
pub mod result;
pub mod window;

pub use capabilities::{HdrConfig, HdrMode, HdrSupport, WindowCapabilities};
pub use command::{
    AttentionKind, CommandSequence, CommandSequencer, ImeRequest, ResizeDirection, WindowCommand,
    WindowCommandEnvelope, WindowModeRequest, WindowTarget,
};
pub use cursor::{CursorGrabMode, CursorIcon, CursorOptions};
pub use envelope::{
    EventSource, MonotonicTimestamp, PlatformEventSequence, PlatformEventStamp, WindowEventEnvelope,
};
pub use event::WindowEvent;
pub use geometry::{LogicalPosition, LogicalSize, PhysicalPosition, PhysicalSize};
pub use mode::{CompositeAlphaMode, PresentMode, WindowLevel, WindowMode, WindowTheme};
pub use monitor::{Monitor, MonitorId, VideoMode};
pub use registry::WindowRegistry;
pub use resolution::{WindowResizeConstraints, WindowResolution};
pub use result::{
    AdjustmentReason, Capability, CommandStatus, PlatformDenial, RealizedWindowDelta,
    WindowBackendError, WindowCommandResult,
};
pub use window::{Window, WindowAttributes, WindowId};

/// The common types most consumers import.
pub mod prelude {
    pub use crate::capabilities::{HdrConfig, HdrMode, HdrSupport, WindowCapabilities};
    pub use crate::command::{
        AttentionKind, CommandSequence, CommandSequencer, ImeRequest, ResizeDirection,
        WindowCommand, WindowCommandEnvelope, WindowModeRequest, WindowTarget,
    };
    pub use crate::cursor::{CursorGrabMode, CursorIcon, CursorOptions};
    pub use crate::envelope::{
        EventSource, MonotonicTimestamp, PlatformEventSequence, PlatformEventStamp,
        WindowEventEnvelope,
    };
    pub use crate::event::WindowEvent;
    pub use crate::geometry::{LogicalPosition, LogicalSize, PhysicalPosition, PhysicalSize};
    pub use crate::mode::{CompositeAlphaMode, PresentMode, WindowLevel, WindowMode, WindowTheme};
    pub use crate::monitor::{Monitor, MonitorId, VideoMode};
    pub use crate::registry::WindowRegistry;
    pub use crate::resolution::{WindowResizeConstraints, WindowResolution};
    pub use crate::result::{
        AdjustmentReason, Capability, CommandStatus, PlatformDenial, RealizedWindowDelta,
        WindowBackendError, WindowCommandResult,
    };
    pub use crate::window::{Window, WindowAttributes, WindowId};
}

#[cfg(test)]
mod tests;
