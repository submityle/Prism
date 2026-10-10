//! # `prism_window_winit`
//!
//! The winit 0.30 platform backend for [`prism_window`]. The deterministic
//! window *model* (geometry, events, commands, three-state sync policy) lives
//! in `prism_window` and is `no_std + alloc`, zero-dependency, and `unsafe`-free.
//! This crate owns the opposite half: the `std`-only, platform-specific dirty
//! work of driving the real OS event loop on the main thread.
//!
//! ## Responsibilities
//! - Own the winit [`EventLoop`](winit::event_loop::EventLoop) and the live
//!   platform windows ([`runner`]).
//! - Translate platform messages into the kernel's `Copy`
//!   [`WindowEventEnvelope`](prism_window::WindowEventEnvelope) stream
//!   ([`translate`]), keeping input-device traffic for `prism_input`.
//! - Apply [`WindowCommand`](prism_window::WindowCommand)s through the
//!   Desired/Applied/Realized model so OS feedback never starts a resend loop
//!   ([`sync`], [`command`]).
//! - Map stable kernel [`WindowId`](prism_window::WindowId)s to opaque
//!   [`winit::window::WindowId`]s ([`id`]).
//! - Coalesce high-frequency events (resize/move/cursor/scale) once per frame
//!   ([`batch`]).
//! - Convert geometry at exactly one boundary ([`convert`]).
//!
//! The pure modules ([`convert`], [`id`], [`translate`], [`sync`], [`batch`],
//! [`command`]) are fully unit-tested without a display server; only the real
//! event loop in [`runner`] needs a windowing system to execute.

extern crate alloc;

pub mod batch;
pub mod command;
pub mod convert;
pub mod error;
pub mod id;
pub mod runner;
pub mod sync;
pub mod translate;

pub use error::{BackendError, BackendResult};
pub use id::WindowIdMap;
pub use runner::{FrameOutcome, WinitBackend, WinitRunner};
pub use sync::{RealizedState, WindowConfig, WindowSync};

/// The types most backend consumers import.
pub mod prelude {
    pub use crate::error::{BackendError, BackendResult};
    pub use crate::id::WindowIdMap;
    pub use crate::runner::{FrameOutcome, WinitBackend, WinitRunner};
    pub use crate::sync::{RealizedState, WindowConfig, WindowSync};
    pub use prism_window::prelude::*;
}
