//! The unified window-level event stream.
//!
//! These are *window* events (resize, focus, close, scale change, cursor
//! enter/leave) as opposed to *input device* events, which live in the
//! `prism_input` crate. Backends translate OS/windowing messages into these;
//! higher layers apply them to [`Window`](crate::Window) state.

use crate::geometry::{PhysicalPosition, PhysicalSize};
use crate::mode::WindowTheme;

/// A single window-level event for one window.
///
/// Kept [`Copy`] so it can be recorded, replayed, or batched cheaply; the
/// window id is carried alongside by the backend, not inside the event.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum WindowEvent {
    /// The drawable (inner) area changed to this physical size.
    Resized(PhysicalSize),
    /// The display scale factor changed; the backend also supplies the new
    /// inner size, since many platforms resize and rescale together.
    ScaleFactorChanged {
        /// The new scale factor (physical per logical).
        scale_factor_milli: u32,
        /// The new inner size in physical pixels.
        new_inner_size: PhysicalSize,
    },
    /// The window's top-left moved to this physical desktop position.
    Moved(PhysicalPosition),
    /// The user requested the window be closed (e.g. clicked the close button).
    CloseRequested,
    /// The window was destroyed by the OS.
    Destroyed,
    /// The window gained (`true`) or lost (`false`) keyboard focus.
    Focused(bool),
    /// The cursor moved to this physical position within the window.
    CursorMoved {
        /// Cursor position in physical pixels, relative to the window.
        position: PhysicalPosition,
    },
    /// The cursor entered the window.
    CursorEntered,
    /// The cursor left the window.
    CursorLeft,
    /// The window became fully hidden (`true`) or visible again (`false`),
    /// e.g. covered by another window; a hint to pause rendering.
    Occluded(bool),
    /// The system color theme changed.
    ThemeChanged(WindowTheme),
    /// The window was minimized/iconified.
    Minimized,
    /// The window was maximized.
    Maximized,
    /// The window returned to its normal state from minimized/maximized.
    Restored,
}
