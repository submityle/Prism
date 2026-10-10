//! Translation of winit platform messages into the kernel's `Copy`
//! [`WindowEvent`] vocabulary.
//!
//! The functions here are deliberately small and pure so each mapping decision
//! is unit-tested in isolation. Several winit variants carry handles that are
//! not publicly constructible (`DeviceId`, `InnerSizeWriter`); those are read
//! through narrow helpers so the bulk of the logic stays testable from plain
//! values.
//!
//! Input-device events (keyboard, mouse buttons, wheel, touch) are *not*
//! translated here: they belong to `prism_input`. This module only emits
//! *window-level* events, matching the kernel's split.

use prism_window::event::WindowEvent;
use prism_window::geometry::PhysicalSize;
use prism_window::mode::WindowTheme;

use crate::convert;

/// Maps a winit theme to the kernel theme.
#[must_use]
pub fn map_theme(theme: winit::window::Theme) -> WindowTheme {
    match theme {
        winit::window::Theme::Light => WindowTheme::Light,
        winit::window::Theme::Dark => WindowTheme::Dark,
    }
}

/// Builds a [`WindowEvent::ScaleFactorChanged`] from the raw scale factor and
/// the window's resolved new inner size. winit 0.30 reports the scale change
/// without the new size inline (it hands back an `InnerSizeWriter`), so the
/// caller queries the size and passes it here.
#[must_use]
pub fn map_scale_factor_changed(scale_factor: f64, new_inner_size: PhysicalSize) -> WindowEvent {
    WindowEvent::ScaleFactorChanged {
        scale_factor_milli: convert::scale_factor_to_milli(scale_factor),
        new_inner_size,
    }
}

/// Translates a window-level winit event into the kernel event, or `None` when
/// the event is either input-device traffic (owned by `prism_input`), a
/// no-op for the window model, or a case the caller must handle with extra
/// context ([`winit::event::WindowEvent::ScaleFactorChanged`], which needs the
/// freshly-resolved inner size: use [`map_scale_factor_changed`]).
#[must_use]
pub fn translate(event: &winit::event::WindowEvent) -> Option<WindowEvent> {
    use winit::event::WindowEvent as W;
    match event {
        W::Resized(size) => Some(WindowEvent::Resized(convert::size_from_winit(*size))),
        W::Moved(pos) => Some(WindowEvent::Moved(convert::position_from_winit(*pos))),
        W::CloseRequested => Some(WindowEvent::CloseRequested),
        W::Destroyed => Some(WindowEvent::Destroyed),
        W::Focused(focused) => Some(WindowEvent::Focused(*focused)),
        W::Occluded(occluded) => Some(WindowEvent::Occluded(*occluded)),
        W::ThemeChanged(theme) => Some(WindowEvent::ThemeChanged(map_theme(*theme))),
        W::CursorMoved { position, .. } => Some(WindowEvent::CursorMoved {
            position: convert::cursor_from_winit(*position),
        }),
        W::CursorEntered { .. } => Some(WindowEvent::CursorEntered),
        W::CursorLeft { .. } => Some(WindowEvent::CursorLeft),

        // Everything below maps to `None`, for three distinct reasons:
        // * `ScaleFactorChanged` needs the resolved inner size and is handled
        //   by the runner via `map_scale_factor_changed`.
        // * Keyboard/mouse/touch/gesture/IME traffic is input-device data owned
        //   by `prism_input`, routed with `is_input_device_event`.
        // * File drag/drop, activation tokens, and redraw requests are not yet
        //   part of the kernel window-event vocabulary; dropped until modelled.
        W::ScaleFactorChanged { .. }
        | W::KeyboardInput { .. }
        | W::ModifiersChanged(_)
        | W::Ime(_)
        | W::MouseWheel { .. }
        | W::MouseInput { .. }
        | W::Touch(_)
        | W::TouchpadPressure { .. }
        | W::PinchGesture { .. }
        | W::PanGesture { .. }
        | W::DoubleTapGesture { .. }
        | W::RotationGesture { .. }
        | W::AxisMotion { .. }
        | W::DroppedFile(_)
        | W::HoveredFile(_)
        | W::HoveredFileCancelled
        | W::ActivationTokenDone { .. }
        | W::RedrawRequested => None,
    }
}

/// Whether a winit event is input-device traffic that `prism_input` owns.
/// Backends can use this to route a single winit stream to both subsystems
/// without re-matching the whole enum.
#[must_use]
pub fn is_input_device_event(event: &winit::event::WindowEvent) -> bool {
    use winit::event::WindowEvent as W;
    matches!(
        event,
        W::KeyboardInput { .. }
            | W::ModifiersChanged(_)
            | W::Ime(_)
            | W::MouseWheel { .. }
            | W::MouseInput { .. }
            | W::Touch(_)
            | W::TouchpadPressure { .. }
            | W::PinchGesture { .. }
            | W::PanGesture { .. }
            | W::DoubleTapGesture { .. }
            | W::RotationGesture { .. }
            | W::AxisMotion { .. }
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_window::geometry::{PhysicalPosition, PhysicalSize};

    #[test]
    fn translates_resized() {
        let ev = winit::event::WindowEvent::Resized(winit::dpi::PhysicalSize::new(800, 600));
        assert_eq!(
            translate(&ev),
            Some(WindowEvent::Resized(PhysicalSize::new(800, 600)))
        );
    }

    #[test]
    fn translates_moved_with_negative_coords() {
        let ev = winit::event::WindowEvent::Moved(winit::dpi::PhysicalPosition::new(-5, 7));
        assert_eq!(
            translate(&ev),
            Some(WindowEvent::Moved(PhysicalPosition::new(-5, 7)))
        );
    }

    #[test]
    fn translates_focus_close_destroy_occluded() {
        assert_eq!(
            translate(&winit::event::WindowEvent::Focused(true)),
            Some(WindowEvent::Focused(true))
        );
        assert_eq!(
            translate(&winit::event::WindowEvent::CloseRequested),
            Some(WindowEvent::CloseRequested)
        );
        assert_eq!(
            translate(&winit::event::WindowEvent::Destroyed),
            Some(WindowEvent::Destroyed)
        );
        assert_eq!(
            translate(&winit::event::WindowEvent::Occluded(true)),
            Some(WindowEvent::Occluded(true))
        );
    }

    #[test]
    fn translates_theme() {
        assert_eq!(
            translate(&winit::event::WindowEvent::ThemeChanged(
                winit::window::Theme::Dark
            )),
            Some(WindowEvent::ThemeChanged(WindowTheme::Dark))
        );
        assert_eq!(map_theme(winit::window::Theme::Light), WindowTheme::Light);
    }

    #[test]
    fn scale_factor_change_is_deferred_to_runner() {
        // The pure enum translation cannot produce the new inner size, so it
        // must defer; the dedicated helper builds the event.
        let built = map_scale_factor_changed(2.0, PhysicalSize::new(1024, 768));
        assert_eq!(
            built,
            WindowEvent::ScaleFactorChanged {
                scale_factor_milli: 2000,
                new_inner_size: PhysicalSize::new(1024, 768),
            }
        );
    }

    #[test]
    fn redraw_and_dropped_file_are_not_window_model_events() {
        assert_eq!(translate(&winit::event::WindowEvent::RedrawRequested), None);
        assert_eq!(
            translate(&winit::event::WindowEvent::HoveredFileCancelled),
            None
        );
    }
}
