//! Unit tests for the window kernel: geometry conversions, resolution and
//! constraint math, monitor mode selection, and window event application.

use alloc::vec;

use crate::cursor::{CursorGrabMode, CursorOptions};
use crate::event::WindowEvent;
use crate::geometry::{LogicalSize, PhysicalPosition, PhysicalSize};
use crate::mode::{PresentMode, WindowMode, WindowTheme};
use crate::monitor::{Monitor, MonitorId, VideoMode};
use crate::resolution::{WindowResizeConstraints, WindowResolution};
use crate::window::{Window, WindowAttributes};

// --- geometry --------------------------------------------------------------

#[test]
fn physical_to_logical_roundtrip() {
    let physical = PhysicalSize::new(2560, 1440);
    let logical = physical.to_logical(2.0);
    assert_eq!(logical, LogicalSize::new(1280.0, 720.0));
    assert_eq!(logical.to_physical(2.0), physical);
}

#[test]
fn to_physical_rounds_half_up() {
    // 1.5 * 1.0 = 1.5 -> rounds to 2; 1.49 -> 1.
    assert_eq!(
        LogicalSize::new(1.5, 1.49).to_physical(1.0),
        PhysicalSize::new(2, 1)
    );
}

#[test]
fn invalid_scale_factor_folds_to_one() {
    let physical = PhysicalSize::new(100, 50);
    assert_eq!(physical.to_logical(0.0), LogicalSize::new(100.0, 50.0));
    assert_eq!(physical.to_logical(f32::NAN), LogicalSize::new(100.0, 50.0));
    assert_eq!(physical.to_logical(-2.0), LogicalSize::new(100.0, 50.0));
}

#[test]
fn to_physical_saturates_and_never_panics() {
    // Non-finite / negative collapse to 0 rather than panicking.
    assert_eq!(
        LogicalSize::new(-5.0, 10.0).to_physical(1.0),
        PhysicalSize::new(0, 10)
    );
    assert_eq!(
        LogicalSize::new(f32::INFINITY, 10.0).to_physical(1.0).width,
        u32::MAX
    );
}

#[test]
fn aspect_ratio_guards_zero_height() {
    assert_eq!(
        PhysicalSize::new(1920, 1080).aspect_ratio(),
        1920.0 / 1080.0
    );
    assert_eq!(PhysicalSize::new(100, 0).aspect_ratio(), 0.0);
}

#[test]
fn is_empty_detects_zero_dimension() {
    assert!(PhysicalSize::new(0, 10).is_empty());
    assert!(PhysicalSize::new(10, 0).is_empty());
    assert!(!PhysicalSize::new(1, 1).is_empty());
}

// --- resolution & constraints ---------------------------------------------

#[test]
fn resolution_logical_size_uses_scale() {
    let res = WindowResolution::new(1920, 1080).with_scale_factor(1.5);
    assert_eq!(res.scale_factor(), 1.5);
    assert_eq!(res.logical_size(), LogicalSize::new(1280.0, 720.0));
}

#[test]
fn constraints_clamp_both_dimensions() {
    let c = WindowResizeConstraints::new(640, 480, 1920, 1080);
    assert_eq!(
        c.clamp(PhysicalSize::new(100, 100)),
        PhysicalSize::new(640, 480)
    );
    assert_eq!(
        c.clamp(PhysicalSize::new(4000, 4000)),
        PhysicalSize::new(1920, 1080)
    );
    assert_eq!(
        c.clamp(PhysicalSize::new(800, 600)),
        PhysicalSize::new(800, 600)
    );
}

#[test]
fn constraints_new_normalizes_inverted_bounds() {
    // max below min is lifted to min; min of 0 is lifted to 1.
    let c = WindowResizeConstraints::new(0, 0, 10, 10);
    assert_eq!(c.min_width, 1);
    assert_eq!(c.min_height, 1);
    let c2 = WindowResizeConstraints::new(800, 600, 100, 100);
    assert_eq!(c2.max_width, 800);
    assert_eq!(c2.max_height, 600);
}

// --- monitor ---------------------------------------------------------------

#[test]
fn best_video_mode_prefers_exact_then_refresh() {
    let monitor = Monitor {
        id: MonitorId(1),
        name: None,
        position: PhysicalPosition::new(0, 0),
        size: PhysicalSize::new(1920, 1080),
        scale_factor: 1.0,
        refresh_rate_millihertz: Some(60_000),
        video_modes: vec![
            VideoMode {
                size: PhysicalSize::new(1280, 720),
                bit_depth: 32,
                refresh_rate_millihertz: 60_000,
            },
            VideoMode {
                size: PhysicalSize::new(1920, 1080),
                bit_depth: 32,
                refresh_rate_millihertz: 60_000,
            },
            VideoMode {
                size: PhysicalSize::new(1920, 1080),
                bit_depth: 32,
                refresh_rate_millihertz: 144_000,
            },
        ],
    };
    let best = monitor
        .best_video_mode(PhysicalSize::new(1920, 1080))
        .expect("has modes");
    assert_eq!(best.size, PhysicalSize::new(1920, 1080));
    assert_eq!(best.refresh_rate_millihertz, 144_000);
    assert_eq!(best.refresh_rate_hz(), 144.0);
}

#[test]
fn best_video_mode_none_when_empty() {
    let monitor = Monitor {
        id: MonitorId(0),
        name: None,
        position: PhysicalPosition::default(),
        size: PhysicalSize::new(800, 600),
        scale_factor: 1.0,
        refresh_rate_millihertz: None,
        video_modes: vec![],
    };
    assert!(monitor
        .best_video_mode(PhysicalSize::new(800, 600))
        .is_none());
}

// --- window state & events -------------------------------------------------

#[test]
fn window_builds_from_attributes() {
    let attrs = WindowAttributes::new("Prism Test")
        .with_size(1600, 900)
        .with_mode(WindowMode::BorderlessFullscreen)
        .with_present_mode(PresentMode::Immediate);
    let window = Window::new(attrs);
    assert_eq!(window.title(), "Prism Test");
    assert_eq!(window.physical_size(), PhysicalSize::new(1600, 900));
    assert_eq!(window.mode(), WindowMode::BorderlessFullscreen);
    assert_eq!(window.present_mode(), PresentMode::Immediate);
    assert!(window.mode().is_fullscreen());
}

#[test]
fn resized_event_clamps_and_reports_change() {
    let attrs = WindowAttributes::default();
    let mut window = Window::new(attrs);
    assert!(window.apply(WindowEvent::Resized(PhysicalSize::new(1920, 1080))));
    assert_eq!(window.physical_size(), PhysicalSize::new(1920, 1080));
    // Re-applying the same size is a no-op (no change reported).
    assert!(!window.apply(WindowEvent::Resized(PhysicalSize::new(1920, 1080))));
}

#[test]
fn scale_factor_changed_updates_scale_and_size() {
    let mut window = Window::new(WindowAttributes::default());
    assert!(window.apply(WindowEvent::ScaleFactorChanged {
        scale_factor_milli: 2_000,
        new_inner_size: PhysicalSize::new(2560, 1440),
    }));
    assert_eq!(window.scale_factor(), 2.0);
    assert_eq!(window.physical_size(), PhysicalSize::new(2560, 1440));
    assert_eq!(window.logical_size(), LogicalSize::new(1280.0, 720.0));
}

#[test]
fn focus_minimize_maximize_restore_cycle() {
    let mut window = Window::new(WindowAttributes::default());
    window.apply(WindowEvent::Focused(false));
    assert!(!window.focused());
    assert!(window.apply(WindowEvent::Minimized));
    assert!(window.is_minimized());
    assert!(window.apply(WindowEvent::Maximized));
    assert!(window.is_maximized());
    assert!(!window.is_minimized());
    assert!(window.apply(WindowEvent::Restored));
    assert!(!window.is_maximized() && !window.is_minimized());
}

#[test]
fn cursor_enter_move_leave() {
    let mut window = Window::new(WindowAttributes::default());
    window.apply(WindowEvent::CursorEntered);
    assert!(window.cursor_inside());
    window.apply(WindowEvent::CursorMoved {
        position: PhysicalPosition::new(10, 20),
    });
    assert_eq!(
        window.physical_cursor_position(),
        Some(PhysicalPosition::new(10, 20))
    );
    window.apply(WindowEvent::CursorLeft);
    assert!(!window.cursor_inside());
    assert_eq!(window.physical_cursor_position(), None);
}

#[test]
fn close_requested_latches() {
    let mut window = Window::new(WindowAttributes::default());
    assert!(!window.close_requested());
    assert!(window.apply(WindowEvent::CloseRequested));
    assert!(window.close_requested());
    // Already latched: no further change.
    assert!(!window.apply(WindowEvent::CloseRequested));
}

#[test]
fn resize_clamps_to_constraints() {
    let attrs = WindowAttributes {
        resize_constraints: WindowResizeConstraints::new(640, 480, 1920, 1080),
        ..WindowAttributes::default()
    };
    let mut window = Window::new(attrs);
    window.resize(PhysicalSize::new(100, 100));
    assert_eq!(window.physical_size(), PhysicalSize::new(640, 480));
}

#[test]
fn cursor_mut_edits_options() {
    let mut window = Window::new(WindowAttributes::default());
    assert_eq!(window.cursor(), &CursorOptions::default());
    window.cursor_mut().grab_mode = CursorGrabMode::Locked;
    window.cursor_mut().visible = false;
    assert_eq!(window.cursor().grab_mode, CursorGrabMode::Locked);
    assert!(!window.cursor().visible);
}

#[test]
fn theme_change_is_tracked_as_no_state_change() {
    let mut window = Window::new(WindowAttributes::default());
    // Theme is forwarded to app-level state, not stored on Window.
    assert!(!window.apply(WindowEvent::ThemeChanged(WindowTheme::Dark)));
}
