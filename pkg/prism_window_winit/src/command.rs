//! Applying kernel [`WindowCommand`]s to live winit windows, plus the pure
//! enum/attribute translations that back them.
//!
//! The enum mappers are pure and unit-tested. [`apply`] drives a live
//! `winit::window::Window`; it is exercised at runtime by the event loop
//! rather than in unit tests (a real window needs a display connection).

use prism_window::command::WindowCommand;
use prism_window::cursor::{CursorGrabMode, CursorIcon, CursorOptions};
use prism_window::mode::{WindowLevel, WindowMode, WindowTheme};
use prism_window::command::{AttentionKind, ResizeDirection};

use crate::convert;
use crate::sync::WindowConfig;

/// Maps a kernel window level to winit's.
#[must_use]
pub fn to_winit_level(level: WindowLevel) -> winit::window::WindowLevel {
    match level {
        WindowLevel::AlwaysOnBottom => winit::window::WindowLevel::AlwaysOnBottom,
        WindowLevel::Normal => winit::window::WindowLevel::Normal,
        WindowLevel::AlwaysOnTop => winit::window::WindowLevel::AlwaysOnTop,
    }
}

/// Maps a kernel theme to winit's.
#[must_use]
pub fn to_winit_theme(theme: WindowTheme) -> winit::window::Theme {
    match theme {
        WindowTheme::Light => winit::window::Theme::Light,
        WindowTheme::Dark => winit::window::Theme::Dark,
    }
}

/// Maps a kernel resize-drag direction to winit's.
#[must_use]
pub fn to_winit_resize_direction(dir: ResizeDirection) -> winit::window::ResizeDirection {
    use winit::window::ResizeDirection as R;
    match dir {
        ResizeDirection::East => R::East,
        ResizeDirection::North => R::North,
        ResizeDirection::NorthEast => R::NorthEast,
        ResizeDirection::NorthWest => R::NorthWest,
        ResizeDirection::South => R::South,
        ResizeDirection::SouthEast => R::SouthEast,
        ResizeDirection::SouthWest => R::SouthWest,
        ResizeDirection::West => R::West,
    }
}

/// Maps a kernel attention kind to winit's user-attention type.
#[must_use]
pub fn to_winit_attention(kind: AttentionKind) -> winit::window::UserAttentionType {
    match kind {
        AttentionKind::Informational => winit::window::UserAttentionType::Informational,
        AttentionKind::Critical => winit::window::UserAttentionType::Critical,
    }
}

/// Maps a kernel cursor grab mode to winit's.
#[must_use]
pub fn to_winit_grab(mode: CursorGrabMode) -> winit::window::CursorGrabMode {
    match mode {
        CursorGrabMode::None => winit::window::CursorGrabMode::None,
        CursorGrabMode::Confined => winit::window::CursorGrabMode::Confined,
        CursorGrabMode::Locked => winit::window::CursorGrabMode::Locked,
    }
}

/// Maps a kernel cursor shape to winit's (names mirror the CSS cursor set).
#[must_use]
pub fn to_winit_cursor_icon(icon: CursorIcon) -> winit::window::CursorIcon {
    use winit::window::CursorIcon as C;
    match icon {
        CursorIcon::Default => C::Default,
        CursorIcon::ContextMenu => C::ContextMenu,
        CursorIcon::Help => C::Help,
        CursorIcon::Pointer => C::Pointer,
        CursorIcon::Progress => C::Progress,
        CursorIcon::Wait => C::Wait,
        CursorIcon::Cell => C::Cell,
        CursorIcon::Crosshair => C::Crosshair,
        CursorIcon::Text => C::Text,
        CursorIcon::VerticalText => C::VerticalText,
        CursorIcon::Alias => C::Alias,
        CursorIcon::Copy => C::Copy,
        CursorIcon::Move => C::Move,
        CursorIcon::NoDrop => C::NoDrop,
        CursorIcon::NotAllowed => C::NotAllowed,
        CursorIcon::Grab => C::Grab,
        CursorIcon::Grabbing => C::Grabbing,
        CursorIcon::EResize => C::EResize,
        CursorIcon::NResize => C::NResize,
        CursorIcon::NeResize => C::NeResize,
        CursorIcon::NwResize => C::NwResize,
        CursorIcon::SResize => C::SResize,
        CursorIcon::SeResize => C::SeResize,
        CursorIcon::SwResize => C::SwResize,
        CursorIcon::WResize => C::WResize,
        CursorIcon::EwResize => C::EwResize,
        CursorIcon::NsResize => C::NsResize,
        CursorIcon::NeswResize => C::NeswResize,
        CursorIcon::NwseResize => C::NwseResize,
        CursorIcon::ColResize => C::ColResize,
        CursorIcon::RowResize => C::RowResize,
        CursorIcon::AllScroll => C::AllScroll,
        CursorIcon::ZoomIn => C::ZoomIn,
        CursorIcon::ZoomOut => C::ZoomOut,
    }
}

/// Maps a kernel window mode to a winit fullscreen request.
///
/// Exclusive modes ([`WindowMode::Fullscreen`], [`WindowMode::SizedFullscreen`])
/// currently fall back to borderless fullscreen on the current monitor until
/// video-mode enumeration (§12.6) is wired; borderless avoids a mode switch and
/// never leaves the user on a black screen if enumeration fails.
#[must_use]
pub fn to_winit_fullscreen(mode: WindowMode) -> Option<winit::window::Fullscreen> {
    match mode {
        WindowMode::Windowed => None,
        WindowMode::BorderlessFullscreen
        | WindowMode::Fullscreen
        | WindowMode::SizedFullscreen => Some(winit::window::Fullscreen::Borderless(None)),
    }
}

/// Builds the winit creation attributes from a desired [`WindowConfig`].
#[must_use]
pub fn build_attributes(config: &WindowConfig) -> winit::window::WindowAttributes {
    let mut attrs = winit::window::WindowAttributes::default()
        .with_title(config.title.clone())
        .with_inner_size(convert::size_to_winit(config.inner_size))
        .with_resizable(config.resizable)
        .with_decorations(config.decorations)
        .with_visible(config.visible)
        .with_window_level(to_winit_level(config.level))
        .with_fullscreen(to_winit_fullscreen(config.mode));

    if let Some(theme) = config.theme {
        attrs = attrs.with_theme(Some(to_winit_theme(theme)));
    }
    if let Some(pos) = config.outer_position {
        attrs = attrs.with_position(convert::position_to_winit(pos));
    }
    attrs
}

/// Applies cursor options to a live window (shape, visibility, grab). Grab
/// failures are swallowed: not every platform supports confine/lock, and a
/// missing grab must not abort command application.
pub fn apply_cursor(window: &winit::window::Window, cursor: CursorOptions) {
    window.set_cursor_visible(cursor.visible);
    window.set_cursor(winit::window::Cursor::Icon(to_winit_cursor_icon(cursor.icon)));
    let _ = window.set_cursor_grab(to_winit_grab(cursor.grab_mode));
    let _ = window.set_cursor_hittest(cursor.hit_test);
}

/// Applies a single [`WindowCommand`] to a live window. Commands that create or
/// destroy windows ([`WindowCommand::Create`], [`WindowCommand::Destroy`]) are
/// handled by the runner (they need the event loop), so they are ignored here.
pub fn apply(window: &winit::window::Window, command: &WindowCommand) {
    match command {
        WindowCommand::SetTitle(title) => window.set_title(title),
        WindowCommand::SetVisible(v) => window.set_visible(*v),
        WindowCommand::SetOuterPosition(pos) => {
            window.set_outer_position(convert::position_to_winit(*pos));
        }
        WindowCommand::RequestInnerSize(size) => {
            let _ = window.request_inner_size(convert::size_to_winit(*size));
        }
        WindowCommand::SetMinInnerSize(size) => {
            window.set_min_inner_size(size.map(convert::size_to_winit));
        }
        WindowCommand::SetMaxInnerSize(size) => {
            window.set_max_inner_size(size.map(convert::size_to_winit));
        }
        WindowCommand::SetResizable(v) => window.set_resizable(*v),
        WindowCommand::SetDecorations(v) => window.set_decorations(*v),
        WindowCommand::SetLevel(level) => window.set_window_level(to_winit_level(*level)),
        WindowCommand::SetMode(req) => window.set_fullscreen(to_winit_fullscreen(req.mode)),
        WindowCommand::SetTheme(theme) => window.set_theme(theme.map(to_winit_theme)),
        WindowCommand::SetCursor(cursor) => apply_cursor(window, *cursor),
        WindowCommand::RequestFocus => window.focus_window(),
        WindowCommand::RequestAttention(kind) => {
            window.request_user_attention(kind.map(to_winit_attention));
        }
        WindowCommand::RequestRedraw => window.request_redraw(),
        WindowCommand::BeginDragWindow => {
            let _ = window.drag_window();
        }
        WindowCommand::BeginResizeDrag(dir) => {
            let _ = window.drag_resize_window(to_winit_resize_direction(*dir));
        }
        WindowCommand::SetContentProtected(v) => window.set_content_protected(*v),

        // Present mode / HDR / IME are surface- and compositor-level concerns
        // handled by the renderer and IME subsystems, not winit's Window API.
        // Create/Destroy are the runner's responsibility.
        WindowCommand::SetPresentMode(_)
        | WindowCommand::SetHdr(_)
        | WindowCommand::SetIme(_)
        | WindowCommand::Create(_)
        | WindowCommand::Destroy => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_window::geometry::PhysicalSize;

    #[test]
    fn level_mapping_is_total() {
        assert_eq!(
            to_winit_level(WindowLevel::AlwaysOnTop),
            winit::window::WindowLevel::AlwaysOnTop
        );
        assert_eq!(
            to_winit_level(WindowLevel::AlwaysOnBottom),
            winit::window::WindowLevel::AlwaysOnBottom
        );
    }

    #[test]
    fn grab_mapping() {
        assert_eq!(
            to_winit_grab(CursorGrabMode::Locked),
            winit::window::CursorGrabMode::Locked
        );
    }

    #[test]
    fn windowed_has_no_fullscreen_and_fullscreen_is_borderless() {
        assert!(to_winit_fullscreen(WindowMode::Windowed).is_none());
        assert!(matches!(
            to_winit_fullscreen(WindowMode::Fullscreen),
            Some(winit::window::Fullscreen::Borderless(None))
        ));
    }

    #[test]
    fn cursor_icon_maps_representative_shapes() {
        assert_eq!(
            to_winit_cursor_icon(CursorIcon::Pointer),
            winit::window::CursorIcon::Pointer
        );
        assert_eq!(
            to_winit_cursor_icon(CursorIcon::NwseResize),
            winit::window::CursorIcon::NwseResize
        );
    }

    #[test]
    fn build_attributes_does_not_panic() {
        // Smoke test: attribute construction must be total for a default config.
        let cfg = WindowConfig::new("Prism", PhysicalSize::new(800, 600));
        let _ = build_attributes(&cfg);
    }
}
