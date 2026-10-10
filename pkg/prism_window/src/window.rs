//! Window identity, desired attributes, and live per-window runtime state.

use alloc::string::{String, ToString};

use crate::cursor::CursorOptions;
use crate::event::WindowEvent;
use crate::geometry::{LogicalSize, PhysicalPosition, PhysicalSize};
use crate::mode::{CompositeAlphaMode, PresentMode, WindowLevel, WindowMode};
use crate::resolution::{WindowResizeConstraints, WindowResolution};

/// Stable identifier for a window, assigned by the backend.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct WindowId(pub u64);

/// The *desired* configuration of a window, supplied when creating it.
///
/// This is the request the backend tries to honor; the realized state lives in
/// [`Window`], which may differ (e.g. the OS clamps the size or denies
/// fullscreen).
#[derive(Clone, PartialEq, Debug)]
pub struct WindowAttributes {
    /// Title text shown in the title bar / taskbar.
    pub title: String,
    /// Requested resolution (physical size + scale factor).
    pub resolution: WindowResolution,
    /// Display mode (windowed / fullscreen).
    pub mode: WindowMode,
    /// Swapchain present mode (v-sync behavior).
    pub present_mode: PresentMode,
    /// Whether the user may resize the window.
    pub resizable: bool,
    /// Whether the OS draws a title bar and borders.
    pub decorations: bool,
    /// Whether the window surface has a transparent background.
    pub transparent: bool,
    /// Whether the window is initially visible.
    pub visible: bool,
    /// Bounds applied to user/OS resizes.
    pub resize_constraints: WindowResizeConstraints,
    /// Desktop stacking level.
    pub window_level: WindowLevel,
    /// Surface alpha compositing mode.
    pub composite_alpha_mode: CompositeAlphaMode,
    /// Cursor appearance and confinement.
    pub cursor: CursorOptions,
}

impl Default for WindowAttributes {
    fn default() -> Self {
        Self {
            title: "Prism".to_string(),
            resolution: WindowResolution::default(),
            mode: WindowMode::default(),
            present_mode: PresentMode::default(),
            resizable: true,
            decorations: true,
            transparent: false,
            visible: true,
            resize_constraints: WindowResizeConstraints::default(),
            window_level: WindowLevel::default(),
            composite_alpha_mode: CompositeAlphaMode::default(),
            cursor: CursorOptions::default(),
        }
    }
}

impl WindowAttributes {
    /// Starts from the defaults with a given title.
    #[must_use]
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            ..Self::default()
        }
    }

    /// Sets the requested physical size (builder style).
    #[must_use]
    pub fn with_size(mut self, width: u32, height: u32) -> Self {
        self.resolution
            .set_physical_size(PhysicalSize::new(width, height));
        self
    }

    /// Sets the display mode (builder style).
    #[must_use]
    pub fn with_mode(mut self, mode: WindowMode) -> Self {
        self.mode = mode;
        self
    }

    /// Sets the present mode (builder style).
    #[must_use]
    pub fn with_present_mode(mut self, present_mode: PresentMode) -> Self {
        self.present_mode = present_mode;
        self
    }
}

/// The live, realized state of a single window.
///
/// Backends create one from [`WindowAttributes`] and then drive it with
/// [`apply`](Self::apply) as [`WindowEvent`]s arrive; gameplay/render code
/// reads the realized size, focus, and cursor state from here.
#[derive(Clone, PartialEq, Debug)]
pub struct Window {
    attributes: WindowAttributes,
    position: Option<PhysicalPosition>,
    focused: bool,
    minimized: bool,
    maximized: bool,
    occluded: bool,
    close_requested: bool,
    cursor_inside: bool,
    physical_cursor_position: Option<PhysicalPosition>,
}

impl Window {
    /// Creates a window in the state its attributes request.
    #[must_use]
    pub fn new(attributes: WindowAttributes) -> Self {
        let focused = attributes.visible;
        Self {
            attributes,
            position: None,
            focused,
            minimized: false,
            maximized: false,
            occluded: false,
            close_requested: false,
            cursor_inside: false,
            physical_cursor_position: None,
        }
    }

    /// The desired attributes this window was built from.
    #[must_use]
    pub const fn attributes(&self) -> &WindowAttributes {
        &self.attributes
    }

    /// The current resolution.
    #[must_use]
    pub const fn resolution(&self) -> &WindowResolution {
        &self.attributes.resolution
    }

    /// The realized physical size.
    #[must_use]
    pub const fn physical_size(&self) -> PhysicalSize {
        self.attributes.resolution.physical_size()
    }

    /// The realized logical size.
    #[must_use]
    pub fn logical_size(&self) -> LogicalSize {
        self.attributes.resolution.logical_size()
    }

    /// The current scale factor.
    #[must_use]
    pub const fn scale_factor(&self) -> f32 {
        self.attributes.resolution.scale_factor()
    }

    /// The window title.
    #[must_use]
    pub fn title(&self) -> &str {
        &self.attributes.title
    }

    /// Sets the window title.
    pub fn set_title(&mut self, title: impl Into<String>) {
        self.attributes.title = title.into();
    }

    /// The display mode.
    #[must_use]
    pub const fn mode(&self) -> WindowMode {
        self.attributes.mode
    }

    /// Sets the display mode.
    pub fn set_mode(&mut self, mode: WindowMode) {
        self.attributes.mode = mode;
    }

    /// The present mode.
    #[must_use]
    pub const fn present_mode(&self) -> PresentMode {
        self.attributes.present_mode
    }

    /// Cursor options.
    #[must_use]
    pub const fn cursor(&self) -> &CursorOptions {
        &self.attributes.cursor
    }

    /// Mutable access to cursor options (shape, visibility, grab mode).
    pub fn cursor_mut(&mut self) -> &mut CursorOptions {
        &mut self.attributes.cursor
    }

    /// Whether the window currently holds keyboard focus.
    #[must_use]
    pub const fn focused(&self) -> bool {
        self.focused
    }

    /// Whether the window is minimized/iconified.
    #[must_use]
    pub const fn is_minimized(&self) -> bool {
        self.minimized
    }

    /// Whether the window is maximized.
    #[must_use]
    pub const fn is_maximized(&self) -> bool {
        self.maximized
    }

    /// Whether the window is fully occluded (a hint to pause rendering).
    #[must_use]
    pub const fn is_occluded(&self) -> bool {
        self.occluded
    }

    /// Whether a close has been requested but not yet honored.
    #[must_use]
    pub const fn close_requested(&self) -> bool {
        self.close_requested
    }

    /// The last known cursor position in physical pixels, if the cursor is
    /// inside the window.
    #[must_use]
    pub const fn physical_cursor_position(&self) -> Option<PhysicalPosition> {
        self.physical_cursor_position
    }

    /// Whether the cursor is currently inside the window.
    #[must_use]
    pub const fn cursor_inside(&self) -> bool {
        self.cursor_inside
    }

    /// The window's top-left desktop position in physical pixels, if known.
    #[must_use]
    pub const fn position(&self) -> Option<PhysicalPosition> {
        self.position
    }

    /// Resizes the realized surface, clamping to the resize constraints.
    pub fn resize(&mut self, size: PhysicalSize) {
        let clamped = self.attributes.resize_constraints.clamp(size);
        self.attributes.resolution.set_physical_size(clamped);
    }

    /// Applies one window event, updating realized state, and returns whether
    /// the event changed anything observable.
    pub fn apply(&mut self, event: WindowEvent) -> bool {
        match event {
            WindowEvent::Resized(size) => {
                let clamped = self.attributes.resize_constraints.clamp(size);
                let changed = clamped != self.attributes.resolution.physical_size();
                self.attributes.resolution.set_physical_size(clamped);
                changed
            }
            WindowEvent::ScaleFactorChanged {
                scale_factor_milli,
                new_inner_size,
            } => {
                let scale = scale_factor_milli as f32 / 1000.0;
                self.attributes.resolution.set_scale_factor(scale);
                let clamped = self.attributes.resize_constraints.clamp(new_inner_size);
                self.attributes.resolution.set_physical_size(clamped);
                true
            }
            WindowEvent::Moved(position) => {
                let changed = self.position != Some(position);
                self.position = Some(position);
                changed
            }
            WindowEvent::CloseRequested => {
                let changed = !self.close_requested;
                self.close_requested = true;
                changed
            }
            WindowEvent::Destroyed => true,
            WindowEvent::Focused(focused) => {
                let changed = self.focused != focused;
                self.focused = focused;
                changed
            }
            WindowEvent::CursorMoved { position } => {
                let changed = self.physical_cursor_position != Some(position);
                self.physical_cursor_position = Some(position);
                self.cursor_inside = true;
                changed
            }
            WindowEvent::CursorEntered => {
                let changed = !self.cursor_inside;
                self.cursor_inside = true;
                changed
            }
            WindowEvent::CursorLeft => {
                let changed = self.cursor_inside;
                self.cursor_inside = false;
                self.physical_cursor_position = None;
                changed
            }
            WindowEvent::Occluded(occluded) => {
                let changed = self.occluded != occluded;
                self.occluded = occluded;
                changed
            }
            WindowEvent::Minimized => {
                let changed = !self.minimized;
                self.minimized = true;
                self.maximized = false;
                changed
            }
            WindowEvent::Maximized => {
                let changed = !self.maximized;
                self.maximized = true;
                self.minimized = false;
                changed
            }
            WindowEvent::Restored => {
                let changed = self.minimized || self.maximized;
                self.minimized = false;
                self.maximized = false;
                changed
            }
            WindowEvent::ThemeChanged(_) => false,
        }
    }
}
