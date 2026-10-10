//! The three-state Desired / Applied / Realized synchronization model (§9).
//!
//! * **Desired** is what the engine wants the window to be. The engine mutates
//!   it freely each frame.
//! * **Applied** is the last state we actually *commanded* the OS toward. It is
//!   advanced only when we emit the corresponding command, so a field is never
//!   commanded twice for the same value.
//! * **Realized** is the ground truth the OS reports back through events.
//!
//! [`WindowSync::diff`] turns `Desired != Applied` into the minimal command
//! list; the caller sends those commands and calls [`WindowSync::mark_applied`].
//! OS feedback flows through [`WindowSync::observe`], which only touches
//! *Realized*. Because feedback never rewinds *Applied*, a size the OS clamps
//! (§9.4) cannot start a resend loop (§9.3): we asked once, we do not nag.

use prism_window::command::{WindowCommand, WindowModeRequest};
use prism_window::cursor::CursorOptions;
use prism_window::geometry::{PhysicalPosition, PhysicalSize};
use prism_window::mode::{PresentMode, WindowLevel, WindowMode, WindowTheme};

/// The diffable, runtime-mutable surface of a window's configuration.
///
/// Only fields that can meaningfully change *after* creation live here; one-
/// shot creation-only attributes (transparency, alpha mode) are set at build
/// time and are intentionally excluded.
#[derive(Clone, Debug, PartialEq)]
pub struct WindowConfig {
    /// Title bar / taskbar text.
    pub title: String,
    /// Requested inner (drawable) size in physical pixels.
    pub inner_size: PhysicalSize,
    /// Requested top-left desktop position (`None` = leave to the OS).
    pub outer_position: Option<PhysicalPosition>,
    /// Whether the window is shown.
    pub visible: bool,
    /// Whether the user may resize.
    pub resizable: bool,
    /// Whether OS decorations are drawn.
    pub decorations: bool,
    /// Desktop stacking level.
    pub level: WindowLevel,
    /// Windowed / fullscreen request.
    pub mode: WindowMode,
    /// Forced theme, or `None` to follow the system.
    pub theme: Option<WindowTheme>,
    /// Swapchain present mode.
    pub present_mode: PresentMode,
    /// Cursor appearance, visibility, and grab.
    pub cursor: CursorOptions,
}

impl WindowConfig {
    /// A reasonable default used before the engine expresses any intent.
    #[must_use]
    pub fn new(title: impl Into<String>, inner_size: PhysicalSize) -> Self {
        Self {
            title: title.into(),
            inner_size,
            outer_position: None,
            visible: true,
            resizable: true,
            decorations: true,
            level: WindowLevel::Normal,
            mode: WindowMode::Windowed,
            theme: None,
            present_mode: PresentMode::AutoVsync,
            cursor: CursorOptions::default(),
        }
    }
}

/// What the OS has actually reported for a window.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RealizedState {
    /// Last OS-reported inner size.
    pub inner_size: Option<PhysicalSize>,
    /// Last OS-reported outer position.
    pub outer_position: Option<PhysicalPosition>,
    /// Last OS-reported scale factor in milli-units.
    pub scale_factor_milli: Option<u32>,
    /// Whether the window currently holds focus.
    pub focused: bool,
    /// Whether the window is currently occluded (hidden by other surfaces).
    pub occluded: bool,
}

/// Per-window three-state synchronizer.
#[derive(Clone, Debug)]
pub struct WindowSync {
    desired: WindowConfig,
    applied: WindowConfig,
    realized: RealizedState,
}

impl WindowSync {
    /// Starts a synchronizer whose `desired` and `applied` both equal `config`
    /// (i.e. the window was just created from `config`, so nothing is pending).
    #[must_use]
    pub fn new(config: WindowConfig) -> Self {
        Self {
            desired: config.clone(),
            applied: config,
            realized: RealizedState::default(),
        }
    }

    /// Read-only view of the desired configuration.
    #[must_use]
    pub fn desired(&self) -> &WindowConfig {
        &self.desired
    }

    /// Read-only view of the last-commanded configuration.
    #[must_use]
    pub fn applied(&self) -> &WindowConfig {
        &self.applied
    }

    /// Read-only view of the OS-reported state.
    #[must_use]
    pub fn realized(&self) -> &RealizedState {
        &self.realized
    }

    /// Mutable access to the desired configuration for the engine to edit.
    #[must_use]
    pub fn desired_mut(&mut self) -> &mut WindowConfig {
        &mut self.desired
    }

    /// Computes the minimal command list to move the OS from `applied` toward
    /// `desired`. The result is empty when the window is already in sync.
    ///
    /// This does not mutate anything: the caller sends the commands and then
    /// calls [`Self::mark_applied`] so the same value is never commanded twice.
    #[must_use]
    pub fn diff(&self) -> Vec<WindowCommand> {
        let d = &self.desired;
        let a = &self.applied;
        let mut cmds = Vec::new();

        if d.title != a.title {
            cmds.push(WindowCommand::set_title(&d.title));
        }
        if d.inner_size != a.inner_size {
            cmds.push(WindowCommand::RequestInnerSize(d.inner_size));
        }
        if d.outer_position != a.outer_position
            && let Some(pos) = d.outer_position
        {
            cmds.push(WindowCommand::SetOuterPosition(pos));
        }
        if d.visible != a.visible {
            cmds.push(WindowCommand::SetVisible(d.visible));
        }
        if d.resizable != a.resizable {
            cmds.push(WindowCommand::SetResizable(d.resizable));
        }
        if d.decorations != a.decorations {
            cmds.push(WindowCommand::SetDecorations(d.decorations));
        }
        if d.level != a.level {
            cmds.push(WindowCommand::SetLevel(d.level));
        }
        if d.mode != a.mode {
            cmds.push(WindowCommand::SetMode(WindowModeRequest {
                mode: d.mode,
                monitor: None,
                video_mode: None,
            }));
        }
        if d.theme != a.theme {
            cmds.push(WindowCommand::SetTheme(d.theme));
        }
        if d.present_mode != a.present_mode {
            cmds.push(WindowCommand::SetPresentMode(d.present_mode));
        }
        if d.cursor != a.cursor {
            cmds.push(WindowCommand::SetCursor(d.cursor));
        }

        cmds
    }

    /// Whether `desired` and `applied` differ (i.e. [`Self::diff`] is non-empty).
    #[must_use]
    pub fn is_dirty(&self) -> bool {
        self.desired != self.applied
    }

    /// Snapshots `desired` into `applied` after the diff commands were emitted.
    /// From here, each field is considered "already asked for" and will not be
    /// re-commanded until `desired` changes again — the core of §9.3 feedback
    /// suppression.
    pub fn mark_applied(&mut self) {
        self.applied = self.desired.clone();
    }

    /// Folds an OS-reported window event into `realized` only. `applied` is
    /// deliberately left untouched so clamped/adjusted sizes (§9.4) never cause
    /// a resend. Returns `true` if the realized state changed.
    pub fn observe(&mut self, event: &prism_window::event::WindowEvent) -> bool {
        use prism_window::event::WindowEvent as E;
        let before = self.realized;
        match event {
            E::Resized(size) => self.realized.inner_size = Some(*size),
            E::Moved(pos) => self.realized.outer_position = Some(*pos),
            E::ScaleFactorChanged {
                scale_factor_milli,
                new_inner_size,
            } => {
                self.realized.scale_factor_milli = Some(*scale_factor_milli);
                self.realized.inner_size = Some(*new_inner_size);
            }
            E::Focused(f) => self.realized.focused = *f,
            E::Occluded(o) => self.realized.occluded = *o,
            _ => {}
        }
        self.realized != before
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_window::event::WindowEvent;

    fn cfg() -> WindowConfig {
        WindowConfig::new("Prism", PhysicalSize::new(1280, 720))
    }

    #[test]
    fn fresh_sync_is_clean() {
        let s = WindowSync::new(cfg());
        assert!(!s.is_dirty());
        assert!(s.diff().is_empty());
    }

    #[test]
    fn single_field_change_produces_single_command() {
        let mut s = WindowSync::new(cfg());
        s.desired_mut().title = "Renamed".to_string();
        assert!(s.is_dirty());
        let cmds = s.diff();
        assert_eq!(cmds.len(), 1);
        assert!(matches!(cmds[0], WindowCommand::SetTitle(_)));
    }

    #[test]
    fn multiple_field_changes_batch_together() {
        let mut s = WindowSync::new(cfg());
        {
            let d = s.desired_mut();
            d.inner_size = PhysicalSize::new(1920, 1080);
            d.visible = false;
            d.resizable = false;
        }
        let cmds = s.diff();
        assert_eq!(cmds.len(), 3);
    }

    #[test]
    fn mark_applied_clears_the_diff() {
        let mut s = WindowSync::new(cfg());
        s.desired_mut().inner_size = PhysicalSize::new(1920, 1080);
        assert!(!s.diff().is_empty());
        s.mark_applied();
        assert!(s.diff().is_empty());
        assert!(!s.is_dirty());
    }

    #[test]
    fn clamped_resize_does_not_resend() {
        // Engine asks for 1920x1080; we command it and mark applied.
        let mut s = WindowSync::new(cfg());
        s.desired_mut().inner_size = PhysicalSize::new(1920, 1080);
        let _ = s.diff();
        s.mark_applied();
        // OS clamps to 1600x1000 and reports it back.
        let changed = s.observe(&WindowEvent::Resized(PhysicalSize::new(1600, 1000)));
        assert!(changed);
        assert_eq!(s.realized().inner_size, Some(PhysicalSize::new(1600, 1000)));
        // We must NOT fight the OS by re-requesting 1920x1080.
        assert!(s.diff().is_empty());
    }

    #[test]
    fn new_desire_after_clamp_does_resend() {
        let mut s = WindowSync::new(cfg());
        s.desired_mut().inner_size = PhysicalSize::new(1920, 1080);
        let _ = s.diff();
        s.mark_applied();
        s.observe(&WindowEvent::Resized(PhysicalSize::new(1600, 1000)));
        // A genuinely new target must be commanded.
        s.desired_mut().inner_size = PhysicalSize::new(1024, 768);
        let cmds = s.diff();
        assert_eq!(cmds.len(), 1);
        assert!(matches!(cmds[0], WindowCommand::RequestInnerSize(_)));
    }

    #[test]
    fn observe_updates_focus_and_scale() {
        let mut s = WindowSync::new(cfg());
        assert!(s.observe(&WindowEvent::Focused(true)));
        assert!(s.realized().focused);
        assert!(s.observe(&WindowEvent::ScaleFactorChanged {
            scale_factor_milli: 2000,
            new_inner_size: PhysicalSize::new(2560, 1440),
        }));
        assert_eq!(s.realized().scale_factor_milli, Some(2000));
        assert_eq!(s.realized().inner_size, Some(PhysicalSize::new(2560, 1440)));
        // Idempotent re-observation reports no change.
        assert!(!s.observe(&WindowEvent::Focused(true)));
    }
}
