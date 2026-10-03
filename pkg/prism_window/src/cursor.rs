//! Cursor appearance, visibility, and grab/confinement options.

/// How the cursor is confined to the window.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum CursorGrabMode {
    /// The cursor moves freely and can leave the window.
    #[default]
    None,
    /// The cursor is confined to the window bounds but still moves visibly.
    Confined,
    /// The cursor is locked in place (position frozen); motion is delivered as
    /// relative deltas, used for mouse-look.
    Locked,
}

/// The shape drawn for the system cursor.
///
/// Mirrors the common CSS / native cursor set; backends map these to the
/// platform's own cursors.
#[expect(
    missing_docs,
    reason = "self-describing standard cursor shapes; documenting each variant adds no information"
)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum CursorIcon {
    #[default]
    Default,
    ContextMenu,
    Help,
    Pointer,
    Progress,
    Wait,
    Cell,
    Crosshair,
    Text,
    VerticalText,
    Alias,
    Copy,
    Move,
    NoDrop,
    NotAllowed,
    Grab,
    Grabbing,
    EResize,
    NResize,
    NeResize,
    NwResize,
    SResize,
    SeResize,
    SwResize,
    WResize,
    EwResize,
    NsResize,
    NeswResize,
    NwseResize,
    ColResize,
    RowResize,
    AllScroll,
    ZoomIn,
    ZoomOut,
}

/// Per-window cursor state: shape, visibility, confinement, and hit-testing.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct CursorOptions {
    /// The cursor shape.
    pub icon: CursorIcon,
    /// Whether the cursor is drawn while over the window.
    pub visible: bool,
    /// How the cursor is confined to the window.
    pub grab_mode: CursorGrabMode,
    /// Whether the window participates in cursor hit-testing. When `false` the
    /// window is click-through (events pass to the window behind it).
    pub hit_test: bool,
}

impl Default for CursorOptions {
    fn default() -> Self {
        Self {
            icon: CursorIcon::Default,
            visible: true,
            grab_mode: CursorGrabMode::None,
            hit_test: true,
        }
    }
}
