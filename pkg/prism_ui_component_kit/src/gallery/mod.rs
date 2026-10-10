//! Dev-only showcase: mounts *every* kit control as a real, named instance so
//! the `gallery_real` example can render one sheet containing the whole
//! library (light + dark). Gated behind the `gallery` feature so it never
//! ships in normal builds.
//!
//! Each family module exposes `instances() -> Vec<Showcase>`; [`all`]
//! concatenates them in family order. Entries are the honest output of each
//! control's real `render`, mounted via
//! [`mount_component`](prism_ui_component::mount_component).
//!
//! Contains no Unreal Engine source or derived code.

use alloc::vec::Vec;

use prism_ui::Element;

mod basics;
mod containers;
mod display;
mod editor;
mod feedback;
mod inputs;
mod motion;
mod nav;
mod node_editor;
mod pickers;
mod utils;

/// A single named, fully mounted control instance for the gallery sheet.
pub struct Showcase {
    /// Human-readable label painted beneath the tile (e.g. `"Button / Filled"`).
    pub name: &'static str,
    /// The real element subtree produced by the control's `render`.
    pub element: Element,
}

impl Showcase {
    /// Builds a showcase entry from a label and a mounted element.
    #[must_use]
    pub fn new(name: &'static str, element: Element) -> Self {
        Self { name, element }
    }
}

/// Every kit control, in family order, as named real instances.
#[must_use]
pub fn all() -> Vec<Showcase> {
    let mut out = Vec::new();
    out.extend(basics::instances());
    out.extend(inputs::instances());
    out.extend(pickers::instances());
    out.extend(display::instances());
    out.extend(containers::instances());
    out.extend(feedback::instances());
    out.extend(nav::instances());
    out.extend(editor::instances());
    out.extend(node_editor::instances());
    out.extend(motion::instances());
    out.extend(utils::instances());
    out
}
