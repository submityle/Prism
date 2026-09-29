//! Main, shadow, reflection, stereo, capture, and offline views.
//!
//! A single frame renders a *family* of views that all draw on the same shared
//! subsystems (virtual-resource residency, geometry and shadow `LOD`). This
//! module owns the `CPU`-side contract for that family:
//!
//! * [`ViewImportance`] and its aggregation live in [`importance`]: three
//!   sanitized, non-negative weights per view, combined across the family by
//!   component-wise maximum or a kind-weighted sum to drive the shared budget.
//! * [`registry`] holds the live views in a [`registry::ViewRegistry`] keyed by
//!   generational [`ViewHandle`], so a handle to a removed view is detected
//!   rather than silently aliasing a reused slot.
//!
//! Views are identified by [`ViewHandle`] (an [`crate::abi::GenerationalHandle`])
//! and never carry a `GPU` handle here; the backend consumes these contracts,
//! pending the `GPU` backend.

use crate::abi::GenerationalHandle;

pub mod importance;
pub mod registry;

pub use registry::{ViewEntry, ViewRegistry};

/// Stable, generational identifier for a registered view.
pub type ViewHandle = GenerationalHandle;

/// What role a view plays in the frame, which sets its default importance.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ViewKind {
    /// Primary camera view.
    Main,
    /// One eye of a stereo pair.
    StereoEye,
    /// A shadow-casting light's depth view.
    Shadow,
    /// A planar or probe reflection view.
    Reflection,
    /// A runtime scene-capture view (e.g. a `RenderTarget`).
    SceneCapture,
    /// An editor viewport.
    Editor,
    /// One tile of an offline / path-traced render.
    OfflineTile,
}

/// How aggressively the shared subsystems should serve a view.
///
/// Each component is a sanitized, non-negative weight (see [`importance`]):
/// `streaming` drives virtual-resource residency, `geometry_lod` biases geometry
/// detail, and `shadow_lod` biases shadow-map detail. Holds floating-point data,
/// so it is [`PartialEq`] but deliberately not [`Eq`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewImportance {
    /// Residency / streaming urgency for this view.
    pub streaming: f32,
    /// Geometry level-of-detail bias for this view.
    pub geometry_lod: f32,
    /// Shadow level-of-detail bias for this view.
    pub shadow_lod: f32,
}
