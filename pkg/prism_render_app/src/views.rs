//! The render world's view family.
//!
//! Bevy carries an `ExtractedView` per camera in its render world, built by a
//! `camera_driver`. Prism keeps the *view* on the render side and the *camera*
//! on the scene side: the scene publishes what it wants rendered, and the render
//! world materializes it into a
//! [`ViewRegistry`](prism_render_architecture::view_family::ViewRegistry) of
//! views (main / stereo eye / shadow / reflection / capture / editor / offline
//! tile).
//!
//! There is deliberately no camera type here. A camera is scene/gameplay state;
//! the renderer only consumes the resulting views, which is what this module
//! owns. The registry's per-view [`ViewImportance`] is what the shared
//! virtual-resource budget aggregates, so building it here is the render side of
//! the view contract.

use prism_ecs::prelude::{Res, ResMut, Resource};
use prism_render_architecture::view_family::{
    ViewHandle, ViewImportance, ViewKind, ViewRegistry,
};

use crate::record::RenderRecord;

/// The views the current frame renders, keyed by generational [`ViewHandle`].
#[derive(Resource)]
pub struct RenderViews {
    /// The live view family.
    registry: ViewRegistry,
    /// The primary view's handle, once the frame has built the family.
    primary: Option<ViewHandle>,
}

impl RenderViews {
    /// An empty view family.
    #[must_use]
    pub fn new() -> Self {
        Self {
            registry: ViewRegistry::new(),
            primary: None,
        }
    }

    /// The live view registry.
    #[must_use]
    pub fn registry(&self) -> &ViewRegistry {
        &self.registry
    }

    /// Mutable access to the view registry, for other render systems.
    pub fn registry_mut(&mut self) -> &mut ViewRegistry {
        &mut self.registry
    }

    /// The primary view's handle, if the frame built one.
    #[must_use]
    pub fn primary_view(&self) -> Option<ViewHandle> {
        self.primary
    }

    /// Registers the primary view on first use, then refreshes its importance.
    pub(crate) fn upsert_primary(&mut self, importance: ViewImportance) -> ViewHandle {
        let handle = match self.primary {
            Some(handle) if self.registry.contains(handle) => {
                self.registry.set_importance(handle, importance);
                handle
            }
            _ => {
                let handle = self.registry.register_with(ViewKind::Main, importance);
                self.primary = Some(handle);
                handle
            }
        };
        self.registry.set_enabled(handle, true);
        handle
    }

    /// Marks the primary view disabled (a minimized/zero-sized frame) without
    /// removing it, so its handle survives skipped frames.
    pub(crate) fn disable_primary(&mut self) {
        if let Some(handle) = self.primary {
            self.registry.set_enabled(handle, false);
        }
    }
}

impl Default for RenderViews {
    fn default() -> Self {
        Self::new()
    }
}

/// Materializes the frame's view family from the extracted render record.
///
/// Runs in [`RenderSystems::Prepare`](crate::RenderSystems::Prepare), before the
/// queue stage consumes the family. An empty viewport disables the primary view
/// rather than removing it, so a minimized frame does not churn the registry.
///
/// A single primary view is built today from the extracted viewport; per-view
/// specs (stereo / shadow / reflection / capture, each with its own importance)
/// arrive with scene-side view extraction, and the registry already models them
/// via [`ViewKind`].
pub(crate) fn build_view_family(record: Res<RenderRecord>, mut views: ResMut<RenderViews>) {
    if record.viewport.is_empty() {
        views.disable_primary();
        return;
    }
    views.upsert_primary(ViewImportance::uniform(1.0));
}
