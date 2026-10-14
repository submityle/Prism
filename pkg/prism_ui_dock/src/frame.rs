//! Host-side orchestration for action, history, solve, and cache ordering.

use crate::{
    Action, DockHistory, DockLayout, LayoutMetrics, RenderPlan, RenderPlanCache, RenderPlanError,
    Viewport,
};

/// Error returned by [`DockHost`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DockHostError {
    /// The initial or resulting layout was rejected.
    Layout(crate::LayoutError),
    /// The history controller rejected the action or traversal.
    History(crate::HistoryError),
    /// The current layout cannot be solved for the requested viewport.
    Render(RenderPlanError),
}

impl From<crate::HistoryError> for DockHostError {
    fn from(error: crate::HistoryError) -> Self {
        Self::History(error)
    }
}

/// Deterministic dock host state.
///
/// This type is the narrow application-facing seam: it owns the current
/// layout, serializes reducer commits, records bounded history, increments a
/// revision, and solves through the exact-input cache. Window/surface code can
/// consume the returned [`RenderPlan`] without owning a second dock state.
#[derive(Clone, Debug)]
pub struct DockHost {
    layout: DockLayout,
    history: DockHistory,
    cache: RenderPlanCache,
    revision: u64,
}

impl DockHost {
    /// Creates a host after validating the initial layout.
    pub fn new(layout: DockLayout, history_capacity: usize) -> Result<Self, DockHostError> {
        layout.validate().map_err(DockHostError::Layout)?;
        if history_capacity == 0 {
            return Err(DockHostError::History(crate::HistoryError::ZeroCapacity));
        }
        Ok(Self {
            layout,
            history: DockHistory::new(history_capacity),
            cache: RenderPlanCache::new(),
            revision: 0,
        })
    }

    /// Returns the current layout.
    pub fn layout(&self) -> &DockLayout {
        &self.layout
    }

    /// Returns the monotonically increasing layout revision.
    pub const fn revision(&self) -> u64 {
        self.revision
    }

    /// Returns the bounded history controller.
    pub fn history(&self) -> &DockHistory {
        &self.history
    }

    /// Commits one action atomically and invalidates the plan cache.
    pub fn dispatch(&mut self, action: Action) -> Result<bool, DockHostError> {
        let next = self.history.execute(&self.layout, action)?;
        if next == self.layout {
            return Ok(false);
        }
        self.layout = next;
        self.revision = self.revision.wrapping_add(1);
        self.cache.clear();
        Ok(true)
    }

    /// Undoes one committed action and invalidates the plan cache.
    pub fn undo(&mut self) -> Result<(), DockHostError> {
        self.layout = self.history.undo(&self.layout)?;
        self.revision = self.revision.wrapping_add(1);
        self.cache.clear();
        Ok(())
    }

    /// Redoes one previously undone action and invalidates the plan cache.
    pub fn redo(&mut self) -> Result<(), DockHostError> {
        self.layout = self.history.redo(&self.layout)?;
        self.revision = self.revision.wrapping_add(1);
        self.cache.clear();
        Ok(())
    }

    /// Solves the current layout through the exact-input cache.
    pub fn plan(
        &mut self,
        viewport: Viewport,
        metrics: LayoutMetrics,
    ) -> Result<&RenderPlan, DockHostError> {
        self.cache
            .get_or_solve(&self.layout, viewport, metrics)
            .map_err(DockHostError::Render)
    }

    /// Returns cache hit/miss counters for diagnostics.
    pub fn cache_counters(&self) -> (u64, u64) {
        (self.cache.hits(), self.cache.misses())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DockNode, Rect};

    #[test]
    fn host_serializes_commit_history_and_plan_cache() {
        let mut host = DockHost::new(
            DockLayout {
                root: DockNode::panel("panel"),
                ..DockLayout::default()
            },
            4,
        )
        .unwrap();
        let viewport = Viewport::new(640, 480);
        host.plan(viewport, LayoutMetrics::default()).unwrap();
        host.plan(viewport, LayoutMetrics::default()).unwrap();
        assert_eq!(host.cache_counters(), (1, 1));
        host.dispatch(Action::Float {
            panel: "panel".into(),
            rect: Rect {
                x: 10,
                y: 10,
                width: 300,
                height: 200,
            },
        })
        .unwrap();
        assert_eq!(host.revision(), 1);
        host.undo().unwrap();
        assert_eq!(host.layout().root, DockNode::panel("panel"));
        host.redo().unwrap();
        assert!(host.layout().root == DockNode::Empty);
    }
}
