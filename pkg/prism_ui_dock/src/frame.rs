//! Host-side orchestration for action, history, solve, and cache ordering.

use crate::{
    input::process_input, transition, Action, DockFrameSample, DockHistory, DockInputError,
    DockInputEvent, DockInputOutput, DockLayout, DockRuntimeMetrics, DragState, LayoutMetrics,
    RenderPlan, RenderPlanCache, RenderPlanError, Viewport,
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
    /// The render-feature adapter could not assemble the Element tree.
    #[cfg(feature = "render")]
    View(crate::DockViewError),
    /// A frame sample did not complete the Host-to-present instrumentation
    /// contract and was not committed.
    Metrics,
    /// A platform-neutral input event could not be converted into an action.
    Input(DockInputError),
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
    metrics: DockRuntimeMetrics,
    drag: DragState,
    pointer_capture: Option<u64>,
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
            metrics: DockRuntimeMetrics::new(),
            drag: DragState::Idle,
            pointer_capture: None,
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

    /// Starts instrumentation for one application-owned frame.
    ///
    /// The returned sample is intentionally independent of any platform
    /// clock. Callers record each real boundary as it completes, then pass the
    /// completed sample to [`Self::commit_frame`].
    #[must_use]
    pub const fn begin_frame(&self, frame_id: u64) -> DockFrameSample {
        DockRuntimeMetrics::begin_frame(frame_id)
    }

    /// Solves the current layout and records the Host stage in `sample`.
    ///
    /// `solve_duration_ns` and `dropped_frame` must be measured/reported by
    /// the caller around the actual host solve. Element counts are recorded
    /// separately by [`DockFrameSample::element_updated`]. This method only
    /// connects the timing observation to the host's canonical solve path.
    pub fn plan_instrumented<'a>(
        &'a mut self,
        viewport: Viewport,
        metrics: LayoutMetrics,
        sample: &mut DockFrameSample,
        solve_duration_ns: u64,
        dropped_frame: bool,
    ) -> Result<&'a RenderPlan, DockHostError> {
        let plan = self.plan(viewport, metrics)?;
        if !sample.host_solved(solve_duration_ns, dropped_frame) {
            return Err(DockHostError::Metrics);
        }
        Ok(plan)
    }

    /// Commits one complete Host-to-present sample to the host-owned
    /// collector. Incomplete or out-of-order samples are rejected.
    pub fn commit_frame(&mut self, sample: DockFrameSample) -> Result<(), DockHostError> {
        if self.metrics.record(sample) {
            Ok(())
        } else {
            Err(DockHostError::Metrics)
        }
    }

    /// Processes one platform-neutral pointer event through the canonical
    /// hit-test, preview, reducer, and plan path.
    pub fn handle_input(
        &mut self,
        event: DockInputEvent,
        viewport: Viewport,
        metrics: LayoutMetrics,
    ) -> Result<DockInputOutput, DockHostError> {
        let before = self.plan(viewport, metrics)?.clone();
        let (action, preview, committed) =
            process_input(&mut self.drag, &mut self.pointer_capture, event, before)
                .map_err(DockHostError::Input)?;
        if let Some(action) = action {
            self.dispatch(action)?;
        }
        if matches!(self.drag, DragState::Drop { .. } | DragState::Cancel { .. }) {
            self.drag = transition(&self.drag, crate::DragEvent::Reset)
                .map_err(DockInputError::Drag)
                .map_err(DockHostError::Input)?;
        }
        let plan = self.plan(viewport, metrics)?.clone();
        Ok(DockInputOutput {
            plan,
            preview,
            committed,
            drag: self.drag.clone(),
        })
    }

    /// Returns the host-owned bounded frame metrics collector.
    #[must_use]
    pub fn metrics(&self) -> &DockRuntimeMetrics {
        &self.metrics
    }

    /// Returns cache hit/miss counters for diagnostics.
    pub fn cache_counters(&self) -> (u64, u64) {
        (self.cache.hits(), self.cache.misses())
    }
}
#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;
    use crate::{Axis, DockNode, Ratio, Rect, SplitChild};

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

    #[test]
    fn host_wires_solve_to_the_canonical_frame_collector() {
        let mut host = DockHost::new(
            DockLayout {
                root: DockNode::panel("panel"),
                ..DockLayout::default()
            },
            4,
        )
        .unwrap();
        let viewport = Viewport::new(640, 480);
        let mut sample = host.begin_frame(9);
        host.plan_instrumented(viewport, LayoutMetrics::default(), &mut sample, 11, false)
            .unwrap();
        assert!(sample.element_updated(13, 1, 1));
        assert!(sample.draw_list_built(17, 24));
        assert!(sample.gpu_submitted(19));
        assert!(sample.presented(23));
        host.commit_frame(sample).unwrap();
        let summary = host.metrics().summary();
        assert_eq!(summary.samples, 1);
        assert_eq!(summary.solve_p50_ns, 11);
        assert_eq!(summary.draw_list_p50_ns, 17);
        assert_eq!(summary.gpu_p50_ns, 19);
        assert_eq!(summary.present_p50_ns, 23);
    }

    #[test]
    fn host_rejects_an_incomplete_frame_without_recording_it() {
        let mut host = DockHost::new(
            DockLayout {
                root: DockNode::panel("panel"),
                ..DockLayout::default()
            },
            4,
        )
        .unwrap();
        let sample = host.begin_frame(1);
        assert_eq!(host.commit_frame(sample), Err(DockHostError::Metrics));
        assert!(host.metrics().is_empty());
    }

    #[test]
    fn input_hit_test_preview_and_pointer_up_commit_one_action() {
        let layout = DockLayout {
            root: DockNode::Split {
                axis: Axis::Horizontal,
                children: vec![
                    SplitChild {
                        ratio: Ratio::HALF,
                        node: DockNode::panel("source"),
                    },
                    SplitChild {
                        ratio: Ratio::HALF,
                        node: DockNode::panel("target"),
                    },
                ],
            },
            ..DockLayout::default()
        };
        let mut host = DockHost::new(layout, 4).unwrap();
        let viewport = Viewport::new(640, 480);
        host.handle_input(
            DockInputEvent::PointerDown {
                pointer_id: 7,
                position: crate::PointerPosition { x: 120, y: 240 },
            },
            viewport,
            LayoutMetrics::default(),
        )
        .unwrap();
        let moved = host
            .handle_input(
                DockInputEvent::PointerMove {
                    pointer_id: 7,
                    position: crate::PointerPosition { x: 520, y: 240 },
                },
                viewport,
                LayoutMetrics::default(),
            )
            .unwrap();
        assert!(moved.preview.is_some());
        let released = host
            .handle_input(
                DockInputEvent::PointerUp {
                    pointer_id: 7,
                    position: crate::PointerPosition { x: 520, y: 240 },
                },
                viewport,
                LayoutMetrics::default(),
            )
            .unwrap();
        assert!(released.committed);
        assert!(matches!(released.drag, DragState::Idle));
        assert!(host.layout().root.contains_panel("source"));
    }
}
