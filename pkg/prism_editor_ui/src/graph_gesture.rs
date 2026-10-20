//! Canvas-local pointer capture with bounded previews and one release command.
use crate::{
    graph::{
        port_point, GraphPanelAction, GraphPanelController, GraphPanelError, GraphPanelViewport,
        Rect,
    },
    graph_view::BoundGraphPanel,
    SelectionMode,
};
use prism_editor_graph::{
    GraphCommand, GraphEdit, GraphPoint, GraphSnapshotToken, NodeId, PortDirection, PortId,
};
use std::sync::Arc;

/// Native button intent; hosts map primary, middle or modified input explicitly.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GraphPointerButton {
    /// Node drag, typed port connection or background rectangle selection.
    Primary,
    /// Pan regardless of the painted target below the pointer.
    Pan,
}
/// Target from the exact painted publication, in reverse painter order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GraphPointerHit {
    /// Empty canvas, including wires which do not consume drag capture.
    Canvas,
    /// Painted stable node identity.
    Node(NodeId),
    /// Painted typed port, above its owning node box.
    Port(PortId),
}
/// Small preview descriptor. No graph or payload is copied on pointer movement.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GraphGesturePreview {
    /// Integer translation applied to the captured bounded selection.
    Nodes {
        /// Rounded document-space translation from the initial press.
        delta: GraphPoint,
        /// Captured node count, bounded by the selection policy.
        count: usize,
    },
    /// Rubber-band endpoints in canvas logical pixels.
    Connection {
        /// Starting port, either input or output.
        port: PortId,
        /// Exact un-clipped port anchor.
        from: [f64; 2],
        /// Latest bounded pointer position.
        to: [f64; 2],
    },
    /// Canvas-clamped rectangle in document coordinates, left/top/right/bottom.
    Marquee {
        /// Normalized document-space rectangle.
        rect: [f64; 4],
    },
    /// Candidate viewport; committed viewport remains unchanged until release.
    Pan {
        /// Validated candidate viewport.
        viewport: GraphPanelViewport,
    },
}
#[derive(Debug)]
pub(crate) struct GraphGesture {
    stamp: Arc<()>,
    token: GraphSnapshotToken,
    viewport: GraphPanelViewport,
    start: [f64; 2],
    originals: Vec<(NodeId, GraphPoint)>,
    mode: SelectionMode,
    pub(crate) preview: GraphGesturePreview,
}
impl GraphPanelController {
    fn check_pointer_view(&self, view: &BoundGraphPanel) -> Result<(), GraphPanelError> {
        if Arc::ptr_eq(&self.stamp, &view.stamp) {
            Ok(())
        } else {
            Err(GraphPanelError::StaleView)
        }
    }
    fn check_pointer(&self) -> Result<&GraphGesture, GraphPanelError> {
        let capture = self.gesture.as_ref().ok_or(GraphPanelError::GestureState)?;
        if !Arc::ptr_eq(&capture.stamp, &self.stamp)
            || capture.token != self.editor.snapshot_token()
        {
            return Err(GraphPanelError::StaleView);
        }
        Ok(capture)
    }
    /// Whether this controller owns a pointer until release or cancellation.
    pub fn pointer_active(&self) -> bool {
        self.gesture.is_some()
    }
    /// Current bounded preview. Semantic actions invalidate and discard captures.
    pub fn gesture_preview(&self) -> Option<GraphGesturePreview> {
        self.check_pointer().ok().map(|g| g.preview)
    }
    /// Acquire one pointer using the current painted view and canvas-local pixels.
    /// Pressing an already selected node preserves the selection for group drag.
    /// Toggle-removing a node changes selection without acquiring a drag.
    pub fn pointer_press(
        &mut self,
        view: &BoundGraphPanel,
        position: [f64; 2],
        button: GraphPointerButton,
        mode: SelectionMode,
    ) -> Result<GraphPointerHit, GraphPanelError> {
        self.check_pointer_view(view)?;
        if self.pointer_active() {
            return Err(GraphPanelError::GestureState);
        }
        let hit = view.hit_test(position).ok_or(GraphPanelError::Geometry)?;
        let mut originals = Vec::new();
        let preview = if button == GraphPointerButton::Pan {
            GraphGesturePreview::Pan {
                viewport: self.viewport,
            }
        } else {
            match hit {
                GraphPointerHit::Node(node) => {
                    if mode != SelectionMode::Replace || !self.selected.contains(&node) {
                        self.apply(GraphPanelAction::Select(node, mode))?;
                    }
                    if !self.selected.contains(&node) {
                        return Ok(hit);
                    }
                    for id in &self.selected {
                        originals.push((
                            *id,
                            self.editor.position(*id).ok_or(GraphPanelError::Target)?,
                        ));
                    }
                    GraphGesturePreview::Nodes {
                        delta: GraphPoint::default(),
                        count: originals.len(),
                    }
                }
                GraphPointerHit::Port(port) => {
                    let anchor = port_point(&self.editor, port);
                    GraphGesturePreview::Connection {
                        port,
                        from: [
                            (anchor[0] - self.viewport.origin[0]) * self.viewport.zoom,
                            (anchor[1] - self.viewport.origin[1]) * self.viewport.zoom,
                        ],
                        to: position,
                    }
                }
                GraphPointerHit::Canvas => {
                    let point = self
                        .viewport
                        .document_point(position)
                        .ok_or(GraphPanelError::Geometry)?;
                    GraphGesturePreview::Marquee {
                        rect: [point[0], point[1], point[0], point[1]],
                    }
                }
            }
        };
        self.stamp = Arc::new(());
        self.gesture = Some(GraphGesture {
            stamp: self.stamp.clone(),
            token: self.editor.snapshot_token(),
            viewport: self.viewport,
            start: position,
            originals,
            mode,
            preview,
        });
        Ok(hit)
    }
    /// Preview captured movement; finite coordinates outside the canvas are
    /// accepted up to +/- one million logical pixels. Invalid moves retain the
    /// last preview. Work is bounded by selected nodes, with no model mutation.
    pub fn pointer_move(&mut self, position: [f64; 2]) -> Result<(), GraphPanelError> {
        let capture = self.check_pointer()?;
        if position
            .iter()
            .any(|p| !p.is_finite() || p.abs() > 1_000_000.)
        {
            return Err(GraphPanelError::Geometry);
        }
        let delta = [
            (position[0] - capture.start[0]) / capture.viewport.zoom,
            (position[1] - capture.start[1]) / capture.viewport.zoom,
        ];
        let preview = match capture.preview {
            GraphGesturePreview::Nodes { count, .. } => {
                let rounded = [delta[0].round(), delta[1].round()];
                if rounded.iter().any(|p| p.abs() > 2_000_000.) {
                    return Err(GraphPanelError::Geometry);
                }
                let delta = GraphPoint {
                    x: rounded[0] as i32,
                    y: rounded[1] as i32,
                };
                for (_, point) in &capture.originals {
                    if (i64::from(point.x) + i64::from(delta.x)).abs() > 1_000_000
                        || (i64::from(point.y) + i64::from(delta.y)).abs() > 1_000_000
                    {
                        return Err(GraphPanelError::Geometry);
                    }
                }
                GraphGesturePreview::Nodes { delta, count }
            }
            GraphGesturePreview::Connection { port, from, .. } => GraphGesturePreview::Connection {
                port,
                from,
                to: position,
            },
            GraphGesturePreview::Marquee { .. } => {
                let start = capture
                    .viewport
                    .document_point(capture.start)
                    .expect("validated press");
                let end = [
                    capture.viewport.origin[0]
                        + position[0].clamp(0., capture.viewport.size[0]) / capture.viewport.zoom,
                    capture.viewport.origin[1]
                        + position[1].clamp(0., capture.viewport.size[1]) / capture.viewport.zoom,
                ];
                GraphGesturePreview::Marquee {
                    rect: [
                        start[0].min(end[0]),
                        start[1].min(end[1]),
                        start[0].max(end[0]),
                        start[1].max(end[1]),
                    ],
                }
            }
            GraphGesturePreview::Pan { .. } => {
                let mut viewport = capture.viewport;
                for (origin, delta) in viewport.origin.iter_mut().zip(delta) {
                    *origin -= delta;
                }
                viewport.validate()?;
                GraphGesturePreview::Pan { viewport }
            }
        };
        self.stamp = Arc::new(());
        let capture = self.gesture.as_mut().expect("checked capture");
        capture.preview = preview;
        capture.stamp = self.stamp.clone();
        Ok(())
    }
    /// Finish using the latest painted publication. Node release is one atomic
    /// undo step; port release uses the real kernel type/cycle/cardinality checks.
    /// Every failed release discards capture. Outside/non-finite release cancels.
    pub fn pointer_release(
        &mut self,
        view: &BoundGraphPanel,
        position: [f64; 2],
    ) -> Result<(), GraphPanelError> {
        let result = (|| {
            self.check_pointer_view(view)?;
            self.check_pointer()?;
            let hit = view.hit_test(position).ok_or(GraphPanelError::Geometry)?;
            self.pointer_move(position)?;
            let capture = self.gesture.take().expect("checked capture");
            match capture.preview {
                GraphGesturePreview::Nodes { delta, .. } if delta != GraphPoint::default() => {
                    let edits = capture
                        .originals
                        .iter()
                        .map(|(node, p)| GraphEdit::MoveNode {
                            node: *node,
                            position: GraphPoint {
                                x: p.x + delta.x,
                                y: p.y + delta.y,
                            },
                        })
                        .collect();
                    self.apply(GraphPanelAction::Command(GraphCommand::Edit(edits)))?;
                }
                GraphGesturePreview::Nodes { .. } => {}
                GraphGesturePreview::Connection { port, .. } => {
                    let GraphPointerHit::Port(target) = hit else {
                        return Err(GraphPanelError::Target);
                    };
                    let (output, input) = if self
                        .editor
                        .graph()
                        .port(port)
                        .ok_or(GraphPanelError::Target)?
                        .direction()
                        == PortDirection::Output
                    {
                        (port, target)
                    } else {
                        (target, port)
                    };
                    self.apply(GraphPanelAction::Command(GraphCommand::Edit(vec![
                        GraphEdit::Connect {
                            edge: self.editor.next_ids().2,
                            output,
                            input,
                        },
                    ])))?;
                }
                GraphGesturePreview::Marquee { rect } => {
                    self.apply(GraphPanelAction::SelectRect(rect, capture.mode))?
                }
                GraphGesturePreview::Pan { viewport } => {
                    self.apply(GraphPanelAction::Viewport(viewport))?
                }
            }
            Ok(())
        })();
        self.gesture = None;
        self.stamp = Arc::new(());
        result
    }
    /// Discard preview without changing graph/history or committing a pan.
    /// Selection changes made on press remain selected. Idempotent.
    pub fn cancel_pointer(&mut self) -> bool {
        let active = self.gesture.take().is_some();
        if active {
            self.stamp = Arc::new(());
        }
        active
    }
    /// Anchor-preserving canvas zoom. Active capture must be cancelled first.
    /// Invalid geometry leaves the viewport and publication identity unchanged.
    pub fn zoom_at(
        &mut self,
        view: &BoundGraphPanel,
        position: [f64; 2],
        factor: f64,
    ) -> Result<(), GraphPanelError> {
        self.check_pointer_view(view)?;
        if self.pointer_active() {
            return Err(GraphPanelError::GestureState);
        }
        let anchor = self
            .viewport
            .document_point(position)
            .ok_or(GraphPanelError::Geometry)?;
        let mut viewport = self.viewport;
        viewport.zoom *= factor;
        for i in 0..2 {
            viewport.origin[i] = anchor[i] - position[i] / viewport.zoom;
        }
        self.apply(GraphPanelAction::Viewport(viewport))
    }
    pub(crate) fn display_viewport(&self) -> GraphPanelViewport {
        match self.gesture_preview() {
            Some(GraphGesturePreview::Pan { viewport }) => viewport,
            _ => self.viewport,
        }
    }
    pub(crate) fn preview_offset(&self, node: NodeId) -> [f64; 2] {
        let Ok(g) = self.check_pointer() else {
            return [0.; 2];
        };
        if let GraphGesturePreview::Nodes { delta, .. } = g.preview
            && g.originals
                .binary_search_by_key(&node, |(id, _)| *id)
                .is_ok()
        {
            return [f64::from(delta.x), f64::from(delta.y)];
        }
        [0.; 2]
    }
    pub(crate) fn preview_query(&self) -> Rect {
        let mut bounds = self.display_viewport().bounds().0;
        if let Some(GraphGesturePreview::Nodes { delta, .. }) = self.gesture_preview() {
            for (i, d) in [delta.x, delta.y].into_iter().enumerate() {
                bounds[i] = bounds[i].min(bounds[i] - f64::from(d));
                bounds[i + 2] = bounds[i + 2].max(bounds[i + 2] - f64::from(d));
            }
        }
        Rect(bounds)
    }
}
