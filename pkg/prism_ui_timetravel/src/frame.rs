//! A single recorded UI frame on a timeline.
//!
//! A [`Frame`] couples a human-readable `label` with an owned
//! [`TreeSnapshot`] of the view at that moment, plus an optional
//! [`OpTrace`] describing the backend mutations that produced it. Because the
//! snapshot is fully owned, a frame can be stored for the lifetime of the
//! debugging session and compared or replayed later.

use alloc::string::String;

use prism_ui::Element;
use prism_ui_devtools::{snapshot, OpTrace, TreeSnapshot};

/// One recorded point on a [`crate::Timeline`].
///
/// A frame bundles everything time-travel debugging needs about a single UI
/// state: a `label`, the owned [`TreeSnapshot`], and an optional [`OpTrace`]
/// of the mutations that produced it. Frames are value types: they derive
/// [`Clone`], [`Debug`], [`PartialEq`], and [`Eq`], so equality compares the
/// label, the captured tree, and the trace structurally.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Frame {
    /// Human-readable name for this frame, e.g. `"after click"`.
    label: String,
    /// The owned snapshot of the view at this point in time.
    snapshot: TreeSnapshot,
    /// Optional tally of the backend ops that produced this frame.
    trace: Option<OpTrace>,
}

impl Frame {
    /// Creates a frame from an already-captured [`TreeSnapshot`], with no
    /// attached [`OpTrace`].
    #[must_use]
    pub fn new(label: impl Into<String>, snapshot: TreeSnapshot) -> Self {
        Self {
            label: label.into(),
            snapshot,
            trace: None,
        }
    }

    /// Captures a frame directly from a live [`Element`] tree.
    ///
    /// This walks `element` with [`prism_ui_devtools::snapshot`] into an owned
    /// [`TreeSnapshot`], so the returned frame does not borrow `element`.
    #[must_use]
    pub fn capture(label: impl Into<String>, element: &Element) -> Self {
        Self::new(label, snapshot(element))
    }

    /// Attaches an [`OpTrace`] to this frame, consuming and returning it so
    /// calls can be chained after [`Frame::new`] or [`Frame::capture`].
    #[must_use]
    pub fn with_trace(mut self, trace: OpTrace) -> Self {
        self.trace = Some(trace);
        self
    }

    /// Returns the frame's human-readable label.
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Returns the captured [`TreeSnapshot`].
    #[must_use]
    pub fn snapshot(&self) -> &TreeSnapshot {
        &self.snapshot
    }

    /// Returns the attached [`OpTrace`], or [`None`] when none was recorded.
    #[must_use]
    pub fn trace(&self) -> Option<&OpTrace> {
        self.trace.as_ref()
    }

    /// Returns `true` when an [`OpTrace`] is attached to this frame.
    #[must_use]
    pub fn has_trace(&self) -> bool {
        self.trace.is_some()
    }

    /// Total number of nodes in the captured tree, including the root.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.snapshot.node_count()
    }

    /// Depth of the captured tree. A lone root has depth `1`.
    #[must_use]
    pub fn depth(&self) -> usize {
        self.snapshot.depth()
    }
}

#[cfg(test)]
mod tests {
    use super::Frame;
    use prism_ui::backend::{BackendId, BackendOp};
    use prism_ui::{Element, ElementKind};
    use prism_ui_devtools::OpTrace;

    fn sample_trace() -> OpTrace {
        OpTrace::from_ops(&[
            BackendOp::Create {
                id: BackendId(1),
                kind: ElementKind::Box,
                parent: None,
                index: 0,
            },
            BackendOp::SetText {
                id: BackendId(2),
                text: "hi".into(),
            },
        ])
    }

    #[test]
    fn capture_reads_element_tree() {
        let view = Element::box_().child(Element::text("hello"));
        let frame = Frame::capture("initial", &view);
        assert_eq!(frame.label(), "initial");
        assert_eq!(frame.node_count(), 2);
        assert_eq!(frame.depth(), 2);
        assert_eq!(frame.snapshot().root.kind, "Box");
        assert!(frame.trace().is_none());
        assert!(!frame.has_trace());
    }

    #[test]
    fn with_trace_attaches_op_trace() {
        let frame = Frame::capture("x", &Element::box_()).with_trace(sample_trace());
        assert!(frame.has_trace());
        assert_eq!(frame.trace().unwrap().total(), 2);
        assert_eq!(frame.trace().unwrap().creates(), 1);
        assert_eq!(frame.trace().unwrap().set_texts(), 1);
    }

    #[test]
    fn new_from_owned_snapshot() {
        let snap = prism_ui_devtools::snapshot(&Element::text("t"));
        let frame = Frame::new("leaf", snap.clone());
        assert_eq!(frame.snapshot(), &snap);
        assert_eq!(frame.node_count(), 1);
        assert_eq!(frame.depth(), 1);
    }

    #[test]
    fn node_count_and_depth_follow_nested_tree() {
        let view = Element::box_()
            .child(Element::text("a"))
            .child(Element::box_().child(Element::text("b")));
        let frame = Frame::new("nested", prism_ui_devtools::snapshot(&view));
        // root + two children + one grandchild = 4 nodes, depth 3.
        assert_eq!(frame.node_count(), 4);
        assert_eq!(frame.depth(), 3);
    }

    #[test]
    fn equality_is_structural() {
        let a = Frame::capture("v", &Element::box_().child(Element::text("a")));
        let b = Frame::capture("v", &Element::box_().child(Element::text("a")));
        let c = Frame::capture("v", &Element::box_().child(Element::text("b")));
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}
