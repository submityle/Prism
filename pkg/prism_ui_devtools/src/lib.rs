//! `prism_ui_devtools` — introspection and debugging tools for Loom runtimes.
//!
//! This crate turns the normally transient state of a [`prism_ui`] runtime into
//! owned, deterministic, human-readable artifacts:
//!
//! * [`snapshot`] walks a live [`Element`] tree into an owned [`TreeSnapshot`]
//!   you can store, compare, and query ([`TreeSnapshot::node_count`],
//!   [`TreeSnapshot::depth`]).
//! * [`render_tree`] pretty-prints a [`TreeSnapshot`] into an indented
//!   multi-line `String` for debugging and golden tests.
//! * [`OpTrace`] tallies a slice of [`BackendOp`]s by variant and renders a
//!   stable one-line [`OpTrace::summary`].
//!
//! Everything is `no_std`-friendly (uses `alloc`) and produces deterministic
//! output, which is the key quality bar for the snapshot tests these tools feed.
//!
//! # Example
//!
//! ```
//! use prism_ui::Element;
//! use prism_ui_devtools::{render_tree, snapshot};
//!
//! let view = Element::box_()
//!     .child(Element::text("hello"))
//!     .child(Element::text("world"));
//!
//! let snap = snapshot(&view);
//! assert_eq!(snap.node_count(), 3);
//! assert_eq!(snap.depth(), 2);
//!
//! assert_eq!(
//!     render_tree(&snap),
//!     "Box\n  Text \"hello\"\n  Text \"world\"\n",
//! );
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

pub mod format;
pub mod snapshot;
pub mod trace;

pub use format::render_tree;
pub use snapshot::{snapshot, SnapshotNode, TreeSnapshot};
pub use trace::OpTrace;

#[cfg(all(test, feature = "std"))]
mod tests {
    #![allow(clippy::std_instead_of_alloc, reason = "tests run under std")]

    use prism_ui::layout::{AvailableSpace, Size};
    use prism_ui::{Element, RecordingBackend, Ui};

    use crate::{render_tree, snapshot, OpTrace};

    fn definite(w: f32, h: f32) -> Size<AvailableSpace> {
        Size::new(AvailableSpace::Definite(w), AvailableSpace::Definite(h))
    }

    #[test]
    fn snapshot_counts_and_kinds() {
        let view = Element::box_()
            .child(Element::text("a"))
            .child(Element::text("b"));
        let snap = snapshot(&view);

        assert_eq!(snap.node_count(), 3);
        assert_eq!(snap.depth(), 2);
        assert_eq!(snap.root.kind, "Box");
        assert_eq!(snap.root.text, None);
        assert_eq!(snap.root.children.len(), 2);
        assert_eq!(snap.root.children[0].kind, "Text");
        assert_eq!(snap.root.children[0].text.as_deref(), Some("a"));
        assert_eq!(snap.root.children[1].text.as_deref(), Some("b"));
    }

    #[test]
    fn snapshot_captures_classes() {
        let view = Element::box_().class("card").class("raised");
        let snap = snapshot(&view);
        assert_eq!(snap.root.classes, ["card", "raised"]);
    }

    #[test]
    fn snapshot_nested_depth() {
        let view = Element::box_().child(Element::box_().child(Element::text("deep")));
        let snap = snapshot(&view);
        assert_eq!(snap.node_count(), 3);
        assert_eq!(snap.depth(), 3);
    }

    #[test]
    fn render_tree_exact() {
        let view = Element::box_()
            .child(Element::text("hello"))
            .child(Element::box_().child(Element::text("nested")));
        let snap = snapshot(&view);
        assert_eq!(
            render_tree(&snap),
            "Box\n  Text \"hello\"\n  Box\n    Text \"nested\"\n",
        );
    }

    #[test]
    fn render_empty_box() {
        let view = Element::box_();
        let snap = snapshot(&view);
        assert_eq!(snap.node_count(), 1);
        assert_eq!(snap.depth(), 1);
        assert_eq!(render_tree(&snap), "Box\n");
    }

    #[test]
    fn render_custom_kind_label() {
        let view = Element::custom("Canvas").child(Element::text("x"));
        let snap = snapshot(&view);
        assert_eq!(snap.root.kind, "Canvas");
        assert_eq!(render_tree(&snap), "Canvas\n  Text \"x\"\n");
    }

    #[test]
    fn optrace_counts_from_mount() {
        let view = Element::box_()
            .child(Element::text("a"))
            .child(Element::text("b"));
        let mut ui = Ui::new(RecordingBackend::new());
        ui.mount(&view);
        ui.compute_layout(definite(800.0, 600.0));

        let trace = OpTrace::from_ops(ui.backend().ops());

        // Three creates (box + two texts), two set_texts.
        assert_eq!(trace.creates(), 3);
        assert_eq!(trace.set_texts(), 2);
        assert_eq!(trace.removes(), 0);
        assert_eq!(trace.reorders(), 0);
        // Layout was computed, so every node got a layout.
        assert_eq!(trace.set_layouts(), 3);
        assert_eq!(
            trace.total(),
            trace.creates()
                + trace.removes()
                + trace.set_texts()
                + trace.set_layouts()
                + trace.set_paints()
                + trace.reorders(),
        );
    }

    #[test]
    fn optrace_empty_summary() {
        let trace = OpTrace::from_ops(&[]);
        assert_eq!(trace.total(), 0);
        assert_eq!(
            trace.summary(),
            "0 ops (create=0, remove=0, set_text=0, set_layout=0, set_paint=0, reorder=0)",
        );
    }

    #[test]
    fn optrace_summary_exact() {
        let view = Element::box_().child(Element::text("a"));
        let mut ui = Ui::new(RecordingBackend::new());
        ui.mount(&view);

        let trace = OpTrace::from_ops(ui.backend().ops());
        // Mount without layout: two creates, one set_text, and a paint per node.
        assert_eq!(trace.creates(), 2);
        assert_eq!(trace.set_texts(), 1);
        assert_eq!(trace.set_layouts(), 0);
        assert_eq!(trace.set_paints(), 2);
        assert_eq!(
            trace.summary(),
            "5 ops (create=2, remove=0, set_text=1, set_layout=0, set_paint=2, reorder=0)",
        );
    }
}
