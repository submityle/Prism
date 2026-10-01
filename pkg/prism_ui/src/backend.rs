//! The engine-agnostic render target.
//!
//! The [`Ui`](crate::Ui) runtime never talks to a renderer directly. Instead it
//! emits a stream of [`BackendOp`]s describing the *minimal* mutations needed to
//! bring the engine's retained scene in line with the latest view. A concrete
//! engine (Bevy, a headless test harness, a web canvas, ...) implements
//! [`Backend`] to turn those ops into real nodes.
//!
//! This indirection is what lets Loom honour its core performance contract —
//! *cost ∝ change, not scene size* — in an engine-neutral way: the runtime
//! decides what changed, the backend only hears about the delta.

use alloc::string::String;
use alloc::vec::Vec;

use prism_ui_layout::{Point, Size};

use crate::element::ElementKind;
use crate::paint::PaintStyle;

/// A stable identifier the runtime assigns to each materialised node.
///
/// Ids are allocated monotonically and are never reused within a single
/// [`Ui`](crate::Ui); a removed node's id will not be handed out again. Backends
/// can therefore use it directly as a slot/entity key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BackendId(pub u64);

/// A single mutation the backend must apply to its retained scene.
///
/// Ops are emitted in dependency order within a flush: a node is always
/// `Create`d before it is referenced as a parent, before any text/layout/paint
/// is set on it, and before it is reordered.
#[derive(Clone, Debug, PartialEq)]
pub enum BackendOp {
    /// Materialise a new node of the given [`ElementKind`] under `parent` at
    /// position `index` among its siblings. A `parent` of `None` marks the root.
    Create {
        /// The id assigned to the new node.
        id: BackendId,
        /// The kind of node to create.
        kind: ElementKind,
        /// The parent to attach to, or `None` for the root node.
        parent: Option<BackendId>,
        /// The node's position among its new siblings.
        index: usize,
    },
    /// Set (or update) the text content of a text node.
    SetText {
        /// The node to update.
        id: BackendId,
        /// The new text content.
        text: String,
    },
    /// Set the computed geometry of a node, relative to its parent.
    SetLayout {
        /// The node to update.
        id: BackendId,
        /// Top-left corner relative to the parent's content box.
        location: Point<f32>,
        /// Border-box size.
        size: Size<f32>,
    },
    /// Set the resolved visual appearance of a node.
    SetPaint {
        /// The node to update.
        id: BackendId,
        /// The new paint style.
        paint: PaintStyle,
    },
    /// Destroy a node. Its children are removed in their own `Remove` ops first.
    Remove {
        /// The node to destroy.
        id: BackendId,
    },
    /// Reorder a parent's children to exactly `order`.
    Reorder {
        /// The parent whose children are being reordered.
        parent: BackendId,
        /// The complete, new child order.
        order: Vec<BackendId>,
    },
}

/// A concrete sink for [`BackendOp`]s.
///
/// Implementors translate ops into real engine state. The runtime calls
/// [`Backend::apply`] once per op during a flush; implementations should be
/// cheap and must not call back into the [`Ui`](crate::Ui).
pub trait Backend {
    /// Applies a single op to the retained scene.
    fn apply(&mut self, op: BackendOp);
}

/// A reference [`Backend`] that records every op it receives.
///
/// This is the headless target used by tests and tooling: it performs no
/// rendering, it simply appends ops to a log so assertions can prove the
/// runtime emitted the minimal, correct mutation stream.
#[derive(Clone, Debug, Default)]
pub struct RecordingBackend {
    ops: Vec<BackendOp>,
}

impl RecordingBackend {
    /// Creates an empty recorder.
    #[must_use]
    pub fn new() -> Self {
        Self { ops: Vec::new() }
    }

    /// Returns the ops recorded so far, in application order.
    #[must_use]
    pub fn ops(&self) -> &[BackendOp] {
        &self.ops
    }

    /// Removes and returns the recorded ops, leaving the recorder empty.
    ///
    /// Useful between frames to inspect only the ops produced by the latest
    /// update.
    #[must_use]
    pub fn take(&mut self) -> Vec<BackendOp> {
        core::mem::take(&mut self.ops)
    }

    /// Clears the recorded ops.
    pub fn clear(&mut self) {
        self.ops.clear();
    }

    /// Number of ops recorded so far.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Whether no ops have been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }
}

impl Backend for RecordingBackend {
    fn apply(&mut self, op: BackendOp) {
        self.ops.push(op);
    }
}
