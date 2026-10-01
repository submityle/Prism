//! Reactive graph node primitives shared by signals, memos, and effects.

use alloc::rc::Rc;
use alloc::vec::Vec;
use core::cell::RefCell;

/// Opaque identifier of a node inside a [`crate::Runtime`].
///
/// Identifiers are stable for the lifetime of the node. Disposing a node frees
/// its slot, which may later be reused for a different node; holding a stale
/// [`NodeId`] after disposal is safe (operations become no-ops) but must not be
/// relied upon to address the original node.
pub type NodeId = usize;

/// Sentinel pushed on the observer stack by `untrack` to suppress dependency
/// collection. It can never collide with a real [`NodeId`] because a runtime
/// cannot allocate `usize::MAX` nodes.
pub(crate) const NO_OBSERVER: NodeId = NodeId::MAX;

/// Reactivity state of a node, ordered so that a numerically larger state is
/// "more stale". The ordering is load-bearing: [`State::escalates_to`] relies
/// on `Clean < Check < Dirty`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum State {
    /// Value is known to be up to date.
    Clean = 0,
    /// A transitive source may have changed; the node must verify its sources
    /// before it can be trusted (pull-based check).
    Check = 1,
    /// A direct source changed; the node must recompute.
    Dirty = 2,
}

impl State {
    #[inline]
    pub(crate) fn escalates_to(self, target: State) -> bool {
        (self as u8) < (target as u8)
    }
}

/// Role of a node in the reactive graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum NodeKind {
    /// A writable source value with no computation.
    Signal,
    /// A derived, lazily recomputed, cached value.
    Memo,
    /// A side effect that re-runs when its sources change.
    Effect,
}

/// A boxed recomputation. Returns `true` when the node's observable output
/// changed (always `false` for effects, which have no output).
pub(crate) type Computation = Rc<RefCell<dyn FnMut() -> bool>>;

/// A single node of the dependency graph.
///
/// Edges are stored on both ends: `sources` are the nodes this node read during
/// its last run, and `observers` are the nodes that read this node. Keeping both
/// directions lets writes propagate staleness downstream and lets reruns detach
/// cleanly, which is what keeps dependency tracking dynamic and glitch-free.
pub(crate) struct Node {
    pub(crate) kind: NodeKind,
    pub(crate) state: State,
    pub(crate) sources: Vec<NodeId>,
    pub(crate) observers: Vec<NodeId>,
    pub(crate) computation: Option<Computation>,
}

impl Node {
    pub(crate) fn new(kind: NodeKind, state: State, computation: Option<Computation>) -> Self {
        Self {
            kind,
            state,
            sources: Vec::new(),
            observers: Vec::new(),
            computation,
        }
    }
}
