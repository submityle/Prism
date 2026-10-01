//! The reactive runtime: owns the dependency graph and drives propagation.
//!
//! # Algorithm
//!
//! Propagation follows the widely documented "push the dirt, pull the value"
//! scheme (as used by SolidJS/Reactively):
//!
//! * **Push (write):** setting a signal eagerly marks its direct observers
//!   [`State::Dirty`] and recursively marks their transitive observers
//!   [`State::Check`]. Marking stops as soon as a node is already at least as
//!   stale as the mark, so each edge is visited at most once per write.
//! * **Pull (read):** reading a memo lazily verifies it via
//!   [`Runtime::update_if_necessary`]. A `Check` node first verifies its sources
//!   (depth-first); only if a source actually changed does the node recompute.
//!   This guarantees each node recomputes at most once per update and never
//!   observes an inconsistent (glitchy) intermediate state.
//!
//! Effects are the only eagerly scheduled nodes: when they become stale they are
//! queued and flushed at the end of the current batch.

use alloc::rc::Rc;
use alloc::vec::Vec;
use core::cell::RefCell;

use crate::node::{Computation, Node, NodeId, NodeKind, State, NO_OBSERVER};

/// Defensive cap on a single effect-flush drain to convert an accidental
/// infinite update cycle into a bounded panic instead of a hang.
const MAX_FLUSH_ITERATIONS: usize = 1_000_000;

pub(crate) struct Inner {
    nodes: Vec<Option<Node>>,
    free: Vec<NodeId>,
    observer_stack: Vec<NodeId>,
    pending_effects: Vec<NodeId>,
    batch_depth: u32,
}

impl Inner {
    fn node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(id).and_then(|slot| slot.as_ref())
    }

    fn node_mut(&mut self, id: NodeId) -> Option<&mut Node> {
        self.nodes.get_mut(id).and_then(|slot| slot.as_mut())
    }

    /// Mark `id` (and, when first dirtied, its observers) stale. Pure graph
    /// bookkeeping: never runs user code, so it may safely recurse while the
    /// runtime is borrowed.
    fn stale(&mut self, id: NodeId, target: State) {
        let (was_clean, is_effect, observers) = match self.node_mut(id) {
            Some(node) => {
                if !node.state.escalates_to(target) {
                    return;
                }
                let was_clean = node.state == State::Clean;
                node.state = target;
                let observers = if was_clean {
                    node.observers.clone()
                } else {
                    Vec::new()
                };
                (was_clean, node.kind == NodeKind::Effect, observers)
            }
            None => return,
        };

        if is_effect && was_clean {
            self.pending_effects.push(id);
        }
        if was_clean {
            for observer in observers {
                self.stale(observer, State::Check);
            }
        }
    }
}

/// A cheap, clonable handle to a reactive graph. All signals, memos, and effects
/// created from the same runtime share one dependency graph.
///
/// The runtime is single-threaded by design (UI reactivity is inherently
/// single-threaded); it is neither `Send` nor `Sync`.
#[derive(Clone)]
pub struct Runtime(pub(crate) Rc<RefCell<Inner>>);

impl Default for Runtime {
    fn default() -> Self {
        Self::new()
    }
}

impl Runtime {
    /// Create an empty runtime.
    pub fn new() -> Self {
        Runtime(Rc::new(RefCell::new(Inner {
            nodes: Vec::new(),
            free: Vec::new(),
            observer_stack: Vec::new(),
            pending_effects: Vec::new(),
            batch_depth: 0,
        })))
    }

    /// Number of live nodes. Primarily useful for leak assertions in tests.
    pub fn live_nodes(&self) -> usize {
        let inner = self.0.borrow();
        inner.nodes.iter().filter(|slot| slot.is_some()).count()
    }

    pub(crate) fn insert_node(
        &self,
        kind: NodeKind,
        state: State,
        computation: Option<Computation>,
    ) -> NodeId {
        let mut inner = self.0.borrow_mut();
        let node = Node::new(kind, state, computation);
        if let Some(id) = inner.free.pop() {
            inner.nodes[id] = Some(node);
            id
        } else {
            inner.nodes.push(Some(node));
            inner.nodes.len() - 1
        }
    }

    /// Record that the current observer (if any) depends on `source`.
    pub(crate) fn track(&self, source: NodeId) {
        let mut inner = self.0.borrow_mut();
        let observer = match inner.observer_stack.last().copied() {
            Some(observer) if observer != NO_OBSERVER && observer != source => observer,
            _ => return,
        };
        let already_linked = inner
            .node(observer)
            .map(|node| node.sources.contains(&source))
            .unwrap_or(true);
        if already_linked {
            return;
        }
        if let Some(node) = inner.node_mut(observer) {
            node.sources.push(source);
        }
        if let Some(node) = inner.node_mut(source)
            && !node.observers.contains(&observer)
        {
            node.observers.push(observer);
        }
    }

    /// Write path: propagate staleness from a changed signal, then flush effects
    /// unless we are inside a batch.
    pub(crate) fn notify_changed(&self, source: NodeId) {
        {
            let mut inner = self.0.borrow_mut();
            let observers = inner
                .node(source)
                .map(|node| node.observers.clone())
                .unwrap_or_default();
            for observer in observers {
                inner.stale(observer, State::Dirty);
            }
        }
        if self.0.borrow().batch_depth == 0 {
            self.flush_effects();
        }
    }

    /// Read path: ensure a derived node is up to date before its value is read.
    pub(crate) fn update_if_necessary(&self, id: NodeId) {
        let (needs_check, sources) = match self.0.borrow().node(id) {
            Some(node) if node.state == State::Check => (true, node.sources.clone()),
            Some(_) => (false, Vec::new()),
            None => return,
        };

        if needs_check {
            for source in sources {
                self.update_if_necessary(source);
                if self.state_of(id) == Some(State::Dirty) {
                    break;
                }
            }
        }

        if self.state_of(id) == Some(State::Dirty) {
            self.run_computation(id);
        }

        if let Some(node) = self.0.borrow_mut().node_mut(id) {
            node.state = State::Clean;
        }
    }

    fn state_of(&self, id: NodeId) -> Option<State> {
        self.0.borrow().node(id).map(|node| node.state)
    }

    /// Recompute a memo/effect, re-collecting its dependencies, and dirty its
    /// observers if its output actually changed.
    fn run_computation(&self, id: NodeId) {
        let computation = {
            let mut inner = self.0.borrow_mut();
            let old_sources = match inner.node_mut(id) {
                Some(node) => core::mem::take(&mut node.sources),
                None => return,
            };
            for source in &old_sources {
                if let Some(source_node) = inner.node_mut(*source) {
                    source_node.observers.retain(|observer| *observer != id);
                }
            }
            inner.observer_stack.push(id);
            inner.node(id).and_then(|node| node.computation.clone())
        };

        let changed = match computation {
            Some(computation) => {
                let mut call = computation.borrow_mut();
                (*call)()
            }
            None => false,
        };

        let mut inner = self.0.borrow_mut();
        inner.observer_stack.pop();
        if changed {
            let observers = inner
                .node(id)
                .map(|node| node.observers.clone())
                .unwrap_or_default();
            for observer in observers {
                if let Some(node) = inner.node_mut(observer) {
                    node.state = State::Dirty;
                }
            }
        }
    }

    /// Drain the queued, stale effects until the graph is quiescent.
    pub(crate) fn flush_effects(&self) {
        let mut iterations = 0usize;
        loop {
            let next = {
                let mut inner = self.0.borrow_mut();
                if inner.pending_effects.is_empty() {
                    None
                } else {
                    Some(inner.pending_effects.remove(0))
                }
            };
            match next {
                Some(id) => self.update_if_necessary(id),
                None => break,
            }
            iterations += 1;
            assert!(
                iterations < MAX_FLUSH_ITERATIONS,
                "prism_ui_reactive: effect flush did not converge (cyclic update?)"
            );
        }
    }

    /// Group multiple writes so observers see a single, consistent update and
    /// effects run at most once at the end. Batches may nest; effects flush when
    /// the outermost batch completes.
    pub fn batch<R>(&self, f: impl FnOnce() -> R) -> R {
        self.0.borrow_mut().batch_depth += 1;
        let result = f();
        let depth = {
            let mut inner = self.0.borrow_mut();
            inner.batch_depth -= 1;
            inner.batch_depth
        };
        if depth == 0 {
            self.flush_effects();
        }
        result
    }

    /// Run `f` without recording dependencies on anything it reads.
    pub fn untrack<R>(&self, f: impl FnOnce() -> R) -> R {
        self.0.borrow_mut().observer_stack.push(NO_OBSERVER);
        let result = f();
        self.0.borrow_mut().observer_stack.pop();
        result
    }

    /// Dispose a node, detaching it from both ends of every edge and freeing its
    /// slot for reuse. Idempotent.
    pub(crate) fn dispose(&self, id: NodeId) {
        let mut inner = self.0.borrow_mut();
        let (sources, observers) = match inner.node(id) {
            Some(node) => (node.sources.clone(), node.observers.clone()),
            None => return,
        };
        for source in sources {
            if let Some(node) = inner.node_mut(source) {
                node.observers.retain(|observer| *observer != id);
            }
        }
        for observer in observers {
            if let Some(node) = inner.node_mut(observer) {
                node.sources.retain(|source| *source != id);
            }
        }
        inner.pending_effects.retain(|pending| *pending != id);
        inner.nodes[id] = None;
        inner.free.push(id);
    }
}
