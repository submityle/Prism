//! Side effects that re-run when their reactive dependencies change.

use alloc::rc::Rc;
use core::cell::RefCell;

use crate::node::{NodeId, NodeKind, State};
use crate::runtime::Runtime;

/// A handle to a reactive effect.
///
/// The effect runs once immediately (to collect its dependencies) and re-runs
/// whenever any dependency changes. Dropping the handle does **not** stop the
/// effect; call [`Effect::dispose`] to detach it from the graph.
pub struct Effect {
    runtime: Runtime,
    id: NodeId,
}

impl Effect {
    pub(crate) fn new(runtime: &Runtime, mut run: impl FnMut() + 'static) -> Self {
        let computation = Rc::new(RefCell::new(move || {
            run();
            false
        }));
        let id = runtime.insert_node(NodeKind::Effect, State::Dirty, Some(computation));
        let effect = Self {
            runtime: runtime.clone(),
            id,
        };
        // Run eagerly so dependencies are captured straight away.
        runtime.update_if_necessary(id);
        effect
    }

    /// Dispose the effect so it no longer re-runs.
    pub fn dispose(self) {
        self.runtime.dispose(self.id);
    }
}

impl Runtime {
    /// Create an [`Effect`] that runs now and re-runs on dependency changes.
    pub fn effect(&self, run: impl FnMut() + 'static) -> Effect {
        Effect::new(self, run)
    }
}
