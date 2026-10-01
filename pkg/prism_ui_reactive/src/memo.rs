//! Lazily recomputed, cached derived values.

use alloc::rc::Rc;
use core::cell::RefCell;

use crate::node::{NodeId, NodeKind, State};
use crate::runtime::Runtime;

/// A derived value computed from other reactive sources.
///
/// A memo recomputes lazily — only when read after one of its sources changed —
/// and caches the result. Because the output is compared with [`PartialEq`],
/// downstream dependents are only disturbed when the memo's value actually
/// changes, which prunes redundant propagation.
pub struct Memo<T: 'static> {
    runtime: Runtime,
    id: NodeId,
    value: Rc<RefCell<Option<T>>>,
}

impl<T: 'static> Clone for Memo<T> {
    fn clone(&self) -> Self {
        Self {
            runtime: self.runtime.clone(),
            id: self.id,
            value: self.value.clone(),
        }
    }
}

impl<T: PartialEq + Clone + 'static> Memo<T> {
    pub(crate) fn new(runtime: &Runtime, mut compute: impl FnMut() -> T + 'static) -> Self {
        let value: Rc<RefCell<Option<T>>> = Rc::new(RefCell::new(None));
        let value_in_comp = value.clone();
        let computation = Rc::new(RefCell::new(move || {
            let next = compute();
            let mut slot = value_in_comp.borrow_mut();
            let changed = match slot.as_ref() {
                Some(previous) => *previous != next,
                None => true,
            };
            if changed {
                *slot = Some(next);
            }
            changed
        }));
        // Memos start dirty and are evaluated on first read.
        let id = runtime.insert_node(NodeKind::Memo, State::Dirty, Some(computation));
        Self {
            runtime: runtime.clone(),
            id,
            value,
        }
    }

    /// Read the memo, recomputing on demand if a source changed, and record a
    /// dependency on it.
    pub fn get(&self) -> T {
        self.runtime.track(self.id);
        self.runtime.update_if_necessary(self.id);
        self.value
            .borrow()
            .clone()
            .expect("memo must hold a value after evaluation")
    }

    /// Borrow the up-to-date value through `f`, recording a dependency.
    pub fn with<R>(&self, f: impl FnOnce(&T) -> R) -> R {
        self.runtime.track(self.id);
        self.runtime.update_if_necessary(self.id);
        f(self
            .value
            .borrow()
            .as_ref()
            .expect("memo must hold a value after evaluation"))
    }

    /// Dispose the memo, detaching it from the dependency graph.
    pub fn dispose(self) {
        self.runtime.dispose(self.id);
    }
}

impl Runtime {
    /// Create a [`Memo`] from a pure computation over reactive sources.
    pub fn memo<T: PartialEq + Clone + 'static>(
        &self,
        compute: impl FnMut() -> T + 'static,
    ) -> Memo<T> {
        Memo::new(self, compute)
    }
}
