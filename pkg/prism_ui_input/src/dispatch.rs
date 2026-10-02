//! Capture / target / bubble event dispatch.
//!
//! Given a root-to-target path (as produced by
//! [`hit_test`](crate::hit_test::hit_test)), the [`Dispatcher`] delivers a
//! [`PointerEvent`] in three phases:
//!
//! 1. **Capture** — ancestors from the root down to the target's parent, in
//!    order, running handlers registered for [`Phase::Capture`].
//! 2. **Target** — the target node, running handlers registered for
//!    [`Phase::Target`].
//! 3. **Bubble** — ancestors from the target's parent back up to the root, in
//!    reverse order, running handlers registered for [`Phase::Bubble`].
//!
//! Handlers receive a mutable [`EventContext`]. Calling
//! [`EventContext::stop_propagation`] halts delivery to any node not yet
//! visited (the current node's remaining handlers still run), and
//! [`EventContext::prevent_default`] records that the default action should be
//! suppressed. Both flags are surfaced in the returned [`DispatchOutcome`].

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::event::{NodeId, Phase, PointerEvent};

/// Boxed pointer-event handler stored by the dispatcher.
type PointerHandler = Box<dyn FnMut(&mut EventContext, &PointerEvent)>;

/// Mutable state passed to each handler while an event is being dispatched.
pub struct EventContext {
    current: NodeId,
    phase: Phase,
    propagation_stopped: bool,
    default_prevented: bool,
}

impl EventContext {
    /// Creates a context positioned at `current` in `phase`.
    fn new(current: NodeId, phase: Phase) -> Self {
        Self {
            current,
            phase,
            propagation_stopped: false,
            default_prevented: false,
        }
    }

    /// Returns the node whose handler is currently running.
    pub fn current_target(&self) -> NodeId {
        self.current
    }

    /// Returns the phase in which the current handler is running.
    pub fn phase(&self) -> Phase {
        self.phase
    }

    /// Stops delivery to nodes not yet visited.
    pub fn stop_propagation(&mut self) {
        self.propagation_stopped = true;
    }

    /// Returns `true` once [`EventContext::stop_propagation`] has been called.
    pub fn is_propagation_stopped(&self) -> bool {
        self.propagation_stopped
    }

    /// Records that the event's default action should be suppressed.
    pub fn prevent_default(&mut self) {
        self.default_prevented = true;
    }

    /// Returns `true` once [`EventContext::prevent_default`] has been called.
    pub fn is_default_prevented(&self) -> bool {
        self.default_prevented
    }
}

/// Summary of what happened while dispatching one event.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DispatchOutcome {
    /// Whether any handler called [`EventContext::prevent_default`].
    pub default_prevented: bool,
    /// Whether any handler called [`EventContext::stop_propagation`].
    pub propagation_stopped: bool,
    /// Number of handlers that actually ran.
    pub handlers_run: usize,
}

/// Handlers registered for a single node, grouped by phase.
#[derive(Default)]
struct PhaseHandlers {
    capture: Vec<PointerHandler>,
    target: Vec<PointerHandler>,
    bubble: Vec<PointerHandler>,
}

impl PhaseHandlers {
    fn slot(&mut self, phase: Phase) -> &mut Vec<PointerHandler> {
        match phase {
            Phase::Capture => &mut self.capture,
            Phase::Target => &mut self.target,
            Phase::Bubble => &mut self.bubble,
        }
    }
}

/// Registers handlers per node and delivers events through the three phases.
#[derive(Default)]
pub struct Dispatcher {
    nodes: BTreeMap<NodeId, PhaseHandlers>,
}

impl Dispatcher {
    /// Creates an empty dispatcher.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers `handler` on `node` for `phase`.
    ///
    /// Multiple handlers may be registered for the same node and phase; they
    /// run in registration order.
    pub fn on<F>(&mut self, node: NodeId, phase: Phase, handler: F)
    where
        F: FnMut(&mut EventContext, &PointerEvent) + 'static,
    {
        self.nodes
            .entry(node)
            .or_default()
            .slot(phase)
            .push(Box::new(handler));
    }

    /// Returns the number of nodes that have at least one handler.
    pub fn registered_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Dispatches `event` along `path` (root first, target last).
    ///
    /// Returns a [`DispatchOutcome`] describing which flags were set and how
    /// many handlers ran. An empty path runs no handlers.
    pub fn dispatch(&mut self, path: &[NodeId], event: &PointerEvent) -> DispatchOutcome {
        let mut outcome = DispatchOutcome::default();
        if path.is_empty() {
            return outcome;
        }
        let target_index = path.len() - 1;

        // Capture: root down to the target's parent.
        for &node in &path[..target_index] {
            if self.run_node(node, Phase::Capture, event, &mut outcome) {
                return outcome;
            }
        }

        // Target.
        if self.run_node(path[target_index], Phase::Target, event, &mut outcome) {
            return outcome;
        }

        // Bubble: target's parent up to the root.
        for &node in path[..target_index].iter().rev() {
            if self.run_node(node, Phase::Bubble, event, &mut outcome) {
                return outcome;
            }
        }

        outcome
    }

    /// Runs the handlers for one node in one phase.
    ///
    /// Returns `true` when propagation was stopped and dispatch should halt.
    fn run_node(
        &mut self,
        node: NodeId,
        phase: Phase,
        event: &PointerEvent,
        outcome: &mut DispatchOutcome,
    ) -> bool {
        let mut ctx = EventContext::new(node, phase);
        ctx.propagation_stopped = outcome.propagation_stopped;
        ctx.default_prevented = outcome.default_prevented;

        if let Some(handlers) = self.nodes.get_mut(&node) {
            for handler in handlers.slot(phase).iter_mut() {
                handler(&mut ctx, event);
                outcome.handlers_run += 1;
            }
        }

        if ctx.default_prevented {
            outcome.default_prevented = true;
        }
        if ctx.propagation_stopped {
            outcome.propagation_stopped = true;
            return true;
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::{PointerId, PointerKind};
    use crate::geometry::Point;
    use alloc::rc::Rc;
    use alloc::vec;
    use core::cell::RefCell;

    fn event() -> PointerEvent {
        PointerEvent::new(
            PointerId::new(1),
            PointerKind::Down,
            Point::new(0.0, 0.0),
            0,
        )
    }

    #[test]
    fn phases_run_in_order() {
        let log: Rc<RefCell<Vec<(NodeId, Phase)>>> = Rc::new(RefCell::new(Vec::new()));
        let mut d = Dispatcher::new();
        let path = [NodeId::new(1), NodeId::new(2), NodeId::new(3)];
        for &n in &path {
            for phase in [Phase::Capture, Phase::Target, Phase::Bubble] {
                let log = Rc::clone(&log);
                d.on(n, phase, move |ctx, _ev| {
                    log.borrow_mut().push((ctx.current_target(), ctx.phase()));
                });
            }
        }
        d.dispatch(&path, &event());
        let seen = log.borrow().clone();
        assert_eq!(
            seen,
            vec![
                (NodeId::new(1), Phase::Capture),
                (NodeId::new(2), Phase::Capture),
                (NodeId::new(3), Phase::Target),
                (NodeId::new(2), Phase::Bubble),
                (NodeId::new(1), Phase::Bubble),
            ]
        );
    }

    #[test]
    fn stop_propagation_halts_remaining_nodes() {
        let count = Rc::new(RefCell::new(0usize));
        let mut d = Dispatcher::new();
        let path = [NodeId::new(1), NodeId::new(2), NodeId::new(3)];
        // Capture on the root stops propagation.
        d.on(NodeId::new(1), Phase::Capture, |ctx, _| {
            ctx.stop_propagation();
        });
        for &n in &path {
            let count = Rc::clone(&count);
            d.on(n, Phase::Target, move |_ctx, _| *count.borrow_mut() += 1);
        }
        let outcome = d.dispatch(&path, &event());
        assert!(outcome.propagation_stopped);
        assert_eq!(*count.borrow(), 0);
    }

    #[test]
    fn prevent_default_is_reported() {
        let mut d = Dispatcher::new();
        d.on(NodeId::new(1), Phase::Target, |ctx, _| {
            ctx.prevent_default();
        });
        let outcome = d.dispatch(&[NodeId::new(1)], &event());
        assert!(outcome.default_prevented);
        assert!(!outcome.propagation_stopped);
        assert_eq!(outcome.handlers_run, 1);
    }

    #[test]
    fn empty_path_runs_nothing() {
        let mut d = Dispatcher::new();
        d.on(NodeId::new(1), Phase::Target, |_ctx, _| {});
        let outcome = d.dispatch(&[], &event());
        assert_eq!(outcome.handlers_run, 0);
        assert_eq!(d.registered_nodes(), 1);
    }
}
