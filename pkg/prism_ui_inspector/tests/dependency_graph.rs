//! End-to-end integration tests for the reactive dependency-graph inspector.
//!
//! These drive a real `prism_ui_reactive::Runtime`, snapshot its graph, and
//! assert that `DependencyGraph` reports the expected structure.

extern crate alloc;

use alloc::rc::Rc;
use core::cell::RefCell;

use prism_ui_inspector::DependencyGraph;
use prism_ui_reactive::{NodeKindInfo, Runtime};

#[test]
fn chain_dependents_topo_and_dot() {
    let rt = Runtime::new();
    let count = rt.signal(1i32);
    let doubled = rt.memo({
        let count = count.clone();
        move || count.get() * 2
    });
    let log = Rc::new(RefCell::new(Vec::new()));
    let _effect = rt.effect({
        let doubled = doubled.clone();
        let log = log.clone();
        move || log.borrow_mut().push(doubled.get())
    });

    let snap = rt.graph_snapshot();
    let graph = DependencyGraph::from(&snap);

    let signal_id = snap.signals()[0].id;
    let memo_id = snap.memos()[0].id;
    let effect_id = snap.effects()[0].id;

    // The signal has exactly three nodes and the chain shape.
    assert_eq!(graph.node_count(), 3);
    assert_eq!(graph.kind_of(signal_id), Some(NodeKindInfo::Signal));
    assert_eq!(graph.kind_of(memo_id), Some(NodeKindInfo::Memo));
    assert_eq!(graph.kind_of(effect_id), Some(NodeKindInfo::Effect));

    // Everything downstream of the signal; everything upstream of the effect.
    let mut down = graph.dependents_of(signal_id);
    down.sort_unstable();
    let mut expected_down = vec![memo_id, effect_id];
    expected_down.sort_unstable();
    assert_eq!(down, expected_down);

    let mut up = graph.dependencies_of(effect_id);
    up.sort_unstable();
    let mut expected_up = vec![signal_id, memo_id];
    expected_up.sort_unstable();
    assert_eq!(up, expected_up);

    // Roots are signals; leaves are effects.
    assert_eq!(graph.roots(), vec![signal_id]);
    assert_eq!(graph.leaves(), vec![effect_id]);

    // Topological order respects dependencies.
    let order = graph.topo_order().expect("acyclic graph");
    let pos = |id| order.iter().position(|x| *x == id).unwrap();
    assert!(pos(signal_id) < pos(memo_id));
    assert!(pos(memo_id) < pos(effect_id));

    // DOT output names each kind and carries the signal -> memo edge.
    let dot = graph.to_dot();
    assert!(dot.starts_with("digraph reactive {"));
    assert!(dot.contains("Signal"));
    assert!(dot.contains("Memo"));
    assert!(dot.contains("Effect"));
    assert!(dot.contains(&format!("n{signal_id} -> n{memo_id};")));
}

#[test]
fn diamond_topology_from_runtime() {
    let rt = Runtime::new();
    let base = rt.signal(10i32);
    let left = rt.memo({
        let base = base.clone();
        move || base.get() + 1
    });
    let right = rt.memo({
        let base = base.clone();
        move || base.get() - 1
    });
    let sum = rt.memo({
        let left = left.clone();
        let right = right.clone();
        move || left.get() + right.get()
    });
    // Force the memos to subscribe to their sources.
    assert_eq!(sum.get(), 20);

    let snap = rt.graph_snapshot();
    let graph = DependencyGraph::from(&snap);
    assert_eq!(graph.node_count(), 4);

    let base_id = snap.signals()[0].id;
    // The signal feeds both branches and (transitively) the join.
    assert_eq!(graph.dependents_of(base_id).len(), 3);

    let sum_id = snap.memos().iter().map(|n| n.id).max().expect("has memos");
    // The join depends on everything else.
    assert_eq!(graph.dependencies_of(sum_id).len(), 3);

    let order = graph.topo_order().expect("acyclic graph");
    let pos = |id| order.iter().position(|x| *x == id).unwrap();
    // The signal comes first and the join comes last.
    assert_eq!(pos(base_id), 0);
    assert_eq!(pos(sum_id), order.len() - 1);
}

#[test]
fn render_reflects_runtime_graph() {
    let rt = Runtime::new();
    let a = rt.signal(0i32);
    let b = rt.memo({
        let a = a.clone();
        move || a.get() + 1
    });
    let _ = b.get();

    let snap = rt.graph_snapshot();
    let graph = DependencyGraph::from(&snap);

    let a_id = snap.signals()[0].id;
    let b_id = snap.memos()[0].id;

    let text = graph.render();
    assert!(text.contains(&format!("#{a_id} Signal")));
    assert!(text.contains(&format!("  #{b_id} Memo")));

    // Dependents/dependencies agree across the single edge.
    assert_eq!(graph.dependents_of(a_id), vec![b_id]);
    assert_eq!(graph.dependencies_of(b_id), vec![a_id]);
}
