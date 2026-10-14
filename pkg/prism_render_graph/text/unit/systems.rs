//! Schedule vocabulary regressions; retained intact from the original module.

use super::{configure_render_graph_sets, RenderGraphSystems};
use alloc::vec;
use alloc::vec::Vec;

/// The order marker systems run in, given how they were registered and the
/// set policy they hang off.
///
/// Mirrors `prism_ecs`'s ordering model: a set-ordering edge expands to an
/// edge between every member system, and the topological sort breaks ties
/// by registration order (its documented determinism guarantee). Only
/// [`RenderGraphSystems::ORDER_EDGES`] is consulted for precedence, so a
/// lost policy edge shows up as a wrong run order rather than passing by
/// accident.
fn resolve_run_order(registrations: &[(RenderGraphSystems, &'static str)]) -> Vec<&'static str> {
    let node_count = registrations.len();
    let mut in_degree = vec![0usize; node_count];
    let mut edges: Vec<Vec<usize>> = vec![Vec::new(); node_count];
    for (index, (before, _)) in registrations.iter().enumerate() {
        for (other, (after, _)) in registrations.iter().enumerate() {
            if index != other && precedes(*before, *after) {
                edges[index].push(other);
                in_degree[other] += 1;
            }
        }
    }

    let mut order = Vec::with_capacity(node_count);
    let mut emitted = vec![false; node_count];
    for _ in 0..node_count {
        let next = (0..node_count)
            .find(|&index| !emitted[index] && in_degree[index] == 0)
            .expect("the set policy is acyclic");
        emitted[next] = true;
        order.push(registrations[next].1);
        for &dependent in &edges[next] {
            in_degree[dependent] -= 1;
        }
    }
    order
}

/// Whether the set policy orders `before` ahead of `after` — the transitive
/// closure of [`RenderGraphSystems::ORDER_EDGES`].
fn precedes(before: RenderGraphSystems, after: RenderGraphSystems) -> bool {
    let mut frontier = vec![before];
    let mut seen = vec![before];
    while let Some(current) = frontier.pop() {
        for (from, to) in RenderGraphSystems::ORDER_EDGES {
            if from == current && !seen.contains(&to) {
                if to == after {
                    return true;
                }
                seen.push(to);
                frontier.push(to);
            }
        }
    }
    false
}

#[test]
fn order_and_edges_are_the_weak_chain_begin_to_finish() {
    assert_eq!(
        RenderGraphSystems::ORDER,
        [
            RenderGraphSystems::Begin,
            RenderGraphSystems::Render,
            RenderGraphSystems::Submit,
            RenderGraphSystems::Finish,
        ],
        "the sets run Begin → Render → Submit → Finish"
    );
    assert_eq!(
        RenderGraphSystems::ORDER_EDGES,
        [
            (RenderGraphSystems::Begin, RenderGraphSystems::Render),
            (RenderGraphSystems::Render, RenderGraphSystems::Submit),
            (RenderGraphSystems::Submit, RenderGraphSystems::Finish),
        ],
        "the policy is exactly the weak chain over ORDER"
    );
}

#[test]
fn render_graph_sets_impose_their_order_when_registered_out_of_order() {
    // Registered out of order across sets (and, within a set, in the order
    // the schedule keeps: registration order is the tie-break).
    let registrations = [
        (RenderGraphSystems::Finish, "map-readback"),
        (RenderGraphSystems::Submit, "submit-pending-buffers"),
        (RenderGraphSystems::Render, "record-graph"),
        (RenderGraphSystems::Begin, "reset-frame-scratch"),
        (RenderGraphSystems::Finish, "reclaim-completed-handles"),
        (RenderGraphSystems::Begin, "collect-last-frame-readback"),
    ];

    assert_eq!(
        resolve_run_order(&registrations),
        [
            "reset-frame-scratch",
            "collect-last-frame-readback",
            "record-graph",
            "submit-pending-buffers",
            "map-readback",
            "reclaim-completed-handles",
        ],
        "per-frame setup runs first, then the graph records, then the buffers \
             submit, then finalization — whatever order they were registered in"
    );
}

#[test]
fn configure_emits_every_ordering_edge_in_order() {
    let mut applied = Vec::new();
    configure_render_graph_sets(|before, after| applied.push((before, after)));
    assert_eq!(applied.as_slice(), RenderGraphSystems::ORDER_EDGES);
}
