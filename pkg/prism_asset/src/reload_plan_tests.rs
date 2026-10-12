use alloc::vec;
use alloc::vec::Vec;

use super::{
    AssetIndex, AssetTypeId, DependencyGraph, ReloadPlan, ReloadPlanError, UntypedAssetId,
};

fn asset(index: u32) -> UntypedAssetId {
    UntypedAssetId::new(
        AssetIndex::from_parts(index, 0),
        AssetTypeId::of_name("reload-plan-tests"),
    )
}

fn plan(graph: &DependencyGraph, roots: &[u32]) -> Vec<UntypedAssetId> {
    ReloadPlan::build(graph, roots.iter().copied().map(asset))
        .expect("reload graph should be acyclic")
        .into_order()
}

#[test]
fn plans_transitive_dependencies_before_dependents() {
    let mut graph = DependencyGraph::new();
    graph.add_dependency(asset(40), asset(10));
    graph.add_dependency(asset(40), asset(20));
    graph.add_dependency(asset(20), asset(5));

    assert_eq!(
        plan(&graph, &[40]),
        [asset(5), asset(10), asset(20), asset(40)]
    );
}

#[test]
fn merges_and_deduplicates_multiple_roots() {
    let mut graph = DependencyGraph::new();
    graph.add_dependency(asset(30), asset(10));
    graph.add_dependency(asset(40), asset(10));
    graph.add_dependency(asset(40), asset(20));

    assert_eq!(
        plan(&graph, &[40, 30, 40]),
        [asset(10), asset(20), asset(30), asset(40)]
    );
}

#[test]
fn ordering_is_stable_for_root_and_edge_insertion_order() {
    let mut first = DependencyGraph::new();
    first.add_dependency(asset(9), asset(4));
    first.add_dependency(asset(9), asset(1));
    first.add_dependency(asset(8), asset(2));

    let mut second = DependencyGraph::new();
    second.add_dependency(asset(8), asset(2));
    second.add_dependency(asset(9), asset(1));
    second.add_dependency(asset(9), asset(4));

    let first_order = plan(&first, &[9, 8]);
    let second_order = plan(&second, &[8, 9, 8]);
    assert_eq!(
        first_order,
        [asset(1), asset(2), asset(4), asset(8), asset(9)]
    );
    assert_eq!(first_order, second_order);
}

#[test]
fn empty_roots_produce_empty_plan_even_for_a_nonempty_graph() {
    let mut graph = DependencyGraph::new();
    graph.add_dependency(asset(2), asset(1));

    let plan = ReloadPlan::build(&graph, core::iter::empty()).expect("empty roots are valid");
    assert!(plan.is_empty());
    assert_eq!(plan.len(), 0);
    assert_eq!(plan.as_slice(), &[]);
}

#[test]
fn an_unregistered_root_is_an_isolated_reload_target() {
    let graph = DependencyGraph::new();
    assert_eq!(plan(&graph, &[77]), [asset(77)]);
}

#[test]
fn reports_cycle_participants_in_stable_order() {
    let mut graph = DependencyGraph::new();
    graph.add_dependency(asset(30), asset(10));
    graph.add_dependency(asset(10), asset(20));
    graph.add_dependency(asset(20), asset(30));

    let error = ReloadPlan::build(&graph, [asset(30)]).expect_err("cycle must be rejected");
    assert_eq!(
        error,
        ReloadPlanError::Cycle {
            participants: vec![asset(10), asset(20), asset(30)],
        }
    );
}

#[test]
fn excludes_unrelated_graph_nodes() {
    let mut graph = DependencyGraph::new();
    graph.add_dependency(asset(4), asset(3));
    graph.add_dependency(asset(99), asset(98));

    assert_eq!(plan(&graph, &[4]), [asset(3), asset(4)]);
}
