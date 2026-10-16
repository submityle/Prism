use prism_asset_build_core::{BuildGraph, BuildJob, DirtySetError, JobId};

fn job(id: u128, dependencies: &[u128]) -> BuildJob {
    BuildJob::new(
        JobId(id),
        dependencies.iter().copied().map(JobId).collect::<Vec<_>>(),
        |_: &[&[u8]]| Ok(Vec::new()),
    )
}

#[test]
fn dirty_set_is_exact_transitive_successor_closure() {
    let mut graph = BuildGraph::new();
    graph.add_job(job(4, &[2, 3])).unwrap();
    graph.add_job(job(3, &[1])).unwrap();
    graph.add_job(job(2, &[1])).unwrap();
    graph.add_job(job(1, &[])).unwrap();
    graph.add_job(job(5, &[])).unwrap();

    let dirty = graph.dirty_set([JobId(1)]).unwrap();
    assert_eq!(dirty.direct_jobs().collect::<Vec<_>>(), vec![JobId(1)]);
    assert_eq!(
        dirty.jobs().collect::<Vec<_>>(),
        vec![JobId(1), JobId(2), JobId(3), JobId(4)]
    );
    assert!(!dirty.contains(JobId(5)));
    assert_eq!(
        graph.dirty_execution_order(&dirty).unwrap(),
        vec![JobId(1), JobId(2), JobId(3), JobId(4)]
    );
}

#[test]
fn direct_changes_are_deduplicated_and_order_is_stable() {
    let mut graph = BuildGraph::new();
    graph.add_job(job(3, &[1, 2])).unwrap();
    graph.add_job(job(2, &[])).unwrap();
    graph.add_job(job(1, &[])).unwrap();

    let dirty = graph.dirty_set([JobId(2), JobId(1), JobId(2)]).unwrap();
    assert_eq!(
        dirty.direct_jobs().collect::<Vec<_>>(),
        vec![JobId(1), JobId(2)]
    );
    assert_eq!(
        dirty.jobs().collect::<Vec<_>>(),
        vec![JobId(1), JobId(2), JobId(3)]
    );
}

#[test]
fn unknown_change_and_invalid_graph_fail_closed() {
    let mut graph = BuildGraph::new();
    graph.add_job(job(1, &[])).unwrap();
    assert_eq!(
        graph.dirty_set([JobId(99)]),
        Err(DirtySetError::UnknownJob(JobId(99)))
    );

    let mut cyclic = BuildGraph::new();
    cyclic.add_job(job(1, &[2])).unwrap();
    cyclic.add_job(job(2, &[1])).unwrap();
    assert!(matches!(
        cyclic.dirty_set([JobId(1)]),
        Err(DirtySetError::Graph(_))
    ));
}
