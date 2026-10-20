use prism_asset_build_core::{BuildError, BuildGraph, BuildJob, JobId};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

#[test]
fn parallel_consumers_borrow_the_same_dependency_allocation() {
    let mut graph = BuildGraph::new();
    graph
        .add_job(BuildJob::new(JobId(1), vec![], |_: &[&[u8]]| {
            Ok(vec![37; 1024 * 1024])
        }))
        .unwrap();
    for id in 2..18 {
        graph
            .add_job(BuildJob::new(
                JobId(id),
                vec![JobId(1), JobId(1)],
                |inputs: &[&[u8]]| {
                    assert_eq!(inputs.len(), 2);
                    assert_eq!(inputs[0].as_ptr(), inputs[1].as_ptr());
                    assert_eq!(inputs[0].len(), 1024 * 1024);
                    assert!(inputs[0].iter().all(|&byte| byte == 37));
                    Ok((inputs[0].as_ptr() as usize).to_le_bytes().to_vec())
                },
            ))
            .unwrap();
    }
    let outputs = graph.execute_parallel_with_limit(4).unwrap();
    let expected = (outputs[&JobId(1)].as_ptr() as usize).to_le_bytes();
    for id in 2..18 {
        assert_eq!(outputs[&JobId(id)], expected);
    }
}

#[test]
fn parallel_panic_preserves_the_actual_failing_job() {
    let mut graph = BuildGraph::new();
    for id in [9, 4] {
        graph
            .add_job(BuildJob::new(JobId(id), vec![], |_: &[&[u8]]| {
                panic!("controlled processor panic")
            }))
            .unwrap();
    }
    match graph.execute_parallel_with_limit(2) {
        Err(BuildError::ProcessorFailed { job, source }) => {
            assert_eq!(job, JobId(4));
            assert_eq!(source.message(), "processor thread panicked");
        }
        other => panic!("unexpected result: {other:?}"),
    }
}

#[test]
fn invalid_graph_runs_no_ready_prefix() {
    let mut graph = BuildGraph::new();
    let calls = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&calls);
    graph
        .add_job(BuildJob::new(JobId(1), vec![], move |_: &[&[u8]]| {
            observed.fetch_add(1, Ordering::SeqCst);
            Ok(vec![])
        }))
        .unwrap();
    for (id, dependency) in [(2, 3), (3, 2)] {
        graph
            .add_job(BuildJob::new(
                JobId(id),
                vec![JobId(dependency)],
                |_: &[&[u8]]| Ok(vec![]),
            ))
            .unwrap();
    }
    assert!(matches!(
        graph.execute_parallel_with_limit(2),
        Err(BuildError::Cycle { .. })
    ));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[test]
fn hundred_thousand_node_chain_plans_and_invalidates_without_recursion() {
    const COUNT: u128 = 100_000;
    let mut graph = BuildGraph::new();
    // Reverse insertion challenges stable ordering as well as deep traversal.
    for id in (0..COUNT).rev() {
        let dependencies = if id == 0 { vec![] } else { vec![JobId(id - 1)] };
        graph
            .add_job(BuildJob::new(JobId(id), dependencies, |_: &[&[u8]]| {
                Ok(vec![])
            }))
            .unwrap();
    }
    let order = graph.execution_order().unwrap();
    assert_eq!(order.len(), COUNT as usize);
    assert!(order
        .iter()
        .enumerate()
        .all(|(index, job)| job.0 == index as u128));
    let dirty = graph.dirty_set([JobId(COUNT / 2)]).unwrap();
    assert_eq!(dirty.len(), (COUNT / 2) as usize);
    let selected = graph.dirty_execution_order(&dirty).unwrap();
    assert_eq!(selected, order[(COUNT / 2) as usize..]);
}
