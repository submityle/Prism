//! Tests for the heterogeneous sub-job dependency DAG
//! ([`JobDag`](crate::system::job_dag::JobDag), design §8.3) and the
//! [`JobGraph::dag`](crate::system::job_graph::JobGraph::dag) system-param entry
//! point.
//!
//! The scheduling is pure dataflow, so results are asserted independent of
//! worker count or steal order: for every declared edge `pred → node`, `pred`
//! must appear before `node` in a mutex-serialised completion timeline, and
//! every node must run exactly once.
//!
//! Enabled only under `cfg(all(test, feature = "multi_thread"))`.

use alloc::vec::Vec;
use core::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use prism_tasks::TaskPool;

use crate::system::job_dag::JobDag;
use crate::system::job_graph::{ComputeTaskPool, JobGraph};
use crate::system::{IntoSystem, System};
use crate::world::World;

/// Assert `before` finishes ahead of `after` in a completion `timeline`.
fn assert_before(timeline: &[usize], before: usize, after: usize) {
    let bi = timeline.iter().position(|&x| x == before);
    let ai = timeline.iter().position(|&x| x == after);
    let (bi, ai) = (
        bi.unwrap_or_else(|| panic!("node {before} never ran")),
        ai.unwrap_or_else(|| panic!("node {after} never ran")),
    );
    assert!(
        bi < ai,
        "edge {before} -> {after} violated: {before} ran at {bi}, {after} at {ai} (timeline {timeline:?})"
    );
}

/// A diamond `A -> {B, C} -> D` plus a side edge `A -> E`: every node runs once
/// and all four edges' orderings hold, on a real multi-worker pool.
#[test]
fn diamond_dataflow_respects_every_edge() {
    let pool = TaskPool::with_threads(4);
    let timeline: Mutex<Vec<usize>> = Mutex::new(Vec::new());
    let runs: Vec<AtomicUsize> = (0..5).map(|_| AtomicUsize::new(0)).collect();

    let record = |id: usize| {
        runs[id].fetch_add(1, Ordering::Relaxed);
        // A little spin so overlap actually happens on the pool.
        let mut acc = 0u64;
        for i in 0..5_000u64 {
            acc = acc.wrapping_add(i);
        }
        core::hint::black_box(acc);
        timeline.lock().unwrap().push(id);
    };

    let mut dag = JobDag::new();
    let a = dag.add(|| record(0));
    let b = dag.add_after(&[a], || record(1));
    let c = dag.add_after(&[a], || record(2));
    let _d = dag.add_after(&[b, c], || record(3));
    let _e = dag.add_after(&[a], || record(4));
    dag.dispatch(&pool);

    for (i, r) in runs.iter().enumerate() {
        assert_eq!(r.load(Ordering::Relaxed), 1, "node {i} did not run exactly once");
    }
    let tl = timeline.lock().unwrap();
    assert_eq!(tl.len(), 5);
    assert_before(&tl, 0, 1); // A -> B
    assert_before(&tl, 0, 2); // A -> C
    assert_before(&tl, 1, 3); // B -> D
    assert_before(&tl, 2, 3); // C -> D
    assert_before(&tl, 0, 4); // A -> E
}

/// Producers hand typed values to a consumer through `'env` interior
/// mutability; the dependency edge guarantees the consumer sees them.
#[test]
fn edge_provides_happens_before_for_values() {
    let pool = TaskPool::with_threads(3);
    let left: Mutex<Option<i64>> = Mutex::new(None);
    let right: Mutex<Option<i64>> = Mutex::new(None);
    let sum: Mutex<Option<i64>> = Mutex::new(None);

    let mut dag = JobDag::new();
    let l = dag.add(|| *left.lock().unwrap() = Some(20));
    let r = dag.add(|| *right.lock().unwrap() = Some(22));
    let _s = dag.add_after(&[l, r], || {
        let a = left.lock().unwrap().expect("left producer must have run");
        let b = right.lock().unwrap().expect("right producer must have run");
        *sum.lock().unwrap() = Some(a + b);
    });
    dag.dispatch(&pool);

    assert_eq!(*sum.lock().unwrap(), Some(42));
}

/// A long linear chain built with `add_after` runs in strict order.
#[test]
fn linear_chain_runs_in_order() {
    let pool = TaskPool::with_threads(4);
    let timeline: Mutex<Vec<usize>> = Mutex::new(Vec::new());
    const N: usize = 64;

    let mut dag = JobDag::with_capacity(N);
    let mut prev = dag.add({
        let t = &timeline;
        move || t.lock().unwrap().push(0)
    });
    for id in 1..N {
        let t = &timeline;
        prev = dag.add_after(&[prev], move || t.lock().unwrap().push(id));
    }
    dag.dispatch(&pool);

    let tl = timeline.lock().unwrap();
    assert_eq!(&*tl, &(0..N).collect::<Vec<_>>());
}

/// A wide fan-out: one root, many independent leaves, then one join. All leaves
/// run after the root and before the join.
#[test]
fn fan_out_fan_in() {
    let pool = TaskPool::with_threads(4);
    let timeline: Mutex<Vec<usize>> = Mutex::new(Vec::new());
    const LEAVES: usize = 32;

    let mut dag = JobDag::new();
    let root = dag.add(|| timeline.lock().unwrap().push(0));
    let mut leaves = Vec::new();
    for id in 1..=LEAVES {
        let t = &timeline;
        leaves.push(dag.add_after(&[root], move || t.lock().unwrap().push(id)));
    }
    let join = LEAVES + 1;
    let t = &timeline;
    dag.add_after(&leaves, move || t.lock().unwrap().push(join));
    dag.dispatch(&pool);

    let tl = timeline.lock().unwrap();
    assert_eq!(tl.len(), LEAVES + 2);
    for leaf in 1..=LEAVES {
        assert_before(&tl, 0, leaf);
        assert_before(&tl, leaf, join);
    }
}

/// The single-threaded fallback (a zero-worker pool) runs a valid topological
/// order inline and produces identical results.
#[test]
fn single_threaded_fallback_topological() {
    let pool = TaskPool::with_threads(0);
    assert!(pool.is_single_threaded());
    let timeline: Mutex<Vec<usize>> = Mutex::new(Vec::new());

    let mut dag = JobDag::new();
    let a = dag.add(|| timeline.lock().unwrap().push(0));
    let b = dag.add_after(&[a], || timeline.lock().unwrap().push(1));
    let c = dag.add_after(&[a], || timeline.lock().unwrap().push(2));
    dag.add_after(&[b, c], || timeline.lock().unwrap().push(3));
    dag.dispatch(&pool);

    let tl = timeline.lock().unwrap();
    assert_eq!(tl.len(), 4);
    assert_before(&tl, 0, 1);
    assert_before(&tl, 0, 2);
    assert_before(&tl, 1, 3);
    assert_before(&tl, 2, 3);
}

/// `order` adds extra edges between already-created nodes.
#[test]
fn explicit_order_edges() {
    let pool = TaskPool::with_threads(2);
    let timeline: Mutex<Vec<usize>> = Mutex::new(Vec::new());

    let mut dag = JobDag::new();
    let a = dag.add(|| timeline.lock().unwrap().push(0));
    let b = dag.add(|| timeline.lock().unwrap().push(1));
    // Force b after a even though neither was created with the other as a dep.
    dag.order(a, b);
    dag.dispatch(&pool);

    let tl = timeline.lock().unwrap();
    assert_before(&tl, 0, 1);
}

/// An empty graph is a no-op.
#[test]
fn empty_graph_is_noop() {
    let pool = TaskPool::with_threads(2);
    let dag = JobDag::new();
    assert!(dag.is_empty());
    dag.dispatch(&pool); // must not hang or panic
}

/// Duplicate dependency handles collapse to a single edge (still runs once).
#[test]
fn duplicate_deps_collapse() {
    let pool = TaskPool::with_threads(2);
    let count = AtomicUsize::new(0);
    let mut dag = JobDag::new();
    let a = dag.add(|| {});
    dag.add_after(&[a, a, a], || {
        count.fetch_add(1, Ordering::Relaxed);
    });
    dag.dispatch(&pool);
    assert_eq!(count.load(Ordering::Relaxed), 1);
}

/// A dependency cycle (via `order`) is detected up front and panics instead of
/// deadlocking.
#[test]
#[should_panic(expected = "cycle")]
fn cycle_is_rejected() {
    let pool = TaskPool::with_threads(2);
    let mut dag = JobDag::new();
    let a = dag.add(|| {});
    let b = dag.add_after(&[a], || {});
    dag.order(b, a); // a -> b -> a
    dag.dispatch(&pool);
}

/// A panic in a sub-job is re-raised on the dispatching thread after the graph
/// joins (no pool poisoning).
#[test]
#[should_panic(expected = "boom in sub-job")]
fn sub_job_panic_propagates() {
    let pool = TaskPool::with_threads(4);
    let mut dag = JobDag::new();
    let a = dag.add(|| {});
    dag.add_after(&[a], || panic!("boom in sub-job"));
    dag.dispatch(&pool);
}

/// `add_after` with an out-of-range handle is a programming error caught by a
/// clear assertion.
#[test]
#[should_panic(expected = "not a node of this graph")]
fn foreign_dependency_rejected() {
    let mut other = JobDag::new();
    let foreign = other.add(|| {});

    let mut dag: JobDag = JobDag::new();
    dag.add_after(&[foreign], || {});
}

/// End-to-end through the `JobGraph` system param: a system takes `JobGraph`
/// and drives a diamond DAG via `jobs.dag(..)`, mutating disjoint `'env` slots.
#[test]
fn job_graph_dag_entry_point() {
    let mut world = World::new();
    world.insert_resource(ComputeTaskPool(TaskPool::with_threads(4)));

    fn run_dag(jobs: JobGraph) {
        let outs: Vec<AtomicUsize> = (0..4).map(|_| AtomicUsize::new(0)).collect();
        let timeline: Mutex<Vec<usize>> = Mutex::new(Vec::new());
        jobs.dag(|dag| {
            let a = dag.add(|| {
                outs[0].store(1, Ordering::Relaxed);
                timeline.lock().unwrap().push(0);
            });
            let b = dag.add_after(&[a], || {
                outs[1].store(2, Ordering::Relaxed);
                timeline.lock().unwrap().push(1);
            });
            let c = dag.add_after(&[a], || {
                outs[2].store(3, Ordering::Relaxed);
                timeline.lock().unwrap().push(2);
            });
            dag.add_after(&[b, c], || {
                let s = outs[0].load(Ordering::Relaxed)
                    + outs[1].load(Ordering::Relaxed)
                    + outs[2].load(Ordering::Relaxed);
                outs[3].store(s, Ordering::Relaxed);
                timeline.lock().unwrap().push(3);
            });
        });
        assert_eq!(outs[3].load(Ordering::Relaxed), 1 + 2 + 3);
        let tl = timeline.lock().unwrap();
        assert_before(&tl, 0, 1);
        assert_before(&tl, 0, 2);
        assert_before(&tl, 1, 3);
        assert_before(&tl, 2, 3);
    }

    let mut sys = IntoSystem::into_system(run_dag);
    sys.initialize(&mut world);
    sys.run(&mut world);
}
