//! Integration tests for the [`JobGraph`](crate::system::job_graph::JobGraph)
//! system param (design §8.3): a system taking a query plus `JobGraph` fans the
//! query's rows out across the world's [`ComputeTaskPool`] and mutates /
//! reads them exactly as a serial pass would.
//!
//! Enabled only under `cfg(all(test, feature = "multi_thread"))`.

use alloc::vec::Vec;
use core::sync::atomic::{AtomicI64, Ordering};

use prism_tasks::TaskPool;

use crate::component::Component;
use crate::system::job_graph::{ComputeTaskPool, JobGraph};
use crate::system::{IntoSystem, Query, System};
use crate::world::World;

#[derive(Debug, PartialEq, Clone, Copy)]
struct Position(i64);
impl Component for Position {}

#[derive(Debug, PartialEq, Clone, Copy)]
struct Frozen;
impl Component for Frozen {}

/// A `(Query<&mut Position>, JobGraph)` system doubles every matched row in
/// parallel; the mutation is visible on the world after the run.
#[test]
fn job_graph_par_for_each_mut_doubles_rows() {
    let mut world = World::new();
    world.insert_resource(ComputeTaskPool(TaskPool::new()));
    for i in 0..500i64 {
        world.spawn(Position(i));
    }

    fn double(mut q: Query<&mut Position>, jobs: JobGraph) {
        jobs.par_for_each_mut(&mut q, 16, |mut p| p.0 *= 2);
    }

    let mut sys = IntoSystem::into_system(double);
    sys.initialize(&mut world);
    sys.run(&mut world);

    let mut got: Vec<i64> = Vec::new();
    world.for_each::<&Position>(|p| got.push(p.0));
    got.sort_unstable();
    let want: Vec<i64> = (0..500i64).map(|i| i * 2).collect();
    assert_eq!(got, want);
}

/// A read-only `JobGraph::par_for_each` visits every matched row exactly once;
/// an atomic running sum equals the serial sum.
#[test]
fn job_graph_par_for_each_read_only_sums_all_rows() {
    let mut world = World::new();
    world.insert_resource(ComputeTaskPool(TaskPool::with_threads(4)));
    let mut expected = 0i64;
    for i in 0..777i64 {
        world.spawn(Position(i));
        expected += i;
    }

    static SUM: AtomicI64 = AtomicI64::new(0);
    SUM.store(0, Ordering::Relaxed);

    fn accumulate(q: Query<&Position>, jobs: JobGraph) {
        jobs.par_for_each(&q, 32, |p| {
            SUM.fetch_add(p.0, Ordering::Relaxed);
        });
    }

    let mut sys = IntoSystem::into_system(accumulate);
    sys.initialize(&mut world);
    sys.run(&mut world);

    assert_eq!(SUM.load(Ordering::Relaxed), expected);
}

/// `JobGraph` honours query filters: a `Without<Frozen>` query skips the frozen
/// rows, leaving them untouched by the parallel mutation.
#[test]
fn job_graph_respects_query_filters() {
    use crate::query::Without;

    let mut world = World::new();
    world.insert_resource(ComputeTaskPool(TaskPool::new()));
    // Even indices are frozen (Position + Frozen), odd indices are plain.
    for i in 0..200i64 {
        if i % 2 == 0 {
            world.spawn((Position(i), Frozen));
        } else {
            world.spawn(Position(i));
        }
    }

    fn bump(mut q: Query<&mut Position, Without<Frozen>>, jobs: JobGraph) {
        jobs.par_for_each_mut(&mut q, 8, |mut p| p.0 += 1000);
    }

    let mut sys = IntoSystem::into_system(bump);
    sys.initialize(&mut world);
    sys.run(&mut world);

    let mut got: Vec<i64> = Vec::new();
    world.for_each::<&Position>(|p| got.push(p.0));
    got.sort_unstable();

    let mut want: Vec<i64> = Vec::new();
    for i in 0..200i64 {
        if i % 2 == 0 {
            want.push(i); // frozen → untouched
        } else {
            want.push(i + 1000); // bumped
        }
    }
    want.sort_unstable();
    assert_eq!(got, want);
}
