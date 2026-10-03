//! M6 tests: the ordered stage/schedule [`Pipeline`](crate::Pipeline).
//!
//! Anti-vacuous contract: stages run strictly in declaration order (observed
//! through a shared log), jobs within a stage all run (intra-stage
//! parallelism), and empty stages are skipped without disturbing ordering.

use crate::TaskPool;
use alloc::string::String;
use alloc::vec::Vec;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

#[test]
fn stages_run_in_declaration_order() {
    let pool = TaskPool::with_threads(4);
    let log: Mutex<Vec<String>> = Mutex::new(Vec::new());
    let mut pipeline = pool.pipeline();
    for name in ["first", "extract", "simulate", "render", "present"] {
        let log = &log;
        pipeline
            .add_stage(name)
            .job(move || log.lock().unwrap().push(String::from(name)));
    }
    pipeline.run();
    let seen = log.into_inner().unwrap();
    assert_eq!(seen, ["first", "extract", "simulate", "render", "present"]);
}

#[test]
fn later_stage_observes_earlier_stage_effects() {
    // Stage N increments a shared counter; stage N+1 asserts it has advanced,
    // which can only hold if stages are fully joined in order.
    let pool = TaskPool::with_threads(4);
    let state = AtomicUsize::new(0);
    let witnessed: Mutex<Vec<usize>> = Mutex::new(Vec::new());
    let mut pipeline = pool.pipeline();
    for step in 0..5usize {
        let state = &state;
        let witnessed = &witnessed;
        pipeline.add_stage("step").job(move || {
            // Observe the fully-settled value from all previous stages, then
            // contribute this stage's increment.
            witnessed.lock().unwrap().push(state.load(Ordering::Acquire));
            state.store(step + 1, Ordering::Release);
        });
    }
    pipeline.run();
    assert_eq!(state.load(Ordering::Acquire), 5);
    assert_eq!(witnessed.into_inner().unwrap(), [0, 1, 2, 3, 4]);
}

#[test]
fn jobs_within_a_stage_all_run() {
    let pool = TaskPool::with_threads(4);
    let count = AtomicUsize::new(0);
    let mut pipeline = pool.pipeline();
    let stage = pipeline.add_stage("fanout");
    for _ in 0..1000 {
        let count = &count;
        stage.job(move || {
            count.fetch_add(1, Ordering::Relaxed);
        });
    }
    assert_eq!(pipeline.stage_count(), 1);
    pipeline.run();
    assert_eq!(count.load(Ordering::Relaxed), 1000);
}

#[test]
fn empty_stages_are_skipped_but_order_is_preserved() {
    let pool = TaskPool::with_threads(2);
    let log: Mutex<Vec<String>> = Mutex::new(Vec::new());
    let mut pipeline = pool.pipeline();
    pipeline.add_stage("a").job({
        let log = &log;
        move || log.lock().unwrap().push(String::from("a"))
    });
    pipeline.add_stage("empty"); // no jobs
    pipeline.add_stage("b").job({
        let log = &log;
        move || log.lock().unwrap().push(String::from("b"))
    });
    assert_eq!(pipeline.stage_count(), 3);
    assert_eq!(pipeline.stage_names(), ["a", "empty", "b"]);
    pipeline.run();
    assert_eq!(log.into_inner().unwrap(), ["a", "b"]);
}

#[test]
fn empty_pipeline_runs_cleanly() {
    let pool = TaskPool::with_threads(2);
    let pipeline = pool.pipeline();
    assert_eq!(pipeline.stage_count(), 0);
    pipeline.run();
}

#[test]
fn run_traced_records_a_span_per_job() {
    let pool = TaskPool::with_threads(4);
    let trace = pool.new_job_trace();
    let hits = AtomicUsize::new(0);
    let mut pipeline = pool.pipeline();
    pipeline.add_stage("extract").job({
        let hits = &hits;
        move || {
            hits.fetch_add(1, Ordering::Relaxed);
        }
    });
    let simulate = pipeline.add_stage("simulate");
    for _ in 0..7 {
        let hits = &hits;
        simulate.job(move || {
            hits.fetch_add(1, Ordering::Relaxed);
        });
    }
    pipeline.run_traced(&trace);
    assert_eq!(hits.load(Ordering::Relaxed), 8);
    assert_eq!(trace.span_count(), 8, "one span per stage job");
    assert_eq!(trace.total_jobs(), 8);
    let names: Vec<String> = trace.spans().into_iter().map(|s| s.name).collect();
    assert_eq!(names.iter().filter(|n| *n == "extract").count(), 1);
    assert_eq!(names.iter().filter(|n| *n == "simulate").count(), 7);
}
