//! Failure recovery must restore tracking context and retain unfinished work,
//! without executing user effects from an unwind guard.

use std::cell::{Cell, RefCell};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::rc::Rc;

use prism_ui_reactive::Runtime;

#[test]
fn effect_panic_restores_tracking_and_retries_only_the_unfinished_tail() {
    let runtime = Runtime::new();
    let source = runtime.signal(0);
    let unrelated = runtime.signal(0);
    let log = Rc::new(RefCell::new(Vec::new()));
    let fail = Rc::new(Cell::new(false));
    let mut effects = Vec::new();
    for index in 0..3 {
        effects.push(runtime.effect({
            let source = source.clone();
            let log = log.clone();
            let fail = fail.clone();
            move || {
                log.borrow_mut().push(index);
                // Fail before reading even one dependency. The last successful
                // dependency set must survive until a successful retry.
                if index == 1 {
                    assert!(!fail.replace(false), "middle effect fails once");
                }
                source.get();
            }
        }));
    }
    let graph = runtime.graph_snapshot();
    log.borrow_mut().clear();
    fail.set(true);
    assert!(catch_unwind(AssertUnwindSafe(|| source.set(1))).is_err());
    assert_eq!(*log.borrow(), [0, 1]);
    unrelated.get(); // must not attach to the failed observer
    assert_eq!(runtime.graph_snapshot(), graph);
    log.borrow_mut().clear();
    runtime.batch(|| ());
    assert_eq!(*log.borrow(), [1, 2]);
    log.borrow_mut().clear();
    source.set(2);
    assert_eq!(*log.borrow(), [0, 1, 2]);
    for effect in effects {
        effect.dispose();
    }
}

#[test]
fn dependency_memo_panic_keeps_its_waiting_effect_queued() {
    let runtime = Runtime::new();
    let source = runtime.signal(1);
    let unrelated = runtime.signal(0);
    let fail = Rc::new(Cell::new(false));
    let memo = runtime.memo({
        let source = source.clone();
        let fail = fail.clone();
        move || {
            assert!(!fail.replace(false), "memo fails before source read");
            source.get() * 2
        }
    });
    let values = Rc::new(RefCell::new(Vec::new()));
    let effect = runtime.effect({
        let memo = memo.clone();
        let values = values.clone();
        move || values.borrow_mut().push(memo.get())
    });
    let graph = runtime.graph_snapshot();
    fail.set(true);
    assert!(catch_unwind(AssertUnwindSafe(|| source.set(2))).is_err());
    unrelated.get();
    assert_eq!(runtime.graph_snapshot(), graph);
    assert_eq!(*values.borrow(), [2]);
    runtime.batch(|| ());
    assert_eq!(*values.borrow(), [2, 4]);
    source.set(3);
    assert_eq!(*values.borrow(), [2, 4, 6]);
    effect.dispose();
    memo.dispose();
}

#[test]
fn caught_nested_memo_panic_restores_the_parent_observer() {
    let runtime = Runtime::new();
    let tail = runtime.signal(0);
    let fail = Rc::new(Cell::new(true));
    let memo = runtime.memo({
        let fail = fail.clone();
        move || {
            assert!(!fail.replace(false), "lazy memo fails once");
            7
        }
    });
    let runs = Rc::new(Cell::new(0));
    let effect = runtime.effect({
        let memo = memo.clone();
        let tail = tail.clone();
        let runs = runs.clone();
        move || {
            let _ = catch_unwind(AssertUnwindSafe(|| memo.get()));
            tail.get();
            runs.set(runs.get() + 1);
        }
    });
    assert_eq!(runs.get(), 1);
    assert!(!runtime.graph_snapshot().has_cycle());
    tail.set(1);
    assert_eq!(
        runs.get(),
        2,
        "tail belongs to the effect, not the failed memo"
    );
    effect.dispose();
    memo.dispose();
}

#[test]
fn untrack_panic_restores_tracking_inside_a_live_effect() {
    let runtime = Runtime::new();
    let hidden = runtime.signal(0);
    let tracked = runtime.signal(0);
    let runs = Rc::new(Cell::new(0));
    let effect = runtime.effect({
        let runtime = runtime.clone();
        let hidden = hidden.clone();
        let tracked = tracked.clone();
        let runs = runs.clone();
        move || {
            assert!(catch_unwind(AssertUnwindSafe(|| runtime.untrack(|| {
                hidden.get();
                panic!("untracked callback");
            })))
            .is_err());
            tracked.get();
            runs.set(runs.get() + 1);
        }
    });
    hidden.set(1);
    assert_eq!(runs.get(), 1);
    tracked.set(1);
    assert_eq!(runs.get(), 2);
    effect.dispose();
}

#[test]
fn nested_batch_panic_restores_depth_without_flushing_during_unwind() {
    let runtime = Runtime::new();
    let source = runtime.signal(0);
    let values = Rc::new(RefCell::new(Vec::new()));
    let effect = runtime.effect({
        let source = source.clone();
        let values = values.clone();
        move || {
            assert!(!std::thread::panicking());
            values.borrow_mut().push(source.get());
        }
    });
    assert!(catch_unwind(AssertUnwindSafe(|| runtime.batch(|| {
        runtime.batch(|| {
            source.set(1);
            panic!("batch callback");
        });
    })))
    .is_err());
    assert_eq!(*values.borrow(), [0]);
    runtime.batch(|| ());
    assert_eq!(*values.borrow(), [0, 1]);
    source.set(2);
    assert_eq!(*values.borrow(), [0, 1, 2]);
    effect.dispose();
}

#[test]
fn failing_initial_effect_releases_its_node_edges_and_captures() {
    let runtime = Runtime::new();
    let source = runtime.signal(0);
    let capture = Rc::new(());
    let weak = Rc::downgrade(&capture);
    let graph = runtime.graph_snapshot();
    assert!(catch_unwind(AssertUnwindSafe(|| {
        runtime.effect({
            let source = source.clone();
            move || {
                let _keep = &capture;
                source.get();
                panic!("initial effect");
            }
        });
    }))
    .is_err());
    assert_eq!(runtime.graph_snapshot(), graph);
    assert!(weak.upgrade().is_none());
    source.set(1);
    runtime.batch(|| ());
    assert_eq!(runtime.graph_snapshot(), graph);
}

#[test]
fn self_disposal_and_slot_reuse_during_panic_do_not_revive_old_edges() {
    let runtime = Runtime::new();
    let source = runtime.signal(0);
    let fail = Rc::new(Cell::new(false));
    let handle = Rc::new(RefCell::new(None::<prism_ui_reactive::Effect>));
    let replacement = Rc::new(RefCell::new(None));
    let effect = runtime.effect({
        let runtime = runtime.clone();
        let source = source.clone();
        let fail = fail.clone();
        let handle = handle.clone();
        let replacement = replacement.clone();
        move || {
            if fail.get() {
                handle.borrow_mut().take().unwrap().dispose();
                *replacement.borrow_mut() = Some(runtime.signal(99));
                panic!("disposed effect fails before reading its old source");
            }
            source.get();
        }
    });
    *handle.borrow_mut() = Some(effect);
    fail.set(true);
    assert!(catch_unwind(AssertUnwindSafe(|| source.set(1))).is_err());
    let graph = runtime.graph_snapshot();
    assert!(graph.effects().is_empty());
    assert_eq!(graph.node_count(), 2);
    assert_eq!(graph.edge_count(), 0);
    assert!(graph.nodes.iter().all(|node| node.observers.is_empty()));
    runtime.batch(|| ());
    source.set(2);
    assert_eq!(runtime.graph_snapshot(), graph);
    assert_eq!(replacement.borrow().as_ref().unwrap().get(), 99);
}

#[test]
fn failed_partial_dependencies_are_retained_until_a_successful_run_prunes_them() {
    let runtime = Runtime::new();
    let branch = runtime.signal(false);
    let old_source = runtime.signal(0);
    let new_source = runtime.signal(0);
    let fail = Rc::new(Cell::new(false));
    let runs = Rc::new(Cell::new(0));
    let effect = runtime.effect({
        let branch = branch.clone();
        let old_source = old_source.clone();
        let new_source = new_source.clone();
        let fail = fail.clone();
        let runs = runs.clone();
        move || {
            runs.set(runs.get() + 1);
            if branch.get() {
                new_source.get();
            } else {
                old_source.get();
            }
            assert!(!fail.replace(false), "partial dependency collection fails");
        }
    });
    fail.set(true);
    assert!(catch_unwind(AssertUnwindSafe(|| branch.set(true))).is_err());
    assert_eq!(runtime.graph_snapshot().effects()[0].sources.len(), 3);
    runtime.batch(|| ());
    assert_eq!(runtime.graph_snapshot().effects()[0].sources.len(), 2);
    let completed = runs.get();
    old_source.set(1);
    assert_eq!(runs.get(), completed);
    new_source.set(1);
    assert_eq!(runs.get(), completed + 1);
    effect.dispose();
}
