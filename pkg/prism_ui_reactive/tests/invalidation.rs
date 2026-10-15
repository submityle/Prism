//! Deferred notifications must not depend on value reads or effect execution.

use std::cell::Cell;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::rc::Rc;

use prism_ui_reactive::{Runtime, SignalChangeQueue};

#[test]
fn initial_notifications_coalesce_and_follow_subscription_order() {
    // No Clone/PartialEq implementation: subscriptions cannot read values.
    struct Opaque;
    let runtime = Runtime::new();
    let signal = runtime.signal(Opaque);
    let queue = SignalChangeQueue::new();
    let first = signal.subscribe_changes(&queue);
    let second = signal.subscribe_changes(&queue);
    for _ in 0..8 {
        signal.set(Opaque);
    }
    let mut batch = queue.take_pending();
    assert_eq!(batch.first(), Some(first.id()));
    batch.acknowledge(first.id());
    assert_eq!(batch.first(), Some(second.id()));
    batch.acknowledge(second.id());
    assert_eq!(batch.first(), None);
    assert_eq!(queue.take_pending().first(), None);
}

#[test]
fn drop_cancels_pending_and_in_flight_work_without_recycling_ids() {
    let runtime = Runtime::new();
    let signal = runtime.signal(0);
    let queue = SignalChangeQueue::new();
    let lease = signal.subscribe_changes(&queue);
    let old = lease.id();
    let batch = queue.take_pending();
    drop(lease);
    drop(batch);
    signal.set(1);
    assert_eq!(queue.take_pending().first(), None);
    let new = signal.subscribe_changes(&queue);
    assert_ne!(new.id(), old);
    assert_eq!(queue.take_pending().first(), Some(new.id()));
    drop(new);
    assert_eq!(queue.take_pending().first(), None);
}

#[test]
fn unfinished_work_retries_but_new_writes_survive_acknowledgement() {
    let runtime = Runtime::new();
    let signal = runtime.signal(0);
    let queue = SignalChangeQueue::new();
    let first = signal.subscribe_changes(&queue);
    let second = signal.subscribe_changes(&queue);
    let mut batch = queue.take_pending();
    signal.set(1);
    batch.acknowledge(first.id());
    // Simulate consumer unwinding with its second entry unfinished.
    drop(batch);
    let mut retry = queue.take_pending();
    assert_eq!(
        retry.first(),
        Some(first.id()),
        "the new write was not acked"
    );
    retry.acknowledge(first.id());
    assert_eq!(retry.first(), Some(second.id()));
    retry.acknowledge(second.id());
    drop(retry);
    assert_eq!(queue.take_pending().first(), None);
}

#[test]
fn cloned_signal_and_independent_queues_have_independent_leases() {
    let runtime = Runtime::new();
    let signal = runtime.signal(0);
    let first = SignalChangeQueue::new();
    let second = SignalChangeQueue::new();
    let a = signal.subscribe_changes(&first);
    let b = signal.clone().subscribe_changes(&second);
    let mut initial = first.take_pending();
    initial.acknowledge(a.id());
    drop(initial);
    let mut initial = second.take_pending();
    initial.acknowledge(b.id());
    drop(initial);
    drop(a);
    signal.clone().set(1);
    assert_eq!(first.take_pending().first(), None);
    assert_eq!(second.take_pending().first(), Some(b.id()));
    drop(first);
    drop(second);
    // A subscription outliving its queue and signal is harmless.
    drop(signal);
    drop(b);
}

#[test]
fn mutation_unwind_notifies_without_running_effects_during_unwind() {
    let runtime = Runtime::new();
    let signal = runtime.signal(0);
    let effects = Rc::new(Cell::new(0));
    let effect = runtime.effect({
        let signal = signal.clone();
        let effects = effects.clone();
        move || {
            signal.get();
            effects.set(effects.get() + 1);
        }
    });
    let queue = SignalChangeQueue::new();
    let lease = signal.subscribe_changes(&queue);
    let mut initial = queue.take_pending();
    initial.acknowledge(lease.id());
    drop(initial);
    assert!(catch_unwind(AssertUnwindSafe(|| {
        signal.update(|value| {
            *value = 7;
            panic!("partially mutated signal");
        });
    }))
    .is_err());
    assert_eq!(signal.get(), 7);
    assert_eq!(effects.get(), 1);
    assert_eq!(queue.take_pending().first(), Some(lease.id()));
    effect.dispose();
}

#[test]
fn effect_panic_cannot_prevent_deferred_delivery() {
    let runtime = Runtime::new();
    let signal = runtime.signal(0);
    let queue = SignalChangeQueue::new();
    let lease = signal.subscribe_changes(&queue);
    let mut initial = queue.take_pending();
    initial.acknowledge(lease.id());
    drop(initial);
    let effect = runtime.effect({
        let signal = signal.clone();
        move || assert_ne!(signal.get(), 1, "effect failure")
    });
    assert!(catch_unwind(AssertUnwindSafe(|| signal.set(1))).is_err());
    assert_eq!(queue.take_pending().first(), Some(lease.id()));
    effect.dispose();
}
