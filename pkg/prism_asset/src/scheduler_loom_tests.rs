//! Models the production lifecycle transitions, not the task pool, server
//! publication, epoch protocol, or notification queue. Integration regressions
//! exercise those independently. The batch model pins locks in production order.

use super::{FlightLifecycle, FlightPhase};
use loom::sync::atomic::{AtomicUsize, Ordering};
use loom::sync::{Arc, Mutex};
use loom::thread;
use std::task::{Wake, Waker};

type State = Arc<Mutex<FlightLifecycle>>;

loom::thread_local! {
    static HELD: core::cell::Cell<bool> = core::cell::Cell::new(false);
}

fn transition<R>(state: &State, f: impl FnOnce(&mut FlightLifecycle) -> R) -> R {
    let mut guard = state.lock().unwrap();
    HELD.with(|held| assert!(!held.replace(true)));
    let result = f(&mut guard);
    HELD.with(|held| held.set(false));
    drop(guard);
    result
}

fn active() -> State {
    let mut lifecycle = FlightLifecycle::default();
    assert!(lifecycle.attach_consumer());
    Arc::new(Mutex::new(lifecycle))
}

#[derive(Default)]
struct Counts {
    wakes: AtomicUsize,
    drops: AtomicUsize,
}

struct Probe {
    counts: Arc<Counts>,
}

impl Wake for Probe {
    fn wake(self: std::sync::Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &std::sync::Arc<Self>) {
        HELD.with(|held| assert!(!held.get(), "wake under lifecycle lock"));
        self.counts.wakes.fetch_add(1, Ordering::SeqCst);
    }
}

impl Drop for Probe {
    fn drop(&mut self) {
        HELD.with(|held| assert!(!held.get(), "drop under lifecycle lock"));
        self.counts.drops.fetch_add(1, Ordering::SeqCst);
    }
}

fn probe() -> (Waker, Arc<Counts>) {
    let counts = Arc::new(Counts::default());
    let waker = Waker::from(std::sync::Arc::new(Probe {
        counts: Arc::clone(&counts),
    }));
    (waker, counts)
}

fn detach(state: &State) -> bool {
    let (cancelled, wake) = transition(state, FlightLifecycle::detach_consumer);
    if let Some(wake) = wake {
        wake.wake();
    }
    cancelled
}

fn register(state: &State, waker: Waker) -> bool {
    let (cancelled, retired) = transition(state, |state| state.register_worker(waker));
    drop(retired);
    cancelled
}

#[test]
fn joining_and_last_detach_never_resurrect_cancelled_work() {
    loom::model(|| {
        let state = active();
        let joining = Arc::clone(&state);
        let leaving = Arc::clone(&state);
        let join = thread::spawn(move || joining.lock().unwrap().attach_consumer());
        let leave = thread::spawn(move || detach(&leaving));
        let joined = join.join().unwrap();
        let cancelled = leave.join().unwrap();
        assert_ne!(joined, cancelled);
        {
            let state = state.lock().unwrap();
            assert_eq!(state.consumers, usize::from(joined));
            assert!(
                state.phase
                    == if joined {
                        FlightPhase::Active
                    } else {
                        FlightPhase::Cancelling
                    }
            );
        }
        if joined {
            assert!(detach(&state));
        }
    });
}

#[test]
fn last_detach_and_commit_have_exactly_one_winner() {
    loom::model(|| {
        let state = active();
        let committing = Arc::clone(&state);
        let leaving = Arc::clone(&state);
        let commit = thread::spawn(move || committing.lock().unwrap().begin_commit());
        let leave = thread::spawn(move || detach(&leaving));
        let committed = commit.join().unwrap();
        assert_ne!(committed, leave.join().unwrap());
        let state = state.lock().unwrap();
        assert_eq!(state.consumers, 0);
        assert!(
            state.phase
                == if committed {
                    FlightPhase::Committing
                } else {
                    FlightPhase::Cancelling
                }
        );
    });
}

#[test]
fn worker_registration_cannot_lose_last_consumer_cancellation() {
    loom::model(|| {
        let state = active();
        let (waker, counts) = probe();
        let registering = Arc::clone(&state);
        let leaving = Arc::clone(&state);
        let worker = thread::spawn(move || register(&registering, waker));
        let leave = thread::spawn(move || detach(&leaving));
        let observed = worker.join().unwrap();
        assert!(leave.join().unwrap());
        assert_eq!(
            usize::from(observed) + counts.wakes.load(Ordering::SeqCst),
            1
        );
        assert_eq!(counts.drops.load(Ordering::SeqCst), 1);
        assert!(state.lock().unwrap().worker_waker.is_none());
    });
}

#[test]
fn replacing_worker_waker_preserves_one_unlocked_notification() {
    loom::model(|| {
        let state = active();
        let (old, old_counts) = probe();
        let (new, new_counts) = probe();
        assert!(!register(&state, old));
        let registering = Arc::clone(&state);
        let leaving = Arc::clone(&state);
        let worker = thread::spawn(move || register(&registering, new));
        let leave = thread::spawn(move || detach(&leaving));
        worker.join().unwrap();
        assert!(leave.join().unwrap());
        assert_eq!(
            old_counts.wakes.load(Ordering::SeqCst) + new_counts.wakes.load(Ordering::SeqCst),
            1
        );
        assert_eq!(old_counts.drops.load(Ordering::SeqCst), 1);
        assert_eq!(new_counts.drops.load(Ordering::SeqCst), 1);
    });
}

#[test]
fn equivalent_worker_waker_is_retired_outside_lock() {
    loom::model(|| {
        let state = active();
        let (waker, counts) = probe();
        assert!(!register(&state, waker.clone()));
        let registering = Arc::clone(&state);
        let leaving = Arc::clone(&state);
        let worker = thread::spawn(move || register(&registering, waker));
        let leave = thread::spawn(move || detach(&leaving));
        worker.join().unwrap();
        assert!(leave.join().unwrap());
        assert_eq!(counts.wakes.load(Ordering::SeqCst), 1);
        assert_eq!(counts.drops.load(Ordering::SeqCst), 1);
    });
}

#[test]
fn sorted_pinned_batch_is_all_or_nothing_against_final_detaches() {
    loom::model(|| {
        let first = active();
        let second = active();
        let batch_first = Arc::clone(&first);
        let batch_second = Arc::clone(&second);
        let leave_first = Arc::clone(&first);
        let leave_second = Arc::clone(&second);
        let batch = thread::spawn(move || {
            let mut first = batch_first.lock().unwrap();
            let mut second = batch_second.lock().unwrap();
            if !first.accepts_consumer() || !second.accepts_consumer() {
                return false;
            }
            assert!(first.attach_consumer());
            assert!(second.attach_consumer());
            true
        });
        let leave = thread::spawn(move || {
            detach(&leave_first);
            detach(&leave_second);
        });
        let admitted = batch.join().unwrap();
        leave.join().unwrap();
        for state in [&first, &second] {
            {
                let state = state.lock().unwrap();
                assert_eq!(state.consumers, usize::from(admitted));
                assert!(
                    state.phase
                        == if admitted {
                            FlightPhase::Active
                        } else {
                            FlightPhase::Cancelling
                        }
                );
            }
            if admitted {
                assert!(detach(state));
            }
        }
    });
}

#[test]
fn finishing_committed_work_and_final_detach_stay_terminal() {
    loom::model(|| {
        let state = active();
        let (waker, counts) = probe();
        assert!(!register(&state, waker));
        assert!(state.lock().unwrap().begin_commit());
        let finishing = Arc::clone(&state);
        let leaving = Arc::clone(&state);
        let finish = thread::spawn(move || {
            let retired = transition(&finishing, FlightLifecycle::finish);
            drop(retired);
        });
        let leave = thread::spawn(move || detach(&leaving));
        finish.join().unwrap();
        assert!(!leave.join().unwrap());
        let state = state.lock().unwrap();
        assert!(state.phase == FlightPhase::Finished);
        assert_eq!(state.consumers, 0);
        assert!(state.worker_waker.is_none());
        assert_eq!(counts.wakes.load(Ordering::SeqCst), 0);
        assert_eq!(counts.drops.load(Ordering::SeqCst), 1);
    });
}
