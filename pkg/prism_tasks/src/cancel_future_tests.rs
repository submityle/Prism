use super::CancelToken;
use alloc::sync::Arc;
use alloc::task::Wake;
use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
use std::mem::ManuallyDrop;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Barrier;

#[derive(Default)]
struct WakeCount(AtomicUsize);

impl Wake for WakeCount {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

impl WakeCount {
    fn get(&self) -> usize {
        self.0.load(Ordering::SeqCst)
    }
}

fn counter() -> (Arc<WakeCount>, Waker) {
    let count = Arc::new(WakeCount::default());
    let waker = Waker::from(Arc::clone(&count));
    (count, waker)
}

fn poll(future: Pin<&mut impl Future<Output = ()>>, waker: &Waker) -> Poll<()> {
    future.poll(&mut Context::from_waker(waker))
}

fn waiter_count(token: &CancelToken) -> usize {
    token
        .state
        .waiters
        .lock()
        .unwrap()
        .slots
        .iter()
        .filter(|slot| slot.is_some())
        .count()
}

fn assert_unlocked(token: &CancelToken) {
    assert!(token.state.waiters.try_lock().is_ok(), "waiters locked");
    assert!(token.state.children.try_lock().is_ok(), "children locked");
}

struct Hooks {
    clone: Box<dyn Fn() + Send + Sync>,
    wake: Box<dyn Fn() + Send + Sync>,
    drop: Box<dyn Fn() + Send + Sync>,
}

impl Default for Hooks {
    fn default() -> Self {
        Self {
            clone: Box::new(|| {}),
            wake: Box::new(|| {}),
            drop: Box::new(|| {}),
        }
    }
}

// Each raw waker owns exactly one Arc. Borrowed callbacks use ManuallyDrop;
// consuming callbacks recover that reference once, including when a hook panics.
#[expect(
    unsafe_code,
    reason = "test RawWaker callbacks for clone/drop/reentrancy"
)]
fn hooked_waker(hooks: Hooks) -> Waker {
    /// # Safety
    /// `data` must borrow a live `Arc<Hooks>` reference created by `raw`.
    /// That reference must remain owned by the caller throughout this call.
    unsafe fn clone(data: *const ()) -> RawWaker {
        // SAFETY: The vtable receives a live reference from `raw`; ManuallyDrop
        // preserves the caller's reference, including if the clone hook panics.
        let hooks = ManuallyDrop::new(unsafe { Arc::from_raw(data.cast::<Hooks>()) });
        (hooks.clone)();
        raw(Arc::clone(&hooks))
    }

    /// # Safety
    /// `data` must own one live `Arc<Hooks>` reference created by `raw`.
    /// The caller must relinquish that reference to this consuming callback.
    unsafe fn wake(data: *const ()) {
        // SAFETY: This consuming callback takes exactly the reference from
        // `raw`; the local Arc releases it even if the wake hook panics.
        let hooks = unsafe { Arc::from_raw(data.cast::<Hooks>()) };
        (hooks.wake)();
    }

    /// # Safety
    /// `data` must borrow a live `Arc<Hooks>` reference created by `raw`.
    /// That reference must remain owned by the caller throughout this call.
    unsafe fn wake_by_ref(data: *const ()) {
        // SAFETY: The vtable receives a live reference from `raw`; ManuallyDrop
        // preserves the caller's reference, including if the wake hook panics.
        let hooks = ManuallyDrop::new(unsafe { Arc::from_raw(data.cast::<Hooks>()) });
        (hooks.wake)();
    }

    /// # Safety
    /// `data` must own one live `Arc<Hooks>` reference created by `raw`.
    /// The caller must relinquish that reference to this consuming callback.
    unsafe fn drop(data: *const ()) {
        // SAFETY: This consuming callback takes exactly the reference from
        // `raw`; the local Arc releases it even if the drop hook panics.
        let hooks = unsafe { Arc::from_raw(data.cast::<Hooks>()) };
        (hooks.drop)();
    }

    fn raw(hooks: Arc<Hooks>) -> RawWaker {
        const VTABLE: RawWakerVTable = RawWakerVTable::new(clone, wake, wake_by_ref, drop);
        RawWaker::new(Arc::into_raw(hooks).cast(), &VTABLE)
    }

    // SAFETY: `raw` transfers one Arc reference into the waker. The vtable
    // preserves borrowed references and releases consumed ones exactly once;
    // Hooks is Send + Sync, so its callbacks may run on any thread.
    unsafe { Waker::from_raw(raw(Arc::new(hooks))) }
}

#[test]
fn cancelled_future_is_send_and_token_can_move_into_an_async_task() {
    fn send_static(_: impl Future<Output = ()> + Send + 'static) {}
    fn send(_: &(impl Future<Output = ()> + Send)) {}

    let token = CancelToken::new();
    send(&token.cancelled());
    let task_token = token.clone();
    send_static(async move { task_token.cancelled().await });

    let pool = crate::TaskPool::with_threads(1);
    let task_token = token.clone();
    let task = pool.spawn_async(async move { task_token.cancelled().await });
    token.cancel();
    pool.block_on(task);
}

#[test]
fn registration_is_lazy_and_already_cancelled_never_clones_the_waker() {
    let token = CancelToken::new();
    let future = token.cancelled();
    assert_eq!(waiter_count(&token), 0);
    drop(future);
    token.cancel();
    let waker = hooked_waker(Hooks {
        clone: Box::new(|| panic!("already cancelled must not clone")),
        ..Hooks::default()
    });
    let mut future = Box::pin(token.cancelled());
    assert_eq!(poll(future.as_mut(), &waker), Poll::Ready(()));
    assert_eq!(poll(future.as_mut(), &waker), Poll::Ready(()));
    assert_eq!(waiter_count(&token), 0);
}

#[test]
fn root_and_subtree_cancel_wake_every_independent_waiter() {
    let root = CancelToken::new();
    let child = root.child();
    let grandchild = child.child();
    let sibling = root.child();
    let tokens = [&root, &child, &grandchild, &sibling];
    let counters: Vec<_> = tokens.iter().map(|_| counter()).collect();
    let mut futures: Vec<_> = tokens
        .iter()
        .flat_map(|token| (0..3).map(move |_| Box::pin(token.cancelled())))
        .collect();
    for (i, future) in futures.iter_mut().enumerate() {
        assert_eq!(poll(future.as_mut(), &counters[i / 3].1), Poll::Pending);
    }
    child.cancel();
    assert_eq!(counters[0].0.get(), 0);
    assert_eq!(counters[1].0.get(), 3);
    assert_eq!(counters[2].0.get(), 3);
    assert_eq!(counters[3].0.get(), 0);
    root.cancel();
    root.cancel();
    for (i, future) in futures.iter_mut().enumerate() {
        assert_eq!(poll(future.as_mut(), &counters[i / 3].1), Poll::Ready(()));
    }
    for (count, _) in counters {
        assert_eq!(count.get(), 3);
    }
    assert!(tokens.iter().all(|token| waiter_count(token) == 0));
}

#[test]
fn repoll_replaces_the_waker_without_adding_registrations() {
    let token = CancelToken::new();
    let (old_count, old) = counter();
    let (latest_count, latest) = counter();
    let mut future = Box::pin(token.cancelled());
    for _ in 0..16 {
        assert_eq!(poll(future.as_mut(), &old), Poll::Pending);
        assert_eq!(poll(future.as_mut(), &latest), Poll::Pending);
    }
    assert_eq!(waiter_count(&token), 1);
    assert_eq!(token.state.waiters.lock().unwrap().slots.len(), 1);
    token.cancel();
    assert_eq!(old_count.get(), 0);
    assert_eq!(latest_count.get(), 1);
}

#[test]
fn drop_unregisters_and_slots_are_reused() {
    let token = CancelToken::new();
    let (count, waker) = counter();
    for _ in 0..512 {
        let mut future = Box::pin(token.cancelled());
        assert_eq!(poll(future.as_mut(), &waker), Poll::Pending);
        assert_eq!(waiter_count(&token), 1);
        drop(future);
        assert_eq!(waiter_count(&token), 0);
    }
    assert_eq!(token.state.waiters.lock().unwrap().slots.len(), 1);
    token.cancel();
    assert_eq!(count.get(), 0);
}

#[test]
fn dropping_one_waiter_keeps_another_with_the_same_waker() {
    let token = CancelToken::new();
    let (count, waker) = counter();
    let mut first = Box::pin(token.cancelled());
    let mut second = Box::pin(token.cancelled());
    assert_eq!(poll(first.as_mut(), &waker), Poll::Pending);
    assert_eq!(poll(second.as_mut(), &waker), Poll::Pending);
    drop(first);
    assert_eq!(waiter_count(&token), 1);
    token.cancel();
    assert_eq!(count.get(), 1);
    assert_eq!(poll(second.as_mut(), &waker), Poll::Ready(()));
}

#[test]
fn dropped_intermediate_tokens_still_propagate_to_live_descendants() {
    let root = CancelToken::new();
    let child = root.child();
    let child_state = Arc::downgrade(&child.state);
    let grandchild = child.child();
    drop(child);
    assert!(child_state.upgrade().is_some());
    let (count, waker) = counter();
    let mut future = Box::pin(grandchild.cancelled());
    assert_eq!(poll(future.as_mut(), &waker), Poll::Pending);
    root.cancel();
    assert_eq!(count.get(), 1);
    assert_eq!(poll(future.as_mut(), &waker), Poll::Ready(()));
    drop(future);
    drop(grandchild);
    assert!(child_state.upgrade().is_none());
}

#[test]
fn long_lived_parent_releases_abandoned_subtrees_and_prunes_weak_links() {
    let root = CancelToken::new();
    let child = root.child();
    let capacity = root.state.children.lock().unwrap().capacity();
    drop(child);
    for _ in 0..1024 {
        let child = root.child();
        let leaf = child.child();
        let child_state = Arc::downgrade(&child.state);
        let leaf_state = Arc::downgrade(&leaf.state);
        drop(child);
        drop(leaf);
        assert!(child_state.upgrade().is_none());
        assert!(leaf_state.upgrade().is_none());
    }
    assert_eq!(root.state.children.lock().unwrap().capacity(), capacity);
    root.cancel();
    for _ in 0..16 {
        assert!(root.child().is_cancelled());
    }
    assert!(root.state.children.lock().unwrap().is_empty());
}

#[test]
fn deep_tree_cancellation_and_ancestry_drop_are_iterative() {
    for cancel in [false, true] {
        let root = CancelToken::new();
        let root_state = Arc::downgrade(&root.state);
        let mut leaf = root.clone();
        for _ in 0..16_384 {
            leaf = leaf.child();
        }
        if cancel {
            root.cancel();
            assert!(leaf.is_cancelled());
        }
        drop(root);
        drop(leaf);
        assert!(root_state.upgrade().is_none());
    }
}

#[test]
fn cancellation_during_waker_clone_cannot_lose_a_wake() {
    let token = CancelToken::new();
    let cloning = Arc::new(Barrier::new(2));
    let release = Arc::new(Barrier::new(2));
    let waker = hooked_waker(Hooks {
        clone: {
            let cloning = Arc::clone(&cloning);
            let release = Arc::clone(&release);
            Box::new(move || {
                cloning.wait();
                release.wait();
            })
        },
        ..Hooks::default()
    });
    std::thread::scope(|scope| {
        let waiter = scope.spawn(|| {
            let mut future = Box::pin(token.cancelled());
            poll(future.as_mut(), &waker)
        });
        cloning.wait();
        token.cancel();
        release.wait();
        assert_eq!(waiter.join().unwrap(), Poll::Ready(()));
    });
    assert_eq!(waiter_count(&token), 0);
}

#[test]
fn concurrent_registration_and_cancel_either_complete_or_notify() {
    for _ in 0..64 {
        let token = CancelToken::new();
        let start = Barrier::new(2);
        let finish = Barrier::new(2);
        let (count, waker) = counter();
        std::thread::scope(|scope| {
            let waiter = scope.spawn(|| {
                let mut future = Box::pin(token.cancelled());
                start.wait();
                let first_poll = poll(future.as_mut(), &waker);
                // Keep a Pending registration alive until cancellation ends.
                finish.wait();
                assert_eq!(poll(future.as_mut(), &waker), Poll::Ready(()));
                first_poll
            });
            start.wait();
            token.cancel();
            finish.wait();
            if waiter.join().unwrap().is_pending() {
                assert_eq!(count.get(), 1);
            }
        });
        assert_eq!(waiter_count(&token), 0);
    }
}

#[test]
fn concurrent_child_registration_cannot_escape_root_cancellation() {
    for _ in 0..64 {
        let root = CancelToken::new();
        let start = Barrier::new(2);
        std::thread::scope(|scope| {
            let children = scope.spawn(|| {
                start.wait();
                let child = root.child();
                let leaf = child.child();
                (child, leaf)
            });
            start.wait();
            root.cancel();
            let (child, leaf) = children.join().unwrap();
            assert!(child.is_cancelled());
            assert!(leaf.is_cancelled());
        });
    }
}

#[test]
fn all_raw_waker_callbacks_run_outside_tree_and_waiter_mutexes() {
    let token = CancelToken::new();
    let calls = Arc::new(AtomicUsize::new(0));
    let hook = || {
        let token = token.clone();
        let calls = Arc::clone(&calls);
        Box::new(move || {
            assert_unlocked(&token);
            calls.fetch_add(1, Ordering::SeqCst);
        }) as Box<dyn Fn() + Send + Sync>
    };
    let waker = hooked_waker(Hooks {
        clone: hook(),
        wake: hook(),
        drop: hook(),
    });
    let mut first = Box::pin(token.cancelled());
    let mut second = Box::pin(token.cancelled());
    assert_eq!(poll(first.as_mut(), &waker), Poll::Pending);
    assert_eq!(poll(first.as_mut(), &waker), Poll::Pending);
    assert_eq!(poll(second.as_mut(), &waker), Poll::Pending);
    drop(first);
    token.cancel();
    drop(second);
    drop(waker);
    assert_eq!(calls.load(Ordering::SeqCst), 8);
}

#[test]
fn clone_callback_can_reenter_cancellation() {
    let token = CancelToken::new();
    let waker = hooked_waker(Hooks {
        clone: {
            let token = token.clone();
            Box::new(move || {
                assert_unlocked(&token);
                token.cancel();
            })
        },
        ..Hooks::default()
    });
    let mut future = Box::pin(token.cancelled());
    assert_eq!(poll(future.as_mut(), &waker), Poll::Ready(()));
    assert_eq!(waiter_count(&token), 0);
}

#[test]
fn replacing_waker_drop_can_cancel_and_wake_the_replacement() {
    let token = CancelToken::new();
    let armed = Arc::new(AtomicBool::new(false));
    let old = hooked_waker(Hooks {
        drop: {
            let token = token.clone();
            let armed = Arc::clone(&armed);
            Box::new(move || {
                assert_unlocked(&token);
                if armed.swap(false, Ordering::SeqCst) {
                    token.cancel();
                }
            })
        },
        ..Hooks::default()
    });
    let mut future = Box::pin(token.cancelled());
    assert_eq!(poll(future.as_mut(), &old), Poll::Pending);
    drop(old);
    armed.store(true, Ordering::SeqCst);
    let (count, latest) = counter();
    assert_eq!(poll(future.as_mut(), &latest), Poll::Pending);
    assert_eq!(count.get(), 1);
    assert_eq!(poll(future.as_mut(), &latest), Poll::Ready(()));
}

#[test]
fn unregistering_waker_drop_can_reenter_and_notify_other_waiters() {
    let token = CancelToken::new();
    let armed = Arc::new(AtomicBool::new(false));
    let waker = hooked_waker(Hooks {
        drop: {
            let token = token.clone();
            let armed = Arc::clone(&armed);
            Box::new(move || {
                assert_unlocked(&token);
                if armed.swap(false, Ordering::SeqCst) {
                    token.cancel();
                }
            })
        },
        ..Hooks::default()
    });
    let mut first = Box::pin(token.cancelled());
    assert_eq!(poll(first.as_mut(), &waker), Poll::Pending);
    drop(waker);
    let (count, other) = counter();
    let mut second = Box::pin(token.cancelled());
    assert_eq!(poll(second.as_mut(), &other), Poll::Pending);
    armed.store(true, Ordering::SeqCst);
    drop(first);
    assert_eq!(count.get(), 1);
    assert_eq!(waiter_count(&token), 0);
}

#[test]
fn wake_can_reenter_after_all_descendants_are_cancelled() {
    let root = CancelToken::new();
    let leaf = root.child().child();
    let waker = hooked_waker(Hooks {
        wake: {
            let root = root.clone();
            let leaf = leaf.clone();
            Box::new(move || {
                assert_unlocked(&root);
                assert_unlocked(&leaf);
                assert!(leaf.is_cancelled());
                root.cancel();
                leaf.cancel();
                let child = root.child();
                let mut future = Box::pin(child.cancelled());
                assert_eq!(poll(future.as_mut(), Waker::noop()), Poll::Ready(()));
            })
        },
        ..Hooks::default()
    });
    let mut future = Box::pin(root.cancelled());
    assert_eq!(poll(future.as_mut(), &waker), Poll::Pending);
    root.cancel();
}

#[test]
fn panicking_wakes_do_not_skip_other_waiters_or_descendants() {
    let root = CancelToken::new();
    let leaf = root.child().child();
    let failures = Arc::new(AtomicUsize::new(0));
    let bad = hooked_waker(Hooks {
        wake: {
            let root = root.clone();
            let leaf = leaf.clone();
            let failures = Arc::clone(&failures);
            Box::new(move || {
                assert_unlocked(&root);
                assert_unlocked(&leaf);
                assert!(leaf.is_cancelled());
                failures.fetch_add(1, Ordering::SeqCst);
                panic!("wake failure");
            })
        },
        ..Hooks::default()
    });
    let (count, good) = counter();
    let mut first = Box::pin(root.cancelled());
    let mut second = Box::pin(root.cancelled());
    let mut third = Box::pin(root.cancelled());
    let mut descendant = Box::pin(leaf.cancelled());
    assert_eq!(poll(first.as_mut(), &bad), Poll::Pending);
    assert_eq!(poll(second.as_mut(), &bad), Poll::Pending);
    assert_eq!(poll(third.as_mut(), &good), Poll::Pending);
    assert_eq!(poll(descendant.as_mut(), &good), Poll::Pending);
    assert!(catch_unwind(AssertUnwindSafe(|| root.cancel())).is_err());
    assert_eq!(failures.load(Ordering::SeqCst), 2);
    assert_eq!(count.get(), 2);
    assert_eq!(waiter_count(&root), 0);
    assert_eq!(waiter_count(&leaf), 0);
    root.cancel();
    assert_eq!(poll(first.as_mut(), &good), Poll::Ready(()));
    assert_eq!(poll(descendant.as_mut(), &good), Poll::Ready(()));
    assert_eq!(count.get(), 2);
}

#[test]
fn panicking_waker_drop_does_not_skip_other_notifications() {
    let root = CancelToken::new();
    let child = root.child();
    let armed = Arc::new(AtomicBool::new(false));
    let bad = hooked_waker(Hooks {
        drop: {
            let root = root.clone();
            let armed = Arc::clone(&armed);
            Box::new(move || {
                assert_unlocked(&root);
                assert!(!armed.swap(false, Ordering::SeqCst), "drop failure");
            })
        },
        ..Hooks::default()
    });
    let (count, good) = counter();
    let mut first = Box::pin(root.cancelled());
    let mut second = Box::pin(root.cancelled());
    let mut descendant = Box::pin(child.cancelled());
    assert_eq!(poll(first.as_mut(), &bad), Poll::Pending);
    assert_eq!(poll(second.as_mut(), &good), Poll::Pending);
    assert_eq!(poll(descendant.as_mut(), &good), Poll::Pending);
    drop(bad);
    armed.store(true, Ordering::SeqCst);
    assert!(catch_unwind(AssertUnwindSafe(|| root.cancel())).is_err());
    assert!(child.is_cancelled());
    assert_eq!(count.get(), 2);
    root.cancel();
    assert_eq!(waiter_count(&root), 0);
}

#[test]
fn clone_panic_preserves_the_previous_registration_and_leaves_mutexes_usable() {
    let token = CancelToken::new();
    let (count, good) = counter();
    let mut future = Box::pin(token.cancelled());
    assert_eq!(poll(future.as_mut(), &good), Poll::Pending);
    let bad = hooked_waker(Hooks {
        clone: {
            let token = token.clone();
            Box::new(move || {
                assert_unlocked(&token);
                panic!("clone failure");
            })
        },
        ..Hooks::default()
    });
    assert!(catch_unwind(AssertUnwindSafe(|| poll(future.as_mut(), &bad))).is_err());
    assert_eq!(waiter_count(&token), 1);
    token.cancel();
    assert_eq!(count.get(), 1);
    assert_eq!(poll(future.as_mut(), &good), Poll::Ready(()));
}

// Run abort-sensitive regressions in their own process so a future double
// panic produces a failed test instead of aborting the entire test harness.
fn isolated_panic_regression(name: &str, body: impl FnOnce()) {
    const CASE_ENV: &str = "PRISM_CANCEL_PANIC_REGRESSION";
    let exact = format!("cancel::cancel_future_tests::{name}");
    if std::env::var(CASE_ENV).as_deref() == Ok(exact.as_str()) {
        body();
        return;
    }
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .arg("--exact")
        .arg(&exact)
        .arg("--nocapture")
        .arg("--test-threads=1")
        .env(CASE_ENV, exact)
        .env("RUST_BACKTRACE", "0")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "panic regression subprocess failed: {}\n{}\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[derive(Clone)]
struct PayloadProbe {
    token: CancelToken,
    notifications: Arc<WakeCount>,
    expected_notifications: usize,
    payload_drops: Arc<AtomicUsize>,
    cleanup_payload_drops: Arc<AtomicUsize>,
}

impl PayloadProbe {
    fn panic(&self) -> ! {
        assert_unlocked(&self.token);
        std::panic::panic_any(PanickingPayload(self.clone()));
    }

    fn waker(&self, drop_armed: &Arc<AtomicBool>) -> Waker {
        hooked_waker(Hooks {
            wake: {
                let probe = self.clone();
                Box::new(move || probe.panic())
            },
            drop: {
                let probe = self.clone();
                let armed = Arc::clone(drop_armed);
                Box::new(move || {
                    if armed.load(Ordering::SeqCst) {
                        probe.panic();
                    }
                })
            },
            ..Hooks::default()
        })
    }
}

struct PanickingPayload(PayloadProbe);

impl Drop for PanickingPayload {
    fn drop(&mut self) {
        assert_unlocked(&self.0.token);
        assert!(self.0.token.is_cancelled());
        // No payload cleanup is allowed to run before all notifications.
        assert_eq!(self.0.notifications.get(), self.0.expected_notifications);
        self.0.payload_drops.fetch_add(1, Ordering::SeqCst);
        self.0.token.cancel();
        std::panic::panic_any(CleanupPayload(Arc::clone(&self.0.cleanup_payload_drops)));
    }
}

struct CleanupPayload(Arc<AtomicUsize>);

impl Drop for CleanupPayload {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
        panic!("a panic from payload cleanup must not be dropped recursively");
    }
}

#[test]
fn multiple_payload_destructor_panics_preserve_first_panic_and_all_notifications() {
    isolated_panic_regression(
        "multiple_payload_destructor_panics_preserve_first_panic_and_all_notifications",
        || {
            let root = CancelToken::new();
            let child = root.child();
            let leaf = child.child();
            let (count, good) = counter();
            let probe = PayloadProbe {
                token: root.clone(),
                notifications: Arc::clone(&count),
                expected_notifications: 3,
                payload_drops: Arc::new(AtomicUsize::new(0)),
                cleanup_payload_drops: Arc::new(AtomicUsize::new(0)),
            };
            let first = hooked_waker(Hooks {
                wake: Box::new(|| panic!("first waiter panic")),
                ..Hooks::default()
            });
            let armed = Arc::new(AtomicBool::new(false));
            let bad = probe.waker(&armed);
            let mut first_future = Box::pin(root.cancelled());
            let mut bad_futures = [Box::pin(root.cancelled()), Box::pin(root.cancelled())];
            let mut good_futures = [
                Box::pin(root.cancelled()),
                Box::pin(child.cancelled()),
                Box::pin(leaf.cancelled()),
            ];
            assert_eq!(poll(first_future.as_mut(), &first), Poll::Pending);
            for future in &mut bad_futures {
                assert_eq!(poll(future.as_mut(), &bad), Poll::Pending);
            }
            for future in &mut good_futures {
                assert_eq!(poll(future.as_mut(), &good), Poll::Pending);
            }
            drop(bad);
            armed.store(true, Ordering::SeqCst);

            let payload = catch_unwind(AssertUnwindSafe(|| root.cancel())).unwrap_err();
            assert_eq!(payload.downcast_ref::<&str>(), Some(&"first waiter panic"));
            assert_eq!(count.get(), 3);
            // Both wake and drop panic for each of the two bad registrations.
            assert_eq!(probe.payload_drops.load(Ordering::SeqCst), 4);
            assert_eq!(probe.cleanup_payload_drops.load(Ordering::SeqCst), 0);
            assert_eq!(waiter_count(&root), 0);
            assert_eq!(waiter_count(&child), 0);
            assert_eq!(waiter_count(&leaf), 0);
            for future in &mut good_futures {
                assert_eq!(poll(future.as_mut(), &good), Poll::Ready(()));
            }
            root.cancel();
            assert_eq!(count.get(), 3);
        },
    );
}

#[test]
fn cancel_during_unwind_suppresses_waiter_and_payload_panics() {
    isolated_panic_regression(
        "cancel_during_unwind_suppresses_waiter_and_payload_panics",
        || {
            struct CancelOnDrop {
                token: CancelToken,
                returned: Arc<AtomicBool>,
            }

            impl Drop for CancelOnDrop {
                fn drop(&mut self) {
                    assert!(std::thread::panicking());
                    self.token.cancel();
                    self.returned.store(true, Ordering::SeqCst);
                }
            }

            let root = CancelToken::new();
            let child = root.child();
            let leaf = child.child();
            let (count, good) = counter();
            let probe = PayloadProbe {
                token: root.clone(),
                notifications: Arc::clone(&count),
                expected_notifications: 3,
                payload_drops: Arc::new(AtomicUsize::new(0)),
                cleanup_payload_drops: Arc::new(AtomicUsize::new(0)),
            };
            let armed = Arc::new(AtomicBool::new(false));
            let bad = probe.waker(&armed);
            let mut bad_futures = [Box::pin(root.cancelled()), Box::pin(root.cancelled())];
            let mut good_futures = [
                Box::pin(root.cancelled()),
                Box::pin(child.cancelled()),
                Box::pin(leaf.cancelled()),
            ];
            for future in &mut bad_futures {
                assert_eq!(poll(future.as_mut(), &bad), Poll::Pending);
            }
            for future in &mut good_futures {
                assert_eq!(poll(future.as_mut(), &good), Poll::Pending);
            }
            drop(bad);
            armed.store(true, Ordering::SeqCst);
            let returned = Arc::new(AtomicBool::new(false));
            let original = catch_unwind(AssertUnwindSafe(|| {
                let _guard = CancelOnDrop {
                    token: root.clone(),
                    returned: Arc::clone(&returned),
                };
                panic!("original operation panic");
            }))
            .unwrap_err();

            assert_eq!(
                original.downcast_ref::<&str>(),
                Some(&"original operation panic")
            );
            assert!(returned.load(Ordering::SeqCst));
            assert!(!std::thread::panicking());
            assert!(child.is_cancelled());
            assert!(leaf.is_cancelled());
            assert_eq!(count.get(), 3);
            assert_eq!(probe.payload_drops.load(Ordering::SeqCst), 4);
            assert_eq!(probe.cleanup_payload_drops.load(Ordering::SeqCst), 0);
            for token in [&root, &child, &leaf] {
                assert_eq!(waiter_count(token), 0);
            }
            for future in &mut good_futures {
                assert_eq!(poll(future.as_mut(), &good), Poll::Ready(()));
            }
            root.cancel();
        },
    );
}
