//! Subscription admission, detachment, and cancellation regressions.

use prism_asset_build_core::bake::BakeSubscriptionLimits;
use prism_asset_build_core::{
    BakeError, BakeKernelRegistry, BakeRequest, BakeService, BakeServiceConfigError,
    CancellationToken, KernelError, RuntimeCacheConfig, StableInputKey,
};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Barrier, Condvar, Mutex};
use std::thread;
use std::time::Duration;

const TIMEOUT: Duration = Duration::from_secs(10);

fn request(input: &[u8]) -> BakeRequest {
    BakeRequest::new(
        StableInputKey::from_bytes(input),
        input.to_vec(),
        1,
        2,
        1,
        Vec::new(),
        3,
        1,
        16,
    )
}

#[derive(Default)]
struct Gate {
    open: Mutex<bool>,
    changed: Condvar,
}

impl Gate {
    fn wait(&self) -> Result<(), KernelError> {
        let (open, _) = self
            .changed
            .wait_timeout_while(self.open.lock().unwrap(), TIMEOUT, |open| !*open)
            .unwrap();
        if *open {
            Ok(())
        } else {
            Err(KernelError::new("test kernel release timed out"))
        }
    }

    fn release(&self) {
        *self.open.lock().unwrap() = true;
        self.changed.notify_all();
    }
}

struct BlockedService {
    service: BakeService,
    entered: mpsc::Receiver<CancellationToken>,
    gate: Arc<Gate>,
    calls: Arc<AtomicUsize>,
}

impl BlockedService {
    fn new(max_in_flight: usize, limits: BakeSubscriptionLimits) -> Self {
        let gate = Arc::new(Gate::default());
        let worker_gate = Arc::clone(&gate);
        let calls = Arc::new(AtomicUsize::new(0));
        let worker_calls = Arc::clone(&calls);
        let (entered, receiver) = mpsc::channel();
        let service = BakeService::new_with_subscription_limits(
            max_in_flight,
            limits,
            Arc::new(move |request: &BakeRequest, token: &CancellationToken| {
                worker_calls.fetch_add(1, Ordering::SeqCst);
                entered
                    .send(token.clone())
                    .map_err(|_| KernelError::new("test observer disconnected"))?;
                worker_gate.wait()?;
                Ok(request.input_bytes.clone())
            }),
        )
        .unwrap();
        Self {
            service,
            entered: receiver,
            gate,
            calls,
        }
    }

    fn entered(&self) -> CancellationToken {
        self.entered
            .recv_timeout(TIMEOUT)
            .expect("kernel did not start")
    }
}

impl Drop for BlockedService {
    fn drop(&mut self) {
        // Release workers before service teardown, also when an assertion fails.
        self.gate.release();
    }
}

#[test]
fn subscription_limits_reject_zero_and_existing_constructor_has_finite_defaults() {
    for limits in [
        BakeSubscriptionLimits {
            max_total: 0,
            max_per_job: 1,
        },
        BakeSubscriptionLimits {
            max_total: 1,
            max_per_job: 0,
        },
    ] {
        assert!(matches!(
            BakeService::new_with_subscription_limits(
                1,
                limits,
                Arc::new(|_: &BakeRequest, _: &CancellationToken| Ok(Vec::new())),
            ),
            Err(BakeServiceConfigError::ZeroSubscriptionLimit)
        ));
    }
    let service = BakeService::new(
        1,
        Arc::new(|_: &BakeRequest, _: &CancellationToken| Ok(Vec::new())),
    )
    .unwrap();
    assert_eq!(
        service.subscription_limits(),
        BakeSubscriptionLimits::default()
    );
    assert_eq!(service.active_subscriptions(), 0);
}

#[test]
fn duplicate_limit_and_submit_drop_churn_release_slots_without_cancelling_kernel() {
    let blocked = BlockedService::new(
        1,
        BakeSubscriptionLimits {
            max_total: 4,
            max_per_job: 2,
        },
    );
    let input = request(b"one-shared-execution");
    let first = blocked.service.submit(input.clone()).unwrap();
    let token = blocked.entered();
    let second = blocked.service.submit(input.clone()).unwrap();
    assert_eq!(blocked.service.active_subscriptions(), 2);
    assert!(matches!(
        blocked.service.submit(input.clone()),
        Err(BakeError::JobSubscriptionLimit { max_per_job: 2 })
    ));
    assert_eq!(blocked.service.active_subscriptions(), 2);
    drop(second);
    for _ in 0..128 {
        let duplicate = blocked.service.submit(input.clone()).unwrap();
        assert_eq!(blocked.service.active_subscriptions(), 2);
        drop(duplicate);
        assert_eq!(blocked.service.active_subscriptions(), 1);
    }
    drop(first);
    assert_eq!(blocked.service.active_subscriptions(), 0);
    assert!(!token.is_cancelled());
    let reattached = blocked.service.submit(input).unwrap();
    assert_eq!(blocked.calls.load(Ordering::SeqCst), 1);
    blocked.gate.release();
    assert_eq!(
        reattached.wait().unwrap().output_bytes,
        b"one-shared-execution"
    );
    assert_eq!(blocked.service.active_subscriptions(), 0);
}

#[test]
fn global_limit_is_shared_across_jobs_and_service_clones_and_failures_refund_slots() {
    let blocked = BlockedService::new(
        2,
        BakeSubscriptionLimits {
            max_total: 2,
            max_per_job: 4,
        },
    );
    let clone = blocked.service.clone();
    let first = blocked.service.submit(request(b"first")).unwrap();
    let first_token = blocked.entered();
    let second = clone.submit(request(b"second")).unwrap();
    blocked.entered();
    assert!(matches!(
        clone.submit(request(b"first")),
        Err(BakeError::SubscriptionLimit { max_total: 2 })
    ));
    assert_eq!(clone.active_subscriptions(), 2);
    drop(first);
    assert!(!first_token.is_cancelled());
    assert!(matches!(
        clone.submit(request(b"third")),
        Err(BakeError::QueueFull { max_in_flight: 2 })
    ));
    assert_eq!(clone.active_subscriptions(), 1);
    let reattached = clone.submit(request(b"first")).unwrap();
    assert_eq!(blocked.service.active_subscriptions(), 2);
    blocked.gate.release();
    assert_eq!(second.wait().unwrap().output_bytes, b"second");
    assert_eq!(reattached.wait().unwrap().output_bytes, b"first");
    assert_eq!(clone.active_subscriptions(), 0);
}

#[test]
fn concurrent_duplicate_admission_obeys_both_subscription_budgets() {
    for limits in [
        BakeSubscriptionLimits {
            max_total: 3,
            max_per_job: 16,
        },
        BakeSubscriptionLimits {
            max_total: 16,
            max_per_job: 3,
        },
    ] {
        let blocked = BlockedService::new(1, limits);
        let input = request(b"concurrent-budget");
        let first = blocked.service.submit(input.clone()).unwrap();
        let token = blocked.entered();
        let start = Arc::new(Barrier::new(9));
        let submitters: Vec<_> = (0..8)
            .map(|_| {
                let service = blocked.service.clone();
                let input = input.clone();
                let start = Arc::clone(&start);
                thread::spawn(move || {
                    start.wait();
                    service.submit(input)
                })
            })
            .collect();
        start.wait();
        let mut admitted = Vec::new();
        let mut rejected = 0;
        for submitter in submitters {
            match submitter.join().unwrap() {
                Ok(handle) => admitted.push(handle),
                Err(BakeError::SubscriptionLimit { max_total: 3 }) if limits.max_total == 3 => {
                    rejected += 1;
                }
                Err(BakeError::JobSubscriptionLimit { max_per_job: 3 })
                    if limits.max_per_job == 3 =>
                {
                    rejected += 1;
                }
                Err(error) => panic!("unexpected admission error: {error}"),
            }
        }
        assert_eq!(admitted.len(), 2);
        assert_eq!(rejected, 6);
        assert_eq!(blocked.service.active_subscriptions(), 3);
        drop(admitted);
        assert_eq!(blocked.service.active_subscriptions(), 1);
        assert!(!token.is_cancelled());
        blocked.gate.release();
        first.wait().unwrap();
        assert_eq!(blocked.service.active_subscriptions(), 0);
    }
}

#[test]
fn cancelled_undrained_handles_keep_their_budget_until_dropped() {
    let blocked = BlockedService::new(
        2,
        BakeSubscriptionLimits {
            max_total: 2,
            max_per_job: 2,
        },
    );
    let input = request(b"cancelled-budget");
    let first = blocked.service.submit(input.clone()).unwrap();
    let old_token = blocked.entered();
    let second = blocked.service.submit(input.clone()).unwrap();
    assert!(first.cancel());
    assert!(second.cancel());
    assert!(old_token.is_cancelled());
    assert_eq!(blocked.service.active_subscriptions(), 2);
    assert!(matches!(
        blocked.service.submit(input.clone()),
        Err(BakeError::SubscriptionLimit { max_total: 2 })
    ));
    drop(first);
    let fresh = blocked.service.submit(input).unwrap();
    let fresh_token = blocked.entered();
    assert!(!fresh_token.is_cancelled());
    blocked.gate.release();
    assert!(matches!(second.wait(), Err(BakeError::Cancelled)));
    assert_eq!(fresh.wait().unwrap().output_bytes, b"cancelled-budget");
    assert_eq!(blocked.service.active_subscriptions(), 0);
}

#[test]
fn undrained_cache_hits_consume_global_budget_until_handle_drop() {
    let registry = Arc::new(BakeKernelRegistry::new());
    registry
        .register(
            1,
            Arc::new(|request: &BakeRequest, _: &CancellationToken| {
                Ok(request.input_bytes.clone())
            }),
        )
        .unwrap();
    let service = BakeService::new_with_dispatch_and_subscription_limits(
        1,
        RuntimeCacheConfig::new(2, 1024),
        BakeSubscriptionLimits {
            max_total: 2,
            max_per_job: 1,
        },
        registry,
    )
    .unwrap();
    let input = request(b"cached");
    service.submit(input.clone()).unwrap().wait().unwrap();
    let first = service.submit(input.clone()).unwrap();
    let second = service.submit(input.clone()).unwrap();
    assert_eq!(service.in_flight(), 0);
    assert_eq!(service.active_subscriptions(), 2);
    assert!(matches!(
        service.submit(input.clone()),
        Err(BakeError::SubscriptionLimit { max_total: 2 })
    ));
    assert!(!first.cancel());
    drop(first);
    let replacement = service.submit(input).unwrap();
    assert_eq!(second.wait().unwrap().output_bytes, b"cached");
    assert_eq!(replacement.wait().unwrap().output_bytes, b"cached");
    assert_eq!(service.active_subscriptions(), 0);
}

#[test]
fn cancelling_one_subscriber_preserves_an_already_joined_subscriber() {
    let blocked = BlockedService::new(
        2,
        BakeSubscriptionLimits {
            max_total: 4,
            max_per_job: 2,
        },
    );
    let input = request(b"join-before-cancel");
    let first = blocked.service.submit(input.clone()).unwrap();
    let token = blocked.entered();
    let clone = blocked.service.clone();
    let second = thread::spawn(move || clone.submit(input).unwrap())
        .join()
        .unwrap();
    assert!(first.cancel());
    assert!(!first.cancel());
    assert!(!token.is_cancelled());
    assert_eq!(blocked.calls.load(Ordering::SeqCst), 1);
    blocked.gate.release();
    assert!(matches!(first.wait(), Err(BakeError::Cancelled)));
    assert_eq!(second.wait().unwrap().output_bytes, b"join-before-cancel");
    assert_eq!(blocked.service.active_subscriptions(), 0);
}

#[test]
fn cancelling_last_subscriber_retires_the_job_before_a_new_join() {
    let blocked = BlockedService::new(
        2,
        BakeSubscriptionLimits {
            max_total: 4,
            max_per_job: 2,
        },
    );
    let input = request(b"cancel-before-join");
    let first = blocked.service.submit(input.clone()).unwrap();
    let first_token = blocked.entered();
    assert!(first.cancel());
    assert!(first_token.is_cancelled());
    let second = blocked.service.submit(input).unwrap();
    let second_token = blocked.entered();
    assert!(!second_token.is_cancelled());
    assert_eq!(blocked.calls.load(Ordering::SeqCst), 2);
    blocked.gate.release();
    assert!(matches!(first.wait(), Err(BakeError::Cancelled)));
    assert_eq!(second.wait().unwrap().output_bytes, b"cancel-before-join");
    assert_eq!(blocked.service.active_subscriptions(), 0);
}

#[test]
fn concurrent_last_cancel_and_same_key_join_never_cancel_the_fresh_handle() {
    for _ in 0..32 {
        let blocked = BlockedService::new(
            2,
            BakeSubscriptionLimits {
                max_total: 4,
                max_per_job: 2,
            },
        );
        let input = request(b"racing-join");
        let first = blocked.service.submit(input.clone()).unwrap();
        blocked.entered();
        let start = Arc::new(Barrier::new(3));
        let cancel_start = Arc::clone(&start);
        let cancel = thread::spawn(move || {
            cancel_start.wait();
            assert!(first.cancel());
            first
        });
        let join_start = Arc::clone(&start);
        let service = blocked.service.clone();
        let join = thread::spawn(move || {
            join_start.wait();
            service.submit(input)
        });
        start.wait();
        let first = cancel.join().unwrap();
        let second = join.join().unwrap().unwrap();
        blocked.gate.release();
        assert!(matches!(first.wait(), Err(BakeError::Cancelled)));
        assert_eq!(second.wait().unwrap().output_bytes, b"racing-join");
        assert_eq!(blocked.service.active_subscriptions(), 0);
    }
}
