//! Real frame consumption with controlled worker interleavings.
#![cfg(feature = "ecs")]

use prism_app::{App, Update};
use prism_asset_build_core::{
    BakeEcsConfig, BakeEcsConfigError, BakeEcsError, BakeEcsPlugin, BakeError, BakeHandle,
    BakeKernelRegistry, BakeOutput, BakeRequest, BakeRequestComponent, BakeRevision, BakeRuntime,
    BakeService, BakeState, BakeStatus, BakeSubscriptionLimits, CancellationToken, ContentHash,
    KernelDispatchError, KernelError, RuntimeCacheConfig, StableInputKey,
};
use prism_ecs::prelude::{Entity, Resource, World};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

fn config() -> BakeEcsConfig {
    BakeEcsConfig {
        max_queued: 8,
        max_pending: 8,
        max_admissions_per_frame: 8,
        max_completions_per_frame: 8,
    }
}

fn request(bytes: &[u8]) -> BakeRequest {
    BakeRequest::new(
        StableInputKey::from_bytes(bytes),
        bytes.to_vec(),
        17,
        23,
        1,
        b"canonical".to_vec(),
        31,
        2,
        16,
    )
}

fn app(service: &BakeService, config: BakeEcsConfig) -> App {
    let mut app = App::new();
    app.add_plugins(BakeEcsPlugin::new(service.clone(), config).unwrap());
    app
}

fn put(app: &mut App, entity: Entity, revision: u128, bytes: &[u8]) {
    assert!(BakeRuntime::insert_request(
        app.world_mut(),
        entity,
        BakeRequestComponent::new(BakeRevision(revision), request(bytes))
    ));
}

fn spawn(app: &mut App, revision: u128, bytes: &[u8]) -> Entity {
    let entity = app.world_mut().spawn(());
    put(app, entity, revision, bytes);
    entity
}

fn state(app: &App, entity: Entity) -> &BakeState {
    &app.world().get::<BakeStatus>(entity).unwrap().state
}

fn until(mut condition: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while !condition() {
        assert!(
            Instant::now() < deadline,
            "controlled worker did not progress"
        );
        std::thread::yield_now();
    }
}

fn update_until(app: &mut App, condition: impl Fn(&App) -> bool) {
    until(|| {
        app.update();
        condition(app)
    });
}

#[derive(Default)]
struct Gate {
    released: Mutex<bool>,
    wake: Condvar,
    token: Mutex<Option<CancellationToken>>,
    entered: AtomicUsize,
    calls: AtomicUsize,
}
impl Gate {
    fn release(&self) {
        *self.released.lock().unwrap() = true;
        self.wake.notify_all();
    }
    fn cancelled(&self) -> bool {
        self.token
            .lock()
            .unwrap()
            .as_ref()
            .is_some_and(CancellationToken::is_cancelled)
    }
    fn bake(
        &self,
        request: &BakeRequest,
        token: &CancellationToken,
    ) -> Result<Vec<u8>, KernelError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if request.input_bytes == b"slow" {
            *self.token.lock().unwrap() = Some(token.clone());
            self.entered.fetch_add(1, Ordering::Release);
            let (released, timeout) = self
                .wake
                .wait_timeout_while(
                    self.released.lock().unwrap(),
                    Duration::from_secs(5),
                    |released| !*released,
                )
                .unwrap();
            if timeout.timed_out() && !*released {
                return Err(KernelError::new("test gate timeout"));
            }
        }
        if request.input_bytes == b"fail" {
            Err(KernelError::new("controlled failure"))
        } else {
            Ok(request.input_bytes.clone())
        }
    }
}

// Releases gates before the final service/pool reference drops, including on
// assertion failure. No test can leave a worker waiting forever.
struct Controlled {
    service: BakeService,
    gate: Arc<Gate>,
}
impl Controlled {
    fn new(workers: usize) -> Self {
        let gate = Arc::new(Gate::default());
        let worker_gate = gate.clone();
        let service = BakeService::new_with_cache(
            workers,
            RuntimeCacheConfig::new(16, 4096),
            Arc::new(move |request: &BakeRequest, token: &CancellationToken| {
                worker_gate.bake(request, token)
            }),
        )
        .unwrap();
        Self { service, gate }
    }
    fn started(&self) {
        until(|| self.gate.entered.load(Ordering::Acquire) != 0);
    }
    fn seed(&self, bytes: &[u8]) {
        self.service.submit(request(bytes)).unwrap().wait().unwrap();
    }
}
impl Drop for Controlled {
    fn drop(&mut self) {
        self.gate.release();
    }
}

#[test]
fn zero_budgets_are_rejected_and_subscription_limits_are_root_exported() {
    let controlled = Controlled::new(1);
    let limits: BakeSubscriptionLimits = controlled.service.subscription_limits();
    assert!(limits.max_total > 0 && limits.max_per_job > 0);
    for field in 0..4 {
        let mut bounds = config();
        match field {
            0 => bounds.max_queued = 0,
            1 => bounds.max_pending = 0,
            2 => bounds.max_admissions_per_frame = 0,
            _ => bounds.max_completions_per_frame = 0,
        }
        assert!(matches!(
            BakeEcsPlugin::new(controlled.service.clone(), bounds),
            Err(BakeEcsConfigError::ZeroBudget)
        ));
    }
}

#[test]
fn cached_owned_output_and_revision_are_consumed_once_by_real_updates() {
    let controlled = Controlled::new(1);
    controlled.seed(b"result");
    let mut app = app(&controlled.service, config());
    let entity = spawn(&mut app, 101, b"result");
    assert!(matches!(state(&app, entity), BakeState::Queued));
    app.update();
    assert!(matches!(state(&app, entity), BakeState::Running));
    assert!(app.world().get::<BakeOutput>(entity).is_none());
    app.update();
    let output = app.world().get::<BakeOutput>(entity).unwrap();
    assert_eq!(output.revision, BakeRevision(101));
    assert_eq!(output.result.key, request(b"result").request_key());
    assert_eq!(output.result.output_bytes, b"result");
    assert_eq!(
        output.result.descriptor.content_hash,
        ContentHash::digest(b"result")
    );
    assert_eq!(
        (
            output.result.descriptor.kind,
            output.result.descriptor.schema,
            output.result.descriptor.alignment
        ),
        (31, 2, 16)
    );
    assert!(matches!(state(&app, entity), BakeState::Ready));
    for _ in 0..4 {
        app.update();
    }
    let runtime = app.world().resource::<BakeRuntime>();
    assert_eq!(
        (
            runtime.queued(),
            runtime.pending(),
            runtime.frame().admissions
        ),
        (0, 0, 0)
    );
    assert_eq!(controlled.gate.calls.load(Ordering::SeqCst), 1);
}

#[derive(Default)]
struct Frames(usize);
impl Resource for Frames {}
fn count_frame(world: &mut World) {
    world.resource_mut::<Frames>().0 += 1;
}

#[test]
fn blocked_kernel_never_blocks_update_or_a_later_ready_ticket() {
    let controlled = Controlled::new(2);
    controlled.seed(b"fast");
    let mut app = app(&controlled.service, config());
    app.world_mut().insert_resource(Frames::default());
    app.add_systems(Update, count_frame);
    let slow = spawn(&mut app, 1, b"slow");
    app.update();
    controlled.started();
    let fast = spawn(&mut app, 2, b"fast");
    app.update();
    app.update();
    assert!(matches!(state(&app, slow), BakeState::Running));
    assert!(matches!(state(&app, fast), BakeState::Ready));
    assert_eq!(app.world().resource::<Frames>().0, 3);
    assert!(!*controlled.gate.released.lock().unwrap());
    controlled.gate.release();
    update_until(&mut app, |app| matches!(state(app, slow), BakeState::Ready));
}

#[test]
fn same_revision_replacement_rejects_old_completion_and_keeps_new_output() {
    let controlled = Controlled::new(2);
    controlled.seed(b"new");
    let mut app = app(&controlled.service, config());
    let entity = spawn(&mut app, 7, b"slow");
    app.update();
    controlled.started();
    put(&mut app, entity, 7, b"new");
    assert!(
        controlled.gate.cancelled(),
        "replacement cancels before another frame"
    );
    app.update();
    app.update();
    assert_eq!(
        app.world()
            .get::<BakeOutput>(entity)
            .unwrap()
            .result
            .output_bytes,
        b"new"
    );
    controlled.gate.release();
    update_until(&mut app, |app| {
        app.world().resource::<BakeRuntime>().pending() == 0
    });
    assert!(matches!(state(&app, entity), BakeState::Ready));
    assert_eq!(
        app.world().get::<BakeOutput>(entity).unwrap().revision,
        BakeRevision(7)
    );
    assert_eq!(
        app.world()
            .get::<BakeOutput>(entity)
            .unwrap()
            .result
            .output_bytes,
        b"new"
    );
}

#[test]
fn cancellation_after_delivery_and_replacement_before_drain_preserve_last_good() {
    let controlled = Controlled::new(1);
    controlled.seed(b"good");
    controlled.seed(b"obsolete");
    let mut app = app(&controlled.service, config());
    let entity = spawn(&mut app, 1, b"good");
    app.update();
    app.update();
    put(&mut app, entity, 2, b"obsolete");
    app.update(); // cache-hit result has already been delivered to its ticket
    BakeRuntime::request(app.world(), entity).unwrap().cancel();
    app.update();
    assert!(matches!(state(&app, entity), BakeState::Cancelled));
    assert_eq!(app.world().resource::<BakeRuntime>().frame().discarded, 1);
    assert_eq!(
        app.world().get::<BakeOutput>(entity).unwrap().revision,
        BakeRevision(1)
    );
    put(&mut app, entity, 3, b"obsolete");
    app.update();
    put(&mut app, entity, 3, b"good"); // same revision, distinct request instance
    app.update();
    assert_eq!(app.world().resource::<BakeRuntime>().frame().discarded, 1);
    app.update();
    assert_eq!(
        app.world()
            .get::<BakeOutput>(entity)
            .unwrap()
            .result
            .output_bytes,
        b"good"
    );
}

#[test]
fn removal_and_despawn_cancel_immediately_and_entity_reuse_is_safe() {
    let controlled = Controlled::new(2);
    controlled.seed(b"good");
    let mut app = app(&controlled.service, config());
    let entity = spawn(&mut app, 1, b"good");
    app.update();
    app.update();
    put(&mut app, entity, 2, b"slow");
    app.update();
    controlled.started();
    assert!(BakeRuntime::remove_request(app.world_mut(), entity));
    assert!(controlled.gate.cancelled());
    assert!(matches!(state(&app, entity), BakeState::Cancelled));
    assert_eq!(
        app.world().get::<BakeOutput>(entity).unwrap().revision,
        BakeRevision(1)
    );
    put(&mut app, entity, 2, b"good");
    app.update();
    app.update();
    assert_eq!(
        app.world().get::<BakeOutput>(entity).unwrap().revision,
        BakeRevision(2)
    );
    assert!(app.world_mut().despawn(entity));
    let recycled = spawn(&mut app, 9, b"good");
    assert_eq!(recycled.index(), entity.index());
    assert_ne!(recycled.generation(), entity.generation());
    controlled.gate.release();
    update_until(&mut app, |app| {
        matches!(state(app, recycled), BakeState::Ready)
            && app.world().resource::<BakeRuntime>().pending() == 0
    });
    assert!(app.world().get::<BakeOutput>(entity).is_none());
    assert_eq!(
        app.world().get::<BakeOutput>(recycled).unwrap().revision,
        BakeRevision(9)
    );
}

#[test]
fn despawn_cancels_running_work_and_frees_queued_admission_immediately() {
    let controlled = Controlled::new(1);
    let mut bounds = config();
    bounds.max_queued = 1;
    let mut app = app(&controlled.service, bounds);
    let running = spawn(&mut app, 1, b"slow");
    app.update();
    controlled.started();
    let queued = spawn(&mut app, 2, b"never submitted");
    assert!(app.world_mut().despawn(queued));
    assert_eq!(app.world().resource::<BakeRuntime>().queued(), 0);
    let replacement = spawn(&mut app, 3, b"also queued");
    assert!(matches!(state(&app, replacement), BakeState::Queued));
    assert!(app.world_mut().despawn(running));
    assert!(controlled.gate.cancelled());
    assert_eq!(app.world().resource::<BakeRuntime>().pending(), 1);
    assert_eq!(controlled.service.in_flight(), 1);
    assert!(BakeRuntime::remove_request(app.world_mut(), replacement));
    controlled.gate.release();
    update_until(&mut app, |app| {
        app.world().resource::<BakeRuntime>().pending() == 0
    });
    assert_eq!(controlled.gate.calls.load(Ordering::SeqCst), 1);
}

#[test]
fn cancellation_before_admission_never_submits_or_publishes() {
    let controlled = Controlled::new(1);
    let mut app = app(&controlled.service, config());
    let cancelled = BakeRequestComponent::new(BakeRevision(1), request(b"never submitted"));
    cancelled.cancel();
    let before_insert = app.world_mut().spawn(());
    assert!(BakeRuntime::insert_request(
        app.world_mut(),
        before_insert,
        cancelled
    ));
    assert!(matches!(state(&app, before_insert), BakeState::Cancelled));
    assert_eq!(app.world().resource::<BakeRuntime>().queued(), 0);
    let before_update = spawn(&mut app, 2, b"also never submitted");
    BakeRuntime::request(app.world(), before_update)
        .unwrap()
        .cancel();
    app.update();
    assert!(matches!(state(&app, before_update), BakeState::Cancelled));
    assert_eq!(app.world().resource::<BakeRuntime>().queued(), 0);
    assert_eq!(app.world().resource::<BakeRuntime>().pending(), 0);
    assert_eq!(controlled.gate.calls.load(Ordering::SeqCst), 0);
    assert!(app.world().get::<BakeOutput>(before_insert).is_none());
    assert!(app.world().get::<BakeOutput>(before_update).is_none());
}

#[test]
fn dropping_runtime_cancels_workers_and_disables_frame_admission() {
    let controlled = Controlled::new(1);
    let mut app = app(&controlled.service, config());
    let running = spawn(&mut app, 1, b"slow");
    app.update();
    controlled.started();
    let queued = spawn(&mut app, 2, b"never submitted");
    drop(app.world_mut().remove_resource::<BakeRuntime>().unwrap());
    assert!(controlled.gate.cancelled());
    assert_eq!(controlled.service.active_subscriptions(), 0);
    assert_eq!(controlled.service.in_flight(), 1);
    app.update();
    assert_eq!(controlled.gate.calls.load(Ordering::SeqCst), 1);
    assert!(app.world().get::<BakeOutput>(running).is_none());
    assert!(app.world().get::<BakeOutput>(queued).is_none());
    assert!(!BakeRuntime::insert_request(
        app.world_mut(),
        queued,
        BakeRequestComponent::new(BakeRevision(3), request(b"rejected"))
    ));
    controlled.gate.release();
    until(|| controlled.service.in_flight() == 0);
}

#[test]
fn failure_validation_and_dispatch_errors_are_observable_and_keep_last_good() {
    let controlled = Controlled::new(1);
    controlled.seed(b"good");
    let mut app = app(&controlled.service, config());
    let entity = spawn(&mut app, 1, b"good");
    app.update();
    app.update();
    put(&mut app, entity, 2, b"fail");
    update_until(&mut app, |app| {
        matches!(state(app, entity), BakeState::Failed(_))
    });
    assert!(
        matches!(state(&app, entity), BakeState::Failed(BakeEcsError::Bake(BakeError::Kernel(error))) if error.message() == "controlled failure")
    );
    let mut invalid = request(b"invalid");
    invalid.output_alignment = 3;
    assert!(BakeRuntime::insert_request(
        app.world_mut(),
        entity,
        BakeRequestComponent::new(BakeRevision(3), invalid)
    ));
    app.update();
    assert!(matches!(
        state(&app, entity),
        BakeState::Failed(BakeEcsError::Bake(BakeError::InvalidRequest(_)))
    ));
    assert_eq!(
        app.world().get::<BakeOutput>(entity).unwrap().revision,
        BakeRevision(1)
    );
    let registry = Arc::new(BakeKernelRegistry::new());
    let service =
        BakeService::new_with_registry(1, RuntimeCacheConfig::disabled(), registry).unwrap();
    let mut unknown_app = self::app(&service, config());
    let unknown = spawn(&mut unknown_app, 8, b"unknown");
    unknown_app.update();
    assert!(matches!(
        state(&unknown_app, unknown),
        BakeState::Failed(BakeEcsError::Bake(BakeError::KernelDispatch(
            KernelDispatchError::UnknownKernel { kernel_id: 17 }
        )))
    ));
}

#[test]
fn bounded_hook_queue_and_frame_budgets_do_not_scan_or_retry_all_requests() {
    let controlled = Controlled::new(1);
    controlled.seed(b"cached");
    let mut bounds = config();
    bounds.max_queued = 3;
    bounds.max_admissions_per_frame = 1;
    bounds.max_completions_per_frame = 1;
    let mut app = app(&controlled.service, bounds);
    let first = spawn(&mut app, 1, b"cached");
    let removed = spawn(&mut app, 2, b"cached");
    let third = spawn(&mut app, 3, b"cached");
    let overflow = spawn(&mut app, 4, b"cached");
    assert!(matches!(
        state(&app, overflow),
        BakeState::Failed(BakeEcsError::QueueFull { max_queued: 3 })
    ));
    assert!(BakeRuntime::remove_request(app.world_mut(), removed));
    assert_eq!(app.world().resource::<BakeRuntime>().queued(), 2);
    for expected in [1, 0] {
        app.update();
        let runtime = app.world().resource::<BakeRuntime>();
        assert_eq!(runtime.frame().admissions, 1);
        assert!(runtime.frame().completions <= 1);
        assert_eq!(runtime.queued(), expected);
    }
    app.update();
    assert!(matches!(state(&app, first), BakeState::Ready));
    assert!(matches!(state(&app, third), BakeState::Ready));
    assert!(matches!(
        state(&app, overflow),
        BakeState::Failed(BakeEcsError::QueueFull { .. })
    ));
    assert_eq!(app.world().resource::<BakeRuntime>().frame().admissions, 0);
    put(&mut app, overflow, 4, b"cached");
    app.update();
    app.update();
    assert!(matches!(state(&app, overflow), BakeState::Ready));
}

#[test]
fn completion_budget_counts_stale_messages_and_retains_undrained_results() {
    let controlled = Controlled::new(1);
    controlled.seed(b"cached");
    let mut bounds = config();
    bounds.max_completions_per_frame = 1;
    let mut app = app(&controlled.service, bounds);
    let stale = spawn(&mut app, 1, b"cached");
    let second = spawn(&mut app, 2, b"cached");
    let third = spawn(&mut app, 3, b"cached");
    app.update();
    assert!(BakeRuntime::remove_request(app.world_mut(), stale));
    for (pending, applied) in [(2, 0), (1, 1), (0, 1)] {
        app.update();
        let runtime = app.world().resource::<BakeRuntime>();
        assert_eq!(runtime.pending(), pending);
        assert_eq!(
            (runtime.frame().completions, runtime.frame().applied),
            (1, applied)
        );
    }
    assert!(app.world().get::<BakeOutput>(stale).is_none());
    assert!(matches!(state(&app, second), BakeState::Ready));
    assert!(matches!(state(&app, third), BakeState::Ready));
}

#[test]
fn cancelled_ticket_retains_capacity_until_worker_finishes_and_failures_do_not_retry() {
    let controlled = Controlled::new(1);
    let mut bounds = config();
    bounds.max_pending = 1;
    let mut app = app(&controlled.service, bounds);
    let slow = spawn(&mut app, 1, b"slow");
    app.update();
    controlled.started();
    BakeRuntime::request(app.world(), slow).unwrap().cancel();
    app.update();
    assert_eq!(controlled.service.in_flight(), 1);
    assert_eq!(app.world().resource::<BakeRuntime>().pending(), 1);
    let rejected = spawn(&mut app, 2, b"fast");
    app.update();
    assert!(matches!(
        state(&app, rejected),
        BakeState::Failed(BakeEcsError::AtCapacity { max_pending: 1 })
    ));
    for _ in 0..5 {
        app.update();
    }
    assert_eq!(controlled.service.in_flight(), 1);
    assert_eq!(app.world().resource::<BakeRuntime>().pending(), 1);
    controlled.gate.release();
    update_until(&mut app, |app| {
        app.world().resource::<BakeRuntime>().pending() == 0
    });
    assert!(matches!(state(&app, rejected), BakeState::Failed(_)));
    put(&mut app, rejected, 2, b"fast");
    update_until(&mut app, |app| {
        matches!(state(app, rejected), BakeState::Ready)
    });
}

#[test]
fn cancelling_one_entity_does_not_cancel_another_subscriber_of_shared_work() {
    let controlled = Controlled::new(1);
    let mut app = app(&controlled.service, config());
    let cancelled = spawn(&mut app, 1, b"slow");
    let live = spawn(&mut app, 2, b"slow");
    app.update();
    controlled.started();
    assert!(BakeRuntime::remove_request(app.world_mut(), cancelled));
    assert!(!controlled.gate.cancelled());
    controlled.gate.release();
    update_until(&mut app, |app| {
        app.world().resource::<BakeRuntime>().pending() == 0
    });
    assert!(app.world().get::<BakeOutput>(cancelled).is_none());
    assert!(matches!(state(&app, live), BakeState::Ready));
    assert_eq!(controlled.gate.calls.load(Ordering::SeqCst), 1);
}

#[test]
fn plugin_shutdown_cancels_running_and_queued_work_and_prevents_future_admission() {
    let controlled = Controlled::new(1);
    let mut bounds = config();
    bounds.max_admissions_per_frame = 1;
    let mut app = app(&controlled.service, bounds);
    let slow = spawn(&mut app, 1, b"slow");
    app.update();
    controlled.started();
    let queued = spawn(&mut app, 2, b"fast");
    app.run_shutdown();
    assert!(controlled.gate.cancelled());
    assert!(matches!(state(&app, slow), BakeState::Cancelled));
    assert!(matches!(state(&app, queued), BakeState::Cancelled));
    assert_eq!(app.world().resource::<BakeRuntime>().queued(), 0);
    let after = spawn(&mut app, 3, b"after");
    assert!(matches!(state(&app, after), BakeState::Cancelled));
    app.update();
    assert_eq!(app.world().resource::<BakeRuntime>().frame().admissions, 0);
    assert_eq!(controlled.gate.calls.load(Ordering::SeqCst), 1);
    controlled.gate.release();
    update_until(&mut app, |app| {
        app.world().resource::<BakeRuntime>().pending() == 0
    });
    assert!(app.world().get::<BakeOutput>(slow).is_none());
}

#[test]
fn nonblocking_ticket_poll_reports_pending_success_and_cancellation() {
    let controlled = Controlled::new(1);
    let ticket: BakeHandle = controlled.service.submit(request(b"slow")).unwrap();
    controlled.started();
    assert!(ticket.try_recv().is_none());
    assert!(ticket.cancel());
    assert!(
        ticket.try_recv().is_none(),
        "cancellation does not finish a running worker"
    );
    assert_eq!(controlled.service.in_flight(), 1);
    controlled.gate.release();
    let mut terminal = None;
    until(|| {
        terminal = ticket.try_recv();
        terminal.is_some()
    });
    assert!(matches!(terminal.unwrap(), Err(BakeError::Cancelled)));
    drop(ticket);
    controlled.seed(b"ready");
    let ready = controlled.service.submit(request(b"ready")).unwrap();
    assert_eq!(ready.try_recv().unwrap().unwrap().output_bytes, b"ready");
}
