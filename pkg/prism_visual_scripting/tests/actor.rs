use async_trait::async_trait;
use prism_visual_scripting::prelude::*;
use std::sync::Arc;

#[test]
fn lease_expiry_overflow_is_rejected_without_installing_lease() {
    let mut actor = ActorState::new(ActorId::new("overflow", "1"), Value::Unit);
    let error = actor.acquire_lease("worker", u64::MAX, 1).unwrap_err();
    assert!(error.to_string().contains("lease expiry exhausted"));
    assert!(actor.lease.is_none());

    let lease = actor.acquire_lease("worker", u64::MAX - 2, 1).unwrap();
    let error = actor
        .renew_lease(&lease.owner, &lease.token, u64::MAX - 2, 3)
        .unwrap_err();
    assert!(error.to_string().contains("lease expiry exhausted"));
    assert_eq!(actor.lease.as_ref().unwrap().expires_at_ms, u64::MAX - 1);
}

#[test]
fn actor_mailbox_is_idempotent_and_commits_with_fencing() {
    let mut actor = ActorState::new(ActorId::new("order", "42"), Value::I64(0));
    let first = actor.acquire_lease("worker-a", 100, 1000).unwrap();
    assert!(actor
        .enqueue(
            &first.owner,
            &first.token,
            101,
            "event-1",
            "order.created",
            Value::I64(7),
        )
        .unwrap());
    assert!(!actor
        .enqueue(
            &first.owner,
            &first.token,
            102,
            "event-1",
            "order.created",
            Value::I64(999),
        )
        .unwrap());
    let signal = actor
        .claim_next(&first.owner, &first.token, 103)
        .unwrap()
        .unwrap();
    assert_eq!(signal.sequence, 1);
    actor
        .commit(
            &first.owner,
            &first.token,
            104,
            &signal.signal_id,
            Value::I64(7),
        )
        .unwrap();
    assert_eq!(actor.revision, 1);
    assert_eq!(actor.value, Value::I64(7));
}

#[test]
fn expired_lease_fences_old_worker_and_allows_takeover() {
    let mut actor = ActorState::new(ActorId::new("deployment", "prod"), Value::Unit);
    let old = actor.acquire_lease("worker-a", 0, 10).unwrap();
    let new = actor.acquire_lease("worker-b", 10, 10).unwrap();
    assert_ne!(old.token, new.token);
    let error = actor
        .enqueue(
            &old.owner,
            &old.token,
            11,
            "event-1",
            "deploy.requested",
            Value::Unit,
        )
        .unwrap_err();
    assert!(error.to_string().contains("fencing token"));
    assert!(actor
        .enqueue(
            &new.owner,
            &new.token,
            11,
            "event-1",
            "deploy.requested",
            Value::Unit,
        )
        .unwrap());
}

#[test]
fn actor_state_round_trips_through_store_and_reclaims_in_flight_signal() {
    let store = InMemoryActorStore::default();
    let id = ActorId::new("ticket", "7");
    let mut actor = ActorState::new(id.clone(), Value::String("open".into()));
    let first = actor.acquire_lease("worker-a", 0, 10).unwrap();
    actor
        .enqueue(
            &first.owner,
            &first.token,
            1,
            "signal-1",
            "ticket.updated",
            Value::String("pending".into()),
        )
        .unwrap();
    actor.claim_next(&first.owner, &first.token, 2).unwrap();
    store.save(actor).unwrap();
    let mut restored = store.load(&id).unwrap();
    let second = restored.acquire_lease("worker-b", 10, 10).unwrap();
    let signal = restored
        .claim_next(&second.owner, &second.token, 11)
        .unwrap()
        .unwrap();
    assert_eq!(signal.signal_id, "signal-1");
    restored
        .commit(
            &second.owner,
            &second.token,
            12,
            "signal-1",
            Value::String("closed".into()),
        )
        .unwrap();
    assert_eq!(restored.value, Value::String("closed".into()));
}

struct AddHandler;

#[async_trait]
impl ActorHandler for AddHandler {
    async fn handle(&self, state: &ActorState, signal: &ActorSignal) -> VsResult<Value> {
        let current = state.value.as_i64().unwrap();
        let delta = signal.payload.as_i64().unwrap();
        Ok(Value::I64(current + delta))
    }
}

#[tokio::test]
async fn durable_actor_runtime_dispatches_mailbox_and_commits_state() {
    let store: Arc<dyn DurableStore> = Arc::new(InMemoryDurableStore::default());
    let runtime = DurableActorRuntime::new(store);
    let id = ActorId::new("counter", "7");
    runtime.create(id.clone(), Value::I64(1)).unwrap();
    assert!(runtime
        .enqueue(
            &id,
            "worker",
            0,
            100,
            ActorMessage {
                signal_id: "add-1".into(),
                topic: "add".into(),
                payload: Value::I64(4),
            },
        )
        .unwrap());
    let signal = runtime
        .process_next(&id, "worker", 100, &VirtualClock::new(1), &AddHandler)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(signal.signal_id, "add-1");
    let state = runtime.load(&id).unwrap().unwrap();
    assert_eq!(state.value, Value::I64(5));
    assert_eq!(state.revision, 1);
    assert!(state.in_flight.is_none());
}

struct ExpiringHandler<'a> {
    clock: &'a VirtualClock,
}

#[async_trait]
impl ActorHandler for ExpiringHandler<'_> {
    async fn handle(&self, _state: &ActorState, _signal: &ActorSignal) -> VsResult<Value> {
        self.clock.advance(10);
        Ok(Value::I64(99))
    }
}

#[tokio::test]
async fn durable_actor_runtime_fences_a_handler_that_outlives_its_lease() {
    let store: Arc<dyn DurableStore> = Arc::new(InMemoryDurableStore::default());
    let runtime = DurableActorRuntime::new(store);
    let id = ActorId::new("counter", "expired");
    runtime.create(id.clone(), Value::I64(1)).unwrap();
    runtime
        .enqueue(
            &id,
            "worker",
            0,
            100,
            ActorMessage {
                signal_id: "add-1".into(),
                topic: "add".into(),
                payload: Value::I64(4),
            },
        )
        .unwrap();
    let clock = VirtualClock::new(1);
    let error = runtime
        .process_next(&id, "worker", 5, &clock, &ExpiringHandler { clock: &clock })
        .await
        .unwrap_err();
    assert!(error.to_string().contains("lease expired"));
    let state = runtime.load(&id).unwrap().unwrap();
    assert_eq!(state.value, Value::I64(1));
    assert_eq!(state.in_flight.unwrap().signal_id, "add-1");
}
