//! Client-side atomic authority hand-off regressions. Driver internals are
//! deliberately exercised only through the public transaction API.

use super::super::ecs::{DeferredEcsError, NetworkRollbackDriver};
use super::tests::{int, schema, snapshot_at};
use super::*;
use crate::delta::FieldValue;
use crate::frame::encode_frame;
use crate::session::ServerSession;
use crate::transport::memory::InMemoryTransport;
use prism_app::{EcsRollbackError, SnapshotHistory};
use prism_ecs::{component::Component, entity::Entity, world::World};
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc,
};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Counter(i64);
impl Component for Counter {}

struct Unregistered;
impl Component for Unregistered {}

struct Strategy {
    entity: Entity,
    authority_calls: Arc<AtomicUsize>,
    input_calls: Arc<AtomicUsize>,
    fail_authority: Arc<AtomicBool>,
}

impl ReplicationStrategy for Strategy {
    type Input = i64;

    fn simulate(&self, record: &mut Record, input: &i64) {
        record.set(0, FieldValue::Int(int(record) + input));
    }

    fn encode(&self, input: &i64) -> Vec<u8> {
        input.to_le_bytes().to_vec()
    }
}

impl EcsReplicationStrategy for Strategy {
    fn apply_input(&self, world: &mut World, input: &i64) {
        self.input_calls.fetch_add(1, Ordering::SeqCst);
        world.get_mut::<Counter>(self.entity).unwrap().0 += input;
    }

    fn apply_authority(&self, world: &mut World, state: &Record) {
        self.authority_calls.fetch_add(1, Ordering::SeqCst);
        world.get_mut::<Counter>(self.entity).unwrap().0 = int(state);
        if self.fail_authority.load(Ordering::SeqCst) {
            world.spawn(Unregistered);
        }
    }
}

struct Fixture {
    app: App,
    client: Replication<Strategy>,
    driver: NetworkRollbackDriver<i64>,
    peer: ChannelMux<Box<dyn Transport>>,
}

impl Fixture {
    fn new(queue_limit: usize) -> Self {
        let mut app = App::new();
        let entity = app.world_mut().spawn(Counter(0));
        app.world_mut()
            .register_snapshot_component_hashable::<Counter>();
        app.world_mut().insert_resource(SnapshotHistory::new(16));
        let strategy = Strategy {
            entity,
            authority_calls: Arc::new(AtomicUsize::new(0)),
            input_calls: Arc::new(AtomicUsize::new(0)),
            fail_authority: Arc::new(AtomicBool::new(false)),
        };
        let (local, remote) = InMemoryTransport::pair_secure();
        let mut config = ReplicationConfig::for_capacity(16);
        config.mux.reliable_queue = queue_limit;
        let client = Replication::new_with_config(
            schema(),
            16,
            &snapshot_at(0, 0),
            strategy,
            Box::new(local),
            config,
        )
        .unwrap();
        let mut peer_config = config.mux;
        peer_config.reliable_queue = 64;
        let mut peer = ChannelMux::new(Box::new(remote) as Box<dyn Transport>, peer_config);
        peer.send(
            NetChannel::RpcReliable,
            &super::super::encode_accept(NegotiatedProtocol {
                version: config.protocol.version,
                rollback_window: config.protocol.rollback_window,
            }),
        )
        .unwrap();
        let mut result = Self {
            app,
            client,
            driver: NetworkRollbackDriver::new(),
            peer,
        };
        result.pump();
        assert!(result.client.negotiated().is_some());
        result
    }

    fn pump(&mut self) {
        for _ in 0..12 {
            self.peer.tick();
            self.client.pump_deferred();
        }
    }

    fn predict(&mut self, input: i64) {
        self.driver.commit(self.app.world_mut(), input).unwrap();
        self.client.predict(input).unwrap();
        self.client
            .apply_input_to_world(self.app.world_mut(), &input);
    }

    fn reconcile(&mut self) -> Result<bool, DeferredEcsError> {
        self.client
            .reconcile_deferred_ecs(self.app.world_mut(), &mut self.driver)
    }

    fn value(&self) -> i64 {
        self.app
            .world()
            .get::<Counter>(self.client.strategy.entity)
            .unwrap()
            .0
    }

    fn history(&self) -> &SnapshotHistory {
        self.app.world().resource::<SnapshotHistory>()
    }

    fn send_delta(&mut self, delta: &[u8]) {
        self.peer
            .send(
                NetChannel::EventsReliable,
                &super::super::encode_delta(delta),
            )
            .unwrap();
    }

    fn send_snapshot(&mut self, snapshot: &[u8]) {
        self.peer
            .send(
                NetChannel::EventsReliable,
                &super::super::encode_snapshot(snapshot),
            )
            .unwrap();
    }
}

fn snapshot(revision: u32, value: i64) -> Vec<u8> {
    let mut framed = Vec::new();
    encode_frame(&snapshot_at(revision, value), &mut framed).unwrap();
    framed
}

fn server() -> ServerSession {
    ServerSession::new(schema(), 16, &snapshot_at(0, 0)).unwrap()
}

fn advance(server: &mut ServerSession, value: i64) -> Vec<u8> {
    let mut framed = Vec::new();
    server
        .advance(
            |record| {
                record.set(0, FieldValue::Int(value));
            },
            &mut framed,
        )
        .unwrap();
    framed
}

#[test]
fn deferred_queue_is_bounded_and_drains_without_losing_authority() {
    let mut f = Fixture::new(2);
    let mut server = server();
    for revision in 1..=8 {
        f.predict(1);
        f.send_delta(&advance(&mut server, revision * 3));
    }
    f.pump();
    assert_eq!(f.client.deferred_authority.len(), 2);
    assert_eq!(f.client.acked(), 0);
    for _ in 0..30 {
        while f.reconcile().unwrap() {}
        f.pump();
        assert!(f.client.deferred_authority.len() <= 2);
    }
    assert_eq!(f.client.acked(), 8);
    assert_eq!(f.value(), 24);
    assert_eq!(int(f.client.state().unwrap()), 24);
}

#[test]
fn snapshot_is_staged_then_atomically_resets_zero_and_nonzero_baselines() {
    for revision in [0, 42] {
        let mut f = Fixture::new(2);
        f.predict(5);
        f.predict(1);
        f.client.authority_updates.push_back(AuthorityUpdate {
            revision: 1,
            state: Record::new(alloc::vec![FieldValue::Int(99)]),
            _input: core::marker::PhantomData,
        });
        f.send_snapshot(&snapshot(revision, 30));
        f.pump();
        assert!(f.client.deferred_reseed_pending());
        assert_eq!(
            (f.client.acked(), f.client.predicted(), f.value()),
            (0, 2, 6)
        );
        assert_eq!(f.history().len(), 2);
        assert_eq!(f.driver.inputs(), &[5, 1]);
        assert!(f.reconcile().unwrap());
        assert_eq!(
            (f.client.acked(), f.client.predicted(), f.value()),
            (revision, revision, 30)
        );
        assert!(f.history().is_empty());
        assert_eq!(f.history().next_frame(), u64::from(revision));
        assert_eq!(f.driver.input_start(), u64::from(revision));
        assert!(f.driver.inputs().is_empty());
        assert!(f.client.recv_authority_update().is_none());
        assert!(!f.client.deferred_reseed_pending());
        // A new prediction and delta must use the new absolute input base.
        f.predict(2);
        let mut authority = ServerSession::new(schema(), 16, &snapshot_at(revision, 30)).unwrap();
        f.send_delta(&advance(&mut authority, 35));
        f.pump();
        assert!(f.reconcile().unwrap());
        assert_eq!(f.value(), 35);
        assert_eq!(f.client.acked(), revision + 1);
        assert_eq!(f.history().next_frame(), u64::from(revision + 1));
    }
}

#[test]
fn failed_reset_preserves_session_world_history_and_inputs_then_retries() {
    let mut f = Fixture::new(2);
    f.predict(5);
    f.send_snapshot(&snapshot(42, 30));
    f.pump();
    f.client
        .strategy
        .fail_authority
        .store(true, Ordering::SeqCst);
    assert!(matches!(
        f.reconcile(),
        Err(DeferredEcsError::Rollback(
            EcsRollbackError::MissingSnapshotComponents(_)
        ))
    ));
    assert_eq!(
        (f.client.acked(), f.client.predicted(), f.value()),
        (0, 1, 5)
    );
    assert_eq!(f.history().next_frame(), 1);
    assert!(f.history().contains(0));
    assert_eq!(f.driver.inputs(), &[5]);
    assert_eq!(f.driver.input_start(), 0);
    assert!(f.client.deferred_reseed_pending());
    f.client
        .strategy
        .fail_authority
        .store(false, Ordering::SeqCst);
    assert!(f.reconcile().unwrap());
    assert_eq!((f.client.acked(), f.value()), (42, 30));
    assert!(f.history().is_empty());
    assert!(f.driver.inputs().is_empty());
}

#[test]
fn missing_history_keeps_snapshot_retryable() {
    let mut f = Fixture::new(2);
    f.predict(5);
    f.send_snapshot(&snapshot(0, 30));
    f.pump();
    let history = f
        .app
        .world_mut()
        .remove_resource::<SnapshotHistory>()
        .unwrap();
    assert!(matches!(
        f.reconcile(),
        Err(DeferredEcsError::Rollback(EcsRollbackError::MissingHistory))
    ));
    assert_eq!(
        (f.client.acked(), f.client.predicted(), f.value()),
        (0, 1, 5)
    );
    assert_eq!(f.driver.inputs(), &[5]);
    assert!(f.client.deferred_reseed_pending());
    f.app.world_mut().insert_resource(history);
    assert!(f.reconcile().unwrap());
}

#[cfg(feature = "determinism")]
#[test]
fn snapshot_resets_overlapping_audit_ticks_and_preserves_following_wire_hashes() {
    use super::super::{FrameHashReceipt, FrameHashUpdate};
    use prism_ecs::world::snapshot::FrameHash;

    for baseline in [0, 1, 42] {
        for queue_limit in [1, 4] {
            let mut f = Fixture::new(queue_limit);
            f.predict(5);
            f.predict(1);
            f.client
                .audit
                .sync_history(f.app.world().resource::<SnapshotHistory>(), 0)
                .unwrap();
            let old = f.client.audit.local(1).unwrap();
            assert_eq!(f.client.audit.receive(old), Ok(FrameHashReceipt::Inserted));
            f.send_snapshot(&snapshot(baseline, 30));
            let new_remote = FrameHashUpdate {
                frame: FrameHash {
                    tick: u64::from(baseline),
                    hash: 123,
                },
                sequence: old.sequence + 1,
            };
            f.peer
                .send(
                    NetChannel::EventsReliable,
                    &super::super::encode_frame_hash(new_remote),
                )
                .unwrap();
            f.pump();
            // Neither reset nor post-snapshot admission is visible before the
            // World transaction; a nonzero future baseline may be outside the
            // old audit window altogether.
            assert_eq!(f.client.audit.remote(1), Some(old));
            assert!(f.client.take_audit_error().is_none());
            assert!(f.reconcile().unwrap());
            assert!(f.client.audit.local_log().is_empty());
            assert!(f.client.audit.remote(1).is_none());
            assert_eq!(f.client.audit.pending_len(), 0);
            // At capacity one the hash stays in the mux until the snapshot
            // leaves the bounded deferred queue; both paths must preserve it.
            f.pump();
            while f.reconcile().unwrap() {}
            assert_eq!(f.client.audit.remote(u64::from(baseline)), Some(new_remote));
            assert!(f.client.take_audit_error().is_none());
            f.predict(2);
            f.client
                .audit
                .sync_history(
                    f.app.world().resource::<SnapshotHistory>(),
                    u64::from(baseline),
                )
                .unwrap();
            assert!(f.client.audit.local(u64::from(baseline)).unwrap().sequence > old.sequence);
        }
    }
}

#[cfg(feature = "determinism")]
#[test]
fn immediate_snapshot_also_abandons_old_audit_branch() {
    let mut f = Fixture::new(4);
    f.predict(5);
    f.client
        .audit
        .sync_history(f.app.world().resource::<SnapshotHistory>(), 0)
        .unwrap();
    let old = f.client.audit.local(0).unwrap();
    f.client.audit.receive(old).unwrap();
    f.send_snapshot(&snapshot(0, 30));
    for _ in 0..12 {
        f.peer.tick();
        f.client.pump();
    }
    assert!(f.client.audit.local_log().is_empty());
    assert!(f.client.audit.remote(0).is_none());
    assert_eq!(f.client.audit.pending_len(), 0);
    assert!(!f.client.audit.comparison().retained_window_matches());
}

#[test]
fn multiple_observers_rebase_only_final_authority_and_replay_once() {
    let mut f = Fixture::new(2);
    for input in [1, 2, 3] {
        f.predict(input);
    }
    f.client.strategy.input_calls.store(0, Ordering::SeqCst);
    let mut server = server();
    let mut batch = advance(&mut server, 10);
    batch.extend(advance(&mut server, 20));
    // The current wire envelope admits one frame. Exercise the stronger
    // transaction contract directly at the deferred hand-off boundary.
    f.client
        .deferred_authority
        .push_back(DeferredAuthority::Delta(batch));
    assert!(f.reconcile().unwrap());
    assert_eq!(f.client.strategy.authority_calls.load(Ordering::SeqCst), 1);
    assert_eq!(f.client.strategy.input_calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        (f.client.acked(), f.client.predicted(), f.value()),
        (2, 3, 23)
    );
    assert_eq!(int(f.client.state().unwrap()), 23);
}

#[test]
fn later_invalid_frame_does_not_publish_earlier_observer() {
    let mut f = Fixture::new(2);
    f.predict(1);
    f.predict(2);
    let delta = advance(&mut server(), 10);
    let mut batch = delta.clone();
    batch.extend_from_slice(&delta); // stale base after the first valid update
    f.client
        .deferred_authority
        .push_back(DeferredAuthority::Delta(batch));
    let queued = f.client.deferred_authority.clone();
    assert!(matches!(f.reconcile(), Err(DeferredEcsError::Session(_))));
    assert_eq!(
        (f.client.acked(), f.client.predicted(), f.value()),
        (0, 2, 3)
    );
    assert_eq!(f.client.strategy.authority_calls.load(Ordering::SeqCst), 0);
    assert_eq!(f.client.deferred_authority, queued);
    assert_eq!(f.history().next_frame(), 2);
    assert_eq!(f.driver.inputs(), &[1, 2]);
    assert_eq!(f.client.session.buffered(), 0);
}

#[test]
fn refused_rebase_keeps_record_and_wire_message_unchanged() {
    let mut f = Fixture::new(2);
    // Authority is valid for the record, but no World input frame exists yet.
    f.send_delta(&advance(&mut server(), 10));
    f.pump();
    let queued = f.client.deferred_authority.clone();
    assert!(matches!(
        f.reconcile(),
        Err(DeferredEcsError::Rollback(
            EcsRollbackError::UnavailableFrame { .. }
        ))
    ));
    assert_eq!(
        (f.client.acked(), f.client.predicted(), f.value()),
        (0, 0, 0)
    );
    assert_eq!(f.client.deferred_authority, queued);
    assert!(f.history().is_empty());
    assert!(f.driver.inputs().is_empty());
}

#[test]
fn staged_fragment_continuation_is_not_dropped() {
    let mut f = Fixture::new(1);
    f.predict(1);
    let delta = advance(&mut server(), 10);
    let split = delta.len() / 2;
    // Wire envelopes reject fragments; this tests incremental session state
    // should a future/internal producer hand off an incomplete byte run.
    f.client
        .deferred_authority
        .push_back(DeferredAuthority::Delta(delta[..split].to_vec()));
    assert!(f.reconcile().unwrap());
    assert_eq!(f.client.acked(), 0);
    assert!(f.client.session.buffered() > 0);
    f.client
        .deferred_authority
        .push_back(DeferredAuthority::Delta(delta[split..].to_vec()));
    assert!(f.reconcile().unwrap());
    assert_eq!((f.client.acked(), f.value()), (1, 10));
    assert_eq!(f.client.session.buffered(), 0);
}

#[test]
fn malformed_snapshot_does_not_supersede_queued_work() {
    for invalid in [
        Vec::new(),
        alloc::vec![0],
        {
            let mut bytes = snapshot(42, 30);
            bytes.push(0);
            bytes
        },
        {
            let mut bytes = snapshot(42, 30);
            bytes.extend(snapshot(43, 31));
            bytes
        },
        {
            let mut bytes = Vec::new();
            encode_frame(&[255], &mut bytes).unwrap();
            bytes
        },
    ] {
        let mut f = Fixture::new(3);
        f.predict(1);
        let delta = advance(&mut server(), 10);
        f.send_delta(&delta);
        f.pump();
        let queued = f.client.deferred_authority.clone();
        f.send_snapshot(&invalid);
        // A single pump at a time: subsequent pump calls clear last_error.
        let mut rejected = false;
        for _ in 0..12 {
            f.peer.tick();
            f.client.pump_deferred();
            rejected |=
                f.client.take_last_error().is_some() || f.client.take_protocol_error().is_some();
        }
        assert!(rejected);
        assert_eq!(f.client.deferred_authority, queued);
        assert_eq!((f.client.acked(), f.value()), (0, 1));
        assert!(f.reconcile().unwrap());
        assert_eq!((f.client.acked(), f.value()), (1, 10));
    }
}

#[test]
fn valid_snapshot_supersedes_only_the_queued_prefix() {
    let mut f = Fixture::new(3);
    f.predict(1);
    f.send_delta(&advance(&mut server(), 10));
    f.send_snapshot(&snapshot(42, 30));
    let mut authority = ServerSession::new(schema(), 16, &snapshot_at(42, 30)).unwrap();
    f.send_delta(&advance(&mut authority, 40));
    f.pump();
    assert_eq!(f.client.deferred_authority.len(), 2);
    assert!(f.client.deferred_reseed_pending());
    assert_eq!(
        (f.client.acked(), f.client.predicted(), f.value()),
        (0, 1, 1)
    );
    assert!(f.reconcile().unwrap());
    assert_eq!((f.client.acked(), f.value()), (42, 30));
    assert_eq!(f.client.deferred_authority.len(), 1);
    assert!(!f.client.deferred_reseed_pending());
    f.predict(2);
    assert!(f.reconcile().unwrap());
    assert_eq!((f.client.acked(), f.value()), (43, 40));
    assert_eq!(f.client.strategy.authority_calls.load(Ordering::SeqCst), 2);
}

#[test]
fn failed_batched_rebase_restores_record_world_history_and_retries_once() {
    let mut f = Fixture::new(2);
    for input in [1, 2, 3] {
        f.predict(input);
    }
    let mut authority = server();
    let mut batch = advance(&mut authority, 10);
    batch.extend(advance(&mut authority, 20));
    f.client
        .deferred_authority
        .push_back(DeferredAuthority::Delta(batch));
    let queued = f.client.deferred_authority.clone();
    let original_hashes = (0..3)
        .map(|frame| f.history().state_hash(frame))
        .collect::<Vec<_>>();
    f.client
        .strategy
        .fail_authority
        .store(true, Ordering::SeqCst);
    assert!(matches!(
        f.reconcile(),
        Err(DeferredEcsError::Rollback(
            EcsRollbackError::MissingSnapshotComponents(_)
        ))
    ));
    assert_eq!(f.client.strategy.authority_calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        (f.client.acked(), f.client.predicted(), f.value()),
        (0, 3, 6)
    );
    assert_eq!(int(f.client.state().unwrap()), 6);
    assert_eq!(f.client.deferred_authority, queued);
    assert_eq!(f.history().next_frame(), 3);
    assert_eq!(
        (0..3)
            .map(|frame| f.history().state_hash(frame))
            .collect::<Vec<_>>(),
        original_hashes
    );
    assert_eq!(f.driver.inputs(), &[1, 2, 3]);
    f.client
        .strategy
        .fail_authority
        .store(false, Ordering::SeqCst);
    f.client.strategy.input_calls.store(0, Ordering::SeqCst);
    assert!(f.reconcile().unwrap());
    assert_eq!(f.client.strategy.authority_calls.load(Ordering::SeqCst), 2);
    assert_eq!(f.client.strategy.input_calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        (f.client.acked(), f.client.predicted(), f.value()),
        (2, 3, 23)
    );
}

#[test]
fn unavailable_anchor_at_capacity_one_can_receive_a_snapshot() {
    let mut f = Fixture::new(1);
    f.send_delta(&advance(&mut server(), 10));
    f.pump();
    assert!(matches!(
        f.reconcile(),
        Err(DeferredEcsError::Rollback(
            EcsRollbackError::UnavailableFrame { .. }
        ))
    ));
    f.send_snapshot(&snapshot(42, 30));
    f.pump();
    assert!(f.client.deferred_reseed_pending());
    assert!(f.reconcile().unwrap());
    assert_eq!((f.client.acked(), f.value()), (42, 30));
}

#[test]
fn wire_fragments_and_concatenated_deltas_are_rejected_before_staging() {
    let delta = advance(&mut server(), 10);
    let mut batch = delta.clone();
    batch.extend_from_slice(&delta);
    for malformed in [&delta[..delta.len() / 2], batch.as_slice()] {
        let mut f = Fixture::new(2);
        f.predict(1);
        f.send_delta(malformed);
        f.pump();
        assert!(matches!(
            f.client.connection_state(),
            ConnectionState::Rejected(ProtocolError::MalformedMessage {
                channel: NetChannel::EventsReliable
            })
        ));
        assert!(f.client.deferred_authority.is_empty());
        assert_eq!(
            (f.client.acked(), f.client.predicted(), f.value()),
            (0, 1, 1)
        );
    }
}

#[test]
fn full_failed_queue_can_recover_through_later_ordered_snapshot() {
    let mut f = Fixture::new(1);
    f.predict(1);
    let mut server = server();
    advance(&mut server, 10); // omit base 0, so the admitted base 1 fails
    let invalid = advance(&mut server, 20);
    f.send_delta(&invalid);
    f.pump();
    assert!(matches!(f.reconcile(), Err(DeferredEcsError::Session(_))));
    f.send_delta(&invalid);
    f.send_delta(&invalid);
    f.send_snapshot(&snapshot(42, 30));
    for _ in 0..20 {
        f.peer.tick();
        f.client.pump_deferred();
        assert!(f.client.deferred_authority.len() <= 1);
    }
    assert!(f.client.deferred_reseed_pending());
    assert_eq!((f.client.acked(), f.value()), (0, 1));
    assert!(f.reconcile().unwrap());
    assert_eq!((f.client.acked(), f.value()), (42, 30));
    assert!(f.driver.inputs().is_empty());
}

#[test]
fn application_event_staging_is_independently_bounded() {
    let mut f = Fixture::new(2);
    for byte in 0..8 {
        f.peer
            .send(NetChannel::EventsReliable, &encode_event(&[byte]))
            .unwrap();
    }
    f.pump();
    assert_eq!(f.client.inbound_events.len(), 2);
    assert!(f.client.deferred_authority.is_empty());
    let mut received = Vec::new();
    for _ in 0..30 {
        while let Some(event) = f.client.recv_event() {
            received.push(event[0]);
        }
        f.pump();
        assert!(f.client.inbound_events.len() <= 2);
    }
    assert_eq!(received, (0..8).collect::<Vec<u8>>());
}
