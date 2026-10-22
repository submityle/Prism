use super::*;
use crate::delta::{FieldKind, FieldValue};
use crate::event_log::{
    EventLogAppend, EventLogBatch, EventLogCursor, EventLogEntry, FileEventLogBackend,
    MemoryEventLogBackend,
};
use crate::plugin::{DurableLogOperation, Replication, ReplicationStrategy};
use crate::transport::{memory::InMemoryTransport, TransportSecurity};
use alloc::vec;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

const MAX_PAYLOAD: usize = 16_384;

fn schema() -> Schema {
    Schema::new(vec![FieldKind::Int { bits: 16 }])
}

fn snapshot() -> Vec<u8> {
    let mut bytes = Vec::new();
    crate::delta::encode_snapshot(
        &schema(),
        0,
        &Record::new(vec![FieldValue::Int(0)]),
        &mut bytes,
    )
    .unwrap();
    bytes
}

fn value(record: &Record) -> i64 {
    match record.value(0).unwrap() {
        FieldValue::Int(value) => value,
        _ => panic!("expected integer"),
    }
}

struct Add(Arc<AtomicUsize>);

impl ServerStrategy for Add {
    type Command = i64;
    fn decode(&self, bytes: &[u8]) -> Option<i64> {
        Some(i64::from_le_bytes(bytes.try_into().ok()?))
    }
    fn apply(&self, record: &mut Record, command: i64) {
        self.0.fetch_add(1, Ordering::Relaxed);
        record.set(0, FieldValue::Int(value(record) + command));
    }
}

struct Predict;
impl ReplicationStrategy for Predict {
    type Input = i64;
    fn simulate(&self, record: &mut Record, input: &i64) {
        record.set(0, FieldValue::Int(value(record) + input));
    }
    fn encode(&self, input: &i64) -> Vec<u8> {
        input.to_le_bytes().to_vec()
    }
}

struct FaultState {
    log: MemoryEventLogBackend,
    fail_append_before: usize,
    fail_append_after: usize,
    fail_replay: usize,
    fail_sync: usize,
    appends: usize,
    syncs: usize,
    overhead_after_sync: Option<Arc<AtomicUsize>>,
    checkpoint_enabled: bool,
    fail_checkpoint_before: usize,
    fail_checkpoint_after: usize,
    checkpoints: Vec<EventLogEntry>,
}

#[derive(Clone)]
struct FaultLog(Arc<Mutex<FaultState>>);

impl FaultLog {
    fn new(capacity: usize) -> Self {
        Self(Arc::new(Mutex::new(FaultState {
            log: MemoryEventLogBackend::new(capacity, MAX_PAYLOAD),
            fail_append_before: 0,
            fail_append_after: 0,
            fail_replay: 0,
            fail_sync: 0,
            appends: 0,
            syncs: 0,
            overhead_after_sync: None,
            checkpoint_enabled: false,
            fail_checkpoint_before: 0,
            fail_checkpoint_after: 0,
            checkpoints: Vec::new(),
        })))
    }
    fn replay_log(&self) -> ReplayEventLog<Self> {
        ReplayEventLog::new(self.clone(), MAX_PAYLOAD)
    }
    fn len(&self) -> usize {
        self.0.lock().unwrap().log.len()
    }
}

impl EventLogBackend for FaultLog {
    type Error = &'static str;
    fn append(&mut self, key: u128, bytes: &[u8]) -> Result<EventLogAppend, Self::Error> {
        let mut state = self.0.lock().unwrap();
        state.appends += 1;
        if state.fail_append_before > 0 {
            state.fail_append_before -= 1;
            return Err("before append");
        }
        let outcome = state.log.append(key, bytes).map_err(|_| "append bounds")?;
        if state.fail_append_after > 0 {
            state.fail_append_after -= 1;
            return Err("ambiguous append");
        }
        Ok(outcome)
    }
    fn cursor(&self) -> EventLogCursor {
        self.0.lock().unwrap().log.cursor()
    }
    fn start_cursor(&self) -> EventLogCursor {
        self.0.lock().unwrap().log.start_cursor()
    }
    fn remaining_capacity(&self) -> Option<usize> {
        let state = self.0.lock().unwrap();
        state
            .checkpoint_enabled
            .then(|| state.log.remaining_capacity().unwrap())
    }
    fn checkpoint(&mut self, entry: &EventLogEntry) -> Option<Result<(), Self::Error>> {
        let mut state = self.0.lock().unwrap();
        if !state.checkpoint_enabled {
            return None;
        }
        state.checkpoints.push(entry.clone());
        if state.fail_checkpoint_before > 0 {
            state.fail_checkpoint_before -= 1;
            return Some(Err("before checkpoint"));
        }
        if state.log.checkpoint(entry).unwrap().is_err() {
            return Some(Err("checkpoint bounds"));
        }
        if state.fail_checkpoint_after > 0 {
            state.fail_checkpoint_after -= 1;
            return Some(Err("ambiguous checkpoint"));
        }
        Some(Ok(()))
    }
    fn replay(&self, cursor: EventLogCursor, limit: usize) -> Result<EventLogBatch, Self::Error> {
        let mut state = self.0.lock().unwrap();
        if state.fail_replay > 0 {
            state.fail_replay -= 1;
            return Err("replay read");
        }
        state.log.replay(cursor, limit).map_err(|_| "replay bounds")
    }
    fn flush(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }
    fn sync(&mut self) -> Result<(), Self::Error> {
        let mut state = self.0.lock().unwrap();
        state.syncs += 1;
        if state.fail_sync > 0 {
            state.fail_sync -= 1;
            return Err("sync barrier");
        }
        if let Some(overhead) = state.overhead_after_sync.take() {
            overhead.store(100_000, Ordering::Relaxed);
        }
        Ok(())
    }
}

struct AdjustableTransport {
    inner: Box<dyn Transport>,
    overhead: Arc<AtomicUsize>,
}
impl Transport for AdjustableTransport {
    fn send(&mut self, bytes: &[u8]) {
        self.inner.send(bytes);
    }
    fn recv(&mut self) -> Option<Vec<u8>> {
        self.inner.recv()
    }
    fn supports_secure_session(&self) -> bool {
        self.inner.supports_secure_session()
    }
    fn security_state(&self) -> TransportSecurity {
        self.inner.security_state()
    }
    fn security_overhead(&self) -> usize {
        self.inner
            .security_overhead()
            .saturating_add(self.overhead.load(Ordering::Relaxed))
    }
}

type Peer = ChannelMux<Box<dyn Transport>>;

fn connected(
    log: &FaultLog,
) -> (
    Peer,
    ServerReplication<Add>,
    Arc<AtomicUsize>,
    Arc<AtomicUsize>,
) {
    connected_backend(log.replay_log(), Arc::new(AtomicUsize::new(0)))
}

fn connected_backend<B>(
    log: ReplayEventLog<B>,
    overhead: Arc<AtomicUsize>,
) -> (
    Peer,
    ServerReplication<Add>,
    Arc<AtomicUsize>,
    Arc<AtomicUsize>,
)
where
    B: EventLogBackend + Send + Sync + 'static,
    B::Error: core::fmt::Debug,
{
    connected_seed(log, overhead, &snapshot())
}

fn connected_seed<B>(
    log: ReplayEventLog<B>,
    overhead: Arc<AtomicUsize>,
    seed: &[u8],
) -> (
    Peer,
    ServerReplication<Add>,
    Arc<AtomicUsize>,
    Arc<AtomicUsize>,
)
where
    B: EventLogBackend + Send + Sync + 'static,
    B::Error: core::fmt::Debug,
{
    let (left, right) = InMemoryTransport::pair_secure();
    let config = ReplicationConfig::for_capacity(8);
    let mut peer = ChannelMux::new(Box::new(left) as Box<dyn Transport>, config.mux);
    let applies = Arc::new(AtomicUsize::new(0));
    let mut server = ServerReplication::new_durable(
        schema(),
        8,
        seed,
        Add(applies.clone()),
        Box::new(AdjustableTransport {
            inner: Box::new(right),
            overhead: overhead.clone(),
        }),
        log,
    )
    .unwrap();
    peer.send(
        NetChannel::RpcReliable,
        &super::super::encode_hello(config.protocol),
    )
    .unwrap();
    for _ in 0..8 {
        peer.tick();
        server.pump();
    }
    assert!(server.negotiated().is_some());
    (peer, server, applies, overhead)
}

fn command(peer: &mut Peer, amount: i64) {
    peer.send(NetChannel::RpcReliable, &encode_rpc(&amount.to_le_bytes()))
        .unwrap();
    peer.tick();
}

#[test]
fn append_replay_and_sync_failures_retry_the_exact_candidate_once() {
    for failure in 0..4 {
        let log = FaultLog::new(32);
        let (mut peer, mut server, applies, _) = connected(&log);
        {
            let mut state = log.0.lock().unwrap();
            match failure {
                0 => state.fail_append_before = 1,
                1 => state.fail_append_after = 1,
                2 => state.fail_replay = 1,
                _ => state.fail_sync = 1,
            }
        }
        command(&mut peer, 7);
        assert_eq!(server.pump(), 0);
        assert_eq!(server.revision(), Some(0));
        assert_eq!(value(server.record().unwrap()), 0);
        assert_eq!(server.pending_commands.len(), 1);
        assert_eq!(server.unacked(), 0, "no delta may be queued before sync");
        peer.tick();
        assert!(peer.recv(NetChannel::EventsReliable).is_none());
        assert_eq!(server.journal_cursor(), Some(EventLogCursor::new(1)));
        let candidate = server.prepared.as_ref().unwrap().record.clone();
        let expected_operation = match failure {
            0 | 1 => DurableLogOperation::Append,
            2 => DurableLogOperation::Replay,
            _ => DurableLogOperation::Sync,
        };
        assert!(
            matches!(server.take_durable_error(), Some(DurableRecoveryError::Log {
            operation, ..
        }) if operation == expected_operation)
        );
        assert_eq!(server.pump(), 1);
        assert_eq!(server.revision(), Some(1));
        assert_eq!(value(server.record().unwrap()), 7);
        assert_eq!(applies.load(Ordering::Relaxed), 1);
        assert_eq!(server.applied(), 1);
        assert!(server.pending_commands.is_empty());
        assert_eq!(log.len(), 2);
        assert_eq!(
            log.replay_log()
                .replay(EventLogCursor::new(1), 1)
                .unwrap()
                .records()[0]
                .record,
            candidate
        );
        assert_eq!(server.journal_cursor(), Some(EventLogCursor::new(2)));
        for _ in 0..4 {
            peer.tick();
            server.pump();
        }
        assert_eq!(server.applied(), 1);
    }
}

#[test]
fn durable_authority_refuses_external_command_handoff() {
    let log = FaultLog::new(32);
    let (_peer, mut server, _, _) = connected(&log);

    assert_eq!(
        server.poll_inbound_commands(),
        Err(ExternalCommandModeError::DurableJournalInstalled)
    );
    assert!(!server.external_command_consumer);
}

#[test]
fn ambiguous_io_then_mtu_refusal_never_discards_the_candidate() {
    for failure in 0..4 {
        let log = FaultLog::new(32);
        let (mut peer, mut server, applies, overhead) = connected(&log);
        {
            let mut state = log.0.lock().unwrap();
            match failure {
                0 => state.fail_append_before = 1,
                1 => state.fail_append_after = 1,
                2 => state.fail_replay = 1,
                _ => state.fail_sync = 1,
            }
        }
        command(&mut peer, 7);
        assert_eq!(server.pump(), 0);
        let candidate = server.prepared.as_ref().unwrap().record.clone();
        overhead.store(100_000, Ordering::Relaxed);
        for _ in 0..3 {
            assert_eq!(server.pump(), 0);
            assert_eq!(server.revision(), Some(0));
            assert_eq!(server.pending_commands.len(), 1);
            assert_eq!(server.rejected(), 0);
            assert_eq!(server.prepared.as_ref().unwrap().record, candidate);
            assert_eq!(server.unacked(), 0);
        }
        overhead.store(0, Ordering::Relaxed);
        assert_eq!(server.pump(), 1);
        assert_eq!(value(server.record().unwrap()), 7);
        assert_eq!(applies.load(Ordering::Relaxed), 1);
        assert_eq!(log.len(), 2);
        assert_eq!(
            log.replay_log()
                .replay(EventLogCursor::new(1), 1)
                .unwrap()
                .records()[0]
                .record,
            candidate
        );
    }
}

#[test]
fn synced_candidate_survives_send_refusal_and_in_memory_restart() {
    let log = FaultLog::new(32);
    let (mut peer, mut server, applies, overhead) = connected(&log);
    log.0.lock().unwrap().overhead_after_sync = Some(overhead.clone());
    command(&mut peer, 9);
    assert_eq!(server.pump(), 0);
    assert!(matches!(
        server.take_send_error(),
        Some(MuxSendError::TooLarge { .. })
    ));
    assert!(server.prepared.as_ref().unwrap().synced);
    assert_eq!(server.revision(), Some(0));
    assert_eq!(log.len(), 2);
    let (appends, syncs) = {
        let state = log.0.lock().unwrap();
        (state.appends, state.syncs)
    };
    overhead.store(0, Ordering::Relaxed);
    assert_eq!(server.pump(), 1);
    assert_eq!(applies.load(Ordering::Relaxed), 1);
    {
        let state = log.0.lock().unwrap();
        assert_eq!((state.appends, state.syncs), (appends, syncs));
    }
    // A second transaction reaches sync but never reaches enqueue. Dropping the
    // server simulates loss of the in-flight candidate and transport state; the
    // separate SIGKILL matrix below provides actual process-crash evidence.
    for _ in 0..4 {
        peer.tick();
        server.pump();
    }
    log.0.lock().unwrap().overhead_after_sync = Some(overhead);
    command(&mut peer, 4);
    assert_eq!(server.pump(), 0);
    assert_eq!(server.revision(), Some(1));
    drop(server);
    let (_, recovered, recovered_applies, _) = connected(&log);
    assert_eq!(recovered.revision(), Some(2));
    assert_eq!(value(recovered.record().unwrap()), 13);
    assert_eq!(
        recovered_applies.load(Ordering::Relaxed),
        0,
        "recovery applies deltas, not commands"
    );
    assert_eq!(log.len(), 3);
}

#[test]
fn journal_capacity_backpressures_without_advancing_or_losing_input() {
    let log = FaultLog::new(1); // only the seed fits
    let (mut peer, mut server, applies, _) = connected(&log);
    command(&mut peer, 3);
    for _ in 0..3 {
        assert_eq!(server.pump(), 0);
        assert_eq!(server.revision(), Some(0));
        assert!(matches!(
            server.take_durable_error(),
            Some(DurableRecoveryError::Log { .. })
        ));
    }
    assert_eq!(server.pending_commands.len(), 1);
    assert_eq!(applies.load(Ordering::Relaxed), 1);
    assert_eq!(log.len(), 1);
}

#[test]
fn recovery_rejects_changed_schema_seed_and_conflicting_keys() {
    let log = FaultLog::new(32);
    let (mut peer, mut server, _, _) = connected(&log);
    command(&mut peer, 1);
    assert_eq!(server.pump(), 1);
    drop(server);
    let mut other_seed = Vec::new();
    crate::delta::encode_snapshot(
        &schema(),
        0,
        &Record::new(vec![FieldValue::Int(99)]),
        &mut other_seed,
    )
    .unwrap();
    for (schema, snapshot) in [
        (schema(), other_seed),
        (Schema::new(vec![FieldKind::Int { bits: 15 }]), snapshot()),
    ] {
        assert!(durable::restore(&schema, 8, &snapshot, log.replay_log()).is_err());
    }
    let mut foreign = crate::replay::ReplayRecord::new(
        2,
        2,
        NetChannel::EventsReliable,
        crate::replay::ReplayDirection::ServerToClient,
        None,
        vec![255],
    );
    // The provider's key-only deduplication must not hide a foreign record.
    foreign.sequence = 7;
    log.replay_log().append(2, foreign).unwrap();
    assert!(durable::restore(&schema(), 8, &snapshot(), log.replay_log()).is_err());
}

#[test]
fn durable_plugin_updates_authority_and_reconciles_via_app_update() {
    let log = FaultLog::new(32);
    let (left, right) = InMemoryTransport::pair_secure();
    let mut client = App::new();
    client.add_plugins(crate::plugin::ReplicationPlugin::new(
        schema(),
        8,
        snapshot(),
        Predict,
        Box::new(left),
    ));
    let mut authority = App::new();
    authority.add_plugins(
        ReplicationServerPlugin::new(
            schema(),
            8,
            snapshot(),
            Add(Arc::new(AtomicUsize::new(0))),
            Box::new(right),
        )
        .with_event_log(log.replay_log())
        .unwrap(),
    );
    for _ in 0..8 {
        client.update();
        authority.update();
    }
    client
        .world_mut()
        .resource_mut::<Replication<Predict>>()
        .predict_and_send(11)
        .unwrap();
    for _ in 0..12 {
        client.update();
        authority.update();
    }
    assert_eq!(
        authority
            .world()
            .resource::<ServerReplication<Add>>()
            .revision(),
        Some(1)
    );
    assert_eq!(
        value(
            client
                .world()
                .resource::<Replication<Predict>>()
                .state()
                .unwrap()
        ),
        11
    );
    assert_eq!(log.len(), 2);
    drop(authority);
    let (_, recovered, _, _) = connected(&log);
    assert_eq!(recovered.revision(), Some(1));
    assert_eq!(value(recovered.record().unwrap()), 11);
}

#[test]
fn checkpoint_failures_retry_exact_bytes_without_reapplying_or_publishing() {
    for ambiguous in [false, true] {
        let log = FaultLog::new(2);
        log.0.lock().unwrap().checkpoint_enabled = true;
        let (mut peer, mut server, applies, _) = connected(&log);
        command(&mut peer, 7);
        assert_eq!(server.pump(), 1);
        for _ in 0..8 {
            peer.tick();
            server.pump();
        }
        {
            let mut state = log.0.lock().unwrap();
            if ambiguous {
                state.fail_checkpoint_after = 1;
            } else {
                state.fail_checkpoint_before = 1;
            }
        }
        command(&mut peer, 4);
        assert_eq!(server.pump(), 0);
        assert_eq!(server.revision(), Some(1));
        assert_eq!(value(server.record().unwrap()), 7);
        assert_eq!(server.pending_commands.len(), 1);
        assert_eq!(server.unacked(), 0);
        assert!(matches!(
            server.take_durable_error(),
            Some(DurableRecoveryError::Log {
                operation: DurableLogOperation::Checkpoint,
                ..
            })
        ));
        let candidate = server.prepared.as_ref().unwrap().record.clone();
        assert_eq!(server.pump(), 1);
        assert_eq!(value(server.record().unwrap()), 11);
        assert_eq!(applies.load(Ordering::Relaxed), 2);
        let state = log.0.lock().unwrap();
        assert_eq!(state.checkpoints.len(), 2);
        assert_eq!(state.checkpoints[0], state.checkpoints[1]);
        assert_eq!(state.log.start_cursor(), EventLogCursor::new(1));
        drop(state);
        assert_eq!(
            log.replay_log()
                .replay(EventLogCursor::new(2), 1)
                .unwrap()
                .records()[0]
                .record,
            candidate
        );
        drop(server);
        let (_, recovered, recovered_applies, _) = connected(&log);
        assert_eq!(value(recovered.record().unwrap()), 11);
        assert_eq!(recovered.revision(), Some(2));
        assert_eq!(recovered_applies.load(Ordering::Relaxed), 0);
    }
}

#[test]
fn oversized_checkpoint_backpressures_exact_candidate_without_mutating_memory_log() {
    let log = FaultLog::new(2);
    let (mut peer, mut server, applies, _) = connected(&log);
    command(&mut peer, 1);
    assert_eq!(server.pump(), 1);
    for _ in 0..8 {
        peer.tick();
        server.pump();
    }
    let before = {
        let mut state = log.0.lock().unwrap();
        let batch = state.log.replay(EventLogCursor::new(0), 2).unwrap();
        let max = batch
            .entries()
            .iter()
            .map(|entry| entry.payload.len())
            .max()
            .unwrap();
        let mut bounded = MemoryEventLogBackend::new(2, max);
        for entry in batch.entries() {
            bounded
                .append(entry.idempotency_key, &entry.payload)
                .unwrap();
        }
        state.log = bounded;
        state.checkpoint_enabled = true;
        state.log.clone()
    };
    command(&mut peer, 2);
    for _ in 0..3 {
        assert_eq!(server.pump(), 0);
        assert_eq!(server.revision(), Some(1));
        assert_eq!(server.pending_commands.len(), 1);
        assert_eq!(server.unacked(), 0);
        assert_eq!(log.0.lock().unwrap().log, before);
    }
    let state = log.0.lock().unwrap();
    assert!(state.checkpoints[0].payload.len() > before.max_payload());
    assert!(state.checkpoints.windows(2).all(|pair| pair[0] == pair[1]));
    assert_eq!(applies.load(Ordering::Relaxed), 2);
}

#[test]
fn rotated_checkpoint_rejects_wrong_schema_seed_revision_and_noncanonical_payload() {
    let log = FaultLog::new(2);
    log.0.lock().unwrap().checkpoint_enabled = true;
    let (mut peer, mut server, _, _) = connected(&log);
    for _ in 0..4 {
        command(&mut peer, 1);
        assert_eq!(server.pump(), 1);
        for _ in 0..8 {
            peer.tick();
            server.pump();
        }
    }
    drop(server);
    let start = log.start_cursor();
    assert_eq!(start, EventLogCursor::new(3));
    assert!(durable::restore(
        &Schema::new(vec![FieldKind::Int { bits: 15 }]),
        8,
        &snapshot(),
        log.replay_log()
    )
    .is_err());
    let mut changed_seed = Vec::new();
    crate::delta::encode_snapshot(
        &schema(),
        0,
        &Record::new(vec![FieldValue::Int(99)]),
        &mut changed_seed,
    )
    .unwrap();
    assert!(durable::restore(&schema(), 8, &changed_seed, log.replay_log()).is_err());
    let original = log.replay_log().replay(start, 1).unwrap().records()[0]
        .record
        .clone();
    for mutation in 0..6 {
        let mut invalid = original.clone();
        match mutation {
            0 => invalid.tick += 1,
            1 => invalid.payload.push(0),
            2 => {
                invalid.payload.truncate(12);
            }
            3 => invalid.payload[4] ^= 1,
            4 => invalid.state_hash = Some(0),
            _ => invalid.sequence += 1,
        }
        let mut broken =
            ReplayEventLog::new(MemoryEventLogBackend::new(8, MAX_PAYLOAD), MAX_PAYLOAD);
        // Give the provider the right absolute position, then replace its prefix
        // with a semantically malformed checkpoint, not a corrupt outer codec.
        for sequence in 0..=original.sequence {
            let mut filler = original.clone();
            filler.sequence = sequence;
            broken.append(u128::from(sequence), filler).unwrap();
        }
        if mutation == 5 {
            // Keep the provider sequence correct while corrupting the inner one.
            let mut encoded = crate::replay::ReplayLog::new(1, MAX_PAYLOAD);
            encoded.append(invalid).unwrap();
            broken
                .backend_mut()
                .checkpoint(&EventLogEntry {
                    sequence: original.sequence,
                    idempotency_key: u128::from(original.sequence),
                    payload: encoded.encode().unwrap(),
                })
                .unwrap()
                .unwrap();
        } else {
            broken
                .checkpoint(u128::from(original.sequence), invalid)
                .unwrap()
                .unwrap();
        }
        assert!(
            durable::restore(&schema(), 8, &snapshot(), broken).is_err(),
            "mutation {mutation}"
        );
    }
    log.0.lock().unwrap().fail_sync = 1;
    assert!(matches!(
        durable::restore(&schema(), 8, &snapshot(), log.replay_log()),
        Err(DurableRecoveryError::Log {
            operation: DurableLogOperation::Sync,
            ..
        })
    ));
    assert_eq!(
        durable::restore(&schema(), 8, &snapshot(), log.replay_log())
            .unwrap()
            .session
            .revision(),
        Some(4)
    );
}

/// Real file provider with one-shot ambiguous I/O results. All successful file
/// operations are real; failures are injected only at their explicit boundaries.
struct FileFaultLog {
    inner: FileEventLogBackend,
    fault: Arc<AtomicUsize>,
    overhead: Arc<AtomicUsize>,
}

impl FileFaultLog {
    fn fail(&self, mode: usize) -> bool {
        self.fault
            .compare_exchange(mode, 0, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
    }
}

impl EventLogBackend for FileFaultLog {
    type Error = std::string::String;
    fn append(&mut self, key: u128, payload: &[u8]) -> Result<EventLogAppend, Self::Error> {
        if self.fail(1) {
            return Err("before append".into());
        }
        let result = self
            .inner
            .append(key, payload)
            .map_err(|error| std::format!("{error:?}"))?;
        if self.fail(2) {
            return Err("after append".into());
        }
        Ok(result)
    }
    fn cursor(&self) -> EventLogCursor {
        self.inner.cursor()
    }
    fn start_cursor(&self) -> EventLogCursor {
        self.inner.start_cursor()
    }
    fn remaining_capacity(&self) -> Option<usize> {
        self.inner.remaining_capacity()
    }
    fn checkpoint(&mut self, entry: &EventLogEntry) -> Option<Result<(), Self::Error>> {
        self.inner
            .checkpoint(entry)
            .map(|result| result.map_err(|error| std::format!("{error:?}")))
    }
    fn replay(&self, cursor: EventLogCursor, limit: usize) -> Result<EventLogBatch, Self::Error> {
        self.inner
            .replay(cursor, limit)
            .map_err(|error| std::format!("{error:?}"))
    }
    fn flush(&mut self) -> Result<(), Self::Error> {
        self.inner
            .flush()
            .map_err(|error| std::format!("{error:?}"))
    }
    fn sync(&mut self) -> Result<(), Self::Error> {
        if self.fail(3) {
            return Err("before sync".into());
        }
        self.inner
            .sync()
            .map_err(|error| std::format!("{error:?}"))?;
        if self.fail(4) {
            return Err("after sync".into());
        }
        if self.fail(5) {
            self.overhead.store(100_000, Ordering::Relaxed);
        }
        Ok(())
    }
}

/// Parent-only fixture. Children are deliberately killed before running Drop.
struct ProcessDirectory(std::path::PathBuf);
impl ProcessDirectory {
    fn new() -> Self {
        let path = std::env::temp_dir().join(std::format!(
            "prism-durable-process-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn journal(&self) -> std::path::PathBuf {
        self.0.join("authority")
    }
}
impl Drop for ProcessDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn file_ambiguous_io_retries_exact_candidate_without_duplicate_append_or_publication() {
    for mode in 1..=5 {
        let directory = ProcessDirectory::new();
        let path = directory.journal();
        let fault = Arc::new(AtomicUsize::new(0));
        let overhead = Arc::new(AtomicUsize::new(0));
        let log = FileFaultLog {
            inner: FileEventLogBackend::open(&path, 2, MAX_PAYLOAD).unwrap(),
            fault: fault.clone(),
            overhead: overhead.clone(),
        };
        let (mut peer, mut server, applies, _) =
            connected_backend(ReplayEventLog::new(log, MAX_PAYLOAD), overhead.clone());
        // Fill, reclaim, then inject at a nonzero sequence (not only the seed).
        for _ in 0..4 {
            command(&mut peer, 1);
            assert_eq!(server.pump(), 1);
            for _ in 0..8 {
                peer.tick();
                server.pump();
            }
        }
        fault.store(mode, Ordering::Relaxed);
        command(&mut peer, 9);
        assert_eq!(server.pump(), 0);
        assert_eq!(server.revision(), Some(4));
        assert_eq!(value(server.record().unwrap()), 4);
        assert_eq!(server.unacked(), 0);
        assert_eq!(server.pending_commands.len(), 1);
        let candidate = server.prepared.as_ref().unwrap().record.clone();
        assert_eq!(candidate.sequence, 5);
        let written = std::fs::read(&path).unwrap();
        // After any attempted journal write, an MTU refusal must retain the
        // exact candidate, including failures before the first byte was written.
        overhead.store(100_000, Ordering::Relaxed);
        for _ in 0..3 {
            assert_eq!(server.pump(), 0);
            assert_eq!(server.prepared.as_ref().unwrap().record, candidate);
            assert_eq!(std::fs::read(&path).unwrap(), written);
            assert_eq!(server.revision(), Some(4));
            assert_eq!(server.unacked(), 0);
        }
        overhead.store(0, Ordering::Relaxed);
        assert_eq!(server.pump(), 1);
        assert_eq!(server.revision(), Some(5));
        assert_eq!(value(server.record().unwrap()), 13);
        assert_eq!(applies.load(Ordering::Relaxed), 5);
        if mode != 1 {
            assert_eq!(std::fs::read(&path).unwrap(), written);
        }
        drop(server);
        let mut log = ReplayEventLog::new(
            FileEventLogBackend::open(&path, 2, MAX_PAYLOAD).unwrap(),
            MAX_PAYLOAD,
        );
        assert_eq!(log.backend().cursor(), EventLogCursor::new(6));
        assert_eq!(log.backend().start_cursor(), EventLogCursor::new(4));
        assert_eq!(
            log.replay(EventLogCursor::new(5), 1).unwrap().records()[0].record,
            candidate
        );
        assert!(!log.append(5, candidate).unwrap().inserted);
        let (_, restored, recovered_applies, _) =
            connected_backend(log, Arc::new(AtomicUsize::new(0)));
        assert_eq!(restored.revision(), Some(5));
        assert_eq!(value(restored.record().unwrap()), 13);
        assert_eq!(recovered_applies.load(Ordering::Relaxed), 0);
    }
}

#[test]
fn file_nonzero_seed_revision_rotates_and_restarts_without_sequence_aliasing() {
    let directory = ProcessDirectory::new();
    let path = directory.journal();
    let mut seed = Vec::new();
    crate::delta::encode_snapshot(
        &schema(),
        41,
        &Record::new(vec![FieldValue::Int(300)]),
        &mut seed,
    )
    .unwrap();
    for round in 0..3 {
        let fault = Arc::new(AtomicUsize::new(0));
        let overhead = Arc::new(AtomicUsize::new(0));
        let log = FileFaultLog {
            inner: FileEventLogBackend::open(&path, 2, MAX_PAYLOAD).unwrap(),
            fault: fault.clone(),
            overhead: overhead.clone(),
        };
        let (mut peer, mut server, applies, _) =
            connected_seed(ReplayEventLog::new(log, MAX_PAYLOAD), overhead, &seed);
        assert_eq!(server.revision(), Some(41 + round * 6));
        assert_eq!(applies.load(Ordering::Relaxed), 0);
        for offset in 0..6 {
            if offset == 5 {
                fault.store(4, Ordering::Relaxed);
            }
            command(&mut peer, 1);
            if offset == 5 {
                assert_eq!(server.pump(), 0);
                assert_eq!(server.revision(), Some(41 + round * 6 + offset));
                assert_eq!(server.pending_commands.len(), 1);
            }
            assert_eq!(server.pump(), 1);
            for _ in 0..8 {
                peer.tick();
                server.pump();
            }
        }
        assert_eq!(applies.load(Ordering::Relaxed), 6);
        assert_eq!(server.revision(), Some(47 + round * 6));
        assert_eq!(
            server.journal_cursor(),
            Some(EventLogCursor::new(u64::from(7 + round * 6)))
        );
    }
    let file = FileEventLogBackend::open(&path, 2, MAX_PAYLOAD).unwrap();
    assert_eq!(file.start_cursor(), EventLogCursor::new(17));
    assert_eq!(file.cursor(), EventLogCursor::new(19));
    let restored =
        durable::restore(&schema(), 8, &seed, ReplayEventLog::new(file, MAX_PAYLOAD)).unwrap();
    assert_eq!(restored.session.revision(), Some(59));
    assert_eq!(value(restored.session.record().unwrap()), 318);
    assert_eq!(restored.next_sequence, 19);
}

#[test]
fn file_single_slot_capacity_keeps_seed_and_backpressures_without_false_reclamation() {
    let directory = ProcessDirectory::new();
    let path = directory.journal();
    let file = FileEventLogBackend::open(&path, 1, MAX_PAYLOAD).unwrap();
    let (mut peer, mut server, applies, _) = connected_backend(
        ReplayEventLog::new(file, MAX_PAYLOAD),
        Arc::new(AtomicUsize::new(0)),
    );
    let bytes = std::fs::read(&path).unwrap();
    command(&mut peer, 5);
    for _ in 0..3 {
        assert_eq!(server.pump(), 0);
        assert_eq!(server.revision(), Some(0));
        assert_eq!(server.pending_commands.len(), 1);
        assert_eq!(server.unacked(), 0);
        assert!(matches!(
            server.take_durable_error(),
            Some(DurableRecoveryError::Log {
                operation: DurableLogOperation::Append,
                ..
            })
        ));
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
    }
    assert_eq!(applies.load(Ordering::Relaxed), 1);
    drop(server);
    // Increasing capacity is an explicit operator action, not silent data loss.
    let file = FileEventLogBackend::open(&path, 2, MAX_PAYLOAD).unwrap();
    let (mut peer, mut server, _, _) = connected_backend(
        ReplayEventLog::new(file, MAX_PAYLOAD),
        Arc::new(AtomicUsize::new(0)),
    );
    assert_eq!(server.revision(), Some(0));
    command(&mut peer, 5);
    assert_eq!(server.pump(), 1);
    assert_eq!(server.revision(), Some(1));
}

fn kill_at_barrier(path: &std::path::Path, phase: &str) {
    use std::io::{BufRead, BufReader, Read};
    use std::process::{Command, Stdio};
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "plugin::server::durable_tests::durable_process_child",
            "--nocapture",
        ])
        .env("PRISM_DURABLE_CHILD_PATH", path)
        .env("PRISM_DURABLE_CHILD_PHASE", phase)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let stdout = child.stdout.take().unwrap();
    let (sender, receiver) = std::sync::mpsc::channel();
    let reader = std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines() {
            if line.unwrap() == "PRISM_DURABLE_CRASH_READY" {
                let _ = sender.send(());
                break;
            }
        }
    });
    let ready = receiver.recv_timeout(std::time::Duration::from_secs(20));
    // Kill on both success and timeout: never leak a paused subprocess.
    let _ = child.kill();
    let status = child.wait().unwrap();
    reader.join().unwrap();
    let mut stderr = std::string::String::new();
    child
        .stderr
        .take()
        .unwrap()
        .read_to_string(&mut stderr)
        .unwrap();
    assert!(
        ready.is_ok(),
        "child failed to reach {phase}: {status}, {stderr}"
    );
    assert!(!status.success());
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        assert_eq!(
            status.signal(),
            Some(9),
            "parent must SIGKILL, not orderly exit"
        );
    }
}

#[test]
fn durable_process_child() {
    let Some(path) = std::env::var_os("PRISM_DURABLE_CHILD_PATH") else {
        return;
    };
    let phase = std::env::var("PRISM_DURABLE_CHILD_PHASE").unwrap();
    let file = FileEventLogBackend::open(path, 2, MAX_PAYLOAD).unwrap();
    let fault = Arc::new(AtomicUsize::new(0));
    let overhead = Arc::new(AtomicUsize::new(0));
    let log = FileFaultLog {
        inner: file,
        fault: fault.clone(),
        overhead: overhead.clone(),
    };
    let (mut peer, mut server, applies, _) =
        connected_backend(ReplayEventLog::new(log, MAX_PAYLOAD), overhead);
    if phase.starts_with("checkpoint-") {
        command(&mut peer, 7);
        assert_eq!(server.pump(), 1);
        for _ in 0..8 {
            peer.tick();
            server.pump();
        }
        assert_eq!(server.unacked(), 0);
        let point = match phase.as_str() {
            "checkpoint-written" => "checkpoint-written",
            "checkpoint-synced" => "checkpoint-synced",
            "checkpoint-renamed" => "checkpoint-renamed",
            "checkpoint-published" => "checkpoint-published",
            _ => panic!("unknown checkpoint phase"),
        };
        crate::event_log::CHECKPOINT_CRASH.with(|slot| slot.set(Some(point)));
        command(&mut peer, 4);
        server.pump();
        panic!("checkpoint crash point not reached");
    }
    if phase == "resume-rotations" {
        let previous = server.revision().unwrap();
        for _ in 0..6 {
            command(&mut peer, 1);
            assert_eq!(server.pump(), 1);
            for _ in 0..8 {
                peer.tick();
                server.pump();
            }
        }
        assert_eq!(server.revision(), Some(previous + 6));
        crate::event_log::process_crash_barrier();
    }
    let rotated = phase == "rotated-sync";
    if rotated {
        for _ in 0..6 {
            command(&mut peer, 1);
            assert_eq!(server.pump(), 1);
            for _ in 0..8 {
                peer.tick();
                server.pump();
            }
        }
    }
    let mode = match phase.as_str() {
        "before-append" => 1,
        "after-append" => 2,
        "before-sync" => 3,
        "after-sync" | "rotated-sync" => 4,
        "before-publication" => 5,
        "after-publication" => 0,
        _ => panic!("unknown child phase"),
    };
    fault.store(mode, Ordering::Relaxed);
    let baseline = server.revision().unwrap();
    command(&mut peer, if rotated { 1 } else { 9 });
    let advanced = server.pump();
    if phase == "after-publication" {
        assert_eq!(advanced, 1);
        assert_eq!(server.revision(), Some(baseline + 1));
        assert!(server.pending_commands.is_empty());
    } else {
        assert_eq!(advanced, 0);
        assert_eq!(server.revision(), Some(baseline));
        assert_eq!(server.pending_commands.len(), 1);
        assert_eq!(
            server.unacked(),
            0,
            "no enqueue before the successful barrier"
        );
        assert_eq!(server.prepared.as_ref().unwrap().synced, mode == 5);
        assert_eq!(applies.load(Ordering::Relaxed), baseline as usize + 1);
        peer.tick();
        // Prior transactions may remain in the peer's receive queue after rotation.
        if !rotated {
            assert!(peer.recv(NetChannel::EventsReliable).is_none());
        }
    }
    crate::event_log::process_crash_barrier();
}

#[test]
fn file_process_sigkill_recovers_append_sync_publication_and_checkpoint_boundaries() {
    for phase in [
        "before-append",
        "after-append",
        "before-sync",
        "after-sync",
        "before-publication",
        "after-publication",
        "rotated-sync",
        "checkpoint-written",
        "checkpoint-synced",
        "checkpoint-renamed",
        "checkpoint-published",
    ] {
        let directory = ProcessDirectory::new();
        let path = directory.journal();
        kill_at_barrier(&path, phase);
        let checkpoint = phase.starts_with("checkpoint-");
        let (revision, expected_value) = if checkpoint {
            (1, 7)
        } else if phase == "before-append" {
            (0, 0)
        } else if phase == "rotated-sync" {
            (7, 7)
        } else {
            (1, 9)
        };
        let mut log = ReplayEventLog::new(
            FileEventLogBackend::open(&path, 2, MAX_PAYLOAD).unwrap(),
            MAX_PAYLOAD,
        );
        if checkpoint {
            let expected_start = if matches!(phase, "checkpoint-renamed" | "checkpoint-published") {
                1
            } else {
                0
            };
            assert_eq!(
                log.backend().start_cursor(),
                EventLogCursor::new(expected_start),
                "{phase}"
            );
        }
        let end = log.backend().cursor();
        assert_eq!(end.next_sequence(), u64::from(revision) + 1, "{phase}");
        let last = log
            .replay(EventLogCursor::new(end.next_sequence() - 1), 1)
            .unwrap()
            .records()[0]
            .clone();
        // Reopening and retrying the same physical key cannot insert a duplicate.
        assert!(
            !log.append(last.idempotency_key, last.record.clone())
                .unwrap()
                .inserted
        );
        assert_eq!(log.backend().cursor(), end);
        let (mut peer, mut recovered, applies, _) =
            connected_backend(log, Arc::new(AtomicUsize::new(0)));
        assert_eq!(recovered.revision(), Some(revision), "{phase}");
        assert_eq!(
            value(recovered.record().unwrap()),
            expected_value,
            "{phase}"
        );
        assert_eq!(
            applies.load(Ordering::Relaxed),
            0,
            "replay must not rerun commands"
        );
        for _ in 0..12 {
            command(&mut peer, 1);
            assert_eq!(recovered.pump(), 1, "{phase}");
            for _ in 0..8 {
                peer.tick();
                recovered.pump();
            }
        }
        drop(recovered);
        let file = FileEventLogBackend::open(&path, 2, MAX_PAYLOAD).unwrap();
        assert_eq!(file.len(), 2);
        assert_eq!(
            file.start_cursor().next_sequence(),
            u64::from(revision) + 11
        );
        assert!(std::fs::metadata(&path).unwrap().len() < (MAX_PAYLOAD * 2 + 128) as u64);
        let restored = durable::restore(
            &schema(),
            8,
            &snapshot(),
            ReplayEventLog::new(file, MAX_PAYLOAD),
        )
        .unwrap();
        assert_eq!(restored.session.revision(), Some(revision + 12));
        assert_eq!(
            value(restored.session.record().unwrap()),
            expected_value + 12
        );
        drop(restored);
        // A second actual process reopens the already-compacted file, rotates it
        // repeatedly, and is killed again. The parent then recovers that result.
        kill_at_barrier(&path, "resume-rotations");
        let file = FileEventLogBackend::open(&path, 2, MAX_PAYLOAD).unwrap();
        let restored = durable::restore(
            &schema(),
            8,
            &snapshot(),
            ReplayEventLog::new(file, MAX_PAYLOAD),
        )
        .unwrap();
        assert_eq!(restored.session.revision(), Some(revision + 18));
        assert_eq!(
            value(restored.session.record().unwrap()),
            expected_value + 18
        );
        assert_eq!(restored.next_sequence, u64::from(revision) + 19);
    }
    // These are process-death results: complete unsynced writes can survive in
    // the OS cache. They are not evidence of persistence through power loss.
}
