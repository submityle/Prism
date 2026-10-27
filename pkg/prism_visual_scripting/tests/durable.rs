use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use prism_visual_scripting::prelude::*;

fn key() -> RecordKey {
    RecordKey::new("workflow", "run/42").unwrap()
}

fn temp_store_path(label: &str) -> std::path::PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "uwu-visual-script-{label}-{}-{nonce}",
        std::process::id()
    ))
}

#[test]
fn event_sequence_allocator_rejects_u64_exhaustion() {
    let error = prism_visual_scripting::durable::next_event_sequence(usize::MAX).unwrap_err();
    assert!(error.to_string().contains("event sequence exhausted"));
}
#[test]
fn in_memory_store_enforces_create_cas_and_payload_integrity() {
    let store = InMemoryDurableStore::default();
    let record = store
        .commit(
            &key(),
            RevisionGuard::Create,
            DurableMutation::default()
                .with_payload(&vec!["checkpoint", "one"])
                .unwrap(),
        )
        .unwrap();
    assert_eq!(record.revision, 1);
    assert_eq!(
        record.decode::<Vec<String>>().unwrap(),
        vec!["checkpoint".to_string(), "one".to_string()]
    );

    let error = store
        .commit(&key(), RevisionGuard::Create, DurableMutation::default())
        .unwrap_err();
    assert!(error.to_string().contains("already exists"));
    let error = store
        .commit(&key(), RevisionGuard::Match(0), DurableMutation::default())
        .unwrap_err();
    assert!(error.to_string().contains("revision conflict"));
}

#[test]
fn durable_events_reject_invalid_execution_identity() {
    for execution_id in [String::new(), "x".repeat(129)] {
        let error = InMemoryDurableStore::default()
            .commit(
                &key(),
                RevisionGuard::Create,
                DurableMutation {
                    events: vec![ExecutionEvent {
                        sequence: 0,
                        execution_id,
                        task_id: 1,
                        frame_depth: 0,
                        node_id: 1,
                        timestamp_ms: 0,
                        event: ExecutionEventKind::NodeStarted {
                            def_id: "test.node".into(),
                        },
                    }],
                    ..Default::default()
                },
            )
            .unwrap_err();
        assert!(error.to_string().contains("execution identity"));
    }
}

#[test]
fn inbox_outbox_and_events_are_atomic_and_idempotent() {
    let store = InMemoryDurableStore::default();
    let k = key();
    let event = ExecutionEvent {
        sequence: 0,
        execution_id: "run/42".into(),
        task_id: 1,
        frame_depth: 0,
        node_id: 7,
        timestamp_ms: 10,
        event: ExecutionEventKind::NodeStarted {
            def_id: "test.node".into(),
        },
    };
    let record = store
        .commit(
            &k,
            RevisionGuard::Create,
            DurableMutation {
                inbox: vec![InboxEntry::new("message-1", b"payload")],
                outbox: vec![OutboxEntry {
                    id: "out-1".into(),
                    topic: "events".into(),
                    payload: Value::I64(7),
                    dispatched: false,
                }],
                acknowledge_outbox: vec!["out-1".into()],
                events: vec![event.clone()],
                ..Default::default()
            },
        )
        .unwrap();
    assert!(record.outbox["out-1"].dispatched);
    assert_eq!(record.events[0].sequence, 1);

    let duplicate = store
        .commit(
            &k,
            RevisionGuard::Match(record.revision),
            DurableMutation {
                inbox: vec![InboxEntry::new("message-1", b"payload")],
                outbox: vec![OutboxEntry {
                    id: "out-1".into(),
                    topic: "events".into(),
                    payload: Value::I64(7),
                    dispatched: false,
                }],
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(duplicate.inbox.len(), 1);
    assert_eq!(duplicate.outbox.len(), 1);

    let error = store
        .commit(
            &k,
            RevisionGuard::Match(duplicate.revision),
            DurableMutation {
                inbox: vec![InboxEntry::new("message-1", b"tampered")],
                ..Default::default()
            },
        )
        .unwrap_err();
    assert!(error.to_string().contains("different content"));
}

#[test]
fn file_store_round_trips_and_rejects_stale_revision() {
    let path = temp_store_path("durable");
    let key = key();
    let store = FileDurableStore::open(&path).unwrap();
    let first = store
        .commit(
            &key,
            RevisionGuard::Create,
            DurableMutation::default()
                .with_payload(&Value::String("persisted".into()))
                .unwrap(),
        )
        .unwrap();
    drop(store);

    let reopened = FileDurableStore::open(&path).unwrap();
    let loaded = reopened.load(&key).unwrap().unwrap();
    assert_eq!(loaded, first);
    assert_eq!(
        loaded.decode::<Value>().unwrap(),
        Value::String("persisted".into())
    );
    let error = reopened
        .commit(&key, RevisionGuard::Match(0), DurableMutation::default())
        .unwrap_err();
    assert!(error.to_string().contains("revision conflict"));
    std::fs::remove_dir_all(path).unwrap();
}

#[test]
fn file_store_serializes_concurrent_cas_on_one_instance() {
    let path = temp_store_path("concurrent");
    let key = key();
    let store = Arc::new(FileDurableStore::open(&path).unwrap());
    store
        .commit(&key, RevisionGuard::Create, DurableMutation::default())
        .unwrap();
    let handles = (0..2)
        .map(|_| {
            let store = store.clone();
            let key = key.clone();
            std::thread::spawn(move || {
                store.commit(&key, RevisionGuard::Match(1), DurableMutation::default())
            })
        })
        .collect::<Vec<_>>();
    let successes = handles
        .into_iter()
        .map(|handle| handle.join().unwrap())
        .filter(Result::is_ok)
        .count();
    assert_eq!(successes, 1);
    assert_eq!(store.load(&key).unwrap().unwrap().revision, 2);
    drop(store);
    std::fs::remove_dir_all(path).unwrap();
}

#[test]
fn run_coordinator_binds_identity_and_revision() {
    let store: Arc<dyn DurableStore> = Arc::new(InMemoryDurableStore::default());
    let coordinator = RunCoordinator::new(store);
    let first = coordinator
        .start("run-1", "execution-1", b"checkpoint-1".to_vec())
        .unwrap();
    let second = coordinator
        .checkpoint(
            "run-1",
            first.revision,
            "execution-1",
            RunStatus::Suspended,
            b"checkpoint-2".to_vec(),
        )
        .unwrap();
    let (_, run) = coordinator.load("run-1").unwrap().unwrap();
    assert_eq!(run.status, RunStatus::Suspended);
    assert_eq!(run.checkpoint, b"checkpoint-2");

    let identity_error = coordinator
        .checkpoint(
            "run-1",
            second.revision,
            "execution-2",
            RunStatus::Running,
            vec![],
        )
        .unwrap_err();
    assert!(identity_error.to_string().contains("identity mismatch"));
    let revision_error = coordinator
        .checkpoint(
            "run-1",
            first.revision,
            "execution-1",
            RunStatus::Running,
            vec![],
        )
        .unwrap_err();
    assert!(revision_error.to_string().contains("revision conflict"));
}
