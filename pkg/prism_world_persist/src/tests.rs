use alloc::vec;
use alloc::vec::Vec;

use super::*;

fn delta(field: u64, value: OverrideValue) -> OverrideDelta {
    OverrideDelta::single(FieldId(field), value).expect("valid test delta")
}

fn transaction(
    id: u128,
    actor: u128,
    clock: u64,
    sequence: u64,
    object: u128,
    value: OverrideValue,
) -> Transaction {
    let forward = delta(1, value);
    let inverse = delta(1, OverrideValue::Delete);
    Transaction::new(
        TransactionId(id),
        ActorId(actor),
        LogicalClock(clock),
        sequence,
        PersistentId(object),
        forward,
        inverse,
    )
    .expect("valid test transaction")
}

#[test]
fn delta_is_canonical_and_codec_round_trips_all_values() {
    let delta = OverrideDelta::new(vec![
        FieldOverride {
            field: FieldId(9),
            value: OverrideValue::Delete,
        },
        FieldOverride {
            field: FieldId(2),
            value: OverrideValue::Bytes(vec![0, 1, 255]),
        },
        FieldOverride {
            field: FieldId(7),
            value: OverrideValue::Float(2.5),
        },
        FieldOverride {
            field: FieldId(4),
            value: OverrideValue::Integer(-12),
        },
        FieldOverride {
            field: FieldId(5),
            value: OverrideValue::Unsigned(12),
        },
        FieldOverride {
            field: FieldId(6),
            value: OverrideValue::Boolean(true),
        },
    ])
    .expect("valid delta");
    assert_eq!(
        delta.iter().next().map(|change| change.field),
        Some(FieldId(2))
    );
    let encoded = encode_delta(&delta).expect("encode");
    assert_eq!(decode_delta(&encoded).expect("decode"), delta);

    let mut damaged = encoded;
    let index = damaged.len() - 5;
    damaged[index] ^= 1;
    assert!(matches!(
        decode_delta(&damaged),
        Err(CodecError::ChecksumMismatch { .. })
    ));
}

#[test]
fn append_only_log_rejects_duplicates_and_out_of_order_transactions() {
    let first = transaction(1, 1, 1, 0, 11, OverrideValue::Integer(10));
    let mut log = AppendOnlyLog::new();
    log.append(first.clone()).expect("first append");
    assert_eq!(
        log.append(first),
        Err(LogError::DuplicateTransaction(TransactionId(1)))
    );
    let skipped = transaction(2, 1, 2, 2, 11, OverrideValue::Integer(20));
    assert_eq!(
        log.append(skipped),
        Err(LogError::OutOfOrder {
            expected: 1,
            actual: 2
        })
    );
    let second = transaction(2, 1, 2, 1, 11, OverrideValue::Integer(20));
    log.append(second).expect("second append");
    let ids: Vec<_> = log
        .iter_object(PersistentId(11))
        .map(|entry| entry.id())
        .collect();
    assert_eq!(ids, vec![TransactionId(1), TransactionId(2)]);
}

#[test]
fn crdt_merge_is_order_independent_and_uses_tie_breakers() {
    let low = transaction(10, 1, 5, 0, 7, OverrideValue::Integer(1));
    let high_actor = transaction(11, 2, 5, 1, 7, OverrideValue::Integer(2));
    let high_id = transaction(12, 2, 5, 2, 7, OverrideValue::Integer(3));
    let mut first = CrdtState::new();
    first
        .merge([&high_id, &low, &high_actor])
        .expect("merge first order");
    let mut second = CrdtState::new();
    second
        .merge([&low, &high_actor, &high_id])
        .expect("merge second order");
    assert_eq!(first, second);
    assert_eq!(
        first.get(PersistentId(7), FieldId(1)),
        Some(&OverrideValue::Integer(3))
    );
    assert_eq!(
        first.merge([&low]),
        Err(CrdtError::DuplicateTransaction(TransactionId(10)))
    );
}

#[test]
fn timeline_snapshot_replay_and_binary_round_trip() {
    let mut timeline = Timeline::new();
    timeline
        .commit(
            TransactionId(20),
            ActorId(1),
            LogicalClock(1),
            PersistentId(3),
            delta(1, OverrideValue::Integer(10)),
        )
        .expect("commit one");
    let snapshot = timeline.snapshot();
    timeline
        .commit(
            TransactionId(21),
            ActorId(1),
            LogicalClock(2),
            PersistentId(3),
            delta(1, OverrideValue::Integer(20)),
        )
        .expect("commit two");
    timeline
        .commit(
            TransactionId(22),
            ActorId(1),
            LogicalClock(3),
            PersistentId(3),
            delta(2, OverrideValue::Boolean(true)),
        )
        .expect("commit three");

    let replayed = Timeline::replay(&snapshot, timeline.log()).expect("replay");
    assert_eq!(&replayed, timeline.state());
    let snapshot_bytes = encode_snapshot(&snapshot).expect("encode snapshot");
    assert_eq!(
        decode_snapshot(&snapshot_bytes).expect("decode snapshot"),
        snapshot
    );
    let log_bytes = encode_log(timeline.log()).expect("encode log");
    let decoded_log = decode_log(&log_bytes).expect("decode log");
    assert_eq!(&decoded_log, timeline.log());
}

#[test]
fn undo_and_redo_append_reverse_transactions() {
    let mut timeline = Timeline::new();
    timeline
        .commit(
            TransactionId(30),
            ActorId(1),
            LogicalClock(1),
            PersistentId(4),
            delta(9, OverrideValue::Integer(1)),
        )
        .expect("initial commit");
    timeline
        .commit(
            TransactionId(31),
            ActorId(1),
            LogicalClock(2),
            PersistentId(4),
            delta(9, OverrideValue::Integer(2)),
        )
        .expect("second commit");
    let undone = timeline
        .undo(TransactionId(32), ActorId(1), LogicalClock(3))
        .expect("undo");
    assert_eq!(
        undone.delta().get(FieldId(9)),
        Some(&OverrideValue::Integer(1))
    );
    assert_eq!(
        timeline.state().get(PersistentId(4), FieldId(9)),
        Some(&OverrideValue::Integer(1))
    );
    timeline
        .redo(TransactionId(33), ActorId(1), LogicalClock(4))
        .expect("redo");
    assert_eq!(
        timeline.state().get(PersistentId(4), FieldId(9)),
        Some(&OverrideValue::Integer(2))
    );
    assert_eq!(timeline.log().len(), 4);
}

#[test]
fn multiple_undo_operations_preserve_redo_order() {
    let mut timeline = Timeline::new();
    for (id, clock, value) in [(40, 1, 1), (41, 2, 2), (42, 3, 3)] {
        timeline
            .commit(
                TransactionId(id),
                ActorId(1),
                LogicalClock(clock),
                PersistentId(5),
                delta(1, OverrideValue::Integer(value)),
            )
            .expect("commit");
    }
    timeline
        .undo(TransactionId(43), ActorId(1), LogicalClock(4))
        .expect("undo three");
    timeline
        .undo(TransactionId(44), ActorId(1), LogicalClock(5))
        .expect("undo two");
    assert_eq!(
        timeline.state().get(PersistentId(5), FieldId(1)),
        Some(&OverrideValue::Integer(1))
    );
    timeline
        .redo(TransactionId(45), ActorId(1), LogicalClock(6))
        .expect("redo two");
    assert_eq!(
        timeline.state().get(PersistentId(5), FieldId(1)),
        Some(&OverrideValue::Integer(2))
    );
    timeline
        .redo(TransactionId(46), ActorId(1), LogicalClock(7))
        .expect("redo three");
    assert_eq!(
        timeline.state().get(PersistentId(5), FieldId(1)),
        Some(&OverrideValue::Integer(3))
    );
}

#[test]
fn invalid_float_and_inverse_are_rejected_without_panicking() {
    assert_eq!(
        OverrideDelta::single(FieldId(1), OverrideValue::Float(f64::NAN)),
        Err(DeltaError::InvalidValue(ValueError::NonFiniteFloat))
    );
    let forward = delta(1, OverrideValue::Integer(1));
    let inverse = delta(2, OverrideValue::Integer(0));
    assert_eq!(
        Transaction::new(
            TransactionId(1),
            ActorId(1),
            LogicalClock(1),
            0,
            PersistentId(1),
            forward,
            inverse,
        ),
        Err(DeltaError::InverseFieldMismatch)
    );
}
