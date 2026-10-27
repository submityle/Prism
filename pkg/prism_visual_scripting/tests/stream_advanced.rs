use prism_visual_scripting::prelude::*;

fn keyed(id: &str, key: &str, partition: &str, time: u64, value: i64) -> KeyedStreamEvent {
    KeyedStreamEvent {
        event_id: id.into(),
        key: key.into(),
        partition: partition.into(),
        event_time_ms: time,
        payload: Value::I64(value),
    }
}

#[test]
fn interval_join_rejects_overflow_without_mutating_state() {
    let mut join = IntervalJoin::new(10, 10, 4).unwrap();
    let error = join
        .ingest(
            JoinSide::Left,
            KeyedStreamEvent {
                event_id: "overflow".into(),
                key: "k".into(),
                partition: "p1".into(),
                event_time_ms: u64::MAX,
                payload: Value::I64(1),
            },
        )
        .unwrap_err();
    assert!(matches!(error, VsError::Runtime(message) if message.contains("interval arithmetic")));
    assert_eq!(join.buffered_events(), 0);
}

#[test]
fn interval_join_matches_keys_and_time_bounds_deterministically() {
    let mut join = IntervalJoin::new(10, 20, 10).unwrap();
    assert!(join
        .ingest(JoinSide::Right, keyed("r2", "order", "p2", 115, 2))
        .unwrap()
        .is_empty());
    join.ingest(JoinSide::Right, keyed("r1", "order", "p2", 95, 1))
        .unwrap();
    join.ingest(JoinSide::Right, keyed("other", "other", "p2", 100, 3))
        .unwrap();
    let matched = join
        .ingest(JoinSide::Left, keyed("left", "order", "p1", 100, 7))
        .unwrap();
    assert_eq!(
        matched
            .iter()
            .map(|item| item.right.event_id.as_str())
            .collect::<Vec<_>>(),
        vec!["r1", "r2"]
    );
    assert_eq!(join.advance_watermark(200), 4);
    assert_eq!(join.buffered_events(), 0);
}

#[test]
fn session_window_bridges_adjacent_sessions_and_closes_on_watermark() {
    let mut sessions = SessionWindowStream::new(10, 10).unwrap();
    sessions.ingest(keyed("a", "user", "p", 0, 1)).unwrap();
    sessions.ingest(keyed("c", "user", "p", 20, 3)).unwrap();
    assert_eq!(sessions.sessions["user"].len(), 2);
    sessions.ingest(keyed("b", "user", "p", 10, 2)).unwrap();
    assert_eq!(sessions.sessions["user"].len(), 1);
    let closed = sessions.advance_watermark(30);
    assert_eq!(closed.len(), 1);
    assert_eq!(
        closed[0]
            .events
            .iter()
            .map(|event| event.event_id.as_str())
            .collect::<Vec<_>>(),
        vec!["a", "b", "c"]
    );
}

#[test]
fn partition_watermark_ignores_idle_partition_without_regression() {
    let mut watermarks = PartitionWatermarks::new(5);
    assert_eq!(watermarks.observe("a", 100).unwrap(), Some(95));
    assert_eq!(watermarks.observe("b", 50).unwrap(), Some(95));
    assert_eq!(watermarks.observe("a", 200).unwrap(), Some(95));
    assert_eq!(watermarks.set_idle("b", true).unwrap(), Some(195));
    assert_eq!(watermarks.observe("b", 60).unwrap(), Some(195));
}

#[test]
fn source_offsets_are_monotonic_and_digest_bound() {
    let mut offsets = SourceOffsets::default();
    assert!(offsets.commit("partition-1", 7, b"event").unwrap());
    assert!(!offsets.commit("partition-1", 7, b"event").unwrap());
    assert!(offsets
        .commit("partition-1", 7, b"changed")
        .unwrap_err()
        .to_string()
        .contains("different content"));
    assert!(offsets.commit("partition-1", 8, b"next").unwrap());
    assert!(offsets
        .commit("partition-1", 6, b"old")
        .unwrap_err()
        .to_string()
        .contains("backwards"));
}
