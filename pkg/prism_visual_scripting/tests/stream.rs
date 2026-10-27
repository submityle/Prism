use prism_visual_scripting::prelude::*;

fn event(id: &str, time: u64, value: i64) -> StreamEvent {
    StreamEvent {
        event_id: id.into(),
        event_time_ms: time,
        payload: Value::I64(value),
    }
}

#[test]
fn window_end_overflow_is_rejected_without_mutating_stream() {
    let mut stream = WindowedStream::new(
        WindowSpec::Tumbling { size_ms: 2 },
        0,
        4,
        BackpressurePolicy::Reject,
    )
    .unwrap();
    let error = stream.ingest(event("overflow", u64::MAX, 1)).unwrap_err();
    assert!(matches!(
        error,
        StreamIngestError::Arithmetic { message } if message.contains("end time exhausted")
    ));
    assert_eq!(stream.buffered_events(), 0);
    assert!(stream.seen_event_ids.is_empty());
}

#[test]
fn lateness_deadline_overflow_is_rejected_without_mutating_stream() {
    let mut stream = WindowedStream::new(
        WindowSpec::Tumbling { size_ms: 1 },
        u64::MAX - 1,
        4,
        BackpressurePolicy::Reject,
    )
    .unwrap();
    let error = stream
        .ingest(event("late-deadline", u64::MAX - 1, 1))
        .unwrap_err();
    assert!(matches!(
        error,
        StreamIngestError::Arithmetic { message } if message.contains("lateness deadline")
    ));
    assert_eq!(stream.buffered_events(), 0);
    assert!(stream.seen_event_ids.is_empty());
    assert!(stream.max_event_time_ms.is_none());
}

#[test]
fn tumbling_window_orders_out_of_order_events_and_closes_by_watermark() {
    let mut stream = WindowedStream::new(
        WindowSpec::Tumbling { size_ms: 100 },
        20,
        10,
        BackpressurePolicy::Reject,
    )
    .unwrap();
    assert_eq!(
        stream.ingest(event("late-arrival", 80, 8)).unwrap().0,
        StreamIngest::Accepted
    );
    assert_eq!(
        stream.ingest(event("early-arrival", 70, 2)).unwrap().0,
        StreamIngest::Accepted
    );
    assert_eq!(
        stream.ingest(event("late-arrival", 80, 8)).unwrap().0,
        StreamIngest::Duplicate
    );
    let closed = stream.advance_watermark(100);
    assert_eq!(closed.len(), 1);
    assert_eq!(closed[0].start_ms, 0);
    assert_eq!(closed[0].events[0].event_id, "early-arrival");
    assert_eq!(closed[0].events[1].event_id, "late-arrival");
}

#[test]
fn late_event_is_rejected_after_window_closes() {
    let mut stream = WindowedStream::new(
        WindowSpec::Tumbling { size_ms: 100 },
        0,
        10,
        BackpressurePolicy::Reject,
    )
    .unwrap();
    stream.ingest(event("first", 10, 1)).unwrap();
    stream.advance_watermark(100);
    let error = stream.ingest(event("late", 10, 2)).unwrap_err();
    assert!(matches!(
        error,
        StreamIngestError::Late {
            reason: LateEvent::AlreadyClosed { window_start_ms: 0 }
        }
    ));
}

#[test]
fn sliding_window_duplicates_events_across_expected_windows() {
    let mut stream = WindowedStream::new(
        WindowSpec::Sliding {
            size_ms: 100,
            step_ms: 50,
        },
        0,
        10,
        BackpressurePolicy::Reject,
    )
    .unwrap();
    stream.ingest(event("e", 75, 1)).unwrap();
    assert_eq!(stream.windows.len(), 2);
    assert!(stream.windows.values().all(|events| events.len() == 1));
}

#[test]
fn drop_oldest_backpressure_is_bounded_and_checkpointable() {
    let mut stream = WindowedStream::new(
        WindowSpec::Tumbling { size_ms: 100 },
        1000,
        2,
        BackpressurePolicy::DropOldest,
    )
    .unwrap();
    stream.ingest(event("a", 0, 1)).unwrap();
    stream.ingest(event("b", 100, 2)).unwrap();
    let (result, _) = stream.ingest(event("c", 200, 3)).unwrap();
    assert_eq!(result, StreamIngest::DroppedOldest);
    assert!(stream.buffered_events() <= 2);
    assert_eq!(stream.dropped_events(), 1);
    let remaining_ids = stream
        .windows
        .values()
        .flatten()
        .map(|event| event.event_id.as_str())
        .collect::<Vec<_>>();
    assert_eq!(remaining_ids, vec!["b", "c"]);
    let encoded = serde_json::to_vec(&stream).unwrap();
    let restored: WindowedStream = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(restored, stream);
}

#[test]
fn reject_backpressure_has_a_distinct_error_contract() {
    let mut stream = WindowedStream::new(
        WindowSpec::Tumbling { size_ms: 100 },
        1000,
        1,
        BackpressurePolicy::Reject,
    )
    .unwrap();
    stream.ingest(event("a", 0, 1)).unwrap();
    let error = stream.ingest(event("b", 100, 2)).unwrap_err();
    assert_eq!(
        error,
        StreamIngestError::Backpressure {
            capacity: 1,
            required: 2,
        }
    );
    assert_eq!(stream.max_event_time_ms, Some(0));
}

fn closed(events: Vec<StreamEvent>) -> ClosedWindow {
    ClosedWindow {
        start_ms: 0,
        end_ms: 100,
        events,
    }
}

#[test]
fn reducers_compute_numeric_and_distinct_aggregates() {
    let window = closed(vec![
        event("a", 10, 2),
        event("b", 20, 4),
        event("c", 30, 4),
    ]);
    let cases = [
        (AggregationSpec::Count, Value::I64(3)),
        (AggregationSpec::Sum, Value::F64(10.0)),
        (AggregationSpec::Average, Value::F64(10.0 / 3.0)),
        (AggregationSpec::Min, Value::F64(2.0)),
        (AggregationSpec::Max, Value::F64(4.0)),
        (AggregationSpec::DistinctCount, Value::I64(2)),
    ];
    for (spec, expected) in cases {
        let outcome = WindowReducer::new(spec).reduce(&window).unwrap();
        let ReductionOutcome::Applied { aggregate } = outcome else {
            panic!("first reduction must apply");
        };
        assert_eq!(aggregate.value, expected);
        assert_eq!(aggregate.event_count, 3);
    }
}

#[test]
fn reducer_replay_is_idempotent_and_detects_changed_window() {
    let window = closed(vec![event("a", 10, 1)]);
    let mut reducer = WindowReducer::new(AggregationSpec::Sum);
    let ReductionOutcome::Applied { aggregate } = reducer.reduce(&window).unwrap() else {
        panic!("first reduction must apply");
    };
    let ReductionOutcome::Replay { aggregate: replay } = reducer.reduce(&window).unwrap() else {
        panic!("same reduction must replay");
    };
    assert_eq!(replay, aggregate);
    let changed = closed(vec![event("a", 10, 2)]);
    let error = reducer.reduce(&changed).unwrap_err();
    assert!(error.to_string().contains("different content"));
    let restored: WindowReducer =
        serde_json::from_slice(&serde_json::to_vec(&reducer).unwrap()).unwrap();
    assert_eq!(restored, reducer);
}

#[test]
fn numeric_reducer_rejects_non_numeric_payload() {
    let window = ClosedWindow {
        start_ms: 0,
        end_ms: 100,
        events: vec![StreamEvent {
            event_id: "text".into(),
            event_time_ms: 10,
            payload: Value::String("not-a-number".into()),
        }],
    };
    let error = WindowReducer::new(AggregationSpec::Sum)
        .reduce(&window)
        .unwrap_err();
    assert!(matches!(error, VsError::TypeMismatch { .. }));
}

#[test]
fn sliding_assignment_cannot_exceed_hard_capacity() {
    let mut stream = WindowedStream::new(
        WindowSpec::Sliding {
            size_ms: 100,
            step_ms: 25,
        },
        0,
        2,
        BackpressurePolicy::DropOldest,
    )
    .unwrap();
    let error = stream.ingest(event("wide", 75, 1)).unwrap_err();
    assert_eq!(
        error,
        StreamIngestError::Backpressure {
            capacity: 2,
            required: 4,
        }
    );
    assert_eq!(stream.buffered_events(), 0);
}

#[test]
fn numeric_reducer_rejects_non_finite_values() {
    let window = ClosedWindow {
        start_ms: 0,
        end_ms: 100,
        events: vec![StreamEvent {
            event_id: "nan".into(),
            event_time_ms: 10,
            payload: Value::F64(f64::NAN),
        }],
    };
    let error = WindowReducer::new(AggregationSpec::Sum)
        .reduce(&window)
        .unwrap_err();
    assert!(error.to_string().contains("non-finite"));
}
