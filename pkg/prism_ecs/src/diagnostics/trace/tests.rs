//! Tests for the `trace` feature: event model, buffer bounding, Chrome export,
//! and the `std` recorder bridge.

use super::buffer::TraceBuffer;
use super::chrome::to_chrome_json;
use super::event::{EventPhase, TraceArgValue, TraceEvent, TrackId};

#[test]
fn push_stamps_monotonic_sequence_in_record_order() {
    let mut buf = TraceBuffer::new();
    buf.begin(TrackId::MAIN, 100, "a");
    buf.end(TrackId::MAIN, 200, "a");
    buf.instant(TrackId(1), 150, "mark");

    let seqs: alloc::vec::Vec<u64> = buf.events().iter().map(TraceEvent::seq).collect();
    assert_eq!(seqs, alloc::vec![0, 1, 2]);
    assert_eq!(buf.total_pushed(), 3);
    assert_eq!(buf.events()[0].name(), "a");
    assert_eq!(buf.events()[2].track(), TrackId(1));
    assert_eq!(buf.events()[2].phase(), EventPhase::Instant);
}

#[test]
fn bounded_buffer_evicts_oldest_and_counts_drops() {
    let mut buf = TraceBuffer::with_capacity(3);
    for i in 0..5u64 {
        buf.instant(TrackId::MAIN, i * 10, "e");
    }
    // Only the 3 most recent are retained; 2 were evicted.
    assert_eq!(buf.len(), 3);
    assert_eq!(buf.dropped(), 2);
    assert_eq!(buf.total_pushed(), 5);
    // Oldest retained is the third pushed (seq 2), newest is seq 4.
    assert_eq!(buf.events().first().unwrap().seq(), 2);
    assert_eq!(buf.events().last().unwrap().seq(), 4);
    assert_eq!(buf.events().first().unwrap().timestamp_ns(), 20);
}

#[test]
fn with_capacity_zero_is_unbounded() {
    let mut buf = TraceBuffer::with_capacity(0);
    for i in 0..10u64 {
        buf.instant(TrackId::MAIN, i, "e");
    }
    assert_eq!(buf.len(), 10);
    assert_eq!(buf.dropped(), 0);
}

#[test]
fn clear_resets_sequence_and_drops() {
    let mut buf = TraceBuffer::with_capacity(2);
    for i in 0..4u64 {
        buf.instant(TrackId::MAIN, i, "e");
    }
    assert_eq!(buf.dropped(), 2);
    buf.clear();
    assert!(buf.is_empty());
    assert_eq!(buf.dropped(), 0);
    assert_eq!(buf.total_pushed(), 0);
    assert_eq!(buf.begin(TrackId::MAIN, 0, "x"), 0);
}

#[test]
fn balance_detects_matched_and_unmatched_spans() {
    let mut ok = TraceBuffer::new();
    ok.begin(TrackId::MAIN, 0, "outer");
    ok.begin(TrackId::MAIN, 1, "inner");
    ok.end(TrackId::MAIN, 2, "inner");
    ok.end(TrackId::MAIN, 3, "outer");
    assert!(ok.is_balanced());

    let mut dangling = TraceBuffer::new();
    dangling.begin(TrackId::MAIN, 0, "outer");
    assert!(!dangling.is_balanced());

    let mut extra_end = TraceBuffer::new();
    extra_end.begin(TrackId::MAIN, 0, "a");
    extra_end.end(TrackId::MAIN, 1, "a");
    extra_end.end(TrackId::MAIN, 2, "a");
    assert!(!extra_end.is_balanced());
}

#[test]
fn balance_is_per_track() {
    let mut buf = TraceBuffer::new();
    buf.begin(TrackId(0), 0, "a");
    buf.begin(TrackId(1), 0, "b");
    buf.end(TrackId(1), 1, "b");
    buf.end(TrackId(0), 2, "a");
    assert!(buf.is_balanced());
}

#[test]
fn chrome_json_empty_buffer_is_valid_object() {
    let buf = TraceBuffer::new();
    assert_eq!(
        to_chrome_json(&buf),
        "{\"displayTimeUnit\":\"ns\",\"traceEvents\":[]}"
    );
}

#[test]
fn chrome_json_complete_event_has_ts_dur_in_micros() {
    let mut buf = TraceBuffer::new();
    // 12_345 ns -> 12.345 us; 8_000 ns -> 8.000 us.
    buf.complete(TrackId(2), 12_345, 8_000, "physics");
    let json = to_chrome_json(&buf);
    assert!(json.contains("\"name\":\"physics\""), "{json}");
    assert!(json.contains("\"ph\":\"X\""), "{json}");
    assert!(json.contains("\"ts\":12.345"), "{json}");
    assert!(json.contains("\"dur\":8.000"), "{json}");
    assert!(json.contains("\"tid\":2"), "{json}");
    assert!(json.contains("\"pid\":1"), "{json}");
}

#[test]
fn chrome_json_instant_has_scope_and_no_dur() {
    let mut buf = TraceBuffer::new();
    buf.instant(TrackId::MAIN, 1_000, "frame");
    let json = to_chrome_json(&buf);
    assert!(json.contains("\"ph\":\"i\""), "{json}");
    assert!(json.contains("\"s\":\"t\""), "{json}");
    assert!(!json.contains("\"dur\""), "{json}");
    assert!(json.contains("\"ts\":1.000"), "{json}");
}

#[test]
fn chrome_json_sub_microsecond_timestamp_zero_pads() {
    let mut buf = TraceBuffer::new();
    buf.instant(TrackId::MAIN, 7, "tiny");
    let json = to_chrome_json(&buf);
    assert!(json.contains("\"ts\":0.007"), "{json}");
}

#[test]
fn chrome_json_serialises_typed_args() {
    let mut buf = TraceBuffer::new();
    buf.push(
        TraceEvent::complete(TrackId::MAIN, 0, 10, "sys")
            .with_category("system")
            .with_arg("count", TraceArgValue::Uint(42))
            .with_arg("delta", TraceArgValue::Int(-3))
            .with_arg("saturated", TraceArgValue::Bool(true)),
    );
    let json = to_chrome_json(&buf);
    assert!(json.contains("\"args\":{"), "{json}");
    assert!(json.contains("\"count\":42"), "{json}");
    assert!(json.contains("\"delta\":-3"), "{json}");
    assert!(json.contains("\"saturated\":true"), "{json}");
    assert!(json.contains("\"cat\":\"system\""), "{json}");
}

#[test]
fn chrome_json_escapes_control_and_quote_characters() {
    let mut buf = TraceBuffer::new();
    buf.push(
        TraceEvent::instant(TrackId::MAIN, 0, "quote\"back\\slash\nnewline\ttab")
            .with_arg("note", TraceArgValue::Str(alloc::string::String::from("bell\u{07}"))),
    );
    let json = to_chrome_json(&buf);
    assert!(json.contains("quote\\\"back\\\\slash\\nnewline\\ttab"), "{json}");
    assert!(json.contains("\\u0007"), "{json}");
}

#[test]
fn complete_counted_sets_category_and_single_arg() {
    let mut buf = TraceBuffer::new();
    buf.complete_counted(TrackId::MAIN, 1_000, 2_000, "system", "ai", "entities", 9);
    let ev = &buf.events()[0];
    assert_eq!(ev.category(), "system");
    assert_eq!(ev.name(), "ai");
    assert_eq!(ev.duration_ns(), 2_000);
    assert_eq!(ev.args().len(), 1);
    assert_eq!(ev.args()[0].key(), "entities");
    assert_eq!(*ev.args()[0].value(), TraceArgValue::Uint(9));
}

#[cfg(feature = "std")]
mod std_recorder {
    use super::super::buffer::TraceBuffer;
    use super::super::event::{EventPhase, TrackId};
    use super::super::recorder::TraceRecorder;
    use crate::diagnostics::profiler::SystemInstrument;

    #[test]
    fn recorder_captures_nested_spans_as_begin_end_stream() {
        let mut rec = TraceRecorder::new().on_track(TrackId(3)).with_category("sys");
        rec.begin("outer");
        rec.begin("inner");
        rec.end();
        rec.mark("midpoint");
        rec.end();
        assert!(rec.is_balanced());

        let buf: TraceBuffer = rec.into_buffer();
        let phases: alloc::vec::Vec<EventPhase> =
            buf.events().iter().map(|e| e.phase()).collect();
        assert_eq!(
            phases,
            alloc::vec![
                EventPhase::Begin,
                EventPhase::Begin,
                EventPhase::End,
                EventPhase::Instant,
                EventPhase::End,
            ]
        );
        assert_eq!(buf.events()[0].name(), "outer");
        assert_eq!(buf.events()[1].name(), "inner");
        // End events are named after the span they close (stack discipline).
        assert_eq!(buf.events()[2].name(), "inner");
        assert_eq!(buf.events()[4].name(), "outer");
        assert_eq!(buf.events()[0].track(), TrackId(3));
        assert_eq!(buf.events()[0].category(), "sys");
        assert!(buf.is_balanced());
    }

    #[test]
    fn recorder_timestamps_are_monotonic_nondecreasing() {
        let mut rec = TraceRecorder::new();
        rec.begin("a");
        rec.end();
        rec.begin("b");
        rec.end();
        let buf = rec.into_buffer();
        let mut last = 0u64;
        for ev in buf.events() {
            assert!(ev.timestamp_ns() >= last, "timestamps must not go backwards");
            last = ev.timestamp_ns();
        }
    }
}
