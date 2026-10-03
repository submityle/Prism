//! Cross-thread flow and async trace events.
//!
//! A complete ([`SpanRecord`](super::ring::SpanRecord)) event lives on exactly
//! one thread, so it cannot express "a job was enqueued on the main thread and
//! finished on a worker thread". Chrome Trace Event Format solves this with two
//! families of connected events, both captured here by [`FlowRecord`]:
//!
//! - **Flow** events (`s` start, `t` step, `f` finish) draw an arrow from the
//!   producing site to the consuming site. The arrow connects every event that
//!   shares the same `(id, cat)` pair.
//! - **Async** events (`b` begin, `e` end) model an operation whose begin and
//!   end may land on different threads; the UI pairs them by `(id, cat, name)`.
//!
//! Flow events are comparatively rare (per job / per hand-off, not per span) and
//! inherently cross-thread, so they share a single process-global
//! [`RingBuffer`] guarded by a mutex rather than a per-thread ring. The buffer
//! overwrites its oldest record when full, exactly like the per-thread span
//! rings, so recording a flow event never blocks or grows unbounded.

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;
use std::sync::{Mutex, OnceLock};

use super::ring::RingBuffer;

/// Default capacity of the process-global flow-event ring.
pub const DEFAULT_FLOW_CAPACITY: usize = 4096;

/// The Chrome Trace phase of a [`FlowRecord`].
///
/// The first three variants are *flow* phases (`s`/`t`/`f`) that render as a
/// connecting arrow; the last two are *async* phases (`b`/`e`) that render as a
/// duration bar spanning the begin/end pair.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FlowPhase {
    /// Flow start (`"ph":"s"`): the arrow originates here.
    Start,
    /// Flow step (`"ph":"t"`): an intermediate waypoint on the arrow.
    Step,
    /// Flow finish (`"ph":"f"`): the arrow terminates here.
    Finish,
    /// Async begin (`"ph":"b"`): the operation's duration bar opens here.
    AsyncBegin,
    /// Async end (`"ph":"e"`): the operation's duration bar closes here.
    AsyncEnd,
}

impl FlowPhase {
    /// The single-character Chrome Trace `ph` code for this phase.
    pub const fn chrome_ph(self) -> &'static str {
        match self {
            FlowPhase::Start => "s",
            FlowPhase::Step => "t",
            FlowPhase::Finish => "f",
            FlowPhase::AsyncBegin => "b",
            FlowPhase::AsyncEnd => "e",
        }
    }

    /// Whether this is a flow phase (`s`/`t`/`f`) rather than an async phase.
    pub const fn is_flow(self) -> bool {
        matches!(self, FlowPhase::Start | FlowPhase::Step | FlowPhase::Finish)
    }

    /// Whether this is an async phase (`b`/`e`) rather than a flow phase.
    pub const fn is_async(self) -> bool {
        matches!(self, FlowPhase::AsyncBegin | FlowPhase::AsyncEnd)
    }
}

/// A single cross-thread flow or async trace event.
///
/// Events are connected in the trace UI by their [`id`](FlowRecord::id) together
/// with their [`category`](FlowRecord::category) (and, for async events, their
/// [`name`](FlowRecord::name)), so a producer and consumer on different threads
/// must share the same id and category to be linked.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FlowRecord {
    /// Human-readable name (Chrome `name`).
    pub name: String,
    /// Category/track label (Chrome `cat`); required for flow linkage.
    pub category: String,
    /// Correlation id shared by every event on the same flow (Chrome `id`).
    pub id: u64,
    /// Which phase of the flow/async pair this record represents.
    pub phase: FlowPhase,
    /// Prism-assigned id of the thread that emitted this event (Chrome `tid`).
    pub thread_id: u64,
    /// Emission timestamp in nanoseconds (monotonic clock).
    pub timestamp_nanos: u64,
}

fn flow_buffer() -> &'static Mutex<RingBuffer<FlowRecord>> {
    static BUF: OnceLock<Mutex<RingBuffer<FlowRecord>>> = OnceLock::new();
    BUF.get_or_init(|| Mutex::new(RingBuffer::with_capacity(DEFAULT_FLOW_CAPACITY)))
}

/// Record one cross-thread flow/async event into the process-global ring.
///
/// Overwrites the oldest record when the ring is full; never blocks beyond the
/// brief buffer lock and never allocates unboundedly.
pub fn record_flow(record: FlowRecord) {
    if let Ok(mut buf) = flow_buffer().lock() {
        buf.push(record);
    }
}

/// Snapshot every retained flow/async event in chronological order.
pub fn flow_records() -> Vec<FlowRecord> {
    flow_buffer()
        .lock()
        .map(|b| b.snapshot())
        .unwrap_or_default()
}

/// Number of flow/async events currently retained.
pub fn flow_len() -> usize {
    flow_buffer().lock().map(|b| b.len()).unwrap_or(0)
}

/// Discard all retained flow/async events (useful for test isolation).
pub fn clear_flow() {
    if let Ok(mut buf) = flow_buffer().lock() {
        buf.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(id: u64, phase: FlowPhase) -> FlowRecord {
        FlowRecord {
            name: String::from("job"),
            category: String::from("flow"),
            id,
            phase,
            thread_id: 1,
            timestamp_nanos: 100,
        }
    }

    #[test]
    fn phase_codes_and_classification() {
        assert_eq!(FlowPhase::Start.chrome_ph(), "s");
        assert_eq!(FlowPhase::Step.chrome_ph(), "t");
        assert_eq!(FlowPhase::Finish.chrome_ph(), "f");
        assert_eq!(FlowPhase::AsyncBegin.chrome_ph(), "b");
        assert_eq!(FlowPhase::AsyncEnd.chrome_ph(), "e");
        assert!(FlowPhase::Start.is_flow() && !FlowPhase::Start.is_async());
        assert!(FlowPhase::AsyncEnd.is_async() && !FlowPhase::AsyncEnd.is_flow());
    }

    #[test]
    fn record_and_snapshot_roundtrip() {
        // Use a unique id and filter rather than clearing, so the assertions
        // hold even while other tests share the process-global flow buffer.
        const ID: u64 = 0x00F1_0A01;
        record_flow(rec(ID, FlowPhase::Start));
        record_flow(rec(ID, FlowPhase::Finish));
        let mine: Vec<_> = flow_records().into_iter().filter(|r| r.id == ID).collect();
        assert_eq!(mine.len(), 2);
        assert_eq!(mine[0].phase, FlowPhase::Start);
        assert_eq!(mine[1].phase, FlowPhase::Finish);
    }
}
