//! Flow-id allocation and the ergonomic flow/async emission helpers.
//!
//! A *flow* ties two trace sites together across threads: a producer calls
//! [`flow_start`] with a freshly allocated [`FlowId`], hands that id to the
//! consumer (through a task payload, channel message, etc.), and the consumer
//! calls [`flow_finish`] with the same id. The Chrome Trace UI then draws an
//! arrow from the producer's thread to the consumer's thread. [`async_begin`]
//! and [`async_end`] model the complementary "one logical operation, two
//! threads" case as a duration bar.
//!
//! Ids are drawn from a single process-global monotonic counter so they never
//! collide within a run. Timestamps are read from the platform monotonic clock
//! at the moment of emission.

extern crate alloc;

use alloc::string::String;
use core::sync::atomic::{AtomicU64, Ordering};

use prism_platform::now;

use crate::trace::flow::{record_flow, FlowPhase, FlowRecord};
use crate::trace::ring::current_thread_id;

/// Default category applied to flow/async events emitted by the job
/// instrumentation when the caller does not choose one.
pub const FLOW_CATEGORY: &str = "flow";

/// A process-unique correlation id connecting the events of one flow.
///
/// Allocate one with [`next_flow_id`] at the producing site and pass the same
/// id to the consuming site so the two are linked in the trace.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub struct FlowId(pub u64);

impl FlowId {
    /// The raw numeric id (as serialized to Chrome `id`).
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Allocate the next process-unique [`FlowId`].
pub fn next_flow_id() -> FlowId {
    static COUNTER: AtomicU64 = AtomicU64::new(1);
    FlowId(COUNTER.fetch_add(1, Ordering::Relaxed))
}

fn emit(id: FlowId, name: impl Into<String>, category: impl Into<String>, phase: FlowPhase) {
    record_flow(FlowRecord {
        name: name.into(),
        category: category.into(),
        id: id.get(),
        phase,
        thread_id: current_thread_id(),
        timestamp_nanos: now().0,
    });
}

/// Emit a flow *start* (`s`) event on the calling thread: the arrow originates
/// here.
pub fn flow_start(id: FlowId, name: impl Into<String>, category: impl Into<String>) {
    emit(id, name, category, FlowPhase::Start);
}

/// Emit a flow *step* (`t`) waypoint on the calling thread.
pub fn flow_step(id: FlowId, name: impl Into<String>, category: impl Into<String>) {
    emit(id, name, category, FlowPhase::Step);
}

/// Emit a flow *finish* (`f`) event on the calling thread: the arrow terminates
/// here.
pub fn flow_finish(id: FlowId, name: impl Into<String>, category: impl Into<String>) {
    emit(id, name, category, FlowPhase::Finish);
}

/// Emit an async *begin* (`b`) event on the calling thread.
pub fn async_begin(id: FlowId, name: impl Into<String>, category: impl Into<String>) {
    emit(id, name, category, FlowPhase::AsyncBegin);
}

/// Emit an async *end* (`e`) event on the calling thread.
pub fn async_end(id: FlowId, name: impl Into<String>, category: impl Into<String>) {
    emit(id, name, category, FlowPhase::AsyncEnd);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::flow::flow_records;

    #[test]
    fn ids_are_monotonic_and_unique() {
        let a = next_flow_id();
        let b = next_flow_id();
        assert_ne!(a, b);
        assert!(b.get() > a.get());
    }

    #[test]
    fn start_and_finish_share_id_across_calls() {
        let id = next_flow_id();
        flow_start(id, "hop", FLOW_CATEGORY);
        flow_finish(id, "hop", FLOW_CATEGORY);
        let mine: Vec<_> =
            flow_records().into_iter().filter(|r| r.id == id.get()).collect();
        assert_eq!(mine.len(), 2);
        assert_eq!(mine[0].phase, FlowPhase::Start);
        assert_eq!(mine[1].phase, FlowPhase::Finish);
        assert_eq!(mine[0].category, FLOW_CATEGORY);
    }
}
