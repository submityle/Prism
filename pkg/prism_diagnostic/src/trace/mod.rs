//! CPU span tracing: per-thread ring buffers, cross-thread flow events, plus
//! Chrome Trace export.
//!
//! [`ring`] holds the thread-local span ring buffers and the global registry;
//! [`flow`] holds the process-global ring of cross-thread flow/async events;
//! [`chrome`] turns the registered spans and flow events into Chrome Trace
//! Event Format JSON.

pub mod chrome;
pub mod flow;
pub mod ring;

pub use chrome::{export_string as export_chrome_string, export_to_file as export_chrome_to_file};
#[cfg(feature = "gpu")]
pub use chrome::{export_string_with_gpu as export_chrome_with_gpu, GPU_TRACK_TID_BASE};
pub use flow::{
    clear_flow, flow_len, flow_records, record_flow, FlowPhase, FlowRecord, DEFAULT_FLOW_CAPACITY,
};
pub use ring::{
    clear_current_thread, current_thread_id, current_thread_spans, registered_threads, RingBuffer,
    SpanRecord, ThreadTrace, DEFAULT_CAPACITY,
};
