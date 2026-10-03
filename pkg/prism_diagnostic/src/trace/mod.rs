//! CPU span tracing: per-thread ring buffers plus Chrome Trace export.
//!
//! [`ring`] holds the thread-local ring buffers and the global registry;
//! [`chrome`] turns the registered spans into Chrome Trace Event Format JSON.

pub mod chrome;
pub mod ring;

pub use chrome::{export_string as export_chrome_string, export_to_file as export_chrome_to_file};
pub use ring::{
    clear_current_thread, current_thread_id, current_thread_spans, registered_threads, RingBuffer,
    SpanRecord, ThreadTrace, DEFAULT_CAPACITY,
};
