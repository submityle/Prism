//! Thread-local ring buffers of completed span records plus the global
//! multi-thread registry.
//!
//! Each thread owns a fixed-capacity ring of [`SpanRecord`]s. Pushing when the
//! ring is full overwrites the oldest record (and bumps a dropped counter) so
//! instrumentation never blocks or grows unbounded on the hot path. Every
//! thread lazily registers its ring in a process-global registry on first use,
//! letting exporters on any thread gather every thread's track.

extern crate alloc;

use alloc::format;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::cell::Cell;
use core::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock, RwLock};

/// Default per-thread ring capacity: the number of completed spans retained
/// before the oldest are overwritten.
pub const DEFAULT_CAPACITY: usize = 4096;

/// A single completed CPU span: one closed scope on one thread.
#[derive(Clone, Debug, PartialEq)]
pub struct SpanRecord {
    /// Human-readable scope name.
    pub name: String,
    /// Optional category/track label (Chrome `cat`).
    pub category: Option<String>,
    /// Owning thread id (Prism-assigned, dense from 1).
    pub thread_id: u64,
    /// Start timestamp in nanoseconds (monotonic clock).
    pub start_nanos: u64,
    /// Measured wall duration in nanoseconds.
    pub duration_nanos: u64,
    /// Nesting depth at entry (0 is top level).
    pub depth: u32,
    /// Optional structured key/value args (Chrome `args`).
    pub args: Vec<(String, String)>,
}

/// A fixed-capacity ring of [`SpanRecord`]s. When full, pushing overwrites the
/// oldest record and increments the dropped counter rather than growing.
#[derive(Debug)]
pub struct RingBuffer {
    records: alloc::collections::VecDeque<SpanRecord>,
    capacity: usize,
    dropped: u64,
}

impl RingBuffer {
    /// Create a ring with `capacity` slots (clamped to at least 1).
    pub fn with_capacity(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            records: alloc::collections::VecDeque::with_capacity(capacity),
            capacity,
            dropped: 0,
        }
    }

    /// Create a ring with [`DEFAULT_CAPACITY`] slots.
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CAPACITY)
    }

    /// Push a completed span, overwriting the oldest record when full.
    pub fn push(&mut self, record: SpanRecord) {
        if self.records.len() == self.capacity {
            self.records.pop_front();
            self.dropped += 1;
        }
        self.records.push_back(record);
    }

    /// Clone the retained records in chronological (oldest-first) order.
    pub fn snapshot(&self) -> Vec<SpanRecord> {
        self.records.iter().cloned().collect()
    }

    /// Remove and return all retained records in chronological order.
    pub fn drain(&mut self) -> Vec<SpanRecord> {
        self.records.drain(..).collect()
    }

    /// Number of retained records.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether no records are retained.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Configured capacity (maximum retained records).
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Count of records overwritten (dropped) because the ring was full.
    pub fn dropped(&self) -> u64 {
        self.dropped
    }

    /// Discard all retained records (leaves the dropped counter unchanged).
    pub fn clear(&mut self) {
        self.records.clear();
    }
}

impl Default for RingBuffer {
    fn default() -> Self {
        Self::new()
    }
}

/// One thread's trace track: identity plus its ring buffer, shared with the
/// global registry so exporters on other threads can read it.
#[derive(Debug)]
pub struct ThreadTrace {
    /// Prism-assigned dense thread id.
    pub thread_id: u64,
    /// Thread name if the runtime provided one, else a synthetic label.
    pub thread_name: String,
    /// This thread's completed-span ring buffer.
    pub buffer: Mutex<RingBuffer>,
}

fn registry() -> &'static RwLock<Vec<Arc<ThreadTrace>>> {
    static REG: OnceLock<RwLock<Vec<Arc<ThreadTrace>>>> = OnceLock::new();
    REG.get_or_init(|| RwLock::new(Vec::new()))
}

fn next_thread_id() -> u64 {
    static COUNTER: AtomicU64 = AtomicU64::new(1);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Snapshot shared handles to every registered thread's track.
pub fn registered_threads() -> Vec<Arc<ThreadTrace>> {
    registry().read().map(|r| r.clone()).unwrap_or_default()
}

struct ThreadState {
    trace: Arc<ThreadTrace>,
    depth: Cell<u32>,
}

impl ThreadState {
    fn register() -> Self {
        let thread_id = next_thread_id();
        let thread_name = std::thread::current()
            .name()
            .map(String::from)
            .unwrap_or_else(|| format!("thread-{thread_id}"));
        let trace = Arc::new(ThreadTrace {
            thread_id,
            thread_name,
            buffer: Mutex::new(RingBuffer::new()),
        });
        if let Ok(mut reg) = registry().write() {
            reg.push(Arc::clone(&trace));
        }
        Self {
            trace,
            depth: Cell::new(0),
        }
    }
}

thread_local! {
    static STATE: ThreadState = ThreadState::register();
}

/// Enter a scope on the calling thread: returns the entry depth and the
/// Prism-assigned thread id, lazily registering the thread on first use.
pub(crate) fn enter() -> (u32, u64) {
    STATE.with(|st| {
        let depth = st.depth.get();
        st.depth.set(depth + 1);
        (depth, st.trace.thread_id)
    })
}

/// Leave a scope on the calling thread, recording `record` into the ring.
pub(crate) fn leave(record: SpanRecord) {
    STATE.with(|st| {
        st.depth.set(st.depth.get().saturating_sub(1));
        if let Ok(mut buf) = st.trace.buffer.lock() {
            buf.push(record);
        }
    });
}

/// Snapshot the calling thread's retained spans without draining them.
pub fn current_thread_spans() -> Vec<SpanRecord> {
    STATE.with(|st| {
        st.trace
            .buffer
            .lock()
            .map(|b| b.snapshot())
            .unwrap_or_default()
    })
}

/// Clear the calling thread's ring buffer (useful for test isolation).
pub fn clear_current_thread() {
    STATE.with(|st| {
        if let Ok(mut buf) = st.trace.buffer.lock() {
            buf.clear();
        }
    });
}

/// The calling thread's Prism-assigned id (registers the thread if needed).
pub fn current_thread_id() -> u64 {
    STATE.with(|st| st.trace.thread_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(start: u64) -> SpanRecord {
        SpanRecord {
            name: String::from("s"),
            category: None,
            thread_id: 1,
            start_nanos: start,
            duration_nanos: 10,
            depth: 0,
            args: Vec::new(),
        }
    }

    #[test]
    fn ring_overwrites_oldest_when_full() {
        let mut ring = RingBuffer::with_capacity(3);
        assert!(ring.is_empty());
        for i in 0..5 {
            ring.push(rec(i));
        }
        assert_eq!(ring.capacity(), 3);
        assert_eq!(ring.len(), 3);
        assert_eq!(ring.dropped(), 2);
        let snap = ring.snapshot();
        assert_eq!(snap.len(), 3);
        // Oldest two (start 0,1) were overwritten; 2,3,4 remain in order.
        assert_eq!(snap[0].start_nanos, 2);
        assert_eq!(snap[1].start_nanos, 3);
        assert_eq!(snap[2].start_nanos, 4);
    }

    #[test]
    fn ring_capacity_floor_is_one() {
        let mut ring = RingBuffer::with_capacity(0);
        assert_eq!(ring.capacity(), 1);
        ring.push(rec(7));
        ring.push(rec(8));
        assert_eq!(ring.len(), 1);
        assert_eq!(ring.snapshot()[0].start_nanos, 8);
    }
}
