//! Per-worker job trace, steal/migration + occupancy counters, and a
//! chrome-tracing (flamegraph-compatible) `JSON` export (design §16).
//!
//! A [`JobTrace`] is a cheap-to-clone handle around shared, lock-protected
//! span storage plus per-worker atomic counters. Instrument any job with
//! [`JobTrace::instrument`] (which wraps a closure so it is timed and recorded
//! when it actually runs on a worker) or [`JobTrace::record`] for synchronous
//! inline work. The [`crate::Pipeline`] can record a span per stage job via
//! [`crate::Pipeline::run_traced`].
//!
//! Everything a `JobTrace` reports is a *real measured value*:
//! - A [`Span`] carries the executing worker id, start/end timestamps relative
//!   to the trace origin, and the id of the enclosing span (its parent in the
//!   job graph, captured on the enqueuing thread).
//! - The **migration counter** (`steal_rate`) counts jobs that executed on a
//!   different worker than the one that enqueued them — the observable effect of
//!   work-stealing. It is always between `0.0` and `1.0`.
//! - **Occupancy** is each worker's measured busy time (sum of its span
//!   durations) over the trace's wall-clock span.
//!
//! The chrome `JSON` export ([`JobTrace::to_chrome_json`]) is gated behind the
//! `trace` feature and emits the Trace Event Format "complete" (`ph:"X"`)
//! events that `chrome://tracing`, Perfetto, and speedscope read.

use alloc::string::String;
use alloc::vec::Vec;
use core::cell::Cell;
use core::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use alloc::sync::Arc;

use crate::TaskPool;

thread_local! {
    /// Id of the span currently executing on this thread, used so a nested
    /// [`JobTrace::instrument`]/[`JobTrace::record`] can record its parent.
    static CURRENT_SPAN: Cell<Option<u64>> = const { Cell::new(None) };
}

/// One recorded job execution: who ran it, when, and under which parent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Span {
    /// Unique id of this span within its [`JobTrace`].
    pub id: u64,
    /// Id of the enclosing span (the job that spawned this one), if any.
    pub parent: Option<u64>,
    /// Human-readable span name.
    pub name: String,
    /// Executing worker index, or `None` if it ran on an external/main thread
    /// (which also helps drive the pool during `wait`).
    pub worker: Option<usize>,
    /// Start time in nanoseconds since the trace origin.
    pub start_nanos: u64,
    /// End time in nanoseconds since the trace origin.
    pub end_nanos: u64,
}

impl Span {
    /// Duration of the span in nanoseconds.
    #[must_use]
    pub fn duration_nanos(&self) -> u64 {
        self.end_nanos.saturating_sub(self.start_nanos)
    }
}

/// Shared trace state behind the [`JobTrace`] handle.
struct TraceInner {
    /// Pool whose workers this trace observes (used to resolve the executing
    /// worker index at span time).
    pool: TaskPool,
    /// Reference instant; all span timestamps are relative to it.
    origin: Instant,
    /// Recorded spans.
    spans: Mutex<Vec<Span>>,
    /// Monotonic span-id source.
    next_id: AtomicU64,
    /// Number of worker buckets. The external/main thread uses bucket
    /// `worker_count` (the extra trailing slot).
    worker_count: usize,
    /// Jobs executed per bucket.
    executed: Vec<AtomicU64>,
    /// Jobs that migrated (ran on a different worker than enqueued) per bucket.
    migrated: Vec<AtomicU64>,
    /// Busy nanoseconds accumulated per bucket.
    busy_nanos: Vec<AtomicU64>,
}

/// A cheap-to-clone handle to a job trace (design §16). Clones share the same
/// underlying span store and counters.
#[derive(Clone)]
pub struct JobTrace {
    inner: Arc<TraceInner>,
}

impl JobTrace {
    /// Create a trace that observes `pool`'s workers.
    #[must_use]
    pub fn new(pool: &TaskPool) -> Self {
        let worker_count = pool.worker_count();
        let buckets = worker_count + 1;
        let executed = (0..buckets).map(|_| AtomicU64::new(0)).collect();
        let migrated = (0..buckets).map(|_| AtomicU64::new(0)).collect();
        let busy_nanos = (0..buckets).map(|_| AtomicU64::new(0)).collect();
        Self {
            inner: Arc::new(TraceInner {
                pool: pool.clone(),
                origin: Instant::now(),
                spans: Mutex::new(Vec::new()),
                next_id: AtomicU64::new(0),
                worker_count,
                executed,
                migrated,
                busy_nanos,
            }),
        }
    }

    /// Map an optional worker index to its counter bucket (external threads map
    /// to the trailing bucket).
    fn bucket(&self, worker: Option<usize>) -> usize {
        match worker {
            Some(index) if index < self.inner.worker_count => index,
            _ => self.inner.worker_count,
        }
    }

    /// Nanoseconds since the trace origin, saturated into a `u64`.
    fn now_nanos(&self) -> u64 {
        u64::try_from(self.inner.origin.elapsed().as_nanos()).unwrap_or(u64::MAX)
    }

    /// Wrap `body` so that, when it runs, it is timed and recorded as a span.
    ///
    /// Call this on the thread that *enqueues* the job (e.g. inside a
    /// [`scope`](crate::TaskPool::scope) body); the returned closure is what you
    /// hand to [`crate::Scope::spawn`]. The enqueuing worker and parent span are
    /// captured now; the executing worker, start, and end are captured when the
    /// returned closure runs. If the two workers differ, the job counts as a
    /// migration (steal).
    pub fn instrument<F, R>(&self, name: impl Into<String>, body: F) -> impl FnOnce() -> R
    where
        F: FnOnce() -> R,
    {
        let name = name.into();
        let id = self.inner.next_id.fetch_add(1, Ordering::Relaxed);
        let parent = CURRENT_SPAN.with(Cell::get);
        let origin_worker = self.inner.pool.current_worker_index();
        let trace = self.clone();
        move || {
            let exec_worker = trace.inner.pool.current_worker_index();
            let start = trace.now_nanos();
            let previous = CURRENT_SPAN.with(|c| c.replace(Some(id)));
            let result = body();
            CURRENT_SPAN.with(|c| c.set(previous));
            let end = trace.now_nanos();
            trace.commit(Span {
                id,
                parent,
                name,
                worker: exec_worker,
                start_nanos: start,
                end_nanos: end,
            });
            let bucket = trace.bucket(exec_worker);
            trace.inner.executed[bucket].fetch_add(1, Ordering::Relaxed);
            trace.inner.busy_nanos[bucket].fetch_add(end.saturating_sub(start), Ordering::Relaxed);
            if exec_worker != origin_worker {
                trace.inner.migrated[bucket].fetch_add(1, Ordering::Relaxed);
            }
            result
        }
    }

    /// Record `body` as a span, running it inline on the calling thread now.
    /// Equivalent to building an [`JobTrace::instrument`] wrapper and invoking
    /// it immediately (so the enqueuing and executing worker are the same, and
    /// no migration is counted).
    pub fn record<F, R>(&self, name: impl Into<String>, body: F) -> R
    where
        F: FnOnce() -> R,
    {
        self.instrument(name, body)()
    }

    /// Push a finished span into the store.
    fn commit(&self, span: Span) {
        self.inner.spans.lock().unwrap().push(span);
    }

    /// Number of recorded spans.
    #[must_use]
    pub fn span_count(&self) -> usize {
        self.inner.spans.lock().unwrap().len()
    }

    /// Snapshot of the recorded spans, ordered by completion.
    #[must_use]
    pub fn spans(&self) -> Vec<Span> {
        self.inner.spans.lock().unwrap().clone()
    }

    /// Number of worker buckets (worker count plus one external/main bucket).
    #[must_use]
    pub fn bucket_count(&self) -> usize {
        self.inner.worker_count + 1
    }

    /// Total jobs executed (across all buckets).
    #[must_use]
    pub fn total_jobs(&self) -> u64 {
        self.inner
            .executed
            .iter()
            .map(|c| c.load(Ordering::Relaxed))
            .sum()
    }

    /// Jobs executed on `bucket` (use [`JobTrace::bucket_count`] for bounds; the
    /// last bucket is the external/main thread).
    #[must_use]
    pub fn jobs_on(&self, bucket: usize) -> u64 {
        self.inner
            .executed
            .get(bucket)
            .map_or(0, |c| c.load(Ordering::Relaxed))
    }

    /// Total jobs that migrated between workers (ran on a different worker than
    /// the one that enqueued them).
    #[must_use]
    pub fn migrated_jobs(&self) -> u64 {
        self.inner
            .migrated
            .iter()
            .map(|c| c.load(Ordering::Relaxed))
            .sum()
    }

    /// Fraction of executed jobs that migrated, in `0.0..=1.0`. Returns `0.0`
    /// when no jobs ran. This is the observable work-stealing / steal rate.
    #[must_use]
    pub fn steal_rate(&self) -> f64 {
        let total = self.total_jobs();
        if total == 0 {
            return 0.0;
        }
        // Precision note: counts stay far below 2^53 in practice, so the f64
        // conversion is exact for realistic traces.
        self.migrated_jobs() as f64 / total as f64
    }

    /// Busy nanoseconds accumulated on `bucket` (sum of its span durations).
    #[must_use]
    pub fn busy_nanos_on(&self, bucket: usize) -> u64 {
        self.inner
            .busy_nanos
            .get(bucket)
            .map_or(0, |c| c.load(Ordering::Relaxed))
    }

    /// Occupancy of `bucket`: its busy time over the trace's wall-clock span,
    /// clamped to `0.0..=1.0`. Returns `0.0` before any time has elapsed.
    #[must_use]
    pub fn occupancy(&self, bucket: usize) -> f64 {
        let wall = self.now_nanos();
        if wall == 0 {
            return 0.0;
        }
        let busy = self.busy_nanos_on(bucket);
        (busy as f64 / wall as f64).min(1.0)
    }

    /// Export the recorded spans as a chrome-tracing (Trace Event Format)
    /// `JSON` string of "complete" (`ph:"X"`) events. The output is accepted by
    /// `chrome://tracing`, Perfetto, and speedscope for flamegraph viewing.
    /// Timestamps and durations are in microseconds, as the format requires.
    #[cfg(feature = "trace")]
    #[must_use]
    pub fn to_chrome_json(&self) -> String {
        use core::fmt::Write as _;

        let spans = self.spans();
        let mut out = String::from("[");
        for (index, span) in spans.iter().enumerate() {
            if index > 0 {
                out.push(',');
            }
            let tid = span.worker.map_or(-1_i64, |w| w as i64);
            let ts = span.start_nanos as f64 / 1000.0;
            let dur = span.duration_nanos() as f64 / 1000.0;
            // Each field is written explicitly; `name` is JSON-escaped.
            out.push_str("{\"name\":\"");
            escape_json_into(&mut out, &span.name);
            out.push_str("\",\"ph\":\"X\",\"pid\":1,\"tid\":");
            let _ = write!(out, "{tid}");
            out.push_str(",\"ts\":");
            let _ = write!(out, "{ts:.3}");
            out.push_str(",\"dur\":");
            let _ = write!(out, "{dur:.3}");
            out.push_str(",\"args\":{\"id\":");
            let _ = write!(out, "{}", span.id);
            if let Some(parent) = span.parent {
                out.push_str(",\"parent\":");
                let _ = write!(out, "{parent}");
            }
            out.push_str("}}");
        }
        out.push(']');
        out
    }
}

/// Append `text` to `out` with the minimal `JSON` string escaping.
#[cfg(feature = "trace")]
fn escape_json_into(out: &mut String, text: &str) {
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                use core::fmt::Write as _;
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
}

impl TaskPool {
    /// Create a [`JobTrace`] that observes this pool's workers (design §16).
    #[must_use]
    pub fn new_job_trace(&self) -> JobTrace {
        JobTrace::new(self)
    }
}
