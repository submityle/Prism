//! Backend-neutral real-time profiling surface (zones / frames / plots).
//!
//! M5 adds a live profiling seam distinct from the offline Chrome-trace path in
//! [`crate::trace`]. Where the timeline *buffers* completed spans for later
//! export, the profiler *streams* events to a live backend as they happen: a
//! frame-by-frame flamegraph, counters, and messages in a tool like Tracy, or a
//! custom remote panel.
//!
//! ## Shape
//! - [`Profiler`] is the backend trait. Every method has a no-op default so a
//!   backend implements only what it supports.
//! - A process-global backend slot is installed with [`set_profiler`] and read
//!   on the hot path through the free functions [`zone`], [`frame_mark`],
//!   [`plot`], [`message`], and [`gpu_zone`]. When no backend is installed (the
//!   default), these are a cheap `RwLock` read and return — the
//!   [`NoopProfiler`] is never even allocated.
//! - [`ProfiledZone`] is an `RAII` guard: construct it at the top of a scope and
//!   it emits a completed [`Zone`] (with measured duration and nesting depth)
//!   when it drops. The [`profiled_zone!`](crate::profiled_zone!) macro wraps the
//!   common case.
//!
//! ## Feeding from the existing timeline
//! The profiler reuses the timeline's dense per-thread id
//! ([`current_thread_id`](crate::trace::current_thread_id)) so zones and spans
//! share a thread axis. [`emit_span`] forwards an already-recorded
//! [`SpanRecord`](crate::trace::SpanRecord) to the live backend, letting the
//! buffered timeline double as a live feed without re-instrumenting code.
//!
//! ## Backends
//! The default build ships only [`NoopProfiler`]. The optional
//! [`tracy`](mod@tracy) backend (behind the `tracy` feature) is a self-contained
//! Tracy-compatible emitter — no external `tracy-client` dependency.

pub mod zone;

#[cfg(feature = "tracy")]
pub mod tracy;

extern crate alloc;

use alloc::sync::Arc;
use core::cell::Cell;
use std::sync::{OnceLock, RwLock};
use String;

use prism_platform::now;

use crate::model::Level;
use crate::trace::{current_thread_id, SpanRecord};

pub use zone::{FrameMark, GpuZone, PlotValue, Zone};

/// A live profiling backend.
///
/// All methods default to no-ops so a backend only implements the event kinds
/// it cares about. Implementations must be cheap and non-blocking: they run on
/// the instrumented hot path. They must also tolerate being called from any
/// thread (the trait requires [`Send`] + [`Sync`]).
pub trait Profiler: Send + Sync {
    /// Record one completed [`Zone`] (scoped span).
    fn zone(&self, zone: &Zone) {
        let _ = zone;
    }

    /// Record a frame boundary at `timestamp_nanos`.
    fn frame_mark(&self, mark: &FrameMark, timestamp_nanos: u64) {
        let _ = (mark, timestamp_nanos);
    }

    /// Record a plot/counter sample for the series `name` at `timestamp_nanos`.
    fn plot(&self, name: &str, value: PlotValue, timestamp_nanos: u64) {
        let _ = (name, value, timestamp_nanos);
    }

    /// Record a free-form message at `level`.
    fn message(&self, level: Level, text: &str, timestamp_nanos: u64) {
        let _ = (level, text, timestamp_nanos);
    }

    /// Record one resolved [`GpuZone`].
    fn gpu_zone(&self, zone: &GpuZone) {
        let _ = zone;
    }
}

/// A backend that discards every event. Installed implicitly when no backend is
/// set; also useful as an explicit "profiling off" marker in tests.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoopProfiler;

impl Profiler for NoopProfiler {}

type ProfilerSlot = RwLock<Option<Arc<dyn Profiler>>>;

fn slot() -> &'static ProfilerSlot {
    static SLOT: OnceLock<ProfilerSlot> = OnceLock::new();
    SLOT.get_or_init(|| RwLock::new(None))
}

/// Install `profiler` as the process-global live backend, replacing any
/// previous one.
pub fn set_profiler(profiler: Arc<dyn Profiler>) {
    *slot().write().unwrap() = Some(profiler);
}

/// Remove the global backend (events are then dropped by [`NoopProfiler`]
/// semantics).
pub fn clear_profiler() {
    *slot().write().unwrap() = None;
}

/// Whether a (non-no-op) backend is currently installed.
pub fn is_active() -> bool {
    slot().read().unwrap().is_some()
}

fn with_backend<R>(f: impl FnOnce(&dyn Profiler) -> R) -> Option<R> {
    let guard = slot().read().unwrap();
    guard.as_ref().map(|p| f(p.as_ref()))
}

/// Emit a completed [`Zone`] to the installed backend, if any.
pub fn zone(zone: &Zone) {
    with_backend(|p| p.zone(zone));
}

/// Emit a frame boundary to the installed backend, if any.
pub fn frame_mark(mark: &FrameMark) {
    let ts = now().0;
    with_backend(|p| p.frame_mark(mark, ts));
}

/// Emit a plot/counter sample to the installed backend, if any.
pub fn plot(name: &str, value: PlotValue) {
    let ts = now().0;
    with_backend(|p| p.plot(name, value, ts));
}

/// Emit a free-form message to the installed backend, if any.
pub fn message(level: Level, text: &str) {
    let ts = now().0;
    with_backend(|p| p.message(level, text, ts));
}

/// Emit a resolved [`GpuZone`] to the installed backend, if any.
pub fn gpu_zone(zone: &GpuZone) {
    with_backend(|p| p.gpu_zone(zone));
}

/// Forward an already-recorded timeline [`SpanRecord`] to the live backend as a
/// [`Zone`], letting the buffered timeline double as a live feed.
pub fn emit_span(span: &SpanRecord) {
    with_backend(|p| p.zone(&Zone::from(span)));
}

thread_local! {
    static DEPTH: Cell<u32> = const { Cell::new(0) };
}

/// An `RAII` profiling guard: emits a completed [`Zone`] to the live backend
/// when it drops.
///
/// Construction captures the current nesting depth and thread id and reads the
/// clock last (so guard setup is excluded from the measured interval); drop
/// measures the elapsed time and emits the zone. Depth tracking is independent
/// of the timeline's [`Scope`](crate::span::Scope) so the two can be used
/// together or separately.
#[derive(Debug)]
pub struct ProfiledZone {
    name: String,
    category: Option<String>,
    thread_id: u64,
    depth: u32,
    start: prism_platform::MonotonicNanos,
}

impl ProfiledZone {
    /// Open a profiling zone named `name` on the calling thread.
    pub fn new(name: impl Into<String>) -> Self {
        let depth = DEPTH.with(|d| {
            let cur = d.get();
            d.set(cur + 1);
            cur
        });
        let thread_id = current_thread_id();
        Self {
            name: name.into(),
            category: None,
            thread_id,
            depth,
            // Read the clock last so setup is excluded from the interval.
            start: now(),
        }
    }

    /// Attach a category/track label (builder style).
    pub fn with_category(mut self, category: impl Into<String>) -> Self {
        self.category = Some(category.into());
        self
    }

    /// This zone's nesting depth at entry (0 is top level).
    pub fn depth(&self) -> u32 {
        self.depth
    }
}

impl Drop for ProfiledZone {
    fn drop(&mut self) {
        let duration_nanos = now().saturating_since(self.start);
        let z = Zone {
            name: core::mem::take(&mut self.name),
            category: self.category.take(),
            thread_id: self.thread_id,
            start_nanos: self.start.0,
            duration_nanos,
            depth: self.depth,
        };
        DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
        zone(&z);
    }
}

/// Open a [`ProfiledZone`] bound to a hidden local guard that closes at the end
/// of the enclosing block.
///
/// Accepts an optional category: `profiled_zone!(name)` or
/// `profiled_zone!(name, category)`.
///
/// # Examples
///
/// ```
/// use prism_diagnostic::profiled_zone;
///
/// {
///     profiled_zone!("physics_step", "physics");
///     // ... timed work ...
/// }
/// ```
#[macro_export]
macro_rules! profiled_zone {
    ($name:expr) => {
        let _prism_profiled_zone = $crate::profiler::ProfiledZone::new($name);
    };
    ($name:expr, $category:expr) => {
        let _prism_profiled_zone =
            $crate::profiler::ProfiledZone::new($name).with_category($category);
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use std::sync::Mutex;
    use String;

    #[derive(Default)]
    struct Recorder {
        zones: Mutex<Vec<Zone>>,
        frames: Mutex<Vec<(FrameMark, u64)>>,
        plots: Mutex<Vec<(String, PlotValue)>>,
        messages: Mutex<Vec<(Level, String)>>,
        gpu: Mutex<Vec<GpuZone>>,
    }

    impl Profiler for Recorder {
        fn zone(&self, zone: &Zone) {
            self.zones.lock().unwrap().push(zone.clone());
        }
        fn frame_mark(&self, mark: &FrameMark, ts: u64) {
            self.frames.lock().unwrap().push((mark.clone(), ts));
        }
        fn plot(&self, name: &str, value: PlotValue, _ts: u64) {
            self.plots.lock().unwrap().push((String::from(name), value));
        }
        fn message(&self, level: Level, text: &str, _ts: u64) {
            self.messages
                .lock()
                .unwrap()
                .push((level, String::from(text)));
        }
        fn gpu_zone(&self, zone: &GpuZone) {
            self.gpu.lock().unwrap().push(zone.clone());
        }
    }

    // These tests share the process-global profiler slot, so they must not run
    // concurrently; a module mutex serializes them.
    fn lock() -> std::sync::MutexGuard<'static, ()> {
        static GUARD: Mutex<()> = Mutex::new(());
        GUARD
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    #[test]
    fn noop_backend_is_inactive_feed() {
        let _g = lock();
        clear_profiler();
        assert!(!is_active());
        set_profiler(Arc::new(NoopProfiler));
        assert!(is_active());
        // Dispatching with the no-op backend must not panic.
        zone(&Zone {
            name: String::from("z"),
            category: None,
            thread_id: 1,
            start_nanos: 0,
            duration_nanos: 1,
            depth: 0,
        });
        clear_profiler();
    }

    #[test]
    fn profiled_zone_records_depth_and_duration() {
        let _g = lock();
        let rec = Arc::new(Recorder::default());
        set_profiler(rec.clone());
        {
            let _outer = ProfiledZone::new("outer");
            {
                let inner = ProfiledZone::new("inner").with_category("phys");
                assert_eq!(inner.depth(), 1);
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
        }
        clear_profiler();

        let zones = rec.zones.lock().unwrap();
        assert_eq!(zones.len(), 2);
        // Inner closes first.
        assert_eq!(zones[0].name, "inner");
        assert_eq!(zones[0].depth, 1);
        assert_eq!(zones[0].category.as_deref(), Some("phys"));
        assert_eq!(zones[1].name, "outer");
        assert_eq!(zones[1].depth, 0);
        assert!(zones[0].duration_nanos > 0);
    }

    #[test]
    fn frame_plot_message_gpu_dispatch() {
        let _g = lock();
        let rec = Arc::new(Recorder::default());
        set_profiler(rec.clone());

        frame_mark(&FrameMark::Continuous);
        frame_mark(&FrameMark::Named(String::from("render")));
        plot("drawcalls", PlotValue::U64(1234));
        message(Level::Warn, "budget exceeded");
        gpu_zone(&GpuZone {
            name: String::from("shadow_pass"),
            queue_id: 1,
            start_nanos: 10,
            duration_nanos: 20,
            correlation: Some(99),
        });
        clear_profiler();

        assert_eq!(rec.frames.lock().unwrap().len(), 2);
        assert_eq!(rec.plots.lock().unwrap()[0].0, "drawcalls");
        assert_eq!(rec.messages.lock().unwrap()[0].0, Level::Warn);
        assert_eq!(rec.gpu.lock().unwrap()[0].queue_id, 1);
    }

    #[test]
    fn emit_span_forwards_as_zone() {
        let _g = lock();
        let rec = Arc::new(Recorder::default());
        set_profiler(rec.clone());
        let span = SpanRecord {
            name: String::from("sys"),
            category: None,
            thread_id: 2,
            start_nanos: 5,
            duration_nanos: 7,
            depth: 0,
            args: Vec::new(),
        };
        emit_span(&span);
        clear_profiler();
        let zones = rec.zones.lock().unwrap();
        assert_eq!(zones.len(), 1);
        assert_eq!(zones[0].name, "sys");
        assert_eq!(zones[0].duration_nanos, 7);
    }
}
