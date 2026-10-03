//! Per-thread / per-system load aggregation (M3 load visualization).
//!
//! The Chrome export already yields a flame graph; this module produces the
//! complementary *load summary* a "负载可视化" view needs: how busy each worker
//! thread was over a window, and how that busy time splits across systems.
//!
//! Aggregation is a pure function of recorded [`SpanRecord`]s and performs no
//! timing itself. To avoid double-counting nested work, only **top-level**
//! spans (`depth == 0`) contribute: a top-level span's wall time already
//! includes the time of its children. As a result the invariant
//! `Σ systems.total_nanos == Σ threads.busy_nanos` holds, so the per-system and
//! per-thread breakdowns are two consistent slices of the same busy time.
//!
//! The observation **window** spans from the earliest top-level span start to
//! the latest top-level span end seen across all threads. Utilization is busy
//! time divided by the available thread-time (`window × thread count`).

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use crate::trace::ring::{registered_threads, SpanRecord};

/// Aggregated timing for one system (grouped by span name) within a window.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SystemLoad {
    /// System name (the top-level span `name`).
    pub name: String,
    /// Summed wall time across all invocations in the window, in nanoseconds.
    pub total_nanos: u64,
    /// Number of invocations (top-level spans) in the window.
    pub call_count: u64,
    /// Longest single invocation in the window, in nanoseconds.
    pub max_nanos: u64,
}

impl SystemLoad {
    /// Mean invocation duration in nanoseconds (`0` when never called).
    pub fn mean_nanos(&self) -> u64 {
        if self.call_count == 0 {
            0
        } else {
            self.total_nanos / self.call_count
        }
    }
}

/// Utilization of one thread within a window.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ThreadLoad {
    /// Prism-assigned thread id.
    pub thread_id: u64,
    /// Summed top-level span wall time on this thread, in nanoseconds.
    pub busy_nanos: u64,
    /// Number of top-level spans attributed to this thread in the window.
    pub span_count: u64,
}

/// A consistent per-thread and per-system load breakdown over an observation
/// window.
///
/// Build one from the live rings with [`LoadProfile::capture`], or from
/// explicit per-thread span snapshots with [`LoadProfile::from_thread_spans`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoadProfile {
    /// Earliest top-level span start observed, in nanoseconds (`0` if empty).
    pub window_start_nanos: u64,
    /// Latest top-level span end observed, in nanoseconds (`0` if empty).
    pub window_end_nanos: u64,
    /// Per-thread utilization, ordered by `thread_id`.
    pub threads: Vec<ThreadLoad>,
    /// Per-system totals, ordered by descending `total_nanos` then name.
    pub systems: Vec<SystemLoad>,
}

impl LoadProfile {
    /// Capture a profile from every registered thread's currently retained
    /// spans.
    pub fn capture() -> Self {
        let threads = registered_threads();
        let mut per_thread: Vec<(u64, Vec<SpanRecord>)> = Vec::with_capacity(threads.len());
        for trace in &threads {
            let spans = match trace.buffer.lock() {
                Ok(buf) => buf.snapshot(),
                Err(_) => Vec::new(),
            };
            per_thread.push((trace.thread_id, spans));
        }
        Self::from_thread_spans(&per_thread)
    }

    /// Aggregate a profile from explicit per-thread span snapshots.
    ///
    /// Each entry is `(thread_id, spans)`. Only `depth == 0` spans contribute.
    pub fn from_thread_spans(threads: &[(u64, Vec<SpanRecord>)]) -> Self {
        let mut thread_loads: Vec<ThreadLoad> = Vec::with_capacity(threads.len());
        let mut systems: BTreeMap<String, SystemLoad> = BTreeMap::new();
        let mut window_start: Option<u64> = None;
        let mut window_end: Option<u64> = None;

        for (thread_id, spans) in threads {
            let mut busy_nanos: u64 = 0;
            let mut span_count: u64 = 0;
            for span in spans {
                if span.depth != 0 {
                    continue;
                }
                span_count += 1;
                busy_nanos = busy_nanos.saturating_add(span.duration_nanos);

                let start = span.start_nanos;
                let end = span.start_nanos.saturating_add(span.duration_nanos);
                window_start = Some(window_start.map_or(start, |w| w.min(start)));
                window_end = Some(window_end.map_or(end, |w| w.max(end)));

                let entry = systems.entry(span.name.clone()).or_insert_with(|| SystemLoad {
                    name: span.name.clone(),
                    total_nanos: 0,
                    call_count: 0,
                    max_nanos: 0,
                });
                entry.total_nanos = entry.total_nanos.saturating_add(span.duration_nanos);
                entry.call_count += 1;
                entry.max_nanos = entry.max_nanos.max(span.duration_nanos);
            }
            thread_loads.push(ThreadLoad {
                thread_id: *thread_id,
                busy_nanos,
                span_count,
            });
        }

        thread_loads.sort_by_key(|t| t.thread_id);

        let mut systems: Vec<SystemLoad> = systems.into_values().collect();
        systems.sort_by(|a, b| {
            b.total_nanos
                .cmp(&a.total_nanos)
                .then_with(|| a.name.cmp(&b.name))
        });

        Self {
            window_start_nanos: window_start.unwrap_or(0),
            window_end_nanos: window_end.unwrap_or(0),
            threads: thread_loads,
            systems,
        }
    }

    /// Window length in nanoseconds (`0` when empty).
    pub fn window_nanos(&self) -> u64 {
        self.window_end_nanos.saturating_sub(self.window_start_nanos)
    }

    /// Number of threads represented in the profile.
    pub fn thread_count(&self) -> usize {
        self.threads.len()
    }

    /// Total busy time summed across every thread, in nanoseconds.
    pub fn total_busy_nanos(&self) -> u64 {
        self.threads
            .iter()
            .fold(0u64, |acc, t| acc.saturating_add(t.busy_nanos))
    }

    /// Overall utilization in `[0, 1]`: total busy time over the available
    /// thread-time (`window × thread count`). Returns `0.0` when the window or
    /// thread count is zero.
    pub fn utilization(&self) -> f64 {
        let available = self.window_nanos().saturating_mul(self.thread_count() as u64);
        if available == 0 {
            0.0
        } else {
            self.total_busy_nanos() as f64 / available as f64
        }
    }

    /// Utilization in `[0, 1]` of a single thread over the window, or `None`
    /// when the thread is absent. Returns `Some(0.0)` for a zero-length window.
    pub fn thread_utilization(&self, thread_id: u64) -> Option<f64> {
        let thread = self.threads.iter().find(|t| t.thread_id == thread_id)?;
        let window = self.window_nanos();
        Some(if window == 0 {
            0.0
        } else {
            thread.busy_nanos as f64 / window as f64
        })
    }

    /// The busiest system by total time, if any spans were recorded.
    pub fn hottest_system(&self) -> Option<&SystemLoad> {
        self.systems.first()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span(name: &str, start: u64, dur: u64, depth: u32) -> SpanRecord {
        SpanRecord {
            name: String::from(name),
            category: None,
            thread_id: 0,
            start_nanos: start,
            duration_nanos: dur,
            depth,
            args: Vec::new(),
        }
    }

    #[test]
    fn aggregates_per_thread_and_per_system_consistently() {
        let threads = alloc::vec![
            (
                1u64,
                alloc::vec![
                    span("physics", 0, 100, 0),
                    span("narrow_phase", 10, 40, 1), // nested: excluded
                    span("render", 100, 200, 0),
                ],
            ),
            (2u64, alloc::vec![span("physics", 50, 150, 0)]),
        ];
        let profile = LoadProfile::from_thread_spans(&threads);

        // Per-thread busy = sum of depth-0 durations only.
        let t1 = &profile.threads[0];
        assert_eq!(t1.thread_id, 1);
        assert_eq!(t1.busy_nanos, 300); // 100 + 200 (nested 40 excluded)
        assert_eq!(t1.span_count, 2);
        let t2 = &profile.threads[1];
        assert_eq!(t2.busy_nanos, 150);
        assert_eq!(t2.span_count, 1);

        // Window spans earliest start (0) to latest end (render ends at 300).
        assert_eq!(profile.window_start_nanos, 0);
        assert_eq!(profile.window_end_nanos, 300);
        assert_eq!(profile.window_nanos(), 300);

        // Invariant: Σ systems total == Σ threads busy.
        let sys_total: u64 = profile.systems.iter().map(|s| s.total_nanos).sum();
        assert_eq!(sys_total, profile.total_busy_nanos());
        assert_eq!(profile.total_busy_nanos(), 450);

        // physics = 100 + 150 across two threads, called twice.
        let physics = profile
            .systems
            .iter()
            .find(|s| s.name == "physics")
            .unwrap();
        assert_eq!(physics.total_nanos, 250);
        assert_eq!(physics.call_count, 2);
        assert_eq!(physics.max_nanos, 150);
        assert_eq!(physics.mean_nanos(), 125);

        // Systems sorted by descending total: physics(250) > render(200).
        assert_eq!(profile.systems[0].name, "physics");
        assert_eq!(profile.hottest_system().unwrap().name, "physics");
    }

    #[test]
    fn utilization_math() {
        // Two threads, window 1000ns. Thread 1 busy 500, thread 2 busy 250.
        let threads = alloc::vec![
            (1u64, alloc::vec![span("a", 0, 500, 0)]),
            (2u64, alloc::vec![span("b", 0, 250, 0)]),
        ];
        // Extend the window to 1000 with a short span ending at 1000.
        let mut threads = threads;
        threads[0].1.push(span("tail", 999, 1, 0));
        let profile = LoadProfile::from_thread_spans(&threads);

        assert_eq!(profile.window_nanos(), 1000);
        // Thread 1 busy = 500 + 1 = 501.
        assert_eq!(profile.thread_utilization(1).unwrap(), 501.0 / 1000.0);
        assert_eq!(profile.thread_utilization(2).unwrap(), 250.0 / 1000.0);
        assert_eq!(profile.thread_utilization(99), None);

        // Overall = (501 + 250) / (1000 * 2 threads).
        let expected = 751.0 / 2000.0;
        assert!((profile.utilization() - expected).abs() < 1e-12);
    }

    #[test]
    fn empty_profile_is_zeroed() {
        let profile = LoadProfile::from_thread_spans(&[]);
        assert_eq!(profile.window_nanos(), 0);
        assert_eq!(profile.total_busy_nanos(), 0);
        assert_eq!(profile.utilization(), 0.0);
        assert!(profile.hottest_system().is_none());
    }
}
