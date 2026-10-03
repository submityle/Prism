//! Built-in frame statistics.
//!
//! [`FrameTimer`] tracks per-frame wall time and derived rates over a sliding
//! window, plus a set of generic named counters (drawcalls, triangles, ...)
//! that reset every frame. Drive it with a [`begin_frame`]/[`end_frame`] cycle
//! around each rendered frame, or feed synthetic deltas with
//! [`record_frame_delta_ms`] when the frame time comes from an external clock.
//!
//! [`begin_frame`]: FrameTimer::begin_frame
//! [`end_frame`]: FrameTimer::end_frame
//! [`record_frame_delta_ms`]: FrameTimer::record_frame_delta_ms

extern crate alloc;

use alloc::collections::{BTreeMap, VecDeque};
use alloc::string::String;
use alloc::vec::Vec;

use prism_platform::{now, MonotonicNanos};

/// Default sliding-window length (number of frames retained for statistics).
pub const DEFAULT_WINDOW: usize = 120;

/// Tracks frame timing and per-frame named counters.
///
/// The timer keeps the last `window` frame deltas (in milliseconds) and derives
/// min/avg/max frame time and FPS from them. Named counters accumulate during a
/// frame via [`count`](FrameTimer::count); at [`end_frame`](FrameTimer::end_frame)
/// the live counters become the "last frame" values and are reset for the next
/// [`begin_frame`](FrameTimer::begin_frame).
#[derive(Debug)]
pub struct FrameTimer {
    window: usize,
    deltas_ms: VecDeque<f64>,
    frame_index: u64,
    begin_at: Option<MonotonicNanos>,
    live: BTreeMap<String, u64>,
    last: BTreeMap<String, u64>,
}

impl Default for FrameTimer {
    fn default() -> Self {
        Self::with_window(DEFAULT_WINDOW)
    }
}

impl FrameTimer {
    /// Create a timer with the [`DEFAULT_WINDOW`] sliding window.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a timer whose sliding window retains `window` frames (clamped to
    /// at least 1).
    pub fn with_window(window: usize) -> Self {
        let window = window.max(1);
        Self {
            window,
            deltas_ms: VecDeque::with_capacity(window),
            frame_index: 0,
            begin_at: None,
            live: BTreeMap::new(),
            last: BTreeMap::new(),
        }
    }

    /// Begin a frame: capture the start timestamp and clear the live counters.
    pub fn begin_frame(&mut self) {
        self.begin_at = Some(now());
        self.live.clear();
    }

    /// End a frame: measure the elapsed time since [`begin_frame`], record it,
    /// and roll the live counters into the "last frame" snapshot. Returns the
    /// measured frame delta in milliseconds (`0.0` if `begin_frame` was not
    /// called).
    ///
    /// [`begin_frame`]: FrameTimer::begin_frame
    pub fn end_frame(&mut self) -> f64 {
        let delta_ms = match self.begin_at.take() {
            Some(start) => now().saturating_since(start) as f64 / 1_000_000.0,
            None => 0.0,
        };
        self.record_frame(delta_ms);
        delta_ms
    }

    /// Record a frame with an explicit delta in milliseconds.
    ///
    /// Use this when the frame time is sourced externally (or in tests). The
    /// live counters are rolled into the "last frame" snapshot exactly as in
    /// [`end_frame`](FrameTimer::end_frame).
    pub fn record_frame_delta_ms(&mut self, delta_ms: f64) {
        self.begin_at = None;
        self.record_frame(delta_ms);
    }

    fn record_frame(&mut self, delta_ms: f64) {
        if self.deltas_ms.len() == self.window {
            self.deltas_ms.pop_front();
        }
        self.deltas_ms.push_back(delta_ms);
        self.frame_index += 1;
        self.last = core::mem::take(&mut self.live);
    }

    /// Add `amount` to the named per-frame counter for the current frame.
    pub fn count(&mut self, name: &str, amount: u64) {
        *self.live.entry(name.into()).or_insert(0) += amount;
    }

    /// Increment the named per-frame counter by one.
    pub fn incr(&mut self, name: &str) {
        self.count(name, 1);
    }

    /// The named counter's accumulated value for the in-progress frame.
    pub fn live_count(&self, name: &str) -> u64 {
        self.live.get(name).copied().unwrap_or(0)
    }

    /// The named counter's value from the most recently completed frame.
    pub fn last_frame_count(&self, name: &str) -> u64 {
        self.last.get(name).copied().unwrap_or(0)
    }

    /// Names of counters recorded in the most recently completed frame.
    pub fn last_frame_counter_names(&self) -> Vec<String> {
        self.last.keys().cloned().collect()
    }

    /// Total number of frames recorded so far.
    pub fn frame_index(&self) -> u64 {
        self.frame_index
    }

    /// Take an immutable snapshot of the current frame statistics.
    pub fn stats(&self) -> FrameStatsSnapshot {
        let sample_count = self.deltas_ms.len();
        let last_delta_ms = self.deltas_ms.back().copied().unwrap_or(0.0);
        let (min_ms, max_ms, sum_ms) = self.deltas_ms.iter().fold(
            (f64::INFINITY, f64::NEG_INFINITY, 0.0),
            |(mn, mx, sum), &d| (mn.min(d), mx.max(d), sum + d),
        );
        let (min_ms, max_ms) = if sample_count == 0 {
            (0.0, 0.0)
        } else {
            (min_ms, max_ms)
        };
        let avg_ms = if sample_count == 0 {
            0.0
        } else {
            sum_ms / sample_count as f64
        };
        FrameStatsSnapshot {
            frame_index: self.frame_index,
            sample_count,
            last_delta_ms,
            min_ms,
            max_ms,
            avg_ms,
            fps: ms_to_fps(avg_ms),
            min_fps: ms_to_fps(max_ms),
            max_fps: ms_to_fps(min_ms),
            instant_fps: ms_to_fps(last_delta_ms),
        }
    }
}

/// Convert a frame time in milliseconds to frames per second (`0.0` for a
/// non-positive input).
fn ms_to_fps(ms: f64) -> f64 {
    if ms > 0.0 {
        1000.0 / ms
    } else {
        0.0
    }
}

/// An immutable snapshot of [`FrameTimer`] statistics.
///
/// FPS fields are derived from the frame-time fields: `fps` from `avg_ms`,
/// `min_fps` from `max_ms` (the slowest frame), `max_fps` from `min_ms` (the
/// fastest frame), and `instant_fps` from `last_delta_ms`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameStatsSnapshot {
    /// Total number of frames recorded so far.
    pub frame_index: u64,
    /// Number of frame samples currently in the sliding window.
    pub sample_count: usize,
    /// Most recent frame delta in milliseconds.
    pub last_delta_ms: f64,
    /// Fastest (smallest) frame time in the window, in milliseconds.
    pub min_ms: f64,
    /// Slowest (largest) frame time in the window, in milliseconds.
    pub max_ms: f64,
    /// Average frame time over the window, in milliseconds.
    pub avg_ms: f64,
    /// Average FPS, derived from `avg_ms`.
    pub fps: f64,
    /// Lowest FPS over the window, derived from `max_ms`.
    pub min_fps: f64,
    /// Highest FPS over the window, derived from `min_ms`.
    pub max_fps: f64,
    /// Instantaneous FPS, derived from `last_delta_ms`.
    pub instant_fps: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fps_min_avg_max_over_synthetic_deltas() {
        let mut timer = FrameTimer::new();
        for d in [10.0, 20.0, 30.0, 15.0, 25.0] {
            timer.record_frame_delta_ms(d);
        }
        let s = timer.stats();
        assert_eq!(s.frame_index, 5);
        assert_eq!(s.sample_count, 5);
        assert!((s.min_ms - 10.0).abs() < 1e-9);
        assert!((s.max_ms - 30.0).abs() < 1e-9);
        assert!((s.avg_ms - 20.0).abs() < 1e-9);
        assert!((s.fps - 50.0).abs() < 1e-9);
        assert!((s.max_fps - 100.0).abs() < 1e-9);
        assert!((s.last_delta_ms - 25.0).abs() < 1e-9);
    }

    #[test]
    fn sliding_window_drops_old_samples() {
        let mut timer = FrameTimer::with_window(3);
        for d in [100.0, 1.0, 2.0, 3.0] {
            timer.record_frame_delta_ms(d);
        }
        let s = timer.stats();
        assert_eq!(s.sample_count, 3);
        // The 100.0 sample has rolled out of the window.
        assert!((s.max_ms - 3.0).abs() < 1e-9);
        assert!((s.min_ms - 1.0).abs() < 1e-9);
        assert_eq!(s.frame_index, 4);
    }

    #[test]
    fn per_frame_counters_reset_across_begin_end() {
        let mut timer = FrameTimer::new();

        timer.begin_frame();
        timer.count("drawcalls", 10);
        timer.incr("drawcalls");
        timer.count("triangles", 5000);
        assert_eq!(timer.live_count("drawcalls"), 11);
        timer.end_frame();
        assert_eq!(timer.last_frame_count("drawcalls"), 11);
        assert_eq!(timer.last_frame_count("triangles"), 5000);

        timer.begin_frame();
        // Counters start fresh for the new frame.
        assert_eq!(timer.live_count("drawcalls"), 0);
        timer.count("drawcalls", 3);
        timer.end_frame();
        assert_eq!(timer.last_frame_count("drawcalls"), 3);
        assert_eq!(timer.last_frame_count("triangles"), 0);
    }
}
