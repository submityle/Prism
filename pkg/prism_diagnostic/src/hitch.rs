//! Hitch (卡顿) detection: frame-time monitoring that flags frames exceeding a
//! fixed budget or a rolling-percentile spike threshold (design §17, M6).
//!
//! An AAA frame time is a contract (60/120Hz), and the hardest problems are
//! *intermittent* stalls: a frame that is suddenly far slower than its
//! neighbours. [`HitchDetector`] watches a stream of per-frame durations and
//! emits a [`HitchEvent`] when a frame is either over an absolute budget or a
//! configurable multiple of a rolling percentile of recent frames (the
//! "baseline"). It performs no capture itself; it produces the signal a capture
//! or sampler can act on.
//!
//! The math is intentionally explicit and uses only `core` floating-point
//! arithmetic (no `std`-only rounding intrinsics) so the detector works in a
//! `no_std` + `alloc` build.

extern crate alloc;

use alloc::collections::VecDeque;
use alloc::vec::Vec;

/// Configuration for a [`HitchDetector`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HitchConfig {
    /// Absolute per-frame budget in nanoseconds. A frame longer than this is a
    /// hitch regardless of history. `0` disables the absolute-budget check.
    pub budget_nanos: u64,
    /// Number of recent frames retained for the rolling baseline.
    pub window: usize,
    /// Baseline percentile in `[0, 1]` (e.g. `0.95` for p95), computed over the
    /// retained window via the nearest-rank method.
    pub percentile: f64,
    /// Spike multiplier: a frame longer than `spike_ratio * baseline` is a
    /// hitch. `0.0` disables the baseline check.
    pub spike_ratio: f64,
    /// Minimum samples required before the baseline check activates. Until the
    /// window holds at least this many frames, only the absolute budget applies.
    pub min_samples: usize,
}

impl HitchConfig {
    /// A 60Hz preset: ~16.67ms budget, 120-frame window, p95 baseline, 2x
    /// spike ratio, warming up after 30 frames.
    pub fn fps_60() -> Self {
        Self {
            budget_nanos: 16_667_000,
            window: 120,
            percentile: 0.95,
            spike_ratio: 2.0,
            min_samples: 30,
        }
    }

    /// A 120Hz preset: ~8.33ms budget, otherwise identical to [`Self::fps_60`].
    pub fn fps_120() -> Self {
        Self {
            budget_nanos: 8_333_000,
            window: 120,
            percentile: 0.95,
            spike_ratio: 2.0,
            min_samples: 30,
        }
    }
}

impl Default for HitchConfig {
    fn default() -> Self {
        Self::fps_60()
    }
}

/// A detected hitch: one frame flagged as anomalously slow.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HitchEvent {
    /// Zero-based index of the offending frame in the observed stream.
    pub frame_index: u64,
    /// Observed duration of the offending frame, in nanoseconds.
    pub duration_nanos: u64,
    /// The absolute budget in effect, in nanoseconds (`0` if disabled).
    pub budget_nanos: u64,
    /// The rolling baseline (percentile of recent frames) at detection time, in
    /// nanoseconds (`0` if the baseline was not yet active).
    pub baseline_nanos: u64,
    /// Whether the frame exceeded the absolute budget.
    pub over_budget: bool,
    /// Whether the frame exceeded `spike_ratio * baseline`.
    pub over_baseline: bool,
    /// Monotonic timestamp (nanoseconds) sampled when the frame was recorded.
    pub timestamp_nanos: u64,
}

/// A rolling frame-time monitor that flags hitches.
#[derive(Clone, Debug)]
pub struct HitchDetector {
    config: HitchConfig,
    samples: VecDeque<u64>,
    frame_index: u64,
    hitch_count: u64,
    worst_nanos: u64,
}

impl HitchDetector {
    /// Create a detector with the given configuration.
    ///
    /// The window is clamped to at least 1 so the retained history is never
    /// empty.
    pub fn new(config: HitchConfig) -> Self {
        let window = config.window.max(1);
        Self {
            config: HitchConfig { window, ..config },
            samples: VecDeque::with_capacity(window),
            frame_index: 0,
            hitch_count: 0,
            worst_nanos: 0,
        }
    }

    /// Record a frame duration (nanoseconds) with an explicit capture
    /// timestamp, returning a [`HitchEvent`] when the frame is a hitch.
    ///
    /// The baseline is computed over the frames observed *before* this one, so a
    /// slow frame cannot mask itself by inflating its own baseline. The sample
    /// is then folded into the rolling window.
    pub fn record_frame_at(&mut self, duration_nanos: u64, timestamp_nanos: u64) -> Option<HitchEvent> {
        let baseline = self.baseline_nanos();

        let over_budget = self.config.budget_nanos > 0 && duration_nanos > self.config.budget_nanos;
        let over_baseline = baseline > 0
            && self.config.spike_ratio > 0.0
            && (duration_nanos as f64) > self.config.spike_ratio * (baseline as f64);

        self.push_sample(duration_nanos);
        let frame_index = self.frame_index;
        self.frame_index += 1;
        self.worst_nanos = self.worst_nanos.max(duration_nanos);

        if over_budget || over_baseline {
            self.hitch_count += 1;
            Some(HitchEvent {
                frame_index,
                duration_nanos,
                budget_nanos: self.config.budget_nanos,
                baseline_nanos: baseline,
                over_budget,
                over_baseline,
                timestamp_nanos,
            })
        } else {
            None
        }
    }

    /// Record a frame duration, sampling the platform monotonic clock for the
    /// event timestamp. See [`Self::record_frame_at`].
    pub fn record_frame(&mut self, duration_nanos: u64) -> Option<HitchEvent> {
        self.record_frame_at(duration_nanos, prism_platform::now().0)
    }

    /// Current rolling baseline in nanoseconds: the configured percentile over
    /// the retained window, or `0` while the window holds fewer than
    /// `min_samples` frames.
    pub fn baseline_nanos(&self) -> u64 {
        if self.samples.len() < self.config.min_samples.max(1) {
            return 0;
        }
        percentile_nearest_rank(&self.samples, self.config.percentile)
    }

    /// Total number of hitches flagged so far.
    pub fn hitch_count(&self) -> u64 {
        self.hitch_count
    }

    /// Number of frames observed so far.
    pub fn frames_observed(&self) -> u64 {
        self.frame_index
    }

    /// Worst (longest) frame duration observed so far, in nanoseconds.
    pub fn worst_nanos(&self) -> u64 {
        self.worst_nanos
    }

    /// Borrow the active configuration.
    pub fn config(&self) -> &HitchConfig {
        &self.config
    }

    fn push_sample(&mut self, duration_nanos: u64) {
        if self.samples.len() == self.config.window {
            self.samples.pop_front();
        }
        self.samples.push_back(duration_nanos);
    }
}

/// Nearest-rank percentile of a sample window, computed with integer-friendly
/// `core` arithmetic (no `std` rounding intrinsics).
///
/// Returns `0` for an empty input. `p` is clamped to `[0, 1]`.
fn percentile_nearest_rank(samples: &VecDeque<u64>, p: f64) -> u64 {
    let n = samples.len();
    if n == 0 {
        return 0;
    }
    let mut sorted: Vec<u64> = samples.iter().copied().collect();
    sorted.sort_unstable();

    let p = p.clamp(0.0, 1.0);
    // Nearest-rank: rank = ceil(p * n), 1-based, clamped to [1, n]. Emulate
    // `ceil` without the std-only intrinsic.
    let product = p * (n as f64);
    let mut rank = product as usize;
    if (rank as f64) < product {
        rank += 1;
    }
    let rank = rank.clamp(1, n);
    sorted[rank - 1]
}
