//! CPU↔GPU clock calibration and timeline alignment.
//!
//! GPU timestamp queries count in device *ticks*. To draw a GPU span on the
//! same timeline as CPU spans we need an affine map
//!
//! ```text
//! cpu_ns = slope · (tick − ref_tick) + ref_cpu_ns
//! ```
//!
//! that converts a raw tick into the CPU monotonic nanosecond base used by
//! [`prism_platform::now`] (and therefore by [`SpanRecord`](crate::trace::SpanRecord)).
//!
//! The nominal slope is the queue's `timestamp_period` (nanoseconds-per-tick),
//! but the GPU and CPU clocks drift relative to each other, so we periodically
//! take a *calibration sample*: a near-simultaneous `(cpu_now, gpu_now)` pair.
//! Over a sliding window of samples we fit the slope/offset by ordinary least
//! squares, optionally smoothing the slope with an exponential moving average
//! (EMA) to resist per-sample jitter. For a fixed sequence of samples the fit —
//! and therefore every projection — is fully deterministic.
//!
//! This module is pure arithmetic over `f64` + an `alloc` ring; it has no GPU
//! dependency. A backend supplies the `timestamp_period` and the calibration
//! samples through the ingestion API.

extern crate alloc;

use alloc::collections::VecDeque;

use super::scope::{GpuSpan, GpuTick};
use super::timeline::ProjectedGpuSpan;

/// Default number of calibration samples retained for the least-squares fit.
pub const DEFAULT_CALIBRATION_WINDOW: usize = 16;

/// A single near-simultaneous CPU/GPU clock reading used to anchor the fit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CalibrationSample {
    /// CPU monotonic time (nanoseconds) when the sample was taken.
    pub cpu_nanos: u64,
    /// GPU timestamp (ticks) read as close as possible to `cpu_nanos`.
    pub gpu_tick: GpuTick,
}

/// A fitted affine map from GPU ticks to CPU nanoseconds.
///
/// Projection is `cpu_ns = slope · (tick − ref_tick) + ref_cpu_ns`. The
/// reference point is the sample centroid, which both keeps the `f64`
/// arithmetic well-conditioned for huge tick values and makes the regression
/// line pass through the centre of the sample window.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AffineFit {
    /// Nanoseconds per tick.
    pub slope_ns_per_tick: f64,
    /// Reference tick (projection is expressed relative to this).
    pub ref_tick: f64,
    /// CPU nanoseconds corresponding to `ref_tick`.
    pub ref_cpu_nanos: f64,
}

impl AffineFit {
    /// Project a raw GPU tick onto the CPU nanosecond timeline (unrounded).
    #[must_use]
    pub fn project_tick_f64(&self, tick: GpuTick) -> f64 {
        self.slope_ns_per_tick * (tick.0 as f64 - self.ref_tick) + self.ref_cpu_nanos
    }

    /// Project a raw GPU tick onto the CPU nanosecond timeline, rounded and
    /// clamped to a non-negative `u64`.
    #[must_use]
    pub fn project_tick(&self, tick: GpuTick) -> u64 {
        let ns = self.project_tick_f64(tick).round();
        if ns <= 0.0 {
            0
        } else if ns >= u64::MAX as f64 {
            u64::MAX
        } else {
            ns as u64
        }
    }
}

/// Maintains CPU↔GPU calibration from a sliding window of samples.
///
/// Feed samples with [`add_sample`](Self::add_sample); read the current fit with
/// [`fit`](Self::fit) and project ticks/spans with
/// [`project_tick`](Self::project_tick) / [`project_span`](Self::project_span).
#[derive(Clone, Debug)]
pub struct GpuClockCalibration {
    timestamp_period_ns: f64,
    window: usize,
    samples: VecDeque<CalibrationSample>,
    fit: AffineFit,
    ema_alpha: Option<f64>,
    smoothed_slope: Option<f64>,
}

impl GpuClockCalibration {
    /// Create a calibration with the queue's `timestamp_period` (nanoseconds
    /// per tick) as the initial slope and a window of
    /// [`DEFAULT_CALIBRATION_WINDOW`] samples.
    ///
    /// `timestamp_period_ns` is clamped to a tiny positive value so the fit can
    /// never be degenerate before any sample arrives.
    #[must_use]
    pub fn new(timestamp_period_ns: f64) -> Self {
        Self::with_window(timestamp_period_ns, DEFAULT_CALIBRATION_WINDOW)
    }

    /// Create a calibration with an explicit sample-window size (clamped to at
    /// least 1).
    #[must_use]
    pub fn with_window(timestamp_period_ns: f64, window: usize) -> Self {
        let period = if timestamp_period_ns.is_finite() && timestamp_period_ns > 0.0 {
            timestamp_period_ns
        } else {
            f64::MIN_POSITIVE
        };
        Self {
            timestamp_period_ns: period,
            window: window.max(1),
            samples: VecDeque::new(),
            fit: AffineFit {
                slope_ns_per_tick: period,
                ref_tick: 0.0,
                ref_cpu_nanos: 0.0,
            },
            ema_alpha: None,
            smoothed_slope: None,
        }
    }

    /// Enable EMA smoothing of the fitted slope with factor `alpha` in `(0, 1]`
    /// (clamped). A smaller `alpha` reacts more slowly and rejects more jitter;
    /// `alpha == 1.0` disables smoothing (pure least squares). Builder style.
    #[must_use]
    pub fn with_ema(mut self, alpha: f64) -> Self {
        let a = if alpha.is_finite() {
            alpha.clamp(f64::MIN_POSITIVE, 1.0)
        } else {
            1.0
        };
        self.ema_alpha = Some(a);
        self
    }

    /// The queue's nominal nanoseconds-per-tick.
    #[must_use]
    pub fn timestamp_period_ns(&self) -> f64 {
        self.timestamp_period_ns
    }

    /// Number of calibration samples currently retained.
    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// The current affine fit.
    #[must_use]
    pub fn fit(&self) -> AffineFit {
        self.fit
    }

    /// Add a near-simultaneous `(cpu_nanos, gpu_tick)` calibration sample and
    /// refit. Overwrites the oldest sample when the window is full.
    pub fn add_sample(&mut self, cpu_nanos: u64, gpu_tick: GpuTick) {
        if self.samples.len() == self.window {
            self.samples.pop_front();
        }
        self.samples.push_back(CalibrationSample {
            cpu_nanos,
            gpu_tick,
        });
        self.refit();
    }

    /// Recompute [`fit`](Self::fit) from the retained samples.
    fn refit(&mut self) {
        let n = self.samples.len();
        if n == 0 {
            return;
        }

        // Centroid (also the numeric reference point).
        let mut mean_tick = 0.0;
        let mut mean_cpu = 0.0;
        for s in &self.samples {
            mean_tick += s.gpu_tick.0 as f64;
            mean_cpu += s.cpu_nanos as f64;
        }
        mean_tick /= n as f64;
        mean_cpu /= n as f64;

        // Ordinary least squares slope: Σ(x-x̄)(y-ȳ) / Σ(x-x̄)².
        let mut sxx = 0.0;
        let mut sxy = 0.0;
        for s in &self.samples {
            let dx = s.gpu_tick.0 as f64 - mean_tick;
            let dy = s.cpu_nanos as f64 - mean_cpu;
            sxx += dx * dx;
            sxy += dx * dy;
        }

        // With a single sample (or all ticks equal) the slope is undetermined;
        // fall back to the device's nominal timestamp_period.
        let lsq_slope = if sxx > 0.0 {
            sxy / sxx
        } else {
            self.timestamp_period_ns
        };

        let slope = match (self.ema_alpha, self.smoothed_slope) {
            (Some(alpha), Some(prev)) => alpha * lsq_slope + (1.0 - alpha) * prev,
            _ => lsq_slope,
        };
        self.smoothed_slope = Some(slope);

        self.fit = AffineFit {
            slope_ns_per_tick: slope,
            ref_tick: mean_tick,
            ref_cpu_nanos: mean_cpu,
        };
    }

    /// Project a raw GPU tick onto the CPU nanosecond timeline.
    #[must_use]
    pub fn project_tick(&self, tick: GpuTick) -> u64 {
        self.fit.project_tick(tick)
    }

    /// Project a resolved [`GpuSpan`] onto the CPU timeline.
    ///
    /// The start is the projected begin tick; the duration is derived from the
    /// span's tick delta scaled by the fitted slope (so it is independent of the
    /// offset and never negative). Deterministic for a given fit.
    #[must_use]
    pub fn project_span(&self, span: &GpuSpan) -> ProjectedGpuSpan {
        let cpu_start = self.fit.project_tick(span.begin_tick);
        let dur_f64 = (self.fit.slope_ns_per_tick * span.duration_ticks() as f64).round();
        let cpu_duration = if dur_f64 <= 0.0 {
            0
        } else if dur_f64 >= u64::MAX as f64 {
            u64::MAX
        } else {
            dur_f64 as u64
        };
        ProjectedGpuSpan {
            label: span.label.clone(),
            queue: span.queue,
            cpu_start_nanos: cpu_start,
            cpu_duration_nanos: cpu_duration,
            correlation: span.correlation,
            frame: span.frame,
            depth: span.depth,
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use alloc::string::String;

    use super::*;
    use crate::gpu::scope::{CorrelationId, GpuQueueId};

    #[test]
    fn exact_affine_recovered_from_clean_samples() {
        // cpu_ns = 2.0 * tick + 1000 (slope 2 ns/tick, offset 1000 ns).
        let mut cal = GpuClockCalibration::new(2.0);
        for tick in [0u64, 100, 200, 300, 400] {
            cal.add_sample(2 * tick + 1000, GpuTick(tick));
        }
        let fit = cal.fit();
        assert!((fit.slope_ns_per_tick - 2.0).abs() < 1e-9, "slope {}", fit.slope_ns_per_tick);
        // Projection is exact on and off the sample grid.
        assert_eq!(cal.project_tick(GpuTick(150)), 1300);
        assert_eq!(cal.project_tick(GpuTick(1000)), 3000);
    }

    #[test]
    fn single_sample_falls_back_to_timestamp_period() {
        let mut cal = GpuClockCalibration::new(3.0);
        cal.add_sample(500, GpuTick(10));
        let fit = cal.fit();
        assert_eq!(fit.slope_ns_per_tick, 3.0);
        // Line passes through the single sample with the nominal slope.
        assert_eq!(cal.project_tick(GpuTick(10)), 500);
        assert_eq!(cal.project_tick(GpuTick(20)), 530);
    }

    #[test]
    fn projection_is_deterministic() {
        let mut a = GpuClockCalibration::new(1.5);
        let mut b = GpuClockCalibration::new(1.5);
        for tick in [5u64, 25, 60, 90, 140, 210] {
            let cpu = (tick as f64 * 1.5) as u64 + 42;
            a.add_sample(cpu, GpuTick(tick));
            b.add_sample(cpu, GpuTick(tick));
        }
        for probe in [0u64, 7, 123, 999, 50_000] {
            assert_eq!(a.project_tick(GpuTick(probe)), b.project_tick(GpuTick(probe)));
        }
    }

    #[test]
    fn ema_smoothing_damps_slope_jitter() {
        // Feed a jittery slope and confirm the smoothed slope swings less than
        // the raw least-squares slope would across the last update.
        let mut raw = GpuClockCalibration::with_window(2.0, 2);
        let mut ema = GpuClockCalibration::with_window(2.0, 2).with_ema(0.25);

        // Seed both with a steady 2 ns/tick pair of samples.
        for (cpu, tick) in [(0u64, 0u64), (200, 100)] {
            raw.add_sample(cpu, GpuTick(tick));
            ema.add_sample(cpu, GpuTick(tick));
        }
        let ema_before = ema.fit().slope_ns_per_tick;

        // Now a window that implies a much steeper slope (~6 ns/tick).
        raw.add_sample(800, GpuTick(200));
        ema.add_sample(800, GpuTick(200));

        let raw_slope = raw.fit().slope_ns_per_tick;
        let ema_slope = ema.fit().slope_ns_per_tick;
        assert!(raw_slope > 5.0, "raw slope should jump: {raw_slope}");
        // EMA moves toward the new slope but stays much closer to the prior.
        assert!(ema_slope < raw_slope, "ema {ema_slope} should lag raw {raw_slope}");
        assert!((ema_slope - ema_before).abs() < (raw_slope - ema_before).abs());
    }

    #[test]
    fn drifting_clock_tracked_by_sliding_window() {
        // GPU clock runs slightly fast then drifts; the sliding window keeps the
        // fit tracking the most recent slope.
        let mut cal = GpuClockCalibration::with_window(2.0, 3);
        // Early regime: slope 2.0.
        for tick in [0u64, 100, 200] {
            cal.add_sample(2 * tick, GpuTick(tick));
        }
        assert!((cal.fit().slope_ns_per_tick - 2.0).abs() < 1e-9);
        // Later regime: slope 4.0 (window rolls off the slope-2 samples).
        for tick in [300u64, 400, 500] {
            cal.add_sample(400 + 4 * (tick - 300), GpuTick(tick));
        }
        assert!((cal.fit().slope_ns_per_tick - 4.0).abs() < 1e-9, "{}", cal.fit().slope_ns_per_tick);
    }

    #[test]
    fn span_projection_scales_duration_by_slope() {
        let mut cal = GpuClockCalibration::new(2.0);
        for tick in [0u64, 100, 200, 300] {
            cal.add_sample(2 * tick + 10, GpuTick(tick));
        }
        let span = GpuSpan {
            label: String::from("ShadowPass"),
            queue: GpuQueueId::GRAPHICS,
            begin_tick: GpuTick(50),
            end_tick: GpuTick(90),
            correlation: Some(CorrelationId(1)),
            frame: 2,
            depth: 1,
        };
        let projected = cal.project_span(&span);
        assert_eq!(projected.cpu_start_nanos, 110); // 2*50 + 10
        assert_eq!(projected.cpu_duration_nanos, 80); // 2 * (90-50)
        assert_eq!(projected.label, "ShadowPass");
        assert_eq!(projected.correlation, Some(CorrelationId(1)));
        assert_eq!(projected.frame, 2);
        assert_eq!(projected.depth, 1);
    }
}
