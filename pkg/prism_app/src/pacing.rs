//! Frame pacing and frame limiting (design §13).
//!
//! The main loop can run as fast as the machine allows, but on real hardware
//! that is rarely what you want: an uncapped loop burns power and heat for no
//! visible benefit, and a naive `sleep(target - elapsed)` limiter *drifts* —
//! each frame's rounding error accumulates, so the achieved rate wanders away
//! from the target. This module provides the two policy pieces the design
//! calls for without any windowing dependency:
//!
//! - [`FrameLimit`] — the cap policy (unlimited / target FPS / explicit
//!   period), the thing a quality tier or a dedicated-server tickrate selects.
//! - [`FramePacer`] — a **drift-free** limiter that paces to a moving cadence
//!   (`deadline += period`, not `now + period`) so rounding error never
//!   accumulates, plus an anti-death-spiral clamp that resyncs the cadence
//!   when the loop falls too far behind (so a hitch does not trigger a burst
//!   of zero-length catch-up frames), plus rolling [`FrameStats`] over a
//!   bounded window for the diagnostics in design §16.
//!
//! # One-way relationship with `prism_time`
//!
//! Per design §25.1 the App drives time; it does not reimplement it. The pacer
//! reads [`prism_time::Instant::now`] for the monotonic source and sleeps the
//! calling thread — it owns the *policy* (how long to wait), while the clock
//! itself stays `prism_time`'s single source of truth.
//!
//! # Honestly deferred
//!
//! Two design §13 refinements are intentionally **not** implemented here, to
//! avoid pretending to a capability that needs lower layers:
//!
//! - **Present-timestamp alignment / VRR** aligns the sleep target to the
//!   display's predicted next scan-out (design §13, §25.1). That needs a
//!   present timestamp from `prism_window`/RHI, which does not exist yet, so
//!   the pacer paces to a *fixed cadence* only. When a present estimate lands,
//!   it becomes the deadline source without changing this API.
//! - **Low-latency pipeline depth** (shrinking render-ahead) is a
//!   runner/pipeline concern, not a sleep policy, and is wired where the
//!   pipeline lives (the `pipelined` feature); the pacer deliberately carries
//!   no do-nothing toggle for it.

use std::collections::VecDeque;

use prism_time::{Duration, Instant};

/// How aggressively the main loop caps its frame rate (design §13).
///
/// This is pure policy; [`FramePacer`] turns it into actual sleeps. `Off` is
/// the default so a loop is never silently throttled unless a caller opts in
/// (e.g. a quality tier for a mobile frame limiter, or a dedicated-server
/// tickrate).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FrameLimit {
    /// No software limiting: present as fast as the loop runs (relying on
    /// vsync, or deliberately uncapped). The default.
    #[default]
    Off,
    /// Cap to a target frames-per-second. Converted to a per-frame period by
    /// [`period`](FrameLimit::period).
    Fps(core::num::NonZeroU32),
    /// Cap to an explicit minimum frame period. A zero period is treated as
    /// [`Off`](FrameLimit::Off).
    Period(Duration),
}

impl FrameLimit {
    /// Build a limit from an integer FPS. `0` maps to [`Off`](FrameLimit::Off)
    /// so callers can pass a user/config value without a separate guard.
    #[must_use]
    pub fn from_fps(fps: u32) -> Self {
        match core::num::NonZeroU32::new(fps) {
            Some(n) => FrameLimit::Fps(n),
            None => FrameLimit::Off,
        }
    }

    /// The target minimum frame period, or `None` when unlimited.
    ///
    /// `Fps(n)` becomes `1s / n` (nanosecond-truncated, which is well under a
    /// microsecond of error even at thousands of FPS). A zero `Period` is
    /// unlimited.
    #[must_use]
    pub fn period(self) -> Option<Duration> {
        match self {
            FrameLimit::Off => None,
            FrameLimit::Fps(fps) => {
                Some(Duration::from_nanos(1_000_000_000u64 / u64::from(fps.get())))
            }
            FrameLimit::Period(p) if p.is_zero() => None,
            FrameLimit::Period(p) => Some(p),
        }
    }

    /// Whether this limit imposes any cap at all.
    #[must_use]
    pub fn is_limited(self) -> bool {
        self.period().is_some()
    }
}

/// Rolling frame-interval statistics over a bounded window (design §16).
///
/// Records the wall-clock interval between successive frame boundaries so the
/// loop can report achieved frame time, average, and the window's worst frame
/// (a stand-in for the "1% low" the design cares about more than mean FPS).
/// The window is bounded, so memory is O(window) regardless of run length.
#[derive(Clone, Debug)]
pub struct FrameStats {
    window: usize,
    samples: VecDeque<Duration>,
    total_frames: u64,
}

impl FrameStats {
    /// A stats buffer keeping at most `window` recent intervals. A `window` of
    /// `0` is clamped to `1` so there is always room for the last interval.
    #[must_use]
    pub fn new(window: usize) -> Self {
        let window = window.max(1);
        Self {
            window,
            samples: VecDeque::with_capacity(window),
            total_frames: 0,
        }
    }

    /// Record one frame interval, evicting the oldest sample past the window.
    pub fn record(&mut self, interval: Duration) {
        if self.samples.len() == self.window {
            self.samples.pop_front();
        }
        self.samples.push_back(interval);
        self.total_frames = self.total_frames.saturating_add(1);
    }

    /// Total intervals recorded over the pacer's whole lifetime (not just the
    /// window).
    #[must_use]
    pub fn total_frames(&self) -> u64 {
        self.total_frames
    }

    /// Number of samples currently in the window.
    #[must_use]
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Whether no interval has been recorded yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// The most recent frame interval, or `None` before the first record.
    #[must_use]
    pub fn last(&self) -> Option<Duration> {
        self.samples.back().copied()
    }

    /// Mean frame interval over the window, or `None` when empty.
    #[must_use]
    pub fn average(&self) -> Option<Duration> {
        if self.samples.is_empty() {
            return None;
        }
        let total: Duration = self.samples.iter().copied().sum();
        Some(total / self.samples.len() as u32)
    }

    /// Worst (longest) frame interval in the window — the hitch metric.
    #[must_use]
    pub fn worst(&self) -> Option<Duration> {
        self.samples.iter().copied().max()
    }

    /// Best (shortest) frame interval in the window.
    #[must_use]
    pub fn best(&self) -> Option<Duration> {
        self.samples.iter().copied().min()
    }
}

/// A drift-free frame limiter plus rolling [`FrameStats`] (design §13, §16).
///
/// Call [`throttle`](FramePacer::throttle) exactly once per frame, after the
/// frame's work. It records the achieved interval and, when a
/// [`FrameLimit`] is set, blocks the calling thread until the next cadence
/// boundary. The cadence advances by whole periods (`deadline += period`), so
/// the limiter does not drift the way a `sleep(period - work)` limiter would.
pub struct FramePacer {
    limit: FrameLimit,
    max_catch_up: Duration,
    next_deadline: Option<Instant>,
    last_frame_start: Option<Instant>,
    stats: FrameStats,
}

impl Default for FramePacer {
    fn default() -> Self {
        Self::new(FrameLimit::Off)
    }
}

impl FramePacer {
    /// How far behind the cadence may fall before it resyncs instead of
    /// catching up (anti death-spiral). A hitch longer than this skips the
    /// missed boundaries rather than firing a burst of zero-length frames.
    pub const DEFAULT_MAX_CATCH_UP: Duration = Duration::from_millis(100);

    /// Default size of the rolling statistics window (frames).
    pub const DEFAULT_STATS_WINDOW: usize = 120;

    /// A pacer with the given limit, default catch-up clamp and stats window.
    #[must_use]
    pub fn new(limit: FrameLimit) -> Self {
        Self {
            limit,
            max_catch_up: Self::DEFAULT_MAX_CATCH_UP,
            next_deadline: None,
            last_frame_start: None,
            stats: FrameStats::new(Self::DEFAULT_STATS_WINDOW),
        }
    }

    /// Override the rolling statistics window (frames). Builder-style.
    #[must_use]
    pub fn with_stats_window(mut self, window: usize) -> Self {
        self.stats = FrameStats::new(window);
        self
    }

    /// Override the anti-death-spiral catch-up clamp. Builder-style.
    #[must_use]
    pub fn with_max_catch_up(mut self, max_catch_up: Duration) -> Self {
        self.max_catch_up = max_catch_up;
        self
    }

    /// The active limit.
    #[must_use]
    pub fn limit(&self) -> FrameLimit {
        self.limit
    }

    /// Change the limit at runtime (e.g. a cvar flipping the frame cap). The
    /// cadence is reset so the new rate takes effect cleanly on the next frame.
    pub fn set_limit(&mut self, limit: FrameLimit) {
        self.limit = limit;
        self.next_deadline = None;
    }

    /// Rolling frame-interval statistics.
    #[must_use]
    pub fn stats(&self) -> &FrameStats {
        &self.stats
    }

    /// Drop the pacing cadence (not the stats). Call after a large time gap —
    /// e.g. resuming from suspend (design §12) — so the loop does not try to
    /// "catch up" the wall-clock time it was parked.
    pub fn reset_cadence(&mut self) {
        self.next_deadline = None;
        self.last_frame_start = None;
    }

    /// Record this frame's interval and, if limited, block until the next
    /// cadence boundary. Returns the duration actually slept (`ZERO` when
    /// unlimited or already at/behind the boundary).
    pub fn throttle(&mut self) -> Duration {
        let now = Instant::now();
        self.record_interval(now);

        let Some(period) = self.limit.period() else {
            // Unlimited: keep the cadence clear so re-enabling a limit starts
            // fresh instead of chasing a stale deadline.
            self.next_deadline = None;
            return Duration::ZERO;
        };

        let (sleep, next) =
            Self::next_boundary(self.next_deadline, now, period, self.max_catch_up);
        self.next_deadline = Some(next);
        if !sleep.is_zero() {
            std::thread::sleep(sleep);
        }
        sleep
    }

    /// Fold the interval since the previous frame boundary into the stats.
    fn record_interval(&mut self, now: Instant) {
        if let Some(prev) = self.last_frame_start {
            self.stats.record(now.saturating_duration_since(prev));
        }
        self.last_frame_start = Some(now);
    }

    /// Pure cadence arithmetic, split out so it is testable without a real
    /// clock. Given the current `deadline` (if any), the current time `now`,
    /// the frame `period`, and the catch-up clamp, returns the duration to
    /// sleep and the next deadline.
    ///
    /// - No deadline yet (first limited frame): don't sleep, anchor the cadence
    ///   one period out.
    /// - Boundary still ahead: sleep until it, advance the cadence by one
    ///   period (drift-free).
    /// - Boundary already passed but within `max_catch_up`: don't sleep, still
    ///   advance by one period so a small overrun is absorbed over the next few
    ///   frames.
    /// - Boundary passed by more than `max_catch_up`: don't sleep and *resync*
    ///   the cadence to `now + period`, dropping the missed boundaries.
    fn next_boundary(
        deadline: Option<Instant>,
        now: Instant,
        period: Duration,
        max_catch_up: Duration,
    ) -> (Duration, Instant) {
        let Some(deadline) = deadline else {
            return (Duration::ZERO, now + period);
        };
        let until = deadline.saturating_duration_since(now);
        if !until.is_zero() {
            return (until, deadline + period);
        }
        let behind = now.saturating_duration_since(deadline);
        if behind > max_catch_up {
            (Duration::ZERO, now + period)
        } else {
            (Duration::ZERO, deadline + period)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fps_round_trips_to_a_period() {
        assert_eq!(FrameLimit::from_fps(0), FrameLimit::Off);
        assert_eq!(
            FrameLimit::from_fps(60).period(),
            Some(Duration::from_nanos(16_666_666))
        );
        assert!(FrameLimit::from_fps(60).is_limited());
        assert!(!FrameLimit::Off.is_limited());
        assert_eq!(FrameLimit::Period(Duration::ZERO).period(), None);
    }

    #[test]
    fn first_limited_frame_anchors_without_sleeping() {
        let base = Instant::now();
        let period = Duration::from_millis(10);
        let (sleep, next) = FramePacer::next_boundary(None, base, period, Duration::from_millis(100));
        assert_eq!(sleep, Duration::ZERO);
        assert_eq!(next.saturating_duration_since(base), period);
    }

    #[test]
    fn boundary_ahead_sleeps_until_it_and_advances_by_one_period() {
        let base = Instant::now();
        let period = Duration::from_millis(10);
        let deadline = base + period; // 10ms out
        let now = base + Duration::from_millis(4); // 4ms into the frame
        let (sleep, next) =
            FramePacer::next_boundary(Some(deadline), now, period, Duration::from_millis(100));
        // Sleep the remaining 6ms, then the cadence moves to 20ms (drift-free).
        assert_eq!(sleep, Duration::from_millis(6));
        assert_eq!(next.saturating_duration_since(base), Duration::from_millis(20));
    }

    #[test]
    fn small_overrun_absorbs_without_sleeping() {
        let base = Instant::now();
        let period = Duration::from_millis(10);
        let deadline = base + period;
        // 3ms late, well within the 100ms clamp: no sleep, cadence still += one
        // period so the small overrun is paid back over following frames.
        let now = base + Duration::from_millis(13);
        let (sleep, next) =
            FramePacer::next_boundary(Some(deadline), now, period, Duration::from_millis(100));
        assert_eq!(sleep, Duration::ZERO);
        assert_eq!(next.saturating_duration_since(base), Duration::from_millis(20));
    }

    #[test]
    fn large_hitch_resyncs_cadence_instead_of_bursting() {
        let base = Instant::now();
        let period = Duration::from_millis(10);
        let deadline = base + period;
        // 500ms hitch, far past the 100ms clamp: resync to now + period rather
        // than firing ~50 zero-length catch-up frames (anti death-spiral).
        let now = base + Duration::from_millis(510);
        let (sleep, next) =
            FramePacer::next_boundary(Some(deadline), now, period, Duration::from_millis(100));
        assert_eq!(sleep, Duration::ZERO);
        assert_eq!(next.saturating_duration_since(base), Duration::from_millis(520));
    }

    #[test]
    fn stats_track_window_average_and_worst() {
        let mut stats = FrameStats::new(3);
        assert!(stats.is_empty());
        stats.record(Duration::from_millis(10));
        stats.record(Duration::from_millis(20));
        stats.record(Duration::from_millis(30));
        // Window full at 3; recording a 4th evicts the oldest (10ms).
        stats.record(Duration::from_millis(40));
        assert_eq!(stats.len(), 3);
        assert_eq!(stats.total_frames(), 4);
        assert_eq!(stats.last(), Some(Duration::from_millis(40)));
        assert_eq!(stats.worst(), Some(Duration::from_millis(40)));
        assert_eq!(stats.best(), Some(Duration::from_millis(20)));
        // (20 + 30 + 40) / 3 = 30ms.
        assert_eq!(stats.average(), Some(Duration::from_millis(30)));
    }

    #[test]
    fn unlimited_throttle_never_sleeps_but_still_counts_frames() {
        let mut pacer = FramePacer::new(FrameLimit::Off);
        assert_eq!(pacer.throttle(), Duration::ZERO);
        assert_eq!(pacer.throttle(), Duration::ZERO);
        // Two throttles → one measured interval between them.
        assert_eq!(pacer.stats().total_frames(), 1);
        assert_eq!(pacer.limit(), FrameLimit::Off);
    }

    #[test]
    fn set_limit_resets_cadence() {
        let mut pacer = FramePacer::new(FrameLimit::from_fps(60));
        pacer.throttle(); // anchors a cadence
        pacer.set_limit(FrameLimit::Off);
        assert_eq!(pacer.limit(), FrameLimit::Off);
        // Back to unlimited: no sleep.
        assert_eq!(pacer.throttle(), Duration::ZERO);
    }
}
