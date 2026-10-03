//! Adaptive frame-rate selection within a quality tier (design §13).
//!
//! The design calls for an adaptive refinement on top of the fixed-cadence
//! [`FramePacer`](super::FramePacer): *"按最近帧时间直方图在 quality tier 内微调"* —
//! watch the recent frame-time histogram and nudge the frame cap up or down
//! inside the tier so the loop targets the highest cadence it can actually
//! sustain, rather than a single fixed guess.
//!
//! This is **pure policy** — it never reads a clock and never sleeps. The
//! caller feeds it each frame's measured *work* time (how long the frame took,
//! not the paced interval) and it returns a new [`FrameLimit`] whenever the
//! sustained histogram says a different rung of the ladder is a better fit. The
//! runner then applies that limit to its [`FramePacer`](super::FramePacer), so
//! this type composes with the drift-free limiter without duplicating any of
//! its cadence math.
//!
//! # What this is *not*
//!
//! Resolution scaling, render-ahead depth, and present-timestamp alignment are
//! all explicitly out of scope (they live in the render/window layers, per the
//! §13 notes in the module docs). This type only selects *which frame-rate cap*
//! the pacer should target; the design is explicit that the adaptive policy
//! "只管节奏" (only governs cadence).

use std::collections::VecDeque;

use prism_time::Duration;

use super::FrameLimit;

/// A fixed, ordered ladder of candidate frame-rate caps.
///
/// The adaptive limiter only ever moves one rung at a time along this ladder,
/// so the set of reachable caps is bounded and predictable (a "quality tier"
/// of cadences, in the design's words). Rungs are stored from **most
/// permissive** (lowest FPS / longest period, index `0`) to **most demanding**
/// (highest FPS / shortest period, the last index), so a lower index is always
/// the easier target.
///
/// Every rung is a *limited* [`FrameLimit`] (it has a concrete period); an
/// unlimited cap is meaningless as an adaptive target and is filtered out at
/// construction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrameRateLadder {
    /// Caps ordered most-permissive → most-demanding, de-duplicated by period.
    steps: Vec<FrameLimit>,
}

impl FrameRateLadder {
    /// Build a ladder from a set of target FPS values.
    ///
    /// Zero entries (which [`FrameLimit::from_fps`] maps to unlimited) are
    /// dropped, the rest are ordered most-permissive → most-demanding and
    /// de-duplicated by period. Returns `None` when no usable rung remains, so
    /// a caller never ends up with an empty ladder.
    #[must_use]
    pub fn from_fps(fps: impl IntoIterator<Item = u32>) -> Option<Self> {
        Self::new(fps.into_iter().map(FrameLimit::from_fps))
    }

    /// Build a ladder from explicit [`FrameLimit`]s.
    ///
    /// Unlimited rungs (no period) are dropped, the rest are ordered
    /// most-permissive → most-demanding and de-duplicated by period. Returns
    /// `None` when no limited rung remains.
    #[must_use]
    pub fn new(limits: impl IntoIterator<Item = FrameLimit>) -> Option<Self> {
        let mut steps: Vec<FrameLimit> =
            limits.into_iter().filter(|l| l.period().is_some()).collect();
        if steps.is_empty() {
            return None;
        }
        // Longest period first = lowest FPS first = most permissive first.
        steps.sort_by_key(|l| core::cmp::Reverse(l.period()));
        steps.dedup_by(|a, b| a.period() == b.period());
        Some(Self { steps })
    }

    /// The rungs, most-permissive first.
    #[must_use]
    pub fn steps(&self) -> &[FrameLimit] {
        &self.steps
    }

    /// Number of rungs (always at least `1`).
    #[must_use]
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Always `false`: a ladder is never empty by construction. Provided for
    /// clippy's `len_without_is_empty` and for symmetry.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// The cap at `index`, or `None` when out of range.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<FrameLimit> {
        self.steps.get(index).copied()
    }
}

/// The decision an [`AdaptiveFrameLimiter`] reaches for the current window.
///
/// Split out from the mutation so the decision rule can be unit-tested without
/// driving a real limiter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdaptiveAction {
    /// Keep the current rung (window not full, in-band, or already clamped).
    Hold,
    /// Move one rung toward a *more demanding* cap (higher FPS).
    StepUp,
    /// Move one rung toward a *more permissive* cap (lower FPS).
    StepDown,
}

/// A histogram-driven cap selector that walks a [`FrameRateLadder`] (design §13).
///
/// Feed every frame's measured work time to [`record`](Self::record). Once a
/// full window of samples has accumulated it decides whether the loop is
/// sustaining its current cap:
///
/// - **Step down** (toward a lower FPS) when the window's *average* work time
///   already exceeds the current period — a sustained miss the loop cannot pace
///   away.
/// - **Step up** (toward a higher FPS) when even the window's *worst* frame
///   fits inside the next rung's period with a configurable headroom margin —
///   so a climb only happens when there is real slack, not on a lucky average.
/// - **Hold** otherwise.
///
/// On any step the window is cleared, which gives natural hysteresis: a fresh
/// window of evidence must accumulate before the next move, so the limiter does
/// not oscillate between adjacent rungs. A single-rung ladder always holds.
///
/// All comparisons are done in integer nanoseconds, so the decision is
/// deterministic and independent of floating-point rounding.
#[derive(Clone, Debug)]
pub struct AdaptiveFrameLimiter {
    ladder: FrameRateLadder,
    index: usize,
    window: usize,
    headroom_percent: u32,
    samples: VecDeque<Duration>,
}

impl AdaptiveFrameLimiter {
    /// Default number of work-time samples per decision window.
    pub const DEFAULT_WINDOW: usize = 120;

    /// Default headroom margin (percent) required before stepping up.
    pub const DEFAULT_HEADROOM_PERCENT: u32 = 10;

    /// Upper bound on the headroom margin, so there is always at least a sliver
    /// of the next period available as a step-up budget.
    pub const MAX_HEADROOM_PERCENT: u32 = 90;

    /// A limiter over `ladder`, starting at the most demanding rung (highest
    /// FPS) and backing off under load. Use [`with_start_index`](Self::with_start_index)
    /// to start lower / more conservatively.
    #[must_use]
    pub fn new(ladder: FrameRateLadder) -> Self {
        let index = ladder.len().saturating_sub(1);
        Self {
            ladder,
            index,
            window: Self::DEFAULT_WINDOW,
            headroom_percent: Self::DEFAULT_HEADROOM_PERCENT,
            samples: VecDeque::with_capacity(Self::DEFAULT_WINDOW),
        }
    }

    /// Override the decision-window size (frames). Clamped to at least `1`.
    /// Builder-style.
    #[must_use]
    pub fn with_window(mut self, window: usize) -> Self {
        self.window = window.max(1);
        self.samples = VecDeque::with_capacity(self.window);
        self
    }

    /// Override the step-up headroom margin (percent). Clamped to
    /// `[0, MAX_HEADROOM_PERCENT]`. Builder-style.
    #[must_use]
    pub fn with_headroom_percent(mut self, headroom_percent: u32) -> Self {
        self.headroom_percent = headroom_percent.min(Self::MAX_HEADROOM_PERCENT);
        self
    }

    /// Start at a specific rung instead of the most demanding one. The index is
    /// clamped to the ladder's range. Builder-style.
    #[must_use]
    pub fn with_start_index(mut self, index: usize) -> Self {
        self.index = index.min(self.ladder.len().saturating_sub(1));
        self
    }

    /// The cap the loop should currently target.
    #[must_use]
    pub fn current_limit(&self) -> FrameLimit {
        self.ladder.get(self.index).unwrap_or_default()
    }

    /// The current rung index (`0` = most permissive).
    #[must_use]
    pub fn index(&self) -> usize {
        self.index
    }

    /// The ladder this limiter walks.
    #[must_use]
    pub fn ladder(&self) -> &FrameRateLadder {
        &self.ladder
    }

    /// The decision-window size (frames).
    #[must_use]
    pub fn window(&self) -> usize {
        self.window
    }

    /// The step-up headroom margin (percent).
    #[must_use]
    pub fn headroom_percent(&self) -> u32 {
        self.headroom_percent
    }

    /// Number of samples currently in the window.
    #[must_use]
    pub fn sample_len(&self) -> usize {
        self.samples.len()
    }

    /// Record one frame's measured *work* time.
    ///
    /// Returns `Some(new_limit)` when the sustained histogram warrants moving to
    /// a different rung (and resets the window), or `None` to hold. Pass the
    /// time the frame's work actually took — not the paced interval — so the
    /// decision reflects headroom rather than the cap already in force.
    pub fn record(&mut self, work: Duration) -> Option<FrameLimit> {
        if self.samples.len() == self.window {
            self.samples.pop_front();
        }
        self.samples.push_back(work);
        match self.evaluate() {
            AdaptiveAction::Hold => None,
            AdaptiveAction::StepUp => {
                self.index += 1;
                self.samples.clear();
                Some(self.current_limit())
            }
            AdaptiveAction::StepDown => {
                self.index -= 1;
                self.samples.clear();
                Some(self.current_limit())
            }
        }
    }

    /// The decision for the current window, without mutating anything.
    ///
    /// `Hold` until the window is full. Then step down on a sustained miss
    /// (window average over the current period), step up when the window's
    /// worst frame fits the next rung's period with headroom, else hold.
    #[must_use]
    pub fn evaluate(&self) -> AdaptiveAction {
        if self.samples.len() < self.window {
            return AdaptiveAction::Hold;
        }
        let current_period = period_nanos(self.current_limit());
        let average = average_nanos(&self.samples);
        if average > current_period {
            // Missing even the current cap: ease off if we can, never climb.
            return if self.index > 0 {
                AdaptiveAction::StepDown
            } else {
                AdaptiveAction::Hold
            };
        }
        // Room to try a more demanding rung?
        if let Some(next) = self.ladder.get(self.index + 1) {
            let next_period = period_nanos(next);
            let budget = next_period
                .saturating_mul(u128::from(100 - self.headroom_percent))
                / 100;
            if worst_nanos(&self.samples) <= budget {
                return AdaptiveAction::StepUp;
            }
        }
        AdaptiveAction::Hold
    }
}

/// The period of a cap in integer nanoseconds; unlimited caps map to
/// `u128::MAX` so they can never be "missed".
fn period_nanos(limit: FrameLimit) -> u128 {
    limit.period().map_or(u128::MAX, |p| p.as_nanos())
}

/// Mean of the window in integer nanoseconds. Callers guarantee a non-empty
/// window before relying on this.
fn average_nanos(samples: &VecDeque<Duration>) -> u128 {
    let total: u128 = samples.iter().map(Duration::as_nanos).sum();
    total / samples.len() as u128
}

/// Longest sample in the window in integer nanoseconds (`0` when empty).
fn worst_nanos(samples: &VecDeque<Duration>) -> u128 {
    samples.iter().map(Duration::as_nanos).max().unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ladder() -> FrameRateLadder {
        FrameRateLadder::from_fps([30, 60, 120]).expect("non-empty")
    }

    #[test]
    fn ladder_orders_permissive_first_and_dedups() {
        let l = FrameRateLadder::from_fps([120, 30, 60, 60, 0]).expect("non-empty");
        // Zero dropped, duplicate 60 collapsed, ordered 30 → 60 → 120.
        assert_eq!(l.len(), 3);
        assert_eq!(l.get(0), Some(FrameLimit::from_fps(30)));
        assert_eq!(l.get(1), Some(FrameLimit::from_fps(60)));
        assert_eq!(l.get(2), Some(FrameLimit::from_fps(120)));
        assert_eq!(l.get(3), None);
    }

    #[test]
    fn ladder_rejects_all_unlimited() {
        assert!(FrameRateLadder::from_fps([0, 0]).is_none());
        assert!(FrameRateLadder::new([FrameLimit::Off]).is_none());
    }

    #[test]
    fn new_starts_at_most_demanding_rung() {
        let a = AdaptiveFrameLimiter::new(ladder());
        assert_eq!(a.index(), 2);
        assert_eq!(a.current_limit(), FrameLimit::from_fps(120));
    }

    #[test]
    fn window_not_full_holds() {
        let mut a = AdaptiveFrameLimiter::new(ladder()).with_window(4);
        // Huge work times, but fewer than a full window → no decision yet.
        for _ in 0..3 {
            assert_eq!(a.record(Duration::from_millis(100)), None);
        }
        assert_eq!(a.index(), 2);
    }

    #[test]
    fn sustained_miss_steps_down_and_clears_window() {
        let mut a = AdaptiveFrameLimiter::new(ladder()).with_window(4);
        // 120fps period is ~8.3ms; 20ms work misses it on average.
        let mut moved = None;
        for _ in 0..4 {
            if let Some(l) = a.record(Duration::from_millis(20)) {
                moved = Some(l);
            }
        }
        assert_eq!(moved, Some(FrameLimit::from_fps(60)));
        assert_eq!(a.index(), 1);
        // Window was cleared on the step (hysteresis).
        assert_eq!(a.sample_len(), 0);
    }

    #[test]
    fn ample_headroom_steps_up() {
        let mut a = AdaptiveFrameLimiter::new(ladder())
            .with_window(4)
            .with_start_index(0); // start at 30fps
        // 60fps period is ~16.6ms; with 10% headroom the budget is ~15ms.
        // 2ms work is far inside it → climb.
        let mut moved = None;
        for _ in 0..4 {
            if let Some(l) = a.record(Duration::from_millis(2)) {
                moved = Some(l);
            }
        }
        assert_eq!(moved, Some(FrameLimit::from_fps(60)));
        assert_eq!(a.index(), 1);
    }

    #[test]
    fn in_band_holds() {
        // Work that beats the current 60fps period but does not fit the 120fps
        // period with headroom → neither step fires.
        let mut a = AdaptiveFrameLimiter::new(ladder())
            .with_window(4)
            .with_start_index(1); // 60fps, period ~16.6ms
        // 120fps budget with 10% headroom is ~7.5ms; 12ms exceeds it but is
        // under the current 16.6ms period, so hold.
        for _ in 0..4 {
            assert_eq!(a.record(Duration::from_millis(12)), None);
        }
        assert_eq!(a.index(), 1);
    }

    #[test]
    fn headroom_margin_blocks_a_marginal_step_up() {
        let mut a = AdaptiveFrameLimiter::new(ladder())
            .with_window(2)
            .with_start_index(0) // 30fps
            .with_headroom_percent(20);
        // 60fps period ~16.6ms; 20% headroom budget ~13.3ms. A worst frame of
        // 15ms would fit the raw period but not the headroom budget → hold.
        a.record(Duration::from_millis(15));
        assert_eq!(a.record(Duration::from_millis(15)), None);
        assert_eq!(a.index(), 0);
    }

    #[test]
    fn single_rung_ladder_never_moves() {
        let mut a =
            AdaptiveFrameLimiter::new(FrameRateLadder::from_fps([60]).expect("non-empty"))
                .with_window(2);
        assert_eq!(a.record(Duration::from_millis(100)), None);
        assert_eq!(a.record(Duration::from_millis(100)), None);
        assert_eq!(a.index(), 0);
        assert_eq!(a.record(Duration::from_micros(1)), None);
    }

    #[test]
    fn does_not_step_below_the_floor() {
        let mut a = AdaptiveFrameLimiter::new(ladder())
            .with_window(2)
            .with_start_index(0); // already at the most permissive rung
        // Missing even 30fps (33ms): nowhere lower to go → hold.
        a.record(Duration::from_millis(50));
        assert_eq!(a.record(Duration::from_millis(50)), None);
        assert_eq!(a.index(), 0);
    }

    #[test]
    fn does_not_step_above_the_ceiling() {
        let mut a = AdaptiveFrameLimiter::new(ladder()).with_window(2); // starts at 120fps (top)
        // Trivial work, but already at the most demanding rung → hold.
        a.record(Duration::from_micros(10));
        assert_eq!(a.record(Duration::from_micros(10)), None);
        assert_eq!(a.index(), 2);
    }
}
