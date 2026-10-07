//! §24.2 — present-cadence smoothing ([`FramePacer`]) and the low-latency
//! begin-work decision ([`ReflexGate`]).

use super::ema_step;

/// Outcome of feeding one present timestamp to a [`FramePacer`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PresentInfo {
    /// Measured interval since the previous present (`0` on the first present).
    pub interval_ns: u64,
    /// Smoothed (EMA) present interval after this sample.
    pub smoothed_ns: u64,
    /// Signed deviation of this present from the predicted cadence slot
    /// (`actual - predicted`); positive means the present landed late. `0` on
    /// the first present.
    pub jitter_ns: i64,
    /// Predicted time of the next present (the cadence slot to pace toward).
    pub next_present_ns: u64,
    /// Whether this present deviated past [`FramePacer::resync_threshold_ns`]
    /// and forced a hard re-anchor of the cadence onto the actual timestamp.
    pub resynced: bool,
}

/// Phase-locked present-cadence smoother (console-style frame pacing, §24.2).
///
/// Fed a monotonically increasing stream of caller-supplied present timestamps
/// (nanoseconds — never read from a clock here), it maintains a steady
/// *predicted* next-present time advancing by exactly `target_interval_ns` each
/// frame. Small jitter does not move the prediction (so micro-stutter is not
/// re-radiated into submission timing); only a deviation beyond
/// [`resync_threshold_ns`](Self::resync_threshold_ns) hard re-anchors the
/// cadence onto the actual timestamp. A separate EMA tracks the observed
/// interval for diagnostics / dynamic-cadence consumers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FramePacer {
    target_ns: u64,
    smoothed_ns: u64,
    alpha_num: u32,
    alpha_den: u32,
    resync_threshold_ns: u64,
    last_present_ns: Option<u64>,
    next_present_ns: u64,
    last_jitter_ns: i64,
    max_jitter_ns: u64,
    presents: u64,
    resyncs: u64,
}

impl FramePacer {
    /// New pacer targeting `target_interval_ns` per present (e.g. `16_666_667`
    /// for 60 Hz). Smoothing defaults to a `1/8` EMA weight and the re-anchor
    /// threshold to half the target interval. `target_interval_ns` is clamped to
    /// at least `1`.
    #[inline]
    #[must_use]
    pub const fn new(target_interval_ns: u64) -> Self {
        let target = if target_interval_ns == 0 {
            1
        } else {
            target_interval_ns
        };
        Self {
            target_ns: target,
            smoothed_ns: target,
            alpha_num: 1,
            alpha_den: 8,
            resync_threshold_ns: target / 2,
            last_present_ns: None,
            next_present_ns: 0,
            last_jitter_ns: 0,
            max_jitter_ns: 0,
            presents: 0,
            resyncs: 0,
        }
    }

    /// New pacer from a target refresh rate in whole hertz (clamped to `>= 1`).
    #[inline]
    #[must_use]
    pub const fn from_hz(hz: u32) -> Self {
        let hz = if hz == 0 { 1 } else { hz };
        Self::new(1_000_000_000 / hz as u64)
    }

    /// Override the EMA smoothing weight (`num/den`) for the observed-interval
    /// estimate. `den` is clamped to `>= 1` and `num` to `<= den`.
    #[inline]
    #[must_use]
    pub const fn with_smoothing(mut self, num: u32, den: u32) -> Self {
        let den = if den == 0 { 1 } else { den };
        let num = if num > den { den } else { num };
        self.alpha_num = num;
        self.alpha_den = den;
        self
    }

    /// Override the deviation (ns) beyond which a present hard re-anchors the
    /// predicted cadence instead of being absorbed as jitter.
    #[inline]
    #[must_use]
    pub const fn with_resync_threshold(mut self, threshold_ns: u64) -> Self {
        self.resync_threshold_ns = threshold_ns;
        self
    }

    /// Target present interval (ns).
    #[inline]
    #[must_use]
    pub const fn target_interval_ns(&self) -> u64 {
        self.target_ns
    }

    /// Smoothed observed present interval (ns).
    #[inline]
    #[must_use]
    pub const fn smoothed_interval_ns(&self) -> u64 {
        self.smoothed_ns
    }

    /// Predicted time of the next present (the cadence slot to aim submission
    /// at). Meaningful once at least one present has been recorded.
    #[inline]
    #[must_use]
    pub const fn next_present_ns(&self) -> u64 {
        self.next_present_ns
    }

    /// Re-anchor threshold (ns).
    #[inline]
    #[must_use]
    pub const fn resync_threshold_ns(&self) -> u64 {
        self.resync_threshold_ns
    }

    /// Signed cadence deviation of the most recent present (`actual - predicted`).
    #[inline]
    #[must_use]
    pub const fn last_jitter_ns(&self) -> i64 {
        self.last_jitter_ns
    }

    /// Largest absolute cadence deviation observed since construction / reset.
    #[inline]
    #[must_use]
    pub const fn max_jitter_ns(&self) -> u64 {
        self.max_jitter_ns
    }

    /// Number of presents recorded since construction / reset.
    #[inline]
    #[must_use]
    pub const fn presents(&self) -> u64 {
        self.presents
    }

    /// Number of hard cadence re-anchors since construction / reset.
    #[inline]
    #[must_use]
    pub const fn resyncs(&self) -> u64 {
        self.resyncs
    }

    /// Record a present at `now_ns` and advance the pacing state.
    ///
    /// The first present only anchors the cadence (interval / jitter are `0`).
    /// A non-monotonic timestamp (`now_ns` not after the previous present) is
    /// treated as a zero-length interval and forces a re-anchor, keeping the
    /// predicted cadence sane.
    pub fn on_present(&mut self, now_ns: u64) -> PresentInfo {
        self.presents = self.presents.saturating_add(1);

        let Some(last) = self.last_present_ns else {
            // First present: anchor the cadence.
            self.last_present_ns = Some(now_ns);
            self.next_present_ns = now_ns.saturating_add(self.target_ns);
            self.last_jitter_ns = 0;
            return PresentInfo {
                interval_ns: 0,
                smoothed_ns: self.smoothed_ns,
                jitter_ns: 0,
                next_present_ns: self.next_present_ns,
                resynced: false,
            };
        };

        let interval_ns = now_ns.saturating_sub(last);
        self.smoothed_ns = ema_step(
            self.smoothed_ns,
            interval_ns,
            self.alpha_num,
            self.alpha_den,
        );

        let jitter = i128::from(now_ns) - i128::from(self.next_present_ns);
        let jitter_ns = jitter.clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64;
        self.last_jitter_ns = jitter_ns;
        let abs_jitter = jitter.unsigned_abs().min(u128::from(u64::MAX)) as u64;
        if abs_jitter > self.max_jitter_ns {
            self.max_jitter_ns = abs_jitter;
        }

        // Absorb small jitter by keeping the steady predicted cadence; hard
        // re-anchor only on a large deviation (or a non-monotonic timestamp).
        let non_monotonic = now_ns <= last;
        let resynced = abs_jitter > self.resync_threshold_ns || non_monotonic;
        let base = if resynced {
            self.resyncs = self.resyncs.saturating_add(1);
            now_ns
        } else {
            self.next_present_ns
        };
        self.next_present_ns = base.saturating_add(self.target_ns);
        self.last_present_ns = Some(now_ns);

        PresentInfo {
            interval_ns,
            smoothed_ns: self.smoothed_ns,
            jitter_ns,
            next_present_ns: self.next_present_ns,
            resynced,
        }
    }

    /// Clear all history, keeping the configuration.
    #[inline]
    pub fn reset(&mut self) {
        self.smoothed_ns = self.target_ns;
        self.last_present_ns = None;
        self.next_present_ns = 0;
        self.last_jitter_ns = 0;
        self.max_jitter_ns = 0;
        self.presents = 0;
        self.resyncs = 0;
    }
}

/// A begin-work decision from a [`ReflexGate`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BeginDecision {
    /// Begin simulation now — this is the just-in-time moment.
    Begin,
    /// Late-latch: hold `wait_ns` before beginning so input is sampled as close
    /// to simulation start as possible (compresses input-to-photon latency).
    Wait {
        /// Nanoseconds to wait before beginning the frame.
        wait_ns: u64,
    },
    /// The render queue already holds `max_frames_in_flight` frames; do not
    /// begin a new one until a present drains the queue.
    QueueFull,
}

/// Low-latency begin-work gate (NVIDIA Reflex / AMD Anti-Lag *form*, §24.2).
///
/// Encodes "begin work just-in-time, do not queue too many frames ahead": given
/// the present deadline, an estimate of CPU+GPU work for the frame, and how many
/// frames are already in flight, it decides whether to [`Begin`](BeginDecision::Begin),
/// [`Wait`](BeginDecision::Wait) (late-latch), or hold on
/// [`QueueFull`](BeginDecision::QueueFull). Deterministic and clock-free; the
/// caller supplies all timings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReflexGate {
    max_frames_in_flight: u32,
    safety_margin_ns: u64,
}

impl ReflexGate {
    /// New gate allowing at most `max_frames_in_flight` frames queued ahead
    /// (clamped to `>= 1`), beginning work `safety_margin_ns` earlier than the
    /// bare just-in-time point as a cushion against work-estimate error.
    #[inline]
    #[must_use]
    pub const fn new(max_frames_in_flight: u32, safety_margin_ns: u64) -> Self {
        Self {
            max_frames_in_flight: if max_frames_in_flight == 0 {
                1
            } else {
                max_frames_in_flight
            },
            safety_margin_ns,
        }
    }

    /// Maximum frames allowed in flight.
    #[inline]
    #[must_use]
    pub const fn max_frames_in_flight(&self) -> u32 {
        self.max_frames_in_flight
    }

    /// Safety margin (ns) subtracted from the just-in-time begin point.
    #[inline]
    #[must_use]
    pub const fn safety_margin_ns(&self) -> u64 {
        self.safety_margin_ns
    }

    /// The latest moment work may begin and still finish by `present_deadline_ns`
    /// given `estimated_work_ns` (CPU + GPU), including the safety margin.
    /// Saturates at `0` if the deadline is already too close.
    #[inline]
    #[must_use]
    pub const fn latest_begin_ns(&self, present_deadline_ns: u64, estimated_work_ns: u64) -> u64 {
        present_deadline_ns
            .saturating_sub(estimated_work_ns)
            .saturating_sub(self.safety_margin_ns)
    }

    /// Decide whether to begin the next frame.
    ///
    /// - `frames_in_flight >= max_frames_in_flight` → [`QueueFull`](BeginDecision::QueueFull).
    /// - otherwise, with `latest = latest_begin_ns(deadline, work)`:
    ///   - `now >= latest` → [`Begin`](BeginDecision::Begin) (at/past just-in-time),
    ///   - `now < latest` → [`Wait`](BeginDecision::Wait) for `latest - now` (late-latch).
    #[must_use]
    pub const fn decide(
        &self,
        now_ns: u64,
        present_deadline_ns: u64,
        estimated_work_ns: u64,
        frames_in_flight: u32,
    ) -> BeginDecision {
        if frames_in_flight >= self.max_frames_in_flight {
            return BeginDecision::QueueFull;
        }
        let latest = self.latest_begin_ns(present_deadline_ns, estimated_work_ns);
        if now_ns >= latest {
            BeginDecision::Begin
        } else {
            BeginDecision::Wait {
                wait_ns: latest - now_ns,
            }
        }
    }
}
