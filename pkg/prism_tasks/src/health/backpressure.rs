//! Deterministic, `no_std`-friendly core for queue-depth backpressure
//! (design §24.8 背压).
//!
//! This module owns the *decision* half of the backpressure policy and holds
//! no threads, no clock, and no allocation. Whether a submission is admitted,
//! deferred, or rejected is a pure function of the current queue depth, the
//! configured watermarks, and the submission's [`Priority`], so it is directly
//! unit-testable against a serial oracle. The thread-pool façade that actually
//! dispatches admitted jobs lives in the parent [`health`](crate::health)
//! module as [`BackpressureQueue`](crate::health::BackpressureQueue).
//!
//! # Policy
//! A [`QueueBackpressure`] tracks the number of outstanding items (`depth`)
//! against three limits:
//!
//! - `capacity` is a hard cap: once `depth` reaches it, *every* submission is
//!   [`Admission::Rejected`], regardless of [`Priority`]. This is the
//!   memory-blowup guard — the queue never grows past `capacity`.
//! - `high_watermark` arms background shedding: once `depth` reaches it,
//!   [`Priority::Background`] submissions are [`Admission::Deferred`] (shed)
//!   while foreground lanes keep flowing until `capacity`.
//! - `low_watermark` disarms shedding: background shedding stays on (hysteresis)
//!   until `depth` drains back down to `low_watermark`, so the queue does not
//!   oscillate at the high watermark.
//!
//! Shedding degrades deferrable [`Priority::Background`] work first (the §24.8
//! "拒绝/降级 Background 投递" contract) while never dropping committed
//! foreground work until the hard cap.

use crate::priority::Priority;

/// Depth thresholds governing a [`QueueBackpressure`].
///
/// Construct with [`BackpressureLimits::new`], which clamps the inputs so that
/// `low_watermark <= high_watermark <= capacity` always holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BackpressureLimits {
    /// Hard cap on queue depth; at this depth every submission is rejected.
    capacity: usize,
    /// Depth at which background shedding arms.
    high_watermark: usize,
    /// Depth at or below which background shedding disarms (hysteresis).
    low_watermark: usize,
}

impl BackpressureLimits {
    /// Create limits, clamping the inputs so that
    /// `low_watermark <= high_watermark <= capacity`.
    ///
    /// `capacity` is raised to at least `1` (a zero-capacity queue could never
    /// admit anything). `high_watermark` is clamped into `1..=capacity` and
    /// `low_watermark` into `0..=high_watermark`.
    #[must_use]
    pub fn new(capacity: usize, high_watermark: usize, low_watermark: usize) -> Self {
        let capacity = capacity.max(1);
        let high_watermark = high_watermark.clamp(1, capacity);
        let low_watermark = low_watermark.min(high_watermark);
        Self {
            capacity,
            high_watermark,
            low_watermark,
        }
    }

    /// The hard depth cap.
    #[must_use]
    #[inline]
    pub fn capacity(self) -> usize {
        self.capacity
    }

    /// The depth at which background shedding arms.
    #[must_use]
    #[inline]
    pub fn high_watermark(self) -> usize {
        self.high_watermark
    }

    /// The depth at or below which background shedding disarms.
    #[must_use]
    #[inline]
    pub fn low_watermark(self) -> usize {
        self.low_watermark
    }
}

/// The decision for one submission attempt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Admission {
    /// Accepted onto the queue; the depth is incremented.
    Admitted,
    /// A [`Priority::Background`] submission shed under backpressure; the caller
    /// should retry later or drop the deferrable work. Depth is unchanged.
    Deferred,
    /// Rejected because the queue is at `capacity`; applies to every
    /// [`Priority`]. Depth is unchanged.
    Rejected,
}

impl Admission {
    /// Whether the submission was accepted onto the queue.
    #[must_use]
    #[inline]
    pub fn is_admitted(self) -> bool {
        matches!(self, Admission::Admitted)
    }

    /// Whether the submission was shed (deferred) under backpressure.
    #[must_use]
    #[inline]
    pub fn is_deferred(self) -> bool {
        matches!(self, Admission::Deferred)
    }

    /// Whether the submission was rejected at the hard cap.
    #[must_use]
    #[inline]
    pub fn is_rejected(self) -> bool {
        matches!(self, Admission::Rejected)
    }
}

/// A deterministic queue-depth backpressure gate (design §24.8).
///
/// Query [`QueueBackpressure::peek`] for the decision without mutating, or
/// [`QueueBackpressure::offer`] to apply it (incrementing `depth` on an
/// admission). As work finishes, call [`QueueBackpressure::complete`] /
/// [`QueueBackpressure::complete_many`] to drain the depth back down; shedding
/// disarms once the depth reaches `low_watermark`.
#[derive(Clone, Copy, Debug)]
pub struct QueueBackpressure {
    /// The configured depth thresholds.
    limits: BackpressureLimits,
    /// Current number of outstanding (admitted, not-yet-completed) items.
    depth: usize,
    /// Whether background shedding is currently armed (hysteresis latch).
    shedding: bool,
}

impl QueueBackpressure {
    /// Create an empty gate (`depth == 0`, shedding disarmed) with `limits`.
    #[must_use]
    pub fn new(limits: BackpressureLimits) -> Self {
        Self {
            limits,
            depth: 0,
            shedding: false,
        }
    }

    /// The configured [`BackpressureLimits`].
    #[must_use]
    #[inline]
    pub fn limits(self) -> BackpressureLimits {
        self.limits
    }

    /// Current outstanding depth.
    #[must_use]
    #[inline]
    pub fn depth(self) -> usize {
        self.depth
    }

    /// Whether background shedding is currently armed.
    #[must_use]
    #[inline]
    pub fn is_shedding(self) -> bool {
        self.shedding
    }

    /// Remaining admissions before the hard `capacity` cap is hit.
    #[must_use]
    #[inline]
    pub fn headroom(self) -> usize {
        self.limits.capacity - self.depth
    }

    /// Decide what [`QueueBackpressure::offer`] would do for `priority` *without*
    /// mutating the gate.
    ///
    /// The decision is: reject at the hard `capacity` cap (any priority); else
    /// defer [`Priority::Background`] while shedding is armed or the depth has
    /// already reached `high_watermark`; else admit.
    #[must_use]
    pub fn peek(self, priority: Priority) -> Admission {
        if self.depth >= self.limits.capacity {
            return Admission::Rejected;
        }
        if priority == Priority::Background
            && (self.shedding || self.depth >= self.limits.high_watermark)
        {
            return Admission::Deferred;
        }
        Admission::Admitted
    }

    /// Apply the decision for a submission of `priority`: on
    /// [`Admission::Admitted`] the depth is incremented and the shedding latch
    /// is recomputed; otherwise the gate is unchanged. Returns the decision.
    pub fn offer(&mut self, priority: Priority) -> Admission {
        let decision = self.peek(priority);
        if decision.is_admitted() {
            self.depth += 1;
            self.recompute_shedding();
        }
        decision
    }

    /// Record that one admitted item finished, decrementing the depth and
    /// recomputing the shedding latch. Saturates at `0`.
    pub fn complete(&mut self) {
        self.complete_many(1);
    }

    /// Record that `n` admitted items finished, decrementing the depth and
    /// recomputing the shedding latch. Saturates at `0`.
    pub fn complete_many(&mut self, n: usize) {
        self.depth = self.depth.saturating_sub(n);
        self.recompute_shedding();
    }

    /// Recompute the hysteresis latch: arm shedding at or above
    /// `high_watermark`, disarm it at or below `low_watermark`, and otherwise
    /// hold the current state.
    fn recompute_shedding(&mut self) {
        if self.depth >= self.limits.high_watermark {
            self.shedding = true;
        } else if self.depth <= self.limits.low_watermark {
            self.shedding = false;
        }
    }
}
