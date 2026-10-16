//! Bounded frames-in-flight pacing for the render runtime.
//!
//! A GPU runs behind the CPU: a submission does not finish the moment its
//! command buffer is handed to the queue. Recording an unbounded number of
//! frames back-to-back therefore grows the CPU/GPU gap without limit, inflating
//! input latency and the memory the driver holds live. The fix is the standard
//! *frames in flight* bound: never let more than `N` submissions be outstanding
//! at once, and block the CPU on the oldest one before a new frame may start.
//!
//! [`FramePacer`] is the bookkeeping half of that policy and nothing else: it
//! holds a bounded, oldest-first queue of opaque completion tokens and answers
//! "may I start a new frame, or must I first wait on this token?". The tokens
//! are supplied by the backend (see [`FrameSync`](crate::FrameSync)); the pacer
//! itself is pure, allocation-light, and has no notion of a GPU, which is what
//! lets the bound be unit-tested on hardware-free CI.

use alloc::collections::VecDeque;

/// The default number of frames the render runtime keeps in flight.
///
/// Two is the conventional double-buffered floor: it hides the latency of one
/// frame behind the recording of the next while keeping at most one frame of
/// *extra* latency under load.
pub const DEFAULT_FRAMES_IN_FLIGHT: usize = 2;

/// A bounded, oldest-first pipeline of in-flight frame completion tokens.
///
/// The pacer enforces `in_flight <= max_in_flight`. Pair [`FramePacer::begin_frame`]
/// (called before submitting a frame) with [`FramePacer::record`] (called with
/// the token the submission returned):
///
/// ```
/// # use prism_render_runtime::FramePacer;
/// let mut pacer = FramePacer::new(2);
/// // Frames 0 and 1 start without blocking.
/// assert!(pacer.begin_frame().is_none());
/// pacer.record(0_u64);
/// assert!(pacer.begin_frame().is_none());
/// pacer.record(1_u64);
/// // Frame 2 must first retire frame 0.
/// assert_eq!(pacer.begin_frame(), Some(0));
/// pacer.record(2_u64);
/// assert_eq!(pacer.in_flight(), 2);
/// ```
pub struct FramePacer<Token> {
    /// The hard upper bound on outstanding frames; never zero.
    max_in_flight: usize,
    /// Outstanding completion tokens, oldest at the front.
    outstanding: VecDeque<Token>,
}

impl<Token> FramePacer<Token> {
    /// Creates a pacer that allows at most `max_in_flight` outstanding frames.
    ///
    /// A `max_in_flight` of zero is clamped to one: a pipeline that never
    /// permits a frame to be in flight could not make progress.
    #[must_use]
    pub fn new(max_in_flight: usize) -> Self {
        Self {
            max_in_flight: max_in_flight.max(1),
            outstanding: VecDeque::new(),
        }
    }

    /// The configured upper bound on outstanding frames.
    #[must_use]
    pub const fn max_in_flight(&self) -> usize {
        self.max_in_flight
    }

    /// How many frames are currently outstanding.
    #[must_use]
    pub fn in_flight(&self) -> usize {
        self.outstanding.len()
    }

    /// Whether the pipeline has reached its configured bound.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.outstanding.len() >= self.max_in_flight
    }

    /// Opens a frame, handing back the token that must complete first.
    ///
    /// Returns `None` when the pipeline has room and the caller may proceed
    /// immediately. When the pipeline is full, this removes and returns the
    /// *oldest* outstanding token — the frame at exactly `N − max` — and the
    /// caller must block on it before submitting the new frame. Pair every
    /// call that returns `None` or a retired token with one [`Self::record`]
    /// of the frame's own token so the bound keeps holding.
    pub fn begin_frame(&mut self) -> Option<Token> {
        if self.is_full() {
            self.outstanding.pop_front()
        } else {
            None
        }
    }

    /// Records the completion token of the frame just submitted.
    pub fn record(&mut self, token: Token) {
        self.outstanding.push_back(token);
    }

    /// Restores the token removed by `begin_frame` when completion failed.
    /// No replacement frame may be recorded before this token is confirmed.
    pub(crate) fn restore_oldest(&mut self, token: Token) {
        self.outstanding.push_front(token);
    }

    /// Drops oldest tokens that the caller has independently confirmed as
    /// complete. Completion is queue-ordered, so removing only from the front
    /// preserves the pacing order and avoids retaining stale tokens.
    pub(crate) fn discard_while(&mut self, mut is_complete: impl FnMut(&Token) -> bool) {
        while self
            .outstanding
            .front()
            .is_some_and(|token| is_complete(token))
        {
            self.outstanding.pop_front();
        }
    }

    /// Removes and returns every outstanding token, oldest first.
    ///
    /// Used by the runtime's idle/shutdown path to hand the backend a complete
    /// list of submissions to wait on and to leave the pipeline empty.
    pub fn drain(&mut self) -> impl Iterator<Item = Token> {
        self.outstanding.drain(..)
    }
}

impl<Token> Default for FramePacer<Token> {
    fn default() -> Self {
        Self::new(DEFAULT_FRAMES_IN_FLIGHT)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_two_frames_in_flight() {
        let pacer: FramePacer<u32> = FramePacer::default();
        assert_eq!(pacer.max_in_flight(), DEFAULT_FRAMES_IN_FLIGHT);
        assert_eq!(pacer.in_flight(), 0);
    }

    #[test]
    fn zero_max_is_clamped_to_one() {
        let pacer: FramePacer<u32> = FramePacer::new(0);
        assert_eq!(pacer.max_in_flight(), 1);
    }

    #[test]
    fn retires_only_when_full() {
        let mut pacer: FramePacer<u32> = FramePacer::new(3);
        assert!(pacer.begin_frame().is_none(), "room for frame 0");
        pacer.record(0);
        assert!(pacer.begin_frame().is_none(), "room for frame 1");
        pacer.record(1);
        assert!(pacer.begin_frame().is_none(), "room for frame 2");
        pacer.record(2);
        assert!(pacer.is_full());
        // Frame 3 may only start after the oldest (frame 0) retires.
        assert_eq!(pacer.begin_frame(), Some(0));
    }

    #[test]
    fn never_exceeds_the_bound_over_a_long_run() {
        const MAX: usize = 2;
        let mut pacer: FramePacer<u64> = FramePacer::new(MAX);
        let mut retired = Vec::new();
        for frame in 0..100_u64 {
            if let Some(token) = pacer.begin_frame() {
                retired.push(token);
            }
            pacer.record(frame);
            assert!(
                pacer.in_flight() <= MAX,
                "frame {frame} left {} outstanding, bound is {MAX}",
                pacer.in_flight()
            );
        }
        // Every frame except the final MAX must have been retired, oldest first.
        let expected: Vec<u64> = (0..98).collect();
        assert_eq!(retired, expected);
        assert_eq!(pacer.in_flight(), MAX);
    }

    #[test]
    fn drain_empties_the_pipeline_in_order() {
        let mut pacer: FramePacer<u32> = FramePacer::new(4);
        pacer.record(10);
        pacer.record(11);
        pacer.record(12);
        assert_eq!(pacer.drain().collect::<Vec<_>>(), vec![10, 11, 12]);
        assert_eq!(pacer.in_flight(), 0);
    }
}
