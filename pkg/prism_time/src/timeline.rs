//! Addressable, seekable time sources: [`Timeline`] and [`Sequencer`].
//!
//! A gameplay clock advances only forward and only in real steps. Cutscenes,
//! sequencers, and tooling need something different: a *position* on a fixed
//! duration that can be scrubbed, looped, reversed, and queried for progress.
//! [`Timeline`] is that source. It owns a `position` in `[0, duration]`, a
//! playback `speed`, a `playing` flag, and a [`PlaybackMode`], and it reports a
//! [`TimelineTick`] describing how far it moved and whether it wrapped or
//! finished.
//!
//! [`Sequencer`] layers fixed-capacity, sorted [`TimelineMarker`]s on top so a
//! cutscene can fire keyframe events as the playhead crosses them. It is
//! const-generic over the marker capacity and allocates nothing.

use crate::Duration;

/// How a [`Timeline`] behaves when the playhead reaches a boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PlaybackMode {
    /// Play once to the end (or start, when reversed) and stop there.
    #[default]
    Once,
    /// Wrap around to the opposite boundary and keep playing.
    Loop,
    /// Reverse direction at each boundary (back-and-forth).
    PingPong,
}

/// The outcome of advancing a [`Timeline`] by one step.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct TimelineTick {
    /// Signed-magnitude distance the playhead actually moved this step. Always
    /// non-negative; combine with the timeline's direction if you need a sign.
    pub applied: Duration,
    /// How many times the playhead crossed a boundary this step (`0` for the
    /// common sub-step advance). [`PlaybackMode::Once`] reports at most `1`.
    pub wrapped: u32,
    /// Whether a [`PlaybackMode::Once`] timeline came to rest at a boundary.
    pub finished: bool,
}

/// A scrubbable position on a fixed duration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Timeline {
    position: Duration,
    prev: Duration,
    duration: Duration,
    speed: f64,
    playing: bool,
    mode: PlaybackMode,
    /// `+1` playing forward, `-1` reversed (only ever `-1` under
    /// [`PlaybackMode::PingPong`]).
    direction: i8,
}

impl Timeline {
    /// Create a playing timeline of length `duration` at position zero.
    ///
    /// A zero `duration` is degenerate (every advance immediately finishes);
    /// prefer a positive duration for a real sequence.
    #[inline]
    pub fn new(duration: Duration) -> Self {
        Self::with_mode(duration, PlaybackMode::Once)
    }

    /// Create a playing timeline with an explicit [`PlaybackMode`].
    #[inline]
    pub fn with_mode(duration: Duration, mode: PlaybackMode) -> Self {
        Self {
            position: Duration::ZERO,
            prev: Duration::ZERO,
            duration,
            speed: 1.0,
            playing: true,
            mode,
            direction: 1,
        }
    }

    /// The playhead position in `[0, duration]`.
    #[inline]
    pub fn position(&self) -> Duration {
        self.position
    }

    /// The playhead position before the last [`advance`](Self::advance).
    #[inline]
    pub fn previous_position(&self) -> Duration {
        self.prev
    }

    /// The total length of the timeline.
    #[inline]
    pub fn duration(&self) -> Duration {
        self.duration
    }

    /// The playback mode.
    #[inline]
    pub fn mode(&self) -> PlaybackMode {
        self.mode
    }

    /// Normalised progress in `[0, 1]`. A zero-length timeline reports `1.0`.
    #[inline]
    pub fn progress(&self) -> f64 {
        if self.duration.is_zero() {
            1.0
        } else {
            (self.position.as_secs_f64() / self.duration.as_secs_f64()).clamp(0.0, 1.0)
        }
    }

    /// Whether the timeline is currently playing.
    #[inline]
    pub fn is_playing(&self) -> bool {
        self.playing
    }

    /// Whether a [`PlaybackMode::Once`] timeline has come to rest at a boundary.
    ///
    /// Looping and ping-pong timelines never finish, so this is always `false`
    /// for them.
    #[inline]
    pub fn is_finished(&self) -> bool {
        if self.mode != PlaybackMode::Once {
            return false;
        }
        if self.direction >= 0 {
            self.position >= self.duration
        } else {
            self.position.is_zero()
        }
    }

    /// Start (or resume) playback.
    #[inline]
    pub fn play(&mut self) {
        self.playing = true;
    }

    /// Pause playback, holding the current position.
    #[inline]
    pub fn pause(&mut self) {
        self.playing = false;
    }

    /// Stop playback and rewind to the start (forward direction restored).
    #[inline]
    pub fn stop(&mut self) {
        self.playing = false;
        self.position = Duration::ZERO;
        self.prev = Duration::ZERO;
        self.direction = 1;
    }

    /// The playback speed multiplier (`1.0` = real time). Negative values are
    /// clamped to `0.0` on assignment.
    #[inline]
    pub fn speed(&self) -> f64 {
        self.speed
    }

    /// Set the playback speed. Negative or non-finite values are rejected
    /// (negative clamps to `0.0`, non-finite is ignored); reverse playback is
    /// expressed through [`PlaybackMode::PingPong`], not a negative speed.
    #[inline]
    pub fn set_speed(&mut self, speed: f64) {
        if speed.is_finite() {
            self.speed = speed.max(0.0);
        }
    }

    /// Jump the playhead to `position`, clamped to `[0, duration]`. Resets the
    /// ping-pong direction to forward and does not change the playing state.
    #[inline]
    pub fn seek(&mut self, position: Duration) {
        let clamped = if position > self.duration {
            self.duration
        } else {
            position
        };
        self.position = clamped;
        self.prev = clamped;
        self.direction = 1;
    }

    /// Jump to a fractional progress in `[0, 1]` of the duration.
    #[inline]
    pub fn seek_progress(&mut self, progress: f64) {
        let p = progress.clamp(0.0, 1.0);
        self.seek(self.duration.mul_f64(p));
    }

    /// Advance the playhead by `real_delta` of wall-clock time, scaled by the
    /// playback speed and interpreted through the [`PlaybackMode`].
    ///
    /// Returns a [`TimelineTick`] with the magnitude moved, the boundary-cross
    /// count, and whether a [`PlaybackMode::Once`] timeline finished. A paused
    /// timeline, a zero speed, or a zero-length duration reports a zero tick
    /// (finished immediately for the degenerate zero-length case).
    #[inline]
    pub fn advance(&mut self, real_delta: Duration) -> TimelineTick {
        self.prev = self.position;
        if !self.playing || self.speed == 0.0 {
            return TimelineTick {
                applied: Duration::ZERO,
                wrapped: 0,
                finished: self.is_finished(),
            };
        }
        if self.duration.is_zero() {
            self.playing = self.mode != PlaybackMode::Once;
            return TimelineTick {
                applied: Duration::ZERO,
                wrapped: 0,
                finished: self.mode == PlaybackMode::Once,
            };
        }

        let step = if self.speed == 1.0 {
            real_delta
        } else {
            real_delta.mul_f64(self.speed)
        };

        match self.mode {
            PlaybackMode::Once => self.advance_once(step),
            PlaybackMode::Loop => self.advance_loop(step),
            PlaybackMode::PingPong => self.advance_ping_pong(step),
        }
    }

    #[inline]
    fn advance_once(&mut self, step: Duration) -> TimelineTick {
        let start = self.position;
        let end = self.position.saturating_add(step);
        if end >= self.duration {
            self.position = self.duration;
            self.playing = false;
            TimelineTick {
                applied: self.duration.saturating_sub(start),
                wrapped: 1,
                finished: true,
            }
        } else {
            self.position = end;
            TimelineTick {
                applied: step,
                wrapped: 0,
                finished: false,
            }
        }
    }

    #[inline]
    fn advance_loop(&mut self, step: Duration) -> TimelineTick {
        let mut remaining = step;
        let mut wrapped = 0;
        // `duration` is non-zero here, so each wrap consumes a positive amount
        // and the loop terminates.
        loop {
            let room = self.duration.saturating_sub(self.position);
            if remaining < room {
                self.position = self.position.saturating_add(remaining);
                break;
            }
            remaining = remaining.saturating_sub(room);
            self.position = Duration::ZERO;
            wrapped += 1;
        }
        TimelineTick {
            applied: step,
            wrapped,
            finished: false,
        }
    }

    #[inline]
    fn advance_ping_pong(&mut self, step: Duration) -> TimelineTick {
        let mut remaining = step;
        let mut wrapped = 0;
        loop {
            let room = if self.direction >= 0 {
                self.duration.saturating_sub(self.position)
            } else {
                self.position
            };
            if remaining < room {
                self.position = if self.direction >= 0 {
                    self.position.saturating_add(remaining)
                } else {
                    self.position.saturating_sub(remaining)
                };
                break;
            }
            remaining = remaining.saturating_sub(room);
            // Land exactly on the boundary and flip.
            self.position = if self.direction >= 0 {
                self.duration
            } else {
                Duration::ZERO
            };
            self.direction = -self.direction;
            wrapped += 1;
        }
        TimelineTick {
            applied: step,
            wrapped,
            finished: false,
        }
    }
}

/// A keyframe marker on a [`Sequencer`] timeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TimelineMarker {
    /// The playhead position at which this marker fires.
    pub time: Duration,
    /// A caller-defined identifier (event id, cue number, ...).
    pub id: u32,
}

/// A [`Timeline`] with up to `N` sorted [`TimelineMarker`]s that fire as the
/// playhead crosses them.
///
/// The marker array is fixed-capacity (`N`) and kept sorted by time, so the
/// sequencer allocates nothing and is `no_std`-friendly.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sequencer<const N: usize> {
    timeline: Timeline,
    markers: [TimelineMarker; N],
    count: usize,
}

impl<const N: usize> Sequencer<N> {
    /// Create an empty sequencer wrapping `timeline`.
    #[inline]
    pub fn new(timeline: Timeline) -> Self {
        Self {
            timeline,
            markers: [TimelineMarker {
                time: Duration::ZERO,
                id: 0,
            }; N],
            count: 0,
        }
    }

    /// Shared access to the underlying timeline.
    #[inline]
    pub fn timeline(&self) -> &Timeline {
        &self.timeline
    }

    /// Mutable access to the underlying timeline. Seeking directly through this
    /// bypasses marker firing; use [`advance_with`](Self::advance_with) to fire
    /// markers.
    #[inline]
    pub fn timeline_mut(&mut self) -> &mut Timeline {
        &mut self.timeline
    }

    /// The markers registered so far, in ascending time order.
    #[inline]
    pub fn markers(&self) -> &[TimelineMarker] {
        &self.markers[..self.count]
    }

    /// How many markers are registered.
    #[inline]
    pub fn marker_count(&self) -> usize {
        self.count
    }

    /// Whether the marker buffer is full.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.count == N
    }

    /// Register a marker at `time` with identifier `id`, keeping the buffer
    /// sorted by time (stable among equal times).
    ///
    /// # Errors
    /// Returns [`SequencerFull`] when the fixed-capacity buffer is already full.
    #[inline]
    pub fn try_add_marker(&mut self, id: u32, time: Duration) -> Result<(), SequencerFull> {
        if self.count == N {
            return Err(SequencerFull);
        }
        // Insertion sort: find the first slot with a strictly greater time.
        let mut i = self.count;
        while i > 0 && self.markers[i - 1].time > time {
            self.markers[i] = self.markers[i - 1];
            i -= 1;
        }
        self.markers[i] = TimelineMarker { time, id };
        self.count += 1;
        Ok(())
    }

    /// Advance the timeline and invoke `on_marker` for every marker crossed by
    /// the playhead this step, in playback order.
    ///
    /// Markers are fired for the half-open interval the playhead swept. For a
    /// non-wrapping forward step from `prev` to `now`, that is `(prev, now]`.
    /// When the step wraps once (`wrapped == 1`, [`PlaybackMode::Loop`]) the
    /// swept region is `(prev, duration]` followed by `(0, now]`. Callers that
    /// need every marker on a multi-wrap step should keep the per-step advance
    /// smaller than the timeline duration (the documented single-wrap
    /// invariant); larger steps still advance the playhead correctly but only
    /// fire the final lap's markers.
    #[inline]
    pub fn advance_with<F: FnMut(TimelineMarker)>(
        &mut self,
        real_delta: Duration,
        mut on_marker: F,
    ) -> TimelineTick {
        let prev = self.timeline.position();
        let tick = self.timeline.advance(real_delta);
        let now = self.timeline.position();

        if tick.wrapped == 0 {
            // Simple sub-step. Forward or (ping-pong) reverse.
            if now >= prev {
                self.fire_range(prev, now, true, &mut on_marker);
            } else {
                self.fire_range(now, prev, false, &mut on_marker);
            }
        } else {
            // Single documented wrap: tail of this lap, then head of the next.
            let dur = self.timeline.duration();
            self.fire_range(prev, dur, true, &mut on_marker);
            self.fire_range(Duration::ZERO, now, true, &mut on_marker);
        }
        tick
    }

    /// Fire markers whose time is in the half-open interval `(lo, hi]`,
    /// iterating ascending when `forward`, descending otherwise.
    #[inline]
    fn fire_range<F: FnMut(TimelineMarker)>(
        &self,
        lo: Duration,
        hi: Duration,
        forward: bool,
        on_marker: &mut F,
    ) {
        if forward {
            for m in &self.markers[..self.count] {
                if m.time > lo && m.time <= hi {
                    on_marker(*m);
                }
            }
        } else {
            for m in self.markers[..self.count].iter().rev() {
                if m.time > lo && m.time <= hi {
                    on_marker(*m);
                }
            }
        }
    }
}

/// Error returned by [`Sequencer::try_add_marker`] when the buffer is full.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SequencerFull;

impl core::fmt::Display for SequencerFull {
    #[inline]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("sequencer marker buffer is full")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for SequencerFull {}
