//! Compose many timed animations into one coordinated schedule.
//!
//! A [`Choreography`] is an ordered set of [`Track`]s. Each track has a start
//! offset, a duration and an [`Easing`], and reports an eased `0.0..=1.0`
//! progress for a given global time. This is the building block for the three
//! classic composition patterns:
//!
//! * **parallel** — every track starts at the same time
//!   ([`Choreography::parallel`]);
//! * **sequence** — each track starts when the previous one ends
//!   ([`Choreography::sequence`], [`Choreography::then`]);
//! * **stagger** — tracks of equal duration start at a fixed cadence, so a
//!   list animates in like a cascade ([`Choreography::stagger`]).
//!
//! All timing is pure `+ - * /` arithmetic (no transcendental functions), so
//! the schedule is fully deterministic and `no_std`-friendly. The easing of a
//! track is applied by [`Easing::sample`].
//!
//! # Example
//!
//! ```
//! use prism_ui_anim::{Choreography, Easing};
//!
//! // Three items appearing 100 ms apart, each a 200 ms ease-out.
//! let choreo = Choreography::stagger(3, 0.2, 0.1, Easing::EaseOut);
//! assert_eq!(choreo.len(), 3);
//! assert!((choreo.duration() - 0.4).abs() < 1e-6); // 0.2 (last start) + 0.2
//!
//! // At t = 0 only the first item has started.
//! assert_eq!(choreo.progress(0, 0.0), 0.0);
//! assert_eq!(choreo.progress(1, 0.0), 0.0);
//! assert!(!choreo.is_complete(0.0));
//! ```

use alloc::vec::Vec;

use crate::easing::Easing;

/// One timed, eased track within a [`Choreography`].
///
/// A track is defined by when it `start`s, how long it lasts (`duration`) and
/// the `easing` applied to its normalized time.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Track {
    start: f32,
    duration: f32,
    easing: Easing,
}

impl Track {
    /// Creates a track beginning at `start` seconds, lasting `duration`
    /// seconds, shaped by `easing`.
    ///
    /// A non-positive `duration` yields a track that is instantly complete
    /// once `start` is reached.
    #[must_use]
    pub fn new(start: f32, duration: f32, easing: Easing) -> Self {
        Self {
            start,
            duration,
            easing,
        }
    }

    /// The global time at which this track begins.
    #[must_use]
    pub fn start(&self) -> f32 {
        self.start
    }

    /// The track's duration in seconds.
    #[must_use]
    pub fn duration(&self) -> f32 {
        self.duration
    }

    /// The global time at which this track finishes (`start + duration`).
    #[must_use]
    pub fn end(&self) -> f32 {
        self.start + self.duration
    }

    /// The easing curve applied to this track.
    #[must_use]
    pub fn easing(&self) -> Easing {
        self.easing
    }

    /// The raw, un-eased progress of the track at global time `t`.
    ///
    /// Returns `0.0` before the track starts and `1.0` once it has ended. A
    /// zero (or negative) duration track jumps straight to `1.0` at `start`.
    #[must_use]
    pub fn linear_progress(&self, t: f32) -> f32 {
        if self.duration <= 0.0 {
            return if t >= self.start { 1.0 } else { 0.0 };
        }
        if t <= self.start {
            return 0.0;
        }
        if t >= self.end() {
            return 1.0;
        }
        (t - self.start) / self.duration
    }

    /// The eased progress of the track at global time `t`, in `0.0..=1.0`.
    #[must_use]
    pub fn progress(&self, t: f32) -> f32 {
        self.easing.sample(self.linear_progress(t))
    }

    /// Whether the track has finished by global time `t`.
    #[must_use]
    pub fn is_complete(&self, t: f32) -> bool {
        t >= self.end()
    }
}

/// An ordered schedule of eased [`Track`]s sharing one global clock.
///
/// Build one with [`Choreography::parallel`], [`Choreography::sequence`] or
/// [`Choreography::stagger`], or assemble it imperatively with
/// [`Choreography::new`] + [`Choreography::push`] / [`Choreography::then`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Choreography {
    tracks: Vec<Track>,
}

impl Choreography {
    /// Creates an empty choreography.
    #[must_use]
    pub fn new() -> Self {
        Self { tracks: Vec::new() }
    }

    /// Adds a fully specified [`Track`] and returns `self` for chaining.
    #[must_use]
    pub fn push(mut self, track: Track) -> Self {
        self.tracks.push(track);
        self
    }

    /// Appends a track that starts exactly when the current schedule ends,
    /// extending it into a sequence.
    ///
    /// With an empty schedule the new track starts at `0.0`.
    #[must_use]
    pub fn then(mut self, duration: f32, easing: Easing) -> Self {
        let start = self.duration();
        self.tracks.push(Track::new(start, duration, easing));
        self
    }

    /// `count` tracks that all start at `0.0` and share `duration` and
    /// `easing`.
    #[must_use]
    pub fn parallel(count: usize, duration: f32, easing: Easing) -> Self {
        let mut tracks = Vec::with_capacity(count);
        for _ in 0..count {
            tracks.push(Track::new(0.0, duration, easing));
        }
        Self { tracks }
    }

    /// A back-to-back sequence from `(duration, easing)` pairs; each track
    /// starts when the previous one ends.
    #[must_use]
    pub fn sequence(segments: &[(f32, Easing)]) -> Self {
        let mut tracks = Vec::with_capacity(segments.len());
        let mut cursor = 0.0_f32;
        for &(duration, easing) in segments {
            tracks.push(Track::new(cursor, duration, easing));
            cursor += duration;
        }
        Self { tracks }
    }

    /// `count` equal-duration tracks that begin `stagger` seconds apart.
    ///
    /// Track `i` starts at `i * stagger`; the whole schedule finishes at
    /// `(count - 1) * stagger + duration`.
    #[must_use]
    pub fn stagger(count: usize, duration: f32, stagger: f32, easing: Easing) -> Self {
        let mut tracks = Vec::with_capacity(count);
        let mut start = 0.0_f32;
        for _ in 0..count {
            tracks.push(Track::new(start, duration, easing));
            start += stagger;
        }
        Self { tracks }
    }

    /// The number of tracks.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tracks.len()
    }

    /// Whether the schedule has no tracks.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tracks.is_empty()
    }

    /// The tracks, in insertion order.
    #[must_use]
    pub fn tracks(&self) -> &[Track] {
        &self.tracks
    }

    /// The total duration: the latest [`Track::end`] across all tracks, or
    /// `0.0` when empty.
    #[must_use]
    pub fn duration(&self) -> f32 {
        let mut max_end = 0.0_f32;
        for track in &self.tracks {
            let end = track.end();
            if end > max_end {
                max_end = end;
            }
        }
        max_end
    }

    /// The eased progress of track `index` at global time `t`.
    ///
    /// Returns `0.0` for an out-of-range index so callers can sample defensively.
    #[must_use]
    pub fn progress(&self, index: usize, t: f32) -> f32 {
        match self.tracks.get(index) {
            Some(track) => track.progress(t),
            None => 0.0,
        }
    }

    /// Collects the eased progress of every track at global time `t`.
    #[must_use]
    pub fn sample_all(&self, t: f32) -> Vec<f32> {
        self.tracks.iter().map(|track| track.progress(t)).collect()
    }

    /// Whether every track has finished by global time `t`.
    ///
    /// An empty schedule is complete at any time.
    #[must_use]
    pub fn is_complete(&self, t: f32) -> bool {
        t >= self.duration()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-6;

    fn approx(a: f32, b: f32) {
        assert!((a - b).abs() < EPS, "expected {a} ~= {b}");
    }

    #[test]
    fn track_progress_clamps_outside_its_window() {
        let track = Track::new(1.0, 2.0, Easing::Linear);
        approx(track.linear_progress(0.0), 0.0);
        approx(track.linear_progress(1.0), 0.0);
        approx(track.linear_progress(2.0), 0.5);
        approx(track.linear_progress(3.0), 1.0);
        approx(track.linear_progress(9.0), 1.0);
        assert!(track.is_complete(3.0));
        assert!(!track.is_complete(2.999));
    }

    #[test]
    fn zero_duration_track_is_instant_at_start() {
        let track = Track::new(0.5, 0.0, Easing::Linear);
        approx(track.linear_progress(0.4), 0.0);
        approx(track.linear_progress(0.5), 1.0);
        assert!(track.is_complete(0.5));
    }

    #[test]
    fn parallel_tracks_share_one_window() {
        let choreo = Choreography::parallel(3, 1.0, Easing::Linear);
        approx(choreo.duration(), 1.0);
        for i in 0..3 {
            approx(choreo.progress(i, 0.5), 0.5);
        }
        assert!(choreo.is_complete(1.0));
    }

    #[test]
    fn sequence_chains_back_to_back() {
        let choreo = Choreography::sequence(&[
            (1.0, Easing::Linear),
            (2.0, Easing::Linear),
            (1.0, Easing::Linear),
        ]);
        approx(choreo.duration(), 4.0);
        // At t = 1.5 the first track is done, the second is 25% in, the third
        // has not started.
        approx(choreo.progress(0, 1.5), 1.0);
        approx(choreo.progress(1, 1.5), 0.25);
        approx(choreo.progress(2, 1.5), 0.0);
    }

    #[test]
    fn then_appends_after_current_end() {
        let choreo = Choreography::new()
            .then(1.0, Easing::Linear)
            .then(1.0, Easing::Linear);
        approx(choreo.duration(), 2.0);
        approx(choreo.tracks()[1].start(), 1.0);
    }

    #[test]
    fn stagger_offsets_each_item() {
        let choreo = Choreography::stagger(4, 0.2, 0.1, Easing::Linear);
        approx(choreo.tracks()[0].start(), 0.0);
        approx(choreo.tracks()[1].start(), 0.1);
        approx(choreo.tracks()[3].start(), 0.3);
        // Last item starts at 0.3 and lasts 0.2 -> total 0.5.
        approx(choreo.duration(), 0.5);
    }

    #[test]
    fn sample_all_returns_per_track_progress() {
        let choreo = Choreography::stagger(3, 1.0, 1.0, Easing::Linear);
        // At t = 1.5: item0 done, item1 half, item2 not started.
        let samples = choreo.sample_all(1.5);
        approx(samples[0], 1.0);
        approx(samples[1], 0.5);
        approx(samples[2], 0.0);
    }

    #[test]
    fn out_of_range_index_is_zero() {
        let choreo = Choreography::parallel(1, 1.0, Easing::Linear);
        approx(choreo.progress(5, 0.5), 0.0);
    }

    #[test]
    fn empty_choreography_is_complete_and_zero_duration() {
        let choreo = Choreography::new();
        assert!(choreo.is_empty());
        approx(choreo.duration(), 0.0);
        assert!(choreo.is_complete(0.0));
    }
}
