//! Multi-tap (double-tap, triple-tap) recognition.
//!
//! A [`MultiTapRecognizer`] fires once a configured number of quick taps land
//! close together — the gesture behind double-click selection and double-tap
//! zoom. It is a one-shot recognizer in the same family as
//! [`TapRecognizer`](crate::gesture::TapRecognizer): it reports
//! [`GestureState::Accepted`] on the final qualifying tap,
//! [`GestureState::Rejected`] once the multi-tap can no longer complete, and
//! [`GestureState::Possible`] while it is still gathering taps. Call
//! [`GestureRecognizer::reset`] to detect another multi-tap with the same
//! instance.
//!
//! # Qualifying rules
//!
//! A tap counts when a press and release stay within `slop` logical pixels of
//! the press point. Consecutive taps must start within `max_interval_ms` of the
//! previous release; a slower follow-up, movement past `slop`, or a cancel
//! fails the whole gesture. Because the inter-tap window is time-based, drive
//! [`GestureRecognizer::tick`] so the recognizer can reject a stalled sequence
//! even when no further pointer events arrive.
//!
//! ```
//! use prism_ui_input::{GestureRecognizer, GestureState, MultiTapRecognizer};
//! use prism_ui_input::{NodeId, PointerEvent, PointerId, PointerKind};
//! use prism_ui_input::geometry::Point;
//!
//! let mut dbl = MultiTapRecognizer::new(NodeId::new(1), 2, 8.0, 300);
//! let p = Point::new(10.0, 10.0);
//! let tap = |kind, t| PointerEvent::new(PointerId::new(1), kind, p, t);
//!
//! assert_eq!(dbl.handle(&tap(PointerKind::Down, 0)), GestureState::Possible);
//! assert_eq!(dbl.handle(&tap(PointerKind::Up, 10)), GestureState::Possible);
//! assert_eq!(dbl.handle(&tap(PointerKind::Down, 60)), GestureState::Possible);
//! assert_eq!(dbl.handle(&tap(PointerKind::Up, 70)), GestureState::Accepted);
//! ```

use crate::event::{PointerEvent, PointerKind};
use crate::geometry::{distance_squared, Point};
use crate::gesture::{GestureRecognizer, GestureState};
use crate::NodeId;

/// Recognizes a run of `required_taps` quick taps within `slop` and
/// `max_interval_ms` of each other.
pub struct MultiTapRecognizer {
    target: NodeId,
    required_taps: u32,
    slop: f32,
    max_interval_ms: u64,
    /// Number of qualifying taps gathered so far.
    count: u32,
    /// Press point of the tap currently in progress, if the pointer is down.
    down_origin: Option<Point<f32>>,
    /// Release timestamp of the most recent qualifying tap.
    last_up_ms: Option<u64>,
    state: GestureState,
}

impl MultiTapRecognizer {
    /// Creates a recognizer that accepts after `required_taps` qualifying taps.
    ///
    /// `required_taps` is clamped to at least 1 (a value of 2 yields a
    /// double-tap). `slop` is the per-tap movement tolerance in logical pixels
    /// and `max_interval_ms` is the longest gap allowed between a release and
    /// the next press.
    #[must_use]
    pub fn new(target: NodeId, required_taps: u32, slop: f32, max_interval_ms: u64) -> Self {
        Self {
            target,
            required_taps: required_taps.max(1),
            slop,
            max_interval_ms,
            count: 0,
            down_origin: None,
            last_up_ms: None,
            state: GestureState::Possible,
        }
    }

    /// Returns the number of qualifying taps gathered so far.
    #[must_use]
    pub fn count(&self) -> u32 {
        self.count
    }

    /// Returns the tap count required to accept.
    #[must_use]
    pub fn required_taps(&self) -> u32 {
        self.required_taps
    }

    /// Returns `true` when `point` is within `slop` of `origin`.
    fn within_slop(&self, origin: Point<f32>, point: Point<f32>) -> bool {
        distance_squared(origin, point) <= self.slop * self.slop
    }

    /// Returns `true` when the gap from the previous release to `now_ms`
    /// exceeds the inter-tap window.
    fn interval_elapsed(&self, now_ms: u64) -> bool {
        match self.last_up_ms {
            Some(prev) => now_ms.saturating_sub(prev) > self.max_interval_ms,
            None => false,
        }
    }
}

impl GestureRecognizer for MultiTapRecognizer {
    fn target(&self) -> NodeId {
        self.target
    }

    fn handle(&mut self, event: &PointerEvent) -> GestureState {
        if self.state != GestureState::Possible {
            return self.state;
        }
        match event.kind {
            PointerKind::Down => {
                // A press that arrives after the inter-tap window fails the
                // in-progress sequence rather than silently restarting it.
                if self.interval_elapsed(event.timestamp_ms) {
                    self.state = GestureState::Rejected;
                } else {
                    self.down_origin = Some(event.position);
                }
            }
            PointerKind::Move => {
                if let Some(origin) = self.down_origin
                    && !self.within_slop(origin, event.position)
                {
                    self.state = GestureState::Rejected;
                }
            }
            PointerKind::Up => match self.down_origin.take() {
                Some(origin) if self.within_slop(origin, event.position) => {
                    self.count += 1;
                    self.last_up_ms = Some(event.timestamp_ms);
                    if self.count >= self.required_taps {
                        self.state = GestureState::Accepted;
                    }
                }
                // Release without a matching press, or past the slop radius.
                _ => self.state = GestureState::Rejected,
            },
            PointerKind::Cancel => self.state = GestureState::Rejected,
            PointerKind::Enter | PointerKind::Leave => {}
        }
        self.state
    }

    fn tick(&mut self, now_ms: u64) -> GestureState {
        // Only the gap *between* taps expires on a tick; a long hold of the tap
        // currently in progress is governed by `slop`, not by this window.
        if self.state == GestureState::Possible
            && self.down_origin.is_none()
            && self.interval_elapsed(now_ms)
        {
            self.state = GestureState::Rejected;
        }
        self.state
    }

    fn state(&self) -> GestureState {
        self.state
    }

    fn reset(&mut self) {
        self.count = 0;
        self.down_origin = None;
        self.last_up_ms = None;
        self.state = GestureState::Possible;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::PointerId;

    const SLOP: f32 = 8.0;
    const INTERVAL: u64 = 300;

    fn recognizer(taps: u32) -> MultiTapRecognizer {
        MultiTapRecognizer::new(NodeId::new(1), taps, SLOP, INTERVAL)
    }

    fn ev(kind: PointerKind, x: f32, y: f32, t: u64) -> PointerEvent {
        PointerEvent::new(PointerId::new(1), kind, Point::new(x, y), t)
    }

    /// Drives one complete tap (down then up) at `(x, y)` and returns the state
    /// reported after the release.
    fn tap(r: &mut MultiTapRecognizer, x: f32, y: f32, down_t: u64, up_t: u64) -> GestureState {
        r.handle(&ev(PointerKind::Down, x, y, down_t));
        r.handle(&ev(PointerKind::Up, x, y, up_t))
    }

    #[test]
    fn double_tap_accepts_on_second_release() {
        let mut r = recognizer(2);
        assert_eq!(tap(&mut r, 10.0, 10.0, 0, 10), GestureState::Possible);
        assert_eq!(r.count(), 1);
        assert_eq!(tap(&mut r, 11.0, 10.0, 60, 70), GestureState::Accepted);
        assert_eq!(r.count(), 2);
    }

    #[test]
    fn single_tap_stays_possible() {
        let mut r = recognizer(2);
        assert_eq!(tap(&mut r, 10.0, 10.0, 0, 10), GestureState::Possible);
    }

    #[test]
    fn triple_tap_needs_three_releases() {
        let mut r = recognizer(3);
        assert_eq!(tap(&mut r, 5.0, 5.0, 0, 5), GestureState::Possible);
        assert_eq!(tap(&mut r, 5.0, 5.0, 40, 45), GestureState::Possible);
        assert_eq!(tap(&mut r, 5.0, 5.0, 80, 85), GestureState::Accepted);
    }

    #[test]
    fn slow_second_tap_rejects_on_press() {
        let mut r = recognizer(2);
        assert_eq!(tap(&mut r, 10.0, 10.0, 0, 10), GestureState::Possible);
        // Second press arrives 400 ms after the first release (> INTERVAL).
        let s = r.handle(&ev(PointerKind::Down, 10.0, 10.0, 410));
        assert_eq!(s, GestureState::Rejected);
    }

    #[test]
    fn movement_during_tap_rejects() {
        let mut r = recognizer(2);
        r.handle(&ev(PointerKind::Down, 10.0, 10.0, 0));
        let s = r.handle(&ev(PointerKind::Move, 30.0, 10.0, 5));
        assert_eq!(s, GestureState::Rejected);
    }

    #[test]
    fn release_outside_slop_rejects() {
        let mut r = recognizer(2);
        r.handle(&ev(PointerKind::Down, 10.0, 10.0, 0));
        let s = r.handle(&ev(PointerKind::Up, 40.0, 40.0, 10));
        assert_eq!(s, GestureState::Rejected);
    }

    #[test]
    fn small_movement_within_slop_still_taps() {
        let mut r = recognizer(2);
        r.handle(&ev(PointerKind::Down, 10.0, 10.0, 0));
        // A sub-slop wobble keeps the tap alive.
        assert_eq!(
            r.handle(&ev(PointerKind::Move, 14.0, 10.0, 3)),
            GestureState::Possible
        );
        assert_eq!(
            r.handle(&ev(PointerKind::Up, 13.0, 11.0, 10)),
            GestureState::Possible
        );
        assert_eq!(r.count(), 1);
    }

    #[test]
    fn tick_rejects_stalled_sequence() {
        let mut r = recognizer(2);
        assert_eq!(tap(&mut r, 10.0, 10.0, 0, 10), GestureState::Possible);
        // No second tap arrives; the window lapses.
        assert_eq!(r.tick(10 + INTERVAL), GestureState::Possible);
        assert_eq!(r.tick(10 + INTERVAL + 1), GestureState::Rejected);
    }

    #[test]
    fn tick_does_not_reject_mid_press() {
        let mut r = recognizer(2);
        assert_eq!(tap(&mut r, 10.0, 10.0, 0, 10), GestureState::Possible);
        // Second press is in progress; a long hold must not time out here.
        r.handle(&ev(PointerKind::Down, 10.0, 10.0, 50));
        assert_eq!(r.tick(10_000), GestureState::Possible);
        assert_eq!(r.handle(&ev(PointerKind::Up, 10.0, 10.0, 10_050)), GestureState::Accepted);
    }

    #[test]
    fn cancel_rejects() {
        let mut r = recognizer(2);
        r.handle(&ev(PointerKind::Down, 10.0, 10.0, 0));
        assert_eq!(r.handle(&ev(PointerKind::Cancel, 10.0, 10.0, 5)), GestureState::Rejected);
    }

    #[test]
    fn reset_allows_redetection() {
        let mut r = recognizer(2);
        assert_eq!(tap(&mut r, 10.0, 10.0, 0, 10), GestureState::Possible);
        assert_eq!(tap(&mut r, 10.0, 10.0, 60, 70), GestureState::Accepted);
        r.reset();
        assert_eq!(r.count(), 0);
        assert_eq!(r.state(), GestureState::Possible);
        assert_eq!(tap(&mut r, 0.0, 0.0, 0, 10), GestureState::Possible);
        assert_eq!(tap(&mut r, 0.0, 0.0, 40, 50), GestureState::Accepted);
    }

    #[test]
    fn required_taps_clamped_to_one() {
        let mut r = MultiTapRecognizer::new(NodeId::new(1), 0, SLOP, INTERVAL);
        assert_eq!(r.required_taps(), 1);
        assert_eq!(tap(&mut r, 1.0, 1.0, 0, 5), GestureState::Accepted);
    }

    #[test]
    fn target_is_reported() {
        let r = recognizer(2);
        assert_eq!(r.target(), NodeId::new(1));
    }
}
