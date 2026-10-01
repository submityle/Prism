//! `FLIP` layout animation (First, Last, Invert, Play).
//!
//! The `FLIP` technique animates a layout change cheaply:
//!
//! * **First** — record the element's rectangle before the change.
//! * **Last** — read its rectangle after the layout change.
//! * **Invert** — compute the [`Transform`] that makes the new box *appear* at
//!   the old box ([`Transform::from_rects`]).
//! * **Play** — animate that transform back toward
//!   [`Transform::IDENTITY`], so the element visually glides from its old
//!   position and size to its new ones.
//!
//! [`FlipState`] holds the previous rectangle and produces a [`FlipAnimation`]
//! whenever the layout moves. The animation is pure arithmetic driven by a
//! [`prism_ui_anim::Tween`].

use prism_ui_anim::{Easing, Lerp, Tween};

use crate::geometry::{Rect, Transform};

/// A playing `FLIP` animation: a transform tweening from the inverted start
/// toward identity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlipAnimation {
    tween: Tween<Transform>,
}

impl FlipAnimation {
    /// Builds an animation that moves an element from `prev` to `current`.
    ///
    /// The element is assumed to be laid out at `current`; the animation starts
    /// with the inverted transform (so it looks like it is still at `prev`) and
    /// plays toward identity over `duration` seconds.
    #[must_use]
    pub fn new(prev: Rect, current: Rect, duration: f32, easing: Easing) -> Self {
        let invert = Transform::from_rects(&prev, &current);
        Self {
            tween: Tween::new(invert, Transform::IDENTITY, duration, easing),
        }
    }

    /// The initial (inverted) transform at the start of the animation.
    #[must_use]
    pub fn invert(&self) -> Transform {
        self.tween.from
    }

    /// Samples the transform at a normalized progress `t`, independent of the
    /// animation's own elapsed clock.
    ///
    /// `t` is clamped into `[0, 1]`, then shaped by the easing. At `t == 0` the
    /// result is the inverted transform; at `t == 1` it is the identity.
    #[must_use]
    pub fn sample(&self, t: f32) -> Transform {
        let eased = self.tween.easing.sample(t.clamp(0.0, 1.0));
        self.tween.from.lerp(&self.tween.to, eased)
    }

    /// Advances the animation by `dt` seconds and returns the current transform.
    pub fn step(&mut self, dt: f32) -> Transform {
        self.tween.step(dt)
    }

    /// The current transform for the elapsed clock, without advancing it.
    #[must_use]
    pub fn current(&self) -> Transform {
        self.tween.value()
    }

    /// Normalized, un-eased progress in `[0, 1]`.
    #[must_use]
    pub fn progress(&self) -> f32 {
        self.tween.progress()
    }

    /// Whether the animation has returned to identity.
    #[must_use]
    pub fn finished(&self) -> bool {
        self.tween.finished()
    }
}

/// Remembers an element's rectangle across frames to drive `FLIP` animations.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FlipState {
    last: Option<Rect>,
}

impl FlipState {
    /// Creates an empty state with no recorded rectangle.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a state already seeded with a known rectangle.
    #[must_use]
    pub fn seeded(rect: Rect) -> Self {
        Self { last: Some(rect) }
    }

    /// The last recorded rectangle, if any.
    #[must_use]
    pub fn last(&self) -> Option<Rect> {
        self.last
    }

    /// Records `rect` as the latest known rectangle without animating.
    pub fn record(&mut self, rect: Rect) {
        self.last = Some(rect);
    }

    /// The inverted transform from the recorded rectangle to `current`.
    ///
    /// Returns [`Transform::IDENTITY`] when there is no recorded rectangle.
    #[must_use]
    pub fn invert(&self, current: Rect) -> Transform {
        match self.last {
            Some(prev) => Transform::from_rects(&prev, &current),
            None => Transform::IDENTITY,
        }
    }

    /// Observes a new `current` rectangle and starts a `FLIP` animation if the
    /// element actually moved or resized.
    ///
    /// Always updates the recorded rectangle to `current`. Returns `None` when
    /// there was no previous rectangle or when the box did not change.
    pub fn flip(&mut self, current: Rect, duration: f32, easing: Easing) -> Option<FlipAnimation> {
        let animation = match self.last {
            Some(prev) if prev != current => {
                Some(FlipAnimation::new(prev, current, duration, easing))
            }
            _ => None,
        };
        self.last = Some(current);
        animation
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1.0e-4;

    fn close(a: f32, b: f32) {
        assert!((a - b).abs() < EPS, "expected {a} ~= {b}");
    }

    #[test]
    fn animation_starts_inverted_and_ends_at_identity() {
        let prev = Rect::new(0.0, 0.0, 100.0, 100.0);
        let current = Rect::new(200.0, 50.0, 100.0, 100.0);
        let flip = FlipAnimation::new(prev, current, 1.0, Easing::Linear);
        let start = flip.sample(0.0);
        close(start.tx, -200.0);
        close(start.ty, -50.0);
        close(start.sx, 1.0);
        assert!(flip.sample(1.0).is_identity());
    }

    #[test]
    fn sample_is_linear_midpoint() {
        let prev = Rect::new(0.0, 0.0, 100.0, 100.0);
        let current = Rect::new(100.0, 0.0, 200.0, 100.0);
        let flip = FlipAnimation::new(prev, current, 1.0, Easing::Linear);
        let mid = flip.sample(0.5);
        // sx goes from 0.5 toward 1.0 -> 0.75 at the midpoint.
        close(mid.sx, 0.75);
    }

    #[test]
    fn sample_clamps_out_of_range() {
        let prev = Rect::new(0.0, 0.0, 100.0, 100.0);
        let current = Rect::new(100.0, 0.0, 100.0, 100.0);
        let flip = FlipAnimation::new(prev, current, 1.0, Easing::Linear);
        close(flip.sample(-5.0).tx, -100.0);
        assert!(flip.sample(5.0).is_identity());
    }

    #[test]
    fn step_advances_toward_identity() {
        let prev = Rect::new(0.0, 0.0, 100.0, 100.0);
        let current = Rect::new(100.0, 0.0, 100.0, 100.0);
        let mut flip = FlipAnimation::new(prev, current, 1.0, Easing::Linear);
        flip.step(0.5);
        close(flip.current().tx, -50.0);
        flip.step(0.5);
        assert!(flip.finished());
        assert!(flip.current().is_identity());
    }

    #[test]
    fn state_flip_requires_prior_and_change() {
        let mut state = FlipState::new();
        assert!(state
            .flip(Rect::new(0.0, 0.0, 10.0, 10.0), 1.0, Easing::Linear)
            .is_none());
        // Same rect -> no animation.
        assert!(state
            .flip(Rect::new(0.0, 0.0, 10.0, 10.0), 1.0, Easing::Linear)
            .is_none());
        // Moved -> animation.
        let anim = state.flip(Rect::new(5.0, 0.0, 10.0, 10.0), 1.0, Easing::Linear);
        assert!(anim.is_some());
        close(anim.unwrap().invert().tx, -5.0);
    }

    #[test]
    fn seeded_state_inverts_immediately() {
        let state = FlipState::seeded(Rect::new(0.0, 0.0, 10.0, 10.0));
        let t = state.invert(Rect::new(10.0, 0.0, 10.0, 10.0));
        close(t.tx, -10.0);
        assert_eq!(
            FlipState::new().invert(Rect::new(1.0, 2.0, 3.0, 4.0)),
            Transform::IDENTITY
        );
    }
}
