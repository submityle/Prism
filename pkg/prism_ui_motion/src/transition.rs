//! Implicit style transitions.
//!
//! A [`TransitionTracker`] remembers the last value seen for each
//! [`StyleProp`]. When a new value arrives that differs from the previous one
//! and the property has a declared [`TransitionSpec`] (duration + easing), the
//! tracker starts a [`PropertyTransition`] that interpolates between the two
//! values over time, in the spirit of Framer Motion's automatic transitions.
//!
//! Interrupting an in-flight transition retargets from the *current* animated
//! value rather than snapping, so rapid value changes stay continuous.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use prism_ui_anim::{Easing, Tween};
use prism_ui_style::{StyleProp, StyleValue};

use crate::value_anim::AnimatableValue;

/// How a property should animate when its value changes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TransitionSpec {
    /// Duration of the transition in seconds. Non-positive is instant.
    pub duration: f32,
    /// Easing applied to the normalized progress.
    pub easing: Easing,
}

impl TransitionSpec {
    /// Builds a spec from a duration (seconds) and an easing curve.
    #[must_use]
    pub const fn new(duration: f32, easing: Easing) -> Self {
        Self { duration, easing }
    }

    /// A linear transition of the given duration.
    #[must_use]
    pub const fn linear(duration: f32) -> Self {
        Self::new(duration, Easing::Linear)
    }

    /// An ease-out transition of the given duration, a sensible UI default.
    #[must_use]
    pub const fn ease_out(duration: f32) -> Self {
        Self::new(duration, Easing::EaseOut)
    }
}

/// A single property's in-flight transition between two [`StyleValue`]s.
///
/// Backed by a [`Tween`] over [`AnimatableValue`], so continuous values blend
/// numerically and discrete ones switch at the midpoint.
#[derive(Clone, Debug, PartialEq)]
pub struct PropertyTransition {
    tween: Tween<AnimatableValue>,
}

impl PropertyTransition {
    /// Starts a transition from `from` to `to` using `spec`.
    #[must_use]
    pub fn new(from: StyleValue, to: StyleValue, spec: &TransitionSpec) -> Self {
        Self {
            tween: Tween::new(
                AnimatableValue::new(from),
                AnimatableValue::new(to),
                spec.duration,
                spec.easing,
            ),
        }
    }

    /// Advances by `dt` seconds and returns the new interpolated value.
    pub fn step(&mut self, dt: f32) -> StyleValue {
        self.tween.step(dt).into_inner()
    }

    /// The current interpolated value without advancing time.
    #[must_use]
    pub fn value(&self) -> StyleValue {
        self.tween.value().into_inner()
    }

    /// Normalized, un-eased progress in `[0, 1]`.
    #[must_use]
    pub fn progress(&self) -> f32 {
        self.tween.progress()
    }

    /// Whether the transition has reached its target.
    #[must_use]
    pub fn finished(&self) -> bool {
        self.tween.finished()
    }

    /// The starting value of the transition.
    #[must_use]
    pub fn from(&self) -> &StyleValue {
        self.tween.from.get()
    }

    /// The target value of the transition.
    #[must_use]
    pub fn to(&self) -> &StyleValue {
        self.tween.to.get()
    }
}

/// Tracks per-property values and drives implicit transitions between them.
///
/// Declare which properties animate with [`TransitionTracker::with_transition`]
/// or [`TransitionTracker::set_transition`]. Feed new target values each frame
/// with [`TransitionTracker::observe`], then advance all active transitions
/// with [`TransitionTracker::step`]. Query the current (possibly animated)
/// value with [`TransitionTracker::value`].
#[derive(Clone, Debug, Default)]
pub struct TransitionTracker {
    specs: BTreeMap<StyleProp, TransitionSpec>,
    last: BTreeMap<StyleProp, StyleValue>,
    active: BTreeMap<StyleProp, PropertyTransition>,
}

impl TransitionTracker {
    /// Creates an empty tracker with no declared transitions.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Declares a transition for `prop`, returning `self` for chaining.
    #[must_use]
    pub fn with_transition(mut self, prop: StyleProp, spec: TransitionSpec) -> Self {
        self.specs.insert(prop, spec);
        self
    }

    /// Declares (or replaces) a transition for `prop`.
    pub fn set_transition(&mut self, prop: StyleProp, spec: TransitionSpec) {
        self.specs.insert(prop, spec);
    }

    /// Removes any declared transition for `prop`.
    ///
    /// Returns the previously declared spec, if any.
    pub fn clear_transition(&mut self, prop: StyleProp) -> Option<TransitionSpec> {
        self.specs.remove(&prop)
    }

    /// The declared transition spec for `prop`, if any.
    #[must_use]
    pub fn spec(&self, prop: StyleProp) -> Option<&TransitionSpec> {
        self.specs.get(&prop)
    }

    /// Observes a new target `value` for `prop`.
    ///
    /// Returns `true` if this started (or retargeted) a transition. A
    /// transition is started only when `prop` has a declared spec, a previous
    /// value exists, and the new value differs from it. The first value for a
    /// property is recorded without animating. A retarget mid-flight starts
    /// from the current animated value so motion stays continuous.
    pub fn observe(&mut self, prop: StyleProp, value: StyleValue) -> bool {
        let started = match (self.specs.get(&prop).copied(), self.last.get(&prop)) {
            (Some(spec), Some(prev)) if *prev != value => {
                let start_from = match self.active.get(&prop) {
                    Some(active) => active.value(),
                    None => prev.clone(),
                };
                self.active.insert(
                    prop,
                    PropertyTransition::new(start_from, value.clone(), &spec),
                );
                true
            }
            _ => false,
        };
        self.last.insert(prop, value);
        started
    }

    /// Advances every active transition by `dt` seconds.
    ///
    /// Returns the current `(prop, value)` for each still-active transition.
    /// Transitions that finish are retired, and their property's recorded
    /// value is pinned to the target.
    pub fn step(&mut self, dt: f32) -> Vec<(StyleProp, StyleValue)> {
        let mut out = Vec::new();
        let mut finished = Vec::new();
        for (prop, transition) in &mut self.active {
            let value = transition.step(dt);
            out.push((*prop, value));
            if transition.finished() {
                finished.push(*prop);
            }
        }
        for prop in finished {
            if let Some(transition) = self.active.remove(&prop) {
                self.last.insert(prop, transition.to().clone());
            }
        }
        out
    }

    /// The current value for `prop`: the animated value while a transition is
    /// active, otherwise the last observed value.
    #[must_use]
    pub fn value(&self, prop: StyleProp) -> Option<StyleValue> {
        match self.active.get(&prop) {
            Some(transition) => Some(transition.value()),
            None => self.last.get(&prop).cloned(),
        }
    }

    /// Whether `prop` currently has an active transition.
    #[must_use]
    pub fn is_animating(&self, prop: StyleProp) -> bool {
        self.active.contains_key(&prop)
    }

    /// Whether any property currently has an active transition.
    #[must_use]
    pub fn is_idle(&self) -> bool {
        self.active.is_empty()
    }

    /// The number of currently active transitions.
    #[must_use]
    pub fn active_len(&self) -> usize {
        self.active.len()
    }

    /// The properties with active transitions, in [`StyleProp`] order.
    #[must_use]
    pub fn active_props(&self) -> Vec<StyleProp> {
        self.active.keys().copied().collect()
    }

    /// The current value of every known property, in [`StyleProp`] order.
    #[must_use]
    pub fn snapshot(&self) -> Vec<(StyleProp, StyleValue)> {
        self.last
            .keys()
            .copied()
            .filter_map(|prop| self.value(prop).map(|value| (prop, value)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1.0e-6;

    fn px(value: f32) -> StyleValue {
        StyleValue::px(value)
    }

    fn as_px(value: &StyleValue) -> f32 {
        match value {
            StyleValue::Length(prism_ui_style::Length::Px(v)) => *v,
            other => panic!("expected px, got {other:?}"),
        }
    }

    #[test]
    fn first_observation_does_not_animate() {
        let mut tracker =
            TransitionTracker::new().with_transition(StyleProp::Width, TransitionSpec::linear(1.0));
        assert!(!tracker.observe(StyleProp::Width, px(10.0)));
        assert!(!tracker.is_animating(StyleProp::Width));
        assert_eq!(tracker.value(StyleProp::Width), Some(px(10.0)));
    }

    #[test]
    fn change_starts_transition_and_blends_over_time() {
        let mut tracker =
            TransitionTracker::new().with_transition(StyleProp::Width, TransitionSpec::linear(1.0));
        tracker.observe(StyleProp::Width, px(0.0));
        assert!(tracker.observe(StyleProp::Width, px(100.0)));
        assert!(tracker.is_animating(StyleProp::Width));

        tracker.step(0.5);
        assert!((as_px(&tracker.value(StyleProp::Width).unwrap()) - 50.0).abs() < EPS);

        tracker.step(0.5);
        assert!(!tracker.is_animating(StyleProp::Width));
        assert_eq!(tracker.value(StyleProp::Width), Some(px(100.0)));
    }

    #[test]
    fn without_spec_values_snap() {
        let mut tracker = TransitionTracker::new();
        tracker.observe(StyleProp::Height, px(0.0));
        assert!(!tracker.observe(StyleProp::Height, px(50.0)));
        assert!(!tracker.is_animating(StyleProp::Height));
        assert_eq!(tracker.value(StyleProp::Height), Some(px(50.0)));
    }

    #[test]
    fn same_value_does_not_restart() {
        let mut tracker =
            TransitionTracker::new().with_transition(StyleProp::Width, TransitionSpec::linear(1.0));
        tracker.observe(StyleProp::Width, px(10.0));
        assert!(!tracker.observe(StyleProp::Width, px(10.0)));
    }

    #[test]
    fn interrupting_retargets_from_current_value() {
        let mut tracker =
            TransitionTracker::new().with_transition(StyleProp::Width, TransitionSpec::linear(1.0));
        tracker.observe(StyleProp::Width, px(0.0));
        tracker.observe(StyleProp::Width, px(100.0));
        tracker.step(0.5); // now at 50
                           // Retarget to 0; it should start from the current 50, not from 100.
        assert!(tracker.observe(StyleProp::Width, px(0.0)));
        assert!((as_px(tracker.active.get(&StyleProp::Width).unwrap().from()) - 50.0).abs() < EPS);
        tracker.step(0.5);
        assert!((as_px(&tracker.value(StyleProp::Width).unwrap()) - 25.0).abs() < EPS);
    }

    #[test]
    fn step_reports_active_values_and_retires_finished() {
        let mut tracker = TransitionTracker::new()
            .with_transition(StyleProp::Opacity, TransitionSpec::linear(1.0));
        tracker.observe(StyleProp::Opacity, StyleValue::number(0.0));
        tracker.observe(StyleProp::Opacity, StyleValue::number(1.0));
        let frame = tracker.step(0.25);
        assert_eq!(frame.len(), 1);
        assert_eq!(frame[0].0, StyleProp::Opacity);
        tracker.step(1.0);
        assert!(tracker.is_idle());
        assert_eq!(tracker.active_props(), Vec::new());
    }

    #[test]
    fn snapshot_lists_all_known_props() {
        let mut tracker =
            TransitionTracker::new().with_transition(StyleProp::Width, TransitionSpec::linear(1.0));
        tracker.observe(StyleProp::Width, px(10.0));
        tracker.observe(StyleProp::Height, px(20.0));
        let snap = tracker.snapshot();
        assert_eq!(snap.len(), 2);
    }
}
