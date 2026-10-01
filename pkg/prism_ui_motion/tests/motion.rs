//! Integration tests exercising the public motion API end to end.

use prism_ui::Key;
use prism_ui_anim::Easing;
use prism_ui_motion::{
    FlipState, Rect, SharedElementTransition, Transform, TransitionSpec, TransitionTracker,
};
use prism_ui_style::{Length, StyleProp, StyleValue};

const EPS: f32 = 1.0e-4;

fn as_px(value: &StyleValue) -> f32 {
    match value {
        StyleValue::Length(Length::Px(v)) => *v,
        other => panic!("expected px, got {other:?}"),
    }
}

#[test]
fn implicit_transition_runs_to_completion_over_frames() {
    let mut tracker = TransitionTracker::new()
        .with_transition(StyleProp::Width, TransitionSpec::ease_out(1.0))
        .with_transition(StyleProp::Opacity, TransitionSpec::linear(1.0));

    tracker.observe(StyleProp::Width, StyleValue::px(0.0));
    tracker.observe(StyleProp::Opacity, StyleValue::number(1.0));

    // Change both properties; both should begin animating.
    assert!(tracker.observe(StyleProp::Width, StyleValue::px(200.0)));
    assert!(tracker.observe(StyleProp::Opacity, StyleValue::number(0.0)));
    assert_eq!(tracker.active_len(), 2);

    // Drive 10 frames of 0.1s each.
    for _ in 0..10 {
        tracker.step(0.1);
    }
    assert!(tracker.is_idle());
    assert_eq!(tracker.value(StyleProp::Width), Some(StyleValue::px(200.0)));
    assert_eq!(
        tracker.value(StyleProp::Opacity),
        Some(StyleValue::number(0.0))
    );
}

#[test]
fn flip_sequence_glides_from_old_to_new_layout() {
    let mut state = FlipState::seeded(Rect::new(0.0, 0.0, 100.0, 100.0));
    let current = Rect::new(400.0, 0.0, 200.0, 100.0);
    let mut anim = state
        .flip(current, 1.0, Easing::Linear)
        .expect("layout changed so a FLIP animation is produced");

    // At the start the inverted transform maps the new box back onto the old.
    let start = anim.sample(0.0);
    let mapped = start.apply_rect(&current);
    assert!((mapped.x - 0.0).abs() < EPS);
    assert!((mapped.width - 100.0).abs() < EPS);

    // Playing forward lands exactly on identity.
    anim.step(0.5);
    assert!(!anim.finished());
    anim.step(0.5);
    assert!(anim.finished());
    assert!(anim.current().is_identity());

    // The state now remembers the new rectangle.
    assert_eq!(state.last(), Some(current));
}

#[test]
fn shared_element_hero_matches_and_falls_back() {
    let from = [
        (Key::Int(1), Rect::new(0.0, 0.0, 100.0, 100.0)),
        (Key::Str("avatar".into()), Rect::new(10.0, 10.0, 40.0, 40.0)),
    ];
    let to = [
        (Key::Int(1), Rect::new(300.0, 50.0, 100.0, 100.0)),
        (Key::Int(2), Rect::new(0.0, 0.0, 10.0, 10.0)),
    ];

    let transition = SharedElementTransition::from_frames(&from, &to, 0.5, Easing::Linear);

    // Key 1 is shared and should produce a transform that resolves to identity.
    assert!(transition.pair(&Key::Int(1)).is_some());
    let end = transition.sample(&Key::Int(1), 1.0).unwrap();
    assert!(end.is_identity());

    // The midpoint transform is strictly between source and destination.
    let mid: Transform = transition.sample(&Key::Int(1), 0.5).unwrap();
    assert!(mid.tx < 0.0 && mid.tx > -300.0);

    // The avatar only exists in `from` (leaving); key 2 only in `to` (entering).
    assert_eq!(transition.leaving_keys(), vec![Key::Str("avatar".into())]);
    assert_eq!(transition.entering_keys(), vec![Key::Int(2)]);
    assert!((transition.fallback_appearance(&Key::Int(2), 0.5).unwrap() - 0.5).abs() < EPS);
    assert!(transition.sample(&Key::Int(2), 0.5).is_none());
}

#[test]
fn discrete_properties_switch_at_midpoint() {
    use prism_ui_style::Keyword;

    let mut tracker = TransitionTracker::new()
        .with_transition(StyleProp::FlexDirection, TransitionSpec::linear(1.0));
    tracker.observe(StyleProp::FlexDirection, StyleValue::keyword(Keyword::Row));
    assert!(tracker.observe(
        StyleProp::FlexDirection,
        StyleValue::keyword(Keyword::Column)
    ));

    // Before the midpoint it still reports the old keyword.
    tracker.step(0.25);
    assert_eq!(
        tracker.value(StyleProp::FlexDirection),
        Some(StyleValue::keyword(Keyword::Row))
    );
    // After the midpoint it reports the new keyword.
    tracker.step(0.5);
    assert_eq!(
        tracker.value(StyleProp::FlexDirection),
        Some(StyleValue::keyword(Keyword::Column))
    );
}

#[test]
fn interrupting_a_transition_stays_continuous() {
    let mut tracker =
        TransitionTracker::new().with_transition(StyleProp::Height, TransitionSpec::linear(1.0));
    tracker.observe(StyleProp::Height, StyleValue::px(0.0));
    tracker.observe(StyleProp::Height, StyleValue::px(100.0));
    tracker.step(0.5); // at 50

    // Reverse direction mid-flight: should start from 50, not 100.
    tracker.observe(StyleProp::Height, StyleValue::px(0.0));
    tracker.step(0.5); // halfway from 50 -> 0 == 25
    assert!((as_px(&tracker.value(StyleProp::Height).unwrap()) - 25.0).abs() < EPS);
}
