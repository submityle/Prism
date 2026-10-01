//! Integration tests for `prism_ui_anim`.
#![allow(
    clippy::std_instead_of_alloc,
    reason = "integration tests always run under std"
)]

use prism_ui_anim::{
    Easing, Keyframe, Spring, SpringState, StepPosition, Timeline, Transition, TransitionPhase,
    Tween,
};

fn abs(x: f32) -> f32 {
    if x < 0.0 {
        -x
    } else {
        x
    }
}

fn approx(a: f32, b: f32, eps: f32) {
    assert!(abs(a - b) <= eps, "expected {a} ~= {b} (eps {eps})");
}

// ---------------------------------------------------------------------------
// Lerp
// ---------------------------------------------------------------------------

#[test]
fn lerp_scalar_endpoints_and_midpoint() {
    use prism_ui_anim::Lerp;
    approx(2.0_f32.lerp(&4.0, 0.0), 2.0, 1e-6);
    approx(2.0_f32.lerp(&4.0, 1.0), 4.0, 1e-6);
    approx(2.0_f32.lerp(&4.0, 0.5), 3.0, 1e-6);
    // Extrapolation is allowed (no clamping).
    approx(0.0_f32.lerp(&10.0, 2.0), 20.0, 1e-6);
}

#[test]
fn lerp_f64_and_compound() {
    use prism_ui_anim::Lerp;
    let d = 1.0_f64.lerp(&3.0, 0.25);
    assert!((d - 1.5).abs() <= 1e-9);

    let a = [0.0_f32, 10.0, -4.0];
    let b = [10.0_f32, 0.0, 4.0];
    let m = a.lerp(&b, 0.5);
    approx(m[0], 5.0, 1e-6);
    approx(m[1], 5.0, 1e-6);
    approx(m[2], 0.0, 1e-6);

    let t = (0.0_f32, 100.0_f32).lerp(&(100.0, 0.0), 0.25);
    approx(t.0, 25.0, 1e-6);
    approx(t.1, 75.0, 1e-6);

    let q = (0.0f32, 0.0, 0.0, 0.0).lerp(&(4.0, 8.0, 12.0, 16.0), 0.5);
    approx(q.0, 2.0, 1e-6);
    approx(q.3, 8.0, 1e-6);
}

// ---------------------------------------------------------------------------
// Easing
// ---------------------------------------------------------------------------

const STANDARD_EASINGS: &[Easing] = &[
    Easing::Linear,
    Easing::EaseIn,
    Easing::EaseOut,
    Easing::EaseInOut,
    Easing::QuadIn,
    Easing::QuadOut,
    Easing::QuadInOut,
    Easing::CubicIn,
    Easing::CubicOut,
    Easing::CubicInOut,
    Easing::QuartIn,
    Easing::QuartOut,
    Easing::QuartInOut,
    Easing::QuintIn,
    Easing::QuintOut,
    Easing::QuintInOut,
    Easing::SineIn,
    Easing::SineOut,
    Easing::SineInOut,
    Easing::ExpoIn,
    Easing::ExpoOut,
    Easing::ExpoInOut,
    Easing::CircIn,
    Easing::CircOut,
    Easing::CircInOut,
    Easing::CubicBezier(0.33, 0.0, 0.67, 1.0),
];

#[test]
fn easing_endpoints_are_pinned() {
    for e in STANDARD_EASINGS {
        approx(e.sample(0.0), 0.0, 1e-4);
        approx(e.sample(1.0), 1.0, 1e-4);
    }
    // Back family pins endpoints too.
    for e in [Easing::BackIn, Easing::BackOut, Easing::BackInOut] {
        approx(e.sample(0.0), 0.0, 1e-4);
        approx(e.sample(1.0), 1.0, 1e-4);
    }
}

#[test]
fn easing_input_is_clamped() {
    for e in STANDARD_EASINGS {
        approx(e.sample(-5.0), 0.0, 1e-4);
        approx(e.sample(5.0), 1.0, 1e-4);
    }
}

#[test]
fn standard_easings_are_monotonic_nondecreasing() {
    for e in STANDARD_EASINGS {
        let mut prev = e.sample(0.0);
        let mut i = 1;
        while i <= 100 {
            let t = i as f32 / 100.0;
            let cur = e.sample(t);
            assert!(
                cur >= prev - 1e-4,
                "{e:?} not monotonic at t={t}: {prev} -> {cur}"
            );
            prev = cur;
            i += 1;
        }
    }
}

#[test]
fn linear_is_identity() {
    let mut i = 0;
    while i <= 10 {
        let t = i as f32 / 10.0;
        approx(Easing::Linear.sample(t), t, 1e-6);
        i += 1;
    }
}

#[test]
fn back_in_undershoots_and_back_out_overshoots() {
    // BackIn dips below zero shortly after the start.
    assert!(Easing::BackIn.sample(0.2) < 0.0);
    // BackOut rises above one shortly before the end.
    assert!(Easing::BackOut.sample(0.8) > 1.0);
}

#[test]
fn cubic_bezier_matches_linear_control_points() {
    // Control points on the diagonal reproduce the identity curve.
    let e = Easing::CubicBezier(0.333_333, 0.333_333, 0.666_666, 0.666_666);
    let mut i = 0;
    while i <= 10 {
        let t = i as f32 / 10.0;
        approx(e.sample(t), t, 2e-3);
        i += 1;
    }
}

#[test]
fn steps_jump_end() {
    let e = Easing::Steps(4, StepPosition::JumpEnd);
    approx(e.sample(0.0), 0.0, 1e-6);
    approx(e.sample(0.1), 0.0, 1e-6);
    approx(e.sample(0.3), 0.25, 1e-6);
    approx(e.sample(0.6), 0.5, 1e-6);
    approx(e.sample(1.0), 1.0, 1e-6);
}

#[test]
fn steps_jump_start() {
    let e = Easing::Steps(4, StepPosition::JumpStart);
    approx(e.sample(0.0), 0.25, 1e-6);
    approx(e.sample(0.3), 0.5, 1e-6);
    approx(e.sample(1.0), 1.0, 1e-6);
}

#[test]
fn steps_jump_none_and_both_endpoints() {
    let none = Easing::Steps(5, StepPosition::JumpNone);
    approx(none.sample(0.0), 0.0, 1e-6);
    approx(none.sample(1.0), 1.0, 1e-6);

    let both = Easing::Steps(5, StepPosition::JumpBoth);
    approx(both.sample(0.0), 1.0 / 6.0, 1e-6);
    approx(both.sample(1.0), 1.0, 1e-6);
}

// ---------------------------------------------------------------------------
// Spring
// ---------------------------------------------------------------------------

fn steps_to_settle(spring: &Spring, target: f32, start: f32, dt: f32, max_steps: u32) -> u32 {
    let mut state = SpringState::new(start, 0.0);
    let mut n = 0;
    while n < max_steps {
        state.step(spring, target, dt);
        n += 1;
        if state.is_settled(target, 1e-3) {
            return n;
        }
    }
    panic!("spring did not settle within {max_steps} steps");
}

#[test]
fn underdamped_spring_settles() {
    let spring = Spring::wobbly();
    assert!(spring.damping_ratio() < 1.0);
    let n = steps_to_settle(&spring, 1.0, 0.0, 1.0 / 60.0, 1000);
    assert!(n < 1000);
}

#[test]
fn critically_damped_spring_settles_without_overshoot() {
    // zeta == 1 exactly: c = 2*sqrt(k*m).
    let k = 100.0_f32;
    let m = 1.0_f32;
    let c = 2.0 * (k * m).sqrt();
    let spring = Spring::new(k, c, m);
    approx(spring.damping_ratio(), 1.0, 1e-3);

    let mut state = SpringState::new(0.0, 0.0);
    let dt = 1.0 / 120.0;
    let mut n = 0;
    while n < 2000 {
        state.step(&spring, 1.0, dt);
        // Critically damped from rest never overshoots the target.
        assert!(state.value <= 1.0 + 1e-3, "overshoot: {}", state.value);
        if state.is_settled(1.0, 1e-3) {
            break;
        }
        n += 1;
    }
    assert!(n < 2000);
    approx(state.value, 1.0, 1e-3);
}

#[test]
fn overdamped_spring_settles_without_overshoot() {
    let k = 50.0_f32;
    let m = 1.0_f32;
    let c = 40.0_f32; // zeta > 1
    let spring = Spring::new(k, c, m);
    assert!(spring.damping_ratio() > 1.0);

    let mut state = SpringState::new(0.0, 0.0);
    let dt = 1.0 / 120.0;
    let mut n = 0;
    while n < 5000 {
        state.step(&spring, 1.0, dt);
        assert!(state.value <= 1.0 + 1e-3, "overshoot: {}", state.value);
        if state.is_settled(1.0, 1e-3) {
            break;
        }
        n += 1;
    }
    assert!(n < 5000);
    approx(state.value, 1.0, 1e-3);
}

#[test]
fn spring_zero_dt_is_noop() {
    let spring = Spring::default();
    let mut state = SpringState::new(0.0, 0.0);
    state.step(&spring, 1.0, 0.0);
    approx(state.value, 0.0, 1e-9);
    approx(state.velocity, 0.0, 1e-9);
}

#[test]
fn spring_presets_have_expected_regimes() {
    assert!(Spring::gentle().damping_ratio() < 1.0);
    assert!(Spring::wobbly().damping_ratio() < 1.0);
    assert!(Spring::stiff().damping_ratio() < 1.0);
    assert!(Spring::default().damping_ratio() < 1.0);
}

// ---------------------------------------------------------------------------
// Timeline
// ---------------------------------------------------------------------------

#[test]
fn timeline_samples_keyframes_and_midpoints() {
    let mut tl = Timeline::<f32>::new();
    tl.push(Keyframe::linear(0.0, 0.0));
    tl.push(Keyframe::linear(1.0, 10.0));
    tl.push(Keyframe::linear(2.0, 10.0));

    // Exact keyframe hits.
    approx(tl.sample(0.0), 0.0, 1e-6);
    approx(tl.sample(1.0), 10.0, 1e-6);
    approx(tl.sample(2.0), 10.0, 1e-6);

    // Linear midpoint of the first segment.
    approx(tl.sample(0.5), 5.0, 1e-6);
    // Flat second segment.
    approx(tl.sample(1.5), 10.0, 1e-6);

    // Clamping outside the range.
    approx(tl.sample(-1.0), 0.0, 1e-6);
    approx(tl.sample(5.0), 10.0, 1e-6);

    approx(tl.duration(), 2.0, 1e-6);
    assert_eq!(tl.len(), 3);
}

#[test]
fn timeline_sorts_unordered_input_and_eases_per_segment() {
    let tl = Timeline::from_keyframes(alloc_vec_of([
        Keyframe::new(2.0, 100.0, Easing::Linear),
        Keyframe::new(0.0, 0.0, Easing::QuadIn),
        Keyframe::new(1.0, 100.0, Easing::Linear),
    ]));
    // Sorted so the first segment (0->1) uses QuadIn: midpoint eases low.
    approx(tl.sample(0.5), 100.0 * 0.25, 1e-4);
    // Second segment is flat.
    approx(tl.sample(1.5), 100.0, 1e-4);
    // Keyframes still land exactly.
    approx(tl.sample(0.0), 0.0, 1e-6);
    approx(tl.sample(1.0), 100.0, 1e-6);
}

fn alloc_vec_of<const N: usize>(items: [Keyframe<f32>; N]) -> Vec<Keyframe<f32>> {
    items.into_iter().collect()
}

#[test]
fn empty_timeline_try_sample_is_none() {
    let tl = Timeline::<f32>::new();
    assert!(tl.is_empty());
    assert!(tl.try_sample(0.5).is_none());
}

// ---------------------------------------------------------------------------
// Tween + Transition
// ---------------------------------------------------------------------------

#[test]
fn tween_reaches_target_at_end() {
    let mut tween = Tween::new(0.0_f32, 100.0, 1.0, Easing::Linear);
    approx(tween.value(), 0.0, 1e-6);
    approx(tween.step(0.5), 50.0, 1e-6);
    assert!(!tween.finished());
    approx(tween.step(0.5), 100.0, 1e-6);
    assert!(tween.finished());
    // Overstepping stays pinned at the target.
    approx(tween.step(10.0), 100.0, 1e-6);
}

#[test]
fn tween_zero_duration_is_instant() {
    let mut tween = Tween::new(0.0_f32, 5.0, 0.0, Easing::Linear);
    assert!(tween.finished());
    approx(tween.value(), 5.0, 1e-6);
    approx(tween.step(0.0), 5.0, 1e-6);
}

#[test]
fn tween_reset_restarts() {
    let mut tween = Tween::new(0.0_f32, 10.0, 1.0, Easing::Linear);
    tween.step(1.0);
    assert!(tween.finished());
    tween.reset();
    assert!(!tween.finished());
    approx(tween.value(), 0.0, 1e-6);
}

#[test]
fn transition_enter_and_exit() {
    let mut tr = Transition::new(1.0, Easing::Linear);
    assert_eq!(tr.phase(), TransitionPhase::Exited);
    approx(tr.value(), 0.0, 1e-6);

    tr.enter();
    assert_eq!(tr.phase(), TransitionPhase::Entering);
    approx(tr.step(0.5), 0.5, 1e-6);
    approx(tr.step(0.5), 1.0, 1e-6);
    assert_eq!(tr.phase(), TransitionPhase::Entered);

    tr.exit();
    assert_eq!(tr.phase(), TransitionPhase::Exiting);
    approx(tr.step(1.0), 0.0, 1e-6);
    assert_eq!(tr.phase(), TransitionPhase::Exited);
}

#[test]
fn transition_interrupt_is_continuous() {
    let mut tr = Transition::new(1.0, Easing::Linear);
    tr.enter();
    let mid = tr.step(0.5);
    approx(mid, 0.5, 1e-6);
    // Interrupt the enter half-way; exit should start from the current value.
    tr.exit();
    approx(tr.value(), mid, 1e-6);
    approx(tr.step(0.5), 0.25, 1e-6);
}
