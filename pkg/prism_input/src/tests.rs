//! Unit tests for the input kernel: edge tracking, axis math, gamepad curves,
//! deadzones, touch tracking, and determinism.

use alloc::vec::Vec;

use crate::axis::Axis;
use crate::button::ButtonInput;
use crate::gamepad::{AxisSettings, ButtonSettings, GamepadAxis, GamepadSettings, radial_deadzone};
use crate::keyboard::{KeyCode, ModifiersState};
use crate::mouse::MouseButton;
use crate::touch::{TouchInput, TouchPhase, Touches};

// --- ButtonInput -----------------------------------------------------------

#[test]
fn press_sets_pressed_and_just_pressed() {
    let mut input = ButtonInput::<MouseButton>::new();
    input.press(MouseButton::Left);
    assert!(input.pressed(MouseButton::Left));
    assert!(input.just_pressed(MouseButton::Left));
    assert!(!input.just_released(MouseButton::Left));
}

#[test]
fn clear_retires_edges_but_keeps_held() {
    let mut input = ButtonInput::<MouseButton>::new();
    input.press(MouseButton::Left);
    input.clear();
    assert!(input.pressed(MouseButton::Left));
    assert!(!input.just_pressed(MouseButton::Left));
}

#[test]
fn release_sets_just_released_and_clears_pressed() {
    let mut input = ButtonInput::<MouseButton>::new();
    input.press(MouseButton::Left);
    input.clear();
    input.release(MouseButton::Left);
    assert!(!input.pressed(MouseButton::Left));
    assert!(input.just_released(MouseButton::Left));
}

#[test]
fn double_press_does_not_re_edge() {
    let mut input = ButtonInput::<MouseButton>::new();
    input.press(MouseButton::Left);
    input.clear();
    // Pressing an already-held button must not raise a new just_pressed edge.
    input.press(MouseButton::Left);
    assert!(input.pressed(MouseButton::Left));
    assert!(!input.just_pressed(MouseButton::Left));
}

#[test]
fn release_all_clears_held_with_edges() {
    let mut input = ButtonInput::<MouseButton>::new();
    input.press(MouseButton::Left);
    input.press(MouseButton::Right);
    input.clear();
    input.release_all();
    assert!(!input.pressed(MouseButton::Left));
    assert!(!input.pressed(MouseButton::Right));
    assert!(input.just_released(MouseButton::Left));
    assert!(input.just_released(MouseButton::Right));
}

#[test]
fn reset_removes_all_state_for_input() {
    let mut input = ButtonInput::<MouseButton>::new();
    input.press(MouseButton::Left);
    input.reset(MouseButton::Left);
    assert!(!input.pressed(MouseButton::Left));
    assert!(!input.just_pressed(MouseButton::Left));
    assert!(!input.just_released(MouseButton::Left));
}

#[test]
fn any_and_all_pressed() {
    let mut input = ButtonInput::<MouseButton>::new();
    input.press(MouseButton::Left);
    input.press(MouseButton::Right);
    assert!(input.any_pressed([MouseButton::Left, MouseButton::Middle]));
    assert!(input.all_pressed([MouseButton::Left, MouseButton::Right]));
    assert!(!input.all_pressed([MouseButton::Left, MouseButton::Middle]));
}

#[test]
fn pressed_iteration_is_deterministic() {
    let mut input = ButtonInput::<u16>::new();
    // Insert out of order; BTreeSet must yield ascending order.
    for key in [5_u16, 1, 9, 3, 7] {
        input.press(key);
    }
    let order: Vec<u16> = input.get_pressed().copied().collect();
    assert_eq!(order, [1, 3, 5, 7, 9]);
}

// --- Axis ------------------------------------------------------------------

#[test]
fn axis_get_clamps_but_unclamped_does_not() {
    let mut axis = Axis::<GamepadAxis>::new();
    axis.set(GamepadAxis::LeftStickX, 2.5);
    assert_eq!(axis.get(GamepadAxis::LeftStickX), Some(Axis::<GamepadAxis>::MAX));
    assert_eq!(axis.get_unclamped(GamepadAxis::LeftStickX), Some(2.5));
}

#[test]
fn axis_remove_and_contains() {
    let mut axis = Axis::<GamepadAxis>::new();
    assert!(!axis.contains(GamepadAxis::LeftZ));
    axis.set(GamepadAxis::LeftZ, 0.5);
    assert!(axis.contains(GamepadAxis::LeftZ));
    assert_eq!(axis.remove(GamepadAxis::LeftZ), Some(0.5));
    assert!(!axis.contains(GamepadAxis::LeftZ));
    assert_eq!(axis.get(GamepadAxis::LeftZ), None);
}

// --- AxisSettings ----------------------------------------------------------

#[test]
fn axis_filter_collapses_deadzone() {
    let settings = AxisSettings::new(0.1, 0.95, 0.01);
    assert_eq!(settings.filter(0.05), 0.0);
    assert_eq!(settings.filter(-0.1), 0.0);
}

#[test]
fn axis_filter_saturates_livezone() {
    let settings = AxisSettings::new(0.1, 0.95, 0.01);
    assert_eq!(settings.filter(0.95), 1.0);
    assert_eq!(settings.filter(1.5), 1.0);
    assert_eq!(settings.filter(-0.99), -1.0);
}

#[test]
fn axis_filter_is_monotonic_and_sign_preserving() {
    let settings = AxisSettings::new(0.1, 0.95, 0.01);
    let a = settings.filter(0.3);
    let b = settings.filter(0.6);
    assert!(a > 0.0 && b > 0.0 && b > a);
    assert!(settings.filter(-0.6) < 0.0);
    assert!((settings.filter(0.6) + settings.filter(-0.6)).abs() < 1e-6);
}

#[test]
fn axis_settings_new_clamps_bounds() {
    // livezone < deadzone must be lifted to deadzone; bounds clamp to [0,1].
    let settings = AxisSettings::new(0.5, 0.2, -1.0);
    assert!(settings.livezone() >= settings.deadzone());
    assert!((0.0..=1.0).contains(&settings.deadzone()));
}

#[test]
fn axis_should_report_respects_threshold() {
    let settings = AxisSettings::new(0.1, 0.95, 0.05);
    assert!(!settings.should_report(0.50, 0.52));
    assert!(settings.should_report(0.50, 0.60));
}

// --- ButtonSettings --------------------------------------------------------

#[test]
fn button_settings_hysteresis() {
    let settings = ButtonSettings::new(0.75, 0.65);
    assert!(settings.is_pressed(0.8));
    assert!(!settings.is_pressed(0.7));
    assert!(settings.is_released(0.6));
    assert!(!settings.is_released(0.7));
}

#[test]
fn button_settings_new_keeps_release_below_press() {
    let settings = ButtonSettings::new(0.5, 0.9);
    // release is clamped to <= press so the band is non-inverted.
    assert!(settings.is_released(0.5));
}

// --- GamepadSettings -------------------------------------------------------

#[test]
fn gamepad_settings_override_vs_default() {
    let mut settings = GamepadSettings::new();
    let tight = AxisSettings::new(0.2, 0.9, 0.01);
    settings.set_axis(GamepadAxis::LeftStickX, tight);
    assert_eq!(settings.axis(GamepadAxis::LeftStickX), tight);
    // An un-overridden axis falls back to the default curve.
    assert_eq!(settings.axis(GamepadAxis::RightStickX), settings.default_axis);
}

// --- radial_deadzone -------------------------------------------------------

#[test]
fn radial_deadzone_inside_collapses() {
    assert_eq!(radial_deadzone(0.05, 0.05, 0.2), (0.0, 0.0));
    assert_eq!(radial_deadzone(0.0, 0.0, 0.2), (0.0, 0.0));
}

#[test]
fn radial_deadzone_preserves_direction() {
    let (x, y) = radial_deadzone(0.6, 0.0, 0.2);
    assert!(x > 0.0 && y.abs() < 1e-6);
    // A diagonal keeps equal components.
    let (dx, dy) = radial_deadzone(0.5, 0.5, 0.2);
    assert!((dx - dy).abs() < 1e-6);
    assert!(dx > 0.0);
}

#[test]
fn radial_deadzone_remaps_magnitude() {
    // At the deadzone edge output magnitude is ~0; at full tilt it is ~1.
    let (x, _) = radial_deadzone(0.2, 0.0, 0.2);
    assert!(x.abs() < 1e-6);
    let (fx, _) = radial_deadzone(1.0, 0.0, 0.2);
    assert!((fx - 1.0).abs() < 1e-5);
}

// --- Keyboard --------------------------------------------------------------

#[test]
fn key_code_is_modifier() {
    assert!(KeyCode::ShiftLeft.is_modifier());
    assert!(KeyCode::ControlRight.is_modifier());
    assert!(!KeyCode::KeyA.is_modifier());
    assert!(!KeyCode::Space.is_modifier());
}

#[test]
fn modifiers_state_is_empty() {
    let mut mods = ModifiersState::default();
    assert!(mods.is_empty());
    mods.shift = true;
    assert!(!mods.is_empty());
}

// --- Touch -----------------------------------------------------------------

#[test]
fn touch_started_tracks_point_with_edge() {
    let mut touches = Touches::new();
    touches.process(TouchInput::new(1, TouchPhase::Started, 10.0, 20.0));
    assert_eq!(touches.len(), 1);
    assert!(touches.just_pressed(1));
    let touch = touches.get(1).expect("touch 1 is active");
    assert_eq!(touch.position, (10.0, 20.0));
    assert_eq!(touch.start_position, (10.0, 20.0));
}

#[test]
fn touch_move_updates_delta() {
    let mut touches = Touches::new();
    touches.process(TouchInput::new(1, TouchPhase::Started, 0.0, 0.0));
    touches.clear();
    touches.process(TouchInput::new(1, TouchPhase::Moved, 5.0, 3.0));
    let touch = touches.get(1).expect("touch 1 is active");
    assert_eq!(touch.position, (5.0, 3.0));
    assert_eq!(touch.delta(), (5.0, 3.0));
    assert_eq!(touch.distance_from_start(), (5.0, 3.0));
}

#[test]
fn touch_end_edges_then_drops_on_clear() {
    let mut touches = Touches::new();
    touches.process(TouchInput::new(1, TouchPhase::Started, 0.0, 0.0));
    touches.clear();
    touches.process(TouchInput::new(1, TouchPhase::Ended, 0.0, 0.0));
    assert!(touches.just_released(1));
    assert_eq!(touches.len(), 1);
    touches.clear();
    assert!(touches.is_empty());
    assert!(!touches.just_released(1));
}

#[test]
fn touch_iteration_is_deterministic() {
    let mut touches = Touches::new();
    for id in [4_u64, 1, 3, 2] {
        touches.process(TouchInput::new(id, TouchPhase::Started, 0.0, 0.0));
    }
    let order: Vec<u64> = touches.iter().map(|t| t.id).collect();
    assert_eq!(order, [1, 2, 3, 4]);
}

#[test]
fn touch_clear_advances_previous_position() {
    let mut touches = Touches::new();
    touches.process(TouchInput::new(1, TouchPhase::Started, 0.0, 0.0));
    touches.process(TouchInput::new(1, TouchPhase::Moved, 10.0, 0.0));
    touches.clear();
    // After clear, delta resets because previous caught up to current.
    let touch = touches.get(1).expect("touch 1 is active");
    assert_eq!(touch.delta(), (0.0, 0.0));
}
