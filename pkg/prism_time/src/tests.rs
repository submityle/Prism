//! M0 tests: monotonic advancement and seconds conversions.

use crate::{Duration, Instant, Real, Time};

#[test]
fn first_update_reports_zero_delta() {
    let mut t = Time::<Real>::new();
    let base = Instant::now();
    t.update_with_instant(base);
    assert_eq!(t.delta(), Duration::ZERO);
    assert_eq!(t.elapsed(), Duration::ZERO);
    assert_eq!(t.startup(), Some(base));
}

#[test]
fn delta_and_elapsed_accumulate() {
    let mut t = Time::<Real>::new();
    let base = Instant::now();
    t.update_with_instant(base);
    t.update_with_instant(base + Duration::from_millis(16));
    assert_eq!(t.delta(), Duration::from_millis(16));
    assert_eq!(t.elapsed(), Duration::from_millis(16));

    t.update_with_instant(base + Duration::from_millis(48));
    assert_eq!(t.delta(), Duration::from_millis(32));
    assert_eq!(t.elapsed(), Duration::from_millis(48));
}

#[test]
fn monotonic_backwards_clamps_to_zero() {
    let mut t = Time::<Real>::new();
    let base = Instant::now();
    t.update_with_instant(base + Duration::from_millis(100));
    // A clock reading that went "backwards" must not produce a negative delta.
    t.update_with_instant(base);
    assert_eq!(t.delta(), Duration::ZERO);
}

#[test]
fn seconds_conversions() {
    let mut t = Time::<Real>::new();
    let base = Instant::now();
    t.update_with_instant(base);
    t.update_with_instant(base + Duration::from_millis(500));
    assert!((t.delta_secs() - 0.5).abs() < 1e-6);
    assert!((t.delta_secs_f64() - 0.5).abs() < 1e-9);
    assert!((t.elapsed_secs() - 0.5).abs() < 1e-6);
}

#[test]
fn explicit_delta_feed() {
    let mut t = Time::<Real>::new();
    t.update_with_delta(Duration::from_millis(10));
    t.update_with_delta(Duration::from_millis(20));
    assert_eq!(t.elapsed(), Duration::from_millis(30));
    assert_eq!(t.delta(), Duration::from_millis(20));
}

// ---------------------------------------------------------------------------
// M1 tests: virtual (scale/pause/clamp), fixed accumulator, default switch.
// ---------------------------------------------------------------------------

use crate::{Clocks, DefaultSource, Fixed, Virtual};

#[test]
fn virtual_default_matches_real_delta() {
    let mut t = Time::<Virtual>::new();
    t.advance_by(Duration::from_millis(16));
    assert_eq!(t.delta(), Duration::from_millis(16));
    assert_eq!(t.elapsed(), Duration::from_millis(16));
    assert_eq!(t.effective_speed(), 1.0);
}

#[test]
fn virtual_scale_runs_twice_as_fast() {
    let mut t = Time::<Virtual>::new();
    t.set_relative_speed(2.0);
    t.advance_by(Duration::from_millis(10));
    assert_eq!(t.delta(), Duration::from_millis(20));
    assert_eq!(t.elapsed(), Duration::from_millis(20));

    // Half speed advances half as fast.
    t.set_relative_speed(0.5);
    t.advance_by(Duration::from_millis(10));
    assert_eq!(t.delta(), Duration::from_millis(5));
    assert_eq!(t.elapsed(), Duration::from_millis(25));
}

#[test]
fn virtual_pause_freezes_delta_but_holds_elapsed() {
    let mut t = Time::<Virtual>::new();
    t.advance_by(Duration::from_millis(10));
    assert_eq!(t.elapsed(), Duration::from_millis(10));

    t.pause();
    assert!(t.is_paused());
    t.advance_by(Duration::from_millis(10));
    assert_eq!(t.delta(), Duration::ZERO);
    assert_eq!(t.elapsed(), Duration::from_millis(10)); // frozen
    assert_eq!(t.effective_speed(), 0.0);

    // Unpausing restores the previously set speed (1.0 here).
    t.unpause();
    t.advance_by(Duration::from_millis(10));
    assert_eq!(t.delta(), Duration::from_millis(10));
    assert_eq!(t.elapsed(), Duration::from_millis(20));
}

#[test]
fn virtual_max_delta_clamps_a_huge_frame() {
    let mut t = Time::<Virtual>::new();
    assert_eq!(t.max_delta(), Virtual::DEFAULT_MAX_DELTA);
    // A 10 s hitch is clamped to the 0.25 s max before scaling.
    t.advance_by(Duration::from_secs(10));
    assert_eq!(t.delta(), Duration::from_millis(250));

    // Clamp happens before scaling, so 2x of the clamp is 0.5 s.
    let mut t2 = Time::<Virtual>::new();
    t2.set_relative_speed(2.0);
    t2.advance_by(Duration::from_secs(10));
    assert_eq!(t2.delta(), Duration::from_millis(500));
}

#[test]
fn virtual_negative_and_nonfinite_speed_guarded() {
    let mut t = Time::<Virtual>::new();
    t.set_relative_speed(-3.0); // clamped to 0.0
    assert_eq!(t.relative_speed(), 0.0);
    t.advance_by(Duration::from_millis(10));
    assert_eq!(t.delta(), Duration::ZERO);

    t.set_relative_speed_f64(f64::NAN); // ignored, stays 0.0
    assert_eq!(t.relative_speed_f64(), 0.0);
}

#[test]
fn fixed_default_timestep_is_one_sixtyfourth() {
    let t = Time::<Fixed>::new();
    assert_eq!(t.timestep(), Fixed::DEFAULT_TIMESTEP);
    assert_eq!(t.timestep(), Duration::from_micros(15_625));
}

#[test]
fn fixed_accumulator_yields_right_number_of_steps() {
    let mut t = Time::<Fixed>::from_hz(100.0); // 10 ms timestep
    assert_eq!(t.timestep(), Duration::from_millis(10));

    // 25 ms of virtual time => two whole steps, 5 ms left over.
    t.accumulate(Duration::from_millis(25));
    assert_eq!(t.expend_all(), 2);
    assert_eq!(t.elapsed(), Duration::from_millis(20));
    assert_eq!(t.overstep(), Duration::from_millis(5));
    assert!((t.overstep_fraction() - 0.5).abs() < 1e-6);

    // Feeding another 7 ms crosses the next step boundary (5 + 7 = 12 ms).
    t.accumulate(Duration::from_millis(7));
    assert_eq!(t.expend_all(), 1);
    assert_eq!(t.overstep(), Duration::from_millis(2));
    assert!((t.overstep_fraction() - 0.2).abs() < 1e-6);
}

#[test]
fn fixed_expend_reports_step_availability() {
    let mut t = Time::<Fixed>::from_hz(100.0);
    t.accumulate(Duration::from_millis(10));
    assert!(t.expend());
    assert!(!t.expend());
    assert_eq!(t.overstep(), Duration::ZERO);
}

#[test]
fn fixed_caps_substeps_against_death_spiral() {
    let mut t = Time::<Fixed>::from_hz(100.0); // 10 ms timestep
    t.set_max_substeps(4);
    // A 10 s hitch would be 1000 steps; capped to 4.
    t.accumulate(Duration::from_secs(10));
    assert_eq!(t.overstep(), Duration::from_millis(40));
    assert_eq!(t.expend_all(), 4);
}

#[test]
fn default_switch_returns_expected_delta() {
    let mut clocks = Clocks::new();
    assert_eq!(clocks.source(), DefaultSource::Virtual);

    // Variable phase: default mirrors Virtual.
    clocks.virtual_time_mut().set_relative_speed(2.0);
    clocks.virtual_time_mut().advance_by(Duration::from_millis(10));
    clocks.sync_default();
    assert_eq!(clocks.default_time().delta(), Duration::from_millis(20));
    assert!((clocks.delta_secs() - 0.02).abs() < 1e-6);

    // Feed the virtual delta into fixed and switch the default to Fixed.
    let vdelta = clocks.virtual_time().delta();
    clocks.fixed_mut().accumulate(vdelta); // 20 ms into a 1/64 s step
    clocks.fixed_mut().expend_all();
    clocks.set_source(DefaultSource::Fixed);
    assert_eq!(clocks.source(), DefaultSource::Fixed);
    assert_eq!(clocks.default_time().delta(), Fixed::DEFAULT_TIMESTEP);
}
