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
