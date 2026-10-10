//! §24.5 tests: long-session precision and drift correction — wrap-safe
//! monotonic accumulation and bounded-slew reference reconciliation. All
//! oracles are hand-computed from fixed inputs; the correction path is pure
//! integer arithmetic, so every expectation is exact.

use crate::{DriftCorrector, Duration, MonotonicBaseline};

fn ns(n: u64) -> Duration {
    Duration::from_nanos(n)
}

// --- MonotonicBaseline -----------------------------------------------------

#[test]
fn baseline_accumulates_without_wrap() {
    // 1 GHz, full-width => one tick is one nanosecond.
    let mut base = MonotonicBaseline::full_width(1_000_000_000);
    assert!(!base.is_started());

    // First reading only establishes the baseline (zero delta).
    assert_eq!(base.update(0), ns(0));
    assert!(base.is_started());
    assert_eq!(base.elapsed_ticks(), 0);

    assert_eq!(base.update(1_000), ns(1_000));
    assert_eq!(base.update(3_000), ns(2_000));
    assert_eq!(base.elapsed_ticks(), 3_000);
    assert_eq!(base.elapsed(), ns(3_000));
}

#[test]
fn baseline_absorbs_32_bit_wrap() {
    // 1 GHz counter in a 32-bit register.
    let mut base = MonotonicBaseline::new(1_000_000_000, 32);
    // Start near the top of the 32-bit range.
    assert_eq!(base.update(0xFFFF_FFF0), ns(0));
    // Wrap past 2^32 to 0x10: forward distance is 16 + 16 = 32 ticks.
    assert_eq!(base.update(0x0000_0010), ns(32));
    assert_eq!(base.elapsed_ticks(), 32);
}

#[test]
fn baseline_masks_out_of_width_bits() {
    // 16-bit register: bits above bit 15 are ignored.
    let mut base = MonotonicBaseline::new(1_000_000_000, 16);
    assert_eq!(base.update(0xFFFF_0000), ns(0)); // masks to 0x0000
    assert_eq!(base.update(0xDEAD_0064), ns(100)); // masks to 0x0064 = 100
    assert_eq!(base.elapsed_ticks(), 100);
}

#[test]
fn baseline_converts_frequency_exactly() {
    // 2 ticks per second => one tick is 0.5 s = 500_000_000 ns.
    let mut base = MonotonicBaseline::new(2, 64);
    assert_eq!(base.update(0), ns(0));
    assert_eq!(base.update(3), ns(1_500_000_000)); // 3 ticks = 1.5 s
    assert_eq!(base.elapsed_ticks(), 3);
    assert!((base.elapsed_secs_f64() - 1.5).abs() < 1e-12);
}

#[test]
fn baseline_reset_reestablishes_baseline() {
    let mut base = MonotonicBaseline::full_width(1_000_000_000);
    base.update(0);
    base.update(5_000);
    assert_eq!(base.elapsed_ticks(), 5_000);
    base.reset();
    assert!(!base.is_started());
    assert_eq!(base.elapsed_ticks(), 0);
    // After reset the next reading is a fresh baseline, not a huge jump.
    assert_eq!(base.update(9_000), ns(0));
    assert_eq!(base.update(9_100), ns(100));
}

#[test]
#[should_panic(expected = "width")]
fn baseline_zero_width_panics() {
    let _ = MonotonicBaseline::new(1_000_000_000, 0);
}

#[test]
#[should_panic(expected = "frequency")]
fn baseline_zero_frequency_panics() {
    let _ = MonotonicBaseline::new(0, 32);
}

// --- DriftCorrector --------------------------------------------------------

#[test]
fn no_offset_is_pure_passthrough() {
    let mut dc = DriftCorrector::new();
    assert_eq!(dc.residual_nanos(), 0);
    for i in 1..=5u64 {
        assert_eq!(dc.advance(ns(1_000)), ns(1_000));
        assert_eq!(dc.corrected(), ns(1_000 * i));
    }
    assert_eq!(dc.residual_nanos(), 0);
}

#[test]
fn behind_reference_speeds_up_within_the_bound() {
    // 50% slew for an easy oracle: max slew per step = rd/2.
    let mut dc = DriftCorrector::new().with_max_slew_ppm(500_000);
    dc.observe(ns(10_000)); // corrected is 0 => behind by 10_000 ns
    assert_eq!(dc.residual_nanos(), 10_000);

    // One step: rd = 1_000, max slew = 500, applied slew = +500.
    assert_eq!(dc.advance(ns(1_000)), ns(1_500));
    assert_eq!(dc.corrected(), ns(1_500));
    assert_eq!(dc.residual_nanos(), 9_500);
}

#[test]
fn behind_reference_converges_to_zero_offset() {
    let mut dc = DriftCorrector::new().with_max_slew_ppm(500_000);
    dc.observe(ns(10_000));
    // 10_000 / 500 per step = exactly 20 steps to converge.
    for _ in 0..20 {
        dc.advance(ns(1_000));
    }
    assert_eq!(dc.residual_nanos(), 0);
    assert!(dc.is_converged(ns(0)));
    // Each of the 20 steps applied a 1_500 ns corrected delta.
    assert_eq!(dc.corrected(), ns(30_000));
    // Converged: now a pure passthrough again.
    assert_eq!(dc.advance(ns(1_000)), ns(1_000));
}

#[test]
fn ahead_of_reference_slows_down_but_stays_monotonic() {
    let mut dc = DriftCorrector::new().with_max_slew_ppm(500_000);
    // Reference is behind corrected by 3_000 ns.
    dc.observe_offset(ns(3_000), false);
    assert_eq!(dc.residual_nanos(), -3_000);

    // rd = 1_000, max slew = 500, applied slew = -500 => corrected delta 500.
    assert_eq!(dc.advance(ns(1_000)), ns(500));
    assert_eq!(dc.corrected(), ns(500));
    assert_eq!(dc.residual_nanos(), -2_500);
}

#[test]
fn corrected_clock_never_steps_backward_even_under_max_slew() {
    // Maximum cap and a large negative offset: the clock must still inch
    // forward, never reverse.
    let mut dc = DriftCorrector::new().with_max_slew_ppm(DriftCorrector::MAX_SLEW_PPM);
    dc.observe_offset(ns(1_000_000_000), false); // reference far behind
    let mut prev = dc.corrected();
    for _ in 0..100 {
        let step = dc.advance(ns(1_000));
        // rd = 1_000, max slew floor = 999 => corrected delta >= 1 ns.
        assert_eq!(step, ns(1));
        let now = dc.corrected();
        assert!(now > prev, "corrected clock went backward");
        prev = now;
    }
}

#[test]
fn resync_steps_to_reference_and_reports_jump() {
    let mut dc = DriftCorrector::new();
    dc.advance(ns(10_000_000)); // corrected = 10 ms, residual 0
    assert_eq!(dc.corrected(), ns(10_000_000));

    // Session boundary: step forward to 25 ms.
    let jump = dc.resync(ns(25_000_000));
    assert_eq!(jump, 15_000_000);
    assert_eq!(dc.corrected(), ns(25_000_000));
    assert_eq!(dc.residual_nanos(), 0);

    // Resync can also report a negative jump (reference behind).
    let back = dc.resync(ns(5_000_000));
    assert_eq!(back, -20_000_000);
    assert_eq!(dc.corrected(), ns(5_000_000));
}

#[test]
fn slew_cap_is_clamped_to_the_valid_range() {
    let dc = DriftCorrector::new().with_max_slew_ppm(5_000_000);
    assert_eq!(dc.max_slew_ppm(), DriftCorrector::MAX_SLEW_PPM);

    let mut dc2 = DriftCorrector::new();
    dc2.set_max_slew_ppm(2_000_000);
    assert_eq!(dc2.max_slew_ppm(), DriftCorrector::MAX_SLEW_PPM);
    dc2.set_max_slew_ppm(250);
    assert_eq!(dc2.max_slew_ppm(), 250);
}

#[test]
fn deterministic_double_run_matches_bit_for_bit() {
    fn run() -> (u128, i128) {
        let mut dc = DriftCorrector::new().with_max_slew_ppm(300);
        let mut t = 0u64;
        for frame in 0..200u64 {
            // Re-observe an offset every 50 frames, like periodic NTP samples.
            if frame % 50 == 0 {
                dc.observe(ns(t + 7_000));
            }
            let dt = if frame % 3 == 0 {
                16_000_000
            } else {
                17_000_000
            };
            t += dt;
            dc.advance(ns(dt));
        }
        (dc.corrected_nanos(), dc.residual_nanos())
    }
    assert_eq!(run(), run());
}
