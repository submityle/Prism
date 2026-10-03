//! M4 fixed-point determinism tests.
//!
//! Covers: raw-bit / integer / float round-trips, operator correctness against
//! an exact rational (`i128`) reference, transcendental accuracy within stated
//! error bounds versus `f64` references, compensated summation versus a naive
//! `f32` sum, and — the headline guarantee — cross-platform bit-exactness via
//! golden [`StateHasher`] digests plus a double-run agreement check.
//!
//! Tests run with `std` available (the crate is only `no_std` for non-test
//! builds), so `f64` library functions are used purely as *reference oracles*;
//! they never participate in the deterministic `Fixed` path under test.

use crate::prelude::*;

// --------------------------------------------------------------------------
// helpers
// --------------------------------------------------------------------------

/// Reference value of a [`Fixed`] as an exact rational `raw / 2^32` in `f64`.
fn ref_f64(x: Fixed) -> f64 {
    (x.to_bits() as f64) / ((1i64 << 32) as f64)
}

fn approx(a: f64, b: f64, eps: f64) -> bool {
    (a - b).abs() <= eps
}

// --------------------------------------------------------------------------
// round-trips
// --------------------------------------------------------------------------

#[test]
fn fixed_bits_round_trip() {
    for &raw in &[0i64, 1, -1, i64::MAX, i64::MIN, 1 << 32, -(1 << 32), 123_456_789] {
        assert_eq!(Fixed::from_bits(raw).to_bits(), raw);
    }
}

#[test]
fn fixed_int_round_trip() {
    for n in -1000i64..=1000 {
        assert_eq!(Fixed::from_int(n).to_int(), n);
    }
    // Exact integer constants land on exact raw multiples of ONE_BITS.
    assert_eq!(Fixed::from_int(5).to_bits(), 5 * Fixed::ONE_BITS);
    assert_eq!(Fixed::from_int(-3).to_bits(), -3 * Fixed::ONE_BITS);
}

#[test]
fn fixed_float_round_trip_is_tight() {
    for &v in &[0.0f64, 1.0, -1.0, 0.5, -0.25, 42.123, -123.456, 1000.0] {
        let f = Fixed::from_f64(v);
        assert!(approx(f.to_f64(), v, 1e-9), "{v} -> {}", f.to_f64());
    }
    // f32 entry point agrees with the f64 one.
    assert_eq!(Fixed::from_f32(0.5f32).to_bits(), Fixed::HALF.to_bits());
}

#[test]
fn fixed_constants_have_expected_bits() {
    // Precomputed Q32.32 constants (value * 2^32, rounded).
    assert_eq!(Fixed::ONE.to_bits(), 1 << 32);
    assert_eq!(Fixed::HALF.to_bits(), 1 << 31);
    assert_eq!(Fixed::TWO.to_bits(), 1 << 33);
    assert_eq!(Fixed::NEG_ONE.to_bits(), -(1 << 32));
    // Mathematical constants match f64 to sub-ulp.
    assert!(approx(ref_f64(Fixed::PI), core::f64::consts::PI, 1e-9));
    assert!(approx(ref_f64(Fixed::TAU), core::f64::consts::TAU, 1e-9));
    assert!(approx(ref_f64(Fixed::FRAC_PI_2), core::f64::consts::FRAC_PI_2, 1e-9));
    assert!(approx(ref_f64(Fixed::LN_2), core::f64::consts::LN_2, 1e-9));
    assert!(approx(ref_f64(Fixed::E), core::f64::consts::E, 1e-9));
}

#[test]
fn i16f16_round_trips_and_widening() {
    for n in -1000i32..=1000 {
        assert_eq!(I16F16::from_int(n).to_int(), n);
    }
    // Widen Q16.16 -> Q32.32 is exact; narrow back recovers the value.
    for &raw in &[0i32, 1, -1, 1 << 16, -(1 << 16), 12345, -98765] {
        let a = I16F16::from_bits(raw);
        let wide = a.to_fixed();
        assert_eq!(wide.to_bits(), (raw as i64) << 16);
        assert_eq!(I16F16::from_fixed(wide).to_bits(), raw);
    }
    // 0.5 is representable exactly in both formats.
    assert_eq!(I16F16::HALF.to_fixed().to_bits(), Fixed::HALF.to_bits());
}

// --------------------------------------------------------------------------
// operator correctness vs exact rational (i128) reference
// --------------------------------------------------------------------------

/// Exact Q32.32 add on raw bits (saturating), as an independent oracle.
fn ref_add(a: i64, b: i64) -> i64 {
    (a as i128 + b as i128).clamp(i64::MIN as i128, i64::MAX as i128) as i64
}
/// Exact Q32.32 multiply with round-to-nearest (ties toward +inf), saturating.
fn ref_mul(a: i64, b: i64) -> i64 {
    let p = a as i128 * b as i128;
    let rounded = (p + (1 << 31)) >> 32;
    rounded.clamp(i64::MIN as i128, i64::MAX as i128) as i64
}
/// Exact Q32.32 divide truncating toward zero, saturating.
fn ref_div(a: i64, b: i64) -> i64 {
    if b == 0 {
        return if a > 0 { i64::MAX } else if a < 0 { i64::MIN } else { 0 };
    }
    (((a as i128) << 32) / b as i128).clamp(i64::MIN as i128, i64::MAX as i128) as i64
}

#[test]
fn operators_match_rational_reference() {
    let samples = [
        0i64,
        Fixed::ONE_BITS,
        -Fixed::ONE_BITS,
        3 * Fixed::ONE_BITS,
        Fixed::ONE_BITS / 2,
        -(Fixed::ONE_BITS / 4),
        1_234_567_890,
        -987_654_321,
        Fixed::PI.to_bits(),
        Fixed::E.to_bits(),
    ];
    for &ar in &samples {
        for &br in &samples {
            let a = Fixed::from_bits(ar);
            let b = Fixed::from_bits(br);
            assert_eq!((a + b).to_bits(), ref_add(ar, br), "add {ar}+{br}");
            assert_eq!((a * b).to_bits(), ref_mul(ar, br), "mul {ar}*{br}");
            assert_eq!((a / b).to_bits(), ref_div(ar, br), "div {ar}/{br}");
        }
    }
}

#[test]
fn sub_neg_match_reference() {
    let samples = [0i64, Fixed::ONE_BITS, -Fixed::ONE_BITS, 1_234_567_890, -987_654_321, i64::MAX, i64::MIN];
    for &ar in &samples {
        for &br in &samples {
            let a = Fixed::from_bits(ar);
            let b = Fixed::from_bits(br);
            let expect = (ar as i128 - br as i128).clamp(i64::MIN as i128, i64::MAX as i128) as i64;
            assert_eq!((a - b).to_bits(), expect, "sub {ar}-{br}");
        }
        // negation saturates: -MIN == MAX.
        let expect_neg = (-(ar as i128)).clamp(i64::MIN as i128, i64::MAX as i128) as i64;
        assert_eq!((-Fixed::from_bits(ar)).to_bits(), expect_neg, "neg {ar}");
    }
}

#[test]
fn saturation_is_deterministic() {
    assert_eq!(Fixed::MAX + Fixed::ONE, Fixed::MAX);
    assert_eq!(Fixed::MIN - Fixed::ONE, Fixed::MIN);
    assert_eq!(-Fixed::MIN, Fixed::MAX);
    assert_eq!(Fixed::MAX.abs(), Fixed::MAX);
    assert_eq!(Fixed::MIN.abs(), Fixed::MAX);
    // divide-by-zero saturates by sign; 0/0 == 0.
    assert_eq!(Fixed::ONE / Fixed::ZERO, Fixed::MAX);
    assert_eq!(Fixed::NEG_ONE / Fixed::ZERO, Fixed::MIN);
    assert_eq!(Fixed::ZERO / Fixed::ZERO, Fixed::ZERO);
}

#[test]
fn checked_and_wrapping_arithmetic() {
    assert_eq!(Fixed::MAX.checked_add(Fixed::ONE), None);
    assert_eq!(Fixed::ONE.checked_add(Fixed::ONE), Some(Fixed::TWO));
    assert_eq!(Fixed::ONE.checked_div(Fixed::ZERO), None);
    // wrapping add wraps around the raw i64.
    assert_eq!(Fixed::MAX.wrapping_add(Fixed::EPSILON).to_bits(), i64::MIN);
}

#[test]
fn rounding_helpers() {
    let a = Fixed::from_f64(2.75);
    assert_eq!(a.floor(), Fixed::from_int(2));
    assert_eq!(a.ceil(), Fixed::from_int(3));
    assert_eq!(a.round(), Fixed::from_int(3));
    assert_eq!(a.trunc(), Fixed::from_int(2));
    assert!(approx(a.fract().to_f64(), 0.75, 1e-9));
    let b = Fixed::from_f64(-2.25);
    assert_eq!(b.floor(), Fixed::from_int(-3));
    assert_eq!(b.ceil(), Fixed::from_int(-2));
    assert_eq!(b.trunc(), Fixed::from_int(-2));
}

// --------------------------------------------------------------------------
// transcendental accuracy (vs f64 oracle, within stated bounds)
// --------------------------------------------------------------------------

#[test]
fn sqrt_accuracy() {
    // sqrt accurate to < 1e-6 over a wide domain; sqrt of negatives is 0.
    let mut x = 0.0f64;
    while x <= 1000.0 {
        let got = ref_f64(Fixed::from_f64(x).sqrt());
        assert!(approx(got, x.sqrt(), 1e-6), "sqrt({x}) = {got}");
        x += 0.37;
    }
    assert_eq!(Fixed::from_int(-5).sqrt(), Fixed::ZERO);
    // perfect squares are near-exact.
    assert!(approx(ref_f64(Fixed::from_int(144).sqrt()), 12.0, 1e-6));
}

#[test]
fn sin_cos_accuracy() {
    let mut t = -7.0f64;
    while t <= 7.0 {
        let s = ref_f64(Fixed::from_f64(t).sin());
        let c = ref_f64(Fixed::from_f64(t).cos());
        assert!(approx(s, t.sin(), 2e-5), "sin({t}) = {s} vs {}", t.sin());
        assert!(approx(c, t.cos(), 2e-5), "cos({t}) = {c} vs {}", t.cos());
        t += 0.013;
    }
    // Pythagorean identity holds in fixed point.
    let a = Fixed::from_f64(1.2345);
    let (s, c) = a.sin_cos();
    assert!(approx(ref_f64(s * s + c * c), 1.0, 1e-4));
}

#[test]
fn atan2_accuracy_all_quadrants() {
    let pts = [
        (1.0, 1.0),
        (-1.0, 1.0),
        (-1.0, -1.0),
        (1.0, -1.0),
        (0.0, 1.0),
        (0.0, -1.0),
        (1.0, 0.0),
        (-1.0, 0.0),
        (3.0, 4.0),
        (-7.0, 2.0),
        (0.25, -8.0),
    ];
    for &(y, x) in &pts {
        let got = ref_f64(Fixed::from_f64(y).atan2(Fixed::from_f64(x)));
        assert!(approx(got, y.atan2(x), 3e-4), "atan2({y},{x}) = {got} vs {}", y.atan2(x));
    }
    assert_eq!(Fixed::ZERO.atan2(Fixed::ZERO), Fixed::ZERO);
}

#[test]
fn exp_accuracy() {
    let mut x = -10.0f64;
    while x <= 10.0 {
        let got = ref_f64(Fixed::from_f64(x).exp());
        let want = x.exp();
        assert!((got - want).abs() <= want.abs() * 1e-4 + 1e-9, "exp({x}) = {got} vs {want}");
        x += 0.11;
    }
    assert!(approx(ref_f64(Fixed::ZERO.exp()), 1.0, 1e-9));
}

#[test]
fn ln_accuracy() {
    let mut x = 1e-3f64;
    while x <= 1e3 {
        let got = ref_f64(Fixed::from_f64(x).ln());
        assert!(approx(got, x.ln(), 1e-4), "ln({x}) = {got} vs {}", x.ln());
        x *= 1.07;
    }
    // ln of non-positive input returns the MIN sentinel.
    assert_eq!(Fixed::ZERO.ln(), Fixed::MIN);
    assert_eq!(Fixed::NEG_ONE.ln(), Fixed::MIN);
    // exp/ln are inverse within bound.
    assert!(approx(ref_f64(Fixed::E.ln()), 1.0, 1e-4));
}

// --------------------------------------------------------------------------
// fixed-point vectors
// --------------------------------------------------------------------------

#[test]
fn fxvec_algebra() {
    let a = fxvec3(Fixed::from_int(1), Fixed::from_int(2), Fixed::from_int(3));
    let b = fxvec3(Fixed::from_int(4), Fixed::from_int(5), Fixed::from_int(6));
    assert_eq!(a + b, fxvec3(Fixed::from_int(5), Fixed::from_int(7), Fixed::from_int(9)));
    assert_eq!(b - a, fxvec3(Fixed::from_int(3), Fixed::from_int(3), Fixed::from_int(3)));
    assert_eq!(a.dot(b), Fixed::from_int(32));
    assert_eq!(FxVec3::X.cross(FxVec3::Y), FxVec3::Z);
    assert_eq!(a * Fixed::TWO, fxvec3(Fixed::from_int(2), Fixed::from_int(4), Fixed::from_int(6)));
    assert_eq!(Fixed::TWO * a, a * Fixed::TWO);
}

#[test]
fn fxvec_length_and_normalize() {
    let v = fxvec2(Fixed::from_int(3), Fixed::from_int(4));
    assert!(approx(ref_f64(v.length()), 5.0, 1e-6));
    let n = v.normalize_or_zero(Fixed::EPSILON);
    assert!(approx(ref_f64(n.length()), 1.0, 1e-4));
    assert_eq!(FxVec2::ZERO.normalize_or_zero(Fixed::EPSILON), FxVec2::ZERO);
    // lerp midpoint.
    let m = FxVec2::ZERO.lerp(fxvec2(Fixed::from_int(2), Fixed::from_int(4)), Fixed::HALF);
    assert_eq!(m, fxvec2(Fixed::ONE, Fixed::TWO));
}

// --------------------------------------------------------------------------
// compensated summation vs naive f32 sum
// --------------------------------------------------------------------------

#[test]
fn compensated_sum_beats_naive() {
    // A classic ill-conditioned sum: 1 + many tiny terms. The exact total is
    // 1.0 + 1e6 * 1e-3 = 1001.0 (choose terms that cancel the naive drift).
    let mut data: Vec<f32> = Vec::new();
    data.push(1.0e8f32);
    data.extend(core::iter::repeat_n(1.0f32, 1_000_000));
    data.push(-1.0e8f32);
    let exact = 1_000_000.0f32;

    let mut naive = 0.0f32;
    for &v in &data {
        naive += v;
    }
    let kahan = kahan_sum(&data);
    let neumaier = neumaier_sum(&data);

    let naive_err = (naive - exact).abs();
    let kahan_err = (kahan - exact).abs();
    let neumaier_err = (neumaier - exact).abs();

    assert!(kahan_err <= naive_err, "kahan {kahan_err} should beat naive {naive_err}");
    assert!(neumaier_err <= naive_err, "neumaier {neumaier_err} should beat naive {naive_err}");
    assert!(neumaier_err <= 1.0, "neumaier error {neumaier_err} too large");
}

#[test]
fn neumaier_handles_large_then_small() {
    // Neumaier is specifically more robust than Kahan here.
    let data = [1.0f64, 1.0e100, 1.0, -1.0e100];
    assert_eq!(neumaier_sum(&data), 2.0);
    // accumulator AddAssign API.
    let mut acc = NeumaierSum::<f64>::new();
    for &v in &data {
        acc += v;
    }
    assert_eq!(acc.sum(), 2.0);
    let mut k = KahanSum::<f32>::new();
    k += 1.0;
    k += 2.0;
    k += 3.0;
    assert_eq!(k.sum(), 6.0);
}

// --------------------------------------------------------------------------
// cross-platform bit-exactness: golden hashes + double-run agreement
// --------------------------------------------------------------------------

/// A deterministic walk over the scalar `Fixed` surface, hashed by raw bits.
fn scalar_digest() -> u64 {
    let mut h = StateHasher::new();
    let a = Fixed::from_int(3);
    let b = Fixed::from_int(7);
    h.write_fixed(a + b);
    h.write_fixed(a - b);
    h.write_fixed(a * b);
    h.write_fixed(b / a);
    h.write_fixed(b % a);
    h.write_fixed(Fixed::from_int(2).sqrt());
    h.write_fixed(Fixed::from_int(144).sqrt());
    h.write_fixed(Fixed::FRAC_PI_4.sin());
    h.write_fixed(Fixed::FRAC_PI_4.cos());
    h.write_fixed(Fixed::PI.sin());
    h.write_fixed(Fixed::ONE.exp());
    h.write_fixed(Fixed::E.ln());
    h.write_fixed(Fixed::ONE.atan2(Fixed::ONE));
    h.write_fixed(Fixed::NEG_ONE.atan2(Fixed::NEG_ONE));
    h.finish()
}

/// A deterministic walk over the vector surface, hashed by raw bits.
fn vector_digest() -> u64 {
    let mut h = StateHasher::new();
    let a = fxvec3(Fixed::from_int(1), Fixed::from_int(2), Fixed::from_int(3));
    let b = fxvec3(Fixed::from_int(4), Fixed::from_int(5), Fixed::from_int(6));
    h.write_fxvec3(a + b);
    h.write_fxvec3(a.cross(b));
    h.write_fixed(a.dot(b));
    h.write_fxvec2(fxvec2(Fixed::from_int(3), Fixed::from_int(4)));
    h.write_fxvec4(fxvec4(Fixed::ONE, Fixed::TWO, Fixed::HALF, Fixed::NEG_ONE));
    h.write_i16f16(I16F16::from_int(42));
    h.finish()
}

#[test]
fn golden_hashes_are_bit_exact() {
    // These constants are the authoritative cross-platform expectation. If a
    // change to the fixed-point algorithms alters any raw bit, these fail and
    // must be re-reviewed (a determinism-breaking change).
    assert_eq!(scalar_digest(), 3_089_695_298_389_754_697);
    assert_eq!(vector_digest(), 10_664_501_399_365_532_450);
}

#[test]
fn double_run_agreement() {
    // Running the identical op sequence twice yields byte-identical digests —
    // the double-run determinism contract.
    assert_eq!(scalar_digest(), scalar_digest());
    assert_eq!(vector_digest(), vector_digest());
    // Order sensitivity: a different op order must change the digest (proving
    // the hash actually depends on the full sequence, not a fixed constant).
    let mut h1 = StateHasher::new();
    h1.write_fixed(Fixed::ONE);
    h1.write_fixed(Fixed::TWO);
    let mut h2 = StateHasher::new();
    h2.write_fixed(Fixed::TWO);
    h2.write_fixed(Fixed::ONE);
    assert_ne!(h1.finish(), h2.finish());
}

