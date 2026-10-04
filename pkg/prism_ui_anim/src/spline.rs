//! Smooth scalar keyframe interpolation using cubic Hermite segments with
//! Catmull-Rom auto-tangents.
//!
//! [`Timeline`](crate::timeline::Timeline) interpolates each segment with an
//! independent [`Easing`](crate::easing::Easing) curve, which is ideal when an
//! author hand-tunes the shape of every segment. For data-driven "float
//! curves" — recorded motion, sampled sensor values, procedurally generated
//! waypoints — authors instead want a single smooth curve that passes through
//! every key with a continuous slope, exactly like the float-curve editors in
//! mature animation tools.
//!
//! [`HermiteCurve`] provides that primitive. Between two adjacent keys it
//! evaluates a cubic Hermite polynomial whose endpoint slopes are derived from
//! neighbouring keys with the (non-uniform) Catmull-Rom rule:
//!
//! ```text
//! m_i = (v_{i+1} - v_{i-1}) / (t_{i+1} - t_{i-1})
//! ```
//!
//! with one-sided slopes at the ends. The result:
//!
//! * **interpolates** every key exactly (`sample(t_i) == v_i`),
//! * is **C1 continuous** (the slope matches `m_i` on both sides of a key),
//! * **reproduces straight lines** — if the keys lie on `v = a*t + b` the
//!   curve is exactly that line everywhere, so it never overshoots linear
//!   data, and
//! * is **affine in value** — scaling/offsetting every key value scales/
//!   offsets the whole curve, because the Hermite value weights sum to one.
//!
//! Multi-channel values (positions, colours) animate component-wise: build one
//! [`HermiteCurve`] per channel. The module is pure arithmetic (no
//! transcendentals), so it is available with and without default features.

use alloc::vec::Vec;

use crate::math::clampf;

/// A single `(time, value)` sample of a scalar float curve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FloatKeyframe {
    /// Time of the key, in the curve's own time units.
    pub time: f32,
    /// Scalar value at `time`.
    pub value: f32,
}

impl FloatKeyframe {
    /// Builds a keyframe at `time` holding `value`.
    #[must_use]
    pub fn new(time: f32, value: f32) -> Self {
        Self { time, value }
    }
}

/// A smooth scalar curve sampled through sorted [`FloatKeyframe`]s with
/// Catmull-Rom auto-tangents.
///
/// Keys are stored sorted by strictly increasing time; keys sharing a time are
/// coalesced (the first occurrence wins) so every segment has positive width.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct HermiteCurve {
    keys: Vec<FloatKeyframe>,
}

impl HermiteCurve {
    /// Creates an empty curve.
    #[must_use]
    pub fn new() -> Self {
        Self { keys: Vec::new() }
    }

    /// Builds a curve from `keys`, sorting them by time and dropping later
    /// keys that duplicate an earlier key's time.
    #[must_use]
    pub fn from_keyframes(mut keys: Vec<FloatKeyframe>) -> Self {
        keys.sort_by(|a, b| a.time.total_cmp(&b.time));
        keys.dedup_by(|a, b| a.time == b.time);
        Self { keys }
    }

    /// Inserts a key, keeping the curve sorted by time. A key whose time
    /// already exists overwrites that key's value.
    pub fn push(&mut self, key: FloatKeyframe) {
        match self.keys.binary_search_by(|k| k.time.total_cmp(&key.time)) {
            Ok(idx) => self.keys[idx] = key,
            Err(idx) => self.keys.insert(idx, key),
        }
    }

    /// Returns the sorted keys backing the curve.
    #[must_use]
    pub fn keyframes(&self) -> &[FloatKeyframe] {
        &self.keys
    }

    /// Returns `true` when the curve holds no keys.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Returns the number of keys.
    #[must_use]
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Returns the span from the first to the last key, or `0.0` when the
    /// curve has fewer than two keys.
    #[must_use]
    pub fn duration(&self) -> f32 {
        match (self.keys.first(), self.keys.last()) {
            (Some(first), Some(last)) => last.time - first.time,
            _ => 0.0,
        }
    }

    /// Returns the Catmull-Rom slope (`dv/dt`) at key `index`, or `0.0` when
    /// the index is out of range or the curve has a single key.
    ///
    /// Interior keys use the centred difference over the surrounding keys;
    /// the first and last keys use the one-sided slope of their only segment.
    #[must_use]
    pub fn tangent_at(&self, index: usize) -> f32 {
        let n = self.keys.len();
        if n < 2 || index >= n {
            return 0.0;
        }
        let lo = if index == 0 { 0 } else { index - 1 };
        let hi = if index == n - 1 { index } else { index + 1 };
        let span = self.keys[hi].time - self.keys[lo].time;
        if span <= 0.0 {
            return 0.0;
        }
        (self.keys[hi].value - self.keys[lo].value) / span
    }

    /// Samples the curve at `time`, or returns `None` when the curve is empty.
    ///
    /// `time` is clamped to the key range, so values before the first key hold
    /// the first value and values after the last key hold the last value.
    #[must_use]
    pub fn try_sample(&self, time: f32) -> Option<f32> {
        let n = self.keys.len();
        if n == 0 {
            return None;
        }
        if n == 1 {
            return Some(self.keys[0].value);
        }
        let first = self.keys[0].time;
        let last = self.keys[n - 1].time;
        let time = clampf(time, first, last);
        if time <= first {
            return Some(self.keys[0].value);
        }
        if time >= last {
            return Some(self.keys[n - 1].value);
        }
        // Find the segment [i, i+1] with keys[i].time <= time < keys[i+1].time.
        // `partition_point` returns the count of keys at or before `time`.
        let upper = self.keys.partition_point(|k| k.time <= time);
        let i = upper - 1;
        let k0 = self.keys[i];
        let k1 = self.keys[i + 1];
        let dt = k1.time - k0.time;
        if dt <= 0.0 {
            return Some(k0.value);
        }
        let m0 = self.tangent_at(i);
        let m1 = self.tangent_at(i + 1);
        let u = (time - k0.time) / dt;
        Some(hermite(k0.value, k1.value, m0, m1, dt, u))
    }

    /// Samples the curve at `time`, returning `0.0` for an empty curve.
    #[must_use]
    pub fn sample(&self, time: f32) -> f32 {
        self.try_sample(time).unwrap_or(0.0)
    }
}

/// Evaluates one cubic Hermite segment.
///
/// `v0`/`v1` are the endpoint values, `m0`/`m1` the endpoint slopes (`dv/dt`),
/// `dt` the segment width in time and `u` the normalized position in `[0, 1]`.
/// The value basis weights (`h00`, `h01`) sum to one, which is what makes the
/// curve affine in the key values.
#[inline]
fn hermite(v0: f32, v1: f32, m0: f32, m1: f32, dt: f32, u: f32) -> f32 {
    let u2 = u * u;
    let u3 = u2 * u;
    let h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
    let h10 = u3 - 2.0 * u2 + u;
    let h01 = -2.0 * u3 + 3.0 * u2;
    let h11 = u3 - u2;
    h00 * v0 + h10 * dt * m0 + h01 * v1 + h11 * dt * m1
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::std_instead_of_alloc,
        reason = "tests run under std where Vec/vec! are convenient"
    )]
    use super::*;
    use alloc::vec;

    struct SplitMix64(u64);
    impl SplitMix64 {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        /// Uniform f32 in `[0, 1)`.
        fn unit(&mut self) -> f32 {
            // Top 24 bits give an exact dyadic fraction with no rounding bias.
            let bits = self.next_u64() >> 40;
            (bits as f32) / (1u32 << 24) as f32
        }
        /// Uniform f32 in `[lo, hi)`.
        fn range(&mut self, lo: f32, hi: f32) -> f32 {
            lo + (hi - lo) * self.unit()
        }
    }

    #[test]
    fn empty_and_single() {
        let empty = HermiteCurve::new();
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);
        assert_eq!(empty.duration(), 0.0);
        assert_eq!(empty.try_sample(0.0), None);
        assert_eq!(empty.sample(0.0), 0.0);

        let single = HermiteCurve::from_keyframes(vec![FloatKeyframe::new(3.0, 7.0)]);
        assert_eq!(single.len(), 1);
        assert_eq!(single.duration(), 0.0);
        assert_eq!(single.sample(-100.0), 7.0);
        assert_eq!(single.sample(3.0), 7.0);
        assert_eq!(single.sample(100.0), 7.0);
    }

    #[test]
    fn sorts_and_coalesces_duplicate_times() {
        let curve = HermiteCurve::from_keyframes(vec![
            FloatKeyframe::new(2.0, 20.0),
            FloatKeyframe::new(0.0, 0.0),
            FloatKeyframe::new(1.0, 10.0),
            FloatKeyframe::new(1.0, 999.0), // duplicate time dropped
        ]);
        let times: Vec<f32> = curve.keyframes().iter().map(|k| k.time).collect();
        assert_eq!(times, vec![0.0, 1.0, 2.0]);
        // First occurrence of t=1.0 wins (value 10.0, not 999.0).
        assert_eq!(curve.keyframes()[1].value, 10.0);
    }

    #[test]
    fn push_keeps_sorted_and_overwrites() {
        let mut curve = HermiteCurve::new();
        curve.push(FloatKeyframe::new(2.0, 2.0));
        curve.push(FloatKeyframe::new(0.0, 0.0));
        curve.push(FloatKeyframe::new(1.0, 1.0));
        curve.push(FloatKeyframe::new(1.0, 5.0)); // overwrite t=1.0
        let times: Vec<f32> = curve.keyframes().iter().map(|k| k.time).collect();
        assert_eq!(times, vec![0.0, 1.0, 2.0]);
        assert_eq!(curve.keyframes()[1].value, 5.0);
    }

    #[test]
    fn interpolates_every_key_exactly() {
        // ROUND-TRIP ORACLE: the curve must pass through every key value.
        let mut rng = SplitMix64(0x0FED_CBA9_8765_4321);
        for _ in 0..2000 {
            let n = 2 + (rng.next_u64() % 7) as usize;
            let mut t = rng.range(-5.0, 5.0);
            let mut keys = Vec::with_capacity(n);
            for _ in 0..n {
                keys.push(FloatKeyframe::new(t, rng.range(-100.0, 100.0)));
                t += rng.range(0.05, 2.0);
            }
            let curve = HermiteCurve::from_keyframes(keys.clone());
            for k in &keys {
                let got = curve.sample(k.time);
                assert!(
                    (got - k.value).abs() <= 1e-3 * (1.0 + k.value.abs()),
                    "sample({}) = {got}, expected {}",
                    k.time,
                    k.value
                );
            }
        }
    }

    #[test]
    fn reproduces_straight_lines_exactly() {
        // STRONG ORACLE: Catmull-Rom tangents of collinear data equal the line
        // slope, and a cubic Hermite with the correct slopes reproduces a
        // degree-1 polynomial exactly. We recompute the expected value with an
        // entirely independent method: direct line evaluation `a*t + b`.
        let mut rng = SplitMix64(0x00C0_FFEE_0BAD_F00D);
        for _ in 0..2000 {
            let a = rng.range(-10.0, 10.0);
            let b = rng.range(-50.0, 50.0);
            let n = 2 + (rng.next_u64() % 8) as usize;
            let mut t = rng.range(-5.0, 5.0);
            let mut keys = Vec::with_capacity(n);
            let (t_first, mut t_last) = (t, t);
            for _ in 0..n {
                keys.push(FloatKeyframe::new(t, a * t + b));
                t_last = t;
                t += rng.range(0.05, 2.0);
            }
            let curve = HermiteCurve::from_keyframes(keys);
            // Dense independent resampling across the whole span.
            for s in 0..=200 {
                let x = t_first + (t_last - t_first) * (s as f32 / 200.0);
                let expected = a * x + b;
                let got = curve.sample(x);
                let tol = 1e-3 * (1.0 + expected.abs());
                assert!(
                    (got - expected).abs() <= tol,
                    "line a={a} b={b}: sample({x}) = {got}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn affine_in_value() {
        // STRONG ORACLE: because the Hermite value weights sum to one and the
        // tangents are differences of values, the curve is affine in the key
        // values. Verify f(a*v + b) == a*f(v) + b for independent random
        // affine maps, sampled against a freshly transformed curve.
        let mut rng = SplitMix64(0x5151_5151_A0A0_A0A0);
        for _ in 0..2000 {
            let a = rng.range(-6.0, 6.0);
            let b = rng.range(-30.0, 30.0);
            let n = 2 + (rng.next_u64() % 8) as usize;
            let mut t = rng.range(-5.0, 5.0);
            let mut base = Vec::with_capacity(n);
            let mut mapped = Vec::with_capacity(n);
            let (t_first, mut t_last) = (t, t);
            for _ in 0..n {
                let v = rng.range(-100.0, 100.0);
                base.push(FloatKeyframe::new(t, v));
                mapped.push(FloatKeyframe::new(t, a * v + b));
                t_last = t;
                t += rng.range(0.05, 2.0);
            }
            let base = HermiteCurve::from_keyframes(base);
            let mapped = HermiteCurve::from_keyframes(mapped);
            for s in 0..=100 {
                let x = t_first + (t_last - t_first) * (s as f32 / 100.0);
                let expected = a * base.sample(x) + b;
                let got = mapped.sample(x);
                let tol = 1e-2 * (1.0 + expected.abs());
                assert!(
                    (got - expected).abs() <= tol,
                    "affine a={a} b={b}: got {got}, expected {expected} at x={x}"
                );
            }
        }
    }

    #[test]
    fn two_keys_match_linear_interpolation() {
        // With exactly two keys the Catmull-Rom slopes on both ends equal the
        // chord slope, so the curve is the straight chord: an independent
        // lerp is the oracle.
        let mut rng = SplitMix64(0x2222_3333_4444_5555);
        for _ in 0..2000 {
            let t0 = rng.range(-5.0, 5.0);
            let t1 = t0 + rng.range(0.05, 3.0);
            let v0 = rng.range(-50.0, 50.0);
            let v1 = rng.range(-50.0, 50.0);
            let curve = HermiteCurve::from_keyframes(vec![
                FloatKeyframe::new(t0, v0),
                FloatKeyframe::new(t1, v1),
            ]);
            for s in 0..=50 {
                let f = s as f32 / 50.0;
                let x = t0 + (t1 - t0) * f;
                let expected = v0 + (v1 - v0) * f;
                let got = curve.sample(x);
                let tol = 1e-3 * (1.0 + expected.abs());
                assert!(
                    (got - expected).abs() <= tol,
                    "lerp: got {got}, expected {expected} at f={f}"
                );
            }
        }
    }

    #[test]
    fn c1_continuous_at_interior_keys() {
        // Estimate the one-sided slope on each side of every interior key with
        // a second-order one-sided difference (truncation O(h^2)), then assert
        // both sides converge to the same value and to the analytic tangent.
        // A slope jump at a key would break C1 continuity.
        let mut rng = SplitMix64(0x7A7A_7A7A_1B1B_1B1B);
        for _ in 0..500 {
            let n = 3 + (rng.next_u64() % 6) as usize;
            let mut t = rng.range(-3.0, 3.0);
            let mut keys = Vec::with_capacity(n);
            for _ in 0..n {
                keys.push(FloatKeyframe::new(t, rng.range(-20.0, 20.0)));
                t += rng.range(1.0, 2.0);
            }
            let curve = HermiteCurve::from_keyframes(keys);
            let ks = curve.keyframes();
            for i in 1..ks.len() - 1 {
                let ti = ks[i].time;
                let left_gap = ti - ks[i - 1].time;
                let right_gap = ks[i + 1].time - ti;
                let h = 1e-2 * left_gap.min(right_gap);
                let f = |x: f32| curve.sample(x);
                // Second-order one-sided derivatives at `ti`.
                let left = (3.0 * f(ti) - 4.0 * f(ti - h) + f(ti - 2.0 * h)) / (2.0 * h);
                let right = (-3.0 * f(ti) + 4.0 * f(ti + h) - f(ti + 2.0 * h)) / (2.0 * h);
                let analytic = curve.tangent_at(i);
                let tol = 2e-2 * (1.0 + analytic.abs());
                assert!(
                    (left - right).abs() <= tol,
                    "slope jump at key {i}: left {left} right {right}"
                );
                assert!(
                    (left - analytic).abs() <= tol && (right - analytic).abs() <= tol,
                    "slope != tangent at key {i}: left {left} right {right} analytic {analytic}"
                );
            }
        }
    }

    #[test]
    fn clamps_outside_the_key_range() {
        let curve = HermiteCurve::from_keyframes(vec![
            FloatKeyframe::new(0.0, 1.0),
            FloatKeyframe::new(1.0, 2.0),
            FloatKeyframe::new(2.0, -3.0),
        ]);
        assert_eq!(curve.sample(-10.0), 1.0);
        assert_eq!(curve.sample(10.0), -3.0);
        assert_eq!(curve.duration(), 2.0);
    }
}
