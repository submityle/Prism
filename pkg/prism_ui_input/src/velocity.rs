//! Pointer velocity estimation for flings and momentum scrolling.
//!
//! A [`VelocityTracker`] accumulates recent pointer samples and fits a
//! low-degree polynomial to each axis by unweighted least squares, then reports
//! the instantaneous velocity — the fitted first derivative — at the most
//! recent sample. This is the estimator Flutter and Android use to turn a drag
//! *release* into a fling: the recovered velocity seeds momentum scrolling and
//! inertial animation.
//!
//! # Why a polynomial fit
//!
//! A naive "last two points" difference is dominated by pointer jitter and the
//! final, often-stationary, touch-up sample. Fitting a degree-2 polynomial over
//! a short recent window (the `HORIZON_MS` horizon) smooths jitter while still
//! tracking acceleration, and evaluating its derivative at the newest sample
//! gives a stable release velocity.
//!
//! # Determinism
//!
//! The fit runs in `f64` using only `+ - * /` (no transcendental functions), so
//! it is bit-reproducible across targets. Sample times are normalised relative
//! to the newest sample, keeping the normal-equations system well-conditioned.

use alloc::vec::Vec;

use crate::geometry::Point;

/// Longest history retained, matching Android's `VelocityTracker` horizon.
const MAX_SAMPLES: usize = 20;

/// Only samples within this many milliseconds of the newest sample feed the
/// fit; older samples are stale and would bias a release velocity.
const HORIZON_MS: u64 = 100;

#[derive(Clone, Copy, Debug)]
struct Sample {
    t_ms: u64,
    x: f32,
    y: f32,
}

/// Accumulates pointer samples and estimates the current velocity in logical
/// pixels per second.
///
/// Feed every move with [`VelocityTracker::record`] and read the release
/// velocity with [`VelocityTracker::velocity`]. Reuse one tracker per active
/// pointer, calling [`VelocityTracker::clear`] when a new gesture starts.
///
/// ```
/// use prism_ui_input::VelocityTracker;
/// use prism_ui_input::geometry::Point;
///
/// let mut tracker = VelocityTracker::new();
/// // A finger moving at 300 px/s in x, sampled every 16 ms.
/// for i in 0..5u64 {
///     let t = i * 16;
///     let secs = t as f32 / 1000.0;
///     tracker.record(t, Point::new(300.0 * secs, 0.0));
/// }
/// let v = tracker.velocity().expect("enough samples");
/// assert!((v.x - 300.0).abs() < 1.0);
/// assert!(v.y.abs() < 1.0);
/// ```
#[derive(Clone, Default)]
pub struct VelocityTracker {
    samples: Vec<Sample>,
}

impl VelocityTracker {
    /// Creates an empty tracker.
    #[must_use]
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
        }
    }

    /// Discards all recorded samples, readying the tracker for a new gesture.
    pub fn clear(&mut self) {
        self.samples.clear();
    }

    /// Returns the number of retained samples.
    #[must_use]
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Returns `true` when no samples are retained.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Records a pointer `position` observed at monotonic time `t_ms`.
    ///
    /// If `t_ms` precedes the previous sample's timestamp the clock is assumed
    /// to belong to a fresh interaction and the history is restarted. History
    /// is capped at `MAX_SAMPLES`; the oldest sample is evicted first.
    pub fn record(&mut self, t_ms: u64, position: Point<f32>) {
        if let Some(last) = self.samples.last()
            && t_ms < last.t_ms
        {
            self.samples.clear();
        }
        self.samples.push(Sample {
            t_ms,
            x: position.x,
            y: position.y,
        });
        if self.samples.len() > MAX_SAMPLES {
            self.samples.remove(0);
        }
    }

    /// Estimates the instantaneous velocity in logical pixels per second.
    ///
    /// Returns `None` when fewer than two samples fall inside the horizon or the
    /// fit is degenerate (for example every retained sample shares a timestamp).
    #[must_use]
    pub fn velocity(&self) -> Option<Point<f32>> {
        let newest = self.samples.last()?;
        let mut times = Vec::new();
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for s in &self.samples {
            let dt = newest.t_ms - s.t_ms;
            if dt <= HORIZON_MS {
                // Normalise to seconds relative to the newest sample (t == 0).
                times.push(-(dt as f64) / 1000.0);
                xs.push(f64::from(s.x));
                ys.push(f64::from(s.y));
            }
        }
        let vx = fit_axis(&times, &xs)?;
        let vy = fit_axis(&times, &ys)?;
        Some(Point::new(vx as f32, vy as f32))
    }
}

/// Fits the highest workable polynomial degree and returns the velocity (the
/// linear coefficient, i.e. the derivative at the normalised time origin).
///
/// Prefers a degree-2 fit when at least three samples are present, falling back
/// to a linear fit if the quadratic system is singular, then to `None`.
fn fit_axis(times: &[f64], values: &[f64]) -> Option<f64> {
    let n = times.len();
    if n < 2 {
        return None;
    }
    let highest = if n >= 3 { 2 } else { 1 };
    let mut degree = highest;
    while degree >= 1 {
        if let Some(v) = fit_velocity(times, values, degree) {
            return Some(v);
        }
        degree -= 1;
    }
    None
}

/// Solves the unweighted least-squares normal equations for a polynomial of the
/// given `degree` and returns its linear coefficient.
fn fit_velocity(times: &[f64], values: &[f64], degree: usize) -> Option<f64> {
    let m = degree + 1;
    // Power sums s_k = sum(t^k) for k in 0..=2*degree populate the symmetric
    // normal matrix, whose (i, j) entry is s_{i+j}.
    let mut power_sums = [0.0_f64; 5];
    for &t in times {
        let mut p = 1.0;
        for ps in power_sums.iter_mut().take(2 * degree + 1) {
            *ps += p;
            p *= t;
        }
    }
    let mut mat = [[0.0_f64; 3]; 3];
    let mut rhs = [0.0_f64; 3];
    for (i, mrow) in mat.iter_mut().enumerate().take(m) {
        for (j, cell) in mrow.iter_mut().enumerate().take(m) {
            *cell = power_sums[i + j];
        }
    }
    for (k, &t) in times.iter().enumerate() {
        let mut p = 1.0;
        for r in rhs.iter_mut().take(m) {
            *r += values[k] * p;
            p *= t;
        }
    }
    let solution = solve(&mut mat, &mut rhs, m)?;
    Some(solution[1])
}

/// Gaussian elimination with partial pivoting for an `m`x`m` system (`m <= 3`).
fn solve(mat: &mut [[f64; 3]; 3], rhs: &mut [f64; 3], m: usize) -> Option<[f64; 3]> {
    for col in 0..m {
        let mut pivot = col;
        let mut best = abs64(mat[col][col]);
        for (row, mrow) in mat.iter().enumerate().take(m).skip(col + 1) {
            let v = abs64(mrow[col]);
            if v > best {
                best = v;
                pivot = row;
            }
        }
        if best < 1e-12 {
            return None;
        }
        if pivot != col {
            mat.swap(col, pivot);
            rhs.swap(col, pivot);
        }
        for row in (col + 1)..m {
            let factor = mat[row][col] / mat[col][col];
            #[expect(
                clippy::needless_range_loop,
                reason = "elimination cross-indexes the pivot row and target row of the same matrix; index form is clearer than split_at_mut"
            )]
            for c in col..m {
                mat[row][c] -= factor * mat[col][c];
            }
            rhs[row] -= factor * rhs[col];
        }
    }
    let mut x = [0.0_f64; 3];
    for col in (0..m).rev() {
        let mut sum = rhs[col];
        for c in (col + 1)..m {
            sum -= mat[col][c] * x[c];
        }
        x[col] = sum / mat[col][col];
    }
    Some(x)
}

#[inline]
fn abs64(v: f64) -> f64 {
    if v < 0.0 {
        -v
    } else {
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn next_rand(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn rand_in(state: &mut u64, lo: f32, hi: f32) -> f32 {
        let bits = (next_rand(state) >> 40) as u32;
        let unit = (bits as f32) / 16_777_216.0;
        lo + (hi - lo) * unit
    }

    #[test]
    fn recovers_constant_velocity() {
        let mut vt = VelocityTracker::new();
        let (vx, vy) = (200.0_f32, -120.0_f32);
        for i in 0..6u64 {
            let t = i * 16;
            let secs = t as f32 / 1000.0;
            vt.record(t, Point::new(10.0 + vx * secs, 5.0 + vy * secs));
        }
        let v = vt.velocity().expect("velocity");
        assert!((v.x - vx).abs() < 0.5, "vx {}", v.x);
        assert!((v.y - vy).abs() < 0.5, "vy {}", v.y);
    }

    #[test]
    fn recovers_quadratic_instantaneous_velocity() {
        // Position x(t) = a + b t + c t^2; the instantaneous velocity at the
        // newest sample t_last is b + 2 c t_last. Oracle recomputed by hand.
        let mut vt = VelocityTracker::new();
        let (a, b, c) = (3.0_f64, 50.0_f64, 400.0_f64);
        for &t in &[0u64, 20, 40, 60, 80] {
            let secs = t as f64 / 1000.0;
            let x = a + b * secs + c * secs * secs;
            vt.record(t, Point::new(x as f32, 0.0));
        }
        let t_last = 80.0_f64 / 1000.0;
        let expected = b + 2.0 * c * t_last; // 50 + 64 = 114
        let v = vt.velocity().expect("velocity");
        assert!(
            (f64::from(v.x) - expected).abs() < 1.0,
            "vx {} vs {expected}",
            v.x
        );
    }

    #[test]
    fn needs_two_samples() {
        let mut vt = VelocityTracker::new();
        assert!(vt.velocity().is_none());
        vt.record(0, Point::new(0.0, 0.0));
        assert!(vt.velocity().is_none());
    }

    #[test]
    fn stationary_is_zero() {
        let mut vt = VelocityTracker::new();
        for i in 0..5u64 {
            vt.record(i * 16, Point::new(7.0, 7.0));
        }
        let v = vt.velocity().expect("velocity");
        assert!(v.x.abs() < 1e-3 && v.y.abs() < 1e-3, "{v:?}");
    }

    #[test]
    fn backward_time_restarts() {
        let mut vt = VelocityTracker::new();
        for i in 0..5u64 {
            vt.record(i * 16, Point::new((i * 160) as f32, 0.0));
        }
        // A new gesture whose clock precedes the last sample clears history.
        vt.record(0, Point::new(0.0, 0.0));
        assert_eq!(vt.len(), 1);
        assert!(vt.velocity().is_none());
    }

    #[test]
    fn horizon_excludes_stale_samples() {
        let mut vt = VelocityTracker::new();
        // A wild outlier far in the past, then a clean recent run at 300 px/s.
        vt.record(0, Point::new(-9999.0, 0.0));
        let vx = 300.0_f32;
        for i in 0..5u64 {
            let t = 1000 + i * 16;
            let secs = t as f32 / 1000.0;
            vt.record(t, Point::new(vx * secs, 0.0));
        }
        let v = vt.velocity().expect("velocity");
        assert!((v.x - vx).abs() < 1.0, "stale sample leaked: vx {}", v.x);
    }

    #[test]
    fn property_random_linear_motion() {
        let mut state = 0xBEEF_1234_5678_9ABC_u64;
        for _ in 0..200 {
            let vx = rand_in(&mut state, -500.0, 500.0);
            let vy = rand_in(&mut state, -500.0, 500.0);
            let x0 = rand_in(&mut state, -50.0, 50.0);
            let y0 = rand_in(&mut state, -50.0, 50.0);
            let mut vt = VelocityTracker::new();
            for i in 0..5u64 {
                let t = i * 16;
                let secs = t as f32 / 1000.0;
                vt.record(t, Point::new(x0 + vx * secs, y0 + vy * secs));
            }
            let v = vt.velocity().expect("velocity");
            assert!((v.x - vx).abs() < 1.0, "vx {} vs {vx}", v.x);
            assert!((v.y - vy).abs() < 1.0, "vy {} vs {vy}", v.y);
        }
    }

    #[test]
    fn duplicate_timestamps_fall_back_to_linear() {
        // Three samples but only two distinct times: the quadratic system is
        // singular, so the tracker falls back to a linear fit instead of None.
        let mut vt = VelocityTracker::new();
        vt.record(0, Point::new(0.0, 0.0));
        vt.record(0, Point::new(0.0, 0.0));
        vt.record(10, Point::new(1.0, 0.0));
        let v = vt.velocity().expect("linear fallback");
        // 1 px over 10 ms = 100 px/s.
        assert!((v.x - 100.0).abs() < 1.0, "vx {}", v.x);
    }
}
