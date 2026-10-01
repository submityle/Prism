//! Photometric candela grid: storage + bilinear sampling for IES profiles.
//!
//! An IES / LM-63 luminaire measurement is a two-dimensional table of luminous
//! intensity (candela) sampled on a *polar goniometer* grid: a set of vertical
//! angles `theta` (0° at the luminaire's aim/nadir, increasing toward the back
//! hemisphere, up to 180°) crossed with a set of horizontal angles `phi`
//! (azimuth around the aim axis, 0..360°).  Both axes may be **non-uniform**
//! (measurement labs frequently cluster samples where the beam changes fastest),
//! so this module stores the raw angle axes alongside the candela values and
//! interpolates with an explicit per-axis binary search rather than assuming a
//! fixed step.
//!
//! The core query is [`PhotometricGrid::sample`], a defensive bilinear
//! interpolation:
//! * the vertical (`theta`) axis is **clamped** to its stored range — there is
//!   no light defined outside the measured polar sweep;
//! * the horizontal (`phi`) axis is treated as **periodic** with a 360° period,
//!   so queries wrap smoothly across the seam between the last and first stored
//!   azimuths (symmetry folding into a sub-range is a separate concern handled
//!   by [`super::symmetry`]);
//! * empty tables return `0`, and single-sample axes degrade to a constant
//!   along that axis instead of dividing by a zero span.
//!
//! # Conventions
//! * Angles are in **degrees** to match the LM-63 file format; conversion to
//!   radians happens only where trigonometry is required ([`super::normalize`]).
//! * Candela values are stored **row-major by horizontal angle**: the value for
//!   horizontal index `h` and vertical index `v` lives at `h * n_vertical + v`.
//! * Deterministic pure functions, no RNG / I/O / GPU / unsafe / allocation on
//!   the hot path (only construction allocates the backing [`Vec`]s).
//! * Every output is finite and non-negative; degenerate inputs fall back to
//!   `0` rather than emitting `NaN`/`inf`.
//!
//! # References
//! * IESNA LM-63, *Standard File Format for Electronic Transfer of
//!   Photometric Data*.
//! * Ashdown 1993, *Near-Field Photometry: A New Approach*.

use alloc::vec::Vec;

/// Smallest axis span treated as resolvable; narrower spans collapse to a
/// constant to avoid dividing by (near-)zero.
const MIN_SPAN: f32 = 1.0e-9;

/// Azimuthal period of the horizontal axis, in degrees.
const PHI_PERIOD: f32 = 360.0;

/// A photometric candela table on a (possibly non-uniform) polar grid.
///
/// The two angle axes are stored ascending; the candela buffer is row-major by
/// horizontal angle (see the module header).  Construct via
/// [`PhotometricGrid::new`], which validates the dimensions, or
/// [`PhotometricGrid::uniform_single`] for a trivial isotropic table.
#[derive(Clone, Debug, PartialEq)]
pub struct PhotometricGrid {
    /// Vertical angles `theta` in degrees, ascending (nadir 0° .. 180°).
    vertical: Vec<f32>,
    /// Horizontal angles `phi` in degrees, ascending (azimuth 0° .. 360°).
    horizontal: Vec<f32>,
    /// Candela values, `horizontal.len() * vertical.len()` entries, row-major by
    /// horizontal angle (`candela[h * vertical.len() + v]`).
    candela: Vec<f32>,
}

/// Linear interpolation `a + (b - a) * t` with a finite guard.
#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    let value = a + (b - a) * t;
    if value.is_finite() { value } else { 0.0 }
}

/// Locates a clamped interpolation bracket `(i0, i1, t)` for `x` in `axis`.
///
/// `axis` must be ascending.  Values at or below the first sample clamp to the
/// first entry (`t == 0`); values at or above the last sample clamp to the last
/// entry.  Returns `(i0, i0, 0)` for single-sample axes and non-finite inputs.
#[inline]
fn locate(axis: &[f32], x: f32) -> (usize, usize, f32) {
    let n = axis.len();
    if n <= 1 || !x.is_finite() {
        return (0, 0, 0.0);
    }
    if x <= axis[0] {
        return (0, 0, 0.0);
    }
    let last = n - 1;
    if x >= axis[last] {
        return (last, last, 0.0);
    }
    // First index whose angle exceeds `x`; guaranteed in `1..n` by the bounds
    // checks above, so `hi - 1` is a valid lower bracket index.
    let hi = axis.partition_point(|&a| a <= x);
    let i0 = hi - 1;
    let a0 = axis[i0];
    let a1 = axis[hi];
    let span = a1 - a0;
    let t = if span > MIN_SPAN { (x - a0) / span } else { 0.0 };
    (i0, hi, t.clamp(0.0, 1.0))
}

/// Locates a *periodic* interpolation bracket for a horizontal angle `phi`.
///
/// `phi` is reduced modulo 360° into the stored axis frame.  When it lands
/// inside the stored range it behaves like [`locate`]; when it lands in the
/// wrap gap between the last stored azimuth and the first (plus 360°), it blends
/// the last and first rows across the seam, returning `(last, 0, t)`.
#[inline]
fn locate_phi(axis: &[f32], phi: f32) -> (usize, usize, f32) {
    let n = axis.len();
    if n <= 1 || !phi.is_finite() {
        return (0, 0, 0.0);
    }
    let first = axis[0];
    let last_angle = axis[n - 1];
    // Reduce into `[first, first + 360)` so the wrap seam is explicit.
    let reduced = first + (phi - first).rem_euclid(PHI_PERIOD);
    if reduced <= last_angle {
        return locate(axis, reduced);
    }
    // In the gap between the last stored angle and `first + 360`.
    let gap = (first + PHI_PERIOD) - last_angle;
    let t = if gap > MIN_SPAN {
        (reduced - last_angle) / gap
    } else {
        0.0
    };
    (n - 1, 0, t.clamp(0.0, 1.0))
}

impl PhotometricGrid {
    /// Builds a grid from ascending angle axes and a row-major candela buffer.
    ///
    /// Returns `None` when either axis is empty or when
    /// `candela.len() != vertical.len() * horizontal.len()`.  Angle axes are
    /// assumed ascending; callers that cannot guarantee that should sort before
    /// construction (sampling clamps/wraps but does not re-sort).
    #[inline]
    pub fn new(vertical: Vec<f32>, horizontal: Vec<f32>, candela: Vec<f32>) -> Option<Self> {
        let nv = vertical.len();
        let nh = horizontal.len();
        if nv == 0 || nh == 0 || candela.len() != nv * nh {
            return None;
        }
        Some(Self {
            vertical,
            horizontal,
            candela,
        })
    }

    /// Builds a trivial isotropic grid: a single candela value shared by every
    /// direction (one vertical sample at 0°, one horizontal sample at 0°).
    #[inline]
    pub fn uniform_single(candela: f32) -> Self {
        let value = if candela.is_finite() && candela >= 0.0 {
            candela
        } else {
            0.0
        };
        let mut vertical = Vec::with_capacity(1);
        vertical.push(0.0);
        let mut horizontal = Vec::with_capacity(1);
        horizontal.push(0.0);
        let mut c = Vec::with_capacity(1);
        c.push(value);
        Self {
            vertical,
            horizontal,
            candela: c,
        }
    }

    /// Number of vertical (`theta`) samples.
    #[inline]
    pub fn vertical_count(&self) -> usize {
        self.vertical.len()
    }

    /// Number of horizontal (`phi`) samples.
    #[inline]
    pub fn horizontal_count(&self) -> usize {
        self.horizontal.len()
    }

    /// Read-only view of the ascending vertical angle axis (degrees).
    #[inline]
    pub fn vertical_angles(&self) -> &[f32] {
        &self.vertical
    }

    /// Read-only view of the ascending horizontal angle axis (degrees).
    #[inline]
    pub fn horizontal_angles(&self) -> &[f32] {
        &self.horizontal
    }

    /// Read-only view of the row-major candela buffer.
    #[inline]
    pub fn candela(&self) -> &[f32] {
        &self.candela
    }

    /// Returns `true` when the table carries no usable candela samples.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.candela.is_empty() || self.vertical.is_empty() || self.horizontal.is_empty()
    }

    /// Fetches the raw stored candela at integer grid coordinates, clamped to a
    /// finite, non-negative value.  Indices are clamped into range.
    #[inline]
    fn raw(&self, h: usize, v: usize) -> f32 {
        let nv = self.vertical.len();
        let nh = self.horizontal.len();
        if nv == 0 || nh == 0 {
            return 0.0;
        }
        let hc = h.min(nh - 1);
        let vc = v.min(nv - 1);
        let value = self.candela[hc * nv + vc];
        if value.is_finite() && value >= 0.0 {
            value
        } else {
            0.0
        }
    }

    /// Peak candela across the whole table (`0` for an empty grid).
    #[inline]
    pub fn max_candela(&self) -> f32 {
        let mut m = 0.0_f32;
        for &c in &self.candela {
            if c.is_finite() && c > m {
                m = c;
            }
        }
        m
    }

    /// Bilinearly samples the candela table at `(theta, phi)` in **degrees**.
    ///
    /// The vertical axis clamps to its measured range; the horizontal axis wraps
    /// with a 360° period.  Single-sample axes degrade to a constant along that
    /// axis.  The result is always finite and non-negative; an empty grid yields
    /// `0`.
    #[inline]
    pub fn sample(&self, theta: f32, phi: f32) -> f32 {
        if self.is_empty() {
            return 0.0;
        }
        let (v0, v1, tv) = locate(&self.vertical, theta);
        let (h0, h1, th) = locate_phi(&self.horizontal, phi);

        let c00 = self.raw(h0, v0);
        let c01 = self.raw(h0, v1);
        let c10 = self.raw(h1, v0);
        let c11 = self.raw(h1, v1);

        // Interpolate along the vertical axis first, then the horizontal axis.
        let cv0 = lerp(c00, c01, tv);
        let cv1 = lerp(c10, c11, tv);
        let value = lerp(cv0, cv1, th);
        if value.is_finite() && value >= 0.0 {
            value
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn new_rejects_dimension_mismatch() {
        assert!(PhotometricGrid::new(alloc::vec![0.0, 90.0], alloc::vec![0.0], alloc::vec![1.0]).is_none());
        assert!(PhotometricGrid::new(Vec::new(), alloc::vec![0.0], Vec::new()).is_none());
    }

    #[test]
    fn new_accepts_matching_dimensions() {
        let g = PhotometricGrid::new(
            alloc::vec![0.0, 90.0, 180.0],
            alloc::vec![0.0, 180.0],
            alloc::vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        .unwrap();
        assert_eq!(g.vertical_count(), 3);
        assert_eq!(g.horizontal_count(), 2);
        assert_eq!(g.candela().len(), 6);
    }

    #[test]
    fn uniform_single_is_constant_everywhere() {
        let g = PhotometricGrid::uniform_single(42.0);
        assert!(approx(g.sample(0.0, 0.0), 42.0, 1e-6));
        assert!(approx(g.sample(90.0, 123.0), 42.0, 1e-6));
        assert!(approx(g.sample(180.0, 359.0), 42.0, 1e-6));
    }

    #[test]
    fn uniform_single_rejects_bad_value() {
        let g = PhotometricGrid::uniform_single(f32::NAN);
        assert_eq!(g.max_candela(), 0.0);
        let g2 = PhotometricGrid::uniform_single(-5.0);
        assert_eq!(g2.sample(0.0, 0.0), 0.0);
    }

    #[test]
    fn empty_grid_samples_zero() {
        // Constructed directly to exercise the empty guard.
        let g = PhotometricGrid {
            vertical: Vec::new(),
            horizontal: Vec::new(),
            candela: Vec::new(),
        };
        assert!(g.is_empty());
        assert_eq!(g.sample(10.0, 10.0), 0.0);
        assert_eq!(g.max_candela(), 0.0);
    }

    #[test]
    fn vertical_interpolation_is_linear() {
        // Single azimuth, three polar samples 0/100/0.
        let g = PhotometricGrid::new(
            alloc::vec![0.0, 90.0, 180.0],
            alloc::vec![0.0],
            alloc::vec![0.0, 100.0, 0.0],
        )
        .unwrap();
        assert!(approx(g.sample(45.0, 0.0), 50.0, 1e-4));
        assert!(approx(g.sample(135.0, 0.0), 50.0, 1e-4));
        assert!(approx(g.sample(90.0, 0.0), 100.0, 1e-4));
    }

    #[test]
    fn vertical_clamps_outside_range() {
        let g = PhotometricGrid::new(
            alloc::vec![10.0, 20.0],
            alloc::vec![0.0],
            alloc::vec![3.0, 7.0],
        )
        .unwrap();
        assert!(approx(g.sample(-5.0, 0.0), 3.0, 1e-6));
        assert!(approx(g.sample(999.0, 0.0), 7.0, 1e-6));
    }

    #[test]
    fn non_uniform_axis_interpolates_correctly() {
        // Vertical axis is non-uniform: 0, 10, 90.
        let g = PhotometricGrid::new(
            alloc::vec![0.0, 10.0, 90.0],
            alloc::vec![0.0],
            alloc::vec![0.0, 10.0, 90.0],
        )
        .unwrap();
        // Between 10 and 90 (span 80): at theta=50, t=(50-10)/80=0.5 -> 50.
        assert!(approx(g.sample(50.0, 0.0), 50.0, 1e-4));
        // Between 0 and 10: at theta=5, t=0.5 -> 5.
        assert!(approx(g.sample(5.0, 0.0), 5.0, 1e-4));
    }

    #[test]
    fn horizontal_wraps_periodically() {
        // Two azimuths 0 and 180, both with a single polar sample.
        let g = PhotometricGrid::new(
            alloc::vec![0.0],
            alloc::vec![0.0, 180.0],
            alloc::vec![10.0, 20.0],
        )
        .unwrap();
        // phi=90 is halfway from 0 to 180 -> 15.
        assert!(approx(g.sample(0.0, 90.0), 15.0, 1e-4));
        // phi=270 is in the wrap gap halfway from 180 back to 360(=0) -> 15.
        assert!(approx(g.sample(0.0, 270.0), 15.0, 1e-4));
        // phi=360 wraps to 0.
        assert!(approx(g.sample(0.0, 360.0), 10.0, 1e-4));
    }

    #[test]
    fn bilinear_blends_all_four_corners() {
        let g = PhotometricGrid::new(
            alloc::vec![0.0, 90.0],
            alloc::vec![0.0, 90.0],
            // row-major by horizontal: h0 -> [0,10], h1 -> [20,30]
            alloc::vec![0.0, 10.0, 20.0, 30.0],
        )
        .unwrap();
        // Center of the cell: average of all four = 15.
        assert!(approx(g.sample(45.0, 45.0), 15.0, 1e-4));
    }

    #[test]
    fn sample_is_always_finite_nonnegative() {
        let g = PhotometricGrid::new(
            alloc::vec![0.0, 90.0],
            alloc::vec![0.0, 90.0],
            alloc::vec![f32::NAN, 10.0, 20.0, f32::INFINITY],
        )
        .unwrap();
        let v = g.sample(30.0, 30.0);
        assert!(v.is_finite());
        assert!(v >= 0.0);
    }
}
