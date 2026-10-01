//! 3D-LUT golden reference with trilinear sampling and 1D shaper pre-shaping.
//!
//! A 3D colour LUT (lookup table) bakes an arbitrary RGB → RGB transfer into a
//! cube of `N³` samples. At runtime the input colour indexes the cube and a
//! **trilinear** interpolation of the eight surrounding lattice points produces
//! the output. This is the universal "creative LUT" / `.cube` mechanism that
//! lets a colourist's full grade — however complex — be applied as a single
//! cheap texture fetch.
//!
//! Because a 3D LUT distributes its samples uniformly in the input domain, a
//! transfer with a lot of action in the shadows (log or HDR footage) wastes most
//! of its lattice on the highlights. A **1D shaper LUT** fixes that: it reshapes
//! each channel through a monotone curve *before* the 3D lookup, concentrating
//! lattice resolution where the grade needs it. [`ShaperLut1d`] provides that
//! pre-shaping stage and [`Lut3d::sample_with_shaper`] chains the two.
//!
//! # Conventions
//! * `no_std`: storage uses [`alloc::vec::Vec`]; the crate already declares
//!   `extern crate alloc`.
//! * Inputs are clamped to `[0, 1]` before indexing, so the cube is sampled
//!   within its domain and sub-/super-range colours clamp to the boundary
//!   lattice (standard LUT clamp-to-edge behaviour).
//! * `floor` is taken via [`bevy_math::ops::floor`]; colours are
//!   [`bevy_math::Vec3`]. All maths is deterministic `f32`.
//! * Degenerate sizes (`< 2`) are rejected at construction; sampling an empty or
//!   malformed table falls back to the clamped input, never `NaN`.

use alloc::vec::Vec;

use bevy_math::Vec3;
use bevy_math::ops;

/// Minimum lattice edge length. A LUT needs at least the two endpoints `0` and
/// `1` per axis to interpolate between.
pub const MIN_LUT_SIZE: usize = 2;

/// Maximum lattice edge length, bounding `N³` storage to a sane size (`64³ =
/// 262144` entries, the common hardware ceiling).
pub const MAX_LUT_SIZE: usize = 64;

/// Clamp a single coordinate into the `[0, 1]` sampling domain, mapping any
/// non-finite input to `0`.
#[must_use]
fn clamp01(x: f32) -> f32 {
    if x.is_finite() { x.clamp(0.0, 1.0) } else { 0.0 }
}

/// Clamp an RGB colour component-wise into `[0, 1]`.
#[must_use]
fn clamp01_vec(c: Vec3) -> Vec3 {
    Vec3::new(clamp01(c.x), clamp01(c.y), clamp01(c.z))
}

// --- 3D LUT ---------------------------------------------------------------

/// A cubic 3D colour LUT of edge length `size`, storing `size³` RGB samples in
/// row-major `(r, g, b)` order (`r` fastest).
#[derive(Clone, Debug)]
pub struct Lut3d {
    size: usize,
    data: Vec<Vec3>,
}

impl Lut3d {
    /// Clamp a requested edge length into the supported `[MIN_LUT_SIZE,
    /// MAX_LUT_SIZE]` range.
    #[must_use]
    fn clamp_size(n: usize) -> usize {
        n.clamp(MIN_LUT_SIZE, MAX_LUT_SIZE)
    }

    /// Flatten an `(ir, ig, ib)` lattice coordinate to a linear index.
    #[must_use]
    fn index(&self, ir: usize, ig: usize, ib: usize) -> usize {
        (ib * self.size + ig) * self.size + ir
    }

    /// Edge length `N` of the cube.
    #[must_use]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Number of stored samples, `N³`.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the table holds no samples (never true for a constructed LUT).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Fetch the stored sample at a lattice coordinate, clamping the coordinate
    /// to the valid range (clamp-to-edge).
    #[must_use]
    pub fn texel(&self, ir: usize, ig: usize, ib: usize) -> Vec3 {
        let last = self.size - 1;
        let idx = self.index(ir.min(last), ig.min(last), ib.min(last));
        self.data[idx]
    }

    /// Build an identity LUT of edge length `n`: the sample at lattice
    /// `(i, j, k)` is `(i, j, k) / (N - 1)`, so sampling reproduces the input.
    #[must_use]
    pub fn identity(n: usize) -> Self {
        Self::from_fn(n, |c| c)
    }

    /// Bake an arbitrary RGB → RGB transfer `f` into a LUT of edge length `n`.
    ///
    /// `f` is evaluated at every lattice point (its grid coordinate normalised
    /// to `[0, 1]`) and the result is clamped component-wise to `[0, 1]`.
    #[must_use]
    pub fn from_fn(n: usize, f: impl Fn(Vec3) -> Vec3) -> Self {
        let size = Self::clamp_size(n);
        let denom = (size - 1) as f32;
        let mut data = Vec::with_capacity(size * size * size);
        for ib in 0..size {
            let b = ib as f32 / denom;
            for ig in 0..size {
                let g = ig as f32 / denom;
                for ir in 0..size {
                    let r = ir as f32 / denom;
                    data.push(clamp01_vec(f(Vec3::new(r, g, b))));
                }
            }
        }
        Self { size, data }
    }

    /// Sample the LUT at an RGB input via trilinear interpolation.
    ///
    /// The input is clamped to `[0, 1]`, scaled to lattice coordinates, split
    /// into integer cell and fractional position, and the eight corner samples
    /// are blended. On a degenerate table the clamped input is returned.
    #[must_use]
    pub fn sample(&self, rgb: Vec3) -> Vec3 {
        let c = clamp01_vec(rgb);
        if self.size < MIN_LUT_SIZE || self.data.len() != self.size * self.size * self.size {
            return c;
        }
        let last = self.size - 1;
        let scale = last as f32;

        let fr = c.x * scale;
        let fg = c.y * scale;
        let fb = c.z * scale;

        let (ir0, dr) = split(fr, last);
        let (ig0, dg) = split(fg, last);
        let (ib0, db) = split(fb, last);
        let ir1 = (ir0 + 1).min(last);
        let ig1 = (ig0 + 1).min(last);
        let ib1 = (ib0 + 1).min(last);

        // Eight corners of the enclosing cell.
        let c000 = self.texel(ir0, ig0, ib0);
        let c100 = self.texel(ir1, ig0, ib0);
        let c010 = self.texel(ir0, ig1, ib0);
        let c110 = self.texel(ir1, ig1, ib0);
        let c001 = self.texel(ir0, ig0, ib1);
        let c101 = self.texel(ir1, ig0, ib1);
        let c011 = self.texel(ir0, ig1, ib1);
        let c111 = self.texel(ir1, ig1, ib1);

        // Interpolate along r, then g, then b.
        let c00 = lerp(c000, c100, dr);
        let c10 = lerp(c010, c110, dr);
        let c01 = lerp(c001, c101, dr);
        let c11 = lerp(c011, c111, dr);

        let c0 = lerp(c00, c10, dg);
        let c1 = lerp(c01, c11, dg);

        lerp(c0, c1, db)
    }

    /// Sample the LUT after reshaping the input through a per-channel 1D shaper.
    ///
    /// The shaper is applied first (concentrating lattice resolution), then the
    /// shaped colour drives the trilinear 3D lookup.
    #[must_use]
    pub fn sample_with_shaper(&self, rgb: Vec3, shaper: &ShaperLut1d) -> Vec3 {
        self.sample(shaper.apply(rgb))
    }
}

/// Split a lattice coordinate into its integer cell index (clamped so the next
/// cell stays in range) and the fractional position within that cell.
#[must_use]
fn split(coord: f32, last: usize) -> (usize, f32) {
    let base = ops::floor(coord);
    let i = base as i32;
    if i < 0 {
        (0, 0.0)
    } else if i as usize >= last {
        (last, 0.0)
    } else {
        (i as usize, (coord - base).clamp(0.0, 1.0))
    }
}

/// Component-wise linear interpolation `a + (b - a) * t`.
#[must_use]
fn lerp(a: Vec3, b: Vec3, t: f32) -> Vec3 {
    a + (b - a) * t
}

// --- 1D shaper LUT --------------------------------------------------------

/// A per-channel 1D shaper LUT of length `size`, applied before a 3D lookup to
/// redistribute lattice resolution. Each channel shares the same monotone curve
/// sampled with linear interpolation over `[0, 1]`.
#[derive(Clone, Debug)]
pub struct ShaperLut1d {
    size: usize,
    data: Vec<f32>,
}

impl ShaperLut1d {
    /// Clamp a requested length into the supported range.
    #[must_use]
    fn clamp_size(n: usize) -> usize {
        n.clamp(MIN_LUT_SIZE, MAX_LUT_SIZE)
    }

    /// Length of the 1D table.
    #[must_use]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Whether the table holds no samples (never true for a constructed shaper).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Build the identity shaper of length `n`: sample `i` is `i / (N - 1)`.
    #[must_use]
    pub fn identity(n: usize) -> Self {
        Self::from_fn(n, |x| x)
    }

    /// Bake a scalar curve `f` into a shaper of length `n`.
    ///
    /// `f` is evaluated at each normalised sample position and clamped to
    /// `[0, 1]`.
    #[must_use]
    pub fn from_fn(n: usize, f: impl Fn(f32) -> f32) -> Self {
        let size = Self::clamp_size(n);
        let denom = (size - 1) as f32;
        let mut data = Vec::with_capacity(size);
        for i in 0..size {
            let x = i as f32 / denom;
            data.push(clamp01(f(x)));
        }
        Self { size, data }
    }

    /// Shape a single channel value via linear interpolation of the curve.
    ///
    /// The input is clamped to `[0, 1]` before lookup; a degenerate table
    /// returns the clamped input.
    #[must_use]
    pub fn shape(&self, x: f32) -> f32 {
        let v = clamp01(x);
        if self.size < MIN_LUT_SIZE || self.data.len() != self.size {
            return v;
        }
        let last = self.size - 1;
        let f = v * last as f32;
        let (i0, d) = split(f, last);
        let i1 = (i0 + 1).min(last);
        let a = self.data[i0];
        a + (self.data[i1] - a) * d
    }

    /// Apply the shaper to each channel of an RGB colour.
    #[must_use]
    pub fn apply(&self, rgb: Vec3) -> Vec3 {
        Vec3::new(self.shape(rgb.x), self.shape(rgb.y), self.shape(rgb.z))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1.0e-5;

    fn approx(a: f32, b: f32) {
        assert!((a - b).abs() < 1.0e-4, "{a} != {b}");
    }

    fn approx3(a: Vec3, b: Vec3) {
        approx(a.x, b.x);
        approx(a.y, b.y);
        approx(a.z, b.z);
    }

    #[test]
    fn identity_dimensions() {
        let lut = Lut3d::identity(17);
        assert_eq!(lut.size(), 17);
        assert_eq!(lut.len(), 17 * 17 * 17);
        assert!(!lut.is_empty());
    }

    #[test]
    fn identity_maps_grid_points_exactly() {
        let lut = Lut3d::identity(9);
        let n = lut.size();
        let denom = (n - 1) as f32;
        for ib in 0..n {
            for ig in 0..n {
                for ir in 0..n {
                    let c = Vec3::new(
                        ir as f32 / denom,
                        ig as f32 / denom,
                        ib as f32 / denom,
                    );
                    approx3(lut.sample(c), c);
                }
            }
        }
    }

    #[test]
    fn identity_maps_arbitrary_inputs_exactly() {
        // Trilinear interpolation of a linear (identity) field is exact.
        let lut = Lut3d::identity(5);
        for &c in &[
            Vec3::new(0.123, 0.456, 0.789),
            Vec3::new(0.0, 1.0, 0.5),
            Vec3::new(0.333, 0.667, 0.1),
            Vec3::new(0.95, 0.05, 0.5),
        ] {
            approx3(lut.sample(c), c);
        }
    }

    #[test]
    fn constant_lut_returns_constant() {
        let k = Vec3::new(0.25, 0.5, 0.75);
        let lut = Lut3d::from_fn(8, |_| k);
        for &c in &[
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.3, 0.7, 0.2),
            Vec3::new(1.0, 1.0, 1.0),
        ] {
            approx3(lut.sample(c), k);
        }
    }

    #[test]
    fn sample_clamps_out_of_range_inputs() {
        let lut = Lut3d::identity(4);
        // Below 0 and above 1 clamp to the boundary lattice.
        approx3(lut.sample(Vec3::new(-1.0, -2.0, -0.5)), Vec3::ZERO);
        approx3(lut.sample(Vec3::new(2.0, 3.0, 10.0)), Vec3::ONE);
    }

    #[test]
    fn trilinear_midpoint_is_mean_of_corners() {
        // A LUT whose value equals r reproduces r; test the midpoint between
        // two grid points equals the average.
        let lut = Lut3d::from_fn(2, |c| Vec3::splat(c.x));
        let mid = lut.sample(Vec3::new(0.5, 0.0, 0.0));
        approx3(mid, Vec3::splat(0.5));
    }

    #[test]
    fn from_fn_bakes_and_clamps() {
        // A function returning out-of-range values is clamped at bake time.
        let lut = Lut3d::from_fn(4, |c| c * 4.0 - Vec3::ONE);
        // Sampling white: baked value clamped to 1.
        approx3(lut.sample(Vec3::ONE), Vec3::ONE);
        // Sampling black: baked value clamped to 0.
        approx3(lut.sample(Vec3::ZERO), Vec3::ZERO);
    }

    #[test]
    fn degenerate_size_is_clamped_up() {
        let lut = Lut3d::identity(0);
        assert_eq!(lut.size(), MIN_LUT_SIZE);
        let big = Lut3d::identity(10_000);
        assert_eq!(big.size(), MAX_LUT_SIZE);
    }

    #[test]
    fn sample_is_never_nan() {
        let lut = Lut3d::identity(8);
        let out = lut.sample(Vec3::new(f32::NAN, f32::INFINITY, -f32::INFINITY));
        assert!(out.is_finite());
    }

    #[test]
    fn gamma_lut_matches_direct_evaluation_at_grid() {
        let gamma = |c: Vec3| Vec3::new(
            ops::powf(c.x, 2.2),
            ops::powf(c.y, 2.2),
            ops::powf(c.z, 2.2),
        );
        let lut = Lut3d::from_fn(33, gamma);
        for &c in &[Vec3::splat(0.25), Vec3::splat(0.5), Vec3::splat(0.75)] {
            // 33 lattice includes 0.25/0.5/0.75 only approximately, so compare
            // the sampled grade to the baked grade within interpolation error.
            let direct = gamma(c);
            let sampled = lut.sample(c);
            assert!((sampled - direct).length() < 1.0e-3, "{sampled:?} vs {direct:?}");
        }
    }

    #[test]
    fn shaper_identity_round_trips() {
        let shaper = ShaperLut1d::identity(16);
        for &x in &[0.0_f32, 0.123, 0.5, 0.9, 1.0] {
            approx(shaper.shape(x), x);
        }
        approx3(shaper.apply(Vec3::new(0.2, 0.5, 0.8)), Vec3::new(0.2, 0.5, 0.8));
    }

    #[test]
    fn shaper_clamps_and_is_monotonic() {
        let shaper = ShaperLut1d::from_fn(32, |x| ops::powf(x, 2.2));
        assert!(shaper.shape(-1.0).abs() < EPS);
        approx(shaper.shape(2.0), 1.0);
        let mut prev = -1.0_f32;
        let mut x = 0.0_f32;
        while x <= 1.0 {
            let v = shaper.shape(x);
            assert!(v >= prev - EPS, "shaper not monotonic at {x}");
            prev = v;
            x += 0.05;
        }
    }

    #[test]
    fn sample_with_identity_shaper_matches_plain_sample() {
        let lut = Lut3d::from_fn(8, |c| Vec3::new(c.z, c.x, c.y));
        let shaper = ShaperLut1d::identity(8);
        let c = Vec3::new(0.3, 0.6, 0.9);
        approx3(lut.sample_with_shaper(c, &shaper), lut.sample(c));
    }

    #[test]
    fn shaper_degenerate_size_clamped() {
        let s = ShaperLut1d::identity(1);
        assert_eq!(s.size(), MIN_LUT_SIZE);
        assert!(!s.is_empty());
    }
}
