//! Compile-time (`const`) math: zero-cost constant matrices and baked tables.
//!
//! Everything here is evaluated by the compiler and lives in the read-only data
//! segment, so there is no runtime construction, no heap allocation, and no
//! one-time initialisation. Two families are provided:
//!
//! 1. **Coordinate-system conversion matrices.** Engines, DCC tools, and asset
//!    formats disagree on the up axis and on handedness (glTF is Y-up
//!    right-handed, Blender/CAD are typically Z-up right-handed, some engines
//!    are left-handed). The `const` [`Mat4`] constants here express those
//!    interop transforms exactly — every entry is `0` or `±1`, so the matrices
//!    are bit-exact and fold at compile time wherever they are used.
//! 2. **Baked lookup tables.** [`LookupTable`] stores a fixed-size sample array
//!    plus its domain as a `const`-constructible value. Easing curves, sampled
//!    response curves, and precomputed numeric tables bake straight into the
//!    binary and are read with a branch-light clamped linear interpolation.
//!
//! No transcendental functions are used, so the constants are identical on
//! every target.

use crate::mat::Mat4;
use crate::vec::{Vec3, Vec4};

/// Convert a right-handed Y-up point to a right-handed Z-up point.
///
/// This is a `+90°` rotation about the X axis, mapping `(x, y, z)` to
/// `(x, -z, y)` — the standard glTF (Y-up) → Blender/CAD (Z-up) basis change.
/// Being a pure rotation it preserves handedness (determinant `+1`).
pub const Y_UP_TO_Z_UP: Mat4 = Mat4::from_cols(
    Vec4::new(1.0, 0.0, 0.0, 0.0),
    Vec4::new(0.0, 0.0, 1.0, 0.0),
    Vec4::new(0.0, -1.0, 0.0, 0.0),
    Vec4::new(0.0, 0.0, 0.0, 1.0),
);

/// Convert a right-handed Z-up point back to a right-handed Y-up point.
///
/// The exact inverse (and transpose) of [`Y_UP_TO_Z_UP`]: a `-90°` rotation
/// about the X axis mapping `(x, y, z)` to `(x, z, -y)`.
pub const Z_UP_TO_Y_UP: Mat4 = Mat4::from_cols(
    Vec4::new(1.0, 0.0, 0.0, 0.0),
    Vec4::new(0.0, 0.0, -1.0, 0.0),
    Vec4::new(0.0, 1.0, 0.0, 0.0),
    Vec4::new(0.0, 0.0, 0.0, 1.0),
);

/// Flip handedness by negating the Z axis (`diag(1, 1, -1, 1)`).
///
/// Converts between a right-handed and a left-handed frame that share the same
/// up axis (the common OpenGL ↔ Direct3D / Unity depth convention). The
/// determinant is `-1`, so applying it an even number of times is the identity.
pub const FLIP_HANDEDNESS_Z: Mat4 = Mat4::from_cols(
    Vec4::new(1.0, 0.0, 0.0, 0.0),
    Vec4::new(0.0, 1.0, 0.0, 0.0),
    Vec4::new(0.0, 0.0, -1.0, 0.0),
    Vec4::new(0.0, 0.0, 0.0, 1.0),
);

/// A compile-time baked, uniformly sampled lookup table over `[min, max]`.
///
/// The `N` samples are evenly spaced across the inclusive domain, with
/// `samples[0]` at `min` and `samples[N - 1]` at `max`. Construction is
/// `const`, so a table literal bakes into read-only data with no runtime setup.
/// [`LookupTable::sample`] reads it back with clamped linear interpolation
/// (inputs outside the domain saturate to the end samples), which is the usual
/// cost model for easing curves and precomputed response tables.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct LookupTable<const N: usize> {
    samples: [f32; N],
    min: f32,
    max: f32,
}

impl<const N: usize> LookupTable<N> {
    /// Build a table of `N` evenly spaced samples spanning `[min, max]`.
    ///
    /// # Panics
    /// Panics (at `const` evaluation when used in a `const` context) if `N < 2`
    /// or if `max <= min`; a table needs at least two samples and a
    /// non-degenerate domain to interpolate.
    #[must_use]
    pub const fn new(samples: [f32; N], min: f32, max: f32) -> Self {
        assert!(N >= 2, "LookupTable needs at least two samples");
        assert!(max > min, "LookupTable domain must be non-empty");
        Self { samples, min, max }
    }

    /// Number of stored samples.
    #[must_use]
    pub const fn len(&self) -> usize {
        N
    }

    /// Always `false`: a [`LookupTable`] stores at least two samples.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        false
    }

    /// Inclusive domain `[min, max]` the samples span.
    #[must_use]
    pub const fn domain(&self) -> (f32, f32) {
        (self.min, self.max)
    }

    /// Sample the table at `x` with clamped linear interpolation.
    ///
    /// `x` is clamped to the domain, mapped to a fractional sample index, and
    /// linearly blended between the two bracketing samples. No transcendental
    /// functions are involved, so results are deterministic across targets.
    #[must_use]
    pub fn sample(&self, x: f32) -> f32 {
        let t = ((x - self.min) / (self.max - self.min)).clamp(0.0, 1.0);
        let pos = t * (N - 1) as f32;
        // `floor` on a value already clamped to `[0, N-1]` keeps `lo` in range.
        let lo = pos as usize;
        if lo >= N - 1 {
            return self.samples[N - 1];
        }
        let frac = pos - lo as f32;
        let a = self.samples[lo];
        let b = self.samples[lo + 1];
        a + (b - a) * frac
    }
}

/// Image of a point under a `const` conversion matrix, for `const`-site use.
///
/// A thin `const fn` wrapper over the affine transform so callers can fold a
/// converted point at compile time without pulling in the non-`const`
/// [`Mat4::transform_point3`].
#[must_use]
pub const fn convert_point(m: &Mat4, p: Vec3) -> Vec3 {
    // Column-major: result = x*col0 + y*col1 + z*col2 + col3 (w = 1).
    Vec3::new(
        m.x_axis.x * p.x + m.y_axis.x * p.y + m.z_axis.x * p.z + m.w_axis.x,
        m.x_axis.y * p.x + m.y_axis.y * p.y + m.z_axis.y * p.z + m.w_axis.y,
        m.x_axis.z * p.x + m.y_axis.z * p.y + m.z_axis.z * p.z + m.w_axis.z,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn y_up_to_z_up_maps_axes() {
        // +Y (up) in a Y-up frame becomes +Z (up) in a Z-up frame.
        let up = Y_UP_TO_Z_UP.transform_point3(Vec3::new(0.0, 1.0, 0.0));
        assert!((up - Vec3::new(0.0, 0.0, 1.0)).length() < 1e-6);
        // A general point: (x, y, z) -> (x, -z, y).
        let p = Y_UP_TO_Z_UP.transform_point3(Vec3::new(2.0, 3.0, 5.0));
        assert!((p - Vec3::new(2.0, -5.0, 3.0)).length() < 1e-6);
    }

    #[test]
    fn up_conversions_round_trip() {
        let m = Z_UP_TO_Y_UP * Y_UP_TO_Z_UP;
        let p = Vec3::new(-1.5, 4.25, 9.0);
        let back = m.transform_point3(p);
        assert!((back - p).length() < 1e-6);
    }

    #[test]
    fn rotations_preserve_handedness_flip_inverts() {
        assert!((Y_UP_TO_Z_UP.determinant() - 1.0).abs() < 1e-6);
        assert!((Z_UP_TO_Y_UP.determinant() - 1.0).abs() < 1e-6);
        assert!((FLIP_HANDEDNESS_Z.determinant() + 1.0).abs() < 1e-6);
        // Flipping twice is the identity.
        let twice = FLIP_HANDEDNESS_Z * FLIP_HANDEDNESS_Z;
        let p = Vec3::new(3.0, -2.0, 7.0);
        assert!((twice.transform_point3(p) - p).length() < 1e-6);
    }

    #[test]
    fn const_convert_point_matches_runtime() {
        const P: Vec3 = convert_point(&Y_UP_TO_Z_UP, Vec3::new(2.0, 3.0, 5.0));
        let runtime = Y_UP_TO_Z_UP.transform_point3(Vec3::new(2.0, 3.0, 5.0));
        assert!((P - runtime).length() < 1e-6);
    }

    // A baked linear ramp over [0, 10]; endpoints and midpoint are exact.
    const RAMP: LookupTable<3> = LookupTable::new([0.0, 5.0, 10.0], 0.0, 10.0);

    #[test]
    fn lookup_table_interpolates_and_clamps() {
        assert_eq!(RAMP.len(), 3);
        assert!(!RAMP.is_empty());
        assert_eq!(RAMP.domain(), (0.0, 10.0));
        // Exact samples.
        assert!((RAMP.sample(0.0) - 0.0).abs() < 1e-6);
        assert!((RAMP.sample(5.0) - 5.0).abs() < 1e-6);
        assert!((RAMP.sample(10.0) - 10.0).abs() < 1e-6);
        // Interpolated midpoints of each segment.
        assert!((RAMP.sample(2.5) - 2.5).abs() < 1e-6);
        assert!((RAMP.sample(7.5) - 7.5).abs() < 1e-6);
        // Out-of-domain inputs saturate to the end samples.
        assert!((RAMP.sample(-3.0) - 0.0).abs() < 1e-6);
        assert!((RAMP.sample(42.0) - 10.0).abs() < 1e-6);
    }

    #[test]
    fn lookup_table_nonlinear_curve() {
        // A quadratic ease sampled at 5 points over [0, 1].
        const EASE: LookupTable<5> = LookupTable::new([0.0, 0.0625, 0.25, 0.5625, 1.0], 0.0, 1.0);
        // Midway between the 0.25 and 0.5625 samples (x = 0.625).
        let mid = EASE.sample(0.625);
        assert!((mid - 0.406_25).abs() < 1e-6, "{mid}");
    }
}
