//! Octahedral direction <-> unit-square mapping.
//!
//! Screen probes store their per-direction radiance in a small octahedral
//! atlas: the unit sphere is projected onto the faces of an octahedron, which
//! is then unfolded into the `[0, 1]^2` unit square.  Compared with a lat-long
//! parameterisation the octahedral map wastes far less area and has no polar
//! singularity, which is why Lumen-style probes adopt it.
//!
//! # Conventions
//! * Directions are right-handed `(x, y, z)` unit vectors; `+z` is the front
//!   hemisphere and `-z` the back hemisphere.  The mapping is symmetric, so the
//!   choice of "up" axis is irrelevant to the round trip.
//! * The returned UV lives in `[0, 1]^2`.  UV `(0.5, 0.5)` corresponds to the
//!   `+z` pole; the four corners collapse onto the `-z` pole.
//! * `oct_to_dir` and `dir_to_oct` are exact inverses of each other for every
//!   unit direction (round-trip error < 1e-5), which the tests assert across
//!   the sphere including the poles and the diagonal fold seams.

use bevy_math::{Vec2, Vec3};

/// Branchless sign that maps zero to `+1`, matching the GLSL `signNotZero`
/// helper used in the reference octahedral papers.  Using `+1` for zero keeps
/// the fold on the seam continuous.
#[inline]
fn sign_not_zero(value: f32) -> f32 {
    if value >= 0.0 {
        1.0
    } else {
        -1.0
    }
}

/// Encodes a (not necessarily normalised) direction into octahedral UV space.
///
/// The direction is first projected onto the octahedron via the L1 norm, then
/// the lower (`z < 0`) hemisphere is folded outward onto the square's border
/// before remapping from `[-1, 1]` to `[0, 1]`.
#[inline]
pub fn dir_to_oct(dir: Vec3) -> Vec2 {
    let l1 = dir.x.abs() + dir.y.abs() + dir.z.abs();
    // Guard against the zero vector so we never divide by zero; it maps to the
    // `+z` pole at the centre of the square.
    if l1 <= f32::MIN_POSITIVE {
        return Vec2::splat(0.5);
    }
    let inv = l1.recip();
    let mut p = Vec2::new(dir.x * inv, dir.y * inv);
    if dir.z < 0.0 {
        // Fold the back hemisphere out to the border.
        let folded = Vec2::new(
            (1.0 - p.y.abs()) * sign_not_zero(p.x),
            (1.0 - p.x.abs()) * sign_not_zero(p.y),
        );
        p = folded;
    }
    // Map [-1, 1] -> [0, 1].
    p * 0.5 + Vec2::splat(0.5)
}

/// Decodes an octahedral UV in `[0, 1]^2` back to a unit direction.
///
/// The inverse of [`dir_to_oct`]: it undoes the `[0, 1]` remap, reconstructs
/// `z` from the L1 constraint, folds the border back into the lower
/// hemisphere, and finally renormalises.
#[inline]
pub fn oct_to_dir(uv: Vec2) -> Vec3 {
    // Map [0, 1] -> [-1, 1].
    let e = uv * 2.0 - Vec2::ONE;
    let z = 1.0 - e.x.abs() - e.y.abs();
    let mut x = e.x;
    let mut y = e.y;
    if z < 0.0 {
        // Undo the outward fold applied to the back hemisphere.
        let t = -z;
        x = (1.0 - e.y.abs()) * sign_not_zero(e.x);
        y = (1.0 - e.x.abs()) * sign_not_zero(e.y);
        let _ = t;
    }
    Vec3::new(x, y, z).normalize()
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::{ops, Vec3};

    fn assert_round_trip(dir: Vec3) {
        let n = dir.normalize();
        let uv = dir_to_oct(n);
        assert!(
            uv.x >= -1e-6 && uv.x <= 1.0 + 1e-6 && uv.y >= -1e-6 && uv.y <= 1.0 + 1e-6,
            "uv {uv:?} out of range for dir {n:?}"
        );
        let back = oct_to_dir(uv);
        let err = (back - n).length();
        assert!(err < 1e-5, "round trip error {err} for dir {n:?} (uv {uv:?})");
    }

    #[test]
    fn round_trip_axes_and_poles() {
        for dir in [
            Vec3::X,
            Vec3::NEG_X,
            Vec3::Y,
            Vec3::NEG_Y,
            Vec3::Z,
            Vec3::NEG_Z,
        ] {
            assert_round_trip(dir);
        }
    }

    #[test]
    fn round_trip_diagonals_and_seams() {
        for dir in [
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(-1.0, 1.0, 1.0),
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::new(1.0, 1.0, -1.0),
            Vec3::new(-1.0, -1.0, 1.0),
            // Points exactly on the equator (z = 0) sit on the fold seam.
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.7, 0.7, 0.0),
        ] {
            assert_round_trip(dir);
        }
    }

    #[test]
    fn round_trip_dense_sphere() {
        let n = 24;
        for i in 0..n {
            for j in 0..n {
                let u = (i as f32 + 0.5) / n as f32;
                let v = (j as f32 + 0.5) / n as f32;
                // Fibonacci-ish spherical spread via spherical coords.
                let theta = core::f32::consts::PI * u;
                let phi = core::f32::consts::TAU * v;
                let dir = Vec3::new(
                    ops::sin(theta) * ops::cos(phi),
                    ops::sin(theta) * ops::sin(phi),
                    ops::cos(theta),
                );
                assert_round_trip(dir);
            }
        }
    }

    #[test]
    fn front_pole_maps_to_centre() {
        let uv = dir_to_oct(Vec3::Z);
        assert!((uv - Vec2::splat(0.5)).length() < 1e-6, "uv {uv:?}");
    }

    #[test]
    fn zero_vector_is_safe() {
        let uv = dir_to_oct(Vec3::ZERO);
        assert_eq!(uv, Vec2::splat(0.5));
        // decode of the centre is the +z pole.
        let dir = oct_to_dir(Vec2::splat(0.5));
        assert!((dir - Vec3::Z).length() < 1e-6, "dir {dir:?}");
    }

    #[test]
    fn decoded_directions_are_unit_length() {
        let n = 16;
        for i in 0..=n {
            for j in 0..=n {
                let uv = Vec2::new(i as f32 / n as f32, j as f32 / n as f32);
                let dir = oct_to_dir(uv);
                assert!((dir.length() - 1.0).abs() < 1e-5, "dir {dir:?} uv {uv:?}");
            }
        }
    }
}
