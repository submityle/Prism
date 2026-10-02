//! **Stereographic (conformal) full-sphere normal encoding** -- the
//! angle-preserving companion to the equal-area [`super::spheremap`] map.
//!
//! Stereographic projection from the `-Z` pole sends a unit normal `n` to the
//! plane point `p = n.xy / (1 + n.z)`. Unlike the Lambert azimuthal map (which
//! preserves *area* and lives on a bounded disk), the stereographic map
//! preserves *angles* (it is conformal) at the cost of an unbounded plane: the
//! `+Z` pole lands at the origin, the equator on the unit circle `|p| = 1`, and
//! normals approaching the antipodal `-Z` pole shoot off to infinity. Conformal
//! encodings keep the local shape of a normal distribution undistorted, which is
//! why they appear in Pranckevicius's G-buffer survey alongside the equal-area
//! option and in spherical-harmonic / environment remapping work.
//!
//! Decoding inverts it in closed form: with `d = |p|^2`,
//! `n = (2 p.x, 2 p.y, 1 - d) / (1 + d)`, which is an exact unit vector for every
//! finite `p` (the denominator `1 + d >= 1` never vanishes), so there is no
//! singular branch on decode.
//!
//! [`stereographic_decode`] is the exact inverse of [`stereographic_encode`] for
//! every unit normal off the `-Z` pole (round-trip error below float tolerance),
//! the primary anti-fake oracle, and the defining identity
//! `|p|^2 (1 + n.z) = 1 - n.z` distinguishes this map from the equal-area one.
//! Because the plane is unbounded there is no fixed `[0, 1]` storage mapping, so
//! only the signed-plane transforms are offered. Everything is deterministic
//! analytic `f32` arithmetic (no transcendentals, no AI/ML), so a CPU golden
//! matches a GPU twin to floating-point tolerance.
//!
//! # References
//! * Pranckevicius, "Compact Normal Storage for Small G-Buffers" (2009) -- the
//!   stereographic option in the normal-encoding survey.
//! * Snyder, *Map Projections: A Working Manual*, USGS (1987) -- the
//!   stereographic (conformal) projection.

/// Encode a full-sphere unit normal to its stereographic plane coordinate
/// `p = n.xy / (1 + n.z)`.
///
/// The `+Z` pole maps to the origin and the equator (`n.z = 0`) to the unit
/// circle `|p| = 1`; as `n.z -> -1` the magnitude grows without bound. The
/// antipodal `-Z` pole (`n.z = -1`) is the single projection singularity; it is
/// returned as `(0, 0)` rather than dividing by zero (callers must treat the
/// pole specially, as with every stereographic chart).
#[inline]
#[must_use]
pub fn stereographic_encode(n: [f32; 3]) -> [f32; 2] {
    let denom = 1.0 + n[2];
    if denom > 0.0 {
        [n[0] / denom, n[1] / denom]
    } else {
        [0.0, 0.0]
    }
}

/// Decode a stereographic plane coordinate back into a unit normal.
///
/// With `d = |p|^2`, returns `(2 p.x, 2 p.y, 1 - d) / (1 + d)`. The result is an
/// exact unit vector for every finite `p`; there is no singular branch because
/// the denominator `1 + d` is always at least `1`.
#[inline]
#[must_use]
pub fn stereographic_decode(p: [f32; 2]) -> [f32; 3] {
    let d = p[0] * p[0] + p[1] * p[1];
    let inv = 1.0 / (1.0 + d);
    [2.0 * p[0] * inv, 2.0 * p[1] * inv, (1.0 - d) * inv]
}

#[cfg(test)]
mod tests {
    use super::{stereographic_decode, stereographic_encode};
    use alloc::vec::Vec;
    use bevy_math::ops;

    fn unit(v: [f32; 3]) -> [f32; 3] {
        let inv = 1.0 / (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        [v[0] * inv, v[1] * inv, v[2] * inv]
    }

    fn sphere_samples() -> Vec<[f32; 3]> {
        let mut out = Vec::new();
        // Avoid the exact -Z pole (the projection singularity).
        for iz in -98..=100 {
            let z = iz as f32 / 100.0;
            let r = (1.0 - z * z).max(0.0).sqrt();
            for ia in 0..16 {
                let a = ia as f32 / 16.0 * core::f32::consts::TAU;
                let (s, c) = ops::sin_cos(a);
                out.push([r * c, r * s, z]);
            }
        }
        out
    }

    /// Decoding an encoded normal recovers it exactly across the whole sphere
    /// (off the -Z pole): the primary anti-fake round-trip oracle.
    #[test]
    fn round_trips_full_sphere() {
        for n in sphere_samples() {
            let got = stereographic_decode(stereographic_encode(n));
            for k in 0..3 {
                assert!((got[k] - n[k]).abs() < 1e-5, "{got:?} vs {n:?}");
            }
        }
    }

    /// The +Z pole maps to the origin and decodes back to +Z.
    #[test]
    fn plus_z_pole_is_origin() {
        let p = stereographic_encode([0.0, 0.0, 1.0]);
        assert!(p[0].abs() < 1e-6 && p[1].abs() < 1e-6, "+Z -> origin {p:?}");
        assert_eq!(stereographic_decode([0.0, 0.0]), [0.0, 0.0, 1.0]);
    }

    /// The equator (n.z = 0) maps onto the unit circle |p| = 1.
    #[test]
    fn equator_maps_to_unit_circle() {
        for ia in 0..32 {
            let a = ia as f32 / 32.0 * core::f32::consts::TAU;
            let (s, c) = ops::sin_cos(a);
            let p = stereographic_encode([c, s, 0.0]);
            let r = (p[0] * p[0] + p[1] * p[1]).sqrt();
            assert!((r - 1.0).abs() < 1e-5, "equator r {r}");
        }
    }

    /// The defining conformal identity `|p|^2 (1 + n.z) = 1 - n.z` holds for
    /// every sample -- this is what separates the stereographic map from the
    /// equal-area spheremap (whose law is `|p|^2 = 2 (1 - n.z)`).
    #[test]
    fn radius_identity_distinguishes_from_equal_area() {
        for n in sphere_samples() {
            let p = stereographic_encode(n);
            let d = p[0] * p[0] + p[1] * p[1];
            assert!(
                (d * (1.0 + n[2]) - (1.0 - n[2])).abs() < 1e-4,
                "identity z {}",
                n[2]
            );
        }
    }

    /// Decoding any finite plane point yields an exact unit vector (no holes, no
    /// division by zero), including far-out points that map near the -Z pole.
    #[test]
    fn decode_is_always_unit() {
        for iy in -50..=50 {
            for ix in -50..=50 {
                let p = [ix as f32 / 5.0, iy as f32 / 5.0];
                let n = stereographic_decode(p);
                let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
                assert!((len - 1.0).abs() < 1e-5, "p {p:?} len {len}");
            }
        }
    }

    /// The projection is azimuthal: rotating the normal about Z rotates the
    /// plane coordinate by the same angle.
    #[test]
    fn rotation_about_z_rotates_plane() {
        let theta = 0.7_f32;
        let (s, c) = ops::sin_cos(theta);
        for n in sphere_samples() {
            let rot = [c * n[0] - s * n[1], s * n[0] + c * n[1], n[2]];
            let p = stereographic_encode(n);
            let pr = stereographic_encode(rot);
            let want = [c * p[0] - s * p[1], s * p[0] + c * p[1]];
            assert!((pr[0] - want[0]).abs() < 1e-4, "rot x {pr:?} {want:?}");
            assert!((pr[1] - want[1]).abs() < 1e-4, "rot y {pr:?} {want:?}");
        }
    }

    /// A unit normal lying in the X-Z plane at the given latitude `z`.
    fn on_meridian(z: f32) -> [f32; 3] {
        unit([(1.0 - z * z).max(0.0).sqrt(), 0.0, z])
    }

    /// Normals nearer the -Z pole project strictly farther out: the chart grows
    /// without bound toward the singularity.
    #[test]
    fn magnitude_grows_toward_antipode() {
        let near = on_meridian(-0.99);
        let far = on_meridian(-0.9);
        let rn = {
            let p = stereographic_encode(near);
            (p[0] * p[0] + p[1] * p[1]).sqrt()
        };
        let rf = {
            let p = stereographic_encode(far);
            (p[0] * p[0] + p[1] * p[1]).sqrt()
        };
        assert!(rn > rf, "nearer -Z projects farther: {rn} vs {rf}");
        assert!(rn > 10.0, "near -0.99 pole radius should be large: {rn}");
    }
}
