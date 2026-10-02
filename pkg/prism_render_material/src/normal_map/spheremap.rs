//! **Lambert azimuthal equal-area ("spheremap transform") normal encoding** --
//! a two-channel packing that stores a *full-sphere* unit normal in a disk.
//!
//! Where the hemi-octahedral map in [`super::octahedral`] assumes a tangent-space
//! normal (`z >= 0`), deferred renderers also need to pack *view-space* normals,
//! whose `z` can be negative at grazing silhouettes. The **spheremap transform**
//! (Lambert azimuthal equal-area projection from the `-Z` pole) folds the entire
//! sphere -- except the single antipodal point `(0, 0, -1)` -- onto the disk of
//! radius `2`, and it is *area preserving*: equal solid angles on the sphere map
//! to equal areas in the disk, so quantisation error is spread uniformly instead
//! of bunching at a pole. This is the classic `CryEngine 3` / "best-fit" G-buffer
//! normal encoding popularised by Pranckevicius's survey.
//!
//! Encoding a unit normal `n` is `g = sqrt((1 + n.z) / 2)`, `p = n.xy / g`; the
//! disk coordinate `p` satisfies `|p|^2 = 2 (1 - n.z)`, so the `+Z` pole lands at
//! the origin and the equator on the circle `|p| = sqrt(2)`. Decoding inverts it
//! with `f = |p|^2`, `n.z = 1 - f / 2`, `n.xy = p sqrt(1 - f / 4)`; the result is
//! an exact unit vector for every `|p| <= 2` with no square-root of a negative.
//!
//! [`spheremap_decode`] is the exact inverse of [`spheremap_encode`] for every
//! unit normal off the `-Z` pole (round-trip error below float tolerance), which
//! is the primary anti-fake oracle. The `unorm` helpers only remap the disk from
//! `[-2, 2]` to `[0, 1]` for texel storage. Everything is deterministic analytic
//! `f32` arithmetic with the square roots routed through `bevy_math::ops` (no
//! AI/ML), so a CPU golden matches a GPU twin to floating-point tolerance.
//!
//! # References
//! * Pranckevicius, "Compact Normal Storage for Small G-Buffers" (2009) -- the
//!   spheremap transform survey (Lambert azimuthal equal-area).
//! * Snyder, *Map Projections: A Working Manual*, USGS (1987) -- the Lambert
//!   azimuthal equal-area projection.

use bevy_math::ops;

/// Encode a full-sphere unit normal into its signed Lambert azimuthal disk
/// coordinate `p` (components in `[-2, 2]`, `|p| <= 2`).
///
/// `g = sqrt((1 + n.z) / 2)` then `p = n.xy / g`. The `+Z` pole maps to the
/// origin and the equator to the circle `|p| = sqrt(2)`. The antipodal `-Z` pole
/// (`n.z = -1`) is the single projection singularity; it is clamped to the disk
/// rim `(2, 0)` instead of dividing by zero.
#[inline]
#[must_use]
pub fn spheremap_encode(n: [f32; 3]) -> [f32; 2] {
    let g = ops::sqrt((1.0 + n[2]) * 0.5);
    if g > 0.0 {
        [n[0] / g, n[1] / g]
    } else {
        // Antipodal -Z pole: the whole disk rim collapses here; pick a point on
        // the circle |p| = 2 so a decode still returns a unit vector near -Z.
        [2.0, 0.0]
    }
}

/// Decode a signed Lambert azimuthal disk coordinate back into a unit normal.
///
/// `f = |p|^2`, `n.z = 1 - f / 2`, `n.xy = p sqrt(1 - f / 4)`. The output is an
/// exact unit vector for every `|p| <= 2`; coordinates past the rim are clamped
/// so the inner square root stays non-negative.
#[inline]
#[must_use]
pub fn spheremap_decode(p: [f32; 2]) -> [f32; 3] {
    let f = (p[0] * p[0] + p[1] * p[1]).min(4.0);
    let g = ops::sqrt(1.0 - f * 0.25);
    [p[0] * g, p[1] * g, 1.0 - f * 0.5]
}

/// Encode a unit normal into a `[0, 1]^2` texel coordinate for storage.
///
/// Identical to [`spheremap_encode`] followed by the affine remap
/// `p * 0.25 + 0.5` that maps the disk `[-2, 2]` into the unit square.
#[inline]
#[must_use]
pub fn spheremap_encode_unorm(n: [f32; 3]) -> [f32; 2] {
    let p = spheremap_encode(n);
    [p[0] * 0.25 + 0.5, p[1] * 0.25 + 0.5]
}

/// Decode a `[0, 1]^2` texel coordinate (as written by
/// [`spheremap_encode_unorm`]) back into a unit normal.
#[inline]
#[must_use]
pub fn spheremap_decode_unorm(e: [f32; 2]) -> [f32; 3] {
    spheremap_decode([e[0] * 4.0 - 2.0, e[1] * 4.0 - 2.0])
}

#[cfg(test)]
mod tests {
    use super::{
        spheremap_decode, spheremap_decode_unorm, spheremap_encode, spheremap_encode_unorm,
    };
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
            let got = spheremap_decode(spheremap_encode(n));
            for k in 0..3 {
                assert!((got[k] - n[k]).abs() < 1e-5, "{got:?} vs {n:?}");
            }
        }
    }

    /// The `[0,1]^2` storage form round-trips too and stays inside the unit square.
    #[test]
    fn unorm_round_trips_and_in_range() {
        for n in sphere_samples() {
            let e = spheremap_encode_unorm(n);
            for c in e {
                assert!((-1e-4..=1.0 + 1e-4).contains(&c), "unorm {c} out of range");
            }
            let got = spheremap_decode_unorm(e);
            for k in 0..3 {
                assert!((got[k] - n[k]).abs() < 1e-5, "{got:?} vs {n:?}");
            }
        }
    }

    /// The +Z pole maps to the disk origin and decodes back to +Z.
    #[test]
    fn plus_z_pole_is_origin() {
        let p = spheremap_encode([0.0, 0.0, 1.0]);
        assert!(p[0].abs() < 1e-6 && p[1].abs() < 1e-6, "+Z -> origin {p:?}");
        let n = spheremap_decode([0.0, 0.0]);
        assert_eq!(n, [0.0, 0.0, 1.0]);
    }

    /// Equal-area law: the squared disk radius equals `2 (1 - n.z)`, so the
    /// equator lands on the circle |p| = sqrt(2).
    #[test]
    fn disk_radius_follows_equal_area_law() {
        for n in sphere_samples() {
            let p = spheremap_encode(n);
            let r2 = p[0] * p[0] + p[1] * p[1];
            assert!((r2 - 2.0 * (1.0 - n[2])).abs() < 1e-4, "r2 {r2} z {}", n[2]);
        }
        // Equator sample sits on |p| = sqrt(2).
        let p = spheremap_encode(unit([1.0, 0.0, 0.0]));
        let r = (p[0] * p[0] + p[1] * p[1]).sqrt();
        assert!(
            (r - core::f32::consts::SQRT_2).abs() < 1e-5,
            "equator r {r}"
        );
    }

    /// Decoding any point inside the disk yields an exact unit vector (no holes,
    /// no imaginary square root), including the rim which maps near the -Z pole.
    #[test]
    fn decode_is_always_unit_on_disk() {
        for iy in -20..=20 {
            for ix in -20..=20 {
                let p = [ix as f32 / 10.0, iy as f32 / 10.0];
                let n = spheremap_decode(p);
                let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
                assert!((len - 1.0).abs() < 1e-5, "p {p:?} len {len}");
            }
        }
        // The rim |p| = 2 maps onto the -Z pole.
        let n = spheremap_decode([2.0, 0.0]);
        assert!((n[2] + 1.0).abs() < 1e-5, "rim -> -Z {n:?}");
    }

    /// The unorm form is exactly the signed disk coordinate remapped by
    /// `p * 0.25 + 0.5`.
    #[test]
    fn unorm_is_signed_remapped() {
        for n in sphere_samples() {
            let p = spheremap_encode(n);
            let e = spheremap_encode_unorm(n);
            assert!((e[0] - (p[0] * 0.25 + 0.5)).abs() < 1e-6, "x remap");
            assert!((e[1] - (p[1] * 0.25 + 0.5)).abs() < 1e-6, "y remap");
        }
    }

    /// The projection is azimuthal: rotating the normal about Z by an angle
    /// rotates the disk coordinate by the same angle (azimuth is preserved).
    #[test]
    fn rotation_about_z_rotates_disk() {
        let theta = 0.7_f32;
        let (s, c) = ops::sin_cos(theta);
        for n in sphere_samples() {
            let rot = [c * n[0] - s * n[1], s * n[0] + c * n[1], n[2]];
            let p = spheremap_encode(n);
            let pr = spheremap_encode(rot);
            let want = [c * p[0] - s * p[1], s * p[0] + c * p[1]];
            assert!((pr[0] - want[0]).abs() < 1e-4, "rot x {pr:?} {want:?}");
            assert!((pr[1] - want[1]).abs() < 1e-4, "rot y {pr:?} {want:?}");
        }
    }
}
