//! **Equirectangular (lat-long) full-sphere normal encoding** -- the classic
//! cylindrical parameterisation that maps a unit normal to a `[0, 1]^2` panel
//! whose axes are *linear in longitude and latitude*.
//!
//! Where the equal-area [`super::spheremap`] and the conformal
//! [`super::stereographic`] maps fold the sphere onto a plane, the
//! equirectangular (a.k.a. lat-long or plate carree) map lays it out on a
//! rectangle: the horizontal axis is longitude `lambda = atan2(n.y, n.x)` and
//! the vertical axis is latitude `phi = asin(n.z)`, each affinely rescaled to
//! `[0, 1]`. This is the on-disk layout of lat-long HDR environment panoramas
//! and the addressing convention a shader uses to sample them, so it is the
//! natural normal/direction encoding whenever the storage must line up with an
//! equirectangular texture rather than a packed G-buffer channel pair.
//!
//! Decoding is the exact inverse off the poles: `lambda = (u - 0.5) * 2*pi`,
//! `phi = (v - 0.5) * pi`, then `n = (cos(phi) cos(lambda), cos(phi) sin(lambda),
//! sin(phi))`, which is a unit vector for every `(u, v)` (it is a point on the
//! sphere by construction). The two poles (`n.z = +-1`) collapse the whole
//! `u` row to a single direction, so longitude is not recoverable there -- the
//! familiar lat-long pole singularity.
//!
//! [`equirect_decode`] is the exact inverse of [`equirect_encode`] for every
//! unit normal off the poles (round-trip error below float tolerance), the
//! primary anti-fake oracle. Because the output already lives in `[0, 1]^2`
//! there is no separate `unorm` storage form. Everything is deterministic
//! `f32` arithmetic routed through [`bevy_math::ops`] for libm determinism (no
//! AI/ML), so a CPU golden matches a GPU twin to floating-point tolerance.
//!
//! # References
//! * Reinhard et al., *High Dynamic Range Imaging* -- lat-long environment maps.
//! * Snyder, *Map Projections: A Working Manual*, USGS (1987) -- the
//!   equirectangular (plate carree) projection.

use bevy_math::ops;
use core::f32::consts::{PI, TAU};

/// Encode a full-sphere unit normal to its equirectangular panel coordinate
/// `(u, v)` in `[0, 1]^2`.
///
/// The horizontal coordinate is longitude `u = atan2(n.y, n.x) / (2*pi) + 0.5`
/// and the vertical coordinate is latitude `v = asin(n.z) / pi + 0.5`, so the
/// `+X` direction sits at the panel centre `(0.5, 0.5)`, the equator on the row
/// `v = 0.5`, and the `+Z` / `-Z` poles on the rows `v = 1` / `v = 0`. At a pole
/// the longitude is degenerate (`atan2(0, 0) = 0`), exactly as for every
/// lat-long chart.
#[inline]
#[must_use]
pub fn equirect_encode(n: [f32; 3]) -> [f32; 2] {
    let lambda = ops::atan2(n[1], n[0]);
    let phi = ops::asin(n[2].clamp(-1.0, 1.0));
    [lambda / TAU + 0.5, phi / PI + 0.5]
}

/// Decode an equirectangular panel coordinate back into a unit normal.
///
/// Inverts [`equirect_encode`]: `lambda = (u - 0.5) * 2*pi`,
/// `phi = (v - 0.5) * pi`, then `n = (cos(phi) cos(lambda),
/// cos(phi) sin(lambda), sin(phi))`. The result is a unit vector for every
/// `(u, v)` by construction (no singular branch).
#[inline]
#[must_use]
pub fn equirect_decode(uv: [f32; 2]) -> [f32; 3] {
    let lambda = (uv[0] - 0.5) * TAU;
    let phi = (uv[1] - 0.5) * PI;
    let (sin_phi, cos_phi) = ops::sin_cos(phi);
    let (sin_lambda, cos_lambda) = ops::sin_cos(lambda);
    [cos_phi * cos_lambda, cos_phi * sin_lambda, sin_phi]
}

#[cfg(test)]
mod tests {
    use super::{equirect_decode, equirect_encode};
    use alloc::vec::Vec;
    use bevy_math::ops;
    use core::f32::consts::TAU;

    fn sphere_samples() -> Vec<[f32; 3]> {
        let mut out = Vec::new();
        // Avoid the exact poles (longitude is degenerate there).
        for iz in -98..=98 {
            let z = iz as f32 / 100.0;
            let r = (1.0 - z * z).max(0.0).sqrt();
            for ia in 0..16 {
                let a = ia as f32 / 16.0 * TAU;
                let (s, c) = ops::sin_cos(a);
                out.push([r * c, r * s, z]);
            }
        }
        out
    }

    /// Decoding an encoded normal recovers it exactly across the whole sphere
    /// (off the poles): the primary anti-fake round-trip oracle.
    #[test]
    fn round_trips_full_sphere() {
        for n in sphere_samples() {
            let got = equirect_decode(equirect_encode(n));
            for k in 0..3 {
                assert!((got[k] - n[k]).abs() < 1e-5, "{got:?} vs {n:?}");
            }
        }
    }

    /// Every panel coordinate decodes to an exact unit vector (the map covers
    /// the whole sphere with no holes).
    #[test]
    fn decode_is_always_unit() {
        for iv in 0..=40 {
            for iu in 0..=40 {
                let uv = [iu as f32 / 40.0, iv as f32 / 40.0];
                let n = equirect_decode(uv);
                let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
                assert!((len - 1.0).abs() < 1e-5, "uv {uv:?} len {len}");
            }
        }
    }

    /// The +X axis sits at the panel centre and the poles / equator land on the
    /// expected latitude rows.
    #[test]
    fn landmarks_map_to_expected_rows() {
        let c = equirect_encode([1.0, 0.0, 0.0]);
        assert!(
            (c[0] - 0.5).abs() < 1e-6 && (c[1] - 0.5).abs() < 1e-6,
            "+X centre {c:?}"
        );
        // +Y is a quarter turn of longitude east of +X.
        let py = equirect_encode([0.0, 1.0, 0.0]);
        assert!((py[0] - 0.75).abs() < 1e-6, "+Y longitude {py:?}");
        // Poles sit on the top / bottom rows and decode back to +-Z.
        assert!(
            (equirect_encode([0.0, 0.0, 1.0])[1] - 1.0).abs() < 1e-6,
            "+Z row"
        );
        assert!(equirect_encode([0.0, 0.0, -1.0])[1].abs() < 1e-6, "-Z row");
        assert!(
            (equirect_decode([0.3, 1.0])[2] - 1.0).abs() < 1e-5,
            "v=1 -> +Z"
        );
        assert!(
            (equirect_decode([0.7, 0.0])[2] + 1.0).abs() < 1e-5,
            "v=0 -> -Z"
        );
    }

    /// Latitude is linear in `v`: a normal at elevation angle `phi` lands on
    /// row `phi/pi + 0.5` regardless of longitude.
    #[test]
    fn latitude_is_linear_in_v() {
        for ip in -8..=8 {
            let phi = ip as f32 / 8.0 * (core::f32::consts::FRAC_PI_2 * 0.99);
            let (sz, cz) = ops::sin_cos(phi);
            let v = equirect_encode([cz, 0.0, sz])[1];
            assert!(
                (v - (phi / core::f32::consts::PI + 0.5)).abs() < 1e-5,
                "lat {phi} v {v}"
            );
        }
    }

    /// Rotating the normal about Z shifts the longitude coordinate by a constant
    /// (modulo the `[0, 1]` wrap): the map is azimuthally uniform.
    #[test]
    fn rotation_about_z_shifts_u() {
        let alpha = 0.6_f32;
        let (s, c) = ops::sin_cos(alpha);
        let shift = alpha / TAU;
        for n in sphere_samples() {
            // Skip near-polar samples where u is numerically unstable.
            if n[2].abs() > 0.95 {
                continue;
            }
            let rot = [c * n[0] - s * n[1], s * n[0] + c * n[1], n[2]];
            let u0 = equirect_encode(n)[0];
            let u1 = equirect_encode(rot)[0];
            let mut d = (u1 - u0) - shift;
            d -= d.round();
            assert!(d.abs() < 1e-4, "u shift {d} (u0 {u0} u1 {u1})");
        }
    }

    /// All encoded coordinates stay within the `[0, 1]^2` storage panel.
    #[test]
    fn encoded_coords_stay_in_unit_square() {
        for n in sphere_samples() {
            let uv = equirect_encode(n);
            assert!((0.0..=1.0).contains(&uv[0]), "u {}", uv[0]);
            assert!((0.0..=1.0).contains(&uv[1]), "v {}", uv[1]);
        }
    }
}
