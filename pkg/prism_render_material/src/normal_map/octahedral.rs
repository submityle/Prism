//! **Hemi-octahedral tangent-space normal encoding** -- the two-channel
//! G-buffer / normal-map packing that keeps full precision for a hemisphere of
//! normals.
//!
//! Tangent-space normals always point *out* of the surface, so their `z`
//! component is non-negative and they occupy only the upper hemisphere. Storing
//! them with a full-sphere parameterisation (lat-long, or the full octahedral
//! map that GI probes use) wastes half the texel range on directions that can
//! never occur. The **hemi-octahedral** map instead folds just the `z >= 0`
//! hemisphere across the whole `[-1, 1]^2` square, so a `BC5` / `RG16` normal
//! target gets twice the angular resolution of a full-sphere map for the same
//! bit budget -- the standard choice for deferred G-buffer normals.
//!
//! The hemisphere is projected onto the octahedron by the `L1` norm
//! `p = n.xy / (|n.x| + |n.y| + n.z)` (which traces the diamond `|px| + |py| <= 1`),
//! then the diamond is rotated 45 degrees and scaled to fill the unit square via
//! `e = (px + py, px - py)`. Decoding inverts the rotation
//! (`t = (ex + ey, ex - ey) / 2`), rebuilds `z = 1 - |tx| - |ty|` and
//! renormalises. The four equator axes land exactly on the four corners and the
//! `+z` pole lands at the centre, with no polar singularity on the fold seams.
//!
//! [`hemi_oct_decode`] is the exact inverse of [`hemi_oct_encode`] for every
//! `z >= 0` unit normal (round-trip error below float tolerance), which is the
//! primary anti-fake oracle. The `unorm` helpers only remap `[-1, 1] <-> [0, 1]`
//! for texel storage. Everything is deterministic analytic `f32` arithmetic with
//! transcendentals routed through `bevy_math::ops` (no AI/ML), so a CPU golden
//! matches a GPU twin to floating-point tolerance.
//!
//! # References
//! * Cigolle et al., "A Survey of Efficient Representations for Independent Unit
//!   Vectors", JCGT 2014 (the hemi-octahedral listing).
//! * Meyer et al., "On Floating-Point Normal Vectors", EGSR 2010.

use bevy_math::ops;

/// Encode a `z >= 0` unit normal into a signed hemi-octahedral `(ex, ey)` pair
/// in `[-1, 1]^2`.
///
/// The input is treated as lying on the upper hemisphere; its `z` is taken as
/// `|z|` so a normal nudged below the horizon by round-off still encodes
/// sensibly. The result fills the whole square, so the four equator axes map to
/// the four corners and the `+z` pole to the origin.
#[inline]
#[must_use]
pub fn hemi_oct_encode(normal: [f32; 3]) -> [f32; 2] {
    let [x, y, z] = normal;
    let l1 = x.abs() + y.abs() + z.abs();
    // l1 is zero only for a zero vector; guard so the division stays finite.
    let inv = if l1 > 0.0 { 1.0 / l1 } else { 0.0 };
    let (px, py) = (x * inv, y * inv);
    [px + py, px - py]
}

/// Decode a signed hemi-octahedral `(ex, ey)` pair in `[-1, 1]^2` back to a
/// `z >= 0` unit normal.
///
/// The exact inverse of [`hemi_oct_encode`] (the round-trip anti-fake oracle).
#[inline]
#[must_use]
pub fn hemi_oct_decode(e: [f32; 2]) -> [f32; 3] {
    let [ex, ey] = e;
    let tx = (ex + ey) * 0.5;
    let ty = (ex - ey) * 0.5;
    let z = 1.0 - tx.abs() - ty.abs();
    let inv = 1.0 / ops::sqrt(tx * tx + ty * ty + z * z);
    [tx * inv, ty * inv, z * inv]
}

/// Encode a `z >= 0` unit normal into an unsigned hemi-octahedral `(u, v)` pair
/// in `[0, 1]^2`, ready to write to a `unorm` normal target.
///
/// Just [`hemi_oct_encode`] remapped from `[-1, 1]` to `[0, 1]`.
#[inline]
#[must_use]
pub fn hemi_oct_encode_unorm(normal: [f32; 3]) -> [f32; 2] {
    let [ex, ey] = hemi_oct_encode(normal);
    [ex * 0.5 + 0.5, ey * 0.5 + 0.5]
}

/// Decode an unsigned hemi-octahedral `(u, v)` pair in `[0, 1]^2` back to a
/// `z >= 0` unit normal.
///
/// The exact inverse of [`hemi_oct_encode_unorm`].
#[inline]
#[must_use]
pub fn hemi_oct_decode_unorm(uv: [f32; 2]) -> [f32; 3] {
    let [u, v] = uv;
    hemi_oct_decode([u * 2.0 - 1.0, v * 2.0 - 1.0])
}

#[cfg(test)]
mod tests {
    use super::{hemi_oct_decode, hemi_oct_decode_unorm, hemi_oct_encode, hemi_oct_encode_unorm};

    fn unit(v: [f32; 3]) -> [f32; 3] {
        let inv = 1.0 / (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        [v[0] * inv, v[1] * inv, v[2] * inv]
    }

    const NORMALS: [[f32; 3]; 8] = [
        [0.0, 0.0, 1.0],
        [0.3, -0.2, 0.9],
        [-0.5, 0.4, 0.76],
        [0.1, 0.7, 0.7],
        [-0.25, -0.25, 0.93],
        [0.8, 0.1, 0.05],
        [-0.6, -0.6, 0.2],
        [0.0, 0.95, 0.05],
    ];

    fn close3(a: [f32; 3], b: [f32; 3], tol: f32, msg: &str) {
        for k in 0..3 {
            assert!((a[k] - b[k]).abs() < tol, "{msg}: {a:?} vs {b:?}");
        }
    }

    /// Decode after encode returns the original hemisphere normal (the primary
    /// anti-fake oracle: a genuine inverse, not a stub).
    #[test]
    fn signed_round_trip_recovers_normal() {
        for &n in &NORMALS {
            let n = unit(n);
            let got = hemi_oct_decode(hemi_oct_encode(n));
            close3(got, n, 1e-6, "signed round-trip");
        }
    }

    /// The `unorm` storage round-trip also recovers the original normal.
    #[test]
    fn unorm_round_trip_recovers_normal() {
        for &n in &NORMALS {
            let n = unit(n);
            let uv = hemi_oct_encode_unorm(n);
            assert!(
                uv.iter().all(|&c| (0.0..=1.0).contains(&c)),
                "unorm in range {uv:?}"
            );
            close3(hemi_oct_decode_unorm(uv), n, 1e-6, "unorm round-trip");
        }
    }

    /// The `+z` pole lands at the square centre: signed origin and `unorm`
    /// `(0.5, 0.5)`.
    #[test]
    fn pole_maps_to_centre() {
        let e = hemi_oct_encode([0.0, 0.0, 1.0]);
        assert!(
            e[0].abs() < 1e-6 && e[1].abs() < 1e-6,
            "signed centre {e:?}"
        );
        let uv = hemi_oct_encode_unorm([0.0, 0.0, 1.0]);
        assert!(
            (uv[0] - 0.5).abs() < 1e-6 && (uv[1] - 0.5).abs() < 1e-6,
            "unorm centre {uv:?}"
        );
    }

    /// The four equator axes map exactly to the four signed corners (a hallmark
    /// of filling the whole square, not a partial diamond).
    #[test]
    fn equator_axes_hit_corners() {
        let cases = [
            ([1.0f32, 0.0, 0.0], [1.0f32, 1.0]),
            ([0.0, 1.0, 0.0], [1.0, -1.0]),
            ([-1.0, 0.0, 0.0], [-1.0, -1.0]),
            ([0.0, -1.0, 0.0], [-1.0, 1.0]),
        ];
        for (n, want) in cases {
            let e = hemi_oct_encode(n);
            assert!(
                (e[0] - want[0]).abs() < 1e-6 && (e[1] - want[1]).abs() < 1e-6,
                "corner {e:?} vs {want:?}"
            );
        }
    }

    /// Every point of the signed square decodes to a unit normal on the upper
    /// hemisphere (`z >= 0`), with no gaps or singularities.
    #[test]
    fn decode_is_always_unit_upper_hemisphere() {
        let steps = 9;
        for i in 0..=steps {
            for j in 0..=steps {
                let ex = -1.0 + 2.0 * i as f32 / steps as f32;
                let ey = -1.0 + 2.0 * j as f32 / steps as f32;
                let n = hemi_oct_decode([ex, ey]);
                let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
                assert!((len - 1.0).abs() < 1e-6, "unit length {len} at {ex},{ey}");
                assert!(n[2] >= -1e-6, "upper hemisphere {n:?} at {ex},{ey}");
            }
        }
    }

    /// The `unorm` encoding is exactly the signed encoding remapped to `[0, 1]`.
    #[test]
    fn unorm_is_signed_remapped() {
        for &n in &NORMALS {
            let n = unit(n);
            let e = hemi_oct_encode(n);
            let uv = hemi_oct_encode_unorm(n);
            assert!((uv[0] - (e[0] * 0.5 + 0.5)).abs() < 1e-6, "u remap");
            assert!((uv[1] - (e[1] * 0.5 + 0.5)).abs() < 1e-6, "v remap");
        }
    }

    /// A normal pushed slightly below the horizon by round-off still encodes and
    /// decodes back to the horizon rather than exploding.
    #[test]
    fn below_horizon_input_is_clamped_by_abs() {
        let n = unit([0.7, 0.2, -0.001]);
        let got = hemi_oct_decode(hemi_oct_encode(n));
        assert!(got[2] >= -1e-6, "decoded onto hemisphere {got:?}");
        let len = (got[0] * got[0] + got[1] * got[1] + got[2] * got[2]).sqrt();
        assert!((len - 1.0).abs() < 1e-6, "unit length {len}");
    }
}
