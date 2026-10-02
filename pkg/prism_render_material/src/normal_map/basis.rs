//! **Tangent-basis orthonormalization and tangent/object-space normal
//! transforms** -- the `TBN` plumbing that moves a sampled normal map between
//! the space it is authored in and the space the geometry lives in.
//!
//! A tangent-space normal map stores perturbations relative to the surface, in
//! the frame spanned by the mesh tangent `T`, bitangent `B` and geometric
//! normal `N`. To light the surface (or to blend a decal / terrain layer that
//! was baked in a different frame) that normal must be rotated into object or
//! world space by the `TBN` matrix `[T | B | N]`. Interpolating `T` and `N`
//! across a triangle leaves them slightly non-orthogonal and non-unit, so the
//! per-pixel basis is first re-orthonormalized exactly the way `MikkTSpace`
//! bakers assume: `N` is normalized, `T` is Gram-Schmidt-projected off `N`
//! (`T' = normalize(T - N (N·T))`), and `B` is rebuilt as
//! `handedness · (N × T')` so a mirrored `UV` chart keeps the correct chirality.
//!
//! Because the resulting basis is orthonormal, its inverse is its transpose, so
//! [`object_to_tangent`] is the exact inverse of [`tangent_to_object`] for every
//! normal (the primary anti-fake oracle), the transforms preserve length, and
//! the signed basis determinant equals the stored handedness. Everything is
//! deterministic analytic `f32` arithmetic with the one square root routed
//! through `bevy_math::ops` (no AI/ML), so a CPU golden matches a GPU twin to
//! floating-point tolerance.
//!
//! # References
//! * Mikkelsen, "Simulation of Wrinkled Surfaces Revisited" (2008) -- the
//!   `MikkTSpace` tangent convention.
//! * Lengyel, "Computing Tangent Space Basis Vectors for an Arbitrary Mesh"
//!   (2001) -- Gram-Schmidt orthonormalization and the handedness sign.
//! * Akenine-Moller et al., *Real-Time Rendering* 4th ed., Section 6.8.

use bevy_math::ops;

#[inline]
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = ops::sqrt(dot(v, v));
    let inv = if len > 0.0 { 1.0 / len } else { 0.0 };
    [v[0] * inv, v[1] * inv, v[2] * inv]
}

/// An orthonormal tangent frame `(T, B, N)` ready to transform normals.
///
/// The three axes are mutually orthogonal unit vectors; `cross(tangent,
/// bitangent)` equals `normal` scaled by the basis handedness.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TangentBasis {
    /// Tangent axis `T` (the `U` texture direction).
    pub tangent: [f32; 3],
    /// Bitangent axis `B` (the `V` texture direction).
    pub bitangent: [f32; 3],
    /// Surface normal axis `N`.
    pub normal: [f32; 3],
}

/// Build an orthonormal [`TangentBasis`] from a raw interpolated tangent and
/// normal plus a handedness sign.
///
/// `N` is normalized, `T` is Gram-Schmidt-projected to be orthogonal to `N` and
/// normalized, and `B = handedness · (N × T)`. `handedness` is the sign stored
/// in the fourth tangent channel (`+1` or `-1`); any non-negative value is
/// treated as `+1` and any negative value as `-1`.
#[inline]
#[must_use]
pub fn orthonormalize_basis(tangent: [f32; 3], normal: [f32; 3], handedness: f32) -> TangentBasis {
    let n = normalize(normal);
    let d = dot(n, tangent);
    let t = normalize([
        tangent[0] - n[0] * d,
        tangent[1] - n[1] * d,
        tangent[2] - n[2] * d,
    ]);
    let sign = if handedness < 0.0 { -1.0 } else { 1.0 };
    let bitangent_raw = cross(n, t);
    let b = [
        bitangent_raw[0] * sign,
        bitangent_raw[1] * sign,
        bitangent_raw[2] * sign,
    ];
    TangentBasis {
        tangent: t,
        bitangent: b,
        normal: n,
    }
}

/// Rotate a tangent-space normal into the basis's parent (object / world) space.
///
/// Applies `[T | B | N] · n` so tangent-space `+Z` maps to the surface normal.
#[inline]
#[must_use]
pub fn tangent_to_object(basis: &TangentBasis, n: [f32; 3]) -> [f32; 3] {
    [
        basis.tangent[0] * n[0] + basis.bitangent[0] * n[1] + basis.normal[0] * n[2],
        basis.tangent[1] * n[0] + basis.bitangent[1] * n[1] + basis.normal[1] * n[2],
        basis.tangent[2] * n[0] + basis.bitangent[2] * n[1] + basis.normal[2] * n[2],
    ]
}

/// Rotate an object / world-space normal back into tangent space.
///
/// Applies the transpose `[T | B | N]^T · v`, which is the exact inverse of
/// [`tangent_to_object`] because the basis is orthonormal.
#[inline]
#[must_use]
pub fn object_to_tangent(basis: &TangentBasis, v: [f32; 3]) -> [f32; 3] {
    [
        dot(basis.tangent, v),
        dot(basis.bitangent, v),
        dot(basis.normal, v),
    ]
}

#[cfg(test)]
mod tests {
    use super::{
        cross, dot, object_to_tangent, orthonormalize_basis, tangent_to_object, TangentBasis,
    };

    fn unit(v: [f32; 3]) -> [f32; 3] {
        let inv = 1.0 / (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        [v[0] * inv, v[1] * inv, v[2] * inv]
    }

    const RAW: [([f32; 3], [f32; 3], f32); 4] = [
        ([1.0, 0.1, 0.0], [0.0, 0.0, 1.0], 1.0),
        ([0.9, -0.2, 0.3], [0.1, 0.2, 0.97], 1.0),
        ([0.5, 0.8, -0.1], [-0.2, 0.1, 0.9], -1.0),
        ([1.0, 0.0, 0.4], [0.0, 1.0, 0.0], -1.0),
    ];

    fn make(i: usize) -> TangentBasis {
        let (t, n, h) = RAW[i];
        orthonormalize_basis(t, n, h)
    }

    /// The constructed basis is orthonormal: unit axes, mutually perpendicular.
    #[test]
    fn basis_is_orthonormal() {
        for i in 0..RAW.len() {
            let b = make(i);
            for axis in [b.tangent, b.bitangent, b.normal] {
                assert!((dot(axis, axis) - 1.0).abs() < 1e-6, "unit axis");
            }
            assert!(dot(b.tangent, b.bitangent).abs() < 1e-6, "T.B");
            assert!(dot(b.tangent, b.normal).abs() < 1e-6, "T.N");
            assert!(dot(b.bitangent, b.normal).abs() < 1e-6, "B.N");
        }
    }

    /// Transforming to object space then back recovers the original normal (the
    /// primary anti-fake oracle: transpose is the exact orthonormal inverse).
    #[test]
    fn transform_round_trips() {
        let normals = [
            unit([0.0, 0.0, 1.0]),
            unit([0.3, -0.2, 0.9]),
            unit([-0.5, 0.4, 0.76]),
            unit([0.1, 0.7, 0.7]),
        ];
        for i in 0..RAW.len() {
            let b = make(i);
            for &n in &normals {
                let obj = tangent_to_object(&b, n);
                let back = object_to_tangent(&b, obj);
                for k in 0..3 {
                    assert!(
                        (back[k] - n[k]).abs() < 1e-6,
                        "round-trip {back:?} vs {n:?}"
                    );
                }
            }
        }
    }

    /// Tangent-space `+Z` maps exactly to the surface normal axis.
    #[test]
    fn tangent_z_maps_to_normal() {
        for i in 0..RAW.len() {
            let b = make(i);
            let got = tangent_to_object(&b, [0.0, 0.0, 1.0]);
            for k in 0..3 {
                assert!((got[k] - b.normal[k]).abs() < 1e-6, "+Z -> N");
            }
        }
    }

    /// The identity basis gives the identity transform.
    #[test]
    fn identity_basis_is_identity() {
        let b = orthonormalize_basis([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], 1.0);
        assert_eq!(b.tangent, [1.0, 0.0, 0.0]);
        assert_eq!(b.bitangent, [0.0, 1.0, 0.0]);
        assert_eq!(b.normal, [0.0, 0.0, 1.0]);
        let n = [0.3, -0.4, 0.866_025_4];
        assert_eq!(tangent_to_object(&b, n), n);
    }

    /// The transform preserves vector length (it is a pure rotation / reflection).
    #[test]
    fn transform_preserves_length() {
        let b = make(1);
        let v = [0.6, -0.3, 0.74];
        let out = tangent_to_object(&b, v);
        let li = dot(v, v);
        let lo = dot(out, out);
        assert!((li - lo).abs() < 1e-6, "length preserved {li} vs {lo}");
    }

    /// The signed basis determinant equals the requested handedness: the
    /// bitangent follows the right/left-hand rule as stored in the tangent `w`.
    #[test]
    fn handedness_matches_sign() {
        for i in 0..RAW.len() {
            let (_, _, h) = RAW[i];
            let b = make(i);
            let signed = dot(cross(b.tangent, b.bitangent), b.normal);
            let want = if h < 0.0 { -1.0 } else { 1.0 };
            assert!(
                (signed - want).abs() < 1e-6,
                "handedness {signed} vs {want}"
            );
        }
    }

    /// Gram-Schmidt removes the normal component even from a tangent that leans
    /// hard into the normal.
    #[test]
    fn gram_schmidt_removes_normal_component() {
        let n = unit([0.1, 0.2, 0.97]);
        // A tangent with a large component along N.
        let t = [n[0] * 2.0 + 1.0, n[1] * 2.0, n[2] * 2.0];
        let b = orthonormalize_basis(t, n, 1.0);
        assert!(dot(b.tangent, b.normal).abs() < 1e-6, "T orthogonal to N");
        assert!((dot(b.tangent, b.tangent) - 1.0).abs() < 1e-6, "T unit");
    }
}
