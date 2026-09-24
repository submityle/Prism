//! Tangent-frame reconstruction for normal mapping and anisotropic shading.
//!
//! The visibility-buffer resolve needs a per-pixel orthonormal tangent basis
//! `(T, B, N)` to (a) rotate tangent-space normal-map samples into world space
//! and (b) orient the anisotropic GGX lobe.  Three sources are tried in order,
//! matching Unreal's derived-tangent policy:
//!
//! 1. **Authored tangents** interpolated from the vertex table (produced by a
//!    MikkTSpace pass at import time), re-orthonormalized against the shaded
//!    normal so the basis stays orthogonal after interpolation.
//! 2. **Analytic tangents** derived from the triangle's position/UV gradients
//!    (Lengyel's method) when the mesh carries no authored tangent attribute.
//! 3. **A canonical orthonormal basis** (Duff, Cigolle, Jimenez, Wyman & Ku
//!    2017, "Building an Orthonormal Basis, Revisited") when the UVs are also
//!    degenerate; this path is flagged so downstream passes can react.
//!
//! The arithmetic is deliberately restricted to `+ - * /` and a single
//! `sqrt` (through [`normalize_or`]) so the CPU golden reference stays
//! byte-for-byte in step with the `tangent.wesl` GPU twin.

use crate::vecmath::{add, cross, dot, mul_scalar, normalize_or, sub};

/// Squared length below which a tangent candidate is treated as degenerate.
const DEGENERATE_EPSILON_SQ: f32 = 1.0e-12;

/// Determinant magnitude below which the UV gradient is treated as degenerate.
const UV_DETERMINANT_EPSILON: f32 = 1.0e-8;

/// An orthonormal tangent basis paired with the surface normal.
///
/// `tangent` and `bitangent` are unit length and orthogonal to the normal the
/// basis was built against; `bitangent` already carries the handedness sign so
/// callers never re-derive it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TangentBasis {
    pub tangent: [f32; 3],
    pub bitangent: [f32; 3],
    /// `true` when neither authored nor analytic tangents were usable and the
    /// canonical fallback basis was synthesized from the normal alone.
    pub degenerate: bool,
}

/// Branchless orthonormal basis from a **unit** normal (Duff et al. 2017).
///
/// The returned tangent/bitangent are exactly orthogonal to `normal` for any
/// unit input and never hit the `z == -1` singularity of the naive frame.
pub fn orthonormal_basis(normal: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    // `copysign(1, n.z)` expressed as a branch so CPU and WESL agree bit-wise.
    let sign = if normal[2] >= 0.0 { 1.0 } else { -1.0 };
    let a = -1.0 / (sign + normal[2]);
    let b = normal[0] * normal[1] * a;
    let tangent = [
        1.0 + sign * normal[0] * normal[0] * a,
        sign * b,
        -sign * normal[0],
    ];
    let bitangent = [b, sign + normal[1] * normal[1] * a, -normal[1]];
    (tangent, bitangent)
}

/// Builds the per-pixel tangent basis from the best available source.
///
/// `authored` carries the interpolated authored tangent (`xyz`) and handedness
/// (`w`); pass `None` when the mesh supplied no tangent attribute.  `positions`
/// and `uvs` are the triangle's three corners, used for the analytic fallback.
pub fn resolve_tangent_basis(
    normal: [f32; 3],
    authored: Option<[f32; 4]>,
    positions: [[f32; 3]; 3],
    uvs: [[f32; 2]; 3],
) -> TangentBasis {
    // 1. Authored tangent, re-orthonormalized against the shaded normal.
    if let Some(tangent) = authored {
        let raw = [tangent[0], tangent[1], tangent[2]];
        let projected = sub(raw, mul_scalar(normal, dot(normal, raw)));
        if dot(projected, projected) > DEGENERATE_EPSILON_SQ {
            let unit = normalize_or(projected, normal);
            let handedness = if tangent[3] < 0.0 { -1.0 } else { 1.0 };
            let bitangent = mul_scalar(cross(normal, unit), handedness);
            return TangentBasis { tangent: unit, bitangent, degenerate: false };
        }
    }
    // 2. Analytic tangent from the triangle's position/UV gradients.
    if let Some(basis) = analytic_tangent_basis(normal, positions, uvs) {
        return basis;
    }
    // 3. Canonical fallback from the normal alone.
    let (tangent, bitangent) = orthonormal_basis(normal);
    TangentBasis { tangent, bitangent, degenerate: true }
}

/// Derives a tangent basis from the triangle's edge and UV gradients.
///
/// Returns `None` when the UV parameterization is degenerate (zero-area in UV
/// space) or the resulting tangent collapses onto the normal.
fn analytic_tangent_basis(
    normal: [f32; 3],
    positions: [[f32; 3]; 3],
    uvs: [[f32; 2]; 3],
) -> Option<TangentBasis> {
    let edge1 = sub(positions[1], positions[0]);
    let edge2 = sub(positions[2], positions[0]);
    let du1 = uvs[1][0] - uvs[0][0];
    let dv1 = uvs[1][1] - uvs[0][1];
    let du2 = uvs[2][0] - uvs[0][0];
    let dv2 = uvs[2][1] - uvs[0][1];
    let determinant = du1 * dv2 - du2 * dv1;
    if !determinant.is_finite() || determinant.abs() <= UV_DETERMINANT_EPSILON {
        return None;
    }
    let inverse = 1.0 / determinant;
    let raw_tangent = mul_scalar(sub(mul_scalar(edge1, dv2), mul_scalar(edge2, dv1)), inverse);
    let raw_bitangent = mul_scalar(sub(mul_scalar(edge2, du1), mul_scalar(edge1, du2)), inverse);
    let projected = sub(raw_tangent, mul_scalar(normal, dot(normal, raw_tangent)));
    if dot(projected, projected) <= DEGENERATE_EPSILON_SQ {
        return None;
    }
    let unit = normalize_or(projected, normal);
    let handedness = if dot(cross(normal, unit), raw_bitangent) < 0.0 {
        -1.0
    } else {
        1.0
    };
    let bitangent = mul_scalar(cross(normal, unit), handedness);
    Some(TangentBasis { tangent: unit, bitangent, degenerate: false })
}

/// Rotates a tangent-space normal-map sample (`x` along T, `y` along B, `z`
/// along N) into world space, renormalizing the result.
pub fn apply_tangent_space_normal(
    basis: TangentBasis,
    normal: [f32; 3],
    tangent_space_normal: [f32; 3],
) -> [f32; 3] {
    let world = add(
        add(
            mul_scalar(basis.tangent, tangent_space_normal[0]),
            mul_scalar(basis.bitangent, tangent_space_normal[1]),
        ),
        mul_scalar(normal, tangent_space_normal[2]),
    );
    normalize_or(world, normal)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: [f32; 3], b: [f32; 3]) -> bool {
        (0..3).all(|i| (a[i] - b[i]).abs() < 1.0e-5)
    }

    #[test]
    fn analytic_basis_matches_uv_aligned_triangle() {
        // A triangle in the XY plane whose UVs align with world X/Y should
        // yield tangent = +X, bitangent = +Y, normal = +Z.
        let basis = resolve_tangent_basis(
            [0.0, 0.0, 1.0],
            None,
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        );
        assert!(!basis.degenerate);
        assert!(approx(basis.tangent, [1.0, 0.0, 0.0]), "{:?}", basis.tangent);
        assert!(approx(basis.bitangent, [0.0, 1.0, 0.0]), "{:?}", basis.bitangent);
    }

    #[test]
    fn authored_tangent_is_reorthonormalized_against_normal() {
        // Authored tangent leans out of the tangent plane; the basis must
        // project it back so it stays orthogonal to the normal and unit-length.
        let normal = [0.0, 0.0, 1.0];
        let basis = resolve_tangent_basis(
            normal,
            Some([1.0, 0.0, 0.5, 1.0]),
            [[0.0; 3]; 3],
            [[0.0; 2]; 3],
        );
        assert!(!basis.degenerate);
        assert!(dot(basis.tangent, normal).abs() < 1.0e-6);
        assert!((dot(basis.tangent, basis.tangent) - 1.0).abs() < 1.0e-5);
        // Right-handed: T x B should point along +N.
        assert!(approx(cross(basis.tangent, basis.bitangent), normal));
    }

    #[test]
    fn authored_handedness_flips_the_bitangent() {
        let normal = [0.0, 0.0, 1.0];
        let right = resolve_tangent_basis(normal, Some([1.0, 0.0, 0.0, 1.0]), [[0.0; 3]; 3], [[0.0; 2]; 3]);
        let left = resolve_tangent_basis(normal, Some([1.0, 0.0, 0.0, -1.0]), [[0.0; 3]; 3], [[0.0; 2]; 3]);
        assert!(approx(right.tangent, left.tangent));
        assert!(approx(right.bitangent, mul_scalar(left.bitangent, -1.0)));
    }

    #[test]
    fn degenerate_uv_falls_back_to_canonical_basis() {
        // Collapsed UVs (all identical) leave no analytic gradient; the basis
        // must synthesize an orthonormal frame and flag the fallback.
        let normal = normalize_or([0.3, 0.4, 0.866], [0.0, 0.0, 1.0]);
        let basis = resolve_tangent_basis(
            normal,
            None,
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
        );
        assert!(basis.degenerate);
        assert!(dot(basis.tangent, normal).abs() < 1.0e-5);
        assert!(dot(basis.bitangent, normal).abs() < 1.0e-5);
        assert!((dot(basis.tangent, basis.tangent) - 1.0).abs() < 1.0e-4);
    }

    #[test]
    fn orthonormal_basis_is_orthogonal_for_backfacing_normals() {
        // The Duff construction must stay finite and orthogonal near z = -1.
        for normal in [[0.0, 0.0, -1.0], [0.0, 0.0, 1.0], [0.1, -0.2, -0.97]] {
            let n = normalize_or(normal, [0.0, 0.0, 1.0]);
            let (t, b) = orthonormal_basis(n);
            assert!(dot(t, n).abs() < 1.0e-5, "t·n {n:?}");
            assert!(dot(b, n).abs() < 1.0e-5, "b·n {n:?}");
            assert!(dot(t, b).abs() < 1.0e-5, "t·b {n:?}");
        }
    }

    #[test]
    fn apply_tangent_space_normal_rotates_into_world_space() {
        let basis = TangentBasis {
            tangent: [1.0, 0.0, 0.0],
            bitangent: [0.0, 1.0, 0.0],
            degenerate: false,
        };
        let normal = [0.0, 0.0, 1.0];
        // A flat tangent-space normal (0,0,1) must return the surface normal.
        assert!(approx(apply_tangent_space_normal(basis, normal, [0.0, 0.0, 1.0]), normal));
        // A fully tilted sample (1,0,0) must return the tangent direction.
        assert!(approx(
            apply_tangent_space_normal(basis, normal, [1.0, 0.0, 0.0]),
            [1.0, 0.0, 0.0]
        ));
    }
}
