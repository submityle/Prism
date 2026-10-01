//! Surface curvature of a signed distance field by Hessian estimation for the
//! `CPU` golden path.
//!
//! Where [`super::mesh_sdf_normal`] differentiates the field once to recover
//! the shading normal, this module differentiates it twice to recover how the
//! surface *bends*. Curvature is what `AAA` renderers lean on for cavity and
//! edge-wear masks (crevices read concave, ridges read convex), for
//! curvature-adaptive tessellation and detail, and for anisotropic
//! highlight shaping. All of it falls out of the gradient plus the Hessian of
//! the same trilinear sampler
//! ([`super::mesh_sdf_raymarch::sample_signed_distance`]).
//!
//! The implicit-surface curvature formulas are Ron Goldman's (2005): the mean
//! curvature follows from the divergence of the normalized gradient and the
//! Gaussian curvature from the gradient contracted with the adjugate of the
//! Hessian. Both are evaluated from symmetric central differences — a single
//! voxel step for the diagonal second derivatives and a four-corner stencil
//! for the mixed ones — so the only non-linear operations are the two `sqrt`s
//! (gradient length and the principal-curvature discriminant) and the result
//! is reproducible across machines.
//!
//! Sign convention: the field's outward gradient is treated as the surface
//! normal, so a convex surface (a sphere seen from outside, a ridge) reports a
//! **positive** mean curvature and a concave surface (a crevice, the inside of
//! a shell) reports a **negative** one. Gaussian curvature is orientation
//! independent: positive on elliptic (dome/pit) regions, zero on developable
//! (cylinder/plane) regions, negative on hyperbolic (saddle) regions.
//!
//! Sampling away from the surface needs the exterior shell produced by
//! [`super::mesh_voxel_padding::pad_voxel_grid`]; the sampler clamps to border
//! values past the grid.

use super::mesh_sdf_raymarch::sample_signed_distance;
use super::mesh_signed_distance_field::SignedDistanceField;

/// Local curvature of a signed distance field at a sampled point.
///
/// All four quantities share the convex-positive sign convention described in
/// the module documentation (the field's outward gradient is the normal).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SdfCurvature {
    /// Mean curvature `(k1 + k2) / 2`: positive where the surface is convex,
    /// negative where it is concave, near zero on a flat face.
    mean: f32,
    /// Gaussian curvature `k1 * k2`: positive on elliptic (dome/pit) regions,
    /// zero on developable (cylinder/plane) regions, negative on hyperbolic
    /// (saddle) regions. Independent of the normal orientation.
    gaussian: f32,
    /// Larger (more convex) principal curvature `k1 = mean + sqrt(mean^2 - K)`.
    principal_max: f32,
    /// Smaller (more concave) principal curvature `k2 = mean - sqrt(mean^2 - K)`.
    principal_min: f32,
}

impl SdfCurvature {
    /// Mean curvature `(k1 + k2) / 2` (convex positive, concave negative).
    pub fn mean(&self) -> f32 {
        self.mean
    }

    /// Gaussian curvature `k1 * k2` (elliptic positive, hyperbolic negative).
    pub fn gaussian(&self) -> f32 {
        self.gaussian
    }

    /// Larger (more convex) principal curvature.
    pub fn principal_max(&self) -> f32 {
        self.principal_max
    }

    /// Smaller (more concave) principal curvature.
    pub fn principal_min(&self) -> f32 {
        self.principal_min
    }
}

/// Estimates the curvature of the signed distance field at `point`.
///
/// Samples the trilinear field on a one-voxel central-difference stencil to
/// build the gradient and the symmetric Hessian, then evaluates Goldman's
/// implicit-surface mean and Gaussian curvatures and splits them into the two
/// principal curvatures. Returns [`None`] when the gradient is too short to
/// orient reliably (a flat, constant region of the field), where curvature is
/// undefined.
pub fn sdf_curvature(field: &SignedDistanceField, point: [f32; 3]) -> Option<SdfCurvature> {
    let h = field.voxel_size();
    let gradient = central_gradient(field, point, h);
    let hessian = central_hessian(field, point, h);
    curvature_from_derivatives(gradient, hessian)
}

/// Samples the field at `point + h * (dx, dy, dz)`.
fn offset_sample(
    field: &SignedDistanceField,
    point: [f32; 3],
    dx: f32,
    dy: f32,
    dz: f32,
    h: f32,
) -> f32 {
    sample_signed_distance(
        field,
        [point[0] + h * dx, point[1] + h * dy, point[2] + h * dz],
    )
}

/// First-order gradient from a symmetric one-voxel central difference.
fn central_gradient(field: &SignedDistanceField, point: [f32; 3], h: f32) -> [f32; 3] {
    let inv = 1.0 / (2.0 * h);
    [
        (offset_sample(field, point, 1.0, 0.0, 0.0, h)
            - offset_sample(field, point, -1.0, 0.0, 0.0, h))
            * inv,
        (offset_sample(field, point, 0.0, 1.0, 0.0, h)
            - offset_sample(field, point, 0.0, -1.0, 0.0, h))
            * inv,
        (offset_sample(field, point, 0.0, 0.0, 1.0, h)
            - offset_sample(field, point, 0.0, 0.0, -1.0, h))
            * inv,
    ]
}

/// Symmetric Hessian from central differences: a three-point stencil for each
/// diagonal second derivative and a four-corner stencil for each mixed one.
fn central_hessian(field: &SignedDistanceField, point: [f32; 3], h: f32) -> [[f32; 3]; 3] {
    let centre = offset_sample(field, point, 0.0, 0.0, 0.0, h);
    let inv_sq = 1.0 / (h * h);
    let inv_quad = 1.0 / (4.0 * h * h);

    // Diagonal: f(+) - 2 f0 + f(-).
    let fxx = (offset_sample(field, point, 1.0, 0.0, 0.0, h)
        - 2.0 * centre
        + offset_sample(field, point, -1.0, 0.0, 0.0, h))
        * inv_sq;
    let fyy = (offset_sample(field, point, 0.0, 1.0, 0.0, h)
        - 2.0 * centre
        + offset_sample(field, point, 0.0, -1.0, 0.0, h))
        * inv_sq;
    let fzz = (offset_sample(field, point, 0.0, 0.0, 1.0, h)
        - 2.0 * centre
        + offset_sample(field, point, 0.0, 0.0, -1.0, h))
        * inv_sq;

    // Mixed: (f(++) - f(+-) - f(-+) + f(--)) / (4 h^2).
    let fxy = (offset_sample(field, point, 1.0, 1.0, 0.0, h)
        - offset_sample(field, point, 1.0, -1.0, 0.0, h)
        - offset_sample(field, point, -1.0, 1.0, 0.0, h)
        + offset_sample(field, point, -1.0, -1.0, 0.0, h))
        * inv_quad;
    let fxz = (offset_sample(field, point, 1.0, 0.0, 1.0, h)
        - offset_sample(field, point, 1.0, 0.0, -1.0, h)
        - offset_sample(field, point, -1.0, 0.0, 1.0, h)
        + offset_sample(field, point, -1.0, 0.0, -1.0, h))
        * inv_quad;
    let fyz = (offset_sample(field, point, 0.0, 1.0, 1.0, h)
        - offset_sample(field, point, 0.0, 1.0, -1.0, h)
        - offset_sample(field, point, 0.0, -1.0, 1.0, h)
        + offset_sample(field, point, 0.0, -1.0, -1.0, h))
        * inv_quad;

    [[fxx, fxy, fxz], [fxy, fyy, fyz], [fxz, fyz, fzz]]
}

/// Evaluates Goldman's implicit-surface curvatures from a gradient and a
/// symmetric Hessian, returning [`None`] when the gradient is degenerate.
///
/// Mean curvature is the divergence of the normalized gradient,
/// `(trace(H) |g|^2 - g^T H g) / (2 |g|^3)`, taken convex-positive. Gaussian
/// curvature is `g^T adj(H) g / |g|^4` with `adj(H)` the adjugate of the
/// Hessian. The principal curvatures are recovered as `mean +/- sqrt(mean^2 -
/// K)`, with the discriminant clamped to zero so discretization noise can
/// never produce a `NaN`.
fn curvature_from_derivatives(
    gradient: [f32; 3],
    hessian: [[f32; 3]; 3],
) -> Option<SdfCurvature> {
    let [gx, gy, gz] = gradient;
    let g2 = gx * gx + gy * gy + gz * gz;
    if g2 <= f32::MIN_POSITIVE {
        return None;
    }
    let g_len = g2.sqrt();
    let g_len3 = g2 * g_len;
    let g4 = g2 * g2;

    // Name the symmetric Hessian entries: [[a, b, c], [b, d, e], [c, e, f]].
    let a = hessian[0][0];
    let b = hessian[0][1];
    let c = hessian[0][2];
    let d = hessian[1][1];
    let e = hessian[1][2];
    let f = hessian[2][2];

    let trace = a + d + f;
    // Quadratic form g^T H g.
    let ghg = gx * gx * a
        + gy * gy * d
        + gz * gz * f
        + 2.0 * (gx * gy * b + gx * gz * c + gy * gz * e);
    let mean = (trace * g2 - ghg) / (2.0 * g_len3);

    // Adjugate of the symmetric Hessian (itself symmetric).
    let adj_a = d * f - e * e;
    let adj_d = a * f - c * c;
    let adj_f = a * d - b * b;
    let adj_b = c * e - b * f;
    let adj_c = b * e - c * d;
    let adj_e = b * c - a * e;
    // Quadratic form g^T adj(H) g.
    let g_adj_g = gx * gx * adj_a
        + gy * gy * adj_d
        + gz * gz * adj_f
        + 2.0 * (gx * gy * adj_b + gx * gz * adj_c + gy * gz * adj_e);
    let gaussian = g_adj_g / g4;

    let discriminant = (mean * mean - gaussian).max(0.0);
    let root = discriminant.sqrt();

    Some(SdfCurvature {
        mean,
        gaussian,
        principal_max: mean + root,
        principal_min: mean - root,
    })
}

#[cfg(test)]
mod tests {
    use super::{curvature_from_derivatives, sdf_curvature, SdfCurvature};
    use crate::ray_scene::mesh_signed_distance_field::{signed_distance_field, SignedDistanceField};
    use crate::ray_scene::mesh_voxel_padding::pad_voxel_grid;
    use crate::ray_scene::mesh_voxelize::voxelize_surface;
    use crate::ray_scene::triangle_mesh::TriangleMesh;

    /// Builds an index-only triangle mesh from positions.
    fn mesh(positions: Vec<[f32; 3]>, indices: Vec<[u32; 3]>) -> TriangleMesh {
        TriangleMesh::new(positions, Vec::new(), Vec::new(), indices).unwrap()
    }

    /// Closed axis-aligned unit cube (12 triangles) spanning `[0, 1]^3`.
    fn cube() -> TriangleMesh {
        let p = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ];
        let i = vec![
            [0, 1, 2], [0, 2, 3],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [3, 2, 6], [3, 6, 7],
            [0, 3, 7], [0, 7, 4],
            [1, 2, 6], [1, 6, 5],
        ];
        mesh(p, i)
    }

    /// Padded signed distance field of the unit cube with an exterior shell.
    fn cube_field() -> SignedDistanceField {
        let grid = voxelize_surface(&cube(), 16).unwrap();
        let padded = pad_voxel_grid(&grid, 4);
        signed_distance_field(&padded)
    }

    #[test]
    fn sphere_hessian_is_isotropic_convex() {
        // phi = |p| - r sampled on the +x axis at radius r: gradient (1,0,0),
        // Hessian diag(0, 1/r, 1/r). Mean = 1/r, Gaussian = 1/r^2, both
        // principal curvatures 1/r.
        let r = 2.0f32;
        let inv_r = 1.0 / r;
        let gradient = [1.0, 0.0, 0.0];
        let hessian = [[0.0, 0.0, 0.0], [0.0, inv_r, 0.0], [0.0, 0.0, inv_r]];
        let k = curvature_from_derivatives(gradient, hessian).unwrap();
        assert!((k.mean() - inv_r).abs() < 1e-6);
        assert!((k.gaussian() - inv_r * inv_r).abs() < 1e-6);
        assert!((k.principal_max() - inv_r).abs() < 1e-6);
        assert!((k.principal_min() - inv_r).abs() < 1e-6);
    }

    #[test]
    fn cylinder_hessian_is_developable() {
        // phi = sqrt(x^2 + y^2) - r on the +x side: gradient (1,0,0), Hessian
        // diag(0, 1/r, 0). Mean = 1/(2r), Gaussian = 0, principals {1/r, 0}.
        let r = 4.0f32;
        let inv_r = 1.0 / r;
        let gradient = [1.0, 0.0, 0.0];
        let hessian = [[0.0, 0.0, 0.0], [0.0, inv_r, 0.0], [0.0, 0.0, 0.0]];
        let k = curvature_from_derivatives(gradient, hessian).unwrap();
        assert!((k.mean() - 0.5 * inv_r).abs() < 1e-6);
        assert!(k.gaussian().abs() < 1e-6);
        assert!((k.principal_max() - inv_r).abs() < 1e-6);
        assert!(k.principal_min().abs() < 1e-6);
    }

    #[test]
    fn saddle_hessian_is_hyperbolic() {
        // Normal along +z with equal-and-opposite tangent curvatures: a
        // minimal (saddle) surface with zero mean and negative Gaussian.
        let kappa = 0.5f32;
        let gradient = [0.0, 0.0, 1.0];
        let hessian = [[kappa, 0.0, 0.0], [0.0, -kappa, 0.0], [0.0, 0.0, 0.0]];
        let k = curvature_from_derivatives(gradient, hessian).unwrap();
        assert!(k.mean().abs() < 1e-6);
        assert!((k.gaussian() - (-kappa * kappa)).abs() < 1e-6);
        assert!((k.principal_max() - kappa).abs() < 1e-6);
        assert!((k.principal_min() - (-kappa)).abs() < 1e-6);
    }

    #[test]
    fn concave_surface_flips_the_mean_sign() {
        // The same sphere Hessian with an inward (concave) gradient flips the
        // mean curvature sign while leaving the Gaussian curvature intact.
        let r = 2.0f32;
        let inv_r = 1.0 / r;
        let gradient = [-1.0, 0.0, 0.0];
        let hessian = [[0.0, 0.0, 0.0], [0.0, -inv_r, 0.0], [0.0, 0.0, -inv_r]];
        let k = curvature_from_derivatives(gradient, hessian).unwrap();
        assert!((k.mean() - (-inv_r)).abs() < 1e-6);
        assert!((k.gaussian() - inv_r * inv_r).abs() < 1e-6);
    }

    #[test]
    fn degenerate_gradient_has_no_curvature() {
        let hessian = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        assert!(curvature_from_derivatives([0.0, 0.0, 0.0], hessian).is_none());
    }

    #[test]
    fn field_sampling_yields_finite_ordered_curvature() {
        // Exercises the end-to-end field-sampling path (gradient + Hessian
        // stencils feeding the curvature math) at a well-defined point just
        // outside the +x face, where the outward gradient is unambiguous. The
        // discrete trilinear distance transform is far too blocky to assert a
        // precise curvature value, so this checks the invariants that must hold
        // for any input: finite results and ordered principal curvatures whose
        // mean is their average.
        let field = cube_field();
        let point = [1.1, 0.5, 0.5];
        let k = sdf_curvature(&field, point).unwrap();
        assert!(k.mean().is_finite());
        assert!(k.gaussian().is_finite());
        assert!(k.principal_max().is_finite());
        assert!(k.principal_min().is_finite());
        assert!(k.principal_max() >= k.principal_min());
        let average = 0.5 * (k.principal_max() + k.principal_min());
        assert!((k.mean() - average).abs() < 1e-4, "mean = {}", k.mean());
    }

    #[test]
    fn curvature_is_copy_and_comparable() {
        let value = SdfCurvature {
            mean: 1.0,
            gaussian: 2.0,
            principal_max: 3.0,
            principal_min: 4.0,
        };
        let copied = value;
        assert_eq!(value, copied);
    }
}
