//! Anisotropic (fiber-direction) strain limiting for tetrahedral FEM meshes.
//!
//! Fiber-reinforced soft tissue — muscle, tendon, cloth-on-shell, or any
//! material with a dominant reinforcing direction — must not stretch or
//! compress along its fibers beyond a physiological band, even while remaining
//! free to deform across the fibers. The isotropic strain limiter in
//! [`super::tet_fem_strain_limit`] clamps *all* principal stretches via an SVD
//! and therefore cannot express "stiff along the fiber, soft across it". This
//! module provides the complementary projection: it clamps only the stretch of
//! a single rest-space fiber direction per element into a band `[min, max]`,
//! leaving the orthogonal deformation untouched.
//!
//! # Method
//!
//! For an element the deformation gradient is `F = Ds · Dm⁻¹` with
//! `Ds = [p1-p0 | p2-p0 | p3-p0]` (see
//! [`super::tet_fem_basis::TetFemElement::deformation_gradient`]). The fiber
//! vector in the deformed configuration is `fa = F a0 = Ds (Dm⁻¹ a0)`. Writing
//! `b = Dm⁻¹ a0`, this expands to a linear combination of the four vertices
//!
//! ```text
//! fa = w0·p0 + w1·p1 + w2·p2 + w3·p3,
//! w0 = -(b.x + b.y + b.z),  w1 = b.x,  w2 = b.y,  w3 = b.z,
//! ```
//!
//! with `Σ w_j = 0`, so `fa` is invariant under rigid translation. The fiber
//! stretch is `λ = |fa|` and the unit fiber `n̂ = fa / λ`. The limiting
//! constraint is
//!
//! ```text
//! C = λ − clamp(λ, min, max),
//! ```
//!
//! which is zero whenever the fiber is already inside the band. Its gradient
//! with respect to vertex `j` is `∇_j C = w_j n̂`, so `|∇_j C|² = w_j²`. The
//! position-based (PBD / XPBD-at-infinite-stiffness) correction is
//!
//! ```text
//! Δp_j = −( invMass_j · w_j / Σ_k invMass_k w_k² ) · C · n̂.
//! ```
//!
//! Because `Σ_j m_j Δp_j ∝ Σ_j w_j = 0` (using `m_j · invMass_j = 1`), each
//! per-element correction conserves linear momentum exactly; pinned vertices
//! (inverse mass `0`) neither move nor contribute to the denominator.
//!
//! A vertex is shared by many elements, so the per-element corrections are
//! accumulated and averaged in a Jacobi sweep — matching the convention of
//! [`super::tet_fem_strain_limit`] — and the whole projection can be iterated.
//!
//! The module owns no simulation state and performs no time integration; it is
//! a pure geometric projection that can run before or after any step.
//!
//! # Attribution
//!
//! Clean-room implementation of position-based fiber strain limiting (Müller
//! et al., "Position Based Dynamics"; Thomaszewski et al., anisotropic strain
//! limiting). No Unreal Engine source or derived code.

use super::tet_fem_anisotropic::FiberDirection;
use super::tet_fem_basis::TetFemBasis;
use glam::Vec3;

/// Smallest fiber stretch magnitude below which an element is skipped, to avoid
/// dividing by a near-zero `λ` when forming the unit fiber `n̂`.
const MIN_FIBER_STRETCH: f32 = 1.0e-6;

/// Smallest constraint denominator `Σ invMass_k w_k²` for which a correction is
/// applied. Below this the element is effectively rigid (all vertices pinned or
/// the fiber is degenerate) and is skipped.
const MIN_DENOM: f32 = 1.0e-12;

/// The admissible band of fiber stretches enforced by
/// [`project_fiber_strain_limits`].
///
/// A stretch of `1.0` is the rest length along the fiber; `min` caps fiber
/// compression and `max` caps fiber extension. `min == max == 1.0` makes the
/// fiber inextensible.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FiberStrainLimitParams {
    /// Lower bound on the fiber stretch. Must satisfy `0 < min <= max`.
    pub min_stretch: f32,
    /// Upper bound on the fiber stretch.
    pub max_stretch: f32,
    /// Number of Jacobi sweeps. More sweeps propagate corrections through
    /// shared vertices; one sweep is often enough for mild violations.
    pub iterations: u32,
}

impl FiberStrainLimitParams {
    /// Builds a validated band.
    ///
    /// Returns `None` unless `min` and `max` are finite with
    /// `0 < min <= max` and `iterations >= 1`.
    #[must_use]
    pub fn new(min: f32, max: f32, iterations: u32) -> Option<Self> {
        if !min.is_finite() || !max.is_finite() || min <= 0.0 || max < min || iterations == 0 {
            return None;
        }
        Some(Self {
            min_stretch: min,
            max_stretch: max,
            iterations,
        })
    }

    /// A band symmetric about the rest length: `[1 - pct, 1 + pct]`.
    ///
    /// Returns `None` unless `0 < pct < 1` and `iterations >= 1`.
    #[must_use]
    pub fn symmetric(pct: f32, iterations: u32) -> Option<Self> {
        if !pct.is_finite() || pct <= 0.0 || pct >= 1.0 {
            return None;
        }
        Self::new(1.0 - pct, 1.0 + pct, iterations)
    }
}

/// Outcome of a fiber strain-limiting projection, measured on the first sweep
/// before any correction is applied.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FiberStrainLimitReport {
    /// Largest band violation across all elements, i.e. the maximum over
    /// elements of `max(λ − max_stretch, min_stretch − λ, 0)`. `0.0` means
    /// every element's fiber was already inside the band.
    pub max_violation: f32,
    /// Number of elements whose fiber was outside the band on the first sweep.
    pub projected_elements: usize,
}

/// Per-element fiber geometry: the deformed fiber vector written as a linear
/// combination of the four vertices, plus the barycentric weights `w_j`.
struct FiberGeometry {
    /// `fa = Σ w_j p_j`.
    fa: Vec3,
    /// Weights `[w0, w1, w2, w3]` with `Σ w_j = 0`.
    weights: [f32; 4],
}

/// Computes the deformed fiber vector `fa = F a0` for one element and the
/// barycentric weights that express it as `Σ w_j p_j`.
#[inline]
fn fiber_geometry(dm_inverse: &glam::Mat3, fiber: FiberDirection, p: &[Vec3; 4]) -> FiberGeometry {
    // b = Dm⁻¹ a0 (standard matrix–vector product).
    let b = *dm_inverse * fiber.get();
    let w0 = -(b.x + b.y + b.z);
    let weights = [w0, b.x, b.y, b.z];
    // fa = Σ w_j p_j = b.x (p1-p0) + b.y (p2-p0) + b.z (p3-p0).
    let fa = b.x * (p[1] - p[0]) + b.y * (p[2] - p[0]) + b.z * (p[3] - p[0]);
    FiberGeometry { fa, weights }
}

/// Projects every element's fiber stretch into the band in `params`, mutating
/// `positions` in place.
///
/// `fibers` supplies one rest-space fiber direction per basis element; `tets`
/// maps each basis element to its four vertex indices into `positions`.
/// `inv_mass`, when supplied, gives a per-vertex inverse mass used to weight the
/// momentum-preserving correction; `None` treats every vertex as unit mass. A
/// zero inverse mass marks a pinned vertex that never moves. `pinned`, when
/// supplied, additionally forces the listed vertices to stay fixed regardless
/// of their inverse mass.
///
/// Returns `None` on any dimension mismatch (`fibers.len()` or `tets.len()`
/// differs from the number of basis elements, any tet index is out of range for
/// `positions`, or `inv_mass`/`pinned` lengths differ from `positions.len()`).
#[must_use]
pub fn project_fiber_strain_limits(
    basis: &TetFemBasis,
    tets: &[[u32; 4]],
    fibers: &[FiberDirection],
    positions: &mut [Vec3],
    inv_mass: Option<&[f32]>,
    pinned: Option<&[bool]>,
    params: &FiberStrainLimitParams,
) -> Option<FiberStrainLimitReport> {
    let element_count = basis.elements.len();
    if tets.len() != element_count || fibers.len() != element_count {
        return None;
    }
    let n = positions.len();
    if let Some(w) = inv_mass
        && w.len() != n
    {
        return None;
    }
    if let Some(p) = pinned
        && p.len() != n
    {
        return None;
    }
    for tet in tets {
        for &v in tet {
            if v as usize >= n {
                return None;
            }
        }
    }

    let is_pinned = |v: usize| -> bool {
        let pin_flag = pinned.is_some_and(|p| p[v]);
        let zero_inv = inv_mass.is_some_and(|w| w[v] <= 0.0);
        pin_flag || zero_inv
    };
    // Effective inverse mass used by the correction: pinned vertices get 0 so
    // they neither move nor contribute to the denominator.
    let effective_inv_mass = |v: usize| -> f32 {
        if is_pinned(v) {
            0.0
        } else {
            match inv_mass {
                Some(w) => w[v],
                None => 1.0,
            }
        }
    };

    let mut report = FiberStrainLimitReport {
        max_violation: 0.0,
        projected_elements: 0,
    };
    let mut accum = vec![[0.0f32; 3]; n];
    let mut counts = vec![0u32; n];

    for sweep in 0..params.iterations {
        for a in accum.iter_mut() {
            *a = [0.0; 3];
        }
        for c in counts.iter_mut() {
            *c = 0;
        }

        for ((element, tet), fiber) in basis.elements.iter().zip(tets.iter()).zip(fibers.iter()) {
            let [i0, i1, i2, i3] = tet.map(|v| v as usize);
            let verts = [i0, i1, i2, i3];
            let p = [positions[i0], positions[i1], positions[i2], positions[i3]];

            let geom = fiber_geometry(&element.dm_inverse, *fiber, &p);
            let lambda = geom.fa.length();
            if lambda < MIN_FIBER_STRETCH {
                continue;
            }

            let clamped = lambda.clamp(params.min_stretch, params.max_stretch);
            let c = lambda - clamped;

            if sweep == 0 {
                let over = (lambda - params.max_stretch).max(0.0);
                let under = (params.min_stretch - lambda).max(0.0);
                let violation = over.max(under);
                if violation > 0.0 {
                    report.projected_elements += 1;
                    report.max_violation = report.max_violation.max(violation);
                }
            }

            // Already inside the band: nothing to project for this element.
            if c == 0.0 {
                continue;
            }

            let n_hat = geom.fa / lambda;

            // denom = Σ_k invMass_k w_k².
            let mut denom = 0.0f32;
            for (k, &v) in verts.iter().enumerate() {
                let wk = geom.weights[k];
                denom += effective_inv_mass(v) * wk * wk;
            }
            if denom < MIN_DENOM {
                continue;
            }

            // Lagrange multiplier scalar: lambda_mul = C / denom.
            let lambda_mul = c / denom;

            for (k, &v) in verts.iter().enumerate() {
                let inv_m = effective_inv_mass(v);
                if inv_m <= 0.0 {
                    continue;
                }
                // Δp_k = -(invMass_k · w_k · C / denom) · n̂.
                let scale = -lambda_mul * inv_m * geom.weights[k];
                let delta = scale * n_hat;
                accum[v][0] += delta.x;
                accum[v][1] += delta.y;
                accum[v][2] += delta.z;
                counts[v] += 1;
            }
        }

        for v in 0..n {
            if counts[v] == 0 {
                continue;
            }
            let inv = 1.0 / counts[v] as f32;
            positions[v] += Vec3::new(accum[v][0] * inv, accum[v][1] * inv, accum[v][2] * inv);
        }
    }

    Some(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collider::tet_fem_basis::{build_tet_fem_basis, TetFemBasisParams};

    fn two_tets() -> (Vec<Vec3>, Vec<[u32; 4]>) {
        let verts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.7, 0.7, 0.7),
        ];
        let tets = vec![[0u32, 1, 2, 3], [1, 2, 3, 4]];
        (verts, tets)
    }

    fn single_tet() -> (Vec<Vec3>, Vec<[u32; 4]>) {
        let verts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        let tets = vec![[0u32, 1, 2, 3]];
        (verts, tets)
    }

    fn basis_of(verts: &[Vec3], tets: &[[u32; 4]]) -> TetFemBasis {
        build_tet_fem_basis(verts, tets, &TetFemBasisParams::default()).unwrap()
    }

    fn fibers_x(count: usize) -> Vec<FiberDirection> {
        vec![FiberDirection::new(Vec3::X).unwrap(); count]
    }

    /// Scales `pos` about its centroid by `factor` (uniform stretch).
    fn inflate(pos: &mut [Vec3], factor: f32) {
        let centroid: Vec3 = pos.iter().copied().sum::<Vec3>() / pos.len() as f32;
        for p in pos.iter_mut() {
            *p = centroid + (*p - centroid) * factor;
        }
    }

    /// Fiber stretch `λ = |F a0|` of one element under positions `pos`.
    fn fiber_stretch(
        basis: &TetFemBasis,
        tets: &[[u32; 4]],
        fiber: FiberDirection,
        pos: &[Vec3],
        element: usize,
    ) -> f32 {
        let e = &basis.elements[element];
        let t = tets[element].map(|v| v as usize);
        let p = [pos[t[0]], pos[t[1]], pos[t[2]], pos[t[3]]];
        fiber_geometry(&e.dm_inverse, fiber, &p).fa.length()
    }

    #[test]
    fn rejects_invalid_params() {
        assert!(FiberStrainLimitParams::new(1.1, 0.9, 1).is_none());
        assert!(FiberStrainLimitParams::new(0.9, 1.1, 0).is_none());
        assert!(FiberStrainLimitParams::new(0.0, 1.1, 1).is_none());
        assert!(FiberStrainLimitParams::new(f32::NAN, 1.0, 1).is_none());
        assert!(FiberStrainLimitParams::symmetric(0.0, 1).is_none());
        assert!(FiberStrainLimitParams::symmetric(1.0, 1).is_none());
        assert!(FiberStrainLimitParams::new(0.9, 1.1, 1).is_some());
        assert!(FiberStrainLimitParams::symmetric(0.1, 1).is_some());
    }

    #[test]
    fn rejects_dimension_mismatch() {
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let mut pos = verts.clone();
        let params = FiberStrainLimitParams::symmetric(0.1, 1).unwrap();

        // fibers.len() mismatch.
        let bad_fibers = fibers_x(1);
        assert!(project_fiber_strain_limits(
            &basis,
            &tets,
            &bad_fibers,
            &mut pos,
            None,
            None,
            &params
        )
        .is_none());

        let fibers = fibers_x(tets.len());

        // tets.len() mismatch.
        let bad_tets = vec![[0u32, 1, 2, 3]];
        assert!(project_fiber_strain_limits(
            &basis, &bad_tets, &fibers, &mut pos, None, None, &params
        )
        .is_none());

        // inv_mass length mismatch.
        let bad_inv = vec![1.0; pos.len() - 1];
        assert!(project_fiber_strain_limits(
            &basis,
            &tets,
            &fibers,
            &mut pos,
            Some(&bad_inv),
            None,
            &params
        )
        .is_none());

        // pinned length mismatch.
        let bad_pin = vec![false; pos.len() - 1];
        assert!(project_fiber_strain_limits(
            &basis,
            &tets,
            &fibers,
            &mut pos,
            None,
            Some(&bad_pin),
            &params
        )
        .is_none());

        // out-of-range tet index.
        let oob_tets = vec![[0u32, 1, 2, 99], [1, 2, 3, 4]];
        assert!(project_fiber_strain_limits(
            &basis, &oob_tets, &fibers, &mut pos, None, None, &params
        )
        .is_none());
    }

    #[test]
    fn rest_pose_is_left_untouched() {
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let fibers = fibers_x(tets.len());
        let mut pos = verts.clone();
        let params = FiberStrainLimitParams::symmetric(0.1, 3).unwrap();
        let report =
            project_fiber_strain_limits(&basis, &tets, &fibers, &mut pos, None, None, &params)
                .unwrap();
        assert_eq!(report.projected_elements, 0);
        assert_eq!(report.max_violation, 0.0);
        for (a, b) in pos.iter().zip(verts.iter()) {
            assert!((*a - *b).length() < 1e-6);
        }
    }

    #[test]
    fn reduces_fiber_violation() {
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let fibers = fibers_x(tets.len());
        let mut pos = verts.clone();
        inflate(&mut pos, 1.5);

        let before = fiber_stretch(&basis, &tets, fibers[0], &pos, 0);
        let params = FiberStrainLimitParams::symmetric(0.1, 40).unwrap();
        let report =
            project_fiber_strain_limits(&basis, &tets, &fibers, &mut pos, None, None, &params)
                .unwrap();
        assert!(report.projected_elements >= 1);
        assert!(
            report.max_violation > 0.3,
            "violation = {}",
            report.max_violation
        );

        let after = fiber_stretch(&basis, &tets, fibers[0], &pos, 0);
        assert!(after < before, "after {after} should be < before {before}");
        assert!(after <= 1.1 + 2e-2, "fiber stretch after = {after}");
    }

    #[test]
    fn off_axis_fiber_is_untouched() {
        // Single tet stretched 2x along x. A fiber along y has λ = 1 and must
        // not be projected; a fiber along x is at λ = 2 and must be pulled back.
        let (verts, tets) = single_tet();
        let basis = basis_of(&verts, &tets);
        let mut pos = verts.clone();
        pos[1] = Vec3::new(2.0, 0.0, 0.0);

        let fiber_y = vec![FiberDirection::new(Vec3::Y).unwrap()];
        let params = FiberStrainLimitParams::symmetric(0.1, 20).unwrap();
        let before = pos.clone();
        let report =
            project_fiber_strain_limits(&basis, &tets, &fiber_y, &mut pos, None, None, &params)
                .unwrap();
        assert_eq!(
            report.projected_elements, 0,
            "y-fiber should be inside band"
        );
        for (a, b) in pos.iter().zip(before.iter()) {
            assert!(
                (*a - *b).length() < 1e-6,
                "y-fiber projection moved a vertex"
            );
        }

        // Now the x fiber on the same stretched state must be projected.
        let fiber_x = fibers_x(1);
        let report_x =
            project_fiber_strain_limits(&basis, &tets, &fiber_x, &mut pos, None, None, &params)
                .unwrap();
        assert_eq!(report_x.projected_elements, 1);
        let after = fiber_stretch(&basis, &tets, fiber_x[0], &pos, 0);
        assert!(after <= 1.1 + 2e-2, "x fiber stretch after = {after}");
    }

    #[test]
    fn preserves_pinned_vertex() {
        let (verts, tets) = single_tet();
        let basis = basis_of(&verts, &tets);
        let fibers = fibers_x(tets.len());
        let mut pos = verts.clone();
        pos[1] = Vec3::new(2.0, 0.0, 0.0);
        let pinned = vec![true, false, false, false];
        let params = FiberStrainLimitParams::symmetric(0.1, 20).unwrap();
        project_fiber_strain_limits(
            &basis,
            &tets,
            &fibers,
            &mut pos,
            None,
            Some(&pinned),
            &params,
        )
        .unwrap();
        assert!(
            (pos[0] - verts[0]).length() < 1e-6,
            "pinned vertex moved to {:?}",
            pos[0]
        );
    }

    #[test]
    fn zero_inverse_mass_pins_vertex() {
        let (verts, tets) = single_tet();
        let basis = basis_of(&verts, &tets);
        let fibers = fibers_x(tets.len());
        let mut pos = verts.clone();
        pos[1] = Vec3::new(2.0, 0.0, 0.0);
        let inv_mass = vec![0.0, 1.0, 1.0, 1.0];
        let params = FiberStrainLimitParams::symmetric(0.1, 20).unwrap();
        project_fiber_strain_limits(
            &basis,
            &tets,
            &fibers,
            &mut pos,
            Some(&inv_mass),
            None,
            &params,
        )
        .unwrap();
        assert!(
            (pos[0] - verts[0]).length() < 1e-6,
            "pinned vertex moved to {:?}",
            pos[0]
        );
    }

    #[test]
    fn momentum_preserved_single_element() {
        // On a single element every vertex has count 1, so the Jacobi average
        // does not distort the per-element momentum-preserving correction and
        // the centroid is preserved to tight tolerance.
        let (verts, tets) = single_tet();
        let basis = basis_of(&verts, &tets);
        let fibers = fibers_x(tets.len());
        let mut pos = verts.clone();
        pos[1] = Vec3::new(2.2, 0.1, -0.1);
        let before: Vec3 = pos.iter().copied().sum::<Vec3>() / pos.len() as f32;
        let params = FiberStrainLimitParams::symmetric(0.1, 10).unwrap();
        project_fiber_strain_limits(&basis, &tets, &fibers, &mut pos, None, None, &params).unwrap();
        let after: Vec3 = pos.iter().copied().sum::<Vec3>() / pos.len() as f32;
        assert!(
            (after - before).length() < 1e-5,
            "centroid drift = {}",
            (after - before).length()
        );
    }

    #[test]
    fn deterministic() {
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let fibers = fibers_x(tets.len());
        let params = FiberStrainLimitParams::symmetric(0.1, 15).unwrap();

        let mut a = verts.clone();
        inflate(&mut a, 1.4);
        let mut b = a.clone();
        project_fiber_strain_limits(&basis, &tets, &fibers, &mut a, None, None, &params).unwrap();
        project_fiber_strain_limits(&basis, &tets, &fibers, &mut b, None, None, &params).unwrap();
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(*x, *y);
        }
    }
}
