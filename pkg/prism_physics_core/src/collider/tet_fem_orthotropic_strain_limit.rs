//! Orthotropic (multi-fiber) strain limiting for tetrahedral FEM meshes.
//!
//! Woven cloth, fiber-reinforced composites, and biological tissue are commonly
//! *orthotropic*: they carry two (warp + weft) or more reinforcing directions,
//! each with its own stiffness and its own admissible stretch band. The
//! single-family limiter in [`super::tet_fem_fiber_strain_limit`] clamps one
//! rest-space fiber per element; this module generalizes it to an arbitrary
//! number of fiber families and limits every family in turn.
//!
//! # Method
//!
//! Each family reuses the exact per-element, momentum-preserving position-based
//! correction of the single-fiber limiter (see that module for the derivation
//! of the fiber vector `fa = F a0 = Σ w_j p_j`, the constraint
//! `C = λ − clamp(λ, min, max)`, and the correction
//! `Δp_j = −(invMass_j w_j C / Σ invMass_k w_k²) n̂`). The families are swept in
//! a *Gauss–Seidel* fashion within each outer iteration — family `f`'s Jacobi
//! correction is applied before family `f+1` is evaluated — so that coupling
//! between the warp and weft constraints is resolved rather than averaged. The
//! whole pass is repeated for `iterations` outer sweeps.
//!
//! Fibers are supplied as a single **family-major** slice: all elements of
//! family `0`, then all elements of family `1`, and so on. Family `f` occupies
//! the contiguous block `fibers[f * elements .. (f + 1) * elements]`, which lets
//! the limiter hand each family straight to the shared single-family kernel
//! without copying.
//!
//! The module owns no simulation state and performs no time integration; it is
//! a pure geometric projection that can run before or after any step.
//!
//! # Attribution
//!
//! Clean-room implementation of orthotropic position-based strain limiting
//! (Müller et al., "Position Based Dynamics"; Thomaszewski et al., anisotropic
//! strain limiting). No Unreal Engine source or derived code.

use super::tet_fem_anisotropic::FiberDirection;
use super::tet_fem_basis::TetFemBasis;
use super::tet_fem_fiber_strain_limit::{
    apply_fiber_jacobi_sweep, measure_fiber_family, FiberBand,
};
use glam::Vec3;

/// Validated parameters for [`project_orthotropic_strain_limits`]: one stretch
/// band per fiber family plus the number of outer Gauss–Seidel sweeps.
#[derive(Clone, Debug, PartialEq)]
pub struct OrthotropicStrainLimitParams {
    /// One band per fiber family, in the same order as the family-major fiber
    /// slice. Always non-empty.
    bands: Vec<FiberBand>,
    /// Number of outer sweeps over all families.
    iterations: u32,
}

impl OrthotropicStrainLimitParams {
    /// Builds validated parameters from one `(min, max)` band per family.
    ///
    /// Returns `None` unless `bands` is non-empty, `iterations >= 1`, and every
    /// band is finite with `0 < min <= max`.
    #[must_use]
    pub fn new(bands: &[(f32, f32)], iterations: u32) -> Option<Self> {
        if bands.is_empty() || iterations == 0 {
            return None;
        }
        let mut validated = Vec::with_capacity(bands.len());
        for &(min, max) in bands {
            if !min.is_finite() || !max.is_finite() || min <= 0.0 || max < min {
                return None;
            }
            validated.push(FiberBand {
                min_stretch: min,
                max_stretch: max,
            });
        }
        Some(Self {
            bands: validated,
            iterations,
        })
    }

    /// Builds parameters from one symmetric percentage band `[1 - pct, 1 + pct]`
    /// per family.
    ///
    /// Returns `None` unless `pcts` is non-empty, `iterations >= 1`, and every
    /// `pct` satisfies `0 < pct < 1`.
    #[must_use]
    pub fn symmetric(pcts: &[f32], iterations: u32) -> Option<Self> {
        if pcts.is_empty() || iterations == 0 {
            return None;
        }
        let mut bands = Vec::with_capacity(pcts.len());
        for &pct in pcts {
            if !pct.is_finite() || pct <= 0.0 || pct >= 1.0 {
                return None;
            }
            bands.push((1.0 - pct, 1.0 + pct));
        }
        Self::new(&bands, iterations)
    }

    /// Builds parameters with the same symmetric band on every one of
    /// `families` fiber families.
    ///
    /// Returns `None` unless `families >= 1`, `iterations >= 1`, and
    /// `0 < pct < 1`.
    #[must_use]
    pub fn symmetric_uniform(pct: f32, families: usize, iterations: u32) -> Option<Self> {
        if families == 0 {
            return None;
        }
        let pcts = vec![pct; families];
        Self::symmetric(&pcts, iterations)
    }

    /// Number of fiber families (always `>= 1`).
    #[must_use]
    pub fn families(&self) -> usize {
        self.bands.len()
    }

    /// Number of outer sweeps.
    #[must_use]
    pub fn iterations(&self) -> u32 {
        self.iterations
    }
}

/// Outcome of an orthotropic projection, measured on the untouched input before
/// any correction.
#[derive(Clone, Debug, PartialEq)]
pub struct OrthotropicStrainLimitReport {
    /// Largest band violation across all families and elements.
    pub max_violation: f32,
    /// Number of *distinct* elements that violated at least one family's band.
    pub projected_elements: usize,
    /// Largest band violation of each family individually, in family order.
    pub per_family_violation: Vec<f32>,
}

/// Projects every element's fiber stretch into the per-family bands in
/// `params`, mutating `positions` in place.
///
/// `fibers` is a **family-major** slice of length `params.families() * E`, where
/// `E` is the number of basis elements: family `f`'s rest-space direction for
/// element `e` lives at `fibers[f * E + e]`. `tets` maps each basis element to
/// its four vertex indices into `positions`. `inv_mass` and `pinned` behave as
/// in [`super::tet_fem_fiber_strain_limit::project_fiber_strain_limits`].
///
/// Returns `None` on any dimension mismatch (`tets.len() != E`,
/// `fibers.len() != families * E`, any tet index out of range for `positions`,
/// or `inv_mass`/`pinned` length differing from `positions.len()`).
#[must_use]
pub fn project_orthotropic_strain_limits(
    basis: &TetFemBasis,
    tets: &[[u32; 4]],
    fibers: &[FiberDirection],
    positions: &mut [Vec3],
    inv_mass: Option<&[f32]>,
    pinned: Option<&[bool]>,
    params: &OrthotropicStrainLimitParams,
) -> Option<OrthotropicStrainLimitReport> {
    let element_count = basis.elements.len();
    let families = params.bands.len();
    if tets.len() != element_count || fibers.len() != families * element_count {
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

    let family_slice =
        |f: usize| -> &[FiberDirection] { &fibers[f * element_count..(f + 1) * element_count] };

    // Measure every family on the untouched input, OR-ing the violating-element
    // flags so distinct elements are counted once.
    let mut violated = vec![false; element_count];
    let mut per_family_violation = vec![0.0f32; families];
    let mut max_violation = 0.0f32;
    for (f, band) in params.bands.iter().enumerate() {
        let (fam_max, _) = measure_fiber_family(
            basis,
            tets,
            family_slice(f),
            positions,
            *band,
            Some(&mut violated),
        );
        per_family_violation[f] = fam_max;
        max_violation = max_violation.max(fam_max);
    }
    let projected_elements = violated.iter().filter(|&&v| v).count();

    // Gauss–Seidel over families, Jacobi over elements within a family.
    let mut accum = vec![[0.0f32; 3]; n];
    let mut counts = vec![0u32; n];
    for _ in 0..params.iterations {
        for (f, band) in params.bands.iter().enumerate() {
            apply_fiber_jacobi_sweep(
                basis,
                tets,
                family_slice(f),
                positions,
                inv_mass,
                pinned,
                *band,
                &mut accum,
                &mut counts,
            );
        }
    }

    Some(OrthotropicStrainLimitReport {
        max_violation,
        projected_elements,
        per_family_violation,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collider::tet_fem_basis::{build_tet_fem_basis, TetFemBasisParams};
    use crate::collider::tet_fem_fiber_strain_limit::fiber_geometry;

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

    /// Family-major fibers: all elements of family 0 then all of family 1.
    fn warp_weft(elements: usize) -> Vec<FiberDirection> {
        let mut v = vec![FiberDirection::new(Vec3::X).unwrap(); elements];
        v.extend(vec![FiberDirection::new(Vec3::Y).unwrap(); elements]);
        v
    }

    fn inflate(pos: &mut [Vec3], factor: f32) {
        let centroid: Vec3 = pos.iter().copied().sum::<Vec3>() / pos.len() as f32;
        for p in pos.iter_mut() {
            *p = centroid + (*p - centroid) * factor;
        }
    }

    fn fiber_stretch(
        basis: &TetFemBasis,
        tets: &[[u32; 4]],
        dir: FiberDirection,
        pos: &[Vec3],
        element: usize,
    ) -> f32 {
        let e = &basis.elements[element];
        let t = tets[element].map(|v| v as usize);
        let p = [pos[t[0]], pos[t[1]], pos[t[2]], pos[t[3]]];
        fiber_geometry(&e.dm_inverse, dir, &p).fa.length()
    }

    #[test]
    fn rejects_invalid_params() {
        assert!(OrthotropicStrainLimitParams::new(&[], 1).is_none());
        assert!(OrthotropicStrainLimitParams::new(&[(0.9, 1.1)], 0).is_none());
        assert!(OrthotropicStrainLimitParams::new(&[(1.1, 0.9)], 1).is_none());
        assert!(OrthotropicStrainLimitParams::new(&[(0.0, 1.1)], 1).is_none());
        assert!(OrthotropicStrainLimitParams::new(&[(0.9, f32::NAN)], 1).is_none());
        assert!(OrthotropicStrainLimitParams::new(&[(0.9, 1.1), (0.8, 1.2)], 2).is_some());
        assert!(OrthotropicStrainLimitParams::symmetric(&[], 1).is_none());
        assert!(OrthotropicStrainLimitParams::symmetric(&[0.0], 1).is_none());
        assert!(OrthotropicStrainLimitParams::symmetric(&[0.1, 0.2], 1).is_some());
        assert!(OrthotropicStrainLimitParams::symmetric_uniform(0.1, 0, 1).is_none());
        let p = OrthotropicStrainLimitParams::symmetric_uniform(0.1, 2, 3).unwrap();
        assert_eq!(p.families(), 2);
        assert_eq!(p.iterations(), 3);
    }

    #[test]
    fn rejects_dimension_mismatch() {
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let mut pos = verts.clone();
        let params = OrthotropicStrainLimitParams::symmetric_uniform(0.1, 2, 1).unwrap();
        let fibers = warp_weft(tets.len());

        // fibers length mismatch (3 instead of 2*2).
        let bad_fibers = vec![FiberDirection::new(Vec3::X).unwrap(); 3];
        assert!(project_orthotropic_strain_limits(
            &basis,
            &tets,
            &bad_fibers,
            &mut pos,
            None,
            None,
            &params
        )
        .is_none());

        // tets length mismatch.
        let bad_tets = vec![[0u32, 1, 2, 3]];
        assert!(project_orthotropic_strain_limits(
            &basis, &bad_tets, &fibers, &mut pos, None, None, &params
        )
        .is_none());

        // inv_mass / pinned length mismatch.
        let bad_inv = vec![1.0; pos.len() - 1];
        assert!(project_orthotropic_strain_limits(
            &basis,
            &tets,
            &fibers,
            &mut pos,
            Some(&bad_inv),
            None,
            &params
        )
        .is_none());
        let bad_pin = vec![false; pos.len() - 1];
        assert!(project_orthotropic_strain_limits(
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
        let oob = vec![[0u32, 1, 2, 99], [1, 2, 3, 4]];
        assert!(project_orthotropic_strain_limits(
            &basis, &oob, &fibers, &mut pos, None, None, &params
        )
        .is_none());
    }

    #[test]
    fn rest_pose_is_left_untouched() {
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let fibers = warp_weft(tets.len());
        let params = OrthotropicStrainLimitParams::symmetric_uniform(0.1, 2, 3).unwrap();
        let mut pos = verts.clone();
        let report = project_orthotropic_strain_limits(
            &basis, &tets, &fibers, &mut pos, None, None, &params,
        )
        .unwrap();
        assert_eq!(report.projected_elements, 0);
        assert_eq!(report.max_violation, 0.0);
        assert_eq!(report.per_family_violation, vec![0.0, 0.0]);
        for (a, b) in pos.iter().zip(verts.iter()) {
            assert!((*a - *b).length() < 1e-6);
        }
    }

    #[test]
    fn limits_both_families_under_uniform_inflation() {
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let fibers = warp_weft(tets.len());
        let mut pos = verts.clone();
        inflate(&mut pos, 1.5);

        let x = FiberDirection::new(Vec3::X).unwrap();
        let y = FiberDirection::new(Vec3::Y).unwrap();
        let before_x = fiber_stretch(&basis, &tets, x, &pos, 0);
        let before_y = fiber_stretch(&basis, &tets, y, &pos, 0);

        let params = OrthotropicStrainLimitParams::symmetric_uniform(0.1, 2, 60).unwrap();
        let report = project_orthotropic_strain_limits(
            &basis, &tets, &fibers, &mut pos, None, None, &params,
        )
        .unwrap();
        assert!(report.projected_elements >= 1);
        assert!(report.per_family_violation[0] > 0.3);
        assert!(report.per_family_violation[1] > 0.3);

        let after_x = fiber_stretch(&basis, &tets, x, &pos, 0);
        let after_y = fiber_stretch(&basis, &tets, y, &pos, 0);
        assert!(after_x < before_x && after_y < before_y);
        assert!(after_x <= 1.1 + 3e-2, "x stretch after = {after_x}");
        assert!(after_y <= 1.1 + 3e-2, "y stretch after = {after_y}");
    }

    #[test]
    fn stretched_only_along_one_family() {
        // Single tet stretched 2x along x. Warp = x (violates), weft = y (ok).
        let (verts, tets) = single_tet();
        let basis = basis_of(&verts, &tets);
        let mut pos = verts.clone();
        pos[1] = Vec3::new(2.0, 0.0, 0.0);
        let fibers = warp_weft(tets.len()); // [X ; Y], family-major, 1 element each

        let params = OrthotropicStrainLimitParams::symmetric_uniform(0.1, 2, 20).unwrap();
        let report = project_orthotropic_strain_limits(
            &basis, &tets, &fibers, &mut pos, None, None, &params,
        )
        .unwrap();
        assert_eq!(report.projected_elements, 1);
        assert!(report.per_family_violation[0] > 0.8, "warp violation");
        assert_eq!(report.per_family_violation[1], 0.0, "weft already in band");

        let x = FiberDirection::new(Vec3::X).unwrap();
        let after_x = fiber_stretch(&basis, &tets, x, &pos, 0);
        assert!(after_x <= 1.1 + 2e-2, "x stretch after = {after_x}");
    }

    #[test]
    fn preserves_pinned_vertex() {
        let (verts, tets) = single_tet();
        let basis = basis_of(&verts, &tets);
        let fibers = warp_weft(tets.len());
        let mut pos = verts.clone();
        pos[1] = Vec3::new(2.0, 0.0, 0.0);
        let pinned = vec![true, false, false, false];
        let params = OrthotropicStrainLimitParams::symmetric_uniform(0.1, 2, 20).unwrap();
        project_orthotropic_strain_limits(
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
    fn momentum_preserved_single_element() {
        let (verts, tets) = single_tet();
        let basis = basis_of(&verts, &tets);
        let fibers = warp_weft(tets.len());
        let mut pos = verts.clone();
        pos[1] = Vec3::new(2.2, 0.1, -0.1);
        let before: Vec3 = pos.iter().copied().sum::<Vec3>() / pos.len() as f32;
        let params = OrthotropicStrainLimitParams::symmetric_uniform(0.1, 2, 10).unwrap();
        project_orthotropic_strain_limits(&basis, &tets, &fibers, &mut pos, None, None, &params)
            .unwrap();
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
        let fibers = warp_weft(tets.len());
        let params = OrthotropicStrainLimitParams::symmetric_uniform(0.1, 2, 15).unwrap();

        let mut a = verts.clone();
        inflate(&mut a, 1.4);
        let mut b = a.clone();
        project_orthotropic_strain_limits(&basis, &tets, &fibers, &mut a, None, None, &params)
            .unwrap();
        project_orthotropic_strain_limits(&basis, &tets, &fibers, &mut b, None, None, &params)
            .unwrap();
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(*x, *y);
        }
    }
}
