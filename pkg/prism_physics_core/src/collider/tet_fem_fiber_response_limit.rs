//! Response-aware (tension-only vs. bidirectional) fiber strain limiting.
//!
//! The orthotropic limiter in [`super::tet_fem_orthotropic_strain_limit`] clamps
//! every fiber family to a symmetric admissible band `[min, max]`, resisting
//! both over-stretch *and* compression. That is correct for bonded
//! reinforcement that cannot buckle, but it is wrong for the physically common
//! case of slender fibers — muscle, tendon, yarn — which carry load only in
//! tension and simply buckle (go slack) under compression. Forcing a slack
//! fiber back up to `min` injects spurious energy and makes cloth and soft
//! tissue look rubbery.
//!
//! This module generalizes the per-family band with a
//! [`FiberResponse`](super::tet_fem_anisotropic::FiberResponse):
//!
//! * [`FiberResponse::Bidirectional`] keeps the full `[min, max]` band.
//! * [`FiberResponse::TensionOnly`] drops the lower bound so a stretch below
//!   `max` — including any compression `λ < 1` — is never projected; only
//!   over-stretch beyond `max` is pulled back.
//!
//! # Method
//!
//! Each family is mapped to an *effective* [`FiberBand`] encoding its response,
//! then handed to the crate's shared single-family kernel
//! ([`measure_fiber_family`] / [`apply_fiber_jacobi_sweep`]). The families are
//! swept Gauss–Seidel within each outer iteration (family `f`'s correction is
//! applied before family `f+1` is evaluated) and the whole pass is repeated for
//! `iterations` sweeps — identical scheduling to the orthotropic limiter, so
//! the two agree exactly when every family is `Bidirectional`.
//!
//! Fibers are supplied **family-major**: all elements of family `0`, then all
//! of family `1`, and so on; family `f` occupies
//! `fibers[f * elements .. (f + 1) * elements]`.
//!
//! The module owns no simulation state and performs no time integration; it is
//! a pure geometric projection reusing this crate's own primitives.
//!
//! # Attribution
//!
//! Clean-room composition over the crate's single-family fiber limiter kernel.
//! No Unreal Engine source or derived code.

use super::tet_fem_anisotropic::{FiberDirection, FiberResponse};
use super::tet_fem_basis::TetFemBasis;
use super::tet_fem_fiber_strain_limit::{
    apply_fiber_jacobi_sweep, measure_fiber_family, FiberBand,
};
use glam::Vec3;

/// One fiber family's admissible stretch band plus its load response.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FiberResponseBand {
    /// Lower stretch bound (used only when `response` is `Bidirectional`).
    pub min_stretch: f32,
    /// Upper stretch bound; over-stretch beyond this is always projected.
    pub max_stretch: f32,
    /// Whether the family resists compression or only tension.
    pub response: FiberResponse,
}

impl FiberResponseBand {
    /// Builds a validated band.
    ///
    /// Returns `None` unless `min_stretch` and `max_stretch` are finite with
    /// `0 < min_stretch <= max_stretch`.
    #[must_use]
    pub fn new(min_stretch: f32, max_stretch: f32, response: FiberResponse) -> Option<Self> {
        if !min_stretch.is_finite() || !max_stretch.is_finite() {
            return None;
        }
        if min_stretch <= 0.0 || max_stretch < min_stretch {
            return None;
        }
        Some(Self {
            min_stretch,
            max_stretch,
            response,
        })
    }

    /// The effective geometric band handed to the shared kernel. A
    /// tension-only family drops its lower bound to zero so no realistic
    /// stretch (`λ >= 0`) is ever flagged as a compression violation.
    #[must_use]
    fn effective(self) -> FiberBand {
        let min_stretch = match self.response {
            FiberResponse::Bidirectional => self.min_stretch,
            FiberResponse::TensionOnly => 0.0,
        };
        FiberBand {
            min_stretch,
            max_stretch: self.max_stretch,
        }
    }
}

/// Validated parameters for [`project_fiber_response_strain_limits`]: one
/// response-aware band per fiber family plus the outer sweep count.
#[derive(Clone, Debug, PartialEq)]
pub struct FiberResponseLimitParams {
    bands: Vec<FiberResponseBand>,
    iterations: u32,
}

impl FiberResponseLimitParams {
    /// Builds parameters from one `(min, max, response)` tuple per family.
    ///
    /// Returns `None` unless `families` is non-empty, `iterations >= 1`, and
    /// every band validates via [`FiberResponseBand::new`].
    #[must_use]
    pub fn new(families: &[(f32, f32, FiberResponse)], iterations: u32) -> Option<Self> {
        if families.is_empty() || iterations == 0 {
            return None;
        }
        let mut bands = Vec::with_capacity(families.len());
        for &(min, max, response) in families {
            bands.push(FiberResponseBand::new(min, max, response)?);
        }
        Some(Self { bands, iterations })
    }

    /// Builds parameters from pre-validated bands.
    ///
    /// Returns `None` unless `bands` is non-empty and `iterations >= 1`.
    #[must_use]
    pub fn from_bands(bands: &[FiberResponseBand], iterations: u32) -> Option<Self> {
        if bands.is_empty() || iterations == 0 {
            return None;
        }
        Some(Self {
            bands: bands.to_vec(),
            iterations,
        })
    }

    /// Builds a symmetric band `[1-pct, 1+pct]` for every family, each sharing
    /// `response`.
    ///
    /// Returns `None` unless `0 < pct < 1`, `families >= 1`, `iterations >= 1`.
    #[must_use]
    pub fn symmetric_uniform(
        pct: f32,
        response: FiberResponse,
        families: usize,
        iterations: u32,
    ) -> Option<Self> {
        if !(pct.is_finite() && pct > 0.0 && pct < 1.0) || families == 0 || iterations == 0 {
            return None;
        }
        let band = FiberResponseBand::new(1.0 - pct, 1.0 + pct, response)?;
        Some(Self {
            bands: vec![band; families],
            iterations,
        })
    }

    /// Number of fiber families.
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

/// Diagnostics of a response-aware fiber strain-limiting pass, measured on the
/// untouched input pose.
#[derive(Clone, Debug, PartialEq)]
pub struct FiberResponseLimitReport {
    /// Largest band violation across all families and elements.
    pub max_violation: f32,
    /// Number of distinct elements violating at least one family.
    pub projected_elements: usize,
    /// Per-family largest violation, in family order.
    pub per_family_violation: Vec<f32>,
}

/// Projects `positions` so every fiber family satisfies its response-aware
/// stretch band.
///
/// `tets` maps each basis element to its four vertex indices. `fibers` is the
/// family-major fiber field; its length must equal `families * elements`.
/// `inv_mass`, when supplied, weights the momentum-preserving corrections (a
/// zero entry pins the vertex); `None` treats every vertex as unit mass.
/// `pinned`, when supplied, additionally freezes the listed vertices.
///
/// Returns `None` on any dimension mismatch (in which case `positions` is left
/// untouched), or the [`FiberResponseLimitReport`] measured on the untouched
/// input otherwise.
#[must_use]
pub fn project_fiber_response_strain_limits(
    basis: &TetFemBasis,
    tets: &[[u32; 4]],
    fibers: &[FiberDirection],
    positions: &mut [Vec3],
    inv_mass: Option<&[f32]>,
    pinned: Option<&[bool]>,
    params: &FiberResponseLimitParams,
) -> Option<FiberResponseLimitReport> {
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

    // Measure every family on the untouched input, OR-ing violating-element
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
            band.effective(),
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
                band.effective(),
                &mut accum,
                &mut counts,
            );
        }
    }

    Some(FiberResponseLimitReport {
        max_violation,
        projected_elements,
        per_family_violation,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collider::tet_fem_anisotropic::FiberResponse;
    use crate::collider::tet_fem_basis::{build_tet_fem_basis, TetFemBasisParams};
    use crate::collider::tet_fem_fiber_strain_limit::fiber_geometry;

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

    fn basis_of(verts: &[Vec3], tets: &[[u32; 4]]) -> TetFemBasis {
        build_tet_fem_basis(verts, tets, &TetFemBasisParams::default()).unwrap()
    }

    fn x_fibers(elements: usize) -> Vec<FiberDirection> {
        vec![FiberDirection::new(Vec3::X).unwrap(); elements]
    }

    fn scale(pos: &mut [Vec3], factor: f32) {
        let centroid: Vec3 = pos.iter().copied().sum::<Vec3>() / pos.len() as f32;
        for p in pos.iter_mut() {
            *p = centroid + (*p - centroid) * factor;
        }
    }

    fn fiber_stretch(basis: &TetFemBasis, tets: &[[u32; 4]], pos: &[Vec3], element: usize) -> f32 {
        let e = &basis.elements[element];
        let t = tets[element].map(|v| v as usize);
        let p = [pos[t[0]], pos[t[1]], pos[t[2]], pos[t[3]]];
        fiber_geometry(&e.dm_inverse, FiberDirection::new(Vec3::X).unwrap(), &p)
            .fa
            .length()
    }

    #[test]
    fn rejects_invalid_bands_and_params() {
        assert!(FiberResponseBand::new(0.0, 1.1, FiberResponse::Bidirectional).is_none());
        assert!(FiberResponseBand::new(1.1, 0.9, FiberResponse::Bidirectional).is_none());
        assert!(FiberResponseBand::new(0.9, f32::NAN, FiberResponse::Bidirectional).is_none());
        assert!(FiberResponseBand::new(0.9, 1.1, FiberResponse::TensionOnly).is_some());

        assert!(FiberResponseLimitParams::new(&[], 1).is_none());
        assert!(
            FiberResponseLimitParams::new(&[(0.9, 1.1, FiberResponse::TensionOnly)], 0).is_none()
        );
        assert!(
            FiberResponseLimitParams::new(&[(0.9, 1.1, FiberResponse::TensionOnly)], 2).is_some()
        );
        assert!(
            FiberResponseLimitParams::symmetric_uniform(0.0, FiberResponse::TensionOnly, 2, 1)
                .is_none()
        );
        let p = FiberResponseLimitParams::symmetric_uniform(0.1, FiberResponse::TensionOnly, 2, 3)
            .unwrap();
        assert_eq!(p.families(), 2);
        assert_eq!(p.iterations(), 3);
    }

    #[test]
    fn effective_band_drops_lower_bound_only_for_tension_only() {
        let bi = FiberResponseBand::new(0.8, 1.2, FiberResponse::Bidirectional).unwrap();
        assert_eq!(bi.effective().min_stretch, 0.8);
        let to = FiberResponseBand::new(0.8, 1.2, FiberResponse::TensionOnly).unwrap();
        assert_eq!(to.effective().min_stretch, 0.0);
        assert_eq!(to.effective().max_stretch, 1.2);
    }

    #[test]
    fn rejects_dimension_mismatch_leaves_positions_untouched() {
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let mut pos = verts.clone();
        scale(&mut pos, 1.3);
        let snapshot = pos.clone();
        // 1 family, 2 elements => needs 2 fibers; give 1.
        let bad = x_fibers(1);
        let params =
            FiberResponseLimitParams::new(&[(0.9, 1.1, FiberResponse::TensionOnly)], 2).unwrap();
        assert!(project_fiber_response_strain_limits(
            &basis, &tets, &bad, &mut pos, None, None, &params
        )
        .is_none());
        assert_eq!(pos, snapshot);
    }

    #[test]
    fn tension_only_ignores_compression() {
        // Compress the single tet so the X fiber stretch λ < min.
        let (verts, tets) = single_tet();
        let basis = basis_of(&verts, &tets);
        let fibers = x_fibers(tets.len());
        let mut pos = verts.clone();
        scale(&mut pos, 0.6);
        let lambda_before = fiber_stretch(&basis, &tets, &pos, 0);
        assert!(lambda_before < 0.9, "setup: fiber should be compressed");
        let before = pos.clone();

        let params =
            FiberResponseLimitParams::new(&[(0.9, 1.1, FiberResponse::TensionOnly)], 8).unwrap();
        let report = project_fiber_response_strain_limits(
            &basis, &tets, &fibers, &mut pos, None, None, &params,
        )
        .unwrap();

        assert_eq!(
            report.max_violation, 0.0,
            "tension-only ignores compression"
        );
        assert_eq!(report.projected_elements, 0);
        for (a, b) in pos.iter().zip(before.iter()) {
            assert!((*a - *b).length() < 1e-6, "compressed pose must not move");
        }
    }

    #[test]
    fn bidirectional_corrects_compression() {
        // Same compression, but a bidirectional family must pull it back up.
        let (verts, tets) = single_tet();
        let basis = basis_of(&verts, &tets);
        let fibers = x_fibers(tets.len());
        let mut pos = verts.clone();
        scale(&mut pos, 0.6);
        let lambda_before = fiber_stretch(&basis, &tets, &pos, 0);

        let params =
            FiberResponseLimitParams::new(&[(0.9, 1.1, FiberResponse::Bidirectional)], 16).unwrap();
        let report = project_fiber_response_strain_limits(
            &basis, &tets, &fibers, &mut pos, None, None, &params,
        )
        .unwrap();

        assert!(report.max_violation > 0.0, "bidirectional sees compression");
        let lambda_after = fiber_stretch(&basis, &tets, &pos, 0);
        assert!(
            lambda_after > lambda_before,
            "compression {lambda_before} should be pulled toward min, got {lambda_after}"
        );
    }

    #[test]
    fn tension_only_still_corrects_overstretch() {
        let (verts, tets) = single_tet();
        let basis = basis_of(&verts, &tets);
        let fibers = x_fibers(tets.len());
        let mut pos = verts.clone();
        scale(&mut pos, 1.5);
        let lambda_before = fiber_stretch(&basis, &tets, &pos, 0);
        assert!(lambda_before > 1.1, "setup: fiber over-stretched");

        let params =
            FiberResponseLimitParams::new(&[(0.9, 1.1, FiberResponse::TensionOnly)], 16).unwrap();
        let report = project_fiber_response_strain_limits(
            &basis, &tets, &fibers, &mut pos, None, None, &params,
        )
        .unwrap();

        assert!(report.max_violation > 0.0);
        let lambda_after = fiber_stretch(&basis, &tets, &pos, 0);
        assert!(
            lambda_after < lambda_before,
            "over-stretch {lambda_before} should be pulled toward max, got {lambda_after}"
        );
    }

    #[test]
    fn preserves_pinned_vertices() {
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let fibers = x_fibers(tets.len());
        let mut pos = verts.clone();
        scale(&mut pos, 1.4);
        let pinned = vec![true, false, false, false, false];
        let pinned_before = pos[0];
        let params =
            FiberResponseLimitParams::new(&[(0.95, 1.05, FiberResponse::TensionOnly)], 6).unwrap();
        project_fiber_response_strain_limits(
            &basis,
            &tets,
            &fibers,
            &mut pos,
            None,
            Some(&pinned),
            &params,
        )
        .unwrap();
        assert_eq!(pos[0], pinned_before);
    }

    #[test]
    fn is_deterministic() {
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let fibers = x_fibers(tets.len());
        let params =
            FiberResponseLimitParams::new(&[(0.95, 1.05, FiberResponse::TensionOnly)], 6).unwrap();
        let mut a = verts.clone();
        scale(&mut a, 1.45);
        let mut b = a.clone();
        project_fiber_response_strain_limits(&basis, &tets, &fibers, &mut a, None, None, &params)
            .unwrap();
        project_fiber_response_strain_limits(&basis, &tets, &fibers, &mut b, None, None, &params)
            .unwrap();
        assert_eq!(a, b);
    }
}
