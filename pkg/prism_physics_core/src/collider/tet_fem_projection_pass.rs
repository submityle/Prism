//! Unified post-step geometric projection pass composing strain limiting and
//! volume preservation into one ordered, iterated constraint sweep.
//!
//! Position-based solvers typically finish a frame with a short sequence of
//! geometric projections that pull the mesh back onto feasible manifolds the
//! time integrator does not enforce exactly: a strain-limiting band
//! ([`project_strain_limits`]) and a volume/incompressibility constraint
//! ([`project_volume`]). Running them individually leaves the caller to decide
//! the order, how many outer iterations to interleave them for, and how to
//! aggregate their diagnostics. This module bundles that policy into a single
//! all-or-nothing pass.
//!
//! # Semantics
//!
//! * **Composable.** Either sub-pass may be disabled; at least one must be
//!   enabled or the parameters are rejected.
//! * **Ordered.** [`ProjectionOrder`] selects which constraint is projected
//!   first within each outer iteration. Order matters because each projection
//!   perturbs the stretch/volume the other measures.
//! * **Iterated.** `outer_iterations` interleaves the enabled sub-passes so
//!   corrections from one propagate into the other, Gauss–Seidel style across
//!   passes.
//! * **All-or-nothing.** The pass runs on a private copy of `positions` and
//!   only writes the result back once every sub-pass of every outer iteration
//!   has validated and succeeded. A dimension mismatch anywhere leaves the
//!   caller's `positions` untouched and returns `None`.
//! * **Reported.** The returned [`ProjectionPassReport`] carries the enabled
//!   sub-passes' diagnostics measured on the *first* outer iteration, i.e. the
//!   incoming (pre-projection) violation — the quantity a caller watches to
//!   decide whether the pass was needed.
//!
//! Both sub-passes apply mass-weighted corrections and keep pinned/zero-mass
//! vertices fixed; those invariants are preserved by the composition.
//!
//! # Attribution
//!
//! Clean-room composition over the crate's own strain-limit and volume
//! projection primitives. No Unreal Engine source or derived code.

use super::tet_fem_basis::TetFemBasis;
use super::tet_fem_strain_limit::{project_strain_limits, StrainLimitParams, StrainLimitReport};
use super::tet_fem_volume_projection::{
    project_volume, VolumeProjectionParams, VolumeProjectionReport,
};
use glam::Vec3;

/// Order in which the enabled sub-passes run within each outer iteration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProjectionOrder {
    /// Project the strain band first, then volume.
    StrainThenVolume,
    /// Project volume first, then the strain band.
    VolumeThenStrain,
}

/// Parameters controlling a unified projection pass.
#[derive(Clone, Copy, Debug)]
pub struct ProjectionPassParams {
    /// Strain-limiting band, or `None` to skip strain limiting.
    pub strain: Option<StrainLimitParams>,
    /// Volume constraint, or `None` to skip volume preservation.
    pub volume: Option<VolumeProjectionParams>,
    /// Which sub-pass runs first within an outer iteration.
    pub order: ProjectionOrder,
    /// Number of outer iterations interleaving the enabled sub-passes; must be
    /// at least one.
    pub outer_iterations: u32,
}

impl ProjectionPassParams {
    /// Builds a validated pass configuration.
    ///
    /// Returns `None` unless at least one sub-pass is enabled and
    /// `outer_iterations >= 1`.
    #[must_use]
    pub fn new(
        strain: Option<StrainLimitParams>,
        volume: Option<VolumeProjectionParams>,
        order: ProjectionOrder,
        outer_iterations: u32,
    ) -> Option<Self> {
        if outer_iterations == 0 {
            return None;
        }
        if strain.is_none() && volume.is_none() {
            return None;
        }
        Some(Self {
            strain,
            volume,
            order,
            outer_iterations,
        })
    }
}

/// Aggregated diagnostics of a unified projection pass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProjectionPassReport {
    /// First-outer-iteration strain report, if strain limiting was enabled.
    pub strain: Option<StrainLimitReport>,
    /// First-outer-iteration volume report, if volume preservation was enabled.
    pub volume: Option<VolumeProjectionReport>,
    /// Number of outer iterations actually run (equals
    /// `params.outer_iterations`).
    pub outer_iterations_run: u32,
}

/// Runs the configured strain-limit and volume projections over `positions`.
///
/// `tets` maps each basis element to its four vertex indices. `inv_mass`, when
/// supplied, weights the momentum-preserving corrections (a zero entry pins the
/// vertex); `None` treats every vertex as unit mass. `pinned`, when supplied,
/// additionally freezes the listed vertices.
///
/// Returns `None` on any dimension mismatch surfaced by a sub-pass (in which
/// case `positions` is left untouched), or the aggregated
/// [`ProjectionPassReport`] on success.
#[must_use]
pub fn run_projection_pass(
    basis: &TetFemBasis,
    tets: &[[u32; 4]],
    positions: &mut [Vec3],
    inv_mass: Option<&[f32]>,
    pinned: Option<&[bool]>,
    params: &ProjectionPassParams,
) -> Option<ProjectionPassReport> {
    // Work on a private copy so a mid-pass validation failure cannot leave the
    // caller's buffer half-projected.
    let mut work = positions.to_vec();

    let mut strain_report: Option<StrainLimitReport> = None;
    let mut volume_report: Option<VolumeProjectionReport> = None;

    for outer in 0..params.outer_iterations {
        let first = outer == 0;
        match params.order {
            ProjectionOrder::StrainThenVolume => {
                run_strain(
                    basis,
                    tets,
                    &mut work,
                    inv_mass,
                    pinned,
                    params,
                    first,
                    &mut strain_report,
                )?;
                run_volume(
                    basis,
                    tets,
                    &mut work,
                    inv_mass,
                    pinned,
                    params,
                    first,
                    &mut volume_report,
                )?;
            }
            ProjectionOrder::VolumeThenStrain => {
                run_volume(
                    basis,
                    tets,
                    &mut work,
                    inv_mass,
                    pinned,
                    params,
                    first,
                    &mut volume_report,
                )?;
                run_strain(
                    basis,
                    tets,
                    &mut work,
                    inv_mass,
                    pinned,
                    params,
                    first,
                    &mut strain_report,
                )?;
            }
        }
    }

    positions.copy_from_slice(&work);
    Some(ProjectionPassReport {
        strain: strain_report,
        volume: volume_report,
        outer_iterations_run: params.outer_iterations,
    })
}

#[expect(
    clippy::too_many_arguments,
    reason = "an internal sub-pass dispatcher threading the shared projection \
              inputs plus the first-iteration report sink; splitting it would \
              only duplicate the identical argument list"
)]
fn run_strain(
    basis: &TetFemBasis,
    tets: &[[u32; 4]],
    work: &mut [Vec3],
    inv_mass: Option<&[f32]>,
    pinned: Option<&[bool]>,
    params: &ProjectionPassParams,
    capture: bool,
    sink: &mut Option<StrainLimitReport>,
) -> Option<()> {
    if let Some(p) = params.strain {
        let report = project_strain_limits(basis, tets, work, inv_mass, pinned, &p)?;
        if capture {
            *sink = Some(report);
        }
    }
    Some(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "an internal sub-pass dispatcher threading the shared projection \
              inputs plus the first-iteration report sink; splitting it would \
              only duplicate the identical argument list"
)]
fn run_volume(
    basis: &TetFemBasis,
    tets: &[[u32; 4]],
    work: &mut [Vec3],
    inv_mass: Option<&[f32]>,
    pinned: Option<&[bool]>,
    params: &ProjectionPassParams,
    capture: bool,
    sink: &mut Option<VolumeProjectionReport>,
) -> Option<()> {
    if let Some(p) = params.volume {
        let report = project_volume(basis, tets, work, inv_mass, pinned, &p)?;
        if capture {
            *sink = Some(report);
        }
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collider::tet_fem_basis::{build_tet_fem_basis, TetFemBasisParams};
    use glam::Vec3;

    fn mesh() -> (Vec<Vec3>, Vec<[u32; 4]>) {
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

    /// Scales positions about their centroid so the mesh is uniformly stretched
    /// well outside any sane strain band and above its rest volume.
    fn inflate(verts: &[Vec3], factor: f32) -> Vec<Vec3> {
        let n = verts.len() as f32;
        let c = verts.iter().copied().fold(Vec3::ZERO, |a, b| a + b) / n;
        verts.iter().map(|v| c + (*v - c) * factor).collect()
    }

    #[test]
    fn rejects_invalid_params() {
        let s = StrainLimitParams::symmetric(0.1, 2);
        assert!(s.is_some());
        // zero outer iterations
        assert!(ProjectionPassParams::new(s, None, ProjectionOrder::StrainThenVolume, 0).is_none());
        // no sub-pass enabled
        assert!(
            ProjectionPassParams::new(None, None, ProjectionOrder::StrainThenVolume, 1).is_none()
        );
        // valid
        assert!(ProjectionPassParams::new(s, None, ProjectionOrder::StrainThenVolume, 1).is_some());
    }

    #[test]
    fn single_strain_only_matches_direct_call() {
        let (verts, tets) = mesh();
        let basis = basis_of(&verts, &tets);
        let sp = StrainLimitParams::symmetric(0.1, 3).unwrap();

        let inflated = inflate(&verts, 1.5);

        let mut direct = inflated.clone();
        let direct_report =
            project_strain_limits(&basis, &tets, &mut direct, None, None, &sp).unwrap();

        let params =
            ProjectionPassParams::new(Some(sp), None, ProjectionOrder::StrainThenVolume, 1)
                .unwrap();
        let mut via_pass = inflated.clone();
        let pass_report =
            run_projection_pass(&basis, &tets, &mut via_pass, None, None, &params).unwrap();

        assert_eq!(pass_report.strain, Some(direct_report));
        assert_eq!(pass_report.volume, None);
        assert_eq!(pass_report.outer_iterations_run, 1);
        for i in 0..verts.len() {
            assert!((via_pass[i] - direct[i]).length() < 1e-6);
        }
    }

    #[test]
    fn both_passes_report_and_run() {
        let (verts, tets) = mesh();
        let basis = basis_of(&verts, &tets);
        let sp = StrainLimitParams::symmetric(0.05, 2).unwrap();
        let vp = VolumeProjectionParams::incompressible(2).unwrap();
        let params =
            ProjectionPassParams::new(Some(sp), Some(vp), ProjectionOrder::StrainThenVolume, 3)
                .unwrap();

        let mut positions = inflate(&verts, 1.4);
        let report =
            run_projection_pass(&basis, &tets, &mut positions, None, None, &params).unwrap();

        assert!(report.strain.is_some());
        assert!(report.volume.is_some());
        assert_eq!(report.outer_iterations_run, 3);
        // The incoming configuration was over-stretched, so the first-iteration
        // strain report must have flagged violations.
        assert!(report.strain.unwrap().max_violation > 0.0);
    }

    #[test]
    fn reduces_strain_violation() {
        let (verts, tets) = mesh();
        let basis = basis_of(&verts, &tets);
        let sp = StrainLimitParams::symmetric(0.08, 4).unwrap();
        let params =
            ProjectionPassParams::new(Some(sp), None, ProjectionOrder::StrainThenVolume, 6)
                .unwrap();

        let mut positions = inflate(&verts, 1.5);
        let report =
            run_projection_pass(&basis, &tets, &mut positions, None, None, &params).unwrap();
        let incoming = report.strain.unwrap().max_violation;
        assert!(incoming > 0.0);

        // Probe the residual violation with a fresh single-sweep projection.
        let mut probe = positions.clone();
        let residual = project_strain_limits(&basis, &tets, &mut probe, None, None, &sp).unwrap();
        assert!(
            residual.max_violation < incoming,
            "residual {} not below incoming {}",
            residual.max_violation,
            incoming
        );
    }

    #[test]
    fn respects_pinned_vertices() {
        // Pinned vertices must never move, regardless of how over-stretched
        // the surrounding mesh is or how many outer iterations run.
        let (verts, tets) = mesh();
        let basis = basis_of(&verts, &tets);
        let sp = StrainLimitParams::symmetric(0.05, 2).unwrap();
        let vp = VolumeProjectionParams::incompressible(2).unwrap();
        let params =
            ProjectionPassParams::new(Some(sp), Some(vp), ProjectionOrder::VolumeThenStrain, 4)
                .unwrap();

        let mut pinned = vec![false; verts.len()];
        pinned[0] = true;
        pinned[4] = true;

        let mut positions = inflate(&verts, 1.3);
        let before = positions.clone();
        let _ = run_projection_pass(&basis, &tets, &mut positions, None, Some(&pinned), &params)
            .unwrap();

        for (i, p) in pinned.iter().enumerate() {
            if *p {
                assert!(
                    (positions[i] - before[i]).length() < 1e-6,
                    "pinned vertex {i} moved"
                );
            }
        }
        // A free vertex should still have been corrected.
        let moved = (positions[1] - before[1]).length();
        assert!(moved > 1e-5, "free vertex did not move: {moved}");
    }

    #[test]
    fn dimension_mismatch_leaves_positions_untouched() {
        let (verts, tets) = mesh();
        let basis = basis_of(&verts, &tets);
        let sp = StrainLimitParams::symmetric(0.1, 2).unwrap();
        let params =
            ProjectionPassParams::new(Some(sp), None, ProjectionOrder::StrainThenVolume, 2)
                .unwrap();

        // Wrong tet count triggers the sub-pass dimension guard.
        let bad_tets = vec![tets[0]];
        let mut positions = inflate(&verts, 1.5);
        let snapshot = positions.clone();
        assert!(
            run_projection_pass(&basis, &bad_tets, &mut positions, None, None, &params).is_none()
        );
        assert_eq!(positions, snapshot);
    }

    #[test]
    fn order_affects_result() {
        let (verts, tets) = mesh();
        let basis = basis_of(&verts, &tets);
        let sp = StrainLimitParams::symmetric(0.05, 2).unwrap();
        let vp = VolumeProjectionParams::incompressible(2).unwrap();

        let p_sv =
            ProjectionPassParams::new(Some(sp), Some(vp), ProjectionOrder::StrainThenVolume, 1)
                .unwrap();
        let p_vs =
            ProjectionPassParams::new(Some(sp), Some(vp), ProjectionOrder::VolumeThenStrain, 1)
                .unwrap();

        let base = inflate(&verts, 1.4);
        let mut a = base.clone();
        let mut b = base.clone();
        run_projection_pass(&basis, &tets, &mut a, None, None, &p_sv).unwrap();
        run_projection_pass(&basis, &tets, &mut b, None, None, &p_vs).unwrap();

        // A single interleave of two non-commuting projections must differ.
        let max_diff = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (*x - *y).length())
            .fold(0.0_f32, f32::max);
        assert!(max_diff > 1e-6, "orders produced identical result");
    }

    #[test]
    fn deterministic() {
        let (verts, tets) = mesh();
        let basis = basis_of(&verts, &tets);
        let sp = StrainLimitParams::symmetric(0.06, 3).unwrap();
        let vp = VolumeProjectionParams::incompressible(2).unwrap();
        let params =
            ProjectionPassParams::new(Some(sp), Some(vp), ProjectionOrder::StrainThenVolume, 4)
                .unwrap();

        let base = inflate(&verts, 1.4);
        let mut a = base.clone();
        let mut b = base.clone();
        let ra = run_projection_pass(&basis, &tets, &mut a, None, None, &params).unwrap();
        let rb = run_projection_pass(&basis, &tets, &mut b, None, None, &params).unwrap();
        assert_eq!(ra, rb);
        assert_eq!(a, b);
    }
}
