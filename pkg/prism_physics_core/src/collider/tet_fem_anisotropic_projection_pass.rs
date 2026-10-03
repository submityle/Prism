//! Unified anisotropic post-step geometric projection pass.
//!
//! Where [`run_projection_pass`](super::tet_fem_projection_pass::run_projection_pass)
//! composes isotropic strain limiting with volume preservation, cloth- and
//! muscle-like materials additionally need *fiber-aligned* strain limiting:
//! warp/weft yarns and muscle fibers resist stretch along specific material
//! directions far more than the bulk continuum does. This module bundles three
//! geometric projections into one ordered, iterated, all-or-nothing sweep:
//!
//! * isotropic strain limiting
//!   ([`project_strain_limits`]) — the bulk stretch band,
//! * orthotropic fiber strain limiting
//!   ([`project_orthotropic_strain_limits`]) — one or more fiber families,
//! * volume preservation
//!   ([`project_volume`]) — near-incompressibility.
//!
//! # Semantics
//!
//! * **Composable.** Each stage may be disabled; at least one must be enabled
//!   or the parameters are rejected.
//! * **Ordered.** [`AnisotropicStage`] values form a permutation selecting the
//!   intra-iteration order. Order matters because every projection perturbs the
//!   quantities the others measure.
//! * **Iterated.** `outer_iterations` interleaves the enabled stages so one
//!   stage's correction propagates into the next, Gauss–Seidel across stages.
//! * **All-or-nothing.** The pass runs on a private copy of `positions` and
//!   only writes the result back once every enabled stage of every outer
//!   iteration has validated and succeeded. Any dimension mismatch leaves the
//!   caller's `positions` untouched and returns `None`.
//! * **Reported.** The returned [`AnisotropicProjectionReport`] carries the
//!   enabled stages' diagnostics measured on the *first* outer iteration, i.e.
//!   the incoming (pre-projection) violation.
//!
//! All three sub-passes apply mass-weighted, momentum-preserving corrections
//! and keep pinned/zero-mass vertices fixed; those invariants are preserved by
//! the composition.
//!
//! # Attribution
//!
//! Clean-room composition over the crate's own isotropic, orthotropic, and
//! volume projection primitives. No Unreal Engine source or derived code.

use super::tet_fem_anisotropic::FiberDirection;
use super::tet_fem_basis::TetFemBasis;
use super::tet_fem_orthotropic_strain_limit::{
    project_orthotropic_strain_limits, OrthotropicStrainLimitParams, OrthotropicStrainLimitReport,
};
use super::tet_fem_strain_limit::{project_strain_limits, StrainLimitParams, StrainLimitReport};
use super::tet_fem_volume_projection::{
    project_volume, VolumeProjectionParams, VolumeProjectionReport,
};
use glam::Vec3;

/// One projection stage in an anisotropic pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnisotropicStage {
    /// Isotropic (bulk) strain-limiting band.
    IsotropicStrain,
    /// Orthotropic fiber-aligned strain limiting.
    FiberStrain,
    /// Volume / incompressibility preservation.
    Volume,
}

impl AnisotropicStage {
    /// The three stages in canonical order.
    const ALL: [AnisotropicStage; 3] = [
        AnisotropicStage::IsotropicStrain,
        AnisotropicStage::FiberStrain,
        AnisotropicStage::Volume,
    ];
}

/// Parameters controlling a unified anisotropic projection pass.
///
/// Not `Copy`: [`OrthotropicStrainLimitParams`] owns a per-family `Vec` of
/// bands.
#[derive(Clone, Debug)]
pub struct AnisotropicProjectionParams {
    /// Isotropic strain band, or `None` to skip bulk strain limiting.
    pub isotropic: Option<StrainLimitParams>,
    /// Orthotropic fiber bands, or `None` to skip fiber strain limiting.
    pub fiber: Option<OrthotropicStrainLimitParams>,
    /// Volume constraint, or `None` to skip volume preservation.
    pub volume: Option<VolumeProjectionParams>,
    /// Intra-iteration stage order; must be a permutation of the three stages.
    pub order: [AnisotropicStage; 3],
    /// Number of outer iterations interleaving the enabled stages; at least 1.
    pub outer_iterations: u32,
}

impl AnisotropicProjectionParams {
    /// Builds a validated configuration.
    ///
    /// Returns `None` unless all hold:
    /// * at least one stage is enabled,
    /// * `outer_iterations >= 1`,
    /// * `order` lists all three stages exactly once (a permutation).
    #[must_use]
    pub fn new(
        isotropic: Option<StrainLimitParams>,
        fiber: Option<OrthotropicStrainLimitParams>,
        volume: Option<VolumeProjectionParams>,
        order: [AnisotropicStage; 3],
        outer_iterations: u32,
    ) -> Option<Self> {
        if outer_iterations == 0 {
            return None;
        }
        if isotropic.is_none() && fiber.is_none() && volume.is_none() {
            return None;
        }
        if !is_permutation(&order) {
            return None;
        }
        Some(Self {
            isotropic,
            fiber,
            volume,
            order,
            outer_iterations,
        })
    }

    /// Builds a configuration with the canonical order
    /// `[IsotropicStrain, FiberStrain, Volume]`.
    #[must_use]
    pub fn with_default_order(
        isotropic: Option<StrainLimitParams>,
        fiber: Option<OrthotropicStrainLimitParams>,
        volume: Option<VolumeProjectionParams>,
        outer_iterations: u32,
    ) -> Option<Self> {
        Self::new(
            isotropic,
            fiber,
            volume,
            AnisotropicStage::ALL,
            outer_iterations,
        )
    }
}

/// Returns `true` iff `order` contains each stage exactly once.
fn is_permutation(order: &[AnisotropicStage; 3]) -> bool {
    AnisotropicStage::ALL
        .iter()
        .all(|stage| order.iter().filter(|s| *s == stage).count() == 1)
}

/// Aggregated diagnostics of a unified anisotropic projection pass.
#[derive(Clone, Debug, PartialEq)]
pub struct AnisotropicProjectionReport {
    /// First-iteration isotropic report, if that stage was enabled.
    pub isotropic: Option<StrainLimitReport>,
    /// First-iteration fiber report, if that stage was enabled.
    pub fiber: Option<OrthotropicStrainLimitReport>,
    /// First-iteration volume report, if that stage was enabled.
    pub volume: Option<VolumeProjectionReport>,
    /// Number of outer iterations actually run (equals
    /// `params.outer_iterations`).
    pub outer_iterations_run: u32,
}

/// Runs the configured anisotropic projection stages over `positions`.
///
/// `tets` maps each basis element to its four vertex indices. `fibers` is the
/// family-major fiber field required *iff* `params.fiber` is `Some`; its length
/// must equal `families * elements` (validated by the fiber sub-pass). When
/// `params.fiber` is `None`, `fibers` is ignored and may be empty.
///
/// `inv_mass`, when supplied, weights the momentum-preserving corrections (a
/// zero entry pins the vertex); `None` treats every vertex as unit mass.
/// `pinned`, when supplied, additionally freezes the listed vertices.
///
/// Returns `None` on any dimension mismatch surfaced by a sub-pass (in which
/// case `positions` is left untouched), or the aggregated
/// [`AnisotropicProjectionReport`] on success.
#[must_use]
pub fn run_anisotropic_projection_pass(
    basis: &TetFemBasis,
    tets: &[[u32; 4]],
    fibers: &[FiberDirection],
    positions: &mut [Vec3],
    inv_mass: Option<&[f32]>,
    pinned: Option<&[bool]>,
    params: &AnisotropicProjectionParams,
) -> Option<AnisotropicProjectionReport> {
    // Work on a private copy so a mid-pass validation failure cannot leave the
    // caller's buffer half-projected.
    let mut work = positions.to_vec();

    let mut isotropic_report: Option<StrainLimitReport> = None;
    let mut fiber_report: Option<OrthotropicStrainLimitReport> = None;
    let mut volume_report: Option<VolumeProjectionReport> = None;

    for outer in 0..params.outer_iterations {
        let first = outer == 0;
        for stage in params.order {
            match stage {
                AnisotropicStage::IsotropicStrain => run_isotropic(
                    basis,
                    tets,
                    &mut work,
                    inv_mass,
                    pinned,
                    params,
                    first,
                    &mut isotropic_report,
                )?,
                AnisotropicStage::FiberStrain => run_fiber(
                    basis,
                    tets,
                    fibers,
                    &mut work,
                    inv_mass,
                    pinned,
                    params,
                    first,
                    &mut fiber_report,
                )?,
                AnisotropicStage::Volume => run_volume(
                    basis,
                    tets,
                    &mut work,
                    inv_mass,
                    pinned,
                    params,
                    first,
                    &mut volume_report,
                )?,
            }
        }
    }

    positions.copy_from_slice(&work);
    Some(AnisotropicProjectionReport {
        isotropic: isotropic_report,
        fiber: fiber_report,
        volume: volume_report,
        outer_iterations_run: params.outer_iterations,
    })
}

#[expect(
    clippy::too_many_arguments,
    reason = "an internal stage dispatcher threading the shared projection \
              inputs plus the first-iteration report sink; splitting it would \
              only duplicate the identical argument list"
)]
fn run_isotropic(
    basis: &TetFemBasis,
    tets: &[[u32; 4]],
    work: &mut [Vec3],
    inv_mass: Option<&[f32]>,
    pinned: Option<&[bool]>,
    params: &AnisotropicProjectionParams,
    capture: bool,
    sink: &mut Option<StrainLimitReport>,
) -> Option<()> {
    if let Some(p) = params.isotropic {
        let report = project_strain_limits(basis, tets, work, inv_mass, pinned, &p)?;
        if capture {
            *sink = Some(report);
        }
    }
    Some(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "an internal stage dispatcher threading the shared projection \
              inputs plus the fiber field and first-iteration report sink; \
              splitting it would only duplicate the identical argument list"
)]
fn run_fiber(
    basis: &TetFemBasis,
    tets: &[[u32; 4]],
    fibers: &[FiberDirection],
    work: &mut [Vec3],
    inv_mass: Option<&[f32]>,
    pinned: Option<&[bool]>,
    params: &AnisotropicProjectionParams,
    capture: bool,
    sink: &mut Option<OrthotropicStrainLimitReport>,
) -> Option<()> {
    if let Some(ref p) = params.fiber {
        let report =
            project_orthotropic_strain_limits(basis, tets, fibers, work, inv_mass, pinned, p)?;
        if capture {
            *sink = Some(report);
        }
    }
    Some(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "an internal stage dispatcher threading the shared projection \
              inputs plus the first-iteration report sink; splitting it would \
              only duplicate the identical argument list"
)]
fn run_volume(
    basis: &TetFemBasis,
    tets: &[[u32; 4]],
    work: &mut [Vec3],
    inv_mass: Option<&[f32]>,
    pinned: Option<&[bool]>,
    params: &AnisotropicProjectionParams,
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
    use crate::collider::tet_fem_anisotropic::FiberDirection;
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

    /// Family-major fibers: all elements of family 0 (X) then all of family 1 (Y).
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

    fn iso(min: f32, max: f32, iters: u32) -> StrainLimitParams {
        StrainLimitParams::new(min, max, iters).unwrap()
    }

    fn fib(pct: f32, families: usize, iters: u32) -> OrthotropicStrainLimitParams {
        OrthotropicStrainLimitParams::symmetric_uniform(pct, families, iters).unwrap()
    }

    fn vol(ratio: f32, stiffness: f32, iters: u32) -> VolumeProjectionParams {
        VolumeProjectionParams::new(ratio, stiffness, iters).unwrap()
    }

    #[test]
    fn rejects_invalid_params() {
        // All stages disabled.
        assert!(AnisotropicProjectionParams::with_default_order(None, None, None, 1).is_none());
        // Zero outer iterations.
        assert!(AnisotropicProjectionParams::with_default_order(
            Some(iso(0.9, 1.1, 1)),
            None,
            None,
            0
        )
        .is_none());
        // Non-permutation order (duplicate stage, missing Volume).
        let dup = [
            AnisotropicStage::IsotropicStrain,
            AnisotropicStage::IsotropicStrain,
            AnisotropicStage::FiberStrain,
        ];
        assert!(
            AnisotropicProjectionParams::new(Some(iso(0.9, 1.1, 1)), None, None, dup, 1).is_none()
        );
        // Valid permutation + one stage enabled.
        assert!(AnisotropicProjectionParams::with_default_order(
            Some(iso(0.9, 1.1, 1)),
            None,
            None,
            2
        )
        .is_some());
    }

    #[test]
    fn is_permutation_detects_valid_and_invalid() {
        assert!(is_permutation(&AnisotropicStage::ALL));
        assert!(is_permutation(&[
            AnisotropicStage::Volume,
            AnisotropicStage::FiberStrain,
            AnisotropicStage::IsotropicStrain,
        ]));
        assert!(!is_permutation(&[
            AnisotropicStage::Volume,
            AnisotropicStage::Volume,
            AnisotropicStage::IsotropicStrain,
        ]));
    }

    #[test]
    fn rejects_fiber_dimension_mismatch_leaves_positions_untouched() {
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let mut pos = verts.clone();
        inflate(&mut pos, 1.3);
        let snapshot = pos.clone();
        // 2 families, 2 elements => needs 4 fibers; give 3 => mismatch.
        let bad_fibers = vec![FiberDirection::new(Vec3::X).unwrap(); 3];
        let params =
            AnisotropicProjectionParams::with_default_order(None, Some(fib(0.1, 2, 2)), None, 1)
                .unwrap();
        let report = run_anisotropic_projection_pass(
            &basis,
            &tets,
            &bad_fibers,
            &mut pos,
            None,
            None,
            &params,
        );
        assert!(report.is_none());
        assert_eq!(pos, snapshot, "failed pass must not mutate positions");
    }

    #[test]
    fn rest_pose_is_untouched() {
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let fibers = warp_weft(tets.len());
        let mut pos = verts.clone();
        let params = AnisotropicProjectionParams::with_default_order(
            Some(iso(0.9, 1.1, 2)),
            Some(fib(0.1, 2, 2)),
            Some(vol(1.0, 1.0, 2)),
            2,
        )
        .unwrap();
        let before = pos.clone();
        let report =
            run_anisotropic_projection_pass(&basis, &tets, &fibers, &mut pos, None, None, &params)
                .unwrap();
        for (a, b) in pos.iter().zip(before.iter()) {
            assert!((*a - *b).length() < 1e-5, "rest pose should barely move");
        }
        assert_eq!(report.outer_iterations_run, 2);
    }

    #[test]
    fn disabled_fiber_stage_works_with_empty_fibers() {
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let mut pos = verts.clone();
        inflate(&mut pos, 1.4);
        let params = AnisotropicProjectionParams::with_default_order(
            Some(iso(0.95, 1.05, 4)),
            None,
            None,
            3,
        )
        .unwrap();
        let report =
            run_anisotropic_projection_pass(&basis, &tets, &[], &mut pos, None, None, &params)
                .unwrap();
        assert!(report.isotropic.is_some());
        assert!(report.fiber.is_none());
        assert!(report.volume.is_none());
    }

    /// Incoming (pre-projection) isotropic violation of `pos`, measured by a
    /// single-stage, single-iteration pass whose report captures the first
    /// iteration's violation before correction.
    fn iso_violation(basis: &TetFemBasis, tets: &[[u32; 4]], pos: &[Vec3]) -> f32 {
        let probe = AnisotropicProjectionParams::with_default_order(
            Some(iso(0.98, 1.02, 1)),
            None,
            None,
            1,
        )
        .unwrap();
        let mut work = pos.to_vec();
        run_anisotropic_projection_pass(basis, tets, &[], &mut work, None, None, &probe)
            .unwrap()
            .isotropic
            .unwrap()
            .max_violation
    }

    /// Incoming (pre-projection) fiber violation of `pos`.
    fn fib_violation(
        basis: &TetFemBasis,
        tets: &[[u32; 4]],
        fibers: &[FiberDirection],
        pos: &[Vec3],
    ) -> f32 {
        let probe =
            AnisotropicProjectionParams::with_default_order(None, Some(fib(0.02, 2, 1)), None, 1)
                .unwrap();
        let mut work = pos.to_vec();
        run_anisotropic_projection_pass(basis, tets, fibers, &mut work, None, None, &probe)
            .unwrap()
            .fiber
            .unwrap()
            .max_violation
    }

    #[test]
    fn combined_pass_reduces_isotropic_and_fiber_violation() {
        let (verts, tets) = single_tet();
        let basis = basis_of(&verts, &tets);
        let fibers = warp_weft(tets.len()); // 1 element, 2 families => 2 fibers.
        let mut pos = verts.clone();
        inflate(&mut pos, 1.5);

        // Incoming violations measured on the un-projected inflated pose.
        let iso_in = iso_violation(&basis, &tets, &pos);
        let fib_in = fib_violation(&basis, &tets, &fibers, &pos);
        assert!(
            iso_in > 0.0 && fib_in > 0.0,
            "inflation should violate both"
        );

        // Run the full interleaved pass, then re-measure residuals on the result.
        let params = AnisotropicProjectionParams::with_default_order(
            Some(iso(0.98, 1.02, 8)),
            Some(fib(0.02, 2, 8)),
            None,
            6,
        )
        .unwrap();
        run_anisotropic_projection_pass(&basis, &tets, &fibers, &mut pos, None, None, &params)
            .unwrap();

        let iso_out = iso_violation(&basis, &tets, &pos);
        let fib_out = fib_violation(&basis, &tets, &fibers, &pos);
        assert!(iso_out < iso_in, "iso {iso_out} should drop below {iso_in}");
        assert!(fib_out < fib_in, "fib {fib_out} should drop below {fib_in}");
    }

    #[test]
    fn preserves_pinned_vertices() {
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let fibers = warp_weft(tets.len());
        let mut pos = verts.clone();
        inflate(&mut pos, 1.3);
        let pinned = vec![true, false, false, false, false];
        let pinned_before = pos[0];
        let params = AnisotropicProjectionParams::with_default_order(
            Some(iso(0.95, 1.05, 4)),
            Some(fib(0.05, 2, 4)),
            Some(vol(1.0, 0.5, 4)),
            3,
        )
        .unwrap();
        run_anisotropic_projection_pass(
            &basis,
            &tets,
            &fibers,
            &mut pos,
            None,
            Some(&pinned),
            &params,
        )
        .unwrap();
        assert_eq!(pos[0], pinned_before, "pinned vertex must not move");
    }

    #[test]
    fn is_deterministic() {
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let fibers = warp_weft(tets.len());
        let params = AnisotropicProjectionParams::with_default_order(
            Some(iso(0.95, 1.05, 4)),
            Some(fib(0.05, 2, 4)),
            Some(vol(1.0, 0.5, 4)),
            3,
        )
        .unwrap();

        let mut a = verts.clone();
        inflate(&mut a, 1.35);
        let mut b = a.clone();
        run_anisotropic_projection_pass(&basis, &tets, &fibers, &mut a, None, None, &params)
            .unwrap();
        run_anisotropic_projection_pass(&basis, &tets, &fibers, &mut b, None, None, &params)
            .unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn order_permutation_is_accepted_and_runs() {
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let fibers = warp_weft(tets.len());
        let mut pos = verts.clone();
        inflate(&mut pos, 1.3);
        let order = [
            AnisotropicStage::Volume,
            AnisotropicStage::FiberStrain,
            AnisotropicStage::IsotropicStrain,
        ];
        let params = AnisotropicProjectionParams::new(
            Some(iso(0.95, 1.05, 4)),
            Some(fib(0.05, 2, 4)),
            Some(vol(1.0, 0.5, 4)),
            order,
            2,
        )
        .unwrap();
        let report =
            run_anisotropic_projection_pass(&basis, &tets, &fibers, &mut pos, None, None, &params)
                .unwrap();
        assert_eq!(report.outer_iterations_run, 2);
        assert!(report.isotropic.is_some() && report.fiber.is_some() && report.volume.is_some());
    }
}
