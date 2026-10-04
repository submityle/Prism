//! One-call authoring of a validated granular scene.
//!
//! Setting up a discrete-element experiment means stitching together several
//! independent pieces: a grain-size distribution, a non-overlapping packing of
//! that distribution inside a box, a container made of inward-facing
//! half-spaces, an optional gravity-settling pass so the bed starts at rest,
//! and a diagnostic census of the result. Doing that by hand is repetitive and
//! easy to get subtly wrong (mismatched boxes, a container that does not
//! enclose the packing, diagnostics computed against the wrong extent).
//!
//! [`GranularScene`] composes the existing building blocks into a single
//! validated setup:
//!
//! * [`pack_spheres_from_distribution`](crate::collider::distribution_packing::pack_spheres_from_distribution)
//!   fills the packing box,
//! * [`closed_box`](crate::collider::boundary_container::closed_box) /
//!   [`open_top_box`](crate::collider::boundary_container::open_top_box) build
//!   the container from the *same* box,
//! * [`GravitySettler`](crate::collider::gravity_settle::GravitySettler)
//!   optionally relaxes the loose bed to rest, and
//! * [`PackingDiagnostics`](crate::collider::packing_diagnostics::PackingDiagnostics)
//!   measures solid volume, packing fraction, and the coordination census of
//!   the final arrangement.
//!
//! Both constructors are pure and deterministic: identical parameters always
//! yield an identical scene.

use glam::Vec3;

use crate::collider::boundary_container::{closed_box, open_top_box};
use crate::collider::distribution_packing::{
    pack_spheres_from_distribution, DistributionPackingParams,
};
use crate::collider::grain_size_distribution::GrainSizeDistribution;
use crate::collider::gravity_settle::{GravitySettleParams, GravitySettleReport, GravitySettler};
use crate::collider::hertz_contact::HertzModel;
use crate::collider::packing_diagnostics::PackingDiagnostics;
use crate::collider::rotational_boundary_contact::HalfSpace;
use crate::collider::sphere_boundary_driver::SphereBoundaryDriver;
use crate::collider::sphere_cundall_strack_driver::SphereCundallStrackDriver;
use crate::collider::sphere_packing::SpherePacking;
use crate::collider::tangential_history_contact::CundallStrackModel;

/// Which container the scene wraps around the packing box.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContainerKind {
    /// A fully closed box (six inward faces).
    ClosedBox,
    /// A box open at its `+Z` top (five inward faces) so grains can rain in.
    OpenTopBox,
}

impl ContainerKind {
    /// Builds the container half-spaces for this kind over the box
    /// `[min, max]`. Returns `None` for a degenerate box.
    #[must_use]
    fn build(self, min: Vec3, max: Vec3) -> Option<Vec<HalfSpace>> {
        match self {
            ContainerKind::ClosedBox => closed_box(min, max),
            ContainerKind::OpenTopBox => open_top_box(min, max),
        }
    }
}

/// Parameters that fully describe a granular scene before authoring.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GranularSceneParams {
    /// Packing-box extent and dart-throwing budget for the bed.
    pub packing: DistributionPackingParams,
    /// Container wrapped around `packing`'s box.
    pub container: ContainerKind,
    /// Surface-gap tolerance used by the coordination census, in metres.
    pub contact_tolerance: f32,
}

impl GranularSceneParams {
    /// Builds a validated parameter set.
    ///
    /// Returns `None` unless `contact_tolerance` is finite and non-negative. The
    /// `packing` fields are validated downstream by the packer when the scene is
    /// authored.
    #[must_use]
    pub fn new(
        packing: DistributionPackingParams,
        container: ContainerKind,
        contact_tolerance: f32,
    ) -> Option<Self> {
        if !(contact_tolerance.is_finite() && contact_tolerance >= 0.0) {
            return None;
        }
        Some(Self {
            packing,
            container,
            contact_tolerance,
        })
    }
}

/// A fully composed, validated granular scene.
///
/// Build one with [`GranularScene::author`] for a loose (as-packed) bed, or
/// [`GranularScene::author_settled`] to additionally relax it to rest under
/// gravity. All accessors describe the *final* arrangement, so after settling
/// the diagnostics reflect the rested positions.
pub struct GranularScene {
    packing: SpherePacking,
    boundaries: Vec<HalfSpace>,
    diagnostics: PackingDiagnostics,
    box_min: Vec3,
    box_max: Vec3,
    contact_tolerance: f32,
    settle_report: Option<GravitySettleReport>,
}

impl GranularScene {
    /// Authors a loose scene: packs the distribution, builds the container, and
    /// runs the diagnostic census on the as-packed bed.
    ///
    /// Returns `None` if the packer rejects `params.packing`, if the container
    /// box is degenerate, or if the diagnostics reject the packing.
    #[must_use]
    pub fn author(
        params: &GranularSceneParams,
        distribution: &GrainSizeDistribution,
    ) -> Option<Self> {
        let packing = pack_spheres_from_distribution(&params.packing, distribution)?;
        Self::finish(params, packing, None)
    }

    /// Authors a scene and relaxes it to rest.
    ///
    /// The container drives the boundary reaction during settling, so the rested
    /// bed is consistent with the walls the scene reports. `grain_model` governs
    /// grain–grain contacts, `boundary_model` governs grain–wall contacts, and
    /// `settle_params` controls the integration. Returns `None` under the same
    /// conditions as [`GranularScene::author`], or if settling fails.
    #[must_use]
    pub fn author_settled(
        params: &GranularSceneParams,
        distribution: &GrainSizeDistribution,
        grain_model: CundallStrackModel,
        boundary_model: HertzModel,
        settle_params: &GravitySettleParams,
    ) -> Option<Self> {
        let packing = pack_spheres_from_distribution(&params.packing, distribution)?;
        let boundaries = params
            .container
            .build(params.packing.min, params.packing.max)?;

        let boundary_driver = SphereBoundaryDriver::with_boundaries(boundary_model, boundaries);
        let settler =
            GravitySettler::new(SphereCundallStrackDriver::new(grain_model), boundary_driver);
        let (settled, report) = settler.settle(&packing, settle_params)?;
        Self::finish(params, settled, Some(report))
    }

    /// Shared tail: rebuild the container, census the final packing, and store.
    #[must_use]
    fn finish(
        params: &GranularSceneParams,
        packing: SpherePacking,
        settle_report: Option<GravitySettleReport>,
    ) -> Option<Self> {
        let boundaries = params
            .container
            .build(params.packing.min, params.packing.max)?;
        let diagnostics = PackingDiagnostics::analyze(
            packing.positions(),
            packing.radii(),
            params.contact_tolerance,
        )?;
        Some(Self {
            packing,
            boundaries,
            diagnostics,
            box_min: params.packing.min,
            box_max: params.packing.max,
            contact_tolerance: params.contact_tolerance,
            settle_report,
        })
    }

    /// The final sphere bed.
    #[must_use]
    pub fn packing(&self) -> &SpherePacking {
        &self.packing
    }

    /// The container half-spaces (inward normals).
    #[must_use]
    pub fn boundaries(&self) -> &[HalfSpace] {
        &self.boundaries
    }

    /// The diagnostic census of the final bed.
    #[must_use]
    pub fn diagnostics(&self) -> &PackingDiagnostics {
        &self.diagnostics
    }

    /// Lower corner of the packing box.
    #[must_use]
    pub fn box_min(&self) -> Vec3 {
        self.box_min
    }

    /// Upper corner of the packing box.
    #[must_use]
    pub fn box_max(&self) -> Vec3 {
        self.box_max
    }

    /// Surface-gap tolerance used by the coordination census.
    #[must_use]
    pub fn contact_tolerance(&self) -> f32 {
        self.contact_tolerance
    }

    /// Solid-volume fraction of the bed within its packing box, or `None` for a
    /// degenerate box.
    #[must_use]
    pub fn packing_fraction(&self) -> Option<f32> {
        self.diagnostics
            .packing_fraction(self.box_min, self.box_max)
    }

    /// The settling report, present only when built with
    /// [`GranularScene::author_settled`].
    #[must_use]
    pub fn settle_report(&self) -> Option<&GravitySettleReport> {
        self.settle_report.as_ref()
    }

    /// `true` if the scene was relaxed to rest.
    #[must_use]
    pub fn is_settled(&self) -> bool {
        self.settle_report.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn distribution() -> GrainSizeDistribution {
        GrainSizeDistribution::uniform(0.02, 0.03).unwrap()
    }

    fn packing_params(target_count: usize) -> DistributionPackingParams {
        DistributionPackingParams {
            min: Vec3::new(-0.15, -0.15, 0.0),
            max: Vec3::new(0.15, 0.15, 0.4),
            target_count,
            max_attempts: target_count * 200,
            separation: 0.0,
            seed: 7,
        }
    }

    fn scene_params(container: ContainerKind, target_count: usize) -> GranularSceneParams {
        GranularSceneParams::new(packing_params(target_count), container, 1.0e-3).unwrap()
    }

    #[test]
    fn rejects_invalid_contact_tolerance() {
        assert!(
            GranularSceneParams::new(packing_params(10), ContainerKind::ClosedBox, -1.0,).is_none()
        );
        assert!(
            GranularSceneParams::new(packing_params(10), ContainerKind::ClosedBox, f32::NAN,)
                .is_none()
        );
    }

    #[test]
    fn author_builds_a_consistent_loose_scene() {
        let params = scene_params(ContainerKind::ClosedBox, 24);
        let scene = GranularScene::author(&params, &distribution()).expect("scene authored");

        assert!(!scene.packing().is_empty());
        assert_eq!(scene.diagnostics().grain_count(), scene.packing().len());
        assert_eq!(scene.boundaries().len(), 6);
        assert!(!scene.is_settled());
        assert!(scene.settle_report().is_none());

        let phi = scene.packing_fraction().expect("finite fraction");
        assert!(phi > 0.0 && phi < 1.0, "packing fraction = {phi}");
    }

    #[test]
    fn container_kind_sets_face_count() {
        let closed =
            GranularScene::author(&scene_params(ContainerKind::ClosedBox, 12), &distribution())
                .unwrap();
        assert_eq!(closed.boundaries().len(), 6);
        let open = GranularScene::author(
            &scene_params(ContainerKind::OpenTopBox, 12),
            &distribution(),
        )
        .unwrap();
        assert_eq!(open.boundaries().len(), 5);
    }

    #[test]
    fn author_rejects_invalid_packing_params() {
        let mut packing = packing_params(0); // zero target count -> packer rejects
        packing.max_attempts = 0;
        let params = GranularSceneParams::new(packing, ContainerKind::ClosedBox, 1.0e-3).unwrap();
        assert!(GranularScene::author(&params, &distribution()).is_none());
    }

    #[test]
    fn author_is_deterministic() {
        let params = scene_params(ContainerKind::ClosedBox, 20);
        let a = GranularScene::author(&params, &distribution()).unwrap();
        let b = GranularScene::author(&params, &distribution()).unwrap();
        assert_eq!(a.packing().positions(), b.packing().positions());
        assert_eq!(a.packing().radii(), b.packing().radii());
    }

    fn grain_model() -> CundallStrackModel {
        CundallStrackModel::new(2.0e3, 2.0, 2.0e3, 2.0, 0.5).unwrap()
    }

    fn boundary_model() -> HertzModel {
        HertzModel::new(1.0e6, 0.3, 5.0, 5.0, 0.5).unwrap()
    }

    #[test]
    fn author_settled_relaxes_the_bed() {
        let params = scene_params(ContainerKind::OpenTopBox, 10);
        let settle = GravitySettleParams::earth(1.0e3, 1.0e-4, 20_000, 5.0e-3, 1.0, 0.999).unwrap();
        let scene = GranularScene::author_settled(
            &params,
            &distribution(),
            grain_model(),
            boundary_model(),
            &settle,
        )
        .expect("settled scene authored");

        assert!(scene.is_settled());
        let report = scene.settle_report().expect("report present");
        assert!(report.iterations() > 0);
        // Settling preserves the grain population and the container.
        assert_eq!(scene.diagnostics().grain_count(), scene.packing().len());
        assert_eq!(scene.boundaries().len(), 5);
        // Every grain stays inside the box floor (open-top box floor at z = 0).
        let floor_z = scene.box_min().z;
        for (p, &r) in scene
            .packing()
            .positions()
            .iter()
            .zip(scene.packing().radii())
        {
            assert!(
                p.z > floor_z - 0.1 * r,
                "grain sank through the floor: z = {}, r = {r}",
                p.z
            );
        }
    }

    #[test]
    fn settling_compacts_relative_to_the_loose_bed() {
        let params = scene_params(ContainerKind::OpenTopBox, 16);
        let loose = GranularScene::author(&params, &distribution()).unwrap();
        let settle = GravitySettleParams::earth(1.0e3, 1.0e-4, 20_000, 5.0e-3, 1.0, 0.999).unwrap();
        let settled = GranularScene::author_settled(
            &params,
            &distribution(),
            grain_model(),
            boundary_model(),
            &settle,
        )
        .unwrap();

        // The settled bed sits lower on average than the as-packed cloud.
        let mean_z = |s: &GranularScene| {
            let ps = s.packing().positions();
            ps.iter().map(|p| p.z).sum::<f32>() / ps.len() as f32
        };
        assert!(
            mean_z(&settled) < mean_z(&loose),
            "settling should lower the bed: loose {}, settled {}",
            mean_z(&loose),
            mean_z(&settled)
        );
    }
}
