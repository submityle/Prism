//! Authoring → routing assembly: project a validated [`CompiledWaterParams`]
//! snapshot onto the runtime values the per-frame pipeline actually routes.
//!
//! [`asset::WaterBodyAsset::compile`](super::asset::WaterBodyAsset::compile)
//! produces a flat, validated [`CompiledWaterParams`] — the *schema* half of the
//! data-driven authoring path (design §2, "编译为求解器参数 + `WESL`
//! specialization key"). This module closes the *assembly* half by mapping that
//! snapshot onto the two values the runtime consumes:
//!
//! 1. the coarse [`WaterBody`] routing contract read by
//!    [`pipeline::extract`](super::pipeline::extract), and
//! 2. the [`WaterSpecializationKey`] the `GPU` dispatch path specializes on.
//!
//! # What is and is not projected
//!
//! [`WaterBody`] is deliberately a *routing* contract (design doc §2 on
//! [`WaterBody`]): it carries the geometry class, solver bucket, frontend, grid
//! sizing, domain bounds, still-water level, and the aggregate
//! [`WaterSimProfile`] the `CPU` planners read. Fine-grained per-solver inputs
//! that only the `GPU` kernels or the specialization key consume are **not**
//! duplicated into the routing contract; they stay in the retained
//! [`CompiledWaterParams`] and the [`WaterSpecializationKey`] this assembly
//! returns alongside the body. Concretely:
//!
//! - **Projected into the profile**: fluid density (`PBF` rest density + the
//!   coupling fluid density), the spectral optics (`IOR`→Cauchy `A`, dispersion
//!   coefficient→Cauchy `B`, per-channel `Beer-Lambert` extinction, scattering
//!   albedo, underwater visibility floor), and foam persistence.
//! - **Carried only in the returned key / retained params**: viscosity, wave
//!   steepness, directional spread, foam/breaking thresholds, caustic intensity,
//!   and the particle cap. These drive the `GPU` solver specialization keyed by
//!   [`WaterSpecializationKey`]; the particle cap is also arbitrated per frame by
//!   [`budget::plan_water`](super::budget::plan_water) rather than pinned into
//!   the routing contract. Nothing authored is dropped — it is simply routed
//!   through the key instead of being misfiled into a `CPU` profile slice that
//!   has no matching knob.
//!
//! Every function here is a pure, deterministic projection, so the assembled
//! body and key are reproducible on any thread.

use super::asset::{CompiledWaterParams, WaterSpecializationKey};
use super::profile::WaterSimProfile;
use super::underwater::RgbExtinction;
use super::{WaterBody, WaterBodyHandle};
use crate::deformation::DeformationHandle;

/// The runtime values a [`CompiledWaterParams`] snapshot assembles into: the
/// coarse [`WaterBody`] routing contract plus the `GPU` specialization key the
/// dispatch path keys on.
///
/// Kept together because the pipeline needs both from one authored asset: the
/// body routes the `CPU` planners, and the key selects the specialized `GPU`
/// kernel permutation. Returning them as one `Copy` value keeps the two in
/// lockstep with the asset they came from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AssembledWaterBody {
    /// The coarse routing contract consumed by the per-frame pipeline.
    pub body: WaterBody,
    /// The `WESL` specialization key the `GPU` kernels specialize on.
    pub spec_key: WaterSpecializationKey,
}

impl CompiledWaterParams {
    /// Project this validated snapshot onto the runtime [`WaterBody`] routing
    /// contract and the `GPU` specialization key.
    ///
    /// `handle` is the stable body identity and `deformation` is the shared
    /// deformation-cache slot the displacement mesh / height field writes into;
    /// both are assigned by the host that owns the scene, so they are supplied
    /// by the caller rather than invented here.
    ///
    /// The returned [`WaterSimProfile`] starts from
    /// [`WaterSimProfile::physical_water`] and overrides exactly the slices this
    /// snapshot authoritatively owns (density, optics, foam persistence); every
    /// other planner knob keeps the coherent clear-water default so an author
    /// who only sets physical/optical fields still gets a consistent profile.
    #[must_use]
    pub fn assemble(
        &self,
        handle: WaterBodyHandle,
        deformation: DeformationHandle,
    ) -> AssembledWaterBody {
        let mut profile = WaterSimProfile::physical_water();

        // Density is shared by the incompressible PBF rest density and the
        // two-way coupling fluid density; both must agree with the authored
        // value so buoyancy and pressure stay consistent.
        profile.sim.pbf.rest_density = self.density;
        profile.coupling.fluid_density = self.density;

        // Spectral optics: the compiled IOR is the Cauchy baseline `A`, the
        // dispersion coefficient is the Cauchy `B` term, and extinction maps
        // per channel. Scattering is an albedo, so it is clamped to the unit
        // range even though the compiler only guaranteed it non-negative.
        profile.optics.cauchy_a = self.ior;
        profile.optics.cauchy_b = self.dispersion_coeff;
        profile.optics.extinction = RgbExtinction {
            r: self.extinction.x,
            g: self.extinction.y,
            b: self.extinction.z,
        };
        profile.optics.scatter_albedo = self.scattering.min(1.0);
        profile.optics.visibility_threshold = self.visibility;

        // Foam persistence floors the semi-Lagrangian foam decay.
        profile.surface_fx.foam.persistence_floor = self.foam_persistence;

        let body = WaterBody {
            handle,
            kind: self.kind,
            solver: self.solver,
            frontend: self.frontend,
            deformation,
            grid_resolution: self.grid_resolution,
            cascade_count: self.cascade_count,
            domain_half_extent: self.domain_half_extent,
            still_water_level: self.still_water_level,
            profile,
        };

        AssembledWaterBody {
            body,
            spec_key: self.spec_key,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::asset::WaterBodyAsset;
    use super::super::EPS;
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= EPS
    }

    #[test]
    fn ocean_assembly_routes_identity_and_geometry() {
        let compiled = WaterBodyAsset::ocean_default(extent())
            .compile()
            .expect("ocean preset compiles");
        let out = compiled.assemble(WaterBodyHandle(7), DeformationHandle(3));

        assert_eq!(out.body.handle, WaterBodyHandle(7));
        assert_eq!(out.body.deformation, DeformationHandle(3));
        assert_eq!(out.body.kind, compiled.kind);
        assert_eq!(out.body.solver, compiled.solver);
        assert_eq!(out.body.frontend, compiled.frontend);
        assert_eq!(out.body.grid_resolution, compiled.grid_resolution);
        assert_eq!(out.body.cascade_count, compiled.cascade_count);
        assert_eq!(out.body.domain_half_extent, compiled.domain_half_extent);
        assert!(approx(
            out.body.still_water_level,
            compiled.still_water_level
        ));
    }

    #[test]
    fn density_feeds_pbf_and_coupling() {
        let mut asset = WaterBodyAsset::volume_default(extent(), 4096);
        asset.physical.density = 1024.0;
        let compiled = asset.compile().expect("volume preset compiles");
        let out = compiled.assemble(WaterBodyHandle(0), DeformationHandle(0));

        assert!(approx(out.body.profile.sim.pbf.rest_density, 1024.0));
        assert!(approx(out.body.profile.coupling.fluid_density, 1024.0));
    }

    #[test]
    fn optics_slice_is_projected() {
        let mut asset = WaterBodyAsset::ocean_default(extent());
        asset.physical.ior = 1.34;
        asset.physical.dispersion_coeff = 0.0041;
        asset.physical.extinction = super::super::Vec3::new(0.5, 0.2, 0.1);
        asset.physical.scattering = 0.6;
        asset.shading.visibility = 0.03;
        let compiled = asset.compile().expect("ocean preset compiles");
        let out = compiled.assemble(WaterBodyHandle(1), DeformationHandle(1));
        let optics = out.body.profile.optics;

        assert!(approx(optics.cauchy_a, 1.34));
        assert!(approx(optics.cauchy_b, 0.0041));
        assert!(approx(optics.extinction.r, 0.5));
        assert!(approx(optics.extinction.g, 0.2));
        assert!(approx(optics.extinction.b, 0.1));
        assert!(approx(optics.scatter_albedo, 0.6));
        assert!(approx(optics.visibility_threshold, 0.03));
    }

    #[test]
    fn scattering_albedo_is_clamped_to_unit_range() {
        let mut asset = WaterBodyAsset::ocean_default(extent());
        asset.physical.scattering = 5.0;
        let compiled = asset.compile().expect("ocean preset compiles");
        let out = compiled.assemble(WaterBodyHandle(0), DeformationHandle(0));

        assert!(approx(out.body.profile.optics.scatter_albedo, 1.0));
    }

    #[test]
    fn foam_persistence_floors_decay() {
        let mut asset = WaterBodyAsset::ocean_default(extent());
        asset.shading.foam_persistence = 0.37;
        let compiled = asset.compile().expect("ocean preset compiles");
        let out = compiled.assemble(WaterBodyHandle(0), DeformationHandle(0));

        assert!(approx(
            out.body.profile.surface_fx.foam.persistence_floor,
            0.37
        ));
    }

    #[test]
    fn specialization_key_is_preserved() {
        let compiled = WaterBodyAsset::ocean_default(extent())
            .compile()
            .expect("ocean preset compiles");
        let out = compiled.assemble(WaterBodyHandle(0), DeformationHandle(0));

        assert_eq!(out.spec_key, compiled.spec_key);
        assert_eq!(out.spec_key.bits(), compiled.spec_key.bits());
    }

    #[test]
    fn frontend_is_carried_through() {
        let mut asset = WaterBodyAsset::river_default(extent());
        asset.shading.frontend = super::super::ShadingFrontend::Npr;
        let compiled = asset.compile().expect("river preset compiles");
        let out = compiled.assemble(WaterBodyHandle(0), DeformationHandle(0));

        assert_eq!(out.body.frontend, super::super::ShadingFrontend::Npr);
    }

    #[test]
    fn untouched_profile_slices_keep_defaults() {
        let compiled = WaterBodyAsset::ocean_default(extent())
            .compile()
            .expect("ocean preset compiles");
        let out = compiled.assemble(WaterBodyHandle(0), DeformationHandle(0));
        let base = WaterSimProfile::physical_water();

        // The shoreline and shading slices carry no compiled override, so they
        // must match the coherent clear-water default verbatim.
        assert_eq!(out.body.profile.shoreline, base.shoreline);
        assert_eq!(out.body.profile.shading, base.shading);
    }

    #[test]
    fn assembly_is_deterministic() {
        let compiled = WaterBodyAsset::ocean_default(extent())
            .compile()
            .expect("ocean preset compiles");
        let a = compiled.assemble(WaterBodyHandle(2), DeformationHandle(5));
        let b = compiled.assemble(WaterBodyHandle(2), DeformationHandle(5));
        assert_eq!(a, b);
    }

    fn extent() -> super::super::Vec3 {
        super::super::Vec3::new(256.0, 32.0, 256.0)
    }
}
