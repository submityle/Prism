//! Voxel cone-tracing GI CPU golden references.
//!
//! Deterministic, GPU-free voxel-cone-traced indirect lighting (Crassin 2011),
//! distinct from the SDF tracing in [`crate::gi::global_sdf`] and the surfel
//! cache in [`crate::gi::surface_cache`].
//!
//! * [`voxel`] — a mip-mapped opacity/radiance voxel grid with trilinear
//!   fetch and anisotropic-free isotropic mip pyramid construction.
//! * [`cone`] — cone marching with LOD = f(cone aperture, distance) and
//!   front-to-back opacity-weighted radiance accumulation.
//! * [`gather`] — diffuse (hemisphere cone set) and specular (single cone)
//!   gathering built on the cone marcher.
//!
//! The top-level [`trace_indirect`] stitches these together: given a voxelised
//! scene, a shading point, its normal, and a view vector, it returns the
//! indirect diffuse and specular radiance arriving at that point.
//!
//! # Conventions
//! * The grid must have its mip pyramid built ([`voxel::VoxelGrid::build_mips`])
//!   before gathering, so wide cones can read coarse LODs; an un-built grid
//!   simply reads level 0 everywhere.
//! * Diffuse and specular are gathered independently and returned separately in
//!   [`IndirectLight`]; callers weight them by their own BRDF terms.
//! * All results are non-negative and finite; every helper is a pure,
//!   deterministic CPU function with no RNG/IO/GPU/`unsafe`.

pub mod cone;
pub mod gather;
pub mod voxel;

pub use cone::{cone_diameter, lod_from_diameter, trace_cone, ConeConfig, ConeResult};
pub use gather::{
    gather_diffuse, gather_specular, reflection_direction, specular_aperture, GatherParams,
    MAX_SPEC_APERTURE, MIN_SPEC_APERTURE,
};
pub use voxel::{VoxelGrid, VoxelLevel, VoxelSample};

use bevy_math::Vec3;

/// Inputs to [`trace_indirect`]: how to gather plus the view/roughness needed
/// for the specular lobe.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IndirectParams {
    /// Shared diffuse/specular cone-marching parameters.
    pub gather: GatherParams,
    /// Outgoing view direction (surface -> camera) for the specular reflection.
    pub view: Vec3,
    /// Surface roughness in `[0, 1]` controlling the specular cone aperture.
    pub roughness: f32,
    /// Whether to gather the specular lobe at all.
    pub enable_specular: bool,
}

impl Default for IndirectParams {
    fn default() -> Self {
        Self {
            gather: GatherParams::default(),
            view: Vec3::Y,
            roughness: 0.5,
            enable_specular: true,
        }
    }
}

/// Indirect radiance arriving at a shading point, split by lobe.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IndirectLight {
    /// Hemispherical indirect diffuse radiance (non-negative RGB).
    pub diffuse: Vec3,
    /// Indirect specular radiance along the reflection cone (non-negative RGB).
    pub specular: Vec3,
}

impl IndirectLight {
    /// No indirect light.
    pub const ZERO: Self = Self {
        diffuse: Vec3::ZERO,
        specular: Vec3::ZERO,
    };
}

/// Gathers indirect diffuse (and optionally specular) radiance at `position`.
///
/// Combines [`gather::gather_diffuse`] with a single reflection cone from
/// [`gather::gather_specular`].  Both lobes are non-negative and finite; the
/// specular lobe is zero when `enable_specular` is `false`.
pub fn trace_indirect(
    grid: &VoxelGrid,
    position: Vec3,
    normal: Vec3,
    params: IndirectParams,
) -> IndirectLight {
    let diffuse = gather_diffuse(grid, position, normal, params.gather);
    let specular = if params.enable_specular {
        gather_specular(
            grid,
            position,
            normal,
            params.view,
            params.roughness,
            params.gather,
        )
    } else {
        Vec3::ZERO
    };
    IndirectLight {
        diffuse: diffuse.max(Vec3::ZERO),
        specular: specular.max(Vec3::ZERO),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::UVec3;

    /// Small scene: one bright, opaque voxel floating above a floor point.
    fn scene_with_emitter() -> (VoxelGrid, Vec3) {
        let mut grid = VoxelGrid::new(UVec3::splat(8), Vec3::ZERO, Vec3::splat(8.0));
        grid.set_voxel(UVec3::new(4, 6, 4), 1.0, Vec3::splat(10.0));
        grid.build_mips();
        // A floor point beneath the emitter, facing up.
        (grid, Vec3::new(4.5, 0.5, 4.5))
    }

    #[test]
    fn nearby_point_receives_indirect_light() {
        let (grid, pos) = scene_with_emitter();
        let params = IndirectParams {
            view: Vec3::Y,
            roughness: 0.4,
            enable_specular: true,
            ..IndirectParams::default()
        };
        let light = trace_indirect(&grid, pos, Vec3::Y, params);
        assert!(
            light.diffuse.length() > 0.0,
            "expected diffuse light, got {:?}",
            light.diffuse
        );
        assert!(light.diffuse.is_finite() && light.specular.is_finite());
    }

    #[test]
    fn empty_scene_returns_no_light() {
        let mut grid = VoxelGrid::new(UVec3::splat(8), Vec3::ZERO, Vec3::splat(8.0));
        grid.build_mips();
        let light = trace_indirect(
            &grid,
            Vec3::splat(4.0),
            Vec3::Y,
            IndirectParams::default(),
        );
        assert!(light.diffuse.length() < 1.0e-6, "{:?}", light.diffuse);
        assert!(light.specular.length() < 1.0e-6, "{:?}", light.specular);
    }

    #[test]
    fn disabling_specular_zeroes_that_lobe() {
        let (grid, pos) = scene_with_emitter();
        let params = IndirectParams {
            enable_specular: false,
            ..IndirectParams::default()
        };
        let light = trace_indirect(&grid, pos, Vec3::Y, params);
        assert_eq!(light.specular, Vec3::ZERO);
    }

    #[test]
    fn light_is_non_negative_everywhere() {
        let (grid, _) = scene_with_emitter();
        for &p in &[
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(4.5, 2.0, 4.5),
            Vec3::new(7.0, 7.0, 7.0),
        ] {
            let light = trace_indirect(&grid, p, Vec3::Y, IndirectParams::default());
            assert!(light.diffuse.x >= 0.0 && light.diffuse.y >= 0.0 && light.diffuse.z >= 0.0);
            assert!(light.specular.x >= 0.0 && light.specular.y >= 0.0 && light.specular.z >= 0.0);
        }
    }
}
