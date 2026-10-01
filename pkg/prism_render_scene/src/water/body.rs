//! The main-world water body component and its per-frame render-world snapshot.
//!
//! A [`WaterBody`] is the authored `CPU` description a game spawns on an entity:
//! the ocean spectrum amplitude pair and analytic `Gerstner` wave train, the
//! `FLIP`/`APIC` particle pool, the `PBF` positions, the spray emitters, the
//! Shallow-Water and foam source fields, the caustics photon histogram, the
//! two-way coupling queries, every per-pass `#[repr(C)]` uniform, the resident
//! element counts and texture extents, and the set of live passes with their
//! substep / iteration / cascade loop counts. It is the single main-world
//! source the extract stage snapshots into the render world each frame.
//!
//! [`ExtractedWater`] is the render-world resource that snapshot lands in. The
//! extract system clears and refills it every frame (mirroring the cloth and
//! lighting extracts), so the prepare stage always sees the current bodies and
//! an entity that despawns simply stops contributing.
//!
//! The split of responsibilities mirrors cloth exactly: this component owns the
//! host state and exposes two borrowing views — [`WaterBody::as_extract`]
//! produces the float-free
//! [`WaterGpuExtract`](prism_render_architecture::water::gpu::pipeline::WaterGpuExtract)
//! the golden `prepare` expands into the ordered dispatch schedule, and
//! [`WaterBody::as_upload`] produces the
//! [`WaterBodyUpload`](super::bind_groups::WaterBodyUpload) the device factory
//! turns into resident buffers and the twelve bind groups. Neither view copies
//! the owned pools.

use bevy_ecs::component::Component;
use bevy_ecs::resource::Resource;

use prism_render_architecture::water::gpu::buffers::WaterBufferCounts;
use prism_render_architecture::water::gpu::pipeline::{
    extract, mac_face_count, WaterGpuExtract, WaterPasses,
};

use super::abi::{
    GpuFlipParticle, GpuFlipSimParams, GpuFlipSurfaceParams, GpuGerstnerWave, GpuPbfParams,
    GpuSprayParams, GpuSpraySource, GpuWaterCausticsParams, GpuWaterCouplingParams,
    GpuWaterCouplingQuery, GpuWaterDispersionParams, GpuWaterFoamParams, GpuWaterGerstnerParams,
    GpuWaterSpectrumParams, GpuWaterSurfaceMeshParams, GpuWaterSweParams, GpuWaterUnderwaterParams,
    GpuWaterWaterlineParams, GpuWaterWetnessParams,
};
use super::bind_groups::{WaterBodyUpload, WaterSurfaceExtent, WaterVolumeExtent};

/// The authored `CPU` state of one water body, spawned on a main-world entity.
///
/// The owned pools are the `#[repr(C)]` mirrors the shaders read directly, so
/// no per-frame repacking of solver types happens here; the author (or an
/// upstream authoring system) fills them. The `passes` record and the loop
/// counts drive the golden dispatch schedule; the `counts` and extents size the
/// resident device resources. A default body has no live pass and no pool, so
/// it produces an empty schedule and contributes no resident body — an honest
/// no-op rather than a fabricated solve.
#[derive(Component, Clone, Debug, Default, PartialEq)]
pub struct WaterBody {
    // -- Ocean: spectral IFFT + analytic Gerstner --------------------------
    /// Initial `Tessendorf` spectrum amplitudes `h0` (`array<vec2<f32>>`).
    pub(crate) spectrum_h0: Vec<[f32; 2]>,
    /// Conjugate spectrum `h0(-k)` (`array<vec2<f32>>`).
    pub(crate) spectrum_h0_neg: Vec<[f32; 2]>,
    /// Per-frame spectrum scalars (the ocean group's direct-sum reference).
    pub(crate) spectrum_params: GpuWaterSpectrumParams,
    /// Per-cascade spectral uniforms, one per stacked atlas tile; production's
    /// `spectrum_fft` groups bind these in cascade order.
    pub(crate) cascade_params: Vec<GpuWaterSpectrumParams>,
    /// Analytic `Gerstner` wave trains; empty when the body is spectrum-only.
    pub(crate) gerstner_waves: Vec<GpuGerstnerWave>,
    /// Per-frame `Gerstner` scalars.
    pub(crate) gerstner_params: GpuWaterGerstnerParams,
    /// Displacement / normal storage-texture extent (`N x N` cascade atlas).
    pub(crate) ocean_extent: WaterSurfaceExtent,

    // -- Volume: FLIP / APIC ------------------------------------------------
    /// `FLIP`/`APIC` particle pool; empty when the body carries no volume sim.
    pub(crate) flip_particles: Vec<GpuFlipParticle>,
    /// Number of `MAC` grid scalar cells (sizes the scatter/pressure buffers).
    pub(crate) flip_grid_cells: u32,
    /// `FLIP` per-substep uniform.
    pub(crate) flip_params: GpuFlipSimParams,
    /// `FLIP` surface-reconstruction uniform.
    pub(crate) flip_surface_params: GpuFlipSurfaceParams,
    /// Screen-space surface reconstruction extent (`surface_normal_tex`).
    pub(crate) flip_surface_extent: WaterSurfaceExtent,

    // -- PBF ---------------------------------------------------------------
    /// `PBF` particle positions (`array<vec4<f32>>`); ping-ponged in-solve.
    pub(crate) pbf_positions: Vec<[f32; 4]>,
    /// Number of `PBF` spatial-hash entries (one `u32` per hash slot).
    pub(crate) pbf_hash_entries: u32,
    /// `PBF` density-solve uniform.
    pub(crate) pbf_params: GpuPbfParams,

    // -- Spray -------------------------------------------------------------
    /// Spray emitter sources; empty when the body sheds no spray.
    pub(crate) spray_sources: Vec<GpuSpraySource>,
    /// Spray-pool particle capacity (sizes the spawn append buffer).
    pub(crate) spray_capacity: u32,
    /// Spray-emit uniform.
    pub(crate) spray_params: GpuSprayParams,

    // -- Shallow Water -----------------------------------------------------
    /// Number of Shallow-Water grid cells (height / velocity field).
    pub(crate) swe_cells: u32,
    /// Shallow-Water source injections (`array<vec4<f32>>`).
    pub(crate) swe_sources: Vec<[f32; 4]>,
    /// Shallow-Water step uniform.
    pub(crate) swe_params: GpuWaterSweParams,

    // -- Foam --------------------------------------------------------------
    /// Number of foam coverage cells (semi-Lagrangian, double-buffered).
    pub(crate) foam_cells: u32,
    /// Reactive foam source field (`array<f32>`).
    pub(crate) foam_sources: Vec<f32>,
    /// Foam advection uniform.
    pub(crate) foam_params: GpuWaterFoamParams,

    // -- Waterline ---------------------------------------------------------
    /// Number of waterline samples (sizes the sample / output arrays).
    pub(crate) waterline_samples: u32,
    /// Waterline mask uniform.
    pub(crate) waterline_params: GpuWaterWaterlineParams,

    // -- Caustics ----------------------------------------------------------
    /// Photon-count bins projected onto the caustics grid (`array<u32>`).
    pub(crate) caustics_photon_counts: Vec<u32>,
    /// Caustics projection uniform.
    pub(crate) caustics_params: GpuWaterCausticsParams,
    /// Caustics `r32float` projection-target extent.
    pub(crate) caustics_extent: WaterSurfaceExtent,
    /// Sampled photon-offset texture extent.
    pub(crate) caustics_offset_extent: WaterSurfaceExtent,

    // -- Dispersion --------------------------------------------------------
    /// Dispersion refraction uniform.
    pub(crate) dispersion_params: GpuWaterDispersionParams,
    /// Dispersion `rgba16float` output + sampled scene / normal extent.
    pub(crate) dispersion_extent: WaterSurfaceExtent,

    // -- Underwater volume -------------------------------------------------
    /// Underwater single-scatter uniform.
    pub(crate) underwater_params: GpuWaterUnderwaterParams,
    /// Underwater `rgba16float` `3D` froxel-volume extent.
    pub(crate) underwater_extent: WaterVolumeExtent,
    /// Sampled surface-light texture extent.
    pub(crate) underwater_light_extent: WaterSurfaceExtent,

    // -- Wetness -----------------------------------------------------------
    /// Number of surface wetness cells (`array<vec2<f32>>` state).
    pub(crate) wetness_cells: u32,
    /// Wetness step uniform.
    pub(crate) wetness_params: GpuWaterWetnessParams,
    /// Wetness `rgba16float` output extent.
    pub(crate) wetness_extent: WaterSurfaceExtent,

    // -- Coupling ----------------------------------------------------------
    /// Two-way coupling buoyancy queries (`array<WaterCouplingQuery>`).
    pub(crate) coupling_queries: Vec<GpuWaterCouplingQuery>,
    /// Number of coupling read-back rows (`array<vec4<f32>>`).
    pub(crate) coupling_readback_rows: u32,
    /// Coupling read-back uniform.
    pub(crate) coupling_params: GpuWaterCouplingParams,

    // -- Schedule drivers --------------------------------------------------
    /// The resident element counts that size the golden persistent buffer set
    /// (used for the frame's byte accounting in the aggregate queue).
    pub(crate) counts: WaterBufferCounts,
    /// The set of live passes this frame; drives which dispatches emit.
    pub(crate) passes: WaterPasses,
    /// Number of spectral cascades to transform (one dispatch each).
    pub(crate) ocean_cascades: u32,
    /// Simulation substeps per frame.
    pub(crate) substeps: u32,
    /// Projection iterations per substep for the pressure / density solves.
    pub(crate) solver_iterations: u32,
    /// Invocation count for one spectral cascade (spectrum texels).
    pub(crate) spectrum_texels: u32,
    /// Invocation count for the `2D` grid passes (`SWE`/`Gerstner`/foam/
    /// waterline height-field texels).
    pub(crate) grid2d_texels: u32,
    /// Invocation count for the `3D` grid passes (`FLIP` `MAC` voxels,
    /// underwater froxels).
    pub(crate) grid3d_voxels: u32,
    /// Invocation count for the particle-domain passes (`PBF`/spray particles).
    pub(crate) particle_count: u32,
    /// Invocation count for the full-screen passes (reconstruction, caustics,
    /// dispersion, `FLIP` scatter/gather framebuffer tiles).
    pub(crate) screen_pixels: u32,
    /// Displaced render-mesh vertex count driving the `water_surface_mesh`
    /// scatter kernel's `DispatchDomain::Vertices` loop and sizing its four
    /// per-vertex storage pools. Zero leaves the surface-mesh pass an
    /// honest no-op.
    pub(crate) surface_vertex_count: u32,
    /// `@group(0)` binding 0 uniform for `water_surface_mesh`: patch grid
    /// dimensions and world-space origin/extent the scatter kernel uses to
    /// place and displace each render-mesh vertex.
    pub(crate) surface_mesh_params: GpuWaterSurfaceMeshParams,
}

impl WaterBody {
    /// Snapshots this body's live passes, counts and loop bounds into the
    /// float-free [`WaterGpuExtract`] the golden `prepare` expands into the
    /// ordered dispatch schedule.
    ///
    /// This is the one place the main-world body meets the architecture-layer
    /// frame plan; every substep / iteration / cascade clamp lives in
    /// [`extract`] so the schedule can never contain an empty active loop.
    #[must_use]
    pub(crate) fn as_extract(&self) -> WaterGpuExtract {
        // The staggered-`MAC` projection dispatches over velocity *faces*, not
        // cells, so the schedule's `DispatchDomain::Faces` loop count comes from
        // the per-axis face sum of the live `FLIP`/`APIC` grid. Derived from the
        // same authored resolution the collocated path uses, so enabling
        // `passes.flip_mac` needs no second sizing input.
        let [nx, ny, nz, _] = self.flip_params.dim;
        let face_count = mac_face_count(nx, ny, nz);
        extract(
            self.counts,
            self.passes,
            self.ocean_cascades,
            self.substeps,
            self.solver_iterations,
            self.spectrum_texels,
            self.grid2d_texels,
            self.grid3d_voxels,
            face_count,
            self.particle_count,
            self.screen_pixels,
            self.surface_vertex_count,
        )
    }

    /// Borrows this body's owned pools and copies its per-pass uniforms,
    /// element counts and texture extents into a [`WaterBodyUpload`] for the
    /// device factory, without cloning any of the owned buffers.
    #[must_use]
    pub(crate) fn as_upload(&self) -> WaterBodyUpload<'_> {
        WaterBodyUpload {
            spectrum_h0: &self.spectrum_h0,
            spectrum_h0_neg: &self.spectrum_h0_neg,
            spectrum_params: self.spectrum_params,
            cascade_params: &self.cascade_params,
            gerstner_waves: &self.gerstner_waves,
            gerstner_params: self.gerstner_params,
            ocean_extent: self.ocean_extent,

            flip_particles: &self.flip_particles,
            flip_grid_cells: self.flip_grid_cells,
            flip_params: self.flip_params,
            flip_surface_params: self.flip_surface_params,
            flip_surface_extent: self.flip_surface_extent,

            pbf_positions: &self.pbf_positions,
            pbf_hash_entries: self.pbf_hash_entries,
            pbf_params: self.pbf_params,

            spray_sources: &self.spray_sources,
            spray_capacity: self.spray_capacity,
            spray_params: self.spray_params,

            swe_cells: self.swe_cells,
            swe_sources: &self.swe_sources,
            swe_params: self.swe_params,

            foam_cells: self.foam_cells,
            foam_sources: &self.foam_sources,
            foam_params: self.foam_params,

            waterline_samples: self.waterline_samples,
            waterline_params: self.waterline_params,

            caustics_photon_counts: &self.caustics_photon_counts,
            caustics_params: self.caustics_params,
            caustics_extent: self.caustics_extent,
            caustics_offset_extent: self.caustics_offset_extent,

            dispersion_params: self.dispersion_params,
            dispersion_extent: self.dispersion_extent,

            underwater_params: self.underwater_params,
            underwater_extent: self.underwater_extent,
            underwater_light_extent: self.underwater_light_extent,

            wetness_cells: self.wetness_cells,
            wetness_params: self.wetness_params,
            wetness_extent: self.wetness_extent,

            coupling_queries: &self.coupling_queries,
            coupling_readback_rows: self.coupling_readback_rows,
            coupling_params: self.coupling_params,

            surface_vertex_count: self.surface_vertex_count,
            surface_mesh_params: self.surface_mesh_params,
        }
    }
}

/// The render-world snapshot of every main-world [`WaterBody`] this frame.
///
/// Rebuilt each frame by the extract stage; the prepare stage turns each entry
/// into a resident `GPU` body. Defaults to empty, which makes the whole water
/// pass an honest no-op when no body is spawned.
#[derive(Resource, Default)]
pub(crate) struct ExtractedWater {
    /// Every extracted body, in main-world iteration order.
    pub(crate) bodies: Vec<WaterBody>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_architecture::water::kernels::WaterKernel;

    /// A body with the spectral ocean plus a `FLIP` volume live, sized so both
    /// solvers emit and both loops expand.
    fn ocean_and_flip_body() -> WaterBody {
        WaterBody {
            spectrum_h0: vec![[0.0, 0.0]; 4],
            spectrum_h0_neg: vec![[0.0, 0.0]; 4],
            gerstner_waves: vec![GpuGerstnerWave::default(); 2],
            cascade_params: vec![GpuWaterSpectrumParams::default(); 4],
            flip_particles: vec![GpuFlipParticle::default(); 8],
            flip_grid_cells: 64 * 64 * 64,
            pbf_positions: Vec::new(),
            passes: WaterPasses {
                ocean_spectrum: true,
                gerstner: true,
                flip: true,
                coupling_readback: true,
                ..WaterPasses::default()
            },
            ocean_cascades: 4,
            substeps: 2,
            solver_iterations: 3,
            spectrum_texels: 256 * 256,
            grid2d_texels: 256 * 256,
            grid3d_voxels: 64 * 64 * 64,
            particle_count: 8,
            screen_pixels: 1920 * 1080,
            ..WaterBody::default()
        }
    }

    #[test]
    fn default_body_yields_an_empty_schedule() {
        let body = WaterBody::default();
        let ex = body.as_extract();
        assert_eq!(ex.ocean_cascades, 0);
        assert!(!ex.ocean_spectrum);
        assert!(!ex.flip);
        // No pass is live, so the golden prepare produces nothing.
        assert!(
            prism_render_architecture::water::gpu::pipeline::prepare(&ex)
                .dispatches
                .is_empty()
        );
    }

    #[test]
    fn as_extract_carries_live_passes_and_clamped_loops() {
        let ex = ocean_and_flip_body().as_extract();
        assert!(ex.ocean_spectrum);
        assert_eq!(ex.ocean_cascades, 4);
        assert!(ex.gerstner);
        assert!(ex.flip);
        assert_eq!(ex.substeps, 2);
        assert_eq!(ex.solver_iterations, 3);
        assert!(ex.coupling_readback);
        // A pass the body did not light stays off.
        assert!(!ex.pbf);
    }

    #[test]
    fn as_extract_feeds_a_non_empty_ordered_schedule() {
        let ex = ocean_and_flip_body().as_extract();
        let plan = prism_render_architecture::water::gpu::pipeline::prepare(&ex);
        assert!(!plan.dispatches.is_empty());
        // The spectral cascades lead and the coupling readback trails, matching
        // the golden frame order.
        assert_eq!(plan.dispatches[0].kernel, WaterKernel::SpectrumIfft);
        assert_eq!(
            plan.dispatches.last().map(|d| d.kernel),
            Some(WaterKernel::CouplingReadback)
        );
    }

    #[test]
    fn as_upload_borrows_pools_without_copying() {
        let body = ocean_and_flip_body();
        let upload = body.as_upload();
        assert_eq!(upload.spectrum_h0.len(), 4);
        assert_eq!(upload.spectrum_h0_neg.len(), 4);
        assert_eq!(upload.cascade_params.len(), 4);
        assert_eq!(upload.gerstner_waves.len(), 2);
        assert_eq!(upload.flip_particles.len(), 8);
        assert_eq!(upload.flip_grid_cells, 64 * 64 * 64);
        assert!(upload.pbf_positions.is_empty());
        // The borrowed slice aliases the component's owned pool.
        assert_eq!(
            upload.spectrum_h0.as_ptr(),
            body.spectrum_h0.as_ptr(),
            "as_upload must borrow, not clone, the spectrum pool"
        );
    }

    #[test]
    fn default_extracted_water_is_empty() {
        let extracted = ExtractedWater::default();
        assert!(extracted.bodies.is_empty());
    }
}
