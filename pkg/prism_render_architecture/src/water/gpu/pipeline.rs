//! The Extract → Prepare → Queue frame plan for the `GPU`-driven water solve.
//!
//! This mirrors the three-stage shape production renderers expose (and the
//! cloth subsystem's [`super::super::gpu`] twin,
//! [`crate::cloth::gpu::pipeline`]): an `Extract` that snapshots one water
//! body's `GPU`-relevant sizes and which solver/render passes are live for the
//! frame, a `Prepare` that expands the substep and projection-iteration loops
//! into a flat, ordered list of concrete compute dispatches, and a `Queue` that
//! aggregates the per-body plans into the per-frame dispatch and resident-byte
//! totals the scheduler arbitrates against a budget.
//!
//! Everything here is integer bookkeeping over the [`super::super::kernels`]
//! dispatch contract and the [`super::buffers`] sizing — no floats, no `GPU`
//! handles, no wall clock — so a whole frame's dispatch schedule can be
//! asserted deterministically in `CPU` tests and diffed across builds. The
//! recorded order is the canonical water frame order: ocean displacement
//! writers → the per-substep simulation stepping (`SWE` / `FLIP`+pressure /
//! `PBF`) → surface reconstruction → foam advection and spray emission →
//! wetness and waterline → the render-effect passes (caustics, dispersion,
//! underwater volume) → the bounded two-way coupling readback. The numerical
//! passes themselves live in `WESL` and mirror the `CPU` golden reference.

use alloc::vec::Vec;

use super::super::kernels::{linear_group_count, DispatchDomain, WaterKernel};
use super::buffers::{WaterBufferCounts, WaterPersistentBufferSet};

/// A snapshot of one water body's `GPU`-relevant sizes and live passes for a
/// single frame.
///
/// Produced by [`extract`] from the body's resident buffer counts, its active
/// solver bucket and render effects, and its stepping counts. It is
/// deliberately flat and float-free so [`prepare`] is a pure function of it.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct WaterGpuExtract {
    /// The resident element counts for the body.
    pub counts: WaterBufferCounts,
    /// Whether the spectral (`Tessendorf`) ocean displacement writer runs.
    pub ocean_spectrum: bool,
    /// Number of spectral cascades to transform (each is one dispatch). Clamped
    /// to at least one by [`extract`] when `ocean_spectrum` is set.
    pub ocean_cascades: u32,
    /// Whether the analytic `Gerstner` displacement writer runs.
    pub gerstner: bool,
    /// Whether the Shallow-Water height-field step runs.
    pub swe: bool,
    /// Whether the `PBF` density solve runs.
    pub pbf: bool,
    /// Whether the collocated `FLIP`/`APIC` three-pass grid solve runs. This is
    /// the legacy cell-centered path kept until the staggered `MAC` chain is
    /// proven end-to-end on real hardware (see [`Self::flip_mac`]).
    pub flip: bool,
    /// Whether the face-centered staggered `MAC` `FLIP`/`APIC` chain runs
    /// (`P2G` scatter, face normalize, compact divergence, `Jacobi` pressure,
    /// orthogonal projection, `G2P` gather). Mutually exclusive with
    /// [`Self::flip`] in a correct body configuration.
    pub flip_mac: bool,
    /// Whether the renderable-surface reconstruction runs.
    pub reconstruct: bool,
    /// Whether the caustics projection runs.
    pub caustics: bool,
    /// Whether the semi-Lagrangian foam advection runs.
    pub foam: bool,
    /// Whether crest-spray emission runs.
    pub spray: bool,
    /// Whether the surface wetness step runs.
    pub wetness: bool,
    /// Whether the waterline mask rasterization runs.
    pub waterline: bool,
    /// Whether screen-space spectral (`RGB`) dispersion refraction runs.
    pub dispersion: bool,
    /// Whether underwater single-scatter volume accumulation runs.
    pub underwater: bool,
    /// Whether the bounded two-way coupling readback runs.
    pub coupling_readback: bool,
    /// Whether the surface meshing pass runs: sample the assembled
    /// displacement/normal textures into the per-vertex storage arrays the
    /// raster draw reads. Sized against [`Self::vertex_count`].
    pub surface_mesh: bool,
    /// Simulation substeps per frame (clamped to at least one by [`extract`]).
    pub substeps: u32,
    /// Projection iterations per substep for the pressure/density solves
    /// (clamped to at least one by [`extract`]).
    pub solver_iterations: u32,
    /// Invocation count for one spectral cascade (spectrum texels).
    pub spectrum_texels: u32,
    /// Invocation count for the 2D grid passes (`SWE`/`Gerstner`/foam/waterline
    /// height-field texels).
    pub grid2d_texels: u32,
    /// Invocation count for the 3D grid passes (`FLIP` `MAC` voxels, underwater
    /// froxels).
    pub grid3d_voxels: u32,
    /// Invocation count for the face-centered staggered `MAC` passes: the sum
    /// of the per-axis face counts of the velocity grid. Use [`mac_face_count`]
    /// to derive it from the grid dimensions.
    pub face_count: u32,
    /// Invocation count for the particle-domain passes (`PBF`/spray particles).
    pub particle_count: u32,
    /// Invocation count for the full-screen passes (reconstruction, caustics,
    /// dispersion, `FLIP` scatter/gather framebuffer tiles).
    pub screen_pixels: u32,
    /// Invocation count for the surface meshing pass: the number of lattice
    /// vertices ([`DispatchDomain::Vertices`]), one invocation each.
    pub vertex_count: u32,
}

impl WaterGpuExtract {
    /// The invocation count a kernel is sized against, mapped from its
    /// [`DispatchDomain`]. The spectral transform is special-cased to the
    /// per-cascade texel count; every other domain reads the matching extent.
    #[must_use]
    fn invocations(&self, kernel: WaterKernel) -> u32 {
        if matches!(
            kernel,
            WaterKernel::SpectrumIfft
                | WaterKernel::SpectrumEvolve
                | WaterKernel::FftBitReverse
                | WaterKernel::FftStage
                | WaterKernel::FftNormalize
                | WaterKernel::SpectrumAssemble
        ) {
            return self.spectrum_texels;
        }
        match kernel.descriptor().domain {
            DispatchDomain::Grid2d => self.grid2d_texels,
            DispatchDomain::Grid3d => self.grid3d_voxels,
            DispatchDomain::Faces => self.face_count,
            DispatchDomain::Particle => self.particle_count,
            DispatchDomain::Screen => self.screen_pixels,
            DispatchDomain::Vertices => self.vertex_count,
        }
    }

    /// The number of workgroups a single dispatch of `kernel` launches, sized
    /// against this body's extents and the kernel's workgroup tile.
    #[must_use]
    fn groups(&self, kernel: WaterKernel) -> u32 {
        let descriptor = kernel.descriptor();
        linear_group_count(
            self.invocations(kernel),
            descriptor.workgroup.invocations_per_group(),
        )
    }
}

/// One fully sized compute dispatch in the recorded schedule.
///
/// `groups` is the number of workgroups to launch, already divided from the
/// domain extent by the kernel's workgroup tile. `substep` records which
/// simulation substep the dispatch belongs to (`None` for the once-per-frame
/// passes), and `iteration` records which projection iteration a pressure or
/// density solve belongs to, so the schedule reads back in exact solver order.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct PlannedDispatch {
    /// The kernel being dispatched.
    pub kernel: WaterKernel,
    /// The number of workgroups to launch.
    pub groups: u32,
    /// The simulation substep this dispatch belongs to, if any.
    pub substep: Option<u32>,
    /// The projection iteration this dispatch belongs to, if any.
    pub iteration: Option<u32>,
}

/// The expanded, ordered dispatch schedule for one water body.
///
/// Built by [`prepare`]. The dispatches are in exact record order: ocean
/// displacement writers (spectral cascades then `Gerstner`), then per substep
/// the active simulation stepping (`SWE`; `FLIP` `P2G` → pressure iterations →
/// `G2P`; `PBF` density iterations), then once per frame the surface
/// reconstruction, foam advection, spray emission, wetness step, waterline
/// mask, and the render-effect passes (caustics, dispersion, underwater
/// volume), and finally the coupling readback.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WaterGpuPrepare {
    /// The resident buffer set sizing for the body.
    pub buffers: WaterPersistentBufferSet,
    /// The ordered dispatch schedule.
    pub dispatches: Vec<PlannedDispatch>,
}

impl WaterGpuPrepare {
    /// The total number of workgroups launched across every dispatch in the
    /// schedule, saturating.
    #[must_use]
    pub fn total_groups(&self) -> u64 {
        self.dispatches
            .iter()
            .fold(0u64, |acc, d| acc.saturating_add(u64::from(d.groups)))
    }
}

/// The per-frame aggregate across every water body's plan.
///
/// Produced by [`queue`]; the scheduler reads these totals to arbitrate the
/// water solve against the frame's compute and memory budget.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct WaterGpuQueue {
    /// Number of water bodies scheduled this frame.
    pub bodies: u32,
    /// Total number of compute dispatches across every body.
    pub dispatches: u64,
    /// Total number of workgroups across every dispatch.
    pub groups: u64,
    /// Total resident bytes across every body's persistent buffer set.
    pub resident_bytes: u64,
}

/// The whole frame plan: every body's extract and prepare, plus the aggregate
/// queue.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WaterGpuFramePlan {
    /// The per-body snapshots.
    pub extracts: Vec<WaterGpuExtract>,
    /// The per-body expanded schedules.
    pub prepares: Vec<WaterGpuPrepare>,
    /// The per-frame aggregate.
    pub queue: WaterGpuQueue,
}

/// Snapshots one water body into a [`WaterGpuExtract`], clamping the substep,
/// iteration and cascade counts to at least one where the corresponding pass is
/// live so [`prepare`] never emits an empty loop for an active solver.
#[expect(
    clippy::too_many_arguments,
    reason = "an extract is a flat frame snapshot of every live water pass and \
              its extents; grouping them into sub-structs would only relocate \
              the same fields without simplifying the contract"
)]
#[must_use]
pub fn extract(
    counts: WaterBufferCounts,
    passes: WaterPasses,
    ocean_cascades: u32,
    substeps: u32,
    solver_iterations: u32,
    spectrum_texels: u32,
    grid2d_texels: u32,
    grid3d_voxels: u32,
    face_count: u32,
    particle_count: u32,
    screen_pixels: u32,
    vertex_count: u32,
) -> WaterGpuExtract {
    WaterGpuExtract {
        counts,
        ocean_spectrum: passes.ocean_spectrum,
        ocean_cascades: if passes.ocean_spectrum {
            ocean_cascades.max(1)
        } else {
            0
        },
        gerstner: passes.gerstner,
        swe: passes.swe,
        pbf: passes.pbf,
        flip: passes.flip,
        flip_mac: passes.flip_mac,
        reconstruct: passes.reconstruct,
        caustics: passes.caustics,
        foam: passes.foam,
        spray: passes.spray,
        wetness: passes.wetness,
        waterline: passes.waterline,
        dispersion: passes.dispersion,
        underwater: passes.underwater,
        coupling_readback: passes.coupling_readback,
        surface_mesh: passes.surface_mesh,
        substeps: substeps.max(1),
        solver_iterations: solver_iterations.max(1),
        spectrum_texels,
        grid2d_texels,
        grid3d_voxels,
        face_count,
        particle_count,
        screen_pixels,
        vertex_count,
    }
}

/// The total number of staggered `MAC` velocity faces for a `(nx, ny, nz)` cell
/// grid: the sum of the per-axis face counts.
///
/// A `MAC` grid stores velocity components at cell faces, so the `u` family has
/// `(nx + 1) * ny * nz` faces, the `v` family `nx * (ny + 1) * nz`, and the `w`
/// family `nx * ny * (nz + 1)`. This is the invocation count the face-domain
/// passes ([`DispatchDomain::Faces`]) are sized against. All arithmetic is
/// saturating so an adversarial dimension clamps to `u32::MAX` rather than
/// wrapping.
#[must_use]
pub fn mac_face_count(nx: u32, ny: u32, nz: u32) -> u32 {
    let u_faces = nx.saturating_add(1).saturating_mul(ny).saturating_mul(nz);
    let v_faces = nx.saturating_mul(ny.saturating_add(1)).saturating_mul(nz);
    let w_faces = nx.saturating_mul(ny).saturating_mul(nz.saturating_add(1));
    u_faces.saturating_add(v_faces).saturating_add(w_faces)
}

/// The set of live passes for a water body this frame, a flat boolean record so
/// [`extract`] stays a single call site rather than a wide positional argument
/// list of raw booleans.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct WaterPasses {
    /// Spectral (`Tessendorf`) ocean displacement writer.
    pub ocean_spectrum: bool,
    /// Analytic `Gerstner` displacement writer.
    pub gerstner: bool,
    /// Shallow-Water height-field step.
    pub swe: bool,
    /// `PBF` density solve.
    pub pbf: bool,
    /// Collocated `FLIP`/`APIC` three-pass grid solve (legacy cell-centered).
    pub flip: bool,
    /// Face-centered staggered `MAC` `FLIP`/`APIC` chain (`P2G`, face normalize,
    /// compact divergence, `Jacobi` pressure, orthogonal projection, `G2P`).
    pub flip_mac: bool,
    /// Renderable-surface reconstruction.
    pub reconstruct: bool,
    /// Caustics projection.
    pub caustics: bool,
    /// Semi-Lagrangian foam advection.
    pub foam: bool,
    /// Crest-spray emission.
    pub spray: bool,
    /// Surface wetness step.
    pub wetness: bool,
    /// Waterline mask rasterization.
    pub waterline: bool,
    /// Screen-space spectral dispersion refraction.
    pub dispersion: bool,
    /// Underwater single-scatter volume accumulation.
    pub underwater: bool,
    /// Bounded two-way coupling readback.
    pub coupling_readback: bool,
    /// Surface meshing pass (sample cascade textures into per-vertex arrays).
    pub surface_mesh: bool,
}

/// Expands one body's [`WaterGpuExtract`] into its ordered dispatch schedule.
#[must_use]
pub fn prepare(extract: &WaterGpuExtract) -> WaterGpuPrepare {
    let mut dispatches = Vec::new();

    // 1. Ocean displacement writers, once per frame.
    if extract.ocean_spectrum {
        for _ in 0..extract.ocean_cascades {
            push(
                &mut dispatches,
                extract,
                WaterKernel::SpectrumIfft,
                None,
                None,
            );
        }
    }
    if extract.gerstner {
        push(
            &mut dispatches,
            extract,
            WaterKernel::GerstnerDisplace,
            None,
            None,
        );
    }

    // 2. Per-substep simulation stepping.
    for substep in 0..extract.substeps {
        let step = Some(substep);
        if extract.swe {
            push(&mut dispatches, extract, WaterKernel::SweStep, step, None);
        }
        if extract.flip {
            push(&mut dispatches, extract, WaterKernel::FlipP2G, step, None);
            for iteration in 0..extract.solver_iterations {
                push(
                    &mut dispatches,
                    extract,
                    WaterKernel::FlipPressureSolve,
                    step,
                    Some(iteration),
                );
            }
            push(&mut dispatches, extract, WaterKernel::FlipG2P, step, None);
        }
        // Face-centered staggered `MAC` chain: scatter to faces, normalize by
        // weight, take the compact divergence once, relax pressure for the
        // configured iterations, project the faces orthogonally, then gather
        // the projected faces back to particles.
        if extract.flip_mac {
            push(
                &mut dispatches,
                extract,
                WaterKernel::FlipMacP2G,
                step,
                None,
            );
            push(
                &mut dispatches,
                extract,
                WaterKernel::FlipMacFacesNormalize,
                step,
                None,
            );
            push(
                &mut dispatches,
                extract,
                WaterKernel::FlipMacDivergence,
                step,
                None,
            );
            for iteration in 0..extract.solver_iterations {
                push(
                    &mut dispatches,
                    extract,
                    WaterKernel::FlipMacPressure,
                    step,
                    Some(iteration),
                );
            }
            push(
                &mut dispatches,
                extract,
                WaterKernel::FlipMacProject,
                step,
                None,
            );
            push(
                &mut dispatches,
                extract,
                WaterKernel::FlipMacG2P,
                step,
                None,
            );
        }
        if extract.pbf {
            for iteration in 0..extract.solver_iterations {
                push(
                    &mut dispatches,
                    extract,
                    WaterKernel::PbfDensitySolve,
                    step,
                    Some(iteration),
                );
            }
        }
    }

    // 3. Surface reconstruction, once per frame.
    if extract.reconstruct {
        push(
            &mut dispatches,
            extract,
            WaterKernel::SurfaceReconstruct,
            None,
            None,
        );
    }

    // 3b. Surface meshing: sample the assembled displacement/normal textures
    // into the per-vertex storage arrays the raster draw reads, once per frame.
    if extract.surface_mesh {
        push(
            &mut dispatches,
            extract,
            WaterKernel::SurfaceMesh,
            None,
            None,
        );
    }

    // 4. Foam advection and spray emission.
    if extract.foam {
        push(
            &mut dispatches,
            extract,
            WaterKernel::FoamAdvect,
            None,
            None,
        );
    }
    if extract.spray {
        push(&mut dispatches, extract, WaterKernel::SprayEmit, None, None);
    }

    // 5. Wetness and waterline.
    if extract.wetness {
        push(
            &mut dispatches,
            extract,
            WaterKernel::WetnessStep,
            None,
            None,
        );
    }
    if extract.waterline {
        push(
            &mut dispatches,
            extract,
            WaterKernel::WaterlineMask,
            None,
            None,
        );
    }

    // 6. Render-effect passes.
    if extract.caustics {
        push(
            &mut dispatches,
            extract,
            WaterKernel::CausticsProject,
            None,
            None,
        );
    }
    if extract.dispersion {
        push(
            &mut dispatches,
            extract,
            WaterKernel::DispersionRefract,
            None,
            None,
        );
    }
    if extract.underwater {
        push(
            &mut dispatches,
            extract,
            WaterKernel::UnderwaterVolume,
            None,
            None,
        );
    }

    // 7. Bounded two-way coupling readback, last.
    if extract.coupling_readback {
        push(
            &mut dispatches,
            extract,
            WaterKernel::CouplingReadback,
            None,
            None,
        );
    }

    WaterGpuPrepare {
        buffers: WaterPersistentBufferSet::new(extract.counts),
        dispatches,
    }
}

/// Appends one planned dispatch, sizing its workgroup count from the extract.
fn push(
    dispatches: &mut Vec<PlannedDispatch>,
    extract: &WaterGpuExtract,
    kernel: WaterKernel,
    substep: Option<u32>,
    iteration: Option<u32>,
) {
    dispatches.push(PlannedDispatch {
        kernel,
        groups: extract.groups(kernel),
        substep,
        iteration,
    });
}

/// Aggregates every body's prepared schedule into the per-frame
/// [`WaterGpuQueue`] totals. Saturating throughout.
#[must_use]
pub fn queue(prepares: &[WaterGpuPrepare]) -> WaterGpuQueue {
    let mut out = WaterGpuQueue {
        bodies: prepares.len() as u32,
        ..WaterGpuQueue::default()
    };
    for prepare in prepares {
        out.dispatches = out
            .dispatches
            .saturating_add(prepare.dispatches.len() as u64);
        out.groups = out.groups.saturating_add(prepare.total_groups());
        out.resident_bytes = out
            .resident_bytes
            .saturating_add(u64::from(prepare.buffers.total_bytes()));
    }
    out
}

/// Builds the whole-frame plan from every body's extract: prepares each and
/// aggregates the queue totals.
#[must_use]
pub fn plan_frame(extracts: Vec<WaterGpuExtract>) -> WaterGpuFramePlan {
    let prepares: Vec<WaterGpuPrepare> = extracts.iter().map(prepare).collect();
    let queue = queue(&prepares);
    WaterGpuFramePlan {
        extracts,
        prepares,
        queue,
    }
}

#[cfg(test)]
mod tests {
    use super::super::buffers::WaterBufferCounts;
    use super::{extract, plan_frame, prepare, queue, WaterKernel, WaterPasses};
    use alloc::vec;
    use alloc::vec::Vec;

    fn ocean_passes() -> WaterPasses {
        WaterPasses {
            ocean_spectrum: true,
            gerstner: true,
            foam: true,
            waterline: true,
            wetness: true,
            caustics: true,
            dispersion: true,
            underwater: true,
            coupling_readback: true,
            ..WaterPasses::default()
        }
    }

    fn ocean_extract() -> super::WaterGpuExtract {
        extract(
            WaterBufferCounts {
                spectrum_texels: 256 * 256,
                gerstner_waves: 32,
                foam_cells: 512 * 512,
                wetness_cells: 256 * 256,
                froxels: 160 * 90 * 64,
                ..WaterBufferCounts::default()
            },
            ocean_passes(),
            4, // cascades
            2, // substeps
            3, // solver iterations
            256 * 256,
            256 * 256,
            160 * 90 * 64,
            0, // face_count (no MAC pass in this body)
            0,
            1920 * 1080,
            0, // vertex_count (no surface mesh in this body)
        )
    }

    #[test]
    fn extract_clamps_active_loops_to_at_least_one() {
        let ex = extract(
            WaterBufferCounts::default(),
            WaterPasses {
                ocean_spectrum: true,
                flip: true,
                ..WaterPasses::default()
            },
            0, // cascades → clamps to 1 because ocean_spectrum is live
            0, // substeps → clamps to 1
            0, // iterations → clamps to 1
            1,
            1,
            1,
            1,
            1,
            1,
            1,
        );
        assert_eq!(ex.ocean_cascades, 1);
        assert_eq!(ex.substeps, 1);
        assert_eq!(ex.solver_iterations, 1);
    }

    #[test]
    fn inactive_spectrum_reports_zero_cascades() {
        let ex = extract(
            WaterBufferCounts::default(),
            WaterPasses::default(),
            8,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
        );
        assert_eq!(ex.ocean_cascades, 0);
    }

    #[test]
    fn spectrum_emits_one_dispatch_per_cascade_first() {
        let plan = prepare(&ocean_extract());
        let spectrum = plan
            .dispatches
            .iter()
            .filter(|d| d.kernel == WaterKernel::SpectrumIfft)
            .count();
        assert_eq!(spectrum, 4);
        // The very first recorded dispatches are the spectral cascades.
        assert_eq!(plan.dispatches[0].kernel, WaterKernel::SpectrumIfft);
        assert_eq!(plan.dispatches[4].kernel, WaterKernel::GerstnerDisplace);
    }

    #[test]
    fn coupling_readback_is_recorded_last() {
        let plan = prepare(&ocean_extract());
        assert_eq!(
            plan.dispatches.last().map(|d| d.kernel),
            Some(WaterKernel::CouplingReadback)
        );
    }

    #[test]
    fn flip_expands_p2g_pressure_iterations_g2p_per_substep() {
        let ex = extract(
            WaterBufferCounts {
                flip_particles: 40_000,
                flip_grid_cells: 64 * 64 * 64,
                ..WaterBufferCounts::default()
            },
            WaterPasses {
                flip: true,
                ..WaterPasses::default()
            },
            0,
            2, // substeps
            3, // pressure iterations
            0,
            0,
            64 * 64 * 64,
            0, // face_count (collocated path under test)
            40_000,
            1920 * 1080,
            0, // vertex_count
        );
        let plan = prepare(&ex);
        let p2g = plan
            .dispatches
            .iter()
            .filter(|d| d.kernel == WaterKernel::FlipP2G)
            .count();
        let pressure = plan
            .dispatches
            .iter()
            .filter(|d| d.kernel == WaterKernel::FlipPressureSolve)
            .count();
        let g2p = plan
            .dispatches
            .iter()
            .filter(|d| d.kernel == WaterKernel::FlipG2P)
            .count();
        // Two substeps, each: 1 P2G + 3 pressure + 1 G2P.
        assert_eq!(p2g, 2);
        assert_eq!(pressure, 2 * 3);
        assert_eq!(g2p, 2);
        // Ordering within the first substep: P2G before its pressure solves
        // before G2P.
        let first_p2g = plan
            .dispatches
            .iter()
            .position(|d| d.kernel == WaterKernel::FlipP2G)
            .expect("p2g present");
        let first_g2p = plan
            .dispatches
            .iter()
            .position(|d| d.kernel == WaterKernel::FlipG2P)
            .expect("g2p present");
        assert!(first_p2g < first_g2p);
    }

    #[test]
    fn flip_mac_chain_expands_in_face_centered_order_per_substep() {
        let ex = extract(
            WaterBufferCounts {
                flip_particles: 40_000,
                flip_grid_cells: 32 * 32 * 32,
                ..WaterBufferCounts::default()
            },
            WaterPasses {
                flip_mac: true,
                ..WaterPasses::default()
            },
            0,
            2, // substeps
            4, // pressure iterations
            0,
            0,
            32 * 32 * 32,
            super::mac_face_count(32, 32, 32),
            40_000,
            0,
            0, // vertex_count
        );
        let plan = prepare(&ex);
        let count = |k: WaterKernel| plan.dispatches.iter().filter(|d| d.kernel == k).count();
        // Each substep: 1 P2G + 1 normalize + 1 divergence + 4 pressure + 1
        // project + 1 G2P. Two substeps double every count.
        assert_eq!(count(WaterKernel::FlipMacP2G), 2);
        assert_eq!(count(WaterKernel::FlipMacFacesNormalize), 2);
        assert_eq!(count(WaterKernel::FlipMacDivergence), 2);
        assert_eq!(count(WaterKernel::FlipMacPressure), 2 * 4);
        assert_eq!(count(WaterKernel::FlipMacProject), 2);
        assert_eq!(count(WaterKernel::FlipMacG2P), 2);
        // The collocated path stays silent when only the MAC chain is live.
        assert_eq!(count(WaterKernel::FlipP2G), 0);
        // Ordering within the first substep: scatter → normalize → divergence →
        // pressure → project → gather.
        let pos = |k: WaterKernel| {
            plan.dispatches
                .iter()
                .position(|d| d.kernel == k)
                .expect("kernel present")
        };
        assert!(pos(WaterKernel::FlipMacP2G) < pos(WaterKernel::FlipMacFacesNormalize));
        assert!(pos(WaterKernel::FlipMacFacesNormalize) < pos(WaterKernel::FlipMacDivergence));
        assert!(pos(WaterKernel::FlipMacDivergence) < pos(WaterKernel::FlipMacPressure));
        assert!(pos(WaterKernel::FlipMacPressure) < pos(WaterKernel::FlipMacProject));
        assert!(pos(WaterKernel::FlipMacProject) < pos(WaterKernel::FlipMacG2P));
    }

    #[test]
    fn mac_face_count_sums_per_axis_faces_and_saturates() {
        // A unit cell has 2 faces per axis → 6 total.
        assert_eq!(super::mac_face_count(1, 1, 1), 6);
        // Explicit per-axis sum for a 2x3x4 grid.
        let (nx, ny, nz) = (2u32, 3, 4);
        let expected = (nx + 1) * ny * nz + nx * (ny + 1) * nz + nx * ny * (nz + 1);
        assert_eq!(super::mac_face_count(nx, ny, nz), expected);
        // A degenerate zero-cell axis is still well defined: the u family keeps
        // its `(nx + 1)` boundary faces, so `(0, 10, 10)` yields `1 * 10 * 10`.
        assert_eq!(super::mac_face_count(0, 10, 10), 100);
        // Adversarial dimensions saturate rather than wrapping.
        assert_eq!(
            super::mac_face_count(u32::MAX, u32::MAX, u32::MAX),
            u32::MAX
        );
    }

    #[test]
    fn surface_mesh_pass_schedules_one_vertex_sized_dispatch() {
        let ex = extract(
            WaterBufferCounts::default(),
            WaterPasses {
                surface_mesh: true,
                ..WaterPasses::default()
            },
            0,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            300, // vertex_count
        );
        let plan = prepare(&ex);
        let meshing: Vec<_> = plan
            .dispatches
            .iter()
            .filter(|d| d.kernel == WaterKernel::SurfaceMesh)
            .collect();
        assert_eq!(meshing.len(), 1);
        // 300 vertices over a 64-lane sweep rounds up to five workgroups.
        assert_eq!(meshing[0].groups, 5);
    }

    #[test]
    fn empty_passes_produce_an_empty_schedule() {
        let ex = extract(
            WaterBufferCounts::default(),
            WaterPasses::default(),
            0,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        );
        let plan = prepare(&ex);
        assert!(plan.dispatches.is_empty());
        assert_eq!(plan.total_groups(), 0);
    }

    #[test]
    fn queue_sums_dispatches_groups_and_bytes() {
        let plan = plan_frame(vec![ocean_extract(), ocean_extract()]);
        assert_eq!(plan.queue.bodies, 2);
        let per_body = prepare(&ocean_extract());
        assert_eq!(plan.queue.dispatches, 2 * per_body.dispatches.len() as u64);
        assert_eq!(plan.queue.groups, 2 * per_body.total_groups());
        assert_eq!(
            plan.queue.resident_bytes,
            2 * u64::from(per_body.buffers.total_bytes())
        );
    }

    #[test]
    fn queue_of_empty_is_zeroed_but_counts_bodies() {
        let empties: Vec<super::WaterGpuPrepare> = Vec::new();
        assert_eq!(queue(&empties).bodies, 0);
    }
}
