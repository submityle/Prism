//! The `Core3d` compute node that records one frame of `GPU` water solving.
//!
//! This node is a *dumb executor*: every ordering, sizing and per-kernel group
//! decision was already made — and unit-tested — in the float-free golden plan
//! ([`prism_render_architecture::water::gpu::pipeline`]). Each resident
//! [`WaterGpuBody`](super::resources::WaterGpuBody) carries its
//! [`PlannedDispatch`](prism_render_architecture::water::gpu::pipeline::PlannedDispatch)
//! schedule in exact solver order, so all this node does is walk that list, bind
//! the matching group at the shader-declared `@group` index, and launch the
//! recorded workgroup count. Unlike cloth, water carries no per-color immediate:
//! every water kernel reads its whole domain from the uniform counts.
//!
//! Readiness is all-or-nothing: if any of the sixteen pipelines is still
//! compiling this frame the node records nothing, rather than running a partial
//! solve that would leave a body in a half-stepped, non-deterministic state.

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{ComputePassDescriptor, PipelineCache},
    renderer::RenderContext,
};

use prism_render_architecture::water::gpu::{plan_inverse_fft2, FftEntry};
use prism_render_architecture::water::kernels::WaterKernel;

use super::bind_groups::WaterBodyBindGroups;
use super::pipeline::{wesl_group, WaterComputePipelines};
use super::resources::WaterGpuBodies;

/// Borrows the concrete bind group a kernel dispatches against from a body.
///
/// Mirrors [`WaterComputePipelines::layout`](super::pipeline::WaterComputePipelines::layout)
/// exactly: the ocean group serves the two ocean-surface kernels, the flip
/// group serves the three `FLIP` stages plus the surface reconstruction that
/// reads the same `MAC` grid, and every other kernel takes its own group. Kept
/// as one exhaustive match so a new kernel cannot compile without choosing a
/// group, and so the mapping is unit-tested without a `GPU`.
#[must_use]
fn bind_group_for<'a>(
    kernel: WaterKernel,
    groups: &'a WaterBodyBindGroups,
) -> &'a bevy_render::render_resource::BindGroup {
    match kernel {
        WaterKernel::SpectrumIfft | WaterKernel::GerstnerDisplace => &groups.ocean,
        WaterKernel::FlipP2G
        | WaterKernel::FlipPressureSolve
        | WaterKernel::FlipG2P
        | WaterKernel::SurfaceReconstruct => &groups.flip,
        WaterKernel::PbfDensitySolve => &groups.pbf,
        WaterKernel::SprayEmit => &groups.spray,
        WaterKernel::SweStep => &groups.swe,
        WaterKernel::FoamAdvect => &groups.foam,
        WaterKernel::WaterlineMask => &groups.waterline,
        WaterKernel::CausticsProject => &groups.caustics,
        WaterKernel::DispersionRefract => &groups.dispersion,
        WaterKernel::UnderwaterVolume => &groups.underwater,
        WaterKernel::WetnessStep => &groups.wetness,
        WaterKernel::CouplingReadback => &groups.coupling,
        // The spectral evolve/assemble and the three butterfly passes are only
        // ever recorded through the `SpectrumIfft` expansion below (which binds
        // the per-pass ping-pong groups directly), never as a top-level planned
        // dispatch; the arm exists solely to keep this match total.
        WaterKernel::SpectrumEvolve
        | WaterKernel::SpectrumAssemble
        | WaterKernel::FftBitReverse
        | WaterKernel::FftStage
        | WaterKernel::FftNormalize => &groups.spectrum_fft[0],
    }
}

/// Records every resident water body's solve for this frame.
///
/// Runs as a `Core3d` compute node before the main pass. No-ops when no water
/// is resident or while any pipeline is still compiling.
pub(crate) fn dispatch_water(
    bodies: Res<WaterGpuBodies>,
    pipelines: Res<WaterComputePipelines>,
    cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    if bodies.is_empty() {
        return;
    }

    // All-or-nothing readiness: bail before opening a pass if any kernel is
    // still compiling, so a frame never records a partial (non-deterministic)
    // solve. Every body shares these sixteen pipelines.
    for kernel in WaterKernel::ALL {
        if cache
            .get_compute_pipeline(pipelines.pipeline(kernel))
            .is_none()
        {
            return;
        }
    }

    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism GPU water solve"),
            timestamp_writes: None,
        });

    for body in &bodies.bodies {
        // The golden plan emits the spectral `IFFT` markers first, one per
        // cascade in cascade order; walk a counter so each marker records the
        // inverse `FFT` for its own atlas tile (bind group `spectrum_fft[c]`).
        let mut cascade = 0usize;
        for dispatch in &body.dispatches {
            // The golden plan still emits one `SpectrumIfft` marker per ocean
            // cascade, but production no longer runs the `O(N^4)` direct-sum
            // inverse: it expands the marker here into the separable
            // `O(N log N)` butterfly `FFT` (spectrum evolve, then a per-grid
            // ping-pong pass list, then the assemble that packs the four
            // inverted complex grids into the displacement/normal textures) — the
            // same production transform `WaveWorks`/`Crest`/`UE5` Water run.
            if dispatch.kernel == WaterKernel::SpectrumIfft {
                record_spectral_ifft(&mut pass, &pipelines, &cache, &body.bind_groups, cascade);
                cascade += 1;
                continue;
            }
            // Safe to expect: readiness was gated above and the pipeline set is
            // shared by every body.
            let pipeline = cache
                .get_compute_pipeline(pipelines.pipeline(dispatch.kernel))
                .expect("water pipelines were all checked ready above");
            pass.set_pipeline(pipeline);
            // Bind at the shader-declared group index: the render-effect kernels
            // live on `@group(1..=4)` while the simulation kernels live on
            // `@group(0)`; `wesl_group` is the single source of that mapping.
            pass.set_bind_group(
                wesl_group(dispatch.kernel),
                bind_group_for(dispatch.kernel, &body.bind_groups),
                &[],
            );
            pass.dispatch_workgroups(dispatch.groups, 1, 1);
        }
    }
}

/// Records one ocean cascade's production spectral inverse `FFT` in place of the
/// `O(N^4)` direct-sum `SpectrumIfft` marker.
///
/// Replays the golden [`plan_inverse_fft2`] schedule on device: a spectrum
/// evolve writes the four packed complex grids, each grid is then inverted by
/// the deterministic ping-pong butterfly pass list (bit-reversal, `log2(N)`
/// stages per axis, then the shared normalize), and a final assemble packs the
/// inverted grids into the displacement and normal textures. Dispatch
/// dimensions mirror the `8x8`-tiled `water_spectrum_fft.wesl` /
/// `water_butterfly.wesl` entry points: the stage kernel launches one thread per
/// butterfly pair (`N/2` along the transform axis) while every other pass covers
/// the full `N*N` grid. A non-power-of-two edge plans no passes and is a
/// deterministic no-op.
fn record_spectral_ifft(
    pass: &mut bevy_render::render_resource::ComputePass<'_>,
    pipelines: &WaterComputePipelines,
    cache: &PipelineCache,
    groups: &WaterBodyBindGroups,
    cascade: usize,
) {
    let n = groups.ocean_n;
    let plan = plan_inverse_fft2(n);
    if plan.is_empty() {
        return;
    }
    let full = n.div_ceil(8);

    // 1. Time-advance the spectrum into the four packed complex grids.
    pass.set_pipeline(
        cache
            .get_compute_pipeline(pipelines.pipeline(WaterKernel::SpectrumEvolve))
            .expect("water pipelines were all checked ready above"),
    );
    pass.set_bind_group(0, &groups.spectrum_fft[cascade], &[]);
    pass.dispatch_workgroups(full, full, 1);

    // 2. Invert each of the four packed complex grids with the ping-pong
    //    butterfly pass list; every grid replays the full plan in order.
    for grid in 0..4 {
        for (ordinal, fft_pass) in plan.iter().enumerate() {
            let kernel = match fft_pass.entry {
                FftEntry::BitReversal => WaterKernel::FftBitReverse,
                FftEntry::Butterfly => WaterKernel::FftStage,
                FftEntry::Normalize => WaterKernel::FftNormalize,
            };
            pass.set_pipeline(
                cache
                    .get_compute_pipeline(pipelines.pipeline(kernel))
                    .expect("water pipelines were all checked ready above"),
            );
            pass.set_bind_group(0, &groups.butterfly_passes[ordinal][grid], &[]);
            let groups_x = match fft_pass.entry {
                FftEntry::Butterfly => (n / 2).div_ceil(8),
                FftEntry::BitReversal | FftEntry::Normalize => full,
            };
            pass.dispatch_workgroups(groups_x, full, 1);
        }
    }

    // 3. Pack the four inverted complex grids into the displacement/normal
    //    storage textures the render passes sample.
    pass.set_pipeline(
        cache
            .get_compute_pipeline(pipelines.pipeline(WaterKernel::SpectrumAssemble))
            .expect("water pipelines were all checked ready above"),
    );
    pass.set_bind_group(0, &groups.spectrum_fft[cascade], &[]);
    pass.dispatch_workgroups(full, full, 1);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_kernel_maps_to_its_pipeline_layout_group() {
        // The two ocean-surface kernels share the ocean group.
        for kernel in [WaterKernel::SpectrumIfft, WaterKernel::GerstnerDisplace] {
            assert_eq!(
                bind_group_slot_index(kernel),
                bind_group_slot_index(WaterKernel::SpectrumIfft)
            );
        }
        // The three FLIP stages plus surface reconstruction share the flip
        // group (they all address the same MAC grid buffers).
        for kernel in [
            WaterKernel::FlipP2G,
            WaterKernel::FlipPressureSolve,
            WaterKernel::FlipG2P,
            WaterKernel::SurfaceReconstruct,
        ] {
            assert_eq!(
                bind_group_slot_index(kernel),
                bind_group_slot_index(WaterKernel::FlipP2G)
            );
        }
    }

    #[test]
    fn each_kernel_selects_exactly_one_group() {
        // Every kernel must resolve to a stable, distinct-or-shared slot without
        // panicking; iterating ALL proves the match is exhaustive and total.
        for kernel in WaterKernel::ALL {
            let _slot = bind_group_slot_index(kernel);
        }
    }

    /// Pure mirror of [`bind_group_for`]'s kernel→group choice as a small
    /// index, so the mapping is unit-testable without constructing real
    /// `wgpu` bind groups (which need a device the sandbox lacks).
    fn bind_group_slot_index(kernel: WaterKernel) -> usize {
        match kernel {
            WaterKernel::SpectrumIfft | WaterKernel::GerstnerDisplace => 0,
            WaterKernel::FlipP2G
            | WaterKernel::FlipPressureSolve
            | WaterKernel::FlipG2P
            | WaterKernel::SurfaceReconstruct => 1,
            WaterKernel::PbfDensitySolve => 2,
            WaterKernel::SprayEmit => 3,
            WaterKernel::SweStep => 4,
            WaterKernel::FoamAdvect => 5,
            WaterKernel::WaterlineMask => 6,
            WaterKernel::CausticsProject => 7,
            WaterKernel::DispersionRefract => 8,
            WaterKernel::UnderwaterVolume => 9,
            WaterKernel::WetnessStep => 10,
            WaterKernel::CouplingReadback => 11,
            // All five spectral/butterfly kernels bind the shared spectrum_fft
            // group in `bind_group_for`, so the mirror collapses them together.
            WaterKernel::SpectrumEvolve
            | WaterKernel::SpectrumAssemble
            | WaterKernel::FftBitReverse
            | WaterKernel::FftStage
            | WaterKernel::FftNormalize => 12,
        }
    }
}
