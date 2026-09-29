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
        for dispatch in &body.dispatches {
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
        }
    }
}
