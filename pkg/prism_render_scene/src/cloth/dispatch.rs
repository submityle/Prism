//! The `Core3d` compute node that records one frame of `GPU` cloth solving.
//!
//! This node is a *dumb executor*: every ordering, sizing and per-color
//! addressing decision was already made — and unit-tested — in the float-free
//! golden plan
//! ([`prism_render_architecture::cloth::gpu::pipeline::prepare`]). Each resident
//! [`ClothGpuPiece`](super::resources::ClothGpuPiece) carries its
//! [`PlannedDispatch`] schedule in exact solver order, so all this node does is
//! walk that list, bind the matching group, push the per-color immediate for
//! the three projection kernels, and launch the recorded workgroup count.
//!
//! Readiness is all-or-nothing: if any of the thirteen pipelines is still
//! compiling this frame the node records nothing, rather than running a partial
//! solve that would leave the cloth in a half-stepped, non-deterministic state.

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{ComputePassDescriptor, PipelineCache},
    renderer::RenderContext,
};

use prism_render_architecture::cloth::gpu::kernels::ClothKernel;

use super::bind_groups::ClothPieceBindGroups;
use super::pipeline::ClothComputePipelines;
use super::resources::ClothGpuPieces;

/// Which of the seven group-0 bind groups a kernel dispatches against.
///
/// Mirrors [`ClothComputePipelines::layout`](super::pipeline::ClothComputePipelines::layout):
/// the six `cloth_sim.wesl` kernels share the simulation group, the two
/// self-collision kernels share the self group, and body / backstop / embed
/// each take their own. Kept as a pure enum so the mapping is exhaustively
/// unit-tested without a `GPU` device.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ClothBindSlot {
    /// The six `cloth_sim.wesl` kernels (predict / three projections / strain /
    /// velocity update).
    Sim,
    /// The body-collision kernel.
    Body,
    /// The two self-collision kernels (hash build + resolve).
    SelfCollision,
    /// The painted-backstop projection kernel.
    Backstop,
    /// The render-mesh skin-embed kernel.
    Embed,
    /// The aerodynamic velocity-snapshot kernel.
    AeroSnapshot,
    /// The aerodynamic per-vertex wind gather kernel.
    Aero,
}

/// Maps a golden [`ClothKernel`] to the bind-group slot its dispatch targets.
///
/// This is the single source of truth the dispatch loop uses to pick a piece's
/// bind group, so it stays lockstep with the shader interface the pipelines
/// were built against.
#[must_use]
fn bind_slot_for(kernel: ClothKernel) -> ClothBindSlot {
    match kernel {
        ClothKernel::Predict
        | ClothKernel::ProjectDistanceBatch
        | ClothKernel::ProjectBendingBatch
        | ClothKernel::ProjectLongRangeBatch
        | ClothKernel::StrainLimit
        | ClothKernel::VelocityUpdate => ClothBindSlot::Sim,
        ClothKernel::BodyCollision => ClothBindSlot::Body,
        ClothKernel::SelfCollisionHashBuild | ClothKernel::SelfCollisionResolve => {
            ClothBindSlot::SelfCollision
        }
        ClothKernel::Backstop => ClothBindSlot::Backstop,
        ClothKernel::SkinEmbed => ClothBindSlot::Embed,
        ClothKernel::AerodynamicsSnapshot => ClothBindSlot::AeroSnapshot,
        ClothKernel::Aerodynamics => ClothBindSlot::Aero,
    }
}

/// Borrows the concrete bind group a kernel dispatches against from a piece.
#[must_use]
fn bind_group_for<'a>(
    kernel: ClothKernel,
    groups: &'a ClothPieceBindGroups,
) -> &'a bevy_render::render_resource::BindGroup {
    match bind_slot_for(kernel) {
        ClothBindSlot::Sim => &groups.sim,
        ClothBindSlot::Body => &groups.body,
        ClothBindSlot::SelfCollision => &groups.self_collision,
        ClothBindSlot::Backstop => &groups.backstop,
        ClothBindSlot::Embed => &groups.embed,
        ClothBindSlot::AeroSnapshot => &groups.aero_snapshot,
        ClothBindSlot::Aero => &groups.aero,
    }
}

/// Records every resident cloth piece's solve for this frame.
///
/// Runs as a `Core3d` compute node before the main pass. No-ops when no cloth
/// is resident or while any pipeline is still compiling.
pub(crate) fn dispatch_cloth(
    pieces: Res<ClothGpuPieces>,
    pipelines: Res<ClothComputePipelines>,
    cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    if pieces.is_empty() {
        return;
    }

    // All-or-nothing readiness: bail before opening a pass if any kernel is
    // still compiling, so a frame never records a partial (non-deterministic)
    // solve. Every piece shares these thirteen pipelines.
    for kernel in ClothKernel::ALL {
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
            label: Some("prism GPU cloth solve"),
            timestamp_writes: None,
        });

    for piece in &pieces.pieces {
        for dispatch in &piece.dispatches {
            // Safe to expect: readiness was gated above and the pipeline set is
            // shared by every piece.
            let pipeline = cache
                .get_compute_pipeline(pipelines.pipeline(dispatch.kernel))
                .expect("cloth pipelines were all checked ready above");
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group_for(dispatch.kernel, &piece.bind_groups), &[]);

            // The three color-serial projection kernels address a single graph
            // color's contiguous constraint slice through the `ClothColorBatch`
            // immediate; the per-particle kernels carry no batch and read their
            // whole domain from the uniform counts.
            if let Some(batch) = dispatch.batch {
                let window: [u32; 2] = [batch.base, batch.count];
                pass.set_immediates(0, bytemuck::bytes_of(&window));
            }

            pass.dispatch_workgroups(dispatch.groups, 1, 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_kernel_maps_to_its_shader_interface_slot() {
        // The six sim kernels share the sim group.
        for kernel in [
            ClothKernel::Predict,
            ClothKernel::ProjectDistanceBatch,
            ClothKernel::ProjectBendingBatch,
            ClothKernel::ProjectLongRangeBatch,
            ClothKernel::StrainLimit,
            ClothKernel::VelocityUpdate,
        ] {
            assert_eq!(bind_slot_for(kernel), ClothBindSlot::Sim);
        }
        assert_eq!(
            bind_slot_for(ClothKernel::BodyCollision),
            ClothBindSlot::Body
        );
        assert_eq!(
            bind_slot_for(ClothKernel::SelfCollisionHashBuild),
            ClothBindSlot::SelfCollision
        );
        assert_eq!(
            bind_slot_for(ClothKernel::SelfCollisionResolve),
            ClothBindSlot::SelfCollision
        );
        assert_eq!(
            bind_slot_for(ClothKernel::Backstop),
            ClothBindSlot::Backstop
        );
        assert_eq!(bind_slot_for(ClothKernel::SkinEmbed), ClothBindSlot::Embed);
        assert_eq!(
            bind_slot_for(ClothKernel::AerodynamicsSnapshot),
            ClothBindSlot::AeroSnapshot
        );
        assert_eq!(
            bind_slot_for(ClothKernel::Aerodynamics),
            ClothBindSlot::Aero
        );
    }

    #[test]
    fn only_the_projection_kernels_are_color_serial() {
        // The kernels that carry a per-color immediate in the plan are exactly
        // the three that target the sim group and are color-serial; this keeps
        // the dispatch loop's immediate push aligned with the golden contract.
        for kernel in ClothKernel::ALL {
            let expects_batch = kernel.is_color_serial();
            let is_sim_projection = matches!(
                kernel,
                ClothKernel::ProjectDistanceBatch
                    | ClothKernel::ProjectBendingBatch
                    | ClothKernel::ProjectLongRangeBatch
            );
            assert_eq!(expects_batch, is_sim_projection);
            if expects_batch {
                assert_eq!(bind_slot_for(kernel), ClothBindSlot::Sim);
            }
        }
    }
}
