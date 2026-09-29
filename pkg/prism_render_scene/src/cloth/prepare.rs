//! Device-side preparation of the resident `GPU` cloth pieces.
//!
//! This system runs in [`bevy_render::Render`] under
//! [`bevy_render::RenderSystems::PrepareResources`] and turns each extracted
//! [`ClothGarment`](super::garment::ClothGarment) into a resident
//! [`ClothGpuPiece`]. It is the device half of the prepare stage: the
//! device-free [`build_solve_plan`] does all the packing, coloring, counting and
//! scheduling, and this system only allocates the `wgpu` buffers, builds the
//! five bind groups and records the resident piece plus its golden dispatch
//! schedule.
//!
//! Pieces are rebuilt every frame from scratch (clear-then-refill), matching the
//! extract stage: the resident buffers are recreated and re-uploaded, which is
//! the honest baseline before a later slice folds in persistent buffers with
//! `queue`-side rewrites. A garment whose plan has zero particles contributes
//! no piece, so an empty or degenerate garment never fabricates a solve.

use bevy_ecs::prelude::*;
use bevy_render::renderer::RenderDevice;

use super::bind_groups::{ClothPieceBindGroups, ClothPieceGpuBuffers, ClothPieceUpload};
use super::garment::ExtractedCloth;
use super::pipeline::ClothComputePipelines;
use super::resources::{ClothGpuPiece, ClothGpuPieces};
use super::solve_plan::build_solve_plan;

/// Rebuilds the resident `GPU` cloth pieces from the extracted garments.
///
/// Clears the existing pieces and, for every extracted garment, builds its
/// device-free solve plan, allocates the resident buffers, builds the five bind
/// groups and pushes the resulting [`ClothGpuPiece`] with its golden dispatch
/// schedule. Garments whose plan has no particles are skipped so the dispatch
/// node never records an empty solve.
pub(crate) fn prepare_cloth_pieces(
    mut pieces: ResMut<ClothGpuPieces>,
    extracted: Res<ExtractedCloth>,
    pipelines: Option<Res<ClothComputePipelines>>,
    device: Res<RenderDevice>,
) {
    pieces.pieces.clear();

    // The pipelines are built once at `RenderStartup`; if that resource is not
    // present yet there is nothing to bind against, so skip this frame rather
    // than fabricate a piece.
    let Some(pipelines) = pipelines else {
        return;
    };

    for garment in &extracted.garments {
        let input = garment.as_solve_input();
        let plan = build_solve_plan(&input);
        if plan.counts.particles == 0 {
            continue;
        }

        // Read the pass-through arrays from the single borrowed `input` view so
        // `ClothSolveInput` is the one source for the raw particle/collider data
        // while `plan` supplies the reordered constraint and bending records.
        let upload = ClothPieceUpload {
            positions: input.positions,
            velocities: input.velocities,
            constraints: &plan.constraints,
            bending: &plan.bending,
            colliders: input.colliders,
            backstops: input.backstops,
            embed_bindings: input.embed_bindings,
            render_vertex_count: plan.counts.render_vertices,
            hash_cell_count: plan.counts.hash_cells,
            sim_params: plan.sim_params,
            body_params: plan.body_params,
            self_params: plan.self_params,
            backstop_params: plan.backstop_params,
            embed_params: plan.embed_params,
        };

        let buffers = ClothPieceGpuBuffers::create(&device, &upload);
        let bind_groups = ClothPieceBindGroups::create(&device, &pipelines, &buffers);
        pieces
            .pieces
            .push(ClothGpuPiece::new(buffers, bind_groups, plan.dispatches));
    }
}
