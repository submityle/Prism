//! Device-side preparation of the resident `GPU` cloth pieces.
//!
//! This system runs in [`bevy_render::Render`] under
//! [`bevy_render::RenderSystems::PrepareResources`] and turns each extracted
//! [`ClothGarment`](super::garment::ClothGarment) into a resident
//! [`ClothGpuPiece`]. It is the device half of the prepare stage: the
//! device-free [`build_solve_plan`] does all the packing, coloring, counting and
//! scheduling, and this system only allocates the `wgpu` buffers, builds the
//! seven bind groups and records the resident piece plus its golden dispatch
//! schedule.
//!
//! Pieces are rebuilt every frame from scratch (clear-then-refill), matching the
//! extract stage: the resident buffers are recreated and re-uploaded, which is
//! the honest baseline before a later slice folds in persistent buffers with
//! `queue`-side rewrites. A garment whose plan has zero particles contributes
//! no piece, so an empty or degenerate garment never fabricates a solve.
//!
//! Two gates upstream of the buffer allocation decide *whether* a garment solves
//! this frame, both honoring the same "no work, no dispatch" contract:
//!
//! * the screen-coverage LOD gate ([`super::lod`]): a garment whose coverage
//!   collapses it to the skinned proxy tier builds no piece;
//! * the shared deformation-budget gate ([`super::budget`]): when several
//!   simulated garments compete for one finite per-frame vertex budget, the
//!   lower-priority (lower-coverage) garments defer to a later frame and build
//!   no piece this frame. The budget defaults to unlimited, so the gate is a
//!   transparent pass-through until a scene opts into a finite ceiling.

use bevy_ecs::prelude::*;
use bevy_render::renderer::RenderDevice;

use super::bind_groups::{ClothPieceBindGroups, ClothPieceGpuBuffers, ClothPieceUpload};
use super::budget::{admitted_garment_mask, ClothDeformationBudget};
use super::garment::ExtractedCloth;
use super::pipeline::ClothComputePipelines;
use super::resources::{ClothGpuPiece, ClothGpuPieces};
use super::solve_plan::build_solve_plan;

/// Rebuilds the resident `GPU` cloth pieces from the extracted garments.
///
/// Clears the existing pieces and resolves the shared deformation-budget verdict
/// for the whole extracted set, then, for every extracted garment, resolves its
/// screen-coverage LOD tier and skips the garment when that tier is not
/// simulated (the skinned proxy) or when the budget deferred it this frame,
/// otherwise builds its device-free solve plan, allocates the resident buffers,
/// builds the seven bind groups and pushes the resulting [`ClothGpuPiece`] with
/// its golden dispatch schedule and resolved LOD decision. Garments whose plan
/// has no particles are likewise skipped so the dispatch node never records an
/// empty solve.
pub(crate) fn prepare_cloth_pieces(
    mut pieces: ResMut<ClothGpuPieces>,
    extracted: Res<ExtractedCloth>,
    budget: Res<ClothDeformationBudget>,
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

    // Shared deformation-budget arbitration over the whole extracted set: a
    // `false` slot is either a non-simulated proxy or a simulated garment the
    // budget deferred to a later frame. The default unlimited budget admits
    // every simulated garment, keeping this a transparent pass-through.
    let admitted = admitted_garment_mask(&extracted.garments, budget.budget);

    for (index, garment) in extracted.garments.iter().enumerate() {
        // Screen-coverage LOD gate: a garment whose coverage collapses it to a
        // non-simulated tier (the skinned proxy) builds no resident piece, so the
        // dispatch node records no compute pass for it. Simulated tiers (full and
        // reduced) still solve the authored mesh. Reuses the architecture-layer
        // golden classifier through the garment's own decision.
        let lod = garment.lod_decision();
        if !lod.tier.is_simulated() {
            continue;
        }

        // Deformation-budget gate: a simulated garment the shared arbiter
        // deferred this frame builds no resident piece, so the dispatch node
        // records no compute pass for it. It keeps last frame's embedded pose and
        // the arbiter re-offers it next frame.
        if !admitted[index] {
            continue;
        }

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
            triangles: input.triangles,
            csr_offsets: &plan.csr_offsets,
            csr_entries: &plan.csr_entries,
            aero_params: plan.aero_params,
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
        pieces.pieces.push(ClothGpuPiece::new(
            buffers,
            bind_groups,
            plan.dispatches,
            lod,
        ));
    }
}
