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
//! Pieces are *persistent*: [`ClothGpuPieces`] keys resident buffers by their
//! main-world entity, so a garment's device state survives across frames. When a
//! garment's buffer topology is unchanged this stage reuses its resident piece in
//! place — leaving the particle position/velocity pools resident so the `GPU`
//! evolves them across frames (the persistent, `GPU`-driven behavior the design
//! mandates) and only restreaming the small per-frame dynamic inputs (the uniform
//! parameter blocks and the kinematic collider proxies) through
//! [`ClothPieceGpuBuffers::write_dynamic`]. A garment that first appears, or whose
//! topology changes (a re-authored mesh or an LOD tier swap that resizes a pool),
//! gets a fresh allocation; a garment that despawns has its resident buffers
//! evicted. A garment whose plan has zero particles contributes no piece, so an
//! empty or degenerate garment never fabricates a solve.
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
use bevy_platform::collections::HashSet;
use bevy_render::renderer::{RenderDevice, RenderQueue};

use super::bind_groups::{ClothPieceBindGroups, ClothPieceGpuBuffers, ClothPieceUpload};
use super::budget::{admitted_garment_mask, ClothDeformationBudget};
use super::garment::ExtractedCloth;
use super::pipeline::ClothComputePipelines;
use super::resources::{ClothGpuPiece, ClothGpuPieces, ClothPieceSignature};
use super::solve_plan::build_solve_plan;

/// Rebuilds the resident `GPU` cloth pieces from the extracted garments.
///
/// Opens a new frame (keeping resident buffers), resolves the shared
/// deformation-budget verdict for the whole extracted set, then, for every
/// extracted garment, resolves its screen-coverage LOD tier and skips the garment
/// when that tier is not simulated (the skinned proxy) or when the budget
/// deferred it this frame, otherwise builds its device-free solve plan and either
/// *reuses* the garment's resident piece in place — restreaming only the dynamic
/// inputs so the `GPU` keeps evolving the resident simulation state — or, when the
/// garment is new or its buffer topology changed, allocates fresh resident buffers
/// and bind groups and installs a new [`ClothGpuPiece`]. Garments whose plan has
/// no particles are skipped so the dispatch node never records an empty solve.
/// Finally, resident pieces whose garment despawned this frame are evicted.
pub(crate) fn prepare_cloth_pieces(
    mut pieces: ResMut<ClothGpuPieces>,
    extracted: Res<ExtractedCloth>,
    budget: Res<ClothDeformationBudget>,
    pipelines: Option<Res<ClothComputePipelines>>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    // Open a new frame: drop last frame's dispatch order but keep every resident
    // piece so a reused garment can evolve its simulation state in place.
    pieces.begin_frame();

    // The pipelines are built once at `RenderStartup`; if that resource is not
    // present yet there is nothing to bind against, so skip this frame rather
    // than fabricate a piece. Resident buffers are left intact so no state is
    // lost while the pipelines finish compiling.
    let Some(pipelines) = pipelines else {
        return;
    };

    // The extract stage keeps the entity and garment arrays strictly parallel;
    // the per-garment entity below indexes straight into it.
    debug_assert_eq!(
        extracted.entities.len(),
        extracted.garments.len(),
        "extract keeps the entity and garment arrays in lockstep"
    );

    // Shared deformation-budget arbitration over the whole extracted set: a
    // `false` slot is either a non-simulated proxy or a simulated garment the
    // budget deferred to a later frame. The default unlimited budget admits
    // every simulated garment, keeping this a transparent pass-through.
    let admitted = admitted_garment_mask(&extracted.garments, budget.budget);

    for (index, garment) in extracted.garments.iter().enumerate() {
        // The garment's stable main-world entity keys its resident piece so its
        // device state persists across frames.
        let entity = extracted.entities[index];

        // Screen-coverage LOD gate: a garment whose coverage collapses it to a
        // non-simulated tier (the skinned proxy) builds no resident piece, so the
        // dispatch node records no compute pass for it. Simulated tiers still
        // solve: full sim solves the authored mesh, and reduced sim solves the
        // garment's coarse LOD mesh when one was authored (else it falls back to
        // the full mesh). Reuses the architecture-layer golden classifier through
        // the garment's own decision.
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

        // Tier-aware solve view: at the reduced-simulation tier a garment that
        // authored a coarse LOD mesh solves that mesh instead of the full one,
        // landing the reduced tier's real per-frame cost reduction. Every other
        // tier (and the reduced tier with no coarse mesh) solves the full mesh.
        let input = garment.as_solve_input_for_tier(lod.tier);
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

        // The schedule is rebuilt every frame — it is cheap, and the substep and
        // iteration counts can change even when the buffer topology is identical —
        // so move it out before borrowing the rest of the plan for the upload.
        let dispatches = plan.dispatches;

        // The fingerprint of every resident pool's size. Reuse is only sound while
        // it is unchanged, because the dynamic-input rewrites assume the existing
        // allocations still fit exactly.
        let signature = ClothPieceSignature::from_upload(&upload);

        // Reuse the resident piece when its topology is unchanged: leave the
        // simulation-state pools resident so the `GPU` evolves them in place, and
        // restream only the per-frame dynamic inputs. Otherwise (re)allocate.
        let can_reuse = pieces
            .resident_mut(entity)
            .is_some_and(|existing| existing.signature == signature);

        if can_reuse {
            let existing = pieces
                .resident_mut(entity)
                .expect("resident piece was just observed present");
            existing.buffers.write_dynamic(&queue, &upload);
            // Teleport gate: a garment whose reference frame jumped this frame has
            // bumped its teleport generation past the one this resident piece last
            // applied. Snap the resident simulation-state pools back to the authored
            // pose exactly once (the generation only advances on `request_teleport`),
            // so the solver does not stretch the stale resident state across the jump.
            // A `Continuous`-latched bump restreams nothing, collapsing to a no-op.
            let generation = garment.teleport_generation();
            if generation != existing.applied_teleport_generation {
                let mode = garment.teleport_mode();
                if mode.restreams_any() {
                    existing
                        .buffers
                        .apply_teleport(&queue, &upload, mode.restream());
                }
                existing.applied_teleport_generation = generation;
            }
            existing.dispatches = dispatches;
            pieces.mark_active(entity);
        } else {
            let buffers = ClothPieceGpuBuffers::create(&device, &upload);
            let bind_groups = ClothPieceBindGroups::create(&device, &pipelines, &buffers);
            // A fresh piece's create upload already streamed the garment's authored
            // pose, so it records the current generation as applied: only a *later*
            // bump restreams. This keeps a garment spawned mid-teleport from
            // re-snapping on its very first resident frame.
            pieces.install(
                entity,
                ClothGpuPiece::new(
                    buffers,
                    bind_groups,
                    dispatches,
                    signature,
                    garment.teleport_generation(),
                ),
            );
        }
    }

    // Evict resident buffers for garments that despawned entirely this frame. A
    // garment merely gated out (LOD-collapsed or budget-deferred) is still present
    // in the extracted set, so its resident state is preserved and re-offered next
    // frame; only a despawned garment is dropped, freeing its `GPU` buffers.
    let live: HashSet<Entity> = extracted.entities.iter().copied().collect();
    pieces.retain_live(&live);
}
