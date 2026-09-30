//! Render-world resources that hold the resident `GPU` cloth pieces.
//!
//! The [`ClothComputePipelines`](super::pipeline::ClothComputePipelines) own the
//! pipelines and layouts once for the whole app, and
//! [`bind_groups`](super::bind_groups) knows how to build one piece's resident
//! buffers and five bind groups. This module is the render-world container that
//! ties a piece's bind groups to the ordered golden dispatch schedule the
//! [`dispatch`](super::dispatch) node records.
//!
//! The resource is deliberately a plain `Vec` of pieces: the extract stage
//! (which snapshots the main-world cloth garments into these render resources)
//! rebuilds it each frame, and an empty vector makes the dispatch node a
//! genuine no-op rather than a fake solve.

use bevy_ecs::resource::Resource;

use prism_render_architecture::cloth::gpu::pipeline::PlannedDispatch;
use prism_render_architecture::cloth::lod::{ClothLodDecision, ClothLodPlan};

use super::bind_groups::{ClothPieceBindGroups, ClothPieceGpuBuffers};

/// One resident cloth piece the dispatch node can record a frame's solve for.
///
/// Owns the resident buffers (so their `wgpu` handles outlive the bind groups
/// that reference them), the five bind groups the eleven kernels dispatch
/// against, and the flat, ordered [`PlannedDispatch`] schedule the golden
/// `prepare` stage produced for this piece's colored constraint graph and
/// substep/iteration counts.
pub(crate) struct ClothGpuPiece {
    /// The resident buffer set backing every bind group of this piece.
    ///
    /// Held purely to keep the `wgpu` buffer handles alive for as long as the
    /// bind groups that reference them; the dispatch node binds through the
    /// bind groups and never reads this field directly, so it is an
    /// intentionally unread `RAII` handle.
    #[expect(
        dead_code,
        reason = "resident `wgpu` buffer handles kept alive purely as an `RAII` owner; the dispatch node binds through the derived bind groups and never reads this field directly"
    )]
    pub(crate) buffers: ClothPieceGpuBuffers,
    /// The seven group-0 bind groups, one per shader-interface layout.
    pub(crate) bind_groups: ClothPieceBindGroups,
    /// The ordered dispatch schedule in exact golden record order.
    pub(crate) dispatches: Vec<PlannedDispatch>,
    /// The LOD decision that admitted this piece this frame (always a simulated
    /// tier — skinned proxies never become resident). Surfaced so the renderer
    /// can bin resident pieces by tier and account for the sim budget through
    /// [`ClothGpuPieces::lod_plan`]; the dispatch node itself walks the golden
    /// schedule and never reads the tier directly, so the field is currently
    /// only consumed by that aggregation.
    pub(crate) lod: ClothLodDecision,
}

impl ClothGpuPiece {
    /// Builds a piece from its resident buffers, bind groups and golden
    /// schedule. Kept explicit (rather than a struct literal at the call site)
    /// so the extract stage constructs pieces through one documented entry.
    #[must_use]
    pub(crate) fn new(
        buffers: ClothPieceGpuBuffers,
        bind_groups: ClothPieceBindGroups,
        dispatches: Vec<PlannedDispatch>,
        lod: ClothLodDecision,
    ) -> Self {
        Self {
            buffers,
            bind_groups,
            dispatches,
            lod,
        }
    }
}

/// The render-world set of all resident `GPU` cloth pieces for this frame.
///
/// Populated by the extract stage and consumed by the
/// [`dispatch_cloth`](super::dispatch::dispatch_cloth) node. Defaults to empty,
/// which the node treats as "no cloth to solve this frame".
#[derive(Resource, Default)]
pub(crate) struct ClothGpuPieces {
    /// Every resident piece scheduled this frame, in a stable order.
    pub(crate) pieces: Vec<ClothGpuPiece>,
}

impl ClothGpuPieces {
    /// Returns `true` when there is no cloth to solve this frame, so the
    /// dispatch node can skip recording a compute pass entirely.
    #[must_use]
    pub(crate) fn is_empty(&self) -> bool {
        self.pieces.is_empty()
    }

    /// Bins the resident pieces by their resolved LOD tier for the frame.
    ///
    /// Every resident piece is a simulated tier (the prepare stage skips the
    /// skinned proxy before a piece is ever built), so the returned plan's
    /// `skinned_proxy` bucket is always empty here; the full/reduced buckets let
    /// the renderer submit and account for cloth work per tier in a stable
    /// input order.
    #[must_use]
    #[expect(
        dead_code,
        reason = "per-frame resident-piece tier accounting surface for the renderer binning path; wired ahead of the tier-aware submission slice that will consume it. The per-piece `lod` it bins is produced by the golden classifier covered in `cloth::lod` tests"
    )]
    pub(crate) fn lod_plan(&self) -> ClothLodPlan {
        let mut plan = ClothLodPlan::default();
        for piece in &self.pieces {
            plan.push(piece.lod);
        }
        plan
    }
}
