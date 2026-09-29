//! Render-world resources that hold the resident `GPU` water bodies.
//!
//! The [`WaterComputePipelines`](super::pipeline::WaterComputePipelines) own the
//! sixteen pipelines and twelve bind-group layouts once for the whole app, and
//! [`bind_groups`](super::bind_groups) knows how to build one body's resident
//! buffers and twelve bind groups. This module is the render-world container
//! that ties a body's bind groups to the ordered golden dispatch schedule the
//! [`dispatch`](super::dispatch) node records.
//!
//! The resource is deliberately a plain `Vec` of bodies: the extract stage
//! (which would snapshot the main-world water bodies into these render
//! resources) rebuilds it each frame, and an empty vector makes the dispatch
//! node a genuine no-op rather than a fake solve. Until a main-world author
//! populates it, the vector stays empty and the pass is honestly skipped.

use bevy_ecs::resource::Resource;

use prism_render_architecture::water::gpu::pipeline::PlannedDispatch;

use super::bind_groups::{WaterBodyBindGroups, WaterBodyGpuBuffers};

/// One resident water body the dispatch node can record a frame's solve for.
///
/// Owns the resident buffers (so their `wgpu` handles outlive the bind groups
/// that reference them), the twelve bind groups the sixteen kernels dispatch
/// against, and the flat, ordered [`PlannedDispatch`] schedule the golden
/// `prepare` stage produced for this body's solver bucket and per-kernel
/// workgroup counts.
pub(crate) struct WaterGpuBody {
    /// The resident buffer set backing every bind group of this body.
    ///
    /// Held purely to keep the `wgpu` buffer and texture handles alive for as
    /// long as the bind groups that reference them; the dispatch node binds
    /// through the bind groups and never reads this field directly, so it is an
    /// intentionally unread `RAII` handle.
    #[expect(
        dead_code,
        reason = "an RAII handle held only to keep the wgpu buffers and \
                  textures alive as long as the bind groups that reference \
                  them; the dispatch node binds through the bind groups and \
                  never reads this field directly"
    )]
    pub(crate) buffers: WaterBodyGpuBuffers,
    /// The twelve bind groups, one per shader-interface layout.
    pub(crate) bind_groups: WaterBodyBindGroups,
    /// The ordered dispatch schedule in exact golden record order.
    pub(crate) dispatches: Vec<PlannedDispatch>,
}

impl WaterGpuBody {
    /// Builds a body from its resident buffers, bind groups and golden
    /// schedule. Kept explicit (rather than a struct literal at the call site)
    /// so the extract stage constructs bodies through one documented entry.
    #[must_use]
    pub(crate) fn new(
        buffers: WaterBodyGpuBuffers,
        bind_groups: WaterBodyBindGroups,
        dispatches: Vec<PlannedDispatch>,
    ) -> Self {
        Self {
            buffers,
            bind_groups,
            dispatches,
        }
    }
}

/// The render-world set of all resident `GPU` water bodies for this frame.
///
/// Populated by the extract stage and consumed by the
/// [`dispatch_water`](super::dispatch::dispatch_water) node. Defaults to empty,
/// which the node treats as "no water to solve this frame".
#[derive(Resource, Default)]
pub(crate) struct WaterGpuBodies {
    /// Every resident body scheduled this frame, in a stable order.
    pub(crate) bodies: Vec<WaterGpuBody>,
}

impl WaterGpuBodies {
    /// Returns `true` when there is no water to solve this frame, so the
    /// dispatch node can skip recording a compute pass entirely.
    #[must_use]
    pub(crate) fn is_empty(&self) -> bool {
        self.bodies.is_empty()
    }
}
