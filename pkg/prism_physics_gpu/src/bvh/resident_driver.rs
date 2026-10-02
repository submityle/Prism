//! End-to-end resident `BVH` refit-versus-rebuild driver.
//!
//! [`ResidentBvhDriver`] owns a device-resident `LBVH` and the three kernels and
//! one host policy needed to keep it query-ready every frame without a full host
//! round-trip: [`GpuLbvh`] to (re)build it, [`GpuBvhRefit`] to refresh its bounds
//! in place, [`GpuBvhSahCost`] to measure its quality on device, and
//! [`RefitQualityTracker`] to decide when a refit no longer pays off and a rebuild
//! is cheaper. The driver fuses them into one call: hand it the frame's primitive
//! boxes and it returns which action it took and the resulting `SAH` cost.
//!
//! The policy mirrors a production incremental `BVH` (Unreal's Chaos keeps the
//! topology and refits bounds for a bounded number of frames, then rebuilds once
//! the tree's surface-area cost has inflated past a factor of its freshly built
//! baseline): a bit-level refit is cheap and preserves traversal determinism, so
//! it is the default; a rebuild is spent only when the cost-growth or staleness
//! trigger fires. Because the cost is read straight from the resident buffers,
//! the whole decision is one scalar readback per sum, not a tree download.
//!
//! # Provenance
//!
//! Composes Prism's own Karras (2012) builder, bottom-up refit, Goldsmith and
//! Salmon (1987) `SAH` reducer, and surface-area rebuild policy. No Unreal Engine
//! source or derived code.

use crate::context::GpuContext;

use super::config::Aabb;
use super::gpu::GpuLbvh;
use super::quality::{
    RebuildDecision, RefitQualityTracker, DEFAULT_MAX_REFITS_BETWEEN_REBUILDS,
    DEFAULT_REBUILD_COST_FACTOR,
};
use super::refit_gpu::GpuBvhRefit;
use super::resident::GpuResidentLbvh;
use super::sah_cost_gpu::GpuBvhSahCost;

/// Which maintenance action the driver took for a frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UpdateAction {
    /// The resident topology was kept and only its bounds were refreshed.
    Refit,
    /// The tree was rebuilt from scratch, resetting the quality baseline.
    Rebuild,
}

/// The outcome of one [`ResidentBvhDriver::update`] call.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameUpdate {
    /// Whether the driver refit the existing tree or rebuilt it.
    pub action: UpdateAction,
    /// The `SAH` cost of the resident tree after the action, measured on device.
    pub cost: f32,
}

/// A self-maintaining device-resident `LBVH`.
///
/// Build it once from an initial primitive set, then call
/// [`ResidentBvhDriver::update`] each frame with that frame's boxes. The driver
/// refits in place while quality holds and rebuilds when the
/// [`RefitQualityTracker`] policy says a fresh tree is worth it, exposing the
/// current tree through [`ResidentBvhDriver::tree`] for traversal.
pub struct ResidentBvhDriver {
    /// Builds and rebuilds the resident tree.
    builder: GpuLbvh,
    /// Refreshes the resident bounds in place without changing topology.
    refit: GpuBvhRefit,
    /// Measures the resident tree's `SAH` cost on device.
    sah: GpuBvhSahCost,
    /// Host-side cost-growth and staleness policy.
    tracker: RefitQualityTracker,
    /// The live resident tree, bind-ready for queries.
    tree: GpuResidentLbvh,
    /// Leaf count of the current topology; a change forces a rebuild.
    leaf_count: usize,
}

impl ResidentBvhDriver {
    /// Builds the initial resident tree over `boxes` with the default rebuild
    /// policy and seeds the quality baseline from its measured cost.
    #[must_use]
    pub fn build(ctx: &GpuContext, boxes: &[Aabb]) -> ResidentBvhDriver {
        ResidentBvhDriver::build_with_policy(
            ctx,
            boxes,
            DEFAULT_REBUILD_COST_FACTOR,
            DEFAULT_MAX_REFITS_BETWEEN_REBUILDS,
        )
    }

    /// Builds the initial resident tree with an explicit rebuild policy.
    ///
    /// `rebuild_cost_factor` is the surface-area-cost growth (relative to the
    /// last rebuild's baseline) that forces a rebuild; `max_refits_between_rebuilds`
    /// is the staleness bound that forces one regardless of cost. See
    /// [`RefitQualityTracker::new`] for the exact trigger semantics.
    ///
    /// # Panics
    ///
    /// Panics under the same conditions as [`RefitQualityTracker::new`]: a
    /// `rebuild_cost_factor` below `1.0` or not finite.
    #[must_use]
    pub fn build_with_policy(
        ctx: &GpuContext,
        boxes: &[Aabb],
        rebuild_cost_factor: f32,
        max_refits_between_rebuilds: u32,
    ) -> ResidentBvhDriver {
        let builder = GpuLbvh::new(ctx);
        let refit = GpuBvhRefit::new(ctx);
        let sah = GpuBvhSahCost::new(ctx);
        let tree = builder.build_resident(ctx, boxes);
        let baseline = sah.sah_cost(ctx, &tree);
        let tracker =
            RefitQualityTracker::new(baseline, rebuild_cost_factor, max_refits_between_rebuilds);
        ResidentBvhDriver {
            builder,
            refit,
            sah,
            tracker,
            tree,
            leaf_count: boxes.len(),
        }
    }

    /// Updates the resident tree for this frame's `boxes` and returns the action
    /// taken and the resulting `SAH` cost.
    ///
    /// When `boxes` has the same leaf count as the current topology the driver
    /// refits in place, measures the new cost, and consults the policy: a
    /// [`RebuildDecision::Rebuild`] triggers a full rebuild and resets the
    /// baseline. A change in leaf count has no compatible topology to refit, so
    /// the driver rebuilds unconditionally and reseeds the baseline.
    pub fn update(&mut self, ctx: &GpuContext, boxes: &[Aabb]) -> FrameUpdate {
        if boxes.len() != self.leaf_count {
            return self.rebuild(ctx, boxes);
        }
        self.refit.refit(ctx, &self.tree, boxes);
        let cost = self.sah.sah_cost(ctx, &self.tree);
        match self.tracker.observe_refit(cost) {
            RebuildDecision::Refit => FrameUpdate {
                action: UpdateAction::Refit,
                cost,
            },
            RebuildDecision::Rebuild => self.rebuild(ctx, boxes),
        }
    }

    /// Rebuilds the tree from `boxes`, reseeds the baseline, and reports the result.
    fn rebuild(&mut self, ctx: &GpuContext, boxes: &[Aabb]) -> FrameUpdate {
        self.tree = self.builder.build_resident(ctx, boxes);
        self.leaf_count = boxes.len();
        let cost = self.sah.sah_cost(ctx, &self.tree);
        self.tracker.record_rebuild(cost);
        FrameUpdate {
            action: UpdateAction::Rebuild,
            cost,
        }
    }

    /// The live resident tree, ready to bind into a query or ray kernel.
    #[must_use]
    pub fn tree(&self) -> &GpuResidentLbvh {
        &self.tree
    }

    /// The quality tracker driving the refit-versus-rebuild policy.
    #[must_use]
    pub fn tracker(&self) -> &RefitQualityTracker {
        &self.tracker
    }

    /// The leaf count of the current resident topology.
    #[must_use]
    pub fn leaf_count(&self) -> usize {
        self.leaf_count
    }
}
