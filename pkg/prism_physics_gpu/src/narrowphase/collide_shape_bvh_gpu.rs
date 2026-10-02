//! `GPU`-driven broad-phase-accelerated collide-shape query: the device twin
//! of [`collide_shape_bvh`](super::collide_shape_bvh::collide_shape_bvh). It
//! composes two kernels that each already carry their own real-device parity
//! suite — the [`GpuBvhOverlap`](crate::bvh::GpuBvhOverlap) gather and the
//! [`GpuConvexConvexManifoldNarrowphase`] contact generator — into the exact
//! gather-then-manifold pipeline `AAA` engines run for scene overlap queries
//! (Jolt `NarrowPhaseQuery::CollideShape`, `PhysX` `PxScene::overlap`, Unreal
//! `Chaos` shape overlaps): descend an acceleration structure over the scene
//! with the query's bounds to collect candidates on the `GPU`, then build the
//! exact convex-versus-convex manifold only on those candidates, also on the
//! `GPU`.
//!
//! # Body layout
//!
//! Body `0` is the query shape and bodies `1..=n` are the targets, so target
//! slot `i` lives at body index `i + 1`. The returned
//! [`CollideShapeHit::target`] is the zero-based target index, matching the
//! `CPU` [`collide_shape_bvh`] convention.
//!
//! # Why the result matches the `CPU` query
//!
//! The gather descends the identical [`gather_boxes`] geometry the `CPU` query
//! builds, so the `GPU` candidate set is the identical conservative superset of
//! the truly-overlapping targets. The survivors are run through the same
//! manifold kernel the per-pair parity suite pins to the `CPU` golden, and the
//! penetration-only [`None`] slots are dropped and the hits re-sorted with the
//! shared [`sort_hits`] rule the `CPU` query uses. The accompanying parity suite
//! asserts the `GPU`-derived hits match the `CPU` [`collide_shape_bvh`] golden
//! slot for slot.
//!
//! [`GpuConvexConvexManifoldNarrowphase`]: super::convex_convex_manifold_gpu::GpuConvexConvexManifoldNarrowphase
//! [`gather_boxes`]: super::collide_shape_bvh::gather_boxes
//! [`sort_hits`]: super::collide_shape_bvh::sort_hits
//! [`collide_shape_bvh`]: super::collide_shape_bvh::collide_shape_bvh
//! [`CollideShapeHit::target`]: super::collide_shape_bvh::CollideShapeHit
//!
//! # Provenance
//!
//! Broad-phase overlap gather (Jolt / `PhysX` / `Chaos` pattern), `LBVH` per
//! Karras 2012, stackless traversal per Hapala 2011, convex-versus-convex
//! manifold via Gilbert-Johnson-Keerthi 1988 / expanding-polytope (van den
//! Bergen 2001) with Sutherland-Hodgman clipping. No Unreal Engine source or
//! derived code.

use crate::bvh::{cpu_build_lbvh, Aabb, GpuBvhOverlap, GpuResidentLbvh};
use crate::GpuContext;

use super::collide_shape_bvh::{gather_boxes, sort_hits, world_aabb, CollideShapeHit};
use super::convex_convex_manifold::ConvexConvexPair;
use super::convex_convex_manifold_gpu::GpuConvexConvexManifoldNarrowphase;
use super::convex_hull::ConvexHull;
use super::convex_pose::ConvexPose;

/// Composes the `BVH` overlap gather and the convex-versus-convex manifold
/// kernels into a `GPU`-driven collide-shape query. Build it once per device
/// and reuse it across queries; both inner kernels compile their pipelines on
/// construction.
pub struct GpuBvhCollideShape {
    overlap: GpuBvhOverlap,
    manifold: GpuConvexConvexManifoldNarrowphase,
}

impl GpuBvhCollideShape {
    /// Compiles the gather and manifold pipelines on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuBvhCollideShape {
        GpuBvhCollideShape {
            overlap: GpuBvhOverlap::new(ctx),
            manifold: GpuConvexConvexManifoldNarrowphase::new(ctx),
        }
    }

    /// Builds the per-target static bounding boxes that seed a resident `BVH`
    /// for repeated queries against an unchanging scene (the Jolt / `PhysX` /
    /// `Chaos` persistent-broad-phase pattern). Feed the result to
    /// [`GpuLbvh::build_resident`](crate::bvh::GpuLbvh::build_resident) or
    /// [`ResidentBvhDriver::build`](crate::bvh::ResidentBvhDriver::build): the
    /// box order is the resident tree's leaf order, which is exactly the
    /// `0`-based target index [`collide_resident`](Self::collide_resident)
    /// returns. `margin` should be `0` here; the per-query margin is applied to
    /// the moving query box at traversal time, not baked into the static tree.
    #[must_use]
    pub fn scene_boxes(
        target_hulls: &[ConvexHull],
        target_poses: &[ConvexPose],
        margin: f32,
    ) -> Vec<Aabb> {
        (0..target_hulls.len())
            .map(|i| world_aabb(&target_hulls[i], &target_poses[i], margin))
            .collect()
    }

    /// Gathers candidate target indices by building a `BVH` over the targets'
    /// bounds and descending it with the query shape's grown box, all on the
    /// `GPU`. The `Err` arm falls back to the whole scene so correctness never
    /// depends on the gather succeeding.
    fn gather(&self, ctx: &GpuContext, hulls: &[ConvexHull], poses: &[ConvexPose], margin: f32) -> Vec<u32> {
        let num_targets = hulls.len() - 1;
        if num_targets == 0 {
            return Vec::new();
        }
        let (target_boxes, query) = gather_boxes(hulls, poses, margin);
        let lbvh = cpu_build_lbvh(&target_boxes);
        let capacity = u32::try_from(num_targets).unwrap_or(u32::MAX);
        match self.overlap.query(ctx, &lbvh, &[query], capacity) {
            Ok(mut per_query) => per_query.pop().unwrap_or_default(),
            Err(_) => (0..capacity).collect(),
        }
    }

    /// Gathers candidate target indices by descending a resident `BVH` already
    /// built from [`scene_boxes`](Self::scene_boxes) with the query shape's
    /// grown box, reusing the on-device tree instead of rebuilding one per
    /// query.
    fn gather_resident(
        &self,
        ctx: &GpuContext,
        lbvh: &GpuResidentLbvh,
        hulls: &[ConvexHull],
        poses: &[ConvexPose],
        margin: f32,
    ) -> Vec<u32> {
        let (_target_boxes, query) = gather_boxes(hulls, poses, margin);
        let capacity = u32::try_from(lbvh.num_leaves()).unwrap_or(u32::MAX);
        match self.overlap.query_resident(ctx, lbvh, &[query], capacity) {
            Ok(mut per_query) => per_query.pop().unwrap_or_default(),
            Err(_) => (0..capacity).collect(),
        }
    }

    /// Runs the manifold kernel over the query shape (body `0`) versus each
    /// gathered candidate, dropping the separated couples and sorting the
    /// survivors by ascending target index.
    fn manifolds_for(
        &self,
        ctx: &GpuContext,
        hulls: &[ConvexHull],
        poses: &[ConvexPose],
        candidates: &[u32],
    ) -> Vec<CollideShapeHit> {
        if candidates.is_empty() {
            return Vec::new();
        }
        let pairs: Vec<ConvexConvexPair> = candidates
            .iter()
            .map(|&target| ConvexConvexPair::new(0, target + 1))
            .collect();
        let manifolds = self.manifold.query(ctx, hulls, poses, &pairs);
        let mut hits: Vec<CollideShapeHit> = candidates
            .iter()
            .zip(manifolds)
            .filter_map(|(&target, slot)| slot.map(|manifold| CollideShapeHit { target, manifold }))
            .collect();
        sort_hits(&mut hits);
        hits
    }

    /// `GPU`-driven form of
    /// [`collide_shape_bvh`](super::collide_shape_bvh::collide_shape_bvh): every
    /// target the query shape (body `0`) overlaps and the manifold against each,
    /// in ascending target order, matching the `CPU` query exactly. Builds a
    /// `BVH` per call; use [`collide_resident`](Self::collide_resident) for an
    /// unchanging scene.
    #[must_use]
    pub fn collide(
        &self,
        ctx: &GpuContext,
        hulls: &[ConvexHull],
        poses: &[ConvexPose],
        margin: f32,
    ) -> Vec<CollideShapeHit> {
        if hulls.len() < 2 {
            return Vec::new();
        }
        let candidates = self.gather(ctx, hulls, poses, margin);
        self.manifolds_for(ctx, hulls, poses, &candidates)
    }

    /// Resident-tree form of [`collide`](Self::collide): reuses a `BVH` built
    /// once instead of rebuilding per query.
    ///
    /// # Contract
    ///
    /// `lbvh` must be the resident tree built from
    /// [`scene_boxes`](Self::scene_boxes) over the targets' `hulls[1..]`,
    /// `poses[1..]` in that order, so its leaf count equals the target count and
    /// leaf `i` is target `i` (body `i + 1`). Build it from a mismatched or
    /// reordered scene and the returned target indices are meaningless. The
    /// result matches the `CPU`
    /// [`collide_shape_bvh`](super::collide_shape_bvh::collide_shape_bvh) golden.
    #[must_use]
    pub fn collide_resident(
        &self,
        ctx: &GpuContext,
        lbvh: &GpuResidentLbvh,
        hulls: &[ConvexHull],
        poses: &[ConvexPose],
        margin: f32,
    ) -> Vec<CollideShapeHit> {
        if hulls.len() < 2 {
            return Vec::new();
        }
        let candidates = self.gather_resident(ctx, lbvh, hulls, poses, margin);
        self.manifolds_for(ctx, hulls, poses, &candidates)
    }
}
