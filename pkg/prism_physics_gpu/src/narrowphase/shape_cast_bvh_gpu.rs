//! GPU-driven broad-phase-accelerated shape cast: the device twin of
//! [`cast_shape_bvh`](super::shape_cast_bvh::cast_shape_bvh). It composes two
//! kernels that each already carry their own real-device parity suite — the
//! [`GpuBvhOverlap`](crate::bvh::GpuBvhOverlap) gather and the
//! [`GpuConvexConvexToiNarrowphase`] per-pair time of impact — into the exact
//! gather-then-sweep pipeline `AAA` engines run (Jolt `NarrowPhaseQuery` cast,
//! `PhysX` scene sweeps, Unreal `Chaos` sweeps): descend an acceleration
//! structure over the scene with the moving query's swept bounds to collect
//! candidates on the `GPU`, then run the exact per-pair sweep only on those
//! candidates, also on the `GPU`.
//!
//! # Body layout
//!
//! The input tables are body-indexed exactly like the per-pair kernel: body `0`
//! is the moving cast shape and bodies `1..=n` are the targets, so target slot
//! `i` lives at body index `i + 1`. The returned [`ShapeCastHit::target`] is the
//! zero-based target index, matching the `CPU` [`cast_shape_bvh`] convention.
//!
//! # Why the result matches the `CPU` cast
//!
//! The gather descends the identical [`gather_boxes`] geometry the `CPU` cast
//! builds, so the `GPU` candidate set is the identical conservative superset of
//! the truly-struck targets. The surviving candidates are swept by the same
//! conservative-advancement kernel the per-pair parity suite pins to the `CPU`
//! golden, and the readback is reduced with the shared [`consider`] earliest-hit
//! rule and [`sort_hits`] ordering the `CPU` cast uses. The accompanying parity
//! suite asserts the `GPU`-derived hit matches the `CPU` [`cast_shape_bvh`]
//! golden slot for slot.
//!
//! [`GpuConvexConvexToiNarrowphase`]: super::conservative_advancement_gpu::GpuConvexConvexToiNarrowphase
//! [`gather_boxes`]: super::shape_cast_bvh::gather_boxes
//! [`consider`]: super::shape_cast_bvh::consider
//! [`sort_hits`]: super::shape_cast_bvh::sort_hits
//! [`cast_shape_bvh`]: super::shape_cast_bvh::cast_shape_bvh
//! [`ShapeCastHit::target`]: super::shape_cast::ShapeCastHit
//!
//! # Provenance
//!
//! Broad-phase sweep gather (Jolt / `PhysX` / `Chaos` pattern), `LBVH` per Karras
//! 2012, conservative advancement per Mirtich 2000 over a `GJK` distance walk
//! (van den Bergen 2004; Gilbert-Johnson-Keerthi 1988). No Unreal Engine source
//! or derived code.

use crate::bvh::{cpu_build_lbvh, Aabb, GpuBvhOverlap, GpuResidentLbvh};
use crate::GpuContext;

use super::body_motion::BodyMotion;
use super::conservative_advancement_gpu::GpuConvexConvexToiNarrowphase;
use super::convex_hull::ConvexHull;
use super::convex_pose::ConvexPose;
use super::shape_cast::{RoundedConvex, ShapeCastHit};
use super::shape_cast_bvh::{consider, gather_boxes, sort_hits, swept_aabb};
use super::{ConvexConvexSweepPair, Toi};

/// Composes the `BVH` overlap gather and the convex-convex time-of-impact
/// kernels into a `GPU`-driven shape cast. Build it once per device and reuse it
/// across casts; both inner kernels compile their pipelines on construction.
pub struct GpuBvhShapeCast {
    overlap: GpuBvhOverlap,
    toi: GpuConvexConvexToiNarrowphase,
}

impl GpuBvhShapeCast {
    /// Compiles the gather and time-of-impact pipelines on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuBvhShapeCast {
        GpuBvhShapeCast {
            overlap: GpuBvhOverlap::new(ctx),
            toi: GpuConvexConvexToiNarrowphase::new(ctx),
        }
    }

    /// Gathers the candidate target indices (into the `0`-based target space) by
    /// descending a `BVH` over the targets' swept bounds with the moving shape's
    /// swept query box, all on the `GPU`.
    ///
    /// The returned indices are a conservative superset of every truly-struck
    /// target. The gather can never report more hits than the target count, so
    /// the per-query capacity is that count; the `Err` arm nonetheless falls back
    /// to the whole scene so correctness never depends on the gather succeeding.
    fn gather(
        &self,
        ctx: &GpuContext,
        shape: &RoundedConvex,
        targets: &[RoundedConvex],
        dt: f32,
        target_sep: f32,
    ) -> Vec<u32> {
        if targets.is_empty() {
            return Vec::new();
        }
        let (target_boxes, query) = gather_boxes(shape, targets, dt, target_sep);
        let lbvh = cpu_build_lbvh(&target_boxes);
        let capacity = u32::try_from(targets.len()).unwrap_or(u32::MAX);
        match self.overlap.query(ctx, &lbvh, &[query], capacity) {
            Ok(mut per_query) => per_query.pop().unwrap_or_default(),
            Err(_) => (0..capacity).collect(),
        }
    }

    /// Runs the per-pair time-of-impact kernel over the cast shape (body `0`)
    /// versus each gathered candidate, returning each candidate paired with its
    /// time of impact (candidates that miss within `dt` are dropped).
    #[expect(
        clippy::too_many_arguments,
        reason = "the body-indexed kernel tables (hulls, poses, motions, radii) plus the gathered candidates, the context, and the two step scalars are each distinct inputs"
    )]
    fn sweep_candidates(
        &self,
        ctx: &GpuContext,
        hulls: &[ConvexHull],
        poses: &[ConvexPose],
        motions: &[BodyMotion],
        radii: &[f32],
        candidates: &[u32],
        dt: f32,
        target_sep: f32,
    ) -> Vec<ShapeCastHit> {
        if candidates.is_empty() {
            return Vec::new();
        }
        let pairs: Vec<ConvexConvexSweepPair> = candidates
            .iter()
            .map(|&c| ConvexConvexSweepPair::new(0, c + 1))
            .collect();
        let results: Vec<Option<Toi>> =
            self.toi
                .query(ctx, hulls, poses, motions, radii, &pairs, dt, target_sep);
        candidates
            .iter()
            .zip(results)
            .filter_map(|(&target, slot)| slot.map(|toi| ShapeCastHit { target, toi }))
            .collect()
    }

    /// Builds the body-indexed `RoundedConvex` views: body `0` is the shape,
    /// bodies `1..` are the targets, in order.
    fn views<'a>(
        hulls: &'a [ConvexHull],
        poses: &[ConvexPose],
        motions: &[BodyMotion],
        radii: &[f32],
    ) -> (RoundedConvex<'a>, Vec<RoundedConvex<'a>>) {
        let shape = RoundedConvex::new(&hulls[0], poses[0], motions[0], radii[0]);
        let targets = (1..hulls.len())
            .map(|i| RoundedConvex::new(&hulls[i], poses[i], motions[i], radii[i]))
            .collect();
        (shape, targets)
    }

    /// `GPU`-driven form of
    /// [`cast_shape_bvh`](super::shape_cast_bvh::cast_shape_bvh): the earliest
    /// contact of the moving shape (body `0`) against the targets (bodies `1..`),
    /// or `None` when nothing is reached within `dt`. The struck target index,
    /// impact time, contact point, and normal match the `CPU` cast, including the
    /// lower-index rule on an exact time tie.
    #[must_use]
    pub fn cast(
        &self,
        ctx: &GpuContext,
        hulls: &[ConvexHull],
        poses: &[ConvexPose],
        motions: &[BodyMotion],
        radii: &[f32],
        dt: f32,
        target_sep: f32,
    ) -> Option<ShapeCastHit> {
        if hulls.is_empty() {
            return None;
        }
        let (shape, targets) = Self::views(hulls, poses, motions, radii);
        let candidates = self.gather(ctx, &shape, &targets, dt, target_sep);
        let hits = self.sweep_candidates(ctx, hulls, poses, motions, radii, &candidates, dt, target_sep);
        let mut best: Option<ShapeCastHit> = None;
        for hit in hits {
            consider(&mut best, hit);
        }
        best
    }

    /// `GPU`-driven form of
    /// [`cast_shape_all_bvh`](super::shape_cast_bvh::cast_shape_all_bvh): every
    /// contact within `dt`, ordered by increasing time of impact with ties broken
    /// by ascending target index, matching the `CPU` cast exactly.
    #[must_use]
    pub fn cast_all(
        &self,
        ctx: &GpuContext,
        hulls: &[ConvexHull],
        poses: &[ConvexPose],
        motions: &[BodyMotion],
        radii: &[f32],
        dt: f32,
        target_sep: f32,
    ) -> Vec<ShapeCastHit> {
        if hulls.is_empty() {
            return Vec::new();
        }
        let (shape, targets) = Self::views(hulls, poses, motions, radii);
        let candidates = self.gather(ctx, &shape, &targets, dt, target_sep);
        let mut hits =
            self.sweep_candidates(ctx, hulls, poses, motions, radii, &candidates, dt, target_sep);
        sort_hits(&mut hits);
        hits
    }

    /// Builds the per-target static bounding boxes that seed a resident `BVH`
    /// for repeated casts against an unchanging scene — the persistent
    /// broad-phase pattern Jolt (`NarrowPhaseQuery` against the `BroadPhase`),
    /// `PhysX` (`PxScene` sweeps against the pruning structure), and Unreal
    /// `Chaos` all run. Feed the result to
    /// [`GpuLbvh::build_resident`](crate::bvh::GpuLbvh::build_resident) or
    /// [`ResidentBvhDriver::build`](crate::bvh::ResidentBvhDriver::build): the
    /// box order is the resident tree's leaf order, which is exactly the
    /// `0`-based target index [`cast_resident`](Self::cast_resident) and
    /// [`cast_all_resident`](Self::cast_all_resident) return.
    ///
    /// The targets must be static. [`swept_aabb`](super::shape_cast_bvh::swept_aabb)
    /// of a still rounded convex does not depend on `dt`, so a tree built here
    /// stays valid across every frame the targets do not move; only the moving
    /// cast shape's query box is re-swept per cast.
    #[must_use]
    pub fn scene_boxes(
        target_hulls: &[ConvexHull],
        target_poses: &[ConvexPose],
        target_radii: &[f32],
    ) -> Vec<Aabb> {
        (0..target_hulls.len())
            .map(|i| {
                let target =
                    RoundedConvex::still(&target_hulls[i], target_poses[i], target_radii[i]);
                swept_aabb(&target, 0.0, 0.0)
            })
            .collect()
    }

    /// Gathers candidate target indices by descending a resident `BVH` already
    /// built from [`scene_boxes`](Self::scene_boxes) with the moving shape's
    /// swept query box, all on the `GPU`. Unlike [`gather`](Self::gather) this
    /// reuses the on-device tree instead of rebuilding one per cast, which is
    /// the whole point of the persistent broad-phase query path.
    ///
    /// The returned indices are a conservative superset of every truly-struck
    /// target. The gather can never report more hits than the leaf count, so the
    /// per-query capacity is that count; the `Err` arm falls back to the whole
    /// scene so correctness never depends on the gather succeeding.
    fn gather_resident(
        &self,
        ctx: &GpuContext,
        lbvh: &GpuResidentLbvh,
        shape: &RoundedConvex,
        dt: f32,
        target_sep: f32,
    ) -> Vec<u32> {
        let capacity = u32::try_from(lbvh.num_leaves()).unwrap_or(u32::MAX);
        // A resident tree with fewer than two leaves owns no traversable
        // internal hierarchy, so the device descent can never reach its lone
        // leaf and would spuriously report an empty candidate set. Fall back to
        // the trivial scene (every leaf) exactly as the per-call gather does for
        // a single-leaf CPU tree; the exact sweep then accepts or rejects it, so
        // correctness never depends on the degenerate descent.
        if lbvh.num_internal() == 0 {
            return (0..capacity).collect();
        }
        let query = swept_aabb(shape, dt, target_sep.max(0.0));
        match self.overlap.query_resident(ctx, lbvh, &[query], capacity) {
            Ok(mut per_query) => per_query.pop().unwrap_or_default(),
            Err(_) => (0..capacity).collect(),
        }
    }

    /// Resident-tree form of [`cast`](Self::cast): the earliest contact of the
    /// moving shape (body `0`) against the static targets (bodies `1..`), or
    /// `None` when nothing is reached within `dt`, reusing a `BVH` built once
    /// instead of rebuilding per cast.
    ///
    /// # Contract
    ///
    /// `lbvh` must be the resident tree built from
    /// [`scene_boxes`](Self::scene_boxes) over the targets' `hulls[1..]`,
    /// `poses[1..]`, `radii[1..]` in that order, so its leaf count equals the
    /// target count and leaf `i` is target `i` (body `i + 1`). Build it from a
    /// mismatched or reordered scene and the returned target indices are
    /// meaningless. The result — struck target index, impact time, contact
    /// point, and normal — matches the `CPU`
    /// [`cast_shape_bvh`](super::shape_cast_bvh::cast_shape_bvh) golden,
    /// including the lower-index rule on an exact time tie.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "the resident tree, the body-indexed kernel tables (hulls, poses, motions, radii), the context, and the two step scalars are each distinct inputs"
    )]
    pub fn cast_resident(
        &self,
        ctx: &GpuContext,
        lbvh: &GpuResidentLbvh,
        hulls: &[ConvexHull],
        poses: &[ConvexPose],
        motions: &[BodyMotion],
        radii: &[f32],
        dt: f32,
        target_sep: f32,
    ) -> Option<ShapeCastHit> {
        if hulls.is_empty() {
            return None;
        }
        let shape = RoundedConvex::new(&hulls[0], poses[0], motions[0], radii[0]);
        let candidates = self.gather_resident(ctx, lbvh, &shape, dt, target_sep);
        let hits =
            self.sweep_candidates(ctx, hulls, poses, motions, radii, &candidates, dt, target_sep);
        let mut best: Option<ShapeCastHit> = None;
        for hit in hits {
            consider(&mut best, hit);
        }
        best
    }

    /// Resident-tree form of [`cast_all`](Self::cast_all): every contact within
    /// `dt`, ordered by increasing time of impact with ties broken by ascending
    /// target index, matching the `CPU`
    /// [`cast_shape_all_bvh`](super::shape_cast_bvh::cast_shape_all_bvh) exactly,
    /// reusing a `BVH` built once instead of rebuilding per cast.
    ///
    /// The `lbvh` contract is identical to [`cast_resident`](Self::cast_resident).
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "the resident tree, the body-indexed kernel tables (hulls, poses, motions, radii), the context, and the two step scalars are each distinct inputs"
    )]
    pub fn cast_all_resident(
        &self,
        ctx: &GpuContext,
        lbvh: &GpuResidentLbvh,
        hulls: &[ConvexHull],
        poses: &[ConvexPose],
        motions: &[BodyMotion],
        radii: &[f32],
        dt: f32,
        target_sep: f32,
    ) -> Vec<ShapeCastHit> {
        if hulls.is_empty() {
            return Vec::new();
        }
        let shape = RoundedConvex::new(&hulls[0], poses[0], motions[0], radii[0]);
        let candidates = self.gather_resident(ctx, lbvh, &shape, dt, target_sep);
        let mut hits =
            self.sweep_candidates(ctx, hulls, poses, motions, radii, &candidates, dt, target_sep);
        sort_hits(&mut hits);
        hits
    }
}
