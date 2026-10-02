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

use crate::bvh::{cpu_build_lbvh, GpuBvhOverlap};
use crate::GpuContext;

use super::body_motion::BodyMotion;
use super::conservative_advancement_gpu::GpuConvexConvexToiNarrowphase;
use super::convex_hull::ConvexHull;
use super::convex_pose::ConvexPose;
use super::shape_cast::{RoundedConvex, ShapeCastHit};
use super::shape_cast_bvh::{consider, gather_boxes, sort_hits};
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
}
