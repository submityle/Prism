//! `GPU`-driven broad-phase-accelerated collide-point query: the device twin
//! of [`collide_point_bvh`](super::collide_point_bvh::collide_point_bvh). A
//! point-containment test is a still, zero-radius point core, so this composes
//! exactly the proven device shape-cast path — the
//! [`GpuBvhShapeCast`](super::shape_cast_bvh_gpu::GpuBvhShapeCast) `BVH` gather
//! and conservative-advancement overlap test, each with its own real-device
//! parity suite — and only discards the degenerate travel distance, keeping the
//! containing target index.
//!
//! # Body layout
//!
//! Body `0` is the synthesised still point core and bodies `1..=n` are the
//! targets, so target slot `i` lives at body index `i + 1` and the returned
//! [`CollidePointHit::target`] is the `0`-based target index, matching the `CPU`
//! [`collide_point_bvh`] convention.
//!
//! # Why the result matches the `CPU` query
//!
//! The synthesised body table is the identical still-point-versus-targets scene
//! the `CPU` query feeds its own shape cast, and the device shape cast is pinned
//! slot-for-slot to the `CPU` shape cast by its parity suite. Mapping both
//! results through the shared
//! [`collide_point_hit`](super::collide_point_bvh::collide_point_hit) rewrite
//! therefore yields the identical containing targets, which the accompanying
//! parity suite asserts.
//!
//! [`CollidePointHit::target`]: super::collide_point_bvh::CollidePointHit::target
//! [`collide_point_bvh`]: super::collide_point_bvh::collide_point_bvh
//!
//! # Provenance
//!
//! Broad-phase overlap gather (Jolt / `PhysX` / `Chaos` pattern), `LBVH` per
//! Karras 2012, stackless traversal per Hapala 2011, conservative advancement
//! per Mirtich 2000 over a Gilbert-Johnson-Keerthi distance walk (van den
//! Bergen, 2004). No Unreal Engine source or derived code.

use crate::bvh::{Aabb, GpuResidentLbvh};
use crate::GpuContext;

use super::body_motion::BodyMotion;
use super::collide_point_bvh::{collide_point_hit, CollidePointHit, ScenePoint};
use super::convex_hull::ConvexHull;
use super::convex_pose::ConvexPose;
use super::shape_cast_bvh_gpu::GpuBvhShapeCast;

/// Composes the device shape-cast path into a `GPU`-driven collide-point query.
/// Build it once per device and reuse it across queries; the inner shape cast
/// compiles its pipelines on construction.
pub struct GpuSceneCollidePoint {
    cast: GpuBvhShapeCast,
}

/// The synthesised body-indexed kernel tables for a collide-point query: body
/// `0` is the still point core, bodies `1..` are the targets in slice order.
type Bodies = (Vec<ConvexHull>, Vec<ConvexPose>, Vec<BodyMotion>, Vec<f32>);

impl GpuSceneCollidePoint {
    /// Compiles the inner shape-cast pipelines on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSceneCollidePoint {
        GpuSceneCollidePoint {
            cast: GpuBvhShapeCast::new(ctx),
        }
    }

    /// Builds the per-target static bounding boxes that seed a resident `BVH`
    /// for repeated queries against an unchanging scene. The box order is the
    /// resident tree's leaf order, which is exactly the `0`-based target index
    /// [`collide_resident`](Self::collide_resident) returns. Delegates to the
    /// inner shape cast so the boxes match the bounds the gather uses.
    #[must_use]
    pub fn scene_boxes(
        target_hulls: &[ConvexHull],
        target_poses: &[ConvexPose],
        target_radii: &[f32],
    ) -> Vec<Aabb> {
        GpuBvhShapeCast::scene_boxes(target_hulls, target_poses, target_radii)
    }

    /// Synthesises the body-indexed kernel tables: the still, zero-radius point
    /// core as body `0`, then the targets in slice order as bodies `1..`.
    fn bodies(
        target_hulls: &[ConvexHull],
        target_poses: &[ConvexPose],
        target_radii: &[f32],
        point: &ScenePoint,
    ) -> Bodies {
        let n = target_hulls.len();
        let mut hulls = Vec::with_capacity(n + 1);
        hulls.push(ConvexHull::from_point());
        hulls.extend_from_slice(target_hulls);

        let mut poses = Vec::with_capacity(n + 1);
        poses.push(point.pose());
        poses.extend_from_slice(target_poses);

        let mut motions = Vec::with_capacity(n + 1);
        motions.push(BodyMotion::still());
        motions.extend(core::iter::repeat_n(BodyMotion::still(), n));

        let mut radii = Vec::with_capacity(n + 1);
        radii.push(0.0);
        radii.extend_from_slice(target_radii);

        (hulls, poses, motions, radii)
    }

    /// `GPU`-driven form of
    /// [`collide_point_bvh`](super::collide_point_bvh::collide_point_bvh): every
    /// target whose convex volume contains `point`, in ascending target index,
    /// matching the `CPU` query exactly. Builds a `BVH` per query; use
    /// [`collide_resident`](Self::collide_resident) for an unchanging scene.
    #[must_use]
    pub fn collide(
        &self,
        ctx: &GpuContext,
        target_hulls: &[ConvexHull],
        target_poses: &[ConvexPose],
        target_radii: &[f32],
        point: &ScenePoint,
    ) -> Vec<CollidePointHit> {
        let (hulls, poses, motions, radii) =
            Self::bodies(target_hulls, target_poses, target_radii, point);
        self.cast
            .cast_all(ctx, &hulls, &poses, &motions, &radii, 1.0, 0.0)
            .into_iter()
            .map(collide_point_hit)
            .collect()
    }

    /// Resident-tree form of [`collide`](Self::collide): reuses a `BVH` built
    /// once from [`scene_boxes`](Self::scene_boxes) instead of rebuilding per
    /// query.
    ///
    /// # Contract
    ///
    /// `lbvh` must be the resident tree built from
    /// [`scene_boxes`](Self::scene_boxes) over the same `target_hulls`,
    /// `target_poses`, `target_radii` in that order, so its leaf count equals the
    /// target count and leaf `i` is target `i`. The result matches the `CPU`
    /// [`collide_point_bvh`](super::collide_point_bvh::collide_point_bvh) golden.
    #[must_use]
    pub fn collide_resident(
        &self,
        ctx: &GpuContext,
        lbvh: &GpuResidentLbvh,
        target_hulls: &[ConvexHull],
        target_poses: &[ConvexPose],
        target_radii: &[f32],
        point: &ScenePoint,
    ) -> Vec<CollidePointHit> {
        let (hulls, poses, motions, radii) =
            Self::bodies(target_hulls, target_poses, target_radii, point);
        self.cast
            .cast_all_resident(ctx, lbvh, &hulls, &poses, &motions, &radii, 1.0, 0.0)
            .into_iter()
            .map(collide_point_hit)
            .collect()
    }
}
