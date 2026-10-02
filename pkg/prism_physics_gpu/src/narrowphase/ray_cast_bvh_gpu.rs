//! `GPU`-driven broad-phase-accelerated ray cast: the device twin of
//! [`ray_cast_bvh`](super::ray_cast_bvh::ray_cast_bvh). A ray is a zero-radius
//! point core swept across the ray span, so this composes exactly the proven
//! device shape-cast path — the
//! [`GpuBvhShapeCast`](super::shape_cast_bvh_gpu::GpuBvhShapeCast) `BVH` gather
//! and conservative-advancement sweep, each with its own real-device parity
//! suite — and only rewrites the substep fraction into a travelled distance.
//!
//! # Body layout
//!
//! Body `0` is the synthesised ray point core and bodies `1..=n` are the
//! targets, so target slot `i` lives at body index `i + 1` and the returned
//! [`RayCastHit::target`] is the `0`-based target index, matching the `CPU`
//! [`ray_cast_bvh`] convention.
//!
//! # Why the result matches the `CPU` query
//!
//! The synthesised body table is the identical point-core-versus-targets scene
//! the `CPU` query feeds its own shape cast, and the device shape cast is pinned
//! slot-for-slot to the `CPU` shape cast by its parity suite. Rewriting both
//! results through the shared
//! [`ray_cast_hit`](super::ray_cast_bvh::ray_cast_hit) map therefore yields the
//! identical ray hits, which the accompanying parity suite asserts.
//!
//! # Provenance
//!
//! Broad-phase ray/sweep gather (Jolt / `PhysX` / `Chaos` pattern), `LBVH` per
//! Karras 2012, stackless traversal per Hapala 2011, conservative advancement
//! per Mirtich 2000 over a Gilbert-Johnson-Keerthi distance walk (van den
//! Bergen, 2004). No Unreal Engine source or derived code.

use crate::bvh::{Aabb, GpuResidentLbvh};
use crate::GpuContext;

use super::body_motion::BodyMotion;
use super::convex_hull::ConvexHull;
use super::convex_pose::ConvexPose;
use super::ray_cast_bvh::{ray_cast_hit, RayCastHit, SceneRay};
use super::shape_cast::ShapeCastHit;
use super::shape_cast_bvh_gpu::GpuBvhShapeCast;

/// Composes the device shape-cast path into a `GPU`-driven ray cast. Build it
/// once per device and reuse it across casts; the inner shape cast compiles its
/// pipelines on construction.
pub struct GpuSceneRayCast {
    cast: GpuBvhShapeCast,
}

/// The synthesised body-indexed kernel tables for a ray cast: body `0` is the
/// ray's point core, bodies `1..` are the targets in slice order.
type Bodies = (Vec<ConvexHull>, Vec<ConvexPose>, Vec<BodyMotion>, Vec<f32>);

impl GpuSceneRayCast {
    /// Compiles the inner shape-cast pipelines on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSceneRayCast {
        GpuSceneRayCast {
            cast: GpuBvhShapeCast::new(ctx),
        }
    }

    /// Builds the per-target static bounding boxes that seed a resident `BVH`
    /// for repeated casts against an unchanging scene. The box order is the
    /// resident tree's leaf order, which is exactly the `0`-based target index
    /// [`cast_resident`](Self::cast_resident) and
    /// [`cast_all_resident`](Self::cast_all_resident) return. Delegates to the
    /// inner shape cast so the boxes match the swept-bounds the gather uses.
    #[must_use]
    pub fn scene_boxes(
        target_hulls: &[ConvexHull],
        target_poses: &[ConvexPose],
        target_radii: &[f32],
    ) -> Vec<Aabb> {
        GpuBvhShapeCast::scene_boxes(target_hulls, target_poses, target_radii)
    }

    /// Synthesises the body-indexed kernel tables: the ray's zero-radius point
    /// core as body `0`, then the targets in slice order as bodies `1..`.
    fn bodies(
        target_hulls: &[ConvexHull],
        target_poses: &[ConvexPose],
        target_radii: &[f32],
        ray: &SceneRay,
    ) -> Bodies {
        let n = target_hulls.len();
        let mut hulls = Vec::with_capacity(n + 1);
        hulls.push(ConvexHull::from_point());
        hulls.extend_from_slice(target_hulls);

        let mut poses = Vec::with_capacity(n + 1);
        poses.push(ray.pose());
        poses.extend_from_slice(target_poses);

        let mut motions = Vec::with_capacity(n + 1);
        motions.push(ray.motion());
        motions.extend(core::iter::repeat_n(BodyMotion::still(), n));

        let mut radii = Vec::with_capacity(n + 1);
        radii.push(0.0);
        radii.extend_from_slice(target_radii);

        (hulls, poses, motions, radii)
    }

    /// `GPU`-driven form of [`ray_cast_bvh`](super::ray_cast_bvh::ray_cast_bvh):
    /// the nearest target the ray pierces, matching the `CPU` query exactly.
    /// Builds a `BVH` per cast; use [`cast_resident`](Self::cast_resident) for an
    /// unchanging scene.
    #[must_use]
    pub fn cast(
        &self,
        ctx: &GpuContext,
        target_hulls: &[ConvexHull],
        target_poses: &[ConvexPose],
        target_radii: &[f32],
        ray: &SceneRay,
    ) -> Option<RayCastHit> {
        let (hulls, poses, motions, radii) =
            Self::bodies(target_hulls, target_poses, target_radii, ray);
        self.cast
            .cast(ctx, &hulls, &poses, &motions, &radii, 1.0, 0.0)
            .map(|hit: ShapeCastHit| ray_cast_hit(ray, hit))
    }

    /// `GPU`-driven form of
    /// [`ray_cast_all_bvh`](super::ray_cast_bvh::ray_cast_all_bvh): every target
    /// the ray pierces, ordered by increasing distance with ties broken by
    /// ascending target index, matching the `CPU` query exactly.
    #[must_use]
    pub fn cast_all(
        &self,
        ctx: &GpuContext,
        target_hulls: &[ConvexHull],
        target_poses: &[ConvexPose],
        target_radii: &[f32],
        ray: &SceneRay,
    ) -> Vec<RayCastHit> {
        let (hulls, poses, motions, radii) =
            Self::bodies(target_hulls, target_poses, target_radii, ray);
        self.cast
            .cast_all(ctx, &hulls, &poses, &motions, &radii, 1.0, 0.0)
            .into_iter()
            .map(|hit| ray_cast_hit(ray, hit))
            .collect()
    }

    /// Resident-tree form of [`cast`](Self::cast): reuses a `BVH` built once from
    /// [`scene_boxes`](Self::scene_boxes) instead of rebuilding per cast.
    ///
    /// # Contract
    ///
    /// `lbvh` must be the resident tree built from
    /// [`scene_boxes`](Self::scene_boxes) over the same `target_hulls`,
    /// `target_poses`, `target_radii` in that order, so its leaf count equals the
    /// target count and leaf `i` is target `i`. The result matches the `CPU`
    /// [`ray_cast_bvh`](super::ray_cast_bvh::ray_cast_bvh) golden.
    #[must_use]
    pub fn cast_resident(
        &self,
        ctx: &GpuContext,
        lbvh: &GpuResidentLbvh,
        target_hulls: &[ConvexHull],
        target_poses: &[ConvexPose],
        target_radii: &[f32],
        ray: &SceneRay,
    ) -> Option<RayCastHit> {
        let (hulls, poses, motions, radii) =
            Self::bodies(target_hulls, target_poses, target_radii, ray);
        self.cast
            .cast_resident(ctx, lbvh, &hulls, &poses, &motions, &radii, 1.0, 0.0)
            .map(|hit| ray_cast_hit(ray, hit))
    }

    /// Resident-tree form of [`cast_all`](Self::cast_all): reuses a `BVH` built
    /// once instead of rebuilding per cast. The `lbvh` contract is identical to
    /// [`cast_resident`](Self::cast_resident).
    #[must_use]
    pub fn cast_all_resident(
        &self,
        ctx: &GpuContext,
        lbvh: &GpuResidentLbvh,
        target_hulls: &[ConvexHull],
        target_poses: &[ConvexPose],
        target_radii: &[f32],
        ray: &SceneRay,
    ) -> Vec<RayCastHit> {
        let (hulls, poses, motions, radii) =
            Self::bodies(target_hulls, target_poses, target_radii, ray);
        self.cast
            .cast_all_resident(ctx, lbvh, &hulls, &poses, &motions, &radii, 1.0, 0.0)
            .into_iter()
            .map(|hit| ray_cast_hit(ray, hit))
            .collect()
    }
}
