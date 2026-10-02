//! `GPU`-driven broad-phase-accelerated box cast: the device twin of
//! [`box_cast_bvh`](super::box_cast_bvh::box_cast_bvh). A box cast is a six-face
//! box hull swept across the cast span, so this composes exactly the proven
//! device shape-cast path — the
//! [`GpuBvhShapeCast`](super::shape_cast_bvh_gpu::GpuBvhShapeCast) `BVH` gather
//! and conservative-advancement sweep, each with its own real-device parity
//! suite — and only rewrites the substep fraction into a travelled distance.
//!
//! # Body layout
//!
//! Body `0` is the synthesised box core and bodies `1..=n` are the targets, so
//! target slot `i` lives at body index `i + 1` and the returned
//! [`BoxCastHit::target`] is the `0`-based target index, matching the `CPU`
//! [`box_cast_bvh`] convention.
//!
//! # Why the result matches the `CPU` query
//!
//! The synthesised body table is the identical box-core-versus-targets scene the
//! `CPU` query feeds its own shape cast, and the device shape cast is pinned
//! slot-for-slot to the `CPU` shape cast by its parity suite. Rewriting both
//! results through the shared
//! [`box_cast_hit`](super::box_cast_bvh::box_cast_hit) map therefore yields the
//! identical box hits, which the accompanying parity suite asserts.
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
use super::box_cast_bvh::{box_cast_hit, BoxCastHit, SceneBoxCast};
use super::convex_hull::ConvexHull;
use super::convex_pose::ConvexPose;
use super::shape_cast::ShapeCastHit;
use super::shape_cast_bvh_gpu::GpuBvhShapeCast;

/// Composes the device shape-cast path into a `GPU`-driven box cast. Build it
/// once per device and reuse it across casts; the inner shape cast compiles its
/// pipelines on construction.
pub struct GpuSceneBoxCast {
    cast: GpuBvhShapeCast,
}

/// The synthesised body-indexed kernel tables for a box cast: body `0` is the
/// box core, bodies `1..` are the targets in slice order.
type Bodies = (Vec<ConvexHull>, Vec<ConvexPose>, Vec<BodyMotion>, Vec<f32>);

impl GpuSceneBoxCast {
    /// Compiles the inner shape-cast pipelines on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSceneBoxCast {
        GpuSceneBoxCast {
            cast: GpuBvhShapeCast::new(ctx),
        }
    }

    /// Builds the per-target static bounding boxes that seed a resident `BVH`
    /// for repeated casts against an unchanging scene. The box order is the
    /// resident tree's leaf order, which is exactly the `0`-based target index
    /// [`cast_resident`](Self::cast_resident) and
    /// [`cast_all_resident`](Self::cast_all_resident) return. Delegates to the
    /// inner shape cast so the boxes match the swept bounds the gather uses.
    #[must_use]
    pub fn scene_boxes(
        target_hulls: &[ConvexHull],
        target_poses: &[ConvexPose],
        target_radii: &[f32],
    ) -> Vec<Aabb> {
        GpuBvhShapeCast::scene_boxes(target_hulls, target_poses, target_radii)
    }

    /// Synthesises the body-indexed kernel tables: the cast's box core as body
    /// `0`, then the targets in slice order as bodies `1..`.
    fn bodies(
        target_hulls: &[ConvexHull],
        target_poses: &[ConvexPose],
        target_radii: &[f32],
        cast: &SceneBoxCast,
    ) -> Bodies {
        let n = target_hulls.len();
        let mut hulls = Vec::with_capacity(n + 1);
        hulls.push(cast.core());
        hulls.extend_from_slice(target_hulls);

        let mut poses = Vec::with_capacity(n + 1);
        poses.push(cast.pose());
        poses.extend_from_slice(target_poses);

        let mut motions = Vec::with_capacity(n + 1);
        motions.push(cast.motion());
        motions.extend(core::iter::repeat_n(BodyMotion::still(), n));

        let mut radii = Vec::with_capacity(n + 1);
        radii.push(0.0);
        radii.extend_from_slice(target_radii);

        (hulls, poses, motions, radii)
    }

    /// `GPU`-driven form of
    /// [`box_cast_bvh`](super::box_cast_bvh::box_cast_bvh): the nearest target
    /// the box touches, matching the `CPU` query exactly. Builds a `BVH` per
    /// cast; use [`cast_resident`](Self::cast_resident) for an unchanging scene.
    #[must_use]
    pub fn cast(
        &self,
        ctx: &GpuContext,
        target_hulls: &[ConvexHull],
        target_poses: &[ConvexPose],
        target_radii: &[f32],
        cast: &SceneBoxCast,
    ) -> Option<BoxCastHit> {
        let (hulls, poses, motions, radii) =
            Self::bodies(target_hulls, target_poses, target_radii, cast);
        self.cast
            .cast(ctx, &hulls, &poses, &motions, &radii, 1.0, 0.0)
            .map(|hit: ShapeCastHit| box_cast_hit(cast, hit))
    }

    /// `GPU`-driven form of
    /// [`box_cast_all_bvh`](super::box_cast_bvh::box_cast_all_bvh): every target
    /// the box touches, ordered by increasing distance with ties broken by
    /// ascending target index, matching the `CPU` query exactly.
    #[must_use]
    pub fn cast_all(
        &self,
        ctx: &GpuContext,
        target_hulls: &[ConvexHull],
        target_poses: &[ConvexPose],
        target_radii: &[f32],
        cast: &SceneBoxCast,
    ) -> Vec<BoxCastHit> {
        let (hulls, poses, motions, radii) =
            Self::bodies(target_hulls, target_poses, target_radii, cast);
        self.cast
            .cast_all(ctx, &hulls, &poses, &motions, &radii, 1.0, 0.0)
            .into_iter()
            .map(|hit| box_cast_hit(cast, hit))
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
    /// [`box_cast_bvh`](super::box_cast_bvh::box_cast_bvh) golden.
    #[must_use]
    pub fn cast_resident(
        &self,
        ctx: &GpuContext,
        lbvh: &GpuResidentLbvh,
        target_hulls: &[ConvexHull],
        target_poses: &[ConvexPose],
        target_radii: &[f32],
        cast: &SceneBoxCast,
    ) -> Option<BoxCastHit> {
        let (hulls, poses, motions, radii) =
            Self::bodies(target_hulls, target_poses, target_radii, cast);
        self.cast
            .cast_resident(ctx, lbvh, &hulls, &poses, &motions, &radii, 1.0, 0.0)
            .map(|hit| box_cast_hit(cast, hit))
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
        cast: &SceneBoxCast,
    ) -> Vec<BoxCastHit> {
        let (hulls, poses, motions, radii) =
            Self::bodies(target_hulls, target_poses, target_radii, cast);
        self.cast
            .cast_all_resident(ctx, lbvh, &hulls, &poses, &motions, &radii, 1.0, 0.0)
            .into_iter()
            .map(|hit| box_cast_hit(cast, hit))
            .collect()
    }
}
