//! Narrow phase: candidate pairs and shape colliders to contact manifolds.
//!
//! The broad phase ([`crate::broadphase`]) prunes the `O(n^2)` collision test to
//! a set of candidate pairs whose bounding volumes *might* touch. The narrow
//! phase closes the remaining gap: it takes each candidate and produces the
//! precise contact a solver needs — a unit normal, a penetration depth, and a
//! world contact point — or rejects it when the shapes do not actually
//! penetrate. This is the missing link between the broad phase and the `XPBD`
//! constraint solver ([`crate::xpbd`]).
//!
//! # Shape library
//!
//! The narrow phase is organised one collider pair per module, each a
//! self-contained vertical slice: a shared geometry function, a `CPU` golden
//! twin, a `WGSL` kernel, and a device pipeline that runs the identical
//! arithmetic.
//!
//! - [`sphere`] / [`GpuNarrowphase`]: sphere versus sphere, the dynamic-dynamic
//!   primitive driving particle-particle collision.
//! - [`halfspace`] / [`GpuHalfspaceNarrowphase`]: sphere versus an infinite
//!   plane, the canonical static collider for grounds, walls, and frustum faces.
//! - [`capsule`] / [`GpuCapsuleNarrowphase`]: sphere versus a capsule (a segment
//!   swept by a radius), collapsing to sphere-sphere against the closest point
//!   on the segment.
//! - [`capsule_capsule`] / [`GpuCapsuleCapsuleNarrowphase`]: capsule versus
//!   capsule (dynamic-dynamic), finding the closest point pair between the two
//!   segments and collapsing to sphere-sphere there.
//! - [`capsule_halfspace`] / [`GpuCapsuleHalfspaceNarrowphase`]: capsule versus
//!   a halfspace, reporting up to two contact points (one per penetrating
//!   axis endpoint) so a capsule resting flat on a plane stays put.
//! - [`capsule_capsule_manifold`] / [`cpu_capsule_capsule_manifold`]:
//!   promotes the capsule-capsule contact to an up-to-two-point
//!   [`ContactManifold`] for near-parallel capsules by clipping the
//!   overlapping stretch of their axes, so two capsules resting side by side
//!   stay flush instead of rocking about one point.
//! - [`capsule_obb`] / [`GpuCapsuleObbNarrowphase`]: capsule versus an
//!   oriented bounding box, taking the point on the capsule axis closest to
//!   the box (an exact convex piecewise-quadratic segment-box minimisation)
//!   and collapsing to the sphere-versus-box manifold there.
//! - [`capsule_obb_manifold`] / [`cpu_capsule_obb_manifold`]: promotes the
//!   capsule-OBB contact to an up-to-two-point [`ContactManifold`] by
//!   clipping the capsule axis to the reference box face with the
//!   Liang-Barsky algorithm, the manifold a solver needs to hold a capsule
//!   resting flat on a face still.
//! - [`obb`] / [`GpuObbNarrowphase`]: sphere versus an oriented bounding box,
//!   clamping the sphere centre in the box frame with an interior push-out
//!   fallback through the least-penetrated face.
//! - [`obb_halfspace`] / [`GpuObbHalfspaceNarrowphase`]: oriented bounding box
//!   versus a halfspace, using the box support function along the plane normal
//!   to report the deepest penetrating vertex.
//! - [`obb_halfspace_manifold`] / [`cpu_obb_halfspace_manifold`]: promotes the
//!   OBB-halfspace contact to the full up-to-four-corner incident-face
//!   [`ContactManifold`], the manifold a solver needs to keep a box resting flat
//!   and stable on the ground.
//! - [`obb_obb`] / [`GpuObbObbNarrowphase`]: oriented bounding box versus
//!   oriented bounding box (dynamic-dynamic), a fifteen-axis separating-axis
//!   test reporting the minimum-translation contact.
//! - [`obb_obb_manifold`] / [`cpu_obb_obb_manifold`]: promotes the OBB-OBB
//!   contact to a multi-point [`ContactManifold`] by clipping the incident face
//!   against the reference face, the manifold a solver needs for stable
//!   stacking.
//!
//! Every pair emits one contact slot per input candidate: a passing real-device
//! parity test is direct evidence the ported kernel builds the same manifolds as
//! its twin.
//!
//! # Dense output
//!
//! Emitting a slot per candidate (rather than compacting on the fly) keeps the
//! contact index aligned with the input index, which the parity tests rely on
//! and which lets a downstream [`crate::scan`] compaction stream the survivors
//! without a second pass. It is a deliberate design choice, not a stub: the
//! `None` / invalid slots carry a real "no penetration" decision.
//!
//! Provenance: textbook collision-manifold construction; no Unreal Engine source
//! or derived code.

mod body_motion;
mod capsule;
mod capsule_capsule;
mod capsule_capsule_gpu;
mod capsule_capsule_manifold;
mod capsule_capsule_manifold_gpu;
mod capsule_gpu;
mod capsule_halfspace;
mod capsule_halfspace_gpu;
mod capsule_obb;
mod capsule_obb_gpu;
mod capsule_obb_manifold;
mod capsule_obb_manifold_gpu;
mod capsule_triangle;
mod capsule_triangle_gpu;
mod capsule_triangle_manifold;
mod capsule_triangle_manifold_gpu;
mod collide_point_bvh;
mod collide_point_bvh_gpu;
mod collide_shape_bvh;
mod collide_shape_bvh_gpu;
mod conservative_advancement;
mod conservative_advancement_gpu;
mod contact;
mod convex_convex_manifold;
mod convex_convex_manifold_gpu;
mod convex_hull;
mod convex_pose;
mod cpu;
mod epa;
mod gjk;
mod gpu;
mod halfspace;
mod halfspace_gpu;
mod heightfield;
mod heightfield_capsule_manifold;
mod heightfield_capsule_manifold_gpu;
mod heightfield_gpu;
mod heightfield_obb_manifold;
mod heightfield_obb_manifold_gpu;
mod layout;
mod manifold;
mod minkowski;
mod obb;
mod obb_gpu;
mod obb_halfspace;
mod obb_halfspace_gpu;
mod obb_halfspace_manifold;
mod obb_halfspace_manifold_gpu;
mod obb_obb;
mod obb_obb_gpu;
mod obb_obb_manifold;
mod obb_obb_manifold_gpu;
mod obb_triangle;
mod obb_triangle_gpu;
mod obb_triangle_manifold;
mod obb_triangle_manifold_gpu;
mod ray_cast_bvh;
mod ray_cast_bvh_gpu;
mod shape_cast;
mod shape_cast_bvh;
mod shape_cast_bvh_gpu;
mod box_cast_bvh;
mod box_cast_bvh_gpu;
mod convex_cast_bvh;
mod convex_cast_bvh_gpu;
mod capsule_cast_bvh;
mod capsule_cast_bvh_gpu;
mod sphere_cast_bvh;
mod sphere_cast_bvh_gpu;
mod sphere;
mod sphere_triangle;
mod sphere_triangle_gpu;

pub use body_motion::BodyMotion;
pub use capsule::{cpu_capsule_narrowphase, Capsule, SphereCapsulePair};
pub use capsule_capsule::{cpu_capsule_capsule_narrowphase, CapsuleCapsulePair};
pub use capsule_capsule_gpu::GpuCapsuleCapsuleNarrowphase;
pub use capsule_capsule_manifold::cpu_capsule_capsule_manifold;
pub use capsule_capsule_manifold_gpu::GpuCapsuleCapsuleManifoldNarrowphase;
pub use capsule_gpu::GpuCapsuleNarrowphase;
pub use capsule_halfspace::{cpu_capsule_halfspace_manifold, CapsulePlanePair};
pub use capsule_halfspace_gpu::GpuCapsuleHalfspaceNarrowphase;
pub use capsule_obb::{cpu_capsule_obb_narrowphase, CapsuleObbPair};
pub use capsule_obb_gpu::GpuCapsuleObbNarrowphase;
pub use capsule_obb_manifold::cpu_capsule_obb_manifold;
pub use capsule_obb_manifold_gpu::GpuCapsuleObbManifoldNarrowphase;
pub use capsule_triangle::{cpu_capsule_triangle_narrowphase, CapsuleTrianglePair};
pub use capsule_triangle_gpu::GpuCapsuleTriangleNarrowphase;
pub use capsule_triangle_manifold::cpu_capsule_triangle_manifold;
pub use capsule_triangle_manifold_gpu::GpuCapsuleTriangleManifoldNarrowphase;
pub use conservative_advancement::{
    conservative_advancement_toi, conservative_advancement_toi_rounded, cpu_convex_convex_toi,
    cpu_convex_convex_toi_rounded, ConvexConvexSweepPair, Toi,
};
pub use contact::Contact;
pub use convex_convex_manifold::{cpu_convex_convex_manifold, ConvexConvexPair};
pub use collide_point_bvh::{collide_point, collide_point_bvh, CollidePointHit, ScenePoint};
pub use collide_point_bvh_gpu::GpuSceneCollidePoint;
pub use collide_shape_bvh::{collide_shape, collide_shape_bvh, CollideShapeHit};
pub use collide_shape_bvh_gpu::GpuBvhCollideShape;
pub use conservative_advancement_gpu::GpuConvexConvexToiNarrowphase;
pub use ray_cast_bvh::{
    ray_cast, ray_cast_all, ray_cast_all_bvh, ray_cast_bvh, RayCastHit, SceneRay,
};
pub use ray_cast_bvh_gpu::GpuSceneRayCast;
pub use sphere_cast_bvh::{
    sphere_cast, sphere_cast_all, sphere_cast_all_bvh, sphere_cast_bvh, SceneSphereCast,
    SphereCastHit,
};
pub use sphere_cast_bvh_gpu::GpuSceneSphereCast;
pub use capsule_cast_bvh::{
    capsule_cast, capsule_cast_all, capsule_cast_all_bvh, capsule_cast_bvh, CapsuleCastHit,
    SceneCapsuleCast,
};
pub use capsule_cast_bvh_gpu::GpuSceneCapsuleCast;
pub use box_cast_bvh::{
    box_cast, box_cast_all, box_cast_all_bvh, box_cast_bvh, BoxCastHit, SceneBoxCast,
};
pub use box_cast_bvh_gpu::GpuSceneBoxCast;
pub use convex_cast_bvh::{
    convex_cast, convex_cast_all, convex_cast_all_bvh, convex_cast_bvh, ConvexCastHit,
    SceneConvexCast,
};
pub use convex_cast_bvh_gpu::GpuSceneConvexCast;
pub use shape_cast::{cast_shape, cast_shape_all, RoundedConvex, ShapeCastHit};
pub use shape_cast_bvh::{cast_shape_all_bvh, cast_shape_bvh};
pub use shape_cast_bvh_gpu::GpuBvhShapeCast;
pub use convex_convex_manifold_gpu::GpuConvexConvexManifoldNarrowphase;
pub use convex_hull::{ConvexEdge, ConvexFace, ConvexHull};
pub use convex_pose::ConvexPose;
pub use cpu::cpu_narrowphase;
pub use epa::{epa, Penetration};
pub use gjk::{gjk, GjkStatus};
pub use gpu::GpuNarrowphase;
pub use halfspace::{cpu_halfspace_narrowphase, Plane, SpherePlanePair};
pub use halfspace_gpu::GpuHalfspaceNarrowphase;
pub use heightfield::{cpu_sphere_heightfield_narrowphase, CellRange, Heightfield, HeightfieldSpherePair, XzAabb};
pub use heightfield_capsule_manifold::{cpu_capsule_heightfield_manifold, HeightfieldCapsulePair};
pub use heightfield_capsule_manifold_gpu::GpuCapsuleHeightfieldManifoldNarrowphase;
pub use heightfield_gpu::GpuSphereHeightfieldNarrowphase;
pub use heightfield_obb_manifold::{cpu_obb_heightfield_manifold, HeightfieldObbPair};
pub use heightfield_obb_manifold_gpu::GpuObbHeightfieldManifoldNarrowphase;
pub use manifold::{ContactManifold, ManifoldPoint, MAX_MANIFOLD_POINTS};
pub use minkowski::{support, SupportPoint};
pub(crate) use manifold::reduce_to_four;
pub use obb::{cpu_obb_narrowphase, Obb, SphereObbPair};
pub use obb_gpu::GpuObbNarrowphase;
pub use obb_halfspace::{cpu_obb_halfspace_narrowphase, ObbPlanePair};
pub use obb_halfspace_gpu::GpuObbHalfspaceNarrowphase;
pub use obb_halfspace_manifold::cpu_obb_halfspace_manifold;
pub use obb_halfspace_manifold_gpu::GpuObbHalfspaceManifoldNarrowphase;
pub use obb_obb::{cpu_obb_obb_narrowphase, ObbObbPair};
pub use obb_obb_gpu::GpuObbObbNarrowphase;
pub use obb_obb_manifold::cpu_obb_obb_manifold;
pub use obb_obb_manifold_gpu::GpuObbObbManifoldNarrowphase;
pub use obb_triangle::{cpu_obb_triangle_narrowphase, ObbTrianglePair};
pub use obb_triangle_gpu::GpuObbTriangleNarrowphase;
pub use obb_triangle_manifold::cpu_obb_triangle_manifold;
pub use obb_triangle_manifold_gpu::GpuObbTriangleManifoldNarrowphase;
pub use sphere_triangle::{cpu_sphere_triangle_narrowphase, SphereTrianglePair, Triangle};
pub use sphere_triangle_gpu::GpuSphereTriangleNarrowphase;
