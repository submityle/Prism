//! Geometry primitives and broad-phase acceleration structures for Prism's
//! physics engine.
//!
//! The M0 surface provides bounding volumes ([`Aabb`], [`BoundingSphere`],
//! [`Ray`], [`Plane`], [`Frustum`]), a dynamic bounding-volume hierarchy ([`DynamicBvh`], a dynamic
//! axis-aligned bounding box tree) used as the broad-phase, and persistent
//! candidate-pair generation. Proxies are identified by an opaque [`ProxyId`],
//! keeping this crate independent of the physics core.
//!
//! # Algorithms
//!
//! The tree follows publicly documented dynamic bounding-volume hierarchy
//! techniques: fat axis-aligned bounding boxes, incremental insert / remove /
//! refit, surface-area-heuristic guided sibling selection, and height-balanced
//! tree rotations. It is engine-agnostic and contains **no Unreal Engine
//! source or derived code**.
#![forbid(unsafe_code)]

extern crate alloc;

pub mod bounding;
pub mod broadphase;
pub mod bvh;
pub mod narrow;
pub mod mesh;
pub mod proxy;

pub use bounding::{Aabb, BoundingSphere, Capsule, Frustum, Obb, Plane, Ray};
pub use broadphase::{generate_pairs, BroadPhasePair, PairChanges, PersistentBroadPhase};
pub use bvh::DynamicBvh;
pub use mesh::{
    MeshCapsuleContact, MeshCapsuleSweepHit, MeshClosestPoint, MeshRayHit, MeshSphereContact,
    MeshSweepHit, TriangleMesh,
};
pub use narrow::{
    closest_point_on_aabb, closest_point_on_segment, closest_point_on_triangle,
    closest_point_segment_triangle, closest_points_segment_segment, capsule_box_manifold,
    conservative_advancement, contact_manifold, gjk_closest_points, gjk_contact, gjk_intersect, ray_capsule, ray_obb,
    ray_sphere, ray_triangle, rotational_conservative_advancement, segment_triangle_intersection,
    speculative_contact, sweep_capsule_triangle, sweep_sphere_triangle, triangle_aabb_overlap,
    CapsuleSweepHit, ClipShape, ClosestPoints, Contact, ContactManifold, FacePolygon, Inflated,
    ManifoldPoint, RayTriangleHit, SegmentClosest, SegmentTriangleClosest, SegmentTriangleHit,
    SpeculativeContact, SphereSweepHit, SupportMap, TimeOfImpact, Transformed, Translated,
};
pub use proxy::ProxyId;
