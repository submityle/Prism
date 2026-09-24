# Prism Physics Geometry

Geometry primitives and broad-phase acceleration structures for Prism's physics
engine. This crate is engine-agnostic and independent of `prism_physics_core`;
it identifies proxies by an opaque `ProxyId` so it can be reused standalone.

It deliberately contains **no Unreal Engine source or derived code**. The
dynamic BVH follows publicly documented dynamic AABB tree techniques
(fat AABBs, incremental insert/remove/refit, rotation rebalancing, SAH-guided
descent).

## M0 scope

- `bounding`: `Aabb`, `BoundingSphere`, and `Ray` with slab intersection.
- `bvh`: `DynamicBvh` dynamic AABB tree — insert / remove / update / refit,
  AABB overlap queries, and ray casts.
- `broadphase`: candidate-pair generation (self-overlap on the tree) with a
  persistent pair cache reporting started/ended pairs.
