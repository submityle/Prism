//! Geometry primitives and broad-phase acceleration structures for Prism's
//! physics engine.
//!
//! The M0 surface provides bounding volumes (`Aabb`, `BoundingSphere`, `Ray`),
//! a dynamic AABB tree (`DynamicBvh`) broad-phase, and persistent candidate
//! pair generation. Proxies are identified by an opaque [`ProxyId`], keeping
//! this crate independent of the physics core.
//!
//! It is engine-agnostic and contains no Unreal Engine source or derived code.
#![forbid(unsafe_code)]
