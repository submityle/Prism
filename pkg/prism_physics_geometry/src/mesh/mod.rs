//! Mesh colliders that combine the broad-phase BVH with narrow-phase
//! primitive tests to answer exact geometric queries against triangle soups.

mod triangle_mesh;

pub use triangle_mesh::{
    MeshCapsuleContact, MeshCapsuleSweepHit, MeshClosestPoint, MeshRayHit, MeshSphereContact,
    MeshSweepHit, TriangleMesh,
};
