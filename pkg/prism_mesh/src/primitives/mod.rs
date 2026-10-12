//! Deterministic primitive mesh generators.
//!
//! Every generator produces a [`Mesh`] with the
//! [`VertexLayout::POSITION_NORMAL_UV`] layout, correct outward-facing normals,
//! UVs within `[0, 1]`, and counter-clockwise (front-facing) winding. The
//! generators are pure functions of their parameters, so the same inputs always
//! produce byte-identical meshes.

mod cube;
mod cylinder;
mod plane;
mod sphere;

pub use cube::cube;
pub use cylinder::cylinder;
pub use plane::plane;
pub use sphere::uv_sphere;

use crate::mesh::Mesh;
use crate::vertex::{Vertex, VertexLayout};

/// The layout every primitive generator produces.
pub(crate) const PRIMITIVE_LAYOUT: VertexLayout = VertexLayout::POSITION_NORMAL_UV;

/// Assembles a primitive from generated vertices and indices.
pub(crate) fn finish(vertices: &[Vertex], indices: Vec<u32>) -> Mesh {
    Mesh::from_vertices(vertices, indices, PRIMITIVE_LAYOUT)
}
