//! The plane primitive.

use super::finish;
use crate::mesh::Mesh;
use crate::vertex::Vertex;

/// Builds a `size × size` plane centred on the origin, lying in the XZ plane
/// and facing `+Y`.
///
/// `size` is the edge length, so the four corners sit at `±size/2` on X and Z.
/// UVs run from `(0, 0)` at `(-X, -Z)` to `(1, 1)` at `(+X, +Z)`, and the two
/// triangles wind counter-clockwise when viewed from `+Y`.
#[must_use]
pub fn plane(size: f32) -> Mesh {
    let half = size * 0.5;
    let up = [0.0, 1.0, 0.0];
    let vertices = [
        Vertex::new([-half, 0.0, -half], up, [0.0, 0.0]),
        Vertex::new([half, 0.0, -half], up, [1.0, 0.0]),
        Vertex::new([half, 0.0, half], up, [1.0, 1.0]),
        Vertex::new([-half, 0.0, half], up, [0.0, 1.0]),
    ];
    // (0, 2, 1) and (0, 3, 2) both wind counter-clockwise seen from +Y.
    let indices = vec![0, 2, 1, 0, 3, 2];
    finish(&vertices, indices)
}
