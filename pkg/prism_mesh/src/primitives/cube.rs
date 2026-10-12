//! The cube primitive.

use super::finish;
use crate::mesh::Mesh;
use crate::vertex::Vertex;

/// The six cube faces, each as `(normal, tangent_u, tangent_v)`. Every face
/// satisfies `tangent_u × tangent_v == normal`, which makes the corner order
/// below wind counter-clockwise when the face is viewed from outside the cube.
const FACES: [([f32; 3], [f32; 3], [f32; 3]); 6] = [
    ([1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]),
    ([-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
    ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]),
    ([0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
    ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
    ([0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
];

/// The four corners of a face, as `(-u, -v)`, `(+u, -v)`, `(+u, +v)`,
/// `(-u, +v)` sign pairs.
const CORNER_SIGNS: [(f32, f32); 4] = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];

/// The UV for each corner, in the same order as [`CORNER_SIGNS`].
const CORNER_UVS: [[f32; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];

/// Builds a cube with edge length `size`, centred on the origin.
///
/// The cube has 24 vertices — four per face, so each face carries its own flat
/// normal — and 36 indices. Face normals point outward and every face winds
/// counter-clockwise when viewed from outside.
#[must_use]
pub fn cube(size: f32) -> Mesh {
    let half = size * 0.5;
    let mut vertices = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);
    for (normal, u, v) in FACES {
        let base = u32::try_from(vertices.len()).expect("a cube has 24 vertices");
        for (corner, (u_sign, v_sign)) in CORNER_SIGNS.into_iter().enumerate() {
            let position = [
                (normal[0] + u[0] * u_sign + v[0] * v_sign) * half,
                (normal[1] + u[1] * u_sign + v[1] * v_sign) * half,
                (normal[2] + u[2] * u_sign + v[2] * v_sign) * half,
            ];
            vertices.push(Vertex::new(position, normal, CORNER_UVS[corner]));
        }
        // The quad (0, 1, 2) + (0, 2, 3) winds counter-clockwise from outside.
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    finish(&vertices, indices)
}
