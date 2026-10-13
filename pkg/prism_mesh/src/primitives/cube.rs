use alloc::vec::Vec;

use super::common::{build_mesh, positive};
use super::PrimitiveError;

/// Builds a hard-edged centered box with one normal/UV chart per face.
#[must_use = "the generated mesh must be retained"]
pub fn cube(size: [f32; 3]) -> Result<crate::Mesh, PrimitiveError> {
    let half = [
        positive("width", size[0])? * 0.5,
        positive("height", size[1])? * 0.5,
        positive("depth", size[2])? * 0.5,
    ];
    let faces: [([[f32; 3]; 4], [f32; 3]); 6] = [
        (
            [
                [-1.0, -1.0, 1.0],
                [1.0, -1.0, 1.0],
                [1.0, 1.0, 1.0],
                [-1.0, 1.0, 1.0],
            ],
            [0.0, 0.0, 1.0],
        ),
        (
            [
                [1.0, -1.0, -1.0],
                [-1.0, -1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [1.0, 1.0, -1.0],
            ],
            [0.0, 0.0, -1.0],
        ),
        (
            [
                [-1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, -1.0],
                [-1.0, 1.0, -1.0],
            ],
            [0.0, 1.0, 0.0],
        ),
        (
            [
                [-1.0, -1.0, -1.0],
                [1.0, -1.0, -1.0],
                [1.0, -1.0, 1.0],
                [-1.0, -1.0, 1.0],
            ],
            [0.0, -1.0, 0.0],
        ),
        (
            [
                [1.0, -1.0, 1.0],
                [1.0, -1.0, -1.0],
                [1.0, 1.0, -1.0],
                [1.0, 1.0, 1.0],
            ],
            [1.0, 0.0, 0.0],
        ),
        (
            [
                [-1.0, -1.0, -1.0],
                [-1.0, -1.0, 1.0],
                [-1.0, 1.0, 1.0],
                [-1.0, 1.0, -1.0],
            ],
            [-1.0, 0.0, 0.0],
        ),
    ];
    let mut positions = Vec::with_capacity(24);
    let mut normals = Vec::with_capacity(24);
    let mut uvs = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);
    for (face_index, (corners, normal)) in faces.into_iter().enumerate() {
        let base = (face_index * 4) as u32;
        for (corner, unit) in corners.into_iter().enumerate() {
            positions.push([unit[0] * half[0], unit[1] * half[1], unit[2] * half[2]]);
            normals.push(normal);
            uvs.push([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]][corner]);
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    build_mesh(positions, normals, uvs, indices)
}
