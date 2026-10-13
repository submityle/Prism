use alloc::vec::Vec;

use super::common::{build_mesh, positive, subdivisions};
use super::PrimitiveError;

/// Builds a centered XZ plane with +Y normals and tiled [0, 1] UVs.
#[must_use = "the generated mesh must be retained"]
pub fn plane(size: [f32; 2], segments: [u32; 2]) -> Result<crate::Mesh, PrimitiveError> {
    let width = positive("width", size[0])?;
    let depth = positive("depth", size[1])?;
    let x_segments = subdivisions("x", segments[0], 1)?;
    let z_segments = subdivisions("z", segments[1], 1)?;
    let mut positions = Vec::with_capacity(((x_segments + 1) * (z_segments + 1)) as usize);
    let mut normals = Vec::with_capacity(positions.capacity());
    let mut uvs = Vec::with_capacity(positions.capacity());
    for z in 0..=z_segments {
        let v = z as f32 / z_segments as f32;
        for x in 0..=x_segments {
            let u = x as f32 / x_segments as f32;
            positions.push([(u - 0.5) * width, 0.0, (v - 0.5) * depth]);
            normals.push([0.0, 1.0, 0.0]);
            uvs.push([u, v]);
        }
    }
    let mut indices = Vec::with_capacity((x_segments * z_segments * 6) as usize);
    let row = x_segments + 1;
    for z in 0..z_segments {
        for x in 0..x_segments {
            let a = z * row + x;
            let b = a + 1;
            let c = a + row;
            let d = c + 1;
            indices.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }
    build_mesh(positions, normals, uvs, indices)
}
