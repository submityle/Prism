use alloc::vec::Vec;

use super::common::{build_mesh, positive, subdivisions};
use super::PrimitiveError;

/// Builds a sampled XZ terrain patch with finite-difference normals.
///
/// `heights` is row-major in Z, with `(segments[0] + 1) * (segments[1] + 1)`
/// samples. The patch is centered at the origin and uses [0, 1] UVs.
#[must_use = "the generated mesh must be retained"]
pub fn terrain_patch(
    size: [f32; 2],
    segments: [u32; 2],
    heights: &[f32],
) -> Result<crate::Mesh, PrimitiveError> {
    let width = positive("width", size[0])?;
    let depth = positive("depth", size[1])?;
    let x_segments = subdivisions("x", segments[0], 1)?;
    let z_segments = subdivisions("z", segments[1], 1)?;
    let expected = ((x_segments + 1) * (z_segments + 1)) as usize;
    if heights.len() != expected {
        return Err(PrimitiveError::HeightCountMismatch {
            expected,
            found: heights.len(),
        });
    }
    if heights.iter().any(|height| !height.is_finite()) {
        return Err(PrimitiveError::NonFiniteHeight);
    }
    let x_step = width / x_segments as f32;
    let z_step = depth / z_segments as f32;
    let row = x_segments + 1;
    let sample = |x: u32, z: u32| heights[(z * row + x) as usize];
    let mut positions = Vec::with_capacity(expected);
    let mut normals = Vec::with_capacity(expected);
    let mut uvs = Vec::with_capacity(expected);
    for z in 0..=z_segments {
        let v = z as f32 / z_segments as f32;
        for x in 0..=x_segments {
            let u = x as f32 / x_segments as f32;
            let left = sample(x.saturating_sub(1), z);
            let right = sample((x + 1).min(x_segments), z);
            let back = sample(x, z.saturating_sub(1));
            let front = sample(x, (z + 1).min(z_segments));
            let dx = (right - left)
                / if x == 0 || x == x_segments {
                    x_step
                } else {
                    2.0 * x_step
                };
            let dz = (front - back)
                / if z == 0 || z == z_segments {
                    z_step
                } else {
                    2.0 * z_step
                };
            let mut normal = [-dx, 2.0, -dz];
            let length =
                (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
            if length > f32::EPSILON {
                normal.iter_mut().for_each(|value| *value /= length);
            }
            positions.push([(u - 0.5) * width, sample(x, z), (v - 0.5) * depth]);
            normals.push(normal);
            uvs.push([u, v]);
        }
    }
    let mut indices = Vec::with_capacity((x_segments * z_segments * 6) as usize);
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
