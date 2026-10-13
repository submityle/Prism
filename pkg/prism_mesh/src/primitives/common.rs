use alloc::vec::Vec;

use prism_render_driver::PrimitiveTopology;

use crate::{Indices, Mesh, VertexAttributeValues};

use super::PrimitiveError;
use super::{NORMAL, POSITION, UV0};

pub(crate) fn positive(name: &'static str, value: f32) -> Result<f32, PrimitiveError> {
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(PrimitiveError::InvalidDimension { name, value })
    }
}

pub(crate) fn subdivisions(
    name: &'static str,
    value: u32,
    minimum: u32,
) -> Result<u32, PrimitiveError> {
    if value >= minimum {
        Ok(value)
    } else {
        Err(PrimitiveError::InvalidSubdivision {
            name,
            value,
            minimum,
        })
    }
}

pub(crate) fn build_mesh(
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    indices: Vec<u32>,
) -> Result<Mesh, PrimitiveError> {
    let mesh = Mesh::new(PrimitiveTopology::TriangleList)
        .with_attribute(POSITION, VertexAttributeValues::Float32x3(positions))?
        .with_attribute(NORMAL, VertexAttributeValues::Float32x3(normals))?
        .with_attribute(UV0, VertexAttributeValues::Float32x2(uvs))?
        .indexed(Indices::Uint32(indices));
    mesh.validate()?;
    Ok(mesh)
}
