use alloc::vec::Vec;

use prism_render_driver::PrimitiveTopology;

use crate::{Indices, Mesh, VertexAttributeValues};

use super::PrimitiveError;

/// Bounds synchronous primitive construction to a predictable temporary
/// working set. Large production meshes belong in the streaming asset builder.
pub(crate) const MAX_GENERATED_BYTES: u64 = 256 * 1024 * 1024;
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

/// Validates generator cardinalities before any allocation or iteration.
/// Mesh indices and counts are represented by u32 in the public data ABI.
pub(crate) fn generated_counts(
    vertices: u64,
    indices: u64,
) -> Result<(usize, usize), PrimitiveError> {
    let max = u64::from(u32::MAX);
    if vertices > max {
        return Err(PrimitiveError::CountOverflow { kind: "vertex" });
    }
    if indices > max {
        return Err(PrimitiveError::CountOverflow { kind: "index" });
    }
    let bytes = vertices
        .checked_mul(32)
        .and_then(|value| {
            indices
                .checked_mul(4)
                .and_then(|index_bytes| value.checked_add(index_bytes))
        })
        .ok_or(PrimitiveError::CountOverflow { kind: "index" })?;
    if bytes > MAX_GENERATED_BYTES {
        return Err(PrimitiveError::WorkingSetExceeded {
            requested: bytes,
            maximum: MAX_GENERATED_BYTES,
        });
    }
    let vertices =
        usize::try_from(vertices).map_err(|_| PrimitiveError::CountOverflow { kind: "vertex" })?;
    let indices =
        usize::try_from(indices).map_err(|_| PrimitiveError::CountOverflow { kind: "index" })?;
    Ok((vertices, indices))
}

pub(crate) fn reserve<T>(kind: &'static str, elements: usize) -> Result<Vec<T>, PrimitiveError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(elements)
        .map_err(|_| PrimitiveError::AllocationFailed { kind, elements })?;
    Ok(values)
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
