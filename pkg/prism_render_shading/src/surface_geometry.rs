//! Backend-neutral construction of compute-friendly surface geometry tables.
//!
//! This module consumes Prism's native mesh contract and deliberately has no
//! ECS, renderer, or backend dependency. The scene adapter is responsible for
//! converting legacy asset representations before calling this builder.

use prism_mesh::{Indices, Mesh, VertexAttributeValues, POSITION_ATTRIBUTE};
use prism_render_driver::PrimitiveTopology;

use crate::{GpuShadingPrimitive, GpuShadingVertex};

/// CPU-side payload that is uploaded to the render-world shading table.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ShadingGeometry {
    pub vertices: Vec<GpuShadingVertex>,
    pub primitives: Vec<GpuShadingPrimitive>,
    pub flags: u32,
}

/// Errors produced while converting a native mesh into the surface table.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShadingGeometryBuildError {
    MissingPositions,
    UnsupportedPositions,
    MismatchedAttributeLength,
    UnsupportedTopology,
    InvalidPrimitiveCount,
    InvalidIndex,
}

/// Builds a stable surface table from a native Prism mesh.
pub fn build_shading_geometry(
    mesh: &Mesh,
) -> Result<ShadingGeometry, ShadingGeometryBuildError> {
    let positions = match attribute_values(mesh, POSITION_ATTRIBUTE) {
        Some(VertexAttributeValues::Float32x3(values)) => values,
        Some(_) => return Err(ShadingGeometryBuildError::UnsupportedPositions),
        None => return Err(ShadingGeometryBuildError::MissingPositions),
    };
    let normals = optional_float3(mesh, "Vertex_Normal")?;
    let uvs = optional_float2(mesh, "Vertex_Uv0")?;
    let tangents = optional_float4(mesh, "Vertex_Tangent")?;
    if normals.is_some_and(|values| values.len() != positions.len())
        || uvs.is_some_and(|values| values.len() != positions.len())
        || tangents.is_some_and(|values| values.len() != positions.len())
    {
        return Err(ShadingGeometryBuildError::MismatchedAttributeLength);
    }
    let mut flags = 0;
    if normals.is_none() {
        flags |= SHADING_GEOMETRY_FLAG_MISSING_NORMAL;
    }
    if uvs.is_none() {
        flags |= SHADING_GEOMETRY_FLAG_MISSING_UV;
    }
    if tangents.is_none() {
        flags |= SHADING_GEOMETRY_FLAG_MISSING_TANGENT;
    }
    let vertices = positions
        .iter()
        .enumerate()
        .map(|(index, position)| GpuShadingVertex {
            position: *position,
            _position_padding: 0.0,
            normal: normals.map_or([0.0; 3], |values| values[index]),
            _normal_padding: 0.0,
            tangent: tangents.map_or([0.0, 0.0, 0.0, 1.0], |values| values[index]),
            uv: uvs.map_or([0.0; 2], |values| values[index]),
            flags,
            _padding: 0,
        })
        .collect();
    let source_indices = mesh.indices();
    let index_count = source_indices.map_or(positions.len(), Indices::len);
    let primitives = triangle_indices(
        mesh.primitive_topology(),
        source_indices,
        index_count,
        positions.len(),
        flags,
    )?;
    Ok(ShadingGeometry {
        vertices,
        primitives,
        flags,
    })
}

fn optional_float3<'a>(
    mesh: &'a Mesh,
    attribute: &'static str,
) -> Result<Option<&'a Vec<[f32; 3]>>, ShadingGeometryBuildError> {
    match attribute_values(mesh, attribute) {
        None => Ok(None),
        Some(VertexAttributeValues::Float32x3(values)) => Ok(Some(values)),
        Some(_) => Err(ShadingGeometryBuildError::MismatchedAttributeLength),
    }
}

fn optional_float2<'a>(
    mesh: &'a Mesh,
    attribute: &'static str,
) -> Result<Option<&'a Vec<[f32; 2]>>, ShadingGeometryBuildError> {
    match attribute_values(mesh, attribute) {
        None => Ok(None),
        Some(VertexAttributeValues::Float32x2(values)) => Ok(Some(values)),
        Some(_) => Err(ShadingGeometryBuildError::MismatchedAttributeLength),
    }
}

fn optional_float4<'a>(
    mesh: &'a Mesh,
    attribute: &'static str,
) -> Result<Option<&'a Vec<[f32; 4]>>, ShadingGeometryBuildError> {
    match attribute_values(mesh, attribute) {
        None => Ok(None),
        Some(VertexAttributeValues::Float32x4(values)) => Ok(Some(values)),
        Some(_) => Err(ShadingGeometryBuildError::MismatchedAttributeLength),
    }
}

fn attribute_values<'a>(mesh: &'a Mesh, name: &'static str) -> Option<&'a VertexAttributeValues> {
    mesh.attributes()
        .find(|(attribute, _)| attribute.name == name)
        .map(|(_, values)| values)
}

fn triangle_indices(
    topology: PrimitiveTopology,
    indices: Option<&Indices>,
    index_count: usize,
    vertex_count: usize,
    flags: u32,
) -> Result<Vec<GpuShadingPrimitive>, ShadingGeometryBuildError> {
    match topology {
        PrimitiveTopology::TriangleList => {
            if !index_count.is_multiple_of(3) {
                return Err(ShadingGeometryBuildError::InvalidPrimitiveCount);
            }
            let mut primitives = Vec::with_capacity(index_count / 3);
            for base in (0..index_count).step_by(3) {
                primitives.push(read_primitive(
                    indices, [base, base + 1, base + 2], vertex_count, flags,
                )?);
            }
            Ok(primitives)
        }
        PrimitiveTopology::TriangleStrip => {
            let mut primitives = Vec::with_capacity(index_count.saturating_sub(2));
            for base in 0..index_count.saturating_sub(2) {
                let order = if base % 2 == 0 {
                    [base, base + 1, base + 2]
                } else {
                    [base + 1, base, base + 2]
                };
                primitives.push(read_primitive(indices, order, vertex_count, flags)?);
            }
            Ok(primitives)
        }
        _ => Err(ShadingGeometryBuildError::UnsupportedTopology),
    }
}

fn read_primitive(
    indices: Option<&Indices>,
    slots: [usize; 3],
    vertex_count: usize,
    flags: u32,
) -> Result<GpuShadingPrimitive, ShadingGeometryBuildError> {
    let mut primitive = [0; 3];
    for (slot, output) in slots.into_iter().zip(&mut primitive) {
        let index = indices
            .map(|indices| match indices {
                Indices::Uint16(values) => u32::from(values[slot]),
                Indices::Uint32(values) => values[slot],
            })
            .unwrap_or_else(|| slot as u32);
        let index = usize::try_from(index).map_err(|_| ShadingGeometryBuildError::InvalidIndex)?;
        if index >= vertex_count {
            return Err(ShadingGeometryBuildError::InvalidIndex);
        }
        *output = index;
    }
    Ok(GpuShadingPrimitive {
        indices: [
            u32::try_from(primitive[0])
                .map_err(|_| ShadingGeometryBuildError::InvalidIndex)?,
            u32::try_from(primitive[1])
                .map_err(|_| ShadingGeometryBuildError::InvalidIndex)?,
            u32::try_from(primitive[2])
                .map_err(|_| ShadingGeometryBuildError::InvalidIndex)?,
        ],
        flags,
    })
}

pub const SHADING_GEOMETRY_FLAG_ACTIVE: u32 = 1 << 0;
pub const SHADING_GEOMETRY_FLAG_MISSING_NORMAL: u32 = 1 << 1;
pub const SHADING_GEOMETRY_FLAG_MISSING_UV: u32 = 1 << 2;
pub const SHADING_GEOMETRY_FLAG_INVALID: u32 = 1 << 3;
pub const SHADING_GEOMETRY_FLAG_MISSING_TANGENT: u32 = 1 << 4;

#[cfg(test)]
mod tests {
    use super::*;
    use prism_mesh::{Indices, MeshVertexAttribute};
    use prism_render_driver::VertexFormat;

    #[test]
    fn builds_indexed_mesh_and_marks_missing_attributes() {
        let mesh = Mesh::new(PrimitiveTopology::TriangleList)
            .with_attribute(
                MeshVertexAttribute::new(POSITION_ATTRIBUTE, VertexFormat::Float32x3, 0),
                VertexAttributeValues::Float32x3(vec![
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ]),
            )
            .unwrap()
            .indexed(Indices::Uint16(vec![0, 1, 2]));
        let table = build_shading_geometry(&mesh).unwrap();
        assert_eq!(table.vertices.len(), 3);
        assert_eq!(
            table.primitives,
            vec![GpuShadingPrimitive {
                indices: [0, 1, 2],
                flags: SHADING_GEOMETRY_FLAG_MISSING_NORMAL
                    | SHADING_GEOMETRY_FLAG_MISSING_UV
                    | SHADING_GEOMETRY_FLAG_MISSING_TANGENT,
            }]
        );
    }

    #[test]
    fn rejects_indices_outside_the_vertex_table() {
        let mesh = Mesh::new(PrimitiveTopology::TriangleList)
            .with_attribute(
                MeshVertexAttribute::new(POSITION_ATTRIBUTE, VertexFormat::Float32x3, 0),
                VertexAttributeValues::Float32x3(vec![
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ]),
            )
            .unwrap()
            .indexed(Indices::Uint32(vec![0, 1, 3]));
        assert_eq!(
            build_shading_geometry(&mesh),
            Err(ShadingGeometryBuildError::InvalidIndex)
        );
    }

    #[test]
    fn expands_non_indexed_triangle_strip_with_alternating_winding() {
        let mesh = Mesh::new(PrimitiveTopology::TriangleStrip)
            .with_attribute(
                MeshVertexAttribute::new(POSITION_ATTRIBUTE, VertexFormat::Float32x3, 0),
                VertexAttributeValues::Float32x3(vec![
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0],
                ]),
            )
            .unwrap();
        let table = build_shading_geometry(&mesh).unwrap();
        assert_eq!(
            table.primitives.iter().map(|primitive| primitive.indices).collect::<Vec<_>>(),
            vec![[0, 1, 2], [2, 1, 3]]
        );
    }

    #[test]
    fn rejects_triangle_list_remainder_instead_of_dropping_indices() {
        let mesh = Mesh::new(PrimitiveTopology::TriangleList)
            .with_attribute(
                MeshVertexAttribute::new(POSITION_ATTRIBUTE, VertexFormat::Float32x3, 0),
                VertexAttributeValues::Float32x3(vec![
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0],
                ]),
            )
            .unwrap()
            .indexed(Indices::Uint16(vec![0, 1, 2, 3]));
        assert_eq!(
            build_shading_geometry(&mesh),
            Err(ShadingGeometryBuildError::InvalidPrimitiveCount)
        );
    }

    #[test]
    fn rejects_non_triangle_topology() {
        let mesh = Mesh::new(PrimitiveTopology::LineList)
            .with_attribute(
                MeshVertexAttribute::new(POSITION_ATTRIBUTE, VertexFormat::Float32x3, 0),
                VertexAttributeValues::Float32x3(vec![
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                ]),
            )
            .unwrap();
        assert_eq!(
            build_shading_geometry(&mesh),
            Err(ShadingGeometryBuildError::UnsupportedTopology)
        );
    }

    #[test]
    fn rejects_mesh_without_positions() {
        let mesh = Mesh::new(PrimitiveTopology::TriangleList);
        assert_eq!(
            build_shading_geometry(&mesh),
            Err(ShadingGeometryBuildError::MissingPositions)
        );
    }
}
