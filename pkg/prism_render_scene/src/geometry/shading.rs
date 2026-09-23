//! Render-world rows for the compute-friendly surface reconstruction table.

use bevy_mesh::{Mesh, PrimitiveTopology, VertexAttributeValues};
use bevy_render::{
    impl_atomic_pod,
    render_resource::{AtomicPod, ShaderType},
};
use bytemuck::{Pod, Zeroable};
use prism_render_shading::{GpuShadingPrimitive, GpuShadingVertex};

/// Geometry-table header addressed by a stable geometry index.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Pod, ShaderType, Zeroable)]
pub struct RenderShadingGeometryHeader {
    pub generation: u32,
    pub revision: u32,
    pub vertex_offset: u32,
    pub vertex_count: u32,
    pub primitive_offset: u32,
    pub primitive_count: u32,
    pub flags: u32,
    pub _padding: u32,
}
impl_atomic_pod!(RenderShadingGeometryHeader, RenderShadingGeometryHeaderBlob);

/// Stable vertex row used by compute resolve and offline export.
pub type RenderShadingVertex = GpuShadingVertex;

/// Stable triangle row used by compute resolve and offline export.
pub type RenderShadingPrimitive = GpuShadingPrimitive;

/// CPU-side payload that is uploaded to the render-world shading table.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RenderShadingGeometry {
    pub vertices: Vec<RenderShadingVertex>,
    pub primitives: Vec<RenderShadingPrimitive>,
    pub flags: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShadingGeometryBuildError {
    MissingPositions,
    UnsupportedPositions,
    MismatchedAttributeLength,
    UnsupportedTopology,
    InvalidIndex,
}

/// Builds the stable surface table while the source mesh is still accessible
/// in the main world. Meshes extracted with `RENDER_WORLD` only cannot provide
/// this payload; callers must retain `MAIN_WORLD` usage for compute shading.
pub fn build_shading_geometry(
    mesh: &Mesh,
) -> Result<RenderShadingGeometry, ShadingGeometryBuildError> {
    let positions = match mesh
        .try_attribute_option(Mesh::ATTRIBUTE_POSITION)
        .map_err(|_| ShadingGeometryBuildError::MissingPositions)?
    {
        Some(VertexAttributeValues::Float32x3(values)) => values,
        Some(_) => return Err(ShadingGeometryBuildError::UnsupportedPositions),
        None => return Err(ShadingGeometryBuildError::MissingPositions),
    };
    let normals = optional_float3(mesh, Mesh::ATTRIBUTE_NORMAL)?;
    let uvs = optional_float2(mesh, Mesh::ATTRIBUTE_UV_0)?;
    if normals.is_some_and(|values| values.len() != positions.len())
        || uvs.is_some_and(|values| values.len() != positions.len())
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
    let vertices = positions
        .iter()
        .enumerate()
        .map(|(index, position)| RenderShadingVertex {
            position: *position,
            _position_padding: 0.0,
            normal: normals.map_or([0.0; 3], |values| values[index]),
            _normal_padding: 0.0,
            uv: uvs.map_or([0.0; 2], |values| values[index]),
            flags,
            _padding: 0,
        })
        .collect();
    let indices: Vec<usize> = mesh
        .try_indices_option()
        .map_err(|_| ShadingGeometryBuildError::InvalidIndex)?
        .map_or_else(|| (0..positions.len()).collect(), |indices| indices.iter().collect());
    let primitives = triangle_indices(mesh.primitive_topology(), &indices)?
        .into_iter()
        .map(|[a, b, c]| {
            if [a, b, c].into_iter().any(|index| index >= positions.len()) {
                return Err(ShadingGeometryBuildError::InvalidIndex);
            }
            Ok(RenderShadingPrimitive {
                indices: [a as u32, b as u32, c as u32],
                flags,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(RenderShadingGeometry { vertices, primitives, flags })
}

fn optional_float3<'a>(
    mesh: &'a Mesh,
    attribute: bevy_mesh::MeshVertexAttribute,
) -> Result<Option<&'a Vec<[f32; 3]>>, ShadingGeometryBuildError> {
    match mesh
        .try_attribute_option(attribute)
        .map_err(|_| ShadingGeometryBuildError::MismatchedAttributeLength)?
    {
        None => Ok(None),
        Some(VertexAttributeValues::Float32x3(values)) => Ok(Some(values)),
        Some(_) => Err(ShadingGeometryBuildError::MismatchedAttributeLength),
    }
}

fn optional_float2<'a>(
    mesh: &'a Mesh,
    attribute: bevy_mesh::MeshVertexAttribute,
) -> Result<Option<&'a Vec<[f32; 2]>>, ShadingGeometryBuildError> {
    match mesh
        .try_attribute_option(attribute)
        .map_err(|_| ShadingGeometryBuildError::MismatchedAttributeLength)?
    {
        None => Ok(None),
        Some(VertexAttributeValues::Float32x2(values)) => Ok(Some(values)),
        Some(_) => Err(ShadingGeometryBuildError::MismatchedAttributeLength),
    }
}

fn triangle_indices(
    topology: PrimitiveTopology,
    indices: &[usize],
) -> Result<Vec<[usize; 3]>, ShadingGeometryBuildError> {
    match topology {
        PrimitiveTopology::TriangleList => Ok(indices
            .chunks_exact(3)
            .map(|chunk| [chunk[0], chunk[1], chunk[2]])
            .collect()),
        PrimitiveTopology::TriangleStrip => Ok(indices
            .windows(3)
            .enumerate()
            .map(|(index, window)| {
                if index % 2 == 0 {
                    [window[0], window[1], window[2]]
                } else {
                    [window[1], window[0], window[2]]
                }
            })
            .collect()),
        _ => Err(ShadingGeometryBuildError::UnsupportedTopology),
    }
}

pub const SHADING_GEOMETRY_FLAG_ACTIVE: u32 = 1 << 0;
pub const SHADING_GEOMETRY_FLAG_MISSING_NORMAL: u32 = 1 << 1;
pub const SHADING_GEOMETRY_FLAG_MISSING_UV: u32 = 1 << 2;
pub const SHADING_GEOMETRY_FLAG_INVALID: u32 = 1 << 3;

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_asset::RenderAssetUsages;
    use bevy_mesh::Indices;

    #[test]
    fn rows_match_compute_surface_abi() {
        assert_eq!(size_of::<RenderShadingGeometryHeader>(), 32);
        assert_eq!(size_of::<RenderShadingVertex>(), 48);
        assert_eq!(size_of::<RenderShadingPrimitive>(), 16);
    }

    #[test]
    fn builds_indexed_mesh_and_marks_missing_attributes() {
        let mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::MAIN_WORLD,
        )
            .with_inserted_attribute(
                Mesh::ATTRIBUTE_POSITION,
                vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            )
            .with_inserted_indices(Indices::U16(vec![0, 1, 2]));
        let table = build_shading_geometry(&mesh).unwrap();
        assert_eq!(table.vertices.len(), 3);
        assert_eq!(
            table.primitives,
            vec![RenderShadingPrimitive {
                indices: [0, 1, 2],
                flags: SHADING_GEOMETRY_FLAG_MISSING_NORMAL
                    | SHADING_GEOMETRY_FLAG_MISSING_UV,
            }]
        );
    }
}
