//! Conversion at the Bevy integration boundary.
//!
//! The render-scene crate still receives `bevy_mesh::Mesh` from Bevy's ECS and
//! asset system. Geometry algorithms consume `prism_mesh::Mesh` instead, so
//! this module is the only place where the legacy source representation is
//! translated into Prism's device-independent mesh contract.

use bevy_mesh::{Indices, Mesh as BevyMesh, VertexAttributeValues};
use prism_mesh::{
    Mesh, MeshVertexAttribute, VertexAttributeValues as PrismVertexAttributeValues,
    POSITION_ATTRIBUTE,
};
use prism_render_driver::{PrimitiveTopology, VertexFormat};

use super::shading::ShadingGeometryBuildError;

const NORMAL_ATTRIBUTE: &str = "Vertex_Normal";
const UV_ATTRIBUTE: &str = "Vertex_Uv0";
const TANGENT_ATTRIBUTE: &str = "Vertex_Tangent";

/// Converts the shading-relevant portion of a Bevy mesh into the Prism mesh
/// contract. Unsupported optional attributes are reported exactly as the
/// former direct Bevy reader reported them.
pub(super) fn to_prism_mesh(mesh: &BevyMesh) -> Result<Mesh, ShadingGeometryBuildError> {
    let topology = match mesh.primitive_topology() {
        bevy_render::render_resource::PrimitiveTopology::PointList => PrimitiveTopology::PointList,
        bevy_render::render_resource::PrimitiveTopology::LineList => PrimitiveTopology::LineList,
        bevy_render::render_resource::PrimitiveTopology::LineStrip => PrimitiveTopology::LineStrip,
        bevy_render::render_resource::PrimitiveTopology::TriangleList => {
            PrimitiveTopology::TriangleList
        }
        bevy_render::render_resource::PrimitiveTopology::TriangleStrip => {
            PrimitiveTopology::TriangleStrip
        }
    };
    let mut converted = Mesh::new(topology);

    let positions = mesh
        .try_attribute_option(BevyMesh::ATTRIBUTE_POSITION)
        .map_err(|_| ShadingGeometryBuildError::MissingPositions)?
        .ok_or(ShadingGeometryBuildError::MissingPositions)?;
    let VertexAttributeValues::Float32x3(positions) = positions else {
        return Err(ShadingGeometryBuildError::UnsupportedPositions);
    };
    converted
        .insert_attribute(
            MeshVertexAttribute::new(POSITION_ATTRIBUTE, VertexFormat::Float32x3, 0),
            PrismVertexAttributeValues::Float32x3(positions.clone()),
        )
        .map_err(|_| ShadingGeometryBuildError::UnsupportedPositions)?;

    insert_optional_float3(
        mesh,
        BevyMesh::ATTRIBUTE_NORMAL,
        NORMAL_ATTRIBUTE,
        1,
        &mut converted,
    )?;
    insert_optional_float2(
        mesh,
        BevyMesh::ATTRIBUTE_UV_0,
        UV_ATTRIBUTE,
        2,
        &mut converted,
    )?;
    insert_optional_float4(
        mesh,
        BevyMesh::ATTRIBUTE_TANGENT,
        TANGENT_ATTRIBUTE,
        3,
        &mut converted,
    )?;

    if let Some(indices) = mesh
        .try_indices_option()
        .map_err(|_| ShadingGeometryBuildError::InvalidIndex)?
    {
        converted.with_indices(match indices {
            Indices::U16(values) => prism_mesh::Indices::Uint16(values.clone()),
            Indices::U32(values) => prism_mesh::Indices::Uint32(values.clone()),
        });
    }

    Ok(converted)
}

fn insert_optional_float3(
    source: &BevyMesh,
    source_attribute: bevy_mesh::MeshVertexAttribute,
    name: &'static str,
    shader_location: u32,
    target: &mut Mesh,
) -> Result<(), ShadingGeometryBuildError> {
    let Some(values) = source
        .try_attribute_option(source_attribute)
        .map_err(|_| ShadingGeometryBuildError::MismatchedAttributeLength)?
    else {
        return Ok(());
    };
    let VertexAttributeValues::Float32x3(values) = values else {
        return Err(ShadingGeometryBuildError::MismatchedAttributeLength);
    };
    target
        .insert_attribute(
            MeshVertexAttribute::new(name, VertexFormat::Float32x3, shader_location),
            PrismVertexAttributeValues::Float32x3(values.clone()),
        )
        .map_err(|_| ShadingGeometryBuildError::MismatchedAttributeLength)
}

fn insert_optional_float2(
    source: &BevyMesh,
    source_attribute: bevy_mesh::MeshVertexAttribute,
    name: &'static str,
    shader_location: u32,
    target: &mut Mesh,
) -> Result<(), ShadingGeometryBuildError> {
    let Some(values) = source
        .try_attribute_option(source_attribute)
        .map_err(|_| ShadingGeometryBuildError::MismatchedAttributeLength)?
    else {
        return Ok(());
    };
    let VertexAttributeValues::Float32x2(values) = values else {
        return Err(ShadingGeometryBuildError::MismatchedAttributeLength);
    };
    target
        .insert_attribute(
            MeshVertexAttribute::new(name, VertexFormat::Float32x2, shader_location),
            PrismVertexAttributeValues::Float32x2(values.clone()),
        )
        .map_err(|_| ShadingGeometryBuildError::MismatchedAttributeLength)
}

fn insert_optional_float4(
    source: &BevyMesh,
    source_attribute: bevy_mesh::MeshVertexAttribute,
    name: &'static str,
    shader_location: u32,
    target: &mut Mesh,
) -> Result<(), ShadingGeometryBuildError> {
    let Some(values) = source
        .try_attribute_option(source_attribute)
        .map_err(|_| ShadingGeometryBuildError::MismatchedAttributeLength)?
    else {
        return Ok(());
    };
    let VertexAttributeValues::Float32x4(values) = values else {
        return Err(ShadingGeometryBuildError::MismatchedAttributeLength);
    };
    target
        .insert_attribute(
            MeshVertexAttribute::new(name, VertexFormat::Float32x4, shader_location),
            PrismVertexAttributeValues::Float32x4(values.clone()),
        )
        .map_err(|_| ShadingGeometryBuildError::MismatchedAttributeLength)
}

#[cfg(test)]
mod tests {
    use bevy_asset::RenderAssetUsages;
    use bevy_mesh::{Indices, Mesh as BevyMesh, PrimitiveTopology as BevyPrimitiveTopology};
    use prism_render_driver::PrimitiveTopology;

    use super::*;

    #[test]
    fn converts_the_shading_contract_without_losing_topology_or_indices() {
        let source = BevyMesh::new(
            BevyPrimitiveTopology::TriangleList,
            RenderAssetUsages::MAIN_WORLD,
        )
        .with_inserted_attribute(
            BevyMesh::ATTRIBUTE_POSITION,
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        )
        .with_inserted_attribute(
            BevyMesh::ATTRIBUTE_UV_0,
            vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        )
        .with_inserted_indices(Indices::U16(vec![0, 1, 2]));

        let converted = to_prism_mesh(&source).expect("the source mesh converts");
        assert_eq!(
            converted.primitive_topology(),
            PrimitiveTopology::TriangleList
        );
        assert_eq!(converted.vertex_count(), 3);
        assert_eq!(converted.indices().map(|indices| indices.len()), Some(3));
        assert!(converted
            .attributes()
            .any(|(attribute, _)| attribute.name == UV_ATTRIBUTE));
    }
}
