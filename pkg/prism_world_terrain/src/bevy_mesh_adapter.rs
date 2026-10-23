//! Bevy `Mesh` conversion at the renderer boundary.
//!
//! Terrain geometry remains engine-neutral in this crate; this optional
//! adapter preserves the ordinary indexed triangle mesh attributes and LOD
//! seam topology when handing a prepared tile to a Bevy `Assets<Mesh>` owner.

use alloc::vec::Vec;
use bevy_asset::RenderAssetUsages;
use bevy_mesh::{Indices, Mesh, PrimitiveTopology};

use crate::{HeightfieldMesh, PlanetHeightfieldMesh};

/// Converts a planar heightfield mesh into an indexed Bevy triangle mesh.
///
/// Geometry arrays are moved without copying. The conversion validates the
/// vertex/index topology before publishing the mesh; the resulting asset is
/// suitable for normal `Mesh3d`/render-asset extraction.
pub fn heightfield_to_bevy_mesh(
    terrain: HeightfieldMesh,
    usages: RenderAssetUsages,
) -> Result<Mesh, TerrainBevyMeshError> {
    let (positions, normals, uvs, indices) = terrain.into_bevy_parts();
    convert(positions, normals, uvs, indices, usages)
}

/// Converts a cube-sphere terrain tile into an indexed Bevy triangle mesh.
///
/// Positions remain tile-origin-relative as produced by the planetary
/// precision contract; callers must apply the matching frame/scene transform.
pub fn planet_heightfield_to_bevy_mesh(
    terrain: PlanetHeightfieldMesh,
    usages: RenderAssetUsages,
) -> Result<Mesh, TerrainBevyMeshError> {
    let (positions, normals, uvs, indices) = terrain.into_bevy_parts();
    convert(positions, normals, uvs, indices, usages)
}

fn convert(
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    indices: Vec<u32>,
    usages: RenderAssetUsages,
) -> Result<Mesh, TerrainBevyMeshError> {
    if positions.is_empty()
        || positions.len() != normals.len()
        || positions.len() != uvs.len()
        || !indices.len().is_multiple_of(3)
        || indices
            .iter()
            .any(|&index| index as usize >= positions.len())
        || positions
            .iter()
            .flatten()
            .chain(normals.iter().flatten())
            .chain(uvs.iter().flatten())
            .any(|value| !value.is_finite())
    {
        return Err(TerrainBevyMeshError::InvalidGeometry);
    }
    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, usages);
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(Indices::U32(indices));
    Ok(mesh)
}

/// Failure while converting an engine-neutral terrain mesh to a Bevy asset.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TerrainBevyMeshError {
    /// Terrain vertex attributes or triangle indices are malformed/non-finite.
    InvalidGeometry,
}

impl core::fmt::Display for TerrainBevyMeshError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidGeometry => formatter.write_str("invalid terrain mesh geometry"),
        }
    }
}

impl core::error::Error for TerrainBevyMeshError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        BakeLimits, CubeFace, DeterministicHeightSampler, HeightfieldSpec, HeightfieldTile,
        PlanetMeshConfig, TileKey,
    };

    #[test]
    fn planetary_mesh_preserves_attributes_indices_and_lod_seam_topology() {
        let tile = TileKey::new(CubeFace::PositiveY, 3, 3, 2).unwrap();
        let source = HeightfieldTile::bake(
            tile,
            HeightfieldSpec::new(3, 10).unwrap(),
            &DeterministicHeightSampler::new(0, 0),
            BakeLimits {
                max_samples: 64,
                max_bytes: 4096,
            },
        )
        .unwrap();
        let mut terrain = PlanetHeightfieldMesh::build(
            &source,
            PlanetMeshConfig::new(10_000, [0; 3], 0.1).unwrap(),
            BakeLimits {
                max_samples: 64,
                max_bytes: 4096,
            },
        )
        .unwrap();
        let neighbor = tile.coarser_neighbor(crate::TileEdge::East).unwrap();
        let plan = crate::HeightfieldLodSeamPlan::build_strict_for_residency(
            tile,
            2,
            &[neighbor],
            crate::LodSeamLimits {
                max_segments: 2,
                max_vertices: 32,
                max_indices: 128,
                max_bytes: 4096,
            },
        )
        .unwrap();
        terrain.apply_lod_seam_plan(&plan, 4096).unwrap();
        let expected_indices = terrain.indices().to_vec();
        let mesh = planet_heightfield_to_bevy_mesh(terrain, RenderAssetUsages::MAIN_WORLD).unwrap();
        assert_eq!(mesh.primitive_topology(), PrimitiveTopology::TriangleList);
        assert!(
            matches!(mesh.attribute(Mesh::ATTRIBUTE_POSITION), Some(bevy_mesh::VertexAttributeValues::Float32x3(values)) if values.len() == 9)
        );
        assert!(
            matches!(mesh.attribute(Mesh::ATTRIBUTE_NORMAL), Some(bevy_mesh::VertexAttributeValues::Float32x3(values)) if values.len() == 9)
        );
        assert!(
            matches!(mesh.attribute(Mesh::ATTRIBUTE_UV_0), Some(bevy_mesh::VertexAttributeValues::Float32x2(values)) if values.len() == 9)
        );
        assert_eq!(mesh.indices().unwrap(), &Indices::U32(expected_indices));
    }
}
