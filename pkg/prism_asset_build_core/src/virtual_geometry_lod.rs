//! QEM-backed static virtual-geometry LOD chain construction.

use alloc::vec::Vec;

use prism_mesh::{Mesh, MeshError, POSITION_ATTRIBUTE};
use prism_render_architecture::ray_scene::{decimate, DecimationError, TriangleMesh};
use prism_render_driver::{PrimitiveTopology, VertexFormat};

use crate::virtual_geometry_builder::{
    build_static_virtual_geometry, StaticClusterBuildError, StaticClusterBuildSettings,
};
use prism_render_architecture::virtual_geometry::VirtualGeometryAsset;

/// One independently serialized level in a static LOD chain.
#[derive(Clone, Debug, PartialEq)]
pub struct StaticVirtualGeometryLod {
    /// Level index in the requested target order.
    pub level: u32,
    /// Triangle target passed to the QEM decimator.
    pub target_triangles: u32,
    /// Maximum distance from an authored vertex to its nearest simplified
    /// vertex. This is a vertex-sample metric, not a full Hausdorff bound.
    pub vertex_error: f32,
    /// Paged static virtual-geometry asset for this level.
    pub asset: VirtualGeometryAsset,
}

/// Failure while building a static QEM LOD chain.
#[derive(Clone, Debug, PartialEq)]
pub enum StaticVirtualGeometryLodError {
    /// Source mesh validation failed.
    InvalidMesh(MeshError),
    /// The source must have indexed triangle-list topology.
    UnsupportedSource,
    /// Simplification across material partitions is not allowed by this slice.
    MaterialSubmeshesUnsupported,
    /// The target is zero or greater than the source triangle count.
    InvalidTarget,
    /// Existing QEM decimator failure.
    Decimation(DecimationError),
    /// The resulting mesh could not be turned into paged geometry.
    Build(StaticClusterBuildError),
}

impl core::fmt::Display for StaticVirtualGeometryLodError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidMesh(error) => write!(f, "invalid source mesh: {error}"),
            Self::UnsupportedSource => f.write_str("LOD source must be indexed TriangleList geometry"),
            Self::MaterialSubmeshesUnsupported => {
                f.write_str("LOD simplification cannot cross material submesh boundaries")
            }
            Self::InvalidTarget => f.write_str("LOD target is outside the source triangle range"),
            Self::Decimation(error) => write!(f, "QEM decimation failed: {error}"),
            Self::Build(error) => write!(f, "LOD page build failed: {error}"),
        }
    }
}

/// Builds independent QEM-reduced paged assets in the caller's target order.
pub fn build_static_virtual_geometry_lod_chain(
    asset_id: u32,
    mesh: &Mesh,
    settings: StaticClusterBuildSettings,
    targets: &[u32],
) -> Result<Vec<StaticVirtualGeometryLod>, StaticVirtualGeometryLodError> {
    if mesh.primitive_topology() != PrimitiveTopology::TriangleList || mesh.indices().is_none() {
        return Err(StaticVirtualGeometryLodError::UnsupportedSource);
    }
    if !mesh.submeshes().is_empty() {
        return Err(StaticVirtualGeometryLodError::MaterialSubmeshesUnsupported);
    }
    mesh.validate()
        .map_err(StaticVirtualGeometryLodError::InvalidMesh)?;
    if targets.is_empty() {
        return Err(StaticVirtualGeometryLodError::InvalidTarget);
    }
    let positions = positions(mesh)?;
    let source_indices = mesh.indices().expect("checked above").to_u32();
    let source_triangles = source_indices.len() / 3;
    if targets
        .iter()
        .any(|target| *target == 0 || *target as usize > source_triangles)
    {
        return Err(StaticVirtualGeometryLodError::InvalidTarget);
    }
    let source = TriangleMesh::new(
        positions.to_vec(),
        Vec::new(),
        Vec::new(),
        source_indices
            .chunks_exact(3)
            .map(|triangle| [triangle[0], triangle[1], triangle[2]])
            .collect(),
    )
    .map_err(|_| StaticVirtualGeometryLodError::UnsupportedSource)?;
    let mut output = Vec::with_capacity(targets.len());
    for (level, target) in targets.iter().copied().enumerate() {
        let simplified = decimate(&source, target as usize)
            .map_err(StaticVirtualGeometryLodError::Decimation)?;
        let simplified_mesh = to_runtime_mesh(&simplified)?;
        let asset = build_static_virtual_geometry(asset_id, &simplified_mesh, settings)
            .map_err(StaticVirtualGeometryLodError::Build)?;
        output.push(StaticVirtualGeometryLod {
            level: level as u32,
            target_triangles: target,
            vertex_error: vertex_sample_error(positions, simplified.positions()),
            asset,
        });
    }
    Ok(output)
}

fn positions(mesh: &Mesh) -> Result<&[[f32; 3]], StaticVirtualGeometryLodError> {
    mesh.attributes()
        .find(|(attribute, _)| attribute.name == POSITION_ATTRIBUTE)
        .and_then(|(_, values)| values.as_positions())
        .ok_or(StaticVirtualGeometryLodError::InvalidMesh(
            MeshError::NoAttributes,
        ))
}

fn to_runtime_mesh(mesh: &TriangleMesh) -> Result<Mesh, StaticVirtualGeometryLodError> {
    let mut output = Mesh::new(PrimitiveTopology::TriangleList)
        .with_attribute(
            prism_mesh::MeshVertexAttribute::new(POSITION_ATTRIBUTE, VertexFormat::Float32x3, 0),
            prism_mesh::VertexAttributeValues::Float32x3(mesh.positions().to_vec()),
        )
        .map_err(StaticVirtualGeometryLodError::InvalidMesh)?;
    let indices = mesh
        .indices()
        .iter()
        .flat_map(|triangle| triangle.iter().copied())
        .collect();
    output.with_indices(prism_mesh::Indices::Uint32(indices));
    output
        .validate()
        .map_err(StaticVirtualGeometryLodError::InvalidMesh)?;
    Ok(output)
}

fn vertex_sample_error(source: &[[f32; 3]], simplified: &[[f32; 3]]) -> f32 {
    source
        .iter()
        .map(|point| {
            simplified
                .iter()
                .map(|candidate| {
                    let dx = point[0] - candidate[0];
                    let dy = point[1] - candidate[1];
                    let dz = point[2] - candidate[2];
                    dx * dx + dy * dy + dz * dz
                })
                .fold(f32::INFINITY, f32::min)
        })
        .fold(0.0, f32::max)
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_mesh::primitives::cube;

    #[test]
    fn qem_chain_builds_real_reduced_assets_with_errors() {
        let source = cube([2.0, 2.0, 2.0]).expect("cube");
        let result = build_static_virtual_geometry_lod_chain(
            21,
            &source,
            StaticClusterBuildSettings::default(),
            &[12, 4],
        )
        .expect("LOD chain");
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].level, 0);
        assert_eq!(result[1].target_triangles, 4);
        assert!(result.iter().all(|lod| lod.vertex_error.is_finite()));
        assert!(result.iter().all(|lod| lod.asset.encode().is_ok()));
    }

    #[test]
    fn qem_chain_refuses_material_partitions_until_boundary_locks_exist() {
        let mut source = cube([2.0, 2.0, 2.0]).expect("cube");
        source = source.submeshed(vec![prism_mesh::SubMesh::new(0, 0..18)]);
        assert_eq!(
            build_static_virtual_geometry_lod_chain(
                1,
                &source,
                StaticClusterBuildSettings::default(),
                &[4]
            ),
            Err(StaticVirtualGeometryLodError::MaterialSubmeshesUnsupported)
        );
    }
}
