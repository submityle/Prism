//! QEM-backed static virtual-geometry LOD chain construction.

use alloc::collections::BTreeMap;
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
    /// Reserved for callers using an older boundary policy.
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
    mesh.validate()
        .map_err(StaticVirtualGeometryLodError::InvalidMesh)?;
    if targets.is_empty() {
        return Err(StaticVirtualGeometryLodError::InvalidTarget);
    }
    let source_positions = positions(mesh)?;
    let source_indices = mesh.indices().expect("checked above").to_u32();
    let source_triangles = source_indices.len() / 3;
    if targets
        .iter()
        .any(|target| *target == 0 || *target as usize > source_triangles)
    {
        return Err(StaticVirtualGeometryLodError::InvalidTarget);
    }
    let mut output = Vec::with_capacity(targets.len());
    for (level, target) in targets.iter().copied().enumerate() {
        let simplified_mesh = simplify_partitions(
            mesh,
            source_positions,
            &source_indices,
            target as usize,
            source_triangles,
        )?;
        let asset = build_static_virtual_geometry(asset_id, &simplified_mesh, settings)
            .map_err(StaticVirtualGeometryLodError::Build)?;
        let simplified_positions = positions(&simplified_mesh)?;
        output.push(StaticVirtualGeometryLod {
            level: level as u32,
            target_triangles: target,
            vertex_error: vertex_sample_error(source_positions, simplified_positions),
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

fn simplify_partitions(
    source: &Mesh,
    positions: &[[f32; 3]],
    indices: &[u32],
    target_triangles: usize,
    source_triangles: usize,
) -> Result<Mesh, StaticVirtualGeometryLodError> {
    let mut ranges: Vec<(u16, u32, u32)> = if source.submeshes().is_empty() {
        vec![(0, 0, indices.len() as u32)]
    } else {
        source
            .submeshes()
            .iter()
            .map(|submesh| {
                (
                    submesh.material_slot,
                    submesh.index_range.start,
                    submesh.index_range.end,
                )
            })
            .collect()
    };
    ranges.sort_by_key(|range| range.1);
    let mut output_positions = Vec::new();
    let mut output_indices: Vec<u32> = Vec::new();
    let mut output_submeshes = Vec::new();
    for (material_slot, start, end) in ranges {
        let mut remap = BTreeMap::new();
        let mut local_positions = Vec::new();
        let mut local_indices = Vec::new();
        for &global in &indices[start as usize..end as usize] {
            let local = if let Some(local) = remap.get(&global) {
                *local
            } else {
                let local = local_positions.len() as u32;
                remap.insert(global, local);
                local_positions.push(positions[global as usize]);
                local
            };
            local_indices.push(local);
        }
        let local_triangles = local_indices.len() / 3;
        let local_target = if target_triangles >= source_triangles {
            local_triangles
        } else {
            ((target_triangles * local_triangles) / source_triangles)
                .max(1)
                .min(local_triangles)
        };
        let local_mesh = TriangleMesh::new(
            local_positions,
            Vec::new(),
            Vec::new(),
            local_indices
                .chunks_exact(3)
                .map(|triangle| [triangle[0], triangle[1], triangle[2]])
                .collect(),
        )
        .map_err(|_| StaticVirtualGeometryLodError::UnsupportedSource)?;
        let simplified = decimate(&local_mesh, local_target)
            .map_err(StaticVirtualGeometryLodError::Decimation)?;
        let vertex_base = output_positions.len() as u32;
        output_positions.extend_from_slice(simplified.positions());
        let index_start = output_indices.len() as u32;
        output_indices.extend(
            simplified
                .indices()
                .iter()
                .flat_map(|triangle| triangle.iter().map(|index| vertex_base + *index)),
        );
        output_submeshes.push(prism_mesh::SubMesh::new(
            material_slot,
            index_start..output_indices.len() as u32,
        ));
    }
    let mut output = Mesh::new(PrimitiveTopology::TriangleList)
        .with_attribute(
            prism_mesh::MeshVertexAttribute::new(POSITION_ATTRIBUTE, VertexFormat::Float32x3, 0),
            prism_mesh::VertexAttributeValues::Float32x3(output_positions),
        )
        .map_err(StaticVirtualGeometryLodError::InvalidMesh)?;
    output.with_indices(prism_mesh::Indices::Uint32(output_indices));
    output.with_submeshes(output_submeshes);
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
    use alloc::vec;

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
    fn qem_chain_simplifies_material_partitions_independently() {
        let mut source = cube([2.0, 2.0, 2.0]).expect("cube");
        source = source.submeshed(vec![
            prism_mesh::SubMesh::new(0, 0..18),
            prism_mesh::SubMesh::new(1, 18..36),
        ]);
        let lods = build_static_virtual_geometry_lod_chain(
            1,
            &source,
            StaticClusterBuildSettings::default(),
            &[4],
        )
        .expect("material partitions simplify independently");
        assert_eq!(lods.len(), 1);
        assert!(lods[0].asset.clusters.iter().any(|cluster| cluster.material_slot == 0));
        assert!(lods[0].asset.clusters.iter().any(|cluster| cluster.material_slot == 1));
    }
}
