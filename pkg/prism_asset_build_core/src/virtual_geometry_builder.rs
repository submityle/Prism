//! Deterministic baseline static-mesh clusterizer and page packer.
//!
//! This is the first executable M2 builder slice. It preserves authored
//! triangle order, respects material submesh boundaries, enforces the fixed
//! cluster limits, and emits real page payloads. It intentionally does not
//! claim geometric simplification or a multi-level LOD DAG yet: each emitted
//! cluster is a leaf/root, so the result is a correct streamed partition that
//! later simplification passes can refine without changing the asset ABI.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use prism_mesh::{content_hash, Mesh, MeshError, POSITION_ATTRIBUTE};
use prism_render_architecture::gpu_scene::SceneBounds;
use prism_render_architecture::virtual_geometry::{
    ClusterNode, GeometryPageKey, VirtualGeometryAsset, VirtualGeometryAssetError,
    VirtualGeometryAssetHeader, VirtualGeometryClusterRecord, VirtualGeometryCompression,
    VirtualGeometryPage, VirtualGeometryPageRecord, DEFAULT_VIRTUAL_GEOMETRY_PAGE_SIZE,
};
use prism_render_driver::PrimitiveTopology;

/// Payload signature for one uncompressed cluster record.
pub const STATIC_CLUSTER_PAYLOAD_MAGIC: [u8; 4] = *b"CLST";
/// Current payload encoding version.
pub const STATIC_CLUSTER_PAYLOAD_VERSION: u16 = 1;

/// Fixed limits and page policy for the baseline clusterizer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StaticClusterBuildSettings {
    /// Physical page size in bytes.
    pub page_size: u32,
    /// Maximum unique vertices in a cluster.
    pub max_vertices_per_cluster: u16,
    /// Maximum triangles in a cluster.
    pub max_triangles_per_cluster: u16,
    /// Codec applied to completed page payloads.
    pub compression: VirtualGeometryCompression,
}

impl Default for StaticClusterBuildSettings {
    fn default() -> Self {
        Self {
            page_size: DEFAULT_VIRTUAL_GEOMETRY_PAGE_SIZE,
            max_vertices_per_cluster: 64,
            max_triangles_per_cluster: 128,
            compression: VirtualGeometryCompression::None,
        }
    }
}

/// Failure while constructing a static virtual-geometry asset.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StaticClusterBuildError {
    /// The source mesh failed its own upload contract.
    InvalidMesh(MeshError),
    /// Only indexed triangle lists are supported by this baseline builder.
    UnsupportedTopology(PrimitiveTopology),
    /// A source mesh has no index buffer.
    RequiresIndices,
    /// The source has no finite Float32x3 position attribute.
    MissingPositions,
    /// Settings violate the runtime ABI limits.
    InvalidSettings(&'static str),
    /// A single cluster cannot fit within one page.
    ClusterTooLarge {
        /// Encoded cluster payload size.
        bytes: u32,
        /// Configured physical page size.
        page_size: u32,
    },
    /// The page cluster count cannot be represented by the asset ABI.
    PageClusterCountOverflow,
    /// Generated asset cross-validation failed.
    InvalidAsset(VirtualGeometryAssetError),
    /// A material submesh table does not cover the source index buffer exactly.
    SubmeshCoverage,
}

impl core::fmt::Display for StaticClusterBuildError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidMesh(error) => write!(f, "source mesh is invalid: {error}"),
            Self::UnsupportedTopology(topology) => {
                write!(
                    f,
                    "static clusterizer requires TriangleList, got {topology:?}"
                )
            }
            Self::RequiresIndices => f.write_str("static clusterizer requires indexed geometry"),
            Self::MissingPositions => {
                f.write_str("mesh has no finite Float32x3 position attribute")
            }
            Self::InvalidSettings(field) => write!(f, "invalid cluster build setting {field}"),
            Self::ClusterTooLarge { bytes, page_size } => {
                write!(
                    f,
                    "cluster payload is {bytes} bytes, larger than page size {page_size}"
                )
            }
            Self::PageClusterCountOverflow => f.write_str("page contains too many clusters"),
            Self::InvalidAsset(error) => {
                write!(f, "generated virtual-geometry asset is invalid: {error}")
            }
            Self::SubmeshCoverage => {
                f.write_str("submesh ranges do not cover the index buffer exactly")
            }
        }
    }
}

/// Builds a deterministic paged static virtual-geometry asset.
pub fn build_static_virtual_geometry(
    asset_id: u32,
    mesh: &Mesh,
    settings: StaticClusterBuildSettings,
) -> Result<VirtualGeometryAsset, StaticClusterBuildError> {
    validate_settings(settings)?;
    mesh.validate()
        .map_err(StaticClusterBuildError::InvalidMesh)?;
    if mesh.primitive_topology() != PrimitiveTopology::TriangleList {
        return Err(StaticClusterBuildError::UnsupportedTopology(
            mesh.primitive_topology(),
        ));
    }
    let indices = mesh
        .indices()
        .ok_or(StaticClusterBuildError::RequiresIndices)?;
    let positions = mesh
        .attributes()
        .find(|(attribute, _)| attribute.name == POSITION_ATTRIBUTE)
        .and_then(|(_, values)| values.as_positions())
        .ok_or(StaticClusterBuildError::MissingPositions)?;
    if positions
        .iter()
        .any(|position| !position.iter().all(|value| value.is_finite()))
    {
        return Err(StaticClusterBuildError::MissingPositions);
    }

    let index_values = indices.to_u32();
    let ranges = submesh_ranges(mesh, index_values.len() as u32)?;
    let mut nodes = Vec::new();
    let mut clusters = Vec::new();
    let mut pages = Vec::new();
    let mut page_payload = Vec::new();
    let mut page_cluster_start = 0u32;
    let mut page_key = GeometryPageKey::new(asset_id, 0);

    for (start, end, material_slot) in ranges {
        let mut local_indices = Vec::new();
        let mut local_vertices = Vec::new();
        let mut remap = BTreeMap::new();
        let mut triangle_count = 0u16;
        for triangle in (start as usize..end as usize).step_by(3) {
            let source_triangle = [
                index_values[triangle],
                index_values[triangle + 1],
                index_values[triangle + 2],
            ];
            let additional_vertices = source_triangle
                .iter()
                .filter(|index| !remap.contains_key(*index))
                .count();
            if triangle_count == settings.max_triangles_per_cluster
                || (!local_indices.is_empty()
                    && local_vertices.len() + additional_vertices
                        > settings.max_vertices_per_cluster as usize)
            {
                flush_cluster(
                    asset_id,
                    material_slot,
                    &local_vertices,
                    &local_indices,
                    settings,
                    &mut page_key,
                    &mut page_payload,
                    &mut page_cluster_start,
                    &mut pages,
                    &mut clusters,
                    &mut nodes,
                    settings.compression,
                )?;
                local_indices.clear();
                local_vertices.clear();
                remap.clear();
                triangle_count = 0;
            }
            for source_index in source_triangle {
                let local_index = if let Some(local_index) = remap.get(&source_index) {
                    *local_index
                } else {
                    let local_index = u16::try_from(local_vertices.len()).map_err(|_| {
                        StaticClusterBuildError::InvalidSettings("max_vertices_per_cluster")
                    })?;
                    remap.insert(source_index, local_index);
                    local_vertices.push(positions[source_index as usize]);
                    local_index
                };
                local_indices.push(local_index);
            }
            triangle_count += 1;
        }
        if !local_indices.is_empty() {
            flush_cluster(
                asset_id,
                material_slot,
                &local_vertices,
                &local_indices,
                settings,
                &mut page_key,
                &mut page_payload,
                &mut page_cluster_start,
                &mut pages,
                &mut clusters,
                &mut nodes,
                settings.compression,
            )?;
        }
    }
    finish_page(
        page_key,
        settings.page_size,
        page_cluster_start,
        &mut page_payload,
        &mut pages,
        clusters.len() as u32,
        settings.compression,
    )?;

    let bounds = mesh
        .bounds()
        .ok_or(StaticClusterBuildError::MissingPositions)?;
    let center = bounds.center();
    let extents = bounds.extents();
    let header = VirtualGeometryAssetHeader {
        asset_id,
        schema_version: prism_render_architecture::virtual_geometry::VIRTUAL_GEOMETRY_ASSET_SCHEMA,
        page_size: settings.page_size,
        build_settings_hash: settings_hash(settings),
        content_hash: content_hash(mesh).as_bytes(),
        bounds: SceneBounds {
            center,
            radius: bounds.radius(),
            half_extents: [extents[0] * 0.5, extents[1] * 0.5, extents[2] * 0.5],
            _padding: 0.0,
        },
        material_slot_count: mesh
            .submeshes()
            .iter()
            .map(|submesh| u32::from(submesh.material_slot) + 1)
            .max()
            .unwrap_or(1),
        fallback_mesh_hash: [0; 32],
    };
    let asset = VirtualGeometryAsset {
        header,
        nodes,
        roots: (0..clusters.len() as u32).collect(),
        pages,
        clusters,
    };
    asset
        .validate()
        .map_err(StaticClusterBuildError::InvalidAsset)?;
    Ok(asset)
}

fn validate_settings(settings: StaticClusterBuildSettings) -> Result<(), StaticClusterBuildError> {
    if settings.page_size < 1024 || !settings.page_size.is_power_of_two() {
        return Err(StaticClusterBuildError::InvalidSettings("page_size"));
    }
    if settings.max_vertices_per_cluster == 0 || settings.max_vertices_per_cluster > 64 {
        return Err(StaticClusterBuildError::InvalidSettings(
            "max_vertices_per_cluster",
        ));
    }
    if settings.max_triangles_per_cluster == 0 || settings.max_triangles_per_cluster > 128 {
        return Err(StaticClusterBuildError::InvalidSettings(
            "max_triangles_per_cluster",
        ));
    }
    Ok(())
}

fn submesh_ranges(
    mesh: &Mesh,
    index_count: u32,
) -> Result<Vec<(u32, u32, u16)>, StaticClusterBuildError> {
    if mesh.submeshes().is_empty() {
        return Ok(alloc::vec![(0, index_count, 0)]);
    }
    let mut ranges: Vec<_> = mesh
        .submeshes()
        .iter()
        .map(|submesh| {
            (
                submesh.index_range.start,
                submesh.index_range.end,
                submesh.material_slot,
            )
        })
        .collect();
    ranges.sort_by_key(|range| range.0);
    let mut cursor = 0;
    for (start, end, _) in &ranges {
        if *start != cursor || *end > index_count || (*end - *start) % 3 != 0 {
            return Err(StaticClusterBuildError::SubmeshCoverage);
        }
        cursor = *end;
    }
    if cursor != index_count {
        return Err(StaticClusterBuildError::SubmeshCoverage);
    }
    Ok(ranges)
}

#[allow(clippy::too_many_arguments)]
fn flush_cluster(
    asset_id: u32,
    material_slot: u16,
    vertices: &[[f32; 3]],
    indices: &[u16],
    settings: StaticClusterBuildSettings,
    page_key: &mut GeometryPageKey,
    page_payload: &mut Vec<u8>,
    page_cluster_start: &mut u32,
    pages: &mut Vec<VirtualGeometryPage>,
    clusters: &mut Vec<VirtualGeometryClusterRecord>,
    nodes: &mut Vec<ClusterNode>,
    compression: VirtualGeometryCompression,
) -> Result<(), StaticClusterBuildError> {
    let payload = encode_cluster_payload(vertices, indices);
    if payload.len() > settings.page_size as usize {
        return Err(StaticClusterBuildError::ClusterTooLarge {
            bytes: payload.len() as u32,
            page_size: settings.page_size,
        });
    }
    if !page_payload.is_empty() && page_payload.len() + payload.len() > settings.page_size as usize
    {
        finish_page(
            *page_key,
            settings.page_size,
            *page_cluster_start,
            page_payload,
            pages,
            clusters.len() as u32,
            compression,
        )?;
        page_key.page += 1;
        *page_cluster_start = clusters.len() as u32;
    }
    let payload_offset = page_payload.len() as u32;
    page_payload.extend_from_slice(&payload);
    let bounds = cluster_bounds(vertices);
    let normal_cone = normal_cone(vertices, indices);
    let node_index = nodes.len() as u32;
    nodes.push(ClusterNode::leaf(bounds, 0.0, *page_key));
    clusters.push(VirtualGeometryClusterRecord {
        node_index,
        page: *page_key,
        material_slot,
        vertex_count: vertices.len() as u16,
        triangle_count: (indices.len() / 3) as u16,
        payload_offset,
        payload_size: payload.len() as u32,
        bounds,
        normal_cone,
    });
    let _ = asset_id;
    Ok(())
}

fn finish_page(
    key: GeometryPageKey,
    page_size: u32,
    cluster_start: u32,
    payload: &mut Vec<u8>,
    pages: &mut Vec<VirtualGeometryPage>,
    cluster_end: u32,
    compression: VirtualGeometryCompression,
) -> Result<(), StaticClusterBuildError> {
    if payload.is_empty() {
        return Ok(());
    }
    let cluster_count = cluster_end.saturating_sub(cluster_start);
    let cluster_count = u16::try_from(cluster_count)
        .map_err(|_| StaticClusterBuildError::PageClusterCountOverflow)?;
    if payload.len() > page_size as usize {
        return Err(StaticClusterBuildError::ClusterTooLarge {
            bytes: payload.len() as u32,
            page_size,
        });
    }
    let bytes = core::mem::take(payload);
    let stored = match compression {
        VirtualGeometryCompression::None => bytes.clone(),
        VirtualGeometryCompression::GDeflate => {
            prism_render_architecture::compression::gdeflate_compress(&bytes)
        }
    };
    pages.push(VirtualGeometryPage {
        record: VirtualGeometryPageRecord {
            key,
            compression,
            uncompressed_size: bytes.len() as u32,
            priority: 0,
            cluster_start,
            cluster_count,
        },
        payload: stored,
    });
    Ok(())
}

fn encode_cluster_payload(vertices: &[[f32; 3]], indices: &[u16]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(12 + vertices.len() * 12 + indices.len() * 2);
    bytes.extend_from_slice(&STATIC_CLUSTER_PAYLOAD_MAGIC);
    bytes.extend_from_slice(&STATIC_CLUSTER_PAYLOAD_VERSION.to_le_bytes());
    bytes.extend_from_slice(&(vertices.len() as u16).to_le_bytes());
    bytes.extend_from_slice(&((indices.len() / 3) as u16).to_le_bytes());
    for vertex in vertices {
        for value in vertex {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    for index in indices {
        bytes.extend_from_slice(&index.to_le_bytes());
    }
    bytes
}

fn cluster_bounds(vertices: &[[f32; 3]]) -> SceneBounds {
    let mut min = vertices[0];
    let mut max = vertices[0];
    for vertex in &vertices[1..] {
        for axis in 0..3 {
            min[axis] = min[axis].min(vertex[axis]);
            max[axis] = max[axis].max(vertex[axis]);
        }
    }
    let center = [
        (min[0] + max[0]) * 0.5,
        (min[1] + max[1]) * 0.5,
        (min[2] + max[2]) * 0.5,
    ];
    let half_extents = [
        (max[0] - min[0]) * 0.5,
        (max[1] - min[1]) * 0.5,
        (max[2] - min[2]) * 0.5,
    ];
    SceneBounds {
        center,
        radius: (half_extents[0] * half_extents[0]
            + half_extents[1] * half_extents[1]
            + half_extents[2] * half_extents[2])
            .sqrt(),
        half_extents,
        _padding: 0.0,
    }
}

fn normal_cone(vertices: &[[f32; 3]], indices: &[u16]) -> [f32; 4] {
    let mut normals = Vec::new();
    for triangle in indices.chunks_exact(3) {
        let a = vertices[triangle[0] as usize];
        let b = vertices[triangle[1] as usize];
        let c = vertices[triangle[2] as usize];
        let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let n = [
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        ];
        let length = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if length > f32::EPSILON {
            normals.push([n[0] / length, n[1] / length, n[2] / length]);
        }
    }
    if normals.is_empty() {
        return [0.0, 0.0, 1.0, -1.0];
    }
    let mut axis = [0.0; 3];
    for normal in &normals {
        for component in 0..3 {
            axis[component] += normal[component];
        }
    }
    let length = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    if length <= f32::EPSILON {
        return [0.0, 0.0, 1.0, -1.0];
    }
    for component in &mut axis {
        *component /= length;
    }
    let cosine = normals
        .iter()
        .map(|normal| axis[0] * normal[0] + axis[1] * normal[1] + axis[2] * normal[2])
        .fold(1.0, f32::min)
        .clamp(-1.0, 1.0);
    [axis[0], axis[1], axis[2], cosine]
}

fn settings_hash(settings: StaticClusterBuildSettings) -> [u8; 32] {
    let mut bytes = [0u8; 9];
    bytes[0..4].copy_from_slice(&settings.page_size.to_le_bytes());
    bytes[4..6].copy_from_slice(&settings.max_vertices_per_cluster.to_le_bytes());
    bytes[6..8].copy_from_slice(&settings.max_triangles_per_cluster.to_le_bytes());
    bytes[8] = settings.compression as u8;
    *blake3::hash(&bytes).as_bytes()
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use prism_mesh::{Indices, Mesh, MeshVertexAttribute, SubMesh, VertexAttributeValues};
    use prism_render_driver::{PrimitiveTopology, VertexFormat};

    use super::*;

    fn mesh(triangle_count: u32) -> Mesh {
        let mut positions = Vec::new();
        let mut indices = Vec::new();
        for triangle in 0..triangle_count {
            let base = positions.len() as u32;
            let x = triangle as f32;
            positions.extend_from_slice(&[[x, 0.0, 0.0], [x + 0.5, 0.0, 0.0], [x, 0.5, 0.0]]);
            indices.extend_from_slice(&[base, base + 1, base + 2]);
        }
        Mesh::new(PrimitiveTopology::TriangleList)
            .with_attribute(
                MeshVertexAttribute::new(POSITION_ATTRIBUTE, VertexFormat::Float32x3, 0),
                VertexAttributeValues::Float32x3(positions),
            )
            .expect("positions fit")
            .indexed(Indices::Uint32(indices))
    }

    #[test]
    fn clusterizer_respects_cluster_limits_and_emits_pages() {
        let asset = build_static_virtual_geometry(
            9,
            &mesh(3),
            StaticClusterBuildSettings {
                page_size: 1024,
                max_vertices_per_cluster: 64,
                max_triangles_per_cluster: 2,
                compression: VirtualGeometryCompression::None,
            },
        )
        .expect("clusterizes");
        assert_eq!(asset.clusters.len(), 2);
        assert_eq!(asset.pages.len(), 1);
        assert!(asset
            .clusters
            .iter()
            .all(|cluster| cluster.triangle_count <= 2));
        assert!(asset.encode().is_ok());
    }

    #[test]
    fn clusterizer_keeps_material_ranges_separate() {
        let source = mesh(2).submeshed(vec![SubMesh::new(0, 0..3), SubMesh::new(1, 3..6)]);
        source.validate().expect("submeshes valid");
        let asset =
            build_static_virtual_geometry(1, &source, StaticClusterBuildSettings::default())
                .expect("clusterizes");
        assert_eq!(asset.clusters.len(), 2);
        assert_eq!(asset.clusters[0].material_slot, 0);
        assert_eq!(asset.clusters[1].material_slot, 1);
    }

    #[test]
    fn clusterizer_can_emit_gdeflate_pages_with_uncompressed_cluster_offsets() {
        let asset = build_static_virtual_geometry(
            3,
            &mesh(2),
            StaticClusterBuildSettings {
                page_size: 1024,
                max_vertices_per_cluster: 64,
                max_triangles_per_cluster: 128,
                compression: VirtualGeometryCompression::GDeflate,
            },
        )
        .expect("clusterizes");
        let page = &asset.pages[0];
        assert_eq!(
            page.record.compression,
            VirtualGeometryCompression::GDeflate
        );
        assert_eq!(
            prism_render_architecture::compression::gdeflate_decompress(&page.payload)
                .unwrap()
                .len(),
            page.record.uncompressed_size as usize
        );
        assert!(asset.encode().is_ok());
    }

    #[test]
    fn clusterizer_rejects_non_indexed_and_non_triangle_meshes() {
        let non_indexed = Mesh::new(PrimitiveTopology::TriangleList)
            .with_attribute(
                MeshVertexAttribute::new(POSITION_ATTRIBUTE, VertexFormat::Float32x3, 0),
                VertexAttributeValues::Float32x3(vec![[0.0, 0.0, 0.0]; 3]),
            )
            .expect("positions fit");
        assert_eq!(
            build_static_virtual_geometry(1, &non_indexed, StaticClusterBuildSettings::default()),
            Err(StaticClusterBuildError::RequiresIndices)
        );
    }
}
