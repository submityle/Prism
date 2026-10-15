//! Deterministic static-mesh clusterizer and page packer.
//!
//! It preserves authored triangle order, respects material submesh ownership,
//! enforces the fixed cluster limits, and emits versioned page payloads. Each
//! emitted cluster is a leaf/root in the asset produced by this function. QEM
//! simplification and LOD selection are supplied by
//! [`crate::virtual_geometry_lod`], which builds independent versioned assets
//! over the same page ABI.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::virtual_geometry_payload::{
    encode_cluster_payload_with_attributes, ClusterPayloadError, ClusterVertexAttribute,
    MAX_CLUSTER_ATTRIBUTES, MAX_CLUSTER_ATTRIBUTE_NAME_BYTES,
};
use prism_mesh::{
    content_hash, Mesh, MeshBounds, MeshError, MeshVertexAttribute, VertexAttributeValues,
    POSITION_ATTRIBUTE,
};
use prism_render_architecture::gpu_scene::SceneBounds;
use prism_render_architecture::virtual_geometry::{
    ClusterNode, GeometryPageKey, VirtualGeometryAsset, VirtualGeometryAssetError,
    VirtualGeometryAssetHeader, VirtualGeometryClusterRecord, VirtualGeometryCompression,
    VirtualGeometryPage, VirtualGeometryPageRecord, DEFAULT_VIRTUAL_GEOMETRY_PAGE_SIZE,
};
use prism_render_driver::PrimitiveTopology;

/// Payload signature and schema shared with the public cluster codec.
pub use crate::virtual_geometry_payload::{
    STATIC_CLUSTER_PAYLOAD_MAGIC, STATIC_CLUSTER_PAYLOAD_VERSION,
};

/// Fixed limits and page policy for the static clusterizer.
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
    /// Only indexed triangle lists are supported by this static builder.
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
    /// A generated table count or range cannot be represented by the asset ABI.
    AbiCountOverflow(&'static str),
    /// The next page key cannot be represented by the asset ABI.
    PageKeyOverflow,
    /// The page cluster count cannot be represented by the asset ABI.
    PageClusterCountOverflow,
    /// Shared cluster payload encoder rejected generated data.
    InvalidPayload(ClusterPayloadError),
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
            Self::AbiCountOverflow(kind) => {
                write!(f, "{kind} count exceeds the virtual-geometry ABI")
            }
            Self::PageKeyOverflow => f.write_str("page key exceeds the virtual-geometry ABI"),
            Self::PageClusterCountOverflow => f.write_str("page contains too many clusters"),
            Self::InvalidPayload(error) => write!(f, "invalid cluster payload: {error}"),
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
    let (position_attribute, position_values) = mesh
        .attributes()
        .find(|(attribute, _)| attribute.name == POSITION_ATTRIBUTE)
        .ok_or(StaticClusterBuildError::MissingPositions)?;
    let positions = position_values
        .as_positions()
        .ok_or(StaticClusterBuildError::MissingPositions)?;
    let attributes: Vec<_> = mesh
        .attributes()
        .filter(|(attribute, _)| attribute.name != POSITION_ATTRIBUTE)
        .map(|(attribute, values)| (attribute, values))
        .collect();
    if attributes.len() >= MAX_CLUSTER_ATTRIBUTES
        || attributes.iter().any(|(attribute, _)| {
            attribute.name.is_empty()
                || attribute.name.len() > MAX_CLUSTER_ATTRIBUTE_NAME_BYTES
                || attribute.name.chars().any(char::is_control)
        })
    {
        return Err(StaticClusterBuildError::InvalidPayload(
            ClusterPayloadError::InvalidAttribute,
        ));
    }
    let source_attributes = SourceAttributes {
        position_shader_location: position_attribute.shader_location,
        attribute_header_bytes: attributes
            .iter()
            .map(|(attribute, _)| 12 + ((attribute.name.len() + 3) & !3))
            .sum(),
        attribute_stride: attributes
            .iter()
            .map(|(attribute, _)| attribute.format.size() as usize)
            .sum(),
        attributes,
    };
    if positions
        .iter()
        .any(|position| !position.iter().all(|value| value.is_finite()))
    {
        return Err(StaticClusterBuildError::MissingPositions);
    }

    let index_values = indices.to_u32();
    let index_count = abi_u32(index_values.len(), "index")?;
    let ranges = submesh_ranges(mesh, index_count)?;
    let mut nodes = Vec::new();
    let mut clusters = Vec::new();
    let mut pages = Vec::new();
    let mut page_payload = Vec::new();
    let mut page_cluster_start = 0u32;
    let mut page_key = GeometryPageKey::new(asset_id, 0);

    for (start, end, material_slot) in ranges {
        let mut local_indices = Vec::new();
        let mut local_vertices = Vec::new();
        let mut local_source_indices = Vec::new();
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
                .enumerate()
                .filter(|(corner, index)| {
                    !remap.contains_key(*index) && !source_triangle[..*corner].contains(index)
                })
                .count();
            let next_payload_bytes = source_attributes.payload_size(
                local_vertices.len() + additional_vertices,
                local_indices.len() + 3,
            );
            if triangle_count == settings.max_triangles_per_cluster
                || (!local_indices.is_empty()
                    && (local_vertices.len() + additional_vertices
                        > settings.max_vertices_per_cluster as usize
                        || next_payload_bytes > settings.page_size as usize))
            {
                flush_cluster(
                    material_slot,
                    &local_vertices,
                    &local_indices,
                    &local_source_indices,
                    &source_attributes,
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
                local_source_indices.clear();
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
                    local_source_indices.push(source_index);
                    local_index
                };
                local_indices.push(local_index);
            }
            triangle_count += 1;
        }
        if !local_indices.is_empty() {
            flush_cluster(
                material_slot,
                &local_vertices,
                &local_indices,
                &local_source_indices,
                &source_attributes,
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
        abi_u32(clusters.len(), "cluster")?,
        settings.compression,
    )?;

    let bounds = mesh
        .bounds()
        .ok_or(StaticClusterBuildError::MissingPositions)?;
    let center = bounds.center();
    let extents = bounds.extents();
    let material_slot_count = mesh.submeshes().iter().try_fold(1u32, |count, submesh| {
        let slot_count = u32::from(submesh.material_slot)
            .checked_add(1)
            .ok_or(StaticClusterBuildError::AbiCountOverflow("material slot"))?;
        Ok::<_, StaticClusterBuildError>(count.max(slot_count))
    })?;
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
        material_slot_count,
        fallback_mesh_hash: content_hash(mesh).as_bytes(),
    };
    let cluster_count = abi_u32(clusters.len(), "cluster")?;
    let asset = VirtualGeometryAsset {
        header,
        nodes,
        roots: (0..cluster_count).collect(),
        pages,
        clusters,
    };
    asset
        .validate()
        .map_err(StaticClusterBuildError::InvalidAsset)?;
    Ok(asset)
}

fn abi_u32(value: usize, kind: &'static str) -> Result<u32, StaticClusterBuildError> {
    u32::try_from(value).map_err(|_| StaticClusterBuildError::AbiCountOverflow(kind))
}

fn validate_settings(settings: StaticClusterBuildSettings) -> Result<(), StaticClusterBuildError> {
    if settings.page_size < 1024 || !settings.page_size.is_power_of_two() {
        return Err(StaticClusterBuildError::InvalidSettings("page_size"));
    }
    if settings.max_vertices_per_cluster < 3 || settings.max_vertices_per_cluster > 64 {
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
        let Some(length) = end.checked_sub(*start) else {
            return Err(StaticClusterBuildError::SubmeshCoverage);
        };
        if *start != cursor || *end > index_count || length % 3 != 0 {
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
    material_slot: u16,
    vertices: &[[f32; 3]],
    indices: &[u16],
    source_indices: &[u32],
    source_attributes: &SourceAttributes<'_>,
    settings: StaticClusterBuildSettings,
    page_key: &mut GeometryPageKey,
    page_payload: &mut Vec<u8>,
    page_cluster_start: &mut u32,
    pages: &mut Vec<VirtualGeometryPage>,
    clusters: &mut Vec<VirtualGeometryClusterRecord>,
    nodes: &mut Vec<ClusterNode>,
    compression: VirtualGeometryCompression,
) -> Result<(), StaticClusterBuildError> {
    let bounds = cluster_bounds(vertices);
    let attributes = source_attributes
        .attributes
        .iter()
        .map(|(attribute, values)| {
            Ok(ClusterVertexAttribute {
                semantic: attribute.name.into(),
                shader_location: attribute.shader_location,
                format: attribute.format,
                data: values.gather_bytes(source_indices).ok_or(
                    StaticClusterBuildError::InvalidPayload(ClusterPayloadError::InvalidAttribute),
                )?,
            })
        })
        .collect::<Result<Vec<_>, StaticClusterBuildError>>()?;
    let payload = encode_cluster_payload_with_attributes(
        bounds,
        vertices,
        indices,
        source_attributes.position_shader_location,
        &attributes,
    )
    .map_err(StaticClusterBuildError::InvalidPayload)?;
    debug_assert_eq!(
        payload.len(),
        source_attributes.payload_size(vertices.len(), indices.len())
    );
    if payload.len() > settings.page_size as usize {
        return Err(StaticClusterBuildError::ClusterTooLarge {
            bytes: abi_u32(payload.len(), "payload byte")?,
            page_size: settings.page_size,
        });
    }
    let combined_len = page_payload.len().checked_add(payload.len()).ok_or(
        StaticClusterBuildError::AbiCountOverflow("page payload byte"),
    )?;
    if !page_payload.is_empty() && combined_len > settings.page_size as usize {
        finish_page(
            *page_key,
            settings.page_size,
            *page_cluster_start,
            page_payload,
            pages,
            abi_u32(clusters.len(), "cluster")?,
            compression,
        )?;
        page_key.page = page_key
            .page
            .checked_add(1)
            .ok_or(StaticClusterBuildError::PageKeyOverflow)?;
        *page_cluster_start = abi_u32(clusters.len(), "cluster")?;
    }
    let payload_offset = abi_u32(page_payload.len(), "payload offset")?;
    let payload_size = abi_u32(payload.len(), "payload byte")?;
    let vertex_count = u16::try_from(vertices.len())
        .map_err(|_| StaticClusterBuildError::AbiCountOverflow("cluster vertex"))?;
    let triangle_count = u16::try_from(indices.len() / 3)
        .map_err(|_| StaticClusterBuildError::AbiCountOverflow("cluster triangle"))?;
    let node_index = abi_u32(nodes.len(), "node")?;
    page_payload.extend_from_slice(&payload);
    let normal_cone = normal_cone(vertices, indices);
    nodes.push(ClusterNode::leaf(bounds, 0.0, *page_key));
    clusters.push(VirtualGeometryClusterRecord {
        node_index,
        page: *page_key,
        material_slot,
        vertex_count,
        triangle_count,
        payload_offset,
        payload_size,
        bounds,
        normal_cone,
    });
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
    let cluster_count =
        cluster_end
            .checked_sub(cluster_start)
            .ok_or(StaticClusterBuildError::AbiCountOverflow(
                "page cluster range",
            ))?;
    let cluster_count = u16::try_from(cluster_count)
        .map_err(|_| StaticClusterBuildError::PageClusterCountOverflow)?;
    if payload.len() > page_size as usize {
        return Err(StaticClusterBuildError::ClusterTooLarge {
            bytes: abi_u32(payload.len(), "payload byte")?,
            page_size,
        });
    }
    let uncompressed_size = abi_u32(payload.len(), "page payload byte")?;
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
            uncompressed_size,
            priority: 0,
            cluster_start,
            cluster_count,
        },
        payload: stored,
    });
    Ok(())
}

fn cluster_bounds(vertices: &[[f32; 3]]) -> SceneBounds {
    // Cluster positions are validated before clustering. Reuse MeshBounds so
    // per-cluster culling metadata follows the shared overflow-safe policy.
    let bounds = MeshBounds::from_positions(vertices).expect("validated cluster positions");
    let extents = bounds.extents();
    SceneBounds {
        center: bounds.center(),
        radius: bounds.radius(),
        half_extents: [extents[0] * 0.5, extents[1] * 0.5, extents[2] * 0.5],
        _padding: 0.0,
    }
}

fn normal_cone(vertices: &[[f32; 3]], indices: &[u16]) -> [f32; 4] {
    let mut normals = Vec::new();
    for triangle in indices.chunks_exact(3) {
        let a = vertices[triangle[0] as usize].map(f64::from);
        let b = vertices[triangle[1] as usize].map(f64::from);
        let c = vertices[triangle[2] as usize].map(f64::from);
        let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let n = [
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        ];
        let length = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if length > f64::EPSILON {
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
    if length <= f64::EPSILON {
        return [0.0, 0.0, 1.0, -1.0];
    }
    for component in &mut axis {
        *component /= length;
    }
    let cosine = normals
        .iter()
        .map(|normal| axis[0] * normal[0] + axis[1] * normal[1] + axis[2] * normal[2])
        .fold(1.0, f64::min)
        .clamp(-1.0, 1.0);
    [
        axis[0] as f32,
        axis[1] as f32,
        axis[2] as f32,
        cosine as f32,
    ]
}

fn settings_hash(settings: StaticClusterBuildSettings) -> [u8; 32] {
    let mut bytes = [0u8; 9];
    bytes[0..4].copy_from_slice(&settings.page_size.to_le_bytes());
    bytes[4..6].copy_from_slice(&settings.max_vertices_per_cluster.to_le_bytes());
    bytes[6..8].copy_from_slice(&settings.max_triangles_per_cluster.to_le_bytes());
    bytes[8] = settings.compression as u8;
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"Prism.StaticClusterBuild.CLST");
    hasher.update(&STATIC_CLUSTER_PAYLOAD_VERSION.to_le_bytes());
    hasher.update(&bytes);
    *hasher.finalize().as_bytes()
}

/// Borrow authored streams once; clusters only gather their referenced vertices.
struct SourceAttributes<'a> {
    position_shader_location: u32,
    attributes: Vec<(&'a MeshVertexAttribute, &'a VertexAttributeValues)>,
    attribute_header_bytes: usize,
    attribute_stride: usize,
}

impl SourceAttributes<'_> {
    fn payload_size(&self, vertices: usize, indices: usize) -> usize {
        // Source and cluster counts are validated before use; at most 31
        // descriptors of 255 bytes and 64 vertices of 16 bytes each.
        ((34 + vertices * 6 + indices * 2 + 3) & !3)
            + 8
            + self.attribute_header_bytes
            + vertices * self.attribute_stride
    }
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
        assert_ne!(asset.header.fallback_mesh_hash, [0; 32]);
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
    fn cluster_bounds_remain_finite_for_large_finite_positions() {
        let max = f32::MAX;
        let source = Mesh::new(PrimitiveTopology::TriangleList)
            .with_attribute(
                MeshVertexAttribute::new(POSITION_ATTRIBUTE, VertexFormat::Float32x3, 0),
                VertexAttributeValues::Float32x3(vec![
                    [max * 0.5, 0.0, 0.0],
                    [max, 0.0, 0.0],
                    [max * 0.75, 1.0, 0.0],
                ]),
            )
            .expect("positions fit")
            .indexed(Indices::Uint32(vec![0, 1, 2]));
        let asset =
            build_static_virtual_geometry(11, &source, StaticClusterBuildSettings::default())
                .expect("large finite positions remain valid");
        let bounds = [asset.header.bounds]
            .into_iter()
            .chain(asset.clusters.iter().map(|cluster| cluster.bounds));
        for bound in bounds {
            assert!(bound.center.iter().all(|value| value.is_finite()));
            assert!(bound.half_extents.iter().all(|value| value.is_finite()));
            assert!(bound.radius.is_finite());
        }
        assert_eq!(asset.header.bounds.half_extents[0], f32::MAX * 0.25);
        assert!(asset
            .clusters
            .iter()
            .flat_map(|cluster| cluster.normal_cone)
            .all(f32::is_finite));
    }

    #[test]
    fn normal_cone_stays_finite_for_extreme_finite_triangles() {
        // SceneBounds stores half-extents, so keep the range representable after
        // converting the shared MeshBounds policy to that ABI.
        let max = f32::MAX * 0.25;
        let source = Mesh::new(PrimitiveTopology::TriangleList)
            .with_attribute(
                MeshVertexAttribute::new(POSITION_ATTRIBUTE, VertexFormat::Float32x3, 0),
                VertexAttributeValues::Float32x3(vec![
                    [max, 0.0, 0.0],
                    [-max, 0.0, 0.0],
                    [0.0, max, 0.0],
                ]),
            )
            .expect("positions fit")
            .indexed(Indices::Uint32(vec![0, 1, 2]));
        let asset =
            build_static_virtual_geometry(12, &source, StaticClusterBuildSettings::default())
                .expect("extreme finite triangle remains valid");
        assert!(asset.clusters[0]
            .normal_cone
            .iter()
            .all(|value| value.is_finite()));
    }

    #[test]
    fn settings_reject_cluster_caps_smaller_than_one_triangle() {
        for max_vertices_per_cluster in [1, 2] {
            let settings = StaticClusterBuildSettings {
                max_vertices_per_cluster,
                ..StaticClusterBuildSettings::default()
            };
            assert_eq!(
                validate_settings(settings),
                Err(StaticClusterBuildError::InvalidSettings(
                    "max_vertices_per_cluster"
                ))
            );
        }
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

    #[test]
    fn attributed_clusters_preserve_remaps_materials_and_compressed_stream_bytes() {
        let mut source = mesh(30)
            .with_attribute(
                MeshVertexAttribute::new("Vertex_Normal", VertexFormat::Float32x3, 1),
                VertexAttributeValues::Float32x3(vec![[0.0, 0.0, 1.0]; 90]),
            )
            .unwrap()
            .with_attribute(
                MeshVertexAttribute::new("Vertex_Uv0", VertexFormat::Float32x2, 2),
                VertexAttributeValues::Float32x2(
                    (0..90)
                        .map(|vertex| [vertex as f32, -(vertex as f32)])
                        .collect(),
                ),
            )
            .unwrap()
            .with_attribute(
                MeshVertexAttribute::new("Half", VertexFormat::Float16x2, 3),
                VertexAttributeValues::Float16x2(vec![[0x3c00, 0xbc00]; 90]),
            )
            .unwrap()
            .with_attribute(
                MeshVertexAttribute::new("SourceIndex", VertexFormat::Uint32, 4),
                VertexAttributeValues::Uint32((0..90).collect()),
            )
            .unwrap()
            .submeshed(vec![SubMesh::new(2, 0..45), SubMesh::new(3, 45..90)]);
        let reordered: Vec<u32> = (0..30)
            .flat_map(|triangle| [triangle * 3 + 2, triangle * 3, triangle * 3 + 1])
            .collect();
        source.with_indices(Indices::Uint32(reordered.clone()));
        for compression in [
            VirtualGeometryCompression::None,
            VirtualGeometryCompression::GDeflate,
        ] {
            let settings = StaticClusterBuildSettings {
                page_size: 1024,
                compression,
                ..Default::default()
            };
            let asset = build_static_virtual_geometry(1, &source, settings).unwrap();
            assert!(
                asset.clusters.len() > 2,
                "page-size-aware splitting must apply before codec rejection"
            );
            assert_eq!(
                build_static_virtual_geometry(1, &source, settings)
                    .unwrap()
                    .encode()
                    .unwrap(),
                asset.encode().unwrap()
            );
            let mut restored_indices = Vec::new();
            for cluster in &asset.clusters {
                let page = &asset.pages[cluster.page.page as usize];
                let raw = match compression {
                    VirtualGeometryCompression::None => page.payload.clone(),
                    _ => prism_render_architecture::compression::gdeflate_decompress(&page.payload)
                        .unwrap(),
                };
                let payload = &raw[cluster.payload_offset as usize
                    ..(cluster.payload_offset + cluster.payload_size) as usize];
                assert_eq!(payload.len() % 4, 0);
                assert_eq!(&payload[4..6], &3u16.to_le_bytes());
                let decoded = crate::decode_cluster_payload(cluster.bounds, payload).unwrap();
                assert_eq!(decoded.attributes.len(), 4);
                let ids: Vec<_> = decoded
                    .attributes
                    .iter()
                    .find(|attribute| attribute.semantic == "SourceIndex")
                    .unwrap()
                    .data
                    .chunks_exact(4)
                    .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
                    .collect();
                for (attribute, values) in source
                    .attributes()
                    .filter(|(attribute, _)| attribute.name != POSITION_ATTRIBUTE)
                {
                    let restored = decoded
                        .attributes
                        .iter()
                        .find(|restored| restored.semantic == attribute.name)
                        .unwrap();
                    assert_eq!(restored.shader_location, attribute.shader_location);
                    assert_eq!(restored.format, attribute.format);
                    assert_eq!(restored.data, values.gather_bytes(&ids).unwrap());
                }
                for local_index in decoded.indices {
                    let source_index = ids[local_index as usize];
                    assert_eq!(cluster.material_slot, if source_index < 45 { 2 } else { 3 });
                    restored_indices.push(source_index);
                }
            }
            assert_eq!(restored_indices, reordered);
            let mut requests = prism_render_architecture::paging::RequestBatch::new();
            for page in &asset.pages {
                requests.record(page.record.key, 1.0);
            }
            let page_count = asset.pages.len();
            let mut stream =
                prism_render_architecture::virtual_geometry::VirtualGeometryPageStream::from_asset(
                    asset.clone(),
                    page_count as u32,
                    page_count,
                )
                .unwrap();
            stream.reconcile(&requests, 1);
            assert_eq!(stream.resident_count(), page_count);
            for (key, slot) in stream.entries() {
                let page = &asset.pages[key.page as usize];
                let raw = match compression {
                    VirtualGeometryCompression::None => page.payload.clone(),
                    _ => prism_render_architecture::compression::gdeflate_decompress(&page.payload)
                        .unwrap(),
                };
                let stored: Vec<_> = stream
                    .storage()
                    .slot_words(slot)
                    .unwrap()
                    .iter()
                    .flat_map(|word| word.to_le_bytes())
                    .collect();
                assert_eq!(&stored[..raw.len()], raw.as_slice());
            }
        }
    }

    #[test]
    fn position_location_is_preserved_and_settings_hash_versions_the_codec() {
        let source = Mesh::new(PrimitiveTopology::TriangleList)
            .with_attribute(
                MeshVertexAttribute::new(POSITION_ATTRIBUTE, VertexFormat::Float32x3, 7),
                VertexAttributeValues::Float32x3(vec![[0.0; 3]; 3]),
            )
            .unwrap()
            .indexed(Indices::Uint16(vec![0, 1, 2]));
        let settings = StaticClusterBuildSettings::default();
        let asset = build_static_virtual_geometry(1, &source, settings).unwrap();
        let cluster = &asset.clusters[0];
        let decoded =
            crate::decode_cluster_payload(cluster.bounds, &asset.pages[0].payload).unwrap();
        assert_eq!(decoded.position_shader_location, 7);
        let mut old_settings = [0u8; 9];
        old_settings[0..4].copy_from_slice(&settings.page_size.to_le_bytes());
        old_settings[4..6].copy_from_slice(&settings.max_vertices_per_cluster.to_le_bytes());
        old_settings[6..8].copy_from_slice(&settings.max_triangles_per_cluster.to_le_bytes());
        old_settings[8] = settings.compression as u8;
        assert_ne!(
            asset.header.build_settings_hash,
            *blake3::hash(&old_settings).as_bytes()
        );
    }

    #[test]
    fn an_indivisible_triangle_larger_than_the_page_is_reported() {
        let mut source = mesh(1);
        for location in 1..32 {
            // Distinct stable semantics, with deliberately large per-triangle data.
            const NAMES: [&str; 31] = [
                "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11", "A12", "A13",
                "A14", "A15", "A16", "A17", "A18", "A19", "A20", "A21", "A22", "A23", "A24", "A25",
                "A26", "A27", "A28", "A29", "A30", "A31",
            ];
            source = source
                .with_attribute(
                    MeshVertexAttribute::new(
                        NAMES[location as usize - 1],
                        VertexFormat::Float32x4,
                        location,
                    ),
                    VertexAttributeValues::Float32x4(vec![[1.0; 4]; 3]),
                )
                .unwrap();
        }
        assert!(
            matches!(build_static_virtual_geometry(1, &source, StaticClusterBuildSettings { page_size: 1024, ..Default::default() }), Err(StaticClusterBuildError::ClusterTooLarge { bytes, page_size: 1024 }) if bytes > 1024)
        );
    }
}
