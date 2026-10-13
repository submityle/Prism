//! Asset-build bridge for the versioned virtual-geometry page format.
//!
//! This module deliberately canonicalizes and validates a prepared asset; it
//! does not pretend to be a clusterizer or simplifier. Those algorithms need
//! authored mesh IR and build settings, while this boundary makes the output
//! contract executable inside the existing deterministic build graph and CAS.

use crate::job::{Processor, ProcessorError};
use crate::types::{ArtifactDescriptor, ArtifactKey, ContentHash};
use prism_render_architecture::virtual_geometry::{
    VirtualGeometryAsset, VirtualGeometryAssetError, VIRTUAL_GEOMETRY_ASSET_SCHEMA,
};

/// Stable artifact kind for serialized virtual-geometry assets.
pub const VIRTUAL_GEOMETRY_ARTIFACT_KIND: u128 = 0x5052_4953_4d5f_5647_4153_5345_5431;
/// Artifact schema published by [`CanonicalVirtualGeometryProcessor`].
pub const VIRTUAL_GEOMETRY_ARTIFACT_SCHEMA: u32 = VIRTUAL_GEOMETRY_ASSET_SCHEMA;

/// Canonicalized virtual-geometry artifact and its content identity.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VirtualGeometryArtifact {
    /// Deterministic serialized asset bytes.
    pub bytes: Vec<u8>,
    /// BLAKE3 hash of [`Self::bytes`].
    pub content_hash: ContentHash,
    /// Page size declared by the asset.
    pub page_size: u32,
    /// Serialized format schema.
    pub schema: u32,
}

impl VirtualGeometryArtifact {
    /// Creates the manifest descriptor used by the existing DDC/CAS publisher.
    #[must_use]
    pub fn descriptor(
        &self,
        key: ArtifactKey,
        dependencies: Vec<ArtifactKey>,
    ) -> ArtifactDescriptor {
        ArtifactDescriptor::for_bytes(
            key,
            VIRTUAL_GEOMETRY_ARTIFACT_KIND,
            VIRTUAL_GEOMETRY_ARTIFACT_SCHEMA,
            1024,
            dependencies,
            &self.bytes,
        )
    }
}

/// Validates and deterministically re-encodes one virtual-geometry input.
#[derive(Clone, Copy, Debug, Default)]
pub struct CanonicalVirtualGeometryProcessor;

impl CanonicalVirtualGeometryProcessor {
    /// Canonicalizes one encoded asset for direct use in a build graph.
    pub fn canonicalize(bytes: &[u8]) -> Result<VirtualGeometryArtifact, ProcessorError> {
        let asset = VirtualGeometryAsset::decode(bytes).map_err(asset_error)?;
        let page_size = asset.header.page_size;
        let schema = asset.header.schema_version;
        let bytes = asset.encode().map_err(asset_error)?;
        let content_hash = ContentHash::digest(&bytes);
        Ok(VirtualGeometryArtifact {
            bytes,
            content_hash,
            page_size,
            schema,
        })
    }
}

impl Processor for CanonicalVirtualGeometryProcessor {
    fn process(&self, inputs: &[&[u8]]) -> Result<Vec<u8>, ProcessorError> {
        let [input] = inputs else {
            return Err(ProcessorError::new(
                "virtual-geometry canonicalizer expects exactly one input",
            ));
        };
        Ok(Self::canonicalize(input)?.bytes)
    }
}

fn asset_error(error: VirtualGeometryAssetError) -> ProcessorError {
    ProcessorError::new(format!("invalid virtual-geometry asset: {error}"))
}

#[cfg(test)]
mod tests {
    use prism_render_architecture::gpu_scene::SceneBounds;
    use prism_render_architecture::virtual_geometry::{
        ClusterNode, GeometryPageKey, VirtualGeometryAssetHeader, VirtualGeometryClusterRecord,
        VirtualGeometryCompression, VirtualGeometryPage, VirtualGeometryPageRecord,
    };

    use super::*;

    fn fixture() -> VirtualGeometryAsset {
        let page = GeometryPageKey::new(11, 0);
        let bounds = SceneBounds {
            center: [0.0, 0.0, 0.0],
            radius: 1.0,
            half_extents: [1.0, 1.0, 1.0],
            _padding: 0.0,
        };
        VirtualGeometryAsset {
            header: VirtualGeometryAssetHeader {
                asset_id: 11,
                schema_version: VIRTUAL_GEOMETRY_ASSET_SCHEMA,
                page_size: 1024,
                build_settings_hash: [1; 32],
                content_hash: [2; 32],
                bounds,
                material_slot_count: 1,
                fallback_mesh_hash: [3; 32],
            },
            nodes: vec![ClusterNode::leaf(bounds, 0.0, page)],
            roots: vec![0],
            pages: vec![VirtualGeometryPage {
                record: VirtualGeometryPageRecord {
                    key: page,
                    compression: VirtualGeometryCompression::None,
                    uncompressed_size: 4,
                    priority: 1,
                    cluster_start: 0,
                    cluster_count: 1,
                },
                payload: vec![0, 1, 2, 3],
            }],
            clusters: vec![VirtualGeometryClusterRecord {
                node_index: 0,
                page,
                material_slot: 0,
                vertex_count: 3,
                triangle_count: 1,
                payload_offset: 0,
                payload_size: 4,
                bounds,
                normal_cone: [0.0, 0.0, 1.0, 1.0],
            }],
        }
    }

    #[test]
    fn canonicalization_is_deterministic_and_hashes_output_bytes() {
        let input = fixture().encode().expect("fixture encodes");
        let first = CanonicalVirtualGeometryProcessor::canonicalize(&input).expect("valid input");
        let second = CanonicalVirtualGeometryProcessor::canonicalize(&input).expect("valid input");
        assert_eq!(first, second);
        assert_eq!(first.content_hash, ContentHash::digest(&first.bytes));
        assert_eq!(first.schema, VIRTUAL_GEOMETRY_ARTIFACT_SCHEMA);
        assert_eq!(first.page_size, 1024);
        let descriptor = first.descriptor(ArtifactKey([4; 32]), Vec::new());
        assert_eq!(descriptor.kind, VIRTUAL_GEOMETRY_ARTIFACT_KIND);
        assert_eq!(descriptor.content_hash, first.content_hash);
    }

    #[test]
    fn processor_rejects_wrong_arity_and_corrupt_bytes() {
        let processor = CanonicalVirtualGeometryProcessor;
        assert!(processor.process(&[]).is_err());
        assert!(processor.process(&[b"bad", b"extra"]).is_err());
        assert!(processor.process(&[b"bad"]).is_err());
    }
}
