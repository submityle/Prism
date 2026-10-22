//! Incremental CLST residency and GPU-publication coordination.
//!
//! This module is the device-independent handoff between the deterministic
//! [`VirtualGeometryPageStream`] and a renderer-owned page store.  It keeps the
//! immutable Cluster/page tables generation-bound, while exposing only the
//! pages and page-table rows changed by one reconcile pass.  A backend can
//! therefore upload newly admitted page words and a few table rows without
//! copying the whole asset or rebuilding a full residency snapshot every
//! frame.

use alloc::{sync::Arc, vec::Vec};

use crate::paging::RequestBatch;

use super::{
    asset::{VirtualGeometryAsset, VirtualGeometryAssetError},
    gpu::{
        VirtualGeometryGpuPage, VirtualGeometryGpuPublication, VirtualGeometryGpuPublicationError,
        VirtualGeometryMaterialBinding, VirtualGeometryPhysicalPage,
    },
    stream_adapter::{VirtualGeometryPageStream, VirtualGeometryStreamingError},
    GeometryPageKey,
};

/// One newly admitted CLST page ready for a backend page-storage upload.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VirtualGeometryPageUpload {
    /// Logical page identity.
    pub key: GeometryPageKey,
    /// Physical slot selected by the deterministic page pool.
    pub physical_page: u32,
    /// Byte offset of the slot in the fixed-stride storage buffer.
    pub byte_offset: u32,
    /// Meaningful decoded bytes; the remaining words are alignment padding.
    pub byte_len: u32,
    /// Complete fixed-size slot contents, including zero padding.
    pub words: Vec<u32>,
}

/// One page whose GPU storage slot must be cleared or retired.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VirtualGeometryPageEviction {
    /// Logical page identity.
    pub key: GeometryPageKey,
    /// Former physical slot.
    pub physical_page: u32,
}

/// One logical page-table row changed by this reconcile pass.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VirtualGeometryPageTableUpdate {
    /// Sorted logical page-table index.
    pub page_index: u32,
    /// New generation-bound row.
    pub row: VirtualGeometryGpuPage,
}

/// Incremental output consumed by a renderer after one page reconcile pass.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VirtualGeometryPublicationDelta {
    /// Publication generation all records in this delta belong to.
    pub generation: u64,
    /// Pages that need physical-storage upload before their table rows become
    /// visible to a draw consumer.
    pub streamed_in: Vec<VirtualGeometryPageUpload>,
    /// Pages whose old physical storage must be cleared or retired.
    pub evicted: Vec<VirtualGeometryPageEviction>,
    /// Requests that remained pending because the budget or pool could not
    /// admit them this pass.
    pub deferred: Vec<GeometryPageKey>,
    /// Page-table rows to publish after the corresponding storage operations.
    pub page_table_updates: Vec<VirtualGeometryPageTableUpdate>,
    /// Decode or validation errors observed while attempting admission.
    pub source_errors: Vec<(GeometryPageKey, VirtualGeometryStreamingError)>,
}

/// Errors raised while constructing or incrementally updating a publication.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VirtualGeometryPublicationCoordinatorError {
    /// The source asset failed validation before a stream was created.
    InvalidAsset(VirtualGeometryAssetError),
    /// The page stream could not be constructed.
    Stream(VirtualGeometryStreamingError),
    /// The physical slot byte offset does not fit the GPU publication ABI.
    PhysicalByteOffsetOverflow { physical_page: u32, page_size: u32 },
    /// The stream reported a resident page but its storage slot was absent.
    MissingResidentWords(GeometryPageKey),
    /// The generation-bound publication rejected a residency transition.
    Publication(VirtualGeometryGpuPublicationError),
}

impl core::fmt::Display for VirtualGeometryPublicationCoordinatorError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidAsset(error) => {
                write!(formatter, "invalid virtual-geometry asset: {error}")
            }
            Self::Stream(error) => write!(formatter, "virtual-geometry stream error: {error}"),
            Self::PhysicalByteOffsetOverflow {
                physical_page,
                page_size,
            } => write!(
                formatter,
                "physical page {physical_page} with page size {page_size} overflows byte offset"
            ),
            Self::MissingResidentWords(key) => {
                write!(
                    formatter,
                    "resident page {key:?} has no physical storage words"
                )
            }
            Self::Publication(error) => write!(formatter, "GPU publication error: {error}"),
        }
    }
}

impl core::error::Error for VirtualGeometryPublicationCoordinatorError {}

impl From<VirtualGeometryAssetError> for VirtualGeometryPublicationCoordinatorError {
    fn from(error: VirtualGeometryAssetError) -> Self {
        Self::InvalidAsset(error)
    }
}

impl From<VirtualGeometryStreamingError> for VirtualGeometryPublicationCoordinatorError {
    fn from(error: VirtualGeometryStreamingError) -> Self {
        Self::Stream(error)
    }
}

impl From<VirtualGeometryGpuPublicationError> for VirtualGeometryPublicationCoordinatorError {
    fn from(error: VirtualGeometryGpuPublicationError) -> Self {
        Self::Publication(error)
    }
}

/// Generation-safe coordinator for one immutable CLST asset publication.
pub struct VirtualGeometryPublicationCoordinator {
    asset: Arc<VirtualGeometryAsset>,
    stream: VirtualGeometryPageStream,
    publication: VirtualGeometryGpuPublication,
    page_size: u32,
}

impl VirtualGeometryPublicationCoordinator {
    /// Creates a coordinator with an empty GPU residency set.
    pub fn new(
        asset: Arc<VirtualGeometryAsset>,
        content_revision: [u8; 32],
        generation: u64,
        physical_capacity: u32,
        resident_budget: usize,
        material_bindings: &[VirtualGeometryMaterialBinding],
    ) -> Result<Self, VirtualGeometryPublicationCoordinatorError> {
        asset.validate()?;
        let page_size = asset.header.page_size;
        let stream = VirtualGeometryPageStream::from_shared_asset(
            Arc::clone(&asset),
            physical_capacity,
            resident_budget,
        )?;
        let publication = VirtualGeometryGpuPublication::build(
            &asset,
            content_revision,
            generation,
            &[],
            material_bindings,
        )?;
        Ok(Self {
            asset,
            stream,
            publication,
            page_size,
        })
    }

    /// Immutable source asset backing this publication.
    #[must_use]
    pub fn asset(&self) -> &Arc<VirtualGeometryAsset> {
        &self.asset
    }

    /// Current generation-bound page/Cluster publication.
    #[must_use]
    pub fn publication(&self) -> &VirtualGeometryGpuPublication {
        &self.publication
    }

    /// Reconciles requests and returns only the physical uploads, evictions and
    /// page-table rows changed by this pass.
    pub fn reconcile(
        &mut self,
        requests: &RequestBatch<GeometryPageKey>,
        frame: u64,
    ) -> Result<VirtualGeometryPublicationDelta, VirtualGeometryPublicationCoordinatorError> {
        let report = self.stream.reconcile(requests, frame);
        let mut delta = VirtualGeometryPublicationDelta {
            generation: self.publication.generation(),
            streamed_in: Vec::with_capacity(report.streamed_in.len()),
            evicted: Vec::with_capacity(report.evicted.len()),
            deferred: report.deferred,
            page_table_updates: Vec::with_capacity(report.streamed_in.len() + report.evicted.len()),
            source_errors: self.stream.take_source_errors(),
        };

        for key in report.evicted {
            let Some(physical_page) = self.publication.clear_resident_page(key)? else {
                continue;
            };
            let page_index = self
                .publication
                .page_index(key)
                .expect("clear_resident_page validated the logical page");
            let row = self
                .publication
                .page(page_index)
                .expect("cleared page row remains addressable");
            delta
                .evicted
                .push(VirtualGeometryPageEviction { key, physical_page });
            delta
                .page_table_updates
                .push(VirtualGeometryPageTableUpdate { page_index, row });
        }

        for key in report.streamed_in {
            let Some((physical_page, words)) = self.stream.resident_page_words(key) else {
                return Err(VirtualGeometryPublicationCoordinatorError::MissingResidentWords(key));
            };
            let byte_offset = physical_page.checked_mul(self.page_size).ok_or(
                VirtualGeometryPublicationCoordinatorError::PhysicalByteOffsetOverflow {
                    physical_page,
                    page_size: self.page_size,
                },
            )?;
            let page = self
                .asset
                .pages
                .binary_search_by_key(&key, |page| page.record.key)
                .ok()
                .and_then(|index| self.asset.pages.get(index))
                .expect("stream admission validated the page key");
            let page_index = self
                .publication
                .apply_resident_page(VirtualGeometryPhysicalPage {
                    key,
                    physical_page,
                    byte_offset,
                    byte_len: page.record.uncompressed_size,
                    generation: self.publication.generation(),
                })?;
            let row = self
                .publication
                .page(page_index)
                .expect("admitted page row remains addressable");
            delta.streamed_in.push(VirtualGeometryPageUpload {
                key,
                physical_page,
                byte_offset,
                byte_len: page.record.uncompressed_size,
                words: words.to_vec(),
            });
            delta
                .page_table_updates
                .push(VirtualGeometryPageTableUpdate { page_index, row });
        }

        Ok(delta)
    }
}

#[cfg(test)]
mod tests {
    use alloc::{sync::Arc, vec};

    use crate::{
        gpu_scene::{SceneBounds, SceneMaterialHandle},
        virtual_geometry::{
            cluster_payload::encode_cluster_payload, ClusterNode, GeometryPageKey,
            VirtualGeometryAssetHeader, VirtualGeometryClusterRecord, VirtualGeometryCompression,
            VirtualGeometryPage, VirtualGeometryPageRecord, VIRTUAL_GEOMETRY_ASSET_SCHEMA,
        },
    };

    use super::*;

    fn bounds() -> SceneBounds {
        SceneBounds {
            center: [0.0; 3],
            radius: 1.0,
            half_extents: [1.0; 3],
            _padding: 0.0,
        }
    }

    fn asset() -> Arc<VirtualGeometryAsset> {
        let bounds = bounds();
        let payload = encode_cluster_payload(
            bounds,
            &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            &[0, 1, 2],
        )
        .expect("valid fixture payload");
        let payload_size = payload.len() as u32;
        let key = GeometryPageKey::new(9, 0);
        Arc::new(VirtualGeometryAsset {
            header: VirtualGeometryAssetHeader {
                asset_id: 9,
                schema_version: VIRTUAL_GEOMETRY_ASSET_SCHEMA,
                page_size: 1024,
                build_settings_hash: [1; 32],
                content_hash: [2; 32],
                bounds,
                material_slot_count: 1,
                fallback_mesh_hash: [3; 32],
            },
            nodes: vec![ClusterNode::leaf(bounds, 0.0, key)],
            roots: vec![0],
            pages: vec![VirtualGeometryPage {
                record: VirtualGeometryPageRecord {
                    key,
                    compression: VirtualGeometryCompression::None,
                    uncompressed_size: payload_size,
                    priority: 1,
                    cluster_start: 0,
                    cluster_count: 1,
                },
                payload,
            }],
            clusters: vec![VirtualGeometryClusterRecord {
                node_index: 0,
                page: key,
                material_slot: 0,
                vertex_count: 3,
                triangle_count: 1,
                payload_offset: 0,
                payload_size,
                bounds,
                normal_cone: [0.0, 0.0, 1.0, 1.0],
            }],
        })
    }

    fn two_page_asset() -> Arc<VirtualGeometryAsset> {
        let mut asset = (*asset()).clone();
        let key = GeometryPageKey::new(9, 1);
        asset.pages.push(VirtualGeometryPage {
            record: VirtualGeometryPageRecord {
                key,
                compression: VirtualGeometryCompression::None,
                uncompressed_size: 4,
                priority: 1,
                cluster_start: 1,
                cluster_count: 0,
            },
            payload: vec![5, 6, 7, 8],
        });
        asset.validate().expect("two-page fixture remains valid");
        Arc::new(asset)
    }

    fn coordinator() -> VirtualGeometryPublicationCoordinator {
        VirtualGeometryPublicationCoordinator::new(
            asset(),
            [7; 32],
            4,
            2,
            2,
            &[VirtualGeometryMaterialBinding {
                material_slot: 0,
                material: SceneMaterialHandle::new(2, 3),
            }],
        )
        .expect("valid coordinator")
    }

    #[test]
    fn reconcile_emits_storage_upload_and_one_page_table_row() {
        let key = GeometryPageKey::new(9, 0);
        let mut coordinator = coordinator();
        let mut requests = RequestBatch::new();
        requests.record(key, 10.0);
        let delta = coordinator
            .reconcile(&requests, 1)
            .expect("reconcile succeeds");

        assert_eq!(delta.generation, 4);
        assert_eq!(delta.streamed_in.len(), 1);
        assert!(delta.evicted.is_empty());
        assert_eq!(delta.page_table_updates.len(), 1);
        let upload = &delta.streamed_in[0];
        assert_eq!(upload.key, key);
        assert_eq!(upload.physical_page, 0);
        assert_eq!(upload.byte_offset, 0);
        assert!(upload.byte_len > 0);
        assert_eq!(upload.words[0], u32::from_le_bytes(*b"CLST"));
        assert!(coordinator.publication().pages()[0].is_resident());
    }

    #[test]
    fn publication_rejects_generation_mismatch_before_mutation() {
        let key = GeometryPageKey::new(9, 0);
        let mut coordinator = coordinator();
        let error = coordinator
            .publication
            .apply_resident_page(VirtualGeometryPhysicalPage {
                key,
                physical_page: 0,
                byte_offset: 0,
                byte_len: 4,
                generation: 5,
            })
            .expect_err("stale generation must fail closed");
        assert!(matches!(
            error,
            VirtualGeometryGpuPublicationError::PageGenerationMismatch { .. }
        ));
        assert!(!coordinator.publication().pages()[0].is_resident());
    }

    #[test]
    fn reconcile_evicts_and_reuses_a_physical_slot_without_rebuilding_clusters() {
        let first = GeometryPageKey::new(9, 0);
        let second = GeometryPageKey::new(9, 1);
        let mut coordinator = VirtualGeometryPublicationCoordinator::new(
            two_page_asset(),
            [7; 32],
            4,
            1,
            1,
            &[VirtualGeometryMaterialBinding {
                material_slot: 0,
                material: SceneMaterialHandle::new(2, 3),
            }],
        )
        .expect("valid two-page coordinator");
        let cluster_bytes_before = coordinator.publication().cluster_table_bytes().to_vec();

        let mut first_request = RequestBatch::new();
        first_request.record(first, 1.0);
        let first_delta = coordinator
            .reconcile(&first_request, 1)
            .expect("first page admission");
        assert_eq!(first_delta.streamed_in[0].physical_page, 0);
        assert!(first_delta.evicted.is_empty());

        let mut second_request = RequestBatch::new();
        second_request.record(second, 2.0);
        let second_delta = coordinator
            .reconcile(&second_request, 2)
            .expect("second page replaces first");

        assert_eq!(
            second_delta.evicted,
            vec![VirtualGeometryPageEviction {
                key: first,
                physical_page: 0,
            }]
        );
        assert_eq!(second_delta.streamed_in[0].key, second);
        assert_eq!(second_delta.streamed_in[0].physical_page, 0);
        assert_eq!(second_delta.page_table_updates.len(), 2);
        assert!(!coordinator.publication().pages()[0].is_resident());
        assert!(coordinator.publication().pages()[1].is_resident());
        assert_eq!(coordinator.publication().pages()[1].physical_page, 0);
        assert_eq!(
            coordinator.publication().cluster_table_bytes(),
            cluster_bytes_before.as_slice()
        );
    }
}
