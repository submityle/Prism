//! Adapter from [`VirtualGeometryAsset`] pages to the shared page streamer.

use alloc::collections::BTreeMap;
use alloc::sync::Arc;
use alloc::vec::Vec;

use crate::compression::gdeflate::{gdeflate_decompress, GDeflateError};
use crate::paging::{PageSource, PageStorage, PageStreamManager, ReconcileReport, RequestBatch};

use super::asset::{
    VirtualGeometryAsset, VirtualGeometryAssetError, VirtualGeometryCompression,
    VirtualGeometryPage,
};
use super::cluster_payload::{decode_cluster_payload, ClusterPayloadError};
use super::GeometryPageKey;

/// A virtual-geometry page source that retains stored pages and decodes only
/// when the residency manager admits a page.
struct AssetPageSource {
    asset: Arc<VirtualGeometryAsset>,
    page_words: u32,
    errors: BTreeMap<GeometryPageKey, VirtualGeometryStreamingError>,
}

impl PageSource<GeometryPageKey> for AssetPageSource {
    fn load(&mut self, key: GeometryPageKey) -> Vec<u32> {
        let Ok(page_index) = self
            .asset
            .pages
            .binary_search_by_key(&key, |page| page.record.key)
        else {
            self.errors
                .insert(key, VirtualGeometryStreamingError::MissingPage(key));
            return Vec::new();
        };
        let page = &self.asset.pages[page_index];
        let decoded = match decode_page_payload(page) {
            Ok(decoded) => decoded,
            Err(error) => {
                self.errors.insert(key, error);
                return Vec::new();
            }
        };
        let actual = u32::try_from(decoded.len()).unwrap_or(u32::MAX);
        if actual != page.record.uncompressed_size {
            self.errors.insert(
                key,
                VirtualGeometryStreamingError::UncompressedSizeMismatch {
                    key,
                    expected: page.record.uncompressed_size,
                    actual,
                },
            );
            return Vec::new();
        }

        let cluster_start = page.record.cluster_start as usize;
        let Some(cluster_end) = cluster_start.checked_add(page.record.cluster_count as usize)
        else {
            self.errors.insert(
                key,
                VirtualGeometryStreamingError::InvalidAsset(
                    VirtualGeometryAssetError::InvalidValue("page cluster range"),
                ),
            );
            return Vec::new();
        };
        let Some(clusters) = self.asset.clusters.get(cluster_start..cluster_end) else {
            self.errors.insert(
                key,
                VirtualGeometryStreamingError::InvalidAsset(
                    VirtualGeometryAssetError::InvalidValue("page cluster range"),
                ),
            );
            return Vec::new();
        };
        for (offset, cluster) in clusters.iter().enumerate() {
            let cluster_index = page.record.cluster_start + offset as u32;
            let payload_start = cluster.payload_offset as usize;
            let Some(payload_end) = payload_start.checked_add(cluster.payload_size as usize) else {
                self.errors.insert(
                    key,
                    VirtualGeometryStreamingError::InvalidClusterPayload {
                        key,
                        cluster: cluster_index,
                        error: ClusterPayloadError::Truncated,
                    },
                );
                return Vec::new();
            };
            let Some(payload) = decoded.get(payload_start..payload_end) else {
                self.errors.insert(
                    key,
                    VirtualGeometryStreamingError::InvalidClusterPayload {
                        key,
                        cluster: cluster_index,
                        error: ClusterPayloadError::Truncated,
                    },
                );
                return Vec::new();
            };
            let decoded_cluster = match decode_cluster_payload(cluster.bounds, payload) {
                Ok(decoded) => decoded,
                Err(error) => {
                    self.errors.insert(
                        key,
                        VirtualGeometryStreamingError::InvalidClusterPayload {
                            key,
                            cluster: cluster_index,
                            error,
                        },
                    );
                    return Vec::new();
                }
            };
            if decoded_cluster.positions.len() != cluster.vertex_count as usize
                || decoded_cluster.indices.len() / 3 != cluster.triangle_count as usize
            {
                self.errors.insert(
                    key,
                    VirtualGeometryStreamingError::InvalidClusterPayload {
                        key,
                        cluster: cluster_index,
                        error: ClusterPayloadError::InvalidCounts,
                    },
                );
                return Vec::new();
            }
        }

        let mut words = vec![0; self.page_words as usize];
        for (index, chunk) in decoded.chunks(4).enumerate() {
            if index >= words.len() {
                break;
            }
            let mut word = [0; 4];
            word[..chunk.len()].copy_from_slice(chunk);
            words[index] = u32::from_le_bytes(word);
        }
        words
    }
}

fn decode_page_payload(
    page: &VirtualGeometryPage,
) -> Result<Vec<u8>, VirtualGeometryStreamingError> {
    match page.record.compression {
        VirtualGeometryCompression::None => Ok(page.payload.clone()),
        VirtualGeometryCompression::GDeflate => {
            gdeflate_decompress(&page.payload).map_err(VirtualGeometryStreamingError::Decompression)
        }
    }
}

/// A virtual-geometry stream wired to the shared pool/residency/storage
/// controller. Its `entries()` and `storage()` outputs are the exact inputs
/// consumed by the GPU page-table and page-storage twins.
pub struct VirtualGeometryPageStream {
    manager: PageStreamManager<GeometryPageKey>,
    source: AssetPageSource,
}

impl VirtualGeometryPageStream {
    /// Creates a stream from an owned asset. Stored page payloads remain in
    /// the asset allocation and are decoded only when admitted.
    pub fn from_asset(
        asset: VirtualGeometryAsset,
        physical_capacity: u32,
        resident_budget: usize,
    ) -> Result<Self, VirtualGeometryStreamingError> {
        Self::from_shared_asset(Arc::new(asset), physical_capacity, resident_budget)
    }

    /// Creates a stream sharing the immutable asset allocation with its
    /// publisher. This avoids duplicating every stored page and descriptor
    /// while allowing an in-flight stream to retain the exact source revision.
    pub fn from_shared_asset(
        asset: Arc<VirtualGeometryAsset>,
        physical_capacity: u32,
        resident_budget: usize,
    ) -> Result<Self, VirtualGeometryStreamingError> {
        asset.validate()?;
        if asset.header.page_size % 4 != 0 {
            return Err(VirtualGeometryStreamingError::PageSizeNotWordAligned(
                asset.header.page_size,
            ));
        }
        let page_words = asset.header.page_size / 4;
        Ok(Self {
            manager: PageStreamManager::new(physical_capacity, page_words, resident_budget),
            source: AssetPageSource {
                asset,
                page_words,
                errors: BTreeMap::new(),
            },
        })
    }

    /// Reconciles requests and admits page payloads into physical storage.
    pub fn reconcile(
        &mut self,
        requests: &RequestBatch<GeometryPageKey>,
        frame: u64,
    ) -> ReconcileReport<GeometryPageKey> {
        self.manager.reconcile(requests, frame, &mut self.source)
    }

    /// Returns and clears page decode errors produced by the most recent
    /// reconciliation. A failed page remains requested and is never admitted;
    /// callers can report the exact asset/page failure and choose a fallback.
    pub fn take_source_errors(&mut self) -> Vec<(GeometryPageKey, VirtualGeometryStreamingError)> {
        core::mem::take(&mut self.source.errors)
            .into_iter()
            .collect()
    }

    /// Exports the sorted page-to-slot table for GPU page-table upload.
    #[must_use]
    pub fn entries(&self) -> Vec<(GeometryPageKey, u32)> {
        self.manager.entries()
    }

    /// Exports the physical page storage for GPU page-storage upload/parity.
    #[must_use]
    pub const fn storage(&self) -> &PageStorage {
        self.manager.storage()
    }

    /// Number of resident pages.
    #[must_use]
    pub fn resident_count(&self) -> usize {
        self.manager.resident_count()
    }
}

/// Failure while adapting a virtual-geometry asset to the page-stream ABI.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VirtualGeometryStreamingError {
    /// Asset validation failed.
    InvalidAsset(VirtualGeometryAssetError),
    /// A request named a page absent from the registered asset.
    MissingPage(GeometryPageKey),
    /// Physical storage uses 32-bit words, so page size must be word aligned.
    PageSizeNotWordAligned(u32),
    /// GDeflate decode failed before the page entered physical storage.
    Decompression(GDeflateError),
    /// A decompressed cluster slice failed the versioned CLST codec or descriptor checks.
    InvalidClusterPayload {
        /// Page identity.
        key: GeometryPageKey,
        /// Global cluster descriptor index.
        cluster: u32,
        /// Validation failure.
        error: ClusterPayloadError,
    },
    /// Decoded payload length differs from the page table metadata.
    UncompressedSizeMismatch {
        /// Page identity.
        key: GeometryPageKey,
        /// Declared decoded length.
        expected: u32,
        /// Observed decoded length.
        actual: u32,
    },
}

impl core::fmt::Display for VirtualGeometryStreamingError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidAsset(error) => write!(f, "invalid virtual-geometry asset: {error}"),
            Self::MissingPage(key) => write!(f, "virtual-geometry page {key:?} is missing"),
            Self::PageSizeNotWordAligned(size) => write!(f, "page size {size} is not word aligned"),
            Self::Decompression(error) => {
                write!(f, "virtual-geometry page decompression failed: {error:?}")
            }
            Self::InvalidClusterPayload {
                key,
                cluster,
                error,
            } => write!(
                f,
                "virtual-geometry page {key:?} cluster {cluster} has invalid payload: {error}"
            ),
            Self::UncompressedSizeMismatch {
                key,
                expected,
                actual,
            } => write!(
                f,
                "virtual-geometry page {key:?} decoded to {actual} bytes, expected {expected}"
            ),
        }
    }
}

impl From<VirtualGeometryAssetError> for VirtualGeometryStreamingError {
    fn from(value: VirtualGeometryAssetError) -> Self {
        Self::InvalidAsset(value)
    }
}

#[cfg(test)]
mod tests {
    use alloc::sync::Arc;
    use alloc::vec;

    use super::*;
    use crate::gpu_scene::SceneBounds;
    use crate::virtual_geometry::cluster_payload::encode_cluster_payload;
    use crate::virtual_geometry::{
        ClusterNode, VirtualGeometryAssetHeader, VirtualGeometryClusterRecord, VirtualGeometryPage,
        VirtualGeometryPageRecord, VIRTUAL_GEOMETRY_ASSET_SCHEMA,
    };

    fn asset() -> VirtualGeometryAsset {
        let key = GeometryPageKey::new(4, 0);
        let bounds = SceneBounds {
            center: [0.0; 3],
            radius: 1.0,
            half_extents: [1.0; 3],
            _padding: 0.0,
        };
        let payload = encode_cluster_payload(
            bounds,
            &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            &[0, 1, 2],
        )
        .expect("valid fixture payload");
        let payload_size = payload.len() as u32;
        VirtualGeometryAsset {
            header: VirtualGeometryAssetHeader {
                asset_id: 4,
                schema_version: VIRTUAL_GEOMETRY_ASSET_SCHEMA,
                page_size: 1024,
                build_settings_hash: [1; 32],
                content_hash: [2; 32],
                bounds,
                material_slot_count: 1,
                fallback_mesh_hash: [0; 32],
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
        }
    }

    #[test]
    fn adapter_feeds_shared_storage_with_word_padding() {
        let key = GeometryPageKey::new(4, 0);
        let shared = Arc::new(asset());
        let mut stream = VirtualGeometryPageStream::from_shared_asset(Arc::clone(&shared), 2, 2)
            .expect("valid shared asset");
        assert!(Arc::ptr_eq(&shared, &stream.source.asset));
        let mut requests = RequestBatch::new();
        requests.record(key, 5.0);
        let report = stream.reconcile(&requests, 1);
        assert_eq!(report.streamed_in, vec![key]);
        assert_eq!(stream.resident_count(), 1);
        let slot = stream.entries()[0].1;
        let words = stream.storage().slot_words(slot).expect("resident words");
        assert_eq!(words[0], u32::from_le_bytes(*b"CLST"));
        assert!(words[1..].iter().any(|word| *word != 0));
    }

    #[test]
    fn adapter_decodes_gdeflate_before_word_storage() {
        let mut compressed = asset();
        let raw = compressed.pages[0].payload.clone();
        compressed.pages[0].record.compression = VirtualGeometryCompression::GDeflate;
        compressed.pages[0].record.uncompressed_size = raw.len() as u32;
        compressed.pages[0].payload = crate::compression::gdeflate::gdeflate_compress(&raw);

        let key = GeometryPageKey::new(4, 0);
        let mut stream = VirtualGeometryPageStream::from_asset(compressed, 2, 2)
            .expect("compressed page should be decoded");
        let mut requests = RequestBatch::new();
        requests.record(key, 5.0);
        stream.reconcile(&requests, 1);
        let slot = stream.entries()[0].1;
        let words = stream.storage().slot_words(slot).expect("resident words");
        assert_eq!(words[0], u32::from_le_bytes(*b"CLST"));
    }

    #[test]
    fn adapter_decodes_and_validates_only_when_page_is_admitted() {
        let key = GeometryPageKey::new(4, 0);
        let mut corrupted = asset();
        corrupted.pages[0].payload[0] ^= 0xff;
        let mut stream = VirtualGeometryPageStream::from_asset(corrupted, 2, 2)
            .expect("page decode is deferred until admission");
        assert_eq!(stream.resident_count(), 0);
        assert!(stream.take_source_errors().is_empty());

        let mut requests = RequestBatch::new();
        requests.record(key, 5.0);
        let report = stream.reconcile(&requests, 1);

        assert!(report.streamed_in.is_empty());
        assert_eq!(report.deferred, vec![key]);
        assert_eq!(stream.resident_count(), 0);
        assert_eq!(
            stream.take_source_errors(),
            vec![(
                key,
                VirtualGeometryStreamingError::InvalidClusterPayload {
                    key,
                    cluster: 0,
                    error: ClusterPayloadError::InvalidHeader,
                }
            )]
        );
    }
}
