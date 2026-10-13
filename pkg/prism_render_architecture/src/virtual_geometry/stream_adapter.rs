//! Adapter from [`VirtualGeometryAsset`] pages to the shared page streamer.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::paging::{PageSource, PageStorage, PageStreamManager, ReconcileReport, RequestBatch};

use super::asset::{VirtualGeometryAsset, VirtualGeometryAssetError, VirtualGeometryCompression};
use super::GeometryPageKey;

/// A virtual-geometry page source that pads uncompressed asset bytes to words.
struct AssetPageSource {
    pages: BTreeMap<GeometryPageKey, Vec<u8>>,
    page_words: u32,
}

impl PageSource<GeometryPageKey> for AssetPageSource {
    fn load(&mut self, key: GeometryPageKey) -> Vec<u32> {
        let mut words = vec![0; self.page_words as usize];
        let Some(bytes) = self.pages.get(&key) else {
            return words;
        };
        for (index, chunk) in bytes.chunks(4).enumerate() {
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

/// A virtual-geometry stream wired to the shared pool/residency/storage
/// controller. Its `entries()` and `storage()` outputs are the exact inputs
/// consumed by the GPU page-table and page-storage twins.
pub struct VirtualGeometryPageStream {
    manager: PageStreamManager<GeometryPageKey>,
    source: AssetPageSource,
}

impl VirtualGeometryPageStream {
    /// Creates a stream from a validated asset.
    ///
    /// GDeflate pages are rejected until a decompressor backend is explicitly
    /// connected; compressed bytes are never silently treated as geometry words.
    pub fn from_asset(
        asset: VirtualGeometryAsset,
        physical_capacity: u32,
        resident_budget: usize,
    ) -> Result<Self, VirtualGeometryStreamingError> {
        asset.validate()?;
        if asset.header.page_size % 4 != 0 {
            return Err(VirtualGeometryStreamingError::PageSizeNotWordAligned(
                asset.header.page_size,
            ));
        }
        let mut pages = BTreeMap::new();
        for page in asset.pages {
            if page.record.compression != VirtualGeometryCompression::None {
                return Err(VirtualGeometryStreamingError::CompressedPageUnsupported(
                    page.record.key,
                ));
            }
            pages.insert(page.record.key, page.payload);
        }
        let page_words = asset.header.page_size / 4;
        Ok(Self {
            manager: PageStreamManager::new(physical_capacity, page_words, resident_budget),
            source: AssetPageSource { pages, page_words },
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
    /// Physical storage uses 32-bit words, so page size must be word aligned.
    PageSizeNotWordAligned(u32),
    /// A decompression backend is required for this page.
    CompressedPageUnsupported(GeometryPageKey),
}

impl core::fmt::Display for VirtualGeometryStreamingError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidAsset(error) => write!(f, "invalid virtual-geometry asset: {error}"),
            Self::PageSizeNotWordAligned(size) => write!(f, "page size {size} is not word aligned"),
            Self::CompressedPageUnsupported(key) => {
                write!(
                    f,
                    "compressed virtual-geometry page {key:?} needs a decompressor backend"
                )
            }
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
    use alloc::vec;

    use super::*;
    use crate::gpu_scene::SceneBounds;
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
                    uncompressed_size: 3,
                    priority: 1,
                    cluster_start: 0,
                    cluster_count: 1,
                },
                payload: vec![7, 8, 9],
            }],
            clusters: vec![VirtualGeometryClusterRecord {
                node_index: 0,
                page: key,
                material_slot: 0,
                vertex_count: 3,
                triangle_count: 1,
                payload_offset: 0,
                payload_size: 3,
                bounds,
                normal_cone: [0.0, 0.0, 1.0, 1.0],
            }],
        }
    }

    #[test]
    fn adapter_feeds_shared_storage_with_word_padding() {
        let key = GeometryPageKey::new(4, 0);
        let mut stream = VirtualGeometryPageStream::from_asset(asset(), 2, 2).expect("valid asset");
        let mut requests = RequestBatch::new();
        requests.record(key, 5.0);
        let report = stream.reconcile(&requests, 1);
        assert_eq!(report.streamed_in, vec![key]);
        assert_eq!(stream.resident_count(), 1);
        let slot = stream.entries()[0].1;
        let words = stream.storage().slot_words(slot).expect("resident words");
        assert_eq!(words[0], 0x0009_0807);
        assert!(words[1..].iter().all(|word| *word == 0));
    }
}
