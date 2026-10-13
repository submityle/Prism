//! Runtime page store backed by a validated [`VirtualGeometryAsset`].

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use super::asset::{VirtualGeometryAsset, VirtualGeometryAssetError, VirtualGeometryPage};
use super::page_table::{GeometryPageTable, PageResidency};
use super::GeometryPageKey;
use crate::compression::gdeflate::{gdeflate_decompress, GDeflateError};

/// A validated virtual-geometry page store with deterministic residency state.
#[derive(Clone, Debug)]
pub struct VirtualGeometryPageStore {
    pages: BTreeMap<GeometryPageKey, VirtualGeometryPage>,
    decoded_pages: BTreeMap<GeometryPageKey, Vec<u8>>,
    residency: GeometryPageTable,
}

/// Failure while making a requested virtual-geometry page resident.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VirtualGeometryPageStoreError {
    /// The stored GDeflate payload failed integrity or decode validation.
    Decompression(GDeflateError),
    /// Decoded bytes disagree with the page metadata.
    UncompressedSizeMismatch { expected: u32, actual: u32 },
}

impl VirtualGeometryPageStore {
    /// Loads all page payloads from a validated asset without marking them resident.
    pub fn from_asset(asset: VirtualGeometryAsset) -> Result<Self, VirtualGeometryAssetError> {
        asset.validate()?;
        let pages = asset
            .pages
            .into_iter()
            .map(|page| (page.record.key, page))
            .collect();
        Ok(Self {
            pages,
            decoded_pages: BTreeMap::new(),
            residency: GeometryPageTable::new(),
        })
    }

    /// Requests a page for a frame at the supplied screen priority.
    pub fn request(&mut self, key: GeometryPageKey, priority: f32, frame: u64) -> bool {
        if !self.pages.contains_key(&key) {
            return false;
        }
        self.residency.request(key, priority, frame);
        true
    }

    /// Completes a backend upload for a requested page.
    pub fn mark_resident(
        &mut self,
        key: GeometryPageKey,
    ) -> Result<bool, VirtualGeometryPageStoreError> {
        if !self.pages.contains_key(&key)
            || self.residency.residency(key) == PageResidency::Unloaded
        {
            return Ok(false);
        }
        let page = self.pages.get(&key).expect("page existence checked");
        let decoded = match page.record.compression {
            super::asset::VirtualGeometryCompression::None => page.payload.clone(),
            super::asset::VirtualGeometryCompression::GDeflate => {
                gdeflate_decompress(&page.payload)
                    .map_err(VirtualGeometryPageStoreError::Decompression)?
            }
        };
        let actual = u32::try_from(decoded.len()).unwrap_or(u32::MAX);
        if actual != page.record.uncompressed_size {
            return Err(VirtualGeometryPageStoreError::UncompressedSizeMismatch {
                expected: page.record.uncompressed_size,
                actual,
            });
        }
        self.decoded_pages.insert(key, decoded);
        self.residency.mark_resident(key);
        Ok(true)
    }

    /// Returns the stored bytes only after the page is resident.
    #[must_use]
    pub fn resident_payload(&self, key: GeometryPageKey) -> Option<&[u8]> {
        (self.residency.residency(key) == PageResidency::Resident)
            .then(|| self.decoded_pages.get(&key).map(|page| page.as_slice()))
            .flatten()
    }

    /// Returns pages that are requested but not yet resident.
    #[must_use]
    pub fn pending_requests(&self) -> Vec<(GeometryPageKey, f32)> {
        self.residency.pending_requests()
    }

    /// Selects and evicts pages until the resident budget is satisfied.
    pub fn evict_to_budget(
        &mut self,
        resident_budget: usize,
        protect_frame: u64,
    ) -> Vec<GeometryPageKey> {
        let victims = self
            .residency
            .select_evictions(resident_budget, protect_frame);
        for key in &victims {
            self.residency.evict(*key);
            self.decoded_pages.remove(key);
        }
        victims
    }

    /// Current residency state for a page.
    #[must_use]
    pub fn residency(&self, key: GeometryPageKey) -> PageResidency {
        self.residency.residency(key)
    }

    /// Number of pages physically resident according to the CPU contract.
    #[must_use]
    pub fn resident_count(&self) -> usize {
        self.residency.resident_count()
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;
    use crate::gpu_scene::SceneBounds;
    use crate::virtual_geometry::{
        ClusterNode, VirtualGeometryAssetHeader, VirtualGeometryClusterRecord,
        VirtualGeometryCompression, VirtualGeometryPageRecord, DEFAULT_VIRTUAL_GEOMETRY_PAGE_SIZE,
        VIRTUAL_GEOMETRY_ASSET_SCHEMA,
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
                page_size: DEFAULT_VIRTUAL_GEOMETRY_PAGE_SIZE,
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
    fn payload_is_hidden_until_request_and_upload_complete() {
        let key = GeometryPageKey::new(4, 0);
        let mut store = VirtualGeometryPageStore::from_asset(asset()).expect("valid asset");
        assert_eq!(store.resident_payload(key), None);
        assert!(store.request(key, 2.0, 10));
        assert_eq!(store.resident_payload(key), None);
        assert!(store.mark_resident(key).unwrap());
        assert_eq!(store.resident_payload(key), Some(&[7, 8, 9][..]));
    }

    #[test]
    fn unknown_pages_are_rejected_and_resident_pages_can_be_evicted() {
        let key = GeometryPageKey::new(4, 0);
        let unknown = GeometryPageKey::new(4, 99);
        let mut store = VirtualGeometryPageStore::from_asset(asset()).expect("valid asset");
        assert!(!store.request(unknown, 1.0, 1));
        store.request(key, 1.0, 1);
        store.mark_resident(key).unwrap();
        assert_eq!(store.evict_to_budget(0, 2), vec![key]);
        assert_eq!(store.resident_count(), 0);
    }

    #[test]
    fn compressed_page_is_decoded_before_becoming_resident() {
        let mut asset = asset();
        let raw = asset.pages[0].payload.clone();
        asset.pages[0].record.compression = VirtualGeometryCompression::GDeflate;
        asset.pages[0].record.uncompressed_size = raw.len() as u32;
        asset.pages[0].payload = crate::compression::gdeflate::gdeflate_compress(&raw);
        let key = GeometryPageKey::new(4, 0);
        let mut store = VirtualGeometryPageStore::from_asset(asset).expect("valid asset");
        store.request(key, 1.0, 1);
        assert!(store.mark_resident(key).unwrap());
        assert_eq!(store.resident_payload(key), Some(&raw[..]));
    }
}
