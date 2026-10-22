//! GPU-facing CLST publication and per-view draw-list contracts.
//!
//! The CPU virtual-geometry planner deals in asset pages and hierarchy nodes.
//! This module is the boundary where those logical identities become records
//! that a renderer can upload and a shader can consume.  A publication is
//! immutable: replacing an asset creates a new generation, and every page,
//! Cluster and draw item carries enough identity to reject stale work.

use alloc::{collections::BTreeMap, vec::Vec};
use core::fmt;

use bytemuck::{Pod, Zeroable};

use crate::gpu_scene::{SceneHandle, SceneMaterialHandle};

use super::{
    asset::{VirtualGeometryAsset, VirtualGeometryAssetError, VirtualGeometryClusterRecord},
    hierarchy::CutCluster,
    GeometryPageKey,
};

/// Version of the storage-buffer records defined by this module.
pub const VIRTUAL_GEOMETRY_GPU_ABI_VERSION: u32 = 1;
/// Sentinel used for a logical page that has no physical allocation yet.
pub const NON_RESIDENT_PHYSICAL_PAGE: u32 = u32::MAX;

/// One resident physical page supplied by the page allocator.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VirtualGeometryPhysicalPage {
    /// Logical page addressed by this allocation.
    pub key: GeometryPageKey,
    /// Physical slot in the page pool.
    pub physical_page: u32,
    /// Byte offset inside the backing storage buffer.
    pub byte_offset: u32,
    /// Number of valid bytes written to the backing storage buffer.
    pub byte_len: u32,
    /// Runtime publication generation that owns the allocation.
    pub generation: u64,
}

/// Explicit mapping from an authored material slot to the global material ABI.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VirtualGeometryMaterialBinding {
    /// Authored slot stored in the CLST cluster descriptor.
    pub material_slot: u16,
    /// Global material slot consumed by shading.
    pub material: SceneMaterialHandle,
}

/// GPU page-table row.  The page table is indexed by the asset's sorted page
/// order, not by an allocator slot, so logical identity remains stable while
/// physical residency changes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Pod, Zeroable)]
pub struct VirtualGeometryGpuPage {
    /// Low/high words of the publication generation.
    pub generation: [u32; 2],
    /// Physical page slot, or [`NON_RESIDENT_PHYSICAL_PAGE`].
    pub physical_page: u32,
    /// Flags; bit zero means resident.
    pub flags: u32,
    /// Byte offset in the physical page storage buffer.
    pub byte_offset: u32,
    /// Valid byte count in the physical page storage buffer.
    pub byte_len: u32,
    /// First Cluster descriptor backed by this logical page.
    pub cluster_start: u32,
    /// Number of Cluster descriptors backed by this logical page.
    pub cluster_count: u32,
}

impl VirtualGeometryGpuPage {
    /// Returns whether the page can be addressed by a draw item.
    #[must_use]
    pub const fn is_resident(self) -> bool {
        self.physical_page != NON_RESIDENT_PHYSICAL_PAGE && (self.flags & 1) != 0
    }
}

/// GPU Cluster descriptor.  The payload itself remains in page storage and is
/// addressed by `page_index + payload_offset`; no Cluster receives a separate
/// allocation or draw call.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Pod, Zeroable)]
pub struct VirtualGeometryGpuCluster {
    /// Hierarchy node identity, useful for visibility/debug validation.
    pub node_index: u32,
    /// Index into the logical page table.
    pub page_index: u32,
    /// Byte offset and length inside the resident page.
    pub payload_offset: u32,
    pub payload_size: u32,
    /// Global material identity, including its generation.
    pub material_index: u32,
    pub material_generation: u32,
    /// CLST decode bounds.
    pub vertex_count: u32,
    pub triangle_count: u32,
}

/// One typed Cluster work item emitted for a view.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Pod, Zeroable)]
pub struct VirtualGeometryGpuDrawItem {
    /// Asset identity in the virtual-geometry registry.
    pub asset_id: u32,
    /// Cluster descriptor index.
    pub cluster_index: u32,
    /// Scene instance identity.
    pub scene_index: u32,
    pub scene_generation: u32,
    /// Global material identity copied from the Cluster descriptor.
    pub material_index: u32,
    pub material_generation: u32,
    /// Logical page and publication generation proof.
    pub page_index: u32,
    pub publication_generation: [u32; 2],
}

/// All records needed to publish one validated CLST asset to a GPU renderer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VirtualGeometryGpuPublication {
    asset_id: u32,
    content_revision: [u8; 32],
    generation: u64,
    page_keys: Vec<GeometryPageKey>,
    pages: Vec<VirtualGeometryGpuPage>,
    clusters: Vec<VirtualGeometryGpuCluster>,
    cluster_by_node: BTreeMap<u32, u32>,
}

impl VirtualGeometryGpuPublication {
    /// Builds a generation-bound GPU publication from validated asset data.
    pub fn build(
        asset: &VirtualGeometryAsset,
        content_revision: [u8; 32],
        generation: u64,
        resident_pages: &[VirtualGeometryPhysicalPage],
        material_bindings: &[VirtualGeometryMaterialBinding],
    ) -> Result<Self, VirtualGeometryGpuPublicationError> {
        if generation == 0 {
            return Err(VirtualGeometryGpuPublicationError::InvalidGeneration);
        }
        asset
            .validate()
            .map_err(VirtualGeometryGpuPublicationError::InvalidAsset)?;

        let page_indices: BTreeMap<_, _> = asset
            .pages
            .iter()
            .enumerate()
            .map(|(index, page)| (page.record.key, index as u32))
            .collect();
        let bindings: BTreeMap<u16, SceneMaterialHandle> = material_bindings
            .iter()
            .map(|binding| (binding.material_slot, binding.material))
            .collect();
        if bindings.len() != material_bindings.len() {
            return Err(VirtualGeometryGpuPublicationError::DuplicateMaterialSlot);
        }

        let page_keys = asset
            .pages
            .iter()
            .map(|page| page.record.key)
            .collect::<Vec<_>>();
        let mut physical_pages = BTreeMap::new();
        let mut pages = Vec::with_capacity(asset.pages.len());
        for page in &asset.pages {
            pages.push(VirtualGeometryGpuPage {
                generation: split_generation(generation),
                physical_page: NON_RESIDENT_PHYSICAL_PAGE,
                flags: 0,
                byte_offset: 0,
                byte_len: 0,
                cluster_start: page.record.cluster_start,
                cluster_count: u32::from(page.record.cluster_count),
            });
        }

        for resident in resident_pages {
            if resident.generation != generation {
                return Err(VirtualGeometryGpuPublicationError::PageGenerationMismatch {
                    key: resident.key,
                    expected: generation,
                    actual: resident.generation,
                });
            }
            let Some(page_index) = page_indices.get(&resident.key).copied() else {
                return Err(VirtualGeometryGpuPublicationError::UnknownPage(
                    resident.key,
                ));
            };
            if resident.physical_page == NON_RESIDENT_PHYSICAL_PAGE {
                return Err(VirtualGeometryGpuPublicationError::InvalidPhysicalPage(
                    resident.key,
                ));
            }
            if physical_pages
                .insert(resident.physical_page, resident.key)
                .is_some()
            {
                return Err(VirtualGeometryGpuPublicationError::DuplicatePhysicalPage(
                    resident.physical_page,
                ));
            }
            let entry = &mut pages[page_index as usize];
            if entry.is_resident() {
                return Err(VirtualGeometryGpuPublicationError::DuplicateResidentPage(
                    resident.key,
                ));
            }
            if resident.byte_len == 0 {
                return Err(VirtualGeometryGpuPublicationError::EmptyResidentPage(
                    resident.key,
                ));
            }
            entry.physical_page = resident.physical_page;
            entry.flags = 1;
            entry.byte_offset = resident.byte_offset;
            entry.byte_len = resident.byte_len;
        }

        let mut clusters = Vec::with_capacity(asset.clusters.len());
        let mut cluster_by_node = BTreeMap::new();
        for (cluster_index, cluster) in asset.clusters.iter().enumerate() {
            let Some(page_index) = page_indices.get(&cluster.page).copied() else {
                return Err(VirtualGeometryGpuPublicationError::UnknownPage(
                    cluster.page,
                ));
            };
            let Some(material) = bindings.get(&cluster.material_slot).copied() else {
                return Err(VirtualGeometryGpuPublicationError::MaterialSlotUnbound {
                    slot: cluster.material_slot,
                });
            };
            if cluster_by_node
                .insert(cluster.node_index, cluster_index as u32)
                .is_some()
            {
                return Err(VirtualGeometryGpuPublicationError::DuplicateClusterNode(
                    cluster.node_index,
                ));
            }
            clusters.push(encode_cluster(cluster, page_index, material));
        }

        Ok(Self {
            asset_id: asset.header.asset_id,
            content_revision,
            generation,
            page_keys,
            pages,
            clusters,
            cluster_by_node,
        })
    }

    /// Stable CLST asset identity.
    #[must_use]
    pub const fn asset_id(&self) -> u32 {
        self.asset_id
    }

    /// Trusted content revision supplied by the asset publisher.
    #[must_use]
    pub const fn content_revision(&self) -> [u8; 32] {
        self.content_revision
    }

    /// Runtime publication generation.
    #[must_use]
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    /// Logical page-table rows in asset order.
    #[must_use]
    pub fn pages(&self) -> &[VirtualGeometryGpuPage] {
        &self.pages
    }

    /// Logical page keys in the same order as [`Self::pages`].
    #[must_use]
    pub fn page_keys(&self) -> &[GeometryPageKey] {
        &self.page_keys
    }

    /// Cluster descriptors in the asset's canonical descriptor order.
    #[must_use]
    pub fn clusters(&self) -> &[VirtualGeometryGpuCluster] {
        &self.clusters
    }

    /// Bytes suitable for one storage-buffer upload.
    #[must_use]
    pub fn page_table_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.pages)
    }

    /// Bytes suitable for one storage-buffer upload.
    #[must_use]
    pub fn cluster_table_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.clusters)
    }

    /// Resolves a cut into a typed draw list for one scene instance.
    pub fn draw_list(
        &self,
        scene: SceneHandle,
        cut: &[CutCluster],
    ) -> Result<VirtualGeometryViewDrawList, VirtualGeometryGpuPublicationError> {
        let mut items = Vec::with_capacity(cut.len());
        let mut seen = BTreeMap::new();
        for selected in cut {
            if selected.page.asset != self.asset_id {
                return Err(VirtualGeometryGpuPublicationError::CutAssetMismatch {
                    expected: self.asset_id,
                    actual: selected.page.asset,
                });
            }
            let Some(&cluster_index) = self.cluster_by_node.get(&selected.node) else {
                return Err(VirtualGeometryGpuPublicationError::CutNodeMissing {
                    node: selected.node,
                });
            };
            let cluster = self.clusters[cluster_index as usize];
            let Some(&page_key) = self.page_keys.get(cluster.page_index as usize) else {
                return Err(VirtualGeometryGpuPublicationError::CutPageMismatch {
                    node: selected.node,
                    expected: cluster.page_index,
                    actual: u32::MAX,
                });
            };
            if page_key != selected.page {
                return Err(
                    VirtualGeometryGpuPublicationError::CutPageIdentityMismatch {
                        node: selected.node,
                        expected: page_key,
                        actual: selected.page,
                    },
                );
            }
            let Some(page) = self.pages.get(cluster.page_index as usize).copied() else {
                return Err(VirtualGeometryGpuPublicationError::PageNotResident {
                    key: selected.page,
                });
            };
            if !page.is_resident() {
                return Err(VirtualGeometryGpuPublicationError::PageNotResident {
                    key: selected.page,
                });
            }
            let page_index = cluster.page_index;
            if seen.insert(cluster_index, ()).is_some() {
                return Err(VirtualGeometryGpuPublicationError::DuplicateCutCluster {
                    node: selected.node,
                });
            }
            items.push(VirtualGeometryGpuDrawItem {
                asset_id: self.asset_id,
                cluster_index,
                scene_index: scene.index,
                scene_generation: scene.generation,
                material_index: cluster.material_index,
                material_generation: cluster.material_generation,
                page_index,
                publication_generation: split_generation(self.generation),
            });
        }
        Ok(VirtualGeometryViewDrawList {
            asset_id: self.asset_id,
            generation: self.generation,
            items,
        })
    }
}

/// Per-view, generation-bound CLST work list.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VirtualGeometryViewDrawList {
    asset_id: u32,
    generation: u64,
    items: Vec<VirtualGeometryGpuDrawItem>,
}

impl VirtualGeometryViewDrawList {
    /// Asset identity represented by this list.
    #[must_use]
    pub const fn asset_id(&self) -> u32 {
        self.asset_id
    }

    /// Publication generation represented by this list.
    #[must_use]
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    /// Ordered draw items; ordering is the supplied cut order.
    #[must_use]
    pub fn items(&self) -> &[VirtualGeometryGpuDrawItem] {
        &self.items
    }

    /// Bytes suitable for an indirect-work storage buffer.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.items)
    }
}

/// Why a CLST GPU publication could not be built or consumed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VirtualGeometryGpuPublicationError {
    /// The source asset failed its cross-table validation.
    InvalidAsset(VirtualGeometryAssetError),
    /// Runtime generation zero is reserved for an invalid handle.
    InvalidGeneration,
    /// A resident page refers to no logical page in the asset.
    UnknownPage(GeometryPageKey),
    /// The resident page belongs to another publication generation.
    PageGenerationMismatch {
        key: GeometryPageKey,
        expected: u64,
        actual: u64,
    },
    /// Two physical publication inputs target one logical page.
    DuplicateResidentPage(GeometryPageKey),
    /// A page cannot be published with no valid bytes.
    EmptyResidentPage(GeometryPageKey),
    /// The physical-page sentinel cannot be used for a resident allocation.
    InvalidPhysicalPage(GeometryPageKey),
    /// Two logical pages alias the same physical allocation.
    DuplicatePhysicalPage(u32),
    /// Two bindings claim the same authored slot.
    DuplicateMaterialSlot,
    /// A cluster's authored material slot has no global mapping.
    MaterialSlotUnbound { slot: u16 },
    /// A node maps to more than one cluster descriptor.
    DuplicateClusterNode(u32),
    /// A cut names another asset.
    CutAssetMismatch { expected: u32, actual: u32 },
    /// A cut node has no CLST descriptor.
    CutNodeMissing { node: u32 },
    /// A selected page is not resident in this publication.
    PageNotResident { key: GeometryPageKey },
    /// A cut page does not match the descriptor's page table index.
    CutPageMismatch {
        node: u32,
        expected: u32,
        actual: u32,
    },
    /// The selected cut uses a different logical page key than the descriptor.
    CutPageIdentityMismatch {
        node: u32,
        expected: GeometryPageKey,
        actual: GeometryPageKey,
    },
    /// A cut repeats one Cluster descriptor.
    DuplicateCutCluster { node: u32 },
}

impl fmt::Display for VirtualGeometryGpuPublicationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidAsset(error) => {
                write!(formatter, "invalid virtual geometry asset: {error}")
            }
            Self::InvalidGeneration => formatter.write_str("virtual geometry generation is zero"),
            Self::UnknownPage(key) => write!(formatter, "unknown virtual geometry page {key:?}"),
            Self::PageGenerationMismatch {
                key,
                expected,
                actual,
            } => write!(
                formatter,
                "page {key:?} belongs to generation {actual}, expected {expected}"
            ),
            Self::DuplicateResidentPage(key) => {
                write!(formatter, "page {key:?} is published twice")
            }
            Self::EmptyResidentPage(key) => write!(formatter, "page {key:?} has no uploaded bytes"),
            Self::InvalidPhysicalPage(key) => {
                write!(
                    formatter,
                    "page {key:?} uses the non-resident physical-page sentinel"
                )
            }
            Self::DuplicatePhysicalPage(page) => {
                write!(
                    formatter,
                    "physical page {page} is aliased by multiple logical pages"
                )
            }
            Self::DuplicateMaterialSlot => formatter.write_str("material slot is published twice"),
            Self::MaterialSlotUnbound { slot } => {
                write!(formatter, "material slot {slot} has no global binding")
            }
            Self::DuplicateClusterNode(node) => {
                write!(formatter, "cluster node {node} is duplicated")
            }
            Self::CutAssetMismatch { expected, actual } => {
                write!(formatter, "cut asset {actual} does not match {expected}")
            }
            Self::CutNodeMissing { node } => {
                write!(formatter, "cut node {node} has no cluster descriptor")
            }
            Self::PageNotResident { key } => write!(formatter, "page {key:?} is not resident"),
            Self::CutPageMismatch {
                node,
                expected,
                actual,
            } => write!(
                formatter,
                "cut node {node} resolves page {actual}, expected {expected}"
            ),
            Self::CutPageIdentityMismatch {
                node,
                expected,
                actual,
            } => write!(
                formatter,
                "cut node {node} names page {actual:?}, descriptor expects {expected:?}"
            ),
            Self::DuplicateCutCluster { node } => {
                write!(formatter, "cut repeats cluster node {node}")
            }
        }
    }
}

impl std::error::Error for VirtualGeometryGpuPublicationError {}

fn split_generation(generation: u64) -> [u32; 2] {
    [generation as u32, (generation >> 32) as u32]
}

fn encode_cluster(
    cluster: &VirtualGeometryClusterRecord,
    page_index: u32,
    material: SceneMaterialHandle,
) -> VirtualGeometryGpuCluster {
    VirtualGeometryGpuCluster {
        node_index: cluster.node_index,
        page_index,
        payload_offset: cluster.payload_offset,
        payload_size: cluster.payload_size,
        material_index: material.index,
        material_generation: material.generation,
        vertex_count: u32::from(cluster.vertex_count),
        triangle_count: u32::from(cluster.triangle_count),
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;
    use crate::{
        gpu_scene::SceneBounds,
        virtual_geometry::{
            asset::{
                VirtualGeometryAssetHeader, VirtualGeometryCompression, VirtualGeometryPage,
                VirtualGeometryPageRecord, VIRTUAL_GEOMETRY_ASSET_SCHEMA,
            },
            hierarchy::ClusterNode,
            VirtualGeometryClusterRecord,
        },
    };

    fn bounds() -> SceneBounds {
        SceneBounds {
            center: [0.0; 3],
            radius: 2.0,
            half_extents: [2.0; 3],
            _padding: 0.0,
        }
    }

    fn asset() -> VirtualGeometryAsset {
        let key = GeometryPageKey::new(7, 0);
        VirtualGeometryAsset {
            header: VirtualGeometryAssetHeader {
                asset_id: 7,
                schema_version: VIRTUAL_GEOMETRY_ASSET_SCHEMA,
                page_size: 1024,
                build_settings_hash: [1; 32],
                content_hash: [2; 32],
                bounds: bounds(),
                material_slot_count: 1,
                fallback_mesh_hash: [3; 32],
            },
            nodes: vec![ClusterNode::leaf(bounds(), 0.0, key)],
            roots: vec![0],
            pages: vec![VirtualGeometryPage {
                record: VirtualGeometryPageRecord {
                    key,
                    compression: VirtualGeometryCompression::None,
                    uncompressed_size: 4,
                    priority: 1,
                    cluster_start: 0,
                    cluster_count: 1,
                },
                payload: vec![1, 2, 3, 4],
            }],
            clusters: vec![VirtualGeometryClusterRecord {
                node_index: 0,
                page: key,
                material_slot: 0,
                vertex_count: 3,
                triangle_count: 1,
                payload_offset: 0,
                payload_size: 4,
                bounds: bounds(),
                normal_cone: [0.0, 0.0, 1.0, 1.0],
            }],
        }
    }

    fn resident(generation: u64) -> VirtualGeometryPhysicalPage {
        VirtualGeometryPhysicalPage {
            key: GeometryPageKey::new(7, 0),
            physical_page: 3,
            byte_offset: 64,
            byte_len: 4,
            generation,
        }
    }

    #[test]
    fn publication_encodes_nonresident_and_resident_page_states() {
        let publication = VirtualGeometryGpuPublication::build(
            &asset(),
            [9; 32],
            11,
            &[resident(11)],
            &[VirtualGeometryMaterialBinding {
                material_slot: 0,
                material: SceneMaterialHandle::new(4, 8),
            }],
        )
        .expect("publication");
        assert!(publication.pages()[0].is_resident());
        assert_eq!(publication.pages()[0].physical_page, 3);
        assert_eq!(publication.clusters()[0].material_index, 4);
        assert_eq!(publication.clusters()[0].material_generation, 8);
        assert_eq!(
            publication.page_table_bytes().len(),
            core::mem::size_of::<VirtualGeometryGpuPage>()
        );
    }

    #[test]
    fn stale_page_generation_is_rejected_before_publication() {
        let error = VirtualGeometryGpuPublication::build(
            &asset(),
            [0; 32],
            11,
            &[resident(10)],
            &[VirtualGeometryMaterialBinding {
                material_slot: 0,
                material: SceneMaterialHandle::new(1, 1),
            }],
        )
        .expect_err("stale page must fail closed");
        assert!(matches!(
            error,
            VirtualGeometryGpuPublicationError::PageGenerationMismatch { .. }
        ));
    }

    #[test]
    fn draw_list_requires_resident_page_and_carries_scene_and_generation() {
        let key = GeometryPageKey::new(7, 0);
        let publication = VirtualGeometryGpuPublication::build(
            &asset(),
            [0; 32],
            5,
            &[resident(5)],
            &[VirtualGeometryMaterialBinding {
                material_slot: 0,
                material: SceneMaterialHandle::new(2, 9),
            }],
        )
        .expect("publication");
        let list = publication
            .draw_list(SceneHandle::new(6, 7), &[CutCluster { node: 0, page: key }])
            .expect("draw list");
        assert_eq!(list.items()[0].scene_index, 6);
        assert_eq!(list.items()[0].scene_generation, 7);
        assert_eq!(list.items()[0].publication_generation, split_generation(5));

        let without_page = VirtualGeometryGpuPublication::build(
            &asset(),
            [0; 32],
            5,
            &[],
            &[VirtualGeometryMaterialBinding {
                material_slot: 0,
                material: SceneMaterialHandle::new(2, 9),
            }],
        )
        .expect("publication with a nonresident page");
        assert!(matches!(
            without_page.draw_list(SceneHandle::new(6, 7), &[CutCluster { node: 0, page: key }]),
            Err(VirtualGeometryGpuPublicationError::PageNotResident { .. })
        ));
    }

    #[test]
    fn draw_list_rejects_page_key_that_does_not_match_descriptor() {
        let publication = VirtualGeometryGpuPublication::build(
            &asset(),
            [0; 32],
            5,
            &[resident(5)],
            &[VirtualGeometryMaterialBinding {
                material_slot: 0,
                material: SceneMaterialHandle::new(2, 9),
            }],
        )
        .expect("publication");
        let error = publication
            .draw_list(
                SceneHandle::new(6, 7),
                &[CutCluster {
                    node: 0,
                    page: GeometryPageKey::new(7, 99),
                }],
            )
            .expect_err("wrong logical page must fail closed");
        assert!(matches!(
            error,
            VirtualGeometryGpuPublicationError::CutPageIdentityMismatch { .. }
        ));
    }

    #[test]
    fn resident_physical_page_sentinel_and_aliases_are_rejected() {
        let mut sentinel = resident(3);
        sentinel.physical_page = NON_RESIDENT_PHYSICAL_PAGE;
        let error = VirtualGeometryGpuPublication::build(
            &asset(),
            [0; 32],
            3,
            &[sentinel],
            &[VirtualGeometryMaterialBinding {
                material_slot: 0,
                material: SceneMaterialHandle::new(1, 1),
            }],
        )
        .expect_err("resident sentinel must fail closed");
        assert!(matches!(
            error,
            VirtualGeometryGpuPublicationError::InvalidPhysicalPage(_)
        ));

        let mut second = asset();
        let second_key = GeometryPageKey::new(7, 1);
        second.pages.push(VirtualGeometryPage {
            record: VirtualGeometryPageRecord {
                key: second_key,
                compression: VirtualGeometryCompression::None,
                uncompressed_size: 4,
                priority: 1,
                cluster_start: 1,
                cluster_count: 0,
            },
            payload: vec![1, 2, 3, 4],
        });
        let first_resident = resident(3);
        let second_resident = VirtualGeometryPhysicalPage {
            key: second_key,
            physical_page: first_resident.physical_page,
            byte_offset: 128,
            byte_len: 4,
            generation: 3,
        };
        let error = VirtualGeometryGpuPublication::build(
            &second,
            [0; 32],
            3,
            &[first_resident, second_resident],
            &[VirtualGeometryMaterialBinding {
                material_slot: 0,
                material: SceneMaterialHandle::new(1, 1),
            }],
        )
        .expect_err("physical page alias must fail closed");
        assert!(matches!(
            error,
            VirtualGeometryGpuPublicationError::DuplicatePhysicalPage(3)
        ));
    }
}
