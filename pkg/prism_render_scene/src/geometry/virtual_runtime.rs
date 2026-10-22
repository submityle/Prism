//! Runtime registry for validated CLST virtual-geometry assets.
//!
//! Content revisions identify immutable asset contents across runs. Runtime
//! generations identify one publication lifetime inside a registry; replacing
//! or retiring an asset invalidates every handle from its previous lifetime.
//! The registry is deliberately independent of [`super::RenderGeometryRegistry`]
//! because CLST pages and clusters are not Bevy `Mesh` LODs.

use std::collections::BTreeMap;
use std::sync::Arc;

use bevy_ecs::prelude::Resource;
use prism_render_architecture::virtual_geometry::{
    GeometryPageKey, VirtualGeometryAsset, VirtualGeometryAssetError, VirtualGeometryClusterRecord,
    VirtualGeometryPageRecord,
};
use thiserror::Error;

/// Stable identity of a CLST asset within the authored virtual-geometry set.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct VirtualGeometryAssetId(u32);

impl VirtualGeometryAssetId {
    /// Creates an identity matching `VirtualGeometryAssetHeader::asset_id`.
    #[must_use]
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    /// Returns the on-disk asset identity.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// Stable identity for the exact content revision supplied by the asset
/// pipeline. The publisher is trusted to provide the canonical CLST artifact
/// digest; registration deliberately does not re-hash all page payloads. Do
/// not use a process-local counter, source-mesh hash, or registry generation.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct VirtualGeometryContentRevision([u8; 32]);

impl VirtualGeometryContentRevision {
    /// Builds a revision from a stable 256-bit artifact digest.
    #[must_use]
    pub const fn from_digest(digest: [u8; 32]) -> Self {
        Self(digest)
    }

    /// Returns the stable artifact digest.
    #[must_use]
    pub const fn digest(self) -> [u8; 32] {
        self.0
    }
}

/// Runtime publication generation. It advances on every successful register
/// or replacement and is never derived from asset content.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct VirtualGeometryGeneration(u64);

impl VirtualGeometryGeneration {
    /// Returns the monotonically increasing generation value.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Opaque handle to one published asset lifetime.
#[derive(Clone, Debug)]
pub struct VirtualGeometryAssetHandle {
    registry: Arc<RegistryIdentity>,
    asset_id: VirtualGeometryAssetId,
    generation: VirtualGeometryGeneration,
    content_revision: VirtualGeometryContentRevision,
}

impl VirtualGeometryAssetHandle {
    /// Returns the stable asset identity.
    #[must_use]
    pub const fn asset_id(&self) -> VirtualGeometryAssetId {
        self.asset_id
    }

    /// Returns this handle's runtime publication generation.
    #[must_use]
    pub const fn generation(&self) -> VirtualGeometryGeneration {
        self.generation
    }

    /// Returns the stable content revision bound to this publication.
    #[must_use]
    pub const fn content_revision(&self) -> VirtualGeometryContentRevision {
        self.content_revision
    }
}

/// Opaque, generation-bound handle to one page descriptor.
#[derive(Clone, Debug)]
pub struct VirtualGeometryPageHandle {
    asset: VirtualGeometryAssetHandle,
    key: GeometryPageKey,
}

impl VirtualGeometryPageHandle {
    /// Returns the page key, including its owning asset identity.
    #[must_use]
    pub const fn key(&self) -> GeometryPageKey {
        self.key
    }

    /// Returns the owning asset publication handle.
    #[must_use]
    pub const fn asset(&self) -> &VirtualGeometryAssetHandle {
        &self.asset
    }
}

/// Opaque, generation-bound handle to one cluster descriptor.
#[derive(Clone, Debug)]
pub struct VirtualGeometryClusterHandle {
    page: VirtualGeometryPageHandle,
    cluster_index: u32,
}

impl VirtualGeometryClusterHandle {
    /// Returns the owning page handle.
    #[must_use]
    pub const fn page(&self) -> &VirtualGeometryPageHandle {
        &self.page
    }

    /// Returns the cluster's index in the validated asset descriptor array.
    #[must_use]
    pub const fn index(&self) -> u32 {
        self.cluster_index
    }
}

/// Registry errors. Validation errors are preserved so invalid candidates can
/// be diagnosed without mutating the currently published asset.
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum VirtualGeometryRegistryError {
    /// The candidate failed the CLST asset's cross-table validation.
    #[error("invalid virtual geometry asset: {0:?}")]
    InvalidAsset(VirtualGeometryAssetError),
    /// An asset with this stable identity is already registered.
    #[error("virtual geometry asset {0} is already registered")]
    AlreadyRegistered(u32),
    /// No live asset exists for the requested identity.
    #[error("virtual geometry asset {0} is not registered")]
    NotRegistered(u32),
    /// The supplied handle belongs to another registry or an older generation.
    #[error("stale or foreign virtual geometry handle")]
    StaleHandle,
    /// The requested page is not present in the live asset.
    #[error("virtual geometry page {0:?} does not exist")]
    PageNotFound(GeometryPageKey),
    /// The requested cluster index is absent or belongs to a different page.
    #[error("virtual geometry cluster {cluster_index} does not belong to page {page:?}")]
    ClusterNotFound {
        /// Page through which the cluster was requested.
        page: GeometryPageKey,
        /// Index in the asset's canonical cluster descriptor array.
        cluster_index: u32,
    },
    /// The generation counter cannot advance without wrapping.
    #[error("virtual geometry generation space is exhausted")]
    GenerationExhausted,
}

#[derive(Debug)]
struct RegistryIdentity;

struct PublishedAsset {
    generation: VirtualGeometryGeneration,
    content_revision: VirtualGeometryContentRevision,
    asset: Arc<VirtualGeometryAsset>,
    page_indices: BTreeMap<GeometryPageKey, usize>,
}

/// Validated CLST asset registry with generation-safe page and cluster lookup.
///
/// All candidate validation and index construction happens before publishing a
/// replacement. The final map insertion is the publication point; callers
/// holding prior handles are rejected after that point.
#[derive(Resource)]
pub struct VirtualGeometryRuntimeRegistry {
    identity: Arc<RegistryIdentity>,
    next_generation: u64,
    assets: BTreeMap<VirtualGeometryAssetId, PublishedAsset>,
}

impl Default for VirtualGeometryRuntimeRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl VirtualGeometryRuntimeRegistry {
    /// Creates an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            identity: Arc::new(RegistryIdentity),
            next_generation: 1,
            assets: BTreeMap::new(),
        }
    }

    /// Validates and registers a previously absent asset identity.
    pub fn register(
        &mut self,
        candidate: VirtualGeometryAsset,
        content_revision: VirtualGeometryContentRevision,
    ) -> Result<VirtualGeometryAssetHandle, VirtualGeometryRegistryError> {
        candidate
            .validate()
            .map_err(VirtualGeometryRegistryError::InvalidAsset)?;
        let asset_id = VirtualGeometryAssetId::new(candidate.header.asset_id);
        if self.assets.contains_key(&asset_id) {
            return Err(VirtualGeometryRegistryError::AlreadyRegistered(
                asset_id.get(),
            ));
        }
        let generation = self.allocate_generation()?;
        let published = prepare_publication(candidate, content_revision, generation);
        self.assets.insert(asset_id, published);
        Ok(self.handle(asset_id, generation, content_revision))
    }

    /// Validates a complete replacement before atomically publishing it.
    ///
    /// The candidate must retain the stable asset identity of `current`; its
    /// content revision may change independently. Failed validation, identity
    /// mismatch, or generation exhaustion leaves the old publication intact.
    pub fn replace(
        &mut self,
        current: &VirtualGeometryAssetHandle,
        candidate: VirtualGeometryAsset,
        content_revision: VirtualGeometryContentRevision,
    ) -> Result<VirtualGeometryAssetHandle, VirtualGeometryRegistryError> {
        self.validate_asset_handle(current)?;
        candidate
            .validate()
            .map_err(VirtualGeometryRegistryError::InvalidAsset)?;
        if candidate.header.asset_id != current.asset_id.get() {
            return Err(VirtualGeometryRegistryError::InvalidAsset(
                VirtualGeometryAssetError::InvalidValue("replacement asset identity"),
            ));
        }

        let generation = self.allocate_generation()?;
        let published = prepare_publication(candidate, content_revision, generation);
        self.assets.insert(current.asset_id, published);
        Ok(self.handle(current.asset_id, generation, content_revision))
    }

    /// Retires the exact live publication represented by `handle`.
    pub fn retire(
        &mut self,
        handle: &VirtualGeometryAssetHandle,
    ) -> Result<(), VirtualGeometryRegistryError> {
        self.validate_asset_handle(handle)?;
        self.assets.remove(&handle.asset_id);
        Ok(())
    }

    /// Returns a fresh handle to the currently published asset, if present.
    #[must_use]
    pub fn current_handle(
        &self,
        asset_id: VirtualGeometryAssetId,
    ) -> Option<VirtualGeometryAssetHandle> {
        self.assets.get(&asset_id).map(|published| {
            self.handle(asset_id, published.generation, published.content_revision)
        })
    }

    /// Resolves a page key to a handle tied to the supplied live generation.
    pub fn page_handle(
        &self,
        asset: &VirtualGeometryAssetHandle,
        key: GeometryPageKey,
    ) -> Result<VirtualGeometryPageHandle, VirtualGeometryRegistryError> {
        let published = self.validate_asset_handle(asset)?;
        if key.asset != asset.asset_id.get() || !published.page_indices.contains_key(&key) {
            return Err(VirtualGeometryRegistryError::PageNotFound(key));
        }
        Ok(VirtualGeometryPageHandle {
            asset: asset.clone(),
            key,
        })
    }

    /// Queries a page descriptor directly by its typed asset handle and
    /// [`GeometryPageKey`].
    pub fn page_descriptor_by_key(
        &self,
        asset: &VirtualGeometryAssetHandle,
        key: GeometryPageKey,
    ) -> Result<&VirtualGeometryPageRecord, VirtualGeometryRegistryError> {
        let page = self.page_handle(asset, key)?;
        self.page_descriptor(&page)
    }

    /// Queries the validated descriptor for a generation-bound page handle.
    pub fn page_descriptor(
        &self,
        page: &VirtualGeometryPageHandle,
    ) -> Result<&VirtualGeometryPageRecord, VirtualGeometryRegistryError> {
        let published = self.validate_asset_handle(&page.asset)?;
        let index = published
            .page_indices
            .get(&page.key)
            .copied()
            .ok_or(VirtualGeometryRegistryError::PageNotFound(page.key))?;
        Ok(&published.asset.pages[index].record)
    }

    /// Queries all cluster descriptors stored in a page, in the canonical
    /// descriptor order encoded by that page's validated range.
    pub fn cluster_descriptors_for_page(
        &self,
        asset: &VirtualGeometryAssetHandle,
        key: GeometryPageKey,
    ) -> Result<&[VirtualGeometryClusterRecord], VirtualGeometryRegistryError> {
        let published = self.validate_asset_handle(asset)?;
        let page_index = published
            .page_indices
            .get(&key)
            .copied()
            .filter(|_| key.asset == asset.asset_id.get())
            .ok_or(VirtualGeometryRegistryError::PageNotFound(key))?;
        let record = published.asset.pages[page_index].record;
        let start = record.cluster_start as usize;
        let end = start + usize::from(record.cluster_count);
        Ok(&published.asset.clusters[start..end])
    }

    /// Creates a typed handle for one cluster index belonging to `page`.
    pub fn cluster_handle(
        &self,
        page: &VirtualGeometryPageHandle,
        cluster_index: u32,
    ) -> Result<VirtualGeometryClusterHandle, VirtualGeometryRegistryError> {
        let published = self.validate_asset_handle(&page.asset)?;
        if page.key.asset != page.asset.asset_id.get()
            || !published.page_indices.contains_key(&page.key)
        {
            return Err(VirtualGeometryRegistryError::PageNotFound(page.key));
        }
        published
            .asset
            .clusters
            .get(cluster_index as usize)
            .filter(|descriptor| descriptor.page == page.key)
            .ok_or(VirtualGeometryRegistryError::ClusterNotFound {
                page: page.key,
                cluster_index,
            })?;
        Ok(VirtualGeometryClusterHandle {
            page: page.clone(),
            cluster_index,
        })
    }

    /// Queries a validated cluster descriptor through its typed handle.
    pub fn cluster_descriptor(
        &self,
        cluster: &VirtualGeometryClusterHandle,
    ) -> Result<&VirtualGeometryClusterRecord, VirtualGeometryRegistryError> {
        let published = self.validate_asset_handle(&cluster.page.asset)?;
        published
            .asset
            .clusters
            .get(cluster.cluster_index as usize)
            .filter(|descriptor| descriptor.page == cluster.page.key)
            .ok_or(VirtualGeometryRegistryError::ClusterNotFound {
                page: cluster.page.key,
                cluster_index: cluster.cluster_index,
            })
    }

    /// Returns the number of currently published assets.
    #[must_use]
    pub fn len(&self) -> usize {
        self.assets.len()
    }

    /// Returns whether no assets are currently published.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.assets.is_empty()
    }

    fn allocate_generation(
        &mut self,
    ) -> Result<VirtualGeometryGeneration, VirtualGeometryRegistryError> {
        let generation = self.next_generation;
        self.next_generation = generation
            .checked_add(1)
            .ok_or(VirtualGeometryRegistryError::GenerationExhausted)?;
        Ok(VirtualGeometryGeneration(generation))
    }

    fn handle(
        &self,
        asset_id: VirtualGeometryAssetId,
        generation: VirtualGeometryGeneration,
        content_revision: VirtualGeometryContentRevision,
    ) -> VirtualGeometryAssetHandle {
        VirtualGeometryAssetHandle {
            registry: Arc::clone(&self.identity),
            asset_id,
            generation,
            content_revision,
        }
    }

    fn validate_asset_handle(
        &self,
        handle: &VirtualGeometryAssetHandle,
    ) -> Result<&PublishedAsset, VirtualGeometryRegistryError> {
        if !Arc::ptr_eq(&self.identity, &handle.registry) {
            return Err(VirtualGeometryRegistryError::StaleHandle);
        }
        let published = self.assets.get(&handle.asset_id).ok_or(
            VirtualGeometryRegistryError::NotRegistered(handle.asset_id.get()),
        )?;
        if published.generation != handle.generation
            || published.content_revision != handle.content_revision
        {
            return Err(VirtualGeometryRegistryError::StaleHandle);
        }
        Ok(published)
    }
}

fn prepare_publication(
    candidate: VirtualGeometryAsset,
    content_revision: VirtualGeometryContentRevision,
    generation: VirtualGeometryGeneration,
) -> PublishedAsset {
    let page_indices = candidate
        .pages
        .iter()
        .enumerate()
        .map(|(index, page)| (page.record.key, index))
        .collect();
    PublishedAsset {
        generation,
        content_revision,
        asset: Arc::new(candidate),
        page_indices,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_architecture::gpu_scene::SceneBounds;
    use prism_render_architecture::virtual_geometry::{
        ClusterNode, VirtualGeometryAssetHeader, VirtualGeometryCompression, VirtualGeometryPage,
    };

    fn bounds() -> SceneBounds {
        SceneBounds {
            center: [0.0; 3],
            radius: 2.0,
            half_extents: [2.0; 3],
            _padding: 0.0,
        }
    }

    fn asset(asset_id: u32, payload_byte: u8) -> VirtualGeometryAsset {
        let key = GeometryPageKey::new(asset_id, 0);
        VirtualGeometryAsset {
            header: VirtualGeometryAssetHeader {
                asset_id,
                schema_version: 1,
                page_size: 1024,
                build_settings_hash: [1; 32],
                content_hash: [payload_byte; 32],
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
                    priority: 10,
                    cluster_start: 0,
                    cluster_count: 1,
                },
                payload: vec![payload_byte; 4],
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

    fn revision(byte: u8) -> VirtualGeometryContentRevision {
        VirtualGeometryContentRevision::from_digest([byte; 32])
    }

    #[test]
    fn registration_validates_and_resolves_page_and_cluster_descriptors() {
        let mut registry = VirtualGeometryRuntimeRegistry::new();
        let asset_handle = registry.register(asset(7, 4), revision(11)).unwrap();
        let key = GeometryPageKey::new(7, 0);
        let page = registry.page_handle(&asset_handle, key).unwrap();

        assert_eq!(asset_handle.content_revision(), revision(11));
        assert_eq!(registry.page_descriptor(&page).unwrap().key, key);
        assert_eq!(
            registry
                .page_descriptor_by_key(&asset_handle, key)
                .unwrap()
                .key,
            key
        );
        assert_eq!(
            registry
                .cluster_descriptors_for_page(&asset_handle, key)
                .unwrap()
                .len(),
            1
        );
        let cluster = registry.cluster_handle(&page, 0).unwrap();
        let descriptor = registry.cluster_descriptor(&cluster).unwrap();
        assert_eq!(descriptor.page, key);
        assert_eq!(descriptor.vertex_count, 3);
    }

    #[test]
    fn cluster_lookup_reports_bad_cluster_identity_without_misreporting_page_absence() {
        let mut registry = VirtualGeometryRuntimeRegistry::new();
        let asset = registry.register(asset(7, 4), revision(11)).unwrap();
        let key = GeometryPageKey::new(7, 0);
        let page = registry.page_handle(&asset, key).unwrap();

        assert!(matches!(
            registry.cluster_handle(&page, 1),
            Err(VirtualGeometryRegistryError::ClusterNotFound {
                page: error_page,
                cluster_index: 1,
            }) if error_page == key
        ));
    }

    #[test]
    fn replacement_advances_generation_and_rejects_old_asset_page_and_cluster_handles() {
        let mut registry = VirtualGeometryRuntimeRegistry::new();
        let old = registry.register(asset(7, 4), revision(11)).unwrap();
        let old_page = registry
            .page_handle(&old, GeometryPageKey::new(7, 0))
            .unwrap();
        let old_cluster = registry.cluster_handle(&old_page, 0).unwrap();

        let current = registry.replace(&old, asset(7, 9), revision(12)).unwrap();

        assert!(current.generation().get() > old.generation().get());
        assert_eq!(current.content_revision(), revision(12));
        assert_eq!(
            registry.page_descriptor(&old_page),
            Err(VirtualGeometryRegistryError::StaleHandle)
        );
        assert_eq!(
            registry.cluster_descriptor(&old_cluster),
            Err(VirtualGeometryRegistryError::StaleHandle)
        );
        assert_eq!(
            registry
                .page_descriptor(
                    &registry
                        .page_handle(&current, GeometryPageKey::new(7, 0))
                        .unwrap()
                )
                .unwrap()
                .key,
            GeometryPageKey::new(7, 0)
        );
    }

    #[test]
    fn invalid_replacement_preserves_the_live_publication() {
        let mut registry = VirtualGeometryRuntimeRegistry::new();
        let current = registry.register(asset(7, 4), revision(11)).unwrap();
        let mut invalid = asset(7, 9);
        invalid.clusters[0].page = GeometryPageKey::new(99, 0);

        assert!(matches!(
            registry.replace(&current, invalid, revision(12)),
            Err(VirtualGeometryRegistryError::InvalidAsset(_))
        ));
        let still_current = registry.current_handle(current.asset_id()).unwrap();
        assert_eq!(still_current.generation(), current.generation());
        assert_eq!(still_current.content_revision(), revision(11));
        let page = registry
            .page_handle(&current, GeometryPageKey::new(7, 0))
            .unwrap();
        assert_eq!(
            registry
                .cluster_descriptor(&registry.cluster_handle(&page, 0).unwrap())
                .unwrap()
                .page,
            GeometryPageKey::new(7, 0)
        );
    }

    #[test]
    fn retire_and_reregister_never_revalidate_old_handles() {
        let mut registry = VirtualGeometryRuntimeRegistry::new();
        let old = registry.register(asset(7, 4), revision(11)).unwrap();
        registry.retire(&old).unwrap();
        assert!(matches!(
            registry.page_handle(&old, GeometryPageKey::new(7, 0)),
            Err(VirtualGeometryRegistryError::NotRegistered(7))
        ));
        let replacement = registry.register(asset(7, 4), revision(11)).unwrap();
        assert!(replacement.generation().get() > old.generation().get());
        assert!(matches!(
            registry.retire(&old),
            Err(VirtualGeometryRegistryError::StaleHandle)
        ));
    }

    #[test]
    fn page_keys_and_handles_are_isolated_between_assets_and_registries() {
        let mut registry = VirtualGeometryRuntimeRegistry::new();
        let first = registry.register(asset(7, 4), revision(11)).unwrap();
        let second = registry.register(asset(8, 5), revision(12)).unwrap();
        assert_ne!(first.asset_id(), second.asset_id());
        assert!(matches!(
            registry.page_handle(&first, GeometryPageKey::new(8, 0)),
            Err(VirtualGeometryRegistryError::PageNotFound(key))
                if key == GeometryPageKey::new(8, 0)
        ));
        assert_eq!(
            registry
                .page_descriptor(
                    &registry
                        .page_handle(&second, GeometryPageKey::new(8, 0))
                        .unwrap()
                )
                .unwrap()
                .key
                .asset,
            second.asset_id().get()
        );

        let other_registry = VirtualGeometryRuntimeRegistry::new();
        assert!(matches!(
            other_registry.page_handle(&first, GeometryPageKey::new(7, 0)),
            Err(VirtualGeometryRegistryError::StaleHandle)
        ));
    }

    #[test]
    fn generation_exhaustion_never_replaces_or_partially_publishes_an_asset() {
        let mut registry = VirtualGeometryRuntimeRegistry::new();
        let current = registry.register(asset(7, 4), revision(11)).unwrap();
        registry.next_generation = u64::MAX;

        assert!(matches!(
            registry.replace(&current, asset(7, 9), revision(12)),
            Err(VirtualGeometryRegistryError::GenerationExhausted)
        ));
        assert!(matches!(
            registry.register(asset(8, 5), revision(13)),
            Err(VirtualGeometryRegistryError::GenerationExhausted)
        ));
        let still_current = registry.current_handle(current.asset_id()).unwrap();
        assert_eq!(still_current.generation(), current.generation());
        assert_eq!(still_current.content_revision(), revision(11));
        assert!(registry
            .current_handle(VirtualGeometryAssetId::new(8))
            .is_none());
    }
}
