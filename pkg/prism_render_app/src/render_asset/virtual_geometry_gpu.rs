//! Render-world CLST publication buffers.
//!
//! [`prism_render_architecture::virtual_geometry::VirtualGeometryPublicationCoordinator`]
//! owns deterministic page admission and generation-safe CPU publication. This
//! module is the device boundary: it creates one page-storage buffer plus the
//! stable page/Cluster tables for each registered asset, then consumes only
//! incremental deltas. Storage writes are always queued before page-table row
//! writes, so a newly resident page cannot become addressable before its bytes
//! are available to the next ordered submission.

use alloc::{collections::BTreeMap, sync::Arc, vec, vec::Vec};
use core::fmt;

use bytemuck::{bytes_of, cast_slice};
use prism_app::App;
use prism_ecs::prelude::{FromWorld, Res, ResMut, Resource};
use prism_ecs::schedule::{IntoSystemConfigs, Schedule, Schedules};
use prism_ecs::world::World;
use prism_render_architecture::{
    paging::RequestBatch,
    virtual_geometry::{
        GeometryPageKey, VirtualGeometryAsset, VirtualGeometryGpuPublication,
        VirtualGeometryMaterialBinding, VirtualGeometryPublicationCoordinator,
        VirtualGeometryPublicationCoordinatorError, VirtualGeometryPublicationDelta,
    },
};
use prism_render_driver::FrameSync;
use prism_render_driver::{
    BufferDescriptor, BufferId, BufferUsages, Instance, RenderDevice, RenderQueue,
};

use crate::runtime_resource::RenderRuntimeResource;
use crate::{label::RenderApp, sets::RenderSystems};

#[cfg(test)]
#[path = "virtual_geometry_gpu_store_tests.rs"]
mod store_tests;

/// One explicit main-world command for the RenderApp publication store.
#[derive(Clone, Debug)]
pub enum VirtualGeometryPublicationRequest {
    /// Create the generation-bound GPU publication and its page pool.
    Register {
        asset: Arc<VirtualGeometryAsset>,
        content_revision: [u8; 32],
        generation: u64,
        physical_capacity: u32,
        resident_budget: usize,
        material_bindings: Vec<VirtualGeometryMaterialBinding>,
    },
    /// Reconcile the requested page set for one render frame.
    Reconcile {
        asset_id: u32,
        /// Required generation from the producer's current scene publication.
        generation: u64,
        requests: RequestBatch<GeometryPageKey>,
        frame: u64,
    },
    /// Retire all publications after queue completion is confirmed.
    RetireAll,
}

/// Main-world command queue. Producers append typed commands; extraction drains
/// it exactly once so failed commands are diagnosed rather than replayed.
#[derive(Default, Resource)]
pub struct VirtualGeometryPublicationRequests {
    next_sequence: u64,
    requests: Vec<(u64, VirtualGeometryPublicationRequest)>,
}

impl VirtualGeometryPublicationRequests {
    /// Appends a command and returns its stable diagnostic sequence.
    pub fn push(&mut self, request: VirtualGeometryPublicationRequest) -> u64 {
        let sequence = self.next_sequence;
        self.next_sequence = self
            .next_sequence
            .checked_add(1)
            .expect("publication request sequence exhausted");
        self.requests.push((sequence, request));
        sequence
    }
}

/// Commands copied into RenderApp by the registered extraction step.
#[derive(Default, Resource)]
pub struct ExtractedVirtualGeometryPublicationRequests {
    requests: Vec<(u64, VirtualGeometryPublicationRequest)>,
}

/// Per-frame outcomes from the publication bridge; no draw work is implied.
#[derive(Default, Resource)]
pub struct VirtualGeometryPublicationDiagnostics {
    /// Number of commands attempted in the latest prepare pass.
    pub attempted: usize,
    /// Number of successfully applied commands in the latest prepare pass.
    pub succeeded: usize,
    /// Command sequence and exact store error for each rejected command.
    pub errors: Vec<(u64, VirtualGeometryGpuStoreError)>,
}

/// Render-world registration guard for the native publication bridge.
///
/// The production render plugin installs the bridge by default, while an
/// application may also call the public registration helper explicitly.  The
/// extract and prepare stages are singletons: registering either twice would
/// make the first prepare consume the commands and the second prepare reset
/// the diagnostics to zero in the same frame.
#[derive(Resource, Default)]
struct VirtualGeometryPublicationBridgeInstalled;

/// Generation-bound residency state exported from RenderApp to the scene
/// planner. Only changed page rows are transported; immutable Cluster tables
/// and asset descriptors stay owned by their existing registries.
#[derive(Clone, Debug, Default, Resource)]
pub struct VirtualGeometryResidencySnapshot {
    /// Per-asset content revision, publication generation, and changed logical
    /// page residency rows. A new generation replaces all rows for that asset.
    pub assets: Vec<VirtualGeometryResidencyAssetSnapshot>,
}

/// One asset's residency delta for a specific publication generation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VirtualGeometryResidencyAssetSnapshot {
    /// Stable CLST asset identity.
    pub asset_id: u32,
    /// Immutable artifact digest tied to this residency state.
    pub content_revision: [u8; 32],
    /// Runtime publication lifetime.
    pub generation: u64,
    /// False retires all scene-side rows for this asset identity.
    pub active: bool,
    /// Changed sorted logical page rows; `resident == false` revokes a prior row.
    pub changed_pages: Vec<(GeometryPageKey, bool)>,
}

impl VirtualGeometryResidencySnapshot {
    /// Drains residency changes for an owning host to forward across another
    /// ECS boundary. RenderApp never writes these results back into MainWorld.
    pub fn drain_updates(&mut self) -> Vec<VirtualGeometryResidencyAssetSnapshot> {
        core::mem::take(&mut self.assets)
    }
}

/// Merges a sorted page-state delta into a sorted accumulated snapshot in
/// place. Existing allocation is retained; each page is read/written at most
/// once, giving O(existing + delta) work rather than repeated Vec insertion.
fn merge_sorted_residency_pages(
    existing: &mut Vec<(GeometryPageKey, bool)>,
    mut delta: Vec<(GeometryPageKey, bool)>,
) {
    if delta.is_empty() {
        return;
    }
    delta.sort_by_key(|row| row.0);
    // Collapse malformed duplicates deterministically. Stable sort preserves
    // source order among equal keys, so the last state wins.
    let mut write = 0;
    for read in 0..delta.len() {
        if write > 0 && delta[write - 1].0 == delta[read].0 {
            delta[write - 1].1 = delta[read].1;
        } else {
            delta[write] = delta[read];
            write += 1;
        }
    }
    delta.truncate(write);

    let existing_len = existing.len();
    let delta_len = delta.len();
    let mut existing_index = 0;
    let mut delta_index = 0;
    let mut overlap_count = 0;
    while existing_index < existing_len && delta_index < delta_len {
        match existing[existing_index].0.cmp(&delta[delta_index].0) {
            core::cmp::Ordering::Less => existing_index += 1,
            core::cmp::Ordering::Greater => delta_index += 1,
            core::cmp::Ordering::Equal => {
                overlap_count += 1;
                existing_index += 1;
                delta_index += 1;
            }
        }
    }
    let merged_len = existing_len + delta_len - overlap_count;
    existing.resize(
        existing_len + delta_len,
        (GeometryPageKey::new(0, 0), false),
    );
    let mut left = existing_len;
    let mut right = delta_len;
    let mut output = merged_len;
    while left > 0 && right > 0 {
        output -= 1;
        match existing[left - 1].0.cmp(&delta[right - 1].0) {
            core::cmp::Ordering::Greater => {
                left -= 1;
                existing[output] = existing[left];
            }
            core::cmp::Ordering::Less => {
                right -= 1;
                existing[output] = delta[right];
            }
            core::cmp::Ordering::Equal => {
                left -= 1;
                right -= 1;
                existing[output] = delta[right];
            }
        }
    }
    if right > 0 {
        existing[..right].copy_from_slice(&delta[..right]);
    }
    existing.truncate(merged_len);
}

fn extract_virtual_geometry_publication_requests(main: &mut World, render: &mut World) {
    if let (Some(source), Some(target)) = (
        main.get_resource_mut::<VirtualGeometryPublicationRequests>(),
        render.get_resource_mut::<ExtractedVirtualGeometryPublicationRequests>(),
    ) {
        target
            .requests
            .extend(core::mem::take(&mut source.requests));
    }
}

/// Installs one-way request extraction and RenderApp PrepareResources
/// publication. Call after `PrismRenderPlugin`; this only publishes buffers
/// and does not create draw records or issue draw commands.
pub fn register_virtual_geometry_publication_bridge<I>(app: &mut App)
where
    I: Instance + Send + Sync + 'static,
    I::Adapter: Send + Sync + 'static,
    I::Device: Send + Sync + 'static,
    I::Queue: Send + Sync + FrameSync + 'static,
    I::Surface: Send + Sync + 'static,
{
    {
        let render = app.sub_app_mut(RenderApp).expect(
            "PrismRenderPlugin must be installed before the virtual-geometry publication bridge",
        );
        if render
            .world
            .contains_resource::<VirtualGeometryPublicationBridgeInstalled>()
        {
            return;
        }
        render
            .world
            .insert_resource(VirtualGeometryPublicationBridgeInstalled);
    }

    app.init_resource::<VirtualGeometryPublicationRequests>();
    app.add_extract(RenderApp, extract_virtual_geometry_publication_requests);
    {
        let render = app.sub_app_mut(RenderApp).expect(
            "PrismRenderPlugin must be installed before the virtual-geometry publication bridge",
        );
        render
            .world
            .init_resource::<ExtractedVirtualGeometryPublicationRequests>();
        render
            .world
            .init_resource::<VirtualGeometryPublicationDiagnostics>();
        render
            .world
            .init_resource::<VirtualGeometryResidencySnapshot>();
        let schedules = render.world.resource_mut::<Schedules>();
        if !schedules.contains(crate::Render) {
            schedules.insert(crate::Render, Schedule::new());
        }
        schedules
            .get_mut(crate::Render)
            .expect("render schedule exists")
            .add_systems(
                prepare_virtual_geometry_publications::<I>.in_set(RenderSystems::PrepareResources),
            );
    }
}

fn prepare_virtual_geometry_publications<I>(
    mut extracted: ResMut<ExtractedVirtualGeometryPublicationRequests>,
    mut store: ResMut<VirtualGeometryGpuPublicationStore<I>>,
    mut diagnostics: ResMut<VirtualGeometryPublicationDiagnostics>,
    mut published_residency: ResMut<VirtualGeometryResidencySnapshot>,
    runtime: Res<RenderRuntimeResource<I>>,
) where
    I: Instance + Send + Sync + 'static,
    I::Adapter: Send + Sync + 'static,
    I::Device: Send + Sync + 'static,
    I::Queue: Send + Sync + FrameSync + 'static,
    I::Surface: Send + Sync + 'static,
{
    diagnostics.attempted = extracted.requests.len();
    diagnostics.succeeded = 0;
    diagnostics.errors.clear();
    let mut residency = VirtualGeometryResidencySnapshot::default();
    for (sequence, request) in extracted.requests.drain(..) {
        let result = match request {
            VirtualGeometryPublicationRequest::Register {
                asset,
                content_revision,
                generation,
                physical_capacity,
                resident_budget,
                material_bindings,
            } => {
                let asset_id = asset.header.asset_id;
                store
                    .register(
                        asset,
                        content_revision,
                        generation,
                        physical_capacity,
                        resident_budget,
                        &material_bindings,
                        runtime.runtime().queue(),
                    )
                    .map(|_| {
                        if let Some(publication) = store.publication(asset_id) {
                            residency
                                .assets
                                .push(VirtualGeometryResidencyAssetSnapshot {
                                    asset_id: publication.asset_id(),
                                    content_revision: publication.content_revision(),
                                    generation: publication.generation(),
                                    active: true,
                                    changed_pages: publication
                                        .page_keys()
                                        .iter()
                                        .enumerate()
                                        .map(|(index, key)| {
                                            (
                                                *key,
                                                publication
                                                    .page(index as u32)
                                                    .is_some_and(|p| p.is_resident()),
                                            )
                                        })
                                        .collect(),
                                });
                        }
                    })
            }
            VirtualGeometryPublicationRequest::Reconcile {
                asset_id,
                generation,
                requests,
                frame,
            } => match store.publication(asset_id) {
                None => Err(VirtualGeometryGpuStoreError::UnknownAsset(asset_id)),
                Some(publication) if publication.generation() != generation => {
                    Err(VirtualGeometryGpuStoreError::GenerationMismatch {
                        asset: asset_id,
                        expected: publication.generation(),
                        actual: generation,
                    })
                }
                Some(_) => store
                    .reconcile(asset_id, &requests, frame, runtime.runtime().queue())
                    .map(|delta| {
                        if !delta.page_table_updates.is_empty() {
                            if let Some(publication) = store.publication(asset_id) {
                                residency
                                    .assets
                                    .push(VirtualGeometryResidencyAssetSnapshot {
                                        asset_id,
                                        content_revision: publication.content_revision(),
                                        generation: delta.generation,
                                        active: true,
                                        changed_pages: delta
                                            .page_table_updates
                                            .iter()
                                            .filter_map(|update| {
                                                let key = publication
                                                    .page_keys()
                                                    .get(update.page_index as usize)
                                                    .copied()?;
                                                Some((key, update.row.is_resident()))
                                            })
                                            .collect(),
                                    });
                            }
                        }
                    }),
            },
            VirtualGeometryPublicationRequest::RetireAll => {
                let retired = store
                    .asset_identities()
                    .into_iter()
                    .filter_map(|asset_id| {
                        let publication = store.publication(asset_id)?;
                        Some((
                            asset_id,
                            publication.content_revision(),
                            publication.generation(),
                        ))
                    })
                    .collect::<Vec<_>>();
                store.retire(runtime.runtime().queue()).map(|()| {
                    residency.assets.extend(retired.into_iter().map(
                        |(asset_id, content_revision, generation)| {
                            VirtualGeometryResidencyAssetSnapshot {
                                asset_id,
                                content_revision,
                                generation,
                                active: false,
                                changed_pages: Vec::new(),
                            }
                        },
                    ));
                })
            }
        };
        match result {
            Ok(()) => diagnostics.succeeded += 1,
            Err(error) => diagnostics.errors.push((sequence, error)),
        }
    }
    if !residency.assets.is_empty() {
        residency.assets.sort_by_key(|asset| asset.asset_id);
        for update in residency.assets {
            if let Some(current) = published_residency
                .assets
                .iter_mut()
                .find(|current| current.asset_id == update.asset_id)
            {
                if update.active
                    && current.active
                    && current.generation == update.generation
                    && current.content_revision == update.content_revision
                {
                    merge_sorted_residency_pages(&mut current.changed_pages, update.changed_pages);
                } else if update.generation > current.generation
                    || (update.generation == current.generation
                        && update.content_revision == current.content_revision
                        && !update.active
                        && current.active)
                {
                    *current = update;
                }
            } else {
                published_residency.assets.push(update);
            }
        }
        published_residency
            .assets
            .sort_by_key(|asset| asset.asset_id);
    }
}

/// The three buffers consumed by a CLST draw/resolve backend.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VirtualGeometryGpuBuffers {
    /// Fixed-stride CLST page storage.
    pub page_storage: BufferId,
    /// Logical page table indexed by the asset's sorted page order.
    pub page_table: BufferId,
    /// Immutable Cluster descriptors.
    pub cluster_table: BufferId,
}

/// Render-side errors while publishing a CLST asset or delta.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VirtualGeometryGpuStoreError {
    /// A stable asset identity is already owned by this device store.
    DuplicateAsset(u32),
    /// A reconcile or lookup named no registered asset.
    UnknownAsset(u32),
    /// A reconcile command belongs to a stale or future publication generation.
    GenerationMismatch {
        asset: u32,
        expected: u64,
        actual: u64,
    },
    /// The source asset requested no physical slots.
    ZeroPhysicalCapacity,
    /// A buffer byte-size calculation overflowed.
    BufferSizeOverflow,
    /// The coordinator rejected the source or its generation-bound update.
    Publication(VirtualGeometryPublicationCoordinatorError),
    /// A delta attempted to address outside the registered page-storage pool.
    PhysicalPageOutOfRange { asset: u32, physical_page: u32 },
    /// A delta's staging word count did not match the fixed page ABI.
    PageWordsMismatch {
        asset: u32,
        expected: u32,
        actual: usize,
    },
    /// A delta attempted to update outside the logical page table.
    PageTableIndexOutOfRange { asset: u32, page_index: u32 },
    /// Queue completion could not be confirmed, so buffers remain owned.
    QueueCompletionUnconfirmed,
}

impl fmt::Display for VirtualGeometryGpuStoreError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateAsset(asset) => {
                write!(formatter, "CLST asset {asset} is registered twice")
            }
            Self::UnknownAsset(asset) => write!(formatter, "CLST asset {asset} is not registered"),
            Self::GenerationMismatch { asset, expected, actual } => write!(formatter, "CLST asset {asset} reconcile generation {actual} does not match registered generation {expected}"),
            Self::ZeroPhysicalCapacity => {
                formatter.write_str("CLST physical page capacity is zero")
            }
            Self::BufferSizeOverflow => formatter.write_str("CLST GPU buffer size overflowed"),
            Self::Publication(error) => write!(formatter, "CLST publication failed: {error}"),
            Self::PhysicalPageOutOfRange {
                asset,
                physical_page,
            } => write!(
                formatter,
                "CLST asset {asset} addresses physical page {physical_page} outside its pool"
            ),
            Self::PageWordsMismatch {
                asset,
                expected,
                actual,
            } => write!(
                formatter,
                "CLST asset {asset} supplied {actual} page words, expected {expected}"
            ),
            Self::PageTableIndexOutOfRange { asset, page_index } => write!(
                formatter,
                "CLST asset {asset} addresses page-table row {page_index} outside its table"
            ),
            Self::QueueCompletionUnconfirmed => formatter.write_str(
                "CLST GPU buffers remain live because queue completion was not confirmed",
            ),
        }
    }
}

impl std::error::Error for VirtualGeometryGpuStoreError {}

impl From<VirtualGeometryPublicationCoordinatorError> for VirtualGeometryGpuStoreError {
    fn from(error: VirtualGeometryPublicationCoordinatorError) -> Self {
        Self::Publication(error)
    }
}

struct RegisteredVirtualGeometry<I>
where
    I: Instance + Send + Sync + 'static,
    I::Adapter: Send + Sync + 'static,
    I::Device: Send + Sync + 'static,
    I::Queue: Send + Sync + FrameSync + 'static,
    I::Surface: Send + Sync + 'static,
{
    coordinator: VirtualGeometryPublicationCoordinator,
    buffers: VirtualGeometryGpuBuffers,
    physical_capacity: u32,
    page_size: u32,
    page_words: u32,
    logical_page_count: u32,
    zero_page: Vec<u8>,
    marker: core::marker::PhantomData<fn() -> I>,
}

/// Device-owned CLST publication buffers for the render world.
pub struct VirtualGeometryGpuPublicationStore<I>
where
    I: Instance + Send + Sync + 'static,
    I::Adapter: Send + Sync + 'static,
    I::Device: Send + Sync + 'static,
    I::Queue: Send + Sync + FrameSync + 'static,
    I::Surface: Send + Sync + 'static,
{
    device: Arc<I::Device>,
    assets: BTreeMap<u32, RegisteredVirtualGeometry<I>>,
}

impl<I> Resource for VirtualGeometryGpuPublicationStore<I>
where
    I: Instance + Send + Sync + 'static,
    I::Adapter: Send + Sync + 'static,
    I::Device: Send + Sync + 'static,
    I::Queue: Send + Sync + FrameSync + 'static,
    I::Surface: Send + Sync + 'static,
{
}

impl<I> FromWorld for VirtualGeometryGpuPublicationStore<I>
where
    I: Instance + Send + Sync + 'static,
    I::Adapter: Send + Sync + 'static,
    I::Device: Send + Sync + 'static,
    I::Queue: Send + Sync + FrameSync + 'static,
    I::Surface: Send + Sync + 'static,
{
    fn from_world(world: &mut prism_ecs::world::World) -> Self {
        let runtime = world.resource::<RenderRuntimeResource<I>>();
        Self {
            device: runtime.runtime().device_handle(),
            assets: BTreeMap::new(),
        }
    }
}

impl<I> Drop for VirtualGeometryGpuPublicationStore<I>
where
    I: Instance + Send + Sync + 'static,
    I::Adapter: Send + Sync + 'static,
    I::Device: Send + Sync + 'static,
    I::Queue: Send + Sync + FrameSync + 'static,
    I::Surface: Send + Sync + 'static,
{
    fn drop(&mut self) {
        for registered in self.assets.values() {
            self.device.destroy_buffer(registered.buffers.page_storage);
            self.device.destroy_buffer(registered.buffers.page_table);
            self.device.destroy_buffer(registered.buffers.cluster_table);
        }
    }
}

impl<I> VirtualGeometryGpuPublicationStore<I>
where
    I: Instance + Send + Sync + 'static,
    I::Adapter: Send + Sync + 'static,
    I::Device: Send + Sync + 'static,
    I::Queue: Send + Sync + FrameSync + 'static,
    I::Surface: Send + Sync + 'static,
{
    /// Registers one validated CLST asset and uploads its immutable tables.
    pub fn register(
        &mut self,
        asset: Arc<VirtualGeometryAsset>,
        content_revision: [u8; 32],
        generation: u64,
        physical_capacity: u32,
        resident_budget: usize,
        material_bindings: &[VirtualGeometryMaterialBinding],
        queue: &dyn RenderQueue,
    ) -> Result<VirtualGeometryGpuBuffers, VirtualGeometryGpuStoreError> {
        let asset_id = asset.header.asset_id;
        if self.assets.contains_key(&asset_id) {
            return Err(VirtualGeometryGpuStoreError::DuplicateAsset(asset_id));
        }
        if physical_capacity == 0 {
            return Err(VirtualGeometryGpuStoreError::ZeroPhysicalCapacity);
        }

        let coordinator = VirtualGeometryPublicationCoordinator::new(
            Arc::clone(&asset),
            content_revision,
            generation,
            physical_capacity,
            resident_budget,
            material_bindings,
        )?;
        let publication = coordinator.publication();
        let page_size = asset.header.page_size;
        let page_storage_size = u64::from(page_size)
            .checked_mul(u64::from(physical_capacity))
            .ok_or(VirtualGeometryGpuStoreError::BufferSizeOverflow)?;
        // The page-table ABI stores byte offsets in u32. Reject a pool whose
        // last physical slot cannot be represented before allocating buffers.
        let last_page_offset = u64::from(page_size)
            .checked_mul(u64::from(physical_capacity - 1))
            .ok_or(VirtualGeometryGpuStoreError::BufferSizeOverflow)?;
        if last_page_offset > u64::from(u32::MAX) {
            return Err(VirtualGeometryGpuStoreError::BufferSizeOverflow);
        }
        let page_table_size = buffer_size(publication.page_table_bytes().len())?;
        let cluster_table_size = buffer_size(publication.cluster_table_bytes().len())?;
        let logical_page_count = u32::try_from(publication.pages().len())
            .map_err(|_| VirtualGeometryGpuStoreError::BufferSizeOverflow)?;
        let zero_page = vec![
            0;
            usize::try_from(page_size)
                .map_err(|_| VirtualGeometryGpuStoreError::BufferSizeOverflow)?
        ];
        let buffers = VirtualGeometryGpuBuffers {
            page_storage: self.device.create_buffer(&BufferDescriptor {
                label: Some(format!("CLST {asset_id} page storage")),
                size: page_storage_size,
                usage: BufferUsages::COPY_DST
                    .union(BufferUsages::COPY_SRC)
                    .union(BufferUsages::STORAGE),
                mapped_at_creation: false,
            }),
            page_table: self.device.create_buffer(&BufferDescriptor {
                label: Some(format!("CLST {asset_id} page table")),
                size: page_table_size,
                usage: BufferUsages::COPY_DST
                    .union(BufferUsages::COPY_SRC)
                    .union(BufferUsages::STORAGE),
                mapped_at_creation: false,
            }),
            cluster_table: self.device.create_buffer(&BufferDescriptor {
                label: Some(format!("CLST {asset_id} Cluster table")),
                size: cluster_table_size,
                usage: BufferUsages::COPY_DST
                    .union(BufferUsages::COPY_SRC)
                    .union(BufferUsages::STORAGE),
                mapped_at_creation: false,
            }),
        };
        for physical_page in 0..physical_capacity {
            let offset = u64::from(physical_page)
                .checked_mul(u64::from(page_size))
                .ok_or(VirtualGeometryGpuStoreError::BufferSizeOverflow)?;
            queue.write_buffer(buffers.page_storage, offset, &zero_page);
        }
        queue.write_buffer(buffers.page_table, 0, publication.page_table_bytes());
        if !publication.cluster_table_bytes().is_empty() {
            queue.write_buffer(buffers.cluster_table, 0, publication.cluster_table_bytes());
        }

        let page_words = asset.header.page_size / 4;
        let registered = RegisteredVirtualGeometry {
            coordinator,
            buffers,
            physical_capacity,
            page_size,
            page_words,
            logical_page_count,
            zero_page,
            marker: core::marker::PhantomData,
        };
        self.assets.insert(asset_id, registered);
        Ok(buffers)
    }

    /// Returns the buffers for a registered asset.
    #[must_use]
    pub fn buffers(&self, asset_id: u32) -> Option<VirtualGeometryGpuBuffers> {
        self.assets.get(&asset_id).map(|asset| asset.buffers)
    }

    /// Returns the CPU publication mirrored by the GPU tables.
    #[must_use]
    pub fn publication(&self, asset_id: u32) -> Option<&VirtualGeometryGpuPublication> {
        self.assets
            .get(&asset_id)
            .map(|asset| asset.coordinator.publication())
    }

    /// Stable asset ids currently owned by this device store.
    #[must_use]
    pub fn asset_identities(&self) -> Vec<u32> {
        self.assets.keys().copied().collect()
    }

    /// Waits for all queued work before releasing every registered buffer.
    ///
    /// On an unconfirmed wait the store is left entirely unchanged so the
    /// caller can retry after device recovery or another completion check.
    /// Dropping the store remains a last-resort owner teardown; normal renderer
    /// shutdown should use this method while the queue is still available.
    pub fn retire(&mut self, queue: &impl FrameSync) -> Result<(), VirtualGeometryGpuStoreError> {
        if !queue.wait_idle() {
            return Err(VirtualGeometryGpuStoreError::QueueCompletionUnconfirmed);
        }
        for registered in self.assets.values() {
            self.device.destroy_buffer(registered.buffers.page_storage);
            self.device.destroy_buffer(registered.buffers.page_table);
            self.device.destroy_buffer(registered.buffers.cluster_table);
        }
        self.assets.clear();
        Ok(())
    }

    /// Reconciles one request batch and queues the minimal storage/table writes.
    ///
    /// All page-storage writes are emitted before any page-table row update.
    /// `RenderQueue` guarantees this order for later submissions, so a page is
    /// never made visible through the new table row before its bytes are queued.
    pub fn reconcile(
        &mut self,
        asset_id: u32,
        requests: &RequestBatch<GeometryPageKey>,
        frame: u64,
        queue: &dyn RenderQueue,
    ) -> Result<VirtualGeometryPublicationDelta, VirtualGeometryGpuStoreError> {
        let registered = self
            .assets
            .get_mut(&asset_id)
            .ok_or(VirtualGeometryGpuStoreError::UnknownAsset(asset_id))?;
        let delta = registered.coordinator.reconcile(requests, frame)?;
        validate_delta(
            asset_id,
            &delta,
            registered.physical_capacity,
            registered.page_size,
            registered.page_words,
            registered.logical_page_count,
        )?;

        for eviction in &delta.evicted {
            let offset = u64::from(eviction.physical_page)
                .checked_mul(u64::from(registered.page_words) * 4)
                .ok_or(VirtualGeometryGpuStoreError::BufferSizeOverflow)?;
            queue.write_buffer(
                registered.buffers.page_storage,
                offset,
                &registered.zero_page,
            );
        }
        for upload in &delta.streamed_in {
            let offset = u64::from(upload.physical_page)
                .checked_mul(u64::from(registered.page_words) * 4)
                .ok_or(VirtualGeometryGpuStoreError::BufferSizeOverflow)?;
            queue.write_buffer(
                registered.buffers.page_storage,
                offset,
                cast_slice(&upload.words),
            );
        }
        for update in &delta.page_table_updates {
            let offset = u64::from(update.page_index)
                .checked_mul(core::mem::size_of::<
                    prism_render_architecture::virtual_geometry::VirtualGeometryGpuPage,
                >() as u64)
                .ok_or(VirtualGeometryGpuStoreError::BufferSizeOverflow)?;
            queue.write_buffer(registered.buffers.page_table, offset, bytes_of(&update.row));
        }
        Ok(delta)
    }
}

fn buffer_size(bytes: usize) -> Result<u64, VirtualGeometryGpuStoreError> {
    u64::try_from(bytes.max(4)).map_err(|_| VirtualGeometryGpuStoreError::BufferSizeOverflow)
}

fn validate_delta(
    asset: u32,
    delta: &VirtualGeometryPublicationDelta,
    physical_capacity: u32,
    page_size: u32,
    page_words: u32,
    logical_page_count: u32,
) -> Result<(), VirtualGeometryGpuStoreError> {
    for upload in &delta.streamed_in {
        if upload.physical_page >= physical_capacity {
            return Err(VirtualGeometryGpuStoreError::PhysicalPageOutOfRange {
                asset,
                physical_page: upload.physical_page,
            });
        }
        if upload.words.len() != page_words as usize {
            return Err(VirtualGeometryGpuStoreError::PageWordsMismatch {
                asset,
                expected: page_words,
                actual: upload.words.len(),
            });
        }
        let expected_offset = u64::from(upload.physical_page)
            .checked_mul(u64::from(page_size))
            .ok_or(VirtualGeometryGpuStoreError::BufferSizeOverflow)?;
        if u64::from(upload.byte_offset) != expected_offset
            || upload.byte_len == 0
            || upload.byte_len > page_size
        {
            return Err(VirtualGeometryGpuStoreError::PageWordsMismatch {
                asset,
                expected: page_words,
                actual: upload.words.len(),
            });
        }
    }
    for eviction in &delta.evicted {
        if eviction.physical_page >= physical_capacity {
            return Err(VirtualGeometryGpuStoreError::PhysicalPageOutOfRange {
                asset,
                physical_page: eviction.physical_page,
            });
        }
    }
    for update in &delta.page_table_updates {
        if update.page_index >= logical_page_count {
            return Err(VirtualGeometryGpuStoreError::PageTableIndexOutOfRange {
                asset,
                page_index: update.page_index,
            });
        }
        let generation =
            u64::from(update.row.generation[0]) | (u64::from(update.row.generation[1]) << 32);
        if generation != delta.generation {
            return Err(VirtualGeometryGpuStoreError::PageTableIndexOutOfRange {
                asset,
                page_index: update.page_index,
            });
        }
        if update.row.is_resident()
            && (update.row.physical_page >= physical_capacity
                || update.row.byte_len == 0
                || update.row.byte_len > page_size)
        {
            return Err(VirtualGeometryGpuStoreError::PhysicalPageOutOfRange {
                asset,
                physical_page: update.row.physical_page,
            });
        }
    }
    Ok(())
}
