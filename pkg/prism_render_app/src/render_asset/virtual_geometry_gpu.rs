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
use prism_ecs::prelude::{FromWorld, Resource};
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

#[cfg(test)]
#[path = "virtual_geometry_gpu_store_tests.rs"]
mod store_tests;

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
