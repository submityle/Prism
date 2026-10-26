//! Lighting virtual-page residency backed by a real fixed-page GPU pool.
//!
//! The architecture page number is a virtual address only. This adapter keeps
//! an independent PageId-to-PageAllocation map and publishes the resulting
//! physical allocation in a compact GPU page table. Logical eviction and GPU
//! reclamation are deliberately separate: an old allocation stays quarantined
//! until the caller supplies a completion proof.

use alloc::collections::{BTreeMap, BTreeSet};

use prism_render_architecture::gpu_scene::GpuCompletionValue;
use prism_render_architecture::lighting::r#virtual::{
    Fallback, NonresidentReason, PageId, PageKind, PageRequest, RepEpoch, Residency, StreamError,
    VirtualPagePool,
};
use prism_render_driver::{BufferDescriptor, BufferId, BufferUsages, RenderDevice, RenderQueue};

use super::{
    FixedPagePool, FixedPagePoolDescriptor, FixedPagePoolEvent, GpuPrepareBudget, PageAllocation,
    PagePoolError, PrepareCancellation, PrepareContext, PrepareContextError,
};

const TABLE_WORDS: usize = 14;
const TABLE_ENTRY_BYTES: u64 = (TABLE_WORDS * core::mem::size_of::<u32>()) as u64;

/// Bytes for one logical lighting page and its source revision.
#[derive(Clone, Copy, Debug)]
pub struct LightingPageUpload<'a> {
    /// Virtual page addressed by the payload.
    pub page: PageId,
    /// Representation epoch authenticated by the producer.
    pub epoch: RepEpoch,
    /// Producer revision carried into the GPU table.
    pub revision: u64,
    /// Exactly one fixed page of aligned bytes.
    pub bytes: &'a [u8],
}

impl<'a> LightingPageUpload<'a> {
    /// Creates a lighting page upload descriptor.
    #[must_use]
    pub const fn new(page: PageId, epoch: RepEpoch, revision: u64, bytes: &'a [u8]) -> Self {
        Self {
            page,
            epoch,
            revision,
            bytes,
        }
    }
}

/// A committed synchronization result.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LightingResidencySync {
    /// Number of new or replacement physical-page uploads.
    pub uploads: usize,
    /// Number of logical pages whose physical allocation was retired.
    pub releases: usize,
    /// Number of table entries written to the GPU.
    pub table_updates: usize,
    /// Number of deferred allocations reclaimed by the completion proof.
    pub reclaimed: usize,
}

/// Errors from validation, logical residency, publication, or lifecycle use.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LightingVirtualResidencyError {
    /// The page pool must be readable as a storage buffer by lighting shaders.
    MissingStorageUsage,
    /// A table with no slots cannot publish any virtual address.
    InvalidTableCapacity,
    /// The adapter was used after its GPU resources were retired.
    ResourcesUnavailable,
    /// The caller used a different device generation without recovery.
    DeviceGenerationMismatch { current: u64, incoming: u64 },
    /// A recovery operation was requested while old resources are still live.
    RecoveryResourcesStillLive,
    /// A logical resident page needs a payload for its first upload or epoch change.
    MissingPayload { page: PageId, epoch: RepEpoch },
    /// Two payloads attempted to publish the same virtual page in one batch.
    DuplicatePayload(PageId),
    /// A payload is not exactly one fixed page.
    PayloadSize { expected: u64, actual: u64 },
    /// The table has no free slot for a new virtual identity.
    TableFull,
    /// A GPU pool transaction rejected the operation.
    Pool(PagePoolError),
    /// Context allocation was rejected before GPU publication.
    Context(PrepareContextError),
    /// The architecture pool rejected the request batch.
    Logical(StreamError),
}

impl From<PagePoolError> for LightingVirtualResidencyError {
    fn from(value: PagePoolError) -> Self {
        Self::Pool(value)
    }
}

impl From<PrepareContextError> for LightingVirtualResidencyError {
    fn from(value: PrepareContextError) -> Self {
        Self::Context(value)
    }
}

impl From<StreamError> for LightingVirtualResidencyError {
    fn from(value: StreamError) -> Self {
        Self::Logical(value)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PublishedPage {
    allocation: PageAllocation,
    epoch: RepEpoch,
    revision: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct TableRecord {
    words: [u32; TABLE_WORDS],
}

/// GPU adapter for architecture-level lighting virtual residency.
pub struct LightingVirtualResidencyGpuAdapter {
    page_descriptor: FixedPagePoolDescriptor,
    table_capacity: u32,
    pool: Option<FixedPagePool>,
    table: Option<BufferId>,
    allocations: BTreeMap<PageId, PublishedPage>,
    slots: BTreeMap<PageId, u32>,
    entries: BTreeMap<PageId, TableRecord>,
    dirty_slots: BTreeSet<u32>,
    device_generation: u64,
}

impl LightingVirtualResidencyGpuAdapter {
    /// Creates the adapter on device generation zero.
    pub fn new(
        context: &mut PrepareContext<'_>,
        page_descriptor: FixedPagePoolDescriptor,
        table_capacity: u32,
    ) -> Result<Self, LightingVirtualResidencyError> {
        Self::new_for_generation(context, page_descriptor, table_capacity, 0)
    }

    /// Creates the adapter for an explicitly identified device generation.
    pub fn new_for_generation(
        context: &mut PrepareContext<'_>,
        page_descriptor: FixedPagePoolDescriptor,
        table_capacity: u32,
        device_generation: u64,
    ) -> Result<Self, LightingVirtualResidencyError> {
        if table_capacity == 0 {
            return Err(LightingVirtualResidencyError::InvalidTableCapacity);
        }
        if !page_descriptor.usage.contains(BufferUsages::STORAGE) {
            return Err(LightingVirtualResidencyError::MissingStorageUsage);
        }
        let (pool, table) = create_resources(context, &page_descriptor, table_capacity)?;
        Ok(Self {
            page_descriptor,
            table_capacity,
            pool: Some(pool),
            table: Some(table),
            allocations: BTreeMap::new(),
            slots: BTreeMap::new(),
            entries: BTreeMap::new(),
            dirty_slots: BTreeSet::new(),
            device_generation,
        })
    }

    /// Returns the backing page buffer, if the device generation is live.
    #[must_use]
    pub fn page_buffer(&self) -> Option<BufferId> {
        self.pool.as_ref().map(FixedPagePool::buffer)
    }

    /// Returns the GPU page-table buffer, if the device generation is live.
    #[must_use]
    pub const fn table_buffer(&self) -> Option<BufferId> {
        self.table
    }

    /// Returns the physical allocation owned by this adapter for a virtual page.
    #[must_use]
    pub fn allocation(&self, page: PageId) -> Option<PageAllocation> {
        self.allocations
            .get(&page)
            .map(|published| published.allocation)
    }

    /// Returns the encoded table entry currently staged for a virtual page.
    #[must_use]
    pub fn table_entry(&self, page: PageId) -> Option<[u32; TABLE_WORDS]> {
        self.entries.get(&page).map(|entry| entry.words)
    }

    /// Returns the current device generation.
    #[must_use]
    pub const fn device_generation(&self) -> u64 {
        self.device_generation
    }

    /// Returns the configured table slot count.
    #[must_use]
    pub const fn table_capacity(&self) -> u32 {
        self.table_capacity
    }

    /// Synchronizes logical requests, payloads, physical allocations, and the
    /// dirty portion of the GPU table in one ordered operation.
    pub fn sync<'a>(
        &mut self,
        logical: &mut VirtualPagePool,
        requests: &[PageRequest],
        uploads: &[LightingPageUpload<'a>],
        reclaim_after: GpuCompletionValue,
        completed: GpuCompletionValue,
        device_generation: u64,
        queue: &dyn RenderQueue,
    ) -> Result<LightingResidencySync, LightingVirtualResidencyError> {
        self.ensure_generation(device_generation)?;
        self.ensure_live()?;
        let reclaimed = self.reclaim_completed(completed)?;
        let outcomes = logical.request(requests)?;
        let mut payloads = BTreeMap::new();
        for upload in uploads {
            if payloads.insert(upload.page, upload).is_some() {
                return Err(LightingVirtualResidencyError::DuplicatePayload(upload.page));
            }
            if upload.bytes.len() as u64 != self.page_descriptor.page_size {
                return Err(LightingVirtualResidencyError::PayloadSize {
                    expected: self.page_descriptor.page_size,
                    actual: upload.bytes.len() as u64,
                });
            }
        }

        let mut outcome_map = BTreeMap::new();
        let mut protected = BTreeSet::new();
        for (page, residency) in outcomes {
            if matches!(residency, Residency::Resident { .. }) {
                protected.insert(page);
            }
            outcome_map.insert(page, residency);
        }
        for (page, residency) in &outcome_map {
            if let Residency::Resident { epoch, .. } = residency {
                let needs_upload = self
                    .allocations
                    .get(page)
                    .is_none_or(|published| published.epoch != *epoch);
                if needs_upload {
                    let Some(payload) = payloads.get(page) else {
                        return Err(LightingVirtualResidencyError::MissingPayload {
                            page: *page,
                            epoch: *epoch,
                        });
                    };
                    if payload.epoch != *epoch {
                        return Err(LightingVirtualResidencyError::MissingPayload {
                            page: *page,
                            epoch: *epoch,
                        });
                    }
                }
            }
        }

        let mut report = LightingResidencySync {
            reclaimed,
            ..LightingResidencySync::default()
        };
        self.retire_logically_nonresident(logical, &protected, reclaim_after, &mut report)?;
        for (page, residency) in outcome_map {
            match residency {
                Residency::Resident { epoch, .. } => {
                    if self
                        .allocations
                        .get(&page)
                        .is_none_or(|published| published.epoch != epoch)
                    {
                        let payload = payloads
                            .get(&page)
                            .expect("resident payload validated before publication");
                        self.upload_page(page, **payload, reclaim_after, queue)?;
                        report.uploads += 1;
                    }
                    self.set_resident_entry(page, epoch)?;
                }
                Residency::Nonresident { fallback, reason } => {
                    self.set_nonresident_entry(page, fallback, reason)?;
                }
            }
        }
        report.table_updates = self.publish_dirty(queue)?;
        Ok(report)
    }

    /// Reclaims only allocations covered by the caller's completion proof.
    pub fn reclaim_completed(
        &mut self,
        completed: GpuCompletionValue,
    ) -> Result<usize, LightingVirtualResidencyError> {
        let pool = self
            .pool
            .as_mut()
            .ok_or(LightingVirtualResidencyError::ResourcesUnavailable)?;
        Ok(pool.reclaim_completed(completed)?.len())
    }

    /// Marks the old device lost and destroys resources owned by that device.
    /// Logical residency is cleared through the architecture pool; bindings
    /// remain available for a later snapshot restore.
    pub fn handle_device_lost(
        &mut self,
        logical: &mut VirtualPagePool,
        old_device: &dyn RenderDevice,
        next_generation: u64,
    ) {
        logical.device_lost();
        if let Some(pool) = self.pool.take() {
            pool.destroy(old_device);
        }
        if let Some(table) = self.table.take() {
            old_device.destroy_buffer(table);
        }
        self.allocations.clear();
        self.device_generation = next_generation;
        let pages: Vec<PageId> = self.entries.keys().copied().collect();
        for page in pages {
            let _ = self.set_nonresident_entry(
                page,
                fallback_for(page.kind),
                NonresidentReason::DeviceLost,
            );
        }
    }

    /// Recreates the physical pool and page table after handle_device_lost.
    pub fn recover(
        &mut self,
        context: &mut PrepareContext<'_>,
        device_generation: u64,
    ) -> Result<usize, LightingVirtualResidencyError> {
        self.ensure_generation(device_generation)?;
        if self.pool.is_some() || self.table.is_some() {
            return Err(LightingVirtualResidencyError::RecoveryResourcesStillLive);
        }
        let (pool, table) = create_resources(context, &self.page_descriptor, self.table_capacity)?;
        self.pool = Some(pool);
        self.table = Some(table);
        let updates = self.publish_dirty(context.queue())?;
        Ok(updates)
    }

    /// Publishes staged entries only; unchanged slots cause no queue writes.
    pub fn publish_dirty(
        &mut self,
        queue: &dyn RenderQueue,
    ) -> Result<usize, LightingVirtualResidencyError> {
        if self.table.is_none() {
            return Err(LightingVirtualResidencyError::ResourcesUnavailable);
        }
        let dirty: Vec<u32> = self.dirty_slots.iter().copied().collect();
        for slot in &dirty {
            let entry = self
                .entries
                .values()
                .zip(self.slots.values())
                .find_map(|(entry, current)| (*current == *slot).then_some(*entry))
                .ok_or(LightingVirtualResidencyError::ResourcesUnavailable)?;
            queue.write_buffer(
                self.table.expect("table checked above"),
                u64::from(*slot) * TABLE_ENTRY_BYTES,
                bytemuck::cast_slice(&entry.words),
            );
        }
        for slot in &dirty {
            self.dirty_slots.remove(slot);
        }
        Ok(dirty.len())
    }

    /// Releases all live resources on the supplied device. Idempotent.
    pub fn destroy(&mut self, device: &dyn RenderDevice) {
        if let Some(pool) = self.pool.take() {
            pool.destroy(device);
        }
        if let Some(table) = self.table.take() {
            device.destroy_buffer(table);
        }
        self.allocations.clear();
    }

    fn ensure_generation(&self, incoming: u64) -> Result<(), LightingVirtualResidencyError> {
        if incoming == self.device_generation {
            Ok(())
        } else {
            Err(LightingVirtualResidencyError::DeviceGenerationMismatch {
                current: self.device_generation,
                incoming,
            })
        }
    }

    fn ensure_live(&self) -> Result<(), LightingVirtualResidencyError> {
        (self.pool.is_some() && self.table.is_some())
            .then_some(())
            .ok_or(LightingVirtualResidencyError::ResourcesUnavailable)
    }

    fn retire_logically_nonresident(
        &mut self,
        logical: &VirtualPagePool,
        protected: &BTreeSet<PageId>,
        reclaim_after: GpuCompletionValue,
        report: &mut LightingResidencySync,
    ) -> Result<(), LightingVirtualResidencyError> {
        let pages: Vec<(PageId, PublishedPage)> = self
            .allocations
            .iter()
            .filter_map(|(page, published)| {
                (!protected.contains(page)).then_some((*page, *published))
            })
            .collect();
        for (page, published) in pages {
            if matches!(
                logical.lookup(page, published.epoch),
                Residency::Resident { .. }
            ) {
                continue;
            }
            let pool = self
                .pool
                .as_mut()
                .ok_or(LightingVirtualResidencyError::ResourcesUnavailable)?;
            pool.begin_release(published.allocation, published.revision)?
                .commit_deferred(reclaim_after)?;
            self.allocations.remove(&page);
            report.releases += 1;
            if let Residency::Nonresident { fallback, reason } =
                logical.lookup(page, published.epoch)
            {
                self.set_nonresident_entry(page, fallback, reason)?;
            }
        }
        Ok(())
    }

    fn upload_page(
        &mut self,
        page: PageId,
        upload: LightingPageUpload<'_>,
        reclaim_after: GpuCompletionValue,
        queue: &dyn RenderQueue,
    ) -> Result<(), LightingVirtualResidencyError> {
        let pool = self
            .pool
            .as_mut()
            .ok_or(LightingVirtualResidencyError::ResourcesUnavailable)?;
        let event = if let Some(previous) = self.allocations.get(&page).copied() {
            let mut transaction = pool.begin_replace(
                previous.allocation,
                super::FixedPageUpload::new(upload.revision, upload.bytes),
            )?;
            transaction.upload(queue)?;
            let replacement = transaction.commit_deferred(reclaim_after)?;
            FixedPagePoolEvent::Replaced {
                previous: replacement.previous,
                allocation: replacement.allocation,
                revision: replacement.revision,
            }
        } else {
            let mut transaction =
                pool.begin_upload(super::FixedPageUpload::new(upload.revision, upload.bytes))?;
            transaction.upload(queue)?;
            transaction.commit()?
        };
        let allocation = match event {
            FixedPagePoolEvent::Allocated { allocation, .. }
            | FixedPagePoolEvent::Replaced { allocation, .. } => allocation,
            FixedPagePoolEvent::Released { .. } => unreachable!("page upload cannot release"),
        };
        self.allocations.insert(
            page,
            PublishedPage {
                allocation,
                epoch: upload.epoch,
                revision: upload.revision,
            },
        );
        Ok(())
    }

    fn set_resident_entry(
        &mut self,
        page: PageId,
        epoch: RepEpoch,
    ) -> Result<(), LightingVirtualResidencyError> {
        let published = self
            .allocations
            .get(&page)
            .copied()
            .ok_or(LightingVirtualResidencyError::ResourcesUnavailable)?;
        let mut words = identity_words(page);
        words[4] = published.allocation.first_page();
        words[5] = epoch.0 as u32;
        words[6] = (epoch.0 >> 32) as u32;
        words[7] = published.revision as u32;
        words[8] = (published.revision >> 32) as u32;
        words[9] = 1;
        words[10] = 0;
        words[11] = 0;
        words[12] = self.device_generation as u32;
        words[13] = (self.device_generation >> 32) as u32;
        self.set_entry(page, words)
    }

    fn set_nonresident_entry(
        &mut self,
        page: PageId,
        fallback: Fallback,
        reason: NonresidentReason,
    ) -> Result<(), LightingVirtualResidencyError> {
        let mut words = identity_words(page);
        words[4] = u32::MAX;
        words[9] = 0;
        words[10] = fallback_code(fallback);
        words[11] = reason_code(reason);
        words[12] = self.device_generation as u32;
        words[13] = (self.device_generation >> 32) as u32;
        self.set_entry(page, words)
    }

    fn set_entry(
        &mut self,
        page: PageId,
        words: [u32; TABLE_WORDS],
    ) -> Result<(), LightingVirtualResidencyError> {
        let slot = if let Some(slot) = self.slots.get(&page).copied() {
            slot
        } else {
            let slot = self.slots.len() as u32;
            if slot >= self.table_capacity {
                return Err(LightingVirtualResidencyError::TableFull);
            }
            self.slots.insert(page, slot);
            slot
        };
        if self.entries.get(&page).map(|entry| entry.words) != Some(words) {
            self.entries.insert(page, TableRecord { words });
            self.dirty_slots.insert(slot);
        }
        Ok(())
    }
}

fn create_resources(
    context: &mut PrepareContext<'_>,
    page_descriptor: &FixedPagePoolDescriptor,
    table_capacity: u32,
) -> Result<(FixedPagePool, BufferId), LightingVirtualResidencyError> {
    let pool = FixedPagePool::new(context, page_descriptor)?;
    let table_size = TABLE_ENTRY_BYTES
        .checked_mul(u64::from(table_capacity))
        .ok_or(PagePoolError::SizeOverflow)?;
    let table = context.create_buffer(&BufferDescriptor {
        label: Some("prism_lighting_virtual_page_table".into()),
        size: table_size,
        usage: BufferUsages::COPY_DST | BufferUsages::STORAGE,
        mapped_at_creation: false,
    })?;
    Ok((pool, table))
}

fn identity_words(page: PageId) -> [u32; TABLE_WORDS] {
    let mut words = [0; TABLE_WORDS];
    words[0] = match page.kind {
        PageKind::Shadow => 0,
        PageKind::Gi => 1,
        PageKind::Cache => 2,
    };
    words[1] = page.rep.0 as u32;
    words[2] = (page.rep.0 >> 32) as u32;
    words[3] = page.page;
    words
}

const fn fallback_for(kind: PageKind) -> Fallback {
    match kind {
        PageKind::Shadow => Fallback::ShadowCascade,
        PageKind::Gi => Fallback::GiProbe,
        PageKind::Cache => Fallback::CacheRecompute,
    }
}

const fn fallback_code(fallback: Fallback) -> u32 {
    match fallback {
        Fallback::ShadowCascade => 0,
        Fallback::ShadowRay => 1,
        Fallback::GiProbe => 2,
        Fallback::GiCoarse => 3,
        Fallback::CacheRecompute => 4,
        Fallback::CacheUnfiltered => 5,
    }
}

const fn reason_code(reason: NonresidentReason) -> u32 {
    match reason {
        NonresidentReason::NotRequested => 0,
        NonresidentReason::Capacity => 1,
        NonresidentReason::StaleEpoch => 2,
        NonresidentReason::DeviceLost => 3,
        NonresidentReason::UnknownBinding => 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::sync::atomic::AtomicBool;
    use prism_render_architecture::lighting::r#virtual::{
        CellBinding, CellId, PagePoolConfig, PageRange, ProviderKind, RepId, RepProvider,
        RequestPriority,
    };
    use prism_render_driver::{DeviceDescriptor, Instance, RequestAdapterOptions};
    use prism_render_driver_null::{NullDevice, NullInstance, NullQueue};

    fn device() -> (NullDevice, NullQueue) {
        let instance = NullInstance::new();
        let adapter = instance
            .request_adapter(&RequestAdapterOptions::default())
            .expect("null adapter");
        instance
            .create_device(&adapter, &DeviceDescriptor::default())
            .expect("null device")
    }

    fn binding(epoch: u64) -> CellBinding {
        CellBinding {
            cell: CellId(1),
            rep: RepId(9),
            provider: RepProvider::new(ProviderKind::Generated, 9, 1),
            epoch: RepEpoch(epoch),
            shadow: Some(PageRange::new(0, 8)),
            gi: None,
            cache: None,
        }
    }

    fn request(page: u32, epoch: u64, priority: u32) -> PageRequest {
        PageRequest {
            cell: CellId(1),
            page: PageId {
                kind: PageKind::Shadow,
                rep: RepId(9),
                page,
            },
            epoch: RepEpoch(epoch),
            priority: RequestPriority(priority),
            frame: u64::from(page),
        }
    }

    fn adapter(device: &NullDevice, queue: &NullQueue) -> LightingVirtualResidencyGpuAdapter {
        let mut budget = GpuPrepareBudget::unlimited();
        let cancellation = PrepareCancellation::new(false);
        let mut context = PrepareContext::new(device, queue, &mut budget, &cancellation);
        let adapter = LightingVirtualResidencyGpuAdapter::new(
            &mut context,
            FixedPagePoolDescriptor::new(
                16,
                3,
                BufferUsages::COPY_DST | BufferUsages::COPY_SRC | BufferUsages::STORAGE,
            ),
            8,
        )
        .expect("adapter creation");
        context.commit();
        adapter
    }

    #[test]
    fn null_adapter_publishes_real_allocation_and_dirty_only_updates() {
        let (device, queue) = device();
        let mut adapter = adapter(&device, &queue);
        let mut logical = VirtualPagePool::new(PagePoolConfig { capacity: 2 }).unwrap();
        logical.bind(binding(1)).unwrap();
        let page = PageId {
            kind: PageKind::Shadow,
            rep: RepId(9),
            page: 0,
        };
        let bytes = [7_u8; 16];
        let first = adapter
            .sync(
                &mut logical,
                &[request(0, 1, 10)],
                &[LightingPageUpload::new(page, RepEpoch(1), 4, &bytes)],
                GpuCompletionValue(3),
                GpuCompletionValue(0),
                0,
                &queue,
            )
            .unwrap();
        assert_eq!(first.uploads, 1);
        assert_eq!(first.table_updates, 1);
        let allocation = adapter.allocation(page).unwrap();
        assert_eq!(
            adapter.table_entry(page).unwrap()[4],
            allocation.first_page()
        );
        let second = adapter
            .sync(
                &mut logical,
                &[request(0, 1, 10)],
                &[],
                GpuCompletionValue(3),
                GpuCompletionValue(0),
                0,
                &queue,
            )
            .unwrap();
        assert_eq!(second.uploads, 0);
        assert_eq!(second.table_updates, 0);
        adapter.destroy(&device);
        assert_eq!(device.live_resource_count(), 0);
    }

    #[test]
    fn logical_physical_index_is_not_reused_as_gpu_allocation() {
        let (device, queue) = device();
        let mut adapter = adapter(&device, &queue);
        let mut logical = VirtualPagePool::new(PagePoolConfig { capacity: 1 }).unwrap();
        logical.bind(binding(1)).unwrap();
        let first_page = PageId {
            kind: PageKind::Shadow,
            rep: RepId(9),
            page: 0,
        };
        let second_page = PageId {
            kind: PageKind::Shadow,
            rep: RepId(9),
            page: 1,
        };
        let bytes = [1_u8; 16];
        adapter
            .sync(
                &mut logical,
                &[request(0, 1, 1)],
                &[LightingPageUpload::new(first_page, RepEpoch(1), 1, &bytes)],
                GpuCompletionValue(8),
                GpuCompletionValue(0),
                0,
                &queue,
            )
            .unwrap();
        let result = adapter
            .sync(
                &mut logical,
                &[request(1, 1, 2)],
                &[LightingPageUpload::new(second_page, RepEpoch(1), 2, &bytes)],
                GpuCompletionValue(8),
                GpuCompletionValue(0),
                0,
                &queue,
            )
            .unwrap();
        assert_eq!(result.uploads, 1);
        assert_eq!(
            logical.lookup(second_page, RepEpoch(1)),
            Residency::Resident {
                physical: 0,
                epoch: RepEpoch(1)
            }
        );
        assert_eq!(adapter.allocation(second_page).unwrap().first_page(), 1);
        adapter.destroy(&device);
    }

    #[test]
    fn stale_epoch_is_rejected_before_gpu_publication() {
        let (device, queue) = device();
        let mut adapter = adapter(&device, &queue);
        let mut logical = VirtualPagePool::new(PagePoolConfig { capacity: 1 }).unwrap();
        logical.bind(binding(1)).unwrap();
        let page = PageId {
            kind: PageKind::Shadow,
            rep: RepId(9),
            page: 0,
        };
        let old = [2_u8; 16];
        adapter
            .sync(
                &mut logical,
                &[request(0, 1, 1)],
                &[LightingPageUpload::new(page, RepEpoch(1), 1, &old)],
                GpuCompletionValue(5),
                GpuCompletionValue(0),
                0,
                &queue,
            )
            .unwrap();
        logical.invalidate(RepId(9), RepEpoch(2));
        let error = adapter
            .sync(
                &mut logical,
                &[request(0, 2, 1)],
                &[LightingPageUpload::new(page, RepEpoch(1), 2, &old)],
                GpuCompletionValue(5),
                GpuCompletionValue(0),
                0,
                &queue,
            )
            .unwrap_err();
        assert_eq!(
            error,
            LightingVirtualResidencyError::MissingPayload {
                page,
                epoch: RepEpoch(2)
            }
        );
        adapter.destroy(&device);
    }

    #[test]
    fn recovery_destroys_old_resources_and_republishes_after_rebuild() {
        let (device, queue) = device();
        let mut adapter = adapter(&device, &queue);
        let mut logical = VirtualPagePool::new(PagePoolConfig { capacity: 1 }).unwrap();
        logical.bind(binding(1)).unwrap();
        let page = PageId {
            kind: PageKind::Shadow,
            rep: RepId(9),
            page: 0,
        };
        let bytes = [3_u8; 16];
        adapter
            .sync(
                &mut logical,
                &[request(0, 1, 1)],
                &[LightingPageUpload::new(page, RepEpoch(1), 1, &bytes)],
                GpuCompletionValue(5),
                GpuCompletionValue(0),
                0,
                &queue,
            )
            .unwrap();
        let snapshot = logical.snapshot();
        adapter.handle_device_lost(&mut logical, &device, 1);
        assert_eq!(device.live_resource_count(), 0);
        let cancellation = AtomicBool::new(false);
        let mut budget = GpuPrepareBudget::unlimited();
        let mut context = PrepareContext::new(&device, &queue, &mut budget, &cancellation);
        adapter.recover(&mut context, 1).unwrap();
        context.commit();
        logical.restore(&snapshot, 1).unwrap();
        adapter
            .sync(
                &mut logical,
                &[request(0, 1, 1)],
                &[LightingPageUpload::new(page, RepEpoch(1), 2, &bytes)],
                GpuCompletionValue(5),
                GpuCompletionValue(0),
                1,
                &queue,
            )
            .unwrap();
        assert_eq!(adapter.table_entry(page).unwrap()[12], 1);
        adapter.destroy(&device);
        assert_eq!(device.live_resource_count(), 0);
    }
}
