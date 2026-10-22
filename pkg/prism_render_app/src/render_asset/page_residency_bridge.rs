//! Narrow bridge from CPU fixed-page residency to the render-side page pool.
//!
//! The bridge consumes only pages that the asset side has already read and
//! marked resident. It does not schedule I/O, decode bytes, or infer feedback.
//! Deferred GPU reclaim is explicit: the submission owner supplies a completion
//! value, and the bridge only forwards that proof to the page pool.

use alloc::collections::BTreeMap;

use prism_asset::{PageId, PageResidency};
use prism_render_architecture::gpu_scene::GpuCompletionValue;
use prism_render_driver::{RenderDevice, RenderQueue};

use super::page_pool::{
    FixedPageDeferredReplacement, FixedPagePool, FixedPagePoolDescriptor, FixedPagePoolEvent,
    FixedPageReclaimEvent, FixedPageUpload, PageAllocation, PagePoolError,
};
use super::page_streaming::FixedPageSlotKey;
use super::{GpuPrepareBudget, PrepareCancellation, PrepareContext};

/// Errors raised while connecting a resident CPU page to the GPU pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PageResidencyBridgeError {
    /// The active runtime does not own this GPU pool.
    DeviceMismatch,
    /// The CPU page is not currently resident or its bytes were discarded.
    PageNotResident(PageId),
    /// The CPU and GPU pools use different fixed page sizes.
    PageSizeMismatch { cpu: u64, gpu: u64 },
    /// The GPU pool rejected the transaction.
    Pool(PagePoolError),
    /// No GPU allocation is tracked for the requested CPU page.
    UnknownPage(PageId),
    /// A CPU identity already belongs to another logical slot.
    PageSlotConflict(PageId),
    /// An older upload or mismatched release revision targeted a published slot.
    StaleRevision {
        /// The revision currently published for the slot.
        current: u64,
        /// The revision carried by the rejected payload.
        incoming: u64,
    },
}

/// A narrow, explicit CPU-page to GPU-page bridge.
pub struct FixedPageResidencyBridge {
    pool: FixedPagePool,
    allocations: BTreeMap<PageId, PageAllocation>,
    slot_allocations: BTreeMap<FixedPageSlotKey, PageAllocation>,
    slot_revisions: BTreeMap<FixedPageSlotKey, u64>,
    page_slots: BTreeMap<PageId, FixedPageSlotKey>,
}

impl FixedPageResidencyBridge {
    /// Creates a bridge whose backing buffer participates in the prepare
    /// transaction supplied by the caller.
    pub fn new(
        context: &mut PrepareContext<'_>,
        descriptor: &FixedPagePoolDescriptor,
    ) -> Result<Self, PagePoolError> {
        Ok(Self {
            pool: FixedPagePool::new(context, descriptor)?,
            allocations: BTreeMap::new(),
            slot_allocations: BTreeMap::new(),
            slot_revisions: BTreeMap::new(),
            page_slots: BTreeMap::new(),
        })
    }

    /// Creates a bridge with a standalone prepare budget for convenience in
    /// tests or small render-owned resource constructors.
    pub fn new_with_budget(
        device: &dyn RenderDevice,
        queue: &dyn RenderQueue,
        descriptor: &FixedPagePoolDescriptor,
        budget: &mut GpuPrepareBudget,
        cancellation: &PrepareCancellation,
    ) -> Result<Self, PagePoolError> {
        let mut context = PrepareContext::new(device, queue, budget, cancellation);
        let bridge = Self::new(&mut context, descriptor)?;
        context.commit();
        Ok(bridge)
    }

    /// Returns the underlying fixed-page pool.
    #[must_use]
    pub fn pool(&self) -> &FixedPagePool {
        &self.pool
    }

    /// Returns the allocation currently published for a CPU page.
    #[must_use]
    pub fn allocation(&self, page: PageId) -> Option<PageAllocation> {
        self.allocations.get(&page).copied()
    }

    /// Returns a committed allocation only when its source revision matches.
    /// Retiring allocations are deliberately absent from this view.
    #[must_use]
    pub fn allocation_for_slot(
        &self,
        slot: &FixedPageSlotKey,
        revision: u64,
    ) -> Option<PageAllocation> {
        (self.slot_revisions.get(slot).copied() == Some(revision))
            .then(|| self.slot_allocations.get(slot).copied())
            .flatten()
    }

    /// Uploads one already-resident CPU page, replacing its prior GPU range
    /// atomically when one exists.
    ///
    /// This compatibility API performs immediate replacement. Callers that
    /// may still have in-flight consumers must use
    /// [`Self::upload_page_deferred`] and provide the actual completion value.
    pub fn upload_page(
        &mut self,
        residency: &PageResidency,
        page: PageId,
        revision: u64,
        queue: &dyn RenderQueue,
    ) -> Result<FixedPagePoolEvent, PageResidencyBridgeError> {
        if residency.page_size().bytes() != self.pool.page_size() {
            return Err(PageResidencyBridgeError::PageSizeMismatch {
                cpu: residency.page_size().bytes(),
                gpu: self.pool.page_size(),
            });
        }
        let bytes = residency
            .resident_bytes(page)
            .ok_or(PageResidencyBridgeError::PageNotResident(page))?;
        let upload = FixedPageUpload::new(revision, bytes);
        let event = if let Some(previous) = self.allocations.get(&page).copied() {
            let mut transaction = self
                .pool
                .begin_replace(previous, upload)
                .map_err(PageResidencyBridgeError::Pool)?;
            transaction
                .upload(queue)
                .map_err(PageResidencyBridgeError::Pool)?;
            transaction
                .commit()
                .map_err(PageResidencyBridgeError::Pool)?
        } else {
            let mut transaction = self
                .pool
                .begin_upload(upload)
                .map_err(PageResidencyBridgeError::Pool)?;
            transaction
                .upload(queue)
                .map_err(PageResidencyBridgeError::Pool)?;
            transaction
                .commit()
                .map_err(PageResidencyBridgeError::Pool)?
        };
        let allocation = match event {
            FixedPagePoolEvent::Allocated { allocation, .. }
            | FixedPagePoolEvent::Replaced { allocation, .. } => allocation,
            FixedPagePoolEvent::Released { .. } => unreachable!("upload cannot release a page"),
        };
        self.allocations.insert(page, allocation);
        Ok(event)
    }

    /// Uploads a resident page and retires its previous GPU allocation until
    /// the caller-provided completion value has been observed.
    ///
    /// The completion value is deliberately supplied by the render submission
    /// owner. The bridge never guesses a frame number or fabricates a fence.
    pub fn upload_page_deferred(
        &mut self,
        residency: &PageResidency,
        page: PageId,
        revision: u64,
        reclaim_after: GpuCompletionValue,
        queue: &dyn RenderQueue,
    ) -> Result<FixedPagePoolEvent, PageResidencyBridgeError> {
        if residency.page_size().bytes() != self.pool.page_size() {
            return Err(PageResidencyBridgeError::PageSizeMismatch {
                cpu: residency.page_size().bytes(),
                gpu: self.pool.page_size(),
            });
        }
        let bytes = residency
            .resident_bytes(page)
            .ok_or(PageResidencyBridgeError::PageNotResident(page))?;
        let upload = FixedPageUpload::new(revision, bytes);
        let Some(previous) = self.allocations.get(&page).copied() else {
            return self.upload_page(residency, page, revision, queue);
        };
        let mut transaction = self
            .pool
            .begin_replace(previous, upload)
            .map_err(PageResidencyBridgeError::Pool)?;
        transaction
            .upload(queue)
            .map_err(PageResidencyBridgeError::Pool)?;
        let replacement: FixedPageDeferredReplacement = transaction
            .commit_deferred(reclaim_after)
            .map_err(PageResidencyBridgeError::Pool)?;
        self.allocations.insert(page, replacement.allocation);
        Ok(FixedPagePoolEvent::Replaced {
            previous: replacement.previous,
            allocation: replacement.allocation,
            revision: replacement.revision,
        })
    }

    /// Uploads an owned extracted page without borrowing the CPU residency
    /// manager. This is the render-world entry point for cross-world page
    /// extraction; completion remains explicit and is attached to a real queue
    /// submission by the caller.
    pub fn upload_page_bytes(
        &mut self,
        page: PageId,
        revision: u64,
        bytes: &[u8],
        reclaim_after: GpuCompletionValue,
        queue: &dyn RenderQueue,
    ) -> Result<FixedPagePoolEvent, PageResidencyBridgeError> {
        let expected = usize::try_from(self.pool.page_size()).unwrap_or(usize::MAX);
        if bytes.len() != expected {
            return Err(PageResidencyBridgeError::PageSizeMismatch {
                cpu: bytes.len() as u64,
                gpu: self.pool.page_size(),
            });
        }
        let upload = FixedPageUpload::new(revision, bytes);
        let event = if let Some(previous) = self.allocations.get(&page).copied() {
            let mut transaction = self
                .pool
                .begin_replace(previous, upload)
                .map_err(PageResidencyBridgeError::Pool)?;
            transaction
                .upload(queue)
                .map_err(PageResidencyBridgeError::Pool)?;
            let deferred = transaction
                .commit_deferred(reclaim_after)
                .map_err(PageResidencyBridgeError::Pool)?;
            FixedPagePoolEvent::Replaced {
                previous: deferred.previous,
                allocation: deferred.allocation,
                revision: deferred.revision,
            }
        } else {
            let mut transaction = self
                .pool
                .begin_upload(upload)
                .map_err(PageResidencyBridgeError::Pool)?;
            transaction
                .upload(queue)
                .map_err(PageResidencyBridgeError::Pool)?;
            transaction
                .commit()
                .map_err(PageResidencyBridgeError::Pool)?
        };
        let allocation = match event {
            FixedPagePoolEvent::Allocated { allocation, .. }
            | FixedPagePoolEvent::Replaced { allocation, .. } => allocation,
            FixedPagePoolEvent::Released { .. } => unreachable!("upload cannot release a page"),
        };
        self.allocations.insert(page, allocation);
        Ok(event)
    }

    /// Uploads an owned page into a stable logical slot, replacing an older
    /// source revision even when CPU residency assigned it a new `PageId`.
    pub fn upload_page_bytes_for_slot(
        &mut self,
        slot: &FixedPageSlotKey,
        page: PageId,
        revision: u64,
        bytes: &[u8],
        reclaim_after: GpuCompletionValue,
        queue: &dyn RenderQueue,
    ) -> Result<FixedPagePoolEvent, PageResidencyBridgeError> {
        if self.allocations.contains_key(&page) && self.page_slots.get(&page) != Some(slot) {
            return Err(PageResidencyBridgeError::PageSlotConflict(page));
        }
        if let Some(current) = self.slot_revisions.get(slot).copied()
            && revision < current
        {
            return Err(PageResidencyBridgeError::StaleRevision {
                current,
                incoming: revision,
            });
        }
        let expected = usize::try_from(self.pool.page_size()).unwrap_or(usize::MAX);
        if bytes.len() != expected {
            return Err(PageResidencyBridgeError::PageSizeMismatch {
                cpu: bytes.len() as u64,
                gpu: self.pool.page_size(),
            });
        }
        let upload = FixedPageUpload::new(revision, bytes);
        let event = if let Some(previous) = self.slot_allocations.get(slot).copied() {
            let mut transaction = self
                .pool
                .begin_replace(previous, upload)
                .map_err(PageResidencyBridgeError::Pool)?;
            transaction
                .upload(queue)
                .map_err(PageResidencyBridgeError::Pool)?;
            let deferred = transaction
                .commit_deferred(reclaim_after)
                .map_err(PageResidencyBridgeError::Pool)?;
            FixedPagePoolEvent::Replaced {
                previous: deferred.previous,
                allocation: deferred.allocation,
                revision: deferred.revision,
            }
        } else {
            let mut transaction = self
                .pool
                .begin_upload(upload)
                .map_err(PageResidencyBridgeError::Pool)?;
            transaction
                .upload(queue)
                .map_err(PageResidencyBridgeError::Pool)?;
            transaction
                .commit()
                .map_err(PageResidencyBridgeError::Pool)?
        };
        let allocation = match event {
            FixedPagePoolEvent::Allocated { allocation, .. }
            | FixedPagePoolEvent::Replaced { allocation, .. } => allocation,
            FixedPagePoolEvent::Released { .. } => unreachable!("upload cannot release a page"),
        };
        if let Some(old_page) = self
            .page_slots
            .iter()
            .find_map(|(old_page, old_slot)| (old_slot == slot).then_some(*old_page))
        {
            self.allocations.remove(&old_page);
            self.page_slots.remove(&old_page);
        }
        self.allocations.insert(page, allocation);
        self.page_slots.insert(page, slot.clone());
        self.slot_allocations.insert(slot.clone(), allocation);
        self.slot_revisions.insert(slot.clone(), revision);
        Ok(event)
    }

    /// Reclaims all retired GPU page ranges whose completion proof is ready.
    pub fn reclaim_completed(
        &mut self,
        completed: GpuCompletionValue,
    ) -> Result<Vec<FixedPageReclaimEvent>, PageResidencyBridgeError> {
        self.pool
            .reclaim_completed(completed)
            .map_err(PageResidencyBridgeError::Pool)
    }

    /// Retires the entire previous source before any identities can be reused.
    pub(super) fn release_all_deferred(
        &mut self,
        reclaim_after: GpuCompletionValue,
    ) -> Result<(), PageResidencyBridgeError> {
        while let Some((slot, &allocation)) = self.slot_allocations.first_key_value() {
            let slot = slot.clone();
            let revision = self.slot_revisions[&slot];
            self.pool
                .begin_release(allocation, revision)
                .map_err(PageResidencyBridgeError::Pool)?
                .commit_deferred(reclaim_after)
                .map_err(PageResidencyBridgeError::Pool)?;
            self.slot_allocations.remove(&slot);
            self.slot_revisions.remove(&slot);
            self.allocations.retain(|_, current| *current != allocation);
            self.page_slots.retain(|_, current| *current != slot);
        }
        while let Some((&page, &allocation)) = self.allocations.first_key_value() {
            self.pool
                .begin_release(allocation, 0)
                .map_err(PageResidencyBridgeError::Pool)?
                .commit_deferred(reclaim_after)
                .map_err(PageResidencyBridgeError::Pool)?;
            self.allocations.remove(&page);
            self.page_slots.remove(&page);
        }
        Ok(())
    }

    /// Releases the GPU range for a page after the caller's fence/consumer
    /// safety point. CPU residency is not changed by this operation.
    pub fn release_page(
        &mut self,
        page: PageId,
        revision: u64,
    ) -> Result<FixedPagePoolEvent, PageResidencyBridgeError> {
        let allocation = self
            .allocations
            .get(&page)
            .copied()
            .ok_or(PageResidencyBridgeError::UnknownPage(page))?;
        let transaction = self
            .pool
            .begin_release(allocation, revision)
            .map_err(PageResidencyBridgeError::Pool)?;
        let event = transaction
            .commit()
            .map_err(PageResidencyBridgeError::Pool)?;
        self.allocations.remove(&page);
        if let Some(slot) = self.page_slots.remove(&page) {
            if self.slot_allocations.get(&slot) == Some(&allocation) {
                self.slot_allocations.remove(&slot);
                self.slot_revisions.remove(&slot);
            }
        }
        Ok(event)
    }

    /// Retires a GPU page after CPU eviction without reusing it before the
    /// caller-provided completion value is observed.
    pub fn release_page_deferred(
        &mut self,
        page: PageId,
        revision: u64,
        reclaim_after: GpuCompletionValue,
    ) -> Result<(), PageResidencyBridgeError> {
        if let Some(slot) = self.page_slots.get(&page) {
            let current = self.slot_revisions[slot];
            if revision != current {
                return Err(PageResidencyBridgeError::StaleRevision {
                    current,
                    incoming: revision,
                });
            }
        }
        let allocation = self
            .allocations
            .get(&page)
            .copied()
            .ok_or(PageResidencyBridgeError::UnknownPage(page))?;
        self.pool
            .begin_release(allocation, revision)
            .map_err(PageResidencyBridgeError::Pool)?
            .commit_deferred(reclaim_after)
            .map_err(PageResidencyBridgeError::Pool)?;
        self.allocations.remove(&page);
        if let Some(slot) = self.page_slots.remove(&page) {
            if self.slot_allocations.get(&slot) == Some(&allocation) {
                self.slot_allocations.remove(&slot);
                self.slot_revisions.remove(&slot);
            }
        }
        Ok(())
    }

    /// Releases only the exact extracted identity; CPU page IDs may be reused
    /// when the main-world source is replaced. Stale releases cannot retire a
    /// replacement in a different logical slot or source revision.
    pub fn release_page_for_slot_deferred(
        &mut self,
        slot: &FixedPageSlotKey,
        page: PageId,
        revision: u64,
        reclaim_after: GpuCompletionValue,
    ) -> Result<(), PageResidencyBridgeError> {
        if self.page_slots.get(&page) != Some(slot) {
            return Err(PageResidencyBridgeError::UnknownPage(page));
        }
        self.release_page_deferred(page, revision, reclaim_after)
    }

    /// Destroys the backing buffer. Call this from the owning render asset's
    /// unload path after all page allocations have been released.
    pub fn destroy(&self, device: &dyn RenderDevice) {
        self.pool.destroy(device);
    }
}
