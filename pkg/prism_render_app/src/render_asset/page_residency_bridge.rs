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
use super::{GpuPrepareBudget, PrepareCancellation, PrepareContext};

/// Errors raised while connecting a resident CPU page to the GPU pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PageResidencyBridgeError {
    /// The CPU page is not currently resident or its bytes were discarded.
    PageNotResident(PageId),
    /// The CPU and GPU pools use different fixed page sizes.
    PageSizeMismatch { cpu: u64, gpu: u64 },
    /// The GPU pool rejected the transaction.
    Pool(PagePoolError),
    /// No GPU allocation is tracked for the requested CPU page.
    UnknownPage(PageId),
}

/// A narrow, explicit CPU-page to GPU-page bridge.
pub struct FixedPageResidencyBridge {
    pool: FixedPagePool,
    allocations: BTreeMap<PageId, PageAllocation>,
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

    /// Reclaims all retired GPU page ranges whose completion proof is ready.
    pub fn reclaim_completed(
        &mut self,
        completed: GpuCompletionValue,
    ) -> Result<Vec<FixedPageReclaimEvent>, PageResidencyBridgeError> {
        self.pool
            .reclaim_completed(completed)
            .map_err(PageResidencyBridgeError::Pool)
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
        Ok(event)
    }

    /// Destroys the backing buffer. Call this from the owning render asset's
    /// unload path after all page allocations have been released.
    pub fn destroy(&self, device: &dyn RenderDevice) {
        self.pool.destroy(device);
    }
}
