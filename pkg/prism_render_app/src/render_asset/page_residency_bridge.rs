//! Narrow bridge from CPU fixed-page residency to the render-side page pool.
//!
//! The bridge consumes only pages that the asset side has already read and
//! marked resident. It does not schedule I/O, decode bytes, infer feedback, or
//! reclaim GPU pages after a fence; those policies remain with their owners.

use alloc::collections::BTreeMap;

use prism_asset::{PageId, PageResidency};
use prism_render_driver::{RenderDevice, RenderQueue};

use super::page_pool::{
    FixedPagePool, FixedPagePoolDescriptor, FixedPagePoolEvent, FixedPageUpload, PageAllocation,
    PagePoolError,
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
