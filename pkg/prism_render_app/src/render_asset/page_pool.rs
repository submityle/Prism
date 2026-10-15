//! Transactional fixed-page GPU buffer residency.
//!
//! This module is deliberately independent of the asset crate. An asset or a
//! streaming system supplies a [`FixedPageUpload`] descriptor and receives a
//! typed [`FixedPagePoolEvent`] after the render-side transaction commits. The
//! pool owns only the RHI buffer and page bookkeeping; it does not know what
//! produced the bytes or how a consumer interprets them.
//!
//! A pool has one fixed-size GPU buffer split into contiguous pages. Uploads
//! first reserve a contiguous range, then issue a real
//! [`RenderQueue::write_buffer`] call, and only become resident when the
//! transaction is committed. Dropping an uncommitted transaction returns its
//! reservation. Replacement follows the same rule: the old range remains
//! resident until the new upload commits, so a failed validation or an
//! abandoned transaction cannot publish a half-replaced resource.

use alloc::{string::String, vec::Vec};
use core::sync::atomic::{AtomicBool, Ordering};

use prism_render_architecture::gpu_scene::GpuCompletionValue;
use prism_render_driver::{BufferDescriptor, BufferId, BufferUsages, RenderDevice, RenderQueue};

use super::{PrepareContext, PrepareContextError};

/// The descriptor for one fixed-page GPU buffer.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct FixedPagePoolDescriptor {
    /// Optional label forwarded to the RHI buffer descriptor.
    pub label: Option<String>,
    /// Size of one page in bytes. It must be non-zero and four-byte aligned.
    pub page_size: u64,
    /// Number of pages in the backing buffer.
    pub page_count: u32,
    /// GPU usages for the backing buffer. [`BufferUsages::COPY_DST`] is
    /// required because uploads use the queue write path.
    pub usage: BufferUsages,
}

impl FixedPagePoolDescriptor {
    /// Creates a fixed-page descriptor without a debug label.
    #[must_use]
    pub const fn new(page_size: u64, page_count: u32, usage: BufferUsages) -> Self {
        Self {
            label: None,
            page_size,
            page_count,
            usage,
        }
    }
}

/// A typed render-side upload request.
///
/// `revision` is opaque to the pool. It lets an asset-side producer correlate
/// a committed render event with its own source revision without making the
/// pool depend on `AssetId`, an asset server, or a particular streaming model.
#[derive(Clone, Copy, Debug)]
pub struct FixedPageUpload<'a> {
    /// Producer-defined revision carried back in the commit event.
    pub revision: u64,
    /// Bytes to write into the allocated page range.
    pub bytes: &'a [u8],
}

impl<'a> FixedPageUpload<'a> {
    /// Creates an upload request for a producer revision.
    #[must_use]
    pub const fn new(revision: u64, bytes: &'a [u8]) -> Self {
        Self { revision, bytes }
    }
}

/// A committed range in a [`FixedPagePool`].
///
/// The handle is only meaningful with the pool that created it. Its buffer id
/// and allocation token make stale or cross-pool handles fail validation
/// instead of silently releasing a newly reused range.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct PageAllocation {
    buffer: BufferId,
    first_page: u32,
    page_count: u32,
    byte_len: u64,
    token: u64,
}

impl PageAllocation {
    /// The backing RHI buffer.
    #[must_use]
    pub const fn buffer(self) -> BufferId {
        self.buffer
    }

    /// The first page in the allocation.
    #[must_use]
    pub const fn first_page(self) -> u32 {
        self.first_page
    }

    /// The number of pages reserved by the allocation.
    #[must_use]
    pub const fn page_count(self) -> u32 {
        self.page_count
    }

    /// The exact payload length supplied at allocation time.
    #[must_use]
    pub const fn byte_len(self) -> u64 {
        self.byte_len
    }

    /// Byte offset of the allocation in the backing buffer.
    #[must_use]
    pub const fn offset_bytes(self, page_size: u64) -> u64 {
        self.first_page as u64 * page_size
    }

    /// Capacity of the allocation in bytes.
    #[must_use]
    pub const fn capacity_bytes(self, page_size: u64) -> u64 {
        self.page_count as u64 * page_size
    }
}

/// Events emitted by a successful page-pool transaction.
///
/// These are typed render-side facts. A caller may translate them into its
/// own asset event, but the pool itself never imports or names an asset type.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FixedPagePoolEvent {
    /// A new allocation became resident.
    Allocated {
        /// The newly resident range.
        allocation: PageAllocation,
        /// The producer revision supplied with the upload.
        revision: u64,
    },
    /// A new allocation replaced an older resident range.
    Replaced {
        /// The range that was resident before the commit.
        previous: PageAllocation,
        /// The newly resident range.
        allocation: PageAllocation,
        /// The producer revision supplied with the upload.
        revision: u64,
    },
    /// A resident range was released after its release transaction committed.
    Released {
        /// The range that became free.
        allocation: PageAllocation,
        /// The producer revision associated with the release.
        revision: u64,
    },
}

/// The typed publication produced by a deferred replacement.
///
/// The new allocation is resident immediately, while `previous` remains
/// quarantined until `reclaim_after` is observed as completed by the caller.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct FixedPageDeferredReplacement {
    /// The allocation retained until the completion proof arrives.
    pub previous: PageAllocation,
    /// The newly resident allocation.
    pub allocation: PageAllocation,
    /// The producer revision supplied with the upload.
    pub revision: u64,
    /// The caller-provided completion value after which `previous` is safe to reclaim.
    pub reclaim_after: GpuCompletionValue,
}

/// A typed event for one prior allocation that was actually reclaimed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct FixedPageReclaimEvent {
    /// The allocation that became free.
    pub allocation: PageAllocation,
    /// The completion value attached when the replacement was published.
    pub reclaim_after: GpuCompletionValue,
    /// The observed completion value that made reclamation safe.
    pub completed: GpuCompletionValue,
}

/// The publication returned when a resident range is retired for deferred
/// release. The range remains unavailable until `reclaim_after` is observed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct FixedPageDeferredRelease {
    /// The allocation waiting for completion.
    pub allocation: PageAllocation,
    /// The producer revision associated with the release.
    pub revision: u64,
    /// The completion value that makes reclamation safe.
    pub reclaim_after: GpuCompletionValue,
}

/// Errors that prevent a page-pool operation from being submitted or
/// committed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PagePoolError {
    /// The page size is zero.
    ZeroPageSize,
    /// The page size is not four-byte aligned.
    PageSizeNotAligned {
        /// The invalid page size in bytes.
        page_size: u64,
    },
    /// The descriptor has no pages.
    ZeroPageCount,
    /// The total backing-buffer size overflowed `u64`.
    SizeOverflow,
    /// The backing buffer cannot receive queue uploads.
    MissingCopyDestination,
    /// The upload has no bytes.
    EmptyUpload,
    /// The upload length is not four-byte aligned.
    UploadNotAligned {
        /// The invalid payload length in bytes.
        byte_len: u64,
    },
    /// The requested payload does not fit in the pool's page range.
    UploadTooLarge {
        /// The requested payload length in bytes.
        byte_len: u64,
        /// The total pool capacity in bytes.
        capacity: u64,
    },
    /// No contiguous free range can satisfy the request.
    OutOfPages {
        /// The number of pages requested.
        requested: u32,
        /// The largest currently free contiguous run.
        largest_contiguous: u32,
    },
    /// The allocation does not belong to this pool or is stale.
    InvalidAllocation,
    /// The transaction is not in the state required by this operation.
    InvalidTransactionState,
    /// The backing buffer has already been destroyed.
    Destroyed,
    /// The prepare context rejected the backing-buffer allocation.
    Context(PrepareContextError),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PageState {
    Free,
    Reserved(u64),
    Uploaded(u64),
    Resident(u64),
    Releasing(u64),
    Retiring {
        token: u64,
        reclaim_after: GpuCompletionValue,
    },
}

impl PageState {
    const fn owner(self) -> Option<u64> {
        match self {
            Self::Free => None,
            Self::Reserved(owner)
            | Self::Uploaded(owner)
            | Self::Resident(owner)
            | Self::Releasing(owner)
            | Self::Retiring { token: owner, .. } => Some(owner),
        }
    }
}

/// A fixed-size GPU page pool backed by one RHI buffer.
///
/// Create the pool through [`Self::new`] during a contextual render-asset
/// prepare so the backing buffer participates in `PrepareContext` rollback.
/// The successful render asset's `unload` hook must call [`Self::destroy`] to
/// release the backing buffer.
pub struct FixedPagePool {
    buffer: BufferId,
    page_size: u64,
    page_count: u32,
    pages: Vec<PageState>,
    pending_reclaims: Vec<PendingPageReclaim>,
    next_token: u64,
    destroyed: AtomicBool,
}

impl FixedPagePool {
    /// Creates the backing buffer through the existing prepare transaction.
    pub fn new(
        context: &mut PrepareContext<'_>,
        descriptor: &FixedPagePoolDescriptor,
    ) -> Result<Self, PagePoolError> {
        validate_descriptor(descriptor)?;
        let size = descriptor
            .page_size
            .checked_mul(u64::from(descriptor.page_count))
            .ok_or(PagePoolError::SizeOverflow)?;
        let buffer = context
            .create_buffer(&BufferDescriptor {
                label: descriptor.label.clone(),
                size,
                usage: descriptor.usage,
                mapped_at_creation: false,
            })
            .map_err(PagePoolError::Context)?;

        Ok(Self {
            buffer,
            page_size: descriptor.page_size,
            page_count: descriptor.page_count,
            pages: alloc::vec![PageState::Free; descriptor.page_count as usize],
            pending_reclaims: Vec::new(),
            next_token: 1,
            destroyed: AtomicBool::new(false),
        })
    }

    /// Returns the backing buffer id for bind-group or draw descriptors.
    #[must_use]
    pub const fn buffer(&self) -> BufferId {
        self.buffer
    }

    /// Returns the fixed page size in bytes.
    #[must_use]
    pub const fn page_size(&self) -> u64 {
        self.page_size
    }

    /// Returns the total number of pages.
    #[must_use]
    pub const fn page_count(&self) -> u32 {
        self.page_count
    }

    /// Returns the number of currently free pages.
    #[must_use]
    pub fn free_page_count(&self) -> u32 {
        self.pages
            .iter()
            .filter(|state| matches!(state, PageState::Free))
            .count() as u32
    }

    /// Returns the number of resident pages.
    #[must_use]
    pub fn resident_page_count(&self) -> u32 {
        self.pages
            .iter()
            .filter(|state| matches!(state, PageState::Resident(_)))
            .count() as u32
    }

    /// Returns the number of pages quarantined for deferred reclamation.
    #[must_use]
    pub fn pending_reclaim_page_count(&self) -> u32 {
        self.pages
            .iter()
            .filter(|state| matches!(state, PageState::Retiring { .. }))
            .count() as u32
    }

    /// Returns whether the backing buffer has been released.
    #[must_use]
    pub fn is_destroyed(&self) -> bool {
        self.destroyed.load(Ordering::Acquire)
    }

    /// Begins a new upload transaction.
    pub fn begin_upload<'pool, 'bytes>(
        &'pool mut self,
        upload: FixedPageUpload<'bytes>,
    ) -> Result<PageUploadTransaction<'pool, 'bytes>, PagePoolError> {
        self.ensure_live()?;
        let allocation = self.reserve(upload.bytes.len())?;
        Ok(PageUploadTransaction {
            pool: self,
            upload,
            allocation,
            state: UploadState::Reserved,
            committed: false,
        })
    }

    /// Begins an atomic replacement of `previous` with a newly uploaded range.
    ///
    /// The old range remains resident while the new range is reserved and
    /// uploaded. The caller must commit the returned transaction to publish
    /// the replacement; dropping it leaves `previous` resident and returns the
    /// new reservation.
    pub fn begin_replace<'pool, 'bytes>(
        &'pool mut self,
        previous: PageAllocation,
        upload: FixedPageUpload<'bytes>,
    ) -> Result<PageReplaceTransaction<'pool, 'bytes>, PagePoolError> {
        self.ensure_live()?;
        self.validate_state(previous, PageStateKind::Resident)?;
        let allocation = self.reserve(upload.bytes.len())?;
        Ok(PageReplaceTransaction {
            pool: self,
            upload,
            previous,
            allocation,
            state: UploadState::Reserved,
            committed: false,
        })
    }

    /// Begins a transactional release of a resident allocation.
    pub fn begin_release<'pool>(
        &'pool mut self,
        allocation: PageAllocation,
        revision: u64,
    ) -> Result<PageReleaseTransaction<'pool>, PagePoolError> {
        self.ensure_live()?;
        self.validate_state(allocation, PageStateKind::Resident)?;
        self.set_state(allocation, PageState::Releasing(allocation.token));
        Ok(PageReleaseTransaction {
            pool: self,
            allocation,
            revision,
            committed: false,
        })
    }

    /// Reclaims prior allocations whose caller-provided completion values have
    /// been observed as complete. No wall-clock or frame-count assumption is
    /// made; the caller supplies the actual completion proof.
    pub fn reclaim_completed(
        &mut self,
        completed: GpuCompletionValue,
    ) -> Result<Vec<FixedPageReclaimEvent>, PagePoolError> {
        self.ensure_live()?;

        for pending in &self.pending_reclaims {
            if pending.reclaim_after <= completed {
                self.validate_state(pending.allocation, PageStateKind::Retiring)?;
            }
        }

        let pending = core::mem::take(&mut self.pending_reclaims);
        let mut ready = Vec::new();
        let mut deferred = Vec::with_capacity(pending.len());
        for pending in pending {
            if pending.reclaim_after <= completed {
                self.set_state(pending.allocation, PageState::Free);
                ready.push(FixedPageReclaimEvent {
                    allocation: pending.allocation,
                    reclaim_after: pending.reclaim_after,
                    completed,
                });
            } else {
                deferred.push(pending);
            }
        }
        self.pending_reclaims = deferred;
        Ok(ready)
    }

    /// Releases the backing buffer once the containing render asset is being
    /// unloaded. The operation is idempotent so an unload path can safely be
    /// retried after a device-loss cleanup pass.
    pub fn destroy(&self, device: &dyn RenderDevice) {
        if !self.destroyed.swap(true, Ordering::AcqRel) {
            device.destroy_buffer(self.buffer);
        }
    }

    fn ensure_live(&self) -> Result<(), PagePoolError> {
        if self.is_destroyed() {
            Err(PagePoolError::Destroyed)
        } else {
            Ok(())
        }
    }

    fn next_token(&mut self) -> u64 {
        let token = self.next_token;
        self.next_token = self.next_token.wrapping_add(1).max(1);
        token
    }

    fn reserve(&mut self, byte_len: usize) -> Result<PageAllocation, PagePoolError> {
        let byte_len = u64::try_from(byte_len).map_err(|_| PagePoolError::SizeOverflow)?;
        validate_payload(byte_len)?;
        let required_pages = byte_len.div_ceil(self.page_size);
        if required_pages > u64::from(self.page_count) {
            return Err(PagePoolError::UploadTooLarge {
                byte_len,
                capacity: self.page_size * u64::from(self.page_count),
            });
        }
        let page_count = u32::try_from(required_pages).map_err(|_| PagePoolError::SizeOverflow)?;
        let Some(first_page) = self.find_free_run(page_count) else {
            return Err(PagePoolError::OutOfPages {
                requested: page_count,
                largest_contiguous: self.largest_free_run(),
            });
        };
        let token = self.next_token();
        let allocation = PageAllocation {
            buffer: self.buffer,
            first_page,
            page_count,
            byte_len,
            token,
        };
        self.set_state(allocation, PageState::Reserved(token));
        Ok(allocation)
    }

    fn find_free_run(&self, requested: u32) -> Option<u32> {
        let mut run_start = 0;
        let mut run_length = 0;
        for (index, state) in self.pages.iter().enumerate() {
            if matches!(state, PageState::Free) {
                if run_length == 0 {
                    run_start = index as u32;
                }
                run_length += 1;
                if run_length >= requested {
                    return Some(run_start);
                }
            } else {
                run_length = 0;
            }
        }
        None
    }

    fn largest_free_run(&self) -> u32 {
        let mut largest = 0;
        let mut current = 0;
        for state in &self.pages {
            if matches!(state, PageState::Free) {
                current += 1;
                largest = largest.max(current);
            } else {
                current = 0;
            }
        }
        largest
    }

    fn validate_state(
        &self,
        allocation: PageAllocation,
        expected: PageStateKind,
    ) -> Result<(), PagePoolError> {
        let Some(end_page) = allocation.first_page.checked_add(allocation.page_count) else {
            return Err(PagePoolError::InvalidAllocation);
        };
        if allocation.buffer != self.buffer
            || end_page > self.page_count
            || allocation.page_count == 0
            || allocation.token == 0
        {
            return Err(PagePoolError::InvalidAllocation);
        }
        let range = allocation.first_page as usize..end_page as usize;
        if self.pages[range.clone()]
            .iter()
            .any(|state| state.owner() != Some(allocation.token) || !expected.matches(*state))
        {
            return Err(PagePoolError::InvalidAllocation);
        }
        Ok(())
    }

    fn set_state(&mut self, allocation: PageAllocation, state: PageState) {
        let range = allocation.first_page as usize
            ..(allocation.first_page + allocation.page_count) as usize;
        for page in &mut self.pages[range] {
            *page = state;
        }
    }

    fn rollback(&mut self, allocation: PageAllocation) {
        if let Some(end_page) = allocation.first_page.checked_add(allocation.page_count)
            && allocation.buffer == self.buffer
            && end_page <= self.page_count
        {
            let range = allocation.first_page as usize..end_page as usize;
            for page in &mut self.pages[range] {
                if page.owner() == Some(allocation.token) {
                    *page = PageState::Free;
                }
            }
        }
    }
}

#[derive(Clone, Copy)]
enum PageStateKind {
    Reserved,
    Uploaded,
    Resident,
    Releasing,
    Retiring,
}

impl PageStateKind {
    const fn matches(self, state: PageState) -> bool {
        matches!(
            (self, state),
            (Self::Reserved, PageState::Reserved(_))
                | (Self::Uploaded, PageState::Uploaded(_))
                | (Self::Resident, PageState::Resident(_))
                | (Self::Releasing, PageState::Releasing(_))
                | (Self::Retiring, PageState::Retiring { .. })
        )
    }
}

#[derive(Clone, Copy)]
struct PendingPageReclaim {
    allocation: PageAllocation,
    reclaim_after: GpuCompletionValue,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum UploadState {
    Reserved,
    Uploaded,
}

/// A new-allocation upload transaction.
pub struct PageUploadTransaction<'pool, 'bytes> {
    pool: &'pool mut FixedPagePool,
    upload: FixedPageUpload<'bytes>,
    allocation: PageAllocation,
    state: UploadState,
    committed: bool,
}

impl PageUploadTransaction<'_, '_> {
    /// The range reserved by this transaction.
    #[must_use]
    pub const fn allocation(&self) -> PageAllocation {
        self.allocation
    }

    /// Issues the real queue upload into the reserved range.
    pub fn upload(&mut self, queue: &dyn RenderQueue) -> Result<(), PagePoolError> {
        if self.state != UploadState::Reserved {
            return Err(PagePoolError::InvalidTransactionState);
        }
        self.pool
            .validate_state(self.allocation, PageStateKind::Reserved)?;
        queue.write_buffer(
            self.pool.buffer,
            self.allocation.offset_bytes(self.pool.page_size),
            self.upload.bytes,
        );
        self.pool
            .set_state(self.allocation, PageState::Uploaded(self.allocation.token));
        self.state = UploadState::Uploaded;
        Ok(())
    }

    /// Publishes the uploaded range as resident and returns its typed event.
    pub fn commit(mut self) -> Result<FixedPagePoolEvent, PagePoolError> {
        if self.state != UploadState::Uploaded {
            return Err(PagePoolError::InvalidTransactionState);
        }
        self.pool
            .validate_state(self.allocation, PageStateKind::Uploaded)?;
        self.pool
            .set_state(self.allocation, PageState::Resident(self.allocation.token));
        self.committed = true;
        Ok(FixedPagePoolEvent::Allocated {
            allocation: self.allocation,
            revision: self.upload.revision,
        })
    }

    /// Abandons the upload and returns its reservation to the free list.
    pub fn abort(mut self) {
        self.pool.rollback(self.allocation);
        self.committed = true;
    }
}

impl Drop for PageUploadTransaction<'_, '_> {
    fn drop(&mut self) {
        if !self.committed {
            self.pool.rollback(self.allocation);
        }
    }
}

/// An atomic replacement transaction for a resident allocation.
pub struct PageReplaceTransaction<'pool, 'bytes> {
    pool: &'pool mut FixedPagePool,
    upload: FixedPageUpload<'bytes>,
    previous: PageAllocation,
    allocation: PageAllocation,
    state: UploadState,
    committed: bool,
}

impl PageReplaceTransaction<'_, '_> {
    /// The old range that remains resident until commit.
    #[must_use]
    pub const fn previous(&self) -> PageAllocation {
        self.previous
    }

    /// The newly reserved range.
    #[must_use]
    pub const fn allocation(&self) -> PageAllocation {
        self.allocation
    }

    /// Issues the real queue upload into the new range.
    pub fn upload(&mut self, queue: &dyn RenderQueue) -> Result<(), PagePoolError> {
        if self.state != UploadState::Reserved {
            return Err(PagePoolError::InvalidTransactionState);
        }
        self.pool
            .validate_state(self.previous, PageStateKind::Resident)?;
        self.pool
            .validate_state(self.allocation, PageStateKind::Reserved)?;
        queue.write_buffer(
            self.pool.buffer,
            self.allocation.offset_bytes(self.pool.page_size),
            self.upload.bytes,
        );
        self.pool
            .set_state(self.allocation, PageState::Uploaded(self.allocation.token));
        self.state = UploadState::Uploaded;
        Ok(())
    }

    /// Publishes the new range and frees the old range in one logical commit.
    pub fn commit(mut self) -> Result<FixedPagePoolEvent, PagePoolError> {
        if self.state != UploadState::Uploaded {
            return Err(PagePoolError::InvalidTransactionState);
        }
        self.pool
            .validate_state(self.previous, PageStateKind::Resident)?;
        self.pool
            .validate_state(self.allocation, PageStateKind::Uploaded)?;
        self.pool
            .set_state(self.allocation, PageState::Resident(self.allocation.token));
        self.pool.set_state(self.previous, PageState::Free);
        self.committed = true;
        Ok(FixedPagePoolEvent::Replaced {
            previous: self.previous,
            allocation: self.allocation,
            revision: self.upload.revision,
        })
    }

    /// Publishes the new range while retaining the prior range until the
    /// caller later proves that `reclaim_after` has completed on the GPU.
    pub fn commit_deferred(
        mut self,
        reclaim_after: GpuCompletionValue,
    ) -> Result<FixedPageDeferredReplacement, PagePoolError> {
        if self.state != UploadState::Uploaded {
            return Err(PagePoolError::InvalidTransactionState);
        }
        self.pool
            .validate_state(self.previous, PageStateKind::Resident)?;
        self.pool
            .validate_state(self.allocation, PageStateKind::Uploaded)?;
        self.pool
            .set_state(self.allocation, PageState::Resident(self.allocation.token));
        self.pool.set_state(
            self.previous,
            PageState::Retiring {
                token: self.previous.token,
                reclaim_after,
            },
        );
        self.pool.pending_reclaims.push(PendingPageReclaim {
            allocation: self.previous,
            reclaim_after,
        });
        self.committed = true;
        Ok(FixedPageDeferredReplacement {
            previous: self.previous,
            allocation: self.allocation,
            revision: self.upload.revision,
            reclaim_after,
        })
    }

    /// Abandons the replacement and preserves the previous resident range.
    pub fn abort(mut self) {
        self.pool.rollback(self.allocation);
        self.committed = true;
    }
}

impl Drop for PageReplaceTransaction<'_, '_> {
    fn drop(&mut self) {
        if !self.committed {
            self.pool.rollback(self.allocation);
        }
    }
}

/// A transactional release of a resident allocation.
pub struct PageReleaseTransaction<'pool> {
    pool: &'pool mut FixedPagePool,
    allocation: PageAllocation,
    revision: u64,
    committed: bool,
}

impl PageReleaseTransaction<'_> {
    /// The range being released.
    #[must_use]
    pub const fn allocation(&self) -> PageAllocation {
        self.allocation
    }

    /// Makes the range free and emits a release event.
    pub fn commit(mut self) -> Result<FixedPagePoolEvent, PagePoolError> {
        self.pool
            .validate_state(self.allocation, PageStateKind::Releasing)?;
        self.pool.set_state(self.allocation, PageState::Free);
        self.committed = true;
        Ok(FixedPagePoolEvent::Released {
            allocation: self.allocation,
            revision: self.revision,
        })
    }

    /// Retires the range until the caller-provided completion proof arrives.
    pub fn commit_deferred(
        mut self,
        reclaim_after: GpuCompletionValue,
    ) -> Result<FixedPageDeferredRelease, PagePoolError> {
        self.pool
            .validate_state(self.allocation, PageStateKind::Releasing)?;
        self.pool.set_state(
            self.allocation,
            PageState::Retiring {
                token: self.allocation.token,
                reclaim_after,
            },
        );
        self.pool.pending_reclaims.push(PendingPageReclaim {
            allocation: self.allocation,
            reclaim_after,
        });
        self.committed = true;
        Ok(FixedPageDeferredRelease {
            allocation: self.allocation,
            revision: self.revision,
            reclaim_after,
        })
    }

    /// Abandons the release and restores residency.
    pub fn abort(mut self) {
        if self
            .pool
            .validate_state(self.allocation, PageStateKind::Releasing)
            .is_ok()
        {
            self.pool
                .set_state(self.allocation, PageState::Resident(self.allocation.token));
        }
        self.committed = true;
    }
}

impl Drop for PageReleaseTransaction<'_> {
    fn drop(&mut self) {
        if !self.committed
            && self
                .pool
                .validate_state(self.allocation, PageStateKind::Releasing)
                .is_ok()
        {
            self.pool
                .set_state(self.allocation, PageState::Resident(self.allocation.token));
        }
    }
}

fn validate_descriptor(descriptor: &FixedPagePoolDescriptor) -> Result<(), PagePoolError> {
    if descriptor.page_size == 0 {
        return Err(PagePoolError::ZeroPageSize);
    }
    if !descriptor.page_size.is_multiple_of(4) {
        return Err(PagePoolError::PageSizeNotAligned {
            page_size: descriptor.page_size,
        });
    }
    if descriptor.page_count == 0 {
        return Err(PagePoolError::ZeroPageCount);
    }
    descriptor
        .page_size
        .checked_mul(u64::from(descriptor.page_count))
        .ok_or(PagePoolError::SizeOverflow)?;
    if !descriptor.usage.contains(BufferUsages::COPY_DST) {
        return Err(PagePoolError::MissingCopyDestination);
    }
    Ok(())
}

fn validate_payload(byte_len: u64) -> Result<(), PagePoolError> {
    if byte_len == 0 {
        return Err(PagePoolError::EmptyUpload);
    }
    if !byte_len.is_multiple_of(4) {
        return Err(PagePoolError::UploadNotAligned { byte_len });
    }
    Ok(())
}
