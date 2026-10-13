//! Transactional context for budgeted render-asset preparation.

use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, Ordering};

use prism_render_driver::{BufferDescriptor, BufferId, RenderDevice, RenderQueue};

/// Why a contextual prepare cannot continue in this pass.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PrepareContextError {
    /// The requested GPU allocation would exceed the attempt's budget.
    BudgetExceeded {
        /// Bytes requested by the prepare.
        requested: u64,
        /// Bytes still available before the request.
        remaining: u64,
    },
    /// The owner cancelled the prepare before the next allocation/checkpoint.
    Cancelled,
}

/// A byte budget shared by a group of render-asset prepare attempts.
///
/// Reservations made by a failed or cancelled attempt are returned. Successful
/// attempts keep their reservations until the caller drops or reuses the
/// budget, which makes the contract observable in tests and lets a caller bound
/// the amount of new GPU allocation work in one pass.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct GpuPrepareBudget {
    limit: u64,
    used: u64,
}

impl GpuPrepareBudget {
    /// Creates a finite byte budget.
    #[must_use]
    pub const fn new(limit: u64) -> Self {
        Self { limit, used: 0 }
    }

    /// Creates a budget that does not constrain allocation size.
    #[must_use]
    pub const fn unlimited() -> Self {
        Self::new(u64::MAX)
    }

    /// Bytes already reserved by successful attempts and the current attempt.
    #[must_use]
    pub const fn used(self) -> u64 {
        self.used
    }

    /// Bytes still available.
    #[must_use]
    pub const fn remaining(self) -> u64 {
        self.limit.saturating_sub(self.used)
    }

    fn reserve(&mut self, bytes: u64) -> Result<(), PrepareContextError> {
        if bytes > self.remaining() {
            return Err(PrepareContextError::BudgetExceeded {
                requested: bytes,
                remaining: self.remaining(),
            });
        }
        self.used += bytes;
        Ok(())
    }

    fn refund(&mut self, bytes: u64) {
        self.used = self.used.saturating_sub(bytes);
    }
}

/// A cancellation flag suitable for passing to a prepare context.
pub type PrepareCancellation = AtomicBool;

/// The transactional context supplied to a contextual render-asset prepare.
///
/// Implementations should allocate through [`Self::create_buffer`] and call
/// [`Self::replace_previous`] only when they return a fresh representation that
/// does not reuse the previous representation. If the prepare returns an
/// error, all buffers created through this context are destroyed and all
/// reservations are refunded before the previous representation is observed
/// again.
pub struct PrepareContext<'a> {
    device: &'a dyn RenderDevice,
    queue: &'a dyn RenderQueue,
    budget: &'a mut GpuPrepareBudget,
    cancellation: &'a PrepareCancellation,
    reserved: u64,
    tracked: Vec<TrackedResource>,
    replaces_previous: bool,
    committed: bool,
}

#[derive(Clone, Copy)]
enum TrackedResource {
    Buffer(BufferId),
}

impl<'a> PrepareContext<'a> {
    /// Starts one transactional prepare attempt.
    #[must_use]
    pub fn new(
        device: &'a dyn RenderDevice,
        queue: &'a dyn RenderQueue,
        budget: &'a mut GpuPrepareBudget,
        cancellation: &'a PrepareCancellation,
    ) -> Self {
        Self {
            device,
            queue,
            budget,
            cancellation,
            reserved: 0,
            tracked: Vec::new(),
            replaces_previous: false,
            committed: false,
        }
    }

    /// The device used by this attempt for operations not wrapped by the
    /// context. Prefer the context allocation helpers for new resources.
    #[must_use]
    pub fn device(&self) -> &dyn RenderDevice {
        self.device
    }

    /// The upload queue used by this attempt.
    #[must_use]
    pub fn queue(&self) -> &dyn RenderQueue {
        self.queue
    }

    /// Returns whether cancellation was requested.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancellation.load(Ordering::Acquire)
    }

    /// Checks cancellation at an implementation-defined safe point.
    pub fn check_cancelled(&self) -> Result<(), PrepareContextError> {
        if self.is_cancelled() {
            Err(PrepareContextError::Cancelled)
        } else {
            Ok(())
        }
    }

    /// Reserves bytes for a non-buffer allocation or a batched upload.
    pub fn reserve(&mut self, bytes: u64) -> Result<(), PrepareContextError> {
        self.check_cancelled()?;
        self.budget.reserve(bytes)?;
        self.reserved += bytes;
        Ok(())
    }

    /// Creates and tracks a buffer as one transactional allocation.
    pub fn create_buffer(
        &mut self,
        descriptor: &BufferDescriptor,
    ) -> Result<BufferId, PrepareContextError> {
        self.reserve(descriptor.size)?;
        let buffer = self.device.create_buffer(descriptor);
        self.tracked.push(TrackedResource::Buffer(buffer));
        Ok(buffer)
    }

    /// Declares that a successful result replaces, rather than reuses, the
    /// previous representation. The sync layer then calls `unload` exactly once
    /// after the new representation is published.
    pub fn replace_previous(&mut self) {
        self.replaces_previous = true;
    }

    pub(crate) fn replaces_previous(&self) -> bool {
        self.replaces_previous
    }

    pub(crate) fn commit(&mut self) {
        self.tracked.clear();
        self.reserved = 0;
        self.committed = true;
    }

    pub(crate) fn rollback(&mut self) {
        for resource in self.tracked.drain(..) {
            match resource {
                TrackedResource::Buffer(buffer) => self.device.destroy_buffer(buffer),
            }
        }
        self.budget.refund(self.reserved);
        self.reserved = 0;
        self.replaces_previous = false;
        self.committed = true;
    }
}

impl Drop for PrepareContext<'_> {
    fn drop(&mut self) {
        if !self.committed {
            self.rollback();
        }
    }
}
