//! Crate-private admission/publication for copies from authenticated GPU tiles.

use super::*;

impl FixedPagePool {
    pub(crate) fn begin_resident_copy(
        &mut self,
        byte_len: u64,
        previous: Option<PageAllocation>,
    ) -> Result<ResidentCopyTransaction<'_>, PagePoolError> {
        self.ensure_live()?;
        if let Some(previous) = previous {
            self.validate_state(previous, PageStateKind::Resident)?;
        }
        if byte_len == 0 {
            return Err(PagePoolError::EmptyUpload);
        }
        let padded = byte_len.checked_add(3).ok_or(PagePoolError::SizeOverflow)? & !3;
        let mut allocation =
            self.reserve(usize::try_from(padded).map_err(|_| PagePoolError::SizeOverflow)?)?;
        allocation.byte_len = byte_len;
        Ok(ResidentCopyTransaction {
            pool: self,
            allocation,
            previous,
            submitted: false,
        })
    }
}

pub(crate) struct ResidentCopyTransaction<'a> {
    pool: &'a mut FixedPagePool,
    allocation: PageAllocation,
    previous: Option<PageAllocation>,
    submitted: bool,
}

impl ResidentCopyTransaction<'_> {
    pub(crate) fn allocation(&self) -> PageAllocation {
        self.allocation
    }

    /// All validation/admission precedes the submission closure. The caller
    /// holds the pool exclusively; publication after submit cannot fail or
    /// allocate. On a submission error or panic, keep the reservation quarantined:
    /// commands might have reached the queue, so rollback would be unsafe.
    pub(crate) fn submit<E>(
        mut self,
        reclaim_after: GpuCompletionValue,
        submit: impl FnOnce() -> Result<(), E>,
    ) -> Result<PageAllocation, E> {
        self.submitted = true;
        submit()?;
        self.pool
            .set_state(self.allocation, PageState::Resident(self.allocation.token));
        if let Some(previous) = self.previous {
            self.pool.set_state(
                previous,
                PageState::Retiring {
                    token: previous.token,
                    reclaim_after,
                },
            );
            self.pool.pending_reclaims.push(PendingPageReclaim {
                allocation: previous,
                reclaim_after,
            });
        }
        Ok(self.allocation)
    }
}

impl Drop for ResidentCopyTransaction<'_> {
    fn drop(&mut self) {
        if !self.submitted {
            self.pool.rollback(self.allocation);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::{GpuPrepareBudget, PrepareCancellation};
    use super::*;
    use prism_render_driver::{DeviceDescriptor, Instance, RequestAdapterOptions};
    use prism_render_driver_null::{NullDevice, NullInstance, NullQueue};

    fn pool() -> (NullDevice, NullQueue, FixedPagePool) {
        let instance = NullInstance::new();
        let adapter = instance
            .request_adapter(&RequestAdapterOptions::default())
            .unwrap();
        let (device, queue) = instance
            .create_device(&adapter, &DeviceDescriptor::default())
            .unwrap();
        let mut budget = GpuPrepareBudget::new(64);
        let cancellation = PrepareCancellation::new(false);
        let mut context = PrepareContext::new(&device, &queue, &mut budget, &cancellation);
        let pool = FixedPagePool::new(
            &mut context,
            &FixedPagePoolDescriptor::new(8, 4, BufferUsages::COPY_DST),
        )
        .unwrap();
        context.commit();
        drop(context);
        (device, queue, pool)
    }

    #[test]
    fn odd_logical_lengths_are_word_padded_but_not_published_as_padding() {
        let (device, _, mut pool) = pool();
        for len in [1, 3, 7, 9, 31] {
            let transaction = pool.begin_resident_copy(len, None).unwrap();
            let allocation = transaction.allocation();
            assert_eq!(allocation.byte_len(), len);
            assert_eq!(allocation.page_count(), len.div_ceil(8) as u32);
            drop(transaction);
            assert_eq!(pool.free_page_count(), 4);
        }
        assert!(matches!(
            pool.begin_resident_copy(0, None),
            Err(PagePoolError::EmptyUpload)
        ));
        assert!(matches!(
            pool.begin_resident_copy(u64::MAX, None),
            Err(PagePoolError::SizeOverflow)
        ));
        pool.destroy(&device);
    }

    #[test]
    fn aborted_resident_replacement_preserves_the_old_publication() {
        let (device, _, mut pool) = pool();
        let old = pool
            .begin_resident_copy(7, None)
            .unwrap()
            .submit(GpuCompletionValue(0), || Ok::<(), ()>(()))
            .unwrap();
        let replacement = pool.begin_resident_copy(9, Some(old)).unwrap();
        drop(replacement);
        assert_eq!(pool.free_page_count(), 3);
        assert_eq!(pool.resident_page_count(), 1);
        assert_eq!(pool.pending_reclaim_page_count(), 0);
        pool.validate_state(old, PageStateKind::Resident).unwrap();
        pool.destroy(&device);
    }

    #[test]
    fn committed_copy_retires_old_pages_until_the_exact_completion_value() {
        let (device, _, mut pool) = pool();
        let old = pool
            .begin_resident_copy(7, None)
            .unwrap()
            .submit(GpuCompletionValue(0), || Ok::<(), ()>(()))
            .unwrap();
        let new = pool
            .begin_resident_copy(9, Some(old))
            .unwrap()
            .submit(GpuCompletionValue(5), || Ok::<(), ()>(()))
            .unwrap();
        assert_eq!(pool.free_page_count(), 1);
        assert_eq!(pool.resident_page_count(), 2);
        assert_eq!(pool.pending_reclaim_page_count(), 1);
        assert!(pool
            .reclaim_completed(GpuCompletionValue(4))
            .unwrap()
            .is_empty());
        let events = pool.reclaim_completed(GpuCompletionValue(5)).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].allocation, old);
        pool.validate_state(new, PageStateKind::Resident).unwrap();
        assert_eq!(pool.free_page_count(), 2);
        pool.destroy(&device);
    }

    #[test]
    fn submission_panic_never_recycles_a_potentially_submitted_range() {
        let (device, _, mut pool) = pool();
        let old = pool
            .begin_resident_copy(7, None)
            .unwrap()
            .submit(GpuCompletionValue(0), || Ok::<(), ()>(()))
            .unwrap();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = pool
                .begin_resident_copy(9, Some(old))
                .unwrap()
                .submit(GpuCompletionValue(5), || -> Result<(), ()> {
                    panic!("submission may already be queued")
                });
        }));
        assert!(result.is_err());
        assert_eq!(pool.free_page_count(), 1);
        assert_eq!(pool.resident_page_count(), 1);
        assert_eq!(pool.pending_reclaim_page_count(), 0);
        pool.validate_state(old, PageStateKind::Resident).unwrap();
        assert!(pool
            .reclaim_completed(GpuCompletionValue(u64::MAX))
            .unwrap()
            .is_empty());
        pool.destroy(&device);
    }

    #[test]
    fn submission_error_preserves_old_pages_and_quarantines_the_new_range() {
        let (device, _, mut pool) = pool();
        let old = pool
            .begin_resident_copy(7, None)
            .unwrap()
            .submit(GpuCompletionValue(0), || Ok::<(), ()>(()))
            .unwrap();
        let result = pool
            .begin_resident_copy(9, Some(old))
            .unwrap()
            .submit(GpuCompletionValue(5), || Err("uncertain queue submission"));
        assert_eq!(result, Err("uncertain queue submission"));
        assert_eq!(pool.free_page_count(), 1);
        assert_eq!(pool.resident_page_count(), 1);
        assert_eq!(pool.pending_reclaim_page_count(), 0);
        pool.validate_state(old, PageStateKind::Resident).unwrap();
        assert!(pool
            .reclaim_completed(GpuCompletionValue(u64::MAX))
            .unwrap()
            .is_empty());
        pool.destroy(&device);
    }
}
