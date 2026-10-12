//! Staging/active double-buffer contract for writable shade pages.

use alloc::collections::{BTreeMap, BTreeSet};

use super::key::ShadePageKey;

/// A physical page handle published to the active sampler.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ShadePageHandle {
    /// Physical atlas slot.
    pub slot: u32,
    /// Content generation written into the slot.
    pub generation: u64,
    /// Material/light epoch represented by the page.
    pub epoch: u64,
}

/// A frame token used to prevent writes from becoming visible mid-frame.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ShadeFrameToken {
    serial: u64,
    epoch: u64,
}

/// A write lease handed to a compute/upload producer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ShadePageWrite {
    token: ShadeFrameToken,
    /// Page identity.
    pub key: ShadePageKey,
    /// Handle reserved in staging.
    pub handle: ShadePageHandle,
}

/// A write submitted to the GPU but not yet safe to publish.
///
/// The render adapter creates this token only after it has encoded and
/// submitted the copy/compute work. It must not call [`ShadePageStore::complete_write`]
/// until the corresponding queue completion is observed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ShadeSubmission {
    token: ShadeFrameToken,
    /// Page identity carried through the completion boundary.
    pub key: ShadePageKey,
    /// Staging handle written by the submitted GPU work.
    pub handle: ShadePageHandle,
}

/// Active page lookup result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShadeStoreLookup {
    /// The page is visible in the active generation.
    Ready(ShadePageHandle),
    /// No active page is available; the caller must use screen-space shading.
    Fallback,
}

/// Store errors that protect the no-half-written-page invariant.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShadeStoreError {
    /// A frame is already being prepared.
    FrameAlreadyOpen,
    /// No matching frame is open.
    NoOpenFrame,
    /// The token belongs to a previous frame.
    InvalidFrame,
    /// The atlas has no free slot.
    CapacityExhausted,
    /// A page key belongs to a different material epoch than the frame.
    EpochMismatch,
    /// The frame serial space is exhausted; reusing a serial would alias tokens.
    FrameSerialExhausted,
    /// A write lease is already open for this page.
    WriteAlreadyOpen,
    /// This page has already been written and committed in this frame.
    WriteAlreadyCommitted,
    /// A write lease has already been committed or belongs to another frame.
    InvalidWrite,
    /// Publishing would expose a page whose upload has not completed.
    PendingWrites,
}

/// Double-buffered logical page table. The active map is read-only until
/// [`publish`](Self::publish), at which point the complete staging generation
/// becomes visible in one swap.
#[derive(Clone, Debug)]
pub struct ShadePageStore {
    capacity: u32,
    next_slot: u32,
    free_slots: BTreeSet<u32>,
    active: BTreeMap<ShadePageKey, ShadePageHandle>,
    staging: Option<BTreeMap<ShadePageKey, ShadePageHandle>>,
    pending_writes: BTreeMap<ShadePageKey, u64>,
    submitted_writes: BTreeMap<ShadePageKey, u64>,
    written_pages: BTreeSet<ShadePageKey>,
    frame_serial: u64,
}

impl ShadePageStore {
    /// Creates an empty store with a bounded physical page capacity.
    #[must_use]
    pub fn new(capacity: u32) -> Self {
        Self {
            capacity,
            next_slot: 0,
            free_slots: BTreeSet::new(),
            active: BTreeMap::new(),
            staging: None,
            pending_writes: BTreeMap::new(),
            submitted_writes: BTreeMap::new(),
            written_pages: BTreeSet::new(),
            frame_serial: 0,
        }
    }

    /// Opens a new staging generation seeded from the currently active map.
    pub fn begin_frame(&mut self, epoch: u64) -> Result<ShadeFrameToken, ShadeStoreError> {
        if self.staging.is_some() {
            return Err(ShadeStoreError::FrameAlreadyOpen);
        }
        self.frame_serial = self
            .frame_serial
            .checked_add(1)
            .ok_or(ShadeStoreError::FrameSerialExhausted)?;
        self.staging = Some(self.active.clone());
        self.pending_writes.clear();
        self.submitted_writes.clear();
        self.written_pages.clear();
        Ok(ShadeFrameToken {
            serial: self.frame_serial,
            epoch,
        })
    }

    /// Reserves a physical slot and records a pending staging write.
    ///
    /// The key's material epoch must match the frame epoch. This keeps a
    /// physically valid upload from becoming visible under the wrong logical
    /// identity.
    pub fn begin_write(
        &mut self,
        token: ShadeFrameToken,
        key: ShadePageKey,
        generation: u64,
    ) -> Result<ShadePageWrite, ShadeStoreError> {
        if self.staging.is_none() {
            return Err(ShadeStoreError::NoOpenFrame);
        }
        if token.serial != self.frame_serial {
            return Err(ShadeStoreError::InvalidFrame);
        }
        if key.material_epoch != token.epoch {
            return Err(ShadeStoreError::EpochMismatch);
        }
        if self.pending_writes.contains_key(&key) {
            return Err(ShadeStoreError::WriteAlreadyOpen);
        }
        if self.written_pages.contains(&key) {
            return Err(ShadeStoreError::WriteAlreadyCommitted);
        }

        // An update must never write into the slot currently exposed by the
        // active table. The sampler may still be consuming that generation
        // while this frame's compute pass writes the replacement. The old
        // slot becomes reclaimable only after publish swaps the tables.
        let slot = if let Some(slot) = self.free_slots.pop_first() {
            slot
        } else if self.next_slot < self.capacity {
            let slot = self.next_slot;
            self.next_slot = self.next_slot.saturating_add(1);
            slot
        } else {
            return Err(ShadeStoreError::CapacityExhausted);
        };
        let handle = ShadePageHandle {
            slot,
            generation,
            epoch: token.epoch,
        };
        self.staging
            .as_mut()
            .expect("staging checked above")
            .insert(key, handle);
        self.pending_writes.insert(key, generation);
        Ok(ShadePageWrite { token, key, handle })
    }

    /// Records that the write has been encoded and submitted to the GPU.
    ///
    /// This does not make the page publishable. The returned token is consumed
    /// by [`complete_write`](Self::complete_write) only after the queue
    /// completion/fence for this submission has been observed.
    pub fn submit_write(
        &mut self,
        write: ShadePageWrite,
    ) -> Result<ShadeSubmission, ShadeStoreError> {
        if self.staging.is_none() {
            return Err(ShadeStoreError::NoOpenFrame);
        }
        if write.token.serial != self.frame_serial
            || self
                .staging
                .as_ref()
                .and_then(|staging| staging.get(&write.key))
                != Some(&write.handle)
            || self.pending_writes.get(&write.key) != Some(&write.handle.generation)
        {
            return Err(ShadeStoreError::InvalidWrite);
        }
        self.pending_writes.remove(&write.key);
        self.submitted_writes
            .insert(write.key, write.handle.generation);
        Ok(ShadeSubmission {
            token: write.token,
            key: write.key,
            handle: write.handle,
        })
    }

    /// Marks a submitted GPU write complete and makes it eligible for publish.
    ///
    /// The store cannot observe a backend fence itself; the caller is required
    /// to invoke this only after the backend reports completion. Keeping this
    /// as a separate transition prevents an encode/submit call from being
    /// accidentally treated as shader completion.
    pub fn complete_write(&mut self, submission: ShadeSubmission) -> Result<(), ShadeStoreError> {
        if self.staging.is_none() {
            return Err(ShadeStoreError::NoOpenFrame);
        }
        if submission.token.serial != self.frame_serial
            || self
                .staging
                .as_ref()
                .and_then(|staging| staging.get(&submission.key))
                != Some(&submission.handle)
            || self.submitted_writes.get(&submission.key) != Some(&submission.handle.generation)
        {
            return Err(ShadeStoreError::InvalidWrite);
        }
        self.submitted_writes.remove(&submission.key);
        self.written_pages.insert(submission.key);
        Ok(())
    }

    /// Abandons an in-flight frame and returns all staging-only slots to the
    /// allocator. This is required on device loss or a failed submission;
    /// leaving the frame open would otherwise leak slots and block future
    /// frames permanently.
    pub fn abort_frame(&mut self, token: ShadeFrameToken) -> Result<(), ShadeStoreError> {
        if token.serial != self.frame_serial {
            return Err(ShadeStoreError::InvalidFrame);
        }
        let Some(staging) = self.staging.take() else {
            return Err(ShadeStoreError::NoOpenFrame);
        };
        for handle in staging.values() {
            if !self
                .active
                .values()
                .any(|candidate| candidate.slot == handle.slot)
            {
                self.free_slots.insert(handle.slot);
            }
        }
        self.pending_writes.clear();
        self.submitted_writes.clear();
        self.written_pages.clear();
        Ok(())
    }

    /// Atomically publishes the complete staging page table.
    pub fn publish(&mut self, token: ShadeFrameToken) -> Result<(), ShadeStoreError> {
        if token.serial != self.frame_serial {
            return Err(ShadeStoreError::InvalidFrame);
        }
        if !self.pending_writes.is_empty() || !self.submitted_writes.is_empty() {
            return Err(ShadeStoreError::PendingWrites);
        }
        let Some(staging) = self.staging.take() else {
            return Err(ShadeStoreError::NoOpenFrame);
        };
        for handle in self.active.values() {
            if !staging
                .values()
                .any(|candidate| candidate.slot == handle.slot)
            {
                self.free_slots.insert(handle.slot);
            }
        }
        self.active = staging;
        Ok(())
    }

    /// Reads only the active generation. Staging writes are invisible here.
    #[must_use]
    pub fn lookup(&self, key: ShadePageKey, epoch: u64) -> ShadeStoreLookup {
        match self.active.get(&key).copied() {
            Some(handle) if handle.epoch == epoch => ShadeStoreLookup::Ready(handle),
            _ => ShadeStoreLookup::Fallback,
        }
    }

    /// Number of active published pages.
    #[must_use]
    pub fn active_len(&self) -> usize {
        self.active.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shading::texel_cache::ShadeChannel;

    fn key(epoch: u64) -> ShadePageKey {
        ShadePageKey {
            object_id: 1,
            material_id: 2,
            material_epoch: epoch,
            face: 0,
            mip: 0,
            channel: ShadeChannel::LitDiffuse,
            page_x: 0,
            page_y: 0,
        }
    }

    #[test]
    fn epoch_mismatch_is_rejected_and_serial_exhaustion_does_not_alias_tokens() {
        let mut store = ShadePageStore::new(1);
        let frame = store.begin_frame(4).unwrap();
        assert_eq!(
            store.begin_write(frame, key(3), 1),
            Err(ShadeStoreError::EpochMismatch)
        );
        store.publish(frame).unwrap();

        store.frame_serial = u64::MAX;
        assert_eq!(
            store.begin_frame(4),
            Err(ShadeStoreError::FrameSerialExhausted)
        );
    }

    #[test]
    fn submitted_write_cannot_publish_before_completion() {
        let mut store = ShadePageStore::new(1);
        let page = key(4);
        let frame = store.begin_frame(4).unwrap();
        let write = store.begin_write(frame, page, 1).unwrap();
        let submission = store.submit_write(write).unwrap();

        assert_eq!(store.publish(frame), Err(ShadeStoreError::PendingWrites));
        store.complete_write(submission).unwrap();
        store.publish(frame).unwrap();
        assert_eq!(store.lookup(page, 4), ShadeStoreLookup::Ready(write.handle));
    }

    #[test]
    fn duplicate_completion_is_rejected() {
        let mut store = ShadePageStore::new(1);
        let page = key(4);
        let frame = store.begin_frame(4).unwrap();
        let write = store.begin_write(frame, page, 1).unwrap();
        let submission = store.submit_write(write).unwrap();

        store.complete_write(submission).unwrap();
        assert_eq!(
            store.complete_write(submission),
            Err(ShadeStoreError::InvalidWrite)
        );
    }

    #[test]
    fn abort_recycles_staging_only_slots_and_allows_a_new_frame() {
        let mut store = ShadePageStore::new(1);
        let page = key(4);
        let frame = store.begin_frame(4).unwrap();
        let write = store.begin_write(frame, page, 1).unwrap();
        store.abort_frame(frame).unwrap();

        let next_frame = store.begin_frame(4).unwrap();
        let next_write = store.begin_write(next_frame, page, 2).unwrap();
        assert_eq!(next_write.handle.slot, write.handle.slot);
        store.abort_frame(next_frame).unwrap();
    }

    #[test]
    fn abort_preserves_active_slots_and_old_submissions_cannot_complete_new_frame() {
        let mut store = ShadePageStore::new(1);
        let page = key(4);
        let initial_frame = store.begin_frame(4).unwrap();
        let initial = store.begin_write(initial_frame, page, 1).unwrap();
        let initial_submission = store.submit_write(initial).unwrap();
        store.complete_write(initial_submission).unwrap();
        store.publish(initial_frame).unwrap();

        let aborted_frame = store.begin_frame(4).unwrap();
        assert_eq!(
            store.begin_write(aborted_frame, page, 2),
            Err(ShadeStoreError::CapacityExhausted)
        );
        store.abort_frame(aborted_frame).unwrap();
        assert_eq!(
            store.lookup(page, 4),
            ShadeStoreLookup::Ready(initial.handle)
        );

        let next_frame = store.begin_frame(4).unwrap();
        let next_write = store.begin_write(next_frame, page, 3);
        assert_eq!(next_write, Err(ShadeStoreError::CapacityExhausted));
        store.abort_frame(next_frame).unwrap();

        let submitted_frame = store.begin_frame(4).unwrap();
        let submitted = store.begin_write(submitted_frame, key(5), 4);
        assert_eq!(submitted, Err(ShadeStoreError::EpochMismatch));
        store.abort_frame(submitted_frame).unwrap();
    }

    #[test]
    fn stale_submission_cannot_complete_a_reused_slot() {
        let mut store = ShadePageStore::new(1);
        let page = key(4);
        let first_frame = store.begin_frame(4).unwrap();
        let first = store.begin_write(first_frame, page, 1).unwrap();
        let stale_submission = store.submit_write(first).unwrap();
        store.abort_frame(first_frame).unwrap();

        let second_frame = store.begin_frame(4).unwrap();
        let second = store.begin_write(second_frame, page, 2).unwrap();
        assert_eq!(
            store.complete_write(stale_submission),
            Err(ShadeStoreError::InvalidWrite)
        );
        let second_submission = store.submit_write(second).unwrap();
        store.complete_write(second_submission).unwrap();
        store.publish(second_frame).unwrap();
    }
}
