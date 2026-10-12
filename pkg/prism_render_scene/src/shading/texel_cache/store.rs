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
            written_pages: BTreeSet::new(),
            frame_serial: 0,
        }
    }

    /// Opens a new staging generation seeded from the currently active map.
    pub fn begin_frame(&mut self, epoch: u64) -> Result<ShadeFrameToken, ShadeStoreError> {
        if self.staging.is_some() {
            return Err(ShadeStoreError::FrameAlreadyOpen);
        }
        self.frame_serial = self.frame_serial.saturating_add(1).max(1);
        self.staging = Some(self.active.clone());
        self.pending_writes.clear();
        self.written_pages.clear();
        Ok(ShadeFrameToken {
            serial: self.frame_serial,
            epoch,
        })
    }

    /// Reserves a physical slot and records a pending staging write.
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

    /// Commits a write lease. The data upload must happen before this call;
    /// this method only records that the complete page is ready for publish.
    pub fn commit_write(&mut self, write: ShadePageWrite) -> Result<(), ShadeStoreError> {
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
        self.written_pages.insert(write.key);
        Ok(())
    }

    /// Atomically publishes the complete staging page table.
    pub fn publish(&mut self, token: ShadeFrameToken) -> Result<(), ShadeStoreError> {
        if token.serial != self.frame_serial {
            return Err(ShadeStoreError::InvalidFrame);
        }
        if !self.pending_writes.is_empty() {
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
