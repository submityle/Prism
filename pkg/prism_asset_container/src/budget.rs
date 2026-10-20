//! Shared admission limits for untrusted container metadata and byte buffers.
//!
//! A budget is shared by clones and, when explicitly supplied, by different
//! mounts. Reservations are acquired before allocation and released on every
//! error path. Returned owned payloads belong to the caller; their retention is
//! governed by the runtime residency budget, not this in-flight I/O budget.

use alloc::sync::Arc;
use alloc::vec::Vec;
use core::ops::{Deref, DerefMut};
use core::sync::atomic::{AtomicU64, Ordering};

use crate::ContainerError;

/// Explicit resource ceilings applied before allocation or decompression.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContainerReadLimits {
    /// Largest immutable container accepted at mount, including padding.
    pub max_container_bytes: u64,
    /// Maximum total encoded directory/index bytes per container.
    pub max_metadata_bytes: u64,
    /// Maximum entries and chunks, separately, admitted from a header.
    pub max_records: u32,
    /// Maximum stored size of one independently decoded payload/chunk/block.
    pub max_stored_unit_bytes: u64,
    /// Maximum decoded size of one independently decoded payload/chunk/block.
    pub max_decoded_unit_bytes: u64,
    /// Maximum bytes returned by one complete read or aggregate batch.
    pub max_output_bytes: u64,
    /// Aggregate live byte-buffer reservations across mounts sharing a budget.
    pub max_in_flight_bytes: u64,
    /// Maximum decoded bytes validated during a single mount.
    pub max_validation_decoded_bytes: u64,
    /// Maximum raw/stored expansion ratio, with a 64 KiB small-unit allowance.
    pub max_expansion_ratio: u32,
}

impl Default for ContainerReadLimits {
    fn default() -> Self {
        Self {
            max_container_bytes: 1024 * 1024 * 1024 * 1024,
            max_metadata_bytes: 128 * 1024 * 1024,
            max_records: 1_000_000,
            max_stored_unit_bytes: 256 * 1024 * 1024,
            max_decoded_unit_bytes: 256 * 1024 * 1024,
            max_output_bytes: 256 * 1024 * 1024,
            max_in_flight_bytes: 768 * 1024 * 1024,
            max_validation_decoded_bytes: 16 * 1024 * 1024 * 1024,
            max_expansion_ratio: 4096,
        }
    }
}

impl ContainerReadLimits {
    pub(crate) fn check(
        self,
        resource: &'static str,
        requested: u64,
        limit: u64,
    ) -> Result<(), ContainerError> {
        if requested > limit {
            return Err(ContainerError::ResourceLimit {
                resource,
                requested,
                limit,
            });
        }
        Ok(())
    }

    pub(crate) fn unit(self, stored: u64, raw: u64) -> Result<(), ContainerError> {
        self.check("stored unit bytes", stored, self.max_stored_unit_bytes)?;
        self.check("decoded unit bytes", raw, self.max_decoded_unit_bytes)?;
        let expanded = stored
            .saturating_mul(u64::from(self.max_expansion_ratio))
            .max(64 * 1024);
        self.check("expansion bytes", raw, expanded)
    }

    pub(crate) fn metadata(
        self,
        directory: u64,
        index: u64,
        entries: u32,
        chunks: u32,
    ) -> Result<(), ContainerError> {
        self.check(
            "entry count",
            u64::from(entries),
            u64::from(self.max_records),
        )?;
        self.check(
            "chunk count",
            u64::from(chunks),
            u64::from(self.max_records),
        )?;
        let bytes = directory
            .checked_add(index)
            .ok_or(ContainerError::IntegerOverflow)?;
        self.check("metadata bytes", bytes, self.max_metadata_bytes)
    }
}

#[derive(Debug)]
struct BudgetState {
    limits: ContainerReadLimits,
    live: AtomicU64,
    peak: AtomicU64,
}

/// A fail-fast byte-buffer budget shared across concurrent container reads.
#[derive(Clone, Debug)]
pub struct ContainerReadBudget(Arc<BudgetState>);

impl Default for ContainerReadBudget {
    fn default() -> Self {
        Self::new(ContainerReadLimits::default())
    }
}

impl ContainerReadBudget {
    /// Creates a budget. A zero limit intentionally denies that resource.
    #[must_use]
    pub fn new(limits: ContainerReadLimits) -> Self {
        Self(Arc::new(BudgetState {
            limits,
            live: AtomicU64::new(0),
            peak: AtomicU64::new(0),
        }))
    }

    /// Returns the immutable admission policy.
    #[must_use]
    pub fn limits(&self) -> ContainerReadLimits {
        self.0.limits
    }

    /// Returns bytes reserved by operations currently executing.
    #[must_use]
    pub fn in_flight_bytes(&self) -> u64 {
        self.0.live.load(Ordering::Acquire)
    }

    /// Returns the highest observed simultaneous byte-buffer reservation.
    #[must_use]
    pub fn peak_in_flight_bytes(&self) -> u64 {
        self.0.peak.load(Ordering::Acquire)
    }

    pub(crate) fn reserve(&self, bytes: u64) -> Result<Reservation, ContainerError> {
        let mut live = self.0.live.load(Ordering::Acquire);
        loop {
            let next = live
                .checked_add(bytes)
                .ok_or(ContainerError::IntegerOverflow)?;
            self.0
                .limits
                .check("in-flight bytes", next, self.0.limits.max_in_flight_bytes)?;
            match self
                .0
                .live
                .compare_exchange_weak(live, next, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => {
                    self.0.peak.fetch_max(next, Ordering::Relaxed);
                    return Ok(Reservation {
                        budget: self.clone(),
                        bytes,
                    });
                }
                Err(current) => live = current,
            }
        }
    }

    pub(crate) fn buffer(&self, len: u64) -> Result<ReadBuffer, ContainerError> {
        let reservation = self.reserve(len)?;
        Ok(ReadBuffer {
            bytes: zeroed(len)?,
            _reservation: reservation,
        })
    }
}

#[derive(Debug)]
pub(crate) struct Reservation {
    budget: ContainerReadBudget,
    bytes: u64,
}

impl Drop for Reservation {
    fn drop(&mut self) {
        self.budget.0.live.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}

pub(crate) struct ReadBuffer {
    bytes: Vec<u8>,
    _reservation: Reservation,
}

impl ReadBuffer {
    pub(crate) fn truncate(&mut self, len: usize) {
        self.bytes.truncate(len);
    }
    pub(crate) fn into_vec(self) -> Vec<u8> {
        self.bytes
    }
}

impl Deref for ReadBuffer {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        &self.bytes
    }
}

impl DerefMut for ReadBuffer {
    fn deref_mut(&mut self) -> &mut [u8] {
        &mut self.bytes
    }
}

/// Allocates only after a fallible capacity reservation, avoiding capacity OOM
/// panics for lengths declared by an untrusted container or range caller.
pub(crate) fn zeroed(len: u64) -> Result<Vec<u8>, ContainerError> {
    let len_usize = usize::try_from(len).map_err(|_| ContainerError::IntegerOverflow)?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(len_usize)
        .map_err(|_| ContainerError::AllocationFailed { bytes: len })?;
    bytes.resize(len_usize, 0);
    Ok(bytes)
}
