//! Broad-phase configuration and error type.
//!
//! [`BroadphaseConfig`] pins the grid resolution and the fixed capacities that
//! the `GPU` path allocates up front (per-bucket entry slots and the output
//! pair buffer). The `CPU` twin honours the *same* capacities and reports the
//! *same* overflow conditions, so any input that would overflow on device also
//! fails on the reference rather than silently diverging.
//!
//! Provenance: standard uniform-grid parameters; no Unreal Engine source or
//! derived code.

/// Configuration for the spatial-hash broad phase.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BroadphaseConfig {
    /// Edge length of a grid cell. Must be strictly positive and at least the
    /// largest sphere diameter for the result to be exact.
    pub cell_size: f32,
    /// Number of hash buckets. Must be non-zero.
    pub table_size: u32,
    /// Maximum particles storable per bucket. Must be non-zero.
    pub max_per_bucket: u32,
    /// Maximum candidate pairs the output buffer can hold. Must be non-zero.
    pub pair_capacity: u32,
}

impl BroadphaseConfig {
    /// Creates a configuration.
    #[must_use]
    pub fn new(
        cell_size: f32,
        table_size: u32,
        max_per_bucket: u32,
        pair_capacity: u32,
    ) -> BroadphaseConfig {
        BroadphaseConfig {
            cell_size,
            table_size,
            max_per_bucket,
            pair_capacity,
        }
    }

    /// Total number of bucket entry slots (`table_size * max_per_bucket`).
    #[must_use]
    pub fn entry_slots(&self) -> u64 {
        u64::from(self.table_size) * u64::from(self.max_per_bucket)
    }

    /// Validates the configuration, returning the first violated invariant.
    ///
    /// # Errors
    ///
    /// Returns [`BroadphaseError::InvalidConfig`] when any capacity is zero or
    /// `cell_size` is not strictly positive and finite.
    pub fn validate(&self) -> Result<(), BroadphaseError> {
        if !self.cell_size.is_finite() || self.cell_size <= 0.0 {
            return Err(BroadphaseError::InvalidConfig(
                "cell_size must be positive and finite",
            ));
        }
        if self.table_size == 0 {
            return Err(BroadphaseError::InvalidConfig(
                "table_size must be non-zero",
            ));
        }
        if self.max_per_bucket == 0 {
            return Err(BroadphaseError::InvalidConfig(
                "max_per_bucket must be non-zero",
            ));
        }
        if self.pair_capacity == 0 {
            return Err(BroadphaseError::InvalidConfig(
                "pair_capacity must be non-zero",
            ));
        }
        Ok(())
    }
}

/// Errors the broad phase can report.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BroadphaseError {
    /// A configuration invariant was violated; carries a static reason.
    InvalidConfig(&'static str),
    /// More than `max_per_bucket` particles hashed to the same bucket.
    BucketOverflow {
        /// Index of the overflowing bucket.
        bucket: u32,
        /// The per-bucket capacity that was exceeded.
        capacity: u32,
    },
    /// More candidate pairs were produced than `pair_capacity` allows.
    PairCapacityExceeded {
        /// The pair capacity that was exceeded.
        capacity: u32,
    },
}

impl core::fmt::Display for BroadphaseError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            BroadphaseError::InvalidConfig(reason) => {
                write!(f, "invalid broad-phase config: {reason}")
            }
            BroadphaseError::BucketOverflow { bucket, capacity } => write!(
                f,
                "bucket {bucket} overflowed its capacity of {capacity} entries"
            ),
            BroadphaseError::PairCapacityExceeded { capacity } => {
                write!(f, "candidate pairs exceeded capacity of {capacity}")
            }
        }
    }
}

impl core::error::Error for BroadphaseError {}
