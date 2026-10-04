//! Leak reconciliation across a scope/frame boundary (§24.3).
//!
//! A pool that must return to its baseline after a bounded lifetime (a frame, a
//! level, a job scope) is the classic leak site: something allocated inside the
//! scope is never freed. [`LeakCheckpoint`] captures the live byte/allocation
//! counters at the opening boundary; [`LeakCheckpoint::reconcile`] diffs them
//! against the closing boundary and reports the residual.
//!
//! The residual is *signed*: a positive residual is a leak (bytes/allocations
//! that outlived the scope), a negative residual is an over-free (more freed
//! than the scope allocated — usually a sign the baseline was captured late).

/// A captured allocation boundary: live bytes and live (unmatched) allocations
/// at a point in time. Build one from raw counters or, with the `alloc-track`
/// feature, from a [`crate::alloc_track::AllocSnapshot`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LeakCheckpoint {
    /// Live bytes (allocated and not yet freed) at capture time.
    pub live_bytes: u64,
    /// Live allocations (allocations not yet matched by a free) at capture time.
    pub live_allocations: u64,
}

impl LeakCheckpoint {
    /// Construct from raw live counters.
    #[must_use]
    pub fn new(live_bytes: u64, live_allocations: u64) -> Self {
        Self {
            live_bytes,
            live_allocations,
        }
    }

    /// Construct from a global allocator snapshot (`alloc-track` feature).
    #[cfg(feature = "alloc-track")]
    #[must_use]
    pub fn from_snapshot(snapshot: &crate::alloc_track::AllocSnapshot) -> Self {
        Self {
            live_bytes: snapshot.live_bytes,
            live_allocations: snapshot.live_allocations(),
        }
    }

    /// Capture the current global allocator state (`alloc-track` feature).
    #[cfg(feature = "alloc-track")]
    #[must_use]
    pub fn capture() -> Self {
        Self::from_snapshot(&crate::alloc_track::snapshot())
    }

    /// Reconcile this opening checkpoint against the `end` closing checkpoint,
    /// reporting the residual that was not released within the scope.
    #[must_use]
    pub fn reconcile(self, end: LeakCheckpoint) -> LeakReport {
        LeakReport {
            residual_bytes: diff(end.live_bytes, self.live_bytes),
            residual_allocations: diff(end.live_allocations, self.live_allocations),
        }
    }
}

/// The result of reconciling two [`LeakCheckpoint`]s.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LeakReport {
    /// Signed `end.live_bytes - start.live_bytes`. Positive is a leak.
    pub residual_bytes: i64,
    /// Signed `end.live_allocations - start.live_allocations`. Positive is a
    /// leak.
    pub residual_allocations: i64,
}

impl LeakReport {
    /// Whether a leak was detected (positive residual bytes or allocations).
    #[must_use]
    pub fn leaked(&self) -> bool {
        self.residual_bytes > 0 || self.residual_allocations > 0
    }

    /// Whether more was freed than allocated within the scope (negative
    /// residual), which usually means the baseline was captured too late.
    #[must_use]
    pub fn over_freed(&self) -> bool {
        self.residual_bytes < 0 || self.residual_allocations < 0
    }

    /// Whether the scope returned exactly to its baseline (no residual).
    #[must_use]
    pub fn balanced(&self) -> bool {
        self.residual_bytes == 0 && self.residual_allocations == 0
    }
}

/// Signed difference `a - b` without `i64` overflow for `u64` inputs.
fn diff(a: u64, b: u64) -> i64 {
    if a >= b {
        (a - b) as i64
    } else {
        -((b - a) as i64)
    }
}
