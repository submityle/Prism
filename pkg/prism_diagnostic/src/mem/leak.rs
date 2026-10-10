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

/// A captured per-tag live-byte boundary from the tagged allocator
/// (`alloc-track` feature).
///
/// Where [`LeakCheckpoint`] reconciles the *global* live byte total,
/// [`TagLeakCheckpoint`] records each registered tag's live residency so a
/// later reconcile can attribute a residual leak to the specific tag (and
/// therefore the specific subsystem) that failed to release its scope-local
/// allocations. It is populated only by
/// [`LiveTrackingAllocator`](crate::alloc_track::LiveTrackingAllocator), the
/// sole allocator that maintains exact per-tag live bytes; under the
/// header-free [`TrackingAllocator`](crate::alloc_track::TrackingAllocator)
/// every tag reads `0` live bytes and the report is empty of leaks.
#[cfg(feature = "alloc-track")]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TagLeakCheckpoint {
    tags: Vec<TagLive>,
}

/// One tag's live residency at capture time.
#[cfg(feature = "alloc-track")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TagLive {
    name: &'static str,
    live_bytes: u64,
}

#[cfg(feature = "alloc-track")]
impl TagLeakCheckpoint {
    /// Capture the current per-tag live residency from
    /// [`crate::alloc_track::tag_report`].
    #[must_use]
    pub fn capture() -> Self {
        let report = crate::alloc_track::tag_report();
        let mut tags = Vec::with_capacity(report.len());
        for stat in &report {
            tags.push(TagLive {
                name: stat.name,
                live_bytes: stat.live_bytes,
            });
        }
        Self { tags }
    }

    /// Look up a tag's live bytes at capture time (`0` if the tag was not yet
    /// registered when this checkpoint was taken).
    fn live_bytes_for(&self, name: &str) -> u64 {
        self.tags
            .iter()
            .find(|t| t.name == name)
            .map_or(0, |t| t.live_bytes)
    }

    /// Reconcile this opening checkpoint against the `end` closing checkpoint,
    /// attributing each tag's residual (`end - start`) to that tag.
    ///
    /// Tags are only ever added to the global table, so `end` is a superset of
    /// `self`; every tag present at `end` is reported, using `0` as the opening
    /// baseline for tags registered after `self` was captured. A positive
    /// residual for a tag is that tag's leak; a negative residual is an
    /// over-free against its baseline.
    #[must_use]
    pub fn reconcile(&self, end: &TagLeakCheckpoint) -> TagLeakReport {
        let mut residuals = Vec::with_capacity(end.tags.len());
        for tag in &end.tags {
            residuals.push(TagLeakResidual {
                tag: tag.name,
                residual_bytes: diff(tag.live_bytes, self.live_bytes_for(tag.name)),
            });
        }
        TagLeakReport { residuals }
    }
}

/// The per-tag residual produced by [`TagLeakCheckpoint::reconcile`].
#[cfg(feature = "alloc-track")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TagLeakResidual {
    /// The tag (subsystem) this residual is attributed to.
    pub tag: &'static str,
    /// Signed `end.live_bytes - start.live_bytes` for this tag. Positive is a
    /// leak, negative an over-free against the baseline.
    pub residual_bytes: i64,
}

#[cfg(feature = "alloc-track")]
impl TagLeakResidual {
    /// Whether this tag leaked (positive residual).
    #[must_use]
    pub fn leaked(&self) -> bool {
        self.residual_bytes > 0
    }
}

/// Per-tag leak attribution: the residual live bytes each tag failed to release
/// across a scope/frame boundary (`alloc-track` feature).
#[cfg(feature = "alloc-track")]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TagLeakReport {
    residuals: Vec<TagLeakResidual>,
}

#[cfg(feature = "alloc-track")]
impl TagLeakReport {
    /// Every tag's residual, in tag registration order.
    #[must_use]
    pub fn residuals(&self) -> &[TagLeakResidual] {
        &self.residuals
    }

    /// Iterate only the tags that leaked (positive residual).
    pub fn leaking_tags(&self) -> impl Iterator<Item = &TagLeakResidual> {
        self.residuals.iter().filter(|r| r.leaked())
    }

    /// Whether any tag leaked.
    #[must_use]
    pub fn leaked(&self) -> bool {
        self.residuals.iter().any(TagLeakResidual::leaked)
    }

    /// The residual attributed to a single tag (`0` if the tag is absent).
    #[must_use]
    pub fn residual_for(&self, tag: &str) -> i64 {
        self.residuals
            .iter()
            .find(|r| r.tag == tag)
            .map_or(0, |r| r.residual_bytes)
    }

    /// The summed residual across all tags (the per-tag analogue of
    /// [`LeakReport::residual_bytes`]).
    #[must_use]
    pub fn total_residual_bytes(&self) -> i64 {
        self.residuals.iter().map(|r| r.residual_bytes).sum()
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

#[cfg(all(test, feature = "alloc-track"))]
#[expect(
    unsafe_code,
    reason = "driving GlobalAlloc::alloc/dealloc directly to exercise per-tag \
              live-byte leak attribution; every block is freed with its own layout"
)]
mod live_leak_tests {
    use super::*;
    use crate::alloc_track::{register_tag, tag_scope, LiveTrackingAllocator};
    use core::alloc::{GlobalAlloc, Layout};
    use std::alloc::System;

    // A block allocated under a tag and never freed across the scope is
    // attributed as that tag's leak; a sibling tag that balances is not.
    #[test]
    fn reconcile_attributes_residual_to_the_leaking_tag() {
        let leaker = "prism::test::leak::live_leaker";
        let clean = "prism::test::leak::live_clean";
        let leak_tag = register_tag(leaker).expect("tag slot");
        let clean_tag = register_tag(clean).expect("tag slot");
        let alloc = LiveTrackingAllocator::new(System);

        let start = TagLeakCheckpoint::capture();

        // The clean tag allocates and frees within the scope (balanced).
        let clean_layout = Layout::from_size_align(2048, 16).unwrap();
        let clean_ptr = {
            let _scope = tag_scope(clean_tag);
            // SAFETY: non-zero layout; freed below with the same layout.
            unsafe { alloc.alloc(clean_layout) }
        };
        assert!(!clean_ptr.is_null());
        // SAFETY: same block/layout as the allocation above.
        unsafe { alloc.dealloc(clean_ptr, clean_layout) };

        // The leaker allocates and does NOT free before the closing checkpoint.
        let leak_layout = Layout::from_size_align(4096, 16).unwrap();
        let leak_ptr = {
            let _scope = tag_scope(leak_tag);
            // SAFETY: non-zero layout; freed after reconcile with same layout.
            unsafe { alloc.alloc(leak_layout) }
        };
        assert!(!leak_ptr.is_null());

        let end = TagLeakCheckpoint::capture();
        let report = start.reconcile(&end);

        assert!(report.leaked(), "the leaking tag must be flagged");
        assert_eq!(report.residual_for(leaker), 4096);
        assert_eq!(report.residual_for(clean), 0);

        let leaking: Vec<&str> = report.leaking_tags().map(|r| r.tag).collect();
        assert!(leaking.contains(&leaker));
        assert!(!leaking.contains(&clean));
        assert!(report.total_residual_bytes() >= 4096);

        // Clean up the deliberate leak so later tests start balanced.
        // SAFETY: same block/layout as the allocation above.
        unsafe { alloc.dealloc(leak_ptr, leak_layout) };

        // After the free the same opening checkpoint reconciles clean for the
        // tag (allowing for unrelated global tags from other tests).
        let settled = start.reconcile(&TagLeakCheckpoint::capture());
        assert_eq!(settled.residual_for(leaker), 0);
    }

    // A balanced scope (every tagged allocation freed) reports no leak.
    #[test]
    fn reconcile_balanced_scope_reports_no_leak() {
        let tag_name = "prism::test::leak::live_balanced";
        let tag = register_tag(tag_name).expect("tag slot");
        let alloc = LiveTrackingAllocator::new(System);

        let start = TagLeakCheckpoint::capture();
        let layout = Layout::from_size_align(1024, 16).unwrap();
        let ptr = {
            let _scope = tag_scope(tag);
            // SAFETY: non-zero layout; freed below with the same layout.
            unsafe { alloc.alloc(layout) }
        };
        assert!(!ptr.is_null());
        // SAFETY: same block/layout as the allocation above.
        unsafe { alloc.dealloc(ptr, layout) };

        let report = start.reconcile(&TagLeakCheckpoint::capture());
        assert_eq!(report.residual_for(tag_name), 0);
    }
}
