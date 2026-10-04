//! Per-frame output of the virtual-texture streamer.
//!
//! One [`StreamerFrame`] bundles everything a frame of
//! [`VirtualTextureStreamer`](super::streamer::VirtualTextureStreamer) resolves:
//! the scheduler plan, the physical uploads the backend must record, the
//! optional atlas copy layout, and compact counters for budget telemetry. It
//! owns no `GPU` handle; the device-side twin records the copies and uploads the
//! page-table words the streamer exposes separately.

use super::atlas::AtlasCopyPlan;
use super::pool::PageUpload;
use super::scheduler::StreamingPlan;
use alloc::vec::Vec;

/// Everything one streamer frame resolved.
///
/// `uploads` is the subset of [`plan`](Self::plan)`.loads` the physical pool
/// could actually place this frame (a load the pool was too full to seat is
/// dropped, not fatal); it is the authoritative "what the backend copies" list
/// and the input the optional [`atlas`](Self::atlas) plan was laid out over.
/// `resident_bytes` is measured from the pool after the plan is applied, so it
/// reflects pages truly resident rather than the scheduler's pre-admission
/// estimate.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct StreamerFrame {
    /// The scheduler's load/evict decision for this frame.
    pub plan: StreamingPlan,
    /// Pages actually uploaded into the pool this frame, in load order.
    pub uploads: Vec<PageUpload>,
    /// Atlas copy layout for [`uploads`](Self::uploads), when atlas geometry is
    /// configured; `None` otherwise.
    pub atlas: Option<AtlasCopyPlan>,
    /// Total physical bytes resident after this frame, measured from the pool.
    pub resident_bytes: u64,
    /// Number of pages resident after this frame.
    pub resident_count: usize,
    /// Number of distinct pages requested by feedback this frame.
    pub demanded_pages: usize,
    /// Requested, non-resident pages withheld from upload this frame because the
    /// load debounce was not yet satisfied.
    pub deferred_loads: usize,
    /// Staging bytes actually uploaded this frame, i.e. the summed byte cost of
    /// [`uploads`](Self::uploads). Never exceeds the configured upload budget
    /// except when a single highest-priority page is larger than the whole
    /// budget, which is always admitted to keep the loop live.
    pub uploaded_bytes: u64,
    /// Pages seated in the physical pool but still awaiting upload after this
    /// frame because the upload budget was exhausted. They carry forward and
    /// drain in priority order on later frames, and are absent from the `GPU`
    /// page table until uploaded. Always `0` when no upload budget is set.
    pub pending_uploads: usize,
}

impl StreamerFrame {
    /// Number of pages uploaded this frame.
    #[must_use]
    pub fn loaded(&self) -> usize {
        self.uploads.len()
    }

    /// Number of pages evicted this frame.
    #[must_use]
    pub fn evicted(&self) -> usize {
        self.plan.evicts.len()
    }

    /// Whether the frame performed no uploads and no evictions.
    ///
    /// A steady state over an unchanging demand set converges to this, which is
    /// the signal that the resident set has stabilized and the loop is no longer
    /// doing streaming work.
    #[must_use]
    pub fn is_steady(&self) -> bool {
        self.uploads.is_empty() && self.plan.evicts.is_empty() && self.pending_uploads == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_frame_is_steady_and_empty() {
        let frame = StreamerFrame::default();
        assert!(frame.is_steady());
        assert_eq!(frame.loaded(), 0);
        assert_eq!(frame.evicted(), 0);
        assert_eq!(frame.resident_bytes, 0);
        assert_eq!(frame.resident_count, 0);
        assert_eq!(frame.uploaded_bytes, 0);
        assert_eq!(frame.pending_uploads, 0);
        assert!(frame.atlas.is_none());
    }
}
