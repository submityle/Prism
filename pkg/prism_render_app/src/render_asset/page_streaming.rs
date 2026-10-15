//! Main-world fixed-page streaming production and render-side handoff.
//!
//! This slice owns source reads and CPU residency only. It copies resident
//! bytes into an owned render queue, while the render-side upload contract is
//! deliberately explicit about bytes and silent about GPU completion.

use alloc::collections::{BTreeMap, BTreeSet, VecDeque};
use alloc::vec::Vec;

use prism_asset::{
    AssetPath, AssetSources, PageId, PageKey, PageLoadError, PageRequestError, PageResidency,
    PageState,
};
use prism_ecs::prelude::Resource;
use prism_tasks::Priority;

/// A CPU-side fixed-page streaming source suitable for main-world callers.
///
/// The source owns the mounted asset table and the CPU residency authority.
/// It performs no GPU work and does not claim that a render upload has
/// completed.
#[derive(Resource)]
pub struct FixedPageStreamingSource {
    sources: AssetSources,
    residency: PageResidency,
    extracted: BTreeMap<PageId, ExtractedPageMetadata>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ExtractedPageMetadata {
    slot: FixedPageSlotKey,
    revision: u64,
}

/// Stable identity of a page slot across source revisions.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FixedPageSlotKey {
    /// Mounted asset path without a source revision.
    pub path: AssetPath,
    /// Zero-based page index within the asset.
    pub page_index: u64,
}

/// Compatibility name for callers that prefer to describe this value as a
/// manager rather than a source.
pub type FixedPageStreamingManager = FixedPageStreamingSource;

impl FixedPageStreamingSource {
    /// Creates a streaming source from an owned source table and residency.
    #[must_use]
    pub fn new(sources: AssetSources, residency: PageResidency) -> Self {
        Self {
            sources,
            residency,
            extracted: BTreeMap::new(),
        }
    }

    /// Returns the mounted source table.
    #[must_use]
    pub fn sources(&self) -> &AssetSources {
        &self.sources
    }

    /// Returns the CPU residency authority.
    #[must_use]
    pub fn residency(&self) -> &PageResidency {
        &self.residency
    }

    /// Returns mutable access to CPU residency for explicit lifecycle work
    /// such as eviction.
    pub fn residency_mut(&mut self) -> &mut PageResidency {
        &mut self.residency
    }

    /// Registers a page request using the source table's current revision.
    pub fn request_page(
        &mut self,
        path: impl Into<AssetPath>,
        page_index: u64,
        priority: Priority,
        importance: u32,
    ) -> Result<PageId, PageRequestError> {
        let key = PageKey::from_sources(&self.sources, path, page_index);
        self.request_page_key(key, priority, importance)
    }

    /// Registers a page request with an explicitly supplied key.
    pub fn request_page_key(
        &mut self,
        key: PageKey,
        priority: Priority,
        importance: u32,
    ) -> Result<PageId, PageRequestError> {
        self.residency.request(key, priority, importance)
    }

    /// Performs at most `max_pages` real synchronous range reads.
    ///
    /// Read failures are recorded in the report and the pump continues with
    /// other pending pages. Admission failures leave the pending page intact,
    /// so the pump stops for this call instead of retrying the same blocked
    /// page until the bound is exhausted.
    #[must_use]
    pub fn pump_io(&mut self, max_pages: usize) -> FixedPageIoReport {
        let mut report = FixedPageIoReport::default();
        while report.attempted < max_pages {
            if self.residency.next_pending().is_none() {
                break;
            }
            report.attempted += 1;
            match self.residency.load_next(&self.sources) {
                Ok(Some(_)) => report.loaded += 1,
                Ok(None) => break,
                Err(error) => {
                    let blocked = matches!(error, PageLoadError::Admission(_));
                    report.errors.push(error);
                    if blocked {
                        break;
                    }
                }
            }
        }
        report
    }

    /// Copies each newly resident page's bytes and source revision into the
    /// supplied render-side queue.
    ///
    /// Page IDs and revisions are tracked after a successful copy, so repeated
    /// extraction calls do not enqueue the same resident revision twice. The
    /// tracking table drops evicted records so long-running sessions do not
    /// accumulate historical page IDs.
    pub fn extract_to_render(&mut self, queue: &mut FixedPageRenderQueue) -> usize {
        let mut extracted = 0;
        let residents = self.residency.eviction_candidates();
        let resident_ids = residents.iter().copied().collect::<BTreeSet<_>>();
        let evicted = self
            .extracted
            .keys()
            .copied()
            .filter(|page| !resident_ids.contains(page))
            .collect::<Vec<_>>();
        for page in evicted {
            let metadata = self
                .extracted
                .get(&page)
                .expect("evicted page metadata is retained until extraction")
                .clone();
            queue.discard_upload(page);
            queue.push_release(FixedPageReleasePayload {
                page,
                slot: metadata.slot,
                revision: metadata.revision,
            });
            self.extracted.remove(&page);
        }
        for page in residents {
            let Some(key) = self.residency.key(page).cloned() else {
                continue;
            };
            let revision = key.source_revision();
            if self
                .extracted
                .get(&page)
                .is_some_and(|metadata| metadata.revision == revision)
            {
                continue;
            }
            let Some(bytes) = self.residency.resident_bytes(page) else {
                continue;
            };
            queue.push(FixedPageRenderPayload {
                page,
                slot: FixedPageSlotKey {
                    path: key.path().clone(),
                    page_index: key.page_index(),
                },
                revision,
                bytes: bytes.to_vec(),
            });
            self.extracted.insert(
                page,
                ExtractedPageMetadata {
                    slot: FixedPageSlotKey {
                        path: key.path().clone(),
                        page_index: key.page_index(),
                    },
                    revision,
                },
            );
            extracted += 1;
        }
        extracted
    }
}

/// Result of one bounded CPU I/O pump.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct FixedPageIoReport {
    /// Number of pending pages inspected and attempted.
    pub attempted: usize,
    /// Number of pages that became resident.
    pub loaded: usize,
    /// Terminal read errors or admission failures observed by the pump.
    pub errors: Vec<PageLoadError>,
}

/// Owned payload crossing from CPU extraction to the render side.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FixedPageRenderPayload {
    /// CPU residency identity of the page.
    pub page: PageId,
    /// Stable slot identity used to replace older source revisions.
    pub slot: FixedPageSlotKey,
    /// Source-table revision that produced the page bytes.
    pub revision: u64,
    /// Owned page bytes; the render side must not borrow CPU residency.
    pub bytes: Vec<u8>,
}

/// A FIFO queue of extracted page payloads.
#[derive(Debug, Default, Resource)]
pub struct FixedPageRenderQueue {
    pages: VecDeque<FixedPageRenderPayload>,
    releases: VecDeque<FixedPageReleasePayload>,
}

/// An extracted page that left CPU residency and must be retired on the GPU.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FixedPageReleasePayload {
    /// CPU residency identity of the page being retired.
    pub page: PageId,
    /// Stable slot identity for diagnostics and stale-release protection.
    pub slot: FixedPageSlotKey,
    /// Source revision that produced the page.
    pub revision: u64,
}

impl FixedPageRenderQueue {
    /// Creates an empty render-side queue.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the number of payloads waiting for explicit upload.
    #[must_use]
    pub fn len(&self) -> usize {
        self.pages.len()
    }

    /// Returns whether no payload is waiting for upload.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pages.is_empty() && self.releases.is_empty()
    }

    /// Removes the oldest payload for a render-side upload decision.
    pub fn pop(&mut self) -> Option<FixedPageRenderPayload> {
        self.pages.pop_front()
    }

    /// Puts a payload back at the front after a render-side admission failure.
    pub fn push_front(&mut self, payload: FixedPageRenderPayload) {
        self.pages.push_front(payload);
    }

    /// Queues a deferred GPU release for a CPU-evicted page.
    pub fn push_release(&mut self, payload: FixedPageReleasePayload) {
        self.releases.push_back(payload);
    }

    /// Removes a page upload that was extracted but not yet prepared.
    pub fn discard_upload(&mut self, page: PageId) {
        let mut retained = VecDeque::with_capacity(self.pages.len());
        while let Some(payload) = self.pages.pop_front() {
            if payload.page != page {
                retained.push_back(payload);
            }
        }
        self.pages = retained;
    }

    /// Removes the oldest deferred GPU release.
    pub fn pop_release(&mut self) -> Option<FixedPageReleasePayload> {
        self.releases.pop_front()
    }

    fn push(&mut self, payload: FixedPageRenderPayload) {
        self.pages.push_back(payload);
    }

    /// Uploads the oldest payload through an explicit byte-upload sink.
    ///
    /// The payload remains queued when the sink rejects it. Completion or
    /// reclamation proof is intentionally outside this contract.
    pub fn upload_next<U: FixedPageBytesUploader>(
        &mut self,
        uploader: &mut U,
    ) -> Result<bool, U::Error> {
        let Some(payload) = self.pages.front() else {
            return Ok(false);
        };
        uploader.upload_page_bytes(payload.page, payload.revision, &payload.bytes)?;
        self.pages.pop_front();
        Ok(true)
    }
}

/// Render-side upload contract for an owned extracted page.
///
/// Implementations may call a real render queue or another explicit upload
/// path. This trait intentionally has no completion method: fences, timeline
/// values, and deferred reclamation belong to the submission owner.
pub trait FixedPageBytesUploader {
    /// Error returned when the explicit byte upload is rejected.
    type Error;

    /// Uploads one page's bytes with its producer identity and revision.
    fn upload_page_bytes(
        &mut self,
        page: PageId,
        revision: u64,
        bytes: &[u8],
    ) -> Result<(), Self::Error>;
}

impl FixedPageStreamingSource {
    /// Returns whether a resident page has already been extracted.
    #[must_use]
    pub fn was_extracted(&self, page: PageId) -> bool {
        self.extracted.contains_key(&page)
    }

    /// Returns the number of page payloads extracted so far.
    #[must_use]
    pub fn extracted_page_count(&self) -> usize {
        self.extracted.len()
    }

    /// Returns the state of a page request, if its record is retained.
    #[must_use]
    pub fn page_state(&self, page: PageId) -> Option<PageState> {
        self.residency.state(page)
    }
}
