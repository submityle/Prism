use alloc::sync::Arc;
use alloc::vec::Vec;

use prism_asset::{
    AssetSources, MemSource, PageBudget, PageId, PageResidency, PageSize, PageState,
};
use prism_tasks::Priority;

use super::{FixedPageBytesUploader, FixedPageRenderQueue, FixedPageStreamingSource};

fn source() -> AssetSources {
    let mut sources = AssetSources::new();
    let mut memory = MemSource::new();
    memory.insert("payload.bin", b"abcdefgh".to_vec());
    sources.mount("memory", Arc::new(memory), 0);
    sources
}

fn streaming(max_bytes: u64) -> FixedPageStreamingSource {
    let residency = PageResidency::new(
        PageSize::new(4).expect("valid page size"),
        PageBudget::new(max_bytes, 1).expect("valid page budget"),
    )
    .expect("aligned residency budget");
    FixedPageStreamingSource::new(source(), residency)
}

#[test]
fn request_load_extract_copies_owned_bytes_and_revision() {
    let mut streaming = streaming(8);
    let page = streaming
        .request_page("memory://payload.bin", 1, Priority::Normal, 3)
        .expect("request is registered");

    let report = streaming.pump_io(1);
    assert_eq!(report.attempted, 1);
    assert_eq!(report.loaded, 1);
    assert!(report.errors.is_empty());
    assert_eq!(streaming.page_state(page), Some(PageState::Resident));

    let mut queue = FixedPageRenderQueue::new();
    assert_eq!(streaming.extract_to_render(&mut queue), 1);
    assert_eq!(queue.len(), 1);
    let payload = queue.pop().expect("resident page payload");
    assert_eq!(payload.page, page);
    assert_eq!(payload.revision, streaming.sources().revision());
    assert_eq!(payload.bytes, b"efgh");
    assert_eq!(streaming.extract_to_render(&mut queue), 0);
    assert!(queue.is_empty());
}

#[test]
fn extraction_tracking_drops_evicted_page_ids() {
    let mut streaming = streaming(4);
    let page = streaming
        .request_page("memory://payload.bin", 0, Priority::Normal, 0)
        .expect("request is registered");
    let _ = streaming.pump_io(1);
    let mut queue = FixedPageRenderQueue::new();
    assert_eq!(streaming.extract_to_render(&mut queue), 1);
    assert_eq!(streaming.extracted_page_count(), 1);
    streaming
        .residency_mut()
        .evict(page)
        .expect("resident page evicts");
    assert_eq!(streaming.extract_to_render(&mut queue), 0);
    assert_eq!(streaming.extracted_page_count(), 0);
}

#[test]
fn pump_is_bounded_and_reports_budget_blocking() {
    let mut streaming = streaming(4);
    streaming
        .request_page("memory://payload.bin", 0, Priority::Normal, 0)
        .expect("first request");
    let second = streaming
        .request_page("memory://payload.bin", 1, Priority::Normal, 0)
        .expect("second request");

    let report = streaming.pump_io(8);
    assert_eq!(report.attempted, 2);
    assert_eq!(report.loaded, 1);
    assert_eq!(report.errors.len(), 1);
    assert!(matches!(
        report.errors[0],
        prism_asset::PageLoadError::Admission(
            prism_asset::PageAdmissionError::BudgetExceeded { .. }
        )
    ));
    assert_eq!(streaming.page_state(second), Some(PageState::Pending));

    let empty = streaming.pump_io(0);
    assert_eq!(empty.attempted, 0);
    assert_eq!(empty.loaded, 0);
    assert!(empty.errors.is_empty());
}

#[test]
fn read_failure_is_reported_and_duplicate_requests_share_one_page() {
    let mut streaming = streaming(4);
    let first = streaming
        .request_page("memory://missing.bin", 0, Priority::Low, 0)
        .expect("missing path still registers");
    let duplicate = streaming
        .request_page("memory://missing.bin", 0, Priority::Critical, 9)
        .expect("duplicate request");
    assert_eq!(first, duplicate);

    let report = streaming.pump_io(1);
    assert_eq!(report.attempted, 1);
    assert_eq!(report.loaded, 0);
    assert_eq!(report.errors.len(), 1);
    assert_eq!(streaming.page_state(first), Some(PageState::Failed));

    let mut queue = FixedPageRenderQueue::new();
    assert_eq!(streaming.extract_to_render(&mut queue), 0);
    assert!(queue.is_empty());
}

#[derive(Default)]
struct RecordingUploader {
    uploads: Vec<(PageId, u64, Vec<u8>)>,
}

impl FixedPageBytesUploader for RecordingUploader {
    type Error = core::convert::Infallible;

    fn upload_page_bytes(
        &mut self,
        page: PageId,
        revision: u64,
        bytes: &[u8],
    ) -> Result<(), Self::Error> {
        self.uploads.push((page, revision, bytes.to_vec()));
        Ok(())
    }
}

#[test]
fn render_queue_uploads_only_explicit_owned_bytes() {
    let mut streaming = streaming(4);
    let page = streaming
        .request_page("memory://payload.bin", 0, Priority::Normal, 0)
        .expect("request is registered");
    let _ = streaming.pump_io(1);

    let mut queue = FixedPageRenderQueue::new();
    streaming.extract_to_render(&mut queue);
    let mut uploader = RecordingUploader::default();
    assert_eq!(queue.upload_next(&mut uploader), Ok(true));
    assert!(queue.is_empty());
    assert_eq!(uploader.uploads.len(), 1);
    assert_eq!(uploader.uploads[0].0, page);
    assert_eq!(uploader.uploads[0].1, streaming.sources().revision());
    assert_eq!(uploader.uploads[0].2, b"abcd");
}
