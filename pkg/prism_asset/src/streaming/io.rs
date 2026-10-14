//! Synchronous runtime range-read execution.
//!
//! This adapter is intentionally CPU-side: it calls the real
//! [`AssetSources::read_range`](crate::AssetSources::read_range) operation and
//! uses the controller's CPU/GPU/staging values only as residency accounting.
//! It does not allocate GPU resources, perform GPU upload, or imply
//! DirectStorage support.

use super::{
    StreamError, StreamFailure, StreamRequest, StreamRequestId, StreamState, StreamingController,
    SubmitError,
};
use crate::{AssetSources, ReadError};
use alloc::vec::Vec;
use core::fmt;

/// Errors reported by [`StreamIoExecutor`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StreamIoError {
    /// The request id is not present in the controller.
    Controller(StreamError),
    /// Admission failed before any bytes were read. The request remains
    /// pending so the caller may retry after making residency available.
    AdmissionRejected {
        id: StreamRequestId,
        error: StreamError,
    },
    /// The request was cancelled before execution.
    Cancelled { id: StreamRequestId },
    /// A request can only be executed once. Resident, failed, cancelled, and
    /// evicted requests are never read again by this adapter.
    AlreadyExecuted {
        id: StreamRequestId,
        state: StreamState,
    },
    /// The source read failed. The controller has already transitioned the
    /// request to `Failed` and released its loading residency.
    Read {
        id: StreamRequestId,
        error: ReadError,
    },
}

impl fmt::Display for StreamIoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Controller(error) => error.fmt(f),
            Self::AdmissionRejected { id, error } => {
                write!(f, "stream request {} was rejected: {error}", id.get())
            }
            Self::Cancelled { id } => write!(f, "stream request {} is cancelled", id.get()),
            Self::AlreadyExecuted { id, state } => {
                write!(f, "stream request {} is already in {state:?}", id.get())
            }
            Self::Read { id, error } => {
                write!(f, "stream request {} read failed: {error}", id.get())
            }
        }
    }
}

impl std::error::Error for StreamIoError {}

/// A synchronous adapter from [`StreamRequest`] to the existing source table.
///
/// Each successful call to [`Self::execute`] performs exactly one real
/// `AssetSources::read_range`, then transitions the request to `Resident`.
/// Failed reads transition to `Failed` and release all loading residency.
/// Cancellation is explicit and never pretends that an in-flight blocking
/// read can be interrupted; callers must cancel before execution starts.
pub struct StreamIoExecutor<'a> {
    sources: &'a AssetSources,
    controller: &'a mut StreamingController,
}

impl<'a> StreamIoExecutor<'a> {
    /// Creates an adapter over an existing source table and controller.
    #[must_use]
    pub fn new(sources: &'a AssetSources, controller: &'a mut StreamingController) -> Self {
        Self {
            sources,
            controller,
        }
    }

    /// Submits a request to the controller without reading it yet.
    pub fn submit(&mut self, request: StreamRequest) -> Result<StreamRequestId, SubmitError> {
        self.controller.submit(request)
    }

    /// Submits and immediately executes one request.
    pub fn submit_and_execute(
        &mut self,
        request: StreamRequest,
    ) -> Result<(StreamRequestId, Vec<u8>), SubmitAndExecuteError> {
        let id = self.submit(request)?;
        let bytes = self.execute(id)?;
        Ok((id, bytes))
    }

    /// Executes the highest-priority pending request, if one exists.
    ///
    /// Selection is delegated to [`StreamingController::next_pending`], so
    /// equal-priority requests retain the controller's deterministic order.
    pub fn execute_next(&mut self) -> Result<Option<(StreamRequestId, Vec<u8>)>, StreamIoError> {
        let id = self.controller.next_pending().map(|(id, _)| id);
        id.map(|id| self.execute(id).map(|bytes| (id, bytes)))
            .transpose()
    }

    /// Executes one pending request by performing its real source range read.
    pub fn execute(&mut self, id: StreamRequestId) -> Result<Vec<u8>, StreamIoError> {
        let state = self
            .controller
            .state(id)
            .ok_or(StreamIoError::Controller(StreamError::UnknownRequest(id)))?;
        match state {
            StreamState::Pending => {}
            StreamState::Cancelled => return Err(StreamIoError::Cancelled { id }),
            state => return Err(StreamIoError::AlreadyExecuted { id, state }),
        }

        let request = self
            .controller
            .request(id)
            .cloned()
            .ok_or(StreamIoError::Controller(StreamError::UnknownRequest(id)))?;

        if let Err(error) = self.controller.begin_load(id) {
            return match error {
                StreamError::BudgetExceeded { .. } | StreamError::BudgetOverflow => {
                    Err(StreamIoError::AdmissionRejected { id, error })
                }
                error => Err(StreamIoError::Controller(error)),
            };
        }

        match self
            .sources
            .read_range(&request.path, request.offset, request.len)
        {
            Ok(bytes) if u64::try_from(bytes.len()).ok() == Some(request.len) => {
                // No public operation between begin_load and this call can
                // change the state because the executor owns the controller
                // borrow. Keep the transition explicit nevertheless.
                self.controller
                    .complete_load(id)
                    .map_err(StreamIoError::Controller)?;
                Ok(bytes)
            }
            Ok(bytes) => {
                let error = ReadError::Io(format!(
                    "short read: requested {} bytes, received {}",
                    request.len,
                    bytes.len()
                ));
                self.controller
                    .fail(id, StreamFailure::Read)
                    .map_err(StreamIoError::Controller)?;
                Err(StreamIoError::Read { id, error })
            }
            Err(error) => {
                self.controller
                    .fail(id, StreamFailure::Read)
                    .map_err(StreamIoError::Controller)?;
                Err(StreamIoError::Read { id, error })
            }
        }
    }

    /// Cancels a pending or loading request and releases loading residency.
    pub fn cancel(&mut self, id: StreamRequestId) -> Result<(), StreamIoError> {
        self.controller
            .cancel(id)
            .map_err(StreamIoError::Controller)
    }

    /// Borrows the controller for inspection and eviction/priority operations.
    #[must_use]
    pub fn controller(&self) -> &StreamingController {
        self.controller
    }

    /// Borrows the controller mutably for explicit lifecycle operations such
    /// as eviction. I/O execution itself remains serialized through `&mut self`.
    pub fn controller_mut(&mut self) -> &mut StreamingController {
        self.controller
    }
}

/// Error returned by [`StreamIoExecutor::submit_and_execute`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SubmitAndExecuteError {
    /// Request validation or id allocation failed before execution.
    Submit(SubmitError),
    /// Execution failed after a request id was assigned.
    Execute(StreamIoError),
}

impl fmt::Display for SubmitAndExecuteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Submit(error) => error.fmt(f),
            Self::Execute(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for SubmitAndExecuteError {}

impl From<SubmitError> for SubmitAndExecuteError {
    fn from(error: SubmitError) -> Self {
        Self::Submit(error)
    }
}

impl From<StreamIoError> for SubmitAndExecuteError {
    fn from(error: StreamIoError) -> Self {
        Self::Execute(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AssetMeta, AssetReader, MemSource};
    use alloc::sync::Arc;
    use prism_tasks::Priority;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn request(path: &str, bytes: u64) -> StreamRequest {
        StreamRequest {
            path: path.into(),
            offset: 0,
            len: bytes,
            residency: super::super::Residency::new(bytes, bytes, bytes),
            priority: Priority::Normal,
            importance: 0,
        }
    }

    fn sources_with(path: &str, bytes: &[u8]) -> AssetSources {
        let mut source = MemSource::new();
        source.insert(path, Arc::<[u8]>::from(bytes));
        let mut sources = AssetSources::new();
        sources.mount("mem", Arc::new(source), 0);
        sources
    }

    struct FailingSource;

    impl AssetReader for FailingSource {
        fn read(&self, _path: &str) -> Result<Vec<u8>, ReadError> {
            Err(ReadError::Io("injected failure".into()))
        }

        fn metadata(&self, _path: &str) -> Result<AssetMeta, ReadError> {
            Err(ReadError::Io("injected failure".into()))
        }
    }

    struct CountingSource {
        reads: Arc<AtomicUsize>,
        bytes: &'static [u8],
    }

    struct ShortSource;

    impl AssetReader for ShortSource {
        fn read(&self, _path: &str) -> Result<Vec<u8>, ReadError> {
            Ok(vec![1, 2])
        }

        fn read_range(&self, _path: &str, _offset: u64, _len: u64) -> Result<Vec<u8>, ReadError> {
            Ok(vec![1, 2])
        }

        fn metadata(&self, _path: &str) -> Result<AssetMeta, ReadError> {
            Ok(AssetMeta {
                size: 4,
                modified_ms: None,
                is_dir: false,
            })
        }
    }

    impl AssetReader for CountingSource {
        fn read(&self, _path: &str) -> Result<Vec<u8>, ReadError> {
            self.reads.fetch_add(1, Ordering::SeqCst);
            Ok(self.bytes.to_vec())
        }

        fn read_range(&self, _path: &str, offset: u64, len: u64) -> Result<Vec<u8>, ReadError> {
            self.reads.fetch_add(1, Ordering::SeqCst);
            let size = self.bytes.len() as u64;
            let end = offset
                .checked_add(len)
                .ok_or(ReadError::OutOfRange { offset, len, size })?;
            let start =
                usize::try_from(offset).map_err(|_| ReadError::OutOfRange { offset, len, size })?;
            let end =
                usize::try_from(end).map_err(|_| ReadError::OutOfRange { offset, len, size })?;
            self.bytes
                .get(start..end)
                .map(ToOwned::to_owned)
                .ok_or(ReadError::OutOfRange { offset, len, size })
        }

        fn metadata(&self, _path: &str) -> Result<AssetMeta, ReadError> {
            Ok(AssetMeta {
                size: self.bytes.len() as u64,
                modified_ms: None,
                is_dir: false,
            })
        }
    }

    #[test]
    fn successful_nonzero_range_read_becomes_resident_and_keeps_residency() {
        let sources = sources_with("hero.bin", b"abcdef");
        let mut controller =
            StreamingController::new(super::super::ResidencyLimits::new(16, 16, 16));
        let mut executor = StreamIoExecutor::new(&sources, &mut controller);
        let id = executor
            .submit(StreamRequest {
                path: "mem://hero.bin".into(),
                offset: 2,
                len: 3,
                residency: super::super::Residency::new(3, 3, 3),
                priority: Priority::Normal,
                importance: 0,
            })
            .expect("request should submit");
        let bytes = executor.execute(id).expect("range read should work");
        assert_eq!(bytes, b"cde");
        assert_eq!(executor.controller().state(id), Some(StreamState::Resident));
        assert_eq!(
            executor.controller().used(),
            super::super::Residency::new(3, 3, 3)
        );
    }

    #[test]
    fn source_failure_marks_failed_and_releases_all_budget() {
        let mut sources = AssetSources::new();
        sources.mount("bad", Arc::new(FailingSource), 0);
        let mut controller =
            StreamingController::new(super::super::ResidencyLimits::new(16, 16, 16));
        let mut executor = StreamIoExecutor::new(&sources, &mut controller);
        let id = executor.submit(request("bad://hero.bin", 4)).unwrap();
        let error = executor.execute(id).unwrap_err();
        assert!(matches!(error, StreamIoError::Read { .. }));
        assert_eq!(executor.controller().state(id), Some(StreamState::Failed));
        assert_eq!(
            executor.controller().used(),
            super::super::Residency::default()
        );
    }

    #[test]
    fn short_read_marks_failed_and_releases_all_budget() {
        let mut sources = AssetSources::new();
        sources.mount("short", Arc::new(ShortSource), 0);
        let mut controller =
            StreamingController::new(super::super::ResidencyLimits::new(16, 16, 16));
        let mut executor = StreamIoExecutor::new(&sources, &mut controller);
        let id = executor.submit(request("short://hero.bin", 4)).unwrap();

        let error = executor.execute(id).unwrap_err();
        assert!(matches!(error, StreamIoError::Read { .. }));
        assert_eq!(executor.controller().state(id), Some(StreamState::Failed));
        assert_eq!(
            executor.controller().used(),
            super::super::Residency::default()
        );
    }

    #[test]
    fn cancellation_prevents_read_and_releases_loading_budget() {
        let sources = sources_with("hero.bin", b"abcdef");
        let mut controller =
            StreamingController::new(super::super::ResidencyLimits::new(16, 16, 16));
        let mut executor = StreamIoExecutor::new(&sources, &mut controller);
        let id = executor.submit(request("mem://hero.bin", 3)).unwrap();
        executor.controller_mut().begin_load(id).unwrap();
        executor.cancel(id).unwrap();
        assert_eq!(
            executor.controller().state(id),
            Some(StreamState::Cancelled)
        );
        assert_eq!(
            executor.controller().used(),
            super::super::Residency::default()
        );
        assert_eq!(executor.execute(id), Err(StreamIoError::Cancelled { id }));
    }

    #[test]
    fn range_overflow_is_rejected_before_a_request_id_is_created() {
        let sources = sources_with("hero.bin", b"abcdef");
        let mut controller =
            StreamingController::new(super::super::ResidencyLimits::new(16, 16, 16));
        let mut executor = StreamIoExecutor::new(&sources, &mut controller);
        let result = executor.submit(request("mem://hero.bin", 1));
        assert!(result.is_ok());
        let overflow = StreamRequest {
            path: "mem://hero.bin".into(),
            offset: u64::MAX,
            len: 1,
            residency: super::super::Residency::default(),
            priority: Priority::Normal,
            importance: 0,
        };
        assert_eq!(
            executor.submit(overflow),
            Err(SubmitError::Invalid(StreamError::RangeOverflow))
        );
    }

    #[test]
    fn budget_rejection_stays_pending_without_read_or_residency() {
        let reads = Arc::new(AtomicUsize::new(0));
        let mut sources = AssetSources::new();
        sources.mount(
            "counting",
            Arc::new(CountingSource {
                reads: Arc::clone(&reads),
                bytes: b"abcdef",
            }),
            0,
        );
        let mut controller = StreamingController::new(super::super::ResidencyLimits::new(2, 2, 2));
        let mut executor = StreamIoExecutor::new(&sources, &mut controller);
        let id = executor.submit(request("counting://hero.bin", 3)).unwrap();
        assert!(matches!(
            executor.execute(id),
            Err(StreamIoError::AdmissionRejected { id: actual, .. }) if actual == id
        ));
        assert_eq!(reads.load(Ordering::SeqCst), 0);
        assert_eq!(executor.controller().state(id), Some(StreamState::Pending));
        assert_eq!(
            executor.controller().used(),
            super::super::Residency::default()
        );
    }

    #[test]
    fn cancellation_does_not_call_the_reader() {
        let reads = Arc::new(AtomicUsize::new(0));
        let mut sources = AssetSources::new();
        sources.mount(
            "counting",
            Arc::new(CountingSource {
                reads: Arc::clone(&reads),
                bytes: b"abcdef",
            }),
            0,
        );
        let mut controller =
            StreamingController::new(super::super::ResidencyLimits::new(16, 16, 16));
        let mut executor = StreamIoExecutor::new(&sources, &mut controller);
        let id = executor.submit(request("counting://hero.bin", 3)).unwrap();
        executor.cancel(id).unwrap();
        assert_eq!(executor.execute(id), Err(StreamIoError::Cancelled { id }));
        assert_eq!(reads.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn unknown_and_repeated_requests_are_explicit_and_deterministic() {
        let sources = sources_with("hero.bin", b"abcdef");
        let mut controller =
            StreamingController::new(super::super::ResidencyLimits::new(16, 16, 16));
        let mut executor = StreamIoExecutor::new(&sources, &mut controller);
        let unknown = StreamRequestId(999);
        assert_eq!(
            executor.execute(unknown),
            Err(StreamIoError::Controller(StreamError::UnknownRequest(
                unknown
            )))
        );
        let id = executor.submit(request("mem://hero.bin", 3)).unwrap();
        assert_eq!(executor.execute(id).unwrap(), b"abc");
        assert_eq!(
            executor.execute(id),
            Err(StreamIoError::AlreadyExecuted {
                id,
                state: StreamState::Resident
            })
        );
        assert_eq!(
            executor.controller().used(),
            super::super::Residency::new(3, 3, 3)
        );
    }

    #[test]
    fn next_pending_uses_controller_order_and_submit_and_execute_is_atomic_at_api_boundary() {
        let sources = sources_with("hero.bin", b"abcdef");
        let mut controller =
            StreamingController::new(super::super::ResidencyLimits::new(16, 16, 16));
        let mut executor = StreamIoExecutor::new(&sources, &mut controller);
        let first = executor.submit(request("mem://hero.bin", 1)).unwrap();
        let second = executor.submit(request("mem://hero.bin", 2)).unwrap();
        assert_eq!(executor.execute_next().unwrap().unwrap().0, first);
        assert_eq!(executor.execute_next().unwrap().unwrap().0, second);
        assert_eq!(executor.execute_next().unwrap(), None);
        let (third, bytes) = executor
            .submit_and_execute(request("mem://hero.bin", 2))
            .unwrap();
        assert_eq!(third.get(), 2);
        assert_eq!(bytes, b"ab");
    }
}
