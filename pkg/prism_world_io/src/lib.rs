#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![doc = "Engine-neutral chunked IO contracts with an optional std scheduler and file provider."]

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;
use core::task::{Context, Poll};

/// Stable identity assigned to a request by [`IoScheduler`].
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct RequestId(u64);

impl RequestId {
    /// Returns the numeric request identity.
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl fmt::Display for RequestId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

/// A deterministic, bounded read request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReadRequest {
    offset: u64,
    length: u64,
    chunk_size: usize,
    priority: u32,
    expected_checksum: Option<u64>,
}

impl ReadRequest {
    /// Creates a request using the default 64 KiB chunk size and priority zero.
    pub const fn new(offset: u64, length: u64) -> Self {
        Self {
            offset,
            length,
            chunk_size: 64 * 1024,
            priority: 0,
            expected_checksum: None,
        }
    }

    /// Sets the maximum number of bytes returned by one poll.
    pub const fn with_chunk_size(mut self, chunk_size: usize) -> Result<Self, ConfigError> {
        if chunk_size == 0 {
            return Err(ConfigError::ZeroChunkSize);
        }
        self.chunk_size = chunk_size;
        Ok(self)
    }

    /// Sets admission priority. Higher values start first; active reads share turns.
    pub const fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    /// Sets an optional FNV-1a checksum to verify when the read completes.
    pub const fn with_expected_checksum(mut self, checksum: u64) -> Self {
        self.expected_checksum = Some(checksum);
        self
    }

    /// Returns the byte offset at which the read starts.
    pub const fn offset(&self) -> u64 {
        self.offset
    }

    /// Returns the requested byte length.
    pub const fn length(&self) -> u64 {
        self.length
    }

    /// Returns the maximum chunk size.
    pub const fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    /// Returns the scheduling priority.
    pub const fn priority(&self) -> u32 {
        self.priority
    }

    /// Returns the expected checksum, if configured.
    pub const fn expected_checksum(&self) -> Option<u64> {
        self.expected_checksum
    }

    #[cfg(feature = "std")]
    fn validate(&self) -> Result<(), IoError> {
        self.offset
            .checked_add(self.length)
            .ok_or(IoError::RangeOverflow)
            .map(|_| ())
    }
}

/// Errors that indicate invalid scheduler or request configuration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConfigError {
    /// A chunk size of zero cannot make progress.
    ZeroChunkSize,
    /// A scheduler must have at least one in-flight slot.
    ZeroInFlight,
    /// A scheduler must accept at least one outstanding request.
    ZeroCapacity,
}

impl fmt::Display for ConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::ZeroChunkSize => "chunk size must be greater than zero",
            Self::ZeroInFlight => "in-flight capacity must be greater than zero",
            Self::ZeroCapacity => "outstanding capacity must be greater than zero",
        };
        formatter.write_str(message)
    }
}

/// Errors produced by providers or by scheduler-level stream validation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum IoError {
    /// The request was cancelled before its next result was delivered.
    Cancelled { request_id: RequestId },
    /// The scheduler's outstanding request limit has been reached.
    Backpressure { capacity: usize },
    /// Every request identity has been assigned; identities are never reused.
    RequestIdExhausted,
    /// The request range overflows `u64`.
    RangeOverflow,
    /// A provider returned a chunk at an unexpected byte offset.
    InvalidChunkOffset { expected: u64, actual: u64 },
    /// A provider returned more bytes than the request can contain.
    InvalidChunkLength { remaining: u64, actual: usize },
    /// The provider reached EOF before returning the requested bytes.
    ShortRead { expected: u64, actual: u64 },
    /// The completed byte stream did not match the expected checksum.
    ChecksumMismatch { expected: u64, actual: u64 },
    /// A provider-specific error represented without exposing its backend type.
    Provider { message: String },
}

impl fmt::Display for IoError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cancelled { request_id } => {
                write!(formatter, "request {request_id} was cancelled")
            }
            Self::Backpressure { capacity } => {
                write!(formatter, "outstanding request capacity {capacity} is full")
            }
            Self::RequestIdExhausted => formatter.write_str("request identities exhausted"),
            Self::RangeOverflow => formatter.write_str("request range overflows u64"),
            Self::InvalidChunkOffset { expected, actual } => {
                write!(
                    formatter,
                    "chunk offset {actual} does not match expected offset {expected}"
                )
            }
            Self::InvalidChunkLength { remaining, actual } => {
                write!(
                    formatter,
                    "chunk length {actual} exceeds remaining request length {remaining}"
                )
            }
            Self::ShortRead { expected, actual } => {
                write!(
                    formatter,
                    "short read: expected {expected} bytes, received {actual}"
                )
            }
            Self::ChecksumMismatch { expected, actual } => {
                write!(
                    formatter,
                    "checksum mismatch: expected {expected:#x}, received {actual:#x}"
                )
            }
            Self::Provider { message } => write!(formatter, "provider error: {message}"),
        }
    }
}

/// One contiguous result emitted by a chunked read operation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReadChunk {
    offset: u64,
    bytes: Vec<u8>,
}

impl ReadChunk {
    /// Creates a chunk at `offset`.
    pub fn new(offset: u64, bytes: Vec<u8>) -> Self {
        Self { offset, bytes }
    }

    /// Returns the chunk's source offset.
    pub const fn offset(&self) -> u64 {
        self.offset
    }

    /// Returns the chunk payload.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Consumes the chunk and returns its payload.
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

/// Final accounting for a successfully completed request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReadSummary {
    request_id: RequestId,
    offset: u64,
    length: u64,
    checksum: u64,
}

impl ReadSummary {
    /// Returns the completed request identity.
    pub const fn request_id(self) -> RequestId {
        self.request_id
    }

    /// Returns the source offset.
    pub const fn offset(self) -> u64 {
        self.offset
    }

    /// Returns the number of bytes delivered.
    pub const fn length(self) -> u64 {
        self.length
    }

    /// Returns the deterministic checksum of the delivered bytes.
    pub const fn checksum(self) -> u64 {
        self.checksum
    }
}

/// Scheduler output: either one chunk or a completion summary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum IoEvent {
    /// A chunk belonging to `request_id`.
    Chunk {
        request_id: RequestId,
        chunk: ReadChunk,
    },
    /// The request has passed length and checksum validation.
    Completed(ReadSummary),
}

/// A pollable provider operation. Implementations may perform bounded work per poll.
pub trait ChunkedRead {
    /// Produces one chunk, reports completion with `None`, or returns an error.
    fn poll_chunk(&mut self, context: &mut Context<'_>)
        -> Poll<Result<Option<ReadChunk>, IoError>>;

    /// Requests cancellation. The scheduler makes cancellation observable on its next poll.
    fn cancel(&mut self);
}

/// Engine-neutral factory for chunked read operations.
pub trait ReadProvider {
    /// The operation type produced by this provider.
    type Operation: ChunkedRead;

    /// Starts one operation for a scheduler-assigned request identity.
    fn start(
        &self,
        request: &ReadRequest,
        request_id: RequestId,
    ) -> Result<Self::Operation, IoError>;
}

/// Computes the deterministic FNV-1a 64-bit checksum used by this crate.
pub const fn checksum(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325;
    let mut index = 0;
    while index < bytes.len() {
        hash ^= bytes[index] as u64;
        hash = hash.wrapping_mul(0x100000001b3);
        index += 1;
    }
    hash
}

#[cfg(feature = "std")]
mod std_support {
    use super::*;
    use alloc::collections::{BTreeMap, BTreeSet, BinaryHeap};
    use alloc::vec;
    use core::cmp::Ordering;
    use core::ops::Bound::{Excluded, Unbounded};
    use std::fs::File;
    use std::io::{ErrorKind, Read, Seek, SeekFrom};
    use std::path::{Path, PathBuf};

    struct QueuedRequest {
        request: ReadRequest,
        request_id: RequestId,
    }

    impl Eq for QueuedRequest {}

    impl PartialEq for QueuedRequest {
        fn eq(&self, other: &Self) -> bool {
            self.request_id == other.request_id
        }
    }

    impl Ord for QueuedRequest {
        fn cmp(&self, other: &Self) -> Ordering {
            self.request
                .priority
                .cmp(&other.request.priority)
                .then_with(|| other.request_id.cmp(&self.request_id))
        }
    }

    impl PartialOrd for QueuedRequest {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    struct ActiveOperation<O> {
        request: ReadRequest,
        operation: O,
        next_offset: u64,
        delivered: u64,
        checksum: u64,
        cancelled: bool,
    }

    /// A bounded scheduler with priority/FIFO admission and round-robin active reads.
    ///
    /// Priority does not preempt already active operations. Each active operation is
    /// visited at most once per poll; a ready chunk rotates the next poll's first turn.
    pub struct IoScheduler<P: ReadProvider> {
        provider: P,
        max_in_flight: usize,
        capacity: usize,
        next_request_id: Option<u64>,
        last_polled: Option<RequestId>,
        queue: BinaryHeap<QueuedRequest>,
        active: BTreeMap<RequestId, ActiveOperation<P::Operation>>,
        cancelled_ready: BTreeSet<RequestId>,
    }

    impl<P: ReadProvider> IoScheduler<P> {
        /// Creates a scheduler with a maximum number of active operations and total requests.
        pub fn new(
            provider: P,
            max_in_flight: usize,
            capacity: usize,
        ) -> Result<Self, ConfigError> {
            if max_in_flight == 0 {
                return Err(ConfigError::ZeroInFlight);
            }
            if capacity == 0 {
                return Err(ConfigError::ZeroCapacity);
            }
            Ok(Self {
                provider,
                max_in_flight,
                capacity,
                next_request_id: Some(1),
                last_polled: None,
                queue: BinaryHeap::new(),
                active: BTreeMap::new(),
                cancelled_ready: BTreeSet::new(),
            })
        }

        #[cfg(test)]
        pub(crate) fn set_next_request_id_for_test(&mut self, next: u64) {
            self.next_request_id = Some(next);
        }

        /// Adds a request, reporting a full capacity or exhausted identities explicitly.
        ///
        /// Nonzero identities increase monotonically for this scheduler's lifetime,
        /// including cancelled and completed requests. Exhaustion never wraps or resets.
        pub fn submit(&mut self, request: ReadRequest) -> Result<RequestId, IoError> {
            if self.len() >= self.capacity {
                return Err(IoError::Backpressure {
                    capacity: self.capacity,
                });
            }
            request.validate()?;
            let next = self.next_request_id.ok_or(IoError::RequestIdExhausted)?;
            let request_id = RequestId(next);
            self.next_request_id = next.checked_add(1);
            let queued = QueuedRequest {
                request,
                request_id,
            };
            self.queue.push(queued);
            Ok(request_id)
        }

        /// Cancels a queued or active request. The cancellation result is delivered by `poll`.
        pub fn cancel(&mut self, request_id: RequestId) -> bool {
            if let Some(active) = self.active.get_mut(&request_id) {
                if !active.cancelled {
                    active.cancelled = true;
                    active.operation.cancel();
                    return true;
                }
                return false;
            }
            let mut found = false;
            self.queue.retain(|queued| {
                if queued.request_id == request_id {
                    found = true;
                }
                queued.request_id != request_id
            });
            if found {
                self.cancelled_ready.insert(request_id);
            }
            found
        }

        /// Returns the number of queued or active requests not yet reported.
        pub fn len(&self) -> usize {
            self.queue.len() + self.active.len() + self.cancelled_ready.len()
        }

        /// Returns whether there is no outstanding work.
        pub fn is_empty(&self) -> bool {
            self.len() == 0
        }

        /// Polls one deterministic scheduler event.
        pub fn poll(&mut self, context: &mut Context<'_>) -> Poll<Result<IoEvent, IoError>> {
            if let Some(request_id) = self.cancelled_ready.pop_first() {
                return Poll::Ready(Err(IoError::Cancelled { request_id }));
            }
            while self.active.len() < self.max_in_flight {
                let Some(queued) = self.queue.pop() else {
                    break;
                };
                let request_id = queued.request_id;
                let request = queued.request;
                let operation = match self.provider.start(&request, request_id) {
                    Ok(operation) => operation,
                    Err(error) => return Poll::Ready(Err(error)),
                };
                self.active.insert(
                    request_id,
                    ActiveOperation {
                        next_offset: request.offset,
                        request,
                        operation,
                        delivered: 0,
                        checksum: 0xcbf29ce484222325,
                        cancelled: false,
                    },
                );
            }

            for _ in 0..self.active.len() {
                let next = self
                    .last_polled
                    .and_then(|last| self.active.range((Excluded(last), Unbounded)).next());
                let Some((&request_id, _)) = next.or_else(|| self.active.first_key_value()) else {
                    break;
                };
                self.last_polled = Some(request_id);
                let active = self
                    .active
                    .get_mut(&request_id)
                    .expect("selected active request");
                let event = active.poll(request_id, context);
                if matches!(event, Poll::Ready(Err(_) | Ok(IoEvent::Completed(_)))) {
                    self.active.remove(&request_id);
                }
                if event.is_ready() {
                    return event;
                }
            }
            Poll::Pending
        }
    }

    impl<O: ChunkedRead> ActiveOperation<O> {
        fn poll(
            &mut self,
            request_id: RequestId,
            context: &mut Context<'_>,
        ) -> Poll<Result<IoEvent, IoError>> {
            if self.cancelled {
                return Poll::Ready(Err(IoError::Cancelled { request_id }));
            }
            match self.operation.poll_chunk(context) {
                Poll::Pending => Poll::Pending,
                Poll::Ready(Err(error)) => Poll::Ready(Err(error)),
                Poll::Ready(Ok(Some(chunk))) => {
                    if chunk.offset != self.next_offset {
                        return Poll::Ready(Err(IoError::InvalidChunkOffset {
                            expected: self.next_offset,
                            actual: chunk.offset,
                        }));
                    }
                    let chunk_length = chunk.bytes.len();
                    let remaining = self.request.length - self.delivered;
                    if chunk_length as u64 > remaining {
                        return Poll::Ready(Err(IoError::InvalidChunkLength {
                            remaining,
                            actual: chunk_length,
                        }));
                    }
                    self.next_offset += chunk_length as u64;
                    self.delivered += chunk_length as u64;
                    for byte in &chunk.bytes {
                        self.checksum ^= *byte as u64;
                        self.checksum = self.checksum.wrapping_mul(0x100000001b3);
                    }
                    Poll::Ready(Ok(IoEvent::Chunk { request_id, chunk }))
                }
                Poll::Ready(Ok(None)) => {
                    if self.delivered != self.request.length {
                        return Poll::Ready(Err(IoError::ShortRead {
                            expected: self.request.length,
                            actual: self.delivered,
                        }));
                    }
                    if let Some(expected) = self.request.expected_checksum
                        && expected != self.checksum
                    {
                        return Poll::Ready(Err(IoError::ChecksumMismatch {
                            expected,
                            actual: self.checksum,
                        }));
                    }
                    Poll::Ready(Ok(IoEvent::Completed(ReadSummary {
                        request_id,
                        offset: self.request.offset,
                        length: self.delivered,
                        checksum: self.checksum,
                    })))
                }
            }
        }
    }

    /// A std-backed provider that reads a regular file in bounded poll-sized chunks.
    #[derive(Clone, Debug, Eq, PartialEq)]
    pub struct FileProvider {
        path: PathBuf,
    }

    impl FileProvider {
        /// Creates a provider for `path` without opening it yet.
        pub fn new(path: impl Into<PathBuf>) -> Self {
            Self { path: path.into() }
        }

        /// Returns the configured path.
        pub fn path(&self) -> &Path {
            &self.path
        }
    }

    /// A pollable file operation owned by [`FileProvider`].
    pub struct FileReadOperation {
        request_id: RequestId,
        file: File,
        next_offset: u64,
        remaining: u64,
        chunk_size: usize,
        cancelled: bool,
    }

    pub(super) fn read_chunk_fully<R: Read>(
        reader: &mut R,
        bytes: &mut [u8],
    ) -> Result<(), IoError> {
        let mut actual = 0;
        while actual < bytes.len() {
            match reader.read(&mut bytes[actual..]) {
                Ok(0) => {
                    return Err(IoError::ShortRead {
                        expected: bytes.len() as u64,
                        actual: actual as u64,
                    });
                }
                Ok(read) => actual += read,
                Err(error) if error.kind() == ErrorKind::Interrupted => continue,
                Err(error) => {
                    return Err(IoError::Provider {
                        message: error.to_string(),
                    });
                }
            }
        }
        Ok(())
    }

    impl ChunkedRead for FileReadOperation {
        fn poll_chunk(
            &mut self,
            _context: &mut Context<'_>,
        ) -> Poll<Result<Option<ReadChunk>, IoError>> {
            if self.cancelled {
                return Poll::Ready(Err(IoError::Cancelled {
                    request_id: self.request_id,
                }));
            }
            if self.remaining == 0 {
                return Poll::Ready(Ok(None));
            }
            let bytes_to_read = self.remaining.min(self.chunk_size as u64) as usize;
            let mut bytes = vec![0; bytes_to_read];
            if let Err(error) = read_chunk_fully(&mut self.file, &mut bytes) {
                return Poll::Ready(Err(error));
            }
            let offset = self.next_offset;
            self.next_offset += bytes_to_read as u64;
            self.remaining -= bytes_to_read as u64;
            Poll::Ready(Ok(Some(ReadChunk::new(offset, bytes))))
        }

        fn cancel(&mut self) {
            self.cancelled = true;
        }
    }

    impl ReadProvider for FileProvider {
        type Operation = FileReadOperation;

        fn start(
            &self,
            request: &ReadRequest,
            request_id: RequestId,
        ) -> Result<Self::Operation, IoError> {
            request.validate()?;
            let mut file = File::open(&self.path).map_err(|error| IoError::Provider {
                message: error.to_string(),
            })?;
            file.seek(SeekFrom::Start(request.offset))
                .map_err(|error| IoError::Provider {
                    message: error.to_string(),
                })?;
            Ok(FileReadOperation {
                request_id,
                file,
                next_offset: request.offset,
                remaining: request.length,
                chunk_size: request.chunk_size,
                cancelled: false,
            })
        }
    }

    pub use self::FileProvider as FileProviderExport;
    pub use self::IoScheduler as IoSchedulerExport;
}

#[cfg(feature = "std")]
pub use std_support::{FileProviderExport as FileProvider, IoSchedulerExport as IoScheduler};

#[cfg(feature = "std")]
mod threaded_file;
#[cfg(feature = "std")]
pub use threaded_file::{
    QueuedThreadedFileProvider, QueuedThreadedFileReadOperation, ThreadedFileConfig,
    ThreadedFileProvider, ThreadedFileReadOperation,
};

#[cfg(all(feature = "reactor", unix))]
mod reactor;
#[cfg(all(feature = "reactor", unix))]
pub use reactor::{
    QueuedReactorFileProvider, QueuedReactorFileReadOperation, ReactorFileConfig,
    ReactorFileProvider, ReactorFileReadOperation,
};

#[cfg(all(test, feature = "reactor", unix))]
mod reactor_tests;

#[cfg(all(test, feature = "std"))]
mod threaded_file_tests;

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use alloc::collections::BTreeMap;
    use alloc::vec;
    use core::task::{Context, Poll, Waker};
    use std::fs;
    use std::io::{Read, Write};
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn context() -> Context<'static> {
        Context::from_waker(Waker::noop())
    }

    fn temp_file(bytes: &[u8]) -> PathBuf {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("prism_world_io_{}_{}.bin", std::process::id(), id));
        let mut file = fs::File::create(&path).expect("temp file creation must succeed");
        file.write_all(bytes).expect("temp file write must succeed");
        path
    }

    struct OneByteReader {
        bytes: &'static [u8],
        offset: usize,
    }

    impl Read for OneByteReader {
        fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
            if self.offset == self.bytes.len() {
                return Ok(0);
            }
            let count = output.len().min(1).min(self.bytes.len() - self.offset);
            output[..count].copy_from_slice(&self.bytes[self.offset..self.offset + count]);
            self.offset += count;
            Ok(count)
        }
    }

    fn collect_file(path: PathBuf, request: ReadRequest) -> (Vec<u8>, ReadSummary) {
        let provider = FileProvider::new(path.clone());
        let mut scheduler =
            IoScheduler::new(provider, 1, 4).expect("scheduler config must succeed");
        let request_id = scheduler.submit(request).expect("request must be accepted");
        let mut context = context();
        let mut bytes = Vec::new();
        let summary = loop {
            match scheduler.poll(&mut context) {
                Poll::Pending => continue,
                Poll::Ready(Ok(IoEvent::Chunk {
                    request_id: id,
                    chunk,
                })) => {
                    assert_eq!(id, request_id);
                    bytes.extend_from_slice(chunk.bytes());
                }
                Poll::Ready(Ok(IoEvent::Completed(summary))) => break summary,
                Poll::Ready(Err(error)) => panic!("unexpected IO error: {error}"),
            }
        };
        fs::remove_file(path).expect("temporary file cleanup must succeed");
        (bytes, summary)
    }

    #[test]
    fn checksum_is_stable() {
        assert_eq!(checksum(b"hello"), 0xa430d84680aabd0b);
        assert_eq!(checksum(b"hello"), checksum(b"hello"));
    }

    #[test]
    fn file_read_accepts_partial_reads_until_the_chunk_is_full() {
        let mut reader = OneByteReader {
            bytes: b"abcd",
            offset: 0,
        };
        let mut output = [0; 4];
        super::std_support::read_chunk_fully(&mut reader, &mut output)
            .expect("partial reads must be accumulated");
        assert_eq!(&output, b"abcd");
    }

    #[test]
    fn request_rejects_zero_chunk_size() {
        assert_eq!(
            ReadRequest::new(0, 1).with_chunk_size(0),
            Err(ConfigError::ZeroChunkSize)
        );
    }

    #[test]
    fn file_provider_returns_deterministic_chunks_and_checksum() {
        let source = b"abcdefghijkl";
        let path = temp_file(source);
        let request = ReadRequest::new(2, 7)
            .with_chunk_size(3)
            .expect("chunk size must be accepted")
            .with_expected_checksum(checksum(b"cdefghi"));
        let (bytes, summary) = collect_file(path, request);
        assert_eq!(bytes, b"cdefghi");
        assert_eq!(summary.length(), 7);
        assert_eq!(summary.offset(), 2);
        assert_eq!(summary.checksum(), checksum(b"cdefghi"));
    }

    #[test]
    fn priority_is_descending_and_fifo_for_equal_priority() {
        let path = temp_file(b"0123456789");
        let provider = FileProvider::new(path.clone());
        let mut scheduler =
            IoScheduler::new(provider, 1, 4).expect("scheduler config must succeed");
        let first = scheduler
            .submit(ReadRequest::new(0, 1))
            .expect("first request must fit");
        let second = scheduler
            .submit(ReadRequest::new(1, 1).with_priority(4))
            .expect("second request must fit");
        let third = scheduler
            .submit(ReadRequest::new(2, 1).with_priority(4))
            .expect("third request must fit");
        let mut context = context();
        let mut completed = Vec::new();
        loop {
            match scheduler.poll(&mut context) {
                Poll::Ready(Ok(IoEvent::Chunk { .. })) | Poll::Pending => {}
                Poll::Ready(Ok(IoEvent::Completed(summary))) => {
                    completed.push(summary.request_id())
                }
                Poll::Ready(Err(error)) => panic!("unexpected IO error: {error}"),
            }
            if completed.len() == 3 {
                break;
            }
        }
        fs::remove_file(path).expect("temporary file cleanup must succeed");
        assert_eq!(completed, vec![second, third, first]);
    }

    #[test]
    fn active_reads_are_round_robin_after_admission() {
        let mut scripts = BTreeMap::new();
        scripts.insert(
            RequestId(1),
            vec![ReadChunk::new(0, vec![1]), ReadChunk::new(1, vec![2])],
        );
        scripts.insert(
            RequestId(2),
            vec![ReadChunk::new(0, vec![3]), ReadChunk::new(1, vec![4])],
        );
        let mut scheduler = IoScheduler::new(ScriptedProvider { scripts }, 2, 2)
            .expect("scheduler config must succeed");
        let first = scheduler
            .submit(ReadRequest::new(0, 2))
            .expect("first request must fit");
        let second = scheduler
            .submit(ReadRequest::new(0, 2))
            .expect("second request must fit");
        let mut context = context();
        let mut chunk_requests = Vec::new();
        let mut completed = 0;
        while completed < 2 {
            match scheduler.poll(&mut context) {
                Poll::Ready(Ok(IoEvent::Chunk { request_id, .. })) => {
                    chunk_requests.push(request_id)
                }
                Poll::Ready(Ok(IoEvent::Completed(_))) => completed += 1,
                Poll::Ready(Err(error)) => panic!("unexpected IO error: {error}"),
                Poll::Pending => {}
            }
        }
        assert_eq!(chunk_requests, vec![first, second, first, second]);
    }

    #[test]
    fn request_ids_fail_closed_without_wraparound() {
        let path = temp_file(b"a");
        let mut scheduler = IoScheduler::new(FileProvider::new(path.clone()), 1, 2)
            .expect("scheduler config must succeed");
        scheduler.set_next_request_id_for_test(u64::MAX);
        assert_eq!(
            scheduler.submit(ReadRequest::new(0, 1)),
            Ok(RequestId(u64::MAX))
        );
        assert_eq!(
            scheduler.submit(ReadRequest::new(0, 1)),
            Err(IoError::RequestIdExhausted)
        );
        fs::remove_file(path).expect("temporary file cleanup must succeed");
    }

    #[test]
    fn backpressure_rejects_when_capacity_is_full() {
        let path = temp_file(b"a");
        let mut scheduler = IoScheduler::new(FileProvider::new(path.clone()), 1, 1)
            .expect("scheduler config must succeed");
        scheduler
            .submit(ReadRequest::new(0, 1))
            .expect("first request must fit");
        assert_eq!(
            scheduler.submit(ReadRequest::new(0, 1)),
            Err(IoError::Backpressure { capacity: 1 })
        );
        fs::remove_file(path).expect("temporary file cleanup must succeed");
    }

    #[test]
    fn queued_cancellation_releases_capacity_and_reports_error() {
        let path = temp_file(b"ab");
        let mut scheduler = IoScheduler::new(FileProvider::new(path.clone()), 1, 1)
            .expect("scheduler config must succeed");
        let request_id = scheduler
            .submit(ReadRequest::new(0, 1))
            .expect("request must fit");
        assert!(scheduler.cancel(request_id));
        assert_eq!(scheduler.len(), 1);
        let mut context = context();
        assert_eq!(
            scheduler.poll(&mut context),
            Poll::Ready(Err(IoError::Cancelled { request_id }))
        );
        assert!(scheduler.is_empty());
        let replacement = scheduler.submit(ReadRequest::new(1, 1));
        assert!(replacement.is_ok());
        fs::remove_file(path).expect("temporary file cleanup must succeed");
    }

    #[test]
    fn active_cancellation_is_observable() {
        let path = temp_file(b"a");
        let mut scheduler = IoScheduler::new(FileProvider::new(path.clone()), 1, 1)
            .expect("scheduler config must succeed");
        let request_id = scheduler
            .submit(ReadRequest::new(0, 1))
            .expect("request must fit");
        let mut context = context();
        assert!(matches!(
            scheduler.poll(&mut context),
            Poll::Ready(Ok(IoEvent::Chunk { .. }))
        ));
        assert!(scheduler.cancel(request_id));
        assert_eq!(
            scheduler.poll(&mut context),
            Poll::Ready(Err(IoError::Cancelled { request_id }))
        );
        fs::remove_file(path).expect("temporary file cleanup must succeed");
    }

    #[test]
    fn short_read_is_reported() {
        let path = temp_file(b"short");
        let provider = FileProvider::new(path.clone());
        let mut scheduler =
            IoScheduler::new(provider, 1, 1).expect("scheduler config must succeed");
        scheduler
            .submit(ReadRequest::new(0, 6))
            .expect("request must fit");
        let mut context = context();
        let error = loop {
            match scheduler.poll(&mut context) {
                Poll::Ready(Err(error)) => break error,
                Poll::Ready(Ok(_)) | Poll::Pending => {}
            }
        };
        assert_eq!(
            error,
            IoError::ShortRead {
                expected: 6,
                actual: 5
            }
        );
        fs::remove_file(path).expect("temporary file cleanup must succeed");
    }

    #[test]
    fn checksum_mismatch_is_reported() {
        let path = temp_file(b"data");
        let mut scheduler = IoScheduler::new(FileProvider::new(path.clone()), 1, 1)
            .expect("scheduler config must succeed");
        scheduler
            .submit(ReadRequest::new(0, 4).with_expected_checksum(checksum(b"other")))
            .expect("request must fit");
        let mut context = context();
        let error = loop {
            match scheduler.poll(&mut context) {
                Poll::Ready(Err(error)) => break error,
                Poll::Ready(Ok(_)) | Poll::Pending => {}
            }
        };
        assert_eq!(
            error,
            IoError::ChecksumMismatch {
                expected: checksum(b"other"),
                actual: checksum(b"data")
            }
        );
        fs::remove_file(path).expect("temporary file cleanup must succeed");
    }

    struct ScriptedProvider {
        scripts: BTreeMap<RequestId, Vec<ReadChunk>>,
    }

    struct ScriptedOperation {
        request_id: RequestId,
        chunks: Vec<ReadChunk>,
        index: usize,
        cancelled: bool,
    }

    impl ChunkedRead for ScriptedOperation {
        fn poll_chunk(
            &mut self,
            _context: &mut Context<'_>,
        ) -> Poll<Result<Option<ReadChunk>, IoError>> {
            if self.cancelled {
                return Poll::Ready(Err(IoError::Cancelled {
                    request_id: self.request_id,
                }));
            }
            if let Some(chunk) = self.chunks.get(self.index).cloned() {
                self.index += 1;
                Poll::Ready(Ok(Some(chunk)))
            } else {
                Poll::Ready(Ok(None))
            }
        }

        fn cancel(&mut self) {
            self.cancelled = true;
        }
    }

    impl ReadProvider for ScriptedProvider {
        type Operation = ScriptedOperation;

        fn start(
            &self,
            _request: &ReadRequest,
            request_id: RequestId,
        ) -> Result<Self::Operation, IoError> {
            Ok(ScriptedOperation {
                request_id,
                chunks: self.scripts.get(&request_id).cloned().unwrap_or_default(),
                index: 0,
                cancelled: false,
            })
        }
    }

    #[test]
    fn scheduler_rejects_non_contiguous_provider_chunks() {
        let mut scripts = BTreeMap::new();
        scripts.insert(RequestId(1), vec![ReadChunk::new(4, vec![1])]);
        let mut scheduler = IoScheduler::new(ScriptedProvider { scripts }, 1, 1)
            .expect("scheduler config must succeed");
        scheduler
            .submit(ReadRequest::new(0, 1))
            .expect("request must fit");
        let mut context = context();
        assert_eq!(
            scheduler.poll(&mut context),
            Poll::Ready(Err(IoError::InvalidChunkOffset {
                expected: 0,
                actual: 4
            }))
        );
    }

    #[test]
    fn empty_reads_complete_with_empty_checksum() {
        let path = temp_file(b"data");
        let request = ReadRequest::new(2, 0).with_expected_checksum(checksum(b""));
        let (bytes, summary) = collect_file(path, request);
        assert!(bytes.is_empty());
        assert_eq!(summary.length(), 0);
        assert_eq!(summary.checksum(), checksum(b""));
    }
}
