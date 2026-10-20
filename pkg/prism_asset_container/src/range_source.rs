use alloc::sync::Arc;
use core::fmt;
use std::fs::File;
use std::path::Path;

#[cfg(feature = "mmap")]
use memmap2::{Mmap, MmapOptions};

#[cfg(unix)]
use std::os::unix::fs::FileExt;
#[cfg(windows)]
use std::os::windows::fs::FileExt;

use crate::manifest::ContainerIdentity;

/// Errors returned by a [`RangeSource`] while serving a byte range.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RangeSourceError {
    /// The requested range is outside the source's declared length.
    OutOfBounds,
    /// The source ended before it could fill the requested range.
    ShortRead {
        /// Number of bytes requested by the caller.
        expected: usize,
        /// Number of bytes returned before the source ended.
        actual: usize,
    },
    /// The source could not complete the read.
    Io(String),
    /// The remote source was configured without any usable request capacity.
    InvalidRequestLimit {
        /// Configured maximum bytes per request.
        maximum: usize,
    },
    /// The requested remote range exceeds the source's configured request cap.
    RequestTooLarge {
        /// Number of bytes requested by the caller.
        requested: usize,
        /// Maximum bytes allowed in one remote request.
        maximum: usize,
    },
    /// A remote response was not bound to the immutable identity requested.
    RemoteIdentityMismatch,
    /// A remote response described a different byte offset than requested.
    RemoteOffsetMismatch {
        /// Requested byte offset.
        expected: u64,
        /// Response byte offset.
        actual: u64,
    },
    /// A remote response returned a different byte count than requested.
    RemoteLengthMismatch {
        /// Requested byte count.
        expected: usize,
        /// Response byte count.
        actual: usize,
    },
    /// The remote fetcher rejected or failed the request.
    RemoteFetch(RemoteRangeFetchError),
}

impl fmt::Display for RangeSourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfBounds => f.write_str("range source request is out of bounds"),
            Self::ShortRead { expected, actual } => write!(
                f,
                "range source short read: expected {expected} bytes, got {actual}"
            ),
            Self::Io(error) => write!(f, "range source I/O failed: {error}"),
            Self::InvalidRequestLimit { maximum } => write!(
                f,
                "remote range request limit must be non-zero, got {maximum}"
            ),
            Self::RequestTooLarge { requested, maximum } => write!(
                f,
                "remote range request is too large: {requested} bytes exceeds {maximum}"
            ),
            Self::RemoteIdentityMismatch => {
                f.write_str("remote range response is bound to a different immutable identity")
            }
            Self::RemoteOffsetMismatch { expected, actual } => write!(
                f,
                "remote range response offset mismatch: expected {expected}, got {actual}"
            ),
            Self::RemoteLengthMismatch { expected, actual } => write!(
                f,
                "remote range response length mismatch: expected {expected}, got {actual}"
            ),
            Self::RemoteFetch(error) => write!(f, "remote range fetch failed: {error}"),
        }
    }
}

impl std::error::Error for RangeSourceError {}

/// A seekable byte source used by a runtime container.
///
/// Implementations must serve exactly `out.len()` bytes starting at `offset`
/// and must return stable contents for the lifetime of every mounted reader.
/// The container reader validates the complete source at mount time, then
/// fetches payload bytes on demand; full-payload reads recheck the entry hash,
/// but partial reads rely on this immutability contract.
pub trait RangeSource: Send + Sync + fmt::Debug {
    /// Returns the immutable source length in bytes.
    fn len(&self) -> u64;

    /// Returns whether the immutable source contains no bytes.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Reads one source range into `out`.
    fn read_range(&self, offset: u64, out: &mut [u8]) -> Result<(), RangeSourceError>;
}

/// One immutable object identity used to bind a remote range request.
///
/// This is intentionally the same identity persisted by [`crate::RuntimeManifest`].
/// A remote adapter must use it as an object-version precondition (for example,
/// an immutable content-addressed URL or an `If-Match`/generation condition),
/// not merely as diagnostic metadata.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RemoteRangeRequest {
    /// Exact byte offset in the immutable object.
    pub offset: u64,
    /// Exact number of bytes requested.
    pub len: usize,
    /// Immutable object identity required by the caller.
    pub identity: ContainerIdentity,
}

impl RemoteRangeRequest {
    /// Creates a request for one exact object range.
    #[must_use]
    pub const fn new(offset: u64, len: usize, identity: ContainerIdentity) -> Self {
        Self {
            offset,
            len,
            identity,
        }
    }
}

/// Successful response from a remote range adapter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RemoteRangeResponse {
    /// Byte offset represented by `bytes`.
    pub offset: u64,
    /// Immutable identity served by the remote endpoint.
    pub identity: ContainerIdentity,
    /// Exactly the bytes requested by the caller.
    pub bytes: Vec<u8>,
}

impl RemoteRangeResponse {
    /// Creates a response for one exact object range.
    #[must_use]
    pub fn new(offset: u64, identity: ContainerIdentity, bytes: impl Into<Vec<u8>>) -> Self {
        Self {
            offset,
            identity,
            bytes: bytes.into(),
        }
    }
}

/// Failure reported by a remote range adapter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RemoteRangeFetchError {
    /// The transport could not complete the request.
    Transport(String),
    /// The remote endpoint rejected the immutable range precondition.
    Rejected(String),
}

impl fmt::Display for RemoteRangeFetchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Transport(error) => write!(f, "transport error: {error}"),
            Self::Rejected(error) => write!(f, "request rejected: {error}"),
        }
    }
}

impl std::error::Error for RemoteRangeFetchError {}

/// Synchronous adapter boundary for HTTP, object storage, or a platform range API.
///
/// The adapter owns transport, authentication, retry, and connection pooling.
/// It must reject a request when the requested immutable identity is not
/// available and must return an exact response for the requested range. The
/// [`RemoteRangeSource`] independently rechecks all response fields before
/// exposing bytes to the container reader.
pub trait RemoteRangeFetcher: Send + Sync + fmt::Debug {
    /// Fetches one exact byte range from one immutable object version.
    fn fetch_range(
        &self,
        request: RemoteRangeRequest,
    ) -> Result<RemoteRangeResponse, RemoteRangeFetchError>;
}

/// Range source backed by an immutable remote object adapter.
///
/// The source never accepts a response from another object generation, a
/// different offset, or a short/overlong body. Bounds and the per-request cap
/// are checked before calling the adapter. Container mounting additionally
/// validates the full container identity and directory/payload hashes, so the
/// adapter is a delivery mechanism rather than a trust root.
#[derive(Clone)]
pub struct RemoteRangeSource {
    fetcher: Arc<dyn RemoteRangeFetcher>,
    identity: ContainerIdentity,
    maximum_request_bytes: usize,
}

impl fmt::Debug for RemoteRangeSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RemoteRangeSource")
            .field("identity", &self.identity)
            .field("maximum_request_bytes", &self.maximum_request_bytes)
            .finish_non_exhaustive()
    }
}

impl RemoteRangeSource {
    /// Creates a remote source with a strict maximum request size.
    pub fn new(
        fetcher: Arc<dyn RemoteRangeFetcher>,
        identity: ContainerIdentity,
        maximum_request_bytes: usize,
    ) -> Result<Self, RangeSourceError> {
        if maximum_request_bytes == 0 {
            return Err(RangeSourceError::InvalidRequestLimit {
                maximum: maximum_request_bytes,
            });
        }
        Ok(Self {
            fetcher,
            identity,
            maximum_request_bytes,
        })
    }

    /// Returns the immutable identity required for every remote request.
    #[must_use]
    pub const fn identity(&self) -> ContainerIdentity {
        self.identity
    }

    /// Returns the maximum number of bytes sent in one remote request.
    #[must_use]
    pub const fn maximum_request_bytes(&self) -> usize {
        self.maximum_request_bytes
    }
}

impl RangeSource for RemoteRangeSource {
    fn len(&self) -> u64 {
        self.identity.byte_len
    }

    fn read_range(&self, offset: u64, out: &mut [u8]) -> Result<(), RangeSourceError> {
        if out.len() > self.maximum_request_bytes {
            return Err(RangeSourceError::RequestTooLarge {
                requested: out.len(),
                maximum: self.maximum_request_bytes,
            });
        }
        let len = u64::try_from(out.len()).map_err(|_| RangeSourceError::OutOfBounds)?;
        let end = offset
            .checked_add(len)
            .ok_or(RangeSourceError::OutOfBounds)?;
        if end > self.identity.byte_len {
            return Err(RangeSourceError::OutOfBounds);
        }
        if out.is_empty() {
            return Ok(());
        }

        let request = RemoteRangeRequest::new(offset, out.len(), self.identity);
        let response = self
            .fetcher
            .fetch_range(request)
            .map_err(RangeSourceError::RemoteFetch)?;
        if response.identity != self.identity {
            return Err(RangeSourceError::RemoteIdentityMismatch);
        }
        if response.offset != offset {
            return Err(RangeSourceError::RemoteOffsetMismatch {
                expected: offset,
                actual: response.offset,
            });
        }
        if response.bytes.len() != out.len() {
            return Err(RangeSourceError::RemoteLengthMismatch {
                expected: out.len(),
                actual: response.bytes.len(),
            });
        }
        out.copy_from_slice(&response.bytes);
        Ok(())
    }
}

/// An immutable in-memory range source.
#[derive(Clone, Debug)]
pub struct MemoryRangeSource {
    bytes: Arc<[u8]>,
}

impl MemoryRangeSource {
    /// Creates a source backed by the supplied immutable bytes.
    #[must_use]
    pub fn new(bytes: impl Into<Arc<[u8]>>) -> Self {
        Self {
            bytes: bytes.into(),
        }
    }
}

impl RangeSource for MemoryRangeSource {
    fn len(&self) -> u64 {
        self.bytes.len() as u64
    }

    fn read_range(&self, offset: u64, out: &mut [u8]) -> Result<(), RangeSourceError> {
        let start = usize::try_from(offset).map_err(|_| RangeSourceError::OutOfBounds)?;
        let end = start
            .checked_add(out.len())
            .ok_or(RangeSourceError::OutOfBounds)?;
        let bytes = self
            .bytes
            .get(start..end)
            .ok_or(RangeSourceError::OutOfBounds)?;
        out.copy_from_slice(bytes);
        Ok(())
    }
}

/// A local file backed range source.
///
/// Each request uses positional I/O and reads only the requested bytes. The
/// source never maps or loads the complete container, and independent range
/// requests do not share a mutable file cursor.
#[derive(Clone)]
pub struct FileRangeSource {
    file: Arc<File>,
    len: u64,
}

impl fmt::Debug for FileRangeSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FileRangeSource")
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

impl FileRangeSource {
    /// Opens a local file and snapshots its length for range validation.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, RangeSourceError> {
        let file = File::open(path).map_err(|error| RangeSourceError::Io(error.to_string()))?;
        Self::from_file(file)
    }

    /// Creates a source from an already opened file.
    pub fn from_file(file: File) -> Result<Self, RangeSourceError> {
        let len = file
            .metadata()
            .map_err(|error| RangeSourceError::Io(error.to_string()))?
            .len();
        Ok(Self {
            file: Arc::new(file),
            len,
        })
    }
}

impl RangeSource for FileRangeSource {
    fn len(&self) -> u64 {
        self.len
    }

    fn read_range(&self, offset: u64, out: &mut [u8]) -> Result<(), RangeSourceError> {
        let output_len = u64::try_from(out.len()).map_err(|_| RangeSourceError::OutOfBounds)?;
        let end = offset
            .checked_add(output_len)
            .ok_or(RangeSourceError::OutOfBounds)?;
        if end > self.len {
            return Err(RangeSourceError::OutOfBounds);
        }

        let mut filled = 0;
        while filled < out.len() {
            let read_offset = offset
                .checked_add(u64::try_from(filled).map_err(|_| RangeSourceError::OutOfBounds)?)
                .ok_or(RangeSourceError::OutOfBounds)?;
            match read_file_at(&self.file, &mut out[filled..], read_offset) {
                Ok(0) => {
                    return Err(RangeSourceError::ShortRead {
                        expected: out.len(),
                        actual: filled,
                    });
                }
                Ok(read) => filled += read,
                Err(error) => return Err(error),
            }
        }
        Ok(())
    }
}

/// An immutable memory-mapped local file range source.
///
/// The mapping is read-only and shared by clones. The file must remain
/// immutable for the lifetime of every mounted reader, just like
/// [`FileRangeSource`]. Mapping a file avoids the per-request `read_at` copy
/// from the kernel page cache; callers still receive their requested bytes in
/// the output buffer and the container reader retains its strict validation
/// and hash checks.
#[cfg(feature = "mmap")]
#[derive(Clone)]
pub struct MmapRangeSource {
    map: Arc<Mmap>,
    len: u64,
}

#[cfg(feature = "mmap")]
impl fmt::Debug for MmapRangeSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MmapRangeSource")
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "mmap")]
impl MmapRangeSource {
    /// Opens and read-only maps a local container file.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, RangeSourceError> {
        let file = File::open(path).map_err(|error| RangeSourceError::Io(error.to_string()))?;
        Self::from_file(file)
    }

    /// Creates a read-only mapping from an already opened file.
    pub fn from_file(file: File) -> Result<Self, RangeSourceError> {
        let len = file
            .metadata()
            .map_err(|error| RangeSourceError::Io(error.to_string()))?
            .len();
        let map_len = usize::try_from(len).map_err(|_| RangeSourceError::OutOfBounds)?;
        if map_len == 0 {
            return Err(RangeSourceError::Io(
                "cannot memory-map an empty file".into(),
            ));
        }
        // `memmap2` requires the caller to uphold that the file is not
        // concurrently mutated. The public source contract already requires
        // immutable bytes for every mounted reader, and the mapping is
        // read-only, so this is the single audited unsafe boundary.
        #[expect(
            unsafe_code,
            reason = "read-only mmap is the single audited OS mapping boundary"
        )]
        // SAFETY: the public source contract requires the file to remain
        // immutable for the lifetime of the mapping, and the mapping is
        // read-only.
        let map = unsafe { MmapOptions::new().len(map_len).map(&file) }
            .map_err(|error| RangeSourceError::Io(error.to_string()))?;
        Ok(Self {
            map: Arc::new(map),
            len,
        })
    }
}

#[cfg(feature = "mmap")]
impl RangeSource for MmapRangeSource {
    fn len(&self) -> u64 {
        self.len
    }

    fn read_range(&self, offset: u64, out: &mut [u8]) -> Result<(), RangeSourceError> {
        let output_len = u64::try_from(out.len()).map_err(|_| RangeSourceError::OutOfBounds)?;
        let end = offset
            .checked_add(output_len)
            .ok_or(RangeSourceError::OutOfBounds)?;
        if end > self.len {
            return Err(RangeSourceError::OutOfBounds);
        }
        let start = usize::try_from(offset).map_err(|_| RangeSourceError::OutOfBounds)?;
        let end = start
            .checked_add(out.len())
            .ok_or(RangeSourceError::OutOfBounds)?;
        out.copy_from_slice(
            self.map
                .get(start..end)
                .ok_or(RangeSourceError::OutOfBounds)?,
        );
        Ok(())
    }
}

fn read_file_at(file: &File, out: &mut [u8], offset: u64) -> Result<usize, RangeSourceError> {
    #[cfg(unix)]
    {
        loop {
            match file.read_at(out, offset) {
                Ok(read) => return Ok(read),
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(RangeSourceError::Io(error.to_string())),
            }
        }
    }
    #[cfg(windows)]
    {
        loop {
            match file.seek_read(out, offset) {
                Ok(read) => return Ok(read),
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(RangeSourceError::Io(error.to_string())),
            }
        }
    }
    #[cfg(not(any(unix, windows)))]
    {
        let mut file = file
            .try_clone()
            .map_err(|error| RangeSourceError::Io(error.to_string()))?;
        use std::io::{Read, Seek, SeekFrom};
        file.seek(SeekFrom::Start(offset))
            .map_err(|error| RangeSourceError::Io(error.to_string()))?;
        loop {
            match file.read(out) {
                Ok(read) => return Ok(read),
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(RangeSourceError::Io(error.to_string())),
            }
        }
    }
}
