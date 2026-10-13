use core::fmt;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;

/// Errors returned by a [`RangeSource`] while serving a byte range.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RangeSourceError {
    /// The requested range is outside the source's declared length.
    OutOfBounds,
    /// The source could not complete the read.
    Io(String),
}

impl fmt::Display for RangeSourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfBounds => f.write_str("range source request is out of bounds"),
            Self::Io(error) => write!(f, "range source I/O failed: {error}"),
        }
    }
}

impl std::error::Error for RangeSourceError {}

/// A seekable byte source used by a runtime container.
///
/// Implementations must serve exactly `out.len()` bytes starting at `offset`.
/// The container reader uses this contract to validate only its header and
/// directory at mount time, then fetches payload bytes on demand.
pub trait RangeSource: Send + Sync + fmt::Debug {
    /// Returns the immutable source length in bytes.
    fn len(&self) -> u64;

    /// Reads one source range into `out`.
    fn read_range(&self, offset: u64, out: &mut [u8]) -> Result<(), RangeSourceError>;
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
/// Each request seeks a cloned file descriptor and reads only the requested
/// bytes. The source never maps or loads the complete container.
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
        let end = offset
            .checked_add(out.len() as u64)
            .ok_or(RangeSourceError::OutOfBounds)?;
        if end > self.len {
            return Err(RangeSourceError::OutOfBounds);
        }

        let mut file = self
            .file
            .try_clone()
            .map_err(|error| RangeSourceError::Io(error.to_string()))?;
        file.seek(SeekFrom::Start(offset))
            .map_err(|error| RangeSourceError::Io(error.to_string()))?;
        file.read_exact(out)
            .map_err(|error| RangeSourceError::Io(error.to_string()))
    }
}
