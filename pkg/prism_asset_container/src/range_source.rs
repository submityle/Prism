use core::fmt;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

#[cfg(unix)]
use std::os::unix::fs::FileExt;
#[cfg(windows)]
use std::os::windows::fs::FileExt;

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
