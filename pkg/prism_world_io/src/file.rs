use std::{
    fmt,
    fs::File,
    io::{self, Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

use crate::{CancellationToken, ChunkProvider, ChunkRange};

/// Errors produced by [`FileProvider`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FileProviderError {
    operation: &'static str,
    kind: io::ErrorKind,
}

impl FileProviderError {
    fn from_io(operation: &'static str, error: io::Error) -> Self {
        Self {
            operation,
            kind: error.kind(),
        }
    }

    /// Returns the operation that failed.
    #[must_use]
    pub const fn operation(self) -> &'static str {
        self.operation
    }

    /// Returns the platform-independent I/O error kind.
    #[must_use]
    pub const fn kind(self) -> io::ErrorKind {
        self.kind
    }
}

impl fmt::Display for FileProviderError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "file {} failed: {}", self.operation, self.kind)
    }
}

impl std::error::Error for FileProviderError {}

/// A filesystem-backed synchronous provider for the standard I/O runtime.
#[derive(Clone, Debug)]
pub struct FileProvider {
    path: PathBuf,
}

impl FileProvider {
    /// Creates a provider that opens the path independently for each request.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    /// Returns the configured path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl ChunkProvider for FileProvider {
    type Error = FileProviderError;

    fn read_chunk(
        &self,
        range: ChunkRange,
        cancel: &CancellationToken,
    ) -> Result<Vec<u8>, Self::Error> {
        if cancel.is_cancelled() {
            return Ok(Vec::new());
        }
        let mut file =
            File::open(&self.path).map_err(|error| FileProviderError::from_io("open", error))?;
        file.seek(SeekFrom::Start(range.offset()))
            .map_err(|error| FileProviderError::from_io("seek", error))?;

        let mut bytes = Vec::with_capacity(range.length());
        let mut buffer = [0_u8; 16 * 1024];
        while bytes.len() < range.length() {
            if cancel.is_cancelled() {
                break;
            }
            let remaining = range.length() - bytes.len();
            let read_length = remaining.min(buffer.len());
            let read = file
                .read(&mut buffer[..read_length])
                .map_err(|error| FileProviderError::from_io("read", error))?;
            if read == 0 {
                break;
            }
            bytes.extend_from_slice(&buffer[..read]);
        }
        Ok(bytes)
    }
}
