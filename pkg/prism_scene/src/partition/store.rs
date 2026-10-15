use std::{
    fs::{self, File, OpenOptions},
    io::{self, Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use prism_ecs::partition::cell::CellCoord;

/// Upper bound for a cell envelope, including its reflected archive.
pub const MAX_CELL_BYTES: usize = 64 * 1024 * 1024;

/// Scene-owned persistence boundary. A successful write must durably replace
/// the complete cell, never expose a partial write. Failure may leave the old
/// or new complete file, but must never authorize eviction. Implementations
/// must enforce [`MAX_CELL_BYTES`] on reads as the filesystem store does.
pub trait CellStore {
    /// Read an existing cell. Missing files are deliberately not empty cells.
    fn read(&self, cell: CellCoord) -> io::Result<Vec<u8>>;
    /// Commit bytes before the scene owner removes any live entity.
    fn write_atomic(&mut self, cell: CellCoord, bytes: &[u8]) -> io::Result<()>;
}

/// Filesystem implementation: a unique same-directory temporary file,
/// write + fsync, atomic rename, then directory fsync. No I/O enters prism_ecs.
#[derive(Debug)]
pub struct FileCellStore {
    root: PathBuf,
}

impl FileCellStore {
    /// Select a directory. It is only created when a cell is saved.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// The directory containing cell files.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Stable per-cell path; coordinates cannot inject path separators.
    pub fn cell_path(&self, cell: CellCoord) -> PathBuf {
        self.root
            .join(format!("{}_{}_{}.prismcell", cell.x, cell.y, cell.z))
    }
}

impl CellStore for FileCellStore {
    fn read(&self, cell: CellCoord) -> io::Result<Vec<u8>> {
        let file = File::open(self.cell_path(cell))?;
        if file.metadata()?.len() > MAX_CELL_BYTES as u64 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "cell too large"));
        }
        let mut bytes = Vec::new();
        file.take(MAX_CELL_BYTES as u64 + 1)
            .read_to_end(&mut bytes)?;
        if bytes.len() > MAX_CELL_BYTES {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "cell too large"));
        }
        Ok(bytes)
    }

    fn write_atomic(&mut self, cell: CellCoord, bytes: &[u8]) -> io::Result<()> {
        if bytes.len() > MAX_CELL_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "cell too large",
            ));
        }
        fs::create_dir_all(&self.root)?;
        // Open before rename, so unsupported directory handles fail while the
        // previous archive still exists. A post-rename fsync failure still
        // returns Err: the owner keeps its entities and retries the save.
        let directory = File::open(&self.root)?;
        static SEQUENCE: AtomicU64 = AtomicU64::new(0);
        let (temporary, mut file) = loop {
            let sequence = SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let temporary = self
                .root
                .join(format!(".prismcell-{}-{sequence}.tmp", std::process::id()));
            match OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temporary)
            {
                Ok(file) => break (temporary, file),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error),
            }
        };
        let result = (|| {
            file.write_all(bytes)?;
            file.sync_all()?;
            drop(file);
            fs::rename(&temporary, self.cell_path(cell))?;
            directory.sync_all()
        })();
        if result.is_err() {
            // Only our create_new temporary is removed; never the prior cell.
            let _ = fs::remove_file(&temporary);
        }
        result
    }
}
