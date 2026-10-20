//! Atomic publication of real source discovery through host asset identities.

use crate::{
    filesystem::{FilesystemEntry, FilesystemEntryKind, FilesystemSnapshot},
    AssetRecord, CatalogSnapshotLimits, ContentCatalog, ContentError, ContentSnapshotToken,
    VcsRecord,
};
use alloc::{collections::BTreeMap, string::String};

/// Bounds a complete asynchronous VCS metadata publication, including paths
/// for files not yet imported. Counts logical UTF-8 bytes, not allocator overhead.
#[derive(Clone, Copy, Debug)]
pub struct VcsSnapshotLimits {
    /// Maximum distinct path observations.
    pub max_paths: usize,
    /// Maximum path bytes for one observation.
    pub max_path_bytes: usize,
    /// Combined path, revision and lock-owner bytes.
    pub max_payload_bytes: usize,
}

impl Default for VcsSnapshotLimits {
    fn default() -> Self {
        Self {
            max_paths: 100_000,
            max_path_bytes: 4096,
            max_payload_bytes: 16 * 1024 * 1024,
        }
    }
}

impl ContentCatalog {
    /// Publishes complete source records only to the exact captured snapshot.
    /// A stale snapshot is rejected before the iterator is consumed.
    pub fn publish_assets(
        &mut self,
        expected: &ContentSnapshotToken,
        records: impl IntoIterator<Item = AssetRecord>,
        limits: CatalogSnapshotLimits,
    ) -> Result<bool, ContentError> {
        self.require_snapshot(expected)?;
        self.replace_assets(self.generation(), records, limits)
    }

    /// Resolves real discovered files under an exact publication fence.
    /// Stale snapshots are rejected before invoking the host resolver.
    pub fn publish_filesystem(
        &mut self,
        expected: &ContentSnapshotToken,
        snapshot: &FilesystemSnapshot,
        limits: CatalogSnapshotLimits,
        resolve: impl FnMut(&FilesystemEntry) -> Result<Option<AssetRecord>, ContentError>,
    ) -> Result<bool, ContentError> {
        self.require_snapshot(expected)?;
        self.replace_from_filesystem(self.generation(), snapshot, limits, resolve)
    }

    /// Publishes a complete bounded provider result atomically. Stale snapshots
    /// are rejected before consuming records; failures retain all live metadata.
    /// Omitted paths become Unknown. Identical observations retain the token.
    pub fn publish_vcs(
        &mut self,
        expected: &ContentSnapshotToken,
        statuses: impl IntoIterator<Item = (String, VcsRecord)>,
        limits: VcsSnapshotLimits,
    ) -> Result<bool, ContentError> {
        self.require_snapshot(expected)?;
        let mut candidate = BTreeMap::new();
        let mut bytes = 0usize;
        for (path, status) in statuses {
            if candidate.len() >= limits.max_paths {
                return Err(ContentError::SnapshotLimitExceeded("VCS path count"));
            }
            if path.len() > limits.max_path_bytes {
                return Err(ContentError::SnapshotLimitExceeded("VCS path bytes"));
            }
            for amount in [
                path.len(),
                status.revision.as_ref().map_or(0, String::len),
                status.lock_owner.as_ref().map_or(0, String::len),
            ] {
                bytes = bytes
                    .checked_add(amount)
                    .filter(|total| *total <= limits.max_payload_bytes)
                    .ok_or(ContentError::SnapshotLimitExceeded("VCS payload bytes"))?;
            }
            let path = crate::canonical_path(path)?;
            if candidate.insert(path.clone(), status).is_some() {
                return Err(ContentError::DuplicatePath(path));
            }
        }
        if candidate == self.vcs {
            return Ok(false);
        }
        self.vcs = candidate;
        self.changed();
        Ok(true)
    }

    fn require_snapshot(&self, expected: &ContentSnapshotToken) -> Result<(), ContentError> {
        if self.matches_snapshot(expected) {
            Ok(())
        } else {
            Err(ContentError::StaleCatalogSnapshot)
        }
    }

    /// Replaces all assets from one complete filesystem scan. The host resolves
    /// stable asset identities and authored metadata (for example from its asset
    /// manifest or sidecars); paths are never hashed into identity here.
    ///
    /// Only files are offered to `resolve`. Return `None` for a source file that
    /// is not an asset. A returned record must retain the exact discovered path.
    /// Resolution, bounds or validation failures publish nothing. Omitted assets
    /// are removed; references to them remain available to diagnostics. Capture
    /// `generation()` before starting the scan to reject stale completions.
    /// The scan is a complete enumeration, not a transactional filesystem view.
    pub fn replace_from_filesystem(
        &mut self,
        expected_generation: u64,
        snapshot: &FilesystemSnapshot,
        limits: CatalogSnapshotLimits,
        mut resolve: impl FnMut(&FilesystemEntry) -> Result<Option<AssetRecord>, ContentError>,
    ) -> Result<bool, ContentError> {
        let records = snapshot.entries.iter().filter_map(|entry| {
            if entry.kind != FilesystemEntryKind::File {
                return None;
            }
            match resolve(entry) {
                Ok(Some(record)) if record.path != entry.path => {
                    Some(Err(ContentError::InvalidPath(record.path)))
                }
                Ok(record) => record.map(Ok),
                Err(error) => Some(Err(error)),
            }
        });
        self.replace_asset_results(expected_generation, records, limits)
    }
}
