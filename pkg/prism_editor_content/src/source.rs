//! Atomic publication of real source discovery through host asset identities.

use crate::{
    filesystem::{FilesystemEntry, FilesystemEntryKind, FilesystemSnapshot},
    AssetRecord, CatalogSnapshotLimits, ContentCatalog, ContentError,
};

impl ContentCatalog {
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
