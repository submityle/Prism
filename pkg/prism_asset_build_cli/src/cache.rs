//! Cache queries never initialize or recover a store. Explicit pruning uses
//! the core's capacity eviction and revision-aware, leased CAS sweep.

use crate::error::CommandError;
use prism_asset_build_core::{
    BuildRevision, CasError, ContentAddressedStore, ContentHash, DdcCapacity, DerivedDataCache,
    RevisionStore,
};
use serde::Serialize;
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::Path;
use std::time::Duration;

pub const DEFAULT_MAX_INDEX_BYTES: u64 = 64 * 1024 * 1024;
const MAX_BLOBS: usize = 1_000_000;

#[derive(Debug, Serialize)]
pub struct CacheStatus {
    pub ddc_index_present: bool,
    pub ddc_entries: usize,
    pub ddc_encoded_bytes: u64,
    pub cas_blobs: usize,
    pub cas_bytes: u64,
}

#[derive(Debug, Serialize)]
pub struct CacheVerification {
    pub status: CacheStatus,
    pub current_revision: Option<u64>,
    pub checked_blobs: usize,
    pub checked_bytes: u64,
    pub checked_ddc_descriptors: usize,
    pub checked_current_descriptors: usize,
}

#[derive(Debug, Serialize)]
pub struct CachePrune {
    pub max_entries: Option<usize>,
    pub max_encoded_bytes: Option<usize>,
    pub grace_seconds: u64,
    pub evicted_keys: Vec<String>,
    /// Cache descriptors removed because their CAS blob was absent after a
    /// completed sweep. These entries are rebuildable, not GC roots.
    pub removed_missing_keys: Vec<String>,
    pub ddc_entries: usize,
    pub ddc_encoded_bytes: usize,
    pub cas_scanned: usize,
    pub cas_deleted: usize,
    pub cas_retained: usize,
    pub deferred_due_to_active_build: bool,
    pub deferred_due_to_pending_transaction: bool,
}

fn invalid(message: impl Into<String>) -> CommandError {
    CommandError::InvalidInput(message.into())
}

fn require_directory(path: &Path) -> Result<(), CommandError> {
    if !fs::symlink_metadata(path)?.is_dir() {
        return Err(invalid(format!(
            "cache directory is not a regular directory: {}",
            path.display()
        )));
    }
    Ok(())
}

fn require_file(path: &Path) -> Result<(), CommandError> {
    if !fs::symlink_metadata(path)?.is_file() {
        return Err(invalid(format!(
            "cache path is not a regular file: {}",
            path.display()
        )));
    }
    Ok(())
}

fn hex_byte(bytes: &[u8]) -> Option<u8> {
    let digit = |byte| match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    };
    Some(digit(*bytes.first()?)? * 16 + digit(*bytes.get(1)?)?)
}

// Follow the CAS two-level layout only. Temporary and unrelated paths are
// outside the inventory. Hash-shaped non-files and noncanonical paths fail
// explicitly, so verify cannot silently skip a malformed blob.
fn inventory(root: &Path) -> Result<BTreeMap<ContentHash, u64>, CommandError> {
    require_directory(root)?;
    let mut blobs = BTreeMap::new();
    for shard in fs::read_dir(root)? {
        let shard = shard?;
        let name = shard.file_name();
        let Some(name) = name.to_str() else { continue };
        if name.len() != 2 || hex_byte(name.as_bytes()).is_none() {
            continue;
        }
        require_directory(&shard.path())?;
        for entry in fs::read_dir(shard.path())? {
            let entry = entry?;
            let tail = entry.file_name();
            let Some(tail) = tail.to_str() else { continue };
            if tail.len() != 62 {
                continue;
            }
            let text = format!("{name}{tail}");
            let mut digest = [0; 32];
            let mut valid = true;
            for (index, byte) in digest.iter_mut().enumerate() {
                match hex_byte(&text.as_bytes()[index * 2..index * 2 + 2]) {
                    Some(value) => *byte = value,
                    None => {
                        valid = false;
                        break;
                    }
                }
            }
            if !valid {
                continue;
            }
            let hash = ContentHash(digest);
            if text != hash.to_hex() {
                return Err(invalid(
                    "CAS blob path is not canonical lowercase hexadecimal",
                ));
            }
            require_file(&entry.path())?;
            if blobs.len() == MAX_BLOBS {
                return Err(invalid("cache inventory exceeds one million blobs"));
            }
            blobs.insert(hash, fs::symlink_metadata(entry.path())?.len());
        }
    }
    Ok(blobs)
}

fn total_bytes(mut lengths: impl Iterator<Item = u64>) -> Result<u64, CommandError> {
    lengths.try_fold(0u64, |sum, length| {
        sum.checked_add(length)
            .ok_or_else(|| invalid("cache byte total overflows u64"))
    })
}

fn snapshot(
    store: &Path,
    max_index_bytes: u64,
) -> Result<
    (
        prism_asset_build_core::ddc::DdcSnapshot,
        BTreeMap<ContentHash, u64>,
    ),
    CommandError,
> {
    require_directory(store)?;
    let ddc_root = store.join("ddc");
    match fs::symlink_metadata(&ddc_root) {
        Ok(_) => require_directory(&ddc_root)?,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    let ddc = DerivedDataCache::read_snapshot(ddc_root, max_index_bytes)?;
    let blobs = inventory(&store.join("cas"))?;
    Ok((ddc, blobs))
}

fn status_from(
    ddc: &prism_asset_build_core::ddc::DdcSnapshot,
    blobs: &BTreeMap<ContentHash, u64>,
) -> Result<CacheStatus, CommandError> {
    Ok(CacheStatus {
        ddc_index_present: ddc.index_present,
        ddc_entries: ddc.entries.len(),
        ddc_encoded_bytes: ddc.encoded_bytes,
        cas_blobs: blobs.len(),
        cas_bytes: total_bytes(blobs.values().copied())?,
    })
}

pub fn status(store: &Path, max_index_bytes: u64) -> Result<CacheStatus, CommandError> {
    let (ddc, blobs) = snapshot(store, max_index_bytes)?;
    status_from(&ddc, &blobs)
}

fn current_revision(store: &Path) -> Result<Option<BuildRevision>, CommandError> {
    let path = store.join("CURRENT");
    match fs::symlink_metadata(&path) {
        Ok(_) => require_file(&path)?,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    }
    let mut bytes = Vec::new();
    File::open(path)?.take(9).read_to_end(&mut bytes)?;
    let bytes: [u8; 8] = bytes
        .try_into()
        .map_err(|_| invalid("CURRENT pointer has invalid length"))?;
    Ok(Some(BuildRevision(u64::from_le_bytes(bytes))))
}

fn is_missing(error: &CasError) -> bool {
    matches!(error, CasError::Io(error) if error.kind() == io::ErrorKind::NotFound)
}

pub fn verify(store: &Path, max_index_bytes: u64) -> Result<CacheVerification, CommandError> {
    let (ddc, blobs) = snapshot(store, max_index_bytes)?;
    let status = status_from(&ddc, &blobs)?;
    let revision = current_revision(store)?;
    let mut expected = BTreeMap::<ContentHash, u64>::new();
    let mut record = |hash: ContentHash, length: u64| -> Result<(), CommandError> {
        if expected
            .insert(hash, length)
            .is_some_and(|previous| previous != length)
        {
            return Err(invalid(format!(
                "cache descriptors disagree on byte length for {}",
                hash.to_hex()
            )));
        }
        Ok(())
    };
    for descriptor in ddc.entries.values() {
        record(descriptor.content_hash, descriptor.byte_len)?;
    }
    let mut checked_current_descriptors = 0;
    if let Some(revision) = revision {
        let (manifest, _) = RevisionStore::read_committed_manifest(store, revision)?;
        checked_current_descriptors = manifest.entries.len();
        for descriptor in manifest.entries.values() {
            record(descriptor.content_hash, descriptor.byte_len)?;
        }
    }
    let mut hashes: std::collections::BTreeSet<_> = blobs.keys().copied().collect();
    hashes.extend(expected.keys().copied());
    let mut checked_bytes = 0u64;
    for &hash in &hashes {
        let length = ContentAddressedStore::verify_at(store.join("cas"), hash)?;
        if let Some(&declared) = expected.get(&hash) {
            if length != declared {
                return Err(CasError::Io(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "CAS blob {} length mismatch: expected {declared}, got {length}",
                        hash.to_hex()
                    ),
                ))
                .into());
            }
        }
        checked_bytes = checked_bytes
            .checked_add(length)
            .ok_or_else(|| invalid("verified cache bytes overflow u64"))?;
    }
    Ok(CacheVerification {
        status,
        current_revision: revision.map(|revision| revision.0),
        checked_blobs: hashes.len(),
        checked_bytes,
        checked_ddc_descriptors: ddc.entries.len(),
        checked_current_descriptors,
    })
}

/// Capacity applies to DDC metadata only. CAS GC removes uncommitted payloads
/// by reachability and grace, not by a disk quota; DDC entries are not GC roots.
/// Prune is an explicit mutation of an existing store and may recover it.
pub fn prune(
    store: &Path,
    capacity: DdcCapacity,
    grace: Duration,
    max_index_bytes: u64,
) -> Result<CachePrune, CommandError> {
    let capacity = DdcCapacity::new(capacity.max_entries, capacity.max_encoded_bytes)?;
    snapshot(store, max_index_bytes)?;
    require_directory(&store.join("revisions"))?;
    require_directory(&store.join("transactions"))?;
    require_file(&store.join("transactions/journal.bin"))?;
    let revisions = RevisionStore::open(store)?;
    // Always reload under the core eviction lock: an index may have been
    // created since the read-only preflight. Only explicit prune may create
    // the DDC directory/lock when an existing store has no index yet.
    let mut ddc = DerivedDataCache::open(store.join("ddc"))?;
    let eviction = ddc.evict_to_capacity(capacity)?;
    let mut ddc_entries = ddc.len();
    let mut ddc_encoded_bytes = eviction.encoded_bytes;
    let evicted_keys = eviction
        .evicted_keys
        .into_iter()
        .map(|key| key.to_hex())
        .collect();
    let report = revisions.sweep_unreachable_cas(grace)?;
    // DDC descriptors are not GC roots. A completed sweep can therefore make
    // an old cache entry unserviceable. Reload after the sweep and remove only
    // entries whose blob is now absent. Corruption and length mismatches stay
    // observable through `cache verify`; prune never repairs either. Entries
    // concurrently added after this snapshot were not present when this sweep
    // selected deletion candidates, and cooperating builders hold the core
    // build-output lease which makes the sweep defer.
    let mut removed_missing_keys = Vec::new();
    if !report.deferred_due_to_active_build && !report.deferred_due_to_pending_transaction {
        let after_sweep = DerivedDataCache::read_snapshot(store.join("ddc"), max_index_bytes)?;
        for (key, descriptor) in after_sweep.entries {
            match ContentAddressedStore::verify_at(store.join("cas"), descriptor.content_hash) {
                Ok(length) if length == descriptor.byte_len => {}
                Ok(_) => {
                    return Err(invalid(format!(
                        "DDC descriptor {} has a CAS length mismatch",
                        key.to_hex()
                    )))
                }
                Err(error) if is_missing(&error) => {
                    ddc.invalidate(key)?;
                    removed_missing_keys.push(key.to_hex());
                }
                Err(error) => return Err(error.into()),
            }
        }
        let final_snapshot = DerivedDataCache::read_snapshot(store.join("ddc"), max_index_bytes)?;
        ddc_entries = final_snapshot.entries.len();
        ddc_encoded_bytes = usize::try_from(final_snapshot.encoded_bytes)
            .map_err(|_| invalid("DDC encoded byte count exceeds usize"))?;
    }
    Ok(CachePrune {
        max_entries: capacity.max_entries,
        max_encoded_bytes: capacity.max_encoded_bytes,
        grace_seconds: grace.as_secs(),
        evicted_keys,
        removed_missing_keys,
        ddc_entries,
        ddc_encoded_bytes,
        cas_scanned: report.scanned,
        cas_deleted: report.deleted,
        cas_retained: report.retained,
        deferred_due_to_active_build: report.deferred_due_to_active_build,
        deferred_due_to_pending_transaction: report.deferred_due_to_pending_transaction,
    })
}

impl CacheStatus {
    pub fn write_human(&self, out: &mut impl Write) -> io::Result<()> {
        writeln!(
            out,
            "ddc_index_present={} ddc_entries={} ddc_encoded_bytes={} cas_blobs={} cas_bytes={}",
            self.ddc_index_present,
            self.ddc_entries,
            self.ddc_encoded_bytes,
            self.cas_blobs,
            self.cas_bytes
        )
    }
}
impl CacheVerification {
    pub fn write_human(&self, out: &mut impl Write) -> io::Result<()> {
        self.status.write_human(out)?;
        writeln!(out, "current_revision={} checked_blobs={} checked_bytes={} checked_ddc_descriptors={} checked_current_descriptors={}",
            self.current_revision.map_or_else(|| "none".into(), |revision| revision.to_string()),
            self.checked_blobs, self.checked_bytes, self.checked_ddc_descriptors, self.checked_current_descriptors)
    }
}
impl CachePrune {
    pub fn write_human(&self, out: &mut impl Write) -> io::Result<()> {
        writeln!(out, "ddc_entries={} ddc_encoded_bytes={} evicted_keys={} removed_missing_keys={} grace_seconds={} cas_scanned={} cas_deleted={} cas_retained={} deferred_due_to_active_build={} deferred_due_to_pending_transaction={}",
            self.ddc_entries, self.ddc_encoded_bytes, self.evicted_keys.join(","), self.removed_missing_keys.join(","), self.grace_seconds,
            self.cas_scanned, self.cas_deleted, self.cas_retained, self.deferred_due_to_active_build, self.deferred_due_to_pending_transaction)
    }
}
