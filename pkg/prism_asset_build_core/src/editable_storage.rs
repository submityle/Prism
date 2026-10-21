//! Workspace-owned payload CAS, live read leases and durable history roots.
//!
//! Every publication and collection cooperates with the editable workspace
//! lock. A collector never waits for a reader while holding that lock: an
//! active snapshot or unpublished writer result defers collection instead.

use crate::editable::{
    EditableAsset, EditableAssetError, EditableWorkspace, MAX_EDITABLE_WORKSPACE_ENCODED_BYTES,
};
use crate::editable_payload::{
    EditablePayload, EditablePayloadRef, MAX_EDITABLE_EXTERNAL_PAYLOAD_BYTES,
};
use crate::{AssetId, CasError, CasSweepReport, ContentAddressedStore, ContentHash};
use fs4::fs_std::FileExt;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

const PAYLOADS: &str = ".prism-editable-payloads";
const HISTORY: &str = ".prism-editable-history";
const LEASE: &str = ".prism-editable-payloads.lock";
const PIN_MAGIC: &[u8] = b"PRISM-EDITABLE-PIN\0\x01";
const MAX_PINS: usize = 65_536;
const MAX_REFS: usize = 65_536;
const MAX_HISTORY_REFS: usize = 1_048_576;
const MAX_PIN_BYTES: usize = PIN_MAGIC.len() + 4 + MAX_REFS * 40 + 32;

fn io_error(operation: &'static str, error: io::Error) -> EditableAssetError {
    EditableAssetError::Io {
        operation,
        kind: error.kind(),
    }
}

fn cas_error(error: CasError) -> EditableAssetError {
    match error {
        CasError::Io(error) => io_error("payload CAS", error),
        CasError::Corrupt { .. } => EditableAssetError::InvalidPayloadReference,
    }
}

fn directory(path: &Path, create: bool) -> Result<(), EditableAssetError> {
    if create {
        match fs::create_dir(path) {
            Ok(()) => {
                sync_directory(
                    path.parent()
                        .ok_or(EditableAssetError::PackageOutsideWorkspace)?,
                )?;
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(io_error("create payload directory", error)),
        }
    }
    let metadata =
        fs::symlink_metadata(path).map_err(|e| io_error("inspect payload directory", e))?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        return Err(EditableAssetError::NonRegularFile);
    }
    Ok(())
}

fn existing_directory(path: &Path) -> Result<bool, EditableAssetError> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => Ok(true),
        Ok(_) => Err(EditableAssetError::NonRegularFile),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(io_error("inspect payload directory", error)),
    }
}

fn sync_directory(path: &Path) -> Result<(), EditableAssetError> {
    #[cfg(unix)]
    File::open(path)
        .and_then(|file| file.sync_all())
        .map_err(|e| io_error("sync payload directory", e))?;
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

fn regular(path: &Path) -> Result<fs::Metadata, EditableAssetError> {
    let metadata = fs::symlink_metadata(path).map_err(|e| io_error("inspect payload file", e))?;
    if !metadata.is_file() || metadata.file_type().is_symlink() {
        return Err(EditableAssetError::NonRegularFile);
    }
    Ok(metadata)
}

fn lease_file(workspace: &EditableWorkspace) -> Result<File, EditableAssetError> {
    let path = workspace.root().join(LEASE);
    match fs::symlink_metadata(&path) {
        Ok(_) => {
            regular(&path)?;
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(io_error("inspect payload lease", error)),
    }
    let file = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(&path)
        .map_err(|e| io_error("open payload lease", e))?;
    regular(&path)?;
    Ok(file)
}

fn shared_lease(workspace: &EditableWorkspace) -> Result<Arc<File>, EditableAssetError> {
    let file = lease_file(workspace)?;
    FileExt::lock_shared(&file).map_err(|e| io_error("lock payload lease", e))?;
    Ok(Arc::new(file))
}

/// Dropping the final clone releases protection for a persisted, unpublished
/// object. Keep this handle until a package or durable history pin names it.
#[derive(Clone, Debug)]
pub struct EditablePayloadLease {
    reference: EditablePayloadRef,
    _lease: Arc<File>,
}

impl EditablePayloadLease {
    pub fn reference(&self) -> EditablePayloadRef {
        self.reference
    }
}

/// Immutable metadata plus a shared GC lease. Cloning preserves the lease.
/// Extracted reference values alone are not roots; retain them explicitly for
/// durable undo/history before releasing the final snapshot clone.
#[derive(Clone, Debug)]
pub struct EditableWorkspaceSnapshot {
    assets: Arc<BTreeMap<AssetId, EditableAsset>>,
    root: PathBuf,
    _lease: Arc<File>,
}

impl Deref for EditableWorkspaceSnapshot {
    type Target = BTreeMap<AssetId, EditableAsset>;
    fn deref(&self) -> &Self::Target {
        &self.assets
    }
}

impl EditableWorkspaceSnapshot {
    pub(crate) fn new(
        workspace: &EditableWorkspace,
        assets: BTreeMap<AssetId, EditableAsset>,
    ) -> Result<Self, EditableAssetError> {
        Ok(Self {
            assets: Arc::new(assets),
            root: workspace.root().to_path_buf(),
            _lease: shared_lease(workspace)?,
        })
    }

    /// Resolves saved authoring bytes under this snapshot's lease. External
    /// bytes are length-checked before allocation and hash-verified before use.
    pub fn resolve_payload(
        &self,
        payload: &EditablePayload,
        max_bytes: u64,
    ) -> Result<Arc<[u8]>, EditableAssetError> {
        check_budget(payload.byte_len(), max_bytes)?;
        match payload {
            EditablePayload::Inline(bytes) => Ok(Arc::from(bytes.as_slice())),
            EditablePayload::External(reference) => {
                read_reference(&self.root, *reference, max_bytes)
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct EditablePayloadGcReport {
    pub sweep: CasSweepReport,
    pub deferred_due_to_active_readers: bool,
    pub history_pins: usize,
}

fn check_budget(length: u64, max_bytes: u64) -> Result<(), EditableAssetError> {
    if max_bytes > MAX_EDITABLE_EXTERNAL_PAYLOAD_BYTES || length > max_bytes {
        return Err(EditableAssetError::PayloadBudgetExceeded {
            requested: length.max(max_bytes),
            limit: max_bytes.min(MAX_EDITABLE_EXTERNAL_PAYLOAD_BYTES),
        });
    }
    Ok(())
}

fn open_reference(root: &Path, reference: EditablePayloadRef) -> Result<File, EditableAssetError> {
    if reference.byte_len > MAX_EDITABLE_EXTERNAL_PAYLOAD_BYTES {
        return Err(EditableAssetError::InvalidPayloadReference);
    }
    let store_root = root.join(PAYLOADS);
    directory(&store_root, false)?;
    let hex = reference.content_hash.to_hex();
    let shard = store_root.join(&hex[..2]);
    directory(&shard, false)?;
    let path = shard.join(&hex[2..]);
    if regular(&path)?.len() != reference.byte_len {
        return Err(EditableAssetError::InvalidPayloadReference);
    }
    let file = File::open(&path).map_err(|e| io_error("open payload", e))?;
    let metadata = file
        .metadata()
        .map_err(|e| io_error("inspect opened payload", e))?;
    if !metadata.is_file() || metadata.len() != reference.byte_len {
        return Err(EditableAssetError::InvalidPayloadReference);
    }
    Ok(file)
}

fn verify_reference(root: &Path, reference: EditablePayloadRef) -> Result<(), EditableAssetError> {
    let mut file = open_reference(root, reference)?;
    let mut hasher = blake3::Hasher::new();
    let mut buffer = [0u8; 64 * 1024];
    let mut left = reference.byte_len;
    while left > 0 {
        let limit = left.min(buffer.len() as u64) as usize;
        file.read_exact(&mut buffer[..limit])
            .map_err(|e| io_error("verify payload", e))?;
        hasher.update(&buffer[..limit]);
        left -= limit as u64;
    }
    if file
        .read(&mut buffer[..1])
        .map_err(|e| io_error("verify payload end", e))?
        != 0
        || ContentHash(*hasher.finalize().as_bytes()) != reference.content_hash
    {
        return Err(EditableAssetError::InvalidPayloadReference);
    }
    Ok(())
}

fn read_reference(
    root: &Path,
    reference: EditablePayloadRef,
    max_bytes: u64,
) -> Result<Arc<[u8]>, EditableAssetError> {
    check_budget(reference.byte_len, max_bytes)?;
    let file = open_reference(root, reference)?;
    let length = usize::try_from(reference.byte_len)
        .map_err(|_| EditableAssetError::InvalidPayloadReference)?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(length).map_err(|_| {
        io_error(
            "allocate payload",
            io::Error::from(io::ErrorKind::OutOfMemory),
        )
    })?;
    file.take(reference.byte_len + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| io_error("read payload", e))?;
    if bytes.len() != length || ContentHash::digest(&bytes) != reference.content_hash {
        return Err(EditableAssetError::InvalidPayloadReference);
    }
    Ok(Arc::from(bytes))
}

fn references<'a>(
    assets: impl IntoIterator<Item = &'a EditableAsset>,
) -> Result<BTreeSet<EditablePayloadRef>, EditableAssetError> {
    let mut refs = BTreeSet::new();
    let mut lengths = BTreeMap::new();
    for asset in assets {
        if let EditablePayload::External(reference) = &asset.authoring_payload {
            if lengths
                .insert(reference.content_hash, reference.byte_len)
                .is_some_and(|n| n != reference.byte_len)
            {
                return Err(EditableAssetError::InvalidPayloadReference);
            }
            refs.insert(*reference);
        }
    }
    Ok(refs)
}

pub(crate) fn validate_workspace_payloads<'a>(
    workspace: &EditableWorkspace,
    assets: impl IntoIterator<Item = &'a EditableAsset>,
) -> Result<(), EditableAssetError> {
    for reference in references(assets)? {
        verify_reference(workspace.root(), reference)?;
    }
    Ok(())
}

impl EditableWorkspace {
    /// Streams an immutable object before publishing a package reference. The
    /// returned shared lease closes the write-to-publication GC race.
    pub fn store_payload(
        &self,
        mut reader: impl Read,
        max_bytes: u64,
    ) -> Result<EditablePayloadLease, EditableAssetError> {
        check_budget(0, max_bytes)?;
        self.with_lock(|| self.store_payload_locked(&mut reader, max_bytes))
    }

    pub(crate) fn store_payload_locked(
        &self,
        mut reader: impl Read,
        max_bytes: u64,
    ) -> Result<EditablePayloadLease, EditableAssetError> {
        check_budget(0, max_bytes)?;
        let root = self.root().join(PAYLOADS);
        directory(&root, true)?;
        // Refuse symlink shards before handing the layout to the CAS writer.
        for (index, entry) in fs::read_dir(&root)
            .map_err(|e| io_error("inspect payload shards", e))?
            .enumerate()
        {
            if index >= 512 {
                return Err(EditableAssetError::InvalidPayloadReference);
            }
            let entry = entry.map_err(|e| io_error("inspect payload shard", e))?;
            let name = entry.file_name();
            if name
                .to_str()
                .is_some_and(|s| s.len() == 2 && s.bytes().all(|c| c.is_ascii_hexdigit()))
            {
                directory(&entry.path(), false)?;
            }
        }
        let lease = shared_lease(self)?;
        let store = ContentAddressedStore::open(root).map_err(cas_error)?;
        let mut writer = store.begin_write(max_bytes).map_err(cas_error)?;
        let mut buffer = [0u8; 64 * 1024];
        loop {
            let count = loop {
                match reader.read(&mut buffer) {
                    Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                    result => break result.map_err(|e| io_error("read authoring payload", e))?,
                }
            };
            if count == 0 {
                break;
            }
            writer
                .write_all(&buffer[..count])
                .map_err(|e| io_error("write authoring payload", e))?;
        }
        let result = writer.finish().map_err(cas_error)?;
        let reference = EditablePayloadRef {
            content_hash: result.content_hash,
            byte_len: result.byte_len,
        };
        open_reference(self.root(), reference)?;
        Ok(EditablePayloadLease {
            reference,
            _lease: lease,
        })
    }

    pub(crate) fn resolve_payload_locked(
        &self,
        payload: &EditablePayload,
        max_bytes: u64,
    ) -> Result<Arc<[u8]>, EditableAssetError> {
        check_budget(payload.byte_len(), max_bytes)?;
        match payload {
            EditablePayload::Inline(bytes) => Ok(Arc::from(bytes.as_slice())),
            EditablePayload::External(reference) => {
                read_reference(self.root(), *reference, max_bytes)
            }
        }
    }

    /// Atomically creates or replaces a durable history root. The caller owns
    /// the nonzero history identity and explicitly releases obsolete entries.
    /// Pin before dropping the snapshot/writer lease from which refs originated.
    pub fn retain_payloads(
        &self,
        history: u128,
        refs: &[EditablePayloadRef],
    ) -> Result<(), EditableAssetError> {
        if history == 0 || refs.len() > MAX_REFS {
            return Err(EditableAssetError::InvalidPayloadRetention);
        }
        let refs = refs.iter().copied().collect::<BTreeSet<_>>();
        if refs
            .iter()
            .zip(refs.iter().skip(1))
            .any(|(a, b)| a.content_hash == b.content_hash && a.byte_len != b.byte_len)
        {
            return Err(EditableAssetError::InvalidPayloadRetention);
        }
        self.with_lock(|| {
            let root = self.root().join(HISTORY);
            directory(&root, true)?;
            let mut pins = load_pins(&root)?;
            pins.insert(history, refs.clone());
            validate_pin_counts(&pins)?;
            for reference in &refs {
                verify_reference(self.root(), *reference)?;
            }
            let bytes = encode_pin(&refs);
            let path = root.join(format!("{history:032x}.pin"));
            match fs::symlink_metadata(&path) {
                Ok(_) => {
                    regular(&path)?;
                }
                Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                Err(error) => return Err(io_error("inspect retention target", error)),
            }
            let mut temporary = tempfile::Builder::new()
                .prefix(".pin-")
                .tempfile_in(&root)
                .map_err(|e| io_error("create retention record", e))?;
            temporary
                .write_all(&bytes)
                .map_err(|e| io_error("write retention record", e))?;
            temporary
                .as_file()
                .sync_all()
                .map_err(|e| io_error("sync retention record", e))?;
            temporary
                .persist(path)
                .map_err(|e| io_error("publish retention record", e.error))?;
            sync_directory(&root)
        })
    }

    pub fn retain_snapshot(
        &self,
        history: u128,
        snapshot: &EditableWorkspaceSnapshot,
    ) -> Result<(), EditableAssetError> {
        if self.root() != snapshot.root {
            return Err(EditableAssetError::InvalidPayloadRetention);
        }
        self.retain_payloads(
            history,
            &references(snapshot.values())?
                .into_iter()
                .collect::<Vec<_>>(),
        )
    }

    /// Idempotently releases one durable root; current packages and live
    /// snapshots remain protected independently.
    pub fn release_payloads(&self, history: u128) -> Result<(), EditableAssetError> {
        if history == 0 {
            return Err(EditableAssetError::InvalidPayloadRetention);
        }
        self.with_lock(|| {
            let root = self.root().join(HISTORY);
            if !existing_directory(&root)? {
                return Ok(());
            }
            directory(&root, false)?;
            let path = root.join(format!("{history:032x}.pin"));
            match fs::symlink_metadata(&path) {
                Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
                Err(error) => return Err(io_error("inspect retention record", error)),
                Ok(_) => {
                    regular(&path)?;
                }
            }
            fs::remove_file(path).map_err(|e| io_error("release retention record", e))?;
            sync_directory(&root)
        })
    }

    /// Recovers interrupted package publication first, then collects only
    /// unreachable payloads. Invalid packages/pins abort before any deletion.
    pub fn collect_payloads(
        &self,
        grace: Duration,
    ) -> Result<EditablePayloadGcReport, EditableAssetError> {
        self.with_lock(|| {
            let assets = self.capture_assets_locked(MAX_EDITABLE_WORKSPACE_ENCODED_BYTES)?;
            let current_refs = references(assets.values())?;
            let history = self.root().join(HISTORY);
            let pins = if existing_directory(&history)? {
                load_pins(&history)?
            } else {
                BTreeMap::new()
            };
            let mut lengths = BTreeMap::new();
            for reference in current_refs.iter().chain(pins.values().flatten()) {
                if lengths
                    .insert(reference.content_hash, reference.byte_len)
                    .is_some_and(|length| length != reference.byte_len)
                {
                    return Err(EditableAssetError::InvalidPayloadRetention);
                }
            }
            let reachable = lengths.into_keys().collect::<BTreeSet<_>>();
            let mut report = EditablePayloadGcReport {
                history_pins: pins.len(),
                ..Default::default()
            };
            let lease = lease_file(self)?;
            match FileExt::try_lock_exclusive(&lease) {
                Ok(true) => {}
                Ok(false) => {
                    report.deferred_due_to_active_readers = true;
                    return Ok(report);
                }
                Err(error) => return Err(io_error("try exclusive payload lease", error)),
            }
            let root = self.root().join(PAYLOADS);
            if existing_directory(&root)? {
                directory(&root, false)?;
                report.sweep = ContentAddressedStore::open(root)
                    .map_err(cas_error)?
                    .sweep_unreachable(&reachable, grace)
                    .map_err(cas_error)?;
            }
            Ok(report)
        })
    }
}

fn validate_pin_counts(
    pins: &BTreeMap<u128, BTreeSet<EditablePayloadRef>>,
) -> Result<(), EditableAssetError> {
    if pins.len() > MAX_PINS || pins.values().map(BTreeSet::len).sum::<usize>() > MAX_HISTORY_REFS {
        return Err(EditableAssetError::InvalidPayloadRetention);
    }
    let mut lengths = BTreeMap::new();
    for reference in pins.values().flatten() {
        if lengths
            .insert(reference.content_hash, reference.byte_len)
            .is_some_and(|length| length != reference.byte_len)
        {
            return Err(EditableAssetError::InvalidPayloadRetention);
        }
    }
    Ok(())
}

fn encode_pin(refs: &BTreeSet<EditablePayloadRef>) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(PIN_MAGIC.len() + 4 + refs.len() * 40 + 32);
    bytes.extend_from_slice(PIN_MAGIC);
    bytes.extend_from_slice(&(refs.len() as u32).to_le_bytes());
    for reference in refs {
        bytes.extend_from_slice(&reference.content_hash.0);
        bytes.extend_from_slice(&reference.byte_len.to_le_bytes());
    }
    let checksum = ContentHash::digest(&bytes);
    bytes.extend_from_slice(&checksum.0);
    bytes
}

fn load_pins(
    root: &Path,
) -> Result<BTreeMap<u128, BTreeSet<EditablePayloadRef>>, EditableAssetError> {
    directory(root, false)?;
    let mut pins = BTreeMap::new();
    let mut entries = 0usize;
    let mut total_refs = 0usize;
    let mut lengths = BTreeMap::new();
    for entry in fs::read_dir(root).map_err(|e| io_error("scan retention records", e))? {
        entries += 1;
        if entries > MAX_PINS * 2 {
            return Err(EditableAssetError::InvalidPayloadRetention);
        }
        let entry = entry.map_err(|e| io_error("read retention entry", e))?;
        let path = entry.path();
        let name = entry.file_name();
        let name = name
            .to_str()
            .ok_or(EditableAssetError::InvalidPayloadRetention)?;
        if name.starts_with(".pin-") {
            regular(&path)?;
            continue;
        }
        let id = name
            .strip_suffix(".pin")
            .filter(|s| {
                s.len() == 32
                    && s.bytes()
                        .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
            })
            .and_then(|s| u128::from_str_radix(s, 16).ok())
            .filter(|id| *id != 0)
            .ok_or(EditableAssetError::InvalidPayloadRetention)?;
        let length = regular(&path)?.len();
        if length > MAX_PIN_BYTES as u64 {
            return Err(EditableAssetError::InvalidPayloadRetention);
        }
        // Count and aggregate reference budgets are checked before allocating
        // each body; the on-disk fixed header cannot amplify metadata into RAM.
        let mut file = File::open(&path).map_err(|e| io_error("open retention record", e))?;
        let opened = file
            .metadata()
            .map_err(|e| io_error("inspect opened retention record", e))?;
        if !opened.is_file() || opened.len() != length {
            return Err(EditableAssetError::InvalidPayloadRetention);
        }
        let mut header = vec![0u8; PIN_MAGIC.len() + 4];
        file.read_exact(&mut header)
            .map_err(|e| io_error("read retention header", e))?;
        if !header.starts_with(PIN_MAGIC) {
            return Err(EditableAssetError::InvalidPayloadRetention);
        }
        let count = u32::from_le_bytes(
            header[PIN_MAGIC.len()..]
                .try_into()
                .expect("four count bytes"),
        ) as usize;
        total_refs = total_refs
            .checked_add(count)
            .ok_or(EditableAssetError::InvalidPayloadRetention)?;
        if count > MAX_REFS
            || total_refs > MAX_HISTORY_REFS
            || length != (header.len() + count * 40 + 32) as u64
        {
            return Err(EditableAssetError::InvalidPayloadRetention);
        }
        let mut bytes = header;
        file.take(length + 1)
            .read_to_end(&mut bytes)
            .map_err(|e| io_error("read retention record", e))?;
        if bytes.len() as u64 != length
            || ContentHash::digest(&bytes[..bytes.len() - 32]).0 != bytes[bytes.len() - 32..]
        {
            return Err(EditableAssetError::InvalidPayloadRetention);
        }
        let mut refs = BTreeSet::new();
        let mut previous = None;
        for record in bytes[PIN_MAGIC.len() + 4..bytes.len() - 32].chunks_exact(40) {
            let reference = EditablePayloadRef {
                content_hash: ContentHash(record[..32].try_into().expect("hash")),
                byte_len: u64::from_le_bytes(record[32..].try_into().expect("length")),
            };
            if reference.byte_len > MAX_EDITABLE_EXTERNAL_PAYLOAD_BYTES
                || previous.is_some_and(|p| p >= reference)
                || lengths
                    .insert(reference.content_hash, reference.byte_len)
                    .is_some_and(|n| n != reference.byte_len)
            {
                return Err(EditableAssetError::InvalidPayloadRetention);
            }
            previous = Some(reference);
            refs.insert(reference);
        }
        pins.insert(id, refs);
        if pins.len() > MAX_PINS {
            return Err(EditableAssetError::InvalidPayloadRetention);
        }
    }
    Ok(pins)
}
