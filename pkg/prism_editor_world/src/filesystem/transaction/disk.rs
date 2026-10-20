//! One isolated read path, two bounded slots and a durable redo decision.
use super::{format::*, *};

const ROOT_MAGIC: [u8; 8] = *b"PEWROOT\0";
const MANIFEST_MAGIC: [u8; 8] = *b"PEWMANF\0";
const JOURNAL_MAGIC: [u8; 8] = *b"PEWJRNL\0";

struct Version {
    pointer: Pointer,
    files: Vec<WorldFileSnapshot>,
    request: Option<WorldTransactionRequest>,
}

fn exists(path: &Path) -> Result<bool, FileCellIoError> {
    match fs::symlink_metadata(path) {
        Ok(_) => {
            check_regular(path)?;
            Ok(true)
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(io_error("inspect transaction file", e)),
    }
}
fn remove(path: &Path) -> Result<(), FileCellIoError> {
    if exists(path)? {
        fs::remove_file(path).map_err(|e| io_error("remove transaction file", e))?;
    }
    Ok(())
}
fn sync(dir: &Directory) -> Result<(), FileCellIoError> {
    dir.handle
        .sync_all()
        .map_err(|e| io_error("sync transaction directory", e))
}
fn reclaim_slot(slot: &Directory, cancel: &Cancellation) -> Result<(), FileCellIoError> {
    let entries = fs::read_dir(&slot.path).map_err(|e| io_error("enumerate inactive slot", e))?;
    let mut paths = Vec::new();
    for entry in entries.take(MAX_ENTRIES + 2) {
        cancel.check()?;
        let entry = entry.map_err(|e| io_error("inspect inactive slot", e))?;
        if paths.len() > MAX_ENTRIES {
            return Err(FileCellIoError::Capacity);
        }
        let name = entry.file_name();
        let name = name
            .to_str()
            .ok_or_else(|| FileCellIoError::UnsafePath(entry.path()))?;
        let numeric = name
            .parse::<usize>()
            .ok()
            .is_some_and(|n| n < MAX_ENTRIES && n.to_string() == name);
        if name != "manifest" && !numeric {
            return Err(FileCellIoError::UnsafePath(entry.path()));
        }
        check_regular(&entry.path())?;
        paths.push(entry.path());
    }
    // Validate the complete bounded inventory before removing any entry.
    for path in paths {
        cancel.check()?;
        remove(&path)?;
    }
    Ok(())
}
fn read_meta(
    path: &Path,
    magic: [u8; 8],
    cancel: &Cancellation,
) -> Result<Vec<u8>, FileCellIoError> {
    let read = read_envelope(
        &mut open_regular(path)?,
        &identity(magic),
        META_LIMIT,
        None,
        cancel,
    )?;
    if read.revision != 0 {
        return Err(FileCellIoError::Corrupt("metadata revision"));
    }
    Ok(read.bytes)
}
fn write_meta(
    path: &Path,
    magic: [u8; 8],
    bytes: &[u8],
    cancel: &Cancellation,
) -> Result<(), FileCellIoError> {
    if bytes.len() > META_LIMIT {
        return Err(FileCellIoError::Capacity);
    }
    remove(path)?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| io_error("create transaction metadata", e))?;
    write_envelope(&mut file, &identity(magic), 0, bytes, cancel)?;
    file.sync_all()
        .map_err(|e| io_error("sync transaction metadata", e))
}
fn root(store: &Directory, cancel: &Cancellation) -> Result<Pointer, FileCellIoError> {
    let path = store.path.join("CURRENT");
    if exists(&path)? {
        Pointer::decode(&read_meta(&path, ROOT_MAGIC, cancel)?)
    } else {
        Ok(Pointer::default())
    }
}
fn publish(store: &Directory, next: Pointer) -> Result<(), FileCellIoError> {
    write_meta(
        &store.path.join("CURRENT.pending"),
        ROOT_MAGIC,
        &next.encode(),
        &Cancellation::default(),
    )?;
    exists(&store.path.join("CURRENT"))?;
    fs::rename(
        store.path.join("CURRENT.pending"),
        store.path.join("CURRENT"),
    )
    .map_err(|e| io_error("publish transaction root", e))?;
    sync(store)
}
fn uncertain(error: FileCellIoError) -> FileCellIoError {
    match error {
        FileCellIoError::Io { source, .. } | FileCellIoError::CommitUncertain(source) => {
            FileCellIoError::CommitUncertain(source)
        }
        e => FileCellIoError::CommitUncertain(io::Error::other(e.to_string())),
    }
}

fn shape(files: &[WorldFileSnapshot]) -> Result<(), FileCellIoError> {
    let mut cells = BTreeSet::new();
    let mut actors = BTreeSet::new();
    for f in files {
        if let WorldFileTarget::Cell(c) = &f.target
            && !cells.insert((c.coord().x, c.coord().y, c.coord().z))
        {
            return Err(FileCellIoError::InvalidRequest);
        }
    }
    for f in files {
        if let WorldFileTarget::Actor(a) = &f.target {
            let c = a.cell();
            if !actors.insert(a.actor()) || !cells.contains(&(c.x, c.y, c.z)) {
                return Err(FileCellIoError::InvalidRequest);
            }
        }
    }
    Ok(())
}

impl FileWorldTransactionProvider {
    fn version(
        &self,
        store: &Directory,
        pointer: Pointer,
        cancel: &Cancellation,
    ) -> Result<Version, FileCellIoError> {
        if pointer.generation == 0 {
            return Ok(Version {
                pointer,
                files: Vec::new(),
                request: None,
            });
        }
        let slot = Directory::child(
            &store.path,
            &format!("slot{}", pointer.generation % 2),
            false,
        )?;
        let bytes = read_meta(&slot.path.join("manifest"), MANIFEST_MAGIC, cancel)?;
        if digest(&bytes) != pointer.hash {
            return Err(FileCellIoError::Corrupt("root manifest hash"));
        }
        let (entries, descriptors) = decode_manifest(&bytes, self.limits)?;
        let mut files = Vec::with_capacity(entries.len());
        for (index, e) in entries.into_iter().enumerate() {
            cancel.check()?;
            let data = read_envelope(
                &mut open_regular(&slot.path.join(index.to_string()))?,
                &e.target.envelope(),
                e.length,
                None,
                cancel,
            )?;
            if data.revision != e.revision
                || data.bytes.len() != e.length
                || digest(&data.bytes) != e.hash
            {
                return Err(FileCellIoError::Corrupt("manifest envelope mismatch"));
            }
            files.push(WorldFileSnapshot {
                target: e.target,
                revision: e.revision,
                bytes: data.bytes.into(),
            });
        }
        shape(&files).map_err(|_| FileCellIoError::Corrupt("manifest ownership"))?;
        let mut changes = Vec::with_capacity(descriptors.len());
        for d in descriptors {
            let file = files.iter().find(|f| f.target == d.target);
            let value = match (d.revision, file) {
                (None, None) => None,
                (Some(r), Some(f)) if r == f.revision => Some((r, f.bytes.clone())),
                _ => return Err(FileCellIoError::Corrupt("request result mismatch")),
            };
            if matches!((d.base, &value), (ActorRevisionBase::Missing, None))
                || matches!((d.base,&value),(ActorRevisionBase::Revision(b),Some((r,_))) if *r<=b)
            {
                return Err(FileCellIoError::Corrupt("request revision"));
            }
            changes.push(WorldFileChange {
                target: d.target,
                base: d.base,
                value,
            });
        }
        let request = WorldTransactionRequest::new(pointer.id, pointer.generation - 1, changes)
            .map_err(|_| FileCellIoError::Corrupt("request metadata"))?;
        Ok(Version {
            pointer,
            files,
            request: Some(request),
        })
    }

    /// Called only with the exclusive root gate held. A decided journal is
    /// authoritative even if publication/cleanup was interrupted. Validation
    /// completes before entering the non-cancellable recovery fence.
    fn recover(&self, store: &Directory, cancel: &Cancellation) -> Result<(), FileCellIoError> {
        let path = store.path.join("journal");
        if !exists(&path)? {
            return Ok(());
        }
        let bytes = read_meta(&path, JOURNAL_MAGIC, cancel)?;
        let mut r = Reader(&bytes);
        let parent = Pointer::read(&mut r)?;
        let next = Pointer::read(&mut r)?;
        r.end()?;
        if parent.generation.checked_add(1) != Some(next.generation) {
            return Err(FileCellIoError::Corrupt("journal generation"));
        }
        let current = root(store, cancel)?;
        if current != parent && current != next {
            return Err(FileCellIoError::Corrupt("journal root divergence"));
        }
        // Fully read every candidate envelope, never publish a partial slot.
        self.version(store, next, cancel)?;
        let _fence = cancel.commit_guard()?;
        if current != next {
            publish(store, next).map_err(uncertain)?;
        }
        remove(&path).map_err(uncertain)?;
        sync(store).map_err(uncertain)
    }

    pub(super) fn read(
        &self,
        cancel: &Cancellation,
    ) -> Result<WorldTransactionSnapshot, FileCellIoError> {
        cancel.check()?;
        let _gate = root_gate(self.root(), true)?;
        let store = Directory::child(self.root(), STORE, false)?;
        self.recover(&store, cancel)?;
        let version = self.version(&store, root(&store, cancel)?, cancel)?;
        Ok(WorldTransactionSnapshot {
            generation: version.pointer.generation,
            id: version.pointer.id,
            files: version.files,
            root: self.root().to_owned(),
        })
    }

    pub(super) fn execute(
        &self,
        request: WorldTransactionRequest,
        cancel: &Cancellation,
    ) -> Result<WorldTransactionReceipt, FileCellIoError> {
        cancel.check()?;
        let _gate = root_gate(self.root(), true)?;
        let store = Directory::child(self.root(), STORE, false)?;
        self.recover(&store, cancel)?;
        let old = self.version(&store, root(&store, cancel)?, cancel)?;
        let receipt = || WorldTransactionReceipt {
            request: request.clone(),
            root: self.root().to_owned(),
        };
        if old.request.as_ref() == Some(&request) {
            sync(&store)?;
            return Ok(receipt());
        }
        if old.pointer.generation != request.base_generation || old.pointer.id == request.id {
            return Err(FileCellIoError::RevisionConflict {
                requested: request.base_generation,
                current: Some(old.pointer.generation),
            });
        }
        // A bounded fixed stripe set prevents permanent lock-file growth.
        // Global gate precedes unique ascending stripes in every executor.
        let lockdir = Directory::child(&store.path, "locks", true)?;
        let mut stripes = BTreeSet::new();
        for c in &request.changes {
            let mut key = Vec::new();
            target(&mut key, &c.target);
            stripes.insert(digest(&key) % 64);
        }
        let mut guards = Vec::with_capacity(stripes.len());
        for n in stripes {
            cancel.check()?;
            guards.push(lockdir.acquire_path(&lockdir.path.join(n.to_string()), true)?);
        }
        for change in &request.changes {
            let current = old
                .files
                .iter()
                .find(|f| f.target == change.target)
                .map(|f| f.revision);
            let expected = match change.base {
                ActorRevisionBase::Missing => None,
                ActorRevisionBase::Revision(r) => Some(r),
            };
            if current != expected {
                return Err(FileCellIoError::RevisionConflict {
                    requested: expected.unwrap_or(0),
                    current,
                });
            }
            // A move creates a new per-cell target, but must still advance
            // the same actor GUID's durable revision across its old owner.
            if let (WorldFileTarget::Actor(actor), Some((revision, _))) =
                (&change.target, &change.value)
                && let Some(previous) = old.files.iter().find(
                    |f| matches!(&f.target,WorldFileTarget::Actor(a) if a.actor()==actor.actor()),
                )
                && *revision <= previous.revision
            {
                return Err(FileCellIoError::RevisionConflict {
                    requested: *revision,
                    current: Some(previous.revision),
                });
            }
        }
        let mut files = old.files;
        for change in &request.changes {
            cancel.check()?;
            files.retain(|f| f.target != change.target);
            if let Some((revision, bytes)) = &change.value {
                files.push(WorldFileSnapshot {
                    target: change.target.clone(),
                    revision: *revision,
                    bytes: bytes.clone(),
                });
            }
        }
        files.sort_by_key(|f| f.target.key());
        shape(&files)?;
        if files.len() > self.limits.entries
            || files.iter().map(|f| f.bytes.len()).sum::<usize>() > self.limits.snapshot_bytes
        {
            return Err(FileCellIoError::Capacity);
        }
        let manifest = manifest(&request, &files);
        let next = Pointer {
            generation: request.base_generation + 1,
            id: request.id,
            hash: digest(&manifest),
        };
        let slot = Directory::child(&store.path, &format!("slot{}", next.generation % 2), true)?;
        // Only known store-owned paths are reclaimed. The published slot is
        // untouched; owned snapshots do not depend on either slot's lifetime.
        reclaim_slot(&slot, cancel)?;
        for (index, f) in files.iter().enumerate() {
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(slot.path.join(index.to_string()))
                .map_err(|e| io_error("stage transaction envelope", e))?;
            write_envelope(
                &mut file,
                &f.target.envelope(),
                f.revision,
                &f.bytes,
                cancel,
            )?;
            file.sync_all()
                .map_err(|e| io_error("sync transaction envelope", e))?;
        }
        write_meta(
            &slot.path.join("manifest"),
            MANIFEST_MAGIC,
            &manifest,
            cancel,
        )?;
        sync(&slot)?;
        sync(&store)?;
        let mut journal = old.pointer.encode();
        journal.extend(next.encode());
        write_meta(
            &store.path.join("journal.pending"),
            JOURNAL_MAGIC,
            &journal,
            cancel,
        )?;
        #[cfg(test)]
        self.io
            .cell
            .shared
            .checkpoint(TestPhase::TransactionPrepared)
            .map_err(|e| io_error("prepared transaction checkpoint", e))?;
        let _fence = cancel.commit_guard()?;
        // This rename plus directory sync durably decides commit. Any error
        // after rename is uncertain and requires recovery, never rollback.
        fs::rename(
            store.path.join("journal.pending"),
            store.path.join("journal"),
        )
        .map_err(|e| io_error("decide transaction", e))?;
        sync(&store).map_err(uncertain)?;
        #[cfg(test)]
        self.io
            .cell
            .shared
            .checkpoint(TestPhase::TransactionDecided)
            .map_err(FileCellIoError::CommitUncertain)?;
        publish(&store, next).map_err(uncertain)?;
        #[cfg(test)]
        self.io
            .cell
            .shared
            .checkpoint(TestPhase::TransactionPublished)
            .map_err(FileCellIoError::CommitUncertain)?;
        remove(&store.path.join("journal")).map_err(uncertain)?;
        sync(&store).map_err(uncertain)?;
        Ok(receipt())
    }
}
