//! Native loading from unique immutable sibling snapshots.

use std::collections::BTreeSet;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use libloading::{Library, Symbol};

use super::abi::{self, Entry, ModuleApi};
use super::{ModuleReloadConfig, ModuleReloadError};

static SNAPSHOT_SEQUENCE: AtomicU64 = AtomicU64::new(0);

pub(super) fn read_binary(path: &Path, limit: usize) -> Result<Vec<u8>, ModuleReloadError> {
    let file = File::open(path).map_err(load_error)?;
    if file.metadata().map_err(load_error)?.len() > limit as u64 {
        return Err(ModuleReloadError::Limit("library"));
    }
    let mut bytes = Vec::new();
    file.take(limit as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(load_error)?;
    if bytes.len() > limit {
        return Err(ModuleReloadError::Limit("library"));
    }
    Ok(bytes)
}

struct Snapshot(PathBuf);

impl Snapshot {
    fn create(source: &Path, bytes: &[u8]) -> Result<Self, ModuleReloadError> {
        // Keep $ORIGIN/@loader_path dependencies relative to the original
        // directory. Never overwrite the linker output or a previous snapshot.
        let source = source.canonicalize().map_err(load_error)?;
        let directory = source
            .parent()
            .ok_or_else(|| ModuleReloadError::Load("library has no parent".into()))?;
        for _ in 0..32 {
            let sequence = SNAPSHOT_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let mut name = std::ffi::OsString::from(format!(
                ".prism-module-{}-{sequence}-",
                std::process::id()
            ));
            name.push(
                source
                    .file_name()
                    .ok_or_else(|| ModuleReloadError::Load("library has no filename".into()))?,
            );
            let path = directory.join(name);
            let mut options = OpenOptions::new();
            options.write(true).create_new(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                options.mode(0o700);
            }
            let mut file = match options.open(&path) {
                Ok(file) => file,
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(load_error(error)),
            };
            let snapshot = Self(path);
            let written = file.write_all(bytes).and_then(|()| file.sync_all());
            drop(file);
            written.map_err(load_error)?;
            return Ok(snapshot);
        }
        Err(ModuleReloadError::Load(
            "could not allocate a unique library snapshot".into(),
        ))
    }
}

impl Drop for Snapshot {
    fn drop(&mut self) {
        // Only this create_new-owned snapshot is removed. The Library field is
        // dropped first, which also permits removal on Windows.
        let _ = fs::remove_file(&self.0);
    }
}

/// Owns both the mapping and copied callbacks. Field order closes the mapping
/// before removing its backing file. No callback escapes this ownership chain.
pub(super) struct LoadedModule {
    _library: Library,
    _snapshot: Snapshot,
    pub(super) api: ModuleApi,
    systems: Vec<u64>,
}

impl LoadedModule {
    /// # Safety
    /// `source` and `bytes` must identify trusted native code that obeys the
    /// complete module ABI. Loading native code cannot sandbox callback behavior.
    pub(super) unsafe fn load(
        source: &Path,
        bytes: &[u8],
        config: ModuleReloadConfig,
    ) -> Result<Arc<Self>, ModuleReloadError> {
        let snapshot = Snapshot::create(source, bytes)?;
        // SAFETY: the outer load contract requires trusted native constructors
        // and code. Snapshot is unique and immutable for the mapping's lifetime.
        let library = unsafe { Library::new(&snapshot.0) }.map_err(load_error)?;
        // SAFETY: the trusted module exports exactly Entry under ENTRY_SYMBOL.
        let entry: Symbol<'_, Entry> =
            unsafe { library.get(abi::ENTRY_SYMBOL) }.map_err(load_error)?;
        let mut api = ModuleApi::default();
        // SAFETY: output points to a live writable descriptor of the exact
        // advertised capacity; the entry obeys the bounded ABI write contract.
        let status = unsafe { entry(abi::ABI_VERSION, &mut api, size_of::<ModuleApi>()) };
        if status != abi::OK {
            return Err(ModuleReloadError::Callback {
                stage: "entry",
                status,
            });
        }
        validate(&api, config)?;
        let mut systems = BTreeSet::new();
        let enumerate = api.system_id.expect("validated callback");
        for index in 0..api.system_count {
            let mut id = 0;
            // SAFETY: descriptor was validated and the mapping is still owned.
            let status = unsafe { enumerate(index, &mut id) };
            if status != abi::OK {
                return Err(ModuleReloadError::Callback {
                    stage: "system_id",
                    status,
                });
            }
            if id == 0 || !systems.insert(id) {
                return Err(ModuleReloadError::Abi(
                    "system IDs must be nonzero and unique".into(),
                ));
            }
        }
        Ok(Arc::new(Self {
            _library: library,
            _snapshot: snapshot,
            api,
            systems: systems.into_iter().collect(),
        }))
    }

    pub(super) fn systems(&self) -> &[u64] {
        &self.systems
    }
}

fn validate(api: &ModuleApi, config: ModuleReloadConfig) -> Result<(), ModuleReloadError> {
    if api.magic != abi::MAGIC
        || api.abi_version != abi::ABI_VERSION
        || api.struct_size as usize != size_of::<ModuleApi>()
    {
        return Err(ModuleReloadError::Abi(
            "descriptor magic, ABI version or size mismatch".into(),
        ));
    }
    if api.module_id != config.module_id || api.host_contract != config.host_contract {
        return Err(ModuleReloadError::Abi(
            "module identity or host protocol mismatch".into(),
        ));
    }
    if api.flags != abi::THREAD_TRANSFER || api.state_schema == 0 {
        return Err(ModuleReloadError::Abi(
            "unsupported flags or zero state schema".into(),
        ));
    }
    if api.system_count == 0 || api.system_count > 4096 {
        return Err(ModuleReloadError::Abi(
            "system count must be in 1..=4096".into(),
        ));
    }
    if api.system_id.is_none()
        || api.create.is_none()
        || api.destroy.is_none()
        || api.export.is_none()
        || api.run.is_none()
    {
        return Err(ModuleReloadError::Abi("required callback missing".into()));
    }
    Ok(())
}

fn load_error(error: impl std::fmt::Display) -> ModuleReloadError {
    ModuleReloadError::Load(error.to_string())
}
