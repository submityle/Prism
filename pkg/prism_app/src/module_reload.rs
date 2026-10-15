//! Native module reload through a versioned host contract (`module-reload`).
//!
//! [`ModuleReload::poll`] snapshots a real dynamic library, checks its ABI and
//! stable system IDs, exports the current state, and either passes it through
//! the ABI contract or migrates it through a host-owned
//! [`ReflectStateContract`]. Only a fully prepared candidate replaces the
//! active instance.
//! Preparation cannot access the World; failed loads and migrations preserve
//! the old module. World entities, resources and schedule nodes stay in place.
//!
//! Bind [`ModuleReload::system`] once into an ordinary ECS Schedule. Its host
//! wrapper resolves the current instance on every run, preserving conditions,
//! ordering, lanes and change ticks across reload. Module systems are exclusive
//! and serialized, including when installed in different schedules.
//!
//! The ABI is deliberately a host operation protocol: ECS values, allocators,
//! Rust `TypeId`s, function objects and module-owned drop glue never cross it.
//! Application-specific operations live in a [`ModuleHost`] compiled into the
//! [`ReflectStateContract`] gives module state typed schema evolution while the
//! ABI remains opaque bytes; the World is retained.
//! Loading native code requires a trusted module obeying [`abi`]'s contract.

pub mod abi;
mod instance;
mod loader;
mod reflect_state;
mod system;

pub use reflect_state::{PreparedState, ReflectStateContract, ReflectStateError};
pub use system::{IsModuleSystem, ModuleSystem};

use std::collections::VecDeque;
use std::fmt;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};

use prism_ecs::resource::Resource;
use prism_ecs::world::{World, WorldId};

use instance::Instance;

use crate::{App, Plugin};

/// Application-owned operations available to module systems. Implementations
/// must not put library-owned pointers or drop glue into the World. An operation
/// failure does not roll back changes the operation already made. Do not reenter
/// this runtime from a host operation. A host panic is caught at the C boundary
/// and resumed after returning to Rust, poisoning the runtime instead of silently
/// continuing after an arbitrary partially completed World operation.
pub trait ModuleHost: Send + 'static {
    /// Dispatch an application operation and return a byte-encoded reply.
    fn call(&mut self, world: &mut World, operation: u64, input: &[u8]) -> Result<Vec<u8>, String>;
}

impl<F> ModuleHost for F
where
    F: FnMut(&mut World, u64, &[u8]) -> Result<Vec<u8>, String> + Send + 'static,
{
    fn call(&mut self, world: &mut World, operation: u64, input: &[u8]) -> Result<Vec<u8>, String> {
        self(world, operation, input)
    }
}

/// Expected identity and bounded I/O. Protocol IDs must change when application
/// payload layouts or operation semantics change incompatibly.
#[derive(Clone, Copy, Debug)]
pub struct ModuleReloadConfig {
    /// Expected nonzero application module ID.
    pub module_id: u64,
    /// Expected host operation protocol ID.
    pub host_contract: u64,
    /// Maximum library file size, checked before loading.
    pub max_library_bytes: usize,
    /// Maximum exported state size.
    pub max_state_bytes: usize,
    /// Maximum single host request/reply size.
    pub max_message_bytes: usize,
}

impl ModuleReloadConfig {
    /// Configure a module with 256 MiB library, 16 MiB state and 1 MiB message
    /// limits. Limits may be adjusted before constructing the reloader.
    pub fn new(module_id: u64, host_contract: u64) -> Self {
        Self {
            module_id,
            host_contract,
            max_library_bytes: 256 * 1024 * 1024,
            max_state_bytes: 16 * 1024 * 1024,
            max_message_bytes: 1024 * 1024,
        }
    }
}

/// Failure before activation, or a module execution failure. Text never owns
/// a plugin-defined error value or a borrowed pointer into an unloaded library.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ModuleReloadError {
    /// File access, shadow copy or native loader error.
    Load(String),
    /// An incompatible or incomplete descriptor.
    Abi(String),
    /// A callback returned a failure status.
    Callback {
        /// Callback operation that failed.
        stage: &'static str,
        /// Nonzero module/host ABI status.
        status: u32,
    },
    /// A configured byte limit was exceeded.
    Limit(&'static str),
    /// No module has successfully loaded yet.
    NotLoaded,
    /// The requested stable ID is not exported.
    UnknownSystem(u64),
    /// No automatic watcher has the requested module identity.
    UnknownModule(u64),
    /// Reload would remove or add schedule bindings.
    SystemSetChanged,
    /// A previous unexpected host unwind poisoned the runtime lock.
    Poisoned,
    /// Host operation rejected execution or panicked.
    Host(String),
    /// Host-owned typed reflect state could not be migrated or verified.
    Reflect(String),
}

impl fmt::Display for ModuleReloadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "module reload: {self:?}")
    }
}
impl std::error::Error for ModuleReloadError {}

/// Metadata copied into host storage; safe to retain after a reload.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModuleInfo {
    /// Stable module ID.
    pub module_id: u64,
    /// Module build version.
    pub module_version: u64,
    /// State schema exported by this build.
    pub state_schema: u64,
    /// Successful activation sequence, starting at one.
    pub generation: u64,
    /// Sorted stable IDs accepted by [`ModuleReload::system`].
    pub systems: Vec<u64>,
}

/// Result of checking the source library contents.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ModuleReloadOutcome {
    /// The source is byte-for-byte identical to the active build.
    Unchanged,
    /// First activation or a successful replacement.
    Activated(ModuleInfo),
}

/// A bounded execution error retained by a scheduled wrapper. Runtime errors
/// do not retire the host wrapper, so a later reload can repair a failed module.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModuleRunFailure {
    /// Stable system ID.
    pub system: u64,
    /// Active generation at the time of execution.
    pub generation: u64,
    /// Host-owned error detail.
    pub error: ModuleReloadError,
}

pub(super) struct Runtime {
    active: Option<Instance>,
    host: Box<dyn ModuleHost>,
    config: ModuleReloadConfig,
    world: Option<WorldId>,
    generation: u64,
    failures: VecDeque<ModuleRunFailure>,
}

impl Runtime {
    fn info(&self) -> Option<ModuleInfo> {
        self.active.as_ref().map(|instance| ModuleInfo {
            module_id: instance.api().module_id,
            module_version: instance.api().module_version,
            state_schema: instance.api().state_schema,
            generation: self.generation,
            systems: instance.systems().to_vec(),
        })
    }

    fn run(&mut self, system: u64, world: &mut World) -> Result<(), ModuleReloadError> {
        self.bind_world(world);
        let active = self.active.as_mut().ok_or(ModuleReloadError::NotLoaded)?;
        active.run(
            system,
            &mut *self.host,
            world,
            self.config.max_message_bytes,
        )
    }

    fn bind_world(&mut self, world: &World) {
        if let Some(id) = self.world {
            assert_eq!(
                id,
                world.id(),
                "module runtime cannot be shared between Worlds"
            );
        } else {
            self.world = Some(world.id());
        }
    }

    fn record_failure(&mut self, system: u64, error: ModuleReloadError) {
        if self.failures.len() == 128 {
            self.failures.pop_front();
        }
        self.failures.push_back(ModuleRunFailure {
            system,
            generation: self.generation,
            error,
        });
    }
}

/// A watched native module and the shared indirection used by schedule systems.
/// Poll only at a frame boundary, after pipelined/sub-app work has joined.
/// The runtime lock also prevents replacement during any in-flight invocation.
/// The source directory must be writable for owned sibling library snapshots;
/// this preserves dependencies resolved relative to the library directory.
pub struct ModuleReload {
    path: PathBuf,
    runtime: Arc<Mutex<Runtime>>,
    applied_bytes: Option<Vec<u8>>,
    reflect: Option<Arc<ReflectStateContract>>,
}

impl ModuleReload {
    /// Watch an absolute or relative source path. This does not load native code.
    pub fn new(
        path: impl Into<PathBuf>,
        config: ModuleReloadConfig,
        host: impl ModuleHost,
    ) -> Result<Self, ModuleReloadError> {
        if config.module_id == 0 || config.host_contract == 0 {
            return Err(ModuleReloadError::Abi(
                "module and host contract IDs must be nonzero".into(),
            ));
        }
        if config.max_library_bytes == 0
            || config.max_library_bytes > isize::MAX as usize
            || config.max_state_bytes > isize::MAX as usize
            || config.max_message_bytes > isize::MAX as usize
        {
            return Err(ModuleReloadError::Limit("invalid byte limits"));
        }
        Ok(Self {
            path: path.into(),
            runtime: Arc::new(Mutex::new(Runtime {
                active: None,
                host: Box::new(host),
                config,
                world: None,
                generation: 0,
                failures: VecDeque::new(),
            })),
            applied_bytes: None,
            reflect: None,
        })
    }

    /// Attach a host-owned typed reflect contract for state migration. The
    /// native ABI remains unchanged; adapters and migrations run before the
    /// candidate receives any state bytes.
    #[must_use]
    pub fn with_reflect_state(mut self, contract: ReflectStateContract) -> Self {
        self.reflect = Some(Arc::new(contract));
        self
    }

    /// The configured application module identity.
    #[must_use]
    pub fn module_id(&self) -> Result<u64, ModuleReloadError> {
        Ok(lock(&self.runtime)?.config.module_id)
    }

    /// Source library path, whose contents are polled rather than its mtime.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Copy active metadata without exposing any library symbols.
    pub fn active(&self) -> Result<Option<ModuleInfo>, ModuleReloadError> {
        Ok(lock(&self.runtime)?.info())
    }

    /// Check for a changed binary and transactionally activate it. Failed
    /// candidates are retried by the next poll; only successful bytes are cached.
    ///
    /// # Safety
    /// Every library published at `path`, including native constructors and
    /// dependencies, must be trusted code obeying the entire [`abi`] contract.
    /// Version checks cannot sandbox native code or verify callback behavior.
    pub unsafe fn poll(&mut self) -> Result<ModuleReloadOutcome, ModuleReloadError> {
        let config = lock(&self.runtime)?.config;
        let bytes = loader::read_binary(&self.path, config.max_library_bytes)?;
        if self.applied_bytes.as_ref() == Some(&bytes) {
            return Ok(ModuleReloadOutcome::Unchanged);
        }
        // SAFETY: caller trusts the source and all callbacks under the ABI contract.
        let candidate = unsafe { loader::LoadedModule::load(&self.path, &bytes, config)? };
        let mut runtime = lock(&self.runtime)?;
        if let Some(old) = &runtime.active {
            if old.systems() != candidate.systems() {
                return Err(ModuleReloadError::SystemSetChanged);
            }
        }
        // Export is read-only; candidate construction has no World capability.
        let previous = runtime
            .active
            .as_ref()
            .map(|old| old.export(config.max_state_bytes))
            .transpose()?;
        let previous_schema = runtime
            .active
            .as_ref()
            .map_or(0, |old| old.api().state_schema);
        let has_previous = runtime.active.is_some();
        let (create_schema, create_bytes, expected) =
            match (&self.reflect, runtime.active.as_ref(), previous.as_deref()) {
                (Some(contract), Some(_), Some(previous)) => {
                    let prepared = catch_unwind(AssertUnwindSafe(|| {
                        contract.prepare(previous_schema, candidate.api.state_schema, previous)
                    }))
                    .map_err(|_| ModuleReloadError::Reflect("reflect migration panicked".into()))?
                    .map_err(|error| ModuleReloadError::Reflect(error.to_string()))?;
                    (prepared.schema, prepared.payload, Some(prepared.expected))
                }
                _ => (previous_schema, previous.unwrap_or_default(), None),
            };
        if create_bytes.len() > config.max_state_bytes {
            return Err(ModuleReloadError::Limit("state"));
        }
        let replacement = Instance::create(candidate, create_schema, &create_bytes)?;
        if let Some(contract) = &self.reflect {
            let exported = replacement.export(config.max_state_bytes)?;
            if let Some(expected) = expected.as_deref() {
                catch_unwind(AssertUnwindSafe(|| {
                    contract.validate_import(create_schema, &exported, expected)
                }))
                .map_err(|_| {
                    ModuleReloadError::Reflect("reflect import validation panicked".into())
                })?
                .map_err(|error| ModuleReloadError::Reflect(error.to_string()))?;
            } else {
                catch_unwind(AssertUnwindSafe(|| {
                    contract.validate_export(
                        if has_previous {
                            create_schema
                        } else {
                            replacement.api().state_schema
                        },
                        &exported,
                    )
                }))
                .map_err(|_| {
                    ModuleReloadError::Reflect("reflect export validation panicked".into())
                })?
                .map_err(|error| ModuleReloadError::Reflect(error.to_string()))?;
            }
        }
        let old = runtime.active.replace(replacement);
        runtime.generation = runtime.generation.saturating_add(1);
        self.applied_bytes = Some(bytes);
        let info = runtime.info().expect("replacement was activated");
        // State is destroyed before its library. No call can hold a stale symbol.
        drop(old);
        Ok(ModuleReloadOutcome::Activated(info))
    }

    /// Create a named stable binding for insertion/configuration in a Schedule.
    /// Reloads must preserve the full ID set, so every existing binding remains
    /// valid. This binding pins the runtime even if the watcher is dropped.
    pub fn system(&self, id: u64) -> Result<ModuleSystem, ModuleReloadError> {
        let runtime = lock(&self.runtime)?;
        let active = runtime
            .active
            .as_ref()
            .ok_or(ModuleReloadError::NotLoaded)?;
        if active.systems().binary_search(&id).is_err() {
            return Err(ModuleReloadError::UnknownSystem(id));
        }
        Ok(ModuleSystem::new(id, Arc::clone(&self.runtime)))
    }

    /// Execute one system explicitly, using the same exclusive path as bindings.
    /// World writes made before an execution error are retained.
    pub fn run(&self, id: u64, world: &mut World) -> Result<(), ModuleReloadError> {
        lock(&self.runtime)?.run(id, world)
    }

    /// Export a read-only snapshot into host-owned bytes.
    pub fn export_state(&self) -> Result<Vec<u8>, ModuleReloadError> {
        let runtime = lock(&self.runtime)?;
        runtime
            .active
            .as_ref()
            .ok_or(ModuleReloadError::NotLoaded)?
            .export(runtime.config.max_state_bytes)
    }

    /// Drain the most recent (at most 128) scheduled execution failures.
    pub fn take_run_failures(&self) -> Result<Vec<ModuleRunFailure>, ModuleReloadError> {
        Ok(lock(&self.runtime)?.failures.drain(..).collect())
    }
}

/// A set of reload watchers polled by [`poll_app_modules`] at an App frame
/// boundary. Each watcher keeps its old instance active when its candidate
/// fails, so one broken module cannot stop the remaining watchers.
pub struct ModuleReloadRegistry {
    watchers: Vec<ModuleReload>,
    failures: VecDeque<ModulePollFailure>,
}

impl Resource for ModuleReloadRegistry {}

/// A host-owned report retained when automatic polling rejects a candidate.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModulePollFailure {
    /// Module identity configured for the watcher.
    pub module_id: u64,
    /// Source path being watched.
    pub path: PathBuf,
    /// The candidate rejection or loader failure.
    pub error: ModuleReloadError,
}

impl ModuleReloadRegistry {
    /// Create an empty polling registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            watchers: Vec::new(),
            failures: VecDeque::new(),
        }
    }

    /// Add a watcher, rejecting duplicate configured module identities.
    pub fn add(&mut self, watcher: ModuleReload) -> Result<(), ModuleReloadError> {
        let module_id = watcher.module_id()?;
        if self
            .watchers
            .iter()
            .filter_map(|watcher| watcher.module_id().ok())
            .any(|id| id == module_id)
        {
            return Err(ModuleReloadError::Abi(
                "duplicate module reload identity".into(),
            ));
        }
        self.watchers.push(watcher);
        Ok(())
    }

    /// Number of registered watchers.
    #[must_use]
    pub fn len(&self) -> usize {
        self.watchers.len()
    }

    /// Whether no watcher is registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.watchers.is_empty()
    }

    /// Read active metadata for one configured module identity.
    pub fn active(&self, module_id: u64) -> Result<Option<ModuleInfo>, ModuleReloadError> {
        let watcher = self
            .watchers
            .iter()
            .find(|watcher| watcher.module_id().ok() == Some(module_id))
            .ok_or(ModuleReloadError::UnknownModule(module_id))?;
        watcher.active()
    }

    /// Create a stable schedule binding for a system in a configured module.
    pub fn system(
        &self,
        module_id: u64,
        system_id: u64,
    ) -> Result<ModuleSystem, ModuleReloadError> {
        let watcher = self
            .watchers
            .iter()
            .find(|watcher| watcher.module_id().ok() == Some(module_id))
            .ok_or(ModuleReloadError::UnknownModule(module_id))?;
        watcher.system(system_id)
    }

    /// Remove and return all automatic poll failures.
    pub fn take_failures(&mut self) -> Vec<ModulePollFailure> {
        self.failures.drain(..).collect()
    }

    fn poll(&mut self) {
        for watcher in &mut self.watchers {
            // SAFETY: the registry is an explicit trusted-code installation
            // point; every watcher enforces its documented native ABI contract.
            let failure = match unsafe { watcher.poll() } {
                Ok(_) => None,
                Err(error) => Some((
                    watcher.module_id().unwrap_or(0),
                    watcher.path().to_owned(),
                    error,
                )),
            };
            if let Some((module_id, path, error)) = failure {
                if self.failures.len() == 128 {
                    self.failures.pop_front();
                }
                self.failures.push_back(ModulePollFailure {
                    module_id,
                    path,
                    error,
                });
            }
        }
    }
}

impl Default for ModuleReloadRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Explicit plugin that installs one or more trusted module watchers.
pub struct ModuleReloadPlugin {
    watchers: Mutex<Vec<ModuleReload>>,
}

impl ModuleReloadPlugin {
    /// Create a plugin with no watchers; callers may add them before assembly.
    #[must_use]
    pub fn new() -> Self {
        Self {
            watchers: Mutex::new(Vec::new()),
        }
    }

    /// Add a watcher to this plugin.
    #[must_use]
    pub fn with_watcher(mut self, watcher: ModuleReload) -> Self {
        self.watchers
            .get_mut()
            .expect("module reload plugin mutex is not poisoned")
            .push(watcher);
        self
    }
}

impl Default for ModuleReloadPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl Plugin for ModuleReloadPlugin {
    fn build(&self, app: &mut App) {
        let watchers = self
            .watchers
            .lock()
            .expect("module reload plugin mutex is not poisoned")
            .drain(..)
            .collect::<Vec<_>>();
        for watcher in watchers {
            app.add_module_reload(watcher)
                .expect("module reload plugin contains a duplicate watcher");
        }
    }

    fn is_unique(&self) -> bool {
        false
    }
}

impl App {
    /// Install a watcher into the automatic frame-boundary polling registry.
    pub fn add_module_reload(
        &mut self,
        watcher: ModuleReload,
    ) -> Result<&mut Self, ModuleReloadError> {
        if self
            .world()
            .get_resource::<ModuleReloadRegistry>()
            .is_none()
        {
            self.insert_resource(ModuleReloadRegistry::new());
        }
        self.world_mut()
            .resource_mut::<ModuleReloadRegistry>()
            .add(watcher)?;
        Ok(self)
    }
}

/// Poll installed module watchers before the next frame's systems run.
///
/// This is `pub(crate)` so the App loop and PIE/stepping integration share one
/// boundary hook without exposing an unsafe native-load operation as a public
/// free function. No sync is performed when no watcher is installed.
pub(crate) fn poll_app_modules(app: &mut App) {
    let has_watchers = app
        .world()
        .get_resource::<ModuleReloadRegistry>()
        .is_some_and(|registry| !registry.is_empty());
    if !has_watchers {
        return;
    }
    app.sync_sub_apps();
    app.world_mut()
        .resource_mut::<ModuleReloadRegistry>()
        .poll();
}

fn lock(runtime: &Mutex<Runtime>) -> Result<MutexGuard<'_, Runtime>, ModuleReloadError> {
    runtime.lock().map_err(|_| ModuleReloadError::Poisoned)
}

#[cfg(test)]
mod tests;
