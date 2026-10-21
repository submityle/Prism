//! Durable, versioned Scene Project bootstrap for the native editor.
//!
//! A project owns one generic PRISMDOC scene cell. Creation publishes the
//! project manifest only after revision zero of that cell is durable. Opening
//! fully validates and installs the document before the native window starts.

use std::{
    fmt,
    fs::{self, File, OpenOptions},
    io::{self, Write},
    path::{Component, Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    task::{Context, Poll, Wake, Waker},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use prism_asset::StableGuid;
use prism_ecs::partition::cell::CellCoord;
use prism_editor_content::ContentCatalog;
use prism_editor_world::{
    CellLoadRequest, EditorWorld, FileCellIoError, FileCellIoLimits, FileCellIoProvider, IoFailure,
    IoRequestId, ResidencyDemand, ValidationError, WorldError,
};

use crate::{
    ControllerLimits, DesktopDocumentSession, DesktopUiError, DocumentError, DocumentFileAdapter,
    DocumentIoEvent, DocumentLimits, DocumentSchemaCatalog, EditorController, EditorDesktopUi,
};

const MANIFEST_NAME: &str = "prism.project";
const SCENE_DIRECTORY: &str = "scene";
const MANIFEST_MAGIC: &[u8; 8] = b"PRISMPRJ";
const MANIFEST_VERSION: u32 = 1;
const MANIFEST_BYTES: usize = 8 + 4 + 16 + 12;
const SCENE_CELL: CellCoord = CellCoord::new(0, 0, 0);
const MAX_STAGE_ATTEMPTS: u64 = 64;

static NEXT_STAGE: AtomicU64 = AtomicU64::new(1);
static NEXT_PROJECT: AtomicU64 = AtomicU64::new(1);

/// Resource policy for project bootstrap and untrusted project files.
#[derive(Clone, Copy, Debug)]
pub struct SceneProjectConfig {
    /// PRISMDOC and reflection bounds.
    pub document_limits: DocumentLimits,
    /// Installed controller bounds.
    pub controller_limits: ControllerLimits,
    /// Maximum wait for one bootstrap filesystem operation.
    pub io_timeout: Duration,
}

impl Default for SceneProjectConfig {
    fn default() -> Self {
        Self {
            document_limits: DocumentLimits::default(),
            controller_limits: ControllerLimits::default(),
            io_timeout: Duration::from_secs(10),
        }
    }
}

impl SceneProjectConfig {
    fn validate(self) -> Result<Self, SceneProjectError> {
        if self.io_timeout.is_zero() {
            return Err(SceneProjectError::InvalidConfiguration);
        }
        Ok(self)
    }
}

/// A completely bootstrapped project ready for `run_desktop_with_document`.
pub struct SceneProjectOpen {
    root: PathBuf,
    project_guid: StableGuid,
    ui: EditorDesktopUi,
    document: DesktopDocumentSession,
}

impl SceneProjectOpen {
    /// Canonical project root.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Stable identity persisted in the manifest.
    #[must_use]
    pub const fn project_guid(&self) -> StableGuid {
        self.project_guid
    }

    /// Loaded retained UI.
    #[must_use]
    pub const fn ui(&self) -> &EditorDesktopUi {
        &self.ui
    }

    /// Loaded document session.
    #[must_use]
    pub const fn document(&self) -> &DesktopDocumentSession {
        &self.document
    }

    /// Loaded document session.
    #[must_use]
    pub const fn document_session(&self) -> &DesktopDocumentSession {
        &self.document
    }

    /// Transfers the initialized UI and document session to the native host.
    #[must_use]
    pub fn into_desktop(self) -> (EditorDesktopUi, DesktopDocumentSession) {
        (self.ui, self.document)
    }
}

/// Versioned Scene Project creation and opening.
pub struct SceneProject;

impl SceneProject {
    /// Creates a project with a generic empty schema catalog.
    pub fn create(path: impl AsRef<Path>) -> Result<SceneProjectOpen, SceneProjectError> {
        Self::create_with(
            path,
            DocumentSchemaCatalog::new(),
            SceneProjectConfig::default(),
        )
    }

    /// Creates a project and durably saves its initial empty scene before
    /// publishing the manifest.
    pub fn create_with(
        path: impl AsRef<Path>,
        catalog: DocumentSchemaCatalog,
        config: SceneProjectConfig,
    ) -> Result<SceneProjectOpen, SceneProjectError> {
        let config = config.validate()?;
        let (root, created_root) = prepare_new_root(path.as_ref())?;
        let scene_root = root.join(SCENE_DIRECTORY);
        let result = (|| {
            fs::create_dir(&scene_root).map_err(|source| SceneProjectError::Io {
                operation: "create scene directory",
                source,
            })?;
            let project_guid = generate_project_guid(&root)?;
            if project_guid.is_nil() {
                return Err(SceneProjectError::InvalidManifest("nil project identity"));
            }
            let (mut adapter, mut controller) =
                new_adapter(&scene_root, project_guid, config.document_limits)?;
            let request = adapter.submit_save(&controller, &catalog)?;
            let event = wait_for_event(&mut adapter, request, config.io_timeout)?;
            match event {
                DocumentIoEvent::Saved(receipt) => {
                    receipt.acknowledge(&mut controller, &catalog)?;
                }
                DocumentIoEvent::Failed { error, .. } => {
                    return Err(SceneProjectError::DocumentIo(error));
                }
                DocumentIoEvent::Cancelled { .. } => {
                    return Err(SceneProjectError::Cancelled);
                }
                DocumentIoEvent::Loaded(_) => {
                    return Err(SceneProjectError::Protocol("save completed as load"));
                }
            }
            publish_manifest(&root, project_guid)?;
            build_open(
                root.clone(),
                project_guid,
                controller,
                adapter,
                catalog,
                config,
            )
        })();
        if result.is_err() {
            cleanup_failed_create(&root, created_root);
        }
        result
    }

    /// Opens and fully installs an existing project before returning.
    pub fn open(path: impl AsRef<Path>) -> Result<SceneProjectOpen, SceneProjectError> {
        Self::open_with(
            path,
            DocumentSchemaCatalog::new(),
            SceneProjectConfig::default(),
        )
    }

    /// Opens a project with a caller-owned schema catalog.
    pub fn open_with(
        path: impl AsRef<Path>,
        catalog: DocumentSchemaCatalog,
        config: SceneProjectConfig,
    ) -> Result<SceneProjectOpen, SceneProjectError> {
        let config = config.validate()?;
        let root = validate_existing_directory(path.as_ref(), "project root")?;
        let manifest_path = root.join(MANIFEST_NAME);
        validate_regular_file(&manifest_path, "project manifest")?;
        let manifest = fs::read(&manifest_path).map_err(|source| SceneProjectError::Io {
            operation: "read project manifest",
            source,
        })?;
        let project_guid = decode_manifest(&manifest)?;
        let scene_root = validate_existing_directory(&root.join(SCENE_DIRECTORY), "scene root")?;
        if scene_root.parent() != Some(root.as_path()) {
            return Err(SceneProjectError::UnsafePath(scene_root));
        }
        let (mut adapter, mut controller) =
            new_adapter(&scene_root, project_guid, config.document_limits)?;
        let request = adapter.submit_load(&controller, &catalog)?;
        let event = wait_for_event(&mut adapter, request, config.io_timeout)?;
        match event {
            DocumentIoEvent::Loaded(receipt) => {
                let prepared = receipt.prepare(&catalog, config.controller_limits)?;
                prepared.install(&mut adapter, &mut controller, &catalog)?;
            }
            DocumentIoEvent::Failed { error, .. } => {
                return Err(SceneProjectError::DocumentIo(error));
            }
            DocumentIoEvent::Cancelled { .. } => return Err(SceneProjectError::Cancelled),
            DocumentIoEvent::Saved(_) => {
                return Err(SceneProjectError::Protocol("load completed as save"));
            }
        }
        build_open(root, project_guid, controller, adapter, catalog, config)
    }
}

fn build_open(
    root: PathBuf,
    project_guid: StableGuid,
    controller: EditorController,
    adapter: DocumentFileAdapter,
    catalog: DocumentSchemaCatalog,
    config: SceneProjectConfig,
) -> Result<SceneProjectOpen, SceneProjectError> {
    let mut ui = EditorDesktopUi::new(controller)?;
    ui.mount_content_browser(ContentCatalog::new())?;
    Ok(SceneProjectOpen {
        root,
        project_guid,
        ui,
        document: DesktopDocumentSession::new(adapter, catalog, config.controller_limits),
    })
}

fn new_adapter(
    scene_root: &Path,
    project_guid: StableGuid,
    limits: DocumentLimits,
) -> Result<(DocumentFileAdapter, EditorController), SceneProjectError> {
    let provider_limits = FileCellIoLimits::new(limits.max_document_bytes, 1)?;
    let provider = FileCellIoProvider::new(scene_root, provider_limits)?;
    let mut world = EditorWorld::new(project_guid)?;
    world.add_cell(SCENE_CELL, [])?;
    let file = world
        .cell(SCENE_CELL)
        .ok_or(SceneProjectError::Protocol("scene cell was not registered"))?
        .file()
        .clone();
    let mut demand = ResidencyDemand::new();
    demand.require(SCENE_CELL);
    let plan = world.plan(&demand)?;
    let tokens = world.begin_plan(&plan)?;
    let token = tokens.first().copied().ok_or(SceneProjectError::Protocol(
        "scene load token was not created",
    ))?;
    let request = CellLoadRequest::new(token, file, limits.max_document_bytes);
    let adapter = DocumentFileAdapter::new(provider, request, limits)?;
    Ok((adapter, EditorController::new()))
}

struct ThreadWake(thread::Thread);

impl Wake for ThreadWake {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
}

fn wait_for_event(
    adapter: &mut DocumentFileAdapter,
    request: IoRequestId,
    timeout: Duration,
) -> Result<DocumentIoEvent, SceneProjectError> {
    let waker = Waker::from(Arc::new(ThreadWake(thread::current())));
    let mut context = Context::from_waker(&waker);
    let deadline = Instant::now()
        .checked_add(timeout)
        .ok_or(SceneProjectError::InvalidConfiguration)?;
    loop {
        match adapter.poll(&mut context) {
            Poll::Ready(Some(event)) => return Ok(event),
            Poll::Ready(None) => return Err(SceneProjectError::Protocol("operation disappeared")),
            Poll::Pending => {}
        }
        let now = Instant::now();
        if now >= deadline {
            let _ = adapter.cancel(request);
            match adapter.poll(&mut context) {
                Poll::Ready(Some(DocumentIoEvent::Cancelled { .. })) | Poll::Ready(None) => {}
                Poll::Ready(Some(_)) | Poll::Pending => {
                    return Err(SceneProjectError::Protocol(
                        "timed-out operation did not terminate after cancellation",
                    ));
                }
            }
            return Err(SceneProjectError::Timeout(timeout));
        }
        thread::park_timeout((deadline - now).min(Duration::from_millis(5)));
    }
}

fn prepare_new_root(path: &Path) -> Result<(PathBuf, bool), SceneProjectError> {
    reject_parent_components(path)?;
    match fs::symlink_metadata(path) {
        Ok(metadata) => {
            if metadata.file_type().is_symlink() || !metadata.is_dir() {
                return Err(SceneProjectError::UnsafePath(path.to_owned()));
            }
            if fs::read_dir(path)
                .map_err(|source| SceneProjectError::Io {
                    operation: "inspect project directory",
                    source,
                })?
                .next()
                .transpose()
                .map_err(|source| SceneProjectError::Io {
                    operation: "inspect project directory",
                    source,
                })?
                .is_some()
            {
                return Err(SceneProjectError::ProjectNotEmpty(path.to_owned()));
            }
            Ok((canonicalize(path, "resolve project directory")?, false))
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            let parent = path
                .parent()
                .filter(|parent| !parent.as_os_str().is_empty())
                .unwrap_or_else(|| Path::new("."));
            validate_existing_directory(parent, "project parent")?;
            fs::create_dir(path).map_err(|source| SceneProjectError::Io {
                operation: "create project directory",
                source,
            })?;
            Ok((
                canonicalize(path, "resolve created project directory")?,
                true,
            ))
        }
        Err(source) => Err(SceneProjectError::Io {
            operation: "inspect project directory",
            source,
        }),
    }
}

fn generate_project_guid(root: &Path) -> Result<StableGuid, SceneProjectError> {
    let elapsed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| SceneProjectError::InvalidConfiguration)?;
    let sequence = NEXT_PROJECT.fetch_add(1, Ordering::Relaxed);
    let seed = format!(
        "prism-project\0{}\0{}\0{}\0{}",
        root.display(),
        std::process::id(),
        elapsed.as_nanos(),
        sequence
    );
    let guid = StableGuid::from_content(seed.as_bytes());
    if guid.is_nil() {
        return Err(SceneProjectError::InvalidManifest("nil project identity"));
    }
    Ok(guid)
}

fn validate_existing_directory(
    path: &Path,
    operation: &'static str,
) -> Result<PathBuf, SceneProjectError> {
    reject_parent_components(path)?;
    let metadata =
        fs::symlink_metadata(path).map_err(|source| SceneProjectError::Io { operation, source })?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(SceneProjectError::UnsafePath(path.to_owned()));
    }
    canonicalize(path, operation)
}

fn validate_regular_file(path: &Path, operation: &'static str) -> Result<(), SceneProjectError> {
    let metadata =
        fs::symlink_metadata(path).map_err(|source| SceneProjectError::Io { operation, source })?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(SceneProjectError::UnsafePath(path.to_owned()));
    }
    Ok(())
}

fn canonicalize(path: &Path, operation: &'static str) -> Result<PathBuf, SceneProjectError> {
    fs::canonicalize(path).map_err(|source| SceneProjectError::Io { operation, source })
}

fn reject_parent_components(path: &Path) -> Result<(), SceneProjectError> {
    if path
        .components()
        .any(|component| component == Component::ParentDir)
    {
        return Err(SceneProjectError::UnsafePath(path.to_owned()));
    }
    Ok(())
}

fn encode_manifest(project_guid: StableGuid) -> [u8; MANIFEST_BYTES] {
    let mut bytes = [0_u8; MANIFEST_BYTES];
    bytes[..8].copy_from_slice(MANIFEST_MAGIC);
    bytes[8..12].copy_from_slice(&MANIFEST_VERSION.to_le_bytes());
    bytes[12..28].copy_from_slice(&project_guid.to_u128().to_le_bytes());
    bytes[28..32].copy_from_slice(&SCENE_CELL.x.to_le_bytes());
    bytes[32..36].copy_from_slice(&SCENE_CELL.y.to_le_bytes());
    bytes[36..40].copy_from_slice(&SCENE_CELL.z.to_le_bytes());
    bytes
}

fn decode_manifest(bytes: &[u8]) -> Result<StableGuid, SceneProjectError> {
    if bytes.len() != MANIFEST_BYTES {
        return Err(SceneProjectError::InvalidManifest("invalid exact length"));
    }
    if &bytes[..8] != MANIFEST_MAGIC {
        return Err(SceneProjectError::InvalidManifest("invalid magic"));
    }
    let version = u32::from_le_bytes(
        bytes[8..12]
            .try_into()
            .map_err(|_| SceneProjectError::InvalidManifest("missing version"))?,
    );
    if version != MANIFEST_VERSION {
        return Err(SceneProjectError::UnsupportedVersion(version));
    }
    let guid =
        StableGuid::from_u128(u128::from_le_bytes(bytes[12..28].try_into().map_err(
            |_| SceneProjectError::InvalidManifest("missing project identity"),
        )?));
    if guid.is_nil() {
        return Err(SceneProjectError::InvalidManifest("nil project identity"));
    }
    let x = i32::from_le_bytes(
        bytes[28..32]
            .try_into()
            .map_err(|_| SceneProjectError::InvalidManifest("missing scene coordinate"))?,
    );
    let y = i32::from_le_bytes(
        bytes[32..36]
            .try_into()
            .map_err(|_| SceneProjectError::InvalidManifest("missing scene coordinate"))?,
    );
    let z = i32::from_le_bytes(
        bytes[36..40]
            .try_into()
            .map_err(|_| SceneProjectError::InvalidManifest("missing scene coordinate"))?,
    );
    if CellCoord::new(x, y, z) != SCENE_CELL {
        return Err(SceneProjectError::InvalidManifest(
            "unsupported scene layout",
        ));
    }
    Ok(guid)
}

fn publish_manifest(root: &Path, project_guid: StableGuid) -> Result<(), SceneProjectError> {
    let destination = root.join(MANIFEST_NAME);
    if fs::symlink_metadata(&destination).is_ok() {
        return Err(SceneProjectError::ProjectExists(destination));
    }
    let bytes = encode_manifest(project_guid);
    for _ in 0..MAX_STAGE_ATTEMPTS {
        let sequence = NEXT_STAGE.fetch_add(1, Ordering::Relaxed);
        let stage = root.join(format!(
            ".prism.project.{}.{}.stage",
            std::process::id(),
            sequence
        ));
        let mut file = match OpenOptions::new().write(true).create_new(true).open(&stage) {
            Ok(file) => file,
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(source) => {
                return Err(SceneProjectError::Io {
                    operation: "create project manifest staging file",
                    source,
                });
            }
        };
        let result = (|| {
            file.write_all(&bytes)
                .map_err(|source| SceneProjectError::Io {
                    operation: "write project manifest",
                    source,
                })?;
            file.sync_all().map_err(|source| SceneProjectError::Io {
                operation: "sync project manifest",
                source,
            })?;
            drop(file);
            fs::hard_link(&stage, &destination).map_err(|source| {
                if source.kind() == io::ErrorKind::AlreadyExists {
                    SceneProjectError::ProjectExists(destination.clone())
                } else {
                    SceneProjectError::Io {
                        operation: "publish project manifest",
                        source,
                    }
                }
            })?;
            fs::remove_file(&stage).map_err(|source| SceneProjectError::Io {
                operation: "remove manifest staging link",
                source,
            })?;
            File::open(root)
                .and_then(|directory| directory.sync_all())
                .map_err(|source| SceneProjectError::Io {
                    operation: "sync project directory",
                    source,
                })
        })();
        if result.is_err() {
            let _ = fs::remove_file(&stage);
        }
        return result;
    }
    Err(SceneProjectError::StageNameExhausted)
}

fn cleanup_failed_create(root: &Path, created_root: bool) {
    let _ = fs::remove_file(root.join(MANIFEST_NAME));
    let _ = fs::remove_dir_all(root.join(SCENE_DIRECTORY));
    if created_root {
        let _ = fs::remove_dir(root);
    }
}

/// Failure to create, validate, load or initialize a Scene Project.
#[derive(Debug)]
pub enum SceneProjectError {
    /// Project limits or timeout are invalid.
    InvalidConfiguration,
    /// A path contains traversal, a symlink, or an unexpected file type.
    UnsafePath(PathBuf),
    /// Creation would overwrite an existing published project.
    ProjectExists(PathBuf),
    /// Creation only accepts a new or empty directory.
    ProjectNotEmpty(PathBuf),
    /// Manifest framing or values are invalid.
    InvalidManifest(&'static str),
    /// Manifest version is not supported by this editor.
    UnsupportedVersion(u32),
    /// A bounded bootstrap operation exceeded its deadline.
    Timeout(Duration),
    /// An accepted bootstrap operation was cancelled.
    Cancelled,
    /// An internal operation completed with the wrong protocol event.
    Protocol(&'static str),
    /// All bounded staging names collided.
    StageNameExhausted,
    /// Filesystem setup or manifest I/O failed.
    Io {
        /// Operation being attempted.
        operation: &'static str,
        /// Underlying operating-system error.
        source: io::Error,
    },
    /// Cell provider setup failed.
    Storage(FileCellIoError),
    /// World identity or catalog validation failed.
    Validation(ValidationError),
    /// World planning or transition failed.
    World(WorldError),
    /// Document codec or adapter operation failed.
    Document(DocumentError),
    /// Accepted document filesystem work failed.
    DocumentIo(IoFailure<FileCellIoError>),
    /// Retained desktop UI construction failed.
    Ui(DesktopUiError),
}

impl fmt::Display for SceneProjectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfiguration => f.write_str("invalid Scene Project configuration"),
            Self::UnsafePath(path) => write!(f, "unsafe Scene Project path: {}", path.display()),
            Self::ProjectExists(path) => {
                write!(f, "Scene Project already exists: {}", path.display())
            }
            Self::ProjectNotEmpty(path) => write!(
                f,
                "Scene Project directory is not empty: {}",
                path.display()
            ),
            Self::InvalidManifest(reason) => write!(f, "invalid Scene Project manifest: {reason}"),
            Self::UnsupportedVersion(version) => {
                write!(f, "unsupported Scene Project version {version}")
            }
            Self::Timeout(timeout) => write!(f, "Scene Project I/O exceeded {timeout:?}"),
            Self::Cancelled => f.write_str("Scene Project I/O was cancelled"),
            Self::Protocol(reason) => write!(f, "Scene Project protocol failure: {reason}"),
            Self::StageNameExhausted => f.write_str("could not allocate a manifest staging file"),
            Self::Io { operation, source } => write!(f, "{operation}: {source}"),
            Self::Storage(error) => error.fmt(f),
            Self::Validation(error) => error.fmt(f),
            Self::World(error) => error.fmt(f),
            Self::Document(error) => error.fmt(f),
            Self::DocumentIo(error) => write!(f, "Scene Project document I/O: {error}"),
            Self::Ui(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for SceneProjectError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Storage(error) => Some(error),
            Self::Validation(error) => Some(error),
            Self::World(error) => Some(error),
            Self::Document(error) => Some(error),
            Self::Ui(error) => Some(error),
            _ => None,
        }
    }
}

impl From<FileCellIoError> for SceneProjectError {
    fn from(value: FileCellIoError) -> Self {
        Self::Storage(value)
    }
}

impl From<ValidationError> for SceneProjectError {
    fn from(value: ValidationError) -> Self {
        Self::Validation(value)
    }
}

impl From<WorldError> for SceneProjectError {
    fn from(value: WorldError) -> Self {
        Self::World(value)
    }
}

impl From<DocumentError> for SceneProjectError {
    fn from(value: DocumentError) -> Self {
        Self::Document(value)
    }
}

impl From<DesktopUiError> for SceneProjectError {
    fn from(value: DesktopUiError) -> Self {
        Self::Ui(value)
    }
}
