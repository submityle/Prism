//! Deterministic, renderer-independent content browser data contracts.
//!
//! The crate owns bounded catalog indexes, virtualized tree and search pages,
//! dependency diagnostics, import/export task state, and version-control
//! metadata. File-system, importer, exporter, and VCS adapters can feed these
//! contracts without introducing a runtime or renderer dependency.

#![forbid(unsafe_code)]

extern crate alloc;

use alloc::{
    collections::{BTreeMap, BTreeSet},
    string::{String, ToString},
    sync::Arc,
    vec::Vec,
};
use core::{fmt, ops::Bound};

pub mod filesystem;
mod source;
pub mod validation;
pub use source::VcsSnapshotLimits;

/// Exact catalog identity captured before asynchronous source or VCS work.
/// Private identity prevents construction from an unrelated numeric generation.
#[derive(Clone, Debug)]
pub struct ContentSnapshotToken(Arc<()>);

/// Stable content identity supplied by an asset system or importer.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct AssetId(u128);

impl AssetId {
    /// Creates an identity from a stable raw value.
    #[must_use]
    pub const fn from_raw(value: u128) -> Self {
        Self(value)
    }

    /// Returns the raw stable identity.
    #[must_use]
    pub const fn raw(self) -> u128 {
        self.0
    }
}

/// Stable content task identity.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct ContentTaskId(u64);

impl ContentTaskId {
    /// Returns the numeric identity.
    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0
    }
}

/// Asset category used for deterministic filtering and display.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct AssetKind(String);

impl AssetKind {
    /// Creates a non-empty category.
    pub fn new(value: impl Into<String>) -> Result<Self, ContentError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(ContentError::EmptyValue("asset kind"));
        }
        Ok(Self(value))
    }

    /// Returns the category string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Version-control state for one canonical path.
#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub enum VcsState {
    /// No provider has supplied a state.
    #[default]
    Unknown,
    /// The path matches the recorded revision.
    Clean,
    /// The path is not tracked by the provider.
    Untracked,
    /// The path is scheduled for addition.
    Added,
    /// The path has local modifications.
    Modified,
    /// The path is scheduled for deletion.
    Deleted,
    /// The path was renamed by the provider.
    Renamed,
    /// The path is intentionally ignored.
    Ignored,
    /// The path has an unresolved merge or checkout conflict.
    Conflicted,
}

/// Version-control metadata attached to a canonical path.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct VcsRecord {
    /// Provider state.
    pub state: VcsState,
    /// Whether the provider reports the change as staged.
    pub staged: bool,
    /// Provider revision or object identifier.
    pub revision: Option<String>,
    /// Optional lock owner reported by the provider.
    pub lock_owner: Option<String>,
}

/// Import/export task lifecycle.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TaskState {
    /// Waiting for an adapter slot.
    Queued,
    /// Adapter work is in progress.
    Running,
    /// Adapter completed successfully.
    Succeeded,
    /// Adapter reported a failure.
    Failed,
    /// Work was cancelled before completion.
    Cancelled,
}

impl TaskState {
    fn is_terminal(self) -> bool {
        matches!(self, Self::Succeeded | Self::Failed | Self::Cancelled)
    }
}

/// Import or export operation represented by a task.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TaskKind {
    /// Import one asset from its source representation.
    Import {
        /// Asset being imported.
        asset: AssetId,
    },
    /// Export a bounded selection of assets.
    Export {
        /// Assets included in the export operation.
        assets: Vec<AssetId>,
    },
}

/// Snapshot of one import/export task.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ContentTask {
    /// Stable task identity.
    pub id: ContentTaskId,
    /// Operation represented by this task.
    pub kind: TaskKind,
    /// Current lifecycle state.
    pub state: TaskState,
    /// Progress in thousandths, from zero through one thousand.
    pub progress_milli: u16,
    /// Optional adapter message.
    pub message: Option<String>,
}

/// Catalog record for one asset.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AssetRecord {
    /// Stable identity.
    pub id: AssetId,
    /// Canonical project-relative path.
    pub path: String,
    /// Display/import category.
    pub kind: AssetKind,
    /// Searchable tags.
    pub tags: BTreeSet<String>,
    /// Direct dependency identities. Missing identities are retained for diagnostics.
    pub dependencies: BTreeSet<AssetId>,
    /// Whether the user marked the asset as a favorite.
    pub favorite: bool,
    /// Whether the asset is pinned in the content browser.
    pub pinned: bool,
}

impl AssetRecord {
    /// Creates a record after canonical path and kind validation.
    pub fn new(
        id: AssetId,
        path: impl Into<String>,
        kind: AssetKind,
    ) -> Result<Self, ContentError> {
        Ok(Self {
            id,
            path: canonical_path(path.into())?,
            kind,
            tags: BTreeSet::new(),
            dependencies: BTreeSet::new(),
            favorite: false,
            pinned: false,
        })
    }

    /// Adds a normalized search tag.
    pub fn add_tag(&mut self, tag: impl Into<String>) -> Result<(), ContentError> {
        self.tags
            .insert(normalize_required(tag.into(), "asset tag")?);
        Ok(())
    }
}

/// Bounded memory and traversal policy for one catalog.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ContentLimits {
    /// Maximum records returned by one page request.
    pub max_page_size: usize,
    /// Maximum number of cached pages across tree and asset queries.
    pub max_cache_entries: usize,
    /// Maximum number of cached rows across all cached pages.
    pub max_cache_items: usize,
    /// Maximum retained import/export task snapshots.
    pub max_task_history: usize,
    /// Maximum assets in one export task.
    pub max_export_assets: usize,
    /// Maximum diagnostics retained in one report.
    pub max_diagnostics: usize,
    /// Maximum assets returned by a dependency walk.
    pub max_dependency_nodes: usize,
    /// Maximum dependency depth inspected by validation and walks; roots have depth zero.
    pub max_dependency_depth: usize,
}

impl Default for ContentLimits {
    fn default() -> Self {
        Self {
            max_page_size: 256,
            max_cache_entries: 64,
            max_cache_items: 4_096,
            max_task_history: 512,
            max_export_assets: 4_096,
            max_diagnostics: 1_024,
            max_dependency_nodes: 4_096,
            max_dependency_depth: 64,
        }
    }
}

impl ContentLimits {
    fn validate(self) -> Result<Self, ContentError> {
        if self.max_page_size == 0
            || self.max_cache_entries == 0
            || self.max_cache_items == 0
            || self.max_task_history == 0
            || self.max_export_assets == 0
            || self.max_diagnostics == 0
            || self.max_dependency_nodes == 0
            || self.max_dependency_depth == 0
        {
            return Err(ContentError::InvalidLimits);
        }
        Ok(self)
    }
}

/// Bounds on one complete asset refresh, applied before building its indexes.
/// Byte accounting covers UTF-8 path/kind/tag text and 16 bytes per identity or
/// dependency. It is a logical payload bound, not an allocator heap measurement.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CatalogSnapshotLimits {
    /// Maximum records in the replacement catalog.
    pub max_assets: usize,
    /// Maximum UTF-8 bytes in each asset path.
    pub max_path_bytes: usize,
    /// Maximum slash-separated components per path, bounding tree expansion.
    pub max_path_depth: usize,
    /// Maximum dependency edges across all records, including missing targets.
    pub max_dependencies: usize,
    /// Maximum tags across all records.
    pub max_tags: usize,
    /// Maximum logical record payload across the entire replacement.
    pub max_payload_bytes: usize,
}

impl Default for CatalogSnapshotLimits {
    fn default() -> Self {
        Self {
            max_assets: 100_000,
            max_path_bytes: 4096,
            max_path_depth: 64,
            max_dependencies: 1_000_000,
            max_tags: 1_000_000,
            max_payload_bytes: 64 * 1024 * 1024,
        }
    }
}

/// A bounded page request shared by tree and asset queries.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PageRequest {
    /// Number of rows requested.
    pub limit: usize,
}

impl PageRequest {
    /// Creates a page request. The catalog validates it against its limits.
    #[must_use]
    pub const fn new(limit: usize) -> Self {
        Self { limit }
    }
}

/// Cursor for a catalog asset query.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AssetCursor {
    generation: u64,
    fingerprint: u64,
    offset: usize,
    after: AssetId,
}

impl AssetCursor {
    /// Returns the number of matching rows skipped by this cursor.
    #[must_use]
    pub const fn offset(self) -> usize {
        self.offset
    }
}

/// Cursor for an immediate-child asset-tree query.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TreeCursor {
    generation: u64,
    fingerprint: u64,
    offset: usize,
    after: String,
}

impl TreeCursor {
    /// Returns the number of matching children skipped by this cursor.
    #[must_use]
    pub const fn offset(&self) -> usize {
        self.offset
    }
}

/// Result of a virtualized asset query.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AssetPage {
    /// Asset identities in canonical path order.
    pub items: Vec<AssetId>,
    /// Cursor for the next page, if more matching rows exist.
    pub next: Option<AssetCursor>,
    /// Whether this result came from the bounded page cache.
    pub cache: CacheStatus,
}

/// Result of a virtualized immediate-child tree query.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TreePage {
    /// Tree entries in canonical name/path order.
    pub items: Vec<TreeEntry>,
    /// Cursor for the next page, if more matching children exist.
    pub next: Option<TreeCursor>,
    /// Whether this result came from the bounded page cache.
    pub cache: CacheStatus,
}

/// Cache result for one page lookup.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CacheStatus {
    /// The page was computed from catalog indexes.
    Miss,
    /// The page was served by the bounded cache.
    Hit,
}

/// Aggregate page-cache counters and occupancy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CacheStats {
    /// Number of cached pages.
    pub entries: usize,
    /// Number of cached rows.
    pub items: usize,
    /// Number of cache hits since construction.
    pub hits: u64,
    /// Number of cache misses since construction.
    pub misses: u64,
}

/// Query filters for the flat asset view.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct AssetQuery {
    /// Optional project-relative root; descendants are included.
    pub root: Option<String>,
    /// Case-insensitive substring searched in path, kind, and tags.
    pub text: Option<String>,
    /// Exact category filter.
    pub kind: Option<AssetKind>,
    /// Required normalized tag.
    pub tag: Option<String>,
    /// Include only favorites.
    pub favorites_only: bool,
    /// Include only pinned assets.
    pub pinned_only: bool,
    /// Optional VCS state filter.
    pub vcs: Option<VcsState>,
}

/// Query filters for immediate children of one asset-tree folder.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TreeQuery {
    /// Project-relative folder path. An empty path means the project root.
    pub root: String,
    /// Case-insensitive substring searched in path, kind, and tags.
    pub text: Option<String>,
    /// Exact category filter for descendant assets.
    pub kind: Option<AssetKind>,
    /// Required normalized tag for descendant assets.
    pub tag: Option<String>,
    /// Include only favorites.
    pub favorites_only: bool,
    /// Include only pinned assets.
    pub pinned_only: bool,
    /// Optional VCS state filter for descendant assets.
    pub vcs: Option<VcsState>,
}

/// One node returned by an asset-tree query.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TreeEntry {
    /// Canonical path of this folder or asset.
    pub path: String,
    /// Name relative to the queried parent folder.
    pub name: String,
    /// Whether this entry is a folder or an asset.
    pub kind: TreeEntryKind,
    /// Asset identity for an asset node.
    pub asset: Option<AssetId>,
    /// Number of catalog assets below this node.
    pub descendant_assets: usize,
}

/// Kind of an asset-tree node.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TreeEntryKind {
    /// A folder with one or more catalog descendants.
    Folder,
    /// A catalog asset.
    Asset,
}

/// A materialized asset snapshot for UI projection.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AssetSnapshot {
    /// Catalog record.
    pub record: AssetRecord,
    /// Current VCS metadata for the record path.
    pub vcs: VcsRecord,
}

/// Severity of a content diagnostic.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum DiagnosticSeverity {
    /// Informational finding.
    Info,
    /// Non-blocking problem.
    Warning,
    /// Problem that prevents a valid content graph.
    Error,
}

/// Stable diagnostic code.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum DiagnosticCode {
    /// A dependency identity is not present in the catalog.
    MissingDependency,
    /// The dependency graph contains a cycle.
    DependencyCycle,
    /// Validation stopped at the configured dependency depth.
    DependencyDepthExceeded,
    /// The VCS provider reports a conflict.
    VersionControlConflict,
}

/// One deterministic content validation finding.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ContentDiagnostic {
    /// Finding severity.
    pub severity: DiagnosticSeverity,
    /// Stable finding code.
    pub code: DiagnosticCode,
    /// Asset that owns the finding, when applicable.
    pub asset: Option<AssetId>,
    /// Path associated with the finding, when applicable.
    pub path: Option<String>,
    /// Human-readable deterministic message.
    pub message: String,
}

/// Bounded validation result.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiagnosticReport {
    diagnostics: Vec<ContentDiagnostic>,
    errors: usize,
    warnings: usize,
    complete: bool,
    /// Whether the configured diagnostic limit omitted additional findings.
    pub truncated: bool,
}

impl DiagnosticReport {
    /// Returns findings in deterministic order.
    #[must_use]
    pub fn diagnostics(&self) -> &[ContentDiagnostic] {
        &self.diagnostics
    }

    /// Returns whether at least one error finding exists.
    #[must_use]
    pub fn has_errors(&self) -> bool {
        self.errors != 0
    }

    /// Number of error findings before the display list was truncated.
    #[must_use]
    pub const fn error_count(&self) -> usize {
        self.errors
    }

    /// Number of warning findings before display truncation.
    #[must_use]
    pub const fn warning_count(&self) -> usize {
        self.warnings
    }

    /// Whether every built-in check completed within its traversal bounds.
    #[must_use]
    pub const fn is_complete(&self) -> bool {
        self.complete
    }
}

/// Failure while mutating or querying the content data layer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ContentError {
    /// A required string was empty.
    EmptyValue(&'static str),
    /// A path was not canonical and project-relative.
    InvalidPath(String),
    /// An asset identity was already registered.
    DuplicateAsset(AssetId),
    /// An asset path was already registered.
    DuplicatePath(String),
    /// A path would be both an asset and a folder.
    PathPrefixConflict(String),
    /// A referenced asset does not exist.
    UnknownAsset(AssetId),
    /// A requested folder or path does not exist as a tree node.
    UnknownPath(String),
    /// A page request exceeds configured bounds or is zero.
    InvalidPageSize,
    /// A cursor belongs to another catalog generation or query.
    InvalidCursor,
    /// A bounded limit is zero.
    InvalidLimits,
    /// A dependency walk exceeded configured bounds.
    TraversalLimitExceeded,
    /// A completed refresh was prepared before a newer catalog edit.
    StaleSnapshot {
        /// Generation observed when the refresh began.
        expected: u64,
        /// Generation of the catalog now.
        actual: u64,
    },
    /// An asynchronous result belongs to another exact catalog snapshot.
    StaleCatalogSnapshot,
    /// A complete asset refresh exceeded one of its configured bounds.
    SnapshotLimitExceeded(&'static str),
}

impl fmt::Display for ContentError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyValue(name) => write!(formatter, "{name} must not be empty"),
            Self::InvalidPath(path) => write!(formatter, "invalid asset path: {path}"),
            Self::DuplicateAsset(id) => {
                write!(formatter, "asset {} is already registered", id.raw())
            }
            Self::DuplicatePath(path) => {
                write!(formatter, "asset path is already registered: {path}")
            }
            Self::PathPrefixConflict(path) => {
                write!(formatter, "asset path conflicts with a folder: {path}")
            }
            Self::UnknownAsset(id) => write!(formatter, "unknown asset {}", id.raw()),
            Self::UnknownPath(path) => write!(formatter, "unknown content path: {path}"),
            Self::InvalidPageSize => formatter.write_str("invalid or excessive page size"),
            Self::InvalidCursor => {
                formatter.write_str("cursor does not match the current catalog query")
            }
            Self::InvalidLimits => formatter.write_str("content limits must all be non-zero"),
            Self::TraversalLimitExceeded => {
                formatter.write_str("dependency traversal exceeded configured bounds")
            }
            Self::StaleSnapshot { expected, actual } => {
                write!(formatter, "catalog refresh generation {expected} is stale; current generation is {actual}")
            }
            Self::StaleCatalogSnapshot => formatter.write_str("content catalog snapshot changed"),
            Self::SnapshotLimitExceeded(name) => {
                write!(formatter, "catalog refresh exceeded {name}")
            }
        }
    }
}

impl core::error::Error for ContentError {}

/// Failure while creating or advancing a content task.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TaskError {
    /// Task history has no evictable terminal slot.
    Capacity,
    /// The requested export has no assets.
    EmptyExport,
    /// The export contains too many assets.
    ExportTooLarge,
    /// The task identity space is exhausted.
    IdExhausted,
    /// The task does not exist.
    UnknownTask(ContentTaskId),
    /// The requested state transition is invalid.
    InvalidState(ContentTaskId),
    /// Progress is outside zero through one thousand or moves backwards.
    InvalidProgress(ContentTaskId),
    /// A task message is empty after trimming.
    EmptyMessage,
}

impl fmt::Display for TaskError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Capacity => formatter.write_str("task history has no available bounded slot"),
            Self::EmptyExport => formatter.write_str("export task must contain at least one asset"),
            Self::ExportTooLarge => {
                formatter.write_str("export task exceeds the configured asset bound")
            }
            Self::IdExhausted => formatter.write_str("content task identity space exhausted"),
            Self::UnknownTask(id) => write!(formatter, "unknown content task {}", id.raw()),
            Self::InvalidState(id) => {
                write!(formatter, "invalid state for content task {}", id.raw())
            }
            Self::InvalidProgress(id) => {
                write!(formatter, "invalid progress for content task {}", id.raw())
            }
            Self::EmptyMessage => formatter.write_str("task message must not be empty"),
        }
    }
}

impl core::error::Error for TaskError {}

/// Bounded deterministic import/export task store.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TaskStore {
    next_id: u64,
    max_history: usize,
    max_export_assets: usize,
    tasks: BTreeMap<ContentTaskId, ContentTask>,
}

impl TaskStore {
    /// Creates a task store with explicit non-zero bounds.
    pub fn new(max_history: usize, max_export_assets: usize) -> Result<Self, TaskError> {
        if max_history == 0 || max_export_assets == 0 {
            return Err(TaskError::Capacity);
        }
        Ok(Self {
            next_id: 1,
            max_history,
            max_export_assets,
            tasks: BTreeMap::new(),
        })
    }

    /// Enqueues an import task.
    pub fn enqueue_import(&mut self, asset: AssetId) -> Result<ContentTask, TaskError> {
        self.enqueue(TaskKind::Import { asset })
    }

    /// Enqueues an export task for a bounded, non-empty selection.
    pub fn enqueue_export<I>(&mut self, assets: I) -> Result<ContentTask, TaskError>
    where
        I: IntoIterator<Item = AssetId>,
    {
        let assets: Vec<_> = assets.into_iter().collect();
        if assets.is_empty() {
            return Err(TaskError::EmptyExport);
        }
        if assets.len() > self.max_export_assets {
            return Err(TaskError::ExportTooLarge);
        }
        self.enqueue(TaskKind::Export { assets })
    }

    /// Starts a queued task.
    pub fn start(&mut self, id: ContentTaskId) -> Result<ContentTask, TaskError> {
        let task = self.task_mut(id)?;
        if task.state != TaskState::Queued {
            return Err(TaskError::InvalidState(id));
        }
        task.state = TaskState::Running;
        Ok(task.clone())
    }

    /// Updates monotonic task progress and moves a queued task to running.
    pub fn set_progress(
        &mut self,
        id: ContentTaskId,
        progress_milli: u16,
    ) -> Result<ContentTask, TaskError> {
        if progress_milli > 1_000 {
            return Err(TaskError::InvalidProgress(id));
        }
        let task = self.task_mut(id)?;
        if task.state.is_terminal() || progress_milli < task.progress_milli {
            return Err(TaskError::InvalidProgress(id));
        }
        if task.state == TaskState::Queued {
            task.state = TaskState::Running;
        }
        task.progress_milli = progress_milli;
        Ok(task.clone())
    }

    /// Marks a running task successful at full progress.
    pub fn succeed(&mut self, id: ContentTaskId) -> Result<ContentTask, TaskError> {
        let task = self.task_mut(id)?;
        if task.state != TaskState::Running {
            return Err(TaskError::InvalidState(id));
        }
        task.state = TaskState::Succeeded;
        task.progress_milli = 1_000;
        Ok(task.clone())
    }

    /// Marks a queued or running task failed with a message.
    pub fn fail(
        &mut self,
        id: ContentTaskId,
        message: impl Into<String>,
    ) -> Result<ContentTask, TaskError> {
        let message = normalize_required(message.into(), "task message")
            .map_err(|_| TaskError::EmptyMessage)?;
        let task = self.task_mut(id)?;
        if task.state.is_terminal() {
            return Err(TaskError::InvalidState(id));
        }
        task.state = TaskState::Failed;
        task.message = Some(message);
        Ok(task.clone())
    }

    /// Cancels a queued or running task.
    pub fn cancel(&mut self, id: ContentTaskId) -> Result<ContentTask, TaskError> {
        let task = self.task_mut(id)?;
        if task.state.is_terminal() {
            return Err(TaskError::InvalidState(id));
        }
        task.state = TaskState::Cancelled;
        Ok(task.clone())
    }

    /// Returns a task snapshot.
    #[must_use]
    pub fn get(&self, id: ContentTaskId) -> Option<&ContentTask> {
        self.tasks.get(&id)
    }

    /// Returns task snapshots in ascending identity order.
    pub fn tasks(&self) -> impl Iterator<Item = &ContentTask> {
        self.tasks.values()
    }

    /// Returns the number of retained task snapshots.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tasks.len()
    }

    /// Returns whether no task snapshots are retained.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tasks.is_empty()
    }

    fn enqueue(&mut self, kind: TaskKind) -> Result<ContentTask, TaskError> {
        if self.tasks.len() >= self.max_history {
            let evict = self
                .tasks
                .iter()
                .find(|(_, task)| task.state.is_terminal())
                .map(|(id, _)| *id)
                .ok_or(TaskError::Capacity)?;
            self.tasks.remove(&evict);
        }
        let id = ContentTaskId(self.next_id);
        self.next_id = self.next_id.checked_add(1).ok_or(TaskError::IdExhausted)?;
        let task = ContentTask {
            id,
            kind,
            state: TaskState::Queued,
            progress_milli: 0,
            message: None,
        };
        self.tasks.insert(id, task.clone());
        Ok(task)
    }

    fn task_mut(&mut self, id: ContentTaskId) -> Result<&mut ContentTask, TaskError> {
        self.tasks.get_mut(&id).ok_or(TaskError::UnknownTask(id))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct TreeChild {
    path: String,
    asset: Option<AssetId>,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct CacheKey {
    kind: u8,
    fingerprint: u64,
    offset: usize,
    limit: usize,
}

#[derive(Clone, Debug)]
struct CachedAssetPage {
    items: Vec<AssetId>,
    has_more: bool,
    last_used: u64,
}

#[derive(Clone, Debug)]
struct CachedTreePage {
    items: Vec<TreeEntry>,
    has_more: bool,
    last_used: u64,
}

#[derive(Clone, Debug)]
struct NormalizedFilter {
    root: Option<String>,
    text: Option<String>,
    kind: Option<AssetKind>,
    tag: Option<String>,
    favorites_only: bool,
    pinned_only: bool,
    vcs: Option<VcsState>,
}

/// Deterministic asset catalog, tree index, dependency index, page cache, and VCS view.
#[derive(Clone, Debug)]
pub struct ContentCatalog {
    limits: ContentLimits,
    generation: u64,
    snapshot_identity: Arc<()>,
    assets: BTreeMap<AssetId, AssetRecord>,
    paths: BTreeMap<String, AssetId>,
    reverse_dependencies: BTreeMap<AssetId, BTreeSet<AssetId>>,
    tree_children: BTreeMap<String, BTreeMap<String, TreeChild>>,
    folder_counts: BTreeMap<String, usize>,
    vcs: BTreeMap<String, VcsRecord>,
    asset_cache: BTreeMap<CacheKey, CachedAssetPage>,
    tree_cache: BTreeMap<CacheKey, CachedTreePage>,
    cache_clock: u64,
    cache_hits: u64,
    cache_misses: u64,
}

impl Default for ContentCatalog {
    fn default() -> Self {
        Self::with_limits(ContentLimits::default()).expect("default content limits are valid")
    }
}

impl ContentCatalog {
    /// Creates an empty catalog with default bounds.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates an empty catalog with explicit bounds.
    pub fn with_limits(limits: ContentLimits) -> Result<Self, ContentError> {
        let limits = limits.validate()?;
        Ok(Self {
            limits,
            generation: 0,
            snapshot_identity: Arc::new(()),
            assets: BTreeMap::new(),
            paths: BTreeMap::new(),
            reverse_dependencies: BTreeMap::new(),
            tree_children: BTreeMap::from([(String::new(), BTreeMap::new())]),
            folder_counts: BTreeMap::from([(String::new(), 0)]),
            vcs: BTreeMap::new(),
            asset_cache: BTreeMap::new(),
            tree_cache: BTreeMap::new(),
            cache_clock: 0,
            cache_hits: 0,
            cache_misses: 0,
        })
    }

    /// Returns the active bounded policy.
    #[must_use]
    pub const fn limits(&self) -> ContentLimits {
        self.limits
    }

    /// Returns the current mutation generation used by cursors.
    #[must_use]
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    /// Capture an exact immutable identity before starting asynchronous work.
    /// Clones share this identity only until an effective mutation occurs.
    #[must_use]
    pub fn snapshot_token(&self) -> ContentSnapshotToken {
        ContentSnapshotToken(self.snapshot_identity.clone())
    }

    /// Whether this catalog is still the exact captured snapshot. Unlike the
    /// cursor generation, this also rejects unrelated catalogs and wraparound.
    #[must_use]
    pub fn matches_snapshot(&self, token: &ContentSnapshotToken) -> bool {
        Arc::ptr_eq(&self.snapshot_identity, &token.0)
    }

    /// Returns the number of registered assets.
    #[must_use]
    pub fn len(&self) -> usize {
        self.assets.len()
    }

    /// Returns whether the catalog contains no assets.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.assets.is_empty()
    }

    /// Atomically replaces the full asset collection and its derived indexes.
    /// Stable identities and authored metadata belong to the caller. Missing
    /// dependencies remain diagnosable; VCS observations remain keyed by path.
    ///
    /// Capture `generation()` before starting asynchronous discovery. A stale
    /// generation rejects the refresh before consuming any records, preventing
    /// a completed scan from overwriting newer edits. Path swaps and removals
    /// are checked against the candidate, never against the old collection.
    /// An identical collection retains generation, cursors, and cached pages.
    pub fn replace_assets(
        &mut self,
        expected_generation: u64,
        records: impl IntoIterator<Item = AssetRecord>,
        limits: CatalogSnapshotLimits,
    ) -> Result<bool, ContentError> {
        self.replace_asset_results(expected_generation, records.into_iter().map(Ok), limits)
    }

    fn replace_asset_results(
        &mut self,
        expected_generation: u64,
        records: impl IntoIterator<Item = Result<AssetRecord, ContentError>>,
        limits: CatalogSnapshotLimits,
    ) -> Result<bool, ContentError> {
        if expected_generation != self.generation {
            return Err(ContentError::StaleSnapshot {
                expected: expected_generation,
                actual: self.generation,
            });
        }
        if limits.max_path_bytes == 0 || limits.max_path_depth == 0 {
            return Err(ContentError::InvalidLimits);
        }
        let mut candidate = Self::with_limits(self.limits)?;
        let mut dependencies = 0usize;
        let mut tags = 0usize;
        let mut payload_bytes = 0usize;
        for record in records {
            let record = record?;
            if candidate.len() >= limits.max_assets {
                return Err(ContentError::SnapshotLimitExceeded("asset count"));
            }
            if record.path.len() > limits.max_path_bytes {
                return Err(ContentError::SnapshotLimitExceeded("path bytes"));
            }
            if record.path.split('/').count() > limits.max_path_depth {
                return Err(ContentError::SnapshotLimitExceeded("path depth"));
            }
            let mut account = |amount: usize| {
                payload_bytes = payload_bytes
                    .checked_add(amount)
                    .filter(|total| *total <= limits.max_payload_bytes)
                    .ok_or(ContentError::SnapshotLimitExceeded("payload bytes"))?;
                Ok::<(), ContentError>(())
            };
            account(16)?;
            account(record.path.len())?;
            account(record.kind.as_str().len())?;
            dependencies = dependencies
                .checked_add(record.dependencies.len())
                .filter(|total| *total <= limits.max_dependencies)
                .ok_or(ContentError::SnapshotLimitExceeded("dependency count"))?;
            account(
                record
                    .dependencies
                    .len()
                    .checked_mul(16)
                    .ok_or(ContentError::SnapshotLimitExceeded("payload bytes"))?,
            )?;
            tags = tags
                .checked_add(record.tags.len())
                .filter(|total| *total <= limits.max_tags)
                .ok_or(ContentError::SnapshotLimitExceeded("tag count"))?;
            for tag in &record.tags {
                if tag.trim().is_empty() {
                    return Err(ContentError::EmptyValue("asset tag"));
                }
                account(tag.len())?;
            }
            let path = canonical_path(record.path.clone())?;
            if candidate.assets.contains_key(&record.id) {
                return Err(ContentError::DuplicateAsset(record.id));
            }
            candidate.check_path_available(&path, None)?;
            candidate.paths.insert(path, record.id);
            candidate.assets.insert(record.id, record);
        }
        if self.assets == candidate.assets {
            return Ok(false);
        }
        if self.paths != candidate.paths {
            let paths = core::mem::take(&mut candidate.paths);
            for (path, id) in &paths {
                candidate.add_tree_path(path, *id);
            }
            self.paths = paths;
            self.tree_children = candidate.tree_children;
            self.folder_counts = candidate.folder_counts;
        }
        // Validation is complete. Apply only edge differences, retaining reverse
        // references to removed targets when their source still references them.
        let previous = core::mem::replace(&mut self.assets, candidate.assets);
        for (id, old) in &previous {
            for dependency in &old.dependencies {
                if !self
                    .assets
                    .get(id)
                    .is_some_and(|new| new.dependencies.contains(dependency))
                {
                    self.remove_reverse_edge(*dependency, *id);
                }
            }
        }
        for (id, new) in &self.assets {
            for dependency in &new.dependencies {
                if !previous
                    .get(id)
                    .is_some_and(|old| old.dependencies.contains(dependency))
                {
                    self.reverse_dependencies
                        .entry(*dependency)
                        .or_default()
                        .insert(*id);
                }
            }
        }
        self.changed();
        Ok(true)
    }

    /// Inserts one asset and updates all deterministic indexes.
    pub fn insert(&mut self, mut record: AssetRecord) -> Result<(), ContentError> {
        record.path = canonical_path(record.path)?;
        if self.assets.contains_key(&record.id) {
            return Err(ContentError::DuplicateAsset(record.id));
        }
        self.check_path_available(&record.path, None)?;
        self.paths.insert(record.path.clone(), record.id);
        self.add_tree_path(&record.path, record.id);
        for dependency in &record.dependencies {
            self.reverse_dependencies
                .entry(*dependency)
                .or_default()
                .insert(record.id);
        }
        self.assets.insert(record.id, record);
        self.changed();
        Ok(())
    }

    /// Replaces an existing asset record while preserving index invariants.
    pub fn update(&mut self, mut record: AssetRecord) -> Result<(), ContentError> {
        record.path = canonical_path(record.path)?;
        let old = self
            .assets
            .get(&record.id)
            .cloned()
            .ok_or(ContentError::UnknownAsset(record.id))?;
        self.check_path_available(&record.path, Some(record.id))?;
        if old == record {
            return Ok(());
        }
        if old.path != record.path {
            self.remove_tree_path(&old.path);
            self.paths.remove(&old.path);
            if let Some(status) = self.vcs.remove(&old.path) {
                self.vcs.entry(record.path.clone()).or_insert(status);
            }
            self.paths.insert(record.path.clone(), record.id);
            self.add_tree_path(&record.path, record.id);
        }
        for dependency in old.dependencies.difference(&record.dependencies) {
            self.remove_reverse_edge(*dependency, record.id);
        }
        for dependency in record.dependencies.difference(&old.dependencies) {
            self.reverse_dependencies
                .entry(*dependency)
                .or_default()
                .insert(record.id);
        }
        self.assets.insert(record.id, record);
        self.changed();
        Ok(())
    }

    /// Removes an asset and its owned indexes. References from other assets remain diagnosable.
    pub fn remove(&mut self, id: AssetId) -> Result<AssetRecord, ContentError> {
        let record = self
            .assets
            .remove(&id)
            .ok_or(ContentError::UnknownAsset(id))?;
        self.paths.remove(&record.path);
        self.remove_tree_path(&record.path);
        self.vcs.remove(&record.path);
        for dependency in &record.dependencies {
            self.remove_reverse_edge(*dependency, id);
        }
        self.changed();
        Ok(record)
    }

    /// Returns an asset by identity.
    #[must_use]
    pub fn get(&self, id: AssetId) -> Option<&AssetRecord> {
        self.assets.get(&id)
    }

    /// Returns a materialized asset and its current VCS status.
    #[must_use]
    pub fn snapshot(&self, id: AssetId) -> Option<AssetSnapshot> {
        self.assets.get(&id).map(|record| AssetSnapshot {
            record: record.clone(),
            vcs: self.vcs.get(&record.path).cloned().unwrap_or_default(),
        })
    }

    /// Queries a bounded page in canonical path order.
    pub fn query_page(
        &mut self,
        query: &AssetQuery,
        request: PageRequest,
        cursor: Option<AssetCursor>,
    ) -> Result<AssetPage, ContentError> {
        let filter = normalize_filter(query)?;
        let fingerprint = filter.fingerprint();
        let offset = self.validate_asset_cursor(cursor, fingerprint)?;
        let limit = self.validate_page(request)?;
        let key = CacheKey {
            kind: 0,
            fingerprint,
            offset,
            limit,
        };
        if let Some(cached) = self.asset_cache.get(&key).cloned() {
            let last_used = self.next_cache_clock();
            if let Some(page) = self.asset_cache.get_mut(&key) {
                page.last_used = last_used;
            }
            self.cache_hits = self.cache_hits.saturating_add(1);
            let next = cached.has_more.then(|| AssetCursor {
                generation: self.generation,
                fingerprint,
                offset: offset.saturating_add(limit),
                after: *cached
                    .items
                    .last()
                    .expect("nonempty page with continuation"),
            });
            return Ok(AssetPage {
                items: cached.items,
                next,
                cache: CacheStatus::Hit,
            });
        }
        self.cache_misses = self.cache_misses.saturating_add(1);
        let mut items = Vec::with_capacity(limit);
        let mut has_more = false;
        let prefix = filter.root.as_ref().map(|root| format!("{root}/"));
        let lower = if let Some(cursor) = cursor {
            Bound::Excluded(
                self.assets
                    .get(&cursor.after)
                    .ok_or(ContentError::InvalidCursor)?
                    .path
                    .clone(),
            )
        } else if let Some(root) = &filter.root {
            Bound::Included(if self.paths.contains_key(root) {
                root.clone()
            } else {
                prefix.clone().expect("root has prefix")
            })
        } else {
            Bound::Unbounded
        };
        for (path, id) in self.paths.range((lower, Bound::Unbounded)) {
            if prefix.as_ref().is_some_and(|prefix| {
                !path.starts_with(prefix) && filter.root.as_ref() != Some(path)
            }) {
                break;
            }
            let Some(asset) = self.assets.get(id) else {
                continue;
            };
            if !filter.matches(self, path, asset) {
                continue;
            }
            if items.len() == limit {
                has_more = true;
                break;
            }
            items.push(*id);
        }
        self.cache_asset_page(key, items.clone(), has_more);
        let next = has_more.then(|| AssetCursor {
            generation: self.generation,
            fingerprint,
            offset: offset.saturating_add(limit),
            after: *items.last().expect("nonempty page with continuation"),
        });
        Ok(AssetPage {
            items,
            next,
            cache: CacheStatus::Miss,
        })
    }

    /// Queries a bounded page of immediate children under one tree folder.
    pub fn tree_page(
        &mut self,
        query: &TreeQuery,
        request: PageRequest,
        cursor: Option<TreeCursor>,
    ) -> Result<TreePage, ContentError> {
        let root = canonical_root(query.root.clone())?;
        if !self.tree_children.contains_key(&root) {
            return Err(ContentError::UnknownPath(root));
        }
        let filter = normalize_filter(&AssetQuery {
            root: Some(root.clone()),
            text: query.text.clone(),
            kind: query.kind.clone(),
            tag: query.tag.clone(),
            favorites_only: query.favorites_only,
            pinned_only: query.pinned_only,
            vcs: query.vcs,
        })?;
        let fingerprint = tree_fingerprint(&root, &filter);
        let offset = self.validate_tree_cursor(cursor.as_ref(), fingerprint)?;
        let limit = self.validate_page(request)?;
        let key = CacheKey {
            kind: 1,
            fingerprint,
            offset,
            limit,
        };
        if let Some(cached) = self.tree_cache.get(&key).cloned() {
            let last_used = self.next_cache_clock();
            if let Some(page) = self.tree_cache.get_mut(&key) {
                page.last_used = last_used;
            }
            self.cache_hits = self.cache_hits.saturating_add(1);
            let next = cached.has_more.then(|| TreeCursor {
                generation: self.generation,
                fingerprint,
                offset: offset.saturating_add(limit),
                after: cached
                    .items
                    .last()
                    .expect("nonempty page with continuation")
                    .name
                    .clone(),
            });
            return Ok(TreePage {
                items: cached.items,
                next,
                cache: CacheStatus::Hit,
            });
        }
        self.cache_misses = self.cache_misses.saturating_add(1);
        let children = self
            .tree_children
            .get(&root)
            .expect("tree root validated before querying immutable child index");
        let mut items = Vec::with_capacity(limit);
        let mut has_more = false;
        let lower = cursor.map_or(Bound::Unbounded, |cursor| Bound::Excluded(cursor.after));
        for (name, child) in children.range((lower, Bound::Unbounded)) {
            let matches = match child.asset {
                Some(id) => self
                    .assets
                    .get(&id)
                    .is_some_and(|asset| filter.matches(self, &child.path, asset)),
                None => self.folder_matches(&child.path, &filter),
            };
            if !matches {
                continue;
            }
            if items.len() == limit {
                has_more = true;
                break;
            }
            items.push(TreeEntry {
                path: child.path.clone(),
                name: name.clone(),
                kind: child
                    .asset
                    .map_or(TreeEntryKind::Folder, |_| TreeEntryKind::Asset),
                asset: child.asset,
                descendant_assets: child.asset.map_or_else(
                    || self.folder_counts.get(&child.path).copied().unwrap_or(0),
                    |_| 1,
                ),
            });
        }
        self.cache_tree_page(key, items.clone(), has_more);
        let next = has_more.then(|| TreeCursor {
            generation: self.generation,
            fingerprint,
            offset: offset.saturating_add(limit),
            after: items
                .last()
                .expect("nonempty page with continuation")
                .name
                .clone(),
        });
        Ok(TreePage {
            items,
            next,
            cache: CacheStatus::Miss,
        })
    }

    /// Returns page-cache statistics.
    #[must_use]
    pub fn cache_stats(&self) -> CacheStats {
        CacheStats {
            entries: self.asset_cache.len() + self.tree_cache.len(),
            items: self
                .asset_cache
                .values()
                .map(|page| page.items.len())
                .sum::<usize>()
                + self
                    .tree_cache
                    .values()
                    .map(|page| page.items.len())
                    .sum::<usize>(),
            hits: self.cache_hits,
            misses: self.cache_misses,
        }
    }

    /// Returns direct dependencies in deterministic identity order.
    #[must_use]
    pub fn dependencies(&self, id: AssetId) -> Option<&BTreeSet<AssetId>> {
        self.assets.get(&id).map(|asset| &asset.dependencies)
    }

    /// Adds a direct dependency. Unknown targets remain indexed for diagnostics.
    pub fn add_dependency(
        &mut self,
        asset: AssetId,
        dependency: AssetId,
    ) -> Result<(), ContentError> {
        let record = self
            .assets
            .get_mut(&asset)
            .ok_or(ContentError::UnknownAsset(asset))?;
        if record.dependencies.insert(dependency) {
            self.reverse_dependencies
                .entry(dependency)
                .or_default()
                .insert(asset);
            self.changed();
        }
        Ok(())
    }

    /// Removes a direct dependency, returning whether an edge existed.
    pub fn remove_dependency(
        &mut self,
        asset: AssetId,
        dependency: AssetId,
    ) -> Result<bool, ContentError> {
        let record = self
            .assets
            .get_mut(&asset)
            .ok_or(ContentError::UnknownAsset(asset))?;
        let removed = record.dependencies.remove(&dependency);
        if removed {
            self.remove_reverse_edge(dependency, asset);
            self.changed();
        }
        Ok(removed)
    }

    /// Returns direct reverse references, including references to missing targets.
    #[must_use]
    pub fn reverse_dependencies(&self, target: AssetId) -> Vec<AssetId> {
        self.reverse_dependencies
            .get(&target)
            .map(|ids| ids.iter().copied().collect())
            .unwrap_or_default()
    }

    /// Returns the bounded transitive dependency closure of an asset.
    pub fn dependency_closure(&self, root: AssetId) -> Result<Vec<AssetId>, ContentError> {
        self.walk_dependencies(root, false)
    }

    /// Returns the bounded transitive dependent closure of an asset.
    pub fn dependent_closure(&self, root: AssetId) -> Result<Vec<AssetId>, ContentError> {
        self.walk_dependencies(root, true)
    }

    /// Returns direct reference impact for a deletion.
    #[must_use]
    pub fn delete_impact(&self, target: AssetId) -> Vec<AssetId> {
        self.reverse_dependencies(target)
    }

    /// Validates dependency graph and VCS conflict state in deterministic order.
    #[must_use]
    pub fn validate(&self) -> DiagnosticReport {
        self.validate_with_diagnostic_limit(self.limits.max_diagnostics)
    }

    fn validate_with_diagnostic_limit(&self, max_diagnostics: usize) -> DiagnosticReport {
        let mut diagnostics = BTreeSet::new();
        let mut truncated = false;
        let mut errors = 0usize;
        let mut warnings = 0usize;
        let mut retain = |diagnostic: ContentDiagnostic| {
            if diagnostic.severity == DiagnosticSeverity::Error {
                errors = errors.saturating_add(1);
            }
            if diagnostic.severity == DiagnosticSeverity::Warning {
                warnings = warnings.saturating_add(1);
            }
            diagnostics.insert(diagnostic);
            if diagnostics.len() > max_diagnostics {
                diagnostics.pop_last();
                truncated = true;
            }
        };
        for record in self.assets.values() {
            for dependency in &record.dependencies {
                if !self.assets.contains_key(dependency) {
                    retain(ContentDiagnostic {
                        severity: DiagnosticSeverity::Error,
                        code: DiagnosticCode::MissingDependency,
                        asset: Some(record.id),
                        path: Some(record.path.clone()),
                        message: format!("missing dependency {}", dependency.raw()),
                    });
                }
            }
            if self
                .vcs
                .get(&record.path)
                .is_some_and(|status| status.state == VcsState::Conflicted)
            {
                retain(ContentDiagnostic {
                    severity: DiagnosticSeverity::Error,
                    code: DiagnosticCode::VersionControlConflict,
                    asset: Some(record.id),
                    path: Some(record.path.clone()),
                    message: "version-control conflict".to_string(),
                });
            }
        }
        let (cycle, depth_exceeded) = self.find_dependency_cycle();
        if depth_exceeded {
            retain(ContentDiagnostic {
                severity: DiagnosticSeverity::Warning,
                code: DiagnosticCode::DependencyDepthExceeded,
                asset: None,
                path: None,
                message: "dependency validation reached its configured depth bound".to_string(),
            });
        }
        if let Some(assets) = cycle {
            retain(ContentDiagnostic {
                severity: DiagnosticSeverity::Error,
                code: DiagnosticCode::DependencyCycle,
                asset: assets.first().copied(),
                path: assets
                    .first()
                    .and_then(|id| self.assets.get(id))
                    .map(|record| record.path.clone()),
                message: format_cycle_message(&assets),
            });
        }
        DiagnosticReport {
            diagnostics: diagnostics.into_iter().collect(),
            errors,
            warnings,
            complete: !depth_exceeded,
            truncated,
        }
    }

    /// Associates VCS metadata with a canonical path, including paths not yet imported.
    pub fn set_vcs_status(
        &mut self,
        path: impl Into<String>,
        status: VcsRecord,
    ) -> Result<(), ContentError> {
        let path = canonical_path(path.into())?;
        if self.vcs.get(&path) != Some(&status) {
            self.vcs.insert(path, status);
            self.changed();
        }
        Ok(())
    }

    /// Atomically replaces a complete provider status snapshot. Duplicate or
    /// invalid paths reject the snapshot without changing metadata or cursors.
    /// Returns whether the effective snapshot changed. An unchanged snapshot
    /// preserves the catalog generation and cached pages.
    pub fn replace_vcs_statuses(
        &mut self,
        statuses: impl IntoIterator<Item = (String, VcsRecord)>,
    ) -> Result<bool, ContentError> {
        let mut candidate = BTreeMap::new();
        for (path, status) in statuses {
            let path = canonical_path(path)?;
            if candidate.insert(path.clone(), status).is_some() {
                return Err(ContentError::DuplicatePath(path));
            }
        }
        if self.vcs == candidate {
            return Ok(false);
        }
        self.vcs = candidate;
        self.changed();
        Ok(true)
    }

    /// Returns VCS metadata for an asset, defaulting to unknown.
    #[must_use]
    pub fn vcs_status(&self, id: AssetId) -> Option<VcsRecord> {
        self.assets
            .get(&id)
            .map(|asset| self.vcs.get(&asset.path).cloned().unwrap_or_default())
    }

    fn remove_reverse_edge(&mut self, dependency: AssetId, asset: AssetId) {
        if let Some(dependents) = self.reverse_dependencies.get_mut(&dependency) {
            dependents.remove(&asset);
            if dependents.is_empty() {
                self.reverse_dependencies.remove(&dependency);
            }
        }
    }

    fn validate_page(&self, request: PageRequest) -> Result<usize, ContentError> {
        if request.limit == 0 || request.limit > self.limits.max_page_size {
            return Err(ContentError::InvalidPageSize);
        }
        Ok(request.limit)
    }

    fn validate_asset_cursor(
        &self,
        cursor: Option<AssetCursor>,
        fingerprint: u64,
    ) -> Result<usize, ContentError> {
        let Some(cursor) = cursor else { return Ok(0) };
        if cursor.generation != self.generation || cursor.fingerprint != fingerprint {
            return Err(ContentError::InvalidCursor);
        }
        Ok(cursor.offset)
    }

    fn validate_tree_cursor(
        &self,
        cursor: Option<&TreeCursor>,
        fingerprint: u64,
    ) -> Result<usize, ContentError> {
        let Some(cursor) = cursor else { return Ok(0) };
        if cursor.generation != self.generation || cursor.fingerprint != fingerprint {
            return Err(ContentError::InvalidCursor);
        }
        Ok(cursor.offset)
    }

    fn check_path_available(
        &self,
        path: &str,
        updating: Option<AssetId>,
    ) -> Result<(), ContentError> {
        if let Some(existing) = self.paths.get(path)
            && Some(*existing) != updating
        {
            return Err(ContentError::DuplicatePath(path.to_string()));
        }
        let parts: Vec<_> = path.split('/').collect();
        let mut ancestor = String::new();
        for (index, part) in parts.iter().enumerate() {
            if index > 0 {
                ancestor.push('/');
            }
            ancestor.push_str(part);
            if index + 1 != parts.len()
                && self
                    .paths
                    .get(&ancestor)
                    .is_some_and(|id| Some(*id) != updating)
            {
                return Err(ContentError::PathPrefixConflict(ancestor));
            }
        }
        let prefix = format!("{path}/");
        for (descendant, id) in self.paths.range(prefix.clone()..) {
            if !descendant.starts_with(&prefix) {
                break;
            }
            if Some(*id) != updating {
                return Err(ContentError::PathPrefixConflict(path.to_string()));
            }
        }
        Ok(())
    }

    fn add_tree_path(&mut self, path: &str, asset: AssetId) {
        let parts: Vec<_> = path.split('/').collect();
        let mut parent = String::new();
        for (index, part) in parts.iter().enumerate() {
            let child_path = join_path(&parent, part);
            let is_asset = index + 1 == parts.len();
            self.tree_children
                .entry(parent.clone())
                .or_default()
                .insert(
                    (*part).to_string(),
                    TreeChild {
                        path: child_path.clone(),
                        asset: is_asset.then_some(asset),
                    },
                );
            if !is_asset {
                self.tree_children.entry(child_path.clone()).or_default();
            }
            *self.folder_counts.entry(parent.clone()).or_default() += 1;
            parent = child_path;
        }
    }

    fn remove_tree_path(&mut self, path: &str) {
        let parts: Vec<_> = path.split('/').collect();
        let mut parent = String::new();
        let mut parents = Vec::with_capacity(parts.len());
        for part in &parts {
            parents.push(parent.clone());
            parent = join_path(&parent, part);
        }
        if let Some(last_parent) = parents.last()
            && let Some(children) = self.tree_children.get_mut(last_parent)
        {
            children.remove(*parts.last().expect("canonical asset path is non-empty"));
        }
        for index in (0..parts.len()).rev() {
            let node = parts[..=index].join("/");
            let containing = if index == 0 {
                String::new()
            } else {
                parts[..index].join("/")
            };
            if let Some(count) = self.folder_counts.get_mut(&containing) {
                *count = count.saturating_sub(1);
            }
            if index < parts.len() - 1 && self.folder_counts.get(&node).copied().unwrap_or(0) == 0 {
                self.folder_counts.remove(&node);
                self.tree_children.remove(&node);
                if let Some(children) = self.tree_children.get_mut(&containing) {
                    children.remove(parts[index]);
                }
            }
        }
    }

    fn folder_matches(&self, folder: &str, filter: &NormalizedFilter) -> bool {
        let prefix = format!("{folder}/");
        for (path, id) in self.paths.range(prefix.clone()..) {
            if !path.starts_with(&prefix) {
                break;
            }
            if let Some(asset) = self.assets.get(id)
                && filter.matches(self, path, asset)
            {
                return true;
            }
        }
        false
    }

    fn walk_dependencies(
        &self,
        root: AssetId,
        reverse: bool,
    ) -> Result<Vec<AssetId>, ContentError> {
        if !self.assets.contains_key(&root) {
            return Err(ContentError::UnknownAsset(root));
        }
        let mut result = Vec::new();
        let mut seen = BTreeSet::from([root]);
        let mut frontier = vec![root];
        for depth in 0..=self.limits.max_dependency_depth {
            let mut next = Vec::new();
            for id in frontier {
                let neighbors = if reverse {
                    self.reverse_dependencies.get(&id)
                } else {
                    self.assets.get(&id).map(|asset| &asset.dependencies)
                };
                for &neighbor in neighbors.into_iter().flatten() {
                    if !seen.contains(&neighbor) {
                        if depth == self.limits.max_dependency_depth
                            || result.len() >= self.limits.max_dependency_nodes
                        {
                            return Err(ContentError::TraversalLimitExceeded);
                        }
                        seen.insert(neighbor);
                        result.push(neighbor);
                        if self.assets.contains_key(&neighbor) {
                            next.push(neighbor);
                        }
                    }
                }
            }
            if next.is_empty() {
                return Ok(result);
            }
            frontier = next;
        }
        Err(ContentError::TraversalLimitExceeded)
    }

    fn find_dependency_cycle(&self) -> (Option<Vec<AssetId>>, bool) {
        let mut colors = BTreeMap::<AssetId, u8>::new();
        let mut path = Vec::new();
        let mut positions = BTreeMap::<AssetId, usize>::new();
        let mut depth_exceeded = false;
        for start in self.assets.keys().copied() {
            if colors.get(&start).copied().unwrap_or(0) != 0 {
                continue;
            }
            colors.insert(start, 1);
            positions.insert(start, path.len());
            path.push(start);
            let mut frames = vec![(start, self.assets[&start].dependencies.iter())];
            while let Some((id, dependencies)) = frames.last_mut() {
                let Some(&dependency) = dependencies.next() else {
                    colors.insert(*id, 2);
                    positions.remove(id);
                    path.pop();
                    frames.pop();
                    continue;
                };
                let Some(record) = self.assets.get(&dependency) else {
                    continue;
                };
                match colors.get(&dependency).copied().unwrap_or(0) {
                    0 => {
                        if path.len() > self.limits.max_dependency_depth {
                            depth_exceeded = true;
                            continue;
                        }
                        colors.insert(dependency, 1);
                        positions.insert(dependency, path.len());
                        path.push(dependency);
                        frames.push((dependency, record.dependencies.iter()));
                    }
                    1 => {
                        let start_index = positions[&dependency];
                        let mut cycle = path[start_index..].to_vec();
                        cycle.push(dependency);
                        return (Some(cycle), depth_exceeded);
                    }
                    _ => {}
                }
            }
        }
        (None, depth_exceeded)
    }

    fn changed(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        self.snapshot_identity = Arc::new(());
        self.asset_cache.clear();
        self.tree_cache.clear();
    }

    fn cache_asset_page(&mut self, key: CacheKey, items: Vec<AssetId>, has_more: bool) {
        if items.len() > self.limits.max_cache_items {
            return;
        }
        self.evict_cache_until(items.len(), key);
        let last_used = self.next_cache_clock();
        self.asset_cache.insert(
            key,
            CachedAssetPage {
                items,
                has_more,
                last_used,
            },
        );
    }

    fn cache_tree_page(&mut self, key: CacheKey, items: Vec<TreeEntry>, has_more: bool) {
        if items.len() > self.limits.max_cache_items {
            return;
        }
        self.evict_cache_until(items.len(), key);
        let last_used = self.next_cache_clock();
        self.tree_cache.insert(
            key,
            CachedTreePage {
                items,
                has_more,
                last_used,
            },
        );
    }

    fn evict_cache_until(&mut self, additional: usize, replacing: CacheKey) {
        let mut entries = self.asset_cache.len() + self.tree_cache.len();
        let mut items = self
            .asset_cache
            .values()
            .map(|page| page.items.len())
            .sum::<usize>()
            + self
                .tree_cache
                .values()
                .map(|page| page.items.len())
                .sum::<usize>();
        if self.asset_cache.contains_key(&replacing) {
            entries = entries.saturating_sub(1);
            items = items.saturating_sub(self.asset_cache[&replacing].items.len());
        }
        if self.tree_cache.contains_key(&replacing) {
            entries = entries.saturating_sub(1);
            items = items.saturating_sub(self.tree_cache[&replacing].items.len());
        }
        while entries >= self.limits.max_cache_entries
            || items.saturating_add(additional) > self.limits.max_cache_items
        {
            let Some(victim) = self.oldest_cache_key(replacing) else {
                break;
            };
            let victim_items = self.cached_items(victim);
            if self.asset_cache.remove(&victim).is_none() {
                self.tree_cache.remove(&victim);
            }
            entries = entries.saturating_sub(1);
            items = items.saturating_sub(victim_items);
        }
    }

    fn oldest_cache_key(&self, excluded: CacheKey) -> Option<CacheKey> {
        let mut candidate: Option<(u64, CacheKey)> = None;
        for (key, page) in &self.asset_cache {
            if *key == excluded {
                continue;
            }
            let value = (page.last_used, *key);
            if candidate.as_ref().is_none_or(|current| value < *current) {
                candidate = Some(value);
            }
        }
        for (key, page) in &self.tree_cache {
            if *key == excluded {
                continue;
            }
            let value = (page.last_used, *key);
            if candidate.as_ref().is_none_or(|current| value < *current) {
                candidate = Some(value);
            }
        }
        candidate.map(|(_, key)| key)
    }

    fn cached_items(&self, key: CacheKey) -> usize {
        self.asset_cache
            .get(&key)
            .map(|page| page.items.len())
            .or_else(|| self.tree_cache.get(&key).map(|page| page.items.len()))
            .unwrap_or(0)
    }

    fn next_cache_clock(&mut self) -> u64 {
        self.cache_clock = self.cache_clock.wrapping_add(1);
        self.cache_clock
    }
}

impl NormalizedFilter {
    fn fingerprint(&self) -> u64 {
        let mut hash = 1_469_598_103_934_665_603u64;
        hash_field(&mut hash, self.root.as_deref());
        hash_field(&mut hash, self.text.as_deref());
        hash_field(&mut hash, self.kind.as_ref().map(AssetKind::as_str));
        hash_field(&mut hash, self.tag.as_deref());
        hash_bool(&mut hash, self.favorites_only);
        hash_bool(&mut hash, self.pinned_only);
        hash_field(&mut hash, self.vcs.map(vcs_name));
        hash
    }

    fn matches(&self, catalog: &ContentCatalog, path: &str, asset: &AssetRecord) -> bool {
        if let Some(root) = &self.root {
            let prefix = format!("{root}/");
            if path != root && !path.starts_with(&prefix) {
                return false;
            }
        }
        if let Some(text) = &self.text {
            let path_lower = path.to_ascii_lowercase();
            let kind_lower = asset.kind.as_str().to_ascii_lowercase();
            if !path_lower.contains(text)
                && !kind_lower.contains(text)
                && !asset
                    .tags
                    .iter()
                    .any(|tag| tag.to_ascii_lowercase().contains(text))
            {
                return false;
            }
        }
        if self.kind.as_ref().is_some_and(|kind| kind != &asset.kind) {
            return false;
        }
        if self
            .tag
            .as_ref()
            .is_some_and(|tag| !asset.tags.contains(tag))
        {
            return false;
        }
        if self.favorites_only && !asset.favorite {
            return false;
        }
        if self.pinned_only && !asset.pinned {
            return false;
        }
        if self.vcs.is_some_and(|vcs| {
            catalog
                .vcs
                .get(path)
                .map_or(VcsState::Unknown, |record| record.state)
                != vcs
        }) {
            return false;
        }
        true
    }
}

fn normalize_filter(query: &AssetQuery) -> Result<NormalizedFilter, ContentError> {
    let root = query
        .root
        .as_ref()
        .map(|root| canonical_root(root.clone()))
        .transpose()?
        .filter(|root| !root.is_empty());
    Ok(NormalizedFilter {
        root,
        text: normalize_optional(query.text.clone()),
        kind: query.kind.clone(),
        tag: query
            .tag
            .as_ref()
            .map(|tag| normalize_text(tag))
            .filter(|tag| !tag.is_empty()),
        favorites_only: query.favorites_only,
        pinned_only: query.pinned_only,
        vcs: query.vcs,
    })
}

fn tree_fingerprint(root: &str, filter: &NormalizedFilter) -> u64 {
    let mut hash = filter.fingerprint();
    hash_field(&mut hash, Some(root));
    hash
}

fn canonical_path(path: String) -> Result<String, ContentError> {
    if path.is_empty()
        || path.starts_with('/')
        || path.contains('\\')
        || path.contains('\0')
        || path
            .split('/')
            .any(|part| part.is_empty() || part == "." || part == "..")
    {
        return Err(ContentError::InvalidPath(path));
    }
    Ok(path)
}

fn canonical_root(path: String) -> Result<String, ContentError> {
    if path.is_empty() {
        Ok(String::new())
    } else {
        canonical_path(path)
    }
}

fn normalize_required(value: String, name: &'static str) -> Result<String, ContentError> {
    let value = normalize_text(&value);
    if value.is_empty() {
        Err(ContentError::EmptyValue(name))
    } else {
        Ok(value)
    }
}

fn normalize_optional(value: Option<String>) -> Option<String> {
    value
        .map(|value| normalize_text(&value))
        .filter(|value| !value.is_empty())
}

fn normalize_text(value: &str) -> String {
    value.trim().to_ascii_lowercase()
}

fn join_path(parent: &str, child: &str) -> String {
    if parent.is_empty() {
        child.to_string()
    } else {
        format!("{parent}/{child}")
    }
}

fn format_cycle_message(cycle: &[AssetId]) -> String {
    let mut message = String::from("dependency cycle:");
    for id in cycle {
        message.push(' ');
        message.push_str(&id.raw().to_string());
    }
    message
}

fn vcs_name(state: VcsState) -> &'static str {
    match state {
        VcsState::Unknown => "unknown",
        VcsState::Clean => "clean",
        VcsState::Untracked => "untracked",
        VcsState::Added => "added",
        VcsState::Modified => "modified",
        VcsState::Deleted => "deleted",
        VcsState::Renamed => "renamed",
        VcsState::Ignored => "ignored",
        VcsState::Conflicted => "conflicted",
    }
}

fn hash_field(hash: &mut u64, value: Option<&str>) {
    hash_bytes(hash, value.unwrap_or("<none>").as_bytes());
    hash_bytes(hash, &[0]);
}

fn hash_bool(hash: &mut u64, value: bool) {
    hash_bytes(hash, &[u8::from(value), 0]);
}

fn hash_bytes(hash: &mut u64, bytes: &[u8]) {
    for byte in bytes {
        *hash ^= u64::from(*byte);
        *hash = hash.wrapping_mul(1_099_511_628_211);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn asset(id: u128, path: &str, kind: &str) -> AssetRecord {
        AssetRecord::new(
            AssetId::from_raw(id),
            path,
            AssetKind::new(kind).expect("kind"),
        )
        .expect("asset")
    }

    fn ids(page: &AssetPage) -> Vec<u128> {
        page.items.iter().map(|id| id.raw()).collect()
    }

    #[test]
    fn tree_and_asset_pages_are_deterministic_and_bounded() {
        let mut catalog = ContentCatalog::with_limits(ContentLimits {
            max_page_size: 2,
            max_cache_entries: 4,
            max_cache_items: 8,
            ..ContentLimits::default()
        })
        .expect("limits");
        catalog
            .insert(asset(2, "textures/z.png", "texture"))
            .expect("insert");
        catalog
            .insert(asset(1, "textures/a.png", "texture"))
            .expect("insert");
        catalog
            .insert(asset(3, "meshes/a.mesh", "mesh"))
            .expect("insert");
        let query = AssetQuery {
            text: Some("TEXTURES".into()),
            ..AssetQuery::default()
        };
        let first = catalog
            .query_page(&query, PageRequest::new(1), None)
            .expect("page");
        assert_eq!(ids(&first), vec![1]);
        let second = catalog
            .query_page(&query, PageRequest::new(1), first.next)
            .expect("page");
        assert_eq!(ids(&second), vec![2]);
        assert!(second.next.is_none());
        let tree = catalog
            .tree_page(&TreeQuery::default(), PageRequest::new(2), None)
            .expect("tree");
        assert_eq!(tree.items[0].name, "meshes");
        assert_eq!(tree.items[1].name, "textures");
        assert_eq!(tree.items[1].descendant_assets, 2);
        assert_eq!(
            catalog
                .query_page(&query, PageRequest::new(1), None)
                .expect("cache")
                .cache,
            CacheStatus::Hit
        );
    }

    #[test]
    fn cursor_is_invalidated_by_catalog_mutation() {
        let mut catalog = ContentCatalog::new();
        catalog.insert(asset(1, "a.asset", "data")).expect("insert");
        catalog.insert(asset(2, "b.asset", "data")).expect("insert");
        let page = catalog
            .query_page(&AssetQuery::default(), PageRequest::new(1), None)
            .expect("page");
        catalog.insert(asset(3, "c.asset", "data")).expect("insert");
        assert_eq!(
            catalog.query_page(&AssetQuery::default(), PageRequest::new(1), page.next),
            Err(ContentError::InvalidCursor)
        );
    }

    #[test]
    fn subtree_queries_and_page_limits_are_explicit() {
        let mut catalog = ContentCatalog::new();
        catalog
            .insert(asset(1, "characters/hero.mesh", "mesh"))
            .expect("insert");
        catalog
            .insert(asset(2, "characters/hero.mat", "material"))
            .expect("insert");
        catalog
            .insert(asset(3, "textures/hero.png", "texture"))
            .expect("insert");
        let page = catalog
            .query_page(
                &AssetQuery {
                    root: Some("characters".into()),
                    ..AssetQuery::default()
                },
                PageRequest::new(8),
                None,
            )
            .expect("subtree");
        assert_eq!(ids(&page), vec![2, 1]);
        assert_eq!(
            catalog.query_page(&AssetQuery::default(), PageRequest::new(0), None),
            Err(ContentError::InvalidPageSize)
        );
        assert_eq!(
            catalog.tree_page(
                &TreeQuery {
                    root: "missing".into(),
                    ..TreeQuery::default()
                },
                PageRequest::new(1),
                None,
            ),
            Err(ContentError::UnknownPath("missing".into()))
        );
    }

    #[test]
    fn search_filters_tags_and_vcs_state() {
        let mut catalog = ContentCatalog::new();
        let mut record = asset(1, "Characters/Hero.asset", "Mesh");
        record.add_tag("Player").expect("tag");
        catalog.insert(record).expect("insert");
        catalog
            .set_vcs_status(
                "Characters/Hero.asset",
                VcsRecord {
                    state: VcsState::Modified,
                    ..VcsRecord::default()
                },
            )
            .expect("vcs");
        let page = catalog
            .query_page(
                &AssetQuery {
                    text: Some("player".into()),
                    vcs: Some(VcsState::Modified),
                    ..AssetQuery::default()
                },
                PageRequest::new(8),
                None,
            )
            .expect("query");
        assert_eq!(ids(&page), vec![1]);
    }

    #[test]
    fn dependency_indexes_allow_missing_edges_and_detect_cycles() {
        let mut catalog = ContentCatalog::new();
        catalog.insert(asset(1, "a.asset", "data")).expect("insert");
        catalog.insert(asset(2, "b.asset", "data")).expect("insert");
        catalog
            .add_dependency(AssetId::from_raw(1), AssetId::from_raw(2))
            .expect("edge");
        catalog
            .add_dependency(AssetId::from_raw(2), AssetId::from_raw(1))
            .expect("cycle");
        catalog
            .add_dependency(AssetId::from_raw(1), AssetId::from_raw(99))
            .expect("missing");
        assert_eq!(
            catalog.reverse_dependencies(AssetId::from_raw(2)),
            vec![AssetId::from_raw(1)]
        );
        let report = catalog.validate();
        assert!(report.has_errors());
        assert!(report
            .diagnostics()
            .iter()
            .any(|diagnostic| diagnostic.code == DiagnosticCode::MissingDependency));
        assert!(report
            .diagnostics()
            .iter()
            .any(|diagnostic| diagnostic.code == DiagnosticCode::DependencyCycle));
        assert_eq!(
            catalog
                .dependency_closure(AssetId::from_raw(1))
                .expect("closure"),
            vec![AssetId::from_raw(2), AssetId::from_raw(99)]
        );
    }

    #[test]
    fn update_and_remove_keep_tree_and_reverse_indexes_consistent() {
        let mut catalog = ContentCatalog::new();
        catalog
            .insert(asset(1, "old/a.asset", "data"))
            .expect("insert");
        catalog
            .insert(asset(2, "use.asset", "scene"))
            .expect("insert");
        catalog
            .add_dependency(AssetId::from_raw(2), AssetId::from_raw(1))
            .expect("edge");
        let mut replacement = asset(1, "new/a.asset", "data");
        replacement.favorite = true;
        catalog.update(replacement).expect("update");
        assert!(catalog
            .get(AssetId::from_raw(1))
            .is_some_and(|record| record.favorite));
        assert_eq!(
            catalog.delete_impact(AssetId::from_raw(1)),
            vec![AssetId::from_raw(2)]
        );
        catalog.remove(AssetId::from_raw(1)).expect("remove");
        let tree = catalog
            .tree_page(&TreeQuery::default(), PageRequest::new(8), None)
            .expect("tree");
        assert_eq!(tree.items.len(), 1);
        assert_eq!(tree.items[0].name, "use.asset");
        assert!(catalog.validate().has_errors());
    }

    #[test]
    fn cache_is_bounded_and_evicts_deterministically() {
        let mut catalog = ContentCatalog::with_limits(ContentLimits {
            max_cache_entries: 1,
            max_cache_items: 1,
            ..ContentLimits::default()
        })
        .expect("limits");
        catalog.insert(asset(1, "a.asset", "data")).expect("insert");
        catalog.insert(asset(2, "b.asset", "data")).expect("insert");
        catalog
            .query_page(&AssetQuery::default(), PageRequest::new(1), None)
            .expect("first");
        catalog
            .query_page(
                &AssetQuery {
                    text: Some("b".into()),
                    ..AssetQuery::default()
                },
                PageRequest::new(1),
                None,
            )
            .expect("second");
        assert_eq!(catalog.cache_stats().entries, 1);
        assert_eq!(catalog.cache_stats().items, 1);
    }

    #[test]
    fn task_store_tracks_import_export_and_bounded_history() {
        let mut tasks = TaskStore::new(1, 2).expect("store");
        let import = tasks.enqueue_import(AssetId::from_raw(7)).expect("import");
        tasks.set_progress(import.id, 500).expect("progress");
        tasks.succeed(import.id).expect("success");
        let export = tasks
            .enqueue_export([AssetId::from_raw(1), AssetId::from_raw(2)])
            .expect("export");
        assert_eq!(
            export.kind,
            TaskKind::Export {
                assets: vec![AssetId::from_raw(1), AssetId::from_raw(2)]
            }
        );
        assert_eq!(tasks.len(), 1);
        assert!(tasks.get(import.id).is_none());
        assert_eq!(
            tasks.start(export.id).expect("start").state,
            TaskState::Running
        );
        tasks.fail(export.id, "adapter failed").expect("fail");
        assert!(tasks.fail(export.id, "again").is_err());
    }

    #[test]
    fn vcs_conflicts_are_diagnostics_and_filterable() {
        let mut catalog = ContentCatalog::new();
        catalog.insert(asset(1, "a.asset", "data")).expect("insert");
        catalog
            .set_vcs_status(
                "a.asset",
                VcsRecord {
                    state: VcsState::Conflicted,
                    ..VcsRecord::default()
                },
            )
            .expect("vcs");
        assert!(catalog.validate().has_errors());
        let page = catalog
            .query_page(
                &AssetQuery {
                    vcs: Some(VcsState::Conflicted),
                    ..AssetQuery::default()
                },
                PageRequest::new(1),
                None,
            )
            .expect("query");
        assert_eq!(ids(&page), vec![1]);
    }

    #[test]
    fn invalid_paths_and_tree_collisions_are_rejected() {
        assert!(AssetRecord::new(
            AssetId::from_raw(1),
            "a/../b",
            AssetKind::new("data").expect("kind")
        )
        .is_err());
        let mut catalog = ContentCatalog::new();
        catalog.insert(asset(1, "folder", "data")).expect("insert");
        assert_eq!(
            catalog.insert(asset(2, "folder/child", "data")),
            Err(ContentError::PathPrefixConflict("folder".into()))
        );
        let mut catalog = ContentCatalog::new();
        catalog
            .insert(asset(1, "old/a.asset", "data"))
            .expect("insert");
        catalog
            .insert(asset(2, "old/b.asset", "data"))
            .expect("insert");
        assert_eq!(
            catalog.update(asset(1, "old", "data")),
            Err(ContentError::PathPrefixConflict("old".into()))
        );
    }
}
