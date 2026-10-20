//! Deterministic content browser and asset-pipeline contracts.

#![forbid(unsafe_code)]

extern crate alloc;

use alloc::{
    collections::{BTreeMap, BTreeSet},
    string::String,
    vec::Vec,
};
use core::fmt;

/// Stable content identity supplied by the asset system or an importer.
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

/// Stable import job identity.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct ImportJobId(u64);

impl ImportJobId {
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

/// Import lifecycle state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImportPhase {
    /// Waiting for an importer slot.
    Queued,
    /// Import work is in progress.
    Running,
    /// Import completed successfully.
    Succeeded,
    /// Import failed in the adapter.
    Failed,
    /// Import was cancelled before completion.
    Cancelled,
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
    /// Direct dependencies.
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
        let path = canonical_path(path.into())?;
        Ok(Self {
            id,
            path,
            kind,
            tags: BTreeSet::new(),
            dependencies: BTreeSet::new(),
            favorite: false,
            pinned: false,
        })
    }
}

fn canonical_path(path: String) -> Result<String, ContentError> {
    let path = path.replace('\\', "/");
    let path = path.trim_matches('/').to_string();
    if path.is_empty()
        || path
            .split('/')
            .any(|part| part.is_empty() || part == "." || part == "..")
    {
        return Err(ContentError::InvalidPath(path));
    }
    Ok(path)
}

/// Content catalog failure.
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
    /// A referenced asset does not exist.
    UnknownAsset(AssetId),
    /// An import job does not exist.
    UnknownJob(ImportJobId),
    /// A job is not in a state where the operation is valid.
    InvalidJobState(ImportJobId),
    /// The import job identity space is exhausted.
    IdExhausted,
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
            Self::UnknownAsset(id) => write!(formatter, "unknown asset {}", id.raw()),
            Self::UnknownJob(id) => write!(formatter, "unknown import job {}", id.raw()),
            Self::InvalidJobState(id) => {
                write!(formatter, "invalid state for import job {}", id.raw())
            }
            Self::IdExhausted => formatter.write_str("content identity space exhausted"),
        }
    }
}

impl core::error::Error for ContentError {}

/// Bounded content-browser query.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ContentQuery {
    /// Case-insensitive path substring.
    pub text: Option<String>,
    /// Exact category filter.
    pub kind: Option<AssetKind>,
    /// Required tag.
    pub tag: Option<String>,
    /// Include only favorites.
    pub favorites_only: bool,
    /// Include only pinned assets.
    pub pinned_only: bool,
    /// Maximum returned records.
    pub limit: usize,
}

/// Deterministic asset catalog and dependency index.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ContentCatalog {
    assets: BTreeMap<AssetId, AssetRecord>,
    paths: BTreeMap<String, AssetId>,
}

impl ContentCatalog {
    /// Creates an empty catalog.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            assets: BTreeMap::new(),
            paths: BTreeMap::new(),
        }
    }

    /// Inserts one asset record.
    pub fn insert(&mut self, record: AssetRecord) -> Result<(), ContentError> {
        if self.assets.contains_key(&record.id) {
            return Err(ContentError::DuplicateAsset(record.id));
        }
        if self.paths.contains_key(&record.path) {
            return Err(ContentError::DuplicatePath(record.path));
        }
        self.paths.insert(record.path.clone(), record.id);
        self.assets.insert(record.id, record);
        Ok(())
    }

    /// Returns an asset by identity.
    #[must_use]
    pub fn get(&self, id: AssetId) -> Option<&AssetRecord> {
        self.assets.get(&id)
    }

    /// Returns all records matching a query in canonical path order.
    #[must_use]
    pub fn query(&self, query: &ContentQuery) -> Vec<&AssetRecord> {
        let text = query.text.as_ref().map(|value| value.to_ascii_lowercase());
        let mut result = Vec::new();
        for id in self.paths.values() {
            let Some(asset) = self.assets.get(id) else {
                continue;
            };
            if let Some(text) = &text
                && !asset.path.to_ascii_lowercase().contains(text)
            {
                continue;
            }
            if query.kind.as_ref().is_some_and(|kind| kind != &asset.kind) {
                continue;
            }
            if query
                .tag
                .as_ref()
                .is_some_and(|tag| !asset.tags.contains(tag))
            {
                continue;
            }
            if query.favorites_only && !asset.favorite {
                continue;
            }
            if query.pinned_only && !asset.pinned {
                continue;
            }
            result.push(asset);
            if query.limit != 0 && result.len() == query.limit {
                break;
            }
        }
        result
    }

    /// Adds a direct dependency while preserving the catalog index.
    pub fn add_dependency(
        &mut self,
        asset: AssetId,
        dependency: AssetId,
    ) -> Result<(), ContentError> {
        if !self.assets.contains_key(&dependency) {
            return Err(ContentError::UnknownAsset(dependency));
        }
        let record = self
            .assets
            .get_mut(&asset)
            .ok_or(ContentError::UnknownAsset(asset))?;
        record.dependencies.insert(dependency);
        Ok(())
    }

    /// Returns assets that directly reference the supplied asset.
    #[must_use]
    pub fn reverse_dependencies(&self, target: AssetId) -> Vec<AssetId> {
        self.assets
            .values()
            .filter(|record| record.dependencies.contains(&target))
            .map(|record| record.id)
            .collect()
    }

    /// Returns the direct reference impact of deleting an asset.
    #[must_use]
    pub fn delete_impact(&self, target: AssetId) -> Vec<AssetId> {
        self.reverse_dependencies(target)
    }

    /// Validates missing references and returns one deterministic cycle, if any.
    pub fn validate(&self) -> Vec<ValidationIssue> {
        let mut issues = Vec::new();
        for record in self.assets.values() {
            for dependency in &record.dependencies {
                if !self.assets.contains_key(dependency) {
                    issues.push(ValidationIssue::MissingDependency {
                        asset: record.id,
                        dependency: *dependency,
                    });
                }
            }
        }
        for id in self.assets.keys() {
            let mut visiting = BTreeSet::new();
            let mut visited = BTreeSet::new();
            if let Some(cycle) = self.find_cycle(*id, &mut visiting, &mut visited) {
                issues.push(ValidationIssue::DependencyCycle { assets: cycle });
                break;
            }
        }
        issues
    }

    fn find_cycle(
        &self,
        id: AssetId,
        visiting: &mut BTreeSet<AssetId>,
        visited: &mut BTreeSet<AssetId>,
    ) -> Option<Vec<AssetId>> {
        if visiting.contains(&id) {
            return Some(vec![id]);
        }
        if !visited.insert(id) {
            return None;
        }
        visiting.insert(id);
        if let Some(record) = self.assets.get(&id) {
            for dependency in &record.dependencies {
                if let Some(mut cycle) = self.find_cycle(*dependency, visiting, visited) {
                    cycle.push(id);
                    return Some(cycle);
                }
            }
        }
        visiting.remove(&id);
        Some(Vec::new()).filter(|cycle| !cycle.is_empty())
    }
}

/// Content validation finding.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ValidationIssue {
    /// A direct reference points to an unknown asset.
    MissingDependency {
        /// Asset that contains the broken reference.
        asset: AssetId,
        /// Referenced identity that is not in the catalog.
        dependency: AssetId,
    },
    /// A dependency cycle prevents a topological import order.
    DependencyCycle {
        /// Cycle members in deterministic traversal order.
        assets: Vec<AssetId>,
    },
}

/// Snapshot of one import job.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ImportJob {
    /// Stable job identity.
    pub id: ImportJobId,
    /// Asset being imported.
    pub asset: AssetId,
    /// Current phase.
    pub phase: ImportPhase,
    /// Progress in thousandths.
    pub progress_milli: u16,
    /// Optional bounded diagnostic message.
    pub message: Option<String>,
}

/// Deterministic, host-driven import job scheduler.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ImportScheduler {
    next_id: u64,
    jobs: BTreeMap<ImportJobId, ImportJob>,
}

impl ImportScheduler {
    /// Creates an empty scheduler.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            next_id: 1,
            jobs: BTreeMap::new(),
        }
    }

    /// Enqueues an asset import.
    pub fn enqueue(&mut self, asset: AssetId) -> Result<ImportJob, ContentError> {
        let id = ImportJobId(self.next_id);
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or(ContentError::IdExhausted)?;
        let job = ImportJob {
            id,
            asset,
            phase: ImportPhase::Queued,
            progress_milli: 0,
            message: None,
        };
        self.jobs.insert(id, job.clone());
        Ok(job)
    }

    /// Advances a queued/running job by a bounded amount.
    pub fn advance(
        &mut self,
        id: ImportJobId,
        amount_milli: u16,
    ) -> Result<ImportJob, ContentError> {
        let job = self.jobs.get_mut(&id).ok_or(ContentError::UnknownJob(id))?;
        if matches!(
            job.phase,
            ImportPhase::Succeeded | ImportPhase::Failed | ImportPhase::Cancelled
        ) {
            return Err(ContentError::InvalidJobState(id));
        }
        job.phase = ImportPhase::Running;
        job.progress_milli = job.progress_milli.saturating_add(amount_milli).min(1000);
        if job.progress_milli == 1000 {
            job.phase = ImportPhase::Succeeded;
        }
        Ok(job.clone())
    }

    /// Cancels a queued or running job.
    pub fn cancel(&mut self, id: ImportJobId) -> Result<ImportJob, ContentError> {
        let job = self.jobs.get_mut(&id).ok_or(ContentError::UnknownJob(id))?;
        if matches!(
            job.phase,
            ImportPhase::Succeeded | ImportPhase::Failed | ImportPhase::Cancelled
        ) {
            return Err(ContentError::InvalidJobState(id));
        }
        job.phase = ImportPhase::Cancelled;
        Ok(job.clone())
    }

    /// Returns a job snapshot.
    #[must_use]
    pub fn get(&self, id: ImportJobId) -> Option<&ImportJob> {
        self.jobs.get(&id)
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

    #[test]
    fn query_is_path_ordered_and_bounded() {
        let mut catalog = ContentCatalog::new();
        catalog
            .insert(asset(2, "textures/z.png", "texture"))
            .expect("insert");
        catalog
            .insert(asset(1, "textures/a.png", "texture"))
            .expect("insert");
        let result = catalog.query(&ContentQuery {
            text: Some("TEXTURES".into()),
            limit: 1,
            ..ContentQuery::default()
        });
        assert_eq!(result[0].id, AssetId::from_raw(1));
    }

    #[test]
    fn dependencies_report_reverse_references_and_cycles() {
        let mut catalog = ContentCatalog::new();
        catalog
            .insert(asset(1, "a.scene", "scene"))
            .expect("insert");
        catalog.insert(asset(2, "b.mesh", "mesh")).expect("insert");
        catalog
            .add_dependency(AssetId::from_raw(1), AssetId::from_raw(2))
            .expect("dependency");
        assert_eq!(
            catalog.reverse_dependencies(AssetId::from_raw(2)),
            vec![AssetId::from_raw(1)]
        );
        catalog
            .add_dependency(AssetId::from_raw(2), AssetId::from_raw(1))
            .expect("cycle");
        assert!(catalog
            .validate()
            .iter()
            .any(|issue| matches!(issue, ValidationIssue::DependencyCycle { .. })));
    }

    #[test]
    fn import_jobs_are_bounded_and_cancellable() {
        let mut scheduler = ImportScheduler::new();
        let job = scheduler.enqueue(AssetId::from_raw(3)).expect("enqueue");
        let running = scheduler.advance(job.id, 400).expect("advance");
        assert_eq!(running.phase, ImportPhase::Running);
        assert_eq!(
            scheduler.cancel(job.id).expect("cancel").phase,
            ImportPhase::Cancelled
        );
        assert!(scheduler.advance(job.id, 100).is_err());
    }
}
