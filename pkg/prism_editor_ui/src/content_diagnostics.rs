//! Explicit dependency navigation and validation report commands.

use std::{collections::VecDeque, fmt, sync::Arc};

use prism_editor_content::{
    validation::{
        ValidationError, ValidationOperation, ValidationPolicy, ValidationReceipt,
        ValidationRegistry, ValidationRejection,
    },
    AssetId, ContentCatalog, ContentError,
};

use crate::{BoundContentDiagnostics, UiEvent, MAX_VISIBLE_ROWS};

/// Maximum retained dependency navigation entries.
pub const MAX_DEPENDENCY_NAVIGATION_HISTORY: usize = 64;

/// Direction of the catalog's bounded transitive traversal.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum DependencyDirection {
    /// Assets referenced by the selected asset, including missing targets.
    #[default]
    Dependencies,
    /// Assets that transitively reference the selected asset.
    Dependents,
}

/// Owned commands independent of the general editor action enum.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ContentDiagnosticsAction {
    /// Synchronize host selection, or navigate to an existing asset.
    Focus(Option<AssetId>),
    /// Return to the most recent retained asset.
    Back,
    /// Switch upstream/downstream traversal and reset its page.
    Direction(DependencyDirection),
    /// Advance the dependency list.
    NextDependencies,
    /// Return one dependency page.
    PreviousDependencies,
    /// Advance the retained validation findings.
    NextIssues,
    /// Return one findings page.
    PreviousIssues,
    /// Set rows per list (nonzero, capped at the UI row limit).
    PageSize(usize),
    /// Run actual built-in and contributed checks once, for this operation.
    RunValidation(ValidationOperation),
    /// Locate a finding by its index in the current immutable receipt.
    LocateIssue(usize),
}

/// An action rejected before changing UI state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ContentDiagnosticsError {
    /// Catalog target does not exist.
    Catalog(ContentError),
    /// The registry could not produce a receipt.
    Validation(ValidationError),
    /// No interactive target matches the semantic event.
    UnboundEvent,
    /// Rebuild after a controller, catalog or registry change.
    StaleView,
    /// Page sizes must be nonzero.
    InvalidPageSize,
    /// Finding is absent, has no existing asset, or belongs to a stale receipt.
    UnlocatableIssue,
}

impl fmt::Display for ContentDiagnosticsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Catalog(error) => error.fmt(f),
            Self::Validation(error) => error.fmt(f),
            Self::UnboundEvent => f.write_str("unbound content diagnostics event"),
            Self::StaleView => f.write_str("content diagnostics view must be rebuilt"),
            Self::InvalidPageSize => f.write_str("diagnostics page size must be nonzero"),
            Self::UnlocatableIssue => f.write_str("finding has no current asset location"),
        }
    }
}
impl std::error::Error for ContentDiagnosticsError {}

/// Owns bounded navigation state and one genuine validation receipt. The host
/// owns the catalog, registry, native event adapter and operation authorization.
pub struct ContentDiagnosticsController {
    pub(crate) selected: Option<AssetId>,
    pub(crate) direction: DependencyDirection,
    pub(crate) history: VecDeque<Option<AssetId>>,
    pub(crate) page_size: usize,
    pub(crate) dependency_offset: usize,
    pub(crate) issue_offset: usize,
    pub(crate) receipt: Option<ValidationReceipt>,
    pub(crate) operation: ValidationOperation,
    generation: Option<u64>,
    pub(crate) revision: u64,
    pub(crate) identity: Arc<()>,
}

impl Default for ContentDiagnosticsController {
    fn default() -> Self {
        Self {
            selected: None,
            direction: DependencyDirection::default(),
            history: VecDeque::new(),
            page_size: 64,
            dependency_offset: 0,
            issue_offset: 0,
            receipt: None,
            operation: ValidationOperation::Save,
            generation: None,
            revision: 0,
            identity: Arc::new(()),
        }
    }
}

impl ContentDiagnosticsController {
    /// Current stable asset identity, also used by hosts after a locate action.
    #[must_use]
    pub const fn selected_asset(&self) -> Option<AssetId> {
        self.selected
    }

    /// Actual retained receipt. Hosts must call its own `authorize` immediately
    /// before preparing an operation; rendering a report authorizes no work.
    #[must_use]
    pub const fn receipt(&self) -> Option<&ValidationReceipt> {
        self.receipt.as_ref()
    }

    /// Current receipt gate result under the default policy. No contributed
    /// checks execute here; freshness comes from the receipt's identity fence.
    #[must_use]
    pub fn report_status(
        &self,
        catalog: &ContentCatalog,
        registry: &ValidationRegistry,
    ) -> Option<Result<(), ValidationRejection>> {
        self.receipt.as_ref().map(|receipt| {
            receipt
                .authorize(
                    catalog,
                    registry,
                    self.operation,
                    ValidationPolicy::default(),
                )
                .map(|_| ())
        })
    }

    /// Build the actual Loom panel. Catalog traversal failures are visible
    /// states, never a silently partial or apparently empty dependency graph.
    pub fn build(
        &mut self,
        catalog: &ContentCatalog,
        registry: &ValidationRegistry,
    ) -> BoundContentDiagnostics {
        self.refresh(catalog);
        let dependencies = self.dependencies(catalog);
        crate::content_diagnostics_view::build(self, catalog, registry, dependencies)
    }

    /// Resolve an event from the displayed Element and apply its owned command.
    pub fn handle_event(
        &mut self,
        catalog: &ContentCatalog,
        registry: &ValidationRegistry,
        view: &BoundContentDiagnostics,
        target: &str,
        event: &UiEvent,
    ) -> Result<ContentDiagnosticsAction, ContentDiagnosticsError> {
        if !Arc::ptr_eq(&view.identity, &self.identity)
            || view.generation() != catalog.generation()
            || view.revision != self.revision
            || view.status != self.report_status(catalog, registry)
        {
            return Err(ContentDiagnosticsError::StaleView);
        }
        let action = view.dispatch(target, event)?;
        self.apply(catalog, registry, action.clone())?;
        Ok(action)
    }

    /// Apply an explicit host command, borrowing catalog and registry read-only.
    /// Only RunValidation invokes contributed rules; paging never reruns them.
    pub fn apply(
        &mut self,
        catalog: &ContentCatalog,
        registry: &ValidationRegistry,
        action: ContentDiagnosticsAction,
    ) -> Result<(), ContentDiagnosticsError> {
        // Validate fallible commands before touching retained state.
        let receipt = match action {
            ContentDiagnosticsAction::RunValidation(operation) => Some(
                registry
                    .validate(catalog, operation)
                    .map_err(ContentDiagnosticsError::Validation)?,
            ),
            _ => None,
        };
        let focus = match action {
            ContentDiagnosticsAction::Focus(Some(id)) => Some(id),
            ContentDiagnosticsAction::LocateIssue(index) => {
                if !self.report_is_current(catalog, registry) {
                    return Err(ContentDiagnosticsError::UnlocatableIssue);
                }
                Some(
                    self.receipt
                        .as_ref()
                        .and_then(|r| r.issues().get(index))
                        .and_then(|issue| issue.asset)
                        .filter(|id| catalog.get(*id).is_some())
                        .ok_or(ContentDiagnosticsError::UnlocatableIssue)?,
                )
            }
            ContentDiagnosticsAction::PageSize(0) => {
                return Err(ContentDiagnosticsError::InvalidPageSize)
            }
            _ => None,
        };
        if let Some(id) = focus {
            catalog.get(id).ok_or(ContentDiagnosticsError::Catalog(
                ContentError::UnknownAsset(id),
            ))?;
        }
        self.refresh(catalog);
        match action {
            ContentDiagnosticsAction::Focus(id) => self.focus(id),
            ContentDiagnosticsAction::LocateIssue(_) => self.focus(focus),
            ContentDiagnosticsAction::Back => {
                if let Some(id) = self.history.pop_back() {
                    self.selected = id;
                    self.dependency_offset = 0;
                }
            }
            ContentDiagnosticsAction::Direction(direction) => {
                self.direction = direction;
                self.dependency_offset = 0;
            }
            ContentDiagnosticsAction::NextDependencies => {
                if let Ok(ids) = self.dependencies(catalog)
                    && self.dependency_offset.saturating_add(self.page_size) < ids.len()
                {
                    self.dependency_offset += self.page_size;
                }
            }
            ContentDiagnosticsAction::PreviousDependencies => {
                self.dependency_offset = self.dependency_offset.saturating_sub(self.page_size);
            }
            ContentDiagnosticsAction::NextIssues => {
                if self.receipt.as_ref().is_some_and(|r| {
                    self.issue_offset.saturating_add(self.page_size) < r.issues().len()
                }) {
                    self.issue_offset += self.page_size;
                }
            }
            ContentDiagnosticsAction::PreviousIssues => {
                self.issue_offset = self.issue_offset.saturating_sub(self.page_size);
            }
            ContentDiagnosticsAction::PageSize(size) => {
                self.page_size = size.min(MAX_VISIBLE_ROWS);
                self.dependency_offset = 0;
                self.issue_offset = 0;
            }
            ContentDiagnosticsAction::RunValidation(operation) => {
                self.receipt = receipt;
                self.operation = operation;
                self.issue_offset = 0;
            }
        }
        self.revision = self.revision.wrapping_add(1);
        Ok(())
    }

    pub(crate) fn report_is_current(
        &self,
        catalog: &ContentCatalog,
        registry: &ValidationRegistry,
    ) -> bool {
        matches!(
            self.report_status(catalog, registry),
            Some(
                Ok(())
                    | Err(ValidationRejection::Errors
                        | ValidationRejection::Warnings
                        | ValidationRejection::Incomplete)
            )
        )
    }

    fn refresh(&mut self, catalog: &ContentCatalog) {
        if self.generation != Some(catalog.generation()) {
            self.generation = Some(catalog.generation());
            self.dependency_offset = 0;
            self.revision = self.revision.wrapping_add(1);
        }
    }

    fn focus(&mut self, id: Option<AssetId>) {
        if self.selected != id {
            if self.history.len() == MAX_DEPENDENCY_NAVIGATION_HISTORY {
                self.history.pop_front();
            }
            self.history.push_back(self.selected);
            self.selected = id;
            self.dependency_offset = 0;
        }
    }

    fn dependencies(&self, catalog: &ContentCatalog) -> Result<Vec<AssetId>, ContentError> {
        let Some(root) = self.selected else {
            return Ok(Vec::new());
        };
        let mut ids = match self.direction {
            DependencyDirection::Dependencies => catalog.dependency_closure(root)?,
            DependencyDirection::Dependents => catalog.dependent_closure(root)?,
        };
        ids.sort_unstable();
        Ok(ids)
    }
}
