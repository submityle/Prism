//! Content browser state transitions over an externally owned catalog.

use std::collections::{BTreeSet, VecDeque};
use std::fmt;

use prism_editor_content::{
    AssetId, AssetKind, ContentCatalog, ContentError, PageRequest, TreeCursor, TreePage, TreeQuery,
    VcsState,
};

use crate::{BoundContentBrowser, SelectionMode, UiEvent, MAX_ACTION_TEXT_BYTES, MAX_VISIBLE_ROWS};

/// Maximum retained selection, including assets on other pages or in other folders.
pub const MAX_CONTENT_SELECTION: usize = 4096;
/// Maximum previous-page cursors retained; older pages remain reachable from Home.
pub const MAX_CONTENT_PAGE_HISTORY: usize = 64;

/// Explicit commands accepted by the content controller; independent of `EditorAction`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ContentBrowserAction {
    /// Search paths, kinds and tags using catalog search semantics.
    Search(String),
    /// Filter by exact kind; `None` clears the filter.
    Kind(Option<AssetKind>),
    /// Filter by normalized tag; `None` clears the filter.
    Tag(Option<String>),
    /// Filter by supplied VCS state; unknown is distinct from clean.
    Vcs(Option<VcsState>),
    /// Include favorites only.
    FavoritesOnly(bool),
    /// Include pinned assets only.
    PinnedOnly(bool),
    /// Open a canonical folder; the empty string opens the catalog root.
    Navigate(String),
    /// Advance one bounded catalog page.
    NextPage,
    /// Return to a retained previous page.
    PreviousPage,
    /// Set a nonzero page size, capped by the UI and catalog limits.
    PageSize(usize),
    /// Select a visible asset with the shared modifier policy.
    Select {
        /// Stable catalog identity.
        asset: AssetId,
        /// Extend selects the inclusive range within the current page.
        mode: SelectionMode,
    },
    /// Clear selection across all pages.
    ClearSelection,
    /// Change one asset's favorite flag through the catalog update API.
    SetFavorite {
        /// Stable catalog identity.
        asset: AssetId,
        /// Desired flag value.
        value: bool,
    },
    /// Change one asset's pinned flag through the catalog update API.
    SetPinned {
        /// Stable catalog identity.
        asset: AssetId,
        /// Desired flag value.
        value: bool,
    },
}

/// Failures leave the requested mutation unapplied.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ContentBrowserError {
    /// Catalog validation or paging failed.
    Catalog(ContentError),
    /// The event did not match an interactive key and trigger.
    UnboundEvent,
    /// A committed value has the wrong type or an unsupported VCS name.
    InvalidValue,
    /// Search, kind, tag or folder text exceeded the UI byte limit.
    TextLimit,
    /// A zero page size is invalid.
    InvalidPageSize,
    /// Selection would exceed the hard bound.
    SelectionLimit,
    /// Selection targeted an asset outside the current visible page.
    AssetNotVisible,
    /// Rebuild after a catalog or controller change before routing more events.
    StaleView,
}

impl fmt::Display for ContentBrowserError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Catalog(error) => error.fmt(f),
            Self::UnboundEvent => f.write_str("unbound content browser event"),
            Self::InvalidValue => f.write_str("invalid content browser input"),
            Self::TextLimit => f.write_str("content browser text limit exceeded"),
            Self::InvalidPageSize => f.write_str("content page size must be nonzero"),
            Self::SelectionLimit => f.write_str("content selection limit exceeded"),
            Self::AssetNotVisible => f.write_str("content asset is not on the current page"),
            Self::StaleView => f.write_str("content browser view must be rebuilt"),
        }
    }
}

impl std::error::Error for ContentBrowserError {}

impl From<ContentError> for ContentBrowserError {
    fn from(error: ContentError) -> Self {
        Self::Catalog(error)
    }
}

/// Owns only query, paging and selection state. The host owns the catalog,
/// source workers, persistence, Loom runtime and native input adapter.
#[derive(Clone, Debug)]
pub struct ContentBrowserController {
    query: TreeQuery,
    selected: BTreeSet<AssetId>,
    anchor: Option<AssetId>,
    page_size: usize,
    cursor: Option<TreeCursor>,
    next: Option<TreeCursor>,
    history: VecDeque<Option<TreeCursor>>,
    visible: Vec<AssetId>,
    generation: Option<u64>,
    revision: u64,
}

impl Default for ContentBrowserController {
    fn default() -> Self {
        Self {
            query: TreeQuery::default(),
            selected: BTreeSet::new(),
            anchor: None,
            page_size: 64,
            cursor: None,
            next: None,
            history: VecDeque::new(),
            visible: Vec::new(),
            generation: None,
            revision: 0,
        }
    }
}

impl ContentBrowserController {
    /// Returns the current query without permitting direct mutation.
    #[must_use]
    pub fn query(&self) -> &TreeQuery {
        &self.query
    }

    /// Returns the bounded selection, including hidden but still existing assets.
    #[must_use]
    pub fn selection(&self) -> &BTreeSet<AssetId> {
        &self.selected
    }

    /// Returns the current matching-row offset (folders count as rows).
    #[must_use]
    pub fn page_offset(&self) -> usize {
        self.cursor.as_ref().map_or(0, TreeCursor::offset)
    }

    /// Builds a real keyed Loom panel and its content-specific event bindings.
    /// A new catalog generation resets cursors, prunes deleted selections, and
    /// falls back to root if the open folder disappeared.
    pub fn build(
        &mut self,
        catalog: &mut ContentCatalog,
    ) -> Result<BoundContentBrowser, ContentBrowserError> {
        let page = self.refresh(catalog)?;
        Ok(crate::content_view::build(self, catalog, &page))
    }

    /// Routes an event from the currently displayed bound tree and applies its
    /// action. Stale views are rejected before any catalog or controller write.
    pub fn handle_event(
        &mut self,
        catalog: &mut ContentCatalog,
        view: &BoundContentBrowser,
        target: &str,
        event: &UiEvent,
    ) -> Result<ContentBrowserAction, ContentBrowserError> {
        if view.generation() != catalog.generation() || view.revision() != self.revision {
            return Err(ContentBrowserError::StaleView);
        }
        let action = view.dispatch(target, event)?;
        self.apply(catalog, action.clone())?;
        Ok(action)
    }

    /// Applies an explicit host command. Favorite and pinned writes clone the
    /// latest record and call `ContentCatalog::update`, preserving other fields.
    pub fn apply(
        &mut self,
        catalog: &mut ContentCatalog,
        action: ContentBrowserAction,
    ) -> Result<(), ContentBrowserError> {
        self.refresh(catalog)?;
        match action {
            ContentBrowserAction::Search(value) => {
                check_text(&value)?;
                self.query.text = nonempty(value);
                self.reset_page();
            }
            ContentBrowserAction::Kind(value) => {
                if let Some(kind) = &value {
                    check_text(kind.as_str())?;
                }
                self.query.kind = value;
                self.reset_page();
            }
            ContentBrowserAction::Tag(value) => {
                if let Some(tag) = &value {
                    check_text(tag)?;
                }
                self.query.tag = value.and_then(nonempty);
                self.reset_page();
            }
            ContentBrowserAction::Vcs(value) => {
                self.query.vcs = value;
                self.reset_page();
            }
            ContentBrowserAction::FavoritesOnly(value) => {
                self.query.favorites_only = value;
                self.reset_page();
            }
            ContentBrowserAction::PinnedOnly(value) => {
                self.query.pinned_only = value;
                self.reset_page();
            }
            ContentBrowserAction::Navigate(root) => {
                check_text(&root)?;
                let query = TreeQuery {
                    root,
                    ..self.query.clone()
                };
                catalog.tree_page(&query, self.request(catalog), None)?;
                self.query = query;
                self.reset_page();
            }
            ContentBrowserAction::NextPage => {
                if let Some(next) = self.next.take() {
                    if self.history.len() == MAX_CONTENT_PAGE_HISTORY {
                        self.history.pop_front();
                    }
                    self.history.push_back(self.cursor.take());
                    self.cursor = Some(next);
                }
            }
            ContentBrowserAction::PreviousPage => {
                if let Some(previous) = self.history.pop_back() {
                    self.cursor = previous;
                    self.next = None;
                }
            }
            ContentBrowserAction::PageSize(size) => {
                if size == 0 {
                    return Err(ContentBrowserError::InvalidPageSize);
                }
                self.page_size = size.min(MAX_VISIBLE_ROWS);
                self.reset_page();
            }
            ContentBrowserAction::Select { asset, mode } => self.select(asset, mode)?,
            ContentBrowserAction::ClearSelection => {
                self.selected.clear();
                self.anchor = None;
            }
            ContentBrowserAction::SetFavorite { asset, value } => {
                let mut record = catalog
                    .get(asset)
                    .cloned()
                    .ok_or(ContentError::UnknownAsset(asset))?;
                record.favorite = value;
                catalog.update(record)?;
            }
            ContentBrowserAction::SetPinned { asset, value } => {
                let mut record = catalog
                    .get(asset)
                    .cloned()
                    .ok_or(ContentError::UnknownAsset(asset))?;
                record.pinned = value;
                catalog.update(record)?;
            }
        }
        self.revision = self.revision.wrapping_add(1);
        Ok(())
    }

    pub(crate) const fn revision(&self) -> u64 {
        self.revision
    }
    pub(crate) fn has_previous(&self) -> bool {
        !self.history.is_empty()
    }

    fn request(&self, catalog: &ContentCatalog) -> PageRequest {
        PageRequest::new(self.page_size.min(catalog.limits().max_page_size))
    }

    fn reset_page(&mut self) {
        self.cursor = None;
        self.next = None;
        self.history.clear();
        self.visible.clear();
    }

    fn refresh(&mut self, catalog: &mut ContentCatalog) -> Result<TreePage, ContentBrowserError> {
        if self.generation != Some(catalog.generation()) {
            self.reset_page();
            self.selected.retain(|id| catalog.get(*id).is_some());
            if self.anchor.is_some_and(|id| !self.selected.contains(&id)) {
                self.anchor = None;
            }
            self.generation = Some(catalog.generation());
            self.revision = self.revision.wrapping_add(1);
        }
        let page = match catalog.tree_page(&self.query, self.request(catalog), self.cursor.clone())
        {
            Err(ContentError::UnknownPath(_)) if !self.query.root.is_empty() => {
                self.query.root.clear();
                self.reset_page();
                self.revision = self.revision.wrapping_add(1);
                catalog.tree_page(&self.query, self.request(catalog), None)?
            }
            result => result?,
        };
        self.next = page.next.clone();
        self.visible = page.items.iter().filter_map(|entry| entry.asset).collect();
        Ok(page)
    }

    fn select(&mut self, asset: AssetId, mode: SelectionMode) -> Result<(), ContentBrowserError> {
        let target = self
            .visible
            .iter()
            .position(|id| *id == asset)
            .ok_or(ContentBrowserError::AssetNotVisible)?;
        let mut selected = self.selected.clone();
        match mode {
            SelectionMode::Replace => {
                selected.clear();
                selected.insert(asset);
            }
            SelectionMode::Add => {
                selected.insert(asset);
            }
            SelectionMode::Toggle => {
                if !selected.remove(&asset) {
                    selected.insert(asset);
                }
            }
            SelectionMode::Extend => {
                let anchor = self
                    .anchor
                    .and_then(|id| self.visible.iter().position(|v| *v == id))
                    .unwrap_or(target);
                selected.clear();
                selected.extend(&self.visible[anchor.min(target)..=anchor.max(target)]);
            }
        }
        if selected.len() > MAX_CONTENT_SELECTION {
            return Err(ContentBrowserError::SelectionLimit);
        }
        self.selected = selected;
        if mode != SelectionMode::Extend || self.anchor.is_none() {
            self.anchor = Some(asset);
        }
        Ok(())
    }
}

pub(crate) fn check_text(value: &str) -> Result<(), ContentBrowserError> {
    if value.len() > MAX_ACTION_TEXT_BYTES {
        Err(ContentBrowserError::TextLimit)
    } else {
        Ok(())
    }
}

fn nonempty(value: String) -> Option<String> {
    (!value.trim().is_empty()).then_some(value)
}
