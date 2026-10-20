//! Content browser state transitions over an externally owned catalog.

use std::collections::{BTreeSet, VecDeque};
use std::fmt;

use prism_editor_content::{
    AssetId, AssetKind, ContentCatalog, ContentError, PageRequest, TreeCursor, TreePage, TreeQuery,
    VcsState,
};

use crate::{BoundContentBrowser, SelectionMode, UiEvent, MAX_ACTION_TEXT_BYTES, MAX_VISIBLE_ROWS};

/// Maximum named collections retained by one browser.
pub const MAX_CONTENT_COLLECTIONS: usize = 64;
/// Maximum stable asset identities in one collection.
pub const MAX_COLLECTION_MEMBERS: usize = 4096;

/// Bounded presentation of the current content page.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ContentLayout {
    /// One asset per row.
    #[default]
    List,
    /// Fixed-width cards arranged in one to eight columns.
    Grid {
        /// Number of columns, provided by the host layout policy.
        columns: usize,
    },
}

/// A named set of stable asset identities; persistence belongs to the host.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ContentCollection {
    id: u64,
    name: String,
    members: BTreeSet<AssetId>,
}

impl ContentCollection {
    /// Browser-local identity; not reused after deletion.
    pub const fn id(&self) -> u64 {
        self.id
    }
    /// Display name, independent of asset paths.
    pub fn name(&self) -> &str {
        &self.name
    }
    /// Members, including deleted catalog IDs until explicitly removed.
    pub fn members(&self) -> &BTreeSet<AssetId> {
        &self.members
    }
}

/// Maximum retained selection, including assets on other pages or in other folders.
pub const MAX_CONTENT_SELECTION: usize = 4096;
/// Maximum previous-page cursors retained; older pages remain reachable from Home.
pub const MAX_CONTENT_PAGE_HISTORY: usize = 64;

/// Explicit commands accepted by the content controller; independent of `EditorAction`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ContentBrowserAction {
    /// Change presentation without changing the query or selection.
    Layout(ContentLayout),
    /// Create and open an empty collection with a bounded nonempty name.
    CreateCollection(String),
    /// Rename a collection without changing its identity or members.
    RenameCollection {
        /// Collection identity.
        id: u64,
        /// New display name.
        name: String,
    },
    /// Delete a collection, preserving catalog assets.
    DeleteCollection(u64),
    /// Open a collection, or return to folder browsing.
    OpenCollection(Option<u64>),
    /// Add the current selection atomically to one collection.
    AddSelectionToCollection(u64),
    /// Remove the current selection from one collection.
    RemoveSelectionFromCollection(u64),
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
    /// Collection does not exist, or its name is empty.
    InvalidCollection,
    /// Collection count, member count or identity space exhausted.
    CollectionLimit,
    /// Grid columns must be in the inclusive range 1..=8.
    InvalidGridColumns,
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
            Self::InvalidCollection => f.write_str("unknown collection or empty collection name"),
            Self::CollectionLimit => f.write_str("content collection limit exceeded"),
            Self::InvalidGridColumns => f.write_str("content grid needs one to eight columns"),
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
    layout: ContentLayout,
    collections: Vec<ContentCollection>,
    next_collection: u64,
    collection: Option<u64>,
    collection_offset: usize,
    collection_has_next: bool,
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
            layout: ContentLayout::List,
            collections: Vec::new(),
            next_collection: 1,
            collection: None,
            collection_offset: 0,
            collection_has_next: false,
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
    /// Current list/grid presentation.
    pub const fn layout(&self) -> ContentLayout {
        self.layout
    }
    /// Named sets owned by this controller; host persists their snapshots.
    pub fn collections(&self) -> &[ContentCollection] {
        &self.collections
    }
    /// Open collection, or None for folders.
    pub const fn active_collection(&self) -> Option<u64> {
        self.collection
    }
    /// Stable identities on the most recently refreshed page.
    pub fn visible_assets(&self) -> &[AssetId] {
        &self.visible
    }
    /// Catalog generation used to build the current page, if any.
    pub const fn generation(&self) -> Option<u64> {
        self.generation
    }
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
        if self.collection.is_some() {
            self.collection_offset
        } else {
            self.cursor.as_ref().map_or(0, TreeCursor::offset)
        }
    }

    /// Builds a real keyed Loom panel and its content-specific event bindings.
    /// A new catalog generation resets cursors, prunes deleted selections, and
    /// falls back to root if the open folder disappeared.
    pub fn build(
        &mut self,
        catalog: &mut ContentCatalog,
    ) -> Result<BoundContentBrowser, ContentBrowserError> {
        let page = self.refresh(catalog)?;
        Ok(crate::content_view::build(self, catalog, &page, None))
    }

    /// Builds image-layout nodes and exposes real ready pixels through
    /// `BoundContentBrowser::images`. Host projects the Loom layout and submits
    /// images to its compositor; queued/failed requests expose status only.
    pub fn build_with_thumbnails(
        &mut self,
        catalog: &mut ContentCatalog,
        thumbnails: &crate::ContentThumbnails,
    ) -> Result<BoundContentBrowser, ContentBrowserError> {
        let page = self.refresh(catalog)?;
        Ok(crate::content_view::build(
            self,
            catalog,
            &page,
            Some(thumbnails),
        ))
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
            ContentBrowserAction::Layout(layout) => {
                if matches!(layout, ContentLayout::Grid { columns } if !(1..=8).contains(&columns))
                {
                    return Err(ContentBrowserError::InvalidGridColumns);
                }
                self.layout = layout;
            }
            ContentBrowserAction::CreateCollection(name) => {
                let name = collection_name(name)?;
                if self.collections.len() == MAX_CONTENT_COLLECTIONS
                    || self.next_collection == u64::MAX
                {
                    return Err(ContentBrowserError::CollectionLimit);
                }
                let id = self.next_collection;
                self.next_collection += 1;
                self.collections.push(ContentCollection {
                    id,
                    name,
                    members: BTreeSet::new(),
                });
                self.collection = Some(id);
                self.reset_page();
            }
            ContentBrowserAction::RenameCollection { id, name } => {
                let name = collection_name(name)?;
                self.collection_mut(id)?.name = name;
            }
            ContentBrowserAction::DeleteCollection(id) => {
                self.collection_mut(id)?;
                self.collections.retain(|collection| collection.id != id);
                if self.collection == Some(id) {
                    self.collection = None;
                }
                self.reset_page();
            }
            ContentBrowserAction::OpenCollection(id) => {
                if let Some(id) = id {
                    self.collection_mut(id)?;
                }
                self.collection = id;
                self.reset_page();
            }
            ContentBrowserAction::AddSelectionToCollection(id) => {
                let selection = self.selected.clone();
                let collection = self.collection_mut(id)?;
                if collection.members.union(&selection).count() > MAX_COLLECTION_MEMBERS {
                    return Err(ContentBrowserError::CollectionLimit);
                }
                collection.members.extend(selection);
                self.reset_page();
            }
            ContentBrowserAction::RemoveSelectionFromCollection(id) => {
                let selection = self.selected.clone();
                self.collection_mut(id)?
                    .members
                    .retain(|asset| !selection.contains(asset));
                self.reset_page();
            }
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
                self.collection = None;
                self.query = query;
                self.reset_page();
            }
            ContentBrowserAction::NextPage => {
                if self.collection.is_some() {
                    if self.collection_has_next {
                        self.collection_offset += self.request(catalog).limit;
                    }
                } else if let Some(next) = self.next.take() {
                    if self.history.len() == MAX_CONTENT_PAGE_HISTORY {
                        self.history.pop_front();
                    }
                    self.history.push_back(self.cursor.take());
                    self.cursor = Some(next);
                }
            }
            ContentBrowserAction::PreviousPage => {
                if self.collection.is_some() {
                    self.collection_offset = self
                        .collection_offset
                        .saturating_sub(self.request(catalog).limit);
                } else if let Some(previous) = self.history.pop_back() {
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
        if self.collection.is_some() {
            self.collection_offset > 0
        } else {
            !self.history.is_empty()
        }
    }

    pub(crate) fn has_next(&self, page: &TreePage) -> bool {
        if self.collection.is_some() {
            self.collection_has_next
        } else {
            page.next.is_some()
        }
    }

    fn collection_mut(&mut self, id: u64) -> Result<&mut ContentCollection, ContentBrowserError> {
        self.collections
            .iter_mut()
            .find(|collection| collection.id == id)
            .ok_or(ContentBrowserError::InvalidCollection)
    }

    fn request(&self, catalog: &ContentCatalog) -> PageRequest {
        PageRequest::new(self.page_size.min(catalog.limits().max_page_size))
    }

    fn reset_page(&mut self) {
        self.collection_offset = 0;
        self.collection_has_next = false;
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
        if let Some(id) = self.collection {
            let members = &self
                .collections
                .iter()
                .find(|collection| collection.id == id)
                .ok_or(ContentBrowserError::InvalidCollection)?
                .members;
            let mut records: Vec<_> = members
                .iter()
                .filter_map(|id| catalog.get(*id))
                .filter(|record| crate::content_collection::matches(catalog, &self.query, record))
                .collect();
            records.sort_by(|a, b| a.path.cmp(&b.path).then(a.id.cmp(&b.id)));
            let end = self
                .collection_offset
                .saturating_add(self.request(catalog).limit);
            self.collection_has_next = end < records.len();
            let items: Vec<_> = records
                .into_iter()
                .skip(self.collection_offset)
                .take(self.request(catalog).limit)
                .map(|record| prism_editor_content::TreeEntry {
                    path: record.path.clone(),
                    name: record.path.clone(),
                    kind: prism_editor_content::TreeEntryKind::Asset,
                    asset: Some(record.id),
                    descendant_assets: 0,
                })
                .collect();
            self.visible = items.iter().filter_map(|entry| entry.asset).collect();
            return Ok(TreePage {
                items,
                next: None,
                cache: prism_editor_content::CacheStatus::Miss,
            });
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

fn collection_name(name: String) -> Result<String, ContentBrowserError> {
    check_text(&name)?;
    if name.trim().is_empty() {
        return Err(ContentBrowserError::InvalidCollection);
    }
    Ok(name.trim().to_owned())
}

fn nonempty(value: String) -> Option<String> {
    (!value.trim().is_empty()).then_some(value)
}
