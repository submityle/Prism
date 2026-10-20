//! Real Loom content rows and semantic event recipes.

use std::collections::BTreeMap;

use prism_editor_content::{AssetId, AssetKind, ContentCatalog, TreeEntryKind, TreePage, VcsState};
use prism_ui::{
    style::{Keyword, StyleProp, StyleValue},
    Element,
};
use prism_ui_component::Component;
use prism_ui_component_kit::inputs::{SearchField, SearchFieldProps};

use crate::content::check_text;
use crate::{
    ContentBrowserAction, ContentBrowserController, ContentBrowserError, UiEvent, UiTrigger,
};

#[derive(Clone, Debug)]
enum Recipe {
    Action(ContentBrowserAction),
    Select(AssetId),
    Search,
    Kind,
    Tag,
    Vcs,
}

/// An owned Element tree paired with the exact interactive keys it contains.
/// Hosts can mount `element()` inside a dock body, retain this value for input,
/// then rebuild after every accepted action or catalog refresh.
#[derive(Clone, Debug)]
pub struct BoundContentBrowser {
    element: Element,
    bindings: BTreeMap<(String, UiTrigger), Recipe>,
    generation: u64,
    revision: u64,
}

impl BoundContentBrowser {
    /// Borrows the real panel tree to mount or reconcile in Loom.
    #[must_use]
    pub fn element(&self) -> &Element {
        &self.element
    }

    /// Returns exact keyed targets and accepted triggers in deterministic order.
    /// Disabled controls are rendered but have no binding.
    pub fn bindings(&self) -> impl Iterator<Item = (&str, UiTrigger)> {
        self.bindings
            .keys()
            .map(|(key, trigger)| (key.as_str(), *trigger))
    }

    /// Returns the catalog generation represented by this tree.
    #[must_use]
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    pub(crate) const fn revision(&self) -> u64 {
        self.revision
    }

    /// Resolves a semantic event to an owned action without mutating state.
    /// Prefer `ContentBrowserController::handle_event` for freshness validation
    /// and execution against the currently displayed catalog.
    pub fn dispatch(
        &self,
        target: &str,
        event: &UiEvent,
    ) -> Result<ContentBrowserAction, ContentBrowserError> {
        let trigger = match event {
            UiEvent::Activate => UiTrigger::Activate,
            UiEvent::Select { .. } => UiTrigger::Select,
            UiEvent::Commit { .. } => UiTrigger::Commit,
        };
        let recipe = self
            .bindings
            .get(&(target.to_owned(), trigger))
            .ok_or(ContentBrowserError::UnboundEvent)?;
        match (recipe, event) {
            (Recipe::Action(action), UiEvent::Activate) => Ok(action.clone()),
            (Recipe::Select(asset), UiEvent::Select { mode }) => Ok(ContentBrowserAction::Select {
                asset: *asset,
                mode: *mode,
            }),
            (recipe, UiEvent::Commit { value }) => {
                let text = value.as_text().ok_or(ContentBrowserError::InvalidValue)?;
                check_text(text)?;
                let optional = (!text.trim().is_empty()).then(|| text.trim().to_owned());
                match recipe {
                    Recipe::Search => Ok(ContentBrowserAction::Search(text.to_owned())),
                    Recipe::Kind => Ok(ContentBrowserAction::Kind(
                        optional.map(AssetKind::new).transpose()?,
                    )),
                    Recipe::Tag => Ok(ContentBrowserAction::Tag(optional)),
                    Recipe::Vcs => Ok(ContentBrowserAction::Vcs(parse_vcs(text)?)),
                    _ => Err(ContentBrowserError::UnboundEvent),
                }
            }
            _ => Err(ContentBrowserError::UnboundEvent),
        }
    }
}

pub(crate) fn build(
    controller: &ContentBrowserController,
    catalog: &ContentCatalog,
    page: &TreePage,
) -> BoundContentBrowser {
    let mut bound = BoundContentBrowser {
        element: Element::box_(),
        bindings: BTreeMap::new(),
        generation: catalog.generation(),
        revision: controller.revision(),
    };
    let query = controller.query();
    let header = column("prism-editor-content__header")
        .child(Element::text("Content"))
        .child(field(
            &mut bound,
            "content:search",
            "Search content",
            query.text.as_deref().unwrap_or(""),
            Recipe::Search,
        ))
        .child(
            row("prism-editor-content__filters")
                .child(field(
                    &mut bound,
                    "content:kind",
                    "Kind (empty: all)",
                    query.kind.as_ref().map_or("", AssetKind::as_str),
                    Recipe::Kind,
                ))
                .child(field(
                    &mut bound,
                    "content:tag",
                    "Tag (empty: all)",
                    query.tag.as_deref().unwrap_or(""),
                    Recipe::Tag,
                ))
                .child(field(
                    &mut bound,
                    "content:vcs",
                    "VCS (empty: all)",
                    query.vcs.map_or("", vcs_name),
                    Recipe::Vcs,
                )),
        )
        .child(
            row("prism-editor-content__toggles")
                .child(button(
                    &mut bound,
                    "content:favorites-only",
                    if query.favorites_only {
                        "Favorites only: on"
                    } else {
                        "Favorites only: off"
                    },
                    Some(ContentBrowserAction::FavoritesOnly(!query.favorites_only)),
                ))
                .child(button(
                    &mut bound,
                    "content:pinned-only",
                    if query.pinned_only {
                        "Pinned only: on"
                    } else {
                        "Pinned only: off"
                    },
                    Some(ContentBrowserAction::PinnedOnly(!query.pinned_only)),
                )),
        );

    let parent = query.root.rsplit_once('/').map_or("", |(parent, _)| parent);
    let navigation = row("prism-editor-content__navigation")
        .child(button(
            &mut bound,
            "content:home",
            "Home",
            Some(ContentBrowserAction::Navigate(String::new())),
        ))
        .child(button(
            &mut bound,
            "content:up",
            "Up",
            (!query.root.is_empty()).then(|| ContentBrowserAction::Navigate(parent.to_owned())),
        ))
        .child(
            Element::text(if query.root.is_empty() {
                "/"
            } else {
                &query.root
            })
            .class("prism-editor-content__path"),
        );

    let mut list = column("prism-editor-content__list").key_str("content:list");
    for entry in &page.items {
        let item = match entry.kind {
            TreeEntryKind::Folder => button(
                &mut bound,
                &format!("content:folder:{}", entry.path),
                &format!(
                    "[Folder] {} ({} assets)",
                    entry.name, entry.descendant_assets
                ),
                Some(ContentBrowserAction::Navigate(entry.path.clone())),
            )
            .class("prism-editor-content__folder"),
            TreeEntryKind::Asset => {
                let id = entry.asset.expect("catalog asset entry has an identity");
                let record = catalog
                    .get(id)
                    .expect("page and catalog share a generation");
                let key = format!("content:asset:{}", id.raw());
                bound
                    .bindings
                    .insert((key.clone(), UiTrigger::Select), Recipe::Select(id));
                let selected = controller.selection().contains(&id);
                let vcs = catalog.vcs_status(id).unwrap_or_default();
                row("prism-editor-content__asset")
                    .key_str(key)
                    .class(if selected {
                        "is-selected"
                    } else {
                        "is-unselected"
                    })
                    .style(
                        StyleProp::BackgroundColor,
                        StyleValue::token(if selected {
                            "color.fill.secondary"
                        } else {
                            "color.bg"
                        }),
                    )
                    .child(
                        Element::text(if selected { "[x]" } else { "[ ]" })
                            .class("prism-editor-content__selection"),
                    )
                    .child(Element::text(&entry.name).class("prism-editor-content__name"))
                    .child(Element::text(record.kind.as_str()).class("prism-editor-content__kind"))
                    .child(
                        Element::text(format!(
                            "VCS: {}{}",
                            vcs_name(vcs.state),
                            if vcs.staged { " (staged)" } else { "" }
                        ))
                        .class("prism-editor-content__vcs"),
                    )
                    .child(button(
                        &mut bound,
                        &format!("content:favorite:{}", id.raw()),
                        if record.favorite {
                            "Favorite: on"
                        } else {
                            "Favorite: off"
                        },
                        Some(ContentBrowserAction::SetFavorite {
                            asset: id,
                            value: !record.favorite,
                        }),
                    ))
                    .child(button(
                        &mut bound,
                        &format!("content:pin:{}", id.raw()),
                        if record.pinned {
                            "Pinned: on"
                        } else {
                            "Pinned: off"
                        },
                        Some(ContentBrowserAction::SetPinned {
                            asset: id,
                            value: !record.pinned,
                        }),
                    ))
            }
        };
        list = list.child(item);
    }
    if page.items.is_empty() {
        list = list.child(crate::support::empty_state(
            "prism-editor-content__empty",
            if catalog.is_empty() {
                "Catalog is empty"
            } else {
                "No matching content in this folder"
            },
        ));
    }
    let offset = controller.page_offset();
    let footer = row("prism-editor-content__footer")
        .child(button(
            &mut bound,
            "content:previous",
            "Previous",
            controller
                .has_previous()
                .then_some(ContentBrowserAction::PreviousPage),
        ))
        .child(Element::text(if page.items.is_empty() {
            "0 rows".to_owned()
        } else {
            format!(
                "Rows {}–{}",
                offset.saturating_add(1),
                offset.saturating_add(page.items.len())
            )
        }))
        .child(button(
            &mut bound,
            "content:next",
            "Next",
            page.next
                .is_some()
                .then_some(ContentBrowserAction::NextPage),
        ))
        .child(Element::text(format!(
            "{} selected",
            controller.selection().len()
        )))
        .child(button(
            &mut bound,
            "content:clear-selection",
            "Clear selection",
            (!controller.selection().is_empty()).then_some(ContentBrowserAction::ClearSelection),
        ));
    bound.element = column("prism-editor-content")
        .key_str("content:browser")
        .child(header)
        .child(navigation)
        .child(list)
        .child(footer);
    bound
}

fn field(
    bound: &mut BoundContentBrowser,
    key: &str,
    placeholder: &str,
    value: &str,
    recipe: Recipe,
) -> Element {
    bound
        .bindings
        .insert((key.to_owned(), UiTrigger::Commit), recipe);
    SearchField
        .render(
            &SearchFieldProps::new()
                .value(value)
                .placeholder(placeholder),
        )
        .key_str(key)
        .class("prism-editor-content__input")
}

fn button(
    bound: &mut BoundContentBrowser,
    key: &str,
    text: &str,
    action: Option<ContentBrowserAction>,
) -> Element {
    let enabled = action.is_some();
    if let Some(action) = action {
        bound.bindings.insert(
            (key.to_owned(), UiTrigger::Activate),
            Recipe::Action(action),
        );
    }
    Element::box_()
        .key_str(key)
        .class("prism-editor-content__button")
        .class(if enabled { "is-enabled" } else { "is-disabled" })
        .style(
            StyleProp::Opacity,
            StyleValue::number(if enabled { 1.0 } else { 0.45 }),
        )
        .style(StyleProp::PaddingLeft, StyleValue::px(4.0))
        .style(StyleProp::PaddingRight, StyleValue::px(4.0))
        .child(Element::text(text))
}

fn column(class: &str) -> Element {
    Element::box_()
        .class(class)
        .style(StyleProp::Display, StyleValue::keyword(Keyword::Flex))
        .style(
            StyleProp::FlexDirection,
            StyleValue::keyword(Keyword::Column),
        )
        .style(StyleProp::Gap, StyleValue::px(4.0))
}

fn row(class: &str) -> Element {
    Element::box_()
        .class(class)
        .style(StyleProp::Display, StyleValue::keyword(Keyword::Flex))
        .style(StyleProp::FlexDirection, StyleValue::keyword(Keyword::Row))
        .style(StyleProp::Gap, StyleValue::px(8.0))
        .style(StyleProp::FlexShrink, StyleValue::number(0.0))
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

fn parse_vcs(text: &str) -> Result<Option<VcsState>, ContentBrowserError> {
    let text = text.trim().to_ascii_lowercase();
    if text.is_empty() || text == "all" {
        return Ok(None);
    }
    [
        VcsState::Unknown,
        VcsState::Clean,
        VcsState::Untracked,
        VcsState::Added,
        VcsState::Modified,
        VcsState::Deleted,
        VcsState::Renamed,
        VcsState::Ignored,
        VcsState::Conflicted,
    ]
    .into_iter()
    .find(|state| vcs_name(*state) == text)
    .map(Some)
    .ok_or(ContentBrowserError::InvalidValue)
}
