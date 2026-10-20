//! Loom assembly for the editor command palette.

use prism_editor_shell::{CommandIndex, CommandSearchResult};
use prism_ui::Element;
use prism_ui_component::Component;
use prism_ui_component_kit::inputs::{SearchField, SearchFieldProps};

use crate::support::{empty_state, label};

/// Inputs for the command palette.
#[derive(Clone, Copy)]
pub struct CommandPaletteProps<'a> {
    /// The command index used to score and order results.
    pub index: &'a CommandIndex,
    /// Current query text.
    pub query: &'a str,
    /// Maximum number of result rows to materialize.
    pub limit: usize,
    /// Stable command ID of the highlighted result, if any.
    pub selected_id: Option<&'a str>,
}

/// Builds a command palette by querying the existing deterministic index.
#[must_use]
pub fn command_palette(props: &CommandPaletteProps<'_>) -> Element {
    let results = props.index.search(props.query, props.limit);
    command_palette_results(props.query, &results, props.selected_id)
}

/// Builds a command palette from already resolved search results.
#[must_use]
pub fn command_palette_results(
    query: &str,
    results: &[CommandSearchResult],
    selected_id: Option<&str>,
) -> Element {
    let search = SearchField.render(
        &SearchFieldProps::new()
            .value(query)
            .placeholder("Search commands"),
    );
    let header = Element::box_()
        .class("prism-editor-command-palette__header")
        .child(label("prism-editor-command-palette__title", "Commands"))
        .child(search.class("prism-editor-command-palette__search"));

    let mut list = Element::box_().class("prism-editor-command-palette__results");
    if results.is_empty() {
        list = list.child(empty_state(
            "prism-editor-command-palette__empty",
            if query.trim().is_empty() {
                "No commands available"
            } else {
                "No matching commands"
            },
        ));
    } else {
        for result in results {
            let mut row = Element::box_()
                .key_str(format!("command:{}", result.id))
                .class("prism-editor-command-palette__row")
                .child(label(
                    "prism-editor-command-palette__label",
                    result.title.clone(),
                ));
            if let Some(category) = &result.category {
                row = row.child(label(
                    "prism-editor-command-palette__category",
                    category.clone(),
                ));
            }
            if !result.enabled {
                row = row.class("is-disabled");
            }
            if selected_id == Some(result.id.as_str()) {
                row = row.class("is-selected");
            }
            list = list.child(row);
        }
    }

    Element::custom("prism.editor.command-palette")
        .class("prism-editor-command-palette")
        .child(header)
        .child(list)
}
