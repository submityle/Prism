//! Loom assembly for the editor command palette.

use prism_editor_shell::{CommandIndex, CommandSearchResult};
use prism_ui::Element;
use prism_ui_component::Component;
use prism_ui_component_kit::inputs::{SearchField, SearchFieldProps};

use crate::action::{ActionBinding, ActionBindings, ActionTemplate, BoundElement, UiBuildError};
use crate::support::{bounded_count, empty_state, label};

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
    let results = props.index.search(props.query, bounded_count(props.limit));
    command_palette_results(props.query, &results, props.selected_id)
}

/// Builds a command palette and binds enabled rows to command actions.
pub fn command_palette_bound(
    props: &CommandPaletteProps<'_>,
) -> Result<BoundElement, UiBuildError> {
    let results = props.index.search(props.query, bounded_count(props.limit));
    command_palette_results_bound(props.query, &results, props.selected_id)
}

/// Builds a command palette from already resolved search results.
#[must_use]
pub fn command_palette_results(
    query: &str,
    results: &[CommandSearchResult],
    selected_id: Option<&str>,
) -> Element {
    build_command_palette(query, results, selected_id, None)
        .expect("command palette without action bindings cannot fail")
        .into_element()
}

/// Builds a command palette from resolved results and binds enabled rows.
pub fn command_palette_results_bound(
    query: &str,
    results: &[CommandSearchResult],
    selected_id: Option<&str>,
) -> Result<BoundElement, UiBuildError> {
    build_command_palette(query, results, selected_id, Some(ActionBindings::new()))
}

fn build_command_palette(
    query: &str,
    results: &[CommandSearchResult],
    selected_id: Option<&str>,
    bindings: Option<ActionBindings>,
) -> Result<BoundElement, UiBuildError> {
    let mut bindings = bindings;
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
        for result in results.iter().take(bounded_count(results.len())) {
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
            if result.enabled
                && let Some(action_bindings) = bindings.as_mut()
            {
                action_bindings.bind(ActionBinding::activate(
                    format!("command:{}", result.id),
                    ActionTemplate::execute_command(result.id.clone())?,
                )?)?;
            }
            list = list.child(row);
        }
    }

    let element = Element::custom("prism.editor.command-palette")
        .class("prism-editor-command-palette")
        .child(header)
        .child(list);
    Ok(BoundElement::new(element, bindings.unwrap_or_default()))
}
