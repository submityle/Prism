//! Loom assembly for the bounded editor Console.

use prism_editor_panels::{Console, ConsoleEntry, LogLevel};
use prism_ui::Element;
use prism_ui_component::Component;
use prism_ui_component_kit::inputs::{SearchField, SearchFieldProps};

use crate::support::{bounded_count, empty_state, label, normalized_filter};

/// Inputs for the Console Element tree.
#[derive(Clone, Copy, Debug)]
pub struct ConsolePanelProps<'a> {
    /// Source bounded console model.
    pub console: &'a Console,
    /// Minimum severity to display.
    pub minimum: LogLevel,
    /// Optional exact channel filter.
    pub channel: Option<&'a str>,
    /// Case-insensitive message/channel search text.
    pub query: &'a str,
    /// Start offset in retained console order.
    pub start: usize,
    /// Number of rows requested before the hard safety cap.
    pub count: usize,
}

/// Builds a bounded, keyed Console panel.
#[must_use]
pub fn console_panel(props: &ConsolePanelProps<'_>) -> Element {
    let search = SearchField.render(
        &SearchFieldProps::new()
            .value(props.query)
            .placeholder("Filter console"),
    );
    let header = Element::box_()
        .class("prism-editor-console__header")
        .child(label("prism-editor-console__title", "Console"))
        .child(search.class("prism-editor-console__search"));

    let query = normalized_filter(props.query);
    let mut list = Element::box_().class("prism-editor-console__list");
    let mut skipped = 0usize;
    let mut emitted = 0usize;
    let limit = bounded_count(props.count);
    for entry in props.console.entries().iter().filter(|entry| {
        entry.level >= props.minimum
            && props.channel.is_none_or(|channel| channel == entry.channel)
            && (query.is_empty()
                || entry.message.to_lowercase().contains(&query)
                || entry.channel.to_lowercase().contains(&query))
    }) {
        if skipped < props.start {
            skipped += 1;
            continue;
        }
        if emitted == limit {
            break;
        }
        emitted += 1;
        list = list.child(console_row(entry));
    }
    if emitted == 0 {
        list = list.child(empty_state(
            "prism-editor-console__empty",
            if query.is_empty() {
                "Console has no retained entries"
            } else {
                "No matching console entries"
            },
        ));
    }

    Element::custom("prism.editor.console")
        .class("prism-editor-console")
        .child(header)
        .child(list)
}

fn console_row(entry: &ConsoleEntry) -> Element {
    Element::box_()
        .key_str(format!("console:{}", entry.sequence))
        .class("prism-editor-console__row")
        .class(format!("is-{}", level_name(entry.level)))
        .child(label(
            "prism-editor-console__level",
            level_name(entry.level),
        ))
        .child(label(
            "prism-editor-console__channel",
            entry.channel.clone(),
        ))
        .child(label(
            "prism-editor-console__message",
            entry.message.clone(),
        ))
}

const fn level_name(level: LogLevel) -> &'static str {
    match level {
        LogLevel::Trace => "trace",
        LogLevel::Debug => "debug",
        LogLevel::Info => "info",
        LogLevel::Warn => "warn",
        LogLevel::Error => "error",
    }
}
