//! [`Console`] / [`LogView`] — a monospace log panel with filter + search.
//!
//! A console renders a `pk-log-view` column: a `__toolbar` strip (a level
//! filter with search slots) above a stack of `__row` entries. Each row carries a
//! per-level modifier `pk-log-view__row--{level}`, a `__level` tag and a
//! `__message` rendered through the monospace [`Code`](crate::basics::Code)
//! control. Only kit class names are attached; color and type resolve from
//! theme tokens via [`crate::preset`].

use alloc::string::String;
use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;

use crate::basics::code::{Code, CodeProps};
use crate::preset::StyleSheet;

/// A log entry's severity level.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum LogLevel {
    /// Fine-grained tracing output.
    Trace,
    /// Debugging output.
    Debug,
    /// Informational output (default).
    #[default]
    Info,
    /// A warning.
    Warn,
    /// An error.
    Error,
}

impl LogLevel {
    /// The modifier suffix used in class names (e.g. `pk-log-view__row--warn`).
    #[must_use]
    pub const fn suffix(self) -> &'static str {
        match self {
            LogLevel::Trace => "trace",
            LogLevel::Debug => "debug",
            LogLevel::Info => "info",
            LogLevel::Warn => "warn",
            LogLevel::Error => "error",
        }
    }

    /// The color token this level paints its tag with.
    #[must_use]
    pub const fn color_token(self) -> &'static str {
        match self {
            LogLevel::Trace => "color.label.tertiary",
            LogLevel::Debug => "color.label.secondary",
            LogLevel::Info => "color.blue",
            LogLevel::Warn => "color.orange",
            LogLevel::Error => "color.red",
        }
    }
}

/// A single log entry: a severity level plus a message.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct LogEntry {
    /// The entry's severity level.
    pub level: LogLevel,
    /// The log message text.
    pub message: String,
}

impl LogEntry {
    /// Creates an entry at `level` with `message`.
    #[must_use]
    pub fn new(level: LogLevel, message: impl Into<String>) -> Self {
        Self {
            level,
            message: message.into(),
        }
    }
}

/// Props for [`Console`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct ConsoleProps {
    /// Optional level-filter control shown in the toolbar.
    pub filter: Option<Element>,
    /// Optional search control shown in the toolbar.
    pub search: Option<Element>,
    /// The log entries, rendered top-to-bottom.
    pub entries: Vec<LogEntry>,
}

impl ConsoleProps {
    /// Creates empty console props.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the level-filter control.
    #[must_use]
    pub fn filter(mut self, filter: Element) -> Self {
        self.filter = Some(filter);
        self
    }

    /// Sets the search control.
    #[must_use]
    pub fn search(mut self, search: Element) -> Self {
        self.search = Some(search);
        self
    }

    /// Appends a log entry.
    #[must_use]
    pub fn entry(mut self, entry: LogEntry) -> Self {
        self.entries.push(entry);
        self
    }

    /// Replaces the log entries.
    #[must_use]
    pub fn entries<I: IntoIterator<Item = LogEntry>>(mut self, entries: I) -> Self {
        self.entries = entries.into_iter().collect();
        self
    }
}

/// The console control. Zero-sized; all configuration lives in [`ConsoleProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Console;

/// Alias matching the `LogView` naming used by the design doc (section 12).
pub type LogView = Console;

impl Console {
    /// The accessibility role a console exposes.
    #[must_use]
    pub const fn role() -> Role {
        Role::List
    }
}

/// Builds the `--{level}` row modifier, e.g. `pk-log-view__row--error`.
fn row_level_class(level: LogLevel) -> String {
    let suffix = level.suffix();
    let mut name = String::with_capacity("pk-log-view__row--".len() + suffix.len());
    name.push_str("pk-log-view__row--");
    name.push_str(suffix);
    name
}

impl Component for Console {
    type Props = ConsoleProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_().class("pk-log-view");

        let mut toolbar = Element::box_().class("pk-log-view__toolbar");
        if let Some(filter) = props.filter.clone() {
            toolbar = toolbar.child(filter.class("pk-log-view__filter"));
        }
        if let Some(search) = props.search.clone() {
            toolbar = toolbar.child(search.class("pk-log-view__search"));
        }
        el = el.child(toolbar);

        for entry in &props.entries {
            let level = Element::text(String::from(entry.level.suffix())).class("pk-log-view__level");
            let message = Code
                .render(&CodeProps::new(entry.message.clone()))
                .class("pk-log-view__message");
            let row = Element::box_()
                .class("pk-log-view__row")
                .class(row_level_class(entry.level))
                .child(level)
                .child(message);
            el = el.child(row);
        }
        el
    }
}

/// Every level, for exhaustive style registration.
const LEVELS: [LogLevel; 5] = [
    LogLevel::Trace,
    LogLevel::Debug,
    LogLevel::Info,
    LogLevel::Warn,
    LogLevel::Error,
];

/// Registers the `pk-log-view` family: panel, toolbar, row + per-level tag.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp};

    use crate::preset::{kw, tok};

    sheet.insert(
        Class::new("pk-log-view")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::Gap, tok("space.xxs"))
            .with(StyleProp::BackgroundColor, tok("color.background.secondary"))
            .with(StyleProp::BorderRadius, tok("radius.md"))
            .with_padding_x(tok("space.sm"))
            .with_padding_y(tok("space.sm")),
    );

    sheet.insert(
        Class::new("pk-log-view__toolbar")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::JustifyContent, kw(Keyword::SpaceBetween))
            .with(StyleProp::Gap, tok("space.sm")),
    );

    sheet.insert(
        Class::new("pk-log-view__row")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Baseline))
            .with(StyleProp::Gap, tok("space.sm"))
            .with(StyleProp::FontSize, tok("font.size.footnote")),
    );

    sheet.insert(
        Class::new("pk-log-view__level")
            .with(StyleProp::FontWeight, tok("font.weight.bold"))
            .with(StyleProp::FontSize, tok("font.size.caption1")),
    );
    for level in LEVELS {
        let mut name = String::from("pk-log-view__row--");
        name.push_str(level.suffix());
        sheet.insert(Class::new(name).with(StyleProp::Color, tok(level.color_token())));
    }

    sheet.insert(
        Class::new("pk-log-view__message").with(StyleProp::Color, tok("color.label")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: ConsoleProps) -> Element {
        Console.render(&props)
    }

    #[test]
    fn empty_console_has_only_toolbar() {
        let el = render(ConsoleProps::new());
        assert_eq!(el.class_names(), ["pk-log-view"]);
        let children = el.child_elements();
        assert_eq!(children.len(), 1);
        assert_eq!(children[0].class_names(), ["pk-log-view__toolbar"]);
    }

    #[test]
    fn row_carries_level_modifier_and_message() {
        let el = render(
            ConsoleProps::new().entry(LogEntry::new(LogLevel::Error, "boom")),
        );
        let row = &el.child_elements()[1];
        assert_eq!(
            row.class_names(),
            ["pk-log-view__row", "pk-log-view__row--error"]
        );
        let parts = row.child_elements();
        assert_eq!(parts[0].text_content(), Some("error"));
        assert!(parts[1]
            .class_names()
            .iter()
            .any(|c| c == "pk-log-view__message"));
        assert!(parts[1].class_names().iter().any(|c| c == "pk-code"));
    }

    #[test]
    fn toolbar_hosts_filter_and_search() {
        let el = render(
            ConsoleProps::new()
                .filter(Element::box_().class("f"))
                .search(Element::box_().class("s")),
        );
        let toolbar = &el.child_elements()[0];
        let parts = toolbar.child_elements();
        assert_eq!(parts.len(), 2);
        assert!(parts[0]
            .class_names()
            .iter()
            .any(|c| c == "pk-log-view__filter"));
        assert!(parts[1]
            .class_names()
            .iter()
            .any(|c| c == "pk-log-view__search"));
    }

    #[test]
    fn default_level_is_info() {
        assert_eq!(LogLevel::default(), LogLevel::Info);
    }

    #[test]
    fn role_is_list() {
        assert_eq!(Console::role(), Role::List);
    }

    #[test]
    fn register_adds_classes() {
        let mut sheet = StyleSheet::new();
        register_styles(&mut sheet);
        for name in [
            "pk-log-view",
            "pk-log-view__toolbar",
            "pk-log-view__row",
            "pk-log-view__row--trace",
            "pk-log-view__row--debug",
            "pk-log-view__row--info",
            "pk-log-view__row--warn",
            "pk-log-view__row--error",
            "pk-log-view__level",
            "pk-log-view__message",
        ] {
            assert!(sheet.get(name).is_some(), "missing {name}");
        }
    }
}
