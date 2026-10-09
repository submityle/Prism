//! [`CommandPalette`] — a searchable command launcher on the overlay base.
//!
//! A command palette pairs a `query` string with a list of [`CommandItem`]s,
//! each a label plus a keyboard-shortcut `hint`. It composes the shared
//! [`Popover`] base per architecture invariant K7, exposes [`Role::List`] for
//! assistive tech, and attaches only kit classes; [`crate::preset`] resolves
//! every value against the active theme.

use alloc::string::String;
use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;

use crate::feedback::popover::{Popover, PopoverProps};
use crate::preset::StyleSheet;

/// A single runnable command in a [`CommandPalette`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct CommandItem {
    /// The command's visible label.
    pub label: String,
    /// The keyboard-shortcut hint shown trailing the label.
    pub hint: String,
}

impl CommandItem {
    /// Creates a command with `label` and no hint.
    #[must_use]
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            ..Self::default()
        }
    }

    /// Sets the keyboard-shortcut hint.
    #[must_use]
    pub fn hint(mut self, hint: impl Into<String>) -> Self {
        self.hint = hint.into();
        self
    }
}

/// Props for [`CommandPalette`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct CommandPaletteProps {
    /// The current search query shown in the search slot.
    pub query: String,
    /// The (already filtered) list of commands.
    pub commands: Vec<CommandItem>,
    /// Whether the palette is currently shown.
    pub open: bool,
}

impl CommandPaletteProps {
    /// Creates empty, closed palette props.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the search query.
    #[must_use]
    pub fn query(mut self, query: impl Into<String>) -> Self {
        self.query = query.into();
        self
    }

    /// Appends a command.
    #[must_use]
    pub fn command(mut self, command: CommandItem) -> Self {
        self.commands.push(command);
        self
    }

    /// Replaces the command list with `commands`.
    #[must_use]
    pub fn commands<I: IntoIterator<Item = CommandItem>>(mut self, commands: I) -> Self {
        self.commands = commands.into_iter().collect();
        self
    }

    /// Sets whether the palette is shown.
    #[must_use]
    pub fn open(mut self, open: bool) -> Self {
        self.open = open;
        self
    }
}

/// The command-palette control. Zero-sized; configuration lives in
/// [`CommandPaletteProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct CommandPalette;

impl CommandPalette {
    /// The accessibility role the command list exposes.
    #[must_use]
    pub const fn role() -> Role {
        Role::List
    }
}

impl Component for CommandPalette {
    type Props = CommandPaletteProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut palette = Element::box_()
            .class("pk-command-palette")
            .child(Element::text(props.query.clone()).class("pk-command-palette__search"));

        for command in &props.commands {
            let item = Element::box_()
                .class("pk-command-palette__item")
                .child(Element::text(command.label.clone()).class("pk-command-palette__label"))
                .child(Element::text(command.hint.clone()).class("pk-command-palette__hint"));
            palette = palette.child(item);
        }

        Popover.render(&PopoverProps::new().open(props.open).child(palette))
    }
}

/// Registers the `pk-command-palette` class family: the panel, the search
/// slot, the command row, its label and the trailing shortcut hint.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, InteractionState, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    // Panel: a wide vertical stack holding the search box then the list.
    sheet.insert(
        Class::new("pk-command-palette")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::Gap, tok("space.xs"))
            .with(StyleProp::MinWidth, StyleValue::px(360.0))
            .with(StyleProp::MaxWidth, StyleValue::px(560.0)),
    );

    // Search: a prominent query field slot.
    sheet.insert(
        Class::new("pk-command-palette__search")
            .with_padding_x(tok("space.sm"))
            .with_padding_y(tok("space.sm"))
            .with(StyleProp::BorderRadius, tok("radius.md"))
            .with(StyleProp::BackgroundColor, tok("color.fill.secondary"))
            .with(StyleProp::FontSize, tok("font.size.headline"))
            .with(StyleProp::Color, tok("color.label")),
    );

    // Item: a row with the label on the leading edge and the hint trailing.
    sheet.insert(
        Class::new("pk-command-palette__item")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::JustifyContent, kw(Keyword::SpaceBetween))
            .with(StyleProp::Gap, tok("space.sm"))
            .with_padding_x(tok("space.sm"))
            .with_padding_y(tok("space.xs"))
            .with(StyleProp::BorderRadius, tok("radius.sm"))
            .with_state(InteractionState::Hover, StyleProp::BackgroundColor, tok("color.fill.secondary")),
    );

    // Label: primary body text.
    sheet.insert(
        Class::new("pk-command-palette__label")
            .with(StyleProp::FontSize, tok("font.size.body"))
            .with(StyleProp::Color, tok("color.label")),
    );

    // Hint: a quiet, monospace-ish shortcut chip.
    sheet.insert(
        Class::new("pk-command-palette__hint")
            .with(StyleProp::FontSize, tok("font.size.footnote"))
            .with(StyleProp::Color, tok("color.label.tertiary")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: CommandPaletteProps) -> Element {
        CommandPalette.render(&props)
    }

    fn palette(el: &Element) -> Element {
        let surface = el
            .child_elements()
            .iter()
            .find(|c| c.class_names().iter().any(|n| n == "pk-popover__surface"))
            .expect("surface")
            .clone();
        surface.child_elements()[0].clone()
    }

    #[test]
    fn role_is_list() {
        assert_eq!(CommandPalette::role(), Role::List);
    }

    #[test]
    fn composes_popover_base() {
        let el = render(CommandPaletteProps::new().open(true));
        assert!(el.class_names().iter().any(|c| c == "pk-popover"));
        assert!(el.class_names().iter().any(|c| c == "is-open"));
    }

    #[test]
    fn search_slot_shows_the_query() {
        let el = render(CommandPaletteProps::new().query("deploy"));
        let palette = palette(&el);
        let search = &palette.child_elements()[0];
        assert!(search.class_names().iter().any(|c| c == "pk-command-palette__search"));
        assert_eq!(search.text_content(), Some("deploy"));
    }

    #[test]
    fn commands_render_label_and_hint() {
        let el = render(
            CommandPaletteProps::new()
                .command(CommandItem::new("Open File").hint("Cmd+O")),
        );
        let palette = palette(&el);
        let item = &palette.child_elements()[1];
        assert!(item.class_names().iter().any(|c| c == "pk-command-palette__item"));
        let parts = item.child_elements();
        assert_eq!(parts[0].text_content(), Some("Open File"));
        assert!(parts[1].class_names().iter().any(|c| c == "pk-command-palette__hint"));
        assert_eq!(parts[1].text_content(), Some("Cmd+O"));
    }
}
