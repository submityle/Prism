//! [`Dialog`] — a modal panel over a dimming backdrop.
//!
//! A dialog layers a `pk-dialog__backdrop` behind a `pk-dialog__panel` whose
//! glass surface is composed from the shared
//! [`Popover`](crate::feedback::Popover) base (architecture invariant K7). The
//! panel stacks an optional `__title`, a `__body` slot and an `__actions` row.
//! It attaches only kit classes and exposes [`Role::Dialog`] for assistive tech;
//! [`crate::preset`] resolves every value against the active theme.

use alloc::string::String;
use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;

use crate::feedback::popover::{Popover, PopoverProps};
use crate::preset::StyleSheet;

/// Props for [`Dialog`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct DialogProps {
    /// Whether the dialog is currently shown.
    pub open: bool,
    /// Optional heading rendered at the top of the panel.
    pub title: Option<String>,
    /// The main dialog content.
    pub body: Vec<Element>,
    /// The trailing action controls (e.g. confirm/cancel buttons).
    pub actions: Vec<Element>,
}

impl DialogProps {
    /// Creates empty, closed dialog props.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets whether the dialog is shown.
    #[must_use]
    pub fn open(mut self, open: bool) -> Self {
        self.open = open;
        self
    }

    /// Sets the panel heading.
    #[must_use]
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Appends a body child.
    #[must_use]
    pub fn body_child(mut self, child: Element) -> Self {
        self.body.push(child);
        self
    }

    /// Replaces the body content with `body`.
    #[must_use]
    pub fn body<I: IntoIterator<Item = Element>>(mut self, body: I) -> Self {
        self.body = body.into_iter().collect();
        self
    }

    /// Appends an action control.
    #[must_use]
    pub fn action(mut self, action: Element) -> Self {
        self.actions.push(action);
        self
    }

    /// Replaces the action controls with `actions`.
    #[must_use]
    pub fn actions<I: IntoIterator<Item = Element>>(mut self, actions: I) -> Self {
        self.actions = actions.into_iter().collect();
        self
    }
}

/// The dialog control. Zero-sized; all configuration lives in [`DialogProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Dialog;

impl Dialog {
    /// The accessibility role a dialog exposes.
    #[must_use]
    pub const fn role() -> Role {
        Role::Dialog
    }
}

impl Component for Dialog {
    type Props = DialogProps;

    fn render(&self, props: &Self::Props) -> Element {
        // Assemble the panel's stacked sections.
        let mut sections: Vec<Element> = Vec::new();
        if let Some(title) = props.title.clone() {
            sections.push(Element::text(title).class("pk-dialog__title"));
        }
        sections.push(
            Element::box_()
                .class("pk-dialog__body")
                .children(props.body.iter().cloned()),
        );
        if !props.actions.is_empty() {
            sections.push(
                Element::box_()
                    .class("pk-dialog__actions")
                    .children(props.actions.iter().cloned()),
            );
        }

        // Compose the overlay base: the panel is a popover glass surface.
        let panel = Popover
            .render(&PopoverProps::new().open(props.open).children(sections))
            .class("pk-dialog__panel");

        let mut el = Element::box_().class("pk-dialog");
        if props.open {
            el = el.class("is-open");
        }
        el.child(Element::box_().class("pk-dialog__backdrop"))
            .child(panel)
    }
}

/// Registers the `pk-dialog` class family: the overlay root, dimming backdrop,
/// composed glass panel, title, body and actions row.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    // Root: a full overlay layer stacking backdrop then panel.
    sheet.insert(
        Class::new("pk-dialog")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::JustifyContent, kw(Keyword::Center)),
    );

    // Backdrop: a dimming scrim behind the panel.
    sheet.insert(
        Class::new("pk-dialog__backdrop")
            .with(StyleProp::BackgroundColor, tok("color.fill.secondary")),
    );

    // Panel: tightens the composed popover surface into a modal card.
    sheet.insert(
        Class::new("pk-dialog__panel")
            .with(StyleProp::MinWidth, StyleValue::px(280.0))
            .with(StyleProp::BorderRadius, tok("radius.alert")),
    );

    // Title: headline type.
    sheet.insert(
        Class::new("pk-dialog__title")
            .with(StyleProp::FontSize, tok("font.size.headline"))
            .with(StyleProp::FontWeight, tok("font.weight.semibold"))
            .with(StyleProp::Color, tok("color.label")),
    );

    // Body: fills the panel with readable body type.
    sheet.insert(
        Class::new("pk-dialog__body")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::Gap, tok("space.sm"))
            .with(StyleProp::FontSize, tok("font.size.body"))
            .with(StyleProp::Color, tok("color.label.secondary")),
    );

    // Actions: a trailing row of controls.
    sheet.insert(
        Class::new("pk-dialog__actions")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::JustifyContent, kw(Keyword::End))
            .with(StyleProp::Gap, tok("space.sm")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: DialogProps) -> Element {
        Dialog.render(&props)
    }

    #[test]
    fn role_is_dialog() {
        assert_eq!(Dialog::role(), Role::Dialog);
    }

    #[test]
    fn open_adds_marker_and_has_backdrop_then_panel() {
        let el = render(DialogProps::new().open(true));
        assert!(el.class_names().iter().any(|c| c == "is-open"));
        let kids = el.child_elements();
        assert_eq!(kids.len(), 2);
        assert!(kids[0].class_names().iter().any(|c| c == "pk-dialog__backdrop"));
        assert!(kids[1].class_names().iter().any(|c| c == "pk-dialog__panel"));
    }

    #[test]
    fn panel_composes_popover_base() {
        let el = render(DialogProps::new());
        let panel = &el.child_elements()[1];
        assert!(panel.class_names().iter().any(|c| c == "pk-popover"));
        assert!(panel.class_names().iter().any(|c| c == "pk-dialog__panel"));
    }

    #[test]
    fn title_body_and_actions_populate_the_surface() {
        let el = render(
            DialogProps::new()
                .title("Delete?")
                .body_child(Element::text("This cannot be undone."))
                .action(Element::box_().class("confirm")),
        );
        let panel = &el.child_elements()[1];
        let surface = panel
            .child_elements()
            .iter()
            .find(|c| c.class_names().iter().any(|n| n == "pk-popover__surface"))
            .expect("surface");
        let sections = surface.child_elements();
        assert_eq!(sections.len(), 3);
        assert_eq!(sections[0].text_content(), Some("Delete?"));
        assert!(sections[0].class_names().iter().any(|c| c == "pk-dialog__title"));
        assert!(sections[1].class_names().iter().any(|c| c == "pk-dialog__body"));
        assert!(sections[2].class_names().iter().any(|c| c == "pk-dialog__actions"));
    }

    #[test]
    fn actions_row_omitted_when_empty() {
        let el = render(DialogProps::new().body_child(Element::text("hi")));
        let panel = &el.child_elements()[1];
        let surface = panel
            .child_elements()
            .iter()
            .find(|c| c.class_names().iter().any(|n| n == "pk-popover__surface"))
            .expect("surface");
        // Only the body section is present (no title, no actions).
        assert_eq!(surface.child_elements().len(), 1);
        assert!(surface.child_elements()[0]
            .class_names()
            .iter()
            .any(|c| c == "pk-dialog__body"));
    }
}
