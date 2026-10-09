//! [`CopyButton`] — a label that reflects copy-to-clipboard state.
//!
//! A copy button pairs a `value` (the text a host would place on the system
//! clipboard) with a visible `label` and an externally-owned `copied` flag.
//! The control performs *no* clipboard I/O — the kit is `no_std` and data-only.
//! A host wires the click to the real clipboard and flips `copied` for a beat
//! so the control can show an acknowledgement; the control simply reflects that
//! state by attaching `is-copied`.
//!
//! It renders a `pk-copy-button` box with a `pk-copy-button__label` text child.
//! The `value` is carried in props (so a host can read what to copy) but is not
//! painted — only the label is shown.

use alloc::string::String;

use prism_ui::Element;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// Props for [`CopyButton`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct CopyButtonProps {
    /// The text a host would copy to the clipboard. Not rendered visibly.
    pub value: String,
    /// The visible label (e.g. "Copy" / "Copied!").
    pub label: String,
    /// Whether the host is currently showing the copied acknowledgement.
    pub copied: bool,
}

impl CopyButtonProps {
    /// Creates props carrying `value`, with the given visible `label`.
    #[must_use]
    pub fn new(value: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            value: value.into(),
            label: label.into(),
            copied: false,
        }
    }

    /// Sets the clipboard value.
    #[must_use]
    pub fn value(mut self, value: impl Into<String>) -> Self {
        self.value = value.into();
        self
    }

    /// Sets the visible label.
    #[must_use]
    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.label = label.into();
        self
    }

    /// Sets the externally-owned copied flag.
    #[must_use]
    pub fn copied(mut self, copied: bool) -> Self {
        self.copied = copied;
        self
    }
}

/// The copy-button control. Zero-sized; config lives in [`CopyButtonProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct CopyButton;

impl Component for CopyButton {
    type Props = CopyButtonProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_().class("pk-copy-button");
        if props.copied {
            el = el.class("is-copied");
        }
        el.child(Element::text(props.label.clone()).class("pk-copy-button__label"))
    }
}

/// Registers the `pk-copy-button` family: a compact capsule control that reads
/// as a tinted glass chip; `is-copied` swaps the label to the accent color as a
/// cheap acknowledgement. All values are token-backed so the chip follows the
/// theme.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, InteractionState, Keyword, StyleProp};

    use crate::preset::{kw, tok};

    sheet.insert(
        Class::new("pk-copy-button")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::JustifyContent, kw(Keyword::Center))
            .with(StyleProp::Gap, tok("space.xs"))
            .with(StyleProp::Height, tok("size.control.height"))
            .with_padding_x(tok("space.sm"))
            .with(StyleProp::BorderRadius, tok("radius.capsule"))
            .with(StyleProp::FontSize, tok("font.size.subheadline"))
            .with(StyleProp::FontWeight, tok("font.weight.semibold"))
            .with(StyleProp::Color, tok("color.label"))
            .with_glass(0.0, tok("color.fill"), Some(tok("glass.highlight")))
            .with_state(InteractionState::Hover, StyleProp::Color, tok("color.tint")),
    );

    sheet.insert(
        Class::new("is-copied").with(StyleProp::Color, tok("color.tint")),
    );
}

/// Alias: `Clipboard` names the copy-to-clipboard affordance.
pub type Clipboard = CopyButton;

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ui::ElementKind;

    fn render(props: CopyButtonProps) -> Element {
        CopyButton.render(&props)
    }

    #[test]
    fn attaches_base_class_without_copied_marker() {
        let el = render(CopyButtonProps::new("abc123", "Copy"));
        assert_eq!(el.class_names(), ["pk-copy-button"]);
    }

    #[test]
    fn copied_adds_marker_class() {
        let el = render(CopyButtonProps::new("abc123", "Copied!").copied(true));
        assert_eq!(el.class_names(), ["pk-copy-button", "is-copied"]);
    }

    #[test]
    fn renders_label_but_not_value() {
        let el = render(CopyButtonProps::new("secret-token", "Copy"));
        let label = el
            .child_elements()
            .iter()
            .find(|c| c.kind() == &ElementKind::Text)
            .expect("label child");
        assert_eq!(label.class_names(), ["pk-copy-button__label"]);
        assert_eq!(label.text_content(), Some("Copy"));
        // The clipboard value is carried in props, never painted.
        assert!(el
            .child_elements()
            .iter()
            .all(|c| c.text_content() != Some("secret-token")));
    }

    #[test]
    fn register_adds_family() {
        let mut sheet = StyleSheet::new();
        register_styles(&mut sheet);
        assert!(sheet.get("pk-copy-button").is_some());
        assert!(sheet.get("is-copied").is_some());
    }
}
