//! [`TagInput`] — a token entry field that collects discrete tags.
//!
//! A tag input renders a `pk-tag-input` box laying out one `__tag` per value
//! (each a `__tag-label` plus a `__tag-remove` affordance) followed by a
//! trailing `__input` slot for the next token. Tags flow inline and share the
//! row with the input. All color comes from theme tokens via [`crate::preset`].
//! [`TokenField`] is a drop-in alias.

use alloc::string::String;
use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// Props for [`TagInput`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct TagInputProps {
    /// The current tags, rendered in order before the input slot.
    pub tags: Vec<String>,
    /// Placeholder text shown in the trailing input slot.
    pub placeholder: String,
    /// Whether the control is non-interactive.
    pub disabled: bool,
}

impl TagInputProps {
    /// Creates empty tag-input props.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Replaces the tags with `tags`.
    #[must_use]
    pub fn tags<I, S>(mut self, tags: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.tags = tags.into_iter().map(Into::into).collect();
        self
    }

    /// Sets the placeholder text.
    #[must_use]
    pub fn placeholder(mut self, placeholder: impl Into<String>) -> Self {
        self.placeholder = placeholder.into();
        self
    }

    /// Marks the control disabled.
    #[must_use]
    pub fn disabled(mut self, disabled: bool) -> Self {
        self.disabled = disabled;
        self
    }
}

/// The tag-input control. Zero-sized; config lives in [`TagInputProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct TagInput;

impl TagInput {
    /// The accessibility role a tag input exposes.
    #[must_use]
    pub const fn role() -> Role {
        Role::Group
    }
}

impl Component for TagInput {
    type Props = TagInputProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_().class("pk-tag-input");
        if props.disabled {
            el = el.class("pk-tag-input--disabled");
        }

        for tag in &props.tags {
            el = el.child(
                Element::box_()
                    .class("pk-tag-input__tag")
                    .child(Element::text(tag.clone()).class("pk-tag-input__tag-label"))
                    .child(Element::box_().class("pk-tag-input__tag-remove")),
            );
        }

        el.child(Element::text(props.placeholder.clone()).class("pk-tag-input__input"))
    }
}

/// A token field. An alias of [`TagInput`] with the same props.
pub type TokenField = TagInput;

/// Registers the `pk-tag-input` class family: box, disabled modifier, tags
/// (label + remove) and the trailing input slot.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, InteractionState, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    sheet.insert(
        Class::new("pk-tag-input")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::Gap, tok("space.xs"))
            .with_padding_x(tok("space.sm"))
            .with_padding_y(tok("space.xs"))
            .with(StyleProp::MinHeight, tok("size.control.height"))
            .with(StyleProp::BorderRadius, tok("radius.md"))
            .with(StyleProp::BorderWidth, StyleValue::px(1.0))
            .with(StyleProp::BorderColor, tok("color.separator"))
            .with_glass(0.0, tok("color.surface.secondary"), Some(tok("glass.highlight")))
            .with(StyleProp::Color, tok("color.label"))
            .with(StyleProp::FontSize, tok("font.size.body")),
    );
    sheet.insert(
        Class::new("pk-tag-input--disabled").with(StyleProp::Opacity, StyleValue::number(0.4)),
    );

    sheet.insert(
        Class::new("pk-tag-input__tag")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::Gap, tok("space.xxs"))
            .with(StyleProp::FlexShrink, StyleValue::number(0.0))
            .with_padding_x(tok("space.sm"))
            .with_padding_y(tok("space.xxs"))
            .with(StyleProp::BorderRadius, tok("radius.capsule"))
            .with(StyleProp::BackgroundColor, tok("color.fill.secondary"))
            .with(StyleProp::Color, tok("color.label"))
            .with(StyleProp::FontSize, tok("font.size.footnote")),
    );
    sheet.insert(
        Class::new("pk-tag-input__tag-label").with(StyleProp::Color, tok("color.label")),
    );
    sheet.insert(
        Class::new("pk-tag-input__tag-remove")
            .with(StyleProp::Width, StyleValue::px(12.0))
            .with(StyleProp::Height, StyleValue::px(12.0))
            .with(StyleProp::MinWidth, StyleValue::px(12.0))
            .with(StyleProp::FlexShrink, StyleValue::number(0.0))
            .with(StyleProp::BorderRadius, tok("radius.capsule"))
            .with(StyleProp::BackgroundColor, tok("color.label.tertiary"))
            .with_state(InteractionState::Hover, StyleProp::BackgroundColor, tok("color.label.secondary")),
    );

    sheet.insert(
        Class::new("pk-tag-input__input")
            .with(StyleProp::FlexGrow, StyleValue::number(1.0))
            .with(StyleProp::MinWidth, StyleValue::px(60.0))
            .with(StyleProp::Color, tok("color.label.tertiary"))
            .with(StyleProp::FontSize, tok("font.size.body")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: TagInputProps) -> Element {
        TagInput.render(&props)
    }

    #[test]
    fn tags_render_before_the_input_slot() {
        let el = render(TagInputProps::new().tags(["red", "green"]).placeholder("Add"));
        assert_eq!(el.class_names(), ["pk-tag-input"]);
        let kids = el.child_elements();
        assert_eq!(kids.len(), 3);
        assert_eq!(kids[0].class_names(), ["pk-tag-input__tag"]);
        assert_eq!(kids[1].class_names(), ["pk-tag-input__tag"]);
        assert_eq!(kids[2].class_names(), ["pk-tag-input__input"]);
        assert_eq!(kids[2].text_content(), Some("Add"));
    }

    #[test]
    fn each_tag_has_label_and_remove() {
        let el = render(TagInputProps::new().tags(["one"]));
        let tag = &el.child_elements()[0];
        let parts = tag.child_elements();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].class_names(), ["pk-tag-input__tag-label"]);
        assert_eq!(parts[0].text_content(), Some("one"));
        assert_eq!(parts[1].class_names(), ["pk-tag-input__tag-remove"]);
    }

    #[test]
    fn no_tags_still_renders_input_slot() {
        let el = render(TagInputProps::new().placeholder("Tag"));
        let kids = el.child_elements();
        assert_eq!(kids.len(), 1);
        assert_eq!(kids[0].class_names(), ["pk-tag-input__input"]);
    }

    #[test]
    fn disabled_adds_block_modifier() {
        let el = render(TagInputProps::new().disabled(true));
        assert!(el.class_names().iter().any(|c| c == "pk-tag-input--disabled"));
    }

    #[test]
    fn alias_renders_like_tag_input() {
        let via_alias = TokenField::default().render(&TagInputProps::new().tags(["x"]));
        let direct = TagInput.render(&TagInputProps::new().tags(["x"]));
        assert_eq!(via_alias.child_elements().len(), direct.child_elements().len());
    }

    #[test]
    fn role_is_group() {
        assert_eq!(TagInput::role(), Role::Group);
    }
}
