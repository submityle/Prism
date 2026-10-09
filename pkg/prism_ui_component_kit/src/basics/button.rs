//! [`Button`] — the kit's primary action control.
//!
//! A button is a `(variant, size, tone)` triple plus a label and optional
//! leading/trailing children. It carries no style of its own: it attaches the
//! `pk-button` class family and lets [`crate::preset`] resolve every value
//! against the active theme. Light/dark and accent changes therefore require no
//! change to this control.

use alloc::string::String;

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

use crate::kit::{classes, ButtonVariant, ControlSize};

/// Props for [`Button`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct ButtonProps {
    /// The visible text label.
    pub label: String,
    /// Fill treatment / emphasis.
    pub variant: ButtonVariant,
    /// Density step.
    pub size: ControlSize,
    /// Whether the button is non-interactive.
    pub disabled: bool,
    /// Optional leading content (e.g. an icon), rendered before the label.
    pub leading: Option<Element>,
    /// Optional trailing content, rendered after the label.
    pub trailing: Option<Element>,
}

impl ButtonProps {
    /// Creates props for a labelled button with default variant and size.
    #[must_use]
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            ..Self::default()
        }
    }

    /// Sets the fill variant.
    #[must_use]
    pub fn variant(mut self, variant: ButtonVariant) -> Self {
        self.variant = variant;
        self
    }

    /// Sets the density step.
    #[must_use]
    pub fn size(mut self, size: ControlSize) -> Self {
        self.size = size;
        self
    }

    /// Marks the button disabled.
    #[must_use]
    pub fn disabled(mut self, disabled: bool) -> Self {
        self.disabled = disabled;
        self
    }

    /// Sets leading content rendered before the label.
    #[must_use]
    pub fn leading(mut self, element: Element) -> Self {
        self.leading = Some(element);
        self
    }

    /// Sets trailing content rendered after the label.
    #[must_use]
    pub fn trailing(mut self, element: Element) -> Self {
        self.trailing = Some(element);
        self
    }
}

/// The button control. Zero-sized; all configuration lives in [`ButtonProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Button;

impl Button {
    /// The accessibility role a button exposes.
    #[must_use]
    pub const fn role() -> Role {
        Role::Button
    }

    /// The accessible name for a set of props (its label).
    #[must_use]
    pub fn accessible_name(props: &ButtonProps) -> &str {
        &props.label
    }
}

impl Component for Button {
    type Props = ButtonProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_();
        for name in classes(
            "pk-button",
            &[props.variant.suffix(), props.size.suffix()],
        ) {
            el = el.class(name);
        }
        if props.disabled {
            el = el.class("is-disabled");
        }

        if let Some(leading) = props.leading.clone() {
            el = el.child(leading);
        }
        if !props.label.is_empty() {
            el = el.child(Element::text(props.label.clone()).class("pk-button__label"));
        }
        if let Some(trailing) = props.trailing.clone() {
            el = el.child(trailing);
        }
        el
    }
}


/// Registers the `pk-button` class family: one base, five variants, three
/// sizes. The cascade applies `base -> variant -> size` in attachment order,
/// so a control emits `["pk-button", "pk-button--filled", "pk-button--md"]`
/// and gets layout from the base, surface from the variant, density from size.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, InteractionState, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok, transparent};

    // Base: a centered flex row, capsule radius, body typography. Disabled
    // dims via opacity (cheap, reads on any surface).
    sheet.insert(
        Class::new("pk-button")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::JustifyContent, kw(Keyword::Center))
            .with(StyleProp::Gap, tok("space.xs"))
            .with(StyleProp::BorderRadius, tok("radius.capsule"))
            .with(StyleProp::BorderWidth, StyleValue::px(0.0))
            .with(StyleProp::FontSize, tok("font.size.body"))
            .with(StyleProp::FontWeight, tok("font.weight.semibold"))
            .with_state(InteractionState::Disabled, StyleProp::Opacity, StyleValue::number(0.4)),
    );

    // Sizes map density to the spacing scale and shared control height.
    sheet.insert(
        Class::new("pk-button--sm")
            .with_padding_x(tok("space.sm"))
            .with_padding_y(tok("space.xs"))
            .with(StyleProp::Height, StyleValue::px(28.0))
            .with(StyleProp::FontSize, tok("font.size.subheadline")),
    );
    sheet.insert(
        Class::new("pk-button--md")
            .with_padding_x(tok("space.lg"))
            .with_padding_y(tok("space.sm"))
            .with(StyleProp::Height, tok("size.control.height"))
            .with(StyleProp::FontSize, tok("font.size.body")),
    );
    sheet.insert(
        Class::new("pk-button--lg")
            .with_padding_x(tok("space.xl"))
            .with_padding_y(tok("space.md"))
            .with(StyleProp::Height, StyleValue::px(44.0))
            .with(StyleProp::FontSize, tok("font.size.headline")),
    );

    // Filled: solid accent, white label, soft drop shadow.
    sheet.insert(
        Class::new("pk-button--filled")
            .with(StyleProp::BackgroundColor, tok("color.tint"))
            .with(StyleProp::Color, StyleValue::rgba8(255, 255, 255, 255))
            .with_shadow(0.0, 4.0, 12.0, tok("glass.shadow"))
            .with_state(InteractionState::Hover, StyleProp::Opacity, StyleValue::number(0.92))
            .with_state(InteractionState::Pressed, StyleProp::Opacity, StyleValue::number(0.82)),
    );

    // Tinted: translucent accent wash over glass, accent-colored label.
    sheet.insert(
        Class::new("pk-button--tinted")
            .with_glass(0.0, tok("color.fill.secondary"), Some(tok("glass.highlight")))
            .with(StyleProp::Color, tok("color.tint"))
            .with_state(InteractionState::Hover, StyleProp::Opacity, StyleValue::number(0.9))
            .with_state(InteractionState::Pressed, StyleProp::Opacity, StyleValue::number(0.78)),
    );

    // Gray: neutral glass surface, primary label.
    sheet.insert(
        Class::new("pk-button--gray")
            .with_glass(0.0, tok("color.fill"), Some(tok("glass.highlight")))
            .with(StyleProp::Color, tok("color.label"))
            .with_state(InteractionState::Hover, StyleProp::Opacity, StyleValue::number(0.9))
            .with_state(InteractionState::Pressed, StyleProp::Opacity, StyleValue::number(0.78)),
    );

    // Glass: the hero chrome surface — frosted tint, lit rim, drop shadow.
    sheet.insert(
        Class::new("pk-button--glass")
            .with_glass(0.0, tok("glass.tint"), Some(tok("glass.highlight")))
            .with(StyleProp::Color, tok("color.label"))
            .with_shadow(0.0, 8.0, 24.0, tok("glass.shadow"))
            .with_state(InteractionState::Hover, StyleProp::Opacity, StyleValue::number(0.94))
            .with_state(InteractionState::Pressed, StyleProp::Opacity, StyleValue::number(0.84)),
    );

    // Plain: no surface, accent label only.
    sheet.insert(
        Class::new("pk-button--plain")
            .with(StyleProp::BackgroundColor, transparent())
            .with(StyleProp::Color, tok("color.tint"))
            .with_state(InteractionState::Hover, StyleProp::Opacity, StyleValue::number(0.7))
            .with_state(InteractionState::Pressed, StyleProp::Opacity, StyleValue::number(0.55)),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use prism_ui::ElementKind;

    fn render(props: ButtonProps) -> Element {
        Button.render(&props)
    }

    #[test]
    fn attaches_base_variant_and_size_classes_in_order() {
        let el = render(
            ButtonProps::new("Save")
                .variant(ButtonVariant::Filled)
                .size(ControlSize::Medium),
        );
        assert_eq!(
            el.class_names(),
            ["pk-button", "pk-button--filled", "pk-button--md"]
        );
    }

    #[test]
    fn disabled_adds_marker_class() {
        let el = render(ButtonProps::new("X").disabled(true));
        assert!(el.class_names().iter().any(|c| c == "is-disabled"));
    }

    #[test]
    fn renders_label_as_text_child() {
        let el = render(ButtonProps::new("Save"));
        let label = el
            .child_elements()
            .iter()
            .find(|c| c.kind() == &ElementKind::Text)
            .expect("label child");
        assert_eq!(label.text_content(), Some("Save"));
    }

    #[test]
    fn leading_and_trailing_wrap_the_label() {
        let el = render(
            ButtonProps::new("Go")
                .leading(Element::box_().class("icon-left"))
                .trailing(Element::box_().class("icon-right")),
        );
        let kinds: Vec<_> = el.child_elements().iter().map(Element::kind).collect();
        assert_eq!(kinds.len(), 3);
        assert_eq!(kinds[1], &ElementKind::Text);
    }

    #[test]
    fn role_is_button() {
        assert_eq!(Button::role(), Role::Button);
    }
}
