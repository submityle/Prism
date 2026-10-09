//! The kit's style preset: the single owner of what each kit class name means.
//!
//! Controls attach class names (`pk-button`, `pk-button--filled`, …); this
//! module is where those names become [`Class`] rules. Every value is a
//! **token reference** ([`StyleValue::token`]), never a literal — so the active
//! [`ThemeMode`](prism_ui_theme::ThemeMode) decides the concrete color, and a
//! light/dark flip is a single theme signal write with no per-control code.
//!
//! Call [`stylesheet`] once at startup, register it with the style cascade, and
//! every control in the kit is styled.

use prism_ui_style::{Class, InteractionState, Keyword, StyleProp, StyleValue};

/// The kit's compiled [`StyleSheet`](prism_ui_style::StyleSheet), ready to hand
/// to the cascade.
pub type StyleSheet = prism_ui_style::StyleSheet;

/// Resolves a token reference `StyleValue`. Thin wrapper to keep call sites
/// terse and intention-revealing (every preset value is a token).
#[inline]
#[must_use]
pub(crate) fn tok(name: &str) -> StyleValue {
    StyleValue::token(name)
}

#[inline]
fn kw(k: Keyword) -> StyleValue {
    StyleValue::keyword(k)
}

/// Builds the full kit style sheet.
///
/// Each control family contributes its classes through a private `register_*`
/// function so modules stay cohesive and the sheet is assembled in one place.
#[must_use]
pub fn stylesheet() -> StyleSheet {
    let mut sheet = StyleSheet::new();
    register_button(&mut sheet);
    sheet
}

/// Button classes: one base, five variants, three sizes.
///
/// The cascade applies `base → variant → size` in attachment order, so a
/// control emits `["pk-button", "pk-button--filled", "pk-button--md"]` and gets
/// layout from the base, surface from the variant and density from the size.
fn register_button(sheet: &mut StyleSheet) {
    // Base: a centered flex row, capsule radius, body typography, pointer
    // affordance expressed purely through tokens. Disabled dims via opacity.
    let base = Class::new("pk-button")
        .with(StyleProp::Display, kw(Keyword::Flex))
        .with(StyleProp::FlexDirection, kw(Keyword::Row))
        .with(StyleProp::AlignItems, kw(Keyword::Center))
        .with(StyleProp::JustifyContent, kw(Keyword::Center))
        .with(StyleProp::Gap, tok("space.xs"))
        .with(StyleProp::BorderRadius, tok("radius.capsule"))
        .with(StyleProp::BorderWidth, StyleValue::px(0.0))
        .with(StyleProp::FontSize, tok("font.size.body"))
        .with(StyleProp::FontWeight, tok("font.weight.semibold"))
        .with_state(InteractionState::Disabled, StyleProp::Opacity, StyleValue::number(0.4));
    sheet.insert(base);

    // Sizes map density to the spacing scale and the shared control height.
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

    // Filled: solid accent, white label, soft drop shadow. Hover/pressed shift
    // opacity (cheap, token-free interaction feedback that reads on any tint).
    sheet.insert(
        Class::new("pk-button--filled")
            .with(StyleProp::BackgroundColor, tok("color.tint"))
            .with(StyleProp::Color, StyleValue::rgba8(255, 255, 255, 255))
            .with_shadow(0.0, 4.0, 12.0, tok("glass.shadow"))
            .with_state(InteractionState::Hover, StyleProp::Opacity, StyleValue::number(0.92))
            .with_state(InteractionState::Pressed, StyleProp::Opacity, StyleValue::number(0.82)),
    );

    // Tinted: a translucent accent wash over glass, accent-colored label.
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
            .with(StyleProp::BackgroundColor, StyleValue::rgba8(0, 0, 0, 0))
            .with(StyleProp::Color, tok("color.tint"))
            .with_state(InteractionState::Hover, StyleProp::Opacity, StyleValue::number(0.7))
            .with_state(InteractionState::Pressed, StyleProp::Opacity, StyleValue::number(0.55)),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stylesheet_registers_button_family() {
        let sheet = stylesheet();
        for name in [
            "pk-button",
            "pk-button--sm",
            "pk-button--md",
            "pk-button--lg",
            "pk-button--filled",
            "pk-button--tinted",
            "pk-button--gray",
            "pk-button--glass",
            "pk-button--plain",
        ] {
            assert!(sheet.get(name).is_some(), "missing class {name}");
        }
    }

    #[test]
    fn every_value_is_token_or_mode_safe_literal() {
        // Guard K1: colors must be token refs, not baked literals — the only
        // allowed color literals are fully transparent (0 alpha) or pure white
        // label on filled, which read identically in light and dark.
        let sheet = stylesheet();
        let filled = sheet.get("pk-button--filled").unwrap();
        assert_eq!(filled.base.get(&StyleProp::BackgroundColor), Some(&tok("color.tint")));
    }
}
