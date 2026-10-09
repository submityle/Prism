//! The kit's style preset: the single aggregation point for all kit classes.
//!
//! Controls attach class names (`pk-button`, `pk-field`, …); each **module**
//! owns the [`Class`](prism_ui_style::Class) rules for its own controls via a
//! `register_styles(&mut StyleSheet)` function. [`stylesheet`] simply calls
//! every module's registrar in turn, so adding a module never edits a shared
//! file — the modules are developed independently.
//!
//! Every value a module emits must be a **token reference**
//! ([`StyleValue::token`]), never a literal, so the active
//! [`ThemeMode`](prism_ui_theme::ThemeMode) decides concrete colors and a
//! light/dark flip is one theme signal write. The only sanctioned literals are
//! mode-invariant ones (fully transparent, or pure white/black that read
//! identically in both appearances).

use prism_ui_style::{Keyword, StyleValue};

/// The kit's style sheet type (reused from `prism_ui_style`).
pub type StyleSheet = prism_ui_style::StyleSheet;

/// Resolves a token reference. Thin, intention-revealing wrapper used by every
/// module's style registrar so call sites read as "this value is a token".
#[inline]
#[must_use]
pub(crate) fn tok(name: &str) -> StyleValue {
    StyleValue::token(name)
}

/// Builds a keyword value (display/flex/justify/align).
#[inline]
#[must_use]
pub(crate) fn kw(k: Keyword) -> StyleValue {
    StyleValue::keyword(k)
}

/// A fully transparent color — the only sanctioned "no surface" literal.
#[inline]
#[must_use]
pub(crate) fn transparent() -> StyleValue {
    StyleValue::rgba8(0, 0, 0, 0)
}

/// Builds the full kit style sheet by aggregating every module's registrar.
///
/// Modules register in layer order (basics first) so lower-emphasis base rules
/// are inserted before anything that may override them. Insertion order into a
/// `StyleSheet` does not affect cascade priority (that is class-attachment
/// order on the element), but a stable order keeps the sheet reproducible.
#[must_use]
pub fn stylesheet() -> StyleSheet {
    let mut sheet = StyleSheet::new();
    crate::basics::register_styles(&mut sheet);
    crate::inputs::register_styles(&mut sheet);
    crate::pickers::register_styles(&mut sheet);
    crate::display::register_styles(&mut sheet);
    crate::containers::register_styles(&mut sheet);
    crate::feedback::register_styles(&mut sheet);
    crate::nav::register_styles(&mut sheet);
    crate::editor::register_styles(&mut sheet);
    crate::node_editor::register_styles(&mut sheet);
    crate::motion::register_styles(&mut sheet);
    crate::utils::register_styles(&mut sheet);
    sheet
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
}
