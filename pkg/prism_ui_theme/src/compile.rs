//! Theme compilation: folding the palette and semantic layers into literals.
//!
//! [`compile_theme`] takes a [`ThemeDefinition`] and a [`ThemeMode`] and returns
//! a [`CompiledTheme`]: a flat table mapping every primitive and semantic token
//! name to a fully-resolved [`StyleValue`]. Resolution reuses
//! [`prism_ui_style::TokenStore`], so reference chains and cycle detection are
//! shared with the style crate rather than duplicated here.
//!
//! The algorithm layers the semantic mappings for the chosen mode *over* the
//! palette inside one combined store, then resolves each name. Because semantic
//! entries are inserted as ordinary tokens, a cycle that runs semantic ->
//! semantic -> palette -> semantic is caught by the same detector.

use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};

use prism_ui_style::{Color, Length, StyleError, StyleValue, TokenStore};

use crate::theme::{ThemeDefinition, ThemeMode};

/// A theme compiled for one mode: token names mapped to literal values.
///
/// Every value is concrete; no [`StyleValue::TokenRef`] remains. Lookup helpers
/// ([`CompiledTheme::color`], [`CompiledTheme::length`],
/// [`CompiledTheme::number`]) extract typed values for the common cases.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CompiledTheme {
    mode: ThemeMode,
    values: BTreeMap<String, StyleValue>,
}

impl CompiledTheme {
    /// Returns the mode this theme was compiled for.
    #[must_use]
    pub fn mode(&self) -> &ThemeMode {
        &self.mode
    }

    /// Returns the resolved literal value for `name`, if present.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&StyleValue> {
        self.values.get(name)
    }

    /// Returns the resolved [`Color`] for `name`, if it resolves to a color.
    #[must_use]
    pub fn color(&self, name: &str) -> Option<Color> {
        match self.values.get(name) {
            Some(StyleValue::Color(color)) => Some(*color),
            _ => None,
        }
    }

    /// Returns the resolved [`Length`] for `name`, if it resolves to a length.
    #[must_use]
    pub fn length(&self, name: &str) -> Option<Length> {
        match self.values.get(name) {
            Some(StyleValue::Length(length)) => Some(*length),
            _ => None,
        }
    }

    /// Returns the resolved number for `name`, if it resolves to a number.
    #[must_use]
    pub fn number(&self, name: &str) -> Option<f32> {
        match self.values.get(name) {
            Some(StyleValue::Number(value)) => Some(*value),
            _ => None,
        }
    }

    /// Iterates over `(name, value)` pairs in sorted name order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &StyleValue)> {
        self.values
            .iter()
            .map(|(name, value)| (name.as_str(), value))
    }

    /// Returns the number of resolved tokens.
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns `true` if there are no resolved tokens.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// Compiles `definition` for `mode` into a [`CompiledTheme`].
///
/// Every primitive and every semantic token (resolved for `mode`, falling back
/// to its default) is folded into one store and resolved to a literal.
///
/// # Errors
///
/// Returns [`StyleError::UnknownToken`] when a token (or a semantic default)
/// references a name that does not exist, and [`StyleError::CycleDetected`]
/// when resolution revisits a name.
///
/// # Example
///
/// ```
/// use prism_ui_theme::{compile_theme, ThemeDefinition, ThemeMode};
/// use prism_ui_style::Color;
///
/// let theme = ThemeDefinition::studio();
/// let dark = compile_theme(&theme, &ThemeMode::Dark).unwrap();
/// // `color.surface` resolves all the way to the dark gray primitive.
/// assert_eq!(dark.color("color.surface"), Some(Color::rgba8(17, 24, 39, 255)));
/// ```
pub fn compile_theme(
    definition: &ThemeDefinition,
    mode: &ThemeMode,
) -> Result<CompiledTheme, StyleError> {
    // Start from the palette's store and layer the mode's semantic mappings on
    // top in a single combined store, so resolution and cycle detection span
    // both layers uniformly.
    let mut store: TokenStore = definition.palette.store().clone();
    for name in definition.semantics.names() {
        let value = definition
            .semantics
            .resolve_for(name, mode)
            .ok_or_else(|| StyleError::UnknownToken(name.to_string()))?;
        store.insert(name, value.clone());
    }

    // Resolve every known name (palette primitives + semantic tokens).
    let mut values = BTreeMap::new();
    for name in definition.palette.names() {
        values.insert(name.to_string(), store.resolve(name)?);
    }
    for name in definition.semantics.names() {
        values.insert(name.to_string(), store.resolve(name)?);
    }

    Ok(CompiledTheme {
        mode: mode.clone(),
        values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantic::SemanticMap;
    use crate::token::Palette;
    use alloc::vec::Vec;

    #[test]
    fn semantic_resolves_through_palette_per_mode() {
        let theme = ThemeDefinition::studio();
        let light = compile_theme(&theme, &ThemeMode::Light).unwrap();
        let dark = compile_theme(&theme, &ThemeMode::Dark).unwrap();

        assert_eq!(
            light.color("color.surface"),
            Some(Color::rgba8(249, 250, 251, 255))
        );
        assert_eq!(
            dark.color("color.surface"),
            Some(Color::rgba8(17, 24, 39, 255))
        );
        assert_eq!(
            light.color("color.text"),
            Some(Color::rgba8(17, 24, 39, 255))
        );
        assert_eq!(
            dark.color("color.text"),
            Some(Color::rgba8(249, 250, 251, 255))
        );
    }

    #[test]
    fn high_contrast_uses_absolute_colors() {
        let theme = ThemeDefinition::studio();
        let hc = compile_theme(&theme, &ThemeMode::HighContrast).unwrap();
        assert_eq!(hc.color("color.surface"), Some(Color::rgba8(0, 0, 0, 255)));
        assert_eq!(
            hc.color("color.text"),
            Some(Color::rgba8(255, 255, 255, 255))
        );
    }

    #[test]
    fn custom_mode_uses_semantic_defaults() {
        let theme = ThemeDefinition::studio();
        let brand = compile_theme(&theme, &ThemeMode::custom("brand")).unwrap();
        // No per-mode entry, so the light default applies.
        assert_eq!(
            brand.color("color.surface"),
            Some(Color::rgba8(249, 250, 251, 255))
        );
    }

    #[test]
    fn compiled_table_includes_primitives_and_semantics() {
        let theme = ThemeDefinition::studio();
        let light = compile_theme(&theme, &ThemeMode::Light).unwrap();
        assert!(light.get("color.gray.50").is_some());
        assert!(light.get("color.surface").is_some());
        assert_eq!(light.length("space.gutter"), Some(Length::Px(16.0)));
        assert_eq!(light.mode(), &ThemeMode::Light);
    }

    #[test]
    fn cycle_in_semantic_layer_is_detected() {
        let palette = Palette::new();
        let semantics = SemanticMap::new()
            .with_default("a", StyleValue::token("b"))
            .with_default("b", StyleValue::token("a"));
        let theme = ThemeDefinition::new(palette, semantics);
        let err = compile_theme(&theme, &ThemeMode::Light).unwrap_err();
        assert!(matches!(err, StyleError::CycleDetected(_)));
    }

    #[test]
    fn missing_reference_is_reported() {
        let palette = Palette::new();
        let semantics =
            SemanticMap::new().with_default("color.x", StyleValue::token("color.missing"));
        let theme = ThemeDefinition::new(palette, semantics);
        let err = compile_theme(&theme, &ThemeMode::Light).unwrap_err();
        assert_eq!(err, StyleError::UnknownToken("color.missing".to_string()));
    }

    #[test]
    fn iter_is_sorted_and_counts_match() {
        let theme = ThemeDefinition::studio();
        let light = compile_theme(&theme, &ThemeMode::Light).unwrap();
        let names: Vec<&str> = light.iter().map(|(name, _)| name).collect();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        assert_eq!(names, sorted);
        assert_eq!(light.len(), names.len());
        assert!(!light.is_empty());
    }
}
