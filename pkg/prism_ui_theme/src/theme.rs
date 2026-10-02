//! Theme modes and the full theme definition.
//!
//! A [`ThemeMode`] names the active appearance (light, dark, high-contrast or a
//! branded [`ThemeMode::Custom`] variant). A [`ThemeDefinition`] bundles the
//! mode-independent [`Palette`] of primitives with the [`SemanticMap`] that
//! redirects semantic names per mode. Compiling a definition for a mode (see
//! [`crate::compile`]) folds the two layers into concrete values.

use alloc::string::String;
use alloc::vec::Vec;

use prism_ui_style::StyleValue;

use crate::semantic::SemanticMap;
use crate::token::Palette;

/// The active theme appearance.
///
/// Switching between modes is a single reactive write (see
/// [`crate::ReactiveTheme`]); the cost of recomputation is proportional to the
/// tokens whose resolved value actually changed, not to the theme size.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum ThemeMode {
    /// A light appearance (dark text on light surfaces).
    #[default]
    Light,
    /// A dark appearance (light text on dark surfaces).
    Dark,
    /// A maximum-contrast appearance for accessibility.
    HighContrast,
    /// A named custom or brand theme.
    Custom(String),
}

impl ThemeMode {
    /// Builds a [`ThemeMode::Custom`] from a name.
    #[must_use]
    pub fn custom(name: impl Into<String>) -> Self {
        ThemeMode::Custom(name.into())
    }

    /// Returns the stable string label for this mode.
    ///
    /// Built-in modes map to fixed labels; a custom mode returns its name.
    #[must_use]
    pub fn label(&self) -> &str {
        match self {
            ThemeMode::Light => "light",
            ThemeMode::Dark => "dark",
            ThemeMode::HighContrast => "high-contrast",
            ThemeMode::Custom(name) => name,
        }
    }

    /// Returns `true` if this is a custom (user-defined) mode.
    #[must_use]
    pub fn is_custom(&self) -> bool {
        matches!(self, ThemeMode::Custom(_))
    }
}

/// A complete theme: primitive palette plus semantic indirection.
///
/// The `modes` field lists the appearances the definition is authored for; it
/// is advisory metadata (compilation accepts any mode and falls back to
/// semantic defaults) that lets tooling enumerate the supported themes.
///
/// # Example
///
/// ```
/// use prism_ui_theme::{ThemeDefinition, ThemeMode};
///
/// let theme = ThemeDefinition::studio();
/// // Surface is a light gray in light mode and a dark gray in dark mode.
/// let light = theme.compile(&ThemeMode::Light).unwrap();
/// let dark = theme.compile(&ThemeMode::Dark).unwrap();
/// assert_ne!(light.get("color.surface"), dark.get("color.surface"));
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct ThemeDefinition {
    /// The mode-independent primitive tokens.
    pub palette: Palette,
    /// The semantic indirection layer.
    pub semantics: SemanticMap,
    /// The appearances this definition is authored for.
    pub modes: Vec<ThemeMode>,
}

impl ThemeDefinition {
    /// Creates a definition from a palette and semantic map, with `modes`
    /// defaulting to the three built-in appearances.
    #[must_use]
    pub fn new(palette: Palette, semantics: SemanticMap) -> Self {
        Self {
            palette,
            semantics,
            modes: alloc::vec![ThemeMode::Light, ThemeMode::Dark, ThemeMode::HighContrast,],
        }
    }

    /// Replaces the advertised mode list, returning `self` for chaining.
    #[must_use]
    pub fn with_modes(mut self, modes: impl IntoIterator<Item = ThemeMode>) -> Self {
        self.modes = modes.into_iter().collect();
        self
    }

    /// A realistic default theme: a gray/blue palette with light, dark and
    /// high-contrast semantic mappings for surface, text, border and primary.
    #[must_use]
    pub fn studio() -> Self {
        let palette = Palette::new()
            // Neutral ramp.
            .with("color.gray.50", StyleValue::rgba8(249, 250, 251, 255))
            .with("color.gray.100", StyleValue::rgba8(243, 244, 246, 255))
            .with("color.gray.200", StyleValue::rgba8(229, 231, 235, 255))
            .with("color.gray.700", StyleValue::rgba8(55, 65, 81, 255))
            .with("color.gray.800", StyleValue::rgba8(31, 41, 55, 255))
            .with("color.gray.900", StyleValue::rgba8(17, 24, 39, 255))
            // Brand ramp.
            .with("color.blue.500", StyleValue::rgba8(59, 130, 246, 255))
            .with("color.blue.600", StyleValue::rgba8(37, 99, 235, 255))
            // Absolutes.
            .with("color.white", StyleValue::rgba8(255, 255, 255, 255))
            .with("color.black", StyleValue::rgba8(0, 0, 0, 255))
            // Spacing + radius primitives.
            .with("space.sm", StyleValue::px(8.0))
            .with("space.md", StyleValue::px(16.0))
            .with("space.lg", StyleValue::px(24.0))
            .with("radius.md", StyleValue::px(8.0));

        let semantics = SemanticMap::new()
            // Surface: light gray / dark gray / pure black.
            .with_default("color.surface", StyleValue::token("color.gray.50"))
            .with_mode(
                "color.surface",
                ThemeMode::Dark,
                StyleValue::token("color.gray.900"),
            )
            .with_mode(
                "color.surface",
                ThemeMode::HighContrast,
                StyleValue::token("color.black"),
            )
            // Text: dark / light / pure white.
            .with_default("color.text", StyleValue::token("color.gray.900"))
            .with_mode(
                "color.text",
                ThemeMode::Dark,
                StyleValue::token("color.gray.50"),
            )
            .with_mode(
                "color.text",
                ThemeMode::HighContrast,
                StyleValue::token("color.white"),
            )
            // Border: subtle / muted / white.
            .with_default("color.border", StyleValue::token("color.gray.200"))
            .with_mode(
                "color.border",
                ThemeMode::Dark,
                StyleValue::token("color.gray.700"),
            )
            .with_mode(
                "color.border",
                ThemeMode::HighContrast,
                StyleValue::token("color.white"),
            )
            // Primary: constant brand color across light and dark, deepened for
            // high contrast. (Used to show that unchanged tokens do not disturb
            // their dependents on a mode switch.)
            .with_default("color.primary", StyleValue::token("color.blue.500"))
            .with_mode(
                "color.primary",
                ThemeMode::HighContrast,
                StyleValue::token("color.blue.600"),
            )
            // A spacing semantic that never changes with the mode.
            .with_default("space.gutter", StyleValue::token("space.md"));

        Self::new(palette, semantics)
    }

    /// Compiles this definition for `mode`, resolving every token to a literal.
    ///
    /// This is a convenience wrapper over [`crate::compile::compile_theme`].
    ///
    /// # Errors
    ///
    /// Propagates any resolution error (missing token or reference cycle).
    pub fn compile(
        &self,
        mode: &ThemeMode,
    ) -> Result<crate::compile::CompiledTheme, prism_ui_style::StyleError> {
        crate::compile::compile_theme(self, mode)
    }
}

impl Default for ThemeDefinition {
    fn default() -> Self {
        Self::studio()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_labels_are_stable() {
        assert_eq!(ThemeMode::Light.label(), "light");
        assert_eq!(ThemeMode::Dark.label(), "dark");
        assert_eq!(ThemeMode::HighContrast.label(), "high-contrast");
        assert_eq!(ThemeMode::custom("brand").label(), "brand");
    }

    #[test]
    fn custom_detection_and_default() {
        assert!(ThemeMode::custom("x").is_custom());
        assert!(!ThemeMode::Light.is_custom());
        assert_eq!(ThemeMode::default(), ThemeMode::Light);
    }

    #[test]
    fn studio_lists_three_builtin_modes() {
        let theme = ThemeDefinition::studio();
        assert_eq!(
            theme.modes,
            alloc::vec![ThemeMode::Light, ThemeMode::Dark, ThemeMode::HighContrast],
        );
    }

    #[test]
    fn with_modes_overrides_metadata() {
        let theme =
            ThemeDefinition::studio().with_modes([ThemeMode::Light, ThemeMode::custom("brand")]);
        assert_eq!(
            theme.modes,
            alloc::vec![ThemeMode::Light, ThemeMode::custom("brand")],
        );
    }
}
