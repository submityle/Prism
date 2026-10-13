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

    /// A modern, "Liquid Glass" inspired theme, authored for light
    /// and dark appearances.
    ///
    /// The palette carries the system primitives (each with an explicit light
    /// and `*.dark` value); the semantic layer exposes friendly names
    /// (`color.tint`, `color.background`, `color.label`, `color.fill`,
    /// `color.separator`, the twelve system accents) plus the
    /// [`ThemeDefinition::studio`]-compatible aliases `color.surface`,
    /// `color.text`, `color.border` and `color.primary`. Typography, spacing
    /// and corner-radius tokens round out the set. Values come straight from
    /// a DTCG design-token export, so semi-transparent labels/fills/separators
    /// keep their exact 8-bit alpha.
    ///
    /// Only light and dark are authored; other modes fall back to the light
    /// defaults.
    ///
    /// # Example
    ///
    /// ```
    /// use prism_ui_theme::{ThemeDefinition, ThemeMode};
    /// use prism_ui_style::Color;
    ///
    /// let theme = ThemeDefinition::glass();
    /// let light = theme.compile(&ThemeMode::Light).unwrap();
    /// let dark = theme.compile(&ThemeMode::Dark).unwrap();
    /// // System blue is the default tint and shifts between appearances.
    /// assert_eq!(light.color("color.tint"), Some(Color::rgba8(0, 136, 255, 255)));
    /// assert_eq!(dark.color("color.tint"), Some(Color::rgba8(0, 145, 255, 255)));
    /// ```
    #[must_use]
    pub fn glass() -> Self {
        let palette = Palette::new()
            // System colors (light / dark).
            .with("sys.red.light", StyleValue::rgba8(255, 56, 60, 255))
            .with("sys.red.dark", StyleValue::rgba8(255, 66, 69, 255))
            .with("sys.orange.light", StyleValue::rgba8(255, 141, 40, 255))
            .with("sys.orange.dark", StyleValue::rgba8(255, 146, 48, 255))
            .with("sys.yellow.light", StyleValue::rgba8(255, 204, 0, 255))
            .with("sys.yellow.dark", StyleValue::rgba8(255, 214, 0, 255))
            .with("sys.green.light", StyleValue::rgba8(52, 199, 89, 255))
            .with("sys.green.dark", StyleValue::rgba8(48, 209, 88, 255))
            .with("sys.mint.light", StyleValue::rgba8(0, 200, 179, 255))
            .with("sys.mint.dark", StyleValue::rgba8(0, 218, 195, 255))
            .with("sys.teal.light", StyleValue::rgba8(0, 195, 208, 255))
            .with("sys.teal.dark", StyleValue::rgba8(0, 210, 224, 255))
            .with("sys.cyan.light", StyleValue::rgba8(0, 192, 232, 255))
            .with("sys.cyan.dark", StyleValue::rgba8(60, 211, 254, 255))
            .with("sys.blue.light", StyleValue::rgba8(0, 136, 255, 255))
            .with("sys.blue.dark", StyleValue::rgba8(0, 145, 255, 255))
            .with("sys.indigo.light", StyleValue::rgba8(97, 85, 245, 255))
            .with("sys.indigo.dark", StyleValue::rgba8(109, 124, 255, 255))
            .with("sys.purple.light", StyleValue::rgba8(203, 48, 224, 255))
            .with("sys.purple.dark", StyleValue::rgba8(219, 52, 242, 255))
            .with("sys.pink.light", StyleValue::rgba8(255, 45, 85, 255))
            .with("sys.pink.dark", StyleValue::rgba8(255, 55, 95, 255))
            .with("sys.brown.light", StyleValue::rgba8(172, 127, 94, 255))
            .with("sys.brown.dark", StyleValue::rgba8(183, 138, 102, 255))
            // Grays 1-6 (light / dark).
            .with("sys.gray.1.light", StyleValue::rgba8(142, 142, 147, 255))
            .with("sys.gray.1.dark", StyleValue::rgba8(142, 142, 147, 255))
            .with("sys.gray.2.light", StyleValue::rgba8(174, 174, 178, 255))
            .with("sys.gray.2.dark", StyleValue::rgba8(99, 99, 102, 255))
            .with("sys.gray.3.light", StyleValue::rgba8(199, 199, 204, 255))
            .with("sys.gray.3.dark", StyleValue::rgba8(72, 72, 74, 255))
            .with("sys.gray.4.light", StyleValue::rgba8(209, 209, 214, 255))
            .with("sys.gray.4.dark", StyleValue::rgba8(58, 58, 60, 255))
            .with("sys.gray.5.light", StyleValue::rgba8(229, 229, 234, 255))
            .with("sys.gray.5.dark", StyleValue::rgba8(44, 44, 46, 255))
            .with("sys.gray.6.light", StyleValue::rgba8(242, 242, 247, 255))
            .with("sys.gray.6.dark", StyleValue::rgba8(28, 28, 30, 255))
            // Backgrounds, base (light / dark).
            .with(
                "sys.bg.primary.light",
                StyleValue::rgba8(255, 255, 255, 255),
            )
            .with("sys.bg.primary.dark", StyleValue::rgba8(0, 0, 0, 255))
            .with(
                "sys.bg.secondary.light",
                StyleValue::rgba8(242, 242, 247, 255),
            )
            .with("sys.bg.secondary.dark", StyleValue::rgba8(28, 28, 30, 255))
            .with(
                "sys.bg.tertiary.light",
                StyleValue::rgba8(255, 255, 255, 255),
            )
            .with("sys.bg.tertiary.dark", StyleValue::rgba8(44, 44, 46, 255))
            // Backgrounds, elevated dark (overlays / cards).
            .with(
                "sys.bg.elevated.primary.dark",
                StyleValue::rgba8(28, 28, 30, 255),
            )
            .with(
                "sys.bg.elevated.secondary.dark",
                StyleValue::rgba8(44, 44, 46, 255),
            )
            .with(
                "sys.bg.elevated.tertiary.dark",
                StyleValue::rgba8(58, 58, 60, 255),
            )
            // Labels (light / dark), hierarchical alpha.
            .with("sys.label.primary.light", StyleValue::rgba8(0, 0, 0, 255))
            .with(
                "sys.label.secondary.light",
                StyleValue::rgba8(60, 60, 67, 153),
            )
            .with(
                "sys.label.tertiary.light",
                StyleValue::rgba8(60, 60, 67, 77),
            )
            .with(
                "sys.label.quaternary.light",
                StyleValue::rgba8(60, 60, 67, 46),
            )
            .with("sys.label.quinary.light", StyleValue::rgba8(60, 60, 67, 23))
            .with(
                "sys.label.primary.dark",
                StyleValue::rgba8(255, 255, 255, 255),
            )
            .with(
                "sys.label.secondary.dark",
                StyleValue::rgba8(235, 235, 245, 153),
            )
            .with(
                "sys.label.tertiary.dark",
                StyleValue::rgba8(235, 235, 245, 77),
            )
            .with(
                "sys.label.quaternary.dark",
                StyleValue::rgba8(235, 235, 245, 41),
            )
            .with(
                "sys.label.quinary.dark",
                StyleValue::rgba8(235, 235, 245, 23),
            )
            // Fills (light / dark).
            .with(
                "sys.fill.primary.light",
                StyleValue::rgba8(120, 120, 120, 51),
            )
            .with(
                "sys.fill.secondary.light",
                StyleValue::rgba8(120, 120, 128, 41),
            )
            .with(
                "sys.fill.tertiary.light",
                StyleValue::rgba8(118, 118, 128, 31),
            )
            .with(
                "sys.fill.quaternary.light",
                StyleValue::rgba8(116, 116, 128, 20),
            )
            .with("sys.fill.quinary.light", StyleValue::rgba8(0, 0, 0, 5))
            .with(
                "sys.fill.primary.dark",
                StyleValue::rgba8(120, 120, 128, 92),
            )
            .with(
                "sys.fill.secondary.dark",
                StyleValue::rgba8(120, 120, 128, 82),
            )
            .with(
                "sys.fill.tertiary.dark",
                StyleValue::rgba8(118, 118, 128, 61),
            )
            .with(
                "sys.fill.quaternary.dark",
                StyleValue::rgba8(118, 118, 128, 46),
            )
            .with("sys.fill.quinary.dark", StyleValue::rgba8(118, 118, 128, 5))
            // Separators (light / dark).
            .with("sys.separator.light", StyleValue::rgba8(0, 0, 0, 31))
            .with("sys.separator.dark", StyleValue::rgba8(255, 255, 255, 31))
            .with(
                "sys.separator.opaque.light",
                StyleValue::rgba8(198, 198, 200, 255),
            )
            .with(
                "sys.separator.opaque.dark",
                StyleValue::rgba8(56, 56, 58, 255),
            )
            // Glass material primitives (translucent tint, lit rim, drop
            // shadow) for the approximate "Liquid Glass" surface. Values are
            // tuned to read as frost over arbitrary content, light / dark.
            .with(
                "sys.glass.tint.light",
                StyleValue::rgba8(255, 255, 255, 160),
            )
            .with("sys.glass.tint.dark", StyleValue::rgba8(30, 30, 30, 160))
            .with(
                "sys.glass.highlight.light",
                StyleValue::rgba8(255, 255, 255, 115),
            )
            .with(
                "sys.glass.highlight.dark",
                StyleValue::rgba8(255, 255, 255, 40),
            )
            .with("sys.glass.shadow.light", StyleValue::rgba8(0, 0, 0, 38))
            .with("sys.glass.shadow.dark", StyleValue::rgba8(0, 0, 0, 120))
            // Glass scalars (mode-independent): backdrop-blur hint and the
            // drop-shadow blur / vertical offset used by `glass_surface`.
            .with("glass.blur", StyleValue::px(20.0))
            .with("glass.shadow.blur", StyleValue::px(24.0))
            .with("glass.shadow.y", StyleValue::px(8.0))
            // Spacing scale (4pt grid).
            .with("space.xxs", StyleValue::px(2.0))
            .with("space.xs", StyleValue::px(4.0))
            .with("space.sm", StyleValue::px(8.0))
            .with("space.md", StyleValue::px(12.0))
            .with("space.lg", StyleValue::px(16.0))
            .with("space.xl", StyleValue::px(20.0))
            .with("space.xxl", StyleValue::px(24.0))
            .with("space.xxxl", StyleValue::px(32.0))
            // Corner radii.
            .with("radius.xs", StyleValue::px(6.0))
            .with("radius.sm", StyleValue::px(8.0))
            .with("radius.md", StyleValue::px(10.0))
            .with("radius.lg", StyleValue::px(12.0))
            .with("radius.alert", StyleValue::px(14.0))
            .with("radius.xl", StyleValue::px(16.0))
            .with("radius.capsule", StyleValue::px(18.0))
            // Control sizing.
            .with("size.control.height", StyleValue::px(36.0))
            // Typography: SF Pro text styles (size / line-height in px).
            .with("font.size.large-title", StyleValue::px(34.0))
            .with("font.line.large-title", StyleValue::px(41.0))
            .with("font.weight.large-title", StyleValue::number(400.0))
            .with("font.size.title1", StyleValue::px(28.0))
            .with("font.line.title1", StyleValue::px(34.0))
            .with("font.weight.title1", StyleValue::number(400.0))
            .with("font.size.title2", StyleValue::px(22.0))
            .with("font.line.title2", StyleValue::px(28.0))
            .with("font.weight.title2", StyleValue::number(400.0))
            .with("font.size.title3", StyleValue::px(20.0))
            .with("font.line.title3", StyleValue::px(25.0))
            .with("font.weight.title3", StyleValue::number(400.0))
            .with("font.size.headline", StyleValue::px(17.0))
            .with("font.line.headline", StyleValue::px(22.0))
            .with("font.weight.headline", StyleValue::number(600.0))
            .with("font.size.body", StyleValue::px(17.0))
            .with("font.line.body", StyleValue::px(22.0))
            .with("font.weight.body", StyleValue::number(400.0))
            .with("font.size.callout", StyleValue::px(16.0))
            .with("font.line.callout", StyleValue::px(21.0))
            .with("font.weight.callout", StyleValue::number(400.0))
            .with("font.size.subheadline", StyleValue::px(15.0))
            .with("font.line.subheadline", StyleValue::px(20.0))
            .with("font.weight.subheadline", StyleValue::number(400.0))
            .with("font.size.footnote", StyleValue::px(13.0))
            .with("font.line.footnote", StyleValue::px(18.0))
            .with("font.weight.footnote", StyleValue::number(400.0))
            .with("font.size.caption1", StyleValue::px(12.0))
            .with("font.line.caption1", StyleValue::px(16.0))
            .with("font.weight.caption1", StyleValue::number(400.0))
            .with("font.size.caption2", StyleValue::px(11.0))
            .with("font.line.caption2", StyleValue::px(13.0))
            .with("font.weight.caption2", StyleValue::number(400.0))
            // Named font weights.
            .with("font.weight.regular", StyleValue::number(400.0))
            .with("font.weight.medium", StyleValue::number(500.0))
            .with("font.weight.semibold", StyleValue::number(600.0))
            .with("font.weight.bold", StyleValue::number(700.0));

        let semantics = SemanticMap::new()
            // Tint (default accent = system blue).
            .with_default("color.tint", StyleValue::token("sys.blue.light"))
            .with_mode(
                "color.tint",
                ThemeMode::Dark,
                StyleValue::token("sys.blue.dark"),
            )
            // System accent colors (light / dark).
            .with_default("color.red", StyleValue::token("sys.red.light"))
            .with_mode(
                "color.red",
                ThemeMode::Dark,
                StyleValue::token("sys.red.dark"),
            )
            .with_default("color.orange", StyleValue::token("sys.orange.light"))
            .with_mode(
                "color.orange",
                ThemeMode::Dark,
                StyleValue::token("sys.orange.dark"),
            )
            .with_default("color.yellow", StyleValue::token("sys.yellow.light"))
            .with_mode(
                "color.yellow",
                ThemeMode::Dark,
                StyleValue::token("sys.yellow.dark"),
            )
            .with_default("color.green", StyleValue::token("sys.green.light"))
            .with_mode(
                "color.green",
                ThemeMode::Dark,
                StyleValue::token("sys.green.dark"),
            )
            .with_default("color.mint", StyleValue::token("sys.mint.light"))
            .with_mode(
                "color.mint",
                ThemeMode::Dark,
                StyleValue::token("sys.mint.dark"),
            )
            .with_default("color.teal", StyleValue::token("sys.teal.light"))
            .with_mode(
                "color.teal",
                ThemeMode::Dark,
                StyleValue::token("sys.teal.dark"),
            )
            .with_default("color.cyan", StyleValue::token("sys.cyan.light"))
            .with_mode(
                "color.cyan",
                ThemeMode::Dark,
                StyleValue::token("sys.cyan.dark"),
            )
            .with_default("color.blue", StyleValue::token("sys.blue.light"))
            .with_mode(
                "color.blue",
                ThemeMode::Dark,
                StyleValue::token("sys.blue.dark"),
            )
            .with_default("color.indigo", StyleValue::token("sys.indigo.light"))
            .with_mode(
                "color.indigo",
                ThemeMode::Dark,
                StyleValue::token("sys.indigo.dark"),
            )
            .with_default("color.purple", StyleValue::token("sys.purple.light"))
            .with_mode(
                "color.purple",
                ThemeMode::Dark,
                StyleValue::token("sys.purple.dark"),
            )
            .with_default("color.pink", StyleValue::token("sys.pink.light"))
            .with_mode(
                "color.pink",
                ThemeMode::Dark,
                StyleValue::token("sys.pink.dark"),
            )
            .with_default("color.brown", StyleValue::token("sys.brown.light"))
            .with_mode(
                "color.brown",
                ThemeMode::Dark,
                StyleValue::token("sys.brown.dark"),
            )
            // Backgrounds.
            .with_default(
                "color.background",
                StyleValue::token("sys.bg.primary.light"),
            )
            .with_mode(
                "color.background",
                ThemeMode::Dark,
                StyleValue::token("sys.bg.primary.dark"),
            )
            .with_default(
                "color.background.secondary",
                StyleValue::token("sys.bg.secondary.light"),
            )
            .with_mode(
                "color.background.secondary",
                ThemeMode::Dark,
                StyleValue::token("sys.bg.secondary.dark"),
            )
            .with_default(
                "color.background.tertiary",
                StyleValue::token("sys.bg.tertiary.light"),
            )
            .with_mode(
                "color.background.tertiary",
                ThemeMode::Dark,
                StyleValue::token("sys.bg.tertiary.dark"),
            )
            // Elevated surfaces (cards, sheets, popovers).
            .with_default("color.surface", StyleValue::token("sys.bg.primary.light"))
            .with_mode(
                "color.surface",
                ThemeMode::Dark,
                StyleValue::token("sys.bg.elevated.primary.dark"),
            )
            .with_default(
                "color.surface.secondary",
                StyleValue::token("sys.bg.secondary.light"),
            )
            .with_mode(
                "color.surface.secondary",
                ThemeMode::Dark,
                StyleValue::token("sys.bg.elevated.secondary.dark"),
            )
            .with_default(
                "color.surface.tertiary",
                StyleValue::token("sys.bg.tertiary.light"),
            )
            .with_mode(
                "color.surface.tertiary",
                ThemeMode::Dark,
                StyleValue::token("sys.bg.elevated.tertiary.dark"),
            )
            // Labels (text hierarchy).
            .with_default("color.label", StyleValue::token("sys.label.primary.light"))
            .with_mode(
                "color.label",
                ThemeMode::Dark,
                StyleValue::token("sys.label.primary.dark"),
            )
            .with_default(
                "color.label.secondary",
                StyleValue::token("sys.label.secondary.light"),
            )
            .with_mode(
                "color.label.secondary",
                ThemeMode::Dark,
                StyleValue::token("sys.label.secondary.dark"),
            )
            .with_default(
                "color.label.tertiary",
                StyleValue::token("sys.label.tertiary.light"),
            )
            .with_mode(
                "color.label.tertiary",
                ThemeMode::Dark,
                StyleValue::token("sys.label.tertiary.dark"),
            )
            .with_default(
                "color.label.quaternary",
                StyleValue::token("sys.label.quaternary.light"),
            )
            .with_mode(
                "color.label.quaternary",
                ThemeMode::Dark,
                StyleValue::token("sys.label.quaternary.dark"),
            )
            .with_default(
                "color.label.quinary",
                StyleValue::token("sys.label.quinary.light"),
            )
            .with_mode(
                "color.label.quinary",
                ThemeMode::Dark,
                StyleValue::token("sys.label.quinary.dark"),
            )
            // Fills (control backgrounds).
            .with_default("color.fill", StyleValue::token("sys.fill.primary.light"))
            .with_mode(
                "color.fill",
                ThemeMode::Dark,
                StyleValue::token("sys.fill.primary.dark"),
            )
            .with_default(
                "color.fill.secondary",
                StyleValue::token("sys.fill.secondary.light"),
            )
            .with_mode(
                "color.fill.secondary",
                ThemeMode::Dark,
                StyleValue::token("sys.fill.secondary.dark"),
            )
            .with_default(
                "color.fill.tertiary",
                StyleValue::token("sys.fill.tertiary.light"),
            )
            .with_mode(
                "color.fill.tertiary",
                ThemeMode::Dark,
                StyleValue::token("sys.fill.tertiary.dark"),
            )
            .with_default(
                "color.fill.quaternary",
                StyleValue::token("sys.fill.quaternary.light"),
            )
            .with_mode(
                "color.fill.quaternary",
                ThemeMode::Dark,
                StyleValue::token("sys.fill.quaternary.dark"),
            )
            .with_default(
                "color.fill.quinary",
                StyleValue::token("sys.fill.quinary.light"),
            )
            .with_mode(
                "color.fill.quinary",
                ThemeMode::Dark,
                StyleValue::token("sys.fill.quinary.dark"),
            )
            // Separators.
            .with_default("color.separator", StyleValue::token("sys.separator.light"))
            .with_mode(
                "color.separator",
                ThemeMode::Dark,
                StyleValue::token("sys.separator.dark"),
            )
            .with_default(
                "color.separator.opaque",
                StyleValue::token("sys.separator.opaque.light"),
            )
            .with_mode(
                "color.separator.opaque",
                ThemeMode::Dark,
                StyleValue::token("sys.separator.opaque.dark"),
            )
            // studio()-compatible aliases.
            .with_default("color.text", StyleValue::token("sys.label.primary.light"))
            .with_mode(
                "color.text",
                ThemeMode::Dark,
                StyleValue::token("sys.label.primary.dark"),
            )
            .with_default(
                "color.border",
                StyleValue::token("sys.separator.opaque.light"),
            )
            .with_mode(
                "color.border",
                ThemeMode::Dark,
                StyleValue::token("sys.separator.opaque.dark"),
            )
            .with_default("color.primary", StyleValue::token("sys.blue.light"))
            .with_mode(
                "color.primary",
                ThemeMode::Dark,
                StyleValue::token("sys.blue.dark"),
            )
            // Grouped neutral scale passthrough (mode-independent handles).
            .with_default("color.gray", StyleValue::token("sys.gray.1.light"))
            .with_mode(
                "color.gray",
                ThemeMode::Dark,
                StyleValue::token("sys.gray.1.dark"),
            )
            // Glass material (semantic handles for `glass_surface`).
            .with_default("glass.tint", StyleValue::token("sys.glass.tint.light"))
            .with_mode(
                "glass.tint",
                ThemeMode::Dark,
                StyleValue::token("sys.glass.tint.dark"),
            )
            .with_default(
                "glass.highlight",
                StyleValue::token("sys.glass.highlight.light"),
            )
            .with_mode(
                "glass.highlight",
                ThemeMode::Dark,
                StyleValue::token("sys.glass.highlight.dark"),
            )
            .with_default("glass.shadow", StyleValue::token("sys.glass.shadow.light"))
            .with_mode(
                "glass.shadow",
                ThemeMode::Dark,
                StyleValue::token("sys.glass.shadow.dark"),
            );

        Self::new(palette, semantics).with_modes([ThemeMode::Light, ThemeMode::Dark])
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
    fn glass_lists_light_and_dark_only() {
        let theme = ThemeDefinition::glass();
        assert_eq!(theme.modes, alloc::vec![ThemeMode::Light, ThemeMode::Dark]);
    }

    #[test]
    fn glass_tint_shifts_between_appearances() {
        let theme = ThemeDefinition::glass();
        let light = theme.compile(&ThemeMode::Light).unwrap();
        let dark = theme.compile(&ThemeMode::Dark).unwrap();
        assert_eq!(
            light.color("color.tint"),
            Some(prism_ui_style::Color::rgba8(0, 136, 255, 255))
        );
        assert_eq!(
            dark.color("color.tint"),
            Some(prism_ui_style::Color::rgba8(0, 145, 255, 255))
        );
    }

    #[test]
    fn glass_backgrounds_follow_mode() {
        let theme = ThemeDefinition::glass();
        let light = theme.compile(&ThemeMode::Light).unwrap();
        let dark = theme.compile(&ThemeMode::Dark).unwrap();
        assert_eq!(
            light.color("color.background"),
            Some(prism_ui_style::Color::rgba8(255, 255, 255, 255))
        );
        assert_eq!(
            dark.color("color.background"),
            Some(prism_ui_style::Color::rgba8(0, 0, 0, 255))
        );
    }

    #[test]
    fn glass_labels_keep_exact_alpha() {
        let theme = ThemeDefinition::glass();
        let light = theme.compile(&ThemeMode::Light).unwrap();
        // Secondary label is #3C3C4399 -> alpha 0x99 == 153.
        assert_eq!(
            light.color("color.label.secondary"),
            Some(prism_ui_style::Color::rgba8(60, 60, 67, 153))
        );
    }

    #[test]
    fn glass_exposes_typography_and_radius() {
        let theme = ThemeDefinition::glass();
        let light = theme.compile(&ThemeMode::Light).unwrap();
        assert_eq!(
            light.length("font.size.body"),
            Some(prism_ui_style::Length::Px(17.0))
        );
        assert_eq!(light.number("font.weight.headline"), Some(600.0));
        assert_eq!(
            light.length("radius.capsule"),
            Some(prism_ui_style::Length::Px(18.0))
        );
    }

    #[test]
    fn glass_material_tokens_shift_between_modes() {
        let theme = ThemeDefinition::glass();
        let light = theme.compile(&ThemeMode::Light).unwrap();
        let dark = theme.compile(&ThemeMode::Dark).unwrap();
        assert_eq!(
            light.color("glass.tint"),
            Some(prism_ui_style::Color::rgba8(255, 255, 255, 160))
        );
        assert_eq!(
            dark.color("glass.tint"),
            Some(prism_ui_style::Color::rgba8(30, 30, 30, 160))
        );
        // Highlight rim: light rim is the DTCG #FFFFFF73 (alpha 115).
        assert_eq!(
            light.color("glass.highlight"),
            Some(prism_ui_style::Color::rgba8(255, 255, 255, 115))
        );
        assert_eq!(
            dark.color("glass.highlight"),
            Some(prism_ui_style::Color::rgba8(255, 255, 255, 40))
        );
        // Drop shadow deepens in dark mode.
        assert_eq!(
            light.color("glass.shadow"),
            Some(prism_ui_style::Color::rgba8(0, 0, 0, 38))
        );
        assert_eq!(
            dark.color("glass.shadow"),
            Some(prism_ui_style::Color::rgba8(0, 0, 0, 120))
        );
    }

    #[test]
    fn glass_scalars_are_mode_independent() {
        let theme = ThemeDefinition::glass();
        let light = theme.compile(&ThemeMode::Light).unwrap();
        let dark = theme.compile(&ThemeMode::Dark).unwrap();
        assert_eq!(
            light.length("glass.blur"),
            Some(prism_ui_style::Length::Px(20.0))
        );
        assert_eq!(
            dark.length("glass.blur"),
            Some(prism_ui_style::Length::Px(20.0))
        );
        assert_eq!(
            light.length("glass.shadow.blur"),
            Some(prism_ui_style::Length::Px(24.0))
        );
        assert_eq!(
            light.length("glass.shadow.y"),
            Some(prism_ui_style::Length::Px(8.0))
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
