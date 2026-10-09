//! Shared vocabulary for every control in the kit.
//!
//! These are the small, orthogonal enums and helpers that keep control APIs
//! consistent: one way to express size, one way to express a semantic tone,
//! and one way to turn a `(block, variant, size)` triple into the class names
//! the [`preset`](crate::preset) layer knows how to style.

use alloc::string::String;
use alloc::vec::Vec;

/// A control's size step.
///
/// Sizes map to spacing/typography tokens in the preset layer rather than to
/// hard-coded pixels, so a size change is a token lookup, not a magic number.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum ControlSize {
    /// Compact density (toolbars, dense tables).
    Small,
    /// The default density.
    #[default]
    Medium,
    /// Roomy density (primary call-to-action, touch targets).
    Large,
}

impl ControlSize {
    /// The modifier suffix used in class names (e.g. `pk-button--sm`).
    #[must_use]
    pub const fn suffix(self) -> &'static str {
        match self {
            ControlSize::Small => "sm",
            ControlSize::Medium => "md",
            ControlSize::Large => "lg",
        }
    }
}

/// A control's semantic tone.
///
/// Tone selects *which* accent token a control paints with; it never selects a
/// literal color. `Neutral` falls back to the label/fill ramp.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Tone {
    /// The theme's primary tint.
    #[default]
    Accent,
    /// Neutral gray ramp (no accent).
    Neutral,
    /// Success / positive affordance.
    Success,
    /// Warning / caution affordance.
    Warning,
    /// Danger / destructive affordance.
    Danger,
}

impl Tone {
    /// The modifier suffix used in class names (e.g. `pk-button--danger`).
    #[must_use]
    pub const fn suffix(self) -> &'static str {
        match self {
            Tone::Accent => "accent",
            Tone::Neutral => "neutral",
            Tone::Success => "success",
            Tone::Warning => "warning",
            Tone::Danger => "danger",
        }
    }

    /// The accent color token this tone resolves against.
    #[must_use]
    pub const fn color_token(self) -> &'static str {
        match self {
            Tone::Accent => "color.tint",
            Tone::Neutral => "color.gray",
            Tone::Success => "color.green",
            Tone::Warning => "color.orange",
            Tone::Danger => "color.red",
        }
    }
}

/// The fill treatment of an interactive surface, following the Liquid-Glass
/// hierarchy of emphasis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum ButtonVariant {
    /// Solid accent fill, white label — highest emphasis.
    #[default]
    Filled,
    /// Translucent accent wash over a glass surface — medium emphasis.
    Tinted,
    /// Neutral gray glass surface — low emphasis.
    Gray,
    /// Glass surface with only a lit edge — chrome / toolbar.
    Glass,
    /// Text only, no surface — lowest emphasis.
    Plain,
}

impl ButtonVariant {
    /// The modifier suffix used in class names (e.g. `pk-button--tinted`).
    #[must_use]
    pub const fn suffix(self) -> &'static str {
        match self {
            ButtonVariant::Filled => "filled",
            ButtonVariant::Tinted => "tinted",
            ButtonVariant::Gray => "gray",
            ButtonVariant::Glass => "glass",
            ButtonVariant::Plain => "plain",
        }
    }
}

/// Builds the ordered list of class names for a block/element/modifier triple.
///
/// Controls attach these names and nothing else; the [`preset`](crate::preset)
/// layer is the single owner of what each name resolves to. The returned order
/// is **base first, modifiers after**, matching the cascade order the style
/// engine applies.
///
/// ```
/// use prism_ui_component_kit::kit::{classes, ButtonVariant, ControlSize};
///
/// let names = classes("pk-button", &[
///     ButtonVariant::Filled.suffix(),
///     ControlSize::Medium.suffix(),
/// ]);
/// assert_eq!(names, ["pk-button", "pk-button--filled", "pk-button--md"]);
/// ```
#[must_use]
pub fn classes(block: &str, modifiers: &[&str]) -> Vec<String> {
    let mut out = Vec::with_capacity(modifiers.len() + 1);
    out.push(String::from(block));
    for m in modifiers {
        let mut name = String::with_capacity(block.len() + 2 + m.len());
        name.push_str(block);
        name.push_str("--");
        name.push_str(m);
        out.push(name);
    }
    out
}
