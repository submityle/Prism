//! Themes: a token store plus the responsive breakpoint scale.
//!
//! A [`Theme`] bundles the [`TokenStore`] used to resolve design tokens with
//! the ordered set of [`Breakpoint`]s the cascade considers. The
//! [`Theme::with_default_palette`] constructor seeds a small, opinionated
//! palette of color, spacing, radius and typography tokens — including a token
//! that references another token — so downstream crates and tests have a
//! realistic starting point.

use crate::selector::Breakpoint;
use crate::token::TokenStore;
use crate::value::StyleValue;

/// A theme: design tokens plus the breakpoint scale.
#[derive(Clone, Debug, PartialEq)]
pub struct Theme {
    /// The design tokens available to the cascade.
    pub tokens: TokenStore,
    /// The responsive breakpoints, in ascending min-width order.
    pub breakpoints: [Breakpoint; 5],
}

impl Theme {
    /// Creates a theme with an empty token store and the standard breakpoints.
    #[must_use]
    pub fn new() -> Self {
        Self {
            tokens: TokenStore::new(),
            breakpoints: Breakpoint::ALL,
        }
    }

    /// Creates a theme seeded with a small default token palette.
    ///
    /// The palette defines:
    ///
    /// * colors: `color.primary`, `color.danger`, `color.text`, `color.bg`,
    ///   and `color.action` (a reference to `color.primary`);
    /// * spacing: `space.xs` .. `space.xl`;
    /// * radii: `radius.sm` .. `radius.lg`;
    /// * typography: `font.sm` .. `font.lg`.
    #[must_use]
    pub fn with_default_palette() -> Self {
        let mut tokens = TokenStore::new();

        // Colors.
        tokens.insert("color.primary", StyleValue::rgba8(59, 130, 246, 255));
        tokens.insert("color.danger", StyleValue::rgba8(239, 68, 68, 255));
        tokens.insert("color.text", StyleValue::rgba8(17, 24, 39, 255));
        tokens.insert("color.bg", StyleValue::rgba8(255, 255, 255, 255));
        // A semantic alias that references another token.
        tokens.insert("color.action", StyleValue::token("color.primary"));

        // Spacing scale.
        tokens.insert("space.xs", StyleValue::px(4.0));
        tokens.insert("space.sm", StyleValue::px(8.0));
        tokens.insert("space.md", StyleValue::px(16.0));
        tokens.insert("space.lg", StyleValue::px(24.0));
        tokens.insert("space.xl", StyleValue::px(32.0));

        // Border radii.
        tokens.insert("radius.sm", StyleValue::px(4.0));
        tokens.insert("radius.md", StyleValue::px(8.0));
        tokens.insert("radius.lg", StyleValue::px(16.0));

        // Typography.
        tokens.insert("font.sm", StyleValue::px(12.0));
        tokens.insert("font.md", StyleValue::px(16.0));
        tokens.insert("font.lg", StyleValue::px(20.0));

        Self {
            tokens,
            breakpoints: Breakpoint::ALL,
        }
    }
}

impl Default for Theme {
    fn default() -> Self {
        Self::with_default_palette()
    }
}
