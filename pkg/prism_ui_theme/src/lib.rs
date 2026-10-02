#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

//! Loom theming pipeline.
//!
//! This crate layers a design-token theming system over [`prism_ui_style`],
//! driven reactively by [`prism_ui_reactive`]. It is organized one concept per
//! module:
//!
//! - [`token`] — the mode-independent [`Palette`] of primitive design tokens
//!   (`color.gray.100`, `space.md`, `radius.md`), a name-tracking wrapper around
//!   [`prism_ui_style::TokenStore`] that reuses its reference-chain resolution
//!   and cycle detection.
//! - [`semantic`] — the [`SemanticMap`] indirection layer that redirects
//!   semantic names (`color.surface`) to primitives per [`ThemeMode`], so
//!   components never hard-code concrete values.
//! - [`theme`] — [`ThemeMode`] and the [`ThemeDefinition`] that bundles a
//!   palette with its semantics.
//! - [`compile`] — folding both layers into a flat [`CompiledTheme`] of literal
//!   values, with cycle and missing-reference detection.
//! - [`reactive`] — [`ReactiveTheme`], where switching appearance is a single
//!   [`prism_ui_reactive::Signal`] write whose cost is proportional to the nodes
//!   whose resolved value actually changed.
//! - [`logical`] — direction-aware `RTL` resolution of logical box sides
//!   (`inline-start`/`inline-end`) to physical [`prism_ui_style::StyleProp`]s.
//!
//! # Example
//!
//! ```
//! use prism_ui_reactive::Runtime;
//! use prism_ui_theme::{Direction, LogicalSide, LogicalSpacing, ReactiveTheme, ThemeDefinition, ThemeMode};
//! use prism_ui_style::{Color, StyleProp, StyleValue};
//!
//! // A theme switch is a single signal write; dependents recompute lazily.
//! let rt = Runtime::new();
//! let theme = ReactiveTheme::new(&rt, ThemeDefinition::studio(), ThemeMode::Light);
//! let surface = theme.color("color.surface");
//! assert_eq!(surface.get().unwrap(), Some(Color::rgba8(249, 250, 251, 255)));
//! theme.set_mode(ThemeMode::Dark);
//! assert_eq!(surface.get().unwrap(), Some(Color::rgba8(17, 24, 39, 255)));
//!
//! // Logical spacing mirrors with the writing direction.
//! let spacing = LogicalSpacing::new().pad(LogicalSide::InlineStart, StyleValue::px(12.0));
//! let rtl = spacing.resolve(Direction::Rtl);
//! assert_eq!(rtl.get(&StyleProp::PaddingRight), Some(&StyleValue::px(12.0)));
//! ```

extern crate alloc;

pub mod compile;
pub mod logical;
pub mod reactive;
pub mod semantic;
pub mod theme;
pub mod token;

pub use compile::{compile_theme, CompiledTheme};
pub use logical::{
    mirror_physical, mirror_prop_map, BoxEdge, Direction, LogicalEdge, LogicalSide, LogicalSpacing,
};
pub use reactive::ReactiveTheme;
pub use semantic::{SemanticMap, SemanticToken};
pub use theme::{ThemeDefinition, ThemeMode};
pub use token::Palette;
