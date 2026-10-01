//! `prism_ui_style` — design tokens, classes, selectors and the cascade for
//! Prism's Loom UI/scene notation.
//!
//! This crate is a pure-data, engine-agnostic style layer inspired by utility
//! CSS (such as Tailwind) and Unity's `USS`. It is deliberately free of any
//! renderer or layout types so style sheets can be authored, serialized and
//! hot-reloaded independently of the rest of the engine.
//!
//! # Pipeline
//!
//! * [`value`] defines [`StyleProp`] (the styleable properties) and
//!   [`StyleValue`] (the values they take, including token references).
//! * [`token`] defines [`DesignToken`] and the [`TokenStore`] that resolves
//!   token-reference chains with cycle detection.
//! * [`class`] defines [`Class`] (a property bag with per-state and
//!   per-breakpoint overrides) and the [`StyleSheet`] that collects them.
//! * [`selector`] defines [`InteractionState`], [`Breakpoint`] and the
//!   [`MatchContext`] used to decide which overrides apply.
//! * [`cascade`] merges an ordered class list into a [`ComputedStyle`] with all
//!   token references resolved.
//! * [`theme`] bundles a [`TokenStore`] with the breakpoint scale and ships a
//!   small default palette.
//!
//! Resolution never panics; failures are reported as a [`StyleError`].
//!
//! # Example
//!
//! ```
//! use prism_ui_style::{
//!     Cascade, Class, InteractionState, MatchContext, StyleProp, StyleSheet,
//!     StyleValue, Theme,
//! };
//!
//! let theme = Theme::with_default_palette();
//! let sheet = StyleSheet::new().with_class(
//!     Class::new("button")
//!         .with(StyleProp::BackgroundColor, StyleValue::token("color.primary"))
//!         .with_padding_x(StyleValue::token("space.md"))
//!         .with_state(
//!             InteractionState::Hover,
//!             StyleProp::BackgroundColor,
//!             StyleValue::token("color.danger"),
//!         ),
//! );
//!
//! let ctx = MatchContext::new(1024.0).with_state(InteractionState::Hover);
//! let computed = Cascade::new(&sheet, &theme.tokens)
//!     .resolve(&["button"], &ctx)
//!     .unwrap();
//!
//! // The hover override wins and its token reference is resolved to a color.
//! assert_eq!(
//!     computed.get(StyleProp::BackgroundColor),
//!     Some(&StyleValue::rgba8(239, 68, 68, 255)),
//! );
//! ```
#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

pub mod cascade;
pub mod class;
pub mod error;
pub mod selector;
pub mod theme;
pub mod token;
pub mod value;

pub use cascade::{resolve, Cascade, ComputedStyle};
pub use class::{Class, PropMap, StyleSheet};
pub use error::StyleError;
pub use selector::{Breakpoint, InteractionState, InteractionStateFlags, MatchContext};
pub use theme::Theme;
pub use token::{DesignToken, TokenStore};
pub use value::{Color, Keyword, Length, StyleProp, StyleValue};
