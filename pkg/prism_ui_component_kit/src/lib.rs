//! `prism_ui_component_kit` — a Liquid-Glass styled component library for Loom.
//!
//! This crate layers a batteries-included set of UI controls on top of the
//! Loom stack (`prism_ui`, `prism_ui_component`, `prism_ui_style`,
//! `prism_ui_theme`, `prism_ui_a11y`). It answers a single question: *given a
//! typed set of props, what [`Element`](prism_ui::Element) tree draws this
//! control, and which theme tokens style it?*
//!
//! # Architecture (three-layer decoupling)
//!
//! 1. **Controls** ([`basics`], [`inputs`], …) emit data-only
//!    [`Element`](prism_ui::Element) trees and attach *class names only*. They
//!    never embed raw colors, sizes or shadows.
//! 2. **Presets** ([`preset`]) translate the kit's class names into
//!    [`Class`](prism_ui_style::Class) rules whose every value is a theme
//!    **token** reference (e.g. `color.tint`, `radius.capsule`). Swapping the
//!    active [`ThemeMode`](prism_ui_theme::ThemeMode) re-resolves every token,
//!    so light/dark is a single signal write — no per-control branching.
//! 3. **Theme** (`prism_ui_theme`) owns the token values. The kit ships
//!    against [`ThemeDefinition::glass`](prism_ui_theme::ThemeDefinition::glass).
//!
//! # Glass, approximated
//!
//! The "Liquid Glass" look is a translucent tint + lit-edge highlight + soft
//! drop shadow, expressed through the `GlassTint`/`GlassHighlight`/`Shadow*`
//! style props. A true backdrop blur is a GPU-backend concern; the kit only
//! records the `GlassBlur` hint so a capable backend can honor it.
//!
//! # Invariants (K1–K7)
//!
//! * **K1** style never lives in a control — only token-backed classes;
//! * **K2** reuse lower layers, never re-implement them;
//! * **K3** a variant is an enum, not a bool soup;
//! * **K4** controls are composable (children in, element out);
//! * **K5** `no_std` friendly (`alloc` only);
//! * **K6** every control ships Workbench stories;
//! * **K7** overlay-backed controls share one Popover base.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

pub mod kit;
pub mod preset;

// Control families (one directory per family; see the design doc, section 5).
pub mod basics;
pub mod containers;
pub mod display;
pub mod editor;
pub mod feedback;
pub mod inputs;
pub mod motion;
pub mod nav;
pub mod node_editor;
pub mod pickers;
pub mod utils;

// Dev-only showcase: mounts every control for the real-instance gallery.
#[cfg(feature = "gallery")]
pub mod gallery;

pub use kit::{ButtonVariant, ControlSize, Tone};
pub use preset::{stylesheet, StyleSheet};
