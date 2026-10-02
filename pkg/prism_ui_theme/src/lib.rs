#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

//! Loom theming pipeline (scaffold).
//!
//! Deepened design lives in `docs/prism_ui_loom_design_zh.md` §9.12. This crate
//! will provide design-token compilation, semantic token mapping, dynamic
//! light/dark/high-contrast themes driven by a reactive signal, and RTL logical
//! -> physical property resolution, layered over [`prism_ui_style`].
