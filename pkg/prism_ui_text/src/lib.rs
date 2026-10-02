#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

//! Loom text stack (scaffold).
//!
//! Deepened design lives in `docs/prism_ui_loom_design_zh.md` §9.6. This crate
//! will provide grapheme/word segmentation, UAX#14 line breaking, rich-text
//! spans, paragraph layout (alignment/ellipsis), a cursor/selection model and a
//! pluggable shaper (deterministic metric default, optional `swash` shaping).

#[cfg(feature = "std")]
extern crate std;
