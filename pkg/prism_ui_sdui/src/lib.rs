#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

//! Loom server-driven UI (scaffold).
//!
//! Deepened design lives in `docs/prism_ui_loom_design_zh.md` §9.10. This crate
//! will provide a versioned node schema, a capability-whitelist sandbox, version
//! negotiation with graceful fallback, and safe decoding of untrusted remote
//! descriptions into [`prism_ui::Element`] trees.
