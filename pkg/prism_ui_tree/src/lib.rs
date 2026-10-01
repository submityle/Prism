//! `prism_ui_tree` — node tree and keyed reconciliation for Prism's Loom
//! UI/scene notation.
//!
//! Scaffold: the retained node tree and keyed list reconciler land here. Built
//! on [`prism_ui_reactive`] for fine-grained updates.
#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;
