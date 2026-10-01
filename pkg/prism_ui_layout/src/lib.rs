//! `prism_ui_layout` — pure-Rust layout solver for Prism's Loom UI/scene
//! notation.
//!
//! Scaffold: the flexbox/grid solver lands here. Engine-agnostic and
//! deterministic so the same inputs always produce the same box geometry.
#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;
