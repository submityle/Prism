//! `MetaSound` Pages-style tiered compilation for Patches.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML. Authoring several
//! quality variants of one graph and resolving one per quality tier, with
//! debounced switching, is a classical scalability technique; every type here
//! is an independent implementation.
//!
//! # Relationship
//! Implements the design section 42 open item "`MetaSound` Pages-style tiered
//! compilation coupled to the section 32 quality governor", layered on the
//! design section 11 Patch model. It never depends on `prism_audio_governor`,
//! keeping the authoring layer (M5) below the budget governor (M7); a host maps
//! the governor's runtime quality tier onto a [`QualityLevel`].
//!
//! # Pipeline
//!
//! 1. [`QualityLevel`] is the total-ordered key selecting a variant.
//! 2. [`PatchPage`] pairs a level floor with a full
//!    [`crate::patch::PatchDescription`].
//! 3. [`PagedPatch`] holds the ordered pages and [`PagedPatch::resolve`]s one
//!    for any requested level (always total thanks to a mandatory base page).
//! 4. [`PageSelector`] debounces the resolved selection with hysteresis and a
//!    switch-rate cap so governor dithering never triggers a recompile storm;
//!    a committed [`PageDecision::Switched`] is the signal to recompile the
//!    resolved page through the ordinary section 11 compiler.
//!
//! Construction and resolution allocate only in [`PagedPatch::new`]; the
//! runtime [`PageSelector::observe`] loop is allocation free and deterministic.

mod error;
mod page;
mod paged_patch;
mod selector;
mod tier;

pub use error::PagesError;
pub use page::PatchPage;
pub use paged_patch::PagedPatch;
pub use selector::{DeferReason, PageDecision, PageSelector};
pub use tier::QualityLevel;
