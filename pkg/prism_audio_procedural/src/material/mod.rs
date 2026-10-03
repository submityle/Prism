//! Acoustic material coupling: material pairs to synthesis parameters.
//!
//! A contact's timbre is decided by the two materials that collide. This module
//! holds the lightweight identity types ([`pair`]) and the deterministic lookup
//! layer ([`lookup`]) that turns a [`MaterialPairId`] into a modal table and a
//! friction template with no gaps: specific authored pairs may be registered to
//! override the defaults, and every other pair falls back through per-material
//! categories to a generic category, so the whole material space always sounds.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the acoustic material coupling of design section 47.5; feeds mode
//! tables to [`crate::modal`] and friction templates to [`crate::continuous`].
//! Standalone so the crate does not hard-depend on `prism_material_pipeline`.

pub mod lookup;
pub mod pair;

pub use lookup::{
    build_modes, category_acoustics, CategoryAcoustics, FrictionTemplate, MaterialLibrary,
    MaterialPairProfile,
};
pub use pair::{MaterialCategory, MaterialId, MaterialPairId};
