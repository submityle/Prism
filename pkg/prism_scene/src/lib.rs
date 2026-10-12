//! Prism Scene: the template / instance / override runtime (design §5-§7, §14).
//!
//! A **Scene** is a first-class entity-subtree template. At runtime it is
//! represented in three layers (design §5):
//!
//! ```text
//! Template (immutable, shareable, disk/resident form)
//!    │ spawn_scene()
//!    ▼
//! Instance (live ECS subtree; its root carries SceneInstance)
//!    │
//!    ▼
//! Override (sparse delta: property overrides + structural ops)
//! ```
//!
//! This crate owns the correctness-critical heart of that model, all of it
//! dependency-free and deterministic:
//!
//! - [`identity`] — the four id kinds (design §6): [`LocalEntityId`] within a
//!   template, the template [`NameTable`], and the deterministic cross-session
//!   [`PersistentId`] derived from a template-GUID chain plus a local path.
//! - [`instance`] — [`SceneInstance`], the root anchor component of a live
//!   instance, and the [`ResidencyTier`] vocabulary (design §5.2, §11).
//! - [`path`] — [`FieldPath`], the reflect-style field path an override targets
//!   (`"transform.translation.x"`).
//! - [`overrides`] — the crown (design §7): sparse [`PropertyOverride`] and
//!   [`StructuralOp`] deltas in an [`OverrideSet`], with unique-key dedup and
//!   orphan detection.
//! - [`remap`] — the per-instance [`RemapTable`] that keeps a template's
//!   internal references pointing inside *its own* instance (design §6.1).
//! - [`template`] — the immutable [`SceneTemplate`] descriptor and its nested
//!   instance [`NestedSlot`]s (design §5.1, §9).
//!
//! The crate is `no_std + alloc` and forbids `unsafe`. It depends only on
//! `prism_asset` (for the [`prism_asset::StableGuid`] identity shared with the
//! asset system) and `prism_ecs` (for the live [`prism_ecs::Entity`] handle),
//! and on no `bevy_*` crate.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

pub mod identity;
pub mod instance;
pub mod overrides;
pub mod path;
pub mod remap;
pub mod template;

pub use identity::{LocalEntityId, NameTable, PersistentId, PersistentIdBuilder};
pub use instance::{ResidencyTier, SceneInstance};
pub use overrides::{ArchetypeSignature, ComponentTypeId, OverrideSet, OverrideValue, PropertyOverride, StructuralOp};
pub use path::{FieldPath, FieldSegment};
pub use remap::RemapTable;
pub use template::{NestedSlot, SceneTemplate};

/// The common types most consumers import.
pub mod prelude {
    pub use crate::identity::{LocalEntityId, NameTable, PersistentId, PersistentIdBuilder};
    pub use crate::instance::{ResidencyTier, SceneInstance};
    pub use crate::overrides::{
        ArchetypeSignature, ComponentTypeId, OverrideSet, OverrideValue, PropertyOverride,
        StructuralOp,
    };
    pub use crate::path::{FieldPath, FieldSegment};
    pub use crate::remap::RemapTable;
    pub use crate::template::{NestedSlot, SceneTemplate};
}

#[cfg(test)]
mod tests;
