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
//! - [`plan`] — [`InstantiationPlan`], the flattened bake-to-runtime contract
//!   (archetype groups + columnar payloads + entity-valued field slots) and its
//!   structural validation (design §8.1).
//! - [`overrides`] — the crown (design §7): sparse [`PropertyOverride`] and
//!   [`StructuralOp`] deltas in an [`OverrideSet`], with unique-key dedup and
//!   orphan detection.
//! - [`layers`] — bottom-up composition of an override-layer stack (base →
//!   variant → … → instance), the Prefab Variant replay (design §7.3).
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
pub mod instantiate;
pub mod layers;
#[cfg(feature = "materialize")]
pub mod materialize;
pub mod overrides;
pub mod path;
pub mod plan;
pub mod plan_codec;
#[cfg(feature = "materialize")]
pub mod property_apply;
pub mod remap;
pub mod residency;
pub mod template;

pub use identity::{LocalEntityId, NameTable, PersistentId, PersistentIdBuilder};
pub use instance::SceneInstance;
pub use instantiate::{
    despawn_instance, spawn_plan, spawn_typed_plan, InstantiateError, SceneAttachError, SceneSpawn,
    SpawnContext, SpawnContextError, TypedSpawnError, TypedSpawnPlan,
};
pub use layers::{compose_layers, OrphanKey, OverrideComposition, OverrideCompositionError};
#[cfg(feature = "materialize")]
pub use materialize::{
    materialize_plan, materialize_plan_with_limits, ComponentMaterializer, MaterializeError,
    MaterializeFailure, MaterializeLimits, MaterializerRegistry, RemappedEntityField, RowLayout,
};
pub use overrides::{
    ArchetypeSignature, ComponentTypeId, OverrideError, OverrideSet, OverrideValue,
    PropertyOverride, StructuralOp,
};
pub use path::{FieldPath, FieldSegment};
pub use plan::{ArchetypeGroup, Column, FieldSlot, InstantiationPlan, PlanError};
pub use plan_codec::{decode_plan, encode_plan, PlanCodecError, PlanDecodeLimits};
#[cfg(feature = "materialize")]
pub use property_apply::{apply_property_override, PropertyApplyError};
pub use remap::RemapTable;
pub use residency::{ResidencyPlan, ResidencyTier};
pub use template::{NestedSlot, SceneTemplate};

/// The common types most consumers import.
pub mod prelude {
    pub use crate::identity::{LocalEntityId, NameTable, PersistentId, PersistentIdBuilder};
    pub use crate::instance::SceneInstance;
    pub use crate::instantiate::{
        despawn_instance, spawn_plan, spawn_typed_plan, InstantiateError, SceneAttachError,
        SceneSpawn, SpawnContext, SpawnContextError, TypedSpawnError, TypedSpawnPlan,
    };
    pub use crate::layers::{
        compose_layers, OrphanKey, OverrideComposition, OverrideCompositionError,
    };
    #[cfg(feature = "materialize")]
    pub use crate::materialize::{
        materialize_plan, materialize_plan_with_limits, ComponentMaterializer, MaterializeError,
        MaterializeFailure, MaterializeLimits, MaterializerRegistry, RemappedEntityField,
        RowLayout,
    };
    pub use crate::overrides::{
        ArchetypeSignature, ComponentTypeId, OverrideError, OverrideSet, OverrideValue,
        PropertyOverride, StructuralOp,
    };
    pub use crate::path::{FieldPath, FieldSegment};
    pub use crate::plan::{ArchetypeGroup, Column, FieldSlot, InstantiationPlan, PlanError};
    pub use crate::plan_codec::{decode_plan, encode_plan, PlanCodecError, PlanDecodeLimits};
    pub use crate::remap::RemapTable;
    pub use crate::residency::{ResidencyPlan, ResidencyTier};
    pub use crate::template::{NestedSlot, SceneTemplate};
}

#[cfg(test)]
mod tests;
