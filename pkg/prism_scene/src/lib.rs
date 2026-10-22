//! Prism Scene: the template / instance / override runtime (design §5-§15).
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
//! - [`runtime`] — transactional scene spawn, including typed structural replay
//!   when the `materialize` feature is enabled.
//! - [`streaming`] — deterministic residency admission and frame budgets.
//! - [`persistence`] / [`reload`] — ECS-free delta capture and template reload
//!   planning with persistent-id-first remapping.
//!
//! The crate is `no_std + alloc` and forbids `unsafe`. It depends only on
//! `prism_asset` (for the [`prism_asset::StableGuid`] identity shared with the
//! asset system) and `prism_ecs` (for the live [`prism_ecs::Entity`] handle),
//! and on no `bevy_*` crate.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

/// Runtime asset kind for compiled scene templates.
pub const SCENE_TEMPLATE_KIND: u128 = 0x5052_4953_4d5f_5343_454e_455f_544d_5031;
/// Current content schema for a scene template.
///
/// This is intentionally separate from the PRST wire version. Wire version 1
/// remains readable as content schema 1; the current publisher emits wire
/// version 2 with this schema field present.
pub const SCENE_TEMPLATE_SCHEMA: u32 = 2;

pub mod audio;
pub mod cost;
pub mod identity;
pub mod instance;
pub mod instantiate;
pub mod layers;
pub mod manifest;
#[cfg(feature = "materialize")]
pub mod materialize;
#[cfg(feature = "materialize")]
pub mod nested_runtime;
#[cfg(feature = "materialize")]
pub mod override_replay;
pub mod overrides;
#[cfg(feature = "partition")]
pub mod partition;
pub mod path;
pub mod persistence;
pub mod plan;
pub mod plan_codec;
#[cfg(feature = "materialize")]
pub mod property_apply;
pub mod reload;
pub mod remap;
pub mod residency;
#[cfg(feature = "materialize")]
pub mod runtime;
pub mod schema;
pub mod streaming;
pub mod template;
pub mod template_codec;

pub use audio::AudioSourceNode;
pub use cost::ScenePlanCost;
pub use identity::{LocalEntityId, NameTable, PersistentId, PersistentIdBuilder};
pub use instance::SceneInstance;
pub use instantiate::{
    despawn_instance, spawn_plan, spawn_typed_plan, InstantiateError, SceneAttachError, SceneSpawn,
    SpawnContext, SpawnContextError, TypedSpawnError, TypedSpawnPlan,
};
pub use layers::{compose_layers, OrphanKey, OverrideComposition, OverrideCompositionError};
pub use manifest::{
    decode_manifest, encode_manifest, ManifestDecodeLimits, ManifestError, SceneManifest,
};
#[cfg(feature = "materialize")]
pub use materialize::{
    materialize_plan, materialize_plan_with_limits, ComponentMaterializer, MaterializeError,
    MaterializeFailure, MaterializeLimits, MaterializerRegistry, RemappedEntityField, RowLayout,
};
#[cfg(feature = "materialize")]
pub use nested_runtime::{
    reload_invalidated_nested_trees, replace_nested_typed, spawn_nested_typed,
    NestedInstantiateError, NestedProviderError, NestedReloadBatchError, NestedReplaceError,
    NestedSceneProvider, NestedSpawnLimits, SceneTreeChild, SceneTreeSpawn,
};
#[cfg(feature = "materialize")]
pub use override_replay::{
    replay_overrides, replay_overrides_with_registry, OverrideReplayError, StructuralDecodeFailure,
    StructuralDecoder, StructuralDecoderRegistrationError, StructuralDecoderRegistry,
};
pub use overrides::{
    ArchetypeSignature, ComponentTypeId, OverrideError, OverrideSet, OverrideValue,
    PropertyOverride, StructuralOp,
};
#[cfg(feature = "partition")]
pub use partition::{FileCellStore, PersistentEntityRef, PersistentPartition};
pub use path::{FieldPath, FieldSegment};
pub use persistence::{
    EntityBinding, PersistedPropertyOverride, PersistenceCapture, PersistenceDiagnostic,
    PersistenceError, PersistentEntityMap, PersistentOverrideSet, SceneInstancePersistence,
};
pub use plan::{ArchetypeGroup, Column, FieldSlot, InstantiationPlan, PlanError};
pub use plan_codec::{decode_plan, encode_plan, PlanCodecError, PlanDecodeLimits};
#[cfg(feature = "materialize")]
pub use property_apply::{apply_property_override, apply_property_overrides, PropertyApplyError};
pub use reload::{
    plan_reload, HierarchyChange, ReloadDiagnostic, ReloadError, ReloadOrphanReason,
    SceneReloadPlan, TemplateDelta,
};
pub use remap::RemapTable;
pub use residency::{ResidencyPlan, ResidencyTier};
#[cfg(feature = "materialize")]
pub use runtime::{spawn_scene, spawn_scene_with_registry, SceneSpawnError};
pub use schema::{SchemaMigration, SchemaMigrationError, SchemaMigrationRegistry};
pub use streaming::{ResidencyAction, ResidencyBudget, ResidencyRequest, ResidencyScheduler};
pub use template::{NestedSlot, SceneTemplate, TemplateError};
pub use template_codec::{
    decode_template, encode_template, TemplateCodecError, TemplateDecodeLimits,
};

/// The common types most consumers import.
pub mod prelude {
    pub use crate::audio::AudioSourceNode;
    pub use crate::cost::ScenePlanCost;
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
    #[cfg(feature = "materialize")]
    pub use crate::override_replay::{
        replay_overrides, replay_overrides_with_registry, OverrideReplayError,
        StructuralDecodeFailure, StructuralDecoder, StructuralDecoderRegistrationError,
        StructuralDecoderRegistry,
    };
    pub use crate::overrides::{
        ArchetypeSignature, ComponentTypeId, OverrideError, OverrideSet, OverrideValue,
        PropertyOverride, StructuralOp,
    };
    #[cfg(feature = "partition")]
    pub use crate::partition::{FileCellStore, PersistentEntityRef, PersistentPartition};
    pub use crate::path::{FieldPath, FieldSegment};
    pub use crate::persistence::{
        EntityBinding, PersistedPropertyOverride, PersistenceCapture, PersistenceDiagnostic,
        PersistenceError, PersistentEntityMap, PersistentOverrideSet, SceneInstancePersistence,
    };
    pub use crate::plan::{ArchetypeGroup, Column, FieldSlot, InstantiationPlan, PlanError};
    pub use crate::plan_codec::{decode_plan, encode_plan, PlanCodecError, PlanDecodeLimits};
    pub use crate::reload::{
        plan_reload, HierarchyChange, ReloadDiagnostic, ReloadError, ReloadOrphanReason,
        SceneReloadPlan, TemplateDelta,
    };
    pub use crate::remap::RemapTable;
    pub use crate::residency::{ResidencyPlan, ResidencyTier};
    #[cfg(feature = "materialize")]
    pub use crate::runtime::{spawn_scene, spawn_scene_with_registry, SceneSpawnError};
    pub use crate::streaming::{
        ResidencyAction, ResidencyBudget, ResidencyRequest, ResidencyScheduler,
    };
    pub use crate::template::{NestedSlot, SceneTemplate, TemplateError};
    pub use crate::template_codec::{
        decode_template, encode_template, TemplateCodecError, TemplateDecodeLimits,
    };
}

#[cfg(test)]
mod tests;
