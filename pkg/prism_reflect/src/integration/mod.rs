//! M6 integration layer: the reflection-side building blocks that downstream
//! engine crates consume to realise design §22's "集成" milestone — ECS
//! component snapshots and scenes, the editor inspector model, the script
//! bridge, and field-level network replication.
//!
//! `prism_reflect` stays entity-component-system (ECS) agnostic: these modules
//! operate purely on `&dyn Reflect` values plus a [`TypeRegistry`](crate::TypeRegistry),
//! so the actual ECS/editor/network crates can build on one shared reflection
//! contract instead of re-deriving traversal, serialization, and diff logic.

pub mod inspector;
pub mod replication;
pub mod scene;
pub mod script;

pub use inspector::{InspectorHints, InspectorKind, InspectorNode, inspect};
pub use replication::{
    NO_REPLICATE_KEY, REPLICATE_KEY, ReplicationError, ReplicationPlan, ReplicationPolicy,
    apply_replicated, replicated_diff,
};
pub use scene::{DynamicScene, SceneEntry, SceneError};
pub use script::{ScriptBridge, ScriptError};
