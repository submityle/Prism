//! Typed ECS context gathering at the `StateTree` adapter boundary.

use prism_ecs::{entity::Entity, world::World};
use prism_state_tree_core::{Schema, SchemaError};

use crate::ValidatedContext;

/// Structured failure while gathering a schema context from an ECS world.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EcsContextError {
    /// The requested owner is stale, despawned, or not materialized.
    MissingOwner(Entity),
    /// The owner exists but lacks the schema-specific context data.
    MissingContext(Entity),
    /// The gathered context failed the core schema identity contract.
    Schema(SchemaError),
}

/// Adapter-side schema contract with an explicit, typed ECS gather operation.
///
/// The associated context remains borrowed from `World`; no snapshot or
/// hidden allocation is introduced. Implementations own the component/query
/// choices for their domain and must return `MissingContext` when required
/// data is not present.
pub trait EcsSchema: Schema {
    /// Gathers this schema's typed context for one owner entity.
    fn gather<'w>(world: &'w World, owner: Entity) -> Result<Self::Context<'w>, EcsContextError>
    where
        Self: 'w;
}

/// Gathers and validates a typed schema context without mutating the world.
pub fn gather_context<'w, S: EcsSchema>(
    world: &'w World,
    owner: Entity,
) -> Result<ValidatedContext<'w, S>, EcsContextError> {
    if !world.contains(owner) {
        return Err(EcsContextError::MissingOwner(owner));
    }
    let context = S::gather(world, owner)?;
    ValidatedContext::new(context).map_err(EcsContextError::Schema)
}
