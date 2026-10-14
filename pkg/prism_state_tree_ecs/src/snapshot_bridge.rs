//! ECS-facing binary snapshots for [`StateTreeInstance`].
//!
//! The bridge deliberately keeps the portable format in the core crate.  It
//! only supplies the ECS/component boundary: component lookup, byte
//! encoding/decoding, and forwarding a decoded core [`Snapshot`] to the
//! instance's atomic restore operation.  No ECS mutation or lifecycle event is
//! produced by a successful restore.

use alloc::vec::Vec;
use core::fmt;

use prism_ecs::{
    entity::{Entity, EntityLocation},
    world::World,
};
use prism_state_tree_core::{RestoreError, Snapshot, SnapshotCodecError};

use crate::StateTreeInstance;

/// A precise reason why an entity cannot provide a `StateTreeInstance`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SnapshotComponentError {
    /// The entity handle is stale or refers to an entity that was despawned.
    StaleEntity,
    /// The entity is reserved but has not yet been materialized into an
    /// archetype.
    NotMaterialized,
    /// The entity is live and materialized but does not carry the component.
    MissingComponent,
}

impl fmt::Display for SnapshotComponentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StaleEntity => f.write_str("state-tree snapshot entity is stale"),
            Self::NotMaterialized => {
                f.write_str("state-tree snapshot entity is not materialized")
            }
            Self::MissingComponent => {
                f.write_str("state-tree snapshot component is missing")
            }
        }
    }
}

/// A failure at the ECS snapshot boundary.
///
/// Decoding is completed before a mutable component borrow is taken.  Core
/// restore validates the complete decoded snapshot before changing the
/// instance, so [`Self::Decode`] and [`Self::Restore`] leave the target
/// component unchanged.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SnapshotBridgeError {
    /// The requested entity did not contain a usable state-tree component.
    Component {
        /// Entity supplied to the bridge.
        entity: Entity,
        /// Exact component lookup failure.
        reason: SnapshotComponentError,
    },
    /// The core snapshot could not be encoded.
    Encode(SnapshotCodecError),
    /// The supplied bytes are not a valid core snapshot.
    Decode(SnapshotCodecError),
    /// The decoded snapshot was rejected by the target instance's plan or
    /// runtime capacity checks.
    Restore(RestoreError),
}

impl fmt::Display for SnapshotBridgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Component { entity, reason } => {
                write!(f, "entity {entity:?}: {reason}")
            }
            Self::Encode(error) => write!(f, "state-tree snapshot encode failed: {error:?}"),
            Self::Decode(error) => write!(f, "state-tree snapshot decode failed: {error:?}"),
            Self::Restore(error) => write!(f, "state-tree snapshot restore failed: {error:?}"),
        }
    }
}

/// Captures an instance component into the core's deterministic binary
/// snapshot format.
pub fn capture_snapshot(instance: &StateTreeInstance) -> Result<Vec<u8>, SnapshotBridgeError> {
    instance
        .snapshot()
        .encode()
        .map_err(SnapshotBridgeError::Encode)
}

/// Decodes and atomically restores a binary snapshot into an instance.
pub fn restore_snapshot(
    instance: &mut StateTreeInstance,
    bytes: &[u8],
) -> Result<(), SnapshotBridgeError> {
    let snapshot = Snapshot::decode(bytes).map_err(SnapshotBridgeError::Decode)?;
    instance
        .restore_snapshot(&snapshot)
        .map_err(SnapshotBridgeError::Restore)
}

/// Captures the [`StateTreeInstance`] component attached to `entity`.
pub fn capture_component_snapshot(
    world: &World,
    entity: Entity,
) -> Result<Vec<u8>, SnapshotBridgeError> {
    let instance = component(world, entity)?;
    capture_snapshot(instance)
}

/// Decodes and atomically restores a snapshot into the component attached to
/// `entity`.
///
/// Entity/component lookup and decoding happen before the mutable component
/// borrow.  Once the borrow exists, the core restore operation validates the
/// snapshot before replacing any semantic state.
pub fn restore_component_snapshot(
    world: &mut World,
    entity: Entity,
    bytes: &[u8],
) -> Result<(), SnapshotBridgeError> {
    component(world, entity)?;
    let snapshot = Snapshot::decode(bytes).map_err(SnapshotBridgeError::Decode)?;
    let mut instance = world
        .get_mut::<StateTreeInstance>(entity)
        .expect("component lookup was validated before mutable borrow");
    instance
        .restore_snapshot(&snapshot)
        .map_err(SnapshotBridgeError::Restore)
}

/// Convenience method for obtaining the portable core snapshot value before
/// encoding it for transport or storage.
impl StateTreeInstance {
    /// Captures the semantic state as a pure-data core [`Snapshot`].
    pub fn snapshot(&self) -> Snapshot {
        self.runtime.snapshot()
    }

    /// Restores a decoded core [`Snapshot`] atomically.
    pub fn restore_snapshot(&mut self, snapshot: &Snapshot) -> Result<(), RestoreError> {
        self.runtime.restore_snapshot(snapshot)
    }

    /// Captures this component as deterministic portable bytes.
    pub fn snapshot_bytes(&self) -> Result<Vec<u8>, SnapshotBridgeError> {
        capture_snapshot(self)
    }

    /// Decodes and atomically restores deterministic portable bytes.
    pub fn restore_snapshot_bytes(&mut self, bytes: &[u8]) -> Result<(), SnapshotBridgeError> {
        restore_snapshot(self, bytes)
    }
}

fn component<'w>(
    world: &'w World,
    entity: Entity,
) -> Result<&'w StateTreeInstance, SnapshotBridgeError> {
    world
        .get::<StateTreeInstance>(entity)
        .ok_or_else(|| SnapshotBridgeError::Component {
            entity,
            reason: component_reason(world, entity),
        })
}

fn component_reason(world: &World, entity: Entity) -> SnapshotComponentError {
    if !world.contains(entity) {
        SnapshotComponentError::StaleEntity
    } else if world
        .entities()
        .location(entity)
        .is_some_and(|location| location == EntityLocation::EMPTY)
    {
        SnapshotComponentError::NotMaterialized
    } else {
        SnapshotComponentError::MissingComponent
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use prism_ecs::world::World;
    use prism_state_tree_core::{
        Event, EventId, MicrostepLimit, RestoreError, StateId, StateNode, Transition,
        TransitionKind, TransitionTarget, TreeDef, Trigger,
    };

    use super::*;
    use crate::StateTreeAsset;

    fn asset() -> StateTreeAsset {
        let transition = Transition::new(
            StateId(1),
            TransitionTarget::State(StateId(2)),
            Trigger::Event(EventId(7)),
            None,
            0,
            TransitionKind::External,
        );
        let plan = TreeDef {
            root: StateId(0),
            nodes: vec![
                StateNode::compound(
                    StateId(0),
                    None,
                    vec![StateId(1), StateId(2)],
                    StateId(1),
                    None,
                ),
                StateNode::atomic(StateId(1), Some(StateId(0))).with_transition(transition),
                StateNode::atomic(StateId(2), Some(StateId(0))),
            ],
        }
        .compile()
        .expect("test plan is valid");
        StateTreeAsset::new(plan)
    }

    fn running_instance() -> StateTreeInstance {
        let mut instance = StateTreeInstance::new(asset(), 4, MicrostepLimit(8)).unwrap();
        instance.step().unwrap();
        instance.request_event(Event::new(EventId(7))).unwrap();
        instance
    }

    #[test]
    fn component_bytes_round_trip_through_core_restore() {
        let source = running_instance();
        let bytes = capture_snapshot(&source).unwrap();
        let mut target = StateTreeInstance::new(asset(), 4, MicrostepLimit(8)).unwrap();

        restore_snapshot(&mut target, &bytes).unwrap();

        assert_eq!(target.active_states(), source.active_states());
        assert_eq!(target.queued_events(), source.queued_events());
        assert_eq!(target.snapshot_bytes().unwrap(), bytes);
    }

    #[test]
    fn world_component_bridge_captures_and_restores() {
        let mut world = World::new();
        let entity = world.spawn(running_instance());
        let bytes = capture_component_snapshot(&world, entity).unwrap();

        world
            .get_mut::<StateTreeInstance>(entity)
            .unwrap()
            .step()
            .unwrap();
        restore_component_snapshot(&mut world, entity, &bytes).unwrap();

        let instance = world.get::<StateTreeInstance>(entity).unwrap();
        assert_eq!(instance.active_states(), &[StateId(0), StateId(1)]);
        assert_eq!(instance.queued_events(), 1);
    }

    #[test]
    fn malformed_bytes_do_not_mutate_component() {
        let mut instance = running_instance();
        let before = instance.snapshot_bytes().unwrap();
        let error = instance.restore_snapshot_bytes(b"not-a-snapshot");

        assert!(matches!(error, Err(SnapshotBridgeError::Decode(_))));
        assert_eq!(instance.snapshot_bytes().unwrap(), before);
    }

    #[test]
    fn rejected_snapshot_is_atomic_at_component_boundary() {
        let source = running_instance();
        let mut source = source;
        source.request_event(Event::new(EventId(8))).unwrap();
        let bytes = source.snapshot_bytes().unwrap();
        let mut target = StateTreeInstance::new(asset(), 1, MicrostepLimit(8)).unwrap();
        target.step().unwrap();
        let before = target.snapshot_bytes().unwrap();

        let error = target.restore_snapshot_bytes(&bytes);

        assert!(matches!(
            error,
            Err(SnapshotBridgeError::Restore(
                RestoreError::QueueCapacityExceeded { .. }
            ))
        ));
        assert_eq!(target.snapshot_bytes().unwrap(), before);
    }

    #[test]
    fn world_lookup_errors_are_structured() {
        let mut world = World::new();
        let missing_component = world.spawn(());
        let stale = world.entities().reserve_entity();
        world.flush_reserved();
        assert!(world.despawn(stale));

        assert_eq!(
            capture_component_snapshot(&world, missing_component),
            Err(SnapshotBridgeError::Component {
                entity: missing_component,
                reason: SnapshotComponentError::MissingComponent,
            })
        );
        assert_eq!(
            capture_component_snapshot(&world, stale),
            Err(SnapshotBridgeError::Component {
                entity: stale,
                reason: SnapshotComponentError::StaleEntity,
            })
        );
    }
}
