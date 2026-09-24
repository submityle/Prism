//! A FIFO buffer of deferred [`PhysicsCommand`]s.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**.

use crate::command::kind::PhysicsCommand;
use crate::world::PhysicsWorld;

/// A first-in, first-out queue of deferred physics commands.
///
/// Game logic pushes commands during a frame; the pipeline drains and applies
/// them at the next step boundary in issue order. Because application order is
/// fixed by insertion order (not wall-clock timing), the same set of commands
/// always produces the same result — the basis for deterministic replay.
#[derive(Clone, Debug, Default)]
pub struct CommandQueue {
    pending: Vec<PhysicsCommand>,
}

impl CommandQueue {
    /// Creates an empty queue.
    #[must_use]
    pub fn new() -> CommandQueue {
        CommandQueue::default()
    }

    /// Creates an empty queue with room for `capacity` commands.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> CommandQueue {
        CommandQueue {
            pending: Vec::with_capacity(capacity),
        }
    }

    /// Enqueues `command` to be applied at the next drain.
    pub fn push(&mut self, command: PhysicsCommand) {
        self.pending.push(command);
    }

    /// Returns the number of queued commands.
    #[must_use]
    pub fn len(&self) -> usize {
        self.pending.len()
    }

    /// Returns `true` if no commands are queued.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// Returns the queued commands in issue order without draining them.
    #[must_use]
    pub fn pending(&self) -> &[PhysicsCommand] {
        &self.pending
    }

    /// Drops all queued commands without applying them.
    pub fn clear(&mut self) {
        self.pending.clear();
    }

    /// Applies every queued command to `world` in issue order and empties the
    /// queue, returning how many commands actually took effect.
    ///
    /// Commands targeting a stale handle are still removed but not counted, so a
    /// return value below [`CommandQueue::len`]-at-entry indicates dropped
    /// commands. The queue's backing allocation is retained for reuse.
    pub fn drain_apply(&mut self, world: &mut PhysicsWorld) -> usize {
        let mut applied = 0;
        for command in self.pending.drain(..) {
            if command.apply(world) {
                applied += 1;
            }
        }
        applied
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::body::BodyDesc;
    use glam::Vec3;

    #[test]
    fn drain_applies_in_issue_order_and_empties() {
        let mut world = PhysicsWorld::default();
        let h = world.spawn(BodyDesc::dynamic_at(Vec3::ZERO));
        let mut queue = CommandQueue::new();
        // Two impulses in the same drain accumulate: 1 + 2 = 3.
        queue.push(PhysicsCommand::ApplyLinearImpulse {
            body: h,
            impulse: Vec3::new(1.0, 0.0, 0.0),
        });
        queue.push(PhysicsCommand::ApplyLinearImpulse {
            body: h,
            impulse: Vec3::new(2.0, 0.0, 0.0),
        });
        assert_eq!(queue.len(), 2);
        let applied = queue.drain_apply(&mut world);
        assert_eq!(applied, 2);
        assert!(queue.is_empty());
        assert_eq!(
            world.bodies.linear_velocity(h),
            Some(Vec3::new(3.0, 0.0, 0.0))
        );
    }

    #[test]
    fn later_set_overrides_earlier_impulse() {
        let mut world = PhysicsWorld::default();
        let h = world.spawn(BodyDesc::dynamic_at(Vec3::ZERO));
        let mut queue = CommandQueue::new();
        queue.push(PhysicsCommand::ApplyLinearImpulse {
            body: h,
            impulse: Vec3::new(5.0, 0.0, 0.0),
        });
        // A later SetLinearVelocity in the same drain wins (issue order).
        queue.push(PhysicsCommand::SetLinearVelocity {
            body: h,
            velocity: Vec3::ZERO,
        });
        queue.drain_apply(&mut world);
        assert_eq!(world.bodies.linear_velocity(h), Some(Vec3::ZERO));
    }

    #[test]
    fn dropped_commands_are_not_counted() {
        let mut world = PhysicsWorld::default();
        let live = world.spawn(BodyDesc::dynamic_at(Vec3::ZERO));
        let dead = world.spawn(BodyDesc::dynamic_at(Vec3::ONE));
        world.bodies.remove(dead);
        let mut queue = CommandQueue::new();
        queue.push(PhysicsCommand::SetPosition {
            body: live,
            position: Vec3::Y,
        });
        queue.push(PhysicsCommand::SetPosition {
            body: dead,
            position: Vec3::Y,
        });
        let applied = queue.drain_apply(&mut world);
        assert_eq!(applied, 1);
    }
}
