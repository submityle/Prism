//! Schedule integration for deterministic `StateTree` execution.
//!
//! This module owns the host-facing system boundary. It runs the existing
//! query-gathered [`crate::tick_all`] exactly once per schedule invocation and
//! keeps the last result in a world resource for diagnostics or later event
//! adapters. Lifecycle observers are deliberately not synthesized here: the
//! current ECS component still uses the core runtime's no-op lifecycle.

use alloc::vec::Vec;

use prism_ecs::{
    entity::Entity, event::Events, query::With, resource::Resource, schedule::Schedule,
    world::World,
};

use crate::{
    tick_all, BatchTickError, BatchTickReport, StateTreeInstance, StateTreeLifecycleNotification,
};

/// The result produced by one installed `StateTree` schedule system.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StateTreeTickOutcome {
    /// The ECS gather and all gathered runtime ticks completed.
    Completed(BatchTickReport),
    /// The adapter could not reserve its gather buffer.
    GatherFailed(BatchTickError),
}

/// World resource containing the most recent scheduled `StateTree` tick.
///
/// The resource is inserted lazily by [`install_tick_system`], so a host can
/// add the system before it constructs its world resources. A schedule run
/// replaces the previous outcome atomically after [`crate::tick_all`] returns.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct StateTreeTickReport {
    last: Option<StateTreeTickOutcome>,
    runs: u64,
}

impl Resource for StateTreeTickReport {}

impl StateTreeTickReport {
    /// Returns the most recent scheduled outcome, if the system has run.
    #[must_use]
    pub fn last(&self) -> Option<&StateTreeTickOutcome> {
        self.last.as_ref()
    }

    /// Returns how many times the installed system has run.
    #[must_use]
    pub const fn run_count(&self) -> u64 {
        self.runs
    }

    fn record(&mut self, outcome: StateTreeTickOutcome) {
        self.last = Some(outcome);
        self.runs = self.runs.saturating_add(1);
    }
}

/// Installs the deterministic `StateTree` tick system into a host schedule.
///
/// Register this once in the schedule that owns the simulation tick. The
/// chained systems first rotate the lifecycle event buffer, then run the
/// exclusive `StateTree` tick. Calling this function more than once
/// intentionally installs more than one system; hosts should register a
/// single owner for each simulation schedule.
pub fn install_tick_system(schedule: &mut Schedule) -> &mut Schedule {
    schedule.add_systems(run_scheduled_tick);
    schedule
}

/// Runs lifecycle rotation followed by one deterministic `StateTree` tick.
///
/// This is public for the optional `prism_app` plugin and remains an exclusive
/// whole-world system at the ECS boundary.
pub fn run_scheduled_tick(world: &mut World) {
    rotate_lifecycle_events(world);
    run_state_tree_tick(world);
}

fn rotate_lifecycle_events(world: &mut World) {
    if let Some(events) = world.get_resource_mut::<Events<StateTreeLifecycleNotification>>() {
        events.update();
    }
}

fn run_state_tree_tick(world: &mut World) {
    let outcome = match tick_all(world) {
        Ok(report) => StateTreeTickOutcome::Completed(report),
        Err(error) => StateTreeTickOutcome::GatherFailed(error),
    };

    if let Some(report) = world.get_resource_mut::<StateTreeTickReport>() {
        report.record(outcome);
    } else {
        let mut report = StateTreeTickReport::default();
        report.record(outcome);
        world.insert_resource(report);
    }

    publish_lifecycle_notifications(world);
}

fn publish_lifecycle_notifications(world: &mut World) {
    let query = world.query_filtered::<Entity, With<StateTreeInstance>>();
    let mut entities: Vec<Entity> = query.iter(world).collect();
    entities.sort_unstable();

    let mut notifications = Vec::new();
    for entity in entities {
        if let Some(mut instance) = world.get_mut::<StateTreeInstance>(entity) {
            notifications.extend(
                instance
                    .drain_lifecycle_events()
                    .into_iter()
                    .map(|event| StateTreeLifecycleNotification { entity, event }),
            );
        }
    }
    if notifications.is_empty() {
        return;
    }

    if let Some(events) = world.get_resource_mut::<Events<StateTreeLifecycleNotification>>() {
        events.send_batch(notifications);
    } else {
        let mut events = Events::new();
        events.send_batch(notifications);
        world.insert_resource(events);
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use prism_ecs::{
        command::CommandQueue,
        event::{EventCursor, Events},
        schedule::Schedule,
        world::World,
    };
    use prism_state_tree_core::{
        Event, EventId, MicrostepLimit, NodeKind, StateId, StateNode, Transition, TransitionKind,
        TransitionTarget, TreeDef, Trigger,
    };

    use super::{install_tick_system, StateTreeTickOutcome, StateTreeTickReport};
    use crate::{
        queue_spawn, StateTreeAsset, StateTreeInstance, StateTreeLifecycleEvent,
        StateTreeLifecycleNotification,
    };

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

    #[test]
    fn installed_system_ticks_once_per_schedule_run_and_keeps_report() {
        let mut world = World::new();
        let mut queue = CommandQueue::new();
        let entity = {
            let mut commands = queue.commands(world.entities());
            queue_spawn(&mut commands, asset(), 8, MicrostepLimit(16)).expect("spawn setup")
        };
        queue.apply(&mut world);

        let mut schedule = Schedule::new();
        install_tick_system(&mut schedule);
        schedule.run(&mut world);

        let report = world.resource::<StateTreeTickReport>();
        assert_eq!(report.run_count(), 1);
        let Some(StateTreeTickOutcome::Completed(batch)) = report.last() else {
            panic!("scheduled tick should complete");
        };
        assert_eq!(batch.completed_count(), 1);
        assert_eq!(batch.failed_count(), 0);
        assert_eq!(batch.entries()[0].entity, entity);
        assert_eq!(
            world
                .get::<StateTreeInstance>(entity)
                .unwrap()
                .active_states(),
            &[StateId(0), StateId(1)]
        );

        world
            .get_mut::<StateTreeInstance>(entity)
            .unwrap()
            .request_event(Event::new(EventId(7)))
            .expect("event fits bounded queue");
        schedule.run(&mut world);
        assert_eq!(world.resource::<StateTreeTickReport>().run_count(), 2);
        assert_eq!(
            world
                .get::<StateTreeInstance>(entity)
                .unwrap()
                .active_states(),
            &[StateId(0), StateId(2)]
        );
    }

    #[test]
    fn schedule_system_uses_the_same_deterministic_gather_as_tick_all() {
        let mut world = World::new();
        let mut queue = CommandQueue::new();
        let first = {
            let mut commands = queue.commands(world.entities());
            queue_spawn(&mut commands, asset(), 8, MicrostepLimit(16)).expect("spawn first")
        };
        let second = {
            let mut commands = queue.commands(world.entities());
            queue_spawn(&mut commands, asset(), 8, MicrostepLimit(16)).expect("spawn second")
        };
        queue.apply(&mut world);

        let mut schedule = Schedule::new();
        install_tick_system(&mut schedule);
        schedule.run(&mut world);

        let report = world.resource::<StateTreeTickReport>();
        let Some(StateTreeTickOutcome::Completed(batch)) = report.last() else {
            panic!("scheduled tick should complete");
        };
        assert_eq!(batch.entries().len(), 2);
        assert_eq!(batch.entries()[0].entity, first.min(second));
        assert_eq!(batch.entries()[1].entity, first.max(second));
        assert!(matches!(
            world
                .get::<StateTreeInstance>(first)
                .unwrap()
                .asset()
                .plan()
                .node(StateId(0))
                .unwrap()
                .kind,
            NodeKind::Compound { .. }
        ));
    }

    #[test]
    fn schedule_publishes_entity_addressed_lifecycle_events_in_order() {
        let mut world = World::new();
        let mut queue = CommandQueue::new();
        let entity = {
            let mut commands = queue.commands(world.entities());
            queue_spawn(&mut commands, asset(), 8, MicrostepLimit(16)).expect("spawn setup")
        };
        queue.apply(&mut world);

        let mut schedule = Schedule::new();
        install_tick_system(&mut schedule);
        schedule.run(&mut world);

        let events = world.resource::<Events<StateTreeLifecycleNotification>>();
        let mut cursor = EventCursor::default();
        let observed: Vec<_> = cursor.read(events).cloned().collect();
        assert_eq!(
            observed,
            vec![
                StateTreeLifecycleNotification {
                    entity,
                    event: StateTreeLifecycleEvent::Entered(StateId(0)),
                },
                StateTreeLifecycleNotification {
                    entity,
                    event: StateTreeLifecycleEvent::Entered(StateId(1)),
                },
            ]
        );
        assert!(world
            .get::<StateTreeInstance>(entity)
            .unwrap()
            .lifecycle_events()
            .is_empty());

        schedule.run(&mut world);
        assert_eq!(
            world
                .resource::<Events<StateTreeLifecycleNotification>>()
                .len(),
            2
        );
        schedule.run(&mut world);
        assert!(world
            .resource::<Events<StateTreeLifecycleNotification>>()
            .is_empty());
    }
}
