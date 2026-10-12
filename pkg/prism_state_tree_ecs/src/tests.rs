use alloc::vec;

use prism_ecs::{command::CommandQueue, world::World};
use prism_state_tree_core::{
    Event, EventId, Guard, GuardContext, GuardId, MicrostepLimit, NodeKind, StateId, StateNode,
    Transition, TransitionKind, TransitionTarget, TreeDef, Trigger,
};

use crate::{queue_spawn, tick_entity, StateTreeAsset, StateTreeInstance};

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
fn deferred_spawn_then_event_request_switches_state_on_tick() {
    let mut world = World::new();
    let mut queue = CommandQueue::new();
    let entity = {
        let mut commands = queue.commands(world.entities());
        queue_spawn(&mut commands, asset(), 8, MicrostepLimit(16)).expect("spawn setup")
    };
    assert!(world.get::<StateTreeInstance>(entity).is_none());
    queue.apply(&mut world);

    let first = tick_entity(&mut world, entity)
        .expect("instance exists")
        .expect("initial tick");
    assert_eq!(first.microsteps, 0);
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
    assert_eq!(
        world
            .get::<StateTreeInstance>(entity)
            .unwrap()
            .active_states(),
        &[StateId(0), StateId(1)]
    );

    let report = tick_entity(&mut world, entity).unwrap().unwrap();
    assert_eq!(report.events_processed, 1);
    assert_eq!(report.microsteps, 1);
    assert_eq!(
        world
            .get::<StateTreeInstance>(entity)
            .unwrap()
            .active_states(),
        &[StateId(0), StateId(2)]
    );
}

#[test]
fn asset_handles_share_the_same_plan_allocation() {
    let shared = asset();
    let second = shared.clone_shared();
    assert!(core::ptr::eq(shared.plan(), second.plan()));
    assert!(matches!(
        shared.plan().node(StateId(0)).unwrap().kind,
        NodeKind::Compound { .. }
    ));
}

#[test]
fn instances_share_plan_but_keep_cursor_and_event_queue_independent() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<StateTreeInstance>();

    let shared = asset();
    let mut first = StateTreeInstance::new(shared.clone_shared(), 8, MicrostepLimit(16))
        .expect("first runtime setup");
    let mut second =
        StateTreeInstance::new(shared, 8, MicrostepLimit(16)).expect("second runtime setup");
    assert!(core::ptr::eq(first.asset().plan(), second.asset().plan()));
    assert!(first.active_states().is_empty());
    assert!(second.active_states().is_empty());

    first.step().expect("first initial tick");
    first
        .request_event(Event::new(EventId(7)))
        .expect("event fits first queue");
    assert_eq!(first.active_states(), &[StateId(0), StateId(1)]);
    assert!(second.active_states().is_empty());

    first.step().expect("first transition tick");
    assert_eq!(first.active_states(), &[StateId(0), StateId(2)]);
    second.step().expect("second initial tick");
    assert_eq!(second.active_states(), &[StateId(0), StateId(1)]);
}

#[test]
fn entity_guard_is_evaluated_by_its_persistent_runtime() {
    struct MatchEvent(EventId);
    impl Guard for MatchEvent {
        fn evaluate(&self, context: GuardContext) -> bool {
            context.event.is_some_and(|event| event.id == self.0)
        }
    }

    let transition = Transition::new(
        StateId(1),
        TransitionTarget::State(StateId(2)),
        Trigger::Event(EventId(7)),
        Some(GuardId(3)),
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
    .expect("guard plan is valid");
    let mut instance = StateTreeInstance::new(StateTreeAsset::new(plan), 4, MicrostepLimit(8))
        .expect("runtime setup");
    instance
        .set_guard(GuardId(3), MatchEvent(EventId(9)));
    instance.step().expect("initial tick");
    instance
        .request_event(Event::new(EventId(7)))
        .expect("event fits bounded queue");
    let report = instance.step().expect("guarded tick");
    assert_eq!(report.events_processed, 1);
    assert_eq!(report.microsteps, 0);
    assert_eq!(instance.active_states(), &[StateId(0), StateId(1)]);

    instance.set_guard(GuardId(3), MatchEvent(EventId(7)));
    instance
        .request_event(Event::new(EventId(7)))
        .expect("event fits bounded queue");
    assert_eq!(instance.step().unwrap().microsteps, 1);
    assert_eq!(instance.active_states(), &[StateId(0), StateId(2)]);
}
