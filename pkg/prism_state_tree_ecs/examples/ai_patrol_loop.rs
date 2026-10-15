//! Small domain example: Patrol -> Alert -> Chase -> Combat -> Disengage.
//!
//! The example deliberately uses only `StateTree` events. Perception, blackboard
//! writes, and animation presentation remain host-owned inputs/tasks.

use prism_state_tree_core::{
    Event, EventId, MicrostepLimit, StateId, StateNode, Transition, TransitionKind,
    TransitionTarget, TreeDef, Trigger,
};
use prism_state_tree_ecs::{StateTreeAsset, StateTreeInstance};

fn main() {
    let root = StateNode::compound(
        StateId(0),
        None,
        vec![StateId(1), StateId(2), StateId(3), StateId(4), StateId(5)],
        StateId(1),
        None,
    );
    let states = [
        (StateId(1), 2, StateId(2)),
        (StateId(2), 3, StateId(3)),
        (StateId(3), 4, StateId(4)),
        (StateId(4), 5, StateId(5)),
    ]
    .into_iter()
    .map(|(source, event, target)| {
        StateNode::atomic(source, Some(StateId(0))).with_transition(Transition::new(
            source,
            TransitionTarget::State(target),
            Trigger::Event(EventId(event)),
            None,
            0,
            TransitionKind::External,
        ))
    })
    .chain([StateNode::atomic(StateId(5), Some(StateId(0)))])
    .collect::<Vec<_>>();
    let plan = TreeDef {
        root: StateId(0),
        nodes: core::iter::once(root).chain(states).collect(),
    }
    .compile()
    .expect("AI loop plan");
    let mut instance = StateTreeInstance::new(StateTreeAsset::new(plan), 8, MicrostepLimit(8))
        .expect("AI instance");

    instance.step().expect("initial patrol");
    for event in [2, 3, 4, 5] {
        instance
            .request_event(Event::new(EventId(event)))
            .expect("queue AI event");
        instance.step().expect("advance AI state");
    }
    assert_eq!(instance.active_states(), &[StateId(0), StateId(5)]);
}
