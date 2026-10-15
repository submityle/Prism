//! Verifies that Flow and Animation domains share one StateTree contract.
//!
//! Both domains use the same compiled-plan shape, ECS instance adapter,
//! snapshot bridge, debugger observations, and structured error handling.

use prism_state_tree_core::{
    Breakpoint, DebugEvent, Debugger, Event, EventId, MicrostepLimit, StateId, StateNode,
    Transition, TransitionKind, TransitionTarget, TreeDef, Trigger,
};
use prism_state_tree_ecs::{SnapshotBridgeError, StateTreeAsset, StateTreeInstance};

#[derive(Clone, Copy, Debug)]
enum Domain {
    Flow,
    Animation,
}

impl Domain {
    fn label(self) -> &'static str {
        match self {
            Self::Flow => "flow",
            Self::Animation => "animation",
        }
    }

    fn event_id(self) -> EventId {
        match self {
            Self::Flow => EventId(10),
            Self::Animation => EventId(20),
        }
    }
}

struct Scenario {
    domain: Domain,
    initial: StateId,
    target: StateId,
    event: EventId,
    instance: StateTreeInstance,
    debugger: Debugger,
}

fn scenario(domain: Domain) -> Scenario {
    let initial = StateId(1);
    let target = StateId(2);
    let event = domain.event_id();
    let transition = Transition::new(
        initial,
        TransitionTarget::State(target),
        Trigger::Event(event),
        None,
        0,
        TransitionKind::External,
    );
    let plan = TreeDef {
        root: StateId(0),
        nodes: vec![
            StateNode::compound(StateId(0), None, vec![initial, target], initial, None),
            StateNode::atomic(initial, Some(StateId(0))).with_transition(transition),
            StateNode::atomic(target, Some(StateId(0))),
        ],
    }
    .compile()
    .expect("cross-domain plan must compile");
    let instance = StateTreeInstance::new(StateTreeAsset::new(plan), 4, MicrostepLimit(8))
        .expect("cross-domain instance must be constructible");
    let mut debugger = Debugger::with_capacities(2, 4, 0).expect("debugger capacity");
    debugger
        .add_breakpoint(Breakpoint::StateEntered(initial))
        .expect("initial breakpoint");
    debugger
        .add_breakpoint(Breakpoint::StateEntered(target))
        .expect("target breakpoint");
    Scenario {
        domain,
        initial,
        target,
        event,
        instance,
        debugger,
    }
}

fn run(mut scenario: Scenario) {
    scenario.instance.step().expect("initial stabilization");
    scenario
        .debugger
        .observe(DebugEvent::StateEntered(scenario.initial))
        .expect("initial observation");
    let checkpoint = scenario
        .instance
        .snapshot_bytes()
        .expect("portable checkpoint");

    scenario
        .instance
        .request_event(Event::new(scenario.event))
        .expect("event enqueue");
    scenario.instance.step().expect("domain transition");
    scenario
        .debugger
        .observe(DebugEvent::StateEntered(scenario.target))
        .expect("target observation");
    assert_eq!(
        scenario.instance.active_states(),
        &[StateId(0), scenario.target]
    );

    assert!(matches!(
        scenario.instance.restore_snapshot_bytes(b"invalid"),
        Err(SnapshotBridgeError::Decode(_))
    ));
    scenario
        .instance
        .restore_snapshot_bytes(&checkpoint)
        .expect("checkpoint restore");
    assert_eq!(
        scenario.instance.active_states(),
        &[StateId(0), scenario.initial]
    );
    assert_eq!(scenario.debugger.hits().len(), 2);
    println!(
        "{} domain shares StateTree, snapshot, debugger, and errors",
        scenario.domain.label()
    );
}

fn main() {
    run(scenario(Domain::Flow));
    run(scenario(Domain::Animation));
}
