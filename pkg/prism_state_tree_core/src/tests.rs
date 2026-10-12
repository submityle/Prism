use alloc::vec;
use std::vec::Vec;

use super::*;

#[derive(Default)]
struct Recorder(Vec<&'static str>);

impl LifecycleCallback for Recorder {
    fn on_enter(&mut self, state: StateId) {
        self.0.push(match state.0 {
            0 => "root+",
            1 => "a+",
            2 => "b+",
            3 => "c+",
            4 => "p+",
            5 => "r1+",
            6 => "r2+",
            _ => "?+",
        });
    }
    fn on_exit(&mut self, state: StateId) {
        self.0.push(match state.0 {
            0 => "root-",
            1 => "a-",
            2 => "b-",
            3 => "c-",
            4 => "p-",
            5 => "r1-",
            6 => "r2-",
            _ => "?-",
        });
    }
}

fn basic_plan() -> Plan {
    let root = StateNode::compound(
        StateId(0),
        None,
        vec![StateId(1), StateId(2)],
        StateId(1),
        None,
    );
    let a = StateNode::atomic(StateId(1), Some(StateId(0))).with_transition(Transition::new(
        StateId(1),
        TransitionTarget::State(StateId(2)),
        Trigger::Event(EventId(1)),
        None,
        1,
        TransitionKind::External,
    ));
    let b = StateNode::atomic(StateId(2), Some(StateId(0)));
    TreeDef {
        root: StateId(0),
        nodes: vec![root, a, b],
    }
    .compile()
    .unwrap()
}

#[test]
fn lifecycle_uses_lca_and_event_queue_is_fifo() {
    let mut instance = Instance::new(basic_plan(), 4, MicrostepLimit(8)).unwrap();
    let mut recorder = Recorder::default();
    instance.start(&mut recorder).unwrap();
    assert_eq!(recorder.0, ["root+", "a+"]);
    instance.send(Event::new(EventId(1))).unwrap();
    let report = instance.run(&mut recorder).unwrap();
    assert_eq!(report.events_processed, 1);
    assert_eq!(recorder.0, ["root+", "a+", "a-", "b+"]);
    assert_eq!(instance.active_states(), &[StateId(0), StateId(2)]);
}

#[test]
fn ancestor_transition_is_inherited_by_active_descendants() {
    let root = StateNode::compound(
        StateId(0),
        None,
        vec![StateId(1), StateId(2)],
        StateId(1),
        None,
    )
    .with_transition(Transition::new(
        StateId(0),
        TransitionTarget::State(StateId(2)),
        Trigger::Event(EventId(4)),
        None,
        0,
        TransitionKind::External,
    ));
    let a = StateNode::atomic(StateId(1), Some(StateId(0)));
    let b = StateNode::atomic(StateId(2), Some(StateId(0)));
    let plan = TreeDef {
        root: StateId(0),
        nodes: vec![root, a, b],
    }
    .compile()
    .unwrap();
    let mut instance = Instance::new(plan, 2, MicrostepLimit(4)).unwrap();
    let mut callback = NoopLifecycle;
    instance.start(&mut callback).unwrap();
    instance.send(Event::new(EventId(4))).unwrap();
    instance.run(&mut callback).unwrap();
    assert!(instance.is_active(StateId(2)));
}

#[test]
fn priority_then_declaration_order_is_deterministic() {
    let root = StateNode::compound(
        StateId(0),
        None,
        vec![StateId(1), StateId(2), StateId(3)],
        StateId(1),
        None,
    );
    let a = StateNode::atomic(StateId(1), Some(StateId(0))).with_transition(Transition::new(
        StateId(1),
        TransitionTarget::State(StateId(2)),
        Trigger::Event(EventId(7)),
        None,
        4,
        TransitionKind::External,
    ));
    let b = StateNode::atomic(StateId(2), Some(StateId(0))).with_transition(Transition::new(
        StateId(2),
        TransitionTarget::State(StateId(3)),
        Trigger::Event(EventId(7)),
        None,
        0,
        TransitionKind::External,
    ));
    let c = StateNode::atomic(StateId(3), Some(StateId(0)));
    let plan = TreeDef {
        root: StateId(0),
        nodes: vec![root, a, b, c],
    }
    .compile()
    .unwrap();
    let mut instance = Instance::new(plan, 2, MicrostepLimit(4)).unwrap();
    let mut callback = NoopLifecycle;
    instance.start(&mut callback).unwrap();
    instance.send(Event::new(EventId(7))).unwrap();
    instance.run(&mut callback).unwrap();
    assert!(instance.is_active(StateId(2)));
}

#[test]
fn internal_and_self_have_distinct_lifecycle_semantics() {
    let root = StateNode::compound(StateId(0), None, vec![StateId(1)], StateId(1), None);
    let a = StateNode::atomic(StateId(1), Some(StateId(0))).with_transition(Transition::new(
        StateId(1),
        TransitionTarget::SelfState,
        Trigger::Event(EventId(1)),
        None,
        0,
        TransitionKind::Internal,
    ));
    let plan = TreeDef {
        root: StateId(0),
        nodes: vec![root, a],
    }
    .compile()
    .unwrap();
    let mut i = Instance::new(plan, 4, MicrostepLimit(4)).unwrap();
    let mut r = Recorder::default();
    i.start(&mut r).unwrap();
    i.send(Event::new(EventId(1))).unwrap();
    i.run(&mut r).unwrap();
    assert_eq!(r.0, ["root+", "a+"]);
    let root = StateNode::compound(StateId(0), None, vec![StateId(1)], StateId(1), None);
    let a = StateNode::atomic(StateId(1), Some(StateId(0))).with_transition(Transition::new(
        StateId(1),
        TransitionTarget::SelfState,
        Trigger::Event(EventId(1)),
        None,
        0,
        TransitionKind::External,
    ));
    let plan = TreeDef {
        root: StateId(0),
        nodes: vec![root, a],
    }
    .compile()
    .unwrap();
    let mut i = Instance::new(plan, 4, MicrostepLimit(4)).unwrap();
    let mut r = Recorder::default();
    i.start(&mut r).unwrap();
    i.send(Event::new(EventId(1))).unwrap();
    i.run(&mut r).unwrap();
    assert_eq!(r.0, ["root+", "a+", "a-", "a+"]);
}

#[test]
fn external_transition_to_ancestor_exits_and_reenters_ancestor() {
    let root = StateNode::compound(StateId(0), None, vec![StateId(1)], StateId(1), None);
    let parent = StateNode::compound(
        StateId(1),
        Some(StateId(0)),
        vec![StateId(2), StateId(3)],
        StateId(2),
        None,
    );
    let active = StateNode::atomic(StateId(2), Some(StateId(1))).with_transition(Transition::new(
        StateId(2),
        TransitionTarget::State(StateId(1)),
        Trigger::Event(EventId(9)),
        None,
        0,
        TransitionKind::External,
    ));
    let other = StateNode::atomic(StateId(3), Some(StateId(1)));
    let plan = TreeDef {
        root: StateId(0),
        nodes: vec![root, parent, active, other],
    }
    .compile()
    .unwrap();
    let mut instance = Instance::new(plan, 2, MicrostepLimit(4)).unwrap();
    let mut recorder = Recorder::default();
    instance.start(&mut recorder).unwrap();
    instance.send(Event::new(EventId(9))).unwrap();
    instance.run(&mut recorder).unwrap();

    assert_eq!(recorder.0, ["root+", "a+", "b+", "b-", "a-", "a+", "b+"]);
    assert_eq!(
        instance.active_states(),
        &[StateId(0), StateId(1), StateId(2)]
    );
}

#[test]
fn eventless_cascade_is_bounded_and_events_do_not_insert() {
    let root = StateNode::compound(
        StateId(0),
        None,
        vec![StateId(1), StateId(2)],
        StateId(1),
        None,
    );
    let a = StateNode::atomic(StateId(1), Some(StateId(0))).with_transition(Transition::new(
        StateId(1),
        TransitionTarget::State(StateId(2)),
        Trigger::Eventless,
        None,
        0,
        TransitionKind::External,
    ));
    let b = StateNode::atomic(StateId(2), Some(StateId(0))).with_transition(Transition::new(
        StateId(2),
        TransitionTarget::State(StateId(1)),
        Trigger::Eventless,
        None,
        0,
        TransitionKind::External,
    ));
    let plan = TreeDef {
        root: StateId(0),
        nodes: vec![root, a, b],
    }
    .compile()
    .unwrap();
    let mut i = Instance::new(plan, 1, MicrostepLimit(3)).unwrap();
    let mut c = NoopLifecycle;
    assert_eq!(
        i.start(&mut c),
        Err(RuntimeError::MicrostepLimit(MicrostepLimit(3)))
    );
    assert_eq!(i.send(Event::new(EventId(9))), Ok(()));
    assert_eq!(i.send(Event::new(EventId(9))), Err(EventQueueError::Full));
}

#[test]
fn shallow_and_deep_history_restore() {
    let root = StateNode::compound(
        StateId(0),
        None,
        vec![StateId(1), StateId(5)],
        StateId(1),
        None,
    );
    let hist = StateNode::compound(
        StateId(1),
        Some(StateId(0)),
        vec![StateId(2), StateId(3), StateId(4)],
        StateId(2),
        Some(HistoryMode::Deep),
    );
    let a = StateNode::atomic(StateId(2), Some(StateId(1))).with_transition(Transition::new(
        StateId(2),
        TransitionTarget::State(StateId(3)),
        Trigger::Event(EventId(2)),
        None,
        0,
        TransitionKind::External,
    ));
    let b = StateNode::atomic(StateId(3), Some(StateId(1)));
    let c = StateNode::atomic(StateId(4), Some(StateId(1)));
    let out = StateNode::atomic(StateId(5), Some(StateId(0))).with_transition(Transition::new(
        StateId(5),
        TransitionTarget::History {
            owner: StateId(1),
            depth: HistoryMode::Deep,
        },
        Trigger::Event(EventId(3)),
        None,
        0,
        TransitionKind::External,
    ));
    let plan = TreeDef {
        root: StateId(0),
        nodes: vec![root, hist, a, b, c, out],
    }
    .compile()
    .unwrap();
    let mut i = Instance::new(plan, 4, MicrostepLimit(8)).unwrap();
    let mut cb = NoopLifecycle;
    i.start(&mut cb).unwrap();
    i.send(Event::new(EventId(2))).unwrap();
    i.run(&mut cb).unwrap();
    assert!(i.is_active(StateId(3)));
    i.send(Event::new(EventId(3))).unwrap();
    i.run(&mut cb).unwrap();
    assert!(i.is_active(StateId(3)));
}

#[test]
fn parallel_regions_are_all_active_in_declaration_order() {
    let root = StateNode::compound(StateId(0), None, vec![StateId(1)], StateId(1), None);
    let parallel = StateNode::parallel(
        StateId(1),
        Some(StateId(0)),
        vec![StateId(2), StateId(3)],
        vec![StateId(2), StateId(3)],
        None,
    );
    let left = StateNode::atomic(StateId(2), Some(StateId(1)));
    let right = StateNode::atomic(StateId(3), Some(StateId(1)));
    let plan = TreeDef {
        root: StateId(0),
        nodes: vec![root, parallel, left, right],
    }
    .compile()
    .unwrap();
    let mut i = Instance::new(plan, 2, MicrostepLimit(4)).unwrap();
    let mut cb = NoopLifecycle;
    i.start(&mut cb).unwrap();
    assert_eq!(
        i.active_states(),
        &[StateId(0), StateId(1), StateId(2), StateId(3)]
    );
}

#[test]
fn queue_rejects_zero_capacity_and_missing_guard_is_false() {
    assert!(matches!(
        Instance::new(basic_plan(), 0, MicrostepLimit(1)),
        Err(RuntimeError::EventQueue(EventQueueError::ZeroCapacity))
    ));
    let root = StateNode::compound(
        StateId(0),
        None,
        vec![StateId(1), StateId(2)],
        StateId(1),
        None,
    );
    let a = StateNode::atomic(StateId(1), Some(StateId(0))).with_transition(Transition::new(
        StateId(1),
        TransitionTarget::State(StateId(2)),
        Trigger::Event(EventId(1)),
        Some(GuardId(99)),
        0,
        TransitionKind::External,
    ));
    let b = StateNode::atomic(StateId(2), Some(StateId(0)));
    let plan = TreeDef {
        root: StateId(0),
        nodes: vec![root, a, b],
    }
    .compile()
    .unwrap();
    let mut i = Instance::new(plan, 1, MicrostepLimit(2)).unwrap();
    let mut c = NoopLifecycle;
    i.start(&mut c).unwrap();
    i.send(Event::new(EventId(1))).unwrap();
    i.run(&mut c).unwrap();
    assert!(i.is_active(StateId(1)));
}

#[test]
fn compiler_rejects_inconsistent_parent_links() {
    let root = StateNode::compound(StateId(0), None, vec![StateId(1)], StateId(1), None);
    let child = StateNode::atomic(StateId(1), Some(StateId(0)));
    let unrelated = StateNode::atomic(StateId(2), Some(StateId(0)));
    assert!(matches!(
        (TreeDef {
            root: StateId(0),
            nodes: vec![root, child, unrelated]
        })
        .compile(),
        Err(PlanError::ParentMismatch(StateId(2)))
    ));
}
