use alloc::{vec, vec::Vec};

use prism_ecs::{command::CommandQueue, component::Component, world::World};
use prism_state_tree_core::{
    ContextRef, Event, EventId, Guard, GuardContext, GuardId, MicrostepLimit, NodeKind, Schema,
    SchemaDescriptor, SchemaError, SchemaId, SchemaVersion, StateId, StateNode, Transition,
    TransitionKind, TransitionTarget, TreeDef, Trigger,
};

use crate::{
    gather_context, queue_spawn, tick_all, tick_entities, tick_entity, validate_context,
    BatchTickFailure, BatchTickMissingReason, BatchTickOutcome, EcsContextError, EcsSchema,
    StateTreeAsset, StateTreeInstance, StateTreeLifecycleEvent, ValidatedContext,
};

struct TestContext<'w> {
    schema: SchemaId,
    value: &'w u32,
}

impl<'w> ContextRef<'w> for TestContext<'w> {
    fn schema_id(&self) -> SchemaId {
        self.schema
    }
}

struct TestSchema;
const TEST_SCHEMA_ID: SchemaId = SchemaId::new(0x1234);
const WRONG_SCHEMA_ID: SchemaId = SchemaId::new(0x9999);

struct Marker;
impl Component for Marker {}

struct ContextValue(u32);
impl Component for ContextValue {}

impl Schema for TestSchema {
    type Context<'w> = TestContext<'w>;

    fn descriptor() -> SchemaDescriptor {
        SchemaDescriptor::new(TEST_SCHEMA_ID, SchemaVersion::new(1, 0), Default::default())
    }
}

impl EcsSchema for TestSchema {
    fn gather<'w>(
        world: &'w World,
        owner: prism_ecs::entity::Entity,
    ) -> Result<Self::Context<'w>, EcsContextError>
    where
        Self: 'w,
    {
        if !world.contains(owner) {
            return Err(EcsContextError::MissingOwner(owner));
        }
        let value = world
            .get::<ContextValue>(owner)
            .ok_or(EcsContextError::MissingContext(owner))?;
        Ok(TestContext {
            schema: TEST_SCHEMA_ID,
            value: &value.0,
        })
    }
}

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
fn lifecycle_callbacks_are_retained_and_drained_in_core_order() {
    let mut instance = StateTreeInstance::new(asset(), 8, MicrostepLimit(16)).expect("setup");

    instance.step().expect("initial entry");
    assert_eq!(
        instance.lifecycle_events(),
        &[
            StateTreeLifecycleEvent::Entered(StateId(0)),
            StateTreeLifecycleEvent::Entered(StateId(1)),
        ]
    );
    assert_eq!(
        instance.drain_lifecycle_events(),
        vec![
            StateTreeLifecycleEvent::Entered(StateId(0)),
            StateTreeLifecycleEvent::Entered(StateId(1)),
        ]
    );
    assert!(instance.lifecycle_events().is_empty());

    instance
        .request_event(Event::new(EventId(7)))
        .expect("event fits bounded queue");
    instance.step().expect("transition tick");
    assert_eq!(
        instance.drain_lifecycle_events(),
        vec![
            StateTreeLifecycleEvent::Exited(StateId(1)),
            StateTreeLifecycleEvent::Entered(StateId(2)),
            StateTreeLifecycleEvent::Transitioned {
                source: StateId(1),
                target: TransitionTarget::State(StateId(2)),
            },
        ]
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
    instance.set_guard(GuardId(3), MatchEvent(EventId(9)));
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

#[test]
fn event_queue_capacity_is_enforced_at_the_ecs_boundary() {
    let mut instance =
        StateTreeInstance::new(asset(), 1, MicrostepLimit(8)).expect("runtime setup");
    instance
        .request_event(Event::new(EventId(7)))
        .expect("first event fits");
    assert_eq!(
        instance.request_event(Event::new(EventId(7))),
        Err(prism_state_tree_core::EventQueueError::Full)
    );
}

#[test]
fn tick_reports_budget_exhaustion_without_losing_the_event() {
    let mut world = World::new();
    let mut queue = CommandQueue::new();
    let entity = {
        let mut commands = queue.commands(world.entities());
        queue_spawn(&mut commands, asset(), 1, MicrostepLimit(0)).expect("spawn setup")
    };
    queue.apply(&mut world);

    tick_entity(&mut world, entity)
        .unwrap()
        .expect("initial entry has no transition");
    world
        .get_mut::<StateTreeInstance>(entity)
        .unwrap()
        .request_event(Event::new(EventId(7)))
        .expect("event fits bounded queue");
    assert_eq!(
        tick_entity(&mut world, entity).unwrap(),
        Err(prism_state_tree_core::RuntimeError::MicrostepLimit(
            MicrostepLimit(0)
        ))
    );
}

#[test]
fn tick_of_missing_or_despawned_entity_is_a_safe_noop() {
    let mut world = World::new();
    let missing = world.entities().reserve_entity();
    world.flush_reserved();
    assert!(tick_entity(&mut world, missing).is_none());

    let entity = world.spawn(StateTreeInstance::new(asset(), 1, MicrostepLimit(8)).unwrap());
    assert!(world.despawn(entity));
    assert!(tick_entity(&mut world, entity).is_none());
}

#[test]
fn batch_tick_preserves_input_order_and_reports_missing_entities() {
    let mut world = World::new();
    let first = world.spawn(StateTreeInstance::new(asset(), 4, MicrostepLimit(8)).unwrap());
    let second = world.spawn(StateTreeInstance::new(asset(), 4, MicrostepLimit(8)).unwrap());
    let missing = world.entities().reserve_entity();
    world.flush_reserved();

    let report = tick_entities(&mut world, &[second, missing, first, second]);
    assert_eq!(report.len(), 4);
    assert_eq!(report.completed_count(), 3);
    assert_eq!(report.missing_count(), 1);
    assert_eq!(report.failed_count(), 0);
    assert_eq!(
        report.entries(),
        &[
            crate::BatchTickEntry {
                entity: second,
                outcome: BatchTickOutcome::Completed(prism_state_tree_core::StepReport {
                    events_processed: 0,
                    microsteps: 0,
                    queued_events: 0,
                }),
            },
            crate::BatchTickEntry {
                entity: missing,
                outcome: BatchTickOutcome::Missing(BatchTickMissingReason::NotMaterialized),
            },
            crate::BatchTickEntry {
                entity: first,
                outcome: BatchTickOutcome::Completed(prism_state_tree_core::StepReport {
                    events_processed: 0,
                    microsteps: 0,
                    queued_events: 0,
                }),
            },
            crate::BatchTickEntry {
                entity: second,
                outcome: BatchTickOutcome::Completed(prism_state_tree_core::StepReport {
                    events_processed: 0,
                    microsteps: 0,
                    queued_events: 0,
                }),
            },
        ]
    );
}

#[test]
fn batch_tick_classifies_missing_entity_reasons() {
    let mut world = World::new();
    let reserved = world.entities().reserve_entity();
    world.flush_reserved();
    let materialized_without_instance = world.spawn(());
    let stale = world.entities().reserve_entity();
    world.flush_reserved();
    assert!(world.despawn(stale));

    let report = tick_entities(
        &mut world,
        &[reserved, materialized_without_instance, stale],
    );
    assert_eq!(
        report.entries()[0].outcome,
        BatchTickOutcome::Missing(BatchTickMissingReason::NotMaterialized)
    );
    assert_eq!(
        report.entries()[1].outcome,
        BatchTickOutcome::Missing(BatchTickMissingReason::MissingComponent)
    );
    assert_eq!(
        report.entries()[2].outcome,
        BatchTickOutcome::Missing(BatchTickMissingReason::StaleEntity)
    );
}

#[test]
fn batch_tick_executes_duplicate_entity_handles_twice() {
    let mut world = World::new();
    let entity = world.spawn(StateTreeInstance::new(asset(), 4, MicrostepLimit(8)).unwrap());
    tick_entity(&mut world, entity)
        .expect("instance exists")
        .expect("initial tick");
    let mut instance = world.get_mut::<StateTreeInstance>(entity).unwrap();
    instance.request_event(Event::new(EventId(7))).unwrap();
    instance.request_event(Event::new(EventId(7))).unwrap();

    let report = tick_entities(&mut world, &[entity, entity]);
    assert_eq!(report.completed_count(), 2);
    assert!(matches!(
        &report.entries()[0].outcome,
        BatchTickOutcome::Completed(step) if step.events_processed == 2 && step.microsteps == 1
    ));
    assert!(matches!(
        &report.entries()[1].outcome,
        BatchTickOutcome::Completed(step) if step.events_processed == 0 && step.microsteps == 0
    ));
    assert_eq!(
        world
            .get::<StateTreeInstance>(entity)
            .unwrap()
            .queued_events(),
        0
    );
}

#[test]
fn batch_tick_keeps_instances_independent_and_continues_after_error() {
    let mut world = World::new();
    let failing = world.spawn(StateTreeInstance::new(asset(), 4, MicrostepLimit(0)).unwrap());
    let healthy = world.spawn(StateTreeInstance::new(asset(), 4, MicrostepLimit(8)).unwrap());

    let initial = tick_entities(&mut world, &[healthy, failing]);
    assert_eq!(initial.completed_count(), 2);

    world
        .get_mut::<StateTreeInstance>(failing)
        .unwrap()
        .request_event(Event::new(EventId(7)))
        .expect("event fits bounded queue");
    world
        .get_mut::<StateTreeInstance>(healthy)
        .unwrap()
        .request_event(Event::new(EventId(7)))
        .expect("event fits bounded queue");

    let report = tick_entities(&mut world, &[failing, healthy]);
    assert_eq!(report.completed_count(), 1);
    assert_eq!(report.failed_count(), 1);
    assert_eq!(report.entries()[0].entity, failing);
    match &report.entries()[0].outcome {
        BatchTickOutcome::Failed(failure) => {
            assert_eq!(
                failure.error,
                prism_state_tree_core::RuntimeError::MicrostepLimit(MicrostepLimit(0))
            );
            assert_eq!(failure.active_states, [StateId(0), StateId(1)]);
            assert_eq!(failure.queued_events, 1);
        }
        other => panic!("expected a failed tick, got {other:?}"),
    }
    assert_eq!(report.entries()[1].entity, healthy);
    assert!(matches!(
        report.entries()[1].outcome,
        BatchTickOutcome::Completed(report) if report.events_processed == 1 && report.microsteps == 1
    ));
    assert_eq!(
        world
            .get::<StateTreeInstance>(failing)
            .unwrap()
            .active_states(),
        &[StateId(0), StateId(1)]
    );
    assert_eq!(
        world
            .get::<StateTreeInstance>(healthy)
            .unwrap()
            .active_states(),
        &[StateId(0), StateId(2)]
    );
}

#[test]
fn batch_tick_failure_reports_post_transition_state() {
    let event_transition = Transition::new(
        StateId(1),
        TransitionTarget::State(StateId(2)),
        Trigger::Event(EventId(7)),
        None,
        0,
        TransitionKind::External,
    );
    let eventless_transition = Transition::new(
        StateId(2),
        TransitionTarget::State(StateId(3)),
        Trigger::Eventless,
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
                vec![StateId(1), StateId(2), StateId(3)],
                StateId(1),
                None,
            ),
            StateNode::atomic(StateId(1), Some(StateId(0))).with_transition(event_transition),
            StateNode::atomic(StateId(2), Some(StateId(0))).with_transition(eventless_transition),
            StateNode::atomic(StateId(3), Some(StateId(0))),
        ],
    }
    .compile()
    .expect("partial-failure plan is valid");
    let asset = StateTreeAsset::new(plan);
    let mut world = World::new();
    let entity = world.spawn(StateTreeInstance::new(asset, 4, MicrostepLimit(1)).unwrap());
    tick_entity(&mut world, entity)
        .expect("instance exists")
        .expect("initial tick");
    world
        .get_mut::<StateTreeInstance>(entity)
        .unwrap()
        .request_event(Event::new(EventId(7)))
        .unwrap();

    let report = tick_entities(&mut world, &[entity]);
    assert_eq!(report.failed_count(), 1);
    assert_eq!(
        report.entries()[0].outcome,
        BatchTickOutcome::Failed(BatchTickFailure {
            error: prism_state_tree_core::RuntimeError::MicrostepLimit(MicrostepLimit(1)),
            active_states: vec![StateId(0), StateId(2)],
            queued_events: 0,
        })
    );
    assert_eq!(
        world
            .get::<StateTreeInstance>(entity)
            .unwrap()
            .active_states(),
        &[StateId(0), StateId(2)]
    );
}

#[test]
fn empty_batch_is_a_structured_noop() {
    let mut world = World::new();
    let report = tick_entities(&mut world, &[]);
    assert!(report.is_empty());
    assert_eq!(report.entries(), &[]);
    assert_eq!(report.completed_count(), 0);
    assert_eq!(report.missing_count(), 0);
    assert_eq!(report.failed_count(), 0);
}

#[test]
fn tick_all_gathers_only_state_tree_components_in_entity_order() {
    let mut world = World::new();
    let first = world.spawn(StateTreeInstance::new(asset(), 4, MicrostepLimit(8)).unwrap());
    let marker = world.spawn((
        StateTreeInstance::new(asset(), 4, MicrostepLimit(8)).unwrap(),
        Marker,
    ));
    let second = world.spawn(StateTreeInstance::new(asset(), 4, MicrostepLimit(8)).unwrap());
    world.despawn(marker);
    let replacement = world.spawn((
        StateTreeInstance::new(asset(), 4, MicrostepLimit(8)).unwrap(),
        Marker,
    ));

    let report = tick_all(&mut world).expect("gather allocation");
    let mut expected = vec![first, second, replacement];
    expected.sort_unstable();
    assert_eq!(
        report
            .entries()
            .iter()
            .map(|entry| entry.entity)
            .collect::<Vec<_>>(),
        expected
    );
    assert_ne!(marker, replacement);
    assert_eq!(report.completed_count(), 3);
    assert_eq!(report.missing_count(), 0);
    assert_eq!(report.failed_count(), 0);
}

#[test]
fn validates_borrowed_schema_context_without_implicit_world_gather() {
    let value = 42;
    let context = TestContext {
        schema: TEST_SCHEMA_ID,
        value: &value,
    };
    assert!(validate_context::<TestSchema>(&context).is_ok());
    let validated = ValidatedContext::<TestSchema>::new(context).expect("schema matches");
    assert_eq!(*validated.context().value, 42);

    let wrong = TestContext {
        schema: WRONG_SCHEMA_ID,
        value: &value,
    };
    assert_eq!(
        validate_context::<TestSchema>(&wrong),
        Err(SchemaError::ContextSchemaMismatch {
            expected: TEST_SCHEMA_ID,
            found: WRONG_SCHEMA_ID,
        })
    );
    assert!(matches!(
        ValidatedContext::<TestSchema>::new(wrong),
        Err(SchemaError::ContextSchemaMismatch {
            expected: TEST_SCHEMA_ID,
            found: WRONG_SCHEMA_ID,
        })
    ));
}

#[test]
fn gathers_typed_context_from_the_owner_without_hidden_world_access() {
    let mut world = World::new();
    let owner = world.spawn(ContextValue(84));
    let gathered = gather_context::<TestSchema>(&world, owner).expect("context gather");
    assert_eq!(*gathered.context().value, 84);

    let missing_component = world.spawn(());
    assert!(matches!(
        gather_context::<TestSchema>(&world, missing_component),
        Err(EcsContextError::MissingContext(entity)) if entity == missing_component
    ));
    let stale = world.entities().reserve_entity();
    assert!(matches!(
        gather_context::<TestSchema>(&world, stale),
        Err(EcsContextError::MissingOwner(entity)) if entity == stale
    ));
}
