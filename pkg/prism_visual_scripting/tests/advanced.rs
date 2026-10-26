use prism_visual_scripting::prelude::*;
use std::sync::Arc;

#[test]
fn migration_registry_requires_a_contiguous_digest_bound_chain() {
    let mut registry = MigrationRegistry::default();
    registry
        .register(
            1,
            2,
            migration_digest("v1-v2", 1, 2),
            Arc::new(|value| match value {
                Value::I64(value) => Ok(Value::I64(value + 1)),
                other => Ok(other),
            }),
        )
        .unwrap();
    registry
        .register(
            2,
            3,
            migration_digest("v2-v3", 2, 3),
            Arc::new(|value| Ok(Value::List(Arc::new(vec![value])))),
        )
        .unwrap();
    assert_eq!(registry.plan(1, 3).unwrap().len(), 2);
    assert_eq!(
        registry.migrate(Value::I64(4), 1, 3).unwrap(),
        Value::List(Arc::new(vec![Value::I64(5)]))
    );
    assert!(registry.plan(1, 4).is_err());
}

#[test]
fn simulation_trace_and_seeded_draws_are_reproducible() {
    let mut first = SimulationHarness::new(7, 100);
    let draw_a = first.next_u64();
    let draw_b = first.next_u64();
    first.advance(25);
    let mut second = SimulationHarness::new(7, 100);
    assert_eq!(second.next_u64(), draw_a);
    assert_eq!(second.next_u64(), draw_b);
    second.advance(25);
    assert_eq!(first.digest().unwrap(), second.digest().unwrap());
    assert_eq!(first.trace(7, &[]).unwrap().seed, 7);
}

#[test]
fn fair_scheduler_bounds_tenants_prioritizes_and_sheds_deadlines() {
    let mut scheduler = FairScheduler::new(SchedulerLimits {
        max_active: 2,
        max_active_per_tenant: 1,
        max_queued: 8,
    })
    .unwrap();
    scheduler.admit("a1", "a", 1, None).unwrap();
    scheduler.admit("a2", "a", 9, None).unwrap();
    scheduler.admit("b1", "b", 2, None).unwrap();
    scheduler.admit("expired", "c", 255, Some(10)).unwrap();
    let first = scheduler.poll(10);
    assert_eq!(first.len(), 2);
    assert!(first.iter().any(|run| run.tenant == "a"));
    assert!(first.iter().any(|run| run.tenant == "b"));
    scheduler.complete(&first[0].run_id).unwrap();
    scheduler.complete(&first[1].run_id).unwrap();
    let second = scheduler.poll(10);
    assert_eq!(second.len(), 1);
    assert_eq!(second[0].run_id, "a1");
}

#[test]
fn scheduler_stops_when_only_tenant_is_at_its_active_limit() {
    let mut scheduler = FairScheduler::new(SchedulerLimits {
        max_active: 2,
        max_active_per_tenant: 1,
        max_queued: 4,
    })
    .unwrap();
    scheduler.admit("a1", "a", 1, None).unwrap();
    scheduler.admit("a2", "a", 1, None).unwrap();
    let ready = scheduler.poll(0);
    assert_eq!(ready.len(), 1);
    assert_eq!(scheduler.queued_len(), 1);
    assert!(scheduler.poll(0).is_empty());
}

#[test]
fn scheduler_releases_run_identity_after_completion_and_expiry() {
    let mut scheduler = FairScheduler::new(SchedulerLimits {
        max_active: 1,
        max_active_per_tenant: 1,
        max_queued: 4,
    })
    .unwrap();
    scheduler.admit("run", "tenant", 1, None).unwrap();
    assert_eq!(
        scheduler.admit("run", "tenant", 1, None).unwrap(),
        AdmitResult::Duplicate
    );
    let active = scheduler.poll(0);
    scheduler.complete(&active[0].run_id).unwrap();
    assert_eq!(
        scheduler.admit("run", "tenant", 1, None).unwrap(),
        AdmitResult::Queued
    );

    scheduler.admit("expiring", "tenant", 1, Some(5)).unwrap();
    scheduler.poll(5);
    scheduler.complete("run").unwrap();
    scheduler.poll(5);
    assert_eq!(
        scheduler.admit("expiring", "tenant", 1, None).unwrap(),
        AdmitResult::Queued
    );
}
#[test]
fn determinism_gate_rejects_unapproved_native_runner() {
    let library = NodeLibrary::with_builtins();
    let graph = Graph {
        nodes: vec![Node {
            id: 1,
            def: NodeDefRef {
                id: "event.begin".into(),
                version: None,
            },
            title: None,
            config: Default::default(),
        }],
        entries: vec![1],
        ..Default::default()
    };
    let vm = Vm::new(compile(&graph, &library).unwrap());
    let denied = SimulationDeterminismGate::default();
    assert!(vm
        .run_all_with_env(
            &mut InMemoryHost::default(),
            &ExecutionEnv::new().with_determinism(&denied),
        )
        .unwrap_err()
        .to_string()
        .contains("not admitted"));
    let mut allowed = SimulationDeterminismGate::default();
    allowed.allow("event.begin", RunnerClass::Sync);
    vm.run_all_with_env(
        &mut InMemoryHost::default(),
        &ExecutionEnv::new().with_determinism(&allowed),
    )
    .unwrap();
}
