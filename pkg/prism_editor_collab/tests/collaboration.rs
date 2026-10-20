//! Integration coverage for clocks, command replication, document CRDT state,
//! bounded history, presence leases, and soft locks.

use prism_editor_collab::{
    ActorId, CommandEnvelope, CommandId, CommandLimits, CommandPayload, DomainId, FieldPath,
    HlcClock, HlcTimestamp, LockError, PresenceError, PresenceRegistry, PresenceStatus, Replica,
    ResourceId, SoftLockRegistry, Value,
};
use std::collections::BTreeMap;

fn actor(name: &'static str) -> ActorId {
    ActorId::from_static(name)
}

fn domain() -> DomainId {
    DomainId::from_static("document")
}

fn path(name: &'static str) -> FieldPath {
    FieldPath::field(name).expect("test path")
}

fn set(
    who: &'static str,
    sequence: u64,
    physical: u64,
    causal: Vec<CommandId>,
    field: &'static str,
    value: Value,
) -> CommandEnvelope {
    CommandEnvelope::new(
        actor(who),
        sequence,
        HlcTimestamp::new(physical, 0),
        causal,
        [domain()].into_iter().collect(),
        CommandPayload::SetField {
            path: path(field),
            value,
        },
        CommandLimits::default(),
    )
    .expect("test command")
}

#[test]
fn hlc_is_monotonic_and_observes_remote_time() {
    let mut clock = HlcClock::new(actor("a"));
    let first = clock.tick(10).expect("first tick");
    let second = clock.tick(10).expect("logical tick");
    let observed = clock
        .observe(HlcTimestamp::new(20, 4), 12)
        .expect("observe remote");
    assert!(first < second);
    assert!(second < observed);
    assert_eq!(observed, HlcTimestamp::new(20, 5));
}

#[test]
fn command_normalizes_causal_order_and_rejects_self_dependency() {
    let dependency = CommandId {
        actor: actor("a"),
        sequence: 1,
    };
    let command = CommandEnvelope::new(
        actor("b"),
        2,
        HlcTimestamp::new(2, 0),
        vec![dependency.clone()],
        [domain()].into_iter().collect(),
        CommandPayload::Opaque(vec![1, 2]),
        CommandLimits::default(),
    )
    .expect("valid command");
    assert_eq!(command.causal, vec![dependency]);

    let self_id = CommandId {
        actor: actor("b"),
        sequence: 2,
    };
    let error = CommandEnvelope::new(
        actor("b"),
        2,
        HlcTimestamp::new(2, 0),
        vec![self_id],
        [domain()].into_iter().collect(),
        CommandPayload::Opaque(Vec::new()),
        CommandLimits::default(),
    )
    .expect_err("self dependency must fail");
    assert!(matches!(
        error,
        prism_editor_collab::CommandError::SelfDependency
    ));
}

#[test]
fn causal_command_is_buffered_until_predecessor_arrives() {
    let predecessor = set("a", 1, 1, Vec::new(), "name", Value::Text("one".into()));
    let successor = set(
        "b",
        1,
        2,
        vec![predecessor.id.clone()],
        "name",
        Value::Text("two".into()),
    );
    let mut replica = Replica::new(actor("local"), 16).expect("replica");
    let buffered = replica.receive(successor).expect("buffer");
    assert!(matches!(
        buffered,
        prism_editor_collab::ReceiveOutcome::Buffered(_)
    ));
    assert_eq!(replica.pending_ids().len(), 1);
    let applied = replica.receive(predecessor).expect("drain");
    assert!(matches!(applied, prism_editor_collab::ReceiveOutcome::Applied(ids) if ids.len() == 2));
    assert_eq!(
        replica.state().get(&path("name")),
        Some(&Value::Text("two".into()))
    );
}

#[test]
fn exact_duplicates_are_ignored_and_conflicting_duplicates_fail() {
    let command = set("a", 1, 1, Vec::new(), "value", Value::Int(1));
    let mut replica = Replica::new(actor("local"), 8).expect("replica");
    let first = replica.receive(command.clone()).expect("first");
    assert!(matches!(
        first,
        prism_editor_collab::ReceiveOutcome::Applied(_)
    ));
    let duplicate = replica.receive(command.clone()).expect("duplicate");
    assert!(matches!(
        duplicate,
        prism_editor_collab::ReceiveOutcome::Duplicate(_)
    ));
    let conflict = set("a", 1, 2, Vec::new(), "value", Value::Int(2));
    let error = replica.receive(conflict).expect_err("conflicting ID");
    assert!(matches!(
        error,
        prism_editor_collab::ApplyError::ConflictingDuplicate(_)
    ));
}

#[test]
fn field_lww_tombstone_prevents_older_resurrection() {
    let newer_delete = CommandEnvelope::new(
        actor("b"),
        1,
        HlcTimestamp::new(20, 0),
        Vec::new(),
        [domain()].into_iter().collect(),
        CommandPayload::DeleteField { path: path("x") },
        CommandLimits::default(),
    )
    .expect("delete");
    let older_write = set("a", 1, 10, Vec::new(), "x", Value::Int(7));
    let mut replica = Replica::new(actor("local"), 8).expect("replica");
    replica.receive(newer_delete).expect("delete apply");
    replica.receive(older_write).expect("older apply");
    assert_eq!(replica.state().get(&path("x")), None);
    assert!(replica.state().has_tombstone(&path("x")));
}

#[test]
fn replicas_converge_under_different_delivery_orders() {
    let a = set("a", 1, 10, Vec::new(), "shared", Value::Text("a".into()));
    let b = set("b", 1, 10, Vec::new(), "shared", Value::Text("b".into()));
    let c = set("a", 2, 11, vec![a.id.clone()], "other", Value::UInt(9));
    let mut left = Replica::new(actor("left"), 16).expect("left");
    let mut right = Replica::new(actor("right"), 16).expect("right");
    left.receive(a.clone()).expect("left a");
    left.receive(b.clone()).expect("left b");
    left.receive(c.clone()).expect("left c");
    right.receive(c).expect("right c buffered");
    right.receive(b).expect("right b");
    right.receive(a).expect("right a");
    assert_eq!(left.state(), right.state());
    assert_eq!(
        left.replay_all().expect("left replay"),
        right.state().clone()
    );
}

#[test]
fn bounded_history_reports_retention_floor_and_snapshot_replay() {
    let mut replica = Replica::new(actor("local"), 2).expect("replica");
    for sequence in 1..=4 {
        replica
            .receive(set(
                "a",
                sequence,
                sequence,
                Vec::new(),
                "field",
                Value::UInt(sequence),
            ))
            .expect("apply");
    }
    let info = replica.history_info();
    assert_eq!(info.retained, 2);
    assert_eq!(info.dropped, 2);
    assert!(replica.history_since(0).is_err());
    let snapshot = replica.snapshot();
    assert_eq!(
        replica
            .replay_from_snapshot(&snapshot)
            .expect("empty delta"),
        snapshot.state
    );
}

#[test]
fn replay_detects_missing_causal_dependency() {
    let missing = CommandId {
        actor: actor("a"),
        sequence: 99,
    };
    let command = set("b", 1, 2, vec![missing.clone()], "x", Value::Int(1));
    let mut replica = Replica::new(actor("local"), 8).expect("replica");
    replica.receive(command).expect("buffer");
    let error = replica.replay_all().expect_err("missing dependency");
    assert!(
        matches!(error, prism_editor_collab::ReplayError::MissingDependency { dependency, .. } if dependency == missing)
    );
}

#[test]
fn receive_rejects_causal_cycles_without_retaining_the_new_command() {
    let a_id = CommandId {
        actor: actor("a"),
        sequence: 1,
    };
    let b_id = CommandId {
        actor: actor("b"),
        sequence: 1,
    };
    let a = set("a", 1, 1, vec![b_id.clone()], "a", Value::Int(1));
    let b = set("b", 1, 1, vec![a_id], "b", Value::Int(2));
    let mut replica = Replica::new(actor("local"), 8).expect("replica");
    replica.receive(b).expect("first half is buffered");
    let error = replica.receive(a).expect_err("cycle");
    assert!(matches!(
        error,
        prism_editor_collab::ApplyError::CausalCycle(_)
    ));
    assert_eq!(replica.pending_ids(), vec![b_id]);
}

#[test]
fn presence_rejects_stale_lease_and_expires_deterministically() {
    let mut registry = PresenceRegistry::new(2);
    let lease = registry
        .join(actor("a"), "Alice", 10, 5, BTreeMap::new())
        .expect("join");
    assert_eq!(registry.snapshot().len(), 1);
    registry
        .heartbeat(&lease, 12, 5, PresenceStatus::Busy)
        .expect("heartbeat");
    let stale = registry
        .join(actor("a"), "duplicate", 13, 5, BTreeMap::new())
        .expect_err("duplicate session");
    assert_eq!(stale, PresenceError::AlreadyPresent);
    let expired = registry.expire(17);
    assert_eq!(expired, vec![actor("a")]);
    assert!(registry.snapshot().is_empty());
}

#[test]
fn presence_epoch_invalidates_old_session_after_rejoin() {
    let mut registry = PresenceRegistry::new(1);
    let first = registry
        .join(actor("a"), "Alice", 0, 1, BTreeMap::new())
        .expect("first");
    registry.expire(1);
    let second = registry
        .join(actor("a"), "Alice", 2, 5, BTreeMap::new())
        .expect("second");
    assert_ne!(first, second);
    assert_eq!(registry.leave(&first), Err(PresenceError::InvalidLease));
    registry.leave(&second).expect("leave current");
}

#[test]
fn soft_lock_has_exclusive_ownership_renewal_and_expiry() {
    let mut locks = SoftLockRegistry::new();
    let resource = ResourceId::from_static("scene/node");
    let lease = locks
        .acquire(actor("a"), resource.clone(), 10, 5)
        .expect("acquire");
    let held = locks
        .acquire(actor("b"), resource.clone(), 11, 5)
        .expect_err("held");
    assert!(matches!(held, LockError::Held(_)));
    locks.renew(&lease, 12, 5).expect("renew");
    assert!(locks.expire(16).is_empty());
    let expired = locks.expire(17);
    assert_eq!(expired.len(), 1);
    let replacement = locks
        .acquire(actor("b"), resource, 18, 2)
        .expect("replacement");
    assert_ne!(replacement, lease);
}

#[test]
fn soft_lock_rejects_stale_release() {
    let mut locks = SoftLockRegistry::new();
    let resource = ResourceId::from_static("asset");
    let lease = locks.acquire(actor("a"), resource, 1, 2).expect("acquire");
    assert_eq!(locks.expire(3).len(), 1);
    assert_eq!(locks.release(&lease), Err(LockError::InvalidLease));
}
