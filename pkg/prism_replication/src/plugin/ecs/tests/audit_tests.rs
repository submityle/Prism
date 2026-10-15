use super::*;
use crate::multiplex::{NetChannel, ProtocolError};
use crate::plugin::{ConnectionState, FrameHashAuditError, FrameHashUpdate, ServerReplication};
use alloc::vec::Vec;
use prism_app::SnapshotHistory;
use prism_ecs::world::snapshot::{FrameHash, TickDivergence};

fn pair(
    capacity: usize,
    config: ReplicationConfig,
) -> (World, ServerReplication<AuthorityStrategy>, Entity) {
    let (a, b) = InMemoryTransport::pair_secure();
    let mut world = World::new();
    let entity = world.spawn(Counter(0));
    world.register_snapshot_component_hashable::<Counter>();
    world.insert_resource(SnapshotHistory::new(capacity));
    world.insert_resource(
        EcsReplication::new_with_config(
            schema(),
            capacity,
            &seed(),
            ClientStrategy { entity },
            Box::new(a),
            config,
        )
        .unwrap(),
    );
    let mut server = ServerReplication::new_with_config(
        schema(),
        capacity,
        &seed(),
        AuthorityStrategy,
        Box::new(b),
        config,
    )
    .unwrap();
    exchange(&mut world, &mut server, 12);
    assert!(world
        .resource::<EcsReplication<ClientStrategy>>()
        .replication()
        .negotiated()
        .is_some());
    (world, server, entity)
}

fn exchange(world: &mut World, server: &mut ServerReplication<AuthorityStrategy>, rounds: usize) {
    for _ in 0..rounds {
        world
            .resource_mut::<EcsReplication<ClientStrategy>>()
            .replication
            .pump();
        server.pump();
    }
}

fn exchange_deferred(
    world: &mut World,
    server: &mut ServerReplication<AuthorityStrategy>,
    rounds: usize,
) {
    for _ in 0..rounds {
        world
            .resource_mut::<EcsReplication<ClientStrategy>>()
            .replication
            .pump_deferred();
        server.pump();
    }
}

fn queue(world: &mut World, input: i64) {
    world
        .resource_mut::<EcsReplication<ClientStrategy>>()
        .queue_input(input);
}

fn packet(tick: u64, hash: u64, sequence: u64) -> FrameHashUpdate {
    FrameHashUpdate {
        frame: FrameHash { tick, hash },
        sequence,
    }
}

#[test]
fn ecs_capture_hashes_cross_real_envelopes_and_match_authoritative_history() {
    let (mut world, mut server, _) = pair(8, ReplicationConfig::for_capacity(8));
    queue(&mut world, 5);
    queue(&mut world, 1);
    drive_ecs_input::<ClientStrategy>(&mut world);

    // An independent authority ECS uses the same schema, allocator and tick
    // cadence. Its Record proxy is intentionally NOT used as the hash source.
    let mut authority = World::new();
    let entity = authority.spawn(Counter(0));
    authority.register_snapshot_component_hashable::<Counter>();
    let mut history = SnapshotHistory::new(8);
    history.capture(&authority);
    authority.get_mut::<Counter>(entity).unwrap().0 = 5;
    history.capture(&authority);
    for tick in 0..2 {
        assert_eq!(
            history.state_hash(tick),
            world.resource::<SnapshotHistory>().state_hash(tick)
        );
    }
    server.sync_frame_hashes(&history, 0).unwrap();
    server.send_event(b"business event").unwrap();
    exchange(&mut world, &mut server, 24);
    let client = world.resource_mut::<EcsReplication<ClientStrategy>>();
    assert_eq!(
        client.replication.recv_event(),
        Some(b"business event".to_vec())
    );
    assert!(client
        .replication
        .frame_hash_audit()
        .comparison()
        .retained_window_matches());
    assert!(server
        .frame_hash_audit()
        .comparison()
        .retained_window_matches());
    assert!(client.replication.take_audit_error().is_none());
    assert!(server.take_audit_error().is_none());
}

#[test]
fn authority_rollback_replaces_already_transmitted_audit_tick_and_clears_divergence() {
    let (mut world, mut server, entity) = pair(8, ReplicationConfig::for_capacity(8));
    queue(&mut world, 5);
    queue(&mut world, 1);
    drive_ecs_input::<ClientStrategy>(&mut world);
    let before = world
        .resource::<EcsReplication<ClientStrategy>>()
        .replication
        .frame_hash_audit()
        .local(1)
        .unwrap();
    // Publish the corrected frame-one hash from a real authoritative history.
    let mut authority = World::new();
    let e = authority.spawn(Counter(0));
    authority.register_snapshot_component_hashable::<Counter>();
    let mut history = SnapshotHistory::new(8);
    history.capture(&authority);
    authority.get_mut::<Counter>(e).unwrap().0 = 3;
    history.capture(&authority);
    server.sync_frame_hashes(&history, 0).unwrap();
    exchange_deferred(&mut world, &mut server, 24);
    assert!(matches!(
        world
            .resource::<EcsReplication<ClientStrategy>>()
            .replication
            .frame_hash_audit()
            .comparison()
            .first_divergence,
        Some(TickDivergence::Hash { tick: 1, .. })
    ));
    assert_eq!(server.frame_hash_audit().remote(1), Some(before));

    // Consume actual network authority revisions, not a manually injected log.
    reconcile_ecs_replication::<ClientStrategy>(&mut world);
    assert_eq!(world.get::<Counter>(entity).unwrap().0, 6);
    let after = world
        .resource::<EcsReplication<ClientStrategy>>()
        .replication
        .frame_hash_audit()
        .local(1)
        .unwrap();
    assert!(after.sequence > before.sequence);
    assert_ne!(after.frame.hash, before.frame.hash);
    assert_eq!(
        Some(after.frame.hash),
        world.resource::<SnapshotHistory>().state_hash(1)
    );
    assert_eq!(Some(after.frame.hash), history.state_hash(1));
    assert!(world
        .resource::<EcsReplication<ClientStrategy>>()
        .replication
        .frame_hash_audit()
        .comparison()
        .retained_window_matches());
    exchange(&mut world, &mut server, 16);
    assert_eq!(server.frame_hash_audit().remote(1), Some(after));
    assert!(server
        .frame_hash_audit()
        .comparison()
        .retained_window_matches());
    assert!(world
        .resource_mut::<EcsReplication<ClientStrategy>>()
        .take_last_error()
        .is_none());
}

#[test]
fn network_audit_handles_out_of_order_duplicate_stale_conflicting_and_expired_packets() {
    let (mut world, mut server, _) = pair(4, ReplicationConfig::for_capacity(4));
    for input in [1, 1, 1] {
        queue(&mut world, input);
    }
    drive_ecs_input::<ClientStrategy>(&mut world);
    let hash = |tick| {
        world
            .resource::<SnapshotHistory>()
            .state_hash(tick)
            .unwrap()
    };
    let (zero, one, two) = (hash(0), hash(1), hash(2));
    for update in [
        packet(2, two ^ 1, 3),
        packet(0, zero, 1),
        packet(1, one ^ 1, 2),
    ] {
        server.send_frame_hash(update).unwrap();
    }
    exchange(&mut world, &mut server, 20);
    assert!(matches!(
        world
            .resource::<EcsReplication<ClientStrategy>>()
            .replication
            .frame_hash_audit()
            .comparison()
            .first_divergence,
        Some(TickDivergence::Hash { tick: 1, .. })
    ));
    for update in [
        packet(1, one, 4),
        packet(1, one, 4),
        packet(1, one ^ 1, 2),
        packet(2, two, 5),
    ] {
        server.send_frame_hash(update).unwrap();
    }
    exchange(&mut world, &mut server, 20);
    let client = world.resource_mut::<EcsReplication<ClientStrategy>>();
    assert!(client
        .replication
        .frame_hash_audit()
        .comparison()
        .retained_window_matches());
    assert!(client.replication.take_audit_error().is_none());
    server.send_frame_hash(packet(1, one ^ 1, 4)).unwrap();
    exchange(&mut world, &mut server, 12);
    assert_eq!(
        world
            .resource_mut::<EcsReplication<ClientStrategy>>()
            .replication
            .take_audit_error(),
        Some(FrameHashAuditError::ConflictingDuplicate {
            tick: 1,
            sequence: 4
        })
    );
    server.send_frame_hash(packet(7, 0, 6)).unwrap();
    exchange(&mut world, &mut server, 12);
    assert_eq!(
        world
            .resource_mut::<EcsReplication<ClientStrategy>>()
            .replication
            .take_audit_error(),
        Some(FrameHashAuditError::OutsideWindow {
            tick: 7,
            end_exclusive: 7
        })
    );

    // Confirm prior predictions before moving beyond the four-frame window.
    reconcile_ecs_replication::<ClientStrategy>(&mut world);
    for _ in 0..3 {
        queue(&mut world, 1);
        drive_ecs_input::<ClientStrategy>(&mut world);
        exchange(&mut world, &mut server, 12);
        reconcile_ecs_replication::<ClientStrategy>(&mut world);
    }
    server.send_frame_hash(packet(0, zero, 10)).unwrap();
    exchange(&mut world, &mut server, 12);
    assert!(matches!(
        world
            .resource_mut::<EcsReplication<ClientStrategy>>()
            .replication
            .take_audit_error(),
        Some(FrameHashAuditError::Expired { tick: 0, .. })
    ));
    assert!(
        world
            .resource::<EcsReplication<ClientStrategy>>()
            .replication
            .frame_hash_audit()
            .local_log()
            .len()
            <= 4
    );
}

#[test]
fn audit_send_backpressure_keeps_and_retries_captured_frames() {
    let mut config = ReplicationConfig::for_capacity(8);
    config.mux.reliable_queue = 1;
    config.mux.reliable_window = 1;
    let (mut world, mut server, _) = pair(8, config);
    let client = world.resource_mut::<EcsReplication<ClientStrategy>>();
    client.replication.send_event(b"occupy queue").unwrap();
    queue(&mut world, 1);
    drive_ecs_input::<ClientStrategy>(&mut world);
    let client = world.resource_mut::<EcsReplication<ClientStrategy>>();
    assert_eq!(client.replication.frame_hash_audit().pending_len(), 1);
    assert_eq!(
        client.replication.take_send_error(),
        Some(MuxSendError::Backpressured(NetChannel::EventsReliable))
    );
    // Drain the unrelated event; otherwise bounded business-event storage
    // intentionally backpressures the shared ordered reliable lane.
    for _ in 0..24 {
        exchange(&mut world, &mut server, 1);
        server.recv_event();
    }
    let client = world.resource::<EcsReplication<ClientStrategy>>();
    assert_eq!(client.replication.frame_hash_audit().pending_len(), 0);
    assert_eq!(
        server.frame_hash_audit().remote(0),
        client.replication.frame_hash_audit().local(0)
    );
}

#[test]
fn failed_authority_replay_discards_abandoned_audit_frames() {
    struct Unregistered;
    impl Component for Unregistered {}

    struct FailingReplay(ClientStrategy);
    impl ReplicationStrategy for FailingReplay {
        type Input = i64;

        fn simulate(&self, record: &mut Record, input: &i64) {
            self.0.simulate(record, input);
        }

        fn encode(&self, input: &i64) -> Vec<u8> {
            self.0.encode(input)
        }
    }
    impl EcsReplicationStrategy for FailingReplay {
        fn apply_input(&self, world: &mut World, input: &i64) {
            self.0.apply_input(world, input);
        }

        fn apply_authority(&self, world: &mut World, record: &Record) {
            self.0.apply_authority(world, record);
            // Force a fallible capture during replay, after truncation.
            world.spawn(Unregistered);
        }
    }

    let (a, b) = InMemoryTransport::pair_secure();
    let mut world = World::new();
    let entity = world.spawn(Counter(0));
    world.register_snapshot_component_hashable::<Counter>();
    world.insert_resource(SnapshotHistory::new(8));
    world.insert_resource(
        EcsReplication::new(
            schema(),
            8,
            &seed(),
            FailingReplay(ClientStrategy { entity }),
            Box::new(a),
        )
        .unwrap(),
    );
    let mut server =
        ServerReplication::new(schema(), 8, &seed(), AuthorityStrategy, Box::new(b)).unwrap();
    for _ in 0..12 {
        world
            .resource_mut::<EcsReplication<FailingReplay>>()
            .replication
            .pump();
        server.pump();
    }
    {
        let bridge = world.resource_mut::<EcsReplication<FailingReplay>>();
        bridge.queue_input(5);
        bridge.queue_input(1);
    }
    drive_ecs_input::<FailingReplay>(&mut world);
    for _ in 0..16 {
        world
            .resource_mut::<EcsReplication<FailingReplay>>()
            .replication
            .pump_deferred();
        server.pump();
    }
    assert!(server.frame_hash_audit().remote(1).is_some());
    reconcile_ecs_replication::<FailingReplay>(&mut world);
    // A rejected replay is transactional: the predicted world, history cursor,
    // retained future and audit branch remain unchanged so the caller can
    // retry or request a resync without a split-brain timeline.
    assert_eq!(world.resource::<SnapshotHistory>().next_frame(), 2);
    assert!(world.resource::<SnapshotHistory>().contains(1));
    let bridge = world.resource_mut::<EcsReplication<FailingReplay>>();
    assert!(matches!(
        bridge.take_last_error(),
        Some(EcsReplicationError::Rollback(
            EcsRollbackError::MissingSnapshotComponents(_)
        ))
    ));
    assert!(bridge.replication.frame_hash_audit().local(1).is_some());
    assert_eq!(bridge.replication.frame_hash_audit().local_log().len(), 2);
    assert_eq!(bridge.replication.frame_hash_audit().pending_len(), 0);
    assert!(!bridge
        .replication
        .frame_hash_audit()
        .comparison()
        .retained_window_matches());
}

#[test]
fn feature_wire_contract_rejects_a_non_auditing_peer() {
    let (a, b) = InMemoryTransport::pair_secure();
    let config = ReplicationConfig::for_capacity(8);
    assert_eq!(config.protocol.version, 2);
    let mut legacy = config;
    legacy.protocol.version = 1;
    let entity = Entity::from_bits(1u64 << 32).unwrap();
    let mut client = Replication::new_with_config(
        schema(),
        8,
        &seed(),
        ClientStrategy { entity },
        Box::new(a),
        legacy,
    )
    .unwrap();
    let mut server = ServerReplication::new_with_config(
        schema(),
        8,
        &seed(),
        AuthorityStrategy,
        Box::new(b),
        config,
    )
    .unwrap();
    for _ in 0..16 {
        client.pump();
        server.pump();
    }
    assert!(matches!(
        client.connection_state(),
        ConnectionState::Rejected(ProtocolError::VersionMismatch { .. })
    ));
    assert!(matches!(
        server.connection_state(),
        ConnectionState::Rejected(ProtocolError::VersionMismatch { .. })
    ));
}
