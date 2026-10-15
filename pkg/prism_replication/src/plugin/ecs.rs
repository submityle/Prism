//! ECS-owned client replication integration.
//!
//! [`ReplicationPlugin`] is useful for record-only consumers. This module adds
//! the full app path: local inputs are committed to the main World snapshot
//! history, predicted into the wire session, applied to ECS in `FixedFirst`,
//! and authoritative records are reconciled before the next fixed loop.

use alloc::boxed::Box;
use alloc::collections::VecDeque;
use alloc::vec::Vec;
use std::sync::Mutex;

use prism_app::{
    BeforeFixedMainLoop, EcsRollbackError, First, FixedFirst, Plugin, SnapshotHistory,
};
use prism_ecs::resource::Resource;
use prism_ecs::world::World;

use crate::delta::Schema;
use crate::multiplex::MuxSendError;
use crate::session::SessionError;
use crate::transport::Transport;

use super::{EcsReplicationStrategy, Replication, ReplicationConfig};

/// Errors returned while an ECS bridge tries to commit one deferred authority
/// delta. The network record is restored by the caller when either branch
/// fails, so retry/resync decisions observe one coherent state.
#[derive(Clone, Debug)]
pub(crate) enum DeferredEcsError {
    /// The record session could not decode or reconcile its delta.
    Session(SessionError),
    /// The World rollback/replay transaction was refused.
    Rollback(EcsRollbackError),
}

/// Single-owner fixed-tick rollback coordinator for an ECS World.
///
/// The World owns the snapshot ring through its [`SnapshotHistory`] resource;
/// this driver owns only the input timeline. Reconciliation first runs a
/// side-effecting-free validation pass against a transaction snapshot. Only
/// after every replay tick can be snapshotted does it truncate/rebase the real
/// history and commit the replay. Consequently a failed authority correction
/// leaves the World, history cursor, and input timeline unchanged.
pub struct NetworkRollbackDriver<I> {
    inputs: Vec<I>,
}

impl<I: Clone> NetworkRollbackDriver<I> {
    /// Create an empty fixed-tick coordinator.
    #[must_use]
    pub const fn new() -> Self {
        Self { inputs: Vec::new() }
    }

    /// Inputs committed at each local pre-input frame.
    #[must_use]
    pub fn inputs(&self) -> &[I] {
        &self.inputs
    }

    /// Capture one pre-input World frame and append its input.
    pub fn commit(&mut self, world: &mut World, input: I) -> Result<u64, EcsRollbackError> {
        let mut history = world
            .remove_resource::<SnapshotHistory>()
            .ok_or(EcsRollbackError::MissingHistory)?;
        let frame = match history.try_capture(world) {
            Ok(frame) => frame,
            Err(missing) => {
                world.insert_resource(history);
                return Err(EcsRollbackError::MissingSnapshotComponents(missing));
            }
        };
        world.insert_resource(history);
        self.store_input(frame, input);
        Ok(frame)
    }

    /// Rebase the World on one authoritative post-input record and replay the
    /// still-pending local inputs through the complete fixed tick.
    pub fn rebase_authority(
        &mut self,
        world: &mut World,
        authority_frame: u64,
        mut apply_authority: impl FnMut(&mut World),
        mut replay_fixed_tick: impl FnMut(&mut World, &I),
    ) -> Result<u64, EcsRollbackError> {
        if authority_frame == 0 {
            return Err(EcsRollbackError::InvalidAuthorityRevision);
        }
        let Some(mut history) = world.remove_resource::<SnapshotHistory>() else {
            return Err(EcsRollbackError::MissingHistory);
        };
        let current = history.next_frame();
        let anchor = authority_frame - 1;
        if authority_frame > current || !history.contains(anchor) {
            world.insert_resource(history);
            return Err(EcsRollbackError::UnavailableFrame {
                frame: authority_frame,
                current,
            });
        }
        let Some(replay_slice) = self.inputs.get(authority_frame as usize..current as usize) else {
            world.insert_resource(history);
            return Err(EcsRollbackError::UnavailableFrame {
                frame: authority_frame,
                current,
            });
        };
        let replay_inputs: Vec<I> = replay_slice.to_vec();

        // This snapshot is the transaction checkpoint. Registered gameplay
        // resources are included by World::snapshot; infrastructure resources
        // intentionally remain outside rollback ownership.
        let before = match world.try_snapshot() {
            Ok(snapshot) => snapshot,
            Err(missing) => {
                world.insert_resource(history);
                return Err(EcsRollbackError::MissingSnapshotComponents(missing));
            }
        };

        // Validation pass: use the existing retained anchor, but do not mutate
        // the real history ring. Every fixed tick must still be snapshottable.
        if !history.rollback(world, anchor) {
            world.insert_resource(history);
            return Err(EcsRollbackError::UnavailableFrame {
                frame: authority_frame,
                current,
            });
        }
        apply_authority(world);
        if let Err(missing) = world.try_snapshot() {
            world.restore(&before);
            world.insert_resource(history);
            return Err(EcsRollbackError::MissingSnapshotComponents(missing));
        }
        for input in &replay_inputs {
            replay_fixed_tick(world, input);
            if let Err(missing) = world.try_snapshot() {
                world.restore(&before);
                world.insert_resource(history);
                return Err(EcsRollbackError::MissingSnapshotComponents(missing));
            }
        }
        world.restore(&before);

        // Commit the already-validated branch. A capture cannot fail here
        // unless user code violates the snapshot contract between the two
        // passes; retain a fallible path rather than panicking in the driver.
        if !history.rollback(world, anchor) {
            world.insert_resource(history);
            return Err(EcsRollbackError::UnavailableFrame {
                frame: authority_frame,
                current,
            });
        }
        apply_authority(world);
        history.truncate_after(anchor);
        history.seek(authority_frame);
        world.insert_resource(history);

        for input in &replay_inputs {
            let mut history = world
                .remove_resource::<SnapshotHistory>()
                .ok_or(EcsRollbackError::MissingHistory)?;
            let capture = history.try_capture(world);
            world.insert_resource(history);
            if let Err(missing) = capture {
                // The validation pass makes this unreachable for deterministic
                // replay, but restoring the checkpoint keeps the failure
                // contract explicit if the World changed between passes.
                world.restore(&before);
                return Err(EcsRollbackError::MissingSnapshotComponents(missing));
            }
            replay_fixed_tick(world, input);
        }
        Ok(authority_frame + replay_inputs.len() as u64)
    }

    fn store_input(&mut self, frame: u64, input: I) {
        let index = frame as usize;
        if index < self.inputs.len() {
            self.inputs[index] = input;
        } else {
            self.inputs.push(input);
        }
    }
}

impl<I: Clone> Default for NetworkRollbackDriver<I> {
    fn default() -> Self {
        Self::new()
    }
}

/// Errors observed while driving the ECS/network bridge.
#[derive(Clone, Debug)]
pub enum EcsReplicationError {
    /// The ECS snapshot/input timeline could not advance.
    Rollback(EcsRollbackError),
    /// The record prediction could not advance.
    Prediction(SessionError),
    /// The already-predicted command could not be queued for transmission.
    Send(MuxSendError),
    /// A hash packet or retained audit window could not be reconciled.
    #[cfg(feature = "determinism")]
    Audit(super::FrameHashAuditError),
}

/// An ECS-backed client replication resource.
pub struct EcsReplication<S: EcsReplicationStrategy> {
    replication: Replication<S>,
    rollback: NetworkRollbackDriver<S::Input>,
    inputs: VecDeque<S::Input>,
    last_error: Option<EcsReplicationError>,
}

impl<S: EcsReplicationStrategy> Resource for EcsReplication<S> {}

impl<S: EcsReplicationStrategy> EcsReplication<S> {
    /// Build a bridge. The caller must install a World `SnapshotHistory` before
    /// queuing the first input; [`EcsReplicationPlugin`] does that automatically.
    pub fn new(
        schema: Schema,
        capacity: usize,
        snapshot: &[u8],
        strategy: S,
        transport: Box<dyn Transport>,
    ) -> Result<Self, SessionError> {
        Ok(Self {
            replication: Replication::new(schema, capacity, snapshot, strategy, transport)?,
            rollback: NetworkRollbackDriver::new(),
            inputs: VecDeque::new(),
            last_error: None,
        })
    }

    /// Override multiplexing and handshake limits.
    pub fn new_with_config(
        schema: Schema,
        capacity: usize,
        snapshot: &[u8],
        strategy: S,
        transport: Box<dyn Transport>,
        config: ReplicationConfig,
    ) -> Result<Self, SessionError> {
        Ok(Self {
            replication: Replication::new_with_config(
                schema, capacity, snapshot, strategy, transport, config,
            )?,
            rollback: NetworkRollbackDriver::new(),
            inputs: VecDeque::new(),
            last_error: None,
        })
    }

    /// Queue one local input for the next fixed simulation step.
    pub fn queue_input(&mut self, input: S::Input) {
        self.inputs.push_back(input);
    }

    /// Number of local inputs waiting for a fixed step.
    #[must_use]
    pub fn queued_inputs(&self) -> usize {
        self.inputs.len()
    }

    /// Read the underlying record/network session.
    #[must_use]
    pub fn replication(&self) -> &Replication<S> {
        &self.replication
    }

    /// Mutate the underlying record/network session for channel-specific app
    /// traffic.
    pub fn replication_mut(&mut self) -> &mut Replication<S> {
        &mut self.replication
    }

    /// Read the ECS input timeline.
    #[must_use]
    pub fn rollback(&self) -> &NetworkRollbackDriver<S::Input> {
        &self.rollback
    }

    /// Take the first bridge error observed by the frame driver.
    pub fn take_last_error(&mut self) -> Option<EcsReplicationError> {
        self.last_error.take()
    }

    fn record_error(&mut self, error: EcsReplicationError) {
        if self.last_error.is_none() {
            self.last_error = Some(error);
        }
    }

    #[cfg(feature = "determinism")]
    fn sync_audit(&mut self, world: &World, from: u64) {
        if let Some(history) = world.get_resource::<SnapshotHistory>() {
            if let Err(error) = self.replication.sync_frame_hashes(history, from) {
                self.record_error(EcsReplicationError::Audit(error));
            }
        } else {
            self.record_error(EcsReplicationError::Rollback(
                EcsRollbackError::MissingHistory,
            ));
        }
    }
}

/// A plugin that installs [`EcsReplication`] and drives it at the fixed-step
/// boundaries of a [`prism_app::App`].
pub struct EcsReplicationPlugin<S: EcsReplicationStrategy> {
    schema: Schema,
    capacity: usize,
    snapshot: Vec<u8>,
    strategy: Mutex<Option<S>>,
    transport: Mutex<Option<Box<dyn Transport>>>,
    config: ReplicationConfig,
}

impl<S: EcsReplicationStrategy> EcsReplicationPlugin<S> {
    /// Construct a bridge from a seed snapshot and transport.
    #[must_use]
    pub fn new(
        schema: Schema,
        capacity: usize,
        snapshot: Vec<u8>,
        strategy: S,
        transport: Box<dyn Transport>,
    ) -> Self {
        Self {
            schema,
            capacity,
            snapshot,
            strategy: Mutex::new(Some(strategy)),
            transport: Mutex::new(Some(transport)),
            config: ReplicationConfig::for_capacity(capacity),
        }
    }

    /// Override multiplexing and handshake limits.
    #[must_use]
    pub fn with_config(mut self, config: ReplicationConfig) -> Self {
        self.config = config;
        self
    }
}

impl<S: EcsReplicationStrategy> Plugin for EcsReplicationPlugin<S> {
    fn build(&self, app: &mut prism_app::App) {
        let strategy = self
            .strategy
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
            .expect("EcsReplicationPlugin::build ran twice");
        let transport = self
            .transport
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
            .expect("EcsReplicationPlugin::build ran twice");
        let bridge = EcsReplication::new_with_config(
            self.schema.clone(),
            self.capacity,
            &self.snapshot,
            strategy,
            transport,
            self.config,
        )
        .expect("EcsReplicationPlugin seed snapshot is malformed");
        app.init_snapshot_history(self.capacity);
        app.insert_resource(bridge);
        app.add_systems(First, pump_ecs_replication::<S>);
        app.add_systems(BeforeFixedMainLoop, reconcile_ecs_replication::<S>);
        app.add_systems(FixedFirst, drive_ecs_input::<S>);
    }
}

fn pump_ecs_replication<S: EcsReplicationStrategy>(
    mut replication: prism_ecs::system::ResMut<EcsReplication<S>>,
) {
    replication.replication.pump_deferred();
    #[cfg(feature = "determinism")]
    if let Some(error) = replication.replication.take_audit_error() {
        replication.record_error(EcsReplicationError::Audit(error));
    }
}

fn reconcile_ecs_replication<S: EcsReplicationStrategy>(world: &mut World) {
    let Some(mut bridge) = world.remove_resource::<EcsReplication<S>>() else {
        return;
    };
    loop {
        let result = bridge
            .replication
            .reconcile_deferred_ecs(world, &mut bridge.rollback);
        let applied = match result {
            Ok(applied) => applied,
            Err(DeferredEcsError::Session(error)) => {
                bridge.record_error(EcsReplicationError::Prediction(error));
                break;
            }
            Err(DeferredEcsError::Rollback(error)) => {
                bridge.record_error(EcsReplicationError::Rollback(error));
                break;
            }
        };
        if !applied {
            break;
        }
        // Reconcile even after partial replay failure: the abandoned tail must
        // not keep advertising old hashes. The pre-input anchor is unchanged.
        #[cfg(feature = "determinism")]
        bridge.sync_audit(world, bridge.replication.acked() as u64);
    }
    #[cfg(feature = "determinism")]
    bridge.replication.flush_frame_hashes();
    world.insert_resource(bridge);
}

fn drive_ecs_input<S: EcsReplicationStrategy>(world: &mut World) {
    let Some(mut bridge) = world.remove_resource::<EcsReplication<S>>() else {
        return;
    };
    while let Some(input) = bridge.inputs.pop_front() {
        if let Err(error) = bridge.replication.check_prediction_capacity() {
            bridge.record_error(EcsReplicationError::Prediction(error));
            bridge.inputs.push_front(input);
            break;
        }
        if let Err(missing) = world.try_snapshot() {
            bridge.record_error(EcsReplicationError::Rollback(
                EcsRollbackError::MissingSnapshotComponents(missing),
            ));
            bridge.inputs.push_front(input);
            break;
        }
        if world.get_resource::<SnapshotHistory>().is_none() {
            bridge.record_error(EcsReplicationError::Rollback(
                EcsRollbackError::MissingHistory,
            ));
            bridge.inputs.push_front(input);
            break;
        }
        match bridge.replication.predict_and_send(input.clone()) {
            Ok(Some(_)) => {}
            Ok(None) => {
                if let Some(error) = bridge.replication.take_send_error() {
                    bridge.record_error(EcsReplicationError::Send(error));
                }
                bridge.inputs.push_front(input);
                break;
            }
            Err(error) => {
                bridge.record_error(EcsReplicationError::Prediction(error));
                bridge.inputs.push_front(input);
                break;
            }
        }
        let _frame = match bridge.rollback.commit(world, input.clone()) {
            Ok(frame) => frame,
            Err(error) => {
                bridge.record_error(EcsReplicationError::Rollback(error));
                bridge.inputs.push_front(input);
                break;
            }
        };
        #[cfg(feature = "determinism")]
        bridge.sync_audit(world, _frame);
        bridge.replication.apply_input_to_world(world, &input);
    }
    #[cfg(feature = "determinism")]
    bridge.replication.flush_frame_hashes();
    world.insert_resource(bridge);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::delta::{encode_snapshot, FieldKind, FieldValue, Record};
    use crate::plugin::{ReplicationServerPlugin, ReplicationStrategy, ServerStrategy};
    use crate::transport::memory::InMemoryTransport;
    use prism_app::{
        App, FixedFirst, FixedLast, FixedPostUpdate, FixedPreUpdate, FixedUpdate, SnapshotHistory,
        TimeUpdateStrategy,
    };
    use prism_ecs::component::Component;
    use prism_ecs::entity::Entity;
    use prism_ecs::system::ResMut;

    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    struct Counter(i64);

    impl Component for Counter {}

    fn schema() -> Schema {
        Schema::new(vec![FieldKind::Int { bits: 12 }])
    }

    fn seed() -> Vec<u8> {
        let mut bytes = Vec::new();
        encode_snapshot(
            &schema(),
            0,
            &Record::new(vec![FieldValue::Int(0)]),
            &mut bytes,
        )
        .unwrap();
        bytes
    }

    fn record_value(record: &Record) -> i64 {
        let FieldValue::Int(value) = record.value(0).expect("record has one field") else {
            panic!("record field is not an integer");
        };
        value
    }

    struct ClientStrategy {
        entity: Entity,
    }

    impl ReplicationStrategy for ClientStrategy {
        type Input = i64;

        fn simulate(&self, record: &mut Record, input: &Self::Input) {
            record.set(
                0,
                FieldValue::Int(record_value(record).saturating_add(*input)),
            );
        }

        fn encode(&self, input: &Self::Input) -> Vec<u8> {
            input.to_le_bytes().to_vec()
        }
    }

    impl EcsReplicationStrategy for ClientStrategy {
        fn apply_input(&self, world: &mut World, input: &Self::Input) {
            let mut counter = world
                .get_mut::<Counter>(self.entity)
                .expect("client entity exists");
            counter.0 = counter.0.saturating_add(*input);
        }

        fn apply_authority(&self, world: &mut World, state: &Record) {
            world
                .get_mut::<Counter>(self.entity)
                .expect("client entity exists")
                .0 = record_value(state);
        }
    }

    struct AuthorityStrategy;

    impl ServerStrategy for AuthorityStrategy {
        type Command = i64;

        fn decode(&self, bytes: &[u8]) -> Option<Self::Command> {
            <[u8; 8]>::try_from(bytes).ok().map(i64::from_le_bytes)
        }

        fn apply(&self, record: &mut Record, _command: Self::Command) {
            // Deliberately disagree with the client's prediction: the authority
            // advances by three for every accepted command.
            record.set(0, FieldValue::Int(record_value(record).saturating_add(3)));
        }
    }

    #[cfg(feature = "determinism")]
    mod audit_tests;

    #[test]
    fn ecs_bridge_reconciles_authority_and_replays_pending_inputs() {
        let (client_transport, server_transport) = InMemoryTransport::pair_secure();
        let mut client_app = App::new();
        client_app.set_time_update_strategy(TimeUpdateStrategy::ManualDelta(
            std::time::Duration::from_millis(20),
        ));
        client_app.set_fixed_timestep_hz(60.0);
        let entity = client_app.world_mut().spawn(Counter(0));
        client_app
            .world_mut()
            .register_snapshot_component::<Counter>();
        client_app.add_plugins(EcsReplicationPlugin::new(
            schema(),
            8,
            seed(),
            ClientStrategy { entity },
            Box::new(client_transport),
        ));

        let mut server_app = App::new();
        server_app.add_plugins(ReplicationServerPlugin::new(
            schema(),
            8,
            seed(),
            AuthorityStrategy,
            Box::new(server_transport),
        ));

        for _ in 0..8 {
            client_app.update();
            server_app.update();
        }
        assert!(client_app
            .world()
            .resource::<EcsReplication<ClientStrategy>>()
            .replication()
            .negotiated()
            .is_some());

        {
            let bridge = client_app
                .world_mut()
                .resource_mut::<EcsReplication<ClientStrategy>>();
            bridge.queue_input(5);
            bridge.queue_input(1);
        }

        for _ in 0..64 {
            client_app.update();
            // Drive one fixed step explicitly so the test is independent of
            // wall-clock accumulator policy while still exercising the plugin's
            // `BeforeFixedMainLoop` and `FixedFirst` systems.
            client_app.world_mut().run_schedule(FixedFirst);
            server_app.update();
        }

        let (queued, acked, predicted, error) = {
            let bridge = client_app
                .world_mut()
                .resource_mut::<EcsReplication<ClientStrategy>>();
            (
                bridge.queued_inputs(),
                bridge.replication().acked(),
                bridge.replication().predicted(),
                bridge.take_last_error(),
            )
        };
        assert_eq!(queued, 0);
        assert!(error.is_none());
        assert_eq!(acked, 2);
        assert_eq!(predicted, 2);
        assert_eq!(
            client_app
                .world()
                .get::<Counter>(entity)
                .expect("client entity exists")
                .0,
            6,
            "authority result (3 + 3) survives rollback and replay"
        );

        let server = server_app
            .world()
            .resource::<crate::plugin::ServerReplication<AuthorityStrategy>>();
        assert_eq!(server.revision(), Some(2));
        assert_eq!(record_value(server.record().expect("authority record")), 6);
    }

    #[test]
    fn network_driver_replays_the_complete_fixed_main_sequence() {
        #[derive(Clone, Debug, PartialEq, Eq, Hash)]
        struct FixedTicks(u32);
        impl Resource for FixedTicks {}

        let mut app = App::new();
        app.add_systems(FixedUpdate, |mut ticks: ResMut<FixedTicks>| {
            ticks.0 += 1;
        });
        let world = app.world_mut();
        let entity = world.spawn(Counter(0));
        world.register_snapshot_component_hashable::<Counter>();
        world.register_snapshot_resource::<FixedTicks>();
        world.insert_resource(FixedTicks(0));
        world.insert_resource(SnapshotHistory::new(8));

        let mut driver = NetworkRollbackDriver::new();
        driver.commit(world, 5).unwrap();
        world.get_mut::<Counter>(entity).unwrap().0 += 5;
        driver.commit(world, 1).unwrap();
        world.get_mut::<Counter>(entity).unwrap().0 += 1;

        let result = driver
            .rebase_authority(
                world,
                1,
                |world| world.get_mut::<Counter>(entity).unwrap().0 = 3,
                |world, input| {
                    world.get_mut::<Counter>(entity).unwrap().0 += *input;
                    world.run_schedule(FixedFirst);
                    world.run_schedule(FixedPreUpdate);
                    world.run_schedule(FixedUpdate);
                    world.run_schedule(FixedPostUpdate);
                    world.run_schedule(FixedLast);
                },
            )
            .unwrap();

        assert_eq!(result, 2);
        // Revision 1 already contains the authority result for input 0;
        // only the still-pending input 1 is replayed here.
        assert_eq!(world.get::<Counter>(entity).unwrap().0, 4);
        assert_eq!(world.resource::<FixedTicks>().0, 1);
        assert_eq!(world.resource::<SnapshotHistory>().next_frame(), 2);
        assert_eq!(driver.inputs(), &[5, 1]);
    }

    #[test]
    fn failed_network_replay_restores_world_history_and_input_timeline() {
        struct Unregistered;
        impl Component for Unregistered {}

        let mut world = World::new();
        let entity = world.spawn(Counter(0));
        world.register_snapshot_component_hashable::<Counter>();
        world.insert_resource(SnapshotHistory::new(8));
        let mut driver = NetworkRollbackDriver::new();
        driver.commit(&mut world, 5).unwrap();
        world.get_mut::<Counter>(entity).unwrap().0 += 5;
        driver.commit(&mut world, 1).unwrap();
        world.get_mut::<Counter>(entity).unwrap().0 += 1;
        let before_hashes = (
            world.resource::<SnapshotHistory>().state_hash(0),
            world.resource::<SnapshotHistory>().state_hash(1),
            world.resource::<SnapshotHistory>().next_frame(),
        );

        let spawned = std::cell::Cell::new(None);
        let error = driver
            .rebase_authority(
                &mut world,
                1,
                |world| world.get_mut::<Counter>(entity).unwrap().0 = 3,
                |world, input| {
                    world.get_mut::<Counter>(entity).unwrap().0 += *input;
                    spawned.set(Some(world.spawn(Unregistered)));
                },
            )
            .expect_err("an unregistered replay component must refuse the transaction");

        assert!(matches!(
            error,
            EcsRollbackError::MissingSnapshotComponents(_)
        ));
        assert_eq!(world.get::<Counter>(entity).unwrap().0, 6);
        assert_eq!(
            (
                world.resource::<SnapshotHistory>().state_hash(0),
                world.resource::<SnapshotHistory>().state_hash(1),
                world.resource::<SnapshotHistory>().next_frame(),
            ),
            before_hashes
        );
        assert_eq!(driver.inputs(), &[5, 1]);
        assert!(spawned
            .get()
            .is_some_and(|entity| world.get::<Unregistered>(entity).is_none()));
    }
}
