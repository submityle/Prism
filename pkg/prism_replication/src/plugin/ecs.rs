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

use prism_app::rollback::SnapshotCaptureUndo;
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
/// this driver owns only a bounded input window. Reconciliation executes each
/// callback once against a replacement history branch. A failed capture restores
/// registered World state and the original history; external callback effects
/// are not undoable and must be suppressed or journaled by the application.
pub struct NetworkRollbackDriver<I> {
    inputs: Vec<I>,
    input_start: u64,
    pending: Option<PendingCommit<I>>,
}

struct PendingCommit<I> {
    frame: u64,
    undo: SnapshotCaptureUndo,
    input_start: u64,
    discarded: Vec<I>,
}

impl<I: Clone> NetworkRollbackDriver<I> {
    /// Create an empty fixed-tick coordinator.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            inputs: Vec::new(),
            input_start: 0,
            pending: None,
        }
    }

    /// Inputs committed at each local pre-input frame.
    #[must_use]
    pub fn inputs(&self) -> &[I] {
        &self.inputs
    }

    /// Absolute pre-input frame represented by the first retained input.
    #[must_use]
    pub fn input_start(&self) -> u64 {
        self.input_start
    }

    /// Capture one pre-input World frame and append its input.
    pub fn commit(&mut self, world: &mut World, input: I) -> Result<u64, EcsRollbackError> {
        let mut history = world
            .remove_resource::<SnapshotHistory>()
            .ok_or(EcsRollbackError::MissingHistory)?;
        let current = history.next_frame();
        if current != self.input_start + self.inputs.len() as u64 {
            world.insert_resource(history);
            return Err(EcsRollbackError::UnavailableFrame {
                frame: current,
                current,
            });
        }
        let capture = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            history.try_capture_reversible(world)
        }));
        let (frame, undo) = match capture {
            Ok(Ok(captured)) => captured,
            Ok(Err(missing)) => {
                world.insert_resource(history);
                return Err(EcsRollbackError::MissingSnapshotComponents(missing));
            }
            Err(payload) => {
                world.insert_resource(history);
                std::panic::resume_unwind(payload);
            }
        };
        let input_start = self.input_start;
        let oldest = history.oldest_frame().unwrap_or(frame);
        let discard = (oldest - self.input_start) as usize;
        let discarded = self.inputs.drain(..discard).collect();
        self.input_start = oldest;
        self.inputs.push(input);
        self.pending = Some(PendingCommit {
            frame,
            undo,
            input_start,
            discarded,
        });
        world.insert_resource(history);
        Ok(frame)
    }

    /// Undo the most recent [`commit`](Self::commit), restoring its pre-input
    /// snapshot and rewinding the history cursor. This is used to keep the ECS
    /// timeline transactional if a later prediction/send step unexpectedly
    /// refuses the input.
    pub fn abort_commit(&mut self, world: &mut World, frame: u64) -> bool {
        if self
            .pending
            .as_ref()
            .is_none_or(|pending| pending.frame != frame)
        {
            return false;
        }
        let Some(mut history) = world.remove_resource::<SnapshotHistory>() else {
            return false;
        };
        let Some(before) = history.snapshot(frame) else {
            world.insert_resource(history);
            return false;
        };
        let pending = self.pending.take().expect("pending commit checked above");
        let valid = history.undo_capture(pending.undo);
        if valid {
            world.restore(&before);
            self.inputs.pop();
            self.inputs.splice(..0, pending.discarded);
            self.input_start = pending.input_start;
        }
        world.insert_resource(history);
        valid
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
        let Some(history) = world.remove_resource::<SnapshotHistory>() else {
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
        let replay_slice = authority_frame
            .checked_sub(self.input_start)
            .and_then(|start| usize::try_from(start).ok())
            .zip(
                current
                    .checked_sub(self.input_start)
                    .and_then(|end| usize::try_from(end).ok()),
            )
            .and_then(|(start, end)| self.inputs.get(start..end));
        let Some(replay_slice) = replay_slice else {
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

        // Keep the original ring intact until all callbacks and captures finish.
        let mut branch = history.branch_through(anchor);
        branch.seek(authority_frame);
        if !history.rollback(world, anchor) {
            world.insert_resource(history);
            return Err(EcsRollbackError::UnavailableFrame {
                frame: authority_frame,
                current,
            });
        }
        world.insert_resource(branch);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            apply_authority(world);
            for input in &replay_inputs {
                let mut branch = world
                    .remove_resource::<SnapshotHistory>()
                    .ok_or(EcsRollbackError::MissingHistory)?;
                let capture = branch.try_capture(world);
                world.insert_resource(branch);
                capture.map_err(EcsRollbackError::MissingSnapshotComponents)?;
                replay_fixed_tick(world, input);
            }
            // The last replay does not have a following pre-input capture.
            world
                .try_snapshot()
                .map_err(EcsRollbackError::MissingSnapshotComponents)?;
            if world.get_resource::<SnapshotHistory>().is_none() {
                return Err(EcsRollbackError::MissingHistory);
            }
            Ok(())
        }));
        match result {
            Ok(Ok(())) => {
                self.pending = None;
                Ok(current)
            }
            Ok(Err(error)) => {
                world.restore(&before);
                world.insert_resource(history);
                Err(error)
            }
            Err(payload) => {
                world.restore(&before);
                world.insert_resource(history);
                std::panic::resume_unwind(payload)
            }
        }
    }

    /// Atomically install a full authority baseline, abandoning old predictions.
    /// The next local input is captured at `frame`, including nonzero resyncs.
    pub fn reset(
        &mut self,
        world: &mut World,
        frame: u64,
        mut apply_authority: impl FnMut(&mut World),
    ) -> Result<(), EcsRollbackError> {
        let history = world
            .remove_resource::<SnapshotHistory>()
            .ok_or(EcsRollbackError::MissingHistory)?;
        let before = match world.try_snapshot() {
            Ok(snapshot) => snapshot,
            Err(missing) => {
                world.insert_resource(history);
                return Err(EcsRollbackError::MissingSnapshotComponents(missing));
            }
        };
        let mut branch = SnapshotHistory::new(history.capacity());
        branch.seek(frame);
        world.insert_resource(branch);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            apply_authority(world);
            world
                .try_snapshot()
                .map_err(EcsRollbackError::MissingSnapshotComponents)?;
            if world.get_resource::<SnapshotHistory>().is_none() {
                return Err(EcsRollbackError::MissingHistory);
            }
            Ok(())
        }));
        match result {
            Ok(Ok(())) => {
                self.inputs.clear();
                self.input_start = frame;
                self.pending = None;
                Ok(())
            }
            Ok(Err(error)) => {
                world.restore(&before);
                world.insert_resource(history);
                Err(error)
            }
            Err(payload) => {
                world.restore(&before);
                world.insert_resource(history);
                std::panic::resume_unwind(payload)
            }
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
    /// The bounded local input queue is full; the input was not admitted.
    InputQueueFull,
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
///
/// Input callbacks finish before their command is queued. A callback panic or
/// refused send restores the registered pre-input World snapshot and retains
/// the input for retry. This does not undo external effects, unregistered
/// resources, or later fixed-phase systems. Callbacks must not replace the
/// World or mutate the bridge/history infrastructure; snapshot Clone/Drop glue
/// must itself be unwind-safe.
pub struct EcsReplication<S: EcsReplicationStrategy> {
    replication: Replication<S>,
    rollback: NetworkRollbackDriver<S::Input>,
    inputs: VecDeque<S::Input>,
    input_capacity: usize,
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
            input_capacity: capacity.max(1),
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
            input_capacity: capacity.max(1),
            last_error: None,
        })
    }

    /// Queue one local input for a fixed simulation step. Returns ownership of
    /// the refused input when the rollback-sized queue is full.
    pub fn queue_input(&mut self, input: S::Input) -> Result<(), S::Input> {
        if self.inputs.len() >= self.input_capacity {
            self.record_error(EcsReplicationError::InputQueueFull);
            return Err(input);
        }
        self.inputs.push_back(input);
        Ok(())
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

/// Return the same bridge to its World before propagating a callback unwind.
/// The inner transaction owns World/history recovery; this owns the resource.
fn with_ecs_bridge<S: EcsReplicationStrategy>(
    world: &mut World,
    run: impl FnOnce(&mut World, &mut EcsReplication<S>),
) {
    let Some(mut bridge) = world.remove_resource::<EcsReplication<S>>() else {
        return;
    };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run(world, &mut bridge);
    }));
    world.insert_resource(bridge);
    if let Err(payload) = result {
        std::panic::resume_unwind(payload);
    }
}

fn reconcile_ecs_replication<S: EcsReplicationStrategy>(world: &mut World) {
    with_ecs_bridge::<S>(world, reconcile_ecs_bridge::<S>);
}

fn reconcile_ecs_bridge<S: EcsReplicationStrategy>(
    world: &mut World,
    bridge: &mut EcsReplication<S>,
) {
    loop {
        let reseed = bridge.replication.deferred_reseed_pending();
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
        if reseed {
            // These commands were queued on the old branch but never driven.
            // A rejected reset must retain them; only a committed reset owns
            // the right to discard both predicted and not-yet-driven inputs.
            bridge.inputs.clear();
        }
        // Successful deltas retain their pre-input anchor; a full snapshot has
        // already cleared the old audit branch and installed an empty history.
        #[cfg(feature = "determinism")]
        bridge.sync_audit(world, bridge.replication.acked() as u64);
    }
    #[cfg(feature = "determinism")]
    bridge.replication.flush_frame_hashes();
}

fn drive_ecs_input<S: EcsReplicationStrategy>(world: &mut World) {
    with_ecs_bridge::<S>(world, drive_ecs_bridge_input::<S>);
}

fn drive_ecs_bridge_input<S: EcsReplicationStrategy>(
    world: &mut World,
    bridge: &mut EcsReplication<S>,
) {
    // The remaining fixed phases run once after FixedFirst. Consuming several
    // inputs here would make live simulation differ from per-input replay.
    'input: {
        // Keep the admitted input owned by the queue until the entire local
        // transaction succeeds, including panicking encode/simulate callbacks.
        let Some(input) = bridge.inputs.front().cloned() else {
            break 'input;
        };
        if let Err(error) = bridge.replication.check_prediction_capacity() {
            bridge.record_error(EcsReplicationError::Prediction(error));
            break 'input;
        }
        if world.get_resource::<SnapshotHistory>().is_none() {
            bridge.record_error(EcsReplicationError::Rollback(
                EcsRollbackError::MissingHistory,
            ));
            break 'input;
        }
        if let Err(error) = bridge.replication.check_predicted_send(&input) {
            bridge.record_error(EcsReplicationError::Send(error));
            break 'input;
        }
        let mut predicted = false;
        let mut committed = None;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            bridge
                .replication
                .predict(input.clone())
                .map_err(EcsReplicationError::Prediction)?;
            predicted = true;
            let frame = bridge
                .rollback
                .commit(world, input.clone())
                .map_err(EcsReplicationError::Rollback)?;
            committed = Some(frame);
            bridge.replication.apply_input_to_world(world, &input);
            // All user encode/simulate/apply code runs before send publishes a
            // command. Reliable send only admits to the mux queue, so a refused
            // send has no remote effect and a retry cannot duplicate a command.
            bridge
                .replication
                .send_predicted(&input)
                .map_err(EcsReplicationError::Send)?;
            Ok::<_, EcsReplicationError>(frame)
        }));
        let _frame = match result {
            Ok(Ok(frame)) => frame,
            failure => {
                if predicted {
                    bridge.replication.cancel_last_prediction();
                }
                if let Some(frame) = committed {
                    assert!(
                        bridge.rollback.abort_commit(world, frame),
                        "input callback changed rollback infrastructure"
                    );
                }
                match failure {
                    Ok(Err(error)) => bridge.record_error(error),
                    Err(payload) => std::panic::resume_unwind(payload),
                    Ok(Ok(_)) => unreachable!(),
                }
                break 'input;
            }
        };
        bridge.rollback.pending = None;
        bridge.inputs.pop_front();
        #[cfg(feature = "determinism")]
        bridge.sync_audit(world, _frame);
    }
    #[cfg(feature = "determinism")]
    bridge.replication.flush_frame_hashes();
}

#[cfg(test)]
mod tests {
    use alloc::vec;

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
    #[cfg(feature = "multi_thread")]
    use prism_ecs::world::snapshot::WorldSnapshot;

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

    mod panic_tests;

    #[test]
    fn ecs_bridge_reconciles_authority_and_replays_pending_inputs() {
        let (client_transport, server_transport) = InMemoryTransport::pair_secure();
        let mut client_app = App::new();
        client_app.set_time_update_strategy(TimeUpdateStrategy::ManualDelta(
            std::time::Duration::from_millis(10),
        ));
        client_app.set_fixed_timestep(std::time::Duration::from_millis(10));
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
            bridge.queue_input(5).unwrap();
            bridge.queue_input(1).unwrap();
        }

        for _ in 0..24 {
            client_app.update();
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

    #[cfg(feature = "multi_thread")]
    struct E2eResult {
        final_snapshot: WorldSnapshot,
        history: Vec<WorldSnapshot>,
        final_hashes: Vec<Option<u64>>,
    }

    #[cfg(feature = "multi_thread")]
    fn run_e2e(worker_count: usize) -> E2eResult {
        let (client_transport, server_transport) = InMemoryTransport::pair_secure();
        let mut client_app = App::new();
        client_app.set_task_pool_threads(worker_count);
        client_app.set_time_update_strategy(TimeUpdateStrategy::ManualDelta(
            std::time::Duration::from_millis(10),
        ));
        client_app.set_fixed_timestep(std::time::Duration::from_millis(10));
        let entity = client_app.world_mut().spawn(Counter(0));
        client_app
            .world_mut()
            .register_snapshot_component_hashable::<Counter>();
        client_app
            .world_mut()
            .register_snapshot_component_equatable::<Counter>();
        client_app.add_plugins(EcsReplicationPlugin::new(
            schema(),
            8,
            seed(),
            ClientStrategy { entity },
            Box::new(client_transport),
        ));

        let mut server_app = App::new();
        server_app.set_task_pool_threads(worker_count);
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
        {
            let bridge = client_app
                .world_mut()
                .resource_mut::<EcsReplication<ClientStrategy>>();
            bridge.queue_input(5).unwrap();
            bridge.queue_input(1).unwrap();
        }
        for _ in 0..24 {
            client_app.update();
            server_app.update();
        }

        let world = client_app.world();
        let history = world.resource::<SnapshotHistory>();
        let history_snapshots = (0..history.next_frame())
            .map(|frame| history.snapshot(frame).expect("retained e2e frame"))
            .collect();
        let final_hashes = (0..history.next_frame())
            .map(|frame| history.state_hash(frame))
            .collect();
        E2eResult {
            final_snapshot: world.try_snapshot().expect("registered e2e snapshot"),
            history: history_snapshots,
            final_hashes,
        }
    }

    #[cfg(feature = "multi_thread")]
    #[test]
    fn ecs_e2e_is_structurally_equal_for_one_and_eight_workers() {
        let one = run_e2e(1);
        let eight = run_e2e(8);
        assert!(one.final_snapshot.structurally_eq(&eight.final_snapshot));
        assert_eq!(one.history.len(), eight.history.len());
        for (left, right) in one.history.iter().zip(&eight.history) {
            assert!(left.structurally_eq(right));
        }
        assert_eq!(one.final_hashes, eight.final_hashes);
    }

    #[test]
    fn failed_ecs_commit_leaves_session_history_and_input_untouched() {
        struct Unregistered;
        impl Component for Unregistered {}

        let (client_transport, _server_transport) = InMemoryTransport::pair_secure();
        let mut world = World::new();
        let entity = world.spawn(Counter(0));
        world.register_snapshot_component_hashable::<Counter>();
        world.spawn(Unregistered);
        world.insert_resource(SnapshotHistory::new(8));
        let bridge = EcsReplication::new(
            schema(),
            8,
            &seed(),
            ClientStrategy { entity },
            Box::new(client_transport),
        )
        .unwrap();
        world.insert_resource(bridge);
        world
            .resource_mut::<EcsReplication<ClientStrategy>>()
            .queue_input(5)
            .unwrap();
        drive_ecs_input::<ClientStrategy>(&mut world);

        {
            let bridge = world.resource::<EcsReplication<ClientStrategy>>();
            assert_eq!(bridge.queued_inputs(), 1);
            assert_eq!(bridge.replication().acked(), 0);
            assert_eq!(bridge.replication().predicted(), 0);
            assert_eq!(bridge.replication().unacked(), 1, "hello only");
            assert_eq!(bridge.rollback().inputs(), &[]);
        }
        assert_eq!(world.resource::<SnapshotHistory>().next_frame(), 0);
        let error = world
            .resource_mut::<EcsReplication<ClientStrategy>>()
            .take_last_error();
        assert!(matches!(
            error,
            Some(EcsReplicationError::Rollback(
                EcsRollbackError::MissingSnapshotComponents(_)
            ))
        ));
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

    #[test]
    fn network_rebase_executes_callbacks_once_even_for_external_effects() {
        let mut world = World::new();
        let entity = world.spawn(Counter(0));
        world.register_snapshot_component_hashable::<Counter>();
        world.insert_resource(SnapshotHistory::new(4));
        let mut driver = NetworkRollbackDriver::new();
        for input in [1, 2, 3] {
            driver.commit(&mut world, input).unwrap();
            world.get_mut::<Counter>(entity).unwrap().0 += input;
        }
        let mut authority_calls = 0;
        let mut replay_calls = Vec::new();
        driver
            .rebase_authority(
                &mut world,
                1,
                |world| {
                    authority_calls += 1;
                    world.get_mut::<Counter>(entity).unwrap().0 = 10;
                },
                |world, input| {
                    replay_calls.push(*input);
                    world.get_mut::<Counter>(entity).unwrap().0 += *input;
                },
            )
            .unwrap();
        assert_eq!(authority_calls, 1);
        assert_eq!(replay_calls, [2, 3]);
        assert_eq!(world.get::<Counter>(entity).unwrap().0, 15);
        assert_eq!(world.resource::<SnapshotHistory>().next_frame(), 3);
    }

    #[test]
    fn aborted_capture_recovers_evicted_anchor_and_input_at_every_capacity() {
        for capacity in [1, 2, 8] {
            let mut world = World::new();
            let entity = world.spawn(Counter(0));
            world.register_snapshot_component_hashable::<Counter>();
            world.insert_resource(SnapshotHistory::new(capacity));
            let mut driver = NetworkRollbackDriver::new();
            for input in 0..32 {
                driver.commit(&mut world, input).unwrap();
                world.get_mut::<Counter>(entity).unwrap().0 += input;
                assert!(driver.inputs().len() <= capacity);
            }
            let before_inputs = driver.inputs().to_vec();
            let before_start = driver.input_start();
            let history = world.resource::<SnapshotHistory>();
            let before_frame = history.next_frame();
            let before_oldest = history.oldest_frame();
            let hashes: Vec<_> = (before_oldest.unwrap()..before_frame)
                .map(|frame| history.state_hash(frame))
                .collect();
            let before_value = world.get::<Counter>(entity).unwrap().0;
            let frame = driver.commit(&mut world, 99).unwrap();
            world.get_mut::<Counter>(entity).unwrap().0 = 999;
            assert!(driver.abort_commit(&mut world, frame));
            assert!(!driver.abort_commit(&mut world, frame));
            let history = world.resource::<SnapshotHistory>();
            assert_eq!(history.next_frame(), before_frame);
            assert_eq!(history.oldest_frame(), before_oldest);
            assert_eq!(
                (before_oldest.unwrap()..before_frame)
                    .map(|frame| history.state_hash(frame))
                    .collect::<Vec<_>>(),
                hashes
            );
            assert_eq!(driver.input_start(), before_start);
            assert_eq!(driver.inputs(), before_inputs);
            assert_eq!(world.get::<Counter>(entity).unwrap().0, before_value);
            assert_eq!(driver.commit(&mut world, 7).unwrap(), frame);
        }
    }

    #[test]
    fn bounded_input_window_rebases_using_absolute_frames() {
        let mut world = World::new();
        let entity = world.spawn(Counter(0));
        world.register_snapshot_component_hashable::<Counter>();
        world.insert_resource(SnapshotHistory::new(3));
        let mut driver = NetworkRollbackDriver::new();
        for input in 0..100 {
            driver.commit(&mut world, input).unwrap();
        }
        assert_eq!(driver.input_start(), 97);
        assert_eq!(driver.inputs(), [97, 98, 99]);
        let mut seen = Vec::new();
        driver
            .rebase_authority(
                &mut world,
                98,
                |world| world.get_mut::<Counter>(entity).unwrap().0 = 1000,
                |world, input| {
                    seen.push(*input);
                    world.get_mut::<Counter>(entity).unwrap().0 += input;
                },
            )
            .unwrap();
        assert_eq!(seen, [98, 99]);
        assert_eq!(world.get::<Counter>(entity).unwrap().0, 1197);
        assert_eq!(driver.commit(&mut world, 100).unwrap(), 100);
        assert_eq!(driver.inputs(), [98, 99, 100]);
    }

    #[test]
    fn authority_panic_restores_owned_state_and_history_before_unwinding() {
        let mut world = World::new();
        let entity = world.spawn(Counter(0));
        world.register_snapshot_component_hashable::<Counter>();
        world.insert_resource(SnapshotHistory::new(2));
        let mut driver = NetworkRollbackDriver::new();
        driver.commit(&mut world, 1).unwrap();
        world.get_mut::<Counter>(entity).unwrap().0 = 1;
        let before_hash = world.resource::<SnapshotHistory>().state_hash(0);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = driver.rebase_authority(
                &mut world,
                1,
                |world| {
                    world.get_mut::<Counter>(entity).unwrap().0 = 900;
                    panic!("authority callback failed");
                },
                |_, _| {},
            );
        }));
        assert!(result.is_err());
        assert_eq!(world.get::<Counter>(entity).unwrap().0, 1);
        assert_eq!(
            world.resource::<SnapshotHistory>().state_hash(0),
            before_hash
        );
        assert_eq!(world.resource::<SnapshotHistory>().next_frame(), 1);
        assert_eq!(driver.inputs(), [1]);
    }

    #[test]
    fn reset_is_transactional_and_accepts_nonzero_baselines() {
        struct Unregistered;
        impl Component for Unregistered {}
        let mut world = World::new();
        let entity = world.spawn(Counter(0));
        world.register_snapshot_component_hashable::<Counter>();
        world.insert_resource(SnapshotHistory::new(2));
        let mut driver = NetworkRollbackDriver::new();
        driver.commit(&mut world, 1).unwrap();
        assert!(matches!(
            driver.reset(&mut world, 50, |world| {
                world.spawn(Unregistered);
            }),
            Err(EcsRollbackError::MissingSnapshotComponents(_))
        ));
        assert_eq!(driver.inputs(), [1]);
        assert_eq!(world.resource::<SnapshotHistory>().next_frame(), 1);
        driver
            .reset(&mut world, 50, |world| {
                world.get_mut::<Counter>(entity).unwrap().0 = 200;
            })
            .unwrap();
        assert!(driver.inputs().is_empty());
        assert_eq!(driver.input_start(), 50);
        assert!(world.resource::<SnapshotHistory>().is_empty());
        assert_eq!(driver.commit(&mut world, 3).unwrap(), 50);
        assert_eq!(driver.inputs(), [3]);
    }

    #[test]
    fn input_queue_is_bounded_and_one_input_is_admitted_per_fixed_tick() {
        let (client_transport, server_transport) = InMemoryTransport::pair_secure();
        let mut world = World::new();
        let entity = world.spawn(Counter(0));
        world.register_snapshot_component_hashable::<Counter>();
        world.insert_resource(SnapshotHistory::new(4));
        let mut bridge = EcsReplication::new(
            schema(),
            4,
            &seed(),
            ClientStrategy { entity },
            Box::new(client_transport),
        )
        .unwrap();
        let mut server = super::super::ServerReplication::new(
            schema(),
            4,
            &seed(),
            AuthorityStrategy,
            Box::new(server_transport),
        )
        .unwrap();
        for _ in 0..12 {
            bridge.replication.pump();
            server.pump();
        }
        for input in [1, 2, 3, 4] {
            bridge.queue_input(input).unwrap();
        }
        assert_eq!(bridge.queue_input(5), Err(5));
        assert!(matches!(
            bridge.take_last_error(),
            Some(EcsReplicationError::InputQueueFull)
        ));
        world.insert_resource(bridge);
        drive_ecs_input::<ClientStrategy>(&mut world);
        assert_eq!(world.get::<Counter>(entity).unwrap().0, 1);
        assert_eq!(world.resource::<SnapshotHistory>().next_frame(), 1);
        assert_eq!(
            world
                .resource::<EcsReplication<ClientStrategy>>()
                .queued_inputs(),
            3
        );
        drive_ecs_input::<ClientStrategy>(&mut world);
        assert_eq!(world.get::<Counter>(entity).unwrap().0, 3);
        assert_eq!(world.resource::<SnapshotHistory>().next_frame(), 2);
        assert_eq!(
            world
                .resource::<EcsReplication<ClientStrategy>>()
                .queued_inputs(),
            2
        );
    }
}
