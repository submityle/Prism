//! The client half of the app seam: predict locally, reconcile authority.

use alloc::boxed::Box;
use alloc::collections::VecDeque;
use alloc::vec::Vec;
use std::sync::Mutex;

use prism_app::{
    App, First, FixedFirst, FixedLast, FixedPostUpdate, FixedPreUpdate, FixedUpdate, Plugin,
};
use prism_ecs::resource::Resource;
use prism_ecs::system::ResMut;

use crate::delta::{Record, Schema};
use crate::frame::{FrameDecoder, FrameError};
use crate::multiplex::{
    negotiate_protocol, ChannelMux, MuxSendError, MuxStats, NegotiatedProtocol, NetChannel,
    ProtocolError, QueueOutcome,
};
use crate::session::{ClientSession, SessionError};
use crate::transport::Transport;

use super::{
    decode_event, decode_rpc, decode_state, encode_delivery_ack, encode_event, encode_hello,
    encode_rpc, encode_state, ConnectionState, EventEnvelope, ReplicationConfig, RpcEnvelope,
};

/// One reliable application event retained until the integration applies it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeliveredEvent {
    /// Durable peer incarnation that owns this sequence namespace.
    pub generation: u64,
    /// Monotonic sequence within `generation`, starting at one.
    pub sequence: u64,
    /// Exact application payload.
    pub payload: Vec<u8>,
}

/// Invalid application delivery cursor or a reliable-channel refusal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DeliveryAckError {
    /// The cursor was not delivered on the selected peer generation or would
    /// move the cumulative acknowledgement backwards.
    InvalidCursor,
    /// The reliable RPC lane could not admit the ACK.
    Send(MuxSendError),
}

/// The application's replication strategy: how a local input advances the record.
pub trait ReplicationStrategy: Send + Sync + 'static {
    /// The local input type.
    type Input: Clone + Send + Sync + 'static;
    /// Advance `record` by one local `input`.
    fn simulate(&self, record: &mut Record, input: &Self::Input);
    /// Encode a local input as the command bytes authority decodes
    /// ([`ServerStrategy::decode`](super::ServerStrategy::decode)); this is the
    /// client end of the command wire format.
    fn encode(&self, input: &Self::Input) -> Vec<u8>;
}

/// A replication strategy that can project the record protocol onto an ECS
/// World. The record remains the wire/prediction representation; the World is
/// the authoritative local simulation state.
pub trait EcsReplicationStrategy: ReplicationStrategy {
    /// Apply one local predicted input to the ECS World.
    fn apply_input(&self, world: &mut prism_ecs::world::World, input: &Self::Input);
    /// Replace the ECS World with an authoritative post-step record.
    fn apply_authority(&self, world: &mut prism_ecs::world::World, state: &Record);
}

/// The fixed-tick projection used by the ECS rollback bridge.
///
/// A normal prediction is applied by [`Self::apply_input`] inside the live
/// `FixedFirst` schedule. During reconciliation the bridge has to reproduce the
/// same tick without re-entering its network system, so the default projection
/// applies the input and then runs the complete `FixedMain` phase sequence.
/// Applications with a different fixed-tick contract may override this method
/// (for example, to install a replay input resource before the phases run).
pub trait NetworkRollbackSimulation: EcsReplicationStrategy {
    /// Re-simulate one fixed tick from the current World state.
    fn replay_fixed_tick(&self, world: &mut prism_ecs::world::World, input: &Self::Input) {
        self.apply_input(world, input);
        world.run_schedule(FixedFirst);
        world.run_schedule(FixedPreUpdate);
        world.run_schedule(FixedUpdate);
        world.run_schedule(FixedPostUpdate);
        world.run_schedule(FixedLast);
    }
}

impl<T: EcsReplicationStrategy> NetworkRollbackSimulation for T {}

/// One authoritative post-step record delivered to an app integration.
///
/// `revision` is one-based relative to the seeded baseline: revision `1` is
/// the authoritative result of the first predicted input. The record is
/// cloned before the session replays local pending inputs, so an ECS bridge can
/// apply the authority to its own World without observing the predicted tail.
#[derive(Clone, Debug)]
pub struct AuthorityUpdate<I> {
    /// The authoritative post-input revision.
    pub revision: u32,
    /// The authoritative state at `revision`.
    pub state: Record,
    /// The input type carried by the owning session. This phantom field keeps
    /// the update tied to the session's strategy without duplicating wire data.
    pub(crate) _input: core::marker::PhantomData<fn() -> I>,
}

/// Authority frames held until the ECS bridge can commit them together with
/// the World/history/input timeline.
///
/// Keeping the kind explicit is important here: a snapshot is a reseed
/// transaction, not a delta that can be passed through `rebase_authority`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum DeferredAuthority {
    /// One or more framed authoritative deltas.
    Delta(Vec<u8>),
    /// A framed full snapshot that requires the driver's atomic reset API.
    Snapshot(Vec<u8>),
    /// Hashes following a staged snapshot must be admitted against its baseline,
    /// not the abandoned World's tick window.
    #[cfg(feature = "determinism")]
    FrameHash(super::FrameHashUpdate),
}

/// The app-facing replication resource.
pub struct Replication<S: ReplicationStrategy> {
    session: ClientSession<S::Input>,
    strategy: S,
    mux: ChannelMux<Box<dyn Transport>>,
    config: ReplicationConfig,
    connection: ConnectionState,
    next_rpc_id: Option<u64>,
    inbound_events: VecDeque<Vec<u8>>,
    inbound_delivery_events: VecDeque<DeliveredEvent>,
    delivery_generation: Option<u64>,
    highest_delivery_received: u64,
    highest_delivery_dequeued: u64,
    acknowledged_delivery_sequence: u64,
    inbound_rpcs: VecDeque<Vec<u8>>,
    inbound_inputs: VecDeque<Vec<u8>>,
    inbound_bulk: VecDeque<Vec<u8>>,
    authority_updates: VecDeque<AuthorityUpdate<S::Input>>,
    /// Raw authority messages held until an ECS bridge has committed the World
    /// rollback. The record session is advanced only after that commit.
    deferred_authority: VecDeque<DeferredAuthority>,
    /// A rejected record stream cannot accept more deltas. Keep its failed
    /// message for diagnostics/retry, but allow a later snapshot through even
    /// when the deferred queue is full.
    deferred_resync_required: bool,
    #[cfg(feature = "determinism")]
    audit: super::FrameHashAudit,
    #[cfg(feature = "determinism")]
    last_audit_error: Option<super::FrameHashAuditError>,
    newest_state: Option<Vec<u8>>,
    /// The first error from the most recent [`pump`](Self::pump), readable by the
    /// app; a malformed or rejected frame must not panic the frame loop.
    last_error: Option<SessionError>,
    last_protocol_error: Option<ProtocolError>,
    last_send_error: Option<MuxSendError>,
}

impl<S: ReplicationStrategy> Resource for Replication<S> {}

impl<S: ReplicationStrategy> Replication<S> {
    /// Build from a snapshot payload, a strategy, and a transport.
    ///
    /// # Errors
    /// [`SessionError`] for a malformed seed snapshot.
    pub fn new(
        schema: Schema,
        capacity: usize,
        snapshot: &[u8],
        strategy: S,
        transport: Box<dyn Transport>,
    ) -> Result<Self, SessionError> {
        Self::new_with_config(
            schema,
            capacity,
            snapshot,
            strategy,
            transport,
            ReplicationConfig::for_capacity(capacity),
        )
    }

    /// Build with explicit multiplexing and handshake limits.
    ///
    /// # Errors
    /// [`SessionError`] for a malformed seed snapshot, or
    /// [`SessionError::Transport`] if the MTU cannot fit the handshake.
    pub fn new_with_config(
        schema: Schema,
        capacity: usize,
        snapshot: &[u8],
        strategy: S,
        transport: Box<dyn Transport>,
        config: ReplicationConfig,
    ) -> Result<Self, SessionError> {
        super::require_secure_transport(transport.as_ref())?;
        let config = config.for_session(capacity);
        let mut mux = ChannelMux::new(transport, config.mux);
        let hello = encode_hello(config.protocol);
        mux.check_send(NetChannel::RpcReliable, hello.len())
            .map_err(SessionError::Transport)?;
        let connection = if config.protocol.rollback_window == 0 {
            ConnectionState::Rejected(ProtocolError::InvalidRollbackWindow { offered: 0 })
        } else {
            mux.send(NetChannel::RpcReliable, &hello)
                .map_err(SessionError::Transport)?;
            ConnectionState::Handshaking
        };
        Ok(Self {
            session: ClientSession::new(schema, capacity, snapshot)?,
            strategy,
            mux,
            config,
            connection,
            next_rpc_id: Some(0),
            inbound_events: VecDeque::new(),
            inbound_delivery_events: VecDeque::new(),
            delivery_generation: None,
            highest_delivery_received: 0,
            highest_delivery_dequeued: 0,
            acknowledged_delivery_sequence: 0,
            inbound_rpcs: VecDeque::new(),
            inbound_inputs: VecDeque::new(),
            inbound_bulk: VecDeque::new(),
            authority_updates: VecDeque::new(),
            deferred_authority: VecDeque::new(),
            deferred_resync_required: false,
            #[cfg(feature = "determinism")]
            audit: super::FrameHashAudit::new(capacity),
            #[cfg(feature = "determinism")]
            last_audit_error: None,
            newest_state: None,
            last_error: None,
            last_protocol_error: match connection {
                ConnectionState::Rejected(error) => Some(error),
                ConnectionState::Handshaking | ConnectionState::Established(_) => None,
            },
            last_send_error: None,
        })
    }

    /// Advance one tick and drain every recovered frame into the prediction,
    /// returning how many authoritative frames were applied. One tick transmits
    /// new / timed-out messages and reassembles inbound ones; it is called once
    /// per frame by the plugin's `First` system. A framing or reconciliation
    /// error stops the drain and is stored for
    /// [`take_last_error`](Self::take_last_error) rather than propagated, so a bad
    /// frame cannot unwind the frame loop.
    pub fn pump(&mut self) -> usize {
        self.pump_inner(false)
    }

    /// Pump transport traffic while deferring authoritative deltas.
    ///
    /// [`EcsReplicationPlugin`](super::EcsReplicationPlugin) uses this mode so
    /// the record prediction is not acknowledged before the corresponding ECS
    /// World rollback/replay has succeeded.
    pub(crate) fn pump_deferred(&mut self) -> usize {
        self.pump_inner(true)
    }

    fn pump_inner(&mut self, defer_authority: bool) -> usize {
        self.last_error = None;
        self.mux.tick();
        if self.mux.security_state() == crate::transport::TransportSecurity::Rejected
            || self.mux.take_security_error().is_some()
        {
            self.reject(ProtocolError::SecurityFailure);
            return 0;
        }
        if matches!(self.connection, ConnectionState::Rejected(_)) {
            return 0;
        }
        self.drain_control();
        if !matches!(self.connection, ConnectionState::Established(_)) {
            return 0;
        }

        let mut applied = 0;
        let event_queue_limit = self.config.mux.reliable_queue.max(1);
        while self.inbound_events.len() + self.inbound_delivery_events.len() < event_queue_limit
            && (!defer_authority
                || self.deferred_authority.len() < event_queue_limit
                || self.deferred_resync_required)
        {
            // Do not pop another reliable event once deferred authority has
            // reached its bounded hand-off queue. Leaving it in the mux keeps
            // backpressure lossless and prevents an unbounded client-side
            // staging queue across pump calls.
            let Some(bytes) = self.mux.recv(NetChannel::EventsReliable) else {
                break;
            };
            match decode_event(&bytes) {
                // A failed delta remains buffered. Do not grow that decoder
                // indefinitely; only a subsequent ordered snapshot can resync it.
                Ok(EventEnvelope::Delta(_))
                    if (defer_authority && self.deferred_resync_required)
                        || (!defer_authority && self.session.buffered() != 0) => {}
                Ok(EventEnvelope::Delta(delta)) if defer_authority => {
                    self.deferred_authority
                        .push_back(DeferredAuthority::Delta(delta.to_vec()));
                }
                Ok(EventEnvelope::Delta(delta)) => {
                    let mut updates = Vec::new();
                    let result = self.session.receive_with_observer(
                        delta,
                        |record, input| self.strategy.simulate(record, input),
                        |revision, state| {
                            updates.push(AuthorityUpdate {
                                revision,
                                state: state.clone(),
                                _input: core::marker::PhantomData,
                            });
                        },
                    );
                    match result {
                        Ok(revisions) => applied += revisions.len(),
                        Err(error) => {
                            self.last_error = Some(error);
                            break;
                        }
                    }
                    self.authority_updates.extend(updates);
                }
                Ok(EventEnvelope::Payload(event)) => {
                    self.inbound_events.push_back(event.to_vec());
                }
                Ok(EventEnvelope::Delivery {
                    generation,
                    sequence,
                    payload,
                }) => {
                    if self.delivery_generation.is_none() {
                        if sequence != 1 {
                            self.reject(ProtocolError::MalformedMessage {
                                channel: NetChannel::EventsReliable,
                            });
                            break;
                        }
                        self.delivery_generation = Some(generation);
                    }
                    if self.delivery_generation != Some(generation) {
                        self.reject(ProtocolError::MalformedMessage {
                            channel: NetChannel::EventsReliable,
                        });
                        break;
                    }
                    if sequence > self.highest_delivery_received {
                        if self.highest_delivery_received.checked_add(1) != Some(sequence) {
                            self.reject(ProtocolError::MalformedMessage {
                                channel: NetChannel::EventsReliable,
                            });
                            break;
                        }
                        self.highest_delivery_received = sequence;
                        self.inbound_delivery_events.push_back(DeliveredEvent {
                            generation,
                            sequence,
                            payload: payload.to_vec(),
                        });
                    }
                }
                #[cfg(feature = "determinism")]
                Ok(EventEnvelope::FrameHash(update))
                    if defer_authority && self.deferred_reseed_pending() =>
                {
                    self.deferred_authority
                        .push_back(DeferredAuthority::FrameHash(update));
                }
                #[cfg(feature = "determinism")]
                Ok(EventEnvelope::FrameHash(update)) => {
                    if let Err(error) = self.audit.receive(update) {
                        self.last_audit_error.get_or_insert(error);
                    }
                }
                Ok(EventEnvelope::Snapshot(snapshot)) => {
                    if defer_authority {
                        // Validate before superseding the queued prefix. A
                        // full snapshot covers all earlier ordered deltas, but
                        // a malformed snapshot must not discard them.
                        if let Err(error) = self.reseeded_session(snapshot) {
                            self.last_error = Some(error);
                            break;
                        }
                        self.deferred_authority.clear();
                        self.deferred_authority
                            .push_back(DeferredAuthority::Snapshot(snapshot.to_vec()));
                        self.deferred_resync_required = false;
                    } else {
                        match self.reseeded_session(snapshot) {
                            Ok(session) => {
                                #[cfg(feature = "determinism")]
                                self.audit.reset_branch(u64::from(session.acked()));
                                self.session = session;
                                self.authority_updates.clear();
                            }
                            Err(error) => {
                                self.last_error = Some(error);
                                break;
                            }
                        }
                    }
                }
                Err(error) => {
                    self.reject(error);
                    break;
                }
            }
        }

        if !matches!(self.connection, ConnectionState::Established(_)) {
            return applied;
        }

        while let Some(bytes) = self.mux.recv(NetChannel::StateUnreliable) {
            match decode_state(&bytes) {
                Ok(state) => self.newest_state = Some(state.to_vec()),
                Err(error) => {
                    self.reject(error);
                    break;
                }
            }
        }

        while self.inbound_inputs.len() < self.config.mux.input_queue.max(1) {
            let Some(input) = self.mux.recv(NetChannel::InputUnreliable) else {
                break;
            };
            self.inbound_inputs.push_back(input);
        }
        while self.inbound_bulk.len() < self.config.mux.reliable_queue.max(1) {
            let Some(bulk) = self.mux.recv(NetChannel::BulkReliable) else {
                break;
            };
            self.inbound_bulk.push_back(bulk);
        }
        #[cfg(feature = "determinism")]
        self.flush_frame_hashes();
        applied
    }

    /// Reconcile one deferred authority message together with an ECS World.
    ///
    /// The client record session is cloned before applying the message. If the
    /// ECS bridge refuses the corresponding World transaction, the record
    /// session and the raw authority message are both left untouched, allowing a
    /// caller to surface the error or resync without creating a split-brain
    /// predicted state.
    /// A validated incoming snapshot supersedes its queued authority prefix;
    /// its replacement session is published only after an atomic driver reset.
    pub(crate) fn reconcile_deferred_ecs(
        &mut self,
        world: &mut prism_ecs::world::World,
        driver: &mut super::ecs::NetworkRollbackDriver<S::Input>,
    ) -> Result<bool, super::ecs::DeferredEcsError>
    where
        S: EcsReplicationStrategy + NetworkRollbackSimulation,
    {
        let Some(authority) = self.deferred_authority.front().cloned() else {
            return Ok(false);
        };

        #[cfg(feature = "determinism")]
        if let DeferredAuthority::FrameHash(update) = authority {
            if let Err(error) = self.audit.receive(update) {
                self.last_audit_error.get_or_insert(error);
            }
            self.deferred_authority.pop_front();
            return Ok(true);
        }

        let mut next_session = self.session.clone();
        let strategy = &self.strategy;
        match authority {
            DeferredAuthority::Delta(delta) => {
                let mut final_authority = None;
                let result = next_session.receive_with_observer(
                    &delta,
                    |record, input| strategy.simulate(record, input),
                    |revision, state| {
                        // A message may contain multiple complete frames.
                        // Finish all fallible record work before touching
                        // the World, then rebase once on the final update.
                        final_authority = Some((revision, state.clone()));
                    },
                );
                if let Err(error) = result {
                    self.deferred_resync_required = true;
                    return Err(super::ecs::DeferredEcsError::Session(error));
                }
                if let Some((revision, state)) = final_authority {
                    if let Err(error) = driver.rebase_authority(
                        world,
                        u64::from(revision),
                        |world| strategy.apply_authority(world, &state),
                        |world, input| strategy.replay_fixed_tick(world, input),
                    ) {
                        // An expired/missing input anchor cannot be repaired
                        // by accepting more deltas. Permit a later snapshot
                        // through even if this rejected head fills the queue.
                        if matches!(error, prism_app::EcsRollbackError::UnavailableFrame { .. }) {
                            self.deferred_resync_required = true;
                        }
                        return Err(super::ecs::DeferredEcsError::Rollback(error));
                    }
                }
            }
            DeferredAuthority::Snapshot(snapshot) => {
                next_session = self
                    .reseeded_session(&snapshot)
                    .map_err(super::ecs::DeferredEcsError::Session)?;
                let revision = next_session.acked();
                let state = next_session
                    .state()
                    .ok_or(super::ecs::DeferredEcsError::Session(
                        SessionError::NoBaseline,
                    ))?;
                driver
                    .reset(world, u64::from(revision), |world| {
                        strategy.apply_authority(world, state);
                    })
                    .map_err(super::ecs::DeferredEcsError::Rollback)?;
                // Observers from the abandoned pre-snapshot branch must not
                // leak out after a successful reseed.
                self.authority_updates.clear();
                #[cfg(feature = "determinism")]
                self.audit.reset_branch(u64::from(revision));
            }
            #[cfg(feature = "determinism")]
            DeferredAuthority::FrameHash(_) => {
                unreachable!("hashes are handled before session clone")
            }
        }
        self.session = next_session;
        self.deferred_resync_required = false;
        self.deferred_authority.pop_front();
        Ok(true)
    }

    /// Whether the next ECS reconciliation is a snapshot reset. The bridge
    /// clears its not-yet-driven input queue only after that call succeeds.
    pub(crate) fn deferred_reseed_pending(&self) -> bool {
        matches!(
            self.deferred_authority.front(),
            Some(DeferredAuthority::Snapshot(_))
        )
    }

    fn reseeded_session(&self, snapshot: &[u8]) -> Result<ClientSession<S::Input>, SessionError> {
        // Session::reseed consumes the first framed snapshot. At the app seam
        // the envelope must contain exactly one complete snapshot, with no
        // second frame or trailing partial bytes silently discarded.
        let mut decoder = FrameDecoder::new();
        decoder.feed(snapshot);
        decoder.next_frame()?.ok_or(FrameError::Truncated)?;
        if decoder.buffered() != 0 {
            return Err(SessionError::Frame(FrameError::Malformed));
        }
        let mut session = self.session.clone();
        session.reseed(snapshot)?;
        Ok(session)
    }

    fn drain_control(&mut self) {
        while self.inbound_rpcs.len() < self.config.mux.reliable_queue.max(1) {
            let Some(bytes) = self.mux.recv(NetChannel::RpcReliable) else {
                break;
            };
            let envelope = match decode_rpc(&bytes) {
                Ok(envelope) => envelope,
                Err(error) => {
                    self.reject(error);
                    break;
                }
            };
            match envelope {
                RpcEnvelope::Accept(remote) => {
                    if !matches!(self.connection, ConnectionState::Handshaking) {
                        self.reject(ProtocolError::UnexpectedHandshake);
                        break;
                    }
                    let remote_offer = crate::multiplex::ProtocolOffer {
                        version: remote.version,
                        rollback_window: remote.rollback_window,
                    };
                    match negotiate_protocol(self.config.protocol, remote_offer) {
                        Ok(negotiated) if negotiated == remote => {
                            self.connection = ConnectionState::Established(negotiated);
                        }
                        Ok(_) => {
                            self.reject(ProtocolError::UnexpectedHandshake);
                            break;
                        }
                        Err(error) => {
                            self.reject(error);
                            break;
                        }
                    }
                }
                RpcEnvelope::Reject(error) => {
                    let error = match error {
                        ProtocolError::VersionMismatch { local, remote } => {
                            ProtocolError::VersionMismatch {
                                local: remote,
                                remote: local,
                            }
                        }
                        other => other,
                    };
                    self.reject(error);
                    break;
                }
                RpcEnvelope::Payload(payload) => {
                    if !matches!(self.connection, ConnectionState::Established(_)) {
                        self.reject(ProtocolError::TrafficBeforeHandshake {
                            channel: NetChannel::RpcReliable,
                        });
                        break;
                    }
                    self.inbound_rpcs.push_back(payload.to_vec());
                }
                RpcEnvelope::Hello(_) => {
                    self.reject(ProtocolError::UnexpectedHandshake);
                    break;
                }
                RpcEnvelope::DeliveryAck { .. } => {
                    self.reject(ProtocolError::UnexpectedHandshake);
                    break;
                }
            }
        }
    }

    fn reject(&mut self, error: ProtocolError) {
        self.connection = ConnectionState::Rejected(error);
        self.last_protocol_error = Some(error);
    }

    /// Take the error the last [`pump`](Self::pump) stopped on, clearing it.
    pub fn take_last_error(&mut self) -> Option<SessionError> {
        self.last_error.take()
    }

    /// Take the most recent handshake or envelope failure.
    pub fn take_protocol_error(&mut self) -> Option<ProtocolError> {
        self.last_protocol_error.take()
    }

    /// Take the most recent explicit send refusal.
    pub fn take_send_error(&mut self) -> Option<MuxSendError> {
        self.last_send_error.take()
    }

    /// Predict one local input immediately.
    ///
    /// # Errors
    /// [`SessionError::Prediction`] when the predicted state is not retained,
    /// [`SessionError::RollbackWindowFull`] when the baseline plus pending inputs
    /// exhausts the rollback window, or [`SessionError::Protocol`] on rejection.
    pub fn predict(&mut self, input: S::Input) -> Result<u32, SessionError> {
        self.ensure_prediction_capacity()?;
        let Self {
            session, strategy, ..
        } = self;
        session.predict(input, |record, input| strategy.simulate(record, input))
    }

    /// Whether one more local prediction fits in the negotiated rollback
    /// window. This is exposed for an ECS driver so it can commit its World
    /// snapshot only when the network prediction can be admitted as well.
    #[must_use]
    pub fn can_predict(&self) -> bool {
        self.ensure_prediction_capacity().is_ok()
    }

    /// Explain why another local prediction cannot currently be admitted.
    pub fn check_prediction_capacity(&self) -> Result<(), SessionError> {
        self.ensure_prediction_capacity()
    }

    /// Queue the command bytes for an input that has already been predicted.
    ///
    /// [`predict`](Self::predict) and this method are intentionally separate
    /// for ECS integrations: the World snapshot is committed between the two
    /// operations, so a failed snapshot never leaves a network-only prediction.
    pub fn send_predicted(&mut self, input: &S::Input) -> Result<u64, MuxSendError> {
        let bytes = self.strategy.encode(input);
        self.try_send_rpc(&bytes)
    }

    /// Remove the newest local prediction after an external ECS transaction
    /// fails. The caller must invoke this only for the immediately preceding
    /// [`predict`](Self::predict) call.
    pub(crate) fn cancel_last_prediction(&mut self) {
        self.session.cancel_last_prediction();
    }

    /// Check whether the command for an input can be queued without mutating
    /// the transport. This is the send-side half of the ECS transaction.
    pub(crate) fn check_predicted_send(&mut self, input: &S::Input) -> Result<(), MuxSendError> {
        let bytes = self.strategy.encode(input);
        let frame = encode_rpc(&bytes);
        self.mux.check_send(NetChannel::RpcReliable, frame.len())
    }

    /// Apply a predicted input to an ECS World through an ECS-capable strategy.
    pub fn apply_input_to_world(&self, world: &mut prism_ecs::world::World, input: &S::Input)
    where
        S: EcsReplicationStrategy,
    {
        self.strategy.apply_input(world, input);
    }

    /// Apply an authoritative record to an ECS World through an ECS-capable
    /// strategy. The record is the post-step authority state, before pending
    /// local inputs are replayed.
    pub fn apply_authority_to_world(&self, world: &mut prism_ecs::world::World, state: &Record)
    where
        S: EcsReplicationStrategy,
    {
        self.strategy.apply_authority(world, state);
    }

    /// Predict `input` locally **and** send it to authority in one call — the
    /// client half of the command path.
    ///
    /// Returns the local RPC call number, or `Ok(None)` on queue backpressure,
    /// MTU overflow or connection rejection. A refused send predicts nothing, so a caller that
    /// retries the same input cannot double-predict it; check
    /// [`is_full`](Self::is_full) to avoid the `None` case. Prediction is local
    /// and independent of delivery — authority will confirm or correct it later.
    ///
    /// # Errors
    /// [`SessionError::Prediction`] when the predicted state is not retained, or
    /// [`SessionError::RollbackWindowFull`] when the negotiated window is full.
    pub fn predict_and_send(&mut self, input: S::Input) -> Result<Option<u64>, SessionError> {
        if matches!(self.connection, ConnectionState::Rejected(_)) {
            self.last_send_error = Some(MuxSendError::ConnectionNotEstablished);
            return Ok(None);
        }
        if self.next_rpc_id.is_none() {
            return Err(SessionError::RpcIdentityExhausted);
        }
        let bytes = self.strategy.encode(&input);
        let frame = encode_rpc(&bytes);
        if let Err(error) = self.mux.check_send(NetChannel::RpcReliable, frame.len()) {
            self.last_send_error = Some(error);
            return Ok(None);
        }
        self.ensure_prediction_capacity()?;
        let Self {
            session,
            strategy,
            mux,
            next_rpc_id,
            ..
        } = self;
        session.predict(input, |record, input| strategy.simulate(record, input))?;
        mux.send(NetChannel::RpcReliable, &frame)
            .expect("send preflight and queue mutation are contiguous");
        let id = next_rpc_id.expect("identity availability checked before prediction");
        *next_rpc_id = id.checked_add(1);
        Ok(Some(id))
    }

    /// Send `bytes` to the peer reliably: the channel numbers and retains it until
    /// the peer acknowledges, retransmitting on later frames if a datagram drops.
    /// Returns a local RPC call number. On backpressure, MTU overflow or
    /// rejection, returns `None`; [`take_send_error`](Self::take_send_error)
    /// distinguishes those cases. Commands may queue behind the initial hello.
    pub fn send(&mut self, bytes: &[u8]) -> Option<u64> {
        self.try_send_rpc(bytes).ok()
    }

    /// Queue one reliable ordered RPC/control payload and return its local call
    /// sequence. Backpressure and MTU failures are explicit.
    pub fn try_send_rpc(&mut self, bytes: &[u8]) -> Result<u64, MuxSendError> {
        let Some(id) = self.next_rpc_id else {
            self.last_send_error = Some(MuxSendError::RpcIdentityExhausted);
            return Err(MuxSendError::RpcIdentityExhausted);
        };
        if matches!(self.connection, ConnectionState::Rejected(_)) {
            let error = MuxSendError::ConnectionNotEstablished;
            self.last_send_error = Some(error);
            return Err(error);
        }
        let frame = encode_rpc(bytes);
        if let Err(error) = self.mux.send(NetChannel::RpcReliable, &frame) {
            self.last_send_error = Some(error);
            return Err(error);
        }
        self.next_rpc_id = id.checked_add(1);
        Ok(id)
    }

    /// Queue a reliable ordered event/proposal independently of RPC traffic.
    /// Requires an established connection.
    pub fn send_event(&mut self, bytes: &[u8]) -> Result<QueueOutcome, MuxSendError> {
        self.require_established()?;
        let frame = encode_event(bytes);
        let result = self.mux.send(NetChannel::EventsReliable, &frame);
        self.record_send(result)
    }

    /// Establish the persisted application cursor before pumping a resumed
    /// durable-delivery stream. The caller must persist this cursor atomically
    /// with applying each event; the transport cannot provide that guarantee.
    pub fn resume_delivery_cursor(&mut self, generation: u64, applied_sequence: u64) -> bool {
        if generation == 0 || !self.inbound_delivery_events.is_empty() {
            return false;
        }
        if self
            .delivery_generation
            .is_some_and(|current| current != generation)
        {
            return false;
        }
        if self.highest_delivery_received != self.acknowledged_delivery_sequence {
            return false;
        }
        self.delivery_generation = Some(generation);
        self.highest_delivery_received = applied_sequence;
        self.highest_delivery_dequeued = applied_sequence;
        self.acknowledged_delivery_sequence = applied_sequence;
        true
    }

    /// Acknowledge an event only after the application has successfully applied
    /// it. ACKs are cumulative and may be re-sent idempotently after reconnect.
    pub fn acknowledge_delivery(
        &mut self,
        generation: u64,
        applied_sequence: u64,
    ) -> Result<(), DeliveryAckError> {
        self.require_established().map_err(DeliveryAckError::Send)?;
        if self.delivery_generation != Some(generation)
            || applied_sequence == 0
            || applied_sequence < self.acknowledged_delivery_sequence
            || applied_sequence > self.highest_delivery_dequeued
        {
            return Err(DeliveryAckError::InvalidCursor);
        }
        let bytes = encode_delivery_ack(generation, applied_sequence);
        let result = self.mux.send(NetChannel::RpcReliable, &bytes);
        self.record_send(result).map_err(DeliveryAckError::Send)?;
        self.acknowledged_delivery_sequence = applied_sequence;
        Ok(())
    }

    /// Retained ECS hash comparison, independent of application event traffic.
    #[cfg(feature = "determinism")]
    #[must_use]
    pub fn frame_hash_audit(&self) -> &super::FrameHashAudit {
        &self.audit
    }

    /// Copy actual captured hashes into the bounded outgoing audit. `from` is
    /// the first changed pre-input tick; rollback callers must also invoke this
    /// after partial replay failures to discard the abandoned future.
    /// Only registered snapshot hash glue covers values; see [`super::FrameHashAudit`].
    #[cfg(feature = "determinism")]
    pub fn sync_frame_hashes(
        &mut self,
        history: &prism_app::SnapshotHistory,
        from: u64,
    ) -> Result<(), super::FrameHashAuditError> {
        self.audit.sync_history(history, from)
    }

    /// First unconsumed audit admission error. Divergence is queried through
    /// `frame_hash_audit().comparison()` and can clear after correction.
    #[cfg(feature = "determinism")]
    pub fn take_audit_error(&mut self) -> Option<super::FrameHashAuditError> {
        self.last_audit_error.take()
    }

    /// Queue an explicit peer hash envelope. Normal ECS bridge captures use the
    /// retained retry queue instead. The sender owns the connection sequence.
    /// Do not mix caller-assigned sequences with `sync_frame_hashes` on the same
    /// connection: the automatic audit owns its own sequence namespace.
    #[cfg(feature = "determinism")]
    pub fn send_frame_hash(
        &mut self,
        update: super::FrameHashUpdate,
    ) -> Result<QueueOutcome, MuxSendError> {
        self.require_established()?;
        let bytes = super::encode_frame_hash(update);
        let result = self.mux.send(NetChannel::EventsReliable, &bytes);
        self.record_send(result)
    }

    /// Flush retained/coalesced hashes. Backpressure never removes an update;
    /// a later pump retries it. Retention expiry is an explicit audit error.
    #[cfg(feature = "determinism")]
    pub fn flush_frame_hashes(&mut self) {
        if self.negotiated().is_none() {
            return;
        }
        while let Some(update) = self.audit.next_pending() {
            match self.send_frame_hash(update) {
                Ok(_) => self.audit.sent(update),
                Err(_) => break,
            }
        }
    }

    /// Queue newest-wins transient state. A newer unsent value replaces the
    /// previous one rather than adding latency. Requires an established connection.
    pub fn send_state(&mut self, bytes: &[u8]) -> Result<QueueOutcome, MuxSendError> {
        self.require_established()?;
        let frame = encode_state(bytes);
        let result = self.mux.send(NetChannel::StateUnreliable, &frame);
        self.record_send(result)
    }

    /// Queue one latency-sensitive input on the bounded unreliable input lane.
    /// Older unsent inputs may be discarded when the lane is full, preserving
    /// the newest-input latency contract. Requires an established connection.
    pub fn send_input(&mut self, bytes: &[u8]) -> Result<QueueOutcome, MuxSendError> {
        self.require_established()?;
        let result = self.mux.send(NetChannel::InputUnreliable, bytes);
        self.record_send(result)
    }

    /// Queue one reliable, ordered bulk payload. The bounded reliable queue
    /// applies explicit backpressure and retransmits until acknowledged.
    /// Requires an established connection.
    pub fn send_bulk(&mut self, bytes: &[u8]) -> Result<QueueOutcome, MuxSendError> {
        self.require_established()?;
        let result = self.mux.send(NetChannel::BulkReliable, bytes);
        self.record_send(result)
    }

    /// Take one reliable event delivered by the authority.
    pub fn recv_event(&mut self) -> Option<Vec<u8>> {
        self.inbound_events.pop_front()
    }

    /// Take the next sequenced delivery event. It is not acknowledged until
    /// the application calls [`Self::acknowledge_delivery`] after applying it.
    pub fn recv_delivery_event(&mut self) -> Option<DeliveredEvent> {
        let event = self.inbound_delivery_events.pop_front()?;
        self.highest_delivery_dequeued = event.sequence;
        Some(event)
    }

    /// Take one reliable RPC response/control payload delivered by the authority.
    pub fn recv_rpc(&mut self) -> Option<Vec<u8>> {
        self.inbound_rpcs.pop_front()
    }

    /// Take the newest transient state received since the previous call.
    pub fn recv_state(&mut self) -> Option<Vec<u8>> {
        self.newest_state.take()
    }

    /// Take the oldest received input still retained by the bounded input lane.
    pub fn recv_input(&mut self) -> Option<Vec<u8>> {
        self.inbound_inputs.pop_front()
    }

    /// Take the next received reliable bulk payload.
    pub fn recv_bulk(&mut self) -> Option<Vec<u8>> {
        self.inbound_bulk.pop_front()
    }

    /// Take the next authoritative post-step record for an ECS or gameplay
    /// integration. The update is ordered exactly like the underlying delta
    /// stream and is consumed only once.
    pub fn recv_authority_update(&mut self) -> Option<AuthorityUpdate<S::Input>> {
        self.authority_updates.pop_front()
    }

    /// Whether the outbound send window is full (a [`send`](Self::send) would
    /// return `None`); the app should not produce more input until
    /// acknowledgements return.
    #[must_use]
    pub fn is_full(&self) -> bool {
        matches!(self.connection, ConnectionState::Rejected(_))
            || self.mux.is_backpressured(NetChannel::RpcReliable)
    }

    /// How many RPC/control messages are pending or unacknowledged, including
    /// the initial hello until the peer acknowledges it.
    #[must_use]
    pub fn unacked(&self) -> usize {
        self.mux.queued(NetChannel::RpcReliable)
    }

    /// Current handshake state.
    #[must_use]
    pub fn connection_state(&self) -> ConnectionState {
        self.connection
    }

    /// Agreed protocol and rollback window, once connected.
    #[must_use]
    pub fn negotiated(&self) -> Option<NegotiatedProtocol> {
        self.connection.negotiated()
    }

    /// Per-connection multiplexing and congestion counters.
    #[must_use]
    pub fn transport_stats(&self) -> MuxStats {
        self.mux.stats()
    }

    /// The authoritative revision confirmed by the server.
    #[must_use]
    pub fn acked(&self) -> u32 {
        self.session.acked()
    }

    /// The locally predicted revision.
    #[must_use]
    pub fn predicted(&self) -> u32 {
        self.session.predicted()
    }

    /// The currently predicted record.
    #[must_use]
    pub fn state(&self) -> Option<&Record> {
        self.session.state()
    }

    fn ensure_prediction_capacity(&self) -> Result<(), SessionError> {
        if let ConnectionState::Rejected(error) = self.connection {
            return Err(SessionError::Protocol(error));
        }
        let window = self
            .negotiated()
            .map_or(self.config.protocol.rollback_window, |protocol| {
                protocol.rollback_window
            });
        let pending = self.session.pending_inputs();
        // The acknowledged baseline occupies one snapshot-ring slot.
        if pending >= window.saturating_sub(1) as usize {
            return Err(SessionError::RollbackWindowFull { pending, window });
        }
        Ok(())
    }

    fn require_established(&mut self) -> Result<(), MuxSendError> {
        if self.negotiated().is_some() {
            Ok(())
        } else {
            self.last_send_error = Some(MuxSendError::ConnectionNotEstablished);
            Err(MuxSendError::ConnectionNotEstablished)
        }
    }

    fn record_send<T>(&mut self, result: Result<T, MuxSendError>) -> Result<T, MuxSendError> {
        if let Err(error) = &result {
            self.last_send_error = Some(*error);
        }
        result
    }
}

/// A `prism_app` plugin that installs a [`Replication`] resource and pumps it
/// every frame.
pub struct ReplicationPlugin<S: ReplicationStrategy> {
    schema: Schema,
    capacity: usize,
    snapshot: Vec<u8>,
    /// Taken by [`Plugin::build`]; `Plugin::is_unique` guarantees one build.
    strategy: Mutex<Option<S>>,
    transport: Mutex<Option<Box<dyn Transport>>>,
    config: ReplicationConfig,
}

impl<S: ReplicationStrategy> ReplicationPlugin<S> {
    /// A plugin that seeds a client from `snapshot` and drives it over
    /// `transport` (the caller keeps the peer endpoint).
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

impl<S: ReplicationStrategy> Plugin for ReplicationPlugin<S> {
    fn build(&self, app: &mut App) {
        let strategy = self
            .strategy
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
            .expect("ReplicationPlugin::build ran twice");
        let transport = self
            .transport
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
            .expect("ReplicationPlugin::build ran twice");
        let replication = Replication::<S>::new_with_config(
            self.schema.clone(),
            self.capacity,
            &self.snapshot,
            strategy,
            transport,
            self.config,
        )
        .expect("replication seed snapshot is malformed");
        app.insert_resource(replication);
        app.add_systems(First, pump_replication::<S>);
    }
}

/// The per-frame system: drain inbound replication frames into the prediction.
fn pump_replication<S: ReplicationStrategy>(mut replication: ResMut<Replication<S>>) {
    replication.pump();
}

#[cfg(test)]
#[path = "client_deferred_tests.rs"]
mod deferred_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::delta::{encode_snapshot, FieldKind, FieldValue};
    use crate::plugin::{ReplicationServerPlugin, ServerReplication, ServerStrategy};
    use crate::transport::memory::InMemoryTransport;
    use alloc::vec;

    /// A strategy whose single int field advances by the input.
    struct Add;
    impl ReplicationStrategy for Add {
        type Input = i64;
        fn simulate(&self, record: &mut Record, input: &i64) {
            let FieldValue::Int(current) = record.value(0).expect("one field") else {
                panic!("field 0 is an int");
            };
            record.set(0, FieldValue::Int(current + input));
        }
        fn encode(&self, input: &i64) -> Vec<u8> {
            input.to_le_bytes().to_vec()
        }
    }

    pub(crate) fn schema() -> Schema {
        Schema::new(vec![FieldKind::Int { bits: 12 }])
    }

    pub(crate) fn snapshot_at(revision: u32, value: i64) -> Vec<u8> {
        let mut bytes = Vec::new();
        encode_snapshot(
            &schema(),
            revision,
            &Record::new(vec![FieldValue::Int(value)]),
            &mut bytes,
        )
        .unwrap();
        bytes
    }

    pub(crate) fn int(record: &Record) -> i64 {
        let FieldValue::Int(value) = record.value(0).expect("one field") else {
            panic!("field 0 is an int");
        };
        value
    }

    struct AddAuthority;
    impl ServerStrategy for AddAuthority {
        type Command = i64;

        fn decode(&self, bytes: &[u8]) -> Option<Self::Command> {
            <[u8; 8]>::try_from(bytes).ok().map(i64::from_le_bytes)
        }

        fn apply(&self, record: &mut Record, command: Self::Command) {
            record.set(0, FieldValue::Int(int(record).saturating_add(command)));
        }
    }

    #[test]
    fn plugins_negotiate_then_reconcile_a_predicted_rpc() {
        let (client_end, server_end) = InMemoryTransport::pair_secure();
        let client_config = ReplicationConfig {
            protocol: crate::multiplex::ProtocolOffer::current(12),
            ..ReplicationConfig::default()
        };
        let server_config = ReplicationConfig {
            protocol: crate::multiplex::ProtocolOffer::current(8),
            ..ReplicationConfig::default()
        };

        let mut client_app = App::new();
        client_app.add_plugins(
            ReplicationPlugin::new(schema(), 16, snapshot_at(0, 0), Add, Box::new(client_end))
                .with_config(client_config),
        );

        let mut server_app = App::new();
        server_app.add_plugins(
            ReplicationServerPlugin::new(
                schema(),
                16,
                snapshot_at(0, 0),
                AddAuthority,
                Box::new(server_end),
            )
            .with_config(server_config),
        );

        for _ in 0..4 {
            client_app.update();
            server_app.update();
        }

        let negotiated = client_app
            .world()
            .resource::<Replication<Add>>()
            .negotiated()
            .expect("handshake completed");
        assert_eq!(negotiated.rollback_window, 8, "the smaller offer wins");

        let client = client_app.world_mut().resource_mut::<Replication<Add>>();
        assert_eq!(client.predict_and_send(3).unwrap(), Some(0));
        assert_eq!(client.predicted(), 1);

        for _ in 0..8 {
            client_app.update();
            server_app.update();
        }

        let server = server_app
            .world()
            .resource::<ServerReplication<AddAuthority>>();
        assert_eq!(server.revision(), Some(1));
        assert_eq!(int(server.record().unwrap()), 3);
        let client = client_app.world().resource::<Replication<Add>>();
        assert_eq!(client.acked(), 1);
        assert_eq!(int(client.state().unwrap()), 3);
    }

    #[test]
    fn durable_delivery_requires_dequeue_and_roundtrips_application_ack() {
        let (client_end, server_end) = InMemoryTransport::pair_secure();
        let mut client_app = App::new();
        client_app.add_plugins(ReplicationPlugin::new(
            schema(),
            8,
            snapshot_at(0, 0),
            Add,
            Box::new(client_end),
        ));
        let mut server_app = App::new();
        server_app.add_plugins(ReplicationServerPlugin::new(
            schema(),
            8,
            snapshot_at(0, 0),
            AddAuthority,
            Box::new(server_end),
        ));

        for _ in 0..4 {
            client_app.update();
            server_app.update();
        }
        assert!(client_app
            .world()
            .resource::<Replication<Add>>()
            .negotiated()
            .is_some());

        server_app
            .world_mut()
            .resource_mut::<ServerReplication<AddAuthority>>()
            .send_delivery(9, 1, b"committed payload")
            .unwrap();
        for _ in 0..3 {
            client_app.update();
            server_app.update();
        }

        let client = client_app.world_mut().resource_mut::<Replication<Add>>();
        assert_eq!(
            client.acknowledge_delivery(9, 1),
            Err(DeliveryAckError::InvalidCursor),
            "wire receipt alone must not be enough to acknowledge"
        );
        let event = client.recv_delivery_event().unwrap();
        assert_eq!(event.generation, 9);
        assert_eq!(event.sequence, 1);
        assert_eq!(event.payload, b"committed payload");
        client.acknowledge_delivery(9, 1).unwrap();

        for _ in 0..3 {
            client_app.update();
            server_app.update();
        }
        assert_eq!(
            server_app
                .world_mut()
                .resource_mut::<ServerReplication<AddAuthority>>()
                .recv_delivery_ack(),
            Some((9, 1))
        );
    }

    #[test]
    fn predict_and_send_refuses_without_predicting_under_rpc_backpressure() {
        let (client_end, _server_end) = InMemoryTransport::pair_secure();
        let config = ReplicationConfig {
            mux: crate::multiplex::MuxConfig {
                reliable_queue: 2,
                ..crate::multiplex::MuxConfig::default()
            },
            ..ReplicationConfig::for_capacity(8)
        };
        let mut app = App::new();
        app.add_plugins(
            ReplicationPlugin::new(schema(), 8, snapshot_at(0, 0), Add, Box::new(client_end))
                .with_config(config),
        );

        let replication = app.world_mut().resource_mut::<Replication<Add>>();
        // The hello already occupies one bounded RPC queue slot.
        assert!(replication.send(b"x").is_some());
        assert!(replication.is_full());
        assert_eq!(
            replication.predict_and_send(3).unwrap(),
            None,
            "the window is full: nothing sent"
        );
        assert_eq!(replication.predicted(), 0, "and nothing predicted either");
        assert!(matches!(
            replication.take_send_error(),
            Some(MuxSendError::Backpressured(NetChannel::RpcReliable))
        ));
    }

    #[test]
    fn rpc_identity_exhaustion_does_not_queue_or_predict() {
        let (client_end, _server_end) = InMemoryTransport::pair_secure();
        let mut app = App::new();
        app.add_plugins(ReplicationPlugin::new(
            schema(),
            8,
            snapshot_at(0, 0),
            Add,
            Box::new(client_end),
        ));
        let replication = app.world_mut().resource_mut::<Replication<Add>>();
        replication.next_rpc_id = None;
        let unacked_before = replication.unacked();
        assert_eq!(
            replication.predict_and_send(3),
            Err(SessionError::RpcIdentityExhausted)
        );
        assert_eq!(replication.predicted(), 0);
        assert_eq!(replication.unacked(), unacked_before);
        assert_eq!(
            replication.try_send_rpc(b"no id"),
            Err(MuxSendError::RpcIdentityExhausted)
        );
        assert_eq!(replication.unacked(), unacked_before);
    }

    #[test]
    fn constructors_refuse_an_mtu_that_cannot_carry_the_handshake() {
        let (a, b) = InMemoryTransport::pair_secure();
        let mut config = ReplicationConfig::for_capacity(8);
        config.mux.path_mtu = 18;
        assert!(matches!(
            Replication::new_with_config(schema(), 8, &snapshot_at(0, 0), Add, Box::new(a), config),
            Err(SessionError::Transport(MuxSendError::TooLarge {
                len: 7,
                max: 0
            }))
        ));
        assert!(matches!(
            ServerReplication::new_with_config(
                schema(),
                8,
                &snapshot_at(0, 0),
                AddAuthority,
                Box::new(b),
                config
            ),
            Err(SessionError::Transport(MuxSendError::TooLarge {
                len: 7,
                max: 0
            }))
        ));
    }

    #[test]
    fn constructor_rejects_a_raw_transport_without_a_provider() {
        let (a, _b) = InMemoryTransport::pair();
        assert!(matches!(
            Replication::new(schema(), 8, &snapshot_at(0, 0), Add, Box::new(a),),
            Err(SessionError::Security(
                crate::security::SecurityError::ProviderRequired
            ))
        ));
    }

    #[test]
    fn rpc_envelope_overhead_is_checked_before_prediction() {
        let (a, _b) = InMemoryTransport::pair_secure();
        let mut config = ReplicationConfig::for_capacity(8);
        config.mux.path_mtu = 35; // eight payload bytes after security + mux headers
        let mut client =
            Replication::new_with_config(schema(), 8, &snapshot_at(0, 0), Add, Box::new(a), config)
                .unwrap();
        assert_eq!(client.predict_and_send(3).unwrap(), None);
        assert_eq!(client.predicted(), 0);
        assert_eq!(
            client.take_send_error(),
            Some(MuxSendError::TooLarge { len: 9, max: 8 })
        );
        assert_eq!(client.unacked(), 1, "only the hello was queued");
    }

    #[test]
    fn rollback_offer_is_clamped_to_the_real_snapshot_ring() {
        let (a, b) = InMemoryTransport::pair_secure();
        let config = ReplicationConfig {
            protocol: crate::multiplex::ProtocolOffer::current(1024),
            ..ReplicationConfig::default()
        };
        let mut client =
            Replication::new_with_config(schema(), 3, &snapshot_at(0, 0), Add, Box::new(a), config)
                .unwrap();
        let mut server = ServerReplication::new_with_config(
            schema(),
            8,
            &snapshot_at(0, 0),
            AddAuthority,
            Box::new(b),
            config,
        )
        .unwrap();
        for _ in 0..8 {
            client.pump();
            server.pump();
        }
        assert_eq!(client.negotiated().unwrap().rollback_window, 3);
        assert_eq!(server.negotiated(), client.negotiated());
        client.predict_and_send(1).unwrap().unwrap();
        client.predict_and_send(2).unwrap().unwrap();
        assert_eq!(
            client.predict(3),
            Err(SessionError::RollbackWindowFull {
                pending: 2,
                window: 3
            })
        );
        for _ in 0..8 {
            client.pump();
            server.pump();
        }
        assert_eq!(client.acked(), 2);
        assert_eq!(int(client.state().unwrap()), 3);
        assert_eq!(client.take_last_error(), None);
    }

    #[test]
    fn invalid_accepts_and_premature_rpcs_never_establish_the_client() {
        let config = ReplicationConfig::for_capacity(8);
        let cases = [
            (
                vec![1],
                ProtocolError::MalformedMessage {
                    channel: NetChannel::RpcReliable,
                },
            ),
            (
                super::super::encode_accept(NegotiatedProtocol {
                    version: config.protocol.version,
                    rollback_window: 9,
                }),
                ProtocolError::UnexpectedHandshake,
            ),
            (
                super::super::encode_accept(NegotiatedProtocol {
                    version: config.protocol.version,
                    rollback_window: 0,
                }),
                ProtocolError::InvalidRollbackWindow { offered: 0 },
            ),
            (
                super::super::encode_accept(NegotiatedProtocol {
                    version: config.protocol.version + 1,
                    rollback_window: 8,
                }),
                ProtocolError::VersionMismatch {
                    local: config.protocol.version,
                    remote: config.protocol.version + 1,
                },
            ),
            (
                encode_rpc(b"premature"),
                ProtocolError::TrafficBeforeHandshake {
                    channel: NetChannel::RpcReliable,
                },
            ),
        ];
        for (bytes, error) in cases {
            let (a, b) = InMemoryTransport::pair_secure();
            let mut client =
                Replication::new(schema(), 8, &snapshot_at(0, 0), Add, Box::new(a)).unwrap();
            let mut peer = ChannelMux::new(b, config.mux);
            peer.send(NetChannel::RpcReliable, &bytes).unwrap();
            peer.tick();
            assert_eq!(client.pump(), 0);
            assert_eq!(client.connection_state(), ConnectionState::Rejected(error));
            assert_eq!(client.take_protocol_error(), Some(error));
            assert_eq!(client.take_protocol_error(), None);
            assert_eq!(client.recv_rpc(), None);
            assert_eq!(client.acked(), 0);
            assert_eq!(client.predict(1), Err(SessionError::Protocol(error)));
        }
    }

    #[test]
    fn truncated_and_concatenated_session_frames_reject_without_mutating_prediction() {
        let config = ReplicationConfig::for_capacity(8);
        let mut frame = Vec::new();
        crate::frame::encode_frame(&snapshot_at(5, 50), &mut frame).unwrap();
        let truncated = frame[..frame.len() - 1].to_vec();
        let mut concatenated = frame.clone();
        concatenated.extend_from_slice(&frame);

        for payload in [truncated, concatenated] {
            for envelope in [
                super::super::encode_delta(&payload),
                super::super::encode_snapshot(&payload),
            ] {
                let (a, b) = InMemoryTransport::pair_secure();
                let mut client =
                    Replication::new(schema(), 8, &snapshot_at(0, 0), Add, Box::new(a)).unwrap();
                let mut peer = ChannelMux::new(b, config.mux);
                peer.send(
                    NetChannel::RpcReliable,
                    &super::super::encode_accept(NegotiatedProtocol {
                        version: config.protocol.version,
                        rollback_window: 8,
                    }),
                )
                .unwrap();
                peer.tick();
                client.pump();
                assert!(client.negotiated().is_some());
                peer.send(NetChannel::EventsReliable, &envelope).unwrap();
                peer.tick();
                assert_eq!(client.pump(), 0);
                assert_eq!(
                    client.connection_state(),
                    ConnectionState::Rejected(ProtocolError::MalformedMessage {
                        channel: NetChannel::EventsReliable
                    })
                );
                assert_eq!(client.acked(), 0);
                assert_eq!(client.predicted(), 0);
                assert_eq!(int(client.state().unwrap()), 0);
                assert_eq!(client.session.buffered(), 0);
            }
        }
    }

    #[test]
    fn zero_local_rollback_capacity_is_observable_without_sending_an_offer() {
        let (a, mut b) = InMemoryTransport::pair_secure();
        let config = ReplicationConfig {
            protocol: crate::multiplex::ProtocolOffer::current(0),
            ..ReplicationConfig::default()
        };
        let mut client =
            Replication::new_with_config(schema(), 8, &snapshot_at(0, 0), Add, Box::new(a), config)
                .unwrap();
        assert_eq!(
            client.connection_state(),
            ConnectionState::Rejected(ProtocolError::InvalidRollbackWindow { offered: 0 })
        );
        assert_eq!(client.pump(), 0);
        assert_eq!(b.recv(), None);
        assert_eq!(client.send(b"command"), None);
    }
}
