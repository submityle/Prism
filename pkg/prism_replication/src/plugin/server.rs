//! The authority half of the app seam: apply inbound commands, broadcast deltas.

use alloc::boxed::Box;
use alloc::collections::VecDeque;
use alloc::vec::Vec;
use std::sync::Mutex;

use prism_app::{App, First, Plugin};
use prism_ecs::resource::Resource;
use prism_ecs::system::ResMut;

use crate::delta::{Record, Schema};
use crate::multiplex::{
    negotiate_protocol, ChannelMux, MuxSendError, MuxStats, NegotiatedProtocol, NetChannel,
    ProtocolError, QueueOutcome,
};
use crate::session::{ServerSession, SessionError};
use crate::transport::Transport;

use super::{
    decode_event, decode_rpc, decode_state, encode_accept, encode_delta, encode_event,
    encode_reject, encode_rpc, encode_snapshot, encode_state, ConnectionState, EventEnvelope,
    ReplicationConfig, RpcEnvelope,
};

/// How the authority turns an inbound command into an authoritative change.
pub trait ServerStrategy: Send + Sync + 'static {
    /// The command type decoded from one inbound payload.
    type Command: Send + Sync + 'static;
    /// Decode one command from the bytes a client sent — the reliable channel has
    /// already reassembled the message, so this is the command's own wire format.
    /// Return `None` for an unknown or malformed command, which the authority
    /// ignores rather than applies.
    fn decode(&self, bytes: &[u8]) -> Option<Self::Command>;
    /// Apply a decoded command to the authoritative record.
    fn apply(&self, record: &mut Record, command: Self::Command);
}

/// The app-facing authority resource: a [`ServerSession`] plus its channel.
pub struct ServerReplication<S: ServerStrategy> {
    session: ServerSession,
    strategy: S,
    mux: ChannelMux<Box<dyn Transport>>,
    config: ReplicationConfig,
    connection: ConnectionState,
    next_rpc_id: u64,
    pending_commands: VecDeque<Vec<u8>>,
    inbound_events: VecDeque<Vec<u8>>,
    inbound_inputs: VecDeque<Vec<u8>>,
    inbound_bulk: VecDeque<Vec<u8>>,
    newest_state: Option<Vec<u8>>,
    /// The first error from the most recent [`pump`](Self::pump), readable by the
    /// app; a rejected command must not panic the frame loop.
    last_error: Option<SessionError>,
    applied: u64,
    /// Commands that arrived but could not be decoded (a protocol/version
    /// mismatch). Counted so a silently dropped input is observable.
    rejected: u64,
    last_protocol_error: Option<ProtocolError>,
    last_send_error: Option<MuxSendError>,
}

impl<S: ServerStrategy> Resource for ServerReplication<S> {}

impl<S: ServerStrategy> ServerReplication<S> {
    /// Build authority from a full snapshot payload, a strategy, and a transport.
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

    /// Build authority with explicit multiplexing and handshake limits.
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
        let config = config.for_session(capacity);
        let mut mux = ChannelMux::new(transport, config.mux);
        // Accept is the largest fixed-size handshake envelope (seven bytes).
        mux.check_send(NetChannel::RpcReliable, 7)
            .map_err(SessionError::Transport)?;
        let connection = if config.protocol.rollback_window == 0 {
            ConnectionState::Rejected(ProtocolError::InvalidRollbackWindow { offered: 0 })
        } else {
            ConnectionState::Handshaking
        };
        if let ConnectionState::Rejected(error) = connection {
            mux.send(NetChannel::RpcReliable, &encode_reject(error))
                .map_err(SessionError::Transport)?;
        }
        Ok(Self {
            session: ServerSession::new(schema, capacity, snapshot)?,
            strategy,
            mux,
            config,
            connection,
            next_rpc_id: 0,
            pending_commands: VecDeque::new(),
            inbound_events: VecDeque::new(),
            inbound_inputs: VecDeque::new(),
            inbound_bulk: VecDeque::new(),
            newest_state: None,
            last_error: None,
            applied: 0,
            rejected: 0,
            last_protocol_error: match connection {
                ConnectionState::Rejected(error) => Some(error),
                ConnectionState::Handshaking | ConnectionState::Established(_) => None,
            },
            last_send_error: None,
        })
    }

    /// One frame: tick the channel, decode + apply every inbound command, and
    /// broadcast the framed authoritative delta back to the peer. Returns how many
    /// commands advanced authority this frame. A rejected command is stored for
    /// [`take_last_error`](Self::take_last_error) and stops the drain rather than
    /// unwinding the frame loop.
    ///
    /// Authority is only advanced when its delta can actually be retained for
    /// delivery: if the outbound window is full the loop stops and leaves the rest
    /// of the commands queued in the channel (the peer's own send window applies
    /// backpressure), so a full window can never advance a revision the client is
    /// never told about. A command that fails [`ServerStrategy::decode`] is counted
    /// in [`rejected`](Self::rejected) rather than applied.
    pub fn pump(&mut self) -> usize {
        self.last_error = None;
        self.mux.tick();
        if matches!(self.connection, ConnectionState::Rejected(_)) {
            return 0;
        }
        self.drain_control();
        if !matches!(self.connection, ConnectionState::Established(_)) {
            return 0;
        }
        self.drain_auxiliary_channels();
        if !matches!(self.connection, ConnectionState::Established(_)) {
            return 0;
        }

        let mut advanced = 0;
        while let Some(bytes) = self.pending_commands.front() {
            if self.mux.is_backpressured(NetChannel::EventsReliable) {
                self.last_send_error =
                    Some(MuxSendError::Backpressured(NetChannel::EventsReliable));
                break;
            }
            let Some(command) = self.strategy.decode(bytes) else {
                self.pending_commands.pop_front();
                self.rejected = self.rejected.saturating_add(1);
                continue;
            };

            // Advance a clone first. Authority is committed only after the
            // resulting delta has entered the reliable event queue.
            let mut candidate = self.session.clone();
            let mut wire = Vec::new();
            if let Err(error) =
                candidate.advance(|record| self.strategy.apply(record, command), &mut wire)
            {
                self.last_error = Some(error);
                break;
            }
            let delta = encode_delta(&wire);
            match self.mux.send(NetChannel::EventsReliable, &delta) {
                Ok(_) => {
                    self.session = candidate;
                    self.pending_commands.pop_front();
                    self.applied = self.applied.saturating_add(1);
                    advanced += 1;
                }
                Err(error @ MuxSendError::Backpressured(_)) => {
                    self.last_send_error = Some(error);
                    break;
                }
                Err(error @ MuxSendError::TooLarge { .. }) => {
                    // The command was never committed. Drop this unrepresentable
                    // result so it cannot permanently block later commands.
                    self.last_send_error = Some(error);
                    self.pending_commands.pop_front();
                    self.rejected = self.rejected.saturating_add(1);
                }
                Err(error @ MuxSendError::ConnectionNotEstablished) => {
                    self.last_send_error = Some(error);
                    break;
                }
            }
        }
        advanced
    }

    fn drain_control(&mut self) {
        while self.pending_commands.len() < self.config.mux.reliable_queue.max(1) {
            let Some(bytes) = self.mux.recv(NetChannel::RpcReliable) else {
                break;
            };
            let envelope = match decode_rpc(&bytes) {
                Ok(envelope) => envelope,
                Err(error) => {
                    self.reject(error, true);
                    break;
                }
            };
            match envelope {
                RpcEnvelope::Hello(remote) => {
                    if !matches!(self.connection, ConnectionState::Handshaking) {
                        self.reject(ProtocolError::UnexpectedHandshake, true);
                        break;
                    }
                    match negotiate_protocol(self.config.protocol, remote) {
                        Ok(negotiated) => {
                            let accept = encode_accept(negotiated);
                            match self.mux.send(NetChannel::RpcReliable, &accept) {
                                Ok(_) => {
                                    self.connection = ConnectionState::Established(negotiated);
                                }
                                Err(error) => {
                                    self.last_send_error = Some(error);
                                    break;
                                }
                            }
                        }
                        Err(error) => {
                            self.reject(error, true);
                            break;
                        }
                    }
                }
                RpcEnvelope::Payload(payload) => {
                    if !matches!(self.connection, ConnectionState::Established(_)) {
                        self.reject(
                            ProtocolError::TrafficBeforeHandshake {
                                channel: NetChannel::RpcReliable,
                            },
                            true,
                        );
                        break;
                    }
                    self.pending_commands.push_back(payload.to_vec());
                }
                RpcEnvelope::Accept(_) | RpcEnvelope::Reject(_) => {
                    self.reject(ProtocolError::UnexpectedHandshake, true);
                    break;
                }
            }
        }
    }

    fn drain_auxiliary_channels(&mut self) {
        while self.inbound_events.len() < self.config.mux.reliable_queue.max(1) {
            let Some(bytes) = self.mux.recv(NetChannel::EventsReliable) else {
                break;
            };
            match decode_event(&bytes) {
                Ok(EventEnvelope::Payload(event)) => {
                    self.inbound_events.push_back(event.to_vec());
                }
                Ok(EventEnvelope::Delta(_) | EventEnvelope::Snapshot(_)) => {
                    self.reject(
                        ProtocolError::MalformedMessage {
                            channel: NetChannel::EventsReliable,
                        },
                        true,
                    );
                    break;
                }
                Err(error) => {
                    self.reject(error, true);
                    break;
                }
            }
        }
        if !matches!(self.connection, ConnectionState::Established(_)) {
            return;
        }
        while let Some(bytes) = self.mux.recv(NetChannel::StateUnreliable) {
            match decode_state(&bytes) {
                Ok(state) => self.newest_state = Some(state.to_vec()),
                Err(error) => {
                    self.reject(error, true);
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
    }

    fn reject(&mut self, error: ProtocolError, notify_peer: bool) {
        if notify_peer {
            let rejection = encode_reject(error);
            if let Err(send_error) = self.mux.send(NetChannel::RpcReliable, &rejection) {
                self.last_send_error = Some(send_error);
            }
        }
        self.connection = ConnectionState::Rejected(error);
        self.last_protocol_error = Some(error);
    }

    /// Send the framed full snapshot to the peer — the seed a joining or
    /// resyncing client applies before any delta. Returns the seeded revision, or
    /// `Ok(None)` on backpressure, MTU overflow or connection rejection; use
    /// [`take_send_error`](Self::take_send_error) to distinguish these cases.
    /// Snapshots share the event lane with deltas to preserve their order.
    ///
    /// The frame is a full **snapshot**, not a delta, so the client must apply it
    /// with [`ClientSession::reseed`](crate::session::ClientSession::reseed) rather
    /// than [`receive`](crate::session::ClientSession::receive).
    ///
    /// # Errors
    /// [`SessionError::NoBaseline`] before any authority exists, or a store error.
    pub fn seed(&mut self) -> Result<Option<u32>, SessionError> {
        if matches!(self.connection, ConnectionState::Rejected(_)) {
            self.last_send_error = Some(MuxSendError::ConnectionNotEstablished);
            return Ok(None);
        }
        let mut wire = Vec::new();
        let revision = self.session.snapshot_frame(&mut wire)?;
        let snapshot = encode_snapshot(&wire);
        match self.mux.send(NetChannel::EventsReliable, &snapshot) {
            Ok(_) => Ok(Some(revision)),
            Err(error) => {
                self.last_send_error = Some(error);
                Ok(None)
            }
        }
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

    /// Queue a reliable authoritative event independently of RPC traffic.
    pub fn send_event(&mut self, bytes: &[u8]) -> Result<QueueOutcome, MuxSendError> {
        self.require_established()?;
        let event = encode_event(bytes);
        let result = self.mux.send(NetChannel::EventsReliable, &event);
        self.record_send(result)
    }

    /// Queue a reliable RPC response/control payload.
    pub fn send_rpc(&mut self, bytes: &[u8]) -> Result<u64, MuxSendError> {
        self.require_established()?;
        let rpc = encode_rpc(bytes);
        let result = self.mux.send(NetChannel::RpcReliable, &rpc);
        self.record_send(result)?;
        let id = self.next_rpc_id;
        self.next_rpc_id = self.next_rpc_id.wrapping_add(1);
        Ok(id)
    }

    /// Queue newest-wins transient state for the client.
    pub fn send_state(&mut self, bytes: &[u8]) -> Result<QueueOutcome, MuxSendError> {
        self.require_established()?;
        let state = encode_state(bytes);
        let result = self.mux.send(NetChannel::StateUnreliable, &state);
        self.record_send(result)
    }

    /// Queue one latency-sensitive input on the bounded unreliable input lane.
    /// Older unsent inputs may be discarded when the lane is full. Requires an
    /// established connection.
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

    /// Take one client event/proposal.
    pub fn recv_event(&mut self) -> Option<Vec<u8>> {
        self.inbound_events.pop_front()
    }

    /// Take the newest client state/input-side payload.
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

    /// The current authoritative revision.
    #[must_use]
    pub fn revision(&self) -> Option<u32> {
        self.session.revision()
    }

    /// The current authoritative record.
    #[must_use]
    pub fn record(&self) -> Option<&Record> {
        self.session.record()
    }

    /// How many commands have advanced authority since construction.
    #[must_use]
    pub fn applied(&self) -> u64 {
        self.applied
    }

    /// How many inbound commands could not be decoded (a protocol mismatch).
    #[must_use]
    pub fn rejected(&self) -> u64 {
        self.rejected
    }

    /// Whether the outbound send window is full.
    #[must_use]
    pub fn is_full(&self) -> bool {
        matches!(self.connection, ConnectionState::Rejected(_))
            || self.mux.is_backpressured(NetChannel::EventsReliable)
    }

    /// How many event-lane messages (snapshots, deltas and application events)
    /// are pending or unacknowledged.
    #[must_use]
    pub fn unacked(&self) -> usize {
        self.mux.queued(NetChannel::EventsReliable)
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

/// A `prism_app` plugin that installs a [`ServerReplication`] resource and pumps
/// it every frame.
pub struct ReplicationServerPlugin<S: ServerStrategy> {
    schema: Schema,
    capacity: usize,
    snapshot: Vec<u8>,
    /// Taken by [`Plugin::build`]; `Plugin::is_unique` guarantees one build.
    strategy: Mutex<Option<S>>,
    transport: Mutex<Option<Box<dyn Transport>>>,
    config: ReplicationConfig,
}

impl<S: ServerStrategy> ReplicationServerPlugin<S> {
    /// A plugin that seeds authority from `snapshot` and drives it over
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

impl<S: ServerStrategy> Plugin for ReplicationServerPlugin<S> {
    fn build(&self, app: &mut App) {
        let strategy = self
            .strategy
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
            .expect("ReplicationServerPlugin::build ran twice");
        let transport = self
            .transport
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
            .expect("ReplicationServerPlugin::build ran twice");
        let server = ServerReplication::<S>::new_with_config(
            self.schema.clone(),
            self.capacity,
            &self.snapshot,
            strategy,
            transport,
            self.config,
        )
        .expect("authority seed snapshot is malformed");
        app.insert_resource(server);
        app.add_systems(First, pump_server::<S>);
    }
}

/// The per-frame system: apply inbound commands and broadcast authority's deltas.
fn pump_server<S: ServerStrategy>(mut server: ResMut<ServerReplication<S>>) {
    server.pump();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::delta::{encode_snapshot, FieldKind, FieldValue};
    use crate::plugin::{Replication, ReplicationPlugin, ReplicationStrategy};
    use crate::transport::memory::InMemoryTransport;
    use alloc::vec;

    fn schema() -> Schema {
        Schema::new(vec![FieldKind::Int { bits: 12 }])
    }

    fn snapshot_at(revision: u32, value: i64) -> Vec<u8> {
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

    /// Authority's command wire format is one little-endian `i64` addend.
    struct AddAuthority;
    impl ServerStrategy for AddAuthority {
        type Command = i64;
        fn decode(&self, bytes: &[u8]) -> Option<i64> {
            <[u8; 8]>::try_from(bytes).ok().map(i64::from_le_bytes)
        }
        fn apply(&self, record: &mut Record, command: i64) {
            record.set(0, FieldValue::Int(int(record).saturating_add(command)));
        }
    }

    /// The client's prediction mirrors the authority's apply.
    struct AddClient;
    impl ReplicationStrategy for AddClient {
        type Input = i64;
        fn simulate(&self, record: &mut Record, input: &i64) {
            record.set(0, FieldValue::Int(int(record).saturating_add(*input)));
        }
        fn encode(&self, input: &i64) -> Vec<u8> {
            input.to_le_bytes().to_vec()
        }
    }

    fn drive(client: &mut Replication<AddClient>, server: &mut ServerReplication<AddAuthority>) {
        for _ in 0..8 {
            client.pump();
            server.pump();
        }
    }

    fn memory_pair(
        client_config: ReplicationConfig,
        server_config: ReplicationConfig,
    ) -> (Replication<AddClient>, ServerReplication<AddAuthority>) {
        let (a, b) = InMemoryTransport::pair();
        (
            Replication::new_with_config(
                schema(),
                8,
                &snapshot_at(0, 0),
                AddClient,
                Box::new(a),
                client_config,
            )
            .unwrap(),
            ServerReplication::new_with_config(
                schema(),
                8,
                &snapshot_at(0, 0),
                AddAuthority,
                Box::new(b),
                server_config,
            )
            .unwrap(),
        )
    }

    /// Exercise the actual app scheduling seam, with the same assertions for
    /// memory and real sockets. Polls are bounded even if delivery fails.
    fn exercise_plugins(
        client_end: Box<dyn Transport>,
        server_end: Box<dyn Transport>,
        wait: bool,
    ) {
        let mut client_app = App::new();
        client_app.add_plugins(ReplicationPlugin::new(
            schema(),
            8,
            snapshot_at(0, 0),
            AddClient,
            client_end,
        ));
        let mut server_app = App::new();
        server_app.add_plugins(ReplicationServerPlugin::new(
            schema(),
            8,
            snapshot_at(0, 0),
            AddAuthority,
            server_end,
        ));
        for _ in 0..1_000 {
            client_app.update();
            server_app.update();
            if client_app
                .world()
                .resource::<Replication<AddClient>>()
                .negotiated()
                .is_some()
            {
                break;
            }
            if wait {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
        assert_eq!(
            client_app
                .world()
                .resource::<Replication<AddClient>>()
                .negotiated(),
            server_app
                .world()
                .resource::<ServerReplication<AddAuthority>>()
                .negotiated()
        );
        assert!(client_app
            .world()
            .resource::<Replication<AddClient>>()
            .negotiated()
            .is_some());
        {
            let client = client_app
                .world_mut()
                .resource_mut::<Replication<AddClient>>();
            assert_eq!(client.predict_and_send(3).unwrap(), Some(0));
            client.send_event(b"client event").unwrap();
            client.send_state(b"client old").unwrap();
            assert_eq!(
                client.send_state(b"client state"),
                Ok(QueueOutcome::ReplacedOlder)
            );
        }
        {
            let server = server_app
                .world_mut()
                .resource_mut::<ServerReplication<AddAuthority>>();
            server.send_rpc(b"server rpc").unwrap();
            server.send_event(b"server event").unwrap();
            server.send_state(b"server old").unwrap();
            assert_eq!(
                server.send_state(b"server state"),
                Ok(QueueOutcome::ReplacedOlder)
            );
        }
        let (mut ce, mut cr, mut cs, mut se, mut ss) = (None, None, None, None, None);
        for _ in 0..1_000 {
            client_app.update();
            server_app.update();
            let client = client_app
                .world_mut()
                .resource_mut::<Replication<AddClient>>();
            ce = ce.or_else(|| client.recv_event());
            cr = cr.or_else(|| client.recv_rpc());
            cs = cs.or_else(|| client.recv_state());
            let acked = client.acked();
            assert_eq!(client.take_last_error(), None);
            assert_eq!(client.take_protocol_error(), None);
            let server = server_app
                .world_mut()
                .resource_mut::<ServerReplication<AddAuthority>>();
            se = se.or_else(|| server.recv_event());
            ss = ss.or_else(|| server.recv_state());
            assert_eq!(server.take_last_error(), None);
            assert_eq!(server.take_protocol_error(), None);
            if acked == 1
                && ce.is_some()
                && cr.is_some()
                && cs.is_some()
                && se.is_some()
                && ss.is_some()
            {
                break;
            }
            if wait {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
        assert_eq!(ce, Some(b"server event".to_vec()));
        assert_eq!(cr, Some(b"server rpc".to_vec()));
        assert_eq!(cs, Some(b"server state".to_vec()));
        assert_eq!(se, Some(b"client event".to_vec()));
        assert_eq!(ss, Some(b"client state".to_vec()));
        let client = client_app.world().resource::<Replication<AddClient>>();
        let server = server_app
            .world()
            .resource::<ServerReplication<AddAuthority>>();
        assert_eq!(client.acked(), 1);
        assert_eq!(client.predicted(), 1);
        assert_eq!(int(client.state().unwrap()), 3);
        assert_eq!(server.applied(), 1);
        assert_eq!(server.revision(), Some(1));
    }

    #[test]
    fn memory_plugins_exchange_events_rpcs_state_and_prediction() {
        let (a, b) = InMemoryTransport::pair();
        exercise_plugins(Box::new(a), Box::new(b), false);
    }

    #[test]
    fn plugins_deliver_input_and_bulk_channels_without_protocol_rejection() {
        let (mut client, mut server) = memory_pair(
            ReplicationConfig::for_capacity(8),
            ReplicationConfig::for_capacity(8),
        );
        drive(&mut client, &mut server);

        assert_eq!(client.send_input(b"client input"), Ok(QueueOutcome::Queued));
        assert_eq!(client.send_bulk(b"client bulk"), Ok(QueueOutcome::Queued));
        assert_eq!(server.send_input(b"server input"), Ok(QueueOutcome::Queued));
        assert_eq!(server.send_bulk(b"server bulk"), Ok(QueueOutcome::Queued));

        for _ in 0..16 {
            client.pump();
            server.pump();
        }

        assert_eq!(server.recv_input(), Some(b"client input".to_vec()));
        assert_eq!(server.recv_bulk(), Some(b"client bulk".to_vec()));
        assert_eq!(client.recv_input(), Some(b"server input".to_vec()));
        assert_eq!(client.recv_bulk(), Some(b"server bulk".to_vec()));
        assert_eq!(client.take_protocol_error(), None);
        assert_eq!(server.take_protocol_error(), None);
    }

    #[test]
    fn loopback_udp_plugins_exchange_events_rpcs_state_and_prediction() {
        fn bind() -> Option<crate::udp::UdpTransport> {
            match crate::udp::UdpTransport::bind("127.0.0.1:0") {
                Ok(socket) => Some(socket),
                Err(error) => {
                    assert!(
                        std::env::var_os("PRISM_REQUIRE_NET").is_none(),
                        "required UDP bind failed: {error}"
                    );
                    std::eprintln!("skipping replication UDP test: {error}");
                    None
                }
            }
        }
        let Some(mut a) = bind() else {
            return;
        };
        let Some(mut b) = bind() else {
            return;
        };
        a.connect_to(b.local_addr().unwrap()).unwrap();
        b.connect_to(a.local_addr().unwrap()).unwrap();
        exercise_plugins(Box::new(a), Box::new(b), true);
    }

    #[test]
    fn incompatible_versions_are_rejected_on_both_endpoints() {
        let mut client_config = ReplicationConfig::for_capacity(8);
        client_config.protocol.version += 1;
        let server_config = ReplicationConfig::for_capacity(8);
        let (mut client, mut server) = memory_pair(client_config, server_config);
        client.send(&3i64.to_le_bytes()).unwrap();
        drive(&mut client, &mut server);
        assert_eq!(
            client.connection_state(),
            ConnectionState::Rejected(ProtocolError::VersionMismatch {
                local: client_config.protocol.version,
                remote: server_config.protocol.version
            })
        );
        assert_eq!(
            server.connection_state(),
            ConnectionState::Rejected(ProtocolError::VersionMismatch {
                local: server_config.protocol.version,
                remote: client_config.protocol.version
            })
        );
        assert_eq!(server.applied(), 0);
        assert_eq!(
            client.try_send_rpc(b"x"),
            Err(MuxSendError::ConnectionNotEstablished)
        );
        assert_eq!(
            client.send_event(b"x"),
            Err(MuxSendError::ConnectionNotEstablished)
        );
        assert_eq!(
            server.send_state(b"x"),
            Err(MuxSendError::ConnectionNotEstablished)
        );
    }

    #[test]
    fn zero_rollback_offer_is_rejected_and_never_applies_commands() {
        let (a, b) = InMemoryTransport::pair();
        let config = ReplicationConfig::for_capacity(8);
        let mut peer = ChannelMux::new(a, config.mux);
        let mut server = ServerReplication::new_with_config(
            schema(),
            8,
            &snapshot_at(0, 0),
            AddAuthority,
            Box::new(b),
            config,
        )
        .unwrap();
        peer.send(
            NetChannel::RpcReliable,
            &super::super::encode_hello(crate::multiplex::ProtocolOffer::current(0)),
        )
        .unwrap();
        peer.send(NetChannel::RpcReliable, &encode_rpc(&3i64.to_le_bytes()))
            .unwrap();
        for _ in 0..8 {
            peer.tick();
            server.pump();
        }
        assert_eq!(
            server.connection_state(),
            ConnectionState::Rejected(ProtocolError::InvalidRollbackWindow { offered: 0 })
        );
        assert_eq!(server.applied(), 0);
        let reply = peer.recv(NetChannel::RpcReliable).unwrap();
        assert!(matches!(
            decode_rpc(&reply),
            Ok(RpcEnvelope::Reject(ProtocolError::InvalidRollbackWindow {
                offered: 0
            }))
        ));
    }

    #[test]
    fn rollback_window_preserves_the_acked_baseline_and_reopens_after_reconciliation() {
        let mut config = ReplicationConfig::for_capacity(8);
        config.protocol.rollback_window = 3;
        let (mut client, mut server) = memory_pair(config, ReplicationConfig::for_capacity(8));
        drive(&mut client, &mut server);
        assert_eq!(client.negotiated().unwrap().rollback_window, 3);
        assert_eq!(client.predict_and_send(1).unwrap(), Some(0));
        assert_eq!(client.predict_and_send(2).unwrap(), Some(1));
        assert_eq!(
            client.predict_and_send(4),
            Err(SessionError::RollbackWindowFull {
                pending: 2,
                window: 3
            })
        );
        assert_eq!(client.predicted(), 2);
        drive(&mut client, &mut server);
        assert_eq!(client.acked(), 2);
        assert_eq!(int(client.state().unwrap()), 3);
        assert_eq!(client.predict_and_send(4).unwrap(), Some(2));
    }

    #[test]
    fn reliable_event_backpressure_does_not_advance_authority() {
        let (a, b) = InMemoryTransport::pair();
        let mut config = ReplicationConfig::for_capacity(8);
        config.mux.reliable_queue = 1;
        config.mux.reliable_window = 1;
        let mut peer = ChannelMux::new(a, config.mux);
        let mut server = ServerReplication::new_with_config(
            schema(),
            8,
            &snapshot_at(0, 0),
            AddAuthority,
            Box::new(b),
            config,
        )
        .unwrap();
        peer.send(
            NetChannel::RpcReliable,
            &super::super::encode_hello(config.protocol),
        )
        .unwrap();
        for _ in 0..8 {
            peer.tick();
            server.pump();
        }
        assert!(server.negotiated().is_some());
        peer.recv(NetChannel::RpcReliable).unwrap();
        peer.send(NetChannel::RpcReliable, &encode_rpc(&3i64.to_le_bytes()))
            .unwrap();
        peer.tick();
        server.send_event(b"in flight").unwrap();
        // Tick only the mux: receive the command and send the event without
        // allowing the peer to acknowledge it or applying the command yet.
        server.mux.tick();
        server.send_event(b"queued").unwrap();
        assert_eq!(server.pump(), 0);
        assert_eq!(server.revision(), Some(0));
        assert_eq!(server.pending_commands.len(), 1);
        assert_eq!(
            server.take_send_error(),
            Some(MuxSendError::Backpressured(NetChannel::EventsReliable))
        );
        for _ in 0..12 {
            peer.tick();
            while peer.recv(NetChannel::EventsReliable).is_some() {}
            server.pump();
        }
        assert_eq!(server.applied(), 1);
        assert_eq!(server.revision(), Some(1));
        assert_eq!(int(server.record().unwrap()), 3);
    }

    #[test]
    fn malformed_event_rejects_before_queued_commands_can_mutate_authority() {
        let (a, b) = InMemoryTransport::pair();
        let config = ReplicationConfig::for_capacity(8);
        let mut peer = ChannelMux::new(a, config.mux);
        let mut server =
            ServerReplication::new(schema(), 8, &snapshot_at(0, 0), AddAuthority, Box::new(b))
                .unwrap();
        peer.send(
            NetChannel::RpcReliable,
            &super::super::encode_hello(config.protocol),
        )
        .unwrap();
        for _ in 0..8 {
            peer.tick();
            server.pump();
        }
        peer.send(NetChannel::RpcReliable, &encode_rpc(&3i64.to_le_bytes()))
            .unwrap();
        peer.send(NetChannel::EventsReliable, &[255]).unwrap();
        peer.tick();
        assert_eq!(server.pump(), 0);
        assert_eq!(server.revision(), Some(0));
        assert_eq!(
            server.connection_state(),
            ConnectionState::Rejected(ProtocolError::MalformedMessage {
                channel: NetChannel::EventsReliable
            })
        );
    }

    /// The full loop, both halves as real `prism_app` plugins over the mux:
    /// a client's command reaches authority, authority advances and
    /// broadcasts the delta, and the client's prediction reconciles to it — all
    /// driven by the two frame loops, no manual pumping.
    #[test]
    fn a_client_command_reaches_authority_and_its_delta_reconciles_the_client() {
        let (client_end, server_end) = InMemoryTransport::pair();

        let mut client_app = App::new();
        client_app.add_plugins(ReplicationPlugin::new(
            schema(),
            8,
            snapshot_at(0, 0),
            AddClient,
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

        // The client issues one command (no local prediction this time).
        assert!(client_app
            .world_mut()
            .resource_mut::<Replication<AddClient>>()
            .send(&3i64.to_le_bytes())
            .is_some());

        // Alternate the two frame loops. The first client frame puts the command
        // on the wire; the server frame applies it and replies with the delta; the
        // next client frame reconciles.
        for _ in 0..8 {
            client_app.update();
            server_app.update();
        }

        let server = server_app
            .world()
            .resource::<ServerReplication<AddAuthority>>();
        assert_eq!(
            server.revision(),
            Some(1),
            "authority advanced exactly once"
        );
        assert_eq!(server.applied(), 1);
        assert_eq!(int(server.record().unwrap()), 3);

        let replication = client_app.world().resource::<Replication<AddClient>>();
        assert_eq!(replication.acked(), 1, "client reconciled to authority");
        assert_eq!(int(replication.state().unwrap()), 3);
    }

    /// A command whose wire format is nonsense is ignored, not applied — the
    /// authority's revision does not move.
    #[test]
    fn a_malformed_command_does_not_advance_authority() {
        let (client_end, server_end) = InMemoryTransport::pair();
        let mut client_app = App::new();
        client_app.add_plugins(ReplicationPlugin::new(
            schema(),
            8,
            snapshot_at(0, 0),
            AddClient,
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

        assert!(client_app
            .world_mut()
            .resource_mut::<Replication<AddClient>>()
            .send(b"not eight bytes")
            .is_some());
        for _ in 0..4 {
            client_app.update();
            server_app.update();
        }
        let server = server_app
            .world()
            .resource::<ServerReplication<AddAuthority>>();
        assert_eq!(server.revision(), Some(0), "the command was ignored");
        assert_eq!(server.applied(), 0);
        assert_eq!(server.rejected(), 1, "the undecodable command is counted");
    }

    /// `seed()` produces a full-snapshot frame a client applies with `reseed` — the
    /// resync path `receive` (which only understands deltas) cannot serve.
    #[test]
    fn a_seed_frame_reseeds_a_client_to_the_authoritative_revision() {
        let (client_end, server_end) = InMemoryTransport::pair();
        let mut server = ServerReplication::new(
            schema(),
            8,
            &snapshot_at(5, 50),
            AddAuthority,
            Box::new(server_end),
        )
        .unwrap();

        let mut client = Replication::new(
            schema(),
            8,
            &snapshot_at(0, 0),
            AddClient,
            Box::new(client_end),
        )
        .unwrap();

        assert_eq!(server.seed().unwrap(), Some(5));
        for _ in 0..8 {
            client.pump();
            server.pump();
        }
        assert_eq!(client.acked(), 5);
        assert_eq!(int(client.state().unwrap()), 50);

        // A later delta must follow the snapshot on the same ordered lane.
        client.send(&2i64.to_le_bytes()).unwrap();
        for _ in 0..8 {
            client.pump();
            server.pump();
        }
        assert_eq!(client.acked(), 6);
        assert_eq!(int(client.state().unwrap()), 52);
        assert_eq!(client.take_last_error(), None);
    }

    /// A **predicted** command whose simulation matches authority is confirmed, not
    /// rolled back: the client predicts +3, authority applies +3, and after
    /// reconciliation the client is still at the predicted value with nothing
    /// pending.
    #[test]
    fn a_predicted_command_that_matches_authority_is_not_rolled_back() {
        let (client_end, server_end) = InMemoryTransport::pair();
        let mut client_app = App::new();
        client_app.add_plugins(ReplicationPlugin::new(
            schema(),
            8,
            snapshot_at(0, 0),
            AddClient,
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

        // The client predicts and sends in one call.
        let replication = client_app
            .world_mut()
            .resource_mut::<Replication<AddClient>>();
        assert_eq!(replication.predict_and_send(3).unwrap(), Some(0));
        assert_eq!(replication.predicted(), 1);
        assert_eq!(int(replication.state().unwrap()), 3);

        for _ in 0..8 {
            client_app.update();
            server_app.update();
        }

        let replication = client_app.world().resource::<Replication<AddClient>>();
        assert_eq!(replication.acked(), 1, "authority confirmed the prediction");
        assert_eq!(replication.predicted(), 1, "no replay left pending");
        assert_eq!(int(replication.state().unwrap()), 3, "not rolled back");
    }
}
