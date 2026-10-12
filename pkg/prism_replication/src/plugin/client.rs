//! The client half of the app seam: predict locally, reconcile authority.

use alloc::boxed::Box;
use alloc::vec::Vec;
use std::sync::Mutex;

use prism_app::{App, First, Plugin};
use prism_ecs::resource::Resource;
use prism_ecs::system::ResMut;

use crate::delta::{Record, Schema};
use crate::reliable::ReliableChannel;
use crate::session::{ClientSession, SessionError};
use crate::transport::Transport;

use super::{DEFAULT_MIN_RTO, DEFAULT_WINDOW};

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

/// The app-facing replication resource.
pub struct Replication<S: ReplicationStrategy> {
    session: ClientSession<S::Input>,
    strategy: S,
    channel: ReliableChannel<Box<dyn Transport>>,
    /// The first error from the most recent [`pump`](Self::pump), readable by the
    /// app; a malformed or rejected frame must not panic the frame loop.
    last_error: Option<SessionError>,
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
        Ok(Self {
            session: ClientSession::new(schema, capacity, snapshot)?,
            strategy,
            channel: ReliableChannel::new(transport, DEFAULT_MIN_RTO, DEFAULT_WINDOW),
            last_error: None,
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
        self.last_error = None;
        self.channel.tick();
        let Self {
            session,
            strategy,
            channel,
            last_error,
        } = self;
        let mut applied = 0;
        while let Some(bytes) = channel.recv() {
            match session.receive(&bytes, |record, input| strategy.simulate(record, input)) {
                Ok(revisions) => applied += revisions.len(),
                Err(error) => {
                    *last_error = Some(error);
                    break;
                }
            }
        }
        applied
    }

    /// Take the error the last [`pump`](Self::pump) stopped on, clearing it.
    pub fn take_last_error(&mut self) -> Option<SessionError> {
        self.last_error.take()
    }

    /// Predict one local input immediately.
    ///
    /// # Errors
    /// [`SessionError::Prediction`] when the predicted state is not retained.
    pub fn predict(&mut self, input: S::Input) -> Result<u32, SessionError> {
        let Self { session, strategy, .. } = self;
        session.predict(input, |record, input| strategy.simulate(record, input))
    }

    /// Predict `input` locally **and** send it to authority in one call — the
    /// client half of the command path.
    ///
    /// Returns the sent sequence number, or `Ok(None)` when the send window is
    /// full (backpressure). A full window predicts nothing, so a caller that
    /// retries the same input cannot double-predict it; check
    /// [`is_full`](Self::is_full) to avoid the `None` case. Prediction is local
    /// and independent of delivery — authority will confirm or correct it later.
    ///
    /// # Errors
    /// [`SessionError::Prediction`] when the predicted state is not retained.
    pub fn predict_and_send(&mut self, input: S::Input) -> Result<Option<u64>, SessionError> {
        if self.channel.is_full() {
            return Ok(None);
        }
        let bytes = self.strategy.encode(&input);
        let Self {
            session,
            strategy,
            channel,
            ..
        } = self;
        session.predict(input, |record, input| strategy.simulate(record, input))?;
        Ok(channel.send(&bytes))
    }

    /// Send `bytes` to the peer reliably: the channel numbers and retains it until
    /// the peer acknowledges, retransmitting on later frames if a datagram drops.
    /// Returns the sequence number, or `None` when the send window is full (the
    /// app should retry a later frame — check [`is_full`](Self::is_full) or
    /// [`unacked`](Self::unacked) first).
    pub fn send(&mut self, bytes: &[u8]) -> Option<u64> {
        self.channel.send(bytes)
    }

    /// Whether the outbound send window is full (a [`send`](Self::send) would
    /// return `None`); the app should not produce more input until
    /// acknowledgements return.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.channel.is_full()
    }

    /// How many outbound messages are still unacknowledged.
    #[must_use]
    pub fn unacked(&self) -> usize {
        self.channel.unacked()
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
        }
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
        let replication = Replication::<S>::new(
            self.schema.clone(),
            self.capacity,
            &self.snapshot,
            strategy,
            transport,
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
mod tests {
    use super::*;
    use crate::delta::{FieldKind, FieldValue, encode_snapshot};
    use crate::session::ServerSession;
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

    /// The whole point of the seam: a real `App` with the plugin installed
    /// predicts locally, and its `First` system drains an authoritative frame from
    /// the peer's reliable channel and reconciles the prediction — no manual
    /// pumping, and the frame is `ACKed` back to authority.
    #[test]
    fn the_plugin_pumps_authoritative_frames_into_the_prediction() {
        let (client_end, server_end) = InMemoryTransport::pair();
        // The authority speaks the same reliable channel as the client, so the
        // datagrams on the wire are real ARQ packets.
        let mut server = ServerSession::new(schema(), 8, &snapshot_at(0, 0)).unwrap();
        let mut server_channel =
            ReliableChannel::new(server_end, DEFAULT_MIN_RTO, DEFAULT_WINDOW);

        let mut app = App::new();
        app.add_plugins(ReplicationPlugin::new(
            schema(),
            8,
            snapshot_at(0, 0),
            Add,
            Box::new(client_end),
        ));

        // The client predicts +5 immediately.
        assert_eq!(
            app.world_mut()
                .resource_mut::<Replication<Add>>()
                .predict(5)
            .unwrap(),
            1
        );
        assert_eq!(
            int(app.world().resource::<Replication<Add>>().state().unwrap()),
            5
        );

        // Authority applies +3 and sends the framed delta over its reliable
        // channel; one tick puts the packet on the wire.
        let mut wire = Vec::new();
        server
            .advance(
                |record| record.set(0, FieldValue::Int(int(record) + 3)),
                &mut wire,
            )
            .unwrap();
        assert!(server_channel.send(&wire).is_some());
        server_channel.tick();

        // One frame: the plugin's `First` system ticks the channel, recovers the
        // frame (ACKing it back), then reconciles the prediction.
        app.update();

        let replication = app.world().resource::<Replication<Add>>();
        assert_eq!(replication.acked(), 1, "authority confirmed");
        assert_eq!(int(replication.state().unwrap()), 3, "prediction rolled back");

        // The client's ACK reached authority: one more tick retires the message.
        server_channel.tick();
        assert_eq!(
            server_channel.unacked(),
            0,
            "authority's frame was acknowledged over the reliable channel"
        );
    }

    /// `predict_and_send` is the client half of the command path: it predicts the
    /// input locally and puts the `encode`d command on the wire in one call.
    #[test]
    fn predict_and_send_predicts_and_puts_the_encoded_command_on_the_wire() {
        let (client_end, mut server_end) = InMemoryTransport::pair();
        let mut app = App::new();
        app.add_plugins(ReplicationPlugin::new(
            schema(),
            8,
            snapshot_at(0, 0),
            Add,
            Box::new(client_end),
        ));

        let replication = app.world_mut().resource_mut::<Replication<Add>>();
        assert_eq!(replication.predict_and_send(3).unwrap(), Some(0));
        assert_eq!(replication.predicted(), 1, "predicted locally");
        assert_eq!(int(replication.state().unwrap()), 3);

        // One frame ticks the channel, putting the command packet on the wire.
        app.update();
        let packet = server_end.recv().expect("the command reached the peer");
        let (kind, _seq, payload) = crate::reliable::decode(&packet).expect("a packet");
        assert_eq!(kind, 0, "a data packet");
        assert_eq!(payload, &3i64.to_le_bytes(), "the command's wire format");
    }

    /// Backpressure: a full send window refuses the command **without** predicting
    /// it, so a caller that retries the same input cannot double-predict it.
    #[test]
    fn predict_and_send_refuses_without_predicting_when_the_window_is_full() {
        let (client_end, _server_end) = InMemoryTransport::pair();
        let mut app = App::new();
        app.add_plugins(ReplicationPlugin::new(
            schema(),
            8,
            snapshot_at(0, 0),
            Add,
            Box::new(client_end),
        ));

        let replication = app.world_mut().resource_mut::<Replication<Add>>();
        for _ in 0..DEFAULT_WINDOW {
            assert!(replication.send(b"x").is_some());
        }
        assert!(replication.is_full());
        assert_eq!(
            replication.predict_and_send(3).unwrap(),
            None,
            "the window is full: nothing sent"
        );
        assert_eq!(replication.predicted(), 0, "and nothing predicted either");
    }
}
