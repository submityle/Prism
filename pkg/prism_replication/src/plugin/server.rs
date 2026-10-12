//! The authority half of the app seam: apply inbound commands, broadcast deltas.

use alloc::boxed::Box;
use alloc::vec::Vec;
use std::sync::Mutex;

use prism_app::{App, First, Plugin};
use prism_ecs::resource::Resource;
use prism_ecs::system::ResMut;

use crate::delta::{Record, Schema};
use crate::reliable::ReliableChannel;
use crate::session::{ServerSession, SessionError};
use crate::transport::Transport;

use super::{DEFAULT_MIN_RTO, DEFAULT_WINDOW};

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
    channel: ReliableChannel<Box<dyn Transport>>,
    /// The first error from the most recent [`pump`](Self::pump), readable by the
    /// app; a rejected command must not panic the frame loop.
    last_error: Option<SessionError>,
    applied: u64,
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
        Ok(Self {
            session: ServerSession::new(schema, capacity, snapshot)?,
            strategy,
            channel: ReliableChannel::new(transport, DEFAULT_MIN_RTO, DEFAULT_WINDOW),
            last_error: None,
            applied: 0,
        })
    }

    /// One frame: tick the channel, decode + apply every inbound command, and
    /// broadcast the framed authoritative delta back to the peer. Returns how many
    /// commands advanced authority this frame. A rejected command is stored for
    /// [`take_last_error`](Self::take_last_error) and stops the drain rather than
    /// unwinding the frame loop.
    pub fn pump(&mut self) -> usize {
        self.last_error = None;
        self.channel.tick();
        let Self {
            session,
            strategy,
            channel,
            last_error,
            applied,
        } = self;
        // Collect inbound messages first: `channel.recv` borrows the channel, and
        // the broadcast below needs it again.
        let mut inbox = Vec::new();
        while let Some(bytes) = channel.recv() {
            inbox.push(bytes);
        }
        let mut advanced = 0;
        for bytes in inbox {
            let Some(command) = strategy.decode(&bytes) else {
                continue;
            };
            let mut wire = Vec::new();
            match session.advance(|record| strategy.apply(record, command), &mut wire) {
                Ok(_) => {
                    channel.send(&wire);
                    *applied += 1;
                    advanced += 1;
                }
                Err(error) => {
                    *last_error = Some(error);
                    break;
                }
            }
        }
        advanced
    }

    /// Send the framed full snapshot to the peer — the seed a joining or
    /// resyncing client applies before any delta.
    ///
    /// # Errors
    /// [`SessionError::NoBaseline`] before any authority exists, or a store error.
    pub fn seed(&mut self) -> Result<u32, SessionError> {
        let Self { session, channel, .. } = self;
        let mut wire = Vec::new();
        let revision = session.snapshot_frame(&mut wire)?;
        channel.send(&wire);
        Ok(revision)
    }

    /// Take the error the last [`pump`](Self::pump) stopped on, clearing it.
    pub fn take_last_error(&mut self) -> Option<SessionError> {
        self.last_error.take()
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

    /// Whether the outbound send window is full.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.channel.is_full()
    }

    /// How many outbound messages are still unacknowledged.
    #[must_use]
    pub fn unacked(&self) -> usize {
        self.channel.unacked()
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
        }
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
        let server = ServerReplication::<S>::new(
            self.schema.clone(),
            self.capacity,
            &self.snapshot,
            strategy,
            transport,
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
    use crate::delta::{FieldKind, FieldValue, encode_snapshot};
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

    /// The full loop, both halves as real `prism_app` plugins over one reliable
    /// channel: a client's command reaches authority, authority advances and
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
        assert!(
            client_app
                .world_mut()
                .resource_mut::<Replication<AddClient>>()
                .send(&3i64.to_le_bytes())
                .is_some()
        );

        // Alternate the two frame loops. The first client frame puts the command
        // on the wire; the server frame applies it and replies with the delta; the
        // next client frame reconciles.
        for _ in 0..8 {
            client_app.update();
            server_app.update();
        }

        let server = server_app.world().resource::<ServerReplication<AddAuthority>>();
        assert_eq!(server.revision(), Some(1), "authority advanced exactly once");
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

        assert!(
            client_app
                .world_mut()
                .resource_mut::<Replication<AddClient>>()
                .send(b"not eight bytes")
                .is_some()
        );
        for _ in 0..4 {
            client_app.update();
            server_app.update();
        }
        let server = server_app.world().resource::<ServerReplication<AddAuthority>>();
        assert_eq!(server.revision(), Some(0), "the command was ignored");
        assert_eq!(server.applied(), 0);
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
