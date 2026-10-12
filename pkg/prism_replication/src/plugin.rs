//! The `prism_app` seam (design §25.2): drive a client replication session from
//! the app's frame loop.
//!
//! Replication is a plugin on the app shell: [`ReplicationPlugin`] installs a
//! [`Replication`] resource (a [`ClientSession`] plus its [`Transport`]) and a
//! `First` system that drains inbound frames into the prediction each frame. The
//! application supplies a [`ReplicationStrategy`] — how one local input advances
//! the replicated [`Record`] — so the same plugin drives any replicated field
//! set. The caller keeps the peer transport endpoint, so it can drive authority
//! or a test peer.

use alloc::boxed::Box;
use alloc::vec::Vec;
use std::sync::Mutex;

use prism_app::{App, First, Plugin};
use prism_ecs::resource::Resource;
use prism_ecs::system::ResMut;

use crate::delta::{Record, Schema};
use crate::session::{ClientSession, SessionError};
use crate::transport::Transport;

/// The application's replication strategy: how a local input advances the record.
pub trait ReplicationStrategy: Send + Sync + 'static {
    /// The local input type.
    type Input: Clone + Send + Sync + 'static;
    /// Advance `record` by one local `input`.
    fn simulate(&self, record: &mut Record, input: &Self::Input);
}

/// The app-facing replication resource.
pub struct Replication<S: ReplicationStrategy> {
    session: ClientSession<S::Input>,
    strategy: S,
    transport: Box<dyn Transport>,
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
            transport,
            last_error: None,
        })
    }

    /// Drain every inbound frame into the prediction, returning how many
    /// authoritative frames were applied. A framing or reconciliation error stops
    /// the drain and is stored for [`take_last_error`](Self::take_last_error)
    /// rather than propagated, so a bad frame cannot unwind the frame loop.
    pub fn pump(&mut self) -> usize {
        self.last_error = None;
        let Self {
            session,
            strategy,
            transport,
            last_error,
        } = self;
        let mut applied = 0;
        while let Some(bytes) = transport.recv() {
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

    /// Send `bytes` to the peer (the app's outbound path).
    pub fn send(&mut self, bytes: &[u8]) {
        self.transport.send(bytes);
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
    }

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

    fn int(record: &Record) -> i64 {
        let FieldValue::Int(value) = record.value(0).expect("one field") else {
            panic!("field 0 is an int");
        };
        value
    }

    /// The whole point of the seam: a real `App` with the plugin installed
    /// predicts locally, and its `First` system drains an authoritative frame from
    /// the peer transport and reconciles the prediction — no manual pumping.
    #[test]
    fn the_plugin_pumps_authoritative_frames_into_the_prediction() {
        let (client_end, mut server_end) = InMemoryTransport::pair();
        let mut server = ServerSession::new(schema(), 8, &snapshot_at(0, 0)).unwrap();

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

        // Authority applies +3 and sends the framed delta over the transport.
        let mut wire = Vec::new();
        server
            .advance(
                |record| record.set(0, FieldValue::Int(int(record) + 3)),
                &mut wire,
            )
            .unwrap();
        server_end.send(&wire);

        // One frame: the plugin's `First` system drains and reconciles.
        app.update();

        let replication = app.world().resource::<Replication<Add>>();
        assert_eq!(replication.acked(), 1, "authority confirmed");
        assert_eq!(int(replication.state().unwrap()), 3, "prediction rolled back");
    }
}
