use super::*;

use crate::delta::{encode_snapshot, FieldKind, FieldValue, Record};
use crate::plugin::{Replication, ReplicationStrategy};
use crate::security::test_support::Provider;
use crate::transport::memory::InMemoryTransport;
use alloc::boxed::Box;
use alloc::vec;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
struct SharedWorld(Arc<Mutex<BTreeMap<u32, Vec<u8>>>>);

impl AuthorityReplicationProvider for SharedWorld {
    fn snapshot_entity(
        &self,
        _peer_id: PeerId,
        entity_id: u32,
        _aoi_revision: u64,
    ) -> Result<Vec<u8>, AuthorityReplicationProviderError> {
        self.0
            .lock()
            .unwrap()
            .get(&entity_id)
            .cloned()
            .ok_or(AuthorityReplicationProviderError::EntityUnavailable)
    }
}

struct AddAuthority;
impl ServerStrategy for AddAuthority {
    type Command = i64;
    fn decode(&self, bytes: &[u8]) -> Option<Self::Command> {
        <[u8; 8]>::try_from(bytes).ok().map(i64::from_le_bytes)
    }
    fn apply(&self, record: &mut Record, amount: Self::Command) {
        let FieldValue::Int(value) = record.value(0).expect("test schema has one field") else {
            unreachable!()
        };
        record.set(0, FieldValue::Int(value.saturating_add(amount)));
    }
}

struct AddClient;
impl ReplicationStrategy for AddClient {
    type Input = i64;
    fn simulate(&self, record: &mut Record, amount: &i64) {
        let FieldValue::Int(value) = record.value(0).expect("test schema has one field") else {
            unreachable!()
        };
        record.set(0, FieldValue::Int(value.saturating_add(*amount)));
    }
    fn encode(&self, amount: &i64) -> Vec<u8> {
        amount.to_le_bytes().to_vec()
    }
}

fn config() -> MultiPeerAuthorityConfig {
    MultiPeerAuthorityConfig {
        aoi: AuthorityAoiConfig {
            max_peers: 4,
            max_entities: 16,
            max_entities_per_peer: 8,
            max_radius: 100.0,
            cell_size: 10.0,
        },
        max_pending_aoi_events_per_peer: 4,
        max_aoi_payload_bytes: 4096,
    }
}

fn schema() -> Schema {
    Schema::new(vec![FieldKind::Int { bits: 16 }])
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

fn entity_blob(id: u32, value: u8) -> Vec<u8> {
    [id.to_le_bytes().as_slice(), &[value]].concat()
}

#[test]
fn replication_payload_round_trips_and_enforces_total_wire_bounds() {
    let payload = AuthorityAoiReplicationPayload {
        peer_id: 0x0102_0304_0506_0708,
        revision: 17,
        entered: vec![
            EntitySnapshotPayload {
                entity_id: 9,
                bytes: vec![1, 2, 3],
            },
            EntitySnapshotPayload {
                entity_id: 12,
                bytes: Vec::new(),
            },
        ],
        left: vec![4, 5],
    };
    let encoded = encode_authority_aoi_replication_payload(&payload).unwrap();
    assert_eq!(
        decode_authority_aoi_replication_payload(&encoded, 2, encoded.len()).unwrap(),
        payload
    );
    assert_eq!(
        decode_authority_aoi_replication_payload(&encoded, 2, encoded.len() - 1),
        Err(AuthorityAoiEventError::BoundExceeded)
    );
    assert_eq!(
        decode_authority_aoi_replication_payload(&encoded, 1, encoded.len()),
        Err(AuthorityAoiEventError::BoundExceeded)
    );
    let mut trailing = encoded;
    trailing.push(0);
    assert_eq!(
        decode_authority_aoi_replication_payload(&trailing, 2, trailing.len()),
        Err(AuthorityAoiEventError::Malformed)
    );
}

fn secure_pair() -> (
    crate::security::SecureTransport<InMemoryTransport, Provider>,
    crate::security::SecureTransport<InMemoryTransport, Provider>,
) {
    InMemoryTransport::pair_secure()
}

fn exchange(
    authority: &mut MultiPeerAuthority<AddAuthority>,
    clients: &mut [(PeerId, Replication<AddClient>)],
    rounds: usize,
) {
    for _ in 0..rounds {
        for (_, client) in clients.iter_mut() {
            client.pump();
        }
        authority.pump();
    }
}

#[test]
fn multi_peer_aoi_enter_snapshots_and_leave_despawns_are_delivered_from_shared_provider() {
    let world = Arc::new(Mutex::new(BTreeMap::new()));
    world.lock().unwrap().insert(7, entity_blob(7, 42));
    let mut authority = MultiPeerAuthority::new(config(), SharedWorld(Arc::clone(&world))).unwrap();
    authority.upsert_entity(7, (0.0, 0.0)).unwrap();
    let mut clients = Vec::new();

    for peer_id in [10, 20] {
        let (client_transport, server_transport) = secure_pair();
        let client =
            Replication::new(schema(), 8, &seed(), AddClient, Box::new(client_transport)).unwrap();
        authority
            .admit_peer(
                peer_id,
                (0.0, 0.0),
                5.0,
                schema(),
                8,
                &seed(),
                AddAuthority,
                Box::new(server_transport),
            )
            .unwrap();
        clients.push((peer_id, client));
    }

    exchange(&mut authority, &mut clients, 20);
    for (peer_id, client) in &mut clients {
        let bytes = client
            .recv_event()
            .expect("initial visible entity snapshot");
        let payload = decode_authority_aoi_replication_payload(&bytes, 8, 1024).unwrap();
        assert_eq!(payload.peer_id, *peer_id);
        assert_eq!(
            payload.entered,
            vec![EntitySnapshotPayload {
                entity_id: 7,
                bytes: entity_blob(7, 42),
            }]
        );
        assert!(payload.left.is_empty());
    }

    world.lock().unwrap().insert(9, entity_blob(9, 99));
    let entered = authority.upsert_entity(9, (1.0, 1.0)).unwrap();
    assert_eq!(
        entered
            .iter()
            .map(|delta| delta.peer_id)
            .collect::<Vec<_>>(),
        vec![10, 20]
    );
    exchange(&mut authority, &mut clients, 20);
    for (peer_id, client) in &mut clients {
        let payload = decode_authority_aoi_replication_payload(
            &client.recv_event().expect("AOI enter snapshot"),
            8,
            1024,
        )
        .unwrap();
        assert_eq!(payload.peer_id, *peer_id);
        assert_eq!(
            payload.entered,
            vec![EntitySnapshotPayload {
                entity_id: 9,
                bytes: entity_blob(9, 99),
            }]
        );
    }

    let left = authority.update_peer_view(10, (100.0, 0.0), 5.0).unwrap();
    assert_eq!(left.left, vec![7, 9]);
    exchange(&mut authority, &mut clients, 20);
    let payload = decode_authority_aoi_replication_payload(
        &clients[0].1.recv_event().expect("AOI leave despawns"),
        8,
        1024,
    )
    .unwrap();
    assert!(payload.entered.is_empty());
    assert_eq!(payload.left, vec![7, 9]);
    assert!(
        clients[1].1.recv_event().is_none(),
        "other peer retains its view"
    );
}

#[test]
fn provider_failure_retains_bounded_aoi_update_until_shared_state_is_available() {
    let world = Arc::new(Mutex::new(BTreeMap::new()));
    let mut authority = MultiPeerAuthority::new(config(), SharedWorld(Arc::clone(&world))).unwrap();
    let (client_transport, server_transport) = secure_pair();
    let mut client =
        Replication::new(schema(), 8, &seed(), AddClient, Box::new(client_transport)).unwrap();
    authority
        .admit_peer(
            1,
            (0.0, 0.0),
            5.0,
            schema(),
            8,
            &seed(),
            AddAuthority,
            Box::new(server_transport),
        )
        .unwrap();
    authority.upsert_entity(4, (0.0, 0.0)).unwrap();
    for _ in 0..8 {
        client.pump();
        authority.pump();
    }
    let initial = decode_authority_aoi_replication_payload(
        &client.recv_event().expect("initial AOI admission event"),
        8,
        1024,
    )
    .unwrap();
    assert!(initial.entered.is_empty());
    assert!(initial.left.is_empty());
    assert_eq!(
        authority.take_provider_error(1),
        Some(AuthorityReplicationProviderError::EntityUnavailable)
    );
    assert_eq!(authority.pending_aoi_events(1), Some(1));
    world.lock().unwrap().insert(4, entity_blob(4, 5));
    for _ in 0..20 {
        client.pump();
        authority.pump();
    }
    let payload = decode_authority_aoi_replication_payload(
        &client.recv_event().expect("retried AOI snapshot"),
        8,
        1024,
    )
    .unwrap();
    assert_eq!(
        payload.entered,
        vec![EntitySnapshotPayload {
            entity_id: 4,
            bytes: entity_blob(4, 5),
        }]
    );
    assert_eq!(authority.pending_aoi_events(1), Some(0));
}

#[test]
fn pending_aoi_queue_bound_refuses_view_mutation_before_revision_advances() {
    let mut bounded = config();
    bounded.max_pending_aoi_events_per_peer = 2;
    let mut authority =
        MultiPeerAuthority::new(bounded, SharedWorld(Arc::new(Mutex::new(BTreeMap::new()))))
            .unwrap();
    authority.upsert_entity(11, (0.0, 0.0)).unwrap();
    let (_client_transport, server_transport) = secure_pair();
    authority
        .admit_peer(
            3,
            (0.0, 0.0),
            5.0,
            schema(),
            8,
            &seed(),
            AddAuthority,
            Box::new(server_transport),
        )
        .unwrap();
    // The initial event remains queued until the handshake is pumped.
    authority.update_peer_view(3, (10.0, 0.0), 5.0).unwrap();
    assert_eq!(authority.pending_aoi_events(3), Some(2));
    assert_eq!(
        authority.update_peer_view(3, (20.0, 0.0), 5.0),
        Err(MultiPeerAuthorityError::PendingQueueFull { peer_id: 3 })
    );
    assert_eq!(authority.pending_aoi_events(3), Some(2));
}
