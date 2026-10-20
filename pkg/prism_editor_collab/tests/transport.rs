#![allow(missing_docs)]

use prism_editor_collab::{
    ActorId, ClientError, ClientEvent, CommandCapabilities, CommandEnvelope, CommandId,
    CommandLimits, CommandPayload, DomainId, FieldPath, HlcTimestamp, RejectionReason, Replica,
    ServerConfig, TcpCollaborationClient, TcpCollaborationServer, TransportLimits, Value,
};
use std::thread;
use std::time::Duration;

fn actor(value: &'static str) -> ActorId {
    ActorId::from_static(value)
}

fn domain(value: &'static str) -> DomainId {
    DomainId::from_static(value)
}

fn command(
    actor_name: &'static str,
    sequence: u64,
    physical: u64,
    causal: Vec<CommandId>,
    domain_name: &'static str,
    field: &'static str,
    value: i64,
) -> CommandEnvelope {
    CommandEnvelope::new(
        actor(actor_name),
        sequence,
        HlcTimestamp::new(physical, 0),
        causal,
        [domain(domain_name)].into_iter().collect(),
        CommandPayload::SetField {
            path: FieldPath::field(field).expect("field path"),
            value: Value::Int(value),
        },
        CommandLimits::default(),
    )
    .expect("command")
}

fn server_config() -> ServerConfig {
    let transport = TransportLimits {
        max_client_queue: 16,
        max_history_commands: 8,
        ..TransportLimits::default()
    };
    ServerConfig {
        transport,
        capabilities: [
            (actor("alice"), CommandCapabilities::unrestricted(1024)),
            (actor("bob"), CommandCapabilities::unrestricted(1024)),
        ]
        .into_iter()
        .collect(),
        ..ServerConfig::default()
    }
}

fn client(name: &'static str) -> TcpCollaborationClient {
    TcpCollaborationClient::new(
        actor(name),
        Replica::new(actor(name), 32).expect("replica"),
        16,
        CommandLimits::default(),
    )
    .expect("client")
}

fn poll_until(client: &mut TcpCollaborationClient, expected_order: u64) -> Vec<ClientEvent> {
    for _ in 0..200 {
        match client.poll() {
            Ok(events) if client.last_server_order() >= expected_order => return events,
            Ok(_) => thread::sleep(Duration::from_millis(2)),
            Err(error) => panic!("poll failed: {error}"),
        }
    }
    panic!(
        "timed out waiting for server order {}, observed {}",
        expected_order,
        client.last_server_order()
    );
}

#[test]
fn tcp_transport_broadcasts_in_server_admission_order() {
    let server =
        TcpCollaborationServer::bind("127.0.0.1:0".parse().expect("address"), server_config())
            .expect("bind")
            .start()
            .expect("start");
    let address = server.local_addr();
    let mut alice = client("alice");
    let mut bob = client("bob");
    alice.connect(address).expect("alice connect");
    bob.connect(address).expect("bob connect");

    let first = command("alice", 1, 1, Vec::new(), "scene", "x", 1);
    let second = command("bob", 1, 2, vec![first.id.clone()], "scene", "y", 2);
    alice.submit(first.clone()).expect("submit first");
    bob.submit(second).expect("submit second");

    poll_until(&mut alice, 2);
    poll_until(&mut bob, 2);
    assert_eq!(alice.last_server_order(), 2);
    assert_eq!(alice.replica().state(), bob.replica().state());
    assert_eq!(alice.pending_len(), 0);
    assert_eq!(
        alice
            .replica()
            .state()
            .get(&FieldPath::field("y").expect("path")),
        Some(&Value::Int(2))
    );

    alice.disconnect();
    bob.disconnect();
    server.shutdown();
}

#[test]
fn capability_rejection_is_explicit_and_unknown_actor_cannot_join() {
    let config = ServerConfig {
        capabilities: [(
            actor("alice"),
            CommandCapabilities {
                allow_all_domains: false,
                allowed_domains: [domain("allowed")].into_iter().collect(),
                allow_set_field: true,
                allow_delete_field: false,
                allow_opaque: false,
                max_payload_bytes: 1024,
            },
        )]
        .into_iter()
        .collect(),
        ..server_config()
    };
    let server = TcpCollaborationServer::bind("127.0.0.1:0".parse().expect("address"), config)
        .expect("bind")
        .start()
        .expect("start");
    let address = server.local_addr();
    let mut alice = client("alice");
    alice.connect(address).expect("connect");
    alice
        .submit(command("alice", 1, 1, Vec::new(), "blocked", "x", 1))
        .expect("queue rejected command for server");
    let events = poll_until_rejection(&mut alice);
    assert!(matches!(
        events.as_slice(),
        [ClientEvent::Rejected {
            reason: RejectionReason::Capability(_),
            ..
        }]
    ));
    assert_eq!(alice.pending_len(), 0);

    let mut unknown = client("unknown");
    assert!(matches!(
        unknown.connect(address),
        Err(ClientError::Handshake(RejectionReason::Unauthorized))
    ));

    alice.disconnect();
    unknown.disconnect();
    server.shutdown();
}

fn poll_until_rejection(client: &mut TcpCollaborationClient) -> Vec<ClientEvent> {
    for _ in 0..200 {
        match client.poll() {
            Ok(events)
                if events
                    .iter()
                    .any(|event| matches!(event, ClientEvent::Rejected { .. })) =>
            {
                return events;
            }
            Ok(_) => thread::sleep(Duration::from_millis(2)),
            Err(error) => panic!("poll failed: {error}"),
        }
    }
    panic!("timed out waiting for rejection");
}

#[test]
fn reconnect_replays_authoritative_history_and_retries_pending_commands() {
    let server =
        TcpCollaborationServer::bind("127.0.0.1:0".parse().expect("address"), server_config())
            .expect("bind")
            .start()
            .expect("start");
    let address = server.local_addr();
    let mut alice = client("alice");
    let offline = command("alice", 1, 1, Vec::new(), "scene", "offline", 1);
    alice
        .submit(offline.clone())
        .expect("retain offline command");
    assert_eq!(alice.pending_len(), 1);
    alice.connect(address).expect("connect and flush");
    poll_until(&mut alice, 1);
    assert_eq!(alice.pending_len(), 0);

    let after_disconnect = command("alice", 2, 2, vec![offline.id], "scene", "after", 2);
    alice.submit(after_disconnect).expect("submit second");
    thread::sleep(Duration::from_millis(20));
    alice.disconnect();
    assert_eq!(alice.pending_len(), 1);
    let replay = alice.reconnect().expect("reconnect");
    assert!(replay
        .iter()
        .any(|event| { matches!(event, ClientEvent::Command { order: 2, .. }) }));
    poll_until(&mut alice, 2);
    assert_eq!(alice.pending_len(), 0);
    assert_eq!(alice.last_server_order(), 2);

    alice.disconnect();
    server.shutdown();
}

#[test]
fn client_queue_is_bounded_without_dropping_the_retained_command() {
    let mut client = TcpCollaborationClient::new(
        actor("alice"),
        Replica::new(actor("alice"), 8).expect("replica"),
        1,
        CommandLimits::default(),
    )
    .expect("client");
    client
        .submit(command("alice", 1, 1, Vec::new(), "scene", "one", 1))
        .expect("first queued command");
    assert!(matches!(
        client.submit(command("alice", 2, 2, Vec::new(), "scene", "two", 2)),
        Err(ClientError::QueueFull)
    ));
    assert_eq!(client.pending_len(), 1);
}
