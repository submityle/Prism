//! Repeatable real-loopback UDP acceptance coverage.
//!
//! This file deliberately uses real `UdpSocket`s for both clients and the
//! fault proxy. The proxy only makes one deterministic fault reproducible; it
//! does not replace a multi-host or public-network test.
//! The cross-process slice launches separate client and server test processes;
//! its parent only coordinates addresses and observes the forced retransmit.

use prism_replication::reliable::ReliableChannel;
use prism_replication::transport::Transport;
use prism_replication::udp::{
    UdpMultiPeer, UdpMultiPeerConfig, UdpPeerBindingState, UdpPeerMigrationError, UdpTransport,
};
use std::io::{self, BufRead, BufReader, Write};
use std::net::{SocketAddr, UdpSocket};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::{Duration, Instant};

const MAX_STEPS: usize = 4_000;
const POLL_SLEEP: Duration = Duration::from_millis(1);
const PROCESS_TIMEOUT: Duration = Duration::from_secs(10);
const CLIENT_HELLO: &[u8] = b"prism-cross-process-client-hello-v1";
const SERVER_ACCEPT: &[u8] = b"prism-cross-process-server-accept-v1";
const CLIENT_DATA: &[u8] = b"prism-cross-process-payload";
const SERVER_DATA_ACK: &[u8] = b"prism-cross-process-payload-accepted";

struct UdpFaultProxy {
    left: UdpSocket,
    right: UdpSocket,
    left_client: SocketAddr,
    right_client: SocketAddr,
    drop_first_left_to_right: bool,
    drop_first_right_to_left: bool,
    dropped: usize,
    dropped_data: usize,
    dropped_ack: usize,
    forwarded: usize,
}

impl UdpFaultProxy {
    fn bind(left_client: SocketAddr, right_client: SocketAddr) -> io::Result<Self> {
        let left = UdpSocket::bind("127.0.0.1:0")?;
        let right = UdpSocket::bind("127.0.0.1:0")?;
        left.set_nonblocking(true)?;
        right.set_nonblocking(true)?;
        Ok(Self {
            left,
            right,
            left_client,
            right_client,
            drop_first_left_to_right: true,
            drop_first_right_to_left: true,
            dropped: 0,
            dropped_data: 0,
            dropped_ack: 0,
            forwarded: 0,
        })
    }

    fn left_addr(&self) -> io::Result<SocketAddr> {
        self.left.local_addr()
    }

    fn right_addr(&self) -> io::Result<SocketAddr> {
        self.right.local_addr()
    }

    fn pump(&mut self) -> io::Result<usize> {
        let mut forwarded_now = 0;
        let mut buffer = [0_u8; 65_507];

        loop {
            match self.left.recv_from(&mut buffer) {
                Ok((len, source)) => {
                    assert_eq!(source, self.left_client, "unexpected left UDP source");
                    if self.drop_first_left_to_right {
                        self.drop_first_left_to_right = false;
                        self.dropped += 1;
                        self.dropped_data += 1;
                        continue;
                    }
                    self.right.send_to(&buffer[..len], self.right_client)?;
                    forwarded_now += 1;
                    self.forwarded += 1;
                }
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => break,
                Err(error) => return Err(error),
            }
        }

        loop {
            match self.right.recv_from(&mut buffer) {
                Ok((len, source)) => {
                    assert_eq!(source, self.right_client, "unexpected right UDP source");
                    if self.drop_first_right_to_left {
                        self.drop_first_right_to_left = false;
                        self.dropped += 1;
                        self.dropped_ack += 1;
                        continue;
                    }
                    self.left.send_to(&buffer[..len], self.left_client)?;
                    forwarded_now += 1;
                    self.forwarded += 1;
                }
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => break,
                Err(error) => return Err(error),
            }
        }

        Ok(forwarded_now)
    }
}

fn recv_socket(socket: &UdpSocket) -> Option<Vec<u8>> {
    let mut buffer = [0_u8; 65_507];
    match socket.recv_from(&mut buffer) {
        Ok((len, _)) => Some(buffer[..len].to_vec()),
        Err(error) if error.kind() == io::ErrorKind::WouldBlock => None,
        Err(error) => panic!("unexpected UDP receive error: {error}"),
    }
}

fn poll_until<T>(mut step: impl FnMut() -> Option<T>, done: impl Fn(&T) -> bool) -> T {
    for _ in 0..MAX_STEPS {
        if let Some(value) = step()
            && done(&value)
        {
            return value;
        }
        thread::sleep(POLL_SLEEP);
    }
    panic!("real UDP acceptance condition did not converge within {MAX_STEPS} steps");
}

fn wait_until(mut step: impl FnMut() -> bool) {
    for _ in 0..MAX_STEPS {
        if step() {
            return;
        }
        thread::sleep(POLL_SLEEP);
    }
    panic!("real UDP acceptance condition did not converge within {MAX_STEPS} steps");
}

struct CrossProcessWorker {
    role: &'static str,
    child: Child,
    stdin: Option<ChildStdin>,
    lines: Receiver<String>,
    reader: Option<thread::JoinHandle<io::Result<()>>>,
    transcript: Vec<String>,
}

impl CrossProcessWorker {
    fn spawn(role: &'static str, peer_addr: SocketAddr) -> Self {
        let mut command = Command::new(std::env::current_exe().expect("current test executable"));
        command
            .args(["--exact", "cross_process_udp_worker", "--nocapture"])
            .env("PRISM_UDP_WORKER", role)
            .env("PRISM_UDP_PEER_ADDR", peer_addr.to_string())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .stdin(Stdio::piped());
        let mut child = command.spawn().expect("spawn cross-process UDP worker");
        let stdout = child.stdout.take().expect("worker stdout");
        let stdin = child.stdin.take().expect("worker stdin");
        let (sender, lines) = mpsc::channel();
        let reader = thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                if sender.send(line?).is_err() {
                    break;
                }
            }
            Ok(())
        });
        Self {
            role,
            child,
            stdin: Some(stdin),
            lines,
            reader: Some(reader),
            transcript: Vec::new(),
        }
    }

    fn read_prefixed(&mut self, prefix: &str) -> String {
        let deadline = Instant::now() + PROCESS_TIMEOUT;
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            let line = self.lines.recv_timeout(remaining).unwrap_or_else(|error| {
                panic!(
                    "{} worker did not print {prefix:?} within {PROCESS_TIMEOUT:?}: {error}; output={:?}",
                    self.role, self.transcript
                )
            });
            self.transcript.push(line.clone());
            if let Some(value) = line.strip_prefix(prefix) {
                return value.trim().to_owned();
            }
        }
    }

    fn write_line(&mut self, line: &str) {
        let stdin = self.stdin.as_mut().expect("worker stdin remains open");
        writeln!(stdin, "{line}").expect("write worker command");
        stdin.flush().expect("flush worker command");
    }

    fn wait_success(&mut self) {
        let deadline = Instant::now() + PROCESS_TIMEOUT;
        let status = loop {
            match self.child.try_wait().expect("poll worker status") {
                Some(status) => break status,
                None if Instant::now() < deadline => thread::sleep(POLL_SLEEP),
                None => {
                    self.terminate();
                    panic!(
                        "{} worker exceeded {PROCESS_TIMEOUT:?}; output={:?}",
                        self.role, self.transcript
                    );
                }
            }
        };
        self.stdin.take();
        self.finish_reader();
        self.transcript.extend(self.lines.try_iter());
        assert!(
            status.success(),
            "{} worker failed with {status}; output={:?}",
            self.role,
            self.transcript
        );
    }

    fn terminate(&mut self) {
        if self
            .child
            .try_wait()
            .expect("poll worker before cleanup")
            .is_none()
        {
            self.child.kill().expect("kill timed-out UDP worker");
            self.child.wait().expect("reap timed-out UDP worker");
        }
        self.stdin.take();
        self.finish_reader();
    }

    fn finish_reader(&mut self) {
        if let Some(reader) = self.reader.take() {
            reader
                .join()
                .expect("worker output reader did not panic")
                .expect("read worker output");
        }
    }
}

impl Drop for CrossProcessWorker {
    fn drop(&mut self) {
        self.terminate();
    }
}

fn parse_data_sequence(packet: &[u8]) -> u64 {
    assert_eq!(packet.first(), Some(&0), "expected a reliable data packet");
    let sequence = packet
        .get(1..9)
        .expect("reliable data header contains a sequence number");
    u64::from_le_bytes(sequence.try_into().expect("eight-byte sequence"))
}

#[test]
fn real_udp_loss_retransmit_is_ordered_and_exactly_once() {
    let mut left = UdpTransport::bind("127.0.0.1:0").expect("bind left UDP client");
    let mut right = UdpTransport::bind("127.0.0.1:0").expect("bind right UDP client");
    let left_client = left.local_addr().expect("left local address");
    let right_client = right.local_addr().expect("right local address");
    let mut proxy = UdpFaultProxy::bind(left_client, right_client).expect("bind UDP proxy");
    left.connect_to(proxy.left_addr().expect("left proxy address"))
        .expect("connect left to proxy");
    right
        .connect_to(proxy.right_addr().expect("right proxy address"))
        .expect("connect right to proxy");

    let expected = (0_u8..8)
        .map(|value| vec![value, value ^ 0xA5])
        .collect::<Vec<_>>();
    let mut sender = ReliableChannel::new(left, 2, expected.len());
    let mut receiver = ReliableChannel::new(right, 2, expected.len());
    for payload in &expected {
        assert!(
            sender.send(payload).is_some(),
            "send window accepted payload"
        );
    }

    let mut received = Vec::new();
    for tick in 1..=MAX_STEPS as u64 {
        sender.tick_at(tick);
        receiver.tick_at(tick);
        proxy.pump().expect("pump real UDP fault proxy");
        while let Some(payload) = receiver.recv() {
            received.push(payload);
        }
        if received.len() == expected.len() && sender.unacked() == 0 {
            break;
        }
        thread::sleep(POLL_SLEEP);
    }

    assert_eq!(
        proxy.dropped, 2,
        "the proxy dropped exactly one data packet and one ACK"
    );
    assert_eq!(proxy.dropped_data, 1, "the first data datagram was dropped");
    assert_eq!(proxy.dropped_ack, 1, "the first reverse ACK was dropped");
    assert!(
        proxy.forwarded > expected.len(),
        "a retransmission crossed the proxy"
    );
    assert_eq!(
        received, expected,
        "real UDP loss recovered in order exactly once"
    );
    assert_eq!(
        sender.unacked(),
        0,
        "all recovered messages were acknowledged"
    );
}

#[test]
fn real_udp_multi_peer_three_clients_keep_reliable_channels_isolated() {
    let endpoint = UdpMultiPeer::bind(
        "127.0.0.1:0",
        UdpMultiPeerConfig {
            max_peers: 3,
            max_queue_per_peer: 32,
            max_total_queued: 96,
            ..UdpMultiPeerConfig::default()
        },
    )
    .expect("bind multi-peer UDP endpoint");
    let endpoint_addr = endpoint.local_addr().expect("endpoint address");

    let mut clients = Vec::with_capacity(3);
    let mut client_addresses = Vec::with_capacity(3);
    for _ in 0..3 {
        let mut transport = UdpTransport::bind("127.0.0.1:0").expect("bind UDP client");
        transport
            .connect_to(endpoint_addr)
            .expect("connect client to multi-peer endpoint");
        client_addresses.push(transport.local_addr().expect("client address"));
        clients.push(ReliableChannel::new(transport, 2, 8));
    }

    let mut server_channels = Vec::with_capacity(3);
    let mut peer_ids = Vec::with_capacity(3);
    for address in client_addresses {
        let peer = endpoint.peer(address).expect("admit UDP client");
        peer_ids.push(peer.id().as_u64());
        server_channels.push(ReliableChannel::new(peer, 2, 8));
    }
    assert_eq!(peer_ids, vec![1, 2, 3], "peer admission is deterministic");

    let client_messages = [b"client-0".as_slice(), b"client-1", b"client-2"];
    for (channel, message) in clients.iter_mut().zip(client_messages) {
        assert_eq!(channel.send(message), Some(0));
    }

    let mut server_received = vec![Vec::new(), Vec::new(), Vec::new()];
    for tick in 1..=MAX_STEPS as u64 {
        for channel in &mut clients {
            channel.tick_at(tick);
        }
        endpoint.poll();
        for (index, channel) in server_channels.iter_mut().enumerate() {
            channel.tick_at(tick);
            while let Some(message) = channel.recv() {
                server_received[index].push(message);
            }
        }
        if server_received.iter().all(|messages| messages.len() == 1)
            && clients.iter().all(|channel| channel.unacked() == 0)
        {
            break;
        }
        thread::sleep(POLL_SLEEP);
    }

    assert_eq!(
        server_received,
        vec![
            vec![b"client-0".to_vec()],
            vec![b"client-1".to_vec()],
            vec![b"client-2".to_vec()],
        ],
        "each reliable channel receives only its admitted peer's payload"
    );

    let server_messages = [b"server-0".as_slice(), b"server-1", b"server-2"];
    for (channel, message) in server_channels.iter_mut().zip(server_messages) {
        assert_eq!(channel.send(message), Some(0));
    }

    let mut client_received = vec![Vec::new(), Vec::new(), Vec::new()];
    for tick in (MAX_STEPS as u64 + 1)..=(MAX_STEPS as u64 * 2) {
        for channel in &mut server_channels {
            channel.tick_at(tick);
        }
        endpoint.poll();
        for (index, channel) in clients.iter_mut().enumerate() {
            channel.tick_at(tick);
            while let Some(message) = channel.recv() {
                client_received[index].push(message);
            }
        }
        if client_received.iter().all(|messages| messages.len() == 1)
            && server_channels.iter().all(|channel| channel.unacked() == 0)
        {
            break;
        }
        thread::sleep(POLL_SLEEP);
    }

    assert_eq!(
        client_received,
        vec![
            vec![b"server-0".to_vec()],
            vec![b"server-1".to_vec()],
            vec![b"server-2".to_vec()],
        ],
        "each client receives only its peer-specific reply"
    );
    assert_eq!(
        endpoint.dropped(),
        0,
        "admitted multi-peer traffic is retained"
    );
}

#[test]
fn real_udp_peer_rebind_reconnects_after_old_socket_is_closed() {
    let endpoint = UdpMultiPeer::bind("127.0.0.1:0", UdpMultiPeerConfig::default())
        .expect("bind multi-peer UDP endpoint");
    let endpoint_addr = endpoint.local_addr().expect("endpoint address");

    let old_client = UdpSocket::bind("127.0.0.1:0").expect("bind old peer socket");
    old_client
        .set_nonblocking(true)
        .expect("old peer nonblocking");
    let old_addr = old_client.local_addr().expect("old peer address");
    let mut peer = endpoint.peer(old_addr).expect("admit old peer");
    let stable_id = peer.id();

    old_client
        .send_to(b"before reconnect", endpoint_addr)
        .expect("send before reconnect");
    poll_until(
        || {
            endpoint.poll();
            peer.recv()
        },
        |payload| payload.as_slice() == b"before reconnect",
    );

    let new_client = UdpSocket::bind("127.0.0.1:0").expect("bind reconnected peer socket");
    new_client
        .set_nonblocking(true)
        .expect("reconnected peer nonblocking");
    let new_addr = new_client.local_addr().expect("reconnected peer address");
    let ticket = peer
        .request_rebind(new_addr)
        .expect("reserve rebind endpoint");

    let dropped_before_candidate = endpoint.dropped();
    new_client
        .send_to(b"before commit", endpoint_addr)
        .expect("send candidate before commit");
    wait_until(|| {
        endpoint.poll();
        endpoint.dropped() > dropped_before_candidate
    });
    assert_eq!(peer.recv(), None, "candidate is not active before commit");

    peer.send_rebind_challenge(ticket)
        .expect("send candidate path challenge");
    let challenge = poll_until(|| recv_socket(&new_client), |packet| !packet.is_empty());
    new_client
        .send_to(&challenge, endpoint_addr)
        .expect("echo candidate path challenge");

    drop(old_client);
    wait_until(|| {
        endpoint.poll();
        peer.commit_rebind(ticket).is_ok()
    });
    assert_eq!(peer.id(), stable_id, "reconnect preserves peer identity");
    assert_eq!(peer.peer(), new_addr, "new socket is the active endpoint");
    assert_eq!(
        peer.binding_state(),
        UdpPeerBindingState::Stable {
            endpoint: new_addr,
            generation: 1,
        }
    );

    new_client
        .send_to(b"after reconnect", endpoint_addr)
        .expect("send after reconnect");
    let received = poll_until(
        || {
            endpoint.poll();
            peer.recv()
        },
        |payload| payload.as_slice() == b"after reconnect",
    );
    assert_eq!(received, b"after reconnect".to_vec());

    peer.send(b"reconnect reply");
    let reply = poll_until(
        || recv_socket(&new_client),
        |payload| payload.as_slice() == b"reconnect reply",
    );
    assert_eq!(reply, b"reconnect reply".to_vec());
    assert_eq!(
        endpoint.peer(new_addr).expect("lookup rebound peer").id(),
        stable_id
    );
    assert!(
        endpoint.dropped() >= 1,
        "pre-commit candidate traffic was rejected"
    );

    assert_eq!(
        peer.request_rebind(new_addr),
        Err(UdpPeerMigrationError::InvalidEndpoint),
        "rebinding to the active endpoint is rejected"
    );
}

#[test]
fn cross_process_udp_worker() {
    let Ok(role) = std::env::var("PRISM_UDP_WORKER") else {
        return;
    };
    let peer_addr = std::env::var("PRISM_UDP_PEER_ADDR")
        .expect("worker peer address")
        .parse::<SocketAddr>()
        .expect("parse worker peer address");
    match role.as_str() {
        "client" => {
            let mut transport = UdpTransport::bind("127.0.0.1:0").expect("bind client worker");
            let local = transport.local_addr().expect("client worker address");
            transport
                .connect_to(peer_addr)
                .expect("connect client worker to initial blackhole");
            let mut channel = ReliableChannel::new(transport, 2, 8);
            let hello_sequence = channel.send(CLIENT_HELLO).expect("queue client hello");

            println!("CLIENT_READY {local}");
            io::stdout().flush().expect("flush client readiness");

            channel.tick_at(1);
            channel.tick_at(3);
            assert_eq!(
                channel.unacked(),
                1,
                "the blackholed hello must remain in flight"
            );
            println!("CLIENT_RETRANSMITTED {hello_sequence}");
            io::stdout().flush().expect("flush retransmit readiness");

            let mut input = String::new();
            BufReader::new(io::stdin())
                .read_line(&mut input)
                .expect("read server address");
            let server_addr = input
                .strip_prefix("SERVER_ADDR ")
                .expect("server address handoff")
                .trim()
                .parse::<SocketAddr>()
                .expect("parse server address");
            channel
                .datagram_mut()
                .connect_to(server_addr)
                .expect("switch client worker to server endpoint");

            let mut accepted = 0;
            let mut data_sent = false;
            let mut data_sequence = None;
            let mut data_accepted = 0;
            for tick in 4..=MAX_STEPS as u64 {
                channel.tick_at(tick);
                while let Some(payload) = channel.recv() {
                    if payload == SERVER_ACCEPT {
                        accepted += 1;
                    } else if payload == SERVER_DATA_ACK {
                        data_accepted += 1;
                    } else {
                        panic!("client received unexpected payload: {payload:?}");
                    }
                }
                if accepted == 1 && channel.unacked() == 0 && !data_sent {
                    data_sequence = Some(channel.send(CLIENT_DATA).expect("queue client data"));
                    data_sent = true;
                }
                if accepted == 1 && data_accepted == 1 && data_sent && channel.unacked() == 0 {
                    println!(
                        "CLIENT_OK hello_seq={hello_sequence} data_seq={}",
                        data_sequence.expect("data sequence was assigned")
                    );
                    io::stdout().flush().expect("flush client result");
                    return;
                }
                thread::sleep(POLL_SLEEP);
            }
            panic!(
                "cross-process client did not complete: accepted={accepted} data_accepted={data_accepted} unacked={}",
                channel.unacked()
            );
        }
        "server" => {
            let endpoint = UdpMultiPeer::bind("127.0.0.1:0", UdpMultiPeerConfig::default())
                .expect("bind server worker");
            let local = endpoint.local_addr().expect("server worker address");
            let peer = endpoint.peer(peer_addr).expect("admit client worker peer");
            let stable_id = peer.id();
            assert_eq!(
                peer.binding_state(),
                UdpPeerBindingState::Stable {
                    endpoint: peer_addr,
                    generation: 0,
                }
            );
            println!("SERVER_READY {local} peer_id={}", stable_id.as_u64());
            io::stdout().flush().expect("flush server readiness");

            let mut channel = ReliableChannel::new(peer, 2, 8);
            let mut hello_count = 0;
            let mut data_count = 0;
            for tick in 1..=MAX_STEPS as u64 {
                endpoint.poll();
                channel.tick_at(tick);
                while let Some(payload) = channel.recv() {
                    if payload == CLIENT_HELLO {
                        hello_count += 1;
                        assert_eq!(hello_count, 1, "hello was delivered exactly once");
                        channel
                            .send(SERVER_ACCEPT)
                            .expect("queue handshake acceptance");
                    } else if payload == CLIENT_DATA {
                        assert_eq!(hello_count, 1, "data arrived after the handshake");
                        data_count += 1;
                        assert_eq!(data_count, 1, "application data was delivered exactly once");
                        channel
                            .send(SERVER_DATA_ACK)
                            .expect("queue application acceptance");
                    } else {
                        panic!("server received unexpected payload: {payload:?}");
                    }
                }
                let transport = channel.datagram_mut();
                assert_eq!(transport.id(), stable_id, "peer identity changed");
                assert_eq!(transport.peer(), peer_addr, "peer endpoint changed");
                assert_eq!(
                    transport.binding_state(),
                    UdpPeerBindingState::Stable {
                        endpoint: peer_addr,
                        generation: 0,
                    },
                    "peer binding stopped being stable"
                );
                if hello_count == 1 && data_count == 1 && channel.unacked() == 0 {
                    assert_eq!(endpoint.dropped(), 0, "server dropped admitted traffic");
                    println!(
                        "SERVER_OK peer_id={} generation=0 hello_count={hello_count} data_count={data_count}",
                        stable_id.as_u64()
                    );
                    io::stdout().flush().expect("flush server result");
                    return;
                }
                thread::sleep(POLL_SLEEP);
            }
            panic!(
                "cross-process server did not complete: hello_count={hello_count} data_count={data_count} unacked={}",
                channel.unacked()
            );
        }
        other => panic!("unknown cross-process worker role: {other}"),
    }
}

#[test]
fn real_udp_cross_process_handshake_retransmit_and_stable_peer() {
    let blackhole = UdpSocket::bind("127.0.0.1:0").expect("bind deterministic UDP blackhole");
    blackhole
        .set_read_timeout(Some(PROCESS_TIMEOUT))
        .expect("set blackhole timeout");
    let blackhole_addr = blackhole.local_addr().expect("blackhole address");

    let mut client = CrossProcessWorker::spawn("client", blackhole_addr);
    let client_addr = client
        .read_prefixed("CLIENT_READY ")
        .parse::<SocketAddr>()
        .expect("parse client readiness");

    let mut retransmissions = Vec::new();
    while retransmissions.len() < 2 {
        let mut packet = [0_u8; 65_507];
        let (length, source) = blackhole
            .recv_from(&mut packet)
            .expect("receive initial send and deterministic retransmission");
        assert_eq!(source, client_addr, "blackhole source is the client worker");
        retransmissions.push(packet[..length].to_vec());
    }
    assert_eq!(
        retransmissions[0], retransmissions[1],
        "the second datagram is the retained reliable hello retransmission"
    );
    assert_eq!(
        parse_data_sequence(&retransmissions[0]),
        0,
        "the retained handshake uses the first reliable sequence"
    );
    assert_eq!(
        client.read_prefixed("CLIENT_RETRANSMITTED "),
        "0",
        "the client retained and retransmitted sequence zero"
    );

    let mut server = CrossProcessWorker::spawn("server", client_addr);
    let server_ready = server.read_prefixed("SERVER_READY ");
    let server_addr = server_ready
        .split_whitespace()
        .next()
        .expect("server readiness address")
        .parse::<SocketAddr>()
        .expect("parse server readiness");
    assert!(
        server_ready.contains("peer_id=1"),
        "server admitted one stable peer: {server_ready}"
    );

    client.write_line(&format!("SERVER_ADDR {server_addr}"));
    let client_result = client.read_prefixed("CLIENT_OK ");
    let server_result = server.read_prefixed("SERVER_OK ");
    assert_eq!(
        client_result, "hello_seq=0 data_seq=1",
        "client completed handshake, payload delivery, and ACK retirement"
    );
    assert_eq!(
        server_result, "peer_id=1 generation=0 hello_count=1 data_count=1",
        "server preserved peer identity and delivered both messages exactly once"
    );

    client.wait_success();
    server.wait_success();
}
