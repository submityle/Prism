//! Repeatable real-loopback UDP acceptance coverage.
//!
//! This file deliberately uses real `UdpSocket`s for both clients and the
//! fault proxy. The proxy only makes one deterministic fault reproducible; it
//! does not replace a multi-host or public-network test.

use prism_replication::reliable::ReliableChannel;
use prism_replication::transport::Transport;
use prism_replication::udp::{
    UdpMultiPeer, UdpMultiPeerConfig, UdpPeerBindingState, UdpPeerMigrationError, UdpTransport,
};
use std::io;
use std::net::{SocketAddr, UdpSocket};
use std::thread;
use std::time::Duration;

const MAX_STEPS: usize = 4_000;
const POLL_SLEEP: Duration = Duration::from_millis(1);

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
