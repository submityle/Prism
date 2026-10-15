//! A real UDP transport (design §6 "可靠 UDP" — the datagram underlayer).
//!
//! [`UdpTransport`] is the provider-seam [`Transport`] over a real `UdpSocket`.
//! UDP is **unreliable and unordered** by design: this layer is best-effort, and
//! reliability, ordering and framing live above it (the [`crate::sequence`]
//! delivery window restores order; [`crate::frame`] splits a stream; an ARQ layer
//! would retransmit). That is exactly why the pieces below it are pure and
//! testable on one machine: the socket is the only part that needs a network.
//!
//! The socket is non-blocking, so [`recv`](Transport::recv) returns `None` when
//! nothing has arrived yet — a frame loop polls it each frame rather than
//! blocking on it.

use alloc::vec;
use alloc::vec::Vec;
use alloc::{collections::BTreeMap, collections::VecDeque, sync::Arc};
use std::io;
use std::net::{SocketAddr, ToSocketAddrs, UdpSocket};
use std::sync::Mutex;
use std::time::Duration;

use crate::transport::Transport;

/// The largest UDP payload (the IPv4 limit): a datagram larger than this can
/// never arrive, so it bounds the receive buffer.
pub const MAX_DATAGRAM: usize = 65_507;

/// Bounded receive limits for [`UdpMultiPeer`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct UdpMultiPeerConfig {
    /// Maximum number of peer addresses admitted at once.
    pub max_peers: usize,
    /// Maximum queued datagrams retained for one peer.
    pub max_queue_per_peer: usize,
    /// Maximum queued datagrams retained across all peers.
    pub max_total_queued: usize,
}

impl Default for UdpMultiPeerConfig {
    fn default() -> Self {
        Self {
            max_peers: 64,
            max_queue_per_peer: 256,
            max_total_queued: 4_096,
        }
    }
}

impl UdpMultiPeerConfig {
    fn validate(self) -> io::Result<Self> {
        if self.max_peers == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "multi-peer UDP max_peers must be positive",
            ));
        }
        if self.max_queue_per_peer == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "multi-peer UDP max_queue_per_peer must be positive",
            ));
        }
        if self.max_total_queued == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "multi-peer UDP max_total_queued must be positive",
            ));
        }
        Ok(self)
    }
}

struct MultiPeerState {
    queues: BTreeMap<SocketAddr, VecDeque<Vec<u8>>>,
    total_queued: usize,
    dropped: u64,
    last_error: Option<io::ErrorKind>,
}

impl MultiPeerState {
    fn new() -> Self {
        Self {
            queues: BTreeMap::new(),
            total_queued: 0,
            dropped: 0,
            last_error: None,
        }
    }
}

/// A bounded, nonblocking UDP endpoint that demultiplexes datagrams by source
/// address.
///
/// [`UdpTransport`] intentionally models one connected peer. This endpoint is
/// the server-side counterpart for a real multi-peer deployment: call
/// [`poll`](Self::poll) from the network tick, then obtain one
/// [`UdpPeerTransport`] per admitted peer and run the normal reliable/multiplex
/// layers over each peer transport. The endpoint owns no protocol state and
/// does not silently evict peers; capacity pressure drops new datagrams and is
/// observable through [`dropped`](Self::dropped).
#[derive(Clone)]
pub struct UdpMultiPeer {
    socket: Arc<UdpSocket>,
    config: UdpMultiPeerConfig,
    state: Arc<Mutex<MultiPeerState>>,
}

impl UdpMultiPeer {
    /// Bind a nonblocking multi-peer endpoint.
    pub fn bind(local: impl ToSocketAddrs, config: UdpMultiPeerConfig) -> io::Result<Self> {
        let config = config.validate()?;
        let socket = UdpSocket::bind(local)?;
        socket.set_nonblocking(true)?;
        Ok(Self {
            socket: Arc::new(socket),
            config,
            state: Arc::new(Mutex::new(MultiPeerState::new())),
        })
    }

    /// The local address the endpoint is bound to.
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        self.socket.local_addr()
    }

    /// Admit a peer and return its transport view.
    ///
    /// Admission is explicit so an application can authenticate or rate-limit
    /// a peer before allocating per-peer protocol state. Repeated admission is
    /// idempotent.
    pub fn peer(&self, peer: SocketAddr) -> io::Result<UdpPeerTransport> {
        let mut state = lock(&self.state);
        if !state.queues.contains_key(&peer) {
            if state.queues.len() >= self.config.max_peers {
                return Err(io::Error::new(
                    io::ErrorKind::WouldBlock,
                    "multi-peer UDP peer capacity is full",
                ));
            }
            state.queues.insert(peer, VecDeque::new());
        }
        drop(state);
        Ok(UdpPeerTransport {
            socket: Arc::clone(&self.socket),
            state: Arc::clone(&self.state),
            config: self.config,
            peer,
            dropped: 0,
            received: 0,
            last_error: None,
        })
    }

    /// Drain all currently readable datagrams into admitted peer queues.
    ///
    /// Datagrams from unknown peers are dropped instead of implicitly growing
    /// the peer table. This makes admission and memory use explicit.
    pub fn poll(&self) -> usize {
        let mut buffer = vec![0_u8; MAX_DATAGRAM];
        let mut received = 0;
        loop {
            match self.socket.recv_from(&mut buffer) {
                Ok((len, peer)) => {
                    received += 1;
                    let mut state = lock(&self.state);
                    let total_full = state.total_queued >= self.config.max_total_queued;
                    let Some(queue) = state.queues.get_mut(&peer) else {
                        state.dropped = state.dropped.saturating_add(1);
                        continue;
                    };
                    if queue.len() >= self.config.max_queue_per_peer || total_full {
                        state.dropped = state.dropped.saturating_add(1);
                        continue;
                    }
                    queue.push_back(buffer[..len].to_vec());
                    state.total_queued += 1;
                }
                Err(ref error) if error.kind() == io::ErrorKind::WouldBlock => break,
                Err(error) => {
                    let mut state = lock(&self.state);
                    state.last_error = Some(error.kind());
                    break;
                }
            }
        }
        received
    }

    /// Number of datagrams rejected or dropped by the endpoint.
    #[must_use]
    pub fn dropped(&self) -> u64 {
        lock(&self.state).dropped
    }

    /// Take the last socket error observed by [`poll`](Self::poll).
    pub fn take_error_kind(&self) -> Option<io::ErrorKind> {
        lock(&self.state).last_error.take()
    }
}

/// A [`Transport`] view of one admitted peer on a [`UdpMultiPeer`] endpoint.
pub struct UdpPeerTransport {
    socket: Arc<UdpSocket>,
    state: Arc<Mutex<MultiPeerState>>,
    config: UdpMultiPeerConfig,
    peer: SocketAddr,
    dropped: u64,
    received: u64,
    last_error: Option<io::ErrorKind>,
}

impl UdpPeerTransport {
    /// The remote peer address represented by this transport.
    #[must_use]
    pub const fn peer(&self) -> SocketAddr {
        self.peer
    }

    /// Number of datagrams rejected while sending through this peer view.
    #[must_use]
    pub const fn dropped(&self) -> u64 {
        self.dropped
    }

    /// Number of datagrams consumed from this peer's queue.
    #[must_use]
    pub const fn received(&self) -> u64 {
        self.received
    }

    /// Take the last send error observed by this peer view.
    pub fn take_error_kind(&mut self) -> Option<io::ErrorKind> {
        self.last_error.take()
    }
}

impl Transport for UdpPeerTransport {
    fn send(&mut self, bytes: &[u8]) {
        if bytes.len() > MAX_DATAGRAM {
            self.last_error = Some(io::ErrorKind::InvalidInput);
            self.dropped = self.dropped.saturating_add(1);
            return;
        }
        if let Err(error) = self.socket.send_to(bytes, self.peer) {
            self.last_error = Some(error.kind());
            self.dropped = self.dropped.saturating_add(1);
        }
    }

    fn recv(&mut self) -> Option<Vec<u8>> {
        let mut buffer = vec![0_u8; MAX_DATAGRAM];
        loop {
            match self.socket.recv_from(&mut buffer) {
                Ok((len, peer)) => {
                    let mut state = lock(&self.state);
                    let total_full = state.total_queued >= self.config.max_total_queued;
                    let Some(queue) = state.queues.get_mut(&peer) else {
                        state.dropped = state.dropped.saturating_add(1);
                        continue;
                    };
                    if queue.len() >= self.config.max_queue_per_peer || total_full {
                        state.dropped = state.dropped.saturating_add(1);
                        continue;
                    }
                    queue.push_back(buffer[..len].to_vec());
                    state.total_queued += 1;
                }
                Err(ref error) if error.kind() == io::ErrorKind::WouldBlock => break,
                Err(error) => {
                    self.last_error = Some(error.kind());
                    return None;
                }
            }
        }
        let mut state = lock(&self.state);
        let message = state.queues.get_mut(&self.peer)?.pop_front();
        if message.is_some() {
            state.total_queued = state.total_queued.saturating_sub(1);
            self.received = self.received.saturating_add(1);
        }
        message
    }
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// An unreliable UDP transport to one peer.
pub struct UdpTransport {
    socket: UdpSocket,
    peer: SocketAddr,
    connected: bool,
    /// The receive scratch buffer, reused across polls.
    buffer: Vec<u8>,
    last_error: Option<io::ErrorKind>,
    dropped: u64,
    received: u64,
}

impl UdpTransport {
    /// Bind a local address (non-blocking). Bind `127.0.0.1:0` for an ephemeral
    /// loopback port and read it back with [`local_addr`](Self::local_addr).
    ///
    /// # Errors
    /// Any bind / `set_nonblocking` error.
    pub fn bind(local: impl ToSocketAddrs) -> io::Result<Self> {
        let socket = UdpSocket::bind(local)?;
        socket.set_nonblocking(true)?;
        Ok(Self {
            socket,
            peer: SocketAddr::from(([0, 0, 0, 0], 0)),
            connected: false,
            buffer: vec![0u8; MAX_DATAGRAM],
            last_error: None,
            dropped: 0,
            received: 0,
        })
    }

    /// Target `peer` for subsequent sends.
    ///
    /// # Errors
    /// [`io::ErrorKind::InvalidInput`] when `peer` resolves to no address.
    pub fn connect_to(&mut self, peer: impl ToSocketAddrs) -> io::Result<()> {
        let peer = peer.to_socket_addrs()?.next().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "peer resolved to nothing")
        })?;
        self.socket.connect(peer)?;
        self.peer = peer;
        self.connected = true;
        Ok(())
    }

    /// The local address the socket is bound to.
    ///
    /// # Errors
    /// The underlying `local_addr` error.
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        self.socket.local_addr()
    }

    /// The peer this transport sends to.
    #[must_use]
    pub fn peer(&self) -> SocketAddr {
        self.peer
    }

    /// Whether [`connect_to`](Self::connect_to) completed successfully. An
    /// unconnected endpoint refuses sends instead of silently targeting the
    /// placeholder `0.0.0.0:0` address.
    #[must_use]
    pub fn is_connected(&self) -> bool {
        self.connected
    }

    /// Take the last socket error observed by send/receive polling.
    pub fn take_error_kind(&mut self) -> Option<io::ErrorKind> {
        self.last_error.take()
    }

    /// Number of datagrams rejected by the local socket or by an unconnected
    /// endpoint.
    #[must_use]
    pub fn dropped(&self) -> u64 {
        self.dropped
    }

    /// Number of datagrams received from the connected peer.
    #[must_use]
    pub fn received(&self) -> u64 {
        self.received
    }

    /// Set a receive timeout (for a blocking-style caller).
    ///
    /// # Errors
    /// The underlying `set_read_timeout` error.
    pub fn set_read_timeout(&self, timeout: Option<Duration>) -> io::Result<()> {
        self.socket.set_read_timeout(timeout)
    }
}

impl Transport for UdpTransport {
    /// Best-effort send to the connected peer. A datagram error is recorded and
    /// counted as a local drop; higher layers may retransmit it.
    fn send(&mut self, bytes: &[u8]) {
        if !self.connected {
            self.last_error = Some(io::ErrorKind::NotConnected);
            self.dropped = self.dropped.saturating_add(1);
            return;
        }
        if let Err(error) = self.socket.send(bytes) {
            self.last_error = Some(error.kind());
            self.dropped = self.dropped.saturating_add(1);
        }
    }

    /// Take the next datagram from the connected peer, or `None` when none has
    /// arrived (`WouldBlock` / timeout). Connecting the socket makes the OS
    /// discard datagrams from other peers before they reach this seam.
    fn recv(&mut self) -> Option<Vec<u8>> {
        match self.socket.recv(&mut self.buffer) {
            Ok(len) => {
                self.received = self.received.saturating_add(1);
                Some(self.buffer[..len].to_vec())
            }
            Err(ref error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
                ) =>
            {
                None
            }
            Err(error) => {
                self.last_error = Some(error.kind());
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Poll a non-blocking transport for up to ~1s (loopback UDP is immediate,
    /// but the scheduler may delay it a moment).
    fn recv_within(mut transport: impl FnMut() -> Option<Vec<u8>>) -> Option<Vec<u8>> {
        for _ in 0..1_000 {
            if let Some(bytes) = transport() {
                return Some(bytes);
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        None
    }

    /// Whether a bind failure must fail the test instead of skipping it.
    fn require_net() -> bool {
        std::env::var_os("PRISM_REQUIRE_NET").is_some()
    }

    /// Bind a loopback socket, or skip when the environment forbids sockets (a
    /// sandbox denying binds). `PRISM_REQUIRE_NET=1` makes the skip a failure,
    /// so a CI run can prove the socket path really executed.
    fn bind_or_skip() -> Option<UdpTransport> {
        match UdpTransport::bind("127.0.0.1:0") {
            Ok(transport) => Some(transport),
            Err(error) => {
                assert!(
                    !require_net(),
                    "PRISM_REQUIRE_NET is set but binding a UDP socket failed: {error}"
                );
                None
            }
        }
    }

    fn bind_multi_or_skip(config: UdpMultiPeerConfig) -> Option<UdpMultiPeer> {
        match UdpMultiPeer::bind("127.0.0.1:0", config) {
            Ok(endpoint) => Some(endpoint),
            Err(error) => {
                assert!(
                    !require_net(),
                    "PRISM_REQUIRE_NET is set but binding a multi-peer UDP socket failed: {error}"
                );
                None
            }
        }
    }

    fn recv_socket_within(socket: &UdpSocket) -> Option<Vec<u8>> {
        let mut buffer = vec![0_u8; MAX_DATAGRAM];
        recv_within(|| match socket.recv_from(&mut buffer) {
            Ok((len, _peer)) => Some(buffer[..len].to_vec()),
            Err(ref error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
                ) =>
            {
                None
            }
            Err(error) => panic!("unexpected UDP receive error: {error}"),
        })
    }

    fn poll_within(endpoint: &UdpMultiPeer, expected: usize) -> usize {
        let mut total = 0;
        for _ in 0..1_000 {
            total += endpoint.poll();
            if total >= expected {
                return total;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        total
    }

    fn bind_client_or_skip() -> Option<UdpSocket> {
        match UdpSocket::bind("127.0.0.1:0") {
            Ok(socket) => {
                socket
                    .set_nonblocking(true)
                    .expect("loopback client nonblocking");
                Some(socket)
            }
            Err(error) => {
                assert!(
                    !require_net(),
                    "PRISM_REQUIRE_NET is set but binding a UDP client socket failed: {error}"
                );
                None
            }
        }
    }

    #[test]
    fn two_udp_transports_exchange_a_datagram_over_loopback() {
        let Some(mut a) = bind_or_skip() else {
            return; // no socket permission here; the wiring is exercised elsewhere
        };
        let Some(mut b) = bind_or_skip() else {
            return;
        };
        let a_addr = a.local_addr().expect("local addr");
        let b_addr = b.local_addr().expect("local addr");
        a.connect_to(b_addr).expect("peer");
        b.connect_to(a_addr).expect("peer");
        assert!(a.is_connected());
        assert!(b.is_connected());

        a.send(b"hello over udp");
        let received = recv_within(|| b.recv());
        assert_eq!(received, Some(b"hello over udp".to_vec()));

        // Nothing else is pending.
        assert_eq!(b.recv(), None);
        assert_eq!(a.dropped(), 0);
        assert_eq!(b.received(), 1);
    }

    #[test]
    fn an_unconnected_socket_reports_send_failure() {
        let Some(mut transport) = bind_or_skip() else {
            return;
        };
        transport.send(b"not connected");
        assert_eq!(transport.dropped(), 1);
        assert_eq!(
            transport.take_error_kind(),
            Some(io::ErrorKind::NotConnected)
        );
    }

    #[test]
    fn multi_peer_demuxes_two_clients_and_sends_replies_to_each() {
        let Some(endpoint) = bind_multi_or_skip(UdpMultiPeerConfig::default()) else {
            return;
        };
        let Some(client_a) = bind_client_or_skip() else {
            return;
        };
        let Some(client_b) = bind_client_or_skip() else {
            return;
        };
        let endpoint_addr = endpoint.local_addr().expect("endpoint addr");
        let client_a_addr = client_a.local_addr().expect("client A addr");
        let client_b_addr = client_b.local_addr().expect("client B addr");
        let mut peer_a = endpoint.peer(client_a_addr).expect("admit client A");
        let mut peer_b = endpoint.peer(client_b_addr).expect("admit client B");

        client_a
            .send_to(b"from A", endpoint_addr)
            .expect("client A send");
        client_b
            .send_to(b"from B", endpoint_addr)
            .expect("client B send");
        assert_eq!(poll_within(&endpoint, 2), 2);
        assert_eq!(peer_a.recv(), Some(b"from A".to_vec()));
        assert_eq!(peer_b.recv(), Some(b"from B".to_vec()));
        assert_eq!(peer_a.received(), 1);
        assert_eq!(peer_b.received(), 1);

        peer_a.send(b"reply A");
        peer_b.send(b"reply B");
        assert_eq!(recv_socket_within(&client_a), Some(b"reply A".to_vec()));
        assert_eq!(recv_socket_within(&client_b), Some(b"reply B".to_vec()));
        assert_eq!(endpoint.dropped(), 0);
    }

    #[test]
    fn multi_peer_drops_unknown_peers_without_admitting_them() {
        let Some(endpoint) = bind_multi_or_skip(UdpMultiPeerConfig::default()) else {
            return;
        };
        let Some(unknown) = bind_client_or_skip() else {
            return;
        };
        unknown
            .send_to(b"unknown", endpoint.local_addr().expect("endpoint addr"))
            .expect("unknown send");
        assert_eq!(poll_within(&endpoint, 1), 1);
        assert_eq!(endpoint.dropped(), 1);
        assert!(endpoint
            .peer(unknown.local_addr().expect("unknown addr"))
            .is_ok());
    }

    #[test]
    fn multi_peer_enforces_peer_and_queue_capacity() {
        let config = UdpMultiPeerConfig {
            max_peers: 1,
            max_queue_per_peer: 1,
            max_total_queued: 1,
        };
        let Some(endpoint) = bind_multi_or_skip(config) else {
            return;
        };
        let Some(client_a) = bind_client_or_skip() else {
            return;
        };
        let Some(client_b) = bind_client_or_skip() else {
            return;
        };
        let endpoint_addr = endpoint.local_addr().expect("endpoint addr");
        let client_a_addr = client_a.local_addr().expect("client A addr");
        let client_b_addr = client_b.local_addr().expect("client B addr");
        let mut peer_a = endpoint.peer(client_a_addr).expect("admit client A");
        match endpoint.peer(client_b_addr) {
            Err(error) => assert_eq!(error.kind(), io::ErrorKind::WouldBlock),
            Ok(_) => panic!("peer capacity must reject the second peer"),
        }

        client_a
            .send_to(b"kept", endpoint_addr)
            .expect("first send");
        client_a
            .send_to(b"dropped", endpoint_addr)
            .expect("second send");
        assert_eq!(poll_within(&endpoint, 2), 2);
        assert_eq!(peer_a.recv(), Some(b"kept".to_vec()));
        assert_eq!(peer_a.recv(), None);
        assert_eq!(endpoint.dropped(), 1);
    }
}
