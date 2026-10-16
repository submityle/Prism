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
use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hash, Hasher};
use std::io;
use std::net::{SocketAddr, ToSocketAddrs, UdpSocket};
use std::sync::Mutex;
use std::time::Duration;

use crate::transport::{Transport, TransportMigrationCapability};

/// The largest UDP payload (the IPv4 limit): a datagram larger than this can
/// never arrive, so it bounds the receive buffer.
pub const MAX_DATAGRAM: usize = 65_507;

const PATH_CHALLENGE_MAGIC: [u8; 4] = *b"PRBV";
const PATH_CHALLENGE_LEN: usize = PATH_CHALLENGE_MAGIC.len() + 16;

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

/// Stable identity for an admitted peer.
///
/// The identity is local to one [`UdpMultiPeer`] endpoint. It is deliberately
/// not derived from an IP address or port, so an authorized endpoint rebind
/// does not force the replication/session layers to be recreated.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct UdpPeerId(u64);

impl UdpPeerId {
    /// Numeric representation useful for diagnostics and persistence in a
    /// caller-owned connection table.
    #[must_use]
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

/// State of an admitted peer's endpoint binding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UdpPeerBindingState {
    /// The peer is bound to one endpoint and has no pending migration.
    Stable {
        /// Current source/destination endpoint.
        endpoint: SocketAddr,
        /// Monotonically increasing binding generation.
        generation: u64,
    },
    /// A candidate endpoint is reserved but has not been committed.
    Rebinding {
        /// Endpoint still used for sends and accepted receives.
        current: SocketAddr,
        /// Candidate endpoint reserved for this peer.
        candidate: SocketAddr,
        /// Generation that the returned ticket must match.
        generation: u64,
    },
}

/// An opaque, one-use authorization handle for a local endpoint rebind.
///
/// The candidate must echo [`Self::challenge_packet`] from the candidate
/// endpoint before [`UdpPeerTransport::commit_rebind`] succeeds. This proves
/// reachability of that endpoint; caller-owned authentication and authorization
/// remain separate concerns.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct UdpPeerRebind {
    peer_id: UdpPeerId,
    from: SocketAddr,
    to: SocketAddr,
    generation: u64,
    challenge: [u8; 16],
}

impl UdpPeerRebind {
    /// Bytes a candidate endpoint must echo to prove it can receive traffic.
    #[must_use]
    pub fn challenge_packet(&self) -> [u8; PATH_CHALLENGE_LEN] {
        path_challenge_packet(self.challenge)
    }
}

/// Errors returned by the bounded peer endpoint migration state machine.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UdpPeerMigrationError {
    /// The transport view no longer refers to an admitted peer.
    UnknownPeer,
    /// The candidate endpoint is unspecified or uses port zero.
    InvalidEndpoint,
    /// The candidate endpoint is already active or reserved by another peer.
    EndpointInUse,
    /// This peer already has a candidate endpoint reserved.
    MigrationInProgress,
    /// The ticket does not match the current binding generation/state.
    StaleMigration,
    /// The candidate endpoint has not completed the challenge/response step.
    PathNotValidated,
}

impl core::fmt::Display for UdpPeerMigrationError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let message = match self {
            Self::UnknownPeer => "UDP peer is no longer admitted",
            Self::InvalidEndpoint => "UDP peer endpoint must be specified and use a non-zero port",
            Self::EndpointInUse => "UDP peer endpoint is already active or reserved",
            Self::MigrationInProgress => "UDP peer already has a rebind in progress",
            Self::StaleMigration => "UDP peer rebind ticket is stale",
            Self::PathNotValidated => "UDP peer rebind candidate path is not validated",
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for UdpPeerMigrationError {}

struct PeerEntry {
    endpoint: SocketAddr,
    queue: VecDeque<Vec<u8>>,
    generation: u64,
    pending_rebind: Option<UdpPeerRebind>,
    path_validated: bool,
}

struct MultiPeerState {
    peers: BTreeMap<UdpPeerId, PeerEntry>,
    endpoints: BTreeMap<SocketAddr, UdpPeerId>,
    reserved_endpoints: BTreeMap<SocketAddr, UdpPeerId>,
    next_peer_id: u64,
    total_queued: usize,
    dropped: u64,
    last_error: Option<io::ErrorKind>,
    challenge_key: RandomState,
    challenge_counter: u64,
}

impl MultiPeerState {
    fn new() -> Self {
        Self {
            peers: BTreeMap::new(),
            endpoints: BTreeMap::new(),
            reserved_endpoints: BTreeMap::new(),
            next_peer_id: 1,
            total_queued: 0,
            dropped: 0,
            last_error: None,
            challenge_key: RandomState::new(),
            challenge_counter: 0,
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
///
/// An admitted peer can be explicitly rebound to a new endpoint through
/// [`UdpPeerTransport::request_rebind`], [`UdpPeerTransport::send_rebind_challenge`],
/// and a candidate echo before [`UdpPeerTransport::commit_rebind`]. The stable
/// [`UdpPeerId`] and all queued datagrams survive the move. This validates
/// candidate reachability, but is not NAT/ICE/relay or a replacement for QUIC
/// connection migration.
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
        if state.reserved_endpoints.contains_key(&peer) {
            return Err(io::Error::new(
                io::ErrorKind::AddrInUse,
                "multi-peer UDP endpoint is reserved for a pending rebind",
            ));
        }
        let peer_id = if let Some(peer_id) = state.endpoints.get(&peer).copied() {
            peer_id
        } else {
            if state.peers.len() >= self.config.max_peers {
                return Err(io::Error::new(
                    io::ErrorKind::WouldBlock,
                    "multi-peer UDP peer capacity is full",
                ));
            }
            let peer_id = UdpPeerId(state.next_peer_id);
            state.next_peer_id = state
                .next_peer_id
                .checked_add(1)
                .ok_or_else(|| io::Error::other("multi-peer UDP peer identity exhausted"))?;
            state.endpoints.insert(peer, peer_id);
            state.peers.insert(
                peer_id,
                PeerEntry {
                    endpoint: peer,
                    queue: VecDeque::new(),
                    generation: 0,
                    pending_rebind: None,
                    path_validated: false,
                },
            );
            peer_id
        };
        drop(state);
        Ok(UdpPeerTransport {
            socket: Arc::clone(&self.socket),
            state: Arc::clone(&self.state),
            config: self.config,
            peer_id,
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
                    if state.reserved_endpoints.contains_key(&peer) {
                        if !state.consume_path_response(peer, &buffer[..len]) {
                            state.dropped = state.dropped.saturating_add(1);
                        }
                        continue;
                    }
                    let total_full = state.total_queued >= self.config.max_total_queued;
                    let Some(peer_id) = state.endpoints.get(&peer).copied() else {
                        state.dropped = state.dropped.saturating_add(1);
                        continue;
                    };
                    let queue_full = state
                        .peers
                        .get(&peer_id)
                        .is_none_or(|entry| entry.queue.len() >= self.config.max_queue_per_peer);
                    if queue_full || total_full {
                        state.dropped = state.dropped.saturating_add(1);
                        continue;
                    }
                    state
                        .peers
                        .get_mut(&peer_id)
                        .expect("endpoint map points at an admitted peer")
                        .queue
                        .push_back(buffer[..len].to_vec());
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

impl MultiPeerState {
    fn make_challenge(
        &mut self,
        peer_id: UdpPeerId,
        from: SocketAddr,
        to: SocketAddr,
        generation: u64,
    ) -> [u8; 16] {
        self.challenge_counter = self.challenge_counter.wrapping_add(1);
        let counter = self.challenge_counter;
        let mut first = self.challenge_key.build_hasher();
        0_u8.hash(&mut first);
        peer_id.hash(&mut first);
        from.hash(&mut first);
        to.hash(&mut first);
        generation.hash(&mut first);
        counter.hash(&mut first);
        let mut second = self.challenge_key.build_hasher();
        1_u8.hash(&mut second);
        peer_id.hash(&mut second);
        from.hash(&mut second);
        to.hash(&mut second);
        generation.hash(&mut second);
        counter.hash(&mut second);
        let mut challenge = [0_u8; 16];
        challenge[..8].copy_from_slice(&first.finish().to_le_bytes());
        challenge[8..].copy_from_slice(&second.finish().to_le_bytes());
        challenge
    }

    fn consume_path_response(&mut self, source: SocketAddr, bytes: &[u8]) -> bool {
        let Some(peer_id) = self.reserved_endpoints.get(&source).copied() else {
            return false;
        };
        let Some(entry) = self.peers.get_mut(&peer_id) else {
            return false;
        };
        let Some(migration) = entry.pending_rebind else {
            return false;
        };
        if migration.to != source || !valid_path_challenge(bytes, migration.challenge) {
            return false;
        }
        entry.path_validated = true;
        true
    }
}

/// A [`Transport`] view of one admitted peer on a [`UdpMultiPeer`] endpoint.
pub struct UdpPeerTransport {
    socket: Arc<UdpSocket>,
    state: Arc<Mutex<MultiPeerState>>,
    config: UdpMultiPeerConfig,
    peer_id: UdpPeerId,
    dropped: u64,
    received: u64,
    last_error: Option<io::ErrorKind>,
}

impl UdpPeerTransport {
    /// The remote peer address represented by this transport.
    #[must_use]
    pub fn peer(&self) -> SocketAddr {
        lock(&self.state)
            .peers
            .get(&self.peer_id)
            .map_or_else(unspecified_peer, |entry| entry.endpoint)
    }

    /// Stable identity for this admitted peer.
    #[must_use]
    pub const fn id(&self) -> UdpPeerId {
        self.peer_id
    }

    /// Current endpoint binding state.
    #[must_use]
    pub fn binding_state(&self) -> UdpPeerBindingState {
        let state = lock(&self.state);
        let Some(entry) = state.peers.get(&self.peer_id) else {
            return UdpPeerBindingState::Stable {
                endpoint: unspecified_peer(),
                generation: 0,
            };
        };
        entry.pending_rebind.map_or(
            UdpPeerBindingState::Stable {
                endpoint: entry.endpoint,
                generation: entry.generation,
            },
            |migration| UdpPeerBindingState::Rebinding {
                current: entry.endpoint,
                candidate: migration.to,
                generation: entry.generation,
            },
        )
    }

    /// Reserve a new endpoint for this peer without changing traffic routing.
    ///
    /// The returned ticket is a local state-machine handle, not a cryptographic
    /// proof. Callers must only create it after an authenticated migration
    /// request and must validate the candidate path before committing it.
    pub fn request_rebind(
        &self,
        new_peer: SocketAddr,
    ) -> Result<UdpPeerRebind, UdpPeerMigrationError> {
        if !valid_peer_endpoint(new_peer) {
            return Err(UdpPeerMigrationError::InvalidEndpoint);
        }
        let mut state = lock(&self.state);
        let current = state
            .peers
            .get(&self.peer_id)
            .ok_or(UdpPeerMigrationError::UnknownPeer)?
            .endpoint;
        if current == new_peer {
            return Err(UdpPeerMigrationError::InvalidEndpoint);
        }
        if state
            .peers
            .get(&self.peer_id)
            .and_then(|entry| entry.pending_rebind)
            .is_some()
        {
            return Err(UdpPeerMigrationError::MigrationInProgress);
        }
        if state.endpoints.contains_key(&new_peer)
            || state.reserved_endpoints.contains_key(&new_peer)
        {
            return Err(UdpPeerMigrationError::EndpointInUse);
        }
        let generation = state
            .peers
            .get(&self.peer_id)
            .expect("peer checked above")
            .generation;
        let challenge = state.make_challenge(self.peer_id, current, new_peer, generation);
        let migration = UdpPeerRebind {
            peer_id: self.peer_id,
            from: current,
            to: new_peer,
            generation,
            challenge,
        };
        state.reserved_endpoints.insert(new_peer, self.peer_id);
        state
            .peers
            .get_mut(&self.peer_id)
            .expect("peer checked above")
            .pending_rebind = Some(migration);
        state
            .peers
            .get_mut(&self.peer_id)
            .expect("peer checked above")
            .path_validated = false;
        Ok(migration)
    }

    /// Send the challenge packet to the candidate endpoint.
    pub fn send_rebind_challenge(&self, migration: UdpPeerRebind) -> io::Result<()> {
        let destination = {
            let state = lock(&self.state);
            let entry = state.peers.get(&self.peer_id).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotConnected,
                    UdpPeerMigrationError::UnknownPeer,
                )
            })?;
            if migration.peer_id != self.peer_id
                || entry.pending_rebind != Some(migration)
                || entry.generation != migration.generation
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    UdpPeerMigrationError::StaleMigration,
                ));
            }
            migration.to
        };
        self.socket
            .send_to(&migration.challenge_packet(), destination)
            .map(|_| ())
    }

    /// Commit a previously requested endpoint rebind atomically.
    pub fn commit_rebind(&self, migration: UdpPeerRebind) -> Result<(), UdpPeerMigrationError> {
        let mut state = lock(&self.state);
        if migration.peer_id != self.peer_id {
            return Err(UdpPeerMigrationError::StaleMigration);
        }
        let entry = state
            .peers
            .get(&self.peer_id)
            .ok_or(UdpPeerMigrationError::UnknownPeer)?;
        if entry.pending_rebind != Some(migration) || entry.generation != migration.generation {
            return Err(UdpPeerMigrationError::StaleMigration);
        }
        if !entry.path_validated {
            return Err(UdpPeerMigrationError::PathNotValidated);
        }
        if state
            .endpoints
            .get(&migration.to)
            .is_some_and(|peer_id| *peer_id != self.peer_id)
        {
            return Err(UdpPeerMigrationError::EndpointInUse);
        }
        state.endpoints.remove(&migration.from);
        state.reserved_endpoints.remove(&migration.to);
        state.endpoints.insert(migration.to, self.peer_id);
        let entry = state
            .peers
            .get_mut(&self.peer_id)
            .expect("peer checked above");
        entry.endpoint = migration.to;
        entry.generation = entry.generation.saturating_add(1);
        entry.pending_rebind = None;
        Ok(())
    }

    /// Cancel a pending rebind and release its endpoint reservation.
    pub fn cancel_rebind(&self, migration: UdpPeerRebind) -> Result<(), UdpPeerMigrationError> {
        let mut state = lock(&self.state);
        if migration.peer_id != self.peer_id {
            return Err(UdpPeerMigrationError::StaleMigration);
        }
        let entry = state
            .peers
            .get(&self.peer_id)
            .ok_or(UdpPeerMigrationError::UnknownPeer)?;
        if entry.pending_rebind != Some(migration) || entry.generation != migration.generation {
            return Err(UdpPeerMigrationError::StaleMigration);
        }
        state.reserved_endpoints.remove(&migration.to);
        state
            .peers
            .get_mut(&self.peer_id)
            .expect("peer checked above")
            .pending_rebind = None;
        state
            .peers
            .get_mut(&self.peer_id)
            .expect("peer checked above")
            .path_validated = false;
        Ok(())
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
        let Some(peer) = lock(&self.state)
            .peers
            .get(&self.peer_id)
            .map(|entry| entry.endpoint)
        else {
            self.last_error = Some(io::ErrorKind::NotConnected);
            self.dropped = self.dropped.saturating_add(1);
            return;
        };
        if let Err(error) = self.socket.send_to(bytes, peer) {
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
                    if state.reserved_endpoints.contains_key(&peer) {
                        if !state.consume_path_response(peer, &buffer[..len]) {
                            state.dropped = state.dropped.saturating_add(1);
                        }
                        continue;
                    }
                    let total_full = state.total_queued >= self.config.max_total_queued;
                    let Some(peer_id) = state.endpoints.get(&peer).copied() else {
                        state.dropped = state.dropped.saturating_add(1);
                        continue;
                    };
                    let queue_full = state
                        .peers
                        .get(&peer_id)
                        .is_none_or(|entry| entry.queue.len() >= self.config.max_queue_per_peer);
                    if queue_full || total_full {
                        state.dropped = state.dropped.saturating_add(1);
                        continue;
                    }
                    state
                        .peers
                        .get_mut(&peer_id)
                        .expect("endpoint map points at an admitted peer")
                        .queue
                        .push_back(buffer[..len].to_vec());
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
        let message = state.peers.get_mut(&self.peer_id)?.queue.pop_front();
        if message.is_some() {
            state.total_queued = state.total_queued.saturating_sub(1);
            self.received = self.received.saturating_add(1);
        }
        message
    }

    fn migration_capability(&self) -> TransportMigrationCapability {
        TransportMigrationCapability::IdentityPreservingPeerRebind
    }
}

fn valid_peer_endpoint(endpoint: SocketAddr) -> bool {
    endpoint.port() != 0 && !endpoint.ip().is_unspecified()
}

fn path_challenge_packet(challenge: [u8; 16]) -> [u8; PATH_CHALLENGE_LEN] {
    let mut packet = [0_u8; PATH_CHALLENGE_LEN];
    packet[..PATH_CHALLENGE_MAGIC.len()].copy_from_slice(&PATH_CHALLENGE_MAGIC);
    packet[PATH_CHALLENGE_MAGIC.len()..].copy_from_slice(&challenge);
    packet
}

fn valid_path_challenge(bytes: &[u8], challenge: [u8; 16]) -> bool {
    bytes == path_challenge_packet(challenge)
}

fn unspecified_peer() -> SocketAddr {
    SocketAddr::from(([0, 0, 0, 0], 0))
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

    #[test]
    fn multi_peer_rebind_preserves_identity_queue_and_transport_capability() {
        let Some(endpoint) = bind_multi_or_skip(UdpMultiPeerConfig::default()) else {
            return;
        };
        let Some(old_client) = bind_client_or_skip() else {
            return;
        };
        let Some(new_client) = bind_client_or_skip() else {
            return;
        };
        let endpoint_addr = endpoint.local_addr().expect("endpoint addr");
        let old_addr = old_client.local_addr().expect("old client addr");
        let new_addr = new_client.local_addr().expect("new client addr");
        let mut peer = endpoint.peer(old_addr).expect("admit old endpoint");
        let peer_id = peer.id();

        assert_eq!(
            peer.migration_capability(),
            TransportMigrationCapability::IdentityPreservingPeerRebind
        );
        old_client
            .send_to(b"before-rebind", endpoint_addr)
            .expect("old client send");
        assert_eq!(poll_within(&endpoint, 1), 1);

        let ticket = peer.request_rebind(new_addr).expect("reserve new endpoint");
        assert_eq!(peer.id(), peer_id);
        assert_eq!(
            peer.binding_state(),
            UdpPeerBindingState::Rebinding {
                current: old_addr,
                candidate: new_addr,
                generation: 0,
            }
        );
        assert_eq!(
            peer.request_rebind(new_addr),
            Err(UdpPeerMigrationError::MigrationInProgress)
        );

        // A candidate is not active until the caller commits an authenticated
        // path decision; traffic from it is still treated as unknown.
        new_client
            .send_to(b"before-commit", endpoint_addr)
            .expect("new client send");
        assert_eq!(poll_within(&endpoint, 1), 1);
        assert_eq!(peer.recv(), Some(b"before-rebind".to_vec()));
        assert_eq!(peer.recv(), None);

        assert_eq!(
            peer.commit_rebind(ticket),
            Err(UdpPeerMigrationError::PathNotValidated)
        );
        peer.send_rebind_challenge(ticket).expect("send challenge");
        let mut challenge_buffer = [0_u8; MAX_DATAGRAM];
        let challenge = recv_within(|| match new_client.recv_from(&mut challenge_buffer) {
            Ok((len, _)) => Some(challenge_buffer[..len].to_vec()),
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => None,
            Err(error) => panic!("unexpected challenge receive error: {error}"),
        })
        .expect("receive challenge");
        new_client
            .send_to(&challenge, endpoint_addr)
            .expect("echo challenge");
        assert_eq!(poll_within(&endpoint, 1), 1);
        peer.commit_rebind(ticket).expect("commit rebind");
        assert_eq!(peer.id(), peer_id);
        assert_eq!(peer.peer(), new_addr);
        assert_eq!(
            peer.binding_state(),
            UdpPeerBindingState::Stable {
                endpoint: new_addr,
                generation: 1,
            }
        );

        old_client
            .send_to(b"old-endpoint", endpoint_addr)
            .expect("old client send after migration");
        new_client
            .send_to(b"after-rebind", endpoint_addr)
            .expect("new client send after migration");
        assert_eq!(poll_within(&endpoint, 2), 2);
        assert_eq!(peer.recv(), Some(b"after-rebind".to_vec()));
        assert_eq!(peer.recv(), None);
        peer.send(b"reply-after-rebind");
        assert_eq!(
            recv_socket_within(&new_client),
            Some(b"reply-after-rebind".to_vec())
        );
        assert_eq!(
            peer.id(),
            endpoint.peer(new_addr).expect("same endpoint").id()
        );
    }

    #[test]
    fn multi_peer_rebind_enforces_endpoint_reservation_and_stale_tickets() {
        let Some(endpoint) = bind_multi_or_skip(UdpMultiPeerConfig::default()) else {
            return;
        };
        let Some(client_a) = bind_client_or_skip() else {
            return;
        };
        let Some(client_b) = bind_client_or_skip() else {
            return;
        };
        let Some(candidate) = bind_client_or_skip() else {
            return;
        };
        let peer_a = endpoint
            .peer(client_a.local_addr().expect("client A addr"))
            .expect("admit A");
        let peer_b = endpoint
            .peer(client_b.local_addr().expect("client B addr"))
            .expect("admit B");

        assert_eq!(
            peer_a.request_rebind(peer_b.peer()),
            Err(UdpPeerMigrationError::EndpointInUse)
        );
        let candidate_addr = candidate.local_addr().expect("candidate addr");
        let ticket = peer_a
            .request_rebind(candidate_addr)
            .expect("reserve candidate");
        match endpoint.peer(candidate_addr) {
            Err(error) => assert_eq!(error.kind(), io::ErrorKind::AddrInUse),
            Ok(_) => panic!("pending rebind must reserve the candidate endpoint"),
        }
        assert_eq!(
            peer_b.request_rebind(candidate_addr),
            Err(UdpPeerMigrationError::EndpointInUse)
        );
        assert_eq!(
            peer_a.cancel_rebind(ticket),
            Ok(()),
            "cancellation releases the reservation"
        );
        assert_eq!(
            peer_a.commit_rebind(ticket),
            Err(UdpPeerMigrationError::StaleMigration)
        );
        let second_ticket = peer_a
            .request_rebind(candidate_addr)
            .expect("candidate released after cancel");
        assert_eq!(
            peer_a.commit_rebind(second_ticket),
            Err(UdpPeerMigrationError::PathNotValidated)
        );
        peer_a
            .send_rebind_challenge(second_ticket)
            .expect("send second challenge");
        let mut challenge_buffer = [0_u8; MAX_DATAGRAM];
        let challenge = recv_within(|| match candidate.recv_from(&mut challenge_buffer) {
            Ok((len, _)) => Some(challenge_buffer[..len].to_vec()),
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => None,
            Err(error) => panic!("unexpected challenge receive error: {error}"),
        })
        .expect("receive second challenge");
        candidate
            .send_to(&challenge, endpoint.local_addr().expect("endpoint address"))
            .expect("echo second challenge");
        let mut committed = false;
        for _ in 0..1_000 {
            endpoint.poll();
            if peer_a.commit_rebind(second_ticket).is_ok() {
                committed = true;
                break;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(committed, "second challenge response did not validate");
    }
}
