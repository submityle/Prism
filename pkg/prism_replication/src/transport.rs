//! The transport seam and a deterministic in-memory link (design §6).
//!
//! Real multiplayer rides a socket (reliable UDP / QUIC), which this crate does
//! not depend on. What it *can* define without a network — and test on one
//! machine — is the shape of a transport and a **loopback link** with a
//! deterministic fault model (latency, jitter, loss). That link is a real
//! bidirectional byte channel usable for local play and tests, and it lets the
//! [`crate::sequence`] delivery window be exercised against reordering and loss
//! exactly as it would be over UDP.
//!
//! A message is opaque bytes: the sequencing / delta layers above decide their
//! own framing. Delivery within a [`poll`](LoopbackLink::poll) is ordered by
//! `(deliver_at, send order)`, so jitter deliberately reorders messages relative
//! to send order — and a given configuration always reorders and drops the
//! *same* messages, so runs are reproducible.

use alloc::boxed::Box;
use alloc::vec::Vec;

use crate::security::SecurityError;

/// Whether a transport can carry a replication session safely.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransportSecurity {
    /// Raw bytes with no authenticated session boundary.
    Unprotected,
    /// A secure wrapper is present but its handshake is still in flight.
    Pending,
    /// The authenticated session is established.
    Established,
    /// The secure wrapper rejected the peer or a protected datagram.
    Rejected,
}

/// Endpoint migration capabilities exposed by a concrete transport.
///
/// The generic transport seam deliberately does not assume that a provider can
/// prove path ownership or preserve a peer identity across endpoint changes.
/// Callers must inspect this capability and use the provider-specific
/// authorization/control plane before attempting a migration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransportMigrationCapability {
    /// The transport has no endpoint migration operation.
    Unsupported,
    /// The provider can atomically rebind an admitted peer identity to a new
    /// endpoint while retaining its transport object and queued datagrams.
    IdentityPreservingPeerRebind,
}

/// The transport provider seam (design §6): bytes out, bytes in.
///
/// A real provider (reliable UDP / QUIC) implements this over a socket; the
/// in-memory pair below implements it over a shared queue, so an app can be wired
/// end-to-end without a network. `Transport` is `no_std`-clean — the std-only
/// pieces are the concrete in-memory endpoints.
pub trait Transport: Send + Sync {
    /// Send `bytes` to the peer.
    fn send(&mut self, bytes: &[u8]);
    /// Take the next inbound message, if any.
    fn recv(&mut self) -> Option<Vec<u8>>;

    /// Whether this transport has an explicit authenticated-session wrapper.
    /// Plain transports deliberately return `false`; replication constructors
    /// reject them instead of silently sending gameplay data in cleartext.
    fn supports_secure_session(&self) -> bool {
        false
    }

    /// Current security state for diagnostics and plugin fail-closed behavior.
    fn security_state(&self) -> TransportSecurity {
        TransportSecurity::Unprotected
    }

    /// Bytes added around a multiplex payload by the security wrapper. The
    /// multiplex uses this to keep the physical datagram within the path MTU.
    fn security_overhead(&self) -> usize {
        0
    }

    /// Retrieve a provider/wire error without changing the basic `Transport`
    /// send/recv ABI.
    fn take_security_error(&mut self) -> Option<SecurityError> {
        None
    }

    /// Whether this provider exposes an endpoint migration operation.
    ///
    /// This is only a capability declaration. It is not proof that a new
    /// endpoint belongs to the peer; authentication and path validation remain
    /// the caller's responsibility.
    fn migration_capability(&self) -> TransportMigrationCapability {
        TransportMigrationCapability::Unsupported
    }
}

/// A boxed transport is itself a transport, so a caller can erase the concrete
/// provider (a [`reliable::ReliableChannel`](crate::reliable::ReliableChannel)
/// over a `Box<dyn Transport>` still works).
impl<T: Transport + ?Sized> Transport for Box<T> {
    fn send(&mut self, bytes: &[u8]) {
        (**self).send(bytes);
    }
    fn recv(&mut self) -> Option<Vec<u8>> {
        (**self).recv()
    }

    fn supports_secure_session(&self) -> bool {
        (**self).supports_secure_session()
    }

    fn security_state(&self) -> TransportSecurity {
        (**self).security_state()
    }

    fn security_overhead(&self) -> usize {
        (**self).security_overhead()
    }

    fn take_security_error(&mut self) -> Option<SecurityError> {
        (**self).take_security_error()
    }

    fn migration_capability(&self) -> TransportMigrationCapability {
        (**self).migration_capability()
    }
}

/// A shared in-memory [`Transport`] pair (std): one endpoint's `send` lands in the
/// other's inbox immediately, so an app can be driven end-to-end in tests and
/// local play without a socket.
#[cfg(feature = "std")]
pub mod memory {
    use super::Transport;
    use alloc::collections::VecDeque;
    use alloc::sync::Arc;
    use alloc::vec::Vec;
    use std::sync::Mutex;

    #[derive(Default)]
    struct Inboxes {
        to_a: VecDeque<Vec<u8>>,
        to_b: VecDeque<Vec<u8>>,
    }

    /// One end of a shared in-memory transport pair.
    #[derive(Clone)]
    pub struct InMemoryTransport {
        inboxes: Arc<Mutex<Inboxes>>,
        as_a: bool,
    }

    impl InMemoryTransport {
        /// Two connected endpoints `(a, b)`.
        #[must_use]
        pub fn pair() -> (Self, Self) {
            let inboxes = Arc::new(Mutex::new(Inboxes::default()));
            (
                Self {
                    inboxes: Arc::clone(&inboxes),
                    as_a: true,
                },
                Self {
                    inboxes,
                    as_a: false,
                },
            )
        }

        /// Test-only authenticated pair used by replication plugin tests. The
        /// ordinary [`pair`](Self::pair) remains a raw transport and is useful
        /// for testing this lower-level byte seam itself.
        #[cfg(test)]
        pub(crate) fn pair_secure() -> (
            crate::security::SecureTransport<Self, crate::security::test_support::Provider>,
            crate::security::SecureTransport<Self, crate::security::test_support::Provider>,
        ) {
            let (client, server) = Self::pair();
            (
                crate::security::SecureTransport::new_client(
                    client,
                    crate::security::test_support::Provider::new(0x31, 1),
                )
                .expect("test provider creates a non-zero client nonce"),
                crate::security::SecureTransport::new_server(
                    server,
                    crate::security::test_support::Provider::new(0x31, 9),
                ),
            )
        }

        /// How many messages are waiting to be received by this endpoint.
        #[must_use]
        pub fn pending(&self) -> usize {
            let inboxes = lock(&self.inboxes);
            if self.as_a {
                inboxes.to_a.len()
            } else {
                inboxes.to_b.len()
            }
        }
    }

    impl Transport for InMemoryTransport {
        fn send(&mut self, bytes: &[u8]) {
            let mut inboxes = lock(&self.inboxes);
            let target = if self.as_a {
                &mut inboxes.to_b
            } else {
                &mut inboxes.to_a
            };
            target.push_back(bytes.to_vec());
        }

        fn recv(&mut self) -> Option<Vec<u8>> {
            let mut inboxes = lock(&self.inboxes);
            let source = if self.as_a {
                &mut inboxes.to_a
            } else {
                &mut inboxes.to_b
            };
            source.pop_front()
        }
    }

    fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
        mutex
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn the_pair_delivers_each_way_once() {
            let (mut a, mut b) = InMemoryTransport::pair();
            a.send(b"from-a"); // lands in B's inbox
            b.send(b"from-b"); // lands in A's inbox
            assert_eq!(b.pending(), 1, "B has A's message");
            assert_eq!(a.pending(), 1, "A has B's message");
            assert_eq!(a.recv(), Some(b"from-b".to_vec()));
            assert_eq!(b.recv(), Some(b"from-a".to_vec()));
            assert_eq!(a.recv(), None);
            assert_eq!(b.pending(), 0);
        }
    }
}

/// A bounded cross-thread byte transport backed by standard-library channels.
///
/// Each endpoint owns one bounded outbound queue and one inbound receiver. The
/// implementation is `Send + Sync`, so one endpoint can be moved to a worker
/// thread while the peer stays on the app thread. `send` is deliberately
/// non-blocking: a full or disconnected queue drops the datagram and increments
/// [`ThreadTransport::dropped`]. Reliable delivery remains the responsibility of
/// the multiplex/ARQ layer above this datagram seam.
#[cfg(feature = "std")]
pub mod threaded {
    use super::Transport;
    use alloc::sync::Arc;
    use alloc::vec::Vec;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::mpsc::{self, Receiver, SyncSender, TrySendError};
    use std::sync::Mutex;

    /// A bounded, non-blocking endpoint for cross-thread replication tests or
    /// an in-process network worker.
    pub struct ThreadTransport {
        outbound: SyncSender<Vec<u8>>,
        inbound: Mutex<Receiver<Vec<u8>>>,
        dropped: Arc<AtomicU64>,
        capacity: usize,
    }

    impl ThreadTransport {
        /// Create a connected pair with `capacity` datagrams per direction.
        ///
        /// # Panics
        /// Panics when `capacity` is zero because an unbuffered channel would
        /// make the non-blocking `Transport::send` contract unusable.
        #[must_use]
        pub fn pair(capacity: usize) -> (Self, Self) {
            assert!(capacity > 0, "thread transport capacity must be positive");
            let (a_tx, a_rx) = mpsc::sync_channel(capacity);
            let (b_tx, b_rx) = mpsc::sync_channel(capacity);
            (
                Self {
                    outbound: a_tx,
                    inbound: Mutex::new(b_rx),
                    dropped: Arc::new(AtomicU64::new(0)),
                    capacity,
                },
                Self {
                    outbound: b_tx,
                    inbound: Mutex::new(a_rx),
                    dropped: Arc::new(AtomicU64::new(0)),
                    capacity,
                },
            )
        }

        /// Number of datagrams that could not be queued by this endpoint.
        #[must_use]
        pub fn dropped(&self) -> u64 {
            self.dropped.load(Ordering::Relaxed)
        }

        /// Configured datagram capacity per direction.
        #[must_use]
        pub const fn capacity(&self) -> usize {
            self.capacity
        }

        #[cfg(test)]
        pub(crate) fn pair_secure(
            capacity: usize,
        ) -> (
            crate::security::SecureTransport<Self, crate::security::test_support::Provider>,
            crate::security::SecureTransport<Self, crate::security::test_support::Provider>,
        ) {
            let (client, server) = Self::pair(capacity);
            (
                crate::security::SecureTransport::new_client(
                    client,
                    crate::security::test_support::Provider::new(0x31, 1),
                )
                .expect("test provider creates a non-zero client nonce"),
                crate::security::SecureTransport::new_server(
                    server,
                    crate::security::test_support::Provider::new(0x31, 9),
                ),
            )
        }
    }

    impl Transport for ThreadTransport {
        fn send(&mut self, bytes: &[u8]) {
            match self.outbound.try_send(bytes.to_vec()) {
                Ok(()) => {}
                Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => {
                    self.dropped.fetch_add(1, Ordering::Relaxed);
                }
            }
        }

        fn recv(&mut self) -> Option<Vec<u8>> {
            let receiver = self
                .inbound
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            receiver.try_recv().ok()
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::thread;

        #[test]
        fn endpoint_can_move_between_threads_without_blocking() {
            let (mut worker, mut app) = ThreadTransport::pair(2);
            let handle = thread::spawn(move || {
                worker.send(b"from worker");
                worker
            });
            worker = handle.join().expect("worker thread completes");
            assert_eq!(app.recv(), Some(b"from worker".to_vec()));
            assert_eq!(worker.dropped(), 0);
        }

        #[test]
        fn bounded_send_reports_backpressure_without_blocking() {
            let (mut a, mut b) = ThreadTransport::pair(1);
            a.send(b"first");
            a.send(b"second");
            assert_eq!(a.dropped(), 1);
            assert_eq!(b.recv(), Some(b"first".to_vec()));
            assert_eq!(b.recv(), None);
        }

        #[test]
        fn secure_session_crosses_the_thread_transport_boundary() {
            let (mut client, mut server) = ThreadTransport::pair_secure(8);
            client.send(b"hello");

            assert_eq!(server.recv(), None, "server consumes the client hello");
            assert_eq!(client.recv(), None, "client consumes the server accept");
            assert_eq!(server.recv(), Some(b"hello".to_vec()));
            assert_eq!(
                client.security_state(),
                super::super::TransportSecurity::Established
            );
            assert_eq!(
                server.security_state(),
                super::super::TransportSecurity::Established
            );
        }
    }
}

/// A deterministic, bounded transport harness that exercises the same
/// [`Transport`] seam used by UDP and secure sessions.
#[cfg(feature = "std")]
pub mod simulated {
    use super::Transport;
    use alloc::collections::VecDeque;
    use alloc::sync::Arc;
    use alloc::vec::Vec;
    use std::sync::Mutex;

    /// Faults applied to every datagram admitted by a simulated link.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct FaultConfig {
        /// Minimum logical ticks before a datagram becomes due.
        pub latency: u32,
        /// Additional deterministic delay in `0..=jitter` ticks.
        pub jitter: u32,
        /// Drop every `n`th attempted send; `None` disables loss.
        pub drop_every: Option<u64>,
        /// Duplicate every `n`th accepted send; `None` disables duplication.
        pub duplicate_every: Option<u64>,
        /// Reverse each due batch when true, making reordering explicit.
        pub reorder_due_batches: bool,
        /// Maximum queued datagrams per endpoint.
        pub queue_limit: usize,
    }

    impl Default for FaultConfig {
        fn default() -> Self {
            Self {
                latency: 1,
                jitter: 0,
                drop_every: None,
                duplicate_every: None,
                reorder_due_batches: false,
                queue_limit: 256,
            }
        }
    }

    impl FaultConfig {
        fn normalized(self) -> Self {
            Self {
                latency: self.latency,
                jitter: self.jitter,
                drop_every: self.drop_every.filter(|value| *value != 0),
                duplicate_every: self.duplicate_every.filter(|value| *value != 0),
                reorder_due_batches: self.reorder_due_batches,
                queue_limit: self.queue_limit.max(1),
            }
        }
    }

    #[derive(Default)]
    struct State {
        tick: u64,
        sends: u64,
        dropped: u64,
        queue_drops: u64,
        in_flight: Vec<Packet>,
        queues: [VecDeque<Vec<u8>>; 2],
        rng: SplitMix64,
    }

    struct Packet {
        deliver_at: u64,
        order: u64,
        endpoint: usize,
        bytes: Vec<u8>,
    }

    /// Shared deterministic network clock and fault injector.
    #[derive(Clone)]
    pub struct SimulatedNetwork {
        config: FaultConfig,
        state: Arc<Mutex<State>>,
    }

    impl SimulatedNetwork {
        /// Create two connected endpoints sharing one deterministic link.
        #[must_use]
        pub fn pair(config: FaultConfig) -> (Self, SimulatedTransport, SimulatedTransport) {
            let network = Self {
                config: config.normalized(),
                state: Arc::new(Mutex::new(State {
                    rng: SplitMix64::new(0x5EED_1234_ABCD_0001),
                    ..State::default()
                })),
            };
            let a = SimulatedTransport {
                network: network.clone(),
                endpoint: 0,
            };
            let b = SimulatedTransport {
                network: network.clone(),
                endpoint: 1,
            };
            (network, a, b)
        }

        /// Advance one logical tick and deliver all due datagrams.
        pub fn advance(&self) {
            let mut state = lock(&self.state);
            state.tick = state.tick.saturating_add(1);
            let tick = state.tick;
            let mut due = [Vec::new(), Vec::new()];
            let mut kept = Vec::with_capacity(state.in_flight.len());
            for packet in state.in_flight.drain(..) {
                if packet.deliver_at <= tick {
                    due[packet.endpoint].push(packet);
                } else {
                    kept.push(packet);
                }
            }
            state.in_flight = kept;
            for (endpoint, packets) in due.iter_mut().enumerate() {
                packets.sort_by_key(|packet| (packet.deliver_at, packet.order));
                if self.config.reorder_due_batches {
                    packets.reverse();
                }
                for packet in packets.drain(..) {
                    if state.queues[endpoint].len() >= self.config.queue_limit {
                        state.queue_drops = state.queue_drops.saturating_add(1);
                    } else {
                        state.queues[endpoint].push_back(packet.bytes);
                    }
                }
            }
        }

        /// Drop all queued/in-flight packets and reset the logical clock.
        ///
        /// Callers create fresh `SecureTransport` wrappers after this method;
        /// this models a reconnect without allowing old authenticated packets
        /// to cross into a new session.
        pub fn reset(&self) {
            let mut state = lock(&self.state);
            state.tick = 0;
            state.in_flight.clear();
            state.queues[0].clear();
            state.queues[1].clear();
        }

        /// Current logical tick.
        #[must_use]
        pub fn tick(&self) -> u64 {
            lock(&self.state).tick
        }

        /// Datagrams rejected by deterministic loss or bounded queues.
        #[must_use]
        pub fn dropped(&self) -> u64 {
            let state = lock(&self.state);
            state.dropped.saturating_add(state.queue_drops)
        }

        /// Datagrams rejected by the configured deterministic loss rule.
        #[must_use]
        pub fn fault_drops(&self) -> u64 {
            lock(&self.state).dropped
        }

        /// Datagrams rejected because an endpoint receive queue was full.
        #[must_use]
        pub fn queue_drops(&self) -> u64 {
            lock(&self.state).queue_drops
        }

        /// Number of datagrams still waiting for delivery or receive polling.
        #[must_use]
        pub fn pending(&self) -> usize {
            let state = lock(&self.state);
            state.in_flight.len() + state.queues[0].len() + state.queues[1].len()
        }
    }

    /// One endpoint of a [`SimulatedNetwork`]. Cloning an endpoint preserves
    /// its endpoint identity and is useful when reconnecting a fresh secure
    /// wrapper after [`SimulatedNetwork::reset`].
    #[derive(Clone)]
    pub struct SimulatedTransport {
        network: SimulatedNetwork,
        endpoint: usize,
    }

    impl SimulatedTransport {
        /// Endpoint index (`0` or `1`).
        #[must_use]
        pub const fn endpoint(&self) -> usize {
            self.endpoint
        }
    }

    impl Transport for SimulatedTransport {
        fn send(&mut self, bytes: &[u8]) {
            let mut state = lock(&self.network.state);
            state.sends = state.sends.saturating_add(1);
            if self
                .network
                .config
                .drop_every
                .is_some_and(|n| state.sends.is_multiple_of(n))
            {
                state.dropped = state.dropped.saturating_add(1);
                return;
            }
            let extra = if self.network.config.jitter == 0 {
                0
            } else {
                state
                    .rng
                    .next_below(u64::from(self.network.config.jitter) + 1)
            };
            let endpoint = 1 - self.endpoint;
            let deliver_at = state
                .tick
                .saturating_add(u64::from(self.network.config.latency))
                .saturating_add(extra);
            let order = state.sends;
            state.in_flight.push(Packet {
                deliver_at,
                order,
                endpoint,
                bytes: bytes.to_vec(),
            });
            if self
                .network
                .config
                .duplicate_every
                .is_some_and(|n| state.sends.is_multiple_of(n))
            {
                state.in_flight.push(Packet {
                    deliver_at,
                    order,
                    endpoint,
                    bytes: bytes.to_vec(),
                });
            }
        }

        fn recv(&mut self) -> Option<Vec<u8>> {
            lock(&self.network.state).queues[self.endpoint].pop_front()
        }
    }

    fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
        mutex
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    #[derive(Default)]
    struct SplitMix64(u64);

    impl SplitMix64 {
        fn new(seed: u64) -> Self {
            Self(seed)
        }

        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        fn next_below(&mut self, bound: u64) -> u64 {
            if bound == 0 {
                0
            } else {
                self.next_u64() % bound
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use alloc::vec;

        #[test]
        fn deterministic_faults_drop_and_reverse_without_unbounded_growth() {
            let config = FaultConfig {
                latency: 1,
                jitter: 2,
                drop_every: Some(5),
                duplicate_every: None,
                reorder_due_batches: true,
                queue_limit: 8,
            };
            let (network, mut a, mut b) = SimulatedNetwork::pair(config);
            for value in 0..20_u8 {
                a.send(&[value]);
            }
            for _ in 0..16 {
                network.advance();
            }
            let mut received = Vec::new();
            while let Some(bytes) = b.recv() {
                received.push(bytes[0]);
            }
            assert_eq!(network.fault_drops(), 4);
            assert_eq!(network.queue_drops(), 8);
            assert_eq!(network.dropped(), 12);
            assert!(received.len() <= 8);
            assert!(received.windows(2).any(|window| window[0] > window[1]));
        }

        #[test]
        fn reset_discards_old_packets_for_reconnect() {
            let (network, mut a, mut b) = SimulatedNetwork::pair(FaultConfig::default());
            a.send(b"old");
            network.reset();
            network.advance();
            assert_eq!(b.recv(), None);
        }

        #[test]
        fn duplicate_fault_is_deterministic_and_bounded() {
            let config = FaultConfig {
                latency: 1,
                jitter: 0,
                drop_every: None,
                duplicate_every: Some(2),
                reorder_due_batches: false,
                queue_limit: 16,
            };
            let (network, mut a, mut b) = SimulatedNetwork::pair(config);
            for value in 0..4_u8 {
                a.send(&[value]);
            }
            network.advance();
            let mut received = Vec::new();
            while let Some(bytes) = b.recv() {
                received.push(bytes[0]);
            }
            assert_eq!(received, vec![0, 1, 1, 2, 3, 3]);
            assert_eq!(network.pending(), 0);
        }
    }

    #[cfg(feature = "crypto")]
    #[cfg(test)]
    mod secure_channel_tests {
        use super::*;
        use crate::multiplex::{ChannelMux, MuxConfig, NetChannel};
        use crate::security::{PskCredential, PskKeyring, RingPskKeyringProvider, SecureTransport};
        use crate::transport::TransportSecurity;
        use alloc::vec;

        fn key(id: u32, byte: u8) -> PskCredential {
            PskCredential::new(id, [byte; 32])
        }

        type SecureSimTransport = SecureTransport<SimulatedTransport, RingPskKeyringProvider>;
        type SecureMux = ChannelMux<SecureSimTransport>;

        fn connect(
            network: &SimulatedNetwork,
            client_link: SimulatedTransport,
            server_link: SimulatedTransport,
            client_keyring: PskKeyring,
            server_keyring: PskKeyring,
        ) -> (SecureMux, SecureMux) {
            let client = SecureTransport::new_client(
                client_link,
                RingPskKeyringProvider::new(client_keyring),
            )
            .expect("client hello is generated");
            let server = SecureTransport::new_server(
                server_link,
                RingPskKeyringProvider::new(server_keyring),
            );
            let config = MuxConfig {
                min_rto: 3,
                reliable_window: 16,
                reliable_queue: 64,
                path_mtu: 1_200,
                ..MuxConfig::default()
            };
            let mut client = ChannelMux::new(client, config);
            let mut server = ChannelMux::new(server, config);
            for tick in 1..=64 {
                network.advance();
                client.tick_at(tick);
                server.tick_at(tick);
                if client.security_state() == TransportSecurity::Established
                    && server.security_state() == TransportSecurity::Established
                {
                    return (client, server);
                }
            }
            panic!(
                "secure simulated handshake did not complete: {:?} / {:?}",
                client.security_state(),
                server.security_state()
            );
        }

        fn drive(network: &SimulatedNetwork, client: &mut SecureMux, server: &mut SecureMux) {
            for tick in 65..=2_000 {
                network.advance();
                client.tick_at(tick);
                server.tick_at(tick);
                if server.queued(NetChannel::EventsReliable) == 0
                    && client.queued(NetChannel::EventsReliable) == 0
                    && network.pending() == 0
                {
                    return;
                }
            }
            panic!(
                "secure simulated transport did not drain: client={:?} server={:?} pending={}",
                client.telemetry(),
                server.telemetry(),
                network.pending()
            );
        }

        #[test]
        fn secure_mux_closes_loss_reorder_duplicate_and_reconnect_loop() {
            let config = FaultConfig {
                latency: 1,
                jitter: 3,
                drop_every: Some(7),
                duplicate_every: Some(5),
                reorder_due_batches: true,
                queue_limit: 128,
            };
            let (network, client_link, server_link) = SimulatedNetwork::pair(config);
            let old = key(101, 0x41);
            let (mut client, mut server) = connect(
                &network,
                client_link.clone(),
                server_link.clone(),
                PskKeyring::new(old),
                PskKeyring::new(old),
            );

            assert_eq!(client.datagram_mut().telemetry().credential_id, Some(101));
            assert_eq!(server.datagram_mut().telemetry().credential_id, Some(101));
            assert!(client.datagram_mut().telemetry().handshakes_completed >= 1);
            assert!(server.datagram_mut().telemetry().handshakes_completed >= 1);

            for sequence in 0..24_u8 {
                client
                    .send(NetChannel::EventsReliable, &[sequence])
                    .expect("bounded event queue accepts the test slice");
            }
            drive(&network, &mut client, &mut server);

            let mut received = Vec::new();
            while let Some(payload) = server.recv(NetChannel::EventsReliable) {
                received.push(payload);
            }
            assert_eq!(
                received,
                (0..24_u8)
                    .map(|sequence| vec![sequence])
                    .collect::<Vec<_>>(),
                "reliable application delivery is ordered and exactly once"
            );
            assert!(
                server.datagram_mut().telemetry().replayed_datagrams > 0,
                "duplicate injection reaches the authenticated replay window"
            );

            network.reset();
            drop(client);
            drop(server);
            let (mut client, mut server) = connect(
                &network,
                client_link,
                server_link,
                PskKeyring::new(old),
                PskKeyring::new(old),
            );
            client
                .send(NetChannel::EventsReliable, b"after-reconnect")
                .unwrap();
            drive(&network, &mut client, &mut server);
            assert_eq!(
                server.recv(NetChannel::EventsReliable),
                Some(b"after-reconnect".to_vec())
            );
            assert_eq!(client.datagram_mut().telemetry().credential_id, Some(101));
            assert!(client.datagram_mut().telemetry().handshakes_completed >= 1);
        }
    }
}

/// A deterministic two-endpoint link with latency, jitter and loss.
///
/// Delivery is driven by [`poll`](Self::poll): each call advances one logical
/// tick and returns the messages that have arrived for the endpoint being
/// polled, oldest first. Jitter comes from a seeded `SplitMix64`, so a given
/// configuration always drops and reorders the same messages.
#[derive(Clone, Debug)]
pub struct LoopbackLink {
    tick: u64,
    latency: u32,
    jitter: u32,
    drop_every: Option<u64>,
    sent: u64,
    dropped: u64,
    in_flight: Vec<InFlight>,
    rng: SplitMix64,
}

#[derive(Clone, Debug)]
struct InFlight {
    deliver_at: u64,
    /// Monotone send order, the tie-break so deliveries are deterministic.
    seq: u64,
    to_a: bool,
    bytes: Vec<u8>,
}

impl LoopbackLink {
    /// A link with a base latency of `latency` ticks and up to `jitter` extra
    /// ticks of per-message variation (both in logical ticks).
    ///
    /// The delay parameters are `u32` ticks deliberately: it keeps
    /// `tick + latency + jitter` far from `u64` saturation, so the drain idiom
    /// `while link.in_flight() > 0 { poll(..) }` always terminates (a `u64`
    /// latency near `u64::MAX` would pin messages at an unreachable tick).
    #[must_use]
    pub fn new(latency: u32, jitter: u32) -> Self {
        Self {
            tick: 0,
            latency,
            jitter,
            drop_every: None,
            sent: 0,
            dropped: 0,
            in_flight: Vec::new(),
            rng: SplitMix64::new(0x5EED_1234_ABCD_0001),
        }
    }

    /// Drop every `n`th send, deterministically (`n == 0` disables loss). Builder
    /// style.
    #[must_use]
    pub fn with_drop_every(mut self, n: u64) -> Self {
        self.drop_every = (n > 0).then_some(n);
        self
    }

    /// The current logical tick.
    #[must_use]
    pub fn tick(&self) -> u64 {
        self.tick
    }

    /// How many messages are still in flight.
    #[must_use]
    pub fn in_flight(&self) -> usize {
        self.in_flight.len()
    }

    /// How many sends were dropped by the fault model.
    #[must_use]
    pub fn dropped(&self) -> u64 {
        self.dropped
    }

    /// Send `bytes` from endpoint A (`from_a == true`) or B, returning whether it
    /// was accepted (a dropped message returns `false`).
    pub fn send(&mut self, from_a: bool, bytes: Vec<u8>) -> bool {
        self.sent = self.sent.saturating_add(1);
        if let Some(n) = self.drop_every
            && self.sent.is_multiple_of(n)
        {
            self.dropped += 1;
            return false;
        }
        let extra = if self.jitter > 0 {
            self.rng.next_below(u64::from(self.jitter) + 1)
        } else {
            0
        };
        self.in_flight.push(InFlight {
            deliver_at: self
                .tick
                .saturating_add(u64::from(self.latency))
                .saturating_add(extra),
            seq: self.sent,
            to_a: !from_a,
            bytes,
        });
        true
    }

    /// Advance one tick and return the messages that have arrived at endpoint A
    /// (`at_a == true`) or B, oldest first.
    pub fn poll(&mut self, at_a: bool) -> Vec<Vec<u8>> {
        self.tick = self.tick.saturating_add(1);
        // Select the due messages, then remove them. `retain` keeps the remaining
        // in-flight messages in send order.
        let mut due: Vec<InFlight> = Vec::new();
        let mut kept: Vec<InFlight> = Vec::with_capacity(self.in_flight.len());
        for message in self.in_flight.drain(..) {
            if message.to_a == at_a && message.deliver_at <= self.tick {
                due.push(message);
            } else {
                kept.push(message);
            }
        }
        self.in_flight = kept;
        due.sort_by_key(|message| (message.deliver_at, message.seq));
        due.into_iter().map(|message| message.bytes).collect()
    }
}

/// A tiny `SplitMix64` PRNG, used only to make fault injection reproducible.
#[derive(Clone, Debug)]
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A value in `0..bound` (`0` when `bound` is zero).
    fn next_below(&mut self, bound: u64) -> u64 {
        if bound == 0 {
            0
        } else {
            self.next_u64() % bound
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sequence::{Delivery, DeliveryWindow};
    use alloc::vec;

    #[test]
    fn a_clean_link_delivers_every_message() {
        let mut link = LoopbackLink::new(2, 0);
        assert!(link.send(true, vec![1]));
        assert!(link.send(true, vec![2]));
        assert_eq!(link.in_flight(), 2);

        // Nothing arrives before the latency elapses.
        assert!(link.poll(false).is_empty());
        assert_eq!(link.poll(false), vec![vec![1], vec![2]]);
        assert_eq!(link.in_flight(), 0);
    }

    #[test]
    fn the_drop_model_is_deterministic() {
        let mut link = LoopbackLink::new(1, 0).with_drop_every(5);
        let accepted: Vec<bool> = (0..20).map(|i| link.send(true, vec![i as u8])).collect();
        assert_eq!(link.dropped(), 4, "every 5th of 20 sends is dropped");
        assert!(!accepted[4] && !accepted[9]);

        // A second, identically configured link drops the same sends.
        let mut other = LoopbackLink::new(1, 0).with_drop_every(5);
        let other_accepted: Vec<bool> = (0..20).map(|i| other.send(true, vec![i as u8])).collect();
        assert_eq!(accepted, other_accepted);
    }

    /// The delay parameters are `u32`, so `latency + jitter` stays far below the
    /// `u64` tick ceiling — the drain idiom always terminates and jitter cannot
    /// overflow (regression: `jitter + 1` overflowed at `u64::MAX`, and a near-
    /// `u64::MAX` latency pinned a message at an unreachable tick).
    #[test]
    fn extreme_fault_parameters_do_not_overflow() {
        let mut link = LoopbackLink::new(u32::MAX, u32::MAX);
        assert!(link.send(true, vec![7]));
        assert_eq!(
            link.in_flight(),
            1,
            "building the delivery tick did not overflow"
        );
    }

    /// The delivery window, fed over the lossy/jittery link, restores in-order
    /// delivery up to the first dropped sequence — exactly its contract.
    #[test]
    fn the_delivery_window_restores_order_over_a_lossy_link() {
        let mut link = LoopbackLink::new(1, 3).with_drop_every(5);
        let mut window: DeliveryWindow<u64> = DeliveryWindow::new(16);

        for seq in 0..20u64 {
            link.send(true, seq.to_le_bytes().to_vec());
        }

        let mut delivered = 0u64;
        while link.in_flight() > 0 {
            for bytes in link.poll(false) {
                let seq = u64::from_le_bytes(bytes.try_into().expect("eight bytes"));
                if let Delivery::Advanced(run) = window.insert(seq, seq) {
                    for (_, value) in run {
                        assert_eq!(value, delivered, "delivered strictly in order");
                        delivered += 1;
                    }
                }
            }
        }
        // Sequence 4 was dropped (the 5th send), so frames 0..4 (four of them)
        // are delivered and everything after waits behind the gap.
        assert_eq!(delivered, 4);
        assert_eq!(window.next(), 4);
        assert!(window.buffered() > 0, "later frames wait behind the gap");
    }
}
