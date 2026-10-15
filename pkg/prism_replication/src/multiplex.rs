//! Multiplexed game-network channels over one datagram transport (design §6).
//!
//! [`ChannelMux`] keeps independent reliable ordering windows for authoritative
//! events, RPC/control traffic, and bulk transfer. Snapshot and input traffic is
//! deliberately unreliable: it carries a monotonically increasing sequence and
//! never waits behind a lost reliable packet. A shared byte congestion window
//! bounds reliable bytes in flight, while a per-tick datagram budget sheds
//! unreliable traffic first under load.

use alloc::collections::{BTreeMap, VecDeque};
use alloc::vec::Vec;

use crate::arq::{ReliableReceiver, ReliableSender};
use crate::transport::{Transport, TransportSecurity};

const WIRE_VERSION: u8 = 1;
const KIND_DATA: u8 = 0;
const KIND_ACK: u8 = 1;
const HEADER_LEN: usize = 12;

/// Conservative Internet-safe datagram size. Staying at or below 1200 bytes
/// avoids IP fragmentation on IPv6 and matches the QUIC minimum datagram size.
pub const DEFAULT_PATH_MTU: usize = 1_200;

/// Minimum rollback capacity that can retain the acknowledged baseline.
pub const MIN_ROLLBACK_WINDOW: u32 = 1;

/// Current application protocol spoken by the app plugins.
///
/// This is intentionally separate from the private multiplex packet format:
/// changing channel framing does not silently claim compatibility with a
/// different replication schema or RPC contract.
#[cfg(not(feature = "determinism"))]
pub const CURRENT_PROTOCOL_VERSION: u16 = 1;
/// Audit-enabled application contract. Hash envelopes are not silently sent to
/// version-one peers. Explicit custom versions must preserve this distinction.
#[cfg(feature = "determinism")]
pub const CURRENT_PROTOCOL_VERSION: u16 = 2;

/// One endpoint's handshake offer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProtocolOffer {
    /// Replication/RPC contract version.
    pub version: u16,
    /// Maximum number of simulation revisions this endpoint can retain for
    /// reconciliation and rollback.
    pub rollback_window: u32,
}

impl ProtocolOffer {
    /// Build an offer for the current protocol and `rollback_window`.
    #[must_use]
    pub const fn current(rollback_window: u32) -> Self {
        Self {
            version: CURRENT_PROTOCOL_VERSION,
            rollback_window,
        }
    }

    /// Validate the offer before it is put on the wire or negotiated.
    ///
    /// The rollback window includes the acknowledged baseline, so zero cannot
    /// represent a usable replication session.
    pub fn validate(self) -> Result<(), ProtocolError> {
        if self.rollback_window < MIN_ROLLBACK_WINDOW {
            return Err(ProtocolError::InvalidRollbackWindow {
                offered: self.rollback_window,
            });
        }
        Ok(())
    }
}

/// The contract selected by a successful handshake.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NegotiatedProtocol {
    /// Exact protocol version shared by both peers.
    pub version: u16,
    /// Smaller of the two rollback capacities, and therefore the maximum either
    /// endpoint may assume is retained by its peer.
    pub rollback_window: u32,
}

impl NegotiatedProtocol {
    /// Maximum number of unconfirmed predictions retained alongside the
    /// acknowledged baseline.
    #[must_use]
    pub const fn max_unconfirmed_predictions(self) -> u32 {
        self.rollback_window.saturating_sub(MIN_ROLLBACK_WINDOW)
    }
}

/// A handshake or plugin-envelope violation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProtocolError {
    /// Peers implement different replication/RPC contracts.
    VersionMismatch {
        /// Version implemented by this endpoint.
        local: u16,
        /// Version offered by the peer.
        remote: u16,
    },
    /// A zero rollback window cannot retain even the current baseline.
    InvalidRollbackWindow {
        /// Invalid value received from the peer.
        offered: u32,
    },
    /// A control or application envelope was truncated or used an unknown tag.
    MalformedMessage {
        /// Logical channel carrying the invalid message.
        channel: NetChannel,
    },
    /// The peer sent application traffic before the handshake completed.
    TrafficBeforeHandshake {
        /// Logical channel carrying the premature message.
        channel: NetChannel,
    },
    /// The peer sent a handshake message that is invalid for this endpoint's
    /// role or current connection state.
    UnexpectedHandshake,
    /// The authenticated transport failed after construction.
    SecurityFailure,
}

/// Select an exact protocol version and a mutually supportable rollback window.
///
/// Protocol versions are not guessed or downgraded: incompatible peers fail
/// explicitly. Rollback capacity is safely reduced to the smaller offer.
pub fn negotiate_protocol(
    local: ProtocolOffer,
    remote: ProtocolOffer,
) -> Result<NegotiatedProtocol, ProtocolError> {
    local.validate()?;
    remote.validate()?;
    if local.version != remote.version {
        return Err(ProtocolError::VersionMismatch {
            local: local.version,
            remote: remote.version,
        });
    }
    Ok(NegotiatedProtocol {
        version: local.version,
        rollback_window: local.rollback_window.min(remote.rollback_window),
    })
}

/// Logical channels sharing one physical datagram transport.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum NetChannel {
    /// Authoritative persistent events: reliable and ordered.
    EventsReliable = 0,
    /// High-frequency state snapshots: unreliable, newest state wins.
    StateUnreliable = 1,
    /// Client inputs: unreliable and latency-sensitive.
    InputUnreliable = 2,
    /// RPC and control traffic: reliable and ordered.
    RpcReliable = 3,
    /// Baselines and other large transfers: reliable, ordered, low priority.
    BulkReliable = 4,
}

impl NetChannel {
    /// Whether this channel retransmits and delivers in order.
    #[must_use]
    pub const fn is_reliable(self) -> bool {
        matches!(
            self,
            Self::EventsReliable | Self::RpcReliable | Self::BulkReliable
        )
    }

    fn from_wire(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::EventsReliable),
            1 => Some(Self::StateUnreliable),
            2 => Some(Self::InputUnreliable),
            3 => Some(Self::RpcReliable),
            4 => Some(Self::BulkReliable),
            _ => None,
        }
    }
}

/// Runtime limits for a multiplexed connection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MuxConfig {
    /// Retransmit timeout floor in ticks.
    pub min_rto: u32,
    /// Per-reliable-channel message window.
    pub reliable_window: usize,
    /// Maximum queued messages per reliable channel before explicit backpressure.
    pub reliable_queue: usize,
    /// Maximum queued client-input datagrams; oldest inputs are discarded first.
    pub input_queue: usize,
    /// Path MTU including this protocol's header.
    pub path_mtu: usize,
    /// Initial shared reliable congestion window in bytes.
    pub initial_cwnd_bytes: usize,
    /// Upper bound for the congestion window.
    pub max_cwnd_bytes: usize,
    /// Unreliable byte budget replenished every tick.
    pub unreliable_budget_per_tick: usize,
}

impl Default for MuxConfig {
    fn default() -> Self {
        Self {
            min_rto: 3,
            reliable_window: 64,
            reliable_queue: 256,
            input_queue: 8,
            path_mtu: DEFAULT_PATH_MTU,
            initial_cwnd_bytes: DEFAULT_PATH_MTU * 10,
            max_cwnd_bytes: DEFAULT_PATH_MTU * 256,
            unreliable_budget_per_tick: DEFAULT_PATH_MTU * 4,
        }
    }
}

impl MuxConfig {
    fn normalized(self) -> Self {
        let path_mtu = self.path_mtu.max(HEADER_LEN + 1);
        let min_cwnd = path_mtu.saturating_mul(2);
        let max_cwnd_bytes = self.max_cwnd_bytes.max(min_cwnd);
        Self {
            min_rto: self.min_rto.max(1),
            reliable_window: self.reliable_window.max(1),
            reliable_queue: self.reliable_queue.max(1),
            input_queue: self.input_queue.max(1),
            path_mtu,
            initial_cwnd_bytes: self.initial_cwnd_bytes.clamp(min_cwnd, max_cwnd_bytes),
            max_cwnd_bytes,
            unreliable_budget_per_tick: self.unreliable_budget_per_tick.max(path_mtu),
        }
    }
}

/// Why a message was not accepted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MuxSendError {
    /// An app plugin cannot accept this send in its current handshake state.
    ConnectionNotEstablished,
    /// The payload would exceed the configured path MTU.
    TooLarge {
        /// Payload bytes supplied by the caller.
        len: usize,
        /// Largest payload accepted without IP fragmentation.
        max: usize,
    },
    /// The bounded reliable queue is full. The caller must retry later.
    Backpressured(NetChannel),
}

/// Result of accepting unreliable traffic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueueOutcome {
    /// The message was queued without evicting another one.
    Queued,
    /// An older expendable message was replaced to keep latency bounded.
    ReplacedOlder,
}

/// Per-connection transport counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MuxStats {
    /// Physical datagrams emitted, including ACKs and retransmissions.
    pub datagrams_sent: u64,
    /// Physical datagrams received with a valid multiplex header.
    pub datagrams_received: u64,
    /// Reliable packets sent more than once.
    pub retransmissions: u64,
    /// Unreliable messages replaced or shed under budget pressure.
    pub unreliable_dropped: u64,
    /// Reliable sends refused because a bounded queue was full.
    pub backpressure_events: u64,
    /// Malformed or unsupported datagrams ignored.
    pub malformed_datagrams: u64,
}

#[derive(Clone, Debug)]
struct ReliableLane {
    sender: ReliableSender,
    receiver: ReliableReceiver,
    pending: VecDeque<Vec<u8>>,
    inbound: VecDeque<Vec<u8>>,
}

impl ReliableLane {
    fn new(config: MuxConfig) -> Self {
        Self {
            sender: ReliableSender::new(config.min_rto, config.reliable_window),
            receiver: ReliableReceiver::new(config.reliable_window),
            pending: VecDeque::new(),
            inbound: VecDeque::new(),
        }
    }
}

#[derive(Clone, Debug, Default)]
struct UnreliableLane {
    next_send: u64,
    newest_received: Option<u64>,
    pending: VecDeque<Vec<u8>>,
    inbound: VecDeque<Vec<u8>>,
}

#[derive(Clone, Copy, Debug)]
struct Flight {
    bytes: usize,
    first_sent: u64,
    transmissions: u32,
}

#[derive(Clone, Debug)]
struct CongestionController {
    mtu: usize,
    cwnd: usize,
    max_cwnd: usize,
    bytes_in_flight: usize,
    min_rtt: Option<u64>,
    bandwidth_bytes_per_tick: usize,
}

impl CongestionController {
    fn new(config: MuxConfig) -> Self {
        Self {
            mtu: config.path_mtu,
            cwnd: config.initial_cwnd_bytes,
            max_cwnd: config.max_cwnd_bytes,
            bytes_in_flight: 0,
            min_rtt: None,
            bandwidth_bytes_per_tick: config.path_mtu,
        }
    }

    fn can_admit(&self, bytes: usize) -> bool {
        self.bytes_in_flight.saturating_add(bytes) <= self.cwnd
    }

    fn admitted(&mut self, bytes: usize) {
        self.bytes_in_flight = self.bytes_in_flight.saturating_add(bytes);
    }

    fn acknowledged(&mut self, bytes: usize, rtt: u64) {
        self.bytes_in_flight = self.bytes_in_flight.saturating_sub(bytes);
        let rtt = rtt.max(1);
        self.min_rtt = Some(self.min_rtt.map_or(rtt, |old| old.min(rtt)));
        let sample_rate = bytes.saturating_add(rtt as usize - 1) / rtt as usize;
        self.bandwidth_bytes_per_tick = self
            .bandwidth_bytes_per_tick
            .saturating_mul(7)
            .saturating_add(sample_rate)
            / 8;

        // BBR-shaped BDP target: estimated delivery rate × minimum RTT with a
        // 2x gain for ACK compression and short bursts. This is byte based, so
        // large reliable messages consume proportionally more of the window.
        let bdp = self
            .bandwidth_bytes_per_tick
            .saturating_mul(self.min_rtt.unwrap_or(1) as usize)
            .saturating_mul(2);
        let target = bdp.clamp(self.mtu.saturating_mul(2), self.max_cwnd);
        self.cwnd = self.cwnd.saturating_mul(7).saturating_add(target) / 8;
        self.cwnd = self.cwnd.clamp(self.mtu.saturating_mul(2), self.max_cwnd);
    }
}

/// Five logical game-network channels over one physical datagram transport.
pub struct ChannelMux<D: Transport> {
    datagram: D,
    config: MuxConfig,
    events: ReliableLane,
    state: UnreliableLane,
    input: UnreliableLane,
    rpc: ReliableLane,
    bulk: ReliableLane,
    congestion: CongestionController,
    flights: BTreeMap<(NetChannel, u64), Flight>,
    now: u64,
    stats: MuxStats,
}

impl<D: Transport> ChannelMux<D> {
    /// Create a multiplexed connection over `datagram`.
    #[must_use]
    pub fn new(datagram: D, config: MuxConfig) -> Self {
        let config = config.normalized();
        Self {
            datagram,
            config,
            events: ReliableLane::new(config),
            state: UnreliableLane::default(),
            input: UnreliableLane::default(),
            rpc: ReliableLane::new(config),
            bulk: ReliableLane::new(config),
            congestion: CongestionController::new(config),
            flights: BTreeMap::new(),
            now: 0,
            stats: MuxStats::default(),
        }
    }

    /// Maximum payload that fits one datagram without IP fragmentation.
    #[must_use]
    pub fn max_payload(&self) -> usize {
        self.config
            .path_mtu
            .saturating_sub(HEADER_LEN.saturating_add(self.datagram.security_overhead()))
    }

    /// Queue a message. Reliable queues apply explicit bounded backpressure;
    /// state keeps only the newest unsent snapshot, and input keeps a short
    /// bounded tail while discarding the oldest input first.
    pub fn send(
        &mut self,
        channel: NetChannel,
        payload: &[u8],
    ) -> Result<QueueOutcome, MuxSendError> {
        if payload.len() > self.max_payload() {
            return Err(MuxSendError::TooLarge {
                len: payload.len(),
                max: self.max_payload(),
            });
        }
        if channel.is_reliable() {
            let queue_limit = self.config.reliable_queue;
            let lane = self.reliable_lane_mut(channel);
            if lane.pending.len() >= queue_limit {
                self.stats.backpressure_events = self.stats.backpressure_events.saturating_add(1);
                return Err(MuxSendError::Backpressured(channel));
            }
            lane.pending.push_back(payload.to_vec());
            return Ok(QueueOutcome::Queued);
        }

        match channel {
            NetChannel::StateUnreliable => {
                let replaced = self.state.pending.pop_back().is_some();
                self.state.pending.push_back(payload.to_vec());
                if replaced {
                    self.stats.unreliable_dropped = self.stats.unreliable_dropped.saturating_add(1);
                    Ok(QueueOutcome::ReplacedOlder)
                } else {
                    Ok(QueueOutcome::Queued)
                }
            }
            NetChannel::InputUnreliable => {
                let replaced = if self.input.pending.len() >= self.config.input_queue {
                    self.input.pending.pop_front();
                    self.stats.unreliable_dropped = self.stats.unreliable_dropped.saturating_add(1);
                    true
                } else {
                    false
                };
                self.input.pending.push_back(payload.to_vec());
                Ok(if replaced {
                    QueueOutcome::ReplacedOlder
                } else {
                    QueueOutcome::Queued
                })
            }
            _ => unreachable!("reliable channels returned above"),
        }
    }

    /// Check whether `payload_len` bytes can be queued now without mutating the
    /// connection. Reliable channels report bounded-queue backpressure;
    /// unreliable channels always accept by replacing expendable older data.
    pub fn check_send(&self, channel: NetChannel, payload_len: usize) -> Result<(), MuxSendError> {
        if payload_len > self.max_payload() {
            return Err(MuxSendError::TooLarge {
                len: payload_len,
                max: self.max_payload(),
            });
        }
        if channel.is_reliable()
            && self.reliable_lane(channel).pending.len() >= self.config.reliable_queue
        {
            return Err(MuxSendError::Backpressured(channel));
        }
        Ok(())
    }

    /// Advance one logical tick and exchange datagrams.
    pub fn tick(&mut self) {
        self.tick_at(self.now.saturating_add(1));
    }

    /// Advance to monotonic `now`, drain inbound packets, schedule queued traffic,
    /// and run retransmissions.
    pub fn tick_at(&mut self, now: u64) {
        self.now = self.now.max(now);
        self.drain_inbound();
        self.admit_reliable();
        self.flush_reliable();
        self.flush_unreliable();
    }

    /// Security state of the wrapped datagram transport.
    #[must_use]
    pub fn security_state(&self) -> TransportSecurity {
        self.datagram.security_state()
    }

    /// Take a security-provider failure reported by the wrapped transport.
    pub fn take_security_error(&mut self) -> Option<crate::security::SecurityError> {
        self.datagram.take_security_error()
    }

    /// Take one delivered message from `channel`.
    pub fn recv(&mut self, channel: NetChannel) -> Option<Vec<u8>> {
        match channel {
            NetChannel::EventsReliable => self.events.inbound.pop_front(),
            NetChannel::StateUnreliable => self.state.inbound.pop_front(),
            NetChannel::InputUnreliable => self.input.inbound.pop_front(),
            NetChannel::RpcReliable => self.rpc.inbound.pop_front(),
            NetChannel::BulkReliable => self.bulk.inbound.pop_front(),
        }
    }

    /// Current reliable bytes charged against the shared congestion window.
    #[must_use]
    pub fn bytes_in_flight(&self) -> usize {
        self.congestion.bytes_in_flight
    }

    /// Current byte congestion window.
    #[must_use]
    pub fn congestion_window(&self) -> usize {
        self.congestion.cwnd
    }

    /// Transport counters.
    #[must_use]
    pub fn stats(&self) -> MuxStats {
        self.stats
    }

    /// Messages accepted on `channel` but not yet acknowledged by the peer.
    /// For unreliable channels this is the number waiting for the next send
    /// budget; delivered unreliable messages are never tracked as outstanding.
    #[must_use]
    pub fn queued(&self, channel: NetChannel) -> usize {
        match channel {
            NetChannel::EventsReliable => self.events.pending.len() + self.events.sender.unacked(),
            NetChannel::StateUnreliable => self.state.pending.len(),
            NetChannel::InputUnreliable => self.input.pending.len(),
            NetChannel::RpcReliable => self.rpc.pending.len() + self.rpc.sender.unacked(),
            NetChannel::BulkReliable => self.bulk.pending.len() + self.bulk.sender.unacked(),
        }
    }

    /// Whether the next reliable message on `channel` would be refused by the
    /// bounded admission queue. Unreliable channels replace older data and
    /// therefore never report backpressure.
    #[must_use]
    pub fn is_backpressured(&self, channel: NetChannel) -> bool {
        channel.is_reliable()
            && self.reliable_lane(channel).pending.len() >= self.config.reliable_queue
    }

    /// Access the wrapped transport for diagnostics or socket configuration.
    pub fn datagram_mut(&mut self) -> &mut D {
        &mut self.datagram
    }

    fn reliable_lane_mut(&mut self, channel: NetChannel) -> &mut ReliableLane {
        match channel {
            NetChannel::EventsReliable => &mut self.events,
            NetChannel::RpcReliable => &mut self.rpc,
            NetChannel::BulkReliable => &mut self.bulk,
            _ => unreachable!("not a reliable channel"),
        }
    }

    fn reliable_lane(&self, channel: NetChannel) -> &ReliableLane {
        match channel {
            NetChannel::EventsReliable => &self.events,
            NetChannel::RpcReliable => &self.rpc,
            NetChannel::BulkReliable => &self.bulk,
            _ => unreachable!("not a reliable channel"),
        }
    }

    fn drain_inbound(&mut self) {
        let now = self.now;
        while let Some(packet) = self.datagram.recv() {
            if packet.len() > self.config.path_mtu {
                self.stats.malformed_datagrams = self.stats.malformed_datagrams.saturating_add(1);
                continue;
            }
            let Some((kind, channel, seq, payload)) = decode_packet(&packet) else {
                self.stats.malformed_datagrams = self.stats.malformed_datagrams.saturating_add(1);
                continue;
            };
            self.stats.datagrams_received = self.stats.datagrams_received.saturating_add(1);
            if kind == KIND_ACK {
                if channel.is_reliable()
                    && self.reliable_lane_mut(channel).sender.ack(seq, now)
                    && let Some(flight) = self.flights.remove(&(channel, seq))
                {
                    self.congestion
                        .acknowledged(flight.bytes, now.saturating_sub(flight.first_sent));
                }
                continue;
            }
            if kind != KIND_DATA {
                self.stats.malformed_datagrams = self.stats.malformed_datagrams.saturating_add(1);
                continue;
            }

            if channel.is_reliable() {
                let queue_limit = self.config.reliable_queue;
                let lane = self.reliable_lane_mut(channel);
                // Do not acknowledge new data when the application stops
                // consuming. A contiguous run can release at most one receive
                // window in addition to this bounded delivery queue. Stale
                // packets are still re-ACKed so a lost ACK cannot stall a peer.
                if lane.inbound.len() >= queue_limit && seq >= lane.receiver.next() {
                    continue;
                }
                let (delivered, ack) = lane.receiver.receive(seq, payload.to_vec());
                lane.inbound.extend(delivered);
                if let Some(ack) = ack {
                    let packet = encode_packet(KIND_ACK, channel, ack, &[]);
                    self.datagram.send(&packet);
                    self.stats.datagrams_sent = self.stats.datagrams_sent.saturating_add(1);
                }
            } else {
                let lane = match channel {
                    NetChannel::StateUnreliable => &mut self.state,
                    NetChannel::InputUnreliable => &mut self.input,
                    _ => unreachable!(),
                };
                if lane.newest_received.is_none_or(|newest| seq > newest) {
                    lane.newest_received = Some(seq);
                    if channel == NetChannel::StateUnreliable {
                        if !lane.inbound.is_empty() {
                            self.stats.unreliable_dropped =
                                self.stats.unreliable_dropped.saturating_add(1);
                        }
                        lane.inbound.clear();
                    } else if lane.inbound.len() >= self.config.input_queue {
                        lane.inbound.pop_front();
                        self.stats.unreliable_dropped =
                            self.stats.unreliable_dropped.saturating_add(1);
                    }
                    lane.inbound.push_back(payload.to_vec());
                } else {
                    self.stats.unreliable_dropped = self.stats.unreliable_dropped.saturating_add(1);
                }
            }
        }
    }

    fn admit_reliable(&mut self) {
        // Latency-sensitive reliable traffic is admitted before bulk. Each lane
        // has its own ARQ window, so a lost bulk packet cannot block events/RPC.
        for channel in [
            NetChannel::EventsReliable,
            NetChannel::RpcReliable,
            NetChannel::BulkReliable,
        ] {
            loop {
                let Some(payload_len) = self
                    .reliable_lane_mut(channel)
                    .pending
                    .front()
                    .map(Vec::len)
                else {
                    break;
                };
                let wire_len = HEADER_LEN
                    .saturating_add(self.datagram.security_overhead())
                    .saturating_add(payload_len);
                if !self.congestion.can_admit(wire_len)
                    || self.reliable_lane_mut(channel).sender.is_full()
                {
                    break;
                }
                let payload = self
                    .reliable_lane_mut(channel)
                    .pending
                    .pop_front()
                    .expect("front existed");
                let seq = self
                    .reliable_lane_mut(channel)
                    .sender
                    .send(payload)
                    .expect("sender capacity checked");
                self.congestion.admitted(wire_len);
                self.flights.insert(
                    (channel, seq),
                    Flight {
                        bytes: wire_len,
                        first_sent: self.now,
                        transmissions: 0,
                    },
                );
            }
        }
    }

    fn flush_reliable(&mut self) {
        let now = self.now;
        for channel in [
            NetChannel::EventsReliable,
            NetChannel::RpcReliable,
            NetChannel::BulkReliable,
        ] {
            let mut outgoing = Vec::new();
            self.reliable_lane_mut(channel)
                .sender
                .tick(now, &mut outgoing);
            for (seq, payload) in outgoing {
                if let Some(flight) = self.flights.get_mut(&(channel, seq)) {
                    if flight.transmissions > 0 {
                        self.stats.retransmissions = self.stats.retransmissions.saturating_add(1);
                    }
                    flight.transmissions = flight.transmissions.saturating_add(1);
                }
                let packet = encode_packet(KIND_DATA, channel, seq, &payload);
                self.datagram.send(&packet);
                self.stats.datagrams_sent = self.stats.datagrams_sent.saturating_add(1);
            }
        }
    }

    fn flush_unreliable(&mut self) {
        let mut budget = self.config.unreliable_budget_per_tick;
        // Current input is more latency-sensitive than a state snapshot.
        for channel in [NetChannel::InputUnreliable, NetChannel::StateUnreliable] {
            loop {
                let lane = match channel {
                    NetChannel::InputUnreliable => &mut self.input,
                    NetChannel::StateUnreliable => &mut self.state,
                    _ => unreachable!(),
                };
                let Some(payload) = lane.pending.front() else {
                    break;
                };
                let wire_len = HEADER_LEN
                    .saturating_add(self.datagram.security_overhead())
                    .saturating_add(payload.len());
                if wire_len > budget {
                    // State is superseded by the next snapshot. Input is retained
                    // in its small bounded tail for a later tick.
                    if channel == NetChannel::StateUnreliable {
                        lane.pending.pop_front();
                        self.stats.unreliable_dropped =
                            self.stats.unreliable_dropped.saturating_add(1);
                    }
                    break;
                }
                let payload = lane.pending.pop_front().expect("front existed");
                let seq = lane.next_send;
                lane.next_send = lane.next_send.wrapping_add(1);
                let packet = encode_packet(KIND_DATA, channel, seq, &payload);
                self.datagram.send(&packet);
                self.stats.datagrams_sent = self.stats.datagrams_sent.saturating_add(1);
                budget -= wire_len;
            }
        }
    }
}

fn encode_packet(kind: u8, channel: NetChannel, seq: u64, payload: &[u8]) -> Vec<u8> {
    let mut packet = Vec::with_capacity(HEADER_LEN + payload.len());
    packet.push(WIRE_VERSION);
    packet.push(kind);
    packet.push(channel as u8);
    packet.push(0); // reserved for future flags; must remain zero in v1
    packet.extend_from_slice(&seq.to_le_bytes());
    packet.extend_from_slice(payload);
    packet
}

fn decode_packet(packet: &[u8]) -> Option<(u8, NetChannel, u64, &[u8])> {
    let header = packet.get(..HEADER_LEN)?;
    if header[0] != WIRE_VERSION || header[3] != 0 {
        return None;
    }
    let kind = header[1];
    let channel = NetChannel::from_wire(header[2])?;
    if !matches!(kind, KIND_DATA | KIND_ACK)
        || (kind == KIND_ACK && (!channel.is_reliable() || packet.len() != HEADER_LEN))
    {
        return None;
    }
    let seq = u64::from_le_bytes(header[4..12].try_into().ok()?);
    Some((kind, channel, seq, &packet[HEADER_LEN..]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[derive(Default)]
    struct MockTransport {
        outbox: VecDeque<Vec<u8>>,
        inbox: VecDeque<Vec<u8>>,
    }

    impl Transport for MockTransport {
        fn send(&mut self, bytes: &[u8]) {
            self.outbox.push_back(bytes.to_vec());
        }

        fn recv(&mut self) -> Option<Vec<u8>> {
            self.inbox.pop_front()
        }
    }

    fn exchange(a: &mut ChannelMux<MockTransport>, b: &mut ChannelMux<MockTransport>) {
        b.datagram.inbox.extend(a.datagram.outbox.drain(..));
        a.datagram.inbox.extend(b.datagram.outbox.drain(..));
    }

    #[test]
    fn reliable_channels_do_not_head_of_line_block_each_other() {
        let config = MuxConfig {
            min_rto: 2,
            ..MuxConfig::default()
        };
        let mut a = ChannelMux::new(MockTransport::default(), config);
        let mut b = ChannelMux::new(MockTransport::default(), config);
        a.send(NetChannel::EventsReliable, b"event").unwrap();
        a.send(NetChannel::RpcReliable, b"rpc").unwrap();
        a.tick();

        // Drop only the event data packet and deliver the RPC packet.
        while let Some(packet) = a.datagram.outbox.pop_front() {
            let (_, channel, _, _) = decode_packet(&packet).unwrap();
            if channel == NetChannel::RpcReliable {
                b.datagram.inbox.push_back(packet);
            }
        }
        b.tick();
        assert_eq!(b.recv(NetChannel::RpcReliable), Some(b"rpc".to_vec()));
        assert_eq!(b.recv(NetChannel::EventsReliable), None);

        // The lost event is retransmitted independently and arrives later.
        exchange(&mut a, &mut b);
        for _ in 0..4 {
            a.tick();
            exchange(&mut a, &mut b);
            b.tick();
            exchange(&mut a, &mut b);
        }
        assert_eq!(b.recv(NetChannel::EventsReliable), Some(b"event".to_vec()));
    }

    #[test]
    fn state_keeps_only_the_newest_unsent_snapshot() {
        let mut a = ChannelMux::new(MockTransport::default(), MuxConfig::default());
        let mut b = ChannelMux::new(MockTransport::default(), MuxConfig::default());
        assert_eq!(
            a.send(NetChannel::StateUnreliable, b"old"),
            Ok(QueueOutcome::Queued)
        );
        assert_eq!(
            a.send(NetChannel::StateUnreliable, b"new"),
            Ok(QueueOutcome::ReplacedOlder)
        );
        a.tick();
        exchange(&mut a, &mut b);
        b.tick();
        assert_eq!(b.recv(NetChannel::StateUnreliable), Some(b"new".to_vec()));
        assert_eq!(b.recv(NetChannel::StateUnreliable), None);
        assert_eq!(a.stats().unreliable_dropped, 1);
    }

    #[test]
    fn shared_congestion_window_bounds_reliable_admission() {
        let config = MuxConfig {
            path_mtu: 128,
            initial_cwnd_bytes: 256,
            max_cwnd_bytes: 1_024,
            reliable_window: 32,
            ..MuxConfig::default()
        };
        let mut a = ChannelMux::new(MockTransport::default(), config);
        let mut b = ChannelMux::new(MockTransport::default(), config);
        for _ in 0..10 {
            a.send(NetChannel::EventsReliable, &[7; 80]).unwrap();
        }
        a.tick();
        assert!(a.bytes_in_flight() <= a.congestion_window());
        assert_eq!(
            a.bytes_in_flight(),
            184,
            "two 92-byte wire packets admitted"
        );

        // ACKs release bytes and allow queued messages to enter on later ticks.
        exchange(&mut a, &mut b);
        b.tick();
        exchange(&mut a, &mut b);
        a.tick();
        assert!(a.bytes_in_flight() <= a.congestion_window());
        assert!(a.stats().datagrams_sent >= 4);
    }

    #[test]
    fn mtu_and_reliable_queue_limits_are_explicit() {
        let config = MuxConfig {
            path_mtu: 64,
            reliable_queue: 1,
            ..MuxConfig::default()
        };
        let mut mux = ChannelMux::new(MockTransport::default(), config);
        assert_eq!(mux.max_payload(), 52);
        assert_eq!(
            mux.send(NetChannel::EventsReliable, &[0; 53]),
            Err(MuxSendError::TooLarge { len: 53, max: 52 })
        );
        mux.send(NetChannel::EventsReliable, b"one").unwrap();
        assert_eq!(
            mux.send(NetChannel::EventsReliable, b"two"),
            Err(MuxSendError::Backpressured(NetChannel::EventsReliable))
        );
        assert_eq!(mux.stats().backpressure_events, 1);
    }

    #[test]
    fn malformed_and_stale_unreliable_datagrams_are_ignored() {
        let mut mux = ChannelMux::new(MockTransport::default(), MuxConfig::default());
        mux.datagram.inbox.push_back(vec![0xFF; 4]);
        mux.datagram.inbox.push_back(encode_packet(
            KIND_DATA,
            NetChannel::StateUnreliable,
            5,
            b"fresh",
        ));
        mux.datagram.inbox.push_back(encode_packet(
            KIND_DATA,
            NetChannel::StateUnreliable,
            4,
            b"stale",
        ));
        mux.tick();
        assert_eq!(
            mux.recv(NetChannel::StateUnreliable),
            Some(b"fresh".to_vec())
        );
        assert_eq!(mux.recv(NetChannel::StateUnreliable), None);
        assert_eq!(mux.stats().malformed_datagrams, 1);
        assert_eq!(mux.stats().unreliable_dropped, 1);
    }

    #[test]
    fn reversed_reliable_packets_are_delivered_once_in_order_per_lane() {
        let mut a = ChannelMux::new(MockTransport::default(), MuxConfig::default());
        let mut b = ChannelMux::new(MockTransport::default(), MuxConfig::default());
        for channel in [NetChannel::EventsReliable, NetChannel::RpcReliable] {
            for byte in 0..4 {
                a.send(channel, &[byte]).unwrap();
            }
        }
        a.tick();
        while let Some(packet) = a.datagram.outbox.pop_back() {
            b.datagram.inbox.push_back(packet.clone());
            b.datagram.inbox.push_back(packet);
        }
        b.tick();
        for channel in [NetChannel::EventsReliable, NetChannel::RpcReliable] {
            for byte in 0..4 {
                assert_eq!(b.recv(channel), Some(vec![byte]));
            }
            assert_eq!(b.recv(channel), None);
        }
        exchange(&mut a, &mut b);
        a.tick();
        assert_eq!(a.queued(NetChannel::EventsReliable), 0);
        assert_eq!(a.queued(NetChannel::RpcReliable), 0);
        assert_eq!(a.bytes_in_flight(), 0);
    }

    #[test]
    fn newest_state_survives_reverse_delivery_and_late_duplicates() {
        let mut mux = ChannelMux::new(MockTransport::default(), MuxConfig::default());
        for seq in [2, 1, 0, 3, 2] {
            mux.datagram.inbox.push_back(encode_packet(
                KIND_DATA,
                NetChannel::StateUnreliable,
                seq,
                &[seq as u8],
            ));
        }
        mux.tick();
        assert_eq!(mux.recv(NetChannel::StateUnreliable), Some(vec![3]));
        assert_eq!(mux.recv(NetChannel::StateUnreliable), None);
        assert_eq!(mux.stats().unreliable_dropped, 4);
        mux.datagram.inbox.push_back(encode_packet(
            KIND_DATA,
            NetChannel::StateUnreliable,
            2,
            b"late",
        ));
        mux.tick();
        assert_eq!(mux.recv(NetChannel::StateUnreliable), None);
    }

    #[test]
    fn full_inbound_queue_withholds_acks_and_recovers_after_consumption() {
        let config = MuxConfig {
            reliable_queue: 1,
            reliable_window: 2,
            min_rto: 2,
            ..MuxConfig::default()
        };
        let mut a = ChannelMux::new(MockTransport::default(), config);
        let mut b = ChannelMux::new(MockTransport::default(), config);
        a.send(NetChannel::EventsReliable, b"first").unwrap();
        a.tick();
        a.send(NetChannel::EventsReliable, b"second").unwrap();
        a.tick();
        exchange(&mut a, &mut b);
        b.tick();
        assert_eq!(b.events.inbound.len(), 1);
        exchange(&mut a, &mut b);
        a.tick();
        assert_eq!(
            a.queued(NetChannel::EventsReliable),
            1,
            "second packet was not acknowledged"
        );
        assert_eq!(b.recv(NetChannel::EventsReliable), Some(b"first".to_vec()));
        for _ in 0..8 {
            a.tick();
            exchange(&mut a, &mut b);
            b.tick();
            exchange(&mut a, &mut b);
        }
        assert_eq!(b.recv(NetChannel::EventsReliable), Some(b"second".to_vec()));
        assert_eq!(b.recv(NetChannel::EventsReliable), None);
        assert_eq!(a.queued(NetChannel::EventsReliable), 0);
    }

    #[test]
    fn unreliable_input_receive_queue_is_bounded() {
        let config = MuxConfig {
            input_queue: 2,
            ..MuxConfig::default()
        };
        let mut mux = ChannelMux::new(MockTransport::default(), config);
        for seq in 0..5 {
            mux.datagram.inbox.push_back(encode_packet(
                KIND_DATA,
                NetChannel::InputUnreliable,
                seq,
                &[seq as u8],
            ));
        }
        mux.tick();
        assert_eq!(mux.recv(NetChannel::InputUnreliable), Some(vec![3]));
        assert_eq!(mux.recv(NetChannel::InputUnreliable), Some(vec![4]));
        assert_eq!(mux.recv(NetChannel::InputUnreliable), None);
        assert_eq!(mux.stats().unreliable_dropped, 3);
    }

    #[test]
    fn wrong_wire_version_oversized_and_invalid_acks_cannot_release_flights() {
        let config = MuxConfig {
            path_mtu: 64,
            ..MuxConfig::default()
        };
        let mut mux = ChannelMux::new(MockTransport::default(), config);
        mux.send(NetChannel::EventsReliable, b"event").unwrap();
        mux.tick();
        let charged = mux.bytes_in_flight();
        let mut version = encode_packet(KIND_ACK, NetChannel::EventsReliable, 0, &[]);
        version[0] = WIRE_VERSION + 1;
        mux.datagram.inbox.push_back(version);
        mux.datagram.inbox.push_back(encode_packet(
            KIND_ACK,
            NetChannel::EventsReliable,
            0,
            b"illegal payload",
        ));
        mux.datagram
            .inbox
            .push_back(encode_packet(KIND_ACK, NetChannel::StateUnreliable, 0, &[]));
        mux.datagram.inbox.push_back(encode_packet(
            KIND_DATA,
            NetChannel::StateUnreliable,
            0,
            &[0; 53],
        ));
        mux.tick();
        assert_eq!(mux.stats().malformed_datagrams, 4);
        assert_eq!(mux.bytes_in_flight(), charged);
        assert_eq!(mux.queued(NetChannel::EventsReliable), 1);
        mux.datagram
            .inbox
            .push_back(encode_packet(KIND_ACK, NetChannel::EventsReliable, 0, &[]));
        mux.tick();
        assert_eq!(mux.bytes_in_flight(), 0);
    }

    #[test]
    fn protocol_negotiation_requires_equal_versions_and_nonzero_windows() {
        let local = ProtocolOffer::current(8);
        assert_eq!(
            ProtocolOffer::current(MIN_ROLLBACK_WINDOW).validate(),
            Ok(())
        );
        assert_eq!(
            ProtocolOffer::current(0).validate(),
            Err(ProtocolError::InvalidRollbackWindow { offered: 0 })
        );
        assert_eq!(
            negotiate_protocol(local, ProtocolOffer::current(3)),
            Ok(NegotiatedProtocol {
                version: local.version,
                rollback_window: 3
            })
        );
        assert_eq!(
            negotiate_protocol(local, ProtocolOffer::current(3))
                .expect("valid offers negotiate")
                .max_unconfirmed_predictions(),
            2
        );
        assert_eq!(
            negotiate_protocol(
                ProtocolOffer::current(8),
                ProtocolOffer::current(MIN_ROLLBACK_WINDOW),
            )
            .expect("baseline-only offer negotiates")
            .max_unconfirmed_predictions(),
            0
        );
        assert_eq!(
            negotiate_protocol(local, ProtocolOffer::current(0)),
            Err(ProtocolError::InvalidRollbackWindow { offered: 0 })
        );
        assert_eq!(
            negotiate_protocol(ProtocolOffer::current(0), local),
            Err(ProtocolError::InvalidRollbackWindow { offered: 0 })
        );
        assert_eq!(
            negotiate_protocol(
                local,
                ProtocolOffer {
                    version: local.version + 1,
                    rollback_window: 8
                }
            ),
            Err(ProtocolError::VersionMismatch {
                local: local.version,
                remote: local.version + 1
            })
        );
    }
}
