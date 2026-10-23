//! A deterministic reliable-session seam over a host-provided transport.
//!
//! The session does not open sockets or perform cryptographic verification.
//! The host authenticates an opaque token before attaching a transport, while
//! this module provides bounded ordered delivery, cumulative acknowledgements,
//! retry limits, and duplicate suppression in `no_std` builds.

use alloc::collections::{BTreeMap, VecDeque};
use alloc::rc::Rc;
use alloc::vec::Vec;
use core::cell::RefCell;

use crate::wire::{decode_packet, encode_packet, ByteTransport, WireCodecError};
use crate::{AoiSubscription, AuthorityPolicy, DeltaEnvelope, PermissionError, SubscriptionId};
use prism_world_persist::ActorId;

/// Stable identity of one authenticated session.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SessionId(pub u128);

/// Opaque credential supplied by the host during session authentication.
///
/// This is a comparison seam rather than a cryptographic token format. A
/// production host should verify it through its authenticated control plane.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct AuthToken(pub u128);

/// Session lifecycle state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SessionState {
    /// No token has been accepted yet.
    Unauthenticated,
    /// The session may send and receive application messages.
    Authenticated,
    /// Delivery failed permanently after the retry budget was exhausted.
    Failed {
        /// Sequence whose retry budget was exhausted.
        sequence: u64,
    },
    /// The host closed the session.
    Closed,
}

/// An application message carried by a session.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum WorldMessage {
    /// A durable world transaction annotated with its routing sector.
    Delta(DeltaEnvelope),
    /// An AOI subscription request.
    Subscribe(AoiSubscription),
    /// Removes a previously registered AOI subscription.
    Unsubscribe(SubscriptionId),
}

/// A packet exchanged through a host-provided transport.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TransportPacket {
    /// An ordered application packet.
    Data {
        /// Session this packet belongs to.
        session: SessionId,
        /// Monotonically increasing packet sequence, starting at one.
        sequence: u64,
        /// Application payload.
        message: WorldMessage,
    },
    /// A cumulative acknowledgement for every data sequence through this one.
    Ack {
        /// Session this acknowledgement belongs to.
        session: SessionId,
        /// Highest contiguous sequence received.
        sequence: u64,
    },
}

impl TransportPacket {
    /// Returns the session identity carried by the packet.
    pub const fn session(&self) -> SessionId {
        match self {
            Self::Data { session, .. } | Self::Ack { session, .. } => *session,
        }
    }
}

/// A non-blocking transport adapter owned by the host application.
pub trait Transport {
    /// Adapter-specific send error.
    type Error;

    /// Attempts to enqueue one packet for the peer. On error, the packet is
    /// considered unsent and may be retried without consuming an attempt.
    fn try_send(&mut self, packet: TransportPacket) -> Result<(), Self::Error>;

    /// Removes the next received packet, if available.
    fn try_recv(&mut self) -> Option<TransportPacket>;
}

/// Errors from the deterministic in-memory transport.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryTransportError {
    /// The configured queue cannot have zero capacity.
    InvalidCapacity,
    /// The peer's bounded receive queue is full.
    Backpressure,
}

type PacketQueue = Rc<RefCell<VecDeque<TransportPacket>>>;

/// A bounded in-memory transport endpoint for tests and host seams.
pub struct MemoryTransport {
    inbound: PacketQueue,
    outbound: PacketQueue,
    capacity: usize,
}

impl MemoryTransport {
    /// Creates two connected bounded endpoints.
    pub fn pair(capacity: usize) -> Result<(Self, Self), MemoryTransportError> {
        if capacity == 0 {
            return Err(MemoryTransportError::InvalidCapacity);
        }
        let left_to_right = Rc::new(RefCell::new(VecDeque::new()));
        let right_to_left = Rc::new(RefCell::new(VecDeque::new()));
        Ok((
            Self {
                inbound: Rc::clone(&right_to_left),
                outbound: Rc::clone(&left_to_right),
                capacity,
            },
            Self {
                inbound: left_to_right,
                outbound: right_to_left,
                capacity,
            },
        ))
    }

    /// Injects a packet at the back of this endpoint's receive queue.
    pub fn inject(&mut self, packet: TransportPacket) -> Result<(), MemoryTransportError> {
        let mut inbound = self.inbound.borrow_mut();
        if inbound.len() >= self.capacity {
            return Err(MemoryTransportError::Backpressure);
        }
        inbound.push_back(packet);
        Ok(())
    }

    /// Returns the number of packets waiting to be received.
    pub fn queued(&self) -> usize {
        self.inbound.borrow().len()
    }
}

impl Transport for MemoryTransport {
    type Error = MemoryTransportError;

    fn try_send(&mut self, packet: TransportPacket) -> Result<(), Self::Error> {
        let mut outbound = self.outbound.borrow_mut();
        if outbound.len() >= self.capacity {
            return Err(MemoryTransportError::Backpressure);
        }
        outbound.push_back(packet);
        Ok(())
    }

    fn try_recv(&mut self) -> Option<TransportPacket> {
        self.inbound.borrow_mut().pop_front()
    }
}

/// Limits and identity used to construct a [`Session`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SessionConfig {
    /// Session identity shared with the peer binding.
    pub session: SessionId,
    /// Local actor used to validate outgoing AOI subscriptions.
    pub actor: ActorId,
    /// Opaque token accepted by [`Session::authenticate`].
    pub token: AuthToken,
    /// Authority and replication policy applied to delta messages.
    pub authority: AuthorityPolicy,
    /// Maximum number of unacknowledged data packets.
    pub outbound_capacity: usize,
    /// Maximum number of out-of-order data packets retained.
    pub inbound_capacity: usize,
    /// Maximum total sends per data packet, including its first send.
    pub max_attempts: u8,
    /// Minimum tick interval between attempts after the first send.
    pub retry_after_ticks: u64,
}

/// Invalid session limits.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SessionConfigError {
    /// A data or inbound reorder queue has no storage.
    ZeroCapacity,
    /// At least one send is required for every packet.
    ZeroAttempts,
}

impl SessionConfig {
    /// Validates limits without allocating a session.
    pub const fn validate(self) -> Result<(), SessionConfigError> {
        if self.outbound_capacity == 0 || self.inbound_capacity == 0 {
            return Err(SessionConfigError::ZeroCapacity);
        }
        if self.max_attempts == 0 {
            return Err(SessionConfigError::ZeroAttempts);
        }
        Ok(())
    }
}

/// Failures while queuing or changing session state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SessionError {
    /// A token was required but the session is not authenticated.
    NotAuthenticated,
    /// The supplied token does not match the configured credential.
    InvalidToken,
    /// Authentication was attempted more than once.
    AlreadyAuthenticated,
    /// The session cannot be used after close or terminal failure.
    Unavailable,
    /// The bounded outbound data queue is full.
    OutboundFull,
    /// The next sequence number cannot be represented.
    SequenceExhausted,
    /// An outgoing subscription names another actor.
    InvalidSubscriber {
        /// Actor configured for this session.
        expected: ActorId,
        /// Actor supplied by the message.
        actual: ActorId,
    },
    /// A delta violates the configured authority rule.
    Permission(PermissionError),
}

/// Failures while accepting a transport packet.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReceiveError {
    /// The packet belongs to another session.
    WrongSession {
        /// Session configured on the receiver.
        expected: SessionId,
        /// Session carried by the packet.
        actual: SessionId,
    },
    /// Session state or message validation rejected the packet.
    Session(SessionError),
    /// Sequence zero or the reserved maximum cannot be delivered.
    InvalidSequence(u64),
    /// The bounded reorder window is full.
    InboundFull,
    /// A data sequence is farther ahead than the configured reorder window.
    OutsideReceiveWindow {
        /// Next contiguous sequence expected by the receiver.
        expected: u64,
        /// Sequence carried by the rejected packet.
        received: u64,
        /// Maximum accepted sequence distance.
        window: usize,
    },
    /// An acknowledgement claims data that has never been sent.
    InvalidAck {
        /// Highest local sequence sent so far.
        highest_sent: u64,
        /// Sequence claimed by the peer.
        acknowledged: u64,
    },
}

/// Adapter failure while flushing a session.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FlushError<E> {
    /// The session has not completed authentication.
    NotAuthenticated,
    /// A packet exhausted its configured retry budget.
    RetryExhausted(u64),
    /// The host transport refused a send.
    Transport(E),
    /// The host clock moved backwards for a pending packet.
    ClockRegression {
        /// Packet whose retry clock regressed.
        sequence: u64,
        /// Previously observed send tick.
        previous: u64,
        /// Current flush tick.
        now: u64,
    },
}

/// Failures while flushing packets through the byte-channel codec.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ByteFlushError<E> {
    /// The session has not completed authentication.
    NotAuthenticated,
    /// A packet exhausted its configured retry budget.
    RetryExhausted(u64),
    /// The host byte transport refused a send.
    Transport(E),
    /// The host clock moved backwards for a pending packet.
    ClockRegression {
        /// Packet whose retry clock regressed.
        sequence: u64,
        /// Previously observed send tick.
        previous: u64,
        /// Current flush tick.
        now: u64,
    },
    /// The packet could not be represented by the bounded wire codec.
    Codec(WireCodecError),
}

/// Failures while receiving and decoding one byte-channel frame.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ByteReceiveError<E> {
    /// The host byte transport returned an I/O error.
    Transport(E),
    /// The frame failed bounded wire decoding.
    Codec(WireCodecError),
    /// The decoded packet failed session validation or ordering.
    Session(ReceiveError),
}

/// Result of accepting one packet.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReceiveOutcome {
    /// The acknowledgement released local pending packets through this point.
    Acknowledged {
        /// Highest acknowledged sequence.
        sequence: u64,
        /// Number of local packets removed from the retry queue.
        released: usize,
    },
    /// A new packet was retained until its missing predecessor arrives.
    Buffered {
        /// Sequence retained until its predecessors arrive.
        sequence: u64,
    },
    /// The packet had already been received and was not delivered again.
    Duplicate {
        /// Sequence that was already delivered or retained.
        sequence: u64,
    },
    /// Contiguous packets were delivered in sequence order.
    Delivered {
        /// Sequence of the packet that triggered this delivery.
        sequence: u64,
        /// Messages delivered, including contiguous buffered packets.
        messages: Vec<WorldMessage>,
    },
}

/// Result counters from one non-blocking flush.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct FlushReport {
    /// Number of packets successfully handed to the transport.
    pub sent: usize,
    /// Number of those packets that were retries.
    pub retries: usize,
    /// Number of data packets still awaiting acknowledgement.
    pub pending: usize,
}

#[derive(Clone, Debug)]
struct PendingData {
    sequence: u64,
    message: WorldMessage,
    last_sent: Option<u64>,
    attempts: u8,
}

/// A bounded authenticated reliable session.
pub struct Session {
    config: SessionConfig,
    state: SessionState,
    next_send_sequence: u64,
    highest_sent_sequence: u64,
    next_receive_sequence: u64,
    outbound: VecDeque<PendingData>,
    inbound: BTreeMap<u64, WorldMessage>,
    pending_ack: Option<u64>,
}

impl Session {
    /// Creates an unauthenticated session after validating all queue limits.
    pub fn new(config: SessionConfig) -> Result<Self, SessionConfigError> {
        config.validate()?;
        Ok(Self {
            config,
            state: SessionState::Unauthenticated,
            next_send_sequence: 1,
            highest_sent_sequence: 0,
            next_receive_sequence: 1,
            outbound: VecDeque::with_capacity(config.outbound_capacity),
            inbound: BTreeMap::new(),
            pending_ack: None,
        })
    }

    /// Returns the current lifecycle state.
    pub const fn state(&self) -> SessionState {
        self.state
    }

    /// Returns the configured session identity.
    pub const fn id(&self) -> SessionId {
        self.config.session
    }

    /// Returns the number of data packets awaiting acknowledgement.
    pub fn pending(&self) -> usize {
        self.outbound.len()
    }

    /// Authenticates this session with its configured opaque token.
    pub fn authenticate(&mut self, token: AuthToken) -> Result<(), SessionError> {
        match self.state {
            SessionState::Unauthenticated => {
                if token != self.config.token {
                    return Err(SessionError::InvalidToken);
                }
                self.state = SessionState::Authenticated;
                Ok(())
            }
            SessionState::Authenticated => Err(SessionError::AlreadyAuthenticated),
            SessionState::Failed { .. } | SessionState::Closed => Err(SessionError::Unavailable),
        }
    }

    /// Closes this session and prevents further sends or receives.
    pub const fn close(&mut self) {
        self.state = SessionState::Closed;
    }

    /// Queues one application message and returns its assigned sequence.
    pub fn send(&mut self, message: WorldMessage) -> Result<u64, SessionError> {
        self.ensure_authenticated()?;
        self.validate_outgoing(&message)?;
        if self.outbound.len() >= self.config.outbound_capacity {
            return Err(SessionError::OutboundFull);
        }
        let sequence = self.next_send_sequence;
        if sequence == u64::MAX {
            return Err(SessionError::SequenceExhausted);
        }
        self.next_send_sequence = sequence + 1;
        self.outbound.push_back(PendingData {
            sequence,
            message,
            last_sent: None,
            attempts: 0,
        });
        Ok(sequence)
    }

    /// Sends all currently eligible acknowledgements and data packets.
    pub fn flush<T: Transport>(
        &mut self,
        transport: &mut T,
        now: u64,
    ) -> Result<FlushReport, FlushError<T::Error>> {
        if self.state != SessionState::Authenticated {
            return Err(FlushError::NotAuthenticated);
        }
        let mut report = FlushReport::default();
        if let Some(sequence) = self.pending_ack {
            transport
                .try_send(TransportPacket::Ack {
                    session: self.config.session,
                    sequence,
                })
                .map_err(FlushError::Transport)?;
            self.pending_ack = None;
            report.sent += 1;
        }

        for pending in &mut self.outbound {
            if let Some(previous) = pending.last_sent {
                if now < previous {
                    return Err(FlushError::ClockRegression {
                        sequence: pending.sequence,
                        previous,
                        now,
                    });
                }
                if now - previous < self.config.retry_after_ticks {
                    continue;
                }
                if pending.attempts >= self.config.max_attempts {
                    self.state = SessionState::Failed {
                        sequence: pending.sequence,
                    };
                    return Err(FlushError::RetryExhausted(pending.sequence));
                }
            }

            let retry = pending.attempts > 0;
            transport
                .try_send(TransportPacket::Data {
                    session: self.config.session,
                    sequence: pending.sequence,
                    message: pending.message.clone(),
                })
                .map_err(FlushError::Transport)?;
            pending.last_sent = Some(now);
            pending.attempts += 1;
            report.sent += 1;
            if retry {
                report.retries += 1;
            }
            self.highest_sent_sequence = self.highest_sent_sequence.max(pending.sequence);
        }
        report.pending = self.outbound.len();
        Ok(report)
    }

    /// Sends eligible acknowledgements and data through the versioned byte
    /// codec. Codec failure occurs before the corresponding pending packet is
    /// marked sent; host backpressure likewise preserves its retry state.
    pub fn flush_bytes<T: ByteTransport>(
        &mut self,
        transport: &mut T,
        now: u64,
    ) -> Result<FlushReport, ByteFlushError<T::Error>> {
        if self.state != SessionState::Authenticated {
            return Err(ByteFlushError::NotAuthenticated);
        }
        let mut report = FlushReport::default();
        if let Some(sequence) = self.pending_ack {
            let frame = encode_packet(&TransportPacket::Ack {
                session: self.config.session,
                sequence,
            })
            .map_err(ByteFlushError::Codec)?;
            transport
                .try_send_bytes(&frame)
                .map_err(ByteFlushError::Transport)?;
            self.pending_ack = None;
            report.sent += 1;
        }

        for pending in &mut self.outbound {
            if let Some(previous) = pending.last_sent {
                if now < previous {
                    return Err(ByteFlushError::ClockRegression {
                        sequence: pending.sequence,
                        previous,
                        now,
                    });
                }
                if now - previous < self.config.retry_after_ticks {
                    continue;
                }
                if pending.attempts >= self.config.max_attempts {
                    self.state = SessionState::Failed {
                        sequence: pending.sequence,
                    };
                    return Err(ByteFlushError::RetryExhausted(pending.sequence));
                }
            }

            let retry = pending.attempts > 0;
            let frame = encode_packet(&TransportPacket::Data {
                session: self.config.session,
                sequence: pending.sequence,
                message: pending.message.clone(),
            })
            .map_err(ByteFlushError::Codec)?;
            transport
                .try_send_bytes(&frame)
                .map_err(ByteFlushError::Transport)?;
            pending.last_sent = Some(now);
            pending.attempts += 1;
            report.sent += 1;
            if retry {
                report.retries += 1;
            }
            self.highest_sent_sequence = self.highest_sent_sequence.max(pending.sequence);
        }
        report.pending = self.outbound.len();
        Ok(report)
    }

    /// Accepts one packet from the host transport.
    pub fn receive(&mut self, packet: TransportPacket) -> Result<ReceiveOutcome, ReceiveError> {
        if packet.session() != self.config.session {
            return Err(ReceiveError::WrongSession {
                expected: self.config.session,
                actual: packet.session(),
            });
        }
        self.ensure_authenticated().map_err(ReceiveError::Session)?;
        match packet {
            TransportPacket::Ack { sequence, .. } => self.receive_ack(sequence),
            TransportPacket::Data {
                sequence, message, ..
            } => self.receive_data(sequence, message),
        }
    }

    /// Decodes and accepts one complete byte-channel frame. A malformed frame
    /// is returned to the host as an explicit error and cannot advance ACK or
    /// receive state.
    pub fn receive_bytes<T: ByteTransport>(
        &mut self,
        transport: &mut T,
    ) -> Result<Option<ReceiveOutcome>, ByteReceiveError<T::Error>> {
        let Some(frame) = transport
            .try_recv_bytes()
            .map_err(ByteReceiveError::Transport)?
        else {
            return Ok(None);
        };
        let packet = decode_packet(&frame).map_err(ByteReceiveError::Codec)?;
        self.receive(packet)
            .map(Some)
            .map_err(ByteReceiveError::Session)
    }

    fn receive_ack(&mut self, sequence: u64) -> Result<ReceiveOutcome, ReceiveError> {
        if sequence == 0 || sequence > self.highest_sent_sequence {
            return Err(ReceiveError::InvalidAck {
                highest_sent: self.highest_sent_sequence,
                acknowledged: sequence,
            });
        }
        let before = self.outbound.len();
        self.outbound.retain(|pending| pending.sequence > sequence);
        Ok(ReceiveOutcome::Acknowledged {
            sequence,
            released: before - self.outbound.len(),
        })
    }

    fn receive_data(
        &mut self,
        sequence: u64,
        message: WorldMessage,
    ) -> Result<ReceiveOutcome, ReceiveError> {
        if sequence == 0 || sequence == u64::MAX {
            return Err(ReceiveError::InvalidSequence(sequence));
        }
        self.validate_incoming(&message)
            .map_err(ReceiveError::Session)?;
        if sequence < self.next_receive_sequence || self.inbound.contains_key(&sequence) {
            self.update_ack();
            return Ok(ReceiveOutcome::Duplicate { sequence });
        }
        if sequence > self.next_receive_sequence {
            let distance = sequence - self.next_receive_sequence;
            if u128::from(distance) > self.config.inbound_capacity as u128 {
                return Err(ReceiveError::OutsideReceiveWindow {
                    expected: self.next_receive_sequence,
                    received: sequence,
                    window: self.config.inbound_capacity,
                });
            }
            if self.inbound.len() >= self.config.inbound_capacity {
                return Err(ReceiveError::InboundFull);
            }
            self.inbound.insert(sequence, message);
            return Ok(ReceiveOutcome::Buffered { sequence });
        }

        let mut messages = Vec::new();
        messages.push(message);
        self.next_receive_sequence += 1;
        while let Some(message) = self.inbound.remove(&self.next_receive_sequence) {
            messages.push(message);
            self.next_receive_sequence += 1;
        }
        self.update_ack();
        Ok(ReceiveOutcome::Delivered { sequence, messages })
    }

    fn update_ack(&mut self) {
        let acknowledged = self.next_receive_sequence - 1;
        self.pending_ack = Some(
            self.pending_ack
                .map_or(acknowledged, |old| old.max(acknowledged)),
        );
    }

    fn validate_outgoing(&self, message: &WorldMessage) -> Result<(), SessionError> {
        match message {
            WorldMessage::Delta(envelope) => self
                .config
                .authority
                .authorize_delta(envelope)
                .map_err(SessionError::Permission),
            WorldMessage::Subscribe(subscription)
                if subscription.subscriber() != self.config.actor =>
            {
                Err(SessionError::InvalidSubscriber {
                    expected: self.config.actor,
                    actual: subscription.subscriber(),
                })
            }
            WorldMessage::Subscribe(_) | WorldMessage::Unsubscribe(_) => Ok(()),
        }
    }

    fn validate_incoming(&self, message: &WorldMessage) -> Result<(), SessionError> {
        if let WorldMessage::Delta(envelope) = message {
            self.config
                .authority
                .authorize_delta(envelope)
                .map_err(SessionError::Permission)?;
        }
        Ok(())
    }

    fn ensure_authenticated(&self) -> Result<(), SessionError> {
        match self.state {
            SessionState::Authenticated => Ok(()),
            SessionState::Unauthenticated => Err(SessionError::NotAuthenticated),
            SessionState::Failed { .. } | SessionState::Closed => Err(SessionError::Unavailable),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AabbI64, GlobalPos, SubscriptionId};
    use alloc::vec;

    fn config(session: u128, actor: u128, capacity: usize, retry_after: u64) -> SessionConfig {
        SessionConfig {
            session: SessionId(session),
            actor: ActorId(actor),
            token: AuthToken(0xfeed),
            authority: AuthorityPolicy::new(ActorId(7), crate::ReplicationPolicy::SharedLww),
            outbound_capacity: capacity,
            inbound_capacity: capacity,
            max_attempts: 3,
            retry_after_ticks: retry_after,
        }
    }

    fn session(session: u128, actor: u128, capacity: usize, retry_after: u64) -> Session {
        Session::new(config(session, actor, capacity, retry_after)).expect("valid config")
    }

    fn subscription(id: u128, actor: u128) -> AoiSubscription {
        AoiSubscription::new(
            SubscriptionId(id),
            ActorId(actor),
            AabbI64::new(GlobalPos::new(0, 0, 0), GlobalPos::new(1, 1, 1))
                .expect("ordered test bounds"),
        )
    }

    #[test]
    fn unauthenticated_and_invalid_policy_messages_are_rejected() {
        let mut session = session(1, 7, 4, 1);
        assert_eq!(
            session.send(WorldMessage::Unsubscribe(SubscriptionId(1))),
            Err(SessionError::NotAuthenticated)
        );
        assert_eq!(
            session.authenticate(AuthToken(0)),
            Err(SessionError::InvalidToken)
        );
        session
            .authenticate(AuthToken(0xfeed))
            .expect("token accepted");
        assert_eq!(
            session.send(WorldMessage::Subscribe(subscription(1, 8))),
            Err(SessionError::InvalidSubscriber {
                expected: ActorId(7),
                actual: ActorId(8),
            })
        );
    }

    #[test]
    fn memory_transport_delivers_order_and_suppresses_duplicates() {
        let mut sender = session(1, 7, 4, 10);
        let mut receiver = session(1, 8, 4, 10);
        sender.authenticate(AuthToken(0xfeed)).expect("sender auth");
        receiver
            .authenticate(AuthToken(0xfeed))
            .expect("receiver auth");
        sender
            .send(WorldMessage::Subscribe(subscription(1, 7)))
            .expect("first message");
        sender
            .send(WorldMessage::Subscribe(subscription(2, 7)))
            .expect("second message");
        let (mut sender_transport, mut receiver_transport) =
            MemoryTransport::pair(8).expect("valid transport");
        sender
            .flush(&mut sender_transport, 0)
            .expect("initial flush");
        let first = receiver_transport.try_recv().expect("first packet");
        let second = receiver_transport.try_recv().expect("second packet");
        receiver_transport
            .inject(second.clone())
            .expect("reorder insert");
        receiver_transport
            .inject(first.clone())
            .expect("reorder insert");
        assert_eq!(
            receiver.receive(receiver_transport.try_recv().expect("out of order")),
            Ok(ReceiveOutcome::Buffered { sequence: 2 })
        );
        assert_eq!(
            receiver.receive(receiver_transport.try_recv().expect("in order")),
            Ok(ReceiveOutcome::Delivered {
                sequence: 1,
                messages: vec![
                    WorldMessage::Subscribe(subscription(1, 7)),
                    WorldMessage::Subscribe(subscription(2, 7))
                ]
            })
        );
        assert_eq!(
            receiver.receive(first),
            Ok(ReceiveOutcome::Duplicate { sequence: 1 })
        );
        receiver
            .flush(&mut receiver_transport, 0)
            .expect("ack flush");
        sender
            .receive(sender_transport.try_recv().expect("ack packet"))
            .expect("ack accepted");
        assert_eq!(sender.pending(), 0);
    }

    #[test]
    fn byte_transport_codec_preserves_reliable_delivery_and_ack_release() {
        let mut sender = session(11, 7, 4, 10);
        let mut receiver = session(11, 8, 4, 10);
        sender.authenticate(AuthToken(0xfeed)).expect("sender auth");
        receiver
            .authenticate(AuthToken(0xfeed))
            .expect("receiver auth");
        sender
            .send(WorldMessage::Subscribe(subscription(9, 7)))
            .expect("message queued");
        let (mut sender_transport, mut receiver_transport) =
            crate::MemoryByteTransport::pair(4).expect("valid byte transport");
        assert_eq!(
            sender
                .flush_bytes(&mut sender_transport, 0)
                .expect("byte flush")
                .sent,
            1
        );
        assert!(matches!(
            receiver
                .receive_bytes(&mut receiver_transport)
                .expect("byte receive"),
            Some(ReceiveOutcome::Delivered { sequence: 1, .. })
        ));
        receiver
            .flush_bytes(&mut receiver_transport, 0)
            .expect("ack byte flush");
        assert!(matches!(
            sender
                .receive_bytes(&mut sender_transport)
                .expect("ack byte receive"),
            Some(ReceiveOutcome::Acknowledged {
                sequence: 1,
                released: 1
            })
        ));
        assert_eq!(sender.pending(), 0);
    }

    #[test]
    fn malformed_byte_frame_is_reported_without_advancing_receive_state() {
        let mut receiver = session(12, 8, 2, 10);
        receiver
            .authenticate(AuthToken(0xfeed))
            .expect("receiver auth");
        let (mut sender_transport, mut receiver_transport) =
            crate::MemoryByteTransport::pair(2).expect("valid byte transport");
        sender_transport
            .try_send_bytes(&[0, 1, 2])
            .expect("inject malformed frame");
        assert!(matches!(
            receiver.receive_bytes(&mut receiver_transport),
            Err(ByteReceiveError::Codec(
                crate::WireCodecError::UnexpectedEof
            ))
        ));
        assert_eq!(receiver.pending_ack, None);
        assert_eq!(receiver.next_receive_sequence, 1);
    }

    #[test]
    fn receive_window_rejects_far_future_packets_without_poisoning_retry() {
        let mut receiver = session(1, 8, 2, 10);
        receiver
            .authenticate(AuthToken(0xfeed))
            .expect("receiver auth");
        let far = TransportPacket::Data {
            session: SessionId(1),
            sequence: 4,
            message: WorldMessage::Subscribe(subscription(4, 7)),
        };
        assert_eq!(
            receiver.receive(far.clone()),
            Err(ReceiveError::OutsideReceiveWindow {
                expected: 1,
                received: 4,
                window: 2,
            })
        );
        assert_eq!(receiver.inbound.len(), 0);

        let in_window = TransportPacket::Data {
            session: SessionId(1),
            sequence: 2,
            message: WorldMessage::Subscribe(subscription(2, 7)),
        };
        assert_eq!(
            receiver.receive(in_window),
            Ok(ReceiveOutcome::Buffered { sequence: 2 })
        );
        assert_eq!(
            receiver.receive(TransportPacket::Data {
                session: SessionId(1),
                sequence: 3,
                message: WorldMessage::Subscribe(subscription(3, 7)),
            }),
            Ok(ReceiveOutcome::Buffered { sequence: 3 })
        );
        assert_eq!(receiver.inbound.len(), 2);
        assert_eq!(
            receiver.receive(far),
            Err(ReceiveError::OutsideReceiveWindow {
                expected: 1,
                received: 4,
                window: 2,
            })
        );
        assert_eq!(receiver.inbound.len(), 2);
        assert!(matches!(
            receiver.receive(TransportPacket::Data {
                session: SessionId(1),
                sequence: 1,
                message: WorldMessage::Subscribe(subscription(1, 7)),
            }),
            Ok(ReceiveOutcome::Delivered { sequence: 1, messages }) if messages.len() == 3
        ));
        assert_eq!(receiver.inbound.len(), 0);
    }

    #[test]
    fn outbound_and_transport_backpressure_are_bounded() {
        let mut sender = session(1, 7, 1, 1);
        let mut receiver = session(1, 8, 1, 1);
        sender.authenticate(AuthToken(0xfeed)).expect("sender auth");
        receiver
            .authenticate(AuthToken(0xfeed))
            .expect("receiver auth");
        sender
            .send(WorldMessage::Subscribe(subscription(1, 7)))
            .expect("first queue slot");
        assert_eq!(
            sender.send(WorldMessage::Subscribe(subscription(2, 7))),
            Err(SessionError::OutboundFull)
        );
        let (mut sender_transport, mut receiver_transport) =
            MemoryTransport::pair(1).expect("valid transport");
        sender
            .flush(&mut sender_transport, 0)
            .expect("transport has one slot");
        assert_eq!(receiver_transport.queued(), 1);
        receiver
            .receive(receiver_transport.try_recv().expect("data packet"))
            .expect("receiver accepts data");
        receiver
            .flush(&mut receiver_transport, 0)
            .expect("ack fits after receive drain");
        sender
            .receive(sender_transport.try_recv().expect("ack packet"))
            .expect("sender accepts ack");
        assert_eq!(sender.pending(), 0);
    }

    #[test]
    fn retry_resends_after_deadline_and_duplicate_is_not_delivered_twice() {
        let mut sender = session(1, 7, 4, 5);
        let mut receiver = session(1, 8, 4, 5);
        sender.authenticate(AuthToken(0xfeed)).expect("sender auth");
        receiver
            .authenticate(AuthToken(0xfeed))
            .expect("receiver auth");
        sender
            .send(WorldMessage::Subscribe(subscription(1, 7)))
            .expect("queue message");
        let (mut sender_transport, mut receiver_transport) =
            MemoryTransport::pair(8).expect("valid transport");
        assert_eq!(
            sender
                .flush(&mut sender_transport, 0)
                .expect("initial send")
                .retries,
            0
        );
        assert_eq!(
            sender.flush(&mut sender_transport, 4),
            Ok(FlushReport {
                sent: 0,
                retries: 0,
                pending: 1
            })
        );
        sender
            .flush(&mut sender_transport, 5)
            .expect("deadline retry");
        let original = receiver_transport.try_recv().expect("original data");
        let retry = receiver_transport.try_recv().expect("retry data");
        assert_eq!(
            receiver.receive(original),
            Ok(ReceiveOutcome::Delivered {
                sequence: 1,
                messages: vec![WorldMessage::Subscribe(subscription(1, 7))]
            })
        );
        assert_eq!(
            receiver.receive(retry),
            Ok(ReceiveOutcome::Duplicate { sequence: 1 })
        );
        receiver
            .flush(&mut receiver_transport, 5)
            .expect("ack retry");
        sender
            .receive(sender_transport.try_recv().expect("ack"))
            .expect("ack accepted");
        assert_eq!(sender.pending(), 0);
    }

    #[test]
    fn unauthenticated_receive_is_rejected_before_payload_validation() {
        let mut receiver = session(1, 8, 4, 1);
        let packet = TransportPacket::Data {
            session: SessionId(1),
            sequence: 1,
            message: WorldMessage::Subscribe(subscription(1, 7)),
        };
        assert_eq!(
            receiver.receive(packet),
            Err(ReceiveError::Session(SessionError::NotAuthenticated))
        );
    }
}
