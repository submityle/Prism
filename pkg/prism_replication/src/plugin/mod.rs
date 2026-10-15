//! The `prism_app` seam (design §25.2): drive replication sessions from the app's
//! frame loop.
//!
//! Replication is wired into the app shell as `prism_app` plugins:
//!
//! - [`ReplicationPlugin`] installs the **client** half — a [`Replication`]
//!   resource (a [`ClientSession`](crate::session::ClientSession) that predicts
//!   local input and reconciles authority) plus a `First` system that ticks its
//!   channel once per frame and drains the recovered frames into the prediction.
//! - [`ReplicationServerPlugin`] installs the **authority** half — a
//!   [`ServerReplication`] resource (a
//!   [`ServerSession`](crate::session::ServerSession) that decodes each inbound
//!   command, applies it, and broadcasts the framed delta) plus a `First` system
//!   that pumps it every frame.
//!
//! Both wrap the caller's [`Transport`](crate::transport::Transport) in a
//! [`ChannelMux`](crate::multiplex::ChannelMux). Commands and control messages use
//! the reliable RPC lane, authoritative deltas and events use an independent
//! reliable event lane, transient state uses the newest-wins unreliable state
//! lane. Full resync snapshots share the event lane so they remain ordered with
//! respect to authoritative deltas. Peers negotiate an exact protocol version
//! and the smaller rollback window before application traffic is accepted.
//!
//! A rollback window counts retained revisions including the acknowledged
//! baseline, so at most `window - 1` unconfirmed predictions are admitted. The
//! legacy client command API can queue behind the initial hello; new event/state
//! sends and server RPC responses require an established connection. Independent
//! lanes may arrive before the accept packet and are buffered until acceptance.
//! This changes the plugin wire format from the former single reliable channel;
//! both endpoints must upgrade together.

use alloc::vec::Vec;

use crate::multiplex::{MuxConfig, NegotiatedProtocol, NetChannel, ProtocolError, ProtocolOffer};
use crate::session::SessionError;
use crate::transport::Transport;

#[cfg(feature = "determinism")]
mod audit;
mod client;
mod ecs;
mod server;

#[cfg(feature = "determinism")]
pub use audit::{
    FrameHashAudit, FrameHashAuditError, FrameHashComparison, FrameHashReceipt, FrameHashUpdate,
};

pub use crate::security::{
    ClientHello, SecureTransport, SecurityDirection, SecurityError, ServerAccept, SessionNonce,
    SessionSecurityProvider,
};
pub use client::{
    AuthorityUpdate, EcsReplicationStrategy, NetworkRollbackSimulation, Replication,
    ReplicationPlugin, ReplicationStrategy,
};
pub use ecs::{EcsReplication, EcsReplicationError, EcsReplicationPlugin, NetworkRollbackDriver};
pub use server::{ReplicationServerPlugin, ServerReplication, ServerStrategy};

/// The retransmit floor (in frames) both plugins use for their channel.
pub(crate) const DEFAULT_MIN_RTO: u32 = 2;
/// How many out-of-order frames a plugin's channel buffers before it refuses.
pub(crate) const DEFAULT_WINDOW: usize = 64;

/// App-plugin transport and handshake configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReplicationConfig {
    /// Datagram multiplexing, queue, retransmit, and congestion limits.
    pub mux: MuxConfig,
    /// Protocol and local rollback capacity offered to the peer. Constructors
    /// clamp the offer to the actual snapshot-ring capacity, including baseline.
    pub protocol: ProtocolOffer,
}

impl Default for ReplicationConfig {
    fn default() -> Self {
        Self {
            mux: MuxConfig {
                min_rto: DEFAULT_MIN_RTO,
                reliable_window: DEFAULT_WINDOW,
                ..MuxConfig::default()
            },
            protocol: ProtocolOffer::current(DEFAULT_WINDOW as u32),
        }
    }
}

impl ReplicationConfig {
    pub(super) fn for_session(mut self, capacity: usize) -> Self {
        self.protocol.rollback_window = self
            .protocol
            .rollback_window
            .min(capacity_to_window(capacity));
        self
    }

    /// Default transport limits with a rollback offer derived from a session's
    /// snapshot capacity.
    #[must_use]
    pub fn for_capacity(capacity: usize) -> Self {
        Self {
            protocol: ProtocolOffer::current(capacity_to_window(capacity)),
            ..Self::default()
        }
    }
}

/// Observable connection handshake state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConnectionState {
    /// The client offer or server response is still in flight.
    Handshaking,
    /// Both peers accepted this exact contract.
    Established(NegotiatedProtocol),
    /// The peer was rejected or violated the handshake/envelope contract.
    Rejected(ProtocolError),
}

impl ConnectionState {
    /// Negotiated contract after a successful handshake.
    #[must_use]
    pub const fn negotiated(self) -> Option<NegotiatedProtocol> {
        match self {
            Self::Established(protocol) => Some(protocol),
            Self::Handshaking | Self::Rejected(_) => None,
        }
    }
}

const RPC_HELLO: u8 = 0;
const RPC_ACCEPT: u8 = 1;
const RPC_REJECT: u8 = 2;
const RPC_PAYLOAD: u8 = 3;

const EVENT_DELTA: u8 = 0;
const EVENT_PAYLOAD: u8 = 1;
const EVENT_SNAPSHOT: u8 = 2;
#[cfg(feature = "determinism")]
const EVENT_FRAME_HASH: u8 = 3;
const STATE_PAYLOAD: u8 = 0;

pub(super) enum RpcEnvelope<'a> {
    Hello(ProtocolOffer),
    Accept(NegotiatedProtocol),
    Reject(ProtocolError),
    Payload(&'a [u8]),
}

pub(super) enum EventEnvelope<'a> {
    Delta(&'a [u8]),
    Payload(&'a [u8]),
    Snapshot(&'a [u8]),
    #[cfg(feature = "determinism")]
    FrameHash(FrameHashUpdate),
}

pub(super) fn encode_hello(offer: ProtocolOffer) -> Vec<u8> {
    encode_contract(RPC_HELLO, offer.version, offer.rollback_window)
}

pub(super) fn encode_accept(protocol: NegotiatedProtocol) -> Vec<u8> {
    encode_contract(RPC_ACCEPT, protocol.version, protocol.rollback_window)
}

pub(super) fn encode_reject(error: ProtocolError) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(10);
    bytes.push(RPC_REJECT);
    match error {
        ProtocolError::VersionMismatch { local, remote } => {
            bytes.push(0);
            bytes.extend_from_slice(&local.to_le_bytes());
            bytes.extend_from_slice(&remote.to_le_bytes());
        }
        ProtocolError::InvalidRollbackWindow { offered } => {
            bytes.push(1);
            bytes.extend_from_slice(&offered.to_le_bytes());
        }
        ProtocolError::MalformedMessage { .. }
        | ProtocolError::TrafficBeforeHandshake { .. }
        | ProtocolError::UnexpectedHandshake => {
            bytes.push(2);
        }
        ProtocolError::SecurityFailure => bytes.push(3),
    }
    bytes
}

pub(super) fn encode_rpc(payload: &[u8]) -> Vec<u8> {
    encode_payload(RPC_PAYLOAD, payload)
}

pub(super) fn decode_rpc(bytes: &[u8]) -> Result<RpcEnvelope<'_>, ProtocolError> {
    let Some((&tag, payload)) = bytes.split_first() else {
        return Err(malformed(NetChannel::RpcReliable));
    };
    match tag {
        RPC_HELLO | RPC_ACCEPT if payload.len() == 6 => {
            let version = u16::from_le_bytes([payload[0], payload[1]]);
            let rollback_window =
                u32::from_le_bytes([payload[2], payload[3], payload[4], payload[5]]);
            if tag == RPC_HELLO {
                Ok(RpcEnvelope::Hello(ProtocolOffer {
                    version,
                    rollback_window,
                }))
            } else {
                Ok(RpcEnvelope::Accept(NegotiatedProtocol {
                    version,
                    rollback_window,
                }))
            }
        }
        RPC_REJECT => {
            let Some((&code, detail)) = payload.split_first() else {
                return Err(malformed(NetChannel::RpcReliable));
            };
            let error = match code {
                0 if detail.len() == 4 => ProtocolError::VersionMismatch {
                    local: u16::from_le_bytes([detail[0], detail[1]]),
                    remote: u16::from_le_bytes([detail[2], detail[3]]),
                },
                1 if detail.len() == 4 => ProtocolError::InvalidRollbackWindow {
                    offered: u32::from_le_bytes([detail[0], detail[1], detail[2], detail[3]]),
                },
                2 if detail.is_empty() => ProtocolError::UnexpectedHandshake,
                3 if detail.is_empty() => ProtocolError::SecurityFailure,
                _ => return Err(malformed(NetChannel::RpcReliable)),
            };
            Ok(RpcEnvelope::Reject(error))
        }
        RPC_PAYLOAD => Ok(RpcEnvelope::Payload(payload)),
        _ => Err(malformed(NetChannel::RpcReliable)),
    }
}

pub(super) fn encode_delta(payload: &[u8]) -> Vec<u8> {
    encode_payload(EVENT_DELTA, payload)
}

pub(super) fn encode_event(payload: &[u8]) -> Vec<u8> {
    encode_payload(EVENT_PAYLOAD, payload)
}

pub(super) fn decode_event(bytes: &[u8]) -> Result<EventEnvelope<'_>, ProtocolError> {
    let Some((&tag, payload)) = bytes.split_first() else {
        return Err(malformed(NetChannel::EventsReliable));
    };
    match tag {
        EVENT_DELTA | EVENT_SNAPSHOT => {
            // Plugin messages contain exactly one complete session frame. A
            // truncated or concatenated frame must not silently stall prediction
            // or advance multiple revisions from one admitted command.
            let mut decoder = crate::frame::FrameDecoder::new();
            decoder.feed(payload);
            if !matches!(decoder.next_frame(), Ok(Some(_))) || decoder.buffered() != 0 {
                return Err(malformed(NetChannel::EventsReliable));
            }
            if tag == EVENT_DELTA {
                Ok(EventEnvelope::Delta(payload))
            } else {
                Ok(EventEnvelope::Snapshot(payload))
            }
        }
        EVENT_PAYLOAD => Ok(EventEnvelope::Payload(payload)),
        #[cfg(feature = "determinism")]
        EVENT_FRAME_HASH => FrameHashUpdate::decode(payload)
            .map(EventEnvelope::FrameHash)
            .ok_or_else(|| malformed(NetChannel::EventsReliable)),
        _ => Err(malformed(NetChannel::EventsReliable)),
    }
}

#[cfg(feature = "determinism")]
pub(super) fn encode_frame_hash(update: FrameHashUpdate) -> Vec<u8> {
    encode_payload(EVENT_FRAME_HASH, &update.encode())
}

pub(super) fn encode_state(payload: &[u8]) -> Vec<u8> {
    encode_payload(STATE_PAYLOAD, payload)
}

pub(super) fn decode_state(bytes: &[u8]) -> Result<&[u8], ProtocolError> {
    decode_payload(bytes, STATE_PAYLOAD, NetChannel::StateUnreliable)
}

pub(super) fn encode_snapshot(payload: &[u8]) -> Vec<u8> {
    encode_payload(EVENT_SNAPSHOT, payload)
}

fn encode_contract(tag: u8, version: u16, rollback_window: u32) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(7);
    bytes.push(tag);
    bytes.extend_from_slice(&version.to_le_bytes());
    bytes.extend_from_slice(&rollback_window.to_le_bytes());
    bytes
}

fn encode_payload(tag: u8, payload: &[u8]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(payload.len().saturating_add(1));
    bytes.push(tag);
    bytes.extend_from_slice(payload);
    bytes
}

fn decode_payload(
    bytes: &[u8],
    expected_tag: u8,
    channel: NetChannel,
) -> Result<&[u8], ProtocolError> {
    let Some((&tag, payload)) = bytes.split_first() else {
        return Err(malformed(channel));
    };
    if tag != expected_tag {
        return Err(malformed(channel));
    }
    Ok(payload)
}

pub(super) const fn malformed(channel: NetChannel) -> ProtocolError {
    ProtocolError::MalformedMessage { channel }
}

pub(super) fn require_secure_transport(transport: &dyn Transport) -> Result<(), SessionError> {
    if transport.supports_secure_session() {
        Ok(())
    } else {
        Err(SessionError::Security(SecurityError::ProviderRequired))
    }
}

fn capacity_to_window(capacity: usize) -> u32 {
    u32::try_from(capacity.max(1)).unwrap_or(u32::MAX)
}

#[cfg(all(test, not(feature = "determinism")))]
#[test]
fn unaudited_contract_keeps_v1_and_rejects_hash_envelopes() {
    assert_eq!(crate::multiplex::CURRENT_PROTOCOL_VERSION, 1);
    assert!(matches!(
        decode_event(&[3; 25]),
        Err(ProtocolError::MalformedMessage {
            channel: NetChannel::EventsReliable
        })
    ));
}
