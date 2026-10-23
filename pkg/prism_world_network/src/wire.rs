//! Bounded little-endian codec for the transport packet seam.
//!
//! Application values remain owned by the session/persistence contracts. This
//! module only supplies the host byte-channel frame: fixed packet metadata,
//! canonical subscription fields, and the versioned persistence transaction
//! codec for delta messages.

use alloc::collections::VecDeque;
use alloc::rc::Rc;
use alloc::vec::Vec;
use core::cell::RefCell;
use prism_world_persist::{decode_transaction, encode_transaction, ActorId, CodecError};

use crate::{
    AabbI64, AoiSubscription, DeltaEnvelope, GlobalPos, SectorId, SessionId, SubscriptionId,
    TransportPacket, WorldMessage,
};

const MAGIC: [u8; 4] = *b"PWN1";
const VERSION: u8 = 1;
const KIND_DELTA: u8 = 1;
const KIND_SUBSCRIBE: u8 = 2;
const KIND_UNSUBSCRIBE: u8 = 3;
const KIND_ACK: u8 = 4;
const HEADER_BYTES: usize = 4 + 1 + 16 + 8 + 1 + 4;
const MAX_PAYLOAD: usize = 16 * 1024 * 1024;

/// Maximum encoded transport payload accepted by this codec.
pub const MAX_TRANSPORT_PAYLOAD: usize = MAX_PAYLOAD;

/// Result of asking a stream decoder for its next complete frame.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FrameDecodeStatus {
    /// More bytes are required before a complete frame is available.
    NeedMore,
    /// One complete, framing-validated `PWN1` frame.
    Frame(Vec<u8>),
}

/// Errors raised while assembling bounded `PWN1` frames from a byte stream.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FrameDecoderError {
    /// The configured per-frame and cumulative buffering limit is invalid.
    InvalidMaxFrameBytes(usize),
    /// The cumulative retained stream would exceed the configured bound.
    BufferLimitExceeded {
        /// Bytes already retained by the decoder.
        buffered: usize,
        /// Bytes supplied by the rejected chunk.
        incoming: usize,
        /// Maximum retained bytes.
        limit: usize,
    },
    /// The frame marker is invalid.
    InvalidMagic,
    /// The frame version is not supported.
    UnsupportedVersion(u8),
    /// The declared payload exceeds the global codec limit.
    PayloadTooLarge(u32),
    /// The declared frame cannot fit within the decoder's per-frame limit.
    FrameTooLarge {
        /// Declared total frame length.
        frame_bytes: usize,
        /// Configured maximum total frame length.
        limit: usize,
    },
    /// An allocation failed before the chunk or frame was published.
    AllocationFailed,
}

/// A bounded framing decoder for socket, QUIC-stream, and other fragmented byte sources.
///
/// `max_frame_bytes` is both the maximum total size of one frame and the maximum
/// number of unconsumed bytes retained by this decoder. Callers must drain frames
/// with [`Self::next_frame`] before feeding more bytes beyond that budget. Framing
/// validation happens before `push` mutates the retained stream, so a malformed
/// chunk leaves the previous state intact.
pub struct ByteFrameDecoder {
    buffer: Vec<u8>,
    max_frame_bytes: usize,
}

impl ByteFrameDecoder {
    /// Creates a decoder with a bounded total-frame and retained-stream budget.
    pub fn new(max_frame_bytes: usize) -> Result<Self, FrameDecoderError> {
        if !(HEADER_BYTES..=HEADER_BYTES.saturating_add(MAX_PAYLOAD)).contains(&max_frame_bytes) {
            return Err(FrameDecoderError::InvalidMaxFrameBytes(max_frame_bytes));
        }
        Ok(Self {
            buffer: Vec::new(),
            max_frame_bytes,
        })
    }

    /// Adds one arbitrary stream chunk. Empty chunks are accepted without effect.
    pub fn push(&mut self, chunk: &[u8]) -> Result<(), FrameDecoderError> {
        if chunk.is_empty() {
            return Ok(());
        }
        let buffered = self.buffer.len();
        let retained =
            buffered
                .checked_add(chunk.len())
                .ok_or(FrameDecoderError::BufferLimitExceeded {
                    buffered,
                    incoming: chunk.len(),
                    limit: self.max_frame_bytes,
                })?;
        if retained > self.max_frame_bytes {
            return Err(FrameDecoderError::BufferLimitExceeded {
                buffered,
                incoming: chunk.len(),
                limit: self.max_frame_bytes,
            });
        }
        validate_stream(&self.buffer, chunk, self.max_frame_bytes)?;
        self.buffer
            .try_reserve_exact(chunk.len())
            .map_err(|_| FrameDecoderError::AllocationFailed)?;
        self.buffer.extend_from_slice(chunk);
        Ok(())
    }

    /// Returns the next complete frame, retaining any later frames or partial bytes.
    pub fn next_frame(&mut self) -> Result<FrameDecodeStatus, FrameDecoderError> {
        if self.buffer.len() < HEADER_BYTES {
            return Ok(FrameDecodeStatus::NeedMore);
        }
        let frame_bytes = validate_header(&self.buffer, self.max_frame_bytes)?;
        if self.buffer.len() < frame_bytes {
            return Ok(FrameDecodeStatus::NeedMore);
        }
        let mut frame = Vec::new();
        frame
            .try_reserve_exact(frame_bytes)
            .map_err(|_| FrameDecoderError::AllocationFailed)?;
        frame.extend_from_slice(&self.buffer[..frame_bytes]);
        self.buffer.copy_within(frame_bytes.., 0);
        self.buffer.truncate(self.buffer.len() - frame_bytes);
        Ok(FrameDecodeStatus::Frame(frame))
    }

    /// Returns the number of bytes retained but not yet returned as a frame.
    pub const fn buffered_len(&self) -> usize {
        self.buffer.len()
    }
}

fn validate_stream(
    buffered: &[u8],
    incoming: &[u8],
    max_frame_bytes: usize,
) -> Result<(), FrameDecoderError> {
    let total = buffered.len().saturating_add(incoming.len());
    let byte = |index: usize| {
        if index < buffered.len() {
            buffered[index]
        } else {
            incoming[index - buffered.len()]
        }
    };
    let mut offset = 0;
    while total.saturating_sub(offset) >= 4 {
        for (index, expected) in MAGIC.iter().enumerate() {
            if byte(offset + index) != *expected {
                return Err(FrameDecoderError::InvalidMagic);
            }
        }
        if total - offset < HEADER_BYTES {
            if total - offset > 4 && byte(offset + 4) != VERSION {
                return Err(FrameDecoderError::UnsupportedVersion(byte(offset + 4)));
            }
            return Ok(());
        }
        if byte(offset + 4) != VERSION {
            return Err(FrameDecoderError::UnsupportedVersion(byte(offset + 4)));
        }
        let mut length_bytes = [0_u8; 4];
        for (index, value) in length_bytes.iter_mut().enumerate() {
            *value = byte(offset + 30 + index);
        }
        let payload = u32::from_le_bytes(length_bytes);
        let payload_usize = payload as usize;
        if payload_usize > MAX_PAYLOAD {
            return Err(FrameDecoderError::PayloadTooLarge(payload));
        }
        let frame_bytes =
            HEADER_BYTES
                .checked_add(payload_usize)
                .ok_or(FrameDecoderError::FrameTooLarge {
                    frame_bytes: usize::MAX,
                    limit: max_frame_bytes,
                })?;
        if frame_bytes > max_frame_bytes {
            return Err(FrameDecoderError::FrameTooLarge {
                frame_bytes,
                limit: max_frame_bytes,
            });
        }
        if total - offset < frame_bytes {
            return Ok(());
        }
        offset += frame_bytes;
    }
    if total > offset {
        for (index, expected) in MAGIC.iter().enumerate().take(total - offset) {
            if byte(offset + index) != *expected {
                return Err(FrameDecoderError::InvalidMagic);
            }
        }
    }
    Ok(())
}

fn validate_header(bytes: &[u8], max_frame_bytes: usize) -> Result<usize, FrameDecoderError> {
    debug_assert!(bytes.len() >= HEADER_BYTES);
    if bytes[..4] != MAGIC {
        return Err(FrameDecoderError::InvalidMagic);
    }
    if bytes[4] != VERSION {
        return Err(FrameDecoderError::UnsupportedVersion(bytes[4]));
    }
    let payload = u32::from_le_bytes(bytes[30..34].try_into().unwrap());
    let payload_usize = payload as usize;
    if payload_usize > MAX_PAYLOAD {
        return Err(FrameDecoderError::PayloadTooLarge(payload));
    }
    let frame_bytes =
        HEADER_BYTES
            .checked_add(payload_usize)
            .ok_or(FrameDecoderError::FrameTooLarge {
                frame_bytes: usize::MAX,
                limit: max_frame_bytes,
            })?;
    if frame_bytes > max_frame_bytes {
        return Err(FrameDecoderError::FrameTooLarge {
            frame_bytes,
            limit: max_frame_bytes,
        });
    }
    Ok(frame_bytes)
}

/// A non-blocking host byte-channel adapter.
///
/// The adapter owns framing boundaries: one successful send hands one complete
/// `PWN1` frame to the host, and one receive returns one complete frame. A real
/// stream transport must perform its own bounded frame assembly before
/// returning from `try_recv_bytes`.
pub trait ByteTransport {
    /// Adapter-specific I/O error.
    type Error;

    /// Attempts to enqueue one complete frame. Backpressure must not consume a
    /// session retry attempt.
    fn try_send_bytes(&mut self, frame: &[u8]) -> Result<(), Self::Error>;

    /// Removes one complete frame, if available.
    fn try_recv_bytes(&mut self) -> Result<Option<Vec<u8>>, Self::Error>;
}

/// Errors from the bounded in-memory byte transport.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryByteTransportError {
    /// The configured queue cannot have zero capacity.
    InvalidCapacity,
    /// The peer's receive queue is full.
    Backpressure,
}

type ByteQueue = Rc<RefCell<VecDeque<Vec<u8>>>>;

/// Connected bounded byte endpoints for deterministic host integration tests.
pub struct MemoryByteTransport {
    inbound: ByteQueue,
    outbound: ByteQueue,
    capacity: usize,
}

impl MemoryByteTransport {
    /// Creates two connected bounded endpoints.
    pub fn pair(capacity: usize) -> Result<(Self, Self), MemoryByteTransportError> {
        if capacity == 0 {
            return Err(MemoryByteTransportError::InvalidCapacity);
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

    /// Returns the number of complete frames waiting at this endpoint.
    pub fn queued(&self) -> usize {
        self.inbound.borrow().len()
    }
}

impl ByteTransport for MemoryByteTransport {
    type Error = MemoryByteTransportError;

    fn try_send_bytes(&mut self, frame: &[u8]) -> Result<(), Self::Error> {
        let mut outbound = self.outbound.borrow_mut();
        if outbound.len() >= self.capacity {
            return Err(MemoryByteTransportError::Backpressure);
        }
        let mut owned = Vec::new();
        owned
            .try_reserve_exact(frame.len())
            .map_err(|_| MemoryByteTransportError::Backpressure)?;
        owned.extend_from_slice(frame);
        outbound.push_back(owned);
        Ok(())
    }

    fn try_recv_bytes(&mut self) -> Result<Option<Vec<u8>>, Self::Error> {
        Ok(self.inbound.borrow_mut().pop_front())
    }
}

/// Errors raised by the transport packet wire codec.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum WireCodecError {
    /// The frame does not contain the requested bytes.
    UnexpectedEof,
    /// The frame marker is invalid.
    InvalidMagic,
    /// The frame version is not supported.
    UnsupportedVersion(u8),
    /// The application message tag is not supported.
    InvalidMessageKind(u8),
    /// The payload exceeds the bounded codec limit.
    PayloadTooLarge(u32),
    /// The declared payload length does not match the frame.
    InvalidLength,
    /// The session sequence is reserved.
    InvalidSequence(u64),
    /// A persist transaction codec rejected the delta payload.
    Persistence(CodecError),
    /// An allocation failed before any output was published.
    AllocationFailed,
}

/// Encodes one transport packet into a versioned bounded frame.
pub fn encode_packet(packet: &TransportPacket) -> Result<Vec<u8>, WireCodecError> {
    let (kind, payload) = match packet {
        TransportPacket::Data { message, .. } => match message {
            WorldMessage::Delta(envelope) => {
                let transaction = encode_transaction(envelope.transaction())
                    .map_err(WireCodecError::Persistence)?;
                let mut payload = Vec::new();
                payload
                    .try_reserve_exact(25 + transaction.len())
                    .map_err(|_| WireCodecError::AllocationFailed)?;
                payload.push(envelope.sector().level);
                for value in [
                    envelope.sector().x,
                    envelope.sector().y,
                    envelope.sector().z,
                ] {
                    payload.extend_from_slice(&value.to_le_bytes());
                }
                payload.extend_from_slice(&transaction);
                (KIND_DELTA, payload)
            }
            WorldMessage::Subscribe(subscription) => {
                let mut payload = Vec::new();
                payload
                    .try_reserve_exact(16 + 16 + 48)
                    .map_err(|_| WireCodecError::AllocationFailed)?;
                payload.extend_from_slice(&subscription.id().0.to_le_bytes());
                payload.extend_from_slice(&subscription.subscriber().0.to_le_bytes());
                for value in [subscription.bounds().min, subscription.bounds().max] {
                    payload.extend_from_slice(&value.x.to_le_bytes());
                    payload.extend_from_slice(&value.y.to_le_bytes());
                    payload.extend_from_slice(&value.z.to_le_bytes());
                }
                (KIND_SUBSCRIBE, payload)
            }
            WorldMessage::Unsubscribe(id) => (KIND_UNSUBSCRIBE, id.0.to_le_bytes().to_vec()),
        },
        TransportPacket::Ack { .. } => (KIND_ACK, Vec::new()),
    };
    let (session, sequence) = match packet {
        TransportPacket::Data {
            session, sequence, ..
        } => (*session, *sequence),
        TransportPacket::Ack { session, sequence } => (*session, *sequence),
    };
    if sequence == 0 || sequence == u64::MAX {
        return Err(WireCodecError::InvalidSequence(sequence));
    }
    if payload.len() > MAX_PAYLOAD || payload.len() > u32::MAX as usize {
        return Err(WireCodecError::PayloadTooLarge(
            payload.len().min(u32::MAX as usize) as u32,
        ));
    }
    let mut frame = Vec::new();
    frame
        .try_reserve_exact(HEADER_BYTES + payload.len())
        .map_err(|_| WireCodecError::AllocationFailed)?;
    frame.extend_from_slice(&MAGIC);
    frame.push(VERSION);
    frame.extend_from_slice(&session.0.to_le_bytes());
    frame.extend_from_slice(&sequence.to_le_bytes());
    frame.push(kind);
    frame.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    frame.extend_from_slice(&payload);
    Ok(frame)
}

/// Decodes one complete frame and validates all lengths before allocation.
pub fn decode_packet(bytes: &[u8]) -> Result<TransportPacket, WireCodecError> {
    if bytes.len() < HEADER_BYTES {
        return Err(WireCodecError::UnexpectedEof);
    }
    if bytes[..4] != MAGIC {
        return Err(WireCodecError::InvalidMagic);
    }
    if bytes[4] != VERSION {
        return Err(WireCodecError::UnsupportedVersion(bytes[4]));
    }
    let session = SessionId(u128::from_le_bytes(bytes[5..21].try_into().unwrap()));
    let sequence = u64::from_le_bytes(bytes[21..29].try_into().unwrap());
    if sequence == 0 || sequence == u64::MAX {
        return Err(WireCodecError::InvalidSequence(sequence));
    }
    let kind = bytes[29];
    let length = u32::from_le_bytes(bytes[30..34].try_into().unwrap()) as usize;
    if length > MAX_PAYLOAD {
        return Err(WireCodecError::PayloadTooLarge(length as u32));
    }
    let end = HEADER_BYTES
        .checked_add(length)
        .ok_or(WireCodecError::InvalidLength)?;
    if bytes.len() != end {
        return Err(WireCodecError::InvalidLength);
    }
    let payload = &bytes[HEADER_BYTES..end];
    let message = match kind {
        KIND_DELTA if payload.len() >= 25 => {
            let sector = SectorId::new(
                payload[0],
                i64::from_le_bytes(payload[1..9].try_into().unwrap()),
                i64::from_le_bytes(payload[9..17].try_into().unwrap()),
                i64::from_le_bytes(payload[17..25].try_into().unwrap()),
            );
            WorldMessage::Delta(DeltaEnvelope::new(
                sector,
                decode_transaction(&payload[25..]).map_err(WireCodecError::Persistence)?,
            ))
        }
        KIND_DELTA => return Err(WireCodecError::InvalidLength),
        KIND_SUBSCRIBE => WorldMessage::Subscribe(decode_subscription(payload)?),
        KIND_UNSUBSCRIBE if payload.len() == 16 => WorldMessage::Unsubscribe(SubscriptionId(
            u128::from_le_bytes(payload.try_into().unwrap()),
        )),
        KIND_UNSUBSCRIBE => return Err(WireCodecError::InvalidLength),
        KIND_ACK if payload.is_empty() => {
            return Ok(TransportPacket::Ack { session, sequence });
        }
        KIND_ACK => return Err(WireCodecError::InvalidLength),
        other => return Err(WireCodecError::InvalidMessageKind(other)),
    };
    Ok(TransportPacket::Data {
        session,
        sequence,
        message,
    })
}

fn decode_subscription(payload: &[u8]) -> Result<AoiSubscription, WireCodecError> {
    if payload.len() != 80 {
        return Err(WireCodecError::InvalidLength);
    }
    let id = SubscriptionId(u128::from_le_bytes(payload[..16].try_into().unwrap()));
    let subscriber = ActorId(u128::from_le_bytes(payload[16..32].try_into().unwrap()));
    let mut values = [0_i64; 6];
    for (index, value) in values.iter_mut().enumerate() {
        let start = 32 + index * 8;
        *value = i64::from_le_bytes(payload[start..start + 8].try_into().unwrap());
    }
    let min = GlobalPos::new(values[0], values[1], values[2]);
    let max = GlobalPos::new(values[3], values[4], values[5]);
    let bounds = AabbI64::new(min, max).ok_or(WireCodecError::InvalidLength)?;
    Ok(AoiSubscription::new(id, subscriber, bounds))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DeltaEnvelope, Transaction, TransactionId};
    use prism_world_persist::{LogicalClock, OverrideDelta, OverrideValue, PersistentId};

    #[test]
    fn subscription_round_trip_preserves_negative_world_bounds() {
        let packet = TransportPacket::Data {
            session: SessionId(7),
            sequence: 2,
            message: WorldMessage::Subscribe(AoiSubscription::new(
                SubscriptionId(4),
                ActorId(5),
                AabbI64::new(GlobalPos::new(-9, 2, -3), GlobalPos::new(8, 4, 7)).unwrap(),
            )),
        };
        let decoded = decode_packet(&encode_packet(&packet).unwrap()).unwrap();
        assert_eq!(decoded, packet);
    }

    #[test]
    fn delta_round_trip_uses_persistence_codec_and_preserves_sector_context() {
        let delta =
            OverrideDelta::single(prism_world_persist::FieldId(3), OverrideValue::Integer(8))
                .unwrap();
        let tx = Transaction::new(
            TransactionId(1),
            ActorId(2),
            LogicalClock(3),
            4,
            PersistentId(5),
            delta.clone(),
            delta,
        )
        .unwrap();
        let packet = TransportPacket::Data {
            session: SessionId(7),
            sequence: 2,
            message: WorldMessage::Delta(DeltaEnvelope::new(SectorId::new(4, -2, 3, 5), tx)),
        };
        let decoded = decode_packet(&encode_packet(&packet).unwrap()).unwrap();
        assert_eq!(decoded, packet);
    }

    #[test]
    fn malformed_header_length_and_sequence_fail_closed() {
        let packet = TransportPacket::Data {
            session: SessionId(1),
            sequence: 1,
            message: WorldMessage::Unsubscribe(SubscriptionId(2)),
        };
        let mut frame = encode_packet(&packet).unwrap();
        frame[0] ^= 1;
        assert_eq!(decode_packet(&frame), Err(WireCodecError::InvalidMagic));
        let mut frame = encode_packet(&packet).unwrap();
        frame[30] = 0xff;
        assert_eq!(decode_packet(&frame), Err(WireCodecError::InvalidLength));
        let ack = TransportPacket::Ack {
            session: SessionId(1),
            sequence: 1,
        };
        assert_eq!(decode_packet(&encode_packet(&ack).unwrap()).unwrap(), ack);
    }

    fn unsubscribe_packet(sequence: u64, id: u128) -> TransportPacket {
        TransportPacket::Data {
            session: SessionId(9),
            sequence,
            message: WorldMessage::Unsubscribe(SubscriptionId(id)),
        }
    }

    #[test]
    fn frame_decoder_handles_header_and_payload_fragmentation() {
        let frame = encode_packet(&unsubscribe_packet(1, 11)).unwrap();
        let mut decoder = ByteFrameDecoder::new(frame.len()).unwrap();
        decoder.push(&frame[..2]).unwrap();
        assert_eq!(decoder.next_frame().unwrap(), FrameDecodeStatus::NeedMore);
        decoder.push(&frame[2..HEADER_BYTES]).unwrap();
        assert_eq!(decoder.next_frame().unwrap(), FrameDecodeStatus::NeedMore);
        decoder.push(&frame[HEADER_BYTES..]).unwrap();
        let decoded = match decoder.next_frame().unwrap() {
            FrameDecodeStatus::Frame(frame) => decode_packet(&frame).unwrap(),
            FrameDecodeStatus::NeedMore => panic!("complete frame expected"),
        };
        assert_eq!(decoded, unsubscribe_packet(1, 11));
        assert_eq!(decoder.buffered_len(), 0);
    }

    #[test]
    fn frame_decoder_handles_multiple_frames_and_partial_tail() {
        let first = encode_packet(&unsubscribe_packet(1, 11)).unwrap();
        let second = encode_packet(&unsubscribe_packet(2, 12)).unwrap();
        let mut decoder = ByteFrameDecoder::new(first.len() + second.len()).unwrap();
        let split = second.len() / 2;
        let mut chunk = first.clone();
        chunk.extend_from_slice(&second[..split]);
        decoder.push(&chunk).unwrap();
        assert_eq!(
            decoder.next_frame().unwrap(),
            FrameDecodeStatus::Frame(first)
        );
        assert_eq!(decoder.next_frame().unwrap(), FrameDecodeStatus::NeedMore);
        decoder.push(&second[split..]).unwrap();
        assert_eq!(
            decoder.next_frame().unwrap(),
            FrameDecodeStatus::Frame(second)
        );
        assert_eq!(decoder.next_frame().unwrap(), FrameDecodeStatus::NeedMore);
    }

    #[test]
    fn frame_decoder_rejects_bad_header_without_poisoning_state() {
        let frame = encode_packet(&unsubscribe_packet(1, 11)).unwrap();
        let mut decoder = ByteFrameDecoder::new(frame.len()).unwrap();
        decoder.push(&frame[..3]).unwrap();
        assert_eq!(decoder.push(b"X"), Err(FrameDecoderError::InvalidMagic));
        assert_eq!(decoder.buffered_len(), 3);
        decoder.push(&frame[3..]).unwrap();
        assert_eq!(
            decoder.next_frame().unwrap(),
            FrameDecodeStatus::Frame(frame)
        );
    }

    #[test]
    fn frame_decoder_rejects_version_and_declared_length_before_completion() {
        let frame = encode_packet(&unsubscribe_packet(1, 11)).unwrap();
        let mut version = frame[..5].to_vec();
        version[4] = 2;
        let mut decoder = ByteFrameDecoder::new(frame.len()).unwrap();
        assert_eq!(
            decoder.push(&version),
            Err(FrameDecoderError::UnsupportedVersion(2))
        );
        assert_eq!(decoder.buffered_len(), 0);

        let mut length = frame[..HEADER_BYTES].to_vec();
        length[30..34].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(
            decoder.push(&length),
            Err(FrameDecoderError::PayloadTooLarge(u32::MAX))
        );
        assert_eq!(decoder.buffered_len(), 0);
    }

    #[test]
    fn frame_decoder_enforces_capacity_and_accepts_exact_limit() {
        let frame = encode_packet(&unsubscribe_packet(1, 11)).unwrap();
        assert!(matches!(
            ByteFrameDecoder::new(HEADER_BYTES - 1),
            Err(FrameDecoderError::InvalidMaxFrameBytes(value)) if value == HEADER_BYTES - 1
        ));
        let mut decoder = ByteFrameDecoder::new(frame.len()).unwrap();
        decoder.push(&frame).unwrap();
        assert_eq!(decoder.buffered_len(), frame.len());
        let extra = [0_u8];
        assert!(matches!(
            decoder.push(&extra),
            Err(FrameDecoderError::BufferLimitExceeded { .. })
        ));
        assert_eq!(
            decoder.next_frame().unwrap(),
            FrameDecodeStatus::Frame(frame)
        );
        decoder.push(&[]).unwrap();
        assert_eq!(decoder.next_frame().unwrap(), FrameDecodeStatus::NeedMore);
    }
}
