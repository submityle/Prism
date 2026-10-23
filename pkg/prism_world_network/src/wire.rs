//! Bounded little-endian codec for the transport packet seam.
//!
//! Application values remain owned by the session/persistence contracts. This
//! module only supplies the host byte-channel frame: fixed packet metadata,
//! canonical subscription fields, and the versioned persistence transaction
//! codec for delta messages.

use alloc::vec::Vec;
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
}
