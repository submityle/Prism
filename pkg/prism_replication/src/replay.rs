//! Deterministic recording and playback of logical replication events.
//!
//! This is deliberately a transport-independent replay core.  It records the
//! logical event envelope after protocol/authentication handling, not encrypted
//! UDP bytes, so the same stream can be fed back into a session or authority
//! harness.  Durable storage remains a provider concern; the bounded codec is
//! suitable as the payload passed to `prism_world_persist::EventLogBackend`.
//! The optional `state_hash` is a semantic authority hash, not a checksum of
//! the encoded bytes; byte-stream integrity is deliberately delegated to the
//! durable backend or transport framing.

use alloc::vec::Vec;

use crate::multiplex::NetChannel;

/// Four-byte replay stream identifier.
pub const REPLAY_MAGIC: [u8; 4] = *b"PRRP";
/// Current replay stream version.
pub const REPLAY_VERSION: u8 = 1;
/// Default maximum logical event payload accepted by the codec.
pub const DEFAULT_MAX_PAYLOAD: usize = 16 * 1024 * 1024;
const HEADER_LEN: usize = 10;
const RECORD_HEADER_LEN: usize = 24;

/// Direction of a logical event in a recorded session.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum ReplayDirection {
    /// Client input or proposal delivered to authority.
    ClientToServer = 0,
    /// Authoritative event or correction delivered to a client.
    ServerToClient = 1,
}

impl ReplayDirection {
    fn from_wire(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::ClientToServer),
            1 => Some(Self::ServerToClient),
            _ => None,
        }
    }
}

/// One logical replication event.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReplayRecord {
    /// Fixed simulation tick at which the event was observed.
    pub tick: u64,
    /// Channel-local sequence, when the channel provides one.
    pub sequence: u64,
    /// Logical replication lane carrying the event.
    pub channel: NetChannel,
    /// Direction relative to the authority.
    pub direction: ReplayDirection,
    /// Optional authority/state hash captured alongside the event.
    pub state_hash: Option<u64>,
    /// Logical envelope bytes, excluding transport encryption and datagram headers.
    pub payload: Vec<u8>,
}

impl ReplayRecord {
    /// Construct a record.  Bounds and ordering are checked by [`ReplayLog::append`].
    #[must_use]
    pub fn new(
        tick: u64,
        sequence: u64,
        channel: NetChannel,
        direction: ReplayDirection,
        state_hash: Option<u64>,
        payload: Vec<u8>,
    ) -> Self {
        Self {
            tick,
            sequence,
            channel,
            direction,
            state_hash,
            payload,
        }
    }
}

/// Errors from bounded recording or replay stream decoding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplayError {
    /// The bounded recorder has no remaining record slots.
    CapacityExceeded {
        /// Maximum number of records retained by the recorder.
        limit: usize,
    },
    /// A payload exceeds the recorder/decoder limit.
    PayloadTooLarge {
        /// Supplied payload length.
        len: usize,
        /// Configured maximum payload length.
        max: usize,
    },
    /// Events must be appended in nondecreasing tick order.
    TickRewind {
        /// Tick of the last accepted record.
        previous: u64,
        /// Tick of the rejected record.
        next: u64,
    },
    /// The stream header does not identify this codec.
    InvalidMagic,
    /// The stream version is not supported.
    UnsupportedVersion {
        /// Version found in the stream header.
        version: u8,
    },
    /// The byte stream ended before a complete field or record was available.
    Truncated,
    /// A wire enum or reserved field contained an invalid value.
    InvalidField,
    /// The stream contained bytes after the declared record count.
    TrailingBytes,
    /// The declared record count cannot fit the caller's bound.
    RecordCountExceeded {
        /// Number declared by the stream or attempted to encode.
        count: usize,
        /// Maximum accepted record count.
        max: usize,
    },
}

/// A bounded, append-ordered logical event recording.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReplayLog {
    records: Vec<ReplayRecord>,
    max_records: usize,
    max_payload: usize,
}

impl ReplayLog {
    /// Create a bounded recording.  A zero record limit intentionally records
    /// nothing and makes every append fail explicitly.
    #[must_use]
    pub fn new(max_records: usize, max_payload: usize) -> Self {
        Self {
            records: Vec::with_capacity(max_records),
            max_records,
            max_payload: max_payload.min(DEFAULT_MAX_PAYLOAD),
        }
    }

    /// Create a recording using the standard payload bound.
    #[must_use]
    pub fn with_capacity(max_records: usize) -> Self {
        Self::new(max_records, DEFAULT_MAX_PAYLOAD)
    }

    /// Append one event without silently evicting evidence.
    pub fn append(&mut self, record: ReplayRecord) -> Result<(), ReplayError> {
        if self.records.len() >= self.max_records {
            return Err(ReplayError::CapacityExceeded {
                limit: self.max_records,
            });
        }
        if record.payload.len() > self.max_payload {
            return Err(ReplayError::PayloadTooLarge {
                len: record.payload.len(),
                max: self.max_payload,
            });
        }
        if let Some(previous) = self.records.last()
            && record.tick < previous.tick
        {
            return Err(ReplayError::TickRewind {
                previous: previous.tick,
                next: record.tick,
            });
        }
        self.records.push(record);
        Ok(())
    }

    /// Borrow records in deterministic append order.
    #[must_use]
    pub fn records(&self) -> &[ReplayRecord] {
        &self.records
    }

    /// Number of recorded events.
    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether no events have been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Create a cursor starting before the first event.
    #[must_use]
    pub const fn cursor(&self) -> ReplayCursor {
        ReplayCursor { index: 0 }
    }

    /// Encode the complete recording into a deterministic bounded stream.
    pub fn encode(&self) -> Result<Vec<u8>, ReplayError> {
        let count =
            u32::try_from(self.records.len()).map_err(|_| ReplayError::RecordCountExceeded {
                count: self.records.len(),
                max: u32::MAX as usize,
            })?;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&REPLAY_MAGIC);
        bytes.push(REPLAY_VERSION);
        bytes.push(0); // reserved flags
        bytes.extend_from_slice(&count.to_le_bytes());
        for record in &self.records {
            let payload_len =
                u32::try_from(record.payload.len()).map_err(|_| ReplayError::PayloadTooLarge {
                    len: record.payload.len(),
                    max: u32::MAX as usize,
                })?;
            bytes.extend_from_slice(&record.tick.to_le_bytes());
            bytes.extend_from_slice(&record.sequence.to_le_bytes());
            bytes.push(record.channel as u8);
            bytes.push(record.direction as u8);
            bytes.push(u8::from(record.state_hash.is_some()));
            bytes.push(0); // reserved flags
            bytes.extend_from_slice(&payload_len.to_le_bytes());
            if let Some(hash) = record.state_hash {
                bytes.extend_from_slice(&hash.to_le_bytes());
            }
            bytes.extend_from_slice(&record.payload);
        }
        Ok(bytes)
    }

    /// Decode a complete recording under explicit record and payload bounds.
    pub fn decode(
        bytes: &[u8],
        max_records: usize,
        max_payload: usize,
    ) -> Result<Self, ReplayError> {
        if bytes.len() < HEADER_LEN || bytes[..4] != REPLAY_MAGIC {
            return Err(ReplayError::InvalidMagic);
        }
        if bytes[4] != REPLAY_VERSION {
            return Err(ReplayError::UnsupportedVersion { version: bytes[4] });
        }
        if bytes[5] != 0 {
            return Err(ReplayError::InvalidField);
        }
        let count = u32::from_le_bytes(bytes[6..10].try_into().expect("header checked")) as usize;
        if count > max_records {
            return Err(ReplayError::RecordCountExceeded {
                count,
                max: max_records,
            });
        }
        let mut log = Self::new(max_records, max_payload);
        let mut offset = HEADER_LEN;
        for _ in 0..count {
            if bytes.len().saturating_sub(offset) < RECORD_HEADER_LEN {
                return Err(ReplayError::Truncated);
            }
            let tick = read_u64(bytes, &mut offset)?;
            let sequence = read_u64(bytes, &mut offset)?;
            let channel = channel_from_wire(read_u8(bytes, &mut offset)?)?;
            let direction = ReplayDirection::from_wire(read_u8(bytes, &mut offset)?)
                .ok_or(ReplayError::InvalidField)?;
            let has_hash = read_u8(bytes, &mut offset)?;
            let flags = read_u8(bytes, &mut offset)?;
            if flags != 0 || has_hash > 1 {
                return Err(ReplayError::InvalidField);
            }
            let payload_len = read_u32(bytes, &mut offset)? as usize;
            if payload_len > log.max_payload {
                return Err(ReplayError::PayloadTooLarge {
                    len: payload_len,
                    max: log.max_payload,
                });
            }
            let state_hash = if has_hash == 1 {
                Some(read_u64(bytes, &mut offset)?)
            } else {
                None
            };
            let end = offset
                .checked_add(payload_len)
                .ok_or(ReplayError::Truncated)?;
            let payload = bytes
                .get(offset..end)
                .ok_or(ReplayError::Truncated)?
                .to_vec();
            offset = end;
            log.append(ReplayRecord::new(
                tick, sequence, channel, direction, state_hash, payload,
            ))?;
        }
        if offset != bytes.len() {
            return Err(ReplayError::TrailingBytes);
        }
        Ok(log)
    }
}

/// A monotonic cursor over a [`ReplayLog`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ReplayCursor {
    index: usize,
}

impl ReplayCursor {
    /// Return the next record and advance the cursor.
    pub fn next<'a>(&mut self, log: &'a ReplayLog) -> Option<&'a ReplayRecord> {
        let record = log.records.get(self.index)?;
        self.index += 1;
        Some(record)
    }

    /// Number of records already consumed.
    #[must_use]
    pub const fn position(self) -> usize {
        self.index
    }

    /// Whether the cursor has consumed `log`.
    #[must_use]
    pub fn is_finished(self, log: &ReplayLog) -> bool {
        self.index >= log.len()
    }
}

fn channel_from_wire(value: u8) -> Result<NetChannel, ReplayError> {
    match value {
        0 => Ok(NetChannel::EventsReliable),
        1 => Ok(NetChannel::StateUnreliable),
        2 => Ok(NetChannel::InputUnreliable),
        3 => Ok(NetChannel::RpcReliable),
        4 => Ok(NetChannel::BulkReliable),
        _ => Err(ReplayError::InvalidField),
    }
}

fn read_u8(bytes: &[u8], offset: &mut usize) -> Result<u8, ReplayError> {
    let value = *bytes.get(*offset).ok_or(ReplayError::Truncated)?;
    *offset += 1;
    Ok(value)
}

fn read_u32(bytes: &[u8], offset: &mut usize) -> Result<u32, ReplayError> {
    let end = (*offset).checked_add(4).ok_or(ReplayError::Truncated)?;
    let value = u32::from_le_bytes(
        bytes
            .get(*offset..end)
            .ok_or(ReplayError::Truncated)?
            .try_into()
            .expect("slice length checked"),
    );
    *offset = end;
    Ok(value)
}

fn read_u64(bytes: &[u8], offset: &mut usize) -> Result<u64, ReplayError> {
    let end = (*offset).checked_add(8).ok_or(ReplayError::Truncated)?;
    let value = u64::from_le_bytes(
        bytes
            .get(*offset..end)
            .ok_or(ReplayError::Truncated)?
            .try_into()
            .expect("slice length checked"),
    );
    *offset = end;
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn record(tick: u64, sequence: u64, payload: &[u8]) -> ReplayRecord {
        ReplayRecord::new(
            tick,
            sequence,
            NetChannel::EventsReliable,
            ReplayDirection::ServerToClient,
            Some(tick ^ sequence),
            payload.to_vec(),
        )
    }

    #[test]
    fn bounded_log_round_trips_and_replays_in_order() {
        let mut log = ReplayLog::with_capacity(2);
        log.append(record(4, 7, b"first")).unwrap();
        log.append(record(4, 8, b"second")).unwrap();
        let bytes = log.encode().unwrap();
        let decoded = ReplayLog::decode(&bytes, 2, DEFAULT_MAX_PAYLOAD).unwrap();
        assert_eq!(decoded, log);

        let mut cursor = decoded.cursor();
        assert_eq!(cursor.next(&decoded).unwrap().payload, b"first");
        assert_eq!(cursor.position(), 1);
        assert_eq!(cursor.next(&decoded).unwrap().payload, b"second");
        assert!(cursor.next(&decoded).is_none());
        assert!(cursor.is_finished(&decoded));
    }

    #[test]
    fn append_rejects_capacity_payload_and_tick_rewind() {
        let mut log = ReplayLog::new(1, 3);
        log.append(record(2, 0, b"abc")).unwrap();
        assert_eq!(
            log.append(record(1, 1, b"x")),
            Err(ReplayError::CapacityExceeded { limit: 1 })
        );

        let mut log = ReplayLog::new(2, 3);
        log.append(record(2, 0, b"x")).unwrap();
        assert_eq!(
            log.append(record(1, 1, b"y")),
            Err(ReplayError::TickRewind {
                previous: 2,
                next: 1
            })
        );
        assert_eq!(
            log.append(record(2, 1, b"long")),
            Err(ReplayError::PayloadTooLarge { len: 4, max: 3 })
        );
    }

    #[test]
    fn decoder_rejects_truncation_invalid_fields_and_trailing_bytes() {
        let mut log = ReplayLog::with_capacity(1);
        log.append(record(1, 2, b"payload")).unwrap();
        let bytes = log.encode().unwrap();
        assert_eq!(
            ReplayLog::decode(&bytes[..bytes.len() - 1], 1, DEFAULT_MAX_PAYLOAD),
            Err(ReplayError::Truncated)
        );

        let mut invalid = bytes.clone();
        invalid[26] = 9; // channel byte in the first record
        assert_eq!(
            ReplayLog::decode(&invalid, 1, DEFAULT_MAX_PAYLOAD),
            Err(ReplayError::InvalidField)
        );

        let mut trailing = bytes;
        trailing.push(0);
        assert_eq!(
            ReplayLog::decode(&trailing, 1, DEFAULT_MAX_PAYLOAD),
            Err(ReplayError::TrailingBytes)
        );
    }

    #[test]
    fn decoder_enforces_record_and_payload_bounds_without_checksum_claims() {
        let mut log = ReplayLog::with_capacity(1);
        log.append(record(1, 2, b"payload")).unwrap();
        let bytes = log.encode().unwrap();
        assert_eq!(
            ReplayLog::decode(&bytes, 0, DEFAULT_MAX_PAYLOAD),
            Err(ReplayError::RecordCountExceeded { count: 1, max: 0 })
        );
        assert_eq!(
            ReplayLog::decode(&bytes, 1, 3),
            Err(ReplayError::PayloadTooLarge { len: 7, max: 3 })
        );

        // state_hash is carried as logical metadata; changing it is not a
        // byte-integrity check and remains valid when the wire field is valid.
        let mut changed_hash = bytes;
        changed_hash[34] ^= 1;
        assert!(ReplayLog::decode(&changed_hash, 1, DEFAULT_MAX_PAYLOAD).is_ok());
    }

    #[test]
    fn all_channels_and_directions_have_stable_wire_values() {
        let channels = [
            NetChannel::EventsReliable,
            NetChannel::StateUnreliable,
            NetChannel::InputUnreliable,
            NetChannel::RpcReliable,
            NetChannel::BulkReliable,
        ];
        let mut log = ReplayLog::new(channels.len(), 8);
        for (index, channel) in channels.into_iter().enumerate() {
            log.append(ReplayRecord::new(
                index as u64,
                index as u64,
                channel,
                if index % 2 == 0 {
                    ReplayDirection::ClientToServer
                } else {
                    ReplayDirection::ServerToClient
                },
                None,
                vec![index as u8],
            ))
            .unwrap();
        }
        assert_eq!(
            ReplayLog::decode(&log.encode().unwrap(), 5, 8).unwrap(),
            log
        );
    }
}
