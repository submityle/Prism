//! Canonical checkpoint codec for a future durable `MultiPeerAuthority`.
//!
//! This stores only an opaque bounded AOI checkpoint and peer delivery
//! checkpoints with the exact wire bytes still in each ordered outbox. It does
//! not contain or restore transports, sessions, ECS/World state, or runtime
//! authority objects; callers must define those recovery boundaries separately.

use alloc::vec::Vec;

const MAGIC: [u8; 4] = *b"MPAO";
const VERSION: u8 = 1;
const HEADER_LEN: usize = 4 + 1 + 8 + 8;
const PEER_HEADER_LEN: usize = 8 + 8 + 8 + 8;

/// Resource limits applied before allocating or accepting checkpoint data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MultiPeerCheckpointLimits {
    /// Maximum encoded checkpoint size, including all framing bytes.
    pub max_total_bytes: usize,
    /// Maximum opaque AOI checkpoint size.
    pub max_aoi_snapshot_bytes: usize,
    /// Maximum number of peers in the checkpoint.
    pub max_peers: usize,
    /// Maximum retained exact wire payloads per peer.
    pub max_events_per_peer: usize,
    /// Maximum size of each individual wire payload.
    pub max_payload_bytes: usize,
}

/// Checkpoint state for durable MultiPeer AOI and per-peer delivery queues.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MultiPeerCheckpoint {
    /// Opaque AOI snapshot bytes, encoded and validated by the AOI owner.
    pub aoi_snapshot: Vec<u8>,
    /// Peers in strictly ascending `peer_id` order.
    pub peers: Vec<MultiPeerPeerCheckpoint>,
}

/// One peer's acknowledged sequence and ordered, exact outbox payloads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MultiPeerPeerCheckpoint {
    /// Stable peer identity.
    pub peer_id: u64,
    /// Highest delivery sequence durably acknowledged by the peer.
    pub acknowledged_sequence: u64,
    /// Sequence to assign to the next newly enqueued payload.
    pub next_sequence: u64,
    /// Exact wire payloads in delivery-sequence order, starting at
    /// `acknowledged_sequence + 1`.
    pub outbox: Vec<Vec<u8>>,
}

/// Invalid, non-canonical, or over-limit MultiPeer checkpoint data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MultiPeerCheckpointCodecError {
    /// The checkpoint is truncated, malformed, out of canonical order, or has
    /// inconsistent sequence metadata.
    Invalid,
    /// A configured byte, peer, event, or payload bound was exceeded.
    LimitExceeded,
    /// A bounded allocation could not be reserved.
    AllocationFailed,
}

impl MultiPeerCheckpoint {
    /// Encode a canonical, versioned little-endian checkpoint.
    pub fn encode(
        &self,
        limits: MultiPeerCheckpointLimits,
    ) -> Result<Vec<u8>, MultiPeerCheckpointCodecError> {
        validate_limits(limits)?;
        if self.aoi_snapshot.len() > limits.max_aoi_snapshot_bytes
            || self.peers.len() > limits.max_peers
        {
            return Err(MultiPeerCheckpointCodecError::LimitExceeded);
        }

        let mut encoded_len = HEADER_LEN;
        encoded_len = encoded_len
            .checked_add(self.aoi_snapshot.len())
            .ok_or(MultiPeerCheckpointCodecError::Invalid)?;
        let mut previous_peer_id = None;
        for peer in &self.peers {
            if previous_peer_id.is_some_and(|previous| previous >= peer.peer_id) {
                return Err(MultiPeerCheckpointCodecError::Invalid);
            }
            previous_peer_id = Some(peer.peer_id);
            validate_peer(peer, limits)?;
            encoded_len = encoded_len
                .checked_add(PEER_HEADER_LEN)
                .and_then(|length| length.checked_add(peer.outbox.len().checked_mul(8)?))
                .ok_or(MultiPeerCheckpointCodecError::Invalid)?;
            for payload in &peer.outbox {
                encoded_len = encoded_len
                    .checked_add(payload.len())
                    .ok_or(MultiPeerCheckpointCodecError::Invalid)?;
            }
        }
        if encoded_len > limits.max_total_bytes {
            return Err(MultiPeerCheckpointCodecError::LimitExceeded);
        }

        let aoi_len = u64::try_from(self.aoi_snapshot.len())
            .map_err(|_| MultiPeerCheckpointCodecError::Invalid)?;
        let peer_count =
            u64::try_from(self.peers.len()).map_err(|_| MultiPeerCheckpointCodecError::Invalid)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(encoded_len)
            .map_err(|_| MultiPeerCheckpointCodecError::AllocationFailed)?;
        bytes.extend_from_slice(&MAGIC);
        bytes.push(VERSION);
        bytes.extend_from_slice(&aoi_len.to_le_bytes());
        bytes.extend_from_slice(&peer_count.to_le_bytes());
        bytes.extend_from_slice(&self.aoi_snapshot);
        for peer in &self.peers {
            bytes.extend_from_slice(&peer.peer_id.to_le_bytes());
            bytes.extend_from_slice(&peer.acknowledged_sequence.to_le_bytes());
            bytes.extend_from_slice(&peer.next_sequence.to_le_bytes());
            bytes.extend_from_slice(
                &u64::try_from(peer.outbox.len())
                    .map_err(|_| MultiPeerCheckpointCodecError::Invalid)?
                    .to_le_bytes(),
            );
            for payload in &peer.outbox {
                bytes.extend_from_slice(
                    &u64::try_from(payload.len())
                        .map_err(|_| MultiPeerCheckpointCodecError::Invalid)?
                        .to_le_bytes(),
                );
                bytes.extend_from_slice(payload);
            }
        }
        debug_assert_eq!(bytes.len(), encoded_len);
        Ok(bytes)
    }

    /// Decode a canonical checkpoint, enforcing all bounds before allocation.
    pub fn decode(
        bytes: &[u8],
        limits: MultiPeerCheckpointLimits,
    ) -> Result<Self, MultiPeerCheckpointCodecError> {
        validate_limits(limits)?;
        if bytes.len() > limits.max_total_bytes {
            return Err(MultiPeerCheckpointCodecError::LimitExceeded);
        }
        let mut cursor = Cursor::new(bytes);
        if cursor.take(4)? != MAGIC || cursor.byte()? != VERSION {
            return Err(MultiPeerCheckpointCodecError::Invalid);
        }
        let aoi_len = cursor.usize_u64()?;
        let peer_count = cursor.usize_u64()?;
        if aoi_len > limits.max_aoi_snapshot_bytes || peer_count > limits.max_peers {
            return Err(MultiPeerCheckpointCodecError::LimitExceeded);
        }
        let aoi_snapshot = cursor.take(aoi_len)?.to_vec_fallible()?;
        let mut peers = Vec::new();
        peers
            .try_reserve_exact(peer_count)
            .map_err(|_| MultiPeerCheckpointCodecError::AllocationFailed)?;
        let mut previous_peer_id = None;
        for _ in 0..peer_count {
            let peer = MultiPeerPeerCheckpoint {
                peer_id: cursor.u64()?,
                acknowledged_sequence: cursor.u64()?,
                next_sequence: cursor.u64()?,
                outbox: Vec::new(),
            };
            if previous_peer_id.is_some_and(|previous| previous >= peer.peer_id) {
                return Err(MultiPeerCheckpointCodecError::Invalid);
            }
            previous_peer_id = Some(peer.peer_id);
            let event_count = cursor.usize_u64()?;
            if event_count > limits.max_events_per_peer {
                return Err(MultiPeerCheckpointCodecError::LimitExceeded);
            }
            let mut peer = peer;
            peer.outbox
                .try_reserve_exact(event_count)
                .map_err(|_| MultiPeerCheckpointCodecError::AllocationFailed)?;
            for _ in 0..event_count {
                let payload_len = cursor.usize_u64()?;
                if payload_len > limits.max_payload_bytes {
                    return Err(MultiPeerCheckpointCodecError::LimitExceeded);
                }
                peer.outbox
                    .push(cursor.take(payload_len)?.to_vec_fallible()?);
            }
            validate_peer(&peer, limits)?;
            peers.push(peer);
        }
        if !cursor.is_empty() {
            return Err(MultiPeerCheckpointCodecError::Invalid);
        }
        Ok(Self {
            aoi_snapshot,
            peers,
        })
    }
}

fn validate_limits(limits: MultiPeerCheckpointLimits) -> Result<(), MultiPeerCheckpointCodecError> {
    if limits.max_total_bytes < HEADER_LEN {
        return Err(MultiPeerCheckpointCodecError::LimitExceeded);
    }
    Ok(())
}

fn validate_peer(
    peer: &MultiPeerPeerCheckpoint,
    limits: MultiPeerCheckpointLimits,
) -> Result<(), MultiPeerCheckpointCodecError> {
    if peer.outbox.len() > limits.max_events_per_peer {
        return Err(MultiPeerCheckpointCodecError::LimitExceeded);
    }
    for payload in &peer.outbox {
        if payload.len() > limits.max_payload_bytes {
            return Err(MultiPeerCheckpointCodecError::LimitExceeded);
        }
    }
    let event_count =
        u64::try_from(peer.outbox.len()).map_err(|_| MultiPeerCheckpointCodecError::Invalid)?;
    let expected_next = peer
        .acknowledged_sequence
        .checked_add(event_count)
        .and_then(|sequence| sequence.checked_add(1))
        .ok_or(MultiPeerCheckpointCodecError::Invalid)?;
    if peer.next_sequence != expected_next {
        return Err(MultiPeerCheckpointCodecError::Invalid);
    }
    Ok(())
}

struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, length: usize) -> Result<&'a [u8], MultiPeerCheckpointCodecError> {
        let end = self
            .offset
            .checked_add(length)
            .ok_or(MultiPeerCheckpointCodecError::Invalid)?;
        let value = self
            .bytes
            .get(self.offset..end)
            .ok_or(MultiPeerCheckpointCodecError::Invalid)?;
        self.offset = end;
        Ok(value)
    }

    fn byte(&mut self) -> Result<u8, MultiPeerCheckpointCodecError> {
        Ok(self.take(1)?[0])
    }

    fn u64(&mut self) -> Result<u64, MultiPeerCheckpointCodecError> {
        let bytes: [u8; 8] = self
            .take(8)?
            .try_into()
            .map_err(|_| MultiPeerCheckpointCodecError::Invalid)?;
        Ok(u64::from_le_bytes(bytes))
    }

    fn usize_u64(&mut self) -> Result<usize, MultiPeerCheckpointCodecError> {
        usize::try_from(self.u64()?).map_err(|_| MultiPeerCheckpointCodecError::Invalid)
    }

    const fn is_empty(&self) -> bool {
        self.offset == self.bytes.len()
    }
}

trait ToVecFallible {
    fn to_vec_fallible(&self) -> Result<Vec<u8>, MultiPeerCheckpointCodecError>;
}

impl ToVecFallible for [u8] {
    fn to_vec_fallible(&self) -> Result<Vec<u8>, MultiPeerCheckpointCodecError> {
        let mut result = Vec::new();
        result
            .try_reserve_exact(self.len())
            .map_err(|_| MultiPeerCheckpointCodecError::AllocationFailed)?;
        result.extend_from_slice(self);
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use alloc::{vec, vec::Vec};

    use super::{
        MultiPeerCheckpoint, MultiPeerCheckpointCodecError as Error, MultiPeerCheckpointLimits,
        MultiPeerPeerCheckpoint,
    };

    const LIMITS: MultiPeerCheckpointLimits = MultiPeerCheckpointLimits {
        max_total_bytes: 256,
        max_aoi_snapshot_bytes: 32,
        max_peers: 4,
        max_events_per_peer: 4,
        max_payload_bytes: 32,
    };

    fn sample() -> MultiPeerCheckpoint {
        MultiPeerCheckpoint {
            aoi_snapshot: vec![0xA0, 0xA1, 0xA2],
            peers: vec![
                MultiPeerPeerCheckpoint {
                    peer_id: 2,
                    acknowledged_sequence: 7,
                    next_sequence: 11,
                    outbox: vec![vec![1, 2], vec![], vec![3, 4, 5]],
                },
                MultiPeerPeerCheckpoint {
                    peer_id: 9,
                    acknowledged_sequence: 0,
                    next_sequence: 1,
                    outbox: vec![],
                },
            ],
        }
    }

    fn fields(bytes: &mut [u8]) -> Vec<(usize, usize)> {
        let mut pos = 21 + 3;
        let mut result = Vec::new();
        for _ in 0..2 {
            pos += 32;
            let count = u64::from_le_bytes(bytes[pos - 8..pos].try_into().unwrap()) as usize;
            for _ in 0..count {
                let len_pos = pos;
                let len = u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap()) as usize;
                pos += 8;
                result.push((len_pos, pos));
                pos += len;
            }
        }
        result
    }

    #[test]
    fn canonical_roundtrip_is_stable() {
        let checkpoint = sample();
        let encoded = checkpoint.encode(LIMITS).unwrap();
        assert_eq!(&encoded[..5], b"MPAO\x01");
        assert_eq!(
            MultiPeerCheckpoint::decode(&encoded, LIMITS).unwrap(),
            checkpoint
        );
        assert_eq!(
            MultiPeerCheckpoint::decode(&encoded, LIMITS)
                .unwrap()
                .encode(LIMITS)
                .unwrap(),
            encoded
        );
    }

    #[test]
    fn rejects_truncation_trailing_bytes_magic_and_version() {
        let encoded = sample().encode(LIMITS).unwrap();
        for end in 0..encoded.len() {
            assert_eq!(
                MultiPeerCheckpoint::decode(&encoded[..end], LIMITS),
                Err(Error::Invalid)
            );
        }
        let mut trailing = encoded.clone();
        trailing.push(0);
        assert_eq!(
            MultiPeerCheckpoint::decode(&trailing, LIMITS),
            Err(Error::Invalid)
        );
        let mut bad_magic = encoded.clone();
        bad_magic[0] ^= 1;
        assert_eq!(
            MultiPeerCheckpoint::decode(&bad_magic, LIMITS),
            Err(Error::Invalid)
        );
        let mut bad_version = encoded;
        bad_version[4] = 2;
        assert_eq!(
            MultiPeerCheckpoint::decode(&bad_version, LIMITS),
            Err(Error::Invalid)
        );
    }

    #[test]
    fn rejects_unsorted_or_duplicate_peers_and_sequence_gaps() {
        let mut checkpoint = sample();
        checkpoint.peers.swap(0, 1);
        assert_eq!(checkpoint.encode(LIMITS), Err(Error::Invalid));
        checkpoint.peers[0].peer_id = checkpoint.peers[1].peer_id;
        assert_eq!(checkpoint.encode(LIMITS), Err(Error::Invalid));
        let mut checkpoint = sample();
        checkpoint.peers[0].next_sequence += 1;
        assert_eq!(checkpoint.encode(LIMITS), Err(Error::Invalid));
    }

    #[test]
    fn rejects_sequence_overflow_and_empty_outbox_at_max_sequence() {
        let checkpoint = MultiPeerCheckpoint {
            aoi_snapshot: Vec::new(),
            peers: vec![MultiPeerPeerCheckpoint {
                peer_id: 1,
                acknowledged_sequence: u64::MAX,
                next_sequence: 0,
                outbox: Vec::new(),
            }],
        };
        assert_eq!(checkpoint.encode(LIMITS), Err(Error::Invalid));
    }

    #[test]
    fn rejects_each_configured_bound() {
        let checkpoint = sample();
        assert_eq!(
            checkpoint.encode(MultiPeerCheckpointLimits {
                max_total_bytes: 20,
                ..LIMITS
            }),
            Err(Error::LimitExceeded)
        );
        assert_eq!(
            checkpoint.encode(MultiPeerCheckpointLimits {
                max_aoi_snapshot_bytes: 2,
                ..LIMITS
            }),
            Err(Error::LimitExceeded)
        );
        assert_eq!(
            checkpoint.encode(MultiPeerCheckpointLimits {
                max_peers: 1,
                ..LIMITS
            }),
            Err(Error::LimitExceeded)
        );
        assert_eq!(
            checkpoint.encode(MultiPeerCheckpointLimits {
                max_events_per_peer: 2,
                ..LIMITS
            }),
            Err(Error::LimitExceeded)
        );
        assert_eq!(
            checkpoint.encode(MultiPeerCheckpointLimits {
                max_payload_bytes: 2,
                ..LIMITS
            }),
            Err(Error::LimitExceeded)
        );
        let bytes = checkpoint.encode(LIMITS).unwrap();
        assert_eq!(
            MultiPeerCheckpoint::decode(
                &bytes,
                MultiPeerCheckpointLimits {
                    max_total_bytes: bytes.len() - 1,
                    ..LIMITS
                }
            ),
            Err(Error::LimitExceeded)
        );
        assert_eq!(
            MultiPeerCheckpoint::decode(
                &bytes,
                MultiPeerCheckpointLimits {
                    max_aoi_snapshot_bytes: 2,
                    ..LIMITS
                }
            ),
            Err(Error::LimitExceeded)
        );
        assert_eq!(
            MultiPeerCheckpoint::decode(
                &bytes,
                MultiPeerCheckpointLimits {
                    max_peers: 1,
                    ..LIMITS
                }
            ),
            Err(Error::LimitExceeded)
        );
    }

    #[test]
    fn rejects_malformed_wire_peer_order_sequence_and_payload_lengths() {
        let encoded = sample().encode(LIMITS).unwrap();
        // First peer's ID starts after the fixed header and opaque AOI bytes.
        let mut unsorted = encoded.clone();
        unsorted[24..32].copy_from_slice(&9u64.to_le_bytes());
        assert_eq!(
            MultiPeerCheckpoint::decode(&unsorted, LIMITS),
            Err(Error::Invalid)
        );

        let mut bad_next = encoded.clone();
        bad_next[40..48].copy_from_slice(&12u64.to_le_bytes());
        assert_eq!(
            MultiPeerCheckpoint::decode(&bad_next, LIMITS),
            Err(Error::Invalid)
        );

        let mut oversized_payload = encoded.clone();
        let (length_offset, _) = fields(&mut oversized_payload)[0];
        oversized_payload[length_offset..length_offset + 8].copy_from_slice(&33u64.to_le_bytes());
        assert_eq!(
            MultiPeerCheckpoint::decode(&oversized_payload, LIMITS),
            Err(Error::LimitExceeded)
        );

        let mut excessive_events = encoded;
        excessive_events[48..56].copy_from_slice(&5u64.to_le_bytes());
        assert_eq!(
            MultiPeerCheckpoint::decode(&excessive_events, LIMITS),
            Err(Error::LimitExceeded)
        );
    }
}
