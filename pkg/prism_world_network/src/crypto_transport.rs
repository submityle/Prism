//! Authenticated datagram wrapping with AES-256-GCM.

use alloc::vec::Vec;
use core::fmt;
use ring::aead::{Aad, LessSafeKey, Nonce, UnboundKey, AES_256_GCM};

use crate::{ByteTransport, UdpByteTransport};

const NONCE_PREFIX_BYTES: usize = 4;
const NONCE_SEQUENCE_BYTES: usize = 8;
const NONCE_BYTES: usize = NONCE_PREFIX_BYTES + NONCE_SEQUENCE_BYTES;
const TAG_BYTES: usize = 16;
const ENVELOPE_HEADER_BYTES: usize = 1 + NONCE_SEQUENCE_BYTES;
const REPLAY_WINDOW_BITS: usize = 128;

/// Authenticated envelope revision.
const ENVELOPE_VERSION: u8 = 1;

/// Errors from authenticated datagram processing.
#[derive(Debug)]
pub enum CryptoTransportError {
    /// A direction key was not a valid AES-256 key.
    InvalidKey,
    /// The configured maximum plaintext frame cannot fit in a UDP packet with overhead.
    InvalidFrameLimit(usize),
    /// The send nonce sequence is exhausted and must never wrap.
    SendSequenceExhausted,
    /// The envelope has an unsupported revision.
    UnsupportedVersion(u8),
    /// The envelope is shorter than its fixed metadata and authentication tag.
    TruncatedEnvelope,
    /// The received ciphertext exceeds the configured plaintext frame budget.
    CiphertextTooLarge {
        /// Ciphertext bytes excluding the fixed envelope and authentication tag.
        ciphertext_bytes: usize,
        /// Configured maximum plaintext frame bytes.
        limit: usize,
    },
    /// A bounded allocation failed before output was published.
    AllocationFailed,
    /// The datagram failed AES-GCM authentication.
    AuthenticationFailed,
    /// The authenticated nonce was already received or fell outside replay history.
    ReplayRejected(u64),
    /// The wrapped transport returned an error.
    Transport(CryptoTransportErrorSource),
}

/// Wrapped transport failure category.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CryptoTransportErrorSource {
    /// Non-blocking socket returned WouldBlock or another I/O failure.
    Io,
    /// The wrapped UDP adapter rejected the datagram size.
    DatagramLimit,
}

impl fmt::Display for CryptoTransportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidKey => f.write_str("invalid AES-256-GCM key"),
            Self::InvalidFrameLimit(limit) => write!(f, "invalid encrypted frame limit: {limit}"),
            Self::SendSequenceExhausted => f.write_str("encrypted send sequence exhausted"),
            Self::UnsupportedVersion(version) => {
                write!(f, "unsupported crypto envelope version {version}")
            }
            Self::TruncatedEnvelope => f.write_str("truncated crypto envelope"),
            Self::CiphertextTooLarge {
                ciphertext_bytes,
                limit,
            } => write!(
                f,
                "ciphertext needs {ciphertext_bytes} bytes, limit {limit}"
            ),
            Self::AuthenticationFailed => f.write_str("AES-GCM authentication failed"),
            Self::ReplayRejected(sequence) => write!(
                f,
                "encrypted datagram sequence {sequence} is replayed or stale"
            ),
            Self::AllocationFailed => f.write_str("encrypted transport allocation failed"),
            Self::Transport(_) => f.write_str("wrapped datagram transport failed"),
        }
    }
}

impl core::error::Error for CryptoTransportError {}

/// Fixed-size sliding replay window for authenticated 64-bit datagram nonces.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ReplayWindow {
    highest: Option<u64>,
    seen_low: u64,
    seen_high: u64,
}

impl ReplayWindow {
    /// Returns whether `sequence` is newer than the window or an unseen member of it.
    pub const fn would_accept(&self, sequence: u64) -> bool {
        match self.highest {
            None => true,
            Some(highest) if sequence > highest => true,
            Some(highest) => {
                let distance = highest - sequence;
                if distance >= REPLAY_WINDOW_BITS as u64 {
                    false
                } else if distance < 64 {
                    self.seen_low & (1_u64 << distance) == 0
                } else {
                    self.seen_high & (1_u64 << (distance - 64)) == 0
                }
            }
        }
    }

    /// Marks an authenticated sequence as received if it is in the replay window.
    pub fn accept(&mut self, sequence: u64) -> bool {
        if !self.would_accept(sequence) {
            return false;
        }
        match self.highest {
            None => {
                self.highest = Some(sequence);
                self.seen_low = 1;
                self.seen_high = 0;
            }
            Some(highest) if sequence > highest => {
                let shift = (sequence - highest).min(REPLAY_WINDOW_BITS as u64) as u32;
                if shift >= 128 {
                    self.seen_low = 1;
                    self.seen_high = 0;
                } else if shift >= 64 {
                    self.seen_high = self.seen_low << (shift - 64);
                    self.seen_low = 1;
                } else if shift > 0 {
                    self.seen_high = (self.seen_high << shift) | (self.seen_low >> (64 - shift));
                    self.seen_low = (self.seen_low << shift) | 1;
                }
                self.highest = Some(sequence);
            }
            Some(highest) => {
                let distance = (highest - sequence) as u32;
                if distance < 64 {
                    self.seen_low |= 1_u64 << distance;
                } else {
                    self.seen_high |= 1_u64 << (distance - 64);
                }
            }
        }
        true
    }

    /// Returns the newest authenticated sequence, if one has been accepted.
    pub const fn highest(&self) -> Option<u64> {
        self.highest
    }
}

/// AES-256-GCM authenticated encryption around a connected UDP byte transport.
///
/// The host supplies distinct 32-byte send/receive keys and four-byte nonce
/// prefixes. A nonce is `prefix || sequence_le`; send sequence numbers are
/// monotonic and never reused by this instance, including when the underlying
/// UDP send fails. The receive replay window is fixed at 128 datagrams. Key
/// exchange, peer authentication, key rotation, and unique prefix assignment
/// are deliberately the host's responsibility.
pub struct Aes256GcmTransport {
    transport: UdpByteTransport,
    send_key: LessSafeKey,
    receive_key: LessSafeKey,
    send_prefix: [u8; NONCE_PREFIX_BYTES],
    receive_prefix: [u8; NONCE_PREFIX_BYTES],
    send_direction: u8,
    next_send_sequence: u64,
    receive_window: ReplayWindow,
    max_frame_bytes: usize,
    session_aad: [u8; 16],
}

impl Aes256GcmTransport {
    /// Wraps a connected UDP adapter with independent directional AES-256-GCM keys.
    pub fn new(
        transport: UdpByteTransport,
        send_key: [u8; 32],
        receive_key: [u8; 32],
        send_prefix: [u8; NONCE_PREFIX_BYTES],
        receive_prefix: [u8; NONCE_PREFIX_BYTES],
        send_direction: u8,
        session_id: u128,
        max_frame_bytes: usize,
    ) -> Result<Self, CryptoTransportError> {
        let overhead = ENVELOPE_HEADER_BYTES + TAG_BYTES;
        if send_direction > 1
            || max_frame_bytes == 0
            || max_frame_bytes
                .checked_add(overhead)
                .is_none_or(|total| total > transport.max_frame_bytes())
        {
            return Err(CryptoTransportError::InvalidFrameLimit(max_frame_bytes));
        }
        let send_key = LessSafeKey::new(
            UnboundKey::new(&AES_256_GCM, &send_key)
                .map_err(|_| CryptoTransportError::InvalidKey)?,
        );
        let receive_key = LessSafeKey::new(
            UnboundKey::new(&AES_256_GCM, &receive_key)
                .map_err(|_| CryptoTransportError::InvalidKey)?,
        );
        Ok(Self {
            transport,
            send_key,
            receive_key,
            send_prefix,
            receive_prefix,
            send_direction,
            next_send_sequence: 1,
            receive_window: ReplayWindow::default(),
            max_frame_bytes,
            session_aad: session_id.to_le_bytes(),
        })
    }

    fn nonce(prefix: [u8; NONCE_PREFIX_BYTES], sequence: u64) -> Nonce {
        let mut bytes = [0_u8; NONCE_BYTES];
        bytes[..NONCE_PREFIX_BYTES].copy_from_slice(&prefix);
        bytes[NONCE_PREFIX_BYTES..].copy_from_slice(&sequence.to_le_bytes());
        Nonce::assume_unique_for_key(bytes)
    }

    fn aad(&self, sequence: u64, direction: u8) -> [u8; 25] {
        let mut aad = [0_u8; 25];
        aad[..16].copy_from_slice(&self.session_aad);
        aad[16] = direction;
        aad[17..].copy_from_slice(&sequence.to_le_bytes());
        aad
    }

    /// Returns the connected UDP peer.
    pub const fn peer(&self) -> core::net::SocketAddr {
        self.transport.peer()
    }
}

impl ByteTransport for Aes256GcmTransport {
    type Error = CryptoTransportError;

    fn try_send_bytes(&mut self, frame: &[u8]) -> Result<(), Self::Error> {
        if frame.len() > self.max_frame_bytes {
            return Err(CryptoTransportError::CiphertextTooLarge {
                ciphertext_bytes: frame.len(),
                limit: self.max_frame_bytes,
            });
        }
        let sequence = self.next_send_sequence;
        if sequence == u64::MAX {
            return Err(CryptoTransportError::SendSequenceExhausted);
        }
        self.next_send_sequence += 1;
        let aad = self.aad(sequence, self.send_direction);
        let mut ciphertext = Vec::new();
        ciphertext
            .try_reserve_exact(frame.len() + TAG_BYTES)
            .map_err(|_| CryptoTransportError::Transport(CryptoTransportErrorSource::Io))?;
        ciphertext.extend_from_slice(frame);
        self.send_key
            .seal_in_place_append_tag(
                Self::nonce(self.send_prefix, sequence),
                Aad::from(aad.as_slice()),
                &mut ciphertext,
            )
            .map_err(|_| CryptoTransportError::AuthenticationFailed)?;
        let mut envelope = Vec::new();
        envelope
            .try_reserve_exact(ENVELOPE_HEADER_BYTES + ciphertext.len())
            .map_err(|_| CryptoTransportError::AllocationFailed)?;
        envelope.push(ENVELOPE_VERSION);
        envelope.extend_from_slice(&sequence.to_le_bytes());
        envelope.extend_from_slice(&ciphertext);
        self.transport
            .try_send_bytes(&envelope)
            .map_err(map_transport_error)
    }

    fn try_recv_bytes(&mut self) -> Result<Option<Vec<u8>>, Self::Error> {
        let Some(mut envelope) = self
            .transport
            .try_recv_bytes()
            .map_err(map_transport_error)?
        else {
            return Ok(None);
        };
        if envelope.len() < ENVELOPE_HEADER_BYTES + TAG_BYTES {
            return Err(CryptoTransportError::TruncatedEnvelope);
        }
        if envelope[0] != ENVELOPE_VERSION {
            return Err(CryptoTransportError::UnsupportedVersion(envelope[0]));
        }
        let sequence = u64::from_le_bytes(envelope[1..9].try_into().unwrap());
        if sequence == 0 || !self.receive_window.would_accept(sequence) {
            return Err(CryptoTransportError::ReplayRejected(sequence));
        }
        let cipher_bytes = envelope.len() - ENVELOPE_HEADER_BYTES;
        if cipher_bytes < TAG_BYTES || cipher_bytes - TAG_BYTES > self.max_frame_bytes {
            return Err(CryptoTransportError::CiphertextTooLarge {
                ciphertext_bytes: cipher_bytes.saturating_sub(TAG_BYTES),
                limit: self.max_frame_bytes,
            });
        }
        let aad = self.aad(sequence, 1 - self.send_direction);
        let plaintext = self
            .receive_key
            .open_in_place(
                Self::nonce(self.receive_prefix, sequence),
                Aad::from(aad.as_slice()),
                &mut envelope[ENVELOPE_HEADER_BYTES..],
            )
            .map_err(|_| CryptoTransportError::AuthenticationFailed)?;
        let length = plaintext.len();
        if !self.receive_window.accept(sequence) {
            return Err(CryptoTransportError::ReplayRejected(sequence));
        }
        envelope.copy_within(ENVELOPE_HEADER_BYTES..ENVELOPE_HEADER_BYTES + length, 0);
        envelope.truncate(length);
        Ok(Some(envelope))
    }
}

fn map_transport_error(error: crate::UdpByteTransportError) -> CryptoTransportError {
    match error {
        crate::UdpByteTransportError::FrameTooLarge { .. }
        | crate::UdpByteTransportError::ReceivedDatagramTooLarge { .. }
        | crate::UdpByteTransportError::InvalidFrameLimit(_) => {
            CryptoTransportError::Transport(CryptoTransportErrorSource::DatagramLimit)
        }
        _ => CryptoTransportError::Transport(CryptoTransportErrorSource::Io),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{UdpByteTransport, UdpByteTransportConfig};
    use std::net::UdpSocket;
    use std::time::{Duration, Instant};

    fn pair() -> (Aes256GcmTransport, Aes256GcmTransport) {
        let left_socket = UdpSocket::bind(("127.0.0.1", 0)).unwrap();
        let right_socket = UdpSocket::bind(("127.0.0.1", 0)).unwrap();
        left_socket
            .connect(right_socket.local_addr().unwrap())
            .unwrap();
        right_socket
            .connect(left_socket.local_addr().unwrap())
            .unwrap();
        let udp_config = UdpByteTransportConfig::new(1200).unwrap();
        let left = UdpByteTransport::new(left_socket, udp_config).unwrap();
        let right = UdpByteTransport::new(right_socket, udp_config).unwrap();
        let key_a = [0x31; 32];
        let key_b = [0x72; 32];
        (
            Aes256GcmTransport::new(left, key_a, key_b, [1; 4], [2; 4], 0, 77, 1100).unwrap(),
            Aes256GcmTransport::new(right, key_b, key_a, [2; 4], [1; 4], 1, 77, 1100).unwrap(),
        )
    }

    fn receive_until(
        transport: &mut Aes256GcmTransport,
    ) -> Result<Option<Vec<u8>>, CryptoTransportError> {
        let deadline = Instant::now() + Duration::from_secs(1);
        loop {
            if let Some(frame) = transport.try_recv_bytes()? {
                return Ok(Some(frame));
            }
            if Instant::now() >= deadline {
                return Ok(None);
            }
            std::thread::yield_now();
        }
    }

    #[test]
    fn encrypted_udp_round_trip_and_nonce_changes_on_retry_payload() {
        let (mut left, mut right) = pair();
        left.try_send_bytes(b"secret frame").unwrap();
        assert_eq!(receive_until(&mut right).unwrap().unwrap(), b"secret frame");
        left.try_send_bytes(b"secret frame").unwrap();
        assert_eq!(receive_until(&mut right).unwrap().unwrap(), b"secret frame");
        assert_eq!(right.receive_window.highest(), Some(2));
    }

    #[test]
    fn invalid_tag_does_not_consume_receive_replay_window() {
        let (left, mut right) = pair();
        left.transport.inject_test_datagram(&[
            1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]);
        assert!(matches!(
            receive_until(&mut right),
            Err(CryptoTransportError::AuthenticationFailed)
        ));
        assert_eq!(right.receive_window.highest(), None);
    }

    #[test]
    fn replay_window_rejects_duplicates_and_old_values() {
        let mut window = ReplayWindow::default();
        assert!(window.accept(10));
        assert!(window.accept(12));
        assert!(window.accept(11));
        assert!(!window.accept(11));
        assert!(window.accept(200));
        assert!(!window.accept(12));
    }
}
