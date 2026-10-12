//! Length-prefixed message framing (design §6).
//!
//! A datagram transport delivers one message per write, but a stream transport
//! (TCP, a QUIC byte stream, a file) delivers a byte run with no message
//! boundaries. [`encode_frame`] prefixes a payload with a LEB128 length so the
//! receiver can split the stream back into messages, and [`FrameDecoder`] does
//! that incrementally: feed it whatever bytes have arrived, take the frames that
//! are now complete, and keep the rest buffered. It is the first thing any stream
//! transport needs and the last thing a datagram one does not.

use alloc::vec::Vec;

/// The largest payload a single frame may carry. A length prefix past this is a
/// protocol error, rejected before any large allocation is attempted.
pub const MAX_FRAME_LEN: usize = 1 << 20;

/// The most bytes a `usize` LEB128 length prefix can occupy.
const MAX_PREFIX_LEN: usize = usize::BITS.div_ceil(7) as usize;

/// A framing failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameError {
    /// A length prefix exceeded [`MAX_FRAME_LEN`].
    Oversized {
        /// The encoded length.
        len: usize,
        /// The maximum allowed.
        max: usize,
    },
    /// A length prefix was not a valid LEB128 `usize` (more than
    /// `ceil(usize::BITS / 7)` bytes, or with bits that do not fit a `usize`).
    Malformed,
    /// A caller required exactly one complete frame but the input ended before one
    /// was available (used by snapshot reseed, which is not incremental).
    Truncated,
}

/// Append `payload` to `out` as a LEB128-length-prefixed frame.
///
/// # Errors
/// [`FrameError::Oversized`] when the payload is longer than [`MAX_FRAME_LEN`],
/// so this encoder can never emit a frame its own [`FrameDecoder`] would reject.
pub fn encode_frame(payload: &[u8], out: &mut Vec<u8>) -> Result<(), FrameError> {
    if payload.len() > MAX_FRAME_LEN {
        return Err(FrameError::Oversized {
            len: payload.len(),
            max: MAX_FRAME_LEN,
        });
    }
    write_varint(payload.len(), out);
    out.extend_from_slice(payload);
    Ok(())
}

/// An incremental frame splitter over a byte stream.
///
/// Bytes accumulate in an internal buffer; [`next_frame`](Self::next_frame)
/// yields each complete frame in turn, leaving a partial frame buffered for the
/// next feed.
#[derive(Clone, Debug, Default)]
pub struct FrameDecoder {
    buffer: Vec<u8>,
    /// Once a frame is malformed the stream is desynchronized and cannot be
    /// resynchronized; the decoder latches the error and refuses further input so
    /// a hostile or corrupt stream cannot grow the buffer without bound.
    poison: Option<FrameError>,
}

impl FrameDecoder {
    /// A fresh decoder with an empty buffer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Buffer `bytes` from the stream.
    pub fn feed(&mut self, bytes: &[u8]) {
        if self.poison.is_some() {
            return;
        }
        self.buffer.extend_from_slice(bytes);
    }

    /// Bytes currently buffered (a partial frame plus any not-yet-taken frames).
    #[must_use]
    pub fn buffered(&self) -> usize {
        self.buffer.len()
    }

    /// Take the next complete frame, or `None` when the buffer does not yet hold
    /// one.
    ///
    /// # Errors
    /// [`FrameError::Oversized`] for a length past [`MAX_FRAME_LEN`] and
    /// [`FrameError::Malformed`] for an invalid length prefix. The first error
    /// **poisons** the decoder: the buffer is cleared and every later call returns
    /// the same error, so a desynchronized stream can neither be mis-parsed nor
    /// grow without bound.
    pub fn next_frame(&mut self) -> Result<Option<Vec<u8>>, FrameError> {
        if let Some(error) = self.poison {
            return Err(error);
        }
        let (len, prefix) = match read_varint(&self.buffer) {
            Ok(Some(header)) => header,
            Ok(None) => return Ok(None),
            Err(error) => return Err(self.poison(error)),
        };
        if len > MAX_FRAME_LEN {
            return Err(self.poison(FrameError::Oversized {
                len,
                max: MAX_FRAME_LEN,
            }));
        }
        let end = prefix + len;
        if self.buffer.len() < end {
            return Ok(None);
        }
        // `drain` in two steps so the payload is copied exactly once.
        let frame: Vec<u8> = self.buffer[prefix..end].to_vec();
        self.buffer.drain(..end);
        Ok(Some(frame))
    }

    /// Like [`next_frame`](Self::next_frame) but leaves the frame buffered: decode
    /// the next complete frame **without** consuming it, so a caller can act on it
    /// (and only then drop it with [`next_frame`](Self::next_frame)) or reject it
    /// while keeping it for a retry.
    ///
    /// # Errors
    /// The same as [`next_frame`](Self::next_frame), and it likewise poisons the
    /// decoder on a malformed or oversized prefix.
    pub fn peek_frame(&mut self) -> Result<Option<Vec<u8>>, FrameError> {
        if let Some(error) = self.poison {
            return Err(error);
        }
        let (len, prefix) = match read_varint(&self.buffer) {
            Ok(Some(header)) => header,
            Ok(None) => return Ok(None),
            Err(error) => return Err(self.poison(error)),
        };
        if len > MAX_FRAME_LEN {
            return Err(self.poison(FrameError::Oversized {
                len,
                max: MAX_FRAME_LEN,
            }));
        }
        let end = prefix + len;
        if self.buffer.len() < end {
            return Ok(None);
        }
        Ok(Some(self.buffer[prefix..end].to_vec()))
    }

    /// Latch `error`, drop the buffered bytes, and return it.
    fn poison(&mut self, error: FrameError) -> FrameError {
        self.buffer.clear();
        self.poison = Some(error);
        error
    }
}

/// Append `value` to `out` as a LEB128 varint.
fn write_varint(mut value: usize, out: &mut Vec<u8>) {
    while value >= 0x80 {
        out.push((value as u8 & 0x7F) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

/// Decode a LEB128 varint: `Ok(Some((value, bytes)))` when complete,
/// `Ok(None)` when the input ends mid-varint, and [`FrameError::Malformed`] when
/// the varint is longer than a `usize` can hold.
///
/// The accumulator is a `u128`, so a chunk that overflows `usize` (e.g. a tenth
/// byte with bits above the top bit) is detected and rejected instead of
/// overflowing the shift — which would panic in a debug build.
fn read_varint(bytes: &[u8]) -> Result<Option<(usize, usize)>, FrameError> {
    let mut value: u128 = 0;
    let mut shift: u32 = 0;
    for (index, &byte) in bytes.iter().enumerate() {
        // A `usize` varint is at most `ceil(bits / 7)` bytes; a longer run can
        // never encode a `usize`.
        if index >= MAX_PREFIX_LEN {
            return Err(FrameError::Malformed);
        }
        value |= u128::from(byte & 0x7F) << shift;
        if byte & 0x80 == 0 {
            let value = usize::try_from(value).map_err(|_| FrameError::Malformed)?;
            return Ok(Some((value, index + 1)));
        }
        shift += 7;
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    #[test]
    fn frames_round_trip_through_the_decoder() {
        let mut stream = Vec::new();
        encode_frame(b"hello", &mut stream).unwrap();
        encode_frame(&[], &mut stream).unwrap();
        encode_frame(&[0u8; 300], &mut stream).unwrap();

        let mut decoder = FrameDecoder::new();
        decoder.feed(&stream);
        assert_eq!(decoder.next_frame().unwrap(), Some(b"hello".to_vec()));
        assert_eq!(decoder.next_frame().unwrap(), Some(Vec::new()));
        assert_eq!(decoder.next_frame().unwrap(), Some(vec![0u8; 300]));
        assert_eq!(decoder.next_frame().unwrap(), None);
        assert_eq!(decoder.buffered(), 0);
    }

    #[test]
    fn a_partial_frame_is_buffered_until_it_completes() {
        let mut stream = Vec::new();
        encode_frame(b"abcdef", &mut stream).unwrap();

        let mut decoder = FrameDecoder::new();
        // Feed one byte at a time: only the last feed yields the frame.
        let mut frames = Vec::new();
        for byte in &stream {
            decoder.feed(core::slice::from_ref(byte));
            while let Some(frame) = decoder.next_frame().unwrap() {
                frames.push(frame);
            }
        }
        assert_eq!(frames, vec![b"abcdef".to_vec()]);
        assert_eq!(decoder.buffered(), 0);
    }

    #[test]
    fn an_oversized_prefix_is_rejected_before_allocating() {
        let mut stream = Vec::new();
        write_varint(MAX_FRAME_LEN + 1, &mut stream);
        let mut decoder = FrameDecoder::new();
        decoder.feed(&stream);
        assert_eq!(
            decoder.next_frame(),
            Err(FrameError::Oversized {
                len: MAX_FRAME_LEN + 1,
                max: MAX_FRAME_LEN
            })
        );
    }

    #[test]
    fn a_runaway_varint_is_malformed() {
        // Eleven continuation bytes can never be a valid usize varint.
        let runaway = [0x80u8; 11];
        let mut decoder = FrameDecoder::new();
        decoder.feed(&runaway);
        assert_eq!(decoder.next_frame(), Err(FrameError::Malformed));
    }

    #[test]
    fn an_incomplete_varint_is_not_an_error() {
        // A single continuation byte: the length is not yet known.
        let mut decoder = FrameDecoder::new();
        decoder.feed(&[0x80]);
        assert_eq!(decoder.next_frame(), Ok(None));
        assert_eq!(decoder.buffered(), 1);
    }

    /// Regression: a tenth byte whose value does not fit in the final bits of a
    /// `usize` must be `Malformed`, not a debug-build shift panic (or, in release,
    /// a silently truncated length).
    #[test]
    fn a_prefix_overflowing_usize_is_malformed_not_a_panic() {
        // Nine 0xFF bytes contribute 63 set bits; a tenth byte of 0x7F adds seven
        // more, far past `usize::MAX`.
        let mut overflowing = [0xFFu8; 10];
        overflowing[9] = 0x7F;
        let mut decoder = FrameDecoder::new();
        decoder.feed(&overflowing);
        assert_eq!(decoder.next_frame(), Err(FrameError::Malformed));

        // The largest valid `usize` (all bits set) still decodes — then is
        // rejected as oversized — never a panic, never silently truncated.
        let mut maximal = Vec::new();
        write_varint(usize::MAX, &mut maximal);
        let mut decoder = FrameDecoder::new();
        decoder.feed(&maximal);
        assert!(matches!(
            decoder.next_frame(),
            Err(FrameError::Oversized { .. })
        ));
    }

    /// A malformed prefix poisons the decoder: later input is refused and the
    /// buffer cannot grow, so a desynchronized stream is bounded.
    #[test]
    fn a_malformed_prefix_poisons_the_decoder_without_growing() {
        let mut decoder = FrameDecoder::new();
        decoder.feed(&[0x80u8; 11]);
        assert_eq!(decoder.next_frame(), Err(FrameError::Malformed));
        assert_eq!(decoder.buffered(), 0, "the offending bytes are dropped");

        // Further feeds are ignored and the same error latches.
        for _ in 0..100 {
            decoder.feed(&[0u8; 64]);
            assert_eq!(decoder.next_frame(), Err(FrameError::Malformed));
        }
        assert_eq!(decoder.buffered(), 0, "no unbounded growth after poisoning");
    }

    /// `encode_frame` refuses a payload its own decoder would reject, so the
    /// encoder and decoder agree on the frame bound.
    #[test]
    fn encode_rejects_a_payload_past_the_max() {
        let oversized = vec![0u8; MAX_FRAME_LEN + 1];
        let mut out = Vec::new();
        assert_eq!(
            encode_frame(&oversized, &mut out),
            Err(FrameError::Oversized {
                len: MAX_FRAME_LEN + 1,
                max: MAX_FRAME_LEN
            })
        );
        assert!(out.is_empty(), "nothing is written on rejection");

        let mut ok = Vec::new();
        assert!(encode_frame(&[0u8; 8], &mut ok).is_ok());
    }
}
