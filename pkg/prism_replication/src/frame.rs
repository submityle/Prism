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
    /// A length prefix was not a valid LEB128 `usize` (more than ten bytes).
    Malformed,
}

/// Append `payload` to `out` as a LEB128-length-prefixed frame.
pub fn encode_frame(payload: &[u8], out: &mut Vec<u8>) {
    write_varint(payload.len(), out);
    out.extend_from_slice(payload);
}

/// An incremental frame splitter over a byte stream.
///
/// Bytes accumulate in an internal buffer; [`next_frame`](Self::next_frame)
/// yields each complete frame in turn, leaving a partial frame buffered for the
/// next feed.
#[derive(Clone, Debug, Default)]
pub struct FrameDecoder {
    buffer: Vec<u8>,
}

impl FrameDecoder {
    /// A fresh decoder with an empty buffer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Buffer `bytes` from the stream.
    pub fn feed(&mut self, bytes: &[u8]) {
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
    /// [`FrameError::Malformed`] for an invalid length prefix. On error the
    /// offending bytes stay buffered, so a caller can log and close the stream.
    pub fn next_frame(&mut self) -> Result<Option<Vec<u8>>, FrameError> {
        let Some((len, prefix)) = read_varint(&self.buffer)? else {
            return Ok(None);
        };
        if len > MAX_FRAME_LEN {
            return Err(FrameError::Oversized {
                len,
                max: MAX_FRAME_LEN,
            });
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
fn read_varint(bytes: &[u8]) -> Result<Option<(usize, usize)>, FrameError> {
    let mut value: usize = 0;
    let mut shift: u32 = 0;
    for (index, &byte) in bytes.iter().enumerate() {
        // A `usize` varint is at most ceil(bits / 7) bytes; the tenth byte may
        // only contribute the final bit(s).
        if index >= 10 {
            return Err(FrameError::Malformed);
        }
        if shift >= usize::BITS {
            return Err(FrameError::Malformed);
        }
        value |= usize::from(byte & 0x7F) << shift;
        if byte & 0x80 == 0 {
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
        encode_frame(b"hello", &mut stream);
        encode_frame(&[], &mut stream);
        encode_frame(&[0u8; 300], &mut stream);

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
        encode_frame(b"abcdef", &mut stream);

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
}
