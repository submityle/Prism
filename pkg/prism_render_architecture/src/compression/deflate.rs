//! Pure, allocation-light `DEFLATE` (`RFC 1951`) decompressor.
//!
//! This is the inner codec every `GDeflate`/`DirectStorage` tile ultimately
//! carries: `GDeflate` reorders a `DEFLATE` bitstream into warp-parallel
//! sub-streams, but each sub-stream is ordinary `DEFLATE`, so a correct,
//! self-contained inflate core is the foundation the tiled container builds on.
//!
//! The implementation follows `RFC 1951` directly and holds no device or OS
//! handle, so it works identically as a `CPU` fallback and as the golden
//! reference the `GPU` decoder is validated against. It supports all three
//! block types — stored, fixed-Huffman, and dynamic-Huffman — including
//! back-references that span block boundaries.
//!
//! Decoding uses the canonical bit-by-bit Huffman walk from Mark Adler's `puff`
//! reference decoder: symbols are kept sorted by `(code length, symbol)` and the
//! accumulated code is compared against the first code of each length. It trades
//! raw throughput for a tiny, obviously-correct table, which is the right
//! trade-off for a reference core.

use alloc::vec::Vec;

/// Largest Huffman code length `DEFLATE` permits.
const MAX_BITS: usize = 15;
/// Number of literal/length symbols (0..=285, plus the two reserved slots).
const MAX_LITLEN_SYMBOLS: usize = 288;
/// Number of distance symbols (0..=29, plus two reserved slots).
const MAX_DIST_SYMBOLS: usize = 32;

/// Base length for each length symbol 257..=285 (`RFC 1951` §3.2.5).
pub(crate) const LENGTH_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];
/// Extra bits read after each length symbol 257..=285.
pub(crate) const LENGTH_EXTRA: [u32; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
/// Base distance for each distance symbol 0..=29 (`RFC 1951` §3.2.5).
pub(crate) const DIST_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
/// Extra bits read after each distance symbol 0..=29.
pub(crate) const DIST_EXTRA: [u32; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];
/// Order the code-length code lengths are stored in (`RFC 1951` §3.2.7).
pub(crate) const CODE_LENGTH_ORDER: [usize; 19] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];

/// Reason a `DEFLATE` stream could not be decoded.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InflateError {
    /// The bitstream ended before the current block was fully decoded.
    UnexpectedEof,
    /// A block declared the reserved `BTYPE` value `11`.
    ReservedBlockType,
    /// A stored block's length did not match its one's-complement check word.
    StoredLengthMismatch,
    /// A Huffman code did not resolve to any symbol of any length.
    InvalidCode,
    /// A dynamic header described a code-length run with no symbol to repeat.
    InvalidRepeat,
    /// A length/distance symbol fell outside the values `RFC 1951` defines.
    InvalidSymbol,
    /// A back-reference pointed before the start of the output.
    DistanceTooFar,
    /// A decoded code-length table was malformed (over- or under-subscribed).
    InvalidCodeLengths,
    /// A decoded write would exceed the caller's output byte limit.
    OutputLimitExceeded,
    /// The allocator rejected an output buffer growth request.
    AllocationFailed,
    /// Whole bytes remained after the final block (not terminal padding bits).
    TrailingData,
}

/// Decompresses a raw `DEFLATE` stream (no `zlib`/`gzip` wrapper) into bytes.
///
/// # Errors
/// Returns an [`InflateError`] when the stream is truncated or violates
/// `RFC 1951`.
pub fn inflate(input: &[u8]) -> Result<Vec<u8>, InflateError> {
    inflate_bounded(input, usize::MAX)
}

/// Decompresses a raw `DEFLATE` stream with a hard decoded byte limit.
///
/// Every stored block, literal, and LZ77 copy is checked before reserving or
/// writing its output. The limit is not preallocated: capacity grows
/// geometrically, never requesting more than `max_output_bytes`, and allocation
/// failure is returned rather than panicking. Huffman tables use fixed-size
/// stack storage independent of input and output lengths. A zero limit accepts
/// valid empty streams but rejects the first decoded byte.
/// This decodes one stream prefix, as [`inflate`] does. Framed callers should
/// use [`inflate_bounded_with_consumed`] to reject whole trailing bytes.
///
/// # Errors
/// Returns [`InflateError::OutputLimitExceeded`] before any write that would
/// exceed the limit, [`InflateError::AllocationFailed`] on allocation failure,
/// or another [`InflateError`] for truncated or malformed streams.
pub fn inflate_bounded(input: &[u8], max_output_bytes: usize) -> Result<Vec<u8>, InflateError> {
    inflate_bounded_with_consumed(input, max_output_bytes).map(|(output, _)| output)
}

/// Inflates one raw stream prefix and returns `(output, consumed_bytes)`.
///
/// `consumed_bytes` includes the last partially used byte, but no following
/// whole byte. Unused bits in that last byte are legal padding, regardless of
/// their value. Like [`inflate_bounded`], this accepts a stream prefix; framed
/// callers must compare `consumed_bytes` to their declared compressed length.
///
/// # Errors
/// Returns the same bounded-write, allocation, and format errors as
/// [`inflate_bounded`].
pub fn inflate_bounded_with_consumed(
    input: &[u8],
    max_output_bytes: usize,
) -> Result<(Vec<u8>, usize), InflateError> {
    let mut reader = BitReader::new(input);
    let mut out = Vec::new();
    inflate_into(&mut reader, &mut out, max_output_bytes)?;
    Ok((out, reader.consumed_bytes()))
}

/// Validates exactly one bounded raw stream without allocating decoded bytes.
///
/// Returns the decoded byte count. This shares the inflate parser, checks LZ77
/// distances and all output growth, but counts references instead of expanding
/// them. It is suitable for validating GPU input, not for producing CPU output
/// or validating a decoded checksum. Whole trailing bytes are rejected; unused
/// bits in the final byte are accepted.
///
/// # Errors
/// Returns [`InflateError::TrailingData`] if any whole byte follows BFINAL, or
/// the same format and output-limit errors as [`inflate_bounded`].
pub fn validate_deflate_bounded(
    input: &[u8],
    max_output_bytes: usize,
) -> Result<usize, InflateError> {
    let mut reader = BitReader::new(input);
    let mut count = OutputCount(0);
    inflate_into(&mut reader, &mut count, max_output_bytes)?;
    if reader.consumed_bytes() != input.len() {
        return Err(InflateError::TrailingData);
    }
    Ok(count.len())
}

fn inflate_into(
    reader: &mut BitReader<'_>,
    out: &mut impl InflateOutput,
    max_output_bytes: usize,
) -> Result<(), InflateError> {
    loop {
        let final_block = reader.bit()? == 1;
        match reader.bits(2)? {
            0 => inflate_stored(reader, out, max_output_bytes)?,
            1 => inflate_block(
                reader,
                out,
                &Huffman::fixed_litlen(),
                &Huffman::fixed_dist(),
                max_output_bytes,
            )?,
            2 => {
                let (litlen, dist) = read_dynamic_tables(reader)?;
                inflate_block(reader, out, &litlen, &dist, max_output_bytes)?;
            }
            _ => return Err(InflateError::ReservedBlockType),
        }
        if final_block {
            break;
        }
    }
    Ok(())
}

// Both materializing and count-only decode use exactly the same block parser.
// Every push/copy is preceded by reserve_bounded for the entire write.
trait InflateOutput {
    fn len(&self) -> usize;
    fn reserve_bounded(&mut self, additional: usize, limit: usize) -> Result<(), InflateError>;
    fn push(&mut self, byte: u8);
    fn copy_match(&mut self, length: usize, distance: usize);
}

impl InflateOutput for Vec<u8> {
    fn len(&self) -> usize {
        self.len()
    }

    fn reserve_bounded(&mut self, additional: usize, limit: usize) -> Result<(), InflateError> {
        reserve_output(self, additional, limit)
    }

    fn push(&mut self, byte: u8) {
        self.push(byte);
    }

    fn copy_match(&mut self, length: usize, distance: usize) {
        let start = self.len() - distance;
        for offset in 0..length {
            self.push(self[start + offset]);
        }
    }
}

struct OutputCount(usize);

impl InflateOutput for OutputCount {
    fn len(&self) -> usize {
        self.0
    }

    fn reserve_bounded(&mut self, additional: usize, limit: usize) -> Result<(), InflateError> {
        self.0
            .checked_add(additional)
            .filter(|&length| length <= limit)
            .ok_or(InflateError::OutputLimitExceeded)?;
        Ok(())
    }

    fn push(&mut self, _byte: u8) {
        self.0 += 1;
    }

    fn copy_match(&mut self, length: usize, _distance: usize) {
        self.0 += length;
    }
}

/// Checks a complete write before reserving, including overlapping LZ77 copies.
fn reserve_output(out: &mut Vec<u8>, additional: usize, limit: usize) -> Result<(), InflateError> {
    let required = out
        .len()
        .checked_add(additional)
        .filter(|&length| length <= limit)
        .ok_or(InflateError::OutputLimitExceeded)?;
    if required > out.capacity() {
        let capacity = out
            .capacity()
            .saturating_mul(2)
            .max(256)
            .min(limit)
            .max(required);
        out.try_reserve_exact(capacity - out.len())
            .map_err(|_| InflateError::AllocationFailed)?;
    }
    Ok(())
}

/// `LSB`-first bit reader over a byte slice, matching `DEFLATE` bit packing.
struct BitReader<'a> {
    data: &'a [u8],
    /// Index of the next unconsumed byte.
    byte_pos: usize,
    /// Buffered bits not yet consumed, held in the low `bit_count` positions.
    bit_buf: u32,
    /// Number of valid bits currently in `bit_buf`.
    bit_count: u32,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            bit_buf: 0,
            bit_count: 0,
        }
    }

    /// Excludes any full bytes buffered ahead of the actual end of the stream.
    fn consumed_bytes(&self) -> usize {
        self.byte_pos - self.bit_count as usize / 8
    }

    /// Reads a single bit (`LSB`-first within each byte).
    fn bit(&mut self) -> Result<u32, InflateError> {
        self.bits(1)
    }

    /// Reads `count` bits (`count <= 24`), least-significant bit first.
    fn bits(&mut self, count: u32) -> Result<u32, InflateError> {
        while self.bit_count < count {
            let byte = *self
                .data
                .get(self.byte_pos)
                .ok_or(InflateError::UnexpectedEof)?;
            self.byte_pos += 1;
            self.bit_buf |= u32::from(byte) << self.bit_count;
            self.bit_count += 8;
        }
        let value = self.bit_buf & ((1 << count) - 1);
        self.bit_buf >>= count;
        self.bit_count -= count;
        Ok(value)
    }

    /// Discards buffered bits up to the next byte boundary.
    fn align_to_byte(&mut self) {
        let drop = self.bit_count % 8;
        self.bit_buf >>= drop;
        self.bit_count -= drop;
    }

    /// Reads a whole aligned byte, draining any buffered full byte first.
    fn read_aligned_byte(&mut self) -> Result<u8, InflateError> {
        if self.bit_count >= 8 {
            let byte = (self.bit_buf & 0xFF) as u8;
            self.bit_buf >>= 8;
            self.bit_count -= 8;
            return Ok(byte);
        }
        debug_assert_eq!(self.bit_count, 0);
        let byte = *self
            .data
            .get(self.byte_pos)
            .ok_or(InflateError::UnexpectedEof)?;
        self.byte_pos += 1;
        Ok(byte)
    }
}

/// Canonical Huffman decoder stored as per-length counts plus symbols sorted by
/// `(length, symbol)`, following the `puff` reference layout.
struct Huffman {
    /// `counts[len]` is the number of codes of bit length `len` (`counts[0]==0`).
    counts: [u16; MAX_BITS + 1],
    /// Symbols ordered by increasing code length, then increasing symbol value.
    symbols: [u16; MAX_LITLEN_SYMBOLS],
}

impl Huffman {
    /// Builds a decoder from one code length per symbol (0 = symbol absent).
    fn from_lengths(lengths: &[u16]) -> Result<Self, InflateError> {
        if lengths.len() > MAX_LITLEN_SYMBOLS {
            return Err(InflateError::InvalidCodeLengths);
        }
        let mut counts = [0u16; MAX_BITS + 1];
        for &len in lengths {
            let count = counts
                .get_mut(len as usize)
                .ok_or(InflateError::InvalidCodeLengths)?;
            *count += 1;
        }
        counts[0] = 0;

        // Reject over-subscribed tables (more codes than a length can hold).
        let mut left: i32 = 1;
        for &count in &counts[1..=MAX_BITS] {
            left <<= 1;
            left -= i32::from(count);
            if left < 0 {
                return Err(InflateError::InvalidCodeLengths);
            }
        }

        let mut offsets = [0u16; MAX_BITS + 2];
        for len in 1..=MAX_BITS {
            offsets[len + 1] = offsets[len] + counts[len];
        }
        let mut symbols = [0u16; MAX_LITLEN_SYMBOLS];
        for (symbol, &len) in lengths.iter().enumerate() {
            if len != 0 {
                symbols[offsets[len as usize] as usize] = symbol as u16;
                offsets[len as usize] += 1;
            }
        }
        Ok(Self { counts, symbols })
    }

    /// Decodes one symbol by walking lengths until the accumulated code lands in
    /// a length's assigned range (`puff` algorithm).
    fn decode(&self, reader: &mut BitReader<'_>) -> Result<u16, InflateError> {
        let mut code: i32 = 0;
        let mut first: i32 = 0;
        let mut index: i32 = 0;
        for len in 1..=MAX_BITS {
            code |= reader.bit()? as i32;
            let count = i32::from(self.counts[len]);
            if code - first < count {
                return Ok(self.symbols[(index + (code - first)) as usize]);
            }
            index += count;
            first += count;
            first <<= 1;
            code <<= 1;
        }
        Err(InflateError::InvalidCode)
    }

    /// Fixed literal/length table from `RFC 1951` §3.2.6.
    fn fixed_litlen() -> Self {
        let mut lengths = [0u16; MAX_LITLEN_SYMBOLS];
        lengths[0..=143].fill(8);
        lengths[144..=255].fill(9);
        lengths[256..=279].fill(7);
        lengths[280..=287].fill(8);
        Self::from_lengths(&lengths).expect("fixed litlen table is well-formed")
    }

    /// Fixed distance table from `RFC 1951` §3.2.6 (all codes are 5 bits).
    fn fixed_dist() -> Self {
        let lengths = [5u16; MAX_DIST_SYMBOLS];
        Self::from_lengths(&lengths).expect("fixed distance table is well-formed")
    }
}

/// Copies a stored (uncompressed) block verbatim (`RFC 1951` §3.2.4).
fn inflate_stored(
    reader: &mut BitReader<'_>,
    out: &mut impl InflateOutput,
    max_output_bytes: usize,
) -> Result<(), InflateError> {
    reader.align_to_byte();
    let len =
        u16::from(reader.read_aligned_byte()?) | (u16::from(reader.read_aligned_byte()?) << 8);
    let nlen =
        u16::from(reader.read_aligned_byte()?) | (u16::from(reader.read_aligned_byte()?) << 8);
    if len != !nlen {
        return Err(InflateError::StoredLengthMismatch);
    }
    out.reserve_bounded(len as usize, max_output_bytes)?;
    for _ in 0..len {
        out.push(reader.read_aligned_byte()?);
    }
    Ok(())
}

/// Reads the dynamic-Huffman header and returns its literal/length and distance
/// decoders (`RFC 1951` §3.2.7).
fn read_dynamic_tables(reader: &mut BitReader<'_>) -> Result<(Huffman, Huffman), InflateError> {
    let hlit = reader.bits(5)? as usize + 257;
    let hdist = reader.bits(5)? as usize + 1;
    let hclen = reader.bits(4)? as usize + 4;

    let mut code_length_lengths = [0u16; 19];
    for &slot in CODE_LENGTH_ORDER.iter().take(hclen) {
        code_length_lengths[slot] = reader.bits(3)? as u16;
    }
    let code_length_table = Huffman::from_lengths(&code_length_lengths)?;

    let total = hlit + hdist;
    let mut storage = [0u16; MAX_LITLEN_SYMBOLS + MAX_DIST_SYMBOLS];
    let lengths = &mut storage[..total];
    let mut i = 0;
    while i < total {
        let symbol = code_length_table.decode(reader)?;
        match symbol {
            0..=15 => {
                lengths[i] = symbol;
                i += 1;
            }
            16 => {
                if i == 0 {
                    return Err(InflateError::InvalidRepeat);
                }
                let prev = lengths[i - 1];
                let repeat = reader.bits(2)? as usize + 3;
                for _ in 0..repeat {
                    if i >= total {
                        return Err(InflateError::InvalidRepeat);
                    }
                    lengths[i] = prev;
                    i += 1;
                }
            }
            17 => {
                let repeat = reader.bits(3)? as usize + 3;
                i = fill_zeros(lengths, i, repeat, total)?;
            }
            18 => {
                let repeat = reader.bits(7)? as usize + 11;
                i = fill_zeros(lengths, i, repeat, total)?;
            }
            _ => return Err(InflateError::InvalidSymbol),
        }
    }

    let litlen = Huffman::from_lengths(&lengths[..hlit])?;
    let dist = Huffman::from_lengths(&lengths[hlit..])?;
    Ok((litlen, dist))
}

/// Writes `repeat` zero code lengths starting at `i`, bounds-checked.
fn fill_zeros(
    lengths: &mut [u16],
    mut i: usize,
    repeat: usize,
    total: usize,
) -> Result<usize, InflateError> {
    for _ in 0..repeat {
        if i >= total {
            return Err(InflateError::InvalidRepeat);
        }
        lengths[i] = 0;
        i += 1;
    }
    Ok(i)
}

/// Decodes one compressed block body with the supplied Huffman tables.
fn inflate_block(
    reader: &mut BitReader<'_>,
    out: &mut impl InflateOutput,
    litlen: &Huffman,
    dist: &Huffman,
    max_output_bytes: usize,
) -> Result<(), InflateError> {
    loop {
        let symbol = litlen.decode(reader)?;
        match symbol {
            0..=255 => {
                out.reserve_bounded(1, max_output_bytes)?;
                out.push(symbol as u8);
            }
            256 => return Ok(()),
            257..=285 => {
                let index = (symbol - 257) as usize;
                let length =
                    LENGTH_BASE[index] as usize + reader.bits(LENGTH_EXTRA[index])? as usize;
                let dist_symbol = dist.decode(reader)? as usize;
                if dist_symbol >= DIST_BASE.len() {
                    return Err(InflateError::InvalidSymbol);
                }
                let distance = DIST_BASE[dist_symbol] as usize
                    + reader.bits(DIST_EXTRA[dist_symbol])? as usize;
                if distance == 0 || distance > out.len() {
                    return Err(InflateError::DistanceTooFar);
                }
                out.reserve_bounded(length, max_output_bytes)?;
                out.copy_match(length, distance);
            }
            _ => return Err(InflateError::InvalidSymbol),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use flate2::write::DeflateEncoder;
    use flate2::Compression;
    use std::io::Write;

    fn deflate(data: &[u8], level: u32) -> Vec<u8> {
        let mut encoder = DeflateEncoder::new(Vec::new(), Compression::new(level));
        encoder.write_all(data).unwrap();
        encoder.finish().unwrap()
    }

    fn round_trip(data: &[u8]) {
        for level in [0, 1, 6, 9] {
            let compressed = deflate(data, level);
            let restored = inflate(&compressed).expect("inflate must succeed");
            assert_eq!(restored, data, "mismatch at compression level {level}");
        }
    }

    #[test]
    fn empty_input_round_trips() {
        round_trip(&[]);
    }

    #[test]
    fn short_literal_round_trips() {
        round_trip(b"hello, deflate");
    }

    #[test]
    fn highly_repetitive_data_uses_back_references() {
        let data = vec![0xABu8; 100_000];
        round_trip(&data);
    }

    #[test]
    fn repeated_phrase_round_trips() {
        let mut data = Vec::new();
        for _ in 0..5000 {
            data.extend_from_slice(b"the quick brown fox ");
        }
        round_trip(&data);
    }

    #[test]
    fn pseudo_random_bytes_round_trip() {
        // Deterministic LCG so the test needs no rng dependency.
        let mut state: u32 = 0x1234_5678;
        let mut data = vec![0u8; 65_537];
        for byte in &mut data {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *byte = (state >> 24) as u8;
        }
        round_trip(&data);
    }

    #[test]
    fn structured_text_round_trips() {
        let mut data = Vec::new();
        for i in 0..20_000u32 {
            data.extend_from_slice(format!("line {i}: value={}\n", i * 7 % 251).as_bytes());
        }
        round_trip(&data);
    }

    #[test]
    fn stored_block_round_trips() {
        // Level 0 forces stored (uncompressed) blocks.
        let data: Vec<u8> = (0..40_000u32).map(|i| (i % 256) as u8).collect();
        let compressed = deflate(&data, 0);
        assert_eq!(inflate(&compressed).unwrap(), data);
    }

    #[test]
    fn truncated_stream_errors() {
        let compressed = deflate(b"some representative payload bytes", 9);
        let truncated = &compressed[..compressed.len() / 2];
        assert_eq!(inflate(truncated), Err(InflateError::UnexpectedEof));
    }

    #[test]
    fn reserved_block_type_errors() {
        // A single byte whose low three bits are BFINAL=1, BTYPE=11.
        assert_eq!(
            inflate(&[0b0000_0111]),
            Err(InflateError::ReservedBlockType)
        );
    }

    #[test]
    fn stored_length_mismatch_errors() {
        // BFINAL=1, BTYPE=00, then LEN=1 with a wrong NLEN check word.
        let stream = [0x01, 0x01, 0x00, 0x00, 0x00, 0x42];
        assert_eq!(inflate(&stream), Err(InflateError::StoredLengthMismatch));
    }

    #[test]
    fn bounded_decode_accepts_exact_and_generous_limits_for_all_encoder_levels() {
        for data in [
            Vec::new(),
            b"literal bytes".to_vec(),
            vec![0xAB; 100_000],
            (0..70_001u32)
                .map(|value| value.wrapping_mul(37) as u8)
                .collect(),
        ] {
            for level in [0, 1, 6, 9] {
                let encoded = deflate(&data, level);
                assert_eq!(inflate_bounded(&encoded, data.len()).unwrap(), data);
                assert_eq!(inflate_bounded(&encoded, usize::MAX).unwrap(), data);
                if !data.is_empty() {
                    assert_eq!(
                        inflate_bounded(&encoded, data.len() - 1),
                        Err(InflateError::OutputLimitExceeded),
                        "level={level}, length={}",
                        data.len(),
                    );
                    assert_eq!(
                        inflate_bounded(&encoded, 0),
                        Err(InflateError::OutputLimitExceeded),
                    );
                }
            }
        }
    }

    #[test]
    fn stored_block_is_rejected_before_reserving_or_reading_payload() {
        // A complete header advertising three bytes, without any payload.
        let encoded = [0x01, 0x03, 0x00, 0xFC, 0xFF];
        let mut reader = BitReader::new(&encoded);
        assert_eq!(reader.bits(3).unwrap(), 1);
        let mut out = Vec::new();
        assert_eq!(
            inflate_stored(&mut reader, &mut out, 2),
            Err(InflateError::OutputLimitExceeded),
        );
        assert_eq!(out.capacity(), 0);
        assert!(out.is_empty());
    }

    /// Independent fixed block: literal 'A', length 258 / distance 1, EOB.
    fn overlapping_copy_stream() -> Vec<u8> {
        let mut bits = vec![true, true, false]; // BFINAL=1, BTYPE=01.
        for (code, count) in [(0x71u16, 8), (0xC5, 8), (0, 5), (0, 7)] {
            for shift in (0..count).rev() {
                bits.push((code >> shift) & 1 != 0);
            }
        }
        let mut bytes = vec![0u8; bits.len().div_ceil(8)];
        for (offset, bit) in bits.into_iter().enumerate() {
            bytes[offset / 8] |= u8::from(bit) << (offset % 8);
        }
        bytes
    }

    #[test]
    fn fixed_literal_limit_is_checked_before_the_first_write() {
        let encoded = overlapping_copy_stream();
        let mut reader = BitReader::new(&encoded);
        assert_eq!(reader.bits(3).unwrap(), 3);
        let mut out = Vec::new();
        assert_eq!(
            inflate_block(
                &mut reader,
                &mut out,
                &Huffman::fixed_litlen(),
                &Huffman::fixed_dist(),
                0
            ),
            Err(InflateError::OutputLimitExceeded),
        );
        assert!(out.is_empty());
        assert_eq!(out.capacity(), 0);
    }

    #[test]
    fn overlapping_lz77_copy_is_rejected_as_a_whole_before_writing() {
        let encoded = overlapping_copy_stream();
        assert_eq!(inflate_bounded(&encoded, 259).unwrap(), vec![b'A'; 259]);
        let mut reader = BitReader::new(&encoded);
        reader.bits(3).unwrap();
        let mut out = Vec::new();
        assert_eq!(
            inflate_block(
                &mut reader,
                &mut out,
                &Huffman::fixed_litlen(),
                &Huffman::fixed_dist(),
                258
            ),
            Err(InflateError::OutputLimitExceeded),
        );
        assert_eq!(out, b"A");
        assert!(out.capacity() <= 258);
    }

    #[test]
    fn output_limit_is_shared_across_stored_and_huffman_blocks() {
        let mut encoded = vec![0x00, 0x01, 0x00, 0xFE, 0xFF, b'X'];
        encoded.extend_from_slice(&overlapping_copy_stream());
        let mut expected = vec![b'X'];
        expected.extend_from_slice(&vec![b'A'; 259]);
        assert_eq!(inflate_bounded(&encoded, 260).unwrap(), expected);
        assert_eq!(
            inflate_bounded(&encoded, 259),
            Err(InflateError::OutputLimitExceeded)
        );
    }

    #[test]
    fn dynamic_huffman_output_limit_applies_before_full_decode() {
        let data = vec![0xAB; 100_000];
        let encoded = deflate(&data, 9);
        assert_eq!((encoded[0] >> 1) & 3, 2, "fixture must use dynamic Huffman");
        assert_eq!(inflate_bounded(&encoded, data.len()).unwrap(), data);
        assert_eq!(
            inflate_bounded(&encoded, 1024),
            Err(InflateError::OutputLimitExceeded)
        );
    }

    #[test]
    fn checked_reservation_rejects_overflow_and_allocation_failure_without_growth() {
        let mut out = vec![1];
        let capacity = out.capacity();
        assert_eq!(
            reserve_output(&mut out, usize::MAX, usize::MAX),
            Err(InflateError::OutputLimitExceeded)
        );
        assert_eq!(out, [1]);
        assert_eq!(out.capacity(), capacity);
        let mut out = Vec::new();
        assert_eq!(
            reserve_output(&mut out, usize::MAX, usize::MAX),
            Err(InflateError::AllocationFailed)
        );
        assert_eq!(out.capacity(), 0);
    }

    #[test]
    fn geometric_growth_never_requests_more_than_the_output_limit() {
        let mut out = Vec::new();
        let mut growths = 0;
        for _ in 0..65_537 {
            let previous = out.capacity();
            reserve_output(&mut out, 1, 65_537).unwrap();
            growths += usize::from(out.capacity() != previous);
            assert!(out.capacity() <= 65_537);
            out.push(7);
        }
        assert!(growths <= 10, "literal streams must not allocate per byte");
    }

    #[test]
    fn huffman_table_storage_rejects_out_of_range_lengths_without_panicking() {
        assert!(matches!(
            Huffman::from_lengths(&[16]),
            Err(InflateError::InvalidCodeLengths)
        ));
        assert!(matches!(
            Huffman::from_lengths(&[0; MAX_LITLEN_SYMBOLS + 1]),
            Err(InflateError::InvalidCodeLengths)
        ));
    }

    fn consumed_fixtures() -> Vec<(Vec<u8>, Vec<u8>)> {
        let dynamic_data = vec![0xAB; 70_000];
        let dynamic = deflate(&dynamic_data, 9);
        assert_eq!((dynamic[0] >> 1) & 3, 2);
        let fixed = overlapping_copy_stream();
        // Fixed block ends at bit 31; a non-final stored header straddles that
        // byte boundary, followed by a byte-aligned final dynamic block.
        let mut mixed = fixed.clone();
        mixed[0] &= !1;
        mixed.push(0);
        mixed.extend_from_slice(&[1, 0, 0xFE, 0xFF, b'X']);
        mixed.extend_from_slice(&dynamic);
        let mut mixed_data = vec![b'A'; 259];
        mixed_data.push(b'X');
        mixed_data.extend_from_slice(&dynamic_data);
        vec![
            (vec![1, 0, 0, 0xFF, 0xFF], Vec::new()),
            (vec![3, 0], Vec::new()),
            (vec![1, 3, 0, 0xFC, 0xFF, b'a', b'b', b'c'], b"abc".to_vec()),
            (fixed, vec![b'A'; 259]),
            (dynamic, dynamic_data),
            (mixed, mixed_data),
        ]
    }

    #[test]
    fn exact_consumed_excludes_trailing_bytes_for_empty_stored_fixed_dynamic_and_mixed() {
        for (raw, data) in consumed_fixtures() {
            let (decoded, consumed) = inflate_bounded_with_consumed(&raw, data.len()).unwrap();
            assert_eq!(decoded, data);
            assert_eq!(consumed, raw.len());
            assert_eq!(validate_deflate_bounded(&raw, data.len()), Ok(data.len()));
            for tail in [&[0][..], &[0xFF], &[3, 0], raw.as_slice()] {
                let mut extended = raw.clone();
                extended.extend_from_slice(tail);
                let (decoded, consumed) =
                    inflate_bounded_with_consumed(&extended, data.len()).unwrap();
                assert_eq!(decoded, data);
                assert_eq!(consumed, raw.len());
                assert_eq!(inflate_bounded(&extended, data.len()).unwrap(), data);
                assert_eq!(inflate(&extended).unwrap(), data);
                assert_eq!(
                    validate_deflate_bounded(&extended, data.len()),
                    Err(InflateError::TrailingData)
                );
            }
            if !data.is_empty() {
                assert_eq!(
                    validate_deflate_bounded(&raw, data.len() - 1),
                    Err(InflateError::OutputLimitExceeded)
                );
            }
            for length in 0..raw.len() {
                assert!(inflate_bounded_with_consumed(&raw[..length], data.len()).is_err());
                assert!(validate_deflate_bounded(&raw[..length], data.len()).is_err());
            }
        }
    }

    #[test]
    fn consumed_accepts_nonzero_terminal_padding_but_not_an_extra_whole_byte() {
        for raw in [&[3, 0xFC][..], &[0xF9, 0, 0, 0xFF, 0xFF]] {
            assert_eq!(
                inflate_bounded_with_consumed(raw, 0),
                Ok((Vec::new(), raw.len()))
            );
            assert_eq!(validate_deflate_bounded(raw, 0), Ok(0));
            let mut extended = raw.to_vec();
            extended.push(0);
            assert_eq!(
                validate_deflate_bounded(&extended, 0),
                Err(InflateError::TrailingData)
            );
        }
        // Simulate lookahead: whole buffered bytes must not count as consumed.
        let mut reader = BitReader::new(&[0; 4]);
        reader.bits(24).unwrap();
        reader.bit_buf = 0;
        reader.bit_count = 17;
        assert_eq!(reader.consumed_bytes(), 1);
    }

    #[test]
    fn exact_consumed_handles_every_final_bit_alignment() {
        // Each fixed literal 144 occupies nine bits, so these cover all eight
        // final-bit offsets, including a byte-aligned EOB with no padding.
        for literals in 0..8 {
            let mut bits = vec![true, true, false];
            for _ in 0..literals {
                bits.extend((0..9).rev().map(|shift| 0x190u16 & (1 << shift) != 0));
            }
            bits.extend([false; 7]); // Fixed EOB.
            let mut raw = vec![0u8; bits.len().div_ceil(8)];
            for (offset, &bit) in bits.iter().enumerate() {
                raw[offset / 8] |= u8::from(bit) << (offset % 8);
            }
            if !bits.len().is_multiple_of(8) {
                *raw.last_mut().unwrap() |= 0xFFu8 << (bits.len() % 8);
            }
            let data = vec![144; literals];
            let consumed = raw.len();
            assert_eq!(
                inflate_bounded_with_consumed(&raw, literals),
                Ok((data.clone(), consumed))
            );
            assert_eq!(validate_deflate_bounded(&raw, literals), Ok(literals));
            raw.extend_from_slice(&[0, 0xFF, 3, 0]);
            assert_eq!(
                inflate_bounded_with_consumed(&raw, literals),
                Ok((data, consumed))
            );
            assert_eq!(
                validate_deflate_bounded(&raw, literals),
                Err(InflateError::TrailingData)
            );
        }
    }

    #[test]
    fn count_only_validation_preserves_cross_block_history_and_invalid_distance_errors() {
        // Non-final stored 'X', then fixed length 258/distance 1 with no literal.
        let mut bits = vec![true, true, false];
        for (code, width) in [(0xC5u16, 8), (0, 5), (0, 7)] {
            bits.extend((0..width).rev().map(|shift| code & (1 << shift) != 0));
        }
        let mut fixed = vec![0u8; bits.len().div_ceil(8)];
        for (offset, bit) in bits.into_iter().enumerate() {
            fixed[offset / 8] |= u8::from(bit) << (offset % 8);
        }
        assert_eq!(
            validate_deflate_bounded(&fixed, 259),
            Err(InflateError::DistanceTooFar)
        );
        let mut mixed = vec![0, 1, 0, 0xFE, 0xFF, b'X'];
        mixed.extend_from_slice(&fixed);
        assert_eq!(validate_deflate_bounded(&mixed, 259), Ok(259));
        assert_eq!(
            inflate_bounded_with_consumed(&mixed, 259),
            Ok((vec![b'X'; 259], mixed.len()))
        );
        assert_eq!(
            validate_deflate_bounded(&mixed, 258),
            Err(InflateError::OutputLimitExceeded)
        );
        let mut count = OutputCount(usize::MAX);
        assert_eq!(
            count.reserve_bounded(1, usize::MAX),
            Err(InflateError::OutputLimitExceeded)
        );
        assert_eq!(count.len(), usize::MAX);
    }

    #[test]
    fn count_only_and_materialized_decode_agree_on_malformed_streams() {
        let mut state = 0x1234_5678u32;
        for length in 0..48 {
            for _ in 0..64 {
                let raw: Vec<u8> = (0..length)
                    .map(|_| {
                        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                        (state >> 24) as u8
                    })
                    .collect();
                let expected = inflate_bounded_with_consumed(&raw, 32).and_then(|(bytes, used)| {
                    if used == raw.len() {
                        Ok(bytes.len())
                    } else {
                        Err(InflateError::TrailingData)
                    }
                });
                assert_eq!(validate_deflate_bounded(&raw, 32), expected, "{raw:?}");
            }
        }
    }
}
