//! Pure, allocation-light `DEFLATE` (`RFC 1951`) decompressor.
//!
//! This is the inner codec every `GDeflate`/`DirectStorage` tile ultimately
//! carries: `GDeflate` reorders a `DEFLATE` bitstream into warp-parallel
//! sub-streams, but each sub-stream is ordinary `DEFLATE`, so a correct,
//! self-contained inflate core is the foundation the tiled container builds on.
//!
//! The implementation follows `RFC 1951` directly and holds no device or OS
//! handle, so it works identically as a `CPU` fallback and as the golden
//! reference a future `GPU` decoder is validated against. It supports all three
//! block types — stored, fixed-Huffman, and dynamic-Huffman — including
//! back-references that span block boundaries.
//!
//! Decoding uses the canonical bit-by-bit Huffman walk from Mark Adler's `puff`
//! reference decoder: symbols are kept sorted by `(code length, symbol)` and the
//! accumulated code is compared against the first code of each length. It trades
//! raw throughput for a tiny, obviously-correct table, which is the right
//! trade-off for a reference core.

use alloc::vec;
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
const CODE_LENGTH_ORDER: [usize; 19] = [
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
}

/// Decompresses a raw `DEFLATE` stream (no `zlib`/`gzip` wrapper) into bytes.
///
/// # Errors
/// Returns an [`InflateError`] when the stream is truncated or violates
/// `RFC 1951`.
pub fn inflate(input: &[u8]) -> Result<Vec<u8>, InflateError> {
    let mut reader = BitReader::new(input);
    let mut out = Vec::new();
    loop {
        let final_block = reader.bit()? == 1;
        match reader.bits(2)? {
            0 => inflate_stored(&mut reader, &mut out)?,
            1 => inflate_block(
                &mut reader,
                &mut out,
                &Huffman::fixed_litlen(),
                &Huffman::fixed_dist(),
            )?,
            2 => {
                let (litlen, dist) = read_dynamic_tables(&mut reader)?;
                inflate_block(&mut reader, &mut out, &litlen, &dist)?;
            }
            _ => return Err(InflateError::ReservedBlockType),
        }
        if final_block {
            break;
        }
    }
    Ok(out)
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
    symbols: Vec<u16>,
}

impl Huffman {
    /// Builds a decoder from one code length per symbol (0 = symbol absent).
    fn from_lengths(lengths: &[u16]) -> Result<Self, InflateError> {
        let mut counts = [0u16; MAX_BITS + 1];
        for &len in lengths {
            counts[len as usize] += 1;
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
        let total = offsets[MAX_BITS + 1] as usize;
        let mut symbols = vec![0u16; total];
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
fn inflate_stored(reader: &mut BitReader<'_>, out: &mut Vec<u8>) -> Result<(), InflateError> {
    reader.align_to_byte();
    let len =
        u16::from(reader.read_aligned_byte()?) | (u16::from(reader.read_aligned_byte()?) << 8);
    let nlen =
        u16::from(reader.read_aligned_byte()?) | (u16::from(reader.read_aligned_byte()?) << 8);
    if len != !nlen {
        return Err(InflateError::StoredLengthMismatch);
    }
    out.reserve(len as usize);
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
    let mut lengths = vec![0u16; total];
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
                i = fill_zeros(&mut lengths, i, repeat, total)?;
            }
            18 => {
                let repeat = reader.bits(7)? as usize + 11;
                i = fill_zeros(&mut lengths, i, repeat, total)?;
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
    out: &mut Vec<u8>,
    litlen: &Huffman,
    dist: &Huffman,
) -> Result<(), InflateError> {
    loop {
        let symbol = litlen.decode(reader)?;
        match symbol {
            0..=255 => out.push(symbol as u8),
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
                let start = out.len() - distance;
                for offset in 0..length {
                    let byte = out[start + offset];
                    out.push(byte);
                }
            }
            _ => return Err(InflateError::InvalidSymbol),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
}
