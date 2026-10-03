//! Minimal, correct `DEFLATE` (`RFC 1951`) *encoder* used only to produce the
//! per-tile payloads the `GDeflate` container carries.
//!
//! The repo already ships a complete inflate core in [`super::deflate`]; this
//! module is its write-side counterpart so the `GDeflate` codec can be
//! round-trip verified end to end on the `CPU` without pulling in an external
//! encoder. It emits standards-compliant streams that [`super::deflate::inflate`]
//! decodes, so it reuses that core as the golden reference rather than
//! duplicating any Huffman *decode* logic.
//!
//! Two block strategies are produced and the smaller result is kept:
//!
//! * a single fixed-Huffman block (`BTYPE=01`) fed by a greedy `LZ77`
//!   hash-chain match finder, and
//! * stored (uncompressed) blocks (`BTYPE=00`) as an anti-expansion fallback
//!   for incompressible input.
//!
//! The length/distance base and extra-bit tables are the shared `RFC 1951`
//! §3.2.5 constants re-exported from [`super::deflate`], so there is a single
//! source of truth for them.

use alloc::vec;
use alloc::vec::Vec;

use super::deflate::{DIST_BASE, DIST_EXTRA, LENGTH_BASE, LENGTH_EXTRA};

/// Shortest back-reference `RFC 1951` encodes.
const MIN_MATCH: usize = 3;
/// Longest back-reference `RFC 1951` encodes.
const MAX_MATCH: usize = 258;
/// Sliding-window size (maximum back-reference distance).
const WINDOW: usize = 32_768;
/// Largest stored-block payload (`LEN` is a `u16`).
const MAX_STORED: usize = 65_535;

/// Number of hash-chain buckets (`2^HASH_BITS`).
const HASH_BITS: u32 = 15;
/// Mask selecting the low `HASH_BITS` of a hash.
const HASH_MASK: usize = (1 << HASH_BITS) - 1;
/// Maximum chain positions examined per match search.
const MAX_CHAIN: usize = 256;

/// Encodes `input` as a raw `DEFLATE` stream (no `zlib`/`gzip` wrapper).
///
/// Returns whichever of the fixed-Huffman or stored encodings is smaller, so
/// the result never expands incompressible input by more than the stored-block
/// framing overhead.
pub(crate) fn deflate(input: &[u8]) -> Vec<u8> {
    let fixed = deflate_fixed(input);
    let stored = deflate_stored(input);
    if stored.len() < fixed.len() {
        stored
    } else {
        fixed
    }
}

/// `LSB`-first bit writer matching `DEFLATE` bit packing.
struct BitWriter {
    out: Vec<u8>,
    /// Pending bits held in the low `bit_count` positions.
    bit_buf: u32,
    /// Number of valid pending bits in `bit_buf`.
    bit_count: u32,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            out: Vec::new(),
            bit_buf: 0,
            bit_count: 0,
        }
    }

    /// Writes `count` bits (`count <= 24`), least-significant bit first.
    fn write_bits(&mut self, value: u32, count: u32) {
        self.bit_buf |= (value & ((1u32 << count) - 1)) << self.bit_count;
        self.bit_count += count;
        while self.bit_count >= 8 {
            self.out.push((self.bit_buf & 0xFF) as u8);
            self.bit_buf >>= 8;
            self.bit_count -= 8;
        }
    }

    /// Writes a canonical Huffman `code` of `len` bits, which `DEFLATE` packs
    /// most-significant bit first; the bits are reversed so the `LSB`-first
    /// writer reproduces that order.
    fn write_code(&mut self, code: u16, len: u32) {
        let mut reversed = 0u32;
        for i in 0..len {
            reversed |= ((u32::from(code) >> i) & 1) << (len - 1 - i);
        }
        self.write_bits(reversed, len);
    }

    /// Flushes any partial byte, zero-padding the high bits to a boundary.
    fn align_to_byte(&mut self) {
        if self.bit_count > 0 {
            self.out.push((self.bit_buf & 0xFF) as u8);
            self.bit_buf = 0;
            self.bit_count = 0;
        }
    }

    /// Appends raw bytes; only valid immediately after [`Self::align_to_byte`].
    fn write_aligned_bytes(&mut self, bytes: &[u8]) {
        debug_assert_eq!(self.bit_count, 0);
        self.out.extend_from_slice(bytes);
    }

    fn finish(mut self) -> Vec<u8> {
        self.align_to_byte();
        self.out
    }
}

/// Encodes `input` as stored (uncompressed) `DEFLATE` blocks.
fn deflate_stored(input: &[u8]) -> Vec<u8> {
    let mut writer = BitWriter::new();
    if input.is_empty() {
        emit_stored_block(&mut writer, &[], true);
        return writer.finish();
    }
    let mut offset = 0;
    while offset < input.len() {
        let end = (offset + MAX_STORED).min(input.len());
        let chunk = &input[offset..end];
        let last = end == input.len();
        emit_stored_block(&mut writer, chunk, last);
        offset = end;
    }
    writer.finish()
}

/// Emits one stored block (`BTYPE=00`) carrying `chunk` verbatim.
fn emit_stored_block(writer: &mut BitWriter, chunk: &[u8], last: bool) {
    writer.write_bits(u32::from(last), 1);
    writer.write_bits(0, 2);
    writer.align_to_byte();
    let len = chunk.len() as u16;
    let nlen = !len;
    writer.write_aligned_bytes(&len.to_le_bytes());
    writer.write_aligned_bytes(&nlen.to_le_bytes());
    writer.write_aligned_bytes(chunk);
}

/// Encodes `input` as a single fixed-Huffman block (`BTYPE=01`) with greedy
/// `LZ77` matching.
fn deflate_fixed(input: &[u8]) -> Vec<u8> {
    let mut writer = BitWriter::new();
    writer.write_bits(1, 1); // BFINAL
    writer.write_bits(1, 2); // BTYPE = 01 (fixed Huffman)

    let len = input.len();
    let mut head = vec![usize::MAX; HASH_MASK + 1];
    let mut prev = vec![usize::MAX; len];

    let mut pos = 0;
    while pos < len {
        let mut best_len = 0;
        let mut best_dist = 0;
        if pos + MIN_MATCH <= len {
            let hash = hash3(&input[pos..]);
            let (mlen, mdist) = find_match(input, pos, head[hash], &prev);
            best_len = mlen;
            best_dist = mdist;
            insert_hash(&mut head, &mut prev, hash, pos);
        }

        if best_len >= MIN_MATCH {
            emit_match(&mut writer, best_len, best_dist);
            // Insert hashes for the interior of the match so later positions
            // can still reference it.
            let run_end = pos + best_len;
            let mut p = pos + 1;
            while p < run_end {
                if p + MIN_MATCH <= len {
                    let hash = hash3(&input[p..]);
                    insert_hash(&mut head, &mut prev, hash, p);
                }
                p += 1;
            }
            pos = run_end;
        } else {
            emit_literal(&mut writer, input[pos]);
            pos += 1;
        }
    }

    emit_end_of_block(&mut writer);
    writer.finish()
}

/// Hashes the three bytes at the slice start into a bucket index.
fn hash3(bytes: &[u8]) -> usize {
    let a = u32::from(bytes[0]);
    let b = u32::from(bytes[1]);
    let c = u32::from(bytes[2]);
    (((a << 10) ^ (b << 5) ^ c).wrapping_mul(0x9E37_79B1) >> (32 - HASH_BITS)) as usize & HASH_MASK
}

/// Links `pos` into the hash chain for `hash`.
fn insert_hash(head: &mut [usize], prev: &mut [usize], hash: usize, pos: usize) {
    prev[pos] = head[hash];
    head[hash] = pos;
}

/// Finds the longest match for the data at `pos`, walking the hash chain.
fn find_match(input: &[u8], pos: usize, chain_head: usize, prev: &[usize]) -> (usize, usize) {
    let len = input.len();
    let max_len = (len - pos).min(MAX_MATCH);
    if max_len < MIN_MATCH {
        return (0, 0);
    }

    let mut best_len = 0;
    let mut best_dist = 0;
    let mut candidate = chain_head;
    let mut chain_left = MAX_CHAIN;
    while candidate != usize::MAX && chain_left > 0 {
        let distance = pos - candidate;
        if distance > WINDOW {
            break;
        }
        let match_len = common_prefix(&input[candidate..], &input[pos..], max_len);
        if match_len > best_len {
            best_len = match_len;
            best_dist = distance;
            if best_len == max_len {
                break;
            }
        }
        candidate = prev[candidate];
        chain_left -= 1;
    }

    if best_len >= MIN_MATCH {
        (best_len, best_dist)
    } else {
        (0, 0)
    }
}

/// Returns the length of the shared prefix of `a` and `b`, capped at `max_len`.
fn common_prefix(a: &[u8], b: &[u8], max_len: usize) -> usize {
    let mut count = 0;
    while count < max_len && a[count] == b[count] {
        count += 1;
    }
    count
}

/// Emits a literal byte with the fixed literal/length code.
fn emit_literal(writer: &mut BitWriter, byte: u8) {
    let (code, bits) = fixed_litlen_code(u16::from(byte));
    writer.write_code(code, bits);
}

/// Emits the end-of-block symbol (256).
fn emit_end_of_block(writer: &mut BitWriter) {
    let (code, bits) = fixed_litlen_code(256);
    writer.write_code(code, bits);
}

/// Emits a `(length, distance)` back-reference with fixed codes and extra bits.
fn emit_match(writer: &mut BitWriter, length: usize, distance: usize) {
    debug_assert!((MIN_MATCH..=MAX_MATCH).contains(&length));
    debug_assert!((1..=WINDOW).contains(&distance));

    let mut li = LENGTH_BASE.len() - 1;
    while LENGTH_BASE[li] as usize > length {
        li -= 1;
    }
    let (code, bits) = fixed_litlen_code(257 + li as u16);
    writer.write_code(code, bits);
    let extra = LENGTH_EXTRA[li];
    if extra > 0 {
        writer.write_bits((length - LENGTH_BASE[li] as usize) as u32, extra);
    }

    let mut di = DIST_BASE.len() - 1;
    while DIST_BASE[di] as usize > distance {
        di -= 1;
    }
    // Fixed distance codes are all 5 bits, equal to the symbol value.
    writer.write_code(di as u16, 5);
    let dist_extra = DIST_EXTRA[di];
    if dist_extra > 0 {
        writer.write_bits((distance - DIST_BASE[di] as usize) as u32, dist_extra);
    }
}

/// Canonical fixed literal/length code and its bit length (`RFC 1951` §3.2.6).
fn fixed_litlen_code(symbol: u16) -> (u16, u32) {
    match symbol {
        0..=143 => (0x30 + symbol, 8),
        144..=255 => (0x190 + (symbol - 144), 9),
        256..=279 => (symbol - 256, 7),
        280..=287 => (0xC0 + (symbol - 280), 8),
        _ => unreachable!("fixed literal/length symbol out of range"),
    }
}
