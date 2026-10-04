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
//! A single `LZ77` token stream (literals plus `(length, distance)`
//! back-references) is produced once by a greedy hash-chain match finder and
//! then emitted under each of three `RFC 1951` block strategies; the smallest
//! encoding is kept:
//!
//! * a dynamic-Huffman block (`BTYPE=10`) whose literal/length and distance
//!   trees are built optimally from the token stream's symbol frequencies via
//!   the package-merge helper in [`super::deflate_huffman`],
//! * a fixed-Huffman block (`BTYPE=01`) using the static `RFC 1951` §3.2.6
//!   tables (cheap framing, best for tiny inputs), and
//! * stored (uncompressed) blocks (`BTYPE=00`) as an anti-expansion fallback
//!   for incompressible input.
//!
//! The length/distance base and extra-bit tables plus the code-length
//! permutation are the shared `RFC 1951` §3.2.5/§3.2.7 constants re-exported
//! from [`super::deflate`], so there is a single source of truth for them.

use alloc::vec;
use alloc::vec::Vec;

use super::deflate::{CODE_LENGTH_ORDER, DIST_BASE, DIST_EXTRA, LENGTH_BASE, LENGTH_EXTRA};
use super::deflate_huffman::{canonical_codes, length_limited_lengths, run_length_encode};

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

/// Number of `literal/length` alphabet symbols actually assignable
/// (`0..=285`); symbols `286`/`287` are reserved by `RFC 1951`.
const LITLEN_SYMBOLS: usize = 286;
/// Number of `distance` alphabet symbols (`0..=29`).
const DIST_SYMBOLS: usize = 30;
/// Maximum code length `RFC 1951` allows for the main alphabets.
const MAX_MAIN_BITS: u32 = 15;
/// Maximum code length `RFC 1951` allows for the code-length alphabet.
const MAX_CL_BITS: u32 = 7;
/// End-of-block symbol in the `literal/length` alphabet.
const END_OF_BLOCK: usize = 256;

/// One `LZ77` output element: a literal byte or a back-reference.
enum Token {
    /// A single literal byte.
    Literal(u8),
    /// A `(length, distance)` back-reference (`length` in `MIN_MATCH..=MAX_MATCH`).
    Match { length: u16, distance: u16 },
}

/// Encodes `input` as a raw `DEFLATE` stream (no `zlib`/`gzip` wrapper).
///
/// The `LZ77` token stream is produced once and emitted under the dynamic-,
/// fixed-Huffman, and stored strategies; whichever is smallest is returned, so
/// the result never expands incompressible input by more than the stored-block
/// framing overhead and spends the fewest bits on compressible input.
pub(crate) fn deflate(input: &[u8]) -> Vec<u8> {
    let tokens = produce_tokens(input);
    let mut best = emit_fixed(&tokens);

    let dynamic = emit_dynamic(&tokens);
    if dynamic.len() < best.len() {
        best = dynamic;
    }

    let stored = deflate_stored(input);
    if stored.len() < best.len() {
        best = stored;
    }
    best
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

/// Runs the greedy `LZ77` match finder once, producing the shared token stream
/// consumed by both Huffman emitters.
fn produce_tokens(input: &[u8]) -> Vec<Token> {
    let len = input.len();
    let mut tokens = Vec::new();
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
            tokens.push(Token::Match {
                length: best_len as u16,
                distance: best_dist as u16,
            });
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
            tokens.push(Token::Literal(input[pos]));
            pos += 1;
        }
    }
    tokens
}

/// Emits the token stream as a single fixed-Huffman block (`BTYPE=01`).
fn emit_fixed(tokens: &[Token]) -> Vec<u8> {
    let mut writer = BitWriter::new();
    writer.write_bits(1, 1); // BFINAL
    writer.write_bits(1, 2); // BTYPE = 01 (fixed Huffman)

    for token in tokens {
        match *token {
            Token::Literal(byte) => {
                let (code, bits) = fixed_litlen_code(u16::from(byte));
                writer.write_code(code, bits);
            }
            Token::Match { length, distance } => {
                emit_fixed_match(&mut writer, length as usize, distance as usize);
            }
        }
    }

    let (code, bits) = fixed_litlen_code(END_OF_BLOCK as u16);
    writer.write_code(code, bits);
    writer.finish()
}

/// Emits the token stream as a single dynamic-Huffman block (`BTYPE=10`),
/// building optimal literal/length and distance trees from its frequencies.
fn emit_dynamic(tokens: &[Token]) -> Vec<u8> {
    // 1. Symbol frequencies (the end-of-block symbol always occurs once).
    let mut litlen_freq = [0u32; LITLEN_SYMBOLS];
    let mut dist_freq = [0u32; DIST_SYMBOLS];
    litlen_freq[END_OF_BLOCK] = 1;
    for token in tokens {
        match *token {
            Token::Literal(byte) => litlen_freq[usize::from(byte)] += 1,
            Token::Match { length, distance } => {
                litlen_freq[257 + length_symbol(length as usize)] += 1;
                dist_freq[distance_symbol(distance as usize)] += 1;
            }
        }
    }

    // 2. Optimal, length-limited code lengths for both main alphabets.
    let litlen_lengths = length_limited_lengths(&litlen_freq, MAX_MAIN_BITS);
    let dist_lengths = length_limited_lengths(&dist_freq, MAX_MAIN_BITS);

    // 3. Trim the transmitted counts. `HLIT` is at least 257 (the end-of-block
    //    symbol lives below that) and `HDIST` at least 1 (one slot is always
    //    sent, carrying length 0 when no back-reference uses a distance).
    let mut num_litlen = LITLEN_SYMBOLS;
    while num_litlen > 257 && litlen_lengths[num_litlen - 1] == 0 {
        num_litlen -= 1;
    }
    let mut num_dist = DIST_SYMBOLS;
    while num_dist > 1 && dist_lengths[num_dist - 1] == 0 {
        num_dist -= 1;
    }

    // 4. Run-length-encode the concatenated length sequence (runs may span the
    //    literal/length -> distance boundary, which the inflate core handles).
    let mut combined = Vec::with_capacity(num_litlen + num_dist);
    combined.extend_from_slice(&litlen_lengths[..num_litlen]);
    combined.extend_from_slice(&dist_lengths[..num_dist]);
    let cl_tokens = run_length_encode(&combined);

    // 5. Code-length alphabet tree and its transmitted (permuted) length count.
    let mut cl_freq = [0u32; 19];
    for token in &cl_tokens {
        cl_freq[usize::from(token.symbol)] += 1;
    }
    let cl_lengths = length_limited_lengths(&cl_freq, MAX_CL_BITS);
    let mut num_cl = 19;
    while num_cl > 4 && cl_lengths[CODE_LENGTH_ORDER[num_cl - 1]] == 0 {
        num_cl -= 1;
    }

    // 6. Canonical codes for every alphabet.
    let litlen_codes = canonical_codes(&litlen_lengths, MAX_MAIN_BITS);
    let dist_codes = canonical_codes(&dist_lengths, MAX_MAIN_BITS);
    let cl_codes = canonical_codes(&cl_lengths, MAX_CL_BITS);

    // 7. Emit the block header (`RFC 1951` §3.2.7).
    let mut writer = BitWriter::new();
    writer.write_bits(1, 1); // BFINAL
    writer.write_bits(2, 2); // BTYPE = 10 (dynamic Huffman)
    writer.write_bits((num_litlen - 257) as u32, 5);
    writer.write_bits((num_dist - 1) as u32, 5);
    writer.write_bits((num_cl - 4) as u32, 4);
    for &slot in CODE_LENGTH_ORDER.iter().take(num_cl) {
        writer.write_bits(u32::from(cl_lengths[slot]), 3);
    }
    for token in &cl_tokens {
        let symbol = usize::from(token.symbol);
        writer.write_code(cl_codes[symbol], u32::from(cl_lengths[symbol]));
        if token.extra_bits > 0 {
            writer.write_bits(u32::from(token.extra_value), u32::from(token.extra_bits));
        }
    }

    // 8. Emit the token stream under the freshly built trees.
    for token in tokens {
        match *token {
            Token::Literal(byte) => {
                let symbol = usize::from(byte);
                writer.write_code(litlen_codes[symbol], u32::from(litlen_lengths[symbol]));
            }
            Token::Match { length, distance } => emit_dynamic_match(
                &mut writer,
                length as usize,
                distance as usize,
                &litlen_codes,
                &litlen_lengths,
                &dist_codes,
                &dist_lengths,
            ),
        }
    }
    writer.write_code(
        litlen_codes[END_OF_BLOCK],
        u32::from(litlen_lengths[END_OF_BLOCK]),
    );
    writer.finish()
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

/// Maps a match `length` to its `RFC 1951` length-symbol index (`0..=28`).
fn length_symbol(length: usize) -> usize {
    let mut li = LENGTH_BASE.len() - 1;
    while LENGTH_BASE[li] as usize > length {
        li -= 1;
    }
    li
}

/// Maps a match `distance` to its `RFC 1951` distance-symbol index (`0..=29`).
fn distance_symbol(distance: usize) -> usize {
    let mut di = DIST_BASE.len() - 1;
    while DIST_BASE[di] as usize > distance {
        di -= 1;
    }
    di
}

/// Emits a `(length, distance)` back-reference with the fixed-Huffman codes.
fn emit_fixed_match(writer: &mut BitWriter, length: usize, distance: usize) {
    debug_assert!((MIN_MATCH..=MAX_MATCH).contains(&length));
    debug_assert!((1..=WINDOW).contains(&distance));

    let li = length_symbol(length);
    let (code, bits) = fixed_litlen_code(257 + li as u16);
    writer.write_code(code, bits);
    let extra = LENGTH_EXTRA[li];
    if extra > 0 {
        writer.write_bits((length - LENGTH_BASE[li] as usize) as u32, extra);
    }

    let di = distance_symbol(distance);
    // Fixed distance codes are all 5 bits, equal to the symbol value.
    writer.write_code(di as u16, 5);
    let dist_extra = DIST_EXTRA[di];
    if dist_extra > 0 {
        writer.write_bits((distance - DIST_BASE[di] as usize) as u32, dist_extra);
    }
}

/// Emits a `(length, distance)` back-reference with dynamic-block codes.
fn emit_dynamic_match(
    writer: &mut BitWriter,
    length: usize,
    distance: usize,
    litlen_codes: &[u16],
    litlen_lengths: &[u8],
    dist_codes: &[u16],
    dist_lengths: &[u8],
) {
    debug_assert!((MIN_MATCH..=MAX_MATCH).contains(&length));
    debug_assert!((1..=WINDOW).contains(&distance));

    let li = length_symbol(length);
    let symbol = 257 + li;
    writer.write_code(litlen_codes[symbol], u32::from(litlen_lengths[symbol]));
    let extra = LENGTH_EXTRA[li];
    if extra > 0 {
        writer.write_bits((length - LENGTH_BASE[li] as usize) as u32, extra);
    }

    let di = distance_symbol(distance);
    writer.write_code(dist_codes[di], u32::from(dist_lengths[di]));
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

#[cfg(test)]
mod tests {
    use super::super::deflate::inflate;
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    fn assert_round_trips(data: &[u8]) {
        let encoded = deflate(data);
        let decoded = inflate(&encoded).expect("inflate must accept our own stream");
        assert_eq!(decoded, data, "round-trip mismatch for len {}", data.len());
    }

    #[test]
    fn empty_round_trips() {
        assert_round_trips(&[]);
    }

    #[test]
    fn single_byte_round_trips() {
        assert_round_trips(&[0x42]);
    }

    #[test]
    fn highly_repetitive_round_trips() {
        let data = vec![0xABu8; 10_000];
        assert_round_trips(&data);
    }

    #[test]
    fn structured_text_round_trips() {
        let mut data = Vec::new();
        for i in 0..2_000u32 {
            data.extend_from_slice(b"the quick brown fox ");
            data.extend_from_slice(&i.to_le_bytes());
        }
        assert_round_trips(&data);
    }

    #[test]
    fn incompressible_round_trips() {
        // A simple LCG gives a deterministic high-entropy stream.
        let mut state = 0x1234_5678u32;
        let mut data = Vec::with_capacity(8_192);
        for _ in 0..8_192 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            data.push((state >> 24) as u8);
        }
        assert_round_trips(&data);
    }

    #[test]
    fn skewed_distribution_prefers_dynamic() {
        // A strongly skewed byte histogram is where dynamic Huffman beats fixed:
        // mostly 'a' with a long tail of rare high bytes and few back-references.
        let mut data = Vec::new();
        for i in 0..20_000u32 {
            if i % 97 == 0 {
                data.push((0x80 + (i % 0x7F)) as u8);
            } else {
                data.push(b'a');
            }
        }
        let dynamic = emit_dynamic(&produce_tokens(&data));
        let fixed = emit_fixed(&produce_tokens(&data));
        assert!(
            dynamic.len() < fixed.len(),
            "dynamic {} should beat fixed {}",
            dynamic.len(),
            fixed.len()
        );
        assert_round_trips(&data);
    }

    #[test]
    fn all_byte_values_round_trip() {
        let data: Vec<u8> = (0..=255u8).collect();
        assert_round_trips(&data);
    }

    #[test]
    fn max_length_match_round_trips() {
        // Exercise the longest back-reference (length 258) and short distances.
        let mut data = vec![0u8; 1];
        data.resize(601, 0u8);
        assert_round_trips(&data);
    }
}
