//! `xxHash32`: Yann Collet's high-speed non-cryptographic 32-bit hash, the
//! fingerprint family shipped alongside the `LZ4` compressor (design §
//! fast-hash). It is used here as a cheap, well-distributed data fingerprint
//! for `GPU` resource blobs, cache keys, and streamed particle payloads.
//!
//! # Algorithm
//! `xxHash32` processes input in 16-byte blocks spread across four parallel
//! `u32` accumulator lanes. Each lane absorbs one little-endian 32-bit word per
//! block with the round `acc = rotl(acc + lane * PRIME32_2, 13) * PRIME32_1`.
//! Once the blocks are exhausted the four lanes are merged by rotating them left
//! by `1`/`7`/`12`/`18` and summing. Inputs shorter than 16 bytes skip the lane
//! machinery entirely and seed the accumulator with `seed + PRIME32_5`. In both
//! paths the total length is folded in, the trailing bytes are consumed 4 bytes
//! then 1 byte at a time, and the result is finished with an `avalanche` step
//! (`xor`-shift interleaved with multiplies by `PRIME32_2` / `PRIME32_3`). All
//! reads are little-endian and every arithmetic step is `wrapping`; the only
//! rotations use the integer intrinsic `u32::rotate_left` (`rotl`).
//!
//! Two entry points are provided and cross-checked against each other in the
//! tests: the one-shot [`xxhash32`] and the streaming [`XxHash32State`]
//! (`update` / `finish`), which produces an identical digest regardless of how
//! the input is split across calls.
//!
//! # Boundary versus the other fingerprint helpers
//! `xxHash32` is algorithmically distinct from its neighbours in this crate and
//! must not be confused with them:
//! - `fnv1a_hash` multiplies then `xor`s a single byte at a time: no lanes, no
//!   block structure, and no final `avalanche`.
//! - `murmur3_hash` mixes with a different set of rotate/multiply constants and
//!   its own tail and finalisation path.
//! - `crc32` is a polynomial remainder over `GF(2)`, not a multiplicative mix.
//! - `fibonacci_hash` is a single multiply-and-shift bucket scatter.
//!
//! `xxHash32` is NOT cryptographic. It offers no collision resistance against a
//! motivated adversary and must never be used for security, signatures, message
//! authentication, or anti-tamper. It is only a fast fingerprint for detecting
//! accidental differences and for hash-table / cache distribution.

/// First `xxHash32` prime (`0x9E3779B1`): the dominant lane multiplier.
const PRIME32_1: u32 = 0x9E37_79B1;
/// Second `xxHash32` prime (`0x85EBCA77`): the per-word lane multiplier.
const PRIME32_2: u32 = 0x85EB_CA77;
/// Third `xxHash32` prime (`0xC2B2AE3D`): the 4-byte tail multiplier.
const PRIME32_3: u32 = 0xC2B2_AE3D;
/// Fourth `xxHash32` prime (`0x27D4EB2F`): the 4-byte tail rotate multiplier.
const PRIME32_4: u32 = 0x27D4_EB2F;
/// Fifth `xxHash32` prime (`0x165667B1`): the short-input and 1-byte tail seed.
const PRIME32_5: u32 = 0x1656_67B1;

/// Reads four bytes as a little-endian `u32`.
///
/// The slice must be at least 4 bytes long; callers only ever pass exact
/// 4-byte windows produced by `chunks_exact(4)` or fixed sub-slices.
fn read_u32_le(bytes: &[u8]) -> u32 {
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

/// One `xxHash32` lane round: `rotl(acc + lane * PRIME32_2, 13) * PRIME32_1`.
///
/// Every step is `wrapping`; the rotation is the `u32::rotate_left` intrinsic.
fn round(acc: u32, lane: u32) -> u32 {
    acc.wrapping_add(lane.wrapping_mul(PRIME32_2))
        .rotate_left(13)
        .wrapping_mul(PRIME32_1)
}

/// Consumes the trailing bytes after the length has been folded in: first every
/// remaining 4-byte word, then any leftover 1..=3 bytes.
fn finalize_tail(mut acc: u32, tail: &[u8]) -> u32 {
    let mut words = tail.chunks_exact(4);
    for word in &mut words {
        let lane = read_u32_le(word);
        acc = acc.wrapping_add(lane.wrapping_mul(PRIME32_3));
        acc = acc.rotate_left(17).wrapping_mul(PRIME32_4);
    }
    for &byte in words.remainder() {
        acc = acc.wrapping_add(u32::from(byte).wrapping_mul(PRIME32_5));
        acc = acc.rotate_left(11).wrapping_mul(PRIME32_1);
    }
    acc
}

/// Final `avalanche`: `xor`-shift by 15/13/16 interleaved with multiplies by
/// `PRIME32_2` and `PRIME32_3`, scattering every input bit across the output.
fn avalanche(mut acc: u32) -> u32 {
    acc ^= acc >> 15;
    acc = acc.wrapping_mul(PRIME32_2);
    acc ^= acc >> 13;
    acc = acc.wrapping_mul(PRIME32_3);
    acc ^= acc >> 16;
    acc
}

/// Merges the four accumulator lanes into a single word by rotating each lane
/// left by `1`/`7`/`12`/`18` and summing (`wrapping`).
fn merge_lanes(v1: u32, v2: u32, v3: u32, v4: u32) -> u32 {
    v1.rotate_left(1)
        .wrapping_add(v2.rotate_left(7))
        .wrapping_add(v3.rotate_left(12))
        .wrapping_add(v4.rotate_left(18))
}

/// Computes the `xxHash32` digest of `data` with the given `seed` in one shot.
///
/// This matches the official `xxHash32` reference implementation: in particular
/// `xxhash32(b"", 0) == 0x02CC5D05`. Little-endian reads, `wrapping` arithmetic.
pub fn xxhash32(data: &[u8], seed: u32) -> u32 {
    let len = data.len();

    let (mut acc, tail) = if len >= 16 {
        let mut v1 = seed.wrapping_add(PRIME32_1).wrapping_add(PRIME32_2);
        let mut v2 = seed.wrapping_add(PRIME32_2);
        let mut v3 = seed;
        let mut v4 = seed.wrapping_sub(PRIME32_1);

        let mut blocks = data.chunks_exact(16);
        for block in &mut blocks {
            v1 = round(v1, read_u32_le(&block[0..4]));
            v2 = round(v2, read_u32_le(&block[4..8]));
            v3 = round(v3, read_u32_le(&block[8..12]));
            v4 = round(v4, read_u32_le(&block[12..16]));
        }
        (merge_lanes(v1, v2, v3, v4), blocks.remainder())
    } else {
        (seed.wrapping_add(PRIME32_5), data)
    };

    acc = acc.wrapping_add(len as u32);
    acc = finalize_tail(acc, tail);
    avalanche(acc)
}

/// Streaming `xxHash32` accumulator: feed arbitrary slices through
/// [`XxHash32State::update`] and read the digest with
/// [`XxHash32State::finish`]. The result is independent of how the byte stream
/// is partitioned across `update` calls and always equals the one-shot
/// [`xxhash32`] over the concatenation.
///
/// Per the crate's contract this type exposes no inherent arithmetic operators;
/// it is driven solely through the `update` / `finish` methods.
#[derive(Clone, Copy, Debug)]
pub struct XxHash32State {
    /// Total number of bytes fed so far (folded in as a `u32` at `finish`).
    total_len: u64,
    /// The four lane accumulators (`v3` doubles as the short-input seed).
    v1: u32,
    v2: u32,
    v3: u32,
    v4: u32,
    /// Partial 16-byte block that has not yet been consumed by a lane round.
    buffer: [u8; 16],
    /// Number of valid bytes currently held in `buffer` (always `0..16`).
    buffer_len: usize,
}

impl XxHash32State {
    /// Creates a fresh streaming state for the given `seed`.
    pub fn new(seed: u32) -> Self {
        Self {
            total_len: 0,
            v1: seed.wrapping_add(PRIME32_1).wrapping_add(PRIME32_2),
            v2: seed.wrapping_add(PRIME32_2),
            v3: seed,
            v4: seed.wrapping_sub(PRIME32_1),
            buffer: [0u8; 16],
            buffer_len: 0,
        }
    }

    /// Absorbs the four lane words held in `buffer[0..16]`.
    fn consume_buffer(&mut self) {
        self.v1 = round(self.v1, read_u32_le(&self.buffer[0..4]));
        self.v2 = round(self.v2, read_u32_le(&self.buffer[4..8]));
        self.v3 = round(self.v3, read_u32_le(&self.buffer[8..12]));
        self.v4 = round(self.v4, read_u32_le(&self.buffer[12..16]));
    }

    /// Feeds another slice of input into the running digest.
    pub fn update(&mut self, mut input: &[u8]) {
        self.total_len = self.total_len.wrapping_add(input.len() as u64);

        // Not enough to complete a block: just buffer and return.
        if self.buffer_len + input.len() < 16 {
            let end = self.buffer_len + input.len();
            self.buffer[self.buffer_len..end].copy_from_slice(input);
            self.buffer_len = end;
            return;
        }

        // Top up a partially filled buffer to a full 16-byte block and consume.
        if self.buffer_len > 0 {
            let need = 16 - self.buffer_len;
            self.buffer[self.buffer_len..16].copy_from_slice(&input[..need]);
            self.consume_buffer();
            input = &input[need..];
            self.buffer_len = 0;
        }

        // Consume as many whole 16-byte blocks as possible straight from input.
        let mut blocks = input.chunks_exact(16);
        for block in &mut blocks {
            self.v1 = round(self.v1, read_u32_le(&block[0..4]));
            self.v2 = round(self.v2, read_u32_le(&block[4..8]));
            self.v3 = round(self.v3, read_u32_le(&block[8..12]));
            self.v4 = round(self.v4, read_u32_le(&block[12..16]));
        }

        // Stash the remainder (0..16 bytes) for the next update / finish.
        let rem = blocks.remainder();
        if !rem.is_empty() {
            self.buffer[..rem.len()].copy_from_slice(rem);
            self.buffer_len = rem.len();
        }
    }

    /// Finishes the digest without consuming the state.
    pub fn finish(&self) -> u32 {
        let mut acc = if self.total_len >= 16 {
            merge_lanes(self.v1, self.v2, self.v3, self.v4)
        } else {
            // `v3` still holds the raw seed on the short-input path.
            self.v3.wrapping_add(PRIME32_5)
        };
        acc = acc.wrapping_add(self.total_len as u32);
        acc = finalize_tail(acc, &self.buffer[..self.buffer_len]);
        avalanche(acc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random byte filler (`LCG`); test-only helper.
    #[cfg(test)]
    fn fill(buf: &mut [u8], seed: u32) {
        let mut x = seed | 1;
        for slot in buf.iter_mut() {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *slot = (x >> 24) as u8;
        }
    }

    /// Asserts the streaming digest equals the one-shot digest for several
    /// different split strategies; test-only helper.
    #[cfg(test)]
    fn check_stream(data: &[u8], seed: u32) {
        let expect = xxhash32(data, seed);

        // Single update.
        let mut s = XxHash32State::new(seed);
        s.update(data);
        assert_eq!(s.finish(), expect);

        // One byte at a time.
        let mut s = XxHash32State::new(seed);
        for b in data {
            s.update(core::slice::from_ref(b));
        }
        assert_eq!(s.finish(), expect);

        // Two halves.
        let mid = data.len() / 2;
        let mut s = XxHash32State::new(seed);
        s.update(&data[..mid]);
        s.update(&data[mid..]);
        assert_eq!(s.finish(), expect);

        // Thirds.
        let a = data.len() / 3;
        let b = 2 * (data.len() / 3);
        let mut s = XxHash32State::new(seed);
        s.update(&data[..a]);
        s.update(&data[a..b]);
        s.update(&data[b..]);
        assert_eq!(s.finish(), expect);
    }

    // ---- Official reference vectors (hard anchors) -----------------------

    #[test]
    fn empty_official_seed0() {
        assert_eq!(xxhash32(b"", 0), 0x02CC_5D05);
    }

    #[test]
    fn empty_official_seed_prime1() {
        assert_eq!(xxhash32(b"", 0x9E37_79B1), 0x36B7_8AE7);
    }

    #[test]
    fn empty_oneshot_eq_stream_seed0() {
        let mut s = XxHash32State::new(0);
        s.update(b"");
        assert_eq!(s.finish(), 0x02CC_5D05);
    }

    #[test]
    fn empty_oneshot_eq_stream_seeded() {
        let mut s = XxHash32State::new(0x9E37_79B1);
        s.update(b"");
        assert_eq!(s.finish(), 0x36B7_8AE7);
    }

    // ---- Short-input length boundaries (< 16 bytes) ----------------------

    #[test]
    fn len1() {
        check_stream(b"a", 0);
    }

    #[test]
    fn len2() {
        check_stream(b"ab", 0);
    }

    #[test]
    fn len3() {
        check_stream(b"abc", 0);
    }

    #[test]
    fn len4_exact_word() {
        check_stream(b"abcd", 0);
    }

    #[test]
    fn len5() {
        check_stream(b"abcde", 0);
    }

    #[test]
    fn len7() {
        check_stream(b"abcdefg", 0);
    }

    #[test]
    fn len8() {
        check_stream(b"abcdefgh", 0);
    }

    #[test]
    fn len12() {
        check_stream(b"0123456789AB", 0);
    }

    #[test]
    fn len13() {
        check_stream(b"0123456789ABC", 0);
    }

    #[test]
    fn len15_just_under_block() {
        check_stream(b"0123456789ABCDE", 0);
    }

    // ---- The 16-byte boundary and just over -------------------------------

    #[test]
    fn len16_exact_block() {
        check_stream(b"0123456789ABCDEF", 0);
    }

    #[test]
    fn len17_one_over_block() {
        check_stream(b"0123456789ABCDEFG", 0);
    }

    #[test]
    fn len31_just_under_two_blocks() {
        let mut buf = [0u8; 31];
        fill(&mut buf, 7);
        check_stream(&buf, 0);
    }

    #[test]
    fn len32_two_blocks() {
        let mut buf = [0u8; 32];
        fill(&mut buf, 11);
        check_stream(&buf, 0);
    }

    #[test]
    fn len33_two_blocks_plus_one() {
        let mut buf = [0u8; 33];
        fill(&mut buf, 13);
        check_stream(&buf, 0);
    }

    #[test]
    fn len64_four_blocks() {
        let mut buf = [0u8; 64];
        fill(&mut buf, 17);
        check_stream(&buf, 0);
    }

    #[test]
    fn len100() {
        let mut buf = [0u8; 100];
        fill(&mut buf, 19);
        check_stream(&buf, 0);
    }

    #[test]
    fn len1000_large() {
        let mut buf = [0u8; 1000];
        fill(&mut buf, 23);
        check_stream(&buf, 0);
    }

    // ---- Seeded inputs ----------------------------------------------------

    #[test]
    fn seed_small_input() {
        check_stream(b"abc", 0xDEAD_BEEF);
    }

    #[test]
    fn seed_large_input() {
        let mut buf = [0u8; 48];
        fill(&mut buf, 29);
        check_stream(&buf, 0xDEAD_BEEF);
    }

    #[test]
    fn seed_prime_values() {
        let mut buf = [0u8; 40];
        fill(&mut buf, 31);
        check_stream(&buf, PRIME32_1);
        check_stream(&buf, PRIME32_2);
        check_stream(&buf, PRIME32_5);
    }

    #[test]
    fn seed_changes_output() {
        let data = b"the quick brown fox";
        assert_ne!(xxhash32(data, 0), xxhash32(data, 1));
        assert_ne!(xxhash32(data, 1), xxhash32(data, 2));
    }

    // ---- Tail lengths (1 / 2 / 3 / 0 bytes after the 4-byte words) -------

    #[test]
    fn tail_one_byte_after_block() {
        // 16 + 1: a full block then one tail byte.
        check_stream(b"0123456789ABCDEFx", 0);
    }

    #[test]
    fn tail_two_bytes_after_block() {
        check_stream(b"0123456789ABCDEFxy", 0);
    }

    #[test]
    fn tail_three_bytes_after_block() {
        check_stream(b"0123456789ABCDEFxyz", 0);
    }

    #[test]
    fn tail_zero_multiple_of_four() {
        // 20 bytes: one 16-byte block plus one 4-byte word, no 1-byte tail.
        check_stream(b"0123456789ABCDEFwxyz", 0);
    }

    #[test]
    fn tail_word_then_bytes() {
        // 16 + 4 + 3 bytes exercises both the word loop and the byte loop.
        check_stream(b"0123456789ABCDEFwxyzPQR", 0);
    }

    // ---- Streaming split points ------------------------------------------

    #[test]
    fn stream_split_at_16() {
        let mut buf = [0u8; 40];
        fill(&mut buf, 37);
        let expect = xxhash32(&buf, 0);
        let mut s = XxHash32State::new(0);
        s.update(&buf[..16]);
        s.update(&buf[16..]);
        assert_eq!(s.finish(), expect);
    }

    #[test]
    fn stream_split_7_then_rest() {
        let mut buf = [0u8; 40];
        fill(&mut buf, 41);
        let expect = xxhash32(&buf, 0);
        let mut s = XxHash32State::new(0);
        s.update(&buf[..7]);
        s.update(&buf[7..]);
        assert_eq!(s.finish(), expect);
    }

    #[test]
    fn stream_many_tiny_splits() {
        let mut buf = [0u8; 70];
        fill(&mut buf, 43);
        let expect = xxhash32(&buf, 0xABCD);
        let mut s = XxHash32State::new(0xABCD);
        let mut i = 0;
        let steps = [1usize, 2, 3, 5, 8, 13, 1, 4, 9, 7, 17];
        for &step in &steps {
            let end = (i + step).min(buf.len());
            s.update(&buf[i..end]);
            i = end;
        }
        s.update(&buf[i..]);
        assert_eq!(s.finish(), expect);
    }

    #[test]
    fn stream_empty_updates_interleaved() {
        let mut buf = [0u8; 50];
        fill(&mut buf, 47);
        let expect = xxhash32(&buf, 0);
        let mut s = XxHash32State::new(0);
        s.update(b"");
        s.update(&buf[..10]);
        s.update(b"");
        s.update(&buf[10..30]);
        s.update(b"");
        s.update(&buf[30..]);
        s.update(b"");
        assert_eq!(s.finish(), expect);
    }

    #[test]
    fn stream_buffer_then_cross_block() {
        // Buffer 10 bytes, then feed 20 more so the fill-and-consume path and a
        // subsequent direct block both run.
        let mut buf = [0u8; 30];
        fill(&mut buf, 53);
        let expect = xxhash32(&buf, 0);
        let mut s = XxHash32State::new(0);
        s.update(&buf[..10]);
        s.update(&buf[10..]);
        assert_eq!(s.finish(), expect);
    }

    // ---- Structural properties -------------------------------------------

    #[test]
    fn determinism_repeat() {
        let mut buf = [0u8; 123];
        fill(&mut buf, 59);
        let a = xxhash32(&buf, 99);
        let b = xxhash32(&buf, 99);
        assert_eq!(a, b);
    }

    #[test]
    fn single_bit_change_changes_hash() {
        let mut buf = [0u8; 64];
        fill(&mut buf, 61);
        let before = xxhash32(&buf, 0);
        buf[20] ^= 0x01;
        let after = xxhash32(&buf, 0);
        assert_ne!(before, after);
    }

    #[test]
    fn byte_order_matters() {
        assert_ne!(xxhash32(&[1, 2, 3, 4], 0), xxhash32(&[4, 3, 2, 1], 0));
    }

    #[test]
    fn all_zeros_block() {
        let buf = [0u8; 16];
        check_stream(&buf, 0);
        // Length is still folded in, so a zero block does not hash to the seed.
        assert_ne!(xxhash32(&buf, 0), xxhash32(b"", 0));
    }

    #[test]
    fn all_ones_block() {
        let buf = [0xFFu8; 16];
        check_stream(&buf, 0);
    }

    #[test]
    fn distinct_lengths_distinct_hashes() {
        // Zero-filled inputs of different lengths differ because the length is
        // mixed into the digest.
        let z1 = [0u8; 4];
        let z2 = [0u8; 8];
        assert_ne!(xxhash32(&z1, 0), xxhash32(&z2, 0));
    }

    // ---- Low-level helper checks -----------------------------------------

    #[test]
    fn read_u32_le_is_little_endian() {
        assert_eq!(read_u32_le(&[0x78, 0x56, 0x34, 0x12]), 0x1234_5678);
        assert_eq!(read_u32_le(&[0x00, 0x00, 0x00, 0x01]), 0x0100_0000);
    }

    #[test]
    fn prime_constant_values() {
        assert_eq!(PRIME32_1, 0x9E37_79B1);
        assert_eq!(PRIME32_2, 0x85EB_CA77);
        assert_eq!(PRIME32_3, 0xC2B2_AE3D);
        assert_eq!(PRIME32_4, 0x27D4_EB2F);
        assert_eq!(PRIME32_5, 0x1656_67B1);
    }

    #[test]
    fn round_is_deterministic() {
        assert_eq!(round(0, 0), round(0, 0));
        assert_eq!(round(123, 456), round(123, 456));
        // A different lane produces a different accumulator.
        assert_ne!(round(0, 1), round(0, 2));
    }

    #[test]
    fn merge_lanes_matches_definition() {
        let (v1, v2, v3, v4) = (1u32, 2u32, 3u32, 4u32);
        let expect = v1
            .rotate_left(1)
            .wrapping_add(v2.rotate_left(7))
            .wrapping_add(v3.rotate_left(12))
            .wrapping_add(v4.rotate_left(18));
        assert_eq!(merge_lanes(v1, v2, v3, v4), expect);
    }

    #[test]
    fn avalanche_spreads_bits() {
        // The avalanche is a bijection-like mix; distinct inputs stay distinct
        // here and a lone low bit is scattered into the high bits.
        assert_ne!(avalanche(1), avalanche(2));
        assert_ne!(avalanche(1) & 0xFFFF_0000, 0);
    }
}
