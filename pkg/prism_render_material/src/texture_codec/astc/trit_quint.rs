//! ASTC trit/quint integer-sequence block unpacking (Khronos DFS 1.3).
//!
//! A trit block packs five base-3 digits into eight bits; a quint block packs
//! three base-5 digits into seven bits (see [`super::bise`]). The forward and
//! inverse packings are fixed bijections defined by the specification. The
//! lookup tables below are transcribed verbatim from the ARM `astcenc`
//! reference decoder (`astcenc_integer_sequence.cpp`, Apache-2.0) -- the same
//! tables every conformant ASTC hardware unit implements -- rather than being
//! reconstructed by hand, and a `#[cfg(test)]` cross-check proves each decode
//! table is the exact inverse of the corresponding encode table over all
//! `3^5` / `5^3` digit tuples.
//!
//! [`decode_trit_sequence`] / [`decode_quint_sequence`] reproduce the
//! `astcenc` `decode_ise` bit-collection order exactly: each value's low bits
//! are interleaved with 1-3 bits of its trit/quint block, the block bytes are
//! reassembled, then expanded through the tables. Round-trip tests against the
//! inverse (encode) tables exercise every supported `(bits, count)`.

use super::block_reader::read_bits;

/// Unpacked trit quintuplets `[t0, t1, t2, t3, t4]` for each 8-bit trit block.
#[rustfmt::skip]
pub(super) const TRITS_OF_INTEGER: [[u8; 5]; 256] = [
    [0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [2, 0, 0, 0, 0], [0, 0, 2, 0, 0],
    [0, 1, 0, 0, 0], [1, 1, 0, 0, 0], [2, 1, 0, 0, 0], [1, 0, 2, 0, 0],
    [0, 2, 0, 0, 0], [1, 2, 0, 0, 0], [2, 2, 0, 0, 0], [2, 0, 2, 0, 0],
    [0, 2, 2, 0, 0], [1, 2, 2, 0, 0], [2, 2, 2, 0, 0], [2, 0, 2, 0, 0],
    [0, 0, 1, 0, 0], [1, 0, 1, 0, 0], [2, 0, 1, 0, 0], [0, 1, 2, 0, 0],
    [0, 1, 1, 0, 0], [1, 1, 1, 0, 0], [2, 1, 1, 0, 0], [1, 1, 2, 0, 0],
    [0, 2, 1, 0, 0], [1, 2, 1, 0, 0], [2, 2, 1, 0, 0], [2, 1, 2, 0, 0],
    [0, 0, 0, 2, 2], [1, 0, 0, 2, 2], [2, 0, 0, 2, 2], [0, 0, 2, 2, 2],
    [0, 0, 0, 1, 0], [1, 0, 0, 1, 0], [2, 0, 0, 1, 0], [0, 0, 2, 1, 0],
    [0, 1, 0, 1, 0], [1, 1, 0, 1, 0], [2, 1, 0, 1, 0], [1, 0, 2, 1, 0],
    [0, 2, 0, 1, 0], [1, 2, 0, 1, 0], [2, 2, 0, 1, 0], [2, 0, 2, 1, 0],
    [0, 2, 2, 1, 0], [1, 2, 2, 1, 0], [2, 2, 2, 1, 0], [2, 0, 2, 1, 0],
    [0, 0, 1, 1, 0], [1, 0, 1, 1, 0], [2, 0, 1, 1, 0], [0, 1, 2, 1, 0],
    [0, 1, 1, 1, 0], [1, 1, 1, 1, 0], [2, 1, 1, 1, 0], [1, 1, 2, 1, 0],
    [0, 2, 1, 1, 0], [1, 2, 1, 1, 0], [2, 2, 1, 1, 0], [2, 1, 2, 1, 0],
    [0, 1, 0, 2, 2], [1, 1, 0, 2, 2], [2, 1, 0, 2, 2], [1, 0, 2, 2, 2],
    [0, 0, 0, 2, 0], [1, 0, 0, 2, 0], [2, 0, 0, 2, 0], [0, 0, 2, 2, 0],
    [0, 1, 0, 2, 0], [1, 1, 0, 2, 0], [2, 1, 0, 2, 0], [1, 0, 2, 2, 0],
    [0, 2, 0, 2, 0], [1, 2, 0, 2, 0], [2, 2, 0, 2, 0], [2, 0, 2, 2, 0],
    [0, 2, 2, 2, 0], [1, 2, 2, 2, 0], [2, 2, 2, 2, 0], [2, 0, 2, 2, 0],
    [0, 0, 1, 2, 0], [1, 0, 1, 2, 0], [2, 0, 1, 2, 0], [0, 1, 2, 2, 0],
    [0, 1, 1, 2, 0], [1, 1, 1, 2, 0], [2, 1, 1, 2, 0], [1, 1, 2, 2, 0],
    [0, 2, 1, 2, 0], [1, 2, 1, 2, 0], [2, 2, 1, 2, 0], [2, 1, 2, 2, 0],
    [0, 2, 0, 2, 2], [1, 2, 0, 2, 2], [2, 2, 0, 2, 2], [2, 0, 2, 2, 2],
    [0, 0, 0, 0, 2], [1, 0, 0, 0, 2], [2, 0, 0, 0, 2], [0, 0, 2, 0, 2],
    [0, 1, 0, 0, 2], [1, 1, 0, 0, 2], [2, 1, 0, 0, 2], [1, 0, 2, 0, 2],
    [0, 2, 0, 0, 2], [1, 2, 0, 0, 2], [2, 2, 0, 0, 2], [2, 0, 2, 0, 2],
    [0, 2, 2, 0, 2], [1, 2, 2, 0, 2], [2, 2, 2, 0, 2], [2, 0, 2, 0, 2],
    [0, 0, 1, 0, 2], [1, 0, 1, 0, 2], [2, 0, 1, 0, 2], [0, 1, 2, 0, 2],
    [0, 1, 1, 0, 2], [1, 1, 1, 0, 2], [2, 1, 1, 0, 2], [1, 1, 2, 0, 2],
    [0, 2, 1, 0, 2], [1, 2, 1, 0, 2], [2, 2, 1, 0, 2], [2, 1, 2, 0, 2],
    [0, 2, 2, 2, 2], [1, 2, 2, 2, 2], [2, 2, 2, 2, 2], [2, 0, 2, 2, 2],
    [0, 0, 0, 0, 1], [1, 0, 0, 0, 1], [2, 0, 0, 0, 1], [0, 0, 2, 0, 1],
    [0, 1, 0, 0, 1], [1, 1, 0, 0, 1], [2, 1, 0, 0, 1], [1, 0, 2, 0, 1],
    [0, 2, 0, 0, 1], [1, 2, 0, 0, 1], [2, 2, 0, 0, 1], [2, 0, 2, 0, 1],
    [0, 2, 2, 0, 1], [1, 2, 2, 0, 1], [2, 2, 2, 0, 1], [2, 0, 2, 0, 1],
    [0, 0, 1, 0, 1], [1, 0, 1, 0, 1], [2, 0, 1, 0, 1], [0, 1, 2, 0, 1],
    [0, 1, 1, 0, 1], [1, 1, 1, 0, 1], [2, 1, 1, 0, 1], [1, 1, 2, 0, 1],
    [0, 2, 1, 0, 1], [1, 2, 1, 0, 1], [2, 2, 1, 0, 1], [2, 1, 2, 0, 1],
    [0, 0, 1, 2, 2], [1, 0, 1, 2, 2], [2, 0, 1, 2, 2], [0, 1, 2, 2, 2],
    [0, 0, 0, 1, 1], [1, 0, 0, 1, 1], [2, 0, 0, 1, 1], [0, 0, 2, 1, 1],
    [0, 1, 0, 1, 1], [1, 1, 0, 1, 1], [2, 1, 0, 1, 1], [1, 0, 2, 1, 1],
    [0, 2, 0, 1, 1], [1, 2, 0, 1, 1], [2, 2, 0, 1, 1], [2, 0, 2, 1, 1],
    [0, 2, 2, 1, 1], [1, 2, 2, 1, 1], [2, 2, 2, 1, 1], [2, 0, 2, 1, 1],
    [0, 0, 1, 1, 1], [1, 0, 1, 1, 1], [2, 0, 1, 1, 1], [0, 1, 2, 1, 1],
    [0, 1, 1, 1, 1], [1, 1, 1, 1, 1], [2, 1, 1, 1, 1], [1, 1, 2, 1, 1],
    [0, 2, 1, 1, 1], [1, 2, 1, 1, 1], [2, 2, 1, 1, 1], [2, 1, 2, 1, 1],
    [0, 1, 1, 2, 2], [1, 1, 1, 2, 2], [2, 1, 1, 2, 2], [1, 1, 2, 2, 2],
    [0, 0, 0, 2, 1], [1, 0, 0, 2, 1], [2, 0, 0, 2, 1], [0, 0, 2, 2, 1],
    [0, 1, 0, 2, 1], [1, 1, 0, 2, 1], [2, 1, 0, 2, 1], [1, 0, 2, 2, 1],
    [0, 2, 0, 2, 1], [1, 2, 0, 2, 1], [2, 2, 0, 2, 1], [2, 0, 2, 2, 1],
    [0, 2, 2, 2, 1], [1, 2, 2, 2, 1], [2, 2, 2, 2, 1], [2, 0, 2, 2, 1],
    [0, 0, 1, 2, 1], [1, 0, 1, 2, 1], [2, 0, 1, 2, 1], [0, 1, 2, 2, 1],
    [0, 1, 1, 2, 1], [1, 1, 1, 2, 1], [2, 1, 1, 2, 1], [1, 1, 2, 2, 1],
    [0, 2, 1, 2, 1], [1, 2, 1, 2, 1], [2, 2, 1, 2, 1], [2, 1, 2, 2, 1],
    [0, 2, 1, 2, 2], [1, 2, 1, 2, 2], [2, 2, 1, 2, 2], [2, 1, 2, 2, 2],
    [0, 0, 0, 1, 2], [1, 0, 0, 1, 2], [2, 0, 0, 1, 2], [0, 0, 2, 1, 2],
    [0, 1, 0, 1, 2], [1, 1, 0, 1, 2], [2, 1, 0, 1, 2], [1, 0, 2, 1, 2],
    [0, 2, 0, 1, 2], [1, 2, 0, 1, 2], [2, 2, 0, 1, 2], [2, 0, 2, 1, 2],
    [0, 2, 2, 1, 2], [1, 2, 2, 1, 2], [2, 2, 2, 1, 2], [2, 0, 2, 1, 2],
    [0, 0, 1, 1, 2], [1, 0, 1, 1, 2], [2, 0, 1, 1, 2], [0, 1, 2, 1, 2],
    [0, 1, 1, 1, 2], [1, 1, 1, 1, 2], [2, 1, 1, 1, 2], [1, 1, 2, 1, 2],
    [0, 2, 1, 1, 2], [1, 2, 1, 1, 2], [2, 2, 1, 1, 2], [2, 1, 2, 1, 2],
    [0, 2, 2, 2, 2], [1, 2, 2, 2, 2], [2, 2, 2, 2, 2], [2, 1, 2, 2, 2],
];

/// Unpacked quint triplets `[q0, q1, q2]` for each 7-bit quint block.
#[rustfmt::skip]
pub(super) const QUINTS_OF_INTEGER: [[u8; 3]; 128] = [
    [0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0],
    [4, 0, 0], [0, 4, 0], [4, 4, 0], [4, 4, 4],
    [0, 1, 0], [1, 1, 0], [2, 1, 0], [3, 1, 0],
    [4, 1, 0], [1, 4, 0], [4, 4, 1], [4, 4, 4],
    [0, 2, 0], [1, 2, 0], [2, 2, 0], [3, 2, 0],
    [4, 2, 0], [2, 4, 0], [4, 4, 2], [4, 4, 4],
    [0, 3, 0], [1, 3, 0], [2, 3, 0], [3, 3, 0],
    [4, 3, 0], [3, 4, 0], [4, 4, 3], [4, 4, 4],
    [0, 0, 1], [1, 0, 1], [2, 0, 1], [3, 0, 1],
    [4, 0, 1], [0, 4, 1], [4, 0, 4], [0, 4, 4],
    [0, 1, 1], [1, 1, 1], [2, 1, 1], [3, 1, 1],
    [4, 1, 1], [1, 4, 1], [4, 1, 4], [1, 4, 4],
    [0, 2, 1], [1, 2, 1], [2, 2, 1], [3, 2, 1],
    [4, 2, 1], [2, 4, 1], [4, 2, 4], [2, 4, 4],
    [0, 3, 1], [1, 3, 1], [2, 3, 1], [3, 3, 1],
    [4, 3, 1], [3, 4, 1], [4, 3, 4], [3, 4, 4],
    [0, 0, 2], [1, 0, 2], [2, 0, 2], [3, 0, 2],
    [4, 0, 2], [0, 4, 2], [2, 0, 4], [3, 0, 4],
    [0, 1, 2], [1, 1, 2], [2, 1, 2], [3, 1, 2],
    [4, 1, 2], [1, 4, 2], [2, 1, 4], [3, 1, 4],
    [0, 2, 2], [1, 2, 2], [2, 2, 2], [3, 2, 2],
    [4, 2, 2], [2, 4, 2], [2, 2, 4], [3, 2, 4],
    [0, 3, 2], [1, 3, 2], [2, 3, 2], [3, 3, 2],
    [4, 3, 2], [3, 4, 2], [2, 3, 4], [3, 3, 4],
    [0, 0, 3], [1, 0, 3], [2, 0, 3], [3, 0, 3],
    [4, 0, 3], [0, 4, 3], [0, 0, 4], [1, 0, 4],
    [0, 1, 3], [1, 1, 3], [2, 1, 3], [3, 1, 3],
    [4, 1, 3], [1, 4, 3], [0, 1, 4], [1, 1, 4],
    [0, 2, 3], [1, 2, 3], [2, 2, 3], [3, 2, 3],
    [4, 2, 3], [2, 4, 3], [0, 2, 4], [1, 2, 4],
    [0, 3, 3], [1, 3, 3], [2, 3, 3], [3, 3, 3],
    [4, 3, 3], [3, 4, 3], [0, 3, 4], [1, 3, 4],
];

/// Decode a BISE **trit** sequence of `count` values, each carrying `bits` low
/// bits plus a shared base-3 trit, starting at bit `start` of `block`.
///
/// Writes the fully assembled quantised value (`low | (trit << bits)`, in
/// `0..3 * 2^bits`) for each element into `out[0..count]`, mirroring the
/// `astcenc` `decode_ise` collection order bit-for-bit.
pub(super) fn decode_trit_sequence(
    block: &[u8; 16],
    start: u32,
    bits: u32,
    count: u32,
    out: &mut [u8],
) {
    const BITS_TO_READ: [u32; 5] = [2, 2, 1, 2, 1];
    const BLOCK_SHIFT: [u32; 5] = [0, 2, 4, 5, 7];
    const NEXT_L: [usize; 5] = [1, 2, 3, 4, 0];
    const H_INCR: [usize; 5] = [0, 0, 0, 0, 1];
    let mut tq = [0u8; 24];
    let mut off = start;
    let (mut lc, mut hc) = (0usize, 0usize);
    for slot in out.iter_mut().take(count as usize) {
        *slot = read_bits(block, off, bits) as u8;
        off += bits;
        let tdata = read_bits(block, off, BITS_TO_READ[lc]);
        off += BITS_TO_READ[lc];
        tq[hc] |= (tdata << BLOCK_SHIFT[lc]) as u8;
        hc += H_INCR[lc];
        lc = NEXT_L[lc];
    }
    let blocks = (count + 4) / 5;
    for i in 0..blocks as usize {
        let t = TRITS_OF_INTEGER[tq[i] as usize];
        for (k, &digit) in t.iter().enumerate() {
            let idx = 5 * i + k;
            if idx < count as usize {
                out[idx] |= digit << bits;
            }
        }
    }
}

/// Decode a BISE **quint** sequence of `count` values, each carrying `bits` low
/// bits plus a shared base-5 quint, starting at bit `start` of `block`.
///
/// Writes `low | (quint << bits)` (in `0..5 * 2^bits`) for each element into
/// `out[0..count]`, mirroring the `astcenc` `decode_ise` collection order.
pub(super) fn decode_quint_sequence(
    block: &[u8; 16],
    start: u32,
    bits: u32,
    count: u32,
    out: &mut [u8],
) {
    const BITS_TO_READ: [u32; 3] = [3, 2, 2];
    const BLOCK_SHIFT: [u32; 3] = [0, 3, 5];
    const NEXT_L: [usize; 3] = [1, 2, 0];
    const H_INCR: [usize; 3] = [0, 0, 1];
    let mut tq = [0u8; 24];
    let mut off = start;
    let (mut lc, mut hc) = (0usize, 0usize);
    for slot in out.iter_mut().take(count as usize) {
        *slot = read_bits(block, off, bits) as u8;
        off += bits;
        let tdata = read_bits(block, off, BITS_TO_READ[lc]);
        off += BITS_TO_READ[lc];
        tq[hc] |= (tdata << BLOCK_SHIFT[lc]) as u8;
        hc += H_INCR[lc];
        lc = NEXT_L[lc];
    }
    let blocks = (count + 2) / 3;
    for i in 0..blocks as usize {
        let q = QUINTS_OF_INTEGER[tq[i] as usize];
        for (k, &digit) in q.iter().enumerate() {
            let idx = 3 * i + k;
            if idx < count as usize {
                out[idx] |= digit << bits;
            }
        }
    }
}

/// Packed trit value for digit tuple `[t0,t1,t2,t3,t4]`, indexed
/// `(((t4*3+t3)*3+t2)*3+t1)*3+t0`. The exact inverse of
/// [`TRITS_OF_INTEGER`] (proven over all `3^5` tuples in tests); transcribed
/// from the ARM `astcenc` reference encoder.
#[rustfmt::skip]
pub(super) const INTEGER_OF_TRITS: [u8; 243] = [
    0, 1, 2, 4, 5, 6, 8, 9, 10, 16, 17, 18,
    20, 21, 22, 24, 25, 26, 3, 7, 15, 19, 23, 27,
    12, 13, 14, 32, 33, 34, 36, 37, 38, 40, 41, 42,
    48, 49, 50, 52, 53, 54, 56, 57, 58, 35, 39, 47,
    51, 55, 59, 44, 45, 46, 64, 65, 66, 68, 69, 70,
    72, 73, 74, 80, 81, 82, 84, 85, 86, 88, 89, 90,
    67, 71, 79, 83, 87, 91, 76, 77, 78, 128, 129, 130,
    132, 133, 134, 136, 137, 138, 144, 145, 146, 148, 149, 150,
    152, 153, 154, 131, 135, 143, 147, 151, 155, 140, 141, 142,
    160, 161, 162, 164, 165, 166, 168, 169, 170, 176, 177, 178,
    180, 181, 182, 184, 185, 186, 163, 167, 175, 179, 183, 187,
    172, 173, 174, 192, 193, 194, 196, 197, 198, 200, 201, 202,
    208, 209, 210, 212, 213, 214, 216, 217, 218, 195, 199, 207,
    211, 215, 219, 204, 205, 206, 96, 97, 98, 100, 101, 102,
    104, 105, 106, 112, 113, 114, 116, 117, 118, 120, 121, 122,
    99, 103, 111, 115, 119, 123, 108, 109, 110, 224, 225, 226,
    228, 229, 230, 232, 233, 234, 240, 241, 242, 244, 245, 246,
    248, 249, 250, 227, 231, 239, 243, 247, 251, 236, 237, 238,
    28, 29, 30, 60, 61, 62, 92, 93, 94, 156, 157, 158,
    188, 189, 190, 220, 221, 222, 31, 63, 127, 159, 191, 255,
    252, 253, 254,
];

/// Packed quint value for digit tuple `[q0,q1,q2]`, indexed
/// `(q2*5+q1)*5+q0`. The exact inverse of [`QUINTS_OF_INTEGER`] (proven over
/// all `5^3` tuples in tests); transcribed from the ARM `astcenc` encoder.
#[rustfmt::skip]
pub(super) const INTEGER_OF_QUINTS: [u8; 125] = [
    0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 16, 17,
    18, 19, 20, 24, 25, 26, 27, 28, 5, 13, 21, 29,
    6, 32, 33, 34, 35, 36, 40, 41, 42, 43, 44, 48,
    49, 50, 51, 52, 56, 57, 58, 59, 60, 37, 45, 53,
    61, 14, 64, 65, 66, 67, 68, 72, 73, 74, 75, 76,
    80, 81, 82, 83, 84, 88, 89, 90, 91, 92, 69, 77,
    85, 93, 22, 96, 97, 98, 99, 100, 104, 105, 106, 107,
    108, 112, 113, 114, 115, 116, 120, 121, 122, 123, 124, 101,
    109, 117, 125, 30, 102, 103, 70, 71, 38, 110, 111, 78,
    79, 46, 118, 119, 86, 87, 54, 126, 127, 94, 95, 62,
    39, 47, 55, 63, 31,
];

/// Set the low `count` bits of `val` into `block` at bit offset `lo`, LSB-first.
fn write_bits_le(block: &mut [u8; 16], lo: u32, count: u32, val: u32) {
    for i in 0..count {
        if (val >> i) & 1 == 1 {
            let bit = lo + i;
            block[(bit >> 3) as usize] |= 1 << (bit & 7);
        }
    }
}

/// Packed trit integer for the five base-3 digits `t` (MSB digit `t[4]`).
#[inline]
pub(super) fn pack_trit(t: [u8; 5]) -> u8 {
    INTEGER_OF_TRITS[((((t[4] as usize * 3 + t[3] as usize) * 3 + t[2] as usize) * 3
        + t[1] as usize)
        * 3)
        + t[0] as usize]
}

/// Packed quint integer for the three base-5 digits `q` (MSB digit `q[2]`).
#[inline]
pub(super) fn pack_quint(q: [u8; 3]) -> u8 {
    INTEGER_OF_QUINTS[(q[2] as usize * 5 + q[1] as usize) * 5 + q[0] as usize]
}

/// Encode a BISE **trit** sequence: the exact inverse of
/// [`decode_trit_sequence`]. Each value in `vals` is `low | (trit << bits)`
/// with `low` in `0..2^bits` and `trit` in `0..3`; the stream is written
/// starting at bit `start` of `block` in `astcenc` collection order.
pub(super) fn encode_trit_sequence(block: &mut [u8; 16], start: u32, bits: u32, vals: &[u8]) {
    let mask = (1u32 << bits) - 1;
    let count = vals.len();
    let mut off = start;
    let mut i = 0usize;
    let full = count / 5;
    let shifts = [0u32, 2, 4, 5, 7];
    let tb = [2u32, 2, 1, 2, 1];
    for _ in 0..full {
        let t = pack_trit([
            vals[i] >> bits,
            vals[i + 1] >> bits,
            vals[i + 2] >> bits,
            vals[i + 3] >> bits,
            vals[i + 4] >> bits,
        ]) as u32;
        for e in 0..5 {
            let pack =
                ((vals[i] as u32) & mask) | (((t >> shifts[e]) & ((1 << tb[e]) - 1)) << bits);
            write_bits_le(block, off, bits + tb[e], pack);
            off += bits + tb[e];
            i += 1;
        }
    }
    if i != count {
        let g = |k: usize| {
            if i + k >= count {
                0
            } else {
                vals[i + k] >> bits
            }
        };
        let t = pack_trit([g(0), g(1), g(2), g(3), 0]) as u32;
        let mut j = 0usize;
        while i < count {
            let pack =
                ((vals[i] as u32) & mask) | (((t >> shifts[j]) & ((1 << tb[j]) - 1)) << bits);
            write_bits_le(block, off, bits + tb[j], pack);
            off += bits + tb[j];
            i += 1;
            j += 1;
        }
    }
}

/// Encode a BISE **quint** sequence: the exact inverse of
/// [`decode_quint_sequence`]. Each value in `vals` is `low | (quint << bits)`
/// with `low` in `0..2^bits` and `quint` in `0..5`; written starting at bit
/// `start` of `block` in `astcenc` collection order.
pub(super) fn encode_quint_sequence(block: &mut [u8; 16], start: u32, bits: u32, vals: &[u8]) {
    let mask = (1u32 << bits) - 1;
    let count = vals.len();
    let mut off = start;
    let mut i = 0usize;
    let full = count / 3;
    let shifts = [0u32, 3, 5];
    let tb = [3u32, 2, 2];
    for _ in 0..full {
        let t = pack_quint([vals[i] >> bits, vals[i + 1] >> bits, vals[i + 2] >> bits]) as u32;
        for e in 0..3 {
            let pack =
                ((vals[i] as u32) & mask) | (((t >> shifts[e]) & ((1 << tb[e]) - 1)) << bits);
            write_bits_le(block, off, bits + tb[e], pack);
            off += bits + tb[e];
            i += 1;
        }
    }
    if i != count {
        let g = |k: usize| {
            if i + k >= count {
                0
            } else {
                vals[i + k] >> bits
            }
        };
        let t = pack_quint([g(0), g(1), 0]) as u32;
        let mut j = 0usize;
        while i < count {
            let pack =
                ((vals[i] as u32) & mask) | (((t >> shifts[j]) & ((1 << tb[j]) - 1)) << bits);
            write_bits_le(block, off, bits + tb[j], pack);
            off += bits + tb[j];
            i += 1;
            j += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    /// Every decode-table entry is the exact inverse of the production encode
    /// table, over all 3^5 trit tuples and 5^3 quint tuples.
    #[test]
    fn decode_tables_invert_encode_tables() {
        for t0 in 0..3 {
            for t1 in 0..3 {
                for t2 in 0..3 {
                    for t3 in 0..3 {
                        for t4 in 0..3 {
                            let tup = [t0, t1, t2, t3, t4];
                            assert_eq!(TRITS_OF_INTEGER[pack_trit(tup) as usize], tup);
                        }
                    }
                }
            }
        }
        for q0 in 0..5 {
            for q1 in 0..5 {
                for q2 in 0..5 {
                    let tup = [q0, q1, q2];
                    assert_eq!(QUINTS_OF_INTEGER[pack_quint(tup) as usize], tup);
                }
            }
        }
    }

    struct Rng(u32);
    impl Rng {
        fn next_u32(&mut self) -> u32 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            self.0 = x;
            x
        }
    }

    /// The production `encode_trit_sequence` is the exact inverse of
    /// `decode_trit_sequence` for every supported width and element count.
    #[test]
    fn trit_sequences_round_trip_every_width_and_count() {
        let mut rng = Rng(0x1234_5678);
        for bits in 0u32..=6 {
            let levels = 3u32 << bits;
            for count in 1u32..=18 {
                if count * bits + (count * 8 + 4) / 5 > 120 {
                    continue;
                }
                for _ in 0..64 {
                    let vals: Vec<u8> = (0..count)
                        .map(|_| (rng.next_u32() % levels) as u8)
                        .collect();
                    let mut blk = [0u8; 16];
                    encode_trit_sequence(&mut blk, 3, bits, &vals);
                    let mut out = vec![0u8; count as usize];
                    decode_trit_sequence(&blk, 3, bits, count, &mut out);
                    assert_eq!(out, vals, "trit bits={bits} count={count}");
                }
            }
        }
    }

    /// The production `encode_quint_sequence` is the exact inverse of
    /// `decode_quint_sequence` for every supported width and element count.
    #[test]
    fn quint_sequences_round_trip_every_width_and_count() {
        let mut rng = Rng(0x9E37_79B9);
        for bits in 0u32..=5 {
            let levels = 5u32 << bits;
            for count in 1u32..=18 {
                if count * bits + (count * 7 + 2) / 3 > 120 {
                    continue;
                }
                for _ in 0..64 {
                    let vals: Vec<u8> = (0..count)
                        .map(|_| (rng.next_u32() % levels) as u8)
                        .collect();
                    let mut blk = [0u8; 16];
                    encode_quint_sequence(&mut blk, 3, bits, &vals);
                    let mut out = vec![0u8; count as usize];
                    decode_quint_sequence(&blk, 3, bits, count, &mut out);
                    assert_eq!(out, vals, "quint bits={bits} count={count}");
                }
            }
        }
    }
}
