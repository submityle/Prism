//! Raw (uncompressed) `11-bit` unorm bitstream packing.
//!
//! This is the **uncompressed / raw** `DM` displacement layout: the `N`
//! micro-vertex codes are written back-to-back as `11-bit` little-endian
//! fields into a flat byte buffer. Code `j` occupies global bit positions
//! `j * 11 .. j * 11 + 11`; within that field bit `i` (bit `0` is the least
//! significant) is written to global bit `j * 11 + i`, and global bit `p`
//! lives in byte `p / 8` at bit `p % 8` (least significant first).
//!
//! This is deliberately **not** the vendor anchor-plus-correction block
//! compression. The raw `11-bit` stream is a legitimate, exactly correct
//! golden: a future compressor can consume these codes and emit the vendor
//! block format without changing anything upstream. Keeping the golden raw
//! means it is trivially verifiable bit-for-bit.

use alloc::vec;
use alloc::vec::Vec;

/// Number of significant bits in one `unorm` displacement code.
pub const BITS_PER_CODE: usize = 11;

/// Returns the number of bytes needed to pack `count` `11-bit` codes.
#[must_use]
pub fn packed_len(count: usize) -> usize {
    (count * BITS_PER_CODE).div_ceil(8)
}

/// Packs `codes` into the raw little-endian `11-bit` bitstream.
///
/// Only the low `11` bits of each code are stored; any higher bits are
/// ignored, mirroring the `0..=2047` domain of a valid `unorm` code.
#[must_use]
pub fn pack_unorm11(codes: &[u16]) -> Vec<u8> {
    let mut bytes = vec![0u8; packed_len(codes.len())];
    for (j, &code) in codes.iter().enumerate() {
        let masked = code & 0x07FF;
        for i in 0..BITS_PER_CODE {
            if (masked >> i) & 1 == 1 {
                let p = j * BITS_PER_CODE + i;
                bytes[p / 8] |= 1u8 << (p % 8);
            }
        }
    }
    bytes
}

/// Unpacks `count` `11-bit` codes from a raw bitstream.
///
/// Returns [`None`] when `bytes` is too short to hold `count` codes. Every
/// returned code is in `0..=2047`.
#[must_use]
pub fn unpack_unorm11(bytes: &[u8], count: usize) -> Option<Vec<u16>> {
    if bytes.len() < packed_len(count) {
        return None;
    }
    let mut out = Vec::with_capacity(count);
    for j in 0..count {
        let mut code = 0u16;
        for i in 0..BITS_PER_CODE {
            let p = j * BITS_PER_CODE + i;
            let bit = (bytes[p / 8] >> (p % 8)) & 1;
            code |= u16::from(bit) << i;
        }
        out.push(code);
    }
    Some(out)
}
