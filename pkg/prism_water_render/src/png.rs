//! A tiny dependency-free `PNG` encoder for `8`-bit `RGBA` images.
//!
//! The renderer reads its frame back as a tightly packed `RGBA8` buffer and
//! needs to persist it as a viewable image. Rather than pull in an image
//! codec, this module writes a minimal but fully valid `PNG`: an `IHDR`
//! header, a single `IDAT` payload wrapped in a `zlib` stream that uses
//! *stored* (uncompressed) `DEFLATE` blocks, and an `IEND` terminator. Stored
//! blocks keep the encoder trivial and exact while still producing a file every
//! `PNG` reader accepts.
//!
//! The checksums (`CRC-32` per chunk and `Adler-32` over the `zlib` payload)
//! are the standard polynomials, written out so the output validates in any
//! viewer.

use alloc::vec::Vec;

/// The eight-byte `PNG` file signature.
const SIGNATURE: [u8; 8] = [137, 80, 78, 71, 13, 10, 26, 10];

/// Maximum payload a single stored `DEFLATE` block can carry.
const MAX_STORED_BLOCK: usize = 0xFFFF;

/// Encodes `rgba` (`width * height * 4` bytes, row-major, top row first) as a
/// complete `PNG` byte stream.
///
/// # Panics
///
/// Panics if `rgba.len()` is not exactly `width * height * 4`, which would mean
/// the caller passed a buffer that does not match the stated dimensions.
#[must_use]
pub fn encode_rgba8(width: u32, height: u32, rgba: &[u8]) -> Vec<u8> {
    let expected = width as usize * height as usize * 4;
    assert_eq!(
        rgba.len(),
        expected,
        "rgba buffer length must equal width * height * 4"
    );

    let mut out = Vec::with_capacity(expected + 1024);
    out.extend_from_slice(&SIGNATURE);

    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.push(8); // bit depth
    ihdr.push(6); // colour type: truecolour with alpha
    ihdr.push(0); // compression: DEFLATE
    ihdr.push(0); // filter: adaptive (only filter 0 is used)
    ihdr.push(0); // interlace: none
    write_chunk(&mut out, b"IHDR", &ihdr);

    let filtered = filter_scanlines(width, height, rgba);
    let zlib = zlib_stored(&filtered);
    write_chunk(&mut out, b"IDAT", &zlib);

    write_chunk(&mut out, b"IEND", &[]);
    out
}

/// Prefixes every scanline with a `0` (no-op) filter byte, as `PNG` requires.
fn filter_scanlines(width: u32, height: u32, rgba: &[u8]) -> Vec<u8> {
    let stride = width as usize * 4;
    let mut out = Vec::with_capacity((stride + 1) * height as usize);
    for row in 0..height as usize {
        out.push(0);
        let start = row * stride;
        out.extend_from_slice(&rgba[start..start + stride]);
    }
    out
}

/// Wraps `data` in a `zlib` stream built entirely from stored `DEFLATE` blocks.
fn zlib_stored(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() + 64);
    // zlib header: CMF = 0x78 (CM=8, CINFO=7), FLG chosen so the pair is a
    // multiple of 31 with no preset dictionary and fastest compression level.
    out.push(0x78);
    out.push(0x01);

    if data.is_empty() {
        // A single empty final stored block.
        out.push(0x01);
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&(!0u16).to_le_bytes());
    } else {
        let mut offset = 0;
        while offset < data.len() {
            let remaining = data.len() - offset;
            let block = remaining.min(MAX_STORED_BLOCK);
            let is_final = offset + block >= data.len();
            out.push(u8::from(is_final)); // BTYPE 00 (stored); bit0 = BFINAL
            let len = block as u16;
            out.extend_from_slice(&len.to_le_bytes());
            out.extend_from_slice(&(!len).to_le_bytes());
            out.extend_from_slice(&data[offset..offset + block]);
            offset += block;
        }
    }

    out.extend_from_slice(&adler32(data).to_be_bytes());
    out
}

/// Appends a length-prefixed, `CRC-32`-suffixed `PNG` chunk to `out`.
fn write_chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    let mut crc_input = Vec::with_capacity(4 + data.len());
    crc_input.extend_from_slice(kind);
    crc_input.extend_from_slice(data);
    out.extend_from_slice(&crc32(&crc_input).to_be_bytes());
}

/// Computes the `Adler-32` checksum of `data`.
fn adler32(data: &[u8]) -> u32 {
    const MOD: u32 = 65521;
    let mut a = 1u32;
    let mut b = 0u32;
    for &byte in data {
        a = (a + u32::from(byte)) % MOD;
        b = (b + a) % MOD;
    }
    (b << 16) | a
}

/// Computes the `CRC-32` checksum used by `PNG` chunks.
fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}
