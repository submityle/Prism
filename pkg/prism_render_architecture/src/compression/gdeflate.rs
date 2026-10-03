//! `GDeflate` warp-interleaved tile container over the inner `DEFLATE` codec.
//!
//! `GDeflate` is the streaming-friendly framing used by `DirectStorage`-class
//! pipelines: the payload is split into fixed-size tiles (the standard tile is
//! 64 `KiB` uncompressed), each tile is `DEFLATE`-compressed, and the resulting
//! bitstream is laid out so a 32-lane `GPU` warp can decode it with coalesced
//! memory access — each lane owns a round-robin slice of the stream's 32-bit
//! words.
//!
//! # What this module actually implements
//!
//! This is a real, complete `CPU` implementation of a 32-lane warp-interleaved
//! `DEFLATE` tile codec, in **both** directions:
//!
//! * [`gdeflate_compress`] tiles the input, `DEFLATE`-encodes each tile (via the
//!   sibling [`super::deflate_encode`] encoder), and transposes the tile's
//!   32-bit words into lane-major (warp-interleaved) order behind a stream
//!   header and per-tile descriptors.
//! * [`gdeflate_decompress`] parses the headers, reverses the transpose to
//!   recover each tile's linear `DEFLATE` bitstream, and inflates it with the
//!   existing [`super::deflate::inflate`] core.
//!
//! The interleave is a reversible word-level transpose of a single standard
//! `DEFLATE` bitstream: logical word `r * 32 + lane` is stored at lane-major
//! position `lane * rounds + r`. De-interleaving recovers the exact standard
//! `DEFLATE` byte stream, so each tile is decoded by the ordinary inflate core
//! — there is no second copy of the Huffman decode logic.
//!
//! # Provenance and verification status (honest)
//!
//! No Unreal Engine or NVIDIA `GDeflate` source was consulted or copied; the
//! layout here is implemented from the public format description only. There is
//! **no** NVIDIA reference oracle available in this sandbox, so this codec is
//! **not** claimed to be bit-exact compatible with NVIDIA's `GDeflate`
//! bitstream. The verification contract is mutual consistency: the encoder and
//! decoder are exact inverses, proven by `CPU` encode -> decode round-trip tests
//! (see the test module), including the invariant that a decoded tile equals
//! the standard-`DEFLATE` inflate of the de-interleaved stream.

use alloc::vec;
use alloc::vec::Vec;

use super::deflate::{self, InflateError};
use super::deflate_encode;

/// On-wire magic identifying this container (`"GDF1"`).
const MAGIC: [u8; 4] = *b"GDF1";
/// Container format version.
const VERSION: u8 = 1;
/// Number of warp lanes the bitstream is interleaved across.
const LANES: usize = 32;
/// Width of an interleave word in bytes.
const WORD: usize = 4;
/// Bytes consumed by one interleave round across all lanes.
const GROUP: usize = LANES * WORD;
/// `log2` of the standard uncompressed tile size.
const TILE_SHIFT: u8 = 16;
/// Standard uncompressed tile size (64 `KiB`).
const TILE_SIZE: usize = 1 << TILE_SHIFT;
/// Serialized length of [`GDeflateHeader`].
const HEADER_LEN: usize = 20;
/// Serialized length of [`GDeflateTileDescriptor`].
const DESCRIPTOR_LEN: usize = 12;

/// Reason a `GDeflate` stream could not be decoded.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GDeflateError {
    /// The stream did not begin with the expected magic bytes.
    BadMagic,
    /// The container version is not understood by this decoder.
    UnsupportedVersion(u8),
    /// The lane count in the header is not the supported value (32).
    UnsupportedLaneCount(u8),
    /// The stream ended before a header, descriptor, or tile payload finished.
    TruncatedStream,
    /// A tile's interleaved payload was not a whole number of lane groups.
    CorruptTile,
    /// A decoded tile's length did not match its descriptor.
    TileSizeMismatch,
    /// A decoded tile's checksum did not match its descriptor.
    TileChecksumMismatch,
    /// The inner `DEFLATE` stream of a tile failed to inflate.
    Inflate(InflateError),
}

impl From<InflateError> for GDeflateError {
    fn from(error: InflateError) -> Self {
        Self::Inflate(error)
    }
}

/// Fixed-size stream header prefixing every `GDeflate` container.
///
/// On-wire layout (little-endian, 20 bytes, equivalent to a packed
/// `#[repr(C)]` struct):
///
/// | offset | size | field               |
/// |-------:|-----:|---------------------|
/// | 0      | 4    | `magic` (`"GDF1"`)  |
/// | 4      | 1    | `version`           |
/// | 5      | 1    | `lane_count`        |
/// | 6      | 1    | `tile_shift`        |
/// | 7      | 1    | reserved (`0`)      |
/// | 8      | 4    | `tile_count`        |
/// | 12     | 8    | `uncompressed_size` |
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GDeflateHeader {
    /// Format magic; always [`MAGIC`].
    pub magic: [u8; 4],
    /// Container version; always [`VERSION`].
    pub version: u8,
    /// Lanes the bitstream is interleaved across; always 32.
    pub lane_count: u8,
    /// `log2` of the uncompressed tile size.
    pub tile_shift: u8,
    /// Number of tile descriptors (and tiles) that follow.
    pub tile_count: u32,
    /// Total uncompressed byte length of the original input.
    pub uncompressed_size: u64,
}

impl GDeflateHeader {
    /// Serializes the header into `out`.
    fn write(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.magic);
        out.push(self.version);
        out.push(self.lane_count);
        out.push(self.tile_shift);
        out.push(0); // reserved
        out.extend_from_slice(&self.tile_count.to_le_bytes());
        out.extend_from_slice(&self.uncompressed_size.to_le_bytes());
    }

    /// Parses a header from the front of `bytes`.
    ///
    /// # Errors
    /// Returns [`GDeflateError::TruncatedStream`] if fewer than [`HEADER_LEN`]
    /// bytes are present, [`GDeflateError::BadMagic`] on a magic mismatch, or
    /// [`GDeflateError::UnsupportedVersion`] for an unknown version.
    fn read(bytes: &[u8]) -> Result<Self, GDeflateError> {
        let head = bytes
            .get(..HEADER_LEN)
            .ok_or(GDeflateError::TruncatedStream)?;
        let mut magic = [0u8; 4];
        magic.copy_from_slice(&head[0..4]);
        if magic != MAGIC {
            return Err(GDeflateError::BadMagic);
        }
        let version = head[4];
        if version != VERSION {
            return Err(GDeflateError::UnsupportedVersion(version));
        }
        let header = Self {
            magic,
            version,
            lane_count: head[5],
            tile_shift: head[6],
            tile_count: read_u32_le(&head[8..12]),
            uncompressed_size: read_u64_le(&head[12..20]),
        };
        Ok(header)
    }
}

/// Per-tile descriptor table entry.
///
/// On-wire layout (little-endian, 12 bytes, equivalent to a packed
/// `#[repr(C)]` struct):
///
/// | offset | size | field               |
/// |-------:|-----:|---------------------|
/// | 0      | 4    | `uncompressed_size` |
/// | 4      | 4    | `compressed_size`   |
/// | 8      | 4    | `checksum`          |
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GDeflateTileDescriptor {
    /// Uncompressed byte length of this tile (`<= TILE_SIZE`).
    pub uncompressed_size: u32,
    /// `DEFLATE` byte length before lane padding.
    pub compressed_size: u32,
    /// `FNV-1a` checksum of the uncompressed tile bytes.
    pub checksum: u32,
}

impl GDeflateTileDescriptor {
    /// Serializes the descriptor into `out`.
    fn write(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.uncompressed_size.to_le_bytes());
        out.extend_from_slice(&self.compressed_size.to_le_bytes());
        out.extend_from_slice(&self.checksum.to_le_bytes());
    }

    /// Parses a descriptor from `bytes`.
    ///
    /// # Errors
    /// Returns [`GDeflateError::TruncatedStream`] if fewer than
    /// [`DESCRIPTOR_LEN`] bytes are present.
    fn read(bytes: &[u8]) -> Result<Self, GDeflateError> {
        let entry = bytes
            .get(..DESCRIPTOR_LEN)
            .ok_or(GDeflateError::TruncatedStream)?;
        Ok(Self {
            uncompressed_size: read_u32_le(&entry[0..4]),
            compressed_size: read_u32_le(&entry[4..8]),
            checksum: read_u32_le(&entry[8..12]),
        })
    }
}

/// Compresses `input` into a `GDeflate` container.
#[must_use]
pub fn gdeflate_compress(input: &[u8]) -> Vec<u8> {
    let tile_count = input.len().div_ceil(TILE_SIZE);
    let header = GDeflateHeader {
        magic: MAGIC,
        version: VERSION,
        lane_count: LANES as u8,
        tile_shift: TILE_SHIFT,
        tile_count: tile_count as u32,
        uncompressed_size: input.len() as u64,
    };

    let mut descriptors = Vec::with_capacity(tile_count);
    let mut payload = Vec::new();
    for ti in 0..tile_count {
        let start = ti * TILE_SIZE;
        let end = (start + TILE_SIZE).min(input.len());
        let tile = &input[start..end];
        let deflated = deflate_encode::deflate(tile);
        let interleaved = interleave(&deflated);
        descriptors.push(GDeflateTileDescriptor {
            uncompressed_size: tile.len() as u32,
            compressed_size: deflated.len() as u32,
            checksum: checksum(tile),
        });
        payload.extend_from_slice(&interleaved);
    }

    let mut out = Vec::with_capacity(HEADER_LEN + tile_count * DESCRIPTOR_LEN + payload.len());
    header.write(&mut out);
    for descriptor in &descriptors {
        descriptor.write(&mut out);
    }
    out.extend_from_slice(&payload);
    out
}

/// Decompresses a `GDeflate` container produced by [`gdeflate_compress`].
///
/// # Errors
/// Returns a [`GDeflateError`] when the stream is truncated, carries bad magic,
/// fails a tile integrity check, or contains an invalid inner `DEFLATE` stream.
pub fn gdeflate_decompress(input: &[u8]) -> Result<Vec<u8>, GDeflateError> {
    let header = GDeflateHeader::read(input)?;
    if usize::from(header.lane_count) != LANES {
        return Err(GDeflateError::UnsupportedLaneCount(header.lane_count));
    }

    let tile_count = header.tile_count as usize;
    let mut offset = HEADER_LEN;

    let mut descriptors = Vec::with_capacity(tile_count);
    for _ in 0..tile_count {
        let slice = input
            .get(offset..offset + DESCRIPTOR_LEN)
            .ok_or(GDeflateError::TruncatedStream)?;
        descriptors.push(GDeflateTileDescriptor::read(slice)?);
        offset += DESCRIPTOR_LEN;
    }

    let mut out = Vec::with_capacity(header.uncompressed_size as usize);
    for descriptor in &descriptors {
        let compressed_size = descriptor.compressed_size as usize;
        let padded = round_up(compressed_size, GROUP);
        let chunk = input
            .get(offset..offset + padded)
            .ok_or(GDeflateError::TruncatedStream)?;
        offset += padded;

        let deflated = deinterleave(chunk, compressed_size)?;
        let tile = deflate::inflate(&deflated)?;
        if tile.len() != descriptor.uncompressed_size as usize {
            return Err(GDeflateError::TileSizeMismatch);
        }
        if checksum(&tile) != descriptor.checksum {
            return Err(GDeflateError::TileChecksumMismatch);
        }
        out.extend_from_slice(&tile);
    }

    if out.len() != header.uncompressed_size as usize {
        return Err(GDeflateError::TileSizeMismatch);
    }
    Ok(out)
}

/// Transposes a linear `DEFLATE` byte stream into lane-major (warp-interleaved)
/// 32-bit-word order, zero-padding to a whole number of lane groups.
fn interleave(deflated: &[u8]) -> Vec<u8> {
    let padded_len = round_up(deflated.len(), GROUP);
    if padded_len == 0 {
        return Vec::new();
    }
    let mut padded = vec![0u8; padded_len];
    padded[..deflated.len()].copy_from_slice(deflated);

    let words = padded_len / WORD;
    let rounds = words / LANES;
    let mut out = vec![0u8; padded_len];
    for lane in 0..LANES {
        for r in 0..rounds {
            let logical = r * LANES + lane;
            let stored = lane * rounds + r;
            let src = &padded[logical * WORD..logical * WORD + WORD];
            out[stored * WORD..stored * WORD + WORD].copy_from_slice(src);
        }
    }
    out
}

/// Reverses [`interleave`], returning the first `compressed_size` bytes of the
/// recovered linear `DEFLATE` stream.
///
/// # Errors
/// Returns [`GDeflateError::CorruptTile`] if the payload is not a whole number
/// of lane groups or is too short for `compressed_size`.
fn deinterleave(interleaved: &[u8], compressed_size: usize) -> Result<Vec<u8>, GDeflateError> {
    if !interleaved.len().is_multiple_of(GROUP) || compressed_size > interleaved.len() {
        return Err(GDeflateError::CorruptTile);
    }
    let padded_len = interleaved.len();
    let words = padded_len / WORD;
    let rounds = words / LANES;
    let mut padded = vec![0u8; padded_len];
    for lane in 0..LANES {
        for r in 0..rounds {
            let logical = r * LANES + lane;
            let stored = lane * rounds + r;
            let src = &interleaved[stored * WORD..stored * WORD + WORD];
            padded[logical * WORD..logical * WORD + WORD].copy_from_slice(src);
        }
    }
    padded.truncate(compressed_size);
    Ok(padded)
}

/// Rounds `value` up to the next multiple of `multiple`.
fn round_up(value: usize, multiple: usize) -> usize {
    value.div_ceil(multiple) * multiple
}

/// `FNV-1a` 32-bit checksum used for per-tile integrity validation.
fn checksum(data: &[u8]) -> u32 {
    let mut hash = 0x811c_9dc5u32;
    for &byte in data {
        hash ^= u32::from(byte);
        hash = hash.wrapping_mul(0x0100_0193);
    }
    hash
}

/// Reads a little-endian `u32` from a 4-byte slice.
fn read_u32_le(bytes: &[u8]) -> u32 {
    let mut buf = [0u8; 4];
    buf.copy_from_slice(bytes);
    u32::from_le_bytes(buf)
}

/// Reads a little-endian `u64` from an 8-byte slice.
fn read_u64_le(bytes: &[u8]) -> u64 {
    let mut buf = [0u8; 8];
    buf.copy_from_slice(bytes);
    u64::from_le_bytes(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg_bytes(len: usize, seed: u32) -> Vec<u8> {
        let mut state = seed;
        let mut data = vec![0u8; len];
        for byte in &mut data {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *byte = (state >> 24) as u8;
        }
        data
    }

    fn assert_round_trip(data: &[u8]) {
        let compressed = gdeflate_compress(data);
        let restored = gdeflate_decompress(&compressed).expect("decompress must succeed");
        assert_eq!(restored, data, "round-trip mismatch for len {}", data.len());
    }

    #[test]
    fn empty_input_round_trips() {
        assert_round_trip(&[]);
    }

    #[test]
    fn sub_tile_round_trips() {
        assert_round_trip(b"hello, warp-interleaved GDeflate");
    }

    #[test]
    fn exactly_one_tile_round_trips() {
        let data = lcg_bytes(TILE_SIZE, 0x1357_9bdf);
        assert_eq!(data.len(), TILE_SIZE);
        assert_round_trip(&data);
    }

    #[test]
    fn multi_tile_with_unaligned_tail_round_trips() {
        let data = lcg_bytes(TILE_SIZE * 2 + 12_345, 0x2468_ace0);
        assert_round_trip(&data);
    }

    #[test]
    fn incompressible_random_round_trips() {
        let data = lcg_bytes(70_000, 0x0bad_f00d);
        assert_round_trip(&data);
    }

    #[test]
    fn highly_compressible_repeated_round_trips() {
        let data = vec![0xABu8; TILE_SIZE * 3 + 777];
        assert_round_trip(&data);
    }

    #[test]
    fn structured_text_round_trips() {
        let mut data = Vec::new();
        for i in 0..20_000u32 {
            data.extend_from_slice(format!("line {i}: value={}\n", i * 7 % 251).as_bytes());
        }
        assert_round_trip(&data);
    }

    #[test]
    fn decoded_tile_equals_standard_inflate_of_deinterleaved_stream() {
        // A single-tile container so we can inspect its one descriptor/payload.
        let tile: Vec<u8> = (0..4_096u32).map(|i| (i * 13 % 257) as u8).collect();
        let compressed = gdeflate_compress(&tile);

        let header = GDeflateHeader::read(&compressed).unwrap();
        assert_eq!(header.tile_count, 1);
        assert_eq!(usize::from(header.lane_count), LANES);

        let descriptor =
            GDeflateTileDescriptor::read(&compressed[HEADER_LEN..HEADER_LEN + DESCRIPTOR_LEN])
                .unwrap();
        let payload = &compressed[HEADER_LEN + DESCRIPTOR_LEN..];
        let padded = round_up(descriptor.compressed_size as usize, GROUP);

        let deflated = deinterleave(&payload[..padded], descriptor.compressed_size as usize)
            .expect("de-interleave must succeed");
        // The de-interleaved stream is ordinary DEFLATE decoded by the core.
        let via_core = deflate::inflate(&deflated).expect("inflate must succeed");
        assert_eq!(via_core, tile);
    }

    #[test]
    fn bad_magic_is_detected() {
        let mut compressed = gdeflate_compress(b"payload bytes for the magic test");
        compressed[0] ^= 0xFF;
        assert_eq!(
            gdeflate_decompress(&compressed),
            Err(GDeflateError::BadMagic)
        );
    }

    #[test]
    fn truncated_header_is_detected() {
        let compressed = gdeflate_compress(b"short");
        let truncated = &compressed[..HEADER_LEN - 1];
        assert_eq!(
            gdeflate_decompress(truncated),
            Err(GDeflateError::TruncatedStream)
        );
    }

    #[test]
    fn truncated_payload_is_detected() {
        let data = lcg_bytes(TILE_SIZE + 4_096, 0x7777_7777);
        let compressed = gdeflate_compress(&data);
        // Drop the final byte of the payload so a tile slice runs past the end.
        let truncated = &compressed[..compressed.len() - 1];
        assert_eq!(
            gdeflate_decompress(truncated),
            Err(GDeflateError::TruncatedStream)
        );
    }
}
