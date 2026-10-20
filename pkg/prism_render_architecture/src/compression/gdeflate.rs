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
//!   bounded [`super::deflate::inflate_bounded`] core.
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
    /// The tile layout, reserved header byte, or a tile's sizes are invalid.
    CorruptTile,
    /// Bytes remained after the last declared tile payload.
    TrailingData,
    /// A decoded tile's length did not match its descriptor.
    TileSizeMismatch,
    /// A decoded tile's checksum did not match its descriptor.
    TileChecksumMismatch,
    /// The inner `DEFLATE` stream of a tile failed to inflate.
    Inflate(InflateError),
    /// The allocator rejected a decode buffer growth request.
    AllocationFailed,
    /// The declared stream output exceeds the caller's aggregate byte budget.
    OutputLimitExceeded,
}

impl From<InflateError> for GDeflateError {
    fn from(error: InflateError) -> Self {
        match error {
            InflateError::AllocationFailed => Self::AllocationFailed,
            error => Self::Inflate(error),
        }
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
/// has an invalid layout or trailing container bytes, fails tile integrity,
/// exceeds a tile's declared output limit, or cannot allocate decode storage.
/// The entire framing is checked before allocating decode buffers. Each tile
/// is limited to 64 KiB and its declared size is enforced before decoded writes.
/// Use [`gdeflate_decompress_bounded`] to also cap the aggregate stream output.
pub fn gdeflate_decompress(input: &[u8]) -> Result<Vec<u8>, GDeflateError> {
    gdeflate_decompress_bounded(input, usize::MAX)
}

/// Decompresses a container with a caller-specified aggregate output budget.
///
/// All descriptors, ranges, and their sum must agree with the header, and that
/// sum must fit `max_output_bytes`, before any decode allocation or inflation.
/// A budget is not an allocation hint. Per-tile bounds still apply, including
/// to empty and noncanonical short tiles. Each raw stream must consume exactly
/// its declared compressed bytes; lane padding and final-byte unused bits are
/// not part of that check.
///
/// # Errors
/// Returns [`GDeflateError::OutputLimitExceeded`] for an over-budget stream,
/// [`GDeflateError::Inflate`] containing [`InflateError::TrailingData`] for
/// trailing raw bytes inside a tile, or the errors of [`gdeflate_decompress`].
pub fn gdeflate_decompress_bounded(
    input: &[u8],
    max_output_bytes: usize,
) -> Result<Vec<u8>, GDeflateError> {
    let header = GDeflateHeader::read(input)?;
    if usize::from(header.lane_count) != LANES {
        return Err(GDeflateError::UnsupportedLaneCount(header.lane_count));
    }
    if header.tile_shift != TILE_SHIFT || input[7] != 0 {
        return Err(GDeflateError::CorruptTile);
    }
    let tile_count =
        usize::try_from(header.tile_count).map_err(|_| GDeflateError::TruncatedStream)?;
    let descriptor_end = tile_count
        .checked_mul(DESCRIPTOR_LEN)
        .and_then(|length| HEADER_LEN.checked_add(length))
        .ok_or(GDeflateError::TruncatedStream)?;
    let descriptors = input
        .get(HEADER_LEN..descriptor_end)
        .ok_or(GDeflateError::TruncatedStream)?;
    let uncompressed_size =
        usize::try_from(header.uncompressed_size).map_err(|_| GDeflateError::TileSizeMismatch)?;

    // Validate the entire framing without allocating or inflating anything.
    // In particular, neither header count nor total size is an allocation hint.
    let mut offset = descriptor_end;
    let mut total = 0usize;
    for bytes in descriptors.chunks_exact(DESCRIPTOR_LEN) {
        let descriptor = GDeflateTileDescriptor::read(bytes)?;
        if descriptor.uncompressed_size as usize > TILE_SIZE || descriptor.compressed_size == 0 {
            return Err(GDeflateError::CorruptTile);
        }
        total = total
            .checked_add(descriptor.uncompressed_size as usize)
            .ok_or(GDeflateError::TileSizeMismatch)?;
        tile_payload(input, &mut offset, descriptor.compressed_size as usize)?;
    }
    if total != uncompressed_size {
        return Err(GDeflateError::TileSizeMismatch);
    }
    if offset != input.len() {
        return Err(GDeflateError::TrailingData);
    }
    if uncompressed_size > max_output_bytes {
        return Err(GDeflateError::OutputLimitExceeded);
    }

    let mut out = Vec::new();
    offset = descriptor_end;
    for bytes in descriptors.chunks_exact(DESCRIPTOR_LEN) {
        let descriptor = GDeflateTileDescriptor::read(bytes)?;
        let compressed_size = descriptor.compressed_size as usize;
        let chunk = tile_payload(input, &mut offset, compressed_size)?;
        let deflated = deinterleave(chunk, compressed_size)?;
        let (tile, consumed) = deflate::inflate_bounded_with_consumed(
            &deflated,
            descriptor.uncompressed_size as usize,
        )?;
        if consumed != compressed_size {
            return Err(InflateError::TrailingData.into());
        }
        if tile.len() != descriptor.uncompressed_size as usize {
            return Err(GDeflateError::TileSizeMismatch);
        }
        if checksum(&tile) != descriptor.checksum {
            return Err(GDeflateError::TileChecksumMismatch);
        }
        out.try_reserve(tile.len())
            .map_err(|_| GDeflateError::AllocationFailed)?;
        out.extend_from_slice(&tile);
    }
    if out.len() != uncompressed_size {
        return Err(GDeflateError::TileSizeMismatch);
    }
    Ok(out)
}

/// Borrows one padded payload, checking rounding and cursor arithmetic.
fn tile_payload<'a>(
    input: &'a [u8],
    offset: &mut usize,
    compressed_size: usize,
) -> Result<&'a [u8], GDeflateError> {
    let padded = compressed_size
        .div_ceil(GROUP)
        .checked_mul(GROUP)
        .ok_or(GDeflateError::TruncatedStream)?;
    let end = offset
        .checked_add(padded)
        .ok_or(GDeflateError::TruncatedStream)?;
    let payload = input
        .get(*offset..end)
        .ok_or(GDeflateError::TruncatedStream)?;
    *offset = end;
    Ok(payload)
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
    let rounds = interleaved.len() / GROUP;
    let mut linear = Vec::new();
    linear
        .try_reserve_exact(compressed_size)
        .map_err(|_| GDeflateError::AllocationFailed)?;
    for offset in 0..compressed_size {
        let word = offset / WORD;
        let stored_word = (word % LANES) * rounds + word / LANES;
        linear.push(interleaved[stored_word * WORD + offset % WORD]);
    }
    Ok(linear)
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

    fn frame_raw_tiles(tiles: &[(&[u8], &[u8])]) -> Vec<u8> {
        let mut out = Vec::new();
        GDeflateHeader {
            magic: MAGIC,
            version: VERSION,
            lane_count: LANES as u8,
            tile_shift: TILE_SHIFT,
            tile_count: tiles.len() as u32,
            uncompressed_size: tiles.iter().map(|(_, data)| data.len() as u64).sum(),
        }
        .write(&mut out);
        for &(raw, data) in tiles {
            GDeflateTileDescriptor {
                uncompressed_size: data.len() as u32,
                compressed_size: raw.len() as u32,
                checksum: checksum(data),
            }
            .write(&mut out);
        }
        for &(raw, _) in tiles {
            out.extend_from_slice(&interleave(raw));
        }
        out
    }

    fn set_single_tile_size(input: &mut [u8], size: u32) {
        input[12..20].copy_from_slice(&u64::from(size).to_le_bytes());
        input[HEADER_LEN..HEADER_LEN + 4].copy_from_slice(&size.to_le_bytes());
    }

    fn raw_block_fixtures() -> Vec<(Vec<u8>, Vec<u8>)> {
        // Independent fixed tree: literal A, length 258/distance 1, EOB.
        let mut bits = vec![true, true, false];
        for (code, width) in [(0x71u16, 8), (0xC5, 8), (0, 5), (0, 7)] {
            bits.extend((0..width).rev().map(|shift| code & (1 << shift) != 0));
        }
        let pack = |bits: &[bool]| {
            let mut bytes = vec![0; bits.len().div_ceil(8)];
            for (i, &bit) in bits.iter().enumerate() {
                bytes[i / 8] |= u8::from(bit) << (i % 8);
            }
            bytes
        };
        let fixed = pack(&bits);
        // Produced by an independent zlib compressor, not Prism's encoder.
        let hex = "edcab10d83301000c0559e0932062d4d0608f0912c2516022301d3932dd25c73d5f5c333e6b3bebe658ab6666ef18852e75cf2476d717dca18ef72b47dcd2e7adbb66ddbb66ddbb66ddbb66ddbb66ddbb66ddbb66ddbb66ddbb6ffb86f";
        let dynamic: Vec<u8> = (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
            .collect();
        let dynamic_data = b"GPU dynamic trees / independent zlib fixture! ".repeat(200);
        assert_eq!((dynamic[0] >> 1) & 3, 2);

        // Non-final fixed, non-final stored X, then a final dynamic block.
        // The stored header starts at bit 31, testing cross-byte alignment.
        bits[0] = false;
        bits.extend_from_slice(&[false, false, false]);
        while !bits.len().is_multiple_of(8) {
            bits.push(false);
        }
        let mut mixed = pack(&bits);
        mixed.extend_from_slice(&[1, 0, 0xFE, 0xFF, b'X']);
        mixed.extend_from_slice(&dynamic);
        let mut mixed_data = vec![b'A'; 259];
        mixed_data.push(b'X');
        mixed_data.extend_from_slice(&dynamic_data);
        vec![
            (vec![1, 3, 0, 0xFC, 0xFF, b'a', b'b', b'c'], b"abc".to_vec()),
            (fixed, vec![b'A'; 259]),
            (dynamic, dynamic_data),
            (mixed, mixed_data),
        ]
    }

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

    #[test]
    fn header_contract_rejects_unknown_layouts() {
        for (offset, value, expected) in [
            (4, 2, GDeflateError::UnsupportedVersion(2)),
            (5, 0, GDeflateError::UnsupportedLaneCount(0)),
            (6, 15, GDeflateError::CorruptTile),
            (6, 17, GDeflateError::CorruptTile),
            (7, 1, GDeflateError::CorruptTile),
        ] {
            let mut input = gdeflate_compress(&[]);
            input[offset] = value;
            assert_eq!(gdeflate_decompress(&input), Err(expected));
        }
    }

    #[test]
    fn forged_counts_totals_and_sizes_are_rejected_during_preflight() {
        let encoded = gdeflate_compress(b"bounded framing");
        for count in [0u32, u32::MAX] {
            let mut input = encoded.clone();
            input[8..12].copy_from_slice(&count.to_le_bytes());
            assert_eq!(
                gdeflate_decompress(&input),
                Err(if count == 0 {
                    GDeflateError::TileSizeMismatch
                } else {
                    GDeflateError::TruncatedStream
                })
            );
        }
        for total in [0, 1, 1u64 << 40, u64::MAX] {
            let mut input = encoded.clone();
            input[12..20].copy_from_slice(&total.to_le_bytes());
            assert_eq!(
                gdeflate_decompress(&input),
                Err(GDeflateError::TileSizeMismatch)
            );
        }
        for (size, expected) in [
            (0, GDeflateError::CorruptTile),
            (u32::MAX, GDeflateError::TruncatedStream),
        ] {
            let mut input = encoded.clone();
            input[HEADER_LEN + 4..HEADER_LEN + 8].copy_from_slice(&size.to_le_bytes());
            assert_eq!(gdeflate_decompress(&input), Err(expected));
        }
        let mut input = encoded;
        set_single_tile_size(&mut input, TILE_SIZE as u32 + 1);
        assert_eq!(gdeflate_decompress(&input), Err(GDeflateError::CorruptTile));
    }

    #[test]
    fn every_truncated_prefix_and_outer_trailing_byte_is_rejected() {
        let input = frame_raw_tiles(&[(&[1, 3, 0, 0xFC, 0xFF, b'a', b'b', b'c'], b"abc")]);
        for length in 0..input.len() {
            assert_eq!(
                gdeflate_decompress(&input[..length]),
                Err(GDeflateError::TruncatedStream),
                "prefix {length}"
            );
        }
        for encoded in [gdeflate_compress(&[]), input] {
            for extra in [1, GROUP] {
                let mut input = encoded.clone();
                input.resize(input.len() + extra, 0);
                assert_eq!(
                    gdeflate_decompress(&input),
                    Err(GDeflateError::TrailingData)
                );
            }
        }
    }

    #[test]
    fn preflight_finishes_before_decoding_a_corrupt_first_tile() {
        let mut input = frame_raw_tiles(&[(&[7], b""), (&[3, 0], b"")]);
        input.pop();
        assert_eq!(
            gdeflate_decompress(&input),
            Err(GDeflateError::TruncatedStream)
        );
        let mut input = frame_raw_tiles(&[(&[7], b"")]);
        input.push(0);
        assert_eq!(
            gdeflate_decompress(&input),
            Err(GDeflateError::TrailingData)
        );
    }

    #[test]
    fn all_block_types_enforce_exact_zero_and_undersized_tile_budgets() {
        for (raw, data) in raw_block_fixtures() {
            let encoded = frame_raw_tiles(&[(&raw, &data)]);
            assert_eq!(gdeflate_decompress(&encoded).unwrap(), data);
            for limit in [0, data.len() - 1] {
                let mut input = encoded.clone();
                set_single_tile_size(&mut input, limit as u32);
                assert_eq!(
                    gdeflate_decompress(&input),
                    Err(GDeflateError::Inflate(InflateError::OutputLimitExceeded))
                );
            }
            let mut input = encoded.clone();
            set_single_tile_size(&mut input, data.len() as u32 + 1);
            assert_eq!(
                gdeflate_decompress(&input),
                Err(GDeflateError::TileSizeMismatch)
            );
            let mut input = encoded;
            input[HEADER_LEN + 8] ^= 1;
            assert_eq!(
                gdeflate_decompress(&input),
                Err(GDeflateError::TileChecksumMismatch)
            );
        }
    }

    #[test]
    fn explicit_empty_and_noncanonical_small_tiles_remain_valid() {
        let empty_stored = [1, 0, 0, 0xFF, 0xFF];
        let empty_fixed = [3, 0];
        let stored = [1, 3, 0, 0xFC, 0xFF, b'a', b'b', b'c'];
        for raw in [&empty_stored[..], &empty_fixed[..]] {
            let input = frame_raw_tiles(&[(raw, b"")]);
            assert_eq!(gdeflate_decompress(&input).unwrap(), b"");
        }
        let input = frame_raw_tiles(&[
            (&empty_stored, b""),
            (&stored, b"abc"),
            (&empty_fixed, b""),
            (&stored, b"abc"),
        ]);
        assert_eq!(gdeflate_decompress(&input).unwrap(), b"abcabc");
    }

    #[test]
    fn tiles_cannot_borrow_another_tiles_output_budget() {
        let raw = [1, 3, 0, 0xFC, 0xFF, b'a', b'b', b'c'];
        let encoded = frame_raw_tiles(&[(&raw, b"abc"), (&raw, b"abc")]);
        for (first, second) in [(2u32, 4u32), (3, 2)] {
            let mut input = encoded.clone();
            input[12..20].copy_from_slice(&u64::from(first + second).to_le_bytes());
            input[HEADER_LEN..HEADER_LEN + 4].copy_from_slice(&first.to_le_bytes());
            input[HEADER_LEN + DESCRIPTOR_LEN..HEADER_LEN + DESCRIPTOR_LEN + 4]
                .copy_from_slice(&second.to_le_bytes());
            assert_eq!(
                gdeflate_decompress(&input),
                Err(GDeflateError::Inflate(InflateError::OutputLimitExceeded))
            );
        }
    }

    #[test]
    fn full_tile_stored_blocks_allow_compressed_bytes_above_tile_size() {
        let data = vec![b'Z'; TILE_SIZE];
        // A stored block has a 16-bit length, so use two blocks for 64 KiB.
        let mut raw = vec![0, 0xFF, 0xFF, 0, 0];
        raw.extend_from_slice(&data[..TILE_SIZE - 1]);
        raw.extend_from_slice(&[1, 1, 0, 0xFE, 0xFF, b'Z']);
        assert!(raw.len() > TILE_SIZE);
        let input = frame_raw_tiles(&[(&raw, &data)]);
        assert_eq!(gdeflate_decompress(&input).unwrap(), data);
        let mut input = input;
        set_single_tile_size(&mut input, TILE_SIZE as u32 - 1);
        assert_eq!(
            gdeflate_decompress(&input),
            Err(GDeflateError::Inflate(InflateError::OutputLimitExceeded))
        );
    }

    #[test]
    fn checked_payload_offsets_fail_without_advancing_the_cursor() {
        for (start, size) in [(usize::MAX - 63, 1), (0, usize::MAX), (1, GROUP)] {
            let mut cursor = start;
            assert_eq!(
                tile_payload(&[0; GROUP], &mut cursor, size),
                Err(GDeflateError::TruncatedStream)
            );
            assert_eq!(cursor, start);
        }
        assert_eq!(
            deinterleave(&[0; GROUP - 1], 1),
            Err(GDeflateError::CorruptTile)
        );
        assert_eq!(
            deinterleave(&[0; GROUP], GROUP + 1),
            Err(GDeflateError::CorruptTile)
        );
    }

    #[test]
    fn allocation_errors_keep_their_structured_category() {
        assert_eq!(
            GDeflateError::from(InflateError::AllocationFailed),
            GDeflateError::AllocationFailed
        );
        assert_eq!(
            GDeflateError::from(InflateError::OutputLimitExceeded),
            GDeflateError::Inflate(InflateError::OutputLimitExceeded)
        );
    }

    #[test]
    fn inner_raw_trailing_bytes_are_rejected_in_every_block_kind_and_tile_position() {
        let mut fixtures = raw_block_fixtures();
        fixtures.extend([
            (vec![3, 0], Vec::new()),
            (vec![1, 0, 0, 0xFF, 0xFF], Vec::new()),
        ]);
        for (raw, data) in fixtures {
            for suffix in [vec![0], vec![0xFF], vec![3, 0], vec![0; GROUP], raw.clone()] {
                let mut extended = raw.clone();
                extended.extend_from_slice(&suffix);
                for tiles in [
                    vec![(extended.as_slice(), data.as_slice()), (&[3, 0], &b""[..])],
                    vec![(&[3, 0], &b""[..]), (extended.as_slice(), data.as_slice())],
                ] {
                    let input = frame_raw_tiles(&tiles);
                    assert_eq!(
                        gdeflate_decompress(&input),
                        Err(GDeflateError::Inflate(InflateError::TrailingData))
                    );
                    assert_eq!(
                        gdeflate_decompress_bounded(&input, data.len()),
                        Err(GDeflateError::Inflate(InflateError::TrailingData))
                    );
                }
            }
        }
    }

    #[test]
    fn terminal_bits_and_lane_padding_are_not_mistaken_for_raw_trailing_bytes() {
        for raw in [&[3, 0xFC][..], &[0xF9, 0, 0, 0xFF, 0xFF]] {
            let mut input = frame_raw_tiles(&[(raw, b"")]);
            // For one round, stored and logical word positions coincide.
            input[HEADER_LEN + DESCRIPTOR_LEN + raw.len()..].fill(0xA5);
            assert_eq!(gdeflate_decompress_bounded(&input, 0).unwrap(), b"");
        }
        // Also distinguish raw bytes from transpose padding across many rounds.
        let data = lcg_bytes(2_003, 0xCAFEBABE);
        let raw = deflate_encode::deflate(&data);
        let mut input = frame_raw_tiles(&[(&raw, &data)]);
        let payload = &mut input[HEADER_LEN + DESCRIPTOR_LEN..];
        let rounds = payload.len() / GROUP;
        assert!(rounds > 1);
        for offset in raw.len()..payload.len() {
            let word = offset / WORD;
            let stored_word = (word % LANES) * rounds + word / LANES;
            payload[stored_word * WORD + offset % WORD] = 0xA5;
        }
        assert_eq!(
            gdeflate_decompress_bounded(&input, data.len()).unwrap(),
            data
        );
    }

    #[test]
    fn aggregate_budget_accepts_exact_empty_multitile_and_full_tile_boundaries() {
        let fixtures = raw_block_fixtures();
        let mut tiles = vec![(&[3, 0][..], &b""[..])];
        tiles.extend(
            fixtures
                .iter()
                .map(|(raw, data)| (raw.as_slice(), data.as_slice())),
        );
        tiles.push((&[1, 0, 0, 0xFF, 0xFF], b""));
        let expected: Vec<u8> = fixtures
            .iter()
            .flat_map(|(_, data)| data.iter().copied())
            .collect();
        let encoded = frame_raw_tiles(&tiles);
        assert_eq!(
            gdeflate_decompress_bounded(&encoded, expected.len()).unwrap(),
            expected
        );
        assert_eq!(
            gdeflate_decompress_bounded(&encoded, usize::MAX).unwrap(),
            expected
        );
        for limit in [0, expected.len() - 1] {
            assert_eq!(
                gdeflate_decompress_bounded(&encoded, limit),
                Err(GDeflateError::OutputLimitExceeded)
            );
        }
        for empty in [
            gdeflate_compress(&[]),
            frame_raw_tiles(&[(&[3, 0], b""), (&[3, 0], b"")]),
        ] {
            assert_eq!(gdeflate_decompress_bounded(&empty, 0).unwrap(), b"");
        }
        let data = vec![b'A'; 2 * TILE_SIZE + 1];
        let encoded = gdeflate_compress(&data);
        assert_eq!(
            gdeflate_decompress_bounded(&encoded, data.len()).unwrap(),
            data
        );
        assert_eq!(
            gdeflate_decompress_bounded(&encoded, data.len() - 1),
            Err(GDeflateError::OutputLimitExceeded)
        );
    }

    #[test]
    fn aggregate_budget_and_complete_framing_are_checked_before_any_inflate() {
        // Sum is 6, but the first raw stream is reserved and cannot inflate.
        let valid = [1, 3, 0, 0xFC, 0xFF, b'a', b'b', b'c'];
        let input = frame_raw_tiles(&[(&[7], b"abc"), (&valid, b"abc")]);
        assert_eq!(
            gdeflate_decompress_bounded(&input, 5),
            Err(GDeflateError::OutputLimitExceeded)
        );
        assert_eq!(
            gdeflate_decompress_bounded(&input, 6),
            Err(GDeflateError::Inflate(InflateError::ReservedBlockType))
        );
        let mut truncated = input.clone();
        truncated.pop();
        assert_eq!(
            gdeflate_decompress_bounded(&truncated, 5),
            Err(GDeflateError::TruncatedStream)
        );
        let mut forged = input.clone();
        forged[12..20].copy_from_slice(&3u64.to_le_bytes());
        assert_eq!(
            gdeflate_decompress_bounded(&forged, 3),
            Err(GDeflateError::TileSizeMismatch)
        );
        let mut oversized = input;
        oversized[HEADER_LEN + DESCRIPTOR_LEN..HEADER_LEN + DESCRIPTOR_LEN + 4]
            .copy_from_slice(&(TILE_SIZE as u32 + 1).to_le_bytes());
        assert_eq!(
            gdeflate_decompress_bounded(&oversized, usize::MAX),
            Err(GDeflateError::CorruptTile)
        );
    }

    #[test]
    fn aggregate_budget_never_replaces_independent_tile_budgets() {
        let raw = [1, 3, 0, 0xFC, 0xFF, b'a', b'b', b'c'];
        let mut input = frame_raw_tiles(&[(&raw, b"abc"), (&raw, b"abc")]);
        input[HEADER_LEN..HEADER_LEN + 4].copy_from_slice(&2u32.to_le_bytes());
        input[HEADER_LEN + DESCRIPTOR_LEN..HEADER_LEN + DESCRIPTOR_LEN + 4]
            .copy_from_slice(&4u32.to_le_bytes());
        assert_eq!(
            gdeflate_decompress_bounded(&input, 6),
            Err(GDeflateError::Inflate(InflateError::OutputLimitExceeded))
        );
        assert_eq!(
            gdeflate_decompress_bounded(&input, 5),
            Err(GDeflateError::OutputLimitExceeded)
        );
        // History is local to a tile: the second fixed stream starts with a
        // match (length 3/distance 1), despite the first tile providing 'A'.
        let input = frame_raw_tiles(&[(&[1, 1, 0, 0xFE, 0xFF, b'A'], b"A"), (&[3, 2, 0], b"AAA")]);
        assert_eq!(
            gdeflate_decompress_bounded(&input, 4),
            Err(GDeflateError::Inflate(InflateError::DistanceTooFar))
        );
    }
}
