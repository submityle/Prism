//! Unified `GDeflate` decode policy with an honest GPU/CPU execution report.
//!
//! Every block type is decoded completely on the GPU, including dynamic tree
//! construction. Framing, exact raw-stream consumption, and checksums remain
//! host-side. A count-only host parse validates consumption without generating
//! decoded output; the GPU still builds its own trees and expands all bytes.
//! GPU rejection or an integrity failure retries the complete stream through
//! the CPU golden path and is reported as a fallback, never as successful GPU
//! decode.

use alloc::vec::Vec;
use core::fmt;

use prism_render_architecture::compression::{
    inflate_bounded_with_consumed, validate_deflate_bounded, GDeflateError, GDeflateTileDescriptor,
    InflateError,
};

use crate::{DeflateDecodeError, GpuContext, GpuDeflate};

const HEADER_LEN: usize = 20;
const DESCRIPTOR_LEN: usize = 12;
const LANES: usize = 32;
const WORD: usize = 4;
const GROUP: usize = LANES * WORD;
const TILE_SHIFT: u8 = 16;
const TILE_SIZE: usize = 1 << TILE_SHIFT;

/// The execution path used for one decoded stream.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecodePath {
    /// The complete stream was decoded by the CPU golden implementation.
    Cpu,
    /// All tile blocks, dynamic trees, and LZ77 history were decoded on GPU.
    /// Container parsing, count-only consumption validation, and checksum
    /// verification are still CPU-side; no CPU-decoded bytes are substituted.
    GpuDeflate,
    /// GPU was attempted but the request fell back to the CPU golden path.
    CpuFallback,
}

/// Why a GPU attempt did not become the successful execution path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuFallbackReason {
    /// The caller did not provide a real device.
    NoDevice,
    /// The complete GPU decoder rejected a tile. Any returned bytes came
    /// from a successful, independently validated CPU golden retry.
    DecodeRejected(DeflateDecodeError),
    /// The GPU result failed the same CPU-side integrity checks as the golden
    /// decoder, so the safe CPU path was used.
    Integrity,
}

/// Stable diagnostics for one decode operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodeReport {
    /// Chosen execution path.
    pub path: DecodePath,
    /// Whether the caller supplied a GPU context and a GPU attempt was made.
    pub gpu_attempted: bool,
    /// Total compressed input bytes.
    pub compressed_bytes: usize,
    /// Total decoded output bytes.
    pub uncompressed_bytes: usize,
    /// Fallback reason, when [`Self::path`] is [`DecodePath::CpuFallback`].
    pub fallback: Option<GpuFallbackReason>,
}

/// Decoded bytes together with the execution evidence needed by telemetry and
/// performance tests.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecodedGDeflate {
    /// Exact decoded payload.
    pub bytes: Vec<u8>,
    /// Execution and size evidence.
    pub report: DecodeReport,
}

/// Failure from the unified decoder.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DecoderError {
    /// The stream framing or tile integrity is invalid.
    Format(GDeflateError),
    /// Host output or validated descriptor storage could not be allocated.
    AllocationFailed,
}

impl fmt::Display for DecoderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Format(error) => write!(f, "invalid GDeflate stream: {error:?}"),
            Self::AllocationFailed => write!(f, "GDeflate host allocation failed"),
        }
    }
}

impl std::error::Error for DecoderError {}

impl From<GDeflateError> for DecoderError {
    fn from(error: GDeflateError) -> Self {
        match error {
            GDeflateError::AllocationFailed => Self::AllocationFailed,
            error => Self::Format(error),
        }
    }
}

enum GpuAttemptError {
    Fallback(GpuFallbackReason),
    AllocationFailed,
}

/// Reusable decoder policy. The pipeline is compiled once per GPU context.
pub struct GDeflateDecoder {
    deflate: GpuDeflate,
}

impl GDeflateDecoder {
    /// Compiles the complete GPU DEFLATE pipeline for `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        Self {
            deflate: GpuDeflate::new(ctx),
        }
    }

    /// Decodes through the CPU golden implementation and records that choice.
    pub fn decode_cpu(input: &[u8]) -> Result<DecodedGDeflate, DecoderError> {
        Self::decode_cpu_bounded(input, usize::MAX)
    }

    /// CPU decode with an aggregate output admission limit. Framing and the
    /// declared total are checked before descriptor or decode allocation.
    pub fn decode_cpu_bounded(
        input: &[u8],
        max_output_bytes: usize,
    ) -> Result<DecodedGDeflate, DecoderError> {
        let parsed = parse_stream(input, max_output_bytes)?;
        let bytes = decode_cpu_tiles(&parsed)?;
        Ok(DecodedGDeflate {
            report: DecodeReport {
                path: DecodePath::Cpu,
                gpu_attempted: false,
                compressed_bytes: input.len(),
                uncompressed_bytes: bytes.len(),
                fallback: None,
            },
            bytes,
        })
    }

    /// Decodes with an optional GPU context and reports a deterministic CPU
    /// fallback when the host has no usable adapter.
    pub fn decode_auto(
        ctx: Option<&GpuContext>,
        input: &[u8],
    ) -> Result<DecodedGDeflate, DecoderError> {
        Self::decode_auto_bounded(ctx, input, usize::MAX)
    }

    /// Optional-device decode with the same stream budget on GPU, CPU, and
    /// fallback paths. Admission precedes even GPU pipeline construction.
    pub fn decode_auto_bounded(
        ctx: Option<&GpuContext>,
        input: &[u8],
        max_output_bytes: usize,
    ) -> Result<DecodedGDeflate, DecoderError> {
        let Some(ctx) = ctx else {
            let mut decoded = Self::decode_cpu_bounded(input, max_output_bytes)?;
            decoded.report.path = DecodePath::CpuFallback;
            decoded.report.fallback = Some(GpuFallbackReason::NoDevice);
            return Ok(decoded);
        };
        let parsed = parse_stream(input, max_output_bytes)?;
        Self::new(ctx).decode_parsed(ctx, input.len(), &parsed)
    }

    /// Prefers complete GPU decode for every tile. A GPU error or checksum
    /// failure retries the entire stream on CPU and reports the exact reason.
    pub fn decode(&self, ctx: &GpuContext, input: &[u8]) -> Result<DecodedGDeflate, DecoderError> {
        self.decode_bounded(ctx, input, usize::MAX)
    }

    /// GPU-preferred decode with aggregate admission before any per-request
    /// descriptor, host-output, or device allocation. Existing pipeline state
    /// was allocated by [`Self::new`], independently of this request.
    pub fn decode_bounded(
        &self,
        ctx: &GpuContext,
        input: &[u8],
        max_output_bytes: usize,
    ) -> Result<DecodedGDeflate, DecoderError> {
        let parsed = parse_stream(input, max_output_bytes)?;
        self.decode_parsed(ctx, input.len(), &parsed)
    }

    /// Decodes one independently stored raw-DEFLATE block with the same
    /// complete GPU kernel used by the framed GDF1 path.
    ///
    /// Block-addressed runtime containers store each block as a standalone
    /// RFC 1951 stream rather than as a GDF1 frame.  This entry point keeps
    /// that container format intact: it performs bounded, count-only host
    /// validation, applies the lane transpose in memory, and dispatches the
    /// raw stream directly to the GPU kernel.  A device rejection is retried
    /// through the CPU golden path and is reported as a fallback.
    /// Valid streams larger than the kernel's 64 KiB output limit also use
    /// that bounded fallback; the raw entry point has no GDF1 tile-size limit.
    pub fn decode_raw_deflate_bounded(
        &self,
        ctx: &GpuContext,
        input: &[u8],
        expected_output_size: usize,
        max_output_bytes: usize,
    ) -> Result<DecodedGDeflate, DecoderError> {
        if expected_output_size > max_output_bytes {
            return Err(GDeflateError::OutputLimitExceeded.into());
        }
        let output_bytes =
            validate_deflate_bounded(input, expected_output_size).map_err(GDeflateError::from)?;
        if output_bytes != expected_output_size {
            return Err(GDeflateError::TileSizeMismatch.into());
        }
        let interleaved = interleave_raw(input).ok_or(DecoderError::AllocationFailed)?;
        let (bytes, path, fallback) =
            match self
                .deflate
                .decompress(ctx, &interleaved, input.len(), expected_output_size)
            {
                Ok(bytes) if bytes.len() == expected_output_size => {
                    (bytes, DecodePath::GpuDeflate, None)
                }
                Ok(_) => {
                    let bytes = inflate_raw_cpu(input, expected_output_size)?;
                    (
                        bytes,
                        DecodePath::CpuFallback,
                        Some(GpuFallbackReason::Integrity),
                    )
                }
                Err(error) => {
                    let bytes = inflate_raw_cpu(input, expected_output_size)?;
                    (
                        bytes,
                        DecodePath::CpuFallback,
                        Some(GpuFallbackReason::DecodeRejected(error)),
                    )
                }
            };
        Ok(DecodedGDeflate {
            bytes,
            report: DecodeReport {
                path,
                gpu_attempted: true,
                compressed_bytes: input.len(),
                uncompressed_bytes: expected_output_size,
                fallback,
            },
        })
    }

    fn decode_parsed(
        &self,
        ctx: &GpuContext,
        compressed_bytes: usize,
        parsed: &ParsedStream<'_>,
    ) -> Result<DecodedGDeflate, DecoderError> {
        // The current device status does not expose consumed input. Validate
        // it with the shared count-only parser; never CPU-inflate GPU output.
        validate_raw_tiles(parsed)?;
        let (bytes, path, fallback) = match self.decode_gpu(ctx, parsed) {
            Ok(bytes) => (bytes, DecodePath::GpuDeflate, None),
            Err(GpuAttemptError::AllocationFailed) => return Err(DecoderError::AllocationFailed),
            Err(GpuAttemptError::Fallback(reason)) => {
                let bytes = decode_cpu_tiles(parsed)?;
                (bytes, DecodePath::CpuFallback, Some(reason))
            }
        };
        Ok(DecodedGDeflate {
            report: DecodeReport {
                path,
                gpu_attempted: true,
                compressed_bytes,
                uncompressed_bytes: bytes.len(),
                fallback,
            },
            bytes,
        })
    }

    fn decode_gpu(
        &self,
        ctx: &GpuContext,
        parsed: &ParsedStream,
    ) -> Result<Vec<u8>, GpuAttemptError> {
        // Never reserve the entire advertised output before validating a tile.
        let mut output = Vec::new();
        for (descriptor, payload) in &parsed.tiles {
            let tile = self
                .deflate
                .decompress(
                    ctx,
                    payload,
                    descriptor.compressed_size as usize,
                    descriptor.uncompressed_size as usize,
                )
                .map_err(|error| {
                    GpuAttemptError::Fallback(GpuFallbackReason::DecodeRejected(error))
                })?;
            if tile.len() != descriptor.uncompressed_size as usize
                || checksum(&tile) != descriptor.checksum
            {
                return Err(GpuAttemptError::Fallback(GpuFallbackReason::Integrity));
            }
            output
                .try_reserve(tile.len())
                .map_err(|_| GpuAttemptError::AllocationFailed)?;
            output.extend_from_slice(&tile);
        }
        if output.len() != parsed.uncompressed_size {
            return Err(GpuAttemptError::Fallback(GpuFallbackReason::Integrity));
        }
        Ok(output)
    }
}

struct ParsedStream<'a> {
    uncompressed_size: usize,
    tiles: Vec<(GDeflateTileDescriptor, &'a [u8])>,
}

fn parse_stream(input: &[u8], max_output_bytes: usize) -> Result<ParsedStream<'_>, DecoderError> {
    let header = input
        .get(..HEADER_LEN)
        .ok_or(GDeflateError::TruncatedStream)?;
    if header.get(..4) != Some(b"GDF1") {
        return Err(GDeflateError::BadMagic.into());
    }
    if header[4] != 1 {
        return Err(GDeflateError::UnsupportedVersion(header[4]).into());
    }
    if usize::from(header[5]) != LANES {
        return Err(GDeflateError::UnsupportedLaneCount(header[5]).into());
    }
    if header[6] != TILE_SHIFT || header[7] != 0 {
        return Err(GDeflateError::CorruptTile.into());
    }
    let tile_count = usize::try_from(u32::from_le_bytes(header[8..12].try_into().unwrap()))
        .map_err(|_| GDeflateError::TruncatedStream)?;
    let uncompressed_size = usize::try_from(u64::from_le_bytes(header[12..20].try_into().unwrap()))
        .map_err(|_| GDeflateError::TileSizeMismatch)?;
    let descriptor_end = HEADER_LEN
        .checked_add(
            tile_count
                .checked_mul(DESCRIPTOR_LEN)
                .ok_or(GDeflateError::TruncatedStream)?,
        )
        .ok_or(GDeflateError::TruncatedStream)?;
    if descriptor_end > input.len() {
        return Err(GDeflateError::TruncatedStream.into());
    }
    let descriptors = &input[HEADER_LEN..descriptor_end];
    // Preflight every descriptor, payload range, and the total before reserving
    // the table. A malformed later tile must not trigger any decode allocation.
    let mut cursor = descriptor_end;
    let mut total = 0usize;
    for bytes in descriptors.chunks_exact(DESCRIPTOR_LEN) {
        let (descriptor, _) = parse_tile(input, bytes, &mut cursor)?;
        total = total
            .checked_add(descriptor.uncompressed_size as usize)
            .ok_or(GDeflateError::TileSizeMismatch)?;
    }
    if total != uncompressed_size {
        return Err(GDeflateError::TileSizeMismatch.into());
    }
    if cursor != input.len() {
        return Err(GDeflateError::TrailingData.into());
    }
    if uncompressed_size > max_output_bytes {
        return Err(GDeflateError::OutputLimitExceeded.into());
    }
    let mut tiles = Vec::new();
    tiles
        .try_reserve_exact(tile_count)
        .map_err(|_| DecoderError::AllocationFailed)?;
    cursor = descriptor_end;
    for bytes in descriptors.chunks_exact(DESCRIPTOR_LEN) {
        tiles.push(parse_tile(input, bytes, &mut cursor)?);
    }
    Ok(ParsedStream {
        uncompressed_size,
        tiles,
    })
}

fn parse_tile<'a>(
    input: &'a [u8],
    bytes: &[u8],
    cursor: &mut usize,
) -> Result<(GDeflateTileDescriptor, &'a [u8]), GDeflateError> {
    let bytes = bytes
        .get(..DESCRIPTOR_LEN)
        .ok_or(GDeflateError::TruncatedStream)?;
    let descriptor = GDeflateTileDescriptor {
        uncompressed_size: u32::from_le_bytes(bytes[0..4].try_into().unwrap()),
        compressed_size: u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
        checksum: u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
    };
    let compressed_size = descriptor.compressed_size as usize;
    if compressed_size == 0 || descriptor.uncompressed_size as usize > TILE_SIZE {
        return Err(GDeflateError::CorruptTile);
    }
    let padded = compressed_size
        .div_ceil(GROUP)
        .checked_mul(GROUP)
        .ok_or(GDeflateError::TruncatedStream)?;
    let end = cursor
        .checked_add(padded)
        .ok_or(GDeflateError::TruncatedStream)?;
    let payload = input
        .get(*cursor..end)
        .ok_or(GDeflateError::TruncatedStream)?;
    *cursor = end;
    Ok((descriptor, payload))
}

// The shared golden core enforces the declared tile size before every decoded
// write. Aggregate output grows only after exact size and checksum validation.
fn decode_cpu_tiles(parsed: &ParsedStream<'_>) -> Result<Vec<u8>, DecoderError> {
    let mut output = Vec::new();
    for (descriptor, payload) in &parsed.tiles {
        let compressed_size = descriptor.compressed_size as usize;
        let linear = linear_tile(payload, compressed_size)?;
        let (tile, consumed) =
            inflate_bounded_with_consumed(&linear, descriptor.uncompressed_size as usize)
                .map_err(GDeflateError::from)?;
        if consumed != compressed_size {
            return Err(GDeflateError::from(InflateError::TrailingData).into());
        }
        if tile.len() != descriptor.uncompressed_size as usize {
            return Err(GDeflateError::TileSizeMismatch.into());
        }
        if checksum(&tile) != descriptor.checksum {
            return Err(GDeflateError::TileChecksumMismatch.into());
        }
        output
            .try_reserve(tile.len())
            .map_err(|_| DecoderError::AllocationFailed)?;
        output.extend_from_slice(&tile);
    }
    Ok(output)
}

fn linear_tile(payload: &[u8], compressed_size: usize) -> Result<Vec<u8>, DecoderError> {
    let rounds = payload.len() / GROUP;
    let mut linear = Vec::new();
    linear
        .try_reserve_exact(compressed_size)
        .map_err(|_| DecoderError::AllocationFailed)?;
    for offset in 0..compressed_size {
        let word = offset / WORD;
        let stored_word = (word % LANES) * rounds + word / LANES;
        linear.push(payload[stored_word * WORD + offset % WORD]);
    }
    Ok(linear)
}

fn interleave_raw(input: &[u8]) -> Option<Vec<u8>> {
    let padded_len = input.len().div_ceil(GROUP).checked_mul(GROUP)?;
    let rounds = padded_len / GROUP;
    let mut out = Vec::new();
    out.try_reserve_exact(padded_len).ok()?;
    out.resize(padded_len, 0);
    for (logical, bytes) in input.chunks(WORD).enumerate() {
        let stored = (logical % LANES) * rounds + logical / LANES;
        out[stored * WORD..stored * WORD + bytes.len()].copy_from_slice(bytes);
    }
    Some(out)
}

fn inflate_raw_cpu(input: &[u8], expected_output_size: usize) -> Result<Vec<u8>, DecoderError> {
    let (bytes, consumed) =
        inflate_bounded_with_consumed(input, expected_output_size).map_err(GDeflateError::from)?;
    if consumed != input.len() {
        return Err(GDeflateError::Inflate(InflateError::TrailingData).into());
    }
    if bytes.len() != expected_output_size {
        return Err(GDeflateError::TileSizeMismatch.into());
    }
    Ok(bytes)
}

fn validate_raw_tiles(parsed: &ParsedStream<'_>) -> Result<(), DecoderError> {
    for (descriptor, payload) in &parsed.tiles {
        let linear = linear_tile(payload, descriptor.compressed_size as usize)?;
        let output_bytes = validate_deflate_bounded(&linear, descriptor.uncompressed_size as usize)
            .map_err(GDeflateError::from)?;
        if output_bytes != descriptor.uncompressed_size as usize {
            return Err(GDeflateError::TileSizeMismatch.into());
        }
    }
    Ok(())
}

fn checksum(data: &[u8]) -> u32 {
    let mut hash = 0x811c_9dc5u32;
    for &byte in data {
        hash ^= u32::from(byte);
        hash = hash.wrapping_mul(0x0100_0193);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_architecture::compression::{
        gdeflate_compress, gdeflate_decompress, gdeflate_decompress_bounded, InflateError,
    };

    fn frame_raw_tiles(tiles: &[(&[u8], &[u8])]) -> Vec<u8> {
        let mut input = b"GDF1\x01\x20\x10\0".to_vec();
        input.extend_from_slice(&(tiles.len() as u32).to_le_bytes());
        let total: u64 = tiles.iter().map(|(_, data)| data.len() as u64).sum();
        input.extend_from_slice(&total.to_le_bytes());
        for &(raw, data) in tiles {
            input.extend_from_slice(&(data.len() as u32).to_le_bytes());
            input.extend_from_slice(&(raw.len() as u32).to_le_bytes());
            input.extend_from_slice(&checksum(data).to_le_bytes());
        }
        for &(raw, _) in tiles {
            let mut payload = vec![0; raw.len().div_ceil(GROUP) * GROUP];
            let rounds = payload.len() / GROUP;
            for (offset, &byte) in raw.iter().enumerate() {
                let word = offset / WORD;
                payload[((word % LANES) * rounds + word / LANES) * WORD + offset % WORD] = byte;
            }
            input.extend_from_slice(&payload);
        }
        input
    }

    fn assert_cpu_entries(input: &[u8], expected: Result<&[u8], GDeflateError>) {
        let shared = gdeflate_decompress(input);
        assert_eq!(shared.as_deref().map_err(|&error| error), expected);
        for no_device in [false, true] {
            let decoded = if no_device {
                GDeflateDecoder::decode_auto(None, input)
            } else {
                GDeflateDecoder::decode_cpu(input)
            };
            match expected {
                Ok(bytes) => {
                    let decoded = decoded.unwrap();
                    assert_eq!(decoded.bytes, bytes);
                    assert_eq!(decoded.report.compressed_bytes, input.len());
                    assert_eq!(decoded.report.uncompressed_bytes, bytes.len());
                    assert!(!decoded.report.gpu_attempted);
                    assert_eq!(
                        decoded.report.path,
                        if no_device {
                            DecodePath::CpuFallback
                        } else {
                            DecodePath::Cpu
                        }
                    );
                    assert_eq!(
                        decoded.report.fallback,
                        no_device.then_some(GpuFallbackReason::NoDevice)
                    );
                }
                Err(error) => assert_eq!(decoded, Err(error.into())),
            }
        }
    }

    fn assert_bounded_cpu_entries(
        input: &[u8],
        limit: usize,
        expected: Result<&[u8], GDeflateError>,
    ) {
        assert_eq!(
            gdeflate_decompress_bounded(input, limit)
                .as_deref()
                .map_err(|&error| error),
            expected
        );
        for no_device in [false, true] {
            let result = if no_device {
                GDeflateDecoder::decode_auto_bounded(None, input, limit)
            } else {
                GDeflateDecoder::decode_cpu_bounded(input, limit)
            };
            match expected {
                Ok(bytes) => {
                    let decoded = result.unwrap();
                    assert_eq!(decoded.bytes, bytes);
                    assert_eq!(decoded.report.uncompressed_bytes, bytes.len());
                    assert_eq!(decoded.report.compressed_bytes, input.len());
                    assert!(!decoded.report.gpu_attempted);
                    assert_eq!(
                        decoded.report.path,
                        if no_device {
                            DecodePath::CpuFallback
                        } else {
                            DecodePath::Cpu
                        }
                    );
                    assert_eq!(
                        decoded.report.fallback,
                        no_device.then_some(GpuFallbackReason::NoDevice)
                    );
                }
                Err(error) => assert_eq!(result, Err(error.into())),
            }
        }
    }

    fn set_single_tile_size(input: &mut [u8], size: u32) {
        input[12..20].copy_from_slice(&u64::from(size).to_le_bytes());
        input[HEADER_LEN..HEADER_LEN + 4].copy_from_slice(&size.to_le_bytes());
    }

    fn raw_block_fixtures() -> Vec<(Vec<u8>, Vec<u8>)> {
        // Independent fixed literal A and overlapping length-258 copy.
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
        let hex = "edcab10d83301000c0559e0932062d4d0608f0912c2516022301d3932dd25c73d5f5c333e6b3bebe658ab6666ef18852e75cf2476d717dca18ef72b47dcd2e7adbb66ddbb66ddbb66ddbb66ddbb66ddbb66ddbb66ddbb66ddbb6ffb86f";
        let dynamic: Vec<u8> = (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
            .collect();
        let dynamic_data = b"GPU dynamic trees / independent zlib fixture! ".repeat(200);
        assert_eq!((dynamic[0] >> 1) & 3, 2);
        // Three block kinds in one tile, including a misaligned stored header.
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

    fn raw_stored_fixture(size: usize) -> (Vec<u8>, Vec<u8>) {
        let data: Vec<u8> = (0..size).map(|index| (index % 251) as u8).collect();
        let mut raw = Vec::new();
        for (index, block) in data.chunks(u16::MAX as usize).enumerate() {
            let final_block = (index + 1) * u16::MAX as usize >= data.len();
            let length = block.len() as u16;
            raw.push(u8::from(final_block));
            raw.extend_from_slice(&length.to_le_bytes());
            raw.extend_from_slice(&(!length).to_le_bytes());
            raw.extend_from_slice(block);
        }
        if data.is_empty() {
            raw.extend_from_slice(&[1, 0, 0, 0xFF, 0xFF]);
        }
        (raw, data)
    }

    #[test]
    fn raw_deflate_interleave_preserves_bytes_and_zero_padding() {
        for size in [
            0,
            1,
            WORD - 1,
            WORD,
            GROUP - 1,
            GROUP,
            GROUP + 1,
            GROUP * 3 + 1,
        ] {
            let input: Vec<u8> = (0..size).map(|index| (index % 251) as u8).collect();
            let interleaved = interleave_raw(&input).unwrap();
            assert_eq!(interleaved.len(), size.div_ceil(GROUP) * GROUP);
            let mut padded = input.clone();
            padded.resize(interleaved.len(), 0);
            assert_eq!(linear_tile(&interleaved, padded.len()).unwrap(), padded);
            // The existing framing helper implements the transpose independently.
            let framed = frame_raw_tiles(&[(&input, b"")]);
            assert_eq!(interleaved, framed[HEADER_LEN + DESCRIPTOR_LEN..]);
        }
    }

    #[test]
    fn raw_deflate_cpu_retry_distinguishes_size_mismatch_from_trailing_bytes() {
        let (raw, data) = raw_stored_fixture(3);
        assert_eq!(inflate_raw_cpu(&raw, data.len()).unwrap(), data);
        assert_eq!(
            inflate_raw_cpu(&raw, data.len() + 1),
            Err(GDeflateError::TileSizeMismatch.into())
        );
        assert_eq!(
            inflate_raw_cpu(&raw, data.len() - 1),
            Err(GDeflateError::Inflate(InflateError::OutputLimitExceeded).into())
        );
        let mut trailing = raw;
        trailing.push(0);
        assert_eq!(
            inflate_raw_cpu(&trailing, data.len()),
            Err(GDeflateError::Inflate(InflateError::TrailingData).into())
        );
    }

    #[test]
    fn raw_deflate_gpu_matches_stored_fixed_dynamic_and_mixed_fixtures() {
        let Some(ctx) = GpuContext::try_headless() else {
            return;
        };
        let decoder = GDeflateDecoder::new(&ctx);
        for (kind, (raw, data)) in ["stored", "fixed", "dynamic", "mixed"]
            .into_iter()
            .zip(raw_block_fixtures())
        {
            let (golden, consumed) = inflate_bounded_with_consumed(&raw, data.len()).unwrap();
            assert_eq!(golden, data, "{kind}");
            assert_eq!(consumed, raw.len(), "{kind}");
            for limit in [data.len(), usize::MAX] {
                // Pass raw bytes directly: no frame_raw_tiles / GDF1 decode.
                let decoded = decoder
                    .decode_raw_deflate_bounded(&ctx, &raw, data.len(), limit)
                    .unwrap();
                assert_eq!(decoded.bytes, golden, "{kind}");
                assert_eq!(
                    decoded.report,
                    DecodeReport {
                        path: DecodePath::GpuDeflate,
                        gpu_attempted: true,
                        compressed_bytes: raw.len(),
                        uncompressed_bytes: data.len(),
                        fallback: None,
                    },
                    "{kind}"
                );
            }
        }
    }

    #[test]
    fn raw_deflate_gpu_rejects_output_limits_and_expected_length_mismatches() {
        let Some(ctx) = GpuContext::try_headless() else {
            return;
        };
        let decoder = GDeflateDecoder::new(&ctx);
        for (raw, data) in raw_block_fixtures() {
            for limit in [0, data.len() - 1] {
                assert_eq!(
                    decoder.decode_raw_deflate_bounded(&ctx, &raw, data.len(), limit),
                    Err(GDeflateError::OutputLimitExceeded.into())
                );
            }
            // A generous caller cap must not weaken the exact expected size.
            for expected in [0, data.len() - 1] {
                assert_eq!(
                    decoder.decode_raw_deflate_bounded(&ctx, &raw, expected, usize::MAX),
                    Err(GDeflateError::Inflate(InflateError::OutputLimitExceeded).into())
                );
            }
            for expected in [data.len() + 1, usize::MAX] {
                assert_eq!(
                    decoder.decode_raw_deflate_bounded(&ctx, &raw, expected, expected),
                    Err(GDeflateError::TileSizeMismatch.into())
                );
            }
        }
        // Admission wins over malformed input and impossible declared lengths.
        for raw in [&[][..], &[7][..]] {
            assert_eq!(
                decoder.decode_raw_deflate_bounded(&ctx, raw, usize::MAX, 0),
                Err(GDeflateError::OutputLimitExceeded.into())
            );
        }
    }

    #[test]
    fn raw_deflate_gpu_rejects_trailing_bytes_truncation_and_invalid_streams() {
        let Some(ctx) = GpuContext::try_headless() else {
            return;
        };
        let decoder = GDeflateDecoder::new(&ctx);
        for (raw, data) in raw_block_fixtures() {
            for suffix in [vec![0], vec![0xFF], vec![3, 0], vec![0; GROUP], raw.clone()] {
                let mut trailing = raw.clone();
                trailing.extend_from_slice(&suffix);
                assert_eq!(
                    decoder.decode_raw_deflate_bounded(&ctx, &trailing, data.len(), usize::MAX),
                    Err(GDeflateError::Inflate(InflateError::TrailingData).into())
                );
            }
            for length in 0..raw.len() {
                let truncated = &raw[..length];
                let error = inflate_bounded_with_consumed(truncated, data.len()).unwrap_err();
                assert_eq!(
                    decoder.decode_raw_deflate_bounded(&ctx, truncated, data.len(), usize::MAX),
                    Err(GDeflateError::from(error).into()),
                    "truncated prefix {length}/{}",
                    raw.len()
                );
            }
        }
        for (raw, expected, error) in [
            (&[][..], 0, InflateError::UnexpectedEof),
            (&[7][..], 0, InflateError::ReservedBlockType),
            (&[1, 1, 0, 0, 0][..], 1, InflateError::StoredLengthMismatch),
            (&[3, 2, 0][..], 3, InflateError::DistanceTooFar),
        ] {
            assert_eq!(
                decoder.decode_raw_deflate_bounded(&ctx, raw, expected, expected),
                Err(GDeflateError::Inflate(error).into())
            );
        }
    }

    #[test]
    fn raw_deflate_gpu_accepts_empty_streams_and_final_byte_padding() {
        let Some(ctx) = GpuContext::try_headless() else {
            return;
        };
        let decoder = GDeflateDecoder::new(&ctx);
        // Empty output still requires a complete raw stream. RFC 1951 permits
        // arbitrary ignored alignment bits and unused bits in the final byte.
        for raw in [
            &[3, 0][..],
            &[3, 0xFC][..],
            &[1, 0, 0, 0xFF, 0xFF][..],
            &[0xF9, 0, 0, 0xFF, 0xFF][..],
            &[0, 0, 0, 0xFF, 0xFF, 3, 0xFC][..],
        ] {
            let decoded = decoder.decode_raw_deflate_bounded(&ctx, raw, 0, 0).unwrap();
            assert!(decoded.bytes.is_empty());
            assert_eq!(
                decoded.report,
                DecodeReport {
                    path: DecodePath::GpuDeflate,
                    gpu_attempted: true,
                    compressed_bytes: raw.len(),
                    uncompressed_bytes: 0,
                    fallback: None,
                }
            );
            assert_eq!(
                decoder.decode_raw_deflate_bounded(&ctx, raw, 1, 1),
                Err(GDeflateError::TileSizeMismatch.into())
            );
            let mut trailing = raw.to_vec();
            trailing.push(0);
            assert_eq!(
                decoder.decode_raw_deflate_bounded(&ctx, &trailing, 0, 0),
                Err(GDeflateError::Inflate(InflateError::TrailingData).into())
            );
            for length in 0..raw.len() {
                assert!(decoder
                    .decode_raw_deflate_bounded(&ctx, &raw[..length], 0, 0)
                    .is_err());
            }
        }
    }

    #[test]
    fn raw_deflate_gpu_falls_back_with_exact_reason_above_kernel_output_limit() {
        let Some(ctx) = GpuContext::try_headless() else {
            return;
        };
        let decoder = GDeflateDecoder::new(&ctx);
        // This is the actual kernel limit, independent of the adapter. The
        // first stream fits exactly despite its compressed size exceeding 64 KiB.
        // Larger valid raw streams are rejected by GpuDeflate as InvalidInput.
        for size in [TILE_SIZE, TILE_SIZE + 1, TILE_SIZE * 2 + 1] {
            let (raw, data) = raw_stored_fixture(size);
            assert!(raw.len() > TILE_SIZE);
            let (golden, consumed) = inflate_bounded_with_consumed(&raw, size).unwrap();
            assert_eq!(golden, data);
            assert_eq!(consumed, raw.len());
            for limit in [size, usize::MAX] {
                let decoded = decoder
                    .decode_raw_deflate_bounded(&ctx, &raw, size, limit)
                    .unwrap();
                assert_eq!(decoded.bytes, golden);
                let fallback = (size > TILE_SIZE).then_some(GpuFallbackReason::DecodeRejected(
                    DeflateDecodeError::InvalidInput,
                ));
                assert_eq!(
                    decoded.report,
                    DecodeReport {
                        path: if fallback.is_some() {
                            DecodePath::CpuFallback
                        } else {
                            DecodePath::GpuDeflate
                        },
                        gpu_attempted: true,
                        compressed_bytes: raw.len(),
                        uncompressed_bytes: size,
                        fallback,
                    }
                );
            }
            assert_eq!(
                decoder.decode_raw_deflate_bounded(&ctx, &raw, size, size - 1),
                Err(GDeflateError::OutputLimitExceeded.into())
            );
            assert_eq!(
                decoder.decode_raw_deflate_bounded(&ctx, &raw, size - 1, usize::MAX),
                Err(GDeflateError::Inflate(InflateError::OutputLimitExceeded).into())
            );
            assert_eq!(
                decoder.decode_raw_deflate_bounded(&ctx, &raw, size + 1, usize::MAX),
                Err(GDeflateError::TileSizeMismatch.into())
            );
            let mut trailing = raw;
            trailing.extend_from_slice(&[3, 0]);
            assert_eq!(
                decoder.decode_raw_deflate_bounded(&ctx, &trailing, size, size),
                Err(GDeflateError::Inflate(InflateError::TrailingData).into())
            );
        }
    }

    #[test]
    fn cpu_decode_reports_the_cpu_path_and_exact_sizes() {
        let input = gdeflate_compress(b"decoder report");
        let decoded = GDeflateDecoder::decode_cpu(&input).unwrap();
        assert_eq!(decoded.bytes, b"decoder report");
        assert_eq!(decoded.report.path, DecodePath::Cpu);
        assert!(!decoded.report.gpu_attempted);
        assert_eq!(decoded.report.compressed_bytes, input.len());
        assert_eq!(decoded.report.uncompressed_bytes, 14);
        assert_eq!(decoded.report.fallback, None);
    }

    #[test]
    fn optional_gpu_context_reports_no_device_fallback() {
        let input = gdeflate_compress(b"no adapter");
        let decoded = GDeflateDecoder::decode_auto(None, &input).unwrap();
        assert_eq!(decoded.bytes, b"no adapter");
        assert_eq!(decoded.report.path, DecodePath::CpuFallback);
        assert!(!decoded.report.gpu_attempted);
        assert_eq!(decoded.report.fallback, Some(GpuFallbackReason::NoDevice));
    }

    #[test]
    fn parser_rejects_a_tile_size_larger_than_the_wire_tile_contract() {
        let mut input = gdeflate_compress(b"small");
        input[HEADER_LEN + 0..HEADER_LEN + 4].copy_from_slice(&((1u32 << 16) + 1).to_le_bytes());
        assert!(matches!(
            parse_stream(&input, usize::MAX),
            Err(DecoderError::Format(GDeflateError::CorruptTile))
        ));
    }

    #[test]
    fn every_cpu_entry_rejects_forged_totals_and_tile_tables_before_allocation() {
        let encoded = gdeflate_compress(b"bounded framing");
        for invalid_total in [u64::MAX, 1 << 40, 0] {
            let mut input = encoded.clone();
            input[12..20].copy_from_slice(&invalid_total.to_le_bytes());
            assert_eq!(
                GDeflateDecoder::decode_cpu(&input),
                Err(DecoderError::Format(GDeflateError::TileSizeMismatch))
            );
            assert_eq!(
                GDeflateDecoder::decode_auto(None, &input),
                Err(DecoderError::Format(GDeflateError::TileSizeMismatch))
            );
        }
        let mut input = encoded;
        input[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(
            GDeflateDecoder::decode_cpu(&input),
            Err(DecoderError::Format(GDeflateError::TruncatedStream))
        );
    }

    #[test]
    fn cpu_entry_preflights_later_tiles_before_inflating_the_first() {
        let mut input = gdeflate_compress(&vec![b'a'; 70_000]);
        input.pop();
        assert_eq!(
            GDeflateDecoder::decode_cpu(&input),
            Err(DecoderError::Format(GDeflateError::TruncatedStream))
        );
    }

    #[test]
    fn cpu_tiles_match_shared_golden_and_reject_checksum_corruption() {
        for size in [0, 1, 65_536, 90_000] {
            let bytes: Vec<u8> = (0..size).map(|index| (index % 251) as u8).collect();
            let mut encoded = gdeflate_compress(&bytes);
            let decoded = GDeflateDecoder::decode_cpu(&encoded).unwrap();
            assert_eq!(decoded.bytes, bytes);
            assert_eq!(
                decoded.bytes,
                prism_render_architecture::compression::gdeflate_decompress(&encoded).unwrap()
            );
            if size > 0 {
                encoded[HEADER_LEN + 8] ^= 1;
                assert_eq!(
                    GDeflateDecoder::decode_cpu(&encoded),
                    Err(DecoderError::Format(GDeflateError::TileChecksumMismatch))
                );
            }
        }
    }

    #[test]
    fn cpu_entries_match_shared_framing_errors_for_all_header_fields() {
        for (offset, value, error) in [
            (0, 0, GDeflateError::BadMagic),
            (4, 2, GDeflateError::UnsupportedVersion(2)),
            (5, 0, GDeflateError::UnsupportedLaneCount(0)),
            (6, 15, GDeflateError::CorruptTile),
            (6, 17, GDeflateError::CorruptTile),
            (7, 1, GDeflateError::CorruptTile),
        ] {
            let mut input = gdeflate_compress(&[]);
            input[offset] = value;
            assert_cpu_entries(&input, Err(error));
        }
        let mut input = gdeflate_compress(&[]);
        input[8..12].copy_from_slice(&1u32.to_le_bytes());
        assert_cpu_entries(&input, Err(GDeflateError::TruncatedStream));
        let encoded = gdeflate_compress(b"abc");
        for (size, error) in [
            (0, GDeflateError::CorruptTile),
            (u32::MAX, GDeflateError::TruncatedStream),
        ] {
            let mut input = encoded.clone();
            input[HEADER_LEN + 4..HEADER_LEN + 8].copy_from_slice(&size.to_le_bytes());
            assert_cpu_entries(&input, Err(error));
        }
        let mut input = encoded;
        set_single_tile_size(&mut input, TILE_SIZE as u32 + 1);
        assert_cpu_entries(&input, Err(GDeflateError::CorruptTile));
    }

    #[test]
    fn cpu_entries_reject_every_truncated_prefix_and_outer_trailing_bytes() {
        let input = frame_raw_tiles(&[(&[1, 3, 0, 0xFC, 0xFF, b'a', b'b', b'c'], b"abc")]);
        for length in 0..input.len() {
            assert_cpu_entries(&input[..length], Err(GDeflateError::TruncatedStream));
        }
        for encoded in [gdeflate_compress(&[]), input] {
            for extra in [1, GROUP] {
                let mut input = encoded.clone();
                input.resize(input.len() + extra, 0);
                assert_cpu_entries(&input, Err(GDeflateError::TrailingData));
            }
        }
    }

    #[test]
    fn cpu_entries_reject_bad_framing_before_decoding_a_corrupt_first_tile() {
        let mut input = frame_raw_tiles(&[(&[7], b""), (&[3, 0], b"")]);
        input.pop();
        assert_cpu_entries(&input, Err(GDeflateError::TruncatedStream));
        let mut input = frame_raw_tiles(&[(&[7], b"")]);
        input.push(0);
        assert_cpu_entries(&input, Err(GDeflateError::TrailingData));
    }

    #[test]
    fn cpu_entries_enforce_all_block_types_bounded_output_and_integrity() {
        for (raw, data) in raw_block_fixtures() {
            let encoded = frame_raw_tiles(&[(&raw, &data)]);
            assert_cpu_entries(&encoded, Ok(&data));
            for limit in [0, data.len() - 1] {
                let mut input = encoded.clone();
                set_single_tile_size(&mut input, limit as u32);
                assert_cpu_entries(
                    &input,
                    Err(GDeflateError::Inflate(InflateError::OutputLimitExceeded)),
                );
            }
            let mut input = encoded.clone();
            set_single_tile_size(&mut input, data.len() as u32 + 1);
            assert_cpu_entries(&input, Err(GDeflateError::TileSizeMismatch));
            let mut input = encoded;
            input[HEADER_LEN + 8] ^= 1;
            assert_cpu_entries(&input, Err(GDeflateError::TileChecksumMismatch));
        }
    }

    #[test]
    fn cpu_entries_accept_empty_tiles_and_reset_budgets_between_small_tiles() {
        let empty_stored = [1, 0, 0, 0xFF, 0xFF];
        let empty_fixed = [3, 0];
        let stored = [1, 3, 0, 0xFC, 0xFF, b'a', b'b', b'c'];
        assert_cpu_entries(&gdeflate_compress(&[]), Ok(b""));
        for raw in [&empty_stored[..], &empty_fixed[..]] {
            assert_cpu_entries(&frame_raw_tiles(&[(raw, b"")]), Ok(b""));
        }
        let input = frame_raw_tiles(&[
            (&empty_stored, b""),
            (&stored, b"abc"),
            (&empty_fixed, b""),
            (&stored, b"abc"),
        ]);
        assert_cpu_entries(&input, Ok(b"abcabc"));
        let encoded = frame_raw_tiles(&[(&stored, b"abc"), (&stored, b"abc")]);
        for (first, second) in [(2u32, 4u32), (3, 2)] {
            let mut input = encoded.clone();
            input[12..20].copy_from_slice(&u64::from(first + second).to_le_bytes());
            input[HEADER_LEN..HEADER_LEN + 4].copy_from_slice(&first.to_le_bytes());
            input[HEADER_LEN + DESCRIPTOR_LEN..HEADER_LEN + DESCRIPTOR_LEN + 4]
                .copy_from_slice(&second.to_le_bytes());
            assert_cpu_entries(
                &input,
                Err(GDeflateError::Inflate(InflateError::OutputLimitExceeded)),
            );
        }
    }

    #[test]
    fn cpu_entries_accept_full_stored_tile_even_if_encoded_size_exceeds_64k() {
        let data = vec![b'Z'; TILE_SIZE];
        let mut raw = vec![0, 0xFF, 0xFF, 0, 0];
        raw.extend_from_slice(&data[..TILE_SIZE - 1]);
        raw.extend_from_slice(&[1, 1, 0, 0xFE, 0xFF, b'Z']);
        assert!(raw.len() > TILE_SIZE);
        let mut input = frame_raw_tiles(&[(&raw, &data)]);
        assert_cpu_entries(&input, Ok(&data));
        set_single_tile_size(&mut input, TILE_SIZE as u32 - 1);
        assert_cpu_entries(
            &input,
            Err(GDeflateError::Inflate(InflateError::OutputLimitExceeded)),
        );
    }

    #[test]
    fn tile_parser_rejects_short_descriptors_and_checked_cursor_overflow() {
        let input = frame_raw_tiles(&[(&[3, 0], b"")]);
        let descriptor = &input[HEADER_LEN..HEADER_LEN + DESCRIPTOR_LEN];
        for length in 0..DESCRIPTOR_LEN {
            let mut cursor = HEADER_LEN + DESCRIPTOR_LEN;
            assert_eq!(
                parse_tile(&input, &descriptor[..length], &mut cursor),
                Err(GDeflateError::TruncatedStream)
            );
            assert_eq!(cursor, HEADER_LEN + DESCRIPTOR_LEN);
        }
        for start in [usize::MAX - 63, input.len() - GROUP + 1] {
            let mut cursor = start;
            assert_eq!(
                parse_tile(&input, descriptor, &mut cursor),
                Err(GDeflateError::TruncatedStream)
            );
            assert_eq!(cursor, start);
        }
    }

    #[test]
    fn allocation_errors_are_not_reported_as_invalid_format() {
        assert_eq!(
            DecoderError::from(GDeflateError::AllocationFailed),
            DecoderError::AllocationFailed
        );
        assert_eq!(
            DecoderError::from(GDeflateError::from(InflateError::AllocationFailed)),
            DecoderError::AllocationFailed
        );
        assert_eq!(
            DecoderError::from(GDeflateError::from(InflateError::OutputLimitExceeded)),
            DecoderError::Format(GDeflateError::Inflate(InflateError::OutputLimitExceeded))
        );
        assert_eq!(
            DecoderError::from(GDeflateError::OutputLimitExceeded),
            DecoderError::Format(GDeflateError::OutputLimitExceeded)
        );
    }

    #[test]
    fn every_entry_rejects_inner_trailing_for_all_block_kinds_and_later_tiles() {
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
                    let error = GDeflateError::Inflate(InflateError::TrailingData);
                    assert_cpu_entries(&input, Err(error));
                    assert_bounded_cpu_entries(&input, data.len(), Err(error));
                    let parsed = parse_stream(&input, data.len()).unwrap();
                    assert_eq!(validate_raw_tiles(&parsed), Err(error.into()));
                }
            }
        }
    }

    #[test]
    fn bounded_entries_accept_exact_and_empty_but_reject_cumulative_multitile_output() {
        let fixtures = raw_block_fixtures();
        let mut tiles = vec![(&[3, 0xFC][..], &b""[..])];
        tiles.extend(
            fixtures
                .iter()
                .map(|(raw, data)| (raw.as_slice(), data.as_slice())),
        );
        tiles.push((&[0xF9, 0, 0, 0xFF, 0xFF], b""));
        let expected: Vec<u8> = fixtures
            .iter()
            .flat_map(|(_, data)| data.iter().copied())
            .collect();
        let input = frame_raw_tiles(&tiles);
        assert_bounded_cpu_entries(&input, expected.len(), Ok(&expected));
        assert_bounded_cpu_entries(&input, usize::MAX, Ok(&expected));
        assert_eq!(
            validate_raw_tiles(&parse_stream(&input, expected.len()).unwrap()),
            Ok(())
        );
        for limit in [0, expected.len() - 1] {
            assert_bounded_cpu_entries(&input, limit, Err(GDeflateError::OutputLimitExceeded));
        }
        for input in [
            gdeflate_compress(&[]),
            frame_raw_tiles(&[(&[3, 0xFC], b""), (&[1, 0, 0, 0xFF, 0xFF], b"")]),
        ] {
            assert_bounded_cpu_entries(&input, 0, Ok(b""));
            assert_eq!(
                validate_raw_tiles(&parse_stream(&input, 0).unwrap()),
                Ok(())
            );
        }
        let data = vec![b'A'; TILE_SIZE * 2 + 1];
        let input = gdeflate_compress(&data);
        assert_bounded_cpu_entries(&input, data.len(), Ok(&data));
        assert_bounded_cpu_entries(
            &input,
            data.len() - 1,
            Err(GDeflateError::OutputLimitExceeded),
        );
    }

    #[test]
    fn bounded_preflight_rejects_late_bad_framing_or_budget_before_corrupt_raw_decode() {
        let stored = [1, 3, 0, 0xFC, 0xFF, b'a', b'b', b'c'];
        let input = frame_raw_tiles(&[(&[7], b"abc"), (&stored, b"abc")]);
        assert_bounded_cpu_entries(&input, 5, Err(GDeflateError::OutputLimitExceeded));
        assert_bounded_cpu_entries(
            &input,
            6,
            Err(GDeflateError::Inflate(InflateError::ReservedBlockType)),
        );
        assert!(matches!(
            parse_stream(&input, 5),
            Err(DecoderError::Format(GDeflateError::OutputLimitExceeded))
        ));
        let mut truncated = input.clone();
        truncated.pop();
        assert_bounded_cpu_entries(&truncated, 5, Err(GDeflateError::TruncatedStream));
        let mut forged = input.clone();
        forged[12..20].copy_from_slice(&3u64.to_le_bytes());
        assert_bounded_cpu_entries(&forged, 3, Err(GDeflateError::TileSizeMismatch));
        let mut oversized = input;
        oversized[HEADER_LEN + DESCRIPTOR_LEN..HEADER_LEN + DESCRIPTOR_LEN + 4]
            .copy_from_slice(&(TILE_SIZE as u32 + 1).to_le_bytes());
        assert_bounded_cpu_entries(&oversized, usize::MAX, Err(GDeflateError::CorruptTile));
    }

    #[test]
    fn aggregate_budgets_neither_transfer_tile_capacity_nor_lz77_history() {
        let raw = [1, 3, 0, 0xFC, 0xFF, b'a', b'b', b'c'];
        let mut input = frame_raw_tiles(&[(&raw, b"abc"), (&raw, b"abc")]);
        input[HEADER_LEN..HEADER_LEN + 4].copy_from_slice(&2u32.to_le_bytes());
        input[HEADER_LEN + DESCRIPTOR_LEN..HEADER_LEN + DESCRIPTOR_LEN + 4]
            .copy_from_slice(&4u32.to_le_bytes());
        assert_bounded_cpu_entries(
            &input,
            6,
            Err(GDeflateError::Inflate(InflateError::OutputLimitExceeded)),
        );
        assert_bounded_cpu_entries(&input, 5, Err(GDeflateError::OutputLimitExceeded));
        assert_eq!(
            validate_raw_tiles(&parse_stream(&input, 6).unwrap()),
            Err(DecoderError::Format(GDeflateError::Inflate(
                InflateError::OutputLimitExceeded
            )))
        );
        let input = frame_raw_tiles(&[(&[1, 1, 0, 0xFE, 0xFF, b'A'], b"A"), (&[3, 2, 0], b"AAA")]);
        let error = GDeflateError::Inflate(InflateError::DistanceTooFar);
        assert_bounded_cpu_entries(&input, 4, Err(error));
        assert_eq!(
            validate_raw_tiles(&parse_stream(&input, 4).unwrap()),
            Err(error.into())
        );
    }

    #[test]
    fn gpu_bounded_entries_validate_consumption_and_admission_without_cpu_output_substitution() {
        let Some(ctx) = GpuContext::try_headless() else {
            return;
        };
        let decoder = GDeflateDecoder::new(&ctx);
        let mut fixtures = raw_block_fixtures();
        fixtures.extend([
            (vec![3, 0xFC], Vec::new()),
            (vec![0xF9, 0, 0, 0xFF, 0xFF], Vec::new()),
        ]);
        for (raw, data) in &fixtures {
            let input = frame_raw_tiles(&[(raw, data)]);
            let decoded = decoder.decode_bounded(&ctx, &input, data.len()).unwrap();
            assert_eq!(&decoded.bytes, data);
            assert_eq!(decoded.report.path, DecodePath::GpuDeflate);
            assert!(decoded.report.gpu_attempted);
            assert_eq!(decoded.report.fallback, None);
            for tail in [&[0][..], &[3, 0], raw.as_slice()] {
                let mut extended = raw.clone();
                extended.extend_from_slice(tail);
                let input = frame_raw_tiles(&[(&[3, 0], b""), (&extended, data)]);
                let expected = Err(DecoderError::Format(GDeflateError::Inflate(
                    InflateError::TrailingData,
                )));
                assert_eq!(decoder.decode_bounded(&ctx, &input, data.len()), expected);
                assert_eq!(
                    GDeflateDecoder::decode_auto_bounded(Some(&ctx), &input, data.len()),
                    expected
                );
            }
        }
        let tiles: Vec<_> = fixtures
            .iter()
            .map(|(raw, data)| (raw.as_slice(), data.as_slice()))
            .collect();
        let input = frame_raw_tiles(&tiles);
        let total: usize = fixtures.iter().map(|(_, data)| data.len()).sum();
        let decoded = decoder.decode_bounded(&ctx, &input, total).unwrap();
        assert_eq!(decoded.report.path, DecodePath::GpuDeflate);
        assert_eq!(
            decoded.bytes,
            gdeflate_decompress_bounded(&input, total).unwrap()
        );
        let auto = GDeflateDecoder::decode_auto_bounded(Some(&ctx), &input, total).unwrap();
        assert_eq!(auto, decoded);
        for limit in [0, total - 1] {
            let expected = Err(DecoderError::Format(GDeflateError::OutputLimitExceeded));
            assert_eq!(decoder.decode_bounded(&ctx, &input, limit), expected);
            assert_eq!(
                GDeflateDecoder::decode_auto_bounded(Some(&ctx), &input, limit),
                expected
            );
        }
        // A malformed raw stream must not run when the declared stream is
        // over budget, including through the optional-device entrypoint.
        let input = frame_raw_tiles(&[(&[7], b"abc"), (&[3, 0], b"")]);
        let expected = Err(DecoderError::Format(GDeflateError::OutputLimitExceeded));
        assert_eq!(decoder.decode_bounded(&ctx, &input, 2), expected);
        assert_eq!(
            GDeflateDecoder::decode_auto_bounded(Some(&ctx), &input, 2),
            expected
        );
    }

    #[test]
    fn gpu_decode_framed_raw_admission_and_integrity_retry_obey_the_same_tile_budgets() {
        let Some(ctx) = GpuContext::try_headless() else {
            return;
        };
        let decoder = GDeflateDecoder::new(&ctx);
        let fixtures = raw_block_fixtures();
        for (raw, data) in &fixtures {
            let encoded = frame_raw_tiles(&[(raw, data)]);
            let decoded = decoder.decode(&ctx, &encoded).unwrap();
            assert_eq!(&decoded.bytes, data);
            assert_eq!(decoded.report.path, DecodePath::GpuDeflate);
            assert!(decoded.report.gpu_attempted);
            assert_eq!(decoded.report.fallback, None);
            // Raw count-only validation rejects size violations before dispatch.
            for limit in [0, data.len() - 1] {
                let mut input = encoded.clone();
                set_single_tile_size(&mut input, limit as u32);
                assert_eq!(
                    decoder.decode(&ctx, &input),
                    Err(DecoderError::Format(GDeflateError::Inflate(
                        InflateError::OutputLimitExceeded
                    )))
                );
            }
            let mut input = encoded.clone();
            set_single_tile_size(&mut input, data.len() as u32 + 1);
            assert_eq!(
                decoder.decode(&ctx, &input),
                Err(DecoderError::Format(GDeflateError::TileSizeMismatch))
            );
            let mut input = encoded;
            input[HEADER_LEN + 8] ^= 1;
            // Raw framing/length is valid: GPU integrity rejects these bytes,
            // and the independently bounded CPU retry rejects the checksum too.
            assert_eq!(
                decoder.decode(&ctx, &input),
                Err(DecoderError::Format(GDeflateError::TileChecksumMismatch))
            );
        }
        let empty = [1, 0, 0, 0xFF, 0xFF];
        let mut tiles = vec![(&empty[..], &b""[..])];
        tiles.extend(
            fixtures
                .iter()
                .map(|(raw, data)| (raw.as_slice(), data.as_slice())),
        );
        tiles.push((&[3, 0], b""));
        let expected: Vec<u8> = fixtures
            .iter()
            .flat_map(|(_, data)| data.iter().copied())
            .collect();
        let input = frame_raw_tiles(&tiles);
        let decoded = decoder.decode(&ctx, &input).unwrap();
        assert_eq!(decoded.bytes, expected);
        assert_eq!(decoded.bytes, gdeflate_decompress(&input).unwrap());
        assert_eq!(decoded.report.path, DecodePath::GpuDeflate);
        assert_eq!(decoded.report.fallback, None);
    }
}
