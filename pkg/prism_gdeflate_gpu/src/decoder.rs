//! Unified `GDeflate` decode policy with an honest GPU/CPU execution report.
//!
//! Every block type is decoded completely on the GPU, including dynamic tree
//! construction. Framing and checksums remain host-side. GPU rejection or an
//! integrity failure retries the complete stream through the CPU golden path
//! and is reported as a fallback, never as successful GPU decode.

use alloc::vec::Vec;
use core::fmt;

use prism_render_architecture::compression::{
    inflate, GDeflateError, GDeflateTileDescriptor,
};

use crate::{DeflateDecodeError, GpuContext, GpuDeflate};

const HEADER_LEN: usize = 20;
const DESCRIPTOR_LEN: usize = 12;
const LANES: usize = 32;
const WORD: usize = 4;
const GROUP: usize = LANES * WORD;

/// The execution path used for one decoded stream.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecodePath {
    /// The complete stream was decoded by the CPU golden implementation.
    Cpu,
    /// All tile blocks, dynamic trees, and LZ77 history were decoded on GPU.
    /// Container parsing and checksum verification are still CPU-side.
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
        Self::Format(error)
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
        let parsed = parse_stream(input)?;
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
        let Some(ctx) = ctx else {
            let mut decoded = Self::decode_cpu(input)?;
            decoded.report.path = DecodePath::CpuFallback;
            decoded.report.fallback = Some(GpuFallbackReason::NoDevice);
            return Ok(decoded);
        };
        Self::new(ctx).decode(ctx, input)
    }

    /// Prefers complete GPU decode for every tile. A GPU error or checksum
    /// failure retries the entire stream on CPU and reports the exact reason.
    pub fn decode(&self, ctx: &GpuContext, input: &[u8]) -> Result<DecodedGDeflate, DecoderError> {
        let parsed = parse_stream(input)?;
        let (bytes, path, fallback) = match self.decode_gpu(ctx, &parsed) {
            Ok(bytes) => (bytes, DecodePath::GpuDeflate, None),
            Err(GpuAttemptError::AllocationFailed) => return Err(DecoderError::AllocationFailed),
            Err(GpuAttemptError::Fallback(reason)) => {
                let bytes = decode_cpu_tiles(&parsed)?;
                (bytes, DecodePath::CpuFallback, Some(reason))
            }
        };
        Ok(DecodedGDeflate {
            report: DecodeReport {
                path,
                gpu_attempted: true,
                compressed_bytes: input.len(),
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
            if checksum(&tile) != descriptor.checksum {
                return Err(GpuAttemptError::Fallback(GpuFallbackReason::Integrity));
            }
            output.try_reserve(tile.len()).map_err(|_| GpuAttemptError::AllocationFailed)?;
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

fn parse_stream(input: &[u8]) -> Result<ParsedStream<'_>, DecoderError> {
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
    let tile_count = u32::from_le_bytes(header[8..12].try_into().unwrap()) as usize;
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
    let mut tiles = Vec::new();
    tiles.try_reserve_exact(tile_count).map_err(|_| DecoderError::AllocationFailed)?;
    let mut cursor = descriptor_end;
    let mut total = 0usize;
    for bytes in input[HEADER_LEN..descriptor_end].chunks_exact(DESCRIPTOR_LEN) {
        let descriptor = GDeflateTileDescriptor {
            uncompressed_size: u32::from_le_bytes(bytes[0..4].try_into().unwrap()),
            compressed_size: u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
            checksum: u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
        };
        let compressed_size = descriptor.compressed_size as usize;
        let padded = compressed_size
            .div_ceil(GROUP)
            .checked_mul(GROUP)
            .ok_or(GDeflateError::TruncatedStream)?;
        let end = cursor
            .checked_add(padded)
            .ok_or(GDeflateError::TruncatedStream)?;
        let payload = input
            .get(cursor..end)
            .ok_or(GDeflateError::TruncatedStream)?;
        if compressed_size == 0 || descriptor.uncompressed_size as usize > (1 << 16) {
            return Err(GDeflateError::CorruptTile.into());
        }
        total = total
            .checked_add(descriptor.uncompressed_size as usize)
            .ok_or(GDeflateError::TileSizeMismatch)?;
        tiles.push((descriptor, payload));
        cursor = end;
    }
    if total != uncompressed_size {
        return Err(GDeflateError::TileSizeMismatch.into());
    }
    Ok(ParsedStream {
        uncompressed_size,
        tiles,
    })
}

// Use the shared CPU golden inflate core, but grow aggregate output only after
// each tile has passed size/checksum validation. The shared inflate API itself
// is still unbounded; this does not claim to fix its decompression-bomb risk.
fn decode_cpu_tiles(parsed: &ParsedStream<'_>) -> Result<Vec<u8>, DecoderError> {
    let mut output = Vec::new();
    for (descriptor, payload) in &parsed.tiles {
        let compressed_size = descriptor.compressed_size as usize;
        let rounds = payload.len() / GROUP;
        let mut linear = Vec::new();
        linear.try_reserve_exact(compressed_size).map_err(|_| DecoderError::AllocationFailed)?;
        for offset in 0..compressed_size {
            let word = offset / WORD;
            let stored_word = (word % LANES) * rounds + word / LANES;
            linear.push(payload[stored_word * WORD + offset % WORD]);
        }
        let tile = inflate(&linear).map_err(GDeflateError::Inflate)?;
        if tile.len() != descriptor.uncompressed_size as usize {
            return Err(GDeflateError::TileSizeMismatch.into());
        }
        if checksum(&tile) != descriptor.checksum {
            return Err(GDeflateError::TileChecksumMismatch.into());
        }
        output.try_reserve(tile.len()).map_err(|_| DecoderError::AllocationFailed)?;
        output.extend_from_slice(&tile);
    }
    Ok(output)
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
    use prism_render_architecture::compression::gdeflate_compress;

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
            parse_stream(&input),
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
}
