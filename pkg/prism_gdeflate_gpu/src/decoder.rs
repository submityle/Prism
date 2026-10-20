//! Unified `GDeflate` decode policy with an honest GPU/CPU execution report.
//!
//! Every block type is decoded completely on the GPU, including dynamic tree
//! construction. Framing and checksums remain host-side. GPU rejection or an
//! integrity failure retries the complete stream through the CPU golden path
//! and is reported as a fallback, never as successful GPU decode.

use alloc::vec::Vec;
use core::fmt;

use prism_render_architecture::compression::{
    gdeflate_decompress, GDeflateError, GDeflateTileDescriptor, InflateError,
};

use crate::{DeflateDecodeError, DeinterleaveError, GpuContext, GpuDeflate};

const HEADER_LEN: usize = 20;
const DESCRIPTOR_LEN: usize = 12;
const LANES: usize = 32;
const WORD: usize = 4;
const GROUP: usize = LANES * WORD;
const MAX_TILES: usize = 1_000_000;

/// The execution path used for one decoded stream.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecodePath {
    /// The complete stream was decoded by the CPU golden implementation.
    Cpu,
    /// All tile blocks, dynamic trees, and LZ77 history were decoded on GPU.
    /// Container parsing and checksum verification are still CPU-side.
    GpuDeflate,
    /// A fixed-Huffman tile stream was decoded entirely by the GPU, including
    /// lane-major word gather, Huffman symbols, and LZ77 back-references.
    /// Retained for report compatibility; the unified decoder now uses
    /// [`Self::GpuDeflate`] for every supported block type.
    GpuFixedHuffman,
    /// GPU de-interleave was used and CPU Huffman inflate completed the tiles.
    /// Legacy report value; no longer emitted by the unified decoder.
    GpuDeinterleaveCpuHuffman,
    /// GPU was attempted but the request fell back to the CPU golden path.
    CpuFallback,
}

/// Why a GPU attempt did not become the successful execution path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuFallbackReason {
    /// The caller did not provide a real device.
    NoDevice,
    /// The adapter rejected the request shape or resource size.
    /// Legacy de-interleave reason; new full-decode errors are reported via
    /// [`Self::DecodeRejected`].
    DeviceLimit(DeinterleaveError),
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
    /// The shared CPU inflate core rejected a tile.
    /// Legacy hybrid-path error; CPU golden retries now return [`Self::Format`].
    Inflate(InflateError),
}

impl fmt::Display for DecoderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Format(error) => write!(f, "invalid GDeflate stream: {error:?}"),
            Self::Inflate(error) => write!(f, "GDeflate tile inflate failed: {error:?}"),
        }
    }
}

impl std::error::Error for DecoderError {}

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
        let bytes = gdeflate_decompress(input).map_err(DecoderError::Format)?;
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
        let parsed = parse_stream(input).map_err(DecoderError::Format)?;
        let (bytes, path, fallback) = match self.decode_gpu(ctx, &parsed) {
            Ok(bytes) => (bytes, DecodePath::GpuDeflate, None),
            Err(reason) => {
                let decoded = Self::decode_cpu(input)?;
                (decoded.bytes, DecodePath::CpuFallback, Some(reason))
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
    ) -> Result<Vec<u8>, GpuFallbackReason> {
        let mut output = Vec::with_capacity(parsed.uncompressed_size as usize);
        for (descriptor, payload) in &parsed.tiles {
            let tile = self
                .deflate
                .decompress(
                    ctx,
                    payload,
                    descriptor.compressed_size as usize,
                    descriptor.uncompressed_size as usize,
                )
                .map_err(GpuFallbackReason::DecodeRejected)?;
            if checksum(&tile) != descriptor.checksum {
                return Err(GpuFallbackReason::Integrity);
            }
            output.extend_from_slice(&tile);
        }
        if output.len() != parsed.uncompressed_size as usize {
            return Err(GpuFallbackReason::Integrity);
        }
        Ok(output)
    }
}

struct ParsedStream<'a> {
    uncompressed_size: u64,
    tiles: Vec<(GDeflateTileDescriptor, &'a [u8])>,
}

fn parse_stream(input: &[u8]) -> Result<ParsedStream<'_>, GDeflateError> {
    let header = input
        .get(..HEADER_LEN)
        .ok_or(GDeflateError::TruncatedStream)?;
    if header.get(..4) != Some(b"GDF1") {
        return Err(GDeflateError::BadMagic);
    }
    if header[4] != 1 {
        return Err(GDeflateError::UnsupportedVersion(header[4]));
    }
    if usize::from(header[5]) != LANES {
        return Err(GDeflateError::UnsupportedLaneCount(header[5]));
    }
    let tile_count = u32::from_le_bytes(header[8..12].try_into().unwrap()) as usize;
    if tile_count > MAX_TILES {
        return Err(GDeflateError::TruncatedStream);
    }
    let uncompressed_size = u64::from_le_bytes(header[12..20].try_into().unwrap());
    let descriptor_end = HEADER_LEN
        .checked_add(
            tile_count
                .checked_mul(DESCRIPTOR_LEN)
                .ok_or(GDeflateError::TruncatedStream)?,
        )
        .ok_or(GDeflateError::TruncatedStream)?;
    if descriptor_end > input.len() {
        return Err(GDeflateError::TruncatedStream);
    }
    let mut descriptors = Vec::with_capacity(tile_count);
    let mut cursor = HEADER_LEN;
    for _ in 0..tile_count {
        let bytes = input
            .get(cursor..cursor + DESCRIPTOR_LEN)
            .ok_or(GDeflateError::TruncatedStream)?;
        descriptors.push(GDeflateTileDescriptor {
            uncompressed_size: u32::from_le_bytes(bytes[0..4].try_into().unwrap()),
            compressed_size: u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
            checksum: u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
        });
        cursor += DESCRIPTOR_LEN;
    }

    let mut tiles = Vec::with_capacity(tile_count);
    let mut total = 0u64;
    for descriptor in descriptors {
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
            return Err(GDeflateError::CorruptTile);
        }
        total = total
            .checked_add(u64::from(descriptor.uncompressed_size))
            .ok_or(GDeflateError::TileSizeMismatch)?;
        tiles.push((descriptor, payload));
        cursor = end;
    }
    if total != uncompressed_size {
        return Err(GDeflateError::TileSizeMismatch);
    }
    Ok(ParsedStream {
        uncompressed_size,
        tiles,
    })
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
            Err(GDeflateError::CorruptTile)
        ));
    }
}
