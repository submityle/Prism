//! Canonical, bounded sparse-SDF cell frames for the existing `SceneHost` path.

use alloc::vec::Vec;
use core::fmt;

use prism_world_physics_bridge::{
    SdfBridgeError, WorldSdfSnapshot, WorldSdfTransform, WORLD_SDF_BRICK_METADATA_BYTES,
    WORLD_SDF_HEADER_BYTES, WORLD_SDF_SAMPLE_BYTES,
};
use prism_world_residency::{BrickCoord, SdfError, SparseSdfBrickGrid};
use prism_world_stream::{CellManifest, SectorId};

use crate::{SdfCellDecoder, SdfCellSnapshot};

/// Stable `PSDC` sparse-SDF cell format revision.
pub const SDF_CELL_CODEC_VERSION: u16 = 1;
const MAGIC: [u8; 4] = *b"PSDC";
const HEADER_BYTES: usize = 128;
const RECORD_BYTES: usize = 16;
const CHECKSUM_BYTES: usize = 4;

/// Independent bounds checked before allocating decoded brick samples.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SdfCellCodecLimits {
    /// Complete frame bytes, including header and checksum.
    pub max_payload_bytes: usize,
    /// Allocated bricks in one decoded cell.
    pub max_bricks: usize,
    /// Aggregate quantized samples in one decoded cell.
    pub max_samples: usize,
    /// Logical snapshot bytes: 48 + 16 per brick + 2 per sample.
    ///
    /// This excludes allocator overhead, `BTreeMap` nodes, the input frame and
    /// encoder coordinate scratch. It is not a process-memory or VRAM bound.
    pub max_decoded_bytes: usize,
}

impl SdfCellCodecLimits {
    /// Creates explicit bounds without selecting a product residency policy.
    #[must_use]
    pub const fn new(
        max_payload_bytes: usize,
        max_bricks: usize,
        max_samples: usize,
        max_decoded_bytes: usize,
    ) -> Self {
        Self {
            max_payload_bytes,
            max_bricks,
            max_samples,
            max_decoded_bytes,
        }
    }
}

/// Resource rejected by a codec admission limit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SdfCellCodecResource {
    /// Complete encoded frame bytes.
    PayloadBytes,
    /// Allocated brick count.
    Bricks,
    /// Total quantized sample count.
    Samples,
    /// Retained logical snapshot bytes.
    DecodedBytes,
}

/// Structural, identity or admission failure in a sparse-SDF cell frame.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SdfCellCodecError {
    /// Checked frame size arithmetic overflowed.
    SizeOverflow,
    /// A fallible sample or output vector reservation failed.
    AllocationFailed,
    /// A configured resource limit was exceeded.
    LimitExceeded {
        /// Resource being bounded.
        resource: SdfCellCodecResource,
        /// Required number of units.
        required: usize,
        /// Configured maximum units.
        limit: usize,
    },
    /// The input ended before its declared complete frame.
    Truncated,
    /// Data followed the declared frame and checksum.
    TrailingBytes,
    /// Input belongs to a different payload format.
    InvalidMagic,
    /// The codec does not implement the encoded version.
    UnsupportedVersion {
        /// Encoded version.
        found: u16,
    },
    /// A fixed header field or reserved byte is invalid.
    InvalidHeader {
        /// Name of the rejected field.
        field: &'static str,
    },
    /// Brick count and fixed record volume do not match the declared body.
    BodyLengthMismatch,
    /// CRC32 IEEE failed; CRC detects corruption, not malicious alteration.
    ChecksumMismatch,
    /// The frame sector differs from the authoritative manifest.
    ManifestSectorMismatch {
        /// Manifest sector.
        manifest: SectorId,
        /// Encoded sector.
        encoded: SectorId,
    },
    /// Duplicate or decreasing complete keys violate canonical ordering.
    NonCanonicalBrickOrder {
        /// Previous full key.
        previous: BrickCoord,
        /// Rejected full key.
        current: BrickCoord,
    },
    /// The grid rejected its dimensions or a brick mip.
    Grid(SdfError),
    /// The physical transform is invalid.
    Transform(SdfBridgeError),
    /// Encoding a dirty grid would silently discard pending invalidations.
    DirtyGrid,
}

impl fmt::Display for SdfCellCodecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SizeOverflow => f.write_str("SDF cell size overflow"),
            Self::AllocationFailed => f.write_str("SDF cell vector allocation failed"),
            Self::LimitExceeded {
                resource,
                required,
                limit,
            } => write!(
                f,
                "SDF cell {resource:?} requires {required}, limit is {limit}"
            ),
            Self::Truncated => f.write_str("truncated SDF cell frame"),
            Self::TrailingBytes => f.write_str("trailing SDF cell frame bytes"),
            Self::InvalidMagic => f.write_str("invalid SDF cell magic"),
            Self::UnsupportedVersion { found } => write!(f, "unsupported SDF cell version {found}"),
            Self::InvalidHeader { field } => write!(f, "invalid SDF cell {field}"),
            Self::BodyLengthMismatch => f.write_str("SDF cell body length mismatch"),
            Self::ChecksumMismatch => f.write_str("SDF cell checksum mismatch"),
            Self::ManifestSectorMismatch { manifest, encoded } => {
                write!(f, "SDF cell sector mismatch: {manifest:?} vs {encoded:?}")
            }
            Self::NonCanonicalBrickOrder { previous, current } => write!(
                f,
                "noncanonical SDF brick order: {previous:?} then {current:?}"
            ),
            Self::Grid(error) => write!(f, "invalid SDF cell grid: {error:?}"),
            Self::Transform(error) => write!(f, "invalid SDF cell transform: {error}"),
            Self::DirtyGrid => f.write_str("cannot encode an SDF grid with pending invalidations"),
        }
    }
}

impl core::error::Error for SdfCellCodecError {}

/// Canonical little-endian `PSDC` v1 codec and concrete [`SdfCellDecoder`].
///
/// Payloads are disposable derived caches. The sector is checked against the
/// manifest; the opaque `derived_hash` is not interpreted as a content digest.
/// Decode performs complete structural preflight before sample allocations.
/// Standard `BTreeMap` insertion and grid coordinate enumeration retain their
/// allocator's infallible allocation behavior; vector reservations are fallible.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SdfCellCodec {
    limits: SdfCellCodecLimits,
}

impl SdfCellCodec {
    /// Creates a codec with independent payload and decoded-resource bounds.
    #[must_use]
    pub const fn new(limits: SdfCellCodecLimits) -> Self {
        Self { limits }
    }

    /// Returns the configured limits.
    #[must_use]
    pub const fn limits(&self) -> SdfCellCodecLimits {
        self.limits
    }

    /// Encodes clean allocated bricks in full `(x,y,z,mip)` key order.
    ///
    /// Samples retain x-fastest order and exact signed i16 values. Dirty grids
    /// must be rebuilt before encoding so their invalidations cannot be lost.
    pub fn encode(&self, cell: &SdfCellSnapshot) -> Result<Vec<u8>, SdfCellCodecError> {
        let snapshot = cell.snapshot();
        let grid = snapshot.grid();
        if grid.dirty_count() != 0 {
            return Err(SdfCellCodecError::DirtyGrid);
        }
        let layout = self.layout(grid.brick_count(), grid.sample_count())?;
        let count =
            u32::try_from(grid.brick_count()).map_err(|_| SdfCellCodecError::SizeOverflow)?;
        let body_bytes =
            u64::try_from(layout.body_bytes).map_err(|_| SdfCellCodecError::SizeOverflow)?;
        let mut encoded = Vec::new();
        encoded
            .try_reserve_exact(layout.frame_bytes)
            .map_err(|_| SdfCellCodecError::AllocationFailed)?;
        encoded.resize(HEADER_BYTES, 0);
        encoded[..4].copy_from_slice(&MAGIC);
        encoded[4..6].copy_from_slice(&SDF_CELL_CODEC_VERSION.to_le_bytes());
        encoded[8..12].copy_from_slice(&(HEADER_BYTES as u32).to_le_bytes());
        encoded[12..16].copy_from_slice(&(RECORD_BYTES as u32).to_le_bytes());
        encoded[16..24].copy_from_slice(&body_bytes.to_le_bytes());
        encoded[24..28].copy_from_slice(&count.to_le_bytes());
        encoded[28] = grid.brick_size();
        encoded[29] = grid.mip_levels();
        encoded[32] = cell.sector().level;
        for (offset, value) in
            [40, 48, 56]
                .into_iter()
                .zip([cell.sector().x, cell.sector().y, cell.sector().z])
        {
            encoded[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
        }
        let transform = snapshot.transform();
        for (offset, value) in [64, 72, 80].into_iter().zip(transform.origin_mm()) {
            encoded[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
        }
        encoded[88..92].copy_from_slice(&transform.voxel_size_mm().to_le_bytes());
        encoded[92..96].copy_from_slice(&transform.distance_scale_mm().to_le_bytes());
        encoded[96..104].copy_from_slice(&snapshot.revision().to_le_bytes());
        encoded[104..108].copy_from_slice(&(grid.sample_count() as u32).to_le_bytes());
        // The existing grid exposes a coordinate Vec; never duplicate samples
        // into an intermediate packed export just to encode the frame.
        for coord in grid.allocated_bricks() {
            for value in [coord.x, coord.y, coord.z] {
                encoded.extend_from_slice(&value.to_le_bytes());
            }
            encoded.extend_from_slice(&[coord.mip, 0, 0, 0]);
            let samples = grid
                .brick_samples(coord)
                .map_err(SdfCellCodecError::Grid)?
                .ok_or(SdfCellCodecError::BodyLengthMismatch)?;
            for sample in samples {
                encoded.extend_from_slice(&sample.to_le_bytes());
            }
        }
        let checksum = crc32(&encoded);
        encoded.extend_from_slice(&checksum.to_le_bytes());
        debug_assert_eq!(encoded.len(), layout.frame_bytes);
        Ok(encoded)
    }

    /// Validates a complete frame before constructing a fresh clean snapshot.
    ///
    /// No header, count, CRC, sector, order or mip rejection returns a partial
    /// grid. Publication and monotonic revision checks belong to the cell host.
    pub fn decode_payload(
        &self,
        manifest: &CellManifest,
        payload: &[u8],
    ) -> Result<SdfCellSnapshot, SdfCellCodecError> {
        self.check(
            SdfCellCodecResource::PayloadBytes,
            payload.len(),
            self.limits.max_payload_bytes,
        )?;
        if payload.len() < HEADER_BYTES + CHECKSUM_BYTES {
            return Err(SdfCellCodecError::Truncated);
        }
        if payload[..4] != MAGIC {
            return Err(SdfCellCodecError::InvalidMagic);
        }
        let version = u16::from_le_bytes([payload[4], payload[5]]);
        if version != SDF_CELL_CODEC_VERSION {
            return Err(SdfCellCodecError::UnsupportedVersion { found: version });
        }
        for (valid, field) in [
            (payload[6..8] == [0; 2], "flags"),
            (read_u32(payload, 8) == HEADER_BYTES as u32, "header size"),
            (read_u32(payload, 12) == RECORD_BYTES as u32, "record size"),
            (payload[30..32] == [0; 2], "grid reserved bytes"),
            (payload[33..40] == [0; 7], "sector reserved bytes"),
            (payload[108..128] == [0; 20], "frame reserved bytes"),
        ] {
            if !valid {
                return Err(SdfCellCodecError::InvalidHeader { field });
            }
        }
        let mut grid =
            SparseSdfBrickGrid::new(payload[28], payload[29]).map_err(SdfCellCodecError::Grid)?;
        if read_u32(payload, 104) as usize != grid.sample_count() {
            return Err(SdfCellCodecError::InvalidHeader {
                field: "sample volume",
            });
        }
        let transform = WorldSdfTransform::new(
            [
                read_i64(payload, 64),
                read_i64(payload, 72),
                read_i64(payload, 80),
            ],
            read_u32(payload, 88),
            read_u32(payload, 92),
        )
        .map_err(SdfCellCodecError::Transform)?;
        let count =
            usize::try_from(read_u32(payload, 24)).map_err(|_| SdfCellCodecError::SizeOverflow)?;
        let layout = self.layout(count, grid.sample_count())?;
        let body_bytes =
            usize::try_from(read_u64(payload, 16)).map_err(|_| SdfCellCodecError::SizeOverflow)?;
        if body_bytes != layout.body_bytes {
            return Err(SdfCellCodecError::BodyLengthMismatch);
        }
        if payload.len() < layout.frame_bytes {
            return Err(SdfCellCodecError::Truncated);
        }
        if payload.len() > layout.frame_bytes {
            return Err(SdfCellCodecError::TrailingBytes);
        }
        let body_end = layout.frame_bytes - CHECKSUM_BYTES;
        if read_u32(payload, body_end) != crc32(&payload[..body_end]) {
            return Err(SdfCellCodecError::ChecksumMismatch);
        }
        let sector = SectorId::new(
            payload[32],
            read_i64(payload, 40),
            read_i64(payload, 48),
            read_i64(payload, 56),
        );
        if sector != manifest.sector {
            return Err(SdfCellCodecError::ManifestSectorMismatch {
                manifest: manifest.sector,
                encoded: sector,
            });
        }
        let mut previous = None;
        for index in 0..count {
            let offset = HEADER_BYTES + index * layout.record_bytes;
            let coord = read_coord(payload, offset);
            if payload[offset + 13..offset + 16] != [0; 3] {
                return Err(SdfCellCodecError::InvalidHeader {
                    field: "brick reserved bytes",
                });
            }
            if coord.mip >= grid.mip_levels() {
                return Err(SdfCellCodecError::Grid(SdfError::InvalidMip));
            }
            if let Some(previous) = previous
                && coord <= previous
            {
                return Err(SdfCellCodecError::NonCanonicalBrickOrder {
                    previous,
                    current: coord,
                });
            }
            previous = Some(coord);
        }
        for index in 0..count {
            let offset = HEADER_BYTES + index * layout.record_bytes;
            let coord = read_coord(payload, offset);
            let mut samples = Vec::new();
            samples
                .try_reserve_exact(grid.sample_count())
                .map_err(|_| SdfCellCodecError::AllocationFailed)?;
            for pair in payload[offset + RECORD_BYTES..offset + layout.record_bytes].chunks_exact(2)
            {
                samples.push(i16::from_le_bytes([pair[0], pair[1]]));
            }
            grid.insert_brick(coord, samples)
                .map_err(SdfCellCodecError::Grid)?;
        }
        Ok(SdfCellSnapshot::new(
            sector,
            WorldSdfSnapshot::new(transform, read_u64(payload, 96), grid),
        ))
    }

    fn check(
        &self,
        resource: SdfCellCodecResource,
        required: usize,
        limit: usize,
    ) -> Result<(), SdfCellCodecError> {
        if required > limit {
            return Err(SdfCellCodecError::LimitExceeded {
                resource,
                required,
                limit,
            });
        }
        Ok(())
    }

    fn layout(&self, bricks: usize, samples_per_brick: usize) -> Result<Layout, SdfCellCodecError> {
        self.check(SdfCellCodecResource::Bricks, bricks, self.limits.max_bricks)?;
        let samples = bricks
            .checked_mul(samples_per_brick)
            .ok_or(SdfCellCodecError::SizeOverflow)?;
        self.check(
            SdfCellCodecResource::Samples,
            samples,
            self.limits.max_samples,
        )?;
        let sample_bytes = samples
            .checked_mul(WORLD_SDF_SAMPLE_BYTES)
            .ok_or(SdfCellCodecError::SizeOverflow)?;
        let logical_bytes = bricks
            .checked_mul(WORLD_SDF_BRICK_METADATA_BYTES)
            .and_then(|n| n.checked_add(WORLD_SDF_HEADER_BYTES))
            .and_then(|n| n.checked_add(sample_bytes))
            .ok_or(SdfCellCodecError::SizeOverflow)?;
        self.check(
            SdfCellCodecResource::DecodedBytes,
            logical_bytes,
            self.limits.max_decoded_bytes,
        )?;
        let record_bytes = samples_per_brick
            .checked_mul(WORLD_SDF_SAMPLE_BYTES)
            .and_then(|n| n.checked_add(RECORD_BYTES))
            .ok_or(SdfCellCodecError::SizeOverflow)?;
        let body_bytes = bricks
            .checked_mul(record_bytes)
            .ok_or(SdfCellCodecError::SizeOverflow)?;
        let frame_bytes = body_bytes
            .checked_add(HEADER_BYTES + CHECKSUM_BYTES)
            .ok_or(SdfCellCodecError::SizeOverflow)?;
        self.check(
            SdfCellCodecResource::PayloadBytes,
            frame_bytes,
            self.limits.max_payload_bytes,
        )?;
        Ok(Layout {
            body_bytes,
            frame_bytes,
            record_bytes,
        })
    }
}

impl SdfCellDecoder for SdfCellCodec {
    type Error = SdfCellCodecError;
    fn decode(
        &mut self,
        manifest: &CellManifest,
        payload: &[u8],
    ) -> Result<SdfCellSnapshot, Self::Error> {
        self.decode_payload(manifest, payload)
    }
}

struct Layout {
    body_bytes: usize,
    frame_bytes: usize,
    record_bytes: usize,
}

fn read_coord(bytes: &[u8], start: usize) -> BrickCoord {
    BrickCoord::new(
        read_u32(bytes, start) as i32,
        read_u32(bytes, start + 4) as i32,
        read_u32(bytes, start + 8) as i32,
        bytes[start + 12],
    )
}
fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}
fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
        bytes[offset + 4],
        bytes[offset + 5],
        bytes[offset + 6],
        bytes[offset + 7],
    ])
}
fn read_i64(bytes: &[u8], offset: usize) -> i64 {
    read_u64(bytes, offset) as i64
}

// A compile-time table keeps large derived-cell CRC scans to one lookup/byte.
const CRC_TABLE: [u32; 256] = {
    let mut table = [0; 256];
    let mut i = 0;
    while i < 256 {
        let mut value = i as u32;
        let mut bit = 0;
        while bit < 8 {
            value = (value >> 1) ^ if value & 1 != 0 { 0xedb8_8320 } else { 0 };
            bit += 1;
        }
        table[i] = value;
        i += 1;
    }
    table
};
fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = u32::MAX;
    for byte in bytes {
        crc = (crc >> 8) ^ CRC_TABLE[((crc ^ u32::from(*byte)) & 255) as usize];
    }
    !crc
}
