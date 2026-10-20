//! Versioned, bounded codec for terrain-derived physics cell payloads.

use alloc::vec::Vec;
use core::fmt;
use core::mem::size_of;

use prism_world_stream::{CellManifest, SectorId};
use prism_world_terrain::{
    BakeLimits, CubeFace, HeightfieldImportError, HeightfieldSpec, HeightfieldTile,
    PlanarHeightfieldEmbedding, TileKey,
};

use crate::{
    DecodedTerrainHeightfield, TerrainCellDecoder, TerrainCellSnapshot, TerrainCellSnapshotError,
};

/// Stable terrain physics cell payload version.
pub const TERRAIN_CELL_CODEC_VERSION: u16 = 1;

const MAGIC: [u8; 4] = *b"PTHC";
const HEADER_BYTES: usize = 64;
const RECORD_HEADER_BYTES: usize = 64;
const CHECKSUM_BYTES: usize = 4;

/// Independent admission limits for encoded and retained decoded cell data.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TerrainCellCodecLimits {
    /// Maximum complete payload bytes, including frame header and checksum.
    pub max_payload_bytes: usize,
    /// Maximum number of heightfield records in one cell.
    pub max_heightfields: usize,
    /// Maximum sample count in any one heightfield.
    pub max_samples_per_heightfield: usize,
    /// Maximum aggregate sample count in one decoded cell.
    pub max_total_samples: usize,
    /// Maximum retained logical decoded bytes.
    ///
    /// This is sample bytes plus one [`DecodedTerrainHeightfield`] value per
    /// record. It does not claim to bound allocator metadata, capacity rounding,
    /// transient caller storage, or process RSS.
    pub max_decoded_bytes: usize,
}

impl TerrainCellCodecLimits {
    /// Creates explicit codec limits without silently selecting product policy.
    #[must_use]
    pub const fn new(
        max_payload_bytes: usize,
        max_heightfields: usize,
        max_samples_per_heightfield: usize,
        max_total_samples: usize,
        max_decoded_bytes: usize,
    ) -> Self {
        Self {
            max_payload_bytes,
            max_heightfields,
            max_samples_per_heightfield,
            max_total_samples,
            max_decoded_bytes,
        }
    }
}

/// Failure while encoding or decoding a terrain physics cell payload.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TerrainCellCodecError {
    /// Checked size arithmetic overflowed.
    SizeOverflow,
    /// A bounded allocation failed.
    AllocationFailed,
    /// The complete encoded payload exceeds its configured limit.
    PayloadLimitExceeded {
        /// Required payload bytes.
        required: usize,
        /// Configured maximum bytes.
        limit: usize,
    },
    /// The cell contains too many heightfields.
    HeightfieldLimitExceeded {
        /// Required record count.
        required: usize,
        /// Configured maximum count.
        limit: usize,
    },
    /// One tile contains too many samples.
    HeightfieldSampleLimitExceeded {
        /// Rejected tile.
        tile: TileKey,
        /// Required sample count.
        required: usize,
        /// Configured maximum count.
        limit: usize,
    },
    /// Aggregate cell samples exceed their configured limit.
    TotalSampleLimitExceeded {
        /// Required aggregate sample count.
        required: usize,
        /// Configured maximum count.
        limit: usize,
    },
    /// Retained logical decoded bytes exceed their configured limit.
    DecodedBytesLimitExceeded {
        /// Required logical decoded bytes.
        required: usize,
        /// Configured maximum bytes.
        limit: usize,
    },
    /// The payload ended before its declared complete frame.
    Truncated,
    /// Bytes remain after the declared frame checksum.
    TrailingBytes,
    /// The four-byte payload identity is not `PTHC`.
    InvalidMagic,
    /// The payload uses an unsupported format version.
    UnsupportedVersion {
        /// Encountered version.
        found: u16,
    },
    /// Unknown frame flags were set.
    InvalidFlags {
        /// Encountered flag bits.
        found: u16,
    },
    /// The frame header size does not match this version.
    InvalidHeaderSize {
        /// Encountered byte count.
        found: u32,
    },
    /// The record header size does not match this version.
    InvalidRecordHeaderSize {
        /// Encountered byte count.
        found: u32,
    },
    /// A reserved field was non-zero.
    NonZeroReserved {
        /// Header region containing the non-zero byte.
        scope: &'static str,
    },
    /// The payload sector does not match the authoritative manifest.
    ManifestSectorMismatch {
        /// Manifest sector.
        manifest: SectorId,
        /// Encoded sector.
        encoded: SectorId,
    },
    /// The frame checksum does not match its header and body.
    ChecksumMismatch {
        /// Checksum stored in the payload.
        expected: u32,
        /// Checksum computed from frame bytes.
        actual: u32,
    },
    /// Record count and body length do not describe the same bytes.
    BodyLengthMismatch,
    /// A face byte is not one of the six stable cube face values.
    InvalidFace {
        /// Rejected face value.
        value: u8,
    },
    /// A tile level or coordinate is outside the terrain contract.
    InvalidTile,
    /// A heightfield resolution or height scale is invalid.
    InvalidHeightfieldSpec,
    /// Sample spacing is zero.
    InvalidSpacing,
    /// An embedded X or Z endpoint cannot be represented as `i64`.
    CoordinateOverflow,
    /// A record sample count differs from its square resolution.
    SampleCountMismatch {
        /// Count implied by the heightfield specification.
        expected: usize,
        /// Count encoded in the record.
        actual: usize,
    },
    /// Records are duplicated or not in strict stable tile order.
    NonCanonicalTileOrder {
        /// Previous record tile.
        previous: TileKey,
        /// Current record tile.
        current: TileKey,
    },
    /// The terrain crate rejected restoration of a fully preflighted tile.
    HeightfieldImport(HeightfieldImportError),
    /// Final snapshot validation failed.
    Snapshot(TerrainCellSnapshotError),
}

impl fmt::Display for TerrainCellCodecError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SizeOverflow => formatter.write_str("terrain cell size arithmetic overflowed"),
            Self::AllocationFailed => formatter.write_str("terrain cell allocation failed"),
            Self::PayloadLimitExceeded { required, limit } => write!(
                formatter,
                "terrain cell payload requires {required} bytes, limit is {limit}"
            ),
            Self::HeightfieldLimitExceeded { required, limit } => write!(
                formatter,
                "terrain cell requires {required} heightfields, limit is {limit}"
            ),
            Self::HeightfieldSampleLimitExceeded {
                tile,
                required,
                limit,
            } => write!(
                formatter,
                "terrain tile {tile:?} requires {required} samples, limit is {limit}"
            ),
            Self::TotalSampleLimitExceeded { required, limit } => write!(
                formatter,
                "terrain cell requires {required} total samples, limit is {limit}"
            ),
            Self::DecodedBytesLimitExceeded { required, limit } => write!(
                formatter,
                "terrain cell requires {required} decoded logical bytes, limit is {limit}"
            ),
            Self::Truncated => formatter.write_str("terrain cell payload is truncated"),
            Self::TrailingBytes => formatter.write_str("terrain cell payload has trailing bytes"),
            Self::InvalidMagic => formatter.write_str("terrain cell payload magic is invalid"),
            Self::UnsupportedVersion { found } => {
                write!(
                    formatter,
                    "unsupported terrain cell payload version {found}"
                )
            }
            Self::InvalidFlags { found } => {
                write!(
                    formatter,
                    "terrain cell payload flags {found:#06x} are invalid"
                )
            }
            Self::InvalidHeaderSize { found } => {
                write!(formatter, "terrain cell header size {found} is invalid")
            }
            Self::InvalidRecordHeaderSize { found } => {
                write!(
                    formatter,
                    "terrain cell record header size {found} is invalid"
                )
            }
            Self::NonZeroReserved { scope } => {
                write!(
                    formatter,
                    "terrain cell {scope} reserved bytes are non-zero"
                )
            }
            Self::ManifestSectorMismatch { manifest, encoded } => write!(
                formatter,
                "terrain cell sector mismatch: manifest {manifest:?}, encoded {encoded:?}"
            ),
            Self::ChecksumMismatch { expected, actual } => write!(
                formatter,
                "terrain cell checksum mismatch: stored {expected:#010x}, computed {actual:#010x}"
            ),
            Self::BodyLengthMismatch => {
                formatter.write_str("terrain cell record count does not match body length")
            }
            Self::InvalidFace { value } => write!(formatter, "invalid cube face value {value}"),
            Self::InvalidTile => formatter.write_str("invalid terrain tile identity"),
            Self::InvalidHeightfieldSpec => {
                formatter.write_str("invalid terrain heightfield specification")
            }
            Self::InvalidSpacing => {
                formatter.write_str("terrain heightfield sample spacing must be positive")
            }
            Self::CoordinateOverflow => {
                formatter.write_str("terrain heightfield embedding endpoint overflows i64")
            }
            Self::SampleCountMismatch { expected, actual } => write!(
                formatter,
                "terrain heightfield sample count {actual} does not match expected {expected}"
            ),
            Self::NonCanonicalTileOrder { previous, current } => write!(
                formatter,
                "terrain records are not strictly ordered: {previous:?} then {current:?}"
            ),
            Self::HeightfieldImport(error) => {
                write!(formatter, "terrain heightfield import failed: {error:?}")
            }
            Self::Snapshot(error) => {
                write!(formatter, "terrain snapshot validation failed: {error}")
            }
        }
    }
}

impl std::error::Error for TerrainCellCodecError {}

/// Deterministic `PTHC` version 1 terrain physics cell codec.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TerrainCellCodec {
    limits: TerrainCellCodecLimits,
}

impl TerrainCellCodec {
    /// Creates a codec with explicit independent admission limits.
    #[must_use]
    pub const fn new(limits: TerrainCellCodecLimits) -> Self {
        Self { limits }
    }

    /// Returns the active admission limits.
    #[must_use]
    pub const fn limits(&self) -> TerrainCellCodecLimits {
        self.limits
    }

    /// Encodes a canonical snapshot into a checksummed little-endian frame.
    pub fn encode(&self, snapshot: &TerrainCellSnapshot) -> Result<Vec<u8>, TerrainCellCodecError> {
        let heightfield_count = snapshot.heightfields().len();
        self.check_heightfield_count(heightfield_count)?;
        let mut body_bytes = 0usize;
        let mut total_samples = 0usize;
        let mut previous = None;
        for input in snapshot.heightfields() {
            let tile = input.tile();
            if let Some(previous) = previous
                && tile <= previous
            {
                return Err(TerrainCellCodecError::NonCanonicalTileOrder {
                    previous,
                    current: tile,
                });
            }
            previous = Some(tile);
            validate_embedding(input.heightfield().spec(), input.embedding())?;
            let samples = input.heightfield().samples().len();
            let side = usize::try_from(input.heightfield().spec().samples_per_edge())
                .map_err(|_| TerrainCellCodecError::SizeOverflow)?;
            let expected_samples = side
                .checked_mul(side)
                .ok_or(TerrainCellCodecError::SizeOverflow)?;
            if samples != expected_samples {
                return Err(TerrainCellCodecError::SampleCountMismatch {
                    expected: expected_samples,
                    actual: samples,
                });
            }
            self.check_per_heightfield(tile, samples)?;
            total_samples = total_samples
                .checked_add(samples)
                .ok_or(TerrainCellCodecError::SizeOverflow)?;
            let sample_bytes = samples
                .checked_mul(size_of::<i16>())
                .ok_or(TerrainCellCodecError::SizeOverflow)?;
            body_bytes = body_bytes
                .checked_add(RECORD_HEADER_BYTES)
                .and_then(|bytes| bytes.checked_add(sample_bytes))
                .ok_or(TerrainCellCodecError::SizeOverflow)?;
        }
        self.check_aggregate(heightfield_count, total_samples)?;
        let total_bytes = HEADER_BYTES
            .checked_add(body_bytes)
            .and_then(|bytes| bytes.checked_add(CHECKSUM_BYTES))
            .ok_or(TerrainCellCodecError::SizeOverflow)?;
        self.check_payload_bytes(total_bytes)?;
        let body_bytes_u64 =
            u64::try_from(body_bytes).map_err(|_| TerrainCellCodecError::SizeOverflow)?;
        let tile_count =
            u32::try_from(heightfield_count).map_err(|_| TerrainCellCodecError::SizeOverflow)?;

        let mut encoded = Vec::new();
        encoded
            .try_reserve_exact(total_bytes)
            .map_err(|_| TerrainCellCodecError::AllocationFailed)?;
        encoded.extend_from_slice(&MAGIC);
        push_u16(&mut encoded, TERRAIN_CELL_CODEC_VERSION);
        push_u16(&mut encoded, 0);
        push_u32(&mut encoded, HEADER_BYTES as u32);
        push_u32(&mut encoded, RECORD_HEADER_BYTES as u32);
        push_u64(&mut encoded, body_bytes_u64);
        push_u32(&mut encoded, tile_count);
        push_u32(&mut encoded, 0);
        encoded.push(snapshot.sector().level);
        encoded.extend_from_slice(&[0; 7]);
        push_i64(&mut encoded, snapshot.sector().x);
        push_i64(&mut encoded, snapshot.sector().y);
        push_i64(&mut encoded, snapshot.sector().z);
        debug_assert_eq!(encoded.len(), HEADER_BYTES);

        for input in snapshot.heightfields() {
            let tile = input.tile();
            let field = input.heightfield();
            let embedding = input.embedding();
            encoded.push(tile.face().index());
            encoded.push(tile.level());
            push_u16(&mut encoded, 0);
            push_u32(&mut encoded, tile.x());
            push_u32(&mut encoded, tile.y());
            push_u32(&mut encoded, field.spec().samples_per_edge());
            push_i64(&mut encoded, field.spec().height_scale_mm());
            push_u64(&mut encoded, field.revision());
            for coordinate in embedding.sample_origin_mm {
                push_i64(&mut encoded, coordinate);
            }
            push_u32(&mut encoded, embedding.sample_spacing_mm);
            push_u32(
                &mut encoded,
                u32::try_from(field.samples().len())
                    .map_err(|_| TerrainCellCodecError::SizeOverflow)?,
            );
            for &sample in field.samples() {
                encoded.extend_from_slice(&sample.to_le_bytes());
            }
        }
        let checksum = crc32(&encoded);
        push_u32(&mut encoded, checksum);
        debug_assert_eq!(encoded.len(), total_bytes);
        Ok(encoded)
    }

    /// Decodes and fully validates a payload against its authoritative manifest.
    pub fn decode_payload(
        &self,
        manifest: &CellManifest,
        payload: &[u8],
    ) -> Result<TerrainCellSnapshot, TerrainCellCodecError> {
        self.check_payload_bytes(payload.len())?;
        if payload.len() < HEADER_BYTES + CHECKSUM_BYTES {
            return Err(TerrainCellCodecError::Truncated);
        }
        if payload[0..4] != MAGIC {
            return Err(TerrainCellCodecError::InvalidMagic);
        }
        let version = read_u16(payload, 4);
        if version != TERRAIN_CELL_CODEC_VERSION {
            return Err(TerrainCellCodecError::UnsupportedVersion { found: version });
        }
        let flags = read_u16(payload, 6);
        if flags != 0 {
            return Err(TerrainCellCodecError::InvalidFlags { found: flags });
        }
        let header_bytes = read_u32(payload, 8);
        if header_bytes != HEADER_BYTES as u32 {
            return Err(TerrainCellCodecError::InvalidHeaderSize {
                found: header_bytes,
            });
        }
        let record_header_bytes = read_u32(payload, 12);
        if record_header_bytes != RECORD_HEADER_BYTES as u32 {
            return Err(TerrainCellCodecError::InvalidRecordHeaderSize {
                found: record_header_bytes,
            });
        }
        if payload[28..32].iter().any(|&byte| byte != 0) {
            return Err(TerrainCellCodecError::NonZeroReserved { scope: "frame" });
        }
        if payload[33..40].iter().any(|&byte| byte != 0) {
            return Err(TerrainCellCodecError::NonZeroReserved { scope: "sector" });
        }
        let body_bytes = usize::try_from(read_u64(payload, 16))
            .map_err(|_| TerrainCellCodecError::SizeOverflow)?;
        let body_end = HEADER_BYTES
            .checked_add(body_bytes)
            .ok_or(TerrainCellCodecError::SizeOverflow)?;
        let expected_payload_bytes = body_end
            .checked_add(CHECKSUM_BYTES)
            .ok_or(TerrainCellCodecError::SizeOverflow)?;
        self.check_payload_bytes(expected_payload_bytes)?;
        if payload.len() < expected_payload_bytes {
            return Err(TerrainCellCodecError::Truncated);
        }
        if payload.len() > expected_payload_bytes {
            return Err(TerrainCellCodecError::TrailingBytes);
        }
        let stored_checksum = read_u32(payload, body_end);
        let actual_checksum = crc32(&payload[..body_end]);
        if stored_checksum != actual_checksum {
            return Err(TerrainCellCodecError::ChecksumMismatch {
                expected: stored_checksum,
                actual: actual_checksum,
            });
        }
        let sector = SectorId::new(
            payload[32],
            read_i64(payload, 40),
            read_i64(payload, 48),
            read_i64(payload, 56),
        );
        if sector != manifest.sector {
            return Err(TerrainCellCodecError::ManifestSectorMismatch {
                manifest: manifest.sector,
                encoded: sector,
            });
        }
        let heightfield_count = usize::try_from(read_u32(payload, 24))
            .map_err(|_| TerrainCellCodecError::SizeOverflow)?;
        self.check_heightfield_count(heightfield_count)?;

        let mut cursor = HEADER_BYTES;
        let mut total_samples = 0usize;
        let mut previous = None;
        for _ in 0..heightfield_count {
            let record = parse_record(payload, cursor, body_end)?;
            if let Some(previous) = previous
                && record.tile <= previous
            {
                return Err(TerrainCellCodecError::NonCanonicalTileOrder {
                    previous,
                    current: record.tile,
                });
            }
            previous = Some(record.tile);
            self.check_per_heightfield(record.tile, record.sample_count)?;
            total_samples = total_samples
                .checked_add(record.sample_count)
                .ok_or(TerrainCellCodecError::SizeOverflow)?;
            cursor = record.end;
        }
        if cursor != body_end {
            return Err(TerrainCellCodecError::BodyLengthMismatch);
        }
        self.check_aggregate(heightfield_count, total_samples)?;

        let mut heightfields = Vec::new();
        heightfields
            .try_reserve_exact(heightfield_count)
            .map_err(|_| TerrainCellCodecError::AllocationFailed)?;
        cursor = HEADER_BYTES;
        for _ in 0..heightfield_count {
            let record = parse_record(payload, cursor, body_end)?;
            let samples = payload[record.samples_start..record.end]
                .chunks_exact(size_of::<i16>())
                .map(|bytes| i16::from_le_bytes([bytes[0], bytes[1]]));
            let tile = HeightfieldTile::from_encoded_samples(
                record.tile,
                record.spec,
                record.revision,
                samples,
                BakeLimits {
                    max_samples: self.limits.max_samples_per_heightfield,
                    max_bytes: self
                        .limits
                        .max_samples_per_heightfield
                        .checked_mul(size_of::<i16>())
                        .ok_or(TerrainCellCodecError::SizeOverflow)?,
                },
            )
            .map_err(TerrainCellCodecError::HeightfieldImport)?;
            heightfields.push(DecodedTerrainHeightfield::new(tile, record.embedding));
            cursor = record.end;
        }
        TerrainCellSnapshot::new(sector, heightfields).map_err(TerrainCellCodecError::Snapshot)
    }

    fn check_payload_bytes(&self, required: usize) -> Result<(), TerrainCellCodecError> {
        if required > self.limits.max_payload_bytes {
            return Err(TerrainCellCodecError::PayloadLimitExceeded {
                required,
                limit: self.limits.max_payload_bytes,
            });
        }
        Ok(())
    }

    fn check_heightfield_count(&self, required: usize) -> Result<(), TerrainCellCodecError> {
        if required > self.limits.max_heightfields {
            return Err(TerrainCellCodecError::HeightfieldLimitExceeded {
                required,
                limit: self.limits.max_heightfields,
            });
        }
        Ok(())
    }

    fn check_per_heightfield(
        &self,
        tile: TileKey,
        required: usize,
    ) -> Result<(), TerrainCellCodecError> {
        if required > self.limits.max_samples_per_heightfield {
            return Err(TerrainCellCodecError::HeightfieldSampleLimitExceeded {
                tile,
                required,
                limit: self.limits.max_samples_per_heightfield,
            });
        }
        Ok(())
    }

    fn check_aggregate(
        &self,
        heightfield_count: usize,
        total_samples: usize,
    ) -> Result<(), TerrainCellCodecError> {
        if total_samples > self.limits.max_total_samples {
            return Err(TerrainCellCodecError::TotalSampleLimitExceeded {
                required: total_samples,
                limit: self.limits.max_total_samples,
            });
        }
        let sample_bytes = total_samples
            .checked_mul(size_of::<i16>())
            .ok_or(TerrainCellCodecError::SizeOverflow)?;
        let metadata_bytes = heightfield_count
            .checked_mul(size_of::<DecodedTerrainHeightfield>())
            .ok_or(TerrainCellCodecError::SizeOverflow)?;
        let decoded_bytes = sample_bytes
            .checked_add(metadata_bytes)
            .ok_or(TerrainCellCodecError::SizeOverflow)?;
        if decoded_bytes > self.limits.max_decoded_bytes {
            return Err(TerrainCellCodecError::DecodedBytesLimitExceeded {
                required: decoded_bytes,
                limit: self.limits.max_decoded_bytes,
            });
        }
        Ok(())
    }
}

impl TerrainCellDecoder for TerrainCellCodec {
    type Error = TerrainCellCodecError;

    fn decode(
        &mut self,
        manifest: &CellManifest,
        payload: &[u8],
    ) -> Result<TerrainCellSnapshot, Self::Error> {
        self.decode_payload(manifest, payload)
    }
}

#[derive(Clone, Copy)]
struct RecordMeta {
    tile: TileKey,
    spec: HeightfieldSpec,
    revision: u64,
    embedding: PlanarHeightfieldEmbedding,
    sample_count: usize,
    samples_start: usize,
    end: usize,
}

fn parse_record(
    payload: &[u8],
    start: usize,
    body_end: usize,
) -> Result<RecordMeta, TerrainCellCodecError> {
    let header_end = start
        .checked_add(RECORD_HEADER_BYTES)
        .ok_or(TerrainCellCodecError::SizeOverflow)?;
    if header_end > body_end {
        return Err(TerrainCellCodecError::Truncated);
    }
    if read_u16(payload, start + 2) != 0 {
        return Err(TerrainCellCodecError::NonZeroReserved { scope: "record" });
    }
    let face = decode_face(payload[start])?;
    let tile = TileKey::new(
        face,
        payload[start + 1],
        read_u32(payload, start + 4),
        read_u32(payload, start + 8),
    )
    .map_err(|_| TerrainCellCodecError::InvalidTile)?;
    let spec = HeightfieldSpec::new(read_u32(payload, start + 12), read_i64(payload, start + 16))
        .map_err(|_| TerrainCellCodecError::InvalidHeightfieldSpec)?;
    let embedding = PlanarHeightfieldEmbedding {
        sample_origin_mm: [
            read_i64(payload, start + 32),
            read_i64(payload, start + 40),
            read_i64(payload, start + 48),
        ],
        sample_spacing_mm: read_u32(payload, start + 56),
    };
    validate_embedding(spec, embedding)?;
    let sample_count = usize::try_from(read_u32(payload, start + 60))
        .map_err(|_| TerrainCellCodecError::SizeOverflow)?;
    let side = usize::try_from(spec.samples_per_edge())
        .map_err(|_| TerrainCellCodecError::SizeOverflow)?;
    let expected = side
        .checked_mul(side)
        .ok_or(TerrainCellCodecError::SizeOverflow)?;
    if sample_count != expected {
        return Err(TerrainCellCodecError::SampleCountMismatch {
            expected,
            actual: sample_count,
        });
    }
    let sample_bytes = sample_count
        .checked_mul(size_of::<i16>())
        .ok_or(TerrainCellCodecError::SizeOverflow)?;
    let end = header_end
        .checked_add(sample_bytes)
        .ok_or(TerrainCellCodecError::SizeOverflow)?;
    if end > body_end {
        return Err(TerrainCellCodecError::Truncated);
    }
    Ok(RecordMeta {
        tile,
        spec,
        revision: read_u64(payload, start + 24),
        embedding,
        sample_count,
        samples_start: header_end,
        end,
    })
}

fn validate_embedding(
    spec: HeightfieldSpec,
    embedding: PlanarHeightfieldEmbedding,
) -> Result<(), TerrainCellCodecError> {
    if embedding.sample_spacing_mm == 0 {
        return Err(TerrainCellCodecError::InvalidSpacing);
    }
    let span = i128::from(spec.samples_per_edge() - 1) * i128::from(embedding.sample_spacing_mm);
    for origin in [embedding.sample_origin_mm[0], embedding.sample_origin_mm[2]] {
        i64::try_from(i128::from(origin) + span)
            .map_err(|_| TerrainCellCodecError::CoordinateOverflow)?;
    }
    Ok(())
}

fn decode_face(value: u8) -> Result<CubeFace, TerrainCellCodecError> {
    match value {
        0 => Ok(CubeFace::PositiveX),
        1 => Ok(CubeFace::NegativeX),
        2 => Ok(CubeFace::PositiveY),
        3 => Ok(CubeFace::NegativeY),
        4 => Ok(CubeFace::PositiveZ),
        5 => Ok(CubeFace::NegativeZ),
        value => Err(TerrainCellCodecError::InvalidFace { value }),
    }
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = u32::MAX;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            let mask = 0_u32.wrapping_sub(crc & 1);
            crc = (crc >> 1) ^ (0xedb8_8320 & mask);
        }
    }
    !crc
}

fn push_u16(bytes: &mut Vec<u8>, value: u16) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_i64(bytes: &mut Vec<u8>, value: i64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn read_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
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
    i64::from_le_bytes([
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

#[cfg(test)]
mod tests {
    use super::*;
    use core::convert::Infallible;

    use prism_world_physics_bridge::{HeightfieldBridgeBudget, HeightfieldResidencyBudget};
    use prism_world_runtime::SceneHost;
    use prism_world_stream::{AabbI64, DataLayerMask, GlobalPos};

    use crate::{PhysicsCellHost, PhysicsCellHostError};

    const FIRST_RECORD: usize = HEADER_BYTES;

    #[derive(Default)]
    struct Host {
        loads: usize,
        evictions: usize,
    }

    impl SceneHost for Host {
        type Error = Infallible;

        fn load_cell(
            &mut self,
            _manifest: &CellManifest,
            _payload: &[u8],
        ) -> Result<(), Self::Error> {
            self.loads += 1;
            Ok(())
        }

        fn evict_cell(&mut self, _sector: SectorId) -> Result<(), Self::Error> {
            self.evictions += 1;
            Ok(())
        }
    }

    fn limits() -> TerrainCellCodecLimits {
        TerrainCellCodecLimits::new(4_096, 4, 16, 64, 4_096)
    }

    fn manifest(sector: SectorId) -> CellManifest {
        CellManifest {
            sector,
            bounds: AabbI64::new(
                GlobalPos::new(sector.x, sector.y, sector.z),
                GlobalPos::new(sector.x + 1, sector.y + 1, sector.z + 1),
            )
            .expect("test bounds are ordered"),
            scene_key: 7,
            content_version: 11,
            derived_hash: 13,
            io_bytes: 17,
            data_layers: DataLayerMask::EMPTY,
        }
    }

    fn terrain(
        tile: TileKey,
        revision: u64,
        samples: [i16; 4],
        origin: [i64; 3],
    ) -> DecodedTerrainHeightfield {
        let spec = HeightfieldSpec::new(2, 25).expect("valid test specification");
        let field = HeightfieldTile::from_encoded_samples(
            tile,
            spec,
            revision,
            samples,
            BakeLimits {
                max_samples: 4,
                max_bytes: 8,
            },
        )
        .expect("valid encoded test tile");
        DecodedTerrainHeightfield::new(
            field,
            PlanarHeightfieldEmbedding {
                sample_origin_mm: origin,
                sample_spacing_mm: 100,
            },
        )
    }

    fn snapshot(sector: SectorId, revision: u64) -> TerrainCellSnapshot {
        TerrainCellSnapshot::new(
            sector,
            vec![
                terrain(
                    TileKey::root(CubeFace::NegativeX),
                    revision + 1,
                    [4, 3, 2, 1],
                    [400, 500, 600],
                ),
                terrain(
                    TileKey::root(CubeFace::PositiveX),
                    revision,
                    [-4, -3, -2, -1],
                    [-400, -500, -600],
                ),
            ],
        )
        .expect("distinct tiles")
    }

    fn one_tile_snapshot(sector: SectorId, revision: u64) -> TerrainCellSnapshot {
        TerrainCellSnapshot::new(
            sector,
            vec![terrain(
                TileKey::root(CubeFace::PositiveZ),
                revision,
                [-2, -1, 0, 1],
                [10, 20, 30],
            )],
        )
        .expect("one tile is canonical")
    }

    fn refresh_crc(bytes: &mut [u8]) {
        let body_end = bytes.len() - CHECKSUM_BYTES;
        let checksum = crc32(&bytes[..body_end]);
        bytes[body_end..].copy_from_slice(&checksum.to_le_bytes());
    }

    fn second_record_offset(bytes: &[u8]) -> usize {
        let first_samples = usize::try_from(read_u32(bytes, FIRST_RECORD + 60))
            .expect("u32 sample count fits usize");
        FIRST_RECORD + RECORD_HEADER_BYTES + first_samples * size_of::<i16>()
    }

    #[test]
    fn round_trip_is_deterministic_complete_and_cache_local() {
        let sector = SectorId::new(3, -7, 8, -9);
        let manifest = manifest(sector);
        let codec = TerrainCellCodec::new(limits());
        let source = snapshot(sector, 41);

        let first = codec.encode(&source).expect("encode succeeds");
        let second = codec.encode(&source).expect("repeat encode succeeds");
        assert_eq!(first, second);
        assert_eq!(&first[..4], b"PTHC");
        assert_eq!(read_u16(&first, 4), TERRAIN_CELL_CODEC_VERSION);

        let decoded = codec
            .decode_payload(&manifest, &first)
            .expect("decode succeeds");
        assert_eq!(decoded.sector(), sector);
        assert_eq!(decoded.heightfields().len(), source.heightfields().len());
        for (actual, expected) in decoded.heightfields().iter().zip(source.heightfields()) {
            assert_eq!(actual.tile(), expected.tile());
            assert_eq!(actual.heightfield().spec(), expected.heightfield().spec());
            assert_eq!(
                actual.heightfield().revision(),
                expected.heightfield().revision()
            );
            assert_eq!(actual.embedding(), expected.embedding());
            assert_eq!(
                actual.heightfield().samples(),
                expected.heightfield().samples()
            );
        }
    }

    #[test]
    fn every_truncated_prefix_is_rejected_without_panicking() {
        let sector = SectorId::new(0, 1, 2, 3);
        let codec = TerrainCellCodec::new(limits());
        let encoded = codec.encode(&snapshot(sector, 1)).expect("encode succeeds");
        for end in 0..encoded.len() {
            assert!(
                matches!(
                    codec.decode_payload(&manifest(sector), &encoded[..end]),
                    Err(TerrainCellCodecError::Truncated)
                ),
                "prefix length {end} was not rejected as truncated"
            );
        }
    }

    #[test]
    fn frame_identity_shape_reserved_sector_checksum_and_trailing_bytes_are_strict() {
        let sector = SectorId::new(0, 1, 2, 3);
        let codec = TerrainCellCodec::new(limits());
        let encoded = codec
            .encode(&one_tile_snapshot(sector, 1))
            .expect("encode succeeds");

        let mut changed = encoded.clone();
        changed[0] ^= 1;
        assert!(matches!(
            codec.decode_payload(&manifest(sector), &changed),
            Err(TerrainCellCodecError::InvalidMagic)
        ));

        let mut changed = encoded.clone();
        changed[4..6].copy_from_slice(&(TERRAIN_CELL_CODEC_VERSION + 1).to_le_bytes());
        assert!(matches!(
            codec.decode_payload(&manifest(sector), &changed),
            Err(TerrainCellCodecError::UnsupportedVersion { .. })
        ));

        let mut changed = encoded.clone();
        changed[6..8].copy_from_slice(&1_u16.to_le_bytes());
        assert!(matches!(
            codec.decode_payload(&manifest(sector), &changed),
            Err(TerrainCellCodecError::InvalidFlags { found: 1 })
        ));

        let mut changed = encoded.clone();
        changed[8..12].copy_from_slice(&63_u32.to_le_bytes());
        assert!(matches!(
            codec.decode_payload(&manifest(sector), &changed),
            Err(TerrainCellCodecError::InvalidHeaderSize { found: 63 })
        ));

        let mut changed = encoded.clone();
        changed[12..16].copy_from_slice(&63_u32.to_le_bytes());
        assert!(matches!(
            codec.decode_payload(&manifest(sector), &changed),
            Err(TerrainCellCodecError::InvalidRecordHeaderSize { found: 63 })
        ));

        let mut changed = encoded.clone();
        changed[28] = 1;
        assert!(matches!(
            codec.decode_payload(&manifest(sector), &changed),
            Err(TerrainCellCodecError::NonZeroReserved { scope: "frame" })
        ));

        let mut changed = encoded.clone();
        changed[33] = 1;
        assert!(matches!(
            codec.decode_payload(&manifest(sector), &changed),
            Err(TerrainCellCodecError::NonZeroReserved { scope: "sector" })
        ));

        assert!(matches!(
            codec.decode_payload(&manifest(SectorId::new(0, 9, 2, 3)), &encoded),
            Err(TerrainCellCodecError::ManifestSectorMismatch { .. })
        ));

        let mut changed = encoded.clone();
        changed[FIRST_RECORD + RECORD_HEADER_BYTES] ^= 1;
        assert!(matches!(
            codec.decode_payload(&manifest(sector), &changed),
            Err(TerrainCellCodecError::ChecksumMismatch { .. })
        ));

        let mut changed = encoded.clone();
        changed.push(0);
        assert!(matches!(
            codec.decode_payload(&manifest(sector), &changed),
            Err(TerrainCellCodecError::TrailingBytes)
        ));
    }

    #[test]
    fn record_fields_and_embedding_endpoints_are_fully_validated() {
        let sector = SectorId::new(0, 0, 0, 0);
        let codec = TerrainCellCodec::new(limits());
        let encoded = codec
            .encode(&one_tile_snapshot(sector, 1))
            .expect("encode succeeds");

        let mut changed = encoded.clone();
        changed[FIRST_RECORD] = 6;
        refresh_crc(&mut changed);
        assert!(matches!(
            codec.decode_payload(&manifest(sector), &changed),
            Err(TerrainCellCodecError::InvalidFace { value: 6 })
        ));

        let mut changed = encoded.clone();
        changed[FIRST_RECORD + 1] = 31;
        refresh_crc(&mut changed);
        assert!(matches!(
            codec.decode_payload(&manifest(sector), &changed),
            Err(TerrainCellCodecError::InvalidTile)
        ));

        let mut changed = encoded.clone();
        changed[FIRST_RECORD + 4..FIRST_RECORD + 8].copy_from_slice(&1_u32.to_le_bytes());
        refresh_crc(&mut changed);
        assert!(matches!(
            codec.decode_payload(&manifest(sector), &changed),
            Err(TerrainCellCodecError::InvalidTile)
        ));

        let mut changed = encoded.clone();
        changed[FIRST_RECORD + 12..FIRST_RECORD + 16].copy_from_slice(&1_u32.to_le_bytes());
        refresh_crc(&mut changed);
        assert!(matches!(
            codec.decode_payload(&manifest(sector), &changed),
            Err(TerrainCellCodecError::InvalidHeightfieldSpec)
        ));

        let mut changed = encoded.clone();
        changed[FIRST_RECORD + 16..FIRST_RECORD + 24].copy_from_slice(&0_i64.to_le_bytes());
        refresh_crc(&mut changed);
        assert!(matches!(
            codec.decode_payload(&manifest(sector), &changed),
            Err(TerrainCellCodecError::InvalidHeightfieldSpec)
        ));

        let mut changed = encoded.clone();
        changed[FIRST_RECORD + 56..FIRST_RECORD + 60].copy_from_slice(&0_u32.to_le_bytes());
        refresh_crc(&mut changed);
        assert!(matches!(
            codec.decode_payload(&manifest(sector), &changed),
            Err(TerrainCellCodecError::InvalidSpacing)
        ));

        let mut changed = encoded.clone();
        changed[FIRST_RECORD + 32..FIRST_RECORD + 40].copy_from_slice(&i64::MAX.to_le_bytes());
        refresh_crc(&mut changed);
        assert!(matches!(
            codec.decode_payload(&manifest(sector), &changed),
            Err(TerrainCellCodecError::CoordinateOverflow)
        ));

        let mut changed = encoded.clone();
        changed[FIRST_RECORD + 60..FIRST_RECORD + 64].copy_from_slice(&3_u32.to_le_bytes());
        refresh_crc(&mut changed);
        assert!(matches!(
            codec.decode_payload(&manifest(sector), &changed),
            Err(TerrainCellCodecError::SampleCountMismatch {
                expected: 4,
                actual: 3
            })
        ));

        let mut changed = encoded.clone();
        changed[FIRST_RECORD + 2] = 1;
        refresh_crc(&mut changed);
        assert!(matches!(
            codec.decode_payload(&manifest(sector), &changed),
            Err(TerrainCellCodecError::NonZeroReserved { scope: "record" })
        ));
    }

    #[test]
    fn record_order_and_body_accounting_are_canonical() {
        let sector = SectorId::new(0, 0, 0, 0);
        let codec = TerrainCellCodec::new(limits());
        let encoded = codec.encode(&snapshot(sector, 1)).expect("encode succeeds");
        let second = second_record_offset(&encoded);

        let mut duplicate = encoded.clone();
        duplicate[second] = duplicate[FIRST_RECORD];
        refresh_crc(&mut duplicate);
        assert!(matches!(
            codec.decode_payload(&manifest(sector), &duplicate),
            Err(TerrainCellCodecError::NonCanonicalTileOrder {
                previous,
                current
            }) if previous == current
        ));

        let mut out_of_order = encoded.clone();
        out_of_order[FIRST_RECORD] = CubeFace::NegativeX.index();
        out_of_order[second] = CubeFace::PositiveX.index();
        refresh_crc(&mut out_of_order);
        assert!(matches!(
            codec.decode_payload(&manifest(sector), &out_of_order),
            Err(TerrainCellCodecError::NonCanonicalTileOrder { .. })
        ));

        let mut wrong_count = encoded.clone();
        wrong_count[24..28].copy_from_slice(&0_u32.to_le_bytes());
        refresh_crc(&mut wrong_count);
        assert!(matches!(
            codec.decode_payload(&manifest(sector), &wrong_count),
            Err(TerrainCellCodecError::BodyLengthMismatch)
        ));
    }

    #[test]
    fn all_independent_admission_budgets_fail_closed() {
        let sector = SectorId::new(0, 0, 0, 0);
        let source = one_tile_snapshot(sector, 1);
        let encoded = TerrainCellCodec::new(limits())
            .encode(&source)
            .expect("encode succeeds");

        let codec = TerrainCellCodec::new(TerrainCellCodecLimits::new(
            encoded.len() - 1,
            4,
            16,
            64,
            4_096,
        ));
        assert!(matches!(
            codec.decode_payload(&manifest(sector), &encoded),
            Err(TerrainCellCodecError::PayloadLimitExceeded { .. })
        ));
        assert!(matches!(
            codec.encode(&source),
            Err(TerrainCellCodecError::PayloadLimitExceeded { .. })
        ));

        let codec = TerrainCellCodec::new(TerrainCellCodecLimits::new(4_096, 0, 16, 64, 4_096));
        assert!(matches!(
            codec.decode_payload(&manifest(sector), &encoded),
            Err(TerrainCellCodecError::HeightfieldLimitExceeded {
                required: 1,
                limit: 0
            })
        ));

        let codec = TerrainCellCodec::new(TerrainCellCodecLimits::new(4_096, 4, 3, 64, 4_096));
        assert!(matches!(
            codec.decode_payload(&manifest(sector), &encoded),
            Err(TerrainCellCodecError::HeightfieldSampleLimitExceeded {
                required: 4,
                limit: 3,
                ..
            })
        ));

        let codec = TerrainCellCodec::new(TerrainCellCodecLimits::new(4_096, 4, 16, 3, 4_096));
        assert!(matches!(
            codec.decode_payload(&manifest(sector), &encoded),
            Err(TerrainCellCodecError::TotalSampleLimitExceeded {
                required: 4,
                limit: 3
            })
        ));

        let required_decoded = size_of::<DecodedTerrainHeightfield>() + 4 * size_of::<i16>();
        let codec = TerrainCellCodec::new(TerrainCellCodecLimits::new(
            4_096,
            4,
            16,
            64,
            required_decoded - 1,
        ));
        assert!(matches!(
            codec.decode_payload(&manifest(sector), &encoded),
            Err(TerrainCellCodecError::DecodedBytesLimitExceeded {
                required,
                limit
            }) if required == required_decoded && limit == required_decoded - 1
        ));
    }

    #[test]
    fn codec_driven_reload_advances_revision_and_malformed_payload_is_atomic() {
        let sector = SectorId::new(0, 0, 0, 0);
        let manifest = manifest(sector);
        let codec = TerrainCellCodec::new(limits());
        let first = codec
            .encode(&one_tile_snapshot(sector, 7))
            .expect("first encode succeeds");
        let second = codec
            .encode(&one_tile_snapshot(sector, 8))
            .expect("second encode succeeds");
        let tile = TileKey::root(CubeFace::PositiveZ);
        let mut host = Host::default();
        let mut adapter = PhysicsCellHost::new(
            &mut host,
            codec,
            HeightfieldBridgeBudget::new(4, 48).expect("valid per-tile budget"),
            HeightfieldResidencyBudget::new(1, 4, 48).expect("valid residency budget"),
        );

        adapter
            .load_cell(&manifest, &first)
            .expect("first load succeeds");
        assert_eq!(
            adapter
                .residency()
                .snapshot(tile)
                .expect("resident after first load")
                .revision(),
            7
        );
        adapter
            .load_cell(&manifest, &second)
            .expect("newer reload succeeds");
        assert_eq!(
            adapter
                .residency()
                .snapshot(tile)
                .expect("resident after reload")
                .revision(),
            8
        );

        let mut malformed = second;
        malformed[FIRST_RECORD + RECORD_HEADER_BYTES] ^= 1;
        assert!(matches!(
            adapter.load_cell(&manifest, &malformed),
            Err(PhysicsCellHostError::Decoder(
                TerrainCellCodecError::ChecksumMismatch { .. }
            ))
        ));
        assert_eq!(adapter.host.loads, 2);
        assert_eq!(
            adapter
                .residency()
                .snapshot(tile)
                .expect("old resident survives malformed reload")
                .revision(),
            8
        );
    }
}
