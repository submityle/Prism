//! Version 1 soft-cache assets, independent of serde and world snapshots.
//!
//! All integers and IEEE-754 `f32` bits are little endian. The 64-byte header
//! contains magic (8), version/header size (4 each), asset size (8), frame dt,
//! quantizer step, key interval, track count, frame count, reserved (4 each),
//! index size (8), index CRC-32 (4), and header CRC-32 (4). The latter covers
//! the first 60 bytes. Magic is `PRSMPC\0\0`; version is 1; reserved is zero.
//!
//! The index contains, for each track, particle/frame/interval/reserved `u32`s,
//! followed by one 32-byte entry per frame: asset-relative offset/length `u64`s,
//! element count, kind (0 = key, 1 = delta), payload CRC-32, reserved `u32`s.
//! Payloads follow the index contiguously in track/frame order, without gaps
//! or aliases. Keys store xyz `i32`s; deltas store an index `u32` plus xyz
//! `i32`s, in strictly increasing particle-index order. Scheduled keys are
//! mandatory; extra keys are legal. CRC uses IEEE CRC-32 (polynomial
//! `0xedb88320`, initial/final xor `0xffffffff`). CRC detects accidental
//! corruption, not malicious replacement/authenticity.
//!
//! [`CacheReader`] opens at the stream's current position; offsets are relative
//! to that position. Trailing bytes may contain other assets. Opening validates
//! the entire index, not all payloads. Payloads are verified on demand, so a
//! random read does not read unrelated tracks/frames. Use [`CacheReader::load`]
//! to verify every payload and all delta arithmetic. Sources must remain
//! unchanged while a reader is in use. Writers require only [`Write`], emit
//! deterministic bytes, and never buffer the complete payload.

use alloc::vec::Vec;
use core::mem::size_of;
use std::io::{self, Read, Seek, SeekFrom, Write};

use glam::Vec3;

use super::track::checked_sum;
use super::{CacheTrack, FrameRecord, PhysicsCache, PositionQuantizer};

/// Current soft-cache disk format version (not a world-snapshot version).
pub const CACHE_FORMAT_VERSION: u32 = 1;
const MAGIC: &[u8; 8] = b"PRSMPC\0\0";
const HEADER_SIZE: u64 = 64;
const TRACK_SIZE: u64 = 16;
const ENTRY_SIZE: u64 = 32;

/// Caller-controlled resource ceilings, checked before count-based allocations.
///
/// Index bytes bound metadata, frame bytes bound individual payloads, and decoded
/// bytes bound the estimated owned track/frame/particle storage of a full load
/// (or a single track read). These are not exact process RSS limits: allocator
/// overhead, the parsed index and a temporary frame/state are additional.
#[derive(Clone, Copy, Debug)]
pub struct CacheLimits {
    /// Maximum encoded asset size, excluding any enclosing stream/trailing data.
    pub max_asset_bytes: u64,
    /// Maximum on-disk index size.
    pub max_index_bytes: u64,
    /// Maximum bytes in one encoded frame.
    pub max_frame_bytes: u64,
    /// Maximum estimated owned decoded storage for `load` / `read_track`.
    pub max_decoded_bytes: u64,
    /// Maximum tracks in an asset.
    pub max_tracks: u32,
    /// Maximum frames per track.
    pub max_frames: u32,
    /// Maximum particles per track, including empty tracks.
    pub max_particles: u32,
}

impl Default for CacheLimits {
    fn default() -> Self {
        Self {
            max_asset_bytes: 16 * 1024 * 1024 * 1024,
            max_index_bytes: 64 * 1024 * 1024,
            max_frame_bytes: 16 * 1024 * 1024,
            max_decoded_bytes: 512 * 1024 * 1024,
            max_tracks: 4096,
            max_frames: 1_000_000,
            max_particles: 1_000_000,
        }
    }
}

/// Global playback metadata validated when opening a cache asset.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CacheMetadata {
    /// Seconds between recorded frames (finite and positive).
    pub frame_dt: f32,
    /// Position grid spacing in metres (finite and positive).
    pub quantizer_step: f32,
    /// Shared, nonzero periodic keyframe interval.
    pub keyframe_interval: u32,
    /// Number of frames shared by all tracks; zero when there are no tracks.
    pub frame_count: u32,
    /// Total byte length of this asset, including header and index.
    pub asset_bytes: u64,
}

/// Validated per-track metadata; track identity is its index in the cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheTrackInfo {
    /// Number of particles in every keyframe of this track.
    pub particle_count: u32,
    /// Number of recorded frames.
    pub frame_count: u32,
    /// Periodic keyframe interval.
    pub keyframe_interval: u32,
}

#[derive(Clone, Copy, Debug)]
struct Entry {
    offset: u64,
    length: u64,
    count: u32,
    kind: u32,
    crc: u32,
}

/// Indexed, bounded reader over an existing soft-cache asset.
///
/// Accepts `File`, `BufReader<File>`, or `Cursor` through `Read + Seek`. Opening
/// and all reads return `InvalidData` for malformed/unsupported assets or limit
/// violations, `InvalidInput` for out-of-range requests, and propagate I/O errors.
pub struct CacheReader<R> {
    source: R,
    base: u64,
    metadata: CacheMetadata,
    tracks: Vec<CacheTrackInfo>,
    entries: Vec<Entry>,
    limits: CacheLimits,
}

impl<R: Read + Seek> CacheReader<R> {
    /// Opens an asset at the current stream position with default limits.
    pub fn open(source: R) -> io::Result<Self> {
        Self::with_limits(source, CacheLimits::default())
    }

    /// Opens an asset with explicit resource limits. Only its index is loaded.
    pub fn with_limits(mut source: R, limits: CacheLimits) -> io::Result<Self> {
        let base = source.stream_position()?;
        let available = source
            .seek(SeekFrom::End(0))?
            .checked_sub(base)
            .ok_or_else(|| invalid("asset starts beyond stream end"))?;
        source.seek(SeekFrom::Start(base))?;
        if available < HEADER_SIZE {
            return Err(invalid("truncated cache header"));
        }
        let mut header = [0u8; HEADER_SIZE as usize];
        source.read_exact(&mut header)?;
        if &header[..8] != MAGIC || get_u32(&header, 8) != CACHE_FORMAT_VERSION {
            return Err(invalid("unrecognized cache magic or version"));
        }
        if get_u32(&header, 12) != HEADER_SIZE as u32 || get_u32(&header, 44) != 0 {
            return Err(invalid("invalid cache header size or reserved bits"));
        }
        if crc32(&header[..60]) != get_u32(&header, 60) {
            return Err(invalid("cache header checksum mismatch"));
        }
        let metadata = CacheMetadata {
            asset_bytes: get_u64(&header, 16),
            frame_dt: f32::from_bits(get_u32(&header, 24)),
            quantizer_step: f32::from_bits(get_u32(&header, 28)),
            keyframe_interval: get_u32(&header, 32),
            frame_count: get_u32(&header, 40),
        };
        let track_count = get_u32(&header, 36);
        validate_metadata(metadata, track_count, limits)?;
        let index_len = index_length(track_count, metadata.frame_count, limits)?;
        if get_u64(&header, 48) != index_len
            || metadata.asset_bytes < add(HEADER_SIZE, index_len)?
            || metadata.asset_bytes > available
        {
            return Err(invalid(
                "invalid cache index/asset length or truncated asset",
            ));
        }
        let mut index = zero_bytes(index_len)?;
        source.read_exact(&mut index)?;
        if crc32(&index) != get_u32(&header, 56) {
            return Err(invalid("cache index checksum mismatch"));
        }
        let mut tracks = reserved(track_count as u64)?;
        let mut entries = reserved(u64::from(track_count) * u64::from(metadata.frame_count))?;
        let mut cursor = 0;
        let mut expected_offset = add(HEADER_SIZE, index_len)?;
        for _ in 0..track_count {
            let info = CacheTrackInfo {
                particle_count: get_u32(&index, cursor),
                frame_count: get_u32(&index, cursor + 4),
                keyframe_interval: get_u32(&index, cursor + 8),
            };
            if get_u32(&index, cursor + 12) != 0
                || info.frame_count != metadata.frame_count
                || info.keyframe_interval != metadata.keyframe_interval
                || info.particle_count > limits.max_particles
            {
                return Err(invalid("invalid cache track metadata"));
            }
            cursor += TRACK_SIZE as usize;
            for frame in 0..info.frame_count {
                let entry = Entry {
                    offset: get_u64(&index, cursor),
                    length: get_u64(&index, cursor + 8),
                    count: get_u32(&index, cursor + 16),
                    kind: get_u32(&index, cursor + 20),
                    crc: get_u32(&index, cursor + 24),
                };
                validate_entry(entry, info, frame, limits)?;
                if get_u32(&index, cursor + 28) != 0 || entry.offset != expected_offset {
                    return Err(invalid("invalid cache payload offset or reserved bits"));
                }
                expected_offset = add(expected_offset, entry.length)?;
                if expected_offset > metadata.asset_bytes {
                    return Err(invalid("cache payload exceeds asset length"));
                }
                entries.push(entry);
                cursor += ENTRY_SIZE as usize;
            }
            tracks.push(info);
        }
        if expected_offset != metadata.asset_bytes {
            return Err(invalid("unindexed bytes inside cache asset"));
        }
        Ok(Self {
            source,
            base,
            metadata,
            tracks,
            entries,
            limits,
        })
    }

    /// Global playback metadata, available without reading any frame payload.
    #[must_use]
    pub fn metadata(&self) -> CacheMetadata {
        self.metadata
    }

    /// Track descriptors, in original cache order.
    #[must_use]
    pub fn tracks(&self) -> &[CacheTrackInfo] {
        &self.tracks
    }

    /// Returns the underlying stream. Its current position is unspecified.
    pub fn into_inner(self) -> R {
        self.source
    }

    fn entry(&self, track: usize, frame: u32) -> io::Result<Entry> {
        let info = self
            .tracks
            .get(track)
            .ok_or_else(|| input("track out of range"))?;
        if frame >= info.frame_count {
            return Err(input("frame out of range"));
        }
        Ok(self.entries[track * info.frame_count as usize + frame as usize])
    }

    /// Reads and checks one encoded frame, without loading any predecessor.
    ///
    /// This checks payload CRC, lengths, and sparse index order/range; use
    /// `read_quantized_frame` to also check accumulated delta arithmetic.
    pub fn read_frame(&mut self, track: usize, frame: u32) -> io::Result<FrameRecord> {
        let entry = self.entry(track, frame)?;
        self.source
            .seek(SeekFrom::Start(add(self.base, entry.offset)?))?;
        let mut bytes = zero_bytes(entry.length)?;
        self.source.read_exact(&mut bytes)?;
        if crc32(&bytes) != entry.crc {
            return Err(invalid("cache frame checksum mismatch"));
        }
        if entry.kind == 0 {
            let mut positions = reserved(u64::from(entry.count))?;
            for chunk in bytes.chunks_exact(12) {
                positions.push(read_xyz(chunk, 0));
            }
            Ok(FrameRecord::Key(positions))
        } else {
            let mut deltas = reserved(u64::from(entry.count))?;
            let mut previous = None;
            for chunk in bytes.chunks_exact(16) {
                let index = get_u32(chunk, 0);
                if index >= self.tracks[track].particle_count
                    || previous.is_some_and(|p| index <= p)
                {
                    return Err(invalid(
                        "cache sparse indices must be ordered, unique and in range",
                    ));
                }
                previous = Some(index);
                deltas.push((index, read_xyz(chunk, 4)));
            }
            Ok(FrameRecord::Delta(deltas))
        }
    }

    /// Reconstructs just one frame, reading from the nearest preceding key.
    ///
    /// Does not load the whole track. At most one keyframe interval is replayed;
    /// an extra overflow key can make the interval shorter. Delta overflow is
    /// rejected, never wrapped. No solver/world state is involved.
    pub fn read_quantized_frame(&mut self, track: usize, frame: u32) -> io::Result<Vec<[i32; 3]>> {
        self.entry(track, frame)?;
        let mut start = frame;
        while self.entry(track, start)?.kind != 0 {
            start = start
                .checked_sub(1)
                .ok_or_else(|| invalid("missing cache keyframe"))?;
        }
        let mut state = Vec::new();
        for current in start..=frame {
            apply_record(&mut state, self.read_frame(track, current)?)?;
        }
        Ok(state)
    }

    /// Reconstructs and dequantizes one frame for soft-cache playback.
    pub fn read_positions(&mut self, track: usize, frame: u32) -> io::Result<Vec<Vec3>> {
        let quantized = self.read_quantized_frame(track, frame)?;
        let quantizer = PositionQuantizer::new(self.metadata.quantizer_step);
        let mut positions = reserved(quantized.len() as u64)?;
        positions.extend(quantized.into_iter().map(|q| quantizer.decode(q)));
        Ok(positions)
    }

    fn decoded_size(&self, track: usize) -> io::Result<u64> {
        let info = self
            .tracks
            .get(track)
            .ok_or_else(|| input("track out of range"))?;
        let mut size = add(
            size_of::<CacheTrack>() as u64,
            u64::from(info.frame_count) * size_of::<FrameRecord>() as u64,
        )?;
        if info.frame_count > 0 {
            size = add(size, u64::from(info.particle_count) * 12)?;
        }
        for frame in 0..info.frame_count {
            size = add(size, self.entry(track, frame)?.length)?;
        }
        Ok(size)
    }

    /// Fully loads and validates one track, with scratch state ready to append.
    pub fn read_track(&mut self, track: usize) -> io::Result<CacheTrack> {
        if self.decoded_size(track)? > self.limits.max_decoded_bytes {
            return Err(invalid("decoded cache track exceeds limit"));
        }
        let info = self.tracks[track];
        let mut frames = reserved(u64::from(info.frame_count))?;
        let mut state = Vec::new();
        for frame in 0..info.frame_count {
            let record = self.read_frame(track, frame)?;
            apply_record_ref(&mut state, &record)?;
            frames.push(record);
        }
        Ok(CacheTrack {
            particle_count: info.particle_count,
            keyframe_interval: info.keyframe_interval,
            frames,
            last_quantized: state,
        })
    }

    /// Fully loads an asset, verifying every payload and reconstruction chain.
    pub fn load(mut self) -> io::Result<PhysicsCache> {
        let mut size = size_of::<PhysicsCache>() as u64;
        for track in 0..self.tracks.len() {
            size = add(size, self.decoded_size(track)?)?;
        }
        if size > self.limits.max_decoded_bytes {
            return Err(invalid("decoded cache asset exceeds limit"));
        }
        let mut cache = PhysicsCache::new(
            self.metadata.frame_dt,
            PositionQuantizer::new(self.metadata.quantizer_step),
            self.metadata.keyframe_interval,
        );
        for track in 0..self.tracks.len() {
            cache.push_track(self.read_track(track)?);
        }
        Ok(cache)
    }
}

impl PhysicsCache {
    /// Writes deterministic version 1 soft-cache bytes using default limits.
    ///
    /// Validation finishes before the first write. I/O errors can still leave
    /// a partial asset; atomic file replacement and flushing belong to callers.
    pub fn write_to<W: Write>(&self, writer: W) -> io::Result<()> {
        self.write_to_with_limits(writer, CacheLimits::default())
    }

    /// Writes a cache with explicit limits, without buffering frame payloads.
    /// Layout/count budgets are checked without allocation before reserving the
    /// index or reconstructing any frame. Payload validation also completes
    /// before the first write.
    pub fn write_to_with_limits<W: Write>(
        &self,
        mut writer: W,
        limits: CacheLimits,
    ) -> io::Result<()> {
        let (metadata, track_count, index_len) = write_layout(self, limits)?;
        let mut index: Vec<u8> = reserved(index_len)?;
        let mut offset = add(HEADER_SIZE, index_len)?;
        for track in self.tracks() {
            let info = CacheTrackInfo {
                particle_count: track.particle_count(),
                frame_count: track.frame_count(),
                keyframe_interval: track.keyframe_interval(),
            };
            put_u32(&mut index, info.particle_count);
            put_u32(&mut index, info.frame_count);
            put_u32(&mut index, info.keyframe_interval);
            put_u32(&mut index, 0);
            let mut state = Vec::new();
            for record in &track.frames {
                let entry = record_entry(record, offset)?;
                apply_record_ref(&mut state, record)?;
                let mut checksum = Crc32::new();
                record_bytes(record, |bytes| {
                    checksum.update(bytes);
                    Ok(())
                })?;
                put_u64(&mut index, offset);
                put_u64(&mut index, entry.length);
                put_u32(&mut index, entry.count);
                put_u32(&mut index, entry.kind);
                put_u32(&mut index, checksum.finish());
                put_u32(&mut index, 0);
                offset = add(offset, entry.length)?;
            }
        }
        let mut header: Vec<u8> = reserved(HEADER_SIZE)?;
        header.extend_from_slice(MAGIC);
        put_u32(&mut header, CACHE_FORMAT_VERSION);
        put_u32(&mut header, HEADER_SIZE as u32);
        put_u64(&mut header, metadata.asset_bytes);
        put_u32(&mut header, metadata.frame_dt.to_bits());
        put_u32(&mut header, metadata.quantizer_step.to_bits());
        put_u32(&mut header, metadata.keyframe_interval);
        put_u32(&mut header, track_count);
        put_u32(&mut header, metadata.frame_count);
        put_u32(&mut header, 0);
        put_u64(&mut header, index_len);
        put_u32(&mut header, crc32(&index));
        let header_crc = crc32(&header);
        put_u32(&mut header, header_crc);
        writer.write_all(&header)?;
        writer.write_all(&index)?;
        for track in self.tracks() {
            for record in &track.frames {
                record_bytes(record, |bytes| writer.write_all(bytes))?;
            }
        }
        Ok(())
    }

    /// Loads a versioned soft-cache asset from a seekable stream.
    pub fn read_from<R: Read + Seek>(reader: R) -> io::Result<Self> {
        CacheReader::open(reader)?.load()
    }

    /// Loads a versioned soft-cache asset with explicit resource limits.
    pub fn read_from_with_limits<R: Read + Seek>(
        reader: R,
        limits: CacheLimits,
    ) -> io::Result<Self> {
        CacheReader::with_limits(reader, limits)?.load()
    }
}

// No buffers, reconstruction or payload traversal: even a late oversized frame
// must be rejected before earlier frames cause any count-based allocation.
fn write_layout(
    cache: &PhysicsCache,
    limits: CacheLimits,
) -> io::Result<(CacheMetadata, u32, u64)> {
    let track_count =
        u32::try_from(cache.track_count()).map_err(|_| invalid("too many cache tracks"))?;
    let mut metadata = CacheMetadata {
        frame_dt: cache.frame_dt(),
        quantizer_step: cache.quantizer().step(),
        keyframe_interval: cache.keyframe_interval(),
        frame_count: cache.frame_count(),
        asset_bytes: HEADER_SIZE,
    };
    validate_metadata(metadata, track_count, limits)?;
    let index_len = index_length(track_count, metadata.frame_count, limits)?;
    metadata.asset_bytes = add(HEADER_SIZE, index_len)?;
    validate_metadata(metadata, track_count, limits)?;
    for track in cache.tracks() {
        let info = CacheTrackInfo {
            particle_count: track.particle_count(),
            frame_count: track.frame_count(),
            keyframe_interval: track.keyframe_interval(),
        };
        if info.frame_count != metadata.frame_count
            || info.keyframe_interval != metadata.keyframe_interval
            || info.particle_count > limits.max_particles
            || track.frames.len() != info.frame_count as usize
        {
            return Err(invalid(
                "cache tracks must share frame count and interval within limits",
            ));
        }
        for (frame, record) in track.frames.iter().enumerate() {
            let entry = record_entry(record, metadata.asset_bytes)?;
            validate_entry(entry, info, frame as u32, limits)?;
            metadata.asset_bytes = add(metadata.asset_bytes, entry.length)?;
            if metadata.asset_bytes > limits.max_asset_bytes {
                return Err(invalid("cache asset exceeds byte limit"));
            }
        }
    }
    Ok((metadata, track_count, index_len))
}

fn record_entry(record: &FrameRecord, offset: u64) -> io::Result<Entry> {
    let (count, kind, stride) = match record {
        FrameRecord::Key(values) => (values.len(), 0, 12),
        FrameRecord::Delta(values) => (values.len(), 1, 16),
    };
    let count = u32::try_from(count).map_err(|_| invalid("cache frame count overflow"))?;
    Ok(Entry {
        offset,
        length: u64::from(count) * stride,
        count,
        kind,
        crc: 0,
    })
}

fn validate_metadata(metadata: CacheMetadata, tracks: u32, limits: CacheLimits) -> io::Result<()> {
    if !metadata.frame_dt.is_finite()
        || metadata.frame_dt <= 0.0
        || !metadata.quantizer_step.is_finite()
        || metadata.quantizer_step <= 0.0
        || metadata.keyframe_interval == 0
        || tracks > limits.max_tracks
        || metadata.frame_count > limits.max_frames
        || (tracks == 0 && metadata.frame_count != 0)
        || metadata.asset_bytes < HEADER_SIZE
        || metadata.asset_bytes > limits.max_asset_bytes
    {
        return Err(invalid("invalid cache metadata or resource limit exceeded"));
    }
    Ok(())
}

fn index_length(tracks: u32, frames: u32, limits: CacheLimits) -> io::Result<u64> {
    let length = u64::from(tracks)
        .checked_mul(add(TRACK_SIZE, u64::from(frames) * ENTRY_SIZE)?)
        .ok_or_else(|| invalid("cache index size overflow"))?;
    if length > limits.max_index_bytes {
        return Err(invalid("cache index exceeds limit"));
    }
    Ok(length)
}

fn validate_entry(
    entry: Entry,
    info: CacheTrackInfo,
    frame: u32,
    limits: CacheLimits,
) -> io::Result<()> {
    let stride = match entry.kind {
        0 if entry.count == info.particle_count => 12,
        1 if entry.count <= info.particle_count
            && !frame.is_multiple_of(info.keyframe_interval) =>
        {
            16
        }
        _ => {
            return Err(invalid(
                "invalid cache frame kind/count or missing periodic key",
            ))
        }
    };
    if entry.length != u64::from(entry.count) * stride || entry.length > limits.max_frame_bytes {
        return Err(invalid(
            "invalid cache frame length or resource limit exceeded",
        ));
    }
    Ok(())
}

fn apply_record(state: &mut Vec<[i32; 3]>, record: FrameRecord) -> io::Result<()> {
    match record {
        FrameRecord::Key(positions) => {
            *state = positions;
            Ok(())
        }
        delta => apply_record_ref(state, &delta),
    }
}

fn apply_record_ref(state: &mut Vec<[i32; 3]>, record: &FrameRecord) -> io::Result<()> {
    match record {
        FrameRecord::Key(positions) => {
            if state.capacity() < positions.len() {
                state
                    .try_reserve_exact(positions.len().saturating_sub(state.len()))
                    .map_err(|_| invalid("cache state allocation failed"))?;
            }
            state.clear();
            state.extend_from_slice(positions);
        }
        FrameRecord::Delta(deltas) => {
            let mut previous = None;
            for &(index, delta) in deltas {
                if previous.is_some_and(|p| index <= p) {
                    return Err(invalid("cache sparse indices must be ordered and unique"));
                }
                previous = Some(index);
                let position = state
                    .get_mut(index as usize)
                    .ok_or_else(|| invalid("cache particle index out of range"))?;
                *position = checked_sum(*position, delta)
                    .ok_or_else(|| invalid("cache delta reconstruction overflow"))?;
            }
        }
    }
    Ok(())
}

fn record_bytes(
    record: &FrameRecord,
    mut emit: impl FnMut(&[u8]) -> io::Result<()>,
) -> io::Result<()> {
    match record {
        FrameRecord::Key(positions) => {
            for xyz in positions {
                let mut bytes = [0u8; 12];
                for (chunk, value) in bytes.chunks_exact_mut(4).zip(xyz) {
                    chunk.copy_from_slice(&value.to_le_bytes());
                }
                emit(&bytes)?;
            }
        }
        FrameRecord::Delta(deltas) => {
            for (index, xyz) in deltas {
                let mut bytes = [0u8; 16];
                bytes[..4].copy_from_slice(&index.to_le_bytes());
                for (chunk, value) in bytes[4..].chunks_exact_mut(4).zip(xyz) {
                    chunk.copy_from_slice(&value.to_le_bytes());
                }
                emit(&bytes)?;
            }
        }
    }
    Ok(())
}

fn reserved<T>(count: u64) -> io::Result<Vec<T>> {
    let count = usize::try_from(count).map_err(|_| invalid("cache allocation count overflow"))?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| invalid("cache allocation failed"))?;
    Ok(values)
}

fn zero_bytes(count: u64) -> io::Result<Vec<u8>> {
    let mut bytes = reserved(count)?;
    bytes.resize(
        usize::try_from(count).map_err(|_| invalid("cache byte count overflow"))?,
        0,
    );
    Ok(bytes)
}

fn add(a: u64, b: u64) -> io::Result<u64> {
    a.checked_add(b)
        .ok_or_else(|| invalid("cache byte size/offset overflow"))
}

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn input(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}

fn get_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(
        bytes[offset..offset + 4]
            .try_into()
            .expect("validated fixed-size field"),
    )
}

fn get_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(
        bytes[offset..offset + 8]
            .try_into()
            .expect("validated fixed-size field"),
    )
}

fn read_xyz(bytes: &[u8], offset: usize) -> [i32; 3] {
    [
        get_u32(bytes, offset) as i32,
        get_u32(bytes, offset + 4) as i32,
        get_u32(bytes, offset + 8) as i32,
    ]
}

fn put_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}
fn put_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

struct Crc32(u32);

impl Crc32 {
    fn new() -> Self {
        Self(u32::MAX)
    }
    fn update(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 ^= u32::from(byte);
            for _ in 0..8 {
                self.0 = (self.0 >> 1) ^ (0xedb8_8320 & 0u32.wrapping_sub(self.0 & 1));
            }
        }
    }
    fn finish(self) -> u32 {
        !self.0
    }
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = Crc32::new();
    crc.update(bytes);
    crc.finish()
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod robustness;
