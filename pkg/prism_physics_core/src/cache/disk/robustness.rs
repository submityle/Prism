//! Semantic corruption uses freshly repaired checksums, so rejection cannot
//! accidentally be attributed to CRC rather than the boundary under test.

use super::*;
use std::cell::RefCell;
use std::io::{Cursor, ErrorKind};
use std::rc::Rc;

fn cache() -> PhysicsCache {
    let mut cache = PhysicsCache::new(0.25, PositionQuantizer::new(0.5), 3);
    for base in [10, 100] {
        let mut track = CacheTrack::new(2, 3);
        for positions in [
            [[base, 20, 30], [-base, 0, 2]],
            [[base + 1, 20, 30], [-base, 1, 2]],
            [[base + 1, 20, 31], [-base, 1, 2]],
            [[base + 2, 20, 31], [-base, 1, 2]],
            [[base + 3, 20, 31], [-base, 2, 2]],
            [[base + 3, 20, 31], [-base, 2, 2]],
        ] {
            assert!(track.push_frame(&positions));
        }
        cache.push_track(track);
    }
    cache
}

fn encode(cache: &PhysicsCache) -> Vec<u8> {
    let mut bytes = Vec::new();
    cache.write_to(&mut bytes).unwrap();
    bytes
}

fn set32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn set64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn header_crc(bytes: &mut [u8]) {
    let crc = crc32(&bytes[..60]);
    set32(bytes, 60, crc);
}

fn index_crc(bytes: &mut [u8]) {
    let end = HEADER_SIZE as usize + get_u64(bytes, 48) as usize;
    let crc = crc32(&bytes[HEADER_SIZE as usize..end]);
    set32(bytes, 56, crc);
    header_crc(bytes);
}

fn entry_at(track: usize, frame: u32) -> usize {
    HEADER_SIZE as usize
        + track * (TRACK_SIZE + 6 * ENTRY_SIZE) as usize
        + TRACK_SIZE as usize
        + frame as usize * ENTRY_SIZE as usize
}

fn payload_range(bytes: &[u8], entry: usize) -> (u64, usize) {
    (get_u64(bytes, entry), get_u64(bytes, entry + 8) as usize)
}

fn payload_crc(bytes: &mut [u8], entry: usize) {
    let (offset, len) = payload_range(bytes, entry);
    let crc = crc32(&bytes[offset as usize..offset as usize + len]);
    set32(bytes, entry + 24, crc);
    index_crc(bytes);
}

fn error(result: io::Result<impl Sized>, kind: ErrorKind) -> io::Error {
    let error = result.err().expect("must reject");
    assert_eq!(error.kind(), kind);
    error
}

#[derive(Default)]
struct Trace {
    reads: Vec<(u64, usize)>,
    seeks: usize,
}

// The stream is immutable; faults are transient I/O failures, not asset edits.
// Logging successful read ranges proves precisely which payloads were visited.
struct Probe {
    source: Cursor<Vec<u8>>,
    trace: Rc<RefCell<Trace>>,
    chunk: usize,
    fail_read: Option<(u64, Option<ErrorKind>)>,
    fail_seek: Option<u64>,
}

impl Probe {
    fn new(bytes: Vec<u8>) -> (Self, Rc<RefCell<Trace>>) {
        let trace = Rc::new(RefCell::new(Trace::default()));
        (
            Self {
                source: Cursor::new(bytes),
                trace: trace.clone(),
                chunk: usize::MAX,
                fail_read: None,
                fail_seek: None,
            },
            trace,
        )
    }
}

impl Read for Probe {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        let position = self.source.position();
        let mut len = output.len().min(self.chunk);
        if let Some((at, kind)) = self.fail_read {
            if position == at {
                self.fail_read = None;
                return match kind {
                    Some(kind) => Err(io::Error::from(kind)),
                    None => Ok(0),
                };
            }
            if position < at {
                len = len.min(usize::try_from(at - position).unwrap_or(usize::MAX));
            }
        }
        let read = self.source.read(&mut output[..len])?;
        if read > 0 {
            self.trace.borrow_mut().reads.push((position, read));
        }
        Ok(read)
    }
}

impl Seek for Probe {
    fn seek(&mut self, from: SeekFrom) -> io::Result<u64> {
        self.trace.borrow_mut().seeks += 1;
        if let SeekFrom::Start(offset) = from {
            if self.fail_seek == Some(offset) {
                self.fail_seek = None;
                return Err(io::Error::from(ErrorKind::PermissionDenied));
            }
        }
        self.source.seek(from)
    }
}

#[test]
fn crc_matches_ieee_vector_and_is_independent_of_chunking() {
    assert_eq!(crc32(b""), 0);
    assert_eq!(crc32(b"123456789"), 0xcbf4_3926);
    let bytes = encode(&cache());
    for chunk in [1, 3, 64, 127] {
        let mut crc = Crc32::new();
        for bytes in bytes.chunks(chunk) {
            crc.update(bytes);
        }
        assert_eq!(crc.finish(), crc32(&bytes));
    }
}

#[test]
fn checksummed_invalid_header_fields_are_rejected_before_index_reads() {
    let bytes = encode(&cache());
    let mut mutations = vec![
        (8, CACHE_FORMAT_VERSION + 1),
        (12, HEADER_SIZE as u32 - 1),
        (32, 0),
        (36, 0), // Nonzero frames without any tracks.
        (44, 1),
    ];
    for offset in [24, 28] {
        for value in [
            0.0_f32,
            -0.0,
            -1.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ] {
            mutations.push((offset, value.to_bits()));
        }
    }
    for (offset, value) in mutations {
        let mut bad = bytes.clone();
        set32(&mut bad, offset, value);
        header_crc(&mut bad);
        let (source, trace) = Probe::new(bad);
        error(CacheReader::open(source), ErrorKind::InvalidData);
        assert_eq!(
            trace.borrow().reads,
            [(0, HEADER_SIZE as usize)],
            "field {offset}"
        );
    }
    for (offset, value) in [
        (16, HEADER_SIZE - 1),
        (16, bytes.len() as u64 + 1),
        (48, get_u64(&bytes, 48) + ENTRY_SIZE),
    ] {
        let mut bad = bytes.clone();
        set64(&mut bad, offset, value);
        header_crc(&mut bad);
        let (source, trace) = Probe::new(bad);
        error(CacheReader::open(source), ErrorKind::InvalidData);
        assert_eq!(trace.borrow().reads, [(0, HEADER_SIZE as usize)]);
    }
}

#[test]
fn index_size_overflow_and_budget_fail_before_index_allocation_or_read() {
    let bytes = encode(&cache());
    for frames in [1, u32::MAX] {
        let mut bad = bytes.clone();
        set32(&mut bad, 36, u32::MAX);
        set32(&mut bad, 40, frames);
        header_crc(&mut bad);
        let (source, trace) = Probe::new(bad);
        let limits = CacheLimits {
            max_tracks: u32::MAX,
            max_frames: u32::MAX,
            max_index_bytes: if frames == 1 { 1024 } else { u64::MAX },
            ..CacheLimits::default()
        };
        let rejected = error(
            CacheReader::with_limits(source, limits),
            ErrorKind::InvalidData,
        );
        assert_eq!(
            rejected.to_string(),
            if frames == 1 {
                "cache index exceeds limit"
            } else {
                "cache index size overflow"
            }
        );
        assert_eq!(trace.borrow().reads, [(0, HEADER_SIZE as usize)]);
    }
}

#[test]
fn checksummed_index_rejects_inconsistent_geometry_and_payload_layout() {
    let bytes = encode(&cache());
    let first = entry_at(0, 0);
    let second = entry_at(0, 1);
    let mut cases = Vec::new();
    for (offset, value) in [
        (64 + 4, 5), // Track frame count differs from header.
        (64 + 8, 0),
        (64 + 8, 4),
        (64 + 12, 1),
        (first + 16, 1), // A key must contain every particle.
        (first + 20, 1), // Frame zero must be a key.
        (first + 20, 2),
        (first + 28, 1),
        (second + 16, 3),         // Too many sparse entries.
        (entry_at(0, 3) + 20, 1), // Scheduled keys cannot become deltas.
    ] {
        let mut bad = bytes.clone();
        set32(&mut bad, offset, value);
        cases.push(bad);
    }
    for (offset, value) in [
        (first, get_u64(&bytes, first) - 1), // Index alias.
        (first, get_u64(&bytes, first) + 1), // Gap.
        (first, u64::MAX),
        (second, get_u64(&bytes, first)), // Previous frame alias.
        (second + 8, 31),                 // Sparse stride mismatch.
        (second + 8, u64::MAX),
    ] {
        let mut bad = bytes.clone();
        set64(&mut bad, offset, value);
        cases.push(bad);
    }
    let mut suffix = bytes.clone();
    suffix.push(0);
    let len = suffix.len() as u64;
    set64(&mut suffix, 16, len);
    cases.push(suffix); // Bytes inside asset must all be indexed.

    for (case, mut bad) in cases.into_iter().enumerate() {
        index_crc(&mut bad);
        let (source, trace) = Probe::new(bad);
        error(CacheReader::open(source), ErrorKind::InvalidData);
        assert_eq!(
            trace.borrow().reads,
            [
                (0, HEADER_SIZE as usize),
                (HEADER_SIZE, get_u64(&bytes, 48) as usize),
            ],
            "case {case} must not read payloads"
        );
    }
}

#[test]
fn all_encoded_resource_limits_reject_at_their_metadata_boundary() {
    let bytes = encode(&cache());
    let defaults = CacheLimits::default();
    let index_len = get_u64(&bytes, 48);
    for limits in [
        CacheLimits {
            max_asset_bytes: bytes.len() as u64 - 1,
            ..defaults
        },
        CacheLimits {
            max_index_bytes: index_len - 1,
            ..defaults
        },
        CacheLimits {
            max_tracks: 1,
            ..defaults
        },
        CacheLimits {
            max_frames: 5,
            ..defaults
        },
    ] {
        let (source, trace) = Probe::new(bytes.clone());
        error(
            CacheReader::with_limits(source, limits),
            ErrorKind::InvalidData,
        );
        assert_eq!(trace.borrow().reads, [(0, HEADER_SIZE as usize)]);
    }
    for limits in [
        CacheLimits {
            max_particles: 1,
            ..defaults
        },
        CacheLimits {
            max_frame_bytes: 31,
            ..defaults
        },
    ] {
        let (source, trace) = Probe::new(bytes.clone());
        error(
            CacheReader::with_limits(source, limits),
            ErrorKind::InvalidData,
        );
        assert_eq!(
            trace.borrow().reads,
            [(0, HEADER_SIZE as usize), (HEADER_SIZE, index_len as usize)]
        );
    }
    let exact = CacheLimits {
        max_asset_bytes: bytes.len() as u64,
        max_index_bytes: index_len,
        max_tracks: 2,
        max_frames: 6,
        max_particles: 2,
        max_frame_bytes: 32,
        ..defaults
    };
    assert_eq!(
        CacheReader::with_limits(Cursor::new(bytes), exact)
            .unwrap()
            .load()
            .unwrap(),
        cache()
    );
    let mut output = Vec::new();
    cache().write_to_with_limits(&mut output, exact).unwrap();
    assert_eq!(output, encode(&cache()));
}

#[test]
fn decoded_budgets_are_preflighted_before_any_payload_io() {
    let cache = cache();
    let bytes = encode(&cache);
    let inspect = CacheReader::open(Cursor::new(bytes.clone())).unwrap();
    let track_size = inspect.decoded_size(0).unwrap();
    let total_size = size_of::<PhysicsCache>() as u64
        + inspect.decoded_size(0).unwrap()
        + inspect.decoded_size(1).unwrap();
    for (budget, full_load) in [(track_size - 1, false), (total_size - 1, true)] {
        let (source, trace) = Probe::new(bytes.clone());
        let mut reader = CacheReader::with_limits(
            source,
            CacheLimits {
                max_decoded_bytes: budget,
                ..CacheLimits::default()
            },
        )
        .unwrap();
        *trace.borrow_mut() = Trace::default();
        if full_load {
            error(reader.load(), ErrorKind::InvalidData);
        } else {
            error(reader.read_track(0), ErrorKind::InvalidData);
            assert!(trace.borrow().reads.is_empty());
            assert_eq!(trace.borrow().seeks, 0);
            // The decoded-track ceiling deliberately does not forbid streaming.
            assert_eq!(
                reader.read_quantized_frame(1, 4).unwrap(),
                cache.track(1).unwrap().reconstruct(4).unwrap()
            );
            continue;
        }
        assert!(trace.borrow().reads.is_empty());
        assert_eq!(trace.borrow().seeks, 0);
    }
    let mut reader = CacheReader::with_limits(
        Cursor::new(bytes.clone()),
        CacheLimits {
            max_decoded_bytes: track_size,
            ..CacheLimits::default()
        },
    )
    .unwrap();
    assert_eq!(&reader.read_track(0).unwrap(), cache.track(0).unwrap());
    assert_eq!(
        CacheReader::with_limits(
            Cursor::new(bytes),
            CacheLimits {
                max_decoded_bytes: total_size,
                ..CacheLimits::default()
            }
        )
        .unwrap()
        .load()
        .unwrap(),
        cache
    );
}

#[test]
fn writer_layout_budget_precedes_payload_reconstruction_and_preserves_output() {
    let mut cache = cache();
    let bytes = encode(&cache);
    // This early delta would overflow during reconstruction. A late byte-budget
    // failure must win, proving payload traversal has not begun in preflight.
    let mut malformed = cache.track(0).unwrap().clone();
    let FrameRecord::Delta(deltas) = &mut malformed.frames[1] else {
        unreachable!()
    };
    deltas[0].1[0] = i32::MAX;
    let second = cache.track(1).unwrap().clone();
    cache = PhysicsCache::new(0.25, PositionQuantizer::new(0.5), 3);
    cache.push_track(malformed);
    cache.push_track(second);
    let defaults = CacheLimits::default();
    let mut limits = vec![
        CacheLimits {
            max_index_bytes: get_u64(&bytes, 48) - 1,
            ..defaults
        },
        CacheLimits {
            max_frame_bytes: 31,
            ..defaults
        },
        CacheLimits {
            max_particles: 1,
            ..defaults
        },
        CacheLimits {
            max_frames: 5,
            ..defaults
        },
        CacheLimits {
            max_tracks: 1,
            ..defaults
        },
    ];
    for max_asset_bytes in [
        HEADER_SIZE,
        HEADER_SIZE + get_u64(&bytes, 48),
        bytes.len() as u64 - 1,
    ] {
        limits.push(CacheLimits {
            max_asset_bytes,
            ..defaults
        });
    }
    for limits in limits {
        let mut output = vec![0xa5, 0x5a];
        let rejected = error(
            cache.write_to_with_limits(&mut output, limits),
            ErrorKind::InvalidData,
        );
        assert!(rejected.to_string().contains("limit"));
        assert!(!rejected.to_string().contains("reconstruction"));
        assert_eq!(output, [0xa5, 0x5a]);
    }
    // With sufficient layout budget, malformed payload validation still occurs
    // before writing the header, not after an externally visible partial write.
    let mut output = vec![0xa5, 0x5a];
    let rejected = error(cache.write_to(&mut output), ErrorKind::InvalidData);
    assert_eq!(rejected.to_string(), "cache delta reconstruction overflow");
    assert_eq!(output, [0xa5, 0x5a]);

    // Preflight reaches metadata in later tracks before reconstructing the
    // malformed first track, rather than validating only the current track.
    let mut invalid_late_track = cache.track(1).unwrap().clone();
    invalid_late_track.keyframe_interval = 4;
    let mut mismatched = PhysicsCache::new(0.25, PositionQuantizer::new(0.5), 3);
    mismatched.push_track(cache.track(0).unwrap().clone());
    mismatched.push_track(invalid_late_track);
    let rejected = error(mismatched.write_to(&mut output), ErrorKind::InvalidData);
    assert_eq!(
        rejected.to_string(),
        "cache tracks must share frame count and interval within limits"
    );
    assert_eq!(output, [0xa5, 0x5a]);
}

#[test]
fn invalid_requests_do_no_io_and_do_not_poison_later_random_reads() {
    let cache = cache();
    let (source, trace) = Probe::new(encode(&cache));
    let mut reader = CacheReader::open(source).unwrap();
    let metadata = reader.metadata();
    let tracks = reader.tracks().to_vec();
    *trace.borrow_mut() = Trace::default();
    error(reader.read_frame(usize::MAX, 0), ErrorKind::InvalidInput);
    error(
        reader.read_quantized_frame(0, u32::MAX),
        ErrorKind::InvalidInput,
    );
    error(reader.read_positions(2, 0), ErrorKind::InvalidInput);
    error(reader.read_track(2), ErrorKind::InvalidInput);
    assert!(trace.borrow().reads.is_empty());
    assert_eq!(trace.borrow().seeks, 0);
    assert_eq!(reader.metadata(), metadata);
    assert_eq!(reader.tracks(), tracks);
    for frame in [5, 0, 4, 1, 3, 2, 5] {
        for track in [1, 0] {
            assert_eq!(
                reader.read_quantized_frame(track, frame).unwrap(),
                cache.track(track).unwrap().reconstruct(frame).unwrap()
            );
        }
    }
}

#[test]
fn crc_failure_does_not_touch_unrelated_tracks_or_preceding_key_intervals() {
    let cache = cache();
    let mut bytes = encode(&cache);
    let (bad_offset, _) = payload_range(&bytes, entry_at(0, 1));
    bytes[bad_offset as usize + 4] ^= 1;
    let expected_reads = [
        payload_range(&bytes, entry_at(0, 3)),
        payload_range(&bytes, entry_at(0, 4)),
    ];
    let (source, trace) = Probe::new(bytes);
    let mut reader = CacheReader::open(source).unwrap();
    error(reader.read_quantized_frame(0, 2), ErrorKind::InvalidData);
    error(reader.read_track(0), ErrorKind::InvalidData);
    *trace.borrow_mut() = Trace::default();
    assert_eq!(
        reader.read_quantized_frame(0, 4).unwrap(),
        cache.track(0).unwrap().reconstruct(4).unwrap()
    );
    assert_eq!(trace.borrow().reads, expected_reads);
    assert_eq!(reader.read_track(1).unwrap(), *cache.track(1).unwrap());
    // Retrying the damaged interval must fail again, not reuse partial state.
    error(reader.read_quantized_frame(0, 2), ErrorKind::InvalidData);
    assert_eq!(
        reader.read_quantized_frame(0, 0).unwrap(),
        cache.track(0).unwrap().reconstruct(0).unwrap()
    );
}

#[test]
fn checksummed_sparse_order_range_and_arithmetic_failures_are_isolated() {
    let cache = cache();
    let bytes = encode(&cache);
    let entry = entry_at(0, 1);
    let (offset, len) = payload_range(&bytes, entry);
    assert_eq!(len, 32);
    for case in 0..4 {
        let mut bad = bytes.clone();
        let offset = offset as usize;
        match case {
            0 => set32(&mut bad, offset + 16, 0), // Duplicate.
            1 => {
                set32(&mut bad, offset, 1);
                set32(&mut bad, offset + 16, 0);
            }
            2 => set32(&mut bad, offset + 16, 2), // Out of range.
            // The first delta succeeds before the second delta underflows.
            _ => set32(&mut bad, offset + 20, i32::MIN as u32),
        }
        payload_crc(&mut bad, entry);
        let mut reader = CacheReader::open(Cursor::new(bad)).unwrap();
        if case == 3 {
            assert!(reader.read_frame(0, 1).is_ok());
        } else {
            error(reader.read_frame(0, 1), ErrorKind::InvalidData);
        }
        error(reader.read_quantized_frame(0, 2), ErrorKind::InvalidData);
        error(reader.read_track(0), ErrorKind::InvalidData);
        for (track, frame) in [(0, 0), (0, 5), (1, 2)] {
            assert_eq!(
                reader.read_quantized_frame(track, frame).unwrap(),
                cache.track(track).unwrap().reconstruct(frame).unwrap()
            );
        }
    }
}

#[test]
fn transient_seek_short_read_and_eof_failures_allow_exact_retry() {
    let cache = cache();
    let bytes = encode(&cache);
    let (offset, _) = payload_range(&bytes, entry_at(0, 1));
    for case in 0..4 {
        let (mut source, _) = Probe::new(bytes.clone());
        source.chunk = 3; // Header, index and payloads all need read_exact retry.
        let mut reader = CacheReader::open(source).unwrap();
        match case {
            0 => reader.source.fail_seek = Some(offset),
            1 => reader.source.fail_read = Some((offset + 3, Some(ErrorKind::ConnectionReset))),
            2 => reader.source.fail_read = Some((offset + 3, None)),
            _ => reader.source.fail_read = Some((offset + 3, Some(ErrorKind::Interrupted))),
        }
        if case == 3 {
            assert_eq!(
                reader.read_quantized_frame(0, 2).unwrap(),
                cache.track(0).unwrap().reconstruct(2).unwrap()
            );
        } else {
            error(
                reader.read_quantized_frame(0, 2),
                match case {
                    0 => ErrorKind::PermissionDenied,
                    1 => ErrorKind::ConnectionReset,
                    _ => ErrorKind::UnexpectedEof,
                },
            );
        }
        assert_eq!(
            reader.read_quantized_frame(0, 2).unwrap(),
            cache.track(0).unwrap().reconstruct(2).unwrap()
        );
        assert_eq!(reader.read_track(0).unwrap(), *cache.track(0).unwrap());
    }
}

#[test]
fn random_access_starts_at_extra_overflow_key_not_damaged_scheduled_key() {
    let mut track = CacheTrack::new(1, 8);
    for value in [i32::MIN, i32::MAX, i32::MAX - 1] {
        assert!(track.push_frame(&[[value, 0, 0]]));
    }
    let mut cache = PhysicsCache::new(0.25, PositionQuantizer::default(), 8);
    cache.push_track(track);
    let mut bytes = encode(&cache);
    let first = (HEADER_SIZE + TRACK_SIZE) as usize;
    let extra_key = first + ENTRY_SIZE as usize;
    let delta = extra_key + ENTRY_SIZE as usize;
    let (offset, _) = payload_range(&bytes, first);
    bytes[offset as usize] ^= 1;
    let expected_reads = [
        payload_range(&bytes, extra_key),
        payload_range(&bytes, delta),
    ];
    let (source, trace) = Probe::new(bytes);
    let mut reader = CacheReader::open(source).unwrap();
    *trace.borrow_mut() = Trace::default();
    assert_eq!(
        reader.read_quantized_frame(0, 2).unwrap(),
        [[i32::MAX - 1, 0, 0]]
    );
    assert_eq!(trace.borrow().reads, expected_reads);
    error(reader.read_quantized_frame(0, 0), ErrorKind::InvalidData);
    assert_eq!(
        reader.read_quantized_frame(0, 1).unwrap(),
        [[i32::MAX, 0, 0]]
    );
}
