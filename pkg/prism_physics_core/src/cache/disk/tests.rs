use std::io::{self, Cursor, ErrorKind, Write};

use super::*;

fn sample_cache() -> PhysicsCache {
    let mut track = CacheTrack::new(2, 3);
    assert!(track.push_frame(&[[10, -5, 3], [0, 1, 2]]));
    assert!(track.push_frame(&[[11, -5, 3], [0, 3, 2]]));
    assert!(track.push_frame(&[[11, -5, 4], [0, 3, 2]]));
    assert!(track.push_frame(&[[12, -5, 4], [0, 3, 2]]));
    let mut cache = PhysicsCache::new(1.0 / 60.0, PositionQuantizer::new(1.0e-3), 3);
    cache.push_track(track);
    cache
}

fn encoded(cache: &PhysicsCache) -> Vec<u8> {
    let mut bytes = Vec::new();
    cache.write_to(&mut bytes).expect("cache writes");
    bytes
}

fn refresh_header_crc(bytes: &mut [u8]) {
    let crc = crc32(&bytes[..60]);
    bytes[60..64].copy_from_slice(&crc.to_le_bytes());
}

fn refresh_index_crc(bytes: &mut [u8]) {
    let index_len = get_u64(bytes, 48) as usize;
    let crc = crc32(&bytes[64..64 + index_len]);
    bytes[56..60].copy_from_slice(&crc.to_le_bytes());
    refresh_header_crc(bytes);
}

fn error_kind(result: io::Result<impl Sized>) -> ErrorKind {
    result.err().expect("operation should fail").kind()
}

struct ShortWriter {
    bytes: Vec<u8>,
    max_write: usize,
}

impl Write for ShortWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let count = bytes.len().min(self.max_write);
        self.bytes.extend_from_slice(&bytes[..count]);
        Ok(count)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

struct ZeroWriter;

impl Write for ZeroWriter {
    fn write(&mut self, _bytes: &[u8]) -> io::Result<usize> {
        Ok(0)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[test]
fn round_trip_load_and_random_access_preserve_soft_cache() {
    let cache = sample_cache();
    let bytes = encoded(&cache);

    let mut reader = CacheReader::open(Cursor::new(bytes.clone())).expect("valid index opens");
    assert_eq!(reader.metadata().frame_count, 4);
    assert_eq!(
        reader.tracks(),
        &[CacheTrackInfo {
            particle_count: 2,
            frame_count: 4,
            keyframe_interval: 3,
        }]
    );
    assert_eq!(
        reader.read_quantized_frame(0, 2).expect("random frame"),
        vec![[11, -5, 4], [0, 3, 2]]
    );
    let loaded = CacheReader::open(Cursor::new(bytes))
        .expect("valid index opens")
        .load()
        .expect("fully loads");
    assert_eq!(loaded, cache);
    assert_eq!(
        PhysicsCache::read_from(Cursor::new(encoded(&cache))).expect("convenience load"),
        cache
    );
}

#[test]
fn empty_cache_and_empty_track_round_trip_without_payload() {
    let empty = PhysicsCache::new(1.0 / 60.0, PositionQuantizer::default(), 4);
    let bytes = encoded(&empty);
    assert_eq!(bytes.len(), 64);
    assert_eq!(
        PhysicsCache::read_from(Cursor::new(bytes)).expect("empty cache"),
        empty
    );

    let mut with_empty_track = PhysicsCache::new(1.0 / 60.0, PositionQuantizer::default(), 4);
    with_empty_track.push_track(CacheTrack::new(0, 4));
    let bytes = encoded(&with_empty_track);
    let reader = CacheReader::open(Cursor::new(bytes)).expect("empty track");
    assert_eq!(reader.metadata().frame_count, 0);
    assert_eq!(reader.tracks()[0].particle_count, 0);
    assert_eq!(reader.load().expect("load empty track"), with_empty_track);
}

#[test]
fn reader_respects_nonzero_stream_origin_and_allows_trailing_bytes() {
    let asset = encoded(&sample_cache());
    let mut stream = vec![0xa5; 11];
    stream.extend_from_slice(&asset);
    stream.extend_from_slice(&[0x5a; 7]);

    let mut source = Cursor::new(stream);
    source.set_position(11);
    let mut reader = CacheReader::open(source).expect("embedded asset");
    assert_eq!(
        reader.read_quantized_frame(0, 3).expect("embedded frame")[0],
        [12, -5, 4]
    );
    let source = reader.into_inner();
    assert_eq!(source.get_ref().len(), 11 + asset.len() + 7);
}

#[test]
fn writer_handles_short_writes_and_reports_write_zero() {
    let cache = sample_cache();
    let expected = encoded(&cache);
    let mut writer = ShortWriter {
        bytes: Vec::new(),
        max_write: 3,
    };
    cache
        .write_to(&mut writer)
        .expect("write_all retries short writes");
    assert_eq!(writer.bytes, expected);

    assert_eq!(
        cache.write_to(ZeroWriter).unwrap_err().kind(),
        ErrorKind::WriteZero
    );
}

#[test]
fn header_index_and_payload_corruption_are_rejected() {
    let cache = sample_cache();
    let bytes = encoded(&cache);

    let mut bad_header = bytes.clone();
    bad_header[24] ^= 1;
    assert_eq!(
        error_kind(CacheReader::open(Cursor::new(bad_header))),
        ErrorKind::InvalidData
    );

    let mut bad_index = bytes.clone();
    bad_index[64] ^= 1;
    assert_eq!(
        error_kind(CacheReader::open(Cursor::new(bad_index))),
        ErrorKind::InvalidData
    );

    let mut bad_payload = bytes;
    let last = bad_payload.len() - 1;
    bad_payload[last] ^= 1;
    let mut reader =
        CacheReader::open(Cursor::new(bad_payload)).expect("payload checks are deferred");
    assert_eq!(
        reader.read_frame(0, 3).unwrap_err().kind(),
        ErrorKind::InvalidData
    );
}

#[test]
fn truncation_never_allocates_or_accepts_a_partial_asset() {
    let bytes = encoded(&sample_cache());
    for length in [0, 1, 63, 64, bytes.len() - 1] {
        assert_eq!(
            error_kind(CacheReader::open(Cursor::new(bytes[..length].to_vec()))),
            ErrorKind::InvalidData,
            "truncation at {length} bytes"
        );
    }
}

#[test]
fn hostile_header_counts_fail_before_count_based_allocation() {
    let mut bytes = encoded(&sample_cache());
    bytes[36..40].copy_from_slice(&u32::MAX.to_le_bytes());
    refresh_header_crc(&mut bytes);
    let limits = CacheLimits {
        max_tracks: 8,
        ..CacheLimits::default()
    };
    assert_eq!(
        error_kind(CacheReader::with_limits(Cursor::new(bytes), limits)),
        ErrorKind::InvalidData
    );
}

#[test]
fn sparse_delta_overflow_becomes_a_keyframe_when_baking() {
    let mut track = CacheTrack::new(1, 8);
    assert!(track.push_frame(&[[i32::MIN, 0, 0]]));
    assert!(track.push_frame(&[[i32::MAX, 0, 0]]));
    assert!(matches!(track.frames[1], FrameRecord::Key(_)));
    assert!(matches!(track.reconstruct(1), Some(ref values) if values == &vec![[i32::MAX, 0, 0]]));
}

#[test]
fn hostile_sparse_delta_overflow_is_rejected_during_stream_reconstruction() {
    let mut track = CacheTrack::new(1, 4);
    assert!(track.push_frame(&[[i32::MAX - 1, 0, 0]]));
    assert!(track.push_frame(&[[i32::MAX, 0, 0]]));
    let mut cache = PhysicsCache::new(1.0 / 60.0, PositionQuantizer::default(), 4);
    cache.push_track(track);
    let mut bytes = encoded(&cache);

    // header (64), one track descriptor (16), two 32-byte frame entries;
    // key payload (12), then delta payload: u32 particle index + xyz i32.
    let second_entry = 64 + 16 + 32;
    let delta_payload = 64 + 16 + 64 + 12;
    bytes[delta_payload + 4..delta_payload + 8].copy_from_slice(&2_i32.to_le_bytes());
    let payload_crc = crc32(&bytes[delta_payload..delta_payload + 16]);
    bytes[second_entry + 24..second_entry + 28].copy_from_slice(&payload_crc.to_le_bytes());
    refresh_index_crc(&mut bytes);

    let mut reader =
        CacheReader::open(Cursor::new(bytes)).expect("metadata and frame CRCs are valid");
    assert_eq!(
        reader.read_quantized_frame(0, 1).unwrap_err().kind(),
        ErrorKind::InvalidData
    );
}
