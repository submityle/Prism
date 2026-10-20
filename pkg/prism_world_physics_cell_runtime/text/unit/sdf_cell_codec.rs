//! Independent wire, corruption and admission checks for disposable SDF caches.

use prism_world_physics_bridge::{WorldSdfSnapshot, WorldSdfTransform};
use prism_world_physics_cell_runtime::{
    SdfCellCodec, SdfCellCodecError as Error, SdfCellCodecLimits as Limits,
    SdfCellCodecResource as Resource, SdfCellSnapshot, SDF_CELL_CODEC_VERSION,
};
use prism_world_residency::{BrickCoord, SdfError, SparseSdfBrickGrid};
use prism_world_stream::{AabbI64, CellManifest, DataLayerMask, GlobalPos, SectorId};

fn codec() -> SdfCellCodec {
    SdfCellCodec::new(Limits::new(1 << 20, 100, 1 << 18, 1 << 20))
}

fn sector() -> SectorId {
    SectorId::new(7, -8, i64::MIN, i64::MAX)
}

fn manifest() -> CellManifest {
    CellManifest {
        sector: sector(),
        bounds: AabbI64::new(
            GlobalPos::new(-100, -100, -100),
            GlobalPos::new(100, 100, 100),
        )
        .unwrap(),
        scene_key: 9,
        content_version: 1,
        derived_hash: 11,
        io_bytes: 0,
        data_layers: DataLayerMask::EMPTY,
    }
}

fn source(edge: u8, mips: u8, coords: &[BrickCoord]) -> SdfCellSnapshot {
    let mut grid = SparseSdfBrickGrid::new(edge, mips).unwrap();
    for (index, &coord) in coords.iter().enumerate() {
        let samples = (0..grid.sample_count())
            .map(|i| (i as i16).wrapping_add(index as i16 * 100))
            .collect();
        grid.insert_brick(coord, samples).unwrap();
    }
    SdfCellSnapshot::new(
        sector(),
        WorldSdfSnapshot::new(
            WorldSdfTransform::new([i64::MAX, i64::MIN, -17], 123, u32::MAX).unwrap(),
            u64::MAX,
            grid,
        ),
    )
}

// Bitwise reference is deliberately independent of the production lookup table.
fn crc(bytes: &[u8]) -> u32 {
    let mut value = u32::MAX;
    for byte in bytes {
        value ^= u32::from(*byte);
        for _ in 0..8 {
            value = (value >> 1) ^ if value & 1 != 0 { 0xedb8_8320 } else { 0 };
        }
    }
    !value
}

fn seal(bytes: &mut [u8]) {
    let end = bytes.len() - 4;
    let checksum = crc(&bytes[..end]);
    bytes[end..].copy_from_slice(&checksum.to_le_bytes());
}

/// A separately assembled 150-byte frame fixes all header offsets and signed ABI.
#[test]
fn exact_wire_oracle() {
    assert_eq!(crc(b"123456789"), 0xcbf4_3926);
    assert_eq!(SDF_CELL_CODEC_VERSION, 1);
    let original = source(1, 2, &[BrickCoord::new(-2, 3, -4, 1)]);
    let mut expected = vec![0; 150];
    expected[..4].copy_from_slice(b"PSDC");
    expected[4..6].copy_from_slice(&1u16.to_le_bytes());
    for (offset, value) in [
        (8, 128u32),
        (12, 16),
        (24, 1),
        (88, 123),
        (92, u32::MAX),
        (104, 1),
    ] {
        expected[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
    expected[16..24].copy_from_slice(&18u64.to_le_bytes());
    expected[28] = 1;
    expected[29] = 2;
    expected[32] = 7;
    for (offset, value) in [
        (40, -8i64),
        (48, i64::MIN),
        (56, i64::MAX),
        (64, i64::MAX),
        (72, i64::MIN),
        (80, -17),
    ] {
        expected[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }
    expected[96..104].copy_from_slice(&u64::MAX.to_le_bytes());
    for (offset, value) in [(128, -2i32), (132, 3), (136, -4)] {
        expected[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
    expected[140] = 1;
    seal(&mut expected);
    assert_eq!(codec().encode(&original).unwrap(), expected);
    assert_eq!(
        codec().decode_payload(&manifest(), &expected).unwrap(),
        original
    );
}

/// Insertion order cannot change the frame, and empty high-mip grids are legal.
#[test]
fn canonical_roundtrip_and_empty_grid() {
    let coords = [
        BrickCoord::new(i32::MAX, 0, i32::MIN, 2),
        BrickCoord::new(-1, 2, 3, 1),
        BrickCoord::new(-1, 2, 3, 0),
    ];
    let original = source(3, 3, &coords);
    let mut reversed = SparseSdfBrickGrid::new(3, 3).unwrap();
    for coord in coords.into_iter().rev() {
        reversed
            .insert_brick(
                coord,
                original
                    .snapshot()
                    .grid()
                    .brick_samples(coord)
                    .unwrap()
                    .unwrap()
                    .to_vec(),
            )
            .unwrap();
    }
    let other = SdfCellSnapshot::new(
        sector(),
        WorldSdfSnapshot::new(original.snapshot().transform(), u64::MAX, reversed),
    );
    let bytes = codec().encode(&original).unwrap();
    assert_eq!(codec().encode(&other).unwrap(), bytes);
    assert_eq!(
        codec().decode_payload(&manifest(), &bytes).unwrap(),
        original
    );
    let empty = source(32, 255, &[]);
    let exact = SdfCellCodec::new(Limits::new(132, 0, 0, 48));
    let bytes = exact.encode(&empty).unwrap();
    assert_eq!(bytes.len(), 132);
    assert_eq!(exact.decode_payload(&manifest(), &bytes).unwrap(), empty);
}

/// Every incomplete prefix and any trailing byte must fail deterministically.
#[test]
fn truncation_and_trailing_bytes() {
    let mut bytes = codec()
        .encode(&source(2, 1, &[BrickCoord::new(0, 0, 0, 0)]))
        .unwrap();
    for end in 0..bytes.len() {
        assert_eq!(
            codec().decode_payload(&manifest(), &bytes[..end]),
            Err(Error::Truncated),
            "prefix {end}"
        );
    }
    bytes.push(0);
    assert_eq!(
        codec().decode_payload(&manifest(), &bytes),
        Err(Error::TrailingBytes)
    );
}

/// Recomputed CRC cannot authorize reserved flags, impossible dimensions or mip.
#[test]
fn sealed_structural_mutations() {
    let bytes = codec()
        .encode(&source(2, 2, &[BrickCoord::new(0, 0, 0, 1)]))
        .unwrap();
    for offset in [6, 7, 8, 12, 30, 31, 33, 39, 104, 108, 127, 141, 142, 143] {
        let mut changed = bytes.clone();
        changed[offset] ^= 1;
        seal(&mut changed);
        assert!(
            matches!(
                codec().decode_payload(&manifest(), &changed),
                Err(Error::InvalidHeader { .. })
            ),
            "offset {offset}"
        );
    }
    let mut changed = bytes.clone();
    changed[0] ^= 1;
    assert_eq!(
        codec().decode_payload(&manifest(), &changed),
        Err(Error::InvalidMagic)
    );
    changed = bytes.clone();
    changed[4] = 2;
    assert_eq!(
        codec().decode_payload(&manifest(), &changed),
        Err(Error::UnsupportedVersion { found: 2 })
    );
    for edge in [0, 33, 255] {
        changed = bytes.clone();
        changed[28] = edge;
        seal(&mut changed);
        assert!(matches!(
            codec().decode_payload(&manifest(), &changed),
            Err(Error::Grid(_))
        ));
    }
    changed = bytes.clone();
    changed[29] = 0;
    seal(&mut changed);
    assert!(matches!(
        codec().decode_payload(&manifest(), &changed),
        Err(Error::Grid(_))
    ));
    for offset in [88, 92] {
        changed = bytes.clone();
        changed[offset..offset + 4].fill(0);
        seal(&mut changed);
        assert!(matches!(
            codec().decode_payload(&manifest(), &changed),
            Err(Error::Transform(_))
        ));
    }
    changed = bytes;
    changed[140] = 2;
    seal(&mut changed);
    assert_eq!(
        codec().decode_payload(&manifest(), &changed),
        Err(Error::Grid(SdfError::InvalidMip))
    );
}

/// Corruption of otherwise valid metadata, samples or CRC is detected.
#[test]
fn crc_covers_metadata_and_samples() {
    let bytes = codec()
        .encode(&source(2, 2, &[BrickCoord::new(0, 0, 0, 1)]))
        .unwrap();
    for offset in [
        32,
        40,
        48,
        56,
        64,
        72,
        80,
        88,
        92,
        96,
        128,
        132,
        136,
        144,
        bytes.len() - 1,
    ] {
        let mut changed = bytes.clone();
        changed[offset] ^= 1;
        assert_eq!(
            codec().decode_payload(&manifest(), &changed),
            Err(Error::ChecksumMismatch),
            "offset {offset}"
        );
    }
}

/// Manifest sector is authoritative, while its opaque derived hash is not a CRC.
#[test]
fn sector_identity_is_checked_after_checksum() {
    let original = source(1, 1, &[]);
    let bytes = codec().encode(&original).unwrap();
    for offset in [32, 40, 48, 56] {
        let mut changed = bytes.clone();
        changed[offset] ^= 1;
        seal(&mut changed);
        assert!(matches!(
            codec().decode_payload(&manifest(), &changed),
            Err(Error::ManifestSectorMismatch { .. })
        ));
    }
    let mut other = manifest();
    other.derived_hash += 1;
    assert_eq!(codec().decode_payload(&other, &bytes).unwrap(), original);
}

/// Full-key order rejects duplicate and descending late records before decoding.
#[test]
fn full_key_order_and_late_record_validation() {
    let bytes = codec()
        .encode(&source(
            1,
            3,
            &[BrickCoord::new(-1, 0, 0, 0), BrickCoord::new(-1, 0, 0, 1)],
        ))
        .unwrap();
    for mip in [0, 3] {
        let mut changed = bytes.clone();
        changed[158] = mip;
        seal(&mut changed);
        let error = codec().decode_payload(&manifest(), &changed).unwrap_err();
        if mip == 0 {
            assert!(matches!(error, Error::NonCanonicalBrickOrder { .. }));
        } else {
            assert_eq!(error, Error::Grid(SdfError::InvalidMip));
        }
    }
    let mut changed = bytes;
    changed[146..150].copy_from_slice(&(-2i32).to_le_bytes());
    seal(&mut changed);
    assert!(matches!(
        codec().decode_payload(&manifest(), &changed),
        Err(Error::NonCanonicalBrickOrder { .. })
    ));
}

/// Each independent exact bound admits, and one unit less rejects both directions.
#[test]
fn independent_resource_limits() {
    let original = source(2, 1, &[BrickCoord::new(0, 0, 0, 0)]);
    let bytes = codec().encode(&original).unwrap();
    assert_eq!(bytes.len(), 164);
    let exact = Limits::new(164, 1, 8, 80);
    assert_eq!(SdfCellCodec::new(exact).limits(), exact);
    assert_eq!(
        SdfCellCodec::new(exact)
            .decode_payload(&manifest(), &bytes)
            .unwrap(),
        original
    );
    for (limits, resource, required, limit) in [
        (Limits::new(163, 1, 8, 80), Resource::PayloadBytes, 164, 163),
        (Limits::new(164, 0, 8, 80), Resource::Bricks, 1, 0),
        (Limits::new(164, 1, 7, 80), Resource::Samples, 8, 7),
        (Limits::new(164, 1, 8, 79), Resource::DecodedBytes, 80, 79),
    ] {
        let restricted = SdfCellCodec::new(limits);
        let expected = Error::LimitExceeded {
            resource,
            required,
            limit,
        };
        assert_eq!(restricted.encode(&original), Err(expected.clone()));
        assert_eq!(
            restricted.decode_payload(&manifest(), &bytes),
            Err(expected)
        );
    }
}

/// Forged counts and body sizes cannot trigger unbounded sample construction.
#[test]
fn forged_lengths_fail_admission() {
    let bytes = codec().encode(&source(1, 1, &[])).unwrap();
    for body in [1, u64::MAX] {
        let mut changed = bytes.clone();
        changed[16..24].copy_from_slice(&body.to_le_bytes());
        seal(&mut changed);
        assert!(matches!(
            codec().decode_payload(&manifest(), &changed),
            Err(Error::BodyLengthMismatch | Error::SizeOverflow)
        ));
    }
    let mut changed = bytes;
    changed[24..28].copy_from_slice(&u32::MAX.to_le_bytes());
    seal(&mut changed);
    assert!(matches!(
        codec().decode_payload(&manifest(), &changed),
        Err(Error::LimitExceeded {
            resource: Resource::Bricks,
            ..
        })
    ));
}

/// Maximum brick volume preserves signed extremes; dirty invalidations are refused.
#[test]
fn signed_extremes_and_dirty_grid() {
    let mut grid = SparseSdfBrickGrid::new(32, 1).unwrap();
    grid.insert_brick(
        BrickCoord::new(0, 0, 0, 0),
        (0..32768)
            .map(|i| if i % 2 == 0 { i16::MIN } else { i16::MAX })
            .collect(),
    )
    .unwrap();
    let transform = WorldSdfTransform::new([0; 3], 1, 1).unwrap();
    let original =
        SdfCellSnapshot::new(sector(), WorldSdfSnapshot::new(transform, 0, grid.clone()));
    let bytes = codec().encode(&original).unwrap();
    assert_eq!(
        codec().decode_payload(&manifest(), &bytes).unwrap(),
        original
    );
    grid.mark_dirty_region([32; 3], [32; 3], 0).unwrap();
    let dirty = SdfCellSnapshot::new(sector(), WorldSdfSnapshot::new(transform, 0, grid));
    assert_eq!(codec().encode(&dirty), Err(Error::DirtyGrid));
}

/// Every single-bit mutation is either rejected or remains an exact canonical frame.
#[test]
fn exhaustive_single_bit_mutations_are_canonical_or_rejected() {
    let bytes = codec()
        .encode(&source(
            3,
            3,
            &[
                BrickCoord::new(-2, 4, 6, 0),
                BrickCoord::new(-2, 4, 6, 1),
                BrickCoord::new(7, -3, 2, 2),
            ],
        ))
        .unwrap();
    for offset in 0..bytes.len() {
        for bit in 0..8 {
            let mut changed = bytes.clone();
            changed[offset] ^= 1 << bit;
            assert!(
                codec().decode_payload(&manifest(), &changed).is_err(),
                "unsealed bit at {offset}:{bit}"
            );
            seal(&mut changed);
            if let Ok(decoded) = codec().decode_payload(&manifest(), &changed) {
                assert_eq!(
                    codec().encode(&decoded).unwrap(),
                    changed,
                    "sealed bit at {offset}:{bit}"
                );
            }
        }
    }
}
