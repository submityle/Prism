//! Placement persistence contracts, using real baked terrain for regeneration.

use alloc::{sync::Arc, vec, vec::Vec};
use core::{convert::Infallible, mem::size_of};

use prism_world_persist::{
    decode_snapshot, encode_snapshot, ActorId, CrdtState, FieldId, LogicalClock, OverrideDelta,
    OverrideValue, PersistentId, Timeline, Transaction, TransactionId,
};
#[cfg(feature = "std")]
use prism_world_persist::{Snapshot, WorldState};
use prism_world_residency::{
    DensityBrush, DensityBrushOp, ScatterCandidateKey, ScatterDensityField, ScatterError,
    ScatterFieldBuildBudget, ScatterFieldConfig, SectorCoord, SCATTER_KERNEL_VERSION,
};
use prism_world_terrain::{
    scatter_on_heightfield, BakeLimits, CubeFace, HeightSample, HeightSampleValue, HeightSampler,
    HeightfieldPlacementBudget, HeightfieldPlacementConfig, HeightfieldPlacementDeletionBudget,
    HeightfieldPlacementDeletionError, HeightfieldPlacementDeletionSet, HeightfieldPlacementError,
    HeightfieldPlacementFilters, HeightfieldPlacementKey, HeightfieldPlacementOutput,
    HeightfieldPlacementRegion, HeightfieldSpec, HeightfieldTile, PlanarHeightfieldEmbedding,
    PlanarHeightfieldSurface, TileKey, HEIGHTFIELD_PLACEMENT_DELETION_VERSION,
    HEIGHTFIELD_PLACEMENT_VERSION,
};

use crate::{
    decode_placement_deletions, encode_placement_deletions, PlacementDeletionBinding,
    PlacementDeletionCodecBudget, PlacementPersistenceError, PLACEMENT_DELETION_CODEC_VERSION,
    PLACEMENT_DELETION_HEADER_BYTES, PLACEMENT_DELETION_KEY_BYTES,
};

fn budget() -> PlacementDeletionCodecBudget {
    PlacementDeletionCodecBudget {
        max_encoded_bytes: 1_000_000,
        deletions: HeightfieldPlacementDeletionBudget {
            max_keys: 10_000,
            max_key_bytes: 1_000_000,
        },
    }
}

fn key(x: i64) -> HeightfieldPlacementKey {
    HeightfieldPlacementKey {
        placement_version: HEIGHTFIELD_PLACEMENT_VERSION,
        source: ScatterCandidateKey {
            kernel_version: SCATTER_KERNEL_VERSION,
            seed: 0x0123_4567_89ab_cdef,
            version: 0xfedc_ba98,
            salt: 0xfedc_ba98_7654_3210,
            grid_spacing_mm: 1000,
            grid: SectorCoord { x, y: 0, z: -3 },
        },
    }
}

fn set(keys: Vec<HeightfieldPlacementKey>) -> HeightfieldPlacementDeletionSet {
    HeightfieldPlacementDeletionSet::new(keys, budget().deletions).unwrap()
}

fn binding() -> PlacementDeletionBinding {
    PlacementDeletionBinding {
        object: PersistentId(u128::MAX - 1),
        field: FieldId(947),
    }
}

fn commit(
    timeline: &mut Timeline,
    id: u128,
    owner: PlacementDeletionBinding,
    deleted: &HeightfieldPlacementDeletionSet,
) -> Transaction {
    timeline
        .commit(
            TransactionId(id),
            ActorId(3),
            LogicalClock(id as u64),
            owner.object,
            owner.delta(deleted, budget().max_encoded_bytes).unwrap(),
        )
        .unwrap()
}

const HEIGHTS: [i64; 9] = [0, 8, 16, -8, 0, 8, -16, -8, 0];

struct GridHeights<'a>(&'a [i64; 9]);

impl HeightSampler for GridHeights<'_> {
    type Error = Infallible;

    fn sample_height(&self, sample: HeightSample) -> Result<HeightSampleValue, Self::Error> {
        let index = |q: u16| ((u64::from(q) * 2 + 32_767) / 65_535) as usize;
        Ok(HeightSampleValue::from_millimetres(
            self.0[index(sample.v_q16()) * 3 + index(sample.u_q16())],
        ))
    }
}

fn surface(heights: &[i64; 9]) -> PlanarHeightfieldSurface {
    let tile = HeightfieldTile::bake(
        TileKey::root(CubeFace::PositiveY),
        HeightfieldSpec::new(3, 1).unwrap(),
        &GridHeights(heights),
        BakeLimits {
            max_samples: 9,
            max_bytes: 9 * size_of::<i16>(),
        },
    )
    .unwrap();
    PlanarHeightfieldSurface::new(
        Arc::new(tile),
        PlanarHeightfieldEmbedding {
            sample_origin_mm: [-16, 40, -16],
            sample_spacing_mm: 16,
        },
    )
    .unwrap()
}

fn config() -> HeightfieldPlacementConfig {
    HeightfieldPlacementConfig {
        seed: 0x3a65_c807_9742_1eab,
        version: 7,
        salt: 0xf09c_1172_d538_208a,
        grid_spacing_mm: 3,
        min_distance_mm: 0,
        vertical_offset_mm: 7,
    }
}

fn source_on(heights: &[i64; 9], config: HeightfieldPlacementConfig) -> HeightfieldPlacementOutput {
    let brushes = vec![DensityBrush {
        id: 2,
        center_mm: [-9, 47, 0],
        inner_radius_mm: 10,
        outer_radius_mm: 24,
        opacity: u16::MAX,
        density: DensityBrushOp::Add(0),
        species: Some(23),
    }];
    let limits = ScatterFieldBuildBudget {
        max_brushes: brushes.len(),
        max_brush_bytes: brushes.capacity() * size_of::<DensityBrush>(),
    };
    let field = ScatterDensityField::new(
        ScatterFieldConfig {
            seed: 91,
            version: 3,
            base_density: u16::MAX,
            base_species: 11,
            noise: None,
        },
        brushes,
        limits,
    )
    .unwrap();
    scatter_on_heightfield(
        config,
        HeightfieldPlacementRegion::new([-13, -11], [14, 13]).unwrap(),
        &field,
        &surface(heights),
        HeightfieldPlacementFilters {
            surface_height_mm: [i64::MIN, i64::MAX],
            max_slope_rise: u32::MAX,
            max_slope_run: 1,
        },
        HeightfieldPlacementBudget {
            max_candidate_evaluations: 1_000_000,
            max_sample_reads: 100_000,
            max_field_work: 1_000_000,
            max_output_points: 1000,
            max_output_bytes: 1_000_000,
        },
    )
    .unwrap()
}

// Each call bakes a new tile and constructs a new surface/field, not a cloned output.
fn source() -> HeightfieldPlacementOutput {
    source_on(&HEIGHTS, config())
}

#[test]
fn canonical_ppld_is_a_16_byte_header_and_56_byte_full_key() {
    let k = key(-1);
    let bytes = encode_placement_deletions(&set(vec![k]), 72).unwrap();
    let mut expected = Vec::new();
    expected.extend_from_slice(b"PPLD");
    expected.extend_from_slice(&1_u32.to_le_bytes());
    expected.extend_from_slice(&1_u64.to_le_bytes());
    expected.extend_from_slice(&k.placement_version.to_le_bytes());
    expected.extend_from_slice(&k.source.kernel_version.to_le_bytes());
    expected.extend_from_slice(&k.source.seed.to_le_bytes());
    expected.extend_from_slice(&k.source.version.to_le_bytes());
    expected.extend_from_slice(&k.source.salt.to_le_bytes());
    expected.extend_from_slice(&1000_u32.to_le_bytes());
    for coordinate in [-1_i64, 0, -3] {
        expected.extend_from_slice(&coordinate.to_le_bytes());
    }
    assert_eq!(PLACEMENT_DELETION_CODEC_VERSION, 1);
    assert_eq!(PLACEMENT_DELETION_HEADER_BYTES, 16);
    assert_eq!(PLACEMENT_DELETION_KEY_BYTES, 56);
    assert_eq!(bytes.len(), 16 + 56);
    assert_eq!(bytes, expected);
    assert_eq!(
        decode_placement_deletions(&bytes, budget()).unwrap().keys(),
        &[k]
    );
}

#[test]
fn empty_wire_needs_a_header_but_absent_and_delete_need_no_budget() {
    let empty = set(vec![]);
    let bytes = encode_placement_deletions(&empty, 16).unwrap();
    assert_eq!(
        bytes,
        [
            b"PPLD".as_slice(),
            &1_u32.to_le_bytes(),
            &0_u64.to_le_bytes()
        ]
        .concat()
    );
    assert_eq!(decode_placement_deletions(&bytes, budget()).unwrap(), empty);
    assert_eq!(
        encode_placement_deletions(&empty, 15),
        Err(PlacementPersistenceError::EncodedByteBudgetExceeded {
            required: 16,
            limit: 15
        })
    );
    let zero = PlacementDeletionCodecBudget {
        max_encoded_bytes: 0,
        deletions: HeightfieldPlacementDeletionBudget {
            max_keys: 0,
            max_key_bytes: 0,
        },
    };
    let mut timeline = Timeline::new();
    let absent = binding().read(timeline.state(), zero).unwrap();
    assert!(absent.deletions().keys().is_empty());
    let delta = binding().delta(&empty, 0).unwrap();
    assert_eq!(delta.iter().next().unwrap().value, OverrideValue::Delete);
    timeline
        .commit(
            TransactionId(1),
            ActorId(3),
            LogicalClock(1),
            binding().object,
            delta,
        )
        .unwrap();
    assert_eq!(
        timeline.state().get(binding().object, binding().field),
        None
    );
    assert_eq!(binding().read(timeline.state(), zero).unwrap(), absent);
}

#[test]
fn sorted_unique_inputs_share_one_canonical_encoding() {
    let a = set(vec![key(8), key(-9), key(0), key(-9)]);
    let b = set(vec![key(-9), key(0), key(8)]);
    let bytes = encode_placement_deletions(&a, 184).unwrap();
    assert_eq!(bytes, encode_placement_deletions(&b, 184).unwrap());
    assert_eq!(decode_placement_deletions(&bytes, budget()).unwrap(), b);
}

#[test]
fn all_valid_identity_fields_and_extreme_signed_coordinates_roundtrip() {
    let mut keys = vec![key(-9)];
    for field in 0..6 {
        let mut changed = key(-9);
        match field {
            0 => changed.source.seed ^= 1 << 63,
            1 => changed.source.version ^= 1 << 31,
            2 => changed.source.salt ^= 1 << 63,
            3 => changed.source.grid_spacing_mm = u32::MAX,
            4 => changed.source.grid.x = 11,
            _ => changed.source.grid.z = -91,
        }
        keys.push(changed);
    }
    let mut extreme = key(i64::MIN);
    extreme.source.grid_spacing_mm = 1;
    extreme.source.grid.z = i64::MAX;
    keys.push(extreme);
    let original = set(keys);
    let bytes = encode_placement_deletions(&original, 1000).unwrap();
    let decoded = decode_placement_deletions(&bytes, budget()).unwrap();
    assert_eq!(decoded, original);
    assert!(decoded.keys().iter().all(|k| k.source.grid.y == 0));
}

#[test]
fn every_truncated_prefix_trailing_byte_and_inconsistent_count_fails() {
    let bytes = encode_placement_deletions(&set(vec![key(0), key(1)]), 128).unwrap();
    for length in 0..bytes.len() {
        assert!(
            decode_placement_deletions(&bytes[..length], budget()).is_err(),
            "prefix {length}"
        );
    }
    let mut extra = bytes.clone();
    extra.push(0);
    assert_eq!(
        decode_placement_deletions(&extra, budget()),
        Err(PlacementPersistenceError::InvalidLength {
            expected: 128,
            actual: 129
        })
    );
    for count in [0_u64, 1, 3] {
        let mut wrong = bytes.clone();
        wrong[8..16].copy_from_slice(&count.to_le_bytes());
        assert_eq!(
            decode_placement_deletions(&wrong, budget()),
            Err(PlacementPersistenceError::InvalidLength {
                expected: 16 + 56 * count as usize,
                actual: 128
            })
        );
    }
}

#[test]
fn invalid_magic_old_scatter_magic_and_unknown_codec_are_rejected() {
    let good = encode_placement_deletions(&set(vec![]), 16).unwrap();
    for magic in [*b"PSDL", *b"PPLX", [0; 4]] {
        let mut bytes = good.clone();
        bytes[..4].copy_from_slice(&magic);
        assert_eq!(
            decode_placement_deletions(&bytes, budget()),
            Err(PlacementPersistenceError::InvalidMagic)
        );
    }
    for version in [0_u32, 2, u32::MAX] {
        let mut bytes = good.clone();
        bytes[4..8].copy_from_slice(&version.to_le_bytes());
        assert_eq!(
            decode_placement_deletions(&bytes, budget()),
            Err(PlacementPersistenceError::UnsupportedVersion(version))
        );
    }
}

#[test]
fn duplicate_and_descending_wire_records_are_rejected_not_normalized() {
    let good = encode_placement_deletions(&set(vec![key(0), key(1)]), 128).unwrap();
    let mut duplicate = good.clone();
    duplicate.copy_within(16..72, 72);
    assert_eq!(
        decode_placement_deletions(&duplicate, budget()),
        Err(PlacementPersistenceError::NonCanonicalKeys)
    );
    let mut reversed = good;
    for byte in 0..56 {
        reversed.swap(16 + byte, 72 + byte);
    }
    assert_eq!(
        decode_placement_deletions(&reversed, budget()),
        Err(PlacementPersistenceError::NonCanonicalKeys)
    );
}

#[test]
fn placement_version_and_nonzero_source_y_are_validated_without_terrain() {
    let good = encode_placement_deletions(&set(vec![key(0)]), 72).unwrap();
    for version in [0_u32, 2, u32::MAX] {
        let mut bytes = good.clone();
        bytes[16..20].copy_from_slice(&version.to_le_bytes());
        assert_eq!(
            decode_placement_deletions(&bytes, budget()),
            Err(PlacementPersistenceError::Deletions(
                HeightfieldPlacementDeletionError::InvalidKey(
                    HeightfieldPlacementError::InvalidKey
                )
            ))
        );
    }
    for y in [i64::MIN, -1, 1, i64::MAX] {
        let mut bytes = good.clone();
        bytes[56..64].copy_from_slice(&y.to_le_bytes());
        assert_eq!(
            decode_placement_deletions(&bytes, budget()),
            Err(PlacementPersistenceError::Deletions(
                HeightfieldPlacementDeletionError::InvalidKey(
                    HeightfieldPlacementError::InvalidKey
                )
            ))
        );
    }
}

#[test]
fn invalid_raw_kernel_zero_spacing_and_coordinate_overflow_are_rejected() {
    let good = encode_placement_deletions(&set(vec![key(0)]), 72).unwrap();
    for (offset, value, expected) in [
        (20, 2_u32, ScatterError::UnsupportedKernelVersion),
        (44, 0, ScatterError::InvalidGridSpacing),
    ] {
        let mut bytes = good.clone();
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        assert_eq!(
            decode_placement_deletions(&bytes, budget()),
            Err(PlacementPersistenceError::Deletions(
                HeightfieldPlacementDeletionError::InvalidKey(HeightfieldPlacementError::Source(
                    expected
                ))
            ))
        );
    }
    for offset in [48, 64] {
        for coordinate in [i64::MIN, i64::MAX] {
            let mut bytes = good.clone();
            bytes[offset..offset + 8].copy_from_slice(&coordinate.to_le_bytes());
            assert_eq!(
                decode_placement_deletions(&bytes, budget()),
                Err(PlacementPersistenceError::Deletions(
                    HeightfieldPlacementDeletionError::InvalidKey(
                        HeightfieldPlacementError::Source(ScatterError::CoordinateOverflow)
                    )
                ))
            );
        }
    }
}

#[test]
fn wire_budget_precedes_parsing_and_has_an_exact_inclusive_boundary() {
    let mut limits = budget();
    limits.max_encoded_bytes = 71;
    assert_eq!(
        decode_placement_deletions(&[0; 72], limits),
        Err(PlacementPersistenceError::EncodedByteBudgetExceeded {
            required: 72,
            limit: 71
        })
    );
    let deleted = set(vec![key(0)]);
    assert_eq!(
        encode_placement_deletions(&deleted, 71),
        Err(PlacementPersistenceError::EncodedByteBudgetExceeded {
            required: 72,
            limit: 71
        })
    );
    let bytes = encode_placement_deletions(&deleted, 72).unwrap();
    limits.max_encoded_bytes = 72;
    assert_eq!(decode_placement_deletions(&bytes, limits).unwrap(), deleted);
}

#[test]
fn decoded_count_and_actual_key_payload_are_independent_wire_budgets() {
    let bytes = encode_placement_deletions(&set(vec![key(0)]), 72).unwrap();
    let mut limits = budget();
    limits.max_encoded_bytes = 72;
    limits.deletions.max_keys = 0;
    assert_eq!(
        decode_placement_deletions(&bytes, limits),
        Err(PlacementPersistenceError::Deletions(
            HeightfieldPlacementDeletionError::KeyBudgetExceeded {
                required: 1,
                limit: 0
            }
        ))
    );
    limits.deletions.max_keys = 1;
    limits.deletions.max_key_bytes = size_of::<HeightfieldPlacementKey>() - 1;
    assert_eq!(
        decode_placement_deletions(&bytes, limits),
        Err(PlacementPersistenceError::Deletions(
            HeightfieldPlacementDeletionError::KeyBytesBudgetExceeded {
                required: size_of::<HeightfieldPlacementKey>(),
                limit: size_of::<HeightfieldPlacementKey>() - 1
            }
        ))
    );
    limits.deletions.max_key_bytes += 1;
    let decoded = decode_placement_deletions(&bytes, limits).unwrap();
    assert_eq!(
        decoded.allocated_key_bytes(),
        size_of::<HeightfieldPlacementKey>()
    );
}

#[test]
fn hostile_declared_counts_fail_before_any_key_reservation() {
    let mut bytes = encode_placement_deletions(&set(vec![]), 16).unwrap();
    bytes[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
    assert_eq!(
        decode_placement_deletions(&bytes, budget()),
        Err(PlacementPersistenceError::ArithmeticOverflow)
    );
    bytes[8..16].copy_from_slice(&1000_u64.to_le_bytes());
    assert_eq!(
        decode_placement_deletions(&bytes, budget()),
        Err(PlacementPersistenceError::InvalidLength {
            expected: 56_016,
            actual: 16
        })
    );
}

#[test]
fn single_bit_mutations_can_only_decode_to_canonical_valid_full_keys() {
    let good = encode_placement_deletions(&set(vec![key(-2), key(3)]), 128).unwrap();
    for index in 0..good.len() {
        for bit in 0..8 {
            let mut bytes = good.clone();
            bytes[index] ^= 1 << bit;
            if let Ok(decoded) = decode_placement_deletions(&bytes, budget()) {
                assert_eq!(
                    encode_placement_deletions(&decoded, bytes.len()).unwrap(),
                    bytes
                );
                assert!(decoded.keys().windows(2).all(|pair| pair[0] < pair[1]));
                assert!(decoded
                    .keys()
                    .iter()
                    .all(|k| k.placement_version == HEIGHTFIELD_PLACEMENT_VERSION
                        && k.source.grid.y == 0
                        && k.source.rebuild().is_ok()));
            }
        }
    }
}

#[test]
fn binding_scopes_authority_to_both_full_object_id_and_field_id() {
    let owner = binding();
    let deleted = set(vec![key(1)]);
    let mut timeline = Timeline::new();
    commit(&mut timeline, 1, owner, &deleted);
    assert_eq!(
        owner.read(timeline.state(), budget()).unwrap().deletions(),
        &deleted
    );
    for other in [
        PlacementDeletionBinding {
            object: PersistentId(owner.object.0 ^ (1_u128 << 127)),
            ..owner
        },
        PlacementDeletionBinding {
            field: FieldId(owner.field.0 + 1),
            ..owner
        },
    ] {
        assert!(other
            .read(timeline.state(), budget())
            .unwrap()
            .deletions()
            .keys()
            .is_empty());
        assert_ne!(
            owner.snapshot(deleted.clone()),
            other.snapshot(deleted.clone())
        );
    }
}

#[test]
fn malformed_or_wrong_type_fields_do_not_mutate_authority_or_old_snapshots() {
    let owner = binding();
    let mut timeline = Timeline::new();
    commit(&mut timeline, 1, owner, &set(vec![key(1)]));
    let old = owner.read(timeline.state(), budget()).unwrap();
    for (id, value, error) in [
        (
            2,
            OverrideValue::Boolean(false),
            PlacementPersistenceError::UnexpectedValueType,
        ),
        (
            3,
            OverrideValue::Bytes(vec![0; 16]),
            PlacementPersistenceError::InvalidMagic,
        ),
        (
            4,
            OverrideValue::Bytes(vec![]),
            PlacementPersistenceError::InvalidLength {
                expected: 16,
                actual: 0,
            },
        ),
    ] {
        timeline
            .commit(
                TransactionId(id),
                ActorId(3),
                LogicalClock(id as u64),
                owner.object,
                OverrideDelta::single(owner.field, value).unwrap(),
            )
            .unwrap();
        let before = timeline.state().clone();
        assert_eq!(owner.read(timeline.state(), budget()), Err(error));
        assert_eq!(timeline.state(), &before);
        assert_eq!(old.deletions().keys(), &[key(1)]);
    }
}

#[test]
fn over_budget_delta_preparation_leaves_the_log_and_prior_snapshot_unchanged() {
    let mut timeline = Timeline::new();
    commit(&mut timeline, 1, binding(), &set(vec![key(0)]));
    let old = binding().read(timeline.state(), budget()).unwrap();
    let next = old
        .deletions()
        .with_deleted(key(1), true, budget().deletions)
        .unwrap();
    let state = timeline.state().clone();
    assert_eq!(
        binding().delta(&next, 127),
        Err(PlacementPersistenceError::EncodedByteBudgetExceeded {
            required: 128,
            limit: 127
        })
    );
    assert_eq!(timeline.state(), &state);
    assert_eq!(timeline.log().len(), 1);
    assert_eq!(binding().read(timeline.state(), budget()).unwrap(), old);
}

#[test]
fn cache_dependency_includes_absent_keys_full_namespace_and_filter_version() {
    let owner = binding();
    let empty = owner.snapshot(set(vec![]));
    let mut absent = key(8000);
    absent.source.seed = 99;
    let with_absent = owner.snapshot(set(vec![absent]));
    let baseline = empty.filter(source()).unwrap();
    let other = with_absent.filter(source()).unwrap();
    assert_eq!(baseline.source, other.source);
    assert_eq!(other.removed, 0);
    assert_ne!(baseline.snapshot, other.snapshot);
    assert_eq!(other.snapshot.deletions().keys(), &[absent]);
    assert_eq!(
        other.snapshot.deletion_version(),
        HEIGHTFIELD_PLACEMENT_DELETION_VERSION
    );
    for owner in [
        PlacementDeletionBinding {
            object: PersistentId(7),
            ..owner
        },
        PlacementDeletionBinding {
            field: FieldId(948),
            ..owner
        },
    ] {
        let output = owner.snapshot(set(vec![])).filter(source()).unwrap();
        assert_eq!(baseline.source, output.source);
        assert_ne!(baseline.snapshot, output.snapshot);
        assert_eq!(output.snapshot.binding(), owner);
    }
}

#[test]
fn deletion_matching_uses_every_source_key_field_not_an_xz_slot_or_geometry() {
    let original = source();
    let target = original.points[0].key;
    for field in 0..6 {
        let mut changed = target;
        match field {
            0 => changed.source.seed ^= 1 << 63,
            1 => changed.source.version ^= 1 << 31,
            2 => changed.source.salt ^= 1 << 63,
            3 => changed.source.grid_spacing_mm += 1,
            4 => changed.source.grid.x = 8000,
            _ => changed.source.grid.z = -8000,
        }
        let filtered = binding()
            .snapshot(set(vec![changed]))
            .filter(source())
            .unwrap();
        assert_eq!(filtered.removed, 0, "identity field {field}");
        assert_eq!(filtered.source, original);
    }
    let filtered = binding()
        .snapshot(set(vec![target]))
        .filter(source())
        .unwrap();
    assert_eq!(filtered.removed, 1);
    assert!(filtered.source.points.iter().all(|p| p.key != target));
}

#[test]
fn fresh_regeneration_filters_stably_in_place_with_attributes_cache_and_counters() {
    let original = source();
    assert!(original.points.len() > 4);
    assert!(original.attributes.iter().any(|a| a.species == 11));
    assert!(original.attributes.iter().any(|a| a.species == 23));
    let deleted = set(original
        .points
        .iter()
        .step_by(3)
        .map(|p| p.key)
        .chain([key(8000)])
        .collect());
    let expected: Vec<_> = original
        .points
        .iter()
        .copied()
        .zip(original.attributes.iter().copied())
        .filter(|(p, _)| !deleted.is_deleted(p.key))
        .collect();
    let cache = original.cache_key.clone();
    let counters = (
        original.candidate_evaluations,
        original.surface_sample_reads,
        original.field_work,
    );
    let bytes = encode_placement_deletions(&deleted, budget().max_encoded_bytes).unwrap();
    drop(original);
    let snapshot = binding().snapshot(decode_placement_deletions(&bytes, budget()).unwrap());
    let regenerated = source();
    let points_ptr = regenerated.points.as_ptr();
    let attrs_ptr = regenerated.attributes.as_ptr();
    let capacities = (
        regenerated.points.capacity(),
        regenerated.attributes.capacity(),
    );
    let filtered = snapshot.filter(regenerated).unwrap();
    assert_eq!(
        filtered
            .source
            .points
            .iter()
            .copied()
            .zip(filtered.source.attributes.iter().copied())
            .collect::<Vec<_>>(),
        expected
    );
    assert_eq!(filtered.source.points.as_ptr(), points_ptr);
    assert_eq!(filtered.source.attributes.as_ptr(), attrs_ptr);
    assert_eq!(
        (
            filtered.source.points.capacity(),
            filtered.source.attributes.capacity()
        ),
        capacities
    );
    assert_eq!(filtered.source.cache_key, cache);
    assert_eq!(
        (
            filtered.source.candidate_evaluations,
            filtered.source.surface_sample_reads,
            filtered.source.field_work
        ),
        counters
    );
    assert_eq!(filtered.removed, deleted.keys().len() - 1);
    assert_eq!(filtered.snapshot, snapshot);
    assert_eq!(snapshot.filter(source()).unwrap().source, filtered.source);
}

#[test]
fn regenerated_changed_terrain_and_offset_keep_deletion_identity_not_geometry() {
    let original = source();
    let deleted = set(original.points.iter().step_by(2).map(|p| p.key).collect());
    let keys: Vec<_> = original.points.iter().map(|p| p.key).collect();
    let old_cache = original.cache_key.clone();
    let changed = source_on(
        &HEIGHTS.map(|h| h + 20),
        HeightfieldPlacementConfig {
            vertical_offset_mm: config().vertical_offset_mm + 13,
            ..config()
        },
    );
    assert_eq!(
        changed.points.iter().map(|p| p.key).collect::<Vec<_>>(),
        keys
    );
    for (before, after) in original.points.iter().zip(&changed.points) {
        assert_eq!(after.position_mm[1], before.position_mm[1] + 33);
        assert_eq!(before.position_mm[0], after.position_mm[0]);
        assert_eq!(before.position_mm[2], after.position_mm[2]);
    }
    let changed_cache = changed.cache_key.clone();
    assert_ne!(changed_cache, old_cache);
    let expected: Vec<_> = changed
        .points
        .iter()
        .copied()
        .filter(|p| !deleted.is_deleted(p.key))
        .collect();
    drop(original);
    let filtered = binding().snapshot(deleted.clone()).filter(changed).unwrap();
    assert_eq!(filtered.source.points, expected);
    assert_eq!(filtered.source.cache_key, changed_cache);
    assert_eq!(filtered.removed, deleted.keys().len());
}

#[test]
fn malformed_output_alignment_fails_even_with_empty_deletions_or_points() {
    let snapshot = binding().snapshot(set(vec![]));
    for mode in 0..3 {
        let mut output = source();
        match mode {
            0 => {
                output.attributes.pop();
            }
            1 => output.points.clear(),
            _ => output.attributes.push(output.attributes[0]),
        }
        assert_eq!(
            snapshot.filter(output),
            Err(PlacementPersistenceError::Deletions(
                HeightfieldPlacementDeletionError::MisalignedAttributes
            ))
        );
    }
    let mut output = source();
    output.points.clear();
    output.attributes.clear();
    assert_eq!(snapshot.filter(output).unwrap().removed, 0);
    let output = source();
    let snapshot = binding().snapshot(set(output.points.iter().map(|p| p.key).collect()));
    let count = output.points.len();
    let ptrs = (output.points.as_ptr(), output.attributes.as_ptr());
    let capacities = (output.points.capacity(), output.attributes.capacity());
    let filtered = snapshot.filter(output).unwrap();
    assert_eq!(filtered.removed, count);
    assert!(filtered.source.points.is_empty());
    assert!(filtered.source.attributes.is_empty());
    assert_eq!(
        (
            filtered.source.points.as_ptr(),
            filtered.source.attributes.as_ptr()
        ),
        ptrs
    );
    assert_eq!(
        (
            filtered.source.points.capacity(),
            filtered.source.attributes.capacity()
        ),
        capacities
    );
}

#[test]
fn timeline_replacement_undo_redo_restores_actual_previous_sets_not_a_union() {
    let owner = binding();
    let mut timeline = Timeline::new();
    let first = set(vec![key(0), key(1)]);
    let replacement = set(vec![key(2)]);
    commit(&mut timeline, 1, owner, &first);
    commit(&mut timeline, 2, owner, &replacement);
    assert_eq!(
        owner.read(timeline.state(), budget()).unwrap().deletions(),
        &replacement
    );
    timeline
        .undo(TransactionId(3), ActorId(3), LogicalClock(3))
        .unwrap();
    assert_eq!(
        owner.read(timeline.state(), budget()).unwrap().deletions(),
        &first
    );
    timeline
        .redo(TransactionId(4), ActorId(3), LogicalClock(4))
        .unwrap();
    assert_eq!(
        owner.read(timeline.state(), budget()).unwrap().deletions(),
        &replacement
    );
    assert_eq!(timeline.log().len(), 4);
}

#[test]
fn first_write_and_clear_have_exact_undo_redo_and_snapshot_suffix_replay() {
    let owner = binding();
    let mut timeline = Timeline::new();
    let deleted = set(vec![key(0)]);
    commit(&mut timeline, 1, owner, &deleted);
    timeline
        .undo(TransactionId(2), ActorId(3), LogicalClock(2))
        .unwrap();
    assert_eq!(timeline.state().get(owner.object, owner.field), None);
    timeline
        .redo(TransactionId(3), ActorId(3), LogicalClock(3))
        .unwrap();
    let checkpoint = decode_snapshot(&encode_snapshot(&timeline.snapshot()).unwrap()).unwrap();
    commit(&mut timeline, 4, owner, &set(vec![]));
    assert_eq!(timeline.state().get(owner.object, owner.field), None);
    timeline
        .undo(TransactionId(5), ActorId(3), LogicalClock(5))
        .unwrap();
    assert_eq!(
        owner.read(timeline.state(), budget()).unwrap().deletions(),
        &deleted
    );
    let replayed = Timeline::replay(&checkpoint, timeline.log()).unwrap();
    assert_eq!(&replayed, timeline.state());
    timeline
        .redo(TransactionId(6), ActorId(3), LogicalClock(6))
        .unwrap();
    let replayed = Timeline::replay(&checkpoint, timeline.log()).unwrap();
    assert_eq!(replayed.get(owner.object, owner.field), None);
    assert!(owner
        .read(&replayed, budget())
        .unwrap()
        .deletions()
        .keys()
        .is_empty());
}

#[test]
fn crdt_merge_is_order_independent_whole_field_lww_including_delete_not_key_union() {
    let mut left = Timeline::new();
    let mut right = Timeline::new();
    let a = commit(&mut left, 1, binding(), &set(vec![key(0)]));
    let b = commit(&mut right, 2, binding(), &set(vec![key(1)]));
    let clear = commit(&mut right, 3, binding(), &set(vec![]));
    for transactions in [[&a, &b], [&b, &a]] {
        let mut merged = CrdtState::new();
        merged.merge(transactions).unwrap();
        let Some(OverrideValue::Bytes(bytes)) = merged.get(binding().object, binding().field)
        else {
            panic!("placement deletion blob")
        };
        assert_eq!(
            decode_placement_deletions(bytes, budget()).unwrap().keys(),
            &[key(1)]
        );
        merged.merge([&clear]).unwrap();
        assert_eq!(merged.get(binding().object, binding().field), None);
        assert_eq!(
            merged
                .register(binding().object, binding().field)
                .unwrap()
                .value(),
            &OverrideValue::Delete
        );
    }
    let mut merged = CrdtState::new();
    merged.merge([&clear, &b, &a]).unwrap();
    assert_eq!(merged.get(binding().object, binding().field), None);
}

#[cfg(feature = "std")]
#[test]
fn real_file_journal_close_reopen_replay_regenerates_placements_and_undo_redo() {
    use prism_world_persist::FileJournal;
    use std::{
        fs,
        time::{SystemTime, UNIX_EPOCH},
    };

    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let directory = std::env::temp_dir().join(alloc::format!(
        "prism-placement-deletions-{}-{nonce}",
        std::process::id()
    ));
    // Exclusively own the directory before opening a create-if-missing journal.
    // A name collision must fail without touching or cleaning any existing file.
    fs::create_dir(&directory).unwrap();
    struct RemoveOnDrop(std::path::PathBuf);
    impl Drop for RemoveOnDrop {
        fn drop(&mut self) {
            let _ = fs::remove_file(self.0.join("placement.journal"));
            let _ = fs::remove_dir(&self.0);
        }
    }
    let path = directory.join("placement.journal");
    let _cleanup = RemoveOnDrop(directory);
    let original = source();
    let deleted = set(original.points.iter().step_by(3).map(|p| p.key).collect());
    let expected: Vec<_> = original
        .points
        .iter()
        .copied()
        .zip(original.attributes.iter().copied())
        .filter(|(p, _)| !deleted.is_deleted(p.key))
        .collect();
    let mut timeline = Timeline::new();
    {
        let mut journal = FileJournal::open(&path).unwrap();
        journal
            .append(commit(&mut timeline, 1, binding(), &deleted))
            .unwrap();
    }
    // Discard all generated geometry before reading authority back from disk.
    drop(original);
    let recovered = {
        let journal = FileJournal::open(&path).unwrap();
        assert_eq!(journal.recovery(), None);
        assert_eq!(journal.log().len(), 1);
        Timeline::replay(&Snapshot::new(0, WorldState::new()), journal.log()).unwrap()
    };
    let snapshot = binding().read(&recovered, budget()).unwrap();
    assert_eq!(snapshot.deletions(), &deleted);
    let filtered = snapshot.filter(source()).unwrap();
    assert_eq!(
        filtered
            .source
            .points
            .iter()
            .copied()
            .zip(filtered.source.attributes.iter().copied())
            .collect::<Vec<_>>(),
        expected
    );
    assert_eq!(filtered.removed, deleted.keys().len());
    drop(filtered);
    {
        let mut journal = FileJournal::open(&path).unwrap();
        journal
            .append(
                timeline
                    .undo(TransactionId(2), ActorId(3), LogicalClock(2))
                    .unwrap(),
            )
            .unwrap();
    }
    {
        let journal = FileJournal::open(&path).unwrap();
        let replayed =
            Timeline::replay(&Snapshot::new(0, WorldState::new()), journal.log()).unwrap();
        assert_eq!(replayed.get(binding().object, binding().field), None);
        let fresh = source();
        let unfiltered = binding()
            .read(&replayed, budget())
            .unwrap()
            .filter(source())
            .unwrap();
        assert_eq!(unfiltered.removed, 0);
        assert_eq!(unfiltered.source, fresh);
    }
    {
        let mut journal = FileJournal::open(&path).unwrap();
        journal
            .append(
                timeline
                    .redo(TransactionId(3), ActorId(3), LogicalClock(3))
                    .unwrap(),
            )
            .unwrap();
    }
    {
        let journal = FileJournal::open(&path).unwrap();
        assert_eq!(journal.recovery(), None);
        assert_eq!(journal.log().len(), 3);
        let replayed =
            Timeline::replay(&Snapshot::new(0, WorldState::new()), journal.log()).unwrap();
        assert_eq!(&replayed, timeline.state());
        let filtered = binding()
            .read(&replayed, budget())
            .unwrap()
            .filter(source())
            .unwrap();
        assert_eq!(
            filtered
                .source
                .points
                .iter()
                .copied()
                .zip(filtered.source.attributes.iter().copied())
                .collect::<Vec<_>>(),
            expected
        );
        assert_eq!(filtered.removed, deleted.keys().len());
    }
}
