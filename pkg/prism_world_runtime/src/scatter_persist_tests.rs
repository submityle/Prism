use alloc::{vec, vec::Vec};
use core::mem::size_of;

use prism_world_persist::{
    decode_snapshot, encode_snapshot, ActorId, CrdtState, FieldId, LogicalClock, OverrideDelta,
    OverrideValue, PersistentId, Timeline, Transaction, TransactionId,
};
#[cfg(feature = "std")]
use prism_world_persist::{Snapshot, WorldState};
use prism_world_residency::{
    scatter, scatter_with_density_field, FieldScatterBudget, ScatterBudget, ScatterCandidateKey,
    ScatterConfig, ScatterDeletionBudget, ScatterDeletionError, ScatterDeletionSet,
    ScatterDensityField, ScatterError, ScatterFieldBuildBudget, ScatterFieldConfig, ScatterOutput,
    ScatterRegion, SectorCoord, SCATTER_DELETION_VERSION, SCATTER_KERNEL_VERSION,
};

use crate::{
    decode_scatter_deletions, encode_scatter_deletions, ScatterDeletionBinding,
    ScatterDeletionCodecBudget, ScatterPersistenceError,
};

fn budget() -> ScatterDeletionCodecBudget {
    ScatterDeletionCodecBudget {
        max_encoded_bytes: 1_000_000,
        deletions: ScatterDeletionBudget {
            max_keys: 10_000,
            max_key_bytes: 1_000_000,
        },
    }
}

fn key(x: i64) -> ScatterCandidateKey {
    ScatterCandidateKey {
        kernel_version: SCATTER_KERNEL_VERSION,
        seed: 0x0123_4567_89ab_cdef,
        version: 0xfedc_ba98,
        salt: 0xfedc_ba98_7654_3210,
        grid_spacing_mm: 1000,
        grid: SectorCoord { x, y: -2, z: 3 },
    }
}

fn set(keys: Vec<ScatterCandidateKey>) -> ScatterDeletionSet {
    ScatterDeletionSet::new(keys, budget().deletions).unwrap()
}

fn binding() -> ScatterDeletionBinding {
    ScatterDeletionBinding {
        object: PersistentId(u128::MAX - 1),
        field: FieldId(931),
    }
}

fn commit(
    timeline: &mut Timeline,
    id: u128,
    owner: ScatterDeletionBinding,
    deletions: &ScatterDeletionSet,
) -> Transaction {
    timeline
        .commit(
            TransactionId(id),
            ActorId(3),
            LogicalClock(id as u64),
            owner.object,
            owner.delta(deletions, budget().max_encoded_bytes).unwrap(),
        )
        .unwrap()
}

fn source(region: ScatterRegion) -> ScatterOutput {
    scatter(
        ScatterConfig {
            seed: 17,
            version: 1,
            salt: 2,
            grid_spacing_mm: 1000,
            min_distance_mm: 600,
        },
        region,
        ScatterBudget {
            max_candidate_evaluations: 1_000_000,
            max_output_points: 10_000,
            max_output_bytes: 1_000_000,
        },
    )
    .unwrap()
}

fn region() -> ScatterRegion {
    ScatterRegion::new([-2000; 3], [2000; 3]).unwrap()
}

#[test]
fn canonical_wire_layout_is_explicit_little_endian_not_rust_layout() {
    let k = key(-1);
    let bytes = encode_scatter_deletions(&set(vec![k]), 68).unwrap();
    let mut expected = Vec::new();
    expected.extend_from_slice(b"PSDL");
    expected.extend_from_slice(&SCATTER_DELETION_VERSION.to_le_bytes());
    expected.extend_from_slice(&1_u64.to_le_bytes());
    expected.extend_from_slice(&1_u32.to_le_bytes());
    expected.extend_from_slice(&k.seed.to_le_bytes());
    expected.extend_from_slice(&k.version.to_le_bytes());
    expected.extend_from_slice(&k.salt.to_le_bytes());
    expected.extend_from_slice(&1000_u32.to_le_bytes());
    for value in [-1_i64, -2, 3] {
        expected.extend_from_slice(&value.to_le_bytes());
    }
    assert_eq!(bytes, expected);
    assert_eq!(bytes.len(), 16 + 52);
    assert_eq!(
        decode_scatter_deletions(&bytes, budget()).unwrap().keys(),
        &[k]
    );
}

#[test]
fn empty_codec_requires_header_but_empty_delta_removes_field_without_blob() {
    let empty = set(vec![]);
    assert_eq!(encode_scatter_deletions(&empty, 16).unwrap().len(), 16);
    assert!(matches!(
        encode_scatter_deletions(&empty, 15),
        Err(ScatterPersistenceError::EncodedByteBudgetExceeded {
            required: 16,
            limit: 15
        })
    ));
    let mut timeline = Timeline::new();
    timeline
        .commit(
            TransactionId(1),
            ActorId(1),
            LogicalClock(1),
            binding().object,
            binding().delta(&empty, 0).unwrap(),
        )
        .unwrap();
    assert_eq!(
        timeline.state().get(binding().object, binding().field),
        None
    );
    let zero = ScatterDeletionCodecBudget {
        max_encoded_bytes: 0,
        deletions: ScatterDeletionBudget {
            max_keys: 0,
            max_key_bytes: 0,
        },
    };
    assert!(binding()
        .read(timeline.state(), zero)
        .unwrap()
        .deletions()
        .keys()
        .is_empty());
}

#[test]
fn sorted_unique_sets_have_one_canonical_encoding() {
    let first = set(vec![key(5), key(-8), key(0), key(-8)]);
    let second = set(vec![key(-8), key(0), key(5)]);
    let bytes = encode_scatter_deletions(&first, usize::MAX).unwrap();
    assert_eq!(
        bytes,
        encode_scatter_deletions(&second, usize::MAX).unwrap()
    );
    assert_eq!(decode_scatter_deletions(&bytes, budget()).unwrap(), second);
}

#[test]
fn every_key_field_survives_codec_roundtrip() {
    let mut keys = vec![key(-9)];
    for offset in 0..7 {
        let mut changed = key(-9);
        match offset {
            0 => changed.seed ^= 1 << 63,
            1 => changed.version ^= 1 << 31,
            2 => changed.salt ^= 1 << 63,
            3 => changed.grid_spacing_mm = u32::MAX,
            4 => changed.grid.x = 11,
            5 => changed.grid.y = 19,
            _ => changed.grid.z = -91,
        }
        keys.push(changed);
    }
    let original = set(keys);
    let bytes = encode_scatter_deletions(&original, 1000).unwrap();
    assert_eq!(
        decode_scatter_deletions(&bytes, budget()).unwrap(),
        original
    );
}

#[test]
fn signed_extreme_grid_coordinates_with_unit_spacing_are_exact() {
    let mut k = key(i64::MIN);
    k.grid_spacing_mm = 1;
    k.grid.y = i64::MAX;
    k.grid.z = i64::MIN + 1;
    let original = set(vec![k]);
    let bytes = encode_scatter_deletions(&original, 68).unwrap();
    assert_eq!(
        decode_scatter_deletions(&bytes, budget()).unwrap(),
        original
    );
}

#[test]
fn every_truncated_prefix_and_trailing_byte_is_rejected() {
    let bytes = encode_scatter_deletions(&set(vec![key(0), key(1)]), 120).unwrap();
    for length in 0..bytes.len() {
        assert!(
            decode_scatter_deletions(&bytes[..length], budget()).is_err(),
            "prefix {length}"
        );
    }
    let mut extra = bytes;
    extra.push(0);
    assert!(matches!(
        decode_scatter_deletions(&extra, budget()),
        Err(ScatterPersistenceError::InvalidLength { .. })
    ));
}

#[test]
fn bad_magic_and_unknown_format_revision_are_not_silently_empty() {
    let mut bytes = encode_scatter_deletions(&set(vec![]), 16).unwrap();
    bytes[0] ^= 1;
    assert_eq!(
        decode_scatter_deletions(&bytes, budget()),
        Err(ScatterPersistenceError::InvalidMagic)
    );
    bytes[0] ^= 1;
    bytes[4..8].copy_from_slice(&2_u32.to_le_bytes());
    assert_eq!(
        decode_scatter_deletions(&bytes, budget()),
        Err(ScatterPersistenceError::UnsupportedVersion(2))
    );
}

#[test]
fn duplicate_and_out_of_order_wire_keys_are_rejected_not_normalized() {
    let good = encode_scatter_deletions(&set(vec![key(0), key(1)]), 120).unwrap();
    let mut duplicate = good.clone();
    duplicate.copy_within(16..68, 68);
    assert_eq!(
        decode_scatter_deletions(&duplicate, budget()),
        Err(ScatterPersistenceError::NonCanonicalKeys)
    );
    let mut reversed = good;
    for i in 0..52 {
        reversed.swap(16 + i, 68 + i);
    }
    assert_eq!(
        decode_scatter_deletions(&reversed, budget()),
        Err(ScatterPersistenceError::NonCanonicalKeys)
    );
}

#[test]
fn malformed_key_kernel_spacing_and_world_overflow_are_rejected() {
    let good = encode_scatter_deletions(&set(vec![key(0)]), 68).unwrap();
    for (offset, value, expected) in [
        (16, 2_u32, ScatterError::UnsupportedKernelVersion),
        (40, 0, ScatterError::InvalidGridSpacing),
    ] {
        let mut bytes = good.clone();
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        assert_eq!(
            decode_scatter_deletions(&bytes, budget()),
            Err(ScatterPersistenceError::Deletions(
                ScatterDeletionError::InvalidKey(expected)
            ))
        );
    }
    let mut bytes = good;
    bytes[44..52].copy_from_slice(&i64::MAX.to_le_bytes());
    assert_eq!(
        decode_scatter_deletions(&bytes, budget()),
        Err(ScatterPersistenceError::Deletions(
            ScatterDeletionError::InvalidKey(ScatterError::CoordinateOverflow)
        ))
    );
}

#[test]
fn wire_budget_is_checked_before_header_parsing() {
    let bytes = vec![0; 70];
    let mut limits = budget();
    limits.max_encoded_bytes = 69;
    assert_eq!(
        decode_scatter_deletions(&bytes, limits),
        Err(ScatterPersistenceError::EncodedByteBudgetExceeded {
            required: 70,
            limit: 69
        })
    );
    assert!(matches!(
        encode_scatter_deletions(&set(vec![key(0)]), 67),
        Err(ScatterPersistenceError::EncodedByteBudgetExceeded {
            required: 68,
            limit: 67
        })
    ));
}

#[test]
fn single_byte_mutations_never_bypass_key_or_canonical_validation() {
    let valid = encode_scatter_deletions(&set(vec![key(-2), key(3)]), 120).unwrap();
    for index in 0..valid.len() {
        for bit in 0..8 {
            let mut bytes = valid.clone();
            bytes[index] ^= 1 << bit;
            if let Ok(decoded) = decode_scatter_deletions(&bytes, budget()) {
                assert_eq!(
                    encode_scatter_deletions(&decoded, bytes.len()).unwrap(),
                    bytes
                );
                assert!(decoded.keys().windows(2).all(|pair| pair[0] < pair[1]));
                assert!(decoded.keys().iter().all(|key| key.rebuild().is_ok()));
            }
        }
    }
}

#[test]
fn rejected_delta_preparation_does_not_write_log_or_change_previous_snapshot() {
    let mut timeline = Timeline::new();
    let old_set = set(vec![key(0)]);
    commit(&mut timeline, 1, binding(), &old_set);
    let old = binding().read(timeline.state(), budget()).unwrap();
    let next = old_set
        .with_deleted(key(1), true, budget().deletions)
        .unwrap();
    let state = timeline.state().clone();
    assert!(binding().delta(&next, 68).is_err());
    assert_eq!(timeline.state(), &state);
    assert_eq!(timeline.log().len(), 1);
    assert_eq!(binding().read(timeline.state(), budget()).unwrap(), old);
}

#[test]
fn identical_current_points_do_not_make_different_snapshots_cache_equivalent() {
    let first = binding().snapshot(set(vec![]));
    let mut absent = key(8000);
    absent.seed = 99;
    let second = binding().snapshot(set(vec![absent]));
    let a = first.filter_source(source(region()));
    let b = second.filter_source(source(region()));
    assert_eq!(a.source, b.source);
    assert_ne!(a.snapshot, b.snapshot);
    let other_node = ScatterDeletionBinding {
        object: PersistentId(7),
        ..binding()
    };
    let c = other_node
        .snapshot(set(vec![]))
        .filter_source(source(region()));
    assert_eq!(a.source, c.source);
    assert_ne!(a.snapshot, c.snapshot);
}

#[test]
fn decoded_count_and_rust_payload_budgets_are_independent_of_wire_bytes() {
    let bytes = encode_scatter_deletions(&set(vec![key(0)]), 68).unwrap();
    let mut limits = budget();
    limits.max_encoded_bytes = bytes.len();
    limits.deletions.max_keys = 0;
    assert_eq!(
        decode_scatter_deletions(&bytes, limits),
        Err(ScatterPersistenceError::Deletions(
            ScatterDeletionError::KeyBudgetExceeded {
                required: 1,
                limit: 0
            }
        ))
    );
    limits.deletions.max_keys = 1;
    limits.deletions.max_key_bytes = size_of::<ScatterCandidateKey>() - 1;
    assert_eq!(
        decode_scatter_deletions(&bytes, limits),
        Err(ScatterPersistenceError::Deletions(
            ScatterDeletionError::KeyBytesBudgetExceeded {
                required: size_of::<ScatterCandidateKey>(),
                limit: size_of::<ScatterCandidateKey>() - 1
            }
        ))
    );
    limits.deletions.max_key_bytes += 1;
    assert_eq!(
        decode_scatter_deletions(&bytes, limits)
            .unwrap()
            .allocated_key_bytes(),
        size_of::<ScatterCandidateKey>()
    );
}

#[test]
fn hostile_declared_counts_fail_without_reservation() {
    let mut bytes = encode_scatter_deletions(&set(vec![]), 16).unwrap();
    bytes[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
    assert_eq!(
        decode_scatter_deletions(&bytes, budget()),
        Err(ScatterPersistenceError::ArithmeticOverflow)
    );
    bytes[8..16].copy_from_slice(&1000_u64.to_le_bytes());
    assert_eq!(
        decode_scatter_deletions(&bytes, budget()),
        Err(ScatterPersistenceError::InvalidLength {
            expected: 52016,
            actual: 16
        })
    );
}

#[test]
fn encoded_field_is_scoped_to_both_author_node_and_field() {
    let mut timeline = Timeline::new();
    let owner = binding();
    let deleted = set(vec![key(1)]);
    commit(&mut timeline, 1, owner, &deleted);
    assert_eq!(
        owner.read(timeline.state(), budget()).unwrap().deletions(),
        &deleted
    );
    for other in [
        ScatterDeletionBinding {
            object: PersistentId(42),
            ..owner
        },
        ScatterDeletionBinding {
            field: FieldId(932),
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
            other.snapshot(deleted.clone()),
            owner.snapshot(deleted.clone())
        );
    }
}

#[test]
fn malformed_or_wrong_type_bound_fields_leave_authority_and_old_snapshot_unchanged() {
    let owner = binding();
    let mut timeline = Timeline::new();
    commit(&mut timeline, 1, owner, &set(vec![key(1)]));
    let old = owner.read(timeline.state(), budget()).unwrap();
    for (id, value, error) in [
        (
            2,
            OverrideValue::Boolean(false),
            ScatterPersistenceError::UnexpectedValueType,
        ),
        (
            3,
            OverrideValue::Bytes(vec![0; 16]),
            ScatterPersistenceError::InvalidMagic,
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
fn replacement_undo_redo_restore_actual_previous_snapshot() {
    let owner = binding();
    let mut timeline = Timeline::new();
    let first = set(vec![key(0)]);
    let second = first
        .with_deleted(key(1), true, budget().deletions)
        .unwrap();
    commit(&mut timeline, 1, owner, &first);
    commit(&mut timeline, 2, owner, &second);
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
        &second
    );
    assert_eq!(timeline.log().len(), 4);
}

#[test]
fn clearing_is_undoable_and_undoing_first_write_removes_field() {
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
    commit(&mut timeline, 4, owner, &set(vec![]));
    assert_eq!(timeline.state().get(owner.object, owner.field), None);
    timeline
        .undo(TransactionId(5), ActorId(3), LogicalClock(5))
        .unwrap();
    assert_eq!(
        owner.read(timeline.state(), budget()).unwrap().deletions(),
        &deleted
    );
}

#[test]
fn snapshot_codec_and_log_suffix_replay_preserve_deletions() {
    let owner = binding();
    let mut timeline = Timeline::new();
    commit(&mut timeline, 1, owner, &set(vec![key(0)]));
    let checkpoint = decode_snapshot(&encode_snapshot(&timeline.snapshot()).unwrap()).unwrap();
    let final_set = set(vec![key(0), key(1)]);
    commit(&mut timeline, 2, owner, &final_set);
    let state = Timeline::replay(&checkpoint, timeline.log()).unwrap();
    assert_eq!(
        owner.read(&state, budget()).unwrap().deletions(),
        &final_set
    );
}

#[test]
fn crdt_is_deliberately_whole_field_lww_not_per_key_union() {
    let mut left = Timeline::new();
    let mut right = Timeline::new();
    let a = commit(&mut left, 1, binding(), &set(vec![key(0)]));
    let b = commit(&mut right, 2, binding(), &set(vec![key(1)]));
    let mut merged = CrdtState::new();
    merged.merge([&a, &b]).unwrap();
    let value = merged.get(binding().object, binding().field).unwrap();
    let OverrideValue::Bytes(bytes) = value else {
        panic!("deletion blob")
    };
    assert_eq!(
        decode_scatter_deletions(bytes, budget()).unwrap().keys(),
        &[key(1)]
    );
}

#[test]
fn regenerated_source_compacts_in_place_and_retains_full_cache_dependency() {
    let original = source(region());
    let delete = set(original
        .points
        .iter()
        .step_by(2)
        .map(|point| point.key)
        .collect());
    let upstream = original.cache_key;
    let capacity = original.points.capacity();
    let ptr = original.points.as_ptr();
    let expected: Vec<_> = original
        .points
        .iter()
        .copied()
        .filter(|point| !delete.is_deleted(point.key))
        .collect();
    let snapshot = binding().snapshot(delete);
    let filtered = snapshot.filter_source(original);
    assert_eq!(filtered.source.points, expected);
    assert_eq!(filtered.source.points.capacity(), capacity);
    assert_eq!(filtered.source.points.as_ptr(), ptr);
    assert_eq!(filtered.source.cache_key, upstream);
    assert_eq!(filtered.snapshot, snapshot);
    assert_eq!(
        snapshot.filter_source(source(region())).source.points,
        expected
    );
}

#[test]
fn author_field_compaction_keeps_attributes_and_binding_aligned() {
    let config = ScatterConfig {
        seed: 17,
        version: 1,
        salt: 2,
        grid_spacing_mm: 1000,
        min_distance_mm: 600,
    };
    let field = ScatterDensityField::new(
        ScatterFieldConfig {
            seed: 9,
            version: 1,
            base_density: u16::MAX,
            base_species: 23,
            noise: None,
        },
        vec![],
        ScatterFieldBuildBudget {
            max_brushes: 0,
            max_brush_bytes: 0,
        },
    )
    .unwrap();
    let output = scatter_with_density_field(
        config,
        region(),
        ScatterBudget {
            max_candidate_evaluations: 1_000_000,
            max_output_points: 10_000,
            max_output_bytes: 1_000_000,
        },
        &field,
        FieldScatterBudget {
            max_field_work: 100_000,
            max_output_bytes: 1_000_000,
        },
    )
    .unwrap();
    let removed_key = output.points[0].key;
    let points_ptr = output.points.as_ptr();
    let attrs_ptr = output.attributes.as_ptr();
    let filtered = binding()
        .snapshot(set(vec![removed_key]))
        .filter_field(output)
        .unwrap();
    assert_eq!(filtered.removed, 1);
    assert_eq!(filtered.source.points.as_ptr(), points_ptr);
    assert_eq!(filtered.source.attributes.as_ptr(), attrs_ptr);
    assert!(filtered.source.points.iter().all(|p| p.key != removed_key));
    assert!(filtered.source.attributes.iter().all(|a| a.species == 23));
    assert_eq!(
        filtered.source.points.len(),
        filtered.source.attributes.len()
    );
    assert_eq!(filtered.snapshot.binding(), binding());
}

#[test]
fn malformed_field_alignment_is_rejected() {
    let config = ScatterConfig {
        seed: 1,
        version: 1,
        salt: 1,
        grid_spacing_mm: 1000,
        min_distance_mm: 0,
    };
    let field = ScatterDensityField::new(
        ScatterFieldConfig {
            seed: 1,
            version: 1,
            base_density: u16::MAX,
            base_species: 1,
            noise: None,
        },
        vec![],
        ScatterFieldBuildBudget {
            max_brushes: 0,
            max_brush_bytes: 0,
        },
    )
    .unwrap();
    let mut output = scatter_with_density_field(
        config,
        region(),
        ScatterBudget {
            max_candidate_evaluations: 100_000,
            max_output_points: 1000,
            max_output_bytes: 100_000,
        },
        &field,
        FieldScatterBudget {
            max_field_work: 100_000,
            max_output_bytes: 100_000,
        },
    )
    .unwrap();
    output.attributes.pop();
    assert!(matches!(
        binding().snapshot(set(vec![])).filter_field(output),
        Err(ScatterPersistenceError::Deletions(
            ScatterDeletionError::MisalignedAttributes
        ))
    ));
}

#[cfg(feature = "std")]
#[test]
fn physical_journal_reopen_replay_filters_regenerated_scatter_and_records_undo_redo() {
    use prism_world_persist::FileJournal;
    use std::{
        fs,
        time::{SystemTime, UNIX_EPOCH},
    };
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(alloc::format!(
        "prism-scatter-deletions-{}-{nonce}.journal",
        std::process::id()
    ));
    struct RemoveOnDrop(std::path::PathBuf);
    impl Drop for RemoveOnDrop {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.0);
        }
    }
    let _cleanup = RemoveOnDrop(path.clone());
    let original = source(region());
    let deleted = set(original.points.iter().step_by(3).map(|p| p.key).collect());
    let mut timeline = Timeline::new();
    {
        let mut journal = FileJournal::open(&path).unwrap();
        journal
            .append(commit(&mut timeline, 1, binding(), &deleted))
            .unwrap();
    }
    let recovered = {
        let journal = FileJournal::open(&path).unwrap();
        assert_eq!(journal.recovery(), None);
        Timeline::replay(&Snapshot::new(0, WorldState::new()), journal.log()).unwrap()
    };
    let snapshot = binding().read(&recovered, budget()).unwrap();
    let expected: Vec<_> = original
        .points
        .iter()
        .copied()
        .filter(|p| !deleted.is_deleted(p.key))
        .collect();
    // Destroy all generated data; persistence does not store a Vec index or proxy.
    drop(original);
    let filtered = snapshot.filter_source(source(region()));
    assert_eq!(filtered.source.points, expected);
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
        let replay = Timeline::replay(&Snapshot::new(0, WorldState::new()), journal.log()).unwrap();
        assert_eq!(
            binding()
                .read(&replay, budget())
                .unwrap()
                .filter_source(source(region()))
                .removed,
            0
        );
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
    let journal = FileJournal::open(&path).unwrap();
    let replay = Timeline::replay(&Snapshot::new(0, WorldState::new()), journal.log()).unwrap();
    assert_eq!(
        binding()
            .read(&replay, budget())
            .unwrap()
            .filter_source(source(region()))
            .source
            .points,
        expected
    );
    assert_eq!(journal.log().len(), 3);
}
