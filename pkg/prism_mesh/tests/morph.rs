//! Contract tests for the standalone morph-target data module.

extern crate alloc;

use alloc::vec;

#[path = "../src/morph.rs"]
mod morph;

use morph::{MorphDelta, MorphError, MorphTarget, MorphTargetData};

fn delta(x: f32, y: f32, z: f32) -> MorphDelta {
    MorphDelta::position([x, y, z])
}

#[test]
fn dense_targets_cover_every_vertex_and_support_lookup() {
    let target = MorphTarget::new_dense("smile", vec![delta(0.0, 0.0, 0.0), delta(0.1, 0.0, 0.0)])
        .expect("valid dense target");

    assert_eq!(target.name(), "smile");
    assert_eq!(target.vertex_count(), 2);
    assert_eq!(target.stored_delta_count(), 2);
    assert_eq!(target.delta_at(1), Some(delta(0.1, 0.0, 0.0)));
    assert_eq!(target.delta_at(2), None);
    assert!(matches!(target.data(), MorphTargetData::Dense(_)));
    assert!(target.validate().is_ok());
}

#[test]
fn sparse_targets_require_in_range_strictly_increasing_indices() {
    let target = MorphTarget::new_sparse(
        "blink",
        5,
        vec![1, 4],
        vec![delta(0.0, 0.2, 0.0), delta(0.0, 0.4, 0.0)],
    )
    .expect("valid sparse target");

    assert_eq!(target.stored_delta_count(), 2);
    assert_eq!(target.delta_at(0), None);
    assert_eq!(target.delta_at(1), Some(delta(0.0, 0.2, 0.0)));
    assert_eq!(target.delta_at(4), Some(delta(0.0, 0.4, 0.0)));
    assert_eq!(target.delta_at(5), None);

    assert!(matches!(
        MorphTarget::new_sparse("duplicate", 5, vec![2, 2], vec![delta(0.0, 0.0, 0.0); 2]),
        Err(MorphError::SparseIndicesNotStrictlyIncreasing {
            previous: 2,
            current: 2,
            ..
        })
    ));
    assert!(matches!(
        MorphTarget::new_sparse("unsorted", 5, vec![3, 1], vec![delta(0.0, 0.0, 0.0); 2]),
        Err(MorphError::SparseIndicesNotStrictlyIncreasing {
            previous: 3,
            current: 1,
            ..
        })
    ));
}

#[test]
fn sparse_indices_and_deltas_must_have_matching_lengths_and_range() {
    assert_eq!(
        MorphTarget::new_sparse("mismatch", 3, vec![0], vec![]),
        Err(MorphError::SparseLengthMismatch {
            indices: 1,
            deltas: 0,
        })
    );
    assert_eq!(
        MorphTarget::new_sparse("range", 3, vec![3], vec![delta(0.0, 0.0, 0.0)]),
        Err(MorphError::SparseIndexOutOfRange {
            entry: 0,
            index: 3,
            vertex_count: 3,
        })
    );
}

#[test]
fn every_position_component_must_be_finite() {
    assert_eq!(
        MorphTarget::new_dense("nan", vec![delta(f32::NAN, 0.0, 0.0)]),
        Err(MorphError::NonFiniteDelta {
            vertex_index: 0,
            component: 0,
        })
    );
    assert_eq!(
        MorphTarget::new_sparse("inf", 2, vec![1], vec![delta(0.0, 0.0, f32::INFINITY)]),
        Err(MorphError::NonFiniteDelta {
            vertex_index: 1,
            component: 2,
        })
    );
}
