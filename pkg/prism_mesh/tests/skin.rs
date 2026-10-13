//! Contract tests for the standalone skinning data module.

extern crate alloc;

use alloc::vec;

#[path = "../src/skin.rs"]
mod skin;

use skin::{SkinError, SkinInfluence, SkinningData};

fn influence(joint_index: u32, weight: f32) -> SkinInfluence {
    SkinInfluence {
        joint_index,
        weight,
    }
}

#[test]
fn valid_influences_are_queryable_and_preserve_authored_weights() {
    let data = SkinningData::new(
        3,
        4,
        vec![
            vec![influence(0, 0.75), influence(1, 0.25)],
            vec![],
            vec![influence(3, 2.0)],
        ],
    )
    .expect("valid skinning data");

    assert_eq!(data.vertex_count(), 3);
    assert_eq!(data.joint_count(), 4);
    assert_eq!(data.max_influences(), 2);
    assert_eq!(data.influences(0).unwrap().len(), 2);
    assert_eq!(data.influences(3), None);
    assert_eq!(data.weight_sum(0), Some(1.0));
    assert_eq!(data.weight_sum(1), Some(0.0));
    assert!(data.validate_with_max_influences(2).is_ok());
}

#[test]
fn vertex_count_and_joint_range_are_checked() {
    assert_eq!(
        SkinningData::new(2, 2, vec![vec![influence(0, 1.0)]]),
        Err(SkinError::VertexCountMismatch {
            expected: 2,
            found: 1,
        })
    );

    assert_eq!(
        SkinningData::new(1, 2, vec![vec![influence(2, 1.0)]]),
        Err(SkinError::JointIndexOutOfRange {
            vertex: 0,
            influence: 0,
            joint_index: 2,
            joint_count: 2,
        })
    );
}

#[test]
fn weights_must_be_finite_non_negative_and_unambiguous() {
    assert!(matches!(
        SkinningData::new(1, 1, vec![vec![influence(0, f32::NAN)]]),
        Err(SkinError::NonFiniteWeight { .. })
    ));
    assert!(matches!(
        SkinningData::new(1, 1, vec![vec![influence(0, -0.1)]]),
        Err(SkinError::NegativeWeight { .. })
    ));
    assert_eq!(
        SkinningData::new(1, 2, vec![vec![influence(0, 0.5), influence(0, 0.5)]]),
        Err(SkinError::DuplicateJoint {
            vertex: 0,
            joint_index: 0,
        })
    );
}

#[test]
fn a_profile_can_reject_excess_influences_without_changing_asset_data() {
    let data = SkinningData::new(
        1,
        4,
        vec![vec![
            influence(0, 0.25),
            influence(1, 0.25),
            influence(2, 0.25),
            influence(3, 0.25),
        ]],
    )
    .expect("the authored list is valid");

    assert_eq!(
        data.validate_with_max_influences(3),
        Err(SkinError::TooManyInfluences {
            vertex: 0,
            found: 4,
            max: 3,
        })
    );
    assert_eq!(data.max_influences(), 4);
}

#[test]
fn finite_weights_that_are_not_normalized_are_allowed() {
    let data = SkinningData::new(1, 2, vec![vec![influence(0, 2.0), influence(1, 3.0)]])
        .expect("normalization is a construction-stage policy, not a validity requirement");
    assert_eq!(data.weight_sum(0), Some(5.0));
}
