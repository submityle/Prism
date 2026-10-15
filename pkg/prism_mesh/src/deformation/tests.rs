use alloc::vec;

use super::*;
use crate::{MorphDelta, SkinInfluence};

fn dense_target(deltas: &[[f32; 3]]) -> MorphTarget {
    MorphTarget::new_dense(
        "shape",
        deltas.iter().copied().map(MorphDelta::position).collect(),
    )
    .unwrap()
}

#[test]
fn evaluation_matches_morph_skin_and_external_correction_order() {
    let target = dense_target(&[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    let skin = SkinningData::new(
        2,
        1,
        vec![
            vec![SkinInfluence {
                joint_index: 0,
                weight: 2.0,
            }],
            vec![],
        ],
    )
    .unwrap();
    let mesh = DeformationMesh::new(
        &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        Some(&[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
        &[target],
        Some(&skin),
        4,
    )
    .unwrap();
    let joint = JointTransform::new([
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 2.0],
        [0.0, 0.0, 1.0, 3.0],
    ])
    .unwrap();
    let output = mesh
        .evaluate(DeformationPose {
            morph_weights: &[1.0],
            joints: &[joint],
            corrections: Some(&[
                DeformationCorrection {
                    position: [4.0, 0.0, 0.0, 0.0],
                    normal: [0.0, 1.0, 0.0, 0.0],
                },
                DeformationCorrection::default(),
            ]),
        })
        .unwrap();

    assert_eq!(output[0].position, [7.0, 2.0, 3.0, 1.0]);
    let expected = 2.0_f32.sqrt().recip();
    assert!((output[0].normal[1] - expected).abs() < 1.0e-6);
    assert!((output[0].normal[2] - expected).abs() < 1.0e-6);
    assert_eq!(output[1].position, [0.0, 1.0, 0.0, 1.0]);
    assert_eq!(output[1].normal, [0.0, 1.0, 0.0, 0.0]);
}

#[test]
fn sparse_morphs_are_transposed_to_vertex_local_spans() {
    let target = MorphTarget::new_sparse(
        "sparse",
        3,
        vec![1],
        vec![MorphDelta::position([0.0, 2.0, 0.0])],
    )
    .unwrap();
    let mesh = DeformationMesh::new(&[[0.0, 0.0, 0.0]; 3], None, &[target], None, 8).unwrap();

    assert_eq!(mesh.morphs().len(), 1);
    assert_eq!(mesh.vertices()[0].spans, [0, 0, 0, 0]);
    assert_eq!(mesh.vertices()[1].spans, [0, 1, 0, 0]);
    assert_eq!(mesh.vertices()[2].spans, [1, 0, 0, 0]);
    let output = mesh
        .evaluate(DeformationPose {
            morph_weights: &[0.5],
            joints: &[],
            corrections: None,
        })
        .unwrap();
    assert_eq!(output[0].position, [0.0, 0.0, 0.0, 1.0]);
    assert_eq!(output[1].position, [0.0, 1.0, 0.0, 1.0]);
}

#[test]
fn inverse_transpose_normals_handle_non_uniform_scale() {
    let target = dense_target(&[[0.0, 0.0, 0.0]]);
    let skin = SkinningData::new(
        1,
        1,
        vec![vec![SkinInfluence {
            joint_index: 0,
            weight: 1.0,
        }]],
    )
    .unwrap();
    let mesh = DeformationMesh::new(
        &[[0.0, 0.0, 0.0]],
        Some(&[[1.0, 1.0, 0.0]]),
        &[target],
        Some(&skin),
        4,
    )
    .unwrap();
    let output = mesh
        .evaluate(DeformationPose {
            morph_weights: &[0.0],
            joints: &[JointTransform::new([
                [2.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ])
            .unwrap()],
            corrections: None,
        })
        .unwrap();
    assert!((output[0].normal[0] - 0.4472136).abs() < 1.0e-5);
    assert!((output[0].normal[1] - 0.8944272).abs() < 1.0e-5);
}

#[test]
fn validation_rejects_invalid_profiles_and_pose_data() {
    let target = dense_target(&[[0.0, 0.0, 0.0]]);
    assert!(matches!(
        DeformationMesh::new(&[[0.0, 0.0, 0.0]], None, &[target.clone()], None, 6),
        Err(DeformationError::InvalidProfile)
    ));
    assert!(matches!(
        JointTransform::new([[0.0; 4]; 3]),
        Err(DeformationError::SingularJoint)
    ));
    assert!(matches!(
        DeformationMesh::new(&[[f32::NAN, 0.0, 0.0]], None, &[target], None, 4),
        Err(DeformationError::NonFinite {
            field: "positions",
            ..
        })
    ));

    let mesh = DeformationMesh::new(&[[0.0, 0.0, 0.0]], None, &[], None, 4).unwrap();
    assert_eq!(
        mesh.validate_pose(DeformationPose {
            morph_weights: &[],
            joints: &[],
            corrections: Some(&[DeformationCorrection {
                position: [0.0, 0.0, 0.0, 1.0],
                normal: [0.0; 4],
            }]),
        }),
        Err(DeformationError::ReservedCorrection)
    );
}

#[test]
fn shared_abi_has_exact_storage_strides() {
    assert_eq!(size_of::<DeformationVertex>(), 48);
    assert_eq!(size_of::<MorphContribution>(), 16);
    assert_eq!(size_of::<JointWeight>(), 8);
    assert_eq!(size_of::<JointTransform>(), 96);
    assert_eq!(size_of::<DeformationCorrection>(), 32);
    assert_eq!(size_of::<DeformedVertex>(), 32);
}

#[test]
fn four_and_eight_profiles_never_truncate() {
    let influences: Vec<_> = (0..8)
        .map(|joint_index| SkinInfluence {
            joint_index,
            weight: 1.0,
        })
        .collect();
    let skin = SkinningData::new(1, 8, vec![influences]).unwrap();
    assert!(matches!(
        DeformationMesh::new(&[[0.0; 3]], None, &[], Some(&skin), 4),
        Err(DeformationError::Skin(_))
    ));
    let mesh = DeformationMesh::new(&[[0.0; 3]], None, &[], Some(&skin), 8).unwrap();
    let joints: Vec<_> = (0..8)
        .map(|i| {
            JointTransform::new([
                [1.0, 0.0, 0.0, i as f32],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ])
            .unwrap()
        })
        .collect();
    assert_eq!(
        mesh.evaluate(DeformationPose {
            morph_weights: &[],
            joints: &joints,
            corrections: None
        })
        .unwrap()[0]
            .position,
        [3.5, 0.0, 0.0, 1.0]
    );
}

#[test]
fn zero_weight_skin_keeps_morphed_bind_pose() {
    let target = dense_target(&[[2.0, 0.0, 0.0]]);
    let skin = SkinningData::new(
        1,
        1,
        vec![vec![SkinInfluence {
            joint_index: 0,
            weight: 0.0,
        }]],
    )
    .unwrap();
    let mesh = DeformationMesh::new(&[[1.0; 3]], None, &[target], Some(&skin), 4).unwrap();
    let output = mesh
        .evaluate(DeformationPose {
            morph_weights: &[0.5],
            joints: &[JointTransform::identity()],
            corrections: None,
        })
        .unwrap();
    assert!(mesh.influences().is_empty());
    assert_eq!(output[0].position, [2.0, 1.0, 1.0, 1.0]);
    assert_eq!(output[0].normal, [0.0; 4]);
}

#[test]
fn forged_joint_bytes_cannot_skip_validation() {
    let skin = SkinningData::new(1, 1, vec![vec![]]).unwrap();
    let mesh = DeformationMesh::new(&[[0.0; 3]], None, &[], Some(&skin), 4).unwrap();
    let pose = |joint| {
        mesh.evaluate(DeformationPose {
            morph_weights: &[],
            joints: &[joint],
            corrections: None,
        })
    };
    assert_eq!(
        pose(JointTransform::zeroed()),
        Err(DeformationError::SingularJoint)
    );
    let mut forged = JointTransform::identity();
    forged.rows[3][0] = 42.0;
    assert_eq!(pose(forged), Err(DeformationError::InvalidJoint));
}

#[test]
fn empty_mesh_and_extreme_normal_scales_are_well_defined() {
    let empty = DeformationMesh::new(&[], None, &[], None, 4).unwrap();
    assert!(empty
        .evaluate(DeformationPose {
            morph_weights: &[],
            joints: &[],
            corrections: None
        })
        .unwrap()
        .is_empty());
    for value in [1e-35, 1e35] {
        let mesh =
            DeformationMesh::new(&[[0.0; 3]], Some(&[[value, 0.0, 0.0]]), &[], None, 4).unwrap();
        assert_eq!(
            mesh.evaluate(DeformationPose {
                morph_weights: &[],
                joints: &[],
                corrections: None
            })
            .unwrap()[0]
                .normal,
            [1.0, 0.0, 0.0, 0.0]
        );
        assert!(JointTransform::new([
            [value, 0.0, 0.0, 0.0],
            [0.0, value, 0.0, 0.0],
            [0.0, 0.0, value, 0.0]
        ])
        .is_ok());
    }
}

#[test]
fn finite_inputs_that_can_overflow_gpu_are_rejected_before_evaluation() {
    let mesh = DeformationMesh::new(
        &[[1.0e30, 0.0, 0.0]],
        None,
        &[dense_target(&[[1.0e30, 0.0, 0.0]])],
        None,
        4,
    )
    .unwrap();
    let pose = DeformationPose {
        morph_weights: &[1.0e30],
        joints: &[],
        corrections: None,
    };
    assert_eq!(
        mesh.validate_pose(pose),
        Err(DeformationError::ArithmeticRange)
    );
    assert_eq!(mesh.evaluate(pose), Err(DeformationError::ArithmeticRange));

    let skin = SkinningData::new(
        1,
        1,
        vec![vec![SkinInfluence {
            joint_index: 0,
            weight: 1.0,
        }]],
    )
    .unwrap();
    let mesh = DeformationMesh::new(&[[1.0e30, 0.0, 0.0]], None, &[], Some(&skin), 4).unwrap();
    let joint = JointTransform::new([
        [1.0e30, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ])
    .unwrap();
    assert_eq!(
        mesh.validate_pose(DeformationPose {
            morph_weights: &[],
            joints: &[joint],
            corrections: None
        }),
        Err(DeformationError::ArithmeticRange)
    );
}
