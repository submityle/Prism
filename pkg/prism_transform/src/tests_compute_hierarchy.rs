//! Oracle + parity tests for the §24.7 compute-hierarchy data model.

use alloc::vec::Vec;

use prism_math::{Quat, Vec3};

use crate::compute_hierarchy::{propagate_by_levels, ComputeHierarchyInput, LevelSchedule};
use crate::gpu_upload::MatrixLayout;
use crate::hierarchy::{Hierarchy, NodeId};
use crate::propagation::propagate;
use crate::{GlobalTransform, Transform};

/// Build the forest:
/// ```text
/// 0            5
/// ├─1          └─6
/// │ ├─3
/// │ └─4
/// └─2
/// ```
/// Depths: 0,5 -> 0; 1,2,6 -> 1; 3,4 -> 2.
fn two_root_forest() -> Hierarchy {
    let parents = [
        None,                   // 0 root
        Some(NodeId::new(0)),   // 1 <- 0
        Some(NodeId::new(0)),   // 2 <- 0
        Some(NodeId::new(1)),   // 3 <- 1
        Some(NodeId::new(1)),   // 4 <- 1
        None,                   // 5 root
        Some(NodeId::new(5)),   // 6 <- 5
    ];
    Hierarchy::from_parents(&parents).expect("valid forest")
}

#[test]
fn schedule_depths_levels_and_deterministic_order() {
    let hier = two_root_forest();
    let schedule = LevelSchedule::build(&hier).expect("acyclic");

    assert_eq!(schedule.len(), 7);
    assert_eq!(schedule.level_count(), 3);
    assert_eq!(schedule.max_depth(), Some(2));

    // Hand-computed depths.
    let expected_depth = [0u32, 1, 1, 2, 2, 0, 1];
    for (i, &d) in expected_depth.iter().enumerate() {
        assert_eq!(schedule.node_depth(NodeId::new(i as u32)), d, "node {i}");
    }

    // Levels, ascending within a level regardless of child insertion order.
    let level0: Vec<usize> = schedule.level(0).iter().map(|n| n.index()).collect();
    let level1: Vec<usize> = schedule.level(1).iter().map(|n| n.index()).collect();
    let level2: Vec<usize> = schedule.level(2).iter().map(|n| n.index()).collect();
    assert_eq!(level0, [0, 5]);
    assert_eq!(level1, [1, 2, 6]);
    assert_eq!(level2, [3, 4]);

    // Flattened dispatch order concatenates the levels.
    let order: Vec<usize> = schedule.order().iter().map(|n| n.index()).collect();
    assert_eq!(order, [0, 5, 1, 2, 6, 3, 4]);

    // Ranges slice the flattened order into the levels.
    let ranges = schedule.level_ranges();
    assert_eq!(ranges, &[0..2, 2..5, 5..7]);

    // Building twice is bit-identical (determinism).
    let again = LevelSchedule::build(&hier).expect("acyclic");
    assert_eq!(schedule.order(), again.order());
}

#[test]
fn empty_hierarchy_schedule() {
    let hier = Hierarchy::from_parents(&[]).expect("empty forest");
    let schedule = LevelSchedule::build(&hier).expect("acyclic");
    assert!(schedule.is_empty());
    assert_eq!(schedule.level_count(), 0);
    assert_eq!(schedule.max_depth(), None);
}

/// A varied local-transform set so composition exercises rotation + non-uniform
/// scale (not just translation).
fn varied_locals() -> Vec<Transform> {
    [
        Transform::from_xyz(1.0, 0.0, 0.0),
        Transform {
            translation: Vec3::new(0.0, 2.0, 0.0),
            rotation: Quat::from_rotation_z(core::f32::consts::FRAC_PI_2),
            scale: Vec3::new(2.0, 1.0, 1.0),
        },
        Transform {
            translation: Vec3::new(-1.0, 0.5, 3.0),
            rotation: Quat::from_rotation_y(0.75),
            scale: Vec3::new(1.0, 1.0, 0.5),
        },
        Transform::from_scale(Vec3::new(3.0, 3.0, 3.0)),
        Transform {
            translation: Vec3::new(0.25, -1.0, 2.0),
            rotation: Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), 1.1),
            scale: Vec3::new(1.5, 0.5, 2.0),
        },
        Transform::from_xyz(10.0, -10.0, 5.0),
        Transform {
            translation: Vec3::new(1.0, 1.0, 1.0),
            rotation: Quat::from_rotation_x(0.3),
            scale: Vec3::ONE,
        },
    ]
    .to_vec()
}

#[test]
fn level_propagation_matches_serial_bit_for_bit() {
    let hier = two_root_forest();
    let schedule = LevelSchedule::build(&hier).expect("acyclic");
    let locals = varied_locals();

    let mut serial = alloc::vec![GlobalTransform::IDENTITY; hier.len()];
    propagate(&hier, &locals, &mut serial).expect("serial");

    let mut by_levels = alloc::vec![GlobalTransform::IDENTITY; hier.len()];
    propagate_by_levels(&hier, &schedule, &locals, &mut by_levels).expect("by levels");

    assert_eq!(serial, by_levels, "level grouping must not change the result");
}

#[test]
fn level_propagation_length_mismatch() {
    let hier = two_root_forest();
    let schedule = LevelSchedule::build(&hier).expect("acyclic");
    let locals = varied_locals();
    let mut too_small = alloc::vec![GlobalTransform::IDENTITY; hier.len() - 1];
    assert!(propagate_by_levels(&hier, &schedule, &locals, &mut too_small).is_err());
}

#[test]
fn input_payload_parents_order_and_ranges() {
    let hier = two_root_forest();
    let schedule = LevelSchedule::build(&hier).expect("acyclic");
    let locals = varied_locals();

    let input = ComputeHierarchyInput::pack(&hier, &schedule, &locals, MatrixLayout::RowMajor3x4);

    assert_eq!(input.node_count(), 7);
    assert_eq!(input.level_count(), 3);
    // Roots are -1; others carry the parent index.
    assert_eq!(input.parents(), &[-1, 0, 0, 1, 1, -1, 5]);
    // Dispatch order mirrors the schedule.
    let order: Vec<u32> = schedule.order().iter().map(|n| n.index() as u32).collect();
    assert_eq!(input.dispatch_order(), order.as_slice());
    assert_eq!(input.level_ranges(), &[(0, 2), (2, 5), (5, 7)]);
    assert_eq!(input.matrix_stride(), 48);
    assert_eq!(input.local_matrices().len(), 7 * 48);
}

#[test]
fn input_payload_matrix_bytes_are_row_major() {
    // Node 0 is a pure translation (1,0,0): identity basis, translation in the
    // 4th column of each row.
    let hier = two_root_forest();
    let schedule = LevelSchedule::build(&hier).expect("acyclic");
    let locals = varied_locals();

    let input = ComputeHierarchyInput::pack(&hier, &schedule, &locals, MatrixLayout::RowMajor3x4);
    let bytes = &input.local_matrices()[0..48];
    let mut floats = [0.0f32; 12];
    for (f, chunk) in floats.iter_mut().zip(bytes.chunks_exact(4)) {
        *f = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    // rows: [bx.x, by.x, bz.x, t.x,  bx.y, by.y, bz.y, t.y,  bx.z, by.z, bz.z, t.z]
    assert_eq!(floats, [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
}

#[test]
fn input_payload_4x4_appends_identity_bottom_row() {
    let hier = two_root_forest();
    let schedule = LevelSchedule::build(&hier).expect("acyclic");
    let locals = varied_locals();

    let input = ComputeHierarchyInput::pack(&hier, &schedule, &locals, MatrixLayout::RowMajor4x4);
    assert_eq!(input.matrix_stride(), 64);
    let bytes = &input.local_matrices()[0..64];
    let mut floats = [0.0f32; 16];
    for (f, chunk) in floats.iter_mut().zip(bytes.chunks_exact(4)) {
        *f = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    // The implicit bottom row [0,0,0,1].
    assert_eq!(&floats[12..16], &[0.0, 0.0, 0.0, 1.0]);
}
