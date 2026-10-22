#![allow(dead_code)]
#![allow(missing_docs)]

extern crate alloc;

#[path = "../src/error.rs"]
mod error;
#[path = "../src/limits.rs"]
mod limits;
#[path = "../src/submesh.rs"]
mod submesh;
#[path = "../src/topology.rs"]
mod topology;

use core::ops::Range;

use error::MeshError;
use limits::MeshLimits;
use prism_render_driver::{Limits, PrimitiveTopology};
use submesh::{validate_submeshes, validate_submeshes_with_limits, SubMesh};
use topology::{elements_per_primitive, primitive_count, PrimitiveTopologyExt};

#[test]
fn topology_helpers_match_the_rhi_enum() {
    assert_eq!(primitive_count(PrimitiveTopology::PointList, 4), Some(4));
    assert_eq!(primitive_count(PrimitiveTopology::LineList, 4), Some(2));
    assert_eq!(primitive_count(PrimitiveTopology::LineList, 3), None);
    assert_eq!(primitive_count(PrimitiveTopology::LineStrip, 4), Some(3));
    assert_eq!(primitive_count(PrimitiveTopology::TriangleList, 6), Some(2));
    assert_eq!(
        primitive_count(PrimitiveTopology::TriangleStrip, 4),
        Some(2)
    );

    assert!(PrimitiveTopology::TriangleStrip.is_strip());
    assert!(PrimitiveTopology::TriangleList.is_list());
    assert!(PrimitiveTopology::TriangleList.is_triangle());
    assert_eq!(PrimitiveTopology::LineList.minimum_element_count(), 2);
    assert_eq!(elements_per_primitive(PrimitiveTopology::TriangleStrip), 3);
    assert!(PrimitiveTopology::PointList.supports_non_indexed());
}

#[test]
fn topology_validation_rejects_only_incomplete_lists() {
    assert_eq!(
        PrimitiveTopology::TriangleList.validate_element_count(4),
        Err(MeshError::TopologyElementCount {
            topology: PrimitiveTopology::TriangleList,
            element_count: 4,
        })
    );
    assert!(PrimitiveTopology::TriangleStrip
        .validate_element_count(4)
        .is_ok());
}

#[test]
fn submesh_ranges_are_bounded_aligned_and_non_overlapping() {
    let first = SubMesh::new(0, 0..3);
    let second = SubMesh::new(1, 3..6);
    assert!(validate_submeshes(
        &[first.clone(), second.clone()],
        6,
        PrimitiveTopology::TriangleList
    )
    .is_ok());

    assert_eq!(
        SubMesh::new(0, 0..4).validate(6, PrimitiveTopology::TriangleList),
        Err(MeshError::TopologyElementCount {
            topology: PrimitiveTopology::TriangleList,
            element_count: 4,
        })
    );
    assert_eq!(
        SubMesh::new(0, 1..4).validate(6, PrimitiveTopology::TriangleList),
        Err(MeshError::SubMeshIndexRangeMisaligned {
            start: 1,
            topology: PrimitiveTopology::TriangleList,
            elements_per_primitive: 3,
        })
    );
    assert_eq!(
        SubMesh::new(0, 1..3).validate(4, PrimitiveTopology::LineList),
        Err(MeshError::SubMeshIndexRangeMisaligned {
            start: 1,
            topology: PrimitiveTopology::LineList,
            elements_per_primitive: 2,
        })
    );
    assert_eq!(
        SubMesh::new(0, 5..7).validate(6, PrimitiveTopology::TriangleList),
        Err(MeshError::SubMeshIndexRangeOutOfBounds {
            start: 5,
            end: 7,
            index_count: 6,
        })
    );
    assert_eq!(
        validate_submeshes(
            &[SubMesh::new(0, 0..6), SubMesh::new(1, 3..6)],
            9,
            PrimitiveTopology::TriangleList
        ),
        Err(MeshError::SubMeshIndexRangesOverlap {
            first: 0,
            second: 1,
        })
    );
}

#[test]
fn cluster_ranges_and_material_slots_are_checked() {
    let limits = MeshLimits::new(100, 4, 2).with_material_slots(2);
    let submesh = SubMesh::new(1, 0..3).with_cluster_range(Range { start: 2, end: 4 });
    assert!(validate_submeshes_with_limits(
        &[submesh],
        3,
        PrimitiveTopology::TriangleList,
        &limits
    )
    .is_ok());

    assert_eq!(
        SubMesh::new(2, 0..3).validate_with_limits(3, PrimitiveTopology::TriangleList, &limits),
        Err(MeshError::MaterialSlotOutOfRange {
            slot: 2,
            maximum: 2,
        })
    );
    assert_eq!(
        SubMesh::new(0, 0..3)
            .with_cluster_range(4..4)
            .validate(3, PrimitiveTopology::TriangleList),
        Err(MeshError::EmptySubMeshClusterRange)
    );
}

#[test]
fn mesh_limits_use_rhi_baseline_and_report_typed_failures() {
    let baseline = MeshLimits::from_render_limits(Limits::baseline());
    assert_eq!(baseline.max_attributes.get(), 16);
    assert_eq!(baseline.max_streams.get(), 8);
    assert!(baseline.validate_counts(10, 16, 8).is_ok());
    assert_eq!(
        baseline.validate_counts(10, 17, 8),
        Err(MeshError::AttributeLimitExceeded {
            actual: 17,
            maximum: 16,
        })
    );
    assert_eq!(
        baseline.validate_counts(10, 16, 9),
        Err(MeshError::StreamLimitExceeded {
            actual: 9,
            maximum: 8,
        })
    );

    let small = MeshLimits::new(3, 2, 1);
    assert_eq!(
        small.validate_counts(4, 1, 1),
        Err(MeshError::VertexLimitExceeded {
            actual: 4,
            maximum: 3,
        })
    );
}
