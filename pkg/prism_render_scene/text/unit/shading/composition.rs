//! Capacity/freshness regressions independent of a GPU or a real-sized allocation.

use super::*;

/// ABI strides and resource sizes must come from the production structs, not vec3 padding.
#[test]
fn plane_capacity_mapping_matches_production_abi() {
    let layout = ResponseCompositionLayout::new(UVec2::new(17, 9), 7, &Limits::default()).unwrap();
    assert_eq!(layout.size(), UVec2::new(17, 9));
    assert_eq!(layout.pixel_count(), 153);
    assert_eq!(layout.auxiliary_jobs(), 7);
    assert_eq!(CompositionPlane::Left.stride(), 64);
    assert_eq!(CompositionPlane::Right.stride(), 64);
    assert_eq!(CompositionPlane::Controls.stride(), 16);
    assert_eq!(CompositionPlane::Auxiliary.stride(), 64);
    assert_eq!(
        CompositionPlane::WaterCoverage.graph_name(),
        "water_coverage"
    );
    assert_eq!(CompositionPlane::WaterCoverage.stride(), 4);
    assert_eq!(layout.bytes(CompositionPlane::Left), 153 * 64);
    assert_eq!(layout.bytes(CompositionPlane::Right), 153 * 64);
    assert_eq!(layout.bytes(CompositionPlane::Controls), 153 * 16);
    assert_eq!(layout.bytes(CompositionPlane::Auxiliary), 7 * 64);
    assert_eq!(layout.bytes(CompositionPlane::WaterCoverage), 153 * 4);
    let no_aux = ResponseCompositionLayout::new(UVec2::ONE, 0, &Limits::default()).unwrap();
    assert_eq!(no_aux.bytes(CompositionPlane::Auxiliary), 0);
}

/// Reject dimensions and linear-index overflow before allocating or narrowing anything.
#[test]
fn invalid_extent_is_rejected_without_allocation() {
    let limits = Limits::default();
    assert_eq!(
        ResponseCompositionLayout::new(UVec2::ZERO, 0, &limits),
        Err(CompositionResourceError::EmptyExtent)
    );
    assert_eq!(
        ResponseCompositionLayout::new(UVec2::new(u32::MAX, 2), 0, &limits),
        Err(CompositionResourceError::PixelCountOverflow)
    );
    assert!(matches!(
        ResponseCompositionLayout::new(
            UVec2::new(limits.max_texture_dimension_2d + 1, 1),
            0,
            &limits
        ),
        Err(CompositionResourceError::ExtentLimit { .. })
    ));
}

/// Both allocation and binding limits are enforced, including equality at the bound.
#[test]
fn binding_and_allocation_limits_are_checked_exactly() {
    let mut limits = Limits::default();
    limits.max_storage_buffer_binding_size = 128;
    limits.max_buffer_size = 128;
    assert!(ResponseCompositionLayout::new(UVec2::new(2, 1), 2, &limits).is_ok());
    assert_eq!(
        ResponseCompositionLayout::new(UVec2::new(3, 1), 0, &limits),
        Err(CompositionResourceError::BufferLimit {
            plane: CompositionPlane::Left,
            requested: 192,
            limit: 128
        })
    );
    assert_eq!(
        ResponseCompositionLayout::new(UVec2::ONE, 3, &limits),
        Err(CompositionResourceError::BufferLimit {
            plane: CompositionPlane::Auxiliary,
            requested: 192,
            limit: 128
        })
    );
    limits.max_buffer_size = 63;
    assert!(matches!(
        ResponseCompositionLayout::new(UVec2::ONE, 0, &limits),
        Err(CompositionResourceError::BufferLimit { limit: 63, .. })
    ));
}

/// The parent's X-only kernel must not be tiled in Y and accidentally reblend indices.
#[test]
fn linear_and_spatial_dispatch_bounds_are_distinct() {
    let mut limits = Limits::default();
    limits.max_compute_workgroups_per_dimension = 2;
    assert!(ResponseCompositionLayout::new(UVec2::splat(8), 128, &limits).is_ok());
    assert!(matches!(
        ResponseCompositionLayout::new(UVec2::splat(12), 0, &limits),
        Err(CompositionResourceError::DispatchLimit {
            plane: CompositionPlane::Left,
            requested: 3,
            limit: 2
        })
    ));
    assert!(matches!(
        ResponseCompositionLayout::new(UVec2::ONE, 129, &limits),
        Err(CompositionResourceError::DispatchLimit {
            plane: CompositionPlane::Auxiliary,
            requested: 3,
            limit: 2
        })
    ));
    // Only one linear group, but three spatial groups in Y for the HDR bridge.
    assert!(matches!(
        ResponseCompositionLayout::new(UVec2::new(1, 17), 0, &limits),
        Err(CompositionResourceError::DispatchLimit { requested: 3, .. })
    ));
}

/// Unsupported invocation/workgroup shapes are rejected even for a single pixel.
#[test]
fn workgroup_shape_is_an_admission_precondition() {
    let mut limits = Limits::default();
    limits.max_compute_workgroup_size_y = 7;
    assert_eq!(
        ResponseCompositionLayout::new(UVec2::ONE, 0, &limits),
        Err(CompositionResourceError::WorkgroupShape)
    );
    limits.max_compute_workgroup_size_y = 8;
    limits.max_compute_workgroup_size_x = 63;
    assert_eq!(
        ResponseCompositionLayout::new(UVec2::ONE, 0, &limits),
        Err(CompositionResourceError::WorkgroupShape)
    );
    limits.max_compute_workgroup_size_x = 64;
    limits.max_compute_invocations_per_workgroup = 63;
    assert_eq!(
        ResponseCompositionLayout::new(UVec2::ONE, 0, &limits),
        Err(CompositionResourceError::WorkgroupShape)
    );
}

/// Stale and duplicate publications cannot replay an in-place blend, including wraparound.
#[test]
fn publication_is_current_frame_and_consumed_once() {
    let mut publication = Publication::default();
    assert!(!publication.take(0));
    publication.0 = Some(7);
    assert!(publication.take(7));
    assert!(!publication.take(7));
    publication.0 = Some(7);
    assert!(!publication.take(8));
    assert!(!publication.take(7));
    publication.0 = Some(u32::MAX);
    assert!(!publication.take(0));
}
