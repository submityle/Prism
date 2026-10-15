use core::ops::Range;

use prism_render_architecture::geometry::{GeometryLodRecord, GeometryPrimitiveKind};

/// Validates an indexed draw against the allocator ranges that will be bound
/// to the render pass. The metadata is untrusted at this boundary: a corrupt
/// or stale record must skip the draw instead of wrapping an integer and
/// submitting an invalid GPU command.
pub(crate) fn indexed_draw_range(
    lod: &GeometryLodRecord,
    vertex_range: Range<u32>,
    index_range: Range<u32>,
    mesh_index_count: u32,
) -> Option<Range<u32>> {
    if lod.primitive_kind != GeometryPrimitiveKind::Indexed
        || lod.element_count > mesh_index_count
        || !contains_range(&index_range, lod.first_element, lod.element_count)
        || lod.base_vertex < 0
    {
        return None;
    }
    let base_vertex = u32::try_from(lod.base_vertex).ok()?;
    let vertex_end = base_vertex.checked_add(lod.vertex_count)?;
    if vertex_end > vertex_range.end || base_vertex < vertex_range.start {
        return None;
    }
    Some(lod.first_element..lod.first_element.checked_add(lod.element_count)?)
}

/// Validates a non-indexed draw against the allocator vertex range.
pub(crate) fn non_indexed_draw_range(
    lod: &GeometryLodRecord,
    vertex_range: Range<u32>,
    mesh_vertex_count: u32,
) -> Option<Range<u32>> {
    if lod.primitive_kind != GeometryPrimitiveKind::NonIndexed
        || lod.base_vertex != 0
        || lod.element_count > mesh_vertex_count
        || !contains_range(&vertex_range, lod.first_element, lod.element_count)
    {
        return None;
    }
    Some(lod.first_element..lod.first_element.checked_add(lod.element_count)?)
}

fn contains_range(container: &Range<u32>, start: u32, count: u32) -> bool {
    start >= container.start
        && start
            .checked_add(count)
            .is_some_and(|end| end <= container.end)
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_architecture::geometry::GeometryPrimitiveKind;

    fn lod(kind: GeometryPrimitiveKind) -> GeometryLodRecord {
        GeometryLodRecord {
            primitive_kind: kind,
            first_element: 12,
            element_count: 6,
            base_vertex: 20,
            vertex_count: 8,
            ..Default::default()
        }
    }

    #[test]
    fn indexed_range_is_checked_against_both_allocator_slices() {
        let range = indexed_draw_range(&lod(GeometryPrimitiveKind::Indexed), 0..32, 8..24, 6);
        assert_eq!(range, Some(12..18));
        assert!(
            indexed_draw_range(&lod(GeometryPrimitiveKind::Indexed), 0..24, 8..16, 6).is_none()
        );
    }

    #[test]
    fn overflow_and_primitive_mismatch_fail_closed() {
        let mut invalid = lod(GeometryPrimitiveKind::Indexed);
        invalid.first_element = u32::MAX;
        assert!(indexed_draw_range(&invalid, 0..32, 0..u32::MAX, u32::MAX).is_none());
        assert!(non_indexed_draw_range(&lod(GeometryPrimitiveKind::Indexed), 0..32, 32).is_none());
    }

    #[test]
    fn non_indexed_range_uses_the_vertex_allocator_slice() {
        let mut value = lod(GeometryPrimitiveKind::NonIndexed);
        value.first_element = 4;
        value.base_vertex = 0;
        value.vertex_count = 6;
        assert_eq!(non_indexed_draw_range(&value, 2..16, 6), Some(4..10));
        assert!(non_indexed_draw_range(&value, 2..9, 6).is_none());
    }
}
