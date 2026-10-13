//! Static mesh submeshes: material ownership plus an index interval.

use core::ops::Range;

use prism_render_driver::PrimitiveTopology;

use crate::error::MeshError;
use crate::limits::MeshLimits;
use crate::topology::PrimitiveTopologyExt;

/// A contiguous section of a mesh's index buffer rendered with one material
/// slot.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct SubMesh {
    /// Material table index used for this section.
    pub material_slot: u16,
    /// Half-open index range `[start, end)` in the parent mesh.
    pub index_range: Range<u32>,
    /// Optional half-open cluster range populated by a virtual-geometry
    /// builder. The static contract only validates its shape, not its global
    /// cluster-table bound.
    pub cluster_range: Option<Range<u32>>,
}

impl SubMesh {
    /// Creates a submesh without a cluster range.
    #[must_use]
    pub const fn new(material_slot: u16, index_range: Range<u32>) -> Self {
        Self {
            material_slot,
            index_range,
            cluster_range: None,
        }
    }

    /// Adds the optional cluster range.
    #[must_use]
    pub fn with_cluster_range(mut self, cluster_range: Range<u32>) -> Self {
        self.cluster_range = Some(cluster_range);
        self
    }

    /// Number of indices covered by this submesh.
    #[must_use]
    pub const fn index_count(&self) -> u32 {
        self.index_range.end.saturating_sub(self.index_range.start)
    }

    /// Validates the range against a parent index buffer and topology.
    pub fn validate(&self, index_count: u32, topology: PrimitiveTopology) -> Result<(), MeshError> {
        validate_range(&self.index_range, index_count, index_range_error)?;

        if self.index_range.is_empty() {
            return Err(MeshError::EmptySubMeshIndexRange);
        }
        if topology.primitive_count(self.index_count()) == Some(0) {
            return Err(MeshError::SubMeshHasNoPrimitives {
                material_slot: self.material_slot,
                index_count: self.index_count(),
                topology,
            });
        }
        topology.validate_element_count(self.index_count())?;

        if let Some(cluster_range) = &self.cluster_range {
            validate_range(cluster_range, u32::MAX, cluster_range_error)?;
            if cluster_range.is_empty() {
                return Err(MeshError::EmptySubMeshClusterRange);
            }
        }

        Ok(())
    }

    /// Validates this submesh against both topology and profile limits.
    pub fn validate_with_limits(
        &self,
        index_count: u32,
        topology: PrimitiveTopology,
        limits: &MeshLimits,
    ) -> Result<(), MeshError> {
        limits.validate_material_slot(self.material_slot)?;
        self.validate(index_count, topology)
    }
}

/// Validates a complete, non-overlapping submesh table.
pub fn validate_submeshes(
    submeshes: &[SubMesh],
    index_count: u32,
    topology: PrimitiveTopology,
) -> Result<(), MeshError> {
    for submesh in submeshes {
        submesh.validate(index_count, topology)?;
    }

    for (first_index, first) in submeshes.iter().enumerate() {
        for (second_index, second) in submeshes.iter().enumerate().skip(first_index + 1) {
            if ranges_overlap(&first.index_range, &second.index_range) {
                return Err(MeshError::SubMeshIndexRangesOverlap {
                    first: first_index,
                    second: second_index,
                });
            }
        }
    }

    Ok(())
}

/// Validates a complete submesh table against topology and mesh-facing limits.
pub fn validate_submeshes_with_limits(
    submeshes: &[SubMesh],
    index_count: u32,
    topology: PrimitiveTopology,
    limits: &MeshLimits,
) -> Result<(), MeshError> {
    for submesh in submeshes {
        submesh.validate_with_limits(index_count, topology, limits)?;
    }

    for (first_index, first) in submeshes.iter().enumerate() {
        for (second_index, second) in submeshes.iter().enumerate().skip(first_index + 1) {
            if ranges_overlap(&first.index_range, &second.index_range) {
                return Err(MeshError::SubMeshIndexRangesOverlap {
                    first: first_index,
                    second: second_index,
                });
            }
        }
    }

    Ok(())
}

fn validate_range(
    range: &Range<u32>,
    upper_bound: u32,
    out_of_bounds: fn(u32, u32, u32) -> MeshError,
) -> Result<(), MeshError> {
    if range.start > range.end || range.end > upper_bound {
        return Err(out_of_bounds(range.start, range.end, upper_bound));
    }
    Ok(())
}

fn index_range_error(start: u32, end: u32, index_count: u32) -> MeshError {
    MeshError::SubMeshIndexRangeOutOfBounds {
        start,
        end,
        index_count,
    }
}

fn cluster_range_error(start: u32, end: u32, cluster_count: u32) -> MeshError {
    MeshError::SubMeshClusterRangeOutOfBounds {
        start,
        end,
        cluster_count,
    }
}

fn ranges_overlap(first: &Range<u32>, second: &Range<u32>) -> bool {
    first.start < second.end && second.start < first.end
}
