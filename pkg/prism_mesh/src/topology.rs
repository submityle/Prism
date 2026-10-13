//! Helpers for the primitive topologies owned by `prism_render_driver`.
//!
//! `prism_mesh` deliberately does not define a second topology enum.  The
//! extension trait below keeps topology arithmetic next to the mesh contract
//! while using the RHI's [`PrimitiveTopology`] as the single source of truth.

use prism_render_driver::PrimitiveTopology;

use crate::error::MeshError;

/// Operations that are useful when validating or partitioning indexed mesh
/// data.
pub trait PrimitiveTopologyExt {
    /// Returns the number of primitives assembled from `element_count`
    /// vertices or indices.
    ///
    /// A list topology requires a complete primitive, so an incomplete list
    /// returns `None`. Strip topologies may have fewer than their minimum
    /// number of elements and then produce zero primitives.
    #[must_use]
    fn primitive_count(self, element_count: u32) -> Option<u32>;

    /// Whether `element_count` is a complete element sequence for this
    /// topology.
    #[must_use]
    fn is_valid_element_count(self, element_count: u32) -> bool;

    /// The minimum number of elements needed to produce one primitive.
    #[must_use]
    fn minimum_element_count(self) -> u32;

    /// Whether primitives are assembled as independent lists.
    #[must_use]
    fn is_list(self) -> bool;

    /// Whether primitives are assembled as connected strips.
    #[must_use]
    fn is_strip(self) -> bool;

    /// Whether this topology assembles triangles.
    #[must_use]
    fn is_triangle(self) -> bool;

    /// Whether the topology can be issued without an index buffer.
    #[must_use]
    fn supports_non_indexed(self) -> bool;

    /// Validates an element sequence and returns the corresponding contract
    /// error when a list is incomplete.
    fn validate_element_count(self, element_count: u32) -> Result<(), MeshError>;
}

impl PrimitiveTopologyExt for PrimitiveTopology {
    fn primitive_count(self, element_count: u32) -> Option<u32> {
        match self {
            Self::PointList => Some(element_count),
            Self::LineList => complete_list_count(element_count, 2),
            Self::LineStrip => Some(element_count.saturating_sub(1)),
            Self::TriangleList => complete_list_count(element_count, 3),
            Self::TriangleStrip => Some(element_count.saturating_sub(2)),
        }
    }

    fn is_valid_element_count(self, element_count: u32) -> bool {
        self.primitive_count(element_count).is_some()
    }

    fn minimum_element_count(self) -> u32 {
        match self {
            Self::PointList => 1,
            Self::LineList | Self::LineStrip => 2,
            Self::TriangleList | Self::TriangleStrip => 3,
        }
    }

    fn is_list(self) -> bool {
        matches!(self, Self::PointList | Self::LineList | Self::TriangleList)
    }

    fn is_strip(self) -> bool {
        matches!(self, Self::LineStrip | Self::TriangleStrip)
    }

    fn is_triangle(self) -> bool {
        matches!(self, Self::TriangleList | Self::TriangleStrip)
    }

    fn supports_non_indexed(self) -> bool {
        true
    }

    fn validate_element_count(self, element_count: u32) -> Result<(), MeshError> {
        if self.is_valid_element_count(element_count) {
            Ok(())
        } else {
            Err(MeshError::TopologyElementCount {
                topology: self,
                element_count,
            })
        }
    }
}

/// Returns the primitive count for a list topology, rejecting a trailing
/// partial primitive.
#[must_use]
pub const fn primitive_count(topology: PrimitiveTopology, element_count: u32) -> Option<u32> {
    match topology {
        PrimitiveTopology::PointList => Some(element_count),
        PrimitiveTopology::LineList => complete_list_count(element_count, 2),
        PrimitiveTopology::LineStrip => Some(element_count.saturating_sub(1)),
        PrimitiveTopology::TriangleList => complete_list_count(element_count, 3),
        PrimitiveTopology::TriangleStrip => Some(element_count.saturating_sub(2)),
    }
}

/// Returns the number of elements consumed by one primitive for list
/// topologies. Strip topologies consume a sliding window and therefore are
/// represented by [`PrimitiveTopologyExt::minimum_element_count`].
#[must_use]
pub const fn elements_per_primitive(topology: PrimitiveTopology) -> u32 {
    match topology {
        PrimitiveTopology::PointList => 1,
        PrimitiveTopology::LineList | PrimitiveTopology::LineStrip => 2,
        PrimitiveTopology::TriangleList | PrimitiveTopology::TriangleStrip => 3,
    }
}

const fn complete_list_count(element_count: u32, elements_per_primitive: u32) -> Option<u32> {
    if element_count % elements_per_primitive == 0 {
        Some(element_count / elements_per_primitive)
    } else {
        None
    }
}
