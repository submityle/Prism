//! Render-graph-ready scheduling contract for world placement streams.
//!
//! The graph is a backend-neutral description of resource hazards and ordered
//! work.  It deliberately does not own a device or silently submit commands;
//! a renderer can translate the plan to its graph API while retaining the same
//! deterministic cell ordering and indirect-draw dependency.

use core::fmt;
use std::vec::Vec;

use prism_world_stream::SectorId;

/// A stable placement stream identified by its source cell and revision.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct PlacementStream {
    /// Source world cell.
    pub cell: SectorId,
    /// Content/terrain revision used to invalidate the stream.
    pub revision: u64,
    /// Number of primary placement candidates.
    pub candidate_count: u32,
}

/// Logical graph resource used by placement and rendering passes.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum PlacementGraphResource {
    /// CPU-authored or decompressed source records.
    Source,
    /// Projected world-space placement records.
    Placement,
    /// Compacted source-slot indices.
    Compact,
    /// Indirect draw arguments.
    Indirect,
    /// Final render target or scene instance consumer.
    Scene,
}

/// A graph stage with explicit read/write hazards.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlacementGraphStage {
    /// Upload/replace a cell's source stream.
    Upload,
    /// Project source candidates onto terrain.
    Project,
    /// Compact accepted candidates and produce draw count.
    Compact,
    /// Consume indirect arguments in the render pass.
    DrawIndirect,
}

impl PlacementGraphStage {
    /// Returns the stable stage order used by the graph compiler.
    #[must_use]
    pub const fn ordinal(self) -> u8 {
        match self {
            Self::Upload => 0,
            Self::Project => 1,
            Self::Compact => 2,
            Self::DrawIndirect => 3,
        }
    }

    /// Returns resources read by this stage.
    #[must_use]
    pub const fn reads(self) -> &'static [PlacementGraphResource] {
        match self {
            Self::Upload => &[],
            Self::Project => &[PlacementGraphResource::Source],
            Self::Compact => &[PlacementGraphResource::Placement],
            Self::DrawIndirect => &[
                PlacementGraphResource::Placement,
                PlacementGraphResource::Compact,
                PlacementGraphResource::Indirect,
            ],
        }
    }

    /// Returns resources written by this stage.
    #[must_use]
    pub const fn writes(self) -> &'static [PlacementGraphResource] {
        match self {
            Self::Upload => &[PlacementGraphResource::Source],
            Self::Project => &[PlacementGraphResource::Placement],
            Self::Compact => &[
                PlacementGraphResource::Compact,
                PlacementGraphResource::Indirect,
            ],
            Self::DrawIndirect => &[PlacementGraphResource::Scene],
        }
    }
}

/// A single cell-scoped graph pass.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlacementGraphPass {
    /// Cell stream consumed by this pass.
    pub stream: PlacementStream,
    /// Ordered operation.
    pub stage: PlacementGraphStage,
}

/// A complete graph plan for a frame.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlacementRenderGraph {
    passes: Vec<PlacementGraphPass>,
}

/// Errors from graph construction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlacementGraphError {
    /// The same cell/revision was submitted twice.
    DuplicateStream { cell: SectorId, revision: u64 },
    /// More than one revision of a cell was submitted for one graph.
    ConflictingCellRevision {
        /// Cell submitted more than once.
        cell: SectorId,
        /// First revision in sorted graph order.
        first_revision: u64,
        /// Later revision in sorted graph order.
        second_revision: u64,
    },
    /// Host allocation failed while reserving the pass list.
    AllocationFailed,
}

impl fmt::Display for PlacementGraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateStream { cell, revision } => {
                write!(f, "duplicate placement stream {cell:?} revision {revision}")
            }
            Self::ConflictingCellRevision {
                cell,
                first_revision,
                second_revision,
            } => {
                write!(
                    f,
                    "placement cell {cell:?} has conflicting revisions {first_revision} and {second_revision}"
                )
            }
            Self::AllocationFailed => f.write_str("placement graph pass reservation failed"),
        }
    }
}

impl core::error::Error for PlacementGraphError {}

impl PlacementRenderGraph {
    /// Builds a deterministic graph. Cells are sorted by full `SectorId`, then
    /// every cell receives upload → project → compact → indirect draw stages.
    pub fn build(streams: &[PlacementStream]) -> Result<Self, PlacementGraphError> {
        let mut ordered = streams.to_vec();
        ordered.sort_unstable_by_key(|stream| (stream.cell, stream.revision));
        for pair in ordered.windows(2) {
            if pair[0].cell == pair[1].cell {
                if pair[0].revision == pair[1].revision {
                    return Err(PlacementGraphError::DuplicateStream {
                        cell: pair[0].cell,
                        revision: pair[0].revision,
                    });
                }
                return Err(PlacementGraphError::ConflictingCellRevision {
                    cell: pair[0].cell,
                    first_revision: pair[0].revision,
                    second_revision: pair[1].revision,
                });
            }
        }
        let mut passes = Vec::new();
        passes
            .try_reserve_exact(ordered.len().saturating_mul(4))
            .map_err(|_| PlacementGraphError::AllocationFailed)?;
        for stream in ordered {
            for stage in [
                PlacementGraphStage::Upload,
                PlacementGraphStage::Project,
                PlacementGraphStage::Compact,
                PlacementGraphStage::DrawIndirect,
            ] {
                passes.push(PlacementGraphPass { stream, stage });
            }
        }
        Ok(Self { passes })
    }

    /// Returns passes in submission order.
    #[must_use]
    pub fn passes(&self) -> &[PlacementGraphPass] {
        &self.passes
    }

    /// Returns whether the plan contains all producer-to-consumer barriers.
    #[must_use]
    pub fn has_complete_hazards(&self) -> bool {
        self.passes.chunks_exact(4).all(|passes| {
            passes[0].stage.ordinal() < passes[1].stage.ordinal()
                && passes[1].stage.ordinal() < passes[2].stage.ordinal()
                && passes[2].stage.ordinal() < passes[3].stage.ordinal()
                && passes[1]
                    .stage
                    .reads()
                    .contains(&PlacementGraphResource::Source)
                && passes[2]
                    .stage
                    .reads()
                    .contains(&PlacementGraphResource::Placement)
                && passes[3]
                    .stage
                    .reads()
                    .contains(&PlacementGraphResource::Indirect)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stream(x: i64, revision: u64) -> PlacementStream {
        PlacementStream {
            cell: SectorId::new(2, x, 0, 0),
            revision,
            candidate_count: 128,
        }
    }

    #[test]
    fn multi_cell_graph_is_sorted_and_has_indirect_dependency_chain() {
        let graph = PlacementRenderGraph::build(&[stream(3, 1), stream(-1, 2)]).unwrap();
        assert!(graph.has_complete_hazards());
        assert_eq!(graph.passes().len(), 8);
        assert_eq!(graph.passes()[0].stream.cell, SectorId::new(2, -1, 0, 0));
        assert_eq!(graph.passes()[3].stage, PlacementGraphStage::DrawIndirect);
        assert!(graph.passes()[3]
            .stage
            .reads()
            .contains(&PlacementGraphResource::Compact));
    }

    #[test]
    fn duplicate_cell_revision_is_rejected_before_graph_publication() {
        let error = PlacementRenderGraph::build(&[stream(1, 4), stream(1, 4)]).unwrap_err();
        assert!(matches!(error, PlacementGraphError::DuplicateStream { .. }));
    }

    #[test]
    fn conflicting_cell_revisions_are_rejected_before_graph_publication() {
        let error = PlacementRenderGraph::build(&[stream(1, 4), stream(1, 5)]).unwrap_err();
        assert!(matches!(
            error,
            PlacementGraphError::ConflictingCellRevision {
                first_revision: 4,
                second_revision: 5,
                ..
            }
        ));
    }
}
