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

/// Consumer invoked for each graph stage in deterministic submission order.
///
/// A renderer implements this small adapter at its graph boundary. The world
/// contract owns ordering and validation; the consumer owns device resources,
/// barriers, and the actual render/compute API. Each method is called exactly
/// once for every matching pass, including empty streams, so a consumer cannot
/// silently skip a cell while still reporting a successful execution.
pub trait PlacementGraphEncoder {
    /// Consumer-specific failure type.
    type Error;

    /// Registers or uploads the source stream for one cell.
    fn upload(&mut self, stream: PlacementStream) -> Result<(), Self::Error>;

    /// Encodes the terrain projection/placement producer for one cell.
    fn project(&mut self, stream: PlacementStream) -> Result<(), Self::Error>;

    /// Encodes compaction and indirect-argument production for one cell.
    fn compact(&mut self, stream: PlacementStream) -> Result<(), Self::Error>;

    /// Registers the indirect draw consumer for one cell.
    fn draw_indirect(&mut self, stream: PlacementStream) -> Result<(), Self::Error>;
}

/// Failure returned when a graph consumer rejects one pass.
#[derive(Debug, Eq, PartialEq)]
pub struct PlacementGraphEncodeError<E> {
    /// Zero-based pass index in [`PlacementRenderGraph::passes`].
    pub pass_index: usize,
    /// Cell/revision associated with the rejected pass.
    pub stream: PlacementStream,
    /// Stage that was rejected.
    pub stage: PlacementGraphStage,
    /// Consumer-specific error.
    pub error: E,
}

impl<E: fmt::Display> fmt::Display for PlacementGraphEncodeError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "placement graph pass {} ({:?}, {:?}) failed: {}",
            self.pass_index, self.stream, self.stage, self.error
        )
    }
}

impl<E: fmt::Debug + fmt::Display> core::error::Error for PlacementGraphEncodeError<E> {}

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

    /// Executes every pass through a renderer-owned consumer.
    ///
    /// The graph is already validated and sorted by [`Self::build`]. This
    /// method performs no allocation and never reorders or coalesces passes;
    /// therefore a renderer can map each callback to one native graph node or
    /// command section while retaining the same deterministic cell ordering.
    /// If a consumer fails, the exact pass and stream are returned. Earlier
    /// callbacks may already have been applied; callers that need rollback must
    /// provide it in their consumer or discard the partially recorded command
    /// encoder.
    pub fn encode<E: PlacementGraphEncoder>(
        &self,
        encoder: &mut E,
    ) -> Result<(), PlacementGraphEncodeError<E::Error>> {
        for (pass_index, pass) in self.passes.iter().copied().enumerate() {
            let result = match pass.stage {
                PlacementGraphStage::Upload => encoder.upload(pass.stream),
                PlacementGraphStage::Project => encoder.project(pass.stream),
                PlacementGraphStage::Compact => encoder.compact(pass.stream),
                PlacementGraphStage::DrawIndirect => encoder.draw_indirect(pass.stream),
            };
            if let Err(error) = result {
                return Err(PlacementGraphEncodeError {
                    pass_index,
                    stream: pass.stream,
                    stage: pass.stage,
                    error,
                });
            }
        }
        Ok(())
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

    #[derive(Default)]
    struct RecordingEncoder {
        calls: Vec<(PlacementGraphStage, PlacementStream)>,
        fail_at: Option<usize>,
    }

    impl PlacementGraphEncoder for RecordingEncoder {
        type Error = &'static str;

        fn upload(&mut self, stream: PlacementStream) -> Result<(), Self::Error> {
            self.record(PlacementGraphStage::Upload, stream)
        }

        fn project(&mut self, stream: PlacementStream) -> Result<(), Self::Error> {
            self.record(PlacementGraphStage::Project, stream)
        }

        fn compact(&mut self, stream: PlacementStream) -> Result<(), Self::Error> {
            self.record(PlacementGraphStage::Compact, stream)
        }

        fn draw_indirect(&mut self, stream: PlacementStream) -> Result<(), Self::Error> {
            self.record(PlacementGraphStage::DrawIndirect, stream)
        }
    }

    impl RecordingEncoder {
        fn record(
            &mut self,
            stage: PlacementGraphStage,
            stream: PlacementStream,
        ) -> Result<(), &'static str> {
            let index = self.calls.len();
            self.calls.push((stage, stream));
            if self.fail_at == Some(index) {
                Err("consumer rejected pass")
            } else {
                Ok(())
            }
        }
    }

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

    #[test]
    fn encode_delivers_every_stage_in_graph_order_without_reordering() {
        let streams = [stream(3, 1), stream(-1, 2)];
        let graph = PlacementRenderGraph::build(&streams).unwrap();
        let mut encoder = RecordingEncoder::default();

        graph.encode(&mut encoder).unwrap();

        assert_eq!(
            encoder.calls,
            graph
                .passes()
                .iter()
                .map(|pass| (pass.stage, pass.stream))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn encode_reports_the_exact_rejected_pass_and_stops() {
        let graph = PlacementRenderGraph::build(&[stream(1, 7)]).unwrap();
        let mut encoder = RecordingEncoder {
            fail_at: Some(2),
            ..Default::default()
        };

        let error = graph.encode(&mut encoder).unwrap_err();

        assert_eq!(error.pass_index, 2);
        assert_eq!(error.stage, PlacementGraphStage::Compact);
        assert_eq!(error.stream, stream(1, 7));
        assert_eq!(error.error, "consumer rejected pass");
        assert_eq!(encoder.calls.len(), 3);
    }
}
