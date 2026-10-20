//! Deterministic heightfield-placement lifecycle and render-graph adapter.
//!
//! This module is the boundary between world residency and Prism's declarative
//! frame graph. It owns no wgpu object and never submits work. Its graph
//! resources are imported declarations: the existing heightfield placement
//! owner remains responsible for raw wgpu buffers and compute encoding, while
//! the compiled plan carries ordering and hazard metadata for a Prism RHI
//! executor.
//!
//! The lifecycle is local to this crate. A runtime coordinator may feed it a
//! PlacementCellManifest, but cannot mutate the resident table through a
//! leaked mutable reference. Successful operations publish a new immutable
//! stream snapshot; failed or stale operations leave the previous snapshot
//! unchanged.

use core::fmt;
use std::collections::BTreeMap;
use std::vec::Vec;

use prism_render_graph::frame_graph::{
    AccessKind, GpuFrameGraphBuilder, PassDescriptor, PassId, QueueClass, ResourceAccess,
    ResourceDescriptor, ResourceId, ResourceLifetime,
};
use prism_render_graph::{BufferUse, CompileError, ExecutionPlan};
use prism_world_stream::{CellManifest, SectorId};

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

/// Manifest information admitted by the placement lifecycle.
///
/// CellManifest remains the source of spatial identity and content metadata;
/// revision is the placement-specific fence. This keeps a terrain rebake or
/// placement recipe change distinct from a runtime cell payload revision.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlacementCellManifest {
    /// Runtime cell manifest used to identify the sector.
    pub manifest: CellManifest,
    /// Monotonically increasing placement revision for this sector.
    pub revision: u64,
    /// Number of primary placement candidates owned by the stream.
    pub candidate_count: u32,
}

impl PlacementCellManifest {
    /// Creates placement metadata for one runtime cell.
    #[must_use]
    pub const fn new(manifest: CellManifest, revision: u64, candidate_count: u32) -> Self {
        Self {
            manifest,
            revision,
            candidate_count,
        }
    }

    /// Returns the manifest's sector without exposing lifecycle storage.
    #[must_use]
    pub const fn sector(self) -> SectorId {
        self.manifest.sector
    }

    /// Converts admitted metadata to the graph-facing stream value.
    #[must_use]
    pub const fn stream(self) -> PlacementStream {
        PlacementStream {
            cell: self.manifest.sector,
            revision: self.revision,
            candidate_count: self.candidate_count,
        }
    }
}

/// A read-only deterministic view of all currently resident streams.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlacementStreamSnapshot {
    streams: Vec<PlacementStream>,
}

impl PlacementStreamSnapshot {
    /// Returns streams sorted by their full SectorId.
    #[must_use]
    pub fn as_slice(&self) -> &[PlacementStream] {
        &self.streams
    }

    /// Returns the number of resident streams in the snapshot.
    #[must_use]
    pub fn len(&self) -> usize {
        self.streams.len()
    }

    /// Returns whether the snapshot contains no resident streams.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.streams.is_empty()
    }

    /// Iterates over the immutable stream snapshot.
    pub fn iter(&self) -> impl Iterator<Item = &PlacementStream> {
        self.streams.iter()
    }
}

/// Lifecycle errors rejected before resident state is changed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PlacementLifecycleError {
    /// A load attempted to publish a second stream for one sector.
    DuplicateSector { sector: SectorId },
    /// A reload or eviction referenced a sector that is not resident.
    MissingSector { sector: SectorId },
    /// A revision older than or equal to the resident stream was rejected.
    StaleRevision {
        sector: SectorId,
        current: u64,
        incoming: u64,
    },
    /// An eviction from a future revision cannot invalidate the current stream.
    RevisionAhead {
        sector: SectorId,
        current: u64,
        incoming: u64,
    },
    /// The graph plan could not reserve host-side declaration storage.
    AllocationFailed,
    /// The shared frame-graph compiler rejected the declaration contract.
    GraphCompile(CompileError),
}

impl fmt::Display for PlacementLifecycleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateSector { sector } => write!(f, "duplicate placement sector {sector:?}"),
            Self::MissingSector { sector } => write!(f, "missing placement sector {sector:?}"),
            Self::StaleRevision {
                sector,
                current,
                incoming,
            } => write!(
                f,
                "stale placement revision {incoming} for {sector:?}; current is {current}"
            ),
            Self::RevisionAhead {
                sector,
                current,
                incoming,
            } => write!(
                f,
                "placement eviction revision {incoming} is ahead of {sector:?} current {current}"
            ),
            Self::AllocationFailed => {
                f.write_str("placement lifecycle snapshot reservation failed")
            }
            Self::GraphCompile(error) => {
                write!(f, "placement render graph compilation failed: {error}")
            }
        }
    }
}

impl std::error::Error for PlacementLifecycleError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::GraphCompile(error) => Some(error),
            _ => None,
        }
    }
}

impl From<CompileError> for PlacementLifecycleError {
    fn from(error: CompileError) -> Self {
        Self::GraphCompile(error)
    }
}

/// Local placement residency with an explicit stale-revision fence.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PlacementLifecycle {
    resident: BTreeMap<SectorId, PlacementStream>,
}

impl PlacementLifecycle {
    /// Creates an empty placement lifecycle.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Loads a previously absent sector and publishes its stream.
    pub fn load(
        &mut self,
        manifest: PlacementCellManifest,
    ) -> Result<PlacementStream, PlacementLifecycleError> {
        let sector = manifest.sector();
        if self.resident.contains_key(&sector) {
            return Err(PlacementLifecycleError::DuplicateSector { sector });
        }
        let stream = manifest.stream();
        self.resident.insert(sector, stream);
        Ok(stream)
    }

    /// Replaces a resident sector only with a strictly newer revision.
    pub fn reload(
        &mut self,
        manifest: PlacementCellManifest,
    ) -> Result<PlacementStream, PlacementLifecycleError> {
        let sector = manifest.sector();
        let current = self
            .resident
            .get(&sector)
            .ok_or(PlacementLifecycleError::MissingSector { sector })?;
        if manifest.revision <= current.revision {
            return Err(PlacementLifecycleError::StaleRevision {
                sector,
                current: current.revision,
                incoming: manifest.revision,
            });
        }
        let stream = manifest.stream();
        self.resident.insert(sector, stream);
        Ok(stream)
    }

    /// Evicts a sector only when the caller names its exact resident revision.
    pub fn evict(
        &mut self,
        sector: SectorId,
        revision: u64,
    ) -> Result<PlacementStream, PlacementLifecycleError> {
        let current = *self
            .resident
            .get(&sector)
            .ok_or(PlacementLifecycleError::MissingSector { sector })?;
        if revision < current.revision {
            return Err(PlacementLifecycleError::StaleRevision {
                sector,
                current: current.revision,
                incoming: revision,
            });
        }
        if revision > current.revision {
            return Err(PlacementLifecycleError::RevisionAhead {
                sector,
                current: current.revision,
                incoming: revision,
            });
        }
        self.resident.remove(&sector);
        Ok(current)
    }

    /// Returns a deterministic read-only snapshot for graph construction.
    pub fn snapshot(&self) -> Result<PlacementStreamSnapshot, PlacementLifecycleError> {
        let mut streams = Vec::new();
        streams
            .try_reserve_exact(self.resident.len())
            .map_err(|_| PlacementLifecycleError::AllocationFailed)?;
        streams.extend(self.resident.values().copied());
        Ok(PlacementStreamSnapshot { streams })
    }

    /// Returns the resident stream for one sector, if present.
    #[must_use]
    pub fn stream(&self, sector: SectorId) -> Option<PlacementStream> {
        self.resident.get(&sector).copied()
    }

    /// Returns the number of resident placement streams.
    #[must_use]
    pub fn len(&self) -> usize {
        self.resident.len()
    }

    /// Returns whether no placement stream is resident.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.resident.is_empty()
    }
}

/// Logical resources used by the heightfield-placement graph.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum PlacementGraphResource {
    /// Uploaded source candidate records.
    Source,
    /// Projected world-space placement records and attributes.
    Placement,
    /// Compacted source-slot indices.
    Compact,
    /// Indirect draw arguments and live count.
    Indirect,
    /// External scene target consumed by the draw stage.
    Scene,
}

/// A graph stage with explicit read/write hazards.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlacementGraphStage {
    /// Upload/replace a cell's source stream.
    Upload,
    /// Project source candidates onto the heightfield.
    Project,
    /// Compact accepted candidates and produce draw count.
    Compact,
    /// Consume indirect arguments in the external scene pass.
    DrawIndirect,
}

impl PlacementGraphStage {
    /// Returns the stable stage order used by the adapter.
    #[must_use]
    pub const fn ordinal(self) -> u8 {
        match self {
            Self::Upload => 0,
            Self::Project => 1,
            Self::Compact => 2,
            Self::DrawIndirect => 3,
        }
    }

    /// Returns resources read by this stage in logical graph vocabulary.
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

    /// Returns resources written by this stage in logical graph vocabulary.
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

    /// Returns the corresponding RHI uses without selecting a raw backend.
    #[must_use]
    pub const fn rhi_uses(self) -> &'static [PlacementRhiUse] {
        match self {
            Self::Upload => &[PlacementRhiUse::Buffer {
                resource: PlacementGraphResource::Source,
                use_kind: BufferUse::CopyDst,
            }],
            Self::Project => &[
                PlacementRhiUse::Buffer {
                    resource: PlacementGraphResource::Source,
                    use_kind: BufferUse::StorageRead,
                },
                PlacementRhiUse::Buffer {
                    resource: PlacementGraphResource::Placement,
                    use_kind: BufferUse::StorageWrite,
                },
            ],
            Self::Compact => &[
                PlacementRhiUse::Buffer {
                    resource: PlacementGraphResource::Placement,
                    use_kind: BufferUse::StorageRead,
                },
                PlacementRhiUse::Buffer {
                    resource: PlacementGraphResource::Compact,
                    use_kind: BufferUse::StorageWrite,
                },
                PlacementRhiUse::Buffer {
                    resource: PlacementGraphResource::Indirect,
                    use_kind: BufferUse::StorageWrite,
                },
            ],
            Self::DrawIndirect => &[
                PlacementRhiUse::Buffer {
                    resource: PlacementGraphResource::Placement,
                    use_kind: BufferUse::StorageRead,
                },
                PlacementRhiUse::Buffer {
                    resource: PlacementGraphResource::Compact,
                    use_kind: BufferUse::StorageRead,
                },
                PlacementRhiUse::Buffer {
                    resource: PlacementGraphResource::Indirect,
                    use_kind: BufferUse::Indirect,
                },
                PlacementRhiUse::ColorAttachment {
                    resource: PlacementGraphResource::Scene,
                },
            ],
        }
    }
}

/// Backend-neutral RHI use emitted by PlacementGraphStage::rhi_uses.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlacementRhiUse {
    /// A buffer access that can be lowered by a RenderDevice backend.
    Buffer {
        /// Logical placement resource.
        resource: PlacementGraphResource,
        /// RHI buffer access mode.
        use_kind: BufferUse,
    },
    /// The external scene target is a render attachment, not a raw buffer.
    ColorAttachment {
        /// Logical scene resource.
        resource: PlacementGraphResource,
    },
}

/// A single cell-scoped graph pass.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlacementGraphPass {
    /// Cell stream consumed by this pass.
    pub stream: PlacementStream,
    /// Ordered operation.
    pub stage: PlacementGraphStage,
}

/// Logical resource byte sizes supplied by the external GPU owner.
///
/// The graph adapter never allocates these resources. Zero is valid for a
/// topology-only declaration; a production owner should provide exact imported
/// byte footprints when it needs graph memory diagnostics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlacementGraphLayout {
    /// Source record buffer bytes.
    pub source_bytes: u64,
    /// Projected placement record buffer bytes.
    pub placement_bytes: u64,
    /// Compact index buffer bytes.
    pub compact_bytes: u64,
    /// Indirect argument buffer bytes.
    pub indirect_bytes: u64,
}

impl PlacementGraphLayout {
    /// A declaration that carries ordering and hazards without owning sizes.
    #[must_use]
    pub const fn topology_only() -> Self {
        Self {
            source_bytes: 0,
            placement_bytes: 0,
            compact_bytes: 0,
            indirect_bytes: 0,
        }
    }
}

/// A complete deterministic graph declaration for a frame.
pub struct PlacementRenderGraph {
    passes: Vec<PlacementGraphPass>,
    declarations: GpuFrameGraphBuilder,
}

impl PlacementRenderGraph {
    /// Builds a topology-only graph using imported external resources.
    pub fn build(streams: &[PlacementStream]) -> Result<Self, PlacementLifecycleError> {
        Self::build_with_layout(streams, PlacementGraphLayout::topology_only())
    }

    /// Builds a graph with caller-supplied imported resource footprints.
    pub fn build_with_layout(
        streams: &[PlacementStream],
        layout: PlacementGraphLayout,
    ) -> Result<Self, PlacementLifecycleError> {
        let mut ordered = streams.to_vec();
        ordered.sort_unstable_by_key(|stream| (stream.cell, stream.revision));
        for pair in ordered.windows(2) {
            if pair[0].cell == pair[1].cell {
                if pair[0].revision == pair[1].revision {
                    return Err(PlacementLifecycleError::DuplicateSector {
                        sector: pair[0].cell,
                    });
                }
                return Err(PlacementLifecycleError::StaleRevision {
                    sector: pair[0].cell,
                    current: pair[0].revision,
                    incoming: pair[1].revision,
                });
            }
        }

        let mut declarations = GpuFrameGraphBuilder::default();
        let scene = declarations
            .try_add_resource(resource(
                "heightfield placement scene",
                0,
                ResourceLifetime::Imported,
            ))
            .map_err(|_| PlacementLifecycleError::AllocationFailed)?;
        let mut passes = Vec::new();
        passes
            .try_reserve_exact(ordered.len().saturating_mul(4))
            .map_err(|_| PlacementLifecycleError::AllocationFailed)?;

        let mut previous_draw: Option<PassId> = None;
        for stream in ordered {
            let source = declarations
                .try_add_resource(resource(
                    "heightfield placement source",
                    layout.source_bytes,
                    ResourceLifetime::Imported,
                ))
                .map_err(|_| PlacementLifecycleError::AllocationFailed)?;
            let placement = declarations
                .try_add_resource(resource(
                    "heightfield placement projected",
                    layout.placement_bytes,
                    ResourceLifetime::Imported,
                ))
                .map_err(|_| PlacementLifecycleError::AllocationFailed)?;
            let compact = declarations
                .try_add_resource(resource(
                    "heightfield placement compact",
                    layout.compact_bytes,
                    ResourceLifetime::Imported,
                ))
                .map_err(|_| PlacementLifecycleError::AllocationFailed)?;
            let indirect = declarations
                .try_add_resource(resource(
                    "heightfield placement indirect",
                    layout.indirect_bytes,
                    ResourceLifetime::Imported,
                ))
                .map_err(|_| PlacementLifecycleError::AllocationFailed)?;

            let upload = declarations
                .try_add_pass(pass(
                    "heightfield placement upload",
                    QueueClass::Transfer,
                    vec![access(source, AccessKind::TransferWrite)],
                    previous_draw.into_iter().collect(),
                ))
                .map_err(|_| PlacementLifecycleError::AllocationFailed)?;
            let project = declarations
                .try_add_pass(pass(
                    "heightfield placement project",
                    QueueClass::Compute,
                    vec![
                        access(source, AccessKind::StorageRead),
                        access(placement, AccessKind::StorageWrite),
                    ],
                    vec![upload],
                ))
                .map_err(|_| PlacementLifecycleError::AllocationFailed)?;
            let compact_pass = declarations
                .try_add_pass(pass(
                    "heightfield placement compact",
                    QueueClass::Compute,
                    vec![
                        access(placement, AccessKind::StorageRead),
                        access(compact, AccessKind::StorageWrite),
                        access(indirect, AccessKind::StorageWrite),
                    ],
                    vec![project],
                ))
                .map_err(|_| PlacementLifecycleError::AllocationFailed)?;
            let draw = declarations
                .try_add_pass(pass(
                    "heightfield placement draw indirect",
                    QueueClass::Graphics,
                    vec![
                        access(placement, AccessKind::StorageRead),
                        access(compact, AccessKind::StorageRead),
                        access(indirect, AccessKind::IndirectRead),
                        access(scene, AccessKind::ColorAttachment),
                    ],
                    vec![compact_pass],
                ))
                .map_err(|_| PlacementLifecycleError::AllocationFailed)?;

            previous_draw = Some(draw);
            passes.extend([
                PlacementGraphPass {
                    stream,
                    stage: PlacementGraphStage::Upload,
                },
                PlacementGraphPass {
                    stream,
                    stage: PlacementGraphStage::Project,
                },
                PlacementGraphPass {
                    stream,
                    stage: PlacementGraphStage::Compact,
                },
                PlacementGraphPass {
                    stream,
                    stage: PlacementGraphStage::DrawIndirect,
                },
            ]);
        }

        Ok(Self {
            passes,
            declarations,
        })
    }

    /// Returns the placement-specific pass view in deterministic order.
    #[must_use]
    pub fn passes(&self) -> &[PlacementGraphPass] {
        &self.passes
    }

    /// Returns the actual Prism frame-graph declarations for external binding.
    #[must_use]
    pub fn declarations(&self) -> &GpuFrameGraphBuilder {
        &self.declarations
    }

    /// Compiles through Prism's shared SSA, barrier and scheduling kernel.
    pub fn compile(&self) -> Result<ExecutionPlan, PlacementLifecycleError> {
        self.declarations.compile().map_err(Into::into)
    }

    /// Returns whether the placement stages are strictly ordered per stream.
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

fn resource(name: &'static str, size: u64, lifetime: ResourceLifetime) -> ResourceDescriptor {
    ResourceDescriptor {
        name: name.into(),
        lifetime,
        size,
        alignment: 16,
    }
}

fn access(resource: ResourceId, kind: AccessKind) -> ResourceAccess {
    ResourceAccess { resource, kind }
}

fn pass(
    name: &'static str,
    queue: QueueClass,
    accesses: Vec<ResourceAccess>,
    depends_on: Vec<PassId>,
) -> PassDescriptor {
    PassDescriptor {
        name: name.into(),
        queue,
        accesses,
        depends_on,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest(x: i64) -> CellManifest {
        CellManifest {
            sector: SectorId::new(2, x, 0, 0),
            bounds: prism_world_stream::AabbI64::new(
                prism_world_stream::GlobalPos::new(0, 0, 0),
                prism_world_stream::GlobalPos::new(1, 1, 1),
            )
            .expect("valid test bounds"),
            scene_key: 7,
            content_version: 1,
            derived_hash: 11,
            io_bytes: 0,
            data_layers: prism_world_stream::DataLayerMask::EMPTY,
        }
    }

    fn cell(x: i64, revision: u64, candidates: u32) -> PlacementCellManifest {
        PlacementCellManifest::new(manifest(x), revision, candidates)
    }

    #[test]
    fn lifecycle_fences_duplicate_stale_and_future_events() {
        let mut lifecycle = PlacementLifecycle::new();
        assert_eq!(lifecycle.load(cell(3, 4, 128)).unwrap().revision, 4);
        assert!(matches!(
            lifecycle.load(cell(3, 4, 128)),
            Err(PlacementLifecycleError::DuplicateSector { .. })
        ));
        assert!(matches!(
            lifecycle.reload(cell(3, 4, 256)),
            Err(PlacementLifecycleError::StaleRevision { .. })
        ));
        assert_eq!(
            lifecycle.reload(cell(3, 5, 256)).unwrap().candidate_count,
            256
        );
        assert!(matches!(
            lifecycle.evict(SectorId::new(2, 3, 0, 0), 4),
            Err(PlacementLifecycleError::StaleRevision { .. })
        ));
        assert!(matches!(
            lifecycle.evict(SectorId::new(2, 3, 0, 0), 6),
            Err(PlacementLifecycleError::RevisionAhead { .. })
        ));
        lifecycle.evict(SectorId::new(2, 3, 0, 0), 5).unwrap();
        assert!(lifecycle.is_empty());
    }

    #[test]
    fn snapshot_is_sorted_and_read_only_for_graph_consumers() {
        let mut lifecycle = PlacementLifecycle::new();
        lifecycle.load(cell(3, 1, 3)).unwrap();
        lifecycle.load(cell(-1, 1, 2)).unwrap();
        let snapshot = lifecycle.snapshot().unwrap();
        assert_eq!(snapshot.as_slice()[0].cell, SectorId::new(2, -1, 0, 0));
        assert_eq!(snapshot.len(), 2);
        assert_eq!(snapshot.iter().map(|s| s.candidate_count).sum::<u32>(), 5);
    }

    #[test]
    fn graph_adapter_compiles_real_frame_graph_declarations() {
        let streams = [cell(3, 1, 128).stream(), cell(-1, 2, 256).stream()];
        let graph = PlacementRenderGraph::build_with_layout(
            &streams,
            PlacementGraphLayout {
                source_bytes: 8_192,
                placement_bytes: 16_384,
                compact_bytes: 1_024,
                indirect_bytes: 16,
            },
        )
        .unwrap();
        assert!(graph.has_complete_hazards());
        assert_eq!(graph.passes().len(), 8);
        assert_eq!(graph.declarations().resources().len(), 9);
        assert_eq!(graph.declarations().passes().len(), 8);
        let plan = graph.compile().unwrap();
        assert_eq!(plan.pass_count(), 8);
        assert!(plan.barrier_count() >= 8);
    }

    #[test]
    fn graph_rejects_duplicate_sector_revision_before_publication() {
        let stream = cell(1, 4, 1).stream();
        assert!(matches!(
            PlacementRenderGraph::build(&[stream, stream]),
            Err(PlacementLifecycleError::DuplicateSector { .. })
        ));
    }

    #[test]
    fn rhi_mapping_keeps_indirect_and_scene_access_distinct() {
        let uses = PlacementGraphStage::DrawIndirect.rhi_uses();
        assert!(uses.iter().any(|use_kind| {
            matches!(
                use_kind,
                PlacementRhiUse::Buffer {
                    resource: PlacementGraphResource::Indirect,
                    use_kind: BufferUse::Indirect,
                }
            )
        }));
        assert!(uses.iter().any(|use_kind| {
            matches!(
                use_kind,
                PlacementRhiUse::ColorAttachment {
                    resource: PlacementGraphResource::Scene
                }
            )
        }));
    }
}
