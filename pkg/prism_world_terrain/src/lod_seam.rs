//! Deterministic 2:1 planar heightfield LOD seam topology.
//!
//! This module owns the engine-neutral contract between a fine heightfield tile
//! and a same-level or one-level coarser/finer neighbor. A coarser neighbor is
//! represented by a transition strip whose boundary contains only the coarse
//! samples; its three-triangle-per-coarse-segment topology cannot leave a
//! fine-only boundary vertex as a T-junction. The returned morph records let a
//! caller blend the omitted fine boundary samples toward the same coarse
//! segment midpoint during a continuous LOD transition.

use alloc::vec::Vec;

use crate::{TileEdge, TileKey};

/// Version of the planar heightfield 2:1 seam contract.
pub const HEIGHTFIELD_LOD_SEAM_VERSION: u32 = 1;

/// A half-open segment interval on the owner tile's fixed-resolution edge.
///
/// The interval is expressed in the owner edge's `0..=segments` parameter
/// space. Its end is exclusive, so two finer children covering one coarse
/// edge are represented by `[0, segments / 2)` and
/// `[segments / 2, segments)`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LodSeamCoverage {
    start: u32,
    end: u32,
}

impl LodSeamCoverage {
    /// Creates a half-open coverage interval.
    pub const fn new(start: u32, end: u32) -> Self {
        Self { start, end }
    }

    /// Creates coverage for the complete edge.
    pub const fn full(segments: u32) -> Self {
        Self {
            start: 0,
            end: segments,
        }
    }

    /// Inclusive lower segment coordinate.
    pub const fn start(self) -> u32 {
        self.start
    }

    /// Exclusive upper segment coordinate.
    pub const fn end(self) -> u32 {
        self.end
    }

    /// Returns the number of covered segments, or `None` on invalid order.
    pub const fn len(self) -> Option<u32> {
        if self.start < self.end {
            Some(self.end - self.start)
        } else {
            None
        }
    }
}

/// A neighbor declaration for one directed edge of a tile.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LodSeamNeighbor {
    edge: TileEdge,
    tile: TileKey,
    boundary_resolution: u32,
    coverage: LodSeamCoverage,
}

impl LodSeamNeighbor {
    /// Creates a neighbor declaration. Adjacency and LOD difference are
    /// validated when a seam plan is built. This compatibility constructor
    /// supplies the strict builder's default full-edge metadata; callers that
    /// need two finer children on one edge must use [`Self::with_boundary`].
    pub const fn new(edge: TileEdge, tile: TileKey) -> Self {
        Self {
            edge,
            tile,
            boundary_resolution: 0,
            coverage: LodSeamCoverage::new(0, 0),
        }
    }

    /// Creates a neighbor declaration with fixed boundary resolution and
    /// owner-edge coverage metadata.
    pub const fn with_boundary(
        edge: TileEdge,
        tile: TileKey,
        boundary_resolution: u32,
        coverage: LodSeamCoverage,
    ) -> Self {
        Self {
            edge,
            tile,
            boundary_resolution,
            coverage,
        }
    }

    /// Edge on the owner tile.
    pub const fn edge(self) -> TileEdge {
        self.edge
    }

    /// Declared adjacent tile.
    pub const fn tile(self) -> TileKey {
        self.tile
    }

    /// Neighbor boundary segment resolution.
    pub const fn boundary_resolution(self) -> u32 {
        self.boundary_resolution
    }

    /// Portion of the owner edge covered by this neighbor.
    pub const fn coverage(self) -> LodSeamCoverage {
        self.coverage
    }
}

/// Published orientation and coverage metadata for one neighbor edge.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LodSeamNeighborMetadata {
    owner_edge: TileEdge,
    neighbor: TileKey,
    neighbor_edge: TileEdge,
    reversed: bool,
    boundary_resolution: u32,
    coverage: LodSeamCoverage,
}

impl LodSeamNeighborMetadata {
    /// Edge on the owner tile.
    pub const fn owner_edge(self) -> TileEdge {
        self.owner_edge
    }

    /// Declared adjacent tile.
    pub const fn neighbor(self) -> TileKey {
        self.neighbor
    }

    /// Edge on the neighbor tile facing the owner edge.
    pub const fn neighbor_edge(self) -> TileEdge {
        self.neighbor_edge
    }

    /// Whether owner edge parameter order is reversed on the neighbor edge.
    pub const fn reversed(self) -> bool {
        self.reversed
    }

    /// Neighbor boundary segment resolution.
    pub const fn boundary_resolution(self) -> u32 {
        self.boundary_resolution
    }

    /// Coverage interval in owner-edge parameter space.
    pub const fn coverage(self) -> LodSeamCoverage {
        self.coverage
    }
}

/// Admission limits for one seam plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LodSeamLimits {
    /// Maximum fine grid segments along one edge.
    pub max_segments: u32,
    /// Maximum generated transition vertices across all coarser edges.
    pub max_vertices: usize,
    /// Maximum generated index values across all coarser edges.
    pub max_indices: usize,
    /// Maximum logical output bytes: grid points, morph records, interface
    /// points, and indices. Container/allocator overhead is excluded.
    pub max_bytes: usize,
}

/// A transition mode on one owner-tile edge.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LodSeamEdgeMode {
    /// No neighbor was declared; the edge is an external boundary.
    Boundary,
    /// The neighbor has the same LOD.
    Same,
    /// The neighbor is one level coarser; this tile owns a transition strip.
    Coarser,
    /// The neighbor is one level finer; the neighbor owns the transition strip.
    Finer,
}

/// A grid-space vertex used by a transition strip.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LodSeamVertex {
    /// Fine-grid integer coordinate `[u, v]` in the owner tile.
    pub grid: [u32; 2],
}

/// A fine boundary vertex and its coarse-segment morph target.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LodSeamMorphVertex {
    /// Fine-grid boundary coordinate that is omitted from the coarse interface.
    pub source: [u32; 2],
    /// First endpoint of the containing coarse boundary segment.
    pub target_a: [u32; 2],
    /// Second endpoint of the containing coarse boundary segment.
    pub target_b: [u32; 2],
    /// Transition alpha in `[0, 1]`; zero is fine geometry and one is coarse.
    pub alpha: f32,
}

/// One deterministic transition strip for an edge with a coarser neighbor.
#[derive(Clone, Debug, PartialEq)]
pub struct LodSeamTransitionEdge {
    edge: TileEdge,
    neighbor: TileKey,
    neighbor_edge: TileEdge,
    reversed: bool,
    boundary_resolution: u32,
    coverage: LodSeamCoverage,
    vertices: Vec<LodSeamVertex>,
    interface: Vec<[u32; 2]>,
    morph_vertices: Vec<LodSeamMorphVertex>,
    indices: Vec<u32>,
}

impl LodSeamTransitionEdge {
    /// Owner-tile edge represented by this strip.
    pub const fn edge(&self) -> TileEdge {
        self.edge
    }

    /// Coarser neighbor consumed by this strip.
    pub const fn neighbor(&self) -> TileKey {
        self.neighbor
    }

    /// Edge on the neighbor tile facing this transition.
    pub const fn neighbor_edge(&self) -> TileEdge {
        self.neighbor_edge
    }

    /// Whether this transition's owner-edge order is reversed on the neighbor.
    pub const fn reversed(&self) -> bool {
        self.reversed
    }

    /// Neighbor boundary segment resolution.
    pub const fn boundary_resolution(&self) -> u32 {
        self.boundary_resolution
    }

    /// Coverage interval in owner-edge parameter space.
    pub const fn coverage(&self) -> LodSeamCoverage {
        self.coverage
    }

    /// Unique grid vertices referenced by [`Self::indices`].
    pub fn vertices(&self) -> &[LodSeamVertex] {
        &self.vertices
    }

    /// Coarse interface points in increasing edge-parameter order.
    pub fn interface(&self) -> &[[u32; 2]] {
        &self.interface
    }

    /// Fine boundary points and their continuous morph targets.
    pub fn morph_vertices(&self) -> &[LodSeamMorphVertex] {
        &self.morph_vertices
    }

    /// Triangle-list indices for the transition strip.
    pub fn indices(&self) -> &[u32] {
        &self.indices
    }
}

/// Exact logical output accounting for one seam plan.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LodSeamCost {
    /// Number of coarser-neighbor transition strips.
    pub edges: usize,
    /// Number of transition vertices.
    pub vertices: usize,
    /// Number of morph records.
    pub morph_vertices: usize,
    /// Number of index values.
    pub indices: usize,
    /// Logical payload bytes covered by [`LodSeamLimits::max_bytes`].
    pub bytes: usize,
}

/// A complete validated 2:1 seam plan for one tile.
#[derive(Clone, Debug, PartialEq)]
pub struct HeightfieldLodSeamPlan {
    tile: TileKey,
    segments: u32,
    morph_alpha: f32,
    modes: [LodSeamEdgeMode; 4],
    neighbor_metadata: Vec<LodSeamNeighborMetadata>,
    transitions: Vec<LodSeamTransitionEdge>,
    cost: LodSeamCost,
}

impl HeightfieldLodSeamPlan {
    /// Builds a deterministic planar 2:1 seam plan.
    ///
    /// `neighbors` may be supplied in any order, but must contain at most one
    /// declaration for each edge. Missing declarations are external boundaries.
    /// The fine grid segment count must be an even power of two so every coarse
    /// segment has exactly two fine segments.
    pub fn build(
        tile: TileKey,
        segments: u32,
        neighbors: &[LodSeamNeighbor],
        morph_alpha: f32,
        limits: LodSeamLimits,
    ) -> Result<Self, LodSeamError> {
        validate_morph_alpha(morph_alpha)?;
        build_internal(tile, segments, neighbors, limits, false, morph_alpha)
    }

    /// Builds the strict fixed-resolution 2:1 seam contract.
    ///
    /// Unlike [`Self::build`], this API derives morph alpha from the validated
    /// LOD relation: a coarser neighbor publishes alpha `1.0`, while same-level
    /// and finer relations publish `0.0`. Each neighbor must carry the fixed
    /// `segments` boundary resolution. A finer edge must contain exactly two
    /// non-overlapping coverage intervals that cover the complete edge.
    pub fn build_strict(
        tile: TileKey,
        segments: u32,
        neighbors: &[LodSeamNeighbor],
        limits: LodSeamLimits,
    ) -> Result<Self, LodSeamError> {
        build_internal(tile, segments, neighbors, limits, true, 0.0)
    }

    /// Owner tile.
    pub const fn tile(&self) -> TileKey {
        self.tile
    }

    /// Fine-grid segment count.
    pub const fn segments(&self) -> u32 {
        self.segments
    }

    /// Plan-wide morph alpha.
    pub const fn morph_alpha(&self) -> f32 {
        self.morph_alpha
    }

    /// Edge modes in [`TileEdge::ALL`] order.
    pub const fn modes(&self) -> [LodSeamEdgeMode; 4] {
        self.modes
    }

    /// Transition strips in stable edge order.
    pub fn transitions(&self) -> &[LodSeamTransitionEdge] {
        &self.transitions
    }

    /// Published neighbor edge, reversal, resolution, and coverage metadata.
    pub fn neighbor_metadata(&self) -> &[LodSeamNeighborMetadata] {
        &self.neighbor_metadata
    }

    /// Exact logical output accounting.
    pub const fn cost(&self) -> LodSeamCost {
        self.cost
    }
}

/// Atomically replaceable CPU seam state.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct HeightfieldLodSeamState {
    plan: Option<HeightfieldLodSeamPlan>,
}

impl HeightfieldLodSeamState {
    /// Creates empty seam state.
    pub const fn new() -> Self {
        Self { plan: None }
    }

    /// Creates state containing an already validated plan.
    pub const fn from_plan(plan: HeightfieldLodSeamPlan) -> Self {
        Self { plan: Some(plan) }
    }

    /// Returns the currently published plan.
    pub fn plan(&self) -> Option<&HeightfieldLodSeamPlan> {
        self.plan.as_ref()
    }

    /// Atomically builds and publishes a strict plan.
    ///
    /// The temporary plan is fully admitted and constructed before `self` is
    /// changed. Validation, budget overflow, or fallible allocation therefore
    /// leaves the previous plan untouched.
    pub fn replace_strict(
        &mut self,
        tile: TileKey,
        segments: u32,
        neighbors: &[LodSeamNeighbor],
        limits: LodSeamLimits,
    ) -> Result<(), LodSeamError> {
        let next = HeightfieldLodSeamPlan::build_strict(tile, segments, neighbors, limits)?;
        self.plan = Some(next);
        Ok(())
    }

    /// Atomically builds and publishes a compatibility plan with explicit
    /// morph alpha.
    pub fn replace(
        &mut self,
        tile: TileKey,
        segments: u32,
        neighbors: &[LodSeamNeighbor],
        morph_alpha: f32,
        limits: LodSeamLimits,
    ) -> Result<(), LodSeamError> {
        let next = HeightfieldLodSeamPlan::build(tile, segments, neighbors, morph_alpha, limits)?;
        self.plan = Some(next);
        Ok(())
    }
}

/// Seam validation, topology, or bounded-allocation failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LodSeamError {
    /// The fine segment count is not an even power of two of at least two.
    InvalidSegments,
    /// The requested segment count exceeds the caller's limit.
    SegmentLimit,
    /// A neighbor edge was declared more than once.
    DuplicateEdge(TileEdge),
    /// A strict neighbor omitted fixed boundary metadata.
    MissingBoundaryMetadata(TileEdge),
    /// A neighbor boundary resolution differs from the fixed owner resolution.
    BoundaryResolutionMismatch {
        /// Owner edge receiving the invalid declaration.
        edge: TileEdge,
        /// Required fixed boundary resolution.
        expected: u32,
        /// Supplied neighbor boundary resolution.
        actual: u32,
    },
    /// A coverage interval is empty or outside the owner edge.
    InvalidCoverage(TileEdge),
    /// A non-finer relation did not cover the complete owner edge.
    CoverageMustBeFull(TileEdge),
    /// One finer child was declared without its sibling coverage.
    IncompleteFinerCoverage(TileEdge),
    /// Finer coverage intervals have a gap or overlap.
    CoverageGapOrOverlap(TileEdge),
    /// A finer coverage interval has the wrong 2:1 length.
    CoverageLengthMismatch(TileEdge),
    /// More than two neighbor declarations were supplied for one edge.
    TooManyNeighbors(TileEdge),
    /// A neighbor is not the tile sharing the declared edge.
    NonAdjacent(TileEdge),
    /// Neighbor LOD differs by more than one level.
    LodDeltaTooLarge(TileEdge),
    /// A checked count or byte calculation overflowed.
    SizeOverflow,
    /// A logical output budget was exceeded before publication.
    BudgetExceeded,
    /// A bounded vector reservation failed.
    AllocationFailed,
    /// Morph alpha was NaN or infinity.
    NonFiniteMorphAlpha,
    /// Morph alpha was outside the closed unit interval.
    MorphAlphaOutOfRange,
    /// Generated topology contained a degenerate triangle.
    DegenerateTriangle,
}

#[derive(Clone, Copy)]
struct PreparedNeighbor {
    tile: TileKey,
    mode: LodSeamEdgeMode,
    neighbor_edge: TileEdge,
    reversed: bool,
    boundary_resolution: u32,
    coverage: LodSeamCoverage,
}

fn validate_morph_alpha(morph_alpha: f32) -> Result<(), LodSeamError> {
    if !morph_alpha.is_finite() {
        return Err(LodSeamError::NonFiniteMorphAlpha);
    }
    if !(0.0..=1.0).contains(&morph_alpha) {
        return Err(LodSeamError::MorphAlphaOutOfRange);
    }
    Ok(())
}

fn validate_segments(segments: u32, limits: LodSeamLimits) -> Result<(), LodSeamError> {
    if segments < 2 || !segments.is_power_of_two() || !segments.is_multiple_of(2) {
        return Err(LodSeamError::InvalidSegments);
    }
    if segments > limits.max_segments {
        return Err(LodSeamError::SegmentLimit);
    }
    Ok(())
}

fn build_internal(
    tile: TileKey,
    segments: u32,
    neighbors: &[LodSeamNeighbor],
    limits: LodSeamLimits,
    strict: bool,
    compatibility_alpha: f32,
) -> Result<HeightfieldLodSeamPlan, LodSeamError> {
    validate_segments(segments, limits)?;

    let mut groups = [[None; 2]; 4];
    for neighbor in neighbors {
        let index = edge_index(neighbor.edge);
        let slot = if groups[index][0].is_none() {
            0
        } else if groups[index][1].is_none() {
            if !strict {
                return Err(LodSeamError::DuplicateEdge(neighbor.edge));
            }
            1
        } else {
            return Err(LodSeamError::TooManyNeighbors(neighbor.edge));
        };

        let (boundary_resolution, coverage) = if neighbor.boundary_resolution == 0
            && neighbor.coverage == LodSeamCoverage::new(0, 0)
        {
            if strict {
                return Err(LodSeamError::MissingBoundaryMetadata(neighbor.edge));
            }
            (segments, LodSeamCoverage::full(segments))
        } else {
            (neighbor.boundary_resolution, neighbor.coverage)
        };
        if boundary_resolution != segments {
            return Err(LodSeamError::BoundaryResolutionMismatch {
                edge: neighbor.edge,
                expected: segments,
                actual: boundary_resolution,
            });
        }
        if coverage.start >= coverage.end || coverage.end > segments {
            return Err(LodSeamError::InvalidCoverage(neighbor.edge));
        }

        let mode = validate_neighbor(tile, neighbor.edge, neighbor.tile)?;
        let (_, neighbor_edge) = tile.neighbor_with_edge(neighbor.edge);
        let reversed = edge_reversed(tile, neighbor.edge, neighbor_edge);
        groups[index][slot] = Some(PreparedNeighbor {
            tile: neighbor.tile,
            mode,
            neighbor_edge,
            reversed,
            boundary_resolution,
            coverage,
        });
    }

    let mut modes = [LodSeamEdgeMode::Boundary; 4];
    let mut metadata_count = 0_usize;
    let mut transition_count = 0_usize;
    let mut cost = LodSeamCost::default();
    let coarse_cost = transition_cost(segments)?;
    for edge in TileEdge::ALL {
        let index = edge_index(edge);
        let first = groups[index][0];
        let second = groups[index][1];
        let Some(first) = first else {
            continue;
        };
        if let Some(second) = second {
            if !strict {
                return Err(LodSeamError::DuplicateEdge(edge));
            }
            validate_finer_pair(edge, segments, first, second)?;
            if first.mode != LodSeamEdgeMode::Finer || second.mode != LodSeamEdgeMode::Finer {
                return Err(LodSeamError::DuplicateEdge(edge));
            }
            modes[index] = LodSeamEdgeMode::Finer;
            metadata_count = metadata_count
                .checked_add(2)
                .ok_or(LodSeamError::SizeOverflow)?;
        } else {
            if strict && first.mode == LodSeamEdgeMode::Finer {
                return Err(LodSeamError::IncompleteFinerCoverage(edge));
            }
            if strict && first.coverage != LodSeamCoverage::full(segments) {
                return Err(LodSeamError::CoverageMustBeFull(edge));
            }
            modes[index] = first.mode;
            metadata_count = metadata_count
                .checked_add(1)
                .ok_or(LodSeamError::SizeOverflow)?;
        }
        if first.mode == LodSeamEdgeMode::Coarser {
            transition_count = transition_count
                .checked_add(1)
                .ok_or(LodSeamError::SizeOverflow)?;
            cost.edges = transition_count;
            cost.vertices = cost
                .vertices
                .checked_add(coarse_cost.vertices)
                .ok_or(LodSeamError::SizeOverflow)?;
            cost.morph_vertices = cost
                .morph_vertices
                .checked_add(coarse_cost.morph_vertices)
                .ok_or(LodSeamError::SizeOverflow)?;
            cost.indices = cost
                .indices
                .checked_add(coarse_cost.indices)
                .ok_or(LodSeamError::SizeOverflow)?;
            cost.bytes = cost
                .bytes
                .checked_add(coarse_cost.bytes)
                .ok_or(LodSeamError::SizeOverflow)?;
        }
    }

    if cost.vertices > limits.max_vertices
        || cost.indices > limits.max_indices
        || cost.bytes > limits.max_bytes
    {
        return Err(LodSeamError::BudgetExceeded);
    }

    // This is the first heap allocation in the build. All output counts and
    // budgets above have already been admitted.
    let mut neighbor_metadata = Vec::new();
    neighbor_metadata
        .try_reserve_exact(metadata_count)
        .map_err(|_| LodSeamError::AllocationFailed)?;
    let mut transitions = Vec::new();
    transitions
        .try_reserve_exact(transition_count)
        .map_err(|_| LodSeamError::AllocationFailed)?;

    let strict_alpha = if transition_count == 0 { 0.0 } else { 1.0 };
    let morph_alpha = if strict {
        strict_alpha
    } else {
        compatibility_alpha
    };
    for edge in TileEdge::ALL {
        let index = edge_index(edge);
        let Some(first) = groups[index][0] else {
            continue;
        };
        let second = groups[index][1];
        let (first, second) = if let Some(second) = second {
            if first.coverage.start <= second.coverage.start {
                (first, second)
            } else {
                (second, first)
            }
        } else {
            (first, first)
        };
        let count = if groups[index][1].is_some() { 2 } else { 1 };
        for prepared in [first, second].into_iter().take(count) {
            neighbor_metadata.push(LodSeamNeighborMetadata {
                owner_edge: edge,
                neighbor: prepared.tile,
                neighbor_edge: prepared.neighbor_edge,
                reversed: prepared.reversed,
                boundary_resolution: prepared.boundary_resolution,
                coverage: prepared.coverage,
            });
            if prepared.mode == LodSeamEdgeMode::Coarser {
                transitions.push(build_transition(edge, prepared, segments, morph_alpha)?);
            }
        }
    }

    Ok(HeightfieldLodSeamPlan {
        tile,
        segments,
        morph_alpha,
        modes,
        neighbor_metadata,
        transitions,
        cost,
    })
}

fn validate_finer_pair(
    edge: TileEdge,
    segments: u32,
    first: PreparedNeighbor,
    second: PreparedNeighbor,
) -> Result<(), LodSeamError> {
    let (low, high) = if first.coverage.start <= second.coverage.start {
        (first, second)
    } else {
        (second, first)
    };
    let half = segments / 2;
    if low.coverage.len() != Some(half) || high.coverage.len() != Some(half) {
        return Err(LodSeamError::CoverageLengthMismatch(edge));
    }
    if low.coverage.start != 0
        || low.coverage.end != high.coverage.start
        || high.coverage.end != segments
    {
        return Err(LodSeamError::CoverageGapOrOverlap(edge));
    }
    Ok(())
}

fn transition_cost(segments: u32) -> Result<LodSeamCost, LodSeamError> {
    let coarse_segments = usize::try_from(segments / 2).map_err(|_| LodSeamError::SizeOverflow)?;
    let fine_segments = usize::try_from(segments).map_err(|_| LodSeamError::SizeOverflow)?;
    let vertices = coarse_segments
        .checked_add(1)
        .and_then(|boundary| {
            fine_segments
                .checked_add(1)
                .and_then(|inner| boundary.checked_add(inner))
        })
        .ok_or(LodSeamError::SizeOverflow)?;
    let indices = coarse_segments
        .checked_mul(9)
        .ok_or(LodSeamError::SizeOverflow)?;
    let bytes = vertices
        .checked_mul(core::mem::size_of::<LodSeamVertex>())
        .and_then(|value| {
            value.checked_add(
                coarse_segments.checked_mul(core::mem::size_of::<LodSeamMorphVertex>())?,
            )
        })
        .and_then(|value| {
            value.checked_add((coarse_segments + 1).checked_mul(core::mem::size_of::<[u32; 2]>())?)
        })
        .and_then(|value| value.checked_add(indices.checked_mul(core::mem::size_of::<u32>())?))
        .ok_or(LodSeamError::SizeOverflow)?;
    Ok(LodSeamCost {
        edges: 1,
        vertices,
        morph_vertices: coarse_segments,
        indices,
        bytes,
    })
}

fn edge_index(edge: TileEdge) -> usize {
    match edge {
        TileEdge::West => 0,
        TileEdge::East => 1,
        TileEdge::South => 2,
        TileEdge::North => 3,
    }
}

fn validate_neighbor(
    tile: TileKey,
    edge: TileEdge,
    neighbor: TileKey,
) -> Result<LodSeamEdgeMode, LodSeamError> {
    let (same_level, neighbor_edge) = tile.neighbor_with_edge(edge);
    let owner_level = i16::from(tile.level());
    let neighbor_level = i16::from(neighbor.level());
    let delta = neighbor_level - owner_level;
    if !(-1..=1).contains(&delta) {
        return Err(LodSeamError::LodDeltaTooLarge(edge));
    }
    match delta {
        0 if neighbor == same_level => Ok(LodSeamEdgeMode::Same),
        0 => Err(LodSeamError::NonAdjacent(edge)),
        -1 if same_level.parent() == Some(neighbor) => Ok(LodSeamEdgeMode::Coarser),
        1 if neighbor.parent() == Some(same_level)
            && child_touches_edge(neighbor, neighbor_edge) =>
        {
            Ok(LodSeamEdgeMode::Finer)
        }
        _ => Err(LodSeamError::NonAdjacent(edge)),
    }
}

fn edge_reversed(tile: TileKey, edge: TileEdge, neighbor_edge: TileEdge) -> bool {
    let same_level = tile.neighbor(edge);
    if same_level.face() == tile.face() || tile.side_len() <= 1 {
        return false;
    }
    let side = tile.side_len();
    let owner_index = edge_parameter(tile, edge);
    let probe_index = if owner_index + 1 < side {
        owner_index + 1
    } else {
        owner_index - 1
    };
    let probe = match edge {
        TileEdge::West | TileEdge::East => {
            TileKey::new(tile.face(), tile.level(), tile.x(), probe_index)
        }
        TileEdge::South | TileEdge::North => {
            TileKey::new(tile.face(), tile.level(), probe_index, tile.y())
        }
    };
    let Ok(probe) = probe else {
        return false;
    };
    let (probe_neighbor, probe_edge) = probe.neighbor_with_edge(edge);
    if probe_edge != neighbor_edge {
        return false;
    }
    let current_parameter = edge_parameter(same_level, neighbor_edge);
    let probe_parameter = edge_parameter(probe_neighbor, neighbor_edge);
    if probe_index > owner_index {
        probe_parameter < current_parameter
    } else {
        probe_parameter > current_parameter
    }
}

fn edge_parameter(tile: TileKey, edge: TileEdge) -> u32 {
    match edge {
        TileEdge::West | TileEdge::East => tile.y(),
        TileEdge::South | TileEdge::North => tile.x(),
    }
}

fn child_touches_edge(child: TileKey, edge: TileEdge) -> bool {
    let Some(parent) = child.parent() else {
        return false;
    };
    match edge {
        TileEdge::West => child.x() == parent.x() * 2,
        TileEdge::East => child.x() == parent.x() * 2 + 1,
        TileEdge::South => child.y() == parent.y() * 2,
        TileEdge::North => child.y() == parent.y() * 2 + 1,
    }
}

fn build_transition(
    edge: TileEdge,
    neighbor: PreparedNeighbor,
    segments: u32,
    morph_alpha: f32,
) -> Result<LodSeamTransitionEdge, LodSeamError> {
    let coarse_segments = segments / 2;
    let vertex_count = usize::try_from(coarse_segments)
        .ok()
        .and_then(|count| count.checked_add(1))
        .and_then(|boundary| {
            usize::try_from(segments)
                .ok()
                .and_then(|fine| fine.checked_add(1))
                .and_then(|inner| boundary.checked_add(inner))
        })
        .ok_or(LodSeamError::SizeOverflow)?;
    let morph_count = usize::try_from(coarse_segments).map_err(|_| LodSeamError::SizeOverflow)?;
    let index_count = morph_count
        .checked_mul(9)
        .ok_or(LodSeamError::SizeOverflow)?;
    let mut vertices = Vec::new();
    vertices
        .try_reserve_exact(vertex_count)
        .map_err(|_| LodSeamError::AllocationFailed)?;
    let mut interface = Vec::new();
    interface
        .try_reserve_exact(
            morph_count
                .checked_add(1)
                .ok_or(LodSeamError::SizeOverflow)?,
        )
        .map_err(|_| LodSeamError::AllocationFailed)?;
    let mut morph_vertices = Vec::new();
    morph_vertices
        .try_reserve_exact(morph_count)
        .map_err(|_| LodSeamError::AllocationFailed)?;
    let mut indices = Vec::new();
    indices
        .try_reserve_exact(index_count)
        .map_err(|_| LodSeamError::AllocationFailed)?;

    for t in (0..=segments).step_by(2) {
        interface.push(edge_point(edge, segments, t, false));
    }
    for t in (0..=segments).step_by(2) {
        vertices.push(LodSeamVertex {
            grid: edge_point(edge, segments, t, false),
        });
    }
    for t in 0..=segments {
        vertices.push(LodSeamVertex {
            grid: edge_point(edge, segments, t, true),
        });
    }
    for k in 0..coarse_segments {
        let t = k.checked_mul(2).ok_or(LodSeamError::SizeOverflow)?;
        morph_vertices.push(LodSeamMorphVertex {
            source: edge_point(edge, segments, t + 1, false),
            target_a: edge_point(edge, segments, t, false),
            target_b: edge_point(edge, segments, t + 2, false),
            alpha: morph_alpha,
        });
        let boundary_a = usize::try_from(k).map_err(|_| LodSeamError::SizeOverflow)?;
        let boundary_b = boundary_a
            .checked_add(1)
            .ok_or(LodSeamError::SizeOverflow)?;
        let inner_base = usize::try_from(coarse_segments)
            .ok()
            .and_then(|value| value.checked_add(1))
            .ok_or(LodSeamError::SizeOverflow)?;
        let inner_a = inner_base
            .checked_add(usize::try_from(t).map_err(|_| LodSeamError::SizeOverflow)?)
            .ok_or(LodSeamError::SizeOverflow)?;
        let inner_m = inner_a.checked_add(1).ok_or(LodSeamError::SizeOverflow)?;
        let inner_b = inner_a.checked_add(2).ok_or(LodSeamError::SizeOverflow)?;
        push_oriented_triangle(&mut indices, boundary_a, boundary_b, inner_b, &vertices)?;
        push_oriented_triangle(&mut indices, boundary_a, inner_b, inner_m, &vertices)?;
        push_oriented_triangle(&mut indices, boundary_a, inner_m, inner_a, &vertices)?;
    }

    Ok(LodSeamTransitionEdge {
        edge,
        neighbor: neighbor.tile,
        neighbor_edge: neighbor.neighbor_edge,
        reversed: neighbor.reversed,
        boundary_resolution: neighbor.boundary_resolution,
        coverage: neighbor.coverage,
        vertices,
        interface,
        morph_vertices,
        indices,
    })
}

fn edge_point(edge: TileEdge, segments: u32, t: u32, inner: bool) -> [u32; 2] {
    match edge {
        TileEdge::West => [u32::from(inner), t],
        TileEdge::East => [segments - u32::from(inner), t],
        TileEdge::South => [t, u32::from(inner)],
        TileEdge::North => [t, segments - u32::from(inner)],
    }
}

fn push_oriented_triangle(
    indices: &mut Vec<u32>,
    a: usize,
    b: usize,
    c: usize,
    vertices: &[LodSeamVertex],
) -> Result<(), LodSeamError> {
    let pa = vertices[a].grid;
    let pb = vertices[b].grid;
    let pc = vertices[c].grid;
    let area = (i64::from(pb[0]) - i64::from(pa[0])) * (i64::from(pc[1]) - i64::from(pa[1]))
        - (i64::from(pb[1]) - i64::from(pa[1])) * (i64::from(pc[0]) - i64::from(pa[0]));
    if area == 0 {
        return Err(LodSeamError::DegenerateTriangle);
    }
    let a = u32::try_from(a).map_err(|_| LodSeamError::SizeOverflow)?;
    let b = u32::try_from(b).map_err(|_| LodSeamError::SizeOverflow)?;
    let c = u32::try_from(c).map_err(|_| LodSeamError::SizeOverflow)?;
    if area < 0 {
        indices.extend_from_slice(&[a, b, c]);
    } else {
        indices.extend_from_slice(&[a, c, b]);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CubeFace;

    fn limits() -> LodSeamLimits {
        LodSeamLimits {
            max_segments: 1 << 10,
            max_vertices: 1 << 20,
            max_indices: 1 << 22,
            max_bytes: 1 << 24,
        }
    }

    fn tile(level: u8, x: u32, y: u32) -> TileKey {
        TileKey::new(CubeFace::PositiveZ, level, x, y).unwrap()
    }

    #[test]
    fn same_level_and_boundary_edges_need_no_transition() {
        let owner = tile(3, 2, 2);
        let east = owner.neighbor(TileEdge::East);
        let plan = HeightfieldLodSeamPlan::build(
            owner,
            8,
            &[LodSeamNeighbor::new(TileEdge::East, east)],
            0.5,
            limits(),
        )
        .unwrap();
        assert_eq!(
            plan.modes()[edge_index(TileEdge::East)],
            LodSeamEdgeMode::Same
        );
        assert_eq!(
            plan.modes()[edge_index(TileEdge::West)],
            LodSeamEdgeMode::Boundary
        );
        assert!(plan.transitions().is_empty());
    }

    #[test]
    fn coarser_neighbor_has_coarse_interface_and_valid_indices() {
        let owner = tile(4, 4, 4);
        let coarse = owner.neighbor(TileEdge::North).parent().unwrap();
        let plan = HeightfieldLodSeamPlan::build(
            owner,
            8,
            &[LodSeamNeighbor::new(TileEdge::North, coarse)],
            0.75,
            limits(),
        )
        .unwrap();
        let edge = &plan.transitions()[0];
        assert_eq!(edge.edge(), TileEdge::North);
        assert_eq!(edge.interface().len(), 5);
        assert_eq!(edge.morph_vertices().len(), 4);
        assert_eq!(edge.indices().len(), 36);
        assert!(edge
            .indices()
            .iter()
            .all(|index| (*index as usize) < edge.vertices().len()));
        assert!(edge
            .vertices()
            .iter()
            .all(|vertex| vertex.grid[1] == 8 || vertex.grid[1] == 7));
        assert!(edge
            .morph_vertices()
            .iter()
            .all(|morph| morph.source[0] % 2 == 1 && morph.alpha == 0.75));
        assert_eq!(plan.cost().edges, 1);
    }

    #[test]
    fn finer_neighbor_is_valid_but_owner_does_not_duplicate_transition() {
        let owner = tile(3, 2, 2);
        let same_level = owner.neighbor(TileEdge::West);
        let fine = same_level.children().unwrap()[3];
        let plan = HeightfieldLodSeamPlan::build(
            owner,
            8,
            &[LodSeamNeighbor::new(TileEdge::West, fine)],
            0.25,
            limits(),
        )
        .unwrap();
        assert_eq!(
            plan.modes()[edge_index(TileEdge::West)],
            LodSeamEdgeMode::Finer
        );
        assert!(plan.transitions().is_empty());

        let interior_fine = same_level.children().unwrap()[2];
        assert_eq!(
            HeightfieldLodSeamPlan::build(
                owner,
                8,
                &[LodSeamNeighbor::new(TileEdge::West, interior_fine)],
                0.25,
                limits(),
            ),
            Err(LodSeamError::NonAdjacent(TileEdge::West))
        );
    }

    #[test]
    fn neighbor_order_does_not_change_the_plan() {
        let owner = tile(4, 4, 4);
        let north = owner.neighbor(TileEdge::North).parent().unwrap();
        let west = owner.neighbor(TileEdge::West).parent().unwrap();
        let a = HeightfieldLodSeamPlan::build(
            owner,
            16,
            &[
                LodSeamNeighbor::new(TileEdge::North, north),
                LodSeamNeighbor::new(TileEdge::West, west),
            ],
            0.5,
            limits(),
        )
        .unwrap();
        let b = HeightfieldLodSeamPlan::build(
            owner,
            16,
            &[
                LodSeamNeighbor::new(TileEdge::West, west),
                LodSeamNeighbor::new(TileEdge::North, north),
            ],
            0.5,
            limits(),
        )
        .unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn invalid_delta_duplicate_and_budget_fail_closed() {
        let owner = tile(4, 4, 4);
        let too_coarse = owner
            .neighbor(TileEdge::East)
            .parent()
            .unwrap()
            .parent()
            .unwrap();
        assert_eq!(
            HeightfieldLodSeamPlan::build(
                owner,
                8,
                &[LodSeamNeighbor::new(TileEdge::East, too_coarse)],
                0.5,
                limits(),
            ),
            Err(LodSeamError::LodDeltaTooLarge(TileEdge::East))
        );
        let same = owner.neighbor(TileEdge::East);
        assert_eq!(
            HeightfieldLodSeamPlan::build(
                owner,
                8,
                &[
                    LodSeamNeighbor::new(TileEdge::East, same),
                    LodSeamNeighbor::new(TileEdge::East, same),
                ],
                0.5,
                limits(),
            ),
            Err(LodSeamError::DuplicateEdge(TileEdge::East))
        );
        let mut tight = limits();
        tight.max_indices = 1;
        assert_eq!(
            HeightfieldLodSeamPlan::build(
                owner,
                8,
                &[LodSeamNeighbor::new(
                    TileEdge::East,
                    owner.neighbor(TileEdge::East).parent().unwrap(),
                )],
                0.5,
                tight,
            ),
            Err(LodSeamError::BudgetExceeded)
        );
    }

    #[test]
    fn morph_and_segment_validation_is_strict() {
        let owner = tile(3, 2, 2);
        assert_eq!(
            HeightfieldLodSeamPlan::build(owner, 3, &[], 0.5, limits()),
            Err(LodSeamError::InvalidSegments)
        );
        assert_eq!(
            HeightfieldLodSeamPlan::build(owner, 8, &[], 1.1, limits()),
            Err(LodSeamError::MorphAlphaOutOfRange)
        );
        assert_eq!(
            HeightfieldLodSeamPlan::build(owner, 8, &[], f32::NAN, limits()),
            Err(LodSeamError::NonFiniteMorphAlpha)
        );
    }

    #[test]
    fn strict_plan_requires_fixed_resolution_and_derives_morph_alpha() {
        let owner = tile(4, 4, 4);
        let coarse = owner.neighbor(TileEdge::North).parent().unwrap();
        let plan = HeightfieldLodSeamPlan::build_strict(
            owner,
            8,
            &[LodSeamNeighbor::with_boundary(
                TileEdge::North,
                coarse,
                8,
                LodSeamCoverage::full(8),
            )],
            limits(),
        )
        .unwrap();

        assert_eq!(plan.morph_alpha(), 1.0);
        assert_eq!(plan.neighbor_metadata().len(), 1);
        let metadata = plan.neighbor_metadata()[0];
        assert_eq!(metadata.owner_edge(), TileEdge::North);
        assert_eq!(metadata.neighbor(), coarse);
        assert_eq!(metadata.boundary_resolution(), 8);
        assert_eq!(metadata.coverage(), LodSeamCoverage::full(8));
        assert_eq!(metadata.neighbor_edge(), TileEdge::South);
        assert!(!metadata.reversed());

        assert_eq!(
            HeightfieldLodSeamPlan::build_strict(
                owner,
                8,
                &[LodSeamNeighbor::new(TileEdge::North, coarse)],
                limits(),
            ),
            Err(LodSeamError::MissingBoundaryMetadata(TileEdge::North))
        );
    }

    #[test]
    fn strict_finer_children_cover_one_edge_and_publish_order_metadata() {
        let owner = tile(3, 2, 2);
        let same_level = owner.neighbor(TileEdge::West);
        let children = same_level.children().unwrap();
        let plan = HeightfieldLodSeamPlan::build_strict(
            owner,
            8,
            &[
                LodSeamNeighbor::with_boundary(
                    TileEdge::West,
                    children[3],
                    8,
                    LodSeamCoverage::new(4, 8),
                ),
                LodSeamNeighbor::with_boundary(
                    TileEdge::West,
                    children[1],
                    8,
                    LodSeamCoverage::new(0, 4),
                ),
            ],
            limits(),
        )
        .unwrap();

        assert_eq!(
            plan.modes()[edge_index(TileEdge::West)],
            LodSeamEdgeMode::Finer
        );
        assert_eq!(plan.morph_alpha(), 0.0);
        assert!(plan.transitions().is_empty());
        assert_eq!(plan.neighbor_metadata().len(), 2);
        assert_eq!(
            plan.neighbor_metadata()[0].coverage(),
            LodSeamCoverage::new(0, 4)
        );
        assert_eq!(
            plan.neighbor_metadata()[1].coverage(),
            LodSeamCoverage::new(4, 8)
        );
        assert!(plan
            .neighbor_metadata()
            .iter()
            .all(|metadata| metadata.neighbor_edge() == TileEdge::East));
    }

    #[test]
    fn strict_finer_coverage_rejects_gaps_overlap_and_wrong_lengths() {
        let owner = tile(3, 2, 2);
        let same_level = owner.neighbor(TileEdge::West);
        let children = same_level.children().unwrap();
        let make = |a: LodSeamCoverage, b: LodSeamCoverage| {
            HeightfieldLodSeamPlan::build_strict(
                owner,
                8,
                &[
                    LodSeamNeighbor::with_boundary(TileEdge::West, children[1], 8, a),
                    LodSeamNeighbor::with_boundary(TileEdge::West, children[3], 8, b),
                ],
                limits(),
            )
        };

        assert_eq!(
            make(LodSeamCoverage::new(0, 3), LodSeamCoverage::new(3, 8)),
            Err(LodSeamError::CoverageLengthMismatch(TileEdge::West))
        );
        assert_eq!(
            make(LodSeamCoverage::new(0, 4), LodSeamCoverage::new(3, 7)),
            Err(LodSeamError::CoverageGapOrOverlap(TileEdge::West))
        );
        assert_eq!(
            make(LodSeamCoverage::new(0, 3), LodSeamCoverage::new(3, 8)),
            Err(LodSeamError::CoverageLengthMismatch(TileEdge::West))
        );
    }

    #[test]
    fn strict_state_replace_is_atomic_on_validation_or_budget_failure() {
        let owner = tile(4, 4, 4);
        let coarse = owner.neighbor(TileEdge::North).parent().unwrap();
        let valid = [LodSeamNeighbor::with_boundary(
            TileEdge::North,
            coarse,
            8,
            LodSeamCoverage::full(8),
        )];
        let mut state = HeightfieldLodSeamState::new();
        state.replace_strict(owner, 8, &valid, limits()).unwrap();
        let before = state.clone();

        assert_eq!(
            state.replace_strict(
                owner,
                8,
                &[LodSeamNeighbor::new(TileEdge::North, coarse)],
                limits(),
            ),
            Err(LodSeamError::MissingBoundaryMetadata(TileEdge::North))
        );
        assert_eq!(state, before);

        let mut tight = limits();
        tight.max_indices = 1;
        assert_eq!(
            state.replace_strict(owner, 8, &valid, tight),
            Err(LodSeamError::BudgetExceeded)
        );
        assert_eq!(state, before);
    }
}
