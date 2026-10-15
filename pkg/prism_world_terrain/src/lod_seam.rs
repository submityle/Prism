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

/// A neighbor declaration for one directed edge of a tile.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LodSeamNeighbor {
    edge: TileEdge,
    tile: TileKey,
}

impl LodSeamNeighbor {
    /// Creates a neighbor declaration. Adjacency and LOD difference are
    /// validated when a seam plan is built.
    pub const fn new(edge: TileEdge, tile: TileKey) -> Self {
        Self { edge, tile }
    }

    /// Edge on the owner tile.
    pub const fn edge(self) -> TileEdge {
        self.edge
    }

    /// Declared adjacent tile.
    pub const fn tile(self) -> TileKey {
        self.tile
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
        if segments < 2 || !segments.is_power_of_two() || !segments.is_multiple_of(2) {
            return Err(LodSeamError::InvalidSegments);
        }
        if segments > limits.max_segments {
            return Err(LodSeamError::SegmentLimit);
        }
        if !morph_alpha.is_finite() {
            return Err(LodSeamError::NonFiniteMorphAlpha);
        }
        if !(0.0..=1.0).contains(&morph_alpha) {
            return Err(LodSeamError::MorphAlphaOutOfRange);
        }

        let mut declared = [None; 4];
        for neighbor in neighbors {
            let index = edge_index(neighbor.edge);
            if declared[index].is_some() {
                return Err(LodSeamError::DuplicateEdge(neighbor.edge));
            }
            declared[index] = Some(neighbor.tile);
        }

        let mut modes = [LodSeamEdgeMode::Boundary; 4];
        let mut transitions = Vec::new();
        let coarse_edges = declared.iter().filter(|tile| tile.is_some()).count();
        transitions
            .try_reserve_exact(coarse_edges)
            .map_err(|_| LodSeamError::AllocationFailed)?;

        let mut cost = LodSeamCost::default();
        for edge in TileEdge::ALL {
            let index = edge_index(edge);
            let Some(neighbor) = declared[index] else {
                continue;
            };
            let mode = validate_neighbor(tile, edge, neighbor)?;
            modes[index] = mode;
            if mode != LodSeamEdgeMode::Coarser {
                continue;
            }
            let transition = build_transition(edge, neighbor, segments, morph_alpha)?;
            cost.edges = cost
                .edges
                .checked_add(1)
                .ok_or(LodSeamError::SizeOverflow)?;
            cost.vertices = cost
                .vertices
                .checked_add(transition.vertices.len())
                .ok_or(LodSeamError::SizeOverflow)?;
            cost.morph_vertices = cost
                .morph_vertices
                .checked_add(transition.morph_vertices.len())
                .ok_or(LodSeamError::SizeOverflow)?;
            cost.indices = cost
                .indices
                .checked_add(transition.indices.len())
                .ok_or(LodSeamError::SizeOverflow)?;
            let transition_bytes = logical_bytes(&transition)?;
            cost.bytes = cost
                .bytes
                .checked_add(transition_bytes)
                .ok_or(LodSeamError::SizeOverflow)?;
            transitions.push(transition);
        }

        if cost.vertices > limits.max_vertices
            || cost.indices > limits.max_indices
            || cost.bytes > limits.max_bytes
        {
            return Err(LodSeamError::BudgetExceeded);
        }

        Ok(Self {
            tile,
            segments,
            morph_alpha,
            modes,
            transitions,
            cost,
        })
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

    /// Exact logical output accounting.
    pub const fn cost(&self) -> LodSeamCost {
        self.cost
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
    neighbor: TileKey,
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
        neighbor,
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

fn logical_bytes(transition: &LodSeamTransitionEdge) -> Result<usize, LodSeamError> {
    transition
        .vertices
        .len()
        .checked_mul(core::mem::size_of::<LodSeamVertex>())
        .and_then(|bytes| {
            transition
                .morph_vertices
                .len()
                .checked_mul(core::mem::size_of::<LodSeamMorphVertex>())
                .and_then(|morph| bytes.checked_add(morph))
        })
        .and_then(|bytes| {
            transition
                .interface
                .len()
                .checked_mul(core::mem::size_of::<[u32; 2]>())
                .and_then(|interface| bytes.checked_add(interface))
        })
        .and_then(|bytes| {
            transition
                .indices
                .len()
                .checked_mul(core::mem::size_of::<u32>())
                .and_then(|indices| bytes.checked_add(indices))
        })
        .ok_or(LodSeamError::SizeOverflow)
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
}
