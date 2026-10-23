//! Origin-relative cube-sphere geometry derived from baked terrain heights.

use alloc::vec::Vec;
use core::convert::Infallible;

use crate::{BakeError, BakeLimits, CubeFace, HeightfieldTile, TileEdge, TileKey};

/// Borrowed one-ring samples from one neighboring planetary tile.
///
/// Heights are decoded millimetres in the neighboring tile's edge order.
/// edge_mm is the shared edge and inner_mm is the adjacent row one sample
/// inward from that edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlanetHeightfieldBoundarySamples<'a> {
    neighbor: TileKey,
    neighbor_edge: TileEdge,
    edge_mm: &'a [i128],
    inner_mm: &'a [i128],
}

impl<'a> PlanetHeightfieldBoundarySamples<'a> {
    /// Creates a borrowed boundary snapshot for one neighboring tile.
    pub const fn new(
        neighbor: TileKey,
        neighbor_edge: TileEdge,
        edge_mm: &'a [i128],
        inner_mm: &'a [i128],
    ) -> Self {
        Self {
            neighbor,
            neighbor_edge,
            edge_mm,
            inner_mm,
        }
    }
}

/// One source for a cross-tile normal derivative.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlanetHeightfieldNeighbor<'a> {
    /// Read the boundary and inner row from a complete baked neighboring tile.
    Tile(&'a HeightfieldTile),
    /// Read the boundary and inner row from a borrowed snapshot.
    Boundary(PlanetHeightfieldBoundarySamples<'a>),
}

/// Optional west/east/south/north neighborhood for planetary normals.
///
/// Missing edges preserve the legacy one-sided normal. Supplied edges are
/// validated before output allocation and are never partially published.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlanetHeightfieldNormalNeighborhood<'a> {
    neighbors: [Option<PlanetHeightfieldNeighbor<'a>>; 4],
}

impl<'a> PlanetHeightfieldNormalNeighborhood<'a> {
    /// Creates an empty neighborhood, preserving PlanetHeightfieldMesh::build.
    pub const fn empty() -> Self {
        Self {
            neighbors: [None, None, None, None],
        }
    }

    /// Creates a neighborhood in west, east, south, north order.
    pub const fn new(
        west: Option<PlanetHeightfieldNeighbor<'a>>,
        east: Option<PlanetHeightfieldNeighbor<'a>>,
        south: Option<PlanetHeightfieldNeighbor<'a>>,
        north: Option<PlanetHeightfieldNeighbor<'a>>,
    ) -> Self {
        Self {
            neighbors: [west, east, south, north],
        }
    }

    fn get(self, edge: TileEdge) -> Option<PlanetHeightfieldNeighbor<'a>> {
        self.neighbors[edge_index(edge)]
    }
}

/// Cube-sphere placement with a checked bound on numeric position error.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlanetMeshConfig {
    radius_mm: i64,
    origin_mm: [i64; 3],
    max_error_mm: f64,
}

impl PlanetMeshConfig {
    /// Creates a placement contract. Geometry exceeding the error budget fails.
    ///
    /// Radii are limited to 2^52 millimetres so integer radius conversion is exact.
    /// Error bounds cover floating arithmetic and local f32 encoding, not source
    /// height quantization, surface approximation between vertices, or LOD seams.
    pub fn new(
        radius_mm: i64,
        origin_mm: [i64; 3],
        max_error_mm: f64,
    ) -> Result<Self, BakeError<Infallible>> {
        if !(1..=1_i64 << 52).contains(&radius_mm)
            || !max_error_mm.is_finite()
            || max_error_mm <= 0.0
        {
            return Err(BakeError::InvalidGeometry);
        }
        Ok(Self {
            radius_mm,
            origin_mm,
            max_error_mm,
        })
    }

    /// Origin retained as integer millimetres; add it after local-space queries.
    pub const fn origin_mm(self) -> [i64; 3] {
        self.origin_mm
    }

    /// Spherical reference radius in millimetres.
    pub const fn radius_mm(self) -> i64 {
        self.radius_mm
    }

    /// Allowed per-coordinate arithmetic and encoding error in millimetres.
    pub const fn max_error_mm(self) -> f64 {
        self.max_error_mm
    }
}

/// Indexed planet patch. Positions are origin-relative metres, axes stay global.
#[derive(Clone, Debug, PartialEq)]
pub struct PlanetHeightfieldMesh {
    cache_id: u64,
    tile: TileKey,
    spec: crate::HeightfieldSpec,
    revision: u64,
    config: PlanetMeshConfig,
    error_bound_mm: f64,
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    indices: Vec<u32>,
    lod_seams_applied: bool,
    lod_seam_neighbors: [Option<TileKey>; 4],
}

fn planet_mesh_edge_index(edge: TileEdge) -> usize {
    match edge {
        TileEdge::West => 0,
        TileEdge::East => 1,
        TileEdge::South => 2,
        TileEdge::North => 3,
    }
}

fn planet_transition_skipped_segments(edge: TileEdge, coarse_edges: [bool; 4]) -> (bool, bool) {
    match edge {
        TileEdge::West => (coarse_edges[2], coarse_edges[3]),
        TileEdge::East => (coarse_edges[2], coarse_edges[3]),
        TileEdge::South => (coarse_edges[0], coarse_edges[1]),
        TileEdge::North => (coarse_edges[0], coarse_edges[1]),
    }
}

fn offset_grid(value: u32, offset: i64, limit: u32) -> Result<u32, BakeError<Infallible>> {
    let result = i64::from(value)
        .checked_add(offset)
        .and_then(|coordinate| u32::try_from(coordinate).ok())
        .filter(|&coordinate| coordinate <= limit)
        .ok_or(BakeError::InvalidGeometry)?;
    Ok(result)
}

fn append_planet_outward_triangle(
    indices: &mut Vec<u32>,
    positions: &[[f32; 3]],
    origin_mm: [i64; 3],
    triangle: [u32; 3],
) -> Result<(), BakeError<Infallible>> {
    let a = positions
        .get(triangle[0] as usize)
        .ok_or(BakeError::InvalidGeometry)?
        .map(f64::from);
    let b = positions
        .get(triangle[1] as usize)
        .ok_or(BakeError::InvalidGeometry)?
        .map(f64::from);
    let c = positions
        .get(triangle[2] as usize)
        .ok_or(BakeError::InvalidGeometry)?
        .map(f64::from);
    let normal = cross(subtract(b, a), subtract(c, a));
    let origin = origin_mm.map(|value| value as f64 * 0.001);
    let outward = (0..3)
        .map(|axis| normal[axis] * ((a[axis] + b[axis] + c[axis]) / 3.0 + origin[axis]))
        .sum::<f64>();
    if !outward.is_finite() || outward == 0.0 {
        return Err(BakeError::InvalidGeometry);
    }
    if outward > 0.0 {
        indices.extend_from_slice(&triangle);
    } else {
        indices.extend_from_slice(&[triangle[0], triangle[2], triangle[1]]);
    }
    Ok(())
}

impl PlanetHeightfieldMesh {
    /// Projects height samples to a sphere, with outward CCW triangle winding.
    ///
    /// Tile coordinates remain f64 through projection, including level 30.
    /// No temporary full-sized position array is allocated. Payload admission
    /// uses 32 bytes per vertex plus 24 per quad, before any allocation.
    /// Boundary normals are one-sided unless a neighborhood is supplied to
    /// build_with_neighborhood.
    pub fn build(
        field: &HeightfieldTile,
        config: PlanetMeshConfig,
        limits: BakeLimits,
    ) -> Result<Self, BakeError<Infallible>> {
        Self::build_with_neighborhood(
            field,
            config,
            limits,
            PlanetHeightfieldNormalNeighborhood::empty(),
        )
    }

    /// Builds a mesh with optional cross-tile continuous boundary normals.
    ///
    /// Each supplied neighbor must be the declared same-level adjacent tile
    /// with the same sample count and exact shared decoded edge. A boundary
    /// snapshot must contain one complete shared edge and its adjacent inner
    /// row, both in the neighboring tile's edge order. Invalid snapshots fail
    /// with BakeError::InvalidGeometry before any output allocation.
    pub fn build_with_neighborhood(
        field: &HeightfieldTile,
        config: PlanetMeshConfig,
        limits: BakeLimits,
        neighborhood: PlanetHeightfieldNormalNeighborhood<'_>,
    ) -> Result<Self, BakeError<Infallible>> {
        let count = field.samples().len();
        if count > u32::MAX as usize {
            return Err(BakeError::SizeOverflow);
        }
        let side = field.spec().samples_per_edge();
        let indices = usize::try_from(side - 1)
            .ok()
            .and_then(|n| n.checked_mul(n))
            .and_then(|n| n.checked_mul(6))
            .ok_or(BakeError::SizeOverflow)?;
        let bytes = count
            .checked_mul(32)
            .and_then(|n| indices.checked_mul(4).and_then(|i| n.checked_add(i)))
            .ok_or(BakeError::SizeOverflow)?;
        if count > limits.max_samples || bytes > limits.max_bytes {
            return Err(BakeError::BudgetExceeded);
        }
        validate_neighborhood(field, neighborhood)?;
        let mut mesh = Self {
            cache_id: field.cache_id(),
            tile: field.tile(),
            spec: field.spec(),
            revision: field.revision(),
            config,
            error_bound_mm: 0.0,
            positions: reserve(count)?,
            normals: reserve(count)?,
            uvs: reserve(count)?,
            indices: reserve(indices)?,
            lod_seams_applied: false,
            lod_seam_neighbors: [None; 4],
        };
        for y in 0..side {
            for x in 0..side {
                let vertex = project_vertex(field, config, neighborhood, x, y)?;
                mesh.error_bound_mm = mesh.error_bound_mm.max(vertex.error_bound_mm);
                mesh.positions.push(vertex.position);
                mesh.normals.push(vertex.normal);
                mesh.uvs
                    .push([x as f32 / (side - 1) as f32, y as f32 / (side - 1) as f32]);
            }
        }
        for y in 0..side - 1 {
            for x in 0..side - 1 {
                let a = y * side + x;
                let b = a + 1;
                let c = a + side;
                mesh.indices.extend_from_slice(&[a, b, c, b, c + 1, c]);
            }
        }
        Ok(mesh)
    }

    /// Applies the next committed heightfield rebake without reallocating mesh
    /// storage or reprojecting unchanged vertices.
    ///
    /// Positions in the changed sample rectangle and normals in its one-sample
    /// ring are recomputed. The returned rectangle is the conservative vertex
    /// upload region. If a changed boundary uses cross-tile normals, the
    /// supplied neighbor must already contain the matching shared edge; a
    /// mismatch fails before any mesh state changes. Neighbor meshes affected
    /// by the shared edge must be updated by their owner as a separate tile
    /// operation. An active LOD seam/morph must be returned to alpha zero and
    /// cleared before applying a heightfield change, so the incremental update
    /// never mixes morphed and newly projected vertices.
    pub fn update(
        &mut self,
        field: &HeightfieldTile,
        change: crate::GridChange,
        neighborhood: PlanetHeightfieldNormalNeighborhood<'_>,
    ) -> Result<crate::SampleBounds, BakeError<Infallible>> {
        if self.lod_seams_applied {
            return Err(BakeError::InvalidGeometry);
        }
        if !change.matches(field, self.cache_id, self.revision)
            || self.tile != field.tile()
            || self.spec != field.spec()
        {
            return Err(BakeError::StaleChange);
        }
        validate_neighborhood(field, neighborhood)?;
        let side = self.spec.samples_per_edge();
        let updated = change.bounds().expanded(side);

        // Preflight every affected vertex. This keeps the commit phase
        // infallible, so a bad radial height or degenerate normal cannot leave
        // a partially updated mesh visible to the caller.
        for y in updated.min()[1]..=updated.max()[1] {
            for x in updated.min()[0]..=updated.max()[0] {
                let _ = project_vertex(field, self.config, neighborhood, x, y)?;
            }
        }
        let mut next_error_bound = self.error_bound_mm;
        for y in updated.min()[1]..=updated.max()[1] {
            for x in updated.min()[0]..=updated.max()[0] {
                let vertex = project_vertex(field, self.config, neighborhood, x, y)?;
                let index = (y * side + x) as usize;
                self.positions[index] = vertex.position;
                self.normals[index] = vertex.normal;
                next_error_bound = next_error_bound.max(vertex.error_bound_mm);
            }
        }
        self.error_bound_mm = next_error_bound;
        self.revision = change.revision();
        Ok(updated)
    }

    /// Local position payload in metres.
    pub fn positions(&self) -> &[[f32; 3]] {
        &self.positions
    }
    /// Outward geometric unit normals in global-axis orientation.
    pub fn normals(&self) -> &[[f32; 3]] {
        &self.normals
    }
    /// Tile-local normalized UVs.
    pub fn uvs(&self) -> &[[f32; 2]] {
        &self.uvs
    }
    /// Stable spatial tile identity.
    pub const fn tile(&self) -> TileKey {
        self.tile
    }
    /// Outward triangle indices.
    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    #[cfg(feature = "bevy_mesh_adapter")]
    pub(crate) fn into_bevy_parts(self) -> (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<[f32; 2]>, Vec<u32>) {
        (self.positions, self.normals, self.uvs, self.indices)
    }

    pub(crate) const fn lod_seam_neighbors(&self) -> [Option<TileKey>; 4] {
        self.lod_seam_neighbors
    }

    /// Replaces planetary edge quads with resident-derived 2:1 seam strips.
    /// Transition indices reuse this mesh's projected grid vertices, preserving
    /// the origin-relative precision and avoiding duplicate attribute storage.
    /// The complete index stream is staged under `max_bytes` and swapped only
    /// after all topology and index checks succeed.
    ///
    /// Up to two coarse edges are supported. Opposite strips remain separate;
    /// adjacent strips omit their overlapping endpoint segments and share one
    /// six-vertex corner fan. Clear the plan before replacing it.
    pub fn apply_lod_seam_plan(
        &mut self,
        plan: &crate::HeightfieldLodSeamPlan,
        max_bytes: usize,
    ) -> Result<(), BakeError<Infallible>> {
        if self.lod_seams_applied || plan.tile() != self.tile {
            return Err(BakeError::InvalidGeometry);
        }
        let side = self.spec.samples_per_edge();
        let segments = side - 1;
        if plan.segments() != segments {
            return Err(BakeError::InvalidGeometry);
        }
        let transitions = plan.transitions();
        if transitions.len() > 2 {
            return Err(BakeError::InvalidGeometry);
        }
        let mut coarse_edges = [false; 4];
        for transition in transitions {
            let slot = planet_mesh_edge_index(transition.edge());
            if coarse_edges[slot] {
                return Err(BakeError::InvalidGeometry);
            }
            coarse_edges[slot] = true;
        }
        if transitions.is_empty() {
            return Ok(());
        }
        let base_indices = usize::try_from(segments)
            .ok()
            .and_then(|n| n.checked_mul(n))
            .and_then(|n| n.checked_mul(6))
            .ok_or(BakeError::SizeOverflow)?;
        if self.indices.len() != base_indices {
            return Err(BakeError::InvalidGeometry);
        }
        let corner_fans = [
            coarse_edges[0] && coarse_edges[2],
            coarse_edges[1] && coarse_edges[2],
            coarse_edges[0] && coarse_edges[3],
            coarse_edges[1] && coarse_edges[3],
        ];
        let corner_count = corner_fans.iter().filter(|&&fan| fan).count();
        let removed_quads = transitions
            .len()
            .checked_mul(segments as usize)
            .and_then(|n| n.checked_sub(corner_count))
            .ok_or(BakeError::SizeOverflow)?;
        let removed = removed_quads
            .checked_mul(6)
            .ok_or(BakeError::SizeOverflow)?;
        let mut added = 0usize;
        for transition in transitions {
            let (skip_start, skip_end) =
                planet_transition_skipped_segments(transition.edge(), coarse_edges);
            if !transition.indices().len().is_multiple_of(9) {
                return Err(BakeError::InvalidGeometry);
            }
            let skipped = skip_start as usize + skip_end as usize;
            let transition_indices = transition
                .indices()
                .len()
                .checked_sub(skipped.checked_mul(9).ok_or(BakeError::SizeOverflow)?)
                .ok_or(BakeError::InvalidGeometry)?;
            added = added
                .checked_add(transition_indices)
                .ok_or(BakeError::SizeOverflow)?;
        }
        added = added
            .checked_add(
                corner_count
                    .checked_mul(12)
                    .ok_or(BakeError::SizeOverflow)?,
            )
            .ok_or(BakeError::SizeOverflow)?;
        let next_len = base_indices
            .checked_sub(removed)
            .and_then(|n| n.checked_add(added))
            .ok_or(BakeError::SizeOverflow)?;
        let bytes = self
            .positions
            .len()
            .checked_mul(32)
            .and_then(|n| {
                next_len
                    .checked_mul(core::mem::size_of::<u32>())
                    .and_then(|i| n.checked_add(i))
            })
            .ok_or(BakeError::SizeOverflow)?;
        if bytes > max_bytes {
            return Err(BakeError::BudgetExceeded);
        }
        let mut indices = Vec::new();
        indices
            .try_reserve_exact(next_len)
            .map_err(|_| BakeError::AllocationFailed)?;
        for y in 0..segments {
            for x in 0..segments {
                let replaced = (coarse_edges[0] && x == 0)
                    || (coarse_edges[1] && x + 1 == segments)
                    || (coarse_edges[2] && y == 0)
                    || (coarse_edges[3] && y + 1 == segments);
                if !replaced {
                    let first = ((y * segments + x) * 6) as usize;
                    indices.extend_from_slice(&self.indices[first..first + 6]);
                }
            }
        }
        for transition in transitions {
            let (skip_start, skip_end) =
                planet_transition_skipped_segments(transition.edge(), coarse_edges);
            for (segment, segment_indices) in transition.indices().chunks_exact(9).enumerate() {
                if (skip_start && segment == 0)
                    || (skip_end && segment + 1 == transition.indices().len() / 9)
                {
                    continue;
                }
                for local_triangle in segment_indices.chunks_exact(3) {
                    let mut triangle = [0u32; 3];
                    for (slot, &local_index) in local_triangle.iter().enumerate() {
                        let grid = transition
                            .vertices()
                            .get(local_index as usize)
                            .ok_or(BakeError::InvalidGeometry)?
                            .grid;
                        triangle[slot] = grid[1]
                            .checked_mul(side)
                            .and_then(|row| row.checked_add(grid[0]))
                            .ok_or(BakeError::SizeOverflow)?;
                        if triangle[slot] as usize >= self.positions.len() {
                            return Err(BakeError::InvalidGeometry);
                        }
                    }
                    append_planet_outward_triangle(
                        &mut indices,
                        &self.positions,
                        self.config.origin_mm,
                        triangle,
                    )?;
                }
            }
        }
        let limit = segments;
        let corner_coordinates = [
            (0, 0, 1_i64, 1_i64),
            (limit, 0, -1, 1),
            (0, limit, 1, -1),
            (limit, limit, -1, -1),
        ];
        for (corner_index, &(x, y, dx, dy)) in corner_coordinates.iter().enumerate() {
            if !corner_fans[corner_index] {
                continue;
            }
            let corner = [x, y];
            let top_outer = [offset_grid(x, dx * 2, segments)?, y];
            let top_inner_far = [
                offset_grid(x, dx * 2, segments)?,
                offset_grid(y, dy, segments)?,
            ];
            let top_inner_near = [offset_grid(x, dx, segments)?, offset_grid(y, dy, segments)?];
            let side_inner_far = [
                offset_grid(x, dx, segments)?,
                offset_grid(y, dy * 2, segments)?,
            ];
            let side_outer = [x, offset_grid(y, dy * 2, segments)?];
            let patch = [
                [corner, top_outer, top_inner_far],
                [corner, top_inner_far, top_inner_near],
                [corner, top_inner_near, side_inner_far],
                [corner, side_inner_far, side_outer],
            ];
            for triangle in patch {
                let mut vertex_indices = [0u32; 3];
                for (slot, grid) in triangle.into_iter().enumerate() {
                    vertex_indices[slot] = grid[1]
                        .checked_mul(side)
                        .and_then(|row| row.checked_add(grid[0]))
                        .ok_or(BakeError::SizeOverflow)?;
                    if vertex_indices[slot] as usize >= self.positions.len() {
                        return Err(BakeError::InvalidGeometry);
                    }
                }
                append_planet_outward_triangle(
                    &mut indices,
                    &self.positions,
                    self.config.origin_mm,
                    vertex_indices,
                )?;
            }
        }
        if indices.len() != next_len {
            return Err(BakeError::InvalidGeometry);
        }
        self.indices = indices;
        self.lod_seams_applied = true;
        for transition in transitions {
            self.lod_seam_neighbors[planet_mesh_edge_index(transition.edge())] =
                Some(transition.neighbor());
        }
        Ok(())
    }

    /// Applies the continuous 2:1 morph targets carried by a resident seam plan.
    ///
    /// Odd fine-edge vertices move along the local 3D chord toward the midpoint
    /// of their two coarse interface endpoints. The operation updates only
    /// those boundary positions and the adjacent normal stencil; it allocates
    /// no mesh buffers. All identities, neighbor edges, coordinates, positions
    /// and normals are preflighted before the first write.
    pub fn apply_lod_morph(
        &mut self,
        field: &HeightfieldTile,
        plan: &crate::HeightfieldLodSeamPlan,
        alpha: f32,
    ) -> Result<(), BakeError<Infallible>> {
        if !alpha.is_finite()
            || !(0.0..=1.0).contains(&alpha)
            || !self.lod_seams_applied
            || plan.tile() != self.tile
            || field.tile() != self.tile
            || field.cache_id() != self.cache_id
            || field.revision() != self.revision
            || field.spec() != self.spec
            || plan.segments().checked_add(1) != Some(self.spec.samples_per_edge())
        {
            return Err(BakeError::InvalidGeometry);
        }
        let mut plan_neighbors = [None; 4];
        for transition in plan.transitions() {
            let edge = planet_mesh_edge_index(transition.edge());
            if plan_neighbors[edge]
                .replace(transition.neighbor())
                .is_some()
            {
                return Err(BakeError::InvalidGeometry);
            }
        }
        if plan_neighbors != self.lod_seam_neighbors {
            return Err(BakeError::InvalidGeometry);
        }
        let side = self.spec.samples_per_edge();
        for transition in plan.transitions() {
            for morph in transition.morph_vertices() {
                for [x, y] in [morph.source, morph.target_a, morph.target_b] {
                    if x >= side || y >= side {
                        return Err(BakeError::InvalidGeometry);
                    }
                }
                let position = self.planet_morphed_position(field, plan, morph.source, alpha)?;
                if position.iter().any(|coordinate| !coordinate.is_finite()) {
                    return Err(BakeError::InvalidGeometry);
                }
            }
        }
        for transition in plan.transitions() {
            for morph in transition.morph_vertices() {
                let [x, y] = morph.source;
                for normal_y in y.saturating_sub(1)..=(y + 1).min(side - 1) {
                    for normal_x in x.saturating_sub(1)..=(x + 1).min(side - 1) {
                        let _ =
                            self.planet_morphed_normal(field, plan, [normal_x, normal_y], alpha)?;
                    }
                }
            }
        }

        for transition in plan.transitions() {
            for morph in transition.morph_vertices() {
                let position = self.planet_morphed_position(field, plan, morph.source, alpha)?;
                let index = (morph.source[1] * side + morph.source[0]) as usize;
                self.positions[index] = position;
            }
        }
        for transition in plan.transitions() {
            for morph in transition.morph_vertices() {
                let [x, y] = morph.source;
                for normal_y in y.saturating_sub(1)..=(y + 1).min(side - 1) {
                    for normal_x in x.saturating_sub(1)..=(x + 1).min(side - 1) {
                        let normal = self.planet_normal_from_positions([normal_x, normal_y])?;
                        let index = (normal_y * side + normal_x) as usize;
                        self.normals[index] = normal;
                    }
                }
            }
        }
        Ok(())
    }

    fn planet_morphed_position(
        &self,
        field: &HeightfieldTile,
        plan: &crate::HeightfieldLodSeamPlan,
        grid: [u32; 2],
        alpha: f32,
    ) -> Result<[f32; 3], BakeError<Infallible>> {
        let side = self.spec.samples_per_edge();
        let index = (grid[1] * side + grid[0]) as usize;
        if self.positions.get(index).is_none() {
            return Err(BakeError::InvalidGeometry);
        }
        for transition in plan.transitions() {
            let parameter = match transition.edge() {
                TileEdge::West if grid[0] == 0 => Some(grid[1]),
                TileEdge::East if grid[0] + 1 == side => Some(grid[1]),
                TileEdge::South if grid[1] == 0 => Some(grid[0]),
                TileEdge::North if grid[1] + 1 == side => Some(grid[0]),
                _ => None,
            };
            let Some(parameter) = parameter.filter(|parameter| parameter % 2 == 1) else {
                continue;
            };
            let morph = transition
                .morph_vertices()
                .get((parameter / 2) as usize)
                .ok_or(BakeError::InvalidGeometry)?;
            if morph.source != grid {
                return Err(BakeError::InvalidGeometry);
            }
            let a_index = (morph.target_a[1] * side + morph.target_a[0]) as usize;
            let b_index = (morph.target_b[1] * side + morph.target_b[0]) as usize;
            let a = self
                .positions
                .get(a_index)
                .ok_or(BakeError::InvalidGeometry)?;
            let b = self
                .positions
                .get(b_index)
                .ok_or(BakeError::InvalidGeometry)?;
            let source = project_position(field, self.config, grid[0], grid[1])?.0;
            let source = source.map(f64::from);
            let a = a.map(f64::from);
            let b = b.map(f64::from);
            let alpha = f64::from(alpha);
            return Ok(core::array::from_fn(|axis| {
                (source[axis] + ((a[axis] + b[axis]) * 0.5 - source[axis]) * alpha) as f32
            }));
        }
        Ok(self.positions[index])
    }

    fn planet_morphed_normal(
        &self,
        field: &HeightfieldTile,
        plan: &crate::HeightfieldLodSeamPlan,
        grid: [u32; 2],
        alpha: f32,
    ) -> Result<[f32; 3], BakeError<Infallible>> {
        let side = self.spec.samples_per_edge();
        let [x, y] = grid;
        let left = self.planet_morphed_position(field, plan, [x.saturating_sub(1), y], alpha)?;
        let right = self.planet_morphed_position(field, plan, [(x + 1).min(side - 1), y], alpha)?;
        let down = self.planet_morphed_position(field, plan, [x, y.saturating_sub(1)], alpha)?;
        let up = self.planet_morphed_position(field, plan, [x, (y + 1).min(side - 1)], alpha)?;
        let mut normal = cross(
            subtract(right.map(f64::from), left.map(f64::from)),
            subtract(up.map(f64::from), down.map(f64::from)),
        );
        let length = libm::sqrt(normal.iter().map(|value| value * value).sum());
        if !length.is_finite() || length <= 0.0 {
            return Err(BakeError::InvalidGeometry);
        }
        normal.iter_mut().for_each(|value| *value /= length);
        let position = self
            .planet_morphed_position(field, plan, grid, alpha)?
            .map(f64::from);
        let origin = self.config.origin_mm.map(|value| value as f64 * 0.001);
        let outward = (0..3)
            .map(|axis| normal[axis] * (position[axis] + origin[axis]))
            .sum::<f64>();
        if !outward.is_finite() || outward == 0.0 {
            return Err(BakeError::InvalidGeometry);
        }
        if outward < 0.0 {
            normal.iter_mut().for_each(|value| *value = -*value);
        }
        let result = normal.map(|value| value as f32);
        if result.iter().any(|value| !value.is_finite()) {
            return Err(BakeError::InvalidGeometry);
        }
        Ok(result)
    }

    fn planet_normal_from_positions(
        &self,
        grid: [u32; 2],
    ) -> Result<[f32; 3], BakeError<Infallible>> {
        let side = self.spec.samples_per_edge();
        let [x, y] = grid;
        let position = |x: u32, y: u32| self.positions[(y * side + x) as usize].map(f64::from);
        let mut normal = cross(
            subtract(
                position((x + 1).min(side - 1), y),
                position(x.saturating_sub(1), y),
            ),
            subtract(
                position(x, (y + 1).min(side - 1)),
                position(x, y.saturating_sub(1)),
            ),
        );
        let length = libm::sqrt(normal.iter().map(|value| value * value).sum());
        if !length.is_finite() || length <= 0.0 {
            return Err(BakeError::InvalidGeometry);
        }
        normal.iter_mut().for_each(|value| *value /= length);
        let position = position(x, y);
        let origin = self.config.origin_mm.map(|value| value as f64 * 0.001);
        let outward = (0..3)
            .map(|axis| normal[axis] * (position[axis] + origin[axis]))
            .sum::<f64>();
        if !outward.is_finite() || outward == 0.0 {
            return Err(BakeError::InvalidGeometry);
        }
        if outward < 0.0 {
            normal.iter_mut().for_each(|value| *value = -*value);
        }
        let result = normal.map(|value| value as f32);
        if result.iter().any(|value| !value.is_finite()) {
            return Err(BakeError::InvalidGeometry);
        }
        Ok(result)
    }

    /// Restores regular cube-sphere grid triangles before applying another plan.
    pub fn clear_lod_seam_plan(&mut self, max_bytes: usize) -> Result<(), BakeError<Infallible>> {
        if !self.lod_seams_applied {
            return Ok(());
        }
        let side = self.spec.samples_per_edge();
        let segments = side - 1;
        let count = usize::try_from(segments)
            .ok()
            .and_then(|n| n.checked_mul(n))
            .and_then(|n| n.checked_mul(6))
            .ok_or(BakeError::SizeOverflow)?;
        let bytes = self
            .positions
            .len()
            .checked_mul(32)
            .and_then(|n| count.checked_mul(4).and_then(|i| n.checked_add(i)))
            .ok_or(BakeError::SizeOverflow)?;
        if bytes > max_bytes {
            return Err(BakeError::BudgetExceeded);
        }
        let mut indices = Vec::new();
        indices
            .try_reserve_exact(count)
            .map_err(|_| BakeError::AllocationFailed)?;
        for y in 0..segments {
            for x in 0..segments {
                let a = y * side + x;
                let b = a + 1;
                let c = a + side;
                indices.extend_from_slice(&[a, b, c, b, c + 1, c]);
            }
        }
        self.indices = indices;
        self.lod_seams_applied = false;
        self.lod_seam_neighbors = [None; 4];
        Ok(())
    }
    /// Integer-origin placement contract.
    pub const fn config(&self) -> PlanetMeshConfig {
        self.config
    }
    /// Maximum per-coordinate arithmetic/encoding error bound in millimetres.
    pub const fn error_bound_mm(&self) -> f64 {
        self.error_bound_mm
    }

    /// Last consumed heightfield cache revision.
    pub const fn revision(&self) -> u64 {
        self.revision
    }

    /// Converts into a native Prism render mesh while preserving local positions.
    #[cfg(feature = "mesh")]
    pub fn into_mesh(self) -> Result<prism_mesh::Mesh, prism_mesh::MeshError> {
        use prism_mesh::{Indices, Mesh, MeshVertexAttribute, VertexAttributeValues};
        use prism_render_driver::VertexFormat;
        let mesh = Mesh::new(prism_render_driver::PrimitiveTopology::TriangleList)
            .with_attribute(
                MeshVertexAttribute::new(
                    prism_mesh::POSITION_ATTRIBUTE,
                    VertexFormat::Float32x3,
                    0,
                ),
                VertexAttributeValues::Float32x3(self.positions),
            )?
            .with_attribute(
                MeshVertexAttribute::new("Vertex_Normal", VertexFormat::Float32x3, 1),
                VertexAttributeValues::Float32x3(self.normals),
            )?
            .with_attribute(
                MeshVertexAttribute::new("Vertex_Uv0", VertexFormat::Float32x2, 2),
                VertexAttributeValues::Float32x2(self.uvs),
            )?
            .indexed(Indices::Uint32(self.indices));
        mesh.validate()?;
        Ok(mesh)
    }
}

fn reserve<T>(count: usize) -> Result<Vec<T>, BakeError<Infallible>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| BakeError::AllocationFailed)?;
    Ok(values)
}

fn edge_index(edge: TileEdge) -> usize {
    match edge {
        TileEdge::West => 0,
        TileEdge::East => 1,
        TileEdge::South => 2,
        TileEdge::North => 3,
    }
}

fn edge_coordinate(edge: TileEdge, side: u32, index: u32, inward: bool) -> [u32; 2] {
    let offset = if inward { 1 } else { 0 };
    match edge {
        TileEdge::West => [offset, index],
        TileEdge::East => [side - 1 - offset, index],
        TileEdge::South => [index, offset],
        TileEdge::North => [index, side - 1 - offset],
    }
}

fn neighbor_height(
    field: &HeightfieldTile,
    edge: TileEdge,
    index: u32,
    inward: bool,
    source: PlanetHeightfieldNeighbor<'_>,
) -> Result<(TileKey, TileEdge, u32, i128), BakeError<Infallible>> {
    let side = field.spec().samples_per_edge();
    let (neighbor, neighbor_edge, neighbor_index) = field
        .tile()
        .neighbor_edge_index(edge, side, index)
        .ok_or(BakeError::InvalidGeometry)?;
    let height = match source {
        PlanetHeightfieldNeighbor::Tile(tile) => {
            if tile.tile() != neighbor {
                return Err(BakeError::InvalidGeometry);
            }
            tile.height_mm(edge_coordinate(neighbor_edge, side, neighbor_index, inward))
                .ok_or(BakeError::InvalidGeometry)?
        }
        PlanetHeightfieldNeighbor::Boundary(snapshot) => {
            if snapshot.neighbor != neighbor || snapshot.neighbor_edge != neighbor_edge {
                return Err(BakeError::InvalidGeometry);
            }
            let values = if inward {
                snapshot.inner_mm
            } else {
                snapshot.edge_mm
            };
            *values
                .get(neighbor_index as usize)
                .ok_or(BakeError::InvalidGeometry)?
        }
    };
    Ok((neighbor, neighbor_edge, neighbor_index, height))
}

fn validate_neighborhood(
    field: &HeightfieldTile,
    neighborhood: PlanetHeightfieldNormalNeighborhood<'_>,
) -> Result<(), BakeError<Infallible>> {
    let side = field.spec().samples_per_edge();
    for edge in TileEdge::ALL {
        let Some(source) = neighborhood.get(edge) else {
            continue;
        };
        match source {
            PlanetHeightfieldNeighbor::Tile(tile) => {
                if tile.spec().samples_per_edge() != side {
                    return Err(BakeError::InvalidGeometry);
                }
            }
            PlanetHeightfieldNeighbor::Boundary(snapshot) => {
                if snapshot.edge_mm.len() != side as usize
                    || snapshot.inner_mm.len() != side as usize
                {
                    return Err(BakeError::InvalidGeometry);
                }
            }
        }
        for index in 0..side {
            let center = field
                .height_mm(edge_coordinate(edge, side, index, false))
                .ok_or(BakeError::InvalidGeometry)?;
            let (neighbor, _, _, boundary) = neighbor_height(field, edge, index, false, source)?;
            let expected = field
                .tile()
                .neighbor_edge_index(edge, side, index)
                .ok_or(BakeError::InvalidGeometry)?
                .0;
            if neighbor != expected || boundary != center {
                return Err(BakeError::InvalidGeometry);
            }
            neighbor_height(field, edge, index, true, source)?;
        }
    }
    Ok(())
}

fn derivative_point(
    field: &HeightfieldTile,
    radius: i64,
    neighborhood: PlanetHeightfieldNormalNeighborhood<'_>,
    edge: TileEdge,
    x: u32,
    y: u32,
) -> Result<Option<[f64; 3]>, BakeError<Infallible>> {
    let index = match edge {
        TileEdge::West if x == 0 => y,
        TileEdge::East if x + 1 == field.spec().samples_per_edge() => y,
        TileEdge::South if y == 0 => x,
        TileEdge::North if y + 1 == field.spec().samples_per_edge() => x,
        _ => return Ok(None),
    };
    let Some(source) = neighborhood.get(edge) else {
        return Ok(None);
    };
    let (neighbor, neighbor_edge, neighbor_index, height) =
        neighbor_height(field, edge, index, true, source)?;
    let side = field.spec().samples_per_edge();
    let coordinate = edge_coordinate(neighbor_edge, side, neighbor_index, true);
    Ok(Some(point_at(
        neighbor,
        side,
        radius,
        coordinate[0],
        coordinate[1],
        height,
    )?))
}

fn point(
    field: &HeightfieldTile,
    radius: i64,
    x: u32,
    y: u32,
) -> Result<[f64; 3], BakeError<Infallible>> {
    let height = field.height_mm([x, y]).ok_or(BakeError::InvalidGeometry)?;
    point_at(
        field.tile(),
        field.spec().samples_per_edge(),
        radius,
        x,
        y,
        height,
    )
}

fn point_at(
    tile: TileKey,
    side: u32,
    radius: i64,
    x: u32,
    y: u32,
    height: i128,
) -> Result<[f64; 3], BakeError<Infallible>> {
    let [u, v] = face_coordinate(tile, side, x, y);
    let cube = match tile.face() {
        CubeFace::PositiveX => [1.0, v, -u],
        CubeFace::NegativeX => [-1.0, v, u],
        CubeFace::PositiveY => [u, 1.0, -v],
        CubeFace::NegativeY => [u, -1.0, v],
        CubeFace::PositiveZ => [u, v, 1.0],
        CubeFace::NegativeZ => [-u, v, -1.0],
    };
    let radial_mm = i128::from(radius) + height;
    if radial_mm <= 0 {
        return Err(BakeError::InvalidGeometry);
    }
    let length = libm::sqrt(1.0 + (u * u + v * v));
    Ok(cube.map(|v| v / length * radial_mm as f64))
}

struct ProjectedPlanetVertex {
    position: [f32; 3],
    normal: [f32; 3],
    error_bound_mm: f64,
}

fn project_vertex(
    field: &HeightfieldTile,
    config: PlanetMeshConfig,
    neighborhood: PlanetHeightfieldNormalNeighborhood<'_>,
    x: u32,
    y: u32,
) -> Result<ProjectedPlanetVertex, BakeError<Infallible>> {
    let (position, error_bound_mm) = project_position(field, config, x, y)?;
    let left = derivative_point(field, config.radius_mm, neighborhood, TileEdge::West, x, y)?
        .unwrap_or(point(field, config.radius_mm, x.saturating_sub(1), y)?);
    let right = derivative_point(field, config.radius_mm, neighborhood, TileEdge::East, x, y)?
        .unwrap_or(point(
            field,
            config.radius_mm,
            (x + 1).min(field.spec().samples_per_edge() - 1),
            y,
        )?);
    let down = derivative_point(field, config.radius_mm, neighborhood, TileEdge::South, x, y)?
        .unwrap_or(point(field, config.radius_mm, x, y.saturating_sub(1))?);
    let up = derivative_point(field, config.radius_mm, neighborhood, TileEdge::North, x, y)?
        .unwrap_or(point(
            field,
            config.radius_mm,
            x,
            (y + 1).min(field.spec().samples_per_edge() - 1),
        )?);
    let normal = cross(subtract(right, left), subtract(up, down));
    let length = libm::sqrt(normal.iter().map(|v| v * v).sum());
    if !length.is_finite() || length <= 0.0 {
        return Err(BakeError::InvalidGeometry);
    }
    Ok(ProjectedPlanetVertex {
        position,
        normal: normal.map(|v| (v / length) as f32),
        error_bound_mm,
    })
}

fn project_position(
    field: &HeightfieldTile,
    config: PlanetMeshConfig,
    x: u32,
    y: u32,
) -> Result<([f32; 3], f64), BakeError<Infallible>> {
    let absolute = point(field, config.radius_mm, x, y)?;
    let mut position = [0.0; 3];
    let radial_mm = (i128::from(config.radius_mm)
        + field.height_mm([x, y]).ok_or(BakeError::InvalidGeometry)?) as f64;
    let mut error_bound_mm: f64 = 0.0;
    for axis in 0..3 {
        let local_mm = absolute[axis] - config.origin_mm[axis] as f64;
        position[axis] = (local_mm * 0.001) as f32;
        // A deliberately conservative operation-count bound covers rational
        // UV, normalization, radius/height and subtraction.
        let arithmetic = 128.0
            * f64::EPSILON
            * (absolute[axis].abs()
                + (config.origin_mm[axis] as f64).abs()
                + radial_mm.abs()
                + 1.0);
        let error = (f64::from(position[axis]) * 1000.0 - local_mm).abs() + arithmetic;
        if !position[axis].is_finite() || error > config.max_error_mm {
            return Err(BakeError::InvalidGeometry);
        }
        error_bound_mm = error_bound_mm.max(error);
    }
    Ok((position, error_bound_mm))
}

fn face_coordinate(tile: TileKey, side: u32, x: u32, y: u32) -> [f64; 2] {
    let tiles = f64::from(tile.side_len());
    let segments = f64::from(side - 1);
    [
        2.0 * (f64::from(tile.x()) + f64::from(x) / segments) / tiles - 1.0,
        2.0 * (f64::from(tile.y()) + f64::from(y) / segments) / tiles - 1.0,
    ]
}

fn subtract(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{HeightSample, HeightSampleValue, HeightSampler, HeightfieldSpec};
    struct Constant(i64);
    impl HeightSampler for Constant {
        type Error = Infallible;
        fn sample_height(&self, _: HeightSample) -> Result<HeightSampleValue, Self::Error> {
            Ok(HeightSampleValue::from_millimetres(self.0))
        }
    }

    struct AdjacentStep;
    impl HeightSampler for AdjacentStep {
        type Error = Infallible;

        fn sample_height(&self, sample: HeightSample) -> Result<HeightSampleValue, Self::Error> {
            let height = if sample.tile().x() == 0 || sample.u_q16() == 0 {
                0
            } else {
                100
            };
            Ok(HeightSampleValue::from_millimetres(height))
        }
    }

    struct Ramp;
    impl HeightSampler for Ramp {
        type Error = Infallible;

        fn sample_height(&self, sample: HeightSample) -> Result<HeightSampleValue, Self::Error> {
            let u = i64::from(sample.u_q16());
            let v = i64::from(sample.v_q16());
            Ok(HeightSampleValue::from_millimetres((u * 7 + v * 11) / 4096))
        }
    }
    const LIMITS: BakeLimits = BakeLimits {
        max_samples: 1024,
        max_bytes: 100_000,
    };
    fn field(tile: TileKey, height: i64) -> HeightfieldTile {
        HeightfieldTile::bake(
            tile,
            HeightfieldSpec::new(9, 1).unwrap(),
            &Constant(height),
            LIMITS,
        )
        .unwrap()
    }

    fn adjacent_field(tile: TileKey) -> HeightfieldTile {
        HeightfieldTile::bake(
            tile,
            HeightfieldSpec::new(9, 1).unwrap(),
            &AdjacentStep,
            LIMITS,
        )
        .unwrap()
    }

    #[test]
    fn all_faces_have_outward_winding_and_unit_normals() {
        for face in CubeFace::ALL {
            let mesh = PlanetHeightfieldMesh::build(
                &field(TileKey::root(face), 0),
                PlanetMeshConfig::new(1_000_000, [0; 3], 1.0).unwrap(),
                LIMITS,
            )
            .unwrap();
            for (p, n) in mesh.positions().iter().zip(mesh.normals()) {
                assert!(p.iter().zip(n).map(|(a, b)| a * b).sum::<f32>() > 0.0);
                assert!((n.iter().map(|v| v * v).sum::<f32>() - 1.0).abs() < 1e-6);
            }
            for tri in mesh.indices().chunks_exact(3) {
                let [a, b, c] =
                    [tri[0], tri[1], tri[2]].map(|i| mesh.positions()[i as usize].map(f64::from));
                assert!(
                    cross(subtract(b, a), subtract(c, a))
                        .iter()
                        .zip(a)
                        .map(|(n, p)| n * p)
                        .sum::<f64>()
                        > 0.0
                );
            }
        }
    }

    #[test]
    fn planetary_mesh_consumes_and_resets_coarse_lod_seams() {
        use crate::{HeightfieldLodSeamPlan, LodSeamLimits};

        let tile = TileKey::new(CubeFace::PositiveZ, 3, 3, 2).unwrap();
        let coarse = tile.coarser_neighbor(TileEdge::East).unwrap();
        let plan = HeightfieldLodSeamPlan::build_strict_for_residency(
            tile,
            8,
            &[coarse],
            LodSeamLimits {
                max_segments: 8,
                max_vertices: 128,
                max_indices: 1024,
                max_bytes: 16_384,
            },
        )
        .unwrap();
        let config = PlanetMeshConfig::new(1_000_000, [0; 3], 1.0).unwrap();
        let mut mesh = PlanetHeightfieldMesh::build(&field(tile, 0), config, LIMITS).unwrap();
        let base = mesh.indices().to_vec();
        let original_len = base.len();
        assert_eq!(
            mesh.apply_lod_seam_plan(&plan, 1),
            Err(BakeError::BudgetExceeded)
        );
        assert_eq!(mesh.indices(), base);
        mesh.apply_lod_seam_plan(&plan, 16_384).unwrap();
        assert_eq!(mesh.indices().len(), original_len - 8 * 6 + 4 * 9);
        assert!(mesh
            .indices()
            .iter()
            .all(|&i| (i as usize) < mesh.positions().len()));
        for tri in mesh.indices().chunks_exact(3) {
            let [a, b, c] =
                [tri[0], tri[1], tri[2]].map(|i| mesh.positions()[i as usize].map(f64::from));
            assert!(
                cross(subtract(b, a), subtract(c, a))
                    .iter()
                    .zip(a)
                    .map(|(n, p)| n * p)
                    .sum::<f64>()
                    > 0.0
            );
        }
        mesh.clear_lod_seam_plan(16_384).unwrap();
        assert_eq!(mesh.indices(), base);
    }

    #[test]
    fn planetary_seams_preserve_winding_on_every_face_and_edge() {
        use crate::{HeightfieldLodSeamPlan, LodSeamCoverage, LodSeamLimits, LodSeamNeighbor};

        let config = PlanetMeshConfig::new(1_000_000, [0; 3], 1.0).unwrap();
        let seam_limits = LodSeamLimits {
            max_segments: 8,
            max_vertices: 128,
            max_indices: 1024,
            max_bytes: 16_384,
        };
        for face in CubeFace::ALL {
            let tile = TileKey::new(face, 3, 3, 3).unwrap();
            let field =
                HeightfieldTile::bake(tile, HeightfieldSpec::new(9, 1).unwrap(), &Ramp, LIMITS)
                    .unwrap();
            for edge in TileEdge::ALL {
                let coarse = tile.coarser_neighbor(edge).unwrap();
                let plan = HeightfieldLodSeamPlan::build_strict(
                    tile,
                    8,
                    &[LodSeamNeighbor::with_boundary(
                        edge,
                        coarse,
                        8,
                        LodSeamCoverage::full(8),
                    )],
                    seam_limits,
                )
                .unwrap();
                let mut mesh = PlanetHeightfieldMesh::build(&field, config, LIMITS).unwrap();
                for transition in plan.transitions() {
                    for tri in transition.indices().chunks_exact(3) {
                        let points = [tri[0], tri[1], tri[2]].map(|i| {
                            let g = transition.vertices()[i as usize].grid;
                            mesh.positions()[(g[1] * 9 + g[0]) as usize].map(f64::from)
                        });
                        let n = cross(
                            subtract(points[1], points[0]),
                            subtract(points[2], points[0]),
                        );
                        let radial = core::array::from_fn::<_, 3, _>(|axis| {
                            (points[0][axis] + points[1][axis] + points[2][axis]) / 3.0
                                + config.origin_mm()[axis] as f64 * 0.001
                        });
                        let d = n.iter().zip(radial).map(|(a, b)| a * b).sum::<f64>();
                        assert!(d != 0.0, "degenerate: {face:?} {edge:?} {tri:?} {points:?}");
                    }
                }
                assert_eq!(plan.transitions().len(), 1, "{face:?} {edge:?}");
                mesh.apply_lod_seam_plan(&plan, 16_384)
                    .unwrap_or_else(|error| panic!("{face:?} {edge:?}: {error:?}"));
                for tri in mesh.indices().chunks_exact(3) {
                    let [a, b, c] = [tri[0], tri[1], tri[2]]
                        .map(|i| mesh.positions()[i as usize].map(f64::from));
                    assert!(
                        cross(subtract(b, a), subtract(c, a))
                            .iter()
                            .zip(a)
                            .map(|(n, p)| n * p)
                            .sum::<f64>()
                            > 0.0,
                        "inward seam triangle on {face:?} {edge:?}"
                    );
                }
            }
            let east = tile.coarser_neighbor(TileEdge::East).unwrap();
            let west = tile.coarser_neighbor(TileEdge::West).unwrap();
            let plan = HeightfieldLodSeamPlan::build_strict(
                tile,
                8,
                &[
                    LodSeamNeighbor::with_boundary(
                        TileEdge::West,
                        west,
                        8,
                        LodSeamCoverage::full(8),
                    ),
                    LodSeamNeighbor::with_boundary(
                        TileEdge::East,
                        east,
                        8,
                        LodSeamCoverage::full(8),
                    ),
                ],
                seam_limits,
            )
            .unwrap();
            let mut mesh = PlanetHeightfieldMesh::build(&field, config, LIMITS).unwrap();
            mesh.apply_lod_seam_plan(&plan, 16_384).unwrap();
            for tri in mesh.indices().chunks_exact(3) {
                let [a, b, c] =
                    [tri[0], tri[1], tri[2]].map(|i| mesh.positions()[i as usize].map(f64::from));
                assert!(
                    cross(subtract(b, a), subtract(c, a))
                        .iter()
                        .zip(a)
                        .map(|(n, p)| n * p)
                        .sum::<f64>()
                        > 0.0
                );
            }
        }
    }

    #[test]
    fn planetary_corner_fans_are_manifold_and_outward_on_all_faces() {
        use crate::{HeightfieldLodSeamPlan, LodSeamCoverage, LodSeamLimits, LodSeamNeighbor};

        let config = PlanetMeshConfig::new(1_000_000, [0; 3], 1.0).unwrap();
        let seam_limits = LodSeamLimits {
            max_segments: 8,
            max_vertices: 128,
            max_indices: 1024,
            max_bytes: 16_384,
        };
        let corners = [
            (TileEdge::West, TileEdge::South),
            (TileEdge::East, TileEdge::South),
            (TileEdge::West, TileEdge::North),
            (TileEdge::East, TileEdge::North),
        ];
        for face in CubeFace::ALL {
            let tile = TileKey::new(face, 3, 3, 3).unwrap();
            let field =
                HeightfieldTile::bake(tile, HeightfieldSpec::new(9, 1).unwrap(), &Ramp, LIMITS)
                    .unwrap();
            for (first_edge, second_edge) in corners {
                let neighbors = [first_edge, second_edge].map(|edge| {
                    LodSeamNeighbor::with_boundary(
                        edge,
                        tile.coarser_neighbor(edge).unwrap(),
                        8,
                        LodSeamCoverage::full(8),
                    )
                });
                let plan =
                    HeightfieldLodSeamPlan::build_strict(tile, 8, &neighbors, seam_limits).unwrap();
                let mut mesh = PlanetHeightfieldMesh::build(&field, config, LIMITS).unwrap();
                let base = mesh.clone();
                assert_eq!(
                    mesh.apply_lod_seam_plan(&plan, 1),
                    Err(BakeError::BudgetExceeded)
                );
                assert_eq!(mesh, base, "budget mutation on {face:?} {corners:?}");
                mesh.apply_lod_seam_plan(&plan, 16_384)
                    .unwrap_or_else(|error| {
                        panic!("{face:?} {first_edge:?}+{second_edge:?}: {error:?}")
                    });

                let mut edge_counts = Vec::new();
                let mut doubled_grid_area = 0i64;
                let mut used_grid_vertices = [false; 81];
                for &index in mesh.indices() {
                    used_grid_vertices[index as usize] = true;
                }
                for triangle in mesh.indices().chunks_exact(3) {
                    let grid = [triangle[0], triangle[1], triangle[2]].map(|index| {
                        let index = i64::from(index);
                        [index % 9, index / 9]
                    });
                    let ab = [grid[1][0] - grid[0][0], grid[1][1] - grid[0][1]];
                    let ac = [grid[2][0] - grid[0][0], grid[2][1] - grid[0][1]];
                    let area = ab[0] * ac[1] - ab[1] * ac[0];
                    assert_ne!(area, 0, "degenerate grid triangle on {face:?}");
                    doubled_grid_area += area.abs();

                    let [a, b, c] = [triangle[0], triangle[1], triangle[2]]
                        .map(|index| mesh.positions()[index as usize].map(f64::from));
                    assert!(
                        cross(subtract(b, a), subtract(c, a))
                            .iter()
                            .zip(a)
                            .map(|(normal, position)| normal * position)
                            .sum::<f64>()
                            > 0.0,
                        "inward corner triangle on {face:?} {first_edge:?}+{second_edge:?}"
                    );
                    for (start, end) in [
                        (triangle[0], triangle[1]),
                        (triangle[1], triangle[2]),
                        (triangle[2], triangle[0]),
                    ] {
                        let start_grid = [i64::from(start % 9), i64::from(start / 9)];
                        let end_grid = [i64::from(end % 9), i64::from(end / 9)];
                        let edge = [end_grid[0] - start_grid[0], end_grid[1] - start_grid[1]];
                        let edge_len_squared = edge[0] * edge[0] + edge[1] * edge[1];
                        for candidate in 0..81i64 {
                            if !used_grid_vertices[candidate as usize] {
                                continue;
                            }
                            let point = [candidate % 9, candidate / 9];
                            let relative = [point[0] - start_grid[0], point[1] - start_grid[1]];
                            let cross = edge[0] * relative[1] - edge[1] * relative[0];
                            let along = edge[0] * relative[0] + edge[1] * relative[1];
                            assert!(
                                cross != 0 || along <= 0 || along >= edge_len_squared,
                                "T-junction at {point:?} on edge {start_grid:?}->{end_grid:?}, triangle {grid:?}, {face:?} {first_edge:?}+{second_edge:?}"
                            );
                        }
                        edge_counts.push((start.min(end), start.max(end)));
                    }
                }
                assert_eq!(
                    doubled_grid_area, 2 * 8 * 8,
                    "corner fan overlaps or leaves a grid-space hole on {face:?} {first_edge:?}+{second_edge:?}"
                );
                edge_counts.sort_unstable();
                let mut boundary_edges = 0;
                for edge_group in edge_counts.chunk_by(|left, right| left == right) {
                    assert!(
                        edge_group.len() <= 2,
                        "non-manifold edge on {face:?} {first_edge:?}+{second_edge:?}"
                    );
                    boundary_edges += usize::from(edge_group.len() == 1);
                }
                assert_eq!(
                    boundary_edges, 24,
                    "unexpected open boundary on {face:?} {first_edge:?}+{second_edge:?}"
                );
            }
        }
    }

    #[test]
    fn planetary_lod_morph_tracks_chord_midpoints_on_every_face_and_edge() {
        use crate::{HeightfieldLodSeamPlan, LodSeamCoverage, LodSeamLimits, LodSeamNeighbor};

        let config = PlanetMeshConfig::new(1_000_000, [0, 0, 1_000_000], 1.0).unwrap();
        let seam_limits = LodSeamLimits {
            max_segments: 8,
            max_vertices: 128,
            max_indices: 1024,
            max_bytes: 16_384,
        };
        for face in CubeFace::ALL {
            let tile = TileKey::new(face, 3, 3, 3).unwrap();
            let field =
                HeightfieldTile::bake(tile, HeightfieldSpec::new(9, 1).unwrap(), &Ramp, LIMITS)
                    .unwrap();
            for edge in TileEdge::ALL {
                let neighbor = tile.coarser_neighbor(edge).unwrap();
                let plan = HeightfieldLodSeamPlan::build_strict(
                    tile,
                    8,
                    &[LodSeamNeighbor::with_boundary(
                        edge,
                        neighbor,
                        8,
                        LodSeamCoverage::full(8),
                    )],
                    seam_limits,
                )
                .unwrap();
                let mut mesh = PlanetHeightfieldMesh::build(&field, config, LIMITS).unwrap();
                mesh.apply_lod_seam_plan(&plan, 16_384).unwrap();
                let fine_positions = mesh.positions().to_vec();
                mesh.apply_lod_morph(&field, &plan, 0.5)
                    .unwrap_or_else(|error| panic!("{face:?} {edge:?}: {error:?}"));
                for morph in plan.transitions()[0].morph_vertices() {
                    let source = (morph.source[1] * 9 + morph.source[0]) as usize;
                    let a = (morph.target_a[1] * 9 + morph.target_a[0]) as usize;
                    let b = (morph.target_b[1] * 9 + morph.target_b[0]) as usize;
                    let expected = core::array::from_fn::<_, 3, _>(|axis| {
                        (f64::from(fine_positions[source][axis])
                            + ((f64::from(fine_positions[a][axis])
                                + f64::from(fine_positions[b][axis]))
                                * 0.5
                                - f64::from(fine_positions[source][axis]))
                                * 0.5) as f32
                    });
                    assert_eq!(mesh.positions()[source], expected, "{face:?} {edge:?}");
                }
                mesh.apply_lod_morph(&field, &plan, 1.0)
                    .unwrap_or_else(|error| panic!("{face:?} {edge:?}: {error:?}"));
                for morph in plan.transitions()[0].morph_vertices() {
                    let source = (morph.source[1] * 9 + morph.source[0]) as usize;
                    let a = (morph.target_a[1] * 9 + morph.target_a[0]) as usize;
                    let b = (morph.target_b[1] * 9 + morph.target_b[0]) as usize;
                    let expected = core::array::from_fn::<_, 3, _>(|axis| {
                        (fine_positions[a][axis] + fine_positions[b][axis]) * 0.5
                    });
                    assert_eq!(mesh.positions()[source], expected, "{face:?} {edge:?}");
                }
                for (position, normal) in mesh.positions().iter().zip(mesh.normals()) {
                    assert!(normal.iter().all(|component| component.is_finite()));
                    assert!(
                        (normal.iter().map(|value| value * value).sum::<f32>() - 1.0).abs() < 1e-5
                    );
                    let absolute = core::array::from_fn::<_, 3, _>(|axis| {
                        position[axis] + config.origin_mm()[axis] as f32 * 0.001
                    });
                    assert!(
                        absolute.iter().zip(normal).map(|(p, n)| p * n).sum::<f32>() > 0.0,
                        "inward morph normal on {face:?} {edge:?}"
                    );
                }
                mesh.apply_lod_morph(&field, &plan, 0.0).unwrap();
                assert_eq!(mesh.positions(), fine_positions);
                let before_invalid = mesh.clone();
                assert_eq!(
                    mesh.apply_lod_morph(&field, &plan, f32::NAN),
                    Err(BakeError::InvalidGeometry)
                );
                assert_eq!(mesh, before_invalid);
            }
        }
    }

    #[test]
    fn planetary_finer_tiles_close_to_coarse_edges_on_all_faces() {
        use crate::{HeightfieldLodSeamPlan, LodSeamLimits};

        let config = PlanetMeshConfig::new(1_000_000, [0, 0, 1_000_000], 100.0).unwrap();
        let seam_limits = LodSeamLimits {
            max_segments: 8,
            max_vertices: 128,
            max_indices: 1024,
            max_bytes: 16_384,
        };
        for face in CubeFace::ALL {
            for edge in TileEdge::ALL {
                let coarse_tile = TileKey::new(face, 2, 1, 1).unwrap();
                let coarse_neighbor = coarse_tile.neighbor(edge);
                let fine_tiles: Vec<_> = coarse_tile
                    .children()
                    .unwrap()
                    .into_iter()
                    .filter(|child| {
                        let boundary_child = match edge {
                            TileEdge::West => child.x() % 2 == 0,
                            TileEdge::East => child.x() % 2 == 1,
                            TileEdge::South => child.y() % 2 == 0,
                            TileEdge::North => child.y() % 2 == 1,
                        };
                        boundary_child
                    })
                    .collect();
                assert_eq!(fine_tiles.len(), 2, "{face:?} {edge:?}");

                let mut resident = alloc::vec![coarse_neighbor, fine_tiles[0], fine_tiles[1]];
                resident.sort_unstable();
                let coarse_field = field(coarse_neighbor, 0);
                let coarse_mesh =
                    PlanetHeightfieldMesh::build(&coarse_field, config, LIMITS).unwrap();
                for fine_tile in fine_tiles {
                    let fine_field = field(fine_tile, 0);
                    let plan = HeightfieldLodSeamPlan::build_strict_for_residency(
                        fine_tile,
                        8,
                        &resident,
                        seam_limits,
                    )
                    .unwrap();
                    let transition = plan
                        .transitions()
                        .iter()
                        .find(|transition| transition.neighbor() == coarse_neighbor)
                        .unwrap_or_else(|| panic!("missing coarse transition: {face:?} {edge:?}"));
                    let mut fine_mesh =
                        PlanetHeightfieldMesh::build(&fine_field, config, LIMITS).unwrap();
                    fine_mesh.apply_lod_seam_plan(&plan, 16_384).unwrap();
                    fine_mesh.apply_lod_morph(&fine_field, &plan, 1.0).unwrap();

                    let same_level_neighbor = fine_tile.neighbor(transition.edge());
                    let neighbor_half = match transition.neighbor_edge() {
                        TileEdge::West | TileEdge::East => same_level_neighbor.y() % 2,
                        TileEdge::South | TileEdge::North => same_level_neighbor.x() % 2,
                    };
                    for fine_parameter in 0..=8_u32 {
                        let local_half_parameter = f64::from(fine_parameter) * 0.5;
                        let coarse_parameter = if transition.reversed() {
                            f64::from(neighbor_half * 4) + 4.0 - local_half_parameter
                        } else {
                            f64::from(neighbor_half * 4) + local_half_parameter
                        };
                        let lower = coarse_parameter.floor() as u32;
                        let upper = coarse_parameter.ceil() as u32;
                        let fraction = coarse_parameter - f64::from(lower);
                        let coarse_grid = |parameter| match transition.neighbor_edge() {
                            TileEdge::West => [0, parameter],
                            TileEdge::East => [8, parameter],
                            TileEdge::South => [parameter, 0],
                            TileEdge::North => [parameter, 8],
                        };
                        let coarse_a = coarse_grid(lower);
                        let coarse_b = coarse_grid(upper);
                        let coarse_position =
                            |[x, y]: [u32; 2]| coarse_mesh.positions()[(y * 9 + x) as usize];
                        let expected = core::array::from_fn::<_, 3, _>(|axis| {
                            (f64::from(coarse_position(coarse_a)[axis])
                                + (f64::from(coarse_position(coarse_b)[axis])
                                    - f64::from(coarse_position(coarse_a)[axis]))
                                    * fraction) as f32
                        });
                        let fine_grid = match transition.edge() {
                            TileEdge::West => [0, fine_parameter],
                            TileEdge::East => [8, fine_parameter],
                            TileEdge::South => [fine_parameter, 0],
                            TileEdge::North => [fine_parameter, 8],
                        };
                        let actual =
                            fine_mesh.positions()[(fine_grid[1] * 9 + fine_grid[0]) as usize];
                        assert_eq!(
                            actual, expected,
                            "{face:?} {edge:?} {fine_tile:?} t={fine_parameter}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn level_thirty_earth_patch_retains_submillimetre_detail() {
        let tile = TileKey::new(CubeFace::PositiveZ, 30, 1 << 29, 1 << 29).unwrap();
        let config = PlanetMeshConfig::new(6_371_000_000, [0, 0, 6_371_000_000], 0.001).unwrap();
        let mesh = PlanetHeightfieldMesh::build(&field(tile, 0), config, LIMITS).unwrap();
        assert!(mesh.positions()[8][0] > 0.01);
        for row in mesh.positions().chunks_exact(9) {
            assert!(row.windows(2).all(|pair| pair[1][0] > pair[0][0]));
        }
        assert!(mesh.error_bound_mm() < 0.001);
    }

    #[test]
    fn adjacent_tiles_and_cube_faces_share_exact_boundary_positions() {
        let config = PlanetMeshConfig::new(1_000_000, [0; 3], 1.0).unwrap();
        let a = PlanetHeightfieldMesh::build(
            &field(TileKey::new(CubeFace::PositiveZ, 1, 0, 0).unwrap(), 0),
            config,
            LIMITS,
        )
        .unwrap();
        let b = PlanetHeightfieldMesh::build(
            &field(TileKey::new(CubeFace::PositiveZ, 1, 1, 0).unwrap(), 0),
            config,
            LIMITS,
        )
        .unwrap();
        for row in 0..9 {
            assert_eq!(a.positions()[row * 9 + 8], b.positions()[row * 9]);
        }
        let z = PlanetHeightfieldMesh::build(
            &field(TileKey::root(CubeFace::PositiveZ), 0),
            config,
            LIMITS,
        )
        .unwrap();
        let x = PlanetHeightfieldMesh::build(
            &field(TileKey::root(CubeFace::PositiveX), 0),
            config,
            LIMITS,
        )
        .unwrap();
        for row in 0..9 {
            assert_eq!(z.positions()[row * 9 + 8], x.positions()[row * 9]);
        }
    }

    #[test]
    fn neighborhood_uses_neighbor_inner_row_for_continuous_boundary_normal() {
        let center_key = TileKey::new(CubeFace::PositiveZ, 1, 0, 0).unwrap();
        let neighbor_key = center_key.neighbor(TileEdge::East);
        let center = adjacent_field(center_key);
        let neighbor = adjacent_field(neighbor_key);
        let flat_neighbor = field(neighbor_key, 0);
        let config = PlanetMeshConfig::new(1_000_000, [0; 3], 1.0).unwrap();
        let legacy = PlanetHeightfieldMesh::build(&center, config, LIMITS).unwrap();
        let smooth = PlanetHeightfieldMesh::build_with_neighborhood(
            &center,
            config,
            LIMITS,
            PlanetHeightfieldNormalNeighborhood::new(
                None,
                Some(PlanetHeightfieldNeighbor::Tile(&neighbor)),
                None,
                None,
            ),
        )
        .unwrap();
        let smooth_flat = PlanetHeightfieldMesh::build_with_neighborhood(
            &center,
            config,
            LIMITS,
            PlanetHeightfieldNormalNeighborhood::new(
                None,
                Some(PlanetHeightfieldNeighbor::Tile(&flat_neighbor)),
                None,
                None,
            ),
        )
        .unwrap();
        let index = 4 * 9 + 8;
        assert_ne!(legacy.normals()[index], smooth.normals()[index]);
        assert_ne!(smooth_flat.normals()[index], smooth.normals()[index]);
    }

    #[test]
    fn incremental_projection_matches_full_rebuild_and_only_expands_normal_ring() {
        let tile = TileKey::new(CubeFace::PositiveZ, 1, 0, 0).unwrap();
        let mut field = field(tile, 0);
        let config = PlanetMeshConfig::new(1_000_000, [0; 3], 1.0).unwrap();
        let mut incremental = PlanetHeightfieldMesh::build(&field, config, LIMITS).unwrap();
        let change = field
            .rebake(
                crate::DirtyRegion::new(tile, 32_768, 32_768, 32_768, 32_768).unwrap(),
                &Constant(100),
                LIMITS,
            )
            .unwrap();
        let full = PlanetHeightfieldMesh::build(&field, config, LIMITS).unwrap();

        let updated = incremental
            .update(&field, change, PlanetHeightfieldNormalNeighborhood::empty())
            .unwrap();
        assert_eq!(updated.min(), [3, 3]);
        assert_eq!(updated.max(), [6, 6]);
        assert_eq!(incremental.positions(), full.positions());
        assert_eq!(incremental.normals(), full.normals());
        assert_eq!(incremental.uvs(), full.uvs());
        assert_eq!(incremental.indices(), full.indices());
        assert_eq!(incremental.revision(), field.revision());
    }

    #[test]
    fn incremental_projection_rejects_active_lod_seam_without_mutation() {
        use crate::{HeightfieldLodSeamPlan, LodSeamCoverage, LodSeamLimits, LodSeamNeighbor};

        let tile = TileKey::new(CubeFace::PositiveZ, 3, 3, 3).unwrap();
        let mut field =
            HeightfieldTile::bake(tile, HeightfieldSpec::new(9, 1).unwrap(), &Ramp, LIMITS)
                .unwrap();
        let config = PlanetMeshConfig::new(1_000_000, [0, 0, 1_000_000], 1.0).unwrap();
        let mut mesh = PlanetHeightfieldMesh::build(&field, config, LIMITS).unwrap();
        let edge = TileEdge::East;
        let plan = HeightfieldLodSeamPlan::build_strict(
            tile,
            8,
            &[LodSeamNeighbor::with_boundary(
                edge,
                tile.coarser_neighbor(edge).unwrap(),
                8,
                LodSeamCoverage::full(8),
            )],
            LodSeamLimits {
                max_segments: 8,
                max_vertices: 128,
                max_indices: 1024,
                max_bytes: 16_384,
            },
        )
        .unwrap();
        mesh.apply_lod_seam_plan(&plan, 16_384).unwrap();
        mesh.apply_lod_morph(&field, &plan, 0.5).unwrap();
        let before = mesh.clone();
        let change = field
            .rebake(
                crate::DirtyRegion::new(tile, 32_768, 32_768, 32_768, 32_768).unwrap(),
                &Ramp,
                LIMITS,
            )
            .unwrap();

        assert_eq!(
            mesh.update(&field, change, PlanetHeightfieldNormalNeighborhood::empty()),
            Err(BakeError::InvalidGeometry)
        );
        assert_eq!(mesh, before);
    }

    #[test]
    fn incremental_projection_rejects_stale_change_without_mutation() {
        let tile = TileKey::root(CubeFace::PositiveZ);
        let mut field = field(tile, 0);
        let config = PlanetMeshConfig::new(1_000_000, [0; 3], 1.0).unwrap();
        let mut mesh = PlanetHeightfieldMesh::build(&field, config, LIMITS).unwrap();
        let old_positions = mesh.positions().to_vec();
        let first = field
            .rebake(
                crate::DirtyRegion::new(tile, 1, 1, 1, 1).unwrap(),
                &Constant(10),
                LIMITS,
            )
            .unwrap();
        let _second = field
            .rebake(
                crate::DirtyRegion::new(tile, 2, 2, 2, 2).unwrap(),
                &Constant(20),
                LIMITS,
            )
            .unwrap();

        assert_eq!(
            mesh.update(&field, first, PlanetHeightfieldNormalNeighborhood::empty(),),
            Err(BakeError::StaleChange)
        );
        assert_eq!(mesh.revision(), 0);
        assert_eq!(mesh.positions(), old_positions.as_slice());
    }

    #[test]
    fn boundary_neighbor_mismatch_fails_before_incremental_projection_commit() {
        let tile = TileKey::new(CubeFace::PositiveZ, 1, 0, 0).unwrap();
        let neighbor_key = tile.neighbor(TileEdge::West);
        let mut center = field(tile, 0);
        let neighbor = field(neighbor_key, 0);
        let config = PlanetMeshConfig::new(1_000_000, [0; 3], 1.0).unwrap();
        let mut mesh = PlanetHeightfieldMesh::build(&center, config, LIMITS).unwrap();
        let old_positions = mesh.positions().to_vec();
        let change = center
            .rebake(
                crate::DirtyRegion::new(tile, 0, 0, 0, 0).unwrap(),
                &Constant(100),
                LIMITS,
            )
            .unwrap();

        assert_eq!(
            mesh.update(
                &center,
                change,
                PlanetHeightfieldNormalNeighborhood::new(
                    Some(PlanetHeightfieldNeighbor::Tile(&neighbor)),
                    None,
                    None,
                    None,
                ),
            ),
            Err(BakeError::InvalidGeometry)
        );
        assert_eq!(mesh.revision(), 0);
        assert_eq!(mesh.positions(), old_positions.as_slice());
    }

    #[test]
    fn boundary_snapshot_handles_cross_face_reversed_edge_order() {
        let center_key = TileKey::new(CubeFace::PositiveX, 1, 0, 0).unwrap();
        let center = field(center_key, 17);
        let neighbor_key = center_key.neighbor(TileEdge::West);
        let neighbor = field(neighbor_key, 17);
        let side = neighbor.spec().samples_per_edge();
        let (_, neighbor_edge, _) = center_key
            .neighbor_edge_index(TileEdge::West, side, 0)
            .unwrap();
        let mut edge_mm = [0_i128; 9];
        let mut inner_mm = [0_i128; 9];
        for index in 0..side {
            let (_, edge, mapped) = center_key
                .neighbor_edge_index(TileEdge::West, side, index)
                .unwrap();
            edge_mm[index as usize] = neighbor
                .height_mm(edge_coordinate(edge, side, mapped, false))
                .unwrap();
            inner_mm[index as usize] = neighbor
                .height_mm(edge_coordinate(edge, side, mapped, true))
                .unwrap();
        }
        let snapshot =
            PlanetHeightfieldBoundarySamples::new(neighbor_key, neighbor_edge, &edge_mm, &inner_mm);
        let config = PlanetMeshConfig::new(1_000_000, [0; 3], 1.0).unwrap();
        let mesh = PlanetHeightfieldMesh::build_with_neighborhood(
            &center,
            config,
            LIMITS,
            PlanetHeightfieldNormalNeighborhood::new(
                Some(PlanetHeightfieldNeighbor::Boundary(snapshot)),
                None,
                None,
                None,
            ),
        )
        .unwrap();
        assert_eq!(mesh.normals().len(), 81);
    }

    #[test]
    fn invalid_neighborhood_fails_before_publish_and_budget_error_precedes_it() {
        let center_key = TileKey::new(CubeFace::PositiveZ, 1, 0, 0).unwrap();
        let center = field(center_key, 0);
        let wrong = field(TileKey::root(CubeFace::PositiveZ), 0);
        let config = PlanetMeshConfig::new(1_000_000, [0; 3], 1.0).unwrap();
        let neighborhood = PlanetHeightfieldNormalNeighborhood::new(
            None,
            Some(PlanetHeightfieldNeighbor::Tile(&wrong)),
            None,
            None,
        );
        assert_eq!(
            PlanetHeightfieldMesh::build_with_neighborhood(
                &center,
                config,
                BakeLimits {
                    max_samples: 80,
                    ..LIMITS
                },
                neighborhood,
            ),
            Err(BakeError::BudgetExceeded)
        );
        assert_eq!(
            PlanetHeightfieldMesh::build_with_neighborhood(&center, config, LIMITS, neighborhood,),
            Err(BakeError::InvalidGeometry)
        );
    }

    #[test]
    fn rejects_encoding_error_budget_nonpositive_radius_and_payload_budget() {
        let f = field(TileKey::root(CubeFace::PositiveZ), 0);
        let tight = PlanetMeshConfig::new(6_371_000_000, [0; 3], 0.001).unwrap();
        assert_eq!(
            PlanetHeightfieldMesh::build(&f, tight, LIMITS),
            Err(BakeError::InvalidGeometry)
        );
        assert!(PlanetMeshConfig::new(0, [0; 3], 1.0).is_err());
        assert!(PlanetMeshConfig::new((1_i64 << 52) + 1, [0; 3], 1.0).is_err());
        let config = PlanetMeshConfig::new(100, [0; 3], 1.0).unwrap();
        assert_eq!(
            PlanetHeightfieldMesh::build(&field(f.tile(), -100), config, LIMITS),
            Err(BakeError::InvalidGeometry)
        );
        assert_eq!(
            PlanetHeightfieldMesh::build(
                &f,
                config,
                BakeLimits {
                    max_samples: 80,
                    ..LIMITS
                }
            ),
            Err(BakeError::BudgetExceeded)
        );
        let exact = 81 * 32 + 64 * 24;
        assert_eq!(
            PlanetHeightfieldMesh::build(
                &f,
                config,
                BakeLimits {
                    max_bytes: exact - 1,
                    ..LIMITS
                }
            ),
            Err(BakeError::BudgetExceeded)
        );
        assert!(PlanetHeightfieldMesh::build(
            &f,
            config,
            BakeLimits {
                max_bytes: exact,
                ..LIMITS
            }
        )
        .is_ok());
    }

    #[cfg(feature = "mesh")]
    #[test]
    fn projected_payload_is_a_valid_native_mesh() {
        let projected = PlanetHeightfieldMesh::build(
            &field(TileKey::new(CubeFace::NegativeY, 8, 128, 128).unwrap(), 12),
            PlanetMeshConfig::new(6_371_000_000, [0, -6_371_000_000, 0], 10.0).unwrap(),
            LIMITS,
        )
        .unwrap();
        let native = projected.into_mesh().unwrap();
        native.validate().unwrap();
        assert_eq!(native.vertex_count(), 81);
        assert_eq!(native.indices().unwrap().len(), 384);
    }
}
