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
    /// operation.
    pub fn update(
        &mut self,
        field: &HeightfieldTile,
        change: crate::GridChange,
        neighborhood: PlanetHeightfieldNormalNeighborhood<'_>,
    ) -> Result<crate::SampleBounds, BakeError<Infallible>> {
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
    /// Outward triangle indices.
    pub fn indices(&self) -> &[u32] {
        &self.indices
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
