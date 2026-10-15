//! Bounded zero-isosurface extraction on a conforming six-tetrahedra lattice.
//!
//! Negative samples are inside; zero belongs to the outside. The same cube
//! decomposition is used everywhere, including region boundaries. This is a
//! piecewise-linear marching-tetrahedra surface, not sharp-feature Dual
//! Contouring or a mixed-resolution transition mesh. Canonical sample/edge keys
//! weld vertices; area-weighted normals describe the delivered triangles.

use alloc::vec::Vec;

use crate::{SdfGrid, VolumeRegion};

const TETS: [[usize; 4]; 6] = [
    [0, 1, 3, 7],
    [0, 3, 2, 7],
    [0, 2, 6, 7],
    [0, 6, 4, 7],
    [0, 4, 5, 7],
    [0, 5, 1, 7],
];

/// Integer spatial embedding and local float encoding policy for an SDF grid.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceGeometry {
    sample_origin_mm: [i64; 3],
    local_origin_mm: [i64; 3],
    spacing_mm: [u32; 3],
    max_error_mm: f64,
}

impl SurfaceGeometry {
    /// Grid sample zero is at `sample_origin_mm`; axes have positive millimetre
    /// spacing. Output positions subtract `local_origin_mm` before f32 metres.
    /// The error budget covers numerical position encoding, not field sampling,
    /// distance quantization, triangulation, or approximation of a true surface.
    pub fn new(
        sample_origin_mm: [i64; 3],
        local_origin_mm: [i64; 3],
        spacing_mm: [u32; 3],
        max_error_mm: f64,
    ) -> Result<Self, SurfaceError> {
        if spacing_mm.contains(&0) || !max_error_mm.is_finite() || max_error_mm <= 0.0 {
            return Err(SurfaceError::InvalidGeometry);
        }
        Ok(Self {
            sample_origin_mm,
            local_origin_mm,
            spacing_mm,
            max_error_mm,
        })
    }
}

/// Per-build work and allocation admission. No partial mesh is published.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SurfaceLimits {
    /// Maximum visited cubes (each contains six tetrahedra), checked up front.
    pub max_cells: usize,
    /// Maximum emitted non-degenerate triangles.
    pub max_triangles: usize,
    /// Maximum welded output vertices, also bounded by u32 index capacity.
    pub max_vertices: usize,
    /// Maximum simultaneously live logical buffer payload allocated here.
    /// Excludes the borrowed grid, old meshes, allocator overhead/capacity
    /// rounding, and the bounded traversal stack; see [`SurfaceCost`].
    pub max_bytes: usize,
}

/// Exact geometry counts and logical payload costs of a successful build.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SurfaceCost {
    /// Number of cubes visited in each of three lattice traversals.
    pub cells: usize,
    /// Number of output triangles.
    pub triangles: usize,
    /// Number of canonical shared vertices.
    pub vertices: usize,
    /// Output payload: 24 bytes per vertex and 12 per triangle.
    pub output_bytes: usize,
    /// Peak of edge sorting/index construction and normal accumulation stages.
    /// `max(60*T + 12*V, 12*T + 48*V)`; sorting is in-place and unstable.
    pub peak_bytes: usize,
}

/// Surface validation, budget, precision, or fallible-allocation failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SurfaceError {
    /// Non-positive spacing or invalid numerical error budget.
    InvalidGeometry,
    /// Requested cubes extend beyond the sample grid.
    RegionOutOfBounds,
    /// A count, sample identity, or byte calculation overflowed.
    SizeOverflow,
    /// Requested work exceeds the cube limit.
    CellLimit,
    /// The output exceeds its triangle limit.
    TriangleLimit,
    /// The output exceeds its vertex limit or u32 index capacity.
    VertexLimit,
    /// A staging or output stage exceeds the logical byte limit.
    ByteLimit,
    /// The allocator rejected a fallible reservation.
    AllocationFailed,
    /// Local integer coordinates exceed the supported +/-2^52 millimetres.
    CoordinateRange,
    /// Local f32 encoding exceeds the error budget or collapses/flips a face.
    PrecisionLoss,
    /// Incident triangle normals cancel; no finite nonzero unit normal exists.
    UndefinedNormal,
}

/// Indexed render/collision geometry extracted from a borrowed baked SDF grid.
#[derive(Clone, Debug, PartialEq)]
pub struct SdfSurfaceMesh {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    indices: Vec<u32>,
    cost: SurfaceCost,
    revision: u64,
    max_position_error_mm: f64,
}

// Endpoints are sorted X-fastest sample identities. [i,i] is an exact zero
// sample, ensuring all incident edges use one vertex even at lattice vertices.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct Edge([u64; 2]);

impl SdfSurfaceMesh {
    /// Extracts the complete grid. A singleton axis has no cubes and produces
    /// an empty mesh. Empty/all-positive/all-negative grids allocate no buffers.
    pub fn build(
        grid: &SdfGrid,
        geometry: SurfaceGeometry,
        limits: SurfaceLimits,
    ) -> Result<Self, SurfaceError> {
        let end = grid.spec().dimensions().map(|d| d - 1);
        if end.contains(&0) {
            return Ok(Self::empty(grid.revision(), 0));
        }
        Self::build_region(
            grid,
            VolumeRegion::new([0; 3], end).expect("nonzero"),
            geometry,
            limits,
        )
    }

    /// Extracts a half-open box of CUBES, not samples. Cube `[x,y,z]` reads
    /// samples through `[x+1,y+1,z+1]`. Regions partition topology exactly at
    /// equal resolution; normal averaging at open boundaries is one-sided.
    /// Rebuilding dirty cube regions is supported; in-place topology patching,
    /// cross-region smooth normals, and mixed-LOD stitching are not implied.
    pub fn build_region(
        grid: &SdfGrid,
        region: VolumeRegion,
        geometry: SurfaceGeometry,
        limits: SurfaceLimits,
    ) -> Result<Self, SurfaceError> {
        if region
            .end()
            .iter()
            .zip(grid.spec().dimensions())
            .any(|(end, dim)| *end >= dim)
        {
            return Err(SurfaceError::RegionOutOfBounds);
        }
        let cells = region
            .sample_count()
            .map_err(|_| SurfaceError::SizeOverflow)?;
        if cells > limits.max_cells {
            return Err(SurfaceError::CellLimit);
        }
        let mut triangles = 0_usize;
        visit(grid, region, |_, _| {
            triangles = triangles.checked_add(1).ok_or(SurfaceError::SizeOverflow)?;
            if triangles > limits.max_triangles {
                return Err(SurfaceError::TriangleLimit);
            }
            Ok(())
        })?;
        if triangles == 0 {
            return Ok(Self::empty(grid.revision(), cells));
        }
        let count = triangles.checked_mul(3).ok_or(SurfaceError::SizeOverflow)?;
        let edge_bytes = count.checked_mul(16).ok_or(SurfaceError::SizeOverflow)?;
        if edge_bytes > limits.max_bytes {
            return Err(SurfaceError::ByteLimit);
        }
        let mut edges = reserve(count)?;
        visit(grid, region, |triangle, _| {
            edges.extend_from_slice(&triangle);
            Ok(())
        })?;
        edges.sort_unstable();
        edges.dedup();
        let vertices = edges.len();
        if vertices > limits.max_vertices || vertices > u32::MAX as usize {
            return Err(SurfaceError::VertexLimit);
        }
        let cost = cost(cells, triangles, vertices)?;
        if cost.peak_bytes > limits.max_bytes {
            return Err(SurfaceError::ByteLimit);
        }
        let mut positions = reserve(vertices)?;
        let mut max_position_error_mm = 0.0_f64;
        for &edge in &edges {
            let (position, error) = position(grid, edge, geometry)?;
            positions.push(position);
            max_position_error_mm = max_position_error_mm.max(error);
        }
        let mut indices = reserve(count)?;
        visit(grid, region, |triangle, expected_normal| {
            let ids =
                triangle.map(|edge| edges.binary_search(&edge).expect("collected edge") as u32);
            let p = ids.map(|i| positions[i as usize].map(f64::from));
            let normal = cross(sub(p[1], p[0]), sub(p[2], p[0]));
            // The reference normal is in grid coordinates; apply the inverse
            // transpose of the positive diagonal embedding before comparing.
            let expected =
                core::array::from_fn(|i| expected_normal[i] / f64::from(geometry.spacing_mm[i]));
            if dot(normal, expected) <= 0.0 {
                return Err(SurfaceError::PrecisionLoss);
            }
            indices.extend_from_slice(&ids);
            Ok(())
        })?;
        drop(edges);
        let mut sums = reserve(vertices)?;
        sums.resize(vertices, [0.0_f64; 3]);
        for triangle in indices.chunks_exact(3) {
            let p = [triangle[0], triangle[1], triangle[2]]
                .map(|i| positions[i as usize].map(f64::from));
            let normal = cross(sub(p[1], p[0]), sub(p[2], p[0]));
            for &i in triangle {
                for (sum, component) in sums[i as usize].iter_mut().zip(normal) {
                    *sum += component;
                }
            }
        }
        let mut normals = reserve(vertices)?;
        for sum in sums {
            let length = libm::sqrt(dot(sum, sum));
            if !length.is_finite() || length <= 0.0 {
                return Err(SurfaceError::UndefinedNormal);
            }
            normals.push(sum.map(|c| (c / length) as f32));
        }
        Ok(Self {
            positions,
            normals,
            indices,
            cost,
            revision: grid.revision(),
            max_position_error_mm,
        })
    }

    fn empty(revision: u64, cells: usize) -> Self {
        Self {
            positions: Vec::new(),
            normals: Vec::new(),
            indices: Vec::new(),
            cost: SurfaceCost {
                cells,
                ..SurfaceCost::default()
            },
            revision,
            max_position_error_mm: 0.0,
        }
    }

    /// Welded origin-relative positions in metres.
    pub fn positions(&self) -> &[[f32; 3]] {
        &self.positions
    }
    /// Area-weighted unit normals in the same local space.
    pub fn normals(&self) -> &[[f32; 3]] {
        &self.normals
    }
    /// Outward CCW triangle indices, pointing from negative to positive samples.
    pub fn indices(&self) -> &[u32] {
        &self.indices
    }
    /// Actual counts and peak logical payload admission.
    pub const fn cost(&self) -> SurfaceCost {
        self.cost
    }
    /// Revision of the source grid at extraction (not a global cache identity).
    pub const fn revision(&self) -> u64 {
        self.revision
    }
    /// Conservative maximum per-axis numerical position bound in millimetres.
    pub const fn max_position_error_mm(&self) -> f64 {
        self.max_position_error_mm
    }

    /// Moves vertex/index storage into a validated native Prism triangle mesh.
    /// No UV chart is fabricated for an arbitrary volumetric surface.
    #[cfg(feature = "mesh")]
    pub fn into_mesh(self) -> Result<prism_mesh::Mesh, prism_mesh::MeshError> {
        use prism_mesh::{Indices, Mesh, MeshVertexAttribute, VertexAttributeValues};
        use prism_render_driver::{PrimitiveTopology, VertexFormat};
        let mesh = Mesh::new(PrimitiveTopology::TriangleList)
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
            .indexed(Indices::Uint32(self.indices));
        mesh.validate()?;
        Ok(mesh)
    }
}

fn reserve<T>(count: usize) -> Result<Vec<T>, SurfaceError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| SurfaceError::AllocationFailed)?;
    Ok(values)
}

fn cost(cells: usize, triangles: usize, vertices: usize) -> Result<SurfaceCost, SurfaceError> {
    let mul = |a: usize, b| a.checked_mul(b).ok_or(SurfaceError::SizeOverflow);
    let add = |a: usize, b| a.checked_add(b).ok_or(SurfaceError::SizeOverflow);
    Ok(SurfaceCost {
        cells,
        triangles,
        vertices,
        output_bytes: add(mul(vertices, 24)?, mul(triangles, 12)?)?,
        peak_bytes: add(mul(triangles, 60)?, mul(vertices, 12)?)?
            .max(add(mul(triangles, 12)?, mul(vertices, 48)?)?),
    })
}

fn coordinate(id: u64, dimensions: [u32; 3]) -> [u32; 3] {
    let x = id % u64::from(dimensions[0]);
    let row = id / u64::from(dimensions[0]);
    [
        x as u32,
        (row % u64::from(dimensions[1])) as u32,
        (row / u64::from(dimensions[1])) as u32,
    ]
}

fn raw(grid: &SdfGrid, p: [u32; 3]) -> i32 {
    grid.sample_q16(p).expect("validated cube bounds")
}

fn position(
    grid: &SdfGrid,
    edge: Edge,
    geometry: SurfaceGeometry,
) -> Result<([f32; 3], f64), SurfaceError> {
    let [a, b] = edge.0.map(|id| coordinate(id, grid.spec().dimensions()));
    let av = f64::from(raw(grid, a));
    let bv = f64::from(raw(grid, b));
    let t = if a == b { 0.0 } else { av / (av - bv) };
    let mut output = [0.0; 3];
    let mut error = 0.0_f64;
    for i in 0..3 {
        let base = i128::from(geometry.sample_origin_mm[i])
            - i128::from(geometry.local_origin_mm[i])
            + i128::from(a[i]) * i128::from(geometry.spacing_mm[i]);
        let other = i128::from(geometry.sample_origin_mm[i])
            - i128::from(geometry.local_origin_mm[i])
            + i128::from(b[i]) * i128::from(geometry.spacing_mm[i]);
        if base.abs().max(other.abs()) > (1_i128 << 52) {
            return Err(SurfaceError::CoordinateRange);
        }
        let span = (other - base) as f64;
        let mm = base as f64 + span * t;
        output[i] = (mm * 0.001) as f32;
        // Deliberately generous arithmetic allowance for interpolation,
        // metre conversion, and reconstructing the f32 encoding error.
        let bound = (f64::from(output[i]) * 1000.0 - mm).abs()
            + 32.0 * f64::EPSILON * ((base as f64).abs() + span.abs() + 1.0);
        if !output[i].is_finite() || bound > geometry.max_error_mm {
            return Err(SurfaceError::PrecisionLoss);
        }
        error = error.max(bound);
    }
    Ok((output, error))
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    core::array::from_fn(|i| a[i] - b[i])
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a.into_iter().zip(b).map(|(a, b)| a * b).sum()
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn visit(
    grid: &SdfGrid,
    region: VolumeRegion,
    mut emit: impl FnMut([Edge; 3], [f64; 3]) -> Result<(), SurfaceError>,
) -> Result<(), SurfaceError> {
    let dims = grid.spec().dimensions();
    let min = region.min();
    let end = region.end();
    for z in min[2]..end[2] {
        for y in min[1]..end[1] {
            for x in min[0]..end[0] {
                let p: [[u32; 3]; 8] = core::array::from_fn(|i| {
                    [
                        x + (i as u32 & 1),
                        y + ((i as u32 >> 1) & 1),
                        z + (i as u32 >> 2),
                    ]
                });
                let values = p.map(|p| raw(grid, p));
                let mut ids = [0; 8];
                for (id, p) in ids.iter_mut().zip(p) {
                    *id = u64::from(p[2])
                        .checked_mul(u64::from(dims[1]))
                        .and_then(|n| n.checked_add(u64::from(p[1])))
                        .and_then(|n| n.checked_mul(u64::from(dims[0])))
                        .and_then(|n| n.checked_add(u64::from(p[0])))
                        .ok_or(SurfaceError::SizeOverflow)?;
                }
                let local = |i: usize| [(i & 1) as f64, ((i >> 1) & 1) as f64, (i >> 2) as f64];
                let edge = |a: usize, b: usize| {
                    if values[a] == 0 {
                        Edge([ids[a]; 2])
                    } else if values[b] == 0 {
                        Edge([ids[b]; 2])
                    } else {
                        Edge([ids[a].min(ids[b]), ids[a].max(ids[b])])
                    }
                };
                let point = |a: usize, b: usize| {
                    let t = f64::from(values[a]) / (f64::from(values[a]) - f64::from(values[b]));
                    let ap = local(a);
                    let bp = local(b);
                    core::array::from_fn(|i| ap[i] + t * (bp[i] - ap[i]))
                };
                for tet in TETS {
                    let mut inside = [0; 4];
                    let mut outside = [0; 4];
                    let (mut ni, mut no) = (0, 0);
                    for i in tet {
                        if values[i] < 0 {
                            inside[ni] = i;
                            ni += 1;
                        } else {
                            outside[no] = i;
                            no += 1;
                        }
                    }
                    if ni == 0 || ni == 4 {
                        continue;
                    }
                    let outward = sub(local(outside[0]), local(inside[0]));
                    let mut triangle = |pairs: [(usize, usize); 3]| {
                        let mut edges = pairs.map(|(a, b)| edge(a, b));
                        if edges[0] == edges[1] || edges[1] == edges[2] || edges[0] == edges[2] {
                            return Ok(());
                        }
                        let points = pairs.map(|(a, b)| point(a, b));
                        let mut normal =
                            cross(sub(points[1], points[0]), sub(points[2], points[0]));
                        if dot(normal, outward) < 0.0 {
                            edges.swap(1, 2);
                            normal = normal.map(|n| -n);
                        }
                        emit(edges, normal)
                    };
                    match ni {
                        1 => triangle([
                            (inside[0], outside[0]),
                            (inside[0], outside[1]),
                            (inside[0], outside[2]),
                        ])?,
                        3 => triangle([
                            (inside[0], outside[0]),
                            (inside[1], outside[0]),
                            (inside[2], outside[0]),
                        ])?,
                        2 => {
                            triangle([
                                (inside[0], outside[0]),
                                (inside[1], outside[0]),
                                (inside[1], outside[1]),
                            ])?;
                            triangle([
                                (inside[0], outside[0]),
                                (inside[1], outside[1]),
                                (inside[0], outside[1]),
                            ])?;
                        }
                        _ => unreachable!(),
                    }
                }
            }
        }
    }
    Ok(())
}
