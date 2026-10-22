//! Deterministic, bounded CPU Dual Contouring for baked signed-distance grids.
//!
//! One QEF vertex is generated per sign-changing cell. Primal-edge crossings
//! connect those vertices into quads split by a stable coordinate rule. Cells
//! containing checkerboard faces are rejected: resolving their topology needs
//! an explicit asymptotic-decider policy and silently choosing one can create
//! non-manifold output.

use alloc::vec::Vec;

use crate::SdfGrid;

const CORNERS: [[u32; 3]; 8] = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
];
const EDGES: [[usize; 2]; 12] = [
    [0, 1],
    [2, 3],
    [4, 5],
    [6, 7],
    [0, 2],
    [1, 3],
    [4, 6],
    [5, 7],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
];
const FACES: [[usize; 4]; 6] = [
    [0, 1, 3, 2],
    [4, 5, 7, 6],
    [0, 1, 5, 4],
    [2, 3, 7, 6],
    [0, 2, 6, 4],
    [1, 3, 7, 5],
];
const MISSING: u32 = u32::MAX;

/// Integer-millimetre embedding and local output origin for the SDF grid.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DualContouringGeometry {
    sample_origin_mm: [i64; 3],
    local_origin_mm: [i64; 3],
    spacing_mm: [u32; 3],
    max_error_mm: f64,
}

impl DualContouringGeometry {
    /// Creates a positive-spacing grid embedding and bounded local-float policy.
    pub fn new(
        sample_origin_mm: [i64; 3],
        local_origin_mm: [i64; 3],
        spacing_mm: [u32; 3],
        max_error_mm: f64,
    ) -> Result<Self, DualContouringError> {
        if spacing_mm.contains(&0) || !max_error_mm.is_finite() || max_error_mm <= 0.0 {
            return Err(DualContouringError::InvalidGeometry);
        }
        Ok(Self {
            sample_origin_mm,
            local_origin_mm,
            spacing_mm,
            max_error_mm,
        })
    }
}

/// Hard work and output admission limits. All counts are checked before buffers
/// are allocated; `max_bytes` bounds the conservative peak of index map and mesh.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DualContouringLimits {
    /// Maximum cells examined.
    pub max_cells: usize,
    /// Maximum deterministic traversal units (cell visits, full-lattice
    /// primal-edge scans, and output-triangle validation visits).
    pub max_work_units: usize,
    /// Maximum generated vertices.
    pub max_vertices: usize,
    /// Maximum output triangles.
    pub max_triangles: usize,
    /// Maximum logical peak payload bytes, excluding allocator overhead.
    pub max_bytes: usize,
}

/// Successful extraction counts and logical peak payload.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DualContouringStats {
    /// Number of visited cells.
    pub cells: usize,
    /// Deterministic cell and primal-edge traversal units.
    pub work_units: usize,
    /// Active cells / output vertices.
    pub vertices: usize,
    /// Output triangles.
    pub triangles: usize,
    /// Conservative peak bytes admitted for cell map plus output buffers.
    pub peak_bytes: usize,
}

/// Invalid, ambiguous, over-budget, numerically degenerate, or allocation failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DualContouringError {
    /// Invalid spacing or precision budget.
    InvalidGeometry,
    /// A count or byte calculation overflowed.
    SizeOverflow,
    /// Cell, vertex, triangle, or byte budget exceeded.
    CellLimit,
    /// Deterministic traversal work budget exceeded.
    WorkLimit,
    /// Vertex budget or `u32` index capacity exceeded.
    VertexLimit,
    /// Triangle budget exceeded.
    TriangleLimit,
    /// Conservative peak payload exceeds `max_bytes`.
    ByteLimit,
    /// A fallible allocation was rejected.
    AllocationFailed,
    /// An alternating-sign cell face has unresolved topology.
    AmbiguousTopology,
    /// A crossing cell has no usable gradient or finite QEF solution.
    DegenerateCell,
    /// Local coordinates exceed the supported exact integer range.
    CoordinateRange,
    /// Local f32 encoding exceeds the requested error or collapses a triangle.
    PrecisionLoss,
}

/// Indexed, locally encoded dual-contouring surface.
#[derive(Clone, Debug, PartialEq)]
pub struct DualContouringMesh {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    indices: Vec<u32>,
    stats: DualContouringStats,
    revision: u64,
}

impl DualContouringMesh {
    /// Extracts a complete grid. A singleton axis has no cells and returns empty.
    pub fn build(
        grid: &SdfGrid,
        geometry: DualContouringGeometry,
        limits: DualContouringLimits,
    ) -> Result<Self, DualContouringError> {
        let dims = grid.spec().dimensions();
        if dims.contains(&1) {
            return Ok(Self {
                positions: Vec::new(),
                normals: Vec::new(),
                indices: Vec::new(),
                stats: DualContouringStats::default(),
                revision: grid.revision(),
            });
        }
        let cd = [dims[0] - 1, dims[1] - 1, dims[2] - 1];
        let cells = product(cd)?;
        if cells > limits.max_cells {
            return Err(DualContouringError::CellLimit);
        }
        let map_bytes = cells
            .checked_mul(4)
            .ok_or(DualContouringError::SizeOverflow)?;
        if map_bytes > limits.max_bytes {
            return Err(DualContouringError::ByteLimit);
        }

        // Validate ambiguous faces before creating any output. The finite field
        // is immutable, so the subsequent deterministic passes see same values.
        for z in 0..cd[2] {
            for y in 0..cd[1] {
                for x in 0..cd[0] {
                    let v = cell_values(grid, [x, y, z]);
                    if has_ambiguous_face(&v) {
                        return Err(DualContouringError::AmbiguousTopology);
                    }
                }
            }
        }

        let mut active = 0usize;
        for z in 0..cd[2] {
            for y in 0..cd[1] {
                for x in 0..cd[0] {
                    let v = cell_values(grid, [x, y, z]);
                    if !v.iter().all(|v| *v < 0) && !v.iter().all(|v| *v >= 0) {
                        active = active
                            .checked_add(1)
                            .ok_or(DualContouringError::SizeOverflow)?;
                    }
                }
            }
        }
        if active > limits.max_vertices || active > u32::MAX as usize {
            return Err(DualContouringError::VertexLimit);
        }
        let mut triangle_count = 0usize;
        for axis in 0..3 {
            let ranges = [dims[0], dims[1], dims[2]];
            for z in 0..ranges[2] {
                for y in 0..ranges[1] {
                    for x in 0..ranges[0] {
                        let a = [x, y, z];
                        if a[axis] + 1 >= dims[axis] {
                            continue;
                        }
                        let mut b = a;
                        b[axis] += 1;
                        if (raw(grid, a) < 0) == (raw(grid, b) < 0) {
                            continue;
                        }
                        let around = incident_cells(axis, a);
                        if around
                            .iter()
                            .any(|c| c.iter().zip(cd).any(|(v, d)| *v >= d))
                        {
                            continue;
                        }
                        triangle_count = triangle_count
                            .checked_add(2)
                            .ok_or(DualContouringError::SizeOverflow)?;
                    }
                }
            }
        }
        if triangle_count > limits.max_triangles {
            return Err(DualContouringError::TriangleLimit);
        }
        // Both primal-edge passes iterate the complete sample lattice for
        // each axis (including skipped boundary slots); charge those visits,
        // plus the three cell passes and final triangle validation.
        let samples = product(dims)?;
        let work_units = cells
            .checked_mul(3)
            .and_then(|n| samples.checked_mul(6).and_then(|e| n.checked_add(e)))
            .and_then(|n| n.checked_add(triangle_count))
            .ok_or(DualContouringError::SizeOverflow)?;
        if work_units > limits.max_work_units {
            return Err(DualContouringError::WorkLimit);
        }
        let position_bytes = active
            .checked_mul(24)
            .ok_or(DualContouringError::SizeOverflow)?;
        let index_bytes = triangle_count
            .checked_mul(12)
            .ok_or(DualContouringError::SizeOverflow)?;
        let peak_bytes = map_bytes
            .checked_add(position_bytes)
            .and_then(|n| n.checked_add(index_bytes))
            .ok_or(DualContouringError::SizeOverflow)?;
        if peak_bytes > limits.max_bytes {
            return Err(DualContouringError::ByteLimit);
        }
        let mut cell_vertex = reserve::<u32>(cells)?;
        cell_vertex.resize(cells, MISSING);
        let mut positions = reserve::<[f32; 3]>(active)?;
        let mut normals = reserve::<[f32; 3]>(active)?;
        for z in 0..cd[2] {
            for y in 0..cd[1] {
                for x in 0..cd[0] {
                    let p = [x, y, z];
                    let values = cell_values(grid, p);
                    if values.iter().all(|v| *v < 0) || values.iter().all(|v| *v >= 0) {
                        continue;
                    }
                    let (vertex, normal) = solve_cell(p, values, geometry)?;
                    cell_vertex[cell_index(p, cd)] = positions.len() as u32;
                    positions.push(vertex);
                    normals.push(normal);
                }
            }
        }

        let mut indices = reserve::<u32>(
            triangle_count
                .checked_mul(3)
                .ok_or(DualContouringError::SizeOverflow)?,
        )?;
        // Each interior primal edge emits one quad (two triangles). Boundary
        // edges intentionally remain open; this API does not invent a cap.
        for axis in 0..3 {
            let ranges = [dims[0], dims[1], dims[2]];
            for z in 0..ranges[2] {
                for y in 0..ranges[1] {
                    for x in 0..ranges[0] {
                        let a = [x, y, z];
                        if a[axis] + 1 >= dims[axis] {
                            continue;
                        }
                        let mut b = a;
                        b[axis] += 1;
                        let av = raw(grid, a);
                        let bv = raw(grid, b);
                        if (av < 0) == (bv < 0) {
                            continue;
                        }
                        let around = incident_cells(axis, a);
                        if around
                            .iter()
                            .any(|c| c.iter().zip(cd).any(|(v, d)| *v >= d))
                        {
                            continue;
                        }
                        let quad = around.map(|c| cell_vertex[cell_index(c, cd)]);
                        if quad.contains(&MISSING) {
                            return Err(DualContouringError::DegenerateCell);
                        }
                        if quad.iter().enumerate().any(|(i, v)| quad[..i].contains(v)) {
                            return Err(DualContouringError::DegenerateCell);
                        }
                        let needed_triangles = 2usize;
                        let next = indices
                            .len()
                            .checked_add(needed_triangles * 3)
                            .ok_or(DualContouringError::SizeOverflow)?;
                        if next / 3 > limits.max_triangles {
                            return Err(DualContouringError::TriangleLimit);
                        }
                        // Orient quad consistently toward increasing SDF. The cyclic
                        // incident-cell order is positive for X/Y/Z respectively.
                        let ordered = if av < 0 {
                            quad
                        } else {
                            [quad[0], quad[3], quad[2], quad[1]]
                        };
                        let parity = (x ^ y ^ z ^ axis as u32) & 1;
                        if parity == 0 {
                            indices.extend_from_slice(&[
                                ordered[0], ordered[1], ordered[2], ordered[0], ordered[2],
                                ordered[3],
                            ]);
                        } else {
                            indices.extend_from_slice(&[
                                ordered[0], ordered[1], ordered[3], ordered[1], ordered[2],
                                ordered[3],
                            ]);
                        }
                    }
                }
            }
        }
        let triangles = indices.len() / 3;
        if indices.len() / 3 != triangle_count || positions.len() != active {
            return Err(DualContouringError::DegenerateCell);
        }
        // Reject collapsed/incorrectly oriented output before publication.
        for t in indices.chunks_exact(3) {
            let a = positions[t[0] as usize].map(f64::from);
            let b = positions[t[1] as usize].map(f64::from);
            let c = positions[t[2] as usize].map(f64::from);
            let n = cross(sub(b, a), sub(c, a));
            let avg = [0, 1, 2].map(|i| {
                (f64::from(normals[t[0] as usize][i])
                    + f64::from(normals[t[1] as usize][i])
                    + f64::from(normals[t[2] as usize][i]))
                    / 3.0
            });
            if dot(n, n) <= 1.0e-24 || dot(n, avg) <= 0.0 {
                return Err(DualContouringError::PrecisionLoss);
            }
        }
        Ok(Self {
            positions,
            normals,
            indices,
            stats: DualContouringStats {
                cells,
                work_units,
                vertices: active,
                triangles,
                peak_bytes,
            },
            revision: grid.revision(),
        })
    }

    /// Local origin-relative positions in metres.
    pub fn positions(&self) -> &[[f32; 3]] {
        &self.positions
    }
    /// Unit finite-difference SDF normals sampled at the cell center.
    pub fn normals(&self) -> &[[f32; 3]] {
        &self.normals
    }
    /// Deterministically ordered triangle indices.
    pub fn indices(&self) -> &[u32] {
        &self.indices
    }
    /// Work and output accounting.
    pub const fn stats(&self) -> DualContouringStats {
        self.stats
    }
    /// Source grid revision.
    pub const fn revision(&self) -> u64 {
        self.revision
    }
}

fn reserve<T>(n: usize) -> Result<Vec<T>, DualContouringError> {
    let mut v = Vec::new();
    v.try_reserve_exact(n)
        .map_err(|_| DualContouringError::AllocationFailed)?;
    Ok(v)
}
fn product(d: [u32; 3]) -> Result<usize, DualContouringError> {
    (d[0] as usize)
        .checked_mul(d[1] as usize)
        .and_then(|n| n.checked_mul(d[2] as usize))
        .ok_or(DualContouringError::SizeOverflow)
}
fn cell_index(p: [u32; 3], d: [u32; 3]) -> usize {
    p[0] as usize + d[0] as usize * (p[1] as usize + d[1] as usize * p[2] as usize)
}
fn raw(g: &SdfGrid, p: [u32; 3]) -> i32 {
    g.sample_q16(p).expect("validated grid coordinate")
}
fn cell_values(g: &SdfGrid, p: [u32; 3]) -> [i32; 8] {
    CORNERS.map(|o| raw(g, [p[0] + o[0], p[1] + o[1], p[2] + o[2]]))
}
fn has_ambiguous_face(v: &[i32; 8]) -> bool {
    FACES.iter().any(|f| {
        (v[f[0]] < 0) == (v[f[2]] < 0)
            && (v[f[1]] < 0) == (v[f[3]] < 0)
            && (v[f[0]] < 0) != (v[f[1]] < 0)
    })
}
fn incident_cells(axis: usize, p: [u32; 3]) -> [[u32; 3]; 4] {
    match axis {
        0 => [
            [p[0], p[1].wrapping_sub(1), p[2].wrapping_sub(1)],
            [p[0], p[1], p[2].wrapping_sub(1)],
            [p[0], p[1], p[2]],
            [p[0], p[1].wrapping_sub(1), p[2]],
        ],
        1 => [
            [p[0].wrapping_sub(1), p[1], p[2].wrapping_sub(1)],
            [p[0].wrapping_sub(1), p[1], p[2]],
            [p[0], p[1], p[2]],
            [p[0], p[1], p[2].wrapping_sub(1)],
        ],
        _ => [
            [p[0].wrapping_sub(1), p[1].wrapping_sub(1), p[2]],
            [p[0], p[1].wrapping_sub(1), p[2]],
            [p[0], p[1], p[2]],
            [p[0].wrapping_sub(1), p[1], p[2]],
        ],
    }
}

fn solve_cell(
    p: [u32; 3],
    v: [i32; 8],
    geom: DualContouringGeometry,
) -> Result<([f32; 3], [f32; 3]), DualContouringError> {
    let mut ata = [[0.0; 3]; 3];
    let mut atb = [0.0; 3];
    let mut mass = [0.0; 3];
    let mut count = 0.0;
    for [ia, ib] in EDGES {
        let (a, b) = (v[ia], v[ib]);
        if (a < 0) == (b < 0) {
            continue;
        }
        let t = f64::from(a) / (f64::from(a) - f64::from(b));
        let pa = CORNERS[ia].map(f64::from);
        let pb = CORNERS[ib].map(f64::from);
        let q = core::array::from_fn(|i| pa[i] + t * (pb[i] - pa[i]));
        let mut grad = [0.0; 3];
        for axis in 0..3 {
            let lo = pa[axis] < pb[axis];
            let scale = if lo { 1.0 } else { -1.0 };
            // Trilinear derivative, evaluated at the edge crossing. Each axis
            // component is the bilinear interpolation of the four parallel
            // corner differences, evaluated in the other two coordinates.
            let others = match axis {
                0 => [1, 2],
                1 => [0, 2],
                _ => [0, 1],
            };
            let mut sum = 0.0;
            for mask in 0..4 {
                let mut c = [0u32; 3];
                c[axis] = if lo { 0 } else { 1 };
                c[others[0]] = mask as u32 & 1;
                c[others[1]] = (mask as u32 >> 1) & 1;
                let ci = CORNERS
                    .iter()
                    .position(|o| *o == c)
                    .ok_or(DualContouringError::DegenerateCell)?;
                c[axis] ^= 1;
                let di = CORNERS
                    .iter()
                    .position(|o| *o == c)
                    .ok_or(DualContouringError::DegenerateCell)?;
                let w = (if c[others[0]] == 0 {
                    1.0 - q[others[0]]
                } else {
                    q[others[0]]
                }) * (if c[others[1]] == 0 {
                    1.0 - q[others[1]]
                } else {
                    q[others[1]]
                });
                sum += w * f64::from(v[di] - v[ci]) * scale;
            }
            grad[axis] = sum / f64::from(geom.spacing_mm[axis]);
        }
        let len = libm::sqrt(dot(grad, grad));
        if !len.is_finite() || len <= 1e-18 {
            return Err(DualContouringError::DegenerateCell);
        }
        let n = grad.map(|x| x / len);
        for i in 0..3 {
            mass[i] += q[i];
            atb[i] += n[i] * dot(n, q);
            for j in 0..3 {
                ata[i][j] += n[i] * n[j];
            }
        }
        count += 1.0;
    }
    if count == 0.0 {
        return Err(DualContouringError::DegenerateCell);
    }
    let center = mass.map(|x| x / count);
    let lambda = 1e-8;
    for i in 0..3 {
        ata[i][i] += lambda;
        atb[i] += lambda * center[i];
    }
    let q = solve3(ata, atb)
        .ok_or(DualContouringError::DegenerateCell)?
        .map(|x| x.clamp(0.0, 1.0));
    let mut normal = [0.0; 3];
    // Stable normal from the accumulated QEF plane normals is formed by the
    // finite-difference gradient at the cell center.
    for axis in 0..3 {
        let lo = match axis {
            0 => [0, 2, 4, 6],
            1 => [0, 1, 4, 5],
            _ => [0, 1, 2, 3],
        };
        let hi = lo.map(|i| i + (1 << axis));
        normal[axis] = lo
            .iter()
            .zip(hi)
            .map(|(&a, b)| f64::from(v[b] - v[a]))
            .sum::<f64>()
            / 4.0
            / f64::from(geom.spacing_mm[axis]);
    }
    let nl = libm::sqrt(dot(normal, normal));
    if !nl.is_finite() || nl <= 1e-18 {
        return Err(DualContouringError::DegenerateCell);
    }
    normal = normal.map(|x| x / nl);
    let mut out = [0.0f32; 3];
    for axis in 0..3 {
        let base = i128::from(geom.sample_origin_mm[axis]) - i128::from(geom.local_origin_mm[axis])
            + i128::from(p[axis]) * i128::from(geom.spacing_mm[axis]);
        let mm = base as f64 + q[axis] * f64::from(geom.spacing_mm[axis]);
        if base
            .abs()
            .max((base + i128::from(geom.spacing_mm[axis])).abs())
            > (1i128 << 52)
        {
            return Err(DualContouringError::CoordinateRange);
        }
        out[axis] = (mm * 0.001) as f32;
        let err = (f64::from(out[axis]) * 1000.0 - mm).abs()
            + 32.0 * f64::EPSILON * (base.abs() as f64 + 1.0);
        if !out[axis].is_finite() || err > geom.max_error_mm {
            return Err(DualContouringError::PrecisionLoss);
        }
    }
    Ok((out, normal.map(|x| x as f32)))
}
fn solve3(mut a: [[f64; 3]; 3], mut b: [f64; 3]) -> Option<[f64; 3]> {
    for i in 0..3 {
        let mut pivot = i;
        for r in i + 1..3 {
            if a[r][i].abs() > a[pivot][i].abs() {
                pivot = r;
            }
        }
        if !a[pivot][i].is_finite() || a[pivot][i].abs() < 1e-20 {
            return None;
        }
        a.swap(i, pivot);
        b.swap(i, pivot);
        let d = a[i][i];
        for c in i..3 {
            a[i][c] /= d;
        }
        b[i] /= d;
        for r in 0..3 {
            if r != i {
                let f = a[r][i];
                for c in i..3 {
                    a[r][c] -= f * a[i][c];
                }
                b[r] -= f * b[i];
            }
        }
    }
    if b.iter().all(|x| x.is_finite()) {
        Some(b)
    } else {
        None
    }
}
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    core::array::from_fn(|i| a[i] - b[i])
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a.into_iter().zip(b).map(|(x, y)| x * y).sum()
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
    use crate::{SdfField, SdfSpec, VolumeBakeLimits};
    use alloc::vec::Vec;
    use core::convert::Infallible;

    struct Field {
        spec: SdfSpec,
        kind: u8,
    }
    impl SdfField for Field {
        type Error = Infallible;
        fn spec(&self) -> SdfSpec {
            self.spec
        }
        fn sample_q16(&self, p: [u32; 3]) -> Result<i32, Self::Error> {
            let [x, y, z] = p.map(i64::from);
            Ok(match self.kind {
                0 => (x - 4) as i32 * 65_536,
                _ => {
                    let dx = x - 4;
                    let dy = y - 4;
                    let dz = z - 4;
                    (dx * dx + dy * dy + dz * dz - 9) as i32 * 65_536
                }
            })
        }
    }
    fn grid(kind: u8) -> SdfGrid {
        let spec = SdfSpec::new([9, 9, 9], 8, 1).unwrap();
        SdfGrid::bake(
            &Field { spec, kind },
            VolumeBakeLimits {
                max_dimensions: [9; 3],
                max_samples: 729,
                max_bytes: 729 * 4,
            },
        )
        .unwrap()
    }
    fn geom() -> DualContouringGeometry {
        DualContouringGeometry::new([0; 3], [0; 3], [1000; 3], 1.0).unwrap()
    }
    fn limits() -> DualContouringLimits {
        DualContouringLimits {
            max_cells: 512,
            max_work_units: 8192,
            max_vertices: 512,
            max_triangles: 1024,
            max_bytes: 512 * 4 + 512 * 24 + 1024 * 12,
        }
    }

    #[test]
    fn plane_is_stable_and_oriented_toward_positive_distance() {
        let grid = grid(0);
        let mesh = DualContouringMesh::build(&grid, geom(), limits()).unwrap();
        assert!(!mesh.indices().is_empty());
        assert!(mesh.positions().iter().all(|p| (p[0] - 4.0).abs() < 1e-5));
        assert!(mesh.normals().iter().all(|n| n[0] > 0.99));
        assert!(mesh.indices().chunks_exact(3).all(|t| {
            let a = mesh.positions()[t[0] as usize];
            let b = mesh.positions()[t[1] as usize];
            let c = mesh.positions()[t[2] as usize];
            cross(
                sub(b.map(f64::from), a.map(f64::from)),
                sub(c.map(f64::from), a.map(f64::from)),
            )[0] > 0.0
        }));
    }

    #[test]
    fn sphere_surface_is_closed_manifold_and_vertices_are_inside_cells() {
        let mesh = DualContouringMesh::build(&grid(1), geom(), limits()).unwrap();
        assert!(mesh.indices().len() >= 36);
        // The radius-three sphere centered at sample coordinate four reaches
        // the x/y/z cell boundaries 1 and 7; every QEF vertex must remain in
        // that enclosing active-cell range.
        for p in mesh.positions() {
            for c in p {
                assert!(*c >= 1.0 && *c <= 7.0);
            }
        }
        let mut edges: Vec<(u32, u32)> = Vec::new();
        for t in mesh.indices().chunks_exact(3) {
            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                edges.push((a.min(b), a.max(b)));
            }
        }
        edges.sort_unstable();
        assert!(
            edges.chunks_exact(2).all(|pair| pair[0] == pair[1]),
            "every sphere edge has exactly two incident triangles"
        );
    }

    #[test]
    fn checkerboard_face_is_rejected_as_ambiguous() {
        let spec = SdfSpec::new([2, 2, 2], 2, 1).unwrap();
        struct Checker(SdfSpec);
        impl SdfField for Checker {
            type Error = Infallible;
            fn spec(&self) -> SdfSpec {
                self.0
            }
            fn sample_q16(&self, p: [u32; 3]) -> Result<i32, Self::Error> {
                Ok(if (p[0] + p[1]) % 2 == 0 { -1 } else { 1 })
            }
        }
        let g = SdfGrid::bake(
            &Checker(spec),
            VolumeBakeLimits {
                max_dimensions: [2; 3],
                max_samples: 8,
                max_bytes: 32,
            },
        )
        .unwrap();
        assert_eq!(
            DualContouringMesh::build(
                &g,
                geom(),
                DualContouringLimits {
                    max_cells: 1,
                    max_work_units: 64,
                    max_vertices: 8,
                    max_triangles: 12,
                    max_bytes: 4096
                }
            ),
            Err(DualContouringError::AmbiguousTopology)
        );
    }

    #[test]
    fn each_budget_fails_before_any_mesh_is_published() {
        let g = grid(1);
        let full = DualContouringMesh::build(&g, geom(), limits()).unwrap();
        let mut l = limits();
        l.max_work_units = full.stats().work_units - 1;
        assert_eq!(
            DualContouringMesh::build(&g, geom(), l),
            Err(DualContouringError::WorkLimit)
        );
        l.max_work_units = full.stats().work_units;
        assert_eq!(
            DualContouringMesh::build(&g, geom(), l).unwrap().stats(),
            full.stats()
        );
        let mut l = limits();
        l.max_cells = 1;
        assert_eq!(
            DualContouringMesh::build(&g, geom(), l),
            Err(DualContouringError::CellLimit)
        );
        let mut l = limits();
        l.max_work_units = 1;
        assert_eq!(
            DualContouringMesh::build(&g, geom(), l),
            Err(DualContouringError::WorkLimit)
        );
        let mut l = limits();
        l.max_vertices = 0;
        assert_eq!(
            DualContouringMesh::build(&g, geom(), l),
            Err(DualContouringError::VertexLimit)
        );
        let mut l = limits();
        l.max_triangles = 0;
        assert_eq!(
            DualContouringMesh::build(&g, geom(), l),
            Err(DualContouringError::TriangleLimit)
        );
        let mut l = limits();
        l.max_bytes = 0;
        assert_eq!(
            DualContouringMesh::build(&g, geom(), l),
            Err(DualContouringError::ByteLimit)
        );
        assert!(!full.indices().is_empty());
    }

    #[test]
    fn extraction_order_is_bitwise_deterministic() {
        let g = grid(1);
        let a = DualContouringMesh::build(&g, geom(), limits()).unwrap();
        let b = DualContouringMesh::build(&g, geom(), limits()).unwrap();
        assert_eq!(a, b);
    }
}
