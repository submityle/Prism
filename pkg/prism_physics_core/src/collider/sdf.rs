//! Signed distance field (SDF) generation for triangle-mesh colliders.
//!
//! A signed distance field samples, on a regular grid, the shortest distance
//! from each grid node to the surface of a closed triangle mesh, with the sign
//! chosen negative inside the solid and positive outside. SDFs are a staple of
//! AAA collision pipelines (UE `Chaos` level-set/particle collision,
//! `Jolt`/`PhysX` cached meshes) because they turn an expensive nearest-triangle
//! search into an `O(1)` trilinear lookup plus a cheap gradient for the surface
//! normal. They are especially useful for particle, soft-body and fluid
//! collision against otherwise static geometry.
//!
//! This module builds the field from an arbitrary triangle soup and exposes the
//! queries downstream stages need:
//!
//! - [`MeshSdf::sample`] -- trilinear-interpolated signed distance anywhere,
//! - [`MeshSdf::gradient`] -- the (unit) field gradient, i.e. the outward
//!   surface normal direction, via central differences, and
//! - [`MeshSdf::project_to_surface`] -- a single Newton step onto the zero
//!   iso-surface, the primitive a collision resolver needs.
//!
//! # Method
//!
//! Each grid node stores the exact unsigned distance to the nearest triangle
//! (closed-form point/triangle distance, Ericson, *Real-Time Collision
//! Detection*). The sign is resolved by ray parity: a ray cast from the node
//! crosses a closed surface an odd number of times when the node is inside. To
//! stay robust against rays grazing shared edges, three axis-aligned rays are
//! cast and the inside/outside decision is a majority vote.
//!
//! Everything here is standard computational geometry (point/triangle distance,
//! Moller-Trumbore ray/triangle intersection, trilinear interpolation); nothing
//! is derived from Unreal Engine source.

use glam::Vec3;

/// Minimum number of nodes along each grid axis.
const MIN_NODES: usize = 2;
/// Hard cap on total nodes, to keep an accidental tiny `cell_size` from
/// allocating an unbounded grid.
const MAX_NODES_TOTAL: usize = 1 << 24; // ~16.7M nodes.
/// Degenerate-triangle area threshold (squared double-area).
const MIN_TRI_AREA2_SQ: f32 = 1e-20;

/// Parameters controlling how a [`MeshSdf`] is discretised.
#[derive(Clone, Copy, Debug)]
pub struct SdfBuildParams {
    /// Uniform spacing between neighbouring grid nodes, in mesh-local units.
    pub cell_size: f32,
    /// Extra margin added around the mesh axis-aligned bounds so the field
    /// resolves a shell of positive distance outside the surface.
    pub padding: f32,
}

impl SdfBuildParams {
    /// Builds parameters from a target resolution along the mesh's longest
    /// axis. The resulting `cell_size` yields roughly `resolution` cells across
    /// that axis; `padding` defaults to two cells.
    #[must_use]
    pub fn from_resolution(longest_extent: f32, resolution: u32) -> Self {
        let res = resolution.max(1) as f32;
        let cell_size = (longest_extent / res).max(f32::MIN_POSITIVE);
        Self {
            cell_size,
            padding: cell_size * 2.0,
        }
    }
}

/// A dense, regular-grid signed distance field for a triangle mesh.
///
/// Node `(i, j, k)` sits at `origin + cell_size * (i, j, k)` and the stored
/// value is the signed distance there (negative inside). Nodes are laid out
/// with `i` fastest, then `j`, then `k`.
#[derive(Clone, Debug)]
pub struct MeshSdf {
    origin: Vec3,
    cell_size: f32,
    dims: [usize; 3],
    data: Vec<f32>,
}

impl MeshSdf {
    /// Builds a signed distance field for the closed triangle mesh described by
    /// `vertices` and triangle `indices`.
    ///
    /// Returns `None` when the mesh is empty, every triangle is degenerate, the
    /// parameters are non-finite/non-positive, or the requested grid would
    /// exceed [`MAX_NODES_TOTAL`].
    #[must_use]
    pub fn from_mesh(
        vertices: &[Vec3],
        indices: &[[u32; 3]],
        params: SdfBuildParams,
    ) -> Option<Self> {
        if !params.cell_size.is_finite()
            || params.cell_size <= 0.0
            || !params.padding.is_finite()
            || params.padding < 0.0
        {
            return None;
        }

        // Gather the valid (non-degenerate, in-range) triangles once.
        let tris = collect_triangles(vertices, indices);
        if tris.is_empty() {
            return None;
        }

        // Local bounds of the triangles, then pad.
        let (mut min, mut max) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
        for t in &tris {
            for v in &t.v {
                min = min.min(*v);
                max = max.max(*v);
            }
        }
        min -= Vec3::splat(params.padding);
        max += Vec3::splat(params.padding);

        let span = max - min;
        let cell = params.cell_size;
        // Node counts: at least MIN_NODES, enough to cover the padded span.
        let node_count = |extent: f32| -> usize {
            let cells = (extent / cell).ceil() as i64;
            (cells.max(1) as usize + 1).max(MIN_NODES)
        };
        let dims = [node_count(span.x), node_count(span.y), node_count(span.z)];
        let total = dims[0].checked_mul(dims[1])?.checked_mul(dims[2])?;
        if total == 0 || total > MAX_NODES_TOTAL {
            return None;
        }

        let mut data = vec![0.0_f32; total];
        for k in 0..dims[2] {
            for j in 0..dims[1] {
                for i in 0..dims[0] {
                    let p = min + Vec3::new(i as f32, j as f32, k as f32) * cell;
                    let unsigned = nearest_distance(p, &tris).sqrt();
                    let sign = if point_is_inside(p, &tris) { -1.0 } else { 1.0 };
                    let idx = i + dims[0] * (j + dims[1] * k);
                    data[idx] = sign * unsigned;
                }
            }
        }

        Some(Self {
            origin: min,
            cell_size: cell,
            dims,
            data,
        })
    }

    /// Grid node counts along each axis.
    #[must_use]
    pub fn dims(&self) -> [usize; 3] {
        self.dims
    }

    /// Uniform node spacing in mesh-local units.
    #[must_use]
    pub fn cell_size(&self) -> f32 {
        self.cell_size
    }

    /// Local position of node `(0, 0, 0)` (the field's minimum corner).
    #[must_use]
    pub fn origin(&self) -> Vec3 {
        self.origin
    }

    /// Local position of the field's maximum corner.
    #[must_use]
    pub fn max_corner(&self) -> Vec3 {
        self.origin
            + Vec3::new(
                (self.dims[0] - 1) as f32,
                (self.dims[1] - 1) as f32,
                (self.dims[2] - 1) as f32,
            ) * self.cell_size
    }

    /// Axis-aligned bounds covered by the field, as `(min, max)`.
    #[must_use]
    pub fn local_aabb(&self) -> (Vec3, Vec3) {
        (self.origin, self.max_corner())
    }

    /// The stored signed distance at an exact grid node, or `None` when the
    /// index is out of range.
    #[must_use]
    pub fn node_distance(&self, i: usize, j: usize, k: usize) -> Option<f32> {
        if i >= self.dims[0] || j >= self.dims[1] || k >= self.dims[2] {
            return None;
        }
        Some(self.data[i + self.dims[0] * (j + self.dims[1] * k)])
    }

    /// Trilinearly interpolated signed distance at an arbitrary local point.
    ///
    /// Queries outside the field are clamped to the boundary, so the returned
    /// value is a conservative lower bound on the true distance there.
    #[must_use]
    pub fn sample(&self, point: Vec3) -> f32 {
        let (base, frac) = self.cell_coords(point);
        let [i0, j0, k0] = base;
        let i1 = (i0 + 1).min(self.dims[0] - 1);
        let j1 = (j0 + 1).min(self.dims[1] - 1);
        let k1 = (k0 + 1).min(self.dims[2] - 1);

        let d = |i: usize, j: usize, k: usize| self.data[i + self.dims[0] * (j + self.dims[1] * k)];

        // Interpolate along x, then y, then z.
        let c00 = lerp(d(i0, j0, k0), d(i1, j0, k0), frac.x);
        let c10 = lerp(d(i0, j1, k0), d(i1, j1, k0), frac.x);
        let c01 = lerp(d(i0, j0, k1), d(i1, j0, k1), frac.x);
        let c11 = lerp(d(i0, j1, k1), d(i1, j1, k1), frac.x);
        let c0 = lerp(c00, c10, frac.y);
        let c1 = lerp(c01, c11, frac.y);
        lerp(c0, c1, frac.z)
    }

    /// Unit gradient of the field at `point`, i.e. the outward surface-normal
    /// direction. Falls back to `+Y` when the local gradient vanishes.
    #[must_use]
    pub fn gradient(&self, point: Vec3) -> Vec3 {
        let h = self.cell_size;
        let dx = self.sample(point + Vec3::X * h) - self.sample(point - Vec3::X * h);
        let dy = self.sample(point + Vec3::Y * h) - self.sample(point - Vec3::Y * h);
        let dz = self.sample(point + Vec3::Z * h) - self.sample(point - Vec3::Z * h);
        let g = Vec3::new(dx, dy, dz);
        let len = g.length();
        if len > 1e-12 {
            g / len
        } else {
            Vec3::Y
        }
    }

    /// A single Newton projection of `point` onto the zero iso-surface: moves
    /// against the gradient by the sampled signed distance. Repeated calls
    /// converge onto the surface; one step is enough for shallow penetration.
    #[must_use]
    pub fn project_to_surface(&self, point: Vec3) -> Vec3 {
        let d = self.sample(point);
        point - self.gradient(point) * d
    }

    /// Maps a local point to its lower grid-cell index and the in-cell
    /// fractional offset in `[0, 1]`, clamping to the field bounds.
    fn cell_coords(&self, point: Vec3) -> ([usize; 3], Vec3) {
        let rel = (point - self.origin) / self.cell_size;
        let axis = |value: f32, nodes: usize| -> (usize, f32) {
            if nodes <= 1 {
                return (0, 0.0);
            }
            let max_base = nodes - 2;
            let clamped = value.clamp(0.0, (nodes - 1) as f32);
            let base = (clamped.floor() as usize).min(max_base);
            (base, clamped - base as f32)
        };
        let (i, fx) = axis(rel.x, self.dims[0]);
        let (j, fy) = axis(rel.y, self.dims[1]);
        let (k, fz) = axis(rel.z, self.dims[2]);
        ([i, j, k], Vec3::new(fx, fy, fz))
    }
}

/// A single triangle of the collision mesh.
struct Tri {
    v: [Vec3; 3],
}

/// Collects every in-range, non-degenerate triangle from the soup.
fn collect_triangles(vertices: &[Vec3], indices: &[[u32; 3]]) -> Vec<Tri> {
    let n = vertices.len() as u32;
    let mut out = Vec::with_capacity(indices.len());
    for tri in indices {
        if tri[0] >= n || tri[1] >= n || tri[2] >= n {
            continue;
        }
        let a = vertices[tri[0] as usize];
        let b = vertices[tri[1] as usize];
        let c = vertices[tri[2] as usize];
        let double_area_sq = (b - a).cross(c - a).length_squared();
        if double_area_sq <= MIN_TRI_AREA2_SQ {
            continue;
        }
        out.push(Tri { v: [a, b, c] });
    }
    out
}

/// Squared distance from `p` to the nearest triangle in `tris`.
fn nearest_distance(p: Vec3, tris: &[Tri]) -> f32 {
    let mut best = f32::INFINITY;
    for t in tris {
        let d = point_triangle_distance_sq(p, t.v[0], t.v[1], t.v[2]);
        if d < best {
            best = d;
        }
    }
    best
}

/// Majority-vote ray-parity inside test: casts three axis rays and counts
/// surface crossings; an odd count means the ray started inside the solid.
fn point_is_inside(p: Vec3, tris: &[Tri]) -> bool {
    // Generic (non-axis-aligned) ray directions. Perfectly axis-aligned rays
    // systematically strike shared triangle edges/diagonals on symmetric meshes
    // (e.g. a quad split along its diagonal), which parity cannot classify;
    // slightly skewed directions avoid that degeneracy, and the majority vote
    // absorbs the rare remaining grazing case.
    let dirs = [
        Vec3::new(1.0, 0.1100, 0.0700).normalize(),
        Vec3::new(0.0900, 1.0, 0.1300).normalize(),
        Vec3::new(0.0500, 0.0800, 1.0).normalize(),
    ];
    let mut inside_votes = 0u32;
    for dir in dirs {
        let mut crossings = 0u32;
        for t in tris {
            if ray_triangle_forward_hit(p, dir, t) {
                crossings += 1;
            }
        }
        if crossings & 1 == 1 {
            inside_votes += 1;
        }
    }
    inside_votes >= 2
}

/// Moller-Trumbore ray/triangle test counting only strictly-forward hits
/// (`t > eps`). Grazing hits (parallel ray or barycentric on the boundary) are
/// rejected so parity stays stable across the three axis rays.
fn ray_triangle_forward_hit(origin: Vec3, dir: Vec3, tri: &Tri) -> bool {
    const EPS: f32 = 1e-7;
    let e1 = tri.v[1] - tri.v[0];
    let e2 = tri.v[2] - tri.v[0];
    let pvec = dir.cross(e2);
    let det = e1.dot(pvec);
    if det.abs() < EPS {
        return false;
    }
    let inv_det = 1.0 / det;
    let tvec = origin - tri.v[0];
    let u = tvec.dot(pvec) * inv_det;
    if u <= EPS || u >= 1.0 - EPS {
        return false;
    }
    let qvec = tvec.cross(e1);
    let v = dir.dot(qvec) * inv_det;
    if v <= EPS || u + v >= 1.0 - EPS {
        return false;
    }
    let t = e2.dot(qvec) * inv_det;
    t > EPS
}

/// Shortest squared distance from point `p` to triangle `abc`.
///
/// Classic Voronoi-region closest-point algorithm (Ericson, *Real-Time
/// Collision Detection*): classify `p` against the triangle's vertex, edge and
/// face regions and return the squared distance to the closest feature.
pub(crate) fn point_triangle_distance_sq(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> f32 {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return ap.length_squared(); // vertex region A
    }

    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return bp.length_squared(); // vertex region B
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let w = d1 / (d1 - d3);
        let proj = a + ab * w;
        return (p - proj).length_squared(); // edge region AB
    }

    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return cp.length_squared(); // vertex region C
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        let proj = a + ac * w;
        return (p - proj).length_squared(); // edge region AC
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        let proj = b + (c - b) * w;
        return (p - proj).length_squared(); // edge region BC
    }

    // Face region: project onto the triangle plane via barycentric coords.
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    let proj = a + ab * v + ac * w;
    (p - proj).length_squared()
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Axis-aligned cube centred at the origin with the given half-extent,
    /// returned as outward-wound triangles.
    fn cube_mesh(half: f32) -> (Vec<Vec3>, Vec<[u32; 3]>) {
        let h = half;
        let verts = vec![
            Vec3::new(-h, -h, -h),
            Vec3::new(h, -h, -h),
            Vec3::new(h, h, -h),
            Vec3::new(-h, h, -h),
            Vec3::new(-h, -h, h),
            Vec3::new(h, -h, h),
            Vec3::new(h, h, h),
            Vec3::new(-h, h, h),
        ];
        // Outward-facing winding (CCW seen from outside).
        let idx = vec![
            [0, 2, 1],
            [0, 3, 2], // -Z
            [4, 5, 6],
            [4, 6, 7], // +Z
            [0, 1, 5],
            [0, 5, 4], // -Y
            [3, 7, 6],
            [3, 6, 2], // +Y
            [0, 4, 7],
            [0, 7, 3], // -X
            [1, 2, 6],
            [1, 6, 5], // +X
        ];
        (verts, idx)
    }

    /// Regular octahedron with the given radius (vertices on the axes).
    fn octahedron_mesh(r: f32) -> (Vec<Vec3>, Vec<[u32; 3]>) {
        let verts = vec![
            Vec3::new(r, 0.0, 0.0),
            Vec3::new(-r, 0.0, 0.0),
            Vec3::new(0.0, r, 0.0),
            Vec3::new(0.0, -r, 0.0),
            Vec3::new(0.0, 0.0, r),
            Vec3::new(0.0, 0.0, -r),
        ];
        // Eight faces, outward wound.
        let idx = vec![
            [0, 2, 4],
            [2, 1, 4],
            [1, 3, 4],
            [3, 0, 4],
            [2, 0, 5],
            [1, 2, 5],
            [3, 1, 5],
            [0, 3, 5],
        ];
        (verts, idx)
    }

    #[test]
    fn rejects_bad_parameters_and_empty_mesh() {
        let (v, i) = cube_mesh(1.0);
        let good = SdfBuildParams {
            cell_size: 0.5,
            padding: 0.5,
        };
        assert!(MeshSdf::from_mesh(&[], &[], good).is_none());
        assert!(MeshSdf::from_mesh(&v, &[], good).is_none());
        assert!(MeshSdf::from_mesh(
            &v,
            &i,
            SdfBuildParams {
                cell_size: 0.0,
                padding: 0.5
            }
        )
        .is_none());
        assert!(MeshSdf::from_mesh(
            &v,
            &i,
            SdfBuildParams {
                cell_size: f32::NAN,
                padding: 0.5
            }
        )
        .is_none());
    }

    #[test]
    fn cube_sign_is_negative_inside_positive_outside() {
        let (v, i) = cube_mesh(1.0);
        let sdf = MeshSdf::from_mesh(
            &v,
            &i,
            SdfBuildParams {
                cell_size: 0.2,
                padding: 0.6,
            },
        )
        .expect("builds");

        // Centre is deep inside.
        assert!(sdf.sample(Vec3::ZERO) < 0.0);
        // A point well outside is positive.
        assert!(sdf.sample(Vec3::new(1.5, 0.0, 0.0)) > 0.0);
        assert!(sdf.sample(Vec3::new(0.0, 1.4, 0.3)) > 0.0);
        // A point just inside a face is negative.
        assert!(sdf.sample(Vec3::new(0.0, 0.0, 0.85)) < 0.0);
    }

    #[test]
    fn cube_centre_distance_matches_half_extent() {
        let (v, i) = cube_mesh(1.0);
        let sdf = MeshSdf::from_mesh(
            &v,
            &i,
            SdfBuildParams {
                cell_size: 0.1,
                padding: 0.5,
            },
        )
        .expect("builds");
        // Interior distance at the centre of a unit cube is -1 (distance to the
        // nearest face). Trilinear sampling at the exact centre is within a
        // cell of the analytic value.
        let d = sdf.sample(Vec3::ZERO);
        assert!((d + 1.0).abs() < 0.15, "centre distance {d}");
    }

    #[test]
    fn cube_exterior_distance_matches_analytic() {
        let (v, i) = cube_mesh(1.0);
        let sdf = MeshSdf::from_mesh(
            &v,
            &i,
            SdfBuildParams {
                cell_size: 0.1,
                padding: 0.8,
            },
        )
        .expect("builds");
        // 0.5 outside the +X face along the axis: exact distance 0.5.
        let p = Vec3::new(1.5, 0.0, 0.0);
        let d = sdf.sample(p);
        assert!((d - 0.5).abs() < 0.08, "exterior distance {d}");
    }

    #[test]
    fn gradient_points_outward_near_face() {
        let (v, i) = cube_mesh(1.0);
        let sdf = MeshSdf::from_mesh(
            &v,
            &i,
            SdfBuildParams {
                cell_size: 0.1,
                padding: 0.6,
            },
        )
        .expect("builds");
        // Just outside the +X face the gradient should point roughly +X.
        let g = sdf.gradient(Vec3::new(1.2, 0.0, 0.0));
        assert!(g.x > 0.8, "gradient {g:?}");
        assert!(g.y.abs() < 0.3 && g.z.abs() < 0.3, "gradient {g:?}");
        // The gradient is a unit vector.
        assert!((g.length() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn project_moves_a_penetrating_point_towards_the_surface() {
        let (v, i) = cube_mesh(1.0);
        let sdf = MeshSdf::from_mesh(
            &v,
            &i,
            SdfBuildParams {
                cell_size: 0.1,
                padding: 0.6,
            },
        )
        .expect("builds");
        // A point slightly inside the +X face, penetrating by ~0.2.
        let p = Vec3::new(0.8, 0.0, 0.0);
        let before = sdf.sample(p).abs();
        let projected = sdf.project_to_surface(p);
        let after = sdf.sample(projected).abs();
        assert!(after < before, "after {after} before {before}");
        assert!(after < 0.1, "residual {after}");
    }

    #[test]
    fn octahedron_sign_and_rough_distance() {
        let (v, i) = octahedron_mesh(1.0);
        let sdf = MeshSdf::from_mesh(
            &v,
            &i,
            SdfBuildParams {
                cell_size: 0.15,
                padding: 0.6,
            },
        )
        .expect("builds");
        assert!(sdf.sample(Vec3::ZERO) < 0.0, "centre inside");
        assert!(sdf.sample(Vec3::new(2.0, 0.0, 0.0)) > 0.0, "far outside");
        // The plane x+y+z=1 (one octahedron face) has distance 1/sqrt(3) from
        // the origin; the field at the centre should be about -that.
        let analytic = -(1.0_f32 / (3.0_f32).sqrt());
        let d = sdf.sample(Vec3::ZERO);
        assert!((d - analytic).abs() < 0.2, "centre {d} vs {analytic}");
    }

    #[test]
    fn grid_bounds_cover_the_padded_mesh() {
        let (v, i) = cube_mesh(1.0);
        let params = SdfBuildParams {
            cell_size: 0.25,
            padding: 0.5,
        };
        let sdf = MeshSdf::from_mesh(&v, &i, params).expect("builds");
        let (lo, hi) = sdf.local_aabb();
        assert!(lo.x <= -1.5 + 1e-5 && lo.y <= -1.5 + 1e-5 && lo.z <= -1.5 + 1e-5);
        assert!(hi.x >= 1.5 - 1e-5 && hi.y >= 1.5 - 1e-5 && hi.z >= 1.5 - 1e-5);
        let [nx, ny, nz] = sdf.dims();
        assert!(nx >= 2 && ny >= 2 && nz >= 2);
        assert!(sdf.node_distance(0, 0, 0).is_some());
        assert!(sdf.node_distance(nx, 0, 0).is_none());
    }

    #[test]
    fn from_resolution_sets_reasonable_cell_size() {
        let p = SdfBuildParams::from_resolution(4.0, 16);
        assert!((p.cell_size - 0.25).abs() < 1e-6);
        assert!(p.padding > 0.0);
    }

    #[test]
    fn point_triangle_distance_matches_known_cases() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(1.0, 0.0, 0.0);
        let c = Vec3::new(0.0, 1.0, 0.0);
        // Directly above the interior: distance equals the height.
        let d = point_triangle_distance_sq(Vec3::new(0.25, 0.25, 2.0), a, b, c);
        assert!((d - 4.0).abs() < 1e-5);
        // Nearest to vertex A.
        let d = point_triangle_distance_sq(Vec3::new(-1.0, -1.0, 0.0), a, b, c);
        assert!((d - 2.0).abs() < 1e-5);
        // Nearest to edge AB midpoint, offset in -Y.
        let d = point_triangle_distance_sq(Vec3::new(0.5, -1.0, 0.0), a, b, c);
        assert!((d - 1.0).abs() < 1e-5);
    }
}
