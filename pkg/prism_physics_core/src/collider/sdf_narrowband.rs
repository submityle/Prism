//! Sparse narrow-band signed distance field for triangle-mesh colliders.
//!
//! A dense [`MeshSdf`](super::sdf::MeshSdf) stores a signed distance at every
//! node of a regular grid, so its memory grows with the *volume* of the padded
//! bounds -- cubic in resolution. Collision queries, however, only ever care
//! about the thin shell of space hugging the surface: once a probe is more than
//! a cell or two away the exact triangle distance is cheap and the interpolated
//! grid value is never consulted. Production engines exploit this by keeping a
//! *narrow band* level set (UE Chaos narrow-band level sets, `OpenVDB`,
//! `VDB`-style sparse grids): only nodes within a fixed distance of the surface
//! are materialised, cutting storage from `O(res^3)` toward `O(res^2)`.
//!
//! [`NarrowBandSdf`] is that representation. During construction every grid node
//! is classified with a [`MeshBvh`] closest-point query; nodes whose unsigned
//! distance falls inside the band are stored in a hash map keyed by node index,
//! and the rest are dropped. The embedded `BVH` is retained so queries outside
//! the band stay exact instead of degrading to a clamped band value:
//! [`NarrowBandSdf::sample`] trilinearly interpolates the stored band when all
//! eight surrounding nodes are present and otherwise falls back to the exact
//! signed triangle distance.
//!
//! The sign convention and ray-parity inside test mirror the dense field;
//! everything here is standard sparse-level-set and closest-point practice and
//! is not derived from Unreal Engine source.

use super::mesh_bvh::MeshBvh;
use glam::Vec3;
use std::collections::HashMap;

/// Minimum number of nodes along each grid axis.
const MIN_NODES: usize = 2;
/// Hard cap on the dense node scan, so an accidental tiny `cell_size` cannot
/// start an unbounded build sweep.
const MAX_NODES_TOTAL: usize = 1 << 24; // ~16.7M nodes.

/// Parameters controlling how a [`NarrowBandSdf`] is discretised.
#[derive(Clone, Copy, Debug)]
pub struct NarrowBandParams {
    /// Uniform spacing between neighbouring grid nodes, in mesh-local units.
    pub cell_size: f32,
    /// Extra margin added around the mesh axis-aligned bounds.
    pub padding: f32,
    /// Half-width of the stored band, in cell units: a node is materialised
    /// when its unsigned distance to the surface is at most
    /// `band_cells * cell_size`. Values below `1.0` are raised to `1.0` so a
    /// cell straddling the surface always has its corners stored.
    pub band_cells: f32,
}

impl NarrowBandParams {
    /// Builds parameters from a target resolution along the mesh's longest
    /// axis and a band half-width in cells. `padding` defaults to the band
    /// width so the stored shell is never clipped by the bounds.
    #[must_use]
    pub fn from_resolution(longest_extent: f32, resolution: u32, band_cells: f32) -> Self {
        let res = resolution.max(1) as f32;
        let cell_size = (longest_extent / res).max(f32::MIN_POSITIVE);
        let band = band_cells.max(1.0);
        Self {
            cell_size,
            padding: cell_size * band,
            band_cells: band,
        }
    }
}

/// A sparse, narrow-band signed distance field for a triangle mesh.
///
/// Node `(i, j, k)` sits at `origin + cell_size * (i, j, k)`. Only nodes within
/// the band are stored; see the module docs for the rationale.
#[derive(Clone, Debug)]
pub struct NarrowBandSdf {
    origin: Vec3,
    cell_size: f32,
    dims: [usize; 3],
    band: f32,
    cells: HashMap<(u32, u32, u32), f32>,
    bvh: MeshBvh,
}

impl NarrowBandSdf {
    /// Builds a narrow-band signed distance field for the closed triangle mesh
    /// described by `vertices` and triangle `indices`.
    ///
    /// Returns `None` when the mesh is empty, every triangle is degenerate, the
    /// parameters are non-finite/non-positive, or the requested grid would
    /// exceed [`MAX_NODES_TOTAL`] nodes.
    #[must_use]
    pub fn from_mesh(
        vertices: &[Vec3],
        indices: &[[u32; 3]],
        params: NarrowBandParams,
    ) -> Option<Self> {
        if !params.cell_size.is_finite()
            || params.cell_size <= 0.0
            || !params.padding.is_finite()
            || params.padding < 0.0
            || !params.band_cells.is_finite()
            || params.band_cells < 1.0
        {
            return None;
        }

        let bvh = MeshBvh::build(vertices, indices)?;

        let (mut min, mut max) = bvh.local_aabb();
        min -= Vec3::splat(params.padding);
        max += Vec3::splat(params.padding);

        let span = max - min;
        let cell = params.cell_size;
        let node_count = |extent: f32| -> usize {
            let cells = (extent / cell).ceil() as i64;
            (cells.max(1) as usize + 1).max(MIN_NODES)
        };
        let dims = [node_count(span.x), node_count(span.y), node_count(span.z)];
        let total = dims[0].checked_mul(dims[1])?.checked_mul(dims[2])?;
        if total == 0 || total > MAX_NODES_TOTAL {
            return None;
        }

        let band = params.band_cells * cell;
        let mut cells = HashMap::new();
        for k in 0..dims[2] {
            for j in 0..dims[1] {
                for i in 0..dims[0] {
                    let p = min + Vec3::new(i as f32, j as f32, k as f32) * cell;
                    let unsigned = bvh.closest_point(p).map_or(0.0, |c| c.distance_sq.sqrt());
                    if unsigned <= band {
                        let sign = if point_is_inside(&bvh, p) { -1.0 } else { 1.0 };
                        cells.insert((i as u32, j as u32, k as u32), sign * unsigned);
                    }
                }
            }
        }

        Some(Self {
            origin: min,
            cell_size: cell,
            dims,
            band,
            cells,
            bvh,
        })
    }

    /// Grid node counts along each axis.
    #[must_use]
    pub fn dims(&self) -> [usize; 3] {
        self.dims
    }

    /// Uniform spacing between neighbouring grid nodes.
    #[must_use]
    pub fn cell_size(&self) -> f32 {
        self.cell_size
    }

    /// The minimum (origin) corner of the grid in mesh-local space.
    #[must_use]
    pub fn origin(&self) -> Vec3 {
        self.origin
    }

    /// Band half-width in mesh-local units: nodes with `|distance|` greater
    /// than this were not stored.
    #[must_use]
    pub fn band(&self) -> f32 {
        self.band
    }

    /// Number of materialised (in-band) nodes.
    #[must_use]
    pub fn stored_nodes(&self) -> usize {
        self.cells.len()
    }

    /// Number of nodes a dense grid of the same dimensions would allocate.
    #[must_use]
    pub fn dense_nodes(&self) -> usize {
        self.dims[0] * self.dims[1] * self.dims[2]
    }

    /// Fraction of the dense grid actually materialised, in `[0, 1]`. Smaller
    /// is a bigger memory win from the narrow band.
    #[must_use]
    pub fn occupancy(&self) -> f32 {
        let dense = self.dense_nodes();
        if dense == 0 {
            0.0
        } else {
            self.stored_nodes() as f32 / dense as f32
        }
    }

    /// Signed distance stored at node `(i, j, k)`, or `None` when the node is
    /// outside the band or outside the grid.
    #[must_use]
    pub fn node_distance(&self, i: usize, j: usize, k: usize) -> Option<f32> {
        if i >= self.dims[0] || j >= self.dims[1] || k >= self.dims[2] {
            return None;
        }
        self.cells.get(&(i as u32, j as u32, k as u32)).copied()
    }

    /// Local-space axis-aligned bounds of the grid.
    #[must_use]
    pub fn local_aabb(&self) -> (Vec3, Vec3) {
        let max = self.origin
            + Vec3::new(
                (self.dims[0] - 1) as f32,
                (self.dims[1] - 1) as f32,
                (self.dims[2] - 1) as f32,
            ) * self.cell_size;
        (self.origin, max)
    }

    /// Signed distance at an arbitrary local-space `point` (negative inside).
    ///
    /// Trilinearly interpolates the stored band when the point lies inside the
    /// grid and all eight surrounding nodes are present; otherwise returns the
    /// exact signed triangle distance from the embedded `BVH`.
    #[must_use]
    pub fn sample(&self, point: Vec3) -> f32 {
        if let Some(value) = self.sample_band(point) {
            value
        } else {
            self.exact_signed_distance(point)
        }
    }

    /// Numerical gradient of the field at `point`, pointing away from the
    /// surface. Falls back to a unit `+X` when the surface is exactly hit.
    #[must_use]
    pub fn gradient(&self, point: Vec3) -> Vec3 {
        let h = self.cell_size * 0.5;
        let dx = self.sample(point + Vec3::X * h) - self.sample(point - Vec3::X * h);
        let dy = self.sample(point + Vec3::Y * h) - self.sample(point - Vec3::Y * h);
        let dz = self.sample(point + Vec3::Z * h) - self.sample(point - Vec3::Z * h);
        let g = Vec3::new(dx, dy, dz);
        g.try_normalize().unwrap_or(Vec3::X)
    }

    /// Projects `point` onto the mesh surface using the exact closest point on
    /// the triangle set.
    #[must_use]
    pub fn project_to_surface(&self, point: Vec3) -> Vec3 {
        self.bvh.closest_point(point).map_or(point, |c| c.point)
    }

    /// Exact signed distance via a closest-point query plus a ray-parity sign.
    fn exact_signed_distance(&self, point: Vec3) -> f32 {
        let unsigned = self
            .bvh
            .closest_point(point)
            .map_or(0.0, |c| c.distance_sq.sqrt());
        let sign = if point_is_inside(&self.bvh, point) {
            -1.0
        } else {
            1.0
        };
        sign * unsigned
    }

    /// Trilinear interpolation of the stored band, or `None` when the point is
    /// outside the grid or any surrounding node is missing.
    fn sample_band(&self, point: Vec3) -> Option<f32> {
        let local = (point - self.origin) / self.cell_size;
        let max_cell = [self.dims[0] - 1, self.dims[1] - 1, self.dims[2] - 1];
        if local.x < 0.0
            || local.y < 0.0
            || local.z < 0.0
            || local.x > max_cell[0] as f32
            || local.y > max_cell[1] as f32
            || local.z > max_cell[2] as f32
        {
            return None;
        }

        let i0 = (local.x.floor() as usize).min(max_cell[0].saturating_sub(1));
        let j0 = (local.y.floor() as usize).min(max_cell[1].saturating_sub(1));
        let k0 = (local.z.floor() as usize).min(max_cell[2].saturating_sub(1));
        let tx = local.x - i0 as f32;
        let ty = local.y - j0 as f32;
        let tz = local.z - k0 as f32;

        let c000 = self.node_distance(i0, j0, k0)?;
        let c100 = self.node_distance(i0 + 1, j0, k0)?;
        let c010 = self.node_distance(i0, j0 + 1, k0)?;
        let c110 = self.node_distance(i0 + 1, j0 + 1, k0)?;
        let c001 = self.node_distance(i0, j0, k0 + 1)?;
        let c101 = self.node_distance(i0 + 1, j0, k0 + 1)?;
        let c011 = self.node_distance(i0, j0 + 1, k0 + 1)?;
        let c111 = self.node_distance(i0 + 1, j0 + 1, k0 + 1)?;

        let c00 = lerp(c000, c100, tx);
        let c10 = lerp(c010, c110, tx);
        let c01 = lerp(c001, c101, tx);
        let c11 = lerp(c011, c111, tx);
        let c0 = lerp(c00, c10, ty);
        let c1 = lerp(c01, c11, ty);
        Some(lerp(c0, c1, tz))
    }
}

/// Ray-parity inside test: cast three skew rays and take the majority of odd
/// crossing counts. Standard for watertight meshes and robust to a ray that
/// grazes an edge.
fn point_is_inside(bvh: &MeshBvh, p: Vec3) -> bool {
    let dirs = [
        Vec3::new(1.0, 0.1100, 0.0700).normalize(),
        Vec3::new(0.0900, 1.0, 0.1300).normalize(),
        Vec3::new(0.0500, 0.0800, 1.0).normalize(),
    ];
    let mut inside_votes = 0u32;
    for dir in dirs {
        if bvh.count_forward_crossings(p, dir) & 1 == 1 {
            inside_votes += 1;
        }
    }
    inside_votes >= 2
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collider::sdf::{MeshSdf, SdfBuildParams};
    use std::collections::HashMap as StdHashMap;

    /// Builds a closed, welded icosahedron subdivided `levels` times: a
    /// watertight 2-manifold unit sphere.
    fn icosphere(levels: u32) -> (Vec<Vec3>, Vec<[u32; 3]>) {
        let t = (1.0 + 5.0_f32.sqrt()) * 0.5;
        let mut verts: Vec<Vec3> = vec![
            Vec3::new(-1.0, t, 0.0),
            Vec3::new(1.0, t, 0.0),
            Vec3::new(-1.0, -t, 0.0),
            Vec3::new(1.0, -t, 0.0),
            Vec3::new(0.0, -1.0, t),
            Vec3::new(0.0, 1.0, t),
            Vec3::new(0.0, -1.0, -t),
            Vec3::new(0.0, 1.0, -t),
            Vec3::new(t, 0.0, -1.0),
            Vec3::new(t, 0.0, 1.0),
            Vec3::new(-t, 0.0, -1.0),
            Vec3::new(-t, 0.0, 1.0),
        ];
        for v in &mut verts {
            *v = v.normalize();
        }
        let mut faces: Vec<[u32; 3]> = vec![
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ];
        for _ in 0..levels {
            let mut cache: StdHashMap<(u32, u32), u32> = StdHashMap::new();
            let mut next: Vec<[u32; 3]> = Vec::with_capacity(faces.len() * 4);
            let mut midpoint = |a: u32, b: u32, verts: &mut Vec<Vec3>| -> u32 {
                let key = if a < b { (a, b) } else { (b, a) };
                if let Some(&m) = cache.get(&key) {
                    return m;
                }
                let m = verts.len() as u32;
                let p = ((verts[a as usize] + verts[b as usize]) * 0.5).normalize();
                verts.push(p);
                cache.insert(key, m);
                m
            };
            for f in &faces {
                let a = midpoint(f[0], f[1], &mut verts);
                let b = midpoint(f[1], f[2], &mut verts);
                let c = midpoint(f[2], f[0], &mut verts);
                next.push([f[0], a, c]);
                next.push([f[1], b, a]);
                next.push([f[2], c, b]);
                next.push([a, b, c]);
            }
            faces = next;
        }
        (verts, faces)
    }

    fn params(cell: f32, band_cells: f32) -> NarrowBandParams {
        NarrowBandParams {
            cell_size: cell,
            padding: cell * 2.0,
            band_cells,
        }
    }

    #[test]
    fn rejects_bad_params() {
        let (v, f) = icosphere(1);
        assert!(NarrowBandSdf::from_mesh(&v, &f, params(0.0, 2.0)).is_none());
        assert!(NarrowBandSdf::from_mesh(&v, &f, params(-0.1, 2.0)).is_none());
        assert!(NarrowBandSdf::from_mesh(&v, &f, params(0.1, 0.5)).is_none());
        assert!(NarrowBandSdf::from_mesh(&v, &f, params(f32::NAN, 2.0)).is_none());
        assert!(NarrowBandSdf::from_mesh(&[], &[], params(0.1, 2.0)).is_none());
    }

    #[test]
    fn band_is_sparser_than_dense_grid() {
        let (v, f) = icosphere(2);
        let nb = NarrowBandSdf::from_mesh(&v, &f, params(0.1, 2.0)).expect("sdf");
        assert!(nb.stored_nodes() > 0);
        assert!(nb.stored_nodes() < nb.dense_nodes());
        // A thin band around a 2-unit sphere must leave the interior/exterior
        // bulk unstored.
        assert!(
            nb.occupancy() < 0.8,
            "occupancy {} not sparse",
            nb.occupancy()
        );
    }

    #[test]
    fn surface_vertices_sample_near_zero() {
        let (v, f) = icosphere(2);
        let nb = NarrowBandSdf::from_mesh(&v, &f, params(0.1, 2.0)).expect("sdf");
        for &p in v.iter().take(12) {
            assert!(
                nb.sample(p).abs() < 0.1,
                "surface sample {} too large",
                nb.sample(p)
            );
        }
    }

    #[test]
    fn inside_is_negative_outside_is_positive() {
        let (v, f) = icosphere(2);
        let nb = NarrowBandSdf::from_mesh(&v, &f, params(0.1, 2.0)).expect("sdf");
        // Centre is deep inside (beyond the band): exact fallback, ~ -radius.
        assert!(nb.sample(Vec3::ZERO) < -0.5);
        // A far exterior point: exact fallback, ~ distance - radius.
        let outside = nb.sample(Vec3::new(3.0, 0.0, 0.0));
        assert!(outside > 1.5, "outside sample {outside}");
    }

    #[test]
    fn band_matches_dense_field_near_surface() {
        let (v, f) = icosphere(3);
        let cell = 0.08;
        let nb = NarrowBandSdf::from_mesh(&v, &f, params(cell, 2.0)).expect("nb");
        let dense = MeshSdf::from_mesh(
            &v,
            &f,
            SdfBuildParams {
                cell_size: cell,
                padding: cell * 2.0,
            },
        )
        .expect("dense");

        // Probe points just outside the surface, inside the band, where both
        // fields trilinearly interpolate the identical node distances.
        for scale in [1.03_f32, 1.05, 1.07] {
            for dir in [
                Vec3::X,
                Vec3::Y,
                Vec3::Z,
                Vec3::new(1.0, 1.0, 1.0).normalize(),
            ] {
                let p = dir * scale;
                let a = nb.sample(p);
                let b = dense.sample(p);
                assert!(
                    (a - b).abs() < 1.0e-3,
                    "narrow band {a} vs dense {b} at {p:?}"
                );
            }
        }
    }

    #[test]
    fn gradient_points_outward() {
        let (v, f) = icosphere(3);
        let nb = NarrowBandSdf::from_mesh(&v, &f, params(0.08, 2.0)).expect("sdf");
        for dir in [
            Vec3::X,
            Vec3::Y,
            Vec3::Z,
            Vec3::new(-1.0, 1.0, -1.0).normalize(),
        ] {
            let p = dir * 1.05;
            let g = nb.gradient(p);
            assert!(g.dot(dir) > 0.5, "gradient {g:?} not outward along {dir:?}");
        }
    }

    #[test]
    fn project_lands_on_unit_sphere() {
        let (v, f) = icosphere(3);
        let nb = NarrowBandSdf::from_mesh(&v, &f, params(0.08, 2.0)).expect("sdf");
        for dir in [Vec3::X, Vec3::Y, Vec3::new(1.0, 2.0, -1.0).normalize()] {
            let p = dir * 2.0;
            let s = nb.project_to_surface(p);
            assert!((s.length() - 1.0).abs() < 0.1, "projected {s:?} off sphere");
        }
    }

    #[test]
    fn node_distance_only_defined_in_band() {
        let (v, f) = icosphere(2);
        let nb = NarrowBandSdf::from_mesh(&v, &f, params(0.1, 2.0)).expect("sdf");
        let dims = nb.dims();
        // The grid corner is a padded exterior node, far outside the band.
        assert!(nb.node_distance(0, 0, 0).is_none());
        // Out-of-range indices are rejected.
        assert!(nb.node_distance(dims[0], 0, 0).is_none());
        // At least some interior nodes are stored.
        let mut stored = 0;
        for k in 0..dims[2] {
            for j in 0..dims[1] {
                for i in 0..dims[0] {
                    if nb.node_distance(i, j, k).is_some() {
                        stored += 1;
                    }
                }
            }
        }
        assert_eq!(stored, nb.stored_nodes());
        assert!(stored > 0);
    }
}
