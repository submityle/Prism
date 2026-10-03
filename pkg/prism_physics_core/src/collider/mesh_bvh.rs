//! Static bounding-volume hierarchy over a triangle mesh.
//!
//! A triangle-mesh BVH is the midphase every AAA engine builds for static
//! collision geometry (`PhysX` `PxTriangleMesh`'s `RTree`, `Jolt`'s
//! `MeshShape` tree, UE `Chaos`'s triangle BVH). It turns the otherwise linear
//! "which triangle does this ray/box/point touch?" scan into a logarithmic
//! descent, which is what lets scene queries against million-triangle levels
//! stay interactive.
//!
//! This module builds a compact, pointer-free BVH with a binned
//! surface-area-heuristic (SAH) splitter and exposes the three queries the rest
//! of the engine needs:
//!
//! - [`MeshBvh::ray_cast`] -- nearest ray/mesh intersection (scene queries,
//!   SDF sign rays, projectile sweeps),
//! - [`MeshBvh::closest_point`] -- the nearest point on the mesh to a query
//!   point, with branch-and-bound pruning (SDF generation, snapping), and
//! - [`MeshBvh::overlapping_triangles`] -- every triangle whose bounds overlap
//!   a query box (narrow-phase candidate gathering).
//!
//! The SAH binning, slab ray test and branch-and-bound traversal are standard
//! textbook techniques (Wald, *On fast Construction of SAH-based Bounding
//! Volume Hierarchies*; Ericson, *Real-Time Collision Detection*); nothing here
//! is derived from Unreal Engine source.

use super::sdf::point_triangle_distance_sq;
use glam::Vec3;

/// Maximum triangles stored in a leaf before the builder stops splitting.
const MAX_LEAF_TRIS: usize = 4;
/// Number of SAH bins evaluated per split.
const SAH_BINS: usize = 12;
/// Degenerate-triangle area threshold (squared double-area).
const MIN_TRI_AREA2_SQ: f32 = 1e-20;

/// A single BVH node in the flat node array.
///
/// A node is a leaf when `tri_count > 0`; then `first` indexes the first of its
/// triangles in the BVH's reordered triangle array. Otherwise it is an interior
/// node whose children are at `first` and `first + 1`.
#[derive(Clone, Copy, Debug)]
struct BvhNode {
    bmin: Vec3,
    bmax: Vec3,
    first: u32,
    tri_count: u32,
}

/// A ray/triangle intersection reported by [`MeshBvh::ray_cast`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshRayHit {
    /// Ray parameter `t` at the hit (`point = origin + dir * t`).
    pub time: f32,
    /// World/local-space hit position.
    pub point: Vec3,
    /// Unit geometric normal of the struck triangle (unoriented w.r.t. the ray).
    pub normal: Vec3,
    /// Original triangle index (into the mesh passed to the builder).
    pub triangle: u32,
}

/// The nearest point on the mesh to a query, from [`MeshBvh::closest_point`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshClosestPoint {
    /// Closest surface point.
    pub point: Vec3,
    /// Squared distance from the query to [`MeshClosestPoint::point`].
    pub distance_sq: f32,
    /// Original triangle index owning the closest point.
    pub triangle: u32,
}

/// A static BVH over a triangle mesh.
#[derive(Clone, Debug)]
pub struct MeshBvh {
    nodes: Vec<BvhNode>,
    /// Triangle vertex triples, reordered into BVH leaf order.
    tris: Vec<[Vec3; 3]>,
    /// Original triangle index for each entry in [`MeshBvh::tris`].
    source: Vec<u32>,
}

/// Scratch triangle record used only during construction.
struct BuildTri {
    verts: [Vec3; 3],
    centroid: Vec3,
    bmin: Vec3,
    bmax: Vec3,
    source: u32,
}

impl MeshBvh {
    /// Builds a BVH for the triangles described by `vertices` and `indices`.
    ///
    /// Out-of-range and degenerate (zero-area) triangles are skipped. Returns
    /// `None` when no valid triangle remains.
    #[must_use]
    pub fn build(vertices: &[Vec3], indices: &[[u32; 3]]) -> Option<Self> {
        let n = vertices.len() as u32;
        let mut build: Vec<BuildTri> = Vec::with_capacity(indices.len());
        for (ti, tri) in indices.iter().enumerate() {
            if tri[0] >= n || tri[1] >= n || tri[2] >= n {
                continue;
            }
            let a = vertices[tri[0] as usize];
            let b = vertices[tri[1] as usize];
            let c = vertices[tri[2] as usize];
            if (b - a).cross(c - a).length_squared() <= MIN_TRI_AREA2_SQ {
                continue;
            }
            let bmin = a.min(b).min(c);
            let bmax = a.max(b).max(c);
            build.push(BuildTri {
                verts: [a, b, c],
                centroid: (a + b + c) / 3.0,
                bmin,
                bmax,
                source: ti as u32,
            });
        }
        if build.is_empty() {
            return None;
        }

        let count = build.len();
        let mut nodes: Vec<BvhNode> = Vec::with_capacity(2 * count);
        // Root node covers [0, count).
        nodes.push(BvhNode {
            bmin: Vec3::ZERO,
            bmax: Vec3::ZERO,
            first: 0,
            tri_count: count as u32,
        });
        subdivide(&mut nodes, 0, &mut build, 0, count);

        let mut tris = Vec::with_capacity(count);
        let mut source = Vec::with_capacity(count);
        for t in &build {
            tris.push(t.verts);
            source.push(t.source);
        }

        Some(Self {
            nodes,
            tris,
            source,
        })
    }

    /// Number of triangles stored in the hierarchy.
    #[must_use]
    pub fn triangle_count(&self) -> usize {
        self.tris.len()
    }

    /// Number of nodes (interior plus leaf) in the hierarchy.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Root-node axis-aligned bounds as `(min, max)`.
    #[must_use]
    pub fn local_aabb(&self) -> (Vec3, Vec3) {
        let r = &self.nodes[0];
        (r.bmin, r.bmax)
    }

    /// Casts a ray and returns the nearest intersection with `t` in
    /// `(0, max_time]`, or `None`.
    #[must_use]
    pub fn ray_cast(&self, origin: Vec3, dir: Vec3, max_time: f32) -> Option<MeshRayHit> {
        if !max_time.is_finite() || max_time <= 0.0 {
            return None;
        }
        let inv_dir = Vec3::new(safe_inv(dir.x), safe_inv(dir.y), safe_inv(dir.z));

        let mut best_t = max_time;
        let mut best: Option<MeshRayHit> = None;
        let mut stack = [0u32; 64];
        let mut sp = 0usize;
        stack[sp] = 0;
        sp += 1;

        while sp > 0 {
            sp -= 1;
            let node = &self.nodes[stack[sp] as usize];
            if !slab_hit(node.bmin, node.bmax, origin, inv_dir, best_t) {
                continue;
            }
            if node.tri_count > 0 {
                let start = node.first as usize;
                for local in 0..node.tri_count as usize {
                    let idx = start + local;
                    let [a, b, c] = self.tris[idx];
                    if let Some(t) = ray_triangle(origin, dir, a, b, c)
                        && t > 1e-7
                        && t < best_t
                    {
                        best_t = t;
                        let normal = (b - a).cross(c - a).normalize_or_zero();
                        best = Some(MeshRayHit {
                            time: t,
                            point: origin + dir * t,
                            normal,
                            triangle: self.source[idx],
                        });
                    }
                }
            } else if sp + 2 <= stack.len() {
                stack[sp] = node.first;
                sp += 1;
                stack[sp] = node.first + 1;
                sp += 1;
            }
        }
        best
    }

    /// Returns the closest point on the mesh to `query`, or `None` for an empty
    /// hierarchy (which [`MeshBvh::build`] never produces).
    #[must_use]
    pub fn closest_point(&self, query: Vec3) -> Option<MeshClosestPoint> {
        let mut best = f32::INFINITY;
        let mut result: Option<MeshClosestPoint> = None;
        let mut stack = [0u32; 64];
        let mut sp = 0usize;
        stack[sp] = 0;
        sp += 1;

        while sp > 0 {
            sp -= 1;
            let node = &self.nodes[stack[sp] as usize];
            if aabb_distance_sq(node.bmin, node.bmax, query) >= best {
                continue; // whole subtree is further than the current best.
            }
            if node.tri_count > 0 {
                let start = node.first as usize;
                for local in 0..node.tri_count as usize {
                    let idx = start + local;
                    let [a, b, c] = self.tris[idx];
                    let d = point_triangle_distance_sq(query, a, b, c);
                    if d < best {
                        best = d;
                        result = Some(MeshClosestPoint {
                            point: closest_point_on_triangle(query, a, b, c),
                            distance_sq: d,
                            triangle: self.source[idx],
                        });
                    }
                }
            } else if sp + 2 <= stack.len() {
                // Visit the nearer child first for tighter pruning.
                let l = node.first as usize;
                let r = l + 1;
                let dl = aabb_distance_sq(self.nodes[l].bmin, self.nodes[l].bmax, query);
                let dr = aabb_distance_sq(self.nodes[r].bmin, self.nodes[r].bmax, query);
                let (near, far) = if dl <= dr { (l, r) } else { (r, l) };
                stack[sp] = far as u32;
                sp += 1;
                stack[sp] = near as u32;
                sp += 1;
            }
        }
        result
    }

    /// Appends the original indices of every triangle whose axis-aligned bounds
    /// overlap the query box `[qmin, qmax]`.
    pub fn overlapping_triangles(&self, qmin: Vec3, qmax: Vec3, out: &mut Vec<u32>) {
        let mut stack = [0u32; 64];
        let mut sp = 0usize;
        stack[sp] = 0;
        sp += 1;

        while sp > 0 {
            sp -= 1;
            let node = &self.nodes[stack[sp] as usize];
            if !aabb_overlap(node.bmin, node.bmax, qmin, qmax) {
                continue;
            }
            if node.tri_count > 0 {
                let start = node.first as usize;
                for local in 0..node.tri_count as usize {
                    let idx = start + local;
                    let [a, b, c] = self.tris[idx];
                    let tmin = a.min(b).min(c);
                    let tmax = a.max(b).max(c);
                    if aabb_overlap(tmin, tmax, qmin, qmax) {
                        out.push(self.source[idx]);
                    }
                }
            } else if sp + 2 <= stack.len() {
                stack[sp] = node.first;
                sp += 1;
                stack[sp] = node.first + 1;
                sp += 1;
            }
        }
    }
}

/// Recursively subdivides `nodes[node_idx]`, which currently owns the triangle
/// slice `build[start..end]`, writing child nodes to the flat array.
fn subdivide(
    nodes: &mut Vec<BvhNode>,
    node_idx: usize,
    build: &mut [BuildTri],
    start: usize,
    end: usize,
) {
    // Compute this node's bounds from the triangles it owns.
    let (mut bmin, mut bmax) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
    let (mut cmin, mut cmax) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
    for t in &build[start..end] {
        bmin = bmin.min(t.bmin);
        bmax = bmax.max(t.bmax);
        cmin = cmin.min(t.centroid);
        cmax = cmax.max(t.centroid);
    }
    nodes[node_idx].bmin = bmin;
    nodes[node_idx].bmax = bmax;

    let count = end - start;
    if count <= MAX_LEAF_TRIS {
        nodes[node_idx].first = start as u32;
        nodes[node_idx].tri_count = count as u32;
        return;
    }

    // Split along the widest centroid-bound axis.
    let extent = cmax - cmin;
    let axis = if extent.x >= extent.y && extent.x >= extent.z {
        0
    } else if extent.y >= extent.z {
        1
    } else {
        2
    };
    let axis_extent = component(extent, axis);
    if axis_extent <= 0.0 {
        // All centroids coincide: cannot split, make a leaf.
        nodes[node_idx].first = start as u32;
        nodes[node_idx].tri_count = count as u32;
        return;
    }

    // Binned SAH: assign each triangle to a bin, accumulate counts and bounds.
    let scale = SAH_BINS as f32 / axis_extent;
    let cmin_axis = component(cmin, axis);
    let mut bin_count = [0usize; SAH_BINS];
    let mut bin_min = [Vec3::splat(f32::INFINITY); SAH_BINS];
    let mut bin_max = [Vec3::splat(f32::NEG_INFINITY); SAH_BINS];
    for t in &build[start..end] {
        let b = (((component(t.centroid, axis) - cmin_axis) * scale) as usize).min(SAH_BINS - 1);
        bin_count[b] += 1;
        bin_min[b] = bin_min[b].min(t.bmin);
        bin_max[b] = bin_max[b].max(t.bmax);
    }

    // Sweep the SAH_BINS-1 candidate planes, prefix (left) and suffix (right).
    let mut left_area = [0.0_f32; SAH_BINS - 1];
    let mut left_count = [0usize; SAH_BINS - 1];
    let (mut acc_count, mut acc_min, mut acc_max) = (
        0usize,
        Vec3::splat(f32::INFINITY),
        Vec3::splat(f32::NEG_INFINITY),
    );
    for i in 0..SAH_BINS - 1 {
        acc_count += bin_count[i];
        acc_min = acc_min.min(bin_min[i]);
        acc_max = acc_max.max(bin_max[i]);
        left_count[i] = acc_count;
        left_area[i] = half_surface_area(acc_min, acc_max);
    }
    let mut right_area = [0.0_f32; SAH_BINS - 1];
    let mut right_count = [0usize; SAH_BINS - 1];
    let (mut racc_count, mut racc_min, mut racc_max) = (
        0usize,
        Vec3::splat(f32::INFINITY),
        Vec3::splat(f32::NEG_INFINITY),
    );
    for i in (0..SAH_BINS - 1).rev() {
        racc_count += bin_count[i + 1];
        racc_min = racc_min.min(bin_min[i + 1]);
        racc_max = racc_max.max(bin_max[i + 1]);
        right_count[i] = racc_count;
        right_area[i] = half_surface_area(racc_min, racc_max);
    }

    // Pick the plane with minimum SAH cost.
    let mut best_cost = f32::INFINITY;
    let mut best_plane = 0usize;
    for i in 0..SAH_BINS - 1 {
        if left_count[i] == 0 || right_count[i] == 0 {
            continue;
        }
        let cost = left_area[i] * left_count[i] as f32 + right_area[i] * right_count[i] as f32;
        if cost < best_cost {
            best_cost = cost;
            best_plane = i;
        }
    }

    // Leaf cost = whole-node area times triangle count; bail out if no split
    // beats keeping everything in one leaf.
    let leaf_cost = half_surface_area(bmin, bmax) * count as f32;
    // When no split beats a single leaf and the node is small enough, stop;
    // otherwise fall through to a median partition so recursion terminates.
    if (!best_cost.is_finite() || best_cost >= leaf_cost) && count <= 2 * MAX_LEAF_TRIS {
        nodes[node_idx].first = start as u32;
        nodes[node_idx].tri_count = count as u32;
        return;
    }

    // Partition in place by bin index around the chosen plane.
    let split_plane = best_plane;
    let mut mid = partition(build, start, end, |t| {
        let b = (((component(t.centroid, axis) - cmin_axis) * scale) as usize).min(SAH_BINS - 1);
        b <= split_plane
    });
    // Guard against a degenerate partition (everything on one side).
    if mid == start || mid == end {
        mid = start + count / 2;
    }

    let left_child = nodes.len() as u32;
    nodes.push(BvhNode {
        bmin: Vec3::ZERO,
        bmax: Vec3::ZERO,
        first: 0,
        tri_count: 0,
    });
    nodes.push(BvhNode {
        bmin: Vec3::ZERO,
        bmax: Vec3::ZERO,
        first: 0,
        tri_count: 0,
    });
    nodes[node_idx].first = left_child;
    nodes[node_idx].tri_count = 0;

    subdivide(nodes, left_child as usize, build, start, mid);
    subdivide(nodes, left_child as usize + 1, build, mid, end);
}

/// Hoare-style stable-ish partition of `build[start..end]` by `pred`; returns
/// the index of the first element for which `pred` is false.
fn partition<F: Fn(&BuildTri) -> bool>(
    build: &mut [BuildTri],
    start: usize,
    end: usize,
    pred: F,
) -> usize {
    let mut i = start;
    for j in start..end {
        if pred(&build[j]) {
            build.swap(i, j);
            i += 1;
        }
    }
    i
}

#[inline]
fn component(v: Vec3, axis: usize) -> f32 {
    match axis {
        0 => v.x,
        1 => v.y,
        _ => v.z,
    }
}

/// Half the surface area of an AABB (the SAH surface metric). Returns 0 for an
/// inverted/empty box.
#[inline]
fn half_surface_area(min: Vec3, max: Vec3) -> f32 {
    let d = max - min;
    if d.x < 0.0 || d.y < 0.0 || d.z < 0.0 {
        return 0.0;
    }
    d.x * d.y + d.y * d.z + d.z * d.x
}

#[inline]
fn safe_inv(x: f32) -> f32 {
    if x.abs() < 1e-20 {
        if x < 0.0 {
            f32::NEG_INFINITY
        } else {
            f32::INFINITY
        }
    } else {
        1.0 / x
    }
}

/// Slab test: does the ray hit the AABB within `[0, max_t]`?
fn slab_hit(bmin: Vec3, bmax: Vec3, origin: Vec3, inv_dir: Vec3, max_t: f32) -> bool {
    let t0 = (bmin - origin) * inv_dir;
    let t1 = (bmax - origin) * inv_dir;
    let tsmall = t0.min(t1);
    let tbig = t0.max(t1);
    let tmin = tsmall.x.max(tsmall.y).max(tsmall.z).max(0.0);
    let tmax = tbig.x.min(tbig.y).min(tbig.z).min(max_t);
    tmin <= tmax
}

/// Moller-Trumbore ray/triangle intersection; returns the hit `t` (which may be
/// negative) when the ray crosses the triangle, else `None`.
fn ray_triangle(origin: Vec3, dir: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Option<f32> {
    const EPS: f32 = 1e-9;
    let e1 = b - a;
    let e2 = c - a;
    let pvec = dir.cross(e2);
    let det = e1.dot(pvec);
    if det.abs() < EPS {
        return None;
    }
    let inv_det = 1.0 / det;
    let tvec = origin - a;
    let u = tvec.dot(pvec) * inv_det;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let qvec = tvec.cross(e1);
    let v = dir.dot(qvec) * inv_det;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    Some(e2.dot(qvec) * inv_det)
}

/// Squared distance from a point to an AABB (0 when inside).
#[inline]
fn aabb_distance_sq(min: Vec3, max: Vec3, p: Vec3) -> f32 {
    let d = (min - p).max(Vec3::ZERO).max(p - max);
    d.length_squared()
}

#[inline]
fn aabb_overlap(amin: Vec3, amax: Vec3, bmin: Vec3, bmax: Vec3) -> bool {
    amin.x <= bmax.x
        && amax.x >= bmin.x
        && amin.y <= bmax.y
        && amax.y >= bmin.y
        && amin.z <= bmax.z
        && amax.z >= bmin.z
}

/// Closest point on triangle `abc` to `p` (Ericson's Voronoi-region method).
fn closest_point_on_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let w = d1 / (d1 - d3);
        return a + ab * w;
    }
    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return a + ac * w;
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + (c - b) * w;
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    a + ab * v + ac * w
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Axis-aligned cube centred at the origin, outward-wound triangles.
    fn cube(half: f32) -> (Vec<Vec3>, Vec<[u32; 3]>) {
        let h = half;
        let v = vec![
            Vec3::new(-h, -h, -h),
            Vec3::new(h, -h, -h),
            Vec3::new(h, h, -h),
            Vec3::new(-h, h, -h),
            Vec3::new(-h, -h, h),
            Vec3::new(h, -h, h),
            Vec3::new(h, h, h),
            Vec3::new(-h, h, h),
        ];
        let i = vec![
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [3, 7, 6],
            [3, 6, 2],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ];
        (v, i)
    }

    /// A grid of triangles in the z=0 plane, exercising a deeper tree.
    fn grid(n: usize, step: f32) -> (Vec<Vec3>, Vec<[u32; 3]>) {
        let mut v = Vec::new();
        for r in 0..=n {
            for c in 0..=n {
                v.push(Vec3::new(c as f32 * step, r as f32 * step, 0.0));
            }
        }
        let w = n + 1;
        let mut idx = Vec::new();
        for r in 0..n {
            for c in 0..n {
                let a = (r * w + c) as u32;
                let b = (r * w + c + 1) as u32;
                let cc = ((r + 1) * w + c) as u32;
                let d = ((r + 1) * w + c + 1) as u32;
                idx.push([a, b, d]);
                idx.push([a, d, cc]);
            }
        }
        (v, idx)
    }

    #[test]
    fn build_rejects_empty_and_degenerate() {
        assert!(MeshBvh::build(&[], &[]).is_none());
        let v = vec![Vec3::ZERO, Vec3::X, Vec3::new(2.0, 0.0, 0.0)]; // collinear
        assert!(MeshBvh::build(&v, &[[0, 1, 2]]).is_none());
        // Out-of-range triangle is skipped -> no valid triangles.
        let v = vec![Vec3::ZERO, Vec3::X, Vec3::Y];
        assert!(MeshBvh::build(&v, &[[0, 1, 9]]).is_none());
    }

    #[test]
    fn build_reports_sizes() {
        let (v, i) = cube(1.0);
        let bvh = MeshBvh::build(&v, &i).expect("builds");
        assert_eq!(bvh.triangle_count(), 12);
        assert!(bvh.node_count() >= 1);
        let (lo, hi) = bvh.local_aabb();
        assert!((lo + Vec3::ONE).length() < 1e-5);
        assert!((hi - Vec3::ONE).length() < 1e-5);
    }

    #[test]
    fn ray_cast_hits_front_face_of_cube() {
        let (v, i) = cube(1.0);
        let bvh = MeshBvh::build(&v, &i).expect("builds");
        // Ray from far -X toward +X through the centre: nearest hit is x=-1.
        let hit = bvh
            .ray_cast(Vec3::new(-5.0, 0.0, 0.0), Vec3::X, 100.0)
            .expect("hits");
        assert!((hit.time - 4.0).abs() < 1e-4, "t {}", hit.time);
        assert!((hit.point.x + 1.0).abs() < 1e-4, "x {}", hit.point.x);
        assert!(hit.normal.length() > 0.5);
    }

    #[test]
    fn ray_cast_misses_and_respects_max_time() {
        let (v, i) = cube(1.0);
        let bvh = MeshBvh::build(&v, &i).expect("builds");
        // Parallel ray that never touches the cube.
        assert!(bvh
            .ray_cast(Vec3::new(-5.0, 5.0, 0.0), Vec3::X, 100.0)
            .is_none());
        // Correct direction but max_time too short to reach the cube.
        assert!(bvh
            .ray_cast(Vec3::new(-5.0, 0.0, 0.0), Vec3::X, 1.0)
            .is_none());
    }

    #[test]
    fn ray_cast_on_grid_matches_brute_force() {
        let (v, i) = grid(8, 1.0);
        let bvh = MeshBvh::build(&v, &i).expect("builds");
        // Fire down -Z onto the plane at several points; all should hit z=0.
        for (x, y) in [(0.5, 0.5), (3.2, 1.1), (7.5, 7.5)] {
            let origin = Vec3::new(x, y, 5.0);
            let hit = bvh
                .ray_cast(origin, -Vec3::Z, 100.0)
                .unwrap_or_else(|| panic!("hit at {x},{y}"));
            assert!((hit.time - 5.0).abs() < 1e-3, "t {}", hit.time);
            assert!(hit.point.z.abs() < 1e-3);
        }
    }

    #[test]
    fn closest_point_matches_brute_force() {
        let (v, i) = grid(6, 1.0);
        let bvh = MeshBvh::build(&v, &i).expect("builds");
        let tris: Vec<[Vec3; 3]> = i
            .iter()
            .map(|t| [v[t[0] as usize], v[t[1] as usize], v[t[2] as usize]])
            .collect();
        for q in [
            Vec3::new(3.0, 3.0, 2.0),
            Vec3::new(-1.0, -1.0, 0.5),
            Vec3::new(5.5, 0.2, -3.0),
        ] {
            let brute = tris
                .iter()
                .map(|t| point_triangle_distance_sq(q, t[0], t[1], t[2]))
                .fold(f32::INFINITY, f32::min);
            let got = bvh.closest_point(q).expect("has result");
            assert!(
                (got.distance_sq - brute).abs() < 1e-3,
                "bvh {} brute {brute}",
                got.distance_sq
            );
            // The reported point really is at that distance.
            assert!(((got.point - q).length_squared() - got.distance_sq).abs() < 1e-3);
        }
    }

    #[test]
    fn overlapping_triangles_gathers_candidates() {
        let (v, i) = grid(8, 1.0);
        let bvh = MeshBvh::build(&v, &i).expect("builds");
        let mut out = Vec::new();
        // A small box over one grid cell should return just that cell's two tris
        // (plus possibly immediate neighbours sharing the boundary).
        bvh.overlapping_triangles(
            Vec3::new(0.1, 0.1, -0.1),
            Vec3::new(0.9, 0.9, 0.1),
            &mut out,
        );
        assert!(!out.is_empty());
        assert!(out.len() <= 8, "too many candidates: {}", out.len());

        // A box far from the mesh returns nothing.
        out.clear();
        bvh.overlapping_triangles(
            Vec3::new(100.0, 100.0, 100.0),
            Vec3::new(101.0, 101.0, 101.0),
            &mut out,
        );
        assert!(out.is_empty());

        // A box covering everything returns all triangles.
        out.clear();
        bvh.overlapping_triangles(
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::new(9.0, 9.0, 1.0),
            &mut out,
        );
        assert_eq!(out.len(), bvh.triangle_count());
    }

    #[test]
    fn deep_tree_build_terminates_and_is_valid() {
        // Many coplanar triangles stress the SAH/median fallback paths.
        let (v, i) = grid(20, 0.5);
        let bvh = MeshBvh::build(&v, &i).expect("builds");
        assert_eq!(bvh.triangle_count(), i.len());
        // Every ray straight down should still find the plane.
        let hit = bvh.ray_cast(Vec3::new(2.3, 2.3, 3.0), -Vec3::Z, 10.0);
        assert!(hit.is_some());
    }
}
