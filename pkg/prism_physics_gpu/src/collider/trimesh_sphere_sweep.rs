//! Swept-sphere-versus-triangle-mesh scene query: the first triangle a moving
//! sphere of fixed radius touches as it travels along a ray through a static
//! [`Trimesh`], reported as a time of impact, the contact point on the mesh
//! surface, and the push-out normal that drives the sphere off that surface.
//!
//! This is the mesh analogue of the convex cast family in
//! [`crate::narrowphase`]: where the convex sweeps cast a rounded shape at a set
//! of convex hulls, this casts a sphere at the triangle soup `AAA` engines use
//! for all static level geometry. It is the primitive behind `PhysX`
//! `PxMeshQuery::sweep`, Jolt `MeshShape::CastShape` with a sphere, and Unreal
//! `Chaos` triangle-mesh sphere sweeps.
//!
//! # Two interchangeable paths
//!
//! * [`cpu_trimesh_sphere_sweep`] is the brute-force golden: it sweeps the sphere
//!   against every triangle and keeps the earliest contact. It is the
//!   independent oracle the accelerated path and the `GPU` twin are pinned to.
//! * [`cpu_trimesh_sphere_sweep_bvh`] descends the `LBVH` built from
//!   [`Trimesh::triangle_aabbs`], expands each node box by the sphere radius, and
//!   prunes any subtree the sphere-centre ray enters only later than the earliest
//!   contact found so far, so it returns the identical hit while touching a
//!   fraction of the triangles.
//!
//! # Reduction to ray tests, shared across paths and the device
//!
//! A swept sphere against a solid triangle is a ray against the triangle grown
//! by the radius (a `Minkowski` sum), which decomposes into one face test and
//! three edge tests:
//!
//! * **Initial overlap**: if the sphere already touches the triangle at the
//!   start of the sweep (its centre lies within the radius of the triangle), the
//!   contact is immediate, so the time of impact is zero.
//! * **Face**: the moving centre reaches the plane offset by the radius on the
//!   sphere's side; if the touch point projects inside the triangle, that is the
//!   face contact.
//! * **Edges**: otherwise each of the three edges is a capsule of the sphere
//!   radius, and the sphere-centre ray is intersected with it. The capsule's
//!   hemispherical end caps automatically cover the three vertices, so no
//!   separate vertex test is needed. The earliest of the three edge contacts and
//!   the face contact wins.
//!
//! Every path and the `GPU` kernel run this exact [`sweep_sphere_triangle`]
//! arithmetic, so their hits agree to within the floating-point tolerance the
//! parity suite allows (only the handful of square roots and reciprocals differ
//! in their low bits).
//!
//! # Determinism
//!
//! The earliest contact wins; the face contact is preferred over an edge contact
//! at an equal time, and across triangles the lowest index wins because every
//! path keeps a candidate only when it is strictly earlier. The result is
//! therefore stable and identical across the brute, `BVH`, and device paths.
//!
//! # Provenance
//!
//! Moving-sphere / rounded-triangle reduction and the ray-versus-capsule and
//! intersecting-moving-sphere tests per Ericson, "Real-Time Collision Detection"
//! (2005), sections 5.3.4 and 5.5.7; `LBVH` per Karras 2012; branch-free slab box
//! test per Williams et al. 2005. No Unreal Engine source or derived code.

use glam::Vec3;

use crate::bvh::{cpu_build_lbvh, Aabb, Lbvh};

use super::trimesh_closest_point::closest_point_on_triangle;
use super::Trimesh;

/// A swept sphere for a mesh query: a start centre, a travel direction, a
/// maximum travel distance, and the sphere radius.
///
/// The sphere centre traces the point set `origin + t * direction` for `t` in
/// `[0, max_distance]`. [`direction`](SphereSweep::direction) should be a unit
/// vector so the reported [`TrimeshSweepHit::toi`] is a world-space length; a
/// non-unit direction scales the reported time of impact by the direction's
/// length.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SphereSweep {
    /// World-space starting centre of the moving sphere.
    pub origin: Vec3,
    /// Sweep direction; should be unit length for a metric time of impact.
    pub direction: Vec3,
    /// Largest `t` a contact may lie at; triangles touched later are rejected.
    pub max_distance: f32,
    /// Radius of the moving sphere.
    pub radius: f32,
}

impl SphereSweep {
    /// Creates a sweep from its `origin`, `direction`, `max_distance`, and
    /// `radius`.
    #[must_use]
    pub fn new(origin: Vec3, direction: Vec3, max_distance: f32, radius: f32) -> SphereSweep {
        SphereSweep {
            origin,
            direction,
            max_distance,
            radius,
        }
    }
}

/// A single triangle-mesh sphere-sweep contact.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrimeshSweepHit {
    /// Zero-based index of the triangle first touched, addressing the mesh in
    /// its original (unsorted) triangle order.
    pub triangle: u32,
    /// Time of impact: the sweep parameter `t` at first contact, a world-space
    /// length for a unit direction.
    pub toi: f32,
    /// World-space contact point on the triangle surface.
    pub point: Vec3,
    /// Unit push-out normal pointing from the triangle surface toward the sphere
    /// centre, matching the sphere-versus-triangle contact convention.
    pub normal: Vec3,
}

/// Magnitude below which a determinant, a squared edge length, or a squared
/// quadratic leading coefficient is treated as degenerate.
const PARALLEL_EPS: f32 = 1e-8;

/// Squared distance below which a point is treated as lying on the triangle
/// surface, so a degenerate offset falls back to the geometric face normal.
const ON_SURFACE_EPS2: f32 = 1e-12;

/// The earliest intersection of the sphere-centre ray with a sphere of radius
/// `r` centred at `centre`, within `[0, max]`, or [`None`] on a miss.
///
/// Assumes a unit `dir` so the quadratic leading coefficient is one; the first
/// (nearest) root is returned, and a start already inside the sphere (a negative
/// first root) is treated as a miss here because the triangle-level initial
/// overlap test handles that case.
#[must_use]
fn ray_sphere(origin: Vec3, dir: Vec3, max: f32, centre: Vec3, r: f32) -> Option<f32> {
    let m = origin - centre;
    let b = m.dot(dir);
    let c = m.dot(m) - r * r;
    let disc = b * b - c;
    if disc < 0.0 {
        return None;
    }
    let s = -b - disc.sqrt();
    if (0.0..=max).contains(&s) {
        Some(s)
    } else {
        None
    }
}

/// Sweeps the sphere against the triangle's plane and returns the face contact
/// when the touch point projects inside the triangle.
///
/// The moving centre reaches the plane offset by `r` on the side it starts on;
/// the contact point is that touch point projected onto the plane, accepted only
/// when it lies inside the solid triangle (tested by the shared Voronoi-region
/// closest-point helper). The normal is the unit face normal oriented toward the
/// sphere's start side.
#[must_use]
fn sweep_face(
    origin: Vec3,
    dir: Vec3,
    max: f32,
    r: f32,
    a: Vec3,
    b: Vec3,
    c: Vec3,
) -> Option<(f32, Vec3, Vec3)> {
    let raw = (b - a).cross(c - a);
    if raw.length_squared() < PARALLEL_EPS {
        return None;
    }
    let n = raw.normalize();
    let dist0 = (origin - a).dot(n);
    let d_n = dir.dot(n);
    if d_n.abs() < PARALLEL_EPS {
        return None;
    }
    let target = if dist0 >= 0.0 { r } else { -r };
    let s = (target - dist0) / d_n;
    if !(0.0..=max).contains(&s) {
        return None;
    }
    let centre_s = origin + dir * s;
    let contact = centre_s - n * target;
    let q = closest_point_on_triangle(contact, a, b, c);
    if (q - contact).length_squared() > ON_SURFACE_EPS2 {
        return None;
    }
    let normal = if dist0 >= 0.0 { n } else { -n };
    Some((s, contact, normal))
}

/// Sweeps the sphere against one edge treated as a capsule of radius `r` and
/// returns the edge contact, or [`None`] on a miss.
///
/// The sphere-centre ray is intersected with the infinite cylinder of the edge
/// (clamped to the segment extent) and with the two end-cap spheres; the capsule
/// caps cover the edge's endpoints, so the shared vertices need no separate test.
/// The earliest valid root wins. The contact point is the closest point on the
/// edge segment to the centre at impact, and the normal points from that point
/// toward the centre.
#[must_use]
fn sweep_edge(
    origin: Vec3,
    dir: Vec3,
    max: f32,
    r: f32,
    p1: Vec3,
    p2: Vec3,
) -> Option<(f32, Vec3, Vec3)> {
    let axis = p2 - p1;
    let axis_len2 = axis.dot(axis);
    let mut best_s: Option<f32> = None;

    // Cylinder side surface.
    if axis_len2 > PARALLEL_EPS {
        let axis_len = axis_len2.sqrt();
        let ax = axis / axis_len;
        let m = origin - p1;
        let dir_perp = dir - ax * dir.dot(ax);
        let m_perp = m - ax * m.dot(ax);
        let aa = dir_perp.dot(dir_perp);
        if aa > PARALLEL_EPS {
            let bb = 2.0 * m_perp.dot(dir_perp);
            let cc = m_perp.dot(m_perp) - r * r;
            let disc = bb * bb - 4.0 * aa * cc;
            if disc >= 0.0 {
                let s = (-bb - disc.sqrt()) / (2.0 * aa);
                if (0.0..=max).contains(&s) {
                    let centre_s = origin + dir * s;
                    let u = (centre_s - p1).dot(ax);
                    if (0.0..=axis_len).contains(&u) {
                        best_s = Some(s);
                    }
                }
            }
        }
    }

    // End-cap spheres at both endpoints; a cap hit is valid only when the
    // contact lies beyond that end of the segment (or the segment is degenerate).
    for (cap, is_p1) in [(p1, true), (p2, false)] {
        let Some(s) = ray_sphere(origin, dir, max, cap, r) else {
            continue;
        };
        let centre_s = origin + dir * s;
        let along = (centre_s - p1).dot(axis);
        let valid = if axis_len2 <= PARALLEL_EPS {
            true
        } else if is_p1 {
            along <= 0.0
        } else {
            along >= axis_len2
        };
        if valid && best_s.is_none_or(|bs| s < bs) {
            best_s = Some(s);
        }
    }

    let s = best_s?;
    let centre_s = origin + dir * s;
    let t = if axis_len2 > PARALLEL_EPS {
        ((centre_s - p1).dot(axis) / axis_len2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let q = p1 + axis * t;
    let normal = (centre_s - q).normalize_or_zero();
    Some((s, q, normal))
}

/// The shared single-triangle swept-sphere solve, used by the brute path, the
/// `BVH` path, and mirrored by the `GPU` kernel.
///
/// Returns the time of impact, the contact point on the triangle surface, and
/// the push-out normal when the moving sphere first touches the triangle within
/// `[0, max_distance]`, or [`None`] on a miss. The initial-overlap case returns a
/// time of impact of zero; otherwise the earliest of the face contact and the
/// three edge contacts wins, with the face preferred on an exact tie.
#[must_use]
pub(crate) fn sweep_sphere_triangle(
    origin: Vec3,
    dir: Vec3,
    max_distance: f32,
    radius: f32,
    a: Vec3,
    b: Vec3,
    c: Vec3,
) -> Option<(f32, Vec3, Vec3)> {
    // Initial overlap: the sphere already touches the triangle at the start.
    let q0 = closest_point_on_triangle(origin, a, b, c);
    let diff0 = origin - q0;
    let d02 = diff0.dot(diff0);
    if d02 < radius * radius {
        let normal = if d02 > ON_SURFACE_EPS2 {
            diff0 / d02.sqrt()
        } else {
            (b - a).cross(c - a).normalize_or_zero()
        };
        return Some((0.0, q0, normal));
    }

    let mut best: Option<(f32, Vec3, Vec3)> = sweep_face(origin, dir, max_distance, radius, a, b, c);
    for (p1, p2) in [(a, b), (b, c), (c, a)] {
        if let Some(hit) = sweep_edge(origin, dir, max_distance, radius, p1, p2)
            && best.is_none_or(|(bt, _, _)| hit.0 < bt)
        {
            best = Some(hit);
        }
    }
    best
}

/// Whether `candidate` should replace `best`: strictly earlier wins, so an exact
/// time tie keeps the earlier (lower-index) triangle. Shared by the `CPU` and
/// `GPU` earliest-contact reductions.
#[must_use]
#[expect(
    clippy::float_cmp,
    reason = "an exact time-of-impact equality is an intentional tie detector; the lower triangle index then wins so the brute, LBVH, and GPU reductions agree on the same triangle regardless of visitation order"
)]
pub(crate) fn closer_hit(candidate: &TrimeshSweepHit, best: &Option<TrimeshSweepHit>) -> bool {
    match best {
        // Lexicographic order on (time of impact, triangle index): a strictly
        // earlier contact always wins, and an exact time tie resolves to the
        // lower triangle index. This makes the reduction a total order, so the
        // index-order brute sweep, the BVH-order LBVH sweep, and the GPU host
        // reduction all converge on the identical triangle.
        Some(b) => {
            candidate.toi < b.toi
                || (candidate.toi == b.toi && candidate.triangle < b.triangle)
        }
        None => true,
    }
}

/// Sweeps `sweep` against `mesh` by brute force, returning the earliest triangle
/// contact.
///
/// Sweeps the sphere against every triangle and keeps the earliest; ties resolve
/// to the lowest triangle index. Returns [`None`] when the sphere touches no
/// triangle within range or the mesh is empty. This is the independent golden
/// the accelerated and device paths are pinned to.
#[must_use]
pub fn cpu_trimesh_sphere_sweep(mesh: &Trimesh, sweep: &SphereSweep) -> Option<TrimeshSweepHit> {
    let mut best: Option<TrimeshSweepHit> = None;
    for i in 0..mesh.triangle_count() {
        let tri = mesh.triangle(i);
        let Some((toi, point, normal)) = sweep_sphere_triangle(
            sweep.origin,
            sweep.direction,
            sweep.max_distance,
            sweep.radius,
            tri.a,
            tri.b,
            tri.c,
        ) else {
            continue;
        };
        let index = u32::try_from(i).unwrap_or(u32::MAX);
        let hit = TrimeshSweepHit {
            triangle: index,
            toi,
            point,
            normal,
        };
        if closer_hit(&hit, &best) {
            best = Some(hit);
        }
    }
    best
}

/// The entry distance where the sphere-centre ray enters `aabb` grown by the
/// sphere radius `r`, or [`None`] on a miss.
///
/// Expanding the triangle box by `r` and running the branch-free slab test of
/// Williams et al. (2005) yields a lower bound on the time of impact of any
/// triangle in the box: when the sphere surface touches a triangle, its centre
/// is within `r` of that triangle and so lies in the expanded box, so the centre
/// ray must enter the box no later than the contact. Pruning a subtree whose
/// entry exceeds the earliest contact so far therefore never discards a nearer
/// triangle.
#[must_use]
fn aabb_sweep_enter(origin: Vec3, dir: Vec3, max: f32, aabb: &Aabb, r: f32) -> Option<f32> {
    let grow = Vec3::splat(r);
    let inv = Vec3::ONE / dir;
    let t0 = (aabb.min - grow - origin) * inv;
    let t1 = (aabb.max + grow - origin) * inv;
    let t_near = t0.min(t1).max_element().max(0.0);
    let t_far = t0.max(t1).min_element();
    if t_near <= t_far && t_near <= max {
        Some(t_near)
    } else {
        None
    }
}

/// The bounding box of an encoded `BVH` node: its leaf box or internal union.
#[must_use]
fn node_aabb(tree: &Lbvh, encoded: u32) -> Aabb {
    if tree.is_leaf(encoded) {
        tree.leaf_aabb[(encoded as usize) - tree.num_internal]
    } else {
        tree.internal_aabb[encoded as usize]
    }
}

/// Accelerated form of [`cpu_trimesh_sphere_sweep`]: descends the `LBVH` over the
/// mesh's per-triangle boxes and prunes any subtree the sphere-centre ray enters
/// only later than the earliest contact found so far.
///
/// `lbvh` must be the hierarchy built from `mesh.triangle_aabbs()`; its leaf
/// slots map back to original triangle indices through
/// [`Lbvh::sorted_indices`]. The result matches [`cpu_trimesh_sphere_sweep`]
/// exactly whenever the earliest contact is unique. Returns [`None`] when the
/// sphere touches no triangle within range or the tree is empty.
#[must_use]
pub fn cpu_trimesh_sphere_sweep_bvh(
    mesh: &Trimesh,
    lbvh: &Lbvh,
    sweep: &SphereSweep,
) -> Option<TrimeshSweepHit> {
    if lbvh.num_leaves == 0 {
        return None;
    }
    let mut best: Option<TrimeshSweepHit> = None;
    let mut stack = vec![lbvh.root];
    while let Some(node) = stack.pop() {
        let Some(enter) = aabb_sweep_enter(
            sweep.origin,
            sweep.direction,
            sweep.max_distance,
            &node_aabb(lbvh, node),
            sweep.radius,
        ) else {
            continue;
        };
        if let Some(b) = best
            && enter > b.toi
        {
            continue;
        }
        if lbvh.is_leaf(node) {
            let leaf = (node as usize) - lbvh.num_internal;
            let prim = lbvh.sorted_indices[leaf] as usize;
            let tri = mesh.triangle(prim);
            let Some((toi, point, normal)) = sweep_sphere_triangle(
                sweep.origin,
                sweep.direction,
                sweep.max_distance,
                sweep.radius,
                tri.a,
                tri.b,
                tri.c,
            ) else {
                continue;
            };
            let index = u32::try_from(prim).unwrap_or(u32::MAX);
            let hit = TrimeshSweepHit {
                triangle: index,
                toi,
                point,
                normal,
            };
            if closer_hit(&hit, &best) {
                best = Some(hit);
            }
        } else {
            stack.push(lbvh.left[node as usize]);
            stack.push(lbvh.right[node as usize]);
        }
    }
    best
}

/// Builds the mesh's `LBVH` and sweeps `sweep` at it in one call, the convenience
/// form of [`cpu_trimesh_sphere_sweep_bvh`] for callers that do not cache the
/// tree.
#[must_use]
pub fn cpu_trimesh_sphere_sweep_built(
    mesh: &Trimesh,
    sweep: &SphereSweep,
) -> Option<TrimeshSweepHit> {
    let lbvh = cpu_build_lbvh(&mesh.triangle_aabbs());
    cpu_trimesh_sphere_sweep_bvh(mesh, &lbvh, sweep)
}

#[cfg(test)]
mod tests {
    use super::{
        cpu_trimesh_sphere_sweep, cpu_trimesh_sphere_sweep_built, cpu_trimesh_sphere_sweep_bvh,
        SphereSweep,
    };
    use crate::collider::Trimesh;
    use glam::Vec3;

    /// A unit quad in the `z = 0` plane spanning `[0, 1]^2`, two CCW triangles
    /// seen from `+z` sharing the `(0, 0)`-to-`(1, 1)` diagonal.
    fn unit_quad() -> Trimesh {
        Trimesh::new(
            vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
            vec![[0, 1, 2], [0, 2, 3]],
        )
    }

    #[test]
    fn sweeps_onto_front_face() {
        let mesh = unit_quad();
        // Sphere descending from +z toward the quad centre; it stops when its
        // surface reaches the plane, i.e. the centre is one radius above z = 0.
        let sweep = SphereSweep::new(Vec3::new(0.3, 0.1, 5.0), Vec3::new(0.0, 0.0, -1.0), 100.0, 0.5);
        let hit = cpu_trimesh_sphere_sweep(&mesh, &sweep).expect("hits the quad");
        assert_eq!(hit.triangle, 0, "lower-right triangle covers (0.3, 0.1)");
        assert!((hit.toi - 4.5).abs() < 1e-4, "toi was {}", hit.toi);
        assert!((hit.point - Vec3::new(0.3, 0.1, 0.0)).length() < 1e-4);
        assert!((hit.normal - Vec3::Z).length() < 1e-4, "normal points at sphere");
    }

    #[test]
    fn misses_when_aimed_away() {
        let mesh = unit_quad();
        let sweep = SphereSweep::new(Vec3::new(0.3, 0.1, 5.0), Vec3::new(0.0, 0.0, 1.0), 100.0, 0.5);
        assert!(cpu_trimesh_sphere_sweep(&mesh, &sweep).is_none(), "sweeps away from quad");
    }

    #[test]
    fn respects_max_distance() {
        let mesh = unit_quad();
        // Quad is 5 away, sphere radius 0.5 so contact is at toi 4.5; a max of
        // 2 cannot reach it.
        let sweep = SphereSweep::new(Vec3::new(0.3, 0.1, 5.0), Vec3::new(0.0, 0.0, -1.0), 2.0, 0.5);
        assert!(cpu_trimesh_sphere_sweep(&mesh, &sweep).is_none(), "contact beyond max");
    }

    #[test]
    fn initial_overlap_reports_zero_toi() {
        let mesh = unit_quad();
        // Sphere centred just above the quad with a radius that already reaches
        // the surface: the contact is immediate.
        let sweep = SphereSweep::new(Vec3::new(0.3, 0.1, 0.2), Vec3::new(0.0, 0.0, -1.0), 100.0, 0.5);
        let hit = cpu_trimesh_sphere_sweep(&mesh, &sweep).expect("already overlapping");
        assert!((hit.toi - 0.0).abs() < 1e-6, "toi was {}", hit.toi);
        assert!((hit.normal - Vec3::Z).length() < 1e-4, "normal from surface to centre");
    }

    #[test]
    fn sweeps_onto_edge() {
        let mesh = unit_quad();
        // In-plane sweep toward the right edge x = 1; the face test is parallel
        // and misses, so the edge capsule must catch it. Centre stops one radius
        // from the edge line.
        let sweep = SphereSweep::new(Vec3::new(1.5, 0.5, 0.0), Vec3::new(-1.0, 0.0, 0.0), 100.0, 0.3);
        let hit = cpu_trimesh_sphere_sweep(&mesh, &sweep).expect("hits the right edge");
        assert_eq!(hit.triangle, 0, "right edge belongs to the lower-right triangle");
        assert!((hit.toi - 0.2).abs() < 1e-4, "toi was {}", hit.toi);
        assert!((hit.point - Vec3::new(1.0, 0.5, 0.0)).length() < 1e-4);
        assert!((hit.normal - Vec3::X).length() < 1e-4, "normal points back along +x");
    }

    #[test]
    fn bvh_matches_brute_on_nearest_of_many() {
        // A fan of parallel quads at increasing z; the nearest to a +z origin
        // must win on both paths. The sweep stays off the shared diagonal
        // (y < x), so the winning triangle is unambiguous.
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        for k in 0..8_u32 {
            let z = k as f32;
            let base = vertices.len() as u32;
            vertices.push(Vec3::new(-1.0, -1.0, z));
            vertices.push(Vec3::new(1.0, -1.0, z));
            vertices.push(Vec3::new(1.0, 1.0, z));
            vertices.push(Vec3::new(-1.0, 1.0, z));
            indices.push([base, base + 1, base + 2]);
            indices.push([base, base + 2, base + 3]);
        }
        let mesh = Trimesh::new(vertices, indices);
        let sweep = SphereSweep::new(Vec3::new(0.3, 0.1, 20.0), Vec3::new(0.0, 0.0, -1.0), 100.0, 0.25);
        let brute = cpu_trimesh_sphere_sweep(&mesh, &sweep).expect("hits");
        let bvh = cpu_trimesh_sphere_sweep_built(&mesh, &sweep).expect("hits");
        assert_eq!(brute.triangle, bvh.triangle, "same winning triangle");
        assert!((brute.toi - bvh.toi).abs() < 1e-5);
        // Nearest quad is z = 7 (centre starts at z = 20); surface touches it one
        // radius early, so toi = 13 - 0.25 = 12.75.
        assert!((brute.toi - 12.75).abs() < 1e-4, "toi was {}", brute.toi);
        assert_eq!(brute.triangle, 14, "nearest lower-right triangle wins");
    }

    #[test]
    fn empty_mesh_misses() {
        let mesh = Trimesh::new(Vec::new(), Vec::new());
        let sweep = SphereSweep::new(Vec3::ZERO, Vec3::X, 100.0, 0.5);
        assert!(cpu_trimesh_sphere_sweep(&mesh, &sweep).is_none());
        let lbvh = crate::cpu_build_lbvh(&mesh.triangle_aabbs());
        assert!(cpu_trimesh_sphere_sweep_bvh(&mesh, &lbvh, &sweep).is_none());
    }
}
