//! Swept-capsule-versus-triangle-mesh scene query: the first triangle a moving
//! capsule of fixed radius touches as it travels along a ray through a static
//! [`Trimesh`], reported as a time of impact, the contact point on the mesh
//! surface, and the push-out normal that drives the capsule off that surface.
//!
//! This is the character-controller workhorse of the mesh query family: a
//! capsule is the standard proxy for a walking or falling actor, and sweeping it
//! against the triangle soup `AAA` engines use for static level geometry is how
//! movement, step-up, and landing are resolved. It is the primitive behind
//! `PhysX` `PxMeshQuery::sweep` with a capsule, Jolt `MeshShape::CastShape` with
//! a capsule, and Unreal `Chaos` triangle-mesh capsule sweeps.
//!
//! # Reduction to conservative advancement, shared across paths and the device
//!
//! A capsule is a segment inflated by a convex radius, so sweeping it against a
//! triangle is a continuous collision query between two convex cores: the moving
//! segment and the static triangle, with the cap radius carried as the sweep's
//! rounding. Rather than enumerate the four closed-form cases (segment face,
//! segment vertex, and two moving edge-edge families, the last of which is the
//! error-prone one), every path runs the same conservative-advancement time of
//! impact that `PhysX` and Jolt use for shape casts:
//! [`conservative_advancement_toi_rounded`](crate::narrowphase::conservative_advancement_toi_rounded)
//! repeatedly measures the `GJK` separation of the segment and triangle cores,
//! bounds how fast the moving segment can close that gap, and advances time by
//! the largest provably safe step until the inflated surfaces touch. Because the
//! advance never overshoots the first contact, the reported impact is the
//! earliest, and because the `CPU` paths and the `GPU` kernel run the identical
//! `GJK` and advance arithmetic, their hits agree to the floating-point
//! tolerance the parity suite allows.
//!
//! # Two interchangeable `CPU` paths
//!
//! * [`cpu_trimesh_capsule_sweep`] is the brute-force golden: it sweeps the
//!   capsule against every triangle and keeps the earliest contact. It is the
//!   independent oracle the accelerated path and the `GPU` twin are pinned to.
//! * [`cpu_trimesh_capsule_sweep_bvh`] descends the `LBVH` built from
//!   [`Trimesh::triangle_aabbs`], expands each node box by the capsule's bounding
//!   radius (its half-length plus the cap radius), and prunes any subtree the
//!   capsule-centre ray enters only later than the earliest contact found so
//!   far. The capsule is contained in a ball of that radius about its centre, so
//!   the grown-box entry time is a lower bound on any contained triangle's time
//!   of impact and the prune never discards a nearer triangle.
//!
//! # Determinism
//!
//! The earliest contact wins, and across triangles the lowest index wins because
//! every path keeps a candidate only when it is strictly earlier. The result is
//! therefore stable and identical across the brute, `BVH`, and device paths.
//!
//! # Provenance
//!
//! Conservative advancement after Brian Mirtich, *Timewarp Rigid Body
//! Simulation* (2000) and the ray-casting formulation of Gino van den Bergen
//! (2004); `GJK` distance per Gilbert-Johnson-Keerthi (1988) with Ericson's
//! Voronoi sub-distance (2005); `LBVH` per Karras 2012; branch-free slab box test
//! per Williams et al. 2005. No Unreal Engine source or derived code.

use glam::{Quat, Vec3};

use crate::bvh::{cpu_build_lbvh, Aabb, Lbvh};
use crate::narrowphase::{
    conservative_advancement_toi_rounded, BodyMotion, ConvexHull, ConvexPose,
};

use super::Trimesh;

/// A swept capsule for a mesh query: the two world-space segment endpoints at
/// the start of the sweep, a travel direction, a maximum travel distance, and
/// the capsule's cap radius.
///
/// The whole capsule translates rigidly along `origin + t * direction` (there is
/// no rotation during the sweep), so both endpoints advance together. The
/// [`direction`](CapsuleSweep::direction) should be a unit vector so the reported
/// [`CapsuleSweepHit::toi`] is a world-space length; a non-unit direction scales
/// the reported time of impact by the direction's length.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CapsuleSweep {
    /// First world-space endpoint of the capsule segment at the sweep start.
    pub point_a: Vec3,
    /// Second world-space endpoint of the capsule segment at the sweep start.
    pub point_b: Vec3,
    /// Sweep direction; should be unit length for a metric time of impact.
    pub direction: Vec3,
    /// Largest `t` a contact may lie at; triangles touched later are rejected.
    pub max_distance: f32,
    /// Cap radius of the moving capsule.
    pub radius: f32,
}

impl CapsuleSweep {
    /// Creates a sweep from its segment endpoints, `direction`, `max_distance`,
    /// and `radius`.
    #[must_use]
    pub fn new(
        point_a: Vec3,
        point_b: Vec3,
        direction: Vec3,
        max_distance: f32,
        radius: f32,
    ) -> CapsuleSweep {
        CapsuleSweep {
            point_a,
            point_b,
            direction,
            max_distance,
            radius,
        }
    }

    /// The capsule segment's centre at the start of the sweep.
    #[must_use]
    fn centre(&self) -> Vec3 {
        (self.point_a + self.point_b) * 0.5
    }

    /// The radius of the ball that bounds the whole capsule about its centre:
    /// the segment half-length plus the cap radius. Used to grow triangle boxes
    /// for the `BVH` prune so no contact is missed.
    #[must_use]
    fn bounding_radius(&self) -> f32 {
        (self.point_b - self.point_a).length() * 0.5 + self.radius
    }
}

/// A single triangle-mesh capsule-sweep contact.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CapsuleSweepHit {
    /// Zero-based index of the triangle first touched, addressing the mesh in
    /// its original (unsorted) triangle order.
    pub triangle: u32,
    /// Time of impact: the sweep parameter `t` at first contact, a world-space
    /// length for a unit direction.
    pub toi: f32,
    /// World-space contact point, the midpoint of the two closest surface
    /// witnesses at the time of impact.
    pub point: Vec3,
    /// Unit push-out normal pointing from the triangle surface toward the
    /// capsule, matching the contact convention of the rounded shape cast.
    pub normal: Vec3,
}

/// The shared single-triangle swept-capsule solve, used by the brute path, the
/// `BVH` path, and mirrored by the `GPU` kernel.
///
/// Models the capsule as a segment core (centred on the sweep's segment
/// endpoints, translating along the sweep direction) inflated by the cap radius,
/// and the triangle as a static zero-radius core, then runs the rounded
/// conservative-advancement time of impact between them. Returns the time of
/// impact, the contact point, and the push-out normal (from the triangle toward
/// the capsule) when the moving capsule first touches the triangle within
/// `[0, max_distance]`, or [`None`] on a miss. An initial overlap is reported at
/// a time of impact of zero.
#[must_use]
pub(crate) fn sweep_capsule_triangle(
    sweep: &CapsuleSweep,
    a: Vec3,
    b: Vec3,
    c: Vec3,
) -> Option<(f32, Vec3, Vec3)> {
    // Capsule core: a segment along the endpoints' axis, centred on the local
    // origin and placed at the segment centre with no rotation. Posed this way
    // the local endpoints map back exactly onto point_a and point_b, and the
    // sweep's linear motion carries the whole segment along the direction.
    let axis = sweep.point_b - sweep.point_a;
    let half_height = axis.length() * 0.5;
    let capsule = ConvexHull::from_segment(axis, half_height);
    let capsule_pose = ConvexPose::new(sweep.centre(), Quat::IDENTITY);
    let capsule_motion = BodyMotion::new(sweep.direction, Vec3::ZERO);

    // Triangle core: the three world corners, static, zero convex radius.
    let triangle = ConvexHull::from_triangle(a, b, c);
    let triangle_pose = ConvexPose::new(Vec3::ZERO, Quat::IDENTITY);
    let triangle_motion = BodyMotion::still();

    let toi = conservative_advancement_toi_rounded(
        &capsule,
        &capsule_pose,
        &capsule_motion,
        sweep.radius,
        &triangle,
        &triangle_pose,
        &triangle_motion,
        0.0,
        sweep.max_distance,
        0.0,
    )?;
    Some((toi.time, toi.point, toi.normal))
}

/// Whether `candidate` should replace `best`: strictly earlier wins, so an exact
/// time tie keeps the earlier (lower-index) triangle. Shared by the `CPU` and
/// `GPU` earliest-contact reductions.
#[must_use]
#[expect(
    clippy::float_cmp,
    reason = "an exact time-of-impact equality is an intentional tie detector; the lower triangle index then wins so the brute, LBVH, and GPU reductions agree on the same triangle regardless of visitation order"
)]
pub(crate) fn closer_hit(candidate: &CapsuleSweepHit, best: &Option<CapsuleSweepHit>) -> bool {
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
/// Sweeps the capsule against every triangle and keeps the earliest; ties
/// resolve to the lowest triangle index. Returns [`None`] when the capsule
/// touches no triangle within range or the mesh is empty. This is the
/// independent golden the accelerated and device paths are pinned to.
#[must_use]
pub fn cpu_trimesh_capsule_sweep(mesh: &Trimesh, sweep: &CapsuleSweep) -> Option<CapsuleSweepHit> {
    let mut best: Option<CapsuleSweepHit> = None;
    for i in 0..mesh.triangle_count() {
        let tri = mesh.triangle(i);
        let Some((toi, point, normal)) = sweep_capsule_triangle(sweep, tri.a, tri.b, tri.c) else {
            continue;
        };
        let hit = CapsuleSweepHit {
            triangle: u32::try_from(i).unwrap_or(u32::MAX),
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

/// The entry distance where the capsule-centre ray enters `aabb` grown by the
/// capsule's bounding radius `r`, or [`None`] on a miss.
///
/// The capsule is contained in a ball of radius `r` about its centre, so when
/// its surface touches a triangle the centre is within `r` of that triangle and
/// therefore lies in the triangle box grown by `r`. The branch-free slab test of
/// Williams et al. (2005) returns when the centre ray enters that grown box,
/// which is a lower bound on the time of impact of any triangle inside: pruning a
/// subtree whose entry exceeds the earliest contact so far never discards a
/// nearer triangle.
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

/// Accelerated form of [`cpu_trimesh_capsule_sweep`]: descends the `LBVH` over
/// the mesh's per-triangle boxes and prunes any subtree the capsule-centre ray
/// enters only later than the earliest contact found so far.
///
/// `lbvh` must be the hierarchy built from `mesh.triangle_aabbs()`; its leaf
/// slots map back to original triangle indices through
/// [`Lbvh::sorted_indices`]. The result matches [`cpu_trimesh_capsule_sweep`]
/// exactly whenever the earliest contact is unique. Returns [`None`] when the
/// capsule touches no triangle within range or the tree is empty.
#[must_use]
pub fn cpu_trimesh_capsule_sweep_bvh(
    mesh: &Trimesh,
    lbvh: &Lbvh,
    sweep: &CapsuleSweep,
) -> Option<CapsuleSweepHit> {
    if lbvh.num_leaves == 0 {
        return None;
    }
    let origin = sweep.centre();
    let r = sweep.bounding_radius();
    let mut best: Option<CapsuleSweepHit> = None;
    let mut stack = vec![lbvh.root];
    while let Some(node) = stack.pop() {
        let Some(enter) = aabb_sweep_enter(
            origin,
            sweep.direction,
            sweep.max_distance,
            &node_aabb(lbvh, node),
            r,
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
            let Some((toi, point, normal)) = sweep_capsule_triangle(sweep, tri.a, tri.b, tri.c)
            else {
                continue;
            };
            let hit = CapsuleSweepHit {
                triangle: u32::try_from(prim).unwrap_or(u32::MAX),
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
/// form of [`cpu_trimesh_capsule_sweep_bvh`] for callers that do not cache the
/// tree.
#[must_use]
pub fn cpu_trimesh_capsule_sweep_built(
    mesh: &Trimesh,
    sweep: &CapsuleSweep,
) -> Option<CapsuleSweepHit> {
    let lbvh = cpu_build_lbvh(&mesh.triangle_aabbs());
    cpu_trimesh_capsule_sweep_bvh(mesh, &lbvh, sweep)
}

#[cfg(test)]
mod tests {
    use super::{
        cpu_trimesh_capsule_sweep, cpu_trimesh_capsule_sweep_built, cpu_trimesh_capsule_sweep_bvh,
        CapsuleSweep,
    };
    use crate::collider::Trimesh;
    use glam::Vec3;

    /// A unit quad in the `z = 0` plane spanning `[0, 1]^2`, two CCW triangles
    /// seen from `+z` sharing the `(0, 0)`-to-`(1, 1)` diagonal. Triangle 0 is the
    /// lower-right (`x >= y`) half, triangle 1 the upper-left.
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
        // A horizontal capsule (axis along x) hovering at z = 5 over the lower-
        // right triangle, swept straight down. Its underside reaches the plane
        // when the segment centre is one radius above z = 0, i.e. after dropping
        // 5 - 0.5 = 4.5.
        let sweep = CapsuleSweep::new(
            Vec3::new(0.4, 0.2, 5.0),
            Vec3::new(0.8, 0.2, 5.0),
            Vec3::new(0.0, 0.0, -1.0),
            100.0,
            0.5,
        );
        let hit = cpu_trimesh_capsule_sweep(&mesh, &sweep).expect("hits the quad");
        assert_eq!(hit.triangle, 0, "segment sits over the lower-right triangle");
        assert!((hit.toi - 4.5).abs() < 1e-3, "toi was {}", hit.toi);
        assert!(hit.point.z.abs() < 1e-3, "contact on the z = 0 plane, was {}", hit.point.z);
        assert!((hit.normal - Vec3::Z).length() < 1e-3, "normal points up at the capsule");
    }

    #[test]
    fn misses_when_aimed_away() {
        let mesh = unit_quad();
        let sweep = CapsuleSweep::new(
            Vec3::new(0.4, 0.2, 5.0),
            Vec3::new(0.8, 0.2, 5.0),
            Vec3::new(0.0, 0.0, 1.0),
            100.0,
            0.5,
        );
        assert!(
            cpu_trimesh_capsule_sweep(&mesh, &sweep).is_none(),
            "sweeps away from the quad"
        );
    }

    #[test]
    fn respects_max_distance() {
        let mesh = unit_quad();
        // Contact is at toi 4.5; a max of 2 cannot reach it.
        let sweep = CapsuleSweep::new(
            Vec3::new(0.4, 0.2, 5.0),
            Vec3::new(0.8, 0.2, 5.0),
            Vec3::new(0.0, 0.0, -1.0),
            2.0,
            0.5,
        );
        assert!(
            cpu_trimesh_capsule_sweep(&mesh, &sweep).is_none(),
            "contact beyond max distance"
        );
    }

    #[test]
    fn initial_overlap_reports_zero_toi() {
        let mesh = unit_quad();
        // Capsule hovering only 0.3 above the plane with a 0.5 radius already
        // overlaps the quad: the contact is immediate.
        let sweep = CapsuleSweep::new(
            Vec3::new(0.4, 0.2, 0.3),
            Vec3::new(0.8, 0.2, 0.3),
            Vec3::new(0.0, 0.0, -1.0),
            100.0,
            0.5,
        );
        let hit = cpu_trimesh_capsule_sweep(&mesh, &sweep).expect("already overlapping");
        assert!(hit.toi.abs() < 1e-6, "toi was {}", hit.toi);
    }

    #[test]
    fn sweeps_onto_edge() {
        let mesh = unit_quad();
        // A vertical capsule (axis along z) standing just right of the quad's
        // right edge (x = 1), swept in -x. The face plane is parallel to the
        // segment so the edge of the quad catches it: the segment centre stops
        // one radius from the edge line, after travelling 0.5 - 0.3 = 0.2.
        let sweep = CapsuleSweep::new(
            Vec3::new(1.5, 0.5, 1.0),
            Vec3::new(1.5, 0.5, -1.0),
            Vec3::new(-1.0, 0.0, 0.0),
            100.0,
            0.3,
        );
        let hit = cpu_trimesh_capsule_sweep(&mesh, &sweep).expect("hits the right edge");
        assert_eq!(hit.triangle, 0, "the right edge belongs to the lower-right triangle");
        assert!((hit.toi - 0.2).abs() < 1e-3, "toi was {}", hit.toi);
        assert!(
            (hit.point - Vec3::new(1.0, 0.5, 0.0)).length() < 1e-3,
            "contact on the edge midpoint, was {:?}",
            hit.point
        );
        assert!((hit.normal - Vec3::X).length() < 1e-3, "normal points back along +x");
    }

    #[test]
    fn bvh_matches_brute_on_nearest_of_many() {
        // A fan of parallel quads at increasing z; the nearest to a high-z origin
        // must win on both paths. The capsule stays in the lower-right (y < x)
        // half so the winning triangle is unambiguous.
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        for k in 0..8_u32 {
            let z = k as f32;
            let base = u32::try_from(vertices.len()).unwrap_or(u32::MAX);
            vertices.push(Vec3::new(-1.0, -1.0, z));
            vertices.push(Vec3::new(1.0, -1.0, z));
            vertices.push(Vec3::new(1.0, 1.0, z));
            vertices.push(Vec3::new(-1.0, 1.0, z));
            indices.push([base, base + 1, base + 2]);
            indices.push([base, base + 2, base + 3]);
        }
        let mesh = Trimesh::new(vertices, indices);
        let sweep = CapsuleSweep::new(
            Vec3::new(0.3, 0.1, 20.0),
            Vec3::new(0.7, 0.1, 20.0),
            Vec3::new(0.0, 0.0, -1.0),
            100.0,
            0.25,
        );
        let brute = cpu_trimesh_capsule_sweep(&mesh, &sweep).expect("hits");
        let bvh = cpu_trimesh_capsule_sweep_built(&mesh, &sweep).expect("hits");
        assert_eq!(brute.triangle, bvh.triangle, "same winning triangle");
        assert!((brute.toi - bvh.toi).abs() < 1e-4, "toi brute {} bvh {}", brute.toi, bvh.toi);
        // Nearest quad is z = 7; the underside stops one radius early, so
        // toi = 20 - 7 - 0.25 = 12.75.
        assert!((brute.toi - 12.75).abs() < 1e-3, "toi was {}", brute.toi);
        assert_eq!(brute.triangle, 14, "nearest lower-right triangle wins");
    }

    #[test]
    fn empty_mesh_misses() {
        let mesh = Trimesh::new(Vec::new(), Vec::new());
        let sweep = CapsuleSweep::new(
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::X,
            100.0,
            0.5,
        );
        assert!(cpu_trimesh_capsule_sweep(&mesh, &sweep).is_none());
        let lbvh = crate::cpu_build_lbvh(&mesh.triangle_aabbs());
        assert!(cpu_trimesh_capsule_sweep_bvh(&mesh, &lbvh, &sweep).is_none());
    }
}
