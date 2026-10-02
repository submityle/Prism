//! Swept-oriented-box-versus-triangle-mesh scene query: the first triangle a
//! moving oriented box (OBB) touches as it translates rigidly along a ray
//! through a static [`Trimesh`], reported as a time of impact, the contact point
//! on the mesh surface, and the push-out normal that drives the box off that
//! surface.
//!
//! This completes the mesh-query sweep family alongside the sphere and capsule
//! sweeps: an oriented box is the standard proxy for crates, platforms, and
//! vehicle hulls, and casting one against the triangle soup `AAA` engines use
//! for static level geometry is how such bodies resolve translation and
//! landing. It is the primitive behind `PhysX` `PxMeshQuery::sweep` with a box,
//! Jolt `MeshShape::CastShape` with a box, and Unreal `Chaos` triangle-mesh box
//! sweeps.
//!
//! # Reduction to conservative advancement, shared across paths and the device
//!
//! An oriented box is a convex polytope with no rounding, so sweeping it against
//! a triangle is a continuous collision query between two sharp convex cores:
//! the moving box and the static triangle. Rather than enumerate the many
//! closed-form face/edge/vertex cases (the moving edge-edge families are the
//! error-prone ones), every path runs the same conservative-advancement time of
//! impact that `PhysX` and Jolt use for shape casts:
//! [`conservative_advancement_toi_rounded`](crate::narrowphase::conservative_advancement_toi_rounded)
//! repeatedly measures the `GJK` separation of the box and triangle cores,
//! bounds how fast the moving box can close that gap, and advances time by the
//! largest provably safe step until the surfaces touch. Both cores carry a zero
//! convex radius, so the rounded cast degenerates to the sharp box-triangle
//! sweep. Because the advance never overshoots the first contact, the reported
//! impact is the earliest, and because the `CPU` paths and the `GPU` kernel run
//! the identical `GJK` and advance arithmetic, their hits agree to the
//! floating-point tolerance the parity suite allows.
//!
//! # Two interchangeable `CPU` paths
//!
//! * [`cpu_trimesh_obb_sweep`] is the brute-force golden: it sweeps the box
//!   against every triangle and keeps the earliest contact. It is the
//!   independent oracle the accelerated path and the `GPU` twin are pinned to.
//! * [`cpu_trimesh_obb_sweep_bvh`] descends the `LBVH` built from
//!   [`Trimesh::triangle_aabbs`], expands each node box by the box's bounding
//!   radius (the distance from its centre to a corner), and prunes any subtree
//!   the box-centre ray enters only later than the earliest contact found so
//!   far. The box is contained in a ball of that radius about its centre, so the
//!   grown-box entry time is a lower bound on any contained triangle's time of
//!   impact and the prune never discards a nearer triangle.
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

/// A swept oriented box for a mesh query: the box centre and orientation at the
/// start of the sweep, its half extents along each local axis, a travel
/// direction, and a maximum travel distance.
///
/// The whole box translates rigidly along `centre + t * direction` (there is no
/// rotation during the sweep), so the orientation stays fixed throughout. The
/// [`direction`](ObbSweep::direction) should be a unit vector so the reported
/// [`ObbSweepHit::toi`] is a world-space length; a non-unit direction scales the
/// reported time of impact by the direction's length.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ObbSweep {
    /// World-space centre of the box at the sweep start.
    pub centre: Vec3,
    /// World-space orientation of the box; local x/y/z axes are `rotation * X`,
    /// `rotation * Y`, `rotation * Z`.
    pub rotation: Quat,
    /// Half the box extent along each local axis; must be non-negative.
    pub half_extents: Vec3,
    /// Sweep direction; should be unit length for a metric time of impact.
    pub direction: Vec3,
    /// Largest `t` a contact may lie at; triangles touched later are rejected.
    pub max_distance: f32,
}

impl ObbSweep {
    /// Creates a sweep from the box `centre`, `rotation`, `half_extents`,
    /// `direction`, and `max_distance`.
    #[must_use]
    pub fn new(
        centre: Vec3,
        rotation: Quat,
        half_extents: Vec3,
        direction: Vec3,
        max_distance: f32,
    ) -> ObbSweep {
        ObbSweep {
            centre,
            rotation,
            half_extents,
            direction,
            max_distance,
        }
    }

    /// The radius of the ball about the box centre that contains the box: the
    /// distance from the centre to any corner, i.e. the half-extent vector's
    /// length (rotation-invariant).
    #[must_use]
    fn bounding_radius(&self) -> f32 {
        self.half_extents.length()
    }
}

/// A single triangle-mesh box-sweep contact.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ObbSweepHit {
    /// Zero-based index of the triangle first touched, addressing the mesh in
    /// its original (unsorted) triangle order.
    pub triangle: u32,
    /// Time of impact: the sweep parameter `t` at first contact, a world-space
    /// length for a unit direction.
    pub toi: f32,
    /// World-space contact point, the midpoint of the two closest surface
    /// witnesses at the time of impact.
    pub point: Vec3,
    /// Unit push-out normal pointing from the triangle surface toward the box,
    /// matching the contact convention of the rounded shape cast.
    pub normal: Vec3,
}

/// The shared single-triangle swept-box solve, used by the brute path, the
/// `BVH` path, and mirrored by the `GPU` kernel.
///
/// Models the box as a sharp (zero-radius) convex core placed at the sweep
/// centre with the sweep rotation and translating along the sweep direction, and
/// the triangle as a static zero-radius core, then runs the rounded
/// conservative-advancement time of impact between them. Returns the time of
/// impact, the contact point, and the push-out normal (from the triangle toward
/// the box) when the moving box first touches the triangle within
/// `[0, max_distance]`, or [`None`] on a miss. An initial overlap is reported at
/// a time of impact of zero.
#[must_use]
pub(crate) fn sweep_obb_triangle(
    sweep: &ObbSweep,
    a: Vec3,
    b: Vec3,
    c: Vec3,
) -> Option<(f32, Vec3, Vec3)> {
    // Box core: the eight local corners of the half-extent box, posed at the
    // sweep centre with the sweep rotation so the local frame maps onto the OBB,
    // translating along the sweep direction with no spin.
    let obb = ConvexHull::from_box(sweep.half_extents);
    let obb_pose = ConvexPose::new(sweep.centre, sweep.rotation);
    let obb_motion = BodyMotion::new(sweep.direction, Vec3::ZERO);

    // Triangle core: the three world corners, static, zero convex radius.
    let triangle = ConvexHull::from_triangle(a, b, c);
    let triangle_pose = ConvexPose::new(Vec3::ZERO, Quat::IDENTITY);
    let triangle_motion = BodyMotion::still();

    let toi = conservative_advancement_toi_rounded(
        &obb,
        &obb_pose,
        &obb_motion,
        0.0,
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
pub(crate) fn closer_hit(candidate: &ObbSweepHit, best: &Option<ObbSweepHit>) -> bool {
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
/// Sweeps the box against every triangle and keeps the earliest; ties resolve to
/// the lowest triangle index. Returns [`None`] when the box touches no triangle
/// within range or the mesh is empty. This is the independent golden the
/// accelerated and device paths are pinned to.
#[must_use]
pub fn cpu_trimesh_obb_sweep(mesh: &Trimesh, sweep: &ObbSweep) -> Option<ObbSweepHit> {
    let mut best: Option<ObbSweepHit> = None;
    for i in 0..mesh.triangle_count() {
        let tri = mesh.triangle(i);
        let Some((toi, point, normal)) = sweep_obb_triangle(sweep, tri.a, tri.b, tri.c) else {
            continue;
        };
        let hit = ObbSweepHit {
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

/// The entry distance where the box-centre ray enters `aabb` grown by the box's
/// bounding radius `r`, or [`None`] on a miss.
///
/// The box is contained in a ball of radius `r` about its centre, so when its
/// surface touches a triangle the centre is within `r` of that triangle and
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

/// Accelerated form of [`cpu_trimesh_obb_sweep`]: descends the `LBVH` over the
/// mesh's per-triangle boxes and prunes any subtree the box-centre ray enters
/// only later than the earliest contact found so far.
///
/// `lbvh` must be the hierarchy built from `mesh.triangle_aabbs()`; its leaf
/// slots map back to original triangle indices through
/// [`Lbvh::sorted_indices`]. The result matches [`cpu_trimesh_obb_sweep`]
/// exactly whenever the earliest contact is unique. Returns [`None`] when the box
/// touches no triangle within range or the tree is empty.
#[must_use]
pub fn cpu_trimesh_obb_sweep_bvh(
    mesh: &Trimesh,
    lbvh: &Lbvh,
    sweep: &ObbSweep,
) -> Option<ObbSweepHit> {
    if lbvh.num_leaves == 0 {
        return None;
    }
    let origin = sweep.centre;
    let r = sweep.bounding_radius();
    let mut best: Option<ObbSweepHit> = None;
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
            let Some((toi, point, normal)) = sweep_obb_triangle(sweep, tri.a, tri.b, tri.c) else {
                continue;
            };
            let hit = ObbSweepHit {
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
/// form of [`cpu_trimesh_obb_sweep_bvh`] for callers that do not cache the tree.
#[must_use]
pub fn cpu_trimesh_obb_sweep_built(mesh: &Trimesh, sweep: &ObbSweep) -> Option<ObbSweepHit> {
    let lbvh = cpu_build_lbvh(&mesh.triangle_aabbs());
    cpu_trimesh_obb_sweep_bvh(mesh, &lbvh, sweep)
}

#[cfg(test)]
mod tests {
    use super::{
        cpu_trimesh_obb_sweep, cpu_trimesh_obb_sweep_built, ObbSweep, ObbSweepHit,
    };
    use crate::collider::Trimesh;
    use glam::{Quat, Vec3};

    /// A unit quad on the `z = 0` plane spanning `[-1, 1]` in `x` and `y`, wound
    /// so its outward normal is `+z`. Two triangles: lower-right (0) and
    /// upper-left (1).
    fn unit_quad() -> Trimesh {
        Trimesh::new(
            vec![
                Vec3::new(-1.0, -1.0, 0.0),
                Vec3::new(1.0, -1.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(-1.0, 1.0, 0.0),
            ],
            vec![[0, 1, 2], [0, 2, 3]],
        )
    }

    #[test]
    fn axis_aligned_box_lands_flat_on_the_quad() {
        let mesh = unit_quad();
        // A 0.5-half-extent box centred high above the plane, dropped straight
        // down. Its bottom face reaches z = 0 when the centre is at z = 0.5, so
        // toi = 5 - 0.5 = 4.5.
        let sweep = ObbSweep::new(
            Vec3::new(0.0, 0.0, 5.0),
            Quat::IDENTITY,
            Vec3::splat(0.5),
            Vec3::new(0.0, 0.0, -1.0),
            100.0,
        );
        let hit = cpu_trimesh_obb_sweep(&mesh, &sweep).expect("lands on the quad");
        assert!((hit.toi - 4.5).abs() < 1e-3, "toi was {}", hit.toi);
        // A sharp (zero-radius) box lands with its face on the plane, so the
        // solver's final step is the intersecting branch, whose witness is the
        // box centre rather than a surface point; the physically meaningful
        // quantities are the time of impact and the push-out normal.
        assert!((hit.normal - Vec3::Z).length() < 1e-3, "normal points up at the box");
    }

    #[test]
    fn rotated_box_lands_on_its_lower_corner() {
        let mesh = unit_quad();
        // The same box rotated 45 degrees about y. Its downward extent grows to
        // 0.5 * (|cos45| + |sin45|) = 0.5 * sqrt(2) ~= 0.70711, so the lower edge
        // reaches z = 0 when the centre is at that height: toi = 5 - 0.70711.
        let sweep = ObbSweep::new(
            Vec3::new(0.0, 0.0, 5.0),
            Quat::from_axis_angle(Vec3::Y, std::f32::consts::FRAC_PI_4),
            Vec3::splat(0.5),
            Vec3::new(0.0, 0.0, -1.0),
            100.0,
        );
        let hit = cpu_trimesh_obb_sweep(&mesh, &sweep).expect("lands on the quad");
        let expected = 5.0 - 0.5 * 2.0_f32.sqrt();
        assert!((hit.toi - expected).abs() < 1e-3, "toi was {} expected {}", hit.toi, expected);
        // Intersecting-branch contact (see the flat-landing test); assert the
        // time of impact and the upward normal only.
        assert!((hit.normal - Vec3::Z).length() < 1e-3, "normal points up at the box");
    }

    #[test]
    fn misses_when_aimed_away() {
        let mesh = unit_quad();
        let sweep = ObbSweep::new(
            Vec3::new(0.0, 0.0, 5.0),
            Quat::IDENTITY,
            Vec3::splat(0.5),
            Vec3::new(0.0, 0.0, 1.0),
            100.0,
        );
        assert!(
            cpu_trimesh_obb_sweep(&mesh, &sweep).is_none(),
            "sweeps away from the quad"
        );
    }

    #[test]
    fn respects_max_distance() {
        let mesh = unit_quad();
        // Contact is at toi 4.5; a max of 2 cannot reach it.
        let sweep = ObbSweep::new(
            Vec3::new(0.0, 0.0, 5.0),
            Quat::IDENTITY,
            Vec3::splat(0.5),
            Vec3::new(0.0, 0.0, -1.0),
            2.0,
        );
        assert!(
            cpu_trimesh_obb_sweep(&mesh, &sweep).is_none(),
            "contact beyond max distance"
        );
    }

    #[test]
    fn initial_overlap_reports_zero_toi() {
        let mesh = unit_quad();
        // A 0.5-half-extent box hovering only 0.3 above the plane already
        // overlaps the quad: the contact is immediate.
        let sweep = ObbSweep::new(
            Vec3::new(0.0, 0.0, 0.3),
            Quat::IDENTITY,
            Vec3::splat(0.5),
            Vec3::new(0.0, 0.0, -1.0),
            100.0,
        );
        let hit = cpu_trimesh_obb_sweep(&mesh, &sweep).expect("already overlapping");
        assert!(hit.toi.abs() < 1e-6, "toi was {}", hit.toi);
    }

    #[test]
    fn sweeps_onto_edge_from_the_side() {
        let mesh = unit_quad();
        // A box to the right of the quad's right edge (x = 1), swept in -x. The
        // face plane is parallel to the sweep so the quad's edge catches the
        // box: the centre stops one half-extent (0.3) from the edge line after
        // travelling 1.5 - 1.0 - 0.3 = 0.2.
        let sweep = ObbSweep::new(
            Vec3::new(1.5, 0.0, 0.0),
            Quat::IDENTITY,
            Vec3::splat(0.3),
            Vec3::new(-1.0, 0.0, 0.0),
            100.0,
        );
        let hit = cpu_trimesh_obb_sweep(&mesh, &sweep).expect("hits the right edge");
        assert!((hit.toi - 0.2).abs() < 1e-3, "toi was {}", hit.toi);
        // The box face meets the quad edge sharply (intersecting branch), so the
        // reported witness is the box centre; the time of impact and the +x
        // push-out normal are the invariants under test.
        assert!((hit.normal - Vec3::X).length() < 1e-3, "normal points back along +x");
    }

    #[test]
    fn bvh_matches_brute_on_nearest_of_many() {
        // A fan of parallel quads at increasing z; the nearest to a high-z origin
        // must win on both paths.
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
        // Keep the box fully inside the lower-right (y < x) half of the quad so
        // one triangle is the unambiguous winner on both paths; a box centred on
        // the diagonal would touch both triangles of a quad at the same instant.
        let sweep = ObbSweep::new(
            Vec3::new(0.4, -0.4, 20.0),
            Quat::IDENTITY,
            Vec3::splat(0.25),
            Vec3::new(0.0, 0.0, -1.0),
            100.0,
        );
        let brute = cpu_trimesh_obb_sweep(&mesh, &sweep).expect("hits");
        let bvh = cpu_trimesh_obb_sweep_built(&mesh, &sweep).expect("hits");
        assert_eq!(brute.triangle, bvh.triangle, "same winning triangle");
        assert_eq!(brute.triangle, 14, "nearest lower-right triangle wins");
        assert!((brute.toi - bvh.toi).abs() < 1e-4, "toi brute {} bvh {}", brute.toi, bvh.toi);
        // Nearest quad is z = 7; the underside stops one half-extent early, so
        // toi = 20 - 7 - 0.25 = 12.75.
        assert!((brute.toi - 12.75).abs() < 1e-3, "toi was {}", brute.toi);
    }

    #[test]
    fn empty_mesh_misses() {
        let mesh = Trimesh::new(Vec::new(), Vec::new());
        let sweep = ObbSweep::new(
            Vec3::new(0.0, 0.0, 1.0),
            Quat::IDENTITY,
            Vec3::splat(0.5),
            Vec3::X,
            100.0,
        );
        assert!(cpu_trimesh_obb_sweep(&mesh, &sweep).is_none());
        let lbvh = crate::cpu_build_lbvh(&mesh.triangle_aabbs());
        assert!(super::cpu_trimesh_obb_sweep_bvh(&mesh, &lbvh, &sweep).is_none());
    }

    // Keep the hit type exercised through a direct field read so the public
    // surface stays covered even if a path stops returning it.
    #[test]
    fn hit_fields_are_readable() {
        let hit = ObbSweepHit {
            triangle: 3,
            toi: 1.0,
            point: Vec3::ZERO,
            normal: Vec3::Z,
        };
        assert_eq!(hit.triangle, 3);
    }
}
