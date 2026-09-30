//! Sphere-versus-OBB contact geometry, shared bit-for-bit with the `WGSL` kernel.
//!
//! [`sphere_obb_contact`] is the single source of truth for the sphere-against
//! oriented-bounding-box collision test and manifold construction; the `CPU`
//! twin ([`cpu_obb_narrowphase`]) and the device kernel
//! (`shaders/narrowphase_obb.wgsl`) both run this exact arithmetic so their
//! contacts agree to within the floating-point tolerance the parity test allows
//! (only the square root and the reciprocal in the outside-face normalisation
//! differ in their low bits).
//!
//! # Geometry
//!
//! An [`Obb`] is a centre `center`, three orthonormal local axes `axes` (the
//! box's local x/y/z), and per-axis half extents `half_extents`. A sphere is a
//! world centre `c` and radius `r`.
//!
//! The test works in the box's local frame. The sphere centre is projected onto
//! each axis, `local = (dot(d, axes[0]), dot(d, axes[1]), dot(d, axes[2]))` with
//! `d = c - center`, then clamped to the box, `q = clamp(local, -he, he)`. The
//! offset `diff = local - q` is the local vector from the nearest box point to
//! the sphere centre.
//!
//! * **Centre outside the box** (`dot(diff, diff) > `[`INSIDE_EPS2`]): the
//!   distance is `dist = |diff|`; the pair contacts only when `dist < r`
//!   (strict, so a grazing `dist == r` is *not* a contact). The local normal is
//!   `diff / dist`, the penetration is `depth = r - dist`, and the nearest box
//!   point is `q`.
//! * **Centre inside the box** (`dot(diff, diff) <= `[`INSIDE_EPS2`]): the
//!   sphere centre has sunk inside, so `diff` is degenerate. The exit face is
//!   the one with the smallest penetration `pen[i] = he[i] - |local[i]|`; ties
//!   resolve to the lower axis index. Along that axis `k` the local normal is
//!   `sign(local[k])` (with `local[k] >= 0` taken as positive), the penetration
//!   is `depth = r + pen[k]` (the centre must be pushed out `r` plus its depth
//!   below the face), and the nearest box point sets `q[k] = sign * he[k]`.
//!
//! The world normal recombines the local normal through the axes,
//! `axes[0] * n_local.x + axes[1] * n_local.y + axes[2] * n_local.z`, and the
//! world contact point is `center + axes[0] * q.x + axes[1] * q.y + axes[2] * q.z`,
//! the closest point on the box surface.
//!
//! # Normal convention
//!
//! The contact normal points **from the box surface toward the sphere**, i.e.
//! the direction that pushes the sphere out of the box. The reported
//! [`Contact`] stores the sphere index in `a` and the box index in `b`; unlike
//! the sphere-sphere manifold (whose normal runs `a` to `b`), this pair type's
//! normal therefore runs `b` (box) to `a` (sphere), matching the solver's
//! push-out convention for a dynamic sphere against a static box.
//!
//! # Per-operation agreement
//!
//! Every step above is performed in the identical order on both paths: the same
//! dot products, the same `clamp`, the same squared-length branch against
//! [`INSIDE_EPS2`], the same strict `dist < r` overlap test, the same
//! smallest-penetration axis search with lower-index tie-breaking, and the same
//! axis recombination. Only `sqrt` and the single reciprocal in `diff / dist`
//! are inexact, so the parity test matches the validity flag exactly and the
//! normal, depth, and point within a tight tolerance.
//!
//! Provenance: textbook sphere-versus-oriented-bounding-box collision manifold;
//! no Unreal Engine source or derived code.

use glam::{Quat, Vec3};

use super::contact::Contact;
use crate::broadphase::Particle;

/// Squared-length threshold below which the clamped offset `diff` is treated as
/// zero, i.e. the sphere centre is inside the box and the inside-face branch
/// runs. Kept identical to the `WGSL` constant so both paths pick the inside
/// branch on the same pairs.
pub(crate) const INSIDE_EPS2: f32 = 1.0e-12;

/// An oriented bounding box: a centre, three orthonormal local axes, and the
/// half extent along each of those axes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Obb {
    /// World-space centre of the box.
    pub center: Vec3,
    /// The box's local x/y/z axes; assumed orthonormal.
    pub axes: [Vec3; 3],
    /// Half the box's extent along each local axis; must be non-negative.
    pub half_extents: Vec3,
}

impl Obb {
    /// Creates an oriented bounding box from an explicit centre, axis triple,
    /// and half extents.
    #[must_use]
    pub fn new(center: Vec3, axes: [Vec3; 3], half_extents: Vec3) -> Obb {
        Obb {
            center,
            axes,
            half_extents,
        }
    }

    /// Creates an oriented bounding box whose axes are the world axes rotated by
    /// `rotation`, a convenient constructor for a rigid box with an orientation
    /// quaternion.
    ///
    /// The local x/y/z axes are `rotation * Vec3::X`, `rotation * Vec3::Y`, and
    /// `rotation * Vec3::Z`, which stay orthonormal for a unit quaternion.
    #[must_use]
    pub fn from_quat(center: Vec3, rotation: Quat, half_extents: Vec3) -> Obb {
        Obb {
            center,
            axes: [rotation * Vec3::X, rotation * Vec3::Y, rotation * Vec3::Z],
            half_extents,
        }
    }
}

/// A candidate sphere-versus-box pair: an index into the sphere (particle) slice
/// and an index into the box slice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SphereObbPair {
    /// Index of the sphere (bounding-sphere particle) in the sphere slice.
    pub sphere: u32,
    /// Index of the oriented bounding box in the box slice.
    pub obb: u32,
}

impl SphereObbPair {
    /// Creates a sphere-versus-box candidate pair.
    #[must_use]
    pub fn new(sphere: u32, obb: u32) -> SphereObbPair {
        SphereObbPair { sphere, obb }
    }
}

/// Tests whether the sphere penetrates the box and, if so, builds the contact.
///
/// Returns [`Some`] with the manifold for the pair `(sphere_id, obb_id)` when
/// the sphere overlaps the box, or [`None`] when it is separated or exactly
/// touching (a grazing `dist == r` on the outside branch). The arithmetic
/// mirrors `narrowphase_obb.wgsl` operation for operation; see the module
/// documentation for the geometry and the normal convention.
#[must_use]
pub(crate) fn sphere_obb_contact(
    sphere_id: u32,
    obb_id: u32,
    c: Vec3,
    r: f32,
    box_: &Obb,
) -> Option<Contact> {
    let a0 = box_.axes[0];
    let a1 = box_.axes[1];
    let a2 = box_.axes[2];
    let he = box_.half_extents;

    // Project the sphere centre into the box's local frame.
    let d = c - box_.center;
    let local = Vec3::new(d.dot(a0), d.dot(a1), d.dot(a2));
    // Nearest local point on (or in) the box, then the offset back to the centre.
    let q = local.clamp(-he, he);
    let diff = local - q;
    let d2 = diff.dot(diff);

    if d2 > INSIDE_EPS2 {
        // Centre is outside the box: nearest feature is the clamped point `q`.
        let dist = d2.sqrt();
        // Strict overlap: a sphere exactly touching the surface carries no
        // penetration, so it is not a contact.
        if dist >= r {
            return None;
        }
        let n_local = diff / dist;
        let normal = a0 * n_local.x + a1 * n_local.y + a2 * n_local.z;
        let depth = r - dist;
        let point = box_.center + a0 * q.x + a1 * q.y + a2 * q.z;
        Some(Contact::new(sphere_id, obb_id, normal, depth, point))
    } else {
        // Centre is inside the box: exit through the least-penetrated face.
        let pen = he - local.abs();
        let mut k = 0usize;
        let mut min_pen = pen.x;
        if pen.y < min_pen {
            min_pen = pen.y;
            k = 1;
        }
        if pen.z < min_pen {
            min_pen = pen.z;
            k = 2;
        }

        let comp = match k {
            0 => local.x,
            1 => local.y,
            _ => local.z,
        };
        // `local[k] >= 0` exits through the positive face, otherwise negative.
        let sign = if comp >= 0.0 { 1.0 } else { -1.0 };

        let mut n_local = Vec3::ZERO;
        let mut q_in = local;
        match k {
            0 => {
                n_local.x = sign;
                q_in.x = sign * he.x;
            }
            1 => {
                n_local.y = sign;
                q_in.y = sign * he.y;
            }
            _ => {
                n_local.z = sign;
                q_in.z = sign * he.z;
            }
        }

        let normal = a0 * n_local.x + a1 * n_local.y + a2 * n_local.z;
        let depth = r + min_pen;
        let point = box_.center + a0 * q_in.x + a1 * q_in.y + a2 * q_in.z;
        Some(Contact::new(sphere_id, obb_id, normal, depth, point))
    }
}

/// `CPU` golden twin of the sphere-versus-OBB narrow phase.
///
/// Turns a set of candidate `pairs` into contact manifolds, running the shared
/// [`sphere_obb_contact`] geometry so its output matches the device kernel
/// operation for operation. It emits one slot per input pair, preserving order,
/// which is what lets the parity test line the `GPU` contacts up against this
/// reference index by index: [`Some`] carrying the manifold when the sphere
/// penetrates the box, or [`None`] when it is separated or exactly touching. A
/// later scan stage compacts the survivors.
///
/// # Panics
///
/// Panics if a pair references a sphere or box index outside the corresponding
/// slice, which is never valid output from a broad phase over the same sets.
#[must_use]
pub fn cpu_obb_narrowphase(
    spheres: &[Particle],
    boxes: &[Obb],
    pairs: &[SphereObbPair],
) -> Vec<Option<Contact>> {
    pairs
        .iter()
        .map(|pair| {
            let sphere = &spheres[pair.sphere as usize];
            let box_ = &boxes[pair.obb as usize];
            sphere_obb_contact(pair.sphere, pair.obb, sphere.position, sphere.radius, box_)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Quat;

    /// A unit-axis box at the origin with the given half extents.
    fn axis_box(he: Vec3) -> Obb {
        Obb::new(Vec3::ZERO, [Vec3::X, Vec3::Y, Vec3::Z], he)
    }

    #[test]
    fn face_contact_axis_aligned() {
        // Sphere just past the +x face of a unit box: nearest point is the face
        // centre (1, 0, 0), depth = r - dist = 0.7 - 0.5 = 0.2.
        let box_ = axis_box(Vec3::ONE);
        let c = sphere_obb_contact(0, 0, Vec3::new(1.5, 0.0, 0.0), 0.7, &box_)
            .expect("sphere overlapping the +x face must contact");
        assert_eq!(c.a, 0);
        assert_eq!(c.b, 0);
        assert!((c.normal - Vec3::X).length() < 1.0e-6);
        assert!((c.depth - 0.2).abs() < 1.0e-6);
        assert!((c.point - Vec3::new(1.0, 0.0, 0.0)).length() < 1.0e-6);
    }

    #[test]
    fn edge_contact_axis_aligned() {
        // Sphere off the +x/+y edge: nearest point is (1, 1, 0), diff = (0.5,
        // 0.5, 0), dist = sqrt(0.5), normal along the diagonal.
        let box_ = axis_box(Vec3::ONE);
        let c = sphere_obb_contact(3, 5, Vec3::new(1.5, 1.5, 0.0), 0.8, &box_)
            .expect("sphere overlapping the edge must contact");
        assert_eq!(c.a, 3);
        assert_eq!(c.b, 5);
        let dist = 0.5f32.sqrt();
        let want_normal = Vec3::new(0.5, 0.5, 0.0) / dist;
        assert!((c.normal - want_normal).length() < 1.0e-6);
        assert!((c.normal.length() - 1.0).abs() < 1.0e-6);
        assert!((c.depth - (0.8 - dist)).abs() < 1.0e-6);
        assert!((c.point - Vec3::new(1.0, 1.0, 0.0)).length() < 1.0e-6);
    }

    #[test]
    fn corner_contact_axis_aligned() {
        // Sphere off the +x/+y/+z corner: nearest point is (1, 1, 1), diff =
        // (0.4, 0.4, 0.4), dist = sqrt(0.48).
        let box_ = axis_box(Vec3::ONE);
        let c = sphere_obb_contact(1, 2, Vec3::new(1.4, 1.4, 1.4), 0.8, &box_)
            .expect("sphere overlapping the corner must contact");
        let dist = 0.48f32.sqrt();
        let want_normal = Vec3::new(0.4, 0.4, 0.4) / dist;
        assert!((c.normal - want_normal).length() < 1.0e-6);
        assert!((c.normal.length() - 1.0).abs() < 1.0e-6);
        assert!((c.depth - (0.8 - dist)).abs() < 1.0e-6);
        assert!((c.point - Vec3::ONE).length() < 1.0e-6);
    }

    #[test]
    fn centre_inside_pushes_out_least_penetrated_face() {
        // Centre inside a 2x2x2 box at (0.5, 0, 0): pen = (1.5, 2, 2), least on
        // the x axis, so it exits through +x. depth = r + pen[0] = 0.3 + 1.5.
        let box_ = axis_box(Vec3::splat(2.0));
        let c = sphere_obb_contact(7, 4, Vec3::new(0.5, 0.0, 0.0), 0.3, &box_)
            .expect("a sphere centred inside always contacts");
        assert_eq!(c.a, 7);
        assert_eq!(c.b, 4);
        assert!((c.normal - Vec3::X).length() < 1.0e-6);
        assert!((c.depth - 1.8).abs() < 1.0e-6);
        // Nearest box point overrides the x component to +he.x = 2.
        assert!((c.point - Vec3::new(2.0, 0.0, 0.0)).length() < 1.0e-6);
    }

    #[test]
    fn rotated_box_contact() {
        // Box rotated 45 degrees about z: its +x-pointing vertical edge sits at
        // world (sqrt(2), 0, 0). A sphere at (1.6, 0, 0) just clears that edge.
        let rot = Quat::from_rotation_z(core::f32::consts::FRAC_PI_4);
        let box_ = Obb::from_quat(Vec3::ZERO, rot, Vec3::ONE);
        let c = sphere_obb_contact(0, 0, Vec3::new(1.6, 0.0, 0.0), 0.7, &box_)
            .expect("sphere clipping the rotated edge must contact");
        // World normal recombines to +x; both local halves cancel off-axis.
        assert!((c.normal - Vec3::X).length() < 1.0e-5);
        assert!((c.normal.length() - 1.0).abs() < 1.0e-5);
        let want_dist = 1.6 - 2.0f32.sqrt();
        assert!((c.depth - (0.7 - want_dist)).abs() < 1.0e-5);
        assert!((c.point - Vec3::new(2.0f32.sqrt(), 0.0, 0.0)).length() < 1.0e-5);
    }

    #[test]
    fn clear_separation_reports_no_contact() {
        // Sphere far from the box on +x: dist 4, radius 1, no overlap.
        let box_ = axis_box(Vec3::ONE);
        assert!(sphere_obb_contact(0, 0, Vec3::new(5.0, 0.0, 0.0), 1.0, &box_).is_none());
    }

    #[test]
    fn exactly_touching_reports_no_contact() {
        // Sphere exactly grazing the +x face: dist == r, strict test rejects.
        let box_ = axis_box(Vec3::ONE);
        assert!(sphere_obb_contact(0, 0, Vec3::new(2.0, 0.0, 0.0), 1.0, &box_).is_none());
    }

    #[test]
    fn batch_preserves_order_and_indices() {
        // Two spheres and one box: a clear overlap then a clear gap, checked in
        // input order with their pair indices intact.
        let spheres = [
            Particle::new(Vec3::new(1.5, 0.0, 0.0), 0.7),
            Particle::new(Vec3::new(6.0, 0.0, 0.0), 0.5),
        ];
        let boxes = [axis_box(Vec3::ONE)];
        let pairs = [SphereObbPair::new(0, 0), SphereObbPair::new(1, 0)];
        let contacts = cpu_obb_narrowphase(&spheres, &boxes, &pairs);
        assert_eq!(contacts.len(), 2);
        let first = contacts[0].expect("first sphere overlaps");
        assert_eq!(first.a, 0);
        assert_eq!(first.b, 0);
        assert!((first.normal - Vec3::X).length() < 1.0e-6);
        assert!(contacts[1].is_none());
    }
}
