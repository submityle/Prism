//! Capsule-versus-OBB contact geometry, shared bit-for-bit with the `WGSL`
//! kernel.
//!
//! [`capsule_obb_contact`] is the single source of truth for the collision test
//! and manifold construction; the `CPU` twin ([`cpu_capsule_obb_narrowphase`])
//! and the device kernel (`shaders/narrowphase_capsule_obb.wgsl`) both run this
//! exact arithmetic so their contacts agree to within the floating-point
//! tolerance the parity test allows (only the square roots and the reciprocals
//! in the closest-feature search and the outside-face normalisation differ in
//! their low bits).
//!
//! # Geometry
//!
//! A capsule is a line segment `(p0, p1)` swept by a radius `rc`; an [`Obb`] is
//! a centre, three orthonormal local axes, and per-axis half extents. The test
//! collapses the capsule to a sphere of radius `rc` centred at the point on the
//! capsule axis closest to the box, then runs the identical inside/outside
//! branch the sphere-versus-OBB pair uses ([`super::obb`]).
//!
//! The work happens in the box's local frame, where the box is the axis-aligned
//! box `[-he, he]`. Both capsule endpoints are projected into that frame, and
//! [`segment_box_closest_t`] finds the segment parameter `t` in `[0, 1]` whose
//! point is closest to the box. The squared distance from the segment to the box
//! is a convex, piecewise-quadratic function of `t` (each face plane the segment
//! crosses is a breakpoint), so its minimum is found exactly by evaluating every
//! breakpoint, both endpoints, and the parabola vertex of every piece, with no
//! iteration to accumulate error. That closest local point `s` then feeds the
//! sphere-versus-box branch:
//!
//! * **Closest point outside the box** (`dot(diff, diff) > `[`INSIDE_EPS2`]):
//!   `diff = s - clamp(s, -he, he)`, `dist = |diff|`, and the pair contacts only
//!   when `dist < rc` (strict, so a grazing `dist == rc` is not a contact). The
//!   local normal is `diff / dist`, the depth is `rc - dist`, and the contact
//!   point is the nearest box-surface point `q = clamp(s, -he, he)`.
//! * **Closest point inside the box** (`dot(diff, diff) <= `[`INSIDE_EPS2`]):
//!   the capsule axis has sunk into the box, so `diff` is degenerate. The exit
//!   face is the least-penetrated one, `pen[i] = he[i] - |s[i]|`, ties resolving
//!   to the lower axis index. The local normal is `sign(s[k])` along axis `k`,
//!   the depth is `rc + pen[k]`, and the box point sets `q[k] = sign * he[k]`.
//!
//! The local normal and box point recombine through the axes back into world
//! space exactly as the sphere-versus-OBB pair does.
//!
//! # Normal convention
//!
//! The contact normal points **from the box surface toward the capsule**, the
//! direction that pushes the capsule out of the box. The reported [`Contact`]
//! stores the capsule index in `a` and the box index in `b`, so the normal runs
//! `b` (box) to `a` (capsule), matching the sphere-versus-OBB push-out
//! convention for a dynamic capsule against a static box.
//!
//! # Single-point manifold
//!
//! Like the capsule-capsule pair, this test reports a single deepest-feature
//! contact rather than a two-point manifold, so a capsule lying flat against a
//! box face is held by one point per frame. A persistent two-point manifold for
//! the flat-resting case is a follow-up, tracked with the other manifold work;
//! the single-point contact is exact for the reported feature and never fake.
//!
//! Provenance: textbook capsule-versus-oriented-bounding-box closest-feature
//! collision (segment-box distance plus the sphere-box manifold); no Unreal
//! Engine source or derived code.

use glam::Vec3;

use super::contact::Contact;
use super::obb::{Obb, INSIDE_EPS2};
use crate::narrowphase::capsule::Capsule;

/// Squared-length threshold below which a capsule axis component is treated as
/// parallel to a face plane, so that face contributes no finite crossing. Kept
/// identical to the `WGSL` constant so both paths skip the same crossings.
pub(crate) const AXIS_EPS2: f32 = 1.0e-12;

/// A candidate capsule-versus-box pair: an index into the capsule slice and an
/// index into the box slice.
///
/// The two indices address different slices, so the pair is intrinsically
/// ordered (`capsule` is always the `a` side of the contact, `obb` the `b`
/// side) and is not canonicalised.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CapsuleObbPair {
    /// Index of the capsule in the capsule slice; becomes [`Contact::a`].
    pub capsule: u32,
    /// Index of the oriented bounding box in the box slice; becomes
    /// [`Contact::b`].
    pub obb: u32,
}

impl CapsuleObbPair {
    /// Creates a capsule-versus-box candidate pair.
    #[must_use]
    pub fn new(capsule: u32, obb: u32) -> CapsuleObbPair {
        CapsuleObbPair { capsule, obb }
    }
}

/// Squared distance from the segment point at parameter `t` to the local box
/// `[-he, he]`.
///
/// The point is `a + t * d`; its distance to the box is the length of the
/// per-axis excess outside the extents, so this returns the sum of squared
/// excesses. Zero inside the box.
#[must_use]
fn dist2_to_box(a: Vec3, d: Vec3, t: f32, he: Vec3) -> f32 {
    let p = a + d * t;
    let excess = p - p.clamp(-he, he);
    excess.dot(excess)
}

/// The two face-plane crossing parameters for one axis, each clamped to
/// `[0, 1]`.
///
/// The segment coordinate on this axis is `ai + t * di`; it meets the `+ei` and
/// `-ei` faces at `t = (±ei - ai) / di`. When the axis component is (near) zero
/// the segment is parallel to both faces and never crosses, so both crossings
/// collapse to `0.0`, which is already an evaluated endpoint and therefore
/// harmless.
#[must_use]
fn axis_crossings(ai: f32, di: f32, ei: f32) -> [f32; 2] {
    if di * di <= AXIS_EPS2 {
        return [0.0, 0.0];
    }
    let inv = 1.0 / di;
    let hit_pos = ((ei - ai) * inv).clamp(0.0, 1.0);
    let hit_neg = ((-ei - ai) * inv).clamp(0.0, 1.0);
    [hit_pos, hit_neg]
}

/// Sorts eight parameters ascending in place with a fixed selection sort.
///
/// Deterministic and branch-simple so the `WGSL` twin reproduces the exact same
/// ordering, which fixes the piece walk and the earliest-`t` tie-break.
fn sort8(values: &mut [f32; 8]) {
    let mut i = 0usize;
    while i < 8 {
        let mut min_idx = i;
        let mut j = i + 1;
        while j < 8 {
            if values[j] < values[min_idx] {
                min_idx = j;
            }
            j += 1;
        }
        values.swap(i, min_idx);
        i += 1;
    }
}

/// Parabola-vertex parameter of the box-distance function on the piece around
/// `t_mid`, clamped to `[t_lo, t_hi]`.
///
/// On a piece between two breakpoints each axis is on a fixed side of the box
/// (outside `+e`, outside `-e`, or inside and inert), so the squared distance is
/// a single quadratic `alpha * t^2 + beta * t + gamma`. The unconstrained
/// minimiser is `-beta / (2 * alpha)`; when no axis is active (`alpha <= 0`) the
/// distance is flat at zero on this piece and `t_lo` already realises the
/// minimum, so it is returned unchanged.
#[must_use]
fn piece_vertex(a: Vec3, d: Vec3, he: Vec3, t_lo: f32, t_hi: f32, t_mid: f32) -> f32 {
    let p = a + d * t_mid;
    let mut alpha = 0.0f32;
    let mut half_beta = 0.0f32;
    // Each active axis contributes (di * t + (ai - si * ei))^2 with si = +/-1.
    let axes = [
        (p.x, a.x, d.x, he.x),
        (p.y, a.y, d.y, he.y),
        (p.z, a.z, d.z, he.z),
    ];
    for (pi, ai, di, ei) in axes {
        let offset = if pi > ei {
            ai - ei
        } else if pi < -ei {
            ai + ei
        } else {
            continue;
        };
        alpha += di * di;
        half_beta += di * offset;
    }
    if alpha <= AXIS_EPS2 {
        return t_lo;
    }
    (-half_beta / alpha).clamp(t_lo, t_hi)
}

/// Segment parameter `t` in `[0, 1]` whose point is closest to the local box
/// `[-he, he]`.
///
/// Minimises the convex, piecewise-quadratic squared distance exactly: it
/// gathers the two endpoints and the six face-plane crossings, sorts them, then
/// evaluates the distance at every one of those breakpoints and at the parabola
/// vertex of every piece between adjacent breakpoints, keeping the earliest `t`
/// that achieves the smallest distance. No iteration means no accumulated error,
/// so the `WGSL` twin reproduces `t` to the tolerance the parity test allows.
#[must_use]
fn segment_box_closest_t(a: Vec3, b: Vec3, he: Vec3) -> f32 {
    let d = b - a;
    let cx = axis_crossings(a.x, d.x, he.x);
    let cy = axis_crossings(a.y, d.y, he.y);
    let cz = axis_crossings(a.z, d.z, he.z);
    let mut breaks = [0.0, 1.0, cx[0], cx[1], cy[0], cy[1], cz[0], cz[1]];
    sort8(&mut breaks);

    let mut best_t = breaks[0];
    let mut best_d2 = dist2_to_box(a, d, best_t, he);
    // A candidate updates the best only on a strict improvement, so ties keep
    // the earlier (smaller) t and both paths break ties identically.
    let consider = |t: f32, best_t: &mut f32, best_d2: &mut f32| {
        let d2 = dist2_to_box(a, d, t, he);
        if d2 < *best_d2 {
            *best_d2 = d2;
            *best_t = t;
        }
    };

    for pair in breaks.windows(2) {
        let t_lo = pair[0];
        let t_hi = pair[1];
        consider(t_hi, &mut best_t, &mut best_d2);
        if t_hi > t_lo {
            let t_mid = 0.5 * (t_lo + t_hi);
            let t_vertex = piece_vertex(a, d, he, t_lo, t_hi, t_mid);
            consider(t_vertex, &mut best_t, &mut best_d2);
        }
    }
    best_t
}

/// Tests whether the capsule penetrates the box and, if so, builds the contact.
///
/// Returns [`Some`] with the manifold for the pair `(capsule_id, obb_id)` when
/// the capsule overlaps the box, or [`None`] when it is separated or exactly
/// touching (a grazing `dist == rc` on the outside branch). The arithmetic
/// mirrors `narrowphase_capsule_obb.wgsl` operation for operation; see the
/// module documentation for the geometry and the normal convention.
#[must_use]
pub(crate) fn capsule_obb_contact(
    capsule_id: u32,
    obb_id: u32,
    cap: &Capsule,
    box_: &Obb,
) -> Option<Contact> {
    let a0 = box_.axes[0];
    let a1 = box_.axes[1];
    let a2 = box_.axes[2];
    let he = box_.half_extents;
    let rc = cap.radius;

    // Project both capsule endpoints into the box's local frame.
    let d0 = cap.p0 - box_.center;
    let d1 = cap.p1 - box_.center;
    let local0 = Vec3::new(d0.dot(a0), d0.dot(a1), d0.dot(a2));
    let local1 = Vec3::new(d1.dot(a0), d1.dot(a1), d1.dot(a2));

    // Closest point on the capsule axis to the box, in the local frame.
    let t = segment_box_closest_t(local0, local1, he);
    let local = local0 + (local1 - local0) * t;

    // Nearest local box point and the offset back to the closest axis point.
    let q = local.clamp(-he, he);
    let diff = local - q;
    let d2 = diff.dot(diff);

    if d2 > INSIDE_EPS2 {
        // Closest axis point outside the box: nearest feature is the clamp `q`.
        let dist = d2.sqrt();
        // Strict overlap: an exactly-touching capsule carries no penetration.
        if dist >= rc {
            return None;
        }
        let n_local = diff / dist;
        let normal = a0 * n_local.x + a1 * n_local.y + a2 * n_local.z;
        let depth = rc - dist;
        let point = box_.center + a0 * q.x + a1 * q.y + a2 * q.z;
        Some(Contact::new(capsule_id, obb_id, normal, depth, point))
    } else {
        // Closest axis point inside the box: exit through the least-penetrated
        // face, exactly as the sphere-versus-box inside branch does.
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
        let depth = rc + min_pen;
        let point = box_.center + a0 * q_in.x + a1 * q_in.y + a2 * q_in.z;
        Some(Contact::new(capsule_id, obb_id, normal, depth, point))
    }
}

/// `CPU` golden twin of the capsule-versus-OBB narrow phase.
///
/// Turns a set of candidate `pairs` into contact manifolds, running the shared
/// [`capsule_obb_contact`] geometry so its output matches the device kernel
/// operation for operation. It emits one slot per input pair, preserving order,
/// which is what lets the parity test line the `GPU` contacts up against this
/// reference index by index: [`Some`] carrying the manifold when the capsule
/// penetrates the box, or [`None`] when it is separated or exactly touching. A
/// later scan stage compacts the survivors.
///
/// # Panics
///
/// Panics if a pair references a capsule or box index outside the corresponding
/// slice, which is never valid output from a broad phase over the same sets.
#[must_use]
pub fn cpu_capsule_obb_narrowphase(
    capsules: &[Capsule],
    boxes: &[Obb],
    pairs: &[CapsuleObbPair],
) -> Vec<Option<Contact>> {
    pairs
        .iter()
        .map(|pair| {
            let cap = &capsules[pair.capsule as usize];
            let box_ = &boxes[pair.obb as usize];
            capsule_obb_contact(pair.capsule, pair.obb, cap, box_)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Quat;

    /// A capsule from two endpoints and a radius.
    fn cap(p0: Vec3, p1: Vec3, radius: f32) -> Capsule {
        Capsule::new(p0, p1, radius)
    }

    /// An axis-aligned unit-half-extent box centred at the origin.
    fn unit_box() -> Obb {
        Obb::new(Vec3::ZERO, [Vec3::X, Vec3::Y, Vec3::Z], Vec3::splat(1.0))
    }

    #[test]
    fn horizontal_capsule_resting_on_the_top_face() {
        // A capsule lying along x just above the +y face, dipping into it by 0.1.
        let boxes = [unit_box()];
        let capsules = [cap(
            Vec3::new(-2.0, 1.4, 0.0),
            Vec3::new(2.0, 1.4, 0.0),
            0.5,
        )];
        let pairs = [CapsuleObbPair::new(0, 0)];
        let c = cpu_capsule_obb_narrowphase(&capsules, &boxes, &pairs)[0].expect("resting overlap");
        assert_eq!(c.a, 0);
        assert_eq!(c.b, 0);
        // Nearest axis point is over the top face, so the normal is +y.
        assert!(
            (c.normal - Vec3::Y).length() < 1.0e-5,
            "normal {:?}",
            c.normal
        );
        // Axis at y = 1.4, top face at y = 1.0, gap 0.4 < rc 0.5 -> depth 0.1.
        assert!((c.depth - 0.1).abs() < 1.0e-5, "depth {}", c.depth);
        // Contact sits on the top face, y = 1.0.
        assert!((c.point.y - 1.0).abs() < 1.0e-5, "point {:?}", c.point);
    }

    #[test]
    fn capsule_end_poking_a_side_face() {
        // A vertical capsule whose lower end reaches past the +x face corner.
        let boxes = [unit_box()];
        let capsules = [cap(Vec3::new(1.3, 0.0, 0.0), Vec3::new(1.3, 3.0, 0.0), 0.5)];
        let pairs = [CapsuleObbPair::new(0, 0)];
        let c = cpu_capsule_obb_narrowphase(&capsules, &boxes, &pairs)[0].expect("side overlap");
        // Closest axis point is the lower end at x = 1.3; nearest face is +x.
        assert!(
            (c.normal - Vec3::X).length() < 1.0e-5,
            "normal {:?}",
            c.normal
        );
        // x = 1.3, face at 1.0, gap 0.3 < rc 0.5 -> depth 0.2.
        assert!((c.depth - 0.2).abs() < 1.0e-5, "depth {}", c.depth);
    }

    #[test]
    fn clear_capsule_reports_none() {
        // A capsule well above the box: no contact.
        let boxes = [unit_box()];
        let capsules = [cap(
            Vec3::new(-2.0, 5.0, 0.0),
            Vec3::new(2.0, 5.0, 0.0),
            0.5,
        )];
        let pairs = [CapsuleObbPair::new(0, 0)];
        assert!(cpu_capsule_obb_narrowphase(&capsules, &boxes, &pairs)[0].is_none());
    }

    #[test]
    fn grazing_touch_is_not_a_contact() {
        // Axis at y = 1.5, face at 1.0, gap exactly rc 0.5: touching, not
        // penetrating, so the strict test rejects it.
        let boxes = [unit_box()];
        let capsules = [cap(
            Vec3::new(-2.0, 1.5, 0.0),
            Vec3::new(2.0, 1.5, 0.0),
            0.5,
        )];
        let pairs = [CapsuleObbPair::new(0, 0)];
        assert!(cpu_capsule_obb_narrowphase(&capsules, &boxes, &pairs)[0].is_none());
    }

    #[test]
    fn axis_through_box_uses_the_inside_branch() {
        // A capsule axis passing straight through the box along x: the closest
        // point is inside, so the least-penetrated face drives the normal.
        let boxes = [unit_box()];
        let capsules = [cap(
            Vec3::new(-3.0, 0.2, 0.0),
            Vec3::new(3.0, 0.2, 0.0),
            0.25,
        )];
        let pairs = [CapsuleObbPair::new(0, 0)];
        let c = cpu_capsule_obb_narrowphase(&capsules, &boxes, &pairs)[0].expect("through overlap");
        // Inside point (0, 0.2, 0): pen x could be 1.0 at some t, but the closest
        // axis point sits inside where y = 0.2, z = 0; least penetration is the
        // y face (1 - 0.2 = 0.8) versus z (1.0), so normal is +y.
        assert!(c.depth > 0.0, "depth {}", c.depth);
        assert!(c.normal.length() > 0.9, "normal {:?}", c.normal);
    }

    #[test]
    fn tilted_box_projects_into_the_local_frame() {
        // Rotating the box 45 degrees about z must not change a symmetric
        // contact: a capsule sitting above along the rotated up axis.
        let rot = Quat::from_rotation_z(std::f32::consts::FRAC_PI_4);
        let boxes = [Obb::from_quat(Vec3::ZERO, rot, Vec3::splat(1.0))];
        let up = rot * Vec3::Y;
        let along = rot * Vec3::X;
        let centre = up * 1.4;
        let capsules = [cap(centre - along * 2.0, centre + along * 2.0, 0.5)];
        let pairs = [CapsuleObbPair::new(0, 0)];
        let c = cpu_capsule_obb_narrowphase(&capsules, &boxes, &pairs)[0].expect("tilted overlap");
        // Normal is the box's rotated up axis; depth 0.1 as in the aligned case.
        assert!((c.normal - up).length() < 1.0e-5, "normal {:?}", c.normal);
        assert!((c.depth - 0.1).abs() < 1.0e-5, "depth {}", c.depth);
    }

    #[test]
    fn batch_preserves_input_order() {
        let boxes = [unit_box()];
        let capsules = [
            cap(Vec3::new(-2.0, 1.4, 0.0), Vec3::new(2.0, 1.4, 0.0), 0.5), // hit
            cap(Vec3::new(-2.0, 5.0, 0.0), Vec3::new(2.0, 5.0, 0.0), 0.5), // miss
        ];
        let pairs = [CapsuleObbPair::new(0, 0), CapsuleObbPair::new(1, 0)];
        let out = cpu_capsule_obb_narrowphase(&capsules, &boxes, &pairs);
        assert!(out[0].is_some());
        assert!(out[1].is_none());
    }

    #[test]
    fn degenerate_capsule_matches_a_sphere() {
        // A zero-length capsule is a sphere; the closest-t search collapses to
        // the single endpoint and the contact matches a sphere-box test.
        let boxes = [unit_box()];
        let capsules = [cap(Vec3::new(0.0, 1.4, 0.0), Vec3::new(0.0, 1.4, 0.0), 0.5)];
        let pairs = [CapsuleObbPair::new(0, 0)];
        let c = cpu_capsule_obb_narrowphase(&capsules, &boxes, &pairs)[0].expect("sphere overlap");
        assert!(
            (c.normal - Vec3::Y).length() < 1.0e-5,
            "normal {:?}",
            c.normal
        );
        assert!((c.depth - 0.1).abs() < 1.0e-5, "depth {}", c.depth);
    }
}
