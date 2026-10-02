//! Expanding-polytope algorithm (`EPA`): recovers the penetration normal and
//! depth of two overlapping convex hulls from the terminal `GJK` simplex.
//!
//! When [`gjk`](super::gjk::gjk) reports [`GjkStatus::Intersecting`](super::gjk::GjkStatus::Intersecting)
//! the Minkowski difference `A (-) B` contains the origin, but the enclosing
//! simplex only certifies overlap; it does not say by how much. `EPA` grows that
//! simplex into a polytope that approximates the surface of the Minkowski
//! difference, always expanding toward the face nearest the origin. The nearest
//! face converges onto the point of the difference's boundary closest to the
//! origin: its outward normal is the minimum-translation direction and its
//! distance is the penetration depth.
//!
//! # Method
//!
//! The simplex is first blown up to a non-degenerate tetrahedron (adding support
//! points along fresh directions when `GJK` terminated on a vertex, edge, or
//! triangle). Each iteration then
//!
//! 1. picks the polytope face whose supporting plane is closest to the origin,
//! 2. queries the Minkowski support along that face's outward normal, and
//! 3. stops when the new support lies on the face's plane (within tolerance),
//!    otherwise carves away every face the new point can "see" and re-triangulates
//!    the resulting horizon so the polytope strictly grows.
//!
//! Barycentric weights of the origin's projection onto the final face split the
//! contact back into a world-space witness on each body, exactly as `GJK` does
//! for the separated case.
//!
//! Provenance: expanding-polytope algorithm (van den Bergen, *Proximity Queries
//! and Penetration Depth Computation on 3D Game Objects*, 2001), horizon
//! re-triangulation after Ericson (2005). No Unreal Engine source or derived code.

use glam::Vec3;

use super::convex_hull::ConvexHull;
use super::convex_pose::ConvexPose;
use super::minkowski::{support, SupportPoint};

/// Squared length below which a vector is treated as degenerate.
const ZERO_EPS2: f32 = 1.0e-12;

/// Growth tolerance: expansion stops when a new support advances the closest
/// face's plane by less than this absolute distance.
const GROWTH_TOL: f32 = 1.0e-4;

/// Hard cap on expansion iterations so a pathological hull terminates.
const MAX_ITERS: u32 = 64;

/// Sentinel distance for a degenerate (zero-area) face so it is never chosen as
/// the closest face.
const DEGENERATE_DISTANCE: f32 = 1.0e30;

/// The penetration recovered for a pair of overlapping convex hulls.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Penetration {
    /// Unit minimum-translation normal pointing from `B` toward `A`: translating
    /// `A` by `normal * depth` just separates the pair.
    pub normal: Vec3,
    /// Penetration depth: the overlap measured along `normal`.
    pub depth: f32,
    /// World-space contact witness on hull `A`.
    pub point_a: Vec3,
    /// World-space contact witness on hull `B`.
    pub point_b: Vec3,
}

/// A triangular face of the expanding polytope: vertex indices wound CCW as seen
/// from outside, its outward unit normal, and the distance from the origin to
/// its supporting plane.
#[derive(Clone, Copy)]
struct Face {
    indices: [usize; 3],
    normal: Vec3,
    distance: f32,
}

/// Expands the `GJK` `simplex` (one to four Minkowski-difference support points
/// enclosing the origin) into the penetration normal and depth of the pair.
///
/// Returns [`None`] only when the overlap is so degenerate that a non-flat
/// tetrahedron cannot be built (exact surface tangency); callers may treat that
/// as a zero-depth touching contact.
#[must_use]
pub fn epa(
    hull_a: &ConvexHull,
    pose_a: &ConvexPose,
    hull_b: &ConvexHull,
    pose_b: &ConvexPose,
    simplex: &[SupportPoint],
) -> Option<Penetration> {
    let mut verts = blow_up_to_tetrahedron(hull_a, pose_a, hull_b, pose_b, simplex)?.to_vec();
    let mut faces = initial_faces(&verts)?;

    let mut best = closest_face(&faces)?;
    for _ in 0..MAX_ITERS {
        let normal = faces[best].normal;
        let w = support(hull_a, pose_a, hull_b, pose_b, normal);
        let reach = w.diff.dot(normal);
        // Converged: the support cannot push the closest plane out any further.
        if reach - faces[best].distance < GROWTH_TOL {
            break;
        }
        // Reject a support already in the polytope (no progress possible).
        if verts
            .iter()
            .any(|v| (v.diff - w.diff).length_squared() < ZERO_EPS2)
        {
            break;
        }
        let w_index = verts.len();
        verts.push(w);
        carve_and_patch(&mut faces, &verts, w.diff, w_index);
        best = match closest_face(&faces) {
            Some(b) => b,
            None => break,
        };
    }

    Some(resolve(&verts, &faces[best]))
}

/// Builds the four outward-wound faces of the seed tetrahedron, or [`None`] when
/// the four vertices are coplanar (degenerate).
fn initial_faces(verts: &[SupportPoint]) -> Option<Vec<Face>> {
    let centroid = (verts[0].diff + verts[1].diff + verts[2].diff + verts[3].diff) * 0.25;
    let tris = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]];
    let mut faces = Vec::with_capacity(4);
    for t in tris {
        faces.push(oriented_face(verts, t[0], t[1], t[2], centroid)?);
    }
    Some(faces)
}

/// Orients the triangle `(i, j, k)` so its stored winding and normal point away
/// from `interior`, returning [`None`] for a zero-area triangle.
fn oriented_face(
    verts: &[SupportPoint],
    i: usize,
    j: usize,
    k: usize,
    interior: Vec3,
) -> Option<Face> {
    let vi = verts[i].diff;
    let raw = (verts[j].diff - vi).cross(verts[k].diff - vi);
    if raw.length_squared() < ZERO_EPS2 {
        return None;
    }
    let n = raw.normalize();
    // Flip so the normal faces away from the interior reference point.
    if n.dot(interior - vi) > 0.0 {
        Some(Face {
            indices: [i, k, j],
            normal: -n,
            distance: (-n).dot(vi),
        })
    } else {
        Some(Face {
            indices: [i, j, k],
            normal: n,
            distance: n.dot(vi),
        })
    }
}

/// Index of the face whose supporting plane is nearest the origin, skipping
/// degenerate faces.
fn closest_face(faces: &[Face]) -> Option<usize> {
    faces
        .iter()
        .enumerate()
        .filter(|(_, f)| f.distance < DEGENERATE_DISTANCE)
        .min_by(|(_, a), (_, b)| a.distance.total_cmp(&b.distance))
        .map(|(i, _)| i)
}

/// Removes every face the new vertex `w` can see and stitches new faces from the
/// resulting horizon to `w`, preserving outward winding.
fn carve_and_patch(faces: &mut Vec<Face>, verts: &[SupportPoint], w: Vec3, w_index: usize) {
    // Collect the horizon as directed edges of visible faces, cancelling any
    // edge shared by two visible faces.
    let mut horizon: Vec<(usize, usize)> = Vec::new();
    let mut kept: Vec<Face> = Vec::with_capacity(faces.len());
    for f in faces.iter() {
        let v0 = verts[f.indices[0]].diff;
        let visible = f.normal.dot(w - v0) > 0.0;
        if visible {
            for e in edges_of(f.indices) {
                add_or_cancel_edge(&mut horizon, e);
            }
        } else {
            kept.push(*f);
        }
    }
    // Patch the horizon: each boundary edge spans a new triangle to {w}.
    for (a, bb) in horizon {
        let va = verts[a].diff;
        let raw = (verts[bb].diff - va).cross(w - va);
        if raw.length_squared() < ZERO_EPS2 {
            // Degenerate patch face: keep it with a sentinel so it is never
            // picked but still seals the polytope topologically.
            kept.push(Face {
                indices: [a, bb, w_index],
                normal: Vec3::ZERO,
                distance: DEGENERATE_DISTANCE,
            });
            continue;
        }
        let n = raw.normalize();
        // The directed horizon edge preserves CCW outward winding; guard the sign
        // against the origin to stay robust to near-degenerate growth.
        if n.dot(va) < 0.0 {
            kept.push(Face {
                indices: [a, w_index, bb],
                normal: -n,
                distance: (-n).dot(va),
            });
        } else {
            kept.push(Face {
                indices: [a, bb, w_index],
                normal: n,
                distance: n.dot(va),
            });
        }
    }
    *faces = kept;
}

/// The three directed edges of a CCW triangle.
fn edges_of(idx: [usize; 3]) -> [(usize, usize); 3] {
    [(idx[0], idx[1]), (idx[1], idx[2]), (idx[2], idx[0])]
}

/// Adds a directed edge to the horizon, or cancels it against the opposing edge
/// contributed by an adjacent visible face.
fn add_or_cancel_edge(horizon: &mut Vec<(usize, usize)>, e: (usize, usize)) {
    if let Some(pos) = horizon.iter().position(|&(x, y)| x == e.1 && y == e.0) {
        horizon.swap_remove(pos);
    } else {
        horizon.push(e);
    }
}

/// Reconstructs the world-space witnesses from the barycentric projection of the
/// origin onto the closest face.
fn resolve(verts: &[SupportPoint], face: &Face) -> Penetration {
    let a = verts[face.indices[0]];
    let b = verts[face.indices[1]];
    let c = verts[face.indices[2]];
    // The closest point of the plane to the origin is {normal * distance}.
    let proj = face.normal * face.distance;
    let (u, v, w) = barycentric(a.diff, b.diff, c.diff, proj);
    let point_a = a.on_a * u + b.on_a * v + c.on_a * w;
    let point_b = a.on_b * u + b.on_b * v + c.on_b * w;
    Penetration {
        // {face.normal} is the Minkowski-boundary outward normal, which points
        // from A toward B; the push-out that separates A from B is its negation.
        normal: -face.normal,
        depth: face.distance,
        point_a,
        point_b,
    }
}

/// Barycentric weights `(u, v, w)` of `p` on triangle `(a, b, c)`.
fn barycentric(a: Vec3, b: Vec3, c: Vec3, p: Vec3) -> (f32, f32, f32) {
    let v0 = b - a;
    let v1 = c - a;
    let v2 = p - a;
    let d00 = v0.dot(v0);
    let d01 = v0.dot(v1);
    let d11 = v1.dot(v1);
    let d20 = v2.dot(v0);
    let d21 = v2.dot(v1);
    let denom = (d00 * d11 - d01 * d01).max(ZERO_EPS2);
    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    (1.0 - v - w, v, w)
}

/// Grows a one-to-four-point simplex into a non-degenerate tetrahedron by adding
/// fresh support points, returning [`None`] when no non-flat tetrahedron exists.
fn blow_up_to_tetrahedron(
    hull_a: &ConvexHull,
    pose_a: &ConvexPose,
    hull_b: &ConvexHull,
    pose_b: &ConvexPose,
    simplex: &[SupportPoint],
) -> Option<[SupportPoint; 4]> {
    let mut pts: Vec<SupportPoint> = simplex.to_vec();
    let get = |dir: Vec3| support(hull_a, pose_a, hull_b, pose_b, dir);

    // 1 -> 2: find any direction yielding a distinct second point.
    if pts.len() == 1 {
        let axes = [Vec3::X, -Vec3::X, Vec3::Y, -Vec3::Y, Vec3::Z, -Vec3::Z];
        for d in axes {
            let s = get(d);
            if (s.diff - pts[0].diff).length_squared() > ZERO_EPS2 {
                pts.push(s);
                break;
            }
        }
        if pts.len() < 2 {
            return None;
        }
    }

    // 2 -> 3: search perpendicular to the segment for a non-collinear point.
    if pts.len() == 2 {
        let ab = pts[1].diff - pts[0].diff;
        let axis = least_aligned_axis(ab);
        let perp1 = ab.cross(axis);
        let perp2 = ab.cross(perp1);
        let mut best: Option<(f32, SupportPoint)> = None;
        for d in [perp1, -perp1, perp2, -perp2] {
            if d.length_squared() < ZERO_EPS2 {
                continue;
            }
            let s = get(d);
            let area = (s.diff - pts[0].diff).cross(ab).length_squared();
            if best.as_ref().is_none_or(|(ba, _)| area > *ba) {
                best = Some((area, s));
            }
        }
        let (area, s) = best?;
        if area < ZERO_EPS2 {
            return None;
        }
        pts.push(s);
    }

    // 3 -> 4: push off the triangle plane on whichever side reaches farther.
    if pts.len() == 3 {
        let n = (pts[1].diff - pts[0].diff).cross(pts[2].diff - pts[0].diff);
        if n.length_squared() < ZERO_EPS2 {
            return None;
        }
        let plus = get(n);
        let minus = get(-n);
        let off_plus = (plus.diff - pts[0].diff).dot(n).abs();
        let off_minus = (minus.diff - pts[0].diff).dot(n).abs();
        let s = if off_plus >= off_minus { plus } else { minus };
        if (s.diff - pts[0].diff).dot(n).abs() < ZERO_EPS2 {
            return None;
        }
        pts.push(s);
    }

    if pts.len() != 4 {
        return None;
    }
    // Reject a flat tetrahedron.
    let vol = (pts[1].diff - pts[0].diff)
        .cross(pts[2].diff - pts[0].diff)
        .dot(pts[3].diff - pts[0].diff);
    if vol.abs() < ZERO_EPS2 {
        return None;
    }
    Some([pts[0], pts[1], pts[2], pts[3]])
}

/// The world axis least aligned with `v`, for building a perpendicular.
fn least_aligned_axis(v: Vec3) -> Vec3 {
    let ax = v.x.abs();
    let ay = v.y.abs();
    let az = v.z.abs();
    if ax <= ay && ax <= az {
        Vec3::X
    } else if ay <= az {
        Vec3::Y
    } else {
        Vec3::Z
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::narrowphase::gjk::{gjk, GjkStatus};
    use glam::{Quat, Vec3};

    fn unit_box_at(x: f32, y: f32, z: f32) -> (ConvexHull, ConvexPose) {
        (
            ConvexHull::from_box(Vec3::splat(1.0)),
            ConvexPose::new(Vec3::new(x, y, z), Quat::IDENTITY),
        )
    }

    /// Runs GJK then EPA, requiring overlap, and returns the penetration.
    fn penetrate(
        a: &ConvexHull,
        pa: &ConvexPose,
        b: &ConvexHull,
        pb: &ConvexPose,
    ) -> Penetration {
        let simplex = match gjk(a, pa, b, pb) {
            GjkStatus::Intersecting(s) => s,
            GjkStatus::Separated { .. } => panic!("expected overlap"),
        };
        epa(a, pa, b, pb, &simplex).expect("non-degenerate penetration")
    }

    #[test]
    fn axis_overlap_reports_depth_normal_and_witnesses() {
        // A spans [0.5, 2.5] in x; B spans [-1, 1]; overlap depth is 0.5 along x.
        let (a, pa) = unit_box_at(1.5, 0.0, 0.0);
        let (b, pb) = unit_box_at(0.0, 0.0, 0.0);
        let pen = penetrate(&a, &pa, &b, &pb);
        assert!((pen.depth - 0.5).abs() < 1.0e-3, "depth {}", pen.depth);
        // Normal points from B toward A: +x.
        assert!((pen.normal - Vec3::X).length() < 1.0e-3, "normal {:?}", pen.normal);
        // Witness on A sits on its -x face (x = 0.5); on B on its +x face (x = 1).
        assert!((pen.point_a.x - 0.5).abs() < 1.0e-3, "point_a {:?}", pen.point_a);
        assert!((pen.point_b.x - 1.0).abs() < 1.0e-3, "point_b {:?}", pen.point_b);
    }

    #[test]
    fn y_axis_overlap_reports_depth_along_y() {
        let (a, pa) = unit_box_at(0.0, 1.5, 0.0);
        let (b, pb) = unit_box_at(0.0, 0.0, 0.0);
        let pen = penetrate(&a, &pa, &b, &pb);
        assert!((pen.depth - 0.5).abs() < 1.0e-3, "depth {}", pen.depth);
        assert!((pen.normal - Vec3::Y).length() < 1.0e-3, "normal {:?}", pen.normal);
    }

    #[test]
    fn concentric_cubes_report_full_half_width_depth() {
        // Coincident unit cubes: every face of the Minkowski cube ties at depth 2.
        let (a, pa) = unit_box_at(0.0, 0.0, 0.0);
        let (b, pb) = unit_box_at(0.0, 0.0, 0.0);
        let pen = penetrate(&a, &pa, &b, &pb);
        assert!((pen.depth - 2.0).abs() < 1.0e-3, "depth {}", pen.depth);
        // The chosen normal is axis-aligned and unit length.
        assert!((pen.normal.length() - 1.0).abs() < 1.0e-3, "normal {:?}", pen.normal);
        let m = pen.normal.x.abs().max(pen.normal.y.abs()).max(pen.normal.z.abs());
        assert!((m - 1.0).abs() < 1.0e-3, "normal {:?}", pen.normal);
    }

    #[test]
    fn shallow_offset_normal_points_toward_a() {
        // A nudged a little along +x: the push-out still points from B toward A.
        let (a, pa) = unit_box_at(0.2, 0.0, 0.0);
        let (b, pb) = unit_box_at(0.0, 0.0, 0.0);
        let pen = penetrate(&a, &pa, &b, &pb);
        // Depth along x is (half+half) - offset = 2 - 0.2 ... minus overlap width;
        // Minkowski x-span is [-1.8, 2.2], nearest face at 1.8.
        assert!((pen.depth - 1.8).abs() < 1.0e-3, "depth {}", pen.depth);
        let to_a = pa.translation - pb.translation;
        assert!(pen.normal.dot(to_a) > 0.0, "normal {:?}", pen.normal);
    }

    #[test]
    fn rotated_overlapping_box_recovers_positive_depth() {
        // A turned 45 degrees about z and overlapping B at the origin: EPA must
        // still return a unit normal and a strictly positive depth.
        let a = ConvexHull::from_box(Vec3::splat(1.0));
        let pa = ConvexPose::new(
            Vec3::new(0.8, 0.0, 0.0),
            Quat::from_rotation_z(core::f32::consts::FRAC_PI_4),
        );
        let (b, pb) = unit_box_at(0.0, 0.0, 0.0);
        let pen = penetrate(&a, &pa, &b, &pb);
        assert!(pen.depth > 0.0, "depth {}", pen.depth);
        assert!((pen.normal.length() - 1.0).abs() < 1.0e-3, "normal {:?}", pen.normal);
        // A sits to the +x of B, so the push-out has a positive x component.
        assert!(pen.normal.x > 0.0, "normal {:?}", pen.normal);
    }
}
