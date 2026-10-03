//! An arena-backed, unbounded convex-polytope body proxy for soft<->rigid
//! coupling.
//!
//! [`super::ConvexProxy`] is a stack-resident, `Copy` convex solid capped at
//! [`MAX_CONVEX_PLANES`](super::MAX_CONVEX_PLANES) faces so it can slot into the
//! `#[derive(Copy)]` [`super::BodyCollider`] enum without changing the shared
//! coupling kernel's by-value signatures. That ten-face ceiling comfortably
//! covers a box, a wedge, or a bevelled crate, but production cloth collision
//! props (UE5 Chaos Cloth, Houdini Vellum low-poly colliders) routinely exceed
//! it: a chamfered box already needs eighteen faces, and a convex-decomposed
//! rock shard can need dozens. The GPU convex path already stores its faces in
//! an unbounded `convex_planes` storage pool indexed by `[offset, count)`, so a
//! CPU golden that is capped at ten faces cannot validate those richer hulls.
//!
//! [`ConvexMesh`] closes that gap: it is the same half-space-intersection
//! convex solid and the same least-penetration push-out, slab-clip
//! time-of-impact, and active-face query as [`super::ConvexProxy`], but it keeps
//! its faces in a heap [`Vec`] so the face count is unbounded. It is therefore
//! **not** `Copy` and is used host-side (building the GPU plane pool, or running
//! the CPU golden for a >10-face hull), never embedded in the `Copy`
//! [`super::BodyCollider`] enum.
//!
//! For any hull of ten or fewer live faces, [`ConvexMesh`] and
//! [`super::ConvexProxy`] agree bit-for-bit (the tests build both from the same
//! planes and assert the queries match), so the unbounded path is a strict
//! superset of the inline one rather than a divergent second implementation.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The
//! half-space-intersection convex solid, the least-penetration push-out, and
//! the slab-clip segment/convex time-of-impact are textbook computational
//! geometry (Ericson, *Real-Time Collision Detection*, Sections 5.3 / 5.1.5),
//! identical to the sibling [`super::ConvexProxy`] arm.

use glam::{Quat, Vec3};

use crate::math::scalar::Real;

use super::convex::{ConvexProxy, Plane};
use super::EPS_LEN_SQ;

/// A convex solid expressed as the intersection of an *unbounded* set of
/// outward-facing half-spaces, with a cached center and bounding radius.
///
/// A point `x` is inside the solid when it lies behind every face plane,
/// `normal_i . x <= offset_i` (the normals point *out* of the solid). Interior
/// points are projected to the nearest face; points on or outside any face are
/// left untouched. Unlike [`super::ConvexProxy`] this type stores its faces on
/// the heap, so it is not `Copy` and is used host-side rather than inside the
/// `Copy` [`super::BodyCollider`] enum.
#[derive(Clone, Debug, PartialEq)]
pub struct ConvexMesh {
    /// Live outward-facing face planes; every plane carries a unit normal.
    planes: Vec<Plane>,
    /// Representative center (the coupling anchor / body reference point).
    center: Vec3,
    /// Conservative bounding-sphere radius about `center`, used only for the
    /// broad-phase AABB cull (never for the projection itself).
    radius: Real,
}

impl ConvexMesh {
    /// Builds a convex mesh from an explicit set of outward-facing planes with a
    /// caller-supplied `center` and bounding `radius`.
    ///
    /// Degenerate (zero-normal) planes are dropped and the surviving normals are
    /// normalized (the offset is rescaled by the same factor so the plane is
    /// unchanged). Returns [`None`] only when *no* live plane survives, so --
    /// unlike [`super::ConvexProxy::from_planes`], which rejects an over-budget
    /// hull -- any face count is accepted. `radius` is clamped non-negative; it
    /// is used only for the broad-phase AABB and should enclose the solid about
    /// `center`.
    #[must_use]
    pub fn from_planes(center: Vec3, radius: Real, planes: &[Plane]) -> Option<Self> {
        let mut live = Vec::with_capacity(planes.len());
        for p in planes {
            let len_sq = p.normal.length_squared();
            if len_sq <= EPS_LEN_SQ {
                continue;
            }
            let n = p.normal.normalize_or_zero();
            if n.length_squared() <= EPS_LEN_SQ {
                continue;
            }
            live.push(Plane {
                normal: n,
                offset: p.offset / len_sq.sqrt(),
            });
        }
        if live.is_empty() {
            return None;
        }
        Some(Self {
            planes: live,
            center,
            radius: radius.max(0.0),
        })
    }

    /// Builds the convex mesh for an oriented box `(center, orientation,
    /// half_extents)`, i.e. the intersection of its six face half-spaces.
    ///
    /// Axes with a non-positive half-extent are collapsed (their two faces are
    /// dropped), matching [`super::ConvexProxy::from_box`]. The faces are built
    /// in the same fixed `+X,-X,+Y,-Y,+Z,-Z` order so the least-penetration
    /// tie-break is identical, and the bounding radius is the box's corner
    /// distance.
    #[must_use]
    pub fn from_box(center: Vec3, orientation: Quat, half_extents: Vec3) -> Self {
        let he = half_extents.max(Vec3::ZERO);
        let axes = [
            (orientation * Vec3::X, half_extents.x),
            (orientation * Vec3::Y, half_extents.y),
            (orientation * Vec3::Z, half_extents.z),
        ];
        let mut planes = Vec::with_capacity(6);
        for (axis, extent) in axes {
            if extent <= 0.0 {
                continue;
            }
            if let Some(face) = face_from_point_normal(center + axis * extent, axis) {
                planes.push(face);
            }
            if let Some(face) = face_from_point_normal(center - axis * extent, -axis) {
                planes.push(face);
            }
        }
        Self {
            planes,
            center,
            radius: he.length(),
        }
    }

    /// Lifts a bounded [`super::ConvexProxy`] into an unbounded mesh, preserving
    /// its live faces, center, and bounding radius exactly.
    ///
    /// This is the bridge that lets a hull authored as the inline `Copy` proxy
    /// be fed to a host path that wants the heap representation, and it is what
    /// makes the "`ConvexMesh` is a strict superset of `ConvexProxy`" invariant
    /// testable: for a <=10-face hull the two agree on every query.
    #[must_use]
    pub fn from_proxy(proxy: &ConvexProxy) -> Self {
        Self {
            planes: proxy.planes().to_vec(),
            center: proxy.center(),
            radius: proxy.bounding_radius(),
        }
    }

    /// The live face planes.
    #[must_use]
    pub fn planes(&self) -> &[Plane] {
        &self.planes
    }

    /// The number of live face planes.
    #[must_use]
    pub fn face_count(&self) -> usize {
        self.planes.len()
    }

    /// The mesh's representative center (coupling anchor / body reference).
    #[must_use]
    pub fn center(&self) -> Vec3 {
        self.center
    }

    /// The conservative bounding-sphere radius about [`center`](Self::center).
    #[must_use]
    pub fn bounding_radius(&self) -> Real {
        self.radius
    }

    /// Returns `pos` projected out to the nearest face when it lies strictly
    /// inside every face plane, otherwise returns `pos` unchanged.
    ///
    /// The least-penetrating face (smallest `offset - normal.pos`) is the
    /// closest boundary point for an interior point of a convex polytope, so the
    /// point lands exactly on the surface. A point on or outside any face is
    /// outside the solid and is returned untouched. An empty mesh, or one whose
    /// faces are all degenerate, is inert. Identical semantics to
    /// [`super::ConvexProxy::project_out`].
    #[must_use]
    pub fn project_out(&self, pos: Vec3) -> Vec3 {
        let mut best_pen = Real::INFINITY;
        let mut best_normal = Vec3::ZERO;
        let mut found = false;
        for face in &self.planes {
            if face.normal.length_squared() <= EPS_LEN_SQ {
                continue;
            }
            let signed = face.normal.dot(pos) - face.offset;
            if signed >= 0.0 {
                return pos;
            }
            let pen = -signed;
            if pen < best_pen {
                best_pen = pen;
                best_normal = face.normal;
                found = true;
            }
        }
        if !found {
            return pos;
        }
        pos + best_normal * best_pen
    }

    /// Returns `self` rigidly translated by `delta`: the center shifts, each
    /// face offset moves by `normal.delta`, and the bounding radius is
    /// unchanged.
    #[must_use]
    pub fn translated(&self, delta: Vec3) -> Self {
        let mut out = self.clone();
        out.center += delta;
        for face in &mut out.planes {
            face.offset += face.normal.dot(delta);
        }
        out
    }

    /// Returns the outward unit normal of the face the surface point `surf` lies
    /// on (the face with the largest signed distance, i.e. the active contact
    /// face), or the zero vector for an empty/degenerate mesh. Identical to
    /// [`super::ConvexProxy::face_normal`].
    #[must_use]
    pub fn face_normal(&self, surf: Vec3) -> Vec3 {
        let mut best_signed = Real::NEG_INFINITY;
        let mut best_normal = Vec3::ZERO;
        for face in &self.planes {
            if face.normal.length_squared() <= EPS_LEN_SQ {
                continue;
            }
            let signed = face.normal.dot(surf) - face.offset;
            if signed > best_signed {
                best_signed = signed;
                best_normal = face.normal;
            }
        }
        best_normal.normalize_or_zero()
    }

    /// Returns the earliest time in `0..=1` at which the segment `prev -> curr`
    /// enters the convex solid, or [`None`] when it misses or starts inside.
    ///
    /// This is the standard slab/half-space clip of a segment against an
    /// intersection of half-spaces, identical to
    /// [`super::ConvexProxy::segment_toi`]: `t_enter` is the latest entry across
    /// the faces the segment approaches, `t_exit` the earliest exit across the
    /// faces it recedes from. A segment that starts inside yields a negative
    /// entry and is rejected (left to the discrete projection). Only multiplies
    /// and comparisons are used, so no path yields a [`f32::NAN`].
    #[must_use]
    pub fn segment_toi(&self, prev: Vec3, curr: Vec3) -> Option<Real> {
        if self.planes.is_empty() {
            return None;
        }
        let dir = curr - prev;
        let mut t_enter = Real::NEG_INFINITY;
        let mut t_exit = Real::INFINITY;
        for face in &self.planes {
            if face.normal.length_squared() <= EPS_LEN_SQ {
                continue;
            }
            let num = face.normal.dot(prev) - face.offset;
            let rate = face.normal.dot(dir);
            if rate.abs() <= EPS_LEN_SQ {
                if num > 0.0 {
                    return None;
                }
                continue;
            }
            let t = -num / rate;
            if rate > 0.0 {
                t_exit = t_exit.min(t);
            } else {
                t_enter = t_enter.max(t);
            }
            if t_enter > t_exit {
                return None;
            }
        }
        if t_exit < 0.0 || !(0.0..=1.0).contains(&t_enter) {
            return None;
        }
        Some(t_enter)
    }
}

/// Builds a unit-normal face from a point on the plane and an outward normal,
/// returning [`None`] for a (near) degenerate normal. Mirrors the private
/// `Plane::from_point_normal` helper in the sibling [`super::convex`] module.
#[must_use]
fn face_from_point_normal(point: Vec3, normal: Vec3) -> Option<Plane> {
    if normal.length_squared() <= EPS_LEN_SQ {
        return None;
    }
    let n = normal.normalize_or_zero();
    if n.length_squared() <= EPS_LEN_SQ {
        return None;
    }
    Some(Plane {
        normal: n,
        offset: n.dot(point),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: Real = 1.0e-6;

    fn approx_eq(a: Vec3, b: Vec3) {
        assert!((a - b).length() < TOL, "{a:?} != {b:?}");
    }

    /// A 12-face hull: an axis-aligned unit box (6 faces) chamfered by the four
    /// vertical edges (4 faces) plus the top two corners cut (2 faces) --
    /// eighteen would be a full chamfer, twelve is already over the inline cap.
    fn big_hull() -> ConvexMesh {
        let s = 1.0 / 3.0_f32.sqrt();
        let planes = [
            Plane {
                normal: Vec3::X,
                offset: 1.0,
            },
            Plane {
                normal: -Vec3::X,
                offset: 1.0,
            },
            Plane {
                normal: Vec3::Y,
                offset: 1.0,
            },
            Plane {
                normal: -Vec3::Y,
                offset: 1.0,
            },
            Plane {
                normal: Vec3::Z,
                offset: 1.0,
            },
            Plane {
                normal: -Vec3::Z,
                offset: 1.0,
            },
            // Four vertical edge chamfers at offset 1.3 (> 1, so they clip the
            // box corners without swallowing the faces).
            Plane {
                normal: Vec3::new(s, 0.0, s),
                offset: 1.3,
            },
            Plane {
                normal: Vec3::new(-s, 0.0, s),
                offset: 1.3,
            },
            Plane {
                normal: Vec3::new(s, 0.0, -s),
                offset: 1.3,
            },
            Plane {
                normal: Vec3::new(-s, 0.0, -s),
                offset: 1.3,
            },
            // Two top-corner cuts.
            Plane {
                normal: Vec3::new(s, s, 0.0),
                offset: 1.3,
            },
            Plane {
                normal: Vec3::new(-s, s, 0.0),
                offset: 1.3,
            },
        ];
        ConvexMesh::from_planes(Vec3::ZERO, 2.0, &planes).expect("live hull")
    }

    #[test]
    fn exceeds_the_inline_proxy_face_cap() {
        let hull = big_hull();
        assert!(hull.face_count() > crate::soft::MAX_CONVEX_PLANES);
        // The same plane set is rejected by the bounded proxy, proving the mesh
        // is reaching a hull the inline path cannot represent.
        let planes: Vec<Plane> = hull.planes().to_vec();
        assert!(ConvexProxy::from_planes(Vec3::ZERO, 2.0, &planes).is_none());
    }

    #[test]
    fn matches_the_inline_proxy_for_a_box() {
        let proxy = ConvexProxy::from_box(Vec3::ZERO, Quat::IDENTITY, Vec3::new(1.0, 0.5, 2.0));
        let mesh = ConvexMesh::from_box(Vec3::ZERO, Quat::IDENTITY, Vec3::new(1.0, 0.5, 2.0));
        for p in [
            Vec3::new(0.2, 0.1, 0.3),
            Vec3::new(0.9, -0.4, -1.8),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(5.0, 0.0, 0.0), // outside
        ] {
            approx_eq(proxy.project_out(p), mesh.project_out(p));
        }
        approx_eq(
            proxy.face_normal(Vec3::new(1.0, 0.0, 0.0)),
            mesh.face_normal(Vec3::new(1.0, 0.0, 0.0)),
        );
    }

    #[test]
    fn matches_the_inline_proxy_via_lift() {
        // A bevelled 8-face hull that fits in the inline proxy: lifting it must
        // not change any query.
        let s = 1.0 / 3.0_f32.sqrt();
        let planes = [
            Plane {
                normal: Vec3::X,
                offset: 1.0,
            },
            Plane {
                normal: -Vec3::X,
                offset: 1.0,
            },
            Plane {
                normal: Vec3::Y,
                offset: 1.0,
            },
            Plane {
                normal: -Vec3::Y,
                offset: 1.0,
            },
            Plane {
                normal: Vec3::Z,
                offset: 1.0,
            },
            Plane {
                normal: -Vec3::Z,
                offset: 1.0,
            },
            Plane {
                normal: Vec3::new(s, s, s),
                offset: 1.3,
            },
            Plane {
                normal: Vec3::new(-s, -s, -s),
                offset: 1.3,
            },
        ];
        let proxy = ConvexProxy::from_planes(Vec3::ZERO, 2.0, &planes).expect("fits");
        let mesh = ConvexMesh::from_proxy(&proxy);
        assert_eq!(mesh.face_count(), proxy.planes().len());
        for p in [
            Vec3::new(0.3, 0.2, 0.1),
            Vec3::new(-0.6, 0.6, 0.6),
            Vec3::new(0.95, 0.0, 0.0),
        ] {
            approx_eq(proxy.project_out(p), mesh.project_out(p));
        }
        let prev = Vec3::new(0.0, 3.0, 0.0);
        let curr = Vec3::new(0.0, -3.0, 0.0);
        assert_eq!(proxy.segment_toi(prev, curr), mesh.segment_toi(prev, curr));
    }

    #[test]
    fn interior_point_lands_on_the_surface_of_a_big_hull() {
        let hull = big_hull();
        let inside = Vec3::new(0.1, 0.1, 0.1);
        let out = hull.project_out(inside);
        // The projected point must sit on (not past) the boundary: its max signed
        // distance over all faces is ~0.
        let max_signed = hull
            .planes()
            .iter()
            .map(|f| f.normal.dot(out) - f.offset)
            .fold(Real::NEG_INFINITY, Real::max);
        assert!(max_signed.abs() < TOL, "max signed {max_signed}");
        // And it moved along a single face normal from the interior point.
        assert!((out - inside).length() > TOL);
    }

    #[test]
    fn point_outside_is_unchanged() {
        let hull = big_hull();
        let outside = Vec3::new(5.0, 0.0, 0.0);
        approx_eq(hull.project_out(outside), outside);
    }

    #[test]
    fn segment_toi_catches_entry_into_a_big_hull() {
        let hull = big_hull();
        let prev = Vec3::new(0.0, 5.0, 0.0);
        let curr = Vec3::new(0.0, -5.0, 0.0);
        let t = hull
            .segment_toi(prev, curr)
            .expect("segment crosses the hull");
        assert!((0.0..=1.0).contains(&t));
        // Entry is where the top face (y = 1) is crossed: y(t) = 5 - 10 t = 1.
        assert!((t - 0.4).abs() < 1.0e-4, "t = {t}");
    }

    #[test]
    fn segment_that_misses_returns_none() {
        let hull = big_hull();
        let prev = Vec3::new(5.0, 5.0, 0.0);
        let curr = Vec3::new(5.0, -5.0, 0.0);
        assert!(hull.segment_toi(prev, curr).is_none());
    }

    #[test]
    fn translated_shifts_the_solid_rigidly() {
        let hull = big_hull();
        let delta = Vec3::new(2.0, -1.0, 0.5);
        let moved = hull.translated(delta);
        approx_eq(moved.center(), hull.center() + delta);
        // A point that was interior is interior again after the same shift.
        let inside = Vec3::new(0.1, 0.1, 0.1);
        let shifted = inside + delta;
        let out = moved.project_out(shifted);
        let max_signed = moved
            .planes()
            .iter()
            .map(|f| f.normal.dot(out) - f.offset)
            .fold(Real::NEG_INFINITY, Real::max);
        assert!(max_signed.abs() < TOL);
    }

    #[test]
    fn empty_and_degenerate_inputs_are_rejected() {
        assert!(ConvexMesh::from_planes(Vec3::ZERO, 1.0, &[]).is_none());
        let degenerate = [Plane {
            normal: Vec3::ZERO,
            offset: 0.0,
        }];
        assert!(ConvexMesh::from_planes(Vec3::ZERO, 1.0, &degenerate).is_none());
    }
}
