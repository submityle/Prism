//! A bounded, `Copy` convex-polytope body proxy for soft↔rigid coupling.
//!
//! The analytic [`super::BodyCollider`] arms (sphere, capsule, half-space,
//! oriented box) each push an *interior* particle out to the primitive's
//! surface. A box prop is the richest shape that fits, yet production cloth
//! (UE5 Chaos Cloth, Houdini Vellum) also collides garments against arbitrary
//! *convex* props: bevelled crates, wedges, low-poly rocks. This module adds
//! that tier as a [`ConvexProxy`]: a convex solid expressed as the intersection
//! of a small set of outward-facing half-spaces.
//!
//! A point `x` is **inside** the solid when it lies behind every face plane,
//! `normal_i · x <= offset_i` for all faces (the planes carry *outward* unit
//! normals, so "behind" is the solid side). An interior point is pushed out
//! along the face of *least penetration* — the exact closest boundary point for
//! an interior point of a convex polytope — which is the direct generalization
//! of the oriented-box arm ([`super::project_out_of_obb`]): a box *is* the
//! intersection of its six face half-spaces, so a [`ConvexProxy::from_box`]
//! reproduces that arm's projection (exactly for an axis-aligned box; to float
//! tolerance for a rotated box, where the two arms reach the same surface via
//! different arithmetic).
//!
//! # Why bounded and inline
//!
//! [`super::BodyCollider`] is `#[derive(Copy)]` and is consumed *by value* by
//! the shared two-way-coupling kernel (`couple_particle_against_body`) and its
//! GPU twins, whose signatures must not change (a `Vec`/`Box`/`&[_]` payload
//! would break `Copy` or add a lifetime that ripples into those frozen
//! signatures). So the face set is a fixed-length, stack-resident
//! `[Plane; MAX_CONVEX_PLANES]` plus a length, keeping the proxy `Copy` and
//! compact. The cap is [`MAX_CONVEX_PLANES`]; see its docs for the honest size
//! trade-off against `clippy::large_enum_variant`.
//!
//! Determinism and robustness match the sibling arms: faces are visited in a
//! fixed order, only [`f32::sqrt`] is used (via `normalize_or_zero`), a
//! degenerate (zero-normal) face is ignored, and an empty proxy is inert, so no
//! path can produce a [`f32::NAN`].
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The
//! half-space-intersection convex solid, the least-penetration push-out, and
//! the slab-clip segment/convex time-of-impact are textbook computational
//! geometry (Ericson, *Real-Time Collision Detection*, §5.3 / §5.1.5).

use glam::{Quat, Vec3};

use crate::math::scalar::Real;

use super::EPS_LEN_SQ;

/// Maximum number of face planes a [`ConvexProxy`] can carry.
///
/// The proxy stores its faces inline as `[Plane; MAX_CONVEX_PLANES]` so it stays
/// `Copy` (see the module docs). Each [`Plane`] is 16 bytes, so the proxy is
/// `16 * N + 20` bytes (faces + center + radius + length). The enclosing
/// [`super::BodyCollider`] enum must stay under clippy's 200-byte
/// `large_enum_variant` threshold: `N = 10` lands the variant at 180 bytes,
/// whereas `N = 12` would reach 212 bytes and trip the lint. Ten faces comfortably
/// covers a box (6), a wedge/prism, or a bevelled crate; richer hulls are the
/// job of the (unbounded, arena-indexed) mesh proxy tracked separately. Ten is
/// therefore the honest cap for an inline `Copy` proxy.
pub const MAX_CONVEX_PLANES: usize = 10;

/// A single oriented face plane of a [`ConvexProxy`].
///
/// `normal` points *out* of the solid and is expected to be unit length (the
/// builders normalize). The plane is the locus `normal · x == offset`; the
/// solid lies on the `normal · x <= offset` side. A (near) zero `normal` is a
/// degenerate face that the proxy ignores.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plane {
    /// Outward unit face normal.
    pub normal: Vec3,
    /// Signed plane offset: a point `x` is on the plane when `normal·x == offset`.
    pub offset: Real,
}

impl Plane {
    /// The inert all-zero face used to pad the unused tail of the inline array.
    const ZERO: Self = Self {
        normal: Vec3::ZERO,
        offset: 0.0,
    };

    /// Builds a face from a point on the plane and an outward normal, returning
    /// [`None`] when the normal is (near) degenerate.
    #[must_use]
    fn from_point_normal(point: Vec3, normal: Vec3) -> Option<Self> {
        if normal.length_squared() <= EPS_LEN_SQ {
            return None;
        }
        let n = normal.normalize_or_zero();
        if n.length_squared() <= EPS_LEN_SQ {
            return None;
        }
        Some(Self {
            normal: n,
            offset: n.dot(point),
        })
    }
}

/// A convex solid expressed as the intersection of up to [`MAX_CONVEX_PLANES`]
/// outward-facing half-spaces, with a cached center and bounding radius.
///
/// Interior points are projected to the nearest face; points on or outside any
/// face are left untouched. The proxy is `Copy` so it slots into
/// [`super::BodyCollider`] without changing the shared coupling kernel's
/// by-value signatures.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConvexProxy {
    /// Inline face planes; only the first `count` entries are live, the rest
    /// are [`Plane::ZERO`] padding.
    planes: [Plane; MAX_CONVEX_PLANES],
    /// Number of live faces in `planes`.
    count: u8,
    /// Representative center (the coupling anchor / body reference point).
    center: Vec3,
    /// Conservative bounding-sphere radius about `center`, used only for the
    /// broad-phase AABB cull (never for the projection itself).
    radius: Real,
}

impl ConvexProxy {
    /// An empty, inert proxy with no faces.
    pub const EMPTY: Self = Self {
        planes: [Plane::ZERO; MAX_CONVEX_PLANES],
        count: 0,
        center: Vec3::ZERO,
        radius: 0.0,
    };

    /// Builds the convex proxy for an oriented box `(center, orientation,
    /// half_extents)`, i.e. the intersection of its six face half-spaces.
    ///
    /// Axes with a non-positive half-extent are collapsed (their two faces are
    /// dropped), matching the oriented-box arm's "inert along that axis"
    /// behavior. The result projects interior points identically to
    /// [`super::project_out_of_obb`]. The bounding radius is the box's corner
    /// distance so the broad-phase AABB fully contains it.
    #[must_use]
    pub fn from_box(center: Vec3, orientation: Quat, half_extents: Vec3) -> Self {
        let mut proxy = Self::EMPTY;
        proxy.center = center;
        let he = half_extents.max(Vec3::ZERO);
        proxy.radius = he.length();
        // Build faces in a fixed +X,-X,+Y,-Y,+Z,-Z order so the least-penetration
        // tie-break matches the oriented-box arm (which prefers X over Y over Z,
        // and the `+` face when a coordinate is exactly centered).
        let axes = [
            (orientation * Vec3::X, half_extents.x),
            (orientation * Vec3::Y, half_extents.y),
            (orientation * Vec3::Z, half_extents.z),
        ];
        for (axis, extent) in axes {
            if extent <= 0.0 {
                continue;
            }
            if let Some(face) = Plane::from_point_normal(center + axis * extent, axis) {
                proxy.push(face);
            }
            if let Some(face) = Plane::from_point_normal(center - axis * extent, -axis) {
                proxy.push(face);
            }
        }
        proxy
    }

    /// Builds a convex proxy from an explicit set of outward-facing planes with
    /// a caller-supplied `center` and bounding `radius`.
    ///
    /// Degenerate (zero-normal) planes are dropped and the normals are
    /// normalized. Returns [`None`] when the live-plane count is zero or exceeds
    /// [`MAX_CONVEX_PLANES`], so an over-budget hull is rejected rather than
    /// silently truncated. `radius` is clamped non-negative; it is used only for
    /// the broad-phase AABB and should enclose the solid about `center`.
    #[must_use]
    pub fn from_planes(center: Vec3, radius: Real, planes: &[Plane]) -> Option<Self> {
        let mut proxy = Self::EMPTY;
        proxy.center = center;
        proxy.radius = radius.max(0.0);
        let mut live = 0usize;
        for p in planes {
            if p.normal.length_squared() <= EPS_LEN_SQ {
                continue;
            }
            let n = p.normal.normalize_or_zero();
            if n.length_squared() <= EPS_LEN_SQ {
                continue;
            }
            live += 1;
            if live > MAX_CONVEX_PLANES {
                return None;
            }
            proxy.push(Plane {
                normal: n,
                offset: p.offset / p.normal.length(),
            });
        }
        if proxy.count == 0 {
            return None;
        }
        Some(proxy)
    }

    /// Appends a face, saturating silently at [`MAX_CONVEX_PLANES`] (callers
    /// that must not overflow use [`from_planes`](Self::from_planes), which
    /// rejects an over-budget set).
    fn push(&mut self, face: Plane) {
        let i = self.count as usize;
        if i < MAX_CONVEX_PLANES {
            self.planes[i] = face;
            self.count += 1;
        }
    }

    /// The live face planes.
    #[must_use]
    pub fn planes(&self) -> &[Plane] {
        &self.planes[..self.count as usize]
    }

    /// The proxy's representative center (coupling anchor / body reference).
    #[must_use]
    pub fn center(self) -> Vec3 {
        self.center
    }

    /// The conservative bounding-sphere radius about [`center`](Self::center).
    #[must_use]
    pub fn bounding_radius(self) -> Real {
        self.radius
    }

    /// Returns `pos` projected out to the nearest face when it lies strictly
    /// inside every face plane, otherwise returns `pos` unchanged.
    ///
    /// The least-penetrating face (smallest `offset - normal·pos`) is the
    /// closest boundary point for an interior point of a convex polytope, so the
    /// point lands exactly on the surface — the same "push to the nearest face"
    /// semantics as the sphere/box arms. A point on or outside any face is
    /// outside the solid and is returned untouched (the `>= 0` test mirrors the
    /// box arm's `|local| >= half_extent` early-out). An empty proxy, or one
    /// whose faces are all degenerate, is inert.
    #[must_use]
    pub fn project_out(self, pos: Vec3) -> Vec3 {
        let mut best_pen = Real::INFINITY;
        let mut best_normal = Vec3::ZERO;
        let mut found = false;
        for face in self.planes() {
            // A padded/degenerate face never constrains the solid.
            if face.normal.length_squared() <= EPS_LEN_SQ {
                continue;
            }
            let signed = face.normal.dot(pos) - face.offset;
            if signed >= 0.0 {
                // Outside (or exactly on) this face => outside the convex solid.
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
        // Faces carry unit normals, so stepping `best_pen` along the normal lands
        // exactly on the least-penetrating face plane.
        pos + best_normal * best_pen
    }

    /// Returns `self` rigidly translated by `delta`.
    ///
    /// The center shifts, each face offset moves by `normal·delta` (a rigid
    /// translation of the plane), and the bounding radius is unchanged.
    #[must_use]
    pub fn translated(self, delta: Vec3) -> Self {
        let mut out = self;
        out.center += delta;
        for i in 0..out.count as usize {
            out.planes[i].offset += out.planes[i].normal.dot(delta);
        }
        out
    }

    /// Returns the outward unit normal of the face the surface point `surf` lies
    /// on (the face with the largest signed distance, i.e. the active contact
    /// face), or the zero vector for an empty/degenerate proxy.
    #[must_use]
    pub fn face_normal(self, surf: Vec3) -> Vec3 {
        let mut best_signed = Real::NEG_INFINITY;
        let mut best_normal = Vec3::ZERO;
        for face in self.planes() {
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
    /// intersection of half-spaces: `t_enter` is the latest entry across the
    /// faces the segment approaches, `t_exit` the earliest exit across the faces
    /// it recedes from. Starting unbounded keeps the true entry time, so a
    /// segment that *starts inside* yields a negative entry and is rejected
    /// (left to the discrete projection), matching the oriented-box solver. Only
    /// multiplies and comparisons are used, so no path yields a [`f32::NAN`].
    #[must_use]
    pub fn segment_toi(self, prev: Vec3, curr: Vec3) -> Option<Real> {
        if self.count == 0 {
            return None;
        }
        let dir = curr - prev;
        let mut t_enter = Real::NEG_INFINITY;
        let mut t_exit = Real::INFINITY;
        for face in self.planes() {
            if face.normal.length_squared() <= EPS_LEN_SQ {
                continue;
            }
            // f(t) = normal·(prev + t*dir) - offset; inside is f <= 0.
            let num = face.normal.dot(prev) - face.offset;
            let rate = face.normal.dot(dir);
            if rate.abs() <= EPS_LEN_SQ {
                // Parallel to this face: a start outside it can never enter.
                if num > 0.0 {
                    return None;
                }
                continue;
            }
            let t = -num / rate;
            if rate > 0.0 {
                // Moving from inside to outside across this face => an exit bound.
                t_exit = t_exit.min(t);
            } else {
                // Moving from outside to inside => an entry bound.
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

#[cfg(test)]
mod tests {
    use super::super::{project_out_of_obb, BodyCollider};
    use super::*;

    const TOL: Real = 1.0e-6;

    fn approx_eq(a: Vec3, b: Vec3) {
        assert!((a - b).length() < TOL, "{a:?} != {b:?}");
    }

    fn unit_box() -> ConvexProxy {
        ConvexProxy::from_box(Vec3::ZERO, Quat::IDENTITY, Vec3::new(1.0, 0.5, 2.0))
    }

    #[test]
    fn body_collider_stays_under_clippy_large_variant_threshold() {
        // The whole point of the inline, bounded proxy: the enum must stay below
        // clippy's 200-byte `large_enum_variant` threshold so no `#[allow]` is
        // needed. See `MAX_CONVEX_PLANES` for the N=10 vs N=12 trade-off.
        assert!(
            size_of::<BodyCollider>() < 200,
            "BodyCollider is {} bytes",
            size_of::<BodyCollider>()
        );
    }

    #[test]
    fn from_box_has_six_faces() {
        assert_eq!(unit_box().planes().len(), 6);
    }

    #[test]
    fn from_box_collapses_zero_extent_axis() {
        // A zero Y half-extent drops the two Y faces, leaving four.
        let proxy = ConvexProxy::from_box(Vec3::ZERO, Quat::IDENTITY, Vec3::new(1.0, 0.0, 2.0));
        assert_eq!(proxy.planes().len(), 4);
    }

    #[test]
    fn convex_box_matches_obb_projection_interior() {
        // The convex-from-box projection must reproduce the oriented-box arm
        // (to float tolerance) on interior points with a unique
        // least-penetration face.
        let center = Vec3::new(0.3, -1.0, 2.0);
        let orientation = Quat::from_rotation_y(0.6) * Quat::from_rotation_x(0.2);
        let he = Vec3::new(1.0, 0.5, 2.0);
        let proxy = ConvexProxy::from_box(center, orientation, he);
        let samples = [
            Vec3::new(0.3, -0.8, 2.0),
            Vec3::new(0.1, -1.0, 1.3),
            Vec3::new(0.9, -1.1, 2.4),
            center + orientation * Vec3::new(0.2, 0.1, -1.5),
        ];
        for p in samples {
            approx_eq(
                proxy.project_out(p),
                project_out_of_obb(p, center, orientation, he),
            );
        }
    }

    #[test]
    fn interior_point_pushed_to_nearest_face() {
        // Point just below the +Y face (thinnest half-extent 0.5): lands on
        // y = 0.5, keeping x and z.
        let out = unit_box().project_out(Vec3::new(0.2, 0.4, -0.3));
        approx_eq(out, Vec3::new(0.2, 0.5, -0.3));
    }

    #[test]
    fn slope_face_pushes_along_tilted_normal() {
        // A box yawed+pitched: an interior point near one face is pushed out
        // along that face's (tilted) world normal and the correction is parallel
        // to the normal (so the particle then slides along the face).
        let orientation = Quat::from_rotation_z(std::f32::consts::FRAC_PI_6);
        let proxy = ConvexProxy::from_box(Vec3::ZERO, orientation, Vec3::new(1.0, 0.4, 1.0));
        // A point just inside the +Y face in local space.
        let inside_local = Vec3::new(0.1, 0.35, -0.2);
        let p = orientation * inside_local;
        let out = proxy.project_out(p);
        let correction = out - p;
        assert!(correction.length() > 1e-4, "expected a push");
        let expected_normal = orientation * Vec3::Y;
        // Correction is parallel to the tilted face normal.
        let along = correction.normalize_or_zero().dot(expected_normal);
        assert!((along - 1.0).abs() < 1e-5, "dot {along}");
    }

    #[test]
    fn point_outside_a_face_is_untouched() {
        let proxy = unit_box();
        let p = Vec3::new(2.0, 0.0, 0.0); // outside the +X face
        approx_eq(proxy.project_out(p), p);
    }

    #[test]
    fn backface_point_behind_box_is_outside() {
        // Far behind the -Z face: outside the solid, untouched.
        let proxy = unit_box();
        let p = Vec3::new(0.0, 0.0, -5.0);
        approx_eq(proxy.project_out(p), p);
    }

    #[test]
    fn empty_proxy_is_inert() {
        let p = Vec3::new(0.1, 0.2, 0.3);
        approx_eq(ConvexProxy::EMPTY.project_out(p), p);
    }

    #[test]
    fn from_planes_rejects_over_budget_and_empty() {
        // Eleven live planes exceed the ten-face cap. The normals only need to
        // be distinct and non-degenerate; they are built with plain arithmetic
        // (no trig) to respect the repo's determinism lint.
        let mut planes = Vec::new();
        for i in 0..(MAX_CONVEX_PLANES + 1) {
            let a = i as Real;
            planes.push(Plane {
                normal: Vec3::new(1.0 + a, 2.0 - a, 0.5 + a * 0.25),
                offset: 1.0,
            });
        }
        assert!(ConvexProxy::from_planes(Vec3::ZERO, 2.0, &planes).is_none());
        // All-degenerate => no live faces => None.
        let degenerate = [Plane {
            normal: Vec3::ZERO,
            offset: 1.0,
        }];
        assert!(ConvexProxy::from_planes(Vec3::ZERO, 1.0, &degenerate).is_none());
    }

    #[test]
    fn translated_moves_center_and_faces_rigidly() {
        let proxy = unit_box();
        let delta = Vec3::new(1.0, -2.0, 0.5);
        let moved = proxy.translated(delta);
        approx_eq(moved.center(), delta);
        // A point that was interior at the origin is interior after both the
        // point and the box move by the same delta, projecting to the shifted
        // face.
        let p = Vec3::new(0.2, 0.4, -0.3);
        let before = proxy.project_out(p);
        let after = moved.project_out(p + delta);
        approx_eq(after, before + delta);
    }

    #[test]
    fn segment_toi_enters_front_face() {
        // A segment coming straight down into the top (+Y) face enters at the
        // half-extent plane y = 0.5.
        let proxy = unit_box();
        let toi = proxy
            .segment_toi(Vec3::new(0.0, 1.5, 0.0), Vec3::new(0.0, -0.5, 0.0))
            .expect("hit");
        // y(t) = 1.5 - 2t; y = 0.5 at t = 0.5.
        assert!((toi - 0.5).abs() < 1e-5, "toi {toi}");
    }

    #[test]
    fn segment_toi_starting_inside_is_rejected() {
        let proxy = unit_box();
        assert!(proxy
            .segment_toi(Vec3::ZERO, Vec3::new(0.0, 0.2, 0.0))
            .is_none());
    }

    #[test]
    fn segment_toi_miss_returns_none() {
        let proxy = unit_box();
        // Passes well above the box.
        assert!(proxy
            .segment_toi(Vec3::new(-5.0, 3.0, 0.0), Vec3::new(5.0, 3.0, 0.0))
            .is_none());
    }

    #[test]
    fn face_normal_on_top_face_points_up() {
        let proxy = unit_box();
        let n = proxy.face_normal(Vec3::new(0.1, 0.5, -0.2));
        approx_eq(n, Vec3::Y);
    }
}
