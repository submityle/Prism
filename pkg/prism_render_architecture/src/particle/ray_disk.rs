//! Analytic `ray`-disk and `ray`-annulus intersection for the particle
//! subsystem's picking, decal-projection, and analytic-primitive raytrace
//! contracts (design §10, §14).
//!
//! This module owns the *closed-form* solution of the flat-circle problem: given
//! a [`Ray`] and either a [`Disk`] (center + normal + radius) or an [`Annulus`]
//! (center + normal + inner/outer radii), it first solves the one-division
//! `ray`-plane crossing `t = (center - origin)·n / (dir·n)`, then classifies the
//! crossing by the *squared* in-plane radial distance of the hit point from the
//! center. A boolean predicate ([`Disk::intersects`] / [`Annulus::intersects`])
//! never leaves squared space, so it needs no `sqrt`; the full hit query
//! ([`Disk::intersect`] / [`Annulus::intersect`]) additionally reports the ray
//! parameter, the world-space point, the hit face (front or back, read straight
//! off the sign of `dir·n`), and — only here, and only via a single `sqrt` — the
//! outward unit surface normal.
//!
//! # Strict scope
//! This is the *flat bounded circle in a plane* kernel and nothing else. It is
//! the analytic sibling of the other single-`ray`-single-primitive contracts and
//! is deliberately distinct from each of them:
//! * [`ray_triangle`](crate::particle::ray_triangle) tests a three-vertex
//!   triangular face with `barycentric` coordinates, not a circular region.
//! * [`ray_sphere`](crate::particle::ray_sphere) solves a quadratic against a
//!   full spherical *surface*, not a flat disk.
//! * [`ray_cylinder`](crate::particle::ray_cylinder) tests an infinite or capped
//!   tube; a disk is only its flat end-cap, decoupled from the side wall.
//! * [`ray_capsule`](crate::particle::ray_capsule) tests a swept-sphere volume.
//! * [`ray_obb`](crate::particle::ray_obb) tests an oriented box `slab`.
//!
//! # No transcendental math
//! Every routine is polynomial plus at most one guarded division (the plane
//! parameter) and, only when a unit normal is requested, one `sqrt`. There is no
//! `sin`/`cos`/`pow`/`atan` anywhere, and no exact `==` / `!=` is ever written on
//! a production `f32`: a direction that is parallel to the plane (its `dir·n`
//! magnitude falls below an explicit epsilon) is rejected as a non-intersection
//! rather than dividing by a vanishing denominator, and a degenerate radius or a
//! degenerate normal likewise reports a miss instead of producing a `NaN`. The
//! disk normal may be supplied *un-normalized*: the plane parameter and the
//! radial test are both invariant to its magnitude, so the `sqrt` is spent only
//! to emit the final unit normal. This keeps the `CPU` reference here in bit-for-
//! bit agreement with a future `GPU` (`WESL`) kernel that packs the same disk
//! through the `std430` helpers in [`crate::particle::gpu_layout`].

use crate::particle::gpu_layout::{storage_bytes, U32_STRIDE, VEC4_STRIDE};

/// Epsilon used to guard the plane-parameter division, to accept a hit that
/// lands exactly on a radius, and to compare parameters against zero without
/// ever writing an exact `==` / `!=` on a production `f32`.
const CMP_EPS: f32 = 1.0e-6;

/// A hand-rolled three-component vector, kept local so the module stays a
/// zero-dependency contract and its vector math is auditable in one place.
///
/// Arithmetic methods are named `plus` / `minus` / `scale` (not the operator
/// names) to keep the closed-form algebra explicit and to match the sibling
/// `ray_*` contracts; operator traits are intentionally not implemented.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec3 {
    /// The x component.
    pub x: f32,
    /// The y component.
    pub y: f32,
    /// The z component.
    pub z: f32,
}

impl Vec3 {
    /// The zero vector `(0, 0, 0)`, used as the degenerate normalization
    /// fallback.
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    /// Builds a vector from its three components.
    #[must_use]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// A vector with all three lanes set to `s`.
    #[must_use]
    pub const fn splat(s: f32) -> Self {
        Self { x: s, y: s, z: s }
    }

    /// Component-wise sum `self + rhs`.
    #[must_use]
    pub fn plus(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }

    /// Component-wise difference `self - rhs`.
    #[must_use]
    pub fn minus(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }

    /// Uniform scale by a scalar.
    #[must_use]
    pub fn scale(self, s: f32) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }

    /// Dot (inner) product `self · rhs`.
    #[must_use]
    pub fn dot(self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    /// Right-handed cross product `self × rhs`, provided for callers that build
    /// an orthonormal frame around a hit normal.
    #[must_use]
    pub fn cross(self, rhs: Self) -> Self {
        Self::new(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
    }

    /// Squared Euclidean length; cheaper than [`Vec3::length`] when only a
    /// comparison is needed, and the sole quantity the radial classification
    /// uses so the boolean tests stay `sqrt`-free.
    #[must_use]
    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }

    /// Euclidean length (one `sqrt`).
    #[must_use]
    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    /// Unit vector in the same direction, or the zero vector when the input is
    /// shorter than [`CMP_EPS`] (so a degenerate input can never yield a
    /// `NaN`).
    #[must_use]
    pub fn normalize_or_zero(self) -> Self {
        let len = self.length();
        if len < CMP_EPS {
            Self::ZERO
        } else {
            self.scale(1.0 / len)
        }
    }
}

/// A parametric ray `origin + t * dir` with `t >= 0` denoting the forward
/// half-line.
///
/// The plane solver works for any non-degenerate `dir`; the ray parameter `t`
/// equals the Euclidean hit distance only when `dir` is unit length.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ray {
    /// The ray origin.
    pub origin: Vec3,
    /// The ray direction (not required to be unit length).
    pub dir: Vec3,
}

impl Ray {
    /// Builds a ray from an origin and a (possibly non-unit) direction.
    #[must_use]
    pub const fn new(origin: Vec3, dir: Vec3) -> Self {
        Self { origin, dir }
    }

    /// Builds a ray whose direction is normalized to unit length, so the ray
    /// parameter `t` equals the Euclidean distance from the origin. A
    /// degenerate direction collapses to the zero vector, which the solver
    /// later rejects as non-intersecting.
    #[must_use]
    pub fn new_normalized(origin: Vec3, dir: Vec3) -> Self {
        Self {
            origin,
            dir: dir.normalize_or_zero(),
        }
    }

    /// Returns a copy of this ray with its direction normalized to unit length.
    #[must_use]
    pub fn normalized(self) -> Self {
        Self::new_normalized(self.origin, self.dir)
    }

    /// Evaluates the ray position at parameter `t`.
    #[must_use]
    pub fn at(self, t: f32) -> Vec3 {
        self.origin.plus(self.dir.scale(t))
    }
}

/// Which side of the plane the ray struck, read straight off the sign of the
/// plane denominator `dir·n`.
///
/// A ray travelling *against* the stored normal (`dir·n < 0`) strikes the
/// [`HitFace::Front`]; one travelling *with* it (`dir·n > 0`) strikes the
/// [`HitFace::Back`]. The exactly-parallel case (`dir·n ≈ 0`) never reaches a
/// face classification because it is rejected as a non-intersection first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HitFace {
    /// The side the stored normal points toward.
    Front,
    /// The side opposite the stored normal.
    Back,
}

impl HitFace {
    /// Whether this is the front face.
    #[must_use]
    pub fn is_front(self) -> bool {
        matches!(self, Self::Front)
    }
}

/// A resolved forward intersection with a flat circular primitive: the ray
/// parameter, the world-space point, the outward unit normal, the hit face, and
/// the squared in-plane radial distance of the hit from the center.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayDiskHit {
    /// The ray parameter at the hit (`>= 0`). Equals the hit distance when the
    /// ray direction is unit length.
    pub t: f32,
    /// The world-space intersection point `ray.at(t)`.
    pub point: Vec3,
    /// The unit surface normal, oriented to point *against* the incoming ray
    /// (i.e. flipped to the [`HitFace`] the ray actually struck). Zero only for
    /// a degenerate normal, which is otherwise rejected before a hit is
    /// reported.
    pub normal: Vec3,
    /// Which face the ray struck.
    pub face: HitFace,
    /// Squared distance from the hit point to the center, measured in the
    /// plane. Compared against the squared radii, this is the quantity that
    /// classifies a disk / annulus hit without a `sqrt`.
    pub radial_dist_sq: f32,
}

/// The plane parameter of a `ray`-plane crossing plus the raw denominator
/// `dir·n`, or `None` when the ray is parallel to the plane.
struct PlaneHit {
    /// The ray parameter `t` at the crossing.
    t: f32,
    /// The raw denominator `dir·n`, whose sign selects the hit face.
    denom: f32,
}

/// Solves the one-division `ray`-plane crossing `t = (center - origin)·n / (dir·n)`.
///
/// Returns `None` when `|dir·n|` falls below [`CMP_EPS`] (the ray is parallel to
/// the plane, so it either never meets it or lies within it — either way there
/// is no isolated crossing to report). The result is invariant to the magnitude
/// of `normal` because both the numerator and the denominator scale with it.
fn solve_plane(center: Vec3, normal: Vec3, ray: Ray) -> Option<PlaneHit> {
    let denom = ray.dir.dot(normal);
    if denom.abs() < CMP_EPS {
        return None;
    }
    let t = center.minus(ray.origin).dot(normal) / denom;
    Some(PlaneHit { t, denom })
}

/// `std430` byte size of one packed [`Disk`]: two `vec4` slots holding
/// `[center.xyz, radius]` and `[normal.xyz, pad]`.
pub const RAY_DISK_STD430_SIZE: usize = 2 * VEC4_STRIDE;

/// `std430` byte size of one packed [`Annulus`]: two `vec4` slots holding
/// `[center.xyz, inner]` and `[normal.xyz, outer]`.
pub const RAY_ANNULUS_STD430_SIZE: usize = 2 * VEC4_STRIDE;

/// An analytic flat disk: a center, a face normal (not required to be unit
/// length), and a radius.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Disk {
    /// The disk center, lying in its plane.
    pub center: Vec3,
    /// The face normal; may be supplied un-normalized.
    pub normal: Vec3,
    /// The disk radius. Values at or below [`CMP_EPS`] are treated as a
    /// degenerate point and never report a hit.
    pub radius: f32,
}

impl Disk {
    /// Builds a disk from a center, normal, and radius.
    #[must_use]
    pub const fn new(center: Vec3, normal: Vec3, radius: f32) -> Self {
        Self {
            center,
            normal,
            radius,
        }
    }

    /// Whether the disk is degenerate: its radius is at or below [`CMP_EPS`], or
    /// its normal is shorter than [`CMP_EPS`] (so the plane is undefined). A
    /// degenerate disk encloses no area and never intersects.
    #[must_use]
    pub fn is_degenerate(&self) -> bool {
        self.radius <= CMP_EPS || self.normal.length_squared() <= CMP_EPS * CMP_EPS
    }

    /// Whether the ray's forward half-line crosses the disk. This is the
    /// `sqrt`-free boolean predicate: it solves the plane crossing and compares
    /// the squared radial distance against `radius²`, accepting a hit that lands
    /// exactly on the rim (within [`CMP_EPS`]).
    #[must_use]
    pub fn intersects(&self, ray: Ray) -> bool {
        if self.is_degenerate() {
            return false;
        }
        let Some(plane) = solve_plane(self.center, self.normal, ray) else {
            return false;
        };
        if plane.t < -CMP_EPS {
            return false;
        }
        let point = ray.at(plane.t.max(0.0));
        let d2 = point.minus(self.center).length_squared();
        d2 <= self.radius * self.radius + CMP_EPS
    }

    /// Returns the nearest forward hit (`t >= 0`), or `None` when the ray is
    /// parallel, points away from the plane, the crossing lands outside the
    /// radius, or the disk is degenerate.
    ///
    /// The returned normal is the unit face normal flipped to point *against*
    /// the incoming ray (the single `sqrt` in this module is spent here). The
    /// [`HitFace`] records which physical side was struck.
    #[must_use]
    pub fn intersect(&self, ray: Ray) -> Option<RayDiskHit> {
        if self.is_degenerate() {
            return None;
        }
        let plane = solve_plane(self.center, self.normal, ray)?;
        if plane.t < -CMP_EPS {
            return None;
        }
        let t = plane.t.max(0.0);
        let point = ray.at(t);
        let radial_dist_sq = point.minus(self.center).length_squared();
        if radial_dist_sq > self.radius * self.radius + CMP_EPS {
            return None;
        }
        Some(build_hit(
            self.normal,
            plane.denom,
            t,
            point,
            radial_dist_sq,
        ))
    }

    /// Serializes the disk to its little-endian `std430` byte image: two
    /// `vec4<f32>` slots, `[center.x, center.y, center.z, radius]` then
    /// `[normal.x, normal.y, normal.z, 0]`.
    #[must_use]
    pub fn to_std430(&self) -> [u8; RAY_DISK_STD430_SIZE] {
        pack_two_vec4(
            [self.center.x, self.center.y, self.center.z, self.radius],
            [self.normal.x, self.normal.y, self.normal.z, 0.0],
        )
    }
}

/// An analytic flat annulus (a disk with a circular hole): a center, a face
/// normal (not required to be unit length), and inner/outer radii.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Annulus {
    /// The annulus center, lying in its plane.
    pub center: Vec3,
    /// The face normal; may be supplied un-normalized.
    pub normal: Vec3,
    /// The inner radius (the hole). Values at or below [`CMP_EPS`] make the
    /// annulus a full disk.
    pub inner: f32,
    /// The outer radius. Values at or below both [`CMP_EPS`] and `inner` make
    /// the annulus degenerate.
    pub outer: f32,
}

impl Annulus {
    /// Builds an annulus from a center, normal, and inner/outer radii.
    #[must_use]
    pub const fn new(center: Vec3, normal: Vec3, inner: f32, outer: f32) -> Self {
        Self {
            center,
            normal,
            inner,
            outer,
        }
    }

    /// Whether the annulus is degenerate: its outer radius is at or below
    /// [`CMP_EPS`], its normal is shorter than [`CMP_EPS`], or the outer radius
    /// does not exceed the inner radius (leaving no band). A degenerate annulus
    /// encloses no area and never intersects.
    #[must_use]
    pub fn is_degenerate(&self) -> bool {
        self.outer <= CMP_EPS
            || self.normal.length_squared() <= CMP_EPS * CMP_EPS
            || self.outer <= self.inner + CMP_EPS
    }

    /// Whether the ray's forward half-line crosses the annular band. This is the
    /// `sqrt`-free boolean predicate: it solves the plane crossing and checks
    /// that the squared radial distance lies within `[inner², outer²]` (each
    /// edge accepted within [`CMP_EPS`]).
    #[must_use]
    pub fn intersects(&self, ray: Ray) -> bool {
        if self.is_degenerate() {
            return false;
        }
        let Some(plane) = solve_plane(self.center, self.normal, ray) else {
            return false;
        };
        if plane.t < -CMP_EPS {
            return false;
        }
        let point = ray.at(plane.t.max(0.0));
        let d2 = point.minus(self.center).length_squared();
        d2 >= self.inner * self.inner - CMP_EPS && d2 <= self.outer * self.outer + CMP_EPS
    }

    /// Returns the nearest forward hit (`t >= 0`) within the band, or `None`
    /// when the ray is parallel, points away, the crossing lands in the hole or
    /// outside the outer radius, or the annulus is degenerate.
    ///
    /// The returned normal is the unit face normal flipped to point *against*
    /// the incoming ray (the single `sqrt` is spent here). The [`HitFace`]
    /// records which physical side was struck.
    #[must_use]
    pub fn intersect(&self, ray: Ray) -> Option<RayDiskHit> {
        if self.is_degenerate() {
            return None;
        }
        let plane = solve_plane(self.center, self.normal, ray)?;
        if plane.t < -CMP_EPS {
            return None;
        }
        let t = plane.t.max(0.0);
        let point = ray.at(t);
        let radial_dist_sq = point.minus(self.center).length_squared();
        let inside_hole = radial_dist_sq < self.inner * self.inner - CMP_EPS;
        let outside_rim = radial_dist_sq > self.outer * self.outer + CMP_EPS;
        if inside_hole || outside_rim {
            return None;
        }
        Some(build_hit(
            self.normal,
            plane.denom,
            t,
            point,
            radial_dist_sq,
        ))
    }

    /// Serializes the annulus to its little-endian `std430` byte image: two
    /// `vec4<f32>` slots, `[center.x, center.y, center.z, inner]` then
    /// `[normal.x, normal.y, normal.z, outer]`.
    #[must_use]
    pub fn to_std430(&self) -> [u8; RAY_ANNULUS_STD430_SIZE] {
        pack_two_vec4(
            [self.center.x, self.center.y, self.center.z, self.inner],
            [self.normal.x, self.normal.y, self.normal.z, self.outer],
        )
    }
}

/// Assembles a [`RayDiskHit`] from the solved plane crossing, orienting the unit
/// normal against the incoming ray and reading the face off `denom = dir·n`.
fn build_hit(normal: Vec3, denom: f32, t: f32, point: Vec3, radial_dist_sq: f32) -> RayDiskHit {
    let unit = normal.normalize_or_zero();
    let (face, oriented) = if denom < 0.0 {
        (HitFace::Front, unit)
    } else {
        (HitFace::Back, unit.scale(-1.0))
    };
    RayDiskHit {
        t,
        point,
        normal: oriented,
        face,
        radial_dist_sq,
    }
}

/// Packs two `vec4<f32>` word groups into a contiguous little-endian `std430`
/// byte image.
fn pack_two_vec4(a: [f32; 4], b: [f32; 4]) -> [u8; 2 * VEC4_STRIDE] {
    let mut bytes = [0u8; 2 * VEC4_STRIDE];
    for (i, word) in a.iter().chain(b.iter()).enumerate() {
        let start = i * U32_STRIDE;
        bytes[start..start + U32_STRIDE].copy_from_slice(&word.to_le_bytes());
    }
    bytes
}

/// Total `std430` byte size for a storage buffer of `count` disks, clamped up to
/// one element so a `WebGPU` binding is never zero-sized.
#[must_use]
pub fn disk_gpu_storage_bytes(count: usize) -> usize {
    storage_bytes(RAY_DISK_STD430_SIZE, count)
}

/// Total `std430` byte size for a storage buffer of `count` annuli, clamped up
/// to one element so a `WebGPU` binding is never zero-sized.
#[must_use]
pub fn annulus_gpu_storage_bytes(count: usize) -> usize {
    storage_bytes(RAY_ANNULUS_STD430_SIZE, count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    /// Absolute tolerance for parameter / position comparisons that may involve
    /// a `sqrt` (the unit normal).
    const EPS: f32 = 1.0e-4;

    fn approx(a: f32, b: f32) {
        assert!((a - b).abs() < EPS, "expected {b}, got {a}");
    }

    fn approx_vec(a: Vec3, b: Vec3) {
        approx(a.x, b.x);
        approx(a.y, b.y);
        approx(a.z, b.z);
    }

    fn unit_z() -> Vec3 {
        Vec3::new(0.0, 0.0, 1.0)
    }

    #[test]
    fn vec3_algebra_is_exact() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, -1.0, 0.5);
        assert_eq!(a.plus(b), Vec3::new(5.0, 1.0, 3.5));
        assert_eq!(a.minus(b), Vec3::new(-3.0, 3.0, 2.5));
        assert_eq!(a.scale(2.0), Vec3::new(2.0, 4.0, 6.0));
        assert_eq!(a.dot(Vec3::new(1.0, 1.0, 1.0)), 6.0);
        assert_eq!(Vec3::splat(3.0), Vec3::new(3.0, 3.0, 3.0));
    }

    #[test]
    fn vec3_cross_is_right_handed() {
        let x = Vec3::new(1.0, 0.0, 0.0);
        let y = Vec3::new(0.0, 1.0, 0.0);
        assert_eq!(x.cross(y), Vec3::new(0.0, 0.0, 1.0));
    }

    #[test]
    fn vec3_length_matches_pythagoras() {
        let v = Vec3::new(3.0, 4.0, 0.0);
        assert_eq!(v.length_squared(), 25.0);
        approx(v.length(), 5.0);
    }

    #[test]
    fn normalize_zero_is_zero_not_nan() {
        let n = Vec3::ZERO.normalize_or_zero();
        assert_eq!(n, Vec3::ZERO);
        assert!(!n.x.is_nan());
    }

    #[test]
    fn normalize_unit_length() {
        let n = Vec3::new(0.0, 0.0, 7.0).normalize_or_zero();
        approx_vec(n, unit_z());
        approx(n.length(), 1.0);
    }

    #[test]
    fn ray_at_evaluates_point() {
        let r = Ray::new(Vec3::new(1.0, 2.0, 3.0), Vec3::new(0.0, 0.0, 2.0));
        assert_eq!(r.at(2.0), Vec3::new(1.0, 2.0, 7.0));
    }

    #[test]
    fn new_normalized_has_unit_direction() {
        let r = Ray::new_normalized(Vec3::ZERO, Vec3::new(0.0, 0.0, 5.0));
        approx(r.dir.length(), 1.0);
        approx_vec(r.dir, unit_z());
    }

    #[test]
    fn front_face_hit_has_known_t_and_point() {
        // Normal points +z; ray travels -z, so it strikes the front face.
        let disk = Disk::new(Vec3::ZERO, unit_z(), 1.0);
        let r = Ray::new_normalized(Vec3::new(0.0, 0.0, 5.0), Vec3::new(0.0, 0.0, -1.0));
        let hit = disk.intersect(r).expect("front hit");
        approx(hit.t, 5.0);
        approx_vec(hit.point, Vec3::ZERO);
        assert_eq!(hit.face, HitFace::Front);
        // Oriented normal points back toward the origin (+z).
        approx_vec(hit.normal, unit_z());
        approx(hit.radial_dist_sq, 0.0);
    }

    #[test]
    fn back_face_hit_flips_normal() {
        // Normal points +z; ray travels +z, so it strikes the back face.
        let disk = Disk::new(Vec3::ZERO, unit_z(), 1.0);
        let r = Ray::new_normalized(Vec3::new(0.0, 0.0, -5.0), unit_z());
        let hit = disk.intersect(r).expect("back hit");
        approx(hit.t, 5.0);
        approx_vec(hit.point, Vec3::ZERO);
        assert_eq!(hit.face, HitFace::Back);
        // Oriented normal points back toward the origin (-z).
        approx_vec(hit.normal, Vec3::new(0.0, 0.0, -1.0));
    }

    #[test]
    fn offset_ray_misses_disk() {
        let disk = Disk::new(Vec3::ZERO, unit_z(), 1.0);
        let r = Ray::new_normalized(Vec3::new(5.0, 0.0, -5.0), unit_z());
        assert!(disk.intersect(r).is_none());
        assert!(!disk.intersects(r));
    }

    #[test]
    fn parallel_ray_does_not_intersect() {
        // Direction lies in the plane (perpendicular to the normal).
        let disk = Disk::new(Vec3::ZERO, unit_z(), 1.0);
        let r = Ray::new_normalized(Vec3::new(0.0, 0.0, -5.0), Vec3::new(1.0, 0.0, 0.0));
        assert!(disk.intersect(r).is_none());
        assert!(!disk.intersects(r));
        assert!(solve_plane(disk.center, disk.normal, r).is_none());
    }

    #[test]
    fn grazing_exactly_on_radius_hits() {
        // Hit point lands at exactly radius 1 on the rim; the epsilon-tolerant
        // rim test must accept it.
        let disk = Disk::new(Vec3::ZERO, unit_z(), 1.0);
        let r = Ray::new_normalized(Vec3::new(1.0, 0.0, -5.0), unit_z());
        let hit = disk.intersect(r).expect("rim hit");
        approx(hit.point.x, 1.0);
        approx(hit.radial_dist_sq, 1.0);
        assert!(disk.intersects(r));
    }

    #[test]
    fn just_outside_radius_misses() {
        let disk = Disk::new(Vec3::ZERO, unit_z(), 1.0);
        let r = Ray::new_normalized(Vec3::new(1.01, 0.0, -5.0), unit_z());
        assert!(disk.intersect(r).is_none());
        assert!(!disk.intersects(r));
    }

    #[test]
    fn negative_direction_ray_points_away_and_misses() {
        // Origin in front of the plane, direction pointing further away.
        let disk = Disk::new(Vec3::ZERO, unit_z(), 1.0);
        let r = Ray::new_normalized(Vec3::new(0.0, 0.0, 5.0), unit_z());
        assert!(disk.intersect(r).is_none());
        assert!(!disk.intersects(r));
    }

    #[test]
    fn behind_origin_crossing_is_rejected() {
        let disk = Disk::new(Vec3::ZERO, unit_z(), 1.0);
        // Plane is behind the origin along the travel direction.
        let r = Ray::new_normalized(Vec3::new(0.0, 0.0, -5.0), Vec3::new(0.0, 0.0, -1.0));
        assert!(disk.intersect(r).is_none());
    }

    #[test]
    fn non_unit_normal_yields_correct_t_and_unit_normal() {
        // Normal has length 2; the plane parameter must be invariant to it and
        // the reported normal must still be unit length.
        let disk = Disk::new(Vec3::ZERO, Vec3::new(0.0, 0.0, 2.0), 1.0);
        let r = Ray::new_normalized(Vec3::new(0.0, 0.0, -5.0), unit_z());
        let hit = disk.intersect(r).expect("hit");
        approx(hit.t, 5.0);
        approx(hit.normal.length(), 1.0);
        assert_eq!(hit.face, HitFace::Back);
        approx_vec(hit.normal, Vec3::new(0.0, 0.0, -1.0));
    }

    #[test]
    fn non_unit_direction_scales_the_parameter() {
        // Direction length 2, so the parameter is half the distance.
        let disk = Disk::new(Vec3::ZERO, unit_z(), 1.0);
        let r = Ray::new(Vec3::new(0.0, 0.0, -5.0), Vec3::new(0.0, 0.0, 2.0));
        let hit = disk.intersect(r).expect("hit");
        approx(hit.t, 2.5);
        approx_vec(hit.point, Vec3::ZERO);
    }

    #[test]
    fn degenerate_radius_never_hits() {
        let disk = Disk::new(Vec3::ZERO, unit_z(), 0.0);
        assert!(disk.is_degenerate());
        let r = Ray::new_normalized(Vec3::new(0.0, 0.0, -5.0), unit_z());
        assert!(disk.intersect(r).is_none());
        assert!(!disk.intersects(r));
    }

    #[test]
    fn degenerate_normal_never_hits() {
        let disk = Disk::new(Vec3::ZERO, Vec3::ZERO, 1.0);
        assert!(disk.is_degenerate());
        let r = Ray::new_normalized(Vec3::new(0.0, 0.0, -5.0), unit_z());
        assert!(disk.intersect(r).is_none());
    }

    #[test]
    fn intersects_bool_matches_full_query() {
        let disk = Disk::new(Vec3::ZERO, unit_z(), 1.0);
        let hitting = Ray::new_normalized(Vec3::new(0.0, 0.0, -5.0), unit_z());
        let missing = Ray::new_normalized(Vec3::new(9.0, 0.0, -5.0), unit_z());
        assert_eq!(disk.intersects(hitting), disk.intersect(hitting).is_some());
        assert_eq!(disk.intersects(missing), disk.intersect(missing).is_some());
    }

    #[test]
    fn hit_face_is_front_helper() {
        assert!(HitFace::Front.is_front());
        assert!(!HitFace::Back.is_front());
    }

    #[test]
    fn annulus_hits_the_band() {
        let ann = Annulus::new(Vec3::ZERO, unit_z(), 1.0, 2.0);
        // Radial distance 1.5 is inside [1, 2].
        let r = Ray::new_normalized(Vec3::new(1.5, 0.0, -5.0), unit_z());
        let hit = ann.intersect(r).expect("band hit");
        approx(hit.radial_dist_sq, 2.25);
        approx(hit.t, 5.0);
        assert_eq!(hit.face, HitFace::Back);
        assert!(ann.intersects(r));
    }

    #[test]
    fn annulus_inner_hole_misses() {
        let ann = Annulus::new(Vec3::ZERO, unit_z(), 1.0, 2.0);
        // Radial distance 0.5 is inside the hole.
        let r = Ray::new_normalized(Vec3::new(0.5, 0.0, -5.0), unit_z());
        assert!(ann.intersect(r).is_none());
        assert!(!ann.intersects(r));
    }

    #[test]
    fn annulus_outside_outer_misses() {
        let ann = Annulus::new(Vec3::ZERO, unit_z(), 1.0, 2.0);
        let r = Ray::new_normalized(Vec3::new(3.0, 0.0, -5.0), unit_z());
        assert!(ann.intersect(r).is_none());
        assert!(!ann.intersects(r));
    }

    #[test]
    fn annulus_grazes_inner_edge() {
        let ann = Annulus::new(Vec3::ZERO, unit_z(), 1.0, 2.0);
        // Exactly on the inner radius.
        let r = Ray::new_normalized(Vec3::new(1.0, 0.0, -5.0), unit_z());
        let hit = ann.intersect(r).expect("inner edge hit");
        approx(hit.radial_dist_sq, 1.0);
    }

    #[test]
    fn annulus_grazes_outer_edge() {
        let ann = Annulus::new(Vec3::ZERO, unit_z(), 1.0, 2.0);
        // Exactly on the outer radius.
        let r = Ray::new_normalized(Vec3::new(2.0, 0.0, -5.0), unit_z());
        let hit = ann.intersect(r).expect("outer edge hit");
        approx(hit.radial_dist_sq, 4.0);
    }

    #[test]
    fn annulus_front_face_hit() {
        let ann = Annulus::new(Vec3::ZERO, unit_z(), 1.0, 2.0);
        let r = Ray::new_normalized(Vec3::new(1.5, 0.0, 5.0), Vec3::new(0.0, 0.0, -1.0));
        let hit = ann.intersect(r).expect("front band hit");
        assert_eq!(hit.face, HitFace::Front);
        approx_vec(hit.normal, unit_z());
    }

    #[test]
    fn annulus_degenerate_band_never_hits() {
        // Outer does not exceed inner: no band.
        let ann = Annulus::new(Vec3::ZERO, unit_z(), 2.0, 2.0);
        assert!(ann.is_degenerate());
        let r = Ray::new_normalized(Vec3::new(0.0, 0.0, -5.0), unit_z());
        assert!(ann.intersect(r).is_none());
        assert!(!ann.intersects(r));
    }

    #[test]
    fn solver_is_deterministic() {
        let disk = Disk::new(Vec3::new(1.0, -2.0, 3.0), Vec3::new(0.3, 0.4, 1.0), 2.0);
        let r = Ray::new_normalized(Vec3::new(0.0, 0.0, -8.0), Vec3::new(0.1, 0.05, 1.0));
        assert_eq!(disk.intersect(r), disk.intersect(r));
        let ann = Annulus::new(Vec3::ZERO, unit_z(), 1.0, 3.0);
        assert_eq!(ann.intersect(r), ann.intersect(r));
    }

    #[test]
    fn oblique_ray_hits_known_point() {
        // Disk in the z=0 plane; ray from (2,0,4) toward (-1,0,-2) (unit-free).
        // Plane crossing at t where z = 4 + t*(-2) = 0 => t = 2, giving
        // x = 2 + 2*(-1) = 0, so the hit is the center.
        let disk = Disk::new(Vec3::ZERO, unit_z(), 1.0);
        let r = Ray::new(Vec3::new(2.0, 0.0, 4.0), Vec3::new(-1.0, 0.0, -2.0));
        let hit = disk.intersect(r).expect("oblique hit");
        approx(hit.t, 2.0);
        approx_vec(hit.point, Vec3::ZERO);
    }

    #[test]
    fn disk_std430_size_is_two_vec4() {
        assert_eq!(RAY_DISK_STD430_SIZE, 32);
        assert_eq!(RAY_DISK_STD430_SIZE, 2 * VEC4_STRIDE);
        assert_eq!(RAY_DISK_STD430_SIZE % 16, 0);
    }

    #[test]
    fn disk_std430_roundtrip_decodes_fields() {
        let disk = Disk::new(Vec3::new(1.5, -2.25, 3.75), Vec3::new(0.0, 1.0, 0.0), 4.5);
        let bytes = disk.to_std430();
        assert_eq!(bytes.len(), 32);
        let mut words = [0.0f32; 8];
        for (i, w) in words.iter_mut().enumerate() {
            let start = i * U32_STRIDE;
            let mut buf = [0u8; 4];
            buf.copy_from_slice(&bytes[start..start + U32_STRIDE]);
            *w = f32::from_le_bytes(buf);
        }
        assert_eq!(words, [1.5, -2.25, 3.75, 4.5, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn annulus_std430_roundtrip_decodes_fields() {
        let ann = Annulus::new(
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, 1.0),
            1.25,
            3.5,
        );
        let bytes = ann.to_std430();
        assert_eq!(bytes.len(), 32);
        let mut words = [0.0f32; 8];
        for (i, w) in words.iter_mut().enumerate() {
            let start = i * U32_STRIDE;
            let mut buf = [0u8; 4];
            buf.copy_from_slice(&bytes[start..start + U32_STRIDE]);
            *w = f32::from_le_bytes(buf);
        }
        assert_eq!(words, [0.0, 0.0, 1.0, 1.25, 0.0, 0.0, 1.0, 3.5]);
    }

    #[test]
    fn gpu_storage_bytes_is_multiple_of_sixteen() {
        for count in [0usize, 1, 2, 7, 64, 1000] {
            assert_eq!(disk_gpu_storage_bytes(count) % 16, 0);
            assert_eq!(annulus_gpu_storage_bytes(count) % 16, 0);
        }
    }

    #[test]
    fn gpu_storage_bytes_empty_reserves_one_element() {
        assert_eq!(disk_gpu_storage_bytes(0), RAY_DISK_STD430_SIZE);
        assert_eq!(annulus_gpu_storage_bytes(0), RAY_ANNULUS_STD430_SIZE);
    }

    #[test]
    fn gpu_storage_bytes_scales_with_count() {
        assert_eq!(disk_gpu_storage_bytes(1), 32);
        assert_eq!(disk_gpu_storage_bytes(4), 128);
        assert_eq!(annulus_gpu_storage_bytes(256), 256 * 32);
    }

    #[test]
    fn packing_many_disks_is_contiguous() {
        let disks = [
            Disk::new(Vec3::ZERO, unit_z(), 1.0),
            Disk::new(Vec3::new(1.0, 2.0, 3.0), Vec3::new(0.0, 1.0, 0.0), 0.5),
        ];
        let mut blob: Vec<u8> = Vec::new();
        for d in &disks {
            blob.extend_from_slice(&d.to_std430());
        }
        assert_eq!(blob.len(), disks.len() * RAY_DISK_STD430_SIZE);
        assert_eq!(blob.len(), disk_gpu_storage_bytes(disks.len()));
    }
}
