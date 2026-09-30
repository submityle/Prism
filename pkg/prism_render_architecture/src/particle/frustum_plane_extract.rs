//! Gribb-Hartmann view-frustum plane extraction for the particle culling pass
//! (design §12, §13).
//!
//! The culling stage needs the six oriented planes that bound the visible
//! volume so it can reject particles, bounds boxes, and whole emitters before
//! they reach simulation or rasterization. Given a combined *view-projection*
//! matrix this module extracts those planes analytically with the classic
//! Gribb-Hartmann method: each clip-space inequality that defines the canonical
//! view volume is a linear combination of the matrix rows, so a plane is just a
//! row sum or difference — no eigenvalues, no iteration, no trigonometry.
//!
//! This is the deterministic `CPU` reference the future `GPU` culling kernel
//! reproduces; the packed [`FrustumPlanes::to_std430`] block gives that kernel a
//! stable `std430` `ABI` (six `vec4` slots, one per plane).
//!
//! # Matrix convention
//! [`Mat4`] stores its sixteen coefficients **row-major**: element `(row, col)`
//! lives at index `row * 4 + col`, and [`Mat4::row`] returns a whole row as a
//! `vec4`. The extraction assumes the matrix transforms a *column* point
//! `p = (x, y, z, 1)` as `clip = M · p`, so the clip components are the four row
//! dot products (`clip.x = row0·p`, …, `clip.w = row3·p`). The canonical clip
//! volume is the OpenGL-style cube `-w <= x,y,z <= w`.
//!
//! # Plane convention
//! A [`Plane`] stores `n = (nx, ny, nz)` and `d` for the equation
//! `n·p + d = 0`. The *inside* half-space is `n·p + d >= 0`, so every extracted
//! plane's normal points **into** the frustum. Each plane is normalized (the
//! coefficients are divided by `|n|` via [`f32::sqrt`]) so `n·p + d` is a true
//! signed Euclidean distance; a degenerate normal shorter than
//! [`NORMALIZE_EPS`] is left unscaled instead of dividing by ~zero.
//!
//! # Determinism
//! The only non-`+ - * /` primitive used is [`f32::sqrt`] (inside
//! [`Plane::normalized`]); there are no transcendental functions and no platform
//! math, so extraction is bit-for-bit reproducible. `f32` magnitudes are never
//! compared with `==`/`!=`; tolerant comparisons go through [`CMP_EPS`] or
//! ordinary `<`/`>=` ordering against zero.
//!
//! # Layout
//! [`FrustumPlanes::to_std430`] packs the six planes as six consecutive `vec4`
//! slots (`nx, ny, nz, d` each), for [`FRUSTUM_PLANES_STD430_SIZE`] = `96`
//! bytes, matching the shared [`crate::particle::gpu_layout`] stride.

use crate::particle::gpu_layout::{storage_bytes, VEC4_STRIDE};

/// Epsilon for tolerant `f32` sign/inequality comparisons; direct `==`/`!=` is
/// forbidden by the crate's math rules.
pub const CMP_EPS: f32 = 1e-6;

/// Squared-length floor below which [`Plane::normalized`] treats a normal as
/// degenerate and returns the plane unscaled rather than dividing by ~zero.
pub const NORMALIZE_EPS: f32 = 1e-12;

/// Number of planes bounding a frustum: left, right, bottom, top, near, far.
pub const PLANE_COUNT: usize = 6;

/// Index of the left clipping plane inside [`FrustumPlanes::planes`].
pub const PLANE_LEFT: usize = 0;
/// Index of the right clipping plane inside [`FrustumPlanes::planes`].
pub const PLANE_RIGHT: usize = 1;
/// Index of the bottom clipping plane inside [`FrustumPlanes::planes`].
pub const PLANE_BOTTOM: usize = 2;
/// Index of the top clipping plane inside [`FrustumPlanes::planes`].
pub const PLANE_TOP: usize = 3;
/// Index of the near clipping plane inside [`FrustumPlanes::planes`].
pub const PLANE_NEAR: usize = 4;
/// Index of the far clipping plane inside [`FrustumPlanes::planes`].
pub const PLANE_FAR: usize = 5;

/// Byte size of one packed [`Plane`] in a `std430` buffer: `nx, ny, nz, d`
/// exactly fill one `vec4` slot
/// ([`crate::particle::gpu_layout::VEC4_STRIDE`]).
pub const PLANE_STD430_SIZE: usize = VEC4_STRIDE;

/// Byte size of the packed six-plane [`FrustumPlanes`] block: `6 × vec4`.
pub const FRUSTUM_PLANES_STD430_SIZE: usize = PLANE_STD430_SIZE * PLANE_COUNT;

/// A `4×4` matrix stored **row-major**: entry `(row, col)` is at
/// `m[row * 4 + col]`.
///
/// This module never multiplies two matrices; it only reads whole rows as
/// `vec4`s to build the Gribb-Hartmann plane combinations, so the row-major
/// convention is the entire contract. The provided constructors
/// ([`Mat4::identity`], [`Mat4::orthographic`], [`Mat4::perspective`]) write the
/// same layout a caller's own view-projection product must follow.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat4 {
    /// The sixteen coefficients in row-major order.
    pub m: [f32; 16],
}

impl Mat4 {
    /// Builds a matrix from sixteen row-major coefficients.
    #[must_use]
    pub const fn new(m: [f32; 16]) -> Self {
        Self { m }
    }

    /// The `4×4` identity matrix. Extracting its frustum yields the canonical
    /// `[-1, 1]³` clip cube with axis-aligned, unit-length planes.
    #[must_use]
    pub const fn identity() -> Self {
        Self {
            m: [
                1.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, 0.0, //
                0.0, 0.0, 0.0, 1.0, //
            ],
        }
    }

    /// An OpenGL-style orthographic projection mapping the box
    /// `[l, r] × [b, t] × [n, f]` to the `[-1, 1]³` clip cube.
    ///
    /// Uses only division; a degenerate extent (`r == l`, `t == b`, or
    /// `f == n`, tested against [`CMP_EPS`]) collapses that axis' scale to `0`
    /// rather than dividing by ~zero, leaving the corresponding planes
    /// degenerate but well defined.
    #[must_use]
    pub fn orthographic(l: f32, r: f32, b: f32, t: f32, n: f32, f: f32) -> Self {
        let rl = r - l;
        let tb = t - b;
        let fnn = f - n;
        let sx = if rl.abs() < CMP_EPS { 0.0 } else { 2.0 / rl };
        let sy = if tb.abs() < CMP_EPS { 0.0 } else { 2.0 / tb };
        let sz = if fnn.abs() < CMP_EPS { 0.0 } else { -2.0 / fnn };
        let tx = if rl.abs() < CMP_EPS {
            0.0
        } else {
            -(r + l) / rl
        };
        let ty = if tb.abs() < CMP_EPS {
            0.0
        } else {
            -(t + b) / tb
        };
        let tz = if fnn.abs() < CMP_EPS {
            0.0
        } else {
            -(f + n) / fnn
        };
        Self {
            m: [
                sx, 0.0, 0.0, tx, //
                0.0, sy, 0.0, ty, //
                0.0, 0.0, sz, tz, //
                0.0, 0.0, 0.0, 1.0, //
            ],
        }
    }

    /// An OpenGL-style perspective projection for the symmetric or asymmetric
    /// frustum `[l, r] × [b, t]` at the near plane and depth range `[n, f]`.
    ///
    /// Uses only division; degenerate extents (tested against [`CMP_EPS`])
    /// collapse the affected terms to `0` instead of dividing by ~zero. The
    /// camera looks down `-z`, so the visible depth range is `z ∈ [-f, -n]`.
    #[must_use]
    pub fn perspective(l: f32, r: f32, b: f32, t: f32, n: f32, f: f32) -> Self {
        let rl = r - l;
        let tb = t - b;
        let fnn = f - n;
        let a = if rl.abs() < CMP_EPS {
            0.0
        } else {
            2.0 * n / rl
        };
        let e = if tb.abs() < CMP_EPS {
            0.0
        } else {
            2.0 * n / tb
        };
        let cx = if rl.abs() < CMP_EPS {
            0.0
        } else {
            (r + l) / rl
        };
        let cy = if tb.abs() < CMP_EPS {
            0.0
        } else {
            (t + b) / tb
        };
        let cz = if fnn.abs() < CMP_EPS {
            0.0
        } else {
            -(f + n) / fnn
        };
        let dz = if fnn.abs() < CMP_EPS {
            0.0
        } else {
            -2.0 * f * n / fnn
        };
        Self {
            m: [
                a, 0.0, cx, 0.0, //
                0.0, e, cy, 0.0, //
                0.0, 0.0, cz, dz, //
                0.0, 0.0, -1.0, 0.0, //
            ],
        }
    }

    /// Returns row `i` (`0..=3`) as a `[f32; 4]`.
    ///
    /// # Panics
    /// Panics when `i >= 4`.
    #[must_use]
    pub fn row(&self, i: usize) -> [f32; 4] {
        assert!(i < 4, "Mat4 row index out of range");
        let base = i * 4;
        [
            self.m[base],
            self.m[base + 1],
            self.m[base + 2],
            self.m[base + 3],
        ]
    }
}

/// An oriented plane written as the coefficients of `n·p + d = 0`.
///
/// The half-space `n·p + d >= 0` is the *inside* the frustum keeps; every plane
/// [`extract_frustum_planes`] returns has its normal pointing inward and is
/// normalized so [`Plane::signed_distance`] is a true Euclidean distance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plane {
    /// The `x` component of the plane normal `n`.
    pub nx: f32,
    /// The `y` component of the plane normal `n`.
    pub ny: f32,
    /// The `z` component of the plane normal `n`.
    pub nz: f32,
    /// The plane constant `d` in `n·p + d = 0`.
    pub d: f32,
}

impl Plane {
    /// Builds a plane from raw `n·p + d = 0` coefficients without normalizing.
    #[must_use]
    pub const fn new(nx: f32, ny: f32, nz: f32, d: f32) -> Self {
        Self { nx, ny, nz, d }
    }

    /// The squared length of the normal `n`.
    #[must_use]
    pub fn normal_length_squared(&self) -> f32 {
        self.nx * self.nx + self.ny * self.ny + self.nz * self.nz
    }

    /// The Euclidean length `|n|` of the normal (uses [`f32::sqrt`]).
    #[must_use]
    pub fn normal_length(&self) -> f32 {
        self.normal_length_squared().sqrt()
    }

    /// Evaluates `n·p + d`, the signed distance scaled by `|n|`.
    ///
    /// The sign tells which half-space `p` lies in (`>= 0` is inside); the
    /// magnitude is a true Euclidean distance once the plane is normalized.
    #[must_use]
    pub fn signed_distance(&self, p: [f32; 3]) -> f32 {
        self.nx * p[0] + self.ny * p[1] + self.nz * p[2] + self.d
    }

    /// Returns an equivalent plane with a unit-length normal.
    ///
    /// Dividing `n` and `d` by `|n|` preserves the zero set and the sign of
    /// [`Plane::signed_distance`] while turning it into a metric distance. A
    /// degenerate normal (squared length below [`NORMALIZE_EPS`]) is returned
    /// unchanged rather than dividing by ~zero.
    #[must_use]
    pub fn normalized(&self) -> Self {
        let len_sq = self.normal_length_squared();
        if len_sq < NORMALIZE_EPS {
            return *self;
        }
        let inv = 1.0 / len_sq.sqrt();
        Self {
            nx: self.nx * inv,
            ny: self.ny * inv,
            nz: self.nz * inv,
            d: self.d * inv,
        }
    }

    /// Whether `p` lies on the inside half-space (`n·p + d >= -CMP_EPS`).
    ///
    /// The small negative tolerance keeps points exactly on the boundary
    /// classified as inside despite `f32` rounding.
    #[must_use]
    pub fn contains(&self, p: [f32; 3]) -> bool {
        self.signed_distance(p) >= -CMP_EPS
    }

    /// Packs the plane into its `std430` block as little-endian
    /// `nx, ny, nz, d`, spanning one `vec4` slot ([`PLANE_STD430_SIZE`] bytes).
    #[must_use]
    pub fn to_std430(&self) -> [u8; PLANE_STD430_SIZE] {
        let mut bytes = [0u8; PLANE_STD430_SIZE];
        bytes[0..4].copy_from_slice(&self.nx.to_le_bytes());
        bytes[4..8].copy_from_slice(&self.ny.to_le_bytes());
        bytes[8..12].copy_from_slice(&self.nz.to_le_bytes());
        bytes[12..16].copy_from_slice(&self.d.to_le_bytes());
        bytes
    }
}

/// The six inward-pointing, normalized planes bounding a view frustum.
///
/// The planes are stored in the fixed order left, right, bottom, top, near,
/// far (see the `PLANE_*` index constants). A point is inside the frustum when
/// it lies on the inside half-space of all six.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrustumPlanes {
    /// The six planes, indexed by the `PLANE_*` constants.
    pub planes: [Plane; PLANE_COUNT],
}

impl FrustumPlanes {
    /// Wraps six pre-built planes without renormalizing them.
    #[must_use]
    pub const fn new(planes: [Plane; PLANE_COUNT]) -> Self {
        Self { planes }
    }

    /// Whether `p` lies inside the frustum: on the inside half-space of every
    /// plane.
    #[must_use]
    pub fn point_inside(&self, p: [f32; 3]) -> bool {
        let mut i = 0;
        while i < PLANE_COUNT {
            if !self.planes[i].contains(p) {
                return false;
            }
            i += 1;
        }
        true
    }

    /// Conservative frustum-cull test for an axis-aligned box `[min, max]`.
    ///
    /// Returns `true` when the box is provably **entirely outside** the
    /// frustum: for some plane even the box corner farthest along that plane's
    /// (inward) normal — the *positive vertex* — still falls on the outside
    /// half-space. A `false` result means the box is inside or straddles a
    /// plane (the usual "keep it" answer for a cull test); this is the standard
    /// conservative test that never wrongly discards a visible box.
    #[must_use]
    pub fn aabb_outside(&self, min: [f32; 3], max: [f32; 3]) -> bool {
        let mut i = 0;
        while i < PLANE_COUNT {
            let plane = &self.planes[i];
            let px = if plane.nx >= 0.0 { max[0] } else { min[0] };
            let py = if plane.ny >= 0.0 { max[1] } else { min[1] };
            let pz = if plane.nz >= 0.0 { max[2] } else { min[2] };
            if plane.signed_distance([px, py, pz]) < -CMP_EPS {
                return true;
            }
            i += 1;
        }
        false
    }

    /// Packs the six planes into one `std430` block as six consecutive `vec4`
    /// slots, for [`FRUSTUM_PLANES_STD430_SIZE`] bytes total.
    #[must_use]
    pub fn to_std430(&self) -> [u8; FRUSTUM_PLANES_STD430_SIZE] {
        let mut bytes = [0u8; FRUSTUM_PLANES_STD430_SIZE];
        let mut i = 0;
        while i < PLANE_COUNT {
            let slot = self.planes[i].to_std430();
            let base = i * PLANE_STD430_SIZE;
            bytes[base..base + PLANE_STD430_SIZE].copy_from_slice(&slot);
            i += 1;
        }
        bytes
    }
}

/// Total `std430` byte size of a storage buffer holding `count` packed
/// [`FrustumPlanes`] blocks, clamped up to a single element per the shared
/// [`storage_bytes`] rule.
#[must_use]
pub fn gpu_storage_bytes(count: usize) -> usize {
    storage_bytes(FRUSTUM_PLANES_STD430_SIZE, count)
}

/// Extracts the six inward-pointing, normalized frustum planes from a
/// view-projection matrix using the Gribb-Hartmann method.
///
/// Writing the matrix rows as `r0..r3`, the raw (un-normalized) planes are the
/// clip-volume inequalities `-w <= x,y,z <= w` rearranged into `n·p + d >= 0`:
///
/// | plane  | combination |
/// |--------|-------------|
/// | left   | `r0 + r3`   |
/// | right  | `r3 - r0`   |
/// | bottom | `r1 + r3`   |
/// | top    | `r3 - r1`   |
/// | near   | `r2 + r3`   |
/// | far    | `r3 - r2`   |
///
/// Each combination is then normalized ([`Plane::normalized`]) so the stored
/// normals are unit length and [`Plane::signed_distance`] is metric.
#[must_use]
pub fn extract_frustum_planes(mat: &Mat4) -> FrustumPlanes {
    let r0 = mat.row(0);
    let r1 = mat.row(1);
    let r2 = mat.row(2);
    let r3 = mat.row(3);

    let left = Plane::new(r3[0] + r0[0], r3[1] + r0[1], r3[2] + r0[2], r3[3] + r0[3]).normalized();
    let right = Plane::new(r3[0] - r0[0], r3[1] - r0[1], r3[2] - r0[2], r3[3] - r0[3]).normalized();
    let bottom =
        Plane::new(r3[0] + r1[0], r3[1] + r1[1], r3[2] + r1[2], r3[3] + r1[3]).normalized();
    let top = Plane::new(r3[0] - r1[0], r3[1] - r1[1], r3[2] - r1[2], r3[3] - r1[3]).normalized();
    let near = Plane::new(r3[0] + r2[0], r3[1] + r2[1], r3[2] + r2[2], r3[3] + r2[3]).normalized();
    let far = Plane::new(r3[0] - r2[0], r3[1] - r2[1], r3[2] - r2[2], r3[3] - r2[3]).normalized();

    FrustumPlanes::new([left, right, bottom, top, near, far])
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_EPS: f32 = 1e-4;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < TEST_EPS
    }

    fn plane_approx(p: &Plane, nx: f32, ny: f32, nz: f32, d: f32) -> bool {
        approx(p.nx, nx) && approx(p.ny, ny) && approx(p.nz, nz) && approx(p.d, d)
    }

    #[test]
    fn plane_count_is_six() {
        let fp = extract_frustum_planes(&Mat4::identity());
        assert_eq!(fp.planes.len(), 6);
        assert_eq!(PLANE_COUNT, 6);
    }

    #[test]
    fn std430_sizes_are_multiples_of_sixteen() {
        assert_eq!(PLANE_STD430_SIZE, 16);
        assert_eq!(FRUSTUM_PLANES_STD430_SIZE, 96);
        assert_eq!(PLANE_STD430_SIZE % VEC4_STRIDE, 0);
        assert_eq!(FRUSTUM_PLANES_STD430_SIZE % VEC4_STRIDE, 0);
        assert_eq!(FRUSTUM_PLANES_STD430_SIZE % 16, 0);
    }

    #[test]
    fn identity_left_plane_is_x_ge_minus_one() {
        let fp = extract_frustum_planes(&Mat4::identity());
        assert!(plane_approx(&fp.planes[PLANE_LEFT], 1.0, 0.0, 0.0, 1.0));
    }

    #[test]
    fn identity_right_plane_is_x_le_one() {
        let fp = extract_frustum_planes(&Mat4::identity());
        assert!(plane_approx(&fp.planes[PLANE_RIGHT], -1.0, 0.0, 0.0, 1.0));
    }

    #[test]
    fn identity_bottom_plane_is_y_ge_minus_one() {
        let fp = extract_frustum_planes(&Mat4::identity());
        assert!(plane_approx(&fp.planes[PLANE_BOTTOM], 0.0, 1.0, 0.0, 1.0));
    }

    #[test]
    fn identity_top_plane_is_y_le_one() {
        let fp = extract_frustum_planes(&Mat4::identity());
        assert!(plane_approx(&fp.planes[PLANE_TOP], 0.0, -1.0, 0.0, 1.0));
    }

    #[test]
    fn identity_near_plane_is_z_ge_minus_one() {
        let fp = extract_frustum_planes(&Mat4::identity());
        assert!(plane_approx(&fp.planes[PLANE_NEAR], 0.0, 0.0, 1.0, 1.0));
    }

    #[test]
    fn identity_far_plane_is_z_le_one() {
        let fp = extract_frustum_planes(&Mat4::identity());
        assert!(plane_approx(&fp.planes[PLANE_FAR], 0.0, 0.0, -1.0, 1.0));
    }

    #[test]
    fn identity_origin_is_inside() {
        let fp = extract_frustum_planes(&Mat4::identity());
        assert!(fp.point_inside([0.0, 0.0, 0.0]));
    }

    #[test]
    fn identity_cube_corner_just_inside() {
        let fp = extract_frustum_planes(&Mat4::identity());
        assert!(fp.point_inside([0.99, -0.99, 0.99]));
    }

    #[test]
    fn point_outside_left_plane_is_rejected() {
        let fp = extract_frustum_planes(&Mat4::identity());
        assert!(!fp.point_inside([-1.5, 0.0, 0.0]));
        assert!(!fp.planes[PLANE_LEFT].contains([-1.5, 0.0, 0.0]));
    }

    #[test]
    fn point_outside_right_plane_is_rejected() {
        let fp = extract_frustum_planes(&Mat4::identity());
        assert!(!fp.point_inside([1.5, 0.0, 0.0]));
        assert!(!fp.planes[PLANE_RIGHT].contains([1.5, 0.0, 0.0]));
    }

    #[test]
    fn point_outside_bottom_plane_is_rejected() {
        let fp = extract_frustum_planes(&Mat4::identity());
        assert!(!fp.planes[PLANE_BOTTOM].contains([0.0, -2.0, 0.0]));
        assert!(!fp.point_inside([0.0, -2.0, 0.0]));
    }

    #[test]
    fn point_outside_top_plane_is_rejected() {
        let fp = extract_frustum_planes(&Mat4::identity());
        assert!(!fp.planes[PLANE_TOP].contains([0.0, 2.0, 0.0]));
        assert!(!fp.point_inside([0.0, 2.0, 0.0]));
    }

    #[test]
    fn point_outside_near_plane_is_rejected() {
        let fp = extract_frustum_planes(&Mat4::identity());
        assert!(!fp.planes[PLANE_NEAR].contains([0.0, 0.0, -2.0]));
        assert!(!fp.point_inside([0.0, 0.0, -2.0]));
    }

    #[test]
    fn point_outside_far_plane_is_rejected() {
        let fp = extract_frustum_planes(&Mat4::identity());
        assert!(!fp.planes[PLANE_FAR].contains([0.0, 0.0, 2.0]));
        assert!(!fp.point_inside([0.0, 0.0, 2.0]));
    }

    #[test]
    fn all_identity_normals_are_unit_length() {
        let fp = extract_frustum_planes(&Mat4::identity());
        for plane in fp.planes.iter() {
            assert!(approx(plane.normal_length(), 1.0));
        }
    }

    #[test]
    fn scaled_ortho_normals_are_unit_length() {
        let m = Mat4::orthographic(-4.0, 4.0, -3.0, 3.0, 1.0, 50.0);
        let fp = extract_frustum_planes(&m);
        for plane in fp.planes.iter() {
            assert!(approx(plane.normal_length(), 1.0));
        }
    }

    #[test]
    fn perspective_normals_are_unit_length() {
        let m = Mat4::perspective(-1.0, 1.0, -1.0, 1.0, 1.0, 101.0);
        let fp = extract_frustum_planes(&m);
        for plane in fp.planes.iter() {
            assert!(approx(plane.normal_length(), 1.0));
        }
    }

    #[test]
    fn scaled_ortho_side_planes_match_extents() {
        // Half-widths 2 in x, 2 in y; the side planes read off those bounds.
        let m = Mat4::orthographic(-2.0, 2.0, -2.0, 2.0, -1.0, 1.0);
        let fp = extract_frustum_planes(&m);
        // left: x >= -2  -> (1,0,0), d = 2
        assert!(plane_approx(&fp.planes[PLANE_LEFT], 1.0, 0.0, 0.0, 2.0));
        // right: x <= 2  -> (-1,0,0), d = 2
        assert!(plane_approx(&fp.planes[PLANE_RIGHT], -1.0, 0.0, 0.0, 2.0));
    }

    #[test]
    fn scaled_ortho_contains_interior_point() {
        let m = Mat4::orthographic(-4.0, 4.0, -3.0, 3.0, -10.0, 10.0);
        let fp = extract_frustum_planes(&m);
        assert!(fp.point_inside([3.0, 2.0, 5.0]));
        assert!(!fp.point_inside([5.0, 0.0, 0.0]));
        assert!(!fp.point_inside([0.0, 4.0, 0.0]));
    }

    #[test]
    fn symmetric_ortho_origin_inside() {
        let m = Mat4::orthographic(-5.0, 5.0, -5.0, 5.0, -5.0, 5.0);
        let fp = extract_frustum_planes(&m);
        assert!(fp.point_inside([0.0, 0.0, 0.0]));
    }

    #[test]
    fn perspective_interior_point_inside_camera_outside() {
        let m = Mat4::perspective(-1.0, 1.0, -1.0, 1.0, 1.0, 101.0);
        let fp = extract_frustum_planes(&m);
        // A point well down -z is inside the perspective frustum...
        assert!(fp.point_inside([0.0, 0.0, -10.0]));
        // ...while the camera origin sits in front of the near plane.
        assert!(!fp.point_inside([0.0, 0.0, 0.0]));
    }

    #[test]
    fn perspective_near_far_planes_bound_depth() {
        let m = Mat4::perspective(-1.0, 1.0, -1.0, 1.0, 2.0, 200.0);
        let fp = extract_frustum_planes(&m);
        // Near plane: z <= -2 inside -> (0,0,-1), d = -2.
        let near = fp.planes[PLANE_NEAR];
        assert!(approx(near.nx, 0.0) && approx(near.ny, 0.0));
        assert!(approx(near.nz, -1.0));
        assert!(approx(near.d, -2.0));
        // Far plane: z >= -200 inside -> (0,0,1), d = 200. Recovering a large
        // distance from the tiny normalized z-normal loses several f32 digits
        // (catastrophic cancellation in `-1 - cz`), so `d` is checked with a
        // tolerance proportional to the far distance rather than TEST_EPS.
        let far = fp.planes[PLANE_FAR];
        assert!(approx(far.nx, 0.0) && approx(far.ny, 0.0));
        assert!(approx(far.nz, 1.0));
        assert!((far.d - 200.0).abs() < 200.0 * 1e-3);
    }

    #[test]
    fn aabb_fully_outside_is_culled() {
        let fp = extract_frustum_planes(&Mat4::identity());
        // Box entirely to the right of x = 1.
        assert!(fp.aabb_outside([2.0, -0.5, -0.5], [3.0, 0.5, 0.5]));
    }

    #[test]
    fn aabb_straddling_plane_is_kept() {
        let fp = extract_frustum_planes(&Mat4::identity());
        // Box crosses the right plane; conservative test keeps it.
        assert!(!fp.aabb_outside([0.5, -0.5, -0.5], [1.5, 0.5, 0.5]));
    }

    #[test]
    fn aabb_inside_is_kept() {
        let fp = extract_frustum_planes(&Mat4::identity());
        assert!(!fp.aabb_outside([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]));
    }

    #[test]
    fn aabb_outside_below_bottom_is_culled() {
        let fp = extract_frustum_planes(&Mat4::identity());
        assert!(fp.aabb_outside([-0.5, -3.0, -0.5], [0.5, -2.0, 0.5]));
    }

    #[test]
    fn aabb_outside_beyond_far_is_culled() {
        let fp = extract_frustum_planes(&Mat4::identity());
        assert!(fp.aabb_outside([-0.5, -0.5, 2.0], [0.5, 0.5, 3.0]));
    }

    #[test]
    fn extraction_is_deterministic() {
        let m = Mat4::perspective(-1.5, 1.5, -1.0, 1.0, 1.0, 75.0);
        let a = extract_frustum_planes(&m);
        let b = extract_frustum_planes(&m);
        assert_eq!(a, b);
        assert_eq!(a.to_std430(), b.to_std430());
    }

    #[test]
    fn std430_layout_size_and_slotting() {
        let fp = extract_frustum_planes(&Mat4::identity());
        let bytes = fp.to_std430();
        assert_eq!(bytes.len(), 96);
        // The first slot equals the left plane's own packing.
        assert_eq!(&bytes[0..16], &fp.planes[PLANE_LEFT].to_std430());
        // The last slot equals the far plane's packing.
        assert_eq!(&bytes[80..96], &fp.planes[PLANE_FAR].to_std430());
    }

    #[test]
    fn plane_to_std430_roundtrips_components() {
        let plane = Plane::new(0.25, -0.5, 0.75, -1.25);
        let bytes = plane.to_std430();
        assert_eq!(
            f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            0.25
        );
        assert_eq!(
            f32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
            -0.5
        );
        assert_eq!(
            f32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]),
            0.75
        );
        assert_eq!(
            f32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]),
            -1.25
        );
    }

    #[test]
    fn gpu_storage_bytes_scales_and_clamps() {
        assert_eq!(gpu_storage_bytes(0), FRUSTUM_PLANES_STD430_SIZE);
        assert_eq!(gpu_storage_bytes(1), FRUSTUM_PLANES_STD430_SIZE);
        assert_eq!(gpu_storage_bytes(4), FRUSTUM_PLANES_STD430_SIZE * 4);
    }

    #[test]
    fn normalize_leaves_degenerate_normal_unchanged() {
        let degenerate = Plane::new(0.0, 0.0, 0.0, 3.0);
        let out = degenerate.normalized();
        assert_eq!(out, degenerate);
    }

    #[test]
    fn normalize_makes_unit_normal_and_keeps_sign() {
        let plane = Plane::new(0.0, 3.0, 4.0, 10.0);
        let n = plane.normalized();
        assert!(approx(n.normal_length(), 1.0));
        // d scaled by 1/|n| = 1/5.
        assert!(approx(n.d, 2.0));
        // Signed distance sign is preserved for a sample point.
        let p = [0.0, 1.0, 1.0];
        assert!(plane.signed_distance(p) > 0.0);
        assert!(n.signed_distance(p) > 0.0);
    }

    #[test]
    fn signed_distance_is_metric_after_normalization() {
        // Plane y = 0 with inward normal +y; point at y = 5 is 5 units inside.
        let plane = Plane::new(0.0, 2.0, 0.0, 0.0).normalized();
        assert!(approx(plane.signed_distance([100.0, 5.0, -7.0]), 5.0));
        assert!(approx(plane.signed_distance([0.0, -3.0, 0.0]), -3.0));
    }

    #[test]
    fn mat4_row_reads_row_major_layout() {
        let m = Mat4::new([
            0.0, 1.0, 2.0, 3.0, //
            4.0, 5.0, 6.0, 7.0, //
            8.0, 9.0, 10.0, 11.0, //
            12.0, 13.0, 14.0, 15.0, //
        ]);
        assert_eq!(m.row(0), [0.0, 1.0, 2.0, 3.0]);
        assert_eq!(m.row(3), [12.0, 13.0, 14.0, 15.0]);
    }

    #[test]
    fn contains_boundary_point_is_inside() {
        let plane = Plane::new(1.0, 0.0, 0.0, 1.0);
        // Exactly on the boundary x = -1.
        assert!(plane.contains([-1.0, 0.0, 0.0]));
    }

    #[test]
    fn frustum_planes_new_preserves_planes() {
        let planes = [
            Plane::new(1.0, 0.0, 0.0, 1.0),
            Plane::new(-1.0, 0.0, 0.0, 1.0),
            Plane::new(0.0, 1.0, 0.0, 1.0),
            Plane::new(0.0, -1.0, 0.0, 1.0),
            Plane::new(0.0, 0.0, 1.0, 1.0),
            Plane::new(0.0, 0.0, -1.0, 1.0),
        ];
        let fp = FrustumPlanes::new(planes);
        assert_eq!(fp.planes, planes);
    }
}
