//! Rigid mass-property contract: mass, centre of mass, and the inertia tensor
//! of a point-mass cloud (design §8, §10 rigid-body coupling).
//!
//! When particles are promoted to rigid debris chunks or when an emitter needs
//! a rest-pose inertia for its spawned bodies, the simulation needs the classic
//! rigid-body mass properties of a finite set of point masses: the total mass,
//! the centre of mass (COM), and the symmetric `3x3` inertia tensor taken about
//! that COM. These are the seed quantities every angular integrator and every
//! `XPBD` rigid constraint consumes.
//!
//! This module is a self-contained, `CPU`-verifiable contract. It hand-rolls
//! its own [`Vec3`] and symmetric [`Mat3`] (no sibling math is imported), draws
//! on nothing transcendental (only `+ - * /`), and never compares `f32` with
//! `==`. A zero-mass cloud is guarded everywhere: the COM degenerates to the
//! origin and [`MassProperties::of`] reports `None` rather than dividing by
//! zero. The optional `std430` packing reuses the shared strides from
//! [`crate::particle::gpu_layout`] so the tensor lands on a `16`-byte-aligned
//! `3 x vec4` (`48` bytes) storage footprint identical to the one a `WESL`
//! kernel would bind.

use crate::particle::gpu_layout::{storage_bytes, VEC4_STRIDE};

/// Masses at or below this magnitude are treated as zero, so the COM and
/// inertia divisions are guarded instead of producing `NaN` / infinities.
pub const MASS_EPS: f32 = 1e-12;

/// A minimal three-component vector; the module owns its own math so it pulls
/// in no sibling module beyond the shared `std430` strides.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3 {
    /// X component.
    pub x: f32,
    /// Y component.
    pub y: f32,
    /// Z component.
    pub z: f32,
}

impl Vec3 {
    /// The zero vector, also the degenerate COM of a zero-mass cloud.
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    /// Builds a vector from components.
    #[must_use]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Component-wise sum.
    #[must_use]
    #[expect(
        clippy::should_implement_trait,
        reason = "The particle math API is specified with named add/sub/neg methods for call-site uniformity, matching the sibling particle contracts; operator traits are intentionally not part of this internal type."
    )]
    pub fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }

    /// Component-wise difference `self - rhs`.
    #[must_use]
    #[expect(
        clippy::should_implement_trait,
        reason = "The particle math API is specified with named add/sub/neg methods for call-site uniformity, matching the sibling particle contracts; operator traits are intentionally not part of this internal type."
    )]
    pub fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }

    /// Uniform scale by a scalar.
    #[must_use]
    pub fn scale(self, s: f32) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }

    /// Squared Euclidean length, `x^2 + y^2 + z^2`.
    #[must_use]
    pub fn length_squared(self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }
}

/// A symmetric `3x3` matrix stored as its six unique entries.
///
/// The inertia tensor is symmetric by construction (`I_xy == I_yx`, etc.), so
/// only the upper triangle plus the diagonal is kept. Accessors reconstruct the
/// full matrix, and the `std430` packing writes the redundant lower triangle so
/// a shader binds a dense `3 x vec4`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Mat3 {
    /// Row/col `(0,0)` — `I_xx`.
    pub xx: f32,
    /// Row/col `(1,1)` — `I_yy`.
    pub yy: f32,
    /// Row/col `(2,2)` — `I_zz`.
    pub zz: f32,
    /// Off-diagonal `(0,1) == (1,0)` — `I_xy`.
    pub xy: f32,
    /// Off-diagonal `(0,2) == (2,0)` — `I_xz`.
    pub xz: f32,
    /// Off-diagonal `(1,2) == (2,1)` — `I_yz`.
    pub yz: f32,
}

impl Mat3 {
    /// The all-zero (empty-cloud) tensor.
    pub const ZERO: Self = Self {
        xx: 0.0,
        yy: 0.0,
        zz: 0.0,
        xy: 0.0,
        xz: 0.0,
        yz: 0.0,
    };

    /// Builds a symmetric matrix from its six unique entries.
    #[must_use]
    pub const fn new(xx: f32, yy: f32, zz: f32, xy: f32, xz: f32, yz: f32) -> Self {
        Self {
            xx,
            yy,
            zz,
            xy,
            xz,
            yz,
        }
    }

    /// Reads entry `(row, col)`, exploiting symmetry. Out-of-range indices read
    /// as `0.0` rather than panicking, keeping the accessor total.
    #[must_use]
    pub fn get(self, row: usize, col: usize) -> f32 {
        match (row, col) {
            (0, 0) => self.xx,
            (1, 1) => self.yy,
            (2, 2) => self.zz,
            (0, 1) | (1, 0) => self.xy,
            (0, 2) | (2, 0) => self.xz,
            (1, 2) | (2, 1) => self.yz,
            _ => 0.0,
        }
    }

    /// Entry-wise sum of two symmetric matrices.
    #[must_use]
    #[expect(
        clippy::should_implement_trait,
        reason = "The particle math API is specified with named add/sub/neg methods for call-site uniformity, matching the sibling particle contracts; operator traits are intentionally not part of this internal type."
    )]
    pub fn add(self, rhs: Self) -> Self {
        Self::new(
            self.xx + rhs.xx,
            self.yy + rhs.yy,
            self.zz + rhs.zz,
            self.xy + rhs.xy,
            self.xz + rhs.xz,
            self.yz + rhs.yz,
        )
    }

    /// Uniform scale of every entry by `s`.
    #[must_use]
    pub fn scale(self, s: f32) -> Self {
        Self::new(
            self.xx * s,
            self.yy * s,
            self.zz * s,
            self.xy * s,
            self.xz * s,
            self.yz * s,
        )
    }

    /// The trace `I_xx + I_yy + I_zz`. For an inertia tensor this equals
    /// `2 * sum(m * r^2)`, a handy scalar invariant.
    #[must_use]
    pub fn trace(self) -> f32 {
        self.xx + self.yy + self.zz
    }

    /// `std430` byte stride of the packed tensor: three `vec4` rows.
    pub const STD430_STRIDE: usize = VEC4_STRIDE * 3;

    /// Packs the (symmetric) matrix into three `vec4` rows, the fourth lane of
    /// each row being `std430` padding set to `0.0`. Layout is row-major:
    /// `[xx, xy, xz, 0, xy, yy, yz, 0, xz, yz, zz, 0]`.
    #[must_use]
    pub fn pack_std430(self) -> [f32; 12] {
        [
            self.xx, self.xy, self.xz, 0.0, //
            self.xy, self.yy, self.yz, 0.0, //
            self.xz, self.yz, self.zz, 0.0,
        ]
    }

    /// Inverse of [`Mat3::pack_std430`]; reads the diagonal and upper triangle
    /// back out of a packed `3 x vec4` block.
    #[must_use]
    pub fn unpack_std430(raw: [f32; 12]) -> Self {
        Self::new(raw[0], raw[5], raw[10], raw[1], raw[2], raw[6])
    }

    /// Adds the parallel-axis (Huygens-Steiner) correction for shifting the
    /// reference point of an inertia tensor by `offset`, given the cloud's
    /// `total_mass`.
    ///
    /// If a tensor is expressed about the COM, `translate` moves it to a point
    /// displaced from the COM by `offset` (the direction of `offset` is
    /// irrelevant since only `|offset|^2` and the outer product `offset x
    /// offset` appear). This is exactly the per-point contribution form, so it
    /// composes with [`MassProperties::merge`].
    #[must_use]
    pub fn translate(self, total_mass: f32, offset: Vec3) -> Self {
        self.add(point_inertia(total_mass, offset))
    }
}

/// The inertia contribution of a single point mass `m` at position `r`
/// (relative to the reference point), using the covariance form
/// `I_xx = m (y^2 + z^2)`, `I_xy = -m x y`, and symmetric partners.
#[must_use]
fn point_inertia(m: f32, r: Vec3) -> Mat3 {
    let (x, y, z) = (r.x, r.y, r.z);
    Mat3::new(
        m * (y * y + z * z),
        m * (x * x + z * z),
        m * (x * x + y * y),
        -m * x * y,
        -m * x * z,
        -m * y * z,
    )
}

/// The rigid mass properties of a point-mass cloud: total mass, centre of mass,
/// and the symmetric inertia tensor taken *about that centre of mass*.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MassProperties {
    /// Sum of the point masses.
    pub total_mass: f32,
    /// Mass-weighted centroid; the origin for a zero-mass cloud.
    pub com: Vec3,
    /// Inertia tensor about [`MassProperties::com`].
    pub inertia: Mat3,
}

impl MassProperties {
    /// The empty / zero-mass properties: no mass, COM at the origin, zero
    /// tensor. Acts as the identity element for [`MassProperties::merge`].
    pub const EMPTY: Self = Self {
        total_mass: 0.0,
        com: Vec3::ZERO,
        inertia: Mat3::ZERO,
    };

    /// Computes the mass properties of `points` (each a `(position, mass)`
    /// pair). Returns `None` when the total mass is at or below [`MASS_EPS`],
    /// so the COM division and any downstream inverse-inertia stay guarded.
    ///
    /// Two deterministic passes: the first accumulates mass and the
    /// mass-weighted position sum (COM); the second accumulates the inertia
    /// tensor of every point relative to that COM.
    #[must_use]
    pub fn of(points: &[(Vec3, f32)]) -> Option<Self> {
        let mut total_mass = 0.0f32;
        let mut weighted = Vec3::ZERO;
        for &(p, m) in points {
            total_mass += m;
            weighted = weighted.add(p.scale(m));
        }
        if total_mass <= MASS_EPS {
            return None;
        }
        let inv = 1.0 / total_mass;
        let com = weighted.scale(inv);

        let mut inertia = Mat3::ZERO;
        for &(p, m) in points {
            inertia = inertia.add(point_inertia(m, p.sub(com)));
        }
        Some(Self {
            total_mass,
            com,
            inertia,
        })
    }

    /// Total mass of `points` without computing the COM or tensor.
    #[must_use]
    pub fn total_mass(points: &[(Vec3, f32)]) -> f32 {
        let mut total = 0.0f32;
        for &(_, m) in points {
            total += m;
        }
        total
    }

    /// The mass-weighted centre of mass of `points`, or `None` when the total
    /// mass is at or below [`MASS_EPS`].
    #[must_use]
    pub fn center_of_mass(points: &[(Vec3, f32)]) -> Option<Vec3> {
        let mut total = 0.0f32;
        let mut weighted = Vec3::ZERO;
        for &(p, m) in points {
            total += m;
            weighted = weighted.add(p.scale(m));
        }
        if total <= MASS_EPS {
            None
        } else {
            Some(weighted.scale(1.0 / total))
        }
    }

    /// This system's inertia tensor re-expressed about an arbitrary reference
    /// point `about`, via the parallel-axis theorem. When `about` equals the
    /// COM this is just [`MassProperties::inertia`].
    #[must_use]
    pub fn inertia_about(self, about: Vec3) -> Mat3 {
        self.inertia.translate(self.total_mass, self.com.sub(about))
    }

    /// Merges two systems into one, combining mass, COM, and the inertia tensor
    /// about the *combined* COM.
    ///
    /// Each system's tensor is first shifted from its own COM to the combined
    /// COM (parallel axis) and then summed, which is exact for point clouds and
    /// makes the operation associative and commutative up to float rounding. A
    /// pair of zero-mass systems merges to [`MassProperties::EMPTY`].
    #[must_use]
    pub fn merge(a: Self, b: Self) -> Self {
        let total_mass = a.total_mass + b.total_mass;
        if total_mass <= MASS_EPS {
            return Self::EMPTY;
        }
        let inv = 1.0 / total_mass;
        let com = a
            .com
            .scale(a.total_mass)
            .add(b.com.scale(b.total_mass))
            .scale(inv);
        let inertia = a
            .inertia
            .translate(a.total_mass, a.com.sub(com))
            .add(b.inertia.translate(b.total_mass, b.com.sub(com)));
        Self {
            total_mass,
            com,
            inertia,
        }
    }
}

/// `std430` byte size of a packed inertia tensor (`3 x vec4 == 48` bytes),
/// routed through the shared [`storage_bytes`] helper so it matches the stride
/// arithmetic every per-pass buffer contract uses.
#[must_use]
pub fn std430_bytes() -> usize {
    storage_bytes(VEC4_STRIDE, 3)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-3;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() <= EPS
    }

    fn close_vec(a: Vec3, b: Vec3) -> bool {
        close(a.x, b.x) && close(a.y, b.y) && close(a.z, b.z)
    }

    fn close_mat(a: Mat3, b: Mat3) -> bool {
        close(a.xx, b.xx)
            && close(a.yy, b.yy)
            && close(a.zz, b.zz)
            && close(a.xy, b.xy)
            && close(a.xz, b.xz)
            && close(a.yz, b.yz)
    }

    #[test]
    fn vec3_algebra_is_exact() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        assert_eq!(a.add(b), Vec3::new(5.0, 7.0, 9.0));
        assert_eq!(b.sub(a), Vec3::new(3.0, 3.0, 3.0));
        assert_eq!(a.scale(2.0), Vec3::new(2.0, 4.0, 6.0));
        assert_eq!(a.length_squared(), 14.0);
    }

    #[test]
    fn mat3_add_and_scale_are_entrywise() {
        let m = Mat3::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        assert_eq!(m.scale(2.0), Mat3::new(2.0, 4.0, 6.0, 8.0, 10.0, 12.0));
        assert_eq!(
            m.add(Mat3::new(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)),
            Mat3::new(2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
        );
    }

    #[test]
    fn mat3_get_respects_symmetry() {
        let m = Mat3::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        assert_eq!(m.get(0, 1), m.get(1, 0));
        assert_eq!(m.get(0, 2), m.get(2, 0));
        assert_eq!(m.get(1, 2), m.get(2, 1));
        assert_eq!(m.get(0, 0), 1.0);
        assert_eq!(m.get(1, 1), 2.0);
        assert_eq!(m.get(2, 2), 3.0);
        assert_eq!(m.get(9, 9), 0.0);
    }

    #[test]
    fn total_mass_sums_points() {
        let pts = [
            (Vec3::new(0.0, 0.0, 0.0), 1.0),
            (Vec3::new(1.0, 0.0, 0.0), 2.0),
            (Vec3::new(0.0, 1.0, 0.0), 3.0),
        ];
        assert!(close(MassProperties::total_mass(&pts), 6.0));
    }

    #[test]
    fn total_mass_of_empty_is_zero() {
        assert!(close(MassProperties::total_mass(&[]), 0.0));
    }

    #[test]
    fn center_of_mass_single_point_is_that_point() {
        let p = Vec3::new(2.0, -3.0, 4.0);
        let com = MassProperties::center_of_mass(&[(p, 5.0)]).unwrap();
        assert!(close_vec(com, p));
    }

    #[test]
    fn center_of_mass_symmetric_pair_is_midpoint() {
        let pts = [
            (Vec3::new(-2.0, 0.0, 0.0), 1.0),
            (Vec3::new(2.0, 0.0, 0.0), 1.0),
        ];
        let com = MassProperties::center_of_mass(&pts).unwrap();
        assert!(close_vec(com, Vec3::ZERO));
    }

    #[test]
    fn center_of_mass_weights_by_mass() {
        let pts = [
            (Vec3::new(0.0, 0.0, 0.0), 1.0),
            (Vec3::new(3.0, 0.0, 0.0), 3.0),
        ];
        let com = MassProperties::center_of_mass(&pts).unwrap();
        // Weighted toward the heavier point at x = 3.
        assert!(close_vec(com, Vec3::new(2.25, 0.0, 0.0)));
    }

    #[test]
    fn center_of_mass_zero_mass_is_none() {
        assert!(MassProperties::center_of_mass(&[(Vec3::new(1.0, 1.0, 1.0), 0.0)]).is_none());
    }

    #[test]
    fn of_zero_mass_is_none() {
        let pts = [
            (Vec3::new(1.0, 0.0, 0.0), 0.0),
            (Vec3::new(0.0, 1.0, 0.0), 0.0),
        ];
        assert!(MassProperties::of(&pts).is_none());
    }

    #[test]
    fn of_empty_is_none() {
        assert!(MassProperties::of(&[]).is_none());
    }

    #[test]
    fn single_point_inertia_about_com_is_zero() {
        // A single point coincides with its own COM, so r = 0 everywhere.
        let mp = MassProperties::of(&[(Vec3::new(5.0, -7.0, 2.0), 3.0)]).unwrap();
        assert!(close_mat(mp.inertia, Mat3::ZERO));
    }

    #[test]
    fn single_point_offset_gives_m_r_squared_terms() {
        // Mass m at distance r along +x; inertia about the origin.
        let m = 2.0;
        let r = 4.0;
        let mp = MassProperties::of(&[(Vec3::new(r, 0.0, 0.0), m)]).unwrap();
        let about_origin = mp.inertia_about(Vec3::ZERO);
        // x-axis carries no moment; y and z carry m r^2.
        assert!(close(about_origin.xx, 0.0));
        assert!(close(about_origin.yy, m * r * r));
        assert!(close(about_origin.zz, m * r * r));
        assert!(close(about_origin.xy, 0.0));
    }

    #[test]
    fn diagonal_point_produces_off_diagonal_term() {
        // Point at (1,1,0) about the origin: I_xy = -m x y = -m.
        let m = 2.0;
        let mp = MassProperties::of(&[(Vec3::new(1.0, 1.0, 0.0), m)]).unwrap();
        let i = mp.inertia_about(Vec3::ZERO);
        assert!(close(i.xy, -m));
        assert!(close(i.xz, 0.0));
        assert!(close(i.yz, 0.0));
    }

    #[test]
    fn symmetric_pair_on_axis_has_expected_diagonal() {
        // Equal masses at +-a along x; COM at origin.
        let m = 1.0;
        let a = 3.0;
        let pts = [(Vec3::new(-a, 0.0, 0.0), m), (Vec3::new(a, 0.0, 0.0), m)];
        let mp = MassProperties::of(&pts).unwrap();
        assert!(close_vec(mp.com, Vec3::ZERO));
        assert!(close(mp.inertia.xx, 0.0));
        assert!(close(mp.inertia.yy, 2.0 * m * a * a));
        assert!(close(mp.inertia.zz, 2.0 * m * a * a));
    }

    #[test]
    fn axis_aligned_cloud_diagonalizes() {
        // Six unit masses on the +-axes: perfectly symmetric, off-diagonals ~0.
        let m = 1.0;
        let pts = [
            (Vec3::new(1.0, 0.0, 0.0), m),
            (Vec3::new(-1.0, 0.0, 0.0), m),
            (Vec3::new(0.0, 1.0, 0.0), m),
            (Vec3::new(0.0, -1.0, 0.0), m),
            (Vec3::new(0.0, 0.0, 1.0), m),
            (Vec3::new(0.0, 0.0, -1.0), m),
        ];
        let i = MassProperties::of(&pts).unwrap().inertia;
        assert!(close(i.xy, 0.0));
        assert!(close(i.xz, 0.0));
        assert!(close(i.yz, 0.0));
        // Each axis pair contributes to the two orthogonal moments: I_xx = 4 m.
        assert!(close(i.xx, 4.0 * m));
        assert!(close(i.yy, 4.0 * m));
        assert!(close(i.zz, 4.0 * m));
    }

    #[test]
    fn inertia_is_positive_semidefinite_diagonal() {
        let pts = [
            (Vec3::new(1.0, 2.0, -1.0), 1.5),
            (Vec3::new(-2.0, 0.5, 3.0), 2.0),
            (Vec3::new(0.0, -1.0, 1.0), 0.5),
        ];
        let i = MassProperties::of(&pts).unwrap().inertia;
        assert!(i.xx >= 0.0);
        assert!(i.yy >= 0.0);
        assert!(i.zz >= 0.0);
    }

    #[test]
    fn trace_equals_twice_sum_m_r_squared() {
        let pts = [
            (Vec3::new(1.0, 2.0, 3.0), 1.0),
            (Vec3::new(-1.0, 0.0, 2.0), 2.0),
        ];
        let mp = MassProperties::of(&pts).unwrap();
        let mut sum = 0.0f32;
        for &(p, m) in &pts {
            sum += m * p.sub(mp.com).length_squared();
        }
        assert!(close(mp.inertia.trace(), 2.0 * sum));
    }

    #[test]
    fn translate_by_zero_is_identity() {
        let i = Mat3::new(1.0, 2.0, 3.0, 0.1, 0.2, 0.3);
        assert_eq!(i.translate(5.0, Vec3::ZERO), i);
    }

    #[test]
    fn parallel_axis_matches_direct_recomputation() {
        // Compute inertia about the COM, translate to a shifted reference, and
        // compare against directly summing point contributions about that same
        // reference.
        let pts = [
            (Vec3::new(1.0, -2.0, 0.5), 1.0),
            (Vec3::new(-1.5, 0.0, 2.0), 2.0),
            (Vec3::new(0.5, 3.0, -1.0), 0.75),
        ];
        let mp = MassProperties::of(&pts).unwrap();
        let reference = Vec3::new(2.0, -1.0, 4.0);
        let via_theorem = mp.inertia_about(reference);

        let mut direct = Mat3::ZERO;
        for &(p, m) in &pts {
            direct = direct.add(point_inertia(m, p.sub(reference)));
        }
        assert!(close_mat(via_theorem, direct));
    }

    #[test]
    fn merge_matches_whole_cloud() {
        let all = [
            (Vec3::new(1.0, 0.0, 0.0), 1.0),
            (Vec3::new(0.0, 2.0, 0.0), 2.0),
            (Vec3::new(-1.0, 0.0, 3.0), 1.5),
            (Vec3::new(0.0, -2.0, -1.0), 0.5),
        ];
        let left = MassProperties::of(&all[..2]).unwrap();
        let right = MassProperties::of(&all[2..]).unwrap();
        let merged = MassProperties::merge(left, right);
        let whole = MassProperties::of(&all).unwrap();

        assert!(close(merged.total_mass, whole.total_mass));
        assert!(close_vec(merged.com, whole.com));
        assert!(close_mat(merged.inertia, whole.inertia));
    }

    #[test]
    fn merge_is_commutative() {
        let a = MassProperties::of(&[(Vec3::new(2.0, 0.0, 0.0), 1.0)]).unwrap();
        let b = MassProperties::of(&[(Vec3::new(-1.0, 3.0, 0.0), 2.0)]).unwrap();
        let ab = MassProperties::merge(a, b);
        let ba = MassProperties::merge(b, a);
        assert!(close(ab.total_mass, ba.total_mass));
        assert!(close_vec(ab.com, ba.com));
        assert!(close_mat(ab.inertia, ba.inertia));
    }

    #[test]
    fn merge_with_empty_is_identity() {
        let a = MassProperties::of(&[
            (Vec3::new(1.0, 1.0, 0.0), 1.0),
            (Vec3::new(-1.0, 0.0, 2.0), 2.0),
        ])
        .unwrap();
        let merged = MassProperties::merge(a, MassProperties::EMPTY);
        assert!(close(merged.total_mass, a.total_mass));
        assert!(close_vec(merged.com, a.com));
        assert!(close_mat(merged.inertia, a.inertia));
    }

    #[test]
    fn merge_two_empty_systems_is_empty() {
        let merged = MassProperties::merge(MassProperties::EMPTY, MassProperties::EMPTY);
        assert_eq!(merged, MassProperties::EMPTY);
    }

    #[test]
    fn compute_is_bitwise_deterministic() {
        let pts = [
            (Vec3::new(0.3, -1.7, 2.9), 1.25),
            (Vec3::new(-2.1, 0.4, -0.6), 0.75),
            (Vec3::new(1.9, 2.2, -3.3), 2.5),
        ];
        let a = MassProperties::of(&pts).unwrap();
        let b = MassProperties::of(&pts).unwrap();
        assert_eq!(a.total_mass.to_bits(), b.total_mass.to_bits());
        assert_eq!(a.com.x.to_bits(), b.com.x.to_bits());
        assert_eq!(a.com.y.to_bits(), b.com.y.to_bits());
        assert_eq!(a.com.z.to_bits(), b.com.z.to_bits());
        assert_eq!(a.inertia.xx.to_bits(), b.inertia.xx.to_bits());
        assert_eq!(a.inertia.xy.to_bits(), b.inertia.xy.to_bits());
        assert_eq!(a.inertia.yz.to_bits(), b.inertia.yz.to_bits());
    }

    #[test]
    fn std430_size_is_48_and_multiple_of_16() {
        assert_eq!(Mat3::STD430_STRIDE, 48);
        assert_eq!(std430_bytes(), 48);
        assert_eq!(std430_bytes() % VEC4_STRIDE, 0);
    }

    #[test]
    fn std430_pack_writes_padding_lanes_zero() {
        let m = Mat3::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        let raw = m.pack_std430();
        // The fourth lane of each vec4 row is std430 padding.
        assert_eq!(raw[3].to_bits(), 0.0f32.to_bits());
        assert_eq!(raw[7].to_bits(), 0.0f32.to_bits());
        assert_eq!(raw[11].to_bits(), 0.0f32.to_bits());
        // Lower triangle mirrors the upper triangle.
        assert_eq!(raw[1].to_bits(), raw[4].to_bits());
        assert_eq!(raw[2].to_bits(), raw[8].to_bits());
        assert_eq!(raw[6].to_bits(), raw[9].to_bits());
    }

    #[test]
    fn std430_roundtrip_is_bit_exact() {
        let m = Mat3::new(1.5, -2.25, 3.75, 0.5, -0.125, 4.0);
        let back = Mat3::unpack_std430(m.pack_std430());
        assert_eq!(m.xx.to_bits(), back.xx.to_bits());
        assert_eq!(m.yy.to_bits(), back.yy.to_bits());
        assert_eq!(m.zz.to_bits(), back.zz.to_bits());
        assert_eq!(m.xy.to_bits(), back.xy.to_bits());
        assert_eq!(m.xz.to_bits(), back.xz.to_bits());
        assert_eq!(m.yz.to_bits(), back.yz.to_bits());
    }
}
