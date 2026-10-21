//! Reduced-coordinate rigid-body articulations.
//!
//! This module implements Featherstone's spatial-vector articulated-body
//! algorithm (ABA) for a rooted tree of rigid links.  The implementation uses
//! fixed-size six-dimensional vectors and matrices, so the cost of one forward
//! dynamics evaluation is linear in the number of links and does not depend on
//! a dense global mass matrix.
//!
//! The public model deliberately lives below [`reduced`](super) because the
//! existing reduced module is the home of coordinate-reduced physics in this
//! crate.  It is independent of the soft-body modal model in the sibling
//! modules.

use glam::{Quat, Vec3};

use crate::math::scalar::Real;
use crate::math::transform::Isometry;

mod mass;
pub(crate) use mass::ArticulationImpulseSolver;
pub mod collider;
pub mod handoff;
pub use collider::{
    ArticulationColliderDesc, ArticulationColliderError, ArticulationColliderId,
    ArticulationColliderSet,
};

const SIX: usize = 6;
const THREE: usize = 3;
const SOLVE_EPS: Real = 1.0e-8;
const AXIS_EPS: Real = 1.0e-6;

type Mat6 = [[Real; SIX]; SIX];
type Vec6 = [Real; SIX];
type Mat3N = [[Real; THREE]; THREE];

fn zero6() -> Vec6 {
    [0.0; SIX]
}

fn zero_mat6() -> Mat6 {
    [[0.0; SIX]; SIX]
}

fn cross_matrix(v: Vec3) -> [[Real; THREE]; THREE] {
    [[0.0, -v.z, v.y], [v.z, 0.0, -v.x], [-v.y, v.x, 0.0]]
}

fn mat3_transpose(m: [[Real; THREE]; THREE]) -> [[Real; THREE]; THREE] {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

fn mat3_mul(a: [[Real; THREE]; THREE], b: [[Real; THREE]; THREE]) -> [[Real; THREE]; THREE] {
    let mut out = [[0.0; THREE]; THREE];
    for i in 0..THREE {
        for j in 0..THREE {
            for k in 0..THREE {
                out[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    out
}

fn mat3_add(a: [[Real; THREE]; THREE], b: [[Real; THREE]; THREE]) -> [[Real; THREE]; THREE] {
    let mut out = [[0.0; THREE]; THREE];
    for i in 0..THREE {
        for j in 0..THREE {
            out[i][j] = a[i][j] + b[i][j];
        }
    }
    out
}

fn mat3_scale(a: [[Real; THREE]; THREE], s: Real) -> [[Real; THREE]; THREE] {
    let mut out = [[0.0; THREE]; THREE];
    for i in 0..THREE {
        for j in 0..THREE {
            out[i][j] = a[i][j] * s;
        }
    }
    out
}

fn vec6_add(a: Vec6, b: Vec6) -> Vec6 {
    let mut out = zero6();
    for i in 0..SIX {
        out[i] = a[i] + b[i];
    }
    out
}

fn vec6_sub(a: Vec6, b: Vec6) -> Vec6 {
    let mut out = zero6();
    for i in 0..SIX {
        out[i] = a[i] - b[i];
    }
    out
}

fn vec6_scale(a: Vec6, s: Real) -> Vec6 {
    let mut out = zero6();
    for i in 0..SIX {
        out[i] = a[i] * s;
    }
    out
}

fn mat6_vec(m: &Mat6, v: Vec6) -> Vec6 {
    let mut out = zero6();
    for i in 0..SIX {
        for j in 0..SIX {
            out[i] += m[i][j] * v[j];
        }
    }
    out
}

fn mat6_transpose(m: &Mat6) -> Mat6 {
    let mut out = zero_mat6();
    for i in 0..SIX {
        for j in 0..SIX {
            out[i][j] = m[j][i];
        }
    }
    out
}

fn mat6_mul(a: &Mat6, b: &Mat6) -> Mat6 {
    let mut out = zero_mat6();
    for i in 0..SIX {
        for j in 0..SIX {
            for k in 0..SIX {
                out[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    out
}

fn mat6_sub(a: &Mat6, b: &Mat6) -> Mat6 {
    let mut out = zero_mat6();
    for i in 0..SIX {
        for j in 0..SIX {
            out[i][j] = a[i][j] - b[i][j];
        }
    }
    out
}

fn mat6_add_in_place(dst: &mut Mat6, src: &Mat6) {
    for i in 0..SIX {
        for j in 0..SIX {
            dst[i][j] += src[i][j];
        }
    }
}

fn vec6_add_in_place(dst: &mut Vec6, src: Vec6) {
    for i in 0..SIX {
        dst[i] += src[i];
    }
}

// Jacobi eigensolve is bounded by fixed 3x3/6x6 blocks. Double precision
// avoids cancellation in articulated elimination; only the stored result is
// rounded to Real. Null inertia modes receive zero mobility, never identity.
fn symmetric_spectrum<const N: usize>(
    matrix: [[Real; N]; N],
    count: usize,
) -> Option<([f64; N], [[f64; N]; N])> {
    let mut a = [[0.0_f64; N]; N];
    let mut vectors = [[0.0_f64; N]; N];
    let mut scale = 0.0_f64;
    for i in 0..count {
        vectors[i][i] = 1.0;
        for j in 0..count {
            a[i][j] = (f64::from(matrix[i][j]) + f64::from(matrix[j][i])) * 0.5;
            if !a[i][j].is_finite() {
                return None;
            }
            scale = scale.max(a[i][j].abs());
        }
    }
    for _ in 0..(N * N * 32) {
        let mut p = 0;
        let mut q = 0;
        let mut maximum = 0.0_f64;
        for i in 0..count {
            for j in i + 1..count {
                if a[i][j].abs() > maximum {
                    maximum = a[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if maximum <= scale * 1.0e-14 {
            break;
        }
        let angle = 0.5 * (2.0 * a[p][q]).atan2(a[q][q] - a[p][p]);
        let (s, c) = angle.sin_cos();
        let app = a[p][p];
        let aqq = a[q][q];
        let apq = a[p][q];
        a[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        a[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;
        for k in 0..count {
            if k != p && k != q {
                let akp = a[k][p];
                let akq = a[k][q];
                a[k][p] = c * akp - s * akq;
                a[p][k] = a[k][p];
                a[k][q] = s * akp + c * akq;
                a[q][k] = a[k][q];
            }
            let vkp = vectors[k][p];
            let vkq = vectors[k][q];
            vectors[k][p] = c * vkp - s * vkq;
            vectors[k][q] = s * vkp + c * vkq;
        }
    }
    Some((std::array::from_fn(|i| a[i][i]), vectors))
}

fn symmetric_inverse<const N: usize>(matrix: [[Real; N]; N], count: usize) -> [[Real; N]; N] {
    let mut result = [[0.0; N]; N];
    if count == 1 {
        let value = matrix[0][0];
        if value > 0.0 && value.recip().is_finite() {
            result[0][0] = value.recip();
        }
        return result;
    }
    let Some((values, vectors)) = symmetric_spectrum(matrix, count) else {
        return result;
    };
    let scale = values.iter().fold(0.0_f64, |s, &v| s.max(v.abs()));
    for i in 0..count {
        for j in 0..count {
            let mut value = 0.0_f64;
            for k in 0..count {
                if values[k] > scale * 1.0e-10 {
                    value += vectors[i][k] * vectors[j][k] / values[k];
                }
            }
            result[i][j] = value as Real;
            if !result[i][j].is_finite() {
                return [[0.0; N]; N];
            }
        }
    }
    result
}

fn invert6(matrix: Mat6) -> Mat6 {
    symmetric_inverse(matrix, SIX)
}

/// A spatial motion vector `(angular, linear)` expressed in one body frame.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct SpatialMotion {
    /// Angular component in radians per second or radians per second squared.
    pub angular: Vec3,
    /// Linear component in metres per second or metres per second squared.
    pub linear: Vec3,
}

impl SpatialMotion {
    /// Creates a spatial motion vector.
    #[must_use]
    pub const fn new(angular: Vec3, linear: Vec3) -> Self {
        Self { angular, linear }
    }

    fn as_array(self) -> Vec6 {
        [
            self.angular.x,
            self.angular.y,
            self.angular.z,
            self.linear.x,
            self.linear.y,
            self.linear.z,
        ]
    }

    fn from_array(v: Vec6) -> Self {
        Self::new(Vec3::new(v[0], v[1], v[2]), Vec3::new(v[3], v[4], v[5]))
    }
}

/// A spatial force/wrench `(torque, force)` expressed in one body frame.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct SpatialForce {
    /// Torque about the frame origin.
    pub torque: Vec3,
    /// Force applied at the frame origin.
    pub force: Vec3,
}

impl SpatialForce {
    /// Creates a spatial force.
    #[must_use]
    pub const fn new(torque: Vec3, force: Vec3) -> Self {
        Self { torque, force }
    }

    /// Creates a force applied at a point in the same frame.
    #[must_use]
    pub fn at_point(point: Vec3, force: Vec3) -> Self {
        Self::new(point.cross(force), force)
    }

    fn as_array(self) -> Vec6 {
        [
            self.torque.x,
            self.torque.y,
            self.torque.z,
            self.force.x,
            self.force.y,
            self.force.z,
        ]
    }

    fn from_array(v: Vec6) -> Self {
        Self::new(Vec3::new(v[0], v[1], v[2]), Vec3::new(v[3], v[4], v[5]))
    }
}

/// A spatial transform mapping motion coordinates from a parent frame into a
/// child frame.
///
/// `rotation` maps vectors from the source frame to the target frame, while
/// `translation` is the vector from source origin to target origin expressed in
/// source coordinates.  This convention makes composition and the force dual
/// explicit and avoids hidden world-frame assumptions in ABA.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct SpatialTransform {
    /// Source-to-target coordinate rotation.
    pub rotation: Quat,
    /// Source-origin to target-origin vector in source coordinates.
    pub translation: Vec3,
}

impl SpatialTransform {
    /// Identity spatial transform.
    pub const IDENTITY: Self = Self {
        rotation: Quat::IDENTITY,
        translation: Vec3::ZERO,
    };

    /// Creates a spatial transform.
    #[must_use]
    pub const fn new(rotation: Quat, translation: Vec3) -> Self {
        Self {
            rotation,
            translation,
        }
    }

    /// Composes `self` (A→B) with `next` (B→C), returning A→C.
    #[must_use]
    pub fn then(self, next: Self) -> Self {
        Self {
            rotation: next.rotation * self.rotation,
            translation: self.translation + self.rotation.inverse() * next.translation,
        }
    }

    /// Returns the inverse transform.
    #[must_use]
    pub fn inverse(self) -> Self {
        let rotation = self.rotation.inverse();
        Self {
            rotation,
            translation: -(rotation * self.translation),
        }
    }

    /// Maps a motion vector from the source frame into the target frame.
    #[must_use]
    pub fn transform_motion(self, motion: SpatialMotion) -> SpatialMotion {
        let angular = self.rotation * motion.angular;
        let linear = self.rotation * (motion.linear + motion.angular.cross(self.translation));
        SpatialMotion::new(angular, linear)
    }

    /// Maps a force/wrench from the target frame back into the source frame.
    #[must_use]
    pub fn transform_force_transpose(self, force: SpatialForce) -> SpatialForce {
        let f = self.rotation.inverse() * force.force;
        let torque = self.rotation.inverse() * force.torque + self.translation.cross(f);
        SpatialForce::new(torque, f)
    }

    fn motion_matrix(self) -> Mat6 {
        let e = quat_matrix(self.rotation);
        let p = cross_matrix(self.translation);
        let ep = mat3_mul(e, p);
        let mut out = zero_mat6();
        for i in 0..THREE {
            for j in 0..THREE {
                out[i][j] = e[i][j];
                out[i + THREE][j + THREE] = e[i][j];
                out[i + THREE][j] = -ep[i][j];
            }
        }
        out
    }
}

fn quat_matrix(q: Quat) -> [[Real; THREE]; THREE] {
    let x = q * Vec3::X;
    let y = q * Vec3::Y;
    let z = q * Vec3::Z;
    [[x.x, y.x, z.x], [x.y, y.y, z.y], [x.z, y.z, z.z]]
}

/// A rigid body's spatial inertia about the frame origin.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct SpatialInertia {
    /// Total mass.
    pub mass: Real,
    /// Centre of mass in body-frame coordinates.
    pub center_of_mass: Vec3,
    /// Rotational inertia about the centre of mass in body-frame coordinates.
    pub inertia_com: [[Real; THREE]; THREE],
}

impl SpatialInertia {
    /// Creates a spatial inertia from a mass, centre of mass, and inertia tensor.
    #[must_use]
    pub fn new(mass: Real, center_of_mass: Vec3, inertia_com: [[Real; THREE]; THREE]) -> Self {
        Self {
            mass,
            center_of_mass,
            inertia_com,
        }
    }

    /// Creates a point-mass inertia at the frame origin.
    #[must_use]
    pub fn point_mass(mass: Real) -> Self {
        Self::new(mass, Vec3::ZERO, [[0.0; THREE]; THREE])
    }

    /// Creates a diagonal centre-of-mass inertia tensor.
    #[must_use]
    pub fn diagonal(mass: Real, center_of_mass: Vec3, principal: Vec3) -> Self {
        Self::new(
            mass,
            center_of_mass,
            [
                [principal.x, 0.0, 0.0],
                [0.0, principal.y, 0.0],
                [0.0, 0.0, principal.z],
            ],
        )
    }

    /// Returns the 6×6 spatial inertia matrix in `(angular, linear)` order.
    #[must_use]
    pub fn matrix(self) -> Mat6 {
        let c = cross_matrix(self.center_of_mass);
        let ct = mat3_transpose(c);
        let rotational = mat3_add(self.inertia_com, mat3_scale(mat3_mul(ct, c), self.mass));
        let top_right = mat3_scale(c, self.mass);
        let bottom_left = mat3_scale(ct, self.mass);
        let mut out = zero_mat6();
        for i in 0..THREE {
            for j in 0..THREE {
                out[i][j] = rotational[i][j];
                out[i][j + THREE] = top_right[i][j];
                out[i + THREE][j] = bottom_left[i][j];
                out[i + THREE][j + THREE] = if i == j { self.mass } else { 0.0 };
            }
        }
        out
    }

    fn gravity_force(self, gravity_in_body: Vec3) -> SpatialForce {
        SpatialForce::from_array(mat6_vec(
            &self.matrix(),
            [
                0.0,
                0.0,
                0.0,
                gravity_in_body.x,
                gravity_in_body.y,
                gravity_in_body.z,
            ],
        ))
    }
}

impl Default for SpatialInertia {
    fn default() -> Self {
        Self::point_mass(0.0)
    }
}

/// The supported minimal-coordinate joint families.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum ArticulationJointKind {
    /// No relative motion.
    Fixed,
    /// One angular coordinate about a normalized child-frame axis.
    Revolute { axis: Vec3 },
    /// One linear coordinate along a normalized child-frame axis.
    Prismatic { axis: Vec3 },
    /// Three angular coordinates, integrated as a local rotation vector.
    Spherical,
}

impl ArticulationJointKind {
    /// Returns the number of generalized coordinates.
    #[must_use]
    pub const fn dofs(self) -> usize {
        match self {
            Self::Fixed => 0,
            Self::Revolute { .. } | Self::Prismatic { .. } => 1,
            Self::Spherical => 3,
        }
    }
}

/// Position limits for a joint.  Only the first `dofs()` entries are used.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct JointLimits {
    /// Inclusive lower limits.
    pub lower: Vec3,
    /// Inclusive upper limits.
    pub upper: Vec3,
    /// Whether limits participate in state projection.
    pub enabled: bool,
}

impl JointLimits {
    /// Returns disabled limits.
    #[must_use]
    pub const fn unlimited() -> Self {
        Self {
            lower: Vec3::splat(-Real::INFINITY),
            upper: Vec3::splat(Real::INFINITY),
            enabled: false,
        }
    }

    /// Creates a scalar limit for revolute or prismatic joints.
    #[must_use]
    pub const fn scalar(lower: Real, upper: Real) -> Self {
        Self {
            lower: Vec3::new(lower, 0.0, 0.0),
            upper: Vec3::new(upper, 0.0, 0.0),
            enabled: true,
        }
    }

    /// Creates component limits for a spherical joint.
    #[must_use]
    pub const fn vector(lower: Vec3, upper: Vec3) -> Self {
        Self {
            lower,
            upper,
            enabled: true,
        }
    }
}

impl Default for JointLimits {
    fn default() -> Self {
        Self::unlimited()
    }
}

/// A bounded generalized motor.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct JointMotor {
    /// Position target in radians or metres.
    pub target_position: Vec3,
    /// Velocity target in radians/s or metres/s.
    pub target_velocity: Vec3,
    /// Position feedback gain.
    pub stiffness: Real,
    /// Velocity feedback gain.
    pub damping: Real,
    /// Per-axis absolute actuator force/torque limit.
    pub max_force: Vec3,
}

impl JointMotor {
    /// Creates a motor with a position and velocity target.
    #[must_use]
    pub const fn new(
        target_position: Vec3,
        target_velocity: Vec3,
        stiffness: Real,
        damping: Real,
        max_force: Vec3,
    ) -> Self {
        Self {
            target_position,
            target_velocity,
            stiffness,
            damping,
            max_force,
        }
    }

    fn torque(self, q: Vec3, qd: Vec3, dofs: usize) -> Vec3 {
        let mut out = Vec3::ZERO;
        for i in 0..dofs {
            let target = self.stiffness * (self.target_position[i] - q[i])
                + self.damping * (self.target_velocity[i] - qd[i]);
            let limit = self.max_force[i].abs();
            out[i] = if limit.is_finite() {
                target.clamp(-limit, limit)
            } else {
                target
            };
        }
        out
    }
}

/// Complete joint configuration used by an articulation link.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ArticulationJoint {
    /// Joint family and motion subspace.
    pub kind: ArticulationJointKind,
    /// Position limits.
    pub limits: JointLimits,
    /// Optional bounded servo.
    pub motor: Option<JointMotor>,
    /// Physical viscous coefficient in joint space.
    ///
    /// Revolute and spherical axes use N*m*s/rad; prismatic axes use N*s/m.
    /// Integration solves `(M + dt * C) * qdd = tau - bias - C * qd`, so high
    /// damping remains stable without changing contact or impulse inverse mass.
    pub damping: Real,
    /// Reflected motor/gear inertia added in joint space.
    pub armature: Real,
}

impl ArticulationJoint {
    /// Creates an unconstrained joint with no motor or damping.
    #[must_use]
    pub const fn new(kind: ArticulationJointKind) -> Self {
        Self {
            kind,
            limits: JointLimits::unlimited(),
            motor: None,
            damping: 0.0,
            armature: 0.0,
        }
    }

    /// Creates a revolute joint about `axis`.
    #[must_use]
    pub fn revolute(axis: Vec3) -> Self {
        Self::new(ArticulationJointKind::Revolute { axis })
    }

    /// Creates a prismatic joint along `axis`.
    #[must_use]
    pub fn prismatic(axis: Vec3) -> Self {
        Self::new(ArticulationJointKind::Prismatic { axis })
    }

    /// Creates a three-degree-of-freedom spherical joint.
    #[must_use]
    pub const fn spherical() -> Self {
        Self::new(ArticulationJointKind::Spherical)
    }

    /// Creates a fixed joint.
    #[must_use]
    pub const fn fixed() -> Self {
        Self::new(ArticulationJointKind::Fixed)
    }

    /// Sets position limits.
    #[must_use]
    pub const fn with_limits(mut self, limits: JointLimits) -> Self {
        self.limits = limits;
        self
    }

    /// Sets the servo motor.
    #[must_use]
    pub const fn with_motor(mut self, motor: JointMotor) -> Self {
        self.motor = Some(motor);
        self
    }

    /// Sets the physical joint-space viscous coefficient.
    ///
    /// This passive damping is distinct from motor feedback damping.
    #[must_use]
    pub const fn with_damping(mut self, damping: Real) -> Self {
        self.damping = damping;
        self
    }

    /// Sets reflected joint-space inertia.
    #[must_use]
    pub const fn with_armature(mut self, armature: Real) -> Self {
        self.armature = armature;
        self
    }
}

/// Fixed data for one link in the articulation tree.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ArticulationLink {
    /// Parent link, or `None` for a link attached to the base frame.
    pub parent: Option<usize>,
    /// Rest transform from parent frame to this link's joint frame.
    pub x_tree: SpatialTransform,
    /// Rigid-body inertia about the link frame.
    pub inertia: SpatialInertia,
    /// Joint configuration connecting this link to its parent.
    pub joint: ArticulationJoint,
}

/// Fixed or floating base mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum ArticulationBase {
    /// The base frame is immovable and has no generalized coordinates.
    Fixed,
    /// The base is a six-degree-of-freedom floating rigid body.
    Floating,
}

/// Errors detected while building an articulation.
#[derive(Clone, Debug, PartialEq)]
pub enum ArticulationBuildError {
    /// A link referred to itself or to a link added later.
    ParentMustPrecedeChild { link: usize, parent: usize },
    /// A parent index was outside the link array.
    ParentOutOfRange { link: usize, parent: usize },
    /// A joint axis was zero, non-finite, or otherwise invalid.
    InvalidAxis { link: usize },
    /// A mass or inertia tensor was not finite or was not physically usable.
    InvalidInertia { link: usize },
    /// A limit had lower greater than upper.
    InvalidLimits { link: usize },
    /// The base inertia was invalid for a floating base.
    InvalidBaseInertia,
    /// A rest transform, base pose, or gravity vector was invalid.
    InvalidTransform { link: Option<usize> },
    /// Motor, damping, or armature parameters were invalid.
    InvalidJointParameters { link: usize },
}

/// Builder for deterministic, parent-before-child articulation trees.
#[derive(Clone, Debug)]
pub struct ArticulationBuilder {
    base_mode: ArticulationBase,
    base_inertia: SpatialInertia,
    base_pose: Isometry,
    gravity: Vec3,
    links: Vec<ArticulationLink>,
}

impl ArticulationBuilder {
    /// Creates a builder with an identity base pose and zero gravity.
    #[must_use]
    pub fn new(base_mode: ArticulationBase, base_inertia: SpatialInertia) -> Self {
        Self {
            base_mode,
            base_inertia,
            base_pose: Isometry::IDENTITY,
            gravity: Vec3::ZERO,
            links: Vec::new(),
        }
    }

    /// Sets the initial/world pose of the base.
    #[must_use]
    pub fn with_base_pose(mut self, pose: Isometry) -> Self {
        self.base_pose = pose;
        self
    }

    /// Sets uniform world gravity.
    #[must_use]
    pub fn with_gravity(mut self, gravity: Vec3) -> Self {
        self.gravity = gravity;
        self
    }

    /// Adds one link.  A parent must already have been added, which fixes the
    /// outward/inward scan order and makes serialized results deterministic.
    pub fn add_link(
        &mut self,
        parent: Option<usize>,
        x_tree: SpatialTransform,
        inertia: SpatialInertia,
        joint: ArticulationJoint,
    ) -> Result<usize, ArticulationBuildError> {
        let link = self.links.len();
        if let Some(parent_index) = parent {
            if parent_index >= link {
                return Err(if parent_index == link {
                    ArticulationBuildError::ParentMustPrecedeChild {
                        link,
                        parent: parent_index,
                    }
                } else {
                    ArticulationBuildError::ParentOutOfRange {
                        link,
                        parent: parent_index,
                    }
                });
            }
        }
        validate_link(link, &inertia, &joint)?;
        if !valid_pose(Isometry::new(x_tree.translation, x_tree.rotation)) {
            return Err(ArticulationBuildError::InvalidTransform { link: Some(link) });
        }
        self.links.push(ArticulationLink {
            parent,
            x_tree,
            inertia,
            joint,
        });
        Ok(link)
    }

    /// Consumes the builder and validates the complete model.
    pub fn build(self) -> Result<Articulation, ArticulationBuildError> {
        if !valid_pose(self.base_pose) || !self.gravity.is_finite() {
            return Err(ArticulationBuildError::InvalidTransform { link: None });
        }
        if self.base_mode == ArticulationBase::Floating {
            validate_inertia(&self.base_inertia)
                .map_err(|_| ArticulationBuildError::InvalidBaseInertia)?;
            if self.base_inertia.mass <= SOLVE_EPS {
                return Err(ArticulationBuildError::InvalidBaseInertia);
            }
        }
        let mut dof_offsets = Vec::with_capacity(self.links.len());
        let mut dofs = 0;
        for link in &self.links {
            dof_offsets.push(dofs);
            dofs += link.joint.kind.dofs();
        }
        Ok(Articulation {
            base_mode: self.base_mode,
            base_inertia: self.base_inertia,
            base_pose: self.base_pose,
            base_velocity: SpatialMotion::default(),
            gravity: self.gravity,
            links: self.links,
            dof_offsets,
            q: vec![0.0; dofs],
            qd: vec![0.0; dofs],
            link_forces: Vec::new(),
            base_force: SpatialForce::default(),
            broken_joint: None,
        })
    }
}

fn validate_inertia(inertia: &SpatialInertia) -> Result<(), ()> {
    if !inertia.mass.is_finite() || inertia.mass < 0.0 || !inertia.center_of_mass.is_finite() {
        return Err(());
    }
    for row in inertia.inertia_com {
        for value in row {
            if !value.is_finite() {
                return Err(());
            }
        }
    }
    let tensor = inertia.inertia_com;
    let scale = tensor
        .iter()
        .flatten()
        .fold(0.0_f64, |s, &v| s.max(f64::from(v).abs()));
    for i in 0..THREE {
        for j in 0..THREE {
            if f64::from(tensor[i][j] - tensor[j][i]).abs() > scale * 1.0e-6 {
                return Err(());
            }
        }
    }
    // A physical inertia is the trace of a positive semidefinite second
    // moment minus that second moment. This also checks the triangle
    // inequalities and admits point/line masses without invented inertia.
    let half_trace = (tensor[0][0] + tensor[1][1] + tensor[2][2]) * 0.5;
    let mut second_moment = [[0.0; THREE]; THREE];
    for i in 0..THREE {
        for j in 0..THREE {
            second_moment[i][j] = if i == j { half_trace } else { 0.0 } - tensor[i][j];
        }
    }
    let (eigenvalues, _) = symmetric_spectrum(second_moment, THREE).ok_or(())?;
    if eigenvalues.iter().any(|&value| value < -scale * 1.0e-6) {
        return Err(());
    }
    let matrix = inertia.matrix();
    for row in matrix {
        for value in row {
            if !value.is_finite() {
                return Err(());
            }
        }
    }
    Ok(())
}

fn valid_pose(pose: Isometry) -> bool {
    pose.translation.is_finite()
        && pose.rotation.is_finite()
        && (pose.rotation.length_squared() - 1.0).abs() <= 1.0e-4
}

fn validate_link(
    link: usize,
    inertia: &SpatialInertia,
    joint: &ArticulationJoint,
) -> Result<(), ArticulationBuildError> {
    validate_inertia(inertia).map_err(|_| ArticulationBuildError::InvalidInertia { link })?;
    if !joint.damping.is_finite()
        || joint.damping < 0.0
        || !joint.armature.is_finite()
        || joint.armature < 0.0
        || joint.motor.is_some_and(|motor| {
            !motor.target_position.is_finite()
                || !motor.target_velocity.is_finite()
                || !motor.stiffness.is_finite()
                || motor.stiffness < 0.0
                || !motor.damping.is_finite()
                || motor.damping < 0.0
                || motor
                    .max_force
                    .to_array()
                    .iter()
                    .any(|v| v.is_nan() || *v < 0.0)
        })
    {
        return Err(ArticulationBuildError::InvalidJointParameters { link });
    }
    if joint.kind == ArticulationJointKind::Fixed && joint.damping != 0.0 {
        return Err(ArticulationBuildError::InvalidJointParameters { link });
    }
    match joint.kind {
        ArticulationJointKind::Revolute { axis } | ArticulationJointKind::Prismatic { axis } => {
            if !axis.is_finite()
                || !axis.length_squared().is_finite()
                || axis.length_squared() <= AXIS_EPS * AXIS_EPS
            {
                return Err(ArticulationBuildError::InvalidAxis { link });
            }
        }
        ArticulationJointKind::Fixed | ArticulationJointKind::Spherical => {}
    }
    if joint.limits.enabled {
        for i in 0..THREE {
            if joint.limits.lower[i] > joint.limits.upper[i]
                || !joint.limits.lower[i].is_finite()
                || !joint.limits.upper[i].is_finite()
            {
                return Err(ArticulationBuildError::InvalidLimits { link });
            }
        }
    }
    Ok(())
}

/// A mutable reduced-coordinate articulation state and its external inputs.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct Articulation {
    base_mode: ArticulationBase,
    base_inertia: SpatialInertia,
    base_pose: Isometry,
    base_velocity: SpatialMotion,
    gravity: Vec3,
    links: Vec<ArticulationLink>,
    dof_offsets: Vec<usize>,
    q: Vec<Real>,
    qd: Vec<Real>,
    link_forces: Vec<SpatialForce>,
    base_force: SpatialForce,
    broken_joint: Option<usize>,
}

impl Articulation {
    /// Number of links.
    #[must_use]
    pub fn link_count(&self) -> usize {
        self.links.len()
    }

    /// Number of joint generalized coordinates, excluding a floating base.
    #[must_use]
    pub fn dof_count(&self) -> usize {
        self.q.len()
    }

    /// Returns the base mode.
    #[must_use]
    pub const fn base_mode(&self) -> ArticulationBase {
        self.base_mode
    }

    /// Returns the immutable link model.
    #[must_use]
    pub fn links(&self) -> &[ArticulationLink] {
        &self.links
    }

    /// Returns the generalized positions.
    #[must_use]
    pub fn positions(&self) -> &[Real] {
        &self.q
    }

    /// Returns the generalized velocities.
    #[must_use]
    pub fn velocities(&self) -> &[Real] {
        &self.qd
    }

    /// Returns the floating-base pose, or identity for a fixed base.
    #[must_use]
    pub fn base_pose(&self) -> Isometry {
        self.base_pose
    }

    /// Returns the floating-base spatial velocity.
    #[must_use]
    pub fn base_velocity(&self) -> SpatialMotion {
        self.base_velocity
    }

    /// Returns the configured gravity.
    #[must_use]
    pub fn gravity(&self) -> Vec3 {
        self.gravity
    }

    /// Sets one generalized position and clears no other state.
    pub fn set_position(&mut self, dof: usize, value: Real) -> bool {
        if !value.is_finite() {
            return false;
        }
        if let Some(slot) = self.q.get_mut(dof) {
            *slot = value;
            true
        } else {
            false
        }
    }

    /// Sets one generalized velocity.
    pub fn set_velocity(&mut self, dof: usize, value: Real) -> bool {
        if !value.is_finite() {
            return false;
        }
        if let Some(slot) = self.qd.get_mut(dof) {
            *slot = value;
            true
        } else {
            false
        }
    }

    /// Sets the floating-base pose when this is a floating articulation.
    pub fn set_base_pose(&mut self, pose: Isometry) -> bool {
        if self.base_mode == ArticulationBase::Floating && valid_pose(pose) {
            self.base_pose = pose;
            true
        } else {
            false
        }
    }

    /// Sets the floating-base velocity when this is a floating articulation.
    pub fn set_base_velocity(&mut self, velocity: SpatialMotion) -> bool {
        if self.base_mode == ArticulationBase::Floating
            && velocity.angular.is_finite()
            && velocity.linear.is_finite()
        {
            self.base_velocity = velocity;
            true
        } else {
            false
        }
    }

    /// Sets a link-frame external wrench retained until replaced or cleared.
    pub fn set_link_force(&mut self, link: usize, force: SpatialForce) -> bool {
        if link >= self.links.len() || !force.as_array().iter().all(|v| v.is_finite()) {
            return false;
        }
        if self.link_forces.len() != self.links.len() {
            self.link_forces = vec![SpatialForce::default(); self.links.len()];
        }
        self.link_forces[link] = force;
        true
    }

    /// Sets the base-frame external wrench.
    pub fn set_base_force(&mut self, force: SpatialForce) -> bool {
        if !force.as_array().iter().all(|v| v.is_finite()) {
            return false;
        }
        self.base_force = force;
        true
    }

    /// Clears all externally applied wrenches.
    pub fn clear_forces(&mut self) {
        self.link_forces.clear();
        self.base_force = SpatialForce::default();
    }

    /// Computes link world poses in insertion order.
    #[must_use]
    pub fn forward_kinematics(&self) -> Vec<Isometry> {
        let mut poses = vec![Isometry::IDENTITY; self.links.len()];
        for i in 0..self.links.len() {
            let parent_pose = self.links[i]
                .parent
                .map(|parent| poses[parent])
                .unwrap_or(self.base_pose);
            let tree_inverse_rotation = self.links[i].x_tree.rotation.inverse();
            let q = self.joint_coordinates(i);
            let joint_rotation = self.joint_rotation(self.links[i].joint.kind, q);
            let joint_translation = self.joint_translation(self.links[i].joint.kind, q);
            let local_translation =
                self.links[i].x_tree.translation + tree_inverse_rotation * joint_translation;
            let local_rotation = tree_inverse_rotation * joint_rotation;
            poses[i] = Isometry::new(
                parent_pose.translation + parent_pose.rotation * local_translation,
                (parent_pose.rotation * local_rotation).normalize(),
            );
        }
        poses
    }

    /// Returns the current link spatial velocities in link frames.
    #[must_use]
    pub fn link_velocities(&self) -> Vec<SpatialMotion> {
        let kin = self.kinematics();
        kin.velocities
    }

    /// Advances all generalized coordinates by one semi-implicit step.
    pub fn step(&mut self, dt: Real) -> ArticulationStepReport {
        if !dt.is_finite() || dt <= 0.0 {
            return ArticulationStepReport::zero(self.broken_joint.is_some());
        }
        if let Some(link) = self.broken_joint {
            return ArticulationStepReport {
                joint_accelerations: vec![0.0; self.q.len()],
                base_acceleration: SpatialMotion::default(),
                degraded: true,
                broken_joint: Some(link),
            };
        }
        let dynamics = self.compute_dynamics(dt);
        for i in 0..self.q.len() {
            self.qd[i] += dt * dynamics.qdd[i];
        }
        for link in 0..self.links.len() {
            let offset = self.dof_offsets[link];
            if self.links[link].joint.kind == ArticulationJointKind::Spherical {
                // qd is body-frame angular velocity, not the componentwise
                // derivative of the stored exponential coordinates.
                let rotation = Quat::from_scaled_axis(self.joint_coordinates(link));
                let updated = (rotation * Quat::from_scaled_axis(self.joint_velocities(link) * dt))
                    .normalize()
                    .to_scaled_axis();
                self.q[offset..offset + THREE].copy_from_slice(&updated.to_array());
            } else {
                for k in 0..self.links[link].joint.kind.dofs() {
                    self.q[offset + k] += dt * self.qd[offset + k];
                }
            }
        }
        self.project_limits();
        if self.base_mode == ArticulationBase::Floating {
            let old_rotation = self.base_pose.rotation;
            self.base_velocity = SpatialMotion::new(
                self.base_velocity.angular + dynamics.base_acceleration.angular * dt,
                self.base_velocity.linear + dynamics.base_acceleration.linear * dt,
            );
            self.base_pose.translation += old_rotation * self.base_velocity.linear * dt;
            self.base_pose.rotation = (old_rotation
                * Quat::from_scaled_axis(self.base_velocity.angular * dt))
            .normalize();
        }
        ArticulationStepReport {
            joint_accelerations: dynamics.qdd,
            base_acceleration: dynamics.base_acceleration,
            degraded: false,
            broken_joint: None,
        }
    }

    /// Computes the generalized velocity change caused by a set of link-frame
    /// contact impulses.  The result is the exact articulated inverse-mass
    /// response from the same ABA factorization used for time stepping.
    #[must_use]
    pub fn generalized_contact_impulse_response(
        &self,
        impulses: &[LinkContactImpulse],
    ) -> ContactImpulseResponse {
        if self.broken_joint.is_some() {
            return ContactImpulseResponse::zero(self.q.len());
        }
        let solver = ArticulationImpulseSolver::new(self);
        let mut forces = vec![SpatialForce::default(); self.links.len()];
        // Contact producers may traverse broad-phase pairs in different orders.
        // Sort valid inputs before floating-point accumulation, not afterwards.
        let mut ordered: Vec<_> = impulses
            .iter()
            .copied()
            .filter(|impulse| impulse.is_valid(self.links.len()))
            .collect();
        ordered.sort_by(|a, b| {
            a.link.cmp(&b.link).then_with(|| {
                a.point
                    .to_array()
                    .into_iter()
                    .chain(a.impulse.to_array())
                    .zip(b.point.to_array().into_iter().chain(b.impulse.to_array()))
                    .map(|(a, b)| a.total_cmp(&b))
                    .find(|order| !order.is_eq())
                    .unwrap_or(core::cmp::Ordering::Equal)
            })
        });
        for impulse in ordered {
            let force = &mut forces[impulse.link];
            force.torque += impulse.point.cross(impulse.impulse);
            force.force += impulse.impulse;
        }
        solver.solve(&forces, SpatialForce::default())
    }

    /// Applies link-frame contact impulses to the articulation velocity state.
    ///
    /// The response is evaluated by the same articulated inverse-mass solve as
    /// [`Self::generalized_contact_impulse_response`].  Keeping the mutation in
    /// this type gives the world scheduler one authoritative operation for
    /// contact writeback and avoids reconstructing generalized coordinates in
    /// a caller.  Impulses are applied in one batch, so a link tree is
    /// factorized once per fixed sub-step rather than once per contact point.
    #[must_use]
    pub fn apply_contact_impulses(
        &mut self,
        impulses: &[LinkContactImpulse],
    ) -> ContactImpulseResponse {
        if self.broken_joint.is_some() || impulses.is_empty() {
            return ContactImpulseResponse::zero(self.q.len());
        }
        let response = self.generalized_contact_impulse_response(impulses);
        for (velocity, delta) in self
            .qd
            .iter_mut()
            .zip(response.generalized_velocity_change.iter().copied())
        {
            *velocity += delta;
        }
        if self.base_mode == ArticulationBase::Floating {
            self.base_velocity.angular += response.base_velocity_change.angular;
            self.base_velocity.linear += response.base_velocity_change.linear;
        }
        response
    }

    /// Commits the coupled contact solve. Position corrections are split
    /// impulses: they remove overlap without injecting kinetic energy.
    pub(crate) fn commit_contact_state(
        &mut self,
        velocity: &ContactImpulseResponse,
        displacement: &ContactImpulseResponse,
    ) {
        self.qd
            .copy_from_slice(&velocity.generalized_velocity_change);
        for link in 0..self.links.len() {
            let offset = self.dof_offsets[link];
            if self.links[link].joint.kind == ArticulationJointKind::Spherical {
                let delta = Vec3::from_slice(
                    &displacement.generalized_velocity_change[offset..offset + THREE],
                );
                if delta != Vec3::ZERO {
                    let rotation = Quat::from_scaled_axis(self.joint_coordinates(link));
                    let updated = (rotation * Quat::from_scaled_axis(delta))
                        .normalize()
                        .to_scaled_axis();
                    self.q[offset..offset + THREE].copy_from_slice(&updated.to_array());
                }
            } else {
                for k in 0..self.links[link].joint.kind.dofs() {
                    self.q[offset + k] += displacement.generalized_velocity_change[offset + k];
                }
            }
        }
        if self.base_mode == ArticulationBase::Floating {
            self.base_velocity = velocity.base_velocity_change;
            let old_rotation = self.base_pose.rotation;
            self.base_pose.translation +=
                self.base_pose.rotation * displacement.base_velocity_change.linear;
            let angular = displacement.base_velocity_change.angular;
            if angular != Vec3::ZERO {
                self.base_pose.rotation =
                    (self.base_pose.rotation * Quat::from_scaled_axis(angular)).normalize();
                // Split positional rotation must not rotate the physical
                // world velocity: the base twist is stored in its local frame.
                let reexpress = self.base_pose.rotation.conjugate() * old_rotation;
                self.base_velocity.linear = reexpress * self.base_velocity.linear;
                self.base_velocity.angular = reexpress * self.base_velocity.angular;
            }
        }
        self.project_limits();
    }

    /// Converts a narrow-phase world-space impulse to the link frame. The
    /// impulse must already have the sign of the reaction on this link.
    /// Invalid links, non-finite inputs and degraded trees are rejected.
    #[must_use]
    pub fn contact_impulse_from_world(
        &self,
        link: usize,
        point: Vec3,
        impulse: Vec3,
    ) -> Option<LinkContactImpulse> {
        if link >= self.link_count() || self.broken_joint.is_some() {
            return None;
        }
        let inverse = self.forward_kinematics()[link].inverse();
        let contact = LinkContactImpulse {
            link,
            point: inverse.transform_point(point),
            impulse: inverse.transform_vector(impulse),
        };
        contact.is_valid(self.link_count()).then_some(contact)
    }

    /// World velocity at a contact point, including the link's angular motion.
    #[must_use]
    pub fn contact_point_velocity(&self, link: usize, point: Vec3) -> Option<Vec3> {
        if link >= self.link_count() || !point.is_finite() || self.broken_joint.is_some() {
            return None;
        }
        let pose = self.forward_kinematics()[link];
        let velocity = self.link_velocities()[link];
        let local_point = pose.inverse().transform_point(point);
        let result = pose.rotation * (velocity.linear + velocity.angular.cross(local_point));
        result.is_finite().then_some(result)
    }

    /// Articulated effective inverse mass at a world point along a unit
    /// direction. Includes every ancestor and the floating base, rather than
    /// treating the collision proxy as an independent rigid body.
    #[must_use]
    pub fn contact_inverse_mass(&self, link: usize, point: Vec3, direction: Vec3) -> Option<Real> {
        if !direction.is_finite() || (direction.length_squared() - 1.0).abs() > 1.0e-4 {
            return None;
        }
        if link >= self.link_count() || !point.is_finite() || self.broken_joint.is_some() {
            return None;
        }
        let solver = ArticulationImpulseSolver::new(self);
        let response = solver.solve_world_impulses(&[(Some(link), point, direction)]);
        let inverse_mass = solver
            .point_velocity(&response, Some(link), point)
            .dot(direction);
        inverse_mass.is_finite().then_some(inverse_mass.max(0.0))
    }

    /// Marks a joint as broken and returns a complete handoff snapshot for a
    /// maximum-coordinate solver.  The reduced articulation stops advancing
    /// after this call; all subtree poses and velocities are preserved in the
    /// returned degradation output.
    pub fn break_joint(&mut self, link: usize) -> Option<ArticulationDegradation> {
        if link >= self.links.len() || self.broken_joint.is_some() {
            return None;
        }
        let poses = self.forward_kinematics();
        let velocities = self.link_velocities();
        let mut detached = Vec::new();
        for candidate in link..self.links.len() {
            if self.is_descendant_or_self(candidate, link) {
                detached.push(candidate);
            }
        }
        self.broken_joint = Some(link);
        Some(ArticulationDegradation {
            broken_link: link,
            detached_links: detached,
            poses,
            velocities,
            positions: self.q.clone(),
            generalized_velocities: self.qd.clone(),
            reason: DegradationReason::JointBroken,
        })
    }

    /// Returns the link that caused degradation, if any.
    #[must_use]
    pub fn broken_joint(&self) -> Option<usize> {
        self.broken_joint
    }

    /// Rebuilds a broken tree as two live reduced-coordinate components.
    ///
    /// The detached root becomes the new floating base: its broken joint and
    /// DOFs are removed, not reset and reattached. All remaining joints keep
    /// their original coordinates, velocities, limits, motors and wrenches.
    /// `retained_links` and `detached_links` map new indices to original links;
    /// the detached base is `broken_joint()`. This operation is read-only so a
    /// world can create collision bindings before atomically replacing owners.
    #[must_use]
    pub fn rebuild_after_break(&self) -> Option<ArticulationRebuild> {
        let broken = self.broken_joint?;
        let poses = self.forward_kinematics();
        let velocities = self.link_velocities();
        let retained_links: Vec<_> = (0..self.links.len())
            .filter(|&link| !self.is_descendant_or_self(link, broken))
            .collect();
        let detached_links: Vec<_> = (0..self.links.len())
            .filter(|&link| link != broken && self.is_descendant_or_self(link, broken))
            .collect();
        let retained = self.rebuild_component(&retained_links, None, &poses, &velocities);
        let detached = self.rebuild_component(&detached_links, Some(broken), &poses, &velocities);
        Some(ArticulationRebuild {
            retained,
            detached,
            retained_links,
            detached_links,
            detached_base_link: broken,
        })
    }

    fn rebuild_component(
        &self,
        selected: &[usize],
        detached_base: Option<usize>,
        poses: &[Isometry],
        velocities: &[SpatialMotion],
    ) -> Articulation {
        let mut result = self.clone();
        result.links.clear();
        result.dof_offsets.clear();
        result.q.clear();
        result.qd.clear();
        result.link_forces.clear();
        result.broken_joint = None;
        if let Some(base) = detached_base {
            result.base_mode = ArticulationBase::Floating;
            result.base_inertia = self.links[base].inertia;
            result.base_pose = poses[base];
            result.base_velocity = velocities[base];
            result.base_force = self.link_forces.get(base).copied().unwrap_or_default();
        }
        let mut remap = vec![None; self.links.len()];
        for (new, &old) in selected.iter().enumerate() {
            remap[old] = Some(new);
        }
        for &old in selected {
            let mut link = self.links[old].clone();
            link.parent = link.parent.and_then(|parent| remap[parent]);
            let offset = self.dof_offsets[old];
            let end = offset + link.joint.kind.dofs();
            result.dof_offsets.push(result.q.len());
            result.q.extend_from_slice(&self.q[offset..end]);
            result.qd.extend_from_slice(&self.qd[offset..end]);
            result
                .link_forces
                .push(self.link_forces.get(old).copied().unwrap_or_default());
            result.links.push(link);
        }
        result
    }

    /// Complete rigid-body handoff state in deterministic base-then-link order.
    /// The COM velocity includes `angular × center_of_mass`; the spatial
    /// velocity in ABA is measured at the link origin, not at its COM.
    #[must_use]
    pub fn maximal_body_states(&self) -> Vec<ArticulationBodyState> {
        let poses = self.forward_kinematics();
        let velocities = self.link_velocities();
        let mut states = Vec::with_capacity(self.links.len() + 1);
        states.push(ArticulationBodyState::new(
            None,
            self.base_pose,
            self.base_velocity,
            self.base_inertia,
            self.base_mode == ArticulationBase::Fixed,
        ));
        for link in 0..self.links.len() {
            states.push(ArticulationBodyState::new(
                Some(link),
                poses[link],
                velocities[link],
                self.links[link].inertia,
                false,
            ));
        }
        states
    }

    fn is_descendant_or_self(&self, candidate: usize, ancestor: usize) -> bool {
        let mut current = Some(candidate);
        while let Some(index) = current {
            if index == ancestor {
                return true;
            }
            current = self.links[index].parent;
        }
        false
    }

    fn project_limits(&mut self) {
        for link in 0..self.links.len() {
            let offset = self.dof_offsets[link];
            let dofs = self.links[link].joint.kind.dofs();
            if !self.links[link].joint.limits.enabled {
                continue;
            }
            for local in 0..dofs {
                let i = offset + local;
                let lower = self.links[link].joint.limits.lower[local];
                let upper = self.links[link].joint.limits.upper[local];
                if self.q[i] < lower {
                    self.q[i] = lower;
                    if self.qd[i] < 0.0 {
                        self.qd[i] = 0.0;
                    }
                } else if self.q[i] > upper {
                    self.q[i] = upper;
                    if self.qd[i] > 0.0 {
                        self.qd[i] = 0.0;
                    }
                }
            }
        }
    }

    fn joint_coordinates(&self, link: usize) -> Vec3 {
        let offset = self.dof_offsets[link];
        let dofs = self.links[link].joint.kind.dofs();
        let mut out = Vec3::ZERO;
        for local in 0..dofs {
            out[local] = self.q[offset + local];
        }
        out
    }

    fn joint_velocities(&self, link: usize) -> Vec3 {
        let offset = self.dof_offsets[link];
        let dofs = self.links[link].joint.kind.dofs();
        let mut out = Vec3::ZERO;
        for local in 0..dofs {
            out[local] = self.qd[offset + local];
        }
        out
    }

    fn joint_rotation(&self, kind: ArticulationJointKind, q: Vec3) -> Quat {
        match kind {
            ArticulationJointKind::Fixed | ArticulationJointKind::Prismatic { .. } => {
                Quat::IDENTITY
            }
            ArticulationJointKind::Revolute { axis } => {
                Quat::from_scaled_axis(axis.normalize_or_zero() * q.x)
            }
            ArticulationJointKind::Spherical => Quat::from_scaled_axis(q),
        }
    }

    fn joint_translation(&self, kind: ArticulationJointKind, q: Vec3) -> Vec3 {
        match kind {
            ArticulationJointKind::Prismatic { axis } => axis.normalize_or_zero() * q.x,
            ArticulationJointKind::Fixed
            | ArticulationJointKind::Revolute { .. }
            | ArticulationJointKind::Spherical => Vec3::ZERO,
        }
    }

    fn joint_axes(&self, link: usize) -> JointAxes {
        let kind = self.links[link].joint.kind;
        match kind {
            ArticulationJointKind::Fixed => JointAxes::zero(),
            ArticulationJointKind::Revolute { axis } => JointAxes::one([
                axis.normalize_or_zero().x,
                axis.normalize_or_zero().y,
                axis.normalize_or_zero().z,
                0.0,
                0.0,
                0.0,
            ]),
            ArticulationJointKind::Prismatic { axis } => {
                let a = axis.normalize_or_zero();
                JointAxes::one([0.0, 0.0, 0.0, a.x, a.y, a.z])
            }
            ArticulationJointKind::Spherical => JointAxes::three([
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ]),
        }
    }

    fn xup(&self, link: usize) -> SpatialTransform {
        let q = self.joint_coordinates(link);
        let tree = self.links[link].x_tree;
        let joint_rotation = self.joint_rotation(self.links[link].joint.kind, q);
        let joint_translation = self.joint_translation(self.links[link].joint.kind, q);
        let joint = SpatialTransform::new(joint_rotation.inverse(), joint_translation);
        tree.then(joint)
    }

    fn kinematics(&self) -> Kinematics {
        let mut velocities = Vec::with_capacity(self.links.len());
        let mut axes = Vec::with_capacity(self.links.len());
        for link in 0..self.links.len() {
            let xup = self.xup(link);
            let joint_axes = self.joint_axes(link);
            let vj = joint_axes.multiply(self.joint_velocities(link));
            let parent_velocity = self.links[link]
                .parent
                .map(|parent| velocities[parent])
                .unwrap_or(self.base_velocity);
            let velocity = xup.transform_motion(parent_velocity);
            velocities.push(SpatialMotion::from_array(vec6_add(velocity.as_array(), vj)));
            axes.push(joint_axes);
        }
        Kinematics { velocities, axes }
    }

    fn compute_dynamics(&self, dt: Real) -> DynamicsResult {
        let kin = self.kinematics();
        let solver = ArticulationImpulseSolver::new_for_integration(self, dt);
        let n = self.links.len();
        let mut forces = Vec::with_capacity(n);
        let mut c = Vec::with_capacity(n);
        let mut torques = Vec::with_capacity(n);
        for i in 0..n {
            let velocity = kin.velocities[i].as_array();
            let joint_velocity = kin.axes[i].multiply(self.joint_velocities(i));
            c.push(cross_motion(velocity, joint_velocity));
            let inertia = self.links[i].inertia;
            let gravity =
                inertia.gravity_force(solver.body_pose(Some(i)).rotation.inverse() * self.gravity);
            let external = self.link_forces.get(i).copied().unwrap_or_default();
            let bias = cross_force(velocity, mat6_vec(&inertia.matrix(), velocity));
            forces.push(SpatialForce::from_array(vec6_sub(
                vec6_add(external.as_array(), gravity.as_array()),
                bias,
            )));
            let mut torque = self.joint_torque(i);
            let damping = self.links[i].joint.damping;
            if damping != 0.0 {
                torque -= self.joint_velocities(i) * damping;
            }
            torques.push(torque.to_array());
        }
        let velocity = if self.base_mode == ArticulationBase::Floating {
            self.base_velocity.as_array()
        } else {
            zero6()
        };
        let base_force = SpatialForce::from_array(vec6_sub(
            vec6_add(
                self.base_force.as_array(),
                self.base_gravity_force().as_array(),
            ),
            cross_force(velocity, mat6_vec(&self.base_inertia.matrix(), velocity)),
        ));
        let response = solver.solve_biased(&forces, base_force, &c, &torques);
        DynamicsResult {
            qdd: response.generalized_velocity_change,
            base_acceleration: response.base_velocity_change,
        }
    }

    fn base_gravity_force(&self) -> SpatialForce {
        if self.base_mode == ArticulationBase::Fixed {
            return SpatialForce::default();
        }
        self.base_inertia
            .gravity_force(self.base_pose.rotation.inverse() * self.gravity)
    }

    fn joint_torque(&self, link: usize) -> Vec3 {
        let q = self.joint_coordinates(link);
        let qd = self.joint_velocities(link);
        let joint = self.links[link].joint;
        joint
            .motor
            .map(|motor| motor.torque(q, qd, joint.kind.dofs()))
            .unwrap_or(Vec3::ZERO)
    }
}

#[derive(Clone, Copy)]
struct JointAxes {
    cols: [[Real; SIX]; THREE],
    count: usize,
}

impl JointAxes {
    fn zero() -> Self {
        Self {
            cols: [[0.0; SIX]; THREE],
            count: 0,
        }
    }

    fn one(column: [Real; SIX]) -> Self {
        let mut out = Self::zero();
        out.cols[0] = column;
        out.count = 1;
        out
    }

    fn three(columns: [[Real; SIX]; THREE]) -> Self {
        Self {
            cols: columns,
            count: 3,
        }
    }

    fn multiply(self, values: Vec3) -> Vec6 {
        self.multiply_array([values.x, values.y, values.z])
    }

    fn multiply_array(self, values: [Real; THREE]) -> Vec6 {
        let mut out = zero6();
        for col in 0..self.count {
            for row in 0..SIX {
                out[row] += self.cols[col][row] * values[col];
            }
        }
        out
    }

    fn transpose_vec(self, value: Vec6) -> [Real; THREE] {
        let mut out = [0.0; THREE];
        for col in 0..self.count {
            for row in 0..SIX {
                out[col] += self.cols[col][row] * value[row];
            }
        }
        out
    }
}

fn cross_motion(v: Vec6, w: Vec6) -> Vec6 {
    let va = Vec3::new(v[0], v[1], v[2]);
    let vl = Vec3::new(v[3], v[4], v[5]);
    let wa = Vec3::new(w[0], w[1], w[2]);
    let wl = Vec3::new(w[3], w[4], w[5]);
    SpatialMotion::new(va.cross(wa), va.cross(wl) + vl.cross(wa)).as_array()
}

fn cross_force(v: Vec6, f: Vec6) -> Vec6 {
    let va = Vec3::new(v[0], v[1], v[2]);
    let vl = Vec3::new(v[3], v[4], v[5]);
    let ft = Vec3::new(f[0], f[1], f[2]);
    let fl = Vec3::new(f[3], f[4], f[5]);
    SpatialForce::new(va.cross(ft) + vl.cross(fl), va.cross(fl)).as_array()
}

fn invert_small(matrix: Mat3N, count: usize) -> Mat3N {
    symmetric_inverse(matrix, count)
}

fn multiply_small_vec(matrix: Mat3N, vector: [Real; THREE], count: usize) -> [Real; THREE] {
    let mut out = [0.0; THREE];
    for i in 0..count {
        for j in 0..count {
            out[i] += matrix[i][j] * vector[j];
        }
    }
    out
}

fn outer_from_cols(cols: &[[Real; THREE]; SIX], inverse: Mat3N, count: usize) -> Mat6 {
    let mut out = zero_mat6();
    for i in 0..SIX {
        for j in 0..SIX {
            for a in 0..count {
                for b in 0..count {
                    out[i][j] += cols[i][a] * inverse[a][b] * cols[j][b];
                }
            }
        }
    }
    out
}

#[derive(Clone)]
struct Kinematics {
    velocities: Vec<SpatialMotion>,
    axes: Vec<JointAxes>,
}

struct DynamicsResult {
    qdd: Vec<Real>,
    base_acceleration: SpatialMotion,
}

/// Result of one ABA step.
#[derive(Clone, Debug, PartialEq)]
pub struct ArticulationStepReport {
    /// Generalized accelerations in joint coordinate order.
    pub joint_accelerations: Vec<Real>,
    /// Floating-base acceleration, or zero for a fixed base.
    pub base_acceleration: SpatialMotion,
    /// Whether this articulation has been handed off after a break.
    pub degraded: bool,
    /// Broken link, if the report is degraded.
    pub broken_joint: Option<usize>,
}

impl ArticulationStepReport {
    fn zero(degraded: bool) -> Self {
        Self {
            joint_accelerations: Vec::new(),
            base_acceleration: SpatialMotion::default(),
            degraded,
            broken_joint: None,
        }
    }
}

/// An impulse applied at a link point, with point and impulse in link frame.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct LinkContactImpulse {
    /// Target link index.
    pub link: usize,
    /// Contact point relative to the link frame origin.
    pub point: Vec3,
    /// Linear impulse in link coordinates.
    pub impulse: Vec3,
}

impl LinkContactImpulse {
    fn is_valid(self, link_count: usize) -> bool {
        self.link < link_count
            && self.point.is_finite()
            && self.impulse.is_finite()
            && self.point.cross(self.impulse).is_finite()
    }
}

/// Generalized inverse-mass response to contact impulses.
#[derive(Clone, Debug, PartialEq)]
pub struct ContactImpulseResponse {
    /// Joint velocity changes in generalized-coordinate order.
    pub generalized_velocity_change: Vec<Real>,
    /// Floating-base spatial velocity change, or zero for a fixed base.
    pub base_velocity_change: SpatialMotion,
}

impl ContactImpulseResponse {
    fn zero(dofs: usize) -> Self {
        Self {
            generalized_velocity_change: vec![0.0; dofs],
            base_velocity_change: SpatialMotion::default(),
        }
    }
}

/// Two live components produced by removing one broken joint.
#[derive(Clone, Debug)]
pub struct ArticulationRebuild {
    /// Original base and all links outside the broken subtree.
    pub retained: Articulation,
    /// Detached root as a floating base, with its surviving descendant joints.
    pub detached: Articulation,
    /// Retained new-link to original-link mapping.
    pub retained_links: Vec<usize>,
    /// Detached new-link to original-link mapping (excludes its new base).
    pub detached_links: Vec<usize>,
    /// Original link now represented by the detached floating base.
    pub detached_base_link: usize,
}

/// Rigid-body state ready for a maximum-coordinate handoff/collision proxy.
#[derive(Clone, Debug, PartialEq)]
pub struct ArticulationBodyState {
    /// Original link, or `None` for the articulation base.
    pub link: Option<usize>,
    /// World pose of the link frame (not its COM frame).
    pub pose: Isometry,
    /// Original link-frame mass, COM offset and COM inertia tensor.
    pub inertia: SpatialInertia,
    /// World-space COM position.
    pub center_of_mass: Vec3,
    /// World-space COM linear velocity.
    pub linear_velocity: Vec3,
    /// World-space angular velocity.
    pub angular_velocity: Vec3,
    /// Whether this body is the immovable base.
    pub fixed: bool,
}

impl ArticulationBodyState {
    fn new(
        link: Option<usize>,
        pose: Isometry,
        velocity: SpatialMotion,
        inertia: SpatialInertia,
        fixed: bool,
    ) -> Self {
        Self {
            link,
            pose,
            inertia,
            center_of_mass: pose.transform_point(inertia.center_of_mass),
            linear_velocity: pose.rotation
                * (velocity.linear + velocity.angular.cross(inertia.center_of_mass)),
            angular_velocity: pose.rotation * velocity.angular,
            fixed,
        }
    }
}

/// Reason a reduced articulation must be handed off to a maximum-coordinate
/// representation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DegradationReason {
    /// A joint broke and the subtree is no longer a fixed topology tree.
    JointBroken,
}

/// State snapshot emitted when an articulation is broken.
#[derive(Clone, Debug, PartialEq)]
pub struct ArticulationDegradation {
    /// Link whose parent joint broke.
    pub broken_link: usize,
    /// Link subtree that must be detached in the maximum-coordinate solver.
    pub detached_links: Vec<usize>,
    /// World-space poses of every link at handoff time.
    pub poses: Vec<Isometry>,
    /// Link-frame spatial velocities at handoff time.
    pub velocities: Vec<SpatialMotion>,
    /// Joint generalized positions at handoff time.
    pub positions: Vec<Real>,
    /// Joint generalized velocities at handoff time.
    pub generalized_velocities: Vec<Real>,
    /// Why the handoff was requested.
    pub reason: DegradationReason,
}

/// A generational handle into [`ArticulationStorage`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ArticulationHandle {
    index: u32,
    generation: u32,
}

impl ArticulationHandle {
    /// A sentinel handle that never refers to a live articulation.
    pub const INVALID: Self = Self {
        index: u32::MAX,
        generation: u32::MAX,
    };

    const fn new(index: u32, generation: u32) -> Self {
        Self { index, generation }
    }

    /// Returns the storage slot index.
    #[must_use]
    pub const fn index(self) -> u32 {
        self.index
    }

    /// Returns the generation captured by this handle.
    #[must_use]
    pub const fn generation(self) -> u32 {
        self.generation
    }
}

impl Default for ArticulationHandle {
    fn default() -> Self {
        Self::INVALID
    }
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
struct ArticulationSlot {
    articulation: Articulation,
    contact_impulses: Vec<LinkContactImpulse>,
    break_request: Option<usize>,
}

/// Deterministic generational storage and fixed-step input queues for
/// world-owned articulations.
///
/// The storage owns only M14 state.  It does not mirror rigid bodies or
/// assign gameplay meaning to links.  Contact detection, an ECS bridge, or a
/// gameplay system queues [`LinkContactImpulse`] values and optional break
/// requests; [`crate::world::PhysicsWorld`] consumes those queues at the fixed
/// sub-step boundary.
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ArticulationStorage {
    slots: Vec<Option<ArticulationSlot>>,
    generations: Vec<u32>,
    free_list: Vec<u32>,
    live: usize,
}

impl ArticulationStorage {
    /// Creates empty articulation storage.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts an articulation and returns its stable handle.
    pub fn insert(&mut self, articulation: Articulation) -> ArticulationHandle {
        self.live += 1;
        let slot = ArticulationSlot {
            articulation,
            contact_impulses: Vec::new(),
            break_request: None,
        };
        if let Some(index) = self.free_list.pop() {
            let slot_index = index as usize;
            self.slots[slot_index] = Some(slot);
            ArticulationHandle::new(index, self.generations[slot_index])
        } else {
            let index = self.slots.len() as u32;
            self.slots.push(Some(slot));
            self.generations.push(0);
            ArticulationHandle::new(index, 0)
        }
    }

    /// Removes an articulation and invalidates its handle.
    pub fn remove(&mut self, handle: ArticulationHandle) -> bool {
        if !self.contains(handle) {
            return false;
        }
        let slot = handle.index as usize;
        self.slots[slot] = None;
        self.generations[slot] = self.generations[slot].wrapping_add(1);
        self.free_list.push(handle.index);
        self.live -= 1;
        true
    }

    /// Returns whether `handle` refers to a live articulation.
    #[must_use]
    pub fn contains(&self, handle: ArticulationHandle) -> bool {
        let slot = handle.index as usize;
        slot < self.slots.len()
            && self.slots[slot].is_some()
            && self.generations[slot] == handle.generation
    }

    /// Returns the number of live articulations.
    #[must_use]
    pub fn len(&self) -> usize {
        self.live
    }

    /// Returns whether the storage has no live articulations.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.live == 0
    }

    /// Returns a shared articulation reference.
    #[must_use]
    pub fn get(&self, handle: ArticulationHandle) -> Option<&Articulation> {
        self.slot(handle).map(|slot| &slot.articulation)
    }

    /// Returns a mutable articulation reference.
    pub fn get_mut(&mut self, handle: ArticulationHandle) -> Option<&mut Articulation> {
        self.slot_mut(handle).map(|slot| &mut slot.articulation)
    }

    /// Returns live handles in deterministic ascending slot order.
    pub fn handles(&self) -> impl Iterator<Item = ArticulationHandle> + '_ {
        self.slots.iter().enumerate().filter_map(|(slot, value)| {
            value
                .as_ref()
                .map(|_| ArticulationHandle::new(slot as u32, self.generations[slot]))
        })
    }

    /// Returns the live handle occupying `slot`, if any.
    #[must_use]
    pub fn handle_at_slot(&self, slot: usize) -> Option<ArticulationHandle> {
        self.slots.get(slot)?.as_ref()?;
        Some(ArticulationHandle::new(
            slot as u32,
            *self.generations.get(slot)?,
        ))
    }

    /// Queues one contact impulse for the next fixed sub-step.
    pub fn queue_contact_impulse(
        &mut self,
        handle: ArticulationHandle,
        impulse: LinkContactImpulse,
    ) -> bool {
        let Some(slot) = self.slot_mut(handle) else {
            return false;
        };
        if !impulse.is_valid(slot.articulation.link_count()) {
            return false;
        }
        if slot.articulation.broken_joint().is_some() {
            return false;
        }
        slot.contact_impulses.push(impulse);
        true
    }

    /// Queues a world-space narrow-phase reaction without exposing link-frame
    /// conversions to collision producers. Stale handles are rejected.
    pub fn queue_world_contact_impulse(
        &mut self,
        handle: ArticulationHandle,
        link: usize,
        point: Vec3,
        impulse: Vec3,
    ) -> bool {
        let Some(contact) = self
            .get(handle)
            .and_then(|tree| tree.contact_impulse_from_world(link, point, impulse))
        else {
            return false;
        };
        self.queue_contact_impulse(handle, contact)
    }

    /// Queues a deterministic joint-break handoff at the next fixed boundary.
    pub fn request_break(&mut self, handle: ArticulationHandle, link: usize) -> bool {
        let Some(slot) = self.slot_mut(handle) else {
            return false;
        };
        if link >= slot.articulation.link_count()
            || slot.articulation.broken_joint().is_some()
            || slot.break_request.is_some()
        {
            return false;
        }
        slot.break_request = Some(link);
        true
    }

    pub(crate) fn take_contact_impulses(
        &mut self,
        handle: ArticulationHandle,
    ) -> Vec<LinkContactImpulse> {
        self.slot_mut(handle)
            .map(|slot| core::mem::take(&mut slot.contact_impulses))
            .unwrap_or_default()
    }

    pub(crate) fn take_break_request(&mut self, handle: ArticulationHandle) -> Option<usize> {
        self.slot_mut(handle)
            .and_then(|slot| slot.break_request.take())
    }

    fn slot(&self, handle: ArticulationHandle) -> Option<&ArticulationSlot> {
        let index = handle.index as usize;
        (index < self.slots.len() && self.generations[index] == handle.generation)
            .then(|| self.slots[index].as_ref())
            .flatten()
    }

    fn slot_mut(&mut self, handle: ArticulationHandle) -> Option<&mut ArticulationSlot> {
        let index = handle.index as usize;
        if index >= self.slots.len() || self.generations[index] != handle.generation {
            return None;
        }
        self.slots[index].as_mut()
    }
}

/// A world-owned articulation degradation event.
#[derive(Clone, Debug, PartialEq)]
pub struct ArticulationDegradationEvent {
    /// The world storage handle of the articulation that degraded.
    pub articulation: ArticulationHandle,
    /// The complete maximum-coordinate handoff snapshot.
    pub degradation: ArticulationDegradation,
}

/// Result of one world articulation fixed sub-step.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ArticulationWorldStepReport {
    /// Number of articulations visited by the scheduler.
    pub stepped: usize,
    /// Number of queued contact impulses consumed at the boundary.
    pub contact_impulses: usize,
    /// Number of new degradation events emitted by the step.
    pub degradations: usize,
    /// Per-articulation ABA reports in deterministic storage order.
    pub reports: Vec<(ArticulationHandle, ArticulationStepReport)>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point_inertia(mass: Real) -> SpatialInertia {
        SpatialInertia::diagonal(mass, Vec3::ZERO, Vec3::splat(0.2 * mass))
    }

    fn one_revolute() -> Articulation {
        let mut builder =
            ArticulationBuilder::new(ArticulationBase::Fixed, SpatialInertia::default());
        builder
            .add_link(
                None,
                SpatialTransform::new(Quat::IDENTITY, Vec3::new(0.0, 0.0, 1.0)),
                point_inertia(1.0),
                ArticulationJoint::revolute(Vec3::Z),
            )
            .expect("valid link");
        builder.build().expect("valid articulation")
    }

    #[test]
    fn spatial_transform_motion_force_duality() {
        let x = SpatialTransform::new(Quat::from_rotation_y(0.4), Vec3::new(0.2, -0.3, 0.5));
        let motion = SpatialMotion::new(Vec3::new(0.3, -0.2, 0.7), Vec3::new(1.0, 0.2, -0.4));
        let force = SpatialForce::new(Vec3::new(-0.5, 0.8, 0.1), Vec3::new(2.0, -1.0, 0.4));
        let lhs = dot6(x.transform_motion(motion).as_array(), force.as_array());
        let rhs = dot6(
            motion.as_array(),
            x.transform_force_transpose(force).as_array(),
        );
        assert!((lhs - rhs).abs() < 2e-5, "duality: {lhs} vs {rhs}");
    }

    fn dot6(a: Vec6, b: Vec6) -> Real {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    #[test]
    fn fk_revolute_has_zero_drift_and_expected_rotation() {
        let mut articulation = one_revolute();
        articulation.set_position(0, core::f32::consts::FRAC_PI_2);
        let pose = articulation.forward_kinematics()[0];
        assert!((pose.translation - Vec3::new(0.0, 0.0, 1.0)).length() < 1e-5);
        let axis = pose.rotation * Vec3::X;
        assert!((axis - Vec3::Y).length() < 1e-5, "axis={axis:?}");
    }

    #[test]
    fn revolute_gravity_matches_single_link_direction() {
        let mut builder =
            ArticulationBuilder::new(ArticulationBase::Fixed, SpatialInertia::default())
                .with_gravity(Vec3::new(0.0, -9.81, 0.0));
        builder
            .add_link(
                None,
                SpatialTransform::new(Quat::IDENTITY, Vec3::new(0.0, 0.0, 1.0)),
                SpatialInertia::diagonal(1.0, Vec3::new(0.0, 0.0, 0.5), Vec3::splat(0.1)),
                ArticulationJoint::revolute(Vec3::X),
            )
            .expect("valid link");
        let mut articulation = builder.build().expect("valid articulation");
        let report = articulation.step(1.0 / 60.0);
        assert!(
            report.joint_accelerations[0].abs() > 1e-4,
            "report={report:?}, positions={:?}, velocities={:?}",
            articulation.positions(),
            articulation.velocities()
        );
        assert!(articulation.velocities()[0].abs() > 0.0);
    }

    #[test]
    fn prismatic_mass_ratio_is_finite_and_deterministic() {
        fn build() -> Articulation {
            let mut builder =
                ArticulationBuilder::new(ArticulationBase::Fixed, SpatialInertia::default());
            builder
                .add_link(
                    None,
                    SpatialTransform::IDENTITY,
                    point_inertia(1.0e-3),
                    ArticulationJoint::prismatic(Vec3::X),
                )
                .expect("valid root");
            builder
                .add_link(
                    Some(0),
                    SpatialTransform::new(Quat::IDENTITY, Vec3::X),
                    point_inertia(1.0e3),
                    ArticulationJoint::revolute(Vec3::Z),
                )
                .expect("valid child");
            builder.build().expect("valid articulation")
        }
        let mut a = build();
        let mut b = build();
        a.set_link_force(1, SpatialForce::new(Vec3::ZERO, Vec3::new(0.0, 10.0, 0.0)));
        b.set_link_force(1, SpatialForce::new(Vec3::ZERO, Vec3::new(0.0, 10.0, 0.0)));
        for _ in 0..32 {
            let ra = a.step(1.0 / 120.0);
            let rb = b.step(1.0 / 120.0);
            assert_eq!(ra, rb);
        }
        assert!(a.positions().iter().all(|v| v.is_finite()));
        assert!(a.velocities().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn spherical_joint_has_three_dofs_and_finite_fk() {
        let mut builder = ArticulationBuilder::new(ArticulationBase::Floating, point_inertia(4.0));
        builder
            .add_link(
                None,
                SpatialTransform::new(Quat::IDENTITY, Vec3::Y),
                point_inertia(1.0),
                ArticulationJoint::spherical(),
            )
            .expect("valid link");
        let mut a = builder.build().expect("valid articulation");
        assert_eq!(a.dof_count(), 3);
        a.set_position(0, 0.2);
        a.set_position(1, -0.1);
        a.set_position(2, 0.3);
        let pose = a.forward_kinematics()[0];
        assert!(pose.rotation.is_finite());
        let report = a.step(1.0 / 60.0);
        assert!(report.base_acceleration.angular.is_finite());
    }

    #[test]
    fn contact_impulse_response_respects_fixed_base() {
        let a = one_revolute();
        let response = a.generalized_contact_impulse_response(&[LinkContactImpulse {
            link: 0,
            point: Vec3::X,
            impulse: Vec3::Y,
        }]);
        assert_eq!(response.base_velocity_change, SpatialMotion::default());
        assert!(response.generalized_velocity_change[0].is_finite());
        assert!(response.generalized_velocity_change[0].abs() > 1e-5);
    }

    #[test]
    fn contact_impulse_response_is_independent_of_current_velocity() {
        let mut moving = one_revolute();
        moving.set_velocity(0, 3.0);
        let stationary = one_revolute();
        let impulse = [LinkContactImpulse {
            link: 0,
            point: Vec3::X,
            impulse: Vec3::Y,
        }];
        let moving_response = moving.generalized_contact_impulse_response(&impulse);
        let stationary_response = stationary.generalized_contact_impulse_response(&impulse);
        assert!(
            (moving_response.generalized_velocity_change[0]
                - stationary_response.generalized_velocity_change[0])
                .abs()
                < 1.0e-6
        );
    }

    #[test]
    fn limits_and_motor_are_applied_in_generalized_space() {
        let mut builder =
            ArticulationBuilder::new(ArticulationBase::Fixed, SpatialInertia::default());
        let joint = ArticulationJoint::revolute(Vec3::Z)
            .with_limits(JointLimits::scalar(-0.2, 0.2))
            .with_motor(JointMotor::new(
                Vec3::splat(0.2),
                Vec3::ZERO,
                100.0,
                4.0,
                Vec3::splat(10.0),
            ));
        builder
            .add_link(None, SpatialTransform::IDENTITY, point_inertia(1.0), joint)
            .expect("valid link");
        let mut a = builder.build().expect("valid articulation");
        for _ in 0..240 {
            a.step(1.0 / 120.0);
        }
        assert!(a.positions()[0] <= 0.20001);
        assert!(a.positions()[0] >= -0.20001);
        assert!((a.positions()[0] - 0.2).abs() < 0.03);
    }

    #[test]
    fn break_returns_subtree_snapshot_and_stops_reduced_step() {
        let mut builder =
            ArticulationBuilder::new(ArticulationBase::Fixed, SpatialInertia::default());
        builder
            .add_link(
                None,
                SpatialTransform::IDENTITY,
                point_inertia(1.0),
                ArticulationJoint::fixed(),
            )
            .expect("root");
        builder
            .add_link(
                Some(0),
                SpatialTransform::IDENTITY,
                point_inertia(1.0),
                ArticulationJoint::revolute(Vec3::Z),
            )
            .expect("child");
        let mut a = builder.build().expect("valid articulation");
        let output = a.break_joint(1).expect("break output");
        assert_eq!(output.detached_links, vec![1]);
        assert_eq!(output.reason, DegradationReason::JointBroken);
        let report = a.step(0.1);
        assert!(report.degraded);
        assert_eq!(report.broken_joint, Some(1));
    }

    #[test]
    fn articulation_storage_preserves_generation_and_batches_inputs() {
        let mut storage = ArticulationStorage::new();
        let handle = storage.insert(one_revolute());
        assert_eq!(storage.handles().collect::<Vec<_>>(), vec![handle]);
        assert!(storage.queue_contact_impulse(
            handle,
            LinkContactImpulse {
                link: 0,
                point: Vec3::X,
                impulse: Vec3::Y,
            },
        ));
        assert!(!storage.queue_contact_impulse(
            handle,
            LinkContactImpulse {
                link: 1,
                point: Vec3::ZERO,
                impulse: Vec3::Y,
            },
        ));
        assert!(storage.remove(handle));
        assert!(!storage.contains(handle));
        let replacement = storage.insert(one_revolute());
        assert_eq!(replacement.index(), handle.index());
        assert_ne!(replacement.generation(), handle.generation());
    }
}
