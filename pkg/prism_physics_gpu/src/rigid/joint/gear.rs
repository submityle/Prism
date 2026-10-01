//! The gear (angular ratio coupling) rigid-body joint.
//!
//! A [`GearJoint`] couples the spin of two bodies about their respective axes
//! at a fixed ratio, the position-based analogue of `PhysX`'s `PxGearJoint` and
//! the gear coupling of Unreal's articulation. It is the single scalar
//! constraint
//!
//! ```text
//! C = ratio * (u_a . dphi_a) + (u_b . dphi_b)
//! ```
//!
//! where `u_a = rotate(orientation_a, axis_a)` and
//! `u_b = rotate(orientation_b, axis_b)` are the two world-space gear axes and
//! `dphi_a` / `dphi_b` are each body's angular displacement since the substep
//! snapshot. Driving `C -> 0` each substep locks the relative spin rates to
//!
//! ```text
//! ratio * (u_a . omega_a) + (u_b . omega_b) = 0,
//! ```
//!
//! so a positive `ratio` makes the two gears counter-rotate (meshing external
//! gears) and a negative `ratio` makes them co-rotate (an internal/annulus
//! gear or a belt), with the rate magnitude set by `|ratio|` — a small gear
//! driving a large one, or vice versa.
//!
//! # Scope and provenance of the coupling
//!
//! Following `PxGearJoint`, the gear joint constrains *only* the ratio
//! coupling; it carries no positional weld and no axis alignment of its own.
//! Each gear is expected to be separately hinged to its mount (for example by a
//! [`RevoluteJoint`](super::RevoluteJoint)) so that its spin axis is already
//! held in place; the gear joint then ties the two free spins together. This
//! matches the real mechanism, where the gear teeth transmit a tangential force
//! whose equal-and-opposite reaction is carried by the two shaft bearings, not
//! by the mating gear — which is why the coupling's angular impulses about the
//! two axes differ by the factor `ratio` and are *not* equal and opposite (the
//! missing reaction torque flows into the mounting frames).
//!
//! The coupling is expressed per substep on the *incremental* relative spin
//! rather than on an absolute accumulated phase: the shared stepper keeps no
//! per-joint persistent state, so the joint locks the relative *rate*
//! `ratio * omega_a + omega_b` to zero every substep. It does not servo an
//! absolute phase offset (a "gear phasing" lock), which would need a persistent
//! wound-angle the bilateral stepper does not store; that is left to a future
//! stateful joint rather than faked here.
//!
//! # Compliance
//!
//! [`compliance`](GearJoint::compliance) is the inverse stiffness of the ratio
//! coupling (reciprocal newton-metre-seconds per radian). Zero compliance is a
//! *rigid* gear that forces the rate coupling exactly each substep (the ideal
//! meshing-teeth limit); a positive value is a *soft* gear whose teeth flex
//! under load, so the coupling is satisfied with a finite torque proportional
//! to the rate error — the model of a compliant belt or a gear train with
//! backlash take-up springs.
//!
//! Provenance: the angular ratio coupling expressed as a per-substep compliant
//! `XPBD` equality on the relative angular displacement about the two gear axes
//! (Macklin et al., "XPBD: Position-Based Simulation of Compliant Constrained
//! Dynamics"), over the world-space inverse inertia and quaternion kinematics of
//! Baraff & Witkin. No Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

use super::coloring::JointBodies;

/// A gear joint coupling the spin of two bodies about their respective axes at
/// a fixed ratio: a single scalar angular constraint with no weld.
///
/// Either body may be static (zero inverse mass and inertia); its corrections
/// scale by zero and it acts as an immovable gear housing.
#[derive(Clone, Copy, Debug)]
pub struct GearJoint {
    /// Index of the first body in the [`RigidBodyState`](super::super::RigidBodyState).
    pub body_a: u32,
    /// Index of the second body in the [`RigidBodyState`](super::super::RigidBodyState).
    pub body_b: u32,
    /// Gear axis on body `a`, in body `a`'s local frame. Need not be
    /// unit-length; only its direction matters.
    pub axis_a: Vec3,
    /// Gear axis on body `b`, in body `b`'s local frame.
    pub axis_b: Vec3,
    /// Gear ratio coupling the two spins. The constraint locks
    /// `ratio * (u_a . omega_a) + (u_b . omega_b) = 0`: a positive ratio makes
    /// the gears counter-rotate (external mesh), a negative ratio makes them
    /// co-rotate (internal/belt), and `|ratio|` sets the relative rate.
    pub ratio: f32,
    /// Inverse stiffness (reciprocal newton-metre-seconds per radian) of the
    /// ratio coupling. Zero is the rigid limit; a positive value is a soft gear
    /// of torque gain `1 / compliance`.
    pub compliance: f32,
}

impl GearJoint {
    /// Creates a gear joint from its full parameter set.
    ///
    /// `axis_a` / `axis_b` are the spin axes in each body's local frame (only
    /// their direction matters). `ratio` is the gear ratio, `compliance` the
    /// inverse coupling stiffness described on the fields.
    #[must_use]
    pub fn new(
        body_a: u32,
        body_b: u32,
        axis_a: Vec3,
        axis_b: Vec3,
        ratio: f32,
        compliance: f32,
    ) -> GearJoint {
        GearJoint {
            body_a,
            body_b,
            axis_a,
            axis_b,
            ratio,
            compliance,
        }
    }

    /// Creates a rigid gear: a coupling with zero compliance, so each substep
    /// forces the rate coupling `ratio * omega_a + omega_b = 0` exactly (the
    /// ideal rigid meshing-teeth limit, unbounded coupling torque).
    #[must_use]
    pub fn rigid(body_a: u32, body_b: u32, axis_a: Vec3, axis_b: Vec3, ratio: f32) -> GearJoint {
        GearJoint::new(body_a, body_b, axis_a, axis_b, ratio, 0.0)
    }

    /// Creates a soft gear of the given coupling stiffness
    /// (`compliance = 1 / stiffness`), so the ratio coupling is satisfied with a
    /// finite torque — a flexing belt or sprung gear train — rather than
    /// instantly. A non-positive stiffness collapses to the rigid gear.
    #[must_use]
    pub fn soft(
        body_a: u32,
        body_b: u32,
        axis_a: Vec3,
        axis_b: Vec3,
        ratio: f32,
        stiffness: f32,
    ) -> GearJoint {
        let compliance = if stiffness > 0.0 {
            1.0 / stiffness
        } else {
            0.0
        };
        GearJoint::new(body_a, body_b, axis_a, axis_b, ratio, compliance)
    }

    /// The two body indices this joint couples, in `(a, b)` order.
    #[must_use]
    pub fn bodies(&self) -> (u32, u32) {
        (self.body_a, self.body_b)
    }

    /// Packs the joint into its `GPU` storage-buffer representation.
    #[must_use]
    pub(crate) fn to_gpu(self) -> GpuGearJoint {
        GpuGearJoint {
            axis_a: [self.axis_a.x, self.axis_a.y, self.axis_a.z, 0.0],
            axis_b: [self.axis_b.x, self.axis_b.y, self.axis_b.z, 0.0],
            body_a: self.body_a,
            body_b: self.body_b,
            ratio: self.ratio,
            compliance: self.compliance,
        }
    }
}

impl JointBodies for GearJoint {
    fn bodies(&self) -> (u32, u32) {
        (self.body_a, self.body_b)
    }
}

/// Device-packed [`GearJoint`]; layout matches `Joint` in
/// `shaders/rigid_joint_gear.wgsl` (`48` bytes). The two gear axes are padded to
/// `vec4` so each starts on the `16`-byte boundary the storage layout requires;
/// the two `w` lanes are unused. The two body indices, the ratio, and the
/// coupling compliance fill the final `16`-byte block.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct GpuGearJoint {
    /// Body-local gear axis on body `a` in `xyz`; `w` unused.
    axis_a: [f32; 4],
    /// Body-local gear axis on body `b` in `xyz`; `w` unused.
    axis_b: [f32; 4],
    /// Index of body `a`.
    body_a: u32,
    /// Index of body `b`.
    body_b: u32,
    /// Gear ratio coupling the two spins.
    ratio: f32,
    /// Inverse stiffness of the ratio coupling.
    compliance: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_packing_round_trips_fields() {
        let joint = GearJoint::new(
            2,
            9,
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            2.5,
            0.002,
        );
        let gpu = joint.to_gpu();
        assert_eq!(gpu.body_a, 2);
        assert_eq!(gpu.body_b, 9);
        assert_eq!(gpu.axis_a, [0.0, 1.0, 0.0, 0.0]);
        assert_eq!(gpu.axis_b, [0.0, 0.0, 1.0, 0.0]);
        assert_eq!(gpu.ratio, 2.5);
        assert_eq!(gpu.compliance, 0.002);
    }

    #[test]
    fn gpu_struct_is_48_bytes() {
        assert_eq!(size_of::<GpuGearJoint>(), 48);
    }

    #[test]
    fn rigid_builds_a_zero_compliance_gear() {
        let joint = GearJoint::rigid(0, 1, Vec3::Z, Vec3::Z, 3.0);
        assert_eq!(joint.compliance, 0.0);
        assert_eq!(joint.ratio, 3.0);
    }

    #[test]
    fn soft_inverts_stiffness_into_compliance() {
        let joint = GearJoint::soft(0, 1, Vec3::Z, Vec3::Z, 2.0, 200.0);
        assert!((joint.compliance - 1.0 / 200.0).abs() < 1e-9);
        assert_eq!(joint.ratio, 2.0);
    }

    #[test]
    fn soft_with_zero_stiffness_is_rigid() {
        let joint = GearJoint::soft(0, 1, Vec3::Z, Vec3::Z, 1.0, 0.0);
        assert_eq!(joint.compliance, 0.0);
    }

    #[test]
    fn bodies_returns_pair_in_order() {
        let joint = GearJoint::rigid(6, 1, Vec3::Y, Vec3::Y, -1.0);
        assert_eq!(joint.bodies(), (6, 1));
        assert_eq!(JointBodies::bodies(&joint), (6, 1));
    }
}
