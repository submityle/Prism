//! The rack-and-pinion rigid-body joint.
//!
//! A [`RackPinionJoint`] couples the spin of a *pinion* body about its axis to
//! the slide of a *rack* body along its axis at a fixed ratio, the
//! position-based analogue of `PhysX`'s `PxRackAndPinionJoint`. It is the
//! single scalar constraint
//!
//! ```text
//! C = (u_a . dphi_a) - ratio * (u_b . dx_b)
//! ```
//!
//! where `u_a = rotate(orientation_a, axis_a)` is the pinion's world spin axis,
//! `u_b = rotate(orientation_b, axis_b)` is the rack's world slide axis,
//! `dphi_a` is the pinion's angular displacement and `dx_b` the rack's linear
//! displacement since the substep snapshot. Driving `C -> 0` each substep locks
//! the pinion's spin rate to the rack's slide rate:
//!
//! ```text
//! (u_a . omega_a) = ratio * (u_b . v_b),
//! ```
//!
//! so `ratio` (radians of pinion rotation per unit of rack travel) sets the
//! transmission: a toothed pinion of pitch radius `r` meshing a straight rack
//! has `ratio = 1 / r`. A negative `ratio` reverses the sense of the coupling.
//!
//! # Scope and provenance of the coupling
//!
//! Following `PxRackAndPinionJoint`, the joint constrains *only* the
//! rotation-to-translation coupling; it carries no positional weld and no axis
//! alignment of its own. The pinion is expected to be separately hinged to its
//! mount (for example by a [`RevoluteJoint`](super::RevoluteJoint)) and the
//! rack separately held on its slide (for example by a
//! [`PrismaticJoint`](super::PrismaticJoint)); the rack-and-pinion joint then
//! ties the free rotation and the free translation together. This matches the
//! real mechanism, where the meshing teeth transmit a tangential force whose
//! equal-and-opposite reaction is carried by the shaft bearing and the rack
//! guide, not between the two coupled degrees of freedom — which is why the
//! coupling applies an angular impulse to the pinion and a linear impulse to the
//! rack that are deliberately *not* a conserving action-reaction pair (the
//! missing reaction flows into the mounting frames).
//!
//! The coupling is expressed per substep on the *incremental* motion rather
//! than on an absolute accumulated phase: the shared stepper keeps no per-joint
//! persistent state, so the joint locks the relative *rate*
//! `(u_a . omega_a) - ratio * (u_b . v_b)` to zero every substep. It does not
//! servo an absolute phase offset, which would need a persistent wound-angle the
//! bilateral stepper does not store; that is left to a future stateful joint
//! rather than faked here.
//!
//! # Compliance
//!
//! [`compliance`](RackPinionJoint::compliance) is the inverse stiffness of the
//! coupling. Zero compliance is a *rigid* rack-and-pinion that forces the rate
//! coupling exactly each substep (the ideal meshing-teeth limit); a positive
//! value is a *soft* coupling whose teeth flex under load, so the coupling is
//! satisfied with a finite force proportional to the rate error — a compliant
//! or sprung transmission.
//!
//! Provenance: the rotation-to-translation ratio coupling expressed as a
//! per-substep compliant `XPBD` equality on the pinion's angular displacement
//! and the rack's linear displacement (Macklin et al., "XPBD: Position-Based
//! Simulation of Compliant Constrained Dynamics"), over the world-space inverse
//! inertia, inverse mass, and quaternion kinematics of Baraff & Witkin. No
//! Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

use super::coloring::JointBodies;

/// A rack-and-pinion joint coupling the pinion body's spin about its axis to the
/// rack body's slide along its axis at a fixed ratio: a single scalar
/// mixed angular-linear constraint with no weld.
///
/// Either body may be static (zero inverse mass and inertia); its corrections
/// scale by zero and it acts as an immovable mount.
#[derive(Clone, Copy, Debug)]
pub struct RackPinionJoint {
    /// Index of the pinion body in the [`RigidBodyState`](super::super::RigidBodyState).
    pub pinion: u32,
    /// Index of the rack body in the [`RigidBodyState`](super::super::RigidBodyState).
    pub rack: u32,
    /// Pinion spin axis, in the pinion body's local frame. Need not be
    /// unit-length; only its direction matters.
    pub pinion_axis: Vec3,
    /// Rack slide axis, in the rack body's local frame.
    pub rack_axis: Vec3,
    /// Transmission ratio (radians of pinion rotation per unit of rack travel):
    /// the constraint locks `(u_a . omega_a) = ratio * (u_b . v_b)`. For a
    /// pinion of pitch radius `r`, `ratio = 1 / r`; a negative ratio reverses
    /// the coupling sense.
    pub ratio: f32,
    /// Inverse stiffness of the coupling. Zero is the rigid limit; a positive
    /// value is a soft coupling of gain `1 / compliance`.
    pub compliance: f32,
}

impl RackPinionJoint {
    /// Creates a rack-and-pinion joint from its full parameter set.
    ///
    /// `pinion_axis` / `rack_axis` are the spin and slide axes in each body's
    /// local frame (only their direction matters). `ratio` is the transmission
    /// ratio, `compliance` the inverse coupling stiffness described on the
    /// fields.
    #[must_use]
    pub fn new(
        pinion: u32,
        rack: u32,
        pinion_axis: Vec3,
        rack_axis: Vec3,
        ratio: f32,
        compliance: f32,
    ) -> RackPinionJoint {
        RackPinionJoint {
            pinion,
            rack,
            pinion_axis,
            rack_axis,
            ratio,
            compliance,
        }
    }

    /// Creates a rigid rack-and-pinion: a coupling with zero compliance, so each
    /// substep forces the rate coupling `(u_a . omega_a) = ratio * (u_b . v_b)`
    /// exactly (the ideal rigid meshing-teeth limit, unbounded coupling force).
    #[must_use]
    pub fn rigid(
        pinion: u32,
        rack: u32,
        pinion_axis: Vec3,
        rack_axis: Vec3,
        ratio: f32,
    ) -> RackPinionJoint {
        RackPinionJoint::new(pinion, rack, pinion_axis, rack_axis, ratio, 0.0)
    }

    /// Creates a soft rack-and-pinion of the given coupling stiffness
    /// (`compliance = 1 / stiffness`), so the coupling is satisfied with a
    /// finite force — a flexing or sprung transmission — rather than instantly.
    /// A non-positive stiffness collapses to the rigid coupling.
    #[must_use]
    pub fn soft(
        pinion: u32,
        rack: u32,
        pinion_axis: Vec3,
        rack_axis: Vec3,
        ratio: f32,
        stiffness: f32,
    ) -> RackPinionJoint {
        let compliance = if stiffness > 0.0 {
            1.0 / stiffness
        } else {
            0.0
        };
        RackPinionJoint::new(pinion, rack, pinion_axis, rack_axis, ratio, compliance)
    }

    /// The two body indices this joint couples, in `(pinion, rack)` order.
    #[must_use]
    pub fn bodies(&self) -> (u32, u32) {
        (self.pinion, self.rack)
    }

    /// Packs the joint into its `GPU` storage-buffer representation.
    #[must_use]
    pub(crate) fn to_gpu(self) -> GpuRackPinionJoint {
        GpuRackPinionJoint {
            pinion_axis: [
                self.pinion_axis.x,
                self.pinion_axis.y,
                self.pinion_axis.z,
                0.0,
            ],
            rack_axis: [self.rack_axis.x, self.rack_axis.y, self.rack_axis.z, 0.0],
            pinion: self.pinion,
            rack: self.rack,
            ratio: self.ratio,
            compliance: self.compliance,
        }
    }
}

impl JointBodies for RackPinionJoint {
    fn bodies(&self) -> (u32, u32) {
        (self.pinion, self.rack)
    }
}

/// Device-packed [`RackPinionJoint`]; layout matches `Joint` in
/// `shaders/rigid_joint_rack_pinion.wgsl` (`48` bytes). The pinion and rack axes
/// are padded to `vec4` so each starts on the `16`-byte boundary the storage
/// layout requires; the two `w` lanes are unused. The two body indices, the
/// ratio, and the coupling compliance fill the final `16`-byte block.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct GpuRackPinionJoint {
    /// Body-local pinion spin axis in `xyz`; `w` unused.
    pinion_axis: [f32; 4],
    /// Body-local rack slide axis in `xyz`; `w` unused.
    rack_axis: [f32; 4],
    /// Index of the pinion body.
    pinion: u32,
    /// Index of the rack body.
    rack: u32,
    /// Transmission ratio coupling pinion rotation to rack translation.
    ratio: f32,
    /// Inverse stiffness of the coupling.
    compliance: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_packing_round_trips_fields() {
        let joint = RackPinionJoint::new(
            2,
            9,
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0, 0.0, 0.0),
            4.0,
            0.002,
        );
        let gpu = joint.to_gpu();
        assert_eq!(gpu.pinion, 2);
        assert_eq!(gpu.rack, 9);
        assert_eq!(gpu.pinion_axis, [0.0, 0.0, 1.0, 0.0]);
        assert_eq!(gpu.rack_axis, [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(gpu.ratio, 4.0);
        assert_eq!(gpu.compliance, 0.002);
    }

    #[test]
    fn gpu_struct_is_48_bytes() {
        assert_eq!(size_of::<GpuRackPinionJoint>(), 48);
    }

    #[test]
    fn rigid_builds_a_zero_compliance_coupling() {
        let joint = RackPinionJoint::rigid(0, 1, Vec3::Z, Vec3::X, 3.0);
        assert_eq!(joint.compliance, 0.0);
        assert_eq!(joint.ratio, 3.0);
    }

    #[test]
    fn soft_inverts_stiffness_into_compliance() {
        let joint = RackPinionJoint::soft(0, 1, Vec3::Z, Vec3::X, 2.0, 200.0);
        assert!((joint.compliance - 1.0 / 200.0).abs() < 1e-9);
        assert_eq!(joint.ratio, 2.0);
    }

    #[test]
    fn soft_with_zero_stiffness_is_rigid() {
        let joint = RackPinionJoint::soft(0, 1, Vec3::Z, Vec3::X, 1.0, 0.0);
        assert_eq!(joint.compliance, 0.0);
    }

    #[test]
    fn bodies_returns_pair_in_order() {
        let joint = RackPinionJoint::rigid(6, 1, Vec3::Y, Vec3::X, -1.0);
        assert_eq!(joint.bodies(), (6, 1));
        assert_eq!(JointBodies::bodies(&joint), (6, 1));
    }
}
