use glam::Vec3;

use super::apply::AeroBodyState;
use super::field::Real;
use super::field::{FlowJet, Slipstream};
use super::sanitize::{finite, non_negative, normalize, rotation, unit, vec3};

/// Which fluid supplies dynamic pressure to a panel or added-mass model.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FluidKind {
    #[default]
    Air,
    Water,
}

/// Continuous lift model with a smooth transition through stall.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LiftCurve {
    pub lift_slope: Real,
    pub stall_angle: Real,
    pub stall_blend: Real,
    pub post_stall_slope: Real,
    pub maximum_lift: Real,
}

impl Default for LiftCurve {
    fn default() -> Self {
        Self {
            lift_slope: 2.0 * core::f32::consts::PI,
            stall_angle: 15.0_f32.to_radians(),
            stall_blend: 4.0_f32.to_radians(),
            post_stall_slope: -0.35,
            maximum_lift: 1.6,
        }
    }
}

impl LiftCurve {
    #[must_use]
    pub fn coefficient(&self, angle: Real) -> Real {
        // A two-sided panel has a pi-periodic polar, including reverse flow.
        let angle = (finite(angle) + core::f32::consts::FRAC_PI_2)
            .rem_euclid(core::f32::consts::PI)
            - core::f32::consts::FRAC_PI_2;
        let slope = finite(self.lift_slope);
        let stall = non_negative(self.stall_angle).clamp(1.0e-4, core::f32::consts::FRAC_PI_2);
        let blend = non_negative(self.stall_blend).max(1.0e-5);
        let max_lift = non_negative(self.maximum_lift);
        let magnitude = angle.abs();
        let linear = (slope * angle).clamp(-max_lift, max_lift);
        let at_stall = (slope * stall).clamp(0.0, max_lift);
        let post = (2.0 * angle).sin()
            * (at_stall + finite(self.post_stall_slope) * (magnitude - stall)).clamp(0.0, max_lift);
        let t = ((magnitude - stall) / blend).clamp(0.0, 1.0);
        let smooth = t * t * (3.0 - 2.0 * t);
        linear + (post - linear) * smooth
    }
}

/// Deflection of a flap, elevator, aileron, rudder, or other control panel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ControlSurface {
    pub deflection: Real,
    pub maximum_deflection: Real,
    pub effectiveness: Real,
}

impl Default for ControlSurface {
    fn default() -> Self {
        Self {
            deflection: 0.0,
            maximum_deflection: 45.0_f32.to_radians(),
            effectiveness: 1.0,
        }
    }
}

impl ControlSurface {
    #[must_use]
    pub fn effective_deflection(&self) -> Real {
        finite(self.deflection).clamp(
            -non_negative(self.maximum_deflection),
            non_negative(self.maximum_deflection),
        ) * unit(self.effectiveness)
    }
}

/// One rigid-body aerodynamic surface element.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AeroSurface {
    pub local_position: Vec3,
    pub local_normal: Vec3,
    pub local_chord: Vec3,
    pub local_span: Vec3,
    pub area: Real,
    pub chord_length: Real,
    pub drag_normal: Real,
    pub drag_chord: Real,
    pub drag_span: Real,
    pub lift_curve: LiftCurve,
    pub control: Option<ControlSurface>,
    pub fluid: FluidKind,
    pub immersion_depth: Real,
}

impl AeroSurface {
    #[must_use]
    pub fn new(local_position: Vec3, local_normal: Vec3, local_chord: Vec3, area: Real) -> Self {
        let normal = normalize(local_normal, Vec3::Y);
        let chord = tangent(normal, local_chord);
        let span = normalize(normal.cross(chord), Vec3::Z);
        Self {
            local_position,
            local_normal: normal,
            local_chord: chord,
            local_span: span,
            area,
            chord_length: 1.0,
            drag_normal: 1.0,
            drag_chord: 0.02,
            drag_span: 0.05,
            lift_curve: LiftCurve::default(),
            control: None,
            fluid: FluidKind::Air,
            immersion_depth: 0.25,
        }
    }

    #[must_use]
    pub fn basis(&self) -> (Vec3, Vec3, Vec3) {
        let normal = normalize(self.local_normal, Vec3::Y);
        let chord = tangent(normal, self.local_chord);
        let span = normal.cross(chord);
        let span = if span.dot(vec3(self.local_span)) < 0.0 {
            -span
        } else {
            span
        };
        (normal, chord, span)
    }
}

fn tangent(normal: Vec3, candidate: Vec3) -> Vec3 {
    let candidate = normalize(candidate, Vec3::ZERO);
    let axis = if normal.x.abs() <= normal.y.abs() && normal.x.abs() <= normal.z.abs() {
        Vec3::X
    } else if normal.y.abs() <= normal.z.abs() {
        Vec3::Y
    } else {
        Vec3::Z
    };
    let fallback = normalize(axis - normal * axis.dot(normal), Vec3::X);
    normalize(candidate - normal * candidate.dot(normal), fallback)
}

/// An offset propulsor. The force and reaction torque are both applied at the
/// same local point, so the evaluator preserves the moment arm exactly.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Thruster {
    pub local_position: Vec3,
    pub local_direction: Vec3,
    pub thrust: Real,
    pub reaction_torque_coefficient: Real,
    pub spin_direction: Real,
    /// Optional body-local wake template, active only while thrust is nonzero.
    /// Its signed speed is authored independently of thrust magnitude.
    pub slipstream: Option<FlowJet>,
}

impl Thruster {
    #[must_use]
    pub fn new(local_position: Vec3, local_direction: Vec3, thrust: Real) -> Self {
        Self {
            local_position,
            local_direction,
            thrust,
            reaction_torque_coefficient: 0.0,
            spin_direction: 1.0,
            slipstream: None,
        }
    }

    #[must_use]
    pub fn with_reaction(mut self, coefficient: Real, spin_direction: Real) -> Self {
        self.reaction_torque_coefficient = coefficient;
        self.spin_direction = spin_direction;
        self
    }

    /// Attaches a body-local wake; the world adapter publishes transformed
    /// wakes before evaluating any body so registry order cannot affect flow.
    #[must_use]
    pub fn with_slipstream(mut self, wake: FlowJet) -> Self {
        self.slipstream = Some(wake);
        self
    }

    /// Returns the wake in world space, or none for a disabled propulsor.
    #[must_use]
    pub fn world_slipstream(&self, state: &AeroBodyState) -> Option<Slipstream> {
        if finite(self.thrust) == 0.0 {
            return None;
        }
        let mut wake = self.slipstream?;
        let orientation = rotation(state.orientation);
        wake.origin = vec3(state.position) + orientation * vec3(wake.origin);
        wake.direction = normalize(orientation * vec3(wake.direction), Vec3::ZERO);
        Some(Slipstream(wake))
    }
}
