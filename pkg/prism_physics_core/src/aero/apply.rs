use glam::{Quat, Vec3};

use super::field::{FlowField, Real};
use super::fluid::{AddedMass, BuoyancyVolume, FluidPlane, MagnusParams};
use super::sanitize::{
    add, clamp_magnitude, cross, finite, non_negative, normalize, rotation, sub, vec3,
};
use super::surface::{AeroSurface, FluidKind, Thruster};

/// Pose and first derivatives supplied by the fixed-step world adapter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AeroBodyState {
    pub position: Vec3,
    pub orientation: Quat,
    pub linear_velocity: Vec3,
    pub angular_velocity: Vec3,
    pub linear_acceleration: Vec3,
    pub angular_acceleration: Vec3,
}

impl Default for AeroBodyState {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            orientation: Quat::IDENTITY,
            linear_velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            linear_acceleration: Vec3::ZERO,
            angular_acceleration: Vec3::ZERO,
        }
    }
}

/// The fluid and inertial constants shared by all panels on one body.
#[derive(Clone, Debug, PartialEq)]
pub struct AeroModel {
    pub air_density: Real,
    pub water_density: Real,
    pub gravity: Vec3,
    pub water_plane: Option<FluidPlane>,
    pub added_mass: Option<AddedMass>,
    pub magnus: Option<MagnusParams>,
}

impl Default for AeroModel {
    fn default() -> Self {
        Self {
            air_density: 1.225,
            water_density: 1000.0,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            water_plane: None,
            added_mass: None,
            magnus: None,
        }
    }
}

/// Aggregate force and torque in world coordinates.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AeroWrench {
    pub force: Vec3,
    pub torque: Vec3,
}

impl AeroWrench {
    pub const ZERO: Self = Self {
        force: Vec3::ZERO,
        torque: Vec3::ZERO,
    };

    pub fn add_force_at_point(&mut self, force: Vec3, point: Vec3, origin: Vec3) {
        let force = vec3(force);
        self.force = add(self.force, force);
        self.add_torque(cross(sub(point, origin), force));
    }

    pub fn add_torque(&mut self, torque: Vec3) {
        self.torque = add(self.torque, torque);
    }

    /// Returns a finite wrench with independently bounded force and torque.
    #[must_use]
    pub fn bounded(self, maximum_force: Real, maximum_torque: Real) -> Self {
        Self {
            force: clamp_magnitude(self.force, maximum_force),
            torque: clamp_magnitude(self.torque, maximum_torque),
        }
    }
}

/// Sink implemented by a world adapter. It deliberately contains only the
/// force/torque operation needed by a fixed sub-step and has no dependency on
/// the storage, solver, or integrator types.
pub trait AeroForceSink {
    fn apply_aero_wrench(&mut self, force: Vec3, torque: Vec3);
}

/// Evaluates every M12 contribution into one world-space wrench.
#[must_use]
pub fn compute_aero_wrench<F: FlowField + ?Sized>(
    state: &AeroBodyState,
    surfaces: &[AeroSurface],
    thrusters: &[Thruster],
    buoyancy: &[BuoyancyVolume],
    field: &F,
    model: &AeroModel,
    time: Real,
) -> AeroWrench {
    let state = sanitized_state(*state);
    let orientation = state.orientation;
    let mut wrench = AeroWrench::ZERO;

    for surface in surfaces {
        let (drag, lift, point) = surface_force(&state, surface, field, model, time);
        wrench.add_force_at_point(add(drag, lift), point, state.position);
    }

    for thruster in thrusters {
        let direction = normalize(orientation * thruster.local_direction, Vec3::ZERO);
        if direction == Vec3::ZERO {
            continue;
        }
        let point = state.position + orientation * vec3(thruster.local_position);
        let thrust = finite(thruster.thrust);
        let force = direction * thrust;
        wrench.add_force_at_point(force, point, state.position);
        let spin = finite(thruster.spin_direction);
        let spin = if spin == 0.0 { 0.0 } else { spin.signum() };
        let reaction =
            -direction * thrust * non_negative(thruster.reaction_torque_coefficient) * spin;
        wrench.add_torque(reaction);
    }

    for volume in buoyancy {
        apply_buoyancy(&mut wrench, &state, volume, model);
    }
    apply_added_mass_and_magnus(&mut wrench, &state, field, model, time);
    AeroWrench {
        force: vec3(wrench.force),
        torque: vec3(wrench.torque),
    }
}

/// Fixed-step adapter entry point. The caller owns body state and applies the
/// returned wrench to its force accumulator through the narrow sink trait.
pub fn apply_aero_forces<S: AeroForceSink, F: FlowField + ?Sized>(
    sink: &mut S,
    state: &AeroBodyState,
    surfaces: &[AeroSurface],
    thrusters: &[Thruster],
    buoyancy: &[BuoyancyVolume],
    field: &F,
    model: &AeroModel,
    time: Real,
) -> AeroWrench {
    let wrench = compute_aero_wrench(state, surfaces, thrusters, buoyancy, field, model, time);
    sink.apply_aero_wrench(wrench.force, wrench.torque);
    wrench
}

pub(super) fn sanitized_state(mut state: AeroBodyState) -> AeroBodyState {
    state.position = vec3(state.position);
    state.orientation = rotation(state.orientation);
    state.linear_velocity = vec3(state.linear_velocity);
    state.angular_velocity = vec3(state.angular_velocity);
    state.linear_acceleration = vec3(state.linear_acceleration);
    state.angular_acceleration = vec3(state.angular_acceleration);
    state
}

pub(super) fn density(model: &AeroModel, fluid: FluidKind) -> Real {
    match fluid {
        FluidKind::Air => non_negative(model.air_density),
        FluidKind::Water => non_negative(model.water_density),
    }
}

pub(super) fn fluid_fraction(
    point: Vec3,
    fluid: FluidKind,
    depth: Real,
    model: &AeroModel,
) -> Real {
    let Some(plane) = model.water_plane else {
        return if fluid == FluidKind::Air { 1.0 } else { 0.0 };
    };
    let height = plane.signed_height(point);
    let depth = non_negative(depth).max(1.0e-5);
    let water = (0.5 - height / (2.0 * depth)).clamp(0.0, 1.0);
    if fluid == FluidKind::Air {
        1.0 - water
    } else {
        water
    }
}

pub(super) fn surface_force<F: FlowField + ?Sized>(
    state: &AeroBodyState,
    surface: &AeroSurface,
    field: &F,
    model: &AeroModel,
    time: Real,
) -> (Vec3, Vec3, Vec3) {
    let orientation = state.orientation;
    let point = state.position + orientation * vec3(surface.local_position);
    let (local_normal, local_chord, local_span) = surface.basis();
    let normal = normalize(orientation * local_normal, Vec3::Y);
    let chord = normalize(orientation * local_chord, Vec3::X);
    let span = normalize(orientation * local_span, normal.cross(chord));
    let point_velocity =
        state.linear_velocity + state.angular_velocity.cross(point - state.position);
    let wind = field.sample(point, time);
    let flow = vec3(wind.velocity - point_velocity);
    let speed = flow.length();
    let area = non_negative(surface.area);
    if !speed.is_finite() || speed <= 1.0e-6 || area <= 0.0 {
        return (Vec3::ZERO, Vec3::ZERO, point);
    }
    let immersion = fluid_fraction(
        point,
        surface.fluid,
        surface.immersion_depth.max(surface.chord_length * 0.5),
        model,
    );
    if immersion <= 0.0 {
        return (Vec3::ZERO, Vec3::ZERO, point);
    }
    let rho = density(model, surface.fluid);
    let q = 0.5 * rho * speed * speed;
    let flow_hat = flow / speed;
    let n_component = flow.dot(normal) / speed;
    let c_component = flow.dot(chord) / speed;
    let s_component = flow.dot(span) / speed;
    let ground = field.ground_effect(point);
    let drag = (normal * (n_component * non_negative(surface.drag_normal))
        + chord * (c_component * non_negative(surface.drag_chord))
        + span * (s_component * non_negative(surface.drag_span)))
        * (q * area * non_negative(ground.drag));
    let mut angle = flow.dot(normal).atan2(flow.dot(chord));
    if let Some(control) = surface.control {
        angle += control.effective_deflection();
    }
    // Spanwise flow contributes drag, not sectional wing dynamic pressure.
    let sectional_speed_squared = flow.dot(normal).powi(2) + flow.dot(chord).powi(2);
    let lift_direction = normalize(normal - flow_hat * normal.dot(flow_hat), Vec3::ZERO);
    let lift = lift_direction
        * (0.5
            * rho
            * sectional_speed_squared
            * area
            * surface.lift_curve.coefficient(angle)
            * non_negative(ground.lift));
    (vec3(drag * immersion), vec3(lift * immersion), point)
}

fn apply_buoyancy(
    wrench: &mut AeroWrench,
    state: &AeroBodyState,
    volume: &BuoyancyVolume,
    model: &AeroModel,
) {
    if volume.fluid == FluidKind::Air && model.water_plane.is_none() {
        let point = state.position + state.orientation * vec3(volume.local_center);
        let force =
            -vec3(model.gravity) * non_negative(model.air_density) * non_negative(volume.volume);
        wrench.add_force_at_point(force, point, state.position);
        return;
    }
    let Some(plane) = model.water_plane else {
        return;
    };
    let n = usize::from(volume.samples_per_axis.clamp(1, 8));
    let half = Vec3::new(
        non_negative(volume.local_half_extents.x),
        non_negative(volume.local_half_extents.y),
        non_negative(volume.local_half_extents.z),
    );
    let voxel_volume = non_negative(volume.volume) / (n * n * n) as Real;
    if voxel_volume <= 0.0 {
        return;
    }
    let axes = [
        state.orientation * Vec3::X,
        state.orientation * Vec3::Y,
        state.orientation * Vec3::Z,
    ];
    let plane_normal = normalize(plane.normal, Vec3::Y);
    let cell_half_projection = (axes[0].dot(plane_normal).abs() * half.x
        + axes[1].dot(plane_normal).abs() * half.y
        + axes[2].dot(plane_normal).abs() * half.z)
        / n as Real;
    let gravity = vec3(model.gravity);
    for ix in 0..n {
        for iy in 0..n {
            for iz in 0..n {
                let f = |index: usize, extent: Real| {
                    -extent + (2.0 * index as Real + 1.0) * extent / n as Real
                };
                let local = vec3(volume.local_center)
                    + Vec3::new(f(ix, half.x), f(iy, half.y), f(iz, half.z));
                let point = state.position + state.orientation * local;
                let height = plane.signed_height(point);
                let water_fraction = if cell_half_projection > 1.0e-6 {
                    (0.5 - height / (2.0 * cell_half_projection)).clamp(0.0, 1.0)
                } else if height <= 0.0 {
                    1.0
                } else {
                    0.0
                };
                let (fraction, rho) = if volume.fluid == FluidKind::Water {
                    (water_fraction, non_negative(plane.density))
                } else {
                    (1.0 - water_fraction, non_negative(model.air_density))
                };
                if fraction <= 0.0 {
                    continue;
                }
                let force = -gravity * rho * voxel_volume * fraction;
                wrench.add_force_at_point(force, point, state.position);
            }
        }
    }
}

fn apply_added_mass_and_magnus<F: FlowField + ?Sized>(
    wrench: &mut AeroWrench,
    state: &AeroBodyState,
    field: &F,
    model: &AeroModel,
    time: Real,
) {
    let center_flow = field.sample(state.position, time);
    if let Some(added) = model.added_mass {
        let fraction = fluid_fraction(state.position, added.fluid, added.reference_radius, model);
        let rho = density(model, added.fluid);
        let inverse = state.orientation.conjugate();
        let relative_linear = inverse * vec3(state.linear_acceleration - center_flow.acceleration);
        let angular = inverse * state.angular_acceleration;
        let force = state.orientation * (-relative_linear * added.mass_diagonal(rho, fraction));
        let torque = state.orientation * (-angular * added.inertia_diagonal(rho, fraction));
        wrench.add_force_at_point(force, state.position, state.position);
        wrench.add_torque(torque);
    }
    if let Some(magnus) = model.magnus {
        let fraction = fluid_fraction(state.position, magnus.fluid, magnus.effective_radius, model);
        let rho = density(model, magnus.fluid) * fraction;
        let relative_velocity = center_flow.velocity - state.linear_velocity;
        let force = state.angular_velocity.cross(relative_velocity)
            * (rho
                * finite(magnus.coefficient)
                * non_negative(magnus.reference_area)
                * non_negative(magnus.effective_radius));
        wrench.add_force_at_point(force, state.position, state.position);
    }
}
