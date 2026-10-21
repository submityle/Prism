use glam::{Quat, Vec3};
use prism_physics_core::aero::{
    self, apply_aero_forces, compute_aero_wrench, AddedMass, AeroBodyState, AeroForceSink,
    AeroModel, AeroSurface, BuoyancyVolume, ControlSurface, FlowField, FluidKind, FluidPlane,
    GroundEffect, LiftCurve, MagnusParams, Slipstream, Thruster, Turbulence, WindField, WindSample,
};
use prism_physics_core::solver::xpbd::{
    contact_constraint::{solve_positions, ContactConstraint},
    rigid::{apply_velocity_impulse, inv_inertia_world, inv_mass_world},
};
use prism_physics_core::{
    AeroBodyConfig, BodyDesc, BodyHandle, ContactManifold, ContactPoint, MassProperties,
    PhysicsMaterial, PhysicsWorld, WorldConfig, XpbdConfig,
};

const EPS: f32 = 1.0e-4;

fn state() -> AeroBodyState {
    AeroBodyState {
        position: Vec3::new(0.0, 2.0, 0.0),
        orientation: Quat::IDENTITY,
        ..AeroBodyState::default()
    }
}

fn model() -> AeroModel {
    AeroModel {
        water_plane: Some(FluidPlane::horizontal(0.0, 1000.0)),
        ..AeroModel::default()
    }
}

#[test]
fn anisotropic_panel_drag_follows_each_body_axis() {
    let mut panel = AeroSurface::new(Vec3::ZERO, Vec3::Y, Vec3::X, 2.0);
    panel.lift_curve = LiftCurve {
        lift_slope: 0.0,
        maximum_lift: 0.0,
        ..LiftCurve::default()
    };
    panel.drag_normal = 4.0;
    panel.drag_chord = 1.0;
    panel.drag_span = 0.0;
    let wind = WindField::new(Vec3::new(3.0, 0.0, 0.0));
    let result = compute_aero_wrench(&state(), &[panel], &[], &[], &wind, &model(), 0.0);
    assert!(result.force.x > 0.0);
    assert!(result.force.y.abs() < EPS);
    assert!(result.force.z.abs() < EPS);
}

#[test]
fn lift_curve_is_finite_and_continuous_through_stall() {
    let curve = LiftCurve::default();
    let stall = curve.stall_angle;
    let left = curve.coefficient(stall - 0.0001);
    let right = curve.coefficient(stall + 0.0001);
    assert!((left - right).abs() < 0.01);
    for i in -100..=100 {
        let angle = i as f32 * 0.03;
        assert!(curve.coefficient(angle).is_finite());
    }
}

#[test]
fn control_deflection_changes_lift_sign_consistently() {
    let mut panel = AeroSurface::new(Vec3::ZERO, Vec3::Y, Vec3::X, 1.0);
    panel.control = Some(ControlSurface {
        deflection: 10.0_f32.to_radians(),
        ..ControlSurface::default()
    });
    let wind = WindField::new(Vec3::new(10.0, 0.0, 0.0));
    let result = compute_aero_wrench(&state(), &[panel], &[], &[], &wind, &model(), 0.0);
    assert!(result.force.y > 0.0);
}

#[test]
fn offset_thruster_preserves_force_moment_and_reaction_torque() {
    let thruster = Thruster::new(Vec3::new(0.0, 0.0, 2.0), Vec3::X, 10.0).with_reaction(0.5, 1.0);
    let result = compute_aero_wrench(
        &state(),
        &[],
        &[thruster],
        &[],
        &WindField::default(),
        &model(),
        0.0,
    );
    assert!((result.force.x - 10.0).abs() < EPS);
    assert!((result.torque.y - 2.0 * 10.0).abs() < EPS);
    assert!((result.torque.x + 5.0).abs() < EPS);
}

#[test]
fn partial_buoyancy_is_between_zero_and_full_volume() {
    let mut body = state();
    body.position.y = 0.0;
    let volume =
        BuoyancyVolume::box_volume(Vec3::ZERO, Vec3::splat(1.0), FluidKind::Water).with_samples(6);
    let result = compute_aero_wrench(
        &body,
        &[],
        &[],
        &[volume],
        &WindField::default(),
        &model(),
        0.0,
    );
    let full = 1000.0 * 9.81 * 8.0;
    assert!(result.force.y > 0.0);
    assert!(result.force.y < full);
}

#[test]
fn air_buoyancy_does_not_require_a_water_plane() {
    let mut body = state();
    body.position.y = 10.0;
    let volume = BuoyancyVolume::box_volume(Vec3::ZERO, Vec3::splat(1.0), FluidKind::Air);
    let result = compute_aero_wrench(
        &body,
        &[],
        &[],
        &[volume],
        &WindField::default(),
        &AeroModel::default(),
        0.0,
    );
    assert!((result.force.y - 1.225 * 9.81 * 8.0).abs() < EPS);
}

#[test]
fn degenerate_surface_axes_still_form_an_orthonormal_basis() {
    let panel = AeroSurface::new(Vec3::ZERO, Vec3::Y, Vec3::Y, 1.0);
    let (normal, chord, span) = panel.basis();
    assert!((normal.length() - 1.0).abs() < EPS);
    assert!((chord.length() - 1.0).abs() < EPS);
    assert!((span.length() - 1.0).abs() < EPS);
    assert!(normal.dot(chord).abs() < EPS);
    assert!(normal.dot(span).abs() < EPS);
    assert!(chord.dot(span).abs() < EPS);
}

#[test]
fn added_mass_opposes_relative_acceleration() {
    let mut m = model();
    m.added_mass = Some(AddedMass {
        fluid: FluidKind::Air,
        volume: 2.0,
        linear_coefficients: Vec3::ONE,
        angular_coefficients: Vec3::ZERO,
        reference_radius: 0.0,
    });
    let mut body = state();
    body.linear_acceleration = Vec3::X * 3.0;
    let result = compute_aero_wrench(&body, &[], &[], &[], &WindField::default(), &m, 0.0);
    assert!(result.force.x < 0.0);
}

#[test]
fn added_mass_uses_body_axes_and_radius_squared_inertia() {
    let added = AddedMass {
        fluid: FluidKind::Water,
        volume: 2.0,
        linear_coefficients: Vec3::new(1.0, 2.0, 3.0),
        angular_coefficients: Vec3::ONE,
        reference_radius: 2.0,
    };
    let mass = added.mass_diagonal(1000.0, 0.5);
    let inertia = added.inertia_diagonal(1000.0, 0.5);
    assert_eq!(mass, Vec3::new(1000.0, 2000.0, 3000.0));
    assert_eq!(inertia, Vec3::splat(4000.0));
    assert_eq!(
        added.implicit_inverse_mass(Vec3::ONE, 1000.0, 0.5),
        Vec3::new(1.0 / 1001.0, 1.0 / 2001.0, 1.0 / 3001.0,)
    );
}

#[test]
fn added_mass_force_rotates_from_body_axes_to_world_axes() {
    let mut m = model();
    m.added_mass = Some(AddedMass {
        fluid: FluidKind::Water,
        volume: 1.0,
        linear_coefficients: Vec3::new(2.0, 5.0, 7.0),
        ..AddedMass::default()
    });
    let mut body = state();
    body.position.y = -1.0;
    body.orientation = Quat::from_rotation_z(core::f32::consts::FRAC_PI_2);
    body.linear_acceleration = Vec3::Y;
    let result = compute_aero_wrench(&body, &[], &[], &[], &WindField::default(), &m, 0.0);
    assert!(result.force.x.abs() < EPS);
    assert!((result.force.y + 2000.0).abs() < 1.0e-3);
}

#[test]
fn added_mass_immersion_is_applied_once() {
    let mut model = model();
    model.added_mass = Some(AddedMass {
        fluid: FluidKind::Water,
        volume: 2.0,
        linear_coefficients: Vec3::ONE,
        ..AddedMass::default()
    });
    let mut body = state();
    body.position.y = 0.0;
    body.linear_acceleration = Vec3::X * 3.0;
    let result = compute_aero_wrench(&body, &[], &[], &[], &WindField::default(), &model, 0.0);
    assert!((result.force.x + 1000.0 * 2.0 * 0.5 * 3.0).abs() < EPS);
}

#[test]
fn added_mass_immersion_helper_matches_the_force_path() {
    let mut model = AeroModel::default();
    model.added_mass = Some(AddedMass {
        fluid: FluidKind::Water,
        volume: 1.0,
        reference_radius: 2.0,
        ..AddedMass::default()
    });
    model.water_plane = Some(FluidPlane::horizontal(0.0, 1000.0));
    let config = AeroBodyConfig::new(model);

    assert!((config.added_mass_immersion(Vec3::new(0.0, -2.0, 0.0)) - 1.0).abs() < EPS);
    assert!((config.added_mass_immersion(Vec3::new(0.0, 2.0, 0.0))).abs() < EPS);
}

#[test]
fn magnus_direction_is_spin_cross_flow() {
    let mut m = model();
    m.magnus = Some(MagnusParams {
        fluid: FluidKind::Air,
        coefficient: 1.0,
        reference_area: 1.0,
        effective_radius: 1.0,
    });
    let mut body = state();
    body.angular_velocity = Vec3::Z;
    let result = compute_aero_wrench(&body, &[], &[], &[], &WindField::new(Vec3::X), &m, 0.0);
    assert!(result.force.y > 0.0);
}

#[test]
fn wind_field_composes_gust_turbulence_and_jets_deterministically() {
    let field = WindField::new(Vec3::X)
        .with_turbulence(Turbulence {
            amplitude: Vec3::splat(0.2),
            cell_size: 2.0,
            temporal_frequency: 0.5,
            seed: 7,
        })
        .with_slipstream(Slipstream(aero::FlowJet::new(
            Vec3::ZERO,
            Vec3::X,
            2.0,
            5.0,
            3.0,
        )))
        .with_ground_effect(GroundEffect {
            plane_point: Vec3::ZERO,
            plane_normal: Vec3::Y,
            influence_height: 4.0,
            lift_gain: 1.0,
            drag_gain: 0.5,
        });
    let a = field.sample(Vec3::new(1.0, 1.0, 0.0), 1.25);
    let b = field.sample(Vec3::new(1.0, 1.0, 0.0), 1.25);
    assert_eq!(a, b);
    assert!(a.velocity.is_finite() && a.acceleration.is_finite());
    assert!(field.ground_effect(Vec3::new(0.0, 0.5, 0.0)).lift > 1.0);
}

#[test]
fn extreme_turbulence_coordinates_remain_finite_and_deterministic() {
    let turbulence = Turbulence {
        amplitude: Vec3::splat(10.0),
        cell_size: 0.001,
        temporal_frequency: 1000.0,
        seed: u32::MAX,
    };
    let position = Vec3::splat(f32::MAX);
    let a = turbulence.sample(position, f32::MAX);
    let b = turbulence.sample(position, f32::MAX);
    assert_eq!(a, b);
    assert!(a.velocity.is_finite() && a.acceleration.is_finite());
}

#[test]
fn ground_effect_can_reduce_drag_without_making_it_negative() {
    let scale = GroundEffect {
        plane_point: Vec3::ZERO,
        plane_normal: Vec3::Y,
        influence_height: 2.0,
        lift_gain: 1.0,
        drag_gain: -0.5,
    }
    .scale_at(Vec3::ZERO);
    assert!(scale.lift > 1.0);
    assert!((scale.drag - 0.5).abs() < EPS);
}

#[test]
fn reverse_flow_lift_is_finite_and_changes_with_small_incidence() {
    let curve = LiftCurve::default();
    assert!(curve.coefficient(core::f32::consts::PI).abs() < EPS);
    let positive = curve.coefficient(core::f32::consts::PI + 0.1);
    let negative = curve.coefficient(core::f32::consts::PI - 0.1);
    assert!(positive.is_finite() && negative.is_finite());
    assert!((positive + negative).abs() < 0.05);
}

#[test]
fn opposite_thruster_reactions_cancel() {
    let left = Thruster::new(Vec3::ZERO, Vec3::X, 10.0).with_reaction(0.5, 1.0);
    let right = Thruster::new(Vec3::ZERO, Vec3::X, 10.0).with_reaction(0.5, -1.0);
    let result = compute_aero_wrench(
        &state(),
        &[],
        &[left, right],
        &[],
        &WindField::default(),
        &model(),
        0.0,
    );
    assert!(result.torque.abs().max_element() < EPS);
}

#[test]
fn zero_thrust_suppresses_its_wake() {
    let thruster = Thruster::new(Vec3::ZERO, Vec3::X, 0.0).with_slipstream(aero::FlowJet::new(
        Vec3::ZERO,
        Vec3::X,
        1.0,
        4.0,
        20.0,
    ));
    assert!(thruster.world_slipstream(&state()).is_none());
}

#[test]
fn bounded_wrench_caps_force_and_torque_without_nan() {
    let bounded = aero::AeroWrench {
        force: Vec3::splat(f32::MAX),
        torque: Vec3::splat(-f32::MAX),
    }
    .bounded(3.0, 4.0);
    assert!((bounded.force.length() - 3.0).abs() < EPS);
    assert!((bounded.torque.length() - 4.0).abs() < EPS);
}

#[test]
fn substep_helper_applies_added_mass_implicitly_without_velocity_history() {
    let mut m = model();
    m.added_mass = Some(AddedMass {
        fluid: FluidKind::Water,
        volume: 1.0,
        linear_coefficients: Vec3::ONE,
        angular_coefficients: Vec3::ONE,
        reference_radius: 1.0,
        ..AddedMass::default()
    });
    let config = AeroBodyConfig::new(m);
    let mut body = state();
    body.position.y = -1.0;
    let external = aero::AeroWrench {
        force: Vec3::X * 1000.0,
        torque: Vec3::ZERO,
    };
    let correction = config.compute_substep_wrench(
        &body,
        &WindField::default(),
        0.0,
        1.0 / 60.0,
        1.0,
        Vec3::ONE,
        external,
        Vec3::ZERO,
    );
    let total = external.force + correction.force;
    assert!((total.x - 1000.0 / 1001.0).abs() < EPS);
    assert!(correction.torque.is_finite());
}

#[test]
fn invalid_inputs_never_escape_as_nan() {
    let mut panel = AeroSurface::new(
        Vec3::new(f32::NAN, 0.0, 0.0),
        Vec3::NAN,
        Vec3::INFINITY,
        f32::NAN,
    );
    panel.drag_normal = f32::NAN;
    panel.lift_curve.lift_slope = f32::INFINITY;
    let mut body = state();
    body.position = Vec3::NAN;
    body.orientation = Quat::from_xyzw(f32::NAN, 0.0, 0.0, 0.0);
    let result = compute_aero_wrench(
        &body,
        &[panel],
        &[],
        &[],
        &WindField::new(Vec3::NAN),
        &model(),
        f32::NAN,
    );
    assert!(result.force.is_finite() && result.torque.is_finite());
}

#[derive(Default)]
struct Sink {
    wrench: aero::AeroWrench,
}

impl AeroForceSink for Sink {
    fn apply_aero_wrench(&mut self, force: Vec3, torque: Vec3) {
        self.wrench = aero::AeroWrench { force, torque };
    }
}

#[test]
fn apply_api_forwards_exact_wrench_to_substep_sink() {
    let panel = AeroSurface::new(Vec3::ZERO, Vec3::Y, Vec3::X, 1.0);
    let mut sink = Sink::default();
    let expected = compute_aero_wrench(
        &state(),
        &[panel],
        &[],
        &[],
        &WindField::new(Vec3::Y),
        &model(),
        0.0,
    );
    let actual = apply_aero_forces(
        &mut sink,
        &state(),
        &[panel],
        &[],
        &[],
        &WindField::new(Vec3::Y),
        &model(),
        0.0,
    );
    assert_eq!(expected, actual);
    assert_eq!(sink.wrench, expected);
}

struct ConstantField;
impl FlowField for ConstantField {
    fn sample(&self, _position: Vec3, _time: f32) -> WindSample {
        WindSample {
            velocity: Vec3::X,
            acceleration: Vec3::ZERO,
        }
    }
}

#[test]
fn custom_flow_field_is_usable_without_world_dependencies() {
    let panel = AeroSurface::new(Vec3::ZERO, Vec3::Y, Vec3::X, 1.0);
    let result = compute_aero_wrench(&state(), &[panel], &[], &[], &ConstantField, &model(), 0.0);
    assert!(result.force.is_finite() && result.torque.is_finite());
}

#[test]
fn world_registration_rejects_stale_body_handles() {
    let mut world = PhysicsWorld::new(WorldConfig::with_gravity(Vec3::ZERO));
    assert!(world
        .register_aero_body(
            prism_physics_core::BodyHandle::INVALID,
            AeroBodyConfig::default()
        )
        .is_err());
}

#[test]
fn world_samples_shared_field_and_queues_aero_wrench() {
    let mut world = PhysicsWorld::new(WorldConfig::with_gravity(Vec3::ZERO));
    world.set_aero_field(WindField::new(Vec3::X));
    let body = world.spawn(BodyDesc::dynamic_at(Vec3::ZERO));
    let mut panel = AeroSurface::new(Vec3::ZERO, Vec3::Y, Vec3::X, 1.0);
    panel.lift_curve = LiftCurve {
        lift_slope: 0.0,
        maximum_lift: 0.0,
        ..LiftCurve::default()
    };
    panel.drag_normal = 0.0;
    panel.drag_chord = 1.0;
    panel.drag_span = 0.0;
    world
        .register_aero_body(body, AeroBodyConfig::default().with_surface(panel))
        .expect("live body must accept aerodynamic data");

    let sample = world
        .sample_aero_body(body, 0.0)
        .expect("registered body must be sampleable");
    assert!(sample.wrench.force.x > 0.0);
    let report = world.step_aero_substep(0.1);
    assert_eq!(
        (report.registered, report.sampled, report.applied),
        (1, 1, 1)
    );
    let pending = world
        .external_forces
        .pending(body)
        .expect("aero pass must queue a force");
    assert_eq!(pending.force, report.wrench.force);
    assert_eq!(pending.torque, report.wrench.torque);
}

#[test]
fn world_fixed_substep_uses_pending_load_for_implicit_added_mass() {
    let mut world = PhysicsWorld::new(WorldConfig::with_gravity(Vec3::ZERO));
    let body = world.spawn(BodyDesc::dynamic_at(Vec3::new(0.0, -1.0, 0.0)));
    let mut model = AeroModel::default();
    model.added_mass = Some(AddedMass {
        fluid: FluidKind::Water,
        volume: 1.0,
        linear_coefficients: Vec3::ONE,
        ..AddedMass::default()
    });
    model.water_plane = Some(FluidPlane::horizontal(0.0, 1000.0));
    world
        .register_aero_body(body, AeroBodyConfig::new(model))
        .expect("live body must accept aerodynamic data");
    assert!(world.apply_force(body, Vec3::X * 1000.0, Vec3::new(0.0, -1.0, 0.0)));

    let report = world.step_aero_substep(1.0 / 60.0);
    assert_eq!((report.sampled, report.applied), (1, 1));
    let pending = world
        .external_forces
        .pending(body)
        .expect("added mass must queue a wrench");
    assert!((pending.force.x - 1000.0 / 1001.0).abs() < EPS);
    assert!((world.aero_time - 1.0 / 60.0).abs() < EPS);
}

fn assert_vec_close(actual: Vec3, expected: Vec3) {
    assert!(
        (actual - expected).abs().max_element() < EPS,
        "actual {actual:?}, expected {expected:?}"
    );
}

fn implicit_added_mass_world(position: Vec3, orientation: Quat) -> (PhysicsWorld, BodyHandle) {
    let mut world = PhysicsWorld::new(WorldConfig::with_gravity(Vec3::ZERO));
    let mut desc = BodyDesc::dynamic_at(position)
        .with_mass_properties(MassProperties {
            inv_mass: 0.5,
            inv_inertia: Vec3::new(0.25, 0.5, 1.0),
        })
        .with_material(PhysicsMaterial::FRICTIONLESS);
    desc.orientation = orientation;
    let body = world.spawn(desc);
    let model = AeroModel {
        // AddedMass reads the model density, not FluidPlane::density.
        water_density: 2.0,
        added_mass: Some(AddedMass {
            fluid: FluidKind::Water,
            volume: 2.0,
            linear_coefficients: Vec3::new(0.5, 1.5, 2.5),
            angular_coefficients: Vec3::new(1.0, 2.0, 3.0),
            reference_radius: 2.0,
        }),
        water_plane: Some(FluidPlane::horizontal(0.0, 2.0)),
        ..AeroModel::default()
    };
    world
        .register_aero_body(body, AeroBodyConfig::new(model))
        .expect("live body must accept added mass");
    (world, body)
}

// Closed-form expectations: m=2, I=(4,2,1), rho*V=4, r^2=4.
// Keep these independent of AddedMass's implicit-inverse helpers.
fn expected_implicit_inverse(fraction: f32) -> (Vec3, Vec3) {
    let mass = Vec3::splat(2.0) + fraction * Vec3::new(2.0, 6.0, 10.0);
    let inertia = Vec3::new(4.0, 2.0, 1.0) + fraction * Vec3::new(16.0, 32.0, 48.0);
    (Vec3::ONE / mass, Vec3::ONE / inertia)
}

#[test]
fn pending_wrench_implicit_added_mass_matches_xpbd_velocity_response() {
    let h = 0.125;
    for (height, fraction) in [(-4.0, 1.0), (0.0, 0.5), (4.0, 0.0)] {
        for orientation in [
            Quat::IDENTITY,
            Quat::from_rotation_z(0.7) * Quat::from_rotation_x(0.4),
        ] {
            // Check the staged queue/consume API and the backend's actual
            // prepare boundary independently; never sample aero twice.
            for prepare in [false, true] {
                let position = Vec3::Y * height;
                let (mut world, body) = implicit_added_mass_world(position, orientation);
                let authored = world.bodies.mass_properties(body).unwrap();
                let force_local = Vec3::new(6.0, -4.0, 3.0);
                let offset_local = Vec3::new(0.2, 0.3, -0.4);
                let torque_local = Vec3::new(2.0, 3.0, -5.0) + offset_local.cross(force_local);
                assert!(world.apply_force(
                    body,
                    orientation * force_local,
                    position + orientation * offset_local,
                ));
                assert!(world.apply_torque(body, orientation * Vec3::new(2.0, 3.0, -5.0)));
                let (inverse_mass, inverse_inertia) = expected_implicit_inverse(fraction);
                let expected_linear = orientation * (inverse_mass * force_local) * h;
                let expected_angular = orientation * (inverse_inertia * torque_local) * h;
                if prepare {
                    assert_eq!(world.prepare_external_substep(h), 1);
                } else {
                    let original = world.external_forces.pending(body).unwrap();
                    let report = world.step_aero_substep(h);
                    assert_eq!((report.sampled, report.applied), (1, 1));
                    let pending = world.external_forces.pending(body).unwrap();
                    // The queue holds a corrected wrench, not an already
                    // integrated velocity or the external wrench added twice.
                    assert_vec_close(pending.force, original.force + report.wrench.force);
                    assert_vec_close(pending.torque, original.torque + report.wrench.torque);
                    assert_vec_close(pending.force, expected_linear / (authored.inv_mass * h));
                    assert_vec_close(
                        orientation.conjugate() * pending.torque,
                        inverse_inertia * torque_local / authored.inv_inertia,
                    );
                    assert_eq!(world.consume_external_forces(h), 1);
                }
                assert!(world.external_forces.is_empty());
                assert!((world.aero_time - h).abs() < EPS);
                assert_vec_close(world.bodies.linear_velocity(body).unwrap(), expected_linear);
                assert_vec_close(
                    world.bodies.angular_velocity(body).unwrap(),
                    expected_angular,
                );
                assert_eq!(world.bodies.position(body), Some(position));
                assert_eq!(world.bodies.orientation(body), Some(orientation));
                assert_eq!(world.bodies.mass_properties(body), Some(authored));
                if prepare {
                    let slot = body.index() as usize;
                    let view = world.bodies.solver_view_mut();
                    assert_vec_close(view.mass_props[slot].inv_mass, inverse_mass);
                    assert_vec_close(view.mass_props[slot].inv_inertia, inverse_inertia);
                    // Feed the same F*h and tau*h through XPBD's impulse
                    // algebra and compare both linear and angular responses.
                    let angular_tensor =
                        inv_inertia_world(view.mass_props[slot].inv_inertia, orientation);
                    let mut linear = Vec3::ZERO;
                    let mut angular = Vec3::ZERO;
                    apply_velocity_impulse(
                        &mut linear,
                        &mut angular,
                        inv_mass_world(view.mass_props[slot].inv_mass, orientation),
                        angular_tensor,
                        orientation * offset_local,
                        orientation * force_local * h,
                    );
                    angular += angular_tensor * (orientation * Vec3::new(2.0, 3.0, -5.0) * h);
                    assert_vec_close(linear, view.linear_velocities[slot]);
                    assert_vec_close(angular, view.angular_velocities[slot]);
                }
            }
        }
    }
}

#[test]
fn pending_force_added_mass_matches_xpbd_off_center_contact_projection() {
    let h = 0.125;
    for (height, fraction) in [(-4.0, 1.0), (0.0, 0.5), (4.0, 0.0)] {
        for orientation in [
            Quat::IDENTITY,
            Quat::from_rotation_z(core::f32::consts::FRAC_PI_4),
        ] {
            for normal in [Vec3::X, Vec3::Y, (Vec3::X + Vec3::Z).normalize()] {
                for arm in [Vec3::ZERO, Vec3::new(0.4, 0.2, -0.3)] {
                    let position = Vec3::Y * height;
                    let (mut world, body) = implicit_added_mass_world(position, orientation);
                    let authored = world.bodies.mass_properties(body).unwrap();
                    let point_a = position + arm;
                    let point_b = point_a - normal * 0.2;
                    let other = world.spawn(
                        BodyDesc::dynamic_at(point_b).with_material(PhysicsMaterial::FRICTIONLESS),
                    );
                    // F*h=n, tau*h=r x n. Thus the force-driven (dv,dw)
                    // are exactly the contact's unit-impulse response.
                    assert!(world.apply_force(body, normal / h, point_a));
                    assert_eq!(world.prepare_external_substep(h), 1);
                    assert!(world.external_forces.is_empty());
                    let (inverse_mass, inverse_inertia) = expected_implicit_inverse(fraction);
                    let linear = orientation * (inverse_mass * (orientation.conjugate() * normal));
                    let moment = arm.cross(normal);
                    let angular =
                        orientation * (inverse_inertia * (orientation.conjugate() * moment));
                    let actual_linear = world.bodies.linear_velocity(body).unwrap();
                    let actual_angular = world.bodies.angular_velocity(body).unwrap();
                    assert_vec_close(actual_linear, linear);
                    assert_vec_close(actual_angular, angular);
                    let w = normal.dot(actual_linear) + moment.dot(actual_angular);
                    let lambda = 0.2 / (w + 1.0);
                    let mut manifold = ContactManifold::new(body, other, normal);
                    manifold.push(ContactPoint::new(point_a, point_b, 0.2));
                    {
                        // Deliberately stage just the contact projection:
                        // prediction would rotate the body and change the
                        // tensor/lever arm being compared to the pending load.
                        let mut view = world.bodies.solver_view_mut();
                        let mut constraints = ContactConstraint::build(&view, &[manifold]);
                        assert_eq!(constraints.len(), 1);
                        solve_positions(&mut view, &mut constraints, &XpbdConfig::default(), h);
                        assert!((constraints[0].normal_lambda(0) - lambda).abs() < EPS);
                        assert_vec_close(
                            view.positions[body.index() as usize],
                            position - linear * lambda,
                        );
                        assert_vec_close(
                            view.positions[other.index() as usize],
                            point_b + normal * lambda,
                        );
                        let dphi = -angular * lambda;
                        let delta = Quat::from_xyzw(0.5 * dphi.x, 0.5 * dphi.y, 0.5 * dphi.z, 1.0)
                            .normalize();
                        let expected = delta * orientation;
                        let actual = view.orientations[body.index() as usize];
                        for (actual, expected) in
                            actual.to_array().into_iter().zip(expected.to_array())
                        {
                            assert!((actual - expected).abs() < EPS);
                        }
                    }
                    assert_eq!(world.bodies.mass_properties(body), Some(authored));
                }
            }
        }
    }
}

#[test]
fn xpbd_solver_view_receives_added_mass_without_mutating_authored_properties() {
    let mut world = PhysicsWorld::new(WorldConfig::with_gravity(Vec3::ZERO));
    let body = world.spawn(BodyDesc::dynamic_at(Vec3::new(0.0, -1.0, 0.0)));
    let authored = world
        .bodies
        .mass_properties(body)
        .expect("spawned body has authored mass");
    let mut model = AeroModel::default();
    model.added_mass = Some(AddedMass {
        fluid: FluidKind::Water,
        volume: 1.0,
        linear_coefficients: Vec3::ONE,
        angular_coefficients: Vec3::ONE,
        reference_radius: 1.0,
        ..AddedMass::default()
    });
    model.water_plane = Some(FluidPlane::horizontal(0.0, 1000.0));
    world
        .register_aero_body(body, AeroBodyConfig::new(model))
        .expect("live body must accept aerodynamic data");

    world.prepare_external_substep(1.0 / 60.0);
    let slot = body.index() as usize;
    let view = world.bodies.solver_view_mut();
    assert_eq!(view.mass_props[slot].inv_mass, Vec3::splat(1.0 / 1001.0));
    assert_eq!(view.mass_props[slot].inv_inertia, Vec3::splat(1.0 / 1001.0));
    assert_eq!(world.bodies.mass_properties(body), Some(authored));
}

#[test]
fn world_fixed_substep_consumes_thruster_and_advances_aero_time() {
    let mut world = PhysicsWorld::new(WorldConfig::with_gravity(Vec3::ZERO));
    let body = world.spawn(BodyDesc::dynamic_at(Vec3::ZERO));
    world
        .register_aero_body(
            body,
            AeroBodyConfig::default().with_thruster(Thruster::new(Vec3::ZERO, Vec3::X, 10.0)),
        )
        .expect("live body must accept aerodynamic data");

    let consumed = world.prepare_external_substep(0.1);
    assert_eq!(consumed, 1);
    assert!((world.bodies.linear_velocity(body).unwrap().x - 1.0).abs() < EPS);
    assert!((world.aero_time - 0.1).abs() < EPS);
    assert!(world.external_forces.is_empty());
}
