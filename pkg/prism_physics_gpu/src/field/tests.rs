use glam::Vec3;
use wgpu::naga::{
    front::wgsl,
    valid::{Capabilities, ValidationFlags, Validator},
};

use super::{
    cpu_sample_fields, Field, FieldBody, FieldEffect, FieldFalloff, FieldId, FieldRegistry,
    FieldShape, FieldTargetMask, GpuFieldSampler,
};
use crate::context::GpuContext;

const HIGH_MASK: u32 = 1 << 31;

fn field_registry() -> FieldRegistry {
    let mut registry = FieldRegistry::new();
    // Insert deliberately out of order: the GPU must upload and fold the
    // registry's stable FieldId order, not this construction order.
    registry.insert(Field {
        id: FieldId(50),
        shape: FieldShape::Global,
        falloff: FieldFalloff::Constant,
        effect: FieldEffect::Drag { coefficient: 0.75 },
        target_mask: FieldTargetMask(HIGH_MASK),
    });
    registry.insert(Field {
        id: FieldId(40),
        shape: FieldShape::Cone {
            apex: Vec3::ZERO,
            axis: Vec3::Z,
            angle: 0.5,
            length: 8.0,
        },
        falloff: FieldFalloff::InverseSquare { softening: 0.25 },
        effect: FieldEffect::Buoyancy {
            fluid_density: 1.25,
            gravity: Vec3::new(0.0, -9.8, 0.5),
            displaced_volume: 0.8,
        },
        target_mask: FieldTargetMask(0b0100),
    });
    registry.insert(Field {
        id: FieldId(30),
        shape: FieldShape::Aabb {
            min: Vec3::splat(-2.0),
            max: Vec3::splat(2.0),
        },
        falloff: FieldFalloff::Linear,
        effect: FieldEffect::Vortex {
            center: Vec3::ZERO,
            axis: Vec3::Y,
            strength: 3.0,
        },
        target_mask: FieldTargetMask(0b0010),
    });
    registry.insert(Field {
        id: FieldId(20),
        shape: FieldShape::Sphere {
            center: Vec3::X,
            radius: 3.0,
        },
        falloff: FieldFalloff::InverseSquare { softening: 0.125 },
        effect: FieldEffect::Radial {
            center: -Vec3::X,
            strength: 4.0,
        },
        target_mask: FieldTargetMask(0b0001),
    });
    registry.insert(Field {
        id: FieldId(10),
        shape: FieldShape::Global,
        falloff: FieldFalloff::Constant,
        effect: FieldEffect::ConstantForce(Vec3::new(1.0, -2.0, 0.5)),
        target_mask: FieldTargetMask::ALL,
    });
    registry
}

fn representative_bodies() -> Vec<FieldBody> {
    vec![
        FieldBody {
            position: Vec3::new(1.0, 0.5, 2.0),
            velocity: Vec3::new(2.0, -1.0, 0.5),
            mass: 2.0,
            target_mask: FieldTargetMask::ALL,
        },
        // Sphere boundary.
        FieldBody {
            position: Vec3::new(4.0, 0.0, 0.0),
            velocity: Vec3::ZERO,
            mass: 1.0,
            target_mask: FieldTargetMask(0b0001),
        },
        // AABB boundary.
        FieldBody {
            position: Vec3::new(2.0, 0.0, 0.0),
            velocity: Vec3::ONE,
            mass: 1.0,
            target_mask: FieldTargetMask(0b0010),
        },
        // Cone apex is inside only at radius zero.
        FieldBody {
            position: Vec3::new(0.5, 0.0, 1.75),
            velocity: Vec3::ZERO,
            mass: 1.0,
            target_mask: FieldTargetMask(0b0010),
        },
        // Both Z faces must have zero linear falloff, just like X/Y faces.
        FieldBody {
            position: Vec3::new(0.5, 0.0, 2.0),
            velocity: Vec3::ZERO,
            mass: 1.0,
            target_mask: FieldTargetMask(0b0010),
        },
        FieldBody {
            position: Vec3::new(0.5, 0.0, -2.0),
            velocity: Vec3::ZERO,
            mass: 1.0,
            target_mask: FieldTargetMask(0b0010),
        },
        // Cone apex is inside only at radius zero.
        FieldBody {
            position: Vec3::ZERO,
            velocity: Vec3::new(0.0, 2.0, 0.0),
            mass: 1.0,
            target_mask: FieldTargetMask(0b0100),
        },
        // Outside every bounded shape.
        FieldBody {
            position: Vec3::splat(12.0),
            velocity: Vec3::new(-1.0, 2.0, -3.0),
            mass: 3.0,
            target_mask: FieldTargetMask::ALL,
        },
        // Negative finite mass is valid but clamps drag to zero.
        FieldBody {
            position: Vec3::new(-1.0, 0.0, 0.0),
            velocity: Vec3::new(3.0, 4.0, 5.0),
            mass: -2.0,
            target_mask: FieldTargetMask(HIGH_MASK),
        },
        // Radial and vortex degeneracy at their shared center.
        FieldBody {
            position: Vec3::ZERO,
            velocity: Vec3::ZERO,
            mass: 1.0,
            target_mask: FieldTargetMask(0b0011),
        },
        // No field can match a zero mask, including FieldTargetMask::ALL.
        FieldBody {
            position: Vec3::ZERO,
            velocity: Vec3::ZERO,
            mass: 1.0,
            target_mask: FieldTargetMask::NONE,
        },
    ]
}

fn assert_sample_close(actual: super::FieldSample, expected: super::FieldSample) {
    let delta_force = actual.force - expected.force;
    let delta_torque = actual.torque - expected.torque;
    assert!(
        delta_force.length() <= 2.0e-5,
        "force mismatch: actual={actual:?}, expected={expected:?}, delta={delta_force:?}"
    );
    assert!(
        delta_torque.length() <= 2.0e-5,
        "torque mismatch: actual={actual:?}, expected={expected:?}, delta={delta_torque:?}"
    );
}

fn assert_matches_core(registry: &FieldRegistry, bodies: &[FieldBody]) {
    let cpu = cpu_sample_fields(registry, bodies);
    for (body, actual) in bodies.iter().zip(cpu) {
        let expected = registry.sample(body.position, body.velocity, body.mass, body.target_mask);
        assert_sample_close(actual, expected);
    }
}

#[test]
fn cpu_twin_matches_core_for_all_m11_shapes_effects_and_falloffs() {
    let registry = field_registry();
    assert_eq!(
        registry
            .fields()
            .iter()
            .map(|field| field.id)
            .collect::<Vec<_>>(),
        vec![
            FieldId(10),
            FieldId(20),
            FieldId(30),
            FieldId(40),
            FieldId(50)
        ]
    );
    assert_matches_core(&registry, &representative_bodies());
}

#[test]
fn cpu_twin_matches_core_rejection_and_non_finite_rules() {
    let mut registry = FieldRegistry::new();
    registry.insert(Field {
        id: FieldId(1),
        shape: FieldShape::Global,
        falloff: FieldFalloff::InverseSquare { softening: -1.0 },
        effect: FieldEffect::ConstantForce(Vec3::splat(99.0)),
        target_mask: FieldTargetMask::ALL,
    });
    registry.insert(Field {
        id: FieldId(2),
        shape: FieldShape::Sphere {
            center: Vec3::new(f32::NAN, 0.0, 0.0),
            radius: 1.0,
        },
        falloff: FieldFalloff::Constant,
        effect: FieldEffect::ConstantForce(Vec3::splat(98.0)),
        target_mask: FieldTargetMask::ALL,
    });
    registry.insert(Field {
        id: FieldId(3),
        shape: FieldShape::Aabb {
            min: Vec3::ZERO,
            max: Vec3::new(0.0, 2.0, 2.0),
        },
        falloff: FieldFalloff::Constant,
        effect: FieldEffect::ConstantForce(Vec3::splat(97.0)),
        target_mask: FieldTargetMask::ALL,
    });
    registry.insert(Field {
        id: FieldId(4),
        shape: FieldShape::Cone {
            apex: Vec3::ZERO,
            axis: Vec3::ZERO,
            angle: std::f32::consts::FRAC_PI_2,
            length: f32::INFINITY,
        },
        falloff: FieldFalloff::Constant,
        effect: FieldEffect::ConstantForce(Vec3::splat(96.0)),
        target_mask: FieldTargetMask::ALL,
    });
    registry.insert(Field {
        id: FieldId(5),
        shape: FieldShape::Global,
        falloff: FieldFalloff::Constant,
        effect: FieldEffect::ConstantForce(Vec3::new(2.0, -3.0, 5.0)),
        target_mask: FieldTargetMask::ALL,
    });

    let bodies = [
        FieldBody {
            position: Vec3::ZERO,
            velocity: Vec3::ZERO,
            mass: 1.0,
            target_mask: FieldTargetMask::ALL,
        },
        FieldBody {
            position: Vec3::new(f32::NAN, 0.0, 0.0),
            velocity: Vec3::ZERO,
            mass: 1.0,
            target_mask: FieldTargetMask::ALL,
        },
        FieldBody {
            position: Vec3::ZERO,
            velocity: Vec3::new(f32::INFINITY, 0.0, 0.0),
            mass: 1.0,
            target_mask: FieldTargetMask::ALL,
        },
        FieldBody {
            position: Vec3::ZERO,
            velocity: Vec3::ZERO,
            mass: f32::NEG_INFINITY,
            target_mask: FieldTargetMask::ALL,
        },
    ];
    assert_matches_core(&registry, &bodies);
}

#[test]
fn field_wgsl_parses_and_validates() {
    let module = wgsl::parse_str(include_str!("field.wgsl")).expect("field WGSL must parse");
    Validator::new(ValidationFlags::all(), Capabilities::empty())
        .validate(&module)
        .expect("field WGSL must validate");
}

fn required_headless() -> Option<GpuContext> {
    match GpuContext::try_headless() {
        Some(context) => Some(context),
        None if std::env::var("PRISM_REQUIRE_GPU").ok().as_deref() == Some("1") => {
            panic!("PRISM_REQUIRE_GPU=1 requires a usable wgpu adapter");
        }
        None => {
            eprintln!("no usable wgpu adapter; skipping field real-device parity");
            None
        }
    }
}

#[test]
fn real_gpu_matches_cpu_twin_and_core_reference() {
    let Some(context) = required_headless() else {
        return;
    };
    eprintln!("field GPU adapter: {:?}", context.adapter().get_info());

    let registry = field_registry();
    let mut bodies = representative_bodies();
    for index in 0..257 {
        let i = index as f32;
        bodies.push(FieldBody {
            position: Vec3::new(
                ((index % 17) as f32 - 8.0) * 0.37,
                ((index % 11) as f32 - 5.0) * 0.29,
                ((index % 13) as f32 - 6.0) * 0.41,
            ),
            velocity: Vec3::new(i * 0.01 - 1.0, 0.5 - i * 0.002, i * 0.003),
            mass: (index % 6) as f32 - 1.0,
            target_mask: FieldTargetMask(match index % 7 {
                0 => FieldTargetMask::NONE.0,
                1 => 0b0001,
                2 => 0b0010,
                3 => 0b0100,
                4 => HIGH_MASK,
                5 => FieldTargetMask::ALL.0,
                _ => 1 << 12,
            }),
        });
    }

    let expected_cpu = cpu_sample_fields(&registry, &bodies);
    let expected_core: Vec<_> = bodies
        .iter()
        .map(|body| registry.sample(body.position, body.velocity, body.mass, body.target_mask))
        .collect();
    let sampler = GpuFieldSampler::new(&context).expect("field pipeline initialization");
    assert_eq!(sampler.buffer_stats().unwrap().allocations, 0);
    let actual = sampler
        .sample(&context, &registry, &bodies)
        .expect("field GPU sampling must accept the batch");
    assert_eq!(actual.len(), bodies.len());
    for ((actual, expected_cpu), expected_core) in
        actual.into_iter().zip(expected_cpu).zip(expected_core)
    {
        assert_sample_close(actual, expected_cpu);
        assert_sample_close(actual, expected_core);
    }

    let initial_buffers = sampler.buffer_stats().unwrap();
    assert_eq!(initial_buffers.allocations, 1);
    let zero = sampler
        .sample(&context, &FieldRegistry::new(), &bodies[..3])
        .expect("empty field registry still dispatches the GPU kernel");
    assert!(zero.into_iter().all(|sample| sample == Default::default()));
    assert_eq!(sampler.buffer_stats().unwrap(), initial_buffers);

    // Grow, shrink and clear/repopulate fields on the same retained buffers.
    // Only a capacity increase can allocate another device buffer set.
    let expanded = vec![bodies[0]; initial_buffers.body_capacity + 1];
    let output = sampler.sample(&context, &registry, &expanded).unwrap();
    let expected = cpu_sample_fields(&registry, &expanded);
    for (sample, expected) in output.into_iter().zip(expected) {
        assert_sample_close(sample, expected);
    }
    let grown_buffers = sampler.buffer_stats().unwrap();
    assert_eq!(grown_buffers.allocations, 2);
    for count in [1, 64, 65, 3, 0, 257] {
        let batch = &bodies[..count];
        let samples = sampler.sample(&context, &registry, batch).unwrap();
        assert_eq!(samples.len(), count);
        for (sample, expected) in samples.into_iter().zip(cpu_sample_fields(&registry, batch)) {
            assert_sample_close(sample, expected);
        }
        assert_eq!(sampler.buffer_stats().unwrap(), grown_buffers);
    }
    let mut expanded_fields = registry.clone();
    for index in 0..grown_buffers.field_capacity {
        expanded_fields.insert(Field::constant(FieldId(100 + index as u64), Vec3::X));
    }
    let samples = sampler
        .sample(&context, &expanded_fields, &bodies[..1])
        .unwrap();
    assert_sample_close(
        samples[0],
        cpu_sample_fields(&expanded_fields, &bodies[..1])[0],
    );
    assert_eq!(sampler.buffer_stats().unwrap().allocations, 3);
}

#[test]
fn real_gpu_field_sampler_rejects_an_unrelated_device_before_allocating() {
    let Some(context) = required_headless() else {
        return;
    };
    let Some(other) = required_headless() else {
        return;
    };
    let sampler = GpuFieldSampler::new(&context).expect("field pipeline initialization");
    assert_eq!(
        sampler.sample(&other, &field_registry(), &representative_bodies()),
        Err(super::FieldGpuError::DeviceMismatch)
    );
    assert_eq!(sampler.buffer_stats().unwrap().allocations, 0);
    assert_eq!(
        sampler.sample(&other, &FieldRegistry::new(), &[]),
        Err(super::FieldGpuError::DeviceMismatch)
    );
    // A rejected context cannot poison or mutate the sampler's workspace.
    let registry = field_registry();
    let bodies = representative_bodies();
    let output = sampler.sample(&context, &registry, &bodies).unwrap();
    for (sample, expected) in output
        .into_iter()
        .zip(cpu_sample_fields(&registry, &bodies))
    {
        assert_sample_close(sample, expected);
    }
    let allocated = sampler.buffer_stats().unwrap();
    assert_eq!(allocated.allocations, 1);
    assert_eq!(
        sampler.sample(&other, &registry, &bodies),
        Err(super::FieldGpuError::DeviceMismatch)
    );
    assert_eq!(sampler.buffer_stats().unwrap(), allocated);
}

#[test]
fn real_gpu_field_runtime_advances_live_xpbd_world_with_contacts_and_rebinding() {
    use super::GpuFieldRuntime;
    use prism_physics_core::{
        BodyDesc, ColliderShape, PhysicsWorld, Solver, WorldConfig, XpbdSolver,
    };

    let Some(context) = required_headless() else {
        return;
    };
    eprintln!(
        "XPBD field runtime adapter: {:?}",
        context.adapter().get_info()
    );
    let mut runtime = GpuFieldRuntime::new(context).expect("field pipeline initialization");
    let mut world = PhysicsWorld::new(WorldConfig::default());
    let floor = world.shapes.insert(ColliderShape::Plane {
        normal: Vec3::Y,
        offset: 0.0,
    });
    world.spawn(BodyDesc::static_at(Vec3::ZERO).with_collider(floor));
    let shape = world.shapes.insert(ColliderShape::Sphere { radius: 0.5 });
    let mut handles: Vec<_> = (0..5)
        .map(|index| {
            world.spawn(
                BodyDesc::dynamic_at(Vec3::new(
                    index as f32 * 1.5,
                    0.45 + index as f32 * 0.05,
                    0.0,
                ))
                .with_collider(shape)
                .with_linear_velocity(Vec3::new(0.5, -0.1, 0.2))
                .with_damping(0.0, 0.0),
            )
        })
        .collect();
    world.fields = field_registry();
    let mut reference = world.clone();
    let mut solver = XpbdSolver::default();
    for frame in 0..12 {
        if frame == 4 {
            // Reuse the same slot at a new generation, with a new body pose.
            let old = handles[2];
            assert!(world.bodies.remove(old));
            assert!(reference.bodies.remove(old));
            let desc = BodyDesc::dynamic_at(Vec3::new(3.0, 2.0, 0.0)).with_collider(shape);
            handles[2] = world.spawn(desc);
            assert_eq!(reference.spawn(desc), handles[2]);
            assert_ne!(old, handles[2]);
        }
        if frame == 8 {
            // Device buffers must not retain removed fields from earlier steps.
            world.fields = FieldRegistry::new();
            reference.fields = FieldRegistry::new();
        }
        for state in [&mut world, &mut reference] {
            assert!(state.apply_force(
                handles[0],
                Vec3::Z,
                state.bodies.position(handles[0]).unwrap()
            ));
        }
        solver.step(&mut reference, 1.0 / 60.0, 3);
        solver
            .step_with_field_sampler(&mut world, 1.0 / 60.0, 3, &mut runtime)
            .unwrap();
        for &body in &handles {
            let position = world.bodies.position(body).unwrap();
            let velocity = world.bodies.linear_velocity(body).unwrap();
            assert!(
                (position - reference.bodies.position(body).unwrap()).length() < 2.0e-4,
                "frame {frame}, body {body:?}, position {position:?}"
            );
            assert!(
                (velocity - reference.bodies.linear_velocity(body).unwrap()).length() < 2.0e-3,
                "frame {frame}, body {body:?}, velocity {velocity:?}"
            );
            assert!(
                position.y >= 0.499,
                "contact must prevent floor penetration"
            );
        }
        assert!(world.external_forces.is_empty());
    }
}
