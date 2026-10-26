use super::*;
use prism_render_architecture::particle::{
    collision::{
        resolve_particle_collision, sd_aabb, sd_capsule, sd_sphere, Collider, ParticleState,
        ResponseParams,
    },
    forces::{Falloff, Force, ForceEntry, ForceField},
    simulation::{integrate, Kinematics},
    sort_cull::Plane,
    IntegratorKind, IterationDomain, Vec3,
};
use prism_render_driver::Instance as _;
use prism_render_driver_wgpu::{require_hardware_gpu, HardwareDeviceError, WgpuInstance};

#[test]
fn null_rhi_records_gpu_simulation_in_authoritative_stage_order() {
    let instance = prism_render_driver_null::NullInstance::new();
    let adapter = instance
        .request_adapter(&Default::default())
        .expect("null adapter");
    let (device, _queue) = instance
        .create_device(&adapter, &Default::default())
        .expect("null device");
    let stage = GpuSimulationStage::new(&device, 128, 64);
    let mut graph =
        prism_render_graph::RenderGraph::new(prism_render_driver::Extent3d::new_2d(1, 1));
    let outputs = stage
        .append_to_graph(&mut graph)
        .expect("simulation graph admission");
    assert!(outputs.particles.version() > 0);
    let plan = graph.compile().expect("simulation graph compiles");
    assert_eq!(plan.order, [0, 1, 2, 3, 4]);
    assert_eq!(plan.declared_pass_count, 5);
    assert!(plan.barrier_count() >= 4);
    stage.destroy(&device);
}

#[test]
fn force_abi_preserves_order_and_exact_record_stride() {
    let mut field = ForceField::new();
    field
        .push(ForceEntry::new(Force::Gravity {
            acceleration: Vec3::new(0.0, -9.8, 0.0),
        }))
        .push(ForceEntry::new(Force::SpringDamper {
            anchor: Vec3::new(1.0, 2.0, 3.0),
            stiffness: 4.0,
            damping: 0.5,
        }));
    let encoded = encode_force_buffer(field.entries());
    assert_eq!(
        encoded.len(),
        MAX_FORCE_ENTRIES as usize * GPU_FORCE_STRIDE as usize
    );
    assert_eq!(u32_at(&encoded[0..4]), FORCE_GRAVITY);
    assert_eq!(f32_at(&encoded, 32 + 4), -9.8);
    let second = GPU_FORCE_STRIDE as usize;
    assert_eq!(u32_at(&encoded[second..second + 4]), FORCE_SPRING);
    assert_eq!(f32_at(&encoded, second + 32), 1.0);
    assert_eq!(f32_at(&encoded, second + 32 + 16), 4.0);
}

#[test]
fn turbulence_abi_preserves_seed_and_octave_integer_bits() {
    // Values above f32's exact integer range must not be rounded by the ABI.
    let bytes = encode_force_buffer(&[ForceEntry::new(Force::Turbulence {
        seed: 0xffff_fffd,
        frequency: 0.25,
        amplitude: 0.5,
        octaves: 0x0100_0001,
        frequency_multiplier: 2.0,
        amplitude_multiplier: 0.5,
        epsilon: 0.125,
    })]);
    assert_eq!(u32_at(&bytes[48..52]), 0xffff_fffd);
    assert_eq!(u32_at(&bytes[52..56]), 0x0100_0001);
    assert_eq!(f32_at(&bytes, 56), 0.125);
}

#[test]
fn collider_abi_keeps_gpu_record_stride_and_payload_offsets() {
    let bytes = encode_collider_buffer(&[Collider::Sphere(Sphere {
        center: Vec3::new(2.0, 3.0, 4.0),
        radius: 5.0,
    })]);
    assert_eq!(
        bytes.len(),
        MAX_COLLIDERS as usize * GPU_COLLIDER_STRIDE as usize
    );
    assert_eq!(u32_at(&bytes[0..4]), COLLIDER_SPHERE);
    assert_eq!(f32_at(&bytes, 16), 2.0);
    assert_eq!(f32_at(&bytes, 28), 5.0);
}

#[test]
fn stage_dispatch_maps_cpu_domains_without_fabricating_other_kernels() {
    let counts = StageCounts {
        live_particles: 129,
        events: 65,
        neighbor_cells: 8,
        ..StageCounts::default()
    };
    let plan = gpu_stage_dispatch_plan(counts, 129);
    assert_eq!(plan[2].domain, IterationDomain::PerParticle);
    assert_eq!(plan[2].workgroups, 3);
    assert_eq!(plan[4].domain, IterationDomain::PerEvent);
    assert_eq!(plan[4].invocations, 65);
    assert_eq!(plan[4].workgroups, 2);
    assert_eq!(workgroup_count(0, SIMULATION_WORKGROUP_SIZE), 0);
}

#[test]
fn shader_has_all_recorded_entry_points_and_no_placeholder_tokens() {
    let shader = include_str!("shaders/particle_sim_stages.wgsl");
    for entry in ["reset", "spawn", "simulate", "compact", "event_scatter"] {
        assert!(shader.contains(&format!("fn {entry}(")), "missing {entry}");
    }
    assert!(!shader.contains("TODO"));
    assert!(!shader.contains("placeholder"));
}

#[test]
fn particle_wgsl_shaders_parse_and_validate_without_a_gpu() {
    let shaders = [
        ("particle.wgsl", include_str!("shaders/particle.wgsl")),
        (
            "particle_draw.wgsl",
            include_str!("shaders/particle_draw.wgsl"),
        ),
        (
            "particle_renderer_args.wgsl",
            include_str!("shaders/particle_renderer_args.wgsl"),
        ),
        (
            "particle_renderer_auxiliary.wgsl",
            include_str!("shaders/particle_renderer_auxiliary.wgsl"),
        ),
        (
            "particle_renderer_compact.wgsl",
            include_str!("shaders/particle_renderer_compact.wgsl"),
        ),
        (
            "particle_renderer_cull.wgsl",
            include_str!("shaders/particle_renderer_cull.wgsl"),
        ),
        (
            "particle_renderer_sort.wgsl",
            include_str!("shaders/particle_renderer_sort.wgsl"),
        ),
        (
            "particle_renderer_surface.wgsl",
            include_str!("shaders/particle_renderer_surface.wgsl"),
        ),
        (
            "particle_sim_stages.wgsl",
            include_str!("shaders/particle_sim_stages.wgsl"),
        ),
    ];

    for (name, source) in shaders {
        let module = naga::front::wgsl::parse_str(source).unwrap_or_else(|error| {
            panic!(
                "{name} WGSL parse failed:\n{}",
                error.emit_to_string(source)
            )
        });
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|error| panic!("{name} WGSL validation failed: {error:?}"));
    }
}

#[test]
fn physical_gpu_executes_force_lifetime_collision_spawn_and_event_compaction() {
    let (device, queue) = match WgpuInstance::new().request_default_hardware_device() {
        Ok(pair) => pair,
        Err(HardwareDeviceError::NoAdapter) => {
            assert!(
                !require_hardware_gpu(),
                "PRISM_REQUIRE_GPU=1 but no adapter was found"
            );
            eprintln!("SKIPPED particle simulation GPU test: no adapter");
            return;
        }
        Err(HardwareDeviceError::NonHardware(evidence)) => {
            assert!(
                !require_hardware_gpu(),
                "PRISM_REQUIRE_GPU=1 requires a physical GPU: {evidence}"
            );
            eprintln!("SKIPPED particle simulation GPU test: software/virtual adapter");
            return;
        }
        Err(HardwareDeviceError::Device(error)) => panic!("GPU device creation failed: {error}"),
    };

    let stage = GpuSimulationStage::new(&device, 4, 16).expect("simulation resources");
    let pool = [
        GpuParticle::live([0.0, 0.05, 0.0], [0.0, -1.0, 0.0], 2.0, 0.1, 10, 0),
        GpuParticle::live([0.0, 2.0, 0.0], [0.0; 3], 0.05, 0.0, 11, 0),
        GpuParticle::dead(12),
        GpuParticle::dead(13),
    ];
    let forces = [ForceEntry::new(Force::Gravity {
        acceleration: Vec3::new(0.0, -1.0, 0.0),
    })];
    let colliders = [Collider::HalfSpace(Plane {
        normal: Vec3::new(0.0, 1.0, 0.0),
        d: 0.0,
    })];
    stage.initialize_particles(&queue, &pool);
    stage.upload_inputs(
        &queue,
        &forces,
        &colliders,
        GpuSimulationParams {
            dt: 0.1,
            restitution: 1.0,
            spawn_count: 1,
            spawn_position: [3.0, 1.0, 0.0],
            spawn_velocity: [0.0; 3],
            spawn_lifetime: 1.0,
            condition_x: 2.0,
            condition_enabled: true,
            ..GpuSimulationParams::default()
        },
    );
    let output = stage.run(&queue).expect("GPU simulation readback");
    assert_eq!(output.particles.len(), 4);
    assert!(
        output.particles[0].position[1] >= output.particles[0].radius - 1.0e-5,
        "half-space contact did not push the particle out: {:?}",
        output.particles[0]
    );
    assert!(output.particles[0].velocity[1] > 0.0);
    assert!(!output.particles[1].alive);
    assert_eq!(output.active_indices.len(), 2);
    assert!(output.particles[2].alive || output.particles[3].alive);
    assert!(output.control[4] >= 1, "death event should be counted");
    assert!(output.control[5] >= 1, "collision event should be counted");
    assert!(output.control[3] >= 1, "spawn event should be counted");
    assert!(output.control[6] >= 1, "condition event should be counted");
    assert_eq!(output.compact_events.len(), output.control[8] as usize);
    for kind in 0..4 {
        assert!(output.compact_events.iter().any(|event| event.kind == kind));
    }
    stage.destroy(&device);
}

#[test]
fn physical_gpu_integrators_match_cpu_reference() {
    let (device, queue) = match WgpuInstance::new().request_default_hardware_device() {
        Ok(pair) => pair,
        Err(HardwareDeviceError::NoAdapter) => {
            assert!(
                !require_hardware_gpu(),
                "PRISM_REQUIRE_GPU=1 but no adapter was found"
            );
            eprintln!("SKIPPED particle CPU/GPU parity test: no adapter");
            return;
        }
        Err(HardwareDeviceError::NonHardware(evidence)) => {
            assert!(
                !require_hardware_gpu(),
                "PRISM_REQUIRE_GPU=1 requires a physical GPU: {evidence}"
            );
            eprintln!("SKIPPED particle CPU/GPU parity test: software/virtual adapter");
            return;
        }
        Err(HardwareDeviceError::Device(error)) => panic!("GPU device creation failed: {error}"),
    };

    let initial = [
        GpuParticle::live([1.25, -0.5, 2.0], [3.0, -1.5, 0.25], 10.0, 0.0, 101, 7),
        GpuParticle::live([-4.0, 8.0, 0.125], [-0.75, 2.5, -1.0], 10.0, 0.0, 102, 9),
    ];
    let mut forces = ForceField::new();
    forces
        .add(Force::Gravity {
            acceleration: Vec3::new(0.375, -9.25, 1.125),
        })
        .add(Force::SpringDamper {
            anchor: Vec3::new(0.5, 1.0, -0.75),
            stiffness: 0.625,
            damping: 0.125,
        });
    let dt = 1.0 / 37.0;

    for integrator in [
        IntegratorKind::SemiImplicitEuler,
        IntegratorKind::Verlet,
        IntegratorKind::Rk2,
    ] {
        let stage = GpuSimulationStage::new(&device, initial.len() as u32, 8)
            .expect("simulation resources");
        stage.initialize_particles(&queue, &initial);
        stage.upload_inputs(
            &queue,
            forces.entries(),
            &[],
            GpuSimulationParams {
                dt,
                integrator,
                ..GpuSimulationParams::default()
            },
        );

        let mut reference = initial.map(|particle| {
            Kinematics::new(
                Vec3::new(
                    particle.position[0],
                    particle.position[1],
                    particle.position[2],
                ),
                Vec3::new(
                    particle.velocity[0],
                    particle.velocity[1],
                    particle.velocity[2],
                ),
            )
        });
        // Advance the resident pool repeatedly without a CPU snapshot upload.
        // A position/velocity-dependent force also checks RK2's midpoint sample.
        for state in &mut reference {
            *state = integrate(integrator, *state, dt, |p, v| forces.accumulate(p, v));
        }
        let mut output = stage.run(&queue).expect("GPU simulation readback");
        for _ in 1..8 {
            for state in &mut reference {
                *state = integrate(integrator, *state, dt, |p, v| forces.accumulate(p, v));
            }
            output = stage.run(&queue).expect("GPU simulation readback");
        }
        assert_eq!(output.particles.len(), initial.len());
        assert_eq!(output.active_indices.len(), initial.len());

        for (index, (input, actual)) in initial.iter().zip(&output.particles).enumerate() {
            let expected = reference[index];
            assert!(actual.alive, "{integrator:?} killed particle {index}");
            assert_eq!(actual.particle_id, input.particle_id);
            assert_eq!(actual.emitter_id, input.emitter_id);
            assert_vec3_close(
                actual.position,
                [
                    expected.position.x,
                    expected.position.y,
                    expected.position.z,
                ],
                integrator,
                index,
                "position",
            );
            assert_vec3_close(
                actual.velocity,
                [
                    expected.velocity.x,
                    expected.velocity.y,
                    expected.velocity.z,
                ],
                integrator,
                index,
                "velocity",
            );
            assert!(actual.age.is_finite());
            assert!((actual.age - 8.0 * dt).abs() <= 1.0e-6);
        }
        stage.destroy(&device);
    }
}

#[test]
fn physical_gpu_force_library_matches_cpu_reference() {
    let (device, queue) = match WgpuInstance::new().request_default_hardware_device() {
        Ok(pair) => pair,
        Err(HardwareDeviceError::NoAdapter) => {
            assert!(
                !require_hardware_gpu(),
                "PRISM_REQUIRE_GPU=1 but no adapter was found"
            );
            eprintln!("SKIPPED force-library parity: no hardware adapter");
            return;
        }
        Err(HardwareDeviceError::NonHardware(evidence)) => {
            assert!(!require_hardware_gpu(), "physical GPU required: {evidence}");
            eprintln!("SKIPPED force-library parity: non-hardware adapter");
            return;
        }
        Err(HardwareDeviceError::Device(error)) => panic!("GPU device creation failed: {error}"),
    };
    let center = Vec3::new(0.25, -0.5, 0.125);
    let axis = Vec3::new(0.25, 2.0, -0.125);
    let gravity = Force::Gravity {
        acceleration: Vec3::new(0.125, -0.25, 0.5),
    };
    let dt = 1.0 / 256.0;
    let mut cases: Vec<(String, Vec<ForceEntry>, f32)> = Vec::new();
    let mut add = |label: &str, force: Force| {
        cases.push((label.into(), vec![ForceEntry::new(force)], dt));
    };
    add("gravity", gravity);
    add(
        "linear-drag",
        Force::LinearDrag {
            coefficient: 0.75,
            dt,
        },
    );
    add(
        "quadratic-drag",
        Force::QuadraticDrag { coefficient: 0.125 },
    );
    add(
        "wind",
        Force::Wind {
            velocity: center,
            coefficient: 0.75,
        },
    );
    add(
        "vortex",
        Force::Vortex {
            center,
            axis,
            strength: 0.25,
        },
    );
    add(
        "orbital",
        Force::Orbital {
            center,
            axis,
            target_radius: 1.5,
            tangential_strength: -0.25,
            radial_stiffness: 0.125,
        },
    );
    add(
        "gravity-well",
        Force::GravityWell {
            center,
            strength: 0.125,
            softening: 0.5,
        },
    );
    add(
        "near-center-well",
        Force::GravityWell {
            center: Vec3::ZERO,
            strength: 1.0e-9,
            softening: -1.0,
        },
    );
    add(
        "turbulence",
        Force::Turbulence {
            seed: 0xffff_fffd,
            frequency: 0.25,
            amplitude: 0.5,
            octaves: 4,
            frequency_multiplier: 2.0,
            amplitude_multiplier: 0.5,
            epsilon: 0.125,
        },
    );
    add(
        "zero-octaves",
        Force::Turbulence {
            seed: 7,
            frequency: 0.25,
            amplitude: 0.5,
            octaves: 0,
            frequency_multiplier: 2.0,
            amplitude_multiplier: 0.5,
            epsilon: 0.125,
        },
    );
    add(
        "spring",
        Force::SpringDamper {
            anchor: center,
            stiffness: 0.625,
            damping: 0.125,
        },
    );
    add(
        "clamped-spring",
        Force::SpringDamper {
            anchor: center,
            stiffness: -1.0,
            damping: -1.0,
        },
    );
    add(
        "negative-quadratic-drag",
        Force::QuadraticDrag { coefficient: -1.0 },
    );
    add(
        "zero-drag-dt",
        Force::LinearDrag {
            coefficient: 2.0,
            dt: 0.0,
        },
    );
    add(
        "negative-wind",
        Force::Wind {
            velocity: center,
            coefficient: -1.0,
        },
    );
    add(
        "degenerate-vortex",
        Force::Vortex {
            center,
            axis: Vec3::ZERO,
            strength: 0.25,
        },
    );
    add(
        "degenerate-orbit",
        Force::Orbital {
            center,
            axis: Vec3::ZERO,
            target_radius: 1.5,
            tangential_strength: 0.25,
            radial_stiffness: 0.125,
        },
    );
    add(
        "clamped-orbit",
        Force::Orbital {
            center,
            axis,
            target_radius: -1.0,
            tangential_strength: 0.25,
            radial_stiffness: -1.0,
        },
    );
    add(
        "sanitized-gravity",
        Force::Gravity {
            acceleration: Vec3::new(f32::NAN, -0.25, f32::INFINITY),
        },
    );
    add(
        "sanitized-wind",
        Force::Wind {
            velocity: Vec3::new(f32::NAN, 0.25, f32::INFINITY),
            coefficient: 0.5,
        },
    );
    for falloff in [
        Falloff::Constant,
        Falloff::Linear,
        Falloff::Smoothstep,
        Falloff::InverseSquare,
    ] {
        add(
            &format!("point-{falloff:?}"),
            Force::PointAttractor {
                point: center,
                strength: -0.5,
                radius: 4.0,
                softening: 0.125,
                falloff,
            },
        );
        add(
            &format!("line-{falloff:?}"),
            Force::LineAttractor {
                point: center,
                direction: axis,
                strength: 0.5,
                radius: 4.0,
                softening: 0.125,
                falloff,
            },
        );
        add(
            &format!("radial-{falloff:?}"),
            Force::Radial {
                center,
                strength: 0.5,
                radius: 4.0,
                softening: 0.125,
                falloff,
            },
        );
    }
    add(
        "degenerate-line",
        Force::LineAttractor {
            point: center,
            direction: Vec3::ZERO,
            strength: 0.5,
            radius: 0.0,
            softening: -1.0,
            falloff: Falloff::Constant,
        },
    );
    add(
        "unbounded-point",
        Force::PointAttractor {
            point: center,
            strength: 0.5,
            radius: -1.0,
            softening: f32::NAN,
            falloff: Falloff::Linear,
        },
    );
    cases.push((
        "tiny-drag-dt".into(),
        vec![ForceEntry::new(Force::LinearDrag {
            coefficient: 2.0e6,
            dt: 1.0e-8,
        })],
        1.0e-8,
    ));
    cases.push((
        "weighted-gates".into(),
        vec![
            ForceEntry::new(gravity)
                .with_weight(0.375)
                .with_gate(Vec3::ZERO, 4.0),
            ForceEntry::new(gravity).with_gate(Vec3::ZERO, 0.25),
            ForceEntry::new(gravity).with_enabled(false),
            ForceEntry::new(gravity).with_weight(-1.0),
            ForceEntry::new(gravity).with_weight(f32::NAN),
            ForceEntry::new(gravity).with_gate(Vec3::new(f32::NAN, 0.0, 0.0), 0.0),
            ForceEntry::new(gravity).with_gate(Vec3::ZERO, f32::INFINITY),
            ForceEntry::new(gravity).with_gate(Vec3::ZERO, f32::NAN),
        ],
        dt,
    ));
    cases.push((
        "invalid-entry-preserves-valid-sum".into(),
        vec![
            ForceEntry::new(gravity),
            ForceEntry::new(Force::SpringDamper {
                anchor: Vec3::new(f32::MAX, 0.0, 0.0),
                stiffness: f32::MAX,
                damping: 0.0,
            }),
            ForceEntry::new(Force::Vortex {
                center,
                axis,
                strength: f32::INFINITY,
            }),
            ForceEntry::new(gravity).with_weight(0.5),
        ],
        dt,
    ));

    let initial = [
        GpuParticle::live([1.25, -0.5, 2.0], [0.75, -0.25, 0.125], 10.0, 0.0, 101, 7),
        GpuParticle::live([-4.0, 8.0, 0.125], [-0.5, 0.125, -0.75], 10.0, 0.0, 102, 9),
        GpuParticle::live([1.0e-4, 0.0, 0.0], [0.0; 3], 10.0, 0.0, 103, 7),
        GpuParticle::live([0.25, -0.5, 0.125], [0.0; 3], 10.0, 0.0, 104, 9),
    ];
    let stage =
        GpuSimulationStage::new(&device, initial.len() as u32, 16).expect("simulation resources");
    for (label, entries, step_dt) in cases {
        let mut field = ForceField::new();
        for entry in entries {
            field.push(entry);
        }
        for integrator in [
            IntegratorKind::SemiImplicitEuler,
            IntegratorKind::Verlet,
            IntegratorKind::Rk2,
        ] {
            stage.initialize_particles(&queue, &initial);
            stage.upload_inputs(
                &queue,
                field.entries(),
                &[],
                GpuSimulationParams {
                    dt: step_dt,
                    integrator,
                    ..Default::default()
                },
            );
            let mut reference = initial.map(|p| {
                Kinematics::new(
                    Vec3::new(p.position[0], p.position[1], p.position[2]),
                    Vec3::new(p.velocity[0], p.velocity[1], p.velocity[2]),
                )
            });
            for _ in 0..3 {
                for state in &mut reference {
                    *state = integrate(integrator, *state, step_dt, |p, v| field.accumulate(p, v));
                }
                let actual = stage.run(&queue).expect("GPU simulation readback");
                assert_eq!(
                    actual.active_indices.len(),
                    initial.len(),
                    "{label} {integrator:?}"
                );
                for (index, (expected, actual)) in
                    reference.iter().zip(&actual.particles).enumerate()
                {
                    assert_eq!(actual.particle_id, initial[index].particle_id);
                    assert_eq!(actual.emitter_id, initial[index].emitter_id);
                    assert_vec3_close(
                        actual.position,
                        [
                            expected.position.x,
                            expected.position.y,
                            expected.position.z,
                        ],
                        integrator,
                        index,
                        &format!("{label} position"),
                    );
                    assert_vec3_close(
                        actual.velocity,
                        [
                            expected.velocity.x,
                            expected.velocity.y,
                            expected.velocity.z,
                        ],
                        integrator,
                        index,
                        &format!("{label} velocity"),
                    );
                }
            }
        }
    }
    stage.destroy(&device);
}

#[test]
fn physical_gpu_collision_library_matches_cpu_reference() {
    let (device, queue) = match WgpuInstance::new().request_default_hardware_device() {
        Ok(pair) => pair,
        Err(HardwareDeviceError::NoAdapter | HardwareDeviceError::NonHardware(_)) => {
            assert!(
                !require_hardware_gpu(),
                "physical GPU required for collision parity"
            );
            eprintln!("SKIPPED collision parity: no hardware adapter");
            return;
        }
        Err(HardwareDeviceError::Device(error)) => panic!("GPU device creation failed: {error}"),
    };
    let plane = |normal, d| Collider::HalfSpace(Plane::new(normal, d));
    let sphere = |center, radius| Collider::Sphere(Sphere { center, radius });
    let capsule = |a, b, radius| Collider::Capsule(Capsule { a, b, radius });
    let sdf = |gradient, distance| Collider::Sdf(SdfSample { gradient, distance });
    let aabb = |min, max| Collider::Box(Aabb { min, max });
    let x = Vec3::new(1.0, 0.0, 0.0);
    let y = Vec3::new(0.0, 1.0, 0.0);
    let invalid = Vec3::new(f32::NAN, 0.0, f32::INFINITY);
    let mut cases = vec![
        ("unit-plane", vec![plane(y, 0.0)]),
        ("non-unit-offset-plane", vec![plane(y.scale(4.0), -2.0)]),
        (
            "oblique-plane",
            vec![plane(Vec3::new(2.0, 3.0, -1.0), 0.25)],
        ),
        ("degenerate-plane", vec![plane(Vec3::ZERO, -2.0)]),
        (
            "invalid-plane",
            vec![plane(invalid, 0.0), plane(y, f32::NAN)],
        ),
        ("overflow-plane", vec![plane(Vec3::splat(f32::MAX), 0.0)]),
        ("sphere", vec![sphere(Vec3::ZERO, 1.0)]),
        ("negative-sphere-radius", vec![sphere(Vec3::ZERO, -1.0)]),
        (
            "invalid-sphere",
            vec![sphere(invalid, 1.0), sphere(Vec3::ZERO, f32::INFINITY)],
        ),
        ("box", vec![aabb(Vec3::splat(-1.0), Vec3::splat(1.0))]),
        ("degenerate-box", vec![aabb(Vec3::ZERO, Vec3::ZERO)]),
        (
            "invalid-box",
            vec![
                aabb(Vec3::splat(1.0), Vec3::splat(-1.0)),
                aabb(invalid, Vec3::splat(1.0)),
            ],
        ),
        ("capsule", vec![capsule(x.scale(-1.0), x, 0.5)]),
        (
            "degenerate-capsule",
            vec![capsule(Vec3::ZERO, Vec3::ZERO, 0.5)],
        ),
        (
            "tiny-capsule",
            vec![capsule(Vec3::ZERO, x.scale(0.5e-6), 0.5)],
        ),
        (
            "negative-capsule-radius",
            vec![capsule(x.scale(-1.0), x, -0.5)],
        ),
        (
            "invalid-capsule",
            vec![capsule(invalid, x, 1.0), capsule(x, y, f32::NAN)],
        ),
        ("sdf", vec![sdf(Vec3::new(2.0, 3.0, -1.0), -0.25)]),
        ("zero-gradient-sdf", vec![sdf(Vec3::ZERO, -0.25)]),
        (
            "invalid-sdf",
            vec![sdf(invalid, -0.25), sdf(y, f32::INFINITY)],
        ),
        ("empty", vec![]),
    ];
    let mut equal = vec![plane(x, 0.0), plane(y, 0.0)];
    cases.push(("equal-depth-xy", equal.clone()));
    equal.reverse();
    cases.push(("equal-depth-yx", equal));
    cases.push(("shallow-positive-contact", vec![sdf(y, -0.25e-6)]));
    cases.push((
        "mixed-deepest",
        vec![plane(y, 0.0), sphere(Vec3::ZERO, 0.25), sdf(x, -0.5)],
    ));

    let positions = [
        [0.0; 3],
        [0.125, -0.25, 0.5],
        [1.125, 0.25, -0.125],
        [0.25, 1.125, -0.5],
        [1.125, 1.125, 0.0],
        [3.0, 4.0, -2.0],
        [-0.75, -0.5, -0.25],
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [f32::NAN, -0.25, f32::INFINITY],
        [0.0, -0.25, 0.0],
    ];
    let mut initial: Vec<_> = positions
        .into_iter()
        .enumerate()
        .map(|(i, pos)| {
            let velocity = if i % 2 == 0 {
                [0.5, -2.0, 0.125]
            } else {
                [-0.25, 1.0, -0.5]
            };
            GpuParticle::live(pos, velocity, 10.0, 0.25, i as u32 + 1, 7)
        })
        .collect();
    initial[8].radius = -1.0;
    initial[9].radius = f32::NAN;
    initial[10].radius = f32::INFINITY;
    initial[11].radius = 0.0;
    initial[13].velocity = [f32::NAN, -2.0, f32::INFINITY];
    let stage =
        GpuSimulationStage::new(&device, initial.len() as u32, 64).expect("simulation resources");
    for (label, colliders) in cases {
        for (restitution, friction) in [(0.625, 0.25), (2.0, -1.0), (f32::NAN, f32::INFINITY)] {
            stage.initialize_particles(&queue, &initial);
            stage.upload_inputs(
                &queue,
                &[],
                &colliders,
                GpuSimulationParams {
                    dt: 0.0,
                    restitution,
                    friction,
                    ..Default::default()
                },
            );
            let mut reference: Vec<_> = initial
                .iter()
                .map(|p| {
                    ParticleState::new(
                        Vec3::new(p.position[0], p.position[1], p.position[2]),
                        Vec3::new(p.velocity[0], p.velocity[1], p.velocity[2]),
                        p.radius,
                    )
                })
                .collect();
            for frame in 0..3 {
                let entering_states = reference.clone();
                let mut hits = Vec::new();
                for (i, state) in reference.iter_mut().enumerate() {
                    let response = resolve_particle_collision(
                        *state,
                        &colliders,
                        ResponseParams {
                            restitution,
                            friction,
                        },
                    );
                    state.pos = response.new_pos;
                    state.vel = response.new_vel;
                    if response.hit {
                        hits.push(i as u32);
                    }
                }
                let actual = stage.run(&queue).expect("GPU simulation readback");
                assert_eq!(actual.active_indices.len(), initial.len(), "{label}");
                let mut event_indices: Vec<_> = actual
                    .compact_events
                    .iter()
                    .filter(|e| e.kind == 2)
                    .map(|e| e.particle_index)
                    .collect();
                event_indices.sort_unstable();
                assert!(event_indices.windows(2).all(|pair| pair[0] < pair[1]));
                assert!(event_indices
                    .iter()
                    .all(|&index| index < initial.len() as u32));
                for (i, state) in entering_states.iter().enumerate() {
                    let gpu_hit = event_indices.binary_search(&(i as u32)).is_ok();
                    let cpu_hit = hits.binary_search(&(i as u32)).is_ok();
                    assert!(
                        gpu_hit == cpu_hit
                            || (frame > 0 && collision_event_on_roundoff_boundary(*state, &colliders)),
                        "{label} frame {frame} particle {i} collision event GPU={gpu_hit} CPU={cpu_hit}"
                    );
                }
                for (i, (expected, actual)) in reference.iter().zip(&actual.particles).enumerate() {
                    assert_eq!(actual.particle_id, initial[i].particle_id);
                    assert_eq!(actual.emitter_id, initial[i].emitter_id);
                    assert_vec3_close(
                        actual.position,
                        [expected.pos.x, expected.pos.y, expected.pos.z],
                        IntegratorKind::SemiImplicitEuler,
                        i,
                        &format!("{label} frame {frame} position"),
                    );
                    assert_vec3_close(
                        actual.velocity,
                        [expected.vel.x, expected.vel.y, expected.vel.z],
                        IntegratorKind::SemiImplicitEuler,
                        i,
                        &format!("{label} frame {frame} velocity"),
                    );
                }
            }
        }
    }
    stage.destroy(&device);
}

// Repeated push-outs can leave a particle a few ulps either side of a surface.
// This only classifies the event bit, never relaxes position/velocity checks or
// first-frame event checks. A deeper contact anywhere in the batch prevents the
// exception. The budget covers the short f32 distance/normal/push-out chain,
// scaled by both geometry and particle coordinates rather than a scene-unit
// epsilon. Invalid geometry is excluded using the same contract as detection.
fn collision_event_on_roundoff_boundary(state: ParticleState, colliders: &[Collider]) -> bool {
    fn finite(v: Vec3) -> bool {
        v.x.is_finite() && v.y.is_finite() && v.z.is_finite()
    }
    fn magnitude(v: Vec3) -> f32 {
        v.x.abs().max(v.y.abs()).max(v.z.abs())
    }
    if !finite(state.pos) {
        return false;
    }
    let radius = if state.radius.is_finite() {
        state.radius.max(0.0)
    } else {
        0.0
    };
    let mut nearest = f32::INFINITY;
    for &collider in colliders {
        let (distance, geometry_scale) = match collider {
            Collider::HalfSpace(plane) => {
                let length = plane.normal.length();
                if !finite(plane.normal)
                    || !plane.d.is_finite()
                    || !length.is_finite()
                    || length <= 1e-6
                {
                    continue;
                }
                let d = plane.d / length;
                (plane.normal.scale(1.0 / length).dot(state.pos) + d, d.abs())
            }
            Collider::Sphere(sphere) => (
                sd_sphere(sphere, state.pos),
                magnitude(sphere.center).max(sphere.radius.abs()),
            ),
            Collider::Capsule(capsule) => (
                sd_capsule(capsule, state.pos),
                magnitude(capsule.a)
                    .max(magnitude(capsule.b))
                    .max(capsule.radius.abs()),
            ),
            Collider::Box(bounds) => (
                sd_aabb(bounds, state.pos),
                magnitude(bounds.min).max(magnitude(bounds.max)),
            ),
            Collider::Sdf(sample) => {
                if !sample.distance.is_finite() || !finite(sample.gradient) {
                    continue;
                }
                (sample.distance, sample.distance.abs())
            }
        };
        if !distance.is_finite() {
            continue;
        }
        let scale = magnitude(state.pos)
            .max(radius)
            .max(geometry_scale)
            .max(1.0);
        let roundoff = 32.0 * f32::EPSILON * scale;
        nearest = nearest.min((distance - radius) / roundoff);
    }
    nearest.abs() <= 1.0
}

#[test]
fn collision_event_roundoff_band_rejects_deep_and_separated_contacts() {
    let floor = Collider::HalfSpace(Plane::new(Vec3::new(0.0, 1.0, 0.0), 0.0));
    let state = ParticleState::new(Vec3::ZERO, Vec3::ZERO, 0.0);
    assert!(collision_event_on_roundoff_boundary(state, &[floor]));
    for y in [-0.01, 0.01] {
        assert!(!collision_event_on_roundoff_boundary(
            ParticleState {
                pos: Vec3::new(0.0, y, 0.0),
                ..state
            },
            &[floor]
        ));
    }
    let deep = Collider::Sdf(SdfSample {
        distance: -0.25,
        gradient: Vec3::new(0.0, 1.0, 0.0),
    });
    assert!(!collision_event_on_roundoff_boundary(state, &[floor, deep]));
    assert!(!collision_event_on_roundoff_boundary(state, &[]));
    assert!(!collision_event_on_roundoff_boundary(
        state,
        &[Collider::HalfSpace(Plane::new(Vec3::ZERO, 0.0))]
    ));
    assert!(!collision_event_on_roundoff_boundary(
        ParticleState {
            pos: Vec3::splat(f32::NAN),
            ..state
        },
        &[floor]
    ));
}

fn assert_vec3_close(
    actual: [f32; 3],
    expected: [f32; 3],
    integrator: IntegratorKind,
    particle: usize,
    field: &str,
) {
    for axis in 0..3 {
        assert!(actual[axis].is_finite(), "{field}[{axis}] is not finite");
        let scale = expected[axis].abs().max(1.0);
        let tolerance = 2.0e-5 * scale;
        assert!(
            (actual[axis] - expected[axis]).abs() <= tolerance,
            "{integrator:?} particle {particle} {field}[{axis}] mismatch: GPU={} CPU={} tolerance={tolerance}",
            actual[axis],
            expected[axis],
        );
    }
}
