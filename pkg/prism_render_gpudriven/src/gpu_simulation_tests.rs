use super::*;
use prism_render_architecture::particle::{
    collision::Collider,
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

    let stage = GpuSimulationStage::new(&device, 4, 16);
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
    let output = stage.run(&queue);
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
        let stage = GpuSimulationStage::new(&device, initial.len() as u32, 8);
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
        let mut output = stage.run(&queue);
        for _ in 1..8 {
            for state in &mut reference {
                *state = integrate(integrator, *state, dt, |p, v| forces.accumulate(p, v));
            }
            output = stage.run(&queue);
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
    let stage = GpuSimulationStage::new(&device, initial.len() as u32, 16);
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
                let actual = stage.run(&queue);
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
