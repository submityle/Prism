use super::*;
use prism_render_architecture::particle::{
    collision::Collider,
    forces::{Force, ForceEntry, ForceField},
    sort_cull::Plane,
    IterationDomain, Vec3,
};
use prism_render_driver_wgpu::{require_hardware_gpu, HardwareDeviceError, WgpuInstance};

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
        GpuParticle::live([0.0, 0.05, 0.0], [0.0, -1.0, 0.0], 2.0, 0.1, 10),
        GpuParticle::live([0.0, 2.0, 0.0], [0.0; 3], 0.05, 0.0, 11),
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
    stage.upload(
        &queue,
        &pool,
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
