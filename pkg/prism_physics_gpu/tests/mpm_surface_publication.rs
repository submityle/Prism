//! Native ABI and real-device publication tests. Readback is test-only.
use std::collections::BTreeMap;
use std::sync::OnceLock;

use glam::{Mat3, Vec3};
use prism_physics_core::mpm::{Grid, MaterialPoints, MpmConfig, MpmMaterial, MpmSolver};
use prism_physics_core::reconstruct::{triangulate, ScalarField};
use prism_physics_gpu::mpm::gpu::surface::{
    ResidentSurface, ResidentSurfaceConfig, ResidentSurfaceError, SurfaceDrawIndirect,
    SurfaceStatus, SurfaceVertex, RESIDENT_SURFACE_ABI,
};
use prism_physics_gpu::{
    buffer, BoundaryMode, GpuContext, GpuMpmResident, MpmBackendPolicy, MpmRuntime,
    MpmRuntimeConfig, StepConfig, StepInputs,
};

fn context() -> Option<GpuContext> {
    static HOST: OnceLock<Option<GpuContext>> = OnceLock::new();
    let result = HOST.get_or_init(GpuContext::try_headless).clone();
    assert!(
        result.is_some() || std::env::var_os("PRISM_REQUIRE_GPU").is_none(),
        "PRISM_REQUIRE_GPU set but no GPU available"
    );
    result
}

fn config() -> ResidentSurfaceConfig {
    ResidentSurfaceConfig {
        origin: Vec3::splat(0.8),
        nodes: [9; 3],
        dx: 0.05,
        radius: 0.16,
        iso: 0.13,
        max_field_nodes: 729,
        max_cells: 512,
        max_sample_visits: 729,
        max_vertices: 512 * 36,
        max_live_publications: 4,
    }
}

fn solver() -> MpmSolver {
    let mut particles = MaterialPoints::new();
    particles.spawn(Vec3::ONE, Vec3::new(0.1, 0.2, 0.05), 1.0, 0.001);
    MpmSolver::new(
        particles,
        Grid::new(16, 16, 16, 0.1, Vec3::ZERO),
        MpmMaterial::default(),
        MpmConfig::new(0.001, Vec3::new(0.0, -9.81, 0.0)),
    )
}

fn runtime(ctx: &GpuContext) -> MpmRuntime {
    MpmRuntime::with_gpu_context(
        solver(),
        MpmRuntimeConfig {
            backend: MpmBackendPolicy::RequireGpu,
            max_steps_per_advance: 1,
        },
        ctx.clone(),
    )
    .unwrap()
}

fn read<T: bytemuck::Pod>(ctx: &GpuContext, src: &wgpu::Buffer, count: usize) -> Vec<T> {
    if count == 0 {
        return Vec::new();
    }
    let bytes = (count * size_of::<T>()) as u64;
    let stage = buffer::staging(ctx.device(), "surface_test_readback", bytes);
    let mut encoder = ctx.device().create_command_encoder(&Default::default());
    buffer::copy(&mut encoder, src, &stage, bytes);
    ctx.queue().submit([encoder.finish()]);
    buffer::read_back(ctx, &stage)
}

fn geometry(ctx: &GpuContext, surface: &ResidentSurface) -> Vec<SurfaceVertex> {
    let args = read::<SurfaceDrawIndirect>(ctx, surface.indirect(), 1)[0];
    assert_eq!(args.instance_count, 1);
    assert_eq!((args.first_vertex, args.first_instance), (0, 0));
    assert_eq!(args.vertex_count % 3, 0);
    read(ctx, surface.vertices(), args.vertex_count as usize)
}

#[test]
fn surface_abi_and_preflight_are_explicit_and_bounded() {
    assert_eq!(RESIDENT_SURFACE_ABI, 1);
    assert_eq!(size_of::<SurfaceVertex>(), 32);
    assert_eq!(std::mem::offset_of!(SurfaceVertex, position), 0);
    assert_eq!(std::mem::offset_of!(SurfaceVertex, normal), 16);
    assert_eq!(size_of::<SurfaceDrawIndirect>(), 16);
    assert_eq!(size_of::<SurfaceStatus>(), 4);
    assert_eq!(config().validate(1), Ok(()));
    assert_eq!(
        config().validate(2),
        Err(ResidentSurfaceError::BudgetExceeded)
    );
    let mut bad = config();
    bad.nodes = [u32::MAX; 3];
    assert_eq!(bad.validate(1), Err(ResidentSurfaceError::InvalidConfig));
    let mut bad = config();
    bad.nodes = [1 << 20; 3];
    assert_eq!(bad.validate(1), Err(ResidentSurfaceError::BudgetExceeded));
    for scalar in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::MIN_POSITIVE / 8.0] {
        let mut bad = config();
        bad.dx = scalar;
        assert_eq!(bad.validate(1), Err(ResidentSurfaceError::InvalidConfig));
        let mut bad = config();
        bad.radius = scalar;
        assert_eq!(bad.validate(1), Err(ResidentSurfaceError::InvalidConfig));
    }
    for scalar in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        let mut bad = config();
        bad.iso = scalar;
        assert_eq!(bad.validate(1), Err(ResidentSurfaceError::InvalidConfig));
    }
    let mut bad = config();
    bad.origin.x = f32::INFINITY;
    assert_eq!(bad.validate(1), Err(ResidentSurfaceError::InvalidConfig));
    let mut bad = config();
    bad.max_vertices = 4;
    assert_eq!(bad.validate(1), Err(ResidentSurfaceError::InvalidConfig));
    let mut bad = config();
    bad.max_live_publications = 0;
    assert_eq!(bad.validate(1), Err(ResidentSurfaceError::InvalidConfig));
    let mut bad = config();
    bad.max_field_nodes = 728;
    assert_eq!(bad.validate(1), Err(ResidentSurfaceError::BudgetExceeded));
    let mut bad = config();
    bad.max_cells = 511;
    assert_eq!(bad.validate(1), Err(ResidentSurfaceError::BudgetExceeded));
}

#[test]
fn surface_domain_rejects_collapsed_interior_nodes_not_just_endpoints() {
    let mut cfg = config();
    cfg.origin = Vec3::new(16_777_216.0, 0.0, 0.0);
    cfg.dx = 1.5;
    cfg.nodes = [4, 2, 2];
    let upper = cfg.bounds().1;
    // Both previous endpoint tests pass, but two interior world nodes coincide.
    assert_ne!(cfg.origin.x + cfg.dx, cfg.origin.x);
    assert_ne!(upper.x - cfg.dx, upper.x);
    assert_eq!(cfg.origin.x + 2.0 * cfg.dx, cfg.origin.x + 3.0 * cfg.dx);
    assert_eq!(cfg.validate(1), Err(ResidentSurfaceError::InvalidConfig));
    cfg.origin.x = -16_777_216.0 - 4.0;
    assert_eq!(cfg.validate(1), Err(ResidentSurfaceError::InvalidConfig));

    // Translation is supported when spacing resolves nodes and normal probes.
    for origin in [Vec3::splat(1024.0), Vec3::splat(-1024.0)] {
        cfg.origin = origin;
        assert_eq!(cfg.validate(1), Ok(()));
        for axis in 0..3 {
            for i in 0..cfg.nodes[axis] - 1 {
                let a = cfg.origin[axis] + i as f32 * cfg.dx;
                let b = cfg.origin[axis] + (i + 1) as f32 * cfg.dx;
                assert!(a < b);
                assert!(a - cfg.dx * 0.5 < a && a + cfg.dx * 0.5 > a);
            }
        }
    }
}

#[test]
fn gpu_surface_matches_cpu_triangles_and_repeats_without_mutating_old_leases() {
    let Some(ctx) = context() else {
        return;
    };
    let mut runtime = runtime(&ctx);
    let cfg = config();
    let first = runtime.publish_surface(cfg).unwrap();
    assert!(first.is_for_context(&ctx));
    assert!(first
        .vertices()
        .usage()
        .contains(wgpu::BufferUsages::VERTEX));
    assert!(first
        .indirect()
        .usage()
        .contains(wgpu::BufferUsages::INDIRECT));
    assert_eq!(read::<SurfaceStatus>(&ctx, first.status(), 1)[0].flags, 0);
    let vertices = geometry(&ctx, &first);
    assert!(!vertices.is_empty());

    let mut field = ScalarField::zeros(9, 9, 9, cfg.dx, cfg.origin);
    for k in 0..9 {
        for j in 0..9 {
            for i in 0..9 {
                let q = (field.node_position(i, j, k) - Vec3::ONE) / cfg.radius;
                let w = (1.0 - q.length_squared()).max(0.0);
                field.add_value(i, j, k, w * w * w);
            }
        }
    }
    let reference = triangulate(&field, cfg.iso);
    assert_eq!(vertices.len(), reference.indices.len());
    let mut edges = BTreeMap::new();
    for (actual, expected) in vertices
        .chunks_exact(3)
        .zip(reference.indices.chunks_exact(3))
    {
        // GPU fixes outward winding; CPU table orientation is not compared.
        for vertex in actual {
            let p = Vec3::from_array(vertex.position[..3].try_into().unwrap());
            let n = Vec3::from_array(vertex.normal[..3].try_into().unwrap());
            assert!(expected
                .iter()
                .any(|&i| p.distance(reference.positions[i as usize]) < 2.0e-5));
            assert!(n.is_finite() && (n.length() - 1.0).abs() < 2.0e-5);
            let gradient = -field.gradient(p);
            if gradient.length() > 1.0e-4 {
                assert!(n.dot(gradient.normalize()) > 0.999);
            }
            assert_eq!((vertex.position[3], vertex.normal[3]), (0.0, 0.0));
        }
        let positions: [Vec3; 3] =
            std::array::from_fn(|i| Vec3::from_array(actual[i].position[..3].try_into().unwrap()));
        let normals = actual
            .iter()
            .map(|v| Vec3::from_array(v.normal[..3].try_into().unwrap()))
            .fold(Vec3::ZERO, |sum, n| sum + n);
        let cross = (positions[1] - positions[0]).cross(positions[2] - positions[0]);
        assert!(cross.dot(normals) >= -1.0e-8, "inward triangle");
        for (i, j) in [(0, 1), (1, 2), (2, 0)] {
            let a = positions[i].to_array().map(f32::to_bits);
            let b = positions[j].to_array().map(f32::to_bits);
            *edges.entry((a.min(b), a.max(b))).or_insert(0) += 1;
        }
    }
    assert!(
        edges.values().all(|&n| n == 2),
        "closed domain must have no cracks"
    );
    let repeated = runtime.publish_surface(cfg).unwrap();
    assert_eq!(geometry(&ctx, &repeated), vertices);
    assert_eq!(repeated.sequence(), first.sequence() + 1);
    runtime.advance(1);
    let advanced = runtime.publish_surface(cfg).unwrap();
    // Publication did not refresh the CPU checkpoint mirror.
    assert_eq!(runtime.cpu_solver().particles.positions(), &[Vec3::ONE]);
    assert_ne!(geometry(&ctx, &advanced), vertices);
    assert_eq!(geometry(&ctx, &first), vertices);
    assert_ne!(runtime.snapshot().unwrap().positions[0], Vec3::ONE);
    drop(runtime);
    assert_eq!(geometry(&ctx, &first), vertices);
}

#[test]
fn gpu_surface_lease_capacity_retries_and_empty_or_overflow_draws_are_safe() {
    let Some(ctx) = context() else {
        return;
    };
    let mut runtime = runtime(&ctx);
    let mut cfg = config();
    cfg.max_live_publications = 1;
    let first = runtime.publish_surface(cfg).unwrap();
    let pin = first.clone();
    drop(first);
    assert_eq!(
        runtime.publish_surface(cfg).err(),
        Some(ResidentSurfaceError::InFlightLimit)
    );
    drop(pin);
    let next = runtime.publish_surface(cfg).unwrap();
    assert_eq!(next.sequence(), 2);
    drop(next);
    cfg.max_vertices = 3;
    let overflow = runtime.publish_surface(cfg).unwrap();
    assert_eq!(
        read::<SurfaceStatus>(&ctx, overflow.status(), 1)[0].flags,
        2
    );
    assert!(geometry(&ctx, &overflow).is_empty());
    drop(overflow);
    cfg.origin = Vec3::splat(20.0);
    let empty = runtime.publish_surface(cfg).unwrap();
    assert_eq!(read::<SurfaceStatus>(&ctx, empty.status(), 1)[0].flags, 0);
    assert!(geometry(&ctx, &empty).is_empty());
    drop(empty);
    cfg.max_vertices = u32::MAX;
    assert_eq!(
        runtime.publish_surface(cfg).err(),
        Some(ResidentSurfaceError::DeviceLimit)
    );
}

fn upload(resident: &mut GpuMpmResident, ctx: &GpuContext, position: Vec3) {
    upload_points(resident, ctx, &[position]);
}

fn upload_points(resident: &mut GpuMpmResident, ctx: &GpuContext, positions: &[Vec3]) {
    let (lambda0, mu0) = MpmMaterial::default().lame();
    resident.upload(
        ctx,
        &StepInputs {
            positions,
            velocities: &vec![Vec3::ZERO; positions.len()],
            affine: &vec![Mat3::ZERO; positions.len()],
            deformation: &vec![Mat3::IDENTITY; positions.len()],
            masses: &vec![1.0; positions.len()],
            volumes: &vec![0.001; positions.len()],
            plastic_det: &vec![1.0; positions.len()],
        },
        &StepConfig {
            dims: (16, 16, 16),
            origin: Vec3::ZERO,
            dx: 0.1,
            dt: 0.001,
            gravity: Vec3::ZERO,
            mu0,
            lambda0,
            hardening: 0.0,
            theta_c: 0.025,
            theta_s: 0.0075,
            boundary: BoundaryMode::Slip,
            boundary_thickness: 2,
            plastic: false,
            steps: 1,
        },
    );
}

#[test]
fn gpu_surface_winding_follows_tetrahedral_field_not_smooth_shading_normals() {
    let Some(ctx) = context() else {
        return;
    };
    let points = [
        Vec3::new(2.217, 2.046, 2.059),
        Vec3::new(2.719, 0.900, 2.568),
    ];
    let cfg = ResidentSurfaceConfig {
        origin: Vec3::ZERO,
        nodes: [4; 3],
        dx: 1.0,
        radius: 1.471,
        iso: 0.302,
        max_field_nodes: 64,
        max_cells: 27,
        max_sample_visits: 128,
        max_vertices: 27 * 36,
        max_live_publications: 1,
    };
    let mut resident = GpuMpmResident::new(&ctx);
    upload_points(&mut resident, &ctx, &points);
    let surface = resident.publish_surface(&ctx, cfg).unwrap();
    assert_eq!(read::<SurfaceStatus>(&ctx, surface.status(), 1)[0].flags, 0);
    let vertices = geometry(&ctx, &surface);
    let density = |p: Vec3| {
        points.iter().fold(0.0, |sum, &particle| {
            let q = (p - particle) / cfg.radius;
            let weight = (1.0 - q.length_squared()).max(0.0);
            sum + weight * weight * weight
        })
    };
    let tetrahedra = [
        [0, 1, 3, 7],
        [0, 1, 5, 7],
        [0, 2, 3, 7],
        [0, 2, 6, 7],
        [0, 4, 5, 7],
        [0, 4, 6, 7],
    ];
    let case_counts = [0, 3, 3, 6, 3, 6, 6, 3, 3, 6, 6, 3, 6, 3, 3, 0];
    let mut cursor = 0;
    let mut regression_seen = false;
    for z in 0..3 {
        for y in 0..3 {
            for x in 0..3 {
                let base = Vec3::new(x as f32, y as f32, z as f32);
                for (tet, corners) in tetrahedra.iter().enumerate() {
                    let positions = corners.map(|c| {
                        base + Vec3::new(
                            (c & 1) as f32,
                            ((c >> 1) & 1) as f32,
                            ((c >> 2) & 1) as f32,
                        )
                    });
                    let values = positions.map(density);
                    let mask = values
                        .iter()
                        .enumerate()
                        .fold(0, |mask, (i, &v)| mask | (usize::from(v > cfg.iso) << i));
                    let count = case_counts[mask];
                    let [e1, e2, e3] = [
                        positions[1] - positions[0],
                        positions[2] - positions[0],
                        positions[3] - positions[0],
                    ];
                    // Solve the affine tetrahedral density gradient independently
                    // of the shader's trilinear finite-difference normal routine.
                    let gradient = ((values[1] - values[0]) * e2.cross(e3)
                        + (values[2] - values[0]) * e3.cross(e1)
                        + (values[3] - values[0]) * e1.cross(e2))
                        / e1.dot(e2.cross(e3));
                    for triangle in vertices[cursor..cursor + count].chunks_exact(3) {
                        let p: [Vec3; 3] = std::array::from_fn(|i| {
                            Vec3::from_array(triangle[i].position[..3].try_into().unwrap())
                        });
                        let mut smooth_sum = Vec3::ZERO;
                        for v in triangle {
                            let n = Vec3::from_array(v.normal[..3].try_into().unwrap());
                            assert!(n.is_finite() && (n.length() - 1.0).abs() < 2.0e-5);
                            smooth_sum += n;
                        }
                        let face = (p[1] - p[0]).cross(p[2] - p[0]);
                        assert!(
                            face.dot(-gradient) >= -1.0e-5 * face.length() * gradient.length(),
                            "inward triangle in cell {x},{y},{z}, tet {tet}, mask {mask}"
                        );
                        if (x, y, z, tet) == (2, 1, 2, 5) {
                            assert_eq!(mask, 1);
                            assert!(face.dot(-gradient) > 0.01);
                            // The old smooth-normal-based correction would
                            // reverse this correctly outward-facing triangle.
                            assert!(face.dot(smooth_sum) < -0.1);
                            regression_seen = true;
                        }
                    }
                    cursor += count;
                }
            }
        }
    }
    assert_eq!(cursor, vertices.len());
    assert!(regression_seen);
}

#[test]
fn gpu_surface_rejects_wrong_host_and_suppresses_invalid_particles_after_upload_replacement() {
    let Some(ctx) = context() else {
        return;
    };
    let foreign = GpuContext::from_wgpu(
        ctx.instance().clone(),
        ctx.adapter().clone(),
        ctx.device().clone(),
        ctx.queue().clone(),
    );
    assert!(!foreign.shares_device(&ctx));
    assert!(ctx.clone().shares_device(&ctx));
    let mut resident = GpuMpmResident::new(&ctx);
    upload(&mut resident, &ctx, Vec3::ONE);
    assert_eq!(
        resident.publish_surface(&foreign, config()).err(),
        Some(ResidentSurfaceError::WrongDevice)
    );
    let old = resident.publish_surface(&ctx, config()).unwrap();
    assert!(!old.is_for_context(&foreign));
    let old_vertices = geometry(&ctx, &old);
    for (i, value) in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY]
        .into_iter()
        .enumerate()
    {
        upload(&mut resident, &ctx, Vec3::splat(value));
        let invalid = resident.publish_surface(&ctx, config()).unwrap();
        assert_eq!(invalid.sequence(), old.sequence() + i as u64 + 1);
        assert_eq!(read::<SurfaceStatus>(&ctx, invalid.status(), 1)[0].flags, 1);
        assert!(geometry(&ctx, &invalid).is_empty());
    }
    // Invalid GPU data must not poison subsequent publications or reset the
    // sequence when the uploaded batch is replaced again.
    upload(&mut resident, &ctx, Vec3::ONE);
    let recovered = resident.publish_surface(&ctx, config()).unwrap();
    assert_eq!(recovered.sequence(), old.sequence() + 4);
    assert_eq!(
        read::<SurfaceStatus>(&ctx, recovered.status(), 1)[0].flags,
        0
    );
    assert_eq!(geometry(&ctx, &recovered), old_vertices);
    assert_eq!(geometry(&ctx, &old), old_vertices);
}

#[test]
fn cpu_runtime_does_not_fake_native_surface_publication() {
    let mut runtime = MpmRuntime::new(
        solver(),
        MpmRuntimeConfig {
            backend: MpmBackendPolicy::Cpu,
            max_steps_per_advance: 1,
        },
    )
    .unwrap();
    assert_eq!(
        runtime.publish_surface(config()).err(),
        Some(ResidentSurfaceError::BackendUnavailable)
    );
}
