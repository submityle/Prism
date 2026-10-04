//! Real-device parity coverage for the production [`GpuRayTraversal`] service.
//!
//! The sibling kernel-parity suites ([`gpu_tests`](super::gpu_tests),
//! [`tlas_gpu_tests`](super::tlas_gpu_tests),
//! [`footprint_gpu_tests`](super::footprint_gpu_tests)) each stand up a raw
//! `wgpu` device *by hand* — compiling the `WESL`, building the bind group and
//! recording the pass inline — to prove the individual kernels match their
//! `CPU` golden walks. Those tests guard the shaders, but they never touch the
//! reusable [`GpuRayTraversal`] facade that production consumers actually call.
//!
//! The tests here close that gap: they build the service through its real
//! [`RenderDevice`] / [`RenderQueue`] constructor on a live device and exercise
//! every one of its five public walks — [`GpuRayTraversal::closest_hits`],
//! [`GpuRayTraversal::any_hits`], [`GpuRayTraversal::tlas_closest_hits`],
//! [`GpuRayTraversal::tlas_any_hits`] and [`GpuRayTraversal::footprint_mips`] —
//! asserting the service's decoded output element-for-element against the
//! authoritative `CPU` golden (`GpuBvhBuffers::{closest_hit, any_hit}`,
//! `GpuTlasBuffers::{closest_hit, any_hit}` and [`RayFootprint`]). A green run
//! is direct on-device evidence that the shared upload → dispatch → read-back
//! plumbing (pipeline compile, bind-group build, staging copy, poll and decode)
//! is wired correctly for all three kernels, not merely that the kernels work
//! in isolation.
//!
//! Adapter acquisition is best-effort: on a headless host with no `wgpu`
//! adapter [`try_solver_device`] returns `None` and each test skips with a
//! printed notice instead of failing, matching the kernel suites so the whole
//! crate stays green everywhere while still exercising the real service on any
//! machine with a device.

use bevy_platform::future::block_on;
use bevy_render::renderer::{RenderDevice, RenderQueue};
use wgpu::{
    BackendOptions, Backends, DeviceDescriptor, Instance as WgpuInstance, InstanceDescriptor,
    InstanceFlags, RequestAdapterOptions,
};

use prism_render_architecture::ray_scene::{
    Affine3, Bvh, BvhBuildConfig, GpuBlasPool, GpuBvhBuffers, GpuTlasBuffers,
    Instance as SceneInstance, Ray, RayFootprint, Tlas, Triangle,
};

use super::abi::{MISS_PRIMITIVE, POSITIVE_INF_BITS};
use super::dispatch::GpuRayTraversal;
use super::resources::FootprintRequest;

/// Absolute per-scalar tolerance for the `GPU`-versus-`CPU` comparison.
///
/// Both paths run the identical `float32` arithmetic, so agreement is far
/// tighter than this in practice; the margin only absorbs a driver's
/// fused-multiply-add contraction and reordering freedom.
const PARITY_EPS: f32 = 1.0e-3;

/// Inclusive mip ceiling the continuous footprint level is clamped to.
const MAX_MIP: u32 = 8;

/// Minimum distance the golden mip level must sit from an integer boundary for
/// the discrete floor bucket to be compared for exact equality; nearer a
/// boundary a legitimate sub-`ULP` difference can flip the floor.
const MIP_FLOOR_GUARD: f32 = 1.0e-3;

/// Best-effort acquisition of a native compute device and queue, wrapped into
/// the render world's [`RenderDevice`] / [`RenderQueue`] handles.
///
/// Returns `None` (rather than panicking) when no adapter is available so the
/// suite stays green on headless hosts; on a machine with a real `GPU` this
/// yields the live render handles the service is built against.
fn try_solver_device() -> Option<(RenderDevice, RenderQueue)> {
    let instance = WgpuInstance::new(InstanceDescriptor {
        backends: Backends::METAL | Backends::VULKAN | Backends::DX12,
        flags: InstanceFlags::default(),
        memory_budget_thresholds: Default::default(),
        display: None,
        backend_options: BackendOptions::default(),
    });
    let adapter = block_on(instance.request_adapter(&RequestAdapterOptions::default())).ok()?;
    let (device, queue) = block_on(adapter.request_device(&DeviceDescriptor {
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .ok()?;
    Some((RenderDevice::from(device), RenderQueue::new(queue)))
}

/// Tiny deterministic `xorshift` `RNG` so the scenes are byte-reproducible
/// across runs and hosts without pulling in a dependency.
struct Rng {
    /// The running 64-bit state; never zero.
    state: u64,
}

impl Rng {
    /// Seeds the generator, forcing a non-zero state.
    fn new(seed: u64) -> Self {
        Self { state: seed | 1 }
    }

    /// Advances the state and returns the high 32 bits.
    fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        (x >> 32) as u32
    }

    /// Uniform `f32` in `[lo, hi)` using only the four basic operators.
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        let unit = (self.next_u32() as f32) / (u32::MAX as f32);
        lo + (hi - lo) * unit
    }

    /// A random point in the cube `[lo, hi)^3`.
    fn point(&mut self, lo: f32, hi: f32) -> [f32; 3] {
        [self.range(lo, hi), self.range(lo, hi), self.range(lo, hi)]
    }
}

/// Builds a deterministic, non-trivial triangle soup that forces a genuine
/// multi-level `BVH` (interior nodes plus several leaves).
fn build_soup(seed: u64, count: u32, lo: f32, hi: f32) -> Vec<Triangle> {
    let mut rng = Rng::new(seed);
    let mut triangles = Vec::new();
    let mut primitive = 0u32;
    while primitive < count {
        let center = rng.point(lo, hi);
        let e0 = rng.point(-0.7, 0.7);
        let e1 = rng.point(-0.7, 0.7);
        let v0 = center;
        let v1 = [center[0] + e0[0], center[1] + e0[1], center[2] + e0[2]];
        let v2 = [center[0] + e1[0], center[1] + e1[1], center[2] + e1[2]];
        triangles.push(Triangle::new(v0, v1, v2, primitive));
        primitive += 1;
    }
    triangles
}

/// Builds a batch of rays that provably exercises both the hit and the miss
/// paths against a single-`BLAS` triangle soup.
fn build_bvh_rays(triangles: &[Triangle]) -> Vec<Ray> {
    let mut rng = Rng::new(0x0BAD_F00D_DEAD_BEEF);
    let mut rays = Vec::new();
    for tri in triangles {
        let c = [
            (tri.v0[0] + tri.v1[0] + tri.v2[0]) / 3.0,
            (tri.v0[1] + tri.v1[1] + tri.v2[1]) / 3.0,
            (tri.v0[2] + tri.v1[2] + tri.v2[2]) / 3.0,
        ];
        let origin = [
            rng.range(-14.0, 14.0),
            rng.range(-14.0, 14.0),
            rng.range(-14.0, 14.0),
        ];
        let dir = [c[0] - origin[0], c[1] - origin[1], c[2] - origin[2]];
        rays.push(Ray::new(origin, dir, 0.0, f32::INFINITY));
    }
    // Guaranteed misses: a finite interval that stops short, and a ray aimed
    // away from the soup.
    rays.push(Ray::new([20.0, 20.0, 20.0], [1.0, 1.0, 1.0], 0.0, 0.5));
    rays.push(Ray::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 100.0, 200.0));
    rays
}

/// A built two-level scene: the pooled `BLAS` buffers, the packed `TLAS`
/// buffers, and the packed-order object→world transforms plus per-instance
/// soups kept to aim rays and recover object-space centroids.
struct TlasScene {
    /// The pooled bottom-level acceleration structures.
    pool: GpuBlasPool,
    /// The packed top-level acceleration structure.
    tlas_buffers: GpuTlasBuffers,
    /// Object→world transform per packed instance.
    object_to_world: Vec<Affine3>,
    /// Triangle soup per packed instance (object space).
    soups: Vec<Vec<Triangle>>,
}

/// Assembles two distinct `BLAS` soups and four affine-transformed instances so
/// the dispatch exercises the object-space transform path and cross-instance
/// pruning, recovering the packed instance order the kernel reads.
fn build_tlas_scene() -> TlasScene {
    let soup0 = build_soup(0x1111_2222_3333_4444, 18, -1.5, 1.5);
    let soup1 = build_soup(0x5555_6666_7777_8888, 22, -1.5, 1.5);
    let blases = vec![Bvh::build(&soup0), Bvh::build(&soup1)];

    let placements: Vec<(Affine3, usize, u32)> = vec![
        (Affine3::from_translation([6.0, 0.0, 0.0]), 0, 10),
        (
            Affine3::from_translation([-6.0, 0.0, 0.0])
                .compose(&Affine3::from_scale([1.5, 0.75, 1.25])),
            0,
            20,
        ),
        (
            Affine3::from_translation([0.0, 6.0, 0.0])
                .compose(&Affine3::from_quaternion([0.9239, 0.0, 0.3827, 0.0])),
            1,
            30,
        ),
        (
            Affine3::from_translation([0.0, -6.0, 3.0])
                .compose(&Affine3::from_scale([2.0, 2.0, 2.0])),
            1,
            40,
        ),
    ];

    let mut instances = Vec::new();
    let mut object_to_world = Vec::new();
    let mut instance_blas = Vec::new();
    for (m, blas, id) in placements {
        let inst = SceneInstance::new(m, blas, id).expect("instance transform must be invertible");
        instances.push(inst);
        object_to_world.push(m);
        instance_blas.push(blas);
    }

    let tlas = Tlas::build_with(
        &instances,
        &blases,
        BvhBuildConfig {
            max_leaf_primitives: 1,
            ..BvhBuildConfig::default()
        },
    );
    let tlas_buffers = GpuTlasBuffers::from_tlas(&tlas);
    let pool = GpuBlasPool::from_blases(&blases);

    // The build reorders instances; recover the packed order so the ray-aim
    // book-keeping tracks the buffer the kernel actually reads.
    let mut reordered_o2w = Vec::new();
    let mut reordered_soups = Vec::new();
    for inst in tlas.instances() {
        let id = inst.instance_id();
        let orig = instances
            .iter()
            .position(|i| i.instance_id() == id)
            .expect("instance id must survive the build");
        reordered_o2w.push(object_to_world[orig]);
        reordered_soups.push(if instance_blas[orig] == 0 {
            soup0.clone()
        } else {
            soup1.clone()
        });
    }

    TlasScene {
        pool,
        tlas_buffers,
        object_to_world: reordered_o2w,
        soups: reordered_soups,
    }
}

/// Builds world-space rays aimed at each instance's transformed centroids plus
/// guaranteed misses.
fn build_tlas_rays(scene: &TlasScene) -> Vec<Ray> {
    let mut rng = Rng::new(0x0BAD_F00D_C0FF_EE00);
    let mut rays = Vec::new();
    for (idx, o2w) in scene.object_to_world.iter().enumerate() {
        for tri in scene.soups[idx].iter().take(6) {
            let c_obj = [
                (tri.v0[0] + tri.v1[0] + tri.v2[0]) / 3.0,
                (tri.v0[1] + tri.v1[1] + tri.v2[1]) / 3.0,
                (tri.v0[2] + tri.v1[2] + tri.v2[2]) / 3.0,
            ];
            let c = o2w.transform_point(c_obj);
            let origin = [
                c[0] + rng.range(-10.0, 10.0),
                c[1] + rng.range(-10.0, 10.0),
                c[2] + rng.range(-10.0, 10.0),
            ];
            let dir = [c[0] - origin[0], c[1] - origin[1], c[2] - origin[2]];
            rays.push(Ray::new(origin, dir, 0.0, f32::INFINITY));
        }
    }
    rays.push(Ray::new([40.0, 40.0, 40.0], [1.0, 1.0, 1.0], 0.0, 0.5));
    rays.push(Ray::new(
        [0.0, 0.0, 40.0],
        [0.0, 0.0, 1.0],
        0.0,
        f32::INFINITY,
    ));
    rays
}

/// Builds a batch of footprint requests spanning mip 0, intermediate mips, the
/// clamp ceiling and the guarded degenerate (zero / non-finite) inputs.
fn build_footprints() -> Vec<FootprintRequest> {
    let mut rng = Rng::new(0x00F0_07FA_1357_9BDF);
    let mut out = Vec::new();
    for _ in 0..48 {
        out.push(FootprintRequest {
            cone_width: rng.range(0.001, 2.0),
            cone_spread_angle: rng.range(0.0, 0.4),
            hit_distance: rng.range(0.0, 40.0),
            texel_world_size: rng.range(0.01, 1.5),
        });
    }
    let edges = [
        // Sub-texel footprint clamps to mip 0.
        FootprintRequest {
            cone_width: 0.25,
            cone_spread_angle: 0.0,
            hit_distance: 0.0,
            texel_world_size: 1.0,
        },
        // 16 texels -> mip 4 exactly.
        FootprintRequest {
            cone_width: 16.0,
            cone_spread_angle: 0.0,
            hit_distance: 0.0,
            texel_world_size: 1.0,
        },
        // Far beyond the ceiling -> clamp to max mip.
        FootprintRequest {
            cone_width: 4096.0,
            cone_spread_angle: 0.0,
            hit_distance: 0.0,
            texel_world_size: 1.0,
        },
        // Guarded zero texel -> span 0 -> mip 0.
        FootprintRequest {
            cone_width: 2.0,
            cone_spread_angle: 0.0,
            hit_distance: 0.0,
            texel_world_size: 0.0,
        },
        // Non-finite / negative slopes sanitize to zero.
        FootprintRequest {
            cone_width: -1.0,
            cone_spread_angle: f32::NAN,
            hit_distance: f32::INFINITY,
            texel_world_size: 1.0,
        },
        // Growing distance drives coarser mips.
        FootprintRequest {
            cone_width: 0.1,
            cone_spread_angle: 0.05,
            hit_distance: 30.0,
            texel_world_size: 0.1,
        },
    ];
    out.extend_from_slice(&edges);
    out
}

/// Service parity: [`GpuRayTraversal::closest_hits`] must reproduce
/// [`GpuBvhBuffers::closest_hit`] for every ray in the batch.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn service_closest_hits_match_cpu_golden() {
    let Some((device, queue)) = try_solver_device() else {
        eprintln!("skipping service closest-hit parity: no wgpu adapter");
        return;
    };

    let triangles = build_soup(0x00C0_FFEE_1234_5678, 96, -6.0, 6.0);
    let bvh = Bvh::build(&triangles);
    assert!(bvh.node_count() > 1, "scene must build a multi-node BVH");
    let buffers = GpuBvhBuffers::from_bvh(&bvh);
    let rays = build_bvh_rays(&triangles);

    let service = GpuRayTraversal::new(&device);
    let gpu = service.closest_hits(&device, &queue, &buffers, &rays);
    assert_eq!(gpu.len(), rays.len(), "one hit per ray");

    let mut hit_count = 0usize;
    let mut miss_count = 0usize;
    for (i, ray) in rays.iter().enumerate() {
        let g = gpu[i];
        match buffers.closest_hit(ray) {
            Some(hit) => {
                hit_count += 1;
                assert_eq!(g.primitive, hit.primitive, "ray {i}: primitive mismatch");
                assert!((g.t - hit.t).abs() <= PARITY_EPS, "ray {i}: t mismatch");
                assert!(
                    (g.u - hit.u).abs() <= PARITY_EPS && (g.v - hit.v).abs() <= PARITY_EPS,
                    "ray {i}: uv mismatch"
                );
            }
            None => {
                miss_count += 1;
                assert_eq!(g.primitive, MISS_PRIMITIVE, "ray {i}: spurious hit");
                assert_eq!(
                    g.t.to_bits(),
                    POSITIVE_INF_BITS,
                    "ray {i}: miss must be +inf"
                );
            }
        }
    }
    assert!(hit_count > 0, "batch must exercise the hit path");
    assert!(miss_count > 0, "batch must exercise the miss path");
}

/// Service parity: [`GpuRayTraversal::any_hits`] must agree with
/// [`GpuBvhBuffers::any_hit`] on whether each ray is occluded.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn service_any_hits_match_cpu_golden() {
    let Some((device, queue)) = try_solver_device() else {
        eprintln!("skipping service any-hit parity: no wgpu adapter");
        return;
    };

    let triangles = build_soup(0x00C0_FFEE_1234_5678, 96, -6.0, 6.0);
    let bvh = Bvh::build(&triangles);
    let buffers = GpuBvhBuffers::from_bvh(&bvh);
    let rays = build_bvh_rays(&triangles);

    let service = GpuRayTraversal::new(&device);
    let gpu = service.any_hits(&device, &queue, &buffers, &rays);
    assert_eq!(gpu.len(), rays.len(), "one occlusion flag per ray");

    let mut blocked_count = 0usize;
    let mut clear_count = 0usize;
    for (i, ray) in rays.iter().enumerate() {
        let cpu = buffers.any_hit(ray);
        assert_eq!(gpu[i], cpu, "ray {i}: occlusion mismatch");
        if cpu {
            blocked_count += 1;
        } else {
            clear_count += 1;
        }
    }
    assert!(blocked_count > 0, "batch must exercise the blocked path");
    assert!(clear_count > 0, "batch must exercise the clear path");
}

/// Service parity: [`GpuRayTraversal::tlas_closest_hits`] must reproduce
/// [`GpuTlasBuffers::closest_hit`] for every ray in the batch.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn service_tlas_closest_hits_match_cpu_golden() {
    let Some((device, queue)) = try_solver_device() else {
        eprintln!("skipping service tlas closest-hit parity: no wgpu adapter");
        return;
    };

    let scene = build_tlas_scene();
    let rays = build_tlas_rays(&scene);

    let service = GpuRayTraversal::new(&device);
    let gpu = service.tlas_closest_hits(&device, &queue, &scene.tlas_buffers, &scene.pool, &rays);
    assert_eq!(gpu.len(), rays.len(), "one hit per ray");

    let mut hit_count = 0usize;
    let mut miss_count = 0usize;
    for (i, ray) in rays.iter().enumerate() {
        let g = gpu[i];
        match scene.tlas_buffers.closest_hit(ray, &scene.pool) {
            Some(hit) => {
                hit_count += 1;
                assert_eq!(g.primitive, hit.primitive, "ray {i}: primitive mismatch");
                assert_eq!(
                    g.instance_id, hit.instance_id,
                    "ray {i}: instance_id mismatch"
                );
                assert_eq!(
                    g.instance_index, hit.instance_index,
                    "ray {i}: instance_index mismatch"
                );
                assert!((g.t - hit.t).abs() <= PARITY_EPS, "ray {i}: t mismatch");
                assert!(
                    (g.u - hit.u).abs() <= PARITY_EPS && (g.v - hit.v).abs() <= PARITY_EPS,
                    "ray {i}: uv mismatch"
                );
            }
            None => {
                miss_count += 1;
                assert_eq!(g.primitive, MISS_PRIMITIVE, "ray {i}: spurious hit");
                assert_eq!(
                    g.t.to_bits(),
                    POSITIVE_INF_BITS,
                    "ray {i}: miss must be +inf"
                );
            }
        }
    }
    assert!(hit_count > 0, "batch must exercise the hit path");
    assert!(miss_count > 0, "batch must exercise the miss path");
}

/// Service parity: [`GpuRayTraversal::tlas_any_hits`] must agree with
/// [`GpuTlasBuffers::any_hit`] on whether each ray is occluded.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn service_tlas_any_hits_match_cpu_golden() {
    let Some((device, queue)) = try_solver_device() else {
        eprintln!("skipping service tlas any-hit parity: no wgpu adapter");
        return;
    };

    let scene = build_tlas_scene();
    let rays = build_tlas_rays(&scene);

    let service = GpuRayTraversal::new(&device);
    let gpu = service.tlas_any_hits(&device, &queue, &scene.tlas_buffers, &scene.pool, &rays);
    assert_eq!(gpu.len(), rays.len(), "one occlusion flag per ray");

    let mut blocked_count = 0usize;
    let mut clear_count = 0usize;
    for (i, ray) in rays.iter().enumerate() {
        let cpu = scene.tlas_buffers.any_hit(ray, &scene.pool);
        assert_eq!(gpu[i], cpu, "ray {i}: occlusion mismatch");
        if cpu {
            blocked_count += 1;
        } else {
            clear_count += 1;
        }
    }
    assert!(blocked_count > 0, "batch must exercise the blocked path");
    assert!(clear_count > 0, "batch must exercise the clear path");
}

/// Service parity: [`GpuRayTraversal::footprint_mips`] must reproduce the
/// [`RayFootprint`] mip math for every request.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn service_footprint_mips_match_cpu_golden() {
    let Some((device, queue)) = try_solver_device() else {
        eprintln!("skipping service footprint parity: no wgpu adapter");
        return;
    };

    let requests = build_footprints();
    let service = GpuRayTraversal::new(&device);
    let gpu = service.footprint_mips(&device, &queue, &requests, MAX_MIP);
    assert_eq!(gpu.len(), requests.len(), "one result per request");

    let mut mip0_count = 0usize;
    let mut clamped_count = 0usize;
    let mut exact_floor_count = 0usize;
    for (i, req) in requests.iter().enumerate() {
        let fp = RayFootprint::new(req.cone_width, req.cone_spread_angle, req.hit_distance);
        let cpu_width = fp.projected_width();
        let cpu_span = fp.texel_span(req.texel_world_size);
        let cpu_level = fp.mip_level(req.texel_world_size, MAX_MIP);
        let cpu_floor = fp.mip_floor(req.texel_world_size, MAX_MIP);
        let g = gpu[i];

        assert!(
            (g.projected_width - cpu_width).abs() <= PARITY_EPS,
            "record {i}: projected_width {} != CPU {}",
            g.projected_width,
            cpu_width
        );
        assert!(
            (g.texel_span - cpu_span).abs() <= PARITY_EPS,
            "record {i}: texel_span {} != CPU {}",
            g.texel_span,
            cpu_span
        );
        assert!(
            (g.mip_level - cpu_level).abs() <= PARITY_EPS,
            "record {i}: mip_level {} != CPU {}",
            g.mip_level,
            cpu_level
        );

        if cpu_level <= 0.0 {
            mip0_count += 1;
        }
        if cpu_level >= MAX_MIP as f32 {
            clamped_count += 1;
        }
        if (cpu_level - cpu_level.round()).abs() > MIP_FLOOR_GUARD {
            exact_floor_count += 1;
            assert_eq!(g.mip_floor, cpu_floor, "record {i}: mip_floor mismatch");
        }
    }
    assert!(mip0_count > 0, "batch must exercise the mip-0 path");
    assert!(
        clamped_count > 0,
        "batch must exercise the max-mip clamp path"
    );
    assert!(
        exact_floor_count > 0,
        "batch must exercise the discrete floor bucket"
    );
}
